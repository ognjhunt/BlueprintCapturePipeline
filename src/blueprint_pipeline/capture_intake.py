"""Immutable, authority-aware admission for one logical customer capture.

This module is the boundary between an upload action and the existing raw
capture materialization path.  It verifies the submitted bytes, places them in
a content-addressed store, and emits a deterministic admission or targeted
recapture report.  It does not reconstruct a scene or upgrade derived assets to
raw, metric, physics, or physical authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .core.security_controls import strict_identifier


CAPTURE_INTAKE_SCHEMA_VERSION = "capture_intake_envelope.v1"
CAPTURE_ADMISSION_SCHEMA_VERSION = "capture_intake_admission.v1"
CAPTURE_AUTHORITY_PROFILES = {
    "iphone_arkit_lidar",
    "iphone_arkit_non_lidar",
    "camera_360_equirectangular",
    "camera_360_native",
    "monocular_video",
    "precomputed_external_reconstruction",
    "public_processed_rgbd_pose_sequence",
}

_SHA256 = "sha256:"
_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024 * 1024
_VIDEO_SUFFIXES = {".mp4", ".mov", ".insv"}
_ALLOWED_SUFFIXES = _VIDEO_SUFFIXES | {
    ".json",
    ".jsonl",
    ".bin",
    ".ply",
    ".obj",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
    ".glb",
    ".gltf",
}
_REQUIRED_STREAMS = {
    "iphone_arkit_lidar": {
        "retained_video",
        "decoded_video_pts",
        "frame_retention_mapping",
        "camera_poses",
        "camera_intrinsics",
        "depth",
        "depth_confidence",
        "tracking_state",
        "coordinate_frame_semantics",
    },
    "iphone_arkit_non_lidar": {
        "retained_video",
        "decoded_video_pts",
        "frame_retention_mapping",
        "camera_poses",
        "camera_intrinsics",
        "tracking_state",
        "coordinate_frame_semantics",
    },
    "camera_360_equirectangular": {"retained_video", "camera_metadata"},
    "camera_360_native": {"retained_original", "camera_metadata"},
    "monocular_video": {"retained_video"},
    "precomputed_external_reconstruction": {"external_reconstruction"},
    "public_processed_rgbd_pose_sequence": {
        "processed_rgb_observations",
        "camera_poses",
        "camera_intrinsics",
        "depth",
    },
}


class CaptureIntakeError(ValueError):
    """Fail-closed intake error with stable identifiers."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest_mapping(value: Mapping[str, Any], *, omit: str | None = None) -> str:
    normalized = json.loads(json.dumps(value))
    if omit:
        normalized.pop(omit, None)
    return _SHA256 + hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return _SHA256 + digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith(_SHA256) and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _safe_relative_path(value: Any) -> str | None:
    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return str(path)


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _validate_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        envelope = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise CaptureIntakeError(["envelope:not_json_serializable"]) from exc
    errors: list[str] = []
    if envelope.get("schema_version") != CAPTURE_INTAKE_SCHEMA_VERSION:
        errors.append(f"schema_version:must_be:{CAPTURE_INTAKE_SCHEMA_VERSION}")
    for key in (
        "intake_id",
        "idempotency_key",
        "source_type",
        "scene_id",
        "customer_id",
        "organization_id",
        "requested_task_evaluation_run_audience",
    ):
        if not str(envelope.get(key) or "").strip():
            errors.append(f"{key}:missing")
    try:
        envelope["intake_id"] = strict_identifier(
            envelope.get("intake_id"), field="intake_id", max_length=128
        )
    except ValueError:
        errors.append("intake_id:invalid_path_identifier")
    profile = str(envelope.get("capture_authority_profile") or "")
    if profile not in CAPTURE_AUTHORITY_PROFILES:
        errors.append("capture_authority_profile:unsupported")
    if str(envelope.get("source_type") or "") != profile:
        errors.append("source_type:must_match_capture_authority_profile")
    files = _rows(envelope.get("original_files"))
    if not files:
        errors.append("original_files:missing_or_invalid")
    seen_paths: set[str] = set()
    for index, row in enumerate(files):
        relative = _safe_relative_path(row.get("relative_path"))
        if relative is None:
            errors.append(f"original_files[{index}].relative_path:unsafe")
        elif relative in seen_paths:
            errors.append(f"original_files[{index}].relative_path:duplicate")
        else:
            seen_paths.add(relative)
        filename = str(row.get("original_filename") or "").strip()
        if not filename or Path(filename).name != filename:
            errors.append(f"original_files[{index}].original_filename:invalid")
        if not _is_sha256(row.get("sha256")):
            errors.append(f"original_files[{index}].sha256:invalid")
        size = row.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or not (0 < size <= _MAX_FILE_SIZE_BYTES):
            errors.append(f"original_files[{index}].size_bytes:invalid")
        suffix = Path(relative or filename).suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            errors.append(f"original_files[{index}].file_type:unsupported")
    for key in ("capture_device", "timing_declaration", "coordinate_frame_declaration"):
        if not isinstance(envelope.get(key), Mapping) or not envelope.get(key):
            errors.append(f"{key}:missing_or_empty")
    streams = _rows(envelope.get("available_sensor_streams"))
    stream_types: list[str] = []
    for index, stream in enumerate(streams):
        stream_type = str(stream.get("stream_type") or "").strip()
        if not stream_type:
            errors.append(f"available_sensor_streams[{index}].stream_type:missing")
        stream_types.append(stream_type)
        if str(stream.get("status") or "") not in {"available", "diagnostic", "unavailable"}:
            errors.append(f"available_sensor_streams[{index}].status:invalid")
        source_path = stream.get("source_relative_path")
        if source_path is not None and _safe_relative_path(source_path) not in seen_paths:
            errors.append(f"available_sensor_streams[{index}].source_relative_path:unknown")
    if len(set(stream_types)) != len(stream_types):
        errors.append("available_sensor_streams:duplicate_stream_type")
    governance = envelope.get("governance")
    if not isinstance(governance, Mapping):
        errors.append("governance:missing")
    else:
        for key in (
            "rights",
            "consent",
            "privacy",
            "retention",
            "revocation",
            "provider_constraints",
            "allowed_uses",
        ):
            if key not in governance:
                errors.append(f"governance.{key}:missing")
    for key in (
        "permitted_reconstruction_providers",
        "permitted_evidence_uses",
        "operator_notes",
    ):
        if not isinstance(envelope.get(key), list):
            errors.append(f"{key}:must_be_list")
    for key in ("upload_validation", "malware_content_validation"):
        if not isinstance(envelope.get(key), Mapping):
            errors.append(f"{key}:missing")
    if profile == "precomputed_external_reconstruction":
        binding = envelope.get("source_capture_binding")
        if not isinstance(binding, Mapping) or not _is_sha256(binding.get("source_capture_digest")):
            errors.append("source_capture_binding.source_capture_digest:missing_or_invalid")
    supplied_digest = str(envelope.get("envelope_digest") or "")
    expected_digest = _digest_mapping(envelope, omit="envelope_digest")
    if supplied_digest and supplied_digest != expected_digest:
        errors.append("envelope_digest:mismatch")
    if errors:
        raise CaptureIntakeError(errors)
    envelope["envelope_digest"] = expected_digest
    return envelope


def validate_capture_intake_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached, integrity-checked intake envelope."""

    return _validate_envelope(value)


def _governance_blockers(envelope: Mapping[str, Any]) -> list[str]:
    governance = envelope.get("governance")
    governance = governance if isinstance(governance, Mapping) else {}
    blockers: list[str] = []
    if str(governance.get("rights") or "") != "accepted":
        blockers.append("rights_not_accepted")
    if str(governance.get("consent") or "") not in {"accepted", "not_required"}:
        blockers.append("consent_not_accepted")
    if str(governance.get("privacy") or "") not in {"cleared", "restricted_local_only"}:
        blockers.append("privacy_admission_incomplete")
    if not isinstance(governance.get("retention"), Mapping):
        blockers.append("retention_policy_missing")
    if not isinstance(governance.get("revocation"), Mapping):
        blockers.append("revocation_policy_missing")
    if not _strings(governance.get("allowed_uses")):
        blockers.append("allowed_uses_missing")
    provider_constraints = governance.get("provider_constraints")
    provider_constraints = (
        provider_constraints if isinstance(provider_constraints, Mapping) else {}
    )
    permitted_providers = set(_strings(envelope.get("permitted_reconstruction_providers")))
    if provider_constraints.get("external_processing_allowed") is False and any(
        provider not in {"local", "local_only"} for provider in permitted_providers
    ):
        blockers.append("provider_restriction_conflict")
    if str((envelope.get("upload_validation") or {}).get("status") or "") != "passed":
        blockers.append("upload_validation_not_passed")
    if str((envelope.get("malware_content_validation") or {}).get("status") or "") != "passed":
        blockers.append("malware_content_validation_not_passed")
    return blockers


def _stream_state(envelope: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row.get("stream_type") or ""): str(row.get("status") or "")
        for row in _rows(envelope.get("available_sensor_streams"))
        if str(row.get("stream_type") or "")
    }


def _recapture_plan(profile: str, missing: Sequence[str]) -> list[dict[str, Any]]:
    instructions = {
        "retained_video": "Upload the retained original video from this capture.",
        "retained_original": "Upload the original native 360-camera container; do not supply only a recompressed derivative.",
        "decoded_video_pts": "Re-export decoded video presentation timestamps for the retained video.",
        "frame_retention_mapping": "Repeat the capture/export so every encoded frame is mapped to decoded video PTS and encoder omissions are recorded.",
        "camera_poses": "Repeat the scan with camera-pose recording enabled and include tracking-reset events.",
        "camera_intrinsics": "Repeat the scan/export with per-format camera intrinsics and distortion metadata.",
        "depth": "Repeat the iPhone Pro scan with LiDAR depth recording enabled.",
        "depth_confidence": "Repeat the iPhone Pro export with the depth-confidence stream retained.",
        "tracking_state": "Repeat the scan while retaining tracking, relocalization, and dropped-frame state.",
        "coordinate_frame_semantics": "Export the site/world coordinate-frame, gravity, up-axis, handedness, and transform semantics.",
        "camera_metadata": "Export the 360 camera model, stitch/equirectangular layout, orientation, and firmware metadata.",
        "external_reconstruction": "Attach the external reconstruction files and their provider/runtime manifest.",
    }
    return [
        {
            "code": f"missing_{stream}",
            "instruction": instructions.get(stream, f"Provide the required {stream} evidence."),
            "reason": f"{profile} requires a verified {stream} stream for its declared authority profile.",
        }
        for stream in sorted(set(missing))
    ]


def _claim_ceiling(profile: str, streams: Mapping[str, str], *, admitted: bool) -> dict[str, Any]:
    available = {key for key, status in streams.items() if status == "available"}
    calibrated_pose = {
        "camera_poses",
        "camera_intrinsics",
        "decoded_video_pts",
        "frame_retention_mapping",
    }.issubset(available)
    metric_scale = profile == "iphone_arkit_lidar" and calibrated_pose and {
        "depth",
        "depth_confidence",
        "coordinate_frame_semantics",
    }.issubset(available)
    if profile == "iphone_arkit_non_lidar":
        metric_scale = calibrated_pose and "verified_scale_anchor" in available
    observed_video = bool(
        {"retained_video", "retained_original", "processed_rgb_observations"}
        & available
    )
    return {
        "capture_admitted": admitted,
        "task_candidate_discovery": admitted and observed_video,
        "captured_observation_review": admitted and observed_video,
        "calibrated_camera_poses": admitted and calibrated_pose,
        "metric_scale": admitted and metric_scale,
        "metric_geometry": admitted and metric_scale,
        "collision_geometry": False,
        "contact_or_articulation": False,
        "physical_task_success": False,
        "deployment_readiness": False,
        "safety_certification": False,
        "comparative_policy_ranking": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
        "external_reconstruction_is_raw_capture_authority": False,
        "generated_completion_upgrades_observed_or_physical_claims": False,
    }


def build_capture_admission(envelope_value: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic admission report from a validated intake envelope."""

    envelope = _validate_envelope(envelope_value)
    profile = str(envelope["capture_authority_profile"])
    streams = _stream_state(envelope)
    missing = sorted(
        stream for stream in _REQUIRED_STREAMS[profile] if streams.get(stream) != "available"
    )
    blockers = _governance_blockers(envelope)
    if blockers:
        status = "rejected"
    elif missing:
        status = "recapture_required"
    else:
        status = "accepted"
    reduced_authority_reasons: list[str] = []
    if profile in {"camera_360_equirectangular", "camera_360_native", "monocular_video"}:
        reduced_authority_reasons.extend(
            [
                "metric_scale_not_inherent_in_video",
                "camera_poses_not_inherent_in_video",
                "depth_or_collision_not_inherent_in_video",
                "physical_outcomes_not_observed",
            ]
        )
    if profile == "precomputed_external_reconstruction":
        reduced_authority_reasons.append("derived_reconstruction_cannot_replace_source_capture_authority")
    if profile == "public_processed_rgbd_pose_sequence":
        reduced_authority_reasons.extend(
            [
                "public_processed_dataset_is_not_customer_capture",
                "original_video_and_encoder_retention_truth_unavailable",
                "dataset_camera_and_depth_calibration_not_independently_verified",
                "physical_outcomes_not_observed",
            ]
        )
    admitted = status == "accepted"
    report = {
        "schema_version": CAPTURE_ADMISSION_SCHEMA_VERSION,
        "intake_id": envelope["intake_id"],
        "idempotency_key": envelope["idempotency_key"],
        "envelope_digest": envelope["envelope_digest"],
        "capture_authority_profile": profile,
        "status": status,
        "state": "capture_accepted" if admitted else (
            "rejected_or_recapture_required" if status == "recapture_required" else "failed"
        ),
        "governance_blockers": sorted(blockers),
        "missing_required_streams": missing,
        "recapture_plan": _recapture_plan(profile, missing),
        "reduced_authority_reasons": sorted(reduced_authority_reasons),
        "claim_ceiling": _claim_ceiling(profile, streams, admitted=admitted),
        "permitted_reconstruction_providers": sorted(
            _strings(envelope.get("permitted_reconstruction_providers"))
        ),
        "permitted_evidence_uses": sorted(_strings(envelope.get("permitted_evidence_uses"))),
        "provider_execution_authorized": False,
        "physical_execution_authorized": False,
        "raw_capture_authority_rewritten": False,
    }
    report["admission_digest"] = _digest_mapping(report, omit="admission_digest")
    return report


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise CaptureIntakeError([f"immutable_artifact_conflict:{path.name}"])


def _store_object(source: Path, object_path: Path, expected_digest: str) -> None:
    object_path.parent.mkdir(parents=True, exist_ok=True)
    if object_path.exists():
        if not object_path.is_file() or _file_digest(object_path) != expected_digest:
            raise CaptureIntakeError(["content_addressed_object_conflict"])
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=".capture-object-", dir=object_path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if _file_digest(temporary) != expected_digest:
            raise CaptureIntakeError(["content_addressed_copy_digest_mismatch"])
        try:
            os.link(temporary, object_path)
        except FileExistsError:
            if _file_digest(object_path) != expected_digest:
                raise CaptureIntakeError(["content_addressed_object_conflict"])
        object_path.chmod(0o440)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class MaterializedCaptureIntake:
    envelope: dict[str, Any]
    admission: dict[str, Any]
    artifact_root: Path
    content_objects: tuple[dict[str, Any], ...]


def verify_capture_intake_bytes(
    envelope_value: Mapping[str, Any], *, upload_root: Path
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Verify every declared original against regular files beneath ``upload_root``."""

    envelope = _validate_envelope(envelope_value)
    upload_root = upload_root.resolve()
    errors: list[str] = []
    verified: list[dict[str, Any]] = []
    for index, row in enumerate(_rows(envelope.get("original_files"))):
        relative = _safe_relative_path(row.get("relative_path"))
        if relative is None:
            errors.append(f"original_files[{index}].relative_path:unsafe")
            continue
        source = upload_root / relative
        try:
            resolved = source.resolve(strict=True)
        except FileNotFoundError:
            errors.append(f"original_files[{index}]:missing")
            continue
        if upload_root != resolved and upload_root not in resolved.parents:
            errors.append(f"original_files[{index}]:outside_upload_root")
            continue
        if source.is_symlink() or not resolved.is_file():
            errors.append(f"original_files[{index}]:not_regular_file")
            continue
        expected_size = int(row["size_bytes"])
        if resolved.stat().st_size != expected_size:
            errors.append(f"original_files[{index}]:size_mismatch")
            continue
        digest = _file_digest(resolved)
        if digest != row["sha256"]:
            errors.append(f"original_files[{index}]:digest_mismatch")
            continue
        verified.append(
            {
                "original_filename": row["original_filename"],
                "source_relative_path": relative,
                "sha256": digest,
                "size_bytes": expected_size,
                "source_path": str(resolved),
            }
        )
    if errors:
        raise CaptureIntakeError(errors)
    return envelope, tuple(verified)


def materialize_capture_intake(
    envelope_value: Mapping[str, Any], *, upload_root: Path, store_root: Path
) -> MaterializedCaptureIntake:
    """Verify upload bytes and persist an idempotent content-addressed intake."""

    envelope, verified_objects = verify_capture_intake_bytes(
        envelope_value, upload_root=upload_root
    )
    idempotency_hash = hashlib.sha256(str(envelope["idempotency_key"]).encode("utf-8")).hexdigest()
    idempotency_path = store_root / "idempotency" / f"{idempotency_hash}.json"
    if idempotency_path.is_file():
        try:
            existing_binding = json.loads(idempotency_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaptureIntakeError(["idempotency_binding_invalid"]) from exc
        if (
            existing_binding.get("envelope_digest") != envelope["envelope_digest"]
            or existing_binding.get("intake_id") != envelope["intake_id"]
        ):
            raise CaptureIntakeError(["idempotency_key_reuse_with_different_envelope"])
    objects: list[dict[str, Any]] = []
    for row in verified_objects:
        relative = str(row["source_relative_path"])
        resolved = Path(str(row["source_path"]))
        expected_size = int(row["size_bytes"])
        digest = str(row["sha256"])
        suffix = digest.removeprefix(_SHA256)
        object_path = store_root / "objects" / "sha256" / suffix[:2] / suffix
        _store_object(resolved, object_path, digest)
        objects.append(
            {
                "original_filename": row["original_filename"],
                "source_relative_path": relative,
                "sha256": digest,
                "size_bytes": expected_size,
                "object_path": str(object_path.relative_to(store_root)),
            }
        )
    admission = build_capture_admission(envelope)
    digest_suffix = envelope["envelope_digest"].removeprefix(_SHA256)
    artifact_root = store_root / "intakes" / str(envelope["intake_id"]) / digest_suffix
    _write_json_once(artifact_root / "capture_intake_envelope.json", envelope)
    _write_json_once(artifact_root / "capture_intake_admission.json", admission)
    object_manifest = {
        "schema_version": "capture_intake_object_manifest.v1",
        "envelope_digest": envelope["envelope_digest"],
        "objects": sorted(objects, key=lambda row: (row["sha256"], row["source_relative_path"])),
        "raw_inputs_content_addressed": True,
        "raw_inputs_mutated": False,
    }
    object_manifest["manifest_digest"] = _digest_mapping(object_manifest, omit="manifest_digest")
    _write_json_once(artifact_root / "capture_intake_object_manifest.json", object_manifest)
    _write_json_once(
        idempotency_path,
        {
            "schema_version": "capture_intake_idempotency_binding.v1",
            "idempotency_key_sha256": _SHA256 + idempotency_hash,
            "intake_id": envelope["intake_id"],
            "envelope_digest": envelope["envelope_digest"],
            "artifact_root": str(artifact_root.relative_to(store_root)),
        },
    )
    return MaterializedCaptureIntake(
        envelope=envelope,
        admission=admission,
        artifact_root=artifact_root,
        content_objects=tuple(object_manifest["objects"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--upload-root", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    args = parser.parse_args(argv)
    value = json.loads(args.envelope.read_text(encoding="utf-8"))
    result = materialize_capture_intake(
        value, upload_root=args.upload_root, store_root=args.store_root
    )
    print(
        json.dumps(
            {
                "artifact_root": str(result.artifact_root),
                "admission": result.admission,
                "content_objects": list(result.content_objects),
            },
            sort_keys=True,
        )
    )
    return 0 if result.admission["status"] == "accepted" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CAPTURE_ADMISSION_SCHEMA_VERSION",
    "CAPTURE_AUTHORITY_PROFILES",
    "CAPTURE_INTAKE_SCHEMA_VERSION",
    "CaptureIntakeError",
    "MaterializedCaptureIntake",
    "build_capture_admission",
    "materialize_capture_intake",
    "verify_capture_intake_bytes",
]
