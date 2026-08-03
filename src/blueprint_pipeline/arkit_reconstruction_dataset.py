"""Compile strict ARKit observations into a reconstruction request.

This compiler joins only already-frozen decoded observations to a previously
validated ARKit metric scaffold. It does not refine poses, filter depth,
validate metric scale, train a reconstruction, or evaluate held-out quality.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json


ARKIT_RECONSTRUCTION_DATASET_SCHEMA_VERSION = "arkit_reconstruction_dataset_export.v1"
ARKIT_RECONSTRUCTION_DATASET_REQUEST_SCHEMA_VERSION = (
    "arkit_reconstruction_dataset_export_request.v1"
)
CAMERA_OBSERVATION_SCHEMA_VERSION = "camera_observation_manifest.v1"
CAMERA_CALIBRATION_SCHEMA_VERSION = "camera_calibration_manifest.v1"
POSE_REFINEMENT_REQUEST_SCHEMA_VERSION = "pose_refinement_request.v1"
ARKIT_CAMERA_AXIS_CONVENTION = "arkit_x_right_y_up_z_backward"


class ArkitReconstructionDatasetError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def build_arkit_reconstruction_dataset_export_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and digest-bind the trusted input to the ARKit exporter."""

    request = json.loads(canonical_json(dict(value)))
    errors: list[str] = []
    if request.get("schema_version") != ARKIT_RECONSTRUCTION_DATASET_REQUEST_SCHEMA_VERSION:
        errors.append("arkit_export_request_schema_invalid")
    if not str(request.get("intake_id") or "").strip() or not _is_digest(
        request.get("source_capture_digest")
    ):
        errors.append("arkit_export_request_source_binding_invalid")
    if not _is_digest(request.get("implementation_digest")):
        errors.append("arkit_export_request_implementation_invalid")
    source_commit_sha = str(request.get("source_commit_sha") or "")
    if len(source_commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit_sha
    ):
        errors.append("arkit_export_request_source_commit_invalid")
    for key in (
        "dataset_manifest",
        "split_manifest",
        "candidate_manifest",
        "metric_scaffold",
        "authority_used",
    ):
        if not isinstance(request.get(key), Mapping):
            errors.append(f"arkit_export_request_{key}_invalid")
    if not _is_digest(request.get("metric_scaffold_digest")):
        errors.append("arkit_export_request_metric_scaffold_digest_invalid")
    if not str(request.get("timestamp") or "").strip():
        errors.append("arkit_export_request_timestamp_missing")
    if isinstance(request.get("candidate_manifest"), Mapping) and request[
        "candidate_manifest"
    ].get("heldout_pixels_included") is not False:
        errors.append("arkit_export_request_hidden_heldout_exposed")
    if not errors:
        _validate_source_artifacts(
            capture_digest=request["source_capture_digest"],
            dataset=request["dataset_manifest"],
            split=request["split_manifest"],
            candidate=request["candidate_manifest"],
            scaffold=request["metric_scaffold"],
            scaffold_digest=request["metric_scaffold_digest"],
        )
    supplied_digest = request.pop("arkit_export_request_digest", None)
    request["arkit_export_request_digest"] = canonical_digest(
        request, digest_field="arkit_export_request_digest"
    )
    if supplied_digest is not None and supplied_digest != request["arkit_export_request_digest"]:
        errors.append("arkit_export_request_digest_mismatch")
    if errors:
        raise ArkitReconstructionDatasetError(errors)
    return request


def export_bound_arkit_reconstruction_dataset(
    *, source_artifact: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Execute the local exporter from a validated, immutable request."""

    request = build_arkit_reconstruction_dataset_export_request(source_artifact)
    return compile_arkit_reconstruction_dataset(
        output_root=output_root,
        intake_id=request["intake_id"],
        capture_digest=request["source_capture_digest"],
        dataset_manifest=request["dataset_manifest"],
        split_manifest=request["split_manifest"],
        candidate_manifest=request["candidate_manifest"],
        metric_scaffold=request["metric_scaffold"],
        metric_scaffold_digest=request["metric_scaffold_digest"],
        implementation_digest=request["implementation_digest"],
        source_commit_sha=request["source_commit_sha"],
        authority_used=request["authority_used"],
        timestamp=request.get("timestamp"),
    )


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _canonical_artifact_digest(value: Mapping[str, Any]) -> str:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_immutable(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json(dict(value)))
    payload = (canonical_json(normalized) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(normalized):
            raise ArkitReconstructionDatasetError(["arkit_export_immutable_conflict"])
        return dict(existing)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json(existing) != canonical_json(normalized):
                raise ArkitReconstructionDatasetError(["arkit_export_immutable_conflict"])
            return dict(existing)
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


def _validate_source_artifacts(
    *,
    capture_digest: str,
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
    candidate: Mapping[str, Any],
    scaffold: Mapping[str, Any],
    scaffold_digest: str,
) -> None:
    errors: list[str] = []
    if dataset.get("schema_version") != "reconstruction_dataset_manifest.v1" or dataset.get(
        "dataset_manifest_digest"
    ) != canonical_digest(dataset, digest_field="dataset_manifest_digest"):
        errors.append("arkit_export_dataset_manifest_invalid")
    if split.get("schema_version") != "frozen_reconstruction_split_manifest.v1" or split.get(
        "split_digest"
    ) != canonical_digest(split, digest_field="split_digest"):
        errors.append("arkit_export_split_manifest_invalid")
    if candidate.get("schema_version") != "candidate_reconstruction_dataset_manifest.v1" or candidate.get(
        "candidate_dataset_digest"
    ) != canonical_digest(candidate, digest_field="candidate_dataset_digest"):
        errors.append("arkit_export_candidate_manifest_invalid")
    if (
        scaffold.get("schema_version") != "arkit_metric_scaffold.v1"
        or not _is_digest(scaffold_digest)
        or scaffold_digest != _canonical_artifact_digest(scaffold)
    ):
        errors.append("arkit_export_metric_scaffold_invalid")
    if any(
        value != capture_digest
        for value in (
            dataset.get("source_capture_digest"),
            split.get("capture_digest"),
            candidate.get("capture_digest"),
            scaffold.get("capture_digest"),
        )
    ):
        errors.append("arkit_export_capture_binding_mismatch")
    if dataset.get("train_heldout_split_digest") != split.get("split_digest") or candidate.get(
        "split_digest"
    ) != split.get("split_digest"):
        errors.append("arkit_export_split_binding_mismatch")
    if candidate.get("heldout_pixels_included") is not False or split.get(
        "candidate_can_change_assignments"
    ) is not False:
        errors.append("arkit_export_hidden_split_contract_invalid")
    if errors:
        raise ArkitReconstructionDatasetError(errors)


def compile_arkit_reconstruction_dataset(
    *,
    output_root: str | Path,
    intake_id: str,
    capture_digest: str,
    dataset_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    metric_scaffold: Mapping[str, Any],
    metric_scaffold_digest: str,
    implementation_digest: str,
    source_commit_sha: str,
    authority_used: Mapping[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Create candidate-only calibrated observations plus a frozen refinement request."""

    if not str(intake_id).strip() or not _is_digest(capture_digest) or not _is_digest(
        implementation_digest
    ):
        raise ArkitReconstructionDatasetError(["arkit_export_source_binding_invalid"])
    if len(source_commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit_sha
    ):
        raise ArkitReconstructionDatasetError(["arkit_export_source_commit_invalid"])
    _validate_source_artifacts(
        capture_digest=capture_digest,
        dataset=dataset_manifest,
        split=split_manifest,
        candidate=candidate_manifest,
        scaffold=metric_scaffold,
        scaffold_digest=metric_scaffold_digest,
    )
    intrinsics = metric_scaffold.get("intrinsics")
    coordinate_system = metric_scaffold.get("coordinate_system")
    if not isinstance(intrinsics, Mapping) or not isinstance(coordinate_system, Mapping):
        raise ArkitReconstructionDatasetError(["arkit_export_calibration_missing"])
    coordinate_system = {
        **dict(coordinate_system),
        "camera_axis_convention": ARKIT_CAMERA_AXIS_CONVENTION,
    }
    calibration = {
        "schema_version": CAMERA_CALIBRATION_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "coordinate_frame_session_id": metric_scaffold.get("coordinate_frame_session_id"),
        "camera_model": "PINHOLE",
        "intrinsics": dict(intrinsics),
        "distortion": {"status": "not_declared", "training_use_allowed": False},
        "rolling_shutter": {"status": "not_declared", "training_use_allowed": False},
        "pixel_orientation": "encoded_source_no_autorotate",
        "coordinate_system": dict(coordinate_system),
        "source_metric_scaffold_digest": metric_scaffold_digest,
    }
    calibration["calibration_digest"] = canonical_digest(
        calibration, digest_field="calibration_digest"
    )
    camera_by_index = {
        int(row["encoded_frame_index"]): dict(row)
        for row in metric_scaffold.get("camera_frames", [])
        if isinstance(row, Mapping) and isinstance(row.get("encoded_frame_index"), int)
    }
    depth_by_frame = {
        str(row.get("frame_id")): dict(row)
        for row in metric_scaffold.get("depth_confidence_pairs", [])
        if isinstance(row, Mapping) and str(row.get("frame_id") or "")
    }
    observations: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in candidate_manifest.get("frames", []):
        if not isinstance(row, Mapping):
            errors.append("arkit_export_candidate_frame_invalid")
            continue
        index = row.get("decoded_frame_index")
        camera = camera_by_index.get(index) if isinstance(index, int) else None
        image_metadata = row.get("image_metadata")
        if camera is None or not isinstance(image_metadata, Mapping):
            errors.append(f"arkit_export_camera_binding_missing:{index}")
            continue
        if (
            image_metadata.get("pixel_orientation") != "encoded_source_no_autorotate"
            or image_metadata.get("width") != intrinsics.get("width")
            or image_metadata.get("height") != intrinsics.get("height")
        ):
            errors.append(f"arkit_export_intrinsics_pixel_binding_mismatch:{index}")
            continue
        if abs(float(row.get("t_video_sec")) - float(camera.get("t_video_sec"))) > 0.0001:
            errors.append(f"arkit_export_pts_binding_mismatch:{index}")
            continue
        capture_frame_id = str(camera.get("frame_id") or "")
        depth = depth_by_frame.get(capture_frame_id)
        observations.append(
            {
                "observation_id": row["frame_id"],
                "capture_frame_id": capture_frame_id,
                "decoded_frame_index": index,
                "t_video_sec": row["t_video_sec"],
                "t_capture_sec": camera.get("t_capture_sec"),
                "split": row["split"],
                "image_relative_path": row["candidate_relative_path"],
                "image_digest": row["frame_digest"],
                "T_world_camera": camera.get("T_world_camera"),
                # Keep the Blueprint-native fields above and also materialize
                # the exact generic camera binding consumed by the shared
                # candidate-only COLMAP exporter. This is a projection of the
                # same accepted values, not a second calibration or pose.
                "camera": {
                    "camera_model": "PINHOLE",
                    "T_world_camera": camera.get("T_world_camera"),
                    "camera_axis_convention": ARKIT_CAMERA_AXIS_CONVENTION,
                    "rgb_intrinsics": dict(intrinsics),
                },
                "calibration_digest": calibration["calibration_digest"],
                "depth_confidence_binding": depth,
            }
        )
    if errors:
        raise ArkitReconstructionDatasetError(errors)
    if not observations:
        raise ArkitReconstructionDatasetError(["arkit_export_candidate_observations_missing"])
    observation_manifest = {
        "schema_version": CAMERA_OBSERVATION_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "dataset_manifest_digest": dataset_manifest["dataset_manifest_digest"],
        "split_digest": split_manifest["split_digest"],
        "calibration_digest": calibration["calibration_digest"],
        "candidate_splits_only": True,
        "hidden_heldout_pixels_included": False,
        "observations": observations,
    }
    observation_manifest["camera_observation_digest"] = canonical_digest(
        observation_manifest, digest_field="camera_observation_digest"
    )
    refinement_request = {
        "schema_version": POSE_REFINEMENT_REQUEST_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "camera_observation_digest": observation_manifest["camera_observation_digest"],
        "calibration_digest": calibration["calibration_digest"],
        "split_digest": split_manifest["split_digest"],
        "initial_pose_source": "verified_arkit_raw_contract_3_2",
        "refinement_policy": "arkit_anchored_bounded_bundle_adjustment_required",
        "maximum_pose_drift_threshold": None,
        "status": "blocked_qa_threshold_not_registered",
        "candidate_may_change_input_poses": False,
        "candidate_may_access_hidden_heldout": False,
    }
    refinement_request["pose_refinement_request_digest"] = canonical_digest(
        refinement_request, digest_field="pose_refinement_request_digest"
    )
    configuration_digest = canonical_digest(
        {
            "compiler": ARKIT_RECONSTRUCTION_DATASET_SCHEMA_VERSION,
            "dataset_manifest_digest": dataset_manifest["dataset_manifest_digest"],
            "split_digest": split_manifest["split_digest"],
            "metric_scaffold_digest": metric_scaffold_digest,
            "implementation_digest": implementation_digest,
            "source_commit_sha": source_commit_sha,
        }
    )
    export_root = Path(output_root).expanduser().resolve()
    root = export_root / f"arkit_export_{configuration_digest[7:23]}"
    calibration_path = root / "camera_calibration_manifest.json"
    observation_path = root / "candidate_camera_observation_manifest.json"
    pose_request_path = root / "pose_refinement_request.json"
    colmap_request_path = root / "colmap_training_dataset_export_request.json"
    calibration = _write_immutable(calibration_path, calibration)
    observation_manifest = _write_immutable(
        observation_path, observation_manifest
    )
    refinement_request = _write_immutable(pose_request_path, refinement_request)
    compiled_at = timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    colmap_request = {
        "schema_version": "colmap_training_dataset_export_request.v1",
        "stable_run_identity": f"arkit-colmap-request-{configuration_digest[7:31]}",
        "source_capture_digest": capture_digest,
        "source_commit_sha": source_commit_sha,
        "reconstruction_dataset_digest": dataset_manifest["dataset_manifest_digest"],
        "frozen_split_digest": split_manifest["split_digest"],
        "camera_observation_manifest": observation_manifest,
        "camera_calibration_manifest": calibration,
        "candidate_dataset_manifest": dict(candidate_manifest),
        "maximum_initialization_points": 100_000,
        "coordinate_frame_declaration": dict(coordinate_system),
        "units": "meters",
        "metric_scale_status": "sensor_metric_unvalidated",
        "authority_used": dict(authority_used),
        "timestamp": compiled_at,
        "blockers": [
            "initialization_surface_not_bound",
            "pose_refinement_not_executed",
        ],
    }
    colmap_request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        colmap_request,
        digest_field="colmap_training_dataset_export_request_digest",
    )
    colmap_request = _write_immutable(
        colmap_request_path, colmap_request
    )
    artifact_references = [
        {
            "artifact_type": artifact_type,
            "relative_path": path.relative_to(export_root).as_posix(),
            "artifact_digest": _file_digest(path),
            "content_digest": content_digest,
        }
        for artifact_type, path, content_digest in (
            (
                CAMERA_CALIBRATION_SCHEMA_VERSION,
                calibration_path,
                calibration["calibration_digest"],
            ),
            (
                CAMERA_OBSERVATION_SCHEMA_VERSION,
                observation_path,
                observation_manifest["camera_observation_digest"],
            ),
            (
                POSE_REFINEMENT_REQUEST_SCHEMA_VERSION,
                pose_request_path,
                refinement_request["pose_refinement_request_digest"],
            ),
            (
                "colmap_training_dataset_export_request.v1",
                colmap_request_path,
                colmap_request["colmap_training_dataset_export_request_digest"],
            ),
        )
    ]
    export = {
        "schema_version": ARKIT_RECONSTRUCTION_DATASET_SCHEMA_VERSION,
        "stable_run_identity": f"arkit-export-{configuration_digest[7:31]}",
        "source_capture_identity": intake_id,
        "source_capture_digest": capture_digest,
        "source_commit_sha": source_commit_sha,
        "implementation_version": implementation_digest,
        "deterministic_configuration_digest": configuration_digest,
        "reconstruction_dataset_digest": dataset_manifest["dataset_manifest_digest"],
        "frozen_split_digest": split_manifest["split_digest"],
        "camera_calibration_digest": calibration["calibration_digest"],
        "camera_observation_digest": observation_manifest["camera_observation_digest"],
        "pose_refinement_request_digest": refinement_request[
            "pose_refinement_request_digest"
        ],
        "colmap_training_dataset_export_request_digest": colmap_request[
            "colmap_training_dataset_export_request_digest"
        ],
        "artifact_references": artifact_references,
        "metric_scaffold_digest": metric_scaffold_digest,
        "candidate_observation_count": len(observations),
        "hidden_heldout_pixels_included": False,
        "raw_arkit_poses_modified": False,
        "depth_confidence_filtering_status": "not_executed",
        "metric_scale_validation_status": "not_executed",
        "colmap_gsplat_export_status": "candidate_only_raw_arkit_pose_request_ready",
        "authority_used": dict(authority_used),
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "warnings": ["local_compilation_duration_not_measured"],
        "blockers": sorted(
            set(dataset_manifest.get("blockers", []))
            | {"pose_refinement_qa_threshold_not_registered"}
        ),
        "proof_effect": "calibrated_reconstruction_request_only",
        "claim_ceiling": "calibrated_camera_trajectory",
        "parent_artifact_or_event": {
            "dataset_manifest_digest": dataset_manifest["dataset_manifest_digest"],
            "metric_scaffold_digest": metric_scaffold_digest,
        },
        "timestamp": compiled_at,
    }
    export["arkit_reconstruction_dataset_export_digest"] = canonical_digest(
        export, digest_field="arkit_reconstruction_dataset_export_digest"
    )
    return _write_immutable(root / "arkit_reconstruction_dataset_export.json", export)


__all__ = [
    "ARKIT_RECONSTRUCTION_DATASET_SCHEMA_VERSION",
    "ARKIT_RECONSTRUCTION_DATASET_REQUEST_SCHEMA_VERSION",
    "ArkitReconstructionDatasetError",
    "CAMERA_CALIBRATION_SCHEMA_VERSION",
    "CAMERA_OBSERVATION_SCHEMA_VERSION",
    "POSE_REFINEMENT_REQUEST_SCHEMA_VERSION",
    "build_arkit_reconstruction_dataset_export_request",
    "compile_arkit_reconstruction_dataset",
    "export_bound_arkit_reconstruction_dataset",
]
