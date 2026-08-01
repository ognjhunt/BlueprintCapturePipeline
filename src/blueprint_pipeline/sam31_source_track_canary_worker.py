"""Deterministic SAM 3.1 canary bundle builder and isolated GPU worker."""

from __future__ import annotations

import argparse
import json
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from PIL import Image

from .common import read_json_any, sha256_file, write_json
from .decision_evidence_contracts import canonical_digest
from .sam31_source_track_provider_stage import run_sam31_source_track_stage
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest


BUNDLE_MANIFEST_SCHEMA_VERSION = "semantic_sam31_source_track_input_bundle.v1"
BUNDLE_RECEIPT_SCHEMA_VERSION = "semantic_sam31_source_track_input_bundle_receipt.v1"
RUNTIME_RESULT_SCHEMA_VERSION = "semantic_sam31_vast_source_track_result.v1"
MAX_BUNDLE_BYTES = 8 * 1024**3
MAX_MEMBERS = 512
MAX_MEMBER_BYTES = 256 * 1024**2
MAX_TOTAL_UNCOMPRESSED_BYTES = 8 * 1024**3

_CANARY_REQUEST_DIGEST_ENV = "BLUEPRINT_SAM31_CANARY_REQUEST_DIGEST"
_BOUND_REQUEST_DIGEST_ENV = "BLUEPRINT_SAM31_BOUND_REQUEST_DIGEST"
_IMAGE_DIGEST_ENV = "BLUEPRINT_CONTAINER_IMAGE_DIGEST"
_BUNDLE_DIGEST_ENV = "BLUEPRINT_SAM31_INPUT_BUNDLE_DIGEST"
_SOURCE_REQUEST_DIGEST_ENV = "BLUEPRINT_SAM31_SOURCE_TRACK_REQUEST_DIGEST"
_CHECKPOINT_DIGEST_ENV = "BLUEPRINT_SAM31_EXPECTED_CHECKPOINT_DIGEST"


class Sam31CanaryWorkerError(ValueError):
    pass


def _object(path: str | Path) -> dict[str, Any]:
    value = read_json_any(Path(path))
    if not isinstance(value, Mapping):
        raise Sam31CanaryWorkerError(f"expected_json_object:{Path(path).name}")
    return dict(value)


def _normalized_digest(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text[7:] if text.startswith("sha256:") else text


def _digest(value: Any) -> str:
    return "sha256:" + _normalized_digest(value)


def _verified_jpeg(path: Path, *, width: int, height: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise Sam31CanaryWorkerError(f"frame_missing_or_symlink:{path.name}")
    try:
        with Image.open(path) as image:
            if image.format != "JPEG" or image.size != (width, height):
                raise Sam31CanaryWorkerError(f"frame_media_or_dimensions_invalid:{path.name}")
            image.verify()
    except (OSError, ValueError) as exc:
        if isinstance(exc, Sam31CanaryWorkerError):
            raise
        raise Sam31CanaryWorkerError(f"frame_jpeg_invalid:{path.name}") from exc


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def build_sam31_source_track_input_bundle(
    *,
    request_path: str | Path,
    bundle_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Create a byte-deterministic, path-portable frame/request bundle."""

    request_file = Path(request_path).expanduser().resolve()
    bundle_file = Path(bundle_path).expanduser().resolve()
    receipt_file = Path(receipt_path).expanduser().resolve()
    for output in (bundle_file, receipt_file):
        if output.is_symlink() or output.exists():
            raise Sam31CanaryWorkerError("immutable_output_already_exists_or_symlink")
    request = _object(request_file)
    frames = request.get("frame_registry")
    artifacts = request.get("frame_artifacts")
    if not isinstance(frames, list) or not frames or not isinstance(artifacts, list):
        raise Sam31CanaryWorkerError("frame_registry_or_artifacts_missing")
    if len(frames) != len(artifacts):
        raise Sam31CanaryWorkerError("frame_registry_artifact_count_mismatch")
    by_id: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise Sam31CanaryWorkerError("frame_artifact_invalid")
        frame_id = str(artifact.get("source_frame_id") or "").strip()
        if not frame_id or frame_id in by_id:
            raise Sam31CanaryWorkerError("frame_artifact_identity_invalid_or_duplicate")
        by_id[frame_id] = artifact

    portable = json.loads(json.dumps(request))
    portable_artifacts: list[dict[str, Any]] = []
    members: list[tuple[str, bytes]] = []
    frame_members: list[dict[str, Any]] = []
    total = 0
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise Sam31CanaryWorkerError("frame_registry_row_invalid")
        frame_id = str(frame.get("source_frame_id") or "").strip()
        artifact = by_id.get(frame_id)
        if artifact is None:
            raise Sam31CanaryWorkerError(f"frame_artifact_missing:{frame_id}")
        source = Path(str(artifact.get("path") or "")).expanduser()
        if not source.is_absolute():
            raise Sam31CanaryWorkerError(f"frame_artifact_path_not_absolute:{frame_id}")
        size = source.stat().st_size if source.is_file() and not source.is_symlink() else 0
        if size <= 0 or size > MAX_MEMBER_BYTES or artifact.get("size_bytes") != size:
            raise Sam31CanaryWorkerError(f"frame_artifact_size_invalid:{frame_id}")
        expected = _normalized_digest(frame.get("analysis_jpeg_digest"))
        if (
            _normalized_digest(artifact.get("sha256")) != expected
            or sha256_file(source) != expected
        ):
            raise Sam31CanaryWorkerError(f"frame_artifact_digest_mismatch:{frame_id}")
        _verified_jpeg(
            source, width=int(frame.get("width") or 0), height=int(frame.get("height") or 0)
        )
        data = source.read_bytes()
        total += len(data)
        if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise Sam31CanaryWorkerError("frame_artifact_total_size_exceeds_limit")
        member = f"frames/{index:06d}.jpg"
        portable_artifact = {
            "source_frame_id": frame_id,
            "path": member,
            "media_type": "image/jpeg",
            "sha256": _digest(expected),
            "size_bytes": size,
        }
        portable_artifacts.append(portable_artifact)
        frame_members.append({"path": member, "sha256": _digest(expected), "size_bytes": size})
        members.append((member, data))
    portable["frame_artifacts"] = portable_artifacts
    portable_request_digest = canonical_json_digest(portable)
    request_bytes = json.dumps(portable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = {
        "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
        "portable_request_path": "request.json",
        "source_track_run_request_digest": portable_request_digest,
        "frame_count": len(frame_members),
        "frame_members": frame_members,
        "raw_capture_authority_established": False,
        "metric_claim_upgrade_permitted": False,
        "physics_claim_upgrade_permitted": False,
        "physical_claim_upgrade_permitted": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    bundle_file.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_file, "x") as archive:
        archive.writestr(_zip_info("manifest.json"), manifest_bytes)
        archive.writestr(_zip_info("request.json"), request_bytes)
        for member, data in members:
            archive.writestr(_zip_info(member), data)
    bundle_digest = _digest(sha256_file(bundle_file))
    receipt = {
        "schema_version": BUNDLE_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "bundle": {
            "filename": bundle_file.name,
            "sha256": bundle_digest,
            "size_bytes": bundle_file.stat().st_size,
        },
        "manifest_digest": manifest["manifest_digest"],
        "source_track_run_request_digest": portable_request_digest,
        "frame_count": len(frame_members),
        "source_frame_bytes_included": total,
        "source_frame_bytes_returned_by_worker": False,
        "raw_secret_values_recorded": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(receipt_file, receipt)
    return receipt


def _safe_extract_bundle(bundle: Path, destination: Path) -> None:
    if (
        bundle.is_symlink()
        or not bundle.is_file()
        or not 0 < bundle.stat().st_size <= MAX_BUNDLE_BYTES
    ):
        raise Sam31CanaryWorkerError("input_bundle_missing_unsafe_or_too_large")
    with zipfile.ZipFile(bundle, "r") as archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_MEMBERS:
            raise Sam31CanaryWorkerError("input_bundle_member_count_invalid")
        total = 0
        names: set[str] = set()
        for info in infos:
            path = PurePosixPath(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            total += info.file_size
            if (
                info.flag_bits & 0x1
                or info.is_dir()
                or info.file_size <= 0
                or info.file_size > MAX_MEMBER_BYTES
                or total > MAX_TOTAL_UNCOMPRESSED_BYTES
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in info.filename
                or info.filename in names
                or stat.S_ISLNK(mode)
                or not (
                    info.filename in {"manifest.json", "request.json"}
                    or (
                        len(path.parts) == 2
                        and path.parts[0] == "frames"
                        and path.suffix.lower() == ".jpg"
                    )
                )
            ):
                raise Sam31CanaryWorkerError(f"input_bundle_member_unsafe:{info.filename}")
            names.add(info.filename)
        if not {"manifest.json", "request.json"}.issubset(names):
            raise Sam31CanaryWorkerError("input_bundle_required_members_missing")
        for info in infos:
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source:
                data = source.read(MAX_MEMBER_BYTES + 1)
            if len(data) != info.file_size:
                raise Sam31CanaryWorkerError(f"input_bundle_member_size_mismatch:{info.filename}")
            target.write_bytes(data)


def _required_env(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise Sam31CanaryWorkerError(f"required_environment_missing:{name}")
    return value


def _runtime_facts() -> dict[str, Any]:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        return {
            "torch_version": str(torch.__version__),
            "cuda_available": cuda,
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_name": str(torch.cuda.get_device_name(0)) if cuda else None,
        }
    except (ImportError, RuntimeError) as exc:
        return {"cuda_available": False, "error_type": type(exc).__name__}


def run_sam31_source_track_canary_worker(
    *, input_bundle: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Verify a portable bundle, run the local stage, and emit a bounded envelope."""

    bundle = Path(input_bundle)
    output = Path(output_path)
    if output.is_symlink() or output.exists():
        raise Sam31CanaryWorkerError("immutable_output_already_exists_or_symlink")
    expected_bundle = _required_env(_BUNDLE_DIGEST_ENV)
    if _digest(sha256_file(bundle)) != expected_bundle:
        raise Sam31CanaryWorkerError("input_bundle_digest_mismatch")
    root = output.parent / "sam31_canary_materialized"
    if root.exists() or root.is_symlink():
        raise Sam31CanaryWorkerError("materialization_root_already_exists_or_symlink")
    root.mkdir(parents=True)
    _safe_extract_bundle(bundle, root)
    manifest = _object(root / "manifest.json")
    supplied_manifest_digest = manifest.get("manifest_digest")
    if supplied_manifest_digest != canonical_digest(manifest, digest_field="manifest_digest"):
        raise Sam31CanaryWorkerError("input_bundle_manifest_digest_mismatch")
    if manifest.get("schema_version") != BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise Sam31CanaryWorkerError("input_bundle_manifest_schema_invalid")
    portable = _object(root / "request.json")
    portable_digest = canonical_json_digest(portable)
    if portable_digest != manifest.get(
        "source_track_run_request_digest"
    ) or portable_digest != _required_env(_SOURCE_REQUEST_DIGEST_ENV):
        raise Sam31CanaryWorkerError("source_track_run_request_digest_mismatch")
    artifacts = portable.get("frame_artifacts")
    frames = portable.get("frame_registry")
    if (
        not isinstance(artifacts, list)
        or not isinstance(frames, list)
        or len(artifacts) != len(frames)
    ):
        raise Sam31CanaryWorkerError("portable_frame_registry_invalid")
    runtime_request = json.loads(json.dumps(portable))
    for artifact in runtime_request["frame_artifacts"]:
        relative = PurePosixPath(str(artifact.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("frames",):
            raise Sam31CanaryWorkerError("portable_frame_path_invalid")
        source = root.joinpath(*relative.parts)
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != artifact.get("size_bytes")
            or _digest(sha256_file(source)) != artifact.get("sha256")
        ):
            raise Sam31CanaryWorkerError("portable_frame_binding_mismatch")
        artifact["path"] = str(source.resolve())
    runtime_request_path = root / "runtime_request.json"
    write_json(runtime_request_path, runtime_request)
    run_path = root / "run_result.json"
    provider_path = root / "provider_result.json"
    import_path = root / "source_track_import_request.json"
    stage = run_sam31_source_track_stage(
        request_path=runtime_request_path,
        run_result_path=run_path,
        provider_result_path=provider_path,
        import_request_path=import_path,
    )
    provider_result = _object(provider_path) if provider_path.is_file() else None
    import_request = _object(import_path) if import_path.is_file() else None
    runtime = _runtime_facts()
    blockers = list(stage.get("blockers") or [])
    if runtime.get("cuda_available") is not True:
        blockers.append("sam31_gpu_cuda_runtime_unavailable")
    status = (
        "passed" if stage.get("status") in {"completed", "abstained"} and not blockers else "failed"
    )
    result = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": status,
        "request_digest": _required_env(_CANARY_REQUEST_DIGEST_ENV),
        "bound_request_digest": _required_env(_BOUND_REQUEST_DIGEST_ENV),
        "worker_image_digest": _required_env(_IMAGE_DIGEST_ENV),
        "input_bundle_digest": expected_bundle,
        "source_track_run_request_digest": portable_digest,
        "checkpoint_digest": _required_env(_CHECKPOINT_DIGEST_ENV),
        "runtime": runtime,
        "stage_run_result": stage,
        "provider_result": provider_result,
        "source_track_import_request": import_request,
        "blockers": sorted(set(blockers)),
        "source_frame_bytes_returned": False,
        "raw_secret_values_recorded": False,
        "network_access_during_inference": False,
        "directly_observed_object_fact": False,
        "metric_box_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "physical_task_success_established": False,
        "model_self_grading_permitted": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "source_bound_2d_binary_mask_tracks_only"
        if status == "passed"
        else "none",
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    result["runtime_result_digest"] = canonical_digest(result, digest_field="runtime_result_digest")
    write_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-bundle")
    build.add_argument("--request", required=True)
    build.add_argument("--bundle", required=True)
    build.add_argument("--receipt", required=True)
    run = commands.add_parser("run")
    run.add_argument("--input-bundle", required=True)
    run.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-bundle":
        build_sam31_source_track_input_bundle(
            request_path=args.request, bundle_path=args.bundle, receipt_path=args.receipt
        )
        return 0
    result = run_sam31_source_track_canary_worker(
        input_bundle=args.input_bundle, output_path=args.output
    )
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BUNDLE_MANIFEST_SCHEMA_VERSION",
    "BUNDLE_RECEIPT_SCHEMA_VERSION",
    "RUNTIME_RESULT_SCHEMA_VERSION",
    "Sam31CanaryWorkerError",
    "build_sam31_source_track_input_bundle",
    "run_sam31_source_track_canary_worker",
]
