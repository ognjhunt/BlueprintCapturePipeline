"""Bounded file/runtime entrypoint for the official SAM 3.1 multiplex adapter."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import read_json_any, sha256_file, write_json
from .scene_placement.sam31_source_track_provider import (
    blocked_sam31_source_track_run,
    execute_sam31_source_track_request,
)
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest


_REQUEST_MAX_BYTES = 16 * 1024 * 1024
_FRAME_MAX_BYTES = 100 * 1024 * 1024
_TOTAL_FRAME_MAX_BYTES = 64 * 1024 * 1024 * 1024
_CHECKPOINT_PATH_ENV = "BLUEPRINT_SAM31_CHECKPOINT_PATH"
_CODE_REVISION_ENV = "BLUEPRINT_SAM31_OFFICIAL_CODE_REVISION"
_RUNTIME_DIGEST_ENV = "BLUEPRINT_SAM31_RUNTIME_DIGEST"


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _normalized_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text[7:] if text.startswith("sha256:") else text


def _load_request(path: Path, blockers: list[str]) -> Dict[str, Any]:
    if path.is_symlink():
        blockers.append("input_symlink_forbidden:request")
        return {}
    if not path.is_file():
        blockers.append("input_missing:request")
        return {}
    try:
        size = path.stat().st_size
    except OSError:
        blockers.append("input_stat_failed:request")
        return {}
    if size <= 0 or size > _REQUEST_MAX_BYTES:
        blockers.append("input_size_invalid:request")
        return {}
    try:
        payload = read_json_any(path)
    except (OSError, TypeError, ValueError):
        blockers.append("input_json_invalid:request")
        return {}
    if not isinstance(payload, Mapping):
        blockers.append("input_json_not_object:request")
        return {}
    return dict(payload)


def _output_paths(
    *,
    request_path: Path,
    run_result_path: Path,
    provider_result_path: Path,
    import_request_path: Path,
) -> None:
    inputs = {request_path.resolve(strict=False)}
    outputs = [run_result_path, provider_result_path, import_request_path]
    resolved = [path.resolve(strict=False) for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output_paths_must_be_distinct")
    if any(path in inputs for path in resolved):
        raise ValueError("output_path_must_not_overwrite_an_input")
    for path in outputs:
        if path.is_symlink():
            raise ValueError("output_symlink_forbidden")
        if path.exists():
            raise ValueError("immutable_output_already_exists")


def _frame_artifacts(
    request: Mapping[str, Any],
    frame_root: Path,
    blockers: list[str],
    *,
    forbidden_output_paths: set[Path],
) -> None:
    frames = request.get("frame_registry")
    artifacts = request.get("frame_artifacts")
    if not isinstance(frames, list) or not isinstance(artifacts, list):
        blockers.append("frame_registry_or_artifacts_missing")
        return
    if len(frames) != len(artifacts):
        blockers.append("frame_registry_artifact_count_mismatch")
        return
    by_id: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            blockers.append("frame_artifact_invalid")
            continue
        frame_id = str(artifact.get("source_frame_id") or "").strip()
        if not frame_id or frame_id in by_id:
            blockers.append("frame_artifact_identity_invalid_or_duplicate")
            continue
        by_id[frame_id] = artifact
    total_bytes = 0
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            blockers.append("frame_registry_row_invalid")
            continue
        frame_id = str(frame.get("source_frame_id") or "").strip()
        artifact = by_id.get(frame_id)
        if artifact is None:
            blockers.append(f"frame_artifact_missing:{frame_id}")
            continue
        if artifact.get("media_type") != "image/jpeg":
            blockers.append(f"frame_artifact_media_type_unsupported:{frame_id}")
        raw_path = str(artifact.get("path") or "").strip()
        source = Path(raw_path).expanduser()
        if not source.is_absolute():
            blockers.append(f"frame_artifact_path_not_absolute:{frame_id}")
            continue
        if source.resolve(strict=False) in forbidden_output_paths:
            blockers.append(f"output_overwrites_frame_artifact:{frame_id}")
            continue
        if source.is_symlink() or not source.is_file():
            blockers.append(f"frame_artifact_missing_or_symlink:{frame_id}")
            continue
        if source.suffix.lower() not in {".jpg", ".jpeg"}:
            blockers.append(f"frame_artifact_extension_unsupported:{frame_id}")
        try:
            size = source.stat().st_size
        except OSError:
            blockers.append(f"frame_artifact_stat_failed:{frame_id}")
            continue
        supplied_size = artifact.get("size_bytes")
        if (
            isinstance(supplied_size, bool)
            or not isinstance(supplied_size, int)
            or supplied_size != size
            or size <= 0
            or size > _FRAME_MAX_BYTES
        ):
            blockers.append(f"frame_artifact_size_mismatch:{frame_id}")
            continue
        total_bytes += size
        if total_bytes > _TOTAL_FRAME_MAX_BYTES:
            blockers.append("frame_artifact_total_size_exceeds_limit")
            break
        supplied_sha = artifact.get("sha256")
        expected_sha = frame.get("analysis_jpeg_digest")
        if (
            not _valid_sha256(supplied_sha)
            or not _valid_sha256(expected_sha)
            or _normalized_sha256(supplied_sha) != _normalized_sha256(expected_sha)
        ):
            blockers.append(f"frame_artifact_declared_digest_mismatch:{frame_id}")
            continue
        try:
            actual_sha = sha256_file(source)
        except OSError:
            blockers.append(f"frame_artifact_hash_failed:{frame_id}")
            continue
        if actual_sha != _normalized_sha256(expected_sha):
            blockers.append(f"frame_artifact_sha256_mismatch:{frame_id}")
            continue
        target = frame_root / f"{index:06d}.jpg"
        try:
            shutil.copyfile(source, target)
        except OSError:
            blockers.append(f"frame_artifact_materialization_failed:{frame_id}")
            continue
        try:
            materialized_sha = sha256_file(target)
            materialized_size = target.stat().st_size
        except OSError:
            blockers.append(f"frame_artifact_materialized_verification_failed:{frame_id}")
            continue
        if materialized_sha != actual_sha or materialized_size != size:
            blockers.append(f"frame_artifact_materialized_bytes_mismatch:{frame_id}")


def _official_predictor_factory(profile: Mapping[str, Any]) -> Any:
    configured_revision = str(os.getenv(_CODE_REVISION_ENV) or "").strip().lower()
    expected_revision = str(profile.get("official_code_revision") or "").strip().lower()
    if configured_revision != expected_revision:
        raise ValueError("sam31_installed_code_revision_mismatch")
    configured_runtime = str(os.getenv(_RUNTIME_DIGEST_ENV) or "").strip().lower()
    expected_runtime = str(profile.get("runtime_digest") or "").strip().lower()
    if configured_runtime != expected_runtime:
        raise ValueError("sam31_installed_runtime_digest_mismatch")
    checkpoint = Path(str(os.getenv(_CHECKPOINT_PATH_ENV) or "")).expanduser()
    if not checkpoint.is_absolute() or checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("sam31_checkpoint_missing_or_unsafe")
    expected_checkpoint = _normalized_sha256(profile.get("checkpoint_digest"))
    if sha256_file(checkpoint) != expected_checkpoint:
        raise ValueError("sam31_checkpoint_digest_mismatch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from sam3.model_builder import (  # type: ignore[import-not-found]
            build_sam3_multiplex_video_predictor,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise ValueError("sam31_runtime_not_installed") from exc
    return build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint),
        max_num_objects=int(profile["max_num_objects"]),
        multiplex_count=int(profile["multiplex_count"]),
        use_fa3=bool(profile["use_fa3"]),
        compile=bool(profile["compile"]),
        warm_up=bool(profile["warm_up"]),
        default_output_prob_thresh=float(profile["output_probability_threshold"]),
        async_loading_frames=bool(profile["async_loading_frames"]),
    )


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "sha256": "sha256:" + sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def run_sam31_source_track_stage(
    *,
    request_path: str | Path,
    run_result_path: str | Path,
    provider_result_path: str | Path,
    import_request_path: str | Path,
    predictor_factory: Any = None,
) -> Dict[str, Any]:
    """Verify exact frame/checkpoint inputs and write immutable terminal artifacts."""

    request_file = Path(request_path)
    run_file = Path(run_result_path)
    provider_file = Path(provider_result_path)
    import_file = Path(import_request_path)
    _output_paths(
        request_path=request_file,
        run_result_path=run_file,
        provider_result_path=provider_file,
        import_request_path=import_file,
    )
    blockers: list[str] = []
    request = _load_request(request_file, blockers)
    forbidden_outputs = {
        run_file.resolve(strict=False),
        provider_file.resolve(strict=False),
        import_file.resolve(strict=False),
    }
    with tempfile.TemporaryDirectory(prefix="blueprint_sam31_frames_") as temp_dir:
        frame_root = Path(temp_dir)
        if not blockers:
            _frame_artifacts(
                request,
                frame_root,
                blockers,
                forbidden_output_paths=forbidden_outputs,
            )
        if blockers:
            result = blocked_sam31_source_track_run(request, blockers)
        else:
            result = execute_sam31_source_track_request(
                request,
                predictor_factory=predictor_factory or _official_predictor_factory,
                materialized_frame_directory=frame_root,
            )
    provider = result.pop("provider_result", None)
    import_request = result.pop("source_track_import_request", None)
    if result["status"] in {"completed", "abstained"}:
        if not isinstance(provider, Mapping) or not isinstance(import_request, Mapping):
            result = blocked_sam31_source_track_run(request, ["sam31_terminal_artifacts_missing"])
        else:
            write_json(provider_file, dict(provider))
            import_payload = dict(import_request)
            import_payload["input_artifacts"] = {"provider_result": _artifact(provider_file)}
            write_json(import_file, import_payload)
            result["provider_result_artifact"] = _artifact(provider_file)
            result["source_track_import_request_artifact"] = _artifact(import_file)
    if request_file.is_file() and not request_file.is_symlink():
        result["run_request_artifact"] = _artifact(request_file)
    result.pop("result_digest", None)
    result["result_digest"] = canonical_json_digest(result)
    write_json(run_file, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an authorized local SAM 3.1 multiplex source-track adapter."
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--run-result", required=True)
    parser.add_argument("--provider-result", required=True)
    parser.add_argument("--import-request", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_sam31_source_track_stage(
        request_path=args.request,
        run_result_path=args.run_result,
        provider_result_path=args.provider_result,
        import_request_path=args.import_request,
    )
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["run_sam31_source_track_stage"]
