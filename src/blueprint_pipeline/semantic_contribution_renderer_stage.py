"""Bounded file entrypoint for the standard-3DGS semantic renderer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import read_json_any, sha256_file, write_json
from .gaussian_splat_decode import read_standard_3dgs_ply
from .scene_placement.semantic_contribution_renderer import (
    blocked_semantic_contribution_render,
    render_semantic_contributions,
)
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest


_REQUEST_MAX_BYTES = 2 * 1024 * 1024
_INPUT_LIMITS = {
    "analysis_splat": 16 * 1024 * 1024 * 1024,
    "gaussian_mapping": 256 * 1024 * 1024,
    "source_tracks": 512 * 1024 * 1024,
    "camera_records": 64 * 1024 * 1024,
}


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _normalized_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text[7:] if text.startswith("sha256:") else text


def _safe_file(path: Path, *, name: str, max_bytes: int, blockers: list[str]) -> bool:
    if path.is_symlink():
        blockers.append(f"input_symlink_forbidden:{name}")
        return False
    if not path.is_file():
        blockers.append(f"input_file_missing:{name}")
        return False
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        blockers.append(f"input_file_size_invalid:{name}")
        return False
    return True


def _load_json(
    path: Path, *, name: str, max_bytes: int, blockers: list[str]
) -> Any:
    if not _safe_file(path, name=name, max_bytes=max_bytes, blockers=blockers):
        return None
    try:
        return read_json_any(path)
    except (OSError, UnicodeError, ValueError):
        blockers.append(f"input_json_unreadable:{name}")
        return None


def _verify_artifact(
    *,
    name: str,
    path: Path,
    request: Mapping[str, Any],
    blockers: list[str],
) -> Dict[str, Any] | None:
    artifacts = request.get("input_artifacts")
    reference = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    if not isinstance(reference, Mapping):
        blockers.append(f"input_artifact_reference_missing:{name}")
        return None
    expected_sha = reference.get("sha256")
    expected_size = reference.get("size_bytes")
    if not _valid_sha256(expected_sha):
        blockers.append(f"input_artifact_sha256_invalid:{name}")
        return None
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0:
        blockers.append(f"input_artifact_size_invalid:{name}")
        return None
    actual_size = path.stat().st_size
    actual_sha = sha256_file(path)
    if expected_size != actual_size:
        blockers.append(f"input_artifact_size_mismatch:{name}")
    if _normalized_sha256(expected_sha) != actual_sha:
        blockers.append(f"input_artifact_sha256_mismatch:{name}")
    return {
        "filename": path.name,
        "sha256": "sha256:" + actual_sha,
        "size_bytes": actual_size,
    }


def run_semantic_contribution_renderer_stage(
    *,
    request_path: str | Path,
    analysis_splat_path: str | Path,
    gaussian_mapping_path: str | Path,
    source_tracks_path: str | Path,
    camera_records_path: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    """Verify immutable inputs, render contributions, and atomically write a receipt."""

    paths = {
        "request": Path(request_path),
        "analysis_splat": Path(analysis_splat_path),
        "gaussian_mapping": Path(gaussian_mapping_path),
        "source_tracks": Path(source_tracks_path),
        "camera_records": Path(camera_records_path),
    }
    output = Path(output_path)
    blockers: list[str] = []
    try:
        output_resolved = output.resolve(strict=False)
        input_resolved = {path.resolve(strict=False) for path in paths.values()}
    except OSError:
        blockers.append("input_or_output_path_unresolvable")
    else:
        if output_resolved in input_resolved:
            raise ValueError("output_path_must_not_overwrite_an_input")
    if output.is_symlink():
        raise ValueError("output_symlink_forbidden")

    request_payload = _load_json(
        paths["request"], name="request", max_bytes=_REQUEST_MAX_BYTES, blockers=blockers
    )
    request = dict(request_payload) if isinstance(request_payload, Mapping) else {}
    if request_payload is not None and not isinstance(request_payload, Mapping):
        blockers.append("request_json_must_be_object")

    payloads: Dict[str, Any] = {}
    for name in ("gaussian_mapping", "source_tracks", "camera_records"):
        payload = _load_json(
            paths[name], name=name, max_bytes=_INPUT_LIMITS[name], blockers=blockers
        )
        payloads[name] = payload
    if payloads.get("gaussian_mapping") is not None and not isinstance(
        payloads["gaussian_mapping"], list
    ):
        blockers.append("gaussian_mapping_json_must_be_array")
    if payloads.get("source_tracks") is not None and not isinstance(
        payloads["source_tracks"], Mapping
    ):
        blockers.append("source_tracks_json_must_be_object")
    if payloads.get("camera_records") is not None and not isinstance(
        payloads["camera_records"], list
    ):
        blockers.append("camera_records_json_must_be_array")
    _safe_file(
        paths["analysis_splat"],
        name="analysis_splat",
        max_bytes=_INPUT_LIMITS["analysis_splat"],
        blockers=blockers,
    )

    verified: Dict[str, Any] = {}
    for name in ("analysis_splat", "gaussian_mapping", "source_tracks", "camera_records"):
        path = paths[name]
        if path.is_file() and not path.is_symlink():
            record = _verify_artifact(
                name=name, path=path, request=request, blockers=blockers
            )
            if record is not None:
                verified[name] = record
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    analysis_record = verified.get("analysis_splat")
    if analysis_record and _normalized_sha256(bindings.get("analysis_splat_digest")) != (
        _normalized_sha256(analysis_record["sha256"])
    ):
        blockers.append("analysis_splat_binding_digest_mismatch")

    splat = None
    if not blockers:
        try:
            splat = read_standard_3dgs_ply(paths["analysis_splat"])
        except (OSError, ValueError):
            blockers.append("analysis_splat_not_standard_3dgs_ply")
    if blockers or splat is None:
        result = blocked_semantic_contribution_render(request, blockers)
    else:
        result = render_semantic_contributions(
            request,
            splat=splat,
            gaussian_mapping=payloads["gaussian_mapping"],
            source_tracks=payloads["source_tracks"],
            camera_records=payloads["camera_records"],
        )
        result.pop("result_digest", None)
        result["stage_input_artifacts"] = {
            "request": {
                "filename": paths["request"].name,
                "sha256": "sha256:" + sha256_file(paths["request"]),
                "size_bytes": paths["request"].stat().st_size,
            },
            **verified,
        }
        result["transport_profile"] = "bounded_canonical_json_reference.v1"
        result["result_digest"] = canonical_json_digest(result)
    write_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render exact standard-3DGS contribution rows for semantic lifting."
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--analysis-splat", required=True)
    parser.add_argument("--gaussian-mapping", required=True)
    parser.add_argument("--source-tracks", required=True)
    parser.add_argument("--camera-records", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_semantic_contribution_renderer_stage(
        request_path=args.request,
        analysis_splat_path=args.analysis_splat,
        gaussian_mapping_path=args.gaussian_mapping,
        source_tracks_path=args.source_tracks,
        camera_records_path=args.camera_records,
        output_path=args.output,
    )
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["run_semantic_contribution_renderer_stage"]
