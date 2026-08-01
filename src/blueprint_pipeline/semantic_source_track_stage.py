"""Bounded file entrypoint for persistent semantic source-track imports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import read_json_any, sha256_file, write_json
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest
from .scene_placement.semantic_source_track_import import (
    blocked_semantic_source_track_import,
    import_semantic_source_tracks,
)


_REQUEST_MAX_BYTES = 2 * 1024 * 1024
_PROVIDER_RESULT_MAX_BYTES = 512 * 1024 * 1024


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _normalized_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text[7:] if text.startswith("sha256:") else text


def _load_object(
    path: Path, *, name: str, max_bytes: int, blockers: list[str]
) -> Dict[str, Any]:
    if path.is_symlink():
        blockers.append(f"input_symlink_forbidden:{name}")
        return {}
    if not path.is_file():
        blockers.append(f"input_file_missing:{name}")
        return {}
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        blockers.append(f"input_file_size_invalid:{name}")
        return {}
    try:
        payload = read_json_any(path)
    except (OSError, UnicodeError, ValueError):
        blockers.append(f"input_json_unreadable:{name}")
        return {}
    if not isinstance(payload, Mapping):
        blockers.append(f"input_json_not_object:{name}")
        return {}
    return dict(payload)


def run_semantic_source_track_stage(
    *, request_path: str | Path, provider_result_path: str | Path, output_path: str | Path
) -> Dict[str, Any]:
    """Verify exact provider bytes and write one terminal normalized artifact."""

    request_file = Path(request_path)
    provider_file = Path(provider_result_path)
    output_file = Path(output_path)
    try:
        output_resolved = output_file.resolve(strict=False)
        inputs = {
            request_file.resolve(strict=False),
            provider_file.resolve(strict=False),
        }
    except OSError as exc:
        raise ValueError("input_or_output_path_unresolvable") from exc
    if output_resolved in inputs:
        raise ValueError("output_path_must_not_overwrite_an_input")
    if output_file.is_symlink():
        raise ValueError("output_symlink_forbidden")

    blockers: list[str] = []
    request = _load_object(
        request_file,
        name="request",
        max_bytes=_REQUEST_MAX_BYTES,
        blockers=blockers,
    )
    provider_result = _load_object(
        provider_file,
        name="provider_result",
        max_bytes=_PROVIDER_RESULT_MAX_BYTES,
        blockers=blockers,
    )
    expected = (
        request.get("input_artifacts", {}).get("provider_result")
        if isinstance(request.get("input_artifacts"), Mapping)
        else None
    )
    if provider_file.is_file() and not provider_file.is_symlink():
        if not isinstance(expected, Mapping):
            blockers.append("input_artifact_reference_missing:provider_result")
        else:
            supplied_size = expected.get("size_bytes")
            supplied_sha = expected.get("sha256")
            if (
                not isinstance(supplied_size, int)
                or isinstance(supplied_size, bool)
                or supplied_size != provider_file.stat().st_size
            ):
                blockers.append("input_artifact_size_mismatch:provider_result")
            if not _valid_sha256(supplied_sha) or _normalized_sha256(
                supplied_sha
            ) != sha256_file(provider_file):
                blockers.append("input_artifact_sha256_mismatch:provider_result")
    if blockers:
        result = blocked_semantic_source_track_import(request, blockers)
    else:
        result = import_semantic_source_tracks(request, provider_result)
    result.pop("result_digest", None)
    stage_inputs: dict[str, Any] = {}
    for name, path in (("request", request_file), ("provider_result", provider_file)):
        if path.is_file() and not path.is_symlink():
            stage_inputs[name] = {
                "filename": path.name,
                "sha256": "sha256:" + sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    result["stage_input_artifacts"] = stage_inputs
    result["transport_profile"] = "bounded_compact_probability_rle.v1"
    result["result_digest"] = canonical_json_digest(result)
    write_json(output_file, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize source-bound persistent semantic tracks."
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--provider-result", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_semantic_source_track_stage(
        request_path=args.request,
        provider_result_path=args.provider_result,
        output_path=args.output,
    )
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["run_semantic_source_track_stage"]
