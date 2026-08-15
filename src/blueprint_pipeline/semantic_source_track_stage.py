"""Bounded file entrypoint for persistent semantic source-track imports."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import read_json_any, sha256_file, write_json
from .decision_evidence_contracts import canonical_digest
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest
from .scene_placement.semantic_source_track_import import (
    blocked_semantic_source_track_import,
    import_semantic_source_tracks,
)


_REQUEST_MAX_BYTES = 2 * 1024 * 1024
_PROVIDER_RESULT_MAX_BYTES = 512 * 1024 * 1024
_TERMINAL_RUNTIME_SCHEMA_VERSION = "semantic_sam31_vast_source_track_result.v1"
_TERMINAL_REIMPORT_RECEIPT_SCHEMA_VERSION = (
    "semantic_source_track_terminal_reimport_receipt.v1"
)
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


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


def run_semantic_source_track_terminal_reimport(
    *,
    terminal_runtime_result_path: str | Path,
    source_commit_sha: str,
    output_path: str | Path,
    receipt_output_path: str | Path,
) -> Dict[str, Any]:
    """Reopen one retained terminal worker result under the current importer."""

    runtime_file = Path(terminal_runtime_result_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    receipt_file = Path(receipt_output_path).expanduser().resolve()
    commit = str(source_commit_sha or "").strip().lower()
    if not _COMMIT_SHA.fullmatch(commit):
        raise ValueError("terminal_reimport_source_commit_invalid")
    if output_file == receipt_file or runtime_file in {output_file, receipt_file}:
        raise ValueError("terminal_reimport_output_must_not_overwrite_input")
    for destination in (output_file, receipt_file):
        if destination.is_symlink() or destination.exists():
            raise ValueError("terminal_reimport_output_not_exclusive")

    blockers: list[str] = []
    runtime = _load_object(
        runtime_file,
        name="terminal_runtime_result",
        max_bytes=_PROVIDER_RESULT_MAX_BYTES,
        blockers=blockers,
    )
    request = runtime.get("source_track_import_request")
    provider_result = runtime.get("provider_result")
    if (
        blockers
        or runtime.get("schema_version") != _TERMINAL_RUNTIME_SCHEMA_VERSION
        or runtime.get("status") != "passed"
        or runtime.get("blockers") != []
        or runtime.get("raw_secret_values_recorded") is not False
        or runtime.get("runtime_result_digest")
        != canonical_digest(runtime, digest_field="runtime_result_digest")
        or not isinstance(request, Mapping)
        or not isinstance(provider_result, Mapping)
    ):
        raise ValueError("terminal_reimport_runtime_result_invalid")

    result = import_semantic_source_tracks(request, provider_result)
    if result.get("status") not in {"completed", "abstained"} or result.get("blockers") != []:
        raise ValueError("terminal_reimport_normalized_result_invalid")
    original = runtime.get("normalized_source_tracks")
    original_digest = (
        str(original.get("result_digest") or "") if isinstance(original, Mapping) else ""
    )
    result.pop("result_digest", None)
    result["terminal_reimport"] = {
        "source_commit_sha": commit,
        "terminal_runtime_result": {
            "path": str(runtime_file),
            "sha256": "sha256:" + sha256_file(runtime_file),
            "size_bytes": runtime_file.stat().st_size,
            "runtime_result_digest": runtime["runtime_result_digest"],
        },
        "original_normalized_result_digest": original_digest,
        "provider_result_digest": provider_result.get("result_digest"),
        "provider_compute_mutation_performed": False,
        "paid_resource_allocated": False,
    }
    result["result_digest"] = canonical_json_digest(result)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    receipt_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_file, result)
    receipt: Dict[str, Any] = {
        "schema_version": _TERMINAL_REIMPORT_RECEIPT_SCHEMA_VERSION,
        "status": "ready",
        "source_commit_sha": commit,
        "terminal_runtime_result": result["terminal_reimport"]["terminal_runtime_result"],
        "normalized_result": {
            "path": str(output_file),
            "sha256": "sha256:" + sha256_file(output_file),
            "size_bytes": output_file.stat().st_size,
            "result_digest": result["result_digest"],
            "frame_count": len(result.get("frame_masks") or []),
            "track_count": len(result.get("track_registry") or []),
        },
        "provider_compute_mutation_performed": False,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(receipt_file, receipt)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize source-bound persistent semantic tracks."
    )
    parser.add_argument("--request")
    parser.add_argument("--provider-result")
    parser.add_argument(
        "--terminal-runtime-result",
        help="Retained terminal semantic_sam31_vast_source_track_result.v1 to re-import.",
    )
    parser.add_argument("--source-commit-sha")
    parser.add_argument("--receipt-output")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.terminal_runtime_result:
        if args.request or args.provider_result or not args.source_commit_sha or not args.receipt_output:
            parser.error(
                "terminal re-import requires --source-commit-sha and --receipt-output, "
                "and forbids --request/--provider-result"
            )
        result = run_semantic_source_track_terminal_reimport(
            terminal_runtime_result_path=args.terminal_runtime_result,
            source_commit_sha=args.source_commit_sha,
            output_path=args.output,
            receipt_output_path=args.receipt_output,
        )
        return 2 if result["status"] == "blocked" else 0
    if not args.request or not args.provider_result or args.source_commit_sha or args.receipt_output:
        parser.error(
            "file import requires --request and --provider-result and forbids terminal re-import arguments"
        )
    result = run_semantic_source_track_stage(
        request_path=args.request,
        provider_result_path=args.provider_result,
        output_path=args.output,
    )
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "run_semantic_source_track_stage",
    "run_semantic_source_track_terminal_reimport",
]
