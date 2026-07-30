"""File-bound entrypoint for deterministic semantic oriented-box fitting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import read_json_any, sha256_file, write_json
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest
from .scene_placement.semantic_oriented_box import (
    FIT_METHOD,
    RESULT_SCHEMA_VERSION,
    fit_semantic_oriented_boxes,
)


_INPUT_LIMITS = {
    "semantic_result": 256 * 1024 * 1024,
    "gaussian_mapping": 256 * 1024 * 1024,
    "support_points": 512 * 1024 * 1024,
}
_REQUEST_MAX_BYTES = 2 * 1024 * 1024


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _normalized_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text[7:] if text.startswith("sha256:") else text


def _blocked(request: Mapping[str, Any], blockers: Sequence[str]) -> Dict[str, Any]:
    bindings = request.get("bindings")
    result: Dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "bindings": dict(bindings) if isinstance(bindings, Mapping) else {},
        "fit_method": FIT_METHOD,
        "objects": [],
        "blockers": sorted(set(blockers)),
        "claim_ceiling": "none_invalid_or_unbound_input_artifacts",
        "metric_obb_candidate_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "prohibited_claims": [
            "collision_or_contact_truth",
            "task_or_physical_success",
        ],
    }
    result["result_digest"] = canonical_json_digest(result)
    return result


def _load_json(path: Path, *, name: str, limit: int, blockers: list[str]) -> Any:
    if path.is_symlink():
        blockers.append(f"input_symlink_forbidden:{name}")
        return None
    if not path.is_file():
        blockers.append(f"input_file_missing:{name}")
        return None
    size = path.stat().st_size
    if size <= 0 or size > limit:
        blockers.append(f"input_file_size_invalid:{name}")
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
    if actual_size != expected_size:
        blockers.append(f"input_artifact_size_mismatch:{name}")
    if _normalized_sha256(expected_sha) != actual_sha:
        blockers.append(f"input_artifact_sha256_mismatch:{name}")
    return {
        "filename": path.name,
        "sha256": "sha256:" + actual_sha,
        "size_bytes": actual_size,
    }


def run_semantic_oriented_box_stage(
    *,
    request_path: str | Path,
    semantic_result_path: str | Path,
    gaussian_mapping_path: str | Path,
    support_points_path: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    """Verify exact files, fit qualified OBB candidates, and write a terminal result."""

    paths = {
        "request": Path(request_path),
        "semantic_result": Path(semantic_result_path),
        "gaussian_mapping": Path(gaussian_mapping_path),
        "support_points": Path(support_points_path),
    }
    output = Path(output_path)
    try:
        output_resolved = output.resolve(strict=False)
        input_resolved = {path.resolve(strict=False) for path in paths.values()}
    except OSError as error:
        raise ValueError("input_or_output_path_unresolvable") from error
    if output_resolved in input_resolved:
        raise ValueError("output_path_must_not_overwrite_an_input")
    if output.is_symlink():
        raise ValueError("output_symlink_forbidden")

    blockers: list[str] = []
    request_payload = _load_json(
        paths["request"], name="request", limit=_REQUEST_MAX_BYTES, blockers=blockers
    )
    request = dict(request_payload) if isinstance(request_payload, Mapping) else {}
    if request_payload is not None and not isinstance(request_payload, Mapping):
        blockers.append("request_json_must_be_object")

    payloads: Dict[str, Any] = {}
    verified: Dict[str, Any] = {}
    for name in ("semantic_result", "gaussian_mapping", "support_points"):
        payload = _load_json(
            paths[name], name=name, limit=_INPUT_LIMITS[name], blockers=blockers
        )
        expected_object = name == "semantic_result"
        if payload is not None and expected_object != isinstance(payload, Mapping):
            blockers.append(
                f"input_json_must_be_{'object' if expected_object else 'array'}:{name}"
            )
        payloads[name] = payload
        if payload is not None and paths[name].is_file() and not paths[name].is_symlink():
            verified_row = _verify_artifact(
                name=name, path=paths[name], request=request, blockers=blockers
            )
            if verified_row is not None:
                verified[name] = verified_row

    if blockers:
        result = _blocked(request, blockers)
    else:
        result = fit_semantic_oriented_boxes(
            request,
            semantic_result=payloads["semantic_result"],
            gaussian_mapping=payloads["gaussian_mapping"],
            support_points=payloads["support_points"],
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
        result["transport_profile"] = "bounded_canonical_json_baseline.v1"
        result["result_digest"] = canonical_json_digest(result)
    write_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit metric Z-up OBB candidates from qualified semantic support."
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--semantic-result", required=True)
    parser.add_argument("--gaussian-mapping", required=True)
    parser.add_argument("--support-points", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_semantic_oriented_box_stage(
        request_path=args.request,
        semantic_result_path=args.semantic_result,
        gaussian_mapping_path=args.gaussian_mapping,
        support_points_path=args.support_points,
        output_path=args.output,
    )
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
