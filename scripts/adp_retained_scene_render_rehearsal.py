#!/usr/bin/env python3
"""Validate a retained-scene provider bundle without requiring host Node.js."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


class RehearsalError(ValueError):
    """The immutable runtime packet failed its zero-cost validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _checked_file(root: Path, record: Any) -> Path:
    if not isinstance(record, dict):
        raise RehearsalError("retained_scene_render_runtime_file_binding_invalid")
    relative = record.get("relative_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RehearsalError("retained_scene_render_runtime_file_binding_invalid")
    candidate = root / relative
    if candidate.is_symlink():
        raise RehearsalError("retained_scene_render_runtime_file_binding_invalid")
    target = candidate.resolve()
    if root != target and root not in target.parents:
        raise RehearsalError("retained_scene_render_runtime_file_binding_invalid")
    if (
        not target.is_file()
        or target.stat().st_size != record.get("size_bytes")
        or _sha256(target) != record.get("sha256")
    ):
        raise RehearsalError("retained_scene_render_runtime_file_binding_invalid")
    return target


def _standard_ply_count(path: Path) -> int:
    with path.open("rb") as stream:
        header = stream.read(1024 * 1024)
    match = re.search(rb"^element vertex ([0-9]+)\r?$", header, flags=re.MULTILINE)
    if match is None:
        raise RehearsalError("retained_scene_render_runtime_ply_count_invalid")
    return int(match.group(1))


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RehearsalError("retained_scene_render_runtime_request_invalid") from exc
    if not isinstance(value, dict):
        raise RehearsalError("retained_scene_render_runtime_request_invalid")
    return value


def rehearse(*, runtime: Path, output: Path) -> dict[str, Any]:
    runtime = runtime.resolve()
    output = output.resolve()
    if not runtime.is_dir() or runtime.is_symlink():
        raise RehearsalError("retained_scene_render_runtime_path_invalid")
    request = _json_object(runtime / "render_request.json")
    deleted = _checked_file(runtime, request.get("shared_deleted_source_layer"))
    retained = _checked_file(runtime, request.get("shared_retained_scene"))
    _checked_file(runtime, request.get("candidate_set"))
    _checked_file(runtime, request.get("execution_authority"))
    deleted_record = request.get("shared_deleted_source_layer")
    if not isinstance(deleted_record, dict):
        raise RehearsalError("retained_scene_render_runtime_ply_count_invalid")
    if _standard_ply_count(deleted) != deleted_record.get("gaussian_count") or _standard_ply_count(
        retained
    ) != request.get("shared_retained_gaussian_count"):
        raise RehearsalError("retained_scene_render_runtime_ply_count_invalid")

    lanes = request.get("lanes")
    if not isinstance(lanes, list):
        raise RehearsalError("retained_scene_render_runtime_lane_contract_invalid")
    for lane in lanes:
        if not isinstance(lane, dict):
            raise RehearsalError("retained_scene_render_runtime_lane_contract_invalid")
        _checked_file(runtime, lane.get("camera_contract"))
        _checked_file(runtime, lane.get("task_freeze"))
        for key in ("task_deleted_source_layer", "task_retained_scene"):
            record = lane.get(key)
            if record is None:
                continue
            path = _checked_file(runtime, record)
            if not isinstance(record, dict) or _standard_ply_count(path) != record.get(
                "gaussian_count"
            ):
                raise RehearsalError("retained_scene_render_runtime_ply_count_invalid")

    receipt = {
        "schema_version": "provider_bundle_rehearsal.v1",
        "status": "passed",
        "released_renderer_executed": False,
        "gpu_runtime_started": False,
        "paid_inference_performed": False,
        "provider_mutations_performed": 0,
        "verified_task_lanes": len(lanes),
        "blockers": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "provider_bundle_rehearsal.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        rehearse(runtime=args.runtime, output=args.output)
    except RehearsalError as exc:
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
