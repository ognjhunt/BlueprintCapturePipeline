"""Narrow paid-allocator adapter for retained native construction feedback."""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .task_evaluation_robot_placement_warm_executor import (
    run_retained_native_construction_feedback,
)


def _mapping(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(blocker) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise ValueError(blocker)
    return dict(value)


def native_feedback_runtime_blockers(packet_dir: str | Path | None) -> list[str]:
    if not packet_dir:
        return []
    request_path = Path(packet_dir) / "native_task_arena_packet_request.v1.json"
    if not request_path.is_file():
        return []
    request = _mapping(
        request_path, blocker="native_construction_feedback_packet_request_invalid"
    )
    if not isinstance(request.get("native_construction_feedback"), Mapping):
        return []
    try:
        version = importlib.metadata.version("optuna")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return (
        []
        if version == "4.9.0"
        else ["native_construction_feedback_optuna_4_9_0_missing"]
    )


def continue_retained_feedback_if_requested(
    *,
    execute: bool,
    construction_requested: bool,
    retain_warm_session: bool,
    result: Mapping[str, Any],
    packet_dir: str | Path,
    runtime_source_packet_receipt_path: str | Path,
    prepared_bundle: Mapping[str, Any],
    native_authority: Mapping[str, Any],
    job_dir: str | Path,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
) -> dict[str, Any]:
    value = dict(result)
    if not (
        execute
        and construction_requested
        and retain_warm_session
        and isinstance(value.get("warm_session"), Mapping)
        and value.get("native_control_result_path")
    ):
        return value
    packet = Path(packet_dir)
    request = _mapping(
        packet / "native_task_arena_packet_request.v1.json",
        blocker="native_construction_feedback_packet_request_invalid",
    )
    feedback = request.get("native_construction_feedback")
    if not isinstance(feedback, Mapping):
        return value
    controller = run_retained_native_construction_feedback(
        cold_allocator_result=value,
        packet_dir=packet,
        runtime_source_packet_receipt_path=runtime_source_packet_receipt_path,
        implementation_commit=str(prepared_bundle["implementation_commit"]),
        output_root=Path(job_dir) / "native-construction-feedback",
        authorization_reference=str(native_authority.get("authority_reference") or ""),
        authorized_by=str(native_authority.get("authorized_by") or ""),
        authorized_on=str(native_authority.get("authorized_on") or ""),
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        maximum_rounds=min(int(feedback.get("maximum_rounds") or 4), 8),
    )
    value["native_construction_feedback_controller"] = controller
    if controller.get("status") != "controls_completed":
        return value
    final_native = controller["history"][-1]["execution"]["native_result"]
    qualified_path = (
        Path(job_dir)
        / "native-construction-feedback"
        / "qualified-native-construction-result.v1.json"
    )
    write_json(qualified_path, final_native)
    continuation = controller["controls_continuation"]
    value.update(
        {
            "status": "completed",
            "blockers": [],
            "native_control_result_path": str(qualified_path),
            "native_control_result_digest": final_native["result_digest"],
            "native_controls_result_path": continuation[
                "native_control_result_path"
            ],
            "native_controls_result_digest": continuation[
                "native_control_result_digest"
            ],
            "continuing_spend_from_this_run": False,
            "retry_cap": 0,
        }
    )
    return value


__all__ = [
    "continue_retained_feedback_if_requested",
    "native_feedback_runtime_blockers",
]
