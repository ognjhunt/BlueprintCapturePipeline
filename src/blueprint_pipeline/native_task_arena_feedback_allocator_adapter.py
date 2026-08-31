"""Narrow paid-allocator adapter for retained native construction feedback."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .task_evaluation_robot_placement_warm_executor import (
    run_retained_native_construction_feedback,
)
from .native_construction_terminal_feedback_contract import (
    validate_terminal_feedback_adoption,
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


def terminal_feedback_bootstrap_blockers(
    *,
    packet_dir: str | Path | None,
    prepared_bundle: Mapping[str, Any] | None,
    adoption_path: str | Path | None,
) -> list[str]:
    if adoption_path is None:
        return []
    try:
        path = Path(adoption_path).expanduser().resolve()
        adoption = validate_terminal_feedback_adoption(
            _mapping(
                path,
                blocker="native_construction_terminal_feedback_adoption_invalid",
            )
        )
        request = _mapping(
            Path(str(packet_dir)) / "native_task_arena_packet_request.v1.json",
            blocker="native_construction_feedback_packet_request_invalid",
        )
        rows = [
            row
            for row in (prepared_bundle or {}).get("bound_runtime_inputs") or []
            if isinstance(row, Mapping)
            and Path(str(row.get("relative_path") or "")).name
            == "native_construction_terminal_feedback_adoption.v1.json"
        ]
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            len(rows) != 1
            or rows[0].get("sha256") != digest
            or adoption.get("packet_request_digest") != request.get("request_digest")
        ):
            raise ValueError("binding_mismatch")
    except (OSError, ValueError, TypeError):
        return ["native_construction_terminal_feedback_bootstrap_invalid"]
    return []
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
    terminal_feedback_adoption_path: str | Path | None = None,
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
    if terminal_feedback_adoption_path is not None:
        adoption_path = Path(terminal_feedback_adoption_path).expanduser().resolve()
        adoption = validate_terminal_feedback_adoption(
            _mapping(
                adoption_path,
                blocker="native_construction_terminal_feedback_adoption_invalid",
            )
        )
        expected = [
            row
            for row in prepared_bundle.get("bound_runtime_inputs") or []
            if isinstance(row, Mapping)
            and Path(str(row.get("relative_path") or "")).name
            == "native_construction_terminal_feedback_adoption.v1.json"
        ]
        digest = "sha256:" + hashlib.sha256(adoption_path.read_bytes()).hexdigest()
        if (
            len(expected) != 1
            or expected[0].get("sha256") != digest
            or adoption.get("packet_request_digest") != request.get("request_digest")
        ):
            raise ValueError("native_construction_terminal_feedback_bundle_mismatch")
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
        terminal_feedback_adoption_path=terminal_feedback_adoption_path,
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
    "terminal_feedback_bootstrap_blockers",
]
