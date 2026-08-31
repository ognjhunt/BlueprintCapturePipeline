"""Provider-runtime verification for no-motion terminal-feedback bootstrap."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .native_construction_terminal_feedback_contract import (
    validate_terminal_feedback_adoption,
)


RELATIVE_PATH = (
    "runtime_inputs/native_construction_terminal_feedback_adoption.v1.json"
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def verified_terminal_feedback_adoption_path(
    runtime: Path, manifest: Mapping[str, Any]
) -> Path | None:
    rows = [
        row
        for row in manifest.get("bound_runtime_inputs") or []
        if isinstance(row, Mapping) and row.get("relative_path") == RELATIVE_PATH
    ]
    if not rows:
        return None
    path = runtime / RELATIVE_PATH
    if (
        len(rows) != 1
        or not path.is_file()
        or path.stat().st_size != rows[0].get("size_bytes")
        or _sha256(path) != rows[0].get("sha256")
    ):
        raise RuntimeError("native_task_terminal_feedback_adoption_invalid")
    return path


def verified_construction_phase_plan_path(
    runtime: Path, manifest: Mapping[str, Any]
) -> Path:
    relative = "runtime_inputs/native_task_construction_phase_plan.v1.json"
    rows = [
        row
        for row in manifest.get("bound_runtime_inputs") or []
        if isinstance(row, Mapping) and row.get("relative_path") == relative
    ]
    path = runtime / relative
    if (
        len(rows) != 1
        or not path.is_file()
        or path.stat().st_size != rows[0].get("size_bytes")
        or _sha256(path) != rows[0].get("sha256")
    ):
        raise RuntimeError("native_task_construction_phase_plan_identity_mismatch")
    return path


def feedback_bootstrap_result(
    *, runtime: Path, manifest: Mapping[str, Any], packet: Path
) -> dict[str, Any] | None:
    path = verified_terminal_feedback_adoption_path(runtime, manifest)
    if path is None:
        return None
    adoption = validate_terminal_feedback_adoption(
        json.loads(path.read_text(encoding="utf-8"))
    )
    request = json.loads(
        (packet / "native_task_arena_packet_request.v1.json").read_text(
            encoding="utf-8"
        )
    )
    if adoption["packet_request_digest"] != request.get("request_digest"):
        raise RuntimeError("native_task_terminal_feedback_adoption_binding_mismatch")
    return {
        "status": "blocked",
        "phase_reached": "feedback_bootstrap_ready",
        "construction_gate_qualified": False,
        "feedback_bootstrap_only": True,
        "baseline_physics_replayed": False,
        "terminal_feedback_adoption_digest": adoption["checkpoint_digest"],
        "blockers": ["native_construction_feedback_bootstrap_ready"],
    }


__all__ = [
    "feedback_bootstrap_result",
    "verified_construction_phase_plan_path",
    "verified_terminal_feedback_adoption_path",
]
