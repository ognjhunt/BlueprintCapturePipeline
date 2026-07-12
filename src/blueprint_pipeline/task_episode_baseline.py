"""Episode-relative task-success baseline capture, binding, and evaluation.

Relative task criteria must be judged against one immutable episode baseline
captured after stage settle and before action zero, never against the
immediately preceding step. Per-step deltas remain diagnostic only. The
baseline is bound to attempt, launch nonce, simulator session, stage
fingerprint, target prim, and task-contract hash; any restart, prim change,
or tampering blocks fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


TASK_EPISODE_BASELINE_SCHEMA_VERSION = "task_episode_baseline.v1"

RELATIVE_CHANGE_COMPARISONS = frozenset(
    {"increase_at_least", "decrease_at_least", "absolute_change_at_least"}
)
ABSOLUTE_TARGET_COMPARISONS = frozenset({"within_tolerance", "at_or_above", "at_or_below"})

_BINDING_STRING_FIELDS = (
    "attempt_id",
    "launch_nonce",
    "simulator_session_id",
    "stage_id",
    "articulation_prim_path",
    "task_contract_sha256",
    "criterion_id",
    "unit",
    "captured_timestamp",
)


def canonical_task_contract_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(contract or {}), sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _baseline_digest(baseline: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": baseline.get("schema_version"),
        "episode_initial_value": baseline.get("episode_initial_value"),
        **{field: baseline.get(field) for field in _BINDING_STRING_FIELDS},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def build_task_episode_baseline(
    *,
    episode_initial_value: float,
    attempt_id: str,
    launch_nonce: str,
    simulator_session_id: str,
    stage_id: str,
    articulation_prim_path: str,
    task_contract_sha256: str,
    criterion_id: str,
    unit: str,
    captured_timestamp: str,
) -> dict[str, Any]:
    fields = {
        "attempt_id": attempt_id,
        "launch_nonce": launch_nonce,
        "simulator_session_id": simulator_session_id,
        "stage_id": stage_id,
        "articulation_prim_path": articulation_prim_path,
        "task_contract_sha256": task_contract_sha256,
        "criterion_id": criterion_id,
        "unit": unit,
        "captured_timestamp": captured_timestamp,
    }
    for field, value in fields.items():
        if not str(value or "").strip():
            raise ValueError(f"task_episode_baseline_field_missing:{field}")
    try:
        initial = float(episode_initial_value)
    except (TypeError, ValueError):
        initial = math.nan
    if not math.isfinite(initial):
        raise ValueError("task_episode_baseline_field_missing:episode_initial_value")
    baseline = {
        "schema_version": TASK_EPISODE_BASELINE_SCHEMA_VERSION,
        "episode_initial_value": initial,
        **{field: str(value) for field, value in fields.items()},
    }
    baseline["baseline_digest"] = _baseline_digest(baseline)
    return baseline


def verify_task_episode_baseline(
    baseline: Any,
    *,
    simulator_session_id: str,
    stage_id: str,
    articulation_prim_path: str,
    task_contract_sha256: str,
    attempt_id: str | None = None,
    launch_nonce: str | None = None,
) -> list[str]:
    if not isinstance(baseline, Mapping):
        return ["task_episode_baseline_missing"]
    blockers: list[str] = []
    if baseline.get("schema_version") != TASK_EPISODE_BASELINE_SCHEMA_VERSION:
        blockers.append("task_episode_baseline_schema_invalid")
    if str(baseline.get("baseline_digest") or "") != _baseline_digest(baseline):
        blockers.append("task_episode_baseline_digest_mismatch")
    expected = {
        "simulator_session_id": (simulator_session_id, "task_episode_baseline_session_mismatch"),
        "stage_id": (stage_id, "task_episode_baseline_stage_mismatch"),
        "articulation_prim_path": (
            articulation_prim_path,
            "task_episode_baseline_prim_mismatch",
        ),
        "task_contract_sha256": (
            task_contract_sha256,
            "task_episode_baseline_task_contract_mismatch",
        ),
    }
    if attempt_id is not None:
        expected["attempt_id"] = (attempt_id, "task_episode_baseline_attempt_mismatch")
    if launch_nonce is not None:
        expected["launch_nonce"] = (launch_nonce, "task_episode_baseline_nonce_mismatch")
    for field, (value, blocker) in expected.items():
        if str(baseline.get(field) or "") != str(value or "") or not str(value or "").strip():
            blockers.append(blocker)
    initial = baseline.get("episode_initial_value")
    if (
        not isinstance(initial, (int, float))
        or isinstance(initial, bool)
        or not math.isfinite(float(initial))
    ):
        blockers.append("task_episode_baseline_initial_value_nonfinite")
    return sorted(set(blockers))


def evaluate_task_criterion(
    criterion: Mapping[str, Any],
    *,
    episode_initial_value: float,
    step_before: float,
    step_after: float,
) -> dict[str, Any]:
    """Evaluate one registered criterion against the immutable episode baseline.

    Relative change comparisons use current-vs-episode-initial truth; the
    per-step delta is emitted for diagnostics only. Absolute target
    comparisons stay separate and judge the current value against the
    registered target.
    """

    values = {
        "episode_initial_value": episode_initial_value,
        "step_before": step_before,
        "step_after": step_after,
    }
    for field, value in values.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = math.nan
        if not math.isfinite(numeric):
            raise ValueError(f"task_episode_measurement_value_nonfinite:{field}")
        values[field] = numeric
    initial = values["episode_initial_value"]
    before = values["step_before"]
    after = values["step_after"]
    comparison = str(criterion.get("comparison") or "")
    tolerance = float(criterion.get("tolerance") or 0.0)
    step_delta = after - before
    episode_delta = after - initial
    if comparison in RELATIVE_CHANGE_COMPARISONS:
        basis = "episode_relative"
        if comparison == "increase_at_least":
            passed = episode_delta >= tolerance
        elif comparison == "decrease_at_least":
            passed = -episode_delta >= tolerance
        else:
            passed = abs(episode_delta) >= tolerance
    elif comparison in ABSOLUTE_TARGET_COMPARISONS:
        basis = "absolute_target"
        target = float(criterion.get("target_value") or 0.0)
        if comparison == "within_tolerance":
            passed = abs(after - target) <= tolerance
        elif comparison == "at_or_above":
            passed = after >= target - tolerance
        else:
            passed = after <= target + tolerance
    else:
        raise ValueError("persistent_isaac_completion_comparison_unsupported")
    return {
        "passed": bool(passed),
        "comparison": comparison,
        "tolerance": tolerance,
        "evaluation_basis": basis,
        "episode_initial_value": initial,
        "step_before": before,
        "step_after": after,
        "step_delta": step_delta,
        "episode_delta": episode_delta,
    }
