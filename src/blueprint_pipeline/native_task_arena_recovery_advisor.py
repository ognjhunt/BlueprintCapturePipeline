"""Let an agent choose the next recovery attempt, without letting it grade.

Thirty-two controls runs spent one hypothesis each, and the loop that finally
found the cause was an agent reading sealed telemetry between runs.  Doing that
*inside* a run is worth minutes rather than a cold start per hypothesis -- but
the deterministic scripted positive control is also the baseline both frozen
policy candidates are scored against, so it cannot make model calls mid-episode
and stay replayable.

Both are satisfiable at once, and this module is the seam.  The agent advises
*before* the run: it reads the previous episode's sealed attempt telemetry and
returns a recovery ladder, which the plan then carries and the episode executes
deterministically.  What the agent may express is exactly a ranking over rungs
the executor already implements, so a model can reorder the search but can
never invent physics, weaken a gate, or author an outcome.

Three refusals define that boundary:

* an advisory naming a rung the executor cannot perform is rejected whole,
  rather than silently dropping the unknown entry;
* an advisory carrying any outcome vocabulary is rejected, so a model cannot
  smuggle a verdict into a plan input;
* an unreachable or malformed advisory falls back to the default ladder, so a
  provider outage degrades the search order rather than disabling recovery.

The module never calls a model itself.  It validates what a caller obtained,
and seals which ladder was used and why, so a receipt distinguishes an
agent-ranked run from a default-ranked one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .adp009d_control_episode import (
    TASK_CONTROL_RECOVERY_LADDER,
    recovery_ladder_for_plan,
)
from .decision_evidence_contracts import canonical_digest


ADVISORY_SCHEMA_VERSION = "native_task_arena_recovery_advisory.v1"

#: Vocabulary an advisory may never contain.  A ranking over strategies is the
#: whole of what an agent is allowed to express here; anything that reads as a
#: verdict belongs to the deterministic scorer alone.
FORBIDDEN_ADVISORY_KEYS = frozenset(
    {
        "arrival_tolerance_m",
        "caller_asserted_success",
        "control_passed",
        "controls_qualified",
        "outcome",
        "target_reached",
        "task_succeeded",
    }
)

BLOCKER_SCHEMA = "recovery_advisory_schema_invalid"
BLOCKER_UNKNOWN_STRATEGY = "recovery_advisory_unknown_strategy"
BLOCKER_FORBIDDEN_KEY = "recovery_advisory_outcome_vocabulary_forbidden"
BLOCKER_EMPTY = "recovery_advisory_ladder_empty"
BLOCKER_DUPLICATE = "recovery_advisory_ladder_duplicated"


class RecoveryAdvisoryError(ValueError):
    """Fail-closed advisory contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def summarize_attempts_for_advice(episode: Mapping[str, Any]) -> dict[str, Any]:
    """The measured facts an adviser may see, and nothing else.

    Deliberately narrow: per-attempt strategy, measured errors, and the
    actuator saturation that took thirty-two runs to notice.  No verdict, no
    tolerance, and no scoring vocabulary, so an adviser reasons about what the
    arm did rather than about whether the run should be called a pass.
    """

    arrivals = [
        row
        for row in (episode.get("phase_arrivals") or [])
        if isinstance(row, Mapping)
    ]
    actions = [
        row for row in (episode.get("action_trace") or []) if isinstance(row, Mapping)
    ]
    phases: dict[str, list[dict[str, Any]]] = {}
    for row in arrivals:
        phases.setdefault(str(row.get("phase_id") or ""), []).append(
            {
                "attempt": row.get("attempt"),
                "strategy": row.get("recovery_strategy"),
                "position_error_m": row.get("terminal_position_error_m"),
                "orientation_error_rad": row.get("terminal_orientation_error_rad"),
                "commanded_position_bias_m": row.get("commanded_position_bias_m"),
            }
        )
    saturation: dict[str, dict[str, Any]] = {}
    for row in actions:
        phase_id = str(row.get("phase_id") or "")
        dynamics = row.get("arm_dynamics_after")
        if not isinstance(dynamics, Mapping):
            continue
        utilization = dynamics.get("joint_effort_utilization")
        if not isinstance(utilization, list) or not utilization:
            continue
        bucket = saturation.setdefault(
            phase_id, {"steps": 0, "saturated_steps": 0, "maximum_utilization": 0.0}
        )
        peak = max(float(value) for value in utilization)
        bucket["steps"] += 1
        bucket["maximum_utilization"] = max(bucket["maximum_utilization"], peak)
        if peak > 0.999:
            bucket["saturated_steps"] += 1
    return {
        "schema_version": "native_task_arena_recovery_attempt_summary.v1",
        "phases": phases,
        "actuator_saturation": saturation,
        "available_strategies": list(TASK_CONTROL_RECOVERY_LADDER),
        "claim_boundary": (
            "measured_attempt_telemetry_only;carries_no_outcome_tolerance_or_"
            "grading_vocabulary"
        ),
    }


def _forbidden_paths(value: Any, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_ADVISORY_KEYS:
                found.append(path)
            found.extend(_forbidden_paths(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, prefix=f"{prefix}[{index}]"))
    return found


def validate_recovery_advisory(value: Mapping[str, Any]) -> dict[str, Any]:
    """Admit a ranking over implemented rungs; refuse anything more."""

    try:
        advisory = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RecoveryAdvisoryError([BLOCKER_SCHEMA]) from exc
    errors: list[str] = []
    if advisory.get("schema_version") != ADVISORY_SCHEMA_VERSION:
        errors.append(BLOCKER_SCHEMA)
    if _forbidden_paths(advisory):
        errors.append(BLOCKER_FORBIDDEN_KEY)
    ladder = advisory.get("recovery_strategy_ladder")
    if not isinstance(ladder, list) or not ladder:
        errors.append(BLOCKER_EMPTY)
        ladder = []
    rungs = [str(entry) for entry in ladder]
    unknown = [rung for rung in rungs if rung not in TASK_CONTROL_RECOVERY_LADDER]
    if unknown:
        # Rejected whole rather than filtered: a caller that asked for physics
        # the executor cannot perform did not mean the subset that remains.
        errors.append(f"{BLOCKER_UNKNOWN_STRATEGY}:{','.join(sorted(set(unknown)))}")
    if len(set(rungs)) != len(rungs):
        errors.append(BLOCKER_DUPLICATE)
    if not str(advisory.get("rationale") or "").strip():
        errors.append(BLOCKER_SCHEMA)
    if errors:
        raise RecoveryAdvisoryError(errors)
    return advisory


def plan_with_advised_ladder(
    *,
    control_plan: Mapping[str, Any],
    advisory: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind an advised ladder into a plan, or fall back and say so."""

    plan = json.loads(json.dumps(dict(control_plan), allow_nan=False))
    receipt: dict[str, Any] = {
        "schema_version": "native_task_arena_recovery_ladder_binding.v1",
        "source": "default_ladder",
        "advisory_digest": None,
        "advisory_rationale": None,
        "blockers": [],
        "ladder": list(recovery_ladder_for_plan(plan)),
        "claim_boundary": (
            "search_order_only;the_arrival_contact_and_task_gates_are_"
            "unchanged_and_the_deterministic_scorer_remains_the_sole_grader"
        ),
    }
    if advisory is None:
        return plan, receipt
    try:
        checked = validate_recovery_advisory(advisory)
    except RecoveryAdvisoryError as exc:
        # A refused advisory degrades the search order, never the recovery.
        receipt["source"] = "default_ladder_after_refused_advisory"
        receipt["blockers"] = list(exc.errors)
        return plan, receipt
    plan["recovery_strategy_ladder"] = list(checked["recovery_strategy_ladder"])
    receipt.update(
        {
            "source": "agent_advisory",
            "advisory_digest": canonical_digest(checked),
            "advisory_rationale": str(checked["rationale"]),
            "ladder": list(recovery_ladder_for_plan(plan)),
        }
    )
    return plan, receipt


__all__ = [
    "ADVISORY_SCHEMA_VERSION",
    "FORBIDDEN_ADVISORY_KEYS",
    "RecoveryAdvisoryError",
    "plan_with_advised_ladder",
    "summarize_attempts_for_advice",
    "validate_recovery_advisory",
]
