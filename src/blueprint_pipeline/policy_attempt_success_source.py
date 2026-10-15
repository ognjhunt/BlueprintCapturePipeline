"""Who decided a policy attempt's task outcome, stated on every attempt.

Beta journey step: L5 team policy evaluation. A recommendation needs a team's
own policy scored by the simulator against the job's frozen success contract,
never a verdict the team's policy or service reported about itself.

The legacy robot-eval policy bundle normalizes several submission paths into
one attempt trace: a trusted native executor's independent outcome, a team's
HTTP response, a team's command output or reference replay, and Blueprint's
synthetic default test policy. Only the first is a simulator score, and none of
them is scored against a website job's success contract. So every attempt names
its source, a missing verdict is never read as success, a team's own verdict is
kept only as ``reported_success``, and no attempt from this bundle is eligible
as recommendation evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SIMULATOR_SCORED = "simulator_scored"
TEAM_REPORTED = "team_reported"
BLUEPRINT_DEFAULT_TEST_POLICY = "blueprint_default_test_policy"
SUCCESS_SOURCES = (SIMULATOR_SCORED, TEAM_REPORTED, BLUEPRINT_DEFAULT_TEST_POLICY)

# Attempts whose scope says no outcome exists at all.
NO_OUTCOME_SCOPES = frozenset(
    {"submitted_skill_intent_only", "controlled_policy_execution_without_outcome"}
)
# Only a trusted executor's own scoring scope can carry a simulator verdict.
SIMULATOR_OUTCOME_SCOPE = "native_simulator_independent_outcome"

_INELIGIBLE_REASONS = {
    SIMULATOR_SCORED: "not_scored_against_the_job_success_contract",
    TEAM_REPORTED: "team_reported_success_is_not_a_task_outcome",
    BLUEPRINT_DEFAULT_TEST_POLICY: "blueprint_default_test_policy_is_not_a_team_policy",
}


def _verdict(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "passed", "success", "succeeded"}


def attempt_outcome(raw: Mapping[str, Any], *, success_source: str) -> dict[str, Any]:
    """Return the outcome fields for one normalized attempt.

    ``success_source`` is declared by the trusted caller from the path it ran,
    never read from the payload: a team's response could name any source.
    """

    if success_source not in SUCCESS_SOURCES:
        raise ValueError("policy_attempt_success_source_invalid")
    scope = raw.get("evidence_scope")
    explicit = None if scope in NO_OUTCOME_SCOPES else _verdict(raw.get("success"))
    if success_source == TEAM_REPORTED:
        success = None
    elif success_source == SIMULATOR_SCORED:
        success = explicit if scope == SIMULATOR_OUTCOME_SCOPE else None
    else:
        success = explicit
    return {
        "success": success,
        "reported_success": explicit if success_source == TEAM_REPORTED else None,
        "success_source": success_source,
        "recommendation_evidence_eligible": False,
        "recommendation_evidence_ineligible_reason": _INELIGIBLE_REASONS[success_source],
    }


def success_source_summary(attempts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate attempt sources for a trace; this bundle never backs a recommendation."""

    counts = {source: 0 for source in SUCCESS_SOURCES}
    for attempt in attempts:
        source = attempt.get("success_source")
        if source in counts:
            counts[source] += 1
    return {
        "success_source_counts": counts,
        "team_reported_success_counted_as_task_success": False,
        "recommendation_evidence_eligible": False,
    }


__all__ = [
    "BLUEPRINT_DEFAULT_TEST_POLICY",
    "SIMULATOR_SCORED",
    "SUCCESS_SOURCES",
    "TEAM_REPORTED",
    "attempt_outcome",
    "success_source_summary",
]
