"""Typed simulator execution evidence; scripted baselines never become learned proof."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

LEARNED = "learned_policy"
SCRIPTED = "scripted_simulator_state_baseline"
SCRIPTED_ACTION_SOURCE = "scripted_simulator_state"
KINDS_FIELD = "candidate_execution_evidence_kinds"


def candidate_execution_kinds(payload: Mapping[str, Any]) -> dict[str, str]:
    candidates = payload.get("candidate_ids") or []
    kinds = payload.get(KINDS_FIELD)
    if kinds is None:
        return {candidate: LEARNED for candidate in candidates}
    if (not isinstance(kinds, Mapping) or set(kinds) != set(candidates)
            or any(kind not in (LEARNED, SCRIPTED) for kind in kinds.values())):
        raise ValueError("policy_execution_evidence_kinds_invalid")
    return dict(kinds)


def evidence_kind(row: Mapping[str, Any]) -> str:
    return row.get("execution_evidence_kind", LEARNED)


def execution_evidence_valid(row: Mapping[str, Any]) -> bool:
    kind = evidence_kind(row)
    if kind == SCRIPTED:
        return (row.get("action_source") == SCRIPTED_ACTION_SOURCE
                and row.get("candidate_policy_queried") is False
                and row.get("candidate_policy_query_attempted") in (None, False)
                and row.get("policy_outcome_interpretable") in (None, False)
                and type(row.get("scripted_controller_invoked")) is bool)
    return (kind == LEARNED and row.get("action_source") != SCRIPTED_ACTION_SOURCE
            and row.get("scripted_controller_invoked") in (None, False)
            and row.get("scripted_outcome_interpretable") in (None, False))


def execution_invoked(row: Mapping[str, Any]) -> bool:
    return execution_evidence_valid(row) and row.get(
        "scripted_controller_invoked" if evidence_kind(row) == SCRIPTED else "candidate_policy_queried"
    ) is True


def rollout_counts(kinds: Mapping[str, str], episodes_per_policy: int) -> dict[str, Any]:
    result = {"learned_policy_rollout_count": sum(kind == LEARNED for kind in kinds.values()) * episodes_per_policy}
    if SCRIPTED in kinds.values():
        result.update({KINDS_FIELD: dict(kinds),
                       "scripted_rollout_count": sum(kind == SCRIPTED for kind in kinds.values()) * episodes_per_policy})
    return result


def retained_rollout_counts(kinds: Mapping[str, str], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count retained candidate episodes, including pre-inference failures, by kind.

    Invocation is a separate witnessed fact, never inferred from these counts.
    This preserves the legacy session's rollout/retained-episode convention.
    """
    result = rollout_counts(kinds, 0)
    for row in rows:
        kind = kinds.get(row.get("candidate_id"))
        if kind != evidence_kind(row) or not execution_evidence_valid(row):
            raise ValueError("policy_episode_execution_evidence_mismatch")
        result["scripted_rollout_count" if kind == SCRIPTED else "learned_policy_rollout_count"] += 1
    if SCRIPTED in kinds.values():
        result["scripted_controller_invoked"] = any(row.get("scripted_controller_invoked") is True for row in rows)
    return result


def rollout_total(payload: Mapping[str, Any]) -> int:
    learned, scripted = payload.get("learned_policy_rollout_count"), payload.get("scripted_rollout_count", 0)
    if any(type(count) is not int or count < 0 for count in (learned, scripted)):
        raise ValueError("policy_rollout_count_invalid")
    return learned + scripted


def rollout_counts_valid(payload: Mapping[str, Any], episodes_per_policy: int) -> bool:
    try:
        kinds = candidate_execution_kinds(payload)
        expected = rollout_counts(kinds, episodes_per_policy)
        return (rollout_total(payload) == len(kinds) * episodes_per_policy
                and all(payload.get(key) == value for key, value in expected.items()))
    except (ValueError, TypeError):
        return False


def session_evidence_binding_valid(manifest: Mapping[str, Any], authority: Mapping[str, Any],
                                   inputs: Mapping[str, Any]) -> bool:
    """Enforce the new typed seal; legacy manifests retain their existing validator."""
    if not any(KINDS_FIELD in payload for payload in (manifest, authority, inputs)):
        return True
    try:
        return (candidate_execution_kinds(manifest) == candidate_execution_kinds(authority)
                == candidate_execution_kinds(inputs)
                and rollout_counts_valid(manifest, authority.get("episodes_per_policy")))
    except (TypeError, ValueError):
        return False


def scripted_fields(*, invoked: bool) -> dict[str, Any]:
    return {"execution_evidence_kind": SCRIPTED, "action_source": SCRIPTED_ACTION_SOURCE,
            "scripted_controller_invoked": invoked, "candidate_policy_queried": False,
            "candidate_policy_query_attempted": False, "policy_outcome_interpretable": False}


def scripted_episode_outcome(row: Mapping[str, Any]) -> tuple[str, str]:
    """Use the actual simulator verdict without claiming a learned-policy outcome."""
    if evidence_kind(row) != SCRIPTED or not execution_evidence_valid(row):
        raise ValueError("scripted_episode_evidence_invalid")
    episode = row.get("episode") or {}
    score = episode.get("score") or {}
    if (row.get("status") == "completed" and execution_invoked(row)
            and row.get("scoring_authority") == "deterministic_simulator_state"
            and score.get("status") == "scored" and type(score.get("task_succeeded")) is bool):
        if (episode.get("motion_evidence") or {}).get("robot_fell") is True:
            return "failure", "robot_fell"
        return ("success", "task_succeeded") if score["task_succeeded"] else ("failure", "task_not_completed")
    if row.get("episode_failure_stage") == "action_delivery_rejected":
        return "failure", "scripted_action_beyond_joint_limits"
    if row.get("episode_failure_cause") == "scripted_controller" and execution_invoked(row):
        return "failure", "scripted_controller_failed"
    return "void", "scripted_simulator_episode_unscored"
