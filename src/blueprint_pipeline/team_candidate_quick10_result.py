"""Report one team's evaluation (practice or scored) as a simulator-scored agent-run result.

Beta journey step: L5 team policy evaluation. The input is every closed session
result of the run (one for a practice run, one per shard for a scored run),
each with provider teardown and provider-zero confirmed; the output is the
terminal receipt the agent-run executor posts to the WebApp, with its scoring
statement and one outcome row per episode.

Each episode is exactly one of:

- **success**: the simulator scored it against the sealed success contract and
  the task succeeded;
- **failure**: the simulator scored it a failure, or the team's policy caused it
  to end: it commanded an action outside the robot's joint limits, its service
  did not answer within the request timeout, or it answered with an action of
  the wrong shape or non-finite values. A robot that cannot act on time, or
  acts outside its own limits, has failed the job;
- **void**: the simulator, not the policy, failed (a reset that never reached
  the first observation, a harness error after a valid action, an undetermined
  score). Void episodes are reported, never counted either way, and a scored
  run with more than ``MAXIMUM_VOID_SHARE`` of them is refused.

``episodes_run`` and ``scored_episode_count`` are the success and failure
episodes; ``episodes_succeeded`` counts successes. Nothing the team's service
returns besides actions is read as an outcome. The claim ceiling stays
``diagnostic_policy_execution``: a bounded simulation estimate that never
certifies physical performance or safety, and never Arm Decision Proof
evidence. A practice run is labelled with its own runtime so the WebApp never
counts it toward a recommendation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .agent_run_result_scoring import SIMULATOR_SCORED, TEAM_RUNTIMES, scoring_statement
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_policy_canary_session import (
    CLAIM_CEILING,
    TEAM_RUN_KIND,
    PolicyCanarySessionError,
    validate_session_result,
)
from .team_evaluation_plan import EPISODES, MAXIMUM_VOID_SHARE, PRACTICE, PURPOSES, SCORED

_POLICY_ERROR_PREFIXES = ("droid_https_policy_", "team_policy_https_")


class TeamCandidateResultError(ValueError):
    """The session results cannot be reported for this admitted team candidate."""


def _scored(row: Mapping[str, Any]) -> bool | None:
    """The simulator's verdict for one episode, or None when it was not scored."""

    score = (row.get("episode") or {}).get("score") if isinstance(row.get("episode"), Mapping) else None
    if (row.get("status") != "completed" or row.get("scoring_authority") != "deterministic_simulator_state"
            or not isinstance(score, Mapping) or score.get("status") != "scored"
            or not isinstance(score.get("task_succeeded"), bool)):
        return None
    return score["task_succeeded"]


def episode_outcome(row: Mapping[str, Any]) -> tuple[str, str]:
    """``(outcome, reason)``: success, failure (scored or policy-caused) or void (simulator)."""

    verdict = _scored(row)
    if verdict is not None:
        return ("success", "task_succeeded") if verdict else ("failure", "task_not_completed")
    if row.get("status") == "completed":
        return "void", "simulator_score_undetermined"
    if row.get("episode_failure_stage") == "action_delivery_rejected":
        return "failure", "policy_action_beyond_joint_limits"
    attempted = row.get("candidate_policy_query_attempted") is True or row.get("candidate_policy_queried") is True
    message = str(row.get("failure_message") or "")
    if attempted:
        if row.get("policy_response_status") not in (None, "received"):
            return "failure", "policy_service_did_not_answer"
        if row.get("candidate_action_returned") is True and not (
                row.get("candidate_action_shape_validated") is True
                and row.get("candidate_action_finite_validated") is True):
            return "failure", "policy_action_invalid"
        if message.startswith(_POLICY_ERROR_PREFIXES):
            return "failure", "policy_service_error"
    if row.get("episode_failure_stage") == "after_first_observation":
        return "void", "simulator_failed_during_episode"
    return "void", "simulator_failed_before_policy"


def team_candidate_terminal(*, session_results: Sequence[Mapping[str, Any]], runtime_binding: Mapping[str, Any],
                            purpose: str) -> dict[str, Any]:
    """Return the terminal receipt (outcomes, counts, scoring) for the agent-run executor."""

    if purpose not in PURPOSES or not session_results:
        raise TeamCandidateResultError("team_candidate_result_purpose_invalid")
    binding = dict(runtime_binding)
    rows: list[dict[str, Any]] = []
    result_digests: list[str] = []
    for value in session_results:
        try:
            result = validate_session_result(value)
        except PolicyCanarySessionError as exc:
            raise TeamCandidateResultError("team_candidate_session_result_invalid:" + str(exc)) from exc
        if (result["run_kind"] != TEAM_RUN_KIND or result["claim_ceiling"] != CLAIM_CEILING
                or result["candidate_ids"] != [binding.get("candidate_id")]
                or result["task_success_contract_digest"] != binding.get("task_success_contract_digest")):
            raise TeamCandidateResultError("team_candidate_session_result_binding_mismatch")
        rows.extend(row for row in result["episodes"] if row.get("candidate_id") == binding["candidate_id"])
        result_digests.append(result["result_digest"])
    cell_ids = [str(row.get("cell_id") or "") for row in rows]
    if len(rows) != EPISODES[purpose] or len(set(cell_ids)) != len(rows):
        raise TeamCandidateResultError("team_candidate_session_result_episode_count_invalid")
    outcomes = []
    for row in rows:
        outcome, reason = episode_outcome(row)
        scenario = row.get("resolved_scenario") if isinstance(row.get("resolved_scenario"), Mapping) else {}
        outcomes.append({
            "cell_id": row.get("cell_id"), "family": row.get("family") or scenario.get("family"),
            "outcome": outcome, "reason": reason,
            # Practice cells are published; a scored run reports only each cell's family.
            **({"parameters": dict(scenario.get("parameters") or {})} if purpose == PRACTICE else {}),
        })
    counted = [row for row in outcomes if row["outcome"] != "void"]
    succeeded = sum(1 for row in counted if row["outcome"] == "success")
    void = len(outcomes) - len(counted)
    if purpose == SCORED and void > math.floor(MAXIMUM_VOID_SHARE * len(outcomes)):
        raise TeamCandidateResultError("team_scored_run_void_share_exceeded")
    label = "Practice run" if purpose == PRACTICE else "Scored evaluation"
    return {
        "status": "completed",
        "episodes_run": len(counted),
        "episodes_succeeded": succeeded,
        "median_cycle_seconds": None,
        "cycle_seconds_p10": None,
        "cycle_seconds_p90": None,
        "note": (f"{label} of one team's own policy: {len(counted)} episodes scored by the simulator against "
                 f"the sealed success contract, {succeeded} succeeded"
                 + (f", {void} void (simulator failure, not counted)" if void else "")
                 + ". Simulation estimate; not physical performance or safety."),
        "artifact_uri": None,
        "evaluation_purpose": purpose,
        "episode_outcomes": outcomes,
        "scoring": scoring_statement(
            runtime=TEAM_RUNTIMES[purpose], success_source=SIMULATOR_SCORED,
            scored_episode_count=len(counted), claim_ceiling=CLAIM_CEILING,
            success_contract_digest=binding.get("task_success_contract_digest"),
            scene_revision_digest=binding.get("scene_revision_digest"),
            robot_free_probe_receipt_digest=binding.get("robot_free_probe_receipt_digest"),
            team_candidate_admission_digest=binding.get("admission_digest"),
            session_result_digest=(result_digests[0] if len(result_digests) == 1
                                   else canonical_digest({"shard_session_results": result_digests})),
        ),
    }


__all__ = ["TeamCandidateResultError", "episode_outcome", "team_candidate_terminal"]
