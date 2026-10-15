"""Report one team candidate's Quick-10 run as a simulator-scored agent-run result.

Beta journey step: L5 team policy evaluation. The input is the sealed, closed
session result of the ``team_candidate_evaluation`` run (provider teardown and
provider-zero already confirmed); the output is the terminal receipt the agent
run executor posts to the WebApp, with its scoring statement.

Counting is conservative and read only from the simulator:

- ``episodes_run`` is every planned episode of the one candidate (ten);
- an episode is *scored* only when it completed, its scoring authority is the
  deterministic simulator state, and the scorer returned ``scored`` with a
  boolean verdict against the run's sealed success contract;
- ``episodes_succeeded`` counts scored successes and nothing else.

A blocked or unscored episode stays in ``episodes_run`` as a non-success and
lowers ``scored_episode_count``; the WebApp then refuses the result as
recommendation evidence because the counts differ. Nothing the team's service
returned besides actions is read. The claim ceiling stays
``diagnostic_policy_execution``: a bounded simulation estimate that never
certifies physical performance or safety, and never enters Arm Decision Proof
qualified evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .agent_run_result_scoring import SIMULATOR_SCORED, TEAM_CANDIDATE_QUICK10, scoring_statement
from .native_task_arena_policy_canary_session import (
    CLAIM_CEILING,
    EPISODES_PER_POLICY,
    TEAM_RUN_KIND,
    PolicyCanarySessionError,
    validate_session_result,
)


class TeamCandidateResultError(ValueError):
    """The session result cannot be reported for this admitted team candidate."""


def _scored(row: Mapping[str, Any]) -> bool | None:
    """The simulator's verdict for one episode, or None when it was not scored."""

    score = (row.get("episode") or {}).get("score") if isinstance(row.get("episode"), Mapping) else None
    if (row.get("status") != "completed" or row.get("scoring_authority") != "deterministic_simulator_state"
            or not isinstance(score, Mapping) or score.get("status") != "scored"
            or not isinstance(score.get("task_succeeded"), bool)):
        return None
    return score["task_succeeded"]


def team_candidate_terminal(*, session_result: Mapping[str, Any],
                            runtime_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Return the terminal receipt (episodes, successes, scoring) for the agent-run executor."""

    try:
        result = validate_session_result(session_result)
    except PolicyCanarySessionError as exc:
        raise TeamCandidateResultError("team_candidate_session_result_invalid:" + str(exc)) from exc
    binding = dict(runtime_binding)
    if (result["run_kind"] != TEAM_RUN_KIND or result["claim_ceiling"] != CLAIM_CEILING
            or result["candidate_ids"] != [binding.get("candidate_id")]
            or result["task_success_contract_digest"] != binding.get("task_success_contract_digest")):
        raise TeamCandidateResultError("team_candidate_session_result_binding_mismatch")
    episodes = [row for row in result["episodes"] if row.get("candidate_id") == binding["candidate_id"]]
    if len(episodes) != EPISODES_PER_POLICY:
        raise TeamCandidateResultError("team_candidate_session_result_episode_count_invalid")
    verdicts = [_scored(row) for row in episodes]
    scored = [verdict for verdict in verdicts if verdict is not None]
    succeeded = sum(1 for verdict in scored if verdict)
    return {
        "status": "completed",
        "episodes_run": len(episodes),
        "episodes_succeeded": succeeded,
        "median_cycle_seconds": None,
        "cycle_seconds_p10": None,
        "cycle_seconds_p90": None,
        "note": (f"Quick-10 simulation of one admitted team candidate: {len(scored)} of {len(episodes)} "
                 f"episodes scored by the simulator against the sealed success contract, {succeeded} succeeded. "
                 "Diagnostic simulation estimate; not physical performance or safety."),
        "artifact_uri": None,
        "scoring": scoring_statement(
            runtime=TEAM_CANDIDATE_QUICK10, success_source=SIMULATOR_SCORED,
            scored_episode_count=len(scored), claim_ceiling=CLAIM_CEILING,
            success_contract_digest=result["task_success_contract_digest"],
            scene_revision_digest=binding.get("scene_revision_digest"),
            robot_free_probe_receipt_digest=binding.get("robot_free_probe_receipt_digest"),
            team_candidate_admission_digest=binding.get("admission_digest"),
            session_result_digest=result["result_digest"],
        ),
    }


__all__ = ["TeamCandidateResultError", "team_candidate_terminal"]
