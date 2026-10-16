"""Report one team's scored evaluation as a simulator-scored agent-run result.

Beta journey step: L5 team policy evaluation. The input is every closed session
result of the run (one per shard),
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
evidence.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from .agent_run_result_scoring import SIMULATOR_SCORED, TEAM_RUNTIMES, scoring_statement
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_policy_canary_session import (
    CLAIM_CEILING,
    TEAM_RUN_KIND,
    PolicyCanarySessionError,
    validate_session_result,
)
from .team_evaluation_plan import EPISODES, MAXIMUM_VOID_SHARE, PURPOSES

from .team_policy_episode_outcome import episode_outcome as episode_outcome


class TeamCandidateResultError(ValueError):
    """The session results cannot be reported for this admitted team candidate."""



def _cycle_seconds(row: Mapping[str, Any]) -> float | None:
    """Conservative completed-episode duration on the simulator's own clock.

    Includes the terminal settling window; never substitutes wall-clock
    render, provider, or policy-service timings. Old receipts stay unmeasured.
    """

    episode = row.get("episode") if isinstance(row.get("episode"), Mapping) else {}
    trace = episode.get("state_trace") if isinstance(episode.get("state_trace"), Mapping) else {}
    samples = trace.get("task_state_samples")
    if (trace.get("trace_digest") != canonical_digest(trace, digest_field="trace_digest")
            or row.get("state_trace_digest") != canonical_digest({"value": trace})
            or not isinstance(samples, list) or len(samples) < 2):
        return None
    times = []
    for sample in samples:
        value = sample.get("simulation_time_s") if isinstance(sample, Mapping) else None
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return None
        times.append(float(value))
    if any(right <= left for left, right in pairwise(times)):
        return None
    return times[-1] - times[0]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    return ordered[lower] + (ordered[math.ceil(index)] - ordered[lower]) * (index - lower)


def team_candidate_terminal(*, session_results: Sequence[Mapping[str, Any]], runtime_binding: Mapping[str, Any],
                            purpose: str, episodes: int | None = None,
                            expected_cells: Sequence[Mapping[str, Any]] | None = None,
                            pre_policy_recovery_cells: Sequence[str] = (),
                            retained_episode_digests: Mapping[str, str | list[str]] | None = None) -> dict[str, Any]:
    """Return the terminal receipt (outcomes, counts, scoring) for the agent-run executor."""

    if purpose not in PURPOSES or not session_results:
        raise TeamCandidateResultError("team_candidate_result_purpose_invalid")
    binding = dict(runtime_binding)
    rows: list[dict[str, Any]] = []
    result_digests: list[str] = []
    session_rows: list[tuple[str, Mapping[str, Any]]] = []
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
        session_rows.extend((result["result_digest"], row) for row in result["episodes"]
                            if row.get("candidate_id") == binding["candidate_id"])
        result_digests.append(result["result_digest"])
    recovery = None
    if pre_policy_recovery_cells:
        from .team_candidate_episode_recovery import select_pre_policy_reset_recovery
        try:
            rows, recovery = select_pre_policy_reset_recovery(session_rows,
                expected_cells=expected_cells or (), recovery_cell_ids=pre_policy_recovery_cells,
                retained_episode_digests=retained_episode_digests or {})
        except ValueError as exc:
            raise TeamCandidateResultError(str(exc)) from exc
    cell_ids = [str(row.get("cell_id") or "") for row in rows]
    # Exactly the admitted episode count (the admission's; the purpose's by default).
    if len(rows) != (EPISODES[purpose] if episodes is None else int(episodes)) or len(set(cell_ids)) != len(rows):
        raise TeamCandidateResultError("team_candidate_session_result_episode_count_invalid")
    outcomes = []
    for row in rows:
        outcome, reason = episode_outcome(row)
        scenario = row.get("resolved_scenario") if isinstance(row.get("resolved_scenario"), Mapping) else {}
        outcomes.append({
            "cell_id": row.get("cell_id"), "family": row.get("family") or scenario.get("family"),
            # Cells are hidden: the team learns each cell's family, outcome and reason, never its draw.
            "outcome": outcome, "reason": reason,
        })
    counted = [row for row in outcomes if row["outcome"] != "void"]
    succeeded = sum(1 for row in counted if row["outcome"] == "success")
    void = len(outcomes) - len(counted)
    if void > math.floor(MAXIMUM_VOID_SHARE * len(outcomes)):
        raise TeamCandidateResultError("team_scored_run_void_share_exceeded")
    successful_rows = [row for row in rows if episode_outcome(row)[0] == "success"]
    cycles = [_cycle_seconds(row) for row in successful_rows]
    measured_cycles = [value for value in cycles if value is not None]
    complete_cycles = bool(cycles) and len(measured_cycles) == len(cycles)
    return {
        "status": "completed",
        **({"pre_policy_reset_recovery": recovery} if recovery is not None else {}),
        "episodes_run": len(counted),
        "episodes_succeeded": succeeded,
        "median_cycle_seconds": statistics.median(measured_cycles) if complete_cycles else None,
        "cycle_seconds_p10": _percentile(measured_cycles, 0.1) if complete_cycles else None,
        "cycle_seconds_p90": _percentile(measured_cycles, 0.9) if complete_cycles else None,
        "note": (f"Scored evaluation of one team's own policy: {len(counted)} episodes scored by the simulator against "
                 f"the sealed success contract, {succeeded} succeeded"
                 + (f", {void} void (simulator failure, not counted)" if void else "")
                 + (". Cycle time is completed-episode simulator duration, including settling"
                    if complete_cycles else ". Cycle time unmeasured")
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
