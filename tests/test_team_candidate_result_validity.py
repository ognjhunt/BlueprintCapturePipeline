"""L5 invalid media stays visible without excluding policy failures or inventing cycle time."""

from __future__ import annotations

import copy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import _episode_evidence_valid
from blueprint_pipeline.policy_canary_worker_evidence import _write_episode_failure_gap
from blueprint_pipeline.team_candidate_quick10_result import (
    TeamCandidateResultError,
    episode_outcome,
    team_candidate_terminal,
)
from tests.test_team_candidate_quick10_rehearsal import (
    _fake_closed_team_result,
    _stage_team_runtime,
)


def _reseal(value):
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def _media_failure(row, *, policy_failure=False):
    row.update(status="blocked", episode_failure_stage="after_first_observation",
               episode_failure_cause="policy" if policy_failure else "media",
               candidate_policy_query_attempted=True, candidate_policy_queried=True,
               candidate_action_returned=True, candidate_action_shape_validated=True,
               candidate_action_finite_validated=True, first_observation_retained=True,
               policy_response_status="received", policy_outcome_interpretable=False,
               episode={"score": {"status": "not_scored"}},
               review_video_digest=None,
               visual_evidence={"status": "incomplete_after_first_observation", "media_gap": {
                   "type": "after_first_observation_media_seal_failed", "reason": "OSError"}})


def test_media_voids_are_reported_and_policy_failures_still_count(tmp_path):
    _root, _out, inputs, _admission = _stage_team_runtime(tmp_path)
    first = _fake_closed_team_result(inputs)
    second = copy.deepcopy(first)
    for row in second["episodes"]:
        row["cell_id"] += "-second"
    _media_failure(first["episodes"][0])
    _media_failure(first["episodes"][1], policy_failure=True)
    first["status"] = "blocked"
    _reseal(first)
    _reseal(second)
    assert _episode_evidence_valid(first["episodes"][0])
    # ADP's complete-media boundary is unchanged.
    assert not _episode_evidence_valid({**first["episodes"][0], "run_kind": "internal_policy_canary"})
    assert episode_outcome(first["episodes"][0]) == ("void", "episode_media_unavailable")
    assert episode_outcome(first["episodes"][1]) == ("failure", "policy_service_error")
    terminal = team_candidate_terminal(session_results=[first, second], purpose="scored", episodes=20,
                                       runtime_binding=inputs["team_candidate_admission"])
    assert len(terminal["episode_outcomes"]) == 20
    assert terminal["episodes_run"] == terminal["scoring"]["scored_episode_count"] == 19
    assert terminal["episodes_succeeded"] == 18
    assert sum(row["outcome"] == "void" for row in terminal["episode_outcomes"]) == 1
    # A second infrastructure void exceeds the sealed 5% allowance.
    _media_failure(first["episodes"][2])
    _reseal(first)
    with pytest.raises(TeamCandidateResultError, match="void_share_exceeded"):
        team_candidate_terminal(session_results=[first, second], purpose="scored", episodes=20,
                                runtime_binding=inputs["team_candidate_admission"])


def test_primary_policy_timeout_survives_secondary_media_failure(tmp_path):
    path = _write_episode_failure_gap(output_root=tmp_path, run_id="run",
        context={"cell_id": "cell", "candidate_id": "team"}, failure=TimeoutError("policy timed out"),
        progress={"first_observation_retained": True, "candidate_policy_query_attempted": True,
                  "policy_inference_failed": True})
    evidence = json.loads(path.read_text())
    assert evidence["episode_failure_cause"] == "policy"
    assert evidence["visual_evidence"]["media_gap"]
    assert episode_outcome(evidence)[0] == "failure"


def test_request_media_disk_failure_is_infrastructure_even_inside_policy_call(tmp_path):
    path = _write_episode_failure_gap(output_root=tmp_path, run_id="run",
        context={"cell_id": "cell", "candidate_id": "team"}, failure=OSError("request media disk full"),
        progress={"first_observation_retained": True, "candidate_policy_query_attempted": True,
                  "policy_inference_failed": True, "policy_request_retention_failed": True})
    evidence = json.loads(path.read_text())
    assert evidence["episode_failure_cause"] == "infrastructure"
    assert episode_outcome(evidence)[0] == "void"


def test_cycle_time_uses_complete_bound_simulator_traces_only(tmp_path):
    _root, _out, inputs, _admission = _stage_team_runtime(tmp_path)
    result = _fake_closed_team_result(inputs)
    for index, row in enumerate(result["episodes"]):
        trace = {"schema_version": "policy_episode_state_trace.v1",
                 "task_state_samples": [{"step_index": 0, "simulation_time_s": 1.0},
                                        {"step_index": 1, "simulation_time_s": 2.0 + index}],
                 "trace_digest": ""}
        trace["trace_digest"] = canonical_digest(trace, digest_field="trace_digest")
        row["episode"]["state_trace"] = trace
        row["state_trace_digest"] = canonical_digest({"value": trace})
        row["episode"]["performance_diagnostics"] = {"timings_seconds": {"episode_total": 999}}
    _reseal(result)
    kwargs = {"purpose": "scored", "episodes": 10, "runtime_binding": inputs["team_candidate_admission"]}
    terminal = team_candidate_terminal(session_results=[result], **kwargs)
    assert terminal["median_cycle_seconds"] == pytest.approx(5.5)
    assert terminal["cycle_seconds_p10"] == pytest.approx(1.9)
    assert terminal["cycle_seconds_p90"] == pytest.approx(9.1)
    # Incomplete or tampered clocks cannot be replaced by provider wall time.
    result["episodes"][0]["episode"]["state_trace"]["task_state_samples"][1]["simulation_time_s"] = 0.0
    _reseal(result)
    assert team_candidate_terminal(session_results=[result], **kwargs)["median_cycle_seconds"] is None
