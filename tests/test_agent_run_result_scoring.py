"""Every result reported to the WebApp states who scored its episodes.

Beta journey step: L5 team policy evaluation. The WebApp admits a result as
recommendation evidence only through a positive allowlist of these statements
(Blueprint-WebApp ``server/utils/runResultScoring.ts``); this pins the Pipeline
side of that wire contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import blueprint_pipeline.agent_run_executor as executor
import blueprint_pipeline.controlled_native_queue as native_queue
import blueprint_pipeline.robot_eval_terminal_artifact_adapter as legacy_adapter
from blueprint_pipeline.agent_run_result_scoring import (
    AgentRunResultScoringError,
    scoring_statement,
    validate_scoring_statement,
)

DIGEST = "sha256:" + "a" * 64


def _admitted(**overrides: Any) -> dict[str, Any]:
    return {**scoring_statement(
        runtime="policy_canary_quick10_team_candidate", success_source="simulator_scored",
        scored_episode_count=10, claim_ceiling="diagnostic_policy_execution",
        success_contract_digest=DIGEST, scene_revision_digest=DIGEST, robot_free_probe_receipt_digest=DIGEST,
        team_candidate_admission_digest=DIGEST, session_result_digest=DIGEST), **overrides}


def test_statement_mirrors_the_webapp_wire_schema_exactly() -> None:
    assert set(_admitted()) == {
        "schema_version", "runtime", "success_source", "scored_episode_count", "success_contract_digest",
        "scene_revision_digest", "robot_free_probe_receipt_digest", "team_candidate_admission_digest",
        "session_result_digest", "claim_ceiling"}
    assert validate_scoring_statement(_admitted(), episodes_run=10)["schema_version"] == (
        "blueprint.agent_run_result_scoring.v1")


@pytest.mark.parametrize("bad", [
    {"recommendation_evidence_eligible": True},
    {"scored_episode_count": 11},
    {"scored_episode_count": True},
    {"success_contract_digest": "sha256:short"},
    {"runtime": "Policy Canary"},
    {"schema_version": "blueprint.agent_run_result_scoring.v0"},
])
def test_malformed_or_impossible_statements_are_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(AgentRunResultScoringError):
        validate_scoring_statement(_admitted(**bad), episodes_run=10)


def test_reported_result_carries_the_statement_verbatim() -> None:
    received: list[tuple[str, dict[str, Any]]] = []

    class Client(executor.AgentRunWebAppClient):
        def __init__(self) -> None:
            pass

        def _json(self, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None):
            received.append((path, payload or {}))
            return {"ok": True}

    row = {"reservation_id": "res-1", "team_id": "team-1", "quoted_episodes": 10, "quoted_usd": 0,
           "execution_admission_digest": DIGEST}
    Client().report_completed(row, "attempt-1", {"episodes_run": 10, "episodes_succeeded": 7,
                                                 "scoring": _admitted()})
    assert received[0][0] == "/api/internal/pipeline/agent-run-results"
    assert received[0][1]["scoring"] == _admitted()
    with pytest.raises(AgentRunResultScoringError):
        Client().report_completed(row, "attempt-1", {"episodes_run": 9, "episodes_succeeded": 7,
                                                     "scoring": _admitted()})


def test_legacy_terminal_states_an_unverified_command_batch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(native_queue, "read_native_terminal", lambda **_kwargs: None)
    monkeypatch.setattr(legacy_adapter, "read_terminal_robot_eval_artifacts", lambda _job_dir: {
        "status": "terminal_observed", "job_id": "job-1", "canonical_job_request_digest": DIGEST,
        "episode_result": {"episodes_run": 10, "episodes_succeeded": 9, "metrics_path": "m.json"},
        "cost": {"observed_cost_usd": 0.0}, "blockers": []})
    terminal = executor._default_terminal_reader(job_dir=tmp_path, expected_job_id="job-1",
                                                 expected_canonical_request_digest=DIGEST)
    assert terminal["episodes_succeeded"] == 9
    assert terminal["scoring"]["runtime"] == "legacy_robot_eval_command_batch"
    assert terminal["scoring"]["success_source"] == "unverified"
    assert terminal["scoring"]["scored_episode_count"] == 0


def test_controlled_native_terminal_states_a_sealed_task_score_not_a_job_score(tmp_path, monkeypatch) -> None:
    sealed = {"status": "completed", "episodes_run": 1, "episodes_succeeded": 1,
              "private_execution_result": {"task_spec_digest": DIGEST, "execution_receipt_digest": DIGEST}}
    monkeypatch.setattr(native_queue, "read_native_terminal", lambda **_kwargs: dict(sealed))
    terminal = executor._default_terminal_reader(job_dir=tmp_path, expected_job_id="job-1",
                                                 expected_canonical_request_digest=DIGEST)
    assert terminal["scoring"]["runtime"] == "controlled_native_sealed_task"
    assert terminal["scoring"]["success_source"] == "simulator_scored"
    assert terminal["scoring"]["robot_free_probe_receipt_digest"] is None
    assert terminal["scoring"]["team_candidate_admission_digest"] is None
    monkeypatch.setattr(native_queue, "read_native_terminal", lambda **_kwargs: {"status": "blocked"})
    assert "scoring" not in executor._default_terminal_reader(
        job_dir=tmp_path, expected_job_id="job-1", expected_canonical_request_digest=DIGEST)
