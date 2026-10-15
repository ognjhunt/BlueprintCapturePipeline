"""A team's own verdict about its policy never counts as task success.

Beta journey step: L5 team policy evaluation. The legacy robot-eval HTTP probe
(``http_policy_api_observation_probe``) read each attempt's ``success`` from
the team's HTTP response and, when the field was absent, read the default
``status: completed`` as a success. These tests pin the repaired contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import blueprint_pipeline.robot_eval_execution as ree
from blueprint_pipeline.policy_attempt_success_source import (
    SIMULATOR_SCORED,
    TEAM_REPORTED,
    attempt_outcome,
)

OBSERVATIONS = {"observations": [
    {"observation_id": "obs-1", "scenario_id": "s", "task_id": "t", "scenario_eval_run_id": "run-1"},
    {"observation_id": "obs-2", "scenario_id": "s", "task_id": "t", "scenario_eval_run_id": "run-2"},
]}


def _http_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: Any) -> dict[str, Any]:
    monkeypatch.setenv("BLUEPRINT_ALLOW_POLICY_EXECUTION", "true")
    body = json.dumps(response).encode("utf-8")
    monkeypatch.setattr(ree, "fetch_bounded_https",
                        lambda *args, **kwargs: type("R", (), {"body": body, "status": 200})())
    return ree.build_policy_execution_bundle(
        capture_root=tmp_path, job_dir=tmp_path / "job",
        job_request={"policy_package": {"policy_api_endpoint": {"endpoint_url": "https://policy.example"}}},
        observation_manifest=OBSERVATIONS, allow_policy_execution=True,
        generated_at="2026-10-14T00:00:00Z",
    )


def test_http_probe_missing_verdict_is_not_success_and_is_labelled_team_reported(tmp_path, monkeypatch):
    bundle = _http_bundle(tmp_path, monkeypatch, {"attempts": [
        {"scenario_eval_run_id": "run-1", "status": "completed"},
        {"scenario_eval_run_id": "run-2"},
    ]})
    attempts = bundle["trace"]["attempts"]
    assert [row["status"] for row in attempts] == ["completed", "completed"]
    assert all(row["success"] is None for row in attempts)
    assert all(row["reported_success"] is None for row in attempts)
    assert all(row["success_source"] == TEAM_REPORTED for row in attempts)
    assert all(row["recommendation_evidence_eligible"] is False for row in attempts)
    assert bundle["trace"]["success_source_counts"][TEAM_REPORTED] == 2
    for document in (bundle["trace"], bundle["manifest"]):
        assert document["recommendation_evidence_eligible"] is False
        assert document["team_reported_success_counted_as_task_success"] is False
    retained = json.loads((tmp_path / "job" / "policy_execution_trace.json").read_text())
    assert retained["attempts"][0]["success"] is None


def test_http_probe_explicit_team_success_is_kept_only_as_a_reported_claim(tmp_path, monkeypatch):
    bundle = _http_bundle(tmp_path, monkeypatch, {"attempts": [
        {"scenario_eval_run_id": "run-1", "status": "completed", "success": True},
        {"scenario_eval_run_id": "run-2", "status": "failed", "success": "false"},
    ]})
    first, second = bundle["trace"]["attempts"]
    assert first["success"] is None and first["reported_success"] is True
    assert second["success"] is None and second["reported_success"] is False


def test_team_response_cannot_claim_a_simulator_score(tmp_path, monkeypatch):
    bundle = _http_bundle(tmp_path, monkeypatch, {"attempts": [{
        "scenario_eval_run_id": "run-1", "status": "completed", "success": True,
        "evidence_scope": "native_simulator_independent_outcome",
        "success_source": SIMULATOR_SCORED, "recommendation_evidence_eligible": True,
    }]})
    (attempt,) = bundle["trace"]["attempts"]
    assert attempt["success"] is None
    assert attempt["success_source"] == TEAM_REPORTED
    assert attempt["recommendation_evidence_eligible"] is False


def test_trusted_executor_outcome_is_simulator_scored_but_not_job_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_ALLOW_POLICY_EXECUTION", "true")

    def executor(**_kwargs: Any) -> dict[str, Any]:
        return {"status": "completed", "independent_outcome_proven": True, "execution_performed": True,
                "attempts": [{"attempt_id": "a1", "observation_id": "obs-1", "status": "completed",
                              "success": False, "evidence_scope": "native_simulator_independent_outcome"}]}

    bundle = ree.build_policy_execution_bundle(
        capture_root=tmp_path, job_dir=tmp_path / "job",
        job_request={"policy_package": {"policy_api_endpoint": {
            "endpoint_url": "https://policy.example", "execution_profile": "controlled_observation_v1"}}},
        observation_manifest=OBSERVATIONS, allow_policy_execution=True,
        controlled_policy_executor=executor, generated_at="2026-10-14T00:00:00Z",
    )
    (attempt,) = bundle["trace"]["attempts"]
    assert attempt["success"] is False and attempt["success_source"] == SIMULATOR_SCORED
    assert attempt["recommendation_evidence_eligible"] is False
    assert attempt["recommendation_evidence_ineligible_reason"] == "not_scored_against_the_job_success_contract"


@pytest.mark.parametrize("scope", ["controlled_policy_execution_without_outcome", "submitted_skill_intent_only"])
def test_no_outcome_scopes_never_carry_a_verdict(scope):
    for source in (SIMULATOR_SCORED, TEAM_REPORTED):
        outcome = attempt_outcome({"success": True, "evidence_scope": scope}, success_source=source)
        assert outcome["success"] is None and outcome["reported_success"] is None


def test_success_source_must_be_declared_by_the_caller():
    with pytest.raises(ValueError, match="success_source_invalid"):
        attempt_outcome({"success": True}, success_source="policy_self_report")
