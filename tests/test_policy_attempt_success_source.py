"""A team's own verdict about its policy never counts as task success.

Beta journey step: L5 team policy evaluation. The legacy robot-eval HTTP probe
(``http_policy_api_observation_probe``) read each attempt's ``success`` from
the team's HTTP response and, when the field was absent, read the default
``status: completed`` as a success. These tests pin the repaired contract.
"""

from __future__ import annotations


import pytest

from blueprint_pipeline.policy_attempt_success_source import (
    SIMULATOR_SCORED,
    TEAM_REPORTED,
    attempt_outcome,
)

OBSERVATIONS = {"observations": [
    {"observation_id": "obs-1", "scenario_id": "s", "task_id": "t", "scenario_eval_run_id": "run-1"},
    {"observation_id": "obs-2", "scenario_id": "s", "task_id": "t", "scenario_eval_run_id": "run-2"},
]}


@pytest.mark.parametrize("scope", ["controlled_policy_execution_without_outcome", "submitted_skill_intent_only"])
def test_no_outcome_scopes_never_carry_a_verdict(scope):
    for source in (SIMULATOR_SCORED, TEAM_REPORTED):
        outcome = attempt_outcome({"success": True, "evidence_scope": scope}, success_source=source)
        assert outcome["success"] is None and outcome["reported_success"] is None


def test_success_source_must_be_declared_by_the_caller():
    with pytest.raises(ValueError, match="success_source_invalid"):
        attempt_outcome({"success": True}, success_source="policy_self_report")
