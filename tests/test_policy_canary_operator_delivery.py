"""User omission remains an admitted diagnostic fact through private delivery."""
from copy import deepcopy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_policy_canary_result import validate_policy_canary_result
from blueprint_pipeline.task_evaluation_result_delivery import materialize_policy_canary_result_delivery
from blueprint_pipeline.task_evaluation_run_webapp_sync import build_task_evaluation_policy_canary_webapp_publication
from blueprint_pipeline.policy_canary_control_result_delivery import WARNINGS
from tests.test_task_evaluation_policy_canary_result_delivery import _result, _closure
from tests.test_task_evaluation_policy_canary_webapp_sync import _projection, _artifact


def _authority(contract):
    value = {
        "schema_version": "task_evaluation_diagnostic_control_omission_authority.v1",
        "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
        "authorized_by": "owner", "authorization_reference": "explicit-user-request",
        "omitted_controls": ["zero_action_negative", "deterministic_scripted_positive"],
        "source_task_success_contract_digest": "sha256:" + "a" * 64,
        "result_task_success_contract_digest": contract["contract_digest"],
        "task_scoring_criteria_changed": False, "qualified_comparison_permitted": False,
    }
    value["authority_digest"] = canonical_digest(value, digest_field="authority_digest")
    return value


@pytest.mark.parametrize("fault", [None, "contract", "qualification", "controls"])
def test_delivery_seals_original_omission_authority_and_refuses_conflicts(tmp_path, fault):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    result = _result(evidence)
    authority = _authority(result["task_success_contract"])
    if fault == "contract":
        authority["result_task_success_contract_digest"] = "sha256:" + "f" * 64
    if fault == "qualification":
        authority["qualified_comparison_permitted"] = True
    if fault == "controls":
        result.update(controls=[], control_episode_count=0)
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    original = deepcopy(result)
    kwargs = dict(run_root=tmp_path, run_id="scene-839873-canary-1", result_status="blocked",
        session_result=result, evidence_root=evidence, control_omission_authority=authority,
        closure_records={name: _closure(tmp_path / f"{name}.json", flag=flag)
            for name, flag in (("billing", "official_billing_sealed"), ("teardown", "teardown_completed"),
                               ("provider_zero", "provider_zero_verified"))})
    if fault:
        with pytest.raises(ValueError, match="control_omission"):
            materialize_policy_canary_result_delivery(**kwargs)
        return
    delivery = materialize_policy_canary_result_delivery(**kwargs)
    assert result == original
    assert delivery["scene_controls_status"] == "controls_omitted_by_user"
    omission = delivery["control_omission"]
    assert omission["authority_digest"] == authority["authority_digest"]
    assert omission["qualified_comparison_permitted"] is False
    assert any(row["artifact_id"] == omission["artifact"]["artifact_id"] for row in delivery["artifacts"])
    assert "controls" not in delivery


@pytest.mark.parametrize("fault", [None, "missing", "strict", "count"])
def test_projection_omission_cannot_claim_verified_controls(fault):
    _, value = _projection()
    value.update(scene_controls_status="controls_omitted_by_user", warning=WARNINGS["controls_omitted_by_user"],
        control_omission={"authority_digest": "sha256:" + "b" * 64,
            "task_success_contract_digest": value["task_success_contract_digest"],
            "qualified_comparison_permitted": False, "artifact": _artifact("c", "omission")})
    value["counts"]["diagnostic_control_rollout_count"] = 0
    if fault == "missing":
        value.pop("control_omission")
    if fault == "strict":
        contract = value["task_success_contract"]
        contract["criteria"]["controls"] = {"mode": "required_per_cell",
            "control_ids": ["zero_action_negative", "deterministic_scripted_positive"]}
        contract["contract_digest"] = cross_runtime_canonical_digest(contract, digest_field="contract_digest")
        value["task_success_contract_digest"] = contract["contract_digest"]
        value["control_omission"]["task_success_contract_digest"] = contract["contract_digest"]
    if fault == "count":
        value["counts"]["diagnostic_control_rollout_count"] = 20
    value["projection_digest"] = cross_runtime_canonical_digest(value, digest_field="projection_digest")
    if fault:
        with pytest.raises(ValueError, match="omission"):
            validate_policy_canary_result(value)
    else:
        assert validate_policy_canary_result(value)["counts"]["diagnostic_control_rollout_count"] == 0


def test_operator_publication_requires_both_registration_and_plan():
    delivery, value = _projection()
    kwargs = dict(capture_session_id="capture-1", intake_id="intake-1", run_id=value["run_id"],
        request_digest=value["request_digest"], configuration_digest=value["configuration_digest"],
        result_status="blocked", result_delivery=delivery, policy_canary_result=value)
    with pytest.raises(ValueError, match="operator_publication_binding_missing"):
        build_task_evaluation_policy_canary_webapp_publication(**kwargs, plan_digest="sha256:" + "d" * 64)
    publication = build_task_evaluation_policy_canary_webapp_publication(**kwargs,
        plan_digest="sha256:" + "d" * 64, operator_registration_digest="sha256:" + "e" * 64)
    assert publication["plan_digest"] == "sha256:" + "d" * 64
    assert publication["operator_registration_digest"] == "sha256:" + "e" * 64
