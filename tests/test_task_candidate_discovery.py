from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from blueprint_pipeline.task_candidate_discovery import (
    TaskCandidateContractError,
    build_task_candidate_discovery,
    compile_approved_task_decision_request,
    record_customer_supplied_task,
    record_task_candidate_decision,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def _beta_fixture(case_id: str) -> dict:
    matrix = json.loads(
        (
            Path(__file__).parents[1]
            / "tests"
            / "fixtures"
            / "design_partner_beta_v1"
            / "fixture_matrix.json"
        ).read_text(encoding="utf-8")
    )
    return next(row for row in matrix["cases"] if row["case_id"] == case_id)


def _source_capture() -> dict:
    return {
        "intake_id": "intake-1",
        "capture_digest": SHA_A,
        "capture_authority_profile": "iphone_arkit_lidar",
    }


def _scene_analysis() -> dict:
    return {
        "observed_site_facts": [
            {
                "fact_id": "fact-tote",
                "description": "A rigid blue tote is directly visible on the table.",
                "confidence": 0.98,
                "supporting_frames": ["frame-20", "frame-10"],
                "supporting_3d_regions": ["region-table"],
            }
        ],
        "inferred_objects_and_affordances": [
            {
                "inference_id": "affordance-tote",
                "description": "The tote may be graspable from its rim.",
                "confidence": 0.75,
                "supporting_frames": ["frame-20"],
                "supporting_3d_regions": ["region-tote"],
            }
        ],
        "unsupported_or_occluded_regions": [
            {
                "region_id": "region-tote-rear",
                "description": "The rear of the tote is occluded.",
                "confidence": 1.0,
                "supporting_frames": ["frame-20"],
                "supporting_3d_regions": ["region-tote-rear"],
            }
        ],
        "hazards": [],
        "privacy_sensitive_areas": [],
    }


def _proposal() -> dict:
    return {
        "description": "Move the blue tote from the table to the floor target.",
        "observed_objects": [
            {
                "object_id": "tote-1",
                "label": "blue tote",
                "observation_fact_ids": ["fact-tote"],
            }
        ],
        "target_regions": [{"region_id": "region-floor-target", "label": "floor target"}],
        "required_robot_capabilities": ["rigid-object grasp", "tabletop reach"],
        "likely_task_family": "rigid_object_pick_place",
        "proposed_measurable_success_condition": {
            "metric": "final_object_center_distance",
            "operator": "<=",
            "threshold": 0.05,
            "units": "m",
        },
        "required_site_reset": "Return the tote to the marked table location.",
        "supporting_frames": ["frame-20", "frame-10"],
        "supporting_3d_regions": ["region-table", "region-floor-target"],
        "confidence": 0.99,
        "coverage": {"task_object": 0.8, "target_region": 0.7},
        "assumptions": ["The tote is movable."],
        "missing_evidence": ["Rear grasp surface is occluded."],
        "prohibited_claims": ["physical_task_success", "deployment_readiness"],
        "estimated_evaluation_cost_usd": 2.5,
        "expected_customer_value": {"source": "customer", "description": "reduce review time"},
    }


def _proposal_method() -> dict:
    return {
        "method_id": "local-task-proposer",
        "version": "1",
        "implementation_digest": SHA_C,
        "proposer_identity": "provider:model-a",
        "origin": "model_provider",
    }


def _discovery() -> dict:
    return build_task_candidate_discovery(
        discovery_id="discovery-1",
        source_capture=_source_capture(),
        capture_qa_report_digest=SHA_B,
        scene_analysis=_scene_analysis(),
        candidate_proposals=[_proposal()],
        proposal_method=_proposal_method(),
    )


def _claim() -> dict:
    return {
        "claim_id": "reach",
        "claim_type": "reachability",
        "subject": "robot-1:tote-1",
        "measurable_threshold": {"operator": ">=", "value": 0.9, "units": "ratio"},
        "false_safe_consequence": "moderate",
        "acceptable_false_safe_risk": 0.05,
        "desired_confidence_or_coverage": {"minimum_coverage": 0.9},
        "permitted_abstention_behavior": {"allowed": True},
        "task_family": "rigid_object_pick_place",
    }


def _approve(discovery: dict) -> tuple[dict, dict]:
    decision, approved = record_task_candidate_decision(
        discovery,
        task_candidate_id=discovery["task_candidates"][0]["task_candidate_id"],
        action="approve",
        actor={"role": "customer", "identity": "customer:user-1"},
        idempotency_key="approval-1",
        rationale="This is the task I want evaluated.",
    )
    assert approved is not None
    return decision, approved


def test_discovery_is_deterministic_and_keeps_observation_and_inference_separate() -> None:
    fixture = _beta_fixture("inferred_task_awaiting_approval")
    first = _discovery()
    second = _discovery()

    assert first == second
    assert first["approval_state"] == "task_approval_required"
    assert first["approval_state"] == fixture["approval_state"]
    assert fixture["decision_evidence_request_allowed"] is False
    candidate = first["task_candidates"][0]
    assert candidate["confidence"] == 0.99
    assert candidate["approval_status"] == "approval_required"
    assert first["scene_analysis"]["observed_site_facts"][0]["observation_status"] == "directly_observed"
    assert (
        first["scene_analysis"]["inferred_objects_and_affordances"][0]["observation_status"]
        == "inferred"
    )
    assert first["claim_boundaries"]["candidate_is_customer_intent"] is False


@pytest.mark.parametrize("action", ["reject", "request_more_capture"])
def test_nonapproval_decisions_never_emit_an_approved_task(action: str) -> None:
    discovery = _discovery()
    decision, approved = record_task_candidate_decision(
        discovery,
        task_candidate_id=discovery["task_candidates"][0]["task_candidate_id"],
        action=action,
        actor={"role": "operator", "identity": "operator:1"},
        idempotency_key=f"decision-{action}",
        rationale="The customer requested a different task or more coverage.",
    )

    assert decision["action"] == action
    assert approved is None


def test_edit_and_approve_binds_the_exact_customer_definition() -> None:
    discovery = _discovery()
    edited_task = {
        "description": "Move the tote into the box until its rim is below the box rim.",
        "task_family": "rigid_object_pick_place",
        "measurable_success_conditions": [
            {"metric": "rim_clearance", "operator": ">=", "threshold": 0.01, "units": "m"}
        ],
        "reset_contract": {"instructions": "Return tote to the table marker."},
        "task_objects": [{"object_id": "tote-1"}],
        "target_regions": [{"region_id": "box-1"}],
    }

    decision, approved = record_task_candidate_decision(
        discovery,
        task_candidate_id=discovery["task_candidates"][0]["task_candidate_id"],
        action="edit_and_approve",
        actor={"role": "customer", "identity": "customer:user-1"},
        idempotency_key="approval-edited-1",
        rationale="Use the box rim as the measurable threshold.",
        edited_task=edited_task,
    )

    assert approved is not None
    assert approved["intent_source"] == "customer_edited_candidate"
    assert approved["task"] == edited_task
    assert approved["approval_decision_digest"] == decision["decision_digest"]
    assert approved["discovery_digest"] == discovery["discovery_digest"]
    assert approved["prohibited_evaluator_identities"] == ["provider:model-a"]


def test_operator_approval_is_not_serialized_as_customer_intent() -> None:
    discovery = _discovery()
    _, approved = record_task_candidate_decision(
        discovery,
        task_candidate_id=discovery["task_candidates"][0]["task_candidate_id"],
        action="approve",
        actor={"role": "operator", "identity": "operator:17"},
        idempotency_key="operator-approval-1",
        rationale="Approve this bounded exploratory proxy task.",
    )

    assert approved is not None
    assert approved["intent_source"] == "operator_approved_candidate"
    assert approved["approval_actor"]["role"] == "operator"


def test_stale_or_tampered_discovery_is_rejected() -> None:
    discovery = _discovery()
    discovery["task_candidates"][0]["description"] = "Changed without a successor artifact"

    with pytest.raises(TaskCandidateContractError, match="candidate_digest:mismatch"):
        _approve(discovery)


def test_discovery_rejects_secret_bearing_proposal_metadata() -> None:
    proposal = _proposal()
    proposal["expected_customer_value"]["provider_api_key"] = "must-not-persist"

    with pytest.raises(TaskCandidateContractError, match="secret_value_forbidden"):
        build_task_candidate_discovery(
            discovery_id="discovery-secret",
            source_capture=_source_capture(),
            capture_qa_report_digest=SHA_B,
            scene_analysis=_scene_analysis(),
            candidate_proposals=[proposal],
            proposal_method=_proposal_method(),
        )


def test_candidate_observed_objects_must_bind_direct_observation_facts() -> None:
    proposal = _proposal()
    proposal["observed_objects"][0]["observation_fact_ids"] = ["inference-only"]

    with pytest.raises(TaskCandidateContractError, match="observation_fact_ids:unknown"):
        build_task_candidate_discovery(
            discovery_id="discovery-ungrounded",
            source_capture=_source_capture(),
            capture_qa_report_digest=SHA_B,
            scene_analysis=_scene_analysis(),
            candidate_proposals=[proposal],
            proposal_method=_proposal_method(),
        )


def test_customer_supplied_task_requires_explicit_thresholds_and_units() -> None:
    fixture = _beta_fixture("explicit_customer_task")
    task = {
        "description": "Place the tote inside the marked box.",
        "task_family": "rigid_object_pick_place",
        "measurable_success_conditions": [
            {"metric": "containment", "operator": "==", "threshold": True, "units": "boolean"}
        ],
        "reset_contract": {"instructions": "Return tote to table."},
    }
    receipt, approved = record_customer_supplied_task(
        source_capture=_source_capture(),
        task=task,
        actor={"role": "customer", "identity": "customer:user-1"},
        idempotency_key="customer-task-1",
    )

    assert receipt["schema_version"] == "customer_supplied_task_receipt.v1"
    assert approved["intent_source"] == "customer_supplied"
    assert approved["intent_source"] == fixture["intent_source"]
    assert fixture["thresholds_and_units_present"] is True
    assert fixture["decision_evidence_request_allowed"] is True
    assert approved["task"] == task

    invalid = copy.deepcopy(task)
    del invalid["measurable_success_conditions"][0]["units"]
    with pytest.raises(TaskCandidateContractError, match="units:missing"):
        record_customer_supplied_task(
            source_capture=_source_capture(),
            task=invalid,
            actor={"role": "customer", "identity": "customer:user-1"},
            idempotency_key="customer-task-2",
        )


def test_only_approved_task_compiles_to_provider_neutral_request_and_blocks_self_grading() -> None:
    _, approved = _approve(_discovery())
    testbed = {
        "testbed_id": "testbed-1",
        "version": "1",
        "testbed_digest": SHA_C,
        "approved_task_definition": {"digest": approved["approved_task_digest"]},
    }
    kwargs = {
        "testbed": testbed,
        "request_id": "request-1",
        "decision_id": "decision-1",
        "candidates": [{"robot_id": "robot-1"}],
        "claims": [_claim()],
        "budget": {"max_cost_usd": 5.0},
        "deadline": "2026-08-01T00:00:00Z",
        "permitted_evidence_methods": ["captured_real_observation", "analytic_geometry_kinematics"],
        "restrictions": {"external_processing_allowed": False},
        "requested_result_audience": "design_partner",
        "caller_identity": "service:task-approval",
        "idempotency_key": "request-1",
    }

    request = compile_approved_task_decision_request(approved, **kwargs)
    assert request["schema_version"] == "decision_evidence_request.v1"
    assert "selected_provider" not in request
    assert request["provenance"]["approved_task_digest"] == approved["approved_task_digest"]
    assert request["restrictions"]["prohibited_evaluator_identities"] == ["provider:model-a"]

    with pytest.raises(TaskCandidateContractError, match="self_grading_forbidden"):
        compile_approved_task_decision_request(
            approved,
            proposed_evaluator_identities=["provider:model-a"],
            **kwargs,
        )


def test_unapproved_or_wrong_testbed_binding_cannot_compile() -> None:
    _, approved = _approve(_discovery())
    unapproved = copy.deepcopy(approved)
    unapproved["approval_status"] = "approval_required"
    with pytest.raises(TaskCandidateContractError, match="approval_status"):
        compile_approved_task_decision_request(
            unapproved,
            testbed={
                "testbed_id": "testbed-1",
                "version": "1",
                "testbed_digest": SHA_C,
                "approved_task_definition": {"digest": approved["approved_task_digest"]},
            },
            request_id="request-1",
            decision_id="decision-1",
            candidates=[],
            claims=[_claim()],
            budget={"max_cost_usd": 5.0},
            deadline="2026-08-01T00:00:00Z",
            permitted_evidence_methods=["analytic_geometry_kinematics"],
            restrictions={"external_processing_allowed": False},
            requested_result_audience="design_partner",
            caller_identity="service:test",
            idempotency_key="request-1",
        )


def test_checked_in_schema_accepts_all_task_control_plane_artifacts() -> None:
    discovery = _discovery()
    decision, approved = _approve(discovery)
    customer_task = {
        "description": "Place the tote inside the marked box.",
        "task_family": "rigid_object_pick_place",
        "measurable_success_conditions": [
            {"metric": "containment", "operator": "==", "threshold": True, "units": "boolean"}
        ],
        "reset_contract": {"instructions": "Return tote to table."},
    }
    receipt, customer_approved = record_customer_supplied_task(
        source_capture=_source_capture(),
        task=customer_task,
        actor={"role": "customer", "identity": "customer:user-1"},
        idempotency_key="customer-task-schema",
    )
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "schemas"
        / "task_candidate_control_plane.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    for artifact in (discovery, decision, approved, receipt, customer_approved):
        validator.validate(artifact)
