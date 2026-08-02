from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_research_admission import (
    MeasurementAdmissionError,
    admission_supports_production_route,
    advance_research_admission,
    apply_requalification_trigger,
    create_research_candidate,
    validate_research_admission_record,
)


def _approval(role: str, *, actor_type: str = "human") -> dict:
    return {
        "role": role,
        "actor_id": f"fixture-{role}",
        "actor_type": actor_type,
        "approved": True,
        "signature_id": f"signature-{role}",
    }


def _r0() -> dict:
    return create_research_candidate(
        candidate_id="candidate-fixture-engine",
        method_id="fixture-engine",
        stage_data={
            "primary_sources": ["fixture://official-docs"],
            "method_identity": {"version": "1", "commit": "abc"},
            "claimed_scope": {"tasks": ["rigid_pick_place"], "claim_ceiling": "C3"},
            "access_status": {"code": "available", "model": "not_applicable"},
        },
        approval=_approval("research_analyst"),
    )


def _r7(*, r6_outcome: str = "approved_narrow_scope", stop_at: str = "R7") -> dict:
    record = advance_research_admission(
        _r0(),
        target_stage="R1",
        stage_data={
            "source_verification": {"official_docs_checked": True},
            "code_access": {"repository_verified": True},
            "license_records": {"source_license": "fixture"},
            "vendor_claim_separation": {"vendor_results_status": "external_claim"},
        },
        approvals=[_approval("research_lead")],
    )
    record = advance_research_admission(
        record,
        target_stage="R2",
        stage_data={
            key: {"status": "accepted"}
            for key in (
                "commercial_use",
                "site_data_ownership",
                "retention",
                "training_use",
                "subprocessors",
                "offline_option",
                "export_rights",
                "output_portability",
                "termination_deletion",
            )
        },
        approvals=[_approval("legal_owner"), _approval("privacy_owner")],
    )
    record = advance_research_admission(
        record,
        target_stage="R3",
        stage_data={
            key: {"status": "verified"}
            for key in (
                "scene_robot_formats",
                "coordinate_units",
                "collider_material_path",
                "controller_action_adapter",
                "sensor_adapter",
                "headless_execution",
                "deterministic_replay",
                "logs_state_access",
                "engineering_burden",
            )
        },
        approvals=[_approval("platform_owner")],
    )
    record = advance_research_admission(
        record,
        target_stage="R4",
        stage_data={
            "frozen_benchmark_preregistration": {
                "task_site_classes": ["rigid_pick_place"],
                "development_split_hash": "sha256:" + "1" * 64,
                "qualification_split_hash": "sha256:" + "2" * 64,
                "robot_controller_digests": ["sha256:" + "3" * 64],
                "capture_bundle_hashes": ["sha256:" + "4" * 64],
                "metrics": ["false_negative_rate"],
                "acceptance_thresholds": {"max": 0.01},
                "comparison_methods": ["baseline"],
                "compute_budget": {"usd": 10},
                "failure_criteria": ["threshold_exceeded"],
                "statistical_method": "bootstrap",
                "claim_ceiling": "C3",
            },
            "heldout_labels_exposed": False,
        },
        approvals=[_approval("benchmark_owner"), _approval("independent_reviewer")],
    )
    record = advance_research_admission(
        record,
        target_stage="R5",
        stage_data={
            "heldout_evaluation": {
                "independent_execution": True,
                "hidden_case_hashes": ["sha256:" + "5" * 64],
                "physical_measurement_ids": ["physical-fixture"],
                "repeated_trial_count": 20,
                "confidence_intervals": {"level": 0.95},
                "harmful_false_negative_analysis": {"rate": 0.0},
                "retained_failure_ids": ["failure-fixture"],
                "clean_environment_rerun_id": "rerun-fixture",
                "qualification_split_hash": "sha256:" + "2" * 64,
            },
            "vendor_graded_qualification": False,
        },
        approvals=[_approval("benchmark_owner")],
    )
    record = advance_research_admission(
        record,
        target_stage="R6",
        stage_data={
            "qualification_decision": {"outcome": r6_outcome},
            "agent_approved": False,
        },
        approvals=[
            _approval("research_lead"),
            _approval("domain_owner"),
            _approval("independent_reviewer"),
        ],
    )
    if stop_at == "R6":
        return record
    return advance_research_admission(
        record,
        target_stage="R7",
        stage_data={
            "catalog_admission": {
                "method_version": "1",
                "capability_profile_digest": "sha256:" + "a" * 64,
                "scope_envelope": {"task_classes": ["rigid_pick_place"]},
                "qualification_ids": ["qualification-fixture"],
                "expiration_date": "2027-08-01",
                "claim_ceiling": "C3",
                "known_failure_modes": ["fixture_failure"],
                "required_site_evidence": ["validated_collider"],
                "prohibited_extrapolations": ["other_robot"],
            }
        },
        approvals=[_approval("catalog_owner"), _approval("independent_reviewer")],
    )


def test_admission_advances_sequentially_and_digest_binds_history() -> None:
    r0 = _r0()
    r1 = advance_research_admission(
        r0,
        target_stage="R1",
        stage_data={
            "source_verification": {"official_docs_checked": True},
            "code_access": {"repository_verified": True},
            "license_records": {"source_license": "fixture"},
            "vendor_claim_separation": {"vendor_results_status": "external_claim"},
        },
        approvals=[_approval("research_lead")],
    )
    assert r1["stage"] == "R1"
    assert r1["transition_history"][0]["predecessor_digest"] == r0["admission_record_digest"]
    assert r1["production_eligible"] is False


def test_agent_cannot_approve_and_stages_cannot_be_skipped() -> None:
    with pytest.raises(MeasurementAdmissionError, match="agent_approval_forbidden"):
        create_research_candidate(
            candidate_id="agent-candidate",
            method_id="agent-method",
            stage_data={
                "primary_sources": ["fixture://source"],
                "method_identity": {"version": "1"},
                "claimed_scope": {"task": "rigid"},
                "access_status": {"code": "available"},
            },
            approval=_approval("research_analyst", actor_type="agent"),
        )
    with pytest.raises(MeasurementAdmissionError, match="must_be_sequential"):
        advance_research_admission(
            _r0(), target_stage="R2", stage_data={}, approvals=[_approval("legal_owner")]
        )


def test_only_human_approved_r7_record_is_production_eligible() -> None:
    record = _r7()
    assert record["production_eligible"] is True
    assert record["completed_stage_data"]["R6"]["qualification_decision"]["outcome"] == (
        "approved_narrow_scope"
    )
    assert admission_supports_production_route(record) is True
    schema = json.loads(
        (
            Path(__file__).parents[1] / "docs/schemas/measurement_research_admission.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(record, schema)
    tampered = copy.deepcopy(record)
    tampered["stage_data"]["catalog_admission"]["claim_ceiling"] = "C8"
    with pytest.raises(MeasurementAdmissionError, match="digest_mismatch"):
        validate_research_admission_record(tampered)


def test_r7_cannot_be_constructed_without_the_full_evidence_chain() -> None:
    with pytest.raises(
        MeasurementAdmissionError,
        match="completed_stage_data_incomplete|transition_history_incomplete",
    ):
        validate_research_admission_record(
            {
                "schema_version": "measurement_research_admission.v1",
                "candidate_id": "shortcut-candidate",
                "method_id": "shortcut-method",
                "stage": "R7",
                "stage_data": {
                    "catalog_admission": {
                        "method_version": "1",
                        "capability_profile_digest": "sha256:" + "a" * 64,
                        "scope_envelope": {"task_classes": ["rigid_pick_place"]},
                        "qualification_ids": ["qualification-fixture"],
                        "expiration_date": "2027-08-01",
                        "claim_ceiling": "C3",
                        "known_failure_modes": ["fixture_failure"],
                        "required_site_evidence": ["validated_collider"],
                        "prohibited_extrapolations": ["other_robot"],
                    }
                },
                "completed_stage_data": {},
                "approvals": [_approval("catalog_owner"), _approval("independent_reviewer")],
                "production_eligible": True,
                "suspended": False,
                "requalification_events": [],
                "transition_history": [],
            }
        )


def test_failed_r6_outcome_blocks_catalog_admission() -> None:
    failed = _r7(r6_outcome="failed", stop_at="R6")
    assert failed["production_eligible"] is False
    with pytest.raises(
        MeasurementAdmissionError, match="admission_r6_outcome_blocks_catalog_admission"
    ):
        advance_research_admission(
            failed,
            target_stage="R7",
            stage_data={"catalog_admission": {}},
            approvals=[_approval("catalog_owner"), _approval("independent_reviewer")],
        )


def test_development_and_qualification_splits_must_differ() -> None:
    record = _r7(stop_at="R6")
    r3 = validate_research_admission_record(
        {
            **{key: value for key, value in record.items() if key != "admission_record_digest"},
            "stage": "R3",
            "stage_data": dict(record["completed_stage_data"]["R3"]),
            "completed_stage_data": {
                stage: data
                for stage, data in record["completed_stage_data"].items()
                if stage in {"R0", "R1", "R2"}
            },
            "approvals": [_approval("platform_owner")],
            "transition_history": record["transition_history"][:3],
            "production_eligible": False,
        }
    )
    leaky_prereg = {
        "frozen_benchmark_preregistration": {
            "task_site_classes": ["rigid_pick_place"],
            "development_split_hash": "sha256:" + "1" * 64,
            "qualification_split_hash": "sha256:" + "1" * 64,
            "robot_controller_digests": ["sha256:" + "3" * 64],
            "capture_bundle_hashes": ["sha256:" + "4" * 64],
            "metrics": ["false_negative_rate"],
            "acceptance_thresholds": {"max": 0.01},
            "comparison_methods": ["baseline"],
            "compute_budget": {"usd": 10},
            "failure_criteria": ["threshold_exceeded"],
            "statistical_method": "bootstrap",
            "claim_ceiling": "C3",
        },
        "heldout_labels_exposed": False,
    }
    with pytest.raises(
        MeasurementAdmissionError,
        match="R4_development_and_qualification_splits_must_differ",
    ):
        advance_research_admission(
            r3,
            target_stage="R4",
            stage_data=leaky_prereg,
            approvals=[_approval("benchmark_owner"), _approval("independent_reviewer")],
        )


def test_requalification_trigger_suspends_production_eligibility() -> None:
    record = _r7()
    suspended = apply_requalification_trigger(
        record,
        trigger="engine_solver_api_or_model_update",
        detail="engine 1 -> 2",
        approval=_approval("catalog_owner"),
    )
    assert suspended["suspended"] is True
    assert suspended["production_eligible"] is False
    assert admission_supports_production_route(suspended) is False
    with pytest.raises(
        MeasurementAdmissionError,
        match="admission_suspended_requires_new_qualification_decision",
    ):
        advance_research_admission(
            suspended,
            target_stage="R8",
            stage_data={"monitoring": {}},
            approvals=[_approval("catalog_owner")],
        )
    with pytest.raises(MeasurementAdmissionError, match="trigger_unknown"):
        apply_requalification_trigger(
            record,
            trigger="vibes_changed",
            detail="",
            approval=_approval("catalog_owner"),
        )
