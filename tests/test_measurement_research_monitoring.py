from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.measurement_research_admission import (
    advance_research_admission,
    apply_requalification_trigger,
    create_research_candidate,
)
from blueprint_pipeline.measurement_research_monitoring import (
    ResearchMonitoringError,
    build_release_observation,
    compile_research_monitoring_report,
)


def _observation(method_id: str, version: str) -> dict:
    return build_release_observation(
        method_id=method_id,
        observed_version=version,
        source_reference=f"https://example.test/releases/{method_id}",
        observed_on="2026-08-02",
    )


def _approval(role: str) -> dict:
    return {
        "role": role,
        "actor_id": f"fixture-{role}",
        "actor_type": "human",
        "approved": True,
        "signature_id": f"signature-{role}",
    }


def _admitted_r7(method_id: str) -> dict:
    record = create_research_candidate(
        candidate_id=f"admission-{method_id}",
        method_id=method_id,
        stage_data={
            "primary_sources": ["fixture://official-docs"],
            "method_identity": {"version": "1"},
            "claimed_scope": {"tasks": ["rigid_pick_place"]},
            "access_status": {"code": "available"},
        },
        approval=_approval("research_analyst"),
    )
    record = advance_research_admission(
        record,
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
            "qualification_decision": {"outcome": "approved_narrow_scope"},
            "agent_approved": False,
        },
        approvals=[
            _approval("research_lead"),
            _approval("domain_owner"),
            _approval("independent_reviewer"),
        ],
    )
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


def test_monitoring_classifies_unchanged_changed_and_new_methods() -> None:
    report = compile_research_monitoring_report(
        [
            _observation("mujoco-3", "3.11.0"),
            _observation("drake-1-55", "1.56.0"),
            _observation("brand-new-solver", "0.1.0"),
        ],
        observed_on="2026-08-02",
    )
    kinds = {row["method_id"]: row["kind"] for row in report["alerts"]}
    assert kinds["mujoco-3"] == "unchanged"
    assert kinds["drake-1-55"] == "version_changed"
    assert kinds["brand-new-solver"] == "new_method_discovered"
    assert report["regression_checks_required"] == ["drake-1-55"]
    drafts = report["r0_intake_drafts"]
    assert len(drafts) == 1
    assert drafts[0]["method_id"] == "brand-new-solver"
    assert drafts[0]["requires_human_research_analyst_approval"] is True
    assert drafts[0]["automation_may_create_catalog_entry"] is False
    assert report["human_action_required"] is True
    assert report["automation_approved_anything"] is False
    assert report["automation_advanced_any_stage"] is False
    assert report["monitoring_report_digest"].startswith("sha256:")


def test_version_change_proposes_but_never_applies_a_requalification_trigger() -> None:
    admitted = _admitted_r7("fixture-engine")
    report = compile_research_monitoring_report(
        [_observation("mujoco-3", "3.12.0")],
        observed_on="2026-08-02",
        admission_records=[admitted],
    )
    # mujoco-3 changed but the admitted record belongs to fixture-engine:
    # no cross-method trigger may be proposed.
    assert report["requalification_trigger_proposals"] == []

    catalog_backed = compile_research_monitoring_report(
        [
            build_release_observation(
                method_id="fixture-engine",
                observed_version="2",
                source_reference="https://example.test/releases/fixture-engine",
                observed_on="2026-08-02",
            )
        ],
        observed_on="2026-08-02",
        admission_records=[admitted],
    )
    # fixture-engine is not in the research catalog, so it surfaces as a new
    # method; admitted-method version changes only arise for catalog entries.
    assert catalog_backed["alerts"][0]["kind"] == "new_method_discovered"

    # Now a catalog method with an admitted record: simulate by reusing the
    # admitted chain under the catalog's mujoco-3 identity.
    mujoco_admitted = _admitted_r7("mujoco-3")
    proposal_report = compile_research_monitoring_report(
        [_observation("mujoco-3", "3.12.0")],
        observed_on="2026-08-02",
        admission_records=[mujoco_admitted],
    )
    proposals = proposal_report["requalification_trigger_proposals"]
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["trigger"] == "engine_solver_api_or_model_update"
    assert proposal["requires_human_approval"] is True
    assert proposal["automation_applied"] is False
    assert proposal["admission_record_digest"] == (mujoco_admitted["admission_record_digest"])
    # The record itself is untouched by monitoring; applying the proposal is a
    # separate human-approved act that suspends production eligibility.
    assert mujoco_admitted["production_eligible"] is True
    suspended = apply_requalification_trigger(
        mujoco_admitted,
        trigger=proposal["trigger"],
        detail=proposal["detail"],
        approval=_approval("catalog_owner"),
    )
    assert suspended["production_eligible"] is False


def test_stale_engine_profile_is_flagged_for_reverification() -> None:
    report = compile_research_monitoring_report(
        [_observation("newton-1-4", "1.5.0")],
        observed_on="2026-08-02",
    )
    alert = report["alerts"][0]
    assert alert["kind"] == "version_changed"
    assert "r1_source_reverification" in alert["actions"]

    profile_only = compile_research_monitoring_report(
        [
            build_release_observation(
                method_id="sapien-maniskill-3",
                observed_version="3.0.3",
                source_reference="https://example.test/releases/sapien",
                observed_on="2026-08-02",
            )
        ],
        observed_on="2026-08-02",
    )
    assert profile_only["alerts"][0]["kind"] == "unchanged"


def test_observation_and_report_inputs_fail_closed() -> None:
    with pytest.raises(ResearchMonitoringError, match="method_id_missing"):
        build_release_observation(
            method_id="",
            observed_version="1",
            source_reference="https://example.test",
            observed_on="2026-08-02",
        )
    with pytest.raises(ResearchMonitoringError, match="observed_on_missing"):
        compile_research_monitoring_report([], observed_on="")
    tampered = _admitted_r7("mujoco-3")
    tampered["production_eligible"] = True
    tampered["suspended"] = True
    report = compile_research_monitoring_report(
        [_observation("mujoco-3", "9.9.9")],
        observed_on="2026-08-02",
        admission_records=[tampered],
    )
    assert report["requalification_trigger_proposals"] == []
    assert any("admission_record_invalid" in row for row in report["input_errors"])


def test_monthly_workflow_is_read_only_bounded_and_archives_the_report() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/measurement-research-monitor.yml").read_text(
        encoding="utf-8"
    )
    assert 'cron: "23 9 1 * *"' in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "timeout-minutes: 45" in workflow
    assert "GITHUB_TOKEN: ${{ github.token }}" in workflow
    assert "scripts/measurement_research_monitor.py" in workflow
    assert "tests/test_task_site_measurement_routing.py" in workflow
    assert "scripts/bootstrap_measurement_chrono_development.py" in workflow
    assert "tests/test_measurement_deformation_granular_chrono_development_suite.py" in workflow
    assert "measurement-chrono-bootstrap.json" in workflow
    assert "actions/upload-artifact@ea165f8d" in workflow
    assert "retention-days: 90" in workflow
    assert "paid_resource_allocator" not in workflow
    assert "apply_requalification_trigger" not in workflow
