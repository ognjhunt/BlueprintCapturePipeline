"""Regression tests for the layered, fail-closed success-claim contracts.

Guards against the audited false-positive families:
- camera motion passing as robot task success,
- visible arm presence passing without reach/contact evidence,
- provider runtime success being treated as task success,
- WAM/generated-media labels being treated as real-world proof,
- stale output artifacts being treated as current truth,
- status-string / stringly-typed verdicts coercing to success.

The artifact-shaped truth gates always run against the committed hermetic fixture
under tests/fixtures/kitchen_task_min. Real local kitchen task artifacts under
output/kitchen_task_scaling_preflight_* (faucet / stovetop / microwave / sink) are an
additional opt-in lane: set BLUEPRINT_TEST_LOCAL_ARTIFACTS=1 to sweep them too.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import success_claim_contracts as scc
from blueprint_pipeline.oscar_cosmos_wam_evaluator import _normalize_wam_success_labels
from blueprint_pipeline.proof_contracts import build_site_package_manifest
from blueprint_pipeline.robot_eval_execution import _simulator_attempts_from_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNNER = REPO_ROOT / "scripts" / "run_isaac_g1_kitchen_parity_eval.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("parity_runner_scc", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNNER = _load_runner()


# --------------------------------------------------------------------------------------
# Strict verdict coercion
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("1", None),
        ("true", None),
        ("passed", None),
        (None, None),
        (1.0, None),
        ({}, None),
    ],
)
def test_coerce_strict_success(value, expected) -> None:
    assert scc.coerce_strict_success(value) is expected


# --------------------------------------------------------------------------------------
# Artifact freshness — stale outputs are not current truth
# --------------------------------------------------------------------------------------


def test_freshness_fails_closed_without_any_signal() -> None:
    evidence = scc.build_artifact_freshness_evidence()
    assert evidence["status"] == "FAIL"
    assert "artifact_freshness_evidence_missing" in evidence["blockers"]


def test_freshness_rejects_run_id_mismatch() -> None:
    evidence = scc.build_artifact_freshness_evidence(
        artifact_run_id="run_old", current_run_id="run_new"
    )
    assert evidence["fresh"] is False
    assert any(b.startswith("stale_artifact_run_id_mismatch") for b in evidence["blockers"])


def test_freshness_rejects_artifact_predating_run() -> None:
    evidence = scc.build_artifact_freshness_evidence(
        artifact_generated_at="2026-07-01T00:00:00Z",
        run_started_at="2026-07-03T00:00:00Z",
    )
    assert evidence["fresh"] is False
    assert "stale_artifact_generated_before_run_start" in evidence["blockers"]


def test_freshness_passes_with_matching_run_id() -> None:
    evidence = scc.build_artifact_freshness_evidence(
        artifact_run_id="run_a", current_run_id="run_a"
    )
    assert evidence["fresh"] is True and evidence["status"] == "PASS"


def _fresh():
    return scc.build_artifact_freshness_evidence(
        artifact_run_id="run_a", current_run_id="run_a"
    )


# --------------------------------------------------------------------------------------
# Layer 1 — media validity
# --------------------------------------------------------------------------------------


def test_media_validity_fails_closed_on_missing_media() -> None:
    contract = scc.build_media_validity(media_present=False)
    assert contract["status"] == "FAIL"
    assert "media_missing" in contract["blockers"]


def test_media_validity_requires_freshness_evidence() -> None:
    contract = scc.build_media_validity(
        media_present=True, frame_count=10, decodable=True
    )
    assert contract["status"] == "FAIL"
    assert "media_freshness_evidence_missing" in contract["blockers"]


def test_media_validity_passes_and_is_not_task_success() -> None:
    contract = scc.build_media_validity(
        media_present=True, frame_count=10, decodable=True, freshness=_fresh()
    )
    assert contract["status"] == "PASS" and contract["media_valid"] is True
    assert "not task success" in contract["claim_boundary"]


# --------------------------------------------------------------------------------------
# Layer 2 — review task success
# --------------------------------------------------------------------------------------


def _valid_media():
    return scc.build_media_validity(
        media_present=True, frame_count=10, decodable=True, freshness=_fresh()
    )


def test_review_rejects_camera_only_motion_root_follow() -> None:
    contract = scc.build_review_task_success(
        media_validity=_valid_media(),
        reviewer_verdicts=[{"success": True, "reviewer": "vlm"}],
        camera_evidence={
            "robot_pov_camera_mode": "root_follow",
            "visible_embodied_robot_action_evidence": True,
        },
    )
    assert contract["status"] == "FAIL"
    assert "camera_motion_is_not_robot_task_evidence:root_follow" in contract["blockers"]


def test_review_rejects_missing_embodied_action_evidence() -> None:
    contract = scc.build_review_task_success(
        media_validity=_valid_media(),
        reviewer_verdicts=[{"success": True, "reviewer": "vlm"}],
        camera_evidence={"robot_pov_camera_mode": "robot_mounted_manipulation"},
    )
    assert "visible_embodied_robot_action_not_proven" in contract["blockers"]


def test_review_rejects_non_boolean_verdict() -> None:
    contract = scc.build_review_task_success(
        media_validity=_valid_media(),
        reviewer_verdicts=[{"success": "true", "reviewer": "vlm"}],
        camera_evidence={
            "robot_pov_camera_mode": "robot_mounted_manipulation",
            "visible_embodied_robot_action_evidence": True,
        },
    )
    assert "reviewer_verdict_not_strict_boolean:index_0" in contract["blockers"]


def test_review_success_is_never_real_world_proof() -> None:
    contract = scc.build_review_task_success(
        media_validity=_valid_media(),
        reviewer_verdicts=[{"success": True, "reviewer": "vlm"}],
        camera_evidence={
            "robot_pov_camera_mode": "robot_mounted_manipulation",
            "visible_embodied_robot_action_evidence": True,
        },
    )
    assert contract["status"] == "PASS"
    assert contract["real_world_proof"] is False


def test_review_fails_when_media_invalid() -> None:
    contract = scc.build_review_task_success(
        media_validity=scc.build_media_validity(media_present=False),
        reviewer_verdicts=[{"success": True, "reviewer": "vlm"}],
        camera_evidence={
            "robot_pov_camera_mode": "robot_mounted_manipulation",
            "visible_embodied_robot_action_evidence": True,
        },
    )
    assert "media_validity_not_passed" in contract["blockers"]


# --------------------------------------------------------------------------------------
# Layer 3 — task success contract; arm presence is not reach evidence
# --------------------------------------------------------------------------------------


def test_task_contract_requires_reach_evidence_when_affordance_declared() -> None:
    contract = scc.build_task_success_contract_result(
        task_metadata={
            "task_success_contract": "visible_reach_to_affordance",
            "affordance_object_ids": ["faucet", "handle"],
        },
        trace_task_success=True,
        reach_evidence=None,
    )
    assert contract["status"] == "FAIL"
    assert "visible_arm_presence_is_not_reach_evidence" in contract["blockers"]


def test_task_contract_rejects_non_boolean_trace_verdict() -> None:
    contract = scc.build_task_success_contract_result(
        task_metadata={"task_success_contract": "root_navigation_to_target"},
        trace_task_success="passed",
    )
    assert "trace_task_success_not_strict_boolean" in contract["blockers"]


def test_task_contract_passes_with_reach_evidence() -> None:
    contract = scc.build_task_success_contract_result(
        task_metadata={
            "task_success_contract": "visible_reach_to_affordance",
            "affordance_object_ids": ["faucet"],
        },
        trace_task_success=True,
        reach_evidence={"status": "PASS", "blockers": []},
    )
    assert contract["status"] == "PASS"


# --------------------------------------------------------------------------------------
# Layer 4 — simulator execution; provider runtime success is not task success
# --------------------------------------------------------------------------------------


def test_provider_completed_without_artifacts_fails() -> None:
    contract = scc.build_simulator_execution(
        provider_runtime_status="completed",
        output_artifacts_present=False,
        artifact_freshness=_fresh(),
    )
    assert contract["status"] == "FAIL"
    assert contract["provider_runtime_operational"] is True
    assert "simulator_output_artifacts_missing" in contract["blockers"]
    assert contract["provider_runtime_success_is_not_task_success"] is True


def test_simulator_execution_rejects_stale_artifacts() -> None:
    stale = scc.build_artifact_freshness_evidence(
        artifact_run_id="old", current_run_id="new"
    )
    contract = scc.build_simulator_execution(
        provider_runtime_status="completed",
        output_artifacts_present=True,
        artifact_freshness=stale,
        frames_rendered=100,
    )
    assert "simulator_output_artifacts_not_proven_fresh" in contract["blockers"]


def test_simulator_execution_passes_with_fresh_output() -> None:
    contract = scc.build_simulator_execution(
        provider_runtime_status="completed",
        output_artifacts_present=True,
        artifact_freshness=_fresh(),
        frames_rendered=100,
    )
    assert contract["status"] == "PASS"
    assert contract["simulator_execution_proven"] is True


# --------------------------------------------------------------------------------------
# Layer 5 — policy action execution; scripted motion is not policy proof
# --------------------------------------------------------------------------------------


def test_kinematic_teleport_is_not_policy_execution() -> None:
    contract = scc.build_policy_action_execution(
        action_source="kinematic_teleport",
        action_trace_present=True,
        actions_executed_in_simulator=True,
    )
    assert contract["status"] == "FAIL"
    assert "action_source_not_policy:kinematic_teleport" in contract["blockers"]


def test_policy_execution_passes_for_learned_policy() -> None:
    contract = scc.build_policy_action_execution(
        action_source="learned_policy",
        policy_id="groot_sonic",
        action_trace_present=True,
        actions_executed_in_simulator=True,
    )
    assert contract["status"] == "PASS"


def test_policy_execution_requires_executed_actions() -> None:
    contract = scc.build_policy_action_execution(
        action_source="learned_policy",
        policy_id="groot_sonic",
        action_trace_present=True,
        actions_executed_in_simulator=None,
    )
    assert "actions_not_proven_executed_in_simulator" in contract["blockers"]


# --------------------------------------------------------------------------------------
# Layer 6 — contact / state change; proximity and masks are not contact
# --------------------------------------------------------------------------------------

_STATE_CHANGE_TASK = {
    "task_success_contract": "visible_reach_to_affordance",
    "affordance_object_ids": ["door", "handle"],
    "success_state_change": {"object": "door", "property": "open_fraction"},
}


def test_mask_overlap_is_rejected_as_contact_proof() -> None:
    requirements = scc.derive_task_proof_requirements(_STATE_CHANGE_TASK)
    contract = scc.build_contact_state_change_proof(
        proof_requirements=requirements,
        contact_reports=[{"mask_overlap_only": True}],
    )
    assert contract["status"] == "FAIL"
    assert (
        "contact_report_rejected_mask_overlap_only_is_not_contact_proof"
        in contract["blockers"]
    )


def test_declared_state_change_requires_measurement() -> None:
    requirements = scc.derive_task_proof_requirements(_STATE_CHANGE_TASK)
    contract = scc.build_contact_state_change_proof(
        proof_requirements=requirements,
        contact_reports=[{"physics_contact_measured": True}],
    )
    # Contact alone does not satisfy a declared state change.
    assert contract["contact_state_change_proven"] is False


def test_unchanged_state_is_not_state_change() -> None:
    requirements = scc.derive_task_proof_requirements(_STATE_CHANGE_TASK)
    contract = scc.build_contact_state_change_proof(
        proof_requirements=requirements,
        state_change_measurement={
            "property": "open_fraction",
            "before": 0.0,
            "after": 0.0,
        },
    )
    assert "state_change_not_observed" in contract["blockers"]


def test_measured_state_change_passes() -> None:
    requirements = scc.derive_task_proof_requirements(_STATE_CHANGE_TASK)
    contract = scc.build_contact_state_change_proof(
        proof_requirements=requirements,
        state_change_measurement={
            "property": "open_fraction",
            "before": 0.0,
            "after": 0.8,
        },
    )
    assert contract["status"] == "PASS"
    assert contract["contact_state_change_proven"] is True


# --------------------------------------------------------------------------------------
# Layer 7 — physical readiness can never come from simulation layers
# --------------------------------------------------------------------------------------


def test_physical_readiness_fails_without_real_robot_evidence() -> None:
    contract = scc.build_physical_readiness()
    assert contract["status"] == "FAIL"
    assert "physical_robot_execution_evidence_missing" in contract["blockers"]
    assert contract["simulation_evidence_cannot_upgrade_this_layer"] is True


def test_physical_readiness_requires_named_approval() -> None:
    contract = scc.build_physical_readiness(
        real_robot_execution_evidence={
            "physical_robot_executed": True,
            "run_manifest_uri": "gs://runs/field_run.json",
        },
        deployment_approval={"approved": True},
    )
    assert "deployment_approver_missing" in contract["blockers"]


def test_physical_readiness_passes_with_full_evidence() -> None:
    contract = scc.build_physical_readiness(
        real_robot_execution_evidence={
            "physical_robot_executed": True,
            "run_manifest_uri": "gs://runs/field_run.json",
        },
        deployment_approval={"approved": True, "approver": "site_operator"},
    )
    assert contract["status"] == "PASS"


# --------------------------------------------------------------------------------------
# Composed ledger — monotone claim ladder
# --------------------------------------------------------------------------------------


def _passing_review():
    return scc.build_review_task_success(
        media_validity=_valid_media(),
        reviewer_verdicts=[{"success": True, "reviewer": "vlm"}],
        camera_evidence={
            "robot_pov_camera_mode": "robot_mounted_manipulation",
            "visible_embodied_robot_action_evidence": True,
        },
    )


def test_ledger_review_success_does_not_claim_simulator_success() -> None:
    ledger = scc.build_success_claim_ledger(
        task_metadata={"task_success_contract": "visible_reach_to_affordance"},
        media_validity=_valid_media(),
        review_task_success=_passing_review(),
    )
    assert ledger["highest_truthful_claim"] == "review_task_success"
    assert ledger["claims"]["simulator_task_success"] is False
    assert ledger["claims"]["physical_deployment_ready"] is False


def test_ledger_state_change_task_withholds_task_claim_without_proof() -> None:
    task = dict(_STATE_CHANGE_TASK)
    simulator = scc.build_simulator_execution(
        provider_runtime_status="completed",
        output_artifacts_present=True,
        artifact_freshness=_fresh(),
        frames_rendered=100,
    )
    contract = scc.build_task_success_contract_result(
        task_metadata=task,
        trace_task_success=True,
        reach_evidence={"status": "PASS", "blockers": []},
    )
    ledger = scc.build_success_claim_ledger(
        task_metadata=task,
        media_validity=_valid_media(),
        review_task_success=_passing_review(),
        task_success_contract=contract,
        simulator_execution=simulator,
    )
    # Task declares a door state change; without measured contact/state-change proof
    # the ledger must stay at review level.
    assert ledger["highest_truthful_claim"] == "review_task_success"
    assert (
        "task_declares_state_change_but_contact_state_change_not_proven"
        in ledger["blockers"]
    )


def test_ledger_never_reports_physical_readiness_from_simulation() -> None:
    task = {"task_success_contract": "root_navigation_to_target"}
    simulator = scc.build_simulator_execution(
        provider_runtime_status="completed",
        output_artifacts_present=True,
        artifact_freshness=_fresh(),
        frames_rendered=100,
    )
    contract = scc.build_task_success_contract_result(
        task_metadata=task, trace_task_success=True
    )
    policy = scc.build_policy_action_execution(
        action_source="learned_policy",
        policy_id="p1",
        action_trace_present=True,
        actions_executed_in_simulator=True,
    )
    ledger = scc.build_success_claim_ledger(
        task_metadata=task,
        media_validity=_valid_media(),
        review_task_success=_passing_review(),
        task_success_contract=contract,
        simulator_execution=simulator,
        policy_action_execution=policy,
    )
    assert ledger["highest_truthful_claim"] == "policy_task_success"
    assert ledger["claims"]["physical_deployment_ready"] is False


# --------------------------------------------------------------------------------------
# Runner ledger — kinematic preview never claims policy task success
# --------------------------------------------------------------------------------------


def _runner_review_evidence(passing: bool) -> dict:
    return {
        "review_task_success": passing,
        "blockers": [] if passing else ["visible_embodied_robot_action_not_proven"],
    }


def test_runner_ledger_kinematic_lane_is_not_policy_success() -> None:
    outcome = {
        "frames_captured": 48,
        "task_success": True,
        "task_success_contract": "visible_reach_to_affordance",
        "task_status": "passed",
    }
    ledger = RUNNER._scenario_success_claim_ledger(
        {"scenario_id": "s1"}, outcome, _runner_review_evidence(True)
    )
    assert ledger["claims"]["policy_task_success"] is False
    assert any(
        b.startswith("policy_action_execution:action_source_not_policy")
        for b in ledger["blockers"]
    )
    assert ledger["highest_truthful_claim"] == "simulator_task_success"
    assert ledger["claims"]["physical_deployment_ready"] is False


def test_runner_ledger_camera_only_motion_stays_at_media_valid() -> None:
    outcome = {
        "frames_captured": 48,
        "task_success": True,
        "task_success_contract": "root_navigation_to_target",
        "task_status": "passed",
    }
    ledger = RUNNER._scenario_success_claim_ledger(
        {"scenario_id": "s1"}, outcome, _runner_review_evidence(False)
    )
    assert ledger["claims"]["review_task_success"] is False
    # Trace/simulator layers may still pass, but review-grade claims must not.
    assert ledger["claims"]["media_valid"] is True


def test_runner_ledger_declared_state_change_blocks_simulator_claim() -> None:
    scenario = {
        "scenario_id": "s1",
        "success_state_change": {"object": "door", "property": "open_fraction"},
    }
    outcome = {
        "frames_captured": 48,
        "task_success": True,
        "task_success_contract": "visible_reach_to_affordance",
        "task_status": "passed",
    }
    ledger = RUNNER._scenario_success_claim_ledger(
        scenario, outcome, _runner_review_evidence(True)
    )
    assert ledger["claims"]["simulator_task_success"] is False
    assert (
        "task_declares_state_change_but_contact_state_change_not_proven"
        in ledger["blockers"]
    )
    assert ledger["highest_truthful_claim"] == "review_task_success"


def test_runner_build_result_attaches_ledger_rows() -> None:
    scenarios = [{"scenario_id": "s1"}]
    outcomes = [
        {
            "frames_captured": 10,
            "task_success": False,
            "task_success_contract": "root_navigation_to_target",
            "task_status": "failed_task_criteria",
        }
    ]
    result = RUNNER.build_result(
        scenarios=scenarios,
        outcomes=outcomes,
        policy_id="g1_walkto_v1",
        kitchen_usd="kitchen.usd",
        g1_usd=None,
        blockers=[],
    )
    row = result["scenarios"][0]
    assert row["success_claim_ledger"]["schema_version"] == "success_claim_ledger.v1"
    assert "success_claim_summary" in result
    assert result["success_claim_summary"]["physical_deployment_ready_count"] == 0


# --------------------------------------------------------------------------------------
# WAM success labels — generated-media review is never authoritative without a verdict
# --------------------------------------------------------------------------------------


def _wam_rollout():
    return [{"rollout_id": "r1", "policy_id": "p1", "generated_video_path": "v.mp4"}]


def test_wam_label_media_validity_alone_is_not_authoritative() -> None:
    payload = {
        "status": "completed",
        "labels": [{"rollout_id": "r1", "success": None}],
    }
    labels = _normalize_wam_success_labels(
        command_payload=payload,
        rollouts=_wam_rollout(),
        generated_at="2026-07-04T00:00:00Z",
        visual_smoke_status="passed",
        visual_rollout_useful=True,
    )
    row = labels["labels"][0]
    assert row["authoritative_task_success_label"] is False
    assert row["reviewer_verdict_strict_boolean"] is False
    assert labels["review_grade_success_labels"] is False
    assert "wam_success_label_verdict_not_strict_boolean" in labels["blockers"]


def test_wam_label_boolean_verdict_with_valid_media_is_review_grade() -> None:
    payload = {
        "status": "completed",
        "labels": [{"rollout_id": "r1", "success": True}],
    }
    labels = _normalize_wam_success_labels(
        command_payload=payload,
        rollouts=_wam_rollout(),
        generated_at="2026-07-04T00:00:00Z",
        visual_smoke_status="passed",
        visual_rollout_useful=True,
    )
    row = labels["labels"][0]
    assert row["authoritative_task_success_label"] is True
    assert row["review_task_success"] is True
    assert labels["review_grade_success_labels"] is True
    # Review-grade never upgrades to real-world proof.
    assert labels["claim_boundary"][
        "success_label_is_from_generated_video_not_physical_robot"
    ] is True


def test_wam_label_degraded_media_blocks_review_grade() -> None:
    payload = {
        "status": "completed",
        "labels": [{"rollout_id": "r1", "success": True}],
    }
    labels = _normalize_wam_success_labels(
        command_payload=payload,
        rollouts=_wam_rollout(),
        generated_at="2026-07-04T00:00:00Z",
        visual_smoke_status="failed",
        visual_rollout_useful=False,
    )
    assert labels["review_grade_success_labels"] is False
    assert labels["labels"][0]["authoritative_task_success_label"] is False


# --------------------------------------------------------------------------------------
# Simulator attempt ingest — status strings never become task success
# --------------------------------------------------------------------------------------


def test_completed_status_without_verdict_fails_closed() -> None:
    attempts = _simulator_attempts_from_payload(
        payload={"episodes": [{"status": "completed"}]},
        simulator="mujoco",
        generated_at="2026-07-04T00:00:00Z",
    )
    attempt = attempts[0]
    assert attempt["task_success"] is False
    assert attempt["task_success_explicit"] is False
    assert "task_success_not_reported_failing_closed" in attempt["failure_mode_ids"]


def test_explicit_task_success_still_accepted() -> None:
    attempts = _simulator_attempts_from_payload(
        payload={"episodes": [{"status": "completed", "task_success": True}]},
        simulator="mujoco",
        generated_at="2026-07-04T00:00:00Z",
    )
    attempt = attempts[0]
    assert attempt["task_success"] is True
    assert attempt["task_success_explicit"] is True


# --------------------------------------------------------------------------------------
# Site package manifest — spec pointer is not runtime launchability
# --------------------------------------------------------------------------------------


def _site_package_kwargs(**overrides):
    kwargs = dict(
        scene_id="scene",
        capture_id="cap",
        site_submission_id=None,
        opportunity_id=None,
        evaluation_prep_manifest={"canonical_package_status": "registered"},
        site_world_spec={"canonical_package_status": "registered"},
        site_world_registration={},
        site_world_health={"launchable": True},
        launchable_export_bundle={"status": "ready"},
        site_identity={"site_id": "site-1"},
        adjacent_systems=None,
        rights_review={"status": "cleared"},
    )
    kwargs.update(overrides)
    return kwargs


def test_site_package_blocks_when_export_bundle_not_ready() -> None:
    manifest = build_site_package_manifest(
        **_site_package_kwargs(launchable_export_bundle={"status": "missing"})
    )
    assert manifest["status"] == "blocked"
    assert "launchable_export_not_ready:missing" in manifest["blockers"]


def test_site_package_blocks_when_runtime_not_launchable() -> None:
    manifest = build_site_package_manifest(
        **_site_package_kwargs(site_world_health={"launchable": False})
    )
    assert manifest["status"] == "blocked"
    assert "site_world_runtime_not_launchable" in manifest["blockers"]


def test_site_package_ready_with_launchable_runtime() -> None:
    manifest = build_site_package_manifest(**_site_package_kwargs())
    assert manifest["status"] == "ready"


# --------------------------------------------------------------------------------------
# Kitchen task artifacts: hermetic committed fixture always runs; local output/
# artifacts are an additional opt-in lane (BLUEPRINT_TEST_LOCAL_ARTIFACTS=1)
# --------------------------------------------------------------------------------------

_FIXTURE_REQUESTS = sorted(
    (REPO_ROOT / "tests" / "fixtures" / "kitchen_task_min").glob(
        "kitchen_task_scaling_request.json"
    )
)

_LOCAL_ARTIFACT_REQUESTS = (
    sorted(
        (REPO_ROOT / "output").glob(
            "kitchen_task_scaling_preflight_*/kitchen_task_scaling_request.json"
        )
    )
    if os.environ.get("BLUEPRINT_TEST_LOCAL_ARTIFACTS") == "1"
    and (REPO_ROOT / "output").is_dir()
    else []
)

_ARTIFACT_REQUESTS = _FIXTURE_REQUESTS + _LOCAL_ARTIFACT_REQUESTS


def test_hermetic_kitchen_fixture_present() -> None:
    # The committed fixture is what keeps the artifact-shaped truth gates below
    # executing in every checkout; if it disappears they would silently degrade
    # to an empty parametrization.
    assert _FIXTURE_REQUESTS, "tests/fixtures/kitchen_task_min fixture is missing"
    fixture_manifest = (
        _FIXTURE_REQUESTS[0].parent / "kitchen_task_scaling_preflight_manifest.json"
    )
    assert fixture_manifest.is_file()
    assert (
        json.loads(fixture_manifest.read_text())["local_preflight_status"] == "passed"
    )


@pytest.mark.parametrize(
    "request_path",
    _ARTIFACT_REQUESTS,
    ids=[p.parent.name for p in _ARTIFACT_REQUESTS],
)
def test_real_kitchen_task_metadata_derives_generic_requirements(request_path) -> None:
    payload = json.loads(request_path.read_text())
    for spec in payload.get("scenarios") or []:
        requirements = scc.derive_task_proof_requirements(spec)
        declares_targets = bool(
            spec.get("affordance_object_ids") or spec.get("target_object_ids")
        )
        # The requirement must be driven purely by the task metadata: any scenario
        # that declares target/affordance objects requires reach evidence, so visible
        # arm presence alone can never pass those tasks.
        assert requirements["requires_reach_to_affordance"] is declares_targets
        if not declares_targets:
            continue
        contract = scc.build_task_success_contract_result(
            task_metadata={
                "task_success_contract": "visible_reach_to_affordance",
                **spec,
            },
            trace_task_success=True,
            reach_evidence=None,
        )
        assert contract["status"] == "FAIL"
        assert "visible_arm_presence_is_not_reach_evidence" in contract["blockers"]


def test_real_preflight_pass_never_reaches_task_success_claim() -> None:
    manifest_paths = [
        p.parent / "kitchen_task_scaling_preflight_manifest.json"
        for p in _ARTIFACT_REQUESTS
    ]
    manifest_paths = [p for p in manifest_paths if p.is_file()]
    assert manifest_paths, "request artifacts exist but no preflight manifests found"
    passed_manifests = 0
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("local_preflight_status") != "passed":
            continue
        passed_manifests += 1
        # A passed local preflight provides no review/simulator/contact evidence, so
        # the ledger must stay at no_claim for every task it covers.
        for task in manifest.get("tasks") or []:
            ledger = scc.build_success_claim_ledger(
                task_metadata={
                    "task_success_contract": "visible_reach_to_affordance",
                    "affordance_object_ids": ["preflight_target"],
                }
            )
            assert ledger["highest_truthful_claim"] == "no_claim"
            assert ledger["claims"]["simulator_task_success"] is False
    assert passed_manifests, "no passed preflight manifest was exercised"
