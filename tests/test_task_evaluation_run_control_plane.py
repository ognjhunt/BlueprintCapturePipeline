from __future__ import annotations

import hmac
import json
from datetime import datetime, timezone
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.decision_evidence_contracts import (
    DecisionEvidenceRequest,
    EvidenceMethodProfile,
    MaintainedSiteTaskTestbed,
    QualificationRecord,
    canonical_digest,
)
from blueprint_pipeline.local_evidence_adapters import ANALYTIC_REACHABILITY_ADAPTER
from blueprint_pipeline.local_evidence_adapters import (
    SWEPT_AABB_COLLISION_SIMULATION_ADAPTER,
)
from blueprint_pipeline.live_pipeline_control_plane import CONTROL_PLANE_OUTPUT_PATH_ENV
from blueprint_pipeline.task_evaluation_run_control_plane import (
    TaskEvaluationRunControlPlaneError,
    authorize_task_evaluation_run,
    execute_and_aggregate_task_evaluation_run,
    prepare_task_evaluation_run,
)
from blueprint_pipeline.task_evaluation_run_webapp_sync import (
    TASK_EVALUATION_RUN_WEBAPP_SYNC_REQUIRED_ENV,
    build_task_evaluation_run_webapp_publication,
    sync_task_evaluation_run_to_webapp,
)
from blueprint_pipeline.task_evaluation_method_catalog import (
    TASK_EVALUATION_METHOD_CATALOG_PATH_ENV,
    validate_task_evaluation_method_catalog,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64


def _testbed() -> dict:
    return MaintainedSiteTaskTestbed.from_mapping({
        "schema_version": "maintained_site_task_testbed.v1",
        "testbed_id": "local-run-testbed",
        "version": "1",
        "predecessor_testbed_digest": None,
        "supersedes": [],
        "source_capture_bundles": [{"bundle_id": "capture-1", "version": "3", "digest": SHA_A}],
        "artifact_references": {
            "site_card": {"uri": "fixture://site", "digest": SHA_A},
            "task_cards": [{"uri": "fixture://task", "digest": SHA_A}],
            "scenario_cards": [{"uri": "fixture://scenario", "digest": SHA_A}],
            "eval_cards": [{"uri": "fixture://eval", "digest": SHA_A}],
            "evaluator": {"uri": "fixture://evaluator", "digest": SHA_B},
            "reset": {"uri": "fixture://reset", "digest": SHA_B},
        },
        "task_distribution": {"task_family": "rigid_object_pick_place", "tasks": ["move-item"]},
        "supported_condition_ranges": {"lighting_lux": [300, 600]},
        "robot_sensor_controller_bindings": {
            "embodiment": {
                "robot_id": "fixture-arm",
                "reach_envelope": {"minimum_m": 0.1, "maximum_m": 1.0},
            },
            "sensors": {"camera": "rgb-v1"},
            "controller_action_representation": {"type": "joint_position"},
            "selected_robot_placement": {
                "candidate_id": "base-1",
                "base_position_site_m": [0.0, 0.0, 0.0],
                "captured_coverage": 0.95,
                "calibration_uncertainty_m": 0.01,
                "method_qualification_status": "analytic_only",
            },
        },
        "governance": {
            "rights": "accepted",
            "consent": "accepted",
            "privacy": "cleared",
            "revocation": "version_invalidates_on_revocation",
            "allowed_uses": ["evaluation"],
        },
        "evidence_inventory": [{"evidence_id": "metric_geometry"}],
        "validation_envelope": {"robot_placement_digest": SHA_D},
        "target_regions": [{
            "region_id": "tote-1",
            "position_site_m": [0.6, 0.1, 0.7],
            "supporting_frames": ["frame-1"],
            "captured_coverage": 0.9,
        }],
        "known_unsupported_conditions": ["physical_task_success"],
        "invalidation_triggers": ["layout_change"],
        "physical_outcome_history_refs": [],
        "lifecycle_state": "active",
    }).to_mapping()


def _request(testbed: dict) -> dict:
    return DecisionEvidenceRequest.from_mapping({
        "schema_version": "decision_evidence_request.v1",
        "request_id": "request-local-1",
        "decision_id": "decision-local-1",
        "testbed_id": testbed["testbed_id"],
        "testbed_version": testbed["version"],
        "testbed_digest": testbed["testbed_digest"],
        "decision_question": "Can the exact robot analytically reach the tote region?",
        "candidates": [{"robot_id": "fixture-arm"}],
        "claims": [{
            "claim_id": "reach-tote",
            "claim_type": "reachability",
            "subject": "tote-1",
            "measurable_threshold": {"operator": "inside", "units": "m"},
            "false_safe_consequence": "moderate",
            "acceptable_false_safe_risk": 0.05,
            "desired_confidence_or_coverage": {
                "minimum_coverage": 0.8,
                "minimum_independent_methods": 1,
            },
            "permitted_abstention_behavior": {"allowed": True},
            "task_family": "rigid_object_pick_place",
            "site_domain_conditions": {"lighting_lux": [300, 600]},
            "embodiment": {"robot_id": "fixture-arm"},
            "sensors": {"camera": "rgb-v1"},
            "controller_action_representation": {"type": "joint_position"},
        }],
        "budget": {"max_cost_usd": 0.0, "max_latency_seconds": 10.0},
        "deadline": "2026-07-30T00:00:00Z",
        "available_physical_evidence": [],
        "permitted_evidence_methods": ["analytic_geometry_kinematics"],
        "restrictions": {"external_processing_allowed": False},
        "requested_result_audience": "design_partner",
        "provenance": {"caller_identity": "customer-approved-task"},
        "idempotency_key": "request-local-1",
    }).to_mapping()


def _profile() -> dict:
    return EvidenceMethodProfile.from_mapping({
        "schema_version": "evidence_method_profile.v1",
        "method_id": "local-analytic-reachability",
        "version": "1",
        "implementation_digest": SHA_B,
        "adapter_reference": ANALYTIC_REACHABILITY_ADAPTER,
        "method_family": "analytic_geometry_kinematics",
        "supported_claim_types": ["reachability"],
        "required_inputs": ["metric_geometry"],
        "applicability_envelope": {
            "testbed_ids": ["local-run-testbed"],
            "testbed_versions": ["1"],
            "task_families": ["rigid_object_pick_place"],
        },
        "calibration_evidence_references": ["fixture://calibration"],
        "authority_tier": 1,
        "proof_tier": "analytic_only",
        "correlation_group": "metric-scaffold",
        "shared_dependencies": ["capture-1"],
        "expected_cost_usd": 0.0,
        "expected_latency_seconds": 0.01,
        "reproducibility_level": "hermetic_local",
        "constraints": {"external_processing": False},
        "provider_availability": {"status": "available"},
        "failure_modes": ["metric_position_missing"],
        "abstention_modes": ["uncertain_boundary"],
        "disqualifying_conditions": [],
        "self_qualified": False,
    }).to_mapping()


def _qualification(profile: dict) -> dict:
    return QualificationRecord.from_mapping({
        "schema_version": "evidence_method_qualification.v1",
        "qualification_id": "qualification-local-reach-1",
        "method_id": profile["method_id"],
        "method_version": profile["version"],
        "method_profile_digest": profile["method_profile_digest"],
        "implementation_digest": profile["implementation_digest"],
        "claim_type": "reachability",
        "task_family": "rigid_object_pick_place",
        "site_domain_conditions": {"lighting_lux": [300, 600]},
        "embodiment": {"robot_id": "fixture-arm"},
        "sensors": {"camera": "rgb-v1"},
        "controller_action_representation": {"type": "joint_position"},
        "evaluator": {"evaluator_id": "independent-geometry-check", "version": "1"},
        "evaluator_digest": SHA_C,
        "predictions": [{"prediction_id": "prediction-1", "value": True}],
        "accepted_real_outcomes": [{"outcome_id": "measurement-anchor-1", "value": True}],
        "calibration_partition": "heldout",
        "confidence_intervals": {"level": 0.95, "lower": 0.9, "upper": 1.0},
        "coverage": 0.95,
        "abstention_rate": 0.05,
        "false_safe_rate": 0.01,
        "false_reject_rate": 0.02,
        "provenance": {"source": "independent-measurement-fixture"},
        "owner_evidence": [{"uri": "fixture://qualification", "digest": SHA_D}],
        "status": "qualified",
        "self_grading": False,
        "subject_provider_id": "blueprint-local-method",
        "evaluator_provider_id": "independent-geometry-check",
    }).to_mapping()


def _collision_run_inputs() -> tuple[dict, dict, dict, dict]:
    scene = {
        "schema_version": "collision_scene_aabb.v1",
        "source_capture_digest": SHA_A,
        "coordinate_frame": "site",
        "scale_status": "metric_verified",
        "generated_geometry": False,
        "primitives": [{
            "primitive_id": "table-obstacle",
            "object_id": "table-obstacle",
            "minimum_site_m": [0.4, -0.2, 0.0],
            "maximum_site_m": [0.6, 0.2, 0.8],
        }],
        "validation": {
            "status": "qualified",
            "independent_validation": True,
            "coverage": 0.95,
            "maximum_spatial_uncertainty_m": 0.01,
        },
    }
    scene["collision_scene_digest"] = canonical_digest(
        scene, digest_field="collision_scene_digest"
    )
    testbed_value = _testbed()
    testbed_value.pop("testbed_digest")
    testbed_value["evidence_inventory"].append({"evidence_id": "collision_scene"})
    testbed_value["validation_envelope"]["reconstruction_layers"] = {
        "physics_layer": [{
            "output": "collision_geometry",
            "result_id": "collision-result-1",
            "result_digest": SHA_D,
            "asset_references": {"collision_scene": scene},
            "generated_regions": [],
            "claim_ceiling": {"collision_geometry": True},
        }],
    }
    testbed = MaintainedSiteTaskTestbed.from_mapping(testbed_value).to_mapping()
    request_value = _request(testbed)
    request_value.pop("request_digest")
    request_value.update({
        "request_id": "request-collision-1",
        "decision_id": "decision-collision-1",
        "decision_question": "Is the modeled trajectory collision-free?",
        "claims": [{
            "claim_id": "collision-clearance",
            "claim_type": "collision_contact",
            "subject": {
                "trajectory_points_site_m": [[0.0, 0.8, 1.0], [1.0, 0.8, 1.0]],
                "swept_radius_m": 0.05,
                "excluded_collision_object_ids": ["item-1"],
            },
            "measurable_threshold": {"operator": "equals", "value": 0, "units": "modeled_contacts"},
            "false_safe_consequence": "moderate",
            "acceptable_false_safe_risk": 0.05,
            "desired_confidence_or_coverage": {
                "minimum_coverage": 0.8,
                "minimum_independent_methods": 1,
            },
            "permitted_abstention_behavior": {"allowed": True},
            "task_family": "rigid_object_pick_place",
            "site_domain_conditions": {"lighting_lux": [300, 600]},
            "embodiment": {"robot_id": "fixture-arm"},
            "sensors": {"camera": "rgb-v1"},
            "controller_action_representation": {"type": "joint_position"},
        }],
        "permitted_evidence_methods": ["traditional_simulation"],
        "idempotency_key": "request-collision-1",
    })
    request = DecisionEvidenceRequest.from_mapping(request_value).to_mapping()
    profile_value = _profile()
    profile_value.pop("method_profile_digest")
    profile_value.update({
        "method_id": "local-swept-aabb-collision",
        "adapter_reference": SWEPT_AABB_COLLISION_SIMULATION_ADAPTER,
        "method_family": "traditional_simulation",
        "supported_claim_types": ["collision_contact"],
        "required_inputs": ["collision_scene"],
        "authority_tier": 2,
        "proof_tier": "sim_only",
        "correlation_group": "qualified-collision-scene",
        "failure_modes": ["collision_scene_invalid"],
        "abstention_modes": ["generated_or_unqualified_geometry"],
        "evaluation_run_template": {
            "schema_version": "evaluation_run.v1",
            "run_id": "template",
            "mode": "evaluate",
            "scene_bundle": {
                "adapter_id": "capture_site_scene_bundle",
                "adapter_version": "1",
                "bundle_id": "capture-1",
                "uri": "fixture://capture-1",
                "entrypoint": "collision-scene.json",
                "content_digest": scene["collision_scene_digest"],
            },
            "robot_adapter": {
                "adapter_id": "robot_profile_adapter",
                "adapter_version": "1",
                "robot_profile_id": "fixture-arm",
                "asset_ref": "fixture://robot",
            },
            "task_scenario_pack": {
                "adapter_id": "robot_eval_matrix_task_scenario_pack",
                "adapter_version": "1",
                "pack_id": "collision-pack",
                "tasks": [{"task_id": "modeled-transfer"}],
                "scenarios": [{"scenario_id": "base", "task_id": "modeled-transfer"}],
            },
            "policy_adapter": {
                "adapter_id": "robot_eval_policy_package",
                "adapter_version": "1",
                "policy_id": "deterministic-trajectory",
                "observation_schema_ref": "fixture_observation.v1",
                "action_schema_ref": "fixture_action.v1",
            },
            "runtime_provider_profile": {
                "adapter_id": "robot_eval_runtime_provider",
                "adapter_version": "1",
                "profile_id": "local-swept-aabb",
                "providers": ["fixture_local"],
                "simulator": "swept_aabb",
                "max_spend_usd": 0,
            },
            "proof_contract": {
                "adapter_id": "robot_eval_proof_contract",
                "adapter_version": "1",
                "contract_id": "modeled-collision-only",
                "required_evidence": ["qualified_collision_scene"],
                "claim_ceiling": {"level": "sim_only"},
                "prohibited_claims": ["physical_success", "deployment_readiness"],
            },
            "metadata": {},
        },
    })
    profile = EvidenceMethodProfile.from_mapping(profile_value).to_mapping()
    qualification_value = _qualification(profile)
    qualification_value.pop("qualification_digest")
    qualification_value.update({
        "qualification_id": "qualification-local-collision-1",
        "claim_type": "collision_contact",
        "evaluator": {"evaluator_id": "independent-collision-check", "version": "1"},
        "subject_provider_id": "blueprint-local-simulation",
        "evaluator_provider_id": "independent-collision-check",
    })
    qualification = QualificationRecord.from_mapping(qualification_value).to_mapping()
    return testbed, request, profile, qualification


def test_run_control_plane_requires_authorization_and_returns_bound_decision(tmp_path) -> None:
    testbed = _testbed()
    request = _request(testbed)
    profile = _profile()
    preparation = prepare_task_evaluation_run(
        state_root=tmp_path,
        run_id="run-local-1",
        capture_session_id="capture-session-1",
        intake_id="capture-1",
        request_value=request,
        testbed_value=testbed,
        method_values=[profile],
        qualification_values=[_qualification(profile)],
        idempotency_key="prepare-local-1",
    )
    assert preparation["state"] == "authorization_required"
    assert preparation["execution_started"] is False
    with pytest.raises(TaskEvaluationRunControlPlaneError, match="execution_authorization"):
        execute_and_aggregate_task_evaluation_run(
            state_root=tmp_path,
            run_id="run-local-1",
        )
    authorization = authorize_task_evaluation_run(
        state_root=tmp_path,
        run_id="run-local-1",
        plan_digest=preparation["evidence_plan"]["plan_digest"],
        authorized_adapter_references=[ANALYTIC_REACHABILITY_ADAPTER],
        actor={"role": "customer", "identity": "firebase:buyer-1"},
        idempotency_key="authorize-local-1",
    )
    assert authorization["physical_robot_run_authorized"] is False
    result = execute_and_aggregate_task_evaluation_run(
        state_root=tmp_path,
        run_id="run-local-1",
    )
    assert result["state"] == "decided"
    assert result["decision_envelope"]["overall_outcome"] == "decision"
    assert result["decision_envelope"]["per_claim_verdicts"][0]["verdict"] == "supported"
    assert result["decision_envelope"]["deployment_approval"] is False
    assert result["decision_envelope"]["uncertainty"]["ranking_science_boundary"] == (
        "thesis_not_supported"
    )
    replay = execute_and_aggregate_task_evaluation_run(
        state_root=tmp_path,
        run_id="run-local-1",
    )
    assert replay["already_exists"] is True
    assert replay["decision_envelope"] == result["decision_envelope"]


def test_run_executes_explicitly_authorized_sim_only_collision_method(tmp_path) -> None:
    testbed, request, profile, qualification = _collision_run_inputs()
    prepared = prepare_task_evaluation_run(
        state_root=tmp_path,
        run_id="run-local-collision",
        capture_session_id="capture-session-collision",
        intake_id="capture-1",
        request_value=request,
        testbed_value=testbed,
        method_values=[profile],
        qualification_values=[qualification],
        idempotency_key="prepare-local-collision",
    )
    assert len(prepared["evidence_plan"]["compiled_evaluation_run_specs"]) == 1
    authorize_task_evaluation_run(
        state_root=tmp_path,
        run_id="run-local-collision",
        plan_digest=prepared["evidence_plan"]["plan_digest"],
        authorized_adapter_references=[SWEPT_AABB_COLLISION_SIMULATION_ADAPTER],
        actor={"role": "customer", "identity": "firebase:buyer-1"},
        idempotency_key="authorize-local-collision",
    )
    result = execute_and_aggregate_task_evaluation_run(
        state_root=tmp_path,
        run_id="run-local-collision",
    )
    evidence = result["evidence_results"][0]
    assert evidence["status"] == "valid"
    assert evidence["supports_claim"] is True
    assert evidence["claim_ceiling"]["sim_only_modeled_collision_clearance"] is True
    assert evidence["claim_ceiling"]["physical_success"] is False
    assert result["decision_envelope"]["overall_outcome"] == "decision"
    assert result["decision_envelope"]["deployment_approval"] is False


def test_run_authorization_rejects_stale_plan_and_unknown_adapter(tmp_path) -> None:
    testbed = _testbed()
    profile = _profile()
    prepared = prepare_task_evaluation_run(
        state_root=tmp_path,
        run_id="run-local-2",
        capture_session_id="capture-session-2",
        intake_id="capture-1",
        request_value=_request(testbed),
        testbed_value=testbed,
        method_values=[profile],
        qualification_values=[_qualification(profile)],
        idempotency_key="prepare-local-2",
    )
    with pytest.raises(TaskEvaluationRunControlPlaneError, match="plan_digest_mismatch"):
        authorize_task_evaluation_run(
            state_root=tmp_path,
            run_id="run-local-2",
            plan_digest=SHA_A,
            authorized_adapter_references=[],
            actor={"role": "customer"},
            idempotency_key="authorize-local-2",
        )
    with pytest.raises(ValueError, match="local_evidence_adapter_not_registered"):
        authorize_task_evaluation_run(
            state_root=tmp_path,
            run_id="run-local-2",
            plan_digest=prepared["evidence_plan"]["plan_digest"],
            authorized_adapter_references=["provider://live-not-authorized"],
            actor={"role": "customer"},
            idempotency_key="authorize-local-2",
        )


def test_run_preparation_rejects_invalid_or_unbound_capture_context(tmp_path) -> None:
    testbed = _testbed()
    profile = _profile()
    with pytest.raises(TaskEvaluationRunControlPlaneError, match="capture_session_id"):
        prepare_task_evaluation_run(
            state_root=tmp_path,
            run_id="run-invalid-context",
            capture_session_id="../escape",
            intake_id="capture-1",
            request_value=_request(testbed),
            testbed_value=testbed,
            method_values=[profile],
            qualification_values=[_qualification(profile)],
            idempotency_key="prepare-invalid-context",
        )
    with pytest.raises(
        TaskEvaluationRunControlPlaneError,
        match="run_intake_testbed_binding_mismatch",
    ):
        prepare_task_evaluation_run(
            state_root=tmp_path,
            run_id="run-unbound-context",
            capture_session_id="capture-session-3",
            intake_id="different-intake",
            request_value=_request(testbed),
            testbed_value=testbed,
            method_values=[profile],
            qualification_values=[_qualification(profile)],
            idempotency_key="prepare-unbound-context",
        )


def test_terminal_run_sync_requires_exact_signed_receipt(tmp_path, monkeypatch) -> None:
    testbed = _testbed()
    profile = _profile()
    prepared = prepare_task_evaluation_run(
        state_root=tmp_path,
        run_id="run-webapp-sync",
        capture_session_id="capture-session-sync",
        intake_id="capture-1",
        request_value=_request(testbed),
        testbed_value=testbed,
        method_values=[profile],
        qualification_values=[_qualification(profile)],
        idempotency_key="prepare-webapp-sync",
    )
    authorize_task_evaluation_run(
        state_root=tmp_path,
        run_id="run-webapp-sync",
        plan_digest=prepared["evidence_plan"]["plan_digest"],
        authorized_adapter_references=[ANALYTIC_REACHABILITY_ADAPTER],
        actor={"role": "customer", "identity": "firebase:buyer-1"},
        idempotency_key="authorize-webapp-sync",
    )
    terminal = execute_and_aggregate_task_evaluation_run(
        state_root=tmp_path,
        run_id="run-webapp-sync",
    )
    assert terminal["webapp_sync"]["status"] == "skipped"
    publication = build_task_evaluation_run_webapp_publication(
        capture_session_id="capture-session-sync",
        intake_id="capture-1",
        run_id="run-webapp-sync",
        state=terminal["state"],
        evidence_plan=prepared["evidence_plan"],
        decision_envelope=terminal["decision_envelope"],
    )
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            receipt = {
                "schema_version": "capture_task_evaluation_run_publication_receipt.v1",
                "status": publication["state"],
                "already_exists": False,
                "capture_session_id": publication["capture_session_id"],
                "intake_id": publication["intake_id"],
                "run_id": publication["run_id"],
                "testbed_digest": publication["testbed_digest"],
                "request_digest": publication["request_digest"],
                "plan_digest": publication["plan_digest"],
                "decision_envelope_digest": publication["decision_envelope"][
                    "decision_envelope_digest"
                ],
                "proof_boundary": publication["proof_boundary"],
            }
            return BytesIO(json.dumps(receipt).encode("utf-8")).read()

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_run_webapp_sync.urllib_request.urlopen",
        fake_urlopen,
    )
    synced = sync_task_evaluation_run_to_webapp(
        capture_session_id="capture-session-sync",
        intake_id="capture-1",
        run_id="run-webapp-sync",
        state=terminal["state"],
        evidence_plan=prepared["evidence_plan"],
        decision_envelope=terminal["decision_envelope"],
        endpoint_url="https://webapp.example/api/internal/pipeline/capture-task-evaluation-runs",
        token="sync-secret",
        max_attempts=1,
    )
    assert synced["status"] == "succeeded"
    assert getattr(captured["request"], "headers")[
        "X-blueprint-pipeline-signature"
    ].startswith("sha256=")
    assert "sync-secret" not in json.dumps(synced)

    class WrongResponse(Response):
        def read(self) -> bytes:
            return b'{"schema_version":"capture_task_evaluation_run_publication_receipt.v1","status":"decided","already_exists":false,"run_id":"wrong"}'

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_run_webapp_sync.urllib_request.urlopen",
        lambda *_args, **_kwargs: WrongResponse(),
    )
    mismatched = sync_task_evaluation_run_to_webapp(
        capture_session_id="capture-session-sync",
        intake_id="capture-1",
        run_id="run-webapp-sync",
        state=terminal["state"],
        evidence_plan=prepared["evidence_plan"],
        decision_envelope=terminal["decision_envelope"],
        endpoint_url="https://webapp.example/api/internal/pipeline/capture-task-evaluation-runs",
        token="sync-secret",
        max_attempts=1,
    )
    assert mismatched["status"] == "failed"
    assert mismatched["reason"] == "response_binding_mismatch"
def test_signed_service_plans_authorizes_executes_and_inspects(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "control" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    token = "run-service-test-secret"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "work"))
    monkeypatch.setenv(
        service.INTAKE_CLIENT_SECRETS_ENV,
        json.dumps({"blueprint-webapp": token}),
    )
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.delenv(service.INTAKE_TOKEN_ENV, raising=False)
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)

    def signed_headers(body: str, nonce: str) -> dict[str, str]:
        timestamp = datetime.now(timezone.utc).isoformat()
        signature = hmac.new(
            token.encode("utf-8"),
            f"{timestamp}.blueprint-webapp.{nonce}.{body}".encode("utf-8"),
            "sha256",
        ).hexdigest()
        return {
            "content-type": "application/json",
            "x-blueprint-pipeline-timestamp": timestamp,
            "x-blueprint-pipeline-client-id": "blueprint-webapp",
            "x-blueprint-pipeline-nonce": nonce,
            "x-blueprint-pipeline-signature": f"sha256={signature}",
        }

    testbed = _testbed()
    profile = _profile()
    catalog = validate_task_evaluation_method_catalog({
        "schema_version": "task_evaluation_method_catalog.v1",
        "catalog_id": "service-local-methods",
        "version": "1",
        "method_profiles": [profile],
        "qualifications": [_qualification(profile)],
    })
    catalog_path = tmp_path / "method-catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setenv(TASK_EVALUATION_METHOD_CATALOG_PATH_ENV, str(catalog_path))
    plan_submission = {
        "schema_version": "task_evaluation_run_plan_submission.v2",
        "run_id": "run-service-1",
        "capture_session_id": "capture-session-service-1",
        "intake_id": "capture-1",
        "decision_evidence_request": _request(testbed),
        "testbed": testbed,
        "idempotency_key": "prepare-run-service-1",
    }
    plan_body = json.dumps(plan_submission, separators=(",", ":"))
    client = TestClient(service.create_app())
    planned = client.post(
        "/api/live-pipeline/task-evaluation-runs/plan",
        content=plan_body,
        headers=signed_headers(plan_body, "run-service-plan-nonce"),
    )
    assert planned.status_code == 200
    plan_result = planned.json()
    assert plan_result["state"] == "authorization_required"
    assert plan_result["method_catalog"] == {
        "catalog_id": "service-local-methods",
        "version": "1",
        "catalog_digest": catalog["catalog_digest"],
        "pipeline_owned": True,
    }
    assert plan_result["authorization_candidates"] == [{
        "adapter_reference": ANALYTIC_REACHABILITY_ADAPTER,
        "method_id": "local-analytic-reachability",
        "method_version": "1",
        "method_profile_digest": profile["method_profile_digest"],
        "method_family": "analytic_geometry_kinematics",
        "expected_cost_usd": 0.0,
        "proof_tier": "analytic_only",
        "execution_authorized": False,
    }]

    authorization_submission = {
        "schema_version": "task_evaluation_run_authorization_submission.v1",
        "plan_digest": plan_result["evidence_plan"]["plan_digest"],
        "authorized_adapter_references": [ANALYTIC_REACHABILITY_ADAPTER],
        "actor": {"role": "customer", "identity": "firebase:buyer-1"},
        "idempotency_key": "authorize-run-service-1",
    }
    authorization_body = json.dumps(authorization_submission, separators=(",", ":"))
    authorized = client.post(
        "/api/live-pipeline/task-evaluation-runs/run-service-1/authorize",
        content=authorization_body,
        headers=signed_headers(authorization_body, "run-service-authorize-nonce"),
    )
    assert authorized.status_code == 200
    assert authorized.json()["physical_robot_run_authorized"] is False

    executed = client.post(
        "/api/live-pipeline/task-evaluation-runs/run-service-1/execute",
        content=b"",
        headers=signed_headers("", "run-service-execute-nonce"),
    )
    assert executed.status_code == 200
    assert executed.json()["state"] == "decided"
    assert executed.json()["decision_envelope"]["overall_outcome"] == "decision"
    assert executed.json()["webapp_sync"]["status"] == "skipped"

    monkeypatch.setenv(TASK_EVALUATION_RUN_WEBAPP_SYNC_REQUIRED_ENV, "true")
    required_sync = client.post(
        "/api/live-pipeline/task-evaluation-runs/run-service-1/execute",
        content=b"",
        headers=signed_headers("", "run-service-required-sync-nonce"),
    )
    assert required_sync.status_code == 502
    assert "task_evaluation_run_webapp_sync_required" in required_sync.json()["detail"]
    monkeypatch.delenv(TASK_EVALUATION_RUN_WEBAPP_SYNC_REQUIRED_ENV, raising=False)

    inspected = client.get(
        "/api/live-pipeline/task-evaluation-runs/run-service-1",
        headers=signed_headers("", "run-service-inspect-nonce"),
    )
    assert inspected.status_code == 200
    assert inspected.json()["state"] == "decided"
    assert inspected.json()["proof_boundary"]["comparative_policy_ranking_verdict"] == (
        "thesis_not_supported"
    )
