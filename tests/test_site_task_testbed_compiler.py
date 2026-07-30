from __future__ import annotations

import copy
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline.capture_intake import validate_capture_intake_envelope
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.live_pipeline_control_plane import CONTROL_PLANE_OUTPUT_PATH_ENV
from blueprint_pipeline.reconstruction_capability import (
    decide_simready_assets,
    plan_reconstruction_methods,
    score_robot_placements,
)
from blueprint_pipeline.site_task_testbed_compiler import (
    SiteTaskTestbedCompilerError,
    compile_site_task_testbed,
    write_testbed_version,
)
from blueprint_pipeline.site_task_testbed_compiler_cli import main as compiler_cli_main
from blueprint_pipeline.site_task_testbed_webapp_sync import (
    build_site_task_testbed_webapp_publication,
    sync_site_task_testbed_to_webapp,
)
from blueprint_pipeline.task_candidate_discovery import compile_approved_task_decision_request


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


def _envelope() -> dict:
    return validate_capture_intake_envelope(
        {
            "schema_version": "capture_intake_envelope.v1",
            "intake_id": "intake-1",
            "idempotency_key": "org-1-intake-1",
            "capture_authority_profile": "iphone_arkit_lidar",
            "source_type": "iphone_arkit_lidar",
            "original_files": [
                {
                    "original_filename": "capture.mov",
                    "relative_path": "capture.mov",
                    "sha256": SHA_A,
                    "size_bytes": 1024,
                    "media_type": "video/quicktime",
                }
            ],
            "scene_id": "site-1",
            "customer_id": "customer-1",
            "organization_id": "org-1",
            "capture_device": {"manufacturer": "Apple", "model": "iPhone Pro"},
            "timing_declaration": {"clock": "monotonic", "monotonic_time_available": True},
            "coordinate_frame_declaration": {"status": "arkit_world", "gravity_aligned": True},
            "available_sensor_streams": [
                {"stream_type": stream, "status": "available", "source_relative_path": "capture.mov"}
                for stream in (
                    "retained_video",
                    "decoded_video_pts",
                    "frame_retention_mapping",
                    "camera_poses",
                    "camera_intrinsics",
                    "depth",
                    "depth_confidence",
                    "tracking_state",
                    "coordinate_frame_semantics",
                )
            ],
            "governance": {
                "rights": "accepted",
                "consent": "accepted",
                "privacy": "cleared",
                "retention": {"max_days": 30},
                "revocation": {"supported": True, "historical_tombstone_retained": True},
                "provider_constraints": {"external_processing_allowed": False},
                "allowed_uses": ["evaluation"],
            },
            "requested_task_evaluation_run_audience": "design_partner",
            "known_task_specification": None,
            "calibration_board_dimensions": None,
            "operator_notes": [],
            "permitted_reconstruction_providers": ["local"],
            "permitted_evidence_uses": ["captured_observation", "analytic_geometry"],
            "upload_validation": {"status": "passed"},
            "malware_content_validation": {"status": "passed"},
        }
    )


def _qa(envelope: dict) -> dict:
    report = {
        "schema_version": "capture_qa_report.v1",
        "intake_id": envelope["intake_id"],
        "envelope_digest": envelope["envelope_digest"],
        "capture_authority_profile": envelope["capture_authority_profile"],
        "status": "accepted",
        "state": "capture_accepted",
        "checks": [],
        "recapture_plan": [],
        "missing_evidence": [],
        "required_analysis": [],
        "next_cheapest_experiment": None,
        "quality_observations_digest": SHA_B,
        "quality_analysis_errors": [],
        "claim_ceiling": {
            "capture_admitted": True,
            "calibrated_camera_poses": True,
            "metric_geometry": True,
            "collision_geometry": False,
            "physical_task_success": False,
        },
        "prohibited_claims": [
            "physical_task_success",
            "deployment_readiness",
            "safety_certification",
            "general_policy_ranking_validity",
        ],
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    report["qa_report_digest"] = canonical_digest(report, digest_field="qa_report_digest")
    return report


def _approved_task() -> dict:
    task = {
        "schema_version": "approved_task_definition.v1",
        "approved_task_id": "approved-task-1",
        "source_capture": {
            "intake_id": "intake-1",
            "capture_digest": SHA_A,
            "capture_authority_profile": "iphone_arkit_lidar",
        },
        "discovery_id": "discovery-1",
        "discovery_digest": SHA_B,
        "task_candidate_id": "candidate-1",
        "candidate_digest": SHA_C,
        "approval_decision_id": "approval-1",
        "approval_decision_digest": SHA_B,
        "approval_actor": {"role": "customer", "identity": "firebase:buyer-1"},
        "intent_source": "customer_approved_candidate",
        "task": {
            "description": "Move the rigid item from the table into the tote.",
            "task_family": "rigid_object_pick_place",
            "measurable_success_conditions": [
                {
                    "metric": "object_center_distance",
                    "operator": "<=",
                    "threshold": 0.05,
                    "units": "m",
                }
            ],
            "reset_contract": {"instructions": "Return the item to the table marker."},
            "task_objects": [{"object_id": "item-1", "label": "rigid item"}],
            "target_regions": [{
                "region_id": "tote-1",
                "label": "tote",
                "position_site_m": [0.6, 0.1, 0.7],
                "supporting_frames": ["frame-1", "frame-2"],
                "captured_coverage": 0.9,
            }],
            "required_robot_capabilities": ["rigid-object grasp"],
        },
        "proposer_identity": "local-rule:fixture",
        "prohibited_evaluator_identities": [],
        "approval_status": "approved",
    }
    task["approved_task_digest"] = canonical_digest(task, digest_field="approved_task_digest")
    return task


def _method() -> dict:
    return {
        "method_id": "local-depth-scaffold",
        "version": "1",
        "implementation_digest": SHA_B,
        "method_kind": "lidar_depth_fusion",
        "provider_identity": "local",
        "execution_mode": "hermetic_local",
        "outputs": ["metric_reference_layer"],
        "required_capture_authority_profiles": ["iphone_arkit_lidar"],
        "required_claim_ceiling_flags": ["metric_geometry"],
        "qualified_claim_types": ["reachability"],
        "execution_authorized": True,
        "qualification_status": "qualified",
        "expected_cost_usd": 0.0,
        "provider_constraints": {},
        "rights_constraints": {},
        "failure_modes": [],
    }


def _plan(qa: dict) -> dict:
    return plan_reconstruction_methods(
        intake_id="intake-1",
        capture_digest=SHA_A,
        capture_authority_profile="iphone_arkit_lidar",
        claim_ceiling=qa["claim_ceiling"],
        requested_claim_types=["reachability"],
        permitted_provider_identities=["local"],
        method_profiles=[_method()],
    )


def _result(plan: dict) -> dict:
    selected = plan["selected_methods"][0]
    return {
        "result_id": "metric-scaffold-1",
        "intake_id": "intake-1",
        "capture_digest": SHA_A,
        "method_id": selected["method_id"],
        "method_version": selected["method_version"],
        "method_profile_digest": selected["method_profile_digest"],
        "implementation_digest": SHA_B,
        "provider_identity": "local",
        "runtime_identity": "fixture-runtime",
        "runtime_digest": SHA_C,
        "outputs": ["metric_reference_layer"],
        "source_frames": {"frame_ids": ["frame-1", "frame-2"]},
        "camera_solution": {"status": "capture_supplied"},
        "coordinate_system": {"up_axis": "Y", "scale_status": "metric_verified"},
        "asset_references": {"metric_scaffold": {"uri": "fixture://scaffold", "digest": SHA_C}},
        "coverage_map": {"covered_fraction": 0.9},
        "observed_regions": [{"region_id": "table"}],
        "generated_regions": [],
        "uncertainty_map": {"uri": "fixture://uncertainty", "digest": SHA_C},
        "invalid_regions": [],
        "validation_metrics": {"held_out_reprojection_error_px": 0.8},
        "cost_usd": 0.0,
        "duration_seconds": 1.0,
        "provider_receipt": None,
        "rights_and_retention": {"external_processing": False},
        "deletion_evidence": None,
        "claim_ceiling": {
            "metric_geometry": True,
            "collision_geometry": False,
            "physical_task_success": False,
        },
    }


def _robot_binding() -> dict:
    return {
        "robot_id": "fixture-arm",
        "embodiment_version": "1",
        "base_footprint": {"shape": "circle", "radius_m": 0.4},
        "sensors": {"camera": "rgb-v1"},
        "controller_id": "joint-position-v1",
        "end_effector_id": "parallel-gripper-v1",
        "reach_envelope": {"minimum_m": 0.1, "maximum_m": 1.0},
    }


def _placement() -> dict:
    return score_robot_placements(
        robot_binding=_robot_binding(),
        approved_task_digest=_approved_task()["approved_task_digest"],
        capture_digest=SHA_A,
        task_object_id="item-1",
        target_region_id="tote-1",
        candidates=[
            {
                "candidate_id": "base-1",
                "site_from_robot_base": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                "base_position_site_m": [0.0, 0.0, 0.0],
                "floor_support_valid": True,
                "footprint_clear": True,
                "access_path_clear": True,
                "collision_free": True,
                "reset_feasible": True,
                "human_clearance_valid": True,
                "captured_coverage": 0.95,
                "reachability_score": 0.9,
                "manipulability_score": 0.8,
                "sensor_visibility_score": 0.8,
                "approach_direction_score": 0.8,
                "cable_controller_score": 0.8,
                "stability_score": 0.9,
                "calibration_uncertainty_m": 0.01,
                "method_qualification_status": "qualified",
                "evidence_digests": [SHA_C],
            }
        ],
    )


def _simready() -> dict:
    return decide_simready_assets(
        approved_task_digest=_approved_task()["approved_task_digest"],
        capture_digest=SHA_A,
        requested_claim_types=["reachability"],
        task_objects=[{"object_id": "item-1"}],
        asset_candidates=[],
    )


def _refs() -> dict:
    return {
        "evaluator": {"uri": "fixture://evaluator", "digest": SHA_C},
        "reset": {"uri": "fixture://reset", "digest": SHA_C},
    }


def _compile(*, previous: dict | None = None, version: str = "1") -> dict:
    envelope = _envelope()
    qa = _qa(envelope)
    plan = _plan(qa)
    return compile_site_task_testbed(
        testbed_id="site-task-1",
        version=version,
        capture_intake_envelope=envelope,
        capture_qa_report=qa,
        approved_task_definition=_approved_task(),
        reconstruction_plan=plan,
        reconstruction_results=[_result(plan)],
        simready_decision=_simready(),
        robot_placement_result=_placement(),
        artifact_references=_refs(),
        supported_condition_ranges={"lighting_lux": [300, 600]},
        previous_testbed=previous,
    )


def _decision_request(testbed: dict, approved: dict | None = None) -> dict:
    return compile_approved_task_decision_request(
        approved or _approved_task(),
        testbed=testbed,
        request_id="request-sync-1",
        decision_id="decision-sync-1",
        candidates=[{"robot_id": "fixture-arm"}],
        claims=[{
            "claim_id": "reach",
            "claim_type": "reachability",
            "subject": "fixture-arm:item-1:tote-1",
            "measurable_threshold": {"operator": ">=", "value": 0.9, "units": "fraction"},
            "false_safe_consequence": "moderate",
            "acceptable_false_safe_risk": 0.05,
            "desired_confidence_or_coverage": {
                "minimum_coverage": 0.9,
                "minimum_independent_methods": 1,
            },
            "permitted_abstention_behavior": {"allowed": True},
            "task_family": "rigid_object_pick_place",
            "site_domain_conditions": {"lighting_lux": [300, 600]},
            "embodiment": {"robot_id": "fixture-arm"},
            "sensors": {"camera": "rgb-v1"},
            "controller_action_representation": {"type": "joint_position"},
        }],
        budget={"max_cost_usd": 0.0},
        deadline="2026-07-30T00:00:00Z",
        permitted_evidence_methods=["analytic_geometry_kinematics"],
        restrictions={"external_processing_allowed": False},
        requested_result_audience="design_partner",
        caller_identity="pipeline:testbed-compiler",
        idempotency_key="request-sync-1",
    )


def test_compiler_emits_deterministic_layered_router_compatible_testbed() -> None:
    first = _compile()
    second = _compile()

    assert first == second
    assert first["schema_version"] == "maintained_site_task_testbed.v1"
    assert first["approved_task_definition"]["digest"] == _approved_task()[
        "approved_task_digest"
    ]
    envelope = first["validation_envelope"]
    assert envelope["reconstruction_layers"]["metric_reference_layer"][0]["output"] == (
        "metric_reference_layer"
    )
    assert envelope["reconstruction_layers"]["physics_layer"] == []
    assert first["compiled_cards"]["task_cards"][0]["approved_task_digest"] == (
        _approved_task()["approved_task_digest"]
    )
    assert first["compiled_cards"]["eval_cards"][0][
        "comparative_policy_ranking_verdict"
    ] == "thesis_not_supported"
    assert first["robot_sensor_controller_bindings"]["selected_robot_placement"] == (
        _placement()["accepted_candidates"][0]
    )
    assert first["proof_boundary"] == {
        "appearance_is_collision_truth": False,
        "generated_completion_is_observed_truth": False,
        "simulation_is_physical_success": False,
        "deployment_or_safety_approved": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }


def test_compiled_testbed_unlocks_provider_neutral_decision_request() -> None:
    testbed = _compile()
    request = compile_approved_task_decision_request(
        _approved_task(),
        testbed=testbed,
        request_id="request-1",
        decision_id="decision-1",
        candidates=[{"robot_id": "fixture-arm"}],
        claims=[
            {
                "claim_id": "reach",
                "claim_type": "reachability",
                "subject": "fixture-arm:item-1:tote-1",
                "measurable_threshold": {
                    "operator": ">=",
                    "value": 0.95,
                    "units": "fraction",
                },
                "false_safe_consequence": "moderate",
                "acceptable_false_safe_risk": 0.05,
                "desired_confidence_or_coverage": {
                    "minimum_coverage": 0.95,
                    "minimum_independent_methods": 1,
                },
                "permitted_abstention_behavior": {"allowed": True},
                "task_family": "rigid_object_pick_place",
                "site_domain_conditions": {"lighting_lux": [300, 600]},
                "embodiment": {"robot_id": "fixture-arm"},
                "sensors": {"camera": "rgb-v1"},
                "controller_action_representation": {"type": "joint_position"},
            }
        ],
        budget={"max_cost_usd": 1.0, "max_latency_seconds": 60},
        deadline="2026-07-30T00:00:00Z",
        permitted_evidence_methods=[
            "analytic_geometry_kinematics",
            "captured_real_observation",
        ],
        restrictions={"external_processing_allowed": False},
        requested_result_audience="design_partner",
        caller_identity="testbed-compiler-test",
        idempotency_key="request-1",
    )

    assert request["testbed_digest"] == testbed["testbed_digest"]
    assert request["provenance"]["approved_task_digest"] == _approved_task()[
        "approved_task_digest"
    ]
    assert "selected_provider" not in request
    assert "selected_method" not in request


def test_compiler_rejects_stale_capture_reconstruction_and_unaccepted_qa() -> None:
    fixture = _beta_fixture("stale_reconstruction_wrong_source_digest")
    assert fixture["source_capture_digest_matches"] is False
    envelope = _envelope()
    qa = _qa(envelope)
    plan = _plan(qa)
    stale = _result(plan)
    stale["capture_digest"] = "sha256:" + "f" * 64
    stale["outputs"] = ["collision_geometry"]
    rejected = copy.deepcopy(qa)
    rejected["status"] = "recapture_required"
    rejected["state"] = "rejected_or_recapture_required"
    rejected["qa_report_digest"] = canonical_digest(rejected, digest_field="qa_report_digest")
    tampered_placement = _placement()
    tampered_placement["robot_binding"]["robot_id"] = "different-robot"
    tampered_placement["robot_placement_digest"] = canonical_digest(
        tampered_placement, digest_field="robot_placement_digest"
    )

    with pytest.raises(SiteTaskTestbedCompilerError) as exc_info:
        compile_site_task_testbed(
            testbed_id="site-task-1",
            version="1",
            capture_intake_envelope=envelope,
            capture_qa_report=rejected,
            approved_task_definition=_approved_task(),
            reconstruction_plan=plan,
            reconstruction_results=[stale],
            simready_decision=_simready(),
            robot_placement_result=tampered_placement,
            artifact_references=_refs(),
            supported_condition_ranges={"lighting_lux": [300, 600]},
        )
    assert "capture_qa_report:not_accepted" in exc_info.value.errors
    assert "reconstruction_results[0]:capture_digest_mismatch" in exc_info.value.errors
    assert fixture["expected_blocker"] == "source_capture_digest_mismatch"
    assert "reconstruction_results[0]:output_not_selected_in_plan" in exc_info.value.errors
    assert "robot_placement_result:robot_binding_digest_mismatch" in exc_info.value.errors


def test_testbed_versions_are_immutable_and_successors_bind_predecessor(tmp_path) -> None:
    first = _compile()
    first_write = write_testbed_version(output_root=tmp_path, testbed=first)
    replay = write_testbed_version(output_root=tmp_path, testbed=first)
    successor = _compile(previous=first, version="2")
    successor_write = write_testbed_version(output_root=tmp_path, testbed=successor)

    assert first_write["already_exists"] is False
    assert replay["already_exists"] is True
    assert successor["predecessor_testbed_digest"] == first["testbed_digest"]
    assert successor["supersedes"] == [first["testbed_digest"]]
    assert successor_write["testbed_digest"] != first_write["testbed_digest"]
    assert len(list(tmp_path.rglob("*.json"))) == 12


def test_compiler_rejects_same_version_successor() -> None:
    first = _compile()
    with pytest.raises(SiteTaskTestbedCompilerError, match="version_must_change"):
        _compile(previous=first, version="1")


def test_writer_rejects_two_digests_for_one_logical_version(tmp_path) -> None:
    first = _compile()
    write_testbed_version(output_root=tmp_path, testbed=first)
    conflicting = copy.deepcopy(first)
    conflicting["supported_condition_ranges"] = {"lighting_lux": [200, 700]}
    conflicting.pop("testbed_digest")

    with pytest.raises(SiteTaskTestbedCompilerError, match="testbed_version_digest_conflict"):
        write_testbed_version(output_root=tmp_path, testbed=conflicting)


def test_compiler_cli_writes_one_immutable_testbed_version(tmp_path, capsys) -> None:
    envelope = _envelope()
    qa = _qa(envelope)
    plan = _plan(qa)
    inputs = {
        "capture-intake-envelope": envelope,
        "capture-qa-report": qa,
        "approved-task-definition": _approved_task(),
        "reconstruction-plan": plan,
        "reconstruction-results": [_result(plan)],
        "simready-decision": _simready(),
        "robot-placement-result": _placement(),
        "artifact-references": _refs(),
        "supported-condition-ranges": {"lighting_lux": [300, 600]},
    }
    arguments = ["--testbed-id", "site-task-cli", "--version", "1"]
    for option, value in inputs.items():
        path = tmp_path / f"{option}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        arguments.extend([f"--{option}", str(path)])
    output_root = tmp_path / "testbeds"
    arguments.extend(["--output-root", str(output_root)])

    assert compiler_cli_main(arguments) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "testbed_ready"
    assert receipt["already_exists"] is False
    assert compiler_cli_main(arguments) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["already_exists"] is True
    assert replay["testbed_digest"] == receipt["testbed_digest"]


def test_signed_service_compiles_only_the_authoritative_approved_task(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "control" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    work_dir = tmp_path / "work"
    session_dir = (
        work_dir
        / "task_candidate_control_plane"
        / "sessions"
        / "capture-session-1"
    )
    result_dir = session_dir / "decisions" / "task-command-1"
    result_dir.mkdir(parents=True)
    approved = _approved_task()
    (session_dir / "state.json").write_text(
        json.dumps(
            {
                "schema_version": "task_candidate_control_plane_state.v1",
                "capture_session_id": "capture-session-1",
                "intake_id": "intake-1",
                "latest_command_request_id": "task-command-1",
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "schema_version": "task_candidate_decision_processing_result.v1",
                "pipeline_approval_status": "approved",
                "capture_session_id": "capture-session-1",
                "intake_id": "intake-1",
                "approved_task_definition": approved,
            }
        ),
        encoding="utf-8",
    )
    token = "test-testbed-service-secret"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(work_dir))
    monkeypatch.setenv(
        service.INTAKE_CLIENT_SECRETS_ENV,
        json.dumps({"blueprint-webapp": token}),
    )
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.delenv(service.INTAKE_TOKEN_ENV, raising=False)
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    envelope = _envelope()
    qa = _qa(envelope)
    plan = _plan(qa)
    execution_digest = SHA_C
    monkeypatch.setattr(
        service,
        "load_reconstruction_compilation_inputs",
        lambda **_: {
            "context": {
                "capture_session_id": "capture-session-1",
                "intake_id": "intake-1",
            },
            "capture_intake_envelope": envelope,
            "capture_qa_report": qa,
            "reconstruction_plan": plan,
            "reconstruction_results": [_result(plan)],
            "execution_result": {
                "state": "completed",
                "execution_result_digest": execution_digest,
            },
        },
    )
    monkeypatch.setenv(service.CAPTURE_UPLOAD_STORE_ROOT_ENV, str(tmp_path / "capture-store"))
    payload = {
        "schema_version": "site_task_testbed_compilation_submission.v2",
        "capture_session_id": "capture-session-1",
        "intake_id": "intake-1",
        "testbed_id": "site-task-service",
        "version": "1",
        "approved_task_digest": approved["approved_task_digest"],
        "reconstruction_plan_id": "reconstruction-plan-1",
        "reconstruction_execution_result_digest": execution_digest,
        "robot_binding": _robot_binding(),
        "decision_request_constraints": {
            "request_id": "request-service-1",
            "decision_id": "decision-service-1",
            "candidates": [{"robot_id": "fixture-arm"}],
            "claims": [{
                "claim_id": "reach",
                "claim_type": "reachability",
                "subject": "fixture-arm:item-1:tote-1",
                "measurable_threshold": {
                    "operator": ">=",
                    "value": 0.95,
                    "units": "fraction",
                },
                "false_safe_consequence": "moderate",
                "acceptable_false_safe_risk": 0.05,
                "desired_confidence_or_coverage": {
                    "minimum_coverage": 0.95,
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
            "permitted_evidence_methods": ["analytic_geometry_kinematics"],
            "restrictions": {"external_processing_allowed": False},
            "requested_result_audience": "design_partner",
            "idempotency_key": "request-service-1",
        },
    }
    body = json.dumps(payload, separators=(",", ":"))
    timestamp = datetime.now(timezone.utc).isoformat()
    nonce = "testbed-compile-nonce-1"
    signature = hmac.new(
        token.encode("utf-8"),
        f"{timestamp}.blueprint-webapp.{nonce}.{body}".encode("utf-8"),
        "sha256",
    ).hexdigest()
    response = TestClient(service.create_app()).post(
        "/api/live-pipeline/testbeds/compile",
        content=body,
        headers={
            "content-type": "application/json",
            "x-blueprint-pipeline-timestamp": timestamp,
            "x-blueprint-pipeline-client-id": "blueprint-webapp",
            "x-blueprint-pipeline-nonce": nonce,
            "x-blueprint-pipeline-signature": f"sha256={signature}",
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "testbed_ready"
    assert result["testbed"]["approved_task_definition"]["digest"] == (
        approved["approved_task_digest"]
    )
    assert "artifact_path" not in json.dumps(result)
    assert result["proof_boundary"]["deployment_or_safety_approved"] is False
    assert result["webapp_sync"]["status"] == "skipped"
    placement_evidence = next(
        row
        for row in result["testbed"]["evidence_inventory"]
        if row["evidence_id"] == "robot_placement"
    )
    assert placement_evidence["status"] == "abstained"
    assert "robot_placement_not_established" in result["testbed"][
        "known_unsupported_conditions"
    ]
    version_root = (
        work_dir / "maintained_site_task_testbeds" / "site-task-service" / "1"
    )
    assert (version_root / "evaluator.json").is_file()
    assert (version_root / "reset.json").is_file()
    assert result["decision_evidence_request"]["testbed_digest"] == result["testbed_digest"]
    assert result["decision_evidence_request_artifact"]["request_digest"] == (
        result["decision_evidence_request"]["request_digest"]
    )
    assert "artifact_path" not in json.dumps(result["decision_evidence_request_artifact"])

    caller_scientific_payload = {**payload, "simready_decision": _simready()}
    caller_scientific_body = json.dumps(caller_scientific_payload, separators=(",", ":"))
    caller_scientific_nonce = "testbed-compile-nonce-2"
    caller_scientific_signature = hmac.new(
        token.encode("utf-8"),
        (
            f"{timestamp}.blueprint-webapp.{caller_scientific_nonce}."
            f"{caller_scientific_body}"
        ).encode("utf-8"),
        "sha256",
    ).hexdigest()
    rejected = TestClient(service.create_app()).post(
        "/api/live-pipeline/testbeds/compile",
        content=caller_scientific_body,
        headers={
            "content-type": "application/json",
            "x-blueprint-pipeline-timestamp": timestamp,
            "x-blueprint-pipeline-client-id": "blueprint-webapp",
            "x-blueprint-pipeline-nonce": caller_scientific_nonce,
            "x-blueprint-pipeline-signature": f"sha256={caller_scientific_signature}",
        },
    )
    assert rejected.status_code == 422
    assert "Pipeline-owned scientific inputs forbidden:simready_decision" in (
        rejected.json()["detail"]
    )


def test_testbed_webapp_publication_is_exactly_bound_and_receipt_verified(
    monkeypatch,
) -> None:
    envelope = _envelope()
    qa = _qa(envelope)
    plan = _plan(qa)
    approved = _approved_task()
    testbed = compile_site_task_testbed(
        testbed_id="sync-testbed",
        version="1",
        capture_intake_envelope=envelope,
        capture_qa_report=qa,
        approved_task_definition=approved,
        reconstruction_plan=plan,
        reconstruction_results=[_result(plan)],
        simready_decision=_simready(),
        robot_placement_result=_placement(),
        artifact_references=_refs(),
        supported_condition_ranges={"lighting_lux": [300, 600]},
    )
    decision_request = _decision_request(testbed, approved)
    publication = build_site_task_testbed_webapp_publication(
        capture_session_id="capture-session-1",
        intake_id="intake-1",
        approved_task_digest=approved["approved_task_digest"],
        testbed=testbed,
        decision_evidence_request=decision_request,
    )
    assert publication["testbed_digest"] == testbed["testbed_digest"]
    assert publication["proof_boundary"]["comparative_policy_ranking_verdict"] == (
        "thesis_not_supported"
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            receipt = {
                "schema_version": "capture_site_task_testbed_publication_receipt.v1",
                "status": "testbed_ready",
                "already_exists": False,
                **{
                    field: publication[field]
                    for field in (
                        "capture_session_id",
                        "intake_id",
                        "approved_task_digest",
                        "testbed_id",
                        "version",
                        "testbed_digest",
                        "artifact_reference",
                        "proof_boundary",
                    )
                },
                "request_digest": decision_request["request_digest"],
            }
            return json.dumps(receipt).encode("utf-8")

    monkeypatch.setattr(
        "blueprint_pipeline.site_task_testbed_webapp_sync.urllib_request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    result = sync_site_task_testbed_to_webapp(
        capture_session_id="capture-session-1",
        intake_id="intake-1",
        approved_task_digest=approved["approved_task_digest"],
        testbed=testbed,
        decision_evidence_request=decision_request,
        endpoint_url="https://webapp.test/api/internal/pipeline/capture-testbeds",
        token="sync-secret",
    )
    assert result["status"] == "succeeded"
    assert result["testbed_digest"] == testbed["testbed_digest"]


def test_testbed_webapp_sync_rejects_mismatched_success_receipt(monkeypatch) -> None:
    envelope = _envelope()
    qa = _qa(envelope)
    plan = _plan(qa)
    approved = _approved_task()
    testbed = compile_site_task_testbed(
        testbed_id="sync-testbed",
        version="1",
        capture_intake_envelope=envelope,
        capture_qa_report=qa,
        approved_task_definition=approved,
        reconstruction_plan=plan,
        reconstruction_results=[_result(plan)],
        simready_decision=_simready(),
        robot_placement_result=_placement(),
        artifact_references=_refs(),
        supported_condition_ranges={"lighting_lux": [300, 600]},
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "schema_version": "capture_site_task_testbed_publication_receipt.v1",
                    "status": "testbed_ready",
                    "already_exists": False,
                    "testbed_digest": "sha256:" + "f" * 64,
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "blueprint_pipeline.site_task_testbed_webapp_sync.urllib_request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    result = sync_site_task_testbed_to_webapp(
        capture_session_id="capture-session-1",
        intake_id="intake-1",
        approved_task_digest=approved["approved_task_digest"],
        testbed=testbed,
        endpoint_url="https://webapp.test/api/internal/pipeline/capture-testbeds",
        token="sync-secret",
        max_attempts=1,
    )
    assert result["status"] == "failed"
    assert result["reason"] == "response_binding_mismatch"
