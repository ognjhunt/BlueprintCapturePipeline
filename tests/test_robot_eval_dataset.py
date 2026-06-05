from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.robot_eval_dataset import build_real_site_robot_eval_dataset


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "raw" / "manifest.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "site_identity": {"site_id": "site-1"},
        },
    )
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {"site_identity": {"site_id": "site-1"}},
        },
    )
    return capture_root


def _write_eval_inputs(capture_root: Path) -> None:
    eval_dir = capture_root / "pipeline" / "evaluation_prep"
    _write_json(
        eval_dir / "object_geometry_manifest.json",
        {
            "schema_version": "v1",
            "objects": [
                {
                    "object_id": "bin_0001",
                    "label": "returns bin",
                    "task_role": "target_container",
                    "collision_hulls": [{"kind": "box"}],
                    "support_surfaces": [{"kind": "rim"}],
                    "provenance": {"grounding_level": "observed"},
                }
            ],
        },
    )
    _write_json(
        eval_dir / "task_anchor_manifest.json",
        {
            "schema_version": "v1",
            "updated_at": "2026-06-03T00:00:00Z",
            "tasks": [
                {
                    "task_id": "place_return_in_bin",
                    "task_text": "Place the return item in the labeled bin",
                    "task_category": "pick_place",
                    "target_object_ids": ["bin_0001"],
                    "start_zone": [0.0, 0.0, 0.0],
                    "goal_zone": [1.0, 0.0, 0.2],
                    "task_critical": True,
                }
            ],
        },
    )
    _write_json(
        eval_dir / "site_world_spec.json",
        {
            "schema_version": "v1",
            "robot_profiles": [
                {
                    "id": "mobile_manipulator_rgb_v1",
                    "display_name": "Mobile manipulator",
                    "embodiment_type": "mobile_manipulator",
                    "action_space": {"name": "ee_delta_pose_gripper", "dim": 7},
                }
            ],
        },
    )
    _write_json(
        eval_dir / "hosted_session_runtime_manifest.json",
        {"schema_version": "v1", "robot_profiles": []},
    )


def _write_review_sources(capture_root: Path) -> None:
    _write_json(
        capture_root / "pipeline" / "simready" / "simready_scene_manifest.json",
        {
            "schema_version": "simready_scene_manifest.v1",
            "status": "prepared_for_review",
            "simulator_execution_proven": False,
        },
    )
    _write_json(
        capture_root / "pipeline" / "simready" / "simready_validation.json",
        {"schema_version": "simready_validation.v1", "overall_status": "prepared_for_review"},
    )
    _write_json(
        capture_root / "pipeline" / "marble_sim_assets" / "marble_simready_bridge.json",
        {
            "schema_version": "marble_simready_bridge.v1",
            "status": "review_ready_with_conversion_required",
        },
    )
    _write_json(
        capture_root / "pipeline" / "marble_sim_assets" / "marble_asset_validation.json",
        {
            "schema_version": "marble_asset_validation.v1",
            "overall_status": "review_ready",
            "physics_collision_review_ready": True,
            "collider_mesh_available": True,
            "collider_mesh_glb_url": "gs://bucket/collider.glb",
        },
    )
    _write_json(
        capture_root / "pipeline" / "rights_and_compliance_summary.json",
        {"schema_version": "v1", "status": "verified"},
    )
    _write_json(
        capture_root / "pipeline" / "privacy_processing_manifest.json",
        {"schema_version": "v1", "status": "person_removed", "fail_closed": True},
    )


def _write_recorded_trace_fixture(capture_root: Path) -> None:
    _write_json(
        capture_root / "pipeline" / "robot_eval_inputs" / "recorded_action_trace_manifest.json",
        {
            "schema_version": "recorded_action_trace_manifest.v1",
            "generated_at": "2026-06-03T00:00:00Z",
            "owner_system": "BlueprintCapturePipeline.fixture",
            "attempts": [
                {
                    "attempt_id": "trace-attempt-1",
                    "trace_id": "trace-1",
                    "scenario_id": "scenario_place_return_in_bin_mobile_manipulator_rgb_v1",
                    "task_id": "place_return_in_bin",
                    "success": False,
                    "metrics": {
                        "cycle_time_seconds": 18.25,
                        "intervention_count": 1,
                        "unsafe_proximity_event_count": 1,
                        "collision_risk_event_count": 1,
                        "object_drop_count": 1,
                        "wrong_object_count": 0,
                        "timeout_count": 0,
                        "recovery_attempt_count": 1,
                        "recovery_success_count": 0,
                    },
                    "failure_mode_ids": [
                        "failure_contact_collision",
                        "failure_safety_threshold_violation",
                    ],
                    "evidence_refs": {"trace": "fixtures/trace-1.json"},
                }
            ],
        },
    )


def test_robot_eval_dataset_emits_fail_closed_contract(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_eval_inputs(capture_root)
    _write_review_sources(capture_root)

    result = build_real_site_robot_eval_dataset(capture_root=capture_root)
    first_fingerprint = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))[
        "deterministic_fingerprint"
    ]
    second = build_real_site_robot_eval_dataset(capture_root=capture_root)
    second_fingerprint = json.loads(Path(second["manifest_path"]).read_text(encoding="utf-8"))[
        "deterministic_fingerprint"
    ]

    robot_eval_root = capture_root / "pipeline" / "robot_eval_dataset"
    primary_manifest = json.loads(
        (robot_eval_root / "robot_eval_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (robot_eval_root / "real_site_robot_eval_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    site_card = json.loads((robot_eval_root / "site_card.json").read_text())
    task_cards = json.loads((robot_eval_root / "task_cards.json").read_text())
    scenario_cards = json.loads((robot_eval_root / "scenario_cards.json").read_text())
    eval_cards = json.loads((robot_eval_root / "eval_cards.json").read_text())
    annotation_backlog = json.loads((robot_eval_root / "annotation_backlog.json").read_text())
    proof_boundaries = json.loads((robot_eval_root / "proof_boundaries.json").read_text())
    task_library = json.loads((robot_eval_root / "robot_task_library.json").read_text())
    scenario_library = json.loads((robot_eval_root / "scenario_library.json").read_text())
    failure_taxonomy = json.loads((robot_eval_root / "failure_taxonomy.json").read_text())
    ledger = json.loads((robot_eval_root / "prediction_outcome_ledger.json").read_text())
    robot_team_submission_modalities = json.loads(
        (robot_eval_root / "robot_team_test_submission_modalities.json").read_text()
    )
    evidence_contract = json.loads(
        (robot_eval_root / "robot_eval_inputs_evidence_contract.json").read_text()
    )
    task_ontology = json.loads((robot_eval_root / "task_ontology_v1.json").read_text())
    scenario_families = json.loads(
        (robot_eval_root / "scenario_family_library.json").read_text()
    )
    scoring = json.loads((robot_eval_root / "scoring_methodology.json").read_text())
    recorded_report = json.loads(
        (robot_eval_root / "recorded_trace_eval_report.json").read_text()
    )
    policy_report = json.loads((robot_eval_root / "policy_eval_report.json").read_text())
    prediction_summary = json.loads(
        (robot_eval_root / "prediction_vs_actual_summary.json").read_text()
    )
    rights_packet = json.loads((robot_eval_root / "rights_packet.json").read_text())
    rights_ledger = json.loads((robot_eval_root / "rights_ledger.json").read_text())
    methodology = (robot_eval_root / "eval_methodology_summary.md").read_text(encoding="utf-8")

    assert result["status"] == "capture_grounded_review_ready"
    assert first_fingerprint == second_fingerprint
    assert primary_manifest["schema_version"] == "real_site_robot_eval_dataset_manifest.v0.1"
    assert primary_manifest["dataset_version"] == "0.1"
    assert primary_manifest == manifest
    assert manifest["dataset_statuses"] == [
        "capture_grounded_ready",
        "needs_robot_pov",
        "needs_human_demo",
        "needs_action_logs",
        "needs_actual_outcome",
        "needs_policy_api_endpoint_ref",
        "needs_docker_container_ref",
        "needs_recorded_action_trace_ref",
        "needs_high_level_skill_trace_ref",
        "needs_teleop_demo_ref",
        "needs_sim_controller_plugin_ref",
        "review_only_no_robot_readiness",
    ]
    assert manifest["claim_boundary"]["robot_readiness_proven"] is False
    assert manifest["claim_boundary"]["simulator_execution_proven"] is False
    assert manifest["site_card_count"] == 1
    assert manifest["task_card_count"] == 1
    assert manifest["scenario_card_count"] == 1
    assert manifest["eval_card_count"] == 2
    assert manifest["annotation_backlog_count"] > 0
    assert manifest["task_ontology_count"] == 10
    assert manifest["scenario_family_count"] == 1
    assert manifest["recorded_trace_eval_status"] == "blocked_missing_recorded_trace"
    assert manifest["prediction_vs_actual_status"] == "blocked_missing_actuals"
    assert manifest["rights_packet_status"] == "review_required"
    assert manifest["robot_team_test_submission_modality_count"] == 6
    assert manifest["output_artifacts"]["robot_eval_inputs_evidence_contract"] == (
        "robot_eval_inputs_evidence_contract.json"
    )
    assert manifest["output_artifacts"]["rights_packet"] == "rights_packet.json"
    assert manifest["output_artifacts"]["task_ontology_v1"] == "task_ontology_v1.json"
    assert manifest["output_artifacts"]["recorded_trace_eval_report"] == (
        "recorded_trace_eval_report.json"
    )
    assert manifest["robot_team_test_submission_missing_evidence_statuses"] == [
        "needs_policy_api_endpoint_ref",
        "needs_docker_container_ref",
        "needs_recorded_action_trace_ref",
        "needs_high_level_skill_trace_ref",
        "needs_teleop_demo_ref",
        "needs_sim_controller_plugin_ref",
    ]
    assert manifest["webapp_sync_boundary"]["must_not_display_as"] == [
        "robot_ready",
        "deployment_ready",
        "safety_validated",
        "simulator_completed",
        "actual_outcome_proven",
    ]
    assert site_card["schema_version"] == "real_site_robot_eval_site_card.v0.1"
    assert site_card["site_type"] == "unknown_site_type"
    assert site_card["geometry"]["collider"]["status"] == "review_input_present"
    assert site_card["geometry"]["collider"]["collision_ready_claim_allowed"] is False
    assert site_card["safety_constraints"]["claim_boundary"] == (
        "safety_constraints_are_review_inputs_not_safety_validation"
    )
    assert task_cards["task_card_count"] == 1
    assert task_cards["cards"][0]["ontology_task_id"] == "place_object_into_bin"
    assert task_cards["cards"][0]["required_metrics"] == [
        "cycle_time_seconds",
        "placement_accuracy",
        "intervention_rate",
        "recovery_success",
    ]
    assert scenario_cards["scenario_card_count"] == 1
    assert scenario_cards["cards"][0]["observed_vs_inferred_labels"]["variation"] == (
        "agent_inferred"
    )
    assert "generated_scenarios_are_not_real_world_proof" in scenario_cards["cards"][0]["known_risk"]
    assert eval_cards["eval_card_count"] == 2
    assert all(
        "robot_policy_execution_proven" in card["blocked_upgrades"]
        for card in eval_cards["cards"]
    )
    assert annotation_backlog["backlog_count"] > 0
    assert any(item["backlog_id"] == "needs_action_logs" for item in annotation_backlog["items"])
    assert any(
        item["backlog_id"] == "needs_policy_api_endpoint_ref"
        for item in annotation_backlog["items"]
    )
    assert proof_boundaries["simulator_execution_proven"] is False
    assert proof_boundaries["physics_contact_validation_proven"] is False
    assert proof_boundaries["robot_policy_execution_proven"] is False
    assert proof_boundaries["safety_validation_proven"] is False
    assert proof_boundaries["rights_cleared_external_licensing_proven"] is False
    assert proof_boundaries["real_pilot_outcome_proven"] is False
    assert proof_boundaries["robot_team_test_submission_refs_present"] is False
    assert robot_team_submission_modalities["schema_version"] == (
        "robot_team_test_submission_modalities.v0.1"
    )
    assert robot_team_submission_modalities["modality_count"] == 6
    assert {
        item["modality_id"] for item in robot_team_submission_modalities["modalities"]
    } == {
        "policy_api_endpoint",
        "docker_container",
        "recorded_action_trace",
        "high_level_skill_trace",
        "teleop_demo",
        "sim_controller_plugin",
    }
    policy_schema = next(
        item
        for item in robot_team_submission_modalities["modalities"]
        if item["modality_id"] == "policy_api_endpoint"
    )
    assert policy_schema["required_reference_fields"] == [
        "endpointUrl",
        "authHandling",
        "observationSchemaRef",
        "actionSchemaRef",
        "runtimeConstraints",
        "callbackLogUri",
        "ownerContact",
    ]
    assert robot_team_submission_modalities["blocked_claim_upgrades"] == [
        "ready_to_deploy_claim",
        "safety_validated_claim",
        "simulator_completed_claim",
        "robot_trial_passed_claim",
        "policy_execution_passed_claim",
        "guaranteed_threshold_claim",
    ]
    assert task_library["task_count"] == 1
    assert task_library["tasks"][0]["ontology_task_id"] == "place_object_into_bin"
    assert task_library["tasks"][0]["required_evidence"] == [
        "robot_pov_evidence",
        "human_demo_evidence",
        "action_log_evidence",
        "prediction_outcome_record",
    ]
    assert scenario_library["scenario_count"] == 1
    assert task_ontology["schema_version"] == "real_site_robot_eval_task_ontology.v1"
    assert {
        item["task_id"] for item in task_ontology["tasks"]
    } >= {
        "navigate_to_station",
        "inspect_shelf",
        "move_tote",
        "cart_to_conveyor_transfer",
        "line_side_delivery",
        "pick_known_object",
        "place_object_into_bin",
        "blocked_path_recovery",
        "human_crossing_safety_response",
        "open_door_enter_room",
    }
    assert scenario_families["schema_version"] == (
        "real_site_robot_eval_scenario_family_library.v1"
    )
    assert set(scenario_families["variation_names_required"]) == {
        "lighting_variation",
        "object_rotation",
        "cart_shifted",
        "blocked_path",
        "human_crossing",
        "forklift_nearby",
        "occlusion",
        "glare",
        "missing_label",
        "wrong_object_nearby",
        "narrow_approach_angle",
    }
    assert {
        variation["scenario_status"]
        for variation in scenario_families["families"][0]["variations"]
    } <= {
        "capture-grounded",
        "representative-mock",
        "agent-inferred-needs-review",
        "accepted",
        "rejected",
        "review-only",
    }
    assert scenario_families["cosmos_or_simulator_proof_claim_allowed"] is False
    assert {
        mode["failure_mode_id"] for mode in failure_taxonomy["failure_modes"]
    } >= {
        "failure_contact_collision",
        "failure_intervention_required",
        "failure_evidence_missing",
    }
    assert ledger["ledger_status"] == "needs_actual_outcome"
    assert ledger["record_count"] == 2
    assert {record["prediction_source"] for record in ledger["records"]} == {
        "marble_review",
        "simready_review",
    }
    assert all(record["actual_status"] == "needs_actual_outcome" for record in ledger["records"])
    assert all(record["actual_success"] is None for record in ledger["records"])
    assert evidence_contract["schema_version"] == "robot_eval_inputs_evidence_contract.v1"
    assert set(evidence_contract["contracts"]) == {
        "robot_pov",
        "human_demo",
        "action_logs",
        "recorded_action_traces",
        "simulator_traces",
        "policy_submissions",
        "actual_outcomes",
    }
    assert "rights_privacy_scope" in evidence_contract["required_cross_cutting_fields"]
    assert "timestamp_alignment" in evidence_contract["required_cross_cutting_fields"]
    assert "owner_system" in evidence_contract["required_cross_cutting_fields"]
    assert "provenance" in evidence_contract["required_cross_cutting_fields"]
    assert scoring["schema_version"] == "real_site_robot_eval_scoring_methodology.v1"
    assert {
        metric["metric_id"] for metric in scoring["metrics"]
    } >= {
        "success_rate",
        "cycle_time",
        "intervention_rate",
        "unsafe_proximity",
        "collision_risk",
        "object_drop",
        "wrong_object",
        "timeout",
        "recovery_success",
        "uncertainty",
        "sim_vs_real_calibration_placeholder",
    }
    assert recorded_report["status"] == "blocked_missing_recorded_trace"
    assert recorded_report["proof_boundary"]["simulator_execution_proven"] is False
    assert recorded_report == policy_report
    assert prediction_summary["status"] == "blocked_missing_actuals"
    assert prediction_summary["missing_actuals_remain_blocked"] is True
    assert rights_packet["commercial_use_claim_allowed"] is False
    assert rights_packet["external_licensing_claim_allowed"] is False
    assert {record["rights_scope"] for record in rights_packet["records"]} >= {
        "raw_confidential_data",
        "derived_deidentified_environment",
        "synthetic_variant_rights",
        "robot_eval_rights",
        "commercial_licensing",
        "revenue_share",
        "exclusivity_limits",
    }
    assert rights_ledger["record_count"] == rights_packet["record_count"]
    assert "No live provider jobs" in methodology


def test_robot_eval_dataset_blocks_missing_rights_privacy_and_keeps_ledger_empty(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_eval_inputs(capture_root)

    result = build_real_site_robot_eval_dataset(capture_root=capture_root)
    robot_eval_root = capture_root / "pipeline" / "robot_eval_dataset"
    manifest = json.loads(
        (robot_eval_root / "real_site_robot_eval_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    ledger = json.loads((robot_eval_root / "prediction_outcome_ledger.json").read_text())
    site_card = json.loads((robot_eval_root / "site_card.json").read_text())
    proof_boundaries = json.loads((robot_eval_root / "proof_boundaries.json").read_text())
    robot_pov = json.loads(
        (robot_eval_root / "robot_pov_evidence_requirements.json").read_text()
    )
    human_demo = json.loads(
        (robot_eval_root / "human_demo_evidence_requirements.json").read_text()
    )

    assert result["status"] == "blocked"
    assert "blocked_rights_privacy" in manifest["dataset_statuses"]
    assert manifest["rights_privacy"]["rights_status"] == "missing"
    assert manifest["claim_boundary"]["deployment_outcome_proven"] is False
    assert site_card["geometry"]["collider"]["status"] == "blocked_missing_collider"
    assert site_card["geometry"]["collider"]["collision_ready_claim_allowed"] is False
    assert proof_boundaries["collider_review_input_present"] is False
    assert "collision_ready_claim" in proof_boundaries["blocked_upgrades"]
    assert "action_policy_eval_claim" in proof_boundaries["blocked_upgrades"]
    assert ledger["ledger_status"] == "needs_prediction_sources"
    assert ledger["prediction_sources_supported"] == [
        "marble_review",
        "simready_review",
        "cosmos_preflight",
        "human_eval",
        "future_provider",
        "simulator_trace",
        "robot_trial",
    ]
    assert "robot_pov_video_uri" in robot_pov["required_fields"]
    assert human_demo["claim_boundary"] == "human_demo_is_support_evidence_not_robot_trial"


def test_robot_eval_dataset_scores_recorded_action_trace_fixture_without_claim_upgrade(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_eval_inputs(capture_root)
    _write_review_sources(capture_root)
    _write_recorded_trace_fixture(capture_root)

    result = build_real_site_robot_eval_dataset(capture_root=capture_root)
    robot_eval_root = capture_root / "pipeline" / "robot_eval_dataset"
    manifest = json.loads(
        (robot_eval_root / "real_site_robot_eval_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    proof_boundaries = json.loads((robot_eval_root / "proof_boundaries.json").read_text())
    report = json.loads((robot_eval_root / "recorded_trace_eval_report.json").read_text())
    policy_report = json.loads((robot_eval_root / "policy_eval_report.json").read_text())
    prediction_summary = json.loads(
        (robot_eval_root / "prediction_vs_actual_summary.json").read_text()
    )

    assert result["recorded_trace_eval_status"] == "scored_advisory"
    assert manifest["recorded_trace_eval_status"] == "scored_advisory"
    assert "needs_action_logs" not in manifest["dataset_statuses"]
    assert proof_boundaries["action_logs_present"] is True
    assert proof_boundaries["robot_policy_execution_proven"] is False
    assert report == policy_report
    assert report["status"] == "scored_advisory"
    assert report["attempt_count"] == 1
    assert report["metrics"]["success_rate"] == 0.0
    assert report["metrics"]["mean_cycle_time_seconds"] == 18.25
    assert report["metrics"]["intervention_rate"] == 1.0
    assert report["metrics"]["unsafe_proximity_event_count"] == 1
    assert report["metrics"]["collision_risk_event_count"] == 1
    assert report["metrics"]["object_drop_count"] == 1
    assert report["metrics"]["recovery_success_rate"] == 0.0
    assert report["proof_boundary"]["simulator_execution_proven"] is False
    assert report["proof_boundary"]["robot_policy_execution_proven"] is False
    assert prediction_summary["status"] == "advisory_actuals_ingested"
    assert prediction_summary["actual_record_count"] == 1
    assert prediction_summary["records"][0]["actual_source"] == "recorded_action_trace"
    assert prediction_summary["records"][0]["missed_failures"] == [
        "failure_contact_collision",
        "failure_safety_threshold_violation",
    ]
    assert prediction_summary["claim_boundary"]["public_claim_upgrade_allowed"] is False


def test_robot_eval_dataset_recognizes_robot_team_submission_modalities(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_eval_inputs(capture_root)
    _write_review_sources(capture_root)
    _write_json(
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "robot_team_test_submission_manifest.json",
        {
            "schema_version": "blueprint.robot_team_test_submission.v1",
            "generated_at": "2026-06-03T00:00:00+00:00",
            "modalities": {
                "policy_api_endpoint": {
                    "selected": True,
                    "fields": {
                        "endpointUrl": "https://robot-team.example/policy",
                        "observationSchemaRef": "gs://robot-team/schemas/obs.json",
                        "actionSchemaRef": "gs://robot-team/schemas/action.json",
                    },
                }
            },
        },
    )

    build_real_site_robot_eval_dataset(capture_root=capture_root)
    robot_eval_root = capture_root / "pipeline" / "robot_eval_dataset"
    manifest = json.loads(
        (robot_eval_root / "real_site_robot_eval_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    proof_boundaries = json.loads((robot_eval_root / "proof_boundaries.json").read_text())
    modality_schema = json.loads(
        (robot_eval_root / "robot_team_test_submission_modalities.json").read_text()
    )

    assert "needs_policy_api_endpoint_ref" not in manifest["dataset_statuses"]
    assert "needs_docker_container_ref" in manifest["dataset_statuses"]
    assert proof_boundaries["robot_team_test_submission_refs_present"] is True
    assert modality_schema["source_input_present"] is True
    policy_modality = next(
        item
        for item in modality_schema["modalities"]
        if item["modality_id"] == "policy_api_endpoint"
    )
    assert policy_modality["review_status"] == "reference_present_requires_owner_system_review"
