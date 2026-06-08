from __future__ import annotations

import json
import sys
from pathlib import Path

from blueprint_pipeline.evaluation_prep_stage import robot_eval_job_evaluation_prep_surface
from blueprint_pipeline.robot_eval_job_orchestrator import (
    AgentsSdkRobotEvalJobAdapter,
    FakeRobotEvalJobAgentAdapter,
    build_robot_eval_job,
    run_robot_eval_job_request_inbox,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {"site_identity": {"site_id": "site-1"}},
        },
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "site_identity": {"site_id": "site-1"},
        },
    )
    return capture_root


def _write_robot_eval_cards(
    capture_root: Path,
    *,
    scenario_variation_label: str = "derived",
    rights_blocked: bool = False,
) -> None:
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(
        robot_eval_dir / "site_card.json",
        {
            "schema_version": "real_site_robot_eval_site_card.v0.1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "site_id": "site-1",
            "site_type": "stockroom",
            "geometry": {
                "collider": {
                    "status": "review_input_present",
                    "collision_ready_claim_allowed": False,
                }
            },
            "provenance_rights_review_status": {
                "rights_privacy": {
                    "blocked": rights_blocked,
                    "rights_status": "blocked" if rights_blocked else "verified",
                }
            },
        },
    )
    _write_json(
        robot_eval_dir / "task_cards.json",
        {
            "schema_version": "real_site_robot_eval_task_cards.v0.1",
            "task_card_count": 1,
            "cards": [
                {
                    "task_card_id": "task_card_place_return_in_bin",
                    "task_id": "place_return_in_bin",
                    "task_statement": "Place the return item in the labeled bin",
                    "task_category": "pick_place",
                    "required_metrics": [
                        "cycle_time_seconds",
                        "placement_accuracy",
                        "intervention_rate",
                        "recovery_success",
                    ],
                    "claim_boundary": "task_card_defines_eval_scope_not_robot_execution",
                }
            ],
        },
    )
    _write_json(
        robot_eval_dir / "scenario_cards.json",
        {
            "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
            "scenario_card_count": 1,
            "cards": [
                {
                    "scenario_card_id": "scenario_card_place_return_in_bin_mobile",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "task_id": "place_return_in_bin",
                    "robot_profile_id": "mobile_manipulator_rgb_v1",
                    "normal_scenario": {
                        "statement": "Run the task under the capture-observed layout.",
                        "ground_truth_status": "derived_from_capture_package",
                    },
                    "variation": {
                        "statement": "Run under clutter variation.",
                        "ground_truth_status": f"{scenario_variation_label}_needs_review",
                    },
                    "edge_case": {
                        "statement": "Blocked path near the bin.",
                        "ground_truth_status": "agent_inferred_needs_review",
                    },
                    "observed_vs_inferred_labels": {
                        "layout": "capture_grounded",
                        "variation": scenario_variation_label,
                        "edge_case": "agent_inferred",
                    },
                    "required_missing_annotations": [
                        "needs_robot_pov",
                        "needs_action_logs",
                        "needs_actual_outcome",
                    ],
                    "claim_boundary": "scenario_card_is_review_scope_not_simulator_or_pilot_result",
                }
            ],
        },
    )
    _write_json(
        robot_eval_dir / "eval_cards.json",
        {
            "schema_version": "real_site_robot_eval_eval_cards.v0.1",
            "eval_card_count": 1,
            "cards": [
                {
                    "eval_card_id": "eval_card_place_return_in_bin_marble",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "task_id": "place_return_in_bin",
                    "prediction_source": "marble_review",
                    "engine_used": "hosted visual review",
                    "validation": {"actual_status": "needs_actual_outcome"},
                    "blocked_upgrades": [
                        "simulator_execution_completed",
                        "robot_policy_execution_proven",
                    ],
                    "proof_boundary": "prediction_only_no_actual_outcome_no_deployment_claim",
                }
            ],
        },
    )
    _write_json(
        robot_eval_dir / "proof_boundaries.json",
        {
            "schema_version": "real_site_robot_eval_proof_boundaries.v0.1",
            "simulator_execution_proven": False,
            "physics_contact_validation_proven": False,
            "robot_policy_execution_proven": False,
            "safety_validation_proven": False,
            "real_pilot_outcome_proven": False,
            "generated_scenarios_are_real_world_proof": False,
        },
    )


def _write_fixture_attempts(capture_root: Path, *, success: bool) -> None:
    _write_json(
        capture_root / "pipeline" / "robot_eval_inputs" / "headless_fixture_attempts.json",
        {
            "schema_version": "site_eval_fixture_attempts.v1",
            "attempts": [
                {
                    "attempt_id": "fixture-attempt-1",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "task_id": "place_return_in_bin",
                    "policy_id": "policy-fixture-a",
                    "success": success,
                    "predicted_success": True,
                    "predicted_cycle_time_seconds": 12.5,
                    "predicted_intervention_count": 0,
                    "predicted_safety_event_count": 0,
                    "metrics": {
                        "cycle_time_seconds": 14.0,
                        "intervention_count": 0 if success else 1,
                        "contact_event_count": 0,
                        "safety_event_count": 0 if success else 1,
                    },
                    "failure_mode_ids": []
                    if success
                    else ["failure_navigation_blocked"],
                    "breakage_categories": [] if success else ["blocked_path"],
                    "artifact_paths": {"trace": "fixtures/attempt-1.json"},
                    "owner_system": "BlueprintCapturePipeline.fixture",
                }
            ],
        },
    )


def _full_job_request(
    capture_root: Path,
    *,
    operation: str = "evaluate_only",
    rights_allowed: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": "robot_eval_job_request.v1",
        "customer": {"id": "robot-team-a", "name": "Robot Team A"},
        "site_package": {
            "capture_root": str(capture_root),
            "site_id": "site-1",
            "package_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline",
        },
        "requested_tasks": [
            {
                "task_id": "place_return_in_bin",
                "scenario_ids": ["scenario_place_return_in_bin_mobile"],
            }
        ],
        "robot_profile": {
            "robot_profile_id": "mobile_manipulator_rgb_v1",
            "embodiment": "mobile_manipulator",
            "sensors": ["rgb", "depth"],
        },
        "policy_package": {
            "policy_api_endpoint": {
                "endpoint_url": "https://robot-team.example/policy",
                "observation_schema_ref": "schemas/obs-v1.json",
                "action_schema_ref": "schemas/action-v1.json",
            },
            "docker_container": {
                "image_ref": "registry.example/robot/policy:2026-06-04",
                "digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            },
            "recorded_action_trace": {
                "trace_manifest_uri": "gs://robot-team/traces/trace-manifest.json",
                "checksum": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "timestamp_alignment": "aligned_to_capture_timestamps",
            },
            "high_level_skill_trace": {
                "skill_taxonomy_version": "skills-v1",
                "ordered_skill_sequence": ["navigate", "pick", "place"],
            },
            "teleop_demo": {
                "demo_artifact_uri": "gs://robot-team/demos/demo-1.json",
                "rights_privacy_attestation": "deidentified_operator_approved",
            },
            "sim_controller_plugin": {
                "simulator_framework": "fixture",
                "plugin_uri": "gs://robot-team/plugins/controller-fixture.json",
            },
        },
        "operation": operation,
        "simulator_preference": "fixture",
        "cosmos_training_preference": {"mode": "export_only"},
        "budget": {"budget_usd": 5.0, "timeout_seconds": 30},
        "rights_privacy_scope": {
            "status": "cleared_for_robot_eval" if rights_allowed else "blocked",
            "external_use_allowed": rights_allowed,
            "privacy_scope": "derived_deidentified_environment",
        },
        "owner_system": {"name": "robot-team-a", "request_id": "req-1"},
        "provenance": {
            "submitted_at": "2026-06-04T00:00:00+00:00",
            "timestamp_alignment": "trace_timestamps_aligned_to_capture",
        },
    }


def test_robot_eval_job_fixture_path_runs_end_to_end_without_claim_upgrade(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-fixture-success",
        agent_adapter=FakeRobotEvalJobAgentAdapter(),
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-fixture-success"
    required_outputs = {
        "job_request.json",
        "job_validation.json",
        "job_plan.json",
        "agent_orchestration_plan.json",
        "gpu_provisioning_request.json",
        "gpu_provisioning_result.json",
        "simulator_service_request.json",
        "simulator_service_result.json",
        "policy_package_manifest.json",
        "robot_pov_observation_manifest.json",
        "robot_pov_observations.jsonl",
        "policy_execution_manifest.json",
        "policy_execution_trace.json",
        "policy_execution_trace.jsonl",
        "training_request.json",
        "training_result.json",
        "evaluation_request.json",
        "evaluation_result.json",
        "normalized_attempt_trace.json",
        "failure_labels.json",
        "prediction_outcome_ledger.json",
        "calibration_report.json",
        "breakage_library.json",
        "deployment_outcome_ledger.json",
        "sim_vs_real_calibration_report.json",
        "prediction_vs_actual_deployment_summary.json",
        "post_training_data_package_export_manifest.json",
        "proof_boundary.json",
        "job_run_manifest.json",
    }

    assert result["status"] == "fixture_evaluation_completed"
    assert required_outputs.issubset({path.name for path in job_dir.iterdir()})
    assert not (job_dir / "blocked_manifest.json").exists()

    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    validation = _read_json(job_dir / "job_validation.json")
    provisioning = _read_json(job_dir / "gpu_provisioning_result.json")
    simulator_result = _read_json(job_dir / "simulator_service_result.json")
    evaluation = _read_json(job_dir / "evaluation_result.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    robot_pov = _read_json(job_dir / "robot_pov_observation_manifest.json")
    policy_execution = _read_json(job_dir / "policy_execution_manifest.json")
    deployment = _read_json(job_dir / "deployment_outcome_ledger.json")
    data_package_export = _read_json(job_dir / "post_training_data_package_export_manifest.json")

    assert validation["status"] == "passed"
    assert provisioning["status"] == "allocated"
    assert provisioning["provider"] == "fixture_local"
    assert simulator_result["status"] == "completed"
    assert simulator_result["framework"] == "fixture"
    assert simulator_result["simulator_execution_proven"] is False
    assert evaluation["status"] == "completed"
    assert trace["attempts"][0]["success"] is True
    assert robot_pov["status"] == "completed"
    assert robot_pov["observation_count"] == 1
    assert policy_execution["status"] == "completed"
    assert policy_execution["modality_results"]["high_level_skill_trace"]["attempt_count"] == 1
    assert policy_execution["robot_policy_execution_proven"] is False
    assert deployment["status"] == "blocked_missing_real_world_outcomes"
    assert run_manifest["state"] == "completed"
    assert run_manifest["scene_asset_preflight_status"] == "blocked"
    assert run_manifest["episode_spec_status"] == "compiled_review_required"
    assert run_manifest["cpu_simulator_preflight_status"] == (
        "ready_blocked_optional_dependencies_or_gates"
    )
    assert run_manifest["cpu_preflight_artifacts"]["episode_spec"] == (
        "../simulation_automation/episode_spec.v1.json"
    )
    assert run_manifest["public_claim_upgrade_allowed"] is False
    assert proof_boundary["robot_readiness_proven"] is False
    assert proof_boundary["robot_policy_execution_proven"] is False
    assert proof_boundary["public_claim_upgrade_allowed"] is False
    assert proof_boundary["fixture_only_proof"] is True
    assert data_package_export["status"] == "export_ready_review_required"
    assert data_package_export["package_type"] == "post_training_data_package"
    assert data_package_export["included_artifacts"]["normalized_attempt_trace"] == (
        "normalized_attempt_trace.json"
    )
    assert data_package_export["included_artifacts"]["robot_pov_observation_manifest"] == (
        "robot_pov_observation_manifest.json"
    )
    assert data_package_export["included_artifacts"]["policy_execution_trace"] == (
        "policy_execution_trace.json"
    )
    assert data_package_export["claim_boundary"]["robot_readiness_proven"] is False


def test_robot_eval_job_runs_policy_command_and_pairs_real_world_outcomes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_POLICY_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    _write_json(
        capture_root / "pipeline" / "robot_eval_inputs" / "actual_outcome_manifest.json",
        {
            "schema_version": "actual_outcome_manifest.v1",
            "records": [
                {
                    "outcome_id": "pilot-outcome-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "policy_id": "policy-command",
                    "actual_success": False,
                    "failure_mode_ids": ["failure_collision_risk"],
                    "cycle_time_seconds": 22.0,
                    "intervention_count": 1,
                    "tuning_hours": 3.5,
                    "tuning_iterations": 2,
                    "tuning_notes": ["slowed approach near bin"],
                    "site_modifications": [
                        {"modification": "moved cart 0.5m from approach path"}
                    ],
                    "site_modifications_helped": True,
                    "evidence_refs": {"pilot_log": "file://pilot-log.json"},
                }
            ],
        },
    )
    policy_script = tmp_path / "policy_adapter.py"
    policy_script.write_text(
        "\n".join(
            [
                "import json, os",
                "out = os.environ['BLUEPRINT_POLICY_EXECUTION_OUTPUT']",
                "payload = {",
                "  'attempts': [{",
                "    'attempt_id': 'policy-command-attempt-1',",
                "    'task_id': 'place_return_in_bin',",
                "    'scenario_id': 'scenario_place_return_in_bin_mobile',",
                "    'policy_id': 'policy-command',",
                "    'status': 'completed',",
                "    'success': True,",
                "    'actions': [{'type': 'move_base', 'target': 'bin_approach'}],",
                "    'metrics': {'policy_latency_ms': 42}",
                "  }]",
                "}",
                "open(out, 'w', encoding='utf-8').write(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-policy-and-real-world",
        provisioner="fixture_local",
        simulator="fixture",
        allow_policy_execution=True,
        policy_execution_commands={
            "policy_api_endpoint": f"{sys.executable} {policy_script}",
        },
    )

    job_dir = Path(result["job_dir"])
    policy_execution = _read_json(job_dir / "policy_execution_manifest.json")
    policy_trace = _read_json(job_dir / "policy_execution_trace.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")
    deployment = _read_json(job_dir / "deployment_outcome_ledger.json")
    calibration = _read_json(job_dir / "sim_vs_real_calibration_report.json")
    deployment_summary = _read_json(job_dir / "prediction_vs_actual_deployment_summary.json")
    package = _read_json(job_dir / "post_training_data_package_export_manifest.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert policy_execution["robot_policy_execution_proven"] is True
    assert policy_execution["modality_results"]["policy_api_endpoint"]["status"] == "completed"
    assert policy_trace["attempt_count"] >= 1
    assert proof_boundary["robot_policy_execution_proven"] is True
    assert proof_boundary["real_world_outcome_proven"] is True

    assert deployment["status"] == "completed"
    assert deployment["real_world_outcome_proven"] is True
    assert calibration["status"] == "completed"
    assert calibration["sim_vs_real_calibration_score"] == 0.0
    assert calibration["missed_failure_count"] == 1
    assert calibration["site_modification_count"] == 1
    assert deployment_summary["how_much_real_world_tuning_was_needed"] == {
        "tuning_hours_total": 3.5,
        "tuning_iterations_total": 2,
        "records_with_tuning": 1,
    }
    assert deployment_summary["whether_site_modifications_helped"][0][
        "site_modifications_helped"
    ] is True
    assert package["included_artifacts"]["sim_vs_real_calibration_report"] == (
        "sim_vs_real_calibration_report.json"
    )
    assert package["included_artifacts"]["deployment_outcome_ledger"] == (
        "deployment_outcome_ledger.json"
    )
    assert package["export_policy"]["policy_execution_trace_included"] is True
    assert package["export_policy"]["sim_vs_real_calibration_included"] is True
    assert run_manifest["robot_policy_execution_proven"] is True
    assert run_manifest["real_world_outcome_proven"] is True
    assert run_manifest["robot_readiness_proven"] is False


def test_robot_eval_job_normalizes_command_backed_simulator_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    simulator_script = tmp_path / "pybullet_runner.py"
    simulator_script.write_text(
        "\n".join(
            [
                "import json, os",
                "out = os.environ['BLUEPRINT_SIMULATOR_OUTPUT']",
                "payload = {",
                "  'attempts': [{",
                "    'attempt_id': 'pybullet-attempt-1',",
                "    'task_id': 'place_return_in_bin',",
                "    'scenario_id': 'scenario_place_return_in_bin_mobile',",
                "    'policy_id': 'policy-command',",
                "    'status': 'completed',",
                "    'success': True,",
                "    'metrics': {'cycle_time_seconds': 11.0, 'intervention_count': 0},",
                "    'actions': [{'type': 'move_base', 'target': 'bin_approach'}]",
                "  }]",
                "}",
                "open(out, 'w', encoding='utf-8').write(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )
    request_path = tmp_path / "job-request.json"
    request = _full_job_request(capture_root)
    request["simulator_preference"] = "pybullet"
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-pybullet-command",
        provisioner="fixture_local",
        simulator="pybullet",
        allow_simulator_execution=True,
        allowed_simulators=["pybullet"],
        simulator_commands={"pybullet": f"{sys.executable} {simulator_script}"},
    )

    job_dir = Path(result["job_dir"])
    simulator_result = _read_json(job_dir / "simulator_service_result.json")
    provider_adapter = _read_json(job_dir / "simulator_provider_adapter_manifest.json")
    eval_result = _read_json(job_dir / "evaluation_result.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    prediction = _read_json(job_dir / "prediction_outcome_ledger.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")
    package = _read_json(job_dir / "post_training_data_package_export_manifest.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert result["status"] == "simulator_command_completed"
    assert simulator_result["status"] == "completed"
    assert simulator_result["simulator_execution_proven"] is True
    assert simulator_result["artifact_paths"]["simulator_provider_adapter_manifest"] == (
        "simulator_provider_adapter_manifest.json"
    )
    assert simulator_result["artifact_paths"]["normalized_attempt_trace"] == (
        "normalized_attempt_trace.json"
    )
    assert provider_adapter["schema_version"] == (
        "robot_eval_simulator_provider_adapter_manifest.v1"
    )
    assert provider_adapter["provider_profile"]["provider_family"] == "cpu_physics_engine"
    assert provider_adapter["gates"] == {
        "env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION": True,
        "allow_simulator_execution_flag": True,
        "simulator_allowlisted": True,
        "command_configured": True,
        "blockers": [],
    }
    assert provider_adapter["command_ref"]["configured"] is True
    assert provider_adapter["command_ref"]["sha256"]
    assert provider_adapter["normalization"]["simulator_output_ingested"] is True
    assert eval_result["status"] == "completed"
    assert trace["attempt_count"] == 1
    assert trace["attempts"][0]["engine"] == "pybullet"
    assert prediction["records"][0]["predicted_success"] is True
    assert proof_boundary["simulator_execution_proven"] is True
    assert proof_boundary["robot_readiness_proven"] is False
    assert package["included_artifacts"]["simulator_command_artifacts_manifest"] == (
        "simulator_command_artifacts_manifest.json"
    )
    assert package["included_artifacts"]["simulator_provider_adapter_manifest"] == (
        "simulator_provider_adapter_manifest.json"
    )
    assert package["export_policy"]["simulator_provider_adapter_included"] is True
    assert run_manifest["simulator_execution_proven"] is True
    assert run_manifest["robot_readiness_proven"] is False


def test_robot_eval_job_request_inbox_runs_webapp_job_request_automatically(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    inbox_dir = tmp_path / "webapp-robot-eval-job-requests"
    request = _full_job_request(capture_root)
    request["job_id"] = "webapp-job-1"
    request["source"] = {
        "system": "Blueprint-WebApp",
        "route": "/sites/sw-chi-01",
        "selection_state": {
            "site_slug": "sw-chi-01",
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_mobile",
            "policy_id": "policy-fixture-a",
        },
    }
    _write_json(inbox_dir / "webapp-job-1.json", request)

    result = run_robot_eval_job_request_inbox(
        capture_root=capture_root,
        inbox_dir=inbox_dir,
        agent_adapter=FakeRobotEvalJobAgentAdapter(),
        provisioner="fixture_local",
        simulator="fixture",
    )

    queue_root = capture_root / "pipeline" / "robot_eval_job_requests"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "webapp-job-1"
    queue_manifest = _read_json(queue_root / "inbox_run_manifest.json")
    queued_request = _read_json(queue_root / "webapp-job-1" / "job_request.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert result["schema_version"] == "robot_eval_job_request_inbox_run.v1"
    assert result["status"] == "completed"
    assert result["processed_count"] == 1
    assert result["jobs"][0]["job_id"] == "webapp-job-1"
    assert result["jobs"][0]["status"] == "fixture_evaluation_completed"
    assert queued_request["schema_version"] == "robot_eval_job_request.v1"
    assert queued_request["source"]["system"] == "Blueprint-WebApp"
    assert queue_manifest["processed_count"] == 1
    assert queue_manifest["jobs"][0]["job_run_manifest_uri"].endswith(
        "/pipeline/robot_eval_jobs/webapp-job-1/job_run_manifest.json"
    )
    assert run_manifest["status"] == "fixture_evaluation_completed"
    assert run_manifest["public_claim_upgrade_allowed"] is False


def test_robot_eval_job_request_inbox_accepts_webapp_queue_envelope(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    inbox_dir = tmp_path / "webapp-robot-eval-job-requests"
    request = _full_job_request(capture_root)
    request["job_id"] = "webapp-envelope-job-1"
    request["buyer_request_id"] = "buyer-request-envelope-1"
    envelope = {
        "queue_contract": "robot_eval_job_request_inbox.v1",
        "status": "queued_for_pipeline",
        "queued_at_iso": "2026-06-07T00:00:00Z",
        "job_id": request["job_id"],
        "buyer_request_id": request["buyer_request_id"],
        "pipeline_command": "blueprint-run-robot-eval-job",
        "pipeline_consumer": "BlueprintCapturePipeline",
        "job_request": request,
    }
    _write_json(inbox_dir / "webapp-envelope-job-1.json", envelope)
    (inbox_dir / "._webapp-envelope-job-1.json").write_text("not json", encoding="utf-8")

    result = run_robot_eval_job_request_inbox(
        capture_root=capture_root,
        inbox_dir=inbox_dir,
        agent_adapter=FakeRobotEvalJobAgentAdapter(),
        provisioner="fixture_local",
        simulator="fixture",
    )

    queued_request = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_job_requests"
        / "webapp-envelope-job-1"
        / "job_request.json"
    )

    assert result["status"] == "completed"
    assert result["processed_count"] == 1
    assert result["jobs"][0]["job_id"] == "webapp-envelope-job-1"
    assert queued_request["schema_version"] == "robot_eval_job_request.v1"
    assert "queue_contract" not in queued_request


def test_robot_eval_job_rights_privacy_block_prevents_execution(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root, rights_blocked=True)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root, rights_allowed=False))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-rights-blocked",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-rights-blocked"
    blocked = _read_json(job_dir / "blocked_manifest.json")
    provisioning = _read_json(job_dir / "gpu_provisioning_result.json")
    simulator_result = _read_json(job_dir / "simulator_service_result.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert blocked["status"] == "blocked"
    assert "blocked_rights_privacy" in blocked["blockers"]
    assert blocked["missing_inputs"] == ["rights_privacy_clearance"]
    assert provisioning["status"] == "blocked"
    assert provisioning["execution_performed"] is False
    assert simulator_result["status"] == "blocked"
    assert simulator_result["execution_performed"] is False
    assert run_manifest["state"] == "blocked"
    assert run_manifest["public_claim_upgrade_allowed"] is False


def test_robot_eval_job_missing_policy_evidence_writes_exact_blockers(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request = _full_job_request(capture_root)
    policy = dict(request["policy_package"])  # type: ignore[index]
    policy["teleop_demo"] = {
        "demo_artifact_uri": "gs://robot-team/demos/demo-1.json",
    }
    policy["docker_container"] = {"image_ref": "registry.example/robot/policy:latest"}
    request["policy_package"] = policy
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, request)

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-policy-blocked",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-policy-blocked"
    validation = _read_json(job_dir / "job_validation.json")
    blocked = _read_json(job_dir / "blocked_manifest.json")
    policy_manifest = _read_json(job_dir / "policy_package_manifest.json")

    assert validation["status"] == "blocked"
    assert validation["missing_evidence_statuses"] == [
        "needs_docker_container_ref",
        "needs_teleop_demo_ref",
    ]
    assert blocked["missing_inputs"] == [
        "policy_package.docker_container.digest",
        "policy_package.teleop_demo.rights_privacy_attestation",
    ]
    assert policy_manifest["modalities"]["docker_container"]["status"] == "blocked"
    assert policy_manifest["modalities"]["teleop_demo"]["status"] == "blocked"


def test_robot_eval_job_accepts_one_complete_policy_modality(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request = _full_job_request(capture_root)
    request["policy_package"] = {
        "policy_api_endpoint": {
            "endpoint_url": "https://robot-team.example/policy",
            "observation_schema_ref": "schemas/obs-v1.json",
            "action_schema_ref": "schemas/action-v1.json",
        }
    }
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, request)

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-single-modality-policy",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-single-modality-policy"
    validation = _read_json(job_dir / "job_validation.json")
    policy_manifest = _read_json(job_dir / "policy_package_manifest.json")
    policy_execution = _read_json(job_dir / "policy_execution_manifest.json")

    assert "needs_robot_team_test_modality" not in validation["missing_evidence_statuses"]
    assert "needs_docker_container_ref" not in validation["missing_evidence_statuses"]
    assert policy_manifest["status"] == "review_required"
    assert policy_manifest["selected_modalities"] == ["policy_api_endpoint"]
    assert policy_manifest["modalities"]["docker_container"]["status"] == "not_selected"
    assert policy_manifest["modalities"]["docker_container"]["owner_system_review_required"] is False
    assert policy_execution["selected_modalities"] == ["policy_api_endpoint"]


def test_robot_eval_job_real_provisioner_fails_closed_without_gates(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-real-provisioner-blocked",
        provisioner="vast",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-real-provisioner-blocked"
    result = _read_json(job_dir / "gpu_provisioning_result.json")
    blocked = _read_json(job_dir / "blocked_manifest.json")

    assert result["status"] == "blocked"
    assert result["provider"] == "vast"
    assert result["blockers"] == [
        "missing_env_BLUEPRINT_ALLOW_GPU_PROVISIONING",
        "missing_cli_allow_gpu_provisioning",
    ]
    assert "gpu_provisioning_blocked" in blocked["blockers"]


def test_robot_eval_job_fixture_failure_records_failed_attempt_without_overclaiming(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=False)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-fixture-failure",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-fixture-failure"
    evaluation = _read_json(job_dir / "evaluation_result.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    labels = _read_json(job_dir / "failure_labels.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")

    assert evaluation["status"] == "completed_with_failures"
    assert trace["attempts"][0]["success"] is False
    assert labels["labels"][0]["failure_mode_ids"] == ["failure_navigation_blocked"]
    assert proof_boundary["robot_readiness_proven"] is False


def test_robot_eval_job_generated_scenarios_stay_review_required_until_accepted(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root, scenario_variation_label="agent_inferred")
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-review-required",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-review-required"
    simulator_result = _read_json(job_dir / "simulator_service_result.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    blocked = _read_json(job_dir / "blocked_manifest.json")

    assert simulator_result["status"] == "blocked"
    assert trace["status"] == "blocked"
    assert "generated_or_inferred_scenarios_require_review" in blocked["blockers"]


def test_command_simulator_blocks_without_env_gate(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-command-missing-env",
        provisioner="fixture_local",
        simulator="mujoco",
        allow_simulator_execution=True,
        allowed_simulators=["mujoco"],
        simulator_commands={"mujoco": f"{sys.executable} -c \"print('sim ok')\""},
    )

    result = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-command-missing-env"
        / "simulator_service_result.json"
    )
    provider_adapter = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-command-missing-env"
        / "simulator_provider_adapter_manifest.json"
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"]
    assert result["artifact_paths"]["simulator_provider_adapter_manifest"] == (
        "simulator_provider_adapter_manifest.json"
    )
    assert provider_adapter["status"] == "blocked"
    assert provider_adapter["provider_profile"]["provider_family"] == "cpu_physics_engine"
    assert provider_adapter["gates"] == {
        "env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION": False,
        "allow_simulator_execution_flag": True,
        "simulator_allowlisted": True,
        "command_configured": True,
        "blockers": ["missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"],
    }
    assert provider_adapter["command_ref"]["configured"] is True


def test_isaac_lab_arena_simulator_surfaces_packet_and_blocks_without_env_gate(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-arena-missing-env",
        provisioner="fixture_local",
        simulator="isaac_lab_arena",
        allow_simulator_execution=True,
        allowed_simulators=["isaac_lab_arena"],
        simulator_commands={
            "isaac_lab_arena": f"{sys.executable} -c \"print('arena sim ok')\""
        },
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-arena-missing-env"
    request = _read_json(job_dir / "simulator_service_request.json")
    result = _read_json(job_dir / "simulator_service_result.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    arena_packet = _read_json(
        capture_root / "pipeline" / "simulation_automation" / "arena_environment_packet.json"
    )

    assert request["framework"] == "isaac_lab_arena"
    assert request["arena_environment_packet_path"] == (
        "../simulation_automation/arena_environment_packet.json"
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"]
    assert run_manifest["cpu_preflight_artifacts"]["arena_environment_packet"] == (
        "../simulation_automation/arena_environment_packet.json"
    )
    assert arena_packet["backend"] == "isaac_lab_arena"
    assert arena_packet["simulator_execution_proven"] is False
    assert run_manifest["simulator_execution_proven"] is False
    assert run_manifest["robot_readiness_proven"] is False


def test_command_simulator_blocks_without_cli_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-command-missing-cli",
        provisioner="fixture_local",
        simulator="mujoco",
        allowed_simulators=["mujoco"],
        simulator_commands={"mujoco": f"{sys.executable} -c \"print('sim ok')\""},
    )

    result = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-command-missing-cli"
        / "simulator_service_result.json"
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["missing_cli_allow_simulator_execution"]


def test_command_simulator_blocks_without_command(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-command-missing-command",
        provisioner="fixture_local",
        simulator="mujoco",
        allow_simulator_execution=True,
        allowed_simulators=["mujoco"],
    )

    result = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-command-missing-command"
        / "simulator_service_result.json"
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["missing_simulator_command_mujoco"]


def test_command_simulator_records_stdout_stderr_and_exit_code_when_allowed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-command-allowed",
        provisioner="fixture_local",
        simulator="mujoco",
        allow_simulator_execution=True,
        allowed_simulators=["mujoco"],
        simulator_commands={
            "mujoco": (
                f"{sys.executable} -c \"import sys; "
                "print('sim stdout'); print('sim stderr', file=sys.stderr)\""
            )
        },
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-command-allowed"
    result = _read_json(job_dir / "simulator_service_result.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    assert "sim stdout" in result["stdout"]
    assert "sim stderr" in result["stderr"]
    assert run_manifest["public_claim_upgrade_allowed"] is False


def test_training_request_is_export_only_and_training_result_blocks_without_gates(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root, operation="train_then_evaluate"))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-training-blocked",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-training-blocked"
    request = _read_json(job_dir / "training_request.json")
    result = _read_json(job_dir / "training_result.json")
    blocked = _read_json(job_dir / "blocked_manifest.json")

    assert request["status"] == "export_manifest_only"
    assert result["status"] == "blocked"
    assert result["blockers"] == [
        "missing_env_BLUEPRINT_ALLOW_COSMOS_TRAINING",
        "missing_cli_allow_training",
        "missing_training_command",
    ]
    assert "training_blocked" in blocked["blockers"]


def test_fake_and_agents_sdk_agent_adapters_write_advisory_plans(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-fake-agent",
        agent_adapter=FakeRobotEvalJobAgentAdapter(),
        provisioner="fixture_local",
        simulator="fixture",
    )
    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-agents-sdk-blocked",
        agent_adapter=AgentsSdkRobotEvalJobAdapter(
            agents_sdk_available=False,
            openai_api_key="",
            live_env_allowed=False,
            allow_live_operator=False,
        ),
        provisioner="fixture_local",
        simulator="fixture",
    )

    fake_plan = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-fake-agent"
        / "agent_orchestration_plan.json"
    )
    agents_plan = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-agents-sdk-blocked"
        / "agent_orchestration_plan.json"
    )

    assert fake_plan["status"] == "completed"
    assert fake_plan["decisions"][0]["next_command"] == "validate_job_request"
    assert agents_plan["status"] == "blocked"
    assert agents_plan["blockers"] == [
        "missing_openai_agents_sdk",
        "missing_openai_api_key",
        "missing_cli_allow_live_agent_operator",
        "missing_env_BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS",
    ]
    assert agents_plan["agent_authority"] == "live_operator_when_gated"
    assert agents_plan["proof_booleans_mutable_by_agent"] is False


def test_agents_sdk_robot_eval_live_operator_logs_decisions_without_proof_upgrade(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-agents-sdk-live",
        agent_adapter=AgentsSdkRobotEvalJobAdapter(
            agents_sdk_available=True,
            openai_api_key="sk-test",
            live_env_allowed=True,
            allow_live_operator=True,
            executor=lambda _prompt, _context: {
                "final_output": "Validation passed; run fixture evaluator, then summarize.",
                "commands_chosen": ["run_fixture_evaluation"],
                "tool_call_summaries": [
                    {"tool_name": "read_manifest", "summary": "checked job_validation.json"}
                ],
                "decisions": [
                    {
                        "decision": "run_fixture_evaluation",
                        "summary": "Deterministic validation already passed.",
                    }
                ],
            },
        ),
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-agents-sdk-live"
    agents_plan = _read_json(job_dir / "agent_orchestration_plan.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert agents_plan["status"] == "operator_completed"
    assert agents_plan["execution_performed"] is True
    assert agents_plan["operator_mode"] == "live_operator"
    assert agents_plan["operator_ledger"]["commands_chosen"] == [
        "choose_next_deterministic_robot_eval_command",
        "run_fixture_evaluation",
    ]
    assert agents_plan["operator_ledger"]["tool_call_summaries"][0]["tool_name"] == (
        "read_manifest"
    )
    assert agents_plan["proof_effect"]["direct_proof_booleans_set_true"] == []
    assert run_manifest["agent_operator_mode"] == "live_operator"
    assert run_manifest["robot_readiness_proven"] is False


def test_evaluation_prep_surfaces_robot_eval_job_artifacts_without_overclaiming(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))
    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-surfaced",
        provisioner="fixture_local",
        simulator="fixture",
    )

    eval_dir = capture_root / "pipeline" / "evaluation_prep"
    eval_dir.mkdir(parents=True, exist_ok=True)
    surface = robot_eval_job_evaluation_prep_surface(
        capture_root=capture_root,
        eval_dir=eval_dir,
    )

    assert surface["schema_version"] == "robot_eval_job_evaluation_prep_surface.v1"
    assert surface["status"] == "fixture_evaluation_completed"
    assert surface["public_claim_upgrade_allowed"] is False
    assert surface["simulator_execution_proven"] is False
    assert surface["robot_readiness_proven"] is False
    assert surface["artifacts"]["robot_eval_job_job-surfaced_run_manifest"] == (
        "../robot_eval_jobs/job-surfaced/job_run_manifest.json"
    )
    assert surface["artifact_uris"][
        "robot_eval_job_job-surfaced_run_manifest_uri"
    ].endswith("/pipeline/robot_eval_jobs/job-surfaced/job_run_manifest.json")


def _write_arena_rollout_results(results_dir: Path, *, count: int = 500) -> None:
    video_dir = results_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "episode.mp4").write_bytes(b"fake arena video bytes")
    (results_dir / "stdout.txt").write_text("arena rollout completed\n", encoding="utf-8")
    (results_dir / "stderr.txt").write_text("", encoding="utf-8")
    episodes = []
    for index in range(count):
        success = (index + 1) % 10 != 0
        episodes.append(
            {
                "episode_id": f"episode-{index + 1:04d}",
                "scenario_id": "scenario_place_return_in_bin_mobile",
                "scenario_run_id": f"scenario_place_return_in_bin_mobile__arena_run_{index + 1:04d}",
                "task_id": "place_return_in_bin",
                "shard_id": f"arena_shard_{(index // 125) + 1:04d}",
                "status": "completed" if success else "failed",
                "success": success,
                "failure_reason": None if success else "threshold_miss_timeout",
                "metrics": {
                    "cycle_time_seconds": 12.0 + (index % 7),
                    "placement_accuracy": 1.0 if success else 0.2,
                },
                "start_time_seconds": float(index),
                "end_time_seconds": float(index + 1),
                "video_path": "videos/episode.mp4",
                "stdout_path": "stdout.txt",
                "stderr_path": "stderr.txt",
            }
        )
    _write_json(
        results_dir / "rollout_manifest.json",
        {
            "schema_version": "isaac_lab_arena_rollout_manifest.fixture.v1",
            "episodes": episodes,
        },
    )
    _write_json(
        results_dir / "review_resolutions.json",
        {
            "schema_version": "arena_review_resolutions.fixture.v1",
            "resolutions": [
                {
                    "label_id": "label_arena_attempt_0010",
                    "decision": "accepted",
                    "reviewer": "fixture-reviewer",
                    "evidence_uri": "review://accepted/0010",
                }
            ],
        },
    )


def test_isaac_lab_arena_results_feed_eval_package_and_delivery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    results_dir = tmp_path / "arena-results"
    _write_arena_rollout_results(results_dir)
    request_path = tmp_path / "job-request.json"
    request = _full_job_request(capture_root)
    request["simulator_preference"] = "isaac_lab_arena"
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-arena-ingest",
        provisioner="fixture_local",
        simulator="isaac_lab_arena",
        arena_results_dir=results_dir,
        arena_scenario_count=500,
        arena_shard_size=125,
        arena_num_envs=32,
        arena_retry_budget=3,
        arena_operator_mode="fake",
    )

    job_dir = Path(result["job_dir"])
    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    simulator_result = _read_json(job_dir / "simulator_service_result.json")
    eval_result = _read_json(job_dir / "evaluation_result.json")
    schedule = _read_json(job_dir / "arena_eval_schedule.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    labels = _read_json(job_dir / "failure_labels.json")
    clips = _read_json(job_dir / "clips_manifest.json")
    package = _read_json(job_dir / "post_training_data_package_export_manifest.json")
    archive = _read_json(job_dir / "archive_manifest.json")
    delivery = _read_json(job_dir / "delivery_manifest.json")
    operators = _read_json(job_dir / "live_operator_ledger.json")

    assert result["status"] == "completed_with_failures"
    assert simulator_result["status"] == "completed_from_supplied_arena_results"
    assert simulator_result["simulator_execution_proven"] is False
    assert eval_result["status"] == "completed_with_failures"
    assert run_manifest["arena_result_ingest_status"] == "completed"
    assert run_manifest["simulator_execution_proven"] is False
    assert run_manifest["robot_readiness_proven"] is False

    assert schedule["scenario_count"] == 500
    assert schedule["shard_count"] == 4
    assert schedule["num_envs"] == 32
    assert trace["attempt_count"] == 500
    assert labels["label_count"] == 50
    assert clips["clip_count"] == 500

    assert (job_dir / "arena_eval_metrics.json").is_file()
    assert (job_dir / "dataset_card.json").is_file()
    assert (job_dir / "license_manifest.json").is_file()
    assert (job_dir / "checksums.json").is_file()
    assert package["status"] == "export_ready_review_required"
    assert package["archive_manifest_path"] == "archive_manifest.json"
    assert archive["archive"]["exists"] is True
    assert delivery["status"] == "local_delivery_bundle_ready"
    assert operators["status"] == "completed"
    assert operators["agents_sdk_operator_performed"] is True
    assert operators["codex_sdk_operator_performed"] is True
    assert operators["public_claim_upgrade_allowed"] is False
