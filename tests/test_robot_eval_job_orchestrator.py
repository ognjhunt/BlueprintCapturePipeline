from __future__ import annotations

import json
import sys
from pathlib import Path

from blueprint_pipeline.evaluation_prep_stage import robot_eval_job_evaluation_prep_surface
from blueprint_pipeline.robot_eval_job_orchestrator import (
    AgentsSdkRobotEvalJobAdapter,
    FakeRobotEvalJobAgentAdapter,
    build_robot_eval_job,
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
        "training_request.json",
        "training_result.json",
        "evaluation_request.json",
        "evaluation_result.json",
        "normalized_attempt_trace.json",
        "failure_labels.json",
        "prediction_outcome_ledger.json",
        "calibration_report.json",
        "breakage_library.json",
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

    assert validation["status"] == "passed"
    assert provisioning["status"] == "allocated"
    assert provisioning["provider"] == "fixture_local"
    assert simulator_result["status"] == "completed"
    assert simulator_result["framework"] == "fixture"
    assert simulator_result["simulator_execution_proven"] is False
    assert evaluation["status"] == "completed"
    assert trace["attempts"][0]["success"] is True
    assert run_manifest["state"] == "completed"
    assert run_manifest["public_claim_upgrade_allowed"] is False
    assert proof_boundary["robot_readiness_proven"] is False
    assert proof_boundary["public_claim_upgrade_allowed"] is False
    assert proof_boundary["fixture_only_proof"] is True


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
    policy.pop("teleop_demo")
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
        "policy_package.teleop_demo",
    ]
    assert policy_manifest["modalities"]["docker_container"]["status"] == "blocked"
    assert policy_manifest["modalities"]["teleop_demo"]["status"] == "blocked"


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
    assert result["status"] == "blocked"
    assert result["blockers"] == ["missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"]


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
            env_gate_allowed=False,
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
        "missing_env_BLUEPRINT_ALLOW_AGENTS_SDK_JOB_ORCHESTRATION",
    ]
    assert agents_plan["agent_authority"] == "advisory_only"


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
