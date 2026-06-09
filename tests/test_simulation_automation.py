from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.episode_spec import FakeEpisodeSpecAgentAdapter
from blueprint_pipeline.simulation_automation import (
    AgentsSdkCodexMCPAdapter,
    CodexSdkSimulationAutomationAgentAdapter,
    FakeSimulationAutomationAgentAdapter,
    build_simulation_automation,
    validate_owner_gpu_system_proof,
)
from blueprint_pipeline.evaluation_prep_stage import simulation_automation_evaluation_prep_surface


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
        {"scene_id": "scene-1", "capture_id": "capture-1", "site_identity": {"site_id": "site-1"}},
    )
    return capture_root


def _write_worldlabs_and_marble_artifacts(capture_root: Path) -> None:
    pipeline_dir = capture_root / "pipeline"
    _write_json(
        pipeline_dir / "worldlabs_request_manifest.json",
        {
            "schema_version": "v1",
            "provider_name": "world_labs",
            "provider_model": "marble-1.1",
            "selected_video_uri": "gs://local-blueprint/privacy/final_walkthrough.mov",
            "privacy_safe_input": True,
        },
    )
    _write_json(
        pipeline_dir / "worldlabs_world_manifest.json",
        {
            "schema_version": "worldlabs_world_manifest.v1",
            "world_id": "world-1",
            "world_marble_url": "https://marble.worldlabs.ai/worlds/world-1",
            "model": "marble-1.1",
            "updated_at": "2026-06-03T00:00:00Z",
            "assets": {
                "mesh": {"collider_mesh_url": "https://cdn.worldlabs.ai/world-1/collider.glb"},
                "splats": {
                    "spz_urls": {"full": "https://cdn.worldlabs.ai/world-1/full.spz"},
                    "semantics_metadata": {
                        "metric_scale_factor": 0.5,
                        "ground_plane_offset": 1.0,
                    },
                },
            },
        },
    )
    _write_json(
        pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json",
        {
            "schema_version": "marble_simready_bridge.v1",
            "status": "review_ready_with_conversion_required",
            "world_id": "world-1",
            "simulator_review_manifests": {
                "isaac_sim": "simulators/isaac_sim_review_manifest.json",
                "mujoco": "simulators/mujoco_review_manifest.json",
                "pybullet": "simulators/pybullet_review_manifest.json",
            },
            "evaluation_prep_summary": {
                "collider_mesh_available": True,
                "metric_alignment_ready": True,
                "robot_readiness_proven": False,
            },
        },
    )
    _write_json(
        pipeline_dir / "marble_sim_assets" / "marble_asset_validation.json",
        {
            "schema_version": "marble_asset_validation.v1",
            "overall_status": "review_ready_with_conversion_required",
            "physics_collision_review_ready": True,
            "isaac_visual_conversion_required": True,
            "robot_readiness_proven": False,
        },
    )
    _write_json(
        pipeline_dir / "simready" / "simready_scene_manifest.json",
        {
            "schema_version": "simready_scene_manifest.v1",
            "status": "prepared_for_review",
            "framework_artifacts": {
                "isaac_sim": {"path": "isaac_sim/site_scene.usda", "load_status": "not_executed"},
                "mujoco": {"path": "mujoco/site_scene.xml", "load_status": "not_executed"},
                "pybullet": {"path": "pybullet/site_scene.urdf", "load_status": "not_executed"},
            },
            "claim_boundary": {
                "simulator_execution_proven": False,
                "robot_readiness_proven": False,
            },
        },
    )
    _write_json(
        pipeline_dir / "simready" / "simready_validation.json",
        {
            "schema_version": "simready_validation.v1",
            "overall_status": "prepared_for_review",
            "claim_boundary": {
                "simulator_execution_proven": False,
                "robot_readiness_proven": False,
            },
        },
    )
    _write_json(
        pipeline_dir / "cosmos_training_export" / "manifest.json",
        {
            "schema_version": "v1",
            "status": "ready",
            "source_mode": "capture_grounded_fixture",
            "trainer_config_path": "trainer_config.json",
        },
    )


def _write_robot_eval_cards(capture_root: Path) -> None:
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(
        robot_eval_dir / "site_card.json",
        {
            "schema_version": "real_site_robot_eval_site_card.v0.1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "site_id": "site-1",
            "site_type": "stockroom",
        },
    )
    _write_json(
        robot_eval_dir / "task_cards.json",
        {
            "schema_version": "real_site_robot_eval_task_cards.v0.1",
            "cards": [
                {
                    "task_id": "place_return_in_bin",
                    "task_statement": "Place the return item in the labeled bin",
                    "task_category": "pick_place",
                    "required_metrics": [
                        "success_rate",
                        "cycle_time",
                        "intervention_rate",
                        "unsafe_proximity",
                        "collision_risk",
                        "object_drop",
                        "wrong_object",
                        "timeout",
                        "recovery_success",
                        "world_model_uncertainty",
                        "sim_vs_real_calibration_score",
                        "placement_accuracy",
                    ],
                }
            ],
        },
    )
    _write_json(
        robot_eval_dir / "scenario_cards.json",
        {
            "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
            "cards": [
                {
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "task_id": "place_return_in_bin",
                    "robot_profile_id": "mobile_manipulator_rgbd_fixture",
                    "normal_scenario": {"statement": "Use the capture-observed layout."},
                    "variation": {"statement": "Add review-required clutter variation."},
                    "observed_vs_inferred_labels": {"layout": "capture_grounded"},
                }
            ],
        },
    )
    _write_json(
        robot_eval_dir / "eval_cards.json",
        {
            "schema_version": "real_site_robot_eval_eval_cards.v0.1",
            "cards": [
                {
                    "eval_card_id": "eval_card_place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "task_id": "place_return_in_bin",
                    "prediction_source": "hosted_review",
                    "validation": {"actual_status": "needs_actual_outcome"},
                    "blocked_upgrades": ["simulator_execution_completed"],
                }
            ],
        },
    )
    _write_json(
        robot_eval_dir / "proof_boundaries.json",
        {
            "schema_version": "real_site_robot_eval_proof_boundaries.v0.1",
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
        },
    )


def _write_scenario_family_library(capture_root: Path) -> None:
    _write_json(
        capture_root / "pipeline" / "robot_eval_dataset" / "scenario_family_library.json",
        {
            "schema_version": "scenario_family_library.v1",
            "families": [
                {
                    "family_id": "family_place_return_in_bin_robustness",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "scenario_family": "stockroom_pick_place_robustness",
                    "variations": [
                        {"variation_id": "lighting_variation", "variation_name": "lighting variation"},
                        {"variation_id": "object_rotation", "variation_name": "object rotation"},
                        {"variation_id": "cart_shifted", "variation_name": "cart shifted"},
                        {"variation_id": "blocked_path", "variation_name": "blocked path"},
                        {"variation_id": "human_crossing", "variation_name": "human crossing"},
                        {"variation_id": "forklift_nearby", "variation_name": "forklift nearby"},
                        {"variation_id": "occlusion", "variation_name": "occlusion"},
                        {"variation_id": "glare", "variation_name": "glare"},
                        {"variation_id": "missing_label", "variation_name": "missing label"},
                        {"variation_id": "wrong_object_nearby", "variation_name": "wrong object nearby"},
                        {
                            "variation_id": "narrow_approach_angle",
                            "variation_name": "narrow approach angle",
                        },
                    ],
                }
            ],
        },
    )


def _write_valid_owner_gpu_proof(capture_root: Path) -> Path:
    automation_root = capture_root / "pipeline" / "simulation_automation"
    proof_root = automation_root / "owner_gpu_proof"
    proof_root.mkdir(parents=True, exist_ok=True)
    stdout = proof_root / "owner_simulator_stdout.log"
    stderr = proof_root / "owner_simulator_stderr.log"
    scene_load = proof_root / "owner_scene_load_trace.json"
    spawn_trace = proof_root / "owner_spawn_pose_trace.json"
    action_trace = proof_root / "owner_action_policy_trace.json"
    artifact_manifest = proof_root / "owner_artifact_manifest.json"
    stdout.write_text("loaded scene\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    _write_json(
        scene_load,
        {
            "status": "loaded",
            "scene_loaded": True,
            "simulator_backend": "isaac_sim",
            "scene_asset": "worldlabs_collider.glb",
        },
    )
    _write_json(
        spawn_trace,
        {
            "status": "validated",
            "spawn_pose_loaded": True,
            "spawn_pose_id": "spawn-1",
        },
    )
    _write_json(
        action_trace,
        {
            "status": "completed",
            "actions": [{"t": 0.0, "action": "noop"}],
            "policy_id": "owner-policy-a",
        },
    )
    _write_json(
        artifact_manifest,
        {
            "status": "complete",
            "artifacts": [{"kind": "scene_load_trace", "path": str(scene_load)}],
        },
    )
    proof_path = automation_root / "gpu_owner_system_proof.json"
    _write_json(
        proof_path,
        {
            "schema_version": "gpu_owner_system_proof.v1",
            "owner_system_id": "owner-system-a",
            "simulator_backend": "isaac_sim",
            "simulator_version": "2026.1",
            "gpu_model": "RTX-6000",
            "command": "isaac-sim --headless --scene worldlabs_collider.glb",
            "started_at": "2026-06-06T10:00:00Z",
            "completed_at": "2026-06-06T10:04:00Z",
            "exit_code": 0,
            "stdout_uri_or_path": str(stdout),
            "stderr_uri_or_path": str(stderr),
            "scene_load_trace_uri_or_path": str(scene_load),
            "spawn_pose_validation_uri_or_path": str(spawn_trace),
            "action_or_policy_trace_uri_or_path": str(action_trace),
            "artifact_manifest_uri_or_path": str(artifact_manifest),
            "pass_fail_criteria": {"passed": True},
            "operator_attestation": {
                "attested_by": "owner-operator-a",
                "attestation": "Owner system ran the simulator and captured these artifacts.",
            },
        },
    )
    return proof_path


def test_owner_gpu_proof_ingestion_validates_required_artifacts_without_robot_claim(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_worldlabs_and_marble_artifacts(capture_root)
    proof_path = _write_valid_owner_gpu_proof(capture_root)

    validation = validate_owner_gpu_system_proof(proof_path=proof_path, capture_root=capture_root)
    result = build_simulation_automation(capture_root=capture_root)

    automation_root = capture_root / "pipeline" / "simulation_automation"
    proof_manifest = _read_json(automation_root / "owner_gpu_simulator_execution_proof_manifest.json")
    gpu_handoff = _read_json(automation_root / "gpu_handoff_packet.json")
    proof_boundary = _read_json(automation_root / "proof_boundary.json")
    run_manifest = _read_json(automation_root / "simulation_automation_run_manifest.json")

    assert validation["status"] == "accepted"
    assert proof_manifest["status"] == "accepted"
    assert proof_manifest["owner_gpu_simulator_execution_proven"] is True
    assert proof_manifest["robot_readiness_proven"] is False
    assert gpu_handoff["owner_gpu_simulator_execution_proven"] is True
    assert "owner_gpu_simulator_execution_not_run" not in gpu_handoff["blockers"]
    assert proof_boundary["simulator_execution_proven"] is True
    assert proof_boundary["robot_readiness_proven"] is False
    assert run_manifest["owner_gpu_simulator_execution_proven"] is True
    assert result["claim_boundary"]["robot_readiness_proven"] is False


def test_simulation_automation_default_is_local_only_and_blocked(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_worldlabs_and_marble_artifacts(capture_root)
    _write_robot_eval_cards(capture_root)

    result = build_simulation_automation(
        capture_root=capture_root,
        agent_adapter=FakeSimulationAutomationAgentAdapter(),
    )

    automation_root = capture_root / "pipeline" / "simulation_automation"
    plan = _read_json(automation_root / "simulation_automation_plan.json")
    run_manifest = _read_json(automation_root / "simulation_automation_run_manifest.json")
    conversion = _read_json(automation_root / "asset_conversion_plan.json")
    simulator_execution = _read_json(automation_root / "simulator_execution_manifest.json")
    engine_registry = _read_json(automation_root / "simulator_engine_plugin_registry.json")
    training = _read_json(automation_root / "training_orchestration_manifest.json")
    proof_boundary = _read_json(automation_root / "proof_boundary.json")
    agent_ledger = _read_json(automation_root / "agent_decision_ledger.json")
    arena_packet = _read_json(automation_root / "arena_environment_packet.json")
    gpu_handoff = _read_json(automation_root / "gpu_handoff_packet.json")
    proof_schema = _read_json(automation_root / "gpu_owner_system_proof_schema.json")
    owner_blocked = _read_json(
        automation_root / "owner_gpu_simulator_execution_blocked_manifest.json"
    )

    assert result["status"] == "blocked"
    assert (automation_root / "scene_asset_inspection.json").is_file()
    assert (automation_root / "scene_asset_inventory.json").is_file()
    assert (automation_root / "scene_asset_dependency_audit.json").is_file()
    assert (automation_root / "collider_proxy_plan.json").is_file()
    assert (automation_root / "spawn_pose_validation_manifest.json").is_file()
    assert (automation_root / "cpu_preflight_manifest.json").is_file()
    assert (automation_root / "pre_gpu_readiness_summary.json").is_file()
    assert (automation_root / "gpu_run_checklist.md").is_file()
    assert (automation_root / "arena_environment_packet.json").is_file()
    assert (automation_root / "episode_spec.v1.json").is_file()
    assert (automation_root / "episode_setup_manifest.json").is_file()
    assert (automation_root / "cpu_simulator_preflight_manifest.json").is_file()
    assert plan["source_artifacts"]["worldlabs_world_manifest"].endswith(
        "../worldlabs_world_manifest.json"
    )
    assert plan["source_artifacts"]["marble_simready_bridge"].endswith(
        "../marble_sim_assets/marble_simready_bridge.json"
    )
    assert plan["world_model_sources"]["worldlabs"]["world_id"] == "world-1"
    assert conversion["frameworks"]["isaac_sim"]["status"] == "planned_requires_conversion"
    assert conversion["frameworks"]["isaac_lab_arena"]["status"] == (
        "planned_requires_owner_asset_mapping"
    )
    assert conversion["frameworks"]["mujoco"]["status"] == "planned_requires_conversion"
    assert conversion["frameworks"]["pybullet"]["status"] == "planned_requires_conversion"
    assert conversion["frameworks"]["newton"]["status"] == "planned_requires_conversion"
    assert simulator_execution["overall_status"] == "blocked"
    assert engine_registry["schema_version"] == "simulator_engine_plugin_registry.v1"
    assert set(engine_registry["engine_targets"]) == {
        "isaac_sim",
        "isaac_lab_arena",
        "mujoco",
        "pybullet",
        "newton",
    }
    assert set(engine_registry["world_model_engine_targets"]) == {
        "worldlabs_world_model",
        "marble_simready",
        "cosmos_predict",
        "native_site_reference",
    }
    assert set(engine_registry["plugins"]) == set(engine_registry["engine_targets"])
    assert set(engine_registry["world_model_plugins"]) == set(
        engine_registry["world_model_engine_targets"]
    )
    for plugin in engine_registry["plugins"].values():
        assert plugin["adapter_contract_status"] == "ready"
        assert plugin["managed_execution_supported"] is True
        assert plugin["execution_manager"]["status"] == "gated_waiting_for_owner_runtime"
        assert plugin["inputs"]["scenario_variation_instances"] == (
            "scenario_variation_instances.json"
        )
        assert plugin["outputs_expected"]["normalized_attempt_trace"] == (
            "normalized_attempt_trace.json"
        )
        assert plugin["proof_boundary"]["simulator_execution_proven"] is False
    for plugin in engine_registry["world_model_plugins"].values():
        assert plugin["adapter_contract_status"] == "ready"
        assert plugin["managed_execution_supported"] is True
        assert plugin["runtime_kind"] == "world_model_support_engine"
        assert plugin["inputs"]["scenario_variation_instances"] == (
            "scenario_variation_instances.json"
        )
        assert plugin["outputs_expected"]["uncertainty_summary"].endswith(
            "/world_model_uncertainty.json"
        )
        assert plugin["proof_boundary"]["world_model_support_assets_generated"] is False
        assert plugin["proof_boundary"]["robot_readiness_proven"] is False
    assert {
        record["framework"]: record["status"]
        for record in simulator_execution["simulator_results"]
    } == {
        "isaac_sim": "blocked",
        "isaac_lab_arena": "blocked",
        "mujoco": "blocked",
        "pybullet": "blocked",
        "newton": "blocked",
    }
    assert all(
        record["reason"] == "approval_required"
        for record in simulator_execution["simulator_results"]
    )
    assert training["status"] == "blocked"
    assert training["reason"] == "approval_required"
    assert training["runner"] == "blueprint_pipeline.synthesis.cosmos_lora_training.run_cosmos_lora_training"
    assert proof_boundary["simulator_execution_proven"] is False
    assert proof_boundary["robot_readiness_proven"] is False
    assert proof_boundary["training_proof"]["training_completed"] is False
    assert proof_boundary["public_claim_upgrade_allowed"] is False
    assert arena_packet["schema_version"] == "arena_environment_packet.v1"
    assert arena_packet["backend"] == "isaac_lab_arena"
    assert arena_packet["status"] == "ready_for_owner_arena_pack_review"
    assert arena_packet["arena_components"]["scene"]["scene_id"] == "scene-1"
    assert arena_packet["arena_components"]["tasks"][0]["task_id"] == "place_return_in_bin"
    assert arena_packet["arena_components"]["scenarios"][0]["scenario_id"] == (
        "scenario_place_return_in_bin_mobile"
    )
    assert arena_packet["arena_components"]["eval_bindings"][0]["task_id"] == (
        "place_return_in_bin"
    )
    assert arena_packet["arena_components"]["episode_bindings"][0]["arena_builder_target"] == (
        "IsaacLabArenaEnvironment"
    )
    assert arena_packet["simulator_execution_proven"] is False
    assert arena_packet["robot_readiness_proven"] is False
    assert "owner_gpu_simulator_execution_not_run" in arena_packet["blockers"]
    assert gpu_handoff["owner_gpu_simulator_execution_proven"] is False
    assert "owner_gpu_simulator_execution_not_run" in gpu_handoff["blockers"]
    assert gpu_handoff["arena_package"]["path"] == "arena_environment_packet.json"
    assert gpu_handoff["arena_package"]["status"] == "ready_for_owner_arena_pack_review"
    assert "isaac_lab_arena" in gpu_handoff["target_backend_guidance"]
    assert "gpu_owner_system_proof.json" in gpu_handoff["output_artifacts_expected"]
    assert "owner_system_id" in proof_schema["required_fields"]
    assert owner_blocked["blocker_id"] == "owner_gpu_simulator_execution_not_run"
    assert owner_blocked["disallowed_workaround"].startswith("Do not mark simulator")
    assert run_manifest["live_provider_calls_performed"] is False
    assert run_manifest["remote_asset_downloads_performed"] is False
    assert run_manifest["scene_asset_preflight_status"] == "blocked"
    assert run_manifest["episode_spec_status"] == "compiled_review_required"
    assert run_manifest["cpu_simulator_preflight_status"] == (
        "ready_blocked_optional_dependencies_or_gates"
    )
    assert run_manifest["gpu_handoff_packet_path"] == "gpu_handoff_packet.json"
    assert run_manifest["arena_environment_packet_path"] == "arena_environment_packet.json"
    assert run_manifest["arena_environment_packet_status"] == (
        "ready_for_owner_arena_pack_review"
    )
    assert run_manifest["owner_gpu_simulator_execution_proven"] is False
    assert run_manifest["simulators_run"] is False
    assert run_manifest["gpu_training_run"] is False
    assert agent_ledger["adapter"] == "fake"
    assert agent_ledger["decisions"][0]["decision"] == "plan_next_actions"


def test_simulation_automation_instantiates_scenario_variations_for_all_engine_targets(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_worldlabs_and_marble_artifacts(capture_root)
    _write_robot_eval_cards(capture_root)
    _write_scenario_family_library(capture_root)

    build_simulation_automation(capture_root=capture_root)

    automation_root = capture_root / "pipeline" / "simulation_automation"
    variation_instances = _read_json(automation_root / "scenario_variation_instances.json")
    arena_packet = _read_json(automation_root / "arena_environment_packet.json")

    required_names = {
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
    expected_mutation_fields = {
        "lighting_variation": "lighting",
        "object_rotation": "object_pose_delta",
        "cart_shifted": "cart_pose_delta",
        "blocked_path": "path_obstacle",
        "human_crossing": "dynamic_actor",
        "forklift_nearby": "forklift_actor",
        "occlusion": "occluder",
        "glare": "glare_source",
        "missing_label": "label_visibility",
        "wrong_object_nearby": "distractor_object",
        "narrow_approach_angle": "approach_constraint",
    }

    assert variation_instances["schema_version"] == "scenario_variation_instances.v1"
    assert set(variation_instances["required_variation_names"]) == required_names
    assert set(variation_instances["variation_names_instantiated"]) == required_names
    assert variation_instances["instance_count"] == len(required_names)
    assert set(variation_instances["engine_targets"]) == {
        "isaac_sim",
        "isaac_lab_arena",
        "mujoco",
        "pybullet",
        "newton",
    }
    assert set(variation_instances["engine_mutation_plan"]) == {
        "isaac_sim",
        "isaac_lab_arena",
        "mujoco",
        "pybullet",
        "newton",
    }
    assert all(
        plan["mutation_count"] == len(required_names)
        and plan["status"] == "ready_for_owner_engine_adapter"
        for plan in variation_instances["engine_mutation_plan"].values()
    )
    for instance in variation_instances["instances"]:
        variation_name = instance["variation_name"]
        assert expected_mutation_fields[variation_name] in instance["concrete_mutation"]
        assert set(instance["engine_mutations"]) == set(variation_instances["engine_targets"])
        assert all(
            mutation["operation_count"] >= 1
            for mutation in instance["engine_mutations"].values()
        )

    assert arena_packet["source_artifacts"]["scenario_variation_instances"] == (
        "scenario_variation_instances.json"
    )
    scenario_component = arena_packet["arena_components"]["scenarios"][0]
    assert set(scenario_component["scenario_variation_instance_ids"]) == {
        instance["instance_id"] for instance in variation_instances["instances"]
    }
    episode_binding = arena_packet["arena_components"]["episode_bindings"][0]
    assert set(episode_binding["scenario_variation_instance_ids"]) == {
        instance["instance_id"] for instance in variation_instances["instances"]
    }
    assert episode_binding["engine_mutation_plan_path"] == "scenario_variation_instances.json"


def test_missing_simulator_dependency_produces_blocked_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_worldlabs_and_marble_artifacts(capture_root)
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")

    build_simulation_automation(
        capture_root=capture_root,
        allow_simulator_execution=True,
        allowed_simulators=["mujoco"],
        simulator_commands={"mujoco": "definitely-missing-blueprint-mujoco"},
    )

    result_path = (
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "simulators"
        / "mujoco_result.json"
    )
    mujoco_result = _read_json(result_path)

    assert mujoco_result["framework"] == "mujoco"
    assert mujoco_result["status"] == "blocked"
    assert mujoco_result["reason"] == "missing_dependency"
    assert mujoco_result["blocked_manifest"] == str(result_path)
    assert mujoco_result["command"] == ["definitely-missing-blueprint-mujoco"]


def test_fake_agent_adapter_can_plan_and_diagnose_without_network(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_worldlabs_and_marble_artifacts(capture_root)
    adapter = FakeSimulationAutomationAgentAdapter()

    build_simulation_automation(
        capture_root=capture_root,
        agent_adapter=adapter,
        episode_agent_adapter=FakeEpisodeSpecAgentAdapter(),
    )

    automation_root = capture_root / "pipeline" / "simulation_automation"
    ledger = _read_json(automation_root / "agent_decision_ledger.json")
    proposals = _read_json(automation_root / "agent_episode_spec_proposals.json")
    assert ledger["adapter"] == "fake"
    assert ledger["network_required"] is False
    assert ledger["decisions"][0]["summary"] == (
        "Use deterministic manifests; keep simulator and training execution blocked until explicit approvals and dependencies exist."
    )
    assert ledger["diagnostics"][0]["status"] == "blocked"
    assert "approval_required" in ledger["diagnostics"][0]["blockers"]
    assert proposals["adapter"] == "fake"
    assert proposals["agent_authority"] == "review_input_proposal_operator"
    assert proposals["proof_booleans_mutable_by_agent"] is False
    assert proposals["proposal_count"] == 1


def test_simulation_live_sdk_operators_log_commands_without_proof_mutation(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_worldlabs_and_marble_artifacts(capture_root)

    build_simulation_automation(
        capture_root=capture_root,
        agent_adapter=AgentsSdkCodexMCPAdapter(
            agents_sdk_available=True,
            openai_api_key="sk-test",
            live_env_allowed=True,
            allow_live_operator=True,
            executor=lambda _prompt, _context: {
                "final_output": "Retry CPU preflight, then summarize owner GPU blockers.",
                "commands_chosen": ["blueprint-run-cpu-simulator-preflight"],
                "tool_call_summaries": [
                    {"tool_name": "read_manifest", "summary": "read cpu preflight status"}
                ],
            },
        ),
    )

    automation_root = capture_root / "pipeline" / "simulation_automation"
    agents_ledger = _read_json(automation_root / "agent_decision_ledger.json")
    run_manifest = _read_json(automation_root / "simulation_automation_run_manifest.json")
    assert agents_ledger["status"] == "operator_completed"
    assert agents_ledger["operator_mode"] == "live_operator"
    assert agents_ledger["operator_ledger"]["commands_chosen"] == [
        "choose_next_simulation_automation_command",
        "blueprint-run-cpu-simulator-preflight",
    ]
    assert agents_ledger["proof_effect"]["direct_proof_booleans_set_true"] == []
    assert run_manifest["agent_operator_mode"] == "live_operator"
    assert run_manifest["robot_readiness_proven"] is False

    build_simulation_automation(
        capture_root=capture_root,
        agent_adapter=CodexSdkSimulationAutomationAgentAdapter(
            codex_sdk_available=True,
            openai_api_key="sk-test",
            live_env_allowed=True,
            allow_live_operator=True,
            executor=lambda _prompt, _context: {
                "final_output": "Patched failing conversion-plan test and reran it.",
                "commands_chosen": ["pytest tests/test_simulation_automation.py"],
                "tool_call_summaries": [
                    {"tool_name": "apply_patch", "summary": "conversion-plan assertion fix"}
                ],
            },
        ),
    )

    codex_ledger = _read_json(automation_root / "agent_decision_ledger.json")
    assert codex_ledger["status"] == "operator_completed"
    assert codex_ledger["operator_mode"] == "live_operator"
    assert codex_ledger["operator_ledger"]["commands_chosen"] == [
        "diagnose_patch_and_test_simulation_automation",
        "pytest tests/test_simulation_automation.py",
    ]
    assert codex_ledger["operator_ledger"]["tool_call_summaries"][0]["tool_name"] == (
        "apply_patch"
    )
    assert codex_ledger["proof_effect"]["proof_booleans_mutable_by_agent"] is False


def test_evaluation_prep_surfaces_simulation_automation_artifacts_without_overclaiming(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_worldlabs_and_marble_artifacts(capture_root)
    build_simulation_automation(capture_root=capture_root)

    eval_dir = capture_root / "pipeline" / "evaluation_prep"
    eval_dir.mkdir(parents=True, exist_ok=True)
    surface = simulation_automation_evaluation_prep_surface(
        capture_root=capture_root,
        eval_dir=eval_dir,
    )

    assert surface["status"] == "blocked"
    assert surface["simulator_execution_proven"] is False
    assert surface["robot_readiness_proven"] is False
    expected_artifacts = {
        "simulation_automation_plan": "../simulation_automation/simulation_automation_plan.json",
        "simulation_automation_run_manifest": "../simulation_automation/simulation_automation_run_manifest.json",
        "asset_conversion_plan": "../simulation_automation/asset_conversion_plan.json",
        "simulator_execution_manifest": "../simulation_automation/simulator_execution_manifest.json",
        "training_orchestration_manifest": "../simulation_automation/training_orchestration_manifest.json",
        "robot_eval_scenario_variation_instances": (
            "../simulation_automation/scenario_variation_instances.json"
        ),
        "robot_eval_simulator_engine_plugin_registry": (
            "../simulation_automation/simulator_engine_plugin_registry.json"
        ),
        "simulation_automation_proof_boundary": "../simulation_automation/proof_boundary.json",
        "simulation_automation_agent_decision_ledger": "../simulation_automation/agent_decision_ledger.json",
    }
    assert expected_artifacts.items() <= surface["artifacts"].items()
    assert surface["artifacts"]["robot_eval_arena_environment_packet"] == (
        "../simulation_automation/arena_environment_packet.json"
    )
    assert surface["artifacts"]["robot_eval_cpu_preflight_scorecard"] == (
        "../simulation_automation/cpu_preflight_scorecard.json"
    )
    assert surface["artifacts"]["robot_eval_cpu_simulator_preflight_manifest"] == (
        "../simulation_automation/cpu_simulator_preflight_manifest.json"
    )
    assert surface["artifacts"]["robot_eval_gpu_handoff_packet"] == (
        "../simulation_automation/gpu_handoff_packet.json"
    )
    assert surface["artifacts"]["robot_eval_owner_gpu_simulator_execution_blocked_manifest"] == (
        "../simulation_automation/owner_gpu_simulator_execution_blocked_manifest.json"
    )
    assert surface["artifact_uris"]["simulation_automation_run_manifest_uri"].endswith(
        "/pipeline/simulation_automation/simulation_automation_run_manifest.json"
    )
    assert surface["artifact_uris"]["simulation_automation_proof_boundary_uri"].endswith(
        "/pipeline/simulation_automation/proof_boundary.json"
    )
    assert surface["artifact_uris"]["robot_eval_scenario_variation_instances_uri"].endswith(
        "/pipeline/simulation_automation/scenario_variation_instances.json"
    )
