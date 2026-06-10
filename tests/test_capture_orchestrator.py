from __future__ import annotations

from pathlib import Path
import json

from blueprint_pipeline.capture_orchestrator import (
    PipelineConfig,
    _build_derived_lane_result,
    resolve_requested_lanes,
    run_capture_pipeline,
)

STANDARD_SCENARIO_VARIATION_NAMES = (
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
)
STANDARD_SIMULATOR_ENGINE_NAMES = ("isaac_sim", "isaac_lab_arena", "mujoco", "pybullet", "newton")
STANDARD_WORLD_MODEL_ENGINE_NAMES = (
    "worldlabs_world_model",
    "marble_simready",
    "cosmos_predict",
    "native_site_reference",
)


def _write_complete_scenario_variation_artifacts(
    capture_root: Path,
    *,
    task_id: str = "place_return_in_bin",
    scenario_id: str = "scenario_place_return_in_bin_mobile",
) -> None:
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    robot_eval_dir.mkdir(parents=True, exist_ok=True)
    automation_dir.mkdir(parents=True, exist_ok=True)
    variations = [
        {
            "variation_id": variation_name,
            "variation_name": variation_name,
            "scenario_status": "review-only",
        }
        for variation_name in STANDARD_SCENARIO_VARIATION_NAMES
    ]
    (robot_eval_dir / "scenario_family_library.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_scenario_family_library.v1",
                "family_count": 1,
                "variation_names_required": list(STANDARD_SCENARIO_VARIATION_NAMES),
                "families": [
                    {
                        "family_id": f"family_{scenario_id}",
                        "scenario_id": scenario_id,
                        "task_id": task_id,
                        "robot_profile_id": "mobile_manipulator_rgb_v1",
                        "status": "review_required",
                        "variation_count": len(variations),
                        "variations": variations,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    instances = [
        {
            "instance_id": f"variation_{task_id}_{scenario_id}_{variation_name}",
            "family_id": f"family_{scenario_id}",
            "scenario_id": scenario_id,
            "task_id": task_id,
            "variation_id": variation_name,
            "variation_name": variation_name,
            "concrete_mutation": {"mutation_type": variation_name, "ordinal": index},
            "engine_mutations": {
                "isaac_sim": {"status": "ready", "mutation_type": variation_name}
            },
        }
        for index, variation_name in enumerate(STANDARD_SCENARIO_VARIATION_NAMES, start=1)
    ]
    (automation_dir / "scenario_variation_instances.json").write_text(
        json.dumps(
            {
                "schema_version": "scenario_variation_instances.v1",
                "status": "completed",
                "required_variation_names": list(STANDARD_SCENARIO_VARIATION_NAMES),
                "variation_names_instantiated": list(STANDARD_SCENARIO_VARIATION_NAMES),
                "family_count": 1,
                "instance_count": len(instances),
                "instances": instances,
            }
        ),
        encoding="utf-8",
    )


def _write_failure_taxonomy(robot_eval_dir: Path) -> None:
    (robot_eval_dir / "failure_taxonomy.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_failure_taxonomy.v1",
                "failure_modes": [
                    {
                        "failure_mode_id": "failure_navigation_blocked",
                        "label": "Robot cannot reach the required zone or route.",
                    },
                    {
                        "failure_mode_id": "failure_collision_risk",
                        "label": "Collision risk was detected.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_required_robot_eval_dataset_inputs(capture_root: Path) -> None:
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    robot_eval_dir.mkdir(parents=True, exist_ok=True)
    (robot_eval_dir / "site_card.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_site_card.v0.1",
                "site_id": "site-1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "site_type": "stockroom",
                "geometry": {
                    "collider": {
                        "status": "review_input_present",
                        "collision_ready_claim_allowed": False,
                    }
                },
                "provenance_rights_review_status": {
                    "rights_privacy": {"blocked": False, "rights_status": "verified"}
                },
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "eval_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_eval_cards.v0.1",
                "eval_card_count": 1,
                "cards": [
                    {
                        "eval_card_id": "eval_card_place_return_in_bin_fixture",
                        "scenario_id": "scenario_place_return_in_bin_mobile",
                        "task_id": "place_return_in_bin",
                        "prediction_source": "fixture_review",
                        "engine_used": "fixture",
                        "validation": {"actual_status": "needs_actual_outcome"},
                        "proof_boundary": "prediction_only_no_actual_outcome_no_deployment_claim",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "proof_boundaries.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_proof_boundaries.v0.1",
                "simulator_execution_proven": False,
                "robot_policy_execution_proven": False,
                "robot_readiness_proven": False,
                "public_claim_upgrade_allowed": False,
            }
        ),
        encoding="utf-8",
    )


def _write_complete_simulation_automation_plugin_inputs(
    capture_root: Path,
    *,
    include_episode_spec: bool = True,
) -> None:
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    automation_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "simulation_automation_plan.json",
        "asset_conversion_plan.json",
        "cpu_simulator_preflight_manifest.json",
        "arena_environment_packet.json",
    ):
        (automation_dir / filename).write_text(
            json.dumps({"schema_version": f"{filename}.test", "status": "ready"}),
            encoding="utf-8",
        )
    if include_episode_spec:
        (automation_dir / "episode_spec.v1.json").write_text(
            json.dumps(
                {
                    "schema_version": "episode_spec.v1",
                    "status": "ready_for_review",
                    "episodes": [{"episode_id": "episode-1"}],
                }
            ),
            encoding="utf-8",
        )


def _write_complete_simulator_plugin_registry(capture_root: Path) -> None:
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    automation_dir.mkdir(parents=True, exist_ok=True)
    (automation_dir / "simulator_engine_plugin_registry.json").write_text(
        json.dumps(
            {
                "schema_version": "simulator_engine_plugin_registry.v1",
                "status": "ready_for_gated_managed_execution",
                "engine_targets": list(STANDARD_SIMULATOR_ENGINE_NAMES),
                "world_model_engine_targets": list(STANDARD_WORLD_MODEL_ENGINE_NAMES),
                "plugin_count": len(STANDARD_SIMULATOR_ENGINE_NAMES),
                "world_model_plugin_count": len(STANDARD_WORLD_MODEL_ENGINE_NAMES),
                "plugins": {
                    engine: {
                        "plugin_id": f"blueprint_{engine}_sim_engine_plugin",
                        "framework": engine,
                        "adapter_contract_status": "ready",
                        "managed_execution_supported": True,
                        "inputs": {
                            "simulation_automation_plan": "simulation_automation_plan.json",
                            "asset_conversion_plan": "asset_conversion_plan.json",
                            "arena_environment_packet": "arena_environment_packet.json"
                            if engine == "isaac_lab_arena"
                            else None,
                            "scenario_variation_instances": "scenario_variation_instances.json",
                            "episode_spec": "episode_spec.v1.json",
                            "cpu_preflight_manifest": "cpu_simulator_preflight_manifest.json",
                        },
                    }
                    for engine in STANDARD_SIMULATOR_ENGINE_NAMES
                },
                "world_model_plugins": {
                    engine: {
                        "plugin_id": f"blueprint_{engine}_engine_plugin",
                        "engine": engine,
                        "adapter_contract_status": "ready",
                        "managed_execution_supported": True,
                        "inputs": {
                            "simulation_automation_plan": "simulation_automation_plan.json",
                            "scenario_variation_instances": "scenario_variation_instances.json",
                            "site_card": "../robot_eval_dataset/site_card.json",
                            "task_cards": "../robot_eval_dataset/task_cards.json",
                            "scenario_cards": "../robot_eval_dataset/scenario_cards.json",
                        },
                    }
                    for engine in STANDARD_WORLD_MODEL_ENGINE_NAMES
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_derived_lane_result_preserves_current_orchestration_shape() -> None:
    result = _build_derived_lane_result(
        lane="evaluation_prep",
        source="evaluation_prep_artifacts",
        qualification_result={
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
        extra_fields={"manifest_path": "pipeline/evaluation_prep/evaluation_prep_manifest.json"},
    )

    assert result == {
        "status": "completed",
        "lane": "evaluation_prep",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        "source": "evaluation_prep_artifacts",
        "manifest_path": "pipeline/evaluation_prep/evaluation_prep_manifest.json",
    }


def test_capture_orchestrator_keeps_supported_lanes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_requested_lanes",
        lambda **_kwargs: ["qualification", "scene_memory", "evaluation_prep"],
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json",
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    assert result["lanes"] == ["qualification", "scene_memory", "evaluation_prep"]
    assert all(item["lane"] != "advanced_geometry" for item in result["results"])


def test_capture_orchestrator_current_lane_runs_simulation_automation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        lane="current",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    assert result["lanes"] == ["qualification", "evaluation_prep", "simulation_automation"]
    assert [item["lane"] for item in result["results"]] == [
        "qualification",
        "evaluation_prep",
        "simulation_automation",
    ]
    assert result["results"][-1]["automation_status"] == "blocked"
    assert result["results"][-1]["robot_eval_job_inbox_status"] == "waiting_for_job_requests"
    assert result["results"][-1]["robot_eval_job_inbox_processed_count"] == 0


def test_capture_orchestrator_processes_robot_eval_job_inbox_when_present(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text("{}", encoding="utf-8")
    inbox = descriptor_path.parent / "pipeline" / "robot_eval_job_requests" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "robot-eval-job.json").write_text("{}", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    def _run_inbox(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"status": "completed", "processed_count": 1}

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_robot_eval_job_request_inbox",
        _run_inbox,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        lane="current",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    assert len(calls) == 1
    assert calls[0]["capture_root"] == descriptor_path.parent
    assert calls[0]["inbox_dir"] == inbox
    assert result["results"][-1]["robot_eval_job_inbox_status"] == "completed"
    assert result["results"][-1]["robot_eval_job_inbox_processed_count"] == 1


def test_capture_orchestrator_auto_stages_task_eval_job_request_when_capture_requests_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "metadata": {"site_identity": {"site_id": "site-1"}},
                "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
            }
        ),
        encoding="utf-8",
    )
    robot_eval_dir = descriptor_path.parent / "pipeline" / "robot_eval_dataset"
    robot_eval_dir.mkdir(parents=True)
    (robot_eval_dir / "task_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_task_cards.v0.1",
                "cards": [
                    {
                        "task_id": "place_return_in_bin",
                        "task_statement": "Place the return item in the labeled bin",
                        "task_category": "pick_place",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scenario_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
                "cards": [
                    {
                        "scenario_id": "scenario_place_return_in_bin_mobile",
                        "task_id": "place_return_in_bin",
                        "robot_profile_id": "mobile_manipulator_rgb_v1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "task_thresholds.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_task_thresholds.v1",
                "thresholds": [
                    {
                        "task_id": "place_return_in_bin",
                        "success_rate_min": 0.95,
                        "timeout_seconds": 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scoring_methodology.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scoring_methodology.v1",
                "metrics": [
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
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_complete_scenario_variation_artifacts(descriptor_path.parent)
    _write_failure_taxonomy(robot_eval_dir)
    _write_required_robot_eval_dataset_inputs(descriptor_path.parent)
    _write_complete_simulation_automation_plugin_inputs(descriptor_path.parent)
    _write_complete_simulator_plugin_registry(descriptor_path.parent)
    calls = []

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    def _run_inbox(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"status": "completed", "processed_count": 1}

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_robot_eval_job_request_inbox",
        _run_inbox,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    inbox = descriptor_path.parent / "pipeline" / "robot_eval_job_requests" / "inbox"
    staged_requests = sorted(inbox.glob("*.json"))
    envelope = json.loads(staged_requests[0].read_text(encoding="utf-8"))
    request = envelope["job_request"]

    assert len(calls) == 1
    assert calls[0]["inbox_dir"] == inbox
    assert result["results"][-1]["robot_eval_job_inbox_status"] == "completed"
    assert result["results"][-1]["robot_eval_job_auto_stage_status"] == "staged"
    assert envelope["queue_contract"] == "robot_eval_job_request_inbox.v1"
    assert request["schema_version"] == "robot_eval_job_request.v1"
    assert request["source"]["system"] == "BlueprintCapturePipeline.auto_stage"
    assert request["site_package"]["capture_root"] == str(descriptor_path.parent.resolve())
    assert request["requested_tasks"][0]["task_id"] == "place_return_in_bin"
    assert request["requested_tasks"][0]["scenario_ids"] == [
        "scenario_place_return_in_bin_mobile"
    ]
    assert request["policy_package"]["high_level_skill_trace"]["source_type"] == (
        "blueprint_default_baseline_trace"
    )


def test_capture_orchestrator_blocks_task_eval_auto_stage_without_eval_methodology(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
            }
        ),
        encoding="utf-8",
    )
    robot_eval_dir = descriptor_path.parent / "pipeline" / "robot_eval_dataset"
    robot_eval_dir.mkdir(parents=True)
    (robot_eval_dir / "task_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_task_cards.v0.1",
                "cards": [{"task_id": "place_return_in_bin"}],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scenario_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
                "cards": [
                    {
                        "scenario_id": "scenario_place_return_in_bin_mobile",
                        "task_id": "place_return_in_bin",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_failure_taxonomy(robot_eval_dir)
    calls = []

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    def _run_inbox(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"status": "completed", "processed_count": 1}

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_robot_eval_job_request_inbox",
        _run_inbox,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    inbox = descriptor_path.parent / "pipeline" / "robot_eval_job_requests" / "inbox"
    auto_stage = result["results"][-1]

    assert calls == []
    assert not inbox.exists()
    assert auto_stage["robot_eval_job_inbox_status"] == "blocked_missing_task_eval_inputs"
    assert auto_stage["robot_eval_job_auto_stage_status"] == "blocked"
    assert auto_stage["robot_eval_job_auto_stage_blockers"] == [
        "robot_eval_task_thresholds_missing",
        "robot_eval_scoring_methodology_missing",
    ]


def test_capture_orchestrator_blocks_task_eval_auto_stage_with_weak_eval_methodology(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
            }
        ),
        encoding="utf-8",
    )
    robot_eval_dir = descriptor_path.parent / "pipeline" / "robot_eval_dataset"
    robot_eval_dir.mkdir(parents=True)
    (robot_eval_dir / "task_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_task_cards.v0.1",
                "cards": [{"task_id": "place_return_in_bin"}],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scenario_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
                "cards": [
                    {
                        "scenario_id": "scenario_place_return_in_bin_mobile",
                        "task_id": "place_return_in_bin",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "task_thresholds.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_task_thresholds.v1",
                "thresholds": [
                    {
                        "task_id": "inspect_shelf",
                        "success_rate_min": 0.95,
                        "timeout_seconds": 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scoring_methodology.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scoring_methodology.v1",
                "metrics": ["success_rate", "cycle_time"],
            }
        ),
        encoding="utf-8",
    )
    _write_failure_taxonomy(robot_eval_dir)
    calls = []

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    def _run_inbox(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"status": "completed", "processed_count": 1}

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_robot_eval_job_request_inbox",
        _run_inbox,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    inbox = descriptor_path.parent / "pipeline" / "robot_eval_job_requests" / "inbox"
    auto_stage = result["results"][-1]

    assert calls == []
    assert not inbox.exists()
    assert auto_stage["robot_eval_job_inbox_status"] == "blocked_missing_task_eval_inputs"
    assert auto_stage["robot_eval_job_auto_stage_status"] == "blocked"
    assert auto_stage["robot_eval_job_auto_stage_blockers"] == [
        "robot_eval_task_thresholds_missing_requested_tasks",
        "robot_eval_scoring_methodology_missing_standard_metrics",
    ]
    assert auto_stage["robot_eval_job_auto_stage_missing_threshold_task_ids"] == [
        "place_return_in_bin"
    ]
    assert auto_stage["robot_eval_job_auto_stage_missing_scorecard_metrics"] == [
        "intervention_rate",
        "unsafe_proximity",
        "collision_risk",
        "object_drop",
        "wrong_object",
        "timeout",
        "recovery_success",
        "world_model_uncertainty",
        "sim_vs_real_calibration_score",
    ]


def test_capture_orchestrator_blocks_task_eval_auto_stage_with_weak_scenario_variations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
            }
        ),
        encoding="utf-8",
    )
    robot_eval_dir = descriptor_path.parent / "pipeline" / "robot_eval_dataset"
    automation_dir = descriptor_path.parent / "pipeline" / "simulation_automation"
    robot_eval_dir.mkdir(parents=True)
    automation_dir.mkdir(parents=True)
    (robot_eval_dir / "task_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_task_cards.v0.1",
                "cards": [{"task_id": "place_return_in_bin"}],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scenario_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
                "cards": [
                    {
                        "scenario_id": "scenario_place_return_in_bin_mobile",
                        "task_id": "place_return_in_bin",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "task_thresholds.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_task_thresholds.v1",
                "thresholds": [
                    {
                        "task_id": "place_return_in_bin",
                        "success_rate_min": 0.95,
                        "timeout_seconds": 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scoring_methodology.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scoring_methodology.v1",
                "metrics": [
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
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scenario_family_library.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_scenario_family_library.v1",
                "family_count": 1,
                "variation_names_required": list(STANDARD_SCENARIO_VARIATION_NAMES),
                "families": [
                    {
                        "family_id": "family_scenario_place_return_in_bin_mobile",
                        "scenario_id": "scenario_place_return_in_bin_mobile",
                        "task_id": "place_return_in_bin",
                        "variation_count": 1,
                        "variations": [
                            {
                                "variation_id": "lighting_variation",
                                "variation_name": "lighting_variation",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (automation_dir / "scenario_variation_instances.json").write_text(
        json.dumps(
            {
                "schema_version": "scenario_variation_instances.v1",
                "status": "completed",
                "required_variation_names": list(STANDARD_SCENARIO_VARIATION_NAMES),
                "variation_names_instantiated": ["lighting_variation"],
                "instance_count": 1,
                "instances": [
                    {
                        "instance_id": "variation_place_return_in_bin_lighting",
                        "scenario_id": "scenario_place_return_in_bin_mobile",
                        "task_id": "place_return_in_bin",
                        "variation_name": "lighting_variation",
                        "concrete_mutation": {"mutation_type": "lighting"},
                        "engine_mutations": {"isaac_sim": {"status": "ready"}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_failure_taxonomy(robot_eval_dir)
    calls = []

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    def _run_inbox(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"status": "completed", "processed_count": 1}

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_robot_eval_job_request_inbox",
        _run_inbox,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    inbox = descriptor_path.parent / "pipeline" / "robot_eval_job_requests" / "inbox"
    auto_stage = result["results"][-1]
    missing_variations = list(STANDARD_SCENARIO_VARIATION_NAMES[1:])

    assert calls == []
    assert not inbox.exists()
    assert auto_stage["robot_eval_job_inbox_status"] == "blocked_missing_task_eval_inputs"
    assert auto_stage["robot_eval_job_auto_stage_status"] == "blocked"
    assert auto_stage["robot_eval_job_auto_stage_blockers"] == [
        "robot_eval_scenario_family_library_missing_required_variations",
        "robot_eval_scenario_variation_instances_missing_required_variations",
        "robot_eval_scenario_variation_instances_missing_required_variations_per_scenario",
    ]
    assert auto_stage["robot_eval_job_auto_stage_missing_scenario_variation_names"] == (
        missing_variations
    )
    assert auto_stage[
        "robot_eval_job_auto_stage_missing_scenario_variation_names_by_scenario"
    ] == [
        {
            "scenario_id": "scenario_place_return_in_bin_mobile",
            "missing_variation_names": missing_variations,
        }
    ]


def test_capture_orchestrator_blocks_task_eval_auto_stage_without_failure_taxonomy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
            }
        ),
        encoding="utf-8",
    )
    robot_eval_dir = descriptor_path.parent / "pipeline" / "robot_eval_dataset"
    robot_eval_dir.mkdir(parents=True)
    (robot_eval_dir / "task_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_task_cards.v0.1",
                "cards": [{"task_id": "place_return_in_bin"}],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scenario_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
                "cards": [
                    {
                        "scenario_id": "scenario_place_return_in_bin_mobile",
                        "task_id": "place_return_in_bin",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "task_thresholds.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_task_thresholds.v1",
                "thresholds": [
                    {
                        "task_id": "place_return_in_bin",
                        "success_rate_min": 0.95,
                        "timeout_seconds": 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scoring_methodology.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scoring_methodology.v1",
                "metrics": [
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
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_complete_scenario_variation_artifacts(descriptor_path.parent)
    calls = []

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    def _run_inbox(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"status": "completed", "processed_count": 1}

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_robot_eval_job_request_inbox",
        _run_inbox,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    inbox = descriptor_path.parent / "pipeline" / "robot_eval_job_requests" / "inbox"
    auto_stage = result["results"][-1]

    assert calls == []
    assert not inbox.exists()
    assert auto_stage["robot_eval_job_inbox_status"] == "blocked_missing_task_eval_inputs"
    assert auto_stage["robot_eval_job_auto_stage_status"] == "blocked"
    assert auto_stage["robot_eval_job_auto_stage_blockers"] == [
        "robot_eval_failure_taxonomy_missing",
    ]
    assert auto_stage["robot_eval_job_auto_stage_failure_taxonomy_mode_count"] == 0


def test_capture_orchestrator_blocks_task_eval_auto_stage_without_plugin_registry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
            }
        ),
        encoding="utf-8",
    )
    robot_eval_dir = descriptor_path.parent / "pipeline" / "robot_eval_dataset"
    robot_eval_dir.mkdir(parents=True)
    (robot_eval_dir / "task_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_task_cards.v0.1",
                "cards": [{"task_id": "place_return_in_bin"}],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scenario_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
                "cards": [
                    {
                        "scenario_id": "scenario_place_return_in_bin_mobile",
                        "task_id": "place_return_in_bin",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "task_thresholds.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_task_thresholds.v1",
                "thresholds": [
                    {
                        "task_id": "place_return_in_bin",
                        "success_rate_min": 0.95,
                        "timeout_seconds": 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scoring_methodology.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scoring_methodology.v1",
                "metrics": [
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
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_complete_scenario_variation_artifacts(descriptor_path.parent)
    _write_failure_taxonomy(robot_eval_dir)
    _write_required_robot_eval_dataset_inputs(descriptor_path.parent)
    calls = []

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    def _run_inbox(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"status": "completed", "processed_count": 1}

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_robot_eval_job_request_inbox",
        _run_inbox,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    inbox = descriptor_path.parent / "pipeline" / "robot_eval_job_requests" / "inbox"
    auto_stage = result["results"][-1]

    assert calls == []
    assert not inbox.exists()
    assert auto_stage["robot_eval_job_inbox_status"] == "blocked_missing_task_eval_inputs"
    assert auto_stage["robot_eval_job_auto_stage_status"] == "blocked"
    assert auto_stage["robot_eval_job_auto_stage_blockers"] == [
        "robot_eval_simulator_engine_plugin_registry_missing",
        "robot_eval_simulator_engine_plugin_registry_missing_required_engines",
        "robot_eval_simulator_engine_plugin_registry_missing_required_world_model_engines",
    ]
    assert auto_stage["robot_eval_job_auto_stage_missing_simulator_plugins"] == sorted(
        STANDARD_SIMULATOR_ENGINE_NAMES
    )
    assert auto_stage["robot_eval_job_auto_stage_missing_world_model_plugins"] == sorted(
        STANDARD_WORLD_MODEL_ENGINE_NAMES
    )


def test_capture_orchestrator_blocks_task_eval_auto_stage_without_required_dataset_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
            }
        ),
        encoding="utf-8",
    )
    robot_eval_dir = descriptor_path.parent / "pipeline" / "robot_eval_dataset"
    robot_eval_dir.mkdir(parents=True)
    (robot_eval_dir / "task_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_task_cards.v0.1",
                "cards": [{"task_id": "place_return_in_bin"}],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scenario_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
                "cards": [
                    {
                        "scenario_id": "scenario_place_return_in_bin_mobile",
                        "task_id": "place_return_in_bin",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "task_thresholds.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_task_thresholds.v1",
                "thresholds": [
                    {
                        "task_id": "place_return_in_bin",
                        "success_rate_min": 0.95,
                        "timeout_seconds": 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scoring_methodology.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scoring_methodology.v1",
                "metrics": [
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
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_complete_scenario_variation_artifacts(descriptor_path.parent)
    _write_failure_taxonomy(robot_eval_dir)
    _write_complete_simulation_automation_plugin_inputs(descriptor_path.parent)
    _write_complete_simulator_plugin_registry(descriptor_path.parent)
    calls = []

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    def _run_inbox(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"status": "completed", "processed_count": 1}

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_robot_eval_job_request_inbox",
        _run_inbox,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    inbox = descriptor_path.parent / "pipeline" / "robot_eval_job_requests" / "inbox"
    auto_stage = result["results"][-1]

    assert calls == []
    assert not inbox.exists()
    assert auto_stage["robot_eval_job_inbox_status"] == "blocked_missing_task_eval_inputs"
    assert auto_stage["robot_eval_job_auto_stage_status"] == "blocked"
    assert auto_stage["robot_eval_job_auto_stage_blockers"] == [
        "robot_eval_site_card_missing",
        "robot_eval_cards_missing",
        "robot_eval_proof_boundaries_missing",
    ]
    assert auto_stage["robot_eval_job_auto_stage_missing_robot_eval_dataset_inputs"] == [
        "robot_eval_site_card",
        "robot_eval_cards",
        "robot_eval_proof_boundaries",
    ]


def test_capture_orchestrator_blocks_task_eval_auto_stage_with_missing_plugin_local_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
            }
        ),
        encoding="utf-8",
    )
    robot_eval_dir = descriptor_path.parent / "pipeline" / "robot_eval_dataset"
    robot_eval_dir.mkdir(parents=True)
    (robot_eval_dir / "task_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_task_cards.v0.1",
                "cards": [{"task_id": "place_return_in_bin"}],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scenario_cards.json").write_text(
        json.dumps(
            {
                "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
                "cards": [
                    {
                        "scenario_id": "scenario_place_return_in_bin_mobile",
                        "task_id": "place_return_in_bin",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "task_thresholds.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_task_thresholds.v1",
                "thresholds": [
                    {
                        "task_id": "place_return_in_bin",
                        "success_rate_min": 0.95,
                        "timeout_seconds": 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (robot_eval_dir / "scoring_methodology.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scoring_methodology.v1",
                "metrics": [
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
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_complete_scenario_variation_artifacts(descriptor_path.parent)
    _write_failure_taxonomy(robot_eval_dir)
    _write_required_robot_eval_dataset_inputs(descriptor_path.parent)
    _write_complete_simulation_automation_plugin_inputs(
        descriptor_path.parent,
        include_episode_spec=False,
    )
    _write_complete_simulator_plugin_registry(descriptor_path.parent)
    calls = []

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    def _run_inbox(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"status": "completed", "processed_count": 1}

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_robot_eval_job_request_inbox",
        _run_inbox,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    inbox = descriptor_path.parent / "pipeline" / "robot_eval_job_requests" / "inbox"
    auto_stage = result["results"][-1]

    assert calls == []
    assert not inbox.exists()
    assert auto_stage["robot_eval_job_inbox_status"] == "blocked_missing_task_eval_inputs"
    assert auto_stage["robot_eval_job_auto_stage_status"] == "blocked"
    assert auto_stage["robot_eval_job_auto_stage_blockers"] == [
        "robot_eval_simulator_engine_plugin_registry_missing_local_input_artifacts",
    ]
    assert set(
        auto_stage[
            "robot_eval_job_auto_stage_missing_simulator_plugin_local_inputs"
        ]
    ) == set(STANDARD_SIMULATOR_ENGINE_NAMES)
    assert auto_stage[
        "robot_eval_job_auto_stage_missing_simulator_plugin_local_inputs"
    ]["mujoco"] == ["episode_spec"]


def test_capture_orchestrator_blocks_task_eval_auto_stage_without_task_cards(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["task_evaluation_run"],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    def _run_inbox(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"status": "completed", "processed_count": 1}

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_robot_eval_job_request_inbox",
        _run_inbox,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    inbox = descriptor_path.parent / "pipeline" / "robot_eval_job_requests" / "inbox"
    auto_stage = result["results"][-1]

    assert calls == []
    assert not inbox.exists()
    assert auto_stage["robot_eval_job_inbox_status"] == "blocked_missing_task_eval_inputs"
    assert auto_stage["robot_eval_job_inbox_processed_count"] == 0
    assert auto_stage["robot_eval_job_auto_stage_status"] == "blocked"
    assert auto_stage["robot_eval_job_auto_stage_blockers"] == [
        "robot_eval_task_cards_missing",
        "robot_eval_scenario_cards_missing",
        "robot_eval_task_thresholds_missing",
        "robot_eval_scoring_methodology_missing",
    ]


def test_capture_orchestrator_runs_single_capture_smoke_lane(monkeypatch, tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_requested_lanes",
        lambda **_kwargs: ["cosmos_single_capture_smoke"],
    )
    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.cosmos_benchmark.run_cosmos_single_capture_smoke_lane",
        lambda **_kwargs: {"status": "blocked", "reason": "runtime_unavailable"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    assert result["lanes"] == ["cosmos_single_capture_smoke"]
    assert result["results"] == [
        {
            "lane": "cosmos_single_capture_smoke",
            "status": "blocked",
            "reason": "runtime_unavailable",
        }
    ]


def test_resolve_requested_lanes_defaults_to_current_stack_for_site_world_candidate(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "capture_mode": {"resolved_mode": "site_world_candidate"},
                "scene_memory_capture": {"world_model_candidate": True},
                "requested_outputs": ["preview_simulation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == ["qualification", "evaluation_prep", "simulation_automation"]


def test_resolve_requested_lanes_honors_explicit_descriptor_requested_lanes(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_lanes": [
                    "qualification",
                    "retrieval_index",
                    "synthesis_coverage_validation",
                ],
                "requested_outputs": [],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "retrieval_index",
        "synthesis_coverage_validation",
    ]


def test_resolve_requested_lanes_accepts_capture_bridge_robot_eval_alias_lanes(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
                "requested_lanes": [
                    "evaluation_prep",
                    "robot_eval_dataset",
                    "task_evaluation_run",
                ],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == ["qualification", "evaluation_prep", "simulation_automation"]


def test_resolve_requested_lanes_infers_current_lanes_from_robot_eval_outputs(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == ["qualification", "evaluation_prep", "simulation_automation"]


def test_resolve_requested_lanes_demotes_bridge_default_scene_memory_pair(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_lanes": ["qualification", "scene_memory"],
                "requested_outputs": [],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == ["qualification"]


def test_resolve_requested_lanes_prefers_explicit_descriptor_lanes_over_output_inference(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_lanes": [
                    "qualification",
                    "synthesis_coverage_validation",
                ],
                "requested_outputs": ["preview_simulation", "deeper_evaluation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "synthesis_coverage_validation",
    ]


def test_resolve_requested_lanes_prefers_explicit_descriptor_lanes_over_native_candidate_default(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "capture_mode": {"resolved_mode": "site_world_candidate"},
                "scene_memory_capture": {"world_model_candidate": True},
                "requested_lanes": [
                    "qualification",
                    "synthesis_coverage_validation",
                ],
                "requested_outputs": ["preview_simulation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "synthesis_coverage_validation",
    ]


def test_resolve_requested_lanes_accepts_camel_case_descriptor_fields(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requestedLanes": [
                    "qualification",
                    "retrieval_index",
                ],
                "requestedOutputs": ["preview_simulation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "retrieval_index",
    ]


def test_resolve_requested_lanes_accepts_scalar_descriptor_requested_lanes(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requestedLanes": "retrieval_index",
                "requestedOutputs": ["preview_simulation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "retrieval_index",
    ]
