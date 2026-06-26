from __future__ import annotations

import json
from pathlib import Path

import blueprint_pipeline.site_eval_director as sed
from blueprint_pipeline.evaluation_prep_stage import site_eval_director_evaluation_prep_surface
from blueprint_pipeline.site_eval_director import (
    AgentsSdkSiteEvalDirectorAdapter,
    CodexSdkCodeMaintainerAdapter,
    build_site_eval_director,
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
        {"scene_id": "scene-1", "capture_id": "capture-1", "site_identity": {"site_id": "site-1"}},
    )
    return capture_root


def _write_robot_eval_cards(
    capture_root: Path,
    *,
    scenario_variation_label: str = "agent_inferred",
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
            "non_ranking_operational_claim_proven": False,
            "real_pilot_outcome_proven": False,
            "generated_scenarios_are_real_world_proof": False,
        },
    )


def _write_fixture_attempts(
    capture_root: Path,
    *,
    success: bool,
    failure_mode_ids: list[str] | None = None,
    breakage_categories: list[str] | None = None,
) -> None:
    _write_json(
        capture_root / "pipeline" / "robot_eval_inputs" / "headless_fixture_attempts.json",
        {
            "schema_version": "site_eval_fixture_attempts.v1",
            "attempts": [
                {
                    "attempt_id": "fixture-attempt-1",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "task_id": "place_return_in_bin",
                    "policy_id": "fixture-policy-a",
                    "success": success,
                    "predicted_success": True,
                    "predicted_cycle_time_seconds": 12.0,
                    "predicted_intervention_count": 0,
                    "predicted_safety_event_count": 0,
                    "metrics": {
                        "cycle_time_seconds": 10.5 if success else 18.0,
                        "intervention_count": 0 if success else 1,
                        "contact_event_count": 0 if success else 1,
                        "safety_event_count": 0 if success else 1,
                    },
                    "action_trace": [{"t": 0.0, "action": "start"}],
                    "contact_trace": [] if success else [{"t": 4.0, "object": "bin"}],
                    "safety_events": [] if success else [{"t": 4.2, "type": "proximity"}],
                    "failure_mode_ids": failure_mode_ids or [],
                    "breakage_categories": breakage_categories or [],
                    "artifact_paths": {"trace": "fixtures/attempt-1.json"},
                }
            ],
        },
    )


def _write_existing_simulation_sources(capture_root: Path) -> None:
    pipeline_dir = capture_root / "pipeline"
    _write_json(
        pipeline_dir / "worldlabs_world_manifest.json",
        {
            "schema_version": "worldlabs_world_manifest.v1",
            "world_id": "world-1",
            "updated_at": "2026-06-03T00:00:00Z",
            "assets": {
                "mesh": {"collider_mesh_url": "gs://local-blueprint/world-1/collider.glb"},
                "splats": {"spz_urls": {"full": "gs://local-blueprint/world-1/full.spz"}},
            },
        },
    )
    _write_json(
        pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json",
        {
            "schema_version": "marble_simready_bridge.v1",
            "status": "review_ready_with_conversion_required",
            "world_id": "world-1",
        },
    )
    _write_json(
        pipeline_dir / "marble_sim_assets" / "marble_asset_validation.json",
        {
            "schema_version": "marble_asset_validation.v1",
            "overall_status": "review_ready_with_conversion_required",
            "physics_collision_review_ready": True,
            "rank_fidelity_result_proven": False,
        },
    )
    _write_json(
        pipeline_dir / "simready" / "simready_scene_manifest.json",
        {
            "schema_version": "simready_scene_manifest.v1",
            "status": "prepared_for_review",
            "claim_boundary": {
                "simulator_execution_proven": False,
                "rank_fidelity_result_proven": False,
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
                "rank_fidelity_result_proven": False,
            },
        },
    )
    _write_json(
        pipeline_dir / "simulation_automation" / "asset_conversion_plan.json",
        {
            "schema_version": "simulation_asset_conversion_plan.v1",
            "status": "planned",
            "frameworks": {
                "isaac_sim": {"status": "planned_requires_conversion", "blockers": []},
                "isaac_lab_arena": {
                    "status": "planned_requires_owner_asset_mapping",
                    "blockers": ["arena_scene_asset_mapping_required"],
                },
                "mujoco": {"status": "planned_requires_conversion", "blockers": []},
                "pybullet": {"status": "planned_requires_conversion", "blockers": []},
                "newton": {"status": "blocked", "blockers": ["missing_collider_mesh_glb"]},
            },
        },
    )
    _write_json(
        pipeline_dir / "simulation_automation" / "simulator_execution_manifest.json",
        {
            "schema_version": "simulator_execution_manifest.v1",
            "overall_status": "blocked",
            "simulators_run": False,
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
        },
    )
    _write_json(
        pipeline_dir / "simulation_automation" / "simulation_automation_run_manifest.json",
        {
            "schema_version": "simulation_automation_run_manifest.v1",
            "status": "blocked",
            "simulators_run": False,
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )


def test_site_eval_director_builds_card_to_scenario_manifests(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root, scenario_variation_label="derived")
    _write_existing_simulation_sources(capture_root)

    result = build_site_eval_director(capture_root=capture_root)

    automation_dir = capture_root / "pipeline" / "simulation_automation"
    plan = _read_json(automation_dir / "scenario_execution_plan.json")
    requests = _read_json(automation_dir / "task_simulation_requests.json")
    matrix = _read_json(automation_dir / "scenario_simulator_matrix.json")
    queue = _read_json(automation_dir / "agent_review_queue.json")
    run_manifest = _read_json(automation_dir / "site_eval_director_run_manifest.json")
    proof_boundary = _read_json(automation_dir / "site_eval_director_proof_boundary.json")

    assert result["status"] == "review_ready"
    assert run_manifest["status"] == "review_ready"
    assert plan["scenario_count"] == 1
    assert plan["scenarios"][0]["scenario_id"] == "scenario_place_return_in_bin_mobile"
    assert plan["scenarios"][0]["task_id"] == "place_return_in_bin"
    assert plan["scenarios"][0]["execution_mode"] == "review_only"
    assert plan["scenarios"][0]["simulator_execution_proven"] is False
    assert requests["task_request_count"] == 1
    assert requests["requests"][0]["task_id"] == "place_return_in_bin"
    assert requests["requests"][0]["scenario_ids"] == ["scenario_place_return_in_bin_mobile"]
    assert matrix["frameworks"] == [
        "isaac_sim",
        "isaac_lab_arena",
        "mujoco",
        "pybullet",
        "newton",
    ]
    assert {
        row["framework"]: row["conversion_status"]
        for row in matrix["matrix"]
        if row["scenario_id"] == "scenario_place_return_in_bin_mobile"
    } == {
        "isaac_sim": "planned_requires_conversion",
        "isaac_lab_arena": "planned_requires_owner_asset_mapping",
        "mujoco": "planned_requires_conversion",
        "pybullet": "planned_requires_conversion",
        "newton": "blocked",
    }
    assert queue["status"] == "review_required"
    assert proof_boundary["rank_fidelity_result_proven"] is False
    assert proof_boundary["simulator_execution_proven"] is False
    assert proof_boundary["public_claim_upgrade_allowed"] is False


def test_site_eval_director_writes_blocked_manifests_when_robot_eval_cards_missing(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)

    result = build_site_eval_director(capture_root=capture_root)

    automation_dir = capture_root / "pipeline" / "simulation_automation"
    blocked = _read_json(automation_dir / "site_eval_director_blocked_manifest.json")
    run_manifest = _read_json(automation_dir / "site_eval_director_run_manifest.json")
    plan = _read_json(automation_dir / "scenario_execution_plan.json")
    proof_boundary = _read_json(automation_dir / "site_eval_director_proof_boundary.json")

    assert result["status"] == "blocked"
    assert blocked["schema_version"] == "site_eval_director_blocked_manifest.v1"
    assert blocked["status"] == "blocked"
    assert "missing_robot_eval_dataset_cards" in blocked["blockers"]
    assert {
        "robot_eval_site_card",
        "robot_eval_task_cards",
        "robot_eval_scenario_cards",
        "robot_eval_cards",
        "robot_eval_proof_boundaries",
    }.issubset(set(blocked["missing_inputs"]))
    assert blocked["attempted_commands"] == ["build_site_eval_director"]
    assert blocked["claim_boundary"]["rank_fidelity_result_proven"] is False
    assert run_manifest["status"] == "blocked"
    assert plan["status"] == "blocked"
    assert proof_boundary["status"] == "blocked"


def test_agent_inferred_scenarios_stay_review_only(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root, scenario_variation_label="agent_inferred")
    _write_existing_simulation_sources(capture_root)

    build_site_eval_director(capture_root=capture_root)

    automation_dir = capture_root / "pipeline" / "simulation_automation"
    plan = _read_json(automation_dir / "scenario_execution_plan.json")
    queue = _read_json(automation_dir / "agent_review_queue.json")

    scenario = plan["scenarios"][0]
    assert scenario["execution_mode"] == "review_only"
    assert scenario["requires_human_review"] is True
    assert scenario["agent_inferred_components"] == ["edge_case", "variation"]
    assert any(
        item["reason"] == "agent_inferred_scenario_requires_operator_review"
        and item["scenario_id"] == "scenario_place_return_in_bin_mobile"
        for item in queue["items"]
    )


def test_missing_agents_sdk_codex_sdk_api_and_live_gates_write_blocked_operator_manifests(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_existing_simulation_sources(capture_root)

    result = build_site_eval_director(
        capture_root=capture_root,
        agents_adapter=AgentsSdkSiteEvalDirectorAdapter(
            agents_sdk_available=False,
            openai_api_key="",
        ),
        codex_adapter=CodexSdkCodeMaintainerAdapter(
            codex_sdk_available=False,
            openai_api_key="",
            codex_mcp_server_available=False,
            codex_cli_path=None,
        ),
    )

    automation_dir = capture_root / "pipeline" / "simulation_automation"
    agents_manifest = _read_json(
        automation_dir / "agents_sdk_site_eval_director_request.json"
    )
    codex_manifest = _read_json(
        automation_dir / "codex_sdk_code_maintainer_request.json"
    )
    run_manifest = _read_json(automation_dir / "site_eval_director_run_manifest.json")

    assert result["status"] == "review_ready"
    assert agents_manifest["status"] == "blocked"
    assert "missing_openai_agents_sdk" in agents_manifest["blockers"]
    assert "missing_openai_api_key" in agents_manifest["blockers"]
    assert "missing_cli_allow_live_agents_sdk_operator" in agents_manifest["blockers"]
    assert "missing_env_BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS" in agents_manifest["blockers"]
    assert agents_manifest["agent_authority"] == "live_operator_when_gated"
    assert agents_manifest["proof_booleans_mutable_by_agent"] is False
    assert codex_manifest["status"] == "blocked"
    assert "missing_codex_sdk" in codex_manifest["blockers"]
    assert "missing_openai_api_key" in codex_manifest["blockers"]
    assert "missing_cli_allow_live_codex_sdk_operator" in codex_manifest["blockers"]
    assert "missing_env_BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS" in codex_manifest["blockers"]
    assert codex_manifest["evidence"]["codex_mcp_server_required_for_live_operator"] is False
    assert codex_manifest["proof_effect"]["proof_booleans_mutable_by_agent"] is False
    assert run_manifest["agent_request_manifests"] == {
        "agents_sdk_site_eval_director": "agents_sdk_site_eval_director_request.json",
        "codex_sdk_code_maintainer": "codex_sdk_code_maintainer_request.json",
    }
    assert run_manifest["agent_operator_manifests"] == run_manifest["agent_request_manifests"]


def test_codex_code_maintainer_live_operator_can_patch_and_test_without_proof_mutation(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_existing_simulation_sources(capture_root)

    build_site_eval_director(
        capture_root=capture_root,
        codex_adapter=CodexSdkCodeMaintainerAdapter(
            codex_sdk_available=True,
            openai_api_key="sk-test",
            codex_mcp_server_available=True,
            codex_cli_path="/usr/local/bin/codex",
            live_env_allowed=True,
            allow_live_operator=True,
            executor=lambda _prompt, _context: {
                "final_output": "Patched the parser and ran the focused site-eval tests.",
                "commands_chosen": ["pytest tests/test_site_eval_director.py"],
                "tool_call_summaries": [
                    {"tool_name": "apply_patch", "summary": "parser fix"},
                    {"tool_name": "shell", "summary": "focused pytest"},
                ],
                "decisions": [
                    {
                        "decision": "patch_and_test",
                        "summary": "Code fix was required before rerun.",
                    }
                ],
            },
        ),
    )

    manifest = _read_json(
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "codex_sdk_code_maintainer_request.json"
    )

    assert manifest["status"] == "operator_completed"
    assert manifest["execution_performed"] is True
    assert manifest["operator_mode"] == "live_operator"
    assert manifest["agent_authority"] == "live_code_maintainer_when_gated"
    assert manifest["request"]["allowed_request_types"] == [
        "implementation_diagnosis",
        "code_fix_patch",
        "test_execution",
        "diff_summary",
    ]
    assert manifest["request"]["mcp_server_command"] == ["codex", "mcp-server"]
    assert "proof_or_readiness_claim_upgrade" in manifest["request"]["prohibited_request_types"]
    assert manifest["operator_ledger"]["commands_chosen"] == [
        "diagnose_patch_and_test_pipeline_failure",
        "pytest tests/test_site_eval_director.py",
    ]
    assert manifest["operator_ledger"]["tool_call_summaries"][0]["tool_name"] == "apply_patch"
    assert manifest["proof_effect"]["proof_booleans_mutable_by_agent"] is False
    assert manifest["proof_effect"]["direct_proof_booleans_set_true"] == []


def test_evaluation_prep_surfaces_site_eval_director_artifacts_without_overclaiming(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_existing_simulation_sources(capture_root)
    build_site_eval_director(capture_root=capture_root)

    eval_dir = capture_root / "pipeline" / "evaluation_prep"
    eval_dir.mkdir(parents=True, exist_ok=True)
    surface = site_eval_director_evaluation_prep_surface(
        capture_root=capture_root,
        eval_dir=eval_dir,
    )

    assert surface["status"] == "review_ready"
    assert surface["simulator_execution_proven"] is False
    assert surface["rank_fidelity_result_proven"] is False
    assert surface["public_claim_upgrade_allowed"] is False
    assert surface["artifacts"]["site_eval_director_run_manifest"] == (
        "../simulation_automation/site_eval_director_run_manifest.json"
    )
    assert surface["artifacts"]["scenario_execution_plan"] == (
        "../simulation_automation/scenario_execution_plan.json"
    )
    assert surface["artifact_uris"]["site_eval_director_run_manifest_uri"].endswith(
        "/pipeline/simulation_automation/site_eval_director_run_manifest.json"
    )
    assert surface["artifact_uris"]["site_eval_director_proof_boundary_uri"].endswith(
        "/pipeline/simulation_automation/site_eval_director_proof_boundary.json"
    )


def test_site_eval_director_fixture_success_runs_headless_loop_without_claim_upgrade(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root, scenario_variation_label="derived")
    _write_existing_simulation_sources(capture_root)
    _write_fixture_attempts(capture_root, success=True)

    result = build_site_eval_director(capture_root=capture_root)

    automation_dir = capture_root / "pipeline" / "simulation_automation"
    run_manifest = _read_json(automation_dir / "site_eval_director_run_manifest.json")
    trace = _read_json(automation_dir / "normalized_simulator_attempt_trace.json")
    labels = _read_json(automation_dir / "failure_labels.json")
    calibration = _read_json(automation_dir / "site_eval_calibration_report.json")
    ledger = _read_json(automation_dir / "site_eval_prediction_outcome_ledger.json")
    updated_eval_cards = _read_json(automation_dir / "updated_eval_cards.json")
    real_evidence_blocked = _read_json(
        automation_dir / "site_eval_real_evidence_blocked_manifest.json"
    )
    cosmos_exports = _read_json(automation_dir / "cosmos_orchestration_exports.json")

    assert result["status"] == "fixture_loop_completed"
    assert run_manifest["fixture_runner_executed"] is True
    assert run_manifest["simulator_execution_proven"] is False
    assert run_manifest["rank_fidelity_result_proven"] is False
    assert run_manifest["public_claim_upgrade_allowed"] is False
    assert trace["status"] == "completed"
    assert trace["attempts"][0]["success"] is True
    assert labels["labels"][0]["label_status"] == "success"
    assert ledger["records"][0]["success_delta"] == 0
    assert ledger["records"][0]["cycle_time_error_seconds"] == -1.5
    assert calibration["aggregates"]["by_task"][0]["actual_success_rate"] == 1.0
    assert updated_eval_cards["cards"][0]["site_eval_director_update"]["status"] == (
        "fixture_outcome_attached"
    )
    assert real_evidence_blocked["missing_inputs"] == [
        "robot_pov",
        "human_demo",
        "action_logs",
        "actual_outcomes",
    ]
    assert cosmos_exports["post_training_dataset_manifest"]["status"] == "blocked"
    assert "missing_env_BLUEPRINT_ALLOW_COSMOS_TRAINING" in cosmos_exports[
        "post_training_dataset_manifest"
    ]["blockers"]


def test_site_eval_director_fixture_failure_labels_calibrates_and_updates_breakage(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root, scenario_variation_label="derived")
    _write_existing_simulation_sources(capture_root)
    _write_fixture_attempts(
        capture_root,
        success=False,
        failure_mode_ids=[
            "failure_navigation_blocked",
            "failure_safety_threshold_violation",
        ],
        breakage_categories=["blocked_path", "human_crossing"],
    )

    build_site_eval_director(capture_root=capture_root)

    automation_dir = capture_root / "pipeline" / "simulation_automation"
    labels = _read_json(automation_dir / "failure_labels.json")
    ledger = _read_json(automation_dir / "site_eval_prediction_outcome_ledger.json")
    breakage = _read_json(automation_dir / "learned_facility_breakage_library.json")

    assert labels["labels"][0]["label_status"] == "automatic"
    assert labels["labels"][0]["failure_mode_ids"] == [
        "failure_navigation_blocked",
        "failure_safety_threshold_violation",
    ]
    assert ledger["records"][0]["success_delta"] == -1
    assert ledger["records"][0]["intervention_delta"] == 1
    assert ledger["records"][0]["safety_event_delta"] == 1
    assert breakage["status"] == "updated"
    assert breakage["category_counts"]["blocked_path"] == 1
    assert breakage["category_counts"]["human_crossing"] == 1
    assert breakage["category_counts"]["safety_proximity"] == 1


def test_site_eval_director_rights_privacy_blocks_fixture_execution(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(
        capture_root,
        scenario_variation_label="derived",
        rights_blocked=True,
    )
    _write_existing_simulation_sources(capture_root)
    _write_fixture_attempts(capture_root, success=True)

    build_site_eval_director(capture_root=capture_root)

    automation_dir = capture_root / "pipeline" / "simulation_automation"
    trace = _read_json(automation_dir / "normalized_simulator_attempt_trace.json")
    blocked = _read_json(automation_dir / "site_eval_fixture_runner_blocked_manifest.json")
    run_manifest = _read_json(automation_dir / "site_eval_director_run_manifest.json")

    assert trace["status"] == "blocked"
    assert trace["blockers"] == ["blocked_rights_privacy"]
    assert blocked["missing_inputs"] == ["rights_privacy_clearance"]
    assert run_manifest["fixture_runner_executed"] is False
    assert run_manifest["public_claim_upgrade_allowed"] is False


def test_site_eval_director_generated_scenario_blocks_fixture_until_reviewed(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root, scenario_variation_label="agent_inferred")
    _write_existing_simulation_sources(capture_root)
    _write_fixture_attempts(capture_root, success=True)

    build_site_eval_director(capture_root=capture_root)

    automation_dir = capture_root / "pipeline" / "simulation_automation"
    trace = _read_json(automation_dir / "normalized_simulator_attempt_trace.json")
    blocked = _read_json(automation_dir / "site_eval_fixture_runner_blocked_manifest.json")
    queue = _read_json(automation_dir / "agent_review_queue.json")

    assert trace["status"] == "blocked"
    assert "generated_or_inferred_scenarios_require_review" in blocked["blockers"]
    assert any(
        item["reason"] == "agent_inferred_scenario_requires_operator_review"
        for item in queue["items"]
    )


def test_site_eval_director_real_engines_fail_closed_even_with_fixture_success(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root, scenario_variation_label="derived")
    _write_existing_simulation_sources(capture_root)
    _write_fixture_attempts(capture_root, success=True)

    build_site_eval_director(capture_root=capture_root)

    run_manifest = _read_json(
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "site_eval_director_run_manifest.json"
    )

    assert {item["framework"] for item in run_manifest["real_engine_execution_requests"]} == {
        "isaac_sim",
        "isaac_lab_arena",
        "mujoco",
        "pybullet",
        "newton",
    }
    assert all(
        item["status"] == "blocked" and item["execution_performed"] is False
        for item in run_manifest["real_engine_execution_requests"]
    )


def test_site_eval_agents_sdk_operator_success_and_failures(monkeypatch) -> None:
    context = {"capture_root": "/tmp/capture", "repo_root": "/tmp/repo"}

    monkeypatch.setattr(
        sed,
        "run_agents_sdk_operator",
        lambda config: {"command": "inspect", "status": "ok"},
    )
    success = AgentsSdkSiteEvalDirectorAdapter(
        agents_sdk_available=True,
        openai_api_key="key",
        live_env_allowed=True,
        allow_live_operator=True,
    ).build_request_manifest(plan_context=context)
    assert success["status"] == "operator_completed"
    assert success["execution_performed"] is True

    def raise_runtime(_config):
        raise RuntimeError("operator refused")

    monkeypatch.setattr(sed, "run_agents_sdk_operator", raise_runtime)
    runtime_failure = AgentsSdkSiteEvalDirectorAdapter(
        agents_sdk_available=True,
        openai_api_key="key",
        live_env_allowed=True,
        allow_live_operator=True,
    ).build_request_manifest(plan_context=context)
    assert runtime_failure["status"] == "operator_failed"
    assert runtime_failure["blockers"] == ["operator refused"]

    def raise_value(_config):
        raise ValueError("bad")

    monkeypatch.setattr(sed, "run_agents_sdk_operator", raise_value)
    generic_failure = AgentsSdkSiteEvalDirectorAdapter(
        agents_sdk_available=True,
        openai_api_key="key",
        live_env_allowed=True,
        allow_live_operator=True,
    ).build_request_manifest(plan_context=context)
    assert generic_failure["status"] == "operator_failed"
    assert generic_failure["blockers"] == ["agents_sdk_operator_execution_failed:ValueError"]


def test_site_eval_codex_operator_branches(monkeypatch) -> None:
    context = {"capture_root": "/tmp/capture", "repo_root": "/tmp/repo"}
    monkeypatch.setattr(sed, "resolve_codex_cli_path", lambda: None)
    blocked = CodexSdkCodeMaintainerAdapter(
        codex_sdk_available=False,
        openai_api_key="",
        codex_cli_path="",
        codex_mcp_server_available=False,
        live_env_allowed=True,
        allow_live_operator=True,
    ).build_request_manifest(plan_context=context)
    assert "missing_codex_cli" in blocked["blockers"]

    def raise_runtime(_config):
        raise RuntimeError("codex refused")

    monkeypatch.setattr(sed, "run_codex_sdk_operator", raise_runtime)
    runtime_failure = CodexSdkCodeMaintainerAdapter(
        codex_sdk_available=True,
        openai_api_key="key",
        live_env_allowed=True,
        allow_live_operator=True,
    ).build_request_manifest(plan_context=context)
    assert runtime_failure["status"] == "operator_failed"
    assert runtime_failure["blockers"] == ["codex refused"]

    def raise_value(_config):
        raise ValueError("bad")

    monkeypatch.setattr(sed, "run_codex_sdk_operator", raise_value)
    generic_failure = CodexSdkCodeMaintainerAdapter(
        codex_sdk_available=True,
        openai_api_key="key",
        live_env_allowed=True,
        allow_live_operator=True,
        sandbox="unsafe",
    ).build_request_manifest(plan_context=context)
    assert generic_failure["status"] == "operator_failed"
    assert generic_failure["blockers"] == ["codex_sdk_operator_execution_failed:ValueError"]
    assert generic_failure["request"]["sandbox"] == "read-only"


def test_site_eval_codex_mcp_probe_and_small_helpers(tmp_path: Path, monkeypatch) -> None:
    assert sed._module_available(("json",)) is True
    assert sed._codex_mcp_server_available(None) is False

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "codex mcp-server help"

    monkeypatch.setattr(sed.subprocess, "run", lambda *_args, **_kwargs: Completed())
    assert sed._codex_mcp_server_available("codex") is True

    monkeypatch.setattr(
        sed.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad command")),
    )
    assert sed._codex_mcp_server_available("codex") is False

    optional = tmp_path / "optional.json"
    optional.write_text("[]", encoding="utf-8")
    assert sed._read_optional_mapping(optional) == {}
    assert sed._string_list("one") == ["one"]
    assert sed._string_list(7) == ["7"]
    assert sed._cards({"cards": "not-list"}) == []
    assert sed._eval_cards_by_scenario([{"eval_card_id": "missing-scenario"}]) == {}

    context = type("Context", (), {"scene_id": "scene-1", "capture_id": "capture-1"})()
    pipeline_dir = tmp_path / "pipeline"
    _write_json(
        pipeline_dir / "robot_eval_inputs" / "headless_fixture_attempts.json",
        {
            "attempts": [
                {"scenario_id": "unknown", "success": False},
                {
                    "scenario_id": "scenario-1",
                    "task_id": "task-1",
                    "success": False,
                    "metrics": {"contact_event_count": 1},
                },
            ]
        },
    )
    normalized = sed.FixtureSimulatorRunner().run(
        context=context,
        pipeline_dir=pipeline_dir,
        automation_dir=tmp_path / "automation",
        scenario_plan={
            "scenarios": [
                {
                    "scenario_id": "scenario-1",
                    "task_id": "task-1",
                    "agent_inferred_components": [],
                }
            ]
        },
        generated_at="2026-06-21T00:00:00Z",
    )
    assert normalized.attempt["attempts"][0]["failure_mode_ids"] == [
        "failure_contact_collision"
    ]
    assert sed._simulator_execution_status(
        framework="mujoco",
        simulator_execution_manifest={
            "simulator_results": [
                {
                    "framework": "mujoco",
                    "status": "succeeded",
                    "reason": "ok",
                    "simulator_execution_proven": True,
                }
            ]
        },
    )["simulator_execution_proven"] is True


def test_site_eval_failure_calibration_and_breakage_helpers() -> None:
    assert sed._infer_failure_modes(
        metrics={
            "intervention_count": 1,
            "safety_event_count": 1,
            "contact_event_count": 1,
        },
        safety_events=[],
        contact_trace=[],
    ) == [
        "failure_intervention_required",
        "failure_safety_threshold_violation",
        "failure_contact_collision",
    ]
    assert sed._infer_failure_modes(metrics={}, safety_events=[], contact_trace=[]) == [
        "failure_task_not_attempted"
    ]

    context = type("Context", (), {"scene_id": "scene-1", "capture_id": "capture-1"})()
    labels = sed._failure_labels(
        context=context,
        normalized_trace={
            "attempts": [
                "skip-me",
                {"attempt_id": "success", "success": True},
                {
                    "attempt_id": "reviewed",
                    "success": False,
                    "label_review_status": "human_reviewed",
                },
                {"attempt_id": "missing", "success": False},
            ]
        },
        generated_at="2026-06-21T00:00:00Z",
    )
    by_id = {item["attempt_id"]: item for item in labels["labels"]}
    assert by_id["success"]["label_status"] == "success"
    assert by_id["reviewed"]["label_status"] == "human_reviewed"
    assert by_id["missing"]["failure_mode_ids"] == ["failure_evidence_missing"]

    rows = sed._calibration_rows({"attempts": ["skip-me", {"attempt_id": "a", "success": True}]})
    assert rows[0]["record_id"] == "site_eval_a"
    assert sed._group_average(rows, "missing_metric") is None
    assert sed._breakage_categories_from_attempt(
        {
            "failure_mode_ids": [
                "failure_navigation_blocked",
                "failure_localization_or_pose_drift",
                "failure_manipulation_miss",
                "failure_perception_occlusion",
                "failure_safety_threshold_violation",
                "failure_contact_collision",
            ]
        }
    ) == [
        "blocked_path",
        "localization_drift",
        "manipulation_miss",
        "narrow_clearance",
        "occlusion",
        "safety_proximity",
    ]


def test_site_eval_matrix_review_and_real_evidence_edges(tmp_path: Path) -> None:
    context = type(
        "Context",
        (),
        {"scene_id": "scene-1", "capture_id": "capture-1", "capture_root": tmp_path},
    )()
    matrix = sed._scenario_simulator_matrix(
        context=context,
        scenario_plan={"scenarios": ["skip-me", {"scenario_id": "scenario-1", "task_id": "task-1"}]},
        framework_statuses={},
        simulator_execution_manifest={},
        generated_at="2026-06-21T00:00:00Z",
    )
    assert matrix["matrix_count"] == len(sed.SIMULATOR_FRAMEWORKS)

    queue = sed._agent_review_queue(
        context=context,
        scenario_plan={"scenarios": ["skip-me", {"scenario_id": "scenario-1", "task_id": "task-1"}]},
        eval_cards=[],
        proof_boundaries={"simulator_execution_proven": True},
        agent_request_manifests={},
        generated_at="2026-06-21T00:00:00Z",
    )
    assert queue["status"] == "empty"

    pipeline_dir = tmp_path / "pipeline"
    for relative_path in sed.REAL_EVIDENCE_INPUTS.values():
        _write_json(pipeline_dir / relative_path, {"status": "present"})
    assert sed._real_evidence_blocked_manifest(
        context=context,
        pipeline_dir=pipeline_dir,
        generated_at="2026-06-21T00:00:00Z",
    ) is None


def test_site_eval_director_main_success_and_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sed,
        "build_site_eval_director",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "manifest.json"), "status": "built"},
    )
    success_code = sed.main(
        [
            "--capture-root",
            str(tmp_path / "capture"),
            "--agents-sdk-site-eval",
            "--codex-sdk-code-maintainer",
            "--allow-live-agents-sdk-operator",
            "--allow-live-codex-sdk-operator",
            "--codex-sandbox",
            "read-only",
            "--codex-cli-path",
            "codex",
            "--allow-simulator-execution",
            "--allow-simulator",
            "mujoco",
            "--allow-training",
        ]
    )
    stdout = capsys.readouterr().out
    assert success_code == 0
    assert "status=built" in stdout

    def raise_value(**_kwargs):
        raise ValueError("bad capture root")

    monkeypatch.setattr(sed, "build_site_eval_director", raise_value)
    failure_code = sed.main(["--capture-root", str(tmp_path / "missing-capture")])
    failure_stdout = capsys.readouterr().out
    assert failure_code == 1
    assert "bad capture root" in failure_stdout
