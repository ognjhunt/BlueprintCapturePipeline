from __future__ import annotations

import json
from pathlib import Path

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
            "robot_readiness_proven": False,
        },
    )
    _write_json(
        pipeline_dir / "simready" / "simready_scene_manifest.json",
        {
            "schema_version": "simready_scene_manifest.v1",
            "status": "prepared_for_review",
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
        pipeline_dir / "simulation_automation" / "asset_conversion_plan.json",
        {
            "schema_version": "simulation_asset_conversion_plan.v1",
            "status": "planned",
            "frameworks": {
                "isaac_sim": {"status": "planned_requires_conversion", "blockers": []},
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
            "robot_readiness_proven": False,
        },
    )
    _write_json(
        pipeline_dir / "simulation_automation" / "simulation_automation_run_manifest.json",
        {
            "schema_version": "simulation_automation_run_manifest.v1",
            "status": "blocked",
            "simulators_run": False,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
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
    assert matrix["frameworks"] == ["isaac_sim", "mujoco", "pybullet", "newton"]
    assert {
        row["framework"]: row["conversion_status"]
        for row in matrix["matrix"]
        if row["scenario_id"] == "scenario_place_return_in_bin_mobile"
    } == {
        "isaac_sim": "planned_requires_conversion",
        "mujoco": "planned_requires_conversion",
        "pybullet": "planned_requires_conversion",
        "newton": "blocked",
    }
    assert queue["status"] == "review_required"
    assert proof_boundary["robot_readiness_proven"] is False
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
    assert blocked["claim_boundary"]["robot_readiness_proven"] is False
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


def test_missing_agents_sdk_codex_sdk_api_and_mcp_write_blocked_advisory_manifests(
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
    assert codex_manifest["status"] == "blocked"
    assert "missing_codex_sdk" in codex_manifest["blockers"]
    assert "missing_openai_api_key" in codex_manifest["blockers"]
    assert "missing_codex_mcp_server" in codex_manifest["blockers"]
    assert run_manifest["agent_request_manifests"] == {
        "agents_sdk_site_eval_director": "agents_sdk_site_eval_director_request.json",
        "codex_sdk_code_maintainer": "codex_sdk_code_maintainer_request.json",
    }


def test_codex_code_maintainer_manifest_limits_subagent_to_code_fix_requests(
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
        ),
    )

    manifest = _read_json(
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "codex_sdk_code_maintainer_request.json"
    )

    assert manifest["status"] == "request_manifest_ready"
    assert manifest["execution_performed"] is False
    assert manifest["agent_authority"] == "advisory_only"
    assert manifest["request"]["allowed_request_types"] == [
        "implementation_diagnosis",
        "code_fix_patch_plan",
    ]
    assert manifest["request"]["mcp_server_command"] == ["codex", "mcp-server"]
    assert "proof_or_readiness_claim_upgrade" in manifest["request"]["prohibited_request_types"]


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
    assert surface["robot_readiness_proven"] is False
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
    assert run_manifest["robot_readiness_proven"] is False
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
        "mujoco",
        "pybullet",
        "newton",
    }
    assert all(
        item["status"] == "blocked" and item["execution_performed"] is False
        for item in run_manifest["real_engine_execution_requests"]
    )
