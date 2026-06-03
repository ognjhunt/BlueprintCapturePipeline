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
        "review_only_no_robot_readiness",
    ]
    assert manifest["claim_boundary"]["robot_readiness_proven"] is False
    assert manifest["claim_boundary"]["simulator_execution_proven"] is False
    assert manifest["site_card_count"] == 1
    assert manifest["task_card_count"] == 1
    assert manifest["scenario_card_count"] == 1
    assert manifest["eval_card_count"] == 2
    assert manifest["annotation_backlog_count"] > 0
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
    assert proof_boundaries["simulator_execution_proven"] is False
    assert proof_boundaries["physics_contact_validation_proven"] is False
    assert proof_boundaries["robot_policy_execution_proven"] is False
    assert proof_boundaries["safety_validation_proven"] is False
    assert proof_boundaries["rights_cleared_external_licensing_proven"] is False
    assert proof_boundaries["real_pilot_outcome_proven"] is False
    assert task_library["task_count"] == 1
    assert task_library["tasks"][0]["required_evidence"] == [
        "robot_pov_evidence",
        "human_demo_evidence",
        "action_log_evidence",
        "prediction_outcome_record",
    ]
    assert scenario_library["scenario_count"] == 1
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
