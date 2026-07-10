from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.kitchen_task_scaling_preflight import (
    _both_hands_wrists_visible,
    affordance_object_id_candidates_for_task,
    export_all_policy_observations_from_preflight,
    build_request,
    default_task_specs,
    evaluate_local_task_gates,
    export_policy_observation_from_preflight,
    perception_target_prompts_for_task,
    run_preflight,
    target_object_id_candidates_for_task,
)


def test_single_auto_selected_arm_uses_flat_geometry_role_evidence() -> None:
    assert _both_hands_wrists_visible(
        {
            "required_arms": ["left"],
            "reach_arm": "left",
            "arm_roles_in_frame": ["hand", "wrist"],
        }
    ) is True
    assert _both_hands_wrists_visible(
        {
            "required_arms": ["left"],
            "reach_arm": "left",
            "arm_roles_in_frame": ["hand"],
        }
    ) is False


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _passing_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "dry_render" / "lightwheel_kitchen_task_01_sink_faucet"
    (task_dir / "dry_render_preview.png").parent.mkdir(parents=True, exist_ok=True)
    (task_dir / "dry_render_preview.png").write_bytes(b"png")
    _write_json(
        task_dir / "dry_render_summary.json",
        {
            "scene": {"object_count": 45},
            "target": {"resolution_source": "scene_placement_task_label"},
        },
    )
    _write_json(
        task_dir / "task_stance_plan.json",
        {
            "status": "accepted",
            "target_resolution": {
                "status": "resolved",
                "source": "scene_placement_task_label",
                "selected": {"target_object_id": "faucet", "target_object_label": "faucet"},
            },
            "affordance_resolution": {
                "status": "resolved",
                "source": "usd_prim_bounds",
                "selected": {
                    "target_object_id": "handle",
                    "prim_path": "/root/Sink054_01/Sink054_handle",
                    "center_xyz": [1.0, 2.0, 0.9],
                },
            },
            "task_affordance_xyz": [1.0, 2.0, 0.9],
            "reachability_selection_enabled": True,
            "selected_candidate_index": 0,
            "candidates": [
                {
                    "pose": [1.0, 1.52, 0.84],
                    "yaw": 1.570796,
                    "placement_validation": {"status": "accepted", "blockers": []},
                    "reachability_estimate": {
                        "status": "PASS",
                        "blockers": [],
                        "nearest_shoulder_to_affordance_m": 0.52,
                        "nearest_seed_effector_to_affordance_m": 0.28,
                        "passing_arms": ["right"],
                    },
                }
            ],
            "task_affordance_bounds": {
                "bbox_min_xyz": [0.95, 1.95, 0.85],
                "bbox_max_xyz": [1.05, 2.05, 0.95],
            },
        },
    )
    _write_json(task_dir / "placement_validation.json", {"status": "PASS", "blockers": []})
    _write_json(
        task_dir / "manipulation_pov_geometry.json",
        {
            "status": "PASS",
            "target_in_frame": True,
            "required_arms": ["left", "right"],
            "arm_roles_in_frame_by_arm": {
                "left": ["hand", "wrist"],
                "right": ["hand", "wrist"],
            },
            "arm_extension": {"status": "PASS", "blockers": []},
            "reach_feasibility": {"status": "PASS", "blockers": []},
            "target_affordance_xyz": [1.0, 2.0, 0.9],
            "target_projection": {"available": True, "u_px": 320, "v_px": 260},
            "camera_meta": {
                "camera_eye_xyz": [1.0, 1.0, 1.5],
                "camera_target_xyz": [1.0, 2.0, 0.9],
                "camera_vfov_deg": 90.0,
                "viewport_size_px": [640, 480],
                "arm_link_points_by_arm_xyz": {
                    "left": {
                        "shoulder": [0.8, 1.0, 1.2],
                        "elbow": [0.8, 1.2, 1.1],
                        "wrist": [0.8, 1.4, 1.0],
                        "hand": [0.8, 1.5, 0.95],
                    },
                    "right": {
                        "shoulder": [1.2, 1.0, 1.2],
                        "elbow": [1.2, 1.2, 1.1],
                        "wrist": [1.2, 1.4, 1.0],
                        "hand": [1.2, 1.5, 0.95],
                    },
                },
            },
        },
    )
    return task_dir


def test_evaluate_local_task_gates_passes_with_exact_sidecars(tmp_path: Path) -> None:
    task_dir = _passing_task_dir(tmp_path)

    report = evaluate_local_task_gates(
        task_spec=default_task_specs()[0],
        task_dir=task_dir,
        min_scene_objects=20,
    )

    assert report["status"] == "passed"
    assert all(gate["status"] == "PASS" for gate in report["local_gates"])
    assert [gate["status"] for gate in report["downstream_gates"]] == ["PENDING", "PENDING"]


def test_evaluate_local_task_gates_blocks_stale_coordinate_resolution(tmp_path: Path) -> None:
    task_dir = _passing_task_dir(tmp_path)
    _write_json(
        task_dir / "task_stance_plan.json",
        {
            "status": "accepted",
            "target_resolution": {
                "status": "resolved",
                "source": "usd_prim_bounds",
                "selected": {"target_object_id": "manual_stale_target"},
            },
        },
    )

    report = evaluate_local_task_gates(
        task_spec=default_task_specs()[0],
        task_dir=task_dir,
        min_scene_objects=20,
    )

    assert report["status"] == "blocked"
    semantic_gate = next(g for g in report["local_gates"] if g["name"] == "target resolves semantically")
    assert semantic_gate["status"] == "FAIL"


def test_evaluate_local_task_gates_accepts_ordered_usd_alias_resolution(tmp_path: Path) -> None:
    task_dir = _passing_task_dir(tmp_path)
    _write_json(
        task_dir / "task_stance_plan.json",
        {
            "status": "accepted",
            "target_resolution": {
                "status": "resolved",
                "source": "usd_prim_bounds",
                "selected": {
                    "target_object_id": "sink",
                    "target_object_priority": 0,
                    "prim_path": "/root/Sink054_01",
                },
            },
        },
    )

    report = evaluate_local_task_gates(
        task_spec=default_task_specs()[0],
        task_dir=task_dir,
        min_scene_objects=20,
    )

    semantic_gate = next(g for g in report["local_gates"] if g["name"] == "target resolves semantically")
    assert semantic_gate["status"] == "PASS"


def test_evaluate_local_task_gates_blocks_unreachable_seed_geometry(tmp_path: Path) -> None:
    task_dir = _passing_task_dir(tmp_path)
    geometry_path = task_dir / "manipulation_pov_geometry.json"
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    payload["reach_feasibility"] = {
        "status": "FAIL",
        "blockers": ["manipulation_pov_effector_too_far_from_affordance"],
    }
    _write_json(geometry_path, payload)

    report = evaluate_local_task_gates(
        task_spec=default_task_specs()[0],
        task_dir=task_dir,
        min_scene_objects=20,
    )

    assert report["status"] == "blocked"
    assert report["reachability_evidence"]["status"] == "FAIL"
    assert report["reachability_evidence"]["static_reach_required_for_local_preflight"] is True


def test_evaluate_local_task_gates_blocks_failed_geometry_sidecar(tmp_path: Path) -> None:
    task_dir = _passing_task_dir(tmp_path)
    geometry_path = task_dir / "manipulation_pov_geometry.json"
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    payload["status"] = "FAIL"
    payload["blockers"] = ["manipulation_pov_camera_pitched_down_too_far"]
    _write_json(geometry_path, payload)

    report = evaluate_local_task_gates(
        task_spec=default_task_specs()[0],
        task_dir=task_dir,
        min_scene_objects=20,
    )

    assert report["status"] == "blocked"
    geometry_gate = next(
        g for g in report["local_gates"]
        if g["name"] == "manipulation POV has no non-reach framing blockers"
    )
    assert geometry_gate["status"] == "FAIL"
    assert geometry_gate["evidence"]["non_reach_blockers"] == [
        "manipulation_pov_camera_pitched_down_too_far"
    ]


def test_evaluate_local_task_gates_blocks_unreachable_reach_ranked_start(tmp_path: Path) -> None:
    task_dir = _passing_task_dir(tmp_path)
    stance_path = task_dir / "task_stance_plan.json"
    payload = json.loads(stance_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "reachability_selection_enabled": True,
            "selected_candidate_index": 0,
            "candidates": [
                {
                    "pose": [1.6, 1.3, 0.84],
                    "reachability_estimate": {
                        "status": "FAIL",
                        "blockers": ["manipulation_pov_affordance_outside_g1_reach_envelope"],
                        "max_shoulder_to_affordance_m": 0.86,
                    },
                }
            ],
        }
    )
    _write_json(stance_path, payload)

    report = evaluate_local_task_gates(
        task_spec=default_task_specs()[0],
        task_dir=task_dir,
        min_scene_objects=20,
    )

    assert report["status"] == "blocked"
    reachability = report["reachability_evidence"]["selected_candidate_reachability"]
    assert reachability["status"] == "FAIL"
    assert reachability["passing_candidate_count"] == 0
    assert (
        report["reachability_evidence"][
            "selected_candidate_reachability_required_for_local_preflight"
        ]
        is True
    )
    assert report["reachability_evidence"]["static_reach_required_for_local_preflight"] is True


def test_evaluate_local_task_gates_reports_reach_clearance_conflict(tmp_path: Path) -> None:
    task_dir = _passing_task_dir(tmp_path)
    stance_path = task_dir / "task_stance_plan.json"
    payload = json.loads(stance_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "reachability_selection_enabled": True,
            "selected_candidate_index": 1,
            "candidates": [
                {
                    "pose": [0.56, 1.59, 0.84],
                    "yaw": 1.570796,
                    "standoff_from_target_surface_m": 0.30,
                    "angle_offset_deg": 90,
                    "placement_validation": {
                        "status": "blocked",
                        "blockers": ["placement_geometry_invalid"],
                    },
                    "reachability_estimate": {
                        "status": "PASS",
                        "blockers": [],
                        "nearest_shoulder_to_affordance_m": 0.6074,
                        "nearest_seed_effector_to_affordance_m": 0.3026,
                    },
                },
                {
                    "pose": [0.56, 1.51, 0.84],
                    "yaw": 1.570796,
                    "standoff_from_target_surface_m": 0.38,
                    "angle_offset_deg": 90,
                    "placement_validation": {"status": "accepted", "blockers": []},
                    "reachability_estimate": {
                        "status": "FAIL",
                        "blockers": [
                            "manipulation_pov_affordance_outside_g1_reach_envelope"
                        ],
                        "nearest_shoulder_to_affordance_m": 0.6756,
                        "nearest_seed_effector_to_affordance_m": 0.3486,
                    },
                },
            ],
        }
    )
    _write_json(stance_path, payload)

    report = evaluate_local_task_gates(
        task_spec=default_task_specs()[1],
        task_dir=task_dir,
        min_scene_objects=20,
    )

    conflict = report["reachability_evidence"]["reach_clearance_conflict"]
    assert report["status"] == "blocked"
    assert conflict["status"] == "detected"
    assert conflict["reachable_but_placement_blocked_count"] == 1
    assert conflict["placement_clean_but_reach_blocked_count"] == 1
    assert conflict["next_step"] == "improve_initial_arm_or_torso_seed_before_wam"


def test_export_policy_observation_from_preflight_requires_passed_task(
    tmp_path: Path,
) -> None:
    task_dir = _passing_task_dir(tmp_path)
    report = evaluate_local_task_gates(
        task_spec=default_task_specs()[0],
        task_dir=task_dir,
        min_scene_objects=20,
    )
    manifest_path = tmp_path / "kitchen_task_scaling_preflight_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "kitchen_task_scaling_preflight.v1",
            "status": "passed_local_preflight",
            "tasks": [report],
        },
    )

    export = export_policy_observation_from_preflight(
        preflight_manifest_path=manifest_path,
        task_id="sink_faucet",
        out_dir=tmp_path / "exported_seed",
        generated_at="now",
    )

    assert export["status"] == "completed"
    assert export["target_object_id"] == "handle"
    assert export["action_projection_bridge_readiness"]["status"] == "ready"
    observation_path = Path(export["policy_observation_path"])
    observation = json.loads(observation_path.read_text(encoding="utf-8"))["observation"]
    visual = observation["visual_observation"]
    assert observation["schema_version"] == "initial_policy_observation.v1"
    assert observation["camera_frame_path"].endswith("dry_render_preview.png")
    assert observation["target_object_id"] == "handle"
    assert observation["unitree_g1_sonic_state"]["projected_gravity"] == [0.0, 0.0, -1.0]
    assert observation["claim_boundary"]["wam_visual_success_proven"] is False
    assert visual["manipulation_pov_geometry_path"].endswith("manipulation_pov_geometry.json")
    assert visual["camera_meta"]["viewport_size_px"] == [640, 480]
    assert Path(observation["wam_auxiliary_observation_manifest_path"]).is_file()


def test_export_policy_observation_blocks_failed_task(tmp_path: Path) -> None:
    task_dir = _passing_task_dir(tmp_path)
    geometry_path = task_dir / "manipulation_pov_geometry.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    geometry["status"] = "FAIL"
    geometry["target_in_frame"] = False
    geometry["blockers"] = ["manipulation_pov_target_not_visible"]
    _write_json(geometry_path, geometry)
    report = evaluate_local_task_gates(
        task_spec=default_task_specs()[0],
        task_dir=task_dir,
        min_scene_objects=20,
    )
    manifest_path = tmp_path / "kitchen_task_scaling_preflight_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "kitchen_task_scaling_preflight.v1",
            "status": "blocked",
            "tasks": [report],
        },
    )

    export = export_policy_observation_from_preflight(
        preflight_manifest_path=manifest_path,
        task_id="sink_faucet",
        out_dir=tmp_path / "blocked_seed",
        generated_at="now",
    )

    assert export["status"] == "blocked"
    assert "kitchen_task_local_preflight_not_passed" in export["blockers"]
    assert "kitchen_task_action_projection_bridge_not_ready" in export["blockers"]
    assert not (tmp_path / "blocked_seed" / "initial_policy_observation.json").exists()


def test_export_policy_observation_marks_dry_render_preview_not_wam_eligible(
    tmp_path: Path,
) -> None:
    task_dir = _passing_task_dir(tmp_path)
    summary_path = task_dir / "dry_render_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["render_source"] = "dry_render_preview"
    summary["render_provenance"] = {
        "render_source_note": "NOT a rendered frame: CPU-only dry-render preview."
    }
    _write_json(summary_path, summary)
    report = evaluate_local_task_gates(
        task_spec=default_task_specs()[0],
        task_dir=task_dir,
        min_scene_objects=20,
    )
    manifest_path = tmp_path / "kitchen_task_scaling_preflight_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "kitchen_task_scaling_preflight.v1",
            "status": "passed_local_preflight",
            "tasks": [report],
        },
    )

    export = export_policy_observation_from_preflight(
        preflight_manifest_path=manifest_path,
        task_id="sink_faucet",
        out_dir=tmp_path / "exported_dry_seed",
        generated_at="now",
    )

    assert export["status"] == "completed"
    assert export["wam_seed_eligibility"]["status"] == "blocked"
    assert "source_policy_observation_is_dry_render_debug_preview" in export[
        "wam_seed_eligibility"
    ]["blockers"]
    assert export["next_step"] == "render_review_quality_isaac_rgb_policy_observation_before_wam"


def test_export_all_policy_observations_writes_task_index(tmp_path: Path) -> None:
    task_dir = _passing_task_dir(tmp_path)
    report = evaluate_local_task_gates(
        task_spec=default_task_specs()[0],
        task_dir=task_dir,
        min_scene_objects=20,
    )
    manifest_path = tmp_path / "kitchen_task_scaling_preflight_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "kitchen_task_scaling_preflight.v1",
            "status": "passed_local_preflight",
            "tasks": [report],
        },
    )

    index = export_all_policy_observations_from_preflight(
        preflight_manifest_path=manifest_path,
        generated_at="now",
    )

    assert index["status"] == "completed"
    assert index["tasks"][0]["task_id"] == "sink_faucet"
    assert index["tasks"][0]["action_projection_bridge_readiness"]["status"] == "ready"
    assert index["all_action_projection_bridges_ready"] is True
    assert index["all_wam_seed_frames_review_quality_eligible"] is True
    assert (tmp_path / "wam_seed" / "sink_faucet" / "initial_policy_observation.json").is_file()
    assert (tmp_path / "wam_seed" / "kitchen_task_policy_observation_export_index.json").is_file()


def test_export_all_policy_observations_indexes_dry_seed_rgb_blocker(tmp_path: Path) -> None:
    task_dir = _passing_task_dir(tmp_path)
    summary_path = task_dir / "dry_render_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["render_source"] = "dry_render_preview"
    summary["render_provenance"] = {
        "render_source_note": "NOT a rendered frame: CPU-only dry-render preview."
    }
    _write_json(summary_path, summary)
    report = evaluate_local_task_gates(
        task_spec=default_task_specs()[0],
        task_dir=task_dir,
        min_scene_objects=20,
    )
    manifest_path = tmp_path / "kitchen_task_scaling_preflight_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "kitchen_task_scaling_preflight.v1",
            "status": "passed_local_preflight",
            "tasks": [report],
        },
    )

    index = export_all_policy_observations_from_preflight(
        preflight_manifest_path=manifest_path,
        generated_at="now",
    )

    assert index["status"] == "completed"
    assert index["all_action_projection_bridges_ready"] is True
    assert index["all_wam_seed_frames_review_quality_eligible"] is False
    assert index["next_step"] == "render_review_quality_isaac_rgb_policy_observations_before_wam"
    assert index["tasks"][0]["wam_seed_eligibility"]["status"] == "blocked"


def test_build_request_keeps_targets_deferred_to_scene_semantics(tmp_path: Path) -> None:
    request = build_request(kitchen_usd=tmp_path / "KitchenRoom.usd", task_specs=default_task_specs())

    assert len(request["scenarios"]) == 3
    for scenario in request["scenarios"]:
        assert scenario["task_target_deferred"] is True
        assert scenario["task_id"]
        assert "target_object_id" not in scenario
        assert "task_target_position_xyz" not in scenario
        assert scenario["description"]
        assert scenario["perception_target_prompts"]
        assert scenario["target_object_ids"]
        assert "affordance_object_ids" in scenario


def test_perception_target_prompts_include_task_affordances() -> None:
    prompts = perception_target_prompts_for_task(default_task_specs()[0])

    assert "faucet" in prompts
    assert "faucet lever" in prompts
    assert "sink faucet handle" in prompts


def test_target_object_candidates_separate_fixture_from_affordance() -> None:
    fixture_candidates = target_object_id_candidates_for_task(default_task_specs()[0])
    affordance_candidates = affordance_object_id_candidates_for_task(default_task_specs()[0])
    top_cabinet_fixture_candidates = target_object_id_candidates_for_task(default_task_specs()[2])

    assert fixture_candidates == ["sink", "basin"]
    assert affordance_candidates[:2] == ["handle", "lever"]
    assert "spout" in affordance_candidates
    assert affordance_candidates.index("handle") < affordance_candidates.index("spout")
    assert top_cabinet_fixture_candidates[0] == "topcabinet"
    assert top_cabinet_fixture_candidates.index("topcabinet") < top_cabinet_fixture_candidates.index("cabinet")
    assert affordance_object_id_candidates_for_task(
        {"task_id": "microwave_door"}
    ) == ["microwave_handle", "handle", "door"]
    assert affordance_object_id_candidates_for_task(
        {"task_id": "dishwasher_door"}
    ) == ["dishwasher_handle", "handle", "door"]


def test_run_preflight_blocks_without_full_kitchen_usd(tmp_path: Path) -> None:
    manifest = run_preflight(
        out_dir=tmp_path / "preflight",
        kitchen_usd=tmp_path / "missing" / "KitchenRoom.usd",
    )

    assert manifest["status"] == "blocked"
    assert "missing_full_kitchen_usd" in manifest["blockers"]
    assert (tmp_path / "preflight" / "kitchen_task_scaling_preflight_manifest.json").is_file()
