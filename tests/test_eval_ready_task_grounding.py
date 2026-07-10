from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import eval_ready_task_grounding as grounding


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_projection_ready_inputs(
    *,
    camera: Path,
    robot_model: Path,
    robot_state: Path,
    state_extras: dict[str, object] | None = None,
) -> None:
    _write_json(
        camera,
        {
            "fx": 800,
            "fy": 800,
            "cx": 320,
            "cy": 240,
            "width": 640,
            "height": 480,
            "camera_from_world": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            "reference_frame": "world",
            "camera_frame": "head_camera",
            "translation_unit": "meters",
            "reprojection_error_px": 0.5,
        },
    )
    robot_model.write_text(
        """<robot name="test_robot">
  <link name="base_link"/>
  <link name="right_end_effector"/>
  <joint name="right_ee_slide" type="prismatic">
    <parent link="base_link"/>
    <child link="right_end_effector"/>
    <origin xyz="0 -0.13 2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="1" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "angle_unit": "radians",
        "linear_unit": "meters",
        "timestamp_unit": "seconds",
        "reference_frame": "world",
        "base_frame": "base_link",
        "reference_tolerance_m": 1e-6,
        "max_link_step_m": 0.1,
        "joint_state_frames": [
            {
                "timestamp": 0.0,
                "joint_names": ["right_ee_slide"],
                "joint_positions": [0.005],
                "world_from_robot_base": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                "expected_link_positions": {
                    "base_link": [0, 0, 0],
                    "right_end_effector": [0.005, -0.13, 2.0],
                },
            },
            {
                "timestamp": 0.1,
                "joint_names": ["right_ee_slide"],
                "joint_positions": [0.006],
                "world_from_robot_base": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                "expected_link_positions": {
                    "base_link": [0, 0, 0],
                    "right_end_effector": [0.006, -0.13, 2.0],
                },
            },
        ],
    }
    payload.update(state_extras or {})
    _write_json(robot_state, payload)


def test_eval_ready_task_grounding_builds_ready_sink_handle_contract(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True)
    _write_json(
        raw_root / "object_index.json",
        {
            "objects": [
                {
                    "object_id": "sink_parent",
                    "label": "sink",
                    "mean_confidence": 0.82,
                    "reference_crop": "object_index_artifacts/crops/sink.png",
                    "all_crops": ["object_index_artifacts/crops/sink.png"],
                },
                {
                    "object_id": "right_sink_handle_01",
                    "label": "right sink handle",
                    "source_prompt": "right sink handle",
                    "mean_confidence": 0.93,
                    "reference_crop": "object_index_artifacts/crops/right_sink_handle.png",
                    "all_crops": ["object_index_artifacts/crops/right_sink_handle.png"],
                    "keypoints": {"center": [322, 188]},
                    "mean_box_px": {"x": 302, "y": 168, "width": 40, "height": 40},
                },
            ]
        },
    )
    _write_json(
        capture_root / "pipeline" / "evaluation_prep" / "task_anchor_manifest.json",
        {
            "tasks": [
                {
                    "task_id": "turn_on_sink_handle",
                    "target_object_ids": ["right_sink_handle_01"],
                    "anchor_source": "operator_reviewed_detection",
                }
            ]
        },
    )
    camera = capture_root / "pipeline" / "geometry" / "camera" / "intrinsics.json"
    scene = tmp_path / "kitchen.splat"
    scene.write_text("static 3dgs placeholder", encoding="utf-8")
    initial_frame = tmp_path / "initial.png"
    initial_frame.write_bytes(b"png")
    robot_model = tmp_path / "unitree_g1.xml"
    robot_state = tmp_path / "robot_state.json"
    _write_projection_ready_inputs(
        camera=camera,
        robot_model=robot_model,
        robot_state=robot_state,
        state_extras={"right_wrist_rotation_delta_deg": 22.0},
    )

    manifest = grounding.build_eval_ready_task_grounding(
        capture_root=capture_root,
        task_id="turn_on_sink_handle",
        task_text="turn on the sink right handle",
        target_label="right sink handle",
        scene_asset=scene,
        initial_frame=initial_frame,
        camera_calibration=camera,
        robot_model=robot_model,
        robot_state=robot_state,
        articulated_handle_proxy=True,
        handle_axis="0,1,0",
    )

    assert manifest["status"] == "ready_for_learned_wam_rollout_request"
    assert manifest["selected_task_target"]["object_id"] == "right_sink_handle_01"
    assert manifest["readiness"]["learned_rollout_request_ready"] is True
    assert manifest["readiness"]["robot_projection_ready"] is True
    assert manifest["robot_fk_projection"]["status"] == "completed"
    assert manifest["robot_fk_projection"]["urdf_or_mjcf_fk_solver_executed"] is True
    assert manifest["robot_fk_projection"]["aligned_step_count"] == 2
    assert manifest["handle_proxy_state_check"]["handle_proxy_state"] == "on_candidate"
    assert manifest["readiness"]["exact_task_success_proven"] is False
    assert manifest["readiness"]["physical_contact_validated"] is False
    assert manifest["articulated_state_proxy"]["axis"] == [0.0, 1.0, 0.0]
    assert manifest["articulated_state_proxy"]["state_success_proven"] is False
    assert manifest["task"]["support_level"] == "support_only"
    assert manifest["readiness"]["buyer_grade_eligible"] is False
    assert Path(manifest["output_path"]).is_file()

    missing_contract = grounding.build_eval_ready_task_grounding(
        capture_root=capture_root,
        task_id="turn_on_sink_handle",
        task_text="turn on the sink right handle",
        target_label="right sink handle",
        scene_asset=scene,
        initial_frame=initial_frame,
        camera_calibration=camera,
        robot_model=robot_model,
        robot_state=robot_state,
        articulated_handle_proxy=True,
        handle_axis="0,1,0",
        buyer_grade=True,
    )
    assert missing_contract["status"] == "blocked"
    assert "task_contract_missing" in missing_contract["readiness"]["blockers"]

    buyer_contract = {
        "schema_version": grounding.TASK_CONTRACT_SCHEMA_VERSION,
        "task_id": "turn_on_sink_handle",
        "task_text": "turn on the sink right handle",
        "target": {
            "object_id": "right_sink_handle_01",
            "label": "right sink handle",
        },
        "transition": {
            "type": "articulation_state_change",
            "source": "off",
            "destination": "on",
        },
        "evidence_requirements": [
            "reviewed target crop",
            "target keypoint",
            "measured joint state",
        ],
        "metric": {
            "name": "handle_joint_angle_error",
            "tolerance": {"value": 5.0, "unit": "degrees", "operator": "<="},
        },
        "evaluator_mapping": {
            "evaluator_id": "sink_handle_state_evaluator",
            "version": "1.0",
        },
    }
    buyer_manifest = grounding.build_eval_ready_task_grounding(
        capture_root=capture_root,
        task_id="turn_on_sink_handle",
        task_text="turn on the sink right handle",
        target_label="right sink handle",
        task_contract=buyer_contract,
        buyer_grade=True,
        scene_asset=scene,
        initial_frame=initial_frame,
        camera_calibration=camera,
        robot_model=robot_model,
        robot_state=robot_state,
        articulated_handle_proxy=True,
        handle_axis="0,1,0",
    )
    assert buyer_manifest["status"] == "ready_for_learned_wam_rollout_request"
    assert buyer_manifest["task"]["support_level"] == "buyer_grade"
    assert buyer_manifest["readiness"]["buyer_grade_eligible"] is True
    assert buyer_manifest["task"]["task_contract_validation"]["status"] == "passed"


def test_eval_ready_task_grounding_blocks_parent_sink_without_handle(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True)
    _write_json(
        raw_root / "object_index.json",
        {
            "objects": [
                {
                    "object_id": "sink_parent",
                    "label": "sink",
                    "mean_confidence": 0.84,
                    "reference_crop": "object_index_artifacts/crops/sink.png",
                    "all_crops": ["object_index_artifacts/crops/sink.png"],
                }
            ]
        },
    )

    manifest = grounding.build_eval_ready_task_grounding(
        capture_root=capture_root,
        task_id="turn_on_sink_handle",
        task_text="turn on the sink right handle",
        target_label="right sink handle",
    )

    assert manifest["status"] == "blocked"
    assert manifest["selected_task_target"]["object_id"] == "sink_parent"
    assert manifest["parent_context_target"]["object_id"] == "sink_parent"
    assert "missing_task_specific_handle_label_or_keypoint" in manifest["readiness"]["blockers"]
    assert "missing_camera_calibration" in manifest["readiness"]["blockers"]
    assert manifest["readiness"]["learned_rollout_request_ready"] is False
    assert manifest["claim_boundary"]["learned_wam_rollout_is_not_physical_success_proof"] is True


def test_eval_ready_grounding_rejects_cartesian_landmark_copy_and_reflected_camera(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    _write_json(
        capture_root / "raw" / "object_index.json",
        {
            "objects": [
                {
                    "object_id": "target_handle",
                    "label": "right sink handle",
                    "keypoints": {"center": [320, 240]},
                    "all_crops": ["crop.png"],
                }
            ]
        },
    )
    camera = capture_root / "raw" / "camera_calibration.json"
    model = tmp_path / "robot.urdf"
    state = tmp_path / "robot_state.json"
    _write_projection_ready_inputs(camera=camera, robot_model=model, robot_state=state)
    calibration = json.loads(camera.read_text(encoding="utf-8"))
    calibration["camera_from_world"][0][0] = -1.0
    _write_json(camera, calibration)
    _write_json(
        state,
        {
            "right_end_effector_xyz": [0.0, 0.0, 2.0],
            "fk_landmarks": {"right_end_effector": [0.0, 0.0, 2.0]},
        },
    )
    scene = tmp_path / "scene.splat"
    scene.write_text("scene", encoding="utf-8")

    manifest = grounding.build_eval_ready_task_grounding(
        capture_root=capture_root,
        task_id="turn_handle",
        task_text="turn the right sink handle",
        target_label="right sink handle",
        scene_asset=scene,
        camera_calibration=camera,
        robot_model=model,
        robot_state=state,
    )

    assert manifest["status"] == "blocked"
    assert manifest["camera_calibration_quality_gate"]["projection_ready"] is False
    assert "camera_from_reference_rotation_not_right_handed" in manifest[
        "camera_calibration_quality_gate"
    ]["blockers"]
    assert manifest["robot_fk_projection"]["urdf_or_mjcf_fk_solver_executed"] is False
    assert "robot_joint_state_sequence_missing" in manifest["robot_fk_projection"]["blockers"]


def test_eval_ready_task_grounding_default_does_not_emit_sink_task_for_sinkless_site(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "warehouse_capture"
    _write_json(
        capture_root / "raw" / "object_index.json",
        {
            "objects": [
                {
                    "object_id": "rack_01",
                    "label": "storage rack",
                    "mean_confidence": 0.87,
                    "reference_crop": "object_index_artifacts/crops/rack.png",
                    "all_crops": ["object_index_artifacts/crops/rack.png"],
                    "keypoints": {"center": [300, 200]},
                }
            ]
        },
    )

    manifest = grounding.build_eval_ready_task_grounding(capture_root=capture_root)

    assert manifest["task"]["task_id"] == "auto_inspect_storage_rack"
    assert manifest["task"]["task_text"] == "inspect the storage rack"
    assert manifest["task"]["target_label"] == "storage rack"
    assert manifest["task"]["default_task_replaces_legacy_template"] is True
    assert "sink" not in json.dumps(manifest["task"]).lower()
    assert "water appears or faucet state visibly changes" not in manifest[
        "success_check_plan"
    ]["vlm_or_human_review_checks"]
    assert "missing_task_specific_handle_label_or_keypoint" not in manifest[
        "readiness"
    ]["blockers"]
    assert manifest["selected_task_target"]["object_id"] == "rack_01"


def test_eval_ready_task_grounding_task_id_only_does_not_emit_legacy_sink_template(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "warehouse_capture"
    _write_json(
        capture_root / "raw" / "object_index.json",
        {
            "objects": [
                {
                    "object_id": "cart_01",
                    "label": "utility cart",
                    "mean_confidence": 0.88,
                    "reference_crop": "object_index_artifacts/crops/cart.png",
                    "all_crops": ["object_index_artifacts/crops/cart.png"],
                    "keypoints": {"center": [280, 220]},
                }
            ]
        },
    )

    manifest = grounding.build_eval_ready_task_grounding(
        capture_root=capture_root,
        task_id="operator_supplied_task_id_without_contract",
    )

    assert manifest["task"]["task_id"] == "operator_supplied_task_id_without_contract"
    assert manifest["task"]["task_text"] == "inspect the utility cart"
    assert manifest["task"]["target_label"] == "utility cart"
    assert manifest["task"]["default_task_replaces_legacy_template"] is True
    assert (
        manifest["task"]["partial_explicit_task_contract_defaulted_to_object_index"]
        is True
    )
    assert manifest["task"]["provided_task_id"] == "operator_supplied_task_id_without_contract"
    assert "sink" not in json.dumps(manifest["task"]).lower()
    assert "right handle visibly moved in the intended direction" not in manifest[
        "success_check_plan"
    ]["vlm_or_human_review_checks"]
    assert "missing_task_specific_handle_label_or_keypoint" not in manifest[
        "readiness"
    ]["blockers"]
    assert manifest["selected_task_target"]["object_id"] == "cart_01"


def test_eval_ready_task_grounding_default_does_not_emit_sink_handle_task_for_site_with_sink(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "mixed_capture"
    _write_json(
        capture_root / "raw" / "object_index.json",
        {
            "objects": [
                {
                    "object_id": "sink_parent",
                    "label": "sink",
                    "mean_confidence": 0.86,
                    "reference_crop": "object_index_artifacts/crops/sink.png",
                    "all_crops": ["object_index_artifacts/crops/sink.png"],
                },
                {
                    "object_id": "rack_01",
                    "label": "storage rack",
                    "mean_confidence": 0.87,
                    "reference_crop": "object_index_artifacts/crops/rack.png",
                    "all_crops": ["object_index_artifacts/crops/rack.png"],
                    "keypoints": {"center": [300, 200]},
                },
            ]
        },
    )

    manifest = grounding.build_eval_ready_task_grounding(capture_root=capture_root)

    assert manifest["task"]["default_task_replaces_legacy_template"] is True
    assert manifest["task"]["task_text"] == "inspect the sink"
    assert manifest["task"]["target_label"] == "sink"
    assert manifest["task"]["task_text"] != "turn on the sink right handle"
    assert "right sink handle" not in json.dumps(manifest["task"]).lower()
    assert "missing_task_specific_handle_label_or_keypoint" not in manifest[
        "readiness"
    ]["blockers"]


def test_eval_ready_task_grounding_builds_industrial_containment_proxy(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "warehouse_capture"
    _write_json(
        capture_root / "raw" / "object_index.json",
        {
            "objects": [
                {
                    "object_id": "target_bin_01",
                    "label": "target bin",
                    "source_prompt": "target bin",
                    "mean_confidence": 0.9,
                    "reference_crop": "object_index_artifacts/crops/bin.png",
                    "all_crops": ["object_index_artifacts/crops/bin.png"],
                    "keypoints": {"center": [310, 210]},
                    "mean_box_px": {"x": 280, "y": 180, "width": 60, "height": 60},
                }
            ]
        },
    )
    _write_json(
        capture_root / "pipeline" / "evaluation_prep" / "task_anchor_manifest.json",
        {
            "tasks": [
                {
                    "task_id": "place_object_into_bin",
                    "target_object_ids": ["target_bin_01"],
                    "target_bin_id": "target_bin_01",
                    "destination_zone_id": "warehouse_packout_bin_zone",
                    "anchor_source": "operator_reviewed_detection",
                }
            ]
        },
    )
    camera = capture_root / "pipeline" / "geometry" / "camera" / "intrinsics.json"
    scene = tmp_path / "warehouse.splat"
    scene.write_text("static warehouse 3dgs placeholder", encoding="utf-8")
    initial_frame = tmp_path / "initial.png"
    initial_frame.write_bytes(b"png")
    robot_model = tmp_path / "unitree_g1.xml"
    robot_state = tmp_path / "robot_state.json"
    _write_projection_ready_inputs(
        camera=camera,
        robot_model=robot_model,
        robot_state=robot_state,
        state_extras={
            "payload_center_xyz": [1.0, 2.0, 0.55],
            "target_zone_center_xyz": [1.02, 2.01, 0.55],
            "target_zone_aabb": {
                "min": [0.8, 1.8, 0.25],
                "max": [1.2, 2.2, 0.85],
            },
            "placement_tolerance_m": 0.2,
        },
    )

    manifest = grounding.build_eval_ready_task_grounding(
        capture_root=capture_root,
        task_id="place_object_into_bin",
        task_text="place the tote into the target bin",
        target_label="target bin",
        scene_asset=scene,
        initial_frame=initial_frame,
        camera_calibration=camera,
        robot_model=robot_model,
        robot_state=robot_state,
    )

    assert manifest["status"] == "ready_for_learned_wam_rollout_request"
    assert manifest["industrial_state_proxy"]["proxy_type"] == "industrial_containment"
    assert manifest["industrial_proxy_state_check"]["status"] == "completed"
    assert manifest["industrial_proxy_state_check"]["containment_candidate"] is True
    assert manifest["industrial_proxy_state_check"]["proxy_success_candidate"] is True
    assert manifest["readiness"]["industrial_proxy_configured"] is True
    assert manifest["readiness"]["industrial_proxy_success_candidate"] is True
    assert manifest["readiness"]["exact_task_success_proven"] is False
    assert manifest["industrial_proxy_state_check"]["state_success_proven"] is False
    assert Path(manifest["generated_artifacts"]["industrial_proxy_state_check"]).is_file()


def test_committed_warehouse_fixture_exercises_industrial_grounding_gate(
    tmp_path: Path,
) -> None:
    fixture = Path("tests/fixtures/warehouse_task_min").resolve()
    output_path = tmp_path / "warehouse_fixture_grounding.json"

    manifest = grounding.build_eval_ready_task_grounding(
        capture_root=fixture,
        task_id="fixture_warehouse_place_tote_into_bin",
        task_text="place the tote into the target bin",
        target_label="target bin",
        scene_asset=fixture / "assets" / "warehouse_min.splat",
        initial_frame=fixture / "assets" / "initial_policy_observation.jpg",
        camera_calibration=fixture / "pipeline" / "geometry" / "camera" / "intrinsics.json",
        robot_model=fixture / "assets" / "unitree_g1.xml",
        robot_state=fixture / "assets" / "robot_state.json",
        output_path=output_path,
    )

    assert manifest["status"] == "ready_for_learned_wam_rollout_request"
    assert manifest["selected_task_target"]["object_id"] == "target_bin_01"
    assert manifest["industrial_state_proxy"]["available"] is True
    assert manifest["industrial_proxy_state_check"]["proxy_success_candidate"] is True
    assert manifest["claim_boundary"]["learned_wam_rollout_is_not_physical_success_proof"] is True
    assert manifest["readiness"]["exact_task_success_proven"] is False


def test_eval_ready_task_grounding_cli_writes_summary(tmp_path: Path, capsys) -> None:
    capture_root = tmp_path / "capture"
    (capture_root / "raw").mkdir(parents=True)
    _write_json(capture_root / "raw" / "object_index.json", {"objects": []})

    assert grounding.main(["--capture-root", str(capture_root)]) == 2
    out = json.loads(capsys.readouterr().out)

    assert out["status"] == "blocked"
    assert out["eval_ready_task_grounding"].endswith("eval_ready_task_grounding.json")
    assert "missing_task_target_label_or_keypoint" in out["blockers"]


def _build_ready_manifest(tmp_path: Path) -> dict:
    """Reuse the ready-fixture flow from the sink-handle contract test."""
    capture_root = tmp_path / "capture"
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True)
    _write_json(
        raw_root / "object_index.json",
        {
            "objects": [
                {
                    "object_id": "sink_parent",
                    "label": "sink",
                    "mean_confidence": 0.82,
                    "reference_crop": "object_index_artifacts/crops/sink.png",
                    "all_crops": ["object_index_artifacts/crops/sink.png"],
                },
                {
                    "object_id": "right_sink_handle_01",
                    "label": "right sink handle",
                    "source_prompt": "right sink handle",
                    "mean_confidence": 0.93,
                    "reference_crop": "object_index_artifacts/crops/right_sink_handle.png",
                    "all_crops": ["object_index_artifacts/crops/right_sink_handle.png"],
                    "keypoints": {"center": [322, 188]},
                    "mean_box_px": {"x": 302, "y": 168, "width": 40, "height": 40},
                },
            ]
        },
    )
    camera = capture_root / "pipeline" / "geometry" / "camera" / "intrinsics.json"
    scene = tmp_path / "kitchen.splat"
    scene.write_text("static 3dgs placeholder", encoding="utf-8")
    initial_frame = tmp_path / "initial.png"
    initial_frame.write_bytes(b"png")
    robot_model = tmp_path / "unitree_g1.xml"
    robot_state = tmp_path / "robot_state.json"
    _write_projection_ready_inputs(
        camera=camera,
        robot_model=robot_model,
        robot_state=robot_state,
        state_extras={"right_wrist_rotation_delta_deg": 22.0},
    )
    return grounding.build_eval_ready_task_grounding(
        capture_root=capture_root,
        task_id="turn_on_sink_handle",
        task_text="turn on the sink right handle",
        target_label="right sink handle",
        scene_asset=scene,
        initial_frame=initial_frame,
        camera_calibration=camera,
        robot_model=robot_model,
        robot_state=robot_state,
        articulated_handle_proxy=True,
        handle_axis="0,1,0",
    )


def test_eval_ready_task_grounding_schema_contract_lock(tmp_path: Path) -> None:
    """Lock every top-level key the hot lane consumes from the v1 schema.

    A ready manifest and a blocked manifest must both carry the schema version
    and the full set of keys downstream WAM rollout requesters read.
    """
    ready = _build_ready_manifest(tmp_path / "ready")
    blocked_root = tmp_path / "blocked" / "capture"
    (blocked_root / "raw").mkdir(parents=True)
    _write_json(blocked_root / "raw" / "object_index.json", {"objects": [{"object_id": "sink", "label": "sink"}]})
    blocked = grounding.build_eval_ready_task_grounding(capture_root=blocked_root)

    required_top_level = {
        "schema_version",
        "generated_at",
        "status",
        "capture_root",
        "task",
        "object_index",
        "selected_task_target",
        "parent_context_target",
        "task_relevant_region_candidates",
        "scene_and_observation_refs",
        "robot_conditioning_refs",
        "articulated_state_proxy",
        "industrial_state_proxy",
        "camera_calibration_quality_gate",
        "robot_fk_projection",
        "handle_proxy_state_check",
        "industrial_proxy_state_check",
        "generated_artifacts",
        "success_check_plan",
        "readiness",
        "claim_boundary",
        "output_path",
    }
    for manifest, expected_status in ((ready, "ready_for_learned_wam_rollout_request"), (blocked, "blocked")):
        assert manifest["schema_version"] == "eval_ready_task_grounding.v1"
        assert manifest["status"] == expected_status
        assert required_top_level <= set(manifest)

        readiness = manifest["readiness"]
        for key in (
            "learned_rollout_request_ready",
            "exact_task_success_proven",
            "physical_contact_validated",
            "real_world_readiness_proven",
            "blockers",
            "warnings",
        ):
            assert key in readiness
        assert isinstance(readiness["blockers"], list)
        assert isinstance(readiness["warnings"], list)

        # Hot lane reads nested check + gate payloads directly.
        assert manifest["success_check_plan"]["vlm_or_human_review_checks"]
        assert manifest["success_check_plan"]["deterministic_or_lightweight_checks"]
        assert "status" in manifest["handle_proxy_state_check"]
        assert "status" in manifest["camera_calibration_quality_gate"]
        assert manifest["camera_calibration_quality_gate"]["schema_version"] == "camera_calibration_quality_gate.v1"

        # generated_artifacts must expose every nested path key the hot lane resolves.
        artifacts = manifest["generated_artifacts"]
        for key in (
            "camera_calibration_quality_gate",
            "robot_fk_projection_manifest",
            "robot_fk_projected_skeleton_trace",
            "handle_proxy_state_check",
            "industrial_proxy_state_check",
        ):
            assert isinstance(artifacts.get(key), str) and artifacts[key]

        task = manifest["task"]
        assert task["task_id"]
        assert task["task_text"]
        assert task["target_label"]
        assert isinstance(task["target_prompts_for_object_index_backends"], list)

    assert ready["readiness"]["learned_rollout_request_ready"] is True
    assert blocked["readiness"]["learned_rollout_request_ready"] is False
    assert blocked["selected_task_target"]["object_id"] == "sink"


def test_derive_task_aware_detection_prompts() -> None:
    """Direct unit coverage for task->detection prompt derivation."""
    prompts = grounding.derive_task_aware_detection_prompts(
        task_text="turn on the sink right handle",
        target_label="right sink handle",
    )
    # Task-specific tokens surface as prompts.
    assert "right sink handle" in prompts
    assert "sink" in prompts
    assert "handle" in prompts
    # Sink/faucet semantics expand to faucet-specific prompts.
    assert "faucet handle" in prompts
    assert "water stream" in prompts
    # Output is de-duplicated and never empty for a real task.
    assert len(prompts) == len(set(prompts))
    assert prompts

    # Door/drawer/cabinet expansion adds a generic handle prompt.
    door_prompts = grounding.derive_task_aware_detection_prompts(
        task_text="open the cabinet drawer",
        target_label="cabinet drawer",
    )
    assert "handle" in door_prompts
    assert len(door_prompts) == len(set(door_prompts))

    industrial_prompts = grounding.derive_task_aware_detection_prompts(
        task_text="place the tote on the conveyor",
        target_label="tote conveyor",
    )
    assert "tote handle" in industrial_prompts
    assert "container interior" in industrial_prompts
    assert "conveyor belt edge" in industrial_prompts

    # Button/switch/panel spatial expansion.
    button_prompts = grounding.derive_task_aware_detection_prompts(
        task_text="press the left button on the panel",
        target_label="left button",
    )
    assert "left button" in button_prompts
    assert "left switch" in button_prompts

    # max_prompts caps the output length.
    capped = grounding.derive_task_aware_detection_prompts(
        task_text="turn on the sink right handle and open the cabinet drawer",
        target_label="right sink handle",
        max_prompts=3,
    )
    assert len(capped) <= 3

    # Empty/garbage task text defaults to an empty prompt list (no crash).
    assert grounding.derive_task_aware_detection_prompts(task_text="", target_label="") == []
    assert grounding.derive_task_aware_detection_prompts(task_text="   ", target_label="   ") == []
    assert grounding.derive_task_aware_detection_prompts(task_text="!!! @@@ ###", target_label="") == []
