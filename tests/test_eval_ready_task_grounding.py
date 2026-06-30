from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import eval_ready_task_grounding as grounding


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
    _write_json(camera, {"fx": 800, "fy": 800, "cx": 320, "cy": 240, "width": 640, "height": 480})
    scene = tmp_path / "kitchen.splat"
    scene.write_text("static 3dgs placeholder", encoding="utf-8")
    initial_frame = tmp_path / "initial.png"
    initial_frame.write_bytes(b"png")
    robot_model = tmp_path / "unitree_g1.xml"
    robot_model.write_text("<mujoco/>", encoding="utf-8")
    robot_state = tmp_path / "robot_state.json"
    _write_json(
        robot_state,
        {
            "right_arm": [0.0] * 7,
            "right_hand": [0.0] * 7,
            "right_end_effector_xyz": [0.005, -0.13, 2.0],
            "right_wrist_rotation_delta_deg": 22.0,
        },
    )

    manifest = grounding.build_eval_ready_task_grounding(
        capture_root=capture_root,
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
    assert manifest["handle_proxy_state_check"]["handle_proxy_state"] == "on_candidate"
    assert manifest["readiness"]["exact_task_success_proven"] is False
    assert manifest["readiness"]["physical_contact_validated"] is False
    assert manifest["articulated_state_proxy"]["axis"] == [0.0, 1.0, 0.0]
    assert manifest["articulated_state_proxy"]["state_success_proven"] is False
    assert Path(manifest["output_path"]).is_file()


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

    manifest = grounding.build_eval_ready_task_grounding(capture_root=capture_root)

    assert manifest["status"] == "blocked"
    assert manifest["selected_task_target"]["object_id"] == "sink_parent"
    assert manifest["parent_context_target"]["object_id"] == "sink_parent"
    assert "missing_task_specific_handle_label_or_keypoint" in manifest["readiness"]["blockers"]
    assert "missing_camera_calibration" in manifest["readiness"]["blockers"]
    assert manifest["readiness"]["learned_rollout_request_ready"] is False
    assert manifest["claim_boundary"]["learned_wam_rollout_is_not_physical_success_proof"] is True


def test_eval_ready_task_grounding_cli_writes_summary(tmp_path: Path, capsys) -> None:
    capture_root = tmp_path / "capture"
    (capture_root / "raw").mkdir(parents=True)
    _write_json(capture_root / "raw" / "object_index.json", {"objects": []})

    assert grounding.main(["--capture-root", str(capture_root)]) == 0
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
    _write_json(camera, {"fx": 800, "fy": 800, "cx": 320, "cy": 240, "width": 640, "height": 480})
    scene = tmp_path / "kitchen.splat"
    scene.write_text("static 3dgs placeholder", encoding="utf-8")
    initial_frame = tmp_path / "initial.png"
    initial_frame.write_bytes(b"png")
    robot_model = tmp_path / "unitree_g1.xml"
    robot_model.write_text("<mujoco/>", encoding="utf-8")
    robot_state = tmp_path / "robot_state.json"
    _write_json(
        robot_state,
        {
            "right_arm": [0.0] * 7,
            "right_hand": [0.0] * 7,
            "right_end_effector_xyz": [0.005, -0.13, 2.0],
            "right_wrist_rotation_delta_deg": 22.0,
        },
    )
    return grounding.build_eval_ready_task_grounding(
        capture_root=capture_root,
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
        "camera_calibration_quality_gate",
        "robot_fk_projection",
        "handle_proxy_state_check",
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
