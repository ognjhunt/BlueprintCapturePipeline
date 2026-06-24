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
