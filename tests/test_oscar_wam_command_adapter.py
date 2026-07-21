from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from tests.runpy_entrypoint import run_module_as_main


pytestmark = [pytest.mark.slow, pytest.mark.integration]

cv2 = pytest.importorskip("cv2")

from blueprint_pipeline import oscar_wam_command_adapter as adapter  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_review_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64))
    assert writer.isOpened()
    for index in range(4):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = 40 + index * 20
        frame[:, :, 1] = 90
        frame[:, :, 2] = 140
        cv2.rectangle(frame, (10 + index, 24), (52, 60), (210, 170, 72), -1)
        cv2.circle(frame, (24, 18), 7, (235, 235, 235), -1)
        writer.write(frame)
    writer.release()


def _write_dark_void_review_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64))
    assert writer.isOpened()
    for index in range(4):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.rectangle(frame, (8, 26), (55, 62), (70 + index * 8, 86, 92), -1)
        cv2.circle(frame, (32, 20), 8, (180, 180, 180), -1)
        writer.write(frame)
    writer.release()


def _write_projected_skeleton_trace(path: Path) -> None:
    _write_jsonl(
        path,
        [
            {
                "schema_version": "blueprint.mujoco_g1.projected_upper_body_skeleton.v1",
                "status": "completed",
                "episode_id": "episode_0001",
                "scenario_eval_run_id": "run",
                "task_id": "contact_or_push_light_object",
                "spawn_id": "doorway",
                "step": step,
                "projected_landmark_count": 4,
                "landmarks": [
                    {
                        "landmark_id": "left_shoulder",
                        "image_projection": {
                            "available": True,
                            "u_px": 6 + step,
                            "v_px": 38,
                            "inside_image": True,
                        },
                    },
                    {
                        "landmark_id": "left_hand",
                        "image_projection": {
                            "available": True,
                            "u_px": 24 + step,
                            "v_px": 24,
                            "inside_image": True,
                        },
                    },
                    {
                        "landmark_id": "right_shoulder",
                        "image_projection": {
                            "available": True,
                            "u_px": 58 - step,
                            "v_px": 38,
                            "inside_image": True,
                        },
                    },
                    {
                        "landmark_id": "right_hand",
                        "image_projection": {
                            "available": True,
                            "u_px": 40 - step,
                            "v_px": 24,
                            "inside_image": True,
                        },
                    },
                ],
                "segments": [
                    {"from": "left_shoulder", "to": "left_hand"},
                    {"from": "right_shoulder", "to": "right_hand"},
                ],
                "claim_boundary": {
                    "uses_unitree_g1_mujoco_body_transforms": True,
                    "not_physical_robot_sensor_proof": True,
                },
            }
            for step in range(4)
        ],
    )


def test_oscar_wam_command_adapter_materializes_inputs_and_blocks_without_cuda(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "oscar-source"
    (source_root / "inference").mkdir(parents=True)
    (source_root / "inference" / "inference_oscar.py").write_text("# entrypoint\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    review_video = tmp_path / "review" / "episode_0001__head_pov.mp4"
    _write_review_video(review_video)
    trace_path = tmp_path / "g1_mujoco_locomotion_trace.jsonl"
    projected_trace_path = tmp_path / "g1_projected_skeleton_trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "episode_id": "episode_0001",
                "root_position": [0.0 + step * 0.05, 0.0, 0.79],
                "root_yaw_rad": 0.1 * step,
                "active_action": {"action_type": "base_velocity", "vx_mps": 0.2, "vy_mps": 0.0},
                "fall_detected": False,
            }
            for step in range(8)
        ],
    )
    _write_projected_skeleton_trace(projected_trace_path)
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(
        rollout_input,
        {
            "source_mujoco_endpoint_eval_job_dir": str(tmp_path),
            "selected_review_videos": [
                {
                    "path": str(review_video),
                    "camera": "head_pov",
                    "egocentric_sensor_view": True,
                    "first_person_policy_observation_candidate": True,
                    "hands_or_end_effectors_expected_in_view": True,
                }
            ],
            "task_prompts": [{"task_prompt": "Move toward the target."}],
            "inputs": {
                "g1_mujoco_locomotion_trace_jsonl": str(trace_path),
                "g1_projected_skeleton_trace_jsonl": str(projected_trace_path),
            },
        },
    )
    output_path = tmp_path / "wam_provider_output.json"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '{\"module_available\":{\"torch\":true,\"torchvision\":true,\"cv2\":true,\"decord\":true,\"einops\":true,\"diffusers\":true,\"transformers\":true,\"worldsim\":true},\"torch_cuda_available\":false,\"platform_system\":\"Linux\"}'\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output_path))

    payload = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            str(fake_python),
            "--work-dir",
            str(tmp_path / "work"),
            "--num-frames",
            "4",
            "--height",
            "64",
            "--width",
            "64",
            "--fps",
            "5",
            "--probe-only",
        ]
    )

    assert payload["status"] == "blocked"
    assert "blocked_oscar_requires_cuda_gpu_runtime" in payload["blockers"]
    written = json.loads(output_path.read_text(encoding="utf-8"))
    package = written["input_package"]
    assert Path(package["first_frame"]["path"]).is_file()
    assert Path(package["skeleton_video"]["path"]).is_file()
    assert package["conditioning_video_review_validation"]["status"] == "completed"
    assert package["conditioning_video_decode_valid_for_review"] is True
    assert package["conditioning_video_visually_useful_for_model_input"] is True
    assert package["skeleton_video"]["conditioning_mode"] == "projected_g1_skeleton"
    assert package["skeleton_video"]["proxy_skeleton_overlay_drawn"] is False
    assert package["skeleton_video"]["egocentric_arm_skeleton_rendered"] is False
    assert package["skeleton_video"]["oscar_gripper_scenario_proxy_rendered"] is False
    assert package["skeleton_video"]["projected_g1_skeleton_rendered"] is True
    assert package["skeleton_video"]["texture_free_egocentric_arm_skeleton_rendered"] is False
    assert package["skeleton_video"]["selected_review_video_background_used"] is False
    assert package["skeleton_video"]["background_frame_count"] == 0
    assert package["skeleton_video"]["skeleton_stream_separate_from_rgb"] is True
    assert package["skeleton_video"]["skeleton_stream_texture_free"] is True
    assert package["skeleton_video"]["skeleton_stream_image_aligned_to_rgb"] is True
    assert package["skeleton_video"]["first_rgb_frame_anchors_scene_and_robot_appearance"] is True
    assert package["skeleton_video"]["alignment_contract"]["width"] == 64
    assert package["skeleton_video"]["alignment_contract"]["height"] == 64
    assert package["projected_skeleton_trace"]["used_for_conditioning"] is True
    assert package["projected_skeleton_trace"]["row_count"] == 4
    assert package["projected_skeleton_trace"]["projectable_row_count"] == 4
    assert package["rgb_video"]["used_for_oscar_rgb_latent_context"] is False
    assert package["rgb_video"]["omitted_for_projected_g1_skeleton_conditioning"] is True
    assert package["rgb_video"]["normalized_for_oscar_inference"] is False
    assert package["oscar_dual_stream_input_contract"]["separate_2d_skeleton_stream"] is True
    assert package["oscar_dual_stream_input_contract"]["skeleton_stream_texture_free"] is True
    assert (
        package["oscar_dual_stream_input_contract"][
            "first_rgb_frame_anchors_scene_and_robot_appearance"
        ]
        is True
    )
    assert package["claim_boundary"]["skeleton_conditioning_is_proxy_from_mujoco_trace"] is True
    assert package["claim_boundary"]["projected_g1_skeleton_conditioning_used"] is True
    assert (
        package["claim_boundary"][
            "projected_g1_skeleton_conditioning_is_simulated_mujoco_state"
        ]
        is True
    )
    assert (
        package["claim_boundary"][
            "projected_g1_skeleton_conditioning_is_not_physical_robot_sensor_evidence"
        ]
        is True
    )
    assert (
        package["claim_boundary"][
            "egocentric_arm_skeleton_conditioning_is_texture_free_action_render"
        ]
        is False
    )
    assert (
        package["claim_boundary"]["oscar_gripper_scenario_proxy_conditioning_is_support_asset_only"]
        is False
    )
    assert (
        package["claim_boundary"]["conditioning_video_overlays_proxy_gripper_action_cues"]
        is False
    )
    assert (
        package["claim_boundary"]["conditioning_video_preserves_selected_egocentric_rgb_context"]
        is False
    )
    assert package["claim_boundary"]["first_person_conditioning_uses_selected_review_video"] is False
    assert package["claim_boundary"]["first_frame_uses_selected_review_video"] is True
    assert (
        package["claim_boundary"]["first_rgb_frame_anchors_scene_and_robot_appearance"]
        is True
    )
    assert (
        package["claim_boundary"]["separate_2d_skeleton_stream_aligned_to_rgb"]
        is True
    )
    assert package["claim_boundary"]["skeleton_stream_is_texture_free"] is True
    assert (
        package["claim_boundary"]["rgb_video_arg_omitted_for_projected_g1_skeleton_conditioning"]
        is True
    )
    assert (
        package["claim_boundary"]["conditioning_video_uses_selected_first_person_g1_mesh_view"]
        is False
    )
    assert package["claim_boundary"]["conditioning_visual_enhancement_applies_to_support_asset_only"] is True
    assert written["raw_credentials_written_to_artifacts"] is False
    assert written["secret_hashes_written_to_artifacts"] is False


def test_oscar_wam_command_adapter_projected_g1_overlay_uses_first_person_background(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_video = tmp_path / "review" / "episode_0001__head_pov.mp4"
    _write_review_video(review_video)
    trace_path = tmp_path / "g1_mujoco_locomotion_trace.jsonl"
    projected_trace_path = tmp_path / "g1_projected_skeleton_trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "episode_id": "episode_0001",
                "root_position": [0.0 + step * 0.05, 0.0, 0.79],
                "root_yaw_rad": 0.1 * step,
                "active_action": {"action_type": "base_velocity", "vx_mps": 0.2},
                "fall_detected": False,
            }
            for step in range(8)
        ],
    )
    _write_projected_skeleton_trace(projected_trace_path)
    monkeypatch.setenv(
        "BLUEPRINT_OSCAR_WAM_CONDITIONING_MODE",
        "projected_g1_skeleton_rgb_overlay",
    )

    package = adapter._materialize_oscar_input_package(
        rollout_manifest={
            "source_mujoco_endpoint_eval_job_dir": str(tmp_path),
            "selected_review_videos": [
                {
                    "path": str(review_video),
                    "camera": "head_pov",
                    "egocentric_sensor_view": True,
                }
            ],
            "task_prompts": [{"task_prompt": "Move toward the target."}],
            "inputs": {
                "g1_mujoco_locomotion_trace_jsonl": str(trace_path),
                "g1_projected_skeleton_trace_jsonl": str(projected_trace_path),
            },
        },
        work_dir=tmp_path / "overlay-work",
        width=64,
        height=64,
        fps=5,
        num_frames=4,
    )

    skeleton = package["skeleton_video"]
    assert skeleton["conditioning_mode"] == "projected_g1_skeleton_rgb_overlay"
    assert skeleton["projected_g1_skeleton_rendered"] is True
    assert skeleton["selected_review_video_background_used"] is True
    assert skeleton["skeleton_stream_separate_from_rgb"] is False
    assert skeleton["skeleton_stream_texture_free"] is False
    assert skeleton["background_frame_count"] == 4
    assert (
        skeleton["background_preprocessing"]["background_alpha_applied_to_projected_g1_skeleton"]
        is True
    )
    assert skeleton["visual_signal"]["mean_non_dark_pixel_fraction"] > 0.5
    assert package["rgb_video"]["used_for_oscar_rgb_latent_context"] is True
    assert package["rgb_video"]["omitted_for_projected_g1_skeleton_conditioning"] is False
    assert package["rgb_video"]["normalized_for_oscar_inference"] is True
    assert package["rgb_video"]["width"] == 64
    assert package["rgb_video"]["height"] == 64
    assert package["rgb_video"]["frame_count"] == 4
    assert Path(package["rgb_video"]["path"]).is_file()
    cap = cv2.VideoCapture(package["rgb_video"]["path"])
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 64
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    finally:
        cap.release()
    assert package["claim_boundary"]["first_person_conditioning_uses_selected_review_video"] is True
    assert (
        package["claim_boundary"]["conditioning_video_uses_selected_first_person_g1_mesh_view"]
        is True
    )
    assert (
        package["claim_boundary"]["conditioning_video_preserves_selected_egocentric_rgb_context"]
        is True
    )
    assert (
        package["claim_boundary"]["rgb_video_arg_omitted_for_projected_g1_skeleton_conditioning"]
        is False
    )


def test_oscar_wam_command_adapter_rgb_context_mode_never_omits_rgb_latent_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_video = tmp_path / "review" / "episode_0001__head_pov.mp4"
    _write_review_video(review_video)
    trace_path = tmp_path / "g1_mujoco_locomotion_trace.jsonl"
    projected_trace_path = tmp_path / "g1_projected_skeleton_trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "episode_id": "episode_0001",
                "root_position": [0.0 + step * 0.05, 0.0, 0.79],
                "root_yaw_rad": 0.1 * step,
                "active_action": {"action_type": "base_velocity", "vx_mps": 0.2},
                "fall_detected": False,
            }
            for step in range(8)
        ],
    )
    _write_projected_skeleton_trace(projected_trace_path)
    monkeypatch.setenv(
        "BLUEPRINT_OSCAR_WAM_CONDITIONING_MODE",
        "projected_g1_skeleton_rgb_overlay",
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_RGB_CONTEXT_MODE", "never")

    package = adapter._materialize_oscar_input_package(
        rollout_manifest={
            "source_mujoco_endpoint_eval_job_dir": str(tmp_path),
            "selected_review_videos": [
                {
                    "path": str(review_video),
                    "camera": "head_pov",
                    "egocentric_sensor_view": True,
                }
            ],
            "task_prompts": [{"task_prompt": "Move toward the target."}],
            "inputs": {
                "g1_mujoco_locomotion_trace_jsonl": str(trace_path),
                "g1_projected_skeleton_trace_jsonl": str(projected_trace_path),
            },
        },
        work_dir=tmp_path / "no-rgb-work",
        width=64,
        height=64,
        fps=5,
        num_frames=4,
    )

    assert package["rgb_video"]["rgb_context_mode"] == "never"
    assert package["rgb_video"]["used_for_oscar_rgb_latent_context"] is False
    assert package["rgb_video"]["omitted_by_rgb_context_mode"] is True
    assert package["rgb_video"]["normalized_for_oscar_inference"] is False
    assert package["claim_boundary"]["rgb_video_uses_selected_review_video"] is False
    assert (
        package["claim_boundary"]["rgb_video_arg_omitted_by_rgb_context_mode"]
        is True
    )


def test_oscar_wam_command_adapter_private_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert adapter._number(True, 1.5) == 1.5
    assert adapter._number("bad", 2.5) == 2.5
    assert adapter._read_jsonl(tmp_path / "missing.jsonl") == []
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text('\n{"row": 1}\n["ignored"]\n', encoding="utf-8")
    assert adapter._read_jsonl(rows_path) == [{"row": 1}]
    assert adapter._repo_src_root().name == "src"

    existing = tmp_path / "existing"
    existing.mkdir()
    assert adapter._first_existing_path(["", str(existing)]) == existing.resolve()
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_SOURCE_ROOT", str(existing))
    monkeypatch.setenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", str(existing))
    assert adapter._source_root_from_env() == existing.resolve()
    assert adapter._checkpoint_from_env() == existing.resolve()

    review_video = tmp_path / "episode_0001__third_person.mp4"
    _write_review_video(review_video)
    selection_manifest = tmp_path / "selection.json"
    _write_json(selection_manifest, {"selected_review_videos": [{"path": str(review_video)}]})
    assert (
        adapter._selected_video_path(
            {"inputs": {"review_video_selection_manifest": str(selection_manifest)}}
        )
        == review_video.resolve()
    )
    robot_pov_video = tmp_path / "episode_0004_contact_or_push_light_object__robot_pov.mp4"
    head_pov_video = tmp_path / "episode_0004_contact_or_push_light_object__head_pov.mp4"
    torso_pov_video = tmp_path / "episode_0004_contact_or_push_light_object__torso_pov.mp4"
    _write_review_video(robot_pov_video)
    _write_review_video(head_pov_video)
    _write_review_video(torso_pov_video)
    default_selected = adapter._selected_video_row(
        {
            "selected_review_videos": [
                {
                    "path": str(review_video),
                    "camera": "third_person",
                    "task_id": "contact_or_push_light_object",
                },
                {
                    "path": str(head_pov_video),
                    "camera": "head_pov",
                    "task_id": "contact_or_push_light_object",
                },
                {
                    "path": str(torso_pov_video),
                    "camera": "torso_pov",
                    "task_id": "contact_or_push_light_object",
                },
            ]
        }
    )
    assert default_selected["path"] == str(head_pov_video.resolve())
    assert default_selected["camera"] == "head_pov"
    manifest_contract_selected = adapter._selected_video_row(
        {
            "wam_input_videos": [
                {
                    "path": str(head_pov_video),
                    "camera": "head_pov",
                    "task_id": "contact_or_push_light_object",
                }
            ],
            "selected_review_videos": [
                {
                    "path": str(review_video),
                    "camera": "third_person",
                    "task_id": "contact_or_push_light_object",
                }
            ],
        }
    )
    assert manifest_contract_selected["path"] == str(head_pov_video.resolve())
    assert manifest_contract_selected["wam_input_video"] is True
    monkeypatch.setenv("BLUEPRINT_WAM_PREFERRED_CAMERA", "head_pov,robot_pov,third_person")
    monkeypatch.setenv("BLUEPRINT_WAM_PREFERRED_TASK_ID", "contact_or_push_light_object")
    selected = adapter._selected_video_row(
        {
            "selected_review_videos": [
                {
                    "path": str(review_video),
                    "camera": "third_person",
                    "scenario_eval_run_id": "run-approach",
                    "task_id": "approach_target",
                },
                {
                    "path": str(robot_pov_video),
                    "camera": "robot_pov",
                    "scenario_eval_run_id": "run-contact",
                    "task_id": "contact_or_push_light_object",
                },
                {
                    "path": str(head_pov_video),
                    "camera": "head_pov",
                    "scenario_eval_run_id": "run-contact",
                    "task_id": "contact_or_push_light_object",
                },
            ],
            "task_prompts": [
                {
                    "scenario_eval_run_id": "run-contact",
                    "task_prompt": "Push the lightweight object from robot POV.",
                    "spawn_id": "doorway",
                }
            ],
        }
    )
    assert selected["path"] == str(head_pov_video.resolve())
    assert selected["camera"] == "head_pov"
    assert selected["task_prompt"] == "Push the lightweight object from robot POV."
    with pytest.raises(FileNotFoundError, match="missing_selected_review_video"):
        adapter._selected_video_path({"selected_review_videos": [{"path": "missing.mp4"}]})
    assert adapter._task_prompt({}) == (
        "Predict the next robot-scene frames from Blueprint action conditioning."
    )

    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(trace_path, [{"episode_id": "other", "root_position": [1, 2, 3]}])
    assert adapter._trace_rows({"inputs": {"g1_mujoco_locomotion_trace_jsonl": str(trace_path)}}) == [
        {"episode_id": "other", "root_position": [1, 2, 3]}
    ]
    projected_trace = tmp_path / "projected_trace.jsonl"
    _write_projected_skeleton_trace(projected_trace)
    projected_rows = adapter._projected_skeleton_rows(
        {"inputs": {"g1_projected_skeleton_trace_jsonl": str(projected_trace)}}
    )
    assert len(projected_rows) == 4
    assert adapter._projected_skeleton_projectable_row_count(projected_rows) == 4
    assert adapter._configured_conditioning_mode(projected_rows) == "projected_g1_skeleton"
    assert adapter._configured_conditioning_mode([]) == "oscar_gripper_scenario_proxy"
    assert adapter._sample_rows([], 3) == []
    assert adapter._sample_rows([{"row": 1}], 3) == [{"row": 1}, {"row": 1}, {"row": 1}]
    assert adapter._point_from_root({"root_position": "bad"}) == (0.0, 0.0, 0.8)

    with pytest.raises(ValueError, match="missing_locomotion_trace"):
        adapter._render_proxy_skeleton_video(
            trace_rows=[],
            output_path=tmp_path / "empty.mp4",
            width=64,
            height=64,
            fps=5.0,
            num_frames=1,
        )

    class ClosedWriter:
        def isOpened(self) -> bool:
            return False

    monkeypatch.setattr(cv2, "VideoWriter", lambda *args, **kwargs: ClosedWriter())
    with pytest.raises(RuntimeError, match="cv2_video_writer_failed"):
        adapter._render_proxy_skeleton_video(
            trace_rows=[{"root_position": [0, 0, 0.8]}],
            output_path=tmp_path / "closed.mp4",
            width=64,
            height=64,
            fps=5.0,
            num_frames=1,
        )

    monkeypatch.undo()
    skeleton = adapter._render_proxy_skeleton_video(
        trace_rows=[
            {
                "root_position": [0, 0, 0.8],
                "root_yaw_rad": 0.1,
                "active_action": {"action_type": "inspect_look"},
                "fall_detected": True,
            },
            {
                "root_position": [0.1, 0.1, 0.82],
                "active_action": {"action_type": "unknown"},
            },
        ],
        output_path=tmp_path / "inspect.mp4",
        width=64,
        height=64,
        fps=5.0,
        num_frames=2,
    )
    assert skeleton["fall_frame_count"] == 1
    assert {"action_type": "inspect_look", "count": 1} in skeleton["action_type_counts"]
    assert skeleton["background_preprocessing"]["near_black_void_fill_enabled"] is True

    dark_review_video = tmp_path / "dark_void.mp4"
    _write_dark_void_review_video(dark_review_video)
    brightened = adapter._render_proxy_skeleton_video(
        trace_rows=[
            {
                "root_position": [0, 0, 0.8],
                "root_yaw_rad": 0.1,
                "active_action": {"action_type": "manipulation_contact"},
                "fall_detected": False,
            }
            for _ in range(4)
        ],
        output_path=tmp_path / "brightened.mp4",
        width=64,
        height=64,
        fps=5.0,
        num_frames=4,
        background_video=dark_review_video,
    )
    assert brightened["background_frame_count"] == 4
    assert brightened["background_preprocessing"]["void_fill_pixel_fraction"] > 0.1
    assert brightened["background_preprocessing"]["background_alpha"] == pytest.approx(0.88)
    assert brightened["visual_signal"]["status"] == "completed"

    bad_video = tmp_path / "bad.mp4"
    bad_video.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="could_not_decode_selected_review_video_first_frame"):
        adapter._extract_first_frame(
            review_video=bad_video,
            output_path=tmp_path / "first.png",
            width=64,
            height=64,
        )


def test_oscar_wam_runtime_probe_subprocess_and_rollout_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "oscar-source"
    (source_root / "inference").mkdir(parents=True)
    (source_root / "inference" / "inference_oscar.py").write_text("# oscar\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("weights", encoding="utf-8")

    monkeypatch.setenv("PYTHONPATH", "existing")
    cuda_lib = tmp_path / "cuda-lib"
    cuda_lib.mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(cuda_lib))
    env = adapter._runtime_env(source_root)
    assert str(source_root) in env["PYTHONPATH"]
    assert "existing" in env["PYTHONPATH"]
    assert str(cuda_lib) in env["LD_LIBRARY_PATH"]

    monkeypatch.setattr(adapter.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="{", stderr=""),
    )
    invalid_probe = adapter._run_import_probe(
        python=sys.executable,
        source_root=source_root,
        timeout_seconds=1,
    )
    assert invalid_probe["status"] == "blocked"
    assert "blocked_oscar_runtime_import_probe_failed" in invalid_probe["blockers"]

    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "module_available": {"torch": True, "worldsim": False},
                    "torch_cuda_available": False,
                    "platform_system": "Linux",
                }
            ),
            stderr="warn",
        ),
    )
    missing_probe = adapter._run_import_probe(
        python=sys.executable,
        source_root=source_root,
        timeout_seconds=1,
    )
    assert missing_probe["status"] == "blocked"
    assert missing_probe["missing_modules"] == ["worldsim"]
    assert missing_probe["stderr_omitted_to_avoid_secret_leakage"] is True

    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "module_available": {"torch": True},
                    "torch_cuda_available": True,
                    "platform_system": "Linux",
                }
            ),
            stderr="",
        ),
    )
    completed_probe = adapter._run_import_probe(
        python=sys.executable,
        source_root=source_root,
        timeout_seconds=1,
    )
    assert completed_probe["status"] == "completed"

    assert str(checkpoint) not in adapter._redacted_argv(["python", str(checkpoint)], checkpoint)

    captured_subprocess: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        captured_subprocess["args"] = args
        captured_subprocess["kwargs"] = kwargs
        return SimpleNamespace(returncode=1, stdout="out", stderr="err")

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    rgb_video = tmp_path / "rgb.mp4"
    _write_review_video(rgb_video)
    stale_output = tmp_path / "out.mp4"
    _write_review_video(stale_output)
    failed = adapter._run_oscar(
        python=sys.executable,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest={
            "first_frame": {"path": str(tmp_path / "first.png")},
            "skeleton_video": {"path": str(tmp_path / "skeleton.mp4")},
            "rgb_video": {"path": str(rgb_video)},
            "prompt": "Move.",
            "num_frames": 2,
            "height": 64,
            "width": 64,
            "fps": 5,
        },
        output_video=stale_output,
        timeout_seconds=1,
        num_steps=2,
        guidance=1.5,
        seed=3,
    )
    assert failed["status"] == "blocked"
    assert failed["blockers"] == ["oscar_inference_command_nonzero"]
    assert failed["stale_output_removed_before_launch"] is True
    assert not stale_output.exists()
    argv = captured_subprocess["args"][0]
    assert "--rgb-video" in argv
    assert str(rgb_video) in argv

    def fake_timeout(*args: Any, **kwargs: Any) -> SimpleNamespace:
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output="partial", stderr="slow")

    monkeypatch.setattr(adapter.subprocess, "run", fake_timeout)
    timed_out = adapter._run_oscar(
        python=sys.executable,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest={
            "first_frame": {"path": str(tmp_path / "first.png")},
            "skeleton_video": {"path": str(tmp_path / "skeleton.mp4")},
            "rgb_video": {"path": str(rgb_video)},
            "prompt": "Move.",
            "num_frames": 2,
            "height": 64,
            "width": 64,
            "fps": 5,
        },
        output_video=tmp_path / "timeout_out.mp4",
        timeout_seconds=1,
        num_steps=2,
        guidance=1.5,
        seed=3,
    )
    assert timed_out["status"] == "blocked"
    assert timed_out["timed_out"] is True
    assert timed_out["blockers"] == ["oscar_inference_command_timeout"]
    assert timed_out["stderr_omitted_to_avoid_secret_leakage"] is True

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    captured_subprocess.clear()
    projected_failed = adapter._run_oscar(
        python=sys.executable,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest={
            "first_frame": {"path": str(tmp_path / "first.png")},
            "skeleton_video": {
                "path": str(tmp_path / "skeleton.mp4"),
                "projected_g1_skeleton_rendered": True,
            },
            "projected_skeleton_trace": {
                "path": str(tmp_path / "g1_projected_skeleton_trace.jsonl"),
                "used_for_conditioning": True,
            },
            "rgb_video": {"path": str(rgb_video)},
            "prompt": "Move.",
            "num_frames": 2,
            "height": 64,
            "width": 64,
            "fps": 5,
            "claim_boundary": {"projected_g1_skeleton_conditioning_used": True},
        },
        output_video=tmp_path / "projected_out.mp4",
        timeout_seconds=1,
        num_steps=2,
        guidance=1.5,
        seed=3,
    )
    assert projected_failed["status"] == "blocked"
    projected_argv = captured_subprocess["args"][0]
    assert "--rgb-video" not in projected_argv
    assert str(rgb_video) not in projected_argv

    captured_subprocess.clear()
    projected_with_rgb_failed = adapter._run_oscar(
        python=sys.executable,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest={
            "first_frame": {"path": str(tmp_path / "first.png")},
            "skeleton_video": {
                "path": str(tmp_path / "skeleton.mp4"),
                "projected_g1_skeleton_rendered": True,
            },
            "projected_skeleton_trace": {
                "path": str(tmp_path / "g1_projected_skeleton_trace.jsonl"),
                "used_for_conditioning": True,
            },
            "rgb_video": {
                "path": str(rgb_video),
                "used_for_oscar_rgb_latent_context": True,
            },
            "prompt": "Move.",
            "num_frames": 2,
            "height": 64,
            "width": 64,
            "fps": 5,
            "claim_boundary": {"projected_g1_skeleton_conditioning_used": True},
        },
        output_video=tmp_path / "projected_with_rgb_out.mp4",
        timeout_seconds=1,
        num_steps=2,
        guidance=1.5,
        seed=3,
    )
    assert projected_with_rgb_failed["status"] == "blocked"
    projected_with_rgb_argv = captured_subprocess["args"][0]
    assert "--rgb-video" in projected_with_rgb_argv
    assert str(rgb_video) in projected_with_rgb_argv

    output_video = tmp_path / "rollout.mp4"
    completed_payload = adapter._rollout_payload(
        package_manifest={"source_review_video_path": "review.mp4"},
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail={"status": "completed"},
        output_video=output_video,
    )
    assert completed_payload["status"] == "blocked"
    assert completed_payload["blockers"] == [
        "blocked_no_generated_oscar_mp4",
        "generated_video_missing",
    ]
    _write_review_video(output_video)
    completed_payload = adapter._rollout_payload(
        package_manifest={"source_review_video_path": "review.mp4"},
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail={"status": "completed", "configured_inference_steps": 35},
        output_video=output_video,
    )
    assert completed_payload["status"] == "completed"
    assert completed_payload["rollouts"][0]["generated_video_path"] == str(output_video)
    assert completed_payload["rollouts"][0]["generated_video_sha256"] == hashlib.sha256(
        output_video.read_bytes()
    ).hexdigest()
    assert completed_payload["fresh_model_run_steps"] == 1
    assert completed_payload["configured_inference_steps_per_model_run"] == 35

    invalid_video = tmp_path / "placeholder.mp4"
    invalid_video.write_bytes(b"mp4-placeholder")
    invalid_payload = adapter._rollout_payload(
        package_manifest={"source_review_video_path": "review.mp4"},
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail={"status": "completed"},
        output_video=invalid_video,
    )
    assert invalid_payload["status"] == "blocked"
    assert "blocked_generated_oscar_mp4_not_reviewable" in invalid_payload["blockers"]


def test_oscar_wam_run_main_and_module_guard_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "missing_runtime_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output_path))
    missing = adapter.run(
        [
            "--source-root",
            str(tmp_path / "missing-source"),
            "--checkpoint",
            str(tmp_path / "missing-checkpoint"),
            "--python",
            str(tmp_path / "missing-python"),
        ]
    )
    assert missing["status"] == "blocked"
    assert {
        "blocked_missing_oscar_inference_entrypoint",
        "blocked_configured_oscar_checkpoint_path_missing",
        "blocked_configured_python_missing",
    }.issubset(set(missing["blockers"]))

    source_root = tmp_path / "oscar-source"
    (source_root / "inference").mkdir(parents=True)
    (source_root / "inference" / "inference_oscar.py").write_text("# oscar\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("weights", encoding="utf-8")

    def raise_materialize(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise RuntimeError("bad inputs")

    monkeypatch.setattr(adapter, "_materialize_oscar_input_package", raise_materialize)
    materialize_blocked = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            sys.executable,
            "--work-dir",
            str(tmp_path / "materialize-work"),
            "--allow-experimental-oscar-version",
        ]
    )
    assert materialize_blocked["blockers"] == [
        "blocked_oscar_input_package_materialization_failed:KeyError"
    ]

    def fake_materialize(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "schema_version": "blueprint_oscar_wam_input_package.v1",
            "first_frame": {"path": str(tmp_path / "first.png")},
            "skeleton_video": {"path": str(tmp_path / "skeleton.mp4")},
            "prompt": "Move.",
        }

    monkeypatch.setattr(adapter, "_materialize_oscar_input_package", fake_materialize)
    rollout_input = tmp_path / "rollout_input.json"
    _write_json(rollout_input, {})
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setattr(
        adapter,
        "_run_import_probe",
        lambda **kwargs: {"status": "blocked", "blockers": ["probe_blocked"]},
    )
    probe_blocked = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            sys.executable,
            "--work-dir",
            str(tmp_path / "probe-work"),
            "--probe-only",
        ]
    )
    assert probe_blocked["probe_only"] is True
    assert probe_blocked["blockers"] == ["probe_blocked"]

    monkeypatch.setattr(
        adapter,
        "_run_import_probe",
        lambda **kwargs: {"status": "completed", "blockers": []},
    )

    def fake_run_oscar(**kwargs: Any) -> dict[str, Any]:
        _write_review_video(Path(kwargs["output_video"]))
        return {"status": "completed", "blockers": []}

    monkeypatch.setattr(adapter, "_run_oscar", fake_run_oscar)
    completed = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            sys.executable,
            "--work-dir",
            str(tmp_path / "completed-work"),
            "--allow-experimental-oscar-version",
            "--num-frames",
            "2",
            "--height",
            "64",
            "--width",
            "64",
            "--fps",
            "5",
            "--num-steps",
            "3",
            "--guidance",
            "1.5",
            "--seed",
            "9",
            "--timeout-seconds",
            "10",
        ]
    )
    assert completed["status"] == "completed"
    assert completed["generated_video_count"] == 1

    stale_work = tmp_path / "stale-work"
    stale_video = stale_work / "oscar_generated_rollout.mp4"
    _write_review_video(stale_video)
    stale_payload = adapter._rollout_payload(
        package_manifest={
            "source_review_video_path": str(tmp_path / "review.mp4"),
            "source_camera": "head_pov",
            "task_id": "approach_target",
        },
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail={"status": "blocked", "blockers": ["oscar_failed"]},
        output_video=stale_video,
    )
    assert stale_payload["status"] == "blocked"
    assert stale_payload["fresh_model_run_claimed"] is False
    assert stale_payload["learned_wam_model_ran"] is False
    assert "blocked_oscar_inference_command_not_completed" in stale_payload["blockers"]

    monkeypatch.setattr(
        adapter,
        "_run_oscar",
        lambda **kwargs: {"status": "blocked", "blockers": ["oscar_failed"]},
    )
    blocked = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            sys.executable,
            "--work-dir",
            str(tmp_path / "blocked-work"),
            "--allow-experimental-oscar-version",
        ]
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == ["oscar_failed"]

    monkeypatch.setattr(adapter, "run", lambda argv: {"status": "completed"})
    assert adapter.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"

    def raise_run(argv: Any) -> dict[str, Any]:
        del argv
        raise RuntimeError("boom")

    exception_output = tmp_path / "exception.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(exception_output))
    monkeypatch.setattr(adapter, "run", raise_run)
    assert adapter.main([]) == 2
    assert "oscar_wam_adapter_exception:RuntimeError" in json.loads(
        exception_output.read_text(encoding="utf-8")
    )["blockers"]

    module_output = tmp_path / "module.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(module_output))
    monkeypatch.setattr(sys, "argv", ["oscar_wam_command_adapter.py"])
    with pytest.raises(SystemExit) as exc:
        run_module_as_main("blueprint_pipeline.oscar_wam_command_adapter")
    assert exc.value.code == 2
    assert json.loads(module_output.read_text(encoding="utf-8"))["status"] == "blocked"


def test_oscar_wam_run_blocks_unpinned_local_official_release_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "oscar-source"
    (source_root / "inference").mkdir(parents=True)
    (source_root / "inference" / "inference_oscar.py").write_text("# oscar\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_path = tmp_path / "blocked_unpinned.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output_path))

    payload = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            sys.executable,
        ]
    )

    assert payload["status"] == "blocked"
    assert "official_oscar_source_url_mismatch" in payload["blockers"]
    assert "official_oscar_source_commit_not_pinned" in payload["blockers"]
    assert "official_oscar_hf_revision_not_pinned" in payload["blockers"]
    assert payload["official_oscar_release"]["official_release_match"] is False
    assert (
        payload["truth_boundary"]["official_oscar_source_and_checkpoint_pinned"]
        is False
    )
