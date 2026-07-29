from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from blueprint_pipeline import oscar_wam_provider_bundle as bundle_module  # noqa: E402
from blueprint_pipeline.oscar_wam_provider_bundle import build_oscar_wam_provider_bundle  # noqa: E402
from blueprint_pipeline.oscar_official_release import (  # noqa: E402
    OFFICIAL_OSCAR_HF_REVISION,
    OFFICIAL_OSCAR_SOURCE_COMMIT,
    official_release_contract,
)


def test_embedded_oscar_provider_runtime_runner_compiles() -> None:
    compile(bundle_module.REMOTE_RUNNER, "<oscar_wam_provider_runtime_runner>", "exec")


def test_transformer_engine_shim_rope_matches_te_split_half_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY", raising=False)
    source_root = tmp_path / "oscar-source"
    source_root.mkdir()
    namespace: dict[str, Any] = {}
    exec(bundle_module.REMOTE_RUNNER, namespace)

    detail = namespace["_apply_oscar_source_compatibility"](source_root)

    assert detail["status"] == "completed"

    def _purge_transformer_engine_modules() -> None:
        for name in list(sys.modules):
            if name == "transformer_engine" or name.startswith("transformer_engine."):
                sys.modules.pop(name, None)

    def _rotate_half(tensor: Any, interleaved: bool) -> Any:
        if not interleaved:
            first, second = torch.chunk(tensor, 2, dim=-1)
            return torch.cat((-second, first), dim=-1)
        first = tensor[..., ::2]
        second = tensor[..., 1::2]
        return torch.stack((-second, first), dim=-1).flatten(-2)

    def _reference_rope(tensor: Any, freqs: Any, *, interleaved: bool = False) -> Any:
        freqs = freqs.transpose(0, 1)
        cos = torch.cos(freqs).to(tensor.dtype)
        sin = torch.sin(freqs).to(tensor.dtype)
        rot_dim = freqs.shape[-1]
        tensor_rot = tensor[..., :rot_dim]
        tensor_pass = tensor[..., rot_dim:]
        tensor_rot = tensor_rot * cos + _rotate_half(tensor_rot, interleaved) * sin
        return torch.cat((tensor_rot, tensor_pass), dim=-1)

    sys.path.insert(0, str(source_root))
    try:
        _purge_transformer_engine_modules()
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb

        tensor = torch.arange(36, dtype=torch.float32).reshape(1, 3, 2, 6) / 10.0
        freqs = torch.linspace(0.01, 0.24, steps=12, dtype=torch.float32).reshape(3, 1, 1, 4)

        assert torch.allclose(
            apply_rotary_pos_emb(tensor, freqs, tensor_format="bshd", fused=True),
            _reference_rope(tensor, freqs),
        )
        assert torch.allclose(
            apply_rotary_pos_emb(
                tensor,
                freqs,
                tensor_format="bshd",
                interleaved=True,
                fused=True,
            ),
            _reference_rope(tensor, freqs, interleaved=True),
        )
    finally:
        _purge_transformer_engine_modules()
        sys.path.remove(str(source_root))


def test_task_label_prompt_is_normalized_for_robot_action_context() -> None:
    step_input = {
        "current_policy_observation": {
            "task_prompt": "open the refrigerator",
            "task_id": "open_refrigerator",
            "target_object_id": "refrigerator",
        }
    }

    assert (
        bundle_module._source_task_prompt_from_wam_generation_step(step_input)
        == "open the refrigerator"
    )
    assert bundle_module._task_prompt_from_wam_generation_step(step_input) == (
        "A robot performs the task: open the refrigerator. "
        "Continue the egocentric first-person manipulation video from the robot's "
        "rigidly head-mounted camera. Keep that same camera viewpoint; never switch "
        "to an external, overhead, or third-person shot. Do not show the robot's "
        "head or torso; only its hands or forearms may enter from the bottom of the frame. "
        "The supplied robot skeleton trajectory is authoritative: show exactly that "
        "arm and hand motion. Keep articulated objects stationary unless a visible "
        "robot hand reaches and remains in contact with the object while it moves; "
        "never make a door, drawer, handle, or appliance move by itself."
    )


def test_robot_context_prompt_still_receives_egocentric_viewpoint_contract() -> None:
    prompt = "A robot reaches toward the refrigerator handle."
    step_input = {"current_policy_observation": {"task_prompt": prompt}}

    normalized = bundle_module._task_prompt_from_wam_generation_step(step_input)

    assert normalized.startswith(prompt)
    assert "rigidly head-mounted camera" in normalized
    assert "never switch to an external, overhead, or third-person shot" in normalized
    assert "skeleton trajectory is authoritative" in normalized
    assert "never make a door, drawer, handle, or appliance move by itself" in normalized


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_review_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :] = (70, 110, 150)
    cv2.rectangle(frame, (18, 12), (46, 54), (220, 190, 80), -1)
    assert cv2.imwrite(str(path), frame)


def _write_useful_conditioning_mp4(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64))
    assert writer.isOpened()
    for index in range(8):
        frame = np.full((64, 64, 3), (62, 72, 82), dtype=np.uint8)
        cv2.rectangle(frame, (8 + index, 14), (50, 52), (210, 170, 65), -1)
        cv2.circle(frame, (32, 18 + index // 2), 7, (235, 235, 235), -1)
        writer.write(frame)
    writer.release()


def _write_dark_conditioning_mp4(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(np.full((64, 64, 3), 4, dtype=np.uint8))
    writer.release()


SCENE_FAITHFUL_ISAAC_POLICY_ACTION_TRACE_CLAIM = {
    "policy_derived_action_conditioning": True,
    "official_wbc_or_sim_bridge_used": False,
    "scene_faithful_isaac_policy_action_projection_bridge_used": True,
    "blueprint_simulator_only_isaac_action_projection_bridge_used": True,
    "uses_isaac_sidecar_link_landmarks_not_hand_drawn_screen_axes": True,
    "sidecar_kinematic_chain_fk_solver_used": True,
    "full_g1_urdf_fk_solver_used": False,
    "sonic_action_delta_is_heuristic_reach_lift_not_official_wbc": False,
    "sonic_action_delta_is_heuristic_joint_delta_not_official_wbc": True,
}


def _write_projected_skeleton_trace(
    path: Path,
    *,
    claim_boundary: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "schema_version": "blueprint.mujoco_g1.projected_upper_body_skeleton.v1",
            "status": "completed",
            "episode_id": "episode_0001",
            "projected_landmark_count": 4,
            "landmarks": [
                {
                    "landmark_id": "left_wrist",
                    "image_projection": {
                        "available": True,
                        "u_px": 18,
                        "v_px": 42,
                    },
                },
                {
                    "landmark_id": "left_hand",
                    "image_projection": {
                        "available": True,
                        "u_px": 24,
                        "v_px": 26,
                    },
                },
                {
                    "landmark_id": "right_wrist",
                    "image_projection": {
                        "available": True,
                        "u_px": 46,
                        "v_px": 42,
                    },
                },
                {
                    "landmark_id": "right_hand",
                    "image_projection": {
                        "available": True,
                        "u_px": 40,
                        "v_px": 26,
                    },
                },
            ],
            "segments": [
                {"from": "left_wrist", "to": "left_hand"},
                {"from": "right_wrist", "to": "right_hand"},
            ],
        },
        {
            "schema_version": "blueprint.mujoco_g1.projected_upper_body_skeleton.v1",
            "status": "completed",
            "episode_id": "episode_0001",
            "projected_landmark_count": 4,
            "landmarks": [
                {
                    "landmark_id": "left_wrist",
                    "image_projection": {
                        "available": True,
                        "u_px": 20,
                        "v_px": 40,
                    },
                },
                {
                    "landmark_id": "left_hand",
                    "image_projection": {
                        "available": True,
                        "u_px": 27,
                        "v_px": 23,
                    },
                },
                {
                    "landmark_id": "right_wrist",
                    "image_projection": {
                        "available": True,
                        "u_px": 44,
                        "v_px": 40,
                    },
                },
                {
                    "landmark_id": "right_hand",
                    "image_projection": {
                        "available": True,
                        "u_px": 37,
                        "v_px": 23,
                    },
                },
            ],
            "segments": [
                {"from": "left_wrist", "to": "left_hand"},
                {"from": "right_wrist", "to": "right_hand"},
            ],
        },
    ]
    if claim_boundary is not None:
        for row in rows:
            row["claim_boundary"] = dict(claim_boundary)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_oscar_wam_provider_bundle_from_existing_inputs(tmp_path: Path) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    projected_trace = tmp_path / "g1_projected_skeleton_trace.jsonl"
    _write_projected_skeleton_trace(projected_trace)
    rollout_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_rollout_input_manifest.v1",
                "source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job"),
                "inputs": {
                    "g1_projected_skeleton_trace_jsonl": str(projected_trace),
                    "g1_projected_skeleton_manifest": str(
                        tmp_path / "g1_projected_skeleton_manifest.json"
                    ),
                },
                "task_prompts": [{"task_prompt": "Reach toward the object."}],
            }
        ),
        encoding="utf-8",
    )
    oscar_input = tmp_path / "oscar_input"
    oscar_input.mkdir()
    _write_review_png(oscar_input / "first_frame.png")
    _write_useful_conditioning_mp4(oscar_input / "blueprint_proxy_skeleton_conditioning.mp4")
    review_video = tmp_path / "head_pov_review.mp4"
    _write_useful_conditioning_mp4(review_video)
    (tmp_path / "oscar_wam_input_package_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "blueprint_oscar_wam_input_package.v1",
                "source_review_video_path": str(review_video),
                "rgb_video": {
                    "path": str(review_video),
                    "source": "selected_review_video_rgb_context",
                    "rgb_context_mode": "never",
                },
                "source_review_video": {
                    "path": str(review_video),
                    "camera": "head_pov",
                    "video_path": str(review_video),
                },
                "projected_skeleton_trace": {
                    "path": str(projected_trace),
                    "used_for_conditioning": True,
                    "row_count": 2,
                    "projectable_row_count": 2,
                },
                "skeleton_video": {
                    "conditioning_mode": "projected_g1_skeleton",
                    "projected_g1_skeleton_rendered": True,
                    "skeleton_stream_separate_from_rgb": True,
                    "skeleton_stream_texture_free": True,
                    "skeleton_stream_image_aligned_to_rgb": True,
                    "first_rgb_frame_anchors_scene_and_robot_appearance": True,
                    "projected_g1_skeleton_landmark_draw_count": 4,
                    "projected_g1_skeleton_visible_landmark_draw_count": 4,
                    "projected_g1_skeleton_visible_segment_count": 2,
                    "projected_g1_skeleton_end_effector_axis_draw_count": 2,
                    "visual_signal": {"status": "completed", "blockers": []},
                },
                "oscar_dual_stream_input_contract": {
                    "first_rgb_frame_path": str(oscar_input / "first_frame.png"),
                    "skeleton_video_path": str(
                        oscar_input / "blueprint_proxy_skeleton_conditioning.mp4"
                    ),
                    "separate_2d_skeleton_stream": True,
                    "skeleton_stream_texture_free": True,
                    "skeleton_stream_image_aligned_to_rgb": True,
                    "first_rgb_frame_anchors_scene_and_robot_appearance": True,
                    "full_rgb_video_required_for_oscar_inference": False,
                },
                "claim_boundary": {
                    "projected_g1_skeleton_conditioning_used": True,
                    "projected_g1_skeleton_conditioning_is_simulated_mujoco_state": True,
                    "projected_g1_skeleton_conditioning_is_not_physical_robot_sensor_evidence": True,
                    "first_rgb_frame_anchors_scene_and_robot_appearance": True,
                    "separate_2d_skeleton_stream_aligned_to_rgb": True,
                    "skeleton_stream_is_texture_free": True,
                },
                "source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job"),
            }
        ),
        encoding="utf-8",
    )

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "bundle-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_input_dir=oscar_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    assert manifest["input_package_conditioning_video_blockers"] == []
    bundle_path = Path(str(manifest["bundle_path"]))
    assert bundle_path.is_file()
    persisted = _read_json(tmp_path / "bundle-job" / "oscar_wam_provider_bundle_manifest.json")
    assert persisted["raw_credentials_written_to_artifacts"] is False
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/run_wam_provider_runtime.sh" in names
    assert "provider_runtime/wam_provider_runtime_runner.py" in names
    assert "provider_runtime/wam_provider_runtime_manifest.json" in names
    assert "provider_runtime/wam_rollout_input_manifest.json" in names
    assert "provider_runtime/oscar_input/first_frame.png" in names
    assert "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4" in names
    assert "provider_runtime/oscar_input/rgb_context.mp4" not in names
    assert "provider_runtime/oscar_input/g1_projected_skeleton_trace.jsonl" in names
    with zipfile.ZipFile(bundle_path) as archive:
        runner_text = archive.read("provider_runtime/wam_provider_runtime_runner.py").decode(
            "utf-8"
        )
        runtime_manifest_text = archive.read(
            "provider_runtime/wam_provider_runtime_manifest.json"
        ).decode("utf-8")
        rollout_manifest_text = archive.read(
            "provider_runtime/wam_rollout_input_manifest.json"
        ).decode("utf-8")
        runtime_manifest = json.loads(runtime_manifest_text)
        runtime_rollout_manifest = json.loads(rollout_manifest_text)
    runtime_input_package = runtime_manifest["input_package"]
    assert runtime_manifest["official_case_smoke"] == ""
    assert runtime_manifest["oscar_source_ref"] == OFFICIAL_OSCAR_SOURCE_COMMIT
    assert runtime_manifest["oscar_hf_revision"] == OFFICIAL_OSCAR_HF_REVISION
    assert runtime_manifest["official_oscar_release"]["official_release_match"] is True
    assert runtime_manifest["truth_boundary"]["official_oscar_source_and_checkpoint_pinned"] is True
    assert runtime_input_package["first_frame"]["path"] == (
        "provider_runtime/oscar_input/first_frame.png"
    )
    assert runtime_input_package["skeleton_video"]["path"] == (
        "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4"
    )
    assert "path" not in runtime_input_package["rgb_video"]
    assert (
        runtime_input_package["rgb_video"]["local_rgb_context_path_omitted_from_runtime_manifest"]
        is True
    )
    assert runtime_input_package["projected_skeleton_trace"]["path"] == (
        "provider_runtime/oscar_input/g1_projected_skeleton_trace.jsonl"
    )
    assert (
        runtime_input_package["oscar_projected_skeleton_runtime_contract"][
            "projected_skeleton_trace_packaged"
        ]
        is True
    )
    assert (
        runtime_input_package["oscar_projected_skeleton_runtime_contract"][
            "separate_2d_skeleton_stream"
        ]
        is True
    )
    assert (
        runtime_input_package["oscar_projected_skeleton_runtime_contract"][
            "skeleton_stream_texture_free"
        ]
        is True
    )
    assert (
        runtime_input_package["oscar_projected_skeleton_runtime_contract"][
            "first_rgb_frame_anchors_scene_and_robot_appearance"
        ]
        is True
    )
    assert (
        runtime_input_package["oscar_dual_stream_input_contract"]["first_rgb_frame_path"]
        == "provider_runtime/oscar_input/first_frame.png"
    )
    assert (
        runtime_input_package["oscar_dual_stream_input_contract"]["skeleton_video_path"]
        == "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4"
    )
    assert (
        runtime_input_package["claim_boundary"][
            "skeleton_conditioning_is_proxy_from_mujoco_trace"
        ]
        is True
    )
    assert runtime_input_package["claim_boundary"]["existing_input_package_used"] is True
    assert (
        runtime_input_package["oscar_dual_stream_input_contract"]["separate_2d_skeleton_stream"]
        is True
    )
    assert (
        runtime_input_package["oscar_dual_stream_input_contract"]["skeleton_stream_texture_free"]
        is True
    )
    assert (
        runtime_manifest["oscar_runtime_argv_contract"]["projected_skeleton_trace_packaged"] is True
    )
    assert runtime_rollout_manifest["inputs"]["g1_projected_skeleton_trace_jsonl"] == (
        "provider_runtime/oscar_input/g1_projected_skeleton_trace.jsonl"
    )
    assert (
        runtime_rollout_manifest["inputs"][
            "local_g1_projected_skeleton_manifest_omitted_from_runtime_manifest"
        ]
        is True
    )
    assert "source_review_video_path" not in runtime_input_package
    assert (
        runtime_input_package["local_source_review_video_path_omitted_from_runtime_manifest"]
        is True
    )
    assert runtime_input_package["source_review_video"]["camera"] == "head_pov"
    assert (
        runtime_input_package["source_review_video"][
            "local_review_video_path_omitted_from_runtime_manifest"
        ]
        is True
    )
    assert (
        runtime_input_package[
            "local_source_mujoco_endpoint_eval_job_dir_omitted_from_runtime_manifest"
        ]
        is True
    )
    assert (
        runtime_rollout_manifest[
            "local_source_mujoco_endpoint_eval_job_dir_omitted_from_runtime_manifest"
        ]
        is True
    )
    assert runtime_input_package["rgb_video"]["used_for_oscar_rgb_latent_context"] is False
    assert (
        runtime_input_package["rgb_video"]["local_rgb_context_path_omitted_from_runtime_manifest"]
        is True
    )
    assert (
        runtime_input_package["oscar_rgb_context_runtime_contract"]["rgb_context_packaged"] is False
    )
    assert (
        runtime_input_package["oscar_rgb_context_runtime_contract"]["rgb_context_mode"] == "never"
    )
    assert (
        runtime_input_package["oscar_rgb_context_runtime_contract"]["expected_inference_arg"]
        is None
    )
    assert (
        runtime_input_package["oscar_rgb_context_runtime_contract"][
            "rgb_video_arg_omitted_by_rgb_context_mode"
        ]
        is True
    )
    assert (
        runtime_input_package["oscar_rgb_context_runtime_contract"][
            "projected_g1_skeleton_conditioning_suppresses_rgb_context"
        ]
        is True
    )
    assert runtime_manifest["oscar_runtime_argv_contract"]["rgb_video_arg_expected"] is False
    assert runtime_manifest["oscar_runtime_argv_contract"]["rgb_context_mode"] == "never"
    assert runtime_manifest["oscar_runtime_argv_contract"]["rgb_video_arg"] is None
    assert (
        runtime_manifest["oscar_runtime_argv_contract"]["rgb_video_arg_omitted_by_rgb_context_mode"]
        is True
    )
    assert (
        runtime_manifest["oscar_runtime_argv_contract"][
            "projected_g1_skeleton_conditioning_suppresses_rgb_context"
        ]
        is True
    )
    assert runtime_input_package["claim_boundary"]["rgb_video_uses_selected_review_video"] is False
    assert (
        runtime_input_package["claim_boundary"][
            "first_rgb_frame_anchors_scene_and_robot_appearance"
        ]
        is True
    )
    assert (
        runtime_input_package["claim_boundary"]["separate_2d_skeleton_stream_aligned_to_rgb"]
        is True
    )
    assert runtime_input_package["claim_boundary"]["skeleton_stream_is_texture_free"] is True
    assert (
        runtime_input_package["claim_boundary"][
            "rgb_context_packaging_is_input_contract_not_rollout_quality_proof"
        ]
        is True
    )
    assert (
        runtime_input_package["claim_boundary"][
            "projected_g1_skeleton_conditioning_suppresses_rgb_context"
        ]
        is True
    )
    assert (
        runtime_manifest["truth_boundary"]["rgb_context_packaging_is_not_visual_usefulness_proof"]
        is True
    )
    assert str(oscar_input) not in runtime_manifest_text
    assert str(review_video) not in runtime_manifest_text
    assert str(tmp_path) not in runtime_manifest_text
    assert str(tmp_path) not in rollout_manifest_text
    assert (
        runtime_input_package["claim_boundary"][
            "runtime_manifest_paths_point_to_provider_runtime_inputs"
        ]
        is True
    )
    assert 'checkpoint_path / "model"' in runner_text
    assert "inference_checkpoint_path" in runner_text
    assert "BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY" in runner_text
    assert "BLUEPRINT_COMPAT_SHIM = True" in runner_text
    assert "DotProductAttention" in runner_text
    assert "scaled_dot_product_attention" in runner_text
    assert "class QuantizedTensor" in runner_text
    assert "class Float8Tensor" in runner_text
    assert "class FP8GlobalStateManager" in runner_text
    assert "class DelayedScaling" in runner_text
    assert "class Linear(torch.nn.Linear)" in runner_text
    assert "class LayerNormLinear" in runner_text
    assert "from . import distributed, ops" in runner_text
    assert "source_compatibility_detail" in runner_text
    assert "compat_shim_paths_removed" in runner_text
    assert "transformer_engine_blueprint_compat_shim" in runner_text
    assert "transformer_engine_optional" in runner_text
    assert "transformer_engine_strategy" in runner_text
    assert "transformer_engine_tensor_api_importable" in runner_text
    assert "image_runtime_transformer_engine_tensor_api_unavailable" in runner_text
    assert "image_runtime_transformer_engine_fp8_api_unavailable" in runner_text
    assert "image_runtime_transformer_engine_recipe_api_unavailable" in runner_text
    assert "image_runtime_transformer_engine_module_api_unavailable" in runner_text
    assert "image_runtime_transformer_engine_ops_api_unavailable" in runner_text
    assert "transformer_engine-2.0.0.dist-info" in runner_text
    assert "Name: transformer-engine" in runner_text
    assert "BLUEPRINT_WAM_PROVIDER_ALLOW_BREAK_SYSTEM_PACKAGES" in runner_text
    assert "--break-system-packages" in runner_text
    assert "nvidia-resiliency-ext>=0.6.0" in runner_text
    assert '"pytest"' in runner_text
    assert '"hf_transfer"' in runner_text
    # required vs best-effort optional dependency split (a flaky optional package must not block
    # the model run) plus a transient retry on the required install
    assert "required_packages" in runner_text
    assert "optional_packages" in runner_text
    assert "optional_best_effort" in runner_text
    assert 'not row.get("optional_best_effort")' in runner_text
    assert "BLUEPRINT_OSCAR_WAM_SKIP_RUNTIME_PIP_INSTALL" in runner_text
    assert "BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS" in runner_text
    assert "BLUEPRINT_OSCAR_WAM_SOURCE_REF" in runner_text
    assert "BLUEPRINT_OSCAR_WAM_HF_REVISION" in runner_text
    assert OFFICIAL_OSCAR_SOURCE_COMMIT in runner_text
    assert OFFICIAL_OSCAR_HF_REVISION in runner_text
    assert "BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER" in runner_text
    assert 'env["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if hf_transfer_enabled else "0"' in runner_text
    assert 'env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")' not in runner_text
    assert "checkpoint_download_waiting" in runner_text
    assert "checkpoint_download_timeout_reached" in runner_text
    assert "start_new_session=True" in runner_text
    assert "os.killpg(process.pid, signal.SIGTERM)" in runner_text
    assert "os.killpg(process.pid, signal.SIGKILL)" in runner_text
    assert "oscar_checkpoint_download_timeout" in runner_text
    assert "retry_command_redacted" in runner_text
    assert "BLUEPRINT_OSCAR_WAM_OMIT_FPS_ARG" in runner_text
    assert "runtime_pip_install_skipped_by_reusable_image" in runner_text
    assert "pynvml_importable" in runner_text
    assert "image_runtime_pynvml_unavailable" in runner_text
    assert "loguru_importable" in runner_text
    assert "image_runtime_loguru_unavailable" in runner_text
    assert "image_runtime_cv2_unavailable" in runner_text
    assert "worldsim_runtime_imports" in runner_text
    assert "'pytest':'pytest'" in runner_text
    assert "image_runtime_worldsim_extra_unavailable" in runner_text
    assert "GUI_OPENCV_REQUIREMENT_NAMES" in runner_text
    assert "_pip_uninstall_argv" in runner_text
    assert "opencv_headless_repair_always_runs" in runner_text
    assert "filtered_gui_opencv_requirements" in runner_text
    assert "opencv_headless_import_failed_after_dependencies" in runner_text
    assert '"opencv-python-headless"' in runner_text
    assert '"--no-deps"' in runner_text
    assert '"opencv-python", "opencv-contrib-python"' in runner_text
    assert "hf_transfer_disabled_retry" in runner_text
    assert 'retry_env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"' in runner_text
    assert "oscar_loader_appends_model_subdirectory" in runner_text
    assert "_prepare_cuda_library_env" in runner_text
    assert "libcudart.so" in runner_text
    assert "cuda_lib_shims" in runner_text
    assert '"cuda_library_env"' in runner_text
    assert "official_case_smoke" in runner_text
    assert runner_text.index("official_case_smoke = str(runtime_manifest") < runner_text.index(
        "checkpoint_resolution_started"
    )
    assert "official_oscar_case_assets_missing" in runner_text
    assert "gripper_scenario.mp4" in runner_text
    assert "caption.pickle" not in runner_text
    assert "pickle.load" not in runner_text
    assert "caption.json" in runner_text
    assert "caption.txt" in runner_text
    assert "official_case_caption_schema_invalid" in runner_text
    assert "official_case_asset_digest_missing" in runner_text
    assert "official_case_asset_digest_mismatch" in runner_text
    assert "rgb_context.mp4" in runner_text
    assert "official_case_rgb_video" in runner_text
    assert "official_case_use_script" in runner_text
    assert "scripts/run_inference.sh" in runner_text
    assert "_prepare_official_script_runtime" in runner_text
    assert "(official_case_smoke or runtime_rgb_expected)" not in runner_text
    assert "(official_case_rgb_video or runtime_rgb_expected)" in runner_text
    assert "--rgb-video" in runner_text
    assert 'inference_checkpoint_path = (\n            checkpoint_path / "model"' not in runner_text


def test_existing_droid_oscar_input_preserves_recorded_joint_provenance(
    tmp_path: Path,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    rollout_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_rollout_input_manifest.v1",
                "task_prompts": [{"task_prompt": "Pick up the bottle."}],
            }
        ),
        encoding="utf-8",
    )
    oscar_input = tmp_path / "oscar_input"
    oscar_input.mkdir()
    _write_review_png(oscar_input / "first_frame.png")
    _write_useful_conditioning_mp4(oscar_input / "blueprint_proxy_skeleton_conditioning.mp4")
    package_manifest = tmp_path / "oscar_wam_input_package_manifest.json"
    package_manifest.write_text(
        json.dumps(
            {
                "schema_version": "blueprint_oscar_wam_input_package.v1",
                "status": "completed",
                "prompt": "robot picks up the bottle",
                "claim_boundary": {
                    "skeleton_conditioning_from_recorded_droid_joint_state": True,
                    "true_robot_proprioceptive_skeleton_available": True,
                },
                "skeleton_video": {
                    "visual_signal": {"status": "completed", "blockers": []}
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "bundle-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_input_dir=oscar_input,
        oscar_input_package_manifest=package_manifest,
        generated_at="2026-07-29T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    with zipfile.ZipFile(Path(str(manifest["bundle_path"]))) as archive:
        runtime_manifest = json.loads(
            archive.read("provider_runtime/wam_provider_runtime_manifest.json").decode("utf-8")
        )
    boundary = runtime_manifest["input_package"]["claim_boundary"]
    assert boundary["skeleton_conditioning_from_recorded_droid_joint_state"] is True
    assert boundary["true_robot_proprioceptive_skeleton_available"] is True
    assert "skeleton_conditioning_is_proxy_from_mujoco_trace" not in boundary
    assert boundary["existing_input_package_used"] is True


def test_build_oscar_wam_provider_bundle_blocks_unpinned_official_source(
    tmp_path: Path,
) -> None:
    rollout_input = tmp_path / "missing_rollout_input_manifest.json"

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "bundle-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_source_ref="main",
        generated_at="2026-06-30T00:00:00+00:00",
    )

    assert manifest["status"] == "blocked"
    assert "official_oscar_source_commit_not_pinned" in manifest["blockers"]
    assert manifest["official_oscar_release"]["source_ref"] == "main"
    assert manifest["truth_boundary"]["official_oscar_source_and_checkpoint_pinned"] is False


def test_official_oscar_release_contract_accepts_github_ssh_origin() -> None:
    contract = official_release_contract(source_url="git@github.com:wuzy2115/oscar-public.git")

    assert contract["source_url_official"] is True
    assert contract["official_release_match"] is True


def test_build_oscar_wam_provider_bundle_records_official_case_smoke_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_SMOKE", "agibot_465")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_USE_SCRIPT", "true")
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    rollout_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_rollout_input_manifest.v1",
                "source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job"),
                "task_prompts": [{"task_prompt": "Reach toward the object."}],
            }
        ),
        encoding="utf-8",
    )
    oscar_input = tmp_path / "oscar_input"
    oscar_input.mkdir()
    _write_review_png(oscar_input / "first_frame.png")
    _write_useful_conditioning_mp4(oscar_input / "blueprint_proxy_skeleton_conditioning.mp4")

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "bundle-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_input_dir=oscar_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    with zipfile.ZipFile(Path(str(manifest["bundle_path"]))) as archive:
        runtime_manifest = json.loads(
            archive.read("provider_runtime/wam_provider_runtime_manifest.json").decode("utf-8")
        )
    assert runtime_manifest["official_case_smoke"] == "agibot_465"
    assert runtime_manifest["official_case_rgb_video"] == ""
    assert runtime_manifest["official_case_use_script"] == "true"


def test_build_oscar_wam_provider_bundle_packages_projected_g1_overlay_rgb_context(
    tmp_path: Path,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    projected_trace = tmp_path / "g1_projected_skeleton_trace.jsonl"
    _write_projected_skeleton_trace(projected_trace)
    rollout_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_rollout_input_manifest.v1",
                "source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job"),
                "inputs": {"g1_projected_skeleton_trace_jsonl": str(projected_trace)},
            }
        ),
        encoding="utf-8",
    )
    oscar_input = tmp_path / "oscar_input"
    oscar_input.mkdir()
    _write_review_png(oscar_input / "first_frame.png")
    _write_useful_conditioning_mp4(oscar_input / "blueprint_proxy_skeleton_conditioning.mp4")
    review_video = tmp_path / "head_pov_review.mp4"
    _write_useful_conditioning_mp4(review_video)
    (tmp_path / "oscar_wam_input_package_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "blueprint_oscar_wam_input_package.v1",
                "source_review_video_path": str(review_video),
                "rgb_video": {
                    "path": str(review_video),
                    "source": "selected_review_video_rgb_context",
                    "used_for_oscar_rgb_latent_context": True,
                },
                "source_review_video": {
                    "path": str(review_video),
                    "camera": "head_pov",
                    "video_path": str(review_video),
                },
                "projected_skeleton_trace": {
                    "path": str(projected_trace),
                    "used_for_conditioning": True,
                    "row_count": 1,
                    "projectable_row_count": 1,
                },
                "skeleton_video": {
                    "projected_g1_skeleton_rendered": True,
                    "selected_review_video_background_used": True,
                    "conditioning_mode": "projected_g1_skeleton_rgb_overlay",
                },
                "claim_boundary": {
                    "projected_g1_skeleton_conditioning_used": True,
                    "projected_g1_skeleton_conditioning_is_simulated_mujoco_state": True,
                    "projected_g1_skeleton_conditioning_is_not_physical_robot_sensor_evidence": True,
                    "rgb_video_arg_omitted_for_projected_g1_skeleton_conditioning": False,
                },
                "source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job"),
            }
        ),
        encoding="utf-8",
    )

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "bundle-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_input_dir=oscar_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    bundle_path = Path(str(manifest["bundle_path"]))
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        runtime_manifest_text = archive.read(
            "provider_runtime/wam_provider_runtime_manifest.json"
        ).decode("utf-8")
        runtime_manifest = json.loads(runtime_manifest_text)
    assert "provider_runtime/oscar_input/rgb_context.mp4" in names
    runtime_input_package = runtime_manifest["input_package"]
    assert (
        runtime_input_package["rgb_video"]["path"] == "provider_runtime/oscar_input/rgb_context.mp4"
    )
    assert runtime_input_package["rgb_video"]["used_for_oscar_rgb_latent_context"] is True
    assert (
        runtime_input_package["oscar_rgb_context_runtime_contract"]["rgb_context_packaged"] is True
    )
    assert (
        runtime_input_package["oscar_rgb_context_runtime_contract"][
            "projected_g1_skeleton_conditioning_suppresses_rgb_context"
        ]
        is False
    )
    assert (
        runtime_input_package["oscar_rgb_context_runtime_contract"][
            "projected_g1_rgb_context_enabled"
        ]
        is True
    )
    assert runtime_manifest["oscar_runtime_argv_contract"]["rgb_video_arg_expected"] is True
    assert runtime_manifest["oscar_runtime_argv_contract"]["rgb_video_arg"] == (
        "provider_runtime/oscar_input/rgb_context.mp4"
    )
    assert (
        runtime_manifest["oscar_runtime_argv_contract"]["projected_g1_rgb_context_enabled"] is True
    )
    assert str(review_video) not in runtime_manifest_text
    assert str(tmp_path) not in runtime_manifest_text


def test_build_oscar_wam_provider_bundle_blocks_missing_rollout_input(tmp_path: Path) -> None:
    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "bundle-job",
        wam_rollout_input_manifest=tmp_path / "missing.json",
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "blocked"
    assert "wam_rollout_input_manifest_missing" in manifest["blockers"]


def test_oscar_wam_provider_bundle_existing_input_edges(tmp_path: Path) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    rollout_input.write_text(
        json.dumps({"source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job")}),
        encoding="utf-8",
    )
    oscar_input = tmp_path / "oscar_input"
    oscar_input.mkdir()
    with pytest.raises(FileNotFoundError, match="oscar_input_first_frame_missing"):
        bundle_module._materialized_package_from_existing(
            oscar_input_dir=oscar_input,
            package_manifest_path=None,
            rollout_manifest={},
        )

    (oscar_input / "first_frame.png").write_bytes(b"png")
    with pytest.raises(
        FileNotFoundError,
        match="oscar_input_skeleton_conditioning_video_missing",
    ):
        bundle_module._materialized_package_from_existing(
            oscar_input_dir=oscar_input,
            package_manifest_path=None,
            rollout_manifest={},
        )

    (oscar_input / "blueprint_proxy_skeleton_conditioning.mp4").write_bytes(b"mp4")
    package_manifest = tmp_path / "package.json"
    package_manifest.write_text(
        json.dumps(
            {
                "first_frame": {"path": "old.png", "width": 64},
                "skeleton_video": {"path": "old.mp4", "fps": 5},
                "claim_boundary": {"existing": True},
            }
        ),
        encoding="utf-8",
    )
    package = bundle_module._materialized_package_from_existing(
        oscar_input_dir=oscar_input,
        package_manifest_path=package_manifest,
        rollout_manifest={},
    )
    assert package["first_frame"]["path"] == str(oscar_input / "first_frame.png")
    assert package["first_frame"]["width"] == 64
    assert package["claim_boundary"]["existing"] is True
    assert package["conditioning_video_decode_valid_for_review"] is False
    assert package["conditioning_video_visually_useful_for_model_input"] is False


def test_build_oscar_wam_provider_bundle_blocks_invalid_existing_conditioning_video(
    tmp_path: Path,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    rollout_input.write_text(
        json.dumps({"source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job")}),
        encoding="utf-8",
    )
    oscar_input = tmp_path / "oscar_input"
    oscar_input.mkdir()
    _write_review_png(oscar_input / "first_frame.png")
    (oscar_input / "blueprint_proxy_skeleton_conditioning.mp4").write_bytes(b"not-an-mp4")

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "invalid-condition-bundle-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_input_dir=oscar_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "blocked"
    assert manifest["local_bundle_ready_for_remote_staging"] is False
    assert "oscar_input_skeleton_conditioning_video_unreadable" in manifest["blockers"]
    assert "oscar_input_skeleton_conditioning_video_decode_invalid" in manifest["blockers"]

    _write_dark_conditioning_mp4(oscar_input / "blueprint_proxy_skeleton_conditioning.mp4")
    dark = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "dark-condition-bundle-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_input_dir=oscar_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )
    assert dark["status"] == "blocked"
    assert "oscar_input_skeleton_conditioning_video_visual_smoke_failed" in dark["blockers"]
    assert "oscar_input_skeleton_conditioning_video_not_visually_useful" in dark["blockers"]

    _write_useful_conditioning_mp4(oscar_input / "blueprint_proxy_skeleton_conditioning.mp4")
    (tmp_path / "oscar_wam_input_package_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "blueprint_oscar_wam_input_package.v1",
                "skeleton_video": {
                    "visual_signal": {
                        "status": "warning_low_signal_proxy_conditioning",
                        "blockers": ["proxy_conditioning_foreground_fraction_too_low"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    low_signal = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "low-signal-condition-bundle-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_input_dir=oscar_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )
    assert low_signal["status"] == "blocked"
    assert "oscar_input_skeleton_conditioning_low_signal" in low_signal["blockers"]
    assert (
        "oscar_input_skeleton_conditioning_proxy_conditioning_foreground_fraction_too_low"
        in low_signal["blockers"]
    )


def test_action_conditioned_projected_skeleton_trace_must_be_temporal(
    tmp_path: Path,
) -> None:
    projected_trace = tmp_path / "g1_projected_skeleton_trace.jsonl"
    projected_trace.write_text(
        json.dumps(
            {
                "projected_landmark_count": 2,
                "landmarks": [
                    {
                        "landmark_id": "left_hand",
                        "image_projection": {"available": True, "u_px": 24, "v_px": 26},
                    },
                    {
                        "landmark_id": "right_hand",
                        "image_projection": {"available": True, "u_px": 40, "v_px": 26},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    package = {
        "schema_version": "blueprint_oscar_wam_input_package.v1",
        "conditioning_video_review_validation": {"status": "completed", "blockers": []},
        "conditioning_video_visual_smoke": {
            "status": "failed_visual_quality_smoke",
            "blockers": ["generated_rollout_first_frame_not_scene_like"],
        },
        "conditioning_video_decode_valid_for_review": True,
        "conditioning_video_visually_useful_for_model_input": True,
        "requested_output": {"action_conditioned_generation_required": True},
        "skeleton_video": {
            "conditioning_mode": "projected_g1_skeleton",
            "projected_g1_skeleton_rendered": True,
            "projected_g1_skeleton_landmark_draw_count": 2,
            "projected_g1_skeleton_max_interframe_motion_px": 0.0,
            "skeleton_stream_separate_from_rgb": True,
            "skeleton_stream_texture_free": True,
            "visual_signal": {"status": "completed", "blockers": []},
        },
        "projected_skeleton_trace": {
            "path": str(projected_trace),
            "used_for_conditioning": True,
            "row_count": 1,
            "projectable_row_count": 1,
            "max_interframe_landmark_motion_px": 0.0,
        },
        "claim_boundary": {"projected_g1_skeleton_conditioning_used": True},
    }

    blockers = bundle_module._conditioning_video_input_blockers(package)

    assert "oscar_input_projected_g1_skeleton_trace_not_temporal_action" in blockers


def test_projected_skeleton_renderer_scales_source_viewport_to_wam_canvas(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "isaac_projected_skeleton_trace.jsonl"
    rows = []
    for index in range(2):
        rows.append(
            {
                "schema_version": "blueprint.g1.isaac_geometry_policy_action_projected_skeleton.v1",
                "status": "completed",
                "frame_index": index,
                "image_width_px": 1280,
                "image_height_px": 960,
                "projected_landmark_count": 4,
                "landmarks": [
                    {
                        "landmark_id": "left_wrist",
                        "image_projection": {
                            "available": True,
                            "u_px": 470 + index * 8,
                            "v_px": 777 - index * 6,
                            "inside_image": True,
                            "image_width_px": 1280,
                            "image_height_px": 960,
                        },
                    },
                    {
                        "landmark_id": "left_hand",
                        "image_projection": {
                            "available": True,
                            "u_px": 552 + index * 8,
                            "v_px": 531 - index * 6,
                            "inside_image": True,
                            "image_width_px": 1280,
                            "image_height_px": 960,
                        },
                    },
                    {
                        "landmark_id": "right_wrist",
                        "image_projection": {
                            "available": True,
                            "u_px": 790 - index * 8,
                            "v_px": 705 - index * 6,
                            "inside_image": True,
                            "image_width_px": 1280,
                            "image_height_px": 960,
                        },
                    },
                    {
                        "landmark_id": "right_hand",
                        "image_projection": {
                            "available": True,
                            "u_px": 738 - index * 8,
                            "v_px": 550 - index * 6,
                            "inside_image": True,
                            "image_width_px": 1280,
                            "image_height_px": 960,
                        },
                    },
                ],
                "segments": [
                    {"from": "left_wrist", "to": "left_hand"},
                    {"from": "right_wrist", "to": "right_hand"},
                ],
            }
        )
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    skeleton_video, _ = bundle_module._render_projected_skeleton_conditioning_video(
        trace_path=trace,
        output_path=tmp_path / "conditioning.mp4",
        width=128,
        height=128,
        fps=5.0,
        num_frames=2,
    )

    assert skeleton_video["visual_signal"]["status"] == "completed"
    assert skeleton_video["projected_g1_skeleton_landmark_draw_count"] == 4
    capture = cv2.VideoCapture(str(tmp_path / "conditioning.mp4"))
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert int(np.count_nonzero(frame)) > 0


def test_projected_skeleton_renderer_accepts_official_g1_link_ids_without_segments(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    trace = tmp_path / "official_g1_trace.jsonl"
    rows = []
    for frame_index in range(2):
        landmarks = []
        for side, x_offset in (("left", -30), ("right", 30)):
            for landmark_id, y_offset in (
                (f"{side}_shoulder_pitch_link", -60),
                (f"{side}_shoulder_roll_link", -45),
                (f"{side}_shoulder_yaw_link", -30),
                (f"{side}_elbow_link", 0),
                (f"{side}_wrist_roll_link", 25),
                (f"{side}_wrist_pitch_link", 35),
                (f"{side}_wrist_yaw_link", 45),
                (f"{side}_hand_index_0_link", 65),
                (f"{side}_hand_index_1_link", 80),
            ):
                landmarks.append(
                    {
                        "landmark_id": landmark_id,
                        "image_projection": {
                            "available": True,
                            "u_px": 320 + x_offset + frame_index,
                            "v_px": 240 + y_offset,
                            "image_width_px": 640,
                            "image_height_px": 480,
                        },
                    }
                )
        rows.append({"frame_index": frame_index, "projected_landmarks": landmarks})
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report, _ = bundle_module._render_projected_skeleton_conditioning_video(
        trace_path=trace,
        output_path=tmp_path / "conditioning.mp4",
        width=640,
        height=480,
        fps=5.0,
        num_frames=2,
    )

    assert report["visual_signal"] == {
        "status": "completed",
        "blockers": [],
        "trace_row_count": 2,
        "projectable_row_count": 2,
        "max_interframe_landmark_motion_px": 1.0,
        "max_visible_landmark_draw_count": 18,
        "minimum_true_in_frame_landmark_count": 18,
        "minimum_true_in_frame_effector_count": 10,
        "max_visible_segment_count": 16,
        "max_clipped_landmark_count": 0,
        "max_offscreen_edge_indicator_count": 0,
        "max_end_effector_axis_draw_count": 2,
    }
    capture = cv2.VideoCapture(str(tmp_path / "conditioning.mp4"))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None


@pytest.mark.parametrize(
    ("motion_px", "expected_status"),
    ((0, "warning_low_signal_projected_skeleton"), (12, "completed")),
)
def test_controller_fk_action_horizon_requires_visible_effector_motion(
    tmp_path: Path,
    motion_px: int,
    expected_status: str,
) -> None:
    pytest.importorskip("cv2")
    trace = tmp_path / f"controller_fk_trace_{motion_px}.jsonl"
    rows = []
    for frame_index in range(2):
        offset = frame_index * motion_px
        landmarks = []
        for side, x in (("left", 220), ("right", 420)):
            landmarks.extend(
                [
                    {
                        "landmark_id": f"{side}_wrist",
                        "image_projection": {
                            "available": True,
                            "u_px": x + offset,
                            "v_px": 300,
                            "image_width_px": 640,
                            "image_height_px": 480,
                        },
                    },
                    {
                        "landmark_id": f"{side}_hand",
                        "image_projection": {
                            "available": True,
                            "u_px": x + offset,
                            "v_px": 250,
                            "image_width_px": 640,
                            "image_height_px": 480,
                        },
                    },
                ]
            )
        rows.append(
            {
                "frame_index": frame_index,
                "projected_landmarks": landmarks,
                "segments": [
                    {"from": "left_wrist", "to": "left_hand"},
                    {"from": "right_wrist", "to": "right_hand"},
                ],
            }
        )
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report, _ = bundle_module._render_projected_skeleton_conditioning_video(
        trace_path=trace,
        output_path=tmp_path / f"controller_fk_{motion_px}.mp4",
        width=640,
        height=480,
        fps=5.0,
        num_frames=2,
        conditioning_mode="controller_fk_action_horizon",
    )

    assert report["visual_signal"]["status"] == expected_status
    if motion_px == 0:
        assert (
            "controller_fk_skeleton_trace_motion_too_low" in (report["visual_signal"]["blockers"])
        )
    else:
        assert report["visual_signal"]["blockers"] == []


def test_controller_fk_action_horizon_ignores_non_action_seed_visibility_for_gate(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cv2")
    trace = tmp_path / "controller_fk_trace_with_offscreen_seed.jsonl"

    def _landmarks(*, offset: int, available: bool) -> list[dict[str, object]]:
        return [
            {
                "landmark_id": f"{side}_{part}",
                "image_projection": {
                    "available": available,
                    "u_px": x + offset,
                    "v_px": y,
                    "image_width_px": 640,
                    "image_height_px": 480,
                    **(
                        {}
                        if available
                        else {"unavailable_reason": "outside_live_camera_viewport"}
                    ),
                },
            }
            for side, x in (("left", 220), ("right", 420))
            for part, y in (("wrist", 300), ("hand", 250))
        ]

    rows = [
        {
            "frame_index": 0,
            "source_controller_horizon_frame_index": -1,
            "projected_landmarks": _landmarks(offset=700, available=False),
            "segments": [
                {"from": "left_wrist", "to": "left_hand"},
                {"from": "right_wrist", "to": "right_hand"},
            ],
        },
        *[
            {
                "frame_index": action_index + 1,
                "source_controller_horizon_frame_index": action_index,
                "projected_landmarks": _landmarks(
                    offset=action_index * 12,
                    available=True,
                ),
                "segments": [
                    {"from": "left_wrist", "to": "left_hand"},
                    {"from": "right_wrist", "to": "right_hand"},
                ],
            }
            for action_index in range(2)
        ],
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report, _ = bundle_module._render_projected_skeleton_conditioning_video(
        trace_path=trace,
        output_path=tmp_path / "conditioning.mp4",
        width=640,
        height=480,
        fps=5.0,
        num_frames=3,
        conditioning_mode="controller_fk_action_horizon",
    )

    assert report["visual_signal"]["status"] == "completed"
    assert report["visual_signal"]["blockers"] == []
    assert report["projected_g1_skeleton_minimum_true_in_frame_landmark_count"] == 0
    assert report["projected_g1_skeleton_action_horizon_frame_count"] == 2
    assert (
        report[
            "projected_g1_skeleton_action_horizon_minimum_true_in_frame_landmark_count"
        ]
        == 4
    )
    assert (
        report[
            "projected_g1_skeleton_action_horizon_minimum_true_in_frame_effector_count"
        ]
        == 4
    )
    assert report["visual_signal"]["gate_scope"] == (
        "controller_fk_action_horizon_frames_only"
    )


def test_projected_skeleton_renderer_encodes_finite_offscreen_fk_at_image_edge(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    trace = tmp_path / "offscreen_g1_trace.jsonl"
    rows = []
    for frame_index in range(2):
        landmarks = []
        for side, x_offset in (("left", -80), ("right", 80)):
            for landmark_id, y_offset in (
                (f"{side}_shoulder_pitch_link", 0),
                (f"{side}_elbow_link", 40),
                (f"{side}_wrist_yaw_link", 80),
                (f"{side}_hand_index_0_link", 120),
            ):
                landmarks.append(
                    {
                        "landmark_id": landmark_id,
                        "image_projection": {
                            "available": False,
                            "unavailable_reason": "outside_live_camera_viewport",
                            "u_px": 320 + x_offset + frame_index * 4,
                            "v_px": 600 + y_offset,
                            "image_width_px": 640,
                            "image_height_px": 480,
                        },
                    }
                )
        rows.append({"frame_index": frame_index, "projected_landmarks": landmarks})
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report, _ = bundle_module._render_projected_skeleton_conditioning_video(
        trace_path=trace,
        output_path=tmp_path / "offscreen_conditioning.mp4",
        width=640,
        height=480,
        fps=5.0,
        num_frames=2,
    )

    assert report["visual_signal"]["status"] == "completed"
    assert report["offscreen_edge_indicators_used"] is True
    assert report["projected_g1_skeleton_offscreen_edge_indicator_count"] == 8
    assert report["projected_g1_skeleton_visible_landmark_draw_count"] == 8
    assert report["projected_g1_skeleton_visible_segment_count"] >= 2
    assert report["projected_g1_skeleton_end_effector_axis_draw_count"] == 2
    capture = cv2.VideoCapture(str(tmp_path / "offscreen_conditioning.mp4"))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None and int(np.count_nonzero(frame)) > 0


def test_short_conditioning_video_can_stage_when_signal_is_useful_for_model_input(
    tmp_path: Path,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    rollout_input.write_text(
        json.dumps({"source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job")}),
        encoding="utf-8",
    )
    oscar_input = tmp_path / "oscar_input"
    oscar_input.mkdir()
    _write_review_png(oscar_input / "first_frame.png")
    _write_useful_conditioning_mp4(oscar_input / "blueprint_proxy_skeleton_conditioning.mp4")
    (tmp_path / "oscar_wam_input_package_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "blueprint_oscar_wam_input_package.v1",
                "skeleton_video": {
                    "visual_signal": {"status": "ok", "blockers": []},
                },
                "claim_boundary": {
                    "conditioning_video_visual_smoke_is_not_wam_output_success": True,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "short-condition-bundle-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_input_dir=oscar_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    assert manifest["local_bundle_ready_for_remote_staging"] is True
    assert "oscar_input_skeleton_conditioning_video_not_visually_useful" not in manifest["blockers"]


def test_oscar_wam_provider_bundle_blocks_missing_projected_skeleton_trace_when_claimed(
    tmp_path: Path,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    rollout_input.write_text(
        json.dumps({"source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job")}),
        encoding="utf-8",
    )
    oscar_input = tmp_path / "oscar_input"
    oscar_input.mkdir()
    _write_review_png(oscar_input / "first_frame.png")
    _write_useful_conditioning_mp4(oscar_input / "blueprint_proxy_skeleton_conditioning.mp4")
    (tmp_path / "oscar_wam_input_package_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "blueprint_oscar_wam_input_package.v1",
                "projected_skeleton_trace": {
                    "path": str(tmp_path / "missing_projected_trace.jsonl"),
                    "used_for_conditioning": True,
                    "row_count": 0,
                    "projectable_row_count": 0,
                },
                "skeleton_video": {"projected_g1_skeleton_rendered": True},
                "claim_boundary": {"projected_g1_skeleton_conditioning_used": True},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "missing-projected-trace-bundle-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_input_dir=oscar_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "blocked"
    assert "oscar_input_projected_g1_skeleton_trace_missing" in manifest["blockers"]
    assert "oscar_input_projected_g1_skeleton_trace_empty" in manifest["blockers"]


def test_oscar_wam_provider_bundle_local_materialization_and_failure_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    rollout_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_rollout_input_manifest.v1",
                "source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job"),
            }
        ),
        encoding="utf-8",
    )
    job_dir = tmp_path / "bundle-job"
    (job_dir / "oscar_wam_provider_bundle").mkdir(parents=True)
    captured: dict[str, Any] = {}

    def fake_materialize(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        first = kwargs["work_dir"] / "first_frame.png"
        skeleton = kwargs["work_dir"] / "blueprint_proxy_skeleton_conditioning.mp4"
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_bytes(b"png")
        skeleton.write_bytes(b"mp4")
        return {
            "schema_version": "blueprint_oscar_wam_input_package.v1",
            "first_frame": {"path": str(first)},
            "skeleton_video": {"path": str(skeleton)},
            "conditioning_video_review_validation": {"status": "completed", "blockers": []},
            "conditioning_video_visual_smoke": {
                "status": "passed_visual_quality_smoke",
                "blockers": [],
                "claim_boundary": {"visual_rollout_useful_for_task_success_review": True},
                "rollouts": [
                    {
                        "generated_video_path": str(skeleton),
                        "sampled_frames": [{"path": str(kwargs["work_dir"] / "frame.jpg")}],
                    }
                ],
            },
            "conditioning_video_decode_valid_for_review": True,
            "conditioning_video_visually_useful_for_model_input": True,
            "claim_boundary": {},
        }

    monkeypatch.setattr(bundle_module, "_materialize_oscar_input_package", fake_materialize)
    manifest = build_oscar_wam_provider_bundle(
        job_dir=job_dir,
        wam_rollout_input_manifest=rollout_input,
        num_frames=49,
        height=352,
        width=640,
        fps=12.5,
        generated_at="2026-06-21T00:00:00+00:00",
    )
    assert manifest["status"] == "completed"
    assert captured["width"] == 640
    assert captured["height"] == 352
    assert captured["num_frames"] == 49
    assert captured["fps"] == 12.5
    assert not (job_dir / "oscar_wam_provider_bundle" / "stale").exists()

    def raise_materialize(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise RuntimeError("cannot render inputs")

    monkeypatch.setattr(bundle_module, "_materialize_oscar_input_package", raise_materialize)
    blocked = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "blocked-job",
        wam_rollout_input_manifest=rollout_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == [
        "oscar_wam_input_package_materialization_failed:RuntimeError",
        "oscar_wam_input_package_materialization_error:cannot render inputs",
    ]
    assert blocked["input_package_materialization_error"] == {
        "type": "RuntimeError",
        "message": "cannot render inputs",
        "raw_message_omitted_if_path_like": False,
    }


def test_oscar_wam_provider_bundle_materializes_wam_generation_step_input(
    tmp_path: Path,
) -> None:
    source_frame = tmp_path / "policy_observation.jpg"
    _write_review_png(source_frame)
    depth = tmp_path / "depth.npy"
    depth.write_bytes(b"depth")
    target_mask = tmp_path / "target_mask.png"
    robot_mask = tmp_path / "robot_mask.png"
    _write_review_png(target_mask)
    _write_review_png(robot_mask)
    step_input = tmp_path / "wam_generation_step_0001_input.json"
    step_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_generation_step_input.v1",
                "step_index": 1,
                "source_policy_observation_frame_path": str(source_frame),
                "source_policy_action": {
                    "action_type": "unitree_g1_sonic_latent_action_chunk",
                    "action_chunk": [0.12, -0.05, 0.18, 0.03],
                    "unitree_groot_n17_sonic_action_chunk_present": True,
                },
                "policy_action_to_skeleton_contract": {
                    "schema_version": "persistent_wam_policy_action_to_skeleton_contract.v1",
                    "status": "no_policy_derived_projected_skeleton_trace_available",
                    "source_policy_action_present": True,
                    "policy_derived_projected_skeleton_trace_present": False,
                    "policy_ranking_claim_safe": False,
                    "blockers": [
                        "policy_action_to_projected_skeleton_decoder_missing_for_ranking_safe_wam"
                    ],
                },
                "current_policy_observation": {
                    "task_id": "turn_on_sink_handle",
                    "target_object_id": "Sink054_handle",
                    "robot_profile_id": "unitree_g1_sonic",
                    "source_kind": "synthetic_gpt_image_2_seed",
                    "visual_observation": {
                        "camera_id": "head_pov",
                        "depth_map_path": str(depth),
                        "target_segmentation_mask_path": str(target_mask),
                        "robot_mask_path": str(robot_mask),
                        "target_bbox": {
                            "x_min": 0.42,
                            "y_min": 0.30,
                            "x_max": 0.60,
                            "y_max": 0.52,
                        },
                        "target_keypoints": {"handle_tip": {"x": 0.54, "y": 0.42}},
                        "affordance_points": {
                            "turn_handle_axis": {"center": {"x": 0.51, "y": 0.41}}
                        },
                    },
                },
                "requested_output": {
                    "next_observation_frame_path": str(tmp_path / "next.jpg"),
                    "action_conditioned_generation_required": True,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "step-bundle-job",
        wam_rollout_input_manifest=step_input,
        num_frames=8,
        height=64,
        width=64,
        fps=5.0,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    assert manifest["input_package_source_schema_version"] == "wam_generation_step_input.v1"
    assert manifest["blockers"] == []
    bundle_path = Path(str(manifest["bundle_path"]))
    assert bundle_path.is_file()
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        runtime_auxiliary = json.loads(
            archive.read("provider_runtime/oscar_input/wam_auxiliary_observation_manifest.json")
        )
    assert "provider_runtime/oscar_input/first_frame.png" in names
    assert "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4" in names
    assert "provider_runtime/oscar_input/rgb_context.mp4" in names
    assert "provider_runtime/oscar_input/wam_auxiliary_observation_manifest.json" in names
    assert runtime_auxiliary["source_image_path"] == "provider_runtime/oscar_input/first_frame.png"
    assert str(tmp_path) not in json.dumps(runtime_auxiliary)
    runtime_manifest = _read_json(
        tmp_path
        / "step-bundle-job"
        / "oscar_wam_provider_bundle"
        / "provider_runtime"
        / "wam_provider_runtime_manifest.json"
    )
    input_package = runtime_manifest["input_package"]
    assert input_package["source_action"]["unitree_groot_n17_sonic_action_chunk_present"] is True
    assert input_package["policy_action_to_skeleton_contract"]["policy_ranking_claim_safe"] is False
    assert input_package["wam_auxiliary_observation_manifest_path"] == (
        "provider_runtime/oscar_input/wam_auxiliary_observation_manifest.json"
    )
    assert input_package["wam_auxiliary_observation"]["modalities_available"]["depth"] is True
    assert (
        input_package["oscar_auxiliary_observation_runtime_contract"][
            "auxiliary_observation_manifest_packaged"
        ]
        is True
    )
    assert input_package["rgb_video"]["path"] == "provider_runtime/oscar_input/rgb_context.mp4"
    assert input_package["rgb_video"]["used_for_oscar_rgb_latent_context"] is True
    assert input_package["rgb_video"]["rgb_context_mode"] == "single_frame_repeat"
    assert input_package["oscar_rgb_context_runtime_contract"]["rgb_context_packaged"] is True
    contract = runtime_manifest["oscar_input_contract_diagnostic"]
    assert contract["status"] == "warning_high_risk"
    assert contract["rgb_context"]["rgb_context_mode"] == "single_frame_repeat"
    assert contract["skeleton_video"]["policy_action_proxy_used"] is True
    assert contract["projected_skeleton_trace"]["used_for_conditioning"] is False
    assert "oscar_contract_rgb_context_single_frame_repeat" in contract["warnings"]
    assert (
        "oscar_contract_policy_action_proxy_conditioning_without_projected_skeleton"
        in contract["warnings"]
    )
    assert "oscar_contract_policy_action_to_skeleton_not_ranking_safe" in contract["warnings"]
    assert "oscar_contract_single_frame_repeat_without_projected_skeleton" in contract["warnings"]
    assert (
        "rgb_context_single_frame_repeat_autoregressive_risk"
        in contract["autoregressive_risk_flags"]
    )
    assert (
        "policy_action_proxy_without_projected_skeleton_autoregressive_risk"
        in contract["autoregressive_risk_flags"]
    )
    assert "single_frame_repeat_without_projected_skeleton_high_risk" in contract["high_risk_flags"]
    assert (
        "policy_action_proxy_without_decoded_skeleton_ranking_risk"
        in contract["ranking_risk_flags"]
    )
    assert "policy_action_to_skeleton_contract_not_ranking_safe" in contract["ranking_risk_flags"]
    assert contract["policy_ranking_claim_safe"] is False
    assert contract["policy_action_to_skeleton_contract"]["status"] == (
        "no_policy_derived_projected_skeleton_trace_available"
    )
    assert "policy_action_to_skeleton_adapter" in contract["likely_debug_focus"]
    assert contract["autoregressive_risk_level"] == "high"
    assert contract["short_rollout_sanity_recommended_before_scale_up"] is True
    assert input_package["oscar_input_contract_diagnostic"] == contract
    assert input_package["skeleton_video"]["visual_signal"]["auxiliary_target_overlay_used"] is True
    assert input_package["skeleton_video"]["visual_signal"]["auxiliary_target_bbox_used"] is True


def test_wam_generation_step_input_prefers_projected_skeleton_trace(
    tmp_path: Path,
) -> None:
    source_frame = tmp_path / "policy_observation.jpg"
    _write_review_png(source_frame)
    projected_trace = tmp_path / "g1_projected_skeleton_trace.jsonl"
    _write_projected_skeleton_trace(projected_trace)
    step_input = tmp_path / "wam_generation_step_0001_input.json"
    step_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_generation_step_input.v1",
                "step_index": 1,
                "source_policy_observation_frame_path": str(source_frame),
                "source_policy_action": {
                    "action_type": "accepted_direct_collision_checked_motion",
                },
                "current_policy_observation": {
                    "task_id": "open_fridge",
                    "target_object_id": "refrigerator",
                    "robot_profile_id": "unitree_g1",
                    "visual_observation": {
                        "camera_id": "head_pov",
                        "g1_projected_skeleton_trace_jsonl": str(projected_trace),
                    },
                },
                "requested_output": {
                    "next_observation_frame_path": str(tmp_path / "next.jpg"),
                    "action_conditioned_generation_required": True,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "step-projected-bundle-job",
        wam_rollout_input_manifest=step_input,
        num_frames=8,
        height=64,
        width=64,
        fps=5.0,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    assert manifest["input_package_conditioning_video_blockers"] == []
    bundle_path = Path(str(manifest["bundle_path"]))
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/oscar_input/g1_projected_skeleton_trace.jsonl" in names
    assert "provider_runtime/oscar_input/rgb_context.mp4" not in names
    runtime_manifest = _read_json(
        tmp_path
        / "step-projected-bundle-job"
        / "oscar_wam_provider_bundle"
        / "provider_runtime"
        / "wam_provider_runtime_manifest.json"
    )
    input_package = runtime_manifest["input_package"]
    assert input_package["skeleton_video"]["conditioning_mode"] == "projected_g1_skeleton"
    assert input_package["skeleton_video"]["projected_g1_skeleton_rendered"] is True
    assert input_package["projected_skeleton_trace"]["used_for_conditioning"] is True
    assert (
        input_package["oscar_projected_skeleton_runtime_contract"][
            "projected_skeleton_trace_packaged"
        ]
        is True
    )
    assert input_package["rgb_video"]["used_for_oscar_rgb_latent_context"] is False
    contract = runtime_manifest["oscar_input_contract_diagnostic"]
    assert manifest["input_package_contract_diagnostic"]["status"] == "warning_high_risk"
    assert contract["status"] == "warning_high_risk"
    assert contract["projected_skeleton_trace"]["used_for_conditioning"] is True
    assert contract["rgb_context"]["used_for_oscar_rgb_latent_context"] is False
    assert (
        contract["rgb_context"]["rgb_context_mode"]
        == "omitted_first_frame_plus_skeleton_public_contract"
    )
    assert "oscar_contract_rgb_context_omitted_with_projected_skeleton" in contract["warnings"]
    assert (
        "oscar_contract_projected_skeleton_not_scene_faithful_policy_action_bridge"
        in contract["warnings"]
    )
    assert "oscar_contract_guidance_high_for_contract_debug" not in contract["warnings"]
    assert "guidance_high_autoregressive_debug_risk" not in contract["autoregressive_risk_flags"]
    assert (
        "projected_skeleton_missing_scene_faithful_policy_action_bridge"
        in contract["ranking_risk_flags"]
    )
    assert contract["autoregressive_risk_level"] == "high"
    assert (
        input_package["oscar_rgb_context_runtime_contract"]["projected_g1_rgb_context_enabled"]
        is False
    )
    assert (
        input_package["oscar_rgb_context_runtime_contract"][
            "projected_g1_skeleton_conditioning_suppresses_rgb_context"
        ]
        is True
    )
    assert input_package["claim_boundary"]["policy_action_conditioning_proxy_video_used"] is False


def test_wam_generation_step_input_can_force_oscar_proxy_with_rgb_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_frame = tmp_path / "policy_observation.jpg"
    _write_review_png(source_frame)
    projected_trace = tmp_path / "policy_action_projected_skeleton_trace.jsonl"
    _write_projected_skeleton_trace(
        projected_trace,
        claim_boundary=SCENE_FAITHFUL_ISAAC_POLICY_ACTION_TRACE_CLAIM,
    )
    step_input = tmp_path / "wam_generation_step_0001_input.json"
    step_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_generation_step_input.v1",
                "step_index": 1,
                "source_policy_observation_frame_path": str(source_frame),
                "source_policy_action": {
                    "action_type": "unitree_g1_sonic_latent_action_chunk",
                    "action_chunk": [0.1, 0.2],
                    "policy_action_projected_skeleton_trace_path": str(projected_trace),
                },
                "current_policy_observation": {
                    "task_id": "open_fridge",
                    "target_object_id": "refrigerator",
                    "robot_profile_id": "unitree_g1",
                    "visual_observation": {"camera_id": "head_pov"},
                },
                "requested_output": {
                    "next_observation_frame_path": str(tmp_path / "next.jpg"),
                    "action_conditioned_generation_required": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CONDITIONING_MODE", "oscar_gripper_scenario_proxy")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_RGB_CONTEXT_MODE", "always")

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "step-forced-oscar-proxy-bundle-job",
        wam_rollout_input_manifest=step_input,
        num_frames=8,
        height=64,
        width=64,
        fps=5.0,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    with zipfile.ZipFile(manifest["bundle_path"]) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/oscar_input/rgb_context.mp4" in names
    runtime_manifest = _read_json(
        tmp_path
        / "step-forced-oscar-proxy-bundle-job"
        / "oscar_wam_provider_bundle"
        / "provider_runtime"
        / "wam_provider_runtime_manifest.json"
    )
    input_package = runtime_manifest["input_package"]
    skeleton_video = input_package["skeleton_video"]
    assert skeleton_video["conditioning_mode"] == "oscar_gripper_scenario_proxy"
    assert skeleton_video["oscar_gripper_scenario_proxy_rendered"] is True
    assert skeleton_video["projected_g1_skeleton_rendered"] is False
    assert input_package["projected_skeleton_trace"]["available"] is True
    assert input_package["projected_skeleton_trace"]["used_for_conditioning"] is False
    assert input_package["claim_boundary"]["projected_g1_skeleton_conditioning_used"] is False
    assert input_package["claim_boundary"]["policy_action_conditioning_proxy_video_used"] is True
    assert input_package["rgb_video"]["configured_rgb_context_mode"] == "always"
    assert input_package["rgb_video"]["used_for_oscar_rgb_latent_context"] is True
    assert input_package["rgb_video"]["rgb_context_mode"] == "single_frame_repeat"
    assert runtime_manifest["oscar_runtime_argv_contract"]["rgb_context_packaged"] is True


def test_wam_generation_step_input_can_render_oscar_projected_gripper_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_frame = tmp_path / "policy_observation.jpg"
    _write_review_png(source_frame)
    projected_trace = tmp_path / "policy_action_projected_skeleton_trace.jsonl"
    _write_projected_skeleton_trace(
        projected_trace,
        claim_boundary=SCENE_FAITHFUL_ISAAC_POLICY_ACTION_TRACE_CLAIM,
    )
    step_input = tmp_path / "wam_generation_step_0001_input.json"
    step_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_generation_step_input.v1",
                "step_index": 1,
                "source_policy_observation_frame_path": str(source_frame),
                "source_policy_action": {
                    "action_type": "unitree_g1_sonic_latent_action_chunk",
                    "action_chunk": [0.1, 0.2],
                    "policy_action_projected_skeleton_trace_path": str(projected_trace),
                },
                "current_policy_observation": {
                    "task_id": "open_fridge",
                    "target_object_id": "refrigerator",
                    "robot_profile_id": "unitree_g1",
                    "visual_observation": {"camera_id": "head_pov"},
                },
                "requested_output": {
                    "next_observation_frame_path": str(tmp_path / "next.jpg"),
                    "action_conditioned_generation_required": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CONDITIONING_MODE", "oscar_projected_gripper_axes")

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "step-oscar-gripper-axes-bundle-job",
        wam_rollout_input_manifest=step_input,
        num_frames=8,
        height=64,
        width=64,
        fps=5.0,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    runtime_manifest = _read_json(
        tmp_path
        / "step-oscar-gripper-axes-bundle-job"
        / "oscar_wam_provider_bundle"
        / "provider_runtime"
        / "wam_provider_runtime_manifest.json"
    )
    input_package = runtime_manifest["input_package"]
    skeleton_video = input_package["skeleton_video"]
    assert skeleton_video["conditioning_mode"] == "oscar_projected_gripper_axes"
    assert skeleton_video["oscar_projected_gripper_axes_rendered"] is True
    assert skeleton_video["projected_g1_skeleton_rendered"] is True
    assert input_package["projected_skeleton_trace"]["used_for_conditioning"] is True
    assert input_package["rgb_video"]["used_for_oscar_rgb_latent_context"] is False
    assert (
        skeleton_video["claim_boundary"]["oscar_style_gripper_axes_proxy_from_projected_g1_trace"]
        is True
    )
    video_path = (
        tmp_path
        / "step-oscar-gripper-axes-bundle-job"
        / "local_input_materialization"
        / "oscar_input"
        / "blueprint_proxy_skeleton_conditioning.mp4"
    )
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    assert ok
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert float(np.count_nonzero(gray > 12)) / float(gray.size) < 0.20
    assert int(gray.max() - gray.min()) > 80


def test_wam_generation_step_input_accepts_policy_action_projected_skeleton_trace(
    tmp_path: Path,
) -> None:
    source_frame = tmp_path / "policy_observation.jpg"
    _write_review_png(source_frame)
    projected_trace = tmp_path / "policy_action_projected_skeleton_trace.jsonl"
    _write_projected_skeleton_trace(
        projected_trace,
        claim_boundary=SCENE_FAITHFUL_ISAAC_POLICY_ACTION_TRACE_CLAIM,
    )
    step_input = tmp_path / "wam_generation_step_0001_input.json"
    step_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_generation_step_input.v1",
                "step_index": 1,
                "source_policy_observation_frame_path": str(source_frame),
                "source_policy_action": {
                    "action_type": "unitree_g1_sonic_latent_action_chunk",
                    "action_chunk": [0.1, 0.2],
                    "policy_action_projected_skeleton_trace_path": str(projected_trace),
                },
                "current_policy_observation": {
                    "task_id": "open_fridge",
                    "target_object_id": "refrigerator",
                    "robot_profile_id": "unitree_g1",
                    "visual_observation": {"camera_id": "head_pov"},
                },
                "requested_output": {
                    "next_observation_frame_path": str(tmp_path / "next.jpg"),
                    "action_conditioned_generation_required": True,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "step-policy-action-projected-bundle-job",
        wam_rollout_input_manifest=step_input,
        num_frames=8,
        height=64,
        width=64,
        fps=5.0,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    runtime_manifest = _read_json(
        tmp_path
        / "step-policy-action-projected-bundle-job"
        / "oscar_wam_provider_bundle"
        / "provider_runtime"
        / "wam_provider_runtime_manifest.json"
    )
    input_package = runtime_manifest["input_package"]
    assert input_package["skeleton_video"]["conditioning_mode"] == "projected_g1_skeleton"
    assert input_package["projected_skeleton_trace"]["used_for_conditioning"] is True
    assert input_package["projected_skeleton_trace"]["path"] == (
        "provider_runtime/oscar_input/g1_projected_skeleton_trace.jsonl"
    )
    assert input_package["source_action"]["action_chunk_value_count"] == 2
    assert input_package["claim_boundary"]["policy_action_conditioning_proxy_video_used"] is False
    diagnostic = runtime_manifest["oscar_input_contract_diagnostic"]
    projected = diagnostic["projected_skeleton_trace"]
    assert projected["official_wbc_or_sim_bridge_used"] is False
    assert projected["scene_faithful_isaac_policy_action_projection_bridge_used"] is True
    assert projected["policy_action_bridge_safe_for_sim_ranking"] is True
    assert projected["full_g1_urdf_fk_solver_used"] is False
    assert projected["sidecar_kinematic_chain_fk_solver_used"] is True


def test_wam_generation_step_blocks_sparse_projected_skeleton_conditioning(
    tmp_path: Path,
) -> None:
    source_frame = tmp_path / "policy_observation.jpg"
    _write_review_png(source_frame)
    projected_trace = tmp_path / "sparse_policy_action_projected_skeleton_trace.jsonl"
    projected_trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema_version": "blueprint.g1.isaac_geometry_policy_action_projected_skeleton.v1",
                    "status": "completed",
                    "frame_index": index,
                    "projected_landmark_count": 2,
                    "landmarks": [
                        {
                            "landmark_id": "left_hand",
                            "image_projection": {
                                "available": True,
                                "u_px": 24 + index,
                                "v_px": 26,
                            },
                        },
                        {
                            "landmark_id": "right_hand",
                            "image_projection": {
                                "available": True,
                                "u_px": 40 - index,
                                "v_px": 26,
                            },
                        },
                    ],
                    "segments": [{"from": "left_hand", "to": "right_hand"}],
                    "claim_boundary": {
                        "policy_derived_action_conditioning": True,
                        "official_wbc_or_sim_bridge_used": True,
                    },
                }
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    step_input = tmp_path / "wam_generation_step_0001_input.json"
    step_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_generation_step_input.v1",
                "step_index": 1,
                "source_policy_observation_frame_path": str(source_frame),
                "source_policy_action": {
                    "action_type": "unitree_g1_sonic_latent_action_chunk",
                    "action_chunk": [0.1, 0.2],
                    "policy_action_projected_skeleton_trace_path": str(projected_trace),
                },
                "current_policy_observation": {
                    "task_id": "open_fridge",
                    "target_object_id": "refrigerator",
                    "robot_profile_id": "unitree_g1",
                    "visual_observation": {"camera_id": "head_pov"},
                },
                "requested_output": {
                    "next_observation_frame_path": str(tmp_path / "next.jpg"),
                    "action_conditioned_generation_required": True,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "step-sparse-projected-bundle-job",
        wam_rollout_input_manifest=step_input,
        num_frames=8,
        height=64,
        width=64,
        fps=5.0,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "blocked"
    blockers = set(manifest["input_package_conditioning_video_blockers"])
    assert "oscar_input_projected_g1_skeleton_low_signal" in blockers
    assert "oscar_input_projected_skeleton_visible_landmark_count_too_low" in blockers
    assert "oscar_input_projected_skeleton_end_effector_axes_missing" in blockers
    contract = manifest["input_package_contract_diagnostic"]
    assert contract["status"] == "blocked"
    assert "oscar_contract_projected_skeleton_visible_landmarks_too_sparse" in contract["blockers"]
    assert "oscar_contract_projected_skeleton_missing_end_effector_axes" in contract["blockers"]


def test_wam_generation_step_input_packages_real_temporal_rgb_context(
    tmp_path: Path,
) -> None:
    source_frame = tmp_path / "policy_observation_0000.jpg"
    next_frame = tmp_path / "policy_observation_0001.jpg"
    _write_review_png(source_frame)
    _write_review_png(next_frame)
    frame = cv2.imread(str(next_frame), cv2.IMREAD_COLOR)
    assert frame is not None
    cv2.circle(frame, (48, 18), 9, (40, 230, 180), -1)
    assert cv2.imwrite(str(next_frame), frame)
    projected_trace = tmp_path / "policy_action_projected_skeleton_trace.jsonl"
    _write_projected_skeleton_trace(
        projected_trace,
        claim_boundary=SCENE_FAITHFUL_ISAAC_POLICY_ACTION_TRACE_CLAIM,
    )
    step_input = tmp_path / "wam_generation_step_0002_input.json"
    step_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_generation_step_input.v1",
                "step_index": 2,
                "source_policy_observation_frame_path": str(next_frame),
                "rgb_context_frame_paths": [str(source_frame), str(next_frame)],
                "source_policy_action": {
                    "action_type": "unitree_g1_sonic_latent_action_chunk",
                    "action_chunk": [0.1, 0.2],
                    "policy_action_projected_skeleton_trace_path": str(projected_trace),
                },
                "current_policy_observation": {
                    "task_id": "open_fridge",
                    "target_object_id": "refrigerator",
                    "robot_profile_id": "unitree_g1",
                    "visual_observation": {"camera_id": "head_pov"},
                },
                "requested_output": {
                    "next_observation_frame_path": str(tmp_path / "next.jpg"),
                    "action_conditioned_generation_required": True,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "step-temporal-rgb-bundle-job",
        wam_rollout_input_manifest=step_input,
        num_frames=8,
        height=64,
        width=64,
        fps=5.0,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    with zipfile.ZipFile(manifest["bundle_path"]) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/oscar_input/rgb_context.mp4" in names
    runtime_manifest = _read_json(
        tmp_path
        / "step-temporal-rgb-bundle-job"
        / "oscar_wam_provider_bundle"
        / "provider_runtime"
        / "wam_provider_runtime_manifest.json"
    )
    input_package = runtime_manifest["input_package"]
    assert input_package["rgb_video"]["used_for_oscar_rgb_latent_context"] is True
    assert input_package["rgb_video"]["rgb_context_mode"] == "temporal_frame_sequence"
    assert input_package["rgb_video"]["source_frame_count"] == 2
    assert (
        input_package["rgb_video"]["single_frame_policy_observation_repeated_for_oscar_rgb_context"]
        is False
    )
    assert runtime_manifest["oscar_runtime_argv_contract"]["rgb_context_packaged"] is True


def test_wam_generation_step_input_flags_nominal_policy_action_projection_risk(
    tmp_path: Path,
) -> None:
    source_frame = tmp_path / "policy_observation.jpg"
    _write_review_png(source_frame)
    projected_trace = tmp_path / "nominal_policy_action_projected_skeleton_trace.jsonl"
    rows = []
    for index in range(2):
        rows.append(
            {
                "schema_version": "blueprint.g1.nominal_policy_action_projected_skeleton.v1",
                "status": "completed",
                "frame_index": index,
                "projected_landmark_count": 4,
                "landmarks": [
                    {
                        "landmark_id": "left_wrist",
                        "image_projection": {
                            "available": True,
                            "u_px": 18 + index * 3,
                            "v_px": 42,
                        },
                    },
                    {
                        "landmark_id": "left_hand",
                        "image_projection": {
                            "available": True,
                            "u_px": 24 + index * 4,
                            "v_px": 26,
                        },
                    },
                    {
                        "landmark_id": "right_wrist",
                        "image_projection": {
                            "available": True,
                            "u_px": 46 - index * 3,
                            "v_px": 42,
                        },
                    },
                    {
                        "landmark_id": "right_hand",
                        "image_projection": {
                            "available": True,
                            "u_px": 40 - index * 4,
                            "v_px": 26,
                        },
                    },
                ],
                "segments": [
                    {"from": "left_wrist", "to": "left_hand"},
                    {"from": "right_wrist", "to": "right_hand"},
                ],
                "claim_boundary": {
                    "policy_derived_action_conditioning": True,
                    "nominal_kinematic_projection_without_scene_or_wbc_bridge": True,
                    "official_wbc_or_sim_bridge_used": False,
                    "simulated_state_not_physical_robot_sensor_evidence": True,
                },
            }
        )
    projected_trace.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    step_input = tmp_path / "wam_generation_step_0001_input.json"
    step_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_generation_step_input.v1",
                "step_index": 1,
                "source_policy_observation_frame_path": str(source_frame),
                "source_policy_action": {
                    "action_type": "unitree_g1_sonic_latent_action_chunk",
                    "action_chunk": [0.1, 0.2],
                    "policy_action_projected_skeleton_trace_path": str(projected_trace),
                },
                "current_policy_observation": {
                    "task_id": "open_fridge",
                    "target_object_id": "refrigerator",
                    "robot_profile_id": "unitree_g1",
                    "visual_observation": {"camera_id": "head_pov"},
                },
                "requested_output": {
                    "next_observation_frame_path": str(tmp_path / "next.jpg"),
                    "action_conditioned_generation_required": True,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "step-nominal-action-projected-bundle-job",
        wam_rollout_input_manifest=step_input,
        num_frames=2,
        height=64,
        width=64,
        fps=5.0,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    runtime_manifest = _read_json(
        tmp_path
        / "step-nominal-action-projected-bundle-job"
        / "oscar_wam_provider_bundle"
        / "provider_runtime"
        / "wam_provider_runtime_manifest.json"
    )
    diagnostic = runtime_manifest["oscar_input_contract_diagnostic"]
    assert manifest["status"] == "completed"
    assert diagnostic["status"] == "warning_high_risk"
    assert "oscar_contract_projected_skeleton_nominal_action_projection" in diagnostic["warnings"]
    assert "projected_skeleton_nominal_action_projection_high_risk" in diagnostic["high_risk_flags"]
    assert (
        "projected_skeleton_nominal_action_projection_without_scene_or_wbc_bridge"
        in diagnostic["ranking_risk_flags"]
    )


def test_seed_derived_projected_skeleton_trace_is_ranking_risk_not_policy_action(
    tmp_path: Path,
) -> None:
    source_frame = tmp_path / "policy_observation.jpg"
    _write_review_png(source_frame)
    projected_trace = tmp_path / "seed_derived_g1_projected_skeleton_trace.jsonl"
    projected_trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema_version": "blueprint.g1.projected_upper_body_skeleton.v1",
                    "status": "completed",
                    "frame_index": index,
                    "projected_landmark_count": 4,
                    "landmarks": [
                        {
                            "landmark_id": "left_wrist",
                            "image_projection": {
                                "available": True,
                                "u_px": 16 + index * 3,
                                "v_px": 44,
                            },
                        },
                        {
                            "landmark_id": "left_hand",
                            "image_projection": {
                                "available": True,
                                "u_px": 20 + index * 4,
                                "v_px": 30,
                            },
                        },
                        {
                            "landmark_id": "right_wrist",
                            "image_projection": {
                                "available": True,
                                "u_px": 48 - index * 3,
                                "v_px": 44,
                            },
                        },
                        {
                            "landmark_id": "right_hand",
                            "image_projection": {
                                "available": True,
                                "u_px": 44 - index * 4,
                                "v_px": 30,
                            },
                        },
                    ],
                    "segments": [
                        {"from": "left_wrist", "to": "left_hand"},
                        {"from": "right_wrist", "to": "right_hand"},
                    ],
                    "claim_boundary": {
                        "projected_skeleton_trace_derived_from_seed_render_geometry": True,
                        "temporal_rows_are_target_conditioning_from_resolved_affordance_projection": True,
                        "not_a_learned_robot_policy_action": True,
                    },
                }
            )
            for index in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    step_input = tmp_path / "wam_generation_step_0001_input.json"
    step_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_generation_step_input.v1",
                "step_index": 1,
                "source_policy_observation_frame_path": str(source_frame),
                "source_policy_action": {
                    "action_type": "accepted_direct_collision_checked_motion",
                },
                "current_policy_observation": {
                    "task_id": "open_fridge",
                    "target_object_id": "refrigerator",
                    "visual_observation": {
                        "camera_id": "head_pov",
                        "projected_skeleton_trace_path": str(projected_trace),
                    },
                },
                "requested_output": {
                    "next_observation_frame_path": str(tmp_path / "next.jpg"),
                    "action_conditioned_generation_required": True,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "seed-derived-bundle-job",
        wam_rollout_input_manifest=step_input,
        num_frames=4,
        height=64,
        width=64,
        fps=5.0,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    runtime_manifest = _read_json(
        tmp_path
        / "seed-derived-bundle-job"
        / "oscar_wam_provider_bundle"
        / "provider_runtime"
        / "wam_provider_runtime_manifest.json"
    )
    input_package = runtime_manifest["input_package"]
    assert input_package["projected_skeleton_trace"]["used_for_conditioning"] is True
    assert input_package["projected_skeleton_trace"]["seed_geometry_derived"] is True
    assert (
        input_package["claim_boundary"][
            "projected_g1_skeleton_conditioning_is_policy_derived_action"
        ]
        is False
    )
    assert (
        input_package["claim_boundary"][
            "projected_g1_skeleton_conditioning_is_seed_or_target_derived"
        ]
        is True
    )
    contract = runtime_manifest["oscar_input_contract_diagnostic"]
    assert contract["status"] == "warning_high_risk"
    assert "oscar_contract_projected_skeleton_not_policy_derived_action" in contract["warnings"]
    assert "oscar_contract_projected_skeleton_target_conditioned" in contract["warnings"]
    assert (
        "projected_skeleton_not_policy_derived_action_ranking_risk"
        in contract["ranking_risk_flags"]
    )
    assert (
        "projected_skeleton_missing_scene_faithful_policy_action_bridge"
        in contract["ranking_risk_flags"]
    )
    assert contract["policy_ranking_risk_level"] == "high"
    assert contract["policy_ranking_claim_safe"] is False


def test_oscar_wam_provider_bundle_zip_integrity_and_cli_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    rollout_input.write_text(json.dumps({"task_prompts": []}), encoding="utf-8")
    oscar_input = tmp_path / "oscar_input"
    oscar_input.mkdir()
    _write_review_png(oscar_input / "first_frame.png")
    _write_useful_conditioning_mp4(oscar_input / "blueprint_proxy_skeleton_conditioning.mp4")
    real_zipfile = bundle_module.zipfile.ZipFile

    class CorruptReadZip:
        def __enter__(self) -> "CorruptReadZip":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def namelist(self) -> list[str]:
            return ["provider_runtime/run_wam_provider_runtime.sh"]

        def testzip(self) -> str:
            return "provider_runtime/run_wam_provider_runtime.sh"

    def zip_factory(path: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if mode == "w":
            return real_zipfile(path, mode, *args, **kwargs)
        return CorruptReadZip()

    monkeypatch.setattr(bundle_module.zipfile, "ZipFile", zip_factory)
    corrupt = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "corrupt-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_input_dir=oscar_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )
    assert corrupt["status"] == "blocked"
    assert corrupt["blockers"] == ["provider_runtime_bundle_zip_integrity_failed"]

    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "blocked", "blockers": ["cli_blocker"]}

    monkeypatch.setattr(bundle_module, "build_oscar_wam_provider_bundle", fake_build)
    code = bundle_module.main(
        [
            "--job-dir",
            str(tmp_path / "cli-job"),
            "--wam-rollout-input-manifest",
            str(rollout_input),
            "--oscar-input-dir",
            str(oscar_input),
            "--oscar-input-package-manifest",
            str(tmp_path / "package.json"),
            "--oscar-source-url",
            "https://example.com/oscar.git",
            "--oscar-source-ref",
            "abc123",
            "--oscar-hf-repo",
            "example/oscar",
            "--oscar-hf-revision",
            "def456",
            "--allow-experimental-oscar-version",
            "--timeout-seconds",
            "12",
            "--num-steps",
            "7",
            "--guidance",
            "3.5",
            "--seed",
            "99",
            "--num-frames",
            "49",
            "--height",
            "352",
            "--width",
            "640",
            "--fps",
            "12.5",
            "--bundle-filename",
            "custom.zip",
        ]
    )
    output = capsys.readouterr().out
    assert code == 1
    assert "[oscar-wam-provider-bundle] blockers=cli_blocker" in output
    assert captured["oscar_source_url"] == "https://example.com/oscar.git"
    assert captured["oscar_source_ref"] == "abc123"
    assert captured["oscar_hf_repo"] == "example/oscar"
    assert captured["oscar_hf_revision"] == "def456"
    assert captured["allow_experimental_oscar_version"] is True
    assert captured["timeout_seconds"] == 12
    assert captured["num_steps"] == 7
    assert captured["guidance"] == 3.5
    assert captured["seed"] == 99
    assert captured["num_frames"] == 49
    assert captured["height"] == 352
    assert captured["width"] == 640
    assert captured["fps"] == 12.5
    assert captured["bundle_filename"] == "custom.zip"
