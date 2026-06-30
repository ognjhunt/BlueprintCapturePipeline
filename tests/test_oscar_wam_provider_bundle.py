from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from blueprint_pipeline import oscar_wam_provider_bundle as bundle_module  # noqa: E402
from blueprint_pipeline.oscar_wam_provider_bundle import build_oscar_wam_provider_bundle  # noqa: E402


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


def _write_projected_skeleton_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "schema_version": "blueprint.mujoco_g1.projected_upper_body_skeleton.v1",
            "status": "completed",
            "episode_id": "episode_0001",
            "projected_landmark_count": 2,
            "landmarks": [
                {
                    "landmark_id": "left_hand",
                    "image_projection": {
                        "available": True,
                        "u_px": 24,
                        "v_px": 26,
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
            "segments": [{"from": "left_hand", "to": "right_hand"}],
        },
        {
            "schema_version": "blueprint.mujoco_g1.projected_upper_body_skeleton.v1",
            "status": "completed",
            "episode_id": "episode_0001",
            "projected_landmark_count": 2,
            "landmarks": [
                {
                    "landmark_id": "left_hand",
                    "image_projection": {
                        "available": True,
                        "u_px": 27,
                        "v_px": 23,
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
            "segments": [{"from": "left_hand", "to": "right_hand"}],
        },
    ]
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
                    "row_count": 1,
                    "projectable_row_count": 1,
                },
                "skeleton_video": {
                    "conditioning_mode": "projected_g1_skeleton",
                    "projected_g1_skeleton_rendered": True,
                    "skeleton_stream_separate_from_rgb": True,
                    "skeleton_stream_texture_free": True,
                    "skeleton_stream_image_aligned_to_rgb": True,
                    "first_rgb_frame_anchors_scene_and_robot_appearance": True,
                    "projected_g1_skeleton_landmark_draw_count": 2,
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
    assert runtime_input_package["oscar_dual_stream_input_contract"][
        "first_rgb_frame_path"
    ] == "provider_runtime/oscar_input/first_frame.png"
    assert runtime_input_package["oscar_dual_stream_input_contract"][
        "skeleton_video_path"
    ] == "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4"
    assert runtime_input_package["oscar_dual_stream_input_contract"][
        "separate_2d_skeleton_stream"
    ] is True
    assert runtime_input_package["oscar_dual_stream_input_contract"][
        "skeleton_stream_texture_free"
    ] is True
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
        runtime_input_package["claim_boundary"][
            "separate_2d_skeleton_stream_aligned_to_rgb"
        ]
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
    assert "rgb_context.mp4" in runner_text
    assert "--rgb-video" in runner_text
    assert 'inference_checkpoint_path = (\n            checkpoint_path / "model"' not in runner_text


def test_build_oscar_wam_provider_bundle_records_official_case_smoke_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_SMOKE", "agibot_465")
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
    assert "oscar_input_skeleton_conditioning_video_not_visually_useful" not in manifest[
        "blockers"
    ]


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
    assert (
        input_package["oscar_rgb_context_runtime_contract"]["rgb_context_packaged"]
        is True
    )
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
    assert "oscar_contract_single_frame_repeat_without_projected_skeleton" in contract[
        "warnings"
    ]
    assert "rgb_context_single_frame_repeat_autoregressive_risk" in contract[
        "autoregressive_risk_flags"
    ]
    assert (
        "policy_action_proxy_without_projected_skeleton_autoregressive_risk"
        in contract["autoregressive_risk_flags"]
    )
    assert "single_frame_repeat_without_projected_skeleton_high_risk" in contract[
        "high_risk_flags"
    ]
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
    assert "provider_runtime/oscar_input/rgb_context.mp4" in names
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
    assert input_package["rgb_video"]["path"] == "provider_runtime/oscar_input/rgb_context.mp4"
    assert input_package["rgb_video"]["used_for_oscar_rgb_latent_context"] is True
    contract = runtime_manifest["oscar_input_contract_diagnostic"]
    assert manifest["input_package_contract_diagnostic"]["status"] == "ready"
    assert contract["status"] == "ready"
    assert contract["projected_skeleton_trace"]["used_for_conditioning"] is True
    assert contract["rgb_context"]["used_for_oscar_rgb_latent_context"] is True
    assert contract["rgb_context"]["rgb_context_mode"] == "single_frame_repeat"
    assert "oscar_contract_rgb_context_single_frame_repeat" in contract["warnings"]
    assert "oscar_contract_guidance_high_for_contract_debug" in contract["warnings"]
    assert "rgb_context_single_frame_repeat_autoregressive_risk" in contract[
        "autoregressive_risk_flags"
    ]
    assert "guidance_high_autoregressive_debug_risk" in contract[
        "autoregressive_risk_flags"
    ]
    assert contract["autoregressive_risk_level"] == "monitor"
    assert (
        input_package["oscar_rgb_context_runtime_contract"]["projected_g1_rgb_context_enabled"]
        is True
    )
    assert (
        input_package["oscar_rgb_context_runtime_contract"][
            "projected_g1_skeleton_conditioning_suppresses_rgb_context"
        ]
        is False
    )
    assert input_package["claim_boundary"]["policy_action_conditioning_proxy_video_used"] is False


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
                    "projected_landmark_count": 2,
                    "landmarks": [
                        {
                            "landmark_id": "left_hand",
                            "image_projection": {
                                "available": True,
                                "u_px": 20 + index * 4,
                                "v_px": 30,
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
                    "segments": [{"from": "left_hand", "to": "right_hand"}],
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
    assert contract["status"] == "ready"
    assert "oscar_contract_projected_skeleton_not_policy_derived_action" in contract[
        "warnings"
    ]
    assert "oscar_contract_projected_skeleton_target_conditioned" in contract["warnings"]
    assert "projected_skeleton_not_policy_derived_action_ranking_risk" in contract[
        "ranking_risk_flags"
    ]
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
            "--oscar-hf-repo",
            "example/oscar",
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
    assert captured["oscar_hf_repo"] == "example/oscar"
    assert captured["timeout_seconds"] == 12
    assert captured["num_steps"] == 7
    assert captured["guidance"] == 3.5
    assert captured["seed"] == 99
    assert captured["num_frames"] == 49
    assert captured["height"] == 352
    assert captured["width"] == 640
    assert captured["fps"] == 12.5
    assert captured["bundle_filename"] == "custom.zip"
