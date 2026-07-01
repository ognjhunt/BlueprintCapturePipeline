from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import oscar_wam_provider_command_adapter as adapter
from blueprint_pipeline import wam_compute_providers as compute_providers
from blueprint_pipeline.oscar_official_release import (
    OFFICIAL_OSCAR_WAM_IMAGE_REF,
    official_release_contract,
)


_PROVIDER_ENV_VARS = (
    "BLUEPRINT_WAM_ROLLOUT_INPUT",
    "BLUEPRINT_WAM_ROLLOUT_OUTPUT",
    "BLUEPRINT_OSCAR_WAM_CHECKPOINT",
    "BLUEPRINT_WAM_MODEL_CHECKPOINT",
    adapter.VAST_WAM_PUBLIC_IMAGE_ENV,
    adapter.RUNPOD_WAM_PUBLIC_IMAGE_ENV,
    adapter.VAST_WAM_MIN_GPU_RAM_MB_ENV,
    adapter.VAST_WAM_EXCLUDED_MACHINE_ID_ENV,
    adapter.VAST_WAM_ALLOWED_MACHINE_ID_ENV,
    adapter.ALLOW_VAST_PROVIDER_LAUNCH_ENV,
    adapter.ALLOW_UNPINNED_OSCAR_WAM_IMAGE_ENV,
    adapter.OSCAR_WAM_COMPUTE_PROVIDER_ENV,
    compute_providers.PROVIDER_ORDER_ENV,
    compute_providers.DEEPINFRA_API_GATE_ENV,
    compute_providers.DEEPINFRA_API_KEY_ENV,
    compute_providers.DEEPINFRA_API_KEY_FILE_ENV,
)


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_test_video(path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    assert writer.isOpened()
    for index in range(4):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :32] = (255, 40 + index * 20, 0)
        frame[:, 32:] = (0, 160, 220 - index * 10)
        writer.write(frame)
    writer.release()


def _write_flat_dark_video(path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 480))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.full((480, 640, 3), 30, dtype=np.uint8))
    writer.release()


def _write_provider_zip(
    path: Path,
    *,
    valid_video: bool = True,
    flat_dark_video: bool = False,
    runtime_model_truth: bool = True,
    include_rollout_context: bool = True,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    video_path = path.parent / "oscar_generated_rollout_source.mp4"
    if valid_video and flat_dark_video:
        _write_flat_dark_video(video_path)
    elif valid_video:
        _write_test_video(video_path)
    with zipfile.ZipFile(path, "w") as archive:
        if valid_video:
            archive.write(video_path, "oscar_generated_rollout.mp4")
        else:
            archive.writestr("oscar_generated_rollout.mp4", b"mp4-placeholder")
        rollout = {
            "rollout_id": "oscar_wam_rollout_0001",
            "policy_id": "oscar_wam_provider_runtime",
            "generated_video_path": "/workspace/runtime_output/oscar_generated_rollout.mp4",
        }
        if include_rollout_context:
            rollout["scenario_eval_run_id"] = "run_1"
        archive.writestr(
            "wam_provider_output.json",
            json.dumps(
                {
                    "schema_version": "oscar_wam_command_adapter.v1",
                    "status": "completed",
                    "adapter_id": "oscar_wam_provider_runtime",
                    "rollouts": [rollout],
                    "blockers": [],
                    "fresh_model_run_claimed": True,
                    "fresh_provider_model_run_claimed": True,
                    "fresh_model_command_executed_this_invocation": True,
                    "fresh_provider_launch_attempted": True,
                }
            ),
        )
        runtime_result = {
            "status": "completed",
            "runtime": "oscar_wam_provider_runtime",
            "model_candidate": "oscar_wam",
            "runtime_settings": {
                "num_frames": 8,
                "height": 480,
                "width": 640,
                "fps": 15.0,
                "num_steps": 35,
                "guidance": 4.5,
                "seed": 42,
            },
            "oscar_runtime_argv_contract": {
                "rgb_context_packaged": True,
                "rgb_video_arg_expected": True,
                "skeleton_video_arg": "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4",
            },
            "oscar_input_contract_diagnostic": {
                "schema_version": "oscar_wam_runtime_input_contract_diagnostic.v1",
                "status": "ready",
                "rgb_context": {
                    "used_for_oscar_rgb_latent_context": True,
                    "rgb_context_mode": "single_frame_repeat",
                },
                "warnings": ["oscar_contract_rgb_context_single_frame_repeat"],
                "blockers": [],
                "claim_boundary": {"diagnostic_is_no_spend": True},
            },
            "input_signal_summary": {
                "projected_skeleton_used": True,
                "projected_skeleton_projectable_row_count": 8,
                "rgb_context_mode": "single_frame_repeat",
                "input_contract_status": "ready",
                "diagnostic_only_not_success_label": True,
            },
        }
        if runtime_model_truth:
            runtime_result.update(
                {
                    "learned_wam_model_ran": True,
                    "official_oscar_release": official_release_contract(),
                    "truth_boundary": {
                        "generated_video_is_model_output": True,
                        "official_oscar_source_and_checkpoint_pinned": True,
                        "generated_world_rank_fidelity_result_proven": False,
                        "generated_world_policy_evaluation_scope_proven": False,
                    },
                }
            )
        archive.writestr("wam_runtime_result.json", json.dumps(runtime_result))
    return path


def test_provider_command_adapter_blocks_without_required_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_env(monkeypatch)
    output = tmp_path / "out.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.delenv("BLUEPRINT_WAM_ROLLOUT_INPUT", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)

    payload = adapter.run(["--mode", "auto", "--work-dir", str(tmp_path / "work")])

    assert payload["status"] == "blocked"
    assert "blocked_missing_BLUEPRINT_WAM_ROLLOUT_INPUT" in payload["blockers"]
    assert "blocked_missing_oscar_checkpoint_contract" in payload["blockers"]
    assert json.loads(output.read_text(encoding="utf-8"))["raw_credentials_written_to_artifacts"] is False


def test_provider_command_adapter_imports_completed_provider_output_zip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"schema_version": "wam_rollout_input_manifest.v1"})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    provider_job = tmp_path / "provider-job"
    _write_provider_zip(provider_job / "vast_provider_runtime_output.zip")
    output = tmp_path / "wam_provider_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(checkpoint))

    payload = adapter.run(
        [
            "--mode",
            "replay-existing-provider-output",
            "--completed-provider-job-dir",
            str(provider_job),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )

    assert payload["status"] == "completed"
    assert payload["mode"] == "replay_existing_provider_output"
    assert payload["provider_output_zip_imported"] is True
    assert payload["provider_output_replayed"] is True
    assert payload["replay_source_completed_provider_job_dir_name"] == provider_job.name
    assert payload["replay_source_completed_provider_job_dir_path_omitted"] is True
    assert payload["provider_output_zip_name"] == "vast_provider_runtime_output.zip"
    assert payload["provider_output_zip_path_omitted"] is True
    assert payload["provider_video_extraction_dir_path_omitted"] is True
    assert payload["fresh_model_run_claimed"] is False
    assert payload["fresh_provider_model_run_claimed"] is False
    assert payload["fresh_model_command_executed_this_invocation"] is False
    assert payload["fresh_provider_launch_attempted"] is False
    assert payload["imported_provider_payload_truth_claims"]["fresh_model_run_claimed"] is True
    rollout = payload["rollouts"][0]
    assert rollout["provider_original_generated_video_name"] == "oscar_generated_rollout.mp4"
    assert rollout["provider_original_generated_video_path_omitted"] is True
    assert Path(rollout["generated_video_path"]).is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["rollouts"][0]["generated_video_path"] == rollout[
        "generated_video_path"
    ]


def test_provider_command_adapter_imports_completed_runpod_provider_output_zip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"schema_version": "wam_rollout_input_manifest.v1"})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    provider_job = tmp_path / "provider-job"
    _write_provider_zip(provider_job / "runpod_provider_runtime_output.zip")
    output = tmp_path / "wam_provider_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(checkpoint))

    payload = adapter.run(
        [
            "--mode",
            "replay-existing-provider-output",
            "--completed-provider-job-dir",
            str(provider_job),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )

    assert payload["status"] == "completed"
    assert payload["mode"] == "replay_existing_provider_output"
    assert payload["provider_output_zip_name"] == "runpod_provider_runtime_output.zip"
    assert payload["provider_output_replayed"] is True
    assert payload["fresh_provider_launch_attempted"] is False


def test_provider_command_adapter_backfills_rollout_context_from_input_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(
        rollout_input,
        {
            "schema_version": "wam_rollout_input_manifest.v1",
            "task_prompts": [
                {
                    "scenario_eval_run_id": "unitree_run_1",
                    "task_id": "contact_or_push_light_object",
                    "spawn_id": "doorway",
                    "task_prompt": "Touch the object.",
                }
            ],
            "wam_input_videos": [
                {
                    "scenario_eval_run_id": "unitree_run_1",
                    "camera": "head_pov",
                    "path": "/source/head_pov.mp4",
                }
            ],
        },
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    provider_job = tmp_path / "provider-job"
    _write_provider_zip(
        provider_job / "vast_provider_runtime_output.zip",
        include_rollout_context=False,
    )
    output = tmp_path / "wam_provider_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(checkpoint))

    payload = adapter.run(
        [
            "--mode",
            "replay-existing-provider-output",
            "--completed-provider-job-dir",
            str(provider_job),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )

    rollout = payload["rollouts"][0]
    assert rollout["scenario_eval_run_id"] == "unitree_run_1"
    assert rollout["task_id"] == "contact_or_push_light_object"
    assert rollout["spawn_id"] == "doorway"
    assert rollout["task_prompt"] == "Touch the object."
    assert rollout["source_wam_input_camera"] == "head_pov"
    assert rollout["provider_rollout_context_backfilled_from_input_manifest"] is True


def test_provider_command_adapter_blocks_invalid_provider_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"schema_version": "wam_rollout_input_manifest.v1"})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    provider_job = tmp_path / "provider-job"
    _write_provider_zip(provider_job / "vast_provider_runtime_output.zip", valid_video=False)
    output = tmp_path / "wam_provider_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(checkpoint))

    payload = adapter.run(
        [
            "--mode",
            "replay-existing-provider-output",
            "--completed-provider-job-dir",
            str(provider_job),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )

    assert payload["status"] == "blocked"
    assert payload["rollouts"] == []
    assert "provider_generated_video_not_reviewable" in payload["blockers"]
    assert payload["provider_output_replayed"] is True
    assert payload["fresh_model_run_claimed"] is False


def test_provider_command_adapter_vast_mode_is_gated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"schema_version": "wam_rollout_input_manifest.v1"})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "wam_provider_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(checkpoint))
    monkeypatch.delenv(adapter.ALLOW_VAST_PROVIDER_LAUNCH_ENV, raising=False)

    payload = adapter.run(["--mode", "vast-provider", "--work-dir", str(tmp_path / "work")])

    assert payload["status"] == "blocked"
    assert f"missing_env_{adapter.ALLOW_VAST_PROVIDER_LAUNCH_ENV}" in payload["blockers"]
    assert "missing_cli_paid_wam_compute_provider_launch_flag" in payload["blockers"]


def test_provider_command_adapter_runpod_provider_no_spend_plan_is_gated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"schema_version": "wam_rollout_input_manifest.v1"})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "wam_provider_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(checkpoint))

    payload = adapter.run(
        [
            "--mode",
            "auto",
            "--provider",
            "runpod",
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )

    assert payload["status"] == "blocked"
    assert payload["mode"] == "runpod_provider"
    assert "missing_cli_paid_wam_compute_provider_launch_flag" in payload["blockers"]
    assert payload["fresh_provider_launch_attempted"] is False


def test_provider_command_adapter_launches_and_imports_vast_provider_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_env(monkeypatch)
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"schema_version": "wam_rollout_input_manifest.v1"})
    output = tmp_path / "wam_provider_output.json"
    work_dir = tmp_path / "work"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    monkeypatch.delenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", raising=False)
    monkeypatch.setenv(adapter.ALLOW_VAST_PROVIDER_LAUNCH_ENV, "true")
    monkeypatch.setenv(
        adapter.OSCAR_WAM_GPU_IMAGE_REF_ENV,
        OFFICIAL_OSCAR_WAM_IMAGE_REF,
    )
    monkeypatch.setenv(
        adapter.VAST_WAM_PUBLIC_IMAGE_ENV,
        OFFICIAL_OSCAR_WAM_IMAGE_REF,
    )
    monkeypatch.setenv(adapter.VAST_WAM_MIN_GPU_RAM_MB_ENV, "48000")
    monkeypatch.setenv(adapter.VAST_WAM_EXCLUDED_MACHINE_ID_ENV, "134862, 42, bad, 134862")
    monkeypatch.setenv(adapter.VAST_WAM_ALLOWED_MACHINE_ID_ENV, "16571, bad, 16571")
    monkeypatch.setenv(adapter.VAST_WAM_MIN_RELIABILITY_ENV, "0.99")
    monkeypatch.setenv(adapter.VAST_WAM_REQUIRE_DIRECT_PORT_ENV, "true")
    monkeypatch.setenv(adapter.VAST_WAM_PREFERRED_GPU_KEYWORDS_ENV, "RTX A6000,L40S,A100")
    monkeypatch.setenv(adapter.VAST_WAM_PREFERRED_GEOLOCATION_REGEX_ENV, "california|oregon")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_NUM_FRAMES", "49")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_HEIGHT", "480")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_WIDTH", "640")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_FPS", "15")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_NUM_STEPS", "41")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_GUIDANCE", "4.5")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_SEED", "123")
    monkeypatch.setenv(adapter.VAST_WAM_POLL_MAX_WAIT_SECONDS_ENV, "900")
    captured_create: dict[str, Any] = {}
    captured_bundle: dict[str, Any] = {}
    captured_poll: dict[str, Any] = {}

    def fake_build_bundle(**kwargs: Any) -> dict[str, Any]:
        captured_bundle.update(kwargs)
        bundle = Path(kwargs["job_dir"]) / "provider_bundle.zip"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_bytes(b"bundle")
        return {"status": "completed", "bundle_path": str(bundle), "blockers": []}

    def fake_create(**kwargs: Any) -> dict[str, Any]:
        captured_create.update(kwargs)
        return {
            "status": "instance_created",
            "job_dir": str(kwargs["job_dir"]),
            "blockers": [],
        }

    def fake_poll(**kwargs: Any) -> dict[str, Any]:
        captured_poll.update(kwargs)
        provider_job = Path(kwargs["job_dir"])
        _write_provider_zip(provider_job / "vast_provider_runtime_output.zip")
        return {"status": "completed", "job_dir": str(provider_job), "blockers": []}

    monkeypatch.setattr(adapter, "build_oscar_wam_provider_bundle", fake_build_bundle)
    monkeypatch.setattr(compute_providers, "create_async_vast_wam_run", fake_create)
    monkeypatch.setattr(compute_providers, "poll_async_vast_wam_run", fake_poll)

    payload = adapter.run(
        [
            "--mode",
            "vast-provider",
            "--allow-paid-vast-launch",
            "--work-dir",
            str(work_dir),
        ]
    )

    assert payload["status"] == "completed"
    assert payload["mode"] == "vast_provider"
    assert payload["provider_output_zip_imported"] is True
    assert payload["provider_output_replayed"] is False
    assert payload["provider_output_imported_from_current_provider_run"] is True
    assert payload["fresh_provider_launch_attempted"] is True
    assert payload["fresh_model_run_claimed"] is True
    assert payload["generated_rollout_visual_smoke_status"] == "passed_visual_quality_smoke"
    assert payload["generated_rollout_visually_useful_for_success_review"] is False
    assert (
        payload["generated_rollout_review_usefulness_status"]
        == "not_reviewable_for_task_success"
    )
    assert (
        "generated_rollout_video_resolution_too_low_for_task_success_review"
        in payload["generated_rollout_review_usefulness_blockers"]
    )
    assert (
        "generated_rollout_video_fps_too_low_for_task_success_review"
        in payload["generated_rollout_review_usefulness_blockers"]
    )
    assert (
        "generated_rollout_video_too_short_for_task_success_review"
        in payload["generated_rollout_review_usefulness_blockers"]
    )
    assert payload["details"]["vast_provider_job_dir"] == str(work_dir / "vast_provider_run")
    assert payload["details"]["wam_compute_provider"] == "vast"
    assert Path(payload["rollouts"][0]["generated_video_path"]).is_file()
    assert (
        captured_create["public_image"]
        == OFFICIAL_OSCAR_WAM_IMAGE_REF
    )
    assert captured_create["min_gpu_ram_mb"] == 48000
    assert captured_create["excluded_machine_ids"] == [134862, 42]
    assert captured_create["allowed_machine_ids"] == [16571]
    assert captured_create["min_reliability"] == 0.99
    assert captured_create["require_direct_port"] is True
    assert captured_create["preferred_gpu_keywords"] == ["RTX A6000", "L40S", "A100"]
    assert captured_create["preferred_geolocation_regex"] == "california|oregon"
    assert captured_create["prefer_isaac_rt"] is False
    assert captured_bundle["num_frames"] == 49
    assert captured_bundle["height"] == 480
    assert captured_bundle["width"] == 640
    assert captured_bundle["fps"] == 15.0
    assert captured_bundle["num_steps"] == 41
    assert captured_bundle["guidance"] == 4.5
    assert captured_bundle["seed"] == 123
    assert captured_poll["max_wait_seconds"] == 900
    assert captured_poll["teardown"] is True


def test_provider_command_adapter_launches_and_imports_runpod_provider_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_env(monkeypatch)
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"schema_version": "wam_rollout_input_manifest.v1"})
    output = tmp_path / "wam_provider_output.json"
    work_dir = tmp_path / "work"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    monkeypatch.delenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", raising=False)
    monkeypatch.setenv(
        adapter.RUNPOD_WAM_PUBLIC_IMAGE_ENV,
        OFFICIAL_OSCAR_WAM_IMAGE_REF,
    )
    captured_create: dict[str, Any] = {}
    captured_bundle: dict[str, Any] = {}
    captured_poll: dict[str, Any] = {}

    def fake_build_bundle(**kwargs: Any) -> dict[str, Any]:
        captured_bundle.update(kwargs)
        bundle = Path(kwargs["job_dir"]) / "provider_bundle.zip"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_bytes(b"bundle")
        return {"status": "completed", "bundle_path": str(bundle), "blockers": []}

    def fake_create(**kwargs: Any) -> dict[str, Any]:
        captured_create.update(kwargs)
        return {
            "status": "pod_created",
            "pod_id": "pod-123",
            "job_dir": str(kwargs["job_dir"]),
            "output_path": str(kwargs["output_path"]),
            "blockers": [],
        }

    def fake_poll(**kwargs: Any) -> dict[str, Any]:
        captured_poll.update(kwargs)
        provider_job = Path(kwargs["job_dir"])
        _write_provider_zip(provider_job / "runpod_provider_runtime_output.zip")
        return {
            "status": "completed",
            "pod_id": "pod-123",
            "provider_runtime_output_zip_path": str(
                provider_job / "runpod_provider_runtime_output.zip"
            ),
            "provider_command_status": "completed",
            "provider_command_blockers": [],
            "output_zip_present": True,
            "runtime_result_status": "completed",
            "runtime_result_blockers": [],
            "mp4_count": 1,
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        }

    monkeypatch.setattr(adapter, "build_oscar_wam_provider_bundle", fake_build_bundle)
    monkeypatch.setattr(compute_providers, "create_runpod_wam_async_run", fake_create)
    monkeypatch.setattr(compute_providers, "poll_runpod_wam_async_run", fake_poll)

    payload = adapter.run(
        [
            "--mode",
            "auto",
            "--provider",
            "runpod",
            "--allow-paid-runpod-launch",
            "--work-dir",
            str(work_dir),
        ]
    )

    assert payload["status"] == "completed"
    assert payload["mode"] == "runpod_provider"
    assert payload["provider_output_zip_imported"] is True
    assert payload["provider_output_replayed"] is False
    assert payload["provider_output_imported_from_current_provider_run"] is True
    assert payload["fresh_provider_launch_attempted"] is True
    assert payload["fresh_model_run_claimed"] is True
    assert payload["details"]["runpod_provider_job_dir"] == str(work_dir / "runpod_provider_run")
    assert payload["details"]["wam_compute_provider"] == "runpod"
    assert captured_create["allow_paid_runpod_launch"] is True
    assert captured_create["image_name"] == OFFICIAL_OSCAR_WAM_IMAGE_REF
    assert captured_create["container_disk_gb"] == 100
    assert captured_create["volume_gb"] == 30
    assert captured_create["min_vcpu_per_gpu"] == 8
    assert captured_create["min_ram_per_gpu"] == 40
    assert captured_bundle["wam_rollout_input_manifest"] == rollout_input.resolve()
    assert captured_poll["teardown"] is True


def test_provider_command_adapter_launches_and_imports_deepinfra_provider_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_env(monkeypatch)
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"schema_version": "wam_rollout_input_manifest.v1"})
    output = tmp_path / "wam_provider_output.json"
    work_dir = tmp_path / "work"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    monkeypatch.delenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", raising=False)
    monkeypatch.setenv(compute_providers.DEEPINFRA_API_GATE_ENV, "1")
    captured_bundle: dict[str, Any] = {}
    captured_compute: dict[str, Any] = {}

    def fake_build_bundle(**kwargs: Any) -> dict[str, Any]:
        captured_bundle.update(kwargs)
        bundle = Path(kwargs["job_dir"]) / "provider_bundle.zip"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_bytes(b"bundle")
        return {"status": "completed", "bundle_path": str(bundle), "blockers": []}

    def fake_run_wam_compute_job(**kwargs: Any) -> compute_providers.WamComputeRunResult:
        captured_compute.update(kwargs)
        provider_job = Path(kwargs["job_dir"]) / "deepinfra_provider_run"
        _write_provider_zip(provider_job / "deepinfra_provider_runtime_output.zip")
        return compute_providers.WamComputeRunResult(
            provider="deepinfra",
            status="completed",
            provider_command_status="completed",
            output_zip_path=str(provider_job / "deepinfra_provider_runtime_output.zip"),
            output_zip_present=True,
            mp4_count=1,
            runtime_result_status="completed",
            runtime_result_blockers=[],
            budget_ledger_path=str(
                provider_job / "deepinfra_cosmos3_cost_control_ledger.json"
            ),
            teardown_status="not_required",
            teardown_performed=False,
            continuing_spend_from_this_run=False,
            output_availability="available",
        )

    monkeypatch.setattr(adapter, "build_oscar_wam_provider_bundle", fake_build_bundle)
    monkeypatch.setattr(adapter, "run_wam_compute_job", fake_run_wam_compute_job)

    payload = adapter.run(
        [
            "--mode",
            "auto",
            "--provider",
            "deepinfra",
            "--allow-paid-provider-launch",
            "--work-dir",
            str(work_dir),
        ]
    )

    assert payload["status"] == "completed"
    assert payload["mode"] == "deepinfra_provider"
    assert payload["provider_output_zip_imported"] is True
    assert payload["provider_output_imported_from_current_provider_run"] is True
    assert payload["fresh_provider_launch_attempted"] is True
    assert payload["fresh_model_run_claimed"] is True
    assert payload["details"]["wam_compute_provider"] == "deepinfra"
    assert payload["details"]["deepinfra_provider_job_dir"] == str(
        work_dir / "deepinfra_provider_run"
    )
    assert payload["details"]["deepinfra_request_manifest_path"].endswith(
        "deepinfra_cosmos3_request_manifest.json"
    )
    assert captured_compute["provider_order"] == ["deepinfra"]
    assert captured_compute["allow_paid_launch"] is True
    assert captured_compute["spec"].image == "deepinfra/nvidia/Cosmos3-Nano"
    assert captured_bundle["wam_rollout_input_manifest"] == rollout_input.resolve()


def test_provider_command_adapter_blocks_failed_rollout_visual_smoke(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_env(monkeypatch)
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"schema_version": "wam_rollout_input_manifest.v1"})
    output = tmp_path / "wam_provider_output.json"
    work_dir = tmp_path / "work"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    monkeypatch.delenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", raising=False)

    def fake_build_bundle(**kwargs: Any) -> dict[str, Any]:
        bundle = Path(kwargs["job_dir"]) / "provider_bundle.zip"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_bytes(b"bundle")
        return {"status": "completed", "bundle_path": str(bundle), "blockers": []}

    def fake_create(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pod_created",
            "pod_id": "pod-123",
            "job_dir": str(kwargs["job_dir"]),
            "output_path": str(kwargs["output_path"]),
            "blockers": [],
        }

    def fake_poll(**kwargs: Any) -> dict[str, Any]:
        provider_job = Path(kwargs["job_dir"])
        _write_provider_zip(
            provider_job / "runpod_provider_runtime_output.zip",
            flat_dark_video=True,
        )
        return {
            "status": "completed",
            "pod_id": "pod-123",
            "provider_runtime_output_zip_path": str(
                provider_job / "runpod_provider_runtime_output.zip"
            ),
            "provider_command_status": "completed",
            "provider_command_blockers": [],
            "output_zip_present": True,
            "runtime_result_status": "completed",
            "runtime_result_blockers": [],
            "mp4_count": 1,
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        }

    monkeypatch.setattr(adapter, "build_oscar_wam_provider_bundle", fake_build_bundle)
    monkeypatch.setattr(compute_providers, "create_runpod_wam_async_run", fake_create)
    monkeypatch.setattr(compute_providers, "poll_runpod_wam_async_run", fake_poll)

    payload = adapter.run(
        [
            "--mode",
            "auto",
            "--provider",
            "runpod",
            "--allow-paid-runpod-launch",
            "--work-dir",
            str(work_dir),
        ]
    )

    assert payload["status"] == "blocked"
    assert payload["generated_rollout_visual_smoke_status"] == "failed_visual_quality_smoke"
    assert "provider_generated_rollout_visual_smoke_failed" in payload["blockers"]
    assert payload["fresh_model_run_claimed"] is False


def test_provider_command_adapter_current_vast_run_requires_runtime_model_truth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"schema_version": "wam_rollout_input_manifest.v1"})
    output = tmp_path / "wam_provider_output.json"
    work_dir = tmp_path / "work"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    monkeypatch.setenv(adapter.ALLOW_VAST_PROVIDER_LAUNCH_ENV, "true")

    bundle = work_dir / "bundle" / "provider_bundle.zip"

    def fake_build_bundle(**_kwargs: Any) -> dict[str, Any]:
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_bytes(b"bundle")
        return {"status": "completed", "bundle_path": str(bundle), "blockers": []}

    def fake_create(**kwargs: Any) -> dict[str, Any]:
        return {"status": "instance_created", "job_dir": str(kwargs["job_dir"]), "blockers": []}

    def fake_poll(**kwargs: Any) -> dict[str, Any]:
        provider_job = Path(kwargs["job_dir"])
        _write_provider_zip(
            provider_job / "vast_provider_runtime_output.zip",
            runtime_model_truth=False,
        )
        return {"status": "completed", "job_dir": str(provider_job), "blockers": []}

    monkeypatch.setattr(adapter, "build_oscar_wam_provider_bundle", fake_build_bundle)
    monkeypatch.setattr(compute_providers, "create_async_vast_wam_run", fake_create)
    monkeypatch.setattr(compute_providers, "poll_async_vast_wam_run", fake_poll)

    payload = adapter.run(
        [
            "--mode",
            "vast-provider",
            "--allow-paid-vast-launch",
            "--work-dir",
            str(work_dir),
        ]
    )

    assert payload["status"] == "completed"
    assert payload["provider_output_imported_from_current_provider_run"] is True
    assert payload["provider_runtime_result_present"] is True
    assert payload["provider_runtime_result_proves_model_output"] is False
    assert payload["provider_learned_wam_model_ran"] is False
    assert payload["provider_generated_video_is_model_output"] is False
    assert payload["provider_runtime_settings"]["guidance"] == 4.5
    assert payload["provider_oscar_runtime_argv_contract"]["rgb_video_arg_expected"] is True
    assert payload["provider_oscar_input_contract_diagnostic"]["status"] == "ready"
    assert (
        payload["provider_oscar_input_contract_diagnostic"]["rgb_context"][
            "rgb_context_mode"
        ]
        == "single_frame_repeat"
    )
    assert payload["provider_input_signal_summary"]["projected_skeleton_used"] is True
    assert payload["provider_input_signal_summary"]["input_contract_status"] == "ready"
    assert payload["fresh_model_run_claimed"] is False
    assert payload["fresh_provider_model_run_claimed"] is False
    assert payload["fresh_model_command_executed_this_invocation"] is False


def test_provider_command_adapter_scrubs_object_store_transport_url_files(
    tmp_path: Path,
) -> None:
    paths = {
        "provider_bundle_url_file": tmp_path / "provider_bundle_url.txt",
        "provider_output_put_url_file": tmp_path / "provider_output_put_url.txt",
        "provider_output_get_url_file": tmp_path / "provider_output_get_url.txt",
    }
    for path in paths.values():
        path.write_text(
            "https://object.example/file.zip?X-Amz-Credential=AKIAEXAMPLE"
            "&X-Amz-Signature=secret\n",
            encoding="utf-8",
        )
        path.chmod(0o600)
    manifest = {key: {"path": str(path)} for key, path in paths.items()}

    scrub = adapter._scrub_object_store_provider_url_files(manifest)

    assert [item["scrubbed"] for item in scrub] == [True, True, True]
    for path in paths.values():
        assert path.read_text(encoding="utf-8").strip() == (
            adapter.REDACTED_PROVIDER_TRANSPORT_URL
        )
        assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_provider_command_adapter_auto_prefers_replay_without_paid_launch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"schema_version": "wam_rollout_input_manifest.v1"})
    provider_job = tmp_path / "provider-job"
    _write_provider_zip(provider_job / "vast_provider_runtime_output.zip")
    output = tmp_path / "wam_provider_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    monkeypatch.delenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", raising=False)
    monkeypatch.setenv(adapter.COMPLETED_PROVIDER_JOB_ENV, str(provider_job))
    monkeypatch.setenv(adapter.ALLOW_VAST_PROVIDER_LAUNCH_ENV, "true")

    def fail_create(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("auto replay should not create a fresh provider run")

    def fail_poll(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("auto replay should not poll a fresh provider run")

    monkeypatch.setattr(compute_providers, "create_async_vast_wam_run", fail_create)
    monkeypatch.setattr(compute_providers, "poll_async_vast_wam_run", fail_poll)

    payload = adapter.run(["--mode", "auto", "--work-dir", str(tmp_path / "work")])

    assert payload["status"] == "completed"
    assert payload["mode"] == "replay_existing_provider_output"
    assert payload["provider_output_replayed"] is True
    assert payload["fresh_provider_launch_attempted"] is False
    assert payload["fresh_model_run_claimed"] is False
