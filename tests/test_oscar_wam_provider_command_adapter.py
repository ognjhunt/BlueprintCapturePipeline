from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import oscar_wam_provider_command_adapter as adapter


_PROVIDER_ENV_VARS = (
    "BLUEPRINT_WAM_ROLLOUT_INPUT",
    "BLUEPRINT_WAM_ROLLOUT_OUTPUT",
    "BLUEPRINT_OSCAR_WAM_CHECKPOINT",
    "BLUEPRINT_WAM_MODEL_CHECKPOINT",
    adapter.VAST_WAM_PUBLIC_IMAGE_ENV,
    adapter.VAST_WAM_MIN_GPU_RAM_MB_ENV,
    adapter.VAST_WAM_EXCLUDED_MACHINE_ID_ENV,
    adapter.ALLOW_VAST_PROVIDER_LAUNCH_ENV,
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


def _write_provider_zip(
    path: Path,
    *,
    valid_video: bool = True,
    runtime_model_truth: bool = True,
    include_rollout_context: bool = True,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    video_path = path.parent / "oscar_generated_rollout_source.mp4"
    if valid_video:
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
        }
        if runtime_model_truth:
            runtime_result.update(
                {
                    "learned_wam_model_ran": True,
                    "truth_boundary": {
                        "generated_video_is_model_output": True,
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
        "docker.io/nijelhunt/blueprint-oscar-wam:20260621-cu128-shim",
    )
    monkeypatch.setenv(
        adapter.VAST_WAM_PUBLIC_IMAGE_ENV,
        "docker.io/nijelhunt/blueprint-oscar-wam:20260621-cu128-shim",
    )
    monkeypatch.setenv(adapter.VAST_WAM_MIN_GPU_RAM_MB_ENV, "48000")
    monkeypatch.setenv(adapter.VAST_WAM_EXCLUDED_MACHINE_ID_ENV, "134862, 42, bad, 134862")
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
    monkeypatch.setattr(adapter, "create_async_vast_wam_run", fake_create)
    monkeypatch.setattr(adapter, "poll_async_vast_wam_run", fake_poll)

    payload = adapter.run(["--mode", "vast-provider", "--work-dir", str(work_dir)])

    assert payload["status"] == "completed"
    assert payload["mode"] == "vast_provider"
    assert payload["provider_output_zip_imported"] is True
    assert payload["provider_output_replayed"] is False
    assert payload["provider_output_imported_from_current_provider_run"] is True
    assert payload["fresh_provider_launch_attempted"] is True
    assert payload["fresh_model_run_claimed"] is True
    assert payload["generated_rollout_visual_smoke_status"] == "passed_visual_quality_smoke"
    assert payload["generated_rollout_visually_useful_for_success_review"] is True
    assert payload["details"]["vast_provider_job_dir"] == str(work_dir / "vast_provider_run")
    assert Path(payload["rollouts"][0]["generated_video_path"]).is_file()
    assert (
        captured_create["public_image"]
        == "docker.io/nijelhunt/blueprint-oscar-wam:20260621-cu128-shim"
    )
    assert captured_create["min_gpu_ram_mb"] == 48000
    assert captured_create["excluded_machine_ids"] == [134862, 42]
    assert captured_bundle["num_frames"] == 49
    assert captured_bundle["height"] == 480
    assert captured_bundle["width"] == 640
    assert captured_bundle["fps"] == 15.0
    assert captured_bundle["num_steps"] == 41
    assert captured_bundle["guidance"] == 4.5
    assert captured_bundle["seed"] == 123
    assert captured_poll["max_wait_seconds"] == 900
    assert captured_poll["teardown"] is True


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
    monkeypatch.setattr(adapter, "create_async_vast_wam_run", fake_create)
    monkeypatch.setattr(adapter, "poll_async_vast_wam_run", fake_poll)

    payload = adapter.run(["--mode", "vast-provider", "--work-dir", str(work_dir)])

    assert payload["status"] == "completed"
    assert payload["provider_output_imported_from_current_provider_run"] is True
    assert payload["provider_runtime_result_present"] is True
    assert payload["provider_runtime_result_proves_model_output"] is False
    assert payload["provider_learned_wam_model_ran"] is False
    assert payload["provider_generated_video_is_model_output"] is False
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

    monkeypatch.setattr(adapter, "create_async_vast_wam_run", fail_create)
    monkeypatch.setattr(adapter, "poll_async_vast_wam_run", fail_poll)

    payload = adapter.run(["--mode", "auto", "--work-dir", str(tmp_path / "work")])

    assert payload["status"] == "completed"
    assert payload["mode"] == "replay_existing_provider_output"
    assert payload["provider_output_replayed"] is True
    assert payload["fresh_provider_launch_attempted"] is False
    assert payload["fresh_model_run_claimed"] is False
