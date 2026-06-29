from __future__ import annotations

import json
import zipfile
from pathlib import Path

from blueprint_pipeline import runpod_wam_async_runner as runner
from blueprint_pipeline.runpod_provider_adapter import RUNPOD_API_GATE_ENV


def _python_heredoc_chunks(script: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] | None = None
    for line in script.splitlines():
        if current is None and "python" in line and line.endswith("<<'PY'"):
            current = []
            continue
        if current is not None and line == "PY":
            chunks.append("\n".join(current) + "\n")
            current = None
            continue
        if current is not None:
            current.append(line)
    return chunks


def test_runpod_wam_defaults_to_48gb_gpu_classes() -> None:
    assert runner.DEFAULT_GPU_TYPE_IDS[:4] == (
        "NVIDIA L40S",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA RTX A6000",
        "NVIDIA A40",
    )
    assert "NVIDIA GeForce RTX 4090" not in runner.DEFAULT_GPU_TYPE_IDS
    assert "NVIDIA GeForce RTX 3090" not in runner.DEFAULT_GPU_TYPE_IDS


def test_runpod_unitree_groot_sonic_persistent_payload_uses_provider_kind() -> None:
    payload = runner._pod_payload(
        job_name="blueprint-unitree-groot-sonic-test",
        image_name="pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime",
        gpu_type_ids=("NVIDIA L40S",),
        provider_bundle_url="https://store.example/bundle.zip?secret",
        provider_output_put_url="https://store.example/out.zip?secret",
        provider_bundle_kind="unitree_groot_n17_sonic",
        model_secret_env={"HF_TOKEN": "hf-not-persisted"},
        provider_runtime_config_env={
            "BLUEPRINT_OSCAR_WAM_FPS": "4",
            "BLUEPRINT_PERSISTENT_SESSION_WAM_STEP_TIMEOUT_SECONDS": "120",
        },
        container_disk_gb=160,
        volume_gb=40,
    )

    assert payload["env"]["BLUEPRINT_RUNPOD_PROVIDER_BUNDLE_KIND"] == "unitree_groot_n17_sonic"
    assert payload["env"]["WORK_DIR"] == "/workspace/blueprint_unitree_groot_sonic_persistent_provider"
    assert payload["env"]["BLUEPRINT_OSCAR_WAM_FPS"] == "4"
    assert payload["env"]["BLUEPRINT_PERSISTENT_SESSION_WAM_STEP_TIMEOUT_SECONDS"] == "120"
    script = payload["dockerStartCmd"][0]
    assert "run_unitree_groot_n17_sonic_runpod_wrapper.sh" in script
    assert "unitree_groot_n17_sonic_wam_persistent_session_bundle.zip" in script
    assert "\n\timport os\n\timport urllib.request" not in script
    assert "runpod_unitree_groot_sonic_remote_heartbeat" not in script
    assert "os.walk(output_dir)" not in script
    assert "runpod_unitree_groot_sonic_outer_bootstrap_failed_before_inner_wrapper_result" in script
    heredocs = _python_heredoc_chunks(script)
    assert len(heredocs) == 2
    for index, chunk in enumerate(heredocs):
        compile(chunk, f"<unitree_groot_sonic_runpod_heredoc_{index}>", "exec")
    assert len(script) < 4500


def test_runpod_wam_payload_wraps_entrypoint_with_timeout_and_log() -> None:
    payload = runner._pod_payload(
        job_name="blueprint-wam-test",
        image_name="pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime",
        gpu_type_ids=("NVIDIA L40S",),
        provider_bundle_url="https://store.example/bundle.zip?secret",
        provider_output_put_url="https://store.example/out.zip?secret",
        provider_bundle_kind="wam",
        model_secret_env={},
        provider_runtime_config_env={
            "BLUEPRINT_RUNPOD_WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS": "240",
            "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_WRAPPER_WATCHDOG_SECONDS": "180",
        },
        container_disk_gb=160,
        volume_gb=40,
    )

    assert payload["env"]["BLUEPRINT_RUNPOD_PROVIDER_BUNDLE_KIND"] == "wam"
    assert payload["env"]["BLUEPRINT_RUNPOD_WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS"] == "240"
    script = payload["dockerStartCmd"][0]
    assert "runpod_wam_provider_entrypoint.log" in script
    assert "timeout \"$WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS\" bash" in script
    assert "runpod_wam_provider_entrypoint_execution.json" in script
    assert "runpod_wam_provider_entrypoint_timeout" in script
    assert "runpod_wam_outer_bootstrap_failed_before_runtime_result" in script
    assert "unitree_groot_n17_sonic_wam_persistent_session_output.v1" in script
    assert "upload_wam_running_heartbeat runpod_wam_outer_wrapper_started" in script
    assert "upload_wam_running_heartbeat runpod_wam_entrypoint_starting" in script
    heredocs = _python_heredoc_chunks(script)
    assert len(heredocs) == 5
    for index, chunk in enumerate(heredocs):
        compile(chunk, f"<wam_runpod_heredoc_{index}>", "exec")


def test_runpod_wam_carrier_flag_is_forwarded_for_unitree_runtime(monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC", "true")
    monkeypatch.setenv("BLUEPRINT_RUNPOD_WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS", "240")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS", "1200")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER", "true")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE", "system_python_minimal")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT", "true")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON", "/opt/conda/bin/python")
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON_INSTALL_REQUIREMENTS",
        "huggingface_hub pyzmq",
    )

    env, meta = runner._read_provider_runtime_config_env("wam")

    assert meta["status"] == "configured"
    assert env["BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC"] == "true"
    assert env["BLUEPRINT_RUNPOD_WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS"] == "240"
    assert env["BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS"] == "1200"
    assert env["BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER"] == "true"
    assert env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE"] == "system_python_minimal"
    assert env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT"] == "true"
    assert env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON"] == "/opt/conda/bin/python"
    assert env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON_INSTALL_REQUIREMENTS"] == (
        "huggingface_hub pyzmq"
    )


def test_runpod_wam_direct_url_files_block_on_launch_gates_without_leaking_urls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "provider_bundle.zip"
    bundle.write_bytes(b"bundle")
    bundle_url_file = tmp_path / "provider_bundle_url.txt"
    output_url_file = tmp_path / "provider_output_put_url.txt"
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    bundle_url_file.write_text(
        "https://spaces.example/bundle.zip?X-Amz-Signature=bundle-secret\n",
        encoding="utf-8",
    )
    output_url_file.write_text(
        "https://spaces.example/output.zip?X-Amz-Signature=output-secret\n",
        encoding="utf-8",
    )
    output_get_url_file.write_text(
        "https://spaces.example/output.zip?X-Amz-Signature=output-get-secret\n",
        encoding="utf-8",
    )
    hf_token_file = tmp_path / "hf_token"
    hf_token_file.write_text("hf-secret-not-persisted\n", encoding="utf-8")
    bundle_url_file.chmod(0o600)
    output_url_file.chmod(0o600)
    output_get_url_file.chmod(0o600)
    hf_token_file.chmod(0o600)
    monkeypatch.setenv("HF_TOKEN_FILE", str(hf_token_file))
    monkeypatch.setenv(
        "BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY",
        "require_real_transformer_engine",
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS", "1200")
    monkeypatch.delenv(RUNPOD_API_GATE_ENV, raising=False)
    monkeypatch.delenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, raising=False)
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: (
            "runpod-secret-not-persisted",
            {"api_key_configured": True, "raw_secret_values_recorded": False},
        ),
    )

    manifest = runner.create_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        bundle_path=bundle,
        provider_bundle_url_file=bundle_url_file,
        provider_output_put_url_file=output_url_file,
        provider_output_get_url_file=output_get_url_file,
        skip_public_staging_verification=True,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["explicit_provider_urls_used"] is True
    assert "paid_runpod_launch_not_authorized_by_runner_flag" in manifest["blockers"]
    assert f"missing_env_{RUNPOD_API_GATE_ENV}" in manifest["blockers"]
    assert f"missing_env_{runner.RUNPOD_POD_LAUNCH_GATE_ENV}" in manifest["blockers"]
    assert manifest["provider_bundle_url_file"]["mode_is_0600"] is True
    assert manifest["provider_output_put_url_file"]["mode_is_0600"] is True
    assert manifest["provider_output_get_url_file"]["mode_is_0600"] is True
    assert manifest["model_secret_env_status"]["status"] == "configured"
    assert manifest["model_secret_env_status"]["env_keys_forwarded"] == [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ]
    assert manifest["model_secret_env_status"]["selected_file"]["mode_is_0600"] is True
    assert manifest["provider_runtime_config_env_status"]["status"] == "configured"
    assert manifest["provider_runtime_config_env_status"]["values"] == {
        "BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS": "1200",
        "BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY": "require_real_transformer_engine"
    }
    persisted = (tmp_path / "job" / "runpod_wam_async_create_manifest.json").read_text(
        encoding="utf-8"
    )
    direct_manifest = (
        tmp_path / "job" / "runpod_wam_direct_provider_urls_manifest.json"
    ).read_text(encoding="utf-8")
    assert "bundle-secret" not in persisted
    assert "output-secret" not in persisted
    assert "output-get-secret" not in persisted
    assert "runpod-secret-not-persisted" not in persisted
    assert "hf-secret-not-persisted" not in persisted
    assert "bundle-secret" not in direct_manifest
    assert "output-secret" not in direct_manifest
    assert "output-get-secret" not in direct_manifest
    assert "hf-secret-not-persisted" not in direct_manifest
    parsed = json.loads(direct_manifest)
    assert parsed["provider_bundle_url_redacted"].endswith("?REDACTED_QUERY")


def test_runpod_create_allows_unitree_groot_sonic_full_loop_bundle_without_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "unitree_groot_sonic_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "provider_runtime/persistent_session_input.json",
            json.dumps(
                {
                    "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_input.v1",
                    "loop_step_count": 12,
                    "use_live_wam": True,
                    "allow_structural_wam_fallback": False,
                }
            ),
        )

    captured: dict[str, object] = {}

    def fake_runpod_request(**kwargs):
        captured["path"] = kwargs["path"]
        captured["payload"] = kwargs["payload"]
        return 200, {"id": "pod-123"}

    monkeypatch.delenv(runner.RUNPOD_UNITREE_GROOT_SONIC_FULL_LOOP_OVERRIDE_ENV, raising=False)
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: (
            "runpod-secret-not-persisted",
            {"api_key_configured": True, "raw_secret_values_recorded": False},
        ),
    )

    manifest = runner.create_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        bundle_path=bundle,
        provider_bundle_url="https://spaces.example/bundle.zip?X-Amz-Signature=bundle-secret",
        provider_output_put_url="https://spaces.example/output.zip?X-Amz-Signature=output-secret",
        allow_paid_runpod_launch=True,
        skip_public_staging_verification=True,
        generated_at="now",
    )

    assert manifest["status"] == "pod_created"
    assert manifest["pod_id"] == "pod-123"
    assert captured["path"] == "/pods"
    assert captured["payload"]["env"]["BLUEPRINT_RUNPOD_PROVIDER_BUNDLE_KIND"] == "wam"
    assert manifest["full_loop_guard"]["status"] == "allowed"
    assert manifest["full_loop_guard"]["requested_loop_step_count"] == 12
    assert manifest["full_loop_guard"]["full_loop_launch_is_default"] is True
    assert (tmp_path / "job" / "runpod_wam_direct_provider_urls_manifest.json").exists()
    persisted = (tmp_path / "job" / "runpod_wam_async_create_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "bundle-secret" not in persisted
    assert "output-secret" not in persisted


def test_runpod_create_clears_stale_output_from_prior_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A re-fire over an existing job dir must clear the prior run's output zip; otherwise the
    poll treats the stale file as this run's result and short-circuits before the worker uploads.
    """
    bundle = tmp_path / "wam_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo hi\n")
    job = tmp_path / "job"
    job.mkdir()
    stale = job / "runpod_provider_runtime_output.zip"
    stale.write_bytes(b"stale-terminal-from-prior-run")
    stale_nonterminal = job / "runpod_provider_runtime_output_nonterminal.zip"
    stale_nonterminal.write_bytes(b"stale-nonterminal-from-prior-run")

    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setattr(runner, "_runpod_request", lambda **kwargs: (200, {"id": "pod-xyz"}))
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: (
            "runpod-secret-not-persisted",
            {"api_key_configured": True, "raw_secret_values_recorded": False},
        ),
    )

    runner.create_runpod_wam_async_run(
        job_dir=job,
        bundle_path=bundle,
        output_path=stale,
        provider_bundle_url="https://spaces.example/bundle.zip?X-Amz-Signature=bundle-secret",
        provider_output_put_url="https://spaces.example/output.zip?X-Amz-Signature=output-secret",
        allow_paid_runpod_launch=True,
        skip_public_staging_verification=True,
        generated_at="now",
    )

    # cleanup runs at create start (before launch), so stale output is gone regardless of outcome
    assert not stale.exists()
    assert not stale_nonterminal.exists()


def test_runpod_poll_downloads_provider_output_get_url_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    output_get_url_file.write_text(
        "https://spaces.example/output.zip?X-Amz-Signature=download-secret\n",
        encoding="utf-8",
    )
    output_get_url_file.chmod(0o600)
    output_zip = tmp_path / "job" / "vast_provider_runtime_output.zip"
    (tmp_path / "job").mkdir()
    downloaded_zip = tmp_path / "downloaded.zip"
    with zipfile.ZipFile(downloaded_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "wam_runtime_result.json",
            json.dumps({"status": "completed", "blockers": []}),
        )
    (tmp_path / "job" / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "pod_id": "pod-123",
                "output_path": str(output_zip),
                "provider_output_get_url_file": {
                    "path": str(output_get_url_file),
                    "raw_secret_values_recorded": False,
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return downloaded_zip.read_bytes()

    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(
        runner,
        "_runpod_request",
        lambda **kwargs: (200, {"desiredStatus": "PENDING"}),
    )
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        max_wait_seconds=1,
        retry_interval_seconds=1,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["output_zip_present"] is True
    assert manifest["runtime_result_status"] == "completed"
    assert output_zip.is_file()
    persisted = (tmp_path / "job" / "runpod_wam_output_download_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "download-secret" not in persisted
    assert "REDACTED_QUERY" in persisted


def test_runpod_poll_tolerates_transient_not_found_after_create(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_zip = tmp_path / "job" / "runpod_provider_runtime_output.zip"
    (tmp_path / "job").mkdir()
    (tmp_path / "job" / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "unitree_groot_n17_sonic",
                "pod_id": "pod-123",
                "created_at_epoch": runner.time.time(),
                "output_path": str(output_zip),
            }
        ),
        encoding="utf-8",
    )

    def write_output_zip() -> None:
        if output_zip.is_file():
            return
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "unitree_groot_n17_sonic_wam_persistent_session_output.json",
                json.dumps(
                    {
                        "status": "completed",
                        "blockers": [],
                        "repeated_policy_calls_count": 2,
                        "generated_next_observation_count": 1,
                        "live_wam_generation_success_count": 1,
                    }
                ),
            )

    def fake_runpod_request(**kwargs):
        raise runner.urllib.error.HTTPError(
            url="https://rest.runpod.io/v1/pods/pod-123",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    def fake_sleep(_seconds: object) -> None:
        write_output_zip()

    monkeypatch.setenv("BLUEPRINT_RUNPOD_POD_STATUS_NOT_FOUND_GRACE_SECONDS", "300")
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)
    monkeypatch.setattr(runner.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        runner,
        "_delete_pod",
        lambda **kwargs: {"status": "completed", "raw_secret_values_recorded": False},
    )

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        max_wait_seconds=10,
        retry_interval_seconds=1,
        teardown=True,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["output_zip_present"] is True
    assert manifest["pod_status"] == "pending_api_visibility"
    assert manifest["pod_status_transient_not_found_count"] == 1
    assert manifest["teardown_performed"] is True
    assert (tmp_path / "job" / "runpod_wam_async_pre_teardown_poll_manifest.json").is_file()


def test_runpod_poll_can_stop_pod_for_warm_reuse_instead_of_delete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_dir = tmp_path / "job"
    output_zip = job_dir / "runpod_provider_runtime_output.zip"
    job_dir.mkdir()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("wam_runtime_result.json", json.dumps({"status": "completed"}))
        archive.writestr("oscar_generated_rollout.mp4", b"fake-mp4")
    (job_dir / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-123",
                "created_at_epoch": runner.time.time(),
                "output_path": str(output_zip),
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["path"] == "/pods/pod-123":
            return 200, {"desiredStatus": "RUNNING"}
        if kwargs["path"] == "/pods/pod-123/stop":
            return 200, {"id": "pod-123", "desiredStatus": "EXITED"}
        raise AssertionError(f"unexpected runpod request: {kwargs}")

    monkeypatch.setenv(runner.RUNPOD_WAM_TEARDOWN_ACTION_ENV, "stop")
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=job_dir,
        max_wait_seconds=1,
        retry_interval_seconds=1,
        teardown=True,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["teardown_action"] == "stop"
    assert manifest["teardown_performed"] is True
    assert manifest["continuing_spend_from_this_run"] is False
    assert (job_dir / "runpod_wam_async_stop_manifest.json").is_file()
    assert not (job_dir / "runpod_wam_async_delete_manifest.json").exists()
    assert any(request["path"] == "/pods/pod-123/stop" for request in requests)


def test_runpod_poll_stops_not_found_grace_when_delete_already_completed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_zip = tmp_path / "job" / "runpod_provider_runtime_output.zip"
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-123",
                "created_at_epoch": runner.time.time(),
                "output_path": str(output_zip),
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "runpod_wam_async_delete_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "runpod_wam_async_delete_manifest.v1",
                "status": "completed",
                "pod_id": "pod-123",
                "http_status_code": 204,
                "continuing_spend_from_this_run": False,
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    delete_called = {"value": False}

    def fake_runpod_request(**kwargs):
        raise runner.urllib.error.HTTPError(
            url="https://rest.runpod.io/v1/pods/pod-123",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    def fake_delete(**kwargs):
        delete_called["value"] = True
        return {"status": "completed", "raw_secret_values_recorded": False}

    monkeypatch.setenv("BLUEPRINT_RUNPOD_POD_STATUS_NOT_FOUND_GRACE_SECONDS", "300")
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)
    monkeypatch.setattr(runner, "_delete_pod", fake_delete)

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=job_dir,
        max_wait_seconds=300,
        retry_interval_seconds=300,
        teardown=True,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["pod_status"] == "not_found"
    assert manifest["pod_status_transient_not_found_count"] == 0
    assert manifest["teardown_performed"] is True
    assert manifest["continuing_spend_from_this_run"] is False
    assert delete_called["value"] is False


def test_runpod_output_download_rejects_empty_zip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_zip = tmp_path / "job" / "runpod_provider_runtime_output.zip"

    class EmptyResponse:
        def __enter__(self) -> "EmptyResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *args, **kwargs: EmptyResponse())

    manifest = runner._download_provider_output_zip(
        job_dir=tmp_path / "job",
        provider_output_get_url="https://store.example/out.zip?X-Amz-Signature=secret",
        output_path=output_zip,
        generated_at="now",
    )

    assert manifest["status"] == "not_available"
    assert manifest["downloaded_size_bytes"] == 0
    assert manifest["empty_download"] is True
    assert manifest["valid_zip"] is False
    assert output_zip.exists() is False
    persisted = (tmp_path / "job" / "runpod_wam_output_download_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "X-Amz-Signature=secret" not in persisted
    assert "REDACTED_QUERY" in persisted


def test_runpod_unitree_unifolm_create_uses_provider_kind_without_leaking_urls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "unitree_unifolm_policy_provider_runtime_bundle.zip"
    bundle.write_bytes(b"bundle")
    bundle_url_file = tmp_path / "provider_bundle_url.txt"
    output_url_file = tmp_path / "provider_output_put_url.txt"
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    bundle_url_file.write_text(
        "https://spaces.example/unitree-bundle.zip?X-Amz-Signature=bundle-secret\n",
        encoding="utf-8",
    )
    output_url_file.write_text(
        "https://spaces.example/unitree-output.zip?X-Amz-Signature=output-secret\n",
        encoding="utf-8",
    )
    output_get_url_file.write_text(
        "https://spaces.example/unitree-output.zip?X-Amz-Signature=output-get-secret\n",
        encoding="utf-8",
    )
    for path in (bundle_url_file, output_url_file, output_get_url_file):
        path.chmod(0o600)
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: (
            "runpod-secret-not-persisted",
            {"api_key_configured": True, "raw_secret_values_recorded": False},
        ),
    )
    requests: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs: object) -> tuple[int, dict[str, object]]:
        requests.append(dict(kwargs))
        return 200, {"id": "pod-unitree-123"}

    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)

    manifest = runner.create_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        bundle_path=bundle,
        provider_bundle_url_file=bundle_url_file,
        provider_output_put_url_file=output_url_file,
        provider_output_get_url_file=output_get_url_file,
        skip_public_staging_verification=True,
        allow_paid_runpod_launch=True,
        provider_bundle_kind="unitree_unifolm",
        image_name="nijelhunt/blueprint-unitree-unifolm:test",
        generated_at="now",
    )

    assert manifest["status"] == "pod_created"
    assert manifest["provider_bundle_kind"] == "unitree_unifolm"
    assert requests
    payload = requests[0]["payload"]
    assert isinstance(payload, dict)
    env = payload["env"]
    assert isinstance(env, dict)
    assert env["BLUEPRINT_RUNPOD_PROVIDER_BUNDLE_KIND"] == "unitree_unifolm"
    assert env["BLUEPRINT_UNITREE_UNIFOLM_COMMAND"] == "/usr/local/bin/run_unitree_unifolm_vla_policy_once"
    assert env["BLUEPRINT_UNITREE_UNIFOLM_CHECKPOINT"] == "unitreerobotics/UnifoLM-VLA-Base"
    assert env["BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT"] == "unitreerobotics/UnifoLM-VLM-Base"
    script = payload["dockerStartCmd"][0]
    assert "run_unitree_unifolm_provider_runtime.sh" in script
    assert "run_wam_provider_runtime.sh" not in script
    persisted = (tmp_path / "job" / "runpod_wam_async_create_manifest.json").read_text(
        encoding="utf-8"
    )
    state = (tmp_path / "job" / "runpod_wam_async_state.json").read_text(
        encoding="utf-8"
    )
    assert "bundle-secret" not in persisted
    assert "output-secret" not in persisted
    assert "output-get-secret" not in persisted
    assert "runpod-secret-not-persisted" not in persisted
    assert "bundle-secret" not in state
    assert "output-secret" not in state
    assert "output-get-secret" not in state
    assert "runpod-secret-not-persisted" not in state


def test_runpod_poll_accepts_unitree_unifolm_output_without_video_requirement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    output_get_url_file.write_text(
        "https://spaces.example/unitree-output.zip?X-Amz-Signature=download-secret\n",
        encoding="utf-8",
    )
    output_get_url_file.chmod(0o600)
    output_zip = tmp_path / "job" / "vast_provider_runtime_output.zip"
    (tmp_path / "job").mkdir()
    downloaded_zip = tmp_path / "downloaded.zip"
    with zipfile.ZipFile(downloaded_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_unifolm_policy_provider_output.json",
            json.dumps(
                {
                    "status": "completed",
                    "blockers": [],
                    "unitree_unifolm_model_executed": True,
                    "unitree_unifolm_policy_action_command_ran": True,
                    "action": {"action_type": "manipulation_contact"},
                }
            ),
        )
    (tmp_path / "job" / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "unitree_unifolm",
                "pod_id": "pod-123",
                "output_path": str(output_zip),
                "provider_output_get_url_file": {
                    "path": str(output_get_url_file),
                    "raw_secret_values_recorded": False,
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return downloaded_zip.read_bytes()

    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(
        runner,
        "_runpod_request",
        lambda **kwargs: (200, {"desiredStatus": "RUNNING"}),
    )
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        max_wait_seconds=1,
        retry_interval_seconds=1,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["provider_bundle_kind"] == "unitree_unifolm"
    assert manifest["output_zip_present"] is True
    assert manifest["runtime_result_status"] == "completed"
    assert manifest["mp4_count"] == 0
    persisted = (tmp_path / "job" / "runpod_wam_output_download_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "download-secret" not in persisted


def test_runpod_poll_accepts_unitree_groot_sonic_persistent_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    output_get_url_file.write_text(
        "https://spaces.example/persistent-output.zip?X-Amz-Signature=download-secret\n",
        encoding="utf-8",
    )
    output_get_url_file.chmod(0o600)
    output_zip = tmp_path / "job" / "runpod_provider_runtime_output.zip"
    (tmp_path / "job").mkdir()
    downloaded_zip = tmp_path / "downloaded.zip"
    with zipfile.ZipFile(downloaded_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_wam_persistent_session_output.json",
            json.dumps(
                {
                    "status": "completed",
                    "blockers": [],
                    "repeated_policy_calls_count": 3,
                    "generated_next_observation_count": 2,
                    "live_wam_generation_success_count": 2,
                    "learned_wam_model_success_count": 2,
                    "policy_observes_wam_generated_next_observation": True,
                    "provider_instance_reused_for_policy_and_wam_loop": True,
                }
            ),
        )
        archive.writestr("wam_worker_steps/step_0001/oscar_runtime_output/oscar.mp4", b"mp4")
    (tmp_path / "job" / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "unitree_groot_n17_sonic",
                "pod_id": "pod-123",
                "output_path": str(output_zip),
                "provider_output_get_url_file": {
                    "path": str(output_get_url_file),
                    "raw_secret_values_recorded": False,
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return downloaded_zip.read_bytes()

    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(
        runner,
        "_runpod_request",
        lambda **kwargs: (200, {"desiredStatus": "RUNNING"}),
    )
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        max_wait_seconds=1,
        retry_interval_seconds=1,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["provider_bundle_kind"] == "unitree_groot_n17_sonic"
    assert manifest["output_zip_present"] is True
    assert manifest["runtime_result_status"] == "completed"
    runtime_result = manifest["runtime_result"]
    assert runtime_result["repeated_policy_calls_count"] == 3
    assert runtime_result["live_wam_generation_success_count"] == 2
    persisted = (tmp_path / "job" / "runpod_wam_output_download_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "download-secret" not in persisted


def test_runpod_poll_ignores_unitree_groot_sonic_nonterminal_heartbeat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    output_get_url_file.write_text(
        "https://spaces.example/persistent-output.zip?X-Amz-Signature=download-secret\n",
        encoding="utf-8",
    )
    output_get_url_file.chmod(0o600)
    output_zip = tmp_path / "job" / "runpod_provider_runtime_output.zip"
    (tmp_path / "job").mkdir()
    running_zip = tmp_path / "running.zip"
    completed_zip = tmp_path / "completed.zip"
    with zipfile.ZipFile(running_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_wam_persistent_session_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "runpod_inline_bootstrap_started",
                    "runpod_unitree_groot_sonic_remote_heartbeat": True,
                    "blockers": [],
                }
            ),
        )
    with zipfile.ZipFile(completed_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_wam_persistent_session_output.json",
            json.dumps(
                {
                    "status": "completed",
                    "blockers": [],
                    "repeated_policy_calls_count": 2,
                    "generated_next_observation_count": 1,
                    "live_wam_generation_success_count": 1,
                    "learned_wam_model_success_count": 1,
                    "policy_observes_wam_generated_next_observation": True,
                }
            ),
        )
    (tmp_path / "job" / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "unitree_groot_n17_sonic",
                "pod_id": "pod-123",
                "output_path": str(output_zip),
                "provider_output_get_url_file": {
                    "path": str(output_get_url_file),
                    "raw_secret_values_recorded": False,
                },
            }
        ),
        encoding="utf-8",
    )
    zip_sequence = [running_zip, completed_zip]
    read_count = {"value": 0}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            index = min(read_count["value"], len(zip_sequence) - 1)
            read_count["value"] += 1
            return zip_sequence[index].read_bytes()

    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(
        runner,
        "_runpod_request",
        lambda **kwargs: (200, {"desiredStatus": "RUNNING"}),
    )
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        max_wait_seconds=2,
        retry_interval_seconds=1,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["runtime_result_status"] == "completed"
    assert manifest["last_nonterminal_output"]["runtime_result_status"] == "running"
    assert read_count["value"] == 2
    assert (tmp_path / "job" / "runpod_provider_runtime_output_nonterminal.zip").is_file()
    assert (tmp_path / "job" / "runpod_wam_nonterminal_output_manifest.json").is_file()


def test_runpod_poll_recognizes_oscar_wam_provider_output_heartbeat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """OSCAR's first heartbeat zip holds only wam_provider_output.json (status=running) with no
    wam_runtime_result.json. The poll must treat it as nonterminal and keep waiting, not mistake
    it for completion and tear the pod down before deps/checkpoint/inference can run.
    """
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    output_get_url_file.write_text(
        "https://spaces.example/oscar-output.zip?X-Amz-Signature=download-secret\n",
        encoding="utf-8",
    )
    output_get_url_file.chmod(0o600)
    output_zip = tmp_path / "job" / "runpod_provider_runtime_output.zip"
    (tmp_path / "job").mkdir()
    running_zip = tmp_path / "running.zip"
    completed_zip = tmp_path / "completed.zip"
    with zipfile.ZipFile(running_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "wam_provider_output.json",
            json.dumps(
                {
                    "schema_version": "wam_provider_output.v1",
                    "status": "running",
                    "runtime_phase": "runpod_wam_system_dependency_install_started",
                    "blockers": [],
                }
            ),
        )
    with zipfile.ZipFile(completed_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "wam_runtime_result.json",
            json.dumps(
                {
                    "status": "completed",
                    "blockers": [],
                    "generated_video_path": "oscar_generated_rollout.mp4",
                    "learned_wam_model_ran": True,
                }
            ),
        )
        archive.writestr("oscar_generated_rollout.mp4", b"\x00\x00fakemp4")
    (tmp_path / "job" / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-oscar-1",
                "output_path": str(output_zip),
                "provider_output_get_url_file": {
                    "path": str(output_get_url_file),
                    "raw_secret_values_recorded": False,
                },
            }
        ),
        encoding="utf-8",
    )
    zip_sequence = [running_zip, completed_zip]
    read_count = {"value": 0}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            index = min(read_count["value"], len(zip_sequence) - 1)
            read_count["value"] += 1
            return zip_sequence[index].read_bytes()

    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(
        runner,
        "_runpod_request",
        lambda **kwargs: (200, {"desiredStatus": "RUNNING"}),
    )
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        max_wait_seconds=2,
        retry_interval_seconds=1,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["runtime_result_status"] == "completed"
    # the wam_provider_output.json running heartbeat was recognized as nonterminal (kept polling)
    assert manifest["last_nonterminal_output"]["runtime_result_status"] == "running"
    assert read_count["value"] == 2


def test_runpod_poll_preserves_running_nonterminal_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    output_get_url_file.write_text(
        "https://spaces.example/persistent-output.zip?X-Amz-Signature=download-secret\n",
        encoding="utf-8",
    )
    output_get_url_file.chmod(0o600)
    output_zip = tmp_path / "job" / "runpod_provider_runtime_output.zip"
    (tmp_path / "job").mkdir()
    running_zip = tmp_path / "running.zip"
    with zipfile.ZipFile(running_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_wam_persistent_session_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "runpod_entrypoint_subprocess_starting",
                    "blockers": [],
                }
            ),
        )
    (tmp_path / "job" / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-123",
                "output_path": str(output_zip),
                "provider_output_get_url_file": {
                    "path": str(output_get_url_file),
                    "raw_secret_values_recorded": False,
                },
            }
        ),
        encoding="utf-8",
    )
    delete_called = {"value": False}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return running_zip.read_bytes()

    def fake_delete(**kwargs):
        delete_called["value"] = True
        return {"status": "completed"}

    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(
        runner,
        "_runpod_request",
        lambda **kwargs: (200, {"desiredStatus": "RUNNING"}),
    )
    monkeypatch.setattr(runner, "_delete_pod", fake_delete)
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        max_wait_seconds=100,
        retry_interval_seconds=200,
        teardown=True,
        generated_at="now",
    )

    assert manifest["status"] == "running"
    assert manifest["provider_command_status"] == "running"
    assert manifest["nonterminal_running_output"] is True
    assert manifest["last_nonterminal_output"]["runtime_result_status"] == "running"
    assert manifest["continuing_spend_from_this_run"] is True
    assert manifest["teardown_performed"] is False
    assert delete_called["value"] is False


def test_runpod_poll_preserves_active_pod_before_first_heartbeat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    output_get_url_file.write_text(
        "https://spaces.example/persistent-output.zip?X-Amz-Signature=download-secret\n",
        encoding="utf-8",
    )
    output_get_url_file.chmod(0o600)
    output_zip = tmp_path / "job" / "runpod_provider_runtime_output.zip"
    (tmp_path / "job").mkdir()
    (tmp_path / "job" / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-123",
                "output_path": str(output_zip),
                "provider_output_get_url_file": {
                    "path": str(output_get_url_file),
                    "raw_secret_values_recorded": False,
                },
            }
        ),
        encoding="utf-8",
    )
    delete_called = {"value": False}

    class MissingOutputResponse:
        def read(self) -> bytes:
            return b"missing"

    def fake_delete(**kwargs):
        delete_called["value"] = True
        return {"status": "completed"}

    def fake_urlopen(*args, **kwargs):
        raise runner.urllib.error.HTTPError(
            url="https://spaces.example/persistent-output.zip",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=MissingOutputResponse(),
        )

    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(
        runner,
        "_runpod_request",
        lambda **kwargs: (200, {"desiredStatus": "PENDING"}),
    )
    monkeypatch.setattr(runner, "_delete_pod", fake_delete)
    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        max_wait_seconds=100,
        retry_interval_seconds=200,
        teardown=True,
        generated_at="now",
    )

    assert manifest["status"] == "running"
    assert manifest["provider_command_status"] == "running"
    assert manifest["output_zip_present"] is False
    assert manifest["nonterminal_running_output"] is False
    assert manifest["remote_runtime_running_without_terminal_output"] is True
    assert manifest["pod_status_is_active"] is True
    assert manifest["provider_command_blockers"] == []
    assert manifest["continuing_spend_from_this_run"] is True
    assert manifest["teardown_performed"] is False
    assert delete_called["value"] is False
