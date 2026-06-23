from __future__ import annotations

import json
import zipfile
from pathlib import Path

from blueprint_pipeline import runpod_wam_async_runner as runner
from blueprint_pipeline.runpod_provider_adapter import RUNPOD_API_GATE_ENV


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
    assert manifest["output_zip_present"] is True
    assert manifest["runtime_result_status"] == "completed"
    assert output_zip.is_file()
    persisted = (tmp_path / "job" / "runpod_wam_output_download_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "download-secret" not in persisted
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
