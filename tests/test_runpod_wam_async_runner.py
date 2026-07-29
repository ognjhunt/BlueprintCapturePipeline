from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from pathlib import Path

from blueprint_pipeline import runpod_wam_async_runner as runner
from blueprint_pipeline.groot_oscar_runpod_carrier_volume import (
    CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
    DEFAULT_MODEL_CACHE_ROOT,
    DEFAULT_RUNTIME_ARCHIVE_PATH,
    DEFAULT_RUNTIME_MANIFEST_PATH,
    DEFAULT_RUNTIME_ROOT,
    RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION,
    RUNTIME_SOURCE_RELEASE_VERIFICATION_SCHEMA_VERSION,
)
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from blueprint_pipeline.runpod_provider_adapter import RUNPOD_API_GATE_ENV

import pytest

pytestmark = pytest.mark.slow


def _paid_grant():
    admission = build_paid_lane_admission(
        resource_class="runpod_wam_async",
        blockers=[],
    )
    return require_paid_resource_admission(
        admission,
        resource_class="runpod_wam_async",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


def _carrier_volume_admission(*, carrier_image_ref: str) -> dict:
    return {
        "schema_version": CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
        "status": "verified",
        "carrier_image_ref": carrier_image_ref,
        "network_volume": {
            "id": "volume123",
            "data_center_id": "EUR-IS-1",
            "size_gib": 120,
        },
        "runtime_bundle": {
            "manifest_schema_version": RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "source_release_image_ref": "docker.io/blueprint/release@sha256:" + "1" * 64,
            "root": DEFAULT_RUNTIME_ROOT,
            "archive_path": DEFAULT_RUNTIME_ARCHIVE_PATH,
            "manifest_path": DEFAULT_RUNTIME_MANIFEST_PATH,
            "archive_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
        },
        "runtime_source_release": {
            "schema_version": RUNTIME_SOURCE_RELEASE_VERIFICATION_SCHEMA_VERSION,
            "status": "verified",
            "release_image_ref": "docker.io/blueprint/release@sha256:" + "1" * 64,
            "source_commit": "a" * 40,
            "thin_release_contract_sha256": "6" * 64,
            "models_externalized": True,
        },
        "model_cache": {
            "status": "verified",
            "root": DEFAULT_MODEL_CACHE_ROOT,
            "manifest_sha256": "5" * 64,
            "manifest_digest": "sha256:" + "7" * 64,
        },
        "s3_transfer_verification": {
            "upload_completed": True,
            "full_redownload_sha256_verified": True,
            "provider_volume_id": "volume123",
            "data_center_id": "EUR-IS-1",
        },
    }


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


def test_runpod_wam_defaults_to_lower_cost_capable_gpu_classes_first() -> None:
    assert runner.DEFAULT_GPU_TYPE_IDS[:4] == (
        "NVIDIA A40",
        "NVIDIA RTX A5000",
        "NVIDIA RTX A6000",
        "NVIDIA L40S",
    )
    assert "NVIDIA RTX 6000 Ada Generation" in runner.DEFAULT_GPU_TYPE_IDS
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
    assert (
        payload["env"]["WORK_DIR"] == "/workspace/blueprint_unitree_groot_sonic_persistent_provider"
    )
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


def test_runpod_small_carrier_payload_attaches_exact_verified_network_volume() -> None:
    carrier_ref = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:" + "2" * 64
    payload = runner._pod_payload(
        job_name="blueprint-carrier-volume-test",
        image_name=carrier_ref,
        gpu_type_ids=("NVIDIA A40",),
        provider_bundle_url="https://store.example/bundle.zip?secret",
        provider_output_put_url="https://store.example/out.zip?secret",
        provider_bundle_kind="unitree_groot_n17_sonic",
        model_secret_env={},
        provider_runtime_config_env={},
        container_disk_gb=240,
        volume_gb=120,
        carrier_volume_admission=_carrier_volume_admission(carrier_image_ref=carrier_ref),
    )

    assert payload["imageName"] == carrier_ref
    assert payload["networkVolumeId"] == "volume123"
    assert payload["dataCenterIds"] == ["EUR-IS-1"]
    assert payload["volumeMountPath"] == "/workspace"
    assert payload["containerDiskInGb"] == 240
    assert payload["volumeInGb"] == 120
    assert payload["gpuTypeIds"] == ["NVIDIA A40"]
    assert payload["env"]["BLUEPRINT_RUNTIME_ARCHIVE_SHA256"] == "3" * 64
    assert payload["env"]["BLUEPRINT_MODEL_CACHE_MANIFEST_SHA256"] == "5" * 64
    script = payload["dockerStartCmd"][0]
    assert script.index("BLUEPRINT_RUNPOD_CARRIER_RUNTIME_BOOTSTRAP_STARTED") < script.index(
        "BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_PERSISTENT_PROVIDER_STARTED"
    )


def test_persistent_carrier_receipt_binds_pending_record_before_and_after_create(
    tmp_path: Path,
) -> None:
    pod_name = "blueprint-groot-oscar-canary-persistent-test"
    receipt_path = tmp_path / "provider_lane_handoff_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "status": "accepted",
                "campaign_kind": "persistent_policy_wam_loop",
                "pod_name_prefix": "blueprint-groot-oscar-canary-",
                "pre_provider_mutation_confirmed_absent": True,
            }
        ),
        encoding="utf-8",
    )
    receipt_path.chmod(0o600)
    pending_path = tmp_path / "pending.json"
    pending = {
        "status": "open",
        "provider": "runpod",
        "lane": runner.RUNPOD_WAM_LANE,
        "resource_kind": "compute_instance",
        "resource_name": pod_name,
        "instance_id": None,
    }
    pending_path.write_text(json.dumps(pending), encoding="utf-8")

    before = runner._update_provider_lane_handoff_receipt(
        receipt_path,
        pod_name=pod_name,
        pending_teardown_record=str(pending_path),
    )
    bound_before = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert before["status"] == "pending_create_bound"
    assert bound_before["pod_pending_teardown_record"] == str(pending_path)
    assert bound_before["pre_provider_mutation_confirmed_absent"] is False
    assert bound_before["pod_id"] is None

    pending["instance_id"] = "pod-123"
    pending_path.write_text(json.dumps(pending), encoding="utf-8")
    after = runner._update_provider_lane_handoff_receipt(
        receipt_path,
        pod_name=pod_name,
        pending_teardown_record=str(pending_path),
        pod_id="pod-123",
    )
    bound_after = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert after["status"] == "pod_id_bound"
    assert bound_after["pod_id"] == "pod-123"
    assert bound_after["provider_mutation_state"] == "pod_id_bound"


def test_persistent_carrier_receipt_returns_to_absent_after_cancelled_create(
    tmp_path: Path,
) -> None:
    pod_name = "blueprint-groot-oscar-canary-persistent-test"
    receipt_path = tmp_path / "provider_lane_handoff_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "status": "accepted",
                "campaign_kind": "persistent_policy_wam_loop",
                "pod_name_prefix": "blueprint-groot-oscar-canary-",
                "pre_provider_mutation_confirmed_absent": True,
            }
        ),
        encoding="utf-8",
    )
    receipt_path.chmod(0o600)
    pending_path = tmp_path / "pending.json"
    pending = {
        "status": "open",
        "provider": "runpod",
        "lane": runner.RUNPOD_WAM_LANE,
        "resource_kind": "compute_instance",
        "resource_name": pod_name,
        "instance_id": None,
    }
    pending_path.write_text(json.dumps(pending), encoding="utf-8")
    runner._update_provider_lane_handoff_receipt(
        receipt_path,
        pod_name=pod_name,
        pending_teardown_record=str(pending_path),
    )
    pending["status"] = "cancelled_no_allocation"
    pending_path.write_text(json.dumps(pending), encoding="utf-8")

    result = runner._confirm_provider_lane_handoff_no_allocation(
        receipt_path,
        pod_name=pod_name,
        pending_teardown_record=str(pending_path),
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result["status"] == "no_allocation_confirmed"
    assert receipt["pre_provider_mutation_confirmed_absent"] is True
    assert receipt["provider_mutation_state"] == "no_allocation_confirmed"
    assert receipt["pod_pending_teardown_record"] is None
    assert receipt["pod_id"] is None


def test_runpod_small_carrier_payload_rejects_h100_and_unverified_volume() -> None:
    carrier_ref = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:" + "2" * 64
    admission = _carrier_volume_admission(carrier_image_ref=carrier_ref)
    admission["s3_transfer_verification"]["full_redownload_sha256_verified"] = False
    with pytest.raises(ValueError, match="carrier_volume_s3_full_redownload_not_verified"):
        runner._pod_payload(
            job_name="blocked",
            image_name=carrier_ref,
            gpu_type_ids=("NVIDIA A40",),
            provider_bundle_url="https://store.example/bundle.zip",
            provider_output_put_url="https://store.example/out.zip",
            provider_bundle_kind="unitree_groot_n17_sonic",
            model_secret_env={},
            provider_runtime_config_env={},
            container_disk_gb=240,
            volume_gb=120,
            carrier_volume_admission=admission,
        )

    with pytest.raises(ValueError, match="carrier_volume_h100_disallowed"):
        runner._pod_payload(
            job_name="blocked-h100",
            image_name=carrier_ref,
            gpu_type_ids=("NVIDIA H100 PCIe",),
            provider_bundle_url="https://store.example/bundle.zip",
            provider_output_put_url="https://store.example/out.zip",
            provider_bundle_kind="unitree_groot_n17_sonic",
            model_secret_env={},
            provider_runtime_config_env={},
            container_disk_gb=240,
            volume_gb=120,
            carrier_volume_admission=_carrier_volume_admission(carrier_image_ref=carrier_ref),
        )


def test_runpod_payload_forwards_success_keepalive_when_keep_requested(monkeypatch) -> None:
    monkeypatch.setenv(runner.RUNPOD_WAM_TEARDOWN_ACTION_ENV, "keep_on_success")

    payload = runner._pod_payload(
        job_name="blueprint-wam-keepalive-test",
        image_name="pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime",
        gpu_type_ids=("NVIDIA A40",),
        provider_bundle_url="https://store.example/bundle.zip?secret",
        provider_output_put_url="https://store.example/out.zip?secret",
        provider_bundle_kind="wam",
        model_secret_env={},
        provider_runtime_config_env={},
        container_disk_gb=160,
        volume_gb=40,
    )

    assert payload["env"]["BLUEPRINT_RUNPOD_KEEPALIVE_AFTER_SUCCESS"] == "1"


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
    assert 'timeout "$WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS" bash' in script
    assert "runpod_wam_provider_entrypoint_execution.json" in script
    assert "runpod_wam_provider_entrypoint_timeout" in script
    assert "runpod_wam_outer_bootstrap_failed_before_runtime_result" in script
    assert "unitree_groot_n17_sonic_wam_persistent_session_output.v1" in script
    assert "upload_wam_running_heartbeat runpod_wam_outer_wrapper_started" in script
    assert "upload_wam_running_heartbeat runpod_wam_entrypoint_starting" in script
    assert "BLUEPRINT_RUNPOD_WAM_ENTRYPOINT_LOG_PATH" in script
    assert "entrypoint_log_tail" in script
    assert "upload_wam_running_heartbeat runpod_wam_entrypoint_running" in script
    assert "entrypoint_heartbeat_pid" in script
    assert (
        'export BLUEPRINT_VAST_PROVIDER_OUTPUT_DIR="$BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR"' in script
    )
    assert 'sleep "$terminal_hold_seconds"' in script
    assert "BLUEPRINT_RUNPOD_WAM_TERMINAL_HOLD_SECONDS" in script
    heredocs = _python_heredoc_chunks(script)
    assert len(heredocs) == 5
    for index, chunk in enumerate(heredocs):
        compile(chunk, f"<wam_runpod_heredoc_{index}>", "exec")


def test_runpod_wam_carrier_flag_is_forwarded_for_unitree_runtime(monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC", "true")
    monkeypatch.setenv("BLUEPRINT_RUNPOD_WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS", "240")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS", "1200")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER", "true")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CONDITIONING_MODE", "oscar_gripper_scenario_proxy")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_RGB_CONTEXT_MODE", "always")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_SMOKE", "agibot_465")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_RGB_VIDEO", "1")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_USE_SCRIPT", "true")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CONDITIONING_BACKGROUND_ALPHA", "0.72")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CONDITIONING_VOID_THRESHOLD", "14")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE", "system_python_minimal")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT", "true")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON", "/opt/conda/bin/python")
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON_INSTALL_REQUIREMENTS",
        "huggingface_hub pyzmq",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_TIMEOUT_SECONDS", "1800")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_ATTEMPTS", "2")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_MAX_WORKERS", "4")

    env, meta = runner._read_provider_runtime_config_env("wam")

    assert meta["status"] == "configured"
    assert env["BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC"] == "true"
    assert env["BLUEPRINT_RUNPOD_WAM_PROVIDER_ENTRYPOINT_TIMEOUT_SECONDS"] == "240"
    assert env["BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS"] == "1200"
    assert env["BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER"] == "true"
    assert env["BLUEPRINT_OSCAR_WAM_CONDITIONING_MODE"] == "oscar_gripper_scenario_proxy"
    assert env["BLUEPRINT_OSCAR_WAM_RGB_CONTEXT_MODE"] == "always"
    assert env["BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_SMOKE"] == "agibot_465"
    assert env["BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_RGB_VIDEO"] == "1"
    assert env["BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_USE_SCRIPT"] == "true"
    assert env["BLUEPRINT_OSCAR_WAM_CONDITIONING_BACKGROUND_ALPHA"] == "0.72"
    assert env["BLUEPRINT_OSCAR_WAM_CONDITIONING_VOID_THRESHOLD"] == "14"
    assert env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE"] == "system_python_minimal"
    assert env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT"] == "true"
    assert env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON"] == "/opt/conda/bin/python"
    assert env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON_INSTALL_REQUIREMENTS"] == (
        "huggingface_hub pyzmq"
    )
    assert env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_TIMEOUT_SECONDS"] == "1800"
    assert env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_ATTEMPTS"] == "2"
    assert env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_MAX_WORKERS"] == "4"


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
    assert "runpod_wam_max_spend_usd_missing" in manifest["blockers"]
    assert manifest["prelaunch_spend_guard"]["can_launch"] is False
    assert manifest["provider_bundle_url_file"]["mode_is_0600"] is True
    assert manifest["provider_output_put_url_file"]["mode_is_0600"] is True
    assert manifest["provider_output_get_url_file"]["mode_is_0600"] is True
    assert manifest["model_secret_env_status"]["status"] == "configured"
    assert manifest["model_secret_env_status"]["env_keys_forwarded"] == [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ]
    assert manifest["model_secret_env_status"]["selected_file"]["mode_is_0600"] is True
    assert manifest["model_secret_env_status"]["selected_file"]["path_redacted"] is True
    assert "path" not in manifest["model_secret_env_status"]["selected_file"]
    assert manifest["provider_runtime_config_env_status"]["status"] == "configured"
    assert manifest["provider_runtime_config_env_status"]["values"] == {
        "BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS": "1200",
        "BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY": "require_real_transformer_engine",
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
    assert str(hf_token_file) not in persisted
    assert "bundle-secret" not in direct_manifest
    assert "output-secret" not in direct_manifest
    assert "output-get-secret" not in direct_manifest
    assert "hf-secret-not-persisted" not in direct_manifest
    parsed = json.loads(direct_manifest)
    assert parsed["provider_bundle_url_redacted"].endswith("?REDACTED_QUERY")


def test_runpod_direct_presigned_bundle_is_verified_with_get(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "provider_bundle.zip"
    bundle.write_bytes(b"bundle")
    captured: dict[str, object] = {}

    def fake_verify(**kwargs):
        captured.update(kwargs)
        return {"status": "passed", "blockers": []}

    monkeypatch.setattr(runner, "verify_public_staging_urls", fake_verify)
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
        provider_bundle_url="https://spaces.example/bundle.zip?signature=secret",
        provider_output_put_url="https://spaces.example/output.zip?signature=secret",
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert captured["bundle_probe_method"] == "GET"
    assert captured["required_consecutive_successes"] == 1


def test_runpod_local_tunnel_bundle_keeps_head_verification(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "provider_bundle.zip"
    bundle.write_bytes(b"bundle")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        runner,
        "prepare_vast_bundle_staging",
        lambda **_kwargs: {"status": "ready", "blockers": []},
    )
    monkeypatch.setattr(
        runner,
        "run_local_staging_self_test",
        lambda **_kwargs: {"status": "passed", "blockers": []},
    )

    def fake_verify(**kwargs):
        captured.update(kwargs)
        return {"status": "passed", "blockers": []}

    monkeypatch.setattr(runner, "verify_public_staging_urls", fake_verify)
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
        public_base_url="https://tunnel.example",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "staging.env",
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert captured["bundle_probe_method"] == "HEAD"
    assert captured["required_consecutive_successes"] == 2


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

    calls: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        calls.append(dict(kwargs))
        return 200, {"id": "pod-123"}

    monkeypatch.delenv(runner.RUNPOD_UNITREE_GROOT_SONIC_FULL_LOOP_OVERRIDE_ENV, raising=False)
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
    monkeypatch.setenv(runner.RUNPOD_WAM_DISABLE_WARM_CANDIDATE_ENV, "true")
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
        paid_resource_admission_grant=_paid_grant(),
        skip_public_staging_verification=True,
        generated_at="now",
    )

    assert manifest["status"] == "pod_created"
    assert manifest["pod_id"] == "pod-123"
    assert manifest["prelaunch_spend_guard"]["can_launch"] is True
    assert manifest["prelaunch_spend_guard"]["requested_budget_usd"] == 0.75
    assert [call["path"] for call in calls] == ["/pods"]
    create_call = calls[0]
    assert create_call["payload"]["env"]["BLUEPRINT_RUNPOD_PROVIDER_BUNDLE_KIND"] == "wam"
    assert manifest["full_loop_guard"]["status"] == "allowed"
    assert manifest["full_loop_guard"]["requested_loop_step_count"] == 12
    assert manifest["full_loop_guard"]["full_loop_launch_is_default"] is True
    assert (tmp_path / "job" / "runpod_wam_direct_provider_urls_manifest.json").exists()
    persisted = (tmp_path / "job" / "runpod_wam_async_create_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "bundle-secret" not in persisted
    assert "output-secret" not in persisted


def test_runpod_public_model_disables_secret_forwarding_and_blocks_before_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "wam_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo hi\n")
    calls: list[dict[str, object]] = []

    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
    monkeypatch.setenv(runner.RUNPOD_WAM_DISABLE_WARM_CANDIDATE_ENV, "true")
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: (
            "runpod-secret-not-persisted",
            {"api_key_configured": True, "raw_secret_values_recorded": False},
        ),
    )
    monkeypatch.setattr(
        runner,
        "_read_model_secret_env",
        lambda: (_ for _ in ()).throw(AssertionError("model secret must not be read")),
    )
    monkeypatch.setattr(
        runner,
        "_runpod_request",
        lambda **kwargs: calls.append(dict(kwargs)),
    )

    manifest = runner.create_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        bundle_path=bundle,
        provider_bundle_url="https://spaces.example/bundle.zip",
        provider_output_put_url="https://spaces.example/output.zip",
        allow_paid_runpod_launch=True,
        paid_resource_admission_grant=_paid_grant(),
        skip_public_staging_verification=True,
        forward_model_secret_env=False,
        pre_provider_mutation_hook=lambda: {
            "status": "blocked",
            "blockers": ["prospective_authorization_not_consumed"],
        },
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["provider_mutations_performed"] == 0
    assert manifest["blockers"] == ["prospective_authorization_not_consumed"]
    assert manifest["pre_provider_mutation_hook"]["status"] == "blocked"
    assert calls == []
    assert not (tmp_path / "job" / "runpod_wam_async_state.json").exists()


def test_runpod_public_model_payload_contains_no_account_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "wam_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo hi\n")
    calls: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        calls.append(dict(kwargs))
        return 200, {"id": "pod-public-model"}

    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
    monkeypatch.setenv(runner.RUNPOD_WAM_DISABLE_WARM_CANDIDATE_ENV, "true")
    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: (
            "runpod-secret-not-persisted",
            {"api_key_configured": True, "raw_secret_values_recorded": False},
        ),
    )
    monkeypatch.setattr(
        runner,
        "_read_model_secret_env",
        lambda: (_ for _ in ()).throw(AssertionError("model secret must not be read")),
    )

    manifest = runner.create_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        bundle_path=bundle,
        provider_bundle_url="https://spaces.example/bundle.zip",
        provider_output_put_url="https://spaces.example/output.zip",
        allow_paid_runpod_launch=True,
        paid_resource_admission_grant=_paid_grant(),
        skip_public_staging_verification=True,
        forward_model_secret_env=False,
        pre_provider_mutation_hook=lambda: {"status": "consumed"},
        generated_at="now",
    )

    assert manifest["status"] == "pod_created"
    assert manifest["model_secret_env_status"]["status"] == "disabled"
    assert manifest["pre_provider_mutation_hook"]["status"] == "consumed"
    env = calls[0]["payload"]["env"]
    assert "HF_TOKEN" not in env
    assert "HUGGING_FACE_HUB_TOKEN" not in env


def test_runpod_create_reuses_dynamic_existing_pod_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "wam_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo hi\n")
    calls: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        calls.append(dict(kwargs))
        if kwargs["path"] == "/pods/warm-pod-123/update":
            assert "gpuTypeIds" not in kwargs["payload"]
            assert "gpuCount" not in kwargs["payload"]
            assert kwargs["payload"]["imageName"] == "docker.io/example/wam:20260629"
            assert kwargs["payload"]["env"]["BLUEPRINT_RUNPOD_PROVIDER_BUNDLE_KIND"] == "wam"
            return 200, {"id": "warm-pod-123", "desiredStatus": "EXITED"}
        if kwargs["path"] == "/pods/warm-pod-123/start":
            assert kwargs["payload"] == {}
            return 200, {"id": "warm-pod-123", "desiredStatus": "RUNNING"}
        raise AssertionError(kwargs["path"])

    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
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
        paid_resource_admission_grant=_paid_grant(),
        skip_public_staging_verification=True,
        image_name="docker.io/example/wam:20260629",
        existing_pod_id="warm-pod-123",
        generated_at="now",
    )

    assert [call["path"] for call in calls] == [
        "/pods/warm-pod-123/update",
        "/pods/warm-pod-123/start",
    ]
    assert manifest["status"] == "pod_created"
    assert manifest["pod_id"] == "warm-pod-123"
    assert manifest["pod_launch_mode"] == "existing_pod_start"
    assert manifest["warm_existing_pod"]["requested"] is True
    state = json.loads((tmp_path / "job" / "runpod_wam_async_state.json").read_text())
    assert state["pod_id"] == "warm-pod-123"
    assert state["pod_launch_mode"] == "existing_pod_start"
    persisted = (tmp_path / "job" / "runpod_wam_async_create_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "runpod-secret-not-persisted" not in persisted
    assert "bundle-secret" not in persisted
    assert "output-secret" not in persisted


def test_runpod_create_reuses_recorded_warm_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "wam_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo hi\n")
    warm_candidate_file = tmp_path / "warm_candidate.json"
    warm_candidate_file.write_text(
        json.dumps(
            {
                "schema_version": runner.RUNPOD_WAM_WARM_CANDIDATE_SCHEMA_VERSION,
                "generated_at": "prior",
                "pod_id": "warm-file-pod-123",
                "provider_bundle_kind": "wam",
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
                "stopped_pod_preserved_for_warm_reuse": True,
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        calls.append(dict(kwargs))
        if kwargs["path"] == "/pods/warm-file-pod-123/update":
            return 200, {"id": "warm-file-pod-123", "desiredStatus": "EXITED"}
        if kwargs["path"] == "/pods/warm-file-pod-123/start":
            return 200, {"id": "warm-file-pod-123", "desiredStatus": "RUNNING"}
        raise AssertionError(kwargs["path"])

    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))
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
        provider_bundle_url="https://spaces.example/bundle.zip",
        provider_output_put_url="https://spaces.example/output.zip",
        allow_paid_runpod_launch=True,
        paid_resource_admission_grant=_paid_grant(),
        skip_public_staging_verification=True,
        image_name="docker.io/example/wam:20260629",
        generated_at="now",
    )

    assert [call["path"] for call in calls] == [
        "/pods/warm-file-pod-123/update",
        "/pods/warm-file-pod-123/start",
    ]
    assert manifest["pod_launch_mode"] == "existing_pod_start"
    assert manifest["warm_existing_pod"]["selection_source"] == "dynamic_warm_candidate"
    assert manifest["warm_existing_pod"]["dynamic_warm_candidate"]["status"] == "selected"
    assert (
        manifest["warm_existing_pod"]["dynamic_warm_candidate"]["reuse_kind"]
        == "stopped_warm_candidate"
    )
    assert manifest["warm_existing_pod"]["existing_pod_id"] == "warm-file-pod-123"


def test_warm_candidate_string_false_flags_are_not_preserved_reuse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    warm_candidate_file = tmp_path / "warm_candidate.json"
    warm_candidate_file.write_text(
        json.dumps(
            {
                "schema_version": runner.RUNPOD_WAM_WARM_CANDIDATE_SCHEMA_VERSION,
                "generated_at": "prior",
                "pod_id": "warm-file-pod-123",
                "provider_bundle_kind": "wam",
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
                "running_pod_preserved_for_hot_reuse": "false",
                "stopped_pod_preserved_for_warm_reuse": "false",
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))

    candidate = runner._read_compatible_warm_candidate(
        provider_bundle_kind="wam",
        image_name="docker.io/example/wam:20260629",
        cloud_type="SECURE",
    )

    assert candidate["status"] == "selected"
    assert candidate["reuse_kind"] == "existing_pod_candidate"
    assert candidate["running_pod_preserved_for_hot_reuse"] is False
    assert candidate["stopped_pod_preserved_for_warm_reuse"] is False
    assert (
        candidate["claim_boundary"]["running_hot_candidate_still_uses_update_start_path"] is False
    )
    assert candidate["claim_boundary"]["resident_in_pod_job_queue_not_proven"] is False


def test_runpod_create_falls_back_when_stopped_warm_candidate_cannot_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "wam_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo hi\n")
    warm_candidate_file = tmp_path / "warm_candidate.json"
    warm_candidate_file.write_text(
        json.dumps(
            {
                "schema_version": runner.RUNPOD_WAM_WARM_CANDIDATE_SCHEMA_VERSION,
                "generated_at": "prior",
                "pod_id": "warm-file-pod-123",
                "provider_bundle_kind": "wam",
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
                "stopped_pod_preserved_for_warm_reuse": True,
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        calls.append(dict(kwargs))
        if kwargs["path"] == "/pods/warm-file-pod-123/update":
            return 200, {"id": "warm-file-pod-123", "desiredStatus": "EXITED"}
        if kwargs["path"] == "/pods/warm-file-pod-123/start":
            raise runner.urllib.error.HTTPError(
                url="https://rest.runpod.io/v1/pods/warm-file-pod-123/start",
                code=500,
                msg="capacity",
                hdrs=None,
                fp=io.BytesIO(
                    b'{"error":"start pod: There are not enough free GPUs on the host machine to start this pod."}'
                ),
            )
        if kwargs["path"] == "/pods":
            return 200, {"id": "fresh-pod-456", "desiredStatus": "RUNNING"}
        raise AssertionError(kwargs["path"])

    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))
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
        provider_bundle_url="https://spaces.example/bundle.zip",
        provider_output_put_url="https://spaces.example/output.zip",
        allow_paid_runpod_launch=True,
        paid_resource_admission_grant=_paid_grant(),
        skip_public_staging_verification=True,
        image_name="docker.io/example/wam:20260629",
        generated_at="now",
    )

    assert [call["path"] for call in calls] == [
        "/pods/warm-file-pod-123/update",
        "/pods/warm-file-pod-123/start",
        "/pods",
    ]
    assert manifest["status"] == "pod_created"
    assert manifest["pod_id"] == "fresh-pod-456"
    assert manifest["pod_launch_mode"] == "fresh_pod_create_after_stopped_warm_start_failed"
    warm = manifest["warm_existing_pod"]
    assert warm["stopped_warm_candidate_start_failed"] is True
    assert warm["warm_candidate_retirement"]["status"] == "not_retired"
    assert warm["warm_candidate_retirement"]["reason"] == (
        "stopped_warm_candidate_start_error_may_be_transient"
    )
    assert warm["fallback_fresh_create_attempted"] is True
    assert warm["claim_boundary"]["stopped_warm_candidate_does_not_reserve_gpu_capacity"] is True
    assert json.loads(warm_candidate_file.read_text())["pod_id"] == "warm-file-pod-123"


def test_runpod_create_retires_missing_stopped_warm_candidate_before_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "wam_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo hi\n")
    warm_candidate_file = tmp_path / "warm_candidate.json"
    warm_candidate_file.write_text(
        json.dumps(
            {
                "schema_version": runner.RUNPOD_WAM_WARM_CANDIDATE_SCHEMA_VERSION,
                "generated_at": "prior",
                "pod_id": "missing-warm-pod-123",
                "provider_bundle_kind": "wam",
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
                "stopped_pod_preserved_for_warm_reuse": True,
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )

    def fake_runpod_request(**kwargs):
        if kwargs["path"] == "/pods/missing-warm-pod-123/update":
            raise runner.urllib.error.HTTPError(
                url="https://rest.runpod.io/v1/pods/missing-warm-pod-123/update",
                code=404,
                msg="missing",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"podUpdate: get pod: pod not found","status":404}'),
            )
        if kwargs["path"] == "/pods":
            return 201, {"id": "fresh-after-missing-456", "desiredStatus": "RUNNING"}
        raise AssertionError(kwargs["path"])

    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))
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
        provider_bundle_url="https://spaces.example/bundle.zip",
        provider_output_put_url="https://spaces.example/output.zip",
        allow_paid_runpod_launch=True,
        paid_resource_admission_grant=_paid_grant(),
        skip_public_staging_verification=True,
        image_name="docker.io/example/wam:20260629",
        generated_at="now",
    )

    assert manifest["status"] == "pod_created"
    assert manifest["pod_id"] == "fresh-after-missing-456"
    warm = manifest["warm_existing_pod"]
    assert warm["stopped_warm_candidate_start_failed"] is True
    assert warm["warm_candidate_retirement"]["status"] == "retired"
    retired = json.loads(warm_candidate_file.read_text())
    assert retired["status"] == "retired"
    assert retired["retired_pod_id"] == "missing-warm-pod-123"


def test_runpod_create_labels_running_hot_candidate_reuse_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "wam_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo hi\n")
    warm_candidate_file = tmp_path / "warm_candidate.json"
    warm_candidate_file.write_text(
        json.dumps(
            {
                "schema_version": runner.RUNPOD_WAM_WARM_CANDIDATE_SCHEMA_VERSION,
                "generated_at": "prior",
                "pod_id": "hot-file-pod-123",
                "provider_bundle_kind": "wam",
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
                "source_keepalive_poll_manifest_path": str(tmp_path / "poll.json"),
                "running_pod_preserved_for_hot_reuse": True,
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        calls.append(dict(kwargs))
        if kwargs["path"] == "/pods/hot-file-pod-123":
            return 200, {
                "id": "hot-file-pod-123",
                "desiredStatus": "RUNNING",
                "runtime": {"uptimeInSeconds": 120},
                "publicIp": "198.51.100.10",
            }
        if kwargs["path"] == "/pods/hot-file-pod-123/update":
            return 200, {"id": "hot-file-pod-123", "desiredStatus": "RUNNING"}
        if kwargs["path"] == "/pods/hot-file-pod-123/start":
            return 200, {"id": "hot-file-pod-123", "desiredStatus": "RUNNING"}
        raise AssertionError(kwargs["path"])

    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))
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
        provider_bundle_url="https://spaces.example/bundle.zip",
        provider_output_put_url="https://spaces.example/output.zip",
        allow_paid_runpod_launch=True,
        paid_resource_admission_grant=_paid_grant(),
        skip_public_staging_verification=True,
        image_name="docker.io/example/wam:20260629",
        generated_at="now",
    )

    assert [call["path"] for call in calls] == [
        "/pods/hot-file-pod-123",
        "/pods/hot-file-pod-123/update",
        "/pods/hot-file-pod-123/start",
    ]
    assert manifest["pod_launch_mode"] == "existing_pod_start"
    warm = manifest["warm_existing_pod"]
    assert warm["candidate_reuse_kind"] == "running_hot_candidate"
    assert warm["dynamic_warm_candidate"]["running_pod_preserved_for_hot_reuse"] is True
    assert warm["dynamic_warm_candidate"]["source_keepalive_poll_manifest_path"] == str(
        tmp_path / "poll.json"
    )
    assert warm["claim_boundary"]["existing_pod_id_reused"] is True
    assert warm["claim_boundary"]["existing_pod_update_start_path_used"] is True
    assert warm["claim_boundary"]["running_hot_candidate_still_uses_update_start_path"] is True
    assert warm["claim_boundary"]["resident_in_pod_job_queue_not_proven"] is True


def test_runpod_create_rejects_stale_running_hot_candidate_without_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "wam_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo hi\n")
    warm_candidate_file = tmp_path / "warm_candidate.json"
    warm_candidate_file.write_text(
        json.dumps(
            {
                "schema_version": runner.RUNPOD_WAM_WARM_CANDIDATE_SCHEMA_VERSION,
                "generated_at": "2026-06-30T20:00:00+00:00",
                "pod_id": "stale-hot-pod-123",
                "provider_bundle_kind": "wam",
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
                "source_keepalive_poll_manifest_path": str(tmp_path / "poll.json"),
                "running_pod_preserved_for_hot_reuse": True,
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        calls.append(dict(kwargs))
        if kwargs["path"] == "/pods/stale-hot-pod-123":
            return 200, {
                "id": "stale-hot-pod-123",
                "desiredStatus": "RUNNING",
                "runtime": None,
                "publicIp": "",
            }
        if kwargs["path"] == "/pods":
            return 200, {"id": "fresh-pod-456", "desiredStatus": "RUNNING"}
        raise AssertionError(kwargs["path"])

    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))
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
        provider_bundle_url="https://spaces.example/bundle.zip",
        provider_output_put_url="https://spaces.example/output.zip",
        allow_paid_runpod_launch=True,
        paid_resource_admission_grant=_paid_grant(),
        skip_public_staging_verification=True,
        image_name="docker.io/example/wam:20260629",
        generated_at="2026-06-30T21:00:00+00:00",
    )

    assert [call["path"] for call in calls] == [
        "/pods/stale-hot-pod-123",
        "/pods",
    ]
    assert manifest["status"] == "pod_created"
    assert manifest["pod_id"] == "fresh-pod-456"
    assert manifest["pod_launch_mode"] == "fresh_pod_create"
    warm = manifest["warm_existing_pod"]["dynamic_warm_candidate"]
    assert warm["status"] == "rejected"
    assert warm["reason"] == "running_warm_candidate_runtime_absent_too_long"
    assert warm["reuse_probe"]["runtime_present"] is False
    assert warm["reuse_probe"]["public_ip_present"] is False
    assert warm["reuse_probe"]["candidate_age_seconds"] == 3600.0


def test_runpod_create_ignores_incompatible_warm_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "wam_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo hi\n")
    warm_candidate_file = tmp_path / "warm_candidate.json"
    warm_candidate_file.write_text(
        json.dumps(
            {
                "schema_version": runner.RUNPOD_WAM_WARM_CANDIDATE_SCHEMA_VERSION,
                "generated_at": "prior",
                "pod_id": "warm-file-pod-123",
                "provider_bundle_kind": "wam",
                "image_name": "docker.io/example/other:20260629",
                "cloud_type": "SECURE",
                "stopped_pod_preserved_for_warm_reuse": True,
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        calls.append(dict(kwargs))
        if kwargs["path"] == "/pods":
            return 200, {"id": "fresh-pod-123"}
        raise AssertionError(kwargs["path"])

    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))
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
        provider_bundle_url="https://spaces.example/bundle.zip",
        provider_output_put_url="https://spaces.example/output.zip",
        allow_paid_runpod_launch=True,
        paid_resource_admission_grant=_paid_grant(),
        skip_public_staging_verification=True,
        image_name="docker.io/example/wam:20260629",
        generated_at="now",
    )

    assert [call["path"] for call in calls] == ["/pods"]
    assert manifest["pod_launch_mode"] == "fresh_pod_create"
    assert manifest["warm_existing_pod"]["dynamic_warm_candidate"]["status"] == "incompatible"
    assert (
        manifest["warm_existing_pod"]["dynamic_warm_candidate"]["reason"]
        == "warm_candidate_request_mismatch"
    )


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
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
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
        paid_resource_admission_grant=_paid_grant(),
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
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
                "gpu_type_ids": ["NVIDIA L40S"],
                "container_disk_gb": 240,
                "volume_gb": 120,
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
    warm_candidate_file = tmp_path / "warm_candidate.json"
    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))
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
    stop_manifest = json.loads((job_dir / "runpod_wam_async_stop_manifest.json").read_text())
    assert stop_manifest["warm_candidate"]["status"] == "recorded"
    assert stop_manifest["warm_candidate_path"] == str(warm_candidate_file)
    warm_candidate = json.loads(warm_candidate_file.read_text())
    assert warm_candidate["pod_id"] == "pod-123"
    assert warm_candidate["image_name"] == "docker.io/example/wam:20260629"
    reliability = json.loads((job_dir / "provider_reliability_manifest.json").read_text())
    assert reliability["open_billing_risk"] is True
    teardown = reliability["phase_contracts"]["teardown"]
    assert teardown["residual_billing_possible"] is True
    assert teardown["billing_sweep_recommended"] is True
    assert teardown["billing_sweep_action"]["allocation_id"] == "pod-123"
    assert reliability["billing_sweep_recommended"] is True
    assert reliability["billing_sweep_action"]["allocation_id"] == "pod-123"


def test_provider_reliability_manifest_parses_string_false_spend_flags(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    manifest_path = runner._write_wam_provider_reliability_manifest(
        job_dir=job_dir,
        state={
            "schema_version": runner.RUNPOD_WAM_STATE_SCHEMA_VERSION,
            "generated_at": "prior",
        },
        poll_manifest={
            "pod_id": "pod-123",
            "pod_status": "RUNNING",
            "provider_bundle_kind": "wam",
            "output_zip_present": False,
            "provider_output_terminal": False,
            "teardown_requested": "false",
            "keep_running_on_success": "false",
            "continuing_spend_from_this_run": "false",
        },
        teardown_manifest=None,
        generated_at="now",
    )

    reliability = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    teardown = reliability["phase_contracts"]["teardown"]
    assert reliability["spend"]["continuing_spend_from_this_run"] is False
    assert teardown["keep_alive_requested"] is False
    assert "teardown_unproven:terminate_never_requested" in teardown["blockers"]


def test_runpod_stop_http_error_rechecks_missing_pod_as_spend_released(
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
                "pod_id": "pod-500-then-gone",
                "created_at_epoch": runner.time.time(),
                "output_path": str(output_zip),
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
                "gpu_type_ids": ["NVIDIA L40S"],
                "container_disk_gb": 240,
                "volume_gb": 120,
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["path"] == "/pods/pod-500-then-gone":
            if kwargs["method"] == "GET" and len(requests) == 1:
                return 200, {"desiredStatus": "RUNNING"}
            raise runner.urllib.error.HTTPError(
                url="https://api.runpod.io/v2/pods/pod-500-then-gone",
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=None,
            )
        if kwargs["path"] == "/pods/pod-500-then-gone/stop":
            raise runner.urllib.error.HTTPError(
                url="https://api.runpod.io/v2/pods/pod-500-then-gone/stop",
                code=500,
                msg="Internal Server Error",
                hdrs=None,
                fp=None,
            )
        raise AssertionError(f"unexpected runpod request: {kwargs}")

    monkeypatch.setenv(runner.RUNPOD_WAM_TEARDOWN_ACTION_ENV, "stop")
    warm_candidate_file = tmp_path / "warm_candidate.json"
    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))
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
    stop_manifest = json.loads((job_dir / "runpod_wam_async_stop_manifest.json").read_text())
    assert stop_manifest["status"] == "completed"
    assert stop_manifest["http_status_code"] == 500
    assert stop_manifest["stop_error_verification"]["pod_status"] == "not_found"
    assert stop_manifest["gpu_spend_released_if_provider_honors_stop"] is True
    assert stop_manifest["stopped_volume_storage_may_continue_billing"] is False
    assert stop_manifest["warm_candidate"]["status"] == "not_recorded"
    assert not warm_candidate_file.exists()


def test_runpod_poll_can_keep_successful_pod_running_for_hot_reuse(
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
                "pod_id": "pod-hot-123",
                "created_at_epoch": runner.time.time(),
                "output_path": str(output_zip),
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
                "gpu_type_ids": ["NVIDIA A40", "NVIDIA L40S"],
                "container_disk_gb": 240,
                "volume_gb": 120,
            }
        ),
        encoding="utf-8",
    )
    warm_candidate_file = tmp_path / "warm_candidate.json"
    requests: list[dict[str, object]] = []

    monkeypatch.setenv(runner.RUNPOD_WAM_TEARDOWN_ACTION_ENV, "keep_on_success")
    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["path"] == "/pods/pod-hot-123":
            return 200, {
                "id": "pod-hot-123",
                "desiredStatus": "RUNNING",
                "runtime": {"container": "running"},
                "publicIp": "203.0.113.10",
            }
        raise AssertionError(kwargs["path"])

    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=job_dir,
        max_wait_seconds=1,
        retry_interval_seconds=1,
        teardown=True,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["teardown_action"] == "keep_on_success"
    assert manifest["teardown_performed"] is False
    assert manifest["keep_running_on_success"] is True
    assert manifest["keepalive_runtime_health"]["runtime_healthy_for_hot_reuse"] is True
    assert manifest["keepalive_performed"] is True
    assert manifest["continuing_spend_from_this_run"] is True
    assert [request["path"] for request in requests] == ["/pods/pod-hot-123"]
    assert (job_dir / "runpod_wam_async_keepalive_manifest.json").is_file()
    assert not (job_dir / "runpod_wam_async_stop_manifest.json").exists()
    assert not (job_dir / "runpod_wam_async_delete_manifest.json").exists()
    warm_candidate = json.loads(warm_candidate_file.read_text())
    assert warm_candidate["pod_id"] == "pod-hot-123"
    assert warm_candidate["running_pod_preserved_for_hot_reuse"] is True
    assert warm_candidate["gpu_type_ids"] == ["NVIDIA A40", "NVIDIA L40S"]


def test_runpod_poll_keeps_successful_running_pod_when_runtime_metadata_sparse(
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
                "pod_id": "pod-fake-hot-123",
                "created_at_epoch": runner.time.time(),
                "output_path": str(output_zip),
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
                "gpu_type_ids": ["NVIDIA A40"],
            }
        ),
        encoding="utf-8",
    )
    warm_candidate_file = tmp_path / "warm_candidate.json"
    requests: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["path"] == "/pods/pod-fake-hot-123":
            return 200, {
                "id": "pod-fake-hot-123",
                "desiredStatus": "RUNNING",
                "costPerHr": 0.44,
            }
        raise AssertionError(kwargs["path"])

    monkeypatch.setenv(runner.RUNPOD_WAM_TEARDOWN_ACTION_ENV, "keep_on_success")
    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))
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
    assert manifest["requested_keep_running_on_success"] is True
    assert manifest["keep_running_on_success"] is True
    assert manifest["keepalive_runtime_unhealthy_on_success"] is False
    assert manifest["keepalive_runtime_health"]["runtime_present"] is False
    assert manifest["keepalive_runtime_health"]["active_status_without_runtime_metadata"] is True
    assert (
        manifest["keepalive_runtime_health"]["health_basis"]
        == "active_pod_status_without_runtime_metadata"
    )
    assert manifest["keepalive_runtime_health"]["runtime_healthy_for_hot_reuse"] is True
    assert manifest["teardown_performed"] is False
    assert manifest["keepalive_performed"] is True
    assert manifest["continuing_spend_from_this_run"] is True
    assert [request["path"] for request in requests] == ["/pods/pod-fake-hot-123"]
    assert (job_dir / "runpod_wam_async_keepalive_manifest.json").is_file()
    assert not (job_dir / "runpod_wam_async_stop_manifest.json").exists()
    assert not (job_dir / "runpod_wam_async_delete_manifest.json").exists()
    warm_candidate = json.loads(warm_candidate_file.read_text())
    assert warm_candidate["pod_id"] == "pod-fake-hot-123"
    assert warm_candidate["running_pod_preserved_for_hot_reuse"] is True


def test_runpod_poll_does_not_keep_blocked_output_zip_running(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_dir = tmp_path / "job"
    output_zip = job_dir / "runpod_provider_runtime_output.zip"
    job_dir.mkdir()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "wam_runtime_result.json",
            json.dumps({"status": "blocked", "blockers": ["missing_module"]}),
        )
    (job_dir / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-blocked-123",
                "created_at_epoch": runner.time.time(),
                "output_path": str(output_zip),
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["method"] == "DELETE" and kwargs["path"] == "/pods/pod-blocked-123":
            return 204, {}
        if (
            kwargs["method"] == "GET"
            and kwargs["path"] == "/pods/pod-blocked-123"
            and any(r.get("method") == "DELETE" for r in requests)
        ):
            # Deleted pods answer the post-delete verification probe with 404.
            raise urllib.error.HTTPError(kwargs["path"], 404, "gone", None, None)
        raise AssertionError(f"unexpected runpod request: {kwargs}")

    monkeypatch.setenv(runner.RUNPOD_WAM_TEARDOWN_ACTION_ENV, "keep_on_success")
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
    assert manifest["runtime_output_success"] is False
    assert manifest["keep_running_on_success"] is False
    assert manifest["teardown_performed"] is True
    assert manifest["continuing_spend_from_this_run"] is False
    assert (job_dir / "runpod_wam_async_delete_manifest.json").is_file()
    assert [request["method"] for request in requests] == ["DELETE", "GET"]
    delete_manifest = json.loads((job_dir / "runpod_wam_async_delete_manifest.json").read_text())
    assert delete_manifest["terminal_state_api_confirmed"] is True
    assert delete_manifest["verified_pod_status"] == "not_found"


def test_runpod_poll_deletes_failed_output_even_when_stop_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_dir = tmp_path / "job"
    output_zip = job_dir / "runpod_provider_runtime_output.zip"
    job_dir.mkdir()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "wam_runtime_result.json",
            json.dumps({"status": "blocked", "blockers": ["policy_server_exited"]}),
        )
    (job_dir / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-failed-stop-123",
                "created_at_epoch": runner.time.time(),
                "output_path": str(output_zip),
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
            }
        ),
        encoding="utf-8",
    )
    warm_candidate_file = tmp_path / "warm_candidate.json"
    requests: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["method"] == "DELETE" and kwargs["path"] == "/pods/pod-failed-stop-123":
            return 202, {}
        if (
            kwargs["method"] == "GET"
            and kwargs["path"] == "/pods/pod-failed-stop-123"
            and any(r.get("method") == "DELETE" for r in requests)
        ):
            # Deleted pods answer the post-delete verification probe with 404.
            raise urllib.error.HTTPError(kwargs["path"], 404, "gone", None, None)
        raise AssertionError(f"unexpected runpod request: {kwargs}")

    monkeypatch.setenv(runner.RUNPOD_WAM_TEARDOWN_ACTION_ENV, "stop")
    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))
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
    assert manifest["runtime_output_success"] is False
    assert manifest["auto_teardown_failure"] is True
    assert manifest["teardown_action"] == "delete"
    assert manifest["teardown_performed"] is True
    assert [request["method"] for request in requests] == ["DELETE", "GET"]
    assert [request["path"] for request in requests] == [
        "/pods/pod-failed-stop-123",
        "/pods/pod-failed-stop-123",
    ]
    assert not warm_candidate_file.exists()
    assert not (job_dir / "runpod_wam_async_stop_manifest.json").exists()
    delete_manifest = json.loads((job_dir / "runpod_wam_async_delete_manifest.json").read_text())
    assert delete_manifest["status"] == "completed"
    assert delete_manifest["terminal_state_api_confirmed"] is True
    assert delete_manifest["verified_pod_status"] == "not_found"


def test_runpod_stop_command_stops_running_pod_without_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-running-123",
                "created_at_epoch": runner.time.time(),
                "output_path": str(job_dir / "missing_output.zip"),
                "image_name": "docker.io/example/wam:20260629",
                "cloud_type": "SECURE",
            }
        ),
        encoding="utf-8",
    )
    warm_candidate_file = tmp_path / "warm_candidate.json"
    requests: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["path"] == "/pods/pod-running-123/stop":
            return 200, {"id": "pod-running-123", "desiredStatus": "EXITED"}
        raise AssertionError(f"unexpected runpod request: {kwargs}")

    monkeypatch.setenv(runner.RUNPOD_WAM_WARM_CANDIDATE_FILE_ENV, str(warm_candidate_file))
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)

    manifest = runner.stop_runpod_wam_async_run(job_dir=job_dir, generated_at="now")

    assert manifest["status"] == "completed"
    assert manifest["pod_id"] == "pod-running-123"
    assert manifest["warm_candidate"]["status"] == "recorded"
    assert warm_candidate_file.is_file()
    assert [request["path"] for request in requests] == ["/pods/pod-running-123/stop"]


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


def test_runpod_poll_blocks_downloaded_zip_without_provider_result_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    output_get_url_file.write_text(
        "https://spaces.example/output.zip?X-Amz-Signature=download-secret\n",
        encoding="utf-8",
    )
    output_get_url_file.chmod(0o600)
    job_dir = tmp_path / "job"
    output_zip = job_dir / "runpod_provider_runtime_output.zip"
    job_dir.mkdir()
    downloaded_zip = tmp_path / "downloaded.zip"
    with zipfile.ZipFile(downloaded_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("logs/provider.log", "entrypoint exited without a result manifest")
    (job_dir / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-missing-result",
                "output_path": str(output_zip),
                "provider_output_get_url_file": {
                    "path": str(output_get_url_file),
                    "raw_secret_values_recorded": False,
                },
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, object]] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return downloaded_zip.read_bytes()

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["method"] == "DELETE" and kwargs["path"] == "/pods/pod-missing-result":
            return 202, {}
        if (
            kwargs["method"] == "GET"
            and kwargs["path"] == "/pods/pod-missing-result"
            and any(r.get("method") == "DELETE" for r in requests)
        ):
            # Deleted pods answer the post-delete verification probe with 404.
            raise urllib.error.HTTPError(kwargs["path"], 404, "gone", None, None)
        raise AssertionError(f"unexpected runpod request: {kwargs}")

    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=job_dir,
        max_wait_seconds=1,
        retry_interval_seconds=1,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["output_zip_present"] is True
    assert manifest["provider_output_terminal"] is False
    assert manifest["provider_output_usable"] is False
    assert manifest["provider_runtime_operational"] is False
    assert manifest["runtime_output_success"] is False
    assert manifest["provider_command_status"] == "blocked"
    assert "provider_runtime_result_manifest_missing" in manifest["provider_command_blockers"]
    assert manifest["provider_output_validation_status"] == "blocked"
    assert manifest["teardown_performed"] is True
    assert [request["method"] for request in requests] == ["DELETE", "GET"]
    assert [request["path"] for request in requests] == [
        "/pods/pod-missing-result",
        "/pods/pod-missing-result",
    ]
    delete_manifest = json.loads((job_dir / "runpod_wam_async_delete_manifest.json").read_text())
    assert delete_manifest["terminal_state_api_confirmed"] is True
    assert delete_manifest["verified_pod_status"] == "not_found"
    assert output_zip.is_file()


def test_runpod_poll_blocks_malformed_provider_result_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_dir = tmp_path / "job"
    output_zip = job_dir / "runpod_provider_runtime_output.zip"
    job_dir.mkdir()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("wam_runtime_result.json", "{not-json")
    (job_dir / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-malformed-result",
                "output_path": str(output_zip),
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["method"] == "DELETE" and kwargs["path"] == "/pods/pod-malformed-result":
            return 202, {}
        if (
            kwargs["method"] == "GET"
            and kwargs["path"] == "/pods/pod-malformed-result"
            and any(r.get("method") == "DELETE" for r in requests)
        ):
            # Deleted pods answer the post-delete verification probe with 404.
            raise urllib.error.HTTPError(kwargs["path"], 404, "gone", None, None)
        raise AssertionError(f"unexpected runpod request: {kwargs}")

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
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["output_zip_present"] is True
    assert manifest["provider_output_terminal"] is False
    assert manifest["provider_output_usable"] is False
    assert manifest["provider_command_status"] == "blocked"
    assert any(
        blocker.startswith("provider_output_manifest_malformed:wam_runtime_result.json")
        for blocker in manifest["provider_command_blockers"]
    )
    assert manifest["provider_runtime_operational"] is False
    assert manifest["teardown_performed"] is True
    assert [request["method"] for request in requests] == ["DELETE", "GET"]
    assert [request["path"] for request in requests] == [
        "/pods/pod-malformed-result",
        "/pods/pod-malformed-result",
    ]
    delete_manifest = json.loads((job_dir / "runpod_wam_async_delete_manifest.json").read_text())
    assert delete_manifest["terminal_state_api_confirmed"] is True
    assert delete_manifest["verified_pod_status"] == "not_found"


def test_runpod_poll_treats_entrypoint_failure_with_stale_heartbeat_as_terminal_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_dir = tmp_path / "job"
    output_zip = job_dir / "runpod_provider_runtime_output.zip"
    job_dir.mkdir()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "wam_provider_output.json",
            json.dumps({"schema_version": "wam_provider_output.v1", "status": "running"}),
        )
        archive.writestr(
            "runpod_wam_provider_entrypoint_execution.json",
            json.dumps(
                {
                    "schema_version": "runpod_wam_provider_entrypoint_execution.v1",
                    "status": "blocked",
                    "returncode": 1,
                    "blockers": ["runpod_wam_provider_entrypoint_nonzero_or_timeout"],
                    "raw_secret_values_recorded": False,
                }
            ),
        )
    (job_dir / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-entrypoint-failed",
                "output_path": str(output_zip),
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["method"] == "DELETE" and kwargs["path"] == "/pods/pod-entrypoint-failed":
            return 202, {}
        if (
            kwargs["method"] == "GET"
            and kwargs["path"] == "/pods/pod-entrypoint-failed"
            and any(r.get("method") == "DELETE" for r in requests)
        ):
            # Deleted pods answer the post-delete verification probe with 404.
            raise urllib.error.HTTPError(kwargs["path"], 404, "gone", None, None)
        raise AssertionError(f"unexpected runpod request: {kwargs}")

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
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["output_zip_present"] is True
    assert manifest["provider_output_terminal"] is True
    assert manifest["provider_output_usable"] is False
    assert manifest["provider_output_validation_status"] == "completed"
    assert manifest["runtime_result_status"] == "blocked"
    assert manifest["provider_command_status"] == "blocked"
    assert (
        "runpod_wam_provider_entrypoint_nonzero_or_timeout"
        in (manifest["provider_command_blockers"])
    )
    assert manifest["provider_runtime_operational"] is False
    assert manifest["runtime_output_success"] is False
    assert manifest["teardown_performed"] is True
    assert [request["method"] for request in requests] == ["DELETE", "GET"]
    assert [request["path"] for request in requests] == [
        "/pods/pod-entrypoint-failed",
        "/pods/pod-entrypoint-failed",
    ]
    delete_manifest = json.loads((job_dir / "runpod_wam_async_delete_manifest.json").read_text())
    assert delete_manifest["terminal_state_api_confirmed"] is True
    assert delete_manifest["verified_pod_status"] == "not_found"


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
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
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
        paid_resource_admission_grant=_paid_grant(),
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
    assert (
        env["BLUEPRINT_UNITREE_UNIFOLM_COMMAND"]
        == "/usr/local/bin/run_unitree_unifolm_vla_policy_once"
    )
    assert env["BLUEPRINT_UNITREE_UNIFOLM_CHECKPOINT"] == "unitreerobotics/UnifoLM-VLA-Base"
    assert env["BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT"] == "unitreerobotics/UnifoLM-VLM-Base"
    script = payload["dockerStartCmd"][0]
    assert "run_unitree_unifolm_provider_runtime.sh" in script
    assert "run_wam_provider_runtime.sh" not in script
    persisted = (tmp_path / "job" / "runpod_wam_async_create_manifest.json").read_text(
        encoding="utf-8"
    )
    state = (tmp_path / "job" / "runpod_wam_async_state.json").read_text(encoding="utf-8")
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
        archive.writestr(
            "wam_provider_output.json",
            json.dumps(
                {
                    "schema_version": "wam_provider_output.v1",
                    "status": "running",
                    "runtime_phase": "runpod_wam_entrypoint_starting",
                    "blockers": [],
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


def test_runpod_poll_accepts_completed_policy_output_with_stale_wam_heartbeat(
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
    completed_zip = tmp_path / "completed.zip"
    with zipfile.ZipFile(completed_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "completed",
                    "blockers": [],
                    "generated_next_observation_count": 4,
                    "live_wam_generation_success_count": 4,
                    "learned_wam_model_success_count": 4,
                    "repeated_policy_calls_count": 5,
                    "unitree_groot_n17_sonic_model_executed": True,
                }
            ),
        )
        archive.writestr(
            "wam_provider_output.json",
            json.dumps(
                {
                    "schema_version": "wam_provider_output.v1",
                    "status": "running",
                    "runtime_phase": "runpod_wam_entrypoint_starting",
                    "blockers": [],
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

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return completed_zip.read_bytes()

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
    assert manifest["runtime_result_status"] == "completed"
    assert manifest["output_zip_present"] is True
    assert manifest["last_nonterminal_output"] is None


def test_runpod_poll_tolerates_transient_pod_status_url_error(
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
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "policy_infer_started",
                    "blockers": [],
                }
            ),
        )
    with zipfile.ZipFile(completed_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "completed",
                    "blockers": [],
                    "generated_next_observation_count": 1,
                    "live_wam_generation_success_count": 1,
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
        lambda **kwargs: (_ for _ in ()).throw(
            runner.urllib.error.URLError("temporary dns failure")
        ),
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
    assert manifest["pod_status"] == "status_probe_error"
    assert manifest["pod_status_transient_error_count"] == 1
    assert manifest["last_pod_status_error"]["error_type"] == "URLError"
    assert "download-secret" not in (
        tmp_path / "job" / "runpod_wam_async_poll_manifest.json"
    ).read_text(encoding="utf-8")


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
    download_manifest = json.loads(
        (tmp_path / "job" / "runpod_wam_output_download_manifest.json").read_text(encoding="utf-8")
    )
    assert download_manifest["status"] == "nonterminal"
    assert download_manifest["terminal_output_present"] is False
    assert download_manifest["nonterminal_runtime_result_status"] == "running"


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


def test_runpod_poll_deletes_active_pod_after_startup_heartbeat_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_dir = tmp_path / "job"
    output_zip = job_dir / "runpod_provider_runtime_output.zip"
    job_dir.mkdir()
    (job_dir / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-startup-stalled",
                "output_path": str(output_zip),
                "created_at_epoch": runner.time.time(),
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, object]] = []
    monotonic_values = iter([0.0, 0.0, 2.0])

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["method"] == "DELETE" and kwargs["path"] == "/pods/pod-startup-stalled":
            return 202, {}
        if (
            kwargs["method"] == "GET"
            and kwargs["path"] == "/pods/pod-startup-stalled"
            and any(r.get("method") == "DELETE" for r in requests)
        ):
            # Deleted pods answer the post-delete verification probe with 404.
            raise urllib.error.HTTPError(kwargs["path"], 404, "gone", None, None)
        if kwargs["path"] == "/pods/pod-startup-stalled":
            return 200, {"desiredStatus": "RUNNING"}
        raise AssertionError(f"unexpected runpod request: {kwargs}")

    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values, 2.0))

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=job_dir,
        max_wait_seconds=100,
        retry_interval_seconds=10,
        teardown=True,
        post_marker_no_progress_timeout_seconds=1,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["runtime_stall_observed"] is True
    assert manifest["stall_teardown_requested"] is True
    assert manifest["stall_evaluation"]["stall_mode"] == "container_startup"
    assert manifest["teardown_performed"] is True
    assert manifest["continuing_spend_from_this_run"] is False
    assert [request["path"] for request in requests] == [
        "/pods/pod-startup-stalled",
        "/pods/pod-startup-stalled",
        # Post-delete state probe: teardown proof requires API confirmation.
        "/pods/pod-startup-stalled",
    ]
    reliability = json.loads((job_dir / "provider_reliability_manifest.json").read_text())
    assert reliability["failed_phase"] == "container_startup"
    assert reliability["open_billing_risk"] is False
    assert (job_dir / "runpod_wam_async_delete_manifest.json").is_file()


def test_runpod_poll_deletes_stalled_pod_without_teardown_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_dir = tmp_path / "job"
    output_zip = job_dir / "runpod_provider_runtime_output.zip"
    job_dir.mkdir()
    (job_dir / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-auto-stalled",
                "output_path": str(output_zip),
                "created_at_epoch": runner.time.time(),
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, object]] = []
    monotonic_values = iter([0.0, 0.0, 2.0])

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["method"] == "DELETE" and kwargs["path"] == "/pods/pod-auto-stalled":
            return 202, {}
        if (
            kwargs["method"] == "GET"
            and kwargs["path"] == "/pods/pod-auto-stalled"
            and any(r.get("method") == "DELETE" for r in requests)
        ):
            # Deleted pods answer the post-delete verification probe with 404.
            raise urllib.error.HTTPError(kwargs["path"], 404, "gone", None, None)
        if kwargs["path"] == "/pods/pod-auto-stalled":
            return 200, {"desiredStatus": "RUNNING"}
        raise AssertionError(f"unexpected runpod request: {kwargs}")

    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values, 2.0))

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=job_dir,
        max_wait_seconds=100,
        retry_interval_seconds=10,
        teardown=False,
        post_marker_no_progress_timeout_seconds=1,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["runtime_stall_observed"] is True
    assert manifest["auto_teardown_failure"] is True
    assert manifest["teardown_requested"] is True
    assert manifest["teardown_action"] == "delete"
    assert manifest["teardown_performed"] is True
    assert manifest["continuing_spend_from_this_run"] is False
    assert [request["path"] for request in requests] == [
        "/pods/pod-auto-stalled",
        "/pods/pod-auto-stalled",
        # Post-delete state probe: teardown proof requires API confirmation.
        "/pods/pod-auto-stalled",
    ]
    reliability = json.loads((job_dir / "provider_reliability_manifest.json").read_text())
    assert reliability["failed_phase"] == "container_startup"
    assert reliability["open_billing_risk"] is False


def test_runpod_poll_deletes_pod_after_nonterminal_heartbeat_stalls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    output_get_url_file.write_text(
        "https://spaces.example/persistent-output.zip?X-Amz-Signature=download-secret\n",
        encoding="utf-8",
    )
    output_get_url_file.chmod(0o600)
    job_dir = tmp_path / "job"
    output_zip = job_dir / "runpod_provider_runtime_output.zip"
    job_dir.mkdir()
    running_zip = tmp_path / "running.zip"
    with zipfile.ZipFile(running_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("wam_provider_output.json", json.dumps({"status": "running"}))
    (job_dir / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-runtime-stalled",
                "output_path": str(output_zip),
                "provider_output_get_url_file": {
                    "path": str(output_get_url_file),
                    "raw_secret_values_recorded": False,
                },
                "created_at_epoch": runner.time.time(),
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, object]] = []
    monotonic_values = iter([0.0, 0.0, 0.0, 0.0, 2.0])

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return running_zip.read_bytes()

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["method"] == "DELETE" and kwargs["path"] == "/pods/pod-runtime-stalled":
            return 202, {}
        if (
            kwargs["method"] == "GET"
            and kwargs["path"] == "/pods/pod-runtime-stalled"
            and any(r.get("method") == "DELETE" for r in requests)
        ):
            # Deleted pods answer the post-delete verification probe with 404.
            raise urllib.error.HTTPError(kwargs["path"], 404, "gone", None, None)
        if kwargs["path"] == "/pods/pod-runtime-stalled":
            return 200, {"desiredStatus": "RUNNING"}
        raise AssertionError(f"unexpected runpod request: {kwargs}")

    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: ("runpod-secret-not-persisted", {"raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values, 2.0))

    manifest = runner.poll_runpod_wam_async_run(
        job_dir=job_dir,
        max_wait_seconds=100,
        retry_interval_seconds=10,
        teardown=True,
        post_marker_no_progress_timeout_seconds=1,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["runtime_stall_observed"] is True
    assert manifest["stall_evaluation"]["stall_mode"] == "runtime_execution"
    assert manifest["last_nonterminal_output"]["runtime_result_status"] == "running"
    assert manifest["teardown_performed"] is True
    assert manifest["continuing_spend_from_this_run"] is False
    assert [request["path"] for request in requests] == [
        "/pods/pod-runtime-stalled",
        "/pods/pod-runtime-stalled",
        # Post-delete state probe: teardown proof requires API confirmation.
        "/pods/pod-runtime-stalled",
    ]
    reliability = json.loads((job_dir / "provider_reliability_manifest.json").read_text())
    assert reliability["failed_phase"] == "runtime_execution"
    assert any(
        blocker.startswith("post_marker_no_progress:")
        for blocker in reliability["failure_blockers"]
    )
    assert reliability["open_billing_risk"] is False


def test_runpod_poll_deletes_blocked_runtime_output_without_teardown_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_dir = tmp_path / "job"
    output_zip = job_dir / "runpod_provider_runtime_output.zip"
    job_dir.mkdir()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "wam_runtime_result.json",
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": ["runpod_wam_outer_bootstrap_failed_before_runtime_result"],
                }
            ),
        )
    (job_dir / "runpod_wam_async_state.json").write_text(
        json.dumps(
            {
                "provider_bundle_kind": "wam",
                "pod_id": "pod-blocked-output",
                "output_path": str(output_zip),
                "created_at_epoch": runner.time.time(),
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        requests.append(dict(kwargs))
        if kwargs["method"] == "DELETE" and kwargs["path"] == "/pods/pod-blocked-output":
            return 202, {}
        if (
            kwargs["method"] == "GET"
            and kwargs["path"] == "/pods/pod-blocked-output"
            and any(r.get("method") == "DELETE" for r in requests)
        ):
            # Deleted pods answer the post-delete verification probe with 404.
            raise urllib.error.HTTPError(kwargs["path"], 404, "gone", None, None)
        raise AssertionError(f"unexpected runpod request: {kwargs}")

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
        teardown=False,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["runtime_output_success"] is False
    assert manifest["auto_teardown_failure"] is True
    assert manifest["teardown_requested"] is True
    assert manifest["teardown_action"] == "delete"
    assert manifest["teardown_performed"] is True
    assert manifest["continuing_spend_from_this_run"] is False
    assert [request["path"] for request in requests] == [
        "/pods/pod-blocked-output",
        # Post-delete state probe: teardown proof requires API confirmation.
        "/pods/pod-blocked-output",
    ]
    reliability = json.loads((job_dir / "provider_reliability_manifest.json").read_text())
    assert reliability["failed_phase"] == "runtime_execution"
    assert "runner_failed:blocked" in reliability["failure_blockers"]
    assert reliability["open_billing_risk"] is False
