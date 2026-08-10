from __future__ import annotations

import fcntl
import json
import os
import shlex
import subprocess
import urllib.error
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

import blueprint_pipeline.vast_provider_adapter as vpa
import blueprint_pipeline.vast_cuda_runtime_probe as vcrp
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from blueprint_pipeline.vast_provider_adapter import (
    DEFAULT_ISAAC_IMAGE,
    DEFAULT_ISAAC_DISK_GB,
    VAST_API_GATE_ENV,
    VAST_API_KEY_FILE_ENV,
    VAST_INSTANCE_LAUNCH_GATE_ENV,
    _redact_text,
    _offer_selection_manifest,
    _offer_summary,
    _select_offer,
    _url_secret_values,
    run_vast_provider_adapter,
)


pytestmark = pytest.mark.slow


def test_gpu_sanity_requires_container_cuda_runtime_for_paid_bundles() -> None:
    incompatible = vcrp.gpu_sanity_from_log(
        "\n".join(
            [
                "BLUEPRINT_VAST_GPU_SANITY_OK",
                "BLUEPRINT_VAST_CUDA_RUNTIME_BLOCKED:cudaGetDeviceCount:803",
                "BLUEPRINT_VAST_CUDA_RUNTIME_EXIT_CODE:3",
            ]
        ),
        require_cuda_runtime=True,
    )
    assert incompatible["nvidia_smi_ok"] is True
    assert incompatible["gpu_ok"] is False
    assert incompatible["blockers"] == ["vast_cuda_runtime_host_image_incompatible"]

    compatible = vcrp.gpu_sanity_from_log(
        "\n".join(
            [
                "BLUEPRINT_VAST_GPU_SANITY_OK",
                "BLUEPRINT_VAST_CUDA_RUNTIME_API_OK:devices=1",
                "BLUEPRINT_VAST_CUDA_RUNTIME_OK",
            ]
        ),
        require_cuda_runtime=True,
    )
    assert compatible["gpu_ok"] is True
    assert compatible["blockers"] == []


def test_retention_requires_healthy_host_and_armed_watchdog() -> None:
    decision = vpa._retention_decision(
        requested=True,
        watchdog_handoff={
            "status": "armed",
            "independent_process": True,
            "watchdog_armed_before_allocation": True,
            "watchdog_pid": 123,
            "watchdog_deadline_epoch": 2_000.0,
        },
        instance_ids=[456],
        startup_probe={"status": "completed", "startup_probe_proven": True},
        gpu_sanity={"status": "completed", "gpu_sanity_proven": True},
        video_smoke={
            "status": "blocked",
            "cosmos_server_loaded": True,
            "cosmos_runtime_status": "blocked",
        },
        observed_now_epoch=1_000.0,
    )

    assert decision["status"] == "retained_owned"
    assert decision["blockers"] == []


def test_lifecycle_record_failure_blocks_result_without_raising() -> None:
    result: dict[str, object] = {"status": "completed", "blockers": ["prior_blocker"]}

    recorded = vpa._record_lifecycle_or_block(
        result,
        operation="terminal",
        recorder=lambda: (_ for _ in ()).throw(ValueError("sensitive details omitted")),
    )

    assert recorded is False
    assert result["status"] == "failed"
    assert result["reason"] == "retained_gpu_lifecycle_record_failed"
    assert result["blockers"] == [
        "prior_blocker",
        "retained_gpu_lifecycle_terminal_record_failed:ValueError",
    ]
    assert "sensitive details omitted" not in json.dumps(result)


def test_retention_reads_loaded_nonterminal_cosmos_evidence_from_output_zip(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "runtime-output.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "cosmos_server_retention.json",
            json.dumps(
                {
                    "status": "retained_loaded",
                    "process_alive": True,
                    "server_remained_loaded": True,
                }
            ),
        )
        archive.writestr("wam_runtime_result.json", json.dumps({"status": "blocked"}))

    decision = vpa._retention_decision(
        requested=True,
        watchdog_handoff={
            "status": "armed",
            "independent_process": True,
            "watchdog_armed_before_allocation": True,
            "watchdog_pid": 123,
            "watchdog_deadline_epoch": 2_000.0,
        },
        instance_ids=[456],
        startup_probe={"status": "completed", "startup_probe_proven": True},
        gpu_sanity={"status": "completed", "gpu_sanity_proven": True},
        video_smoke={"provider_runtime_output_zip_path": str(archive_path)},
        observed_now_epoch=1_000.0,
    )

    assert decision["status"] == "retained_owned"
    assert decision["cosmos_server_loaded"] is True
    assert decision["cosmos_runtime_status"] == "blocked"


@pytest.mark.parametrize(
    ("watchdog", "startup", "gpu", "video", "expected_blocker"),
    [
        (
            {},
            {"status": "completed", "startup_probe_proven": True},
            {"status": "completed", "gpu_sanity_proven": True},
            {"status": "blocked", "cosmos_server_loaded": True, "cosmos_runtime_status": "blocked"},
            "retention_independent_watchdog_not_armed",
        ),
        (
            {
                "status": "armed",
                "independent_process": True,
                "watchdog_armed_before_allocation": True,
                "watchdog_deadline_epoch": 2_000.0,
            },
            {},
            {"status": "completed", "gpu_sanity_proven": True},
            {"status": "blocked", "cosmos_server_loaded": True, "cosmos_runtime_status": "blocked"},
            "retention_container_health_not_proven",
        ),
        (
            {
                "status": "armed",
                "independent_process": True,
                "watchdog_armed_before_allocation": True,
                "watchdog_deadline_epoch": 2_000.0,
            },
            {"status": "completed", "startup_probe_proven": True},
            {},
            {"status": "blocked", "cosmos_server_loaded": True, "cosmos_runtime_status": "blocked"},
            "retention_gpu_health_not_proven",
        ),
        (
            {
                "status": "armed",
                "independent_process": True,
                "watchdog_armed_before_allocation": True,
                "watchdog_deadline_epoch": 2_000.0,
            },
            {"status": "completed", "startup_probe_proven": True},
            {"status": "completed", "gpu_sanity_proven": True},
            {
                "status": "completed",
                "cosmos_server_loaded": True,
                "cosmos_runtime_status": "completed",
            },
            "retention_not_needed_after_terminal_bundle_success",
        ),
    ],
)
def test_retention_fails_closed(
    watchdog: dict[str, object],
    startup: dict[str, object],
    gpu: dict[str, object],
    video: dict[str, object],
    expected_blocker: str,
) -> None:
    decision = vpa._retention_decision(
        requested=True,
        watchdog_handoff=watchdog,
        instance_ids=[456],
        startup_probe=startup,
        gpu_sanity=gpu,
        video_smoke=video,
        observed_now_epoch=1_000.0,
    )

    assert decision["status"] == "teardown_required"
    assert expected_blocker in decision["blockers"]


def test_args_log_hold_wraps_terminal_heredoc_and_preserves_probe_rc() -> None:
    probe = """set -e
python3 - <<'PY'
raise SystemExit(7)
PY
"""
    payload = vpa._create_payload(
        image="image",
        label="heredoc-probe",
        launch_mode="args",
        probe_script=probe,
        disk_gb=20,
    )
    command = shlex.split(payload["args_str"])

    assert command[:2] == ["bash", "-lc"]
    syntax = subprocess.run(
        ["bash", "-n", "-c", command[2]],
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr.decode("utf-8", errors="replace")

    executed = subprocess.run(
        command,
        env={**os.environ, "BLUEPRINT_VAST_ARGS_LOG_HOLD_SECONDS": "0"},
        capture_output=True,
        check=False,
        text=True,
    )
    assert executed.returncode == 7
    assert "BLUEPRINT_VAST_ARGS_LOG_HOLD_STARTED" in executed.stdout
    assert "BLUEPRINT_VAST_ARGS_LOG_HOLD_DONE" in executed.stdout
    assert "syntax error" not in executed.stderr.lower()


def _paid_grant():
    admission = build_paid_lane_admission(
        resource_class="vast_provider_adapter",
        blockers=[],
    )
    return require_paid_resource_admission(
        admission,
        resource_class="vast_provider_adapter",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


@pytest.fixture(autouse=True)
def _isolate_vast_launch_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(vpa.VAST_LAUNCH_LOCK_FILE_ENV, str(tmp_path / "vast_paid_launch.lock"))
    monkeypatch.delenv(vpa.VAST_WAM_MIN_GPU_RAM_MB_ENV, raising=False)
    monkeypatch.delenv(vpa.VAST_MIN_COMPUTE_CAP_ENV, raising=False)
    monkeypatch.delenv(vpa.VAST_IMAGE_LOGIN_MODE_ENV, raising=False)
    monkeypatch.delenv(vpa.VAST_HEARTBEAT_NO_PROGRESS_SECONDS_ENV, raising=False)
    for env_name in vpa.HF_TOKEN_FILE_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv(vpa.HF_TOKEN_FILE_ENV, str(tmp_path / "missing_hf_token"))
    for env_name in (
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT",
        "BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(env_name, raising=False)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_secret(path: Path, value: str = "secret-vast-key") -> None:
    path.write_text(value + "\n", encoding="utf-8")
    path.chmod(0o600)


def _configure_live_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    secret = "secret-vast-key"
    key_file = tmp_path / "vast_api_key"
    _write_secret(key_file, secret)
    monkeypatch.setenv(VAST_API_KEY_FILE_ENV, str(key_file))
    monkeypatch.setenv(VAST_API_GATE_ENV, "true")
    monkeypatch.setenv(VAST_INSTANCE_LAUNCH_GATE_ENV, "true")
    return secret


def test_vast_launch_lock_path_defaults_beside_api_key_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(vpa.VAST_LAUNCH_LOCK_FILE_ENV, raising=False)
    key_file = tmp_path / "secrets" / "vast_api_key"
    key_file.parent.mkdir()
    _write_secret(key_file)
    monkeypatch.setenv(VAST_API_KEY_FILE_ENV, str(key_file))

    assert vpa._vast_launch_lock_path() == key_file.parent / "vast_paid_launch.lock"


def test_vast_session_budget_ledger_path_defaults_beside_api_key_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(vpa.VAST_SESSION_BUDGET_LEDGER_FILE_ENV, raising=False)
    key_file = tmp_path / "secrets" / "vast_api_key"
    key_file.parent.mkdir()
    _write_secret(key_file)
    monkeypatch.setenv(VAST_API_KEY_FILE_ENV, str(key_file))

    assert (
        vpa._vast_session_budget_ledger_path() == key_file.parent / "vast_session_cost_summary.json"
    )
    explicit = tmp_path / "explicit-session-cost.json"
    monkeypatch.setenv(vpa.VAST_SESSION_BUDGET_LEDGER_FILE_ENV, str(explicit))

    assert vpa._vast_session_budget_ledger_path() == explicit


def _write_valid_provider_bundle(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "provider_runtime/run_isaac_realistic_runtime.sh",
            "write_missing_result\n"
            "isaac_runner_process_exited_without_runtime_result\n"
            "blocked_isaac_process_exited_without_result\n",
        )
        archive.writestr(
            "provider_runtime/isaac_realistic_runtime_runner.py",
            "from isaacsim import SimulationApp\n"
            "controller_grade_execution_proven = False\n"
            "official_policy_execution_proven = False\n"
            "generated_world_rank_fidelity_result_proven = False\n"
            "generated_world_policy_evaluation_scope_proven = False\n",
        )
        for name in (
            "provider_runtime/isaac_provider_eval_manifest.json",
            "provider_runtime/scenario_eval_matrix.json",
            "provider_runtime/camera_manifest.json",
            "provider_runtime/episode_spec_manifest.json",
        ):
            archive.writestr(name, "{}\n")
        archive.writestr("provider_runtime/generated_site_scene.usda", "#usda 1.0\n")
        archive.writestr("provider_runtime/generated_site_scene.usd", b"USD")


def _write_valid_wam_provider_bundle(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "provider_runtime/run_wam_provider_runtime.sh",
            "write_missing_result\n"
            "wam_runner_process_exited_without_runtime_result\n"
            "blocked_wam_process_exited_without_result\n",
        )
        archive.writestr(
            "provider_runtime/wam_provider_runtime_runner.py",
            "model_name = 'OSCAR-2B'\n"
            "output = 'wam_runtime_result.json'\n"
            "action_conditioned_video_rollout_generated = True\n",
        )
        archive.writestr("provider_runtime/wam_provider_runtime_manifest.json", "{}\n")
        archive.writestr("provider_runtime/wam_rollout_input_manifest.json", "{}\n")
        archive.writestr("provider_runtime/oscar_input/first_frame.png", b"png")
        archive.writestr(
            "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4",
            b"mp4",
        )


def _write_valid_unitree_unifolm_provider_bundle(path: Path) -> None:
    readiness = {
        "schema_version": "unitree_unifolm_policy_provider_bundle.v1",
        "local_bundle_ready_for_remote_staging": True,
        "ready_for_fresh_model_execution": True,
        "runtime_execution_blockers": [],
    }
    readiness_path = (
        path.parent / "provider_runtime" / "unitree_unifolm_policy_provider_manifest.json"
    )
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "provider_runtime/run_unitree_unifolm_provider_runtime.sh",
            "unitree_unifolm_provider_runner_failed_without_runtime_result\n"
            "blocked_unitree_unifolm_process_exited_without_result\n",
        )
        archive.writestr(
            "provider_runtime/unitree_unifolm_provider_runner.py",
            "unitree_unifolm_policy_provider_output.json\n"
            "unitree_unifolm_model_executed = False\n"
            "unitree_unifolm_policy_action_command_ran = False\n",
        )
        archive.writestr(
            "provider_runtime/unitree_unifolm_policy_provider_manifest.json",
            json.dumps(readiness),
        )
        archive.writestr("provider_runtime/policy_input.json", "{}\n")
        archive.writestr("provider_runtime/input_frame.png", b"png")
        archive.writestr("provider_runtime/blueprint_pipeline/__init__.py", "")
        archive.writestr(
            "provider_runtime/blueprint_pipeline/unitree_unifolm_policy_command_adapter.py",
            "# bundled adapter\n",
        )
        archive.writestr(
            "provider_runtime/blueprint_pipeline/unitree_unifolm_vla_server_bridge.py",
            "# bundled bridge\n",
        )


def _write_valid_unitree_groot_n17_sonic_provider_bundle(path: Path) -> None:
    readiness = {
        "schema_version": "unitree_groot_n17_sonic_policy_provider_bundle.v1",
        "local_bundle_ready_for_remote_staging": True,
        "ready_for_fresh_model_execution": True,
        "runtime_execution_blockers": [],
    }
    readiness_path = (
        path.parent / "provider_runtime" / "unitree_groot_n17_sonic_policy_provider_manifest.json"
    )
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh",
            "unitree_groot_n17_sonic_provider_runner_failed_without_runtime_result\n"
            "blocked_unitree_groot_n17_sonic_process_exited_without_result\n",
        )
        archive.writestr(
            "provider_runtime/unitree_groot_n17_sonic_provider_runner.py",
            "unitree_groot_n17_sonic_policy_provider_output.json\n"
            "unitree_groot_n17_sonic_model_executed = False\n"
            "unitree_groot_n17_sonic_policy_action_command_ran = False\n",
        )
        archive.writestr(
            "provider_runtime/unitree_groot_n17_sonic_policy_provider_manifest.json",
            json.dumps(readiness),
        )
        archive.writestr("provider_runtime/policy_input.json", "{}\n")
        archive.writestr("provider_runtime/input_frame.png", b"png")
        archive.writestr("provider_runtime/blueprint_pipeline/__init__.py", "")
        archive.writestr("provider_runtime/blueprint_pipeline/common.py", "# bundled common\n")
        archive.writestr(
            "provider_runtime/blueprint_pipeline/unitree_groot_n17_sonic_policy_command_adapter.py",
            "# bundled adapter\n",
        )
        archive.writestr(
            "provider_runtime/blueprint_pipeline/unitree_groot_n17_sonic_policy_runtime.py",
            "# bundled runtime\n",
        )
        archive.writestr(
            "provider_runtime/blueprint_pipeline/unitree_groot_n17_sonic_policy_server_command.py",
            "# bundled server command\n",
        )


def test_blocked_phase_artifacts_refresh_stale_video_smoke_result(tmp_path: Path) -> None:
    stale_video_path = tmp_path / "vast_video_smoke_result.json"
    stale_video_path.write_text(
        json.dumps(
            {
                "schema_version": "vast_video_smoke_result.v1",
                "generated_at": "old-run",
                "status": "blocked",
                "blockers": ["dry_run_no_vast_instance_started"],
                "video_smoke_proven": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    vpa._write_blocked_phase_artifacts(
        job_dir=tmp_path,
        generated_at="new-run",
        provider_reason="vast_heartbeat_blocked",
    )

    refreshed = _read_json(stale_video_path)
    assert refreshed["generated_at"] == "new-run"
    assert refreshed["blockers"] == ["vast_heartbeat_blocked"]
    assert refreshed["video_smoke_proven"] is False


def test_blocked_phase_artifacts_preserve_completed_video_smoke_result(tmp_path: Path) -> None:
    completed_video_path = tmp_path / "vast_video_smoke_result.json"
    completed_video_path.write_text(
        json.dumps(
            {
                "schema_version": "vast_video_smoke_result.v1",
                "generated_at": "completed-run",
                "status": "completed",
                "blockers": [],
                "video_smoke_proven": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    vpa._write_blocked_phase_artifacts(
        job_dir=tmp_path,
        generated_at="later-blocked-run",
        provider_reason="vast_heartbeat_blocked",
    )

    preserved = _read_json(completed_video_path)
    assert preserved["generated_at"] == "completed-run"
    assert preserved["status"] == "completed"
    assert preserved["video_smoke_proven"] is True


def test_isaac_image_startup_preflight_blocks_cold_official_pull(tmp_path: Path) -> None:
    manifest = vpa._isaac_image_startup_preflight(
        job_dir=tmp_path,
        generated_at="2026-06-20T00:00:00+00:00",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="isaac",
        selected_container_image=DEFAULT_ISAAC_IMAGE,
        vast_template_hash_id=None,
        use_vast_template_image=False,
        max_live_minutes=8,
        allow_cold_isaac_image_pull=False,
        min_cold_isaac_pull_live_minutes=18,
    )

    assert manifest["status"] == "blocked"
    assert "cold_official_isaac_image_pull_not_authorized" in manifest["blockers"]
    persisted = _read_json(tmp_path / "vast_isaac_image_startup_preflight.json")
    assert persisted["direct_official_isaac_image"] is True


def test_isaac_image_startup_preflight_blocks_short_template_path_without_cache_proof(
    tmp_path: Path,
) -> None:
    manifest = vpa._isaac_image_startup_preflight(
        job_dir=tmp_path,
        generated_at="2026-06-20T00:00:00+00:00",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="isaac",
        selected_container_image=DEFAULT_ISAAC_IMAGE,
        vast_template_hash_id="template-hash",
        use_vast_template_image=True,
        max_live_minutes=8,
        allow_cold_isaac_image_pull=True,
        min_cold_isaac_pull_live_minutes=18,
    )

    assert manifest["status"] == "blocked"
    assert "vast_template_image_cache_not_proven_for_short_live_window" in manifest["blockers"]
    assert manifest["template_image_cache_proven"] is False
    assert manifest["template_image_cache_evidence"] == "not_proven_by_vast_template_hash"


def test_isaac_image_startup_preflight_allows_custom_image_path(tmp_path: Path) -> None:
    manifest = vpa._isaac_image_startup_preflight(
        job_dir=tmp_path,
        generated_at="2026-06-20T00:00:00+00:00",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="isaac",
        selected_container_image="ghcr.io/blueprint/isaac-smoke:runtime-cache-v1",
        vast_template_hash_id=None,
        use_vast_template_image=False,
        max_live_minutes=8,
        allow_cold_isaac_image_pull=False,
        min_cold_isaac_pull_live_minutes=18,
    )

    assert manifest["status"] == "passed"
    assert manifest["custom_or_template_image_path"] is True
    assert manifest["blockers"] == []


def test_wam_bundle_preflight_does_not_require_isaac_smoke(tmp_path: Path) -> None:
    bundle = tmp_path / "wam_bundle.zip"
    _write_valid_wam_provider_bundle(bundle)
    readiness = {
        "schema_version": "oscar_wam_provider_bundle_manifest.v1",
        "local_bundle_ready_for_remote_staging": True,
        "blockers": [],
    }
    (tmp_path / "oscar_wam_provider_bundle_manifest.json").write_text(
        json.dumps(readiness), encoding="utf-8"
    )

    manifest = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path,
        generated_at="2026-06-20T00:00:00+00:00",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="wam",
        bundle_path=bundle,
        provider_bundle_url="https://example.trycloudflare.com/bundle.zip?token=redacted",
        provider_output_put_url="https://example.trycloudflare.com/output.zip?token=redacted",
    )

    assert manifest["status"] == "passed"
    assert manifest["provider_bundle_kind"] == "wam"
    assert manifest["isaac_smoke_enabled"] is False
    assert manifest["missing_zip_entries"] == []
    assert (
        vpa._resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="wam",
        )
        == "ssh_direct"
    )


def test_unitree_unifolm_bundle_preflight_uses_unitree_entrypoint(tmp_path: Path) -> None:
    bundle = tmp_path / "unitree_unifolm_bundle.zip"
    _write_valid_unitree_unifolm_provider_bundle(bundle)

    manifest = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path,
        generated_at="2026-06-22T00:00:00+00:00",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="unitree_unifolm",
        bundle_path=bundle,
        provider_bundle_url="https://example.trycloudflare.com/bundle.zip?token=redacted",
        provider_output_put_url="https://example.trycloudflare.com/output.zip?token=redacted",
    )

    assert manifest["status"] == "passed"
    assert manifest["provider_bundle_kind"] == "unitree_unifolm"
    assert manifest["isaac_smoke_enabled"] is False
    assert manifest["missing_zip_entries"] == []
    assert manifest["provider_bundle_local_ready_for_remote_staging"] is True
    assert (
        vpa._resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="unitree_unifolm",
        )
        == "ssh_direct"
    )


def test_unitree_groot_n17_sonic_bundle_preflight_uses_groot_entrypoint(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "unitree_groot_n17_sonic_bundle.zip"
    _write_valid_unitree_groot_n17_sonic_provider_bundle(bundle)

    manifest = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path,
        generated_at="2026-06-24T00:00:00+00:00",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="unitree_groot_n17_sonic",
        bundle_path=bundle,
        provider_bundle_url="https://example.trycloudflare.com/groot.zip?token=redacted",
        provider_output_put_url="https://example.trycloudflare.com/groot-out.zip?token=redacted",
    )

    assert manifest["status"] == "passed"
    assert manifest["provider_bundle_kind"] == "unitree_groot_n17_sonic"
    assert manifest["isaac_smoke_enabled"] is False
    assert manifest["missing_zip_entries"] == []
    assert manifest["provider_bundle_local_ready_for_remote_staging"] is True
    assert (
        vpa._resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="unitree_groot_n17_sonic",
        )
        == "ssh_direct"
    )


def test_inline_wam_provider_bundle_payload_is_redacted_in_request_summary(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "wam_bundle.zip"
    _write_valid_wam_provider_bundle(bundle)

    inline = vpa._inline_provider_bundle_payload(
        bundle,
        provider_bundle_kind="wam",
        enable_blueprint_bundle=True,
    )
    assert inline["inline_provider_bundle_transport_used"] is True
    assert inline["inline_provider_bundle_size_bytes"] == bundle.stat().st_size
    assert inline["inline_provider_bundle_sha256_present"] is True

    env = vpa._probe_env(
        job_dir=tmp_path / "inline-env",
        enable_isaac_smoke=False,
        provider_bundle_url="https://example.invalid/wam.zip?token=secret-token",
        provider_output_put_url="https://example.invalid/output.zip?token=secret-token",
        provider_bundle_inline_base64=str(inline["inline_provider_bundle_base64"]),
        provider_bundle_inline_sha256=str(inline["inline_provider_bundle_sha256"]),
    )
    assert (
        env[vpa.VAST_INLINE_PROVIDER_BUNDLE_BASE64_ENV] == inline["inline_provider_bundle_base64"]
    )
    assert (
        env[vpa.VAST_INLINE_PROVIDER_BUNDLE_SHA256_ENV] == inline["inline_provider_bundle_sha256"]
    )

    payload = vpa._create_payload(
        image="image",
        label="inline-wam",
        launch_mode="ssh_direct",
        probe_script="echo hi",
        disk_gb=20,
        env=env,
    )
    summary = vpa._create_request_summary(
        payload,
        secret_values=[str(inline["inline_provider_bundle_base64"]), "secret-token"],
    )

    assert summary["inline_provider_bundle_transport_present"] is True
    assert (
        summary["inline_provider_bundle_base64_length"]
        == inline["inline_provider_bundle_base64_length"]
    )
    assert summary["inline_provider_bundle_sha256_present"] is True
    assert (
        summary["raw_payload_redacted"]["env"][vpa.VAST_INLINE_PROVIDER_BUNDLE_BASE64_ENV]
        == vpa.REDACTED_INLINE_PROVIDER_BUNDLE
    )
    assert str(inline["inline_provider_bundle_base64"]) not in json.dumps(summary)


def test_inline_provider_bundle_payload_is_wam_only_and_size_capped(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "wam_bundle.zip"
    _write_valid_wam_provider_bundle(bundle)

    isaac_inline = vpa._inline_provider_bundle_payload(
        bundle,
        provider_bundle_kind="isaac",
        enable_blueprint_bundle=True,
    )
    assert isaac_inline["inline_provider_bundle_transport_used"] is False
    assert (
        isaac_inline["inline_provider_bundle_transport_reason"]
        == "inline_transport_provider_kind_not_supported"
    )

    unitree_bundle = tmp_path / "unitree_bundle.zip"
    _write_valid_unitree_unifolm_provider_bundle(unitree_bundle)
    unitree_inline = vpa._inline_provider_bundle_payload(
        unitree_bundle,
        provider_bundle_kind="unitree_unifolm",
        enable_blueprint_bundle=True,
    )
    assert unitree_inline["inline_provider_bundle_transport_used"] is True
    assert unitree_inline["inline_provider_bundle_sha256_present"] is True

    groot_bundle = tmp_path / "unitree_groot_bundle.zip"
    _write_valid_unitree_groot_n17_sonic_provider_bundle(groot_bundle)
    groot_inline = vpa._inline_provider_bundle_payload(
        groot_bundle,
        provider_bundle_kind="unitree_groot_n17_sonic",
        enable_blueprint_bundle=True,
    )
    assert groot_inline["inline_provider_bundle_transport_used"] is True
    assert groot_inline["inline_provider_bundle_sha256_present"] is True

    too_large = vpa._inline_provider_bundle_payload(
        bundle,
        provider_bundle_kind="wam",
        enable_blueprint_bundle=True,
        max_raw_bytes=1,
    )
    assert too_large["inline_provider_bundle_transport_used"] is False
    assert (
        too_large["inline_provider_bundle_transport_reason"]
        == "provider_bundle_too_large_for_inline_env"
    )


def test_vast_adapter_disables_inline_wam_or_unitree_bundle_when_fetch_url_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_live_gates(tmp_path, monkeypatch)
    bundle = tmp_path / "wam_bundle.zip"
    _write_valid_wam_provider_bundle(bundle)

    result = run_vast_provider_adapter(
        job_dir=tmp_path / "dry-run",
        mode="dry-run",
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle=bundle,
        provider_bundle_url="https://bundle.example/wam.zip?token=redacted",
        provider_output_put_url="https://bundle.example/out.zip?token=redacted",
        enable_blueprint_bundle=True,
        provider_bundle_kind="wam",
        vast_launch_mode="ssh_direct",
        session_max_live_minutes=None,
    )

    assert result["status"] == "dry_run_ready"
    assert result["provider_bundle_inline_transport_used"] is False
    assert (
        result["provider_bundle_inline_transport_reason"]
        == "disabled_for_vast_env_size_with_fetch_url"
    )

    unitree_bundle = tmp_path / "unitree_bundle.zip"
    _write_valid_unitree_unifolm_provider_bundle(unitree_bundle)
    unitree_result = run_vast_provider_adapter(
        job_dir=tmp_path / "unitree-dry-run",
        mode="dry-run",
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle=unitree_bundle,
        provider_bundle_url="https://bundle.example/unitree.zip?token=redacted",
        provider_output_put_url="https://bundle.example/unitree-out.zip?token=redacted",
        enable_blueprint_bundle=True,
        provider_bundle_kind="unitree_unifolm",
        vast_launch_mode="ssh_direct",
        session_max_live_minutes=None,
    )

    assert unitree_result["status"] == "dry_run_ready"
    assert unitree_result["provider_bundle_inline_transport_used"] is False
    assert (
        unitree_result["provider_bundle_inline_transport_reason"]
        == "disabled_for_vast_env_size_with_fetch_url"
    )

    groot_bundle = tmp_path / "unitree-groot-dry-run.zip"
    _write_valid_unitree_groot_n17_sonic_provider_bundle(groot_bundle)
    groot_result = run_vast_provider_adapter(
        job_dir=tmp_path / "unitree-groot-dry-run",
        mode="dry-run",
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle=groot_bundle,
        provider_bundle_url="https://bundle.example/groot.zip?token=redacted",
        provider_output_put_url="https://bundle.example/groot-out.zip?token=redacted",
        enable_blueprint_bundle=True,
        provider_bundle_kind="unitree_groot_n17_sonic",
        vast_launch_mode="ssh_direct",
        session_max_live_minutes=None,
    )

    assert groot_result["status"] == "dry_run_ready"
    assert groot_result["provider_bundle_inline_transport_used"] is False
    assert (
        groot_result["provider_bundle_inline_transport_reason"]
        == "disabled_for_vast_env_size_with_fetch_url"
    )


def test_vast_adapter_blocks_stale_isaac_bundle_prefixed_paths_without_resolver(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "stale_isaac_provider_runtime_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "provider_runtime/run_isaac_realistic_runtime.sh",
            "write_missing_result\n"
            "isaac_runner_process_exited_without_runtime_result\n"
            "blocked_isaac_process_exited_without_result\n",
        )
        archive.writestr(
            "provider_runtime/isaac_realistic_runtime_runner.py",
            "from isaacsim import SimulationApp\n",
        )
        archive.writestr(
            "provider_runtime/isaac_provider_eval_manifest.json",
            json.dumps(
                {
                    "relative_paths": {
                        "generated_site_scene_usda": ("provider_runtime/generated_site_scene.usda"),
                        "camera_manifest": "provider_runtime/camera_manifest.json",
                    }
                }
            ),
        )
        for name in (
            "provider_runtime/scenario_eval_matrix.json",
            "provider_runtime/camera_manifest.json",
            "provider_runtime/episode_spec_manifest.json",
        ):
            archive.writestr(name, "{}\n")
        archive.writestr("provider_runtime/generated_site_scene.usda", "#usda 1.0\n")
        archive.writestr("provider_runtime/generated_site_scene.usd", b"USD")

    manifest = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path,
        generated_at="2026-06-20T00:00:00+00:00",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="isaac",
        bundle_path=bundle,
        provider_bundle_url="https://example.trycloudflare.com/bundle.zip?token=redacted",
        provider_output_put_url="https://example.trycloudflare.com/output.zip?token=redacted",
    )

    assert manifest["status"] == "blocked"
    assert "provider_runtime_bundle_stale_prefixed_paths_without_resolver" in manifest["blockers"]
    assert (
        manifest["provider_eval_manifest_relative_paths"]["generated_site_scene_usda"]
        == "provider_runtime/generated_site_scene.usda"
    )


def test_bundle_preflight_uses_public_dns_fallback_when_normal_head_dns_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "isaac_provider_runtime_bundle.zip"
    _write_valid_provider_bundle(bundle)

    def failing_urlopen(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise urllib.error.URLError(OSError("nodename nor servname provided"))

    monkeypatch.setattr(vpa.urllib.request, "urlopen", failing_urlopen)
    monkeypatch.setattr(
        vpa,
        "_head_with_public_dns_fallback",
        lambda url, timeout_seconds=20: {
            "status": "passed",
            "method": "HEAD_WITH_PUBLIC_DNS_FALLBACK",
            "http_status_code": 200,
            "content_type": "application/zip",
            "content_length": bundle.stat().st_size,
            "public_dns_resolver": "dig @1.1.1.1",
            "resolved_ip_count": 2,
            "resolved_ip_used": "104.16.230.132",
        },
    )

    manifest = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path,
        generated_at="2026-06-20T00:00:00+00:00",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="isaac",
        bundle_path=bundle,
        provider_bundle_url="https://example.trycloudflare.com/bundle.zip?token=secret-token",
        provider_output_put_url="https://example.trycloudflare.com/output.zip?token=secret-token",
        verify_staging_urls=True,
    )

    assert manifest["status"] == "passed"
    assert manifest["blockers"] == []
    assert manifest["bundle_url_probe"]["status"] == "passed"
    assert manifest["bundle_url_probe"]["method"] == "HEAD_WITH_PUBLIC_DNS_FALLBACK"
    assert manifest["bundle_url_probe"]["normal_head_error_type"] == "URLError"
    persisted = (tmp_path / "vast_blueprint_bundle_preflight.json").read_text(encoding="utf-8")
    assert "secret-token" not in persisted


def test_wam_provider_output_zip_accepts_wam_runtime_result(tmp_path: Path) -> None:
    output_zip = tmp_path / "output.zip"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "wam_runtime_result.json",
            json.dumps({"status": "completed", "blockers": []}),
        )

    result = vpa._inspect_provider_runtime_output_zip(
        output_zip,
        expected_video_count=1,
    )

    assert result["runtime_result_present"] is True
    assert result["runtime_result_status"] == "completed"


def test_provider_output_zip_accepts_aura_interiorgs_result(tmp_path: Path) -> None:
    output_zip = tmp_path / "aura-interiorgs-output.zip"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "immutable_execution/adp_aura_interiorgs_result.json",
            json.dumps({"status": "blocked", "blockers": ["reference_lama_failed"]}),
        )

    result = vpa._inspect_provider_runtime_output_zip(
        output_zip,
        expected_video_count=0,
    )

    assert result["runtime_result_present"] is True
    assert result["runtime_result_status"] == "blocked"
    assert result["runtime_result_blockers"] == ["reference_lama_failed"]


def test_unitree_unifolm_provider_output_zip_accepts_policy_output(tmp_path: Path) -> None:
    output_zip = tmp_path / "unitree-output.zip"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_unifolm_policy_provider_output.json",
            json.dumps(
                {
                    "status": "completed",
                    "unitree_unifolm_model_executed": True,
                    "unitree_unifolm_policy_action_command_ran": True,
                    "action": {"action_type": "manipulation_contact"},
                    "blockers": [],
                }
            ),
        )

    result = vpa._inspect_provider_runtime_output_zip(
        output_zip,
        expected_video_count=0,
    )

    assert result["runtime_result_present"] is True
    assert result["runtime_result_status"] == "completed"
    assert result["video_smoke_proven"] is False


def test_unitree_groot_n17_sonic_provider_output_zip_accepts_policy_output(
    tmp_path: Path,
) -> None:
    output_zip = tmp_path / "unitree-groot-output.zip"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "completed",
                    "unitree_groot_n17_sonic_model_executed": True,
                    "unitree_groot_n17_sonic_policy_action_command_ran": True,
                    "action": {"action_type": "unitree_g1_sonic_latent_action_chunk"},
                    "blockers": [],
                }
            ),
        )

    result = vpa._inspect_provider_runtime_output_zip(
        output_zip,
        expected_video_count=0,
    )

    assert result["runtime_result_present"] is True
    assert result["runtime_result_status"] == "completed"
    assert result["video_smoke_proven"] is False


def test_request_logs_breaks_on_missing_container_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_api_json(**kwargs):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        return 200, {"result_url": "https://example.invalid/log.txt"}

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(
        vpa,
        "_fetch_text",
        lambda *_args, **_kwargs: "Error response from daemon: No such container: C.123",
    )

    result = vpa._request_logs_and_fetch(
        instance_id=123,
        api_key="secret",
        output_log_path=tmp_path / "onstart.log",
        secret_values=["secret"],
        wait_seconds=0,
        retry_interval_seconds=99,
        max_wait_seconds=999,
        success_markers=["BLUEPRINT_VAST_ONSTART_DONE"],
        container_missing_retry_attempts=1,
    )

    assert calls["count"] == 1
    assert result["log_poll_attempts"][0]["container_missing_marker_observed"] is True
    assert "No such container" in (tmp_path / "onstart.log").read_text(encoding="utf-8")


def test_request_logs_retries_transient_missing_container_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}
    log_texts = iter(
        [
            "Error response from daemon: No such container: C.123",
            "Error response from daemon: No such container: C.123",
            "BLUEPRINT_VAST_HEARTBEAT_OK\nBLUEPRINT_VAST_ONSTART_DONE\n",
        ]
    )

    def fake_api_json(**kwargs):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        return 200, {"result_url": "https://example.invalid/log.txt"}

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(vpa, "_fetch_text", lambda *_args, **_kwargs: next(log_texts))
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)

    result = vpa._request_logs_and_fetch(
        instance_id=123,
        api_key="secret",
        output_log_path=tmp_path / "onstart.log",
        secret_values=["secret"],
        wait_seconds=0,
        retry_interval_seconds=99,
        max_wait_seconds=999,
        success_markers=["BLUEPRINT_VAST_ONSTART_DONE"],
        container_missing_retry_attempts=5,
    )

    assert calls["count"] == 3
    assert result["log_poll_attempts"][0]["container_missing_marker_observed"] is True
    assert result["log_poll_attempts"][1]["container_missing_observed_count"] == 2
    assert result["log_poll_attempts"][2]["success_marker_found"] is True
    assert "BLUEPRINT_VAST_ONSTART_DONE" in (tmp_path / "onstart.log").read_text(encoding="utf-8")


def test_request_logs_records_api_url_error_without_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_api_json(**_kwargs):  # type: ignore[no-untyped-def]
        raise urllib.error.URLError("temporary dns failure")

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)

    result = vpa._request_logs_and_fetch(
        instance_id=123,
        api_key="secret",
        output_log_path=tmp_path / "onstart.log",
        secret_values=["secret"],
        wait_seconds=0,
        retry_interval_seconds=99,
        max_wait_seconds=0,
        success_markers=["BLUEPRINT_VAST_ONSTART_DONE"],
    )

    assert result["result_url_present"] is False
    assert result["output_size_bytes"] == 0
    assert result["output_fetch_error"].startswith("URLError:")
    assert result["log_poll_attempts"][0]["api_request_error"].startswith("URLError:")
    assert (tmp_path / "onstart.log").read_text(encoding="utf-8") == ""


def test_wam_cold_pull_extends_heartbeat_no_progress_to_admitted_minimum() -> None:
    assert (
        vpa.cold_pull_aware_heartbeat_no_progress_seconds(
            configured_seconds=600,
            provider_bundle_kind="wam",
            allow_cold_image_pull=True,
            min_cold_image_pull_live_minutes=18,
            startup_timeout_seconds=3600,
            max_live_minutes=180,
        )
        == 1080
    )


def test_cold_pull_heartbeat_extension_never_exceeds_startup_window() -> None:
    assert (
        vpa.cold_pull_aware_heartbeat_no_progress_seconds(
            configured_seconds=300,
            provider_bundle_kind="wam",
            allow_cold_image_pull=True,
            min_cold_image_pull_live_minutes=18,
            startup_timeout_seconds=900,
            max_live_minutes=180,
        )
        == 900
    )
    assert (
        vpa.cold_pull_aware_heartbeat_no_progress_seconds(
            configured_seconds=600,
            provider_bundle_kind="isaac",
            allow_cold_image_pull=True,
            min_cold_image_pull_live_minutes=18,
            startup_timeout_seconds=3600,
            max_live_minutes=180,
        )
        == 600
    )


def test_request_logs_breaks_on_no_progress_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr(vpa.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        vpa,
        "_api_json",
        lambda **_kwargs: (200, {"result_url": "https://example.invalid/log.txt"}),
    )
    monkeypatch.setattr(vpa, "_fetch_text", lambda *_args, **_kwargs: "")

    result = vpa._request_logs_and_fetch(
        instance_id=123,
        api_key="secret",
        output_log_path=tmp_path / "onstart.log",
        secret_values=["secret"],
        wait_seconds=0,
        retry_interval_seconds=1,
        max_wait_seconds=999,
        success_markers=["BLUEPRINT_VAST_ONSTART_DONE"],
        no_progress_seconds=2,
    )

    assert result["break_reason"] == "no_log_progress_timeout"
    assert result["no_progress_timeout_reached"] is True
    assert result["log_poll_attempts"][-1]["no_progress_timeout_reached"] is True
    assert result["log_poll_attempts"][-1]["progress_observed"] is False
    assert (tmp_path / "onstart.log").read_text(encoding="utf-8") == ""


def test_request_logs_dud_container_flicker_is_not_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dud offer whose container never materializes flickers between empty logs and a
    Docker 'No such container' error. That changing text must NOT be counted as progress —
    otherwise the no-progress watchdog never fires and the dud idles the whole live window.
    """
    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr(vpa.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        vpa,
        "_api_json",
        lambda **_kwargs: (200, {"result_url": "https://example.invalid/log.txt"}),
    )
    flicker = iter(["", "Error response from daemon: No such container: C.123"] * 50)
    monkeypatch.setattr(vpa, "_fetch_text", lambda *_args, **_kwargs: next(flicker))

    result = vpa._request_logs_and_fetch(
        instance_id=123,
        api_key="secret",
        output_log_path=tmp_path / "onstart.log",
        secret_values=["secret"],
        wait_seconds=0,
        retry_interval_seconds=1,
        max_wait_seconds=999,
        success_markers=["BLUEPRINT_VAST_ONSTART_DONE"],
        # high container-missing tolerance so the win must come from the no-progress path,
        # proving the flicker no longer fakes progress (the actual bug we fixed)
        container_missing_retry_attempts=999,
        no_progress_seconds=4,
    )

    assert result["break_reason"] == "no_log_progress_timeout"
    assert result["no_progress_timeout_reached"] is True
    # none of the flickering error/empty polls counted as progress
    assert all(a["progress_observed"] is False for a in result["log_poll_attempts"])


def test_request_logs_ignores_unstructured_noise_after_worker_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read-only SSH diagnostics and other stdout noise cannot extend paid work."""

    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr(vpa.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        vpa,
        "_api_json",
        lambda **_kwargs: (200, {"result_url": "https://example.invalid/log.txt"}),
    )
    noise = iter(
        [
            "BLUEPRINT_WAM_RUNTIME_PHASE:worker:environment_build:started\n",
            "BLUEPRINT_WAM_RUNTIME_PHASE:worker:environment_build:started\n"
            "Accepted publickey for root\n",
            "BLUEPRINT_WAM_RUNTIME_PHASE:worker:environment_build:started\n"
            "Disconnected from user root\n",
        ]
        * 20
    )
    monkeypatch.setattr(vpa, "_fetch_text", lambda *_args, **_kwargs: next(noise))

    result = vpa._request_logs_and_fetch(
        instance_id=123,
        api_key="secret",
        output_log_path=tmp_path / "onstart.log",
        secret_values=["secret"],
        wait_seconds=0,
        retry_interval_seconds=1,
        max_wait_seconds=999,
        success_markers=["BLUEPRINT_VAST_ONSTART_DONE"],
        no_progress_seconds=4,
    )

    assert result["break_reason"] == "no_log_progress_timeout"
    assert result["no_progress_timeout_reached"] is True
    attempts = result["log_poll_attempts"]
    assert attempts[0]["progress_observed"] is True
    assert all(item["structured_phase_tracking_active"] is True for item in attempts)
    assert all(item["progress_observed"] is False for item in attempts[1:])


def test_request_logs_persists_last_redacted_snapshot_before_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        vpa,
        "_api_json",
        lambda **_kwargs: (200, {"result_url": "https://example.invalid/log.txt"}),
    )
    snapshots = iter(
        [
            "BLUEPRINT_WAM_RUNTIME_PHASE:worker:static_sdf:completed\nsecret\n",
            KeyboardInterrupt(),
        ]
    )

    def fetch(*_args, **_kwargs):
        value = next(snapshots)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(vpa, "_fetch_text", fetch)
    output = tmp_path / "onstart.log"

    with pytest.raises(KeyboardInterrupt):
        vpa._request_logs_and_fetch(
            instance_id=123,
            api_key="secret",
            output_log_path=output,
            secret_values=["secret"],
            wait_seconds=0,
            retry_interval_seconds=1,
            max_wait_seconds=999,
            success_markers=["BLUEPRINT_VAST_ONSTART_DONE"],
            no_progress_seconds=999,
        )

    assert output.read_text(encoding="utf-8") == (
        "BLUEPRINT_WAM_RUNTIME_PHASE:worker:static_sdf:completed\nREDACTED_SECRET\n"
    )


def test_vast_adapter_dry_run_writes_required_artifacts(tmp_path: Path) -> None:
    result = run_vast_provider_adapter(job_dir=tmp_path, mode="dry-run")

    assert result["status"] == "dry_run_ready"
    assert result["api_call_performed"] is False
    assert result["raw_api_key_stored"] is False
    required = [
        "vast_runtime_discovery.json",
        "vast_provider_plan.json",
        "vast_offer_selection_manifest.json",
        "vast_budget_ledger.json",
        "vast_runtime_phase_log.jsonl",
        "vast_startup_probe_manifest.json",
        "vast_gpu_sanity_report.json",
        "vast_isaac_smoke_result.json",
        "vast_provider_command_result.json",
        "vast_teardown_manifest.json",
        "vast_final_validation.json",
        "provider_worker_endpoint_manifest.json",
    ]
    for name in required:
        assert (tmp_path / name).is_file(), name
    endpoint_manifest_path = Path(result["provider_worker_endpoint_manifest_path"])
    endpoint_manifest = _read_json(endpoint_manifest_path)
    assert endpoint_manifest_path == tmp_path / "provider_worker_endpoint_manifest.json"
    assert endpoint_manifest["provider"] == "vast"
    assert endpoint_manifest["status"] == "endpoint_discovery_pending_provider_runtime"
    assert endpoint_manifest["direct_http_worker_endpoint_expected"] is True
    assert endpoint_manifest["direct_policy_infer_from_local_loop_allowed"] is False
    assert endpoint_manifest["claim_boundary"]["readyz_probe_required_before_customer_eval"] is True
    assert endpoint_manifest["blockers"] == ["provider_worker_endpoint_not_discovered_yet"]
    assert endpoint_manifest == result["provider_worker_endpoint_manifest"]
    offer = _read_json(tmp_path / "vast_offer_selection_manifest.json")
    assert offer["status"] == "dry_run_ready"
    assert offer["offer_search_performed"] is False
    validation = _read_json(tmp_path / "vast_final_validation.json")
    assert validation["status"] == "passed"
    assert validation["continuing_spend_from_this_run"] is False


def test_vast_adapter_uses_image_login_mode_env_for_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(vpa.VAST_IMAGE_LOGIN_MODE_ENV, "never")

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="dry-run",
        public_image="docker.io/nijelhunt/blueprint-vast-unitree-groot-sonic:20260624-pydeps-vast1",
    )

    assert result["status"] == "dry_run_ready"
    assert result["ngc_image_login_mode"] == "never"


def test_vast_adapter_template_discovery_blocks_before_api_without_read_only_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(VAST_API_GATE_ENV, raising=False)
    monkeypatch.delenv(VAST_INSTANCE_LAUNCH_GATE_ENV, raising=False)
    monkeypatch.setenv(VAST_API_KEY_FILE_ENV, str(tmp_path / "missing-vast-key"))

    def fail_if_called(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected Vast API call: {kwargs}")

    monkeypatch.setattr(vpa, "_api_json", fail_if_called)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="template-discovery",
        allow_vast_api_call=False,
        allow_instance_launch=False,
    )

    assert result["status"] == "blocked"
    assert result["api_call_performed"] is False
    assert result["vast_side_effects_may_have_occurred"] is False
    assert result["vast_instance_ids"] == []
    assert "missing_read_only_vast_api_gate" in result["blockers"]
    assert "missing_file_based_vast_api_key" in result["blockers"]
    discovery = _read_json(tmp_path / "vast_template_discovery.json")
    assert discovery["status"] == "blocked"
    teardown = _read_json(tmp_path / "vast_teardown_manifest.json")
    assert teardown["continuing_spend_from_this_run"] is False
    validation = _read_json(tmp_path / "vast_final_validation.json")
    assert validation["status"] == "passed"


def test_vast_adapter_template_discovery_reads_templates_without_launch_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "secret-vast-key"
    key_file = tmp_path / "vast_api_key"
    _write_secret(key_file, secret)
    monkeypatch.setenv(VAST_API_KEY_FILE_ENV, str(key_file))
    monkeypatch.setenv(VAST_API_GATE_ENV, "true")
    monkeypatch.delenv(VAST_INSTANCE_LAUNCH_GATE_ENV, raising=False)
    calls: list[tuple[str, str, object]] = []

    def fake_api_json(
        *,
        method: str,
        path: str,
        api_key: str,
        payload=None,
        timeout_seconds: int = 30,
    ):  # type: ignore[no-untyped-def]
        assert api_key == secret
        calls.append((method, path, payload))
        assert method == "GET"
        assert str(path).startswith("https://console.vast.ai/api/v0/template/?")
        assert "select_filters" in str(path)
        assert payload is None
        return 200, {
            "templates_found": 2,
            "templates": [
                {
                    "id": 11,
                    "hash_id": "isaac-template",
                    "name": "NVIDIA Isaac Sim 6",
                    "image": "nvcr.io/nvidia/isaac-sim:6.0.0",
                    "count_created": 9,
                },
                {
                    "id": 12,
                    "hash_id": "pytorch-template",
                    "name": "PyTorch",
                    "image": "pytorch/pytorch:latest",
                    "count_created": 100,
                },
            ],
        }

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="template-discovery",
        allow_vast_api_call=True,
        allow_instance_launch=False,
    )

    assert calls
    assert result["status"] == "completed"
    assert result["api_call_performed"] is True
    assert result["vast_side_effects_may_have_occurred"] is False
    assert result["vast_instance_ids"] == []
    assert result["final_validation_status"] == "passed"
    discovery = _read_json(tmp_path / "vast_template_discovery.json")
    assert discovery["status"] == "completed"
    assert discovery["templates_returned"] == 2
    assert discovery["isaac_candidate_count"] == 1
    assert discovery["isaac_template_candidates"][0]["hash_id"] == "isaac-template"
    teardown = _read_json(tmp_path / "vast_teardown_manifest.json")
    assert teardown["status"] == "not_required_template_discovery"
    assert teardown["continuing_spend_from_this_run"] is False
    persisted = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.json"))
    assert secret not in persisted


def test_vast_adapter_blocks_live_without_gates_or_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(VAST_API_GATE_ENV, raising=False)
    monkeypatch.delenv(VAST_INSTANCE_LAUNCH_GATE_ENV, raising=False)
    monkeypatch.setenv(VAST_API_KEY_FILE_ENV, str(tmp_path / "missing-vast-key"))

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=False,
        allow_instance_launch=False,
    )

    assert result["status"] == "blocked"
    assert result["api_call_performed"] is False
    assert f"missing_env_{VAST_API_GATE_ENV}" in result["blockers"]
    assert f"missing_env_{VAST_INSTANCE_LAUNCH_GATE_ENV}" in result["blockers"]
    assert "missing_cli_allow_vast_api_call" in result["blockers"]
    assert "missing_cli_allow_vast_instance_launch" in result["blockers"]
    assert f"missing_file_based_secret_{VAST_API_KEY_FILE_ENV}" in result["blockers"]
    teardown = _read_json(tmp_path / "vast_teardown_manifest.json")
    assert teardown["continuing_spend_from_this_run"] is False


def test_vast_adapter_blocks_session_runtime_exhaustion_before_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_live_gates(tmp_path, monkeypatch)
    ledger = tmp_path / "vast_session_cost_summary.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "vast_session_cost_summary.v3",
                "attempts": [
                    {
                        "runtime_seconds_observed_by_adapter": 2700,
                        "estimated_cost_usd": 0.05,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fail_if_called(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected Vast API call: {kwargs}")

    monkeypatch.setattr(vpa, "_api_json", fail_if_called)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        max_live_minutes=1,
        session_max_live_minutes=45,
    )

    assert result["status"] == "blocked"
    assert result["api_call_performed"] is False
    assert result["vast_instance_ids"] == []
    assert result["blockers"] == ["session_live_runtime_limit_exhausted"]
    guard = _read_json(tmp_path / "vast_session_budget_guard.json")
    assert guard["status"] == "blocked"
    assert guard["prior_live_runtime_minutes"] == 45
    assert guard["requested_max_live_runtime_minutes"] == 1
    assert guard["blockers"] == ["session_live_runtime_limit_exhausted"]
    offer = _read_json(tmp_path / "vast_offer_selection_manifest.json")
    assert offer["offer_search_performed"] is False
    teardown = _read_json(tmp_path / "vast_teardown_manifest.json")
    assert teardown["status"] == "not_required_session_budget_blocked"
    assert teardown["continuing_spend_from_this_run"] is False


def test_vast_adapter_blocks_paid_launch_when_launch_lock_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_live_gates(tmp_path, monkeypatch)
    lock_path = tmp_path / "busy-vast-paid-launch.lock"
    monkeypatch.setenv(vpa.VAST_LAUNCH_LOCK_FILE_ENV, str(lock_path))
    lock_handle = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_handle.write('{"pid":123,"purpose":"test-holder"}\n')
    lock_handle.flush()

    def fail_if_called(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected Vast API call: {kwargs}")

    monkeypatch.setattr(vpa, "_api_json", fail_if_called)
    try:
        result = run_vast_provider_adapter(
            job_dir=tmp_path / "lock-blocked",
            mode="live-startup-probe",
            paid_resource_admission_grant=_paid_grant(),
            allow_vast_api_call=True,
            allow_instance_launch=True,
            max_live_minutes=1,
            session_max_live_minutes=None,
        )
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()

    assert result["status"] == "blocked"
    assert result["reason"] == "vast_paid_launch_lock_blocked"
    assert result["api_call_performed"] is False
    assert result["vast_side_effects_may_have_occurred"] is False
    assert result["vast_instance_ids"] == []
    assert result["blockers"] == ["vast_paid_launch_lock_busy"]
    job_dir = tmp_path / "lock-blocked"
    lock_manifest = _read_json(job_dir / "vast_launch_lock_manifest.json")
    assert lock_manifest["status"] == "blocked"
    assert lock_manifest["lock_acquired"] is False
    offer = _read_json(job_dir / "vast_offer_selection_manifest.json")
    assert offer["offer_search_performed"] is False
    teardown = _read_json(job_dir / "vast_teardown_manifest.json")
    assert teardown["status"] == "not_required_launch_lock_blocked"
    assert teardown["continuing_spend_from_this_run"] is False
    validation = _read_json(job_dir / "vast_final_validation.json")
    assert validation["continuing_spend_from_this_run"] is False


def test_vast_adapter_blocks_paid_launch_when_existing_instance_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = _configure_live_gates(tmp_path, monkeypatch)
    calls: list[tuple[str, str]] = []

    def fake_api_json(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["api_key"] == secret
        calls.append((kwargs["method"], kwargs["path"]))
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {
                "instances": [
                    {
                        "id": 123,
                        "machine_id": 456,
                        "actual_status": "loading",
                        "cur_state": "running",
                        "gpu_name": "RTX 3090",
                        "dph_total": 0.2,
                    }
                ]
            }
        raise AssertionError(f"unexpected Vast API call: {kwargs}")

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    job_dir = tmp_path / "active-instance-blocked"
    result = run_vast_provider_adapter(
        job_dir=job_dir,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        max_live_minutes=1,
        session_max_live_minutes=None,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "vast_prelaunch_inventory_guard_blocked"
    assert result["api_call_performed"] is True
    assert result["vast_side_effects_may_have_occurred"] is False
    assert result["vast_instance_ids"] == []
    assert result["blockers"] == ["active_vast_instances_detected_before_new_launch"]
    assert calls == [("GET", "/instances/")]
    guard = _read_json(job_dir / "vast_prelaunch_inventory_guard.json")
    assert guard["status"] == "blocked"
    assert guard["active_instance_count"] == 1
    assert guard["continuing_spend_detected_before_new_launch"] is True
    assert guard["active_instances"][0]["id"] == 123
    assert guard["raw_secret_values_recorded"] is False
    offer = _read_json(job_dir / "vast_offer_selection_manifest.json")
    assert offer["offer_search_performed"] is False
    assert offer["blockers"] == ["active_vast_instances_detected_before_new_launch"]
    lock_manifest = _read_json(job_dir / "vast_launch_lock_manifest.json")
    assert lock_manifest["status"] == "released"
    assert lock_manifest["lock_released"] is True
    teardown = _read_json(job_dir / "vast_teardown_manifest.json")
    assert teardown["status"] == "not_required_prelaunch_inventory_guard_blocked"
    assert teardown["continuing_spend_from_this_run"] is False


def test_prelaunch_inventory_guard_allows_only_exact_authorized_active_instances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vpa,
        "_api_json",
        lambda **_kwargs: (
            200,
            {
                "instances": [
                    {"id": 123, "actual_status": "running", "gpu_name": "RTX 4090"},
                    {"id": 124, "actual_status": "running", "gpu_name": "L40S"},
                ]
            },
        ),
    )
    blocked = vpa._prelaunch_inventory_guard(
        job_dir=tmp_path / "blocked",
        generated_at="2026-08-05T00:00:00Z",
        api_key="test-key",
        allowed_active_instance_ids=(123,),
    )
    assert blocked["status"] == "blocked"
    assert blocked["allowed_active_instance_ids"] == [123]
    assert blocked["unexpected_active_instance_count"] == 1
    assert blocked["unexpected_active_instances"][0]["id"] == 124

    passed = vpa._prelaunch_inventory_guard(
        job_dir=tmp_path / "passed",
        generated_at="2026-08-05T00:00:00Z",
        api_key="test-key",
        allowed_active_instance_ids=(123, 124),
    )
    assert passed["status"] == "passed"
    assert passed["continuing_spend_detected_before_new_launch"] is True
    assert passed["unexpected_active_instance_count"] == 0


def test_vast_adapter_blocks_blueprint_bundle_missing_staging_before_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_live_gates(tmp_path, monkeypatch)
    bundle = tmp_path / "isaac_provider_runtime_bundle.zip"
    _write_valid_provider_bundle(bundle)

    def fail_if_called(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected Vast API call: {kwargs}")

    monkeypatch.setattr(vpa, "_api_json", fail_if_called)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle=bundle,
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        session_max_live_minutes=None,
    )

    assert result["status"] == "blocked"
    assert result["api_call_performed"] is False
    assert result["reason"] == "vast_blueprint_bundle_preflight_blocked"
    assert result["blockers"] == [
        "provider_bundle_fetch_url_missing",
        "provider_output_put_url_missing",
    ]
    preflight = _read_json(tmp_path / "vast_blueprint_bundle_preflight.json")
    assert preflight["status"] == "blocked"
    assert preflight["zip_required_entries_present"] is True
    assert preflight["provider_bundle_fetch_url_present"] is False
    assert preflight["provider_output_put_url_present"] is False
    offer = _read_json(tmp_path / "vast_offer_selection_manifest.json")
    assert offer["offer_search_performed"] is False
    provider = _read_json(tmp_path / "vast_provider_command_result.json")
    assert provider["provider_command_path_remote_proven"] is False
    teardown = _read_json(tmp_path / "vast_teardown_manifest.json")
    assert teardown["status"] == "not_required_blueprint_bundle_preflight_blocked"
    assert teardown["continuing_spend_from_this_run"] is False


def test_vast_adapter_blocks_stale_bundle_url_before_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_live_gates(tmp_path, monkeypatch)
    bundle = tmp_path / "isaac_provider_runtime_bundle.zip"
    _write_valid_provider_bundle(bundle)

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        method = request.get_method()
        if method == "HEAD":
            raise urllib.error.HTTPError(
                request.full_url,
                404,
                "not found",
                {},
                BytesIO(b"missing"),
            )
        raise AssertionError(method)

    def fail_if_called(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected Vast API call: {kwargs}")

    monkeypatch.setattr(vpa.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vpa, "_api_json", fail_if_called)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle=bundle,
        provider_bundle_url="https://example.invalid/bundle.zip?token=secret-token",
        provider_output_put_url="https://example.invalid/output.zip?token=secret-token",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        verify_staging_urls=True,
        session_max_live_minutes=None,
    )

    assert result["status"] == "blocked"
    assert result["api_call_performed"] is False
    assert result["blockers"] == ["provider_bundle_fetch_url_unreachable"]
    preflight = _read_json(tmp_path / "vast_blueprint_bundle_preflight.json")
    assert preflight["staging_url_verification_requested"] is True
    assert preflight["bundle_url_probe"]["status"] == "blocked"
    assert preflight["bundle_url_probe"]["http_status_code"] == 404
    assert preflight["output_put_probe"]["status"] == "skipped"
    assert "provider_output_put_url_not_mutation_probed" in preflight["warnings"]
    persisted = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.json"))
    assert "secret-token" not in persisted


def test_vast_adapter_blocks_localhost_staging_urls_before_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_live_gates(tmp_path, monkeypatch)
    bundle = tmp_path / "isaac_provider_runtime_bundle.zip"
    _write_valid_provider_bundle(bundle)

    def fail_if_called(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected Vast API call: {kwargs}")

    monkeypatch.setattr(vpa, "_api_json", fail_if_called)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle=bundle,
        provider_bundle_url="http://127.0.0.1:8765/bundle.zip?token=secret-token",
        provider_output_put_url="http://localhost:8765/output.zip?token=secret-token",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        verify_staging_urls=True,
        session_max_live_minutes=None,
    )

    assert result["status"] == "blocked"
    assert result["api_call_performed"] is False
    assert "provider_bundle_fetch_url_not_publicly_reachable" in result["blockers"]
    assert "provider_output_put_url_not_publicly_reachable" in result["blockers"]
    preflight = _read_json(tmp_path / "vast_blueprint_bundle_preflight.json")
    assert preflight["bundle_url_probe"]["blockers"] == [
        "provider_bundle_fetch_url_not_publicly_reachable"
    ]
    assert preflight["output_put_probe"]["blockers"] == [
        "provider_output_put_url_not_publicly_reachable"
    ]
    persisted = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.json"))
    assert "secret-token" not in persisted


def test_vast_adapter_redacts_url_encoded_signed_url_tokens() -> None:
    url = "https://example.invalid/bundle.zip?token=abc%2Bdef%3D"
    secrets = _url_secret_values(url)

    assert "abc+def=" in secrets
    assert "abc%2Bdef%3D" in secrets
    redacted = _redact_text(url, secrets)
    assert "abc%2Bdef%3D" not in redacted
    assert redacted == "https://example.invalid/bundle.zip?REDACTED_QUERY"

    s3_url = (
        "https://object.example/bundle.zip?"
        "X-Amz-Credential=AKIAEXAMPLE%2F20260621%2Fus-east-1%2Fs3%2Faws4_request"
        "&X-Amz-Signature=s3-secret&X-Amz-Date=20260621"
    )
    s3_secrets = _url_secret_values(s3_url)
    assert "s3-secret" in s3_secrets
    assert "AKIAEXAMPLE/20260621/us-east-1/s3/aws4_request" in s3_secrets
    s3_redacted = _redact_text(s3_url, s3_secrets)
    assert "s3-secret" not in s3_redacted
    assert "X-Amz-Credential" not in s3_redacted
    assert "AKIAEXAMPLE" not in s3_redacted
    assert s3_redacted == "https://object.example/bundle.zip?REDACTED_QUERY"


def test_vast_adapter_mocked_live_heartbeat_gpu_and_teardown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secret = "secret-vast-key"
    key_file = tmp_path / "vast_api_key"
    key_file.write_text(secret + "\n", encoding="utf-8")
    key_file.chmod(0o600)
    monkeypatch.setenv(VAST_API_KEY_FILE_ENV, str(key_file))
    monkeypatch.setenv(VAST_API_GATE_ENV, "true")
    monkeypatch.setenv(VAST_INSTANCE_LAUNCH_GATE_ENV, "true")

    calls: list[tuple[str, str, object]] = []

    def fake_api_json(
        *,
        method: str,
        path: str,
        api_key: str,
        payload=None,
        timeout_seconds: int = 30,
    ):  # type: ignore[no-untyped-def]
        assert api_key == secret
        calls.append((method, path, payload))
        if method == "GET" and path == "/instances/":
            return 200, {"instances": []}
        if method == "POST" and path == "/bundles/":
            return 200, {
                "offers": [
                    {
                        "id": 101,
                        "ask_contract_id": 101,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.42,
                        "num_gpus": 1,
                        "rentable": True,
                        "verified": True,
                    }
                ]
            }
        if method == "PUT" and path == "/asks/101/":
            assert payload["image"].startswith("nvidia/cuda")  # type: ignore[index]
            assert payload["runtype"] == "ssh_direct"  # type: ignore[index]
            return 200, {"success": True, "new_contract": 555}
        if method == "GET" and path == "/instances/555/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if method == "PUT" and path == "/instances/request_logs/555":
            return 200, {"success": True, "result_url": f"https://logs.example/{len(calls)}"}
        if method == "DELETE" and path == "/instances/555/":
            return 200, {"success": True, "msg": "Instance destroyed successfully"}
        raise AssertionError((method, path))

    fetch_outputs = iter(
        [
            "echo response\nBLUEPRINT_VAST_HEARTBEAT_OK\n"
            "RTX 4090, 550.54, 24564 MiB\n---DF---\n/dev/root 10G\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n",
        ]
    )

    def fake_fetch_text(url: str, timeout_seconds: int = 30) -> str:
        assert url.startswith("https://logs.example/")
        return next(fetch_outputs)

    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._fetch_text", fake_fetch_text)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter.time.sleep", lambda _: None)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=20,
    )

    assert result["status"] == "completed"
    assert result["vast_instance_ids"] == [555]
    assert result["continuing_spend_from_this_run"] is False
    lock_manifest = _read_json(tmp_path / "vast_launch_lock_manifest.json")
    assert lock_manifest["status"] == "released"
    assert lock_manifest["lock_released"] is True
    heartbeat = _read_json(tmp_path / "vast_startup_probe_manifest.json")
    assert heartbeat["status"] == "completed"
    assert heartbeat["heartbeat_completed"] is True
    gpu = _read_json(tmp_path / "vast_gpu_sanity_report.json")
    assert gpu["status"] == "completed"
    assert gpu["nvidia_smi_visible"] is True
    isaac = _read_json(tmp_path / "vast_isaac_smoke_result.json")
    assert isaac["status"] == "blocked"
    assert "isaac_smoke_disabled_for_this_bounded_probe" in isaac["blockers"]
    teardown = _read_json(tmp_path / "vast_teardown_manifest.json")
    assert teardown["status"] == "completed"
    assert teardown["continuing_spend_from_this_run"] is False
    session_summary = _read_json(tmp_path / "vast_session_cost_summary.json")
    assert session_summary["status"] == "completed"
    assert session_summary["attempt_count"] == 1
    assert session_summary["attempts"][0]["vast_instance_ids"] == [555]
    assert session_summary["attempts"][0]["status"] == "completed"
    persisted = (tmp_path / "vast_provider_adapter_result.json").read_text(encoding="utf-8")
    assert secret not in persisted
    assert (tmp_path / "vast_final_validation.json").is_file()


def test_vast_adapter_honors_min_gpu_ram_env_in_offer_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secret = _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setenv(vpa.VAST_WAM_MIN_GPU_RAM_MB_ENV, "48000")
    selected_ask_paths: list[str] = []

    def fake_api_json(
        *,
        method: str,
        path: str,
        api_key: str,
        payload=None,
        timeout_seconds: int = 30,
    ):  # type: ignore[no-untyped-def]
        assert api_key == secret
        if method == "GET" and path == "/instances/":
            return 200, {"instances": []}
        if method == "POST" and path == "/bundles/":
            return 200, {
                "offers": [
                    {
                        "id": 201,
                        "ask_contract_id": 201,
                        "gpu_name": "RTX 3090",
                        "gpu_ram": 24576,
                        "dph_total": 0.12,
                        "driver_version": "580.159.03",
                        "machine_id": 9201,
                        "num_gpus": 1,
                        "rentable": True,
                    },
                    {
                        "id": 202,
                        "ask_contract_id": 202,
                        "gpu_name": "RTX A6000",
                        "gpu_ram": 49152,
                        "dph_total": 0.42,
                        "driver_version": "580.159.03",
                        "machine_id": 9202,
                        "num_gpus": 1,
                        "rentable": True,
                    },
                ]
            }
        if method == "PUT" and path == "/asks/202/":
            selected_ask_paths.append(path)
            return 200, {"success": True, "new_contract": 2020}
        if method == "PUT" and path == "/asks/201/":
            raise AssertionError("24GB offer should be excluded by min GPU RAM")
        if method == "GET" and path == "/instances/2020/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if method == "PUT" and path == "/instances/request_logs/2020":
            return 200, {"success": True, "result_url": "https://logs.example/min-gpu"}
        if method == "DELETE" and path == "/instances/2020/":
            return 200, {"success": True}
        raise AssertionError((method, path))

    def fake_fetch_text(url: str, timeout_seconds: int = 30) -> str:
        assert url == "https://logs.example/min-gpu"
        return (
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "RTX A6000, 580.159.03, 49140 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
        )

    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._fetch_text", fake_fetch_text)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter.time.sleep", lambda _: None)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=20,
        session_max_live_minutes=None,
    )

    assert result["status"] == "completed"
    assert selected_ask_paths == ["/asks/202/"]
    assert result["min_gpu_ram_mb"] == 48000
    offer = _read_json(tmp_path / "vast_offer_selection_manifest.json")
    assert offer["min_gpu_ram_mb"] == 48000
    assert offer["selected_offer"]["ask_contract_id"] == 202
    assert offer["selected_offer"]["gpu_ram_mb"] == 49152


def test_vast_adapter_retries_stale_offer_create_before_allocation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secret = _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setenv(vpa.VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS_ENV, "1")
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter.time.sleep", lambda *_: None)
    created_paths: list[str] = []

    def fake_api_json(
        *,
        method: str,
        path: str,
        api_key: str,
        payload=None,
        timeout_seconds: int = 30,
    ):  # type: ignore[no-untyped-def]
        assert api_key == secret
        if method == "GET" and path == "/instances/":
            return 200, {"instances": []}
        if method == "POST" and path == "/bundles/":
            return 200, {
                "offers": [
                    {
                        "id": 301,
                        "ask_contract_id": 301,
                        "gpu_name": "RTX A6000",
                        "gpu_ram": 49152,
                        "dph_total": 0.25,
                        "driver_version": "580.159.03",
                        "machine_id": 9301,
                        "num_gpus": 1,
                        "rentable": True,
                    },
                    {
                        "id": 302,
                        "ask_contract_id": 302,
                        "gpu_name": "RTX A6000",
                        "gpu_ram": 49152,
                        "dph_total": 0.26,
                        "driver_version": "580.159.03",
                        "machine_id": 9302,
                        "num_gpus": 1,
                        "rentable": True,
                    },
                ]
            }
        if method == "PUT" and path == "/asks/301/":
            created_paths.append(path)
            raise urllib.error.HTTPError(
                "https://vast.invalid/api/v0/asks/301/",
                400,
                "bad request",
                {},
                BytesIO(
                    b'{"success":false,"error":"invalid_args",'
                    b'"msg":"error 404/3603: no_such_ask Instance type is not available"}'
                ),
            )
        if method == "PUT" and path == "/asks/302/":
            created_paths.append(path)
            return 200, {"success": True, "new_contract": 3020}
        if method == "GET" and path == "/instances/3020/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if method == "PUT" and path == "/instances/request_logs/3020":
            return 200, {"success": True, "result_url": "https://logs.example/stale-retry"}
        if method == "DELETE" and path == "/instances/3020/":
            return 200, {"success": True}
        raise AssertionError((method, path))

    def fake_fetch_text(url: str, timeout_seconds: int = 30) -> str:
        assert url == "https://logs.example/stale-retry"
        return (
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "RTX A6000, 580.159.03, 49140 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
        )

    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._fetch_text", fake_fetch_text)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=20,
        session_max_live_minutes=None,
    )

    assert result["status"] == "completed"
    assert created_paths == ["/asks/301/", "/asks/302/"]
    assert result["vast_instance_ids"] == [3020]
    assert result["excluded_machine_ids"] == [9301]
    offer = _read_json(tmp_path / "vast_offer_selection_manifest.json")
    assert offer["selected_offer"]["ask_contract_id"] == 302
    assert offer["create_retry_attempts"][0]["http_status_code"] == 400
    assert offer["create_retry_attempts"][0]["machine_id"] == 9301
    assert "no_such_ask" in offer["create_retry_attempts"][0]["error_preview"]
    teardown = _read_json(tmp_path / "vast_teardown_manifest.json")
    assert teardown["status"] == "completed"
    assert teardown["continuing_spend_from_this_run"] is False


def test_vast_adapter_does_not_retry_unrelated_create_http_400() -> None:
    error = urllib.error.HTTPError(
        "https://vast.invalid/api/v0/asks/301/",
        400,
        "bad request",
        {},
        BytesIO(b'{"error":"invalid_args","msg":"invalid image"}'),
    )
    assert vpa._is_stale_offer_create_http_error(error, "invalid image") is False


def test_vast_adapter_mocked_isaac_uses_args_mode_required_env_and_disk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vast_secret = "secret-vast-key"
    vast_key_file = tmp_path / "vast_api_key"
    vast_key_file.write_text(vast_secret + "\n", encoding="utf-8")
    vast_key_file.chmod(0o600)
    monkeypatch.setenv(VAST_API_KEY_FILE_ENV, str(vast_key_file))
    monkeypatch.setenv("NGC_API_KEY_FILE", str(tmp_path / "missing-ngc-key"))
    monkeypatch.setenv(VAST_API_GATE_ENV, "true")
    monkeypatch.setenv(VAST_INSTANCE_LAUNCH_GATE_ENV, "true")

    created_payloads: list[dict[str, object]] = []

    def fake_api_json(
        *,
        method: str,
        path: str,
        api_key: str,
        payload=None,
        timeout_seconds: int = 30,
    ):  # type: ignore[no-untyped-def]
        assert api_key == vast_secret
        if method == "POST" and path == "/bundles/":
            return 200, {
                "offers": [
                    {
                        "id": 202,
                        "ask_contract_id": 202,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.21,
                        "num_gpus": 1,
                        "rentable": True,
                        "verified": True,
                    }
                ]
            }
        if method == "GET" and path == "/instances/":
            return 200, {"instances": []}
        if method == "PUT" and path == "/asks/202/":
            assert payload is not None
            created_payloads.append(dict(payload))
            assert payload["image"] == DEFAULT_ISAAC_IMAGE
            assert payload["runtype"] == "args"
            assert payload["disk"] == DEFAULT_ISAAC_DISK_GB
            assert "entrypoint" not in payload
            assert "onstart" not in payload
            assert "args" not in payload
            assert payload["args_str"].startswith("bash -lc ")
            assert "BLUEPRINT_VAST_ISAAC_SMOKE_OK" in payload["args_str"]
            assert "BLUEPRINT_VAST_ARGS_LOG_HOLD_STARTED" in payload["args_str"]
            assert payload["env"]["ACCEPT_EULA"] == "Y"
            assert payload["env"]["PRIVACY_CONSENT"] == "Y"
            assert payload["env"]["NVIDIA_DRIVER_CAPABILITIES"] == "all"
            assert "image_login" not in payload
            return 200, {"success": True, "new_contract": 777}
        if method == "GET" and path == "/instances/777/":
            return 200, {"instances": {"actual_status": "exited", "cur_state": "exited"}}
        if method == "PUT" and path == "/instances/request_logs/777":
            return 200, {"success": True, "result_url": "https://logs.example/isaac"}
        if method == "DELETE" and path == "/instances/777/":
            return 200, {"success": True, "msg": "Instance destroyed successfully"}
        raise AssertionError((method, path))

    fetch_outputs = iter(
        [
            "BLUEPRINT_VAST_ONSTART_STARTED\n"
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "RTX 4090, 550.54, 24564 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n",
            "BLUEPRINT_VAST_ONSTART_STARTED\n"
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "RTX 4090, 550.54, 24564 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
            "BLUEPRINT_VAST_ISAAC_SMOKE_OK\n"
            "BLUEPRINT_VAST_ONSTART_DONE\n",
        ]
    )

    def fake_fetch_text(url: str, timeout_seconds: int = 30) -> str:
        assert url == "https://logs.example/isaac"
        return next(fetch_outputs)

    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._fetch_text", fake_fetch_text)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter.time.sleep", lambda _: None)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        enable_isaac_smoke=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=20,
    )

    assert created_payloads
    assert result["status"] == "completed"
    assert result["selected_container_image"] == DEFAULT_ISAAC_IMAGE
    assert result["vast_launch_mode"] == "args"
    assert result["disk_gb"] == DEFAULT_ISAAC_DISK_GB
    startup = _read_json(tmp_path / "vast_startup_probe_manifest.json")
    assert startup["status"] == "completed"
    assert len(startup["container_log_result"]["log_poll_attempts"]) == 2
    assert startup["launch_mode_used"] == "args"
    assert startup["disk_gb"] == DEFAULT_ISAAC_DISK_GB
    summary = startup["create_request_summary"]
    assert summary["onstart_present"] is False
    assert summary["args_str_present"] is True
    assert summary["args_str_length"] > 0
    assert "ACCEPT_EULA" in summary["env_keys"]
    assert summary["isaac_required_env_present"]["ACCEPT_EULA"] is True
    assert summary["image_login_supplied"] is False
    assert (
        startup["image_login_summary"]["reason"]
        == "public_official_isaac_image_without_registry_login"
    )
    assert startup["image_login_summary"]["ngc_secret_file_present"] is False
    assert startup["container_image"] == DEFAULT_ISAAC_IMAGE
    isaac = _read_json(tmp_path / "vast_isaac_smoke_result.json")
    assert isaac["status"] == "completed"
    assert isaac["isaac_simulation_app_started"] is True
    persisted = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.json"))
    assert vast_secret not in persisted


def test_vast_adapter_prefers_known_supported_driver_over_known_unsupported_driver() -> None:
    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "RTX 4090",
                "dph_total": 0.20,
                "driver_version": "570.86.10",
            },
            {
                "id": 2,
                "ask_contract_id": 2,
                "gpu_name": "RTX 4090",
                "dph_total": 0.31,
                "driver_version": "580.95.05",
            },
        ],
        max_hourly_rate=0.60,
    )

    assert selected is not None
    assert selected["ask_contract_id"] == 2
    assert (
        selected["isaac_driver_support_status"]
        == "outside_known_unsupported_omniverse_rtx_driver_range"
    )


def test_vast_adapter_enforces_backend_minimum_driver_version() -> None:
    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "L40",
                "dph_total": 0.20,
                "driver_version": "580.82.09",
            },
            {
                "id": 2,
                "ask_contract_id": 2,
                "gpu_name": "RTX 6000 Ada Generation",
                "dph_total": 0.31,
                "driver_version": "580.119.02",
            },
        ],
        max_hourly_rate=0.60,
        minimum_driver_version="580.95.05",
    )

    assert selected is not None
    assert selected["ask_contract_id"] == 2


def test_vast_adapter_can_require_known_supported_driver_for_rendering() -> None:
    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "RTX 4090",
                "dph_total": 0.20,
                "driver_version": "570.86.10",
            },
            {
                "id": 2,
                "ask_contract_id": 2,
                "gpu_name": "RTX 4090",
                "dph_total": 0.25,
                "driver_version": "",
            },
        ],
        max_hourly_rate=0.60,
        require_known_supported_isaac_driver=True,
    )

    assert selected is None


def test_vast_adapter_prefers_newer_supported_driver_branch() -> None:
    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "RTX 3090",
                "dph_total": 0.15,
                "driver_version": "565.57.01",
            },
            {
                "id": 2,
                "ask_contract_id": 2,
                "gpu_name": "RTX 3090",
                "dph_total": 0.20,
                "driver_version": "580.159.03",
            },
        ],
        max_hourly_rate=0.60,
    )

    assert selected is not None
    assert selected["ask_contract_id"] == 2
    assert selected["driver_version"] == "580.159.03"


def test_vast_adapter_can_require_minimum_gpu_memory() -> None:
    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "RTX 3090",
                "gpu_ram_mb": 24576,
                "dph_total": 0.15,
                "driver_version": "580.159.03",
            },
            {
                "id": 2,
                "ask_contract_id": 2,
                "gpu_name": "RTX 6000 Ada",
                "gpu_ram_mb": 49152,
                "dph_total": 0.82,
                "driver_version": "580.159.03",
            },
        ],
        max_hourly_rate=1.00,
        min_gpu_ram_mb=48000,
    )

    assert selected is not None
    assert selected["ask_contract_id"] == 2
    assert selected["gpu_ram_mb"] == 49152


def test_vast_adapter_caps_host_total_memory_for_known_4090_model() -> None:
    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "RTX 4090",
                # Live Vast rows can expose the two-card host total here even
                # though this ask allocates only one 24 GB GPU.
                "gpu_ram": 49140,
                "gpu_frac": 0.5,
                "num_gpus": 1,
                "dph_total": 0.40,
                "driver_version": "590.48.01",
            },
            {
                "id": 2,
                "ask_contract_id": 2,
                "gpu_name": "RTX 6000Ada",
                "gpu_ram": 49140,
                "gpu_frac": 1.0,
                "num_gpus": 1,
                "dph_total": 0.60,
                "driver_version": "580.119.02",
            },
        ],
        max_hourly_rate=1.00,
        min_gpu_ram_mb=40000,
    )

    assert selected is not None
    assert selected["ask_contract_id"] == 2
    assert selected["gpu_ram_mb"] == 49140
    rejected_4090 = vpa._offer_summary(
        {
            "id": 1,
            "gpu_name": "RTX 4090",
            "gpu_ram": 49140,
        }
    )
    assert rejected_4090["provider_reported_gpu_ram_mb"] == 49140
    assert rejected_4090["known_model_vram_cap_mb"] == 24576
    assert rejected_4090["gpu_ram_mb"] == 24576
    assert rejected_4090["gpu_ram_normalization"] == "known_model_cap_applied"


def test_vast_adapter_can_require_minimum_compute_capability() -> None:
    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "Quadro RTX 8000",
                "gpu_ram_mb": 49152,
                "compute_cap": 750,
                "dph_total": 0.24,
                "driver_version": "580.95.05",
            },
            {
                "id": 2,
                "ask_contract_id": 2,
                "gpu_name": "RTX 6000Ada",
                "gpu_ram_mb": 49140,
                "compute_cap": 890,
                "dph_total": 0.57,
                "driver_version": "580.142",
            },
        ],
        max_hourly_rate=0.60,
        min_gpu_ram_mb=48000,
        min_compute_cap=800,
        prefer_isaac_rt=False,
    )

    assert selected is not None
    assert selected["ask_contract_id"] == 2
    assert selected["compute_cap_normalized"] == 890


def test_vast_adapter_rejects_gpus_above_tensorrt_compute_cap_by_default() -> None:
    """Blackwell (sm_120) cannot build the pinned TensorRT 10.4 policy engine.

    Observed live on Vast instance 45771989 (RTX PRO 6000 WS, compute_cap 1200):
    the GEAR-SONIC controller tried to convert policy/release/model_decoder.onnx
    at startup and TensorRT failed with ``Error Code 10: Could not find any
    implementation``, so the controller never came ready and the episode exited 1.

    The ceiling must be a DEFAULT, not opt-in: every current and future Vast
    selection path has to inherit it, otherwise the next lane silently rents an
    incompatible GPU again.
    """

    offers = [
        {
            "id": 1,
            "ask_contract_id": 1,
            "gpu_name": "RTX PRO 6000 WS",
            "gpu_ram_mb": 49140,
            "compute_cap": 1200,
            "dph_total": 0.30,
            "driver_version": "580.95.05",
        },
        {
            "id": 2,
            "ask_contract_id": 2,
            "gpu_name": "RTX 4090",
            "gpu_ram_mb": 24564,
            "compute_cap": 890,
            "dph_total": 0.90,
            "driver_version": "580.159.03",
        },
    ]

    # Blackwell is both cheaper and higher-VRAM here, so only an architecture
    # ceiling can reject it -- a rate or memory filter cannot.
    selected = _select_offer(offers, max_hourly_rate=1.00, prefer_isaac_rt=False)

    assert selected is not None
    assert selected["ask_contract_id"] == 2, "must not select the sm_120 offer"
    assert selected["compute_cap_normalized"] == 890


def test_vast_adapter_compute_cap_ceiling_allows_offers_without_compute_cap() -> None:
    """An unreported architecture is allowed through, deliberately.

    Live Vast offers always carry ``compute_cap`` (verified against the live
    catalogue: 1200/890/860/700), so rejecting unknowns would not catch the
    incompatibility this ceiling exists for.  It would only convert an upstream
    schema change into a total selection outage.  The ceiling is strict about
    architectures it can prove unusable and permissive about ones it cannot see.
    """

    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "Mystery GPU",
                "gpu_ram_mb": 49140,
                "dph_total": 0.10,
                "driver_version": "580.95.05",
            }
        ],
        max_hourly_rate=1.00,
        prefer_isaac_rt=False,
    )

    assert selected is not None
    assert selected["compute_cap_normalized"] is None


def test_vast_adapter_compute_cap_ceiling_can_be_lifted_explicitly() -> None:
    """A lane that does not build a TensorRT engine may opt out."""

    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "RTX PRO 6000 WS",
                "gpu_ram_mb": 49140,
                "compute_cap": 1200,
                "dph_total": 0.30,
                "driver_version": "580.95.05",
            }
        ],
        max_hourly_rate=1.00,
        max_compute_cap=0,
        prefer_isaac_rt=False,
    )

    assert selected is not None
    assert selected["compute_cap_normalized"] == 1200


def test_vast_public_adapter_exposes_the_compute_ceiling_opt_out() -> None:
    """PR #181 review P1: the documented opt-out must be reachable.

    The ceiling defaults on so a future TensorRT lane inherits it, but callers
    running a workload that never builds the pinned engine (generic CUDA probes,
    WAM) must be able to lift it.  Without a public parameter those callers see
    Blackwell-only capacity reported as unavailable with no recourse.
    """

    import inspect

    from blueprint_pipeline.vast_provider_adapter import run_vast_provider_adapter

    params = inspect.signature(run_vast_provider_adapter).parameters
    assert "max_compute_cap" in params, "opt-out unreachable from the public entry point"
    assert params["max_compute_cap"].default is None, "must default to the safe ceiling"


def test_vast_offer_manifest_records_architecture_exclusions() -> None:
    """PR #181 review P2: evidence must name architecture as the reason.

    Otherwise an operator debugging "no capacity" sees only a rate-shaped
    blocker and cannot tell that an affordable, high-VRAM offer was rejected for
    compatibility -- the exact diagnostic gap that let this lane rent an sm_120
    GPU in the first place.
    """

    manifest = _offer_selection_manifest(
        generated_at="2026-07-25T00:00:00Z",
        status_code=200,
        offers=[
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "RTX PRO 6000 WS",
                "gpu_ram_mb": 49140,
                "compute_cap": 1200,
                "dph_total": 0.30,
                "driver_version": "580.95.05",
            }
        ],
        selected_offer=None,
        max_hourly_rate=1.00,
        min_gpu_ram_mb=40_000,
        require_known_supported_isaac_driver=False,
        excluded_machine_ids=(),
        allowed_machine_ids=(),
        machine_avoidlist_path=Path("/tmp/avoidlist.json"),
        avoidlist_status=None,
        blockers=[],
        prefer_isaac_rt=False,
    )

    assert manifest["max_compute_cap"] == 900
    assert manifest["architecture_excluded_offer_count"] == 1


def test_vast_adapter_excludes_avoidlisted_machine_ids() -> None:
    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "RTX 3090",
                "dph_total": 0.15,
                "driver_version": "580.159.03",
                "machine_id": 101,
            },
            {
                "id": 2,
                "ask_contract_id": 2,
                "gpu_name": "RTX 3090",
                "dph_total": 0.22,
                "driver_version": "580.159.03",
                "machine_id": 202,
            },
        ],
        max_hourly_rate=0.60,
        excluded_machine_ids=[101],
    )

    assert selected is not None
    assert selected["ask_contract_id"] == 2
    assert selected["machine_id"] == 202


def test_vast_adapter_restricts_to_allowed_machine_ids() -> None:
    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "RTX A6000",
                "gpu_ram_mb": 49140,
                "dph_total": 0.31,
                "driver_version": "580.159.03",
                "machine_id": 111,
            },
            {
                "id": 2,
                "ask_contract_id": 2,
                "gpu_name": "RTX A6000",
                "gpu_ram_mb": 49140,
                "dph_total": 0.42,
                "driver_version": "580.159.03",
                "machine_id": 222,
            },
        ],
        max_hourly_rate=0.60,
        min_gpu_ram_mb=48000,
        allowed_machine_ids=[222],
    )

    assert selected is not None
    assert selected["ask_contract_id"] == 2
    assert selected["machine_id"] == 222


def test_vast_adapter_wam_selection_can_prefer_workstation_gpu_over_isaac_rt_pool() -> None:
    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "RTX 4090",
                "gpu_ram_mb": 49140,
                "dph_total": 0.39,
                "driver_version": "580.159.03",
                "machine_id": 111,
                "reliability": 0.991,
                "direct_port_count": 99,
                "geolocation": "cn",
            },
            {
                "id": 2,
                "ask_contract_id": 2,
                "gpu_name": "RTX A6000",
                "gpu_ram_mb": 49140,
                "dph_total": 0.55,
                "driver_version": "580.159.03",
                "machine_id": 222,
                "reliability": 0.995,
                "direct_port_count": 32,
                "geolocation": "california_us",
            },
        ],
        max_hourly_rate=0.80,
        min_gpu_ram_mb=48000,
        min_reliability=0.99,
        require_direct_port=True,
        preferred_gpu_keywords=("RTX A6000", "L40S", "A100"),
        preferred_geolocation_regex="california|oregon|texas",
        prefer_isaac_rt=False,
    )

    assert selected is not None
    assert selected["ask_contract_id"] == 2
    assert selected["machine_id"] == 222


def test_vast_adapter_inline_policy_rejects_offer_below_cuda_floor() -> None:
    selected = _select_offer(
        [
            {
                "id": 1,
                "ask_contract_id": 1,
                "gpu_name": "H100 SXM",
                "gpu_ram_mb": 81_559,
                "dph_total": 1.70,
                "driver_version": "555.58.02",
                "cuda_max_good": 12.5,
                "machine_id": 111,
            },
            {
                "id": 2,
                "ask_contract_id": 2,
                "gpu_name": "H100 NVL",
                "gpu_ram_mb": 94_000,
                "dph_total": 2.00,
                "driver_version": "590.48.01",
                "cuda_max_good": 13.1,
                "machine_id": 222,
            },
        ],
        max_hourly_rate=2.50,
        min_gpu_ram_mb=80_000,
        prefer_isaac_rt=False,
        gpu_selection_policy={
            "policy_id": "reasoner_cuda_floor",
            "allowed_gpu_keywords": ["H100"],
            "minimum_cuda_max_good": 12.8,
        },
    )

    assert selected is not None
    assert selected["ask_contract_id"] == 2
    assert selected["cuda_max_good"] == 13.1


def test_vast_adapter_mocked_blueprint_bundle_run_uploads_and_inspects_zip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vast_secret = "secret-vast-key"
    ngc_secret = "secret-ngc-key"
    tunnel_token = "secret-tunnel-token"
    vast_key_file = tmp_path / "vast_api_key"
    ngc_key_file = tmp_path / "ngc_api_key"
    vast_key_file.write_text(vast_secret + "\n", encoding="utf-8")
    ngc_key_file.write_text(ngc_secret + "\n", encoding="utf-8")
    vast_key_file.chmod(0o600)
    ngc_key_file.chmod(0o600)
    monkeypatch.setenv(VAST_API_KEY_FILE_ENV, str(vast_key_file))
    monkeypatch.setenv("NGC_API_KEY_FILE", str(ngc_key_file))
    monkeypatch.setenv(VAST_API_GATE_ENV, "true")
    monkeypatch.setenv(VAST_INSTANCE_LAUNCH_GATE_ENV, "true")

    provider_bundle = tmp_path / "isaac_provider_runtime_bundle.zip"
    _write_valid_provider_bundle(provider_bundle)
    runtime_output_zip = tmp_path / "vast_provider_runtime_output.zip"
    import zipfile

    with zipfile.ZipFile(runtime_output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "isaac_runtime_result.json",
            json.dumps(
                {
                    "status": "blocked_controller_runtime_unavailable",
                    "blockers": ["real_unitree_g1_controller_policy_stack_not_packaged"],
                    "controller_grade_execution_proven": False,
                    "official_policy_execution_proven": False,
                }
            ),
        )

    created_payloads: list[dict[str, object]] = []

    def fake_api_json(
        *,
        method: str,
        path: str,
        api_key: str,
        payload=None,
        timeout_seconds: int = 30,
    ):  # type: ignore[no-untyped-def]
        assert api_key == vast_secret
        if method == "POST" and path == "/bundles/":
            return 200, {
                "offers": [
                    {
                        "id": 303,
                        "ask_contract_id": 303,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.31,
                        "driver_version": "580.95.05",
                        "num_gpus": 1,
                        "rentable": True,
                        "verified": True,
                    }
                ]
            }
        if method == "GET" and path == "/instances/":
            return 200, {"instances": []}
        if method == "PUT" and path == "/asks/303/":
            assert payload is not None
            created_payloads.append(dict(payload))
            assert payload["image"] == DEFAULT_ISAAC_IMAGE
            assert payload["runtype"] == "args"
            assert "onstart" not in payload
            assert "args" not in payload
            assert payload["args_str"].startswith("bash -lc ")
            assert "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED" in payload["args_str"]
            assert "BLUEPRINT_VAST_WORK_DIR:$WORK_DIR" in payload["args_str"]
            assert "/tmp/blueprint_vast_work" in payload["args_str"]
            assert "BLUEPRINT_VAST_ARGS_LOG_HOLD_STARTED" in payload["args_str"]
            env = payload["env"]
            assert env["BLUEPRINT_EVAL_MANIFEST_URI"].endswith(tunnel_token)
            assert env["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"].endswith(tunnel_token)
            return 200, {"success": True, "new_contract": 888}
        if method == "GET" and path == "/instances/888/":
            return 200, {"instances": {"actual_status": "exited", "cur_state": "exited"}}
        if method == "PUT" and path == "/instances/request_logs/888":
            return 200, {"success": True, "result_url": "https://logs.example/provider"}
        if method == "DELETE" and path == "/instances/888/":
            return 200, {"success": True, "msg": "Instance destroyed successfully"}
        raise AssertionError((method, path))

    def fake_fetch_text(url: str, timeout_seconds: int = 30) -> str:
        assert url == "https://logs.example/provider"
        return (
            "BLUEPRINT_VAST_ONSTART_STARTED\n"
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "RTX 4090, 580.95.05, 24564 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
            "BLUEPRINT_VAST_ISAAC_SMOKE_OK\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED\n"
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED\n"
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE:2\n"
            "BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN:512\n"
            "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK\n"
            '{"ok": true, "bytes": 512}\n'
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED\n"
            "BLUEPRINT_VAST_ONSTART_DONE\n"
        )

    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._fetch_text", fake_fetch_text)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter.time.sleep", lambda _: None)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle=provider_bundle,
        provider_bundle_url=f"https://example.invalid/bundle.zip?token={tunnel_token}",
        provider_output_put_url=f"https://example.invalid/output.zip?token={tunnel_token}",
        provider_runtime_output_zip=runtime_output_zip,
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=20,
    )

    assert created_payloads
    assert result["status"] == "blocked"
    assert result["reason"] == "vast_blueprint_video_smoke_blocked"
    assert "mp4_count_below_expected_video_smoke_camera_count" in result["blockers"]
    provider = _read_json(tmp_path / "vast_provider_command_result.json")
    assert provider["status"] == "completed"
    assert provider["provider_command_path_remote_proven"] is True
    assert provider["provider_runtime_output_zip_received"] is True
    assert provider["provider_entrypoint_exit_code"] == 2
    assert provider["blueprint_provider_bundle_execution_proven"] is True
    assert provider["video_smoke_proven"] is False
    assert provider["runtime_result_status"] == "blocked_controller_runtime_unavailable"
    assert (
        "real_unitree_g1_controller_policy_stack_not_packaged"
        in provider["runtime_result_blockers"]
    )
    video = _read_json(tmp_path / "vast_video_smoke_result.json")
    assert video["status"] == "blocked"
    assert video["video_smoke_proven"] is False
    assert video["expected_video_count"] == 6
    assert "mp4_count_below_expected_video_smoke_camera_count" in video["blockers"]
    persisted = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.json"))
    assert vast_secret not in persisted
    assert ngc_secret not in persisted
    assert tunnel_token not in persisted


def test_structured_policy_canary_output_does_not_require_video(tmp_path: Path) -> None:
    native_action = [[float(row * 8 + column) for column in range(8)] for row in range(32)]
    receipt = {
        "native_action_shape": [32, 8],
        "wam_prefix_action_shape": [16, 8],
        "executed_prefix_steps": 8,
        "server_identity_sha256": "a" * 64,
        "observation_sha256": "b" * 64,
        "native_action_sha256": "c" * 64,
        "wam_prefix_action_sha256": "d" * 64,
        "executed_prefix_action_sha256": "e" * 64,
        "commanded_next_state_sha256": "f" * 64,
        "receipt_sha256": "1" * 64,
    }
    output_zip = tmp_path / "policy-output.zip"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "policy_structured_canary.json",
            json.dumps(
                {
                    "status": "passed",
                    "native_action": native_action,
                    "wam_prefix_action": native_action[:16],
                    "executed_action": native_action[:8],
                    "commanded_next_joint_position": native_action[7][:7],
                    "commanded_next_gripper_position": [native_action[7][7]],
                    "policy_endpoint_evidence": {
                        "identity_verified": True,
                        "request_count": 1,
                        "server_metadata": {
                            "policy_id": "model/policy",
                            "model_revision": "2" * 40,
                        },
                    },
                    "policy_request_receipt": receipt,
                }
            ),
        )

    summary = vpa._inspect_structured_policy_canary_output(output_zip)
    runtime_result = {
        "status": "completed",
        "runtime": "policy_structured_canary",
        "structured_policy_canary_passed": True,
        "learned_wam_model_ran": False,
        "action_conditioned_video_rollout_generated": False,
        "native_action_shape": [32, 8],
        "wam_prefix_action_shape": [16, 8],
        "executed_prefix_steps": 8,
        "commanded_state_advance_proven": True,
    }

    assert summary["status"] == "passed"
    assert summary["native_action_sha256"] == "c" * 64
    assert vpa._structured_policy_canary_runtime_passed(runtime_result, summary) is True
    assert vpa._provider_expected_video_count("wam") > 0
    assert (
        vpa._provider_expected_video_count_for_result("wam", structured_policy_canary_passed=True)
        == 0
    )

    runtime_result["learned_wam_model_ran"] = True
    assert vpa._structured_policy_canary_runtime_passed(runtime_result, summary) is False


def test_vast_adapter_unitree_groot_bundle_completes_without_video_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vast_secret = _configure_live_gates(tmp_path, monkeypatch)
    docker_username = "nijelhunt"
    docker_pat = "secret-docker-pat"
    username_file = tmp_path / "docker_username"
    pat_file = tmp_path / "docker_pat"
    _write_secret(username_file, docker_username)
    _write_secret(pat_file, docker_pat)
    monkeypatch.setenv(vpa.DOCKER_USERNAME_FILE_ENV, str(username_file))
    monkeypatch.setenv(vpa.DOCKER_PAT_FILE_ENV, str(pat_file))
    provider_bundle = tmp_path / "unitree_groot_n17_sonic_provider_bundle.zip"
    _write_valid_unitree_groot_n17_sonic_provider_bundle(provider_bundle)
    runtime_output_zip = tmp_path / "vast_provider_runtime_output.zip"
    with zipfile.ZipFile(runtime_output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "schema_version": "unitree_groot_n17_sonic_policy_provider_output.v1",
                    "status": "completed",
                    "canary_only": True,
                    "unitree_groot_n17_sonic_model_executed": False,
                    "unitree_groot_n17_sonic_policy_action_command_ran": False,
                    "policy_action_model_command_ran": False,
                    "action": None,
                    "blockers": [],
                }
            ),
        )

    def fake_api_json(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["api_key"] == vast_secret
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            return 200, {
                "offers": [
                    {
                        "id": 424,
                        "ask_contract_id": 424,
                        "gpu_name": "RTX 3090",
                        "gpu_ram_mb": 24576,
                        "dph_total": 0.13,
                        "driver_version": "580.159.03",
                        "machine_id": 4242,
                    }
                ]
            }
        if kwargs["method"] == "PUT" and kwargs["path"] == "/asks/424/":
            payload = kwargs["payload"]
            assert payload["image"] == (
                "docker.io/nijelhunt/blueprint-vast-unitree-groot-sonic:20260624-pydeps-vast1"
            )
            assert payload["runtype"] == "ssh_direct"
            assert payload["image_login"].startswith("-u nijelhunt -p ")
            return 200, {"new_contract": 4241}
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/4241/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if kwargs["method"] == "PUT" and kwargs["path"] == "/instances/request_logs/4241":
            return 200, {"success": True, "result_url": "https://logs.example/unitree"}
        if kwargs["method"] == "DELETE":
            return 200, {"success": True}
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(
        vpa,
        "_fetch_text",
        lambda *_args, **_kwargs: (
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "NVIDIA GeForce RTX 3090, 580.159.03, 24576 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED\n"
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED\n"
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE:0\n"
            "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED\n"
            "BLUEPRINT_VAST_ONSTART_DONE\n"
        ),
    )
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)

    result = run_vast_provider_adapter(
        job_dir=tmp_path / "unitree-live",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        public_image="docker.io/nijelhunt/blueprint-vast-unitree-groot-sonic:20260624-pydeps-vast1",
        provider_bundle=provider_bundle,
        provider_bundle_url="https://example.invalid/unitree.zip",
        provider_output_put_url="https://example.invalid/out.zip",
        provider_runtime_output_zip=runtime_output_zip,
        enable_blueprint_bundle=True,
        provider_bundle_kind="unitree_groot_n17_sonic",
        ngc_image_login_mode="auto",
        poll_interval_seconds=0,
        startup_timeout_seconds=10,
        session_max_live_minutes=None,
    )

    assert result["status"] == "completed"
    assert result["reason"] == "vast_startup_probe_completed"
    provider = _read_json(tmp_path / "unitree-live" / "vast_provider_command_result.json")
    assert provider["status"] == "completed"
    assert provider["blueprint_provider_bundle_execution_proven"] is True
    assert provider["video_smoke_expected_video_count"] == 0
    video = _read_json(tmp_path / "unitree-live" / "vast_video_smoke_result.json")
    assert video["status"] == "not_required"
    assert video["expected_video_count"] == 0
    persisted = "\n".join(
        path.read_text(encoding="utf-8") for path in (tmp_path / "unitree-live").glob("*.json")
    )
    assert vast_secret not in persisted
    assert docker_pat not in persisted


def test_vast_adapter_infers_provider_start_when_log_tail_drops_early_markers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vast_secret = "secret-vast-key"
    tunnel_token = "secret-tunnel-token"
    vast_key_file = tmp_path / "vast_api_key"
    vast_key_file.write_text(vast_secret + "\n", encoding="utf-8")
    vast_key_file.chmod(0o600)
    monkeypatch.setenv(VAST_API_KEY_FILE_ENV, str(vast_key_file))
    monkeypatch.setenv(VAST_API_GATE_ENV, "true")
    monkeypatch.setenv(VAST_INSTANCE_LAUNCH_GATE_ENV, "true")

    provider_bundle = tmp_path / "isaac_provider_runtime_bundle.zip"
    _write_valid_provider_bundle(provider_bundle)
    runtime_output_zip = tmp_path / "vast_provider_runtime_output.zip"
    import zipfile

    with zipfile.ZipFile(runtime_output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "isaac_runtime_phase_log.jsonl",
            '{"phase":"runner_referencing_official_g1","status":"running"}\n',
        )

    def fake_api_json(
        *,
        method: str,
        path: str,
        api_key: str,
        payload=None,
        timeout_seconds: int = 30,
    ):  # type: ignore[no-untyped-def]
        assert api_key == vast_secret
        if method == "POST" and path == "/bundles/":
            return 200, {
                "offers": [
                    {
                        "id": 404,
                        "ask_contract_id": 404,
                        "gpu_name": "RTX 3090",
                        "dph_total": 0.20,
                        "driver_version": "580.159.03",
                        "num_gpus": 1,
                        "rentable": True,
                    }
                ]
            }
        if method == "GET" and path == "/instances/":
            return 200, {"instances": []}
        if method == "PUT" and path == "/asks/404/":
            return 200, {"success": True, "new_contract": 889}
        if method == "GET" and path == "/instances/889/":
            return 200, {"instances": {"actual_status": "exited", "cur_state": "exited"}}
        if method == "PUT" and path == "/instances/request_logs/889":
            return 200, {"success": True, "result_url": "https://logs.example/tail"}
        if method == "DELETE" and path == "/instances/889/":
            return 200, {"success": True}
        raise AssertionError((method, path))

    def fake_fetch_text(url: str, timeout_seconds: int = 30) -> str:
        assert url == "https://logs.example/tail"
        return (
            "BLUEPRINT_VAST_ONSTART_STARTED\n"
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "RTX 3090, 580.159.03, 24564 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE:139\n"
            "BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN:294\n"
            "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED\n"
            "BLUEPRINT_VAST_ONSTART_DONE\n"
        )

    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._fetch_text", fake_fetch_text)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter.time.sleep", lambda _: None)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        public_image=DEFAULT_ISAAC_IMAGE,
        provider_bundle=provider_bundle,
        provider_bundle_url=f"https://example.invalid/bundle.zip?token={tunnel_token}",
        provider_output_put_url=f"https://example.invalid/output.zip?token={tunnel_token}",
        provider_runtime_output_zip=runtime_output_zip,
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=20,
    )

    assert result["status"] == "blocked"
    provider = _read_json(tmp_path / "vast_provider_command_result.json")
    assert provider["provider_bundle_started"] is True
    assert provider["provider_bundle_downloaded"] is True
    assert provider["provider_entrypoint_started"] is True
    assert provider["provider_entrypoint_exit_code"] == 139
    assert "provider_runtime_result_missing_from_output_zip" in provider["blockers"]
    assert "provider_bundle_start_marker_missing" not in provider["blockers"]
    assert "provider_bundle_download_marker_missing" not in provider["blockers"]
    assert "provider_entrypoint_start_marker_missing" not in provider["blockers"]


def test_vast_adapter_records_machine_avoidlist_on_heartbeat_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vast_secret = "secret-vast-key"
    vast_key_file = tmp_path / "vast_api_key"
    vast_key_file.write_text(vast_secret + "\n", encoding="utf-8")
    vast_key_file.chmod(0o600)
    monkeypatch.setenv(VAST_API_KEY_FILE_ENV, str(vast_key_file))
    monkeypatch.setenv(VAST_API_GATE_ENV, "true")
    monkeypatch.setenv(VAST_INSTANCE_LAUNCH_GATE_ENV, "true")

    def fake_api_json(
        *,
        method: str,
        path: str,
        api_key: str,
        payload=None,
        timeout_seconds: int = 30,
    ):  # type: ignore[no-untyped-def]
        assert api_key == vast_secret
        if method == "POST" and path == "/bundles/":
            return 200, {
                "offers": [
                    {
                        "id": 505,
                        "ask_contract_id": 505,
                        "gpu_name": "RTX 3090",
                        "dph_total": 0.20,
                        "driver_version": "580.159.03",
                        "machine_id": 909,
                        "num_gpus": 1,
                        "rentable": True,
                    }
                ]
            }
        if method == "GET" and path == "/instances/":
            return 200, {"instances": []}
        if method == "PUT" and path == "/asks/505/":
            return 200, {"success": True, "new_contract": 990}
        if method == "GET" and path == "/instances/990/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if method == "PUT" and path == "/instances/request_logs/990":
            return 200, {"success": True, "result_url": "https://logs.example/heartbeat-blocked"}
        if method == "DELETE" and path == "/instances/990/":
            return 200, {"success": True}
        raise AssertionError((method, path))

    def fake_fetch_text(url: str, timeout_seconds: int = 30) -> str:
        assert url == "https://logs.example/heartbeat-blocked"
        return "BLUEPRINT_VAST_ONSTART_STARTED\nBLUEPRINT_VAST_HEARTBEAT_BLOCKED:7\n"

    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._fetch_text", fake_fetch_text)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter.time.sleep", lambda _: None)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=20,
    )

    assert result["status"] == "failed"
    assert result["continuing_spend_from_this_run"] is False
    avoidlist = _read_json(tmp_path / "vast_machine_avoidlist.json")
    assert avoidlist["machine_ids"] == [909]
    assert avoidlist["entries"][0]["instance_id"] == 990
    assert avoidlist["entries"][0]["reason"] == (
        "vast_startup_control_plane_did_not_reach_onstart_heartbeat"
    )
    offer = _read_json(tmp_path / "vast_offer_selection_manifest.json")
    assert offer["selected_offer"]["machine_id"] == 909


def test_vast_adapter_heartbeat_no_progress_has_startup_specific_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setenv(vpa.VAST_WAM_NO_PROGRESS_SECONDS_ENV, "1200")
    monkeypatch.setenv(vpa.VAST_HEARTBEAT_NO_PROGRESS_SECONDS_ENV, "2")
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 1.0
        return clock["now"]

    def fake_api_json(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["api_key"] == secret
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            return 200, {
                "offers": [
                    {
                        "id": 606,
                        "ask_contract_id": 606,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.2,
                        "driver_version": "580.95.05",
                        "machine_id": 6060,
                        "num_gpus": 1,
                        "rentable": True,
                    }
                ]
            }
        if kwargs["method"] == "PUT" and kwargs["path"] == "/asks/606/":
            return 200, {"success": True, "new_contract": 6061}
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/6061/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if kwargs["method"] == "PUT" and kwargs["path"] == "/instances/request_logs/6061":
            return 200, {"success": True, "result_url": "https://logs.example/empty-startup"}
        if kwargs["method"] == "DELETE" and kwargs["path"] == "/instances/6061/":
            return 200, {"success": True}
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(vpa, "_fetch_text", lambda *_args, **_kwargs: "")

    result = run_vast_provider_adapter(
        job_dir=tmp_path / "empty-heartbeat",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=1200,
        session_max_live_minutes=None,
    )

    assert result["status"] == "failed"
    assert result["blockers"] == ["vast_heartbeat_no_log_progress_timeout"]
    assert result["continuing_spend_from_this_run"] is False
    startup = _read_json(tmp_path / "empty-heartbeat" / "vast_startup_probe_manifest.json")
    assert startup["status"] == "blocked"
    assert startup["blockers"][0] == "vast_heartbeat_no_log_progress_timeout"
    log_result = startup["container_log_result"]
    assert log_result["break_reason"] == "no_log_progress_timeout"
    assert log_result["no_progress_timeout_seconds"] == 2
    avoidlist = _read_json(tmp_path / "empty-heartbeat" / "vast_machine_avoidlist.json")
    assert avoidlist["machine_ids"] == [6060]


def test_vast_adapter_accepts_downstream_markers_when_heartbeat_url_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    bundle = tmp_path / "unitree_groot_n17_sonic_bundle.zip"
    _write_valid_unitree_groot_n17_sonic_provider_bundle(bundle)
    output_zip = tmp_path / "provider-output.zip"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_wam_persistent_session_output.json",
            json.dumps(
                {
                    "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_output.v1",
                    "status": "completed",
                    "blockers": [],
                    "persistent_provider_session_used": True,
                    "provider_instance_reused_for_policy_and_wam_loop": True,
                }
            ),
        )

    def fake_api_json(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["api_key"] == secret
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            return 200, {
                "offers": [
                    {
                        "id": 707,
                        "ask_contract_id": 707,
                        "gpu_name": "RTX A6000",
                        "gpu_ram_mb": 49140,
                        "compute_cap": 860,
                        "dph_total": 0.42,
                        # A driver below the Isaac ceiling: this test covers
                        # heartbeat-URL failure, not offer admission, and 595.x
                        # is now rejected before selection (attempt 069).
                        "driver_version": "580.159.03",
                        "machine_id": 7070,
                    }
                ]
            }
        if kwargs["method"] == "PUT" and kwargs["path"] == "/asks/707/":
            return 200, {"new_contract": 7071}
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/7071/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if kwargs["method"] == "PUT" and kwargs["path"] == "/instances/request_logs/7071":
            return 200, {"success": True, "result_url": "https://logs.example/downstream"}
        if kwargs["method"] == "DELETE":
            return 200, {"success": True}
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(
        vpa,
        "_fetch_text",
        lambda *_args, **_kwargs: (
            "BLUEPRINT_VAST_ONSTART_STARTED\n"
            "BLUEPRINT_VAST_HEARTBEAT_BLOCKED:6\n"
            "NVIDIA RTX A6000, 595.71.05, 49140 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED\n"
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED\n"
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE:0\n"
            "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED\n"
            "BLUEPRINT_VAST_ONSTART_DONE\n"
        ),
    )

    result = run_vast_provider_adapter(
        job_dir=tmp_path / "downstream-after-heartbeat",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle_kind="unitree_groot_n17_sonic",
        provider_bundle=bundle,
        provider_bundle_url="https://example.invalid/groot.zip",
        provider_output_put_url="https://example.invalid/out.zip",
        provider_runtime_output_zip=output_zip,
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        poll_interval_seconds=0,
        startup_timeout_seconds=20,
        session_max_live_minutes=None,
    )

    assert result["status"] == "completed"
    assert "vast_heartbeat_output_missing_success_marker" not in result["blockers"]
    startup = _read_json(
        tmp_path / "downstream-after-heartbeat" / "vast_startup_probe_manifest.json"
    )
    assert startup["status"] == "completed"
    assert startup["heartbeat_completed"] is False
    assert startup["startup_probe_proof_source"] == "downstream_provider_marker"
    assert startup["warnings"] == ["vast_heartbeat_url_failed_but_downstream_provider_marker_seen"]


def test_vast_adapter_records_machine_avoidlist_on_probe_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter.time.sleep", lambda _: None)

    def fake_api_json(
        *,
        method: str,
        path: str,
        api_key: str,
        payload=None,
        timeout_seconds: int = 30,
    ):  # type: ignore[no-untyped-def]
        assert api_key == secret
        if method == "POST" and path == "/bundles/":
            return 200, {
                "offers": [
                    {
                        "id": 506,
                        "ask_contract_id": 506,
                        "gpu_name": "RTX 3090",
                        "dph_total": 0.20,
                        "driver_version": "580.159.03",
                        "machine_id": 910,
                        "num_gpus": 1,
                        "rentable": True,
                    }
                ]
            }
        if method == "GET" and path == "/instances/":
            return 200, {"instances": []}
        if method == "PUT" and path == "/asks/506/":
            return 200, {"success": True, "new_contract": 991}
        if method == "GET" and path == "/instances/991/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if method == "PUT" and path == "/instances/request_logs/991":
            raise KeyboardInterrupt("simulated_request_log_interrupt")
        if method == "DELETE" and path == "/instances/991/":
            return 200, {"success": True}
        raise AssertionError((method, path))

    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=20,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "vast_probe_interrupted"
    assert result["continuing_spend_from_this_run"] is False
    avoidlist = _read_json(tmp_path / "vast_machine_avoidlist.json")
    assert avoidlist["machine_ids"] == [910]
    assert avoidlist["entries"][0]["instance_id"] == 991
    assert avoidlist["entries"][0]["blockers"] == ["vast_probe_interrupted_before_completion"]


def test_vast_adapter_private_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert vpa._number(True) is None
    assert vpa._number("1.25") == 1.25
    assert vpa._number("nope") is None
    assert vpa._version_tuple("") is None
    assert vpa._version_tuple("driver unknown") is None
    assert vpa._version_tuple("580.1") == (580, 1, 0)
    assert vpa._driver_sort_rank({"isaac_driver_support_status": "new"}) == 3
    assert vpa._driver_newer_branch_sort_rank({"driver_version": ""}) == 4
    assert vpa._driver_newer_branch_sort_rank({"driver_version": "570.158.1"}) == 1
    assert vpa._driver_newer_branch_sort_rank({"driver_version": "576.0.0"}) == 2
    assert vpa._driver_newer_branch_sort_rank({"driver_version": "565.0.0"}) == 3
    assert vpa._string_list("one") == ["one"]
    assert vpa._string_list(b"raw") == []

    redacted = vpa._redact_runtime_value(
        {"API_KEY": "secret", "items": ("secret", "public", "visible")},
        ["secret"],
    )
    assert redacted == {
        "API_KEY": vpa.REDACTED_SECRET_FIELD,
        "items": [vpa.REDACTED_SECRET, "public", "visible"],
    }
    assert vpa._redact_runtime_value(("API_KEY", "secret"), []) == [
        "API_KEY",
        vpa.REDACTED_SECRET_FIELD,
    ]
    assert vpa._redact_runtime_value(["secret", "public"], ["secret"]) == [
        vpa.REDACTED_SECRET,
        "public",
    ]
    assert vpa._redact_runtime_value(
        {"extra_env": [["HF_TOKEN", "hf_raw_value"], ["VISIBLE", "ok"]]},
        [],
    ) == {
        "extra_env": [["HF_TOKEN", vpa.REDACTED_SECRET_FIELD], ["VISIBLE", "ok"]],
    }
    assert vpa._redact_runtime_value(
        {"last_instance_payload": {"jupyter_token": "raw-vast-token", "label": "ok"}},
        [],
    ) == {
        "last_instance_payload": {
            "jupyter_token": vpa.REDACTED_SECRET_FIELD,
            "label": "ok",
        }
    }

    assert vpa._offers_from_response({"offers": {"id": 11}}) == [{"id": 11}]
    assert vpa._offers_from_response({"offers": {"a": {"id": 12}, "bad": []}}) == [{"id": 12}]
    assert vpa._offers_from_response({"response": [{"id": 13}, "bad"]}) == [{"id": 13}]
    assert vpa._offers_from_response({}) == []
    assert vpa._offer_id({"ask_contract_id": 0, "id": "42"}) == 42
    assert vpa._offer_id({"id": "bad"}) is None
    assert vpa._offer_hourly_rate({"pricing": {"machine": {"discountTotalHour": "0.33"}}}) == 0.33
    assert vpa._offer_hourly_rate({}) is None

    avoidlist = tmp_path / "avoidlist.json"
    avoidlist.write_text("{bad json", encoding="utf-8")
    assert vpa._load_machine_avoidlist(avoidlist)["status"] == "blocked_parse_failed"
    avoidlist.write_text("[]", encoding="utf-8")
    assert vpa._load_machine_avoidlist(avoidlist)["status"] == "blocked_invalid_shape"
    avoidlist.write_text(
        json.dumps({"machine_ids": ["5"], "entries": [{"machine_id": "6"}]}),
        encoding="utf-8",
    )
    assert vpa._avoidlist_machine_ids(avoidlist) == {5, 6}
    assert (
        vpa._select_offer(
            [
                {"id": None, "gpu_name": "RTX 4090", "dph_total": 0.1},
                {"id": 2, "gpu_name": "RTX 4090", "dph_total": 2.0},
                {"id": 3, "gpu_name": "H100", "dph_total": 0.1},
            ],
            max_hourly_rate=0.5,
        )
        is None
    )

    env_string = vpa._make_env_string({"A": "one two", "": "skip", "B": "ok"})
    assert "-e" in env_string
    assert "'A=one two'" in env_string
    with pytest.raises(ValueError, match="unsupported_vast_launch_mode"):
        vpa._resolve_launch_mode(requested="bad", enable_isaac_smoke=False)
    with pytest.raises(ValueError, match="unsupported_provider_bundle_kind"):
        vpa._resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=False,
            provider_bundle_kind="bad",
        )
    assert vpa._resolve_launch_mode(requested="auto", enable_isaac_smoke=False) == "ssh_direct"
    assert (
        vpa._resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="wam",
        )
        == "ssh_direct"
    )
    assert (
        vpa._resolve_launch_mode(requested="jupyter_direct", enable_isaac_smoke=True)
        == "jupyter_direct"
    )
    assert vpa._resolve_disk_gb(requested=42, enable_isaac_smoke=True) == 42

    with pytest.raises(ValueError, match="unsupported_ngc_image_login_mode"):
        vpa._resolve_image_login(image="nvcr.io/private/image:1", ngc_key="", mode="bad")
    assert (
        vpa._resolve_image_login(
            image="ubuntu:22.04",
            ngc_key="secret-ngc",
            mode="always",
        )[1]["reason"]
        == "non_ngc_image"
    )
    login, summary = vpa._resolve_image_login(
        image="docker.io/nijelhunt/blueprint-unitree-unifolm:20260622-cu124-sdpa3",
        ngc_key="",
        docker_username="nijelhunt",
        docker_pat="secret-docker-pat",
        mode="auto",
    )
    assert login == "-u nijelhunt -p secret-docker-pat docker.io"
    assert summary["reason"] == "docker_hub_image_login_supplied"
    assert summary["image_login_supplied"] is True
    assert summary["docker_secret_file_present"] is True
    public_login, public_summary = vpa._resolve_image_login(
        image="docker.io/nijelhunt/blueprint-unitree-unifolm:20260622-cu124-sdpa3",
        ngc_key="",
        docker_username="nijelhunt",
        docker_pat="secret-docker-pat",
        mode="never",
    )
    assert public_login is None
    assert public_summary["reason"] == "docker_hub_image_login_disabled"
    assert public_summary["image_login_supplied"] is False
    missing_login, missing_summary = vpa._resolve_image_login(
        image="docker.io/nijelhunt/blueprint-unitree-unifolm:20260622-cu124-sdpa3",
        ngc_key="",
        docker_username="nijelhunt",
        docker_pat="",
        mode="auto",
    )
    assert missing_login is None
    assert missing_summary["reason"] == "docker_pat_missing"
    assert (
        vpa._resolve_image_login(
            image="nvidia/cuda:12.4.1-runtime-ubuntu22.04",
            ngc_key="",
            docker_username="nijelhunt",
            docker_pat="secret-docker-pat",
            mode="auto",
        )[1]["reason"]
        == "non_ngc_image"
    )
    assert (
        vpa._resolve_image_login(
            image="nvcr.io/private/image:1",
            ngc_key="",
            mode="always",
        )[1]["reason"]
        == "ngc_key_missing"
    )
    assert (
        vpa._resolve_image_login(
            image="nvcr.io/private/image:1",
            ngc_key="secret-ngc",
            mode="never",
        )[1]["reason"]
        == "ngc_image_login_disabled"
    )
    login, summary = vpa._resolve_image_login(
        image="nvcr.io/private/image:1",
        ngc_key="secret-ngc",
        mode="always",
    )
    assert login is not None
    assert login == "-u $oauthtoken -p secret-ngc nvcr.io"
    assert "'$oauthtoken'" not in login
    assert summary["image_login_supplied"] is True

    payload = vpa._create_payload(
        image="image",
        label="label",
        launch_mode="jupyter_direct",
        probe_script="echo hi",
        disk_gb=12,
        env={"A": "B"},
        image_login="login",
    )
    assert payload["use_jupyter_lab"] is True
    assert payload["jupyter_dir"] == "/workspace"
    assert payload["image_login"] == "login"
    summary = vpa._create_request_summary(
        payload,
        secret_values=["login", "secret-docker-pat", "secret-ngc"],
    )
    assert summary["image_login_supplied"] is True
    assert summary["raw_payload_redacted"]["image_login"] == vpa.REDACTED_SECRET_FIELD
    template_payload = vpa._create_payload(
        image=None,
        label="label",
        launch_mode="args",
        probe_script="echo hi",
        disk_gb=12,
        template_hash_id="template-hash",
    )
    assert "image" not in template_payload
    assert template_payload["template_hash_id"] == "template-hash"
    assert vpa._instance_id_from_create_response({"instance": {"id": "99"}}) == 99
    assert vpa._instance_id_from_create_response({}) is None
    assert vpa._instance_status({"status": "queued"}) == "queued"
    assert vpa._instance_status({"instances": {"cur_state": "loading"}}) == "loading"
    assert vpa._instance_list_rows({"instances": {"a": {"id": 1}, "bad": []}}) == [{"id": 1}]
    assert vpa._instance_list_rows({"instances": {"id": 2, "status": "running"}}) == [
        {"id": 2, "status": "running"}
    ]
    assert vpa._instance_list_rows({"results": [{"id": 3}, "bad"]}) == [{"id": 3}]
    assert vpa._instance_list_rows({"data": {"a": {"id": 4}, "bad": []}}) == [{"id": 4}]
    assert vpa._instance_list_rows({"actual_status": "running", "id": 5}) == [
        {"actual_status": "running", "id": 5}
    ]
    assert vpa._instance_list_rows({"unrelated": True}) == []

    secret_dir = tmp_path / "secret-dir"
    secret_dir.mkdir()
    old_env = vpa.os.environ.get(VAST_API_KEY_FILE_ENV)
    try:
        vpa.os.environ[VAST_API_KEY_FILE_ENV] = str(secret_dir)
        key, status = vpa._read_secret_file(VAST_API_KEY_FILE_ENV, str(secret_dir))
    finally:
        if old_env is None:
            vpa.os.environ.pop(VAST_API_KEY_FILE_ENV, None)
        else:
            vpa.os.environ[VAST_API_KEY_FILE_ENV] = old_env
    assert key == ""
    assert status["read_error"] == "IsADirectoryError"

    secret_file = tmp_path / "unreadable-secret"
    secret_file.write_text("secret", encoding="utf-8")
    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == secret_file:
            raise OSError("cannot read")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    old_env = vpa.os.environ.get(VAST_API_KEY_FILE_ENV)
    try:
        vpa.os.environ[VAST_API_KEY_FILE_ENV] = str(secret_file)
        key, status = vpa._read_secret_file(VAST_API_KEY_FILE_ENV, str(secret_file))
    finally:
        if old_env is None:
            vpa.os.environ.pop(VAST_API_KEY_FILE_ENV, None)
        else:
            vpa.os.environ[VAST_API_KEY_FILE_ENV] = old_env
    assert key == ""
    assert status["read_error"] == "OSError"

    assert vpa._attempt_runtime_seconds({"runtime_seconds_observed_by_adapter": -5}) == 0.0
    assert (
        vpa._attempt_runtime_seconds({"estimated_cost_usd": 0.5, "selected_hourly_rate_usd": 1.0})
        == 1800.0
    )
    assert vpa._attempt_runtime_seconds({}) == 0.0
    assert vpa._attempt_estimated_cost({"estimated_cost_usd": -1}) == 0.0
    assert vpa._attempt_estimated_cost({}) == 0.0

    guard_dir = tmp_path / "guard"
    guard_dir.mkdir()
    budget = guard_dir / "budget.json"
    budget.write_text(
        json.dumps(
            {
                "attempts": [
                    {
                        "actual_live_runtime_seconds_observed_by_adapter": 120,
                        "estimated_cost_usd": 0.3,
                    },
                    {
                        "estimated_cost_usd_using_observed_rate": 0.4,
                        "observed_hourly_rate_usd": 1.2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    guard = vpa._session_budget_guard(
        job_dir=guard_dir,
        generated_at="2026-06-20T00:00:00Z",
        budget_path=budget,
        session_max_live_minutes=2,
        requested_max_live_minutes=3,
        target_spend_usd=0.5,
        hard_cap_usd=1.0,
        max_hourly_rate=0.5,
    )
    assert guard["status"] == "blocked"
    assert "session_live_runtime_limit_exhausted" in guard["blockers"]
    assert "session_estimated_spend_target_already_exceeded" in guard["warnings"]
    budget.write_text("{bad json", encoding="utf-8")
    parse_guard = vpa._session_budget_guard(
        job_dir=guard_dir,
        generated_at="2026-06-20T00:00:00Z",
        budget_path=budget,
        session_max_live_minutes=None,
        requested_max_live_minutes=1,
        target_spend_usd=10.0,
        hard_cap_usd=10.0,
        max_hourly_rate=0.1,
    )
    assert "session_budget_ledger_parse_failed" in parse_guard["blockers"]
    budget.write_text(
        json.dumps({"attempts": [{"estimated_cost_usd": 0.2}]}),
        encoding="utf-8",
    )
    projected_spend_guard = vpa._session_budget_guard(
        job_dir=guard_dir,
        generated_at="2026-06-20T00:00:00Z",
        budget_path=budget,
        session_max_live_minutes=None,
        requested_max_live_minutes=60,
        target_spend_usd=10.0,
        hard_cap_usd=0.3,
        max_hourly_rate=0.2,
    )
    assert "requested_max_spend_would_exceed_hard_cap" in projected_spend_guard["blockers"]


def test_vast_adapter_public_dns_socket_and_url_probe_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert vpa._provider_url_public_blocker("ftp://example.com/file", "bundle") == (
        "bundle_url_scheme_not_http"
    )
    assert vpa._provider_url_public_blocker("https:///missing-host", "bundle") == (
        "bundle_url_host_missing"
    )
    assert vpa._provider_url_public_blocker("https://10.0.0.1/file", "bundle") == (
        "bundle_url_not_publicly_reachable"
    )
    assert vpa._provider_url_public_blocker("https://8.8.8.8/file", "bundle") is None
    assert vpa._provider_url_public_blocker("https://example.com/file", "bundle") is None

    monkeypatch.setattr(vpa.shutil, "which", lambda name: None if name == "dig" else name)
    assert vpa._resolve_public_dns_a_records("example.com") == []

    monkeypatch.setattr(vpa.shutil, "which", lambda name: "/usr/bin/dig")
    monkeypatch.setattr(
        vpa.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("dig failed")),
    )
    assert vpa._resolve_public_dns_a_records("example.com") == []

    monkeypatch.setattr(
        vpa.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="not-an-ip\n::1\n127.0.0.1\n10.0.0.2\n8.8.8.8\n8.8.8.8\n",
            stderr="",
        ),
    )
    assert vpa._resolve_public_dns_a_records("example.com") == ["8.8.8.8"]

    class HeaderSocket:
        def __init__(self, chunks: list[bytes]) -> None:
            self.chunks = chunks
            self.sent = b""
            self.closed = False

        def recv(self, _size: int) -> bytes:
            return self.chunks.pop(0) if self.chunks else b""

        def settimeout(self, _timeout: int) -> None:
            return None

        def sendall(self, data: bytes) -> None:
            self.sent += data

        def close(self) -> None:
            self.closed = True

    status_code, headers = vpa._read_http_headers_from_socket(
        HeaderSocket(
            [
                b"HTTP/1.1 204 No Content\r\nContent-Length: 12\r\n",
                b"Content-Type: application/zip\r\nIgnored\r\n\r\nbody",
            ]
        )
    )
    assert status_code == 204
    assert headers["Content-Length"] == "12"
    assert vpa._read_http_headers_from_socket(HeaderSocket([b""])) == (None, {})
    bad_status, _bad_headers = vpa._read_http_headers_from_socket(
        HeaderSocket([b"HTTP/1.1 OK\r\nX-Test: yes\r\n\r\n"])
    )
    assert bad_status is None

    assert vpa._head_with_public_dns_fallback("ftp://example.com/file")["status"] == "blocked"
    time_values = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(vpa.time, "time", lambda: next(time_values))
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(vpa, "_resolve_public_dns_a_records", lambda *_, **__: [])
    no_dns = vpa._head_with_public_dns_fallback(
        "https://example.com/runtime.zip", timeout_seconds=0
    )
    assert no_dns["blockers"] == ["provider_bundle_fetch_url_public_dns_fallback_failed"]

    monkeypatch.setattr(vpa, "_resolve_public_dns_a_records", lambda *_, **__: ["8.8.8.8"])
    monkeypatch.setattr(vpa.time, "time", lambda: 10.0)

    raw_sock = HeaderSocket([])
    wrapped_sock = HeaderSocket(
        [b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nContent-Type: text/plain\r\n\r\nbody"]
    )

    class FakeContext:
        def wrap_socket(self, sock: HeaderSocket, *, server_hostname: str) -> HeaderSocket:
            assert sock is raw_sock
            assert server_hostname == "example.com"
            return wrapped_sock

    monkeypatch.setattr(vpa.socket, "create_connection", lambda *_, **__: raw_sock)
    monkeypatch.setattr(vpa.ssl, "create_default_context", lambda: FakeContext())
    head = vpa._head_with_public_dns_fallback(
        "https://example.com/runtime.zip?token=secret",
        timeout_seconds=1,
    )
    assert head["status"] == "passed"
    assert head["http_status_code"] == 200
    assert head["content_length"] == 4
    assert wrapped_sock.closed is True

    http_sock = HeaderSocket([b"HTTP/1.1 201 Created\r\nContent-Length: 0\r\n\r\n"])
    monkeypatch.setattr(vpa.socket, "create_connection", lambda *_, **__: http_sock)
    http_head = vpa._head_with_public_dns_fallback(
        "http://example.com/runtime.zip", timeout_seconds=1
    )
    assert http_head["http_status_code"] == 201
    assert http_sock.closed is True

    raw_error_sock = HeaderSocket([])

    class FailingContext:
        def wrap_socket(self, sock: HeaderSocket, *, server_hostname: str) -> HeaderSocket:
            raise OSError("tls failed")

    monkeypatch.setattr(vpa.socket, "create_connection", lambda *_, **__: raw_error_sock)
    monkeypatch.setattr(vpa.ssl, "create_default_context", lambda: FailingContext())
    failed = vpa._head_with_public_dns_fallback(
        "https://example.com/runtime.zip", timeout_seconds=1
    )
    assert failed["status"] == "blocked"
    assert failed["connection_errors"][0]["error_type"] == "OSError"
    assert raw_error_sock.closed is True


def test_vast_adapter_template_discovery_and_preservation_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vpa, "_api_json", lambda **_: (200, {"templates": {"bad": True}}))
    missing_list = vpa._discover_vast_templates(
        job_dir=tmp_path / "templates-missing-list",
        generated_at="2026-06-20T00:00:00Z",
        api_key="secret",
    )
    assert missing_list["blockers"] == ["vast_template_response_missing_templates_list"]

    def raise_http_error(**_kwargs):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError("https://vast.invalid", 500, "bad", {}, BytesIO(b"bad"))

    monkeypatch.setattr(vpa, "_api_json", raise_http_error)
    http_error = vpa._discover_vast_templates(
        job_dir=tmp_path / "templates-http-error",
        generated_at="2026-06-20T00:00:00Z",
        api_key="secret",
    )
    assert http_error["blockers"] == ["vast_template_search_http_error:500"]

    monkeypatch.setattr(
        vpa,
        "_api_json",
        lambda **_: (_ for _ in ()).throw(RuntimeError("template api failed")),
    )
    generic_error = vpa._discover_vast_templates(
        job_dir=tmp_path / "templates-generic-error",
        generated_at="2026-06-20T00:00:00Z",
        api_key="secret",
    )
    assert generic_error["blockers"] == ["vast_template_search_failed:RuntimeError"]

    job_dir = tmp_path / "preserve"
    job_dir.mkdir()
    artifact = job_dir / "vast_startup_probe_manifest.json"
    artifact.write_text('{"status":"old"}', encoding="utf-8")
    slug = vpa._attempt_preservation_slug("2026-06-20T00:00:00Z")
    (job_dir / f"attempt_preserved_{slug}").mkdir()

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("copy failed")

    monkeypatch.setattr(vpa.shutil, "copy2", fail_copy)
    preserved = vpa._preserve_existing_live_attempt_artifacts(
        job_dir=job_dir,
        generated_at="2026-06-20T00:00:00Z",
        reason="unit-test",
        artifact_names=["vast_startup_probe_manifest.json"],
    )
    assert preserved is not None
    assert preserved["status"] == "blocked_copy_errors"
    assert preserved["preserve_dir"].endswith("_2")
    assert preserved["copy_errors"][0]["error_type"] == "OSError"

    budget = tmp_path / "bad-session-budget.json"
    budget.write_text("{bad json", encoding="utf-8")
    summary = vpa._append_session_budget_attempt(
        budget_path=budget,
        job_dir=tmp_path / "job",
        generated_at="2026-06-20T00:00:00Z",
        ledger={
            "vast_instance_ids": [1],
            "selected_hourly_rate_usd": 0.2,
            "actual_live_runtime_seconds_observed_by_adapter": 30,
            "estimated_cost_usd": 0.002,
            "continuing_spend_from_this_run": False,
        },
        selected_offer={"gpu_name": "RTX 4090", "machine_id": 7, "ask_contract_id": 9},
        result_status="blocked",
        result_reason="unit-test",
        blockers=["blocked"],
    )
    assert summary["status"] == "completed_after_parse_reset"
    assert summary["parse_error_recovered"].startswith("JSONDecodeError:")


def test_vast_adapter_blueprint_preflight_branch_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="unsupported_provider_bundle_kind"):
        vpa._blueprint_bundle_preflight(
            job_dir=tmp_path / "bad-kind",
            generated_at="2026-06-20T00:00:00Z",
            enable_blueprint_bundle=True,
            enable_isaac_smoke=False,
            provider_bundle_kind="bad",
            bundle_path=None,
            provider_bundle_url=None,
            provider_output_put_url=None,
        )

    wam_bundle = tmp_path / "wam-incomplete.zip"
    with zipfile.ZipFile(wam_bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo no fallback")
        archive.writestr("provider_runtime/wam_provider_runtime_runner.py", "print('no contract')")
        archive.writestr("provider_runtime/wam_provider_runtime_manifest.json", "{}")
        archive.writestr("provider_runtime/wam_rollout_input_manifest.json", "{}")
        archive.writestr("provider_runtime/oscar_input/first_frame.png", b"png")
        archive.writestr(
            "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4",
            b"mp4",
        )
    wam_preflight = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "wam-preflight",
        generated_at="2026-06-20T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="wam",
        bundle_path=wam_bundle,
        provider_bundle_url="https://example.invalid/wam.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert "provider_entrypoint_missing_runtime_result_crash_fallback" in wam_preflight["blockers"]
    assert "provider_runner_missing_wam_runtime_contract" in wam_preflight["blockers"]

    missing_bundle = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "missing-bundle",
        generated_at="2026-06-20T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="wam",
        bundle_path=tmp_path / "missing.zip",
        provider_bundle_url="https://example.invalid/wam.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert "provider_runtime_bundle_missing" in missing_bundle["blockers"]

    not_zip = tmp_path / "not-a-bundle.zip"
    not_zip.write_text("not a zip", encoding="utf-8")
    zip_failed = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "zip-failed",
        generated_at="2026-06-20T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="wam",
        bundle_path=not_zip,
        provider_bundle_url="https://example.invalid/wam.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert "provider_runtime_bundle_zip_inspection_failed:BadZipFile" in zip_failed["blockers"]

    incomplete_zip = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(incomplete_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo hi")
    incomplete = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "incomplete",
        generated_at="2026-06-20T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="wam",
        bundle_path=incomplete_zip,
        provider_bundle_url="https://example.invalid/wam.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert "provider_runtime_bundle_required_entries_missing" in incomplete["blockers"]

    powered_zip = tmp_path / "powered-wam.zip"
    powered_rows = [
        {"initial_observation_relative_path": f"images/session-{index:02d}/window.png"}
        for index in range(51)
    ]
    with zipfile.ZipFile(powered_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "provider_runtime/run_wam_provider_runtime.sh",
            "write_missing_result\nwam_runner_process_exited_without_runtime_result\n"
            "blocked_wam_process_exited_without_result\n",
        )
        archive.writestr(
            "provider_runtime/wam_provider_runtime_runner.py",
            "wam_runtime_result.json\nCosmos3-Nano\naction_conditioned_video_rollout_generated\n",
        )
        archive.writestr("provider_runtime/wam_provider_runtime_manifest.json", "{}")
        archive.writestr("provider_runtime/wam_rollout_input_manifest.json", "{}")
        archive.writestr(
            "provider_runtime/cosmos3_powered_droid/packet.json",
            json.dumps(
                {
                    "schema_version": "policy_ranking_powered_droid_provider_packet.v1",
                    "rows": powered_rows,
                }
            ),
        )
        for row in powered_rows:
            archive.writestr(
                "provider_runtime/cosmos3_powered_droid/"
                + row["initial_observation_relative_path"],
                b"png",
            )
        for name in ("canary_manifest.json", "initial_observation.png", "action_streams.json"):
            archive.writestr(
                "provider_runtime/cosmos3_powered_droid/official_canary/" + name,
                b"{}" if name.endswith(".json") else b"png",
            )
    powered = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "powered",
        generated_at="2026-07-29T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="wam",
        bundle_path=powered_zip,
        provider_bundle_url="https://example.invalid/powered.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert "provider_runtime_bundle_required_entries_missing" not in powered["blockers"]

    integrity_zip = tmp_path / "integrity.zip"
    integrity_zip.write_text("placeholder", encoding="utf-8")

    class FakeZip:
        def __init__(self, _path: Path) -> None:
            self.entries = {
                "provider_runtime/run_wam_provider_runtime.sh": (
                    "write_missing_result\n"
                    "wam_runner_process_exited_without_runtime_result\n"
                    "blocked_wam_process_exited_without_result\n"
                ),
                "provider_runtime/wam_provider_runtime_runner.py": (
                    "wam_runtime_result.json\nOSCAR-2B\n"
                    "action_conditioned_video_rollout_generated\n"
                ),
                "provider_runtime/wam_provider_runtime_manifest.json": "{}",
                "provider_runtime/wam_rollout_input_manifest.json": "{}",
                "provider_runtime/oscar_input/first_frame.png": "png",
                "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4": "mp4",
            }

        def __enter__(self) -> "FakeZip":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def namelist(self) -> list[str]:
            return list(self.entries)

        def testzip(self) -> str:
            return "provider_runtime/wam_provider_runtime_runner.py"

        def read(self, member: str) -> bytes:
            return self.entries[member].encode("utf-8")

    monkeypatch.setattr(vpa.zipfile, "ZipFile", FakeZip)
    integrity_failed = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "integrity-failed",
        generated_at="2026-06-20T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="wam",
        bundle_path=integrity_zip,
        provider_bundle_url="https://example.invalid/wam.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert "provider_runtime_bundle_zip_integrity_failed" in integrity_failed["blockers"]
    monkeypatch.undo()

    isaac_bad_eval = tmp_path / "isaac-bad-eval.zip"
    with zipfile.ZipFile(isaac_bad_eval, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "provider_runtime/run_isaac_realistic_runtime.sh",
            "write_missing_result\nisaac_runner_process_exited_without_runtime_result\nblocked_isaac_process_exited_without_result\n",
        )
        archive.writestr(
            "provider_runtime/isaac_realistic_runtime_runner.py",
            "from isaacsim import SimulationApp\n",
        )
        archive.writestr("provider_runtime/isaac_provider_eval_manifest.json", "{bad json")
        archive.writestr("provider_runtime/scenario_eval_matrix.json", "{bad matrix")
        for name in (
            "provider_runtime/generated_site_scene.usda",
            "provider_runtime/generated_site_scene.usd",
            "provider_runtime/camera_manifest.json",
            "provider_runtime/episode_spec_manifest.json",
        ):
            archive.writestr(name, "{}")
    readiness_path = tmp_path / "isaac_provider_bundle_readiness.json"
    readiness_path.write_text("{bad json", encoding="utf-8")
    isaac_preflight = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "isaac-preflight",
        generated_at="2026-06-20T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="isaac",
        bundle_path=isaac_bad_eval,
        provider_bundle_url="https://example.invalid/isaac.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert "provider_eval_manifest_parse_failed" in isaac_preflight["blockers"]
    assert "provider_runtime_bundle_json_parse_failed" in isaac_preflight["blockers"]
    assert (
        "provider_runtime/scenario_eval_matrix.json:JSONDecodeError"
        in isaac_preflight["json_member_parse_errors"]
    )
    assert "provider_bundle_readiness_parse_failed" in isaac_preflight["blockers"]

    _write_valid_provider_bundle(tmp_path / "valid-isaac.zip")
    valid_bundle = tmp_path / "valid-isaac.zip"
    (tmp_path / "isaac_provider_bundle_readiness.json").write_text(
        json.dumps(
            {
                "local_bundle_ready_for_remote_staging": False,
                "blockers": ["missing-camera", "provider_launch_request_blocked:ignored"],
            }
        ),
        encoding="utf-8",
    )
    readiness_failed = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "readiness-failed",
        generated_at="2026-06-20T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="isaac",
        bundle_path=valid_bundle,
        provider_bundle_url="https://example.invalid/isaac.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert "provider_bundle_readiness_local_failed" in readiness_failed["blockers"]
    assert "provider_bundle_readiness:missing-camera" in readiness_failed["blockers"]

    class FakeResponse:
        def __init__(
            self,
            *,
            status: int,
            headers: dict[str, str] | None = None,
            body: bytes = b"",
        ) -> None:
            self.status = status
            self.headers = headers or {}
            self._body = body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            return self._body

    def run_preflight_with_urlopen(fake_urlopen, *, allow_put: bool = True) -> dict[str, object]:
        monkeypatch.setattr(vpa.urllib.request, "urlopen", fake_urlopen)
        return vpa._blueprint_bundle_preflight(
            job_dir=tmp_path / f"url-preflight-{len(list(tmp_path.glob('url-preflight-*')))}",
            generated_at="2026-06-20T00:00:00Z",
            enable_blueprint_bundle=True,
            enable_isaac_smoke=True,
            provider_bundle_kind="isaac",
            bundle_path=valid_bundle,
            provider_bundle_url="https://storage.example/isaac.zip",
            provider_output_put_url="https://storage.example/out.zip",
            verify_staging_urls=True,
            allow_staging_output_put_probe=allow_put,
        )

    responses = iter(
        [
            FakeResponse(
                status=200,
                headers={"Content-Length": str(valid_bundle.stat().st_size + 1)},
            ),
        ]
    )
    head_size_mismatch = run_preflight_with_urlopen(
        lambda *_a, **_k: next(responses), allow_put=False
    )
    assert "provider_bundle_fetch_url_size_mismatch" in head_size_mismatch["blockers"]
    assert head_size_mismatch["output_put_probe"]["status"] == "skipped"

    head_500 = run_preflight_with_urlopen(
        lambda *_a, **_k: FakeResponse(status=500, headers={"Content-Length": "0"}),
        allow_put=False,
    )
    assert "provider_bundle_fetch_url_unreachable" in head_500["blockers"]

    def http_error_urlopen(request, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        method = request.get_method()
        code = 404 if method == "HEAD" else 403
        raise urllib.error.HTTPError(request.full_url, code, "blocked", {}, BytesIO(b"blocked"))

    http_errors = run_preflight_with_urlopen(http_error_urlopen)
    assert http_errors["bundle_url_probe"]["http_status_code"] == 404
    assert http_errors["output_put_probe"]["http_status_code"] == 403

    def head_403_get_range_urlopen(request, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        method = request.get_method()
        if method == "HEAD":
            raise urllib.error.HTTPError(
                request.full_url,
                403,
                "head denied",
                {},
                BytesIO(b"denied"),
            )
        assert method == "GET"
        range_header = (
            request.get_header("Range")
            or request.headers.get("Range")
            or request.unredirected_hdrs.get("Range")
        )
        assert range_header == "bytes=0-0"
        return FakeResponse(
            status=206,
            headers={
                "Content-Length": "1",
                "Content-Range": f"bytes 0-0/{valid_bundle.stat().st_size}",
            },
            body=b"P",
        )

    head_403_get_range = run_preflight_with_urlopen(
        head_403_get_range_urlopen,
        allow_put=False,
    )
    assert head_403_get_range["bundle_url_probe"]["status"] == "passed"
    assert head_403_get_range["bundle_url_probe"]["method"] == "GET"
    assert head_403_get_range["bundle_url_probe"]["head_http_status_code"] == 403
    assert "provider_bundle_fetch_url_unreachable" not in head_403_get_range["blockers"]

    monkeypatch.setattr(
        vpa,
        "_head_with_public_dns_fallback",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "content_length": valid_bundle.stat().st_size + 2,
        },
    )

    def url_error_urlopen(request, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise urllib.error.URLError(TimeoutError("timed out"))

    url_errors = run_preflight_with_urlopen(url_error_urlopen)
    assert url_errors["bundle_url_probe"]["normal_head_reason_type"] == "TimeoutError"
    assert "provider_bundle_fetch_url_size_mismatch" in url_errors["blockers"]
    assert url_errors["output_put_probe"]["reason_type"] == "TimeoutError"

    monkeypatch.setattr(
        vpa,
        "_head_with_public_dns_fallback",
        lambda *_args, **_kwargs: {"status": "blocked", "blockers": ["dns failed"]},
    )
    fallback_blocked = run_preflight_with_urlopen(url_error_urlopen, allow_put=False)
    assert "provider_bundle_fetch_url_unreachable" in fallback_blocked["blockers"]

    def generic_error_urlopen(_request, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("network shim failed")

    generic_errors = run_preflight_with_urlopen(generic_error_urlopen)
    assert generic_errors["bundle_url_probe"]["error_type"] == "RuntimeError"
    assert generic_errors["output_put_probe"]["error_type"] == "RuntimeError"

    put_responses = iter(
        [
            FakeResponse(
                status=200,
                headers={"Content-Length": str(valid_bundle.stat().st_size)},
            ),
            FakeResponse(status=500, body=b"denied"),
        ]
    )
    put_blocked = run_preflight_with_urlopen(lambda *_a, **_k: next(put_responses))
    assert put_blocked["output_put_probe"]["http_status_code"] == 500
    assert "provider_output_put_url_unwritable" in put_blocked["blockers"]

    blocked_url_probe = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "blocked-staging-urls",
        generated_at="2026-06-20T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="isaac",
        bundle_path=valid_bundle,
        provider_bundle_url="http://localhost/bundle.zip",
        provider_output_put_url="ftp://storage.example/out.zip",
        verify_staging_urls=True,
        allow_staging_output_put_probe=True,
    )
    assert blocked_url_probe["bundle_url_probe"]["blockers"] == [
        "provider_bundle_fetch_url_not_publicly_reachable"
    ]
    assert blocked_url_probe["output_put_probe"]["blockers"] == [
        "provider_output_put_url_scheme_not_http"
    ]


def test_vast_adapter_small_provider_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        vpa._isaac_image_startup_preflight(
            job_dir=tmp_path / "image-not-required",
            generated_at="2026-06-20T00:00:00Z",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="wam",
            selected_container_image=DEFAULT_ISAAC_IMAGE,
            vast_template_hash_id=None,
            use_vast_template_image=False,
            max_live_minutes=5,
            allow_cold_isaac_image_pull=True,
            min_cold_isaac_pull_live_minutes=10,
        )["status"]
        == "not_required"
    )
    with pytest.raises(ValueError, match="unsupported_provider_bundle_kind"):
        vpa._isaac_image_startup_preflight(
            job_dir=tmp_path / "bad-image-kind",
            generated_at="2026-06-20T00:00:00Z",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=False,
            provider_bundle_kind="bad",
            selected_container_image=DEFAULT_ISAAC_IMAGE,
            vast_template_hash_id=None,
            use_vast_template_image=False,
            max_live_minutes=5,
            allow_cold_isaac_image_pull=True,
            min_cold_isaac_pull_live_minutes=10,
        )
    template_missing = vpa._isaac_image_startup_preflight(
        job_dir=tmp_path / "template-missing",
        generated_at="2026-06-20T00:00:00Z",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=False,
        provider_bundle_kind="isaac",
        selected_container_image=DEFAULT_ISAAC_IMAGE,
        vast_template_hash_id=None,
        use_vast_template_image=True,
        max_live_minutes=5,
        allow_cold_isaac_image_pull=True,
        min_cold_isaac_pull_live_minutes=10,
    )
    assert "vast_template_hash_required_when_using_template_image" in template_missing["blockers"]
    cold_short = vpa._isaac_image_startup_preflight(
        job_dir=tmp_path / "cold-short",
        generated_at="2026-06-20T00:00:00Z",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=False,
        provider_bundle_kind="isaac",
        selected_container_image=DEFAULT_ISAAC_IMAGE,
        vast_template_hash_id=None,
        use_vast_template_image=False,
        max_live_minutes=5,
        allow_cold_isaac_image_pull=True,
        min_cold_isaac_pull_live_minutes=10,
    )
    assert "cold_official_isaac_image_pull_live_window_too_short" in cold_short["blockers"]
    assert "-e" not in vpa._make_env_string({"": "skip", "A": None})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported_provider_bundle_kind"):
        vpa._resolve_probe_image(
            public_image="public",
            isaac_image="isaac",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=False,
            provider_bundle_kind="bad",
        )
    assert (
        vpa._resolve_probe_image(
            public_image="public",
            isaac_image="isaac",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="wam",
        )
        == "public"
    )
    for env_name in (
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON_INSTALL_REQUIREMENTS",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_REPO_URL",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_REPO_REF",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_UV_SYNC_TIMEOUT_SECONDS",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SERVER_STARTUP_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(env_name, raising=False)
    assert vpa._probe_env(job_dir=tmp_path / "env", enable_isaac_smoke=False) == {
        "BLUEPRINT_VAST_PROBE": "true",
        "BLUEPRINT_VAST_PROBE_JOB_DIR_BASENAME": "env",
    }
    monkeypatch.setenv(vpa.VAST_FORWARD_SECRET_ENV_VARS_ENV, "SAFE_NAME,MY_API_KEY")
    monkeypatch.setenv("SAFE_NAME", "not-forwarded")
    monkeypatch.setenv("MY_API_KEY", "forwarded-secret")
    forwarded_env = vpa._probe_env(job_dir=tmp_path / "forwarded", enable_isaac_smoke=False)
    assert forwarded_env["MY_API_KEY"] == "forwarded-secret"
    assert "SAFE_NAME" not in forwarded_env
    assert vpa._forwarded_secret_values() == ["forwarded-secret"]
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY", "disabled")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_SKIP_RUNTIME_PIP_INSTALL", "true")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_OMIT_FPS_ARG", "true")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_NUM_STEPS", "35")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "temporal_rgb35")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_NUM_FRAMES", "49")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_HEIGHT", "480")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_WIDTH", "640")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_FPS", "12")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS", "1200")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER", "false")
    runtime_env = vpa._probe_env(job_dir=tmp_path / "runtime", enable_isaac_smoke=False)
    assert runtime_env["BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY"] == "disabled"
    assert runtime_env["BLUEPRINT_OSCAR_WAM_SKIP_RUNTIME_PIP_INSTALL"] == "true"
    assert runtime_env["BLUEPRINT_OSCAR_WAM_OMIT_FPS_ARG"] == "true"
    assert runtime_env["BLUEPRINT_OSCAR_WAM_NUM_STEPS"] == "35"
    assert runtime_env["BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE"] == "temporal_rgb35"
    assert runtime_env["BLUEPRINT_OSCAR_WAM_NUM_FRAMES"] == "49"
    assert runtime_env["BLUEPRINT_OSCAR_WAM_HEIGHT"] == "480"
    assert runtime_env["BLUEPRINT_OSCAR_WAM_WIDTH"] == "640"
    assert runtime_env["BLUEPRINT_OSCAR_WAM_FPS"] == "12"
    assert runtime_env["BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS"] == "1200"
    assert runtime_env["BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER"] == "false"
    assert runtime_env["MY_API_KEY"] == "forwarded-secret"
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        "python3 -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT", "LucaFrat/groot-bs16")
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT", "nvidia/GEAR-SONIC")
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL",
        "tcp://127.0.0.1:5550",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER", "true")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE", "system_python_minimal")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT", "true")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON", "/opt/conda/bin/python")
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON_INSTALL_REQUIREMENTS",
        "huggingface_hub pyzmq",
    )
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_REPO_URL",
        "https://github.com/NVIDIA/Isaac-GR00T.git",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_REPO_REF", "main")
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_UV_SYNC_TIMEOUT_SECONDS",
        "1800",
    )
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SERVER_STARTUP_TIMEOUT_SECONDS",
        "900",
    )
    unitree_env = vpa._probe_env(job_dir=tmp_path / "unitree", enable_isaac_smoke=False)
    assert unitree_env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"].startswith("python3 -m")
    assert unitree_env["BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT"] == "LucaFrat/groot-bs16"
    assert unitree_env["BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT"] == "nvidia/GEAR-SONIC"
    assert (
        unitree_env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL"] == "tcp://127.0.0.1:5550"
    )
    assert unitree_env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER"] == "true"
    assert unitree_env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE"] == (
        "system_python_minimal"
    )
    assert unitree_env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT"] == "true"
    assert unitree_env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON"] == (
        "/opt/conda/bin/python"
    )
    assert unitree_env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_SYSTEM_PYTHON_INSTALL_REQUIREMENTS"] == (
        "huggingface_hub pyzmq"
    )
    assert unitree_env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_REPO_URL"].endswith(
        "/NVIDIA/Isaac-GR00T.git"
    )
    assert unitree_env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_REPO_REF"] == "main"
    assert unitree_env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_UV_SYNC_TIMEOUT_SECONDS"] == "1800"
    assert unitree_env["BLUEPRINT_UNITREE_GROOT_N17_SONIC_SERVER_STARTUP_TIMEOUT_SECONDS"] == "900"
    payload = vpa._create_payload(
        image="image",
        label="label",
        launch_mode="ssh_direct",
        probe_script="echo hi",
        disk_gb=20,
        env={"TOKEN": "secret"},
    )
    summary = vpa._create_request_summary(payload, secret_values=["secret"])
    assert summary["isaac_required_env_present"] == {
        "ACCEPT_EULA": False,
        "PRIVACY_CONSENT": False,
        "NVIDIA_DRIVER_CAPABILITIES": False,
    }
    assert summary["raw_payload_redacted"]["env"]["TOKEN"] == vpa.REDACTED_SECRET_FIELD
    assert vpa._instance_id_from_create_response({"new_contract": "123"}) == 123
    assert (
        vpa._sanitized_instance_row(
            {
                "instance_id": 77,
                "machine_id": 88,
                "gpu_display_name": "RTX 4090",
                "actual_status": "running",
                "price_per_hour": 0.2,
            }
        )["machine_id"]
        == 88
    )
    script = vpa._probe_shell_script(
        "https://heartbeat.example",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="isaac",
    )
    assert "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:isaac_python_missing" in script
    # Isaac images do not reliably expose a standalone libcudart to their
    # bundled Python.  The paid Isaac path therefore defers CUDA admission to
    # the SimulationApp + Warp smoke rather than running the generic ctypes
    # probe used by non-Isaac bundles.
    assert "BLUEPRINT_VAST_CUDA_RUNTIME_DEFERRED_TO_ISAAC_SIMULATION_APP" in script
    assert "wp.get_devices()" in script
    assert "isaac_simulation_app_warp" in script
    assert "cudaGetDeviceCount" not in script
    assert "BLUEPRINT_VAST_CUDA_RUNTIME_OK" in script
    assert "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:cuda_runtime_incompatible" in script
    assert "/isaac-sim/python.sh" in script
    with pytest.raises(ValueError, match="unsupported_provider_bundle_kind"):
        vpa._probe_shell_script("https://heartbeat.example", provider_bundle_kind="bad")
    wam_script = vpa._probe_shell_script(
        "https://heartbeat.example",
        enable_blueprint_bundle=True,
        provider_bundle_kind="wam",
    )
    assert "wam_provider_bundle" in wam_script
    assert vpa.VAST_INLINE_PROVIDER_BUNDLE_BASE64_ENV in wam_script
    assert "BLUEPRINT_VAST_INLINE_BUNDLE_DECODED" in wam_script
    assert "BLUEPRINT_VAST_INLINE_BUNDLE_SHA256_MISMATCH" in wam_script
    assert "BLUEPRINT_VAST_CUDA_RUNTIME_BLOCKED" in wam_script
    assert "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_DIAGNOSTIC_WRITTEN" in wam_script
    assert "provider_entrypoint_diagnostic.json" in wam_script
    assert "BLUEPRINT_VAST_PROVIDER_EARLY_DIAGNOSTIC_UPLOAD_OK" in wam_script
    assert "if [ -x /opt/conda/bin/python ]; then RUNTIME_PY=/opt/conda/bin/python" in wam_script
    assert "elif [ -x /usr/local/bin/python ]; then RUNTIME_PY=/usr/local/bin/python" in wam_script
    unitree_script = vpa._probe_shell_script(
        "https://heartbeat.example",
        enable_blueprint_bundle=True,
        provider_bundle_kind="unitree_unifolm",
    )
    assert "unitree_unifolm_provider_bundle" in unitree_script
    assert "run_unitree_unifolm_provider_runtime.sh" in unitree_script
    assert "unitree_unifolm_policy_provider_output.json" in unitree_script
    groot_script = vpa._probe_shell_script(
        "https://heartbeat.example",
        enable_blueprint_bundle=True,
        provider_bundle_kind="unitree_groot_n17_sonic",
    )
    assert "unitree_groot_n17_sonic_provider_bundle" in groot_script
    assert "run_unitree_groot_n17_sonic_provider_runtime.sh" in groot_script
    assert "unitree_groot_n17_sonic_policy_provider_output.json" in groot_script
    assert "BLUEPRINT_VAST_PROVIDER_PYTHON_DEPS_MISSING" in groot_script
    assert "msgpack-numpy" in groot_script
    evaluator_script = vpa._probe_shell_script(
        "https://heartbeat.example",
        enable_blueprint_bundle=True,
        provider_bundle_kind="evaluator",
    )
    for generated_script in (script, wam_script, unitree_script, groot_script, evaluator_script):
        subprocess.run(
            ["bash", "-n"],
            input=generated_script,
            text=True,
            check=True,
            capture_output=True,
        )

    monkeypatch.setattr(vpa.shutil, "which", lambda name: None if name == "ffprobe" else name)
    missing_video = vpa._ffprobe_video(tmp_path / "missing.mp4")
    assert missing_video["blockers"] == ["mp4_file_missing"]
    empty_video = tmp_path / "empty.mp4"
    empty_video.write_bytes(b"")
    unavailable = vpa._ffprobe_video(empty_video)
    assert "ffprobe_not_available" in unavailable["blockers"]

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(vpa.shutil, "which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        vpa.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [{"duration": "2.5", "nb_read_frames": "75"}],
                    "format": {"duration": "2.5"},
                }
            ),
            stderr="",
        ),
    )
    probed = vpa._ffprobe_video(video)
    assert probed["status"] == "completed"
    assert probed["duration_seconds"] == 2.5
    assert probed["frame_count"] == 75

    monkeypatch.setattr(
        vpa.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ffprobe exploded")),
    )
    failed_probe = vpa._ffprobe_video(video)
    assert failed_probe["blockers"] == ["ffprobe_failed:RuntimeError"]

    monkeypatch.setattr(
        vpa.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=2,
            stdout="{bad json",
            stderr="bad",
        ),
    )
    invalid_probe = vpa._ffprobe_video(video)
    assert "ffprobe_returncode:2" in invalid_probe["blockers"]
    assert "ffprobe_json_parse_failed" in invalid_probe["blockers"]
    assert "mp4_duration_not_positive" in invalid_probe["blockers"]
    assert "mp4_frame_count_not_positive" in invalid_probe["blockers"]

    blocked_video_zip = tmp_path / "blocked-videos.zip"
    with zipfile.ZipFile(blocked_video_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("camera.mp4", b"fake")
    monkeypatch.setattr(
        vpa,
        "_ffprobe_video",
        lambda path: {
            "status": "blocked",
            "path": str(path),
            "blockers": ["mp4_duration_not_positive"],
        },
    )
    blocked_inspection = vpa._inspect_provider_runtime_output_zip(
        blocked_video_zip,
        video_extract_dir=tmp_path / "blocked-extract",
        expected_video_count=1,
    )
    assert (
        "ffprobe_validation_failed_for_one_or_more_mp4s"
        in blocked_inspection["mp4_validation"]["blockers"]
    )
    no_extract_inspection = vpa._inspect_provider_runtime_output_zip(blocked_video_zip)
    assert no_extract_inspection["mp4_validation"]["blockers"] == [
        "mp4_ffprobe_validation_not_requested"
    ]

    invalid_video_dir = tmp_path / "invalid-video-smoke"
    invalid_video_dir.mkdir()
    (invalid_video_dir / "vast_video_smoke_result.json").write_text("{bad json", encoding="utf-8")
    vpa._write_blocked_phase_artifacts(
        job_dir=invalid_video_dir,
        generated_at="2026-06-20T00:00:00Z",
        provider_reason="unit-test",
    )
    assert _read_json(invalid_video_dir / "vast_video_smoke_result.json")["status"] == "blocked"

    final_dir = tmp_path / "final-corrupt-video"
    final_dir.mkdir()
    for name in (
        "vast_runtime_discovery.json",
        "vast_provider_plan.json",
        "vast_offer_selection_manifest.json",
        "vast_budget_ledger.json",
        "vast_startup_probe_manifest.json",
        "vast_gpu_sanity_report.json",
        "vast_isaac_smoke_result.json",
        "vast_provider_command_result.json",
        "vast_teardown_manifest.json",
    ):
        (final_dir / name).write_text("{}", encoding="utf-8")
    (final_dir / "vast_video_smoke_result.json").write_text("{bad json", encoding="utf-8")
    (final_dir / "vast_runtime_phase_log.jsonl").write_text(
        "\n".join(json.dumps({"phase": phase}) for phase in vpa.VAST_REQUIRED_PHASES),
        encoding="utf-8",
    )
    corrupt_video_validation = vpa._final_validation(
        job_dir=final_dir,
        generated_at="2026-06-20T00:00:00Z",
        instance_ids=[],
        continuing_spend=False,
        estimated_cost_usd=0.0,
        hard_cap_usd=1.0,
    )
    assert "json_parse_errors" in corrupt_video_validation["blockers"]
    assert vpa._release_vast_launch_lock(None) is None

    monkeypatch.setattr(
        vpa,
        "_api_json",
        lambda **_: (_ for _ in ()).throw(RuntimeError("inventory failed")),
    )
    inventory = vpa._prelaunch_inventory_guard(
        job_dir=tmp_path / "inventory",
        generated_at="2026-06-20T00:00:00Z",
        api_key="secret",
    )
    assert inventory["blockers"] == ["vast_prelaunch_inventory_query_failed"]

    inventory_calls = {"count": 0}

    def rate_limited_once_then_passes(**_kwargs):  # type: ignore[no-untyped-def]
        inventory_calls["count"] += 1
        if inventory_calls["count"] == 1:
            raise urllib.error.HTTPError(
                "https://console.vast.ai/api/v0/instances/",
                429,
                "Too Many Requests",
                {},
                None,
            )
        return 200, {"instances": []}

    monkeypatch.setattr(vpa, "_api_json", rate_limited_once_then_passes)
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    retried_inventory = vpa._prelaunch_inventory_guard(
        job_dir=tmp_path / "inventory-retried",
        generated_at="2026-06-20T00:00:00Z",
        api_key="secret",
    )
    assert retried_inventory["status"] == "passed"
    assert retried_inventory["query_attempt_count"] == 2
    assert inventory_calls["count"] == 2

    with pytest.raises(ValueError, match="unsupported_provider_bundle_kind"):
        run_vast_provider_adapter(job_dir=tmp_path / "bad-kind-run", provider_bundle_kind="bad")


def test_vast_adapter_request_logs_container_missing_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        vpa,
        "_api_json",
        lambda **_: (200, {"result_url": "https://logs.example/result"}),
    )
    texts = iter(["No such container\n", "No such container\n"])
    monkeypatch.setattr(vpa, "_fetch_text", lambda *_args, **_kwargs: next(texts))
    result = vpa._request_logs_and_fetch(
        instance_id=42,
        api_key="secret",
        output_log_path=tmp_path / "container-missing.log",
        secret_values=[],
        wait_seconds=0,
        retry_interval_seconds=1,
        max_wait_seconds=60,
        success_markers=["SUCCESS"],
        container_missing_retry_attempts=2,
    )
    assert len(result["log_poll_attempts"]) == 2
    assert result["log_poll_attempts"][-1]["container_missing_observed_count"] == 2
    assert (tmp_path / "container-missing.log").read_text(encoding="utf-8") == (
        "No such container\n"
    )


def test_adp_simpler_uses_bounded_cold_pull_container_window() -> None:
    assert vpa._container_missing_max_seconds("isaac") == 720
    assert vpa._container_missing_max_seconds("adp_simready_isaac") == 720
    assert vpa._container_missing_max_seconds("adp_simpler") == 720
    assert vpa._container_missing_max_seconds("unitree_unifolm") == 60


def test_exact_simready_isaac_bundle_does_not_require_policy_video() -> None:
    assert vpa._provider_expected_video_count("isaac") > 0
    assert vpa._provider_expected_video_count("adp_simready_isaac") == 0


def test_exact_simready_isaac_bundle_forces_http1_download() -> None:
    script = vpa._probe_shell_script(
        "https://example.invalid/heartbeat",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_simready_isaac",
    )

    # Intent, not flag order: HTTP/1.1 with retries, however it is spelled.
    # A literal pin here broke when retry flags were added between the two.
    assert '--http1.1' in script and '--retry-all-errors' in script
    assert '-fL "$blueprint_download_src"' in script


def test_vast_adapter_falls_back_to_command_execute_after_missing_container_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "secret-vast-key"
    key_file = tmp_path / "vast_api_key"
    key_file.write_text(secret + "\n", encoding="utf-8")
    key_file.chmod(0o600)
    monkeypatch.setenv(vpa.VAST_API_KEY_FILE_ENV, str(key_file))
    monkeypatch.setenv(vpa.VAST_API_GATE_ENV, "true")
    monkeypatch.setenv(vpa.VAST_INSTANCE_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(vpa.VAST_ALLOW_COMMAND_EXECUTE_SCRIPT_FALLBACK_ENV, "true")
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)

    calls: list[tuple[str, str]] = []

    def fake_api_json(
        *,
        method: str,
        path: str,
        api_key: str,
        payload=None,
        timeout_seconds: int = 30,
    ):  # type: ignore[no-untyped-def]
        assert api_key == secret
        calls.append((method, path))
        if method == "GET" and path == "/instances/":
            return 200, {"instances": []}
        if method == "POST" and path == "/bundles/":
            return 200, {
                "offers": [
                    {
                        "id": 101,
                        "ask_contract_id": 101,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.42,
                        "num_gpus": 1,
                        "rentable": True,
                        "verified": True,
                    }
                ]
            }
        if method == "PUT" and path == "/asks/101/":
            return 200, {"success": True, "new_contract": 556}
        if method == "GET" and path == "/instances/556/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if method == "PUT" and path == "/instances/request_logs/556":
            return 200, {"success": True, "result_url": "https://logs.example/request"}
        if method == "PUT" and path == "/instances/command/556/":
            assert payload is not None
            assert "BLUEPRINT_VAST_ONSTART_STARTED" in payload["command"]
            return 200, {"success": True, "result_url": "https://logs.example/execute"}
        if method == "DELETE" and path == "/instances/556/":
            return 200, {"success": True, "msg": "Instance destroyed successfully"}
        raise AssertionError((method, path))

    def fake_fetch_text(url: str, timeout_seconds: int = 30) -> str:
        if url == "https://logs.example/request":
            return "Error response from daemon: No such container: C.556\n"
        if url == "https://logs.example/execute":
            return (
                "BLUEPRINT_VAST_HEARTBEAT_OK\n"
                "RTX 4090, 590.48, 24576 MiB\n"
                "BLUEPRINT_VAST_GPU_SANITY_OK\n"
            )
        raise AssertionError(url)

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(vpa, "_fetch_text", fake_fetch_text)

    result = run_vast_provider_adapter(
        job_dir=tmp_path,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=20,
    )

    assert result["status"] == "completed"
    assert ("PUT", "/instances/command/556/") in calls
    heartbeat = _read_json(tmp_path / "vast_startup_probe_manifest.json")
    assert heartbeat["status"] == "completed"
    assert heartbeat["container_log_result"]["effective_log_source"] == "command_execute_fallback"
    gpu = _read_json(tmp_path / "vast_gpu_sanity_report.json")
    assert gpu["status"] == "completed"


def test_execute_and_fetch_records_api_error_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_api_json(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("provider execute temporarily unavailable")

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    result = vpa._execute_and_fetch(
        instance_id=123,
        api_key="secret",
        command="echo hi",
        output_log_path=tmp_path / "execute.log",
        secret_values=[],
        wait_seconds=0,
    )

    assert result["http_status_code"] == 0
    assert result["result_url_present"] is False
    assert "RuntimeError" in result["api_request_error"]
    assert (tmp_path / "execute.log").is_file()


def test_vast_adapter_io_zip_poll_and_validation_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __init__(self, payload: bytes, status: int = 200) -> None:
            self._payload = payload
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return self._payload

    responses = iter(
        [
            FakeResponse(b"", status=204),
            FakeResponse(b"[1, 2]", status=200),
            FakeResponse(b"hello", status=200),
        ]
    )
    monkeypatch.setattr(
        "blueprint_pipeline.safe_outbound_http._open_with_policy",
        lambda *_, **__: next(responses),
    )

    assert vpa._api_json(method="GET", path="https://example.invalid", api_key="k") == (204, {})
    assert vpa._api_json(method="POST", path="/path", api_key="k", payload={"a": 1}) == (
        200,
        {"response": [1, 2]},
    )
    assert vpa._fetch_text("https://example.invalid/log") == "hello"

    monkeypatch.setattr(vpa.shutil, "which", lambda _name: None)
    assert vpa._vastai_version() == {"present": False, "path": None}

    assert vpa._poll_instance(
        instance_id=1,
        api_key="k",
        timeout_seconds=0,
        poll_interval_seconds=0,
    ) == ("unknown", [], {})

    poll_calls = iter([(200, {"status": "queued"}), (200, {"status": "running"})])
    monkeypatch.setattr(vpa, "_api_json", lambda **_: next(poll_calls))
    assert (
        vpa._poll_instance(
            instance_id=2,
            api_key="k",
            timeout_seconds=10,
            poll_interval_seconds=0,
        )[0]
        == "running"
    )

    never_started_payload = {
        "actual_status": "created",
        "cur_state": "stopped",
        "intended_status": "stopped",
        "uptime": None,
    }
    monkeypatch.setattr(vpa, "_api_json", lambda **_: (200, never_started_payload))
    stopped_status, stopped_observations, _ = vpa._poll_instance(
        instance_id=3,
        api_key="k",
        timeout_seconds=10,
        poll_interval_seconds=0,
    )
    assert stopped_status == "stopped_before_start"
    assert stopped_observations[0]["status"] == "stopped_before_start"

    monkeypatch.setattr(
        vpa,
        "_api_json",
        lambda **_: (
            200,
            {
                "result_url": (
                    "https://logs.invalid/result?X-Amz-Signature=abc123&X-Amz-Credential=credential"
                )
            },
        ),
    )
    monkeypatch.setattr(vpa, "_fetch_text", lambda *_args, **_kwargs: "secret output")
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    execute_result = vpa._execute_and_fetch(
        instance_id=12,
        api_key="k",
        command="echo secret",
        output_log_path=tmp_path / "execute.log",
        secret_values=["secret"],
        wait_seconds=0,
    )
    assert execute_result["result_url_present"] is True
    assert execute_result["result_url"] == "https://logs.invalid/result?REDACTED_QUERY"
    assert "abc123" not in json.dumps(execute_result)
    assert (tmp_path / "execute.log").read_text(encoding="utf-8") == (
        f"{vpa.REDACTED_SECRET} output"
    )
    log_result = vpa._request_logs_and_fetch(
        instance_id=12,
        api_key="k",
        output_log_path=tmp_path / "request.log",
        secret_values=["secret"],
        wait_seconds=0,
        retry_interval_seconds=1,
        max_wait_seconds=0,
    )
    assert log_result["http_status_code"] == 200
    assert log_result["result_url"] == "https://logs.invalid/result?REDACTED_QUERY"
    assert "abc123" not in json.dumps(log_result)
    assert log_result["log_poll_attempts"][0]["success_marker_found"] is False

    assert vpa._inspect_provider_runtime_output_zip(None)["status"] == "not_configured"
    assert vpa._inspect_provider_runtime_output_zip(tmp_path / "missing.zip")["status"] == "missing"
    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_text("not a zip", encoding="utf-8")
    assert vpa._inspect_provider_runtime_output_zip(bad_zip)["status"] == "blocked"
    invalid_json_zip = tmp_path / "invalid-json.zip"
    with zipfile.ZipFile(invalid_json_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("isaac_runtime_result.json", "{bad json")
    inspected = vpa._inspect_provider_runtime_output_zip(invalid_json_zip)
    assert inspected["status"] == "completed"
    assert inspected["runtime_result_present"] is False
    assert inspected["json_parse_errors"] == ["isaac_runtime_result.json:JSONDecodeError"]
    video_zip = tmp_path / "videos.zip"
    with zipfile.ZipFile(video_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(6):
            archive.writestr(f"realistic_videos/camera_{index}.mp4", b"fake mp4")
    monkeypatch.setattr(
        vpa,
        "_ffprobe_video",
        lambda path: {
            "status": "completed",
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "duration_seconds": 1.0,
            "frame_count": 10,
            "blockers": [],
        },
    )
    video_inspected = vpa._inspect_provider_runtime_output_zip(
        video_zip,
        video_extract_dir=tmp_path / "extracted-videos",
        expected_video_count=6,
    )
    assert video_inspected["video_smoke_proven"] is True
    assert video_inspected["mp4_validation"]["validated_mp4_count"] == 6

    fill_dir = tmp_path / "fill-missing"
    fill_dir.mkdir()
    phase_path = fill_dir / "vast_runtime_phase_log.jsonl"
    phase_path.write_text("\n{bad json\n", encoding="utf-8")
    vpa._fill_missing_phase_rows(fill_dir, reason="unit-test")
    assert "vast_docs_checked" in phase_path.read_text(encoding="utf-8")

    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    vpa._ensure_offer_manifest(
        validation_dir,
        generated_at="2026-06-20T00:00:00Z",
        blockers=["blocked"],
    )
    assert (validation_dir / "vast_offer_selection_manifest.json").is_file()
    (validation_dir / "broken.json").write_text("{bad json", encoding="utf-8")
    (validation_dir / "vast_runtime_phase_log.jsonl").write_text(
        "\n{bad phase json\n" + json.dumps({"phase": "vast_docs_checked"}) + "\n",
        encoding="utf-8",
    )
    validation = vpa._final_validation(
        job_dir=validation_dir,
        generated_at="2026-06-20T00:00:00Z",
        instance_ids=[123],
        continuing_spend=True,
        estimated_cost_usd=2.0,
        hard_cap_usd=1.0,
    )
    assert validation["status"] == "blocked"
    assert "missing_required_vast_artifacts" in validation["blockers"]
    assert "json_parse_errors" in validation["blockers"]
    assert "missing_required_vast_runtime_phases" in validation["blockers"]
    assert "continuing_vast_spend_detected" in validation["blockers"]
    assert "vast_estimated_spend_exceeded_hard_cap" in validation["blockers"]


def test_vast_adapter_missing_grant_dominates_provider_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_live_gates(tmp_path, monkeypatch)

    def inventory_only_api(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        raise AssertionError("Vast create/search reached without opaque grant")

    monkeypatch.setattr(vpa, "_api_json", inventory_only_api)
    result = run_vast_provider_adapter(
        job_dir=tmp_path / "missing-grant",
        mode="live-startup-probe",
        allow_vast_api_call=True,
        allow_instance_launch=True,
        session_max_live_minutes=None,
    )

    assert result["status"] == "blocked"
    assert result["vast_side_effects_may_have_occurred"] is False
    assert "vast_provider_shared_admission_missing_or_invalid" in result["blockers"]


def test_vast_adapter_live_error_paths_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)

    unsupported = run_vast_provider_adapter(job_dir=tmp_path / "unsupported", mode="invalid")
    assert unsupported["status"] == "blocked"
    assert unsupported["blockers"] == ["unsupported_vast_adapter_mode:invalid"]

    def no_offer_api(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            return 200, {"offers": []}
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", no_offer_api)
    no_offer = run_vast_provider_adapter(
        job_dir=tmp_path / "no-offer",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        session_max_live_minutes=None,
    )
    assert no_offer["status"] == "blocked"
    assert no_offer["blockers"] == ["no_vast_offer_at_or_below_max_hourly_rate"]

    no_compute_cap_offer = run_vast_provider_adapter(
        job_dir=tmp_path / "no-compute-cap-offer",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        session_max_live_minutes=None,
        min_compute_cap=800,
    )
    assert no_compute_cap_offer["status"] == "blocked"
    assert no_compute_cap_offer["blockers"] == [
        "no_vast_offer_meeting_min_compute_cap",
        "no_vast_offer_at_or_below_max_hourly_rate",
    ]

    def no_allowed_offer_api(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            return 200, {
                "offers": [
                    {
                        "id": 11,
                        "ask_contract_id": 11,
                        "gpu_name": "RTX A6000",
                        "gpu_ram_mb": 49140,
                        "dph_total": 0.40,
                        "driver_version": "580.159.03",
                        "machine_id": 111,
                    }
                ]
            }
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", no_allowed_offer_api)
    no_allowed_offer = run_vast_provider_adapter(
        job_dir=tmp_path / "no-allowed-offer",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        session_max_live_minutes=None,
        allowed_machine_ids=[222],
    )
    assert no_allowed_offer["status"] == "blocked"
    assert no_allowed_offer["blockers"] == [
        "no_vast_offer_matching_allowed_machine_ids",
        "no_vast_offer_at_or_below_max_hourly_rate",
    ]

    def expensive_offer_api(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            return 200, {
                "offers": [
                    {
                        "id": 10,
                        "ask_contract_id": 10,
                        "gpu_name": "RTX 4090",
                        "dph_total": 2.0,
                        "driver_version": "580.95.05",
                    }
                ]
            }
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", expensive_offer_api)
    original_session_budget_guard = vpa._session_budget_guard
    monkeypatch.setattr(
        vpa,
        "_session_budget_guard",
        lambda **kwargs: {
            "schema_version": "vast_session_budget_guard.v1",
            "generated_at": kwargs["generated_at"],
            "status": "passed",
            "blockers": [],
            "warnings": [],
        },
    )
    expensive = run_vast_provider_adapter(
        job_dir=tmp_path / "expensive",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        max_hourly_rate=3.0,
        hard_cap_usd=0.01,
        max_live_minutes=60,
        session_max_live_minutes=None,
    )
    assert expensive["status"] == "failed"
    assert expensive["blockers"] == ["selected_offer_projected_max_runtime_exceeds_hard_cap"]
    monkeypatch.setattr(vpa, "_session_budget_guard", original_session_budget_guard)

    def create_missing_id_api(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            return 200, {
                "offers": [
                    {
                        "id": 20,
                        "ask_contract_id": 20,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.2,
                        "driver_version": "580.95.05",
                    }
                ]
            }
        if kwargs["method"] == "PUT" and kwargs["path"] == "/asks/20/":
            return 200, {"success": True}
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", create_missing_id_api)
    missing_id = run_vast_provider_adapter(
        job_dir=tmp_path / "missing-id",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        session_max_live_minutes=None,
    )
    assert missing_id["status"] == "failed"
    assert missing_id["blockers"] == ["vast_create_response_missing_instance_id"]

    def failed_instance_api(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            return 200, {
                "offers": [
                    {
                        "id": 30,
                        "ask_contract_id": 30,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.2,
                        "driver_version": "580.95.05",
                    }
                ]
            }
        if kwargs["method"] == "PUT" and kwargs["path"] == "/asks/30/":
            return 200, {"new_contract": 300}
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/300/":
            return 200, {"instances": {"actual_status": "failed", "cur_state": "failed"}}
        if kwargs["method"] == "DELETE":
            return 200, {"success": True}
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", failed_instance_api)
    failed_instance = run_vast_provider_adapter(
        job_dir=tmp_path / "failed-instance",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        session_max_live_minutes=None,
    )
    assert failed_instance["status"] == "failed"
    assert failed_instance["blockers"] == ["vast_instance_not_running:failed"]
    startup = _read_json(tmp_path / "failed-instance" / "vast_startup_probe_manifest.json")
    assert startup["blockers"] == ["vast_instance_status:failed"]

    def http_error_api(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        raise urllib.error.HTTPError(
            "https://vast.invalid",
            503,
            "unavailable",
            {},
            BytesIO(f"{secret} leaked".encode("utf-8")),
        )

    monkeypatch.setattr(vpa, "_api_json", http_error_api)
    http_error = run_vast_provider_adapter(
        job_dir=tmp_path / "http-error",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        session_max_live_minutes=None,
    )
    assert http_error["status"] == "failed"
    assert http_error["http_status_code"] == 503
    assert secret not in http_error["vast_error"]

    def interrupt_api(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        raise KeyboardInterrupt

    monkeypatch.setattr(vpa, "_api_json", interrupt_api)
    interrupted = run_vast_provider_adapter(
        job_dir=tmp_path / "interrupted",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        session_max_live_minutes=None,
    )
    assert interrupted["status"] == "blocked"
    assert interrupted["blockers"] == ["vast_probe_interrupted_before_completion"]


def test_vast_adapter_blocks_isaac_image_preflight_before_offer_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_live_gates(tmp_path, monkeypatch)

    def fail_if_called(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected Vast API call: {kwargs}")

    monkeypatch.setattr(vpa, "_api_json", fail_if_called)
    result = run_vast_provider_adapter(
        job_dir=tmp_path / "image-preflight-blocked",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        enable_isaac_smoke=True,
        use_vast_template_image=True,
        vast_template_hash_id=None,
        max_live_minutes=8,
        session_max_live_minutes=None,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "vast_isaac_image_startup_preflight_blocked"
    assert result["api_call_performed"] is False
    assert "vast_template_hash_required_when_using_template_image" in result["blockers"]
    assert "vast_template_image_cache_not_proven_for_short_live_window" in result["blockers"]
    teardown = _read_json(tmp_path / "image-preflight-blocked" / "vast_teardown_manifest.json")
    assert teardown["status"] == "not_required_isaac_image_startup_preflight_blocked"


def test_vast_adapter_signal_handler_ignore_raise_and_registration_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("BLUEPRINT_VAST_IGNORE_LOCAL_SIGTERM_DURING_PROVIDER_RUN", "true")
    captured: dict[int, object] = {}

    def signal_with_restore_failure(signum: int, handler: object) -> object:
        if callable(handler):
            captured[signum] = handler
            return "previous-handler"
        raise OSError("restore failed")

    monkeypatch.setattr(vpa.signal, "signal", signal_with_restore_failure)

    def ignore_signal_api(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            sigint_handler = captured[vpa.signal.SIGINT]
            sigterm_handler = captured[vpa.signal.SIGTERM]
            assert callable(sigint_handler)
            assert callable(sigterm_handler)
            sigint_handler(vpa.signal.SIGINT, None)
            sigterm_handler(vpa.signal.SIGTERM, None)
            return 200, {"offers": []}
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", ignore_signal_api)
    ignored = run_vast_provider_adapter(
        job_dir=tmp_path / "ignored-signal",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        session_max_live_minutes=None,
    )
    assert ignored["status"] == "blocked"
    signal_manifest = _read_json(tmp_path / "ignored-signal" / "vast_signal_handling_manifest.json")
    assert signal_manifest["status"] == "ignored_local_probe_signal"
    assert signal_manifest["ignored_signal_counts"][str(vpa.signal.SIGINT)] == 1
    assert signal_manifest["ignored_signal_counts"][str(vpa.signal.SIGTERM)] == 1

    monkeypatch.delenv("BLUEPRINT_VAST_IGNORE_LOCAL_SIGTERM_DURING_PROVIDER_RUN", raising=False)
    captured.clear()
    monkeypatch.setattr(
        vpa.signal,
        "signal",
        lambda signum, handler: captured.setdefault(signum, handler) or "previous",
    )

    def raising_signal_api(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            handler = captured[vpa.signal.SIGINT]
            assert callable(handler)
            handler(vpa.signal.SIGINT, None)
            raise AssertionError("handler should raise before this line")
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", raising_signal_api)
    raised = run_vast_provider_adapter(
        job_dir=tmp_path / "raised-signal",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        session_max_live_minutes=None,
    )
    assert raised["status"] == "blocked"
    assert raised["reason"] == "vast_probe_interrupted"

    monkeypatch.setattr(
        vpa.signal,
        "signal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("registration failed")),
    )
    monkeypatch.setattr(
        vpa,
        "_api_json",
        lambda **kwargs: (
            (200, {"instances": []}) if kwargs["method"] == "GET" else (200, {"offers": []})
        ),
    )
    registration_failed = run_vast_provider_adapter(
        job_dir=tmp_path / "signal-registration-failed",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        session_max_live_minutes=None,
    )
    assert registration_failed["status"] == "blocked"
    assert registration_failed["blockers"] == ["no_vast_offer_at_or_below_max_hourly_rate"]


def test_vast_adapter_mocked_wam_bundle_marks_isaac_not_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    bundle = tmp_path / "wam-bundle.zip"
    _write_valid_wam_provider_bundle(bundle)

    def fake_api_json(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["api_key"] == secret
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            return 200, {
                "offers": [
                    {
                        "id": 818,
                        "ask_contract_id": 818,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.2,
                        "driver_version": "580.95.05",
                    }
                ]
            }
        if kwargs["method"] == "PUT" and kwargs["path"] == "/asks/818/":
            return 200, {"new_contract": 8181}
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/8181/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if kwargs["method"] == "PUT" and kwargs["path"] == "/instances/request_logs/8181":
            return 200, {"success": True, "result_url": "https://logs.example/wam"}
        if kwargs["method"] == "DELETE":
            return 200, {"success": True}
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(
        vpa,
        "_fetch_text",
        lambda *_args, **_kwargs: (
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "RTX 4090, 580.95.05, 24564 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED\n"
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED\n"
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE:0\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED\n"
            "BLUEPRINT_VAST_ONSTART_DONE\n"
        ),
    )

    result = run_vast_provider_adapter(
        job_dir=tmp_path / "wam-live",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle_kind="wam",
        provider_bundle=bundle,
        provider_bundle_url="https://example.invalid/wam.zip",
        provider_output_put_url="https://example.invalid/out.zip",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        poll_interval_seconds=0,
        startup_timeout_seconds=10,
        session_max_live_minutes=None,
    )

    assert result["status"] == "blocked"
    isaac = _read_json(tmp_path / "wam-live" / "vast_isaac_smoke_result.json")
    assert isaac["status"] == "not_required"
    assert isaac["provider_bundle_kind"] == "wam"


def test_vast_adapter_isaac_ngc_missing_blocks_smoke_after_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setenv("NGC_API_KEY_FILE", str(tmp_path / "missing-ngc"))
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)

    def fake_api_json(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["api_key"] == secret
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            return 200, {
                "offers": [
                    {
                        "id": 919,
                        "ask_contract_id": 919,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.2,
                        "driver_version": "580.95.05",
                    }
                ]
            }
        if kwargs["method"] == "PUT" and kwargs["path"] == "/asks/919/":
            assert "image_login" not in kwargs["payload"]
            return 200, {"new_contract": 9191}
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/9191/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if kwargs["method"] == "PUT" and kwargs["path"] == "/instances/request_logs/9191":
            return 200, {"success": True, "result_url": "https://logs.example/isaac-ngc"}
        if kwargs["method"] == "DELETE":
            return 200, {"success": True}
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(
        vpa,
        "_fetch_text",
        lambda *_args, **_kwargs: (
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "RTX 4090, 580.95.05, 24564 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
            "BLUEPRINT_VAST_ONSTART_DONE\n"
        ),
    )

    result = run_vast_provider_adapter(
        job_dir=tmp_path / "isaac-ngc-missing",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        enable_isaac_smoke=True,
        isaac_image="nvcr.io/private/isaac:1",
        ngc_image_login_mode="always",
        poll_interval_seconds=0,
        startup_timeout_seconds=10,
        session_max_live_minutes=None,
    )

    assert result["status"] == "completed"
    isaac = _read_json(tmp_path / "isaac-ngc-missing" / "vast_isaac_smoke_result.json")
    assert isaac["status"] == "blocked"
    assert "ngc_api_key_file_missing_or_empty_for_required_ngc_login" in isaac["blockers"]


def test_vast_adapter_provider_blockers_after_mocked_preflight_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        vpa,
        "_blueprint_bundle_preflight",
        lambda **kwargs: {
            "schema_version": vpa.VAST_BLUEPRINT_BUNDLE_PREFLIGHT_SCHEMA_VERSION,
            "generated_at": kwargs["generated_at"],
            "status": "passed",
            "blockers": [],
        },
    )

    def fake_api_json(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["api_key"] == secret
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            return 200, {
                "offers": [
                    {
                        "id": 616,
                        "ask_contract_id": 616,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.2,
                        "driver_version": "580.95.05",
                    }
                ]
            }
        if kwargs["method"] == "PUT" and kwargs["path"] == "/asks/616/":
            return 200, {"new_contract": 6161}
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/6161/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if kwargs["method"] == "PUT" and kwargs["path"] == "/instances/request_logs/6161":
            return 200, {"success": True, "result_url": "https://logs.example/provider-blockers"}
        if kwargs["method"] == "DELETE":
            return 200, {"success": True}
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(
        vpa,
        "_fetch_text",
        lambda *_args, **_kwargs: (
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "RTX 4090, 580.95.05, 24564 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
            "BLUEPRINT_VAST_ONSTART_DONE\n"
        ),
    )

    result = run_vast_provider_adapter(
        job_dir=tmp_path / "provider-blockers",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        max_live_minutes=20,
        poll_interval_seconds=0,
        startup_timeout_seconds=10,
        session_max_live_minutes=None,
    )

    assert result["status"] == "blocked"
    provider = _read_json(tmp_path / "provider-blockers" / "vast_provider_command_result.json")
    assert "blueprint_bundle_execution_requires_isaac_smoke_path" in provider["blockers"]
    assert "isaac_provider_runtime_bundle_missing" not in provider["blockers"]
    assert "provider_bundle_fetch_url_missing" in provider["blockers"]
    assert "provider_output_put_url_missing" in provider["blockers"]


def test_vast_adapter_run_clears_stale_artifacts_and_blocks_session_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_live_gates(tmp_path, monkeypatch)
    job_dir = tmp_path / "stale-job"
    job_dir.mkdir()
    stale_names = [
        "vast_offer_selection_manifest.json",
        "vast_startup_probe_manifest.json",
        "vast_isaac_smoke_result.json",
        "vast_final_validation.json",
        "vast_session_budget_guard.json",
    ]
    for stale_name in stale_names:
        (job_dir / stale_name).write_text('{"stale": true}', encoding="utf-8")
    (job_dir / "vast_runtime_phase_log.jsonl").write_text('{"phase":"old"}\n', encoding="utf-8")
    stale_output = job_dir / "provider_runtime_output.zip"
    stale_output.write_bytes(b"old-provider-output")
    budget = tmp_path / "budget.json"
    budget.write_text(
        json.dumps(
            {
                "attempts": [
                    {
                        "actual_live_runtime_seconds_observed_by_adapter": 60,
                        "estimated_cost_usd": 3.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_vast_provider_adapter(
        job_dir=job_dir,
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        session_budget_ledger_path=budget,
        session_max_live_minutes=10,
        hard_cap_usd=1.0,
        provider_runtime_output_zip=stale_output,
        provider_output_get_url="https://object.example/current-attempt.zip",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "vast_session_budget_guard_blocked"
    assert result["provider_create_attempted"] is False
    guard = _read_json(job_dir / "vast_session_budget_guard.json")
    assert "session_estimated_spend_hard_cap_exhausted" in guard["blockers"]
    preservation = _read_json(job_dir / "vast_latest_attempt_preservation_manifest.json")
    assert preservation["reason"] == "preserve_existing_live_attempt_before_new_vast_run"
    assert preservation["raw_secret_values_recorded"] is False
    assert "vast_runtime_phase_log.jsonl" in preservation["copied_artifacts"]
    assert "vast_startup_probe_manifest.json" in preservation["copied_artifacts"]
    assert "provider_runtime_output.zip" in preservation["copied_artifacts"]
    preserved_dir = Path(preservation["preserve_dir"])
    assert (preserved_dir / "vast_runtime_phase_log.jsonl").read_text(
        encoding="utf-8"
    ) == '{"phase":"old"}\n'
    assert json.loads(
        (preserved_dir / "vast_startup_probe_manifest.json").read_text(encoding="utf-8")
    ) == {"stale": True}
    assert (preserved_dir / "provider_runtime_output.zip").read_bytes() == b"old-provider-output"
    assert not stale_output.exists()
    assert (job_dir / "vast_runtime_phase_log.jsonl").read_text(encoding="utf-8") != (
        '{"phase":"old"}\n'
    )


def test_vast_adapter_non_rt_gpu_and_gpu_failure_block_isaac_and_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setenv("NGC_API_KEY_FILE", str(tmp_path / "missing-ngc-key"))
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    bundle = tmp_path / "bundle.zip"
    _write_valid_provider_bundle(bundle)

    def fake_api_json(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["api_key"] == secret
        if kwargs["method"] == "POST":
            return 200, {
                "offers": [
                    {
                        "id": 606,
                        "ask_contract_id": 606,
                        "gpu_name": "GTX 1080",
                        "dph_total": 0.2,
                        "driver_version": "565.57.01",
                        "machine_id": 6060,
                    }
                ]
            }
        if kwargs["method"] == "PUT" and kwargs["path"] == "/asks/606/":
            return 200, {"new_contract": 6061}
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "GET":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if kwargs["method"] == "PUT" and kwargs["path"] == "/instances/request_logs/6061":
            return 200, {"success": True, "result_url": "https://logs.example/non-rt"}
        if kwargs["method"] == "DELETE":
            return 200, {"success": True}
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(
        vpa,
        "_fetch_text",
        lambda *_args, **_kwargs: (
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "nvidia-smi: command not found\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
            "BLUEPRINT_VAST_ONSTART_DONE\n"
        ),
    )

    result = run_vast_provider_adapter(
        job_dir=tmp_path / "non-rt",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle=bundle,
        provider_bundle_url="https://example.invalid/bundle.zip",
        provider_output_put_url="https://example.invalid/out.zip",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=10,
    )

    assert result["status"] == "blocked"
    isaac = _read_json(tmp_path / "non-rt" / "vast_isaac_smoke_result.json")
    assert "selected_gpu_not_in_isaac_rt_candidate_allowlist" in isaac["blockers"]
    assert "ngc_api_key_file_missing_or_empty_for_required_ngc_login" not in isaac["blockers"]
    assert "gpu_sanity_not_proven" in isaac["blockers"]
    provider = _read_json(tmp_path / "non-rt" / "vast_provider_command_result.json")
    assert "provider_bundle_start_marker_missing" in provider["blockers"]


def test_vast_adapter_provider_marker_missing_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = _configure_live_gates(tmp_path, monkeypatch)
    ngc_key = tmp_path / "ngc_api_key"
    _write_secret(ngc_key, "secret-ngc-key")
    monkeypatch.setenv("NGC_API_KEY_FILE", str(ngc_key))
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    bundle = tmp_path / "bundle.zip"
    _write_valid_provider_bundle(bundle)

    def fake_api_json(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["api_key"] == secret
        if kwargs["method"] == "POST":
            return 200, {
                "offers": [
                    {
                        "id": 707,
                        "ask_contract_id": 707,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.2,
                        "driver_version": "580.95.05",
                    }
                ]
            }
        if kwargs["method"] == "PUT" and kwargs["path"] == "/asks/707/":
            return 200, {"new_contract": 7071}
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "GET":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if kwargs["method"] == "PUT" and kwargs["path"] == "/instances/request_logs/7071":
            return 200, {"success": True, "result_url": "https://logs.example/missing-markers"}
        if kwargs["method"] == "DELETE":
            return 200, {"success": True}
        raise AssertionError(kwargs)

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(
        vpa,
        "_fetch_text",
        lambda *_args, **_kwargs: (
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "RTX 4090, 580.95.05, 24564 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
            "BLUEPRINT_VAST_ISAAC_SMOKE_OK\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:download_failed:23\n"
            "BLUEPRINT_VAST_ONSTART_DONE\n"
        ),
    )

    result = run_vast_provider_adapter(
        job_dir=tmp_path / "missing-markers",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        public_image=DEFAULT_ISAAC_IMAGE,
        provider_bundle=bundle,
        provider_bundle_url="https://example.invalid/bundle.zip",
        provider_output_put_url="https://example.invalid/out.zip",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=10,
    )

    assert result["status"] == "blocked"
    provider = _read_json(tmp_path / "missing-markers" / "vast_provider_command_result.json")
    assert "provider_bundle_start_marker_missing" in provider["blockers"]
    assert "provider_entrypoint_start_marker_missing" in provider["blockers"]
    assert "provider_output_upload_marker_missing" in provider["blockers"]
    assert "provider_remote_blocker:download_failed:23" in provider["blockers"]


def test_vast_adapter_blueprint_bundle_requires_isaac_smoke_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_live_gates(tmp_path, monkeypatch)
    bundle = tmp_path / "bundle.zip"
    _write_valid_provider_bundle(bundle)

    def fail_if_called(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected Vast API call: {kwargs}")

    monkeypatch.setattr(vpa, "_api_json", fail_if_called)

    result = run_vast_provider_adapter(
        job_dir=tmp_path / "bundle-needs-isaac",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle=bundle,
        provider_bundle_url="https://example.invalid/bundle.zip",
        provider_output_put_url="https://example.invalid/out.zip",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        poll_interval_seconds=0,
        startup_timeout_seconds=10,
        session_max_live_minutes=None,
    )

    assert result["status"] == "blocked"
    assert result["api_call_performed"] is False
    assert result["reason"] == "vast_blueprint_bundle_preflight_blocked"
    provider = _read_json(tmp_path / "bundle-needs-isaac" / "vast_provider_command_result.json")
    assert provider["blockers"] == ["blueprint_bundle_execution_requires_isaac_smoke_path"]
    offer = _read_json(tmp_path / "bundle-needs-isaac" / "vast_offer_selection_manifest.json")
    assert offer["offer_search_performed"] is False


def test_vast_adapter_main_prints_success_and_blocked_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []

    def fake_success(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"status": "completed", "vast_instance_ids": [1, 2]}

    monkeypatch.setattr(vpa, "run_vast_provider_adapter", fake_success)
    assert (
        vpa.main(
            [
                "--job-dir",
                str(tmp_path / "cli-success"),
                "--mode",
                "live-startup-probe",
                "--allow-vast-api-call",
                "--allow-vast-instance-launch",
                "--provider-bundle",
                str(tmp_path / "bundle.zip"),
                "--provider-bundle-url",
                "https://example.invalid/bundle.zip?token=abc",
                "--provider-output-put-url",
                "https://example.invalid/out.zip?token=abc",
                "--provider-runtime-output-zip",
                str(tmp_path / "out.zip"),
                "--enable-isaac-smoke",
                "--enable-blueprint-bundle",
                "--vast-launch-mode",
                "args",
                "--ngc-image-login-mode",
                "never",
                "--disk-gb",
                "64",
                "--poll-interval-seconds",
                "0",
                "--startup-timeout-seconds",
                "1",
                "--heartbeat-no-progress-seconds",
                "2",
                "--machine-avoidlist",
                str(tmp_path / "avoid.json"),
                "--session-budget-ledger",
                str(tmp_path / "session-cost.json"),
                "--session-max-live-minutes",
                "12",
                "--verify-staging-urls",
                "--allow-staging-output-put-probe",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "legacy_vast_provider_mutation_cli_disabled" in captured.err
    assert calls == []

    def fake_blocked(**_kwargs):  # type: ignore[no-untyped-def]
        return {"status": "blocked", "vast_instance_ids": [], "blockers": ["blocked"]}

    monkeypatch.setattr(vpa, "run_vast_provider_adapter", fake_blocked)
    assert vpa.main(["--job-dir", str(tmp_path / "cli-blocked")]) == 1
    blocked_output = capsys.readouterr().out
    assert "status=blocked" in blocked_output
    assert "blockers=blocked" in blocked_output


def test_vast_adapter_remaining_small_helper_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert vpa._redact_runtime_value(["token=abc"], ["abc"]) == [f"token={vpa.REDACTED_SECRET}"]
    assert vpa._provider_url_public_blocker("https://8.8.8.8/bundle.zip", "bundle") is None

    class HeaderSocket:
        def __init__(self, chunks: list[bytes]) -> None:
            self.chunks = chunks
            self.closed = False
            self.sent = b""

        def recv(self, _size: int) -> bytes:
            return self.chunks.pop(0) if self.chunks else b""

        def settimeout(self, _timeout: int) -> None:
            return None

        def sendall(self, data: bytes) -> None:
            self.sent += data

        def close(self) -> None:
            self.closed = True

    assert vpa._read_http_headers_from_socket(HeaderSocket([b""])) == (None, {})
    assert vpa._read_http_headers_from_socket(
        HeaderSocket([b"HTTP/1.1 nope\r\nX-Test: yes\r\n\r\n"])
    ) == (None, {"X-Test": "yes"})

    monkeypatch.setattr(vpa, "_resolve_public_dns_a_records", lambda *_, **__: [])
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)
    assert vpa._head_with_public_dns_fallback("https://example.com/file", timeout_seconds=1)[
        "blockers"
    ] == ["provider_bundle_fetch_url_public_dns_fallback_failed"]

    raw_sock = HeaderSocket([b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n"])
    monkeypatch.setattr(vpa, "_resolve_public_dns_a_records", lambda *_, **__: ["8.8.8.8"])
    monkeypatch.setattr(vpa.socket, "create_connection", lambda *_, **__: raw_sock)
    http_head = vpa._head_with_public_dns_fallback("http://example.com/file", timeout_seconds=1)
    assert http_head["status"] == "passed"
    assert raw_sock.closed is True

    with pytest.raises(ValueError, match="unsupported_provider_bundle_kind"):
        vpa._isaac_image_startup_preflight(
            job_dir=tmp_path / "bad-image-kind",
            generated_at="2026-06-20T00:00:00Z",
            enable_isaac_smoke=True,
            enable_blueprint_bundle=False,
            provider_bundle_kind="bad",
            selected_container_image=DEFAULT_ISAAC_IMAGE,
            vast_template_hash_id=None,
            use_vast_template_image=False,
            max_live_minutes=5,
            allow_cold_isaac_image_pull=True,
            min_cold_isaac_pull_live_minutes=10,
        )
    missing_template = vpa._isaac_image_startup_preflight(
        job_dir=tmp_path / "missing-template",
        generated_at="2026-06-20T00:00:00Z",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=False,
        provider_bundle_kind="isaac",
        selected_container_image=DEFAULT_ISAAC_IMAGE,
        vast_template_hash_id=None,
        use_vast_template_image=True,
        max_live_minutes=5,
        allow_cold_isaac_image_pull=True,
        min_cold_isaac_pull_live_minutes=10,
    )
    assert "vast_template_hash_required_when_using_template_image" in missing_template["blockers"]
    assert (
        "vast_template_image_cache_not_proven_for_short_live_window" in missing_template["blockers"]
    )
    short_pull = vpa._isaac_image_startup_preflight(
        job_dir=tmp_path / "short-pull",
        generated_at="2026-06-20T00:00:00Z",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=False,
        provider_bundle_kind="isaac",
        selected_container_image=DEFAULT_ISAAC_IMAGE,
        vast_template_hash_id=None,
        use_vast_template_image=False,
        max_live_minutes=5,
        allow_cold_isaac_image_pull=True,
        min_cold_isaac_pull_live_minutes=10,
    )
    assert "cold_official_isaac_image_pull_live_window_too_short" in short_pull["blockers"]

    with pytest.raises(ValueError, match="unsupported_provider_bundle_kind"):
        vpa._resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="bad",
        )
    with pytest.raises(ValueError, match="unsupported_provider_bundle_kind"):
        vpa._probe_shell_script(
            "https://heartbeat.example",
            enable_blueprint_bundle=True,
            provider_bundle_kind="bad",
        )

    monkeypatch.setenv(vpa.VAST_FORWARD_SECRET_ENV_VARS_ENV, "SAFE_NAME,API_TOKEN,OTHER_SECRET")
    monkeypatch.setenv("API_TOKEN", "token-secret")
    monkeypatch.setenv("OTHER_SECRET", "other-secret")
    env = vpa._probe_env(job_dir=tmp_path / "env", enable_isaac_smoke=False)
    assert env["API_TOKEN"] == "token-secret"
    assert env["OTHER_SECRET"] == "other-secret"
    assert vpa._forwarded_secret_values() == ["token-secret", "other-secret"]

    monkeypatch.delenv(vpa.VAST_FORWARD_SECRET_ENV_VARS_ENV, raising=False)
    hf_token_file = tmp_path / "hf_token"
    _write_secret(hf_token_file, "hf-secret-token")
    monkeypatch.setenv(vpa.HF_TOKEN_FILE_ENV, str(hf_token_file))
    hf_env = vpa._probe_env(job_dir=tmp_path / "hf-env", enable_isaac_smoke=False)
    assert hf_env["HF_TOKEN"] == "hf-secret-token"
    assert hf_env["HUGGING_FACE_HUB_TOKEN"] == "hf-secret-token"
    assert hf_env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert vpa._forwarded_secret_values() == ["hf-secret-token"]
    summary = vpa._create_request_summary(
        {"env": hf_env},
        secret_values=vpa._forwarded_secret_values(),
    )
    assert summary["raw_payload_redacted"]["env"]["HF_TOKEN"] == vpa.REDACTED_SECRET_FIELD
    assert (
        summary["raw_payload_redacted"]["env"]["HUGGING_FACE_HUB_TOKEN"]
        == vpa.REDACTED_SECRET_FIELD
    )

    public_env = vpa._probe_env(
        job_dir=tmp_path / "public-model-env",
        enable_isaac_smoke=False,
        forward_hf_token=False,
    )
    assert "HF_TOKEN" not in public_env
    assert "HUGGING_FACE_HUB_TOKEN" not in public_env

    with pytest.raises(ValueError, match="unsupported_provider_bundle_kind"):
        run_vast_provider_adapter(job_dir=tmp_path / "bad-kind", provider_bundle_kind="bad")
    assert vpa._release_vast_launch_lock(None) is None


def test_vast_adapter_bundle_ffprobe_instance_and_validation_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_bundle = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "missing-bundle",
        generated_at="2026-06-20T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="isaac",
        bundle_path=tmp_path / "missing.zip",
        provider_bundle_url="https://example.invalid/bundle.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert "isaac_provider_runtime_bundle_missing" in missing_bundle["blockers"]

    bad_zip = tmp_path / "bad-runtime.zip"
    bad_zip.write_text("not a zip", encoding="utf-8")
    bad_zip_preflight = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "bad-zip",
        generated_at="2026-06-20T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="isaac",
        bundle_path=bad_zip,
        provider_bundle_url="https://example.invalid/bundle.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert any(
        blocker.startswith("provider_runtime_bundle_zip_inspection_failed:")
        for blocker in bad_zip_preflight["blockers"]
    )

    incomplete_zip = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(incomplete_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_isaac_realistic_runtime.sh", "echo no fallback")
    incomplete = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "incomplete",
        generated_at="2026-06-20T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="isaac",
        bundle_path=incomplete_zip,
        provider_bundle_url="https://example.invalid/bundle.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert "provider_runtime_bundle_required_entries_missing" in incomplete["blockers"]
    assert "provider_entrypoint_missing_runtime_result_crash_fallback" in incomplete["blockers"]
    assert "provider_runner_missing_isaac_simulation_app_smoke" in incomplete["blockers"]

    integrity_zip = tmp_path / "integrity.zip"
    _write_valid_provider_bundle(integrity_zip)
    original_zipfile = vpa.zipfile.ZipFile

    class IntegrityZip:
        def __init__(self, *args, **kwargs):
            self.inner = original_zipfile(*args, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def namelist(self):
            return self.inner.namelist()

        def read(self, name):
            return self.inner.read(name)

        def testzip(self):
            return "provider_runtime/run_isaac_realistic_runtime.sh"

    monkeypatch.setattr(vpa.zipfile, "ZipFile", IntegrityZip)
    integrity = vpa._blueprint_bundle_preflight(
        job_dir=tmp_path / "integrity",
        generated_at="2026-06-20T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="isaac",
        bundle_path=integrity_zip,
        provider_bundle_url="https://example.invalid/bundle.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert "provider_runtime_bundle_zip_integrity_failed" in integrity["blockers"]
    monkeypatch.setattr(vpa.zipfile, "ZipFile", original_zipfile)

    assert vpa._instance_list_rows({"instances": {"a": {"status": "running"}, "b": "skip"}}) == [
        {"status": "running"}
    ]
    assert vpa._instance_list_rows({"instances": {"status": "running"}}) == [{"status": "running"}]
    assert vpa._instance_list_rows({"results": [{"status": "running"}, "skip"]}) == [
        {"status": "running"}
    ]
    assert vpa._instance_list_rows({"data": {"one": {"status": "running"}}}) == [
        {"status": "running"}
    ]
    assert vpa._instance_list_rows({"status": "running"}) == [{"status": "running"}]
    assert vpa._instance_list_rows({"other": True}) == []

    monkeypatch.setattr(
        vpa,
        "_api_json",
        lambda **_: (_ for _ in ()).throw(RuntimeError("inventory down")),
    )
    inventory = vpa._prelaunch_inventory_guard(
        job_dir=tmp_path / "inventory",
        generated_at="2026-06-20T00:00:00Z",
        api_key="secret",
    )
    assert inventory["blockers"] == ["vast_prelaunch_inventory_query_failed"]

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(vpa.shutil, "which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        vpa.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ffprobe crashed")),
    )
    assert vpa._ffprobe_video(video)["blockers"] == ["ffprobe_failed:RuntimeError"]
    monkeypatch.setattr(
        vpa.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=2,
            stdout="{bad json",
            stderr="bad",
        ),
    )
    blocked_probe = vpa._ffprobe_video(video)
    assert {
        "ffprobe_returncode:2",
        "ffprobe_json_parse_failed",
        "mp4_duration_not_positive",
        "mp4_frame_count_not_positive",
    }.issubset(set(blocked_probe["blockers"]))

    mp4_zip = tmp_path / "mp4s.zip"
    with zipfile.ZipFile(mp4_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("camera.mp4", b"fake")
    monkeypatch.setattr(
        vpa, "_ffprobe_video", lambda path: {"status": "blocked", "blockers": ["bad"]}
    )
    invalid_mp4 = vpa._inspect_provider_runtime_output_zip(
        mp4_zip,
        video_extract_dir=tmp_path / "extract",
        expected_video_count=2,
    )
    assert (
        "ffprobe_validation_failed_for_one_or_more_mp4s"
        in invalid_mp4["mp4_validation"]["blockers"]
    )
    no_extract = vpa._inspect_provider_runtime_output_zip(mp4_zip)
    assert no_extract["mp4_validation"]["blockers"] == ["mp4_ffprobe_validation_not_requested"]

    stale_video = tmp_path / "blocked-video" / "vast_video_smoke_result.json"
    stale_video.parent.mkdir()
    stale_video.write_text("{bad json", encoding="utf-8")
    vpa._write_blocked_phase_artifacts(
        job_dir=stale_video.parent,
        generated_at="2026-06-20T00:00:00Z",
        provider_reason="blocked",
    )
    assert _read_json(stale_video)["status"] == "blocked"

    validation_dir = tmp_path / "video-validation"
    validation_dir.mkdir()
    for name in (
        "vast_runtime_discovery.json",
        "vast_provider_plan.json",
        "vast_offer_selection_manifest.json",
        "vast_budget_ledger.json",
        "vast_startup_probe_manifest.json",
        "vast_gpu_sanity_report.json",
        "vast_isaac_smoke_result.json",
        "vast_provider_command_result.json",
        "vast_teardown_manifest.json",
    ):
        (validation_dir / name).write_text("{}", encoding="utf-8")
    (validation_dir / "vast_runtime_phase_log.jsonl").write_text(
        "\n".join(json.dumps({"phase": phase}) for phase in vpa.VAST_REQUIRED_PHASES) + "\n",
        encoding="utf-8",
    )
    (validation_dir / "vast_video_smoke_result.json").write_text("{bad json", encoding="utf-8")
    validation = vpa._final_validation(
        job_dir=validation_dir,
        generated_at="2026-06-20T00:00:00Z",
        instance_ids=[],
        continuing_spend=False,
        estimated_cost_usd=0.0,
        hard_cap_usd=1.0,
    )
    assert "json_parse_errors" in validation["blockers"]


def test_vast_adapter_run_preflight_and_wam_live_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setattr(vpa.time, "sleep", lambda *_args, **_kwargs: None)

    def fail_if_called(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected Vast API call: {kwargs}")

    monkeypatch.setattr(vpa, "_api_json", fail_if_called)
    image_blocked = run_vast_provider_adapter(
        job_dir=tmp_path / "image-blocked",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        enable_isaac_smoke=True,
        allow_cold_isaac_image_pull=False,
        session_max_live_minutes=None,
    )
    assert image_blocked["reason"] == "vast_isaac_image_startup_preflight_blocked"
    assert image_blocked["api_call_performed"] is False

    secret = _configure_live_gates(tmp_path, monkeypatch)
    wam_bundle = tmp_path / "wam.zip"
    _write_valid_wam_provider_bundle(wam_bundle)
    mutation_order: list[str] = []
    drop_bundle_after_preflight = {"enabled": True}

    def fake_api_json(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["api_key"] == secret
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/":
            return 200, {"instances": []}
        if kwargs["method"] == "POST":
            mutation_order.append("offer_search")
            return 200, {
                "offers": [
                    {
                        "id": 808,
                        "ask_contract_id": 808,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.2,
                        "driver_version": "580.95.05",
                    }
                ]
            }
        if kwargs["method"] == "PUT" and kwargs["path"] == "/asks/808/":
            mutation_order.append("provider_create")
            return 200, {"new_contract": 8081}
        if kwargs["method"] == "GET" and kwargs["path"] == "/instances/8081/":
            return 200, {"instances": {"actual_status": "running", "cur_state": "running"}}
        if kwargs["method"] == "PUT" and kwargs["path"] == "/instances/request_logs/8081":
            return 200, {"success": True, "result_url": "https://logs.example/wam"}
        if kwargs["method"] == "DELETE":
            mutation_order.append("provider_delete")
            return 200, {"success": True}
        raise AssertionError(kwargs)

    def consume_authorization() -> dict[str, object]:
        mutation_order.append("authorization_consumed")
        if drop_bundle_after_preflight["enabled"]:
            wam_bundle.unlink()
        return {"status": "consumed", "blockers": []}

    original_write_json = vpa.write_json

    def reject_write_between_consumption_and_create(*args, **kwargs):  # type: ignore[no-untyped-def]
        if "authorization_consumed" in mutation_order and "provider_create" not in mutation_order:
            raise AssertionError("fallible evidence write occurred after authorization consumption")
        return original_write_json(*args, **kwargs)

    monkeypatch.setattr(vpa, "write_json", reject_write_between_consumption_and_create)

    monkeypatch.setattr(vpa, "_api_json", fake_api_json)
    monkeypatch.setattr(
        vpa,
        "_fetch_text",
        lambda *_args, **_kwargs: (
            "BLUEPRINT_VAST_HEARTBEAT_OK\n"
            "RTX 4090, 580.95.05, 24564 MiB\n"
            "BLUEPRINT_VAST_GPU_SANITY_OK\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED\n"
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED\n"
            "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK\n"
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED\n"
            "BLUEPRINT_VAST_ONSTART_DONE\n"
        ),
    )
    wam = run_vast_provider_adapter(
        job_dir=tmp_path / "wam-live",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle=wam_bundle,
        provider_bundle_url="https://example.invalid/wam.zip",
        provider_output_put_url="https://example.invalid/out.zip",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="wam",
        pre_provider_mutation_hook=consume_authorization,
        poll_interval_seconds=0,
        startup_timeout_seconds=10,
        session_max_live_minutes=None,
    )
    assert wam["status"] == "blocked"
    assert wam["reason"] == "vast_blueprint_video_smoke_blocked"
    assert mutation_order.index("offer_search") < mutation_order.index("authorization_consumed")
    assert mutation_order.index("authorization_consumed") < mutation_order.index("provider_create")
    assert wam["pre_provider_mutation_hook_result"]["status"] == "consumed"
    assert (
        "provider_runtime_bundle_missing"
        not in _read_json(tmp_path / "wam-live" / "vast_provider_command_result.json")["blockers"]
    )
    assert (
        _read_json(tmp_path / "wam-live" / "vast_isaac_smoke_result.json")["status"]
        == "not_required"
    )

    monkeypatch.setattr(
        vpa,
        "record_terminal_lifecycle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("journal write failed")),
    )
    _write_valid_wam_provider_bundle(wam_bundle)
    drop_bundle_after_preflight["enabled"] = False
    delete_count_before = mutation_order.count("provider_delete")
    lifecycle_failed = run_vast_provider_adapter(
        job_dir=tmp_path / "wam-lifecycle-failed",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        provider_bundle=wam_bundle,
        provider_bundle_url="https://example.invalid/wam.zip",
        provider_output_put_url="https://example.invalid/out.zip",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="wam",
        retain_instance_on_runtime_failure=True,
        pre_provider_mutation_hook=consume_authorization,
        poll_interval_seconds=0,
        startup_timeout_seconds=10,
        session_max_live_minutes=None,
    )
    assert lifecycle_failed["status"] == "failed"
    assert lifecycle_failed["reason"] == "retained_gpu_lifecycle_record_failed"
    assert mutation_order.count("provider_delete") == delete_count_before + 1
    lifecycle_teardown = _read_json(
        tmp_path / "wam-lifecycle-failed" / "vast_teardown_manifest.json"
    )
    assert lifecycle_teardown["continuing_spend_from_this_run"] is False
    assert lifecycle_teardown["runner_gpu_teardown_completed"] is True

    monkeypatch.setenv("NGC_API_KEY_FILE", str(tmp_path / "missing-ngc-key"))
    ngc_missing = run_vast_provider_adapter(
        job_dir=tmp_path / "ngc-missing",
        mode="live-startup-probe",
        paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True,
        allow_instance_launch=True,
        enable_isaac_smoke=True,
        enable_blueprint_bundle=False,
        ngc_image_login_mode="always",
        poll_interval_seconds=0,
        startup_timeout_seconds=10,
        session_max_live_minutes=None,
    )
    assert ngc_missing["status"] == "completed"
    isaac = _read_json(tmp_path / "ngc-missing" / "vast_isaac_smoke_result.json")
    assert "ngc_api_key_file_missing_or_empty_for_required_ngc_login" in isaac["blockers"]


def _disk_offer(machine_id: int, disk_space, rate: float = 0.5) -> dict:
    return {
        "ask_contract_id": machine_id,
        "id": machine_id,
        "machine_id": machine_id,
        "dph_total": rate,
        "disk_space": disk_space,
        "gpu_name": "RTX 4090",
        "gpu_ram": 24576,
        "num_gpus": 1,
        "cuda_max_good": 12.4,
        "driver_version": "550.90.07",
        "reliability2": 0.99,
        "rentable": True,
        "has_avx": True,
        "direct_port_count": 4,
    }


def test_offer_summary_records_disk_space():
    """Without it, an image-pull failure cannot be attributed after the fact.

    The Arena image is 9.4 GB compressed and Arena's pip install adds several
    more. Five container_missing failures on this lane could not be checked
    against host disk because the retained offer evidence never had the field.
    """

    summary = _offer_summary(_disk_offer(4242, 320))

    assert summary["disk_space_gb"] == 320


def test_offer_summary_tolerates_a_missing_disk_field():
    summary = _offer_summary(_disk_offer(4242, None))

    assert summary["disk_space_gb"] is None


def test_min_disk_space_excludes_hosts_too_small_for_the_image():
    selected = _select_offer(
        [_disk_offer(1, 40, rate=0.10), _disk_offer(2, 400, rate=0.90)],
        max_hourly_rate=1.0,
        min_disk_space_gb=120,
    )

    assert selected is not None
    assert selected["machine_id"] == 2


def test_min_disk_space_excludes_a_host_that_does_not_report_disk():
    """An unreported size is not a small size, but it is not a large one either."""

    selected = _select_offer(
        [_disk_offer(1, None, rate=0.10)],
        max_hourly_rate=1.0,
        min_disk_space_gb=120,
    )

    assert selected is None


def test_min_disk_space_defaults_to_no_filter():
    selected = _select_offer(
        [_disk_offer(1, None, rate=0.10)],
        max_hourly_rate=1.0,
    )

    assert selected is not None
