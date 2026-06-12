from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from blueprint_pipeline.robot_eval_provider_launcher import (
    ALLOW_PROVIDER_LAUNCH_ENV,
    PROVIDER_LAUNCH_COMMAND_ENV,
    main as provider_launcher_main,
    run_gpu_provider_launcher,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ready_provider_launch_request(
    path: Path,
    *,
    status: str = "request_manifest_ready",
) -> Path:
    _write_json(
        path,
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "job_id": "launcher-job-1",
            "provider": "runpod",
            "status": status,
            "live_provider_calls_performed": False,
            "provider_request_shape": {
                "provider_api": "runpod",
                "api_payload_is_provider_adapter_template": True,
                "api_payload_values_are_redacted": True,
                "operation": "enqueue_runpod_serverless_or_on_demand_worker",
                "image": {
                    "owner_published_image_ref_required": True,
                    "configured_image_ref": (
                        "registry.example/blueprint/isaac-eval-worker:2026-06-12"
                    ),
                    "configured_image_ref_present": True,
                    "configured_image_ref_is_versioned": True,
                },
                "command": (
                    "blueprint-run-robot-eval-worker "
                    "--manifest ${BLUEPRINT_EVAL_MANIFEST_URI}"
                ),
                "environment": {
                    "secret_env_var_names": ["RUNPOD_API_KEY"],
                    "secret_values_in_artifact": False,
                },
                "inputs": {
                    "manifest_uri_required_for_provider": True,
                    "manifest_uri": (
                        "r2://blueprint-artifacts/jobs/launcher-job-1/"
                        "worker_manifest.json"
                    ),
                    "manifest_uri_fetchable_by_provider": True,
                    "artifact_output_uri_required": True,
                    "artifact_output_uri": (
                        "r2://blueprint-artifacts/jobs/launcher-job-1"
                    ),
                },
                "limits": {
                    "max_active_workers": 1,
                    "hard_timeout_seconds": 120,
                    "idle_timeout_seconds": 60,
                    "external_watchdog_ttl_seconds": 180,
                },
            },
        },
    )
    return path


def _python_command(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def test_provider_launcher_blocks_when_launch_request_is_not_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_provider_launch_request(
        tmp_path / "gpu_provider_launch_request.json",
        status="blocked_by_scheduler",
    )
    monkeypatch.setenv(ALLOW_PROVIDER_LAUNCH_ENV, "true")

    result = run_gpu_provider_launcher(
        provider_launch_request_path=request_path,
        allow_provider_launch=True,
        provider_launch_command=_python_command("print('should not run')"),
    )

    assert result["status"] == "blocked"
    assert result["execution_performed"] is False
    assert "provider_launch_request_not_ready" in result["blockers"]
    assert not (tmp_path / "gpu_provider_launcher.stdout.log").exists()


def test_provider_launcher_blocks_without_second_gate_or_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_provider_launch_request(
        tmp_path / "gpu_provider_launch_request.json",
    )
    monkeypatch.delenv(ALLOW_PROVIDER_LAUNCH_ENV, raising=False)
    monkeypatch.delenv(PROVIDER_LAUNCH_COMMAND_ENV, raising=False)

    result = run_gpu_provider_launcher(
        provider_launch_request_path=request_path,
        allow_provider_launch=False,
    )

    assert result["status"] == "blocked"
    assert result["execution_performed"] is False
    assert f"missing_env_{ALLOW_PROVIDER_LAUNCH_ENV}" in result["blockers"]
    assert "missing_cli_allow_provider_launch" in result["blockers"]
    assert "missing_gpu_provider_launch_command" in result["blockers"]
    assert result["secret_values_in_artifact"] is False
    assert result["simulator_execution_proven"] is False


def test_provider_launcher_executes_operator_command_with_redacted_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_provider_launch_request(
        tmp_path / "gpu_provider_launch_request.json",
    )
    monkeypatch.setenv(ALLOW_PROVIDER_LAUNCH_ENV, "true")
    monkeypatch.setenv("RUNPOD_API_KEY", "secret-value-that-must-not-appear")
    code = (
        "import os; "
        "print(os.environ['BLUEPRINT_GPU_PROVIDER']); "
        "print(os.environ['BLUEPRINT_ROBOT_EVAL_JOB_ID']); "
        "print(os.environ['BLUEPRINT_WORKER_IMAGE_REF']); "
        "print(os.environ['BLUEPRINT_GPU_PROVIDER_HARD_TIMEOUT_SECONDS']); "
        "print(os.environ['BLUEPRINT_GPU_PROVIDER_EXTERNAL_WATCHDOG_TTL_SECONDS']); "
        "print(os.environ['RUNPOD_API_KEY'])"
    )

    result = run_gpu_provider_launcher(
        provider_launch_request_path=request_path,
        allow_provider_launch=True,
        provider_launch_command=_python_command(code),
    )

    output_path = tmp_path / "gpu_provider_launcher_result.json"
    persisted = output_path.read_text(encoding="utf-8")
    stdout = (tmp_path / "gpu_provider_launcher.stdout.log").read_text(
        encoding="utf-8"
    )
    assert result["status"] == "completed"
    assert result["execution_performed"] is True
    assert result["provider_launcher_command_executed"] is True
    assert result["provider_side_effects_may_have_occurred"] is True
    assert result["live_provider_calls_performed_by_launcher_module"] is False
    assert result["provider_allocation_proven"] is False
    assert result["simulator_execution_proven"] is False
    assert result["robot_readiness_proven"] is False
    assert result["public_claim_upgrade_allowed"] is False
    assert result["secret_values_in_artifact"] is False
    assert result["stdout_stderr_secret_redaction_enabled"] is True
    assert "RUNPOD_API_KEY" in result["redacted_secret_env_var_names"]
    assert result["redacted_secret_value_count"] >= 1
    assert result["command"]["raw_command_stored"] is False  # type: ignore[index]
    assert "raw_command" not in result
    assert "runpod" in stdout
    assert "launcher-job-1" in stdout
    assert "isaac-eval-worker:2026-06-12" in stdout
    assert "120" in stdout
    assert "180" in stdout
    assert "<redacted:RUNPOD_API_KEY>" in stdout
    assert "secret-value-that-must-not-appear" not in persisted
    assert "secret-value-that-must-not-appear" not in stdout


def test_provider_launcher_records_command_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _ready_provider_launch_request(
        tmp_path / "gpu_provider_launch_request.json",
    )
    monkeypatch.setenv(ALLOW_PROVIDER_LAUNCH_ENV, "true")

    result = run_gpu_provider_launcher(
        provider_launch_request_path=request_path,
        allow_provider_launch=True,
        provider_launch_command=_python_command("import sys; print('bad'); sys.exit(7)"),
    )

    assert result["status"] == "failed"
    assert result["exit_code"] == 7
    assert result["execution_performed"] is True
    assert result["provider_launcher_command_executed"] is True
    assert "gpu_provider_launch_command_failed" in result["blockers"]
    assert (tmp_path / "gpu_provider_launcher.stdout.log").read_text(
        encoding="utf-8"
    ).strip() == "bad"


def test_provider_launcher_cli_uses_job_dir(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    job_dir = tmp_path / "pipeline" / "robot_eval_jobs" / "launcher-job-1"
    _ready_provider_launch_request(job_dir / "gpu_provider_launch_request.json")
    monkeypatch.setenv(ALLOW_PROVIDER_LAUNCH_ENV, "true")
    monkeypatch.setenv(
        PROVIDER_LAUNCH_COMMAND_ENV,
        _python_command("import os; print(os.environ['BLUEPRINT_GPU_PROVIDER'])"),
    )

    exit_code = provider_launcher_main(
        [
            "--job-dir",
            str(job_dir),
            "--allow-provider-launch",
            "--timeout-seconds",
            "5",
        ]
    )

    result = _read_json(job_dir / "gpu_provider_launcher_result.json")
    captured = capsys.readouterr()
    assert exit_code == 0
    assert result["status"] == "completed"
    assert "status=completed" in captured.out
