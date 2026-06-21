from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import robot_eval_provider_launcher as launcher
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


def test_provider_launcher_helper_edges() -> None:
    assert launcher._env_value(None) == ""
    assert launcher._env_value(" value ") == "value"
    assert launcher._string_list("one") == ["one"]
    assert launcher._string_list(123) == []
    assert launcher._number(True) is None
    assert launcher._number(7) == 7.0
    assert launcher._number("1.5") == 1.5
    assert launcher._number("not-a-number") is None
    assert launcher._number(object()) is None
    assert launcher._output_text(None) == ""
    assert "\ufffd" in launcher._output_text(b"\xff")


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


def test_provider_launcher_blocks_invalid_request_payloads(tmp_path: Path) -> None:
    list_request = tmp_path / "list_request.json"
    list_request.write_text("[]", encoding="utf-8")
    list_result = run_gpu_provider_launcher(provider_launch_request_path=list_request)

    assert list_result["status"] == "blocked"
    assert list_result["reason"] == "invalid_provider_launch_request"
    assert list_result["blockers"] == ["invalid_provider_launch_request_json"]

    bad_schema_request = tmp_path / "bad_schema_request.json"
    _write_json(bad_schema_request, {"schema_version": "wrong"})
    schema_result = run_gpu_provider_launcher(provider_launch_request_path=bad_schema_request)

    assert schema_result["status"] == "blocked"
    assert schema_result["reason"] == "invalid_provider_launch_request_schema"
    assert schema_result["blockers"] == ["invalid_provider_launch_request_schema"]


def test_provider_launcher_skips_fixture_local_requests(tmp_path: Path) -> None:
    request_path = tmp_path / "gpu_provider_launch_request.json"
    _write_json(
        request_path,
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "job_id": "fixture-job",
            "provider": "fixture_local",
            "status": "request_manifest_ready",
        },
    )

    result = run_gpu_provider_launcher(provider_launch_request_path=request_path)

    assert result["status"] == "not_required_for_fixture_local"
    assert result["execution_performed"] is False
    assert result["blockers"] == []


def test_provider_launcher_reports_request_shape_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "gpu_provider_launch_request.json"
    _write_json(
        request_path,
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "job_id": "bad-shape",
            "provider": "runpod",
            "status": "request_manifest_ready",
            "provider_request_shape": {
                "api_payload_is_provider_adapter_template": False,
                "environment": {"secret_values_in_artifact": True},
                "image": {"owner_published_image_ref_required": True},
                "inputs": {
                    "manifest_uri_required_for_provider": True,
                    "manifest_uri_fetchable_by_provider": False,
                    "artifact_output_uri_required": True,
                },
            },
        },
    )
    monkeypatch.setenv(ALLOW_PROVIDER_LAUNCH_ENV, "true")

    result = run_gpu_provider_launcher(
        provider_launch_request_path=request_path,
        allow_provider_launch=True,
        provider_launch_command=_python_command("print('blocked before execution')"),
    )

    assert result["status"] == "blocked"
    assert "provider_launch_request_not_adapter_template" in result["blockers"]
    assert "provider_launch_request_secret_values_in_artifact" in result["blockers"]
    assert "missing_provider_worker_image_ref" in result["blockers"]
    assert "missing_provider_worker_manifest_uri" in result["blockers"]
    assert "provider_worker_manifest_uri_not_fetchable" in result["blockers"]
    assert "missing_provider_artifact_output_uri" in result["blockers"]


def test_provider_launcher_blocks_invalid_and_empty_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _ready_provider_launch_request(tmp_path / "gpu_provider_launch_request.json")
    monkeypatch.setenv(ALLOW_PROVIDER_LAUNCH_ENV, "true")

    invalid = run_gpu_provider_launcher(
        provider_launch_request_path=request_path,
        allow_provider_launch=True,
        provider_launch_command="'unterminated",
    )
    assert invalid["status"] == "blocked"
    assert invalid["reason"] == "invalid_gpu_provider_launch_command"
    assert "command_parse_error" in invalid

    monkeypatch.setattr(launcher.shlex, "split", lambda _command: [])
    empty = run_gpu_provider_launcher(
        provider_launch_request_path=request_path,
        allow_provider_launch=True,
        provider_launch_command="syntactically-present",
    )
    assert empty["status"] == "blocked"
    assert empty["reason"] == "missing_gpu_provider_launch_command"


def test_provider_launcher_records_command_not_found_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _ready_provider_launch_request(tmp_path / "gpu_provider_launch_request.json")
    monkeypatch.setenv(ALLOW_PROVIDER_LAUNCH_ENV, "true")

    missing = run_gpu_provider_launcher(
        provider_launch_request_path=request_path,
        allow_provider_launch=True,
        provider_launch_command=str(tmp_path / "definitely-not-a-real-blueprint-launcher-binary"),
    )
    assert missing["status"] == "blocked"
    assert missing["reason"] == "gpu_provider_launch_command_not_found"

    def timeout_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            cmd=["fake-provider"],
            timeout=3,
            output=b"partial out",
            stderr=b"partial err",
        )

    monkeypatch.setattr(launcher.subprocess, "run", timeout_run)
    timed_out = run_gpu_provider_launcher(
        provider_launch_request_path=request_path,
        output_path=tmp_path / "timeout_result.json",
        allow_provider_launch=True,
        provider_launch_command="fake-provider",
        timeout_seconds=3,
    )

    assert timed_out["status"] == "failed"
    assert timed_out["reason"] == "provider_launcher_command_timeout"
    assert timed_out["execution_performed"] is True
    assert (tmp_path / "gpu_provider_launcher.stdout.log").read_text(encoding="utf-8") == "partial out"
    assert (tmp_path / "gpu_provider_launcher.stderr.log").read_text(encoding="utf-8") == "partial err"


def test_provider_launcher_cli_provider_request_and_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path = tmp_path / "gpu_provider_launch_request.json"
    _write_json(
        request_path,
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "job_id": "fixture-job",
            "provider": "fixture_local",
            "status": "request_manifest_ready",
        },
    )

    assert provider_launcher_main(["--provider-launch-request", str(request_path)]) == 0
    assert "status=not_required_for_fixture_local" in capsys.readouterr().out

    _ready_provider_launch_request(request_path)
    monkeypatch.delenv(ALLOW_PROVIDER_LAUNCH_ENV, raising=False)
    exit_code = provider_launcher_main(["--provider-launch-request", str(request_path)])
    blocked_output = capsys.readouterr().out
    assert exit_code == 1
    assert "status=blocked" in blocked_output
    assert "blockers=" in blocked_output

    with pytest.raises(SystemExit):
        provider_launcher_main([])
