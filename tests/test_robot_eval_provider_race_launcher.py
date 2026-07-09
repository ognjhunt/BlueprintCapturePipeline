from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from blueprint_pipeline.robot_eval_provider_race_launcher import (
    ALLOW_PROVIDER_RACE_LAUNCH_ENV,
    main as provider_race_launcher_main,
    run_robot_eval_provider_race_launcher,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _python_command(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _launch_request(path: Path, *, can_launch: bool = True) -> Path:
    _write_json(
        path,
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "job_id": "race-job-1",
            "provider": "runpod",
            "status": "request_manifest_ready",
            "live_provider_calls_performed": False,
            "prelaunch_spend_guard": {
                "schema_version": "robot_eval_provider_prelaunch_spend_guard.v1",
                "required_before_provider_launch": True,
                "can_launch": can_launch,
                "blockers": [] if can_launch else ["prelaunch_local_sim_only_prerequisite_not_passed"],
                "provider_race": {
                    "schema_version": "robot_eval_provider_race_contract.v1",
                    "race_module": "blueprint_pipeline.provider_race",
                    "race_required_for_customer_path": True,
                    "customer_path_provider_failover_runtime_wired": can_launch,
                    "customer_path_provider_failover_runtime_status": (
                        "runtime_ready"
                        if can_launch
                        else "blocked_pending_teardown_owned_race_launcher"
                    ),
                    "provider_race_handoff_path": "gpu_provider_race_handoff.json",
                    "launcher_contract": {
                        "provider_race_launcher_available": True,
                        "provider_race_launcher_command": (
                            "blueprint-run-robot-eval-provider-race"
                        ),
                    },
                },
            },
        },
    )
    return path


def _handoff(path: Path, *, ready: bool = True) -> Path:
    first_marker = path.parent / "runpod-attempt"
    second_marker = path.parent / "vast-attempt"
    _write_json(
        path,
        {
            "schema_version": "robot_eval_gpu_provider_race_handoff.v1",
            "generated_at": "2026-07-04T00:00:00Z",
            "job_id": "race-job-1",
            "status": "ready_for_customer_provider_race_runtime"
            if ready
            else "blocked_before_provider_race_launcher",
            "reason": "provider_race_handoff_ready"
            if ready
            else "provider_race_handoff_blocked",
            "blockers": []
            if ready
            else ["prelaunch_local_sim_only_prerequisite_not_passed"],
            "provider_launch_request_path": "gpu_provider_launch_request.json",
            "provider_race_required_for_customer_path": True,
            "customer_path_provider_failover_handoff_wired": True,
            "customer_path_provider_failover_runtime_wired": ready,
            "customer_path_provider_failover_runtime_status": (
                "runtime_ready"
                if ready
                else "blocked_pending_teardown_owned_race_launcher"
            ),
            "customer_path_provider_failover_runtime_blockers": []
            if ready
            else ["runpod_provider_race_teardown_owned_allocation_contract_missing"],
            "provider_race_runtime_launcher_available": True,
            "provider_race_runtime_launcher_blockers": [],
            "provider_race_launcher_result_path": (
                "gpu_provider_race_launcher_result.json"
            ),
            "launcher_command": (
                "blueprint-run-robot-eval-provider-race "
                "--provider-launch-request gpu_provider_launch_request.json "
                "--handoff gpu_provider_race_handoff.json"
            ),
            "live_provider_calls_performed": False,
            "race_candidate_count": 2,
            "runnable_candidate_count": 2,
            "runnable_candidates": [
                {
                    "provider": "runpod",
                    "operation": "enqueue_runpod_serverless_or_on_demand_worker",
                    "adapter_command": _python_command(
                        f"from pathlib import Path; Path({str(first_marker)!r}).write_text('failed'); raise SystemExit(7)"
                    ),
                    "selected": True,
                    "launch_request_path": "gpu_provider_launch_request.json",
                    "adapter_result_path": "runpod_provider_adapter_result.json",
                },
                {
                    "provider": "vast",
                    "operation": "create_vast_instance_and_run_worker",
                    "adapter_command": _python_command(
                        f"from pathlib import Path; Path({str(second_marker)!r}).write_text('won')"
                    ),
                    "selected": False,
                    "launch_request_path": "gpu_provider_launch_request.json",
                    "adapter_result_path": "vast_provider_adapter_result.json",
                },
            ],
            "claim_boundary": {
                "provider_race_handoff_is_not_customer_runtime_failover": not ready,
                "live_provider_calls_performed": False,
            },
        },
    )
    return path


def test_provider_race_launcher_validates_ready_handoff_without_live_calls(
    tmp_path: Path,
) -> None:
    request_path = _launch_request(tmp_path / "gpu_provider_launch_request.json")
    handoff_path = _handoff(tmp_path / "gpu_provider_race_handoff.json")

    result = run_robot_eval_provider_race_launcher(
        provider_launch_request_path=request_path,
        handoff_path=handoff_path,
    )

    assert result["status"] == "ready_for_live_provider_race"
    assert result["blockers"] == []
    assert result["provider_race_launcher_available"] is True
    assert result["provider_race_runtime_launcher_available"] is True
    assert result["live_provider_calls_performed"] is False
    assert result["provider_race_execution_proven"] is False
    assert result["claim_boundary"][
        "provider_race_launcher_result_is_not_provider_execution"
    ] is True
    persisted = _read_json(tmp_path / "gpu_provider_race_launcher_result.json")
    assert persisted["status"] == "ready_for_live_provider_race"


def test_provider_race_launcher_executes_live_gated_serial_failover(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _launch_request(tmp_path / "gpu_provider_launch_request.json")
    handoff_path = _handoff(tmp_path / "gpu_provider_race_handoff.json")
    monkeypatch.setenv(ALLOW_PROVIDER_RACE_LAUNCH_ENV, "true")

    result = run_robot_eval_provider_race_launcher(
        provider_launch_request_path=request_path,
        handoff_path=handoff_path,
        allow_live_provider_race=True,
        timeout_seconds=5,
    )

    assert result["status"] == "completed"
    assert result["provider_race_execution_performed"] is True
    assert result["provider_race_execution_proven"] is True
    assert result["live_provider_calls_performed"] is True
    assert result["winning_provider"] == "vast"
    assert [attempt["provider"] for attempt in result["failover_attempts"]] == [
        "runpod",
        "vast",
    ]
    assert result["failover_attempts"][0]["status"] == "failed"
    assert result["failover_attempts"][1]["status"] == "completed"
    assert (tmp_path / "runpod-attempt").read_text(encoding="utf-8") == "failed"
    assert (tmp_path / "vast-attempt").read_text(encoding="utf-8") == "won"
    assert result["remote_cloud_execution_proven"] is False
    persisted = _read_json(tmp_path / "gpu_provider_race_launcher_result.json")
    assert persisted["status"] == "completed"
    assert persisted["winning_provider"] == "vast"


def test_provider_race_launcher_blocks_on_blocked_handoff(tmp_path: Path) -> None:
    request_path = _launch_request(
        tmp_path / "gpu_provider_launch_request.json",
        can_launch=False,
    )
    handoff_path = _handoff(tmp_path / "gpu_provider_race_handoff.json", ready=False)

    result = run_robot_eval_provider_race_launcher(
        provider_launch_request_path=request_path,
        handoff_path=handoff_path,
    )

    assert result["status"] == "blocked"
    assert "provider_race_handoff_not_ready" in result["blockers"]
    assert "customer_path_provider_failover_runtime_not_wired" in result["blockers"]
    assert "prelaunch_local_sim_only_prerequisite_not_passed" in result["blockers"]
    assert result["live_provider_calls_performed"] is False


def test_provider_race_launcher_blocks_nested_runtime_blockers(
    tmp_path: Path,
) -> None:
    request_path = _launch_request(tmp_path / "gpu_provider_launch_request.json")
    handoff_path = _handoff(tmp_path / "gpu_provider_race_handoff.json")
    handoff = _read_json(handoff_path)
    handoff["customer_path_provider_failover_runtime_blockers"] = [
        "teardown_owned_loser_cleanup_not_proven"
    ]
    handoff["provider_race_runtime_launcher_blockers"] = [
        "provider_race_runtime_launcher_contract_not_signed"
    ]
    _write_json(handoff_path, handoff)

    result = run_robot_eval_provider_race_launcher(
        provider_launch_request_path=request_path,
        handoff_path=handoff_path,
    )

    assert result["status"] == "blocked"
    assert "teardown_owned_loser_cleanup_not_proven" in result["blockers"]
    assert "provider_race_runtime_launcher_contract_not_signed" in result["blockers"]
    assert result["live_provider_calls_performed"] is False


def test_provider_race_launcher_cli_exits_zero_for_ready_handoff(
    tmp_path: Path,
    capsys,
) -> None:
    request_path = _launch_request(tmp_path / "gpu_provider_launch_request.json")
    handoff_path = _handoff(tmp_path / "gpu_provider_race_handoff.json")

    exit_code = provider_race_launcher_main(
        [
            "--provider-launch-request",
            str(request_path),
            "--handoff",
            str(handoff_path),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "status=ready_for_live_provider_race" in captured.out
