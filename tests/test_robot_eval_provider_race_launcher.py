from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.robot_eval_provider_race_launcher import (
    main as provider_race_launcher_main,
    race_eval_providers,
    run_robot_eval_provider_race_launcher,
    run_robot_eval_provider_race_runtime,
)


_NO_SLEEP = lambda *_a, **_k: None  # noqa: E731 - deterministic, no wall-clock in tests


class _FakeRaceProvider:
    """Honors only the surface race_launch touches: launch/terminate/inspect/marker."""

    def __init__(self, name: str, *, boots: bool = True, launches: bool = True) -> None:
        self.name = name
        self.boots = boots
        self._launches = launches
        self.launch_calls = 0
        self.terminate_calls: list[str] = []

    def launch(self, job_dir, request, *, cold=False, **_kwargs):  # noqa: ANN001
        self.launch_calls += 1
        assert isinstance(job_dir, Path)
        if not self._launches:
            return {"status": "blocked", "blockers": ["no_capacity"]}
        return {"status": "launched", "instance_id": f"{self.name}-iid", "mode": "fake_cold"}

    def terminate(self, instance_id):  # noqa: ANN001
        self.terminate_calls.append(instance_id)
        return {"status": "terminated", "http": 204, "instance_id": instance_id}

    def inspect(self, instance_id):  # noqa: ANN001
        # After terminate the allocation is gone -> billing-terminal (API 404).
        if instance_id in self.terminate_calls:
            return {"status": "unavailable", "http": 404}
        return {"status": "observed", "http": 200, "desiredStatus": "RUNNING"}

    def has_marker(self, _launch_result):  # noqa: ANN001
        return self.boots


def _marker_check(provider, launch_result):  # noqa: ANN001
    return provider.has_marker(launch_result)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def test_race_eval_providers_selects_first_booter_and_tears_down_loser(
    tmp_path: Path,
) -> None:
    dud = _FakeRaceProvider("runpod", boots=False)
    healthy = _FakeRaceProvider("vast", boots=True)

    race = race_eval_providers(
        providers=[dud, healthy],
        request={"job_id": "race-job-1"},
        marker_check=_marker_check,
        job_dir=tmp_path / "race",
        marker_timeout=1.0,
        poll_interval=10.0,  # ceil(1/10) == 1 attempt -> no sleeping
        sleep=_NO_SLEEP,
    )

    assert race["status"] == "launched"
    assert race["provider"] == "vast"
    assert healthy.launch_calls == 1
    # the dud that launched but never booted is torn down as a loser
    assert dud.terminate_calls == ["runpod-iid"]
    assert "circuit_breaker" in race


def test_provider_race_runtime_failover_selects_across_mocked_providers(
    tmp_path: Path,
) -> None:
    request_path = _launch_request(tmp_path / "gpu_provider_launch_request.json")
    _handoff(tmp_path / "gpu_provider_race_handoff.json")
    dud = _FakeRaceProvider("runpod", boots=False)
    healthy = _FakeRaceProvider("vast", boots=True)

    result = run_robot_eval_provider_race_runtime(
        provider_launch_request_path=request_path,
        providers=[dud, healthy],
        marker_check=_marker_check,
        marker_timeout=1.0,
        poll_interval=10.0,
        sleep=_NO_SLEEP,
    )

    assert result["status"] == "provider_race_executed"
    assert result["provider_race_execution_performed"] is True
    assert result["winner_provider"] == "vast"
    assert result["first_priority_provider"] == "runpod"
    assert result["failover_selected"] is True
    assert result["provider_race_runtime_launcher_implemented"] is True
    # the runtime is implemented now -> this claim is never emitted
    assert "provider_race_runtime_launcher_not_implemented" not in result["blockers"]
    assert (
        result["claim_boundary"]["provider_race_runtime_launcher_not_implemented"]
        is False
    )
    assert result["claim_boundary"][
        "provider_race_execution_is_not_simulator_or_rank_proof"
    ] is True
    assert result["simulator_execution_proven"] is False
    assert result["rank_fidelity_result_proven"] is False
    # no live-cred flag was set, so no live-call claim is made
    assert result["live_provider_calls_performed"] is False
    persisted = _read_json(tmp_path / "gpu_provider_race_runtime_result.json")
    assert persisted["status"] == "provider_race_executed"


def test_provider_race_runtime_dry_run_is_wired_without_live_creds(
    tmp_path: Path,
) -> None:
    request_path = _launch_request(tmp_path / "gpu_provider_launch_request.json")
    _handoff(tmp_path / "gpu_provider_race_handoff.json")

    result = run_robot_eval_provider_race_runtime(
        provider_launch_request_path=request_path,
    )

    assert result["status"] == "ready_for_live_provider_race_runtime"
    assert result["provider_race_runtime_wired"] is True
    assert result["provider_race_execution_performed"] is False
    assert result["needs_live_provider_credentials"] is True
    assert result["blockers"] == []
    assert result["live_provider_calls_performed"] is False
    assert (
        result["claim_boundary"]["provider_race_runtime_launcher_not_implemented"]
        is False
    )


def test_provider_race_runtime_blocks_on_blocked_handoff_without_touching_providers(
    tmp_path: Path,
) -> None:
    request_path = _launch_request(
        tmp_path / "gpu_provider_launch_request.json",
        can_launch=False,
    )
    _handoff(tmp_path / "gpu_provider_race_handoff.json", ready=False)
    dud = _FakeRaceProvider("runpod", boots=True)
    healthy = _FakeRaceProvider("vast", boots=True)

    result = run_robot_eval_provider_race_runtime(
        provider_launch_request_path=request_path,
        providers=[dud, healthy],
        marker_check=_marker_check,
        sleep=_NO_SLEEP,
    )

    assert result["status"] == "blocked"
    assert result["provider_race_execution_performed"] is False
    assert dud.launch_calls == 0
    assert healthy.launch_calls == 0
    assert "customer_path_provider_failover_runtime_not_wired" in result["blockers"]


def test_provider_race_runtime_live_race_requires_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = _launch_request(tmp_path / "gpu_provider_launch_request.json")
    _handoff(tmp_path / "gpu_provider_race_handoff.json")
    healthy_a = _FakeRaceProvider("runpod", boots=True)
    healthy_b = _FakeRaceProvider("vast", boots=True)
    monkeypatch.delenv("BLUEPRINT_ALLOW_GPU_PROVIDER_RACE_LAUNCH", raising=False)

    result = run_robot_eval_provider_race_runtime(
        provider_launch_request_path=request_path,
        providers=[healthy_a, healthy_b],
        marker_check=_marker_check,
        live_provider_race=True,
        allow_live_provider_race=False,
        sleep=_NO_SLEEP,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "live_provider_race_gate_blocked"
    assert "missing_env_BLUEPRINT_ALLOW_GPU_PROVIDER_RACE_LAUNCH" in result["blockers"]
    assert "missing_cli_allow_live_provider_race" in result["blockers"]
    assert healthy_a.launch_calls == 0
    assert healthy_b.launch_calls == 0


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
