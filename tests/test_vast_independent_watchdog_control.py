from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import vast_independent_watchdog_control as control


@pytest.fixture(autouse=True)
def _clear_ambient_systemd_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv(control.CALLER_EXIT_SURVIVAL_ENV, raising=False)


class _FakeProcess:
    pid = 4321

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.kwargs = kwargs
        self.returncode: int | None = None
        out_dir = Path(command[command.index("--out-dir") + 1])
        prefix = command[command.index("--pod-name-prefix") + 1]
        payload = {
            "status": "armed",
            "independent_process": True,
            "pid": self.pid,
            "pod_name_prefix": prefix,
            "provider": "vast",
        }
        (out_dir / control.EVIDENCE_NAME).write_text(json.dumps(payload), encoding="utf-8")

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _exact_terminal_evidence(
    handle: control.VastWatchdogHandle, instance_id: int
) -> dict[str, Any]:
    lane_zero = {
        "status": "observed",
        "provider": "vast",
        "name_prefix": handle.pod_name_prefix,
        "api_confirmed": True,
        "live_resource_count": 0,
        "resources": [],
    }
    global_zero = {**lane_zero, "name_prefix": ""}
    inspect = {
        "status": "absent",
        "provider": "vast",
        "http": 404,
        "instance_id": str(instance_id),
        "api_confirmed": True,
        "provider_absence_confirmed": True,
    }
    return {
        "status": "provider_terminal",
        "provider": "vast",
        "provider_absence_confirmed": True,
        "pod_name_prefix": handle.pod_name_prefix,
        "pid": handle.process.pid,
        "deadline_epoch": handle.deadline_epoch,
        "owner_teardown_cancel_requested": True,
        "owner_teardown_cancel_request_valid": True,
        "recorded_vast_instance_teardown": {
            "status": "absent",
            "instance_id": str(instance_id),
            "provider_absence_confirmed": True,
            "provider_mutations_performed": 0,
            "inspect_attempts": [{**inspect, "attempt": 1}, {**inspect, "attempt": 2}],
        },
        "initial_inventory": lane_zero,
        "final_inventory": lane_zero,
        "initial_global_inventory": global_zero,
        "final_global_inventory": global_zero,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }

def test_watchdog_is_armed_detached_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv(control.CALLER_EXIT_SURVIVAL_ENV, raising=False)
    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)

    handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-07-27T00:00:00+00:00",
        pod_name_prefix="blueprint-groot-oscar-canary-vast-wam-",
        allowed_active_instance_ids=[47226054],
    )

    assert handoff["status"] == "armed"
    assert handoff["watchdog_armed_before_allocation"] is True
    assert handoff["watchdog_deadline_epoch"] - handoff["watchdog_started_epoch"] == 180
    assert handle is not None
    assert handle.process.kwargs["start_new_session"] is True
    assert handle.process.kwargs["stdin"] is control.subprocess.DEVNULL
    assert handle.pod_name_prefix.startswith("blueprint-groot-oscar-canary-vast-wam-")
    assert handle.allowed_active_instance_ids == (47226054,)
    assert handoff["allowed_active_instance_ids"] == [47226054]
    assert handoff["caller_exit_survival_contract"] == "detached_posix_session"
    index = handle.process.command.index("--allowed-active-instance-id")
    assert handle.process.command[index + 1] == "47226054"


def test_dispatcher_systemd_unit_preserves_retained_watchdog_cgroup() -> None:
    unit = (
        Path(__file__).resolve().parents[1]
        / "deploy/systemd/blueprint-task-evaluation-launch-dispatcher.service"
    ).read_text(encoding="utf-8")

    assert "KillMode=process" in unit
    assert (
        "BLUEPRINT_VAST_WATCHDOG_CALLER_EXIT_SURVIVAL="
        "systemd_dispatcher_kill_mode_process"
    ) in unit
    assert "ExecStopPost=" in unit
    assert "blueprint_pipeline.task_evaluation_dispatcher_cgroup_cleanup" in unit
    assert "watchdog child" in unit


def test_watchdog_preserves_exact_caller_supplied_sam_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)

    handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-08-14T08:13:04+00:00",
        pod_name_prefix="blueprint-sam31-source-tracks-",
    )

    assert handle is not None
    assert handoff["pod_name_prefix"].startswith("blueprint-sam31-source-tracks-")
    assert handle.pod_name_prefix == handoff["pod_name_prefix"]
    assert "blueprint-groot-oscar-canary-vast-wam-" not in handoff["pod_name_prefix"]


def test_paired_native_import_prefix_is_accepted_by_real_watchdog_contract(
    tmp_path: Path,
) -> None:
    import time

    from blueprint_pipeline.groot_oscar_runpod_watchdog import arm_watchdog

    receipt = arm_watchdog(
        out_dir=tmp_path,
        pod_name_prefix="blueprint-adp-paired-native-import-bound-run-",
        deadline_epoch=time.time() + 120,
        provider_name="vast",
    )

    assert receipt["status"] == "armed"
    assert receipt["provider"] == "vast"
    assert receipt["pre_deadline_provider_mutation_allowed"] is False


def test_watchdog_exact_instance_handoff_is_atomic_and_private(tmp_path: Path) -> None:
    path = tmp_path / "watchdog" / "started_vast_instance_id.txt"

    control.write_started_vast_instance_id(path, 46031731)

    assert path.read_text(encoding="utf-8") == "46031731\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not path.with_suffix(".tmp").exists()


def test_watchdog_process_start_failure_blocks_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_start(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic start failure")

    monkeypatch.setattr(control.subprocess, "Popen", fail_start)

    handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-07-27T00:00:00+00:00",
        pod_name_prefix="blueprint-groot-oscar-canary-vast-wam-",
    )

    assert handle is None
    assert handoff["status"] == "blocked"
    assert handoff["watchdog_armed_before_allocation"] is False
    assert handoff["provider_mutations_performed"] == 0
    assert handoff["blockers"] == ["independent_vast_watchdog_process_start_failed"]


def test_watchdog_rejects_one_minute_ttl_before_process_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        control.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("short TTL must block before process start"),
    )

    handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=1,
        generated_at="2026-07-27T00:00:00+00:00",
        pod_name_prefix="blueprint-groot-oscar-canary-vast-wam-",
    )

    assert handle is None
    assert handoff["status"] == "blocked"
    assert handoff["blockers"] == ["independent_vast_watchdog_ttl_too_short"]


def test_watchdog_without_allocation_is_cancelled_without_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)
    _handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-07-27T00:00:00+00:00",
        pod_name_prefix="blueprint-groot-oscar-canary-vast-wam-",
    )
    assert handle is not None

    result = control.close_independent_vast_watchdog(
        job_dir=tmp_path,
        handle=handle,
        instance_ids=[],
        provider_teardown_completed=True,
        provider_allocation_impossible=True,
    )

    assert result["status"] == "cancelled_no_allocation"
    assert result["provider_mutations_performed"] == 0
    assert handle.process.poll() == -15


def test_watchdog_without_allocation_retains_double_api_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)

    class Provider:
        def billable_inventory(self, *, name_prefix: str) -> dict[str, Any]:
            return {
                "status": "available",
                "name_prefix": name_prefix,
                "api_confirmed": True,
                "live_resource_count": 0,
            }

    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers.get_render_provider",
        lambda _provider: Provider(),
    )
    _handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-08-13T00:00:00+00:00",
        pod_name_prefix="blueprint-groot-oscar-canary-vast-wam-",
    )
    assert handle is not None

    result = control.close_independent_vast_watchdog_without_allocation(
        job_dir=tmp_path,
        handle=handle,
    )

    assert result["status"] == "provider_terminal"
    assert result["provider_absence_confirmed"] is True
    assert result["final_global_inventory"]["live_resource_count"] == 0
    assert result["provider_mutations_performed"] == 0


def test_watchdog_stays_armed_when_create_identity_is_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)
    _handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-07-27T00:00:00+00:00",
        pod_name_prefix="blueprint-groot-oscar-canary-vast-wam-",
    )
    assert handle is not None

    result = control.close_independent_vast_watchdog(
        job_dir=tmp_path,
        handle=handle,
        instance_ids=[],
        provider_teardown_completed=True,
    )

    assert result["status"] == "retained_until_hard_ttl"
    assert result["reason"] == "provider_allocation_identity_ambiguous"
    assert handle.process.poll() is None


def test_watchdog_refuses_retained_claim_after_process_already_died(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)
    _handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-08-17T00:00:00+00:00",
        pod_name_prefix="blueprint-adp-paired-native-import-",
    )
    assert handle is not None
    handle.process.returncode = -15

    result = control.close_independent_vast_watchdog(
        job_dir=tmp_path,
        handle=handle,
        instance_ids=[47999991],
        provider_teardown_completed=False,
    )

    assert result["status"] == "watchdog_process_not_live"
    assert result["watchdog_retention_liveness_confirmed"] is False
    assert result["blockers"] == ["independent_vast_watchdog_process_not_live"]


def test_systemd_watchdog_refuses_retention_without_kill_mode_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INVOCATION_ID", "fixture-systemd-invocation")
    monkeypatch.delenv(control.CALLER_EXIT_SURVIVAL_ENV, raising=False)
    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)
    handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-08-17T00:00:00+00:00",
        pod_name_prefix="blueprint-adp-paired-native-import-",
    )
    assert handle is not None
    assert handoff["caller_exit_survival_contract"] == (
        "systemd_cgroup_survival_unproven"
    )

    result = control.close_independent_vast_watchdog(
        job_dir=tmp_path,
        handle=handle,
        instance_ids=[47999991],
        provider_teardown_completed=False,
    )

    assert result["status"] == "watchdog_caller_exit_survival_unproven"
    assert result["watchdog_retention_liveness_confirmed"] is True
    assert result["watchdog_caller_exit_survival_confirmed"] is False
    assert result["blockers"] == [
        "independent_vast_watchdog_caller_exit_survival_unproven"
    ]


def test_close_returns_terminal_as_soon_as_evidence_confirms_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow watchdog must not be misread as unclosed.

    v7/v9 wrote provider_terminal evidence ~2-3 minutes after the owner cancel
    request while close only waited 45 seconds for process exit and read the
    evidence once, so a correctly closing watchdog was recorded as
    joint_agent_independent_watchdog_not_closed. Close now polls the evidence
    during the wait and returns terminal as soon as absence is confirmed.
    """

    import threading
    import time as time_module

    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)
    _handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-08-09T00:00:00+00:00",
        pod_name_prefix="blueprint-groot-oscar-canary-vast-wam-",
    )
    assert handle is not None

    def _write_terminal_evidence() -> None:
        (handle.out_dir / control.EVIDENCE_NAME).write_text(
            json.dumps(_exact_terminal_evidence(handle, 47283980)),
            encoding="utf-8",
        )

    threading.Timer(0.5, _write_terminal_evidence).start()
    started = time_module.monotonic()
    result = control.close_independent_vast_watchdog(
        job_dir=tmp_path,
        handle=handle,
        instance_ids=[47283980],
        provider_teardown_completed=True,
        wait_seconds=30.0,
    )
    elapsed = time_module.monotonic() - started

    assert result["status"] == "provider_terminal"
    assert result["provider_absence_confirmed"] is True
    assert handle.process.poll() is not None
    assert elapsed < 10.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.__setitem__("provider", "runpod"),
        lambda row: row.__setitem__("pod_name_prefix", "blueprint-wrong-"),
        lambda row: row.__setitem__("pid", 999999),
        lambda row: row.__setitem__("deadline_epoch", 0),
        lambda row: row.__setitem__("owner_teardown_cancel_request_valid", False),
        lambda row: row["recorded_vast_instance_teardown"].__setitem__(
            "instance_id", "111"
        ),
        lambda row: row["final_inventory"].update(
            {"live_resource_count": 1, "resources": [{"instance_id": "47999991"}]}
        ),
        lambda row: row["final_inventory"].update(
            {"name_prefix": "blueprint-another-lane-"}
        ),
        lambda row: row["final_inventory"].update({"provider": "runpod"}),
        lambda row: row["final_inventory"].update({"status": "blocked"}),
        lambda row: row["final_global_inventory"].update(
            {"live_resource_count": 1, "resources": [{"instance_id": "unallowed"}]}
        ),
        lambda row: row["final_global_inventory"].update(
            {"name_prefix": "blueprint-wrong-global-"}
        ),
        lambda row: row["recorded_vast_instance_teardown"].update(
            {"provider_mutations_performed": 1}
        ),
        lambda row: row["recorded_vast_instance_teardown"].update(
            {"inspect_attempts": row["recorded_vast_instance_teardown"]["inspect_attempts"][:1]}
        ),
        lambda row: row["recorded_vast_instance_teardown"]["inspect_attempts"][0].update(
            {"provider": "runpod"}
        ),
        lambda row: row["recorded_vast_instance_teardown"]["inspect_attempts"][0].update(
            {"http": None}
        ),
    ],
)
def test_close_refuses_stale_or_misscoped_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate
) -> None:
    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)
    _handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-08-17T00:00:00+00:00",
        pod_name_prefix="blueprint-adp-paired-native-import-",
    )
    assert handle is not None
    evidence = _exact_terminal_evidence(handle, 47999991)
    mutate(evidence)
    (handle.out_dir / control.EVIDENCE_NAME).write_text(json.dumps(evidence))

    result = control.close_independent_vast_watchdog(
        job_dir=tmp_path,
        handle=handle,
        instance_ids=[47999991],
        provider_teardown_completed=True,
        wait_seconds=0.2,
    )

    assert result["status"] == "retained_until_hard_ttl"
    assert result["provider_absence_confirmed"] is False
    assert handle.process.poll() is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX process parenting")
def test_detached_watchdog_survives_parent_exit_and_seals_owner_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child contract survives parent exit; unit tests bind systemd preservation."""

    from blueprint_pipeline import groot_oscar_runpod_watchdog as worker
    from blueprint_pipeline.watchdog_owner_teardown_contract import (
        write_owner_teardown_cancel_request,
    )

    out_dir = tmp_path / "independent_vast_watchdog"
    out_dir.mkdir()
    child_pid_path = tmp_path / "watchdog-child.pid"
    prefix = "blueprint-adp-paired-native-import-parent-exit-fixture-"
    instance_id = "47999991"
    secret = "fixture-secret-must-not-enter-watchdog-evidence"
    monkeypatch.setenv("BLUEPRINT_TEST_WATCHDOG_SECRET", secret)

    dispatcher_pid = os.fork()
    if dispatcher_pid == 0:  # pragma: no cover - asserted by parent process
        watchdog_pid = os.fork()
        if watchdog_pid == 0:
            try:
                os.setsid()

                class Provider:
                    def inspect(self, observed_id: str) -> dict[str, Any]:
                        return {
                            "status": "absent",
                            "provider": "vast",
                            "http": 404,
                            "instance_id": observed_id,
                            "api_confirmed": True,
                            "provider_absence_confirmed": True,
                        }

                worker._billable_inventory = lambda **_kwargs: {  # type: ignore[attr-defined]
                    "status": "observed",
                    "api_confirmed": True,
                    "live_resource_count": 0,
                    "resources": [],
                }
                result = worker.run_watchdog(
                    out_dir=out_dir,
                    pod_name_prefix=prefix,
                    deadline_epoch=time.time() + 120,
                    provider_name="vast",
                    provider_factory=lambda _name: Provider(),
                    sleeper=lambda seconds: time.sleep(min(seconds, 0.05)),
                )
            except Exception:
                os._exit(2)
            os._exit(0 if result.get("status") == "provider_terminal" else 3)
        child_pid_path.write_text(f"{watchdog_pid}\n", encoding="utf-8")
        os._exit(0)

    os.waitpid(dispatcher_pid, 0)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not child_pid_path.is_file():
        time.sleep(0.02)
    watchdog_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
    evidence_path = out_dir / control.EVIDENCE_NAME
    while time.monotonic() < deadline:
        evidence = json.loads(evidence_path.read_text()) if evidence_path.is_file() else {}
        if evidence.get("status") == "armed":
            break
        time.sleep(0.02)
    else:
        os.kill(watchdog_pid, 9)
        pytest.fail("watchdog did not arm after dispatcher parent exit")
    os.kill(watchdog_pid, 0)
    control.write_started_vast_instance_id(
        out_dir / "started_vast_instance_id.txt", int(instance_id)
    )
    write_owner_teardown_cancel_request(
        root=out_dir,
        pod_name_prefix=prefix,
        provider_name="vast",
        instance_id=instance_id,
    )
    terminal_deadline = time.monotonic() + 10
    terminal: dict[str, Any] = {}
    while time.monotonic() < terminal_deadline:
        terminal = json.loads(evidence_path.read_text())
        if terminal.get("status") == "provider_terminal":
            break
        time.sleep(0.05)
    else:
        os.kill(watchdog_pid, 9)
        pytest.fail("watchdog did not consume owner cancel")
    assert terminal["provider_absence_confirmed"] is True
    assert terminal["owner_teardown_cancel_requested"] is True
    assert terminal["recorded_vast_instance_teardown"]["instance_id"] == instance_id
    assert terminal["recorded_vast_instance_teardown"][
        "provider_absence_confirmed"
    ] is True
    persisted = b"\n".join(path.read_bytes() for path in out_dir.rglob("*") if path.is_file())
    assert secret.encode() not in persisted
    process_gone = False
    while time.monotonic() < terminal_deadline:
        try:
            os.kill(watchdog_pid, 0)
        except ProcessLookupError:
            process_gone = True
            break
        time.sleep(0.05)
    if not process_gone:
        os.kill(watchdog_pid, 9)
    assert process_gone, "terminal watchdog must not remain orphaned"


def test_close_default_wait_covers_slow_absence_confirmation() -> None:
    import inspect

    signature = inspect.signature(control.close_independent_vast_watchdog)
    assert signature.parameters["wait_seconds"].default >= 300.0
