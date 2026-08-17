from __future__ import annotations

import json
import signal
import stat
import time
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_dispatcher_cgroup_cleanup import (
    cleanup_dispatcher_cgroup,
)
from blueprint_pipeline.vast_independent_watchdog_control import (
    HANDOFF_NAME,
    HANDOFF_SCHEMA,
    SYSTEMD_KILL_MODE_PROCESS_SURVIVAL,
)
from blueprint_pipeline.groot_oscar_runpod_watchdog import EVIDENCE_NAME


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _fixture(tmp_path: Path, *, terminal: bool = False):
    state = tmp_path / "runs"
    state.mkdir()
    job = state / "paired-run"
    out_dir = job / "independent_vast_watchdog"
    out_dir.mkdir(parents=True)
    watchdog_pid = 12001
    allocator_pid = 12002
    deadline = time.time() + 300
    prefix = "blueprint-adp-paired-native-import-fixture-"
    _write(
        job / HANDOFF_NAME,
        {
            "schema_version": HANDOFF_SCHEMA,
            "status": "retained_until_hard_ttl",
            "watchdog_pid": watchdog_pid,
            "watchdog_deadline_epoch": deadline,
            "pod_name_prefix": prefix,
            "watchdog_out_dir": str(out_dir),
            "caller_exit_survival_contract": SYSTEMD_KILL_MODE_PROCESS_SURVIVAL,
            "raw_secret_values_recorded": False,
        },
    )
    _write(
        out_dir / EVIDENCE_NAME,
        {
            "status": "provider_terminal" if terminal else "armed",
            "independent_process": True,
            "pid": watchdog_pid,
            "provider": "vast",
            "pod_name_prefix": prefix,
            "deadline_epoch": deadline,
            "raw_secret_values_recorded": False,
        },
    )
    proc = tmp_path / "proc"
    (proc / str(watchdog_pid)).mkdir(parents=True)
    (proc / str(allocator_pid)).mkdir(parents=True)
    (proc / str(watchdog_pid) / "cmdline").write_bytes(
        b"python\0-m\0blueprint_pipeline.groot_oscar_runpod_watchdog\0"
    )
    (proc / str(allocator_pid) / "cmdline").write_bytes(
        b"python\0-m\0blueprint_pipeline.paid_resource_allocator\0secret-value\0"
    )
    for pid in (watchdog_pid, allocator_pid):
        stat_fields = ["S", *(["1"] * 18), str(100_000 + pid)]
        (proc / str(pid) / "stat").write_text(
            f"{pid} (fixture) " + " ".join(stat_fields) + "\n",
            encoding="utf-8",
        )
    cgroup = tmp_path / "cgroup.procs"
    cgroup.write_text(f"999\n{watchdog_pid}\n{allocator_pid}\n", encoding="utf-8")
    return state, proc, cgroup, watchdog_pid, allocator_pid


def test_exec_stop_preserves_only_exact_watchdog_and_reaps_allocator(
    tmp_path: Path,
) -> None:
    state, proc, cgroup, watchdog_pid, allocator_pid = _fixture(tmp_path)
    alive = {watchdog_pid, allocator_pid}
    signals: list[tuple[int, int]] = []

    def killer(pid: int, sig: int) -> None:
        signals.append((pid, sig))
        if sig in {signal.SIGTERM, signal.SIGKILL}:
            alive.discard(pid)

    receipt_dir = tmp_path / "receipts"
    result = cleanup_dispatcher_cgroup(
        state_root=state,
        receipt_dir=receipt_dir,
        cgroup_procs_path=cgroup,
        proc_root=proc,
        self_pid=999,
        killer=killer,
        process_alive=lambda pid: pid in alive,
        sleeper=lambda _seconds: None,
    )

    assert [row["pid"] for row in result["preserved_watchdogs"]] == [watchdog_pid]
    assert result["terminated_non_watchdog_pids"] == [allocator_pid]
    assert (watchdog_pid, signal.SIGTERM) not in signals
    assert (allocator_pid, signal.SIGTERM) in signals
    receipt_path = next(receipt_dir.glob("*.json"))
    persisted = receipt_path.read_text(encoding="utf-8")
    assert "secret-value" not in persisted
    assert "paid_resource_allocator" not in persisted
    assert stat.S_IMODE(receipt_path.stat().st_mode) & 0o077 == 0


def test_stop_restart_reaps_terminal_or_invalid_leftovers(tmp_path: Path) -> None:
    state, proc, cgroup, watchdog_pid, allocator_pid = _fixture(
        tmp_path, terminal=True
    )
    alive = {watchdog_pid, allocator_pid}
    signals: list[tuple[int, int]] = []

    def killer(pid: int, sig: int) -> None:
        signals.append((pid, sig))
        if sig in {signal.SIGTERM, signal.SIGKILL}:
            alive.discard(pid)

    result = cleanup_dispatcher_cgroup(
        state_root=state,
        receipt_dir=tmp_path / "receipts",
        cgroup_procs_path=cgroup,
        proc_root=proc,
        self_pid=999,
        killer=killer,
        process_alive=lambda pid: pid in alive,
        sleeper=lambda _seconds: None,
    )

    assert result["preserved_watchdogs"] == []
    assert result["terminated_non_watchdog_pids"] == [watchdog_pid, allocator_pid]
    assert (watchdog_pid, signal.SIGTERM) in signals


@pytest.mark.parametrize(
    "mutation",
    [
        "expired",
        "wrong_prefix",
        "wrong_pid",
        "wrong_contract",
        "wrong_command",
        "world_readable_handoff",
    ],
)
def test_invalid_watchdog_binding_is_never_preserved(
    tmp_path: Path, mutation: str
) -> None:
    state, proc, cgroup, watchdog_pid, allocator_pid = _fixture(tmp_path)
    handoff_path = state / "paired-run" / HANDOFF_NAME
    handoff = json.loads(handoff_path.read_text())
    if mutation == "expired":
        handoff["watchdog_deadline_epoch"] = time.time() - 1
    elif mutation == "wrong_prefix":
        handoff["pod_name_prefix"] = "not-blueprint"
    elif mutation == "wrong_pid":
        handoff["watchdog_pid"] = 99999
    elif mutation == "wrong_contract":
        handoff["caller_exit_survival_contract"] = "systemd_cgroup_survival_unproven"
    elif mutation == "wrong_command":
        (proc / str(watchdog_pid) / "cmdline").write_bytes(
            b"python\0-m\0blueprint_pipeline.paid_resource_allocator\0"
        )
    _write(handoff_path, handoff)
    if mutation == "world_readable_handoff":
        handoff_path.chmod(0o644)
    alive = {watchdog_pid, allocator_pid}

    def killer(pid: int, sig: int) -> None:
        if sig in {signal.SIGTERM, signal.SIGKILL}:
            alive.discard(pid)

    result = cleanup_dispatcher_cgroup(
        state_root=state,
        receipt_dir=tmp_path / "receipts",
        cgroup_procs_path=cgroup,
        proc_root=proc,
        self_pid=999,
        killer=killer,
        process_alive=lambda pid: pid in alive,
        sleeper=lambda _seconds: None,
    )

    assert result["preserved_watchdogs"] == []
    assert watchdog_pid in result["terminated_non_watchdog_pids"]


def test_cleanup_blocks_when_nonwatchdog_cannot_be_signalled(tmp_path: Path) -> None:
    state, proc, cgroup, watchdog_pid, allocator_pid = _fixture(tmp_path)

    def killer(pid: int, sig: int) -> None:
        if pid == allocator_pid and sig == signal.SIGTERM:
            raise PermissionError("fixture denied")

    result = cleanup_dispatcher_cgroup(
        state_root=state,
        receipt_dir=tmp_path / "receipts",
        cgroup_procs_path=cgroup,
        proc_root=proc,
        self_pid=999,
        killer=killer,
        process_alive=lambda _pid: True,
        sleeper=lambda _seconds: None,
    )

    assert result["status"] == "blocked"
    assert "dispatcher_cgroup_signal_failed" in result["blockers"]
    assert result["surviving_non_watchdog_pids"] == [allocator_pid]
    assert result["preserved_watchdogs"][0]["pid"] == watchdog_pid


def test_pid_identity_change_is_not_signalled_and_blocks_cleanup(tmp_path: Path) -> None:
    state, proc, cgroup, watchdog_pid, allocator_pid = _fixture(tmp_path)
    calls: dict[int, int] = {}
    signals: list[tuple[int, int]] = []

    def identity(_proc: Path, pid: int) -> str:
        calls[pid] = calls.get(pid, 0) + 1
        if pid == allocator_pid and calls[pid] > 1:
            return "reused-process-start"
        return f"original-{pid}"

    result = cleanup_dispatcher_cgroup(
        state_root=state,
        receipt_dir=tmp_path / "receipts",
        cgroup_procs_path=cgroup,
        proc_root=proc,
        self_pid=999,
        killer=lambda pid, sig: signals.append((pid, sig)),
        process_alive=lambda _pid: True,
        process_identity=identity,
        sleeper=lambda _seconds: None,
    )

    assert result["status"] == "blocked"
    assert result["pid_identity_changed_before_signal"] == [allocator_pid]
    assert not any(pid == allocator_pid for pid, _sig in signals)
    assert result["surviving_non_watchdog_pids"] == [allocator_pid]


def test_preserved_watchdog_is_reproved_alive_at_final_readback(tmp_path: Path) -> None:
    state, proc, cgroup, watchdog_pid, allocator_pid = _fixture(tmp_path)
    calls: dict[int, int] = {}
    alive = {watchdog_pid, allocator_pid}

    def identity(_proc: Path, pid: int) -> str:
        calls[pid] = calls.get(pid, 0) + 1
        if pid == watchdog_pid and calls[pid] > 1:
            return "watchdog-reused-or-exited"
        return f"original-{pid}"

    def killer(pid: int, sig: int) -> None:
        if sig in {signal.SIGTERM, signal.SIGKILL}:
            alive.discard(pid)

    result = cleanup_dispatcher_cgroup(
        state_root=state,
        receipt_dir=tmp_path / "receipts",
        cgroup_procs_path=cgroup,
        proc_root=proc,
        self_pid=999,
        killer=killer,
        process_alive=lambda pid: pid in alive,
        process_identity=identity,
        sleeper=lambda _seconds: None,
    )

    assert result["status"] == "blocked"
    assert result["preserved_watchdog_failures"] == [watchdog_pid]
    assert "dispatcher_cgroup_preserved_watchdog_not_live" in result["blockers"]


def test_symlinked_handoff_is_not_watchdog_authority(tmp_path: Path) -> None:
    state, proc, cgroup, watchdog_pid, allocator_pid = _fixture(tmp_path)
    handoff = state / "paired-run" / HANDOFF_NAME
    target = state / "outside-handoff.json"
    target.write_text(handoff.read_text(), encoding="utf-8")
    handoff.unlink()
    handoff.symlink_to(target)
    alive = {watchdog_pid, allocator_pid}

    def killer(pid: int, sig: int) -> None:
        if sig in {signal.SIGTERM, signal.SIGKILL}:
            alive.discard(pid)

    result = cleanup_dispatcher_cgroup(
        state_root=state,
        receipt_dir=tmp_path / "receipts",
        cgroup_procs_path=cgroup,
        proc_root=proc,
        self_pid=999,
        killer=killer,
        process_alive=lambda pid: pid in alive,
        sleeper=lambda _seconds: None,
    )

    assert result["preserved_watchdogs"] == []
    assert watchdog_pid in result["terminated_non_watchdog_pids"]


def test_cgroup_process_limit_fails_closed(tmp_path: Path) -> None:
    state = tmp_path / "runs"
    state.mkdir()
    cgroup = tmp_path / "cgroup.procs"
    cgroup.write_text(
        "\n".join(str(20_000 + index) for index in range(4097)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dispatcher_cgroup_processes_invalid"):
        cleanup_dispatcher_cgroup(
            state_root=state,
            receipt_dir=tmp_path / "receipts",
            cgroup_procs_path=cgroup,
            proc_root=tmp_path / "proc",
            self_pid=999,
        )
