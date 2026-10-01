# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_disk_budget.py
#   src/blueprint_pipeline/control_plane_disk_footprints.py
#   src/blueprint_pipeline/control_plane_disk_ledger.py
from __future__ import annotations

import errno
import json
import os
import re
import threading
from collections import namedtuple
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_disk_budget as disk_budget
from blueprint_pipeline import control_plane_disk_footprints as footprints
from blueprint_pipeline.control_plane_disk_reservation_heartbeat import keep_reservation_live
from blueprint_pipeline.control_plane_disk_budget import (
    ControlPlaneDiskBudgetError,
    disk_headroom,
    reserve_control_plane_disk,
)


Usage = namedtuple("Usage", "total used free")
GIB = 1024**3


@pytest.mark.parametrize("preinstalled", [False, True])
def test_disk_admission_works_when_runtime_denies_setgid_syscalls(
    tmp_path, monkeypatch, preinstalled
) -> None:
    """RestrictSUIDSGID rejects flagged mkdir even when the directory exists."""
    ledger = tmp_path / "ledger"
    if preinstalled:
        ledger.mkdir()
        ledger.chmod(0o2770)
        installed_mode = ledger.stat().st_mode & 0o7777
    real_mkdir, real_fchmod = os.mkdir, os.fchmod

    def restricted_mkdir(path, mode=0o777, *, dir_fd=None):
        if mode & 0o6000:
            raise PermissionError(errno.EPERM, "runtime denies privilege bits", path)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    def restricted_fchmod(descriptor, mode):
        if mode & 0o6000:
            raise PermissionError(errno.EPERM, "runtime denies privilege bits")
        return real_fchmod(descriptor, mode)

    monkeypatch.setattr(os, "mkdir", restricted_mkdir)
    monkeypatch.setattr(os, "fchmod", restricted_fchmod)
    reservation = reserve_control_plane_disk(
        "launch_preparation",
        target_root=tmp_path,
        expected_bytes=GIB,
        reservation_root=ledger,
        disk_usage=lambda _path: Usage(100 * GIB, 60 * GIB, 40 * GIB),
        now=lambda: 100.0,
        pid_alive=lambda _pid: True,
    )
    assert reservation.path.is_file()
    assert (ledger / ".lock").stat().st_mode & 0o777 == 0o660
    if preinstalled:
        assert ledger.stat().st_mode & 0o7777 == installed_mode
    reservation.release(outcome="blocked")
    assert not list(ledger.glob("*.json"))


def test_restore_admission_and_renewal_refuse_a_busy_shared_ledger(tmp_path):
    import fcntl
    ledger = disk_budget._prepare_ledger_root(tmp_path / 'ledger')
    options = dict(target_root=tmp_path, expected_bytes=4096, reservation_root=ledger,
        disk_usage=lambda _: Usage(100 * GIB, 60 * GIB, 40 * GIB), lock_nonblocking=True)
    descriptor = disk_budget.open_ledger_lock(ledger, require_mode=True)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ControlPlaneDiskBudgetError, match='lock_busy'):
            reserve_control_plane_disk('experiment_restore', **options)
    finally:
        os.close(descriptor)
    reservation = reserve_control_plane_disk('experiment_restore', **options)
    before = reservation.path.read_bytes()
    descriptor = disk_budget.open_ledger_lock(ledger, require_mode=True)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ControlPlaneDiskBudgetError, match='lock_busy'):
            reservation.renew()
        assert reservation.path.read_bytes() == before
    finally:
        os.close(descriptor)
        reservation.release()


def test_preinstalled_group_writable_lock_is_not_rechmodded(
    tmp_path, monkeypatch
) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir(mode=0o2770)
    lock = ledger / ".lock"
    lock.touch(mode=0o660)
    lock.chmod(0o660)
    original_chmod = disk_budget.os.chmod

    def reject_lock_chmod(path, mode, **kwargs) -> None:
        if path == lock:
            raise PermissionError("root-owned correct lock must not be rechmodded")
        original_chmod(path, mode, **kwargs)

    def reject_lock_fchmod(*_args, **_kwargs) -> None:
        raise PermissionError("root-owned correct lock must not be rechmodded")

    monkeypatch.setattr(disk_budget.os, "chmod", reject_lock_chmod)
    monkeypatch.setattr(disk_budget.os, "fchmod", reject_lock_fchmod)

    reservation = reserve_control_plane_disk(
        "launch_activation",
        target_root=tmp_path,
        expected_bytes=GIB,
        reservation_root=ledger,
        disk_usage=lambda _path: Usage(100 * GIB, 60 * GIB, 40 * GIB),
        now=lambda: 100.0,
        pid_alive=lambda _pid: True,
    )

    assert lock.stat().st_mode & 0o777 == 0o660
    reservation.release()


def test_unsafe_lock_mode_fails_closed_when_owner_rejects_repair(
    tmp_path, monkeypatch
) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir(mode=0o2770)
    lock = ledger / ".lock"
    lock.touch(mode=0o640)
    lock.chmod(0o640)

    def reject_lock_chmod(*_args, **_kwargs) -> None:
        raise PermissionError("runtime account cannot chmod root-owned lock")

    monkeypatch.setattr(disk_budget.os, "fchmod", reject_lock_chmod)

    with pytest.raises(
        ControlPlaneDiskBudgetError,
        match="control_plane_disk_budget_lock_mode_invalid:0640",
    ):
        reserve_control_plane_disk(
            "launch_activation",
            target_root=tmp_path,
            expected_bytes=GIB,
            reservation_root=ledger,
            disk_usage=lambda _path: Usage(100 * GIB, 60 * GIB, 40 * GIB),
        )


def test_reservation_accounts_for_live_concurrent_reservations(tmp_path) -> None:
    ledger = tmp_path / "ledger"

    def usage(_path):
        return Usage(100 * GIB, 60 * GIB, 40 * GIB)

    first = reserve_control_plane_disk(
        "launch_activation",
        target_root=tmp_path / "future-output",
        expected_bytes=20 * GIB,
        reservation_root=ledger,
        disk_usage=usage,
        now=lambda: 100.0,
        pid_alive=lambda _pid: True,
    )
    assert first.path.is_file()
    assert first.path.stat().st_mode & 0o777 == 0o640
    with pytest.raises(
        ControlPlaneDiskBudgetError,
        match=(
            r"control_plane_disk_budget_exceeded:launch_activation:"
            r"need_bytes=21474836480:available_bytes=12884901888"
        ),
    ):
        reserve_control_plane_disk(
            "launch_activation",
            target_root=tmp_path,
            expected_bytes=20 * GIB,
            reservation_root=ledger,
            disk_usage=usage,
            now=lambda: 101.0,
            pid_alive=lambda _pid: True,
        )
    first.release()
    assert not first.path.exists()


def test_reservation_renewal_keeps_live_bytes_past_original_expiry(tmp_path) -> None:
    ledger = tmp_path / "ledger"
    clock = [100.0]
    def usage(_path):
        return Usage(100 * GIB, 60 * GIB, 40 * GIB)
    reservation = reserve_control_plane_disk(
        "handoff_staging", target_root=tmp_path, expected_bytes=20 * GIB,
        reservation_root=ledger, disk_usage=usage, now=lambda: clock[0],
        pid_alive=lambda _pid: True, ttl_seconds=120,
    )
    original_expiry = json.loads(reservation.path.read_text())["expires_at_epoch"]
    clock[0] = 190.0
    reservation.renew()
    renewed_expiry = json.loads(reservation.path.read_text())["expires_at_epoch"]
    assert renewed_expiry == 310.0 > original_expiry
    clock[0] = 225.0
    with pytest.raises(ControlPlaneDiskBudgetError, match="control_plane_disk_budget_exceeded"):
        reserve_control_plane_disk(
            "launch_dispatch", target_root=tmp_path, expected_bytes=20 * GIB,
            reservation_root=ledger, disk_usage=usage, now=lambda: clock[0],
            pid_alive=lambda _pid: True,
        )
    reservation.release()


def test_expired_reservation_cannot_be_resurrected_by_renewal(tmp_path) -> None:
    clock = [100.0]
    reservation = reserve_control_plane_disk(
        "handoff_staging", target_root=tmp_path, expected_bytes=GIB,
        reservation_root=tmp_path / "ledger", ttl_seconds=120,
        disk_usage=lambda _path: Usage(100 * GIB, 60 * GIB, 40 * GIB),
        now=lambda: clock[0],
    )
    clock[0] = 221.0
    with pytest.raises(ControlPlaneDiskBudgetError, match="reservation_expired"):
        reservation.renew()
    reservation.release()


def test_heartbeat_renews_during_a_blocked_copy(tmp_path, monkeypatch) -> None:
    clock = [100.0]
    reservation = reserve_control_plane_disk(
        "handoff_staging", target_root=tmp_path, expected_bytes=GIB,
        reservation_root=tmp_path / "ledger", ttl_seconds=120,
        disk_usage=lambda _path: Usage(100 * GIB, 60 * GIB, 40 * GIB),
        now=lambda: clock[0],
    )
    renewed = threading.Event()
    original = reservation.renew

    def renew_and_signal():
        original()
        renewed.set()

    monkeypatch.setattr(reservation, "renew", renew_and_signal)
    with reservation, keep_reservation_live(reservation, interval_seconds=0.01):
        clock[0] = 190.0
        assert renewed.wait(timeout=1.0)
        assert json.loads(reservation.path.read_text())["expires_at_epoch"] == 310.0


def test_expired_or_dead_reservations_do_not_consume_headroom(tmp_path) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    stale = ledger / "stale.json"
    stale.write_text(
        json.dumps(
            {
                "device": tmp_path.stat().st_dev,
                "pid": 999,
                "expected_bytes": 30 * GIB,
                "expires_at_epoch": 50,
            }
        ),
        encoding="utf-8",
    )
    reservation = reserve_control_plane_disk(
        "launch_activation",
        target_root=tmp_path,
        expected_bytes=20 * GIB,
        reservation_root=ledger,
        disk_usage=lambda _path: Usage(100 * GIB, 60 * GIB, 40 * GIB),
        now=lambda: 100.0,
        pid_alive=lambda _pid: False,
    )
    assert reservation.reserved_bytes == 0
    assert not stale.exists()
    reservation.release()


def test_headroom_projects_refused_roles_without_paths(tmp_path) -> None:
    report = disk_headroom(
        target_root=tmp_path / "not-created",
        reservation_root=tmp_path / "ledger",
        disk_usage=lambda _path: Usage(100 * GIB, 93 * GIB, 7 * GIB),
        now=lambda: 100.0,
        pid_alive=lambda _pid: True,
    )
    assert report["status"] == "low"
    assert set(report["refused_roles"]) == {
        "handoff_staging",
        "launch_preparation",
        "episode_compilation",
        "launch_activation",
        "launch_dispatch",
        "policy_canary_dispatch",
        "evidence_offload",
        "result_artifact_download",
        "stage_replay",
        "semantic_pretraining",
        "cpu_prestage",
        "scene_configuration_output",
        "g1_checkpoint_cache",
        "experiment_restore",
        "policy_canary_output",
    }
    assert next(row for row in report["targets"] if row["role"] == "control_plane_deploy")["refused"] is False
    assert str(tmp_path) not in json.dumps(report)


def test_environment_overrides_floor_and_role_footprint(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", str(GIB))
    monkeypatch.setenv(
        "BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_LAUNCH_ACTIVATION_BYTES",
        str(3 * GIB),
    )
    reservation = reserve_control_plane_disk(
        "launch_activation",
        target_root=tmp_path,
        reservation_root=tmp_path / "ledger",
        disk_usage=lambda _path: Usage(10 * GIB, 5 * GIB, 5 * GIB),
        now=lambda: 100.0,
        pid_alive=lambda _pid: True,
    )
    assert reservation.floor_bytes == GIB
    assert reservation.expected_bytes == 3 * GIB
    reservation.release()


def test_critical_roles_use_the_reserved_band(tmp_path) -> None:
    def usage(_path):
        return Usage(165 * GIB, 160 * GIB, 5 * GIB)
    ledger = tmp_path / "ledger"
    with pytest.raises(ControlPlaneDiskBudgetError, match="control_plane_disk_budget_exceeded"):
        reserve_control_plane_disk(
            "launch_activation", target_root=tmp_path, reservation_root=ledger,
            expected_bytes=450 * 1024**2, disk_usage=usage,
        )

    reservation = reserve_control_plane_disk(
        "control_plane_deploy", target_root=tmp_path, reservation_root=ledger,
        expected_bytes=450 * 1024**2, disk_usage=usage,
    )
    assert reservation.floor_bytes == max(GIB, int(165 * GIB * 0.01))
    reservation.release()


def test_critical_floor_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_CRITICAL_FLOOR_BYTES", str(3 * GIB))
    with pytest.raises(ControlPlaneDiskBudgetError, match="control_plane_disk_budget_exceeded"):
        reserve_control_plane_disk(
            "control_plane_deploy", target_root=tmp_path,
            reservation_root=tmp_path / "ledger", expected_bytes=450 * 1024**2,
            disk_usage=lambda _path: Usage(10 * GIB, 8 * GIB, 2 * GIB),
        )
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_CRITICAL_FLOOR_BYTES", str(512 * 1024**2))
    reservation = reserve_control_plane_disk(
        "control_plane_deploy", target_root=tmp_path,
        reservation_root=tmp_path / "ledger", expected_bytes=450 * 1024**2,
        disk_usage=lambda _path: Usage(10 * GIB, 8 * GIB, 2 * GIB),
    )
    assert reservation.floor_bytes == 512 * 1024**2
    reservation.release()


def test_headroom_is_computed_on_each_role_target_device(tmp_path) -> None:
    system = tmp_path / "system"
    scratch = tmp_path / "scratch"
    system.mkdir()
    scratch.mkdir()
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "scratch-running.json").write_text(json.dumps({
        "device": 202, "pid": 17, "expected_bytes": GIB,
        "expires_at_epoch": 1000,
    }), encoding="utf-8")

    def device_of(path):
        return 202 if Path(path) == scratch else 101

    def usage(path):
        return (Usage(100 * GIB, 94 * GIB, 6 * GIB) if Path(path) == scratch
                else Usage(100 * GIB, 70 * GIB, 30 * GIB))

    report = disk_headroom(
        target_root=system, role_targets={"launch_activation": scratch},
        reservation_root=ledger, disk_usage=usage, device_of=device_of,
        now=lambda: 100.0, pid_alive=lambda _pid: True,
    )

    targets = {row["role"]: row for row in report["targets"]}
    assert report["free_bytes"] == 30 * GIB
    assert report["reserved_bytes"] == 0
    assert targets["launch_activation"]["device"] == 202
    assert targets["launch_activation"]["reserved_bytes"] == GIB
    assert targets["launch_activation"]["refused"] is True
    assert "launch_activation" in report["refused_roles"]
    assert targets["control_plane_deploy"]["device"] == 101
    assert targets["control_plane_deploy"]["refused"] is False


def test_role_targets_parse_and_refuse_garbage() -> None:
    assert disk_budget.parse_role_targets(None) == {}
    assert disk_budget.parse_role_targets("launch_activation=/mnt/scratch,control_plane_deploy=/var/lib/blueprint") == {
        "launch_activation": Path("/mnt/scratch"),
        "control_plane_deploy": Path("/var/lib/blueprint"),
    }
    for raw in ("unknown=/mnt/scratch", "launch_activation=relative", "launch_activation=",
                "launch_activation=/mnt/a,launch_activation=/mnt/b", "launch_activation=/mnt/a,"):
        with pytest.raises(ControlPlaneDiskBudgetError, match="control_plane_disk_budget_role_targets_invalid"):
            disk_budget.parse_role_targets(raw)


def test_invalid_environment_override_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", "unknown")
    with pytest.raises(
        ControlPlaneDiskBudgetError,
        match="control_plane_disk_budget_configuration_invalid",
    ):
        disk_headroom(
            target_root=tmp_path,
            reservation_root=tmp_path / "ledger",
        )


MIB = 1024**2


def _samples(ledger, role, values, outcome="completed"):
    for value in values:
        assert disk_budget.record_footprint_sample(
            reservation_root=ledger, role=role, observed_bytes=value,
            reserved_bytes=2 * GIB, outcome=outcome, now=lambda: 50.0)


def test_short_history_keeps_the_declared_constant(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 9)
    measured = disk_budget.measured_footprint("launch_activation", reservation_root=ledger)
    assert measured == {"role": "launch_activation", "bytes": 2 * GIB, "basis": "declared_default",
                        "sample_count": 9, "declared_bytes": 2 * GIB, "p95_bytes": None}


def test_p95_times_headroom_is_used_once_ten_samples_exist(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 9 + [400 * MIB])
    measured = disk_budget.measured_footprint("launch_activation", reservation_root=ledger)
    assert measured["basis"] == "measured_p95"
    assert measured["p95_bytes"] == 400 * MIB
    assert measured["bytes"] == 500 * MIB


def test_sample_above_the_clamp_cannot_raise_the_reservation(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [10 * GIB] * 12)
    assert disk_budget.effective_footprint_bytes("launch_activation", reservation_root=ledger) == 2 * GIB


def test_tiny_samples_are_floored_and_failed_samples_ignored(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [1] * 10)
    _samples(ledger, "launch_activation", [9 * GIB] * 5, outcome="failed")
    assert disk_budget.effective_footprint_bytes("launch_activation", reservation_root=ledger) == 64 * MIB


def test_unreadable_history_falls_back_to_declared(tmp_path):
    ledger = tmp_path / "ledger"
    (ledger / "history").mkdir(parents=True)
    (ledger / "history" / "launch_activation.jsonl").write_text("{not json\n" * 20)
    assert disk_budget.measured_footprint("launch_activation", reservation_root=ledger)["basis"] == "declared_default"


def test_history_is_compacted_to_its_newest_samples(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [MIB + index for index in range(600)])
    path = ledger / "history" / "launch_activation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # 600 samples outgrow the 128 KiB threshold, so compaction must have dropped
    # the oldest and kept a contiguous window of the newest.
    assert disk_budget.HISTORY_MAX_LINES <= len(rows) < 600
    assert [row["observed_bytes"] for row in rows] == [MIB + index for index in range(600 - len(rows), 600)]
    assert path.stat().st_size <= footprints.HISTORY_COMPACTION_BYTES + 1024


def test_compaction_leaves_room_so_the_next_append_does_not_rewrite(tmp_path, monkeypatch):
    # Every sample is shorter than the line cap, so the newest HISTORY_MAX_LINES
    # always fit well under the threshold: compaction never repeats on each append.
    assert footprints.HISTORY_MAX_LINES * footprints._SAMPLE_MAX_BYTES < footprints.HISTORY_COMPACTION_BYTES
    ledger, compactions = tmp_path / "ledger", []
    compact = footprints._compact_history
    monkeypatch.setattr(footprints, "_compact_history", lambda *args: (compactions.append(1), compact(*args)))
    while not compactions:  # the longest workload label makes the longest lines
        assert disk_budget.record_footprint_sample(
            reservation_root=ledger, role="stage_replay", observed_bytes=GIB, reserved_bytes=4 * GIB,
            workload="w" * 64, baseline_bytes=GIB, fresh=True, device=2**40, duration_seconds=86_400.0)
    rows = (ledger / "history" / "stage_replay.jsonl").read_text().splitlines()
    assert len(rows) == footprints.HISTORY_MAX_LINES
    assert disk_budget.record_footprint_sample(
        reservation_root=ledger, role="stage_replay", observed_bytes=GIB, reserved_bytes=4 * GIB,
        workload="w" * 64)
    assert len(compactions) == 1


def test_live_reservations_filter_device_and_dead_pids(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    for name, device, pid, expires in (("a", 1, 11, 999.0), ("b", 2, 11, 999.0), ("c", 1, 12, 999.0), ("d", 1, 11, 1.0)):
        (ledger / f"{name}.json").write_text(json.dumps({"device": device, "pid": pid, "expected_bytes": GIB,
                                                         "expires_at_epoch": expires}))
    assert disk_budget.live_reservations(ledger, device=1, now=100.0, pid_alive=lambda pid: pid == 11) == (GIB, 1)
    assert sorted(p.name for p in ledger.glob("*.json")) == ["a.json", "b.json", "c.json", "d.json"]


def test_release_records_the_workspace_peak_in_unique_inodes(tmp_path):
    ledger, work = tmp_path / "ledger", tmp_path / "work" / "job-1"
    work.mkdir(parents=True)
    (work / "preexisting.bin").write_bytes(b"p" * 8192)  # part of the baseline
    reservation = reserve_control_plane_disk(
        "launch_activation", target_root=tmp_path, reservation_root=ledger, workspace=work,
        disk_usage=lambda _p: Usage(100 * GIB, 10 * GIB, 90 * GIB), now=lambda: 100.0,
        pid_alive=lambda _pid: True)
    (work / "new.bin").write_bytes(b"n" * 200_000)
    os.link(work / "new.bin", work / "new-link.bin")        # counted once
    reservation.release()
    rows = [json.loads(line) for line in (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()]
    assert rows[-1]["outcome"] == "completed"
    assert 200_000 <= rows[-1]["observed_bytes"] < 200_000 + 64 * 1024


def test_context_manager_marks_a_failed_run(tmp_path):
    ledger, work = tmp_path / "ledger", tmp_path / "job"
    with pytest.raises(RuntimeError):
        with reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
                                        workspace=work, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB),
                                        now=lambda: 1.0, pid_alive=lambda _pid: True):
            raise RuntimeError("boom")
    rows = (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()
    assert json.loads(rows[-1])["outcome"] == "failed"


def test_receipt_names_the_basis_and_sample_count(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 10)
    reservation = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0, pid_alive=lambda _pid: True)
    receipt = reservation.receipt()
    assert receipt["expected_bytes"] == 125 * MIB
    assert receipt["footprint_basis"] == "measured_p95" and receipt["footprint_sample_count"] == 10
    exact = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        expected_bytes=GIB, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0,
        pid_alive=lambda _pid: True)
    assert exact.receipt()["footprint_basis"] == "caller_exact"


def test_measurement_failure_never_breaks_release(tmp_path, monkeypatch):
    ledger, work = tmp_path / "ledger", tmp_path / "job"
    reservation = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        workspace=work, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0,
        pid_alive=lambda _pid: True)
    monkeypatch.setattr(disk_budget, "record_footprint_sample", lambda **_k: (_ for _ in ()).throw(OSError("full")))
    reservation.release()
    assert not reservation.path.exists()


def test_long_roles_outlive_the_default_ttl(tmp_path):
    ledger = tmp_path / "ledger"
    reservation = reserve_control_plane_disk("cpu_prestage", target_root=tmp_path, reservation_root=ledger,
        expected_bytes=GIB, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 0.0,
        pid_alive=lambda _pid: True)
    entry = json.loads(reservation.path.read_text())
    assert entry["expires_at_epoch"] == 12 * 3600


def test_headroom_refuses_roles_by_their_measured_footprint(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 10)
    def usage(_p):
        return Usage(100 * GIB, 0, 8 * GIB + GIB)  # 1 GiB above the 8 GiB floor

    headroom = disk_headroom(target_root=tmp_path, reservation_root=ledger, disk_usage=usage,
                             now=lambda: 1.0, pid_alive=lambda _pid: True)
    assert "launch_activation" not in headroom["refused_roles"]
    assert "launch_preparation" in headroom["refused_roles"]
    assert headroom["footprints"]["launch_activation"]["basis"] == "measured_p95"


def test_recording_never_waits_forever_on_a_held_ledger_lock(tmp_path, monkeypatch):
    # An evictor runs under the exclusive admission lock; if it released a
    # measured reservation, an unbounded wait here would stall every worker.
    import fcntl

    ledger, work = tmp_path / "ledger", tmp_path / "job"
    reservation = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        workspace=work, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0,
        pid_alive=lambda _pid: True)
    monkeypatch.setattr(footprints, "HISTORY_LOCK_WAIT_SECONDS", 0.2)
    with (ledger / ".lock").open("a+b") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        assert disk_budget.record_footprint_sample(
            reservation_root=ledger, role="launch_activation", observed_bytes=1, reserved_bytes=GIB) is False
        reservation.release()
    assert not reservation.path.exists()
    assert not (ledger / "history" / "launch_activation.jsonl").exists()


@pytest.mark.parametrize("outcome", ["failed", "blocked"])
def test_released_outcome_is_recorded_and_only_completed_shapes_admission(tmp_path, outcome):
    ledger = tmp_path / "ledger"
    for index in range(10):
        reservation = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
            workspace=tmp_path / f"job-{index}", disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB),
            now=lambda: 1.0, pid_alive=lambda _pid: True)
        reservation.release(outcome=outcome)
        reservation.release(outcome="completed")  # idempotent: the first outcome stands
    rows = [json.loads(line) for line in (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()]
    assert [row["outcome"] for row in rows] == [outcome] * 10
    measured = disk_budget.measured_footprint("launch_activation", reservation_root=ledger)
    assert (measured["basis"], measured["bytes"], measured["sample_count"]) == ("declared_default", 2 * GIB, 0)


def test_an_unknown_outcome_never_counts_as_completed(tmp_path):
    ledger = tmp_path / "ledger"
    assert disk_budget.record_footprint_sample(reservation_root=ledger, role="launch_activation",
        observed_bytes=1, reserved_bytes=GIB, outcome="finished") is False
    reservation = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        workspace=tmp_path / "job", disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0,
        pid_alive=lambda _pid: True)
    reservation.release(outcome="finished")
    [row] = [json.loads(line) for line in (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()]
    assert row["outcome"] == "failed"


def _roomy_reservation(tmp_path, ledger, **kwargs):
    return reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0, pid_alive=lambda _pid: True,
        **kwargs)


def _history_rows(ledger, role="launch_activation"):
    return [json.loads(line) for line in (ledger / "history" / f"{role}.jsonl").read_text().splitlines()]


def test_a_pass_over_an_already_full_workspace_records_resumed_not_completed(tmp_path):
    # A resumed pass starts from a workspace that already holds the run, so its
    # growth is near zero; counted as completed it would collapse the p95.
    ledger, work = tmp_path / "ledger", tmp_path / "job"
    work.mkdir()
    (work / "run.bin").write_bytes(b"r" * (footprints.FRESH_WORKSPACE_MAX_BYTES + 64 * 1024))
    reservation = _roomy_reservation(tmp_path, ledger, workspace=work)
    (work / "resume.log").write_bytes(b"l" * 4096)
    reservation.release()
    [row] = _history_rows(ledger)
    assert (row["outcome"], row["fresh"]) == ("resumed", False)
    assert row["baseline_bytes"] > footprints.FRESH_WORKSPACE_MAX_BYTES
    assert disk_budget.measured_footprint("launch_activation", reservation_root=ledger)["sample_count"] == 0


def test_a_fresh_workspace_records_completed_with_its_baseline(tmp_path):
    ledger = tmp_path / "ledger"
    reservation = _roomy_reservation(tmp_path, ledger, workspace=tmp_path / "absent-until-run")
    reservation.release()
    [row] = _history_rows(ledger)
    assert (row["outcome"], row["fresh"], row["baseline_bytes"]) == ("completed", True, 0)


def test_callers_can_assert_or_deny_freshness(tmp_path):
    ledger, work = tmp_path / "ledger", tmp_path / "job"
    work.mkdir()
    (work / "leftover.bin").write_bytes(b"x" * (footprints.FRESH_WORKSPACE_MAX_BYTES + 64 * 1024))
    cleared = _roomy_reservation(tmp_path, ledger)
    cleared.bind_workspace(work, fresh=True)  # e.g. measured right after the caller cleared it
    cleared.release()
    resumed = _roomy_reservation(tmp_path, ledger, workspace=tmp_path / "empty", fresh=False)
    resumed.release()
    assert [(row["outcome"], row["fresh"]) for row in _history_rows(ledger)] == [
        ("completed", True), ("resumed", False)]
    assert disk_budget.record_footprint_sample(reservation_root=ledger, role="launch_activation",
        observed_bytes=1, reserved_bytes=GIB, fresh=False)
    assert _history_rows(ledger)[-1]["outcome"] == "resumed"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-0000 directory anyway")
def test_an_unreadable_subtree_makes_the_measurement_incomplete(tmp_path):
    ledger, work = tmp_path / "ledger", tmp_path / "job"
    reservation = _roomy_reservation(tmp_path, ledger, workspace=work)
    hidden = work / "hidden"
    hidden.mkdir(parents=True)
    (hidden / "payload.bin").write_bytes(b"h" * 8192)
    hidden.chmod(0)
    try:
        reservation.release()
    finally:
        hidden.chmod(0o700)
    [row] = _history_rows(ledger)
    # Bytes under the unreadable directory were not counted; the sample must not count either.
    assert row["outcome"] == "incomplete"
    assert disk_budget.measured_footprint("launch_activation", reservation_root=ledger)["sample_count"] == 0


def test_the_footprint_is_the_largest_workload_p95_not_a_blend(tmp_path):
    # Two scene attempts among 48 small preparations vanish in a blended p95,
    # yet each needs its own 1.2 GiB.
    ledger = tmp_path / "ledger"
    for workload, value, count in (("prepared_references", 100 * MIB, 48),
                                   ("scene_preparation_attempt", 1200 * MIB, 2)):
        for _ in range(count):
            assert disk_budget.record_footprint_sample(reservation_root=ledger, role="launch_preparation",
                workload=workload, observed_bytes=value, reserved_bytes=2 * GIB)
    measured = disk_budget.measured_footprint("launch_preparation", reservation_root=ledger)
    assert (measured["basis"], measured["sample_count"]) == ("measured_p95", 50)
    assert measured["p95_bytes"] == 1200 * MIB and measured["bytes"] == 1500 * MIB


def test_a_job_that_declares_more_than_the_footprint_reserves_what_it_declares(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 10)  # measured: 125 MiB
    declared = _roomy_reservation(tmp_path, ledger, minimum_bytes=900 * MIB)
    assert declared.expected_bytes == 900 * MIB
    assert declared.receipt()["footprint_basis"] == "declared_minimum"
    measured = _roomy_reservation(tmp_path, ledger, minimum_bytes=10 * MIB)
    assert measured.expected_bytes == 125 * MIB
    assert measured.receipt()["footprint_basis"] == "measured_p95"


def test_workload_names_are_always_valid_labels():
    assert footprints.workload_name("activation", "native_task_arena_construction") == (
        "activation_native_task_arena_construction")
    odd = footprints.workload_name("activation", "Lane-With.Odd Chars/" + "x" * 80)
    assert disk_budget._ROLE_RE.fullmatch(odd) and odd.startswith("activation_lane_with_odd_chars")


# The systemd units whose jobs hold each role's reservations.  A job holds its
# reservation for at most its unit's start timeout, so the ledger entry's TTL
# must outlive that timeout or a running job's reservation is deleted as stale.
# control_plane_deploy is run by an operator (no unit), and
# result_artifact_download by the long-running intake service (no start timeout),
# and handoff_staging by the long-running listener (no start timeout).
ROLE_WORKER_UNITS = {
    "launch_preparation": ("blueprint-task-evaluation-launch-preparation.service",
                           "blueprint-task-evaluation-scene-progression.service"),
    "episode_compilation": ("blueprint-task-evaluation-episode-compilation.service",
                            "blueprint-task-evaluation-episode-compilation-remote.service"),
    "launch_activation": ("blueprint-task-evaluation-launch-activation.service",),
    "launch_dispatch": ("blueprint-task-evaluation-launch-dispatcher.service",),
    "policy_canary_dispatch": ("blueprint-task-evaluation-policy-canary-dispatcher.service",),
    "evidence_offload": ("blueprint-control-plane-storage-gc.service",),
    "experiment_restore": ("blueprint-control-plane-storage-gc.service",),
    "stage_replay": ("blueprint-agent-stage-replay.service",),
    "semantic_pretraining": ("blueprint-task-evaluation-launch-dispatcher.service",
                             "blueprint-task-evaluation-launch-activation.service"),
    "cpu_prestage": ("blueprint-task-evaluation-launch-dispatcher.service",),
    # Held from before the paid scene-configuration allocation until its
    # result is sealed, inside the dispatcher's allocator child.
    "scene_configuration_output": ("blueprint-task-evaluation-launch-dispatcher.service",),
    # Held from before the Quick-10 session's authority is consumed until its lane
    # seals, inside the canary dispatcher's allocator child (plan 15, review I3).
    "policy_canary_output": ("blueprint-task-evaluation-policy-canary-dispatcher.service",),
}


def _start_timeout_seconds(unit_text):
    [value] = re.findall(r"^TimeoutStartSec=(\S+)\s*$", unit_text, flags=re.MULTILINE)[-1:] or [None]
    match = re.fullmatch(r"(\d+)(h|min|m|s)?", value or "")
    assert match, f"unparseable TimeoutStartSec={value}"
    return int(match[1]) * {"h": 3600, "min": 60, "m": 60, "s": 1}[match[2] or "s"]


def test_every_role_ttl_outlives_its_worker_units_start_timeout():
    units = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
    # Cache fill runs in the fixed root tool, under its actual 4h long-work
    # controller; it is not attributed to an unrelated systemd worker.
    from blueprint_pipeline.control_plane_registered_checkpoint_cache import _LONG_WORK_SECONDS
    assert disk_budget.ROLE_TTL_SECONDS["g1_checkpoint_cache"] >= _LONG_WORK_SECONDS
    assert set(ROLE_WORKER_UNITS) | {"control_plane_deploy", "result_artifact_download", "handoff_staging", "g1_checkpoint_cache"} == set(
        disk_budget.ROLE_FOOTPRINT_BYTES)
    for role, names in ROLE_WORKER_UNITS.items():
        ttl = disk_budget.ROLE_TTL_SECONDS.get(role, disk_budget.DEFAULT_TTL_SECONDS)
        for name in names:
            timeout = _start_timeout_seconds((units / name).read_text(encoding="utf-8"))
            assert ttl >= timeout, f"{role} TTL {ttl}s < {name} TimeoutStartSec {timeout}s"


def test_a_symlinked_lock_is_refused_and_its_target_untouched(tmp_path):
    # The ledger directory is group-writable, so a runtime account could plant a
    # symlink there; root must never open, chmod or lock through it.
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    target = tmp_path / "root-owned-secret"
    target.write_text("secret")
    target.chmod(0o600)
    (ledger / ".lock").symlink_to(target)
    with pytest.raises(ControlPlaneDiskBudgetError, match="control_plane_disk_budget_lock_invalid"):
        _roomy_reservation(tmp_path, ledger, expected_bytes=GIB)
    with pytest.raises(ControlPlaneDiskBudgetError, match="control_plane_disk_budget_lock_invalid"):
        disk_headroom(target_root=tmp_path, reservation_root=ledger,
                      disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0)
    assert disk_budget.record_footprint_sample(reservation_root=ledger, role="launch_activation",
        observed_bytes=1, reserved_bytes=GIB) is False
    assert target.read_text() == "secret" and target.stat().st_mode & 0o777 == 0o600


def test_a_symlinked_history_directory_receives_nothing(tmp_path):
    ledger, elsewhere = tmp_path / "ledger", tmp_path / "elsewhere"
    ledger.mkdir()
    elsewhere.mkdir()
    (ledger / "history").symlink_to(elsewhere, target_is_directory=True)
    assert disk_budget.record_footprint_sample(reservation_root=ledger, role="launch_activation",
        observed_bytes=1, reserved_bytes=GIB) is False
    assert list(elsewhere.iterdir()) == []


def test_a_lock_owned_by_someone_else_with_a_wrong_mode_fails_closed(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    lock = ledger / ".lock"
    lock.touch()
    lock.chmod(0o640)
    monkeypatch.setattr(disk_budget.os, "geteuid", lambda: os.getuid() + 1)  # not the lock's owner
    with pytest.raises(ControlPlaneDiskBudgetError, match="control_plane_disk_budget_lock_mode_invalid:0640"):
        _roomy_reservation(tmp_path, ledger, expected_bytes=GIB)
    assert lock.stat().st_mode & 0o777 == 0o640


def test_a_declared_ceiling_below_the_floor_is_never_exceeded(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_LAUNCH_ACTIVATION_BYTES", str(32 * MIB))
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [1] * 10)
    measured = disk_budget.measured_footprint("launch_activation", reservation_root=ledger)
    assert (measured["basis"], measured["bytes"]) == ("measured_p95", 32 * MIB)


def test_a_short_write_is_not_a_recorded_sample_and_never_swallows_the_next(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger"
    real_write = footprints.os.write
    monkeypatch.setattr(footprints.os, "write", lambda fd, data: real_write(fd, data[: len(data) // 2]))
    assert disk_budget.record_footprint_sample(reservation_root=ledger, role="launch_activation",
        observed_bytes=7, reserved_bytes=GIB) is False  # a torn line is not success
    monkeypatch.setattr(footprints.os, "write", real_write)
    assert disk_budget.record_footprint_sample(reservation_root=ledger, role="launch_activation",
        observed_bytes=9, reserved_bytes=GIB) is True
    parsed = []
    for line in (ledger / "history" / "launch_activation.jsonl").read_text().splitlines():
        try:
            parsed.append(json.loads(line)["observed_bytes"])
        except ValueError:
            continue  # the torn fragment stays unreadable on its own line
    assert parsed == [9]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads mode-0000 entries anyway")
def test_an_unreadable_ledger_is_never_read_as_empty(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    entry = ledger / "held.json"
    entry.write_text(json.dumps({"device": 1, "pid": os.getpid(), "expected_bytes": GIB, "expires_at_epoch": 1e12}))
    entry.chmod(0)
    try:
        with pytest.raises(ControlPlaneDiskBudgetError, match="control_plane_disk_budget_ledger_unreadable"):
            disk_budget.live_reservations(ledger, device=1, now=1.0)
    finally:
        entry.chmod(0o640)
    ledger.chmod(0)
    try:
        with pytest.raises(ControlPlaneDiskBudgetError, match="control_plane_disk_budget_ledger_unreadable"):
            disk_budget.live_reservations(ledger, device=1, now=1.0)
    finally:
        ledger.chmod(0o770)
    assert disk_budget.live_reservations(tmp_path / "absent", device=1, now=1.0) == (0, 0)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads mode-0000 entries anyway")
def test_reservation_writer_refuses_an_unreadable_live_entry(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    entry = ledger / "held.json"
    entry.write_text(json.dumps({
        "device": tmp_path.stat().st_dev,
        "pid": os.getpid(),
        "expected_bytes": GIB,
        "expires_at_epoch": 1e12,
    }))
    entry.chmod(0)
    try:
        with pytest.raises(ControlPlaneDiskBudgetError, match="control_plane_disk_budget_ledger_unreadable"):
            reserve_control_plane_disk(
                "launch_activation", target_root=tmp_path, expected_bytes=GIB,
                reservation_root=ledger,
                disk_usage=lambda _path: Usage(100 * GIB, 60 * GIB, 40 * GIB),
                now=lambda: 100.0, pid_alive=lambda _pid: True,
            )
        assert entry.exists()
    finally:
        entry.chmod(0o640)


def test_admission_preserves_a_live_reservation_on_another_device(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    foreign_device = tmp_path.stat().st_dev + 1
    foreign = ledger / "foreign.json"
    foreign.write_text(json.dumps({
        "device": foreign_device,
        "pid": os.getpid(),
        "expected_bytes": 2 * GIB,
        "expires_at_epoch": 1e12,
    }))
    reservation = reserve_control_plane_disk(
        "launch_activation", target_root=tmp_path, expected_bytes=GIB,
        reservation_root=ledger,
        disk_usage=lambda _path: Usage(100 * GIB, 60 * GIB, 40 * GIB),
        now=lambda: 100.0, pid_alive=lambda _pid: True,
    )
    try:
        assert foreign.exists()
        assert disk_budget.live_reservations(
            ledger, device=foreign_device, now=100.0, pid_alive=lambda _pid: True,
        ) == (2 * GIB, 1)
    finally:
        reservation.release()


def test_scan_budget_exhaustion_cannot_complete_a_footprint(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_disk_usage as usage_module

    monkeypatch.setattr(usage_module, "MAX_TREE_SCAN_ENTRIES", 1)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ledger = tmp_path / "ledger"
    reservation = reserve_control_plane_disk(
        "launch_activation", target_root=tmp_path, expected_bytes=GIB,
        reservation_root=ledger, workspace=workspace,
        disk_usage=lambda _path: Usage(100 * GIB, 60 * GIB, 40 * GIB),
        now=lambda: 100.0, pid_alive=lambda _pid: True,
    )
    (workspace / "one.bin").write_bytes(b"x")
    (workspace / "two.bin").write_bytes(b"x")
    reservation.release(outcome="completed")
    history = ledger / "history" / "launch_activation.jsonl"
    [sample] = [json.loads(line) for line in history.read_text().splitlines()]
    assert sample["outcome"] == "incomplete"


def test_a_reservation_shrinks_in_place_and_never_grows_without_admission(tmp_path):
    """Review I3: a forecast hold taken before the paid run is resized after indexing. A shrink
    needs no admission; growth would be admitted against space other holds already reduced,
    so it is refused and the hold stands as it was."""
    ledger = tmp_path / "ledger"
    usage = lambda _path: Usage(100 * GIB, 0, 20 * GIB)  # noqa: E731
    hold = reserve_control_plane_disk("policy_canary_output", target_root=tmp_path, expected_bytes=GIB,
                                      reservation_root=ledger, disk_usage=usage, now=lambda: 100.0,
                                      pid_alive=lambda _pid: True, workload="quick10_needed_members")
    assert disk_budget.ROLE_FOOTPRINT_BYTES["policy_canary_output"] == GIB
    assert disk_budget.ROLE_TTL_SECONDS["policy_canary_output"] == 6 * 3600

    hold.resize(300 * 1024**2)

    assert hold.expected_bytes == 300 * 1024**2
    entry = json.loads(hold.path.read_text())
    assert entry["expected_bytes"] == 300 * 1024**2 and entry["token"] == hold.token
    reserved, count = disk_budget.live_reservations(ledger, device=hold.device, now=100.0,
                                                    pid_alive=lambda _pid: True)
    assert (reserved, count) == (300 * 1024**2, 1)
    with pytest.raises(ControlPlaneDiskBudgetError, match="^control_plane_disk_budget_resize_growth_refused$"):
        hold.resize(300 * 1024**2 + 1)
    assert json.loads(hold.path.read_text())["expected_bytes"] == 300 * 1024**2
    hold.release(outcome="completed")
    with pytest.raises(ControlPlaneDiskBudgetError, match="^control_plane_disk_budget_reservation_released$"):
        hold.resize(1)
