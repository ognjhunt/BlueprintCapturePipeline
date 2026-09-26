from __future__ import annotations

import json
from collections import namedtuple

import pytest

from blueprint_pipeline import control_plane_disk_budget as disk_budget
from blueprint_pipeline.control_plane_disk_budget import (
    ControlPlaneDiskBudgetError,
    disk_headroom,
    reserve_control_plane_disk,
)


Usage = namedtuple("Usage", "total used free")
GIB = 1024**3


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

    monkeypatch.setattr(disk_budget.os, "chmod", reject_lock_chmod)

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
    original_chmod = disk_budget.os.chmod

    def reject_lock_chmod(path, mode, **kwargs) -> None:
        if path == lock:
            raise PermissionError("runtime account cannot chmod root-owned lock")
        original_chmod(path, mode, **kwargs)

    monkeypatch.setattr(disk_budget.os, "chmod", reject_lock_chmod)

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
    assert report["status"] == "exhausted"
    assert set(report["refused_roles"]) == {
        "control_plane_deploy",
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
    }
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


def test_history_is_compacted(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [MIB] * 400)
    lines = (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()
    assert len(lines) <= 400 and len(lines) >= disk_budget.HISTORY_MAX_LINES


def test_live_reservations_filter_device_and_dead_pids(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    for name, device, pid, expires in (("a", 1, 11, 999.0), ("b", 2, 11, 999.0), ("c", 1, 12, 999.0), ("d", 1, 11, 1.0)):
        (ledger / f"{name}.json").write_text(json.dumps({"device": device, "pid": pid, "expected_bytes": GIB,
                                                         "expires_at_epoch": expires}))
    assert disk_budget.live_reservations(ledger, device=1, now=100.0, pid_alive=lambda pid: pid == 11) == (GIB, 1)
    assert sorted(p.name for p in ledger.glob("*.json")) == ["a.json", "b.json", "c.json", "d.json"]
