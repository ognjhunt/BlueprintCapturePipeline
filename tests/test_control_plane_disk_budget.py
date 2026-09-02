from __future__ import annotations

import json
from collections import namedtuple

import pytest

from blueprint_pipeline.control_plane_disk_budget import (
    ControlPlaneDiskBudgetError,
    disk_headroom,
    reserve_control_plane_disk,
)


Usage = namedtuple("Usage", "total used free")
GIB = 1024**3


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
