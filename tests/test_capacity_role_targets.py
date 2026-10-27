"""Beta scene-to-screening: capacity alerts follow the workers' actual devices."""

from collections import namedtuple

import pytest

from blueprint_pipeline import control_plane_capacity_controller as capacity
from blueprint_pipeline import control_plane_disk_budget as budget

GIB = 1024 ** 3
Usage = namedtuple("Usage", "total used free")


@pytest.fixture
def mounts(tmp_path, monkeypatch):
    root, work = tmp_path / "root", tmp_path / "work"
    root.mkdir()
    work.mkdir()
    target = root / "canaries-bind-mount"
    target.mkdir()
    # An actual bind mount keeps its source device despite its path spelling.
    def device(path):
        return 2 if path == target or path.is_relative_to(work) else 1

    def usage(path):
        free = 50 * GIB if device(path) == 2 else 9 * GIB
        return Usage(150 * GIB, 150 * GIB - free, free)

    monkeypatch.setattr(budget, "target_device", device)
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_ROLE_TARGETS",
                       ",".join(f"{role}={target}" for role in capacity.CHAIN_ROLES))
    return root, work, target, usage


def test_crowded_root_does_not_refuse_roles_on_a_healthy_bind_mount(tmp_path, mounts):
    root, work, _target, usage = mounts
    report = capacity.build_capacity_report(
        mounts=[root, work], reservation_root=tmp_path / "ledger", disk_usage=usage, now=1,
    )
    rows = {row["mount"]: row for row in report["mounts"]}
    assert rows[str(root)]["assigned_roles"] == []
    assert rows[str(root)]["refused_roles"] == []
    assert rows[str(work)]["assigned_roles"] == sorted(capacity.CHAIN_ROLES)
    assert rows[str(work)]["refused_roles"] == []
    assert not any(row["code"] == "admission_refused" for row in report["alerts"])
    assert any(row["code"] == "utilization_critical" for row in report["alerts"])
    assert capacity.capacity_summary(report)["mounts"][0]["assigned_roles"] == []


def test_a_role_that_really_uses_the_crowded_disk_still_refuses(tmp_path, mounts, monkeypatch):
    root, work, target, usage = mounts
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_ROLE_TARGETS", ",".join(
        f"{role}={root if role == 'launch_dispatch' else target}" for role in capacity.CHAIN_ROLES
    ))
    report = capacity.build_capacity_report(
        mounts=[root, work], reservation_root=tmp_path / "ledger", disk_usage=usage, now=1,
    )
    alerts = [row for row in report["alerts"] if row["code"] == "admission_refused"]
    assert [(row["mount"], row["roles"]) for row in alerts] == [(str(root), ["launch_dispatch"])]


@pytest.mark.parametrize("setting", ["launch_dispatch=relative", "unknown=/tmp"])
def test_invalid_targets_remain_a_capacity_failure(tmp_path, monkeypatch, setting):
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_ROLE_TARGETS", setting)
    report = capacity.build_capacity_report(
        mounts=[tmp_path], reservation_root=tmp_path / "ledger",
        disk_usage=lambda _path: Usage(150 * GIB, 50 * GIB, 100 * GIB), now=1,
    )
    assert report["level"] == "critical"
    assert report["mounts"][0]["status"] == "configuration_invalid"
