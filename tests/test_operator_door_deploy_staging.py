"""Fixed deploy scratch admission, path safety and retained capacity replay."""
# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/deploy_staging.py
from __future__ import annotations

from collections import namedtuple
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/operator-door"))
from operator_door import deploy_staging as staging  # noqa: E402
from blueprint_pipeline.control_plane_disk_budget import (  # noqa: E402
    ControlPlaneDiskBudgetError, reserve_control_plane_disk,
)


@pytest.fixture()
def trusted(tmp_path, monkeypatch):
    root = tmp_path / "host"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    for parts in (staging.WORK, staging.STAGING):
        path = root
        for part in parts:
            path /= part
            path.mkdir(exist_ok=True, mode=0o755)
            path.chmod(0o755)
    root_inode = root.stat().st_ino
    monkeypatch.setattr(staging, "ROOT", root)
    monkeypatch.setattr(staging, "ROOT_UID", os.getuid())
    monkeypatch.setattr(staging, "_service_groups", lambda: {os.getgid() + 1})
    monkeypatch.setattr(staging, "_device", lambda fd: 1 if os.fstat(fd).st_ino == root_inode else 2)
    return root, root.joinpath(*staging.STAGING)


def test_fixed_parent_is_readable_traversable_and_work_backed(trusted):
    _, parent = trusted
    assert staging.validated_staging_parent() == str(parent)
    assert parent.stat().st_mode & 0o005 == 0o005


def test_readback_can_traverse_a_root_owned_service_group_directory(trusted, monkeypatch):
    _, parent = trusted
    parent.chmod(0o750)
    monkeypatch.setattr(staging, "_service_groups", lambda: {parent.stat().st_gid})
    assert staging.validated_staging_parent() == str(parent)


def test_installer_creates_only_absent_fixed_leaf_and_preserves_existing(trusted):
    root, parent = trusted
    parent.rmdir()
    before = set(root.rglob("*"))
    assert staging.install_staging_parent() == str(parent)
    assert set(root.rglob("*")) - before == {parent}
    assert parent.stat().st_mode & 0o777 == 0o755
    inode = parent.stat().st_ino
    assert staging.install_staging_parent() == str(parent)
    assert parent.stat().st_ino == inode


def test_installer_leaves_unsafe_existing_directory_unchanged(trusted):
    _, parent = trusted
    parent.chmod(0o777)
    with pytest.raises(ValueError):
        staging.install_staging_parent()
    assert parent.stat().st_mode & 0o777 == 0o777


@pytest.mark.parametrize("case", ["symlink", "ancestor_symlink", "missing", "file", "owner",
                                  "group_write", "world_write", "no_service_traversal"])
def test_parent_cannot_be_redirected_or_modified_by_untrusted_user(trusted, monkeypatch, case):
    root, parent = trusted
    if case in {"symlink", "ancestor_symlink"}:
        selected = parent if case == "symlink" else parent.parent
        moved = selected.with_name(selected.name + "-original")
        selected.rename(moved)
        selected.symlink_to(moved, target_is_directory=True)
    elif case == "missing":
        parent.rmdir()
    elif case == "file":
        parent.rmdir()
        parent.write_text("not a directory")
    elif case == "owner":
        monkeypatch.setattr(staging, "ROOT_UID", os.getuid() + 1)
    else:
        parent.chmod({"group_write": 0o775, "world_write": 0o757,
                      "no_service_traversal": 0o700}[case])
    with pytest.raises((OSError, ValueError)):
        staging.validated_staging_parent()
    assert root.exists()


@pytest.mark.parametrize("devices", [(1, 1, 1), (1, 2, 3)])
def test_parent_requires_the_actual_work_volume_device(trusted, monkeypatch, devices):
    root, parent = trusted
    mapping = dict(zip((root.stat().st_ino, root.joinpath(*staging.WORK).stat().st_ino,
                        parent.stat().st_ino), devices))
    monkeypatch.setattr(staging, "_device", lambda fd: mapping[os.fstat(fd).st_ino])
    with pytest.raises(ValueError, match="wrong_device"):
        staging.validated_staging_parent()


def test_cli_never_accepts_a_path_override(trusted, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["deploy_staging.py", "/caller/path"])
    assert staging.main() == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("full_copy_reason", ["reuse_absent", "reuse_refused"])
def test_retained_full_copy_capacity_case_preserves_reservation(trusted, tmp_path, full_copy_reason):
    root, parent = trusted
    expected = 11_483_929_282
    Usage = namedtuple("Usage", "total used free")
    release = root / "opt/releases"
    release.mkdir(parents=True)
    def device(path):
        return 2 if Path(path).is_relative_to(parent) else 1
    def usage(path):
        total, free = ((316_464_824_320, 51_410_169_856) if device(path) == 2
                       else (165_295_407_104, 8_808_755_200))
        return Usage(total, total - free, free)
    options = dict(reservation_root=tmp_path / "ledger", disk_usage=usage, device_of=device)
    first = reserve_control_plane_disk("control_plane_deploy", target_root=release,
                                      expected_bytes=524_806_723, **options)
    try:
        with pytest.raises(ControlPlaneDiskBudgetError, match="available_bytes=6630994406"):
            reserve_control_plane_disk("control_plane_deploy", target_root=release,
                                       expected_bytes=expected, **options)
        with reserve_control_plane_disk("control_plane_deploy", target_root=parent,
                                        expected_bytes=expected, workload=full_copy_reason, **options) as admitted:
            assert admitted.receipt()["expected_bytes"] == expected
            assert admitted.receipt()["footprint_basis"] == "caller_exact"
        with pytest.raises(ControlPlaneDiskBudgetError):
            reserve_control_plane_disk("control_plane_deploy", target_root=parent,
                                       expected_bytes=expected,
                                       **{**options, "disk_usage": lambda _: Usage(316_464_824_320, 0, 5_000_000_000)})
    finally:
        first.release(outcome="blocked")
