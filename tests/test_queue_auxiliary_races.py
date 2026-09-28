# Covers (for impacted-test selection): src/blueprint_pipeline/control_plane_queue_auxiliary_observation.py src/blueprint_pipeline/control_plane_queue_observation.py
"""Retained descriptors and global final checks observe change without fencing."""
import os
import errno
import stat

import pytest

from blueprint_pipeline import control_plane_queue_auxiliary_observation as auxiliary
from blueprint_pipeline import control_plane_queue_observation as primary
from tests.test_queue_auxiliary_layouts import CHILD, HEX, OTHER, STEM, observe, root_for, write


@pytest.mark.parametrize("target", ["root", "role", "group", "row"])
def test_earlier_root_changes_during_later_root_reads(tmp_path, monkeypatch, target):
    first = root_for(tmp_path, "sam", "a-first")
    later = root_for(tmp_path, "sam", "z-later")
    row = write(first, f"progress/{CHILD}/000001.json")
    write(later, f"started/{CHILD}.json")
    original = auxiliary._AuxScan.observe_aux_root
    def observe_root(self, contract):
        if contract.root_path == str(later):
            if target == "row":
                row.write_text('{"changed":true}')
            elif target == "group":
                (row.parent / "000002.json").write_text('{}')
            elif target == "role":
                (first / "started" / f"{CHILD}.json").write_text('{}')
            else:
                first.rename(tmp_path / "retained-old")
                root_for(tmp_path, "sam", "a-first")
        return original(self, contract)
    monkeypatch.setattr(auxiliary._AuxScan, "observe_aux_root", observe_root)
    result = observe(("sam", first), ("sam", later))
    assert not result.complete and result.blockers


@pytest.mark.parametrize("target", ["root", "role", "group", "row", "ancestor"])
def test_symlinks_are_not_followed(tmp_path, target):
    root = root_for(tmp_path, "preparation")
    outside = tmp_path / "outside"
    outside.mkdir()
    write(outside, "secret.json")
    row = write(root, f"source-progress/{STEM}/000001-{HEX}.json")
    if target == "root":
        root.rename(tmp_path / "old-root")
        root.symlink_to(outside, target_is_directory=True)
    elif target == "role":
        (root / "source-progress").rename(root / "retained-progress")
        (root / "source-progress").symlink_to(outside, target_is_directory=True)
    elif target == "group":
        row.parent.rename(root / "retained-group")
        row.parent.symlink_to(outside, target_is_directory=True)
    elif target == "ancestor":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        root = alias / root.name
    else:
        row.unlink()
        row.symlink_to(outside / "secret.json")
    result = observe(("preparation", root))
    assert not result.complete and result.rows == ()
    assert (outside / "secret.json").read_text() == '{"retained":true}'


@pytest.mark.parametrize("relative", ["results/conflicts", f"source-progress/{STEM}"])
def test_role_or_group_replacement_detected_after_read(tmp_path, monkeypatch, relative):
    root = root_for(tmp_path, "preparation")
    row = (f"{relative}/{STEM}-{OTHER}.json" if relative == "results/conflicts"
           else f"{relative}/000001-{HEX}.json")
    write(root, row)
    original = auxiliary._AuxScan.verify_aux_root
    def verify(self, snapshot):
        directory = root / relative
        directory.rename(root / "retained-directory")
        directory.mkdir()
        return original(self, snapshot)
    monkeypatch.setattr(auxiliary._AuxScan, "verify_aux_root", verify)
    result = observe(("preparation", root))
    assert not result.complete and "queue_directory_changed" in result.blockers


def test_missing_role_appearing_before_final_pass_is_still_unknown(tmp_path, monkeypatch):
    root = root_for(tmp_path, "sam")
    (root / "started").rmdir()
    original = auxiliary._AuxScan.verify_aux_root
    def verify(self, snapshot):
        (root / "started").mkdir()
        return original(self, snapshot)
    monkeypatch.setattr(auxiliary._AuxScan, "verify_aux_root", verify)
    result = observe(("sam", root))
    assert "queue_directory_changed" in result.blockers
    assert "auxiliary_directory_missing_unproven" in result.blockers


def test_row_read_denial_keeps_other_positive(tmp_path, monkeypatch):
    root = root_for(tmp_path, "sam")
    write(root, f"results/{CHILD}.json")
    write(root, f"started/{CHILD}.json")
    original = primary._Scan.read_row
    def read(self, root, state, directory, name):
        if state == "results":
            raise PermissionError("private error text")
        return original(self, root, state, directory, name)
    monkeypatch.setattr(primary._Scan, "read_row", read)
    result = observe(("sam", root))
    assert not result.complete and len(result.rows) == 1
    assert result.blockers == ("queue_row_unavailable",)


def test_descriptor_close_fault_is_bounded_and_owned_handle_retried(tmp_path, monkeypatch):
    root = root_for(tmp_path, "sam")
    original = os.close
    failed = None
    calls = 0
    def close(fd):
        nonlocal failed, calls
        if failed is None:
            failed = fd
            calls += 1
            raise OSError("injected close")
        if fd == failed:
            calls += 1
        return original(fd)
    monkeypatch.setattr(os, "close", close)
    result = observe(("sam", root))
    assert not result.complete and "queue_descriptor_close_failed" in result.blockers
    assert calls == 2
    with pytest.raises(OSError):
        os.fstat(failed)


@pytest.mark.parametrize("fault", ["fstat", "clock", "read"])
def test_acquisition_or_read_fault_with_close_failure_does_not_leak(tmp_path, monkeypatch, fault):
    root = root_for(tmp_path, "sam")
    write(root, f"started/{CHILD}.json")
    opened, fstat, close, read = os.open, os.fstat, os.close, os.read
    handles = set()
    stat_failed = close_failed = False
    def track(*args, **kwargs):
        fd = opened(*args, **kwargs)
        handles.add(fd)
        return fd
    def injected_stat(fd):
        nonlocal stat_failed
        if fault == "fstat" and not stat_failed:
            stat_failed = True
            raise OSError(errno.EIO, "private")
        return fstat(fd)
    def injected_close(fd):
        nonlocal close_failed
        if not close_failed:
            close_failed = True
            raise OSError(errno.EIO, "one shot")
        close(fd)
        handles.discard(fd)
    def injected_read(*args):
        if fault == "read":
            raise OSError(errno.EIO, "private")
        return read(*args)
    monkeypatch.setattr(os, "open", track)
    monkeypatch.setattr(os, "fstat", injected_stat)
    monkeypatch.setattr(os, "close", injected_close)
    monkeypatch.setattr(os, "read", injected_read)
    result = observe(("sam", root), monotonic=lambda: 6 if fault == "clock" and handles else 0)
    assert not result.complete and not handles
    assert "queue_descriptor_close_failed" in result.blockers


def test_ambiguous_close_foreign_reuse_is_preserved(tmp_path, monkeypatch):
    root = root_for(tmp_path, "sam")
    foreign_path = tmp_path / "foreign"
    foreign_path.write_text("owned elsewhere")
    opened, closed, fstat = os.open, os.close, os.fstat
    owned = []
    foreign = []
    def track(*args, **kwargs):
        fd = opened(*args, **kwargs)
        owned.append(fd)
        return fd
    def close(fd):
        if not foreign:
            closed(fd)
            new = opened(foreign_path, os.O_RDONLY)
            if new != fd:
                os.dup2(new, fd)
                closed(new)
            foreign.append(fd)
            raise OSError(errno.EINTR, "ambiguous")
        return closed(fd)
    monkeypatch.setattr(os, "open", track)
    monkeypatch.setattr(os, "close", close)
    try:
        result = observe(("sam", root), monotonic=lambda: 6 if owned else 0)
        assert "queue_descriptor_changed" in result.blockers
        assert stat.S_ISREG(fstat(foreign[0]).st_mode)
    finally:
        for fd in foreign:
            closed(fd)
