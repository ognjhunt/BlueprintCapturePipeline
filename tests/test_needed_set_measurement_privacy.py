"""ADP-081 day-42 compatibility guard: report bytes do not enroll new readers."""

from __future__ import annotations

import fcntl
import json
import os
import stat
from types import SimpleNamespace

import pytest

from blueprint_pipeline import policy_canary_output_members as members


def _record(*, size=1):
    return members.seal_needed_set_measurement(
        contract=members.CONTRACT_VERSION, materialized_members=12, materialized_bytes=size,
        archive={"name": "synthetic-private-task.zip", "size_bytes": 100,
                 "sha256": "sha256:" + "a" * 64, "members": 12},
        quick10_shape=members.QUICK10_SHAPE, measured_at="2026-10-06T00:00:00+00:00")


def _preserved_metadata(info):
    # Reads may update atime. Identity, rights and write metadata must not change.
    return (info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _canonical(tmp_path, monkeypatch):
    state = tmp_path / "installed-state"
    state.mkdir(mode=0o750)
    state.chmod(0o750)
    path = state / "policy-canary-output" / "needed-set-measurement.v1.json"
    monkeypatch.setattr(members, "_CANONICAL_MEASUREMENT_PATH", path)
    monkeypatch.setattr(members, "MEASUREMENT_PATH", path)
    info = state.stat()
    uid, gid = info.st_uid, info.st_gid  # BSD may inherit a parent GID unlike getegid().

    def user(name):
        assert name == "blueprint"
        return SimpleNamespace(pw_uid=uid)

    def group(name):
        assert name == "blueprint"
        return SimpleNamespace(gr_gid=gid)

    # Synthetic exact-role metadata; no account creation, setuid or enrollment.
    monkeypatch.setattr(members.pwd, "getpwnam", user)
    monkeypatch.setattr(members.grp, "getgrnam", group)
    return state, path


def test_custom_mapping_and_every_created_directory_are_owner_only(tmp_path):
    target = tmp_path / "custom" / "nested" / "measurement.json"
    payload = {"capability": "synthetic-secret", "archive": {"name": "private-task-name.zip"}}
    old_umask = os.umask(0)
    try:
        assert members.write_needed_set_measurement(payload, target) == target
    finally:
        os.umask(old_umask)
    assert json.loads(target.read_text()) == payload
    for path in [target.parent.parent, *target.parent.parent.rglob("*")]:
        info = path.lstat()
        assert stat.S_IMODE(info.st_mode) == (0o700 if path.is_dir() else 0o600)
        assert info.st_mode & 0o077 == 0  # Synthetic non-owners, including inherited GID, cannot read.
        if path.is_file():
            assert info.st_nlink == 1
    assert members.needed_set_measurement_refusal(target) == members.AUTO_NEEDED_SET_RECORD_INVALID


def test_redirected_default_does_not_grant_canonical_readership(tmp_path, monkeypatch):
    target = tmp_path / "custom-state" / "needed-set-measurement.v1.json"
    monkeypatch.setattr(members, "MEASUREMENT_PATH", target)
    members.write_needed_set_measurement(_record())
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert members.needed_set_measurement_refusal() is None


def test_exact_installed_canonical_role_preserves_dispatcher_read_contract(tmp_path, monkeypatch):
    state, target = _canonical(tmp_path, monkeypatch)
    record = _record()
    members.write_needed_set_measurement(record)
    assert json.loads(target.read_text()) == record
    assert stat.S_IMODE(state.stat().st_mode) == 0o750
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert members.needed_set_measurement_refusal() is None
    members.write_needed_set_measurement(_record(size=members.NEEDED_SET_BUDGET_BYTES + 1))
    assert members.needed_set_measurement_refusal() == members.AUTO_NEEDED_SET_OVER_BUDGET


@pytest.mark.parametrize("change", ["public_state", "wrong_group", "wrong_owner", "missing_account"])
def test_canonical_spelling_without_installed_role_refuses(tmp_path, monkeypatch, change):
    state, target = _canonical(tmp_path, monkeypatch)
    target.parent.mkdir()
    target.write_text("existing incumbent")
    target.chmod(0o644)
    if change == "public_state":
        state.chmod(0o755)
    elif change == "wrong_group":
        wrong_gid = state.stat().st_gid + 10000
        monkeypatch.setattr(members.grp, "getgrnam", lambda _: SimpleNamespace(gr_gid=wrong_gid))
    elif change == "wrong_owner":
        wrong_uid = state.stat().st_uid + 10000
        monkeypatch.setattr(members.pwd, "getpwnam", lambda _: SimpleNamespace(pw_uid=wrong_uid))
    else:
        def missing(_):
            raise KeyError("synthetic missing account")
        monkeypatch.setattr(members.pwd, "getpwnam", missing)
    before = _preserved_metadata(target.stat()), _preserved_metadata(state.stat())
    with pytest.raises(OSError):
        members.write_needed_set_measurement(_record())
    assert target.read_text() == "existing incumbent"
    assert (_preserved_metadata(target.stat()), _preserved_metadata(state.stat())) == before


def test_arbitrary_mapping_cannot_be_shared_via_canonical_destination(tmp_path, monkeypatch):
    _, target = _canonical(tmp_path, monkeypatch)
    with pytest.raises(OSError):
        members.write_needed_set_measurement({"private_operator_note": "synthetic"})
    assert not target.exists() and not target.parent.exists()


def test_canonical_reader_role_does_not_enroll_an_unregistered_writer(tmp_path, monkeypatch):
    _, target = _canonical(tmp_path, monkeypatch)
    uid = os.geteuid()
    monkeypatch.setattr(members.os, "geteuid", lambda: uid + 10000)
    with pytest.raises(OSError):
        members.write_needed_set_measurement(_record())
    assert not target.parent.exists()


@pytest.mark.parametrize("change", ["symlink", "hardlink", "shared_mode", "foreign_owner"])
def test_unsafe_leaf_is_preserved_without_chmod_or_overwrite(tmp_path, monkeypatch, change):
    source = tmp_path / "unrelated-source.json"
    source.write_text("unrelated synthetic data")
    source.chmod(0o600)
    target = tmp_path / "measurement.json"
    if change == "symlink":
        target.symlink_to(source)
    elif change == "hardlink":
        os.link(source, target)
    else:
        target.write_text("unrelated incumbent")
        target.chmod(0o644 if change == "shared_mode" else 0o600)
    before = _preserved_metadata(target.lstat()), _preserved_metadata(source.stat())
    if change == "foreign_owner":
        real_stat = members.os.stat

        def foreign(name, *args, **kwargs):
            info = real_stat(name, *args, **kwargs)
            if name == target.name and kwargs.get("dir_fd") is not None:
                values = {key: getattr(info, key) for key in ("st_dev", "st_ino", "st_uid", "st_gid",
                          "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")}
                return SimpleNamespace(**{**values, "st_uid": info.st_uid + 10000})
            return info

        monkeypatch.setattr(members.os, "stat", foreign)
    with pytest.raises(OSError):
        members.write_needed_set_measurement(_record(), target)
    assert (_preserved_metadata(target.lstat()), _preserved_metadata(source.stat())) == before
    assert target.read_text() == ("unrelated synthetic data" if change in ("symlink", "hardlink") else "unrelated incumbent")
    assert source.read_text() == "unrelated synthetic data"


@pytest.mark.parametrize("change", ["symlink", "group_writable", "world_writable"])
def test_unadmitted_parent_is_not_followed_or_repaired(tmp_path, change):
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = tmp_path / "unadmitted"
    if change == "symlink":
        parent.symlink_to(outside, target_is_directory=True)
    else:
        parent.mkdir()
        parent.chmod(0o770 if change == "group_writable" else 0o777)
    before = _preserved_metadata(parent.lstat())
    with pytest.raises(OSError):
        members.write_needed_set_measurement(_record(), parent / "measurement.json")
    assert _preserved_metadata(parent.lstat()) == before
    assert list(outside.iterdir()) == []
    if change != "symlink":
        assert list(parent.iterdir()) == []


def test_competing_incumbent_during_staging_is_preserved(tmp_path, monkeypatch):
    target = tmp_path / "measurement.json"
    members.write_needed_set_measurement(_record(), target)
    competitor = tmp_path / "competitor.json"
    competitor.write_text("competing synthetic incumbent")
    competitor.chmod(0o600)
    replacement = competitor.stat()
    real_fsync = members.os.fsync
    inserted = False

    def replace_during_staging(descriptor):
        nonlocal inserted
        if stat.S_ISREG(os.fstat(descriptor).st_mode) and not inserted:
            inserted = True
            os.replace(competitor, target)
        return real_fsync(descriptor)

    monkeypatch.setattr(members.os, "fsync", replace_during_staging)
    with pytest.raises(OSError):
        members.write_needed_set_measurement(_record(size=2), target)
    assert inserted
    assert target.read_text() == "competing synthetic incumbent"
    assert target.stat().st_ino == replacement.st_ino
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert members.needed_set_measurement_refusal(target) == members.AUTO_NEEDED_SET_RECORD_INVALID


def test_exclusive_publication_preserves_late_new_incumbent(tmp_path, monkeypatch):
    target = tmp_path / "measurement.json"
    real_link = members.os.link

    def create_before_link(source, name, **kwargs):
        target.write_text("late synthetic incumbent")
        target.chmod(0o600)
        return real_link(source, name, **kwargs)

    monkeypatch.setattr(members.os, "link", create_before_link)
    with pytest.raises(FileExistsError):
        members.write_needed_set_measurement(_record(), target)
    assert target.read_text() == "late synthetic incumbent"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_directory_retarget_does_not_escape_or_replace_any_incumbent(tmp_path, monkeypatch):
    parent, retired, outside = tmp_path / "private", tmp_path / "retired", tmp_path / "outside"
    outside.mkdir()
    target = parent / "measurement.json"
    members.write_needed_set_measurement(_record(), target)
    before = target.read_bytes()
    real_fsync = members.os.fsync
    retargeted = False

    def retarget_during_staging(descriptor):
        nonlocal retargeted
        if stat.S_ISREG(os.fstat(descriptor).st_mode) and not retargeted:
            retargeted = True
            parent.rename(retired)
            parent.symlink_to(outside, target_is_directory=True)
        return real_fsync(descriptor)

    monkeypatch.setattr(members.os, "fsync", retarget_during_staging)
    with pytest.raises(OSError):
        members.write_needed_set_measurement(_record(size=2), target)
    assert retargeted and list(outside.iterdir()) == []
    assert (retired / target.name).read_bytes() == before


def test_atomic_replacement_preserves_old_and_new_gate_truth(tmp_path, monkeypatch):
    target = tmp_path / "measurement.json"
    members.write_needed_set_measurement(_record(), target)
    old_inode = target.stat().st_ino
    real_replace = members.os.replace
    observations = []

    def observed_replace(source, name, **kwargs):
        observations.append(members.needed_set_measurement_refusal(target))
        result = real_replace(source, name, **kwargs)
        observations.append(members.needed_set_measurement_refusal(target))
        return result

    monkeypatch.setattr(members.os, "replace", observed_replace)
    monkeypatch.setattr(members.os, "chmod", lambda *a, **kw: pytest.fail("published path chmod forbidden"))
    members.write_needed_set_measurement(_record(size=members.NEEDED_SET_BUDGET_BYTES + 1), target)
    assert observations == [None, members.AUTO_NEEDED_SET_OVER_BUDGET]
    assert target.stat().st_ino != old_inode
    assert target.stat().st_nlink == 1 and stat.S_IMODE(target.stat().st_mode) == 0o600


def test_interrupted_writer_preserves_old_record_and_stable_locked_inode(tmp_path, monkeypatch):
    target = tmp_path / "measurement.json"
    members.write_needed_set_measurement(_record(), target)
    before = target.read_bytes()
    lock = tmp_path / f".{target.name}.write.lock"
    lock_inode = lock.stat().st_ino

    def interrupted(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            competing = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(competing)
            raise OSError("synthetic interrupted write")

    monkeypatch.setattr(members.os, "fsync", interrupted)
    with pytest.raises(OSError, match="synthetic interrupted"):
        members.write_needed_set_measurement(_record(size=2), target)
    assert target.read_bytes() == before
    assert members.needed_set_measurement_refusal(target) is None
    assert lock.stat().st_ino == lock_inode and stat.S_IMODE(lock.stat().st_mode) == 0o600
