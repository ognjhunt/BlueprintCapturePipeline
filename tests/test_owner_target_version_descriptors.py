"""ADP-009D/day28: scoped acquisitions preserve unknown foreign descriptors."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_owner_target_io.py
#   src/blueprint_pipeline/control_plane_lane_owner_target_versions.py

import errno
import hashlib
import json
import os
import stat

import pytest

from blueprint_pipeline import control_plane_lane_scratch as scratch
from blueprint_pipeline import control_plane_scratch_lifetime as lifetime
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def tuple_identity(path, *, file=False):
    info = path.stat()
    value = dict(dev=info.st_dev, ino=info.st_ino, type="regular" if file else "directory")
    if file:
        value.update(mode=stat.S_IMODE(info.st_mode), uid=info.st_uid, gid=info.st_gid,
                     nlink=info.st_nlink, size_bytes=info.st_size,
                     mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)
    return value


@pytest.fixture
def enrolled(tmp_path):
    root = tmp_path / "lanes"
    root.mkdir()
    path = scratch.create_lane_scratch("lane", "cache", root=root, owner="owner", run_ref="run1",
        ttl_seconds=100, cleanup="owner_review", reason="cache", class_intent="cache",
        size_budget_bytes=8192, now=lambda: 1000, consumer_lifetime_contract=lifetime.PROTOCOL)
    raw = (path / scratch.LEASE_FILE).read_bytes()
    lease = json.loads(raw)
    expected = dict(root_identity=tuple_identity(root), lane_identity=tuple_identity(path.parent),
        folder_identity=tuple_identity(path), lease_file_identity=tuple_identity(path / scratch.LEASE_FILE, file=True),
        lease_raw_sha256="sha256:" + hashlib.sha256(raw).hexdigest(), lease_raw_size_bytes=len(raw),
        lease_digest=lease["lease_digest"], lane="lane", name="cache",
        lease={key: lease[key] for key in ("owner", "reason", "class_intent", "cleanup",
            "consumer_lifetime_contract", "created_at_epoch", "expires_at_epoch", "released_at_epoch", "size_budget_bytes")}
              | dict(reference_kind="run_ref", reference_value="run1", renewed_at_epoch=1000))
    return path, expected


def files(expected=None, monotonic=lambda: 0):
    from blueprint_pipeline.control_plane_lane_owner_target_io import _TargetFiles
    return _TargetFiles(ReferenceCollectionBudget(monotonic=monotonic, values_limit=10000), expected=expected)


def test_protected_open_proves_named_identity_before_first_fd(tmp_path, monkeypatch):
    a, foreign = tmp_path / "a", tmp_path / "foreign"
    a.write_bytes(b"a")
    foreign.write_bytes(b"foreign")
    real_open, real_stat = os.open, os.stat
    fd = real_open(foreign, os.O_RDONLY)
    calls = []
    def opened(name, flags, *args, **kwargs):
        calls.append("open")
        return fd
    def named(*args, **kwargs):
        calls.append("stat")
        return real_stat(*args, **kwargs)
    owner = files()
    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "stat", named)
    try:
        with pytest.raises(ValueError, match="owner_target_descriptor_ownership_unproven"):
            owner.open(a, os.O_RDONLY | os.O_NOFOLLOW)
        assert calls == ["stat", "open"]
        assert os.fstat(fd).st_ino == foreign.stat().st_ino
        assert owner.unresolved and not owner.owned
    finally:
        os.close(fd)


def test_initial_identity_failure_never_closes_or_adopts(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.write_bytes(b"a")
    real_fstat, real_open = os.fstat, os.open
    foreign = real_open(a, os.O_RDONLY)
    owner = files()
    monkeypatch.setattr(os, "open", lambda *a, **kw: foreign)
    monkeypatch.setattr(os, "fstat", lambda fd: (_ for _ in ()).throw(OSError(errno.EIO, "private")))
    try:
        with pytest.raises(ValueError, match="owner_target_descriptor_ownership_unproven"):
            owner.open(a, os.O_RDONLY)
        assert not owner.owned and owner.unresolved
        assert real_fstat(foreign).st_ino == a.stat().st_ino
    finally:
        os.close(foreign)


def test_known_reused_token_is_not_closed_and_other_handles_finish(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    owner = files()
    first = owner.open(a, os.O_RDONLY)
    other = owner.open(a, os.O_RDONLY)
    replacement = os.open(b, os.O_RDONLY)
    os.dup2(replacement, first)
    os.close(replacement)
    try:
        with pytest.raises(ValueError, match="owner_target_descriptor_ownership_unproven"):
            owner.finish()
        assert os.fstat(first).st_ino == b.stat().st_ino
        with pytest.raises(OSError):
            os.fstat(other)
    finally:
        os.close(first)


def test_parent_reuse_refuses_before_any_child_lookup(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    owner = files()
    parent = owner.open(root, os.O_RDONLY | os.O_DIRECTORY)
    replacement = os.open(other, os.O_RDONLY | os.O_DIRECTORY)
    os.dup2(replacement, parent)
    os.close(replacement)
    monkeypatch.setattr(os, "stat", lambda *a, **kw: pytest.fail("foreign-parent child lookup"))
    try:
        with pytest.raises(ValueError, match="owner_target_descriptor_changed"):
            owner.open("secret", os.O_RDONLY, parent=parent)
    finally:
        owner.close(parent)
        os.close(parent)


def test_close_identity_and_io_retry_are_clock_independent(tmp_path, monkeypatch):
    path = tmp_path / "a"
    path.write_bytes(b"a")
    owner = files()
    fd = owner.open(path, os.O_RDONLY)
    real_close = os.close
    attempts = []
    def close(value):
        attempts.append(value)
        if len(attempts) == 1:
            raise OSError(errno.EIO, "private")
        real_close(value)
    monkeypatch.setattr(os, "close", close)
    owner.budget.close()
    owner.finish()
    assert attempts == [fd, fd] and not owner.owned


def test_scoped_probe_all_paths_avoid_unsafe_native_fallbacks(enrolled, monkeypatch):
    from blueprint_pipeline.control_plane_lane_owner_target_io import _safe_probe_type
    path, expected = enrolled
    owner = files(expected)
    for method in ("_open", "_take", "_visible", "_lease", "_close_one"):
        monkeypatch.setattr(lifetime.LeasedScratchUse, method, lambda *a, **kw: pytest.fail("unsafe native fallback"))
    monkeypatch.setattr(lifetime, "_read_lease", lambda *a, **kw: pytest.fail("unsafe native lease reader"))
    probe = _safe_probe_type(owner).probe(path, now=lambda: 1001, _checkpoint=owner.budget.tick)
    probe.check()
    owner.final_lease_check()
    owner.finish_target()
    probe.close()
    owner.finish()
    assert owner.budget.counts["roots"] == 3
    assert not owner.probe_owned


def test_native_first_root_reuse_preserves_foreign_regular(enrolled, tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_lane_owner_target_io import _safe_probe_type
    path, expected = enrolled
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"untouched")
    fd = os.open(foreign, os.O_RDONLY)
    owner = files(expected)
    real_open = os.open
    monkeypatch.setattr(os, "open", lambda name, *a, **kw: fd if name == "/" else real_open(name, *a, **kw))
    try:
        with pytest.raises(ValueError):
            _safe_probe_type(owner).probe(path, now=lambda: 1001, _checkpoint=owner.budget.tick)
        assert os.fstat(fd).st_ino == foreign.stat().st_ino
        assert not owner.probe_owned
    finally:
        os.close(fd)


def test_changed_target_and_lease_versions_refuse(enrolled):
    from blueprint_pipeline.control_plane_lane_owner_target_io import _safe_probe_type
    path, expected = enrolled
    expected["folder_identity"]["ino"] += 1
    owner = files(expected)
    try:
        with pytest.raises(ValueError, match="owner_target_version_changed"):
            _safe_probe_type(owner).probe(path, now=lambda: 1001, _checkpoint=owner.budget.tick)
    finally:
        owner.finish()


def test_safe_probe_does_not_create_missing_coordination_lock(enrolled):
    from blueprint_pipeline.control_plane_lane_owner_target_io import _safe_probe_type
    path, expected = enrolled
    lock = path.parent.parent / ".lane-scratch.lock"
    lock.unlink()
    owner = files(expected)
    try:
        with pytest.raises(Exception):
            _safe_probe_type(owner).probe(path, now=lambda: 1001, _checkpoint=owner.budget.tick)
        assert not lock.exists()
    finally:
        owner.finish()
