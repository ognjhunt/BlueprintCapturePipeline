"""Focused resource ownership corrections in the existing read-only lane report."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_scratch_retention.py
import errno
import os

import pytest

from blueprint_pipeline import control_plane_lane_scratch_retention as retention


def scanner():
    return retention._Observation((), "/", 0, False, lambda: 0, 10)


def test_initial_identity_failure_preserves_reused_foreign_token(tmp_path, monkeypatch):
    original = tmp_path / "original"
    original.write_bytes(b"o")
    replacement = tmp_path / "foreign"
    replacement.write_bytes(b"f")
    foreign = os.open(replacement, os.O_RDONLY)
    scan = scanner()
    original_stat = os.fstat
    reused = []

    def failing_initial(fd):
        if not reused and fd != foreign:
            os.close(fd)
            os.dup2(foreign, fd)
            reused.append(fd)
            raise OSError(errno.EIO, "injected initial identity fault")
        return original_stat(fd)

    monkeypatch.setattr(os, "fstat", failing_initial)
    try:
        with pytest.raises(OSError):
            scan.open(str(original), os.O_RDONLY)
        scan.close(reused[0])
        assert original_stat(reused[0]).st_ino == original_stat(foreign).st_ino
        assert "lane_descriptor_ownership_unproven" in scan.blockers
        assert not scan.fds
    finally:
        for fd in (*reused, foreign):
            try:
                os.close(fd)
            except OSError:
                pass


def test_known_descriptor_reuse_is_checked_before_first_close(tmp_path):
    original = tmp_path / "original"
    original.write_bytes(b"o")
    replacement = tmp_path / "foreign"
    replacement.write_bytes(b"f")
    scan = scanner()
    fd = scan.open(str(original), os.O_RDONLY)
    foreign = os.open(replacement, os.O_RDONLY)
    os.close(fd)
    os.dup2(foreign, fd)
    try:
        scan.close(fd)
        assert os.fstat(fd).st_ino == os.fstat(foreign).st_ino
        assert "lane_descriptor_changed" in scan.blockers
        assert not scan.fds
    finally:
        for value in (fd, foreign):
            try:
                os.close(value)
            except OSError:
                pass


def test_identity_fault_keeps_known_token_until_retry_and_finalizes_others(tmp_path, monkeypatch):
    target = tmp_path / "tiny"
    target.write_bytes(b"t")
    scan = scanner()
    first = scan.open(str(target), os.O_RDONLY)
    second = scan.open(str(target), os.O_RDONLY)
    original = os.fstat
    once = []

    def fail_once(fd):
        if fd == first and not once:
            once.append(fd)
            raise OSError(errno.EIO, "fixed injected current identity fault")
        return original(fd)

    monkeypatch.setattr(os, "fstat", fail_once)
    try:
        scan.close(first)
        assert first in scan.fds
        scan.close(second)
        assert second not in scan.fds
        scan.close(first)
        assert not scan.fds
        assert "lane_descriptor_identity_unavailable" in scan.blockers
        for fd in (first, second):
            with pytest.raises(OSError):
                original(fd)
    finally:
        for fd in (first, second):
            try:
                os.close(fd)
            except OSError:
                pass


def test_finalization_ignores_expired_clock_and_preserves_foreign_token(tmp_path):
    original = tmp_path / "original"
    original.write_bytes(b"o")
    replacement = tmp_path / "foreign"
    replacement.write_bytes(b"f")
    scan = scanner()
    owned = scan.open(str(original), os.O_RDONLY)
    changed = scan.open(str(original), os.O_RDONLY)
    foreign = os.open(replacement, os.O_RDONLY)
    os.close(changed)
    os.dup2(foreign, changed)
    scan.clock = lambda: pytest.fail("cleanup consulted expired clock")
    try:
        for fd in reversed(tuple(scan.fds)):
            scan.close(fd)
        assert not scan.fds
        assert os.fstat(changed).st_ino == os.fstat(foreign).st_ino
        with pytest.raises(OSError):
            os.fstat(owned)
    finally:
        for fd in (owned, changed, foreign):
            try:
                os.close(fd)
            except OSError:
                pass


def test_known_close_fault_is_retained_and_all_other_handles_are_closed(tmp_path, monkeypatch):
    target = tmp_path / "tiny"
    target.write_bytes(b"t")
    scan = scanner()
    failing = scan.open(str(target), os.O_RDONLY)
    other = scan.open(str(target), os.O_RDONLY)
    original = os.close
    calls = []

    def fail_once(fd):
        if fd == failing and not calls:
            calls.append(fd)
            raise OSError(errno.EIO, "fixed injected close fault")
        original(fd)

    monkeypatch.setattr(os, "close", fail_once)
    try:
        scan.close(failing)
        scan.close(other)
        assert failing in scan.fds and other not in scan.fds
        scan.close(failing)
        assert not scan.fds and "lane_descriptor_close_failed" in scan.blockers
        for fd in (failing, other):
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        for fd in (failing, other):
            try:
                original(fd)
            except OSError:
                pass


def test_public_report_initial_identity_fault_is_fixed_incomplete(tmp_path, monkeypatch):
    root = tmp_path / "lanes"
    root.mkdir()
    pins = tmp_path / "pins"
    pins.mkdir()
    foreign_path = tmp_path / "foreign"
    foreign_path.write_bytes(b"f")
    foreign = os.open(foreign_path, os.O_RDONLY)
    real_stat = os.fstat
    reused = []
    monkeypatch.setattr(retention, "_ALLOWED_ROOTS", frozenset({str(root)}))

    def fail_initial(fd):
        if not reused and fd != foreign:
            os.close(fd)
            os.dup2(foreign, fd)
            reused.append(fd)
            raise OSError(errno.EIO, "injected secret text must not escape")
        return real_stat(fd)

    monkeypatch.setattr(os, "fstat", fail_initial)
    try:
        result = retention.observe_lane_scratch_retention(
            (str(root),),
            pins_root=str(pins),
            observed_at_epoch=50,
            enabled_requested=True,
            monotonic=lambda: 0,
        )
        assert result["complete"] is False
        assert "lane_descriptor_ownership_unproven" in result["blockers"]
        assert (
            result["logical_bytes"]
            is result["allocated_bytes"]
            is result["candidate_bytes"]
            is None
        )
        assert result["execution_authorized"] is result["apply_supported"] is False
        assert result["mutations"] == result["removed_bytes"] == 0
        assert "injected secret" not in repr(result)
        assert real_stat(reused[0]).st_ino == real_stat(foreign).st_ino
    finally:
        for fd in (*reused, foreign):
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.mark.parametrize(("original_kind", "foreign_kind"), [
    ("regular", "regular"), ("directory", "regular"), ("directory", "directory"),
])
def test_first_successful_proof_matches_named_original_before_adoption(tmp_path, monkeypatch, original_kind, foreign_kind):
    original = tmp_path / "original"
    replacement = tmp_path / "foreign"
    for path, kind in ((original, original_kind), (replacement, foreign_kind)):
        path.mkdir() if kind == "directory" else path.write_bytes(b"x")
    real_open, real_close, real_fstat, real_dup2 = os.open, os.close, os.fstat, os.dup2
    foreign = real_open(replacement, os.O_RDONLY)
    proof = real_fstat(foreign)
    scan, reused = scanner(), []

    def substituted(name, *args, **kwargs):
        fd = real_open(name, *args, **kwargs)
        if str(name) == str(original) and not reused:
            real_close(fd)
            real_dup2(foreign, fd)
            reused.append(fd)
        return fd

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "open", substituted)
            flags = os.O_RDONLY | (os.O_DIRECTORY if original_kind == "directory" else 0)
            with pytest.raises(retention._Blocked, match="lane_descriptor_ownership_unproven"):
                scan.open(str(original), flags)
            scan.close(reused[0])
        observed = real_fstat(reused[0])
        assert (observed.st_dev, observed.st_ino) == (proof.st_dev, proof.st_ino)
        assert "lane_descriptor_ownership_unproven" in scan.blockers
        assert not scan.fds
    finally:
        for fd in (*reused, foreign):
            try:
                real_close(fd)
            except OSError:
                pass


def test_reused_parent_is_refused_before_child_named_access(tmp_path, monkeypatch):
    original, replacement = tmp_path / "parent", tmp_path / "foreign"
    original.mkdir()
    replacement.mkdir()
    (original / "tiny").write_bytes(b"x")
    (replacement / "tiny").write_bytes(b"f")
    scan = scanner()
    parent = scan.open(str(original), os.O_RDONLY | os.O_DIRECTORY)
    foreign = os.open(replacement, os.O_RDONLY | os.O_DIRECTORY)
    real_close, real_fstat = os.close, os.fstat
    proof = real_fstat(foreign)
    real_close(parent)
    os.dup2(foreign, parent)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "stat", lambda *a, **k: pytest.fail("child stat used a substituted parent"))
            with pytest.raises(retention._Blocked, match="lane_descriptor_changed"):
                scan.open("tiny", os.O_RDONLY, parent)
        scan.close(parent)
        assert (real_fstat(parent).st_dev, real_fstat(parent).st_ino) == (proof.st_dev, proof.st_ino)
        assert not scan.fds
    finally:
        for fd in (parent, foreign):
            try:
                real_close(fd)
            except OSError:
                pass
