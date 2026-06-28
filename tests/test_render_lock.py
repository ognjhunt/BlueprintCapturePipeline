"""Hermetic tests for the file-based render single-flight lock.

No GPU, no subprocesses for the lock itself, no network. Every test uses an
isolated ``tmp_path`` state directory so concurrent test runs never collide.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from blueprint_pipeline import render_lock as render_lock_module
from blueprint_pipeline.render_lock import (
    LockInfo,
    RenderLock,
    RenderLockError,
    RenderLockTimeout,
    render_lock,
)


def _locks_dir(tmp_path: Path) -> Path:
    return tmp_path / "state" / "render-locks"


def _reaped_dead_pid() -> int:
    """Spawn a trivial process, reap it, and return its now-dead PID."""

    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


def _plant_lockfile(path: Path, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields), encoding="utf-8")


# --------------------------------------------------------------------------
# acquire / release
# --------------------------------------------------------------------------


def test_acquire_writes_pid_label_and_start_time(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    lock = render_lock("isaac-render", state_dir=sd)
    lock.acquire()
    try:
        assert lock.path.exists()
        assert lock.path.parent == sd
        assert lock.path.name.endswith(".lock")
        payload = json.loads(lock.path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["label"] == "isaac-render"
        assert isinstance(payload["start_time"], (int, float))
        assert payload["start_time_iso"]
    finally:
        lock.release()
    assert not lock.path.exists()


def test_acquire_release_round_trip_allows_reacquire(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    first = render_lock("isaac-render", state_dir=sd)
    first.acquire()
    first.release()
    assert not first.path.exists()

    second = render_lock("isaac-render", state_dir=sd)
    second.acquire()  # must not raise: the lock is free again
    try:
        assert second.path.exists()
    finally:
        second.release()


def test_release_without_acquire_is_a_noop(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    lock = render_lock("isaac-render", state_dir=sd)
    lock.release()  # no exception, nothing to remove
    assert not lock.path.exists()


def test_read_holder_none_when_unlocked(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    lock = render_lock("isaac-render", state_dir=sd)
    assert lock.read_holder() is None


# --------------------------------------------------------------------------
# context manager
# --------------------------------------------------------------------------


def test_context_manager_acquires_and_releases(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    with render_lock("isaac-render", state_dir=sd) as lock:
        assert lock.path.exists()
        holder = lock.read_holder()
        assert isinstance(holder, LockInfo)
        assert holder.pid == os.getpid()
        assert holder.label == "isaac-render"
    assert not lock.path.exists()


def test_context_manager_releases_on_exception(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    lock = render_lock("isaac-render", state_dir=sd)
    with pytest.raises(ValueError):
        with lock:
            assert lock.path.exists()
            raise ValueError("boom")
    assert not lock.path.exists()


def test_distinct_labels_do_not_contend(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    with render_lock("isaac-render", state_dir=sd) as a:
        with render_lock("oscar-launch", state_dir=sd) as b:
            assert a.path != b.path
            assert a.path.exists()
            assert b.path.exists()


# --------------------------------------------------------------------------
# contention
# --------------------------------------------------------------------------


def test_timeout_is_a_render_lock_error() -> None:
    assert issubclass(RenderLockTimeout, RenderLockError)


def test_second_acquire_fails_fast_by_default(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    with render_lock("isaac-render", state_dir=sd):
        contender = render_lock("isaac-render", state_dir=sd)
        start = time.monotonic()
        with pytest.raises(RenderLockTimeout):
            contender.acquire()
        # "fail fast" => it must not have polled/slept for any meaningful time.
        assert time.monotonic() - start < 0.5


def test_second_acquire_blocks_until_timeout(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    with render_lock("isaac-render", state_dir=sd):
        contender = render_lock(
            "isaac-render", state_dir=sd, timeout=0.2, poll_interval=0.02
        )
        start = time.monotonic()
        with pytest.raises(RenderLockTimeout):
            contender.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15  # it actually waited for the timeout window


def test_held_lock_is_not_reclaimed_by_live_holder(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    with render_lock("isaac-render", state_dir=sd) as holder:
        token = holder.read_holder()
        contender = render_lock("isaac-render", state_dir=sd)
        with pytest.raises(RenderLockTimeout):
            contender.acquire()
        # The original holder's lockfile is untouched.
        still = holder.read_holder()
        assert still is not None
        assert still.start_time == token.start_time


# --------------------------------------------------------------------------
# stale reclaim
# --------------------------------------------------------------------------


def test_pid_is_alive_primitive(tmp_path: Path) -> None:
    assert render_lock_module._pid_is_alive(os.getpid()) is True
    assert render_lock_module._pid_is_alive(_reaped_dead_pid()) is False


def test_stale_lock_from_dead_pid_is_reclaimed(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    dead = _reaped_dead_pid()
    # Guard against the (astronomically unlikely) PID-recycle race.
    assert render_lock_module._pid_is_alive(dead) is False

    path = render_lock("isaac-render", state_dir=sd).path
    _plant_lockfile(
        path,
        pid=dead,
        label="isaac-render",
        start_time=time.time(),
        start_time_iso="planted",
    )

    lock = render_lock("isaac-render", state_dir=sd)
    lock.acquire()  # the dead holder must be reclaimed
    try:
        holder = lock.read_holder()
        assert holder is not None
        assert holder.pid == os.getpid()
    finally:
        lock.release()


def test_stale_lock_reclaimed_when_holder_not_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deterministic variant: stub liveness so there is zero PID-recycle risk.
    sd = _locks_dir(tmp_path)
    path = render_lock("isaac-render", state_dir=sd).path
    _plant_lockfile(
        path,
        pid=999_999,
        label="isaac-render",
        start_time=time.time(),
        start_time_iso="planted",
    )
    monkeypatch.setattr(render_lock_module, "_pid_is_alive", lambda pid: False)

    lock = render_lock("isaac-render", state_dir=sd)
    lock.acquire()
    try:
        assert lock.read_holder().pid == os.getpid()
    finally:
        lock.release()


def test_malformed_lockfile_is_reclaimed(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    path = render_lock("isaac-render", state_dir=sd).path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-valid-json{{{", encoding="utf-8")

    lock = render_lock("isaac-render", state_dir=sd, poll_interval=0.01)
    lock.acquire()
    try:
        assert lock.read_holder().pid == os.getpid()
    finally:
        lock.release()


def test_expired_lock_reclaimed_via_max_age(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    path = render_lock("isaac-render", state_dir=sd).path
    # Holder PID is alive (this process) but the lock is far too old.
    _plant_lockfile(
        path,
        pid=os.getpid(),
        label="isaac-render",
        start_time=time.time() - 1_000,
        start_time_iso="old",
    )
    lock = render_lock("isaac-render", state_dir=sd, max_age=1.0)
    lock.acquire()
    try:
        holder = lock.read_holder()
        assert holder is not None
        assert holder.start_time > time.time() - 5
    finally:
        lock.release()


def test_release_does_not_delete_a_replacement_lock(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    lock = render_lock("isaac-render", state_dir=sd)
    lock.acquire()
    # Simulate our lock having been reclaimed and replaced by another holder
    # (e.g. we were wrongly judged stale). Releasing must not delete theirs.
    _plant_lockfile(
        lock.path,
        pid=4242,
        label="isaac-render",
        start_time=time.time() + 5,
        start_time_iso="replacement",
    )
    lock.release()
    assert lock.path.exists()
    holder = json.loads(lock.path.read_text(encoding="utf-8"))
    assert holder["pid"] == 4242


# --------------------------------------------------------------------------
# state-dir resolution
# --------------------------------------------------------------------------


def test_env_var_sets_default_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "env-locks"
    monkeypatch.setenv("BLUEPRINT_RENDER_LOCK_DIR", str(target))
    lock = render_lock("isaac-render")
    assert lock.path.parent == target
    lock.acquire()
    try:
        assert lock.path.exists()
    finally:
        lock.release()


def test_label_is_slugified_into_a_safe_filename(tmp_path: Path) -> None:
    sd = _locks_dir(tmp_path)
    lock = render_lock("isaac render/../weird:name", state_dir=sd)
    assert lock.path.parent == sd
    assert "/" not in lock.path.name
    assert ":" not in lock.path.name
    assert lock.path.name.endswith(".lock")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_main_status_reports_holder(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    sd = _locks_dir(tmp_path)
    with render_lock("isaac-render", state_dir=sd):
        rc = render_lock_module.main(
            ["status", "isaac-render", "--state-dir", str(sd)]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["held"] is True
        assert payload["holder"]["pid"] == os.getpid()


def test_main_status_unlocked(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    sd = _locks_dir(tmp_path)
    rc = render_lock_module.main(["status", "isaac-render", "--state-dir", str(sd)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["held"] is False
    assert payload["holder"] is None


def test_main_break_force_removes_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    sd = _locks_dir(tmp_path)
    lock = render_lock("isaac-render", state_dir=sd)
    lock.acquire()
    try:
        rc = render_lock_module.main(
            ["break", "isaac-render", "--state-dir", str(sd), "--force"]
        )
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["broken"] is True
        assert not lock.path.exists()
    finally:
        lock.release()


def test_main_break_refuses_live_holder_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    sd = _locks_dir(tmp_path)
    lock = render_lock("isaac-render", state_dir=sd)
    lock.acquire()
    try:
        rc = render_lock_module.main(["break", "isaac-render", "--state-dir", str(sd)])
        assert rc != 0
        assert json.loads(capsys.readouterr().out)["broken"] is False
        assert lock.path.exists()  # the live holder is preserved
    finally:
        lock.release()
