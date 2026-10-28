"""AUT-05: expiry never weakens atomic, scoped, restart-safe replay protection."""
from __future__ import annotations

import errno
import hashlib
import json
import multiprocessing
import os
import time

import pytest

from blueprint_pipeline import intake_nonce_retention as store
from blueprint_pipeline import live_pipeline_intake_service as service


@pytest.fixture(autouse=True)
def _close_discovery_cursors():
    store._reset_discovery_cursors()
    yield
    store._reset_discovery_cursors()


def _root(tmp_path):
    root = tmp_path / "nonces"
    root.mkdir(mode=0o700)
    return root


def _path(root, nonce, client="client"):
    return root / (hashlib.sha256(f"{client}\0{nonce}".encode()).hexdigest() + ".json")


def test_report_only_then_incremental_expiry_preserves_fresh_claims(tmp_path):
    root = _root(tmp_path)
    for number in range(120):
        assert store.claim(root, client_id="client", nonce=f"expired-{number}", now=1, max_age_seconds=1)
    for number in range(12):
        assert store.claim(root, client_id="client", nonce=f"fresh-{number}", now=1000, max_age_seconds=300)
    report = store.sweep(root, now=1000, limit=16)
    assert report["mode"] == "report_only" and report["reclaimed"] == 0
    assert report["inspected"] == 16 and report["record_inspection_truncated"] is True
    assert len(list(root.glob("*.json"))) == 132
    for _ in range(20):
        store.sweep(root, now=1000, limit=16, apply=True)
    assert len(list(root.glob("*.json"))) == 12
    assert not store.claim(root, client_id="client", nonce="fresh-0", now=1000, max_age_seconds=300)
    assert store.claim(root, client_id="other-client", nonce="fresh-0", now=1000, max_age_seconds=300)
    assert report["free_bytes"] >= 0 and report["free_inodes"] >= 0
    assert report["observation_scope"] == "bounded_discovery_and_record_window"
    assert report["inventory_complete"] is False


def _racing_operation(root, barrier, results, sweep):
    barrier.wait()
    if sweep:
        store.sweep(root, now=1000, limit=100, apply=True)
        results.put("swept")
    else:
        results.put(store.claim(root, client_id="client", nonce="same-expired-nonce", now=1000, max_age_seconds=300))


@pytest.mark.slow
def test_two_process_claimants_and_sweeper_never_accept_the_same_request_twice(tmp_path):
    root = _root(tmp_path)
    assert store.claim(root, client_id="client", nonce="same-expired-nonce", now=1, max_age_seconds=1)
    store.sweep(root, now=1, limit=1)
    parent_cursor = store._DISCOVERY_CURSORS[os.path.abspath(root)]
    processes = multiprocessing.get_context("fork")
    barrier, results = processes.Barrier(3), processes.Queue()
    children = [processes.Process(target=_racing_operation, args=(root, barrier, results, sweep))
                for sweep in (False, False, True)]
    for child in children:
        child.start()
    for child in children:
        child.join(timeout=10)
        assert child.exitcode == 0
    outcomes = [results.get(timeout=1) for _ in children]
    assert outcomes.count(True) == outcomes.count(False) == outcomes.count("swept") == 1
    assert parent_cursor.descriptor >= 0, "forked children must close only their inherited cursor copy"
    assert not store.claim(root, client_id="client", nonce="same-expired-nonce", now=1000, max_age_seconds=300)


@pytest.mark.parametrize("damage", ["malformed", "invalid_expiry", "symlink", "unsafe_mode"])
def test_unknown_records_are_retained_and_fail_closed(tmp_path, damage):
    root = _root(tmp_path)
    store.claim(root, client_id="client", nonce="protected", now=1, max_age_seconds=1)
    path = _path(root, "protected")
    if damage == "malformed":
        path.write_text("{broken")
    elif damage == "invalid_expiry":
        record = json.loads(path.read_text())
        record["expires_at_epoch"] = float("nan")
        path.write_text(json.dumps(record))
    elif damage == "unsafe_mode":
        path.chmod(0o644)
    else:
        target = tmp_path / "retained-target"
        path.rename(target)
        path.symlink_to(target)
    before = path.read_bytes()
    report = store.sweep(root, now=1000, apply=True)
    assert report["unproven"] == 1 and report["reclaimed"] == 0
    with pytest.raises(store.NonceStoreError, match="intake_nonce_record_"):
        store.claim(root, client_id="client", nonce="protected", now=1000, max_age_seconds=300)
    assert path.read_bytes() == before


def test_expiry_grace_and_restart_clock_rollback_cannot_reopen_nonce(tmp_path):
    root = _root(tmp_path)
    assert store.claim(root, client_id="client", nonce="clock", now=1000, max_age_seconds=10)
    expires = json.loads(_path(root, "clock").read_text())["expires_at_epoch"]
    assert not store.claim(root, client_id="client", nonce="clock", now=expires + 1, max_age_seconds=10)
    assert store.sweep(root, now=expires + store.GRACE_SECONDS, apply=True)["reclaimed"] == 0
    # A clock jump only reclaims after full expiry+grace. Returning to the prior
    # clock after restart fails closed against the durable high-water record.
    assert store.sweep(root, now=100000, apply=True)["reclaimed"] == 1
    with pytest.raises(store.NonceStoreError, match="intake_nonce_clock_rollback"):
        store.claim(root, client_id="client", nonce="clock", now=1000, max_age_seconds=10)


def test_failed_nonce_fsync_is_unavailable_and_keeps_replay_protection(tmp_path, monkeypatch):
    root = _root(tmp_path)
    target = _path(root, "full-disk")
    fsync = os.fsync
    def full(descriptor):
        if target.exists() and os.fstat(descriptor).st_ino == target.stat().st_ino:
            raise OSError(errno.ENOSPC, "private filesystem diagnostic")
        return fsync(descriptor)
    monkeypatch.setattr(store.os, "fsync", full)
    with pytest.raises(store.NonceStoreError, match="^intake_nonce_store_unavailable$"):
        store.claim(root, client_id="client", nonce="full-disk", now=1000, max_age_seconds=300)
    monkeypatch.setattr(store.os, "fsync", fsync)
    assert not store.claim(root, client_id="client", nonce="full-disk", now=1000, max_age_seconds=300)


def test_read_permission_failure_is_typed_and_preserves_record(tmp_path, monkeypatch):
    root = _root(tmp_path)
    store.claim(root, client_id="client", nonce="permission", now=1, max_age_seconds=1)
    path = _path(root, "permission")
    original = store.os.open
    def denied(candidate, *args, **kwargs):
        if candidate == path or (str(candidate) == path.name and kwargs.get("dir_fd") is not None):
            raise PermissionError("private diagnostic")
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(store.os, "open", denied)
    with pytest.raises(store.NonceStoreError, match="intake_nonce_record_unavailable"):
        store.claim(root, client_id="client", nonce="permission", now=1000, max_age_seconds=300)
    assert path.exists()


def test_bounded_volume_soak_stabilizes_at_ttl_plus_grace_window(tmp_path):
    root = _root(tmp_path)
    sizes = []
    for tick in range(30):
        now = 1000 + tick * 30
        for request in range(6):
            assert store.claim(root, client_id="client", nonce=f"tick-{tick}-{request}", now=now, max_age_seconds=1)
        store.sweep(root, now=now, limit=32, apply=True)
        sizes.append(len(list(root.glob("*.json"))))
    # Six requests/30s; 61s retention + 60s grace spans five batches,
    # plus at most one additional batch of bounded inspection lag.
    assert max(sizes[10:]) <= 36
    assert all(json.loads(path.read_text())["claimed_at_epoch"] >= now - 180
               for path in root.glob("*.json"))


def test_http_claim_defaults_to_report_only_and_exposes_bounded_health(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(root))
    monkeypatch.delenv("BLUEPRINT_INTAKE_NONCE_CLEANUP_ENABLED", raising=False)
    store.claim(root, client_id="client", nonce="retained-old", now=1, max_age_seconds=1)
    assert service._claim_intake_nonce(client_id="client", nonce="new", now=1000, max_age_seconds=300)
    assert _path(root, "retained-old").is_file()
    health = service._nonce_retention_health()
    assert health["mode"] == "report_only" and health["expired"] == 1
    assert str(root) not in json.dumps(health) and "retained-old" not in json.dumps(health)


def _claim_with_clock_failure(root, results):
    try:
        store.claim(root, client_id="client", nonce="clock-state-test", now=1000, max_age_seconds=300)
    except store.NonceStoreError as exc:
        results.put(str(exc))
    else:
        results.put("unexpected_acceptance")


@pytest.mark.slow
def test_fifo_clock_state_refuses_promptly_and_releases_global_nonce_lock(tmp_path):
    root = _root(tmp_path)
    clock = root / store.STATE_FILE
    os.mkfifo(clock, mode=0o600)
    processes = multiprocessing.get_context("fork")
    results = processes.Queue()
    child = processes.Process(target=_claim_with_clock_failure, args=(root, results))
    child.start()
    child.join(timeout=2)
    blocked = child.is_alive()
    if blocked:
        child.terminate()
        child.join(timeout=2)
    assert not blocked, "FIFO clock state blocked the global nonce lock"
    assert child.exitcode == 0
    assert results.get(timeout=1) == "intake_nonce_clock_state_unsafe"
    assert not list(root.glob("*.json"))
    # Owned fixture repair demonstrates that the failed read released the lock.
    clock.unlink()
    assert store.claim(root, client_id="client", nonce="healthy-after-clock-refusal", now=1000, max_age_seconds=300)


@pytest.mark.parametrize("damage", ["oversized", "unsafe_mode", "symlink", "directory", "non_utf8"])
def test_clock_state_is_private_regular_size_bounded_json(tmp_path, damage):
    root = _root(tmp_path)
    clock = root / store.STATE_FILE
    if damage == "directory":
        clock.mkdir(mode=0o700)
    elif damage == "symlink":
        target = tmp_path / "private-clock-target"
        target.write_text('{"schema_version":"blueprint_intake_nonce_retention_state.v1","high_water_epoch":1}')
        target.chmod(0o600)
        clock.symlink_to(target)
    else:
        clock.write_bytes(b"x" * 4097 if damage == "oversized" else b"\xff" if damage == "non_utf8"
                          else b'{"schema_version":"blueprint_intake_nonce_retention_state.v1","high_water_epoch":1}')
        clock.chmod(0o644 if damage == "unsafe_mode" else 0o600)
    with pytest.raises(store.NonceStoreError, match="^intake_nonce_clock_state_(unsafe|invalid)$"):
        store.claim(root, client_id="client", nonce="clock-state-test", now=1000, max_age_seconds=300)
    assert not list(root.glob("*.json"))


def _legacy_record(root, number):
    nonce = f"legacy-{number}"
    path = _path(root, nonce)
    path.write_text(json.dumps({
        "schema_version": store.SCHEMA_VERSION,
        "client_id_sha256": hashlib.sha256(b"client").hexdigest(),
        "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(),
        "claimed_at_epoch": 1, "expires_at_epoch": 62,
    }))
    path.chmod(0o600)
    return path.name


def _track_scandir(monkeypatch):
    original = store.os.scandir
    streams = []

    class Stream:
        def __init__(self, descriptor):
            assert isinstance(descriptor, int), "discovery must use the anchored directory FD"
            self.iterator = original(descriptor)
            self.names = []
            self.closed = False

        def __next__(self):
            entry = next(self.iterator)
            self.names.append(entry.name)
            return entry

        def close(self):
            self.closed = True
            self.iterator.close()

    def open_stream(descriptor):
        stream = Stream(descriptor)
        streams.append(stream)
        return stream

    monkeypatch.setattr(store.os, "scandir", open_stream)
    return streams


def test_large_legacy_inventory_and_new_request_never_enumerate_beyond_window(tmp_path, monkeypatch):
    root = _root(tmp_path)
    for number in range(5000):
        _legacy_record(root, number)
        (root / f"unknown-{number}").touch()
    streams = _track_scandir(monkeypatch)
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(root))
    monkeypatch.delenv("BLUEPRINT_INTAKE_NONCE_CLEANUP_ENABLED", raising=False)
    seen = []
    original_read = store._read_record

    def read(path, *, directory_fd=None):
        if directory_fd is not None:
            seen.append(path.name)
        return original_read(path, directory_fd=directory_fd)

    monkeypatch.setattr(store, "_read_record", read)
    started = time.monotonic()
    now = time.time()
    assert service._claim_intake_nonce(client_id="client", nonce="fresh-request", now=now,
                                       max_age_seconds=300)
    report = json.loads((root / store.REPORT_FILE).read_text())
    assert time.monotonic() - started < 2
    assert report["inspected"] <= 64
    assert report["discovery_entries_examined"] <= report["discovery_entry_limit"] == 136
    assert len(streams) == 1 and len(streams[0].names) <= 136
    assert report["mode"] == "report_only" and report["reclaimed"] == 0
    assert report["inventory_complete"] is False and report["record_inspection_truncated"] is True
    assert _path(root, "legacy-4999").exists()
    previous = len(streams[0].names)
    for tick in range(20):
        assert store.claim(root, client_id="client", nonce=f"new-{tick}", now=now,
                           max_age_seconds=300)
        report = store.sweep(root, now=now, limit=4)
        assert report["inspected"] <= 4
        assert report["discovery_entries_examined"] <= report["discovery_entry_limit"] == 16
        assert len(streams) == 1, "new claims must not reopen discovery at the directory start"
        assert len(streams[0].names) - previous <= 16
        previous = len(streams[0].names)
    assert len(set(seen)) > 20, "healthy ticks must continue past the first discovery window"
    assert not store.claim(root, client_id="client", nonce="fresh-request", now=now,
                           max_age_seconds=300)


def test_discovery_progresses_fairly_through_unknown_entries_despite_new_claims(tmp_path, monkeypatch):
    root = _root(tmp_path)
    original_names = {_legacy_record(root, number) for number in range(128)}
    for number in range(128):
        (root / f"unknown-{number}").touch()
    seen = set()
    original_read = store._read_record

    def read(path, *, directory_fd=None):
        if directory_fd is not None:
            seen.add(path.name)
        return original_read(path, directory_fd=directory_fd)

    monkeypatch.setattr(store, "_read_record", read)
    for tick in range(100):
        assert store.claim(root, client_id="client", nonce=f"arriving-{tick}", now=1000,
                           max_age_seconds=300)
        report = store.sweep(root, now=1000, limit=8)
        assert report["inspected"] <= 8
        assert report["discovery_entries_examined"] <= 24
        assert report["inventory_complete"] is False
        if original_names <= seen:
            break
    assert original_names <= seen
    assert report["reclaimed"] == 0


def test_unknown_names_consume_discovery_deadline_and_resume_without_full_scan(tmp_path, monkeypatch):
    root = _root(tmp_path)
    for number in range(2000):
        (root / f"unknown-{number}").touch()
    streams = _track_scandir(monkeypatch)
    clock = iter(number / 100 for number in range(1000))
    monkeypatch.setattr(store.time, "monotonic", lambda: next(clock))
    first = store.sweep(root, now=1000, limit=1000)
    names_first = tuple(streams[0].names)
    second = store.sweep(root, now=1000, limit=1000)
    assert first["discovery_time_budget_reached"] is True
    assert second["discovery_time_budget_reached"] is True
    assert 0 < first["discovery_entries_examined"] <= 5
    assert 0 < second["discovery_entries_examined"] <= 5
    assert first["inspected"] == second["inspected"] == 0
    assert len(streams) == 1 and tuple(streams[0].names[:len(names_first)]) == names_first
    assert len(set(streams[0].names)) == len(streams[0].names)
    assert first["inventory_complete"] is second["inventory_complete"] is False


def test_discovery_completion_eviction_and_error_close_scanners_and_directory_fds(tmp_path, monkeypatch):
    roots = []
    cursors = []
    streams = _track_scandir(monkeypatch)
    for number in range(store.DISCOVERY_CACHE_LIMIT + 2):
        root = tmp_path / f"store-{number}"
        root.mkdir(mode=0o700)
        roots.append(root)
        for nonce in range(3):
            _legacy_record(root, nonce)
        store.sweep(root, now=1000, limit=1)
        cursors.append(store._DISCOVERY_CURSORS[os.path.abspath(root)])
        assert len(store._DISCOVERY_CURSORS) <= store.DISCOVERY_CACHE_LIMIT
    assert all(cursor.descriptor == -1 and cursor.iterator is None for cursor in cursors[:2])
    assert all(stream.closed for stream in streams[:2])
    report = store.sweep(roots[-1], now=1000, limit=1000)
    assert report["directory_pass_completed"] is True and report["inventory_complete"] is False
    assert cursors[-1].descriptor == -1 and streams[-1].closed
    root = roots[-2]
    broken = cursors[-2]
    monkeypatch.setattr(type(broken.iterator), "__next__",
                        lambda self: (_ for _ in ()).throw(OSError("private directory error")))
    with pytest.raises(store.NonceStoreError, match="^intake_nonce_store_unavailable$"):
        store.sweep(root, now=1000, limit=1)
    assert broken.descriptor == -1 and broken.iterator is None
    assert os.path.abspath(root) not in store._DISCOVERY_CURSORS


def test_replaced_root_refuses_old_cursor_then_restarts_without_touching_fresh_claim(tmp_path):
    root = _root(tmp_path)
    for number in range(4):
        _legacy_record(root, number)
    store.sweep(root, now=1000, limit=1)
    old_cursor = store._DISCOVERY_CURSORS[os.path.abspath(root)]
    old = tmp_path / "old-nonces"
    root.rename(old)
    root.mkdir(mode=0o700)
    assert store.claim(root, client_id="client", nonce="fresh-replacement", now=1000,
                       max_age_seconds=300)
    before = _path(root, "fresh-replacement").read_bytes()
    with pytest.raises(store.NonceStoreError, match="^intake_nonce_discovery_root_changed$"):
        store.sweep(root, now=1000, apply=True)
    assert old_cursor.descriptor == -1 and old_cursor.iterator is None
    assert _path(root, "fresh-replacement").read_bytes() == before
    assert len(list(old.glob("*.json"))) == 4
    report = store.sweep(root, now=1000, apply=True)
    assert report["fresh"] == 1 and report["reclaimed"] == 0 and report["inventory_complete"] is False


def test_root_replacement_during_record_read_refuses_before_expiry_unlink(tmp_path, monkeypatch):
    root = _root(tmp_path)
    for number in range(4):
        _legacy_record(root, number)
    original_read = store._read_record
    old = tmp_path / "retained-old-store"
    replaced = False

    def replace_after_read(path, *, directory_fd=None):
        nonlocal replaced
        result = original_read(path, directory_fd=directory_fd)
        if directory_fd is not None and not replaced:
            replaced = True
            root.rename(old)
            root.mkdir(mode=0o700)
            assert store.claim(root, client_id="client", nonce="fresh-replacement", now=1000,
                               max_age_seconds=300)
        return result

    monkeypatch.setattr(store, "_read_record", replace_after_read)
    with pytest.raises(store.NonceStoreError, match="^intake_nonce_discovery_root_changed$"):
        store.sweep(root, now=1000, apply=True)
    assert replaced and len(list(old.glob("*.json"))) == 4
    assert not store._DISCOVERY_CURSORS
    assert not store.claim(root, client_id="client", nonce="fresh-replacement", now=1000,
                           max_age_seconds=300)


@pytest.mark.parametrize("damage", ["clock", "root_symlink"])
def test_refusal_before_discovery_also_closes_cached_scanner(tmp_path, damage):
    root = _root(tmp_path)
    for number in range(4):
        _legacy_record(root, number)
    store.sweep(root, now=1000, limit=1)
    cursor = store._DISCOVERY_CURSORS[os.path.abspath(root)]
    if damage == "clock":
        (root / store.STATE_FILE).write_text("invalid")
    else:
        retained = tmp_path / "retained-store"
        root.rename(retained)
        root.symlink_to(retained, target_is_directory=True)
    with pytest.raises(store.NonceStoreError):
        store.sweep(root, now=1000)
    assert cursor.descriptor == -1 and cursor.iterator is None
    assert not store._DISCOVERY_CURSORS


def test_restart_starts_bounded_discovery_without_changing_durable_replay_protection(tmp_path):
    root = _root(tmp_path)
    for number in range(20):
        _legacy_record(root, number)
    assert store.claim(root, client_id="client", nonce="restart-safe", now=1000, max_age_seconds=300)
    store.sweep(root, now=1000, limit=1)
    old_cursor = store._DISCOVERY_CURSORS[os.path.abspath(root)]
    store._reset_discovery_cursors()
    assert old_cursor.descriptor == -1
    assert not store.claim(root, client_id="client", nonce="restart-safe", now=1000,
                           max_age_seconds=300)
    report = store.sweep(root, now=1000, limit=1)
    assert report["inspected"] <= 1 and report["discovery_entries_examined"] <= 10
    assert report["discovery_continuation_scope"] == "process_lifetime"
    assert report["inventory_complete"] is False


@pytest.mark.parametrize("phase", ["final_state", "report"])
def test_final_metadata_commit_never_follows_replaced_root_or_lowers_other_clock(tmp_path, monkeypatch, phase):
    root = _root(tmp_path)
    _legacy_record(root, 1)
    outside = tmp_path / "replacement"
    outside.mkdir(mode=0o700)
    assert store.claim(outside, client_id="client", nonce="replacement-fresh", now=2000,
                       max_age_seconds=300)
    with pytest.raises(store.NonceStoreError, match="intake_nonce_clock_rollback"):
        store.claim(outside, client_id="client", nonce="older-clock-request", now=1000,
                    max_age_seconds=300)
    before = {path.name: path.read_bytes() for path in outside.iterdir()}
    original_open = store.os.open
    state_writes = 0
    replaced = False

    def open_metadata(target, flags, *args, **kwargs):
        nonlocal state_writes, replaced
        name = os.path.basename(target)
        state_temporary = name.startswith(f".{store.STATE_FILE}.") and name.endswith(".tmp")
        report_temporary = name.startswith(f".{store.REPORT_FILE}.") and name.endswith(".tmp")
        if state_temporary and flags & os.O_CREAT:
            state_writes += 1
        boundary = state_temporary and state_writes == 2 if phase == "final_state" else report_temporary
        if boundary and not replaced:
            root.rename(tmp_path / "retained-original-store")
            root.symlink_to(outside, target_is_directory=True)
            replaced = True
        return original_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(store.os, "open", open_metadata)
    with pytest.raises(store.NonceStoreError, match="intake_nonce_discovery_root_changed"):
        store.sweep(root, now=1000, limit=8)
    assert replaced
    assert before == {path.name: path.read_bytes() for path in outside.iterdir()}
    with pytest.raises(store.NonceStoreError, match="intake_nonce_clock_rollback"):
        store.claim(outside, client_id="client", nonce="older-clock-request", now=1000,
                    max_age_seconds=300)
    assert not store._DISCOVERY_CURSORS


def test_claim_creation_never_follows_root_replacement_after_clock_check(tmp_path, monkeypatch):
    root = _root(tmp_path)
    outside = tmp_path / "replacement"
    outside.mkdir(mode=0o700)
    assert store.claim(outside, client_id="client", nonce="replacement-fresh", now=2000,
                       max_age_seconds=300)
    before = {path.name: path.read_bytes() for path in outside.iterdir()}
    target = _path(root, "older-clock-request")
    original_open = store.os.open
    replaced = False

    def open_file(path, flags, *args, **kwargs):
        nonlocal replaced
        if os.fspath(path) in (str(target), target.name) and flags & os.O_CREAT and not replaced:
            root.rename(tmp_path / "retained-original-store")
            root.symlink_to(outside, target_is_directory=True)
            replaced = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(store.os, "open", open_file)
    with pytest.raises(store.NonceStoreError, match="intake_nonce_discovery_root_changed"):
        store.claim(root, client_id="client", nonce="older-clock-request", now=1000,
                    max_age_seconds=300)
    assert replaced
    assert before == {path.name: path.read_bytes() for path in outside.iterdir()}
