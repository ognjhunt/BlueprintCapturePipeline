"""AUT-05: expiry never weakens atomic, scoped, restart-safe replay protection."""
from __future__ import annotations

import errno
import hashlib
import json
import multiprocessing
import os

import pytest

from blueprint_pipeline import intake_nonce_retention as store
from blueprint_pipeline import live_pipeline_intake_service as service


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
    assert report["observation_scope"] == "bounded_record_window"


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
        if candidate == path:
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
