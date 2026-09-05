"""Registry replacement preserves authority under failure and concurrent writers."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_evaluation_intent_registry as module

A, B, C = "a" * 40, "b" * 40, "c" * 40


def _payload(commit, decision="original"):
    return json.dumps({"expected_production_commit": commit, "decision": decision}).encode()


def _install(path, commit, decision="original", service_group=None):
    module.install_release_intent(destination=path, payload=_payload(commit, decision),
        expected_commit=commit, service_group=service_group, validate=lambda value: None)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_supersession_authority", lambda commit: None)
    path = tmp_path / "owner.json"
    _install(path, A)
    return path


@pytest.mark.parametrize("decision", ["original", "changed"])
def test_retired_release_cannot_displace_current_authority(registry, decision):
    _install(registry, B)
    with pytest.raises(module.IntentRegistryError, match="retired_release_reactivation"):
        _install(registry, A, decision)
    assert registry.read_bytes() == _payload(B)
    assert registry.with_name(f"owner.superseded-{A}.json").read_bytes() == _payload(A)


def test_archive_conflict_never_overwrites_old_or_live_bytes(registry):
    archive = registry.with_name(f"owner.superseded-{A}.json")
    archive.write_bytes(b"retained-conflicting-evidence")
    with pytest.raises(module.IntentRegistryError, match="archive_conflict"):
        _install(registry, B)
    assert registry.read_bytes() == _payload(A)
    assert archive.read_bytes() == b"retained-conflicting-evidence"


@pytest.mark.parametrize("failure", ["unknown_group", "ownership", "replacement"])
def test_successor_access_or_publication_failure_keeps_live_intent(registry, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise OSError("injected-before-publication")
    if failure == "unknown_group":
        monkeypatch.setattr(module.grp, "getgrnam", fail)
    elif failure == "ownership":
        monkeypatch.setattr(module.os, "fchmod", fail)
    else:
        monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        _install(registry, B, service_group="unavailable" if failure == "unknown_group" else None)
    assert registry.read_bytes() == _payload(A)
    assert not list(registry.parent.glob(".intent-*"))


def test_two_concurrent_provisioners_keep_every_release_bytes(registry, monkeypatch):
    entered, release, started = Event(), Event(), Event()
    original = module.os.replace
    def pause_first(source, target):
        if Path(source).read_bytes() == _payload(B):
            entered.set()
            assert release.wait(5)
        original(source, target)
    monkeypatch.setattr(module.os, "replace", pause_first)
    def second():
        started.set()
        _install(registry, C)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_install, registry, B)
        assert entered.wait(5)
        other = pool.submit(second)
        assert started.wait(5)
        # The old live name remains readable even after its archive is sealed.
        assert registry.read_bytes() == _payload(A)
        assert not other.done()
        release.set()
        first.result(timeout=5)
        other.result(timeout=5)
    assert registry.read_bytes() == _payload(C)
    for commit in (A, B):
        archive = registry.with_name(f"owner.superseded-{commit}.json")
        assert archive.read_bytes() == _payload(commit)
        assert archive.stat().st_mode & 0o777 == 0o440
    assert (registry.parent / ".owner.json.registry.lock").is_file()


def test_interrupted_switch_can_resume_without_rewriting_archive(registry, monkeypatch):
    original = module.os.replace
    monkeypatch.setattr(module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        _install(registry, B)
    archive = registry.with_name(f"owner.superseded-{A}.json")
    original_inode = archive.stat().st_ino
    _install(registry, A)  # Current live bytes remain idempotent after interruption.
    monkeypatch.setattr(module.os, "replace", original)
    _install(registry, B)
    assert registry.read_bytes() == _payload(B)
    assert archive.stat().st_ino == original_inode
    assert archive.read_bytes() == _payload(A)


@pytest.mark.parametrize("state", ["active", "activating", "reloading", "failed", "unknown"])
def test_supersession_refuses_active_or_unproven_worker_state(monkeypatch, state):
    monkeypatch.setattr(module, "_verified_checkout_head", lambda: B)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(stdout=f"LoadState=loaded\nActiveState={state}\nMainPID=0\n"))
    with pytest.raises(module.IntentRegistryError, match="quiescence_unproven"):
        module._supersession_authority(B)


def test_supersession_checks_actual_release_and_every_trigger(monkeypatch):
    monkeypatch.setattr(module, "_verified_checkout_head", lambda: B)
    with pytest.raises(module.IntentRegistryError, match="execution_commit_mismatch"):
        module._supersession_authority(A)
    calls = []
    def stopped(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="LoadState=loaded\nActiveState=inactive\nMainPID=0\n")
    monkeypatch.setattr(module.subprocess, "run", stopped)
    module._supersession_authority(B)
    assert [call[2].rsplit(".", 1)[-1] for call in calls] == ["service", "path", "timer"]


def test_symlink_lock_cannot_redirect_serialization(registry):
    lock = registry.parent / ".owner.json.registry.lock"
    lock.unlink()
    lock.symlink_to(registry)
    with pytest.raises(OSError):
        _install(registry, B)
    assert registry.read_bytes() == _payload(A)
