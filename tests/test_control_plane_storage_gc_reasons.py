"""The storage GC names why it kept what it kept: typed retention reasons, with their bytes."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_storage_gc_reasons.py

from __future__ import annotations

import errno
import functools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import completed_replay_cache_retention as retention
from blueprint_pipeline import control_plane_storage_gc_reasons as reasons
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, run_storage_gc
from blueprint_pipeline.control_plane_storage_pins import live_pinned_paths, release_storage_pin, write_storage_pin
from tests.test_completed_replay_cache_retention import _refuse_reading
from tests.test_task_evaluation_configured_scene_object_store import _ContentAddressedClient

NOW = 30_000_000.0
# The real sweep, before the fixture below replaces it with an empty process table.
REAL_PROCESS_REFERENCE = retention.process_reference


@pytest.fixture(autouse=True)
def isolated_disk_ledger(tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk

    monkeypatch.setattr("blueprint_pipeline.control_plane_evidence_offload.reserve_control_plane_disk",
                        functools.partial(reserve_control_plane_disk,
                            disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, free=80 * 1024**3)))
    monkeypatch.setattr("blueprint_pipeline.control_plane_evidence_offload.DEFAULT_RESERVATION_ROOT",
                        tmp_path / "disk-reservations")
    # No process on this host references anything unless a test says so.
    monkeypatch.setattr(retention, "process_reference", lambda _root, **_kwargs: None)


def _noclass(*_args, **_kwargs) -> None:
    return None


def _cold_run(evidence: Path, name: str, *, size: int = 4096) -> Path:
    run = evidence / name
    run.mkdir(parents=True)
    (run / "launch_receipt.json").write_text("{}", encoding="utf-8")
    (run / "frames.bin").write_bytes(b"f" * size)
    old = NOW - 30 * 86400
    for path in (run / "launch_receipt.json", run / "frames.bin", run):
        os.utime(path, (old, old))
    return run


def _settlement(root: Path, run_name: str, reopened: str) -> Path:
    record = root / "scene-a" / "cancelled-unstarted-controls" / "controls-1.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"launch_receipt": {
        "path": f"/var/lib/blueprint/launch-runs/{run_name}/{reopened}"}}), encoding="utf-8")
    return record


def test_evidence_protection_reason_keeps_the_existing_order(tmp_path, monkeypatch) -> None:
    """Each check still runs in the order the closure had; the first that holds names the reason."""

    run = _cold_run(tmp_path / "launch-runs", "run-1")
    settlement = tmp_path / "scene-intents"
    _settlement(settlement, "run-1", "launch_profile.json")
    pins = tmp_path / "pins"
    write_storage_pin(pins_root=pins, kind="activation", owner_id="act-1", paths=[run], now=lambda: NOW)
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    (queue / "pending" / "row.json").write_text(json.dumps({"run": "run-1"}), encoding="utf-8")
    process = {"answer": retention.PROCESS_REFERENCED}
    monkeypatch.setattr(retention, "process_reference", lambda _root, **_kwargs: process["answer"])

    def reason(settlement_roots=(settlement,)):
        return reasons.evidence_protection_reason(
            run, settlement_roots=settlement_roots, pins_root=pins, queue_roots=[queue], now=lambda: NOW)

    assert reason(settlement_roots=(tmp_path / "missing-scene-intents",)) == "protected_unreadable_settlement"
    assert reason() == "protected_process"
    process["answer"] = retention.PROCESS_INVENTORY_UNREADABLE
    assert reason() == "protected_process_inventory_unreadable"
    process["answer"] = None
    assert reason() == "protected_pin"
    release_storage_pin(pins_root=pins, kind="activation", owner_id="act-1", now=lambda: NOW)
    assert reason() == "protected_settlement"
    _settlement(settlement, "run-1", "launch_receipt.json")  # retained by the pointer: not a reopen
    assert reason() == "protected_queue"
    (queue / "pending" / "row.json").unlink()
    assert reason() is None
    assert reasons.EVIDENCE_PROTECTION_REASONS == (
        "protected_unreadable_settlement", "protected_process", "protected_process_inventory_unreadable",
        "protected_pin", "protected_settlement", "protected_queue")


def test_unreadable_process_inventory_protects_with_a_reason(tmp_path, monkeypatch) -> None:
    """2026-09-27: six registry runs failed result-artifact offload with a bare PermissionError.

    A /proc entry the GC may not read now protects the run, and the tick says so
    instead of raising.
    """

    evidence = tmp_path / "launch-runs"
    run = _cold_run(evidence, "run-1", size=5000)
    proc = tmp_path / "proc"
    (proc / "4242" / "fd").mkdir(parents=True)
    (proc / "4242" / "cmdline").write_bytes(b"python")
    (proc / "4242" / "environ").write_bytes(b"")
    _refuse_reading(monkeypatch, proc / "4242" / "environ", PermissionError(errno.EACCES, "Permission denied"))
    # The tick sweeps the fake process table instead of this host's.
    monkeypatch.setattr(retention, "process_reference", lambda root, *, process_root=None, ignored_process_ids=():
                        REAL_PROCESS_REFERENCE(root, process_root=proc, ignored_process_ids=ignored_process_ids))
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    common = dict(content_store_roots=[], derived_roots=[], queue_roots=[queue], pins_root=tmp_path / "pins",
                  evidence_roots=[evidence], now=lambda: NOW, classifier=_noclass)

    assert retention.active_reference(run, process_root=proc) is True
    assert reasons.evidence_protection_reason(
        run, settlement_roots=(), pins_root=tmp_path / "pins", queue_roots=[queue], now=lambda: NOW,
    ) == "protected_process_inventory_unreadable"
    planned = run_storage_gc(**common)
    assert planned["evidence_offload"]["retained_by_reason"] == {
        "protected_process_inventory_unreadable": {"count": 1, "bytes": 5002}}
    applied = run_storage_gc(**common, apply=True, ack=RUN_ACK, offload_enabled=True, publisher=functools.partial(
        store.publish_configured_scene_artifact, client=_ContentAddressedClient(), bucket="blueprint-production-inputs"))
    assert "phase_errors" not in applied
    assert applied["evidence_offload"]["offloaded_count"] == 0
    assert run.is_dir() and (run / "frames.bin").stat().st_size == 5000


def test_pin_kinds_name_exactly_the_live_pinned_paths(tmp_path) -> None:
    """The kind map only labels what ``live_pinned_paths`` pins; it never decides."""

    pins = tmp_path / "pins"
    first, shared, expired = (tmp_path / name for name in ("first", "shared", "expired"))
    write_storage_pin(pins_root=pins, kind="activation", owner_id="act", paths=[first, shared], now=lambda: NOW)
    write_storage_pin(pins_root=pins, kind="preparation", owner_id="prep", paths=[shared], now=lambda: NOW)
    write_storage_pin(pins_root=pins, kind="compilation", owner_id="comp", paths=[expired], now=lambda: NOW,
                      ttl_seconds=1)

    kinds = reasons.live_pin_kinds(pins, now=lambda: NOW + 10)

    assert kinds == {str(first): "activation", str(shared): "activation+preparation"}
    assert set(kinds) == live_pinned_paths(pins, now=lambda: NOW + 10)
