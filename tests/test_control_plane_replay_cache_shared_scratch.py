"""Storage GC reclaims lookahead scratch that several lookaheads share, only under its own opt-in."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_replay_cache_shared_scratch.py
#   src/blueprint_pipeline/control_plane_replay_cache_gc.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
#   src/blueprint_pipeline/control_plane_storage_gc_reasons.py
#   src/blueprint_pipeline/completed_replay_cache_retention.py
#   deploy/systemd/pipeline-control-plane.env.example
#   deploy/systemd/blueprint-control-plane-storage-gc.service

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import completed_replay_cache_retention as retention
from blueprint_pipeline import control_plane_replay_cache_gc as replay_gc
from blueprint_pipeline import control_plane_storage_gc as gc_module
from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, run_storage_gc

NOW = 20_000_000.0
BOTH = {"apply": True, "ack": RUN_ACK, "replay_cache_retention_enabled": True,
        "replay_cache_shared_scratch_enabled": True}


@pytest.fixture(autouse=True)
def process_table(tmp_path, monkeypatch) -> Path:
    """A process table the tick really sweeps, empty unless a test adds a reader."""

    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(gc_module, "reclaim_replay_caches",
                        functools.partial(replay_gc.reclaim_replay_caches, process_root=proc))
    return proc


def _noclass(*_args, **_kwargs) -> None:
    return None


def _replay(parent_root: Path, activation: str, name: str, *, closed_seconds_ago: int = 7100,
            **report: object) -> Path:
    """A parent replay in an activation's lookahead with the report replay_parent writes."""

    child = parent_root / activation / "lookahead" / name
    (child / "prepared-references").mkdir(parents=True)
    (child / "launch-preparations" / "pending").mkdir(parents=True)
    path = child / "stage_replay_report.v1.json"
    path.write_text(json.dumps({
        "schema_version": "task_evaluation_parent_replay_report.v1", "nothing_fetched": True,
        "paid_execution_requested": False, "provider_mutation_performed": False,
        "scratch_queue_root": str(child / "launch-preparations"), **report,
    }))
    os.utime(path, (NOW - closed_seconds_ago, NOW - closed_seconds_ago))
    return child


def _store_blob(tmp_path: Path, payload: bytes) -> Path:
    """A content-store blob as it sat on the root disk before 2026-09-20, older than every report."""

    store = tmp_path / "root-disk-store" / "sha256"
    store.mkdir(parents=True, exist_ok=True)
    blob = store / hashlib.sha256(payload).hexdigest()
    blob.write_bytes(payload)
    blob.chmod(0o440)
    os.utime(blob, (NOW - 9000, NOW - 9000))
    return blob


def _linked(blob: Path, child: Path, *materialized: str) -> list[Path]:
    """What a replay made before the move: the store's own inode under its digest name in the replay's
    scratch inputs, and the worker's materialized references linked to it."""

    names = [child / "prepared-references" / "content-addressed" / "sha256" / blob.name]
    names += [child / "prepared-references" / relative for relative in materialized]
    for name in names:
        name.parent.mkdir(parents=True, exist_ok=True)
        os.link(blob, name)
    return names


def _moved(*blobs: Path) -> None:
    """2026-09-20: the store moved to the work volume and its root copies were deleted."""

    for blob in blobs:
        blob.unlink()


def _tick(tmp_path: Path, parent_root: Path, **kwargs) -> dict:
    return run_storage_gc(content_store_roots=[], derived_roots=[], queue_roots=[], pins_root=tmp_path / "pins",
                          now=lambda: NOW, replay_parent_roots=[parent_root], classifier=_noclass, **kwargs)


def test_shared_lookahead_scratch_is_reclaimed_when_every_link_is_in_finished_lookaheads(
        tmp_path, monkeypatch) -> None:
    """2026-09-28: 89 lookahead replays still held scratch after an applied tick, and 957 of their
    964 blob digests (9.07 GB) sat in more than one of them. Replays made before the store moved
    hard-linked its own inodes, and its root copies were deleted, so each inode now lives only in
    names spread across lookaheads; the per-replay rule plans an inode only when one replay holds
    all of its links, so it never took them. With both opt-ins an inode whose every link is in
    finished lookaheads goes with all of its names and counts once, beside what one replay holds
    alone; the directories left empty go, and every report and scratch queue stays. Nothing about
    it rests on a digest, so no byte is read."""

    parent_root = tmp_path / "scene-configuration-activations"
    first = _replay(parent_root, "scene-841007-preparation", "parent-a-20260915T000000Z-1")
    second = _replay(parent_root, "scene-841007-preparation", "parent-a-20260916T000000Z-2")
    other = _replay(parent_root, "scene-841012-preparation", "parent-b-20260917T000000Z-1")
    everywhere = _store_blob(tmp_path, b"a blob every lookahead linked" * 40)
    two = _store_blob(tmp_path, b"a blob two lookaheads linked" * 30)
    names = [
        *_linked(everywhere, first, "scene-841007-preparation/inputs/scene.usd"),
        *_linked(everywhere, second),
        *_linked(everywhere, other, "scene-841012-preparation/inputs/scene.usd"),
        *_linked(two, first), *_linked(two, other),
    ]
    alone = first / "prepared-references" / "scene-841007-preparation" / "derived" / "view.json"
    alone.parent.mkdir(parents=True)
    alone.write_bytes(b"{}" * 300)
    os.utime(alone, (NOW - 9000, NOW - 9000))
    shared_bytes, alone_bytes = everywhere.stat().st_size + two.stat().st_size, alone.stat().st_size
    _moved(everywhere, two)

    def refuse(*args, **_kwargs):
        raise AssertionError(f"read {args[:2]}: nothing about shared scratch rests on its bytes")

    monkeypatch.setattr(retention, "_held_sha", refuse)
    tick = _tick(tmp_path, parent_root, **BOTH)

    phase = tick["replay_caches"]
    block = phase["shared_scratch"]
    assert (block["candidate_groups"], block["candidate_bytes"]) == (2, shared_bytes)
    assert (block["removed_groups"], block["removed_bytes"]) == (2, shared_bytes)
    # The phase's totals carry both rules, each inode once.
    assert phase["candidate_bytes"] == phase["removed_bytes"] == shared_bytes + alone_bytes
    assert (phase["errors"], phase["skipped"], "phase_errors" in tick) == ([], [], False)
    assert not any(path.exists() for path in (*names, alone))
    for child in (first, second, other):
        inputs = child / "prepared-references"
        assert inputs.is_dir() and not any(inputs.iterdir()), "every emptied directory went; the tree stays"
        assert (child / "stage_replay_report.v1.json").is_file() and (child / "launch-preparations" / "pending").is_dir()

    again = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]
    assert (again["removed_bytes"], again["shared_scratch"]["candidate_groups"]) == (0, 0)


def _two_lookaheads(tmp_path: Path) -> tuple[Path, list[Path], int]:
    """Two activations whose finished lookaheads linked one store blob before the move."""

    parent_root = tmp_path / "scene-configuration-activations"
    blob = _store_blob(tmp_path, b"one blob two activations linked" * 50)
    names = [*_linked(blob, _replay(parent_root, "scene-841007-preparation", "parent-a-1"), "prep-a/scene.usd"),
             *_linked(blob, _replay(parent_root, "scene-841012-preparation", "parent-b-1"))]
    size = blob.stat().st_size
    _moved(blob)
    return parent_root, names, size


def test_shared_scratch_is_plan_only_until_its_own_opt_in(tmp_path, monkeypatch) -> None:
    """The owner's replay cache switch must not start deleting anything new on its own, and the new
    switch does nothing without it or on a tick that does not apply. Until both are on and the tick
    applies, the phase plans and reports the shared groups with ``enabled`` saying whether the owner
    has enabled them, and they stay out of the phase's totals, which say what the tick reclaims. A
    tick that does not apply sweeps no process table for them either, as the phase never does."""

    parent_root, names, size = _two_lookaheads(tmp_path)
    real_index, real_reference = retention.process_reference_index, retention.active_reference

    def refuse(*args, **_kwargs):
        raise AssertionError(f"swept the process table for {args[:1]} on a tick that only plans")

    for switches, tick_applies, enabled in (
        ({"replay_cache_retention_enabled": True}, True, False),
        ({"replay_cache_shared_scratch_enabled": True}, True, False),
        ({"replay_cache_retention_enabled": True, "replay_cache_shared_scratch_enabled": True}, False, True),
        ({}, False, False),
    ):
        applying = tick_applies and switches.get("replay_cache_retention_enabled", False)
        sweeps: list[object] = []
        monkeypatch.setattr(retention, "active_reference", refuse if not applying else real_reference)
        monkeypatch.setattr(retention, "process_reference_index", refuse if not applying else (
            lambda **kwargs: sweeps.append(kwargs) or real_index(**kwargs)))
        tick = _tick(tmp_path, parent_root, apply=tick_applies, ack=RUN_ACK if tick_applies else "", **switches)

        phase = tick["replay_caches"]
        block = phase["shared_scratch"]
        assert (block["enabled"], block["status"], block["live_readers_checked"]) == (enabled, "dry_run", applying)
        assert (block["candidate_groups"], block["candidate_bytes"]) == (1, size)
        assert (block["removed_groups"], block["removed_bytes"]) == (0, 0)
        assert phase["candidate_bytes" if applying else "estimated_candidate_bytes"] == 0
        assert (phase["removed_bytes"], phase["errors"], "phase_errors" in tick) == (0, [], False)
        assert len(sweeps) == int(applying), "a tick that applies checks the lookaheads' readers once"
        assert all(path.exists() for path in names)

    monkeypatch.setattr(retention, "active_reference", real_reference)
    monkeypatch.setattr(retention, "process_reference_index", real_index)
    phase = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]
    block = phase["shared_scratch"]
    assert (block["enabled"], block["status"], block["live_readers_checked"]) == (True, "applied", True)
    assert block["removed_bytes"] == phase["removed_bytes"] == phase["candidate_bytes"] == size
    assert not any(path.exists() for path in names)


def _stamped(blob: Path, seconds: float) -> Path:
    os.utime(blob, ns=(int(seconds * 10**9), int(seconds * 10**9)))
    return blob


def test_shared_group_newer_than_a_holders_report_is_kept(tmp_path) -> None:
    """Every replay holding a name must have written its report after the inode last changed, as
    the per-replay rule requires of its own: a file written after any holder's report is not what
    that replay left behind. The gate is not newer, so an inode exactly as old as a report goes."""

    parent_root = tmp_path / "scene-configuration-activations"
    early = _replay(parent_root, "scene-841007-preparation", "parent-a-1", closed_seconds_ago=7100)
    late = _replay(parent_root, "scene-841012-preparation", "parent-b-1", closed_seconds_ago=5000)
    other = _replay(parent_root, "scene-841019-preparation", "parent-c-1", closed_seconds_ago=7100)
    newer = _stamped(_store_blob(tmp_path, b"changed after the early report" * 20), NOW - 7099)
    as_old = _stamped(_store_blob(tmp_path, b"exactly as old as the reports" * 20), NOW - 7100)
    kept_names = [*_linked(newer, early, "prep-a/newer.usd"), *_linked(newer, late)]
    gone_names = [*_linked(as_old, early), *_linked(as_old, other)]
    sizes = {blob: blob.stat().st_size for blob in (newer, as_old)}
    _moved(newer, as_old)

    block = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]["shared_scratch"]

    assert block["kept_by_reason"] == {"newer_than_report": {"groups": 1, "bytes": sizes[newer]}}
    assert block["kept"] == [{"reason": "newer_than_report", "path": str(kept_names[0]), "name_count": 3,
                              "holder_count": 2, "nlink": 3, "size_bytes": sizes[newer]}]
    assert block["candidates"] == [{"path": str(gone_names[0]), "name_count": 2, "holder_count": 2, "nlink": 2,
                                    "size_bytes": sizes[as_old]}]
    assert (block["omitted_kept_count"], block["omitted_candidates_count"]) == (0, 0)
    assert block["removed_bytes"] == sizes[as_old]
    assert all(path.exists() for path in kept_names) and not any(path.exists() for path in gone_names)
