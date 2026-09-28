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
    tick that does not apply sweeps no process table and hashes no report for them either, as the
    phase never does."""

    parent_root, names, size = _two_lookaheads(tmp_path)
    real_index, real_reference, real_sha = (
        retention.process_reference_index, retention.active_reference, retention.file_sha)

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
        monkeypatch.setattr(retention, "file_sha", refuse if not applying else real_sha)
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
    monkeypatch.setattr(retention, "file_sha", real_sha)
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


def test_shared_group_with_a_link_outside_the_lookaheads_is_kept_and_reported(tmp_path) -> None:
    """An inode with a link outside every lookahead (the store's own name, before a store moved;
    a derived directory's materialized reference) is kept: unlinking the names inside would free
    nothing, and the other name is not a replay's. Until 10a.2 nothing said so. Every such group
    is counted with its bytes, however many, and at most 200 are listed."""

    parent_root = tmp_path / "scene-configuration-activations"
    first = _replay(parent_root, "scene-841007-preparation", "parent-a-1")
    second = _replay(parent_root, "scene-841012-preparation", "parent-b-1")
    in_store = _store_blob(tmp_path, b"still named by the store" * 20)
    derived = _store_blob(tmp_path, b"still named by a derived directory" * 20)
    loose = _store_blob(tmp_path, b"named only in the two lookaheads" * 20)
    outside = tmp_path / "task-evaluation-inputs" / "prepared-references" / "prep-x" / "scene.usd"
    outside.parent.mkdir(parents=True)
    os.link(derived, outside)
    kept = [*_linked(in_store, first, "prep-a/scene.usd"), *_linked(in_store, second),
            *_linked(derived, first), *_linked(derived, second)]
    gone = [*_linked(loose, first), *_linked(loose, second)]
    many = [_store_blob(tmp_path, b"store blob %d" % index) for index in range(201)]
    for blob in many:
        kept += [*_linked(blob, first), *_linked(blob, second)]
    kept_bytes = sum(blob.stat().st_size for blob in (in_store, derived, *many))
    _moved(derived, loose)

    block = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]["shared_scratch"]

    assert block["kept_by_reason"] == {"linked_outside_lookaheads": {"groups": 203, "bytes": kept_bytes}}
    assert (len(block["kept"]), block["omitted_kept_count"]) == (200, 3)
    row = next(row for row in block["kept"] if row["path"] == str(kept[0]))
    assert row == {"reason": "linked_outside_lookaheads", "path": str(kept[0]), "name_count": 3, "holder_count": 2,
                   "nlink": 4, "size_bytes": in_store.stat().st_size}
    assert (block["candidate_groups"], block["removed_groups"]) == (1, 1)
    assert all(path.exists() for path in (*kept, in_store, outside)) and not any(path.exists() for path in gone)


def test_shared_group_is_kept_when_any_holder_is_active_or_unfinished(tmp_path, monkeypatch, process_table) -> None:
    """A replay holding a name is eligible exactly when the per-replay rule would take its scratch
    inputs: a finished parent report that says it ran there, with no paid execution and no provider
    mutation, closed for the phase's hour, and no live reader. One holder failing a gate keeps the
    whole group, and the report counts the holders by the gate each failed. An unreadable process
    inventory proves nothing is unreferenced, so it keeps everything."""

    from tests.test_completed_replay_cache_retention import _refuse_reading

    parent_root = tmp_path / "scene-configuration-activations"
    anchor = _replay(parent_root, "scene-841007-preparation", "parent-a-1")
    finished = _replay(parent_root, "scene-841007-preparation", "parent-a-2")
    holders = {
        "active": _replay(parent_root, "scene-841012-preparation", "parent-b-1"),
        "running": _replay(parent_root, "scene-841012-preparation", "parent-b-2"),
        "paid": _replay(parent_root, "scene-841019-preparation", "parent-c-1", paid_execution_requested=True),
        "recent": _replay(parent_root, "scene-841019-preparation", "parent-c-2", closed_seconds_ago=1800),
        # Before 2026-09-05 a parent report recorded no scratch queue, so it cannot say it ran there.
        "unbound": _replay(parent_root, "scene-841023-preparation", "parent-d-1", scratch_queue_root=None),
    }
    (holders["running"] / "stage_replay_report.v1.json").unlink()
    kept, sizes = [], 0
    for name, holder in holders.items():
        blob = _store_blob(tmp_path, f"shared with the {name} replay".encode() * 20)
        kept += [*_linked(blob, anchor, f"prep-a/{name}.usd"), *_linked(blob, holder)]
        sizes += blob.stat().st_size
        _moved(blob)
    between = _store_blob(tmp_path, b"held by two ineligible replays" * 20)
    kept += [*_linked(between, holders["running"]), *_linked(between, holders["recent"])]
    sizes += between.stat().st_size
    free = _store_blob(tmp_path, b"held by two finished replays" * 20)
    gone = [*_linked(free, anchor), *_linked(free, finished)]
    free_size = free.stat().st_size
    _moved(between, free)
    reader = process_table / "4242"
    (reader / "fd").mkdir(parents=True)
    (reader / "cmdline").write_bytes(b"python")
    (reader / "environ").write_bytes(b"")
    (reader / "fd" / "3").symlink_to(holders["active"] / "prepared-references")

    block = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]["shared_scratch"]

    assert block["kept_by_reason"] == {"holder_ineligible": {"groups": 6, "bytes": sizes}}
    assert block["holders_by_gate"] == {"eligible": 2, "active_reference": 1, "no_finished_report": 2,
                                        "closed_too_recently": 1, "report_not_parent_replay": 1}
    assert (block["removed_groups"], block["removed_bytes"]) == (1, free_size)
    assert all(path.exists() for path in kept) and not any(path.exists() for path in gone)

    last = _store_blob(tmp_path, b"held by two finished replays, checked blind" * 20)
    blind = [*_linked(last, anchor), *_linked(last, finished)]
    _moved(last)
    _refuse_reading(monkeypatch, reader / "environ", PermissionError(13, "Permission denied"))
    block = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]["shared_scratch"]

    assert block["kept_by_reason"]["holder_ineligible"]["groups"] == 7 and block["removed_groups"] == 0
    assert "eligible" not in block["holders_by_gate"]
    assert all(path.exists() for path in blind)


def test_shared_group_off_the_holders_device_or_over_counted_is_kept(tmp_path, monkeypatch) -> None:
    """Nothing on another filesystem than the replay holding it is planned: a group whose names show
    another st_dev than their replay (a file mounted into it) is kept as cross_device. A group whose
    names outnumber its links was counted twice somewhere (a bind mount of the same filesystem keeps
    its st_dev) and is kept too, rather than trusted."""

    from tests.test_completed_replay_cache_retention import _OnDevice, _Reported

    parent_root = tmp_path / "scene-configuration-activations"
    first = _replay(parent_root, "scene-841007-preparation", "parent-a-1")
    second = _replay(parent_root, "scene-841012-preparation", "parent-b-1")
    mounted = _store_blob(tmp_path, b"a file mounted into both replays" * 20)
    counted = _store_blob(tmp_path, b"a file seen twice through a bind mount" * 20)
    names = [*_linked(mounted, first), *_linked(mounted, second), *_linked(counted, first), *_linked(counted, second)]
    sizes = {blob: blob.stat().st_size for blob in (mounted, counted)}
    _moved(mounted, counted)
    real_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        if Path(path).name == mounted.name:
            return _OnDevice(info, info.st_dev + 1)
        return _Reported(info, st_nlink=1) if Path(path).name == counted.name else info

    monkeypatch.setattr(os, "lstat", lstat)
    block = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]["shared_scratch"]

    assert block["kept_by_reason"] == {"cross_device": {"groups": 1, "bytes": sizes[mounted]},
                                       "more_names_than_links": {"groups": 1, "bytes": sizes[counted]}}
    assert block["candidate_groups"] == block["removed_groups"] == 0
    assert all(real_lstat(path) for path in names)


def test_shared_scratch_recheck_refuses_a_swapped_or_extra_linked_name(tmp_path, monkeypatch) -> None:
    """Apply rechecks every name through the held replay before it goes, against the planned inode,
    size, mtime and a link count equal to the names still to go. A name swapped for another file, a
    link made anywhere after the plan, or a name gone since, stops its group before anything of it
    is unlinked, with a typed reason; the other groups still go."""

    from blueprint_pipeline import control_plane_replay_cache_shared_scratch as shared

    parent_root = tmp_path / "scene-configuration-activations"
    first = _replay(parent_root, "scene-841007-preparation", "parent-a-1")
    second = _replay(parent_root, "scene-841012-preparation", "parent-b-1")
    blobs = {name: _store_blob(tmp_path, f"a blob to be {name}".encode() * 20)
             for name in ("swapped", "linked", "vanished", "removed")}
    names = {name: [*_linked(blob, first, f"prep-a/{name}.usd"), *_linked(blob, second)]
             for name, blob in blobs.items()}
    sizes = {name: blob.stat().st_size for name, blob in blobs.items()}
    _moved(*blobs.values())
    real_plan = shared.plan_shared_scratch

    def plan_then_change(*args, **kwargs):
        plan = real_plan(*args, **kwargs)
        swapped = names["swapped"][0]
        swapped.unlink()
        swapped.write_bytes(b"a blob to be swapped" * 20)
        os.utime(swapped, (NOW - 9000, NOW - 9000))
        os.link(names["linked"][0], tmp_path / "linked-after-the-plan")
        names["vanished"][-1].unlink()
        return plan

    monkeypatch.setattr(shared, "plan_shared_scratch", plan_then_change)
    phase = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]

    block = phase["shared_scratch"]
    assert block["kept_by_reason"] == {
        "recheck_failed:changed": {"groups": 1, "bytes": sizes["swapped"]},
        "recheck_failed:extra_link": {"groups": 1, "bytes": sizes["linked"]},
        "recheck_failed:vanished": {"groups": 1, "bytes": sizes["vanished"]},
    }
    assert sorted((row["reason"], row["path"]) for row in block["kept"]) == [
        ("recheck_failed:changed", str(names["swapped"][0])),
        ("recheck_failed:extra_link", str(names["linked"][0])),
        ("recheck_failed:vanished", str(names["vanished"][0])),
    ]
    assert (block["candidate_groups"], block["removed_groups"]) == (4, 1)
    assert phase["candidate_bytes"] == sum(sizes.values()) and phase["removed_bytes"] == sizes["removed"]
    for name in ("swapped", "linked"):
        assert all(path.exists() for path in names[name]), name
    assert all(path.exists() for path in names["vanished"][:-1]) and (tmp_path / "linked-after-the-plan").exists()
    assert not any(path.exists() for path in names["removed"])


class _Killed(BaseException):
    """The unit stopped mid-apply (TimeoutStartSec, a reboot): nothing after it runs."""


@pytest.mark.parametrize("interruption", ["unlink_refused", "unit_killed"])
def test_interrupted_shared_reclaim_resumes_next_tick(tmp_path, monkeypatch, interruption) -> None:
    """Apply stops after some of a group's names are unlinked: the names already gone were scratch
    and stay gone, and the group's bytes are not counted, since the inode still lives. The next tick
    plans what is left, whose names are again all of its links: in two or more replays it is shared
    scratch again, in one replay it is the per-replay rule's. Either way it goes then, and its bytes
    are counted once across the ticks. A refused unlink is a typed skip and the tick goes on; a unit
    killed mid-apply writes no report at all."""

    parent_root = tmp_path / "scene-configuration-activations"
    first = _replay(parent_root, "scene-841007-preparation", "parent-a-1")
    second = _replay(parent_root, "scene-841012-preparation", "parent-b-1")
    third = _replay(parent_root, "scene-841019-preparation", "parent-c-1")
    wide = _store_blob(tmp_path, b"held by three replays" * 20)
    narrow = _store_blob(tmp_path, b"held by the last two replays" * 20)
    # Groups go in the order of their first names, so the wide group, first named in the first replay, goes first.
    wide_names = [*_linked(wide, first, "prep-a/wide.usd"), *_linked(wide, second), *_linked(wide, third)]
    narrow_names = [*_linked(narrow, second), *_linked(narrow, third)]
    sizes = {"wide": wide.stat().st_size, "narrow": narrow.stat().st_size}
    _moved(wide, narrow)
    real_unlink, unlinked = os.unlink, []

    def unlink(path, *args, **kwargs):
        # Each group stops at the second replay it is unlinked in.
        name = Path(path).name
        if name in (wide.name, narrow.name) and name in unlinked:
            if interruption == "unit_killed":
                raise _Killed()
            raise PermissionError(13, "Permission denied")
        unlinked.append(name)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    if interruption == "unit_killed":
        with pytest.raises(_Killed):
            _tick(tmp_path, parent_root, **BOTH)
        gone = wide_names[:2]
    else:
        phase = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]
        block = phase["shared_scratch"]
        assert (phase["removed_bytes"], block["removed_groups"], block["removed_bytes"]) == (0, 0, 0)
        assert block["kept_by_reason"] == {"recheck_failed:permission_error": {
            "groups": 2, "bytes": sizes["wide"] + sizes["narrow"]}}
        assert not any((first / "prepared-references").iterdir()), "what the first replay emptied is pruned"
        gone = [*wide_names[:2], narrow_names[0]]
    monkeypatch.setattr(os, "unlink", real_unlink)
    assert not any(path.exists() for path in gone)
    assert all(path.exists() for path in (*wide_names, *narrow_names) if path not in gone)

    phase = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]

    block = phase["shared_scratch"]
    # The narrow group's rest sits in one replay once an unlink was refused: the per-replay rule's.
    shared_rest = {"unit_killed": (2, sizes["wide"] + sizes["narrow"]), "unlink_refused": (1, sizes["wide"])}
    assert (block["removed_groups"], block["removed_bytes"]) == shared_rest[interruption]
    assert phase["removed_bytes"] == sizes["wide"] + sizes["narrow"]
    assert (block["kept_by_reason"], phase["errors"]) == ({}, [])
    assert not any(path.exists() for path in (*wide_names, *narrow_names))
    for child in (second, third):
        assert not any((child / "prepared-references").iterdir())
    assert _tick(tmp_path, parent_root, **BOTH)["replay_caches"]["removed_bytes"] == 0


SHARED_ENV = "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_SHARED_SCRATCH"


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_summary_reports_shared_scratch_and_its_opt_in(tmp_path, monkeypatch, capsys) -> None:
    """The unit reads the switch from the operator's environment file like every opt-in, records it
    in the report's and the summary's opt_in, and an invalid value only plans and alerts. The
    summary the door reads carries the shared block's counters and typed reasons, never a path, and
    the phase's candidate and removed totals include the shared bytes it removed. The unit never sets
    the switch itself, and the example environment documents it off."""

    from blueprint_pipeline import control_plane_storage_gc_reasons as reasons

    parent_root = tmp_path / "scene-configuration-activations"
    first = _replay(parent_root, "scene-841007-preparation", "parent-a-1")
    second = _replay(parent_root, "scene-841012-preparation", "parent-b-1")
    loose = _store_blob(tmp_path, b"named only in two lookaheads" * 20)
    in_store = _store_blob(tmp_path, b"still named by the store" * 30)
    names = [*_linked(loose, first), *_linked(loose, second), *_linked(in_store, first), *_linked(in_store, second)]
    sizes = {blob: blob.stat().st_size for blob in (loose, in_store)}
    _moved(loose)
    for name in (gc_module.CONTENT_STORE_ROOTS_ENV, gc_module.DERIVED_ROOTS_ENV, gc_module.PLAN_ONLY_DERIVED_ROOTS_ENV,
                 gc_module.QUEUE_ROOTS_ENV, gc_module.EVIDENCE_ROOTS_ENV, gc_module.SETTLEMENT_ROOTS_ENV,
                 gc_module.SCRATCH_ROOTS_ENV, gc_module.WORKSPACE_BUNDLE_ROOTS_ENV, gc_module.SCENE_WORKSPACE_ROOTS_ENV,
                 gc_module.EVIDENCE_OFFLOAD_ENV, gc_module.SCENE_WORKSPACE_RETIREMENT_ENV):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv(replay_gc.REPLAY_PARENT_ROOTS_ENV, str(parent_root))
    monkeypatch.setenv(replay_gc.REPLAY_CACHE_RETENTION_ENV, "1")
    monkeypatch.setattr(gc_module, "require_storage_class", _noclass)
    report_dir = tmp_path / "storage-gc"
    command = ["run", "--apply", "--ack", RUN_ACK, "--pins-root", str(tmp_path / "pins"),
               "--report-out", str(report_dir / "latest.json")]

    monkeypatch.setenv(SHARED_ENV, "sometimes")
    assert gc_module.main(command) == 0
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert "storage_gc_alert:replay_cache_shared_scratch_setting_invalid" in capsys.readouterr().err
    assert summary["alerts"] == ["replay_cache_shared_scratch_setting_invalid"]
    assert (summary["opt_in"]["replay_cache_retention"], summary["opt_in"]["replay_cache_shared_scratch"]) == (
        True, False)
    assert summary["phases"]["replay_caches"]["shared_scratch"]["enabled"] is False
    assert all(path.exists() for path in names)

    monkeypatch.setenv(SHARED_ENV, "1")
    assert gc_module.main(command) == 0
    report = json.loads((report_dir / "latest.json").read_text(encoding="utf-8"))
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))

    assert report["opt_in"]["replay_cache_shared_scratch"] is summary["opt_in"]["replay_cache_shared_scratch"] is True
    assert set(summary["opt_in"]) == set(reasons.OPT_INS) and "replay_cache_shared_scratch" in reasons.OPT_INS
    phase = summary["phases"]["replay_caches"]
    assert (phase["status"], phase["candidate_bytes"], phase["removed_or_offloaded_bytes"]) == (
        "applied", sizes[loose], sizes[loose])
    assert phase["shared_scratch"] == {
        "enabled": True, "status": "applied", "live_readers_checked": True,
        "candidate_groups": 1, "candidate_bytes": sizes[loose], "removed_groups": 1, "removed_bytes": sizes[loose],
        "kept_by_reason": {"linked_outside_lookaheads": {"groups": 1, "bytes": sizes[in_store]}},
        "holders_by_gate": {"eligible": 2},
    }
    assert report["replay_caches"]["shared_scratch"]["kept"][0]["path"] == str(names[2])
    assert not [value for value in _strings(summary) if "/" in value], "the summary names no path"
    assert not any(path.exists() for path in names[:2]) and all(path.exists() for path in names[2:])
    # A report from before the shared rule does not claim there was none, nor that its switch was off.
    older = reasons.build_storage_gc_summary({"status": "applied", "replay_caches": {
        "status": "applied", "candidate_bytes": 0, "removed_bytes": 0, "kept": []}})
    assert "shared_scratch" not in older["phases"]["replay_caches"]
    assert older["opt_in"]["replay_cache_shared_scratch"] is None

    root = Path(__file__).resolve().parents[1]
    unit = (root / "deploy/systemd/blueprint-control-plane-storage-gc.service").read_text(encoding="utf-8")
    example = (root / "deploy/systemd/pipeline-control-plane.env.example").read_text(encoding="utf-8")
    assert replay_gc.REPLAY_CACHE_SHARED_SCRATCH_ENV == SHARED_ENV
    assert f"Environment={SHARED_ENV}=" not in unit, "the shared switch stays an operator opt-in"
    assert f"# {SHARED_ENV}=1" in example and not any(row.startswith(f"{SHARED_ENV}=") for row in example.splitlines())
    assert example.index("# BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_RETENTION=1") < example.index(f"# {SHARED_ENV}=1")


def test_shared_scratch_reports_each_directory_its_prune_kept(tmp_path, monkeypatch) -> None:
    """The prune after a shared removal is the per-replay rule's own: a directory it cannot remove
    stays and is a typed skip, now listed in the shared block, at most 200 with the number left out."""

    parent_root = tmp_path / "scene-configuration-activations"
    first = _replay(parent_root, "scene-841007-preparation", "parent-a-1")
    second = _replay(parent_root, "scene-841012-preparation", "parent-b-1")
    blob = _store_blob(tmp_path, b"one blob two activations linked" * 50)
    names = [*_linked(blob, first, "prep-a/scene.usd"), *_linked(blob, second)]
    _moved(blob)
    real_rmdir = os.rmdir

    def rmdir(path, *args, **kwargs):
        if path == "sha256":
            raise PermissionError(13, "Permission denied")
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "rmdir", rmdir)
    block = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]["shared_scratch"]

    stuck = [child / "prepared-references" / "content-addressed" / "sha256" for child in (first, second)]
    assert block["prune_skipped"] == [{"path": str(path), "reason": "rmdir_failed:PermissionError"} for path in stuck]
    assert (block["omitted_prune_skipped_count"], block["removed_groups"]) == (0, 1)
    assert all(path.is_dir() for path in stuck) and not (first / "prepared-references" / "prep-a").exists()
    assert not any(path.exists() for path in names)


def test_a_shared_scan_that_fails_costs_no_lookahead_its_own_pass(tmp_path, monkeypatch) -> None:
    """The shared plan runs after every lookahead's own pass and is isolated from it: if it raises,
    what each replay held alone is still reclaimed, the phase records the failure by type as one of
    its errors, and the summary then leaves the phase's candidate bytes unknown."""

    from blueprint_pipeline import control_plane_replay_cache_shared_scratch as shared
    from blueprint_pipeline import control_plane_storage_gc_reasons as reasons

    parent_root, names, _size = _two_lookaheads(tmp_path)
    alone = names[0].parent.parent.parent / "prep-a" / "derived.json"
    alone.write_bytes(b"{}" * 200)
    os.utime(alone, (NOW - 9000, NOW - 9000))

    def fail(*_args, **_kwargs):
        raise OSError(5, "I/O error")

    monkeypatch.setattr(shared, "plan_shared_scratch", fail)
    tick = _tick(tmp_path, parent_root, **BOTH)

    phase = tick["replay_caches"]
    assert phase["removed_bytes"] == len(b"{}" * 200) and not alone.exists()
    assert phase["shared_scratch"] == {"enabled": True, "status": "error", "error": "OSError"}
    assert phase["errors"] == [{"scope": "shared_scratch", "error": "OSError"}] and "phase_errors" not in tick
    assert all(path.exists() for path in names)
    summary = reasons.build_storage_gc_summary(tick)["phases"]["replay_caches"]
    assert summary["candidate_bytes"] is None
    assert summary["shared_scratch"]["status"] == "error" and summary["shared_scratch"]["candidate_bytes"] is None


def _after_plan(monkeypatch, change) -> None:
    """``change()`` runs once the tick's shared plan is made, before its apply."""

    from blueprint_pipeline import control_plane_replay_cache_shared_scratch as shared

    real_plan = shared.plan_shared_scratch

    def plan_then_change(*args, **kwargs):
        plan = real_plan(*args, **kwargs)
        change()
        return plan

    monkeypatch.setattr(shared, "plan_shared_scratch", plan_then_change)


def _reader(process_table: Path, pid: str, target: Path) -> Path:
    reader = process_table / pid
    (reader / "fd").mkdir(parents=True)
    (reader / "cmdline").write_bytes(b"python")
    (reader / "environ").write_bytes(b"")
    (reader / "fd" / "3").symlink_to(target)
    return reader


def test_a_reader_that_opens_a_holder_after_the_plan_keeps_the_group_whole(
        tmp_path, monkeypatch, process_table) -> None:
    """Apply sweeps the process table again before anything goes: a replay a process opened after
    the plan is no longer eligible, and every group it holds keeps every name, in every replay,
    though its first names sit in a replay nobody reads."""

    parent_root, names, size = _two_lookaheads(tmp_path)
    second = names[-1].parents[3]
    _after_plan(monkeypatch, lambda: _reader(process_table, "4242", second / "prepared-references"))

    block = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]["shared_scratch"]

    assert block["kept_by_reason"] == {"recheck_failed:holder_ineligible": {"groups": 1, "bytes": size}}
    assert (block["candidate_groups"], block["removed_groups"]) == (1, 0)
    assert all(path.exists() for path in names)


def test_a_directory_swapped_for_a_link_while_rechecking_removes_nothing(tmp_path, monkeypatch) -> None:
    """A directory above a planned name swapped for a link to a directory holding a file of the same
    name, after apply opened it and while it rechecks the name, is found by the held chain's check:
    the group is kept as path_changed, nothing of it is unlinked, the file behind the link is not the
    replay's and survives, and so does the planned file where it was moved."""

    parent_root, names, size = _two_lookaheads(tmp_path)
    first = names[0].parents[3]
    content = first / "prepared-references" / "content-addressed"
    victim = tmp_path / "evidence" / "content-addressed"
    (victim / "sha256").mkdir(parents=True)
    (victim / "sha256" / names[0].name).write_bytes(b"evidence that is not the replay's")
    real_leaf = retention._leaf

    def leaf(directory, name):
        if name == names[0].name and not content.is_symlink():
            content.rename(content.with_name("content-addressed-moved"))
            content.symlink_to(victim, target_is_directory=True)
        return real_leaf(directory, name)

    monkeypatch.setattr(retention, "_leaf", leaf)
    block = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]["shared_scratch"]

    assert block["kept_by_reason"] == {"recheck_failed:path_changed": {"groups": 1, "bytes": size}}
    assert (victim / "sha256" / names[0].name).read_bytes() == b"evidence that is not the replay's"
    assert (content.with_name("content-addressed-moved") / "sha256" / names[0].name).stat().st_nlink == len(names)
    assert all(path.exists() for path in names[1:])


@pytest.mark.parametrize("rewrite", ["same_mtime", "new_mtime"])
def test_a_holders_report_rewritten_after_the_plan_keeps_the_group_whole(tmp_path, monkeypatch, rewrite) -> None:
    """Apply requires each holder's report to be the one the plan read: the same path, mtime and
    bytes, as the per-replay rule requires its report's digest. A report rewritten after the plan
    keeps every group its replay holds, even one that still passes every gate under the same mtime."""

    parent_root, names, size = _two_lookaheads(tmp_path)
    report = names[-1].parents[3] / "stage_replay_report.v1.json"
    before = report.stat()

    def rewritten() -> None:
        report.write_text(json.dumps({**json.loads(report.read_text()), "row": {"status": "rewritten"}}))
        mtime_ns = before.st_mtime_ns if rewrite == "same_mtime" else before.st_mtime_ns - 10**9
        os.utime(report, ns=(before.st_atime_ns, mtime_ns))

    _after_plan(monkeypatch, rewritten)
    block = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]["shared_scratch"]

    assert block["kept_by_reason"] == {"recheck_failed:holder_ineligible": {"groups": 1, "bytes": size}}
    assert block["removed_groups"] == 0 and all(path.exists() for path in names)


def test_a_holder_swapped_for_a_link_after_the_plan_keeps_the_group_whole(tmp_path, monkeypatch) -> None:
    """A replay holding a candidate must still be the directory the plan walked, not a link to it:
    one swapped for a link after the plan keeps every group it holds as path_changed, nothing of
    them unlinked in any replay, though its report still reads the same through the link."""

    parent_root, names, size = _two_lookaheads(tmp_path)
    second = names[-1].parents[3]
    moved = second.with_name(second.name + "-moved")

    def swapped() -> None:
        second.rename(moved)
        second.symlink_to(moved, target_is_directory=True)

    _after_plan(monkeypatch, swapped)
    block = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]["shared_scratch"]

    assert block["kept_by_reason"] == {"recheck_failed:path_changed": {"groups": 1, "bytes": size}}
    assert block["removed_groups"] == 0 and all(path.exists() for path in names)
    assert (moved / "stage_replay_report.v1.json").is_file()


def test_a_reader_that_appears_between_two_replays_removals_keeps_the_later_replays_names(
        tmp_path, monkeypatch, process_table) -> None:
    """Apply sweeps the process table again just before each replay's own removals, as the per-replay
    rule does before each replay's, and never trusts a sweep taken before another replay's removals.
    A reader that opens a replay while an earlier one loses its names keeps that replay's names: the
    group is kept as holder_ineligible and its bytes live on there, uncounted. The next tick plans
    the rest, here one replay's own, and its bytes are counted then, once."""

    parent_root, names, size = _two_lookaheads(tmp_path)
    second = names[-1].parents[3]
    real_unlink, readers = os.unlink, []

    def unlink(path, *args, **kwargs):
        if not readers:
            readers.append(_reader(process_table, "4242", second / "prepared-references"))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    phase = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]
    monkeypatch.setattr(os, "unlink", real_unlink)

    block = phase["shared_scratch"]
    assert block["kept_by_reason"] == {"recheck_failed:holder_ineligible": {"groups": 1, "bytes": size}}
    assert (block["removed_groups"], phase["removed_bytes"]) == (0, 0)
    assert not any(path.exists() for path in names[:-1]) and names[-1].exists()

    for entry in sorted(readers[0].rglob("*"), reverse=True):
        entry.unlink() if entry.is_symlink() or entry.is_file() else entry.rmdir()
    readers[0].rmdir()
    phase = _tick(tmp_path, parent_root, **BOTH)["replay_caches"]
    assert (phase["removed_bytes"], phase["shared_scratch"]["removed_bytes"]) == (size, 0)
    assert not names[-1].exists()


def test_a_failed_shared_scan_leaves_the_retention_switchs_numbers_alone(tmp_path, monkeypatch) -> None:
    """Until the shared removal applies, the shared pass only reports, so a failure of it is its own
    block's and nothing else's: the phase has no error for it, and the phase and its summary say
    exactly what they say without the pass (the estimate, the candidate and the removed bytes the
    capacity controller reads). Only a tick that removes shared scratch counts its failure as one of
    the phase's errors."""

    from blueprint_pipeline import control_plane_replay_cache_shared_scratch as shared
    from blueprint_pipeline import control_plane_storage_gc_reasons as reasons

    def fixture(base: Path) -> Path:
        parent_root, names, _size = _two_lookaheads(base)
        alone = names[0].parents[3] / "prepared-references" / "prep-a" / "derived.json"
        alone.write_bytes(b"{}" * 200)
        os.utime(alone, (NOW - 9000, NOW - 9000))
        return parent_root

    failing, base = fixture(tmp_path / "failing"), fixture(tmp_path / "base")
    real_plan = shared.plan_shared_scratch

    def fail(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    def without_the_pass(tick: dict) -> tuple[dict, dict]:
        phase = {key: value for key, value in tick["replay_caches"].items() if key != "shared_scratch"}
        summary = dict(reasons.build_storage_gc_summary(tick)["phases"]["replay_caches"])
        summary.pop("shared_scratch")
        return phase, summary

    for switches in ({}, {"apply": True, "ack": RUN_ACK, "replay_cache_retention_enabled": True}):
        monkeypatch.setattr(shared, "plan_shared_scratch", fail)
        tick = _tick(tmp_path, failing, **switches)
        monkeypatch.setattr(shared, "plan_shared_scratch", real_plan)

        assert tick["replay_caches"]["shared_scratch"] == {"enabled": False, "status": "error",
                                                           "error": "PermissionError"}
        summary = reasons.build_storage_gc_summary(tick)["phases"]["replay_caches"]["shared_scratch"]
        assert (summary["status"], summary["error_type"]) == ("error", "PermissionError")
        assert tick["replay_caches"]["errors"] == [] and "phase_errors" not in tick
        assert without_the_pass(tick) == without_the_pass(_tick(tmp_path, base, **switches))
    assert tick["replay_caches"]["candidate_bytes"] == tick["replay_caches"]["removed_bytes"] == len(b"{}" * 200)
