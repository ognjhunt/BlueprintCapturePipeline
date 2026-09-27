from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import shutil
import signal
import stat
import time
from pathlib import Path

import pytest

from blueprint_pipeline import completed_replay_cache_retention as gc


def setup(tmp_path):
    root = tmp_path / "replays"
    child = root / "parent-finished"
    child.mkdir(parents=True)
    data = child / "working.ply"
    data.write_bytes(b"x" * 100000)
    readonly = child / "readonly.ply"
    readonly.write_bytes(b"y" * 100000)
    readonly.chmod(0o440)
    original = tmp_path / "original.ply"
    original.write_bytes(b"z" * 100000)
    os.link(original, child / "shared.ply")
    (child / "run.log").write_text("retained log")
    report = child / "stage_replay_report.v1.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "task_evaluation_parent_replay_report.v1",
                "nothing_fetched": True,
                "paid_execution_requested": False,
                "provider_mutation_performed": False,
                # replay_parent's scratch queue is <its own root>/launch-preparations.
                "scratch_queue_root": str(child / "launch-preparations"),
            }
        )
    )
    proc = tmp_path / "proc"
    proc.mkdir()
    return root, child, data, proc


def plan(root, proc, **options):
    return gc.plan_replay_cache_retention(
        replay_root=root, process_root=proc, now=time.time() + 120, **options
    )


# The storage GC phase's opt-in; the standalone unit never passes it.
STORE = {"reclaim_store_copies": True}


def apply(p, proc, **options):
    return gc.apply_replay_cache_retention(p, ack=gc.ACK, process_root=proc, **options)


def test_reclaims_only_disposable_binary_copies_and_is_idempotent(tmp_path):
    root, child, data, proc = setup(tmp_path)
    p = plan(root, proc)
    assert p["candidate_bytes"] == 100000
    result = gc.apply_replay_cache_retention(p, ack=gc.ACK, process_root=proc)
    assert result["removed_bytes"] == 100000
    assert not data.exists()
    for name in ["readonly.ply", "shared.ply", "run.log", "stage_replay_report.v1.json"]:
        assert (child / name).exists()
    assert plan(root, proc)["candidate_bytes"] == 0


def test_active_reader_is_retained_before_plan_and_apply(tmp_path):
    root, child, data, proc = setup(tmp_path)
    p = plan(root, proc)
    process = proc / "123"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python")
    (process / "environ").write_bytes(b"")
    (process / "fd/3").symlink_to(data)
    assert plan(root, proc)["candidate_bytes"] == 0
    result = gc.apply_replay_cache_retention(p, ack=gc.ACK, process_root=proc)
    assert result["removed_bytes"] == 0
    assert data.exists()


def test_changed_file_and_unfinished_reports_are_kept(tmp_path):
    root, child, data, proc = setup(tmp_path)
    p = plan(root, proc)
    data.write_bytes(b"changed" * 20000)
    assert gc.apply_replay_cache_retention(p, ack=gc.ACK, process_root=proc)["removed_bytes"] == 0
    assert plan(root, proc)["candidate_bytes"] == 0
    report = child / "stage_replay_report.v1.json"
    d = json.loads(report.read_text())
    d["provider_mutation_performed"] = True
    report.write_text(json.dumps(d))
    assert plan(root, proc)["candidate_bytes"] == 0


def test_source_code_is_protected_even_from_a_resealed_plan(tmp_path):
    root, child, data, proc = setup(tmp_path)
    p = plan(root, proc)
    code = child / "source.py"
    code.write_bytes(data.read_bytes())
    info = code.stat()
    p["rows"][0]["files"] = [
        {
            "relative_path": "source.py",
            "inode": info.st_ino,
            "mtime_ns": info.st_mtime_ns,
            "size_bytes": info.st_size,
            "sha256": gc.file_sha(code),
        }
    ]
    p["plan_digest"] = gc.digest({k: v for k, v in p.items() if k != "plan_digest"})
    assert gc.apply_replay_cache_retention(p, ack=gc.ACK, process_root=proc)["removed_bytes"] == 0
    assert code.exists()


def test_unknown_process_inventory_fails_closed(tmp_path):
    root, child, data, proc = setup(tmp_path)
    with pytest.raises(ValueError, match="process_inventory_unavailable"):
        plan(root, tmp_path / "absent-proc")


def test_gc_ignores_its_own_reference_but_retains_other_live_readers(tmp_path):
    root, child, data, proc = setup(tmp_path)
    for pid in (101, 202):
        process = proc / str(pid)
        (process / "fd").mkdir(parents=True)
        (process / "cmdline").write_bytes(str(child).encode())
        (process / "environ").write_bytes(b"")
    assert gc.active_reference(child, process_root=proc, ignored_process_ids=(101,))
    assert not gc.active_reference(child, process_root=proc, ignored_process_ids=(101, 202))
    assert gc.active_reference(child, process_root=proc)


def _refuse_reading(monkeypatch, denied, error):
    """Reading ``denied`` fails with ``error``, as /proc does for a process this user may not inspect."""
    real_read, real_iterdir, real_readlink = Path.read_bytes, Path.iterdir, os.readlink

    def refuse(path):
        if Path(path) == denied:
            raise error

    monkeypatch.setattr(Path, "read_bytes", lambda self: refuse(self) or real_read(self))
    monkeypatch.setattr(Path, "iterdir", lambda self: refuse(self) or real_iterdir(self))
    monkeypatch.setattr(os, "readlink", lambda path, *args, **kwargs: refuse(path) or real_readlink(path, *args, **kwargs))


@pytest.mark.parametrize("entry", ["cmdline", "environ", "fd", "cwd"])
@pytest.mark.parametrize("error", [PermissionError(errno.EACCES, "Permission denied"), OSError(errno.EIO, "I/O error")])
def test_an_unreadable_process_entry_protects_without_raising(tmp_path, monkeypatch, entry, error):
    """2026-09-27: the storage GC's sweep raised PermissionError out of every check that met such an entry.

    An entry that cannot be read proves nothing about the root, so it protects it, and
    the detail says the inventory was unreadable rather than that a process holds it.
    """
    root, child, data, proc = setup(tmp_path)
    process = proc / "4242"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python")
    (process / "environ").write_bytes(b"")
    _refuse_reading(monkeypatch, process / entry, error)

    assert gc.process_reference(child, process_root=proc) == gc.PROCESS_INVENTORY_UNREADABLE
    assert gc.active_reference(child, process_root=proc) is True
    unread = plan(root, proc)
    assert unread["candidate_bytes"] == 0 and unread["rows"] == []
    assert unread["kept"] == [{"root": str(child), "reason": "active_reference"}]
    # A process seen holding the root is the stronger answer, wherever the sweep meets it.
    holder = proc / "4243"
    (holder / "fd").mkdir(parents=True)
    (holder / "cmdline").write_bytes(b"python")
    (holder / "environ").write_bytes(b"")
    (holder / "fd" / "7").symlink_to(data)
    assert gc.process_reference(child, process_root=proc) == gc.PROCESS_REFERENCED
    assert data.exists()


@pytest.mark.parametrize("error", [FileNotFoundError(errno.ENOENT, "gone"), ProcessLookupError(errno.ESRCH, "gone")])
def test_a_process_that_exited_mid_sweep_is_not_unreadable(tmp_path, monkeypatch, error):
    root, child, data, proc = setup(tmp_path)
    process = proc / "4242"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python")
    (process / "environ").write_bytes(b"")
    _refuse_reading(monkeypatch, process / "environ", error)

    assert gc.process_reference(child, process_root=proc) is None
    assert gc.active_reference(child, process_root=proc) is False


def test_an_unlistable_process_table_protects(tmp_path, monkeypatch):
    """A /proc that refuses its own listing proves nothing is unreferenced."""
    root, child, data, proc = setup(tmp_path)
    _refuse_reading(monkeypatch, proc, PermissionError(errno.EACCES, "Permission denied"))

    assert gc.process_reference(child, process_root=proc) == gc.PROCESS_INVENTORY_UNREADABLE
    assert gc.active_reference(child, process_root=proc) is True
    assert plan(root, proc)["kept"] == [{"root": str(child), "reason": "active_reference"}]
    assert data.exists()


def test_one_unreadable_descriptor_protects_as_inventory_unreadable(tmp_path, monkeypatch):
    """A single fd/N whose link cannot be read protects the root, and the evidence phase's
    reason says the inventory was unreadable, not that a process holds the run."""
    from blueprint_pipeline import control_plane_storage_gc_reasons as reasons

    root, child, data, proc = setup(tmp_path)
    process = proc / "4242"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python")
    (process / "environ").write_bytes(b"")
    (process / "fd" / "3").symlink_to(tmp_path / "elsewhere")
    (process / "fd" / "4").symlink_to(tmp_path / "unrelated")
    _refuse_reading(monkeypatch, process / "fd" / "3", PermissionError(errno.EACCES, "Permission denied"))

    assert gc.process_reference(child, process_root=proc) == gc.PROCESS_INVENTORY_UNREADABLE
    assert gc.active_reference(child, process_root=proc) is True
    assert reasons.evidence_protection_reason(
        child, settlement_roots=(), pins_root=tmp_path / "pins", queue_roots=(), now=lambda: 0.0, process_root=proc,
    ) == "protected_process_inventory_unreadable"


def _store_copy(child, payload, *, directory=("prepared-references", "content-addressed", "sha256"),
                name=None, seconds_before_report=1):
    """A parent replay's copy of a store blob: named by its digest, read-only, older than the report."""
    store = child.joinpath(*directory)
    store.mkdir(parents=True, exist_ok=True)
    path = store / (name or hashlib.sha256(payload).hexdigest())
    path.write_bytes(payload)
    stamp = (child / "stage_replay_report.v1.json").stat().st_mtime_ns - seconds_before_report * 10**9
    os.utime(path, ns=(stamp, stamp))
    path.chmod(0o444)
    return path


def planned_paths(p):
    return {f["relative_path"] for row in p["rows"] for f in row["files"]} | {
        name for row in p["rows"] for key in ("store_copies", "scratch_inputs")
        for group in row.get(key, ()) for name in group["relative_paths"]}


def reseal(p, row):
    forged = {**p, "rows": [row]}
    forged["plan_digest"] = gc.digest({k: v for k, v in forged.items() if k != "plan_digest"})
    return forged


def test_content_addressed_scratch_copies_are_reclaimable(tmp_path):
    """2026-09-27: activation lookaheads copied the whole content store into each parent replay.
    A copy has no suffix, is often under 64 KiB and keeps the store's read-only mode; its name
    matching its digest is what makes it a disposable copy."""
    root, child, data, proc = setup(tmp_path)
    copy = _store_copy(child, b"a small read-only store blob")

    p = plan(root, proc, **STORE)

    assert planned_paths(p) == {"working.ply", str(copy.relative_to(child))}
    [row] = p["rows"]
    info = copy.stat()
    assert row["store_copies"] == [{
        "relative_paths": [str(copy.relative_to(child))], "inode": info.st_ino, "dev": info.st_dev, "nlink": 1,
        "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "sha256": "sha256:" + copy.name}]
    assert p["candidate_bytes"] == 100000 + len(b"a small read-only store blob")
    result = apply(p, proc, **STORE)
    assert result["removed_bytes"] == p["candidate_bytes"]
    assert not copy.exists() and not data.exists()
    for name in ["readonly.ply", "shared.ply", "run.log", "stage_replay_report.v1.json"]:
        assert (child / name).exists()
    assert plan(root, proc, **STORE)["candidate_bytes"] == 0


def test_linked_scratch_pairs_inside_the_replay_are_reclaimed(tmp_path):
    """The worker hard-links each blob a preparation references into prepared-references/
    <preparation>/, so most copies have two names; the bytes are freed only when both go."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    copy = _store_copy(child, b"a copy the replayed preparation materialized")
    materialized = child / "prepared-references" / "preparation-1" / "construction-stage-configurations" / copy.name
    materialized.parent.mkdir(parents=True)
    os.link(copy, materialized)

    p = plan(root, proc, **STORE)

    [row] = p["rows"]
    [group] = row["store_copies"]
    assert row["files"] == []
    assert group["relative_paths"] == sorted([str(copy.relative_to(child)), str(materialized.relative_to(child))])
    assert (group["nlink"], group["sha256"]) == (2, "sha256:" + copy.name)
    assert p["candidate_bytes"] == len(b"a copy the replayed preparation materialized"), "counted once"
    result = apply(p, proc, **STORE)
    assert result["removed_bytes"] == p["candidate_bytes"]
    assert result["removed"] == [{"paths": [str(child / name) for name in group["relative_paths"]],
                                  "sha256": group["sha256"], "size_bytes": group["size_bytes"]}]
    assert not copy.exists() and not materialized.exists()
    assert (child / "stage_replay_report.v1.json").exists()


def test_an_interrupted_copy_removal_is_finished_by_the_next_plan(tmp_path, monkeypatch):
    """The store name is what makes a group a store copy, so it goes last: a removal cut
    short leaves a group the next plan still recognises, never an unclaimable orphan."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    copy = _store_copy(child, b"a copy with two materialized names")
    size = copy.stat().st_size
    for preparation in ("adp-preparation", "scene-preparation"):  # either side of content-addressed
        (child / "prepared-references" / preparation).mkdir()
        os.link(copy, child / "prepared-references" / preparation / copy.name)
    real_unlink = os.unlink
    calls: list[str] = []

    def unlink(path, *args, **kwargs):
        calls.append(path)
        if len(calls) == 3:  # the last of the three names
            raise OSError(errno.EIO, "interrupted")
        return real_unlink(path, *args, **kwargs)

    first = plan(root, proc, **STORE)
    monkeypatch.setattr(os, "unlink", unlink)
    interrupted = apply(first, proc, **STORE)
    monkeypatch.setattr(os, "unlink", real_unlink)

    assert (interrupted["removed_bytes"], [skip["reason"] for skip in interrupted["skipped"]]) == (
        0, ["unlink_failed:OSError"])
    assert copy.exists() and copy.stat().st_nlink == 1
    again = plan(root, proc, **STORE)
    [group] = again["rows"][0]["store_copies"]
    assert (group["relative_paths"], group["nlink"]) == ([str(copy.relative_to(child))], 1)
    assert apply(again, proc, **STORE)["removed_bytes"] == size
    assert not copy.exists()


def _victim(tmp_path, name):
    """A file outside the replay with the same name as one the replay holds."""
    victim_dir = tmp_path / "victim-dir"
    victim_dir.mkdir(exist_ok=True)
    (victim_dir / name).write_text("evidence that is not the replay's")
    return victim_dir, victim_dir / name


def _swap_while_hashing(monkeypatch, directory, target, *, leaf=None):
    """Swap ``directory`` for a symlink to ``target`` while apply hashes, as the review's probe did."""
    real = gc._held_sha
    swapped = []

    def racing(directory_fd, name, expected):
        if not swapped and (leaf is None or name == leaf):
            os.rename(directory, directory.with_name(directory.name + "-moved"))
            directory.symlink_to(target, target_is_directory=True)
            swapped.append(name)
        return real(directory_fd, name, expected)

    monkeypatch.setattr(gc, "_held_sha", racing)
    return swapped


def test_a_directory_swapped_while_hashing_never_redirects_a_store_copy_removal(tmp_path, monkeypatch):
    """Code review, 2026-09-27: apply checked the parents, hashed, then unlinked each name by
    path, so swapping prepared-references/prep for a symlink while it hashed made the root GC
    delete a file outside the replay. Every recheck, hash and unlink now goes through
    directory descriptors held from the replay child down, and a moved directory is a skip."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    copy = _store_copy(child, b"scratch copy bytes" * 100)
    prep = child / "prepared-references" / "prep"
    prep.mkdir()
    os.link(copy, prep / "receipt.json")
    victim_dir, victim = _victim(tmp_path, "receipt.json")
    p = plan(root, proc, **STORE)
    swapped = _swap_while_hashing(monkeypatch, prep, victim_dir)

    result = apply(p, proc, **STORE)

    assert swapped and victim.read_text() == "evidence that is not the replay's"
    assert result["removed_bytes"] == 0
    assert [skip["reason"] for skip in result["skipped"]] == ["path_changed"]
    assert copy.exists() and (prep.with_name("prep-moved") / "receipt.json").exists()


def test_a_directory_swapped_while_hashing_never_redirects_a_file_removal(tmp_path, monkeypatch):
    root, child, data, proc = setup(tmp_path)
    renders = child / "renders"
    renders.mkdir()
    frame = renders / "frame.ply"
    frame.write_bytes(b"r" * 100000)
    stamp = (child / "stage_replay_report.v1.json").stat().st_mtime_ns - 10**9
    os.utime(frame, ns=(stamp, stamp))
    victim_dir, victim = _victim(tmp_path, "frame.ply")
    p = plan(root, proc)
    assert planned_paths(p) == {"working.ply", "renders/frame.ply"}
    swapped = _swap_while_hashing(monkeypatch, renders, victim_dir, leaf="frame.ply")

    result = apply(p, proc)

    assert swapped and victim.read_text() == "evidence that is not the replay's"
    assert result["skipped"] == [{"path": str(frame), "reason": "path_changed"}]
    assert result["removed_bytes"] == 100000 and not data.exists()
    assert (renders.with_name("renders-moved") / "frame.ply").exists()


def test_an_item_that_fails_is_recorded_and_apply_keeps_going(tmp_path, monkeypatch):
    root, child, data, proc = setup(tmp_path)
    stamp = (child / "stage_replay_report.v1.json").stat().st_mtime_ns - 10**9
    for name in ("stuck.ply", "gone.ply"):
        (child / name).write_bytes(b"s" * 100000)
        os.utime(child / name, ns=(stamp, stamp))
    copy = _store_copy(child, b"a store copy")
    p = plan(root, proc, **STORE)
    (child / "gone.ply").unlink()
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        if path == "stuck.ply":
            raise PermissionError(errno.EACCES, "denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    result = apply(p, proc, **STORE)
    monkeypatch.setattr(os, "unlink", real_unlink)

    assert sorted(result["skipped"], key=lambda skip: skip["path"]) == [
        {"path": str(child / "gone.ply"), "reason": "recheck_failed:FileNotFoundError"},
        {"path": str(child / "stuck.ply"), "reason": "unlink_failed:PermissionError"},
    ]
    assert result["removed_bytes"] == 100000 + len(b"a store copy")
    assert not data.exists() and not copy.exists() and (child / "stuck.ply").exists()


def test_the_process_table_is_swept_only_for_a_root_with_something_to_reclaim(tmp_path, monkeypatch):
    """A sweep of /proc per finished replay on every tick cost the most for the replays with
    nothing left: with the opt-in, names, links and sizes come first and only a root with a
    candidate is checked for live readers; an estimate checks none. The standalone unit's
    order is unchanged."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    busy = root / "parent-busy"
    busy.mkdir()
    (busy / "stage_replay_report.v1.json").write_text((child / "stage_replay_report.v1.json").read_text())
    _store_copy(busy, b"a copy left to reclaim")
    real = gc.active_reference
    calls: list[str] = []
    monkeypatch.setattr(gc, "active_reference", lambda path, **kwargs: calls.append(path.name) or real(path, **kwargs))

    assert plan(root, proc, **STORE)["candidate_bytes"] > 0
    assert calls == ["parent-busy"]
    calls.clear()
    assert gc.estimate_replay_cache_retention(replay_root=root, now=time.time() + 120, **STORE)[
        "estimated_candidate_bytes"] > 0
    assert calls == []
    plan(root, proc)
    assert calls == ["parent-busy", "parent-finished"]


def test_scratch_inode_linked_outside_the_replay_is_kept(tmp_path):
    """A copy sharing its inode with a name outside the replay's prepared-references (the
    production store when linking worked, the scratch queue, anything else) is never planned:
    removing the names inside would free nothing, and the other name is not the replay's."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    production = _store_copy(child, b"linked with the production store")
    os.link(production, tmp_path / "production-name")
    queued = _store_copy(child, b"linked from the replay's scratch queue")
    (child / "launch-preparations").mkdir()
    os.link(queued, child / "launch-preparations" / queued.name)

    p = plan(root, proc, **STORE)

    assert (p["rows"], p["candidate_bytes"]) == ([], 0)
    assert production.exists() and queued.exists()


def test_mismatched_or_changed_scratch_copies_are_kept(tmp_path):
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    mismatched = _store_copy(child, b"bytes that are not the named digest", name="0" * 64)
    misplaced = _store_copy(child, b"a store name outside the scratch store", directory=("content-addressed", "sha256"))
    orphan = _store_copy(child, b"a materialized name with no store name", directory=("prepared-references", "prep"))
    newer = _store_copy(child, b"written after the report", seconds_before_report=-5)
    reclaimable = _store_copy(child, b"a copy with a second name")
    second = child / "prepared-references" / "prep" / reclaimable.name
    os.link(reclaimable, second)

    before = plan(root, proc, **STORE)

    assert planned_paths(before) == {str(reclaimable.relative_to(child)), str(second.relative_to(child))}
    [row] = before["rows"]
    [group] = row["store_copies"]

    # A resealed plan cannot make a file whose bytes differ from its digest name a store copy,
    info = mismatched.stat()
    forged = {"relative_paths": [str(mismatched.relative_to(child))], "inode": info.st_ino, "nlink": 1,
              "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "sha256": gc.file_sha(mismatched)}
    assert apply(reseal(before, {**row, "store_copies": [forged]}), proc, **STORE)["removed_bytes"] == 0
    # leave one of a copy's names behind,
    partial = {**group, "relative_paths": [str(reclaimable.relative_to(child))], "nlink": 1}
    assert apply(reseal(before, {**row, "store_copies": [partial]}), proc, **STORE)["removed_bytes"] == 0
    # or reach outside the replay's prepared-references.
    for names in ([group["relative_paths"][0]] * 2, ["stage_replay_report.v1.json"], ["../outside"]):
        with pytest.raises(ValueError, match="replay_cache_member_unsafe"):
            apply(reseal(before, {**row, "store_copies": [{**group, "relative_paths": names}]}), proc, **STORE)

    # A reader that appears after the plan keeps the copy at apply; the next plan keeps its root.
    process = proc / "123"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python")
    (process / "environ").write_bytes(b"")
    (process / "fd" / "3").symlink_to(reclaimable)
    result = apply(before, proc, **STORE)
    assert result["removed_bytes"] == 0
    assert result["skipped"] == [{"root": str(child), "reason": "active_reference"}]
    assert plan(root, proc, **STORE)["kept"] == [{"root": str(child), "reason": "active_reference"}]

    # Linked elsewhere after the plan: apply rechecks the link count and keeps every name.
    (process / "fd" / "3").unlink()
    os.link(reclaimable, tmp_path / "late-link")
    result = apply(before, proc, **STORE)
    assert result["removed_bytes"] == 0
    assert result["skipped"] == [{"paths": [str(child / name) for name in group["relative_paths"]],
                                  "reason": "copy_changed"}]
    for path in (mismatched, misplaced, orphan, newer, reclaimable, second):
        assert path.exists(), path


def test_an_estimate_reads_no_bytes_and_bounds_the_plan(tmp_path, monkeypatch):
    root, child, data, proc = setup(tmp_path)
    copy = _store_copy(child, b"a faithful copy")
    os.link(copy, child / "prepared-references" / copy.name)
    mismatched = _store_copy(child, b"bytes that are not the named digest", name="0" * 64)
    real_file_sha = gc.file_sha
    monkeypatch.setattr(gc, "file_sha", lambda path: pytest.fail(f"an estimate read {path}"))

    estimate = gc.estimate_replay_cache_retention(replay_root=root, now=time.time() + 120, **STORE)

    monkeypatch.setattr(gc, "file_sha", real_file_sha)
    p = plan(root, proc, **STORE)
    assert "rows" not in estimate and estimate["digests_verified"] is False
    # Only a plan reads the bytes, so only a plan can tell that a copy is not what its name says.
    assert p["candidate_bytes"] == data.stat().st_size + copy.stat().st_size
    assert estimate["estimated_candidate_bytes"] == p["candidate_bytes"] + mismatched.stat().st_size


def test_any_finished_parent_replay_counts_under_the_store_copy_opt_in(tmp_path):
    """A parent replay writes its report when it returns, whatever its status, and its fetcher
    refuses every fetch by construction, so nothing_fetched adds no safety there. Requiring it
    kept the copies of every refused, blocked or rowless lookahead forever. Without the opt-in
    the rule is unchanged: only nothing_fetched counts."""
    root, child, data, proc = setup(tmp_path)
    report = child / "stage_replay_report.v1.json"
    clean = {"paid_execution_requested": False, "provider_mutation_performed": False}

    def write(**fields):
        report.write_text(json.dumps({"schema_version": "task_evaluation_parent_replay_report.v1", **fields}))

    for fields in ({"status": "worker_refused"}, {"status": "blocked", "nothing_fetched": False}, {"status": "no_row"}):
        write(**fields, **clean)
        assert gc.completed_report(child, any_parent_status=True) == report, fields
        assert gc.completed_report(child) is None, fields
    assert plan(root, proc, **STORE)["candidate_bytes"] == data.stat().st_size
    assert plan(root, proc)["candidate_bytes"] == 0

    for fields in (
        {"status": "blocked", "paid_execution_requested": True, "provider_mutation_performed": False},
        {"status": "blocked", "paid_execution_requested": False, "provider_mutation_performed": True},
        {"status": "blocked", "provider_mutation_performed": False},
    ):
        write(**fields)
        assert gc.completed_report(child, any_parent_status=True) is None, fields
    write(nothing_fetched=True, **clean)
    assert gc.completed_report(child) == gc.completed_report(child, any_parent_status=True) == report


def test_without_the_opt_in_the_rules_are_the_ones_the_unit_always_had(tmp_path):
    """blueprint-completed-replay-cache-gc runs main() as root every five minutes with --apply
    over stage-replays. The store-copy rules reach it only through their own opt-in, so without
    it the plan is exactly the one it made before them: no store copies, no refused or blocked
    parent replays, and rows with the same keys in the same order."""
    root, child, data, proc = setup(tmp_path)
    copy = _store_copy(child, b"a store copy the unit leaves alone")
    os.link(copy, child / "prepared-references" / copy.name)
    for name, status in (("parent-refused", {"status": "worker_refused"}),
                         ("parent-blocked", {"status": "blocked", "nothing_fetched": False})):
        other = root / name
        other.mkdir()
        (other / "working.ply").write_bytes(b"w" * 100000)
        (other / "stage_replay_report.v1.json").write_text(json.dumps({
            "schema_version": "task_evaluation_parent_replay_report.v1",
            "paid_execution_requested": False, "provider_mutation_performed": False, **status}))
    report = child / "stage_replay_report.v1.json"
    info = data.stat()

    p = plan(root, proc)

    assert list(p) == ["schema_version", "status", "observed_at_epoch", "replay_root", "rows", "kept",
                       "candidate_bytes", "reports_and_original_evidence_removed", "plan_digest"]
    assert [list(row) for row in p["rows"]] == [["root", "report_path", "report_sha256", "files"]]
    assert p["rows"] == [{
        "root": str(child), "report_path": str(report), "report_sha256": gc.file_sha(report),
        "files": [{"relative_path": "working.ply", "inode": info.st_ino, "mtime_ns": info.st_mtime_ns,
                   "size_bytes": 100000, "sha256": gc.file_sha(data)}]}]
    # A plan made with the opt-in cannot be applied without it.
    with pytest.raises(ValueError, match="replay_cache_store_copies_not_admitted"):
        apply(plan(root, proc, **STORE), proc)
    assert [row["path"] for row in apply(p, proc)["removed"]] == [str(data)]
    assert copy.exists() and (root / "parent-refused" / "working.ply").exists()
    assert (root / "parent-blocked" / "working.ply").exists()


def test_single_files_can_be_left_to_their_own_rules(tmp_path):
    """The storage GC phase removes only store copies and scratch inputs: its plans carry no single
    files, and a plan that does is refused when applied without them."""
    root, child, data, proc = setup(tmp_path)
    copy = _store_copy(child, b"a store copy")

    p = plan(root, proc, single_files=False, **STORE)

    assert planned_paths(p) == {str(copy.relative_to(child))}
    with pytest.raises(ValueError, match="replay_cache_single_files_not_admitted"):
        apply(plan(root, proc, **STORE), proc, single_files=False, **STORE)
    assert apply(p, proc, single_files=False, **STORE)["removed_bytes"] == len(b"a store copy")
    assert data.exists() and not copy.exists()


def test_the_standalone_command_takes_store_copies_only_when_asked(tmp_path, monkeypatch):
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(gc, "plan_replay_cache_retention",
                        lambda **options: seen.append(("plan", options)) or {"plan_digest": "sha256:" + "0" * 64})
    monkeypatch.setattr(gc, "apply_replay_cache_retention",
                        lambda plan, **options: seen.append(("apply", options)) or {"status": "applied"})
    for extra in ([], ["--reclaim-store-copies"]):
        monkeypatch.setattr("sys.argv", ["completed_replay_cache_retention", "--replay-root", str(tmp_path),
                                         "--report-root", str(tmp_path / "reports"), "--apply", "--ack", gc.ACK,
                                         *extra])
        gc.main()

    assert [(step, options["reclaim_store_copies"]) for step, options in seen] == [
        ("plan", False), ("apply", False), ("plan", True), ("apply", True)]
    unit = (Path(__file__).resolve().parents[1] / "deploy/systemd/blueprint-completed-replay-cache-gc.service").read_text(
        encoding="utf-8")
    assert "--apply" in unit and "--reclaim-store-copies" not in unit, "the store-copy rules stay out of the unit"


class _Stalled(Exception):
    """Not an OSError, so apply cannot record it as a recheck failure and carry on."""


@contextlib.contextmanager
def _fails_instead_of_blocking(seconds):
    """Turn an apply that blocks into a failure after ``seconds``, instead of a hung test."""

    def alarm(_signum, _frame):
        raise _Stalled(f"apply blocked for {seconds}s")

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_a_fifo_swapped_in_after_the_recheck_cannot_stall_apply(tmp_path, monkeypatch):
    """Code review of PR 10a: a leaf was opened to be hashed without O_NONBLOCK, so a FIFO put in
    a planned file's place after its S_ISREG recheck held the tick in open() until a writer came.
    The open no longer waits, and the identity check refuses what it opened."""
    root, child, data, proc = setup(tmp_path)
    p = plan(root, proc)
    moved = data.with_name("working-moved.ply")
    real = gc._held_sha

    def racing(directory, name, expected):
        os.rename(data, moved)
        os.mkfifo(data)
        return real(directory, name, expected)

    monkeypatch.setattr(gc, "_held_sha", racing)
    with _fails_instead_of_blocking(5):
        result = apply(p, proc)

    assert (result["removed_bytes"], result["skipped"]) == (0, [{"path": str(data), "reason": "file_changed"}])
    assert stat.S_ISFIFO(data.lstat().st_mode) and moved.read_bytes() == b"x" * 100000


@contextlib.contextmanager
def _descriptor_limit(free):
    """Lower this process's RLIMIT_NOFILE so that only ``free`` more descriptors can be opened."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    in_use = {int(name) for name in os.listdir("/dev/fd")}
    limit = 0
    while limit - sum(1 for fd in in_use if fd < limit) < free:
        limit += 1
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_a_row_across_many_directories_stays_within_the_descriptor_limit(tmp_path):
    """Code review of PR 10a: apply held every directory a row's names were in until the row
    ended, so one wide row could run out of descriptors (EMFILE) and skip everything after.
    Each item's directories are closed once it is done: with room for only 16 more descriptors,
    a row across 65 directories is removed whole."""
    root, child, data, proc = setup(tmp_path)
    stamp = (child / "stage_replay_report.v1.json").stat().st_mtime_ns - 10**9
    frames = []
    for index in range(64):
        frame = child / "renders" / f"view-{index:02d}" / "frame.ply"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b"r" * 65536)
        os.utime(frame, ns=(stamp, stamp))
        frames.append(frame)
    p = plan(root, proc)
    assert len(p["rows"][0]["files"]) == 65

    with _descriptor_limit(free=16):
        result = apply(p, proc)

    assert (result["skipped"], result["removed_bytes"]) == ([], 100000 + 64 * 65536)
    assert not data.exists() and not any(frame.exists() for frame in frames)


def test_a_child_opened_through_a_swapped_ancestor_is_refused(tmp_path, monkeypatch):
    """Code review of PR 10a: apply checked the replay root's path for links and then opened it
    by path, so an ancestor swapped for a link in between (and swapped back) re-rooted every
    descriptor it held. The child it holds must still be the one the checked path names."""
    outer = tmp_path / "outer"
    root, child, data, _proc = setup(outer)
    proc = tmp_path / "proc"
    proc.mkdir()
    p = plan(root, proc)
    elsewhere = tmp_path / "elsewhere"
    decoy = elsewhere / "replays" / child.name / data.name
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(data.read_bytes())
    real = gc._HeldChild

    class Rerooted(real):
        def __init__(self, base, name):
            outer.rename(tmp_path / "outer-real")
            outer.symlink_to(elsewhere, target_is_directory=True)
            try:
                super().__init__(base, name)
            finally:
                outer.unlink()
                (tmp_path / "outer-real").rename(outer)

    monkeypatch.setattr(gc, "_HeldChild", Rerooted)
    with pytest.raises(ValueError, match="^replay_cache_root_changed$"):
        apply(p, proc)

    assert data.exists() and decoy.exists()


# The storage GC phase's opt-ins since the lookahead backlog finding; the standalone unit passes neither.
SCRATCH = {"reclaim_store_copies": True, "reclaim_scratch_inputs": True}


def _scratch_input(child, relative, payload, *, seconds_before_report=1):
    """A file a parent replay left in its scratch inputs: any name, any bytes, older than its report."""
    path = child / "prepared-references" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    report = child / "stage_replay_report.v1.json"
    stamp = (report.stat().st_mtime_ns if report.exists() else time.time_ns()) - seconds_before_report * 10**9
    os.utime(path, ns=(stamp, stamp))
    return path


def test_no_double_counting_when_store_and_scratch_rules_both_apply(tmp_path):
    """2026-09-27, 18:30 UTC: the store-copy rule estimated 4.8 GB across 105 lookaheads that held
    12.95 GiB, because most of a lookahead's prepared-references has no digest-matching store
    name. A parent replay made that whole tree in its own temporary root, so under its own opt-in
    all of it is scratch. Every store copy is a scratch input too: with both opt-ins the scratch
    rule plans each inode once, with every name the worker linked to it, and no store_copies entry
    repeats its bytes. An estimate, a plan and an apply agree."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    copy = _store_copy(child, b"a store copy the worker materialized")
    materialized = child / "prepared-references" / "preparation-1" / copy.name
    materialized.parent.mkdir(parents=True)
    os.link(copy, materialized)
    derived = _scratch_input(child, "preparation-1/derived/view.json", b"a copy with no store name")

    store_only = plan(root, proc, **STORE)
    both = plan(root, proc, **SCRATCH)

    assert store_only["candidate_bytes"] == copy.stat().st_size
    [row] = both["rows"]
    info = copy.stat()
    assert row["store_copies"] == []
    assert row["scratch_inputs"][0] == {
        "relative_paths": sorted([str(copy.relative_to(child)), str(materialized.relative_to(child))]),
        "inode": info.st_ino, "dev": info.st_dev, "nlink": 2, "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns}
    assert [group["relative_paths"] for group in row["scratch_inputs"][1:]] == [[str(derived.relative_to(child))]]
    size = info.st_size + derived.stat().st_size
    assert both["candidate_bytes"] == size
    assert gc.estimate_replay_cache_retention(
        replay_root=root, now=time.time() + 120, **SCRATCH)["estimated_candidate_bytes"] == size
    result = apply(both, proc, **SCRATCH)
    assert (result["removed_bytes"], result["skipped"]) == (size, [])
    assert sorted(path for entry in result["removed"] for path in entry["paths"]) == sorted(
        str(path) for path in (copy, materialized, derived))
    assert (child / "stage_replay_report.v1.json").exists()


def test_scratch_input_linked_outside_is_kept(tmp_path):
    """A scratch input sharing its inode with a name outside the replay's prepared-references (the
    production store when linking worked, the replay's scratch queue, anything else) is kept:
    removing the names inside would free nothing, and the other name is not the replay's. So is
    one written after the report. A link made after the plan is found by apply's recheck, and a
    plan cannot name anything outside the scratch inputs."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    production = _scratch_input(child, "preparation-1/linked-with-production.bin", b"production bytes")
    os.link(production, tmp_path / "production-name")
    queued = _scratch_input(child, "preparation-1/queued.json", b"queued bytes")
    (child / "launch-preparations").mkdir()
    os.link(queued, child / "launch-preparations" / "queued.json")
    newer = _scratch_input(child, "preparation-1/newer.bin", b"written after the report", seconds_before_report=-5)
    gone = _scratch_input(child, "preparation-2/gone.bin", b"reclaimable")
    late = _scratch_input(child, "preparation-2/late.bin", b"linked outside after the plan")

    p = plan(root, proc, **SCRATCH)

    assert planned_paths(p) == {str(gone.relative_to(child)), str(late.relative_to(child))}
    [row] = p["rows"]
    group = row["scratch_inputs"][0]
    for names in ([group["relative_paths"][0]] * 2, ["launch-preparations/queued.json"],
                  ["stage_replay_report.v1.json"], ["../outside"]):
        with pytest.raises(ValueError, match="replay_cache_member_unsafe"):
            apply(reseal(p, {**row, "scratch_inputs": [{**group, "relative_paths": names}]}), proc, **SCRATCH)
    os.link(late, tmp_path / "late-link")
    result = apply(p, proc, **SCRATCH)

    assert result["skipped"] == [{"paths": [str(late)], "reason": "scratch_input_changed"}]
    assert result["removed_bytes"] == len(b"reclaimable") and not gone.exists()
    for path in (production, queued, newer, late):
        assert path.exists(), path


def test_unfinished_or_referenced_lookahead_keeps_its_scratch(tmp_path):
    """Only what a finished parent replay left is scratch: its report says no paid execution was
    requested and no provider was mutated, it closed at least the window ago, and nothing live
    reads it. A legacy replay's report is finished but not a parent replay's, so there only the
    digest-verified store copies can go."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    finished = _scratch_input(child, "prep/input.bin", b"finished scratch")
    parent = json.loads((child / "stage_replay_report.v1.json").read_text())
    legacy = {"status": "repair_inputs_and_semantic_request_replay_passed", "model_inference_performed": False,
              "network_fetch_performed": False, "provider_mutation_performed": False}
    kept = []
    for name, report in (
        ("parent-no-report", None),
        ("parent-paid", {**parent, "paid_execution_requested": True}),
        ("parent-mutated", {**parent, "provider_mutation_performed": True}),
        ("parent-unrecorded", {k: v for k, v in parent.items() if k != "paid_execution_requested"}),
        ("parent-recent", parent),
        ("parent-referenced", parent),
        ("replay-legacy", legacy),
    ):
        other = root / name
        other.mkdir()
        if report is not None:
            own = {"scratch_queue_root": str(other / "launch-preparations")}
            (other / "stage_replay_report.v1.json").write_text(json.dumps({**report, **own}))
        if name == "parent-recent":
            soon = time.time() + 100  # plan() looks from 120 s ahead: closed only 20 s ago
            os.utime(other / "stage_replay_report.v1.json", (soon, soon))
        kept.append(_scratch_input(other, "prep/input.bin", b"scratch the rule leaves"))
    legacy_copy = _store_copy(root / "replay-legacy", b"a digest-named copy in a legacy replay")
    process = proc / "123"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python")
    (process / "environ").write_bytes(b"")
    (process / "fd" / "3").symlink_to(root / "parent-referenced" / "prepared-references" / "prep" / "input.bin")

    size = finished.stat().st_size + legacy_copy.stat().st_size

    p = plan(root, proc, **SCRATCH)

    assert p["kept"] == [{"root": str(root / "parent-referenced"), "reason": "active_reference"}]
    assert [(row["root"], planned_paths({"rows": [row]})) for row in p["rows"]] == [
        (str(child), {str(finished.relative_to(child))}),
        (str(root / "replay-legacy"), {str(legacy_copy.relative_to(root / "replay-legacy"))}),
    ]
    # A resealed plan cannot make a legacy replay's files scratch either.
    legacy_input, info = kept[-1], kept[-1].stat()
    forged = {"relative_paths": [str(legacy_input.relative_to(root / "replay-legacy"))], "inode": info.st_ino,
              "dev": info.st_dev, "nlink": 1, "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    assert apply(reseal(p, {**p["rows"][1], "scratch_inputs": [forged]}), proc, **SCRATCH)["skipped"] == [
        {"root": str(root / "replay-legacy"), "reason": "report_not_parent_replay"}]
    result = apply(p, proc, **SCRATCH)
    assert (result["removed_bytes"], result["skipped"]) == (size, [])
    assert not finished.exists() and not legacy_copy.exists()
    for path in kept:
        assert path.exists(), path


def test_scratch_inputs_rule_is_opt_in_and_unit_unchanged(tmp_path, monkeypatch, capsys):
    """Only storage GC passes reclaim_scratch_inputs. Without it a plan leaves a scratch input that
    has no store name, a plan made with it cannot be applied without it, and the standalone
    command, which has no flag for it, writes the very plan, result and summary line it wrote
    before the rule existed."""
    root, child, data, proc = setup(tmp_path)
    report = child / "stage_replay_report.v1.json"
    hour_ago = time.time() - 3600
    os.utime(data, (hour_ago - 60, hour_ago - 60))
    os.utime(report, (hour_ago, hour_ago))
    copy = _store_copy(child, b"a store copy the unit leaves alone")
    scratch = _scratch_input(child, "prep/derived.json", b"scratch with no store name")

    assert planned_paths(plan(root, proc, **STORE)) == {"working.ply", str(copy.relative_to(child))}
    assert "scratch_inputs" not in plan(root, proc, **STORE)["rows"][0]
    with pytest.raises(ValueError, match="replay_cache_scratch_inputs_not_admitted"):
        apply(plan(root, proc, **SCRATCH), proc, **STORE)

    # blueprint-completed-replay-cache-gc's own command line, on a host whose /proc shows no reader.
    monkeypatch.setattr(gc, "active_reference", lambda _root, **_kwargs: False)
    reports = tmp_path / "reports"
    monkeypatch.setattr("sys.argv", ["completed_replay_cache_retention", "--replay-root", str(root),
                                     "--report-root", str(reports), "--apply", "--ack", gc.ACK])
    info, data_sha = data.stat(), gc.file_sha(data)
    gc.main()

    [plan_file] = reports.glob("*-plan.json")
    expected = {
        "schema_version": gc.SCHEMA, "status": "dry_run",
        "observed_at_epoch": json.loads(plan_file.read_text())["observed_at_epoch"], "replay_root": str(root),
        "rows": [{"root": str(child), "report_path": str(report), "report_sha256": gc.file_sha(report),
                  "files": [{"relative_path": "working.ply", "inode": info.st_ino, "mtime_ns": info.st_mtime_ns,
                             "size_bytes": 100000, "sha256": data_sha}]}],
        "kept": [], "candidate_bytes": 100000, "reports_and_original_evidence_removed": False}
    expected["plan_digest"] = gc.digest(expected)
    applied = {"schema_version": gc.SCHEMA, "status": "applied", "plan_digest": expected["plan_digest"],
               "removed_bytes": 100000, "removed": [{"path": str(data), "sha256": data_sha, "size_bytes": 100000}],
               "skipped": [], "reports_and_original_evidence_removed": False}
    assert plan_file.name == expected["plan_digest"][7:] + "-plan.json"
    assert plan_file.read_text() == json.dumps(expected, indent=2) + "\n"
    assert (reports / (expected["plan_digest"][7:] + "-result.json")).read_text() == json.dumps(applied, indent=2) + "\n"
    assert capsys.readouterr().out == json.dumps(
        {"status": "applied", "removed_bytes": 100000, "reports_and_original_evidence_removed": False}) + "\n"
    assert copy.exists() and scratch.exists() and not data.exists()
    monkeypatch.setattr("sys.argv", ["completed_replay_cache_retention", "--replay-root", str(root),
                                     "--report-root", str(reports), "--reclaim-scratch-inputs"])
    with pytest.raises(SystemExit):
        gc.main()


def test_directories_the_scratch_inputs_leave_empty_are_removed_and_nothing_else(tmp_path, monkeypatch):
    """Once a row's scratch inputs are gone, the directories left empty inside prepared-references
    go too, deepest first, each opened O_NOFOLLOW from its parent's descriptor and removed
    relative to it. prepared-references itself stays, as do a directory still holding a kept
    file, a link to a directory (never entered) and every directory outside prepared-references.
    A directory still holding something is simply kept; any other failure is a typed skip.
    The no-digest walk never plans a file a link reaches, directory or file: what is outside
    stays, however old and whatever its name."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    inputs = child / "prepared-references"
    deep = _scratch_input(child, "prep/a/b/c/input.bin", b"deep")
    newer = _scratch_input(child, "prep/kept/newer.bin", b"written after the report", seconds_before_report=-5)
    (inputs / "prep" / "empty" / "nested").mkdir(parents=True)
    (inputs / "prep" / "stuck").mkdir()
    outside = tmp_path / "outside"
    (outside / "empty").mkdir(parents=True)
    behind = outside / "behind-the-link.bin"
    outside_file = tmp_path / "outside-file.bin"
    for path, payload in ((behind, b"behind a linked directory"), (outside_file, b"behind a linked file")):
        path.write_bytes(payload)
        stamp = (child / "stage_replay_report.v1.json").stat().st_mtime_ns - 10**9
        os.utime(path, ns=(stamp, stamp))
    (inputs / "prep" / "link").symlink_to(outside, target_is_directory=True)
    (inputs / "prep" / "file-link.bin").symlink_to(outside_file)
    (child / "launch-preparations" / "pending").mkdir(parents=True)
    p = plan(root, proc, **SCRATCH)
    assert planned_paths(p) == {str(deep.relative_to(child))}
    real_rmdir = os.rmdir

    def rmdir(path, *args, **kwargs):
        if path == "stuck":
            raise PermissionError(errno.EACCES, "denied")
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "rmdir", rmdir)
    result = apply(p, proc, **SCRATCH)
    monkeypatch.setattr(os, "rmdir", real_rmdir)

    assert result["removed_bytes"] == len(b"deep") and not deep.exists()
    assert result["skipped"] == [{"path": str(inputs / "prep" / "stuck"), "reason": "rmdir_failed:PermissionError"}]
    assert not (inputs / "prep" / "a").exists() and not (inputs / "prep" / "empty").exists()
    assert newer.exists() and (inputs / "prep" / "stuck").is_dir() and (inputs / "prep" / "link").is_symlink()
    assert inputs.is_dir() and (outside / "empty").is_dir() and (child / "launch-preparations" / "pending").is_dir()
    assert behind.read_bytes() == b"behind a linked directory" and outside_file.read_bytes() == b"behind a linked file"
    assert (inputs / "prep" / "file-link.bin").is_symlink()


class _OnDevice:
    """A stat result as the kernel reports one past a mount point: the same entry on another st_dev."""

    def __init__(self, info, device):
        self._info, self.st_dev = info, device

    def __getattr__(self, name):
        return getattr(self._info, name)


def test_a_group_off_the_replay_childs_filesystem_is_refused_at_apply(tmp_path, monkeypatch):
    """Review of PR 10a.1: a plan's dev was compared only with the device its own names showed at
    apply, so a group on another filesystem (one mounted under prepared-references) whose plan
    said so would have been unlinked, with no digest to stop it. Every group, and every name in
    it, must now be on the held child's own device."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    target = _scratch_input(child, "prep/on-another-device.bin", b"bytes on a mounted filesystem")
    p = plan(root, proc, **SCRATCH)
    [row] = p["rows"]
    [group] = row["scratch_inputs"]
    elsewhere = group["dev"] + 1
    real_leaf = gc._leaf

    def leaf(directory, name):
        info = real_leaf(directory, name)
        return _OnDevice(info, elsewhere) if name == target.name else info

    monkeypatch.setattr(gc, "_leaf", leaf)
    forged = apply(reseal(p, {**row, "scratch_inputs": [{**group, "dev": elsewhere}]}), proc, **SCRATCH)
    planned = apply(p, proc, **SCRATCH)

    for result in (forged, planned):
        assert result["skipped"] == [{"paths": [str(target)], "reason": "cross_device"}]
        assert result["removed_bytes"] == 0
    assert target.read_bytes() == b"bytes on a mounted filesystem"


def test_a_directory_on_another_device_is_neither_planned_nor_pruned(tmp_path, monkeypatch):
    """Review of PR 10a.1: the scratch walk and the prune crossed mount points, so a filesystem
    mounted under a finished lookahead's prepared-references would have had its files planned
    and unlinked (no digest) and its empty directories removed. Neither leaves the replay child's
    device now; a directory whose st_dev differs is not entered."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    ours = _scratch_input(child, "prep/ours.bin", b"on the replay's own filesystem")
    mounted = child / "prepared-references" / "prep" / "mounted"
    theirs = _scratch_input(child, "prep/mounted/theirs.bin", b"on another filesystem")
    (mounted / "empty").mkdir()
    elsewhere = os.lstat(child).st_dev + 1
    real_lstat, real_leaf = os.lstat, gc._leaf

    def lstat(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        inside = str(path) == str(mounted) or str(path).startswith(str(mounted) + os.sep)
        return _OnDevice(info, elsewhere) if inside else info

    def leaf(directory, name):
        info = real_leaf(directory, name)
        return _OnDevice(info, elsewhere) if name == mounted.name else info

    monkeypatch.setattr(os, "lstat", lstat)
    monkeypatch.setattr(gc, "_leaf", leaf)
    p = plan(root, proc, **SCRATCH)
    estimate = gc.estimate_replay_cache_retention(replay_root=root, now=time.time() + 120, **SCRATCH)
    result = apply(p, proc, **SCRATCH)

    assert planned_paths(p) == {str(ours.relative_to(child))}
    assert p["candidate_bytes"] == estimate["estimated_candidate_bytes"] == len(b"on the replay's own filesystem")
    assert (result["removed_bytes"], result["skipped"]) == (p["candidate_bytes"], [])
    assert theirs.exists() and (mounted / "empty").is_dir() and not ours.exists()


def test_prune_failures_name_the_call_that_failed(tmp_path, monkeypatch):
    """Review of PR 10a.1: every prune failure read rmdir_failed, whatever had failed. A directory
    that cannot be looked at, opened or listed is prune_failed:<type>, rmdir_failed:<type> is
    rmdir's own failure, and a prepared-references already gone is nothing to do."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    prep = child / "prepared-references" / "prep"
    _scratch_input(child, "prep/input.bin", b"scratch")
    for name in ("stuck", "unlisted", "unopened", "unstated"):
        (prep / name / "empty").mkdir(parents=True)
    unlisted = (prep / "unlisted").stat().st_ino
    real_listdir, real_open, real_leaf, real_rmdir = os.listdir, os.open, gc._leaf, os.rmdir
    denied = PermissionError(errno.EACCES, "denied")

    def listdir(path=".", *args, **kwargs):
        if isinstance(path, int) and os.fstat(path).st_ino == unlisted:
            raise denied
        return real_listdir(path, *args, **kwargs)

    def open_(path, *args, **kwargs):
        if path == "unopened":
            raise denied
        return real_open(path, *args, **kwargs)

    def leaf(directory, name):
        if name == "unstated":
            raise denied
        return real_leaf(directory, name)

    def rmdir(path, *args, **kwargs):
        if path == "stuck":
            raise denied
        return real_rmdir(path, *args, **kwargs)

    p = plan(root, proc, **SCRATCH)
    for name, fake in (("listdir", listdir), ("open", open_), ("rmdir", rmdir)):
        monkeypatch.setattr(os, name, fake)
    monkeypatch.setattr(gc, "_leaf", leaf)
    result = apply(p, proc, **SCRATCH)
    monkeypatch.undo()

    assert result["removed_bytes"] == len(b"scratch")
    assert result["skipped"] == [
        {"path": str(prep / "stuck"), "reason": "rmdir_failed:PermissionError"},
        {"path": str(prep / "unlisted"), "reason": "prune_failed:PermissionError"},
        {"path": str(prep / "unopened"), "reason": "prune_failed:PermissionError"},
        {"path": str(prep / "unstated"), "reason": "prune_failed:PermissionError"},
    ]
    assert not (prep / "stuck" / "empty").exists() and (prep / "unlisted" / "empty").is_dir()

    # Gone before apply: its group is a recheck failure, and the prune has nothing to do.
    again = _scratch_input(child, "prep/again.bin", b"again")
    p = plan(root, proc, **SCRATCH)
    shutil.rmtree(child / "prepared-references")
    assert apply(p, proc, **SCRATCH)["skipped"] == [
        {"paths": [str(again)], "reason": "recheck_failed:FileNotFoundError"}]


def test_a_report_makes_scratch_only_of_the_root_its_replay_ran_in(tmp_path):
    """Review of PR 10a.1: the scratch-input rule took a parent report's word that its directory was
    a replay's own. replay_parent has recorded its scratch queue as <its root>/launch-preparations
    since 2026-09-05, so the rule now also requires that of the report: one with no scratch queue,
    or one naming another root's (a report copied from elsewhere), makes nothing there scratch,
    however a plan was sealed. The store-copy rule does not rest on it, and still takes the
    digest-verified copies."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    report = child / "stage_replay_report.v1.json"
    value = json.loads(report.read_text())
    copy = _store_copy(child, b"a digest-named copy")
    scratch = _scratch_input(child, "prep/derived.json", b"scratch with no store name")
    bound = plan(root, proc, **SCRATCH)
    assert planned_paths(bound) == {str(copy.relative_to(child)), str(scratch.relative_to(child))}

    for queue in (None, str(root / "parent-elsewhere" / "launch-preparations"), 42):
        written = {k: v for k, v in value.items() if k != "scratch_queue_root"}
        report.write_text(json.dumps(written if queue is None else {**written, "scratch_queue_root": queue}))
        p = plan(root, proc, **SCRATCH)
        [row] = p["rows"]
        assert (planned_paths(p), row["scratch_inputs"]) == ({str(copy.relative_to(child))}, []), queue
        forged = reseal(p, {**row, "store_copies": [], "scratch_inputs": bound["rows"][0]["scratch_inputs"]})
        assert apply(forged, proc, **SCRATCH)["skipped"] == [
            {"root": str(child), "reason": "report_not_parent_replay"}], queue
    size = copy.stat().st_size
    result = apply(p, proc, **SCRATCH)
    assert (result["removed_bytes"], result["skipped"]) == (size, [])
    assert not copy.exists() and scratch.exists()


@pytest.mark.parametrize("when", ["before_apply", "while_rechecking"])
def test_a_directory_swapped_for_a_link_never_redirects_a_scratch_input_removal(tmp_path, monkeypatch, when):
    """The scratch-input path never hashes, so the swap-while-hashing probes above never reach it.
    A planned scratch input's directory swapped for a link to a directory holding a file of the
    same name, after the plan or while apply rechecks the group, removes nothing: the file
    outside survives, the planned file survives where it was moved, and the group is skipped."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    prep = child / "prepared-references" / "prep"
    planned = _scratch_input(child, "prep/receipt.json", b"a scratch input")
    victim_dir, victim = _victim(tmp_path, "receipt.json")
    p = plan(root, proc, **SCRATCH)

    def swap():
        prep.rename(prep.with_name("prep-moved"))
        prep.symlink_to(victim_dir, target_is_directory=True)

    if when == "before_apply":
        swap()
    else:
        real_leaf = gc._leaf

        def leaf(directory, name):
            if name == planned.name and not prep.is_symlink():
                swap()
            return real_leaf(directory, name)

        monkeypatch.setattr(gc, "_leaf", leaf)
    result = apply(p, proc, **SCRATCH)

    [skip] = result["skipped"]
    assert skip["paths"] == [str(planned)] and result["removed_bytes"] == 0
    assert skip["reason"].startswith("recheck_failed:") if when == "before_apply" else skip["reason"] == "path_changed"
    assert victim.read_text() == "evidence that is not the replay's"
    assert (prep.with_name("prep-moved") / "receipt.json").read_bytes() == b"a scratch input"


@pytest.mark.parametrize("reported_by", ["lstat_and_fstat", "fstat_only"])
def test_a_single_file_on_another_device_is_skipped_as_cross_device(tmp_path, monkeypatch, reported_by):
    """Re-review of PR 10a.1: _remove_file never compared devices, so under the standalone unit's
    default rules a planned binary whose leaf lstat and opened fstat both reported another device
    was removed, and _HeldChild's docstring (nothing on another device is unlinked) was untrue.
    The recheck now requires the held child's device of both the leaf and what is opened to hash
    it: a cross_device skip, and the file stays."""
    root, child, data, proc = setup(tmp_path)
    p = plan(root, proc)
    assert planned_paths(p) == {"working.ply"}
    target, elsewhere = data.lstat().st_ino, data.lstat().st_dev + 1
    real_leaf, real_fstat = gc._leaf, os.fstat

    def leaf(directory, name):
        info = real_leaf(directory, name)
        return _OnDevice(info, elsewhere) if name == data.name else info

    def fstat(fd):
        info = real_fstat(fd)
        return _OnDevice(info, elsewhere) if info.st_ino == target else info

    with monkeypatch.context() as patched:
        if reported_by == "lstat_and_fstat":
            patched.setattr(gc, "_leaf", leaf)
        patched.setattr(os, "fstat", fstat)
        result = apply(p, proc)

    assert (result["removed_bytes"], result["skipped"]) == (0, [{"path": str(data), "reason": "cross_device"}])
    assert data.read_bytes() == b"x" * 100000


def test_a_store_copy_opened_on_another_device_is_skipped_as_cross_device(tmp_path, monkeypatch):
    """The store-copy recheck hashes through the same open as a single file: what it opens must be
    on the held child's device too, and is otherwise a cross_device skip, never an error out of
    apply."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    copy = _store_copy(child, b"a store copy")
    p = plan(root, proc, **STORE)
    target, elsewhere = copy.lstat().st_ino, copy.lstat().st_dev + 1
    real_fstat = os.fstat

    def fstat(fd):
        info = real_fstat(fd)
        return _OnDevice(info, elsewhere) if info.st_ino == target else info

    with monkeypatch.context() as patched:
        patched.setattr(os, "fstat", fstat)
        result = apply(p, proc, **STORE)

    assert (result["removed_bytes"], result["skipped"]) == (0, [{"paths": [str(copy)], "reason": "cross_device"}])
    assert copy.exists()
