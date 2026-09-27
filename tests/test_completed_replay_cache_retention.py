from __future__ import annotations

import errno
import hashlib
import json
import os
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
            }
        )
    )
    proc = tmp_path / "proc"
    proc.mkdir()
    return root, child, data, proc


def plan(root, proc):
    return gc.plan_replay_cache_retention(
        replay_root=root, process_root=proc, now=time.time() + 120
    )


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
        name for row in p["rows"] for copy in row["store_copies"] for name in copy["relative_paths"]}


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

    p = plan(root, proc)

    assert planned_paths(p) == {"working.ply", str(copy.relative_to(child))}
    [row] = p["rows"]
    info = copy.stat()
    assert row["store_copies"] == [{
        "relative_paths": [str(copy.relative_to(child))], "inode": info.st_ino, "nlink": 1,
        "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "sha256": "sha256:" + copy.name}]
    assert p["candidate_bytes"] == 100000 + len(b"a small read-only store blob")
    result = gc.apply_replay_cache_retention(p, ack=gc.ACK, process_root=proc)
    assert result["removed_bytes"] == p["candidate_bytes"]
    assert not copy.exists() and not data.exists()
    for name in ["readonly.ply", "shared.ply", "run.log", "stage_replay_report.v1.json"]:
        assert (child / name).exists()
    assert plan(root, proc)["candidate_bytes"] == 0


def test_linked_scratch_pairs_inside_the_replay_are_reclaimed(tmp_path):
    """The worker hard-links each blob a preparation references into prepared-references/
    <preparation>/, so most copies have two names; the bytes are freed only when both go."""
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    copy = _store_copy(child, b"a copy the replayed preparation materialized")
    materialized = child / "prepared-references" / "preparation-1" / "construction-stage-configurations" / copy.name
    materialized.parent.mkdir(parents=True)
    os.link(copy, materialized)

    p = plan(root, proc)

    [row] = p["rows"]
    [group] = row["store_copies"]
    assert row["files"] == []
    assert group["relative_paths"] == sorted([str(copy.relative_to(child)), str(materialized.relative_to(child))])
    assert (group["nlink"], group["sha256"]) == (2, "sha256:" + copy.name)
    assert p["candidate_bytes"] == len(b"a copy the replayed preparation materialized"), "counted once"
    result = gc.apply_replay_cache_retention(p, ack=gc.ACK, process_root=proc)
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
    real_unlink = Path.unlink
    calls: list[Path] = []

    def unlink(self, *args, **kwargs):
        calls.append(self)
        if len(calls) == 3:  # the last of the three names
            raise OSError(errno.EIO, "interrupted")
        return real_unlink(self, *args, **kwargs)

    first = plan(root, proc)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError):
        gc.apply_replay_cache_retention(first, ack=gc.ACK, process_root=proc)
    monkeypatch.setattr(Path, "unlink", real_unlink)

    assert copy.exists() and copy.stat().st_nlink == 1
    again = plan(root, proc)
    [group] = again["rows"][0]["store_copies"]
    assert (group["relative_paths"], group["nlink"]) == ([str(copy.relative_to(child))], 1)
    assert gc.apply_replay_cache_retention(again, ack=gc.ACK, process_root=proc)["removed_bytes"] == size
    assert not copy.exists()


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

    p = plan(root, proc)

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

    before = plan(root, proc)

    assert planned_paths(before) == {str(reclaimable.relative_to(child)), str(second.relative_to(child))}
    [row] = before["rows"]
    [group] = row["store_copies"]

    # A resealed plan cannot make a file whose bytes differ from its digest name a store copy,
    info = mismatched.stat()
    forged = {"relative_paths": [str(mismatched.relative_to(child))], "inode": info.st_ino, "nlink": 1,
              "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "sha256": gc.file_sha(mismatched)}
    assert gc.apply_replay_cache_retention(reseal(before, {**row, "store_copies": [forged]}),
                                           ack=gc.ACK, process_root=proc)["removed_bytes"] == 0
    # leave one of a copy's names behind,
    partial = {**group, "relative_paths": [str(reclaimable.relative_to(child))], "nlink": 1}
    assert gc.apply_replay_cache_retention(reseal(before, {**row, "store_copies": [partial]}),
                                           ack=gc.ACK, process_root=proc)["removed_bytes"] == 0
    # or reach outside the replay's prepared-references.
    for names in ([group["relative_paths"][0]] * 2, ["stage_replay_report.v1.json"], ["../outside"]):
        with pytest.raises(ValueError, match="replay_cache_member_unsafe"):
            gc.apply_replay_cache_retention(
                reseal(before, {**row, "store_copies": [{**group, "relative_paths": names}]}),
                ack=gc.ACK, process_root=proc)

    # A reader that appears after the plan keeps the copy at apply; the next plan keeps its root.
    process = proc / "123"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python")
    (process / "environ").write_bytes(b"")
    (process / "fd" / "3").symlink_to(reclaimable)
    result = gc.apply_replay_cache_retention(before, ack=gc.ACK, process_root=proc)
    assert result["removed_bytes"] == 0
    assert result["skipped"] == [{"root": str(child), "reason": "active_reference"}]
    assert plan(root, proc)["kept"] == [{"root": str(child), "reason": "active_reference"}]

    # Linked elsewhere after the plan: apply rechecks the link count and keeps every name.
    (process / "fd" / "3").unlink()
    os.link(reclaimable, tmp_path / "late-link")
    result = gc.apply_replay_cache_retention(before, ack=gc.ACK, process_root=proc)
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

    estimate = gc.estimate_replay_cache_retention(replay_root=root, process_root=proc, now=time.time() + 120)

    monkeypatch.setattr(gc, "file_sha", real_file_sha)
    p = plan(root, proc)
    assert "rows" not in estimate and estimate["digests_verified"] is False
    # Only a plan reads the bytes, so only a plan can tell that a copy is not what its name says.
    assert p["candidate_bytes"] == data.stat().st_size + copy.stat().st_size
    assert estimate["estimated_candidate_bytes"] == p["candidate_bytes"] + mismatched.stat().st_size


def test_refused_parent_replay_is_finished_for_retention(tmp_path):
    """A parent replay whose worker pass raised has written its report and returned, and its
    fetcher refuses every fetch by construction. Before the release, that path recorded no
    nothing_fetched, so such a replay's copies were never reclaimed."""
    root, child, data, proc = setup(tmp_path)
    report = child / "stage_replay_report.v1.json"

    def write(**fields):
        report.write_text(json.dumps({"schema_version": "task_evaluation_parent_replay_report.v1", **fields}))

    write(status="worker_refused", paid_execution_requested=False, provider_mutation_performed=False)
    assert gc.completed_report(child) == report
    assert plan(root, proc)["candidate_bytes"] == data.stat().st_size

    for fields in (
        {"status": "refused", "paid_execution_requested": False, "provider_mutation_performed": False},
        {"status": "worker_refused", "paid_execution_requested": True, "provider_mutation_performed": False},
        {"status": "worker_refused", "paid_execution_requested": False, "provider_mutation_performed": True},
        {"status": "worker_refused", "provider_mutation_performed": False},
    ):
        write(**fields)
        assert gc.completed_report(child) is None, fields
