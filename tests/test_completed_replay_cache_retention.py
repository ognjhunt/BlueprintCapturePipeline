from __future__ import annotations

import hashlib
import json
import os
import time

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
    return {f["relative_path"] for row in p["rows"] for f in row["files"]}


def test_content_addressed_scratch_copies_are_reclaimable(tmp_path):
    """2026-09-27: activation lookaheads copied the whole content store into each parent replay.
    A copy has no suffix, is often under 64 KiB and keeps the store's read-only mode; its name
    matching its digest is what makes it a disposable copy."""
    root, child, data, proc = setup(tmp_path)
    copy = _store_copy(child, b"a small read-only store blob")

    p = plan(root, proc)

    assert planned_paths(p) == {"working.ply", str(copy.relative_to(child))}
    [row] = p["rows"]
    assert {f["relative_path"]: f["sha256"] for f in row["files"]}[str(copy.relative_to(child))] == "sha256:" + copy.name
    assert p["candidate_bytes"] == 100000 + len(b"a small read-only store blob")
    result = gc.apply_replay_cache_retention(p, ack=gc.ACK, process_root=proc)
    assert result["removed_bytes"] == p["candidate_bytes"]
    assert not copy.exists() and not data.exists()
    for name in ["readonly.ply", "shared.ply", "run.log", "stage_replay_report.v1.json"]:
        assert (child / name).exists()
    assert plan(root, proc)["candidate_bytes"] == 0


def test_hardlinked_or_mismatched_scratch_blobs_are_kept(tmp_path):
    root, child, data, proc = setup(tmp_path)
    data.unlink()
    projected = _store_copy(child, b"a copy the replayed preparation also projected")
    os.link(projected, child / "prepared-references" / projected.name)
    mismatched = _store_copy(child, b"bytes that are not the named digest", name="0" * 64)
    misplaced = _store_copy(child, b"a store name outside the scratch store", directory=("content-addressed", "sha256"))
    newer = _store_copy(child, b"written after the report", seconds_before_report=-5)
    reclaimable = _store_copy(child, b"a single-link copy")

    before = plan(root, proc)

    assert planned_paths(before) == {str(reclaimable.relative_to(child))}

    # A resealed plan cannot make a file whose bytes differ from its digest name a store copy.
    forged = dict(before)
    info = mismatched.stat()
    forged["rows"] = [{**before["rows"][0], "files": [{
        "relative_path": str(mismatched.relative_to(child)), "inode": info.st_ino, "mtime_ns": info.st_mtime_ns,
        "size_bytes": info.st_size, "sha256": gc.file_sha(mismatched)}]}]
    forged["plan_digest"] = gc.digest({k: v for k, v in forged.items() if k != "plan_digest"})
    assert gc.apply_replay_cache_retention(forged, ack=gc.ACK, process_root=proc)["removed_bytes"] == 0

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

    # Linked after the plan: the copy no longer frees anything and apply rechecks its link count.
    (process / "fd" / "3").unlink()
    os.link(reclaimable, tmp_path / "late-link")
    result = gc.apply_replay_cache_retention(before, ack=gc.ACK, process_root=proc)
    assert result["removed_bytes"] == 0
    assert result["skipped"] == [{"path": str(reclaimable), "reason": "file_changed"}]
    for path in (projected, mismatched, misplaced, newer, reclaimable):
        assert path.exists(), path


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
