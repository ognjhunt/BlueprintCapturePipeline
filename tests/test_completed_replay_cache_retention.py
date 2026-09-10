from __future__ import annotations

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
