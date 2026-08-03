"""Fast-lane contract tests for scripts/agent_workspace_gc.py.

Pins the fail-closed safety contract of the agent scratch reaper: dry-run by
default, ack-gated deletion, and keep rules for dirty/unpushed/recent/
allowlisted/evidence/non-git-workspace directories.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from scripts import agent_workspace_gc as gc

_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "gc-test",
    "GIT_AUTHOR_EMAIL": "gc-test@example.invalid",
    "GIT_COMMITTER_NAME": "gc-test",
    "GIT_COMMITTER_EMAIL": "gc-test@example.invalid",
    "GIT_TERMINAL_PROMPT": "0",
}


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        env=_GIT_ENV,
        check=True,
    )
    return proc.stdout


def _age(path: Path, days: float = 30.0) -> None:
    stamp = time.time() - days * 86400.0
    for dirpath, dirnames, filenames in os.walk(path, topdown=False, followlinks=False):
        for name in filenames + dirnames:
            os.utime(Path(dirpath) / name, (stamp, stamp), follow_symlinks=False)
    os.utime(path, (stamp, stamp), follow_symlinks=False)


def _make_pushed_clone(tmp_path: Path, name: str) -> Path:
    remote = tmp_path / "remotes" / f"{name}.git"
    remote.mkdir(parents=True)
    _git(remote, "init", "--bare", "--quiet")
    work = tmp_path / "ws" / name
    work.mkdir(parents=True)
    _git(work, "init", "--quiet", "-b", "main")
    (work / "file.txt").write_text("payload\n", encoding="utf-8")
    _git(work, "add", "file.txt")
    _git(work, "commit", "--quiet", "-m", "init")
    _git(work, "remote", "add", "origin", str(remote))
    _git(work, "push", "--quiet", "-u", "origin", "main")
    return work


def _run(ws_root: Path, *extra: str) -> int:
    return gc.main(["--workspace-root", str(ws_root), "--age-days", "7", *extra])


def test_dry_run_reports_but_never_deletes(tmp_path: Path, capsys) -> None:
    work = _make_pushed_clone(tmp_path, "scratch-clone")
    _age(work)
    assert _run(tmp_path / "ws") == 0
    out = capsys.readouterr().out
    assert "REAP" in out and "scratch-clone" in out
    assert "dry-run only" in out
    assert work.exists()


def test_apply_without_ack_refuses_and_deletes_nothing(tmp_path: Path) -> None:
    work = _make_pushed_clone(tmp_path, "scratch-clone")
    _age(work)
    assert _run(tmp_path / "ws", "--apply") == 2
    assert work.exists()


def test_apply_with_ack_reaps_clean_pushed_idle_clone(tmp_path: Path) -> None:
    work = _make_pushed_clone(tmp_path, "scratch-clone")
    _age(work)
    assert _run(tmp_path / "ws", "--apply", "--ack", gc.EXECUTE_ACK) == 0
    assert not work.exists()


def test_tracked_modification_is_kept(tmp_path: Path) -> None:
    work = _make_pushed_clone(tmp_path, "dirty-clone")
    (work / "file.txt").write_text("edited\n", encoding="utf-8")
    _age(work)
    assert _run(tmp_path / "ws", "--apply", "--ack", gc.EXECUTE_ACK) == 0
    assert work.exists()


def test_unpushed_commit_is_kept(tmp_path: Path) -> None:
    work = _make_pushed_clone(tmp_path, "unpushed-clone")
    (work / "new.txt").write_text("wip\n", encoding="utf-8")
    _git(work, "add", "new.txt")
    _git(work, "commit", "--quiet", "-m", "local only")
    _age(work)
    assert _run(tmp_path / "ws", "--apply", "--ack", gc.EXECUTE_ACK) == 0
    assert work.exists()


def test_no_remote_clone_is_kept(tmp_path: Path) -> None:
    work = tmp_path / "ws" / "no-remote"
    work.mkdir(parents=True)
    _git(work, "init", "--quiet", "-b", "main")
    (work / "f").write_text("x", encoding="utf-8")
    _git(work, "add", "f")
    _git(work, "commit", "--quiet", "-m", "init")
    _age(work)
    assert _run(tmp_path / "ws", "--apply", "--ack", gc.EXECUTE_ACK) == 0
    assert work.exists()


def test_recent_deep_file_keeps_directory(tmp_path: Path) -> None:
    work = _make_pushed_clone(tmp_path, "active-clone")
    _age(work)
    deep = work / "a" / "b"
    deep.mkdir(parents=True)
    (deep / "fresh.log").write_text("now\n", encoding="utf-8")
    assert _run(tmp_path / "ws", "--apply", "--ack", gc.EXECUTE_ACK) == 0
    assert work.exists()


def test_allowlist_and_evidence_names_are_kept(tmp_path: Path) -> None:
    primary = _make_pushed_clone(tmp_path, "BlueprintCapturePipeline")
    evidence = _make_pushed_clone(tmp_path, "policy-ranking-evidence-20260728")
    _age(primary)
    _age(evidence)
    assert _run(tmp_path / "ws", "--apply", "--ack", gc.EXECUTE_ACK) == 0
    assert primary.exists()
    assert evidence.exists()


def test_non_git_data_dir_in_workspace_root_is_never_deleted(tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "ws" / "experiment-2-downloads"
    data_dir.mkdir(parents=True)
    (data_dir / "blob.bin").write_bytes(b"\x00" * 128)
    _age(data_dir)
    assert _run(tmp_path / "ws", "--apply", "--ack", gc.EXECUTE_ACK) == 0
    assert data_dir.exists()
    assert "manual review" in capsys.readouterr().out


def test_non_git_dir_under_tmp_root_is_reaped_by_age(tmp_path: Path) -> None:
    scratch = tmp_path / "tmp" / "blueprint-lane-fix.abc123"
    scratch.mkdir(parents=True)
    (scratch / "clone.tar").write_bytes(b"\x00" * 64)
    _age(scratch)
    fresh = tmp_path / "tmp" / "still-active"
    fresh.mkdir()
    (fresh / "live.log").write_text("now\n", encoding="utf-8")
    rc = gc.main(
        [
            "--tmp-root",
            str(tmp_path / "tmp"),
            "--age-days",
            "7",
            "--apply",
            "--ack",
            gc.EXECUTE_ACK,
        ]
    )
    assert rc == 0
    assert not scratch.exists()
    assert fresh.exists()


def test_symlinked_dir_entry_is_skipped(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep\n", encoding="utf-8")
    link = tmp_path / "ws"
    link.mkdir()
    (link / "linked").symlink_to(target)
    _age(target)
    assert _run(tmp_path / "ws", "--apply", "--ack", gc.EXECUTE_ACK) == 0
    assert target.exists()
    assert (target / "keep.txt").exists()


def test_codex_prune_deletes_old_files_but_never_sqlite(tmp_path: Path) -> None:
    codex = tmp_path / "codex"
    sessions = codex / "sessions" / "2026" / "06"
    sessions.mkdir(parents=True)
    old_rollout = sessions / "rollout-old.jsonl"
    old_rollout.write_text("{}\n", encoding="utf-8")
    sqlite_file = codex / "sessions" / "index.sqlite"
    sqlite_file.write_bytes(b"db")
    fresh = codex / "sessions" / "rollout-fresh.jsonl"
    fresh.write_text("{}\n", encoding="utf-8")
    _age(sessions, days=90)
    os.utime(sqlite_file, (time.time() - 90 * 86400, time.time() - 90 * 86400))
    rc = gc.main(
        [
            "--workspace-root",
            str(tmp_path / "empty-ws"),
            "--codex",
            "--codex-root",
            str(codex),
            "--codex-age-days",
            "45",
            "--apply",
            "--ack",
            gc.EXECUTE_ACK,
        ]
    )
    assert rc == 0
    assert not old_rollout.exists()
    assert sqlite_file.exists()
    assert fresh.exists()
