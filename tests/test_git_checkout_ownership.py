"""Root-run deploy Git must leave the service account's checkout its own (2026-10-27)."""
import json
import os
import pwd
import subprocess
from pathlib import Path

from blueprint_pipeline import git_checkout_ownership as ownership


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "checkout"
    (repo / "pkg" / "sub").mkdir(parents=True)
    (repo / ".venv" / "lib").mkdir(parents=True)
    for name, text in (("a.txt", "a"), ("pkg/b.py", "b"), ("pkg/sub/c.py", "c"), (".gitignore", ".venv/\n"),
                       (".venv/lib/x.py", "x"), ("scratch.txt", "untracked")):
        (repo / name).write_text(text)
    for argv in (["init", "-q"], ["add", "a.txt", "pkg", ".gitignore"],
                 ["-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "seed"]):
        subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True)
    return repo


def test_git_runs_as_the_checkout_owner_only_when_root_runs_it(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    assert ownership.checkout_owner(repo) is None and ownership.git_run_options(repo) == {}
    monkeypatch.setattr(ownership.os, "geteuid", lambda: 0)
    metadata = (repo / ".git").stat()
    options = ownership.git_run_options(repo)
    account = pwd.getpwuid(metadata.st_uid)
    assert (options["user"], options["group"]) == (metadata.st_uid, metadata.st_gid)
    assert options["env"]["HOME"] == account.pw_dir and options["env"]["USER"] == account.pw_name
    # A checkout root owns stays root's.
    assert ownership.checkout_owner("/") is None


def test_repair_gives_back_git_data_and_tracked_files_but_not_untracked_paths(tmp_path):
    repo = _repo(tmp_path)
    calls = []
    owner = (os.getuid() + 1, os.getgid() + 1)
    result = ownership.repair_checkout_ownership(repo, owner_uid=owner[0], owner_gid=owner[1],
                                                 chown=lambda path, uid, gid: calls.append((path, uid, gid)))
    repaired = {Path(path).relative_to(repo).as_posix() for path, *_ in calls}
    git_entries = {".git", *(p.relative_to(repo).as_posix() for p in (repo / ".git").rglob("*"))}
    assert git_entries <= repaired
    assert {"a.txt", ".gitignore", "pkg", "pkg/b.py", "pkg/sub", "pkg/sub/c.py"} <= repaired
    assert not repaired & {".", ".venv", ".venv/lib", ".venv/lib/x.py", "scratch.txt"}
    assert {(uid, gid) for _, uid, gid in calls} == {owner}
    assert result["repaired_count"] == len(calls) == len(repaired)
    # Entries already the owner's are left alone.
    untouched = []
    ownership.repair_checkout_ownership(repo, owner_uid=os.getuid(), owner_gid=os.getgid(),
                                        chown=lambda *call: untouched.append(call))
    assert untouched == []


def test_the_repair_command_does_nothing_unless_root_runs_it(tmp_path, capsys):
    assert ownership.main(["repair", "--repo", str(_repo(tmp_path))]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "not_needed"}
