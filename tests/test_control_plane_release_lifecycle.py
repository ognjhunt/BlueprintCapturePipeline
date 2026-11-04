"""The storage GC tick retires superseded release trees behind receipts, and nothing it cannot rebuild."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_release_lifecycle as lifecycle
from blueprint_pipeline.control_plane_release_leases import LIVE_QUEUE_STATES, ProtectionSources
from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, run_storage_gc
from blueprint_pipeline.control_plane_storage_gc_reasons import build_storage_gc_summary

DAY = 86_400.0
NOW = 1_800_000_000.0


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _sources(tmp_path: Path) -> ProtectionSources:
    control_plane = tmp_path / "pipeline-control-plane"
    for queue, states in LIVE_QUEUE_STATES.items():
        for state in states:
            (control_plane / queue / state).mkdir(parents=True, exist_ok=True)
    for name in ("standing-authorizations", "task-evaluation-release-retention-bindings", "release-leases"):
        (control_plane / name).mkdir(exist_ok=True)
    profiles = tmp_path / "task-evaluation-launch-profiles"
    profiles.mkdir(exist_ok=True)
    return ProtectionSources(
        control_plane_root=control_plane,
        profile_dir=profiles,
        standing_authorization_dir=control_plane / "standing-authorizations",
        binding_root=control_plane / "task-evaluation-release-retention-bindings",
        lease_root=control_plane / "release-leases",
        intent_root=control_plane / "task-evaluation-scene-intents",
        launch_run_root=control_plane / "task-evaluation-launch-runs",
    )


def _age(path: Path, seconds_old: float) -> None:
    stamp = NOW - seconds_old
    os.utime(path, (stamp, stamp), follow_symlinks=False)


def _host(tmp_path: Path, *, releases_count: int = 5) -> dict:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    (source / ".gitignore").write_text("__pycache__/\noutput/\n", encoding="utf-8")
    commits = []
    for index in range(releases_count):
        (source / "module.py").write_text(f"VALUE = {index}\n", encoding="utf-8")
        _git(source, "add", "-A")
        _git(source, "commit", "-q", "-m", f"release {index}")
        commits.append(_git(source, "rev-parse", "HEAD"))
    releases = tmp_path / "releases"
    runtimes = tmp_path / "system-runtimes"
    releases.mkdir()
    for index, commit in enumerate(commits):
        _git(source, "worktree", "add", "-q", "--detach", str(releases / commit), commit)
        age = (releases_count - index) * DAY  # the newest is one day old
        _age(releases / commit, age)
        for component in ("splat-render", "scene-configuration"):
            tree = runtimes / component / commit
            tree.mkdir(parents=True)
            (tree / "payload.bin").write_bytes(b"r" * 8192)
            receipt = runtimes / component / f"{commit}.publication.v1.json"
            receipt.write_text("{}", encoding="utf-8")
            for path in (tree / "payload.bin", tree, receipt):
                _age(path, age)
    active = tmp_path / "active"
    active.symlink_to(releases / commits[-1], target_is_directory=True)
    proc = tmp_path / "proc"
    proc.mkdir()
    receipts = tmp_path / "release-retention"
    receipts.mkdir()
    return {"source": source, "commits": commits, "releases": releases, "runtimes": runtimes,
            "active": active, "proc": proc, "receipts": receipts, "sources": _sources(tmp_path)}


def _run(host: dict, **overrides: object) -> dict:
    arguments: dict[str, object] = {
        "release_root": host["releases"], "runtime_root": host["runtimes"], "active_link": host["active"],
        "receipt_root": host["receipts"], "sources": host["sources"], "apply": True, "enabled": True,
        "now": lambda: NOW, "proc_root": host["proc"], "keep_last": 2,
    }
    arguments.update(overrides)
    return lifecycle.run_release_lifecycle(**arguments)  # type: ignore[arg-type]


def _receipts(host: dict) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(host["receipts"].glob("release-lifecycle-*.json"))]


def test_superseded_clean_trees_retire_behind_a_receipt_and_can_be_staged_again(tmp_path: Path) -> None:
    host = _host(tmp_path)
    commits = host["commits"]
    entry = _run(host)
    retired = commits[:3]  # the newest two are kept, and the active is the newest
    assert entry["status"] == "applied"
    assert entry["retired_commits"] == sorted(retired)
    assert entry["retired_bytes"] > 0 and entry["candidate_bytes"] == entry["retired_bytes"]
    assert entry["retained_by_reason"]["keep_last"]["count"] == 2
    assert entry["blockers"] == [] and entry["alerts"] == []
    for commit in retired:
        assert not (host["releases"] / commit).exists()
        for component in ("splat-render", "scene-configuration"):
            assert not (host["runtimes"] / component / commit).exists()
            assert not (host["runtimes"] / component / f"{commit}.publication.v1.json").exists()
    for commit in commits[3:]:
        assert (host["releases"] / commit / "module.py").is_file()
    (receipt,) = _receipts(host)
    assert receipt["schema_version"] == lifecycle.RECEIPT_SCHEMA_VERSION
    assert [row["commit"] for row in receipt["retired"]] == sorted(retired)
    assert all(row["restore"] == {"step": lifecycle.RESTORE_STEP, "source_commit": row["commit"],
                                  "runtime_trees": "redeploy_commit"} for row in receipt["retired"])
    assert receipt["retired_bytes"] == entry["retired_bytes"]
    assert receipt["evidence_roots_touched"] is False
    summary = json.loads((host["receipts"] / lifecycle.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["status"] == "applied" and summary["alerts"] == []
    assert summary["retired_commits"] == sorted(retired) and summary["receipt"] == entry["receipt"]
    # Restorable: the stale registration is pruned and the exact commit checks out again.
    _git(host["source"], "worktree", "prune")
    _git(host["source"], "worktree", "add", "-q", "--detach", str(host["releases"] / retired[0]), retired[0])
    assert _git(host["releases"] / retired[0], "rev-parse", "HEAD") == retired[0]
    # A second tick with nothing left to retire writes no new receipt.
    again = _run(host, keep_last=5)
    assert again["status"] == "applied" and again["retired_commits"] == []
    assert len(_receipts(host)) == 1


def test_a_worktree_holding_bytes_its_commit_cannot_rebuild_is_kept(tmp_path: Path) -> None:
    host = _host(tmp_path)
    oldest, second, third = host["commits"][:3]
    (host["releases"] / oldest / "output").mkdir()
    (host["releases"] / oldest / "output" / "result.json").write_text("{}", encoding="utf-8")
    (host["releases"] / second / "notes.txt").write_text("untracked", encoding="utf-8")
    (host["releases"] / third / "__pycache__").mkdir()
    (host["releases"] / third / "__pycache__" / "module.cpython-312.pyc").write_bytes(b"\0" * 64)
    entry = _run(host)
    assert entry["local_state_commits"] == {oldest: "local_changes", second: "local_changes"}
    assert entry["retained_by_reason"][lifecycle.LOCAL_STATE_KIND]["count"] == 2
    assert entry["retired_commits"] == [third]  # bytecode is reproducible
    assert (host["releases"] / oldest / "output" / "result.json").is_file()
    assert (host["releases"] / second / "notes.txt").is_file()
    # Its runtime trees stay with it: a commit's trees retire together or not at all.
    assert (host["runtimes"] / "splat-render" / oldest).is_dir()


def test_a_tree_that_is_not_a_provable_checkout_is_kept(tmp_path: Path) -> None:
    host = _host(tmp_path)
    oldest = host["commits"][0]
    (host["releases"] / oldest / ".git").unlink()  # no longer a worktree of the source clone
    entry = _run(host)
    assert entry["local_state_commits"] == {oldest: "status_unavailable"}
    assert oldest not in entry["retired_commits"]
    assert (host["releases"] / oldest / "module.py").is_file()


def test_a_tree_a_live_process_uses_is_kept(tmp_path: Path) -> None:
    host = _host(tmp_path)
    oldest = host["commits"][0]
    process = host["proc"] / "4242"
    process.mkdir()
    (process / "cwd").symlink_to(host["releases"] / oldest / "scripts")
    (process / "cmdline").write_bytes(b"python\0")
    entry = _run(host)
    assert entry["retained_by_reason"]["in_use_by_live_process"]["count"] == 1
    assert oldest not in entry["retired_commits"]
    assert (host["releases"] / oldest).is_dir()


def test_opted_out_only_plans_and_says_so(tmp_path: Path) -> None:
    host = _host(tmp_path)
    entry = _run(host, enabled=False)
    assert entry["status"] == "skipped"
    assert entry["retired_commits"] == [] and entry["planned_commits"] == sorted(host["commits"][:3])
    assert all((host["releases"] / commit).is_dir() for commit in host["commits"])
    assert _receipts(host) == []
    summary = json.loads((host["receipts"] / lifecycle.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["status"] == "skipped" and summary["reason"] == "release_retirement_opted_out"
    assert not list((host["sources"].lease_root).rglob("*.json"))  # no lease written


def test_a_dry_run_tick_writes_nothing(tmp_path: Path) -> None:
    host = _host(tmp_path)
    entry = _run(host, apply=False)
    assert entry["status"] == "dry_run" and entry["planned_commits"] == sorted(host["commits"][:3])
    assert all((host["releases"] / commit).is_dir() for commit in host["commits"])
    assert not (host["receipts"] / lifecycle.SUMMARY_NAME).exists()


def test_a_protection_blocker_retires_nothing(tmp_path: Path) -> None:
    host = _host(tmp_path)
    launches = host["sources"].control_plane_root / "task-evaluation-launches" / "pending"
    (launches / "launch.json").write_text(json.dumps({"launch_profile_id": "never-published"}), encoding="utf-8")
    entry = _run(host)
    assert entry["status"] == "blocked"
    assert entry["blockers"] == ["release_protection_profile_missing:never-published"]
    assert entry["alerts"] == ["release_retirement_blocked:release_protection_profile_missing:never-published"]
    assert all((host["releases"] / commit).is_dir() for commit in host["commits"])
    summary = json.loads((host["receipts"] / lifecycle.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["status"] == "blocked" and summary["alerts"] == entry["alerts"]


def test_an_invalid_active_link_blocks(tmp_path: Path) -> None:
    host = _host(tmp_path)
    host["active"].unlink()
    entry = _run(host)
    assert entry["status"] == "blocked"
    assert entry["blockers"] == ["release_retirement_active_link_invalid"]
    assert all((host["releases"] / commit).is_dir() for commit in host["commits"])


@pytest.mark.parametrize(("raw", "expected"), [
    (None, (True, None)), ("", (True, None)), ("1", (True, None)), ("yes", (True, None)),
    ("0", (False, None)), ("false", (False, None)),
    ("maybe", (False, lifecycle.RELEASE_RETIREMENT_INVALID)),
])
def test_owner_switch(raw: str | None, expected: tuple[bool, str | None]) -> None:
    environ = {} if raw is None else {lifecycle.RELEASE_RETIREMENT_ENV: raw}
    assert lifecycle.release_retirement_setting(environ) == expected


def test_storage_gc_tick_runs_the_release_phase_and_summarizes_it(tmp_path: Path) -> None:
    calls = []

    def fake_lifecycle(**kwargs: object) -> dict:
        calls.append(kwargs)
        return {"status": "applied", "enabled": True, "candidate_bytes": 4096, "retired_bytes": 4096,
                "retained_by_reason": {"keep_last": {"count": 3, "bytes": None}}}

    releases = tmp_path / "releases"
    releases.mkdir()
    pins = tmp_path / "pins"
    pins.mkdir()
    settings = lifecycle.storage_gc_settings({lifecycle.RELEASE_ROOT_ENV: str(releases)})
    assert settings == {
        "release_root": str(releases), "runtime_root": str(lifecycle.DEFAULT_RUNTIME_ROOT),
        "active_link": str(lifecycle.DEFAULT_ACTIVE_LINK), "receipt_root": str(lifecycle.DEFAULT_RECEIPT_ROOT),
        "enabled": True, "alert": None,
    }
    report = run_storage_gc(
        content_store_roots=(), derived_roots=(), queue_roots=(), pins_root=pins, apply=True, ack=RUN_ACK,
        release_trees={**settings, "run": fake_lifecycle}, now=lambda: NOW,
    )
    assert report["opt_in"]["release_retirement"] is True
    assert report["release_trees"]["status"] == "applied"
    assert calls[0]["release_root"] == releases and calls[0]["apply"] is True and calls[0]["enabled"] is True
    summary = build_storage_gc_summary(report)
    assert summary["opt_in"]["release_retirement"] is True
    assert summary["phases"]["release_trees"]["removed_or_offloaded_bytes"] == 4096
    assert summary["phases"]["release_trees"]["retained_by_reason"] == {"keep_last": {"count": 3, "bytes": None}}


def test_storage_gc_tick_without_a_release_root_runs_no_phase(tmp_path: Path) -> None:
    pins = tmp_path / "pins"
    pins.mkdir()
    unset = lifecycle.storage_gc_settings({})
    assert unset["release_root"] is None and unset["enabled"] is True
    report = run_storage_gc(content_store_roots=(), derived_roots=(), queue_roots=(), pins_root=pins,
                            release_trees={**unset, "run": lambda **_: pytest.fail("ran")}, now=lambda: NOW)
    assert "release_trees" not in report and report["opt_in"]["release_retirement"] is True
    assert report["skipped_roots"] == []
    absent = lifecycle.storage_gc_settings({lifecycle.RELEASE_ROOT_ENV: str(tmp_path / "absent"),
                                            lifecycle.RELEASE_RETIREMENT_ENV: "0"})
    report = run_storage_gc(
        content_store_roots=(), derived_roots=(), queue_roots=(), pins_root=pins,
        release_trees={**absent, "run": lambda **_: pytest.fail("ran")}, now=lambda: NOW,
    )
    assert "release_trees" not in report and report["opt_in"]["release_retirement"] is False
    assert str(tmp_path / "absent") in report["skipped_roots"]
    report = run_storage_gc(content_store_roots=(), derived_roots=(), queue_roots=(), pins_root=pins,
                            now=lambda: NOW)
    assert "release_trees" not in report and report["opt_in"]["release_retirement"] is False


def test_an_invalid_switch_value_keeps_it_planning_and_alerts(tmp_path: Path) -> None:
    host = _host(tmp_path)
    settings = lifecycle.storage_gc_settings({
        lifecycle.RELEASE_ROOT_ENV: str(host["releases"]), lifecycle.RELEASE_RETIREMENT_ENV: "maybe",
        lifecycle.RELEASE_RUNTIME_ROOT_ENV: str(host["runtimes"]),
        lifecycle.RELEASE_ACTIVE_LINK_ENV: str(host["active"]),
        lifecycle.RELEASE_RECEIPT_ROOT_ENV: str(host["receipts"]),
    })

    def run(**kwargs: object) -> dict:
        return _run(host, **kwargs)

    skipped: list[str] = []
    entry = lifecycle.storage_gc_phase({**settings, "run": run}, apply=True, now=lambda: NOW,
                                       skipped_roots=skipped)
    assert entry["status"] == "skipped" and skipped == []
    assert entry["alerts"] == [lifecycle.RELEASE_RETIREMENT_INVALID]
    assert all((host["releases"] / commit).is_dir() for commit in host["commits"])
    summary = json.loads((host["receipts"] / lifecycle.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["alerts"] == [lifecycle.RELEASE_RETIREMENT_INVALID]
