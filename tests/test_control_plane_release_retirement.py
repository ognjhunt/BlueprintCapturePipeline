"""Deploy retires the release and runtime trees it supersedes, and nothing a launch still names."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline.control_plane_release_retirement import (
    EXECUTE_ACK,
    ControlPlaneReleaseRetirementError,
    apply_release_retirement_plan,
    build_release_retirement_plan,
)


DAY = 86_400.0
A, C, D, E, F = ("a" * 40, "c" * 40, "d" * 40, "e" * 40, "f" * 40)


def _tree(root: Path, commit: str, *, mtime: float, receipt: bool = False) -> Path:
    directory = root / commit
    directory.mkdir(parents=True)
    payload = directory / "payload.bin"
    payload.write_bytes(b"x" * 128)
    for path in (payload, directory):
        os.utime(path, (mtime, mtime))
    if receipt:
        receipt_path = root / f"{commit}.publication.v1.json"
        receipt_path.write_text("{}", encoding="utf-8")
        os.utime(receipt_path, (mtime, mtime))
    return directory


def _host(tmp_path: Path, *, now: float) -> dict[str, Path]:
    releases = tmp_path / "releases"
    runtimes = tmp_path / "system-runtimes"
    _tree(releases, A, mtime=now - 1 * DAY)
    _tree(releases, F, mtime=now - 3_600)  # young
    _tree(releases, D, mtime=now - 2 * DAY)
    _tree(releases, C, mtime=now - 3 * DAY)
    _tree(releases, E, mtime=now - 10 * DAY)
    (releases / "README.txt").write_text("not a release", encoding="utf-8")
    for component in ("splat-render", "scene-configuration"):
        _tree(runtimes / component, A, mtime=now - 1 * DAY, receipt=True)
        _tree(runtimes / component, E, mtime=now - 10 * DAY, receipt=True)
    active = tmp_path / "active"
    active.symlink_to(releases / A, target_is_directory=True)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "live.json").write_text(json.dumps({"source_commit": C}), encoding="utf-8")
    standing = tmp_path / "standing-authorizations"
    standing.mkdir()
    queue = tmp_path / "queue" / "pending"
    queue.mkdir(parents=True)
    return {
        "releases": releases,
        "runtimes": runtimes,
        "active": active,
        "profiles": profiles,
        "standing": standing,
        "queue": queue,
    }


def test_plan_protects_active_current_referenced_recent_and_young_commits(tmp_path: Path) -> None:
    now = 5_000_000.0
    host = _host(tmp_path, now=now)

    plan = build_release_retirement_plan(
        release_root=host["releases"],
        runtime_root=host["runtimes"],
        active_link=host["active"],
        current_commit=A,
        protected_reference_roots=[host["profiles"], host["standing"], host["queue"]],
        keep_last=3,
        now=lambda: now,
    )

    assert plan["status"] == "dry_run"
    assert plan["active_commit"] == A
    assert plan["protected_commits"] == {
        A: ["active_release", "current_deploy", "keep_last"],
        C: ["named_by_protected_reference"],
        D: ["keep_last"],
        F: ["keep_last"],
    }
    assert plan["unmanaged_children"] == ["README.txt"]
    assert [row["commit"] for row in plan["candidates"]] == [E]
    assert sorted(Path(path).name for path in plan["candidates"][0]["paths"]) == sorted(
        [E, E, E, f"{E}.publication.v1.json", f"{E}.publication.v1.json"]
    )
    assert plan["candidate_bytes"] == 3 * 128 + 2 * 2
    assert plan["evidence_roots_touched"] is False


def test_plan_blocks_without_protection_sources_or_a_proven_active_release(tmp_path: Path) -> None:
    now = 5_000_000.0
    host = _host(tmp_path, now=now)

    missing = build_release_retirement_plan(
        release_root=host["releases"],
        runtime_root=host["runtimes"],
        active_link=host["active"],
        current_commit=A,
        protected_reference_roots=[host["profiles"], tmp_path / "absent"],
        now=lambda: now,
    )
    assert missing["status"] == "blocked"
    assert missing["candidates"] == []
    assert missing["blockers"] == ["release_retirement_protected_reference_root_missing:absent"]

    broken_link = tmp_path / "broken"
    broken_link.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    unproven = build_release_retirement_plan(
        release_root=host["releases"],
        runtime_root=host["runtimes"],
        active_link=broken_link,
        current_commit=A,
        protected_reference_roots=[host["profiles"]],
        now=lambda: now,
    )
    assert unproven["status"] == "blocked" and unproven["candidates"] == []

    with pytest.raises(
        ControlPlaneReleaseRetirementError, match="release_retirement_apply_not_authorized"
    ):
        apply_release_retirement_plan(
            missing, ack=EXECUTE_ACK, active_link=host["active"], release_root=host["releases"]
        )
    with pytest.raises(ControlPlaneReleaseRetirementError, match="release_retirement_input_invalid"):
        build_release_retirement_plan(
            release_root=host["releases"],
            runtime_root=host["runtimes"],
            active_link=host["active"],
            current_commit="not-a-commit",
            protected_reference_roots=[host["profiles"]],
        )


def test_apply_removes_only_planned_trees_and_re_proves_the_active_release(tmp_path: Path) -> None:
    now = 5_000_000.0
    host = _host(tmp_path, now=now)
    plan = build_release_retirement_plan(
        release_root=host["releases"],
        runtime_root=host["runtimes"],
        active_link=host["active"],
        current_commit=A,
        protected_reference_roots=[host["profiles"], host["standing"], host["queue"]],
        keep_last=3,
        now=lambda: now,
    )

    with pytest.raises(ControlPlaneReleaseRetirementError, match="apply_not_authorized"):
        apply_release_retirement_plan(
            plan, ack="wrong", active_link=host["active"], release_root=host["releases"]
        )

    # The active link moved to a candidate after the plan: that commit survives.
    host["active"].unlink()
    host["active"].symlink_to(host["releases"] / E, target_is_directory=True)
    moved = apply_release_retirement_plan(
        plan, ack=EXECUTE_ACK, active_link=host["active"], release_root=host["releases"]
    )
    assert moved["removed"] == [] and moved["skipped"] == [{"commit": E, "reason": "protected_at_apply"}]
    assert (host["releases"] / E).is_dir()

    host["active"].unlink()
    host["active"].symlink_to(host["releases"] / A, target_is_directory=True)
    receipt = apply_release_retirement_plan(
        plan, ack=EXECUTE_ACK, active_link=host["active"], release_root=host["releases"]
    )
    assert receipt["status"] == "applied"
    assert receipt["active_commit"] == A
    assert receipt["removed_count"] == 5 and receipt["skipped"] == []
    assert not (host["releases"] / E).exists()
    for component in ("splat-render", "scene-configuration"):
        assert not (host["runtimes"] / component / E).exists()
        assert not (host["runtimes"] / component / f"{E}.publication.v1.json").exists()
        assert (host["runtimes"] / component / A).is_dir()
    for kept in (A, C, D, F):
        assert (host["releases"] / kept).is_dir()
    assert (host["releases"] / "README.txt").is_file()
