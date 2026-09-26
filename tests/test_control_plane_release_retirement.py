"""Deploy retires the release and runtime trees it supersedes, and nothing a live lease needs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.control_plane_release_leases import (
    BINDING_SCHEMA,
    DEFAULT_PROTECTION_SOURCES,
    LIVE_QUEUE_STATES,
    ProtectionSources,
    collect_release_protections,
)
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


def _write_json(path: Path, value: object, *, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _sources(tmp_path: Path, *, config_files: tuple[Path, ...] = ()) -> ProtectionSources:
    control_plane = tmp_path / "pipeline-control-plane"
    for queue, states in LIVE_QUEUE_STATES.items():
        for state in states:
            (control_plane / queue / state).mkdir(parents=True, exist_ok=True)
    for name in ("standing-authorizations", "task-evaluation-release-retention-bindings"):
        (control_plane / name).mkdir(exist_ok=True)
    profiles = tmp_path / "task-evaluation-launch-profiles"
    profiles.mkdir(exist_ok=True)
    return ProtectionSources(
        control_plane_root=control_plane,
        profile_dir=profiles,
        standing_authorization_dir=control_plane / "standing-authorizations",
        binding_root=control_plane / "task-evaluation-release-retention-bindings",
        lease_root=control_plane / "release-leases",
        config_files=config_files,
        intent_root=control_plane / "task-evaluation-scene-intents",
        launch_run_root=control_plane / "task-evaluation-launch-runs",
    )


def _host(tmp_path: Path, *, now: float) -> dict:
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
    sources = _sources(tmp_path)
    # A profile alone protects nothing; the queued launch that names it does.
    _write_json(sources.profile_dir / "live.json", {"profile_id": "live", "source_commit": C})
    _write_json(
        sources.control_plane_root / "task-evaluation-launches" / "pending" / "launch.json",
        {"launch_profile_id": "live"},
        mtime=now - 3_600,
    )
    return {
        "releases": releases,
        "runtimes": runtimes,
        "active": active,
        "sources": sources,
        "protections": collect_release_protections(sources, now=now, migrate=False),
    }


def _plan(host: dict, *, now: float, **overrides: object) -> dict:
    arguments: dict[str, object] = {
        "release_root": host["releases"],
        "runtime_root": host["runtimes"],
        "active_link": host["active"],
        "current_commit": A,
        "protections": host["protections"],
        "keep_last": 3,
        "now": lambda: now,
    }
    arguments.update(overrides)
    return build_release_retirement_plan(**arguments)  # type: ignore[arg-type]


def test_plan_protects_active_current_referenced_recent_and_young_commits(tmp_path: Path) -> None:
    now = 5_000_000.0
    host = _host(tmp_path, now=now)

    plan = _plan(host, now=now)

    assert plan["status"] == "dry_run"
    assert plan["active_commit"] == A
    assert plan["protected_commits"] == {
        A: ["active_release", "current_deploy", "keep_last"],
        C: ["live_queue:task-evaluation-launches/pending"],
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


def test_plan_blocks_without_a_proven_active_release_or_valid_inputs(tmp_path: Path) -> None:
    now = 5_000_000.0
    host = _host(tmp_path, now=now)

    broken_link = tmp_path / "broken"
    broken_link.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    unproven = _plan(host, now=now, active_link=broken_link)
    assert unproven["status"] == "blocked" and unproven["candidates"] == []
    assert unproven["alerts"] == ["release_retirement_blocked:release_retirement_active_link_invalid"]

    with pytest.raises(
        ControlPlaneReleaseRetirementError, match="release_retirement_apply_not_authorized"
    ):
        apply_release_retirement_plan(
            unproven, ack=EXECUTE_ACK, active_link=host["active"], release_root=host["releases"]
        )
    with pytest.raises(ControlPlaneReleaseRetirementError, match="release_retirement_input_invalid"):
        _plan(host, now=now, current_commit="not-a-commit")
    for protections in ({}, {**host["protections"], "schema_version": "grep.v0"},
                        {**host["protections"], "leases": [{"commit": "tree", "kind": "x"}]}):
        with pytest.raises(
            ControlPlaneReleaseRetirementError, match="release_retirement_input_invalid"
        ):
            _plan(host, now=now, protections=protections)


def test_apply_removes_only_planned_trees_and_re_proves_the_active_release(tmp_path: Path) -> None:
    now = 5_000_000.0
    host = _host(tmp_path, now=now)
    plan = _plan(host, now=now)

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


def test_configured_preparation_runtime_is_retained_after_queues_empty(tmp_path: Path) -> None:
    assert DEFAULT_PROTECTION_SOURCES.config_files == (
        Path("/etc/blueprint/task-evaluation-public-scene-machinery.json"),
        Path("/etc/blueprint/task-evaluation-scene-preparation-bootstrap.json"),
    )
    now = 5_000_000.0
    host = _host(tmp_path, now=now)
    machinery = tmp_path / "task-evaluation-public-scene-machinery.json"
    renderer = host["runtimes"] / "splat-render" / E
    machinery.write_text(json.dumps({"preparation": {"runtime_root": str(renderer)}}))
    sources = _sources(tmp_path, config_files=(machinery,))
    protections = collect_release_protections(sources, now=now, migrate=False)

    plan = _plan(host, now=now, protections=protections, keep_last=1)

    assert plan["status"] == "dry_run"
    assert plan["protected_commits"][E] == [
        "configured_runtime:task-evaluation-public-scene-machinery.json"
    ]
    result = apply_release_retirement_plan(
        plan, ack=EXECUTE_ACK, active_link=host["active"], release_root=host["releases"]
    )
    assert E not in {row["commit"] for row in result["removed"]}
    assert (renderer / "payload.bin").read_bytes() == b"x" * 128


def test_unreadable_protected_config_cannot_silently_allow_retirement(tmp_path: Path) -> None:
    now = 5_000_000.0
    host = _host(tmp_path, now=now)
    config = tmp_path / "active-machinery.json"
    actual = tmp_path / "actual.json"
    actual.write_text(json.dumps({"runtime_root": str(host["runtimes"] / E)}))
    config.symlink_to(actual)
    protections = collect_release_protections(
        _sources(tmp_path, config_files=(config,)), now=now, migrate=False
    )

    plan = _plan(host, now=now, protections=protections, keep_last=1)

    assert plan["status"] != "dry_run"
    assert "release_protection_config_unreadable:active-machinery.json" in plan["blockers"]


def test_a_release_a_live_paid_run_still_uses_is_never_retired(tmp_path: Path) -> None:
    """Deploys no longer wait out runs in flight, so those runs pin their own tree."""
    now = 5_000_000.0
    host = _host(tmp_path, now=now)

    plan = _plan(host, now=now, in_use_commits=[E])

    assert plan["protected_commits"][E] == ["in_use_by_live_process"]
    assert plan["candidates"] == []


def _commits(count: int) -> list[str]:
    return [f"{index + 1:040x}" for index in range(count)]


def test_stale_bindings_no_longer_protect_and_keep_last_three_retire(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    releases = tmp_path / "task-evaluation-control-plane-releases"
    runtimes = tmp_path / "system-runtimes"
    commits = _commits(30)
    for index, commit in enumerate(commits):
        _tree(releases, commit, mtime=now - (index + 2) * DAY)
    active = tmp_path / "active"
    active.symlink_to(releases / commits[0], target_is_directory=True)
    sources = _sources(tmp_path)
    live = [commits[10], commits[20]]
    for commit in live:
        _write_json(
            sources.control_plane_root / "task-evaluation-launches" / "processing" / f"{commit}.json",
            {"source_commit": commit},
            mtime=now - DAY,
        )
    stale = [commit for commit in commits[3:] if commit not in live]
    assert len(stale) == 25
    for commit in stale:
        _write_json(
            sources.binding_root / f"binding-{commit}.json",
            {
                "schema_version": BINDING_SCHEMA,
                "status": "required",
                "source_commit": commit,
                "reason": "terminal replay once needed this renderer",
            },
        )
    # The first typed deploy, twenty days ago, gave every legacy binding a 14-day lease.
    assert len(collect_release_protections(sources, now=now - 20 * DAY, migrate=True)["migrated"]) == 25
    protections = collect_release_protections(sources, now=now, migrate=True)
    assert protections["blockers"] == [] and len(protections["lapsed"]) == 25

    plan = build_release_retirement_plan(
        release_root=releases,
        runtime_root=runtimes,
        active_link=active,
        current_commit=commits[0],
        protections=protections,
        keep_last=3,
        now=lambda: now,
    )

    assert plan["status"] == "dry_run"
    assert set(plan["protected_commits"]) == set(commits[:3]) | set(live)
    assert plan["protected_commits"][commits[10]] == ["live_queue:task-evaluation-launches/processing"]
    assert plan["protected_commits"][commits[0]] == ["active_release", "current_deploy", "keep_last"]
    assert plan["lease_protected_tree_count"] == 2
    assert plan["protected_tree_count"] == 5
    assert plan["lapsed_count"] == 25
    assert plan["alerts"] == []
    assert sorted(row["commit"] for row in plan["candidates"]) == sorted(stale)


def test_receipt_groups_by_kind_and_alerts_over_twenty(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    releases = tmp_path / "task-evaluation-control-plane-releases"
    runtimes = tmp_path / "system-runtimes"
    commits = _commits(26)
    for index, commit in enumerate(commits):
        _tree(releases, commit, mtime=now - (index + 2) * DAY)
    active = tmp_path / "active"
    active.symlink_to(releases / commits[0], target_is_directory=True)
    machinery = _write_json(
        tmp_path / "task-evaluation-public-scene-machinery.json",
        {"preparation": {"runtimes": [f"{runtimes}/splat-render/{commits[1]}",
                                      f"{runtimes}/splat-render/{commits[22]}"]}},
    )
    sources = _sources(tmp_path, config_files=(machinery,))
    awaiting = sources.control_plane_root / "task-evaluation-launch-preparations" / "awaiting_capacity"
    for commit in [*commits[1:22], F]:  # F names a commit whose trees are already gone
        _write_json(awaiting / f"prep-{commit}.json",
                    {"request": {"expected_production_commit": commit}}, mtime=now - DAY)
    _write_json(
        sources.profile_dir / "authorized.json",
        {"profile_id": "authorized", "profile_digest": "sha256:authorized",
         "source_commit": commits[23], "allocator": {"max_spend_usd": 1.0}},
    )
    _write_json(
        sources.standing_authorization_dir / "authorized.json",
        {"schema_version": "task_evaluation_standing_launch_authorization.v1",
         "profile_id": "authorized", "profile_digest": "sha256:authorized",
         "max_launches": 2, "max_total_spend_usd": 5.0,
         "expires_at": datetime.fromtimestamp(now + DAY, tz=timezone.utc).isoformat()},
    )
    protections = collect_release_protections(sources, now=now, migrate=False)
    assert protections["blockers"] == []

    plan = build_release_retirement_plan(
        release_root=releases,
        runtime_root=runtimes,
        active_link=active,
        current_commit=commits[0],
        protections=protections,
        keep_last=1,
        now=lambda: now,
    )

    assert plan["protected_by_kind"] == {
        "active_release": [commits[0]],
        "configured_runtime": [commits[1], commits[22]],
        "current_deploy": [commits[0]],
        "keep_last": [commits[0]],
        "live_queue": sorted(commits[1:22]),
        "standing_authorization": [commits[23]],
    }
    # commits[1] is also configured, so it is not held by leases alone.
    assert plan["lease_protected_tree_count"] == 21
    assert plan["protected_tree_count"] == 24
    assert plan["alerts"] == ["release_retirement_lease_protected_trees:21"]
    assert sorted(row["commit"] for row in plan["candidates"]) == [commits[24], commits[25]]


def test_in_use_is_rechecked_at_apply_and_covers_runtime_trees(tmp_path: Path) -> None:
    now = 5_000_000.0
    host = _host(tmp_path, now=now)
    runtime_only = "1" * 40
    for component in ("splat-render", "scene-configuration"):
        _tree(host["runtimes"] / component, runtime_only, mtime=now - 10 * DAY, receipt=True)

    plan = _plan(host, now=now, in_use_commits=[runtime_only])
    assert plan["protected_commits"][runtime_only] == ["in_use_by_live_process"]
    assert [row["commit"] for row in plan["candidates"]] == [E]

    # A process started on E after the plan was built: nothing of E is removed.
    probes: list[str] = []
    busy = apply_release_retirement_plan(
        plan,
        ack=EXECUTE_ACK,
        active_link=host["active"],
        release_root=host["releases"],
        in_use_now=lambda: probes.append("probe") or {E},
    )
    assert probes == ["probe"]
    assert busy["removed"] == []
    assert busy["skipped"] == [{"commit": E, "reason": "in_use_at_apply"}]

    def unavailable() -> set[str]:
        raise OSError("process table unreadable")

    failed = apply_release_retirement_plan(
        plan,
        ack=EXECUTE_ACK,
        active_link=host["active"],
        release_root=host["releases"],
        in_use_now=unavailable,
    )
    assert failed["removed"] == []
    assert failed["skipped"] == [{"commit": E, "reason": "in_use_check_failed:OSError"}]
    assert (host["releases"] / E).is_dir()
    for component in ("splat-render", "scene-configuration"):
        assert (host["runtimes"] / component / E).is_dir()
        assert (host["runtimes"] / component / f"{E}.publication.v1.json").is_file()

    idle = apply_release_retirement_plan(
        plan,
        ack=EXECUTE_ACK,
        active_link=host["active"],
        release_root=host["releases"],
        in_use_now=set,
    )
    assert idle["removed_count"] == 5 and idle["skipped"] == []


def test_protection_blockers_retire_nothing(tmp_path: Path) -> None:
    now = 5_000_000.0
    host = _host(tmp_path, now=now)
    torn = host["sources"].control_plane_root / "task-evaluation-launches" / "pending" / "torn.json"
    torn.write_text("{", encoding="utf-8")
    protections = collect_release_protections(host["sources"], now=now, migrate=False)

    plan = _plan(host, now=now, protections=protections)

    blocker = "release_protection_queue_unreadable:task-evaluation-launches/pending/torn.json"
    assert plan["status"] == "blocked"
    assert plan["blockers"] == [blocker]
    assert plan["candidates"] == [] and plan["candidate_count"] == 0
    assert plan["alerts"] == [f"release_retirement_blocked:{blocker}"]
    with pytest.raises(ControlPlaneReleaseRetirementError, match="apply_not_authorized"):
        apply_release_retirement_plan(
            plan, ack=EXECUTE_ACK, active_link=host["active"], release_root=host["releases"]
        )
    assert (host["releases"] / E).is_dir()
