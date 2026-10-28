"""AUT-19 partial snapshots progress without turning unknown bytes into retirement."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline.control_plane_disk_usage import scratch_lifecycle_disposition, survey_usage


def _survey(root: Path, *, previous: dict | None = None, partitions: int = 1, limit: int = 20,
            mountinfo: Path | None = None) -> dict:
    return survey_usage([str(root)], aliases={}, prefixes=(str(root),), classify=lambda _path: None,
        statvfs=lambda _path: SimpleNamespace(f_blocks=10000, f_bfree=5000, f_frsize=4096),
        max_entries=limit, max_seconds=10, mountinfo=str(mountinfo or root / "absent-mountinfo"),
        continuation=previous, partitions_per_mount=partitions)


def test_bounded_passes_rotate_past_a_busy_historical_first_tree(tmp_path: Path) -> None:
    for name in ("a-history", "b-active", "c-active"):
        directory = tmp_path / name
        directory.mkdir()
        for index in range(100 if name == "a-history" else 2):
            (directory / str(index)).write_bytes(b"x" * 4096)
    first = _survey(tmp_path)
    second = _survey(tmp_path, previous=first["continuation"])
    third = _survey(tmp_path, previous=second["continuation"])
    roots = [{row["root"] for row in result["top_roots"]} for result in (first, second, third)]
    assert any("a-history" in name for name in roots[0])
    assert any("b-active" in name for name in roots[1])
    assert any("c-active" in name for name in roots[2])
    assert all(result["status"] == "truncated" and result["retirement_authority"] is False for result in (first, second, third))
    assert all(result["container_runtime_complete"] is False for result in (first, second, third))
    assert first["entries_visited"] <= 20
    assert second["mounts"][0]["survey_coverage_fraction"] == second["mounts"][0]["attributed_fraction"]
    assert second["mounts"][0]["classified_fraction"] == 0
    assert second["mounts"][0]["unvisited_bytes"] > 0


def test_new_jobs_do_not_restart_the_four_partition_active_root_cycle(tmp_path: Path) -> None:
    names = ("pipeline-control-plane", "production-gpu-artifacts", "pubsub-handoffs",
             "task-evaluation-inputs", "workspace")
    for name in names:
        (tmp_path / name).mkdir()
        (tmp_path / name / "data").write_bytes(b"x" * 4096)
    previous = None
    observed = []
    for tick in range(5):
        if tick:
            (tmp_path / f"z-new-job-{tick}").mkdir()
        result = _survey(tmp_path, previous=previous, partitions=4, limit=40)
        observed.append({Path(row["root"]).name for row in result["top_roots"]})
        previous = result["continuation"]
        assert "stale_cursor_identity_refused" not in result["continuation_status"]
        assert result["status"] == "truncated" and result["retirement_authority"] is False
        assert result["mounts"][0]["unvisited_bytes"] > 0
    assert "workspace" not in observed[0]
    assert "workspace" in observed[1]
    assert set(names) <= set.union(*observed)


def test_new_earlier_priority_root_joins_after_previously_unvisited_root(tmp_path: Path) -> None:
    for name in ("pipeline-control-plane", "workspace"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "data").write_bytes(b"x" * 4096)
    first = _survey(tmp_path)
    (tmp_path / "beta-policy-screening").mkdir()
    second = _survey(tmp_path, previous=first["continuation"])
    assert "workspace" in {Path(row["root"]).name for row in second["top_roots"]}
    assert "stale_cursor_identity_refused" not in second["continuation_status"]


def test_inode_and_mount_identity_changes_refuse_old_cursor(tmp_path: Path) -> None:
    child = tmp_path / "a"
    child.mkdir()
    (child / "data").write_bytes(b"x")
    original = _survey(tmp_path)
    child.rename(tmp_path / "old-a")
    child.mkdir()
    changed = _survey(tmp_path, previous=original["continuation"])
    assert "stale_cursor_identity_refused" in changed["continuation_status"]
    mountinfo = tmp_path.parent / "new-mountinfo"
    mountinfo.write_text("1 0 0:1 / /some-new-mount rw - tmpfs none rw\n")
    remounted = _survey(tmp_path, previous=changed["continuation"], mountinfo=mountinfo)
    assert "stale_cursor_identity_refused" in remounted["continuation_status"]


def test_symlink_partition_is_not_followed_and_inventory_stays_unknown(tmp_path: Path) -> None:
    target = tmp_path.parent / "outside"
    target.mkdir()
    (target / "secret-input").write_bytes(b"z" * 100000)
    (tmp_path / "linked").symlink_to(target)
    result = _survey(tmp_path)
    assert result["inventory_complete"] is False
    assert result["unreadable_bytes"] is None
    assert result["mounts"][0]["surveyed_bytes"] == 0


def test_hardlinks_across_selected_partitions_count_once_per_snapshot(tmp_path: Path) -> None:
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    source = tmp_path / "a" / "data"
    source.write_bytes(b"x" * 10000)
    os.link(source, tmp_path / "b" / "linked")
    result = _survey(tmp_path, partitions=2)
    assert result["hardlinks"]["duplicate_names_skipped"] == 1
    assert result["hardlinks"]["shared_inodes"] == 1


def test_missing_or_incomplete_reference_inventory_never_authorizes_retirement() -> None:
    root = "/mnt/blueprint-work/loose-job"
    base = {"schema_version": "control_plane_lane_scratch_census.v1", "observed_at_epoch": 100,
            "status": "complete", "scan_errors": [], "rows": [{"path": root, "references": ["pin"]}]}
    assert scratch_lifecycle_disposition(root, None, now=100)["lifecycle_disposition"] == "unknown"
    assert scratch_lifecycle_disposition(root, {**base, "status": "incomplete"}, now=100)["lifecycle_disposition"] == "unknown"
    disposition = scratch_lifecycle_disposition(root, base, now=100)
    assert disposition["lifecycle_disposition"] == "active_or_referenced" and disposition["retirement_eligible"] is False
    assert scratch_lifecycle_disposition(root, base, now=500)["lifecycle_disposition"] == "unknown"
    no_references = {**base, "rows": [{"path": root, "references": []}]}
    assert scratch_lifecycle_disposition(root, no_references, now=100)["retirement_eligible"] is False
