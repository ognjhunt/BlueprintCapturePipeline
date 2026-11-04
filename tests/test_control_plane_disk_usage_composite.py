# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_disk_usage.py
#   src/blueprint_pipeline/control_plane_disk_usage_cursor.py
"""A partitioned usage survey completes across passes and attributes the whole mount."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline import control_plane_disk_usage_cursor as cursor_module
from blueprint_pipeline.control_plane_disk_usage import survey_usage
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _common(root: Path, *, limit: int) -> dict:
    return {"aliases": {}, "prefixes": (str(root),), "classify": lambda _path: None,
            "statvfs": lambda _path: SimpleNamespace(f_blocks=10000, f_bfree=5000, f_frsize=4096),
            "max_entries": limit, "max_seconds": 30, "mountinfo": str(root / "absent-mountinfo")}


def _passes(root: Path, *, partitions: int = 1, limit: int = 40, previous: dict | None = None,
            clock=None, max_passes: int = 12) -> list[dict]:
    results = []
    for _ in range(max_passes):
        extra = {} if clock is None else {"clock": clock}
        result = survey_usage([str(root)], continuation=previous, partitions_per_mount=partitions,
                              **_common(root, limit=limit), **extra)
        results.append(result)
        previous = result["continuation"]
        if result["status"] == "complete":
            break
    return results


def _full(root: Path) -> dict:
    return survey_usage([str(root)], **_common(root, limit=100_000))


def _root_inode_bytes(root: Path) -> int:
    info = os.lstat(root)
    return info.st_blocks * 512 or info.st_size


def _write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_the_cycle_completes_and_matches_a_full_walk(tmp_path: Path) -> None:
    root = tmp_path / "volume"
    _write(root / "a" / "one.bin", 10_000)
    _write(root / "a" / "two.bin", 20_000)
    _write(root / "b" / "nested" / "three.bin", 30_000)
    _write(root / "loose.bin", 5_000)
    os.link(root / "a" / "one.bin", root / "a" / "one-again.bin")
    results = _passes(root)
    final = results[-1]
    assert [result["status"] for result in results] == ["truncated", "truncated", "complete"]
    assert final["byte_accounting"] == "rolling_partition_composite"
    assert final["scope"] == "rolling_partition_composite"
    assert final["unreadable_bytes"] == 0 and final["container_runtime_complete"] is True
    assert final["composite"]["partitions"] == 3 and final["composite"]["unmeasured"] == 0
    full = _full(root)
    # Partition passes measure every partition, not the mount root's own inode.
    assert final["mounts"][0]["surveyed_bytes"] == full["mounts"][0]["surveyed_bytes"] - _root_inode_bytes(root)
    assert {row["root"]: row["allocated_bytes"] for row in final["top_roots"]} == {
        row["root"]: row["allocated_bytes"] for row in full["top_roots"] if row["root"] != str(root)}
    assert final["retirement_authority"] is False and final["mutations"] == 0


def test_a_partition_too_large_for_one_pass_is_split_until_it_fits(tmp_path: Path) -> None:
    """``/opt`` with 93 release checkouts could never finish inside one pass."""

    root = tmp_path / "volume"
    for child in ("x", "y", "z"):
        for index in range(6):
            _write(root / "big" / child / f"{index}.bin", 4_096)
    _write(root / "big" / "loose.bin", 4_096)
    _write(root / "small" / "file.bin", 4_096)
    results = _passes(root, limit=40)  # each partition may visit at most 10 entries
    final = results[-1]
    assert final["status"] == "complete"
    assert any("partition_split" in result["continuation_status"] for result in results)
    assert all("stale_cursor_identity_refused" not in result["continuation_status"] for result in results)
    # A split is not churn: the parent was replaced by its parts, not lost.
    assert all(result["partition_churn"]["departed"] == 0 for result in results)
    assert final["composite"]["partitions"] == 5  # big's own entries, x, y, z and small
    full = _full(root)
    assert final["mounts"][0]["surveyed_bytes"] == full["mounts"][0]["surveyed_bytes"] - _root_inode_bytes(root)
    split = final["continuation"]["roots"]
    assert len(next(iter(split.values()))["splits"]) == 1


def test_a_pass_fills_its_budget_instead_of_four_partitions(tmp_path: Path) -> None:
    root = tmp_path / "volume"
    for index in range(12):
        _write(root / f"job-{index:02d}" / "data.bin", 4_096)
    (result,) = _passes(root, partitions=cursor_module.MAX_PARTITIONS_PER_MOUNT, limit=10_000, max_passes=1)
    assert result["selected_partitions"] == 12
    assert result["status"] == "complete"


def test_a_measurement_older_than_a_day_keeps_the_composite_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "volume"
    for name in ("a", "b"):
        _write(root / name / "data.bin", 4_096)
    now = [1_000_000.0]
    results = _passes(root, clock=lambda: now[0])
    assert results[-1]["status"] == "complete"
    now[0] += cursor_module.COMPOSITE_MAX_AGE_SECONDS + 3_600
    (later,) = _passes(root, previous=results[-1]["continuation"], clock=lambda: now[0], max_passes=1)
    assert later["composite"]["oldest_age_seconds"] > cursor_module.COMPOSITE_MAX_AGE_SECONDS
    assert later["status"] == "truncated"
    assert later["byte_accounting"] == "rolling_partition_composite_lower_bound"


def test_a_redacted_name_keeps_the_cursor_valid(tmp_path: Path) -> None:
    """Publishing redacts credential-shaped labels; that must not invalidate the cursor's digest."""

    root = tmp_path / "volume"
    _write(root / "api_key-dump" / "data.bin", 4_096)
    _write(root / "plain" / "data.bin", 4_096)
    results = _passes(root)
    assert results[-1]["status"] == "complete"
    assert all("invalid_cursor_refused" not in result["continuation_status"] for result in results)
    assert "<redacted>" in {row["root"] for row in results[-1]["top_roots"]}


def test_a_v2_cursor_upgrades_without_a_refusal(tmp_path: Path) -> None:
    root = tmp_path / "volume"
    _write(root / "a" / "data.bin", 4_096)
    old = {"schema_version": "control_plane_usage_partition_cursor.v2", "mount_snapshot_digest": "0" * 64,
           "roots": {}, "cursor_digest": ""}
    old["cursor_digest"] = canonical_digest(old, digest_field="cursor_digest")
    (result,) = _passes(root, previous=old, max_passes=1)
    assert result["continuation_status"] == ["continued"]
    assert result["continuation"]["schema_version"] == cursor_module.SCHEMA_VERSION


def test_an_empty_root_is_measured_as_itself(tmp_path: Path) -> None:
    root = tmp_path / "volume"
    root.mkdir()
    (result,) = _passes(root, max_passes=1)
    assert result["status"] == "complete"
    assert result["mounts"][0]["surveyed_bytes"] == _root_inode_bytes(root)
