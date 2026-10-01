"""Plan 13c measurement and acceptance for the joined no-paid scene harness.

ADP-009D/day 28. Reports are development_only. The measurement layer cannot
substitute for the required production stage transitions or live worker proof.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.control_plane_disk_usage import tree_usage

REQUIRED_STAGES = (
    "scene_intake", "scene_preparation", "scene_configuration", "launch_preparation",
    "episode_compilation", "launch_activation", "policy_output_ingestion", "result_delivery", "retirement",
)


def p95(values: Sequence[float]) -> float:
    """Nearest-rank p95; an empty or invalid sample is never zero."""
    if not values or any(isinstance(v, bool) or not isinstance(v, (float, int))
                         or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("unmeasured_or_invalid_samples")
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def allocated_tree_bytes(root: Path) -> int:
    usage = tree_usage(root)
    if usage.unreadable:
        raise ValueError("allocation_measurement_incomplete")
    return usage.allocated_bytes


def create_run_roots(control_plane: Path, objects: Path, workers: Path) -> dict[str, Path]:
    """Create exclusive owned roots after checking all paths; never reuse user work."""
    roots = {"control_plane": Path(control_plane), "objects": Path(objects), "workers": Path(workers)}
    resolved = []
    for path in roots.values():
        if path.exists() or path.is_symlink():
            raise ValueError("run_root_exists")
        for parent in path.parents:
            if parent.is_symlink():
                raise ValueError("run_root_parent_symlink")
        resolved.append(path.absolute())
    for i, first in enumerate(resolved):
        for second in resolved[i + 1:]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("run_roots_overlap")
    for path in roots.values():
        path.mkdir(mode=0o700)
        marker = path / ".concurrency-harness-owner"
        with marker.open("x") as stream:
            stream.write(f"control_plane_concurrency_load_test.v1 uid={os.geteuid()}\n")
        marker.chmod(0o600)
    return roots


def build_summary(*, source_commit: str, expected_beta_concurrency: int, owner_confirmed: bool,
                  scenes: list[dict[str, Any]], measured_peak_concurrency: int, concurrency_hold_seconds: float,
                  baseline_allocated_bytes: int, final_allocated_bytes: int, maximum_retained_bytes: int,
                  maximum_delta_bytes: int, external_calls: int) -> dict[str, Any]:
    """Fail acceptance when even one chain, concurrency or retention gate is absent."""
    count = 2 * expected_beta_concurrency
    blockers: list[str] = []
    if len(scenes) != count or len({s["scene_id"] for s in scenes}) != count:
        blockers.append("requested_scene_count_not_completed")
    if measured_peak_concurrency != count or concurrency_hold_seconds <= 0:
        blockers.append("target_concurrency_not_observed")
    if final_allocated_bytes > maximum_retained_bytes:
        blockers.append("control_plane_retained_bytes_exceeded")
    delta = final_allocated_bytes - baseline_allocated_bytes
    if delta > maximum_delta_bytes:
        blockers.append("control_plane_disk_delta_exceeded")
    if external_calls:
        blockers.append("external_provider_calls_observed")
    samples: dict[str, list[dict[str, Any]]] = {name: [] for name in REQUIRED_STAGES}
    completed = 0
    for scene in scenes:
        identity = scene["scene_id"]
        rows = scene["stages"]
        good = [row.get("stage") for row in rows] == list(REQUIRED_STAGES)
        if not good:
            blockers.append(f"scene_chain_incomplete:{identity}")
        for row in rows:
            if row.get("status") != "completed":
                good = False
                blockers.extend(row.get("blockers") or [f"stage_incomplete:{identity}:{row.get('stage')}"])
            if row.get("stage") in samples and row.get("status") == "completed":
                samples[row["stage"]].append(row)
        for field in ("residual_leases", "residual_pins"):
            if scene.get(field) != 0:
                blockers.append(f"{field}:{identity}")
                good = False
        completed += int(good)
    metrics = {name: {field + "_p95": p95([row[field] for row in rows])
                     for field in ("wall_seconds", "cpu_seconds", "peak_allocated_bytes")}
               for name, rows in samples.items() if rows}
    return {"schema_version": "control_plane_concurrency_acceptance.v1", "source_commit": source_commit,
            "claim_ceiling": "development_only", "expected_beta_concurrency": expected_beta_concurrency,
            "requested_scenes": count, "completed_scenes": completed,
            "measured_peak_concurrency": measured_peak_concurrency,
            "concurrency_hold_seconds": concurrency_hold_seconds, "p95_method": "nearest_rank",
            "stage_metrics": metrics, "baseline_allocated_bytes": baseline_allocated_bytes,
            "final_allocated_bytes": final_allocated_bytes, "control_plane_delta_bytes": delta,
            "maximum_retained_bytes": maximum_retained_bytes, "maximum_delta_bytes": maximum_delta_bytes,
            "owner_confirmed_concurrency": owner_confirmed,
            "owner_sized_acceptance_complete": owner_confirmed and not blockers,
            "external_calls": external_calls, "status": "failed" if blockers else "passed",
            "blockers": sorted(set(blockers)), "scenes": scenes}
