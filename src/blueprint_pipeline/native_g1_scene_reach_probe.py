"""Measure static G1 reach bounds for a verified captured rigid task packet.

This is an authoring diagnostic. A clear floor pose and a nominal arm length do
not prove a feasible grasp, and a failed static bound does not rule out walking,
leaning, or crouching during an episode.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_bundle import verify_native_task_arena_packet
from .native_task_arena_packet import REQUEST_SCHEMA_VERSION
from .scene_placement.robot_profile import UNITREE_G1_PROFILE
from .scene_placement.types import SceneObject
from .scene_placement.usd_index import UsdSceneSpatialIndex
from .scene_placement.validation import validate_stand_pose


SCHEMA = "native_g1_scene_reach_probe.v1"


def probe_g1_scene_static_reach(
    *,
    source_packet_dir: Path,
    floor_z_m: float,
    radius_m: float = 1.25,
    grid_step_m: float = 0.05,
) -> dict[str, Any]:
    """Return a digest-bound floor and nominal-arm diagnostic for one packet."""

    if (
        not all(math.isfinite(value) for value in (floor_z_m, radius_m, grid_step_m))
        or radius_m <= 0
        or grid_step_m <= 0
        or radius_m / grid_step_m > 100
    ):
        raise ValueError("g1_reach_probe_grid_invalid")
    source_root, receipt, _ = verify_native_task_arena_packet(source_packet_dir)
    request = json.loads((source_root / f"{REQUEST_SCHEMA_VERSION}.json").read_text())
    task = request["task_spec"]
    start = task.get("start_pose_world")
    target_position = task.get("target_position_world_m")
    bounds = task.get("subject_collision_bounds_scoring_frame_m")
    if (
        task.get("task_kind") != "rigid_pick_place"
        or not isinstance(start, list)
        or len(start) != 7
        or start[3:] != [0.0, 0.0, 0.0, 1.0]
        or not isinstance(target_position, list)
        or len(target_position) != 3
        or not isinstance(bounds, dict)
        or not isinstance(bounds.get("minimum"), list)
        or not isinstance(bounds.get("maximum"), list)
        or len(bounds["minimum"]) != 3
        or len(bounds["maximum"]) != 3
        or not all(
            math.isfinite(float(value))
            for value in (*start, *target_position, *bounds["minimum"], *bounds["maximum"])
        )
    ):
        raise ValueError("g1_reach_probe_task_unsupported")
    subject = SceneObject(
        id=str(task["subject_asset_id"]),
        label=str(task.get("instruction_subject_label") or "task subject"),
        bbox_min=tuple(start[i] + bounds["minimum"][i] for i in range(3)),
        bbox_max=tuple(start[i] + bounds["maximum"][i] for i in range(3)),
        centroid=tuple(start[:3]),
    )
    collision_bindings = [
        row for row in receipt["source_bindings"] if row["semantic_role"] == "scene_collision"
    ]
    if len(collision_bindings) != 1:
        raise ValueError("g1_reach_probe_collision_source_missing")
    collision = source_root / collision_bindings[0]["staged_relative_path"]
    obstacles = UsdSceneSpatialIndex(usd_path=str(collision)).obstacle_boxes()
    profile = UNITREE_G1_PROFILE
    shoulder_z = floor_z_m + profile.pelvis_height_m + profile.shoulder_above_root_m
    extent = math.floor(radius_m / grid_step_m + 1e-9)
    clear_count = 0
    best_pick: tuple[float, list[float]] | None = None
    best_place: tuple[float, list[float]] | None = None
    best_pair: tuple[float, list[float]] | None = None
    for ix in range(-extent, extent + 1):
        for iy in range(-extent, extent + 1):
            x = start[0] + ix * grid_step_m
            y = start[1] + iy * grid_step_m
            yaw = math.atan2(start[1] - y, start[0] - x)
            pose = [x, y, floor_z_m + profile.pelvis_height_m, yaw]
            verdict = validate_stand_pose(
                tuple(pose[:3]),
                yaw,
                subject,
                obstacles,
                floor_z_m,
                robot_profile=profile,
            )
            if not verdict.ok:
                continue
            clear_count += 1
            lateral = profile.shoulder_lateral_offset_m
            shoulders = (
                (x - lateral * math.sin(yaw), y + lateral * math.cos(yaw)),
                (x + lateral * math.sin(yaw), y - lateral * math.cos(yaw)),
            )
            pick_xy = min(math.dist(shoulder, start[:2]) for shoulder in shoulders)
            place_xy = min(math.dist(shoulder, target_position[:2]) for shoulder in shoulders)
            pair_distance = max(
                math.hypot(pick_xy, shoulder_z - start[2]),
                math.hypot(place_xy, shoulder_z - target_position[2]),
            )
            for value, name in ((pick_xy, "pick"), (place_xy, "place"), (pair_distance, "pair")):
                prior = {"pick": best_pick, "place": best_place, "pair": best_pair}[name]
                if prior is None or value < prior[0]:
                    if name == "pick":
                        best_pick = (value, pose)
                    elif name == "place":
                        best_place = (value, pose)
                    else:
                        best_pair = (value, pose)

    def result(best: tuple[float, list[float]] | None, z: float) -> dict[str, Any] | None:
        if best is None:
            return None
        horizontal, pose = best
        distance = math.hypot(horizontal, shoulder_z - z)
        return {
            "minimum_horizontal_shoulder_distance_m": round(horizontal, 6),
            "nominal_shoulder_distance_m": round(distance, 6),
            "sampled_pose_world_xyzyaw": [round(value, 6) for value in pose],
            "within_nominal_arm_span": distance <= profile.arm_span_m,
        }

    report = {
        "schema_version": SCHEMA,
        "claim_ceiling": "development_only_static_geometry",
        "source_packet_receipt_digest": receipt["receipt_digest"],
        "source_scene_plan_digest": receipt["arena_scene_plan_digest"],
        "scene_id": request["scene_id"],
        "task_id": request["task_id"],
        "robot_profile_id": profile.robot_id,
        "floor_z_m": floor_z_m,
        "grid_radius_m": radius_m,
        "grid_step_m": grid_step_m,
        "sampled_nominal_stance_count": clear_count,
        "nominal_shoulder_z_m": round(shoulder_z, 6),
        "nominal_arm_span_m": profile.arm_span_m,
        "minimum_shoulder_lowering_if_horizontally_aligned_m": round(
            max(0.0, shoulder_z - min(start[2], target_position[2]) - profile.arm_span_m), 6
        ),
        "pick": result(best_pick, start[2]),
        "place": result(best_place, target_position[2]),
        "single_pose_pick_and_place": (
            {
                "minimum_worst_nominal_shoulder_distance_m": round(best_pair[0], 6),
                "sampled_pose_world_xyzyaw": [round(value, 6) for value in best_pair[1]],
                "within_nominal_arm_span": best_pair[0] <= profile.arm_span_m,
            }
            if best_pair is not None
            else None
        ),
        "interpretation": (
            "no_floor_clear_pose_sampled"
            if clear_count == 0
            else "nominal_static_reach_candidate"
            if best_pair is not None and best_pair[0] <= profile.arm_span_m
            else "body_motion_or_new_stance_required"
        ),
        "limitations": [
            "floor clearance uses static USD obstacle boxes and the configured G1 footprint",
            "arm span is a shoulder-to-target bound, not inverse kinematics or grasp proof",
            "walking, leaning, crouching, contacts, camera visibility, and policy behavior are untested",
        ],
    }
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-packet", type=Path, required=True)
    parser.add_argument("--floor-z", type=float, required=True)
    parser.add_argument("--radius", type=float, default=1.25)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = probe_g1_scene_static_reach(
        source_packet_dir=args.source_packet,
        floor_z_m=args.floor_z,
        radius_m=args.radius,
        grid_step_m=args.grid_step,
    )
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("g1_reach_probe_output_exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "report_digest": report["report_digest"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
