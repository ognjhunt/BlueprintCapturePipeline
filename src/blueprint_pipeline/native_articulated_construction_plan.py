"""Compile a collision-clear native IK rehearsal around an articulated handle.

The scored positive must touch the exact handle and move the joint.  Before
that paid control, the construction canary needs to answer a narrower question:
can the selected robot reach the approach and the whole door-sweep workspace
without using task contact as an IK crutch?  This plan offsets each contact
waypoint a small, explicit distance along the door's derived outward normal.
It therefore exercises the same workspace while making no grasp or task-success
claim.  The exact-contact program remains a separate control gate.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .articulated_control_planner import plan_articulated_handle_trajectory
from .decision_evidence_contracts import canonical_digest
from .native_articulated_motion_geometry import SCHEMA_VERSION as MOTION_SCHEMA


SCHEMA_VERSION = "native_articulated_construction_phase_plan.v1"
DEFAULT_CLEARANCE_M = 0.025


class NativeArticulatedConstructionPlanError(ValueError):
    """Stable pre-native errors for an unusable scene plan or phase request."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def materialize_articulated_construction_phase_plan(
    scene_plan: Mapping[str, Any],
    *,
    clearance_m: float = DEFAULT_CLEARANCE_M,
    waypoint_count: int = 8,
) -> dict[str, Any]:
    """Resolve task-neutral, contact-clear targets from exact USD motion geometry."""

    try:
        plan = json.loads(json.dumps(dict(scene_plan), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeArticulatedConstructionPlanError(
            ["native_articulated_construction_scene_plan_invalid"]
        ) from exc
    if plan.get("task_kind") != "articulated_open_close":
        raise NativeArticulatedConstructionPlanError(
            ["native_articulated_construction_task_kind_invalid"]
        )
    try:
        clearance = float(clearance_m)
    except (TypeError, ValueError) as exc:
        raise NativeArticulatedConstructionPlanError(
            ["native_articulated_construction_clearance_invalid"]
        ) from exc
    if not math.isfinite(clearance) or clearance <= 0.0:
        raise NativeArticulatedConstructionPlanError(
            ["native_articulated_construction_clearance_invalid"]
        )
    motion = dict(plan.get("articulation", {}).get("motion_geometry") or {})
    if (
        motion.get("schema_version") != MOTION_SCHEMA
        or motion.get("motion_geometry_digest")
        != canonical_digest(motion, digest_field="motion_geometry_digest")
    ):
        raise NativeArticulatedConstructionPlanError(
            ["native_articulated_construction_motion_geometry_invalid"]
        )
    limits = list(motion.get("authored_limits_degrees") or [])
    if len(limits) != 2:
        raise NativeArticulatedConstructionPlanError(
            ["native_articulated_construction_joint_limits_invalid"]
        )
    trajectory = plan_articulated_handle_trajectory(
        hinge_point_world_m=motion["hinge_point_world_m"],
        hinge_axis_world=motion["hinge_axis_world_unit"],
        handle_grasp_point_closed_world_m=motion[
            "handle_grasp_point_closed_world_m"
        ],
        open_angle_degrees=motion["scripted_sweep_angle_degrees"],
        authored_limit_degrees=max(abs(float(value)) for value in limits),
        waypoint_count=int(waypoint_count),
        approach_standoff_m=0.12,
    )

    phases: list[dict[str, Any]] = [
        {
            "phase_id": "approach",
            "position_world_m": trajectory["approach_pose"]["position_world_m"],
            "corresponding_contact_phase": "approach",
            "clearance_m": 0.12,
        }
    ]
    for row in trajectory["waypoints"]:
        position = [
            float(row["position_world_m"][axis])
            + float(row["outward_normal_world"][axis]) * clearance
            for axis in range(3)
        ]
        phase_id = (
            "grasp_clearance"
            if int(row["waypoint_index"]) == 0
            else f"sweep_clearance_{int(row['waypoint_index']):02d}"
        )
        phases.append(
            {
                "phase_id": phase_id,
                "position_world_m": position,
                "corresponding_contact_phase": (
                    "grasp"
                    if int(row["waypoint_index"]) == 0
                    else f"sweep_{int(row['waypoint_index']):02d}"
                ),
                "door_angle_degrees": float(row["door_angle_degrees"]),
                "outward_normal_world": row["outward_normal_world"],
                "clearance_m": clearance,
            }
        )
    phases.extend(
        [
            {
                "phase_id": "release_clearance",
                "position_world_m": phases[-1]["position_world_m"],
                "corresponding_contact_phase": "release",
                "clearance_m": clearance,
            },
            {
                "phase_id": "retreat",
                "position_world_m": trajectory["retreat_pose"]["position_world_m"],
                "corresponding_contact_phase": "retreat",
                "clearance_m": 0.12,
            },
            {
                "phase_id": "recovery",
                "position_world_m": trajectory["approach_pose"]["position_world_m"],
                "corresponding_contact_phase": "recovery",
                "clearance_m": 0.12,
            },
        ]
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scene_plan_digest": plan.get("plan_digest"),
        "motion_geometry_digest": motion["motion_geometry_digest"],
        "target_joint_id": motion["target_joint_id"],
        "orientation_strategy": "preserve_native_reset_controlled_body_orientation",
        "clearance_m": clearance,
        "phases": phases,
        "phase_count": len(phases),
        "exact_contact_program_required_after_this_gate": True,
        "claim_boundary": {
            "phase_positions_are_native_ik_targets": True,
            "targets_are_contact_clear_not_a_grasp": True,
            "joint_motion_or_task_success_not_claimed": True,
            "collision_clearance_requires_native_contact_readback": True,
        },
        "plan_digest": "",
    }
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    return result


__all__ = [
    "DEFAULT_CLEARANCE_M",
    "NativeArticulatedConstructionPlanError",
    "SCHEMA_VERSION",
    "materialize_articulated_construction_phase_plan",
]
