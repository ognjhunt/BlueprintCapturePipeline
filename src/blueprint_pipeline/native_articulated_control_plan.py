"""Freeze one native Cartesian control plan after construction qualification.

The construction canary proves that the selected robot can reach a clearance
copy of the complete handle sweep.  This compiler joins that retained native
receipt to the exact-contact trajectory without naming a scene, appliance, or
joint.  The resulting plan is still only a deterministic simulator control;
it neither predicts a learned policy nor establishes physical performance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .articulated_control_planner import plan_articulated_handle_trajectory
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp_task_control_plan.v1"
CONSTRUCTION_RESULT_SCHEMA_VERSION = "native_task_arena_construction_result.v1"
WAYPOINT_COUNT = 8
ZERO_ACTION_STEPS = 80
MOTION_MINIMUM_STEPS = 1
MOTION_MAXIMUM_STEPS = 35
GRIPPER_DWELL_MINIMUM_STEPS = 8
GRIPPER_DWELL_MAXIMUM_STEPS = 20
ARRIVAL_TOLERANCE_M = 0.02
ARRIVAL_STABILITY_STEPS = 2
MAX_JOINT_DELTA_RAD = 0.03
MAX_JOINT_SETPOINT_LEAD_RAD = 0.20


class NativeArticulatedControlPlanError(ValueError):
    """Stable failures while joining construction evidence to a control plan."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _copy_mapping(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        copied = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeArticulatedControlPlanError([error]) from exc
    if not isinstance(copied, dict):
        raise NativeArticulatedControlPlanError([error])
    return copied


def _pose_phase(
    *,
    phase_id: str,
    position_world_m: Sequence[float],
    gripper_state: str,
    minimum_steps: int = MOTION_MINIMUM_STEPS,
    maximum_steps: int = MOTION_MAXIMUM_STEPS,
) -> dict[str, Any]:
    return {
        "phase_id": phase_id,
        "mode": "ik_pose",
        "target_position_world_m": [float(value) for value in position_world_m],
        # The native environment binds this to the controlled body's measured
        # reset orientation.  No appliance-facing axis is guessed here.
        "target_quaternion_world_xyzw": None,
        "gripper_state": gripper_state,
        "minimum_steps": minimum_steps,
        "maximum_steps": maximum_steps,
        "arrival_tolerance_m": ARRIVAL_TOLERANCE_M,
        "arrival_stability_steps": ARRIVAL_STABILITY_STEPS,
        "max_joint_delta_rad": MAX_JOINT_DELTA_RAD,
        "max_joint_setpoint_lead_rad": MAX_JOINT_SETPOINT_LEAD_RAD,
    }


def materialize_native_articulated_control_plan(
    *, scene_plan: Mapping[str, Any], construction_result: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the frozen exact-contact plan or fail on weak construction evidence."""

    scene = _copy_mapping(
        scene_plan, error="native_articulated_control_scene_plan_invalid"
    )
    construction = _copy_mapping(
        construction_result,
        error="native_articulated_control_construction_result_invalid",
    )
    errors: list[str] = []
    if (
        scene.get("schema_version") != "native_task_arena_scene_plan.v1"
        or scene.get("task_kind") != "articulated_open_close"
        or scene.get("plan_digest")
        != canonical_digest(scene, digest_field="plan_digest")
    ):
        errors.append("native_articulated_control_scene_plan_invalid")
    if (
        construction.get("schema_version") != CONSTRUCTION_RESULT_SCHEMA_VERSION
        or construction.get("result_digest")
        != canonical_digest(construction, digest_field="result_digest")
    ):
        errors.append("native_articulated_control_construction_result_invalid")
    if (
        construction.get("status") != "completed"
        or construction.get("construction_gate_qualified") is not True
        or construction.get("blockers") != []
    ):
        errors.append("native_articulated_control_construction_not_qualified")
    if construction.get("scene_plan_digest") != scene.get("plan_digest"):
        errors.append("native_articulated_control_construction_binding_mismatch")
    clearance_plan = construction.get("construction_phase_plan")
    clearance_phases = (
        list(clearance_plan.get("phases") or [])
        if isinstance(clearance_plan, Mapping)
        else []
    )
    if (
        not isinstance(clearance_plan, Mapping)
        or clearance_plan.get("plan_digest")
        != canonical_digest(clearance_plan, digest_field="plan_digest")
        or clearance_plan.get("scene_plan_digest") != scene.get("plan_digest")
        or not clearance_phases
    ):
        errors.append("native_articulated_control_clearance_plan_invalid")
    phase_results = construction.get("phase_results")
    expected_phase_ids = [
        str(row.get("phase_id") or "")
        for row in clearance_phases
        if isinstance(row, Mapping)
    ]
    observed_phase_ids = [
        str(row.get("phase_id") or "")
        for row in (phase_results or [])
        if isinstance(row, Mapping)
    ]
    if (
        not isinstance(phase_results, list)
        or not phase_results
        or observed_phase_ids != expected_phase_ids
        or any(
            not isinstance(row, Mapping) or row.get("target_reached") is not True
            for row in phase_results
        )
    ):
        errors.append("native_articulated_control_ik_preflight_incomplete")
    camera_gates = construction.get("camera_gates")
    if (
        not isinstance(camera_gates, Mapping)
        or set(camera_gates) != {"external", "wrist", "overview"}
        or any(
            not isinstance(row, Mapping) or row.get("passed") is not True
            for row in camera_gates.values()
        )
    ):
        errors.append("native_articulated_control_camera_preflight_incomplete")
    if not isinstance(construction.get("reset_replay"), Mapping) or (
        construction["reset_replay"].get("passed") is not True
    ):
        errors.append("native_articulated_control_reset_preflight_incomplete")
    if errors:
        raise NativeArticulatedControlPlanError(errors)

    motion = scene["articulation"]["motion_geometry"]
    limits = motion["authored_limits_degrees"]
    trajectory = plan_articulated_handle_trajectory(
        hinge_point_world_m=motion["hinge_point_world_m"],
        hinge_axis_world=motion["hinge_axis_world_unit"],
        handle_grasp_point_closed_world_m=motion[
            "handle_grasp_point_closed_world_m"
        ],
        open_angle_degrees=motion["scripted_sweep_angle_degrees"],
        authored_limit_degrees=max(abs(float(value)) for value in limits),
        waypoint_count=WAYPOINT_COUNT,
        approach_standoff_m=0.12,
    )
    waypoints = trajectory["waypoints"]
    actions = [
        _pose_phase(
            phase_id="approach",
            position_world_m=trajectory["approach_pose"]["position_world_m"],
            gripper_state="open",
        ),
        _pose_phase(
            phase_id="grasp_open",
            position_world_m=waypoints[0]["position_world_m"],
            gripper_state="open",
        ),
        _pose_phase(
            phase_id="grasp_close",
            position_world_m=waypoints[0]["position_world_m"],
            gripper_state="closed",
            minimum_steps=GRIPPER_DWELL_MINIMUM_STEPS,
            maximum_steps=GRIPPER_DWELL_MAXIMUM_STEPS,
        ),
        *[
            _pose_phase(
                phase_id=f"sweep_{int(row['waypoint_index']):02d}",
                position_world_m=row["position_world_m"],
                gripper_state="closed",
            )
            for row in waypoints[1:]
        ],
        _pose_phase(
            phase_id="release",
            position_world_m=waypoints[-1]["position_world_m"],
            gripper_state="open",
            minimum_steps=GRIPPER_DWELL_MINIMUM_STEPS,
            maximum_steps=GRIPPER_DWELL_MAXIMUM_STEPS,
        ),
        _pose_phase(
            phase_id="retreat",
            position_world_m=trajectory["retreat_pose"]["position_world_m"],
            gripper_state="open",
        ),
    ]
    maximum_steps = sum(int(row["maximum_steps"]) for row in actions) + int(
        scene["task_spec"]["settle_window_samples"]
    )
    if maximum_steps > int(scene["task_spec"]["maximum_action_steps"]):
        raise NativeArticulatedControlPlanError(
            ["native_articulated_control_action_budget_exceeded"]
        )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cell_id": scene["scenario"]["cell_id"],
        "task_spec_digest": canonical_digest(scene["task_spec"]),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": construction["result_digest"],
        "zero_action_steps": ZERO_ACTION_STEPS,
        "scripted_positive_actions": actions,
        "maximum_scripted_and_settle_steps": maximum_steps,
        "construction_scene_plan_digest": scene["plan_digest"],
        "construction_clearance_plan_digest": construction[
            "construction_phase_plan"
        ]["plan_digest"],
        "candidate_policy_queried": False,
        "plan_digest": "",
    }
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    return result


__all__ = [
    "NativeArticulatedControlPlanError",
    "SCHEMA_VERSION",
    "materialize_native_articulated_control_plan",
]
