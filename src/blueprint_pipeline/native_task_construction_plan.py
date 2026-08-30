"""Compile task-neutral native construction phases before paid execution.

Articulated construction retains its existing collision-clear sweep compiler.
Rigid construction uses the frozen start/destination/support contract to bind
the complete pregrasp through reset-readback gate sequence.  Neither compiler
claims that a planned pose was reached or that native contact occurred.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_franka_action_math import is_unauthored_identity_quaternion_xyzw
from .native_franka_pose_servo import DEFAULT_VELOCITY_FEEDFORWARD_SCALE
from .articulation_graph_contract import (
    ArticulationGraphContractError,
    validate_articulation_graph,
)
from .native_articulated_construction_plan import (
    materialize_articulated_construction_phase_plan,
)


SCHEMA_VERSION = "native_task_construction_phase_plan.v1"
RIGID_SCHEMA_VERSION = "native_rigid_construction_phase_plan.v1"
SUPPORTED_TASK_KINDS = frozenset({"articulated_open_close", "rigid_pick_place"})
RIGID_AFFORDANCE_SCHEMA_VERSION = "native_rigid_interaction_affordance.v1"
RIGID_MANIPULATION_STRATEGIES = frozenset({"pick_and_place", "planar_push"})
GRAPH_ARTICULATED_SCHEMA_VERSION = "native_articulated_graph_construction_phase_plan.v1"
GRAPH_ARTICULATED_AFFORDANCE_SCHEMA_VERSION = "native_articulated_graph_interaction_affordance.v1"
# Single source of truth for the two bounds that actually reach the native
# joint-position command.  ``max_joint_delta_rad`` limits how far the commanded
# setpoint may move in one control step; ``max_joint_setpoint_lead_rad``
# independently limits how far the command may lead the *measured* joint state,
# which is what caps the achievable position error and therefore the achievable
# actuator torque.  Construction and controls must execute the same pair or the
# controls lane replays a different dynamical system than the one that
# qualified, so both control-plan compilers import these names from here.
#
#: Per-step slew of the commanded setpoint. At 20 Hz control, 0.10 rad/step is
#: 2 rad/s -- inside a Panda's ~2.6 rad/s joint limit. The previous 0.03 was
#: 0.6 rad/s, roughly a quarter of the hardware's capability.
MAX_JOINT_DELTA_RAD = 0.10
#: How far the commanded setpoint may lead the MEASURED position.
#:
#: This is the throttle, not the slew above. The implicit actuator is a PD:
#: its velocity comes from the position error it is shown, so capping the lead
#: caps the achievable speed. At 0.20 rad the arm moved 0.0038 rad/joint/step
#: -- an eighth of what the 0.03 slew already permitted, so the slew never
#: bound. Measured in r17:
#:
#:   total_action_steps        400  (the full budget, exhausted)
#:   joint travel achieved  10.717 rad
#:   travel needed, phase 1  8.721 rad   (of nine phases)
#:
#: Every phase reported native_task_phase_ik_unreached. The IK solution was
#: correct on the first step and 1.88 rad away; the arm simply could not be
#: driven there in the time allowed. Isaac Lab's own IK examples apply no lead
#: cap at all -- they command the solution and let the actuator's effort and
#: velocity limits govern. This keeps a bound for stability, but one that does
#: not sit below the robot's own limits.
MAX_JOINT_SETPOINT_LEAD_RAD = 1.00
#: Fraction of the commanded setpoint advance rate declared as a joint velocity
#: target.
#:
#: The implicit actuator's torque is
#: ``stiffness * (pos_target - pos) + damping * (vel_target - vel)``.  A
#: position-only command leaves ``vel_target`` at zero, so the damping term
#: brakes in proportion to the very motion we asked for and the joint settles
#: at ``(stiffness / damping) * error`` -- 5 rad/s per rad of lag on this arm,
#: reached while using two to three percent of the available torque.  Declaring
#: the intended velocity cancels that braking while tracking and still damps
#: the joint at rest, which is why this is preferred over lowering damping:
#: the damping ratio is unchanged, and this task ends in contact with a hinged
#: door.  1.0 is exact feedforward; 0.0 restores position-only commanding.
VELOCITY_FEEDFORWARD_SCALE = DEFAULT_VELOCITY_FEEDFORWARD_SCALE
# Align the wrist while still well clear of the appliance, then perform the
# existing short Cartesian approach. MoveIt and Isaac's own Franka examples
# separate these operations; commanding a 120+ degree reorientation only at
# the final 0.12 m standoff produced real washer contact in Arena r30.
GRAPH_ARTICULATED_PREALIGN_CLEARANCE_M = 0.30
# The outward standoff must oppose the tool's +Z approach axis.  A perpendicular
# standoff was accepted previously, which made the gripper translate along its
# jaw-closing axis and strike the door before the panel entered between the
# fingers.  Keep the tolerance numeric rather than trusting a source label.
GRAPH_ARTICULATED_STANDOFF_ALIGNMENT_MIN = 1.0 - 1.0e-6


class NativeTaskConstructionPlanError(ValueError):
    """Stable pre-native failures for task-neutral construction planning."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _finite_vector(value: Any, *, length: int, error: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise NativeTaskConstructionPlanError([error]) from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise NativeTaskConstructionPlanError([error])
    return result


def _positive(value: Any, *, error: str, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NativeTaskConstructionPlanError([error]) from exc
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        raise NativeTaskConstructionPlanError([error])
    return result


def joint_command_limits(
    *,
    max_joint_delta_rad: Any,
    max_joint_setpoint_lead_rad: Any,
    error: str,
    velocity_feedforward_scale: Any = VELOCITY_FEEDFORWARD_SCALE,
) -> dict[str, float]:
    """Validate the joint-command bound pair the native servo will execute.

    ``bounded_absolute_joint_setpoint`` rejects a lead smaller than the slew at
    runtime, i.e. mid paid run.  Reject it here, while the plan is still being
    compiled off-GPU, so an unexecutable pair can never reach the simulator.
    """

    delta = _positive(max_joint_delta_rad, error=error)
    lead = _positive(max_joint_setpoint_lead_rad, error=error)
    if lead < delta:
        raise NativeTaskConstructionPlanError([error])
    feedforward = _positive(velocity_feedforward_scale, error=error, allow_zero=True)
    if feedforward > 1.0:
        raise NativeTaskConstructionPlanError([error])
    return {
        "max_joint_delta_rad": delta,
        "max_joint_setpoint_lead_rad": lead,
        "velocity_feedforward_scale": feedforward,
    }


def _unit(value: Any, *, error: str) -> list[float]:
    result = _finite_vector(value, length=3, error=error)
    norm = math.sqrt(sum(item * item for item in result))
    if abs(norm - 1.0) > 1.0e-6:
        raise NativeTaskConstructionPlanError([error])
    return result


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _quaternion(value: Any, *, error: str) -> list[float]:
    result = _finite_vector(value, length=4, error=error)
    if abs(sum(item * item for item in result) - 1.0) > 1.0e-6:
        raise NativeTaskConstructionPlanError([error])
    return result


def _quaternion_product_xyzw(a: Sequence[float], b: Sequence[float]) -> list[float]:
    ax, ay, az, aw = (float(item) for item in a)
    bx, by, bz, bw = (float(item) for item in b)
    result = [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
    norm = math.sqrt(sum(item * item for item in result))
    return [item / norm for item in result]


def _quaternion_rotate_xyzw(
    quaternion: Sequence[float], vector: Sequence[float]
) -> list[float]:
    qx, qy, qz, qw = (float(item) for item in quaternion)
    vx, vy, vz = (float(item) for item in vector)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


def _quaternion_angle_xyzw(a: Sequence[float], b: Sequence[float]) -> float:
    dot = abs(sum(float(x) * float(y) for x, y in zip(a, b, strict=True)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _slerp_xyzw(a: Sequence[float], b: Sequence[float], fraction: float) -> list[float]:
    start = [float(item) for item in a]
    end = [float(item) for item in b]
    dot = sum(x * y for x, y in zip(start, end, strict=True))
    if dot < 0.0:
        end = [-item for item in end]
        dot = -dot
    if dot > 0.9995:
        result = [
            x + fraction * (y - x) for x, y in zip(start, end, strict=True)
        ]
        norm = math.sqrt(sum(item * item for item in result))
        return [item / norm for item in result]
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sine = math.sin(theta)
    left = math.sin((1.0 - fraction) * theta) / sine
    right = math.sin(fraction * theta) / sine
    return [
        left * x + right * y for x, y in zip(start, end, strict=True)
    ]


def _compose_pose(
    parent_position: Sequence[float],
    parent_orientation: Sequence[float],
    child_position: Sequence[float],
    child_orientation: Sequence[float],
) -> list[float]:
    offset = _quaternion_rotate_xyzw(parent_orientation, child_position)
    return [
        *[
            float(parent_position[index]) + offset[index]
            for index in range(3)
        ],
        *_quaternion_product_xyzw(parent_orientation, child_orientation),
    ]


def _affordance(task_spec: Mapping[str, Any], *, subject_asset_id: str) -> dict[str, Any]:
    raw = task_spec.get("interaction_affordance")
    if not isinstance(raw, Mapping):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_interaction_affordance_missing"]
        )
    value = json.loads(json.dumps(raw))
    if (
        value.get("schema_version") != RIGID_AFFORDANCE_SCHEMA_VERSION
        or value.get("affordance_digest")
        != canonical_digest(value, digest_field="affordance_digest")
    ):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_interaction_affordance_invalid"]
        )
    if (
        value.get("subject_asset_id") != subject_asset_id
        or value.get("scoring_frame_id") != "task_scoring_frame"
    ):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_interaction_affordance_binding_mismatch"]
        )
    transform = value.get("asset_root_from_scoring_frame")
    if not isinstance(transform, Mapping):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_scoring_frame_transform_invalid"]
        )
    value["asset_root_from_scoring_frame"] = {
        "position_m": _finite_vector(
            transform.get("position_m"),
            length=3,
            error="native_rigid_construction_scoring_frame_transform_invalid",
        ),
        "orientation_xyzw": _quaternion(
            transform.get("orientation_xyzw"),
            error="native_rigid_construction_scoring_frame_transform_invalid",
        ),
    }
    value["contact_point_scoring_frame_m"] = _finite_vector(
        value.get("contact_point_scoring_frame_m"),
        length=3,
        error="native_rigid_construction_contact_point_invalid",
    )
    value["approach_unit_scoring_frame"] = _unit(
        value.get("approach_unit_scoring_frame"),
        error="native_rigid_construction_approach_direction_invalid",
    )
    value["lift_unit_world"] = _unit(
        value.get("lift_unit_world"),
        error="native_rigid_construction_lift_direction_invalid",
    )
    orientation = value["gripper_orientation_scoring_frame_xyzw"] = _quaternion(
        value.get("gripper_orientation_scoring_frame_xyzw"),
        error="native_rigid_construction_gripper_orientation_invalid",
    )
    if is_unauthored_identity_quaternion_xyzw(orientation):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_gripper_orientation_unauthored"]
        )
    value["pregrasp_clearance_m"] = _positive(
        value.get("pregrasp_clearance_m"),
        error="native_rigid_construction_pregrasp_clearance_invalid",
    )
    value["arrival_orientation_tolerance_rad"] = _positive(
        value.get("arrival_orientation_tolerance_rad"),
        error="native_rigid_construction_arrival_orientation_tolerance_invalid",
    )
    for field, error in (
        ("allowed_contact_prim_paths", "native_rigid_construction_contact_region_invalid"),
        ("intended_support_prim_paths", "native_rigid_construction_support_region_invalid"),
    ):
        paths = value.get(field)
        if (
            not isinstance(paths, list)
            or not paths
            or any(
                not isinstance(path, str)
                or not path.startswith("/")
                or ".." in path.split("/")
                for path in paths
            )
            or len(set(paths)) != len(paths)
        ):
            raise NativeTaskConstructionPlanError([error])
        value[field] = list(paths)
    return value


def _subject(plan: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in plan.get("objects") or []
        if isinstance(row, Mapping) and row.get("task_subject") is True
    ]
    if len(rows) != 1:
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_task_subject_invalid"]
        )
    return rows[0]


def _destination(task_spec: Mapping[str, Any]) -> tuple[list[float], dict[str, list[float]]]:
    raw_bounds = task_spec.get("destination_position_bounds_world_m")
    if isinstance(raw_bounds, Mapping):
        lower = _finite_vector(
            raw_bounds.get("minimum"),
            length=3,
            error="native_rigid_construction_destination_invalid",
        )
        upper = _finite_vector(
            raw_bounds.get("maximum"),
            length=3,
            error="native_rigid_construction_destination_invalid",
        )
        if any(low >= high for low, high in zip(lower, upper, strict=True)):
            raise NativeTaskConstructionPlanError(
                ["native_rigid_construction_destination_invalid"]
            )
        return (
            [(low + high) / 2.0 for low, high in zip(lower, upper, strict=True)],
            {"minimum": lower, "maximum": upper},
        )
    target = task_spec.get("target_position_world_m")
    if target is None:
        target = task_spec.get("target_position_m")
    center = _finite_vector(
        target, length=3, error="native_rigid_construction_destination_missing"
    )
    tolerance = _positive(
        task_spec.get("destination_position_tolerance_m", 0.05),
        error="native_rigid_construction_destination_tolerance_invalid",
    )
    return center, {
        "minimum": [value - tolerance for value in center],
        "maximum": [value + tolerance for value in center],
    }


def _phase(
    phase_id: str,
    position: Sequence[float],
    *,
    gripper_state: str,
    gate_ids: Sequence[str],
    orientation_world_xyzw: Sequence[float],
    arrival_orientation_tolerance_rad: float | None = None,
    position_only_arrival: bool = False,
    expected_scoring_position_world_m: Sequence[float] | None = None,
    expected_scoring_orientation_world_xyzw: Sequence[float] | None = None,
) -> dict[str, Any]:
    result = {
        "phase_id": phase_id,
        "position_world_m": [float(value) for value in position],
        "orientation_world_xyzw": [float(value) for value in orientation_world_xyzw],
        "gripper_state": gripper_state,
        "gate_ids": list(gate_ids),
    }
    if expected_scoring_position_world_m is not None:
        result["expected_scoring_position_world_m"] = [
            float(value) for value in expected_scoring_position_world_m
        ]
    if expected_scoring_orientation_world_xyzw is not None:
        result["expected_scoring_orientation_world_xyzw"] = [
            float(value) for value in expected_scoring_orientation_world_xyzw
        ]
    if position_only_arrival:
        result["arrival_orientation_tolerance_rad"] = None
        result["position_only_arrival"] = True
    elif arrival_orientation_tolerance_rad is not None:
        result["arrival_orientation_tolerance_rad"] = float(
            arrival_orientation_tolerance_rad
        )
    return result


def _graph_articulated_subject(plan: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in plan.get("objects") or []
        if isinstance(row, Mapping) and row.get("task_subject") is True
    ]
    if len(rows) != 1 or rows[0].get("object_type") != "ARTICULATION":
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_task_subject_invalid"]
        )
    return rows[0]


def _graph_articulated_affordance(
    *,
    task_spec: Mapping[str, Any],
    graph: Mapping[str, Any],
    source_graph_digest: str,
    subject_asset_id: str,
    movement_epsilon: float,
) -> dict[str, Any]:
    raw = task_spec.get("interaction_affordance")
    if not isinstance(raw, Mapping):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_construction_general_interaction_affordance_missing"]
        )
    try:
        value = json.loads(json.dumps(dict(raw), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_interaction_affordance_invalid"]
        ) from exc
    if (
        value.get("schema_version")
        != GRAPH_ARTICULATED_AFFORDANCE_SCHEMA_VERSION
        or value.get("affordance_digest")
        != canonical_digest(value, digest_field="affordance_digest")
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_interaction_affordance_invalid"]
        )
    if (
        value.get("subject_asset_id") != subject_asset_id
        or value.get("articulation_graph_digest") != source_graph_digest
        or task_spec.get("articulation_graph_digest") != source_graph_digest
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_affordance_binding_mismatch"]
        )
    if not _digest(value.get("kinematic_path_receipt_digest")):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_kinematic_receipt_invalid"]
        )

    links = {str(row["link_id"]): dict(row) for row in graph["links"]}
    joints = {str(row["joint_id"]): dict(row) for row in graph["joints"]}
    child_joint = {str(row["child_link_id"]): dict(row) for row in graph["joints"]}
    contact_link_id = str(value.get("contact_link_id") or "")
    if contact_link_id not in links:
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_contact_link_invalid"]
        )
    cursor = contact_link_id
    driven_by_target = False
    while cursor in child_joint:
        joint = child_joint[cursor]
        driven_by_target = driven_by_target or joint["role"] == "target"
        cursor = str(joint["parent_link_id"])
    if not driven_by_target:
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_contact_link_not_target_driven"]
        )

    paths = value.get("contact_body_prim_paths")
    if (
        not isinstance(paths, list)
        or not paths
        or len(set(paths)) != len(paths)
        or any(
            not isinstance(path, str)
            or not path.startswith("/")
            or ".." in path.split("/")
            for path in paths
        )
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_contact_region_invalid"]
        )
    value["contact_body_prim_paths"] = list(paths)
    value["contact_point_link_m"] = _finite_vector(
        value.get("contact_point_link_m"),
        length=3,
        error="native_articulated_graph_construction_contact_point_invalid",
    )
    value["approach_unit_asset_root"] = _unit(
        value.get("approach_unit_asset_root"),
        error="native_articulated_graph_construction_approach_direction_invalid",
    )
    value["retreat_unit_asset_root"] = _unit(
        value.get("retreat_unit_asset_root"),
        error="native_articulated_graph_construction_retreat_direction_invalid",
    )
    value["gripper_orientation_contact_xyzw"] = _quaternion(
        value.get("gripper_orientation_contact_xyzw"),
        error="native_articulated_graph_construction_gripper_orientation_invalid",
    )
    try:
        contact_standoff = float(value.get("contact_outward_standoff_m", 0.0))
        lateral_tcp_offset = float(
            value.get("contact_lateral_tcp_surface_offset_m", 0.0)
        )
    except (TypeError, ValueError) as exc:
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_contact_standoff_invalid"]
        ) from exc
    standoff_digest = value.get("grasp_swept_volume_receipt_digest")
    if (
        not math.isfinite(contact_standoff)
        or contact_standoff < 0.0
        or (
            contact_standoff > 0.0
            and (
                not isinstance(standoff_digest, str)
                or len(standoff_digest) != 71
                or not standoff_digest.startswith("sha256:")
            )
        )
        or (contact_standoff == 0.0 and standoff_digest not in (None, ""))
        or not math.isfinite(lateral_tcp_offset)
        or lateral_tcp_offset < 0.0
        or (contact_standoff > 0.0 and lateral_tcp_offset <= 0.0)
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_contact_standoff_invalid"]
        )
    if "contact_outward_standoff_m" in value:
        value["contact_outward_standoff_m"] = contact_standoff
    if lateral_tcp_offset > 0.0:
        value["contact_lateral_tcp_surface_offset_m"] = lateral_tcp_offset
    for field in (
        "precontact_clearance_m",
        "sweep_clearance_m",
        "retreat_clearance_m",
        "arrival_tolerance_m",
        "arrival_orientation_tolerance_rad",
        "max_joint_delta_rad",
        "max_joint_setpoint_lead_rad",
    ):
        value[field] = _positive(
            value.get(field),
            error=f"native_articulated_graph_construction_{field}_invalid",
        )
    for field in (
        "motion_minimum_steps",
        "motion_maximum_steps",
        "gripper_dwell_minimum_steps",
        "gripper_dwell_maximum_steps",
        "arrival_stability_steps",
    ):
        raw_integer = value.get(field)
        if (
            isinstance(raw_integer, bool)
            or not isinstance(raw_integer, int)
            or raw_integer <= 0
        ):
            raise NativeTaskConstructionPlanError(
                [f"native_articulated_graph_construction_{field}_invalid"]
            )
    if (
        value["motion_minimum_steps"] > value["motion_maximum_steps"]
        or value["gripper_dwell_minimum_steps"]
        > value["gripper_dwell_maximum_steps"]
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_control_step_range_invalid"]
        )

    raw_waypoints = value.get("joint_contact_path")
    if not isinstance(raw_waypoints, list) or len(raw_waypoints) < 2:
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_joint_contact_path_invalid"]
        )
    joint_ids = set(joints)
    target_ids = sorted(
        joint_id for joint_id, joint in joints.items() if joint["role"] == "target"
    )
    normalized_waypoints: list[dict[str, Any]] = []
    waypoint_ids: list[str] = []
    for index, raw_waypoint in enumerate(raw_waypoints):
        if not isinstance(raw_waypoint, Mapping):
            raise NativeTaskConstructionPlanError(
                ["native_articulated_graph_construction_joint_contact_path_invalid"]
            )
        waypoint_id = str(raw_waypoint.get("waypoint_id") or "")
        waypoint_ids.append(waypoint_id)
        raw_positions = raw_waypoint.get("joint_positions")
        if not isinstance(raw_positions, Mapping) or set(raw_positions) != joint_ids:
            raise NativeTaskConstructionPlanError(
                [
                    "native_articulated_graph_construction_joint_path_set_invalid:"
                    f"{waypoint_id or index}"
                ]
            )
        positions: dict[str, float] = {}
        for joint_id in sorted(joint_ids):
            try:
                position = float(raw_positions[joint_id])
            except (TypeError, ValueError) as exc:
                raise NativeTaskConstructionPlanError(
                    [
                        "native_articulated_graph_construction_joint_path_value_invalid:"
                        f"{waypoint_id or index}:{joint_id}"
                    ]
                ) from exc
            lower, upper = joints[joint_id]["limits"]
            if not math.isfinite(position) or not float(lower) <= position <= float(upper):
                raise NativeTaskConstructionPlanError(
                    [
                        "native_articulated_graph_construction_joint_path_value_invalid:"
                        f"{waypoint_id or index}:{joint_id}"
                    ]
                )
            positions[joint_id] = position
        for joint_id, joint in joints.items():
            if joint["role"] == "locked" and abs(
                positions[joint_id] - float(joint["reset_position"])
            ) > float(joint["reset_tolerance"]):
                raise NativeTaskConstructionPlanError(
                    [
                        "native_articulated_graph_construction_locked_joint_path_invalid:"
                        f"{waypoint_id or index}:{joint_id}"
                    ]
                )
            dependency = joint.get("dependency")
            if joint["role"] == "dependent" and isinstance(dependency, Mapping):
                expected = (
                    positions[str(dependency["driver_joint_id"])]
                    * float(dependency["multiplier"])
                    + float(dependency["offset"])
                )
                if abs(positions[joint_id] - expected) > float(
                    dependency["tolerance"]
                ):
                    raise NativeTaskConstructionPlanError(
                        [
                            "native_articulated_graph_construction_dependent_joint_path_invalid:"
                            f"{waypoint_id or index}:{joint_id}"
                        ]
                    )
        contact_pose = raw_waypoint.get("contact_pose_asset_root")
        if not isinstance(contact_pose, Mapping):
            raise NativeTaskConstructionPlanError(
                [
                    "native_articulated_graph_construction_contact_pose_invalid:"
                    f"{waypoint_id or index}"
                ]
            )
        normalized_waypoints.append(
            {
                "waypoint_id": waypoint_id,
                "joint_positions": positions,
                "contact_pose_asset_root": {
                    "position_m": _finite_vector(
                        contact_pose.get("position_m"),
                        length=3,
                        error=(
                            "native_articulated_graph_construction_contact_pose_invalid:"
                            f"{waypoint_id or index}"
                        ),
                    ),
                    "orientation_xyzw": _quaternion(
                        contact_pose.get("orientation_xyzw"),
                        error=(
                            "native_articulated_graph_construction_contact_pose_invalid:"
                            f"{waypoint_id or index}"
                        ),
                    ),
                },
                "clearance_unit_asset_root": _unit(
                    raw_waypoint.get("clearance_unit_asset_root"),
                    error=(
                        "native_articulated_graph_construction_clearance_direction_invalid:"
                        f"{waypoint_id or index}"
                    ),
                ),
                **(
                    {
                        "lateral_outward_unit_asset_root": _unit(
                            raw_waypoint.get(
                                "lateral_outward_unit_asset_root"
                            ),
                            error=(
                                "native_articulated_graph_construction_"
                                "contact_lateral_tcp_offset_invalid:"
                                f"{waypoint_id or index}"
                            ),
                        )
                    }
                    if lateral_tcp_offset > 0.0
                    else {}
                ),
            }
        )
    if any(not waypoint_id for waypoint_id in waypoint_ids) or len(
        set(waypoint_ids)
    ) != len(waypoint_ids):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_waypoint_ids_invalid"]
        )

    first_positions = normalized_waypoints[0]["joint_positions"]
    if any(
        abs(first_positions[joint_id] - float(joint["reset_position"]))
        > float(joint["reset_tolerance"])
        for joint_id, joint in joints.items()
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_path_reset_mismatch"]
        )
    final_positions = normalized_waypoints[-1]["joint_positions"]
    intervals = graph["success_predicate"]["joint_intervals"]
    if any(
        not float(intervals[joint_id][0])
        <= final_positions[joint_id]
        <= float(intervals[joint_id][1])
        for joint_id in target_ids
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_path_target_mismatch"]
        )
    if any(
        max(
            abs(row["joint_positions"][joint_id] - first_positions[joint_id])
            for row in normalized_waypoints
        )
        <= movement_epsilon
        for joint_id in target_ids
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_target_path_static"]
        )
    value["joint_contact_path"] = normalized_waypoints
    return value


def materialize_graph_articulated_construction_phase_plan(
    scene_plan: Mapping[str, Any],
    *,
    arrival_tolerance_m: float = 0.02,
    stable_samples: int = 2,
    maximum_steps_per_phase: int = 64,
    prealign_clearance_m: float = GRAPH_ARTICULATED_PREALIGN_CLEARANCE_M,
) -> dict[str, Any]:
    """Compile a complete graph-articulated clearance and contact program."""

    try:
        plan = json.loads(json.dumps(dict(scene_plan), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_scene_plan_invalid"]
        ) from exc
    if plan.get("task_kind") != "articulated_open_close":
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_task_kind_invalid"]
        )
    task_spec = plan.get("task_spec")
    if not isinstance(task_spec, Mapping):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_task_spec_invalid"]
        )
    source_graph = task_spec.get("articulation_graph")
    if not isinstance(source_graph, Mapping):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_task_spec_invalid"]
        )
    source_graph_digest = canonical_digest(dict(source_graph))
    try:
        graph = validate_articulation_graph(source_graph)
    except ArticulationGraphContractError as exc:
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_task_spec_invalid", *exc.errors]
        ) from exc
    movement_epsilon = _positive(
        task_spec.get("movement_epsilon"),
        error="native_articulated_graph_construction_movement_epsilon_invalid",
    )
    joints = list(graph["joints"])
    normalized_spec = {
        "joint_reset_positions": {
            str(row["joint_id"]): float(row["reset_position"]) for row in joints
        },
        "joint_reset_tolerances": {
            str(row["joint_id"]): float(row["reset_tolerance"]) for row in joints
        },
        "joint_roles": {
            str(row["joint_id"]): str(row["role"]) for row in joints
        },
        "target_success_intervals": {
            str(joint_id): list(interval)
            for joint_id, interval in graph["success_predicate"][
                "joint_intervals"
            ].items()
        },
    }
    subject = _graph_articulated_subject(plan)
    affordance = _graph_articulated_affordance(
        task_spec=task_spec,
        graph=graph,
        source_graph_digest=source_graph_digest,
        subject_asset_id=str(subject.get("asset_id") or ""),
        movement_epsilon=movement_epsilon,
    )
    reset_pose = (subject.get("reset_state") or {}).get("root_pose_world") or subject.get(
        "pose_world"
    )
    if not isinstance(reset_pose, Mapping):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_asset_root_pose_invalid"]
        )
    root_position = _finite_vector(
        reset_pose.get("position_world_m"),
        length=3,
        error="native_articulated_graph_construction_asset_root_pose_invalid",
    )
    root_orientation = _quaternion(
        reset_pose.get("orientation_xyzw"),
        error="native_articulated_graph_construction_asset_root_pose_invalid",
    )
    subject_resets = (subject.get("reset_state") or {}).get("joint_positions")
    graph_resets = normalized_spec["joint_reset_positions"]
    if not isinstance(subject_resets, Mapping) or set(subject_resets) != set(
        graph_resets
    ) or any(
        abs(float(subject_resets[joint_id]) - float(reset))
        > float(normalized_spec["joint_reset_tolerances"][joint_id])
        for joint_id, reset in graph_resets.items()
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_subject_reset_mismatch"]
        )
    arrival_tolerance = _positive(
        arrival_tolerance_m,
        error="native_articulated_graph_construction_arrival_tolerance_invalid",
    )
    if (
        isinstance(stable_samples, bool)
        or not isinstance(stable_samples, int)
        or stable_samples <= 0
        or isinstance(maximum_steps_per_phase, bool)
        or not isinstance(maximum_steps_per_phase, int)
        or maximum_steps_per_phase <= 0
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_execution_parameters_invalid"]
        )

    world_rows: list[dict[str, Any]] = []
    for row in affordance["joint_contact_path"]:
        pose = row["contact_pose_asset_root"]
        world_pose = _compose_pose(
            root_position,
            root_orientation,
            pose["position_m"],
            pose["orientation_xyzw"],
        )
        world_rows.append(
            {
                **row,
                "contact_position_world_m": world_pose[:3],
                "contact_orientation_world_xyzw": world_pose[3:],
                "gripper_orientation_world_xyzw": _quaternion_product_xyzw(
                    world_pose[3:],
                    affordance["gripper_orientation_contact_xyzw"],
                ),
                "clearance_unit_world": _quaternion_rotate_xyzw(
                    root_orientation, row["clearance_unit_asset_root"]
                ),
            }
        )
    approach_unit_world = _quaternion_rotate_xyzw(
        root_orientation, affordance["approach_unit_asset_root"]
    )
    retreat_unit_world = _quaternion_rotate_xyzw(
        root_orientation, affordance["retreat_unit_asset_root"]
    )
    contact_standoff = float(affordance.get("contact_outward_standoff_m", 0.0))
    lateral_tcp_offset = float(
        affordance.get("contact_lateral_tcp_surface_offset_m", 0.0)
    )
    if contact_standoff > 0.0:
        for row in world_rows:
            surface_position = list(row["contact_position_world_m"])
            row["surface_contact_position_world_m"] = surface_position
            row["contact_position_world_m"] = [
                surface_position[axis]
                + row["clearance_unit_world"][axis] * contact_standoff
                for axis in range(3)
            ]
    if lateral_tcp_offset > 0.0:
        for row in world_rows:
            approach_standoff_position = list(
                row["contact_position_world_m"]
            )
            row["approach_standoff_contact_position_world_m"] = (
                approach_standoff_position
            )
            lateral_world = _quaternion_rotate_xyzw(
                root_orientation,
                row["lateral_outward_unit_asset_root"],
            )
            row["lateral_outward_unit_world"] = lateral_world
            row["contact_position_world_m"] = [
                approach_standoff_position[axis]
                + lateral_world[axis] * lateral_tcp_offset
                for axis in range(3)
            ]
    first = world_rows[0]
    last = world_rows[-1]
    authored_grasp = not is_unauthored_identity_quaternion_xyzw(
        affordance["gripper_orientation_contact_xyzw"]
    )
    misaligned_waypoints = []
    for row in world_rows:
        tool_approach_world = _quaternion_rotate_xyzw(
            row["gripper_orientation_world_xyzw"], [0.0, 0.0, 1.0]
        )
        outward_alignment = -sum(
            standoff * tool
            for standoff, tool in zip(
                row["clearance_unit_world"], tool_approach_world, strict=True
            )
        )
        if outward_alignment < GRAPH_ARTICULATED_STANDOFF_ALIGNMENT_MIN:
            misaligned_waypoints.append(str(row.get("waypoint_id") or ""))
    if authored_grasp and misaligned_waypoints:
        raise NativeTaskConstructionPlanError(
            [
                "native_articulated_graph_construction_"
                "standoff_not_opposite_gripper_approach:"
                + ",".join(misaligned_waypoints)
            ]
        )
    orientation_tolerance = float(
        affordance["arrival_orientation_tolerance_rad"]
    )
    approach_position = [
        first["contact_position_world_m"][axis]
        + approach_unit_world[axis] * float(affordance["precontact_clearance_m"])
        for axis in range(3)
    ]
    prealign_clearance = _positive(
        prealign_clearance_m,
        error="native_articulated_graph_construction_prealign_clearance_invalid",
    )
    if prealign_clearance <= float(affordance["precontact_clearance_m"]):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_prealign_clearance_invalid"]
        )
    prealign_position = [
        first["contact_position_world_m"][axis]
        + approach_unit_world[axis] * prealign_clearance
        for axis in range(3)
    ]
    retreat_position = [
        last["contact_position_world_m"][axis]
        + retreat_unit_world[axis] * float(affordance["retreat_clearance_m"])
        for axis in range(3)
    ]
    phases = [
        _phase(
            "prealign",
            prealign_position,
            gripper_state="open",
            gate_ids=("precontact_reachability", "base_collision_clearance"),
            orientation_world_xyzw=first["gripper_orientation_world_xyzw"],
            position_only_arrival=True,
        ),
        _phase(
            "approach",
            approach_position,
            gripper_state="open",
            gate_ids=("precontact_reachability", "base_collision_clearance"),
            orientation_world_xyzw=first["gripper_orientation_world_xyzw"],
            arrival_orientation_tolerance_rad=orientation_tolerance,
        )
    ]
    for index, row in enumerate(world_rows):
        phases.append(
            _phase(
                f"contact_sweep_clearance_{index:02d}",
                [
                    row["contact_position_world_m"][axis]
                    + row["clearance_unit_world"][axis]
                    * float(affordance["sweep_clearance_m"])
                    for axis in range(3)
                ],
                gripper_state="open",
                gate_ids=("sweep_workspace_clearance", "joint_limit_clearance"),
                orientation_world_xyzw=row["gripper_orientation_world_xyzw"],
                arrival_orientation_tolerance_rad=orientation_tolerance,
            )
        )
    phases.extend(
        [
            _phase(
                "release_clearance",
                [
                    last["contact_position_world_m"][axis]
                    + last["clearance_unit_world"][axis]
                    * float(affordance["sweep_clearance_m"])
                    for axis in range(3)
                ],
                gripper_state="open",
                gate_ids=("release_clearance",),
                orientation_world_xyzw=last["gripper_orientation_world_xyzw"],
                arrival_orientation_tolerance_rad=orientation_tolerance,
            ),
            _phase(
                "retreat",
                retreat_position,
                gripper_state="open",
                gate_ids=("retreat",),
                orientation_world_xyzw=last["gripper_orientation_world_xyzw"],
                arrival_orientation_tolerance_rad=orientation_tolerance,
            ),
            _phase(
                "recovery",
                approach_position,
                gripper_state="open",
                gate_ids=("recovery", "reset_readback"),
                orientation_world_xyzw=first["gripper_orientation_world_xyzw"],
                arrival_orientation_tolerance_rad=orientation_tolerance,
            ),
        ]
    )
    exact_contact_phases = [
        _phase(
            "prealign",
            prealign_position,
            gripper_state="open",
            gate_ids=("precontact_reachability", "base_collision_clearance"),
            orientation_world_xyzw=first["gripper_orientation_world_xyzw"],
            position_only_arrival=True,
        ),
        _phase(
            "approach",
            approach_position,
            gripper_state="open",
            gate_ids=("precontact_reachability",),
            orientation_world_xyzw=first["gripper_orientation_world_xyzw"],
            arrival_orientation_tolerance_rad=orientation_tolerance,
        ),
        _phase(
            "contact_open",
            first["contact_position_world_m"],
            gripper_state="open",
            gate_ids=("exact_contact_region",),
            orientation_world_xyzw=first["gripper_orientation_world_xyzw"],
            arrival_orientation_tolerance_rad=orientation_tolerance,
        ),
        _phase(
            "contact_close",
            first["contact_position_world_m"],
            gripper_state="closed",
            gate_ids=("task_robot_contact",),
            orientation_world_xyzw=first["gripper_orientation_world_xyzw"],
            arrival_orientation_tolerance_rad=orientation_tolerance,
        ),
    ]
    for index, row in enumerate(world_rows[1:], start=1):
        exact_contact_phases.append(
            {
                **_phase(
                    f"joint_path_{index:02d}",
                    row["contact_position_world_m"],
                    gripper_state="closed",
                    gate_ids=(
                        "task_robot_contact",
                        "target_joint_path",
                        "dependent_joint_path",
                        "locked_joint_containment",
                        "collision_clearance",
                    ),
                    orientation_world_xyzw=row["gripper_orientation_world_xyzw"],
                    arrival_orientation_tolerance_rad=orientation_tolerance,
                ),
                "expected_joint_positions": row["joint_positions"],
                "source_waypoint_id": row["waypoint_id"],
            }
        )
    exact_contact_phases.extend(
        [
            {
                **_phase(
                    "release",
                    last["contact_position_world_m"],
                    gripper_state="open",
                    gate_ids=("release", "target_joint_settle"),
                    orientation_world_xyzw=last["gripper_orientation_world_xyzw"],
                    arrival_orientation_tolerance_rad=orientation_tolerance,
                ),
                "expected_joint_positions": last["joint_positions"],
            },
            _phase(
                "retreat",
                retreat_position,
                gripper_state="open",
                gate_ids=("retreat", "target_joint_settle"),
                orientation_world_xyzw=last["gripper_orientation_world_xyzw"],
                arrival_orientation_tolerance_rad=orientation_tolerance,
            ),
        ]
    )
    joint_roles = normalized_spec["joint_roles"]
    gate_contract = {
        "precontact_reachability": "native_end_effector_pose_readback",
        "base_collision_clearance": "native_robot_scene_contact_readback",
        "sweep_workspace_clearance": "native_end_effector_pose_and_contact_readback",
        "joint_limit_clearance": "digest_bound_complete_joint_path",
        "release_clearance": "native_end_effector_pose_and_contact_readback",
        "retreat": "native_end_effector_pose_readback",
        "recovery": "native_end_effector_pose_readback",
        "reset_readback": "native_robot_and_complete_joint_graph_reset_replay",
    }
    # The qualifying controls episode replays every qualified phase at its
    # exact step count and appends the settle window inside the same
    # ``maximum_action_steps`` cap
    # (``native_articulated_graph_control_action_budget_exceeded``), so the
    # construction budget must reserve that window or a paid construction can
    # qualify and still be unreplayable.
    _articulated_maximum_action_steps = task_spec.get("maximum_action_steps")
    _articulated_settle_samples = task_spec.get("settle_window_samples")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (
            _articulated_maximum_action_steps,
            _articulated_settle_samples,
        )
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_action_budget_invalid"]
        )
    maximum_construction_total_steps = (
        _articulated_maximum_action_steps - _articulated_settle_samples
    )
    if maximum_construction_total_steps < len(phases) * stable_samples:
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_total_budget_infeasible"]
        )
    result: dict[str, Any] = {
        "schema_version": GRAPH_ARTICULATED_SCHEMA_VERSION,
        "task_kind": "articulated_open_close",
        "scene_plan_digest": plan.get("plan_digest"),
        "subject_asset_id": subject.get("asset_id"),
        "articulation_graph": graph,
        "articulation_graph_digest": source_graph_digest,
        "normalized_articulation_graph_digest": canonical_digest(graph),
        "interaction_affordance": affordance,
        "asset_root_reset_pose_world": [*root_position, *root_orientation],
        "joint_ids_by_role": {
            role: sorted(
                joint_id
                for joint_id, observed_role in joint_roles.items()
                if observed_role == role
            )
            for role in ("target", "dependent", "passive", "locked")
        },
        "target_success_intervals": normalized_spec["target_success_intervals"],
        "joint_reset_positions": normalized_spec["joint_reset_positions"],
        "joint_reset_tolerances": normalized_spec["joint_reset_tolerances"],
        "joint_contact_path": world_rows,
        "phases": phases,
        "phase_count": len(phases),
        "exact_contact_phases": exact_contact_phases,
        "prealign_clearance_m": prealign_clearance,
        "execution_parameters": {
            "arrival_tolerance_m": arrival_tolerance,
            "arrival_orientation_tolerance_rad": affordance[
                "arrival_orientation_tolerance_rad"
            ],
            "stable_samples": stable_samples,
            "maximum_steps_per_phase": maximum_steps_per_phase,
            "maximum_construction_total_steps": maximum_construction_total_steps,
            # The sealed affordance already carries the two bounds the servo
            # executes.  Publish them here so construction executes the pair the
            # task author sealed instead of the servo's own defaults.
            **joint_command_limits(
                max_joint_delta_rad=affordance["max_joint_delta_rad"],
                max_joint_setpoint_lead_rad=affordance[
                    "max_joint_setpoint_lead_rad"
                ],
                error=(
                    "native_articulated_graph_construction_"
                    "joint_command_limits_invalid"
                ),
            ),
        },
        "gate_contract": gate_contract,
        # An identity grasp orientation is an unauthored placeholder.  Clearance
        # phases run open-gripper and bind the measured reset orientation, so
        # construction still executes; the contact replay refuses instead.
        "grasp_orientation_authored": not is_unauthored_identity_quaternion_xyzw(
            affordance["gripper_orientation_contact_xyzw"]
        ),
        "required_gate_ids": sorted(gate_contract),
        "claim_boundary": {
            "wrist_alignment_occurs_before_final_appliance_approach": True,
            "clearance_phases_are_native_ik_targets": True,
            "exact_contact_phases_require_qualified_clearance_receipt": True,
            "kinematic_path_correctness_requires_bound_asset_qualification": True,
            "plan_is_not_construction_policy_or_physical_evidence": True,
        },
        "plan_digest": "",
    }
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    return result


def evaluate_graph_articulated_construction_gates(
    *,
    phase_plan: Mapping[str, Any],
    phase_results: Sequence[Mapping[str, Any]],
    reset_replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Qualify every clearance phase from retained native state samples."""

    if (
        phase_plan.get("schema_version") != GRAPH_ARTICULATED_SCHEMA_VERSION
        or phase_plan.get("plan_digest")
        != canonical_digest(dict(phase_plan), digest_field="plan_digest")
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_phase_plan_invalid"]
        )
    expected_ids = [str(row["phase_id"]) for row in phase_plan["phases"]]
    if (
        not isinstance(phase_results, Sequence)
        or isinstance(phase_results, (str, bytes))
        or len(phase_results) != len(expected_ids)
    ):
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_phase_results_invalid"]
        )
    observed = [dict(row) for row in phase_results if isinstance(row, Mapping)]
    if len(observed) != len(expected_ids) or [
        str(row.get("phase_id") or "") for row in observed
    ] != expected_ids:
        raise NativeTaskConstructionPlanError(
            ["native_articulated_graph_construction_phase_results_invalid"]
        )
    all_samples: list[dict[str, Any]] = []
    for row in observed:
        samples = row.get("task_samples")
        if not isinstance(samples, list) or not samples or any(
            not isinstance(sample, Mapping) for sample in samples
        ):
            raise NativeTaskConstructionPlanError(
                [
                    "native_articulated_graph_construction_path_readback_missing:"
                    f"{row['phase_id']}"
                ]
            )
        all_samples.extend(dict(sample) for sample in samples)
    graph = phase_plan["articulation_graph"]
    reset_positions = phase_plan["joint_reset_positions"]
    reset_tolerances = phase_plan["joint_reset_tolerances"]
    roles = {
        str(joint["joint_id"]): str(joint["role"]) for joint in graph["joints"]
    }
    expected_joint_ids = set(reset_positions)
    path_state_valid = True
    for sample in all_samples:
        positions = sample.get("joint_positions")
        if positions is None:
            positions = sample.get("joint_positions_rad")
        if not isinstance(positions, Mapping) or set(positions) != expected_joint_ids:
            raise NativeTaskConstructionPlanError(
                ["native_articulated_graph_construction_joint_readback_invalid"]
            )
        for joint_id in sorted(expected_joint_ids):
            try:
                position = float(positions[joint_id])
            except (TypeError, ValueError) as exc:
                raise NativeTaskConstructionPlanError(
                    ["native_articulated_graph_construction_joint_readback_invalid"]
                ) from exc
            if not math.isfinite(position):
                raise NativeTaskConstructionPlanError(
                    ["native_articulated_graph_construction_joint_readback_invalid"]
                )
            if roles[joint_id] != "passive" and abs(
                position - float(reset_positions[joint_id])
            ) > float(reset_tolerances[joint_id]):
                path_state_valid = False
    clearance_contact_free = all(
        sample.get("task_contact_active") is False for sample in all_samples
    )
    collision_clear = all(
        sample.get("robot_collision_failure") is False
        and sample.get("scene_collision_failure") is False
        and sample.get("containment_violation") is False
        and sample.get("joint_limit_violation") is False
        for sample in all_samples
    )
    reachability = all(row.get("target_reached") is True for row in observed)
    reset_passed = reset_replay.get("passed") is True
    gate_values = {
        "precontact_reachability": reachability,
        "base_collision_clearance": collision_clear,
        "sweep_workspace_clearance": (
            reachability and collision_clear and clearance_contact_free
        ),
        "joint_limit_clearance": path_state_valid,
        "release_clearance": collision_clear and clearance_contact_free,
        "retreat": reachability and collision_clear,
        "recovery": reachability and collision_clear,
        "reset_readback": reset_passed,
    }
    rows = [
        {
            "gate_id": gate_id,
            "measurement_authority": phase_plan["gate_contract"][gate_id],
            "passed": bool(gate_values[gate_id]),
        }
        for gate_id in phase_plan["required_gate_ids"]
    ]
    blockers = [
        f"native_articulated_graph_construction_gate_failed:{row['gate_id']}"
        for row in rows
        if not row["passed"]
    ]
    result = {
        "schema_version": "native_articulated_graph_construction_gate_evaluation.v1",
        "phase_plan_digest": phase_plan["plan_digest"],
        "all_phase_targets_reached": reachability,
        "all_phase_samples_retained": True,
        "gates": rows,
        "passed": not blockers,
        "blockers": sorted(blockers),
        "evaluation_digest": "",
    }
    result["evaluation_digest"] = canonical_digest(
        result, digest_field="evaluation_digest"
    )
    return result


def materialize_rigid_construction_phase_plan(
    scene_plan: Mapping[str, Any],
    *,
    waypoint_count: int = 4,
    arrival_tolerance_m: float = 0.02,
    stable_samples: int = 2,
    maximum_steps_per_phase: int = 64,
    max_joint_delta_rad: float = MAX_JOINT_DELTA_RAD,
    max_joint_setpoint_lead_rad: float = MAX_JOINT_SETPOINT_LEAD_RAD,
) -> dict[str, Any]:
    """Freeze one rigid pregrasp/contact/relocation/release construction gate."""

    try:
        plan = json.loads(json.dumps(dict(scene_plan), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_scene_plan_invalid"]
        ) from exc
    if plan.get("task_kind") != "rigid_pick_place":
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_task_kind_invalid"]
        )
    if isinstance(waypoint_count, bool) or not isinstance(waypoint_count, int) or waypoint_count < 2:
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_waypoint_count_invalid"]
        )
    arrival_tolerance = _positive(
        arrival_tolerance_m,
        error="native_rigid_construction_arrival_tolerance_invalid",
    )
    if (
        isinstance(stable_samples, bool)
        or not isinstance(stable_samples, int)
        or stable_samples <= 0
        or isinstance(maximum_steps_per_phase, bool)
        or not isinstance(maximum_steps_per_phase, int)
        or maximum_steps_per_phase <= 0
    ):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_execution_parameters_invalid"]
        )
    task_spec = plan.get("task_spec")
    if not isinstance(task_spec, Mapping):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_task_spec_invalid"]
        )
    subject = _subject(plan)
    reset_pose = (subject.get("reset_state") or {}).get("root_pose_world") or subject.get(
        "pose_world"
    )
    if not isinstance(reset_pose, Mapping):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_start_pose_invalid"]
        )
    root_position = _finite_vector(
        reset_pose.get("position_world_m"),
        length=3,
        error="native_rigid_construction_asset_root_pose_invalid",
    )
    root_orientation = _quaternion(
        reset_pose.get("orientation_xyzw"),
        error="native_rigid_construction_asset_root_pose_invalid",
    )
    affordance = _affordance(
        task_spec, subject_asset_id=str(subject.get("asset_id") or "")
    )
    scoring_transform = affordance["asset_root_from_scoring_frame"]
    computed_start_pose = _compose_pose(
        root_position,
        root_orientation,
        scoring_transform["position_m"],
        scoring_transform["orientation_xyzw"],
    )
    start_pose = _finite_vector(
        task_spec.get("start_pose_world"),
        length=7,
        error="native_rigid_construction_start_pose_invalid",
    )
    start_orientation = _quaternion(
        start_pose[3:], error="native_rigid_construction_start_pose_invalid"
    )
    destination, destination_bounds = _destination(task_spec)
    support_interval = _finite_vector(
        task_spec.get("support_height_interval_m"),
        length=2,
        error="native_rigid_construction_support_interval_invalid",
    )
    if support_interval[0] >= support_interval[1]:
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_support_interval_invalid"]
        )
    pregrasp_clearance = float(affordance["pregrasp_clearance_m"])
    manipulation_strategy = str(
        task_spec.get("manipulation_strategy") or "pick_and_place"
    )
    if manipulation_strategy not in RIGID_MANIPULATION_STRATEGIES:
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_manipulation_strategy_invalid"]
        )
    lift = _positive(
        task_spec.get("minimum_lift_m"),
        error="native_rigid_construction_lift_invalid",
        allow_zero=manipulation_strategy == "planar_push",
    )
    minimum_translation = _positive(
        task_spec.get("minimum_translation_m"),
        error="native_rigid_construction_translation_invalid",
    )
    contact_force = _positive(
        task_spec.get("task_contact_minimum_force_n"),
        error="native_rigid_construction_contact_threshold_invalid",
    )
    collision_force = _positive(
        task_spec.get("collision_failure_minimum_force_n"),
        error="native_rigid_construction_collision_threshold_invalid",
    )
    reset_translation = _positive(
        task_spec.get("reset_translation_tolerance_m"),
        error="native_rigid_construction_reset_translation_tolerance_invalid",
    )
    reset_orientation = _positive(
        task_spec.get("reset_orientation_tolerance_rad"),
        error="native_rigid_construction_reset_orientation_tolerance_invalid",
    )
    settle_position = _positive(
        task_spec.get("settle_position_tolerance_m"),
        error="native_rigid_construction_settle_position_tolerance_invalid",
    )
    relocation_tracking = _positive(
        task_spec.get("relocation_tracking_tolerance_m"),
        error="native_rigid_construction_relocation_tracking_tolerance_invalid",
    )
    destination_orientation_tolerance = _positive(
        task_spec.get("destination_orientation_tolerance_rad"),
        error="native_rigid_construction_destination_orientation_tolerance_invalid",
    )
    settle_orientation = _positive(
        task_spec.get("settle_orientation_tolerance_rad"),
        error="native_rigid_construction_settle_orientation_tolerance_invalid",
    )
    workspace_bounds = task_spec.get("workspace_position_bounds_world_m")
    if not isinstance(workspace_bounds, Mapping):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_workspace_bounds_missing"]
        )
    workspace_minimum = _finite_vector(
        workspace_bounds.get("minimum"),
        length=3,
        error="native_rigid_construction_workspace_bounds_invalid",
    )
    workspace_maximum = _finite_vector(
        workspace_bounds.get("maximum"),
        length=3,
        error="native_rigid_construction_workspace_bounds_invalid",
    )
    if any(
        low >= high
        for low, high in zip(workspace_minimum, workspace_maximum, strict=True)
    ):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_workspace_bounds_invalid"]
        )
    if (
        math.dist(computed_start_pose[:3], start_pose[:3]) > reset_translation
        or _quaternion_angle_xyzw(computed_start_pose[3:], start_orientation)
        > reset_orientation
    ):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_scoring_frame_reset_mismatch"]
        )
    destination_orientation = _quaternion(
        task_spec.get("destination_orientation_xyzw"),
        error="native_rigid_construction_destination_orientation_invalid",
    )
    settle_samples = task_spec.get("settle_window_samples")
    if (
        isinstance(settle_samples, bool)
        or not isinstance(settle_samples, int)
        or settle_samples <= 0
    ):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_settle_window_invalid"]
        )
    if settle_samples > maximum_steps_per_phase:
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_settle_window_exceeds_phase_budget"]
        )
    # The qualifying controls episode replays every qualified phase at its
    # exact step count and then appends this settle window inside the same
    # ``maximum_action_steps`` cap.  Sealing the reserved construction budget
    # here guarantees any construction this plan qualifies is replayable, so
    # ``native_rigid_control_action_budget_exceeded`` cannot fire after a paid
    # construction succeeded.
    maximum_action_steps = (plan.get("cadence") or {}).get("maximum_action_steps")
    if (
        isinstance(maximum_action_steps, bool)
        or not isinstance(maximum_action_steps, int)
        or maximum_action_steps <= 0
    ):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_action_budget_invalid"]
        )
    maximum_construction_total_steps = maximum_action_steps - settle_samples

    start = start_pose[:3]
    contact_local = affordance["contact_point_scoring_frame_m"]
    approach_world = _quaternion_rotate_xyzw(
        start_orientation, affordance["approach_unit_scoring_frame"]
    )
    destination_approach_world = _quaternion_rotate_xyzw(
        destination_orientation, affordance["approach_unit_scoring_frame"]
    )
    lift_world = affordance["lift_unit_world"]
    contact_start_offset = _quaternion_rotate_xyzw(start_orientation, contact_local)
    contact_start = [start[index] + contact_start_offset[index] for index in range(3)]
    pregrasp = [
        contact_start[index] + approach_world[index] * pregrasp_clearance
        for index in range(3)
    ]
    lifted_start_contact = [
        contact_start[index] + lift_world[index] * lift for index in range(3)
    ]
    contact_destination_offset = _quaternion_rotate_xyzw(
        destination_orientation, contact_local
    )
    contact_destination = [
        destination[index] + contact_destination_offset[index] for index in range(3)
    ]
    lifted_destination_contact = [
        contact_destination[index] + lift_world[index] * lift for index in range(3)
    ]
    start_gripper_orientation = _quaternion_product_xyzw(
        start_orientation, affordance["gripper_orientation_scoring_frame_xyzw"]
    )
    destination_gripper_orientation = _quaternion_product_xyzw(
        destination_orientation,
        affordance["gripper_orientation_scoring_frame_xyzw"],
    )
    if manipulation_strategy == "planar_push":
        phases = [
            _phase(
                "precontact",
                pregrasp,
                gripper_state="open",
                gate_ids=("precontact_reachability", "base_collision_clearance"),
                orientation_world_xyzw=start_gripper_orientation,
                expected_scoring_position_world_m=start,
                expected_scoring_orientation_world_xyzw=start_orientation,
            ),
            _phase(
                "push_contact",
                contact_start,
                gripper_state="closed",
                gate_ids=("push_contact", "support_contact"),
                orientation_world_xyzw=start_gripper_orientation,
                expected_scoring_position_world_m=start,
                expected_scoring_orientation_world_xyzw=start_orientation,
            ),
        ]
        for index in range(1, waypoint_count + 1):
            fraction = index / waypoint_count
            scoring_position = [
                start_value + (destination_value - start_value) * fraction
                for start_value, destination_value in zip(
                    start, destination, strict=True
                )
            ]
            scoring_orientation = _slerp_xyzw(
                start_orientation, destination_orientation, fraction
            )
            contact_offset = _quaternion_rotate_xyzw(
                scoring_orientation, contact_local
            )
            phases.append(
                _phase(
                    f"push_{index:02d}",
                    [
                        scoring_position[axis] + contact_offset[axis]
                        for axis in range(3)
                    ],
                    gripper_state="closed",
                    gate_ids=(
                        "push_path",
                        "push_contact_maintained",
                        "support_contact",
                        "workspace_containment",
                    ),
                    orientation_world_xyzw=_quaternion_product_xyzw(
                        scoring_orientation,
                        affordance["gripper_orientation_scoring_frame_xyzw"],
                    ),
                    expected_scoring_position_world_m=scoring_position,
                    expected_scoring_orientation_world_xyzw=scoring_orientation,
                )
            )
        destination_retreat = [
            contact_destination[index]
            + destination_approach_world[index] * pregrasp_clearance
            for index in range(3)
        ]
        phases.extend(
            [
                _phase(
                    "push_release",
                    destination_retreat,
                    gripper_state="open",
                    gate_ids=("release",),
                    orientation_world_xyzw=destination_gripper_orientation,
                    expected_scoring_position_world_m=destination,
                    expected_scoring_orientation_world_xyzw=destination_orientation,
                ),
                _phase(
                    "settle_observe",
                    destination_retreat,
                    gripper_state="open",
                    gate_ids=("support_stability", "destination_containment"),
                    orientation_world_xyzw=destination_gripper_orientation,
                    expected_scoring_position_world_m=destination,
                    expected_scoring_orientation_world_xyzw=destination_orientation,
                ),
                _phase(
                    "retreat",
                    destination_retreat,
                    gripper_state="open",
                    gate_ids=("retreat",),
                    orientation_world_xyzw=destination_gripper_orientation,
                    expected_scoring_position_world_m=destination,
                    expected_scoring_orientation_world_xyzw=destination_orientation,
                ),
                _phase(
                    "recovery",
                    pregrasp,
                    gripper_state="open",
                    gate_ids=("recovery", "reset_readback"),
                    orientation_world_xyzw=start_gripper_orientation,
                    expected_scoring_position_world_m=start,
                    expected_scoring_orientation_world_xyzw=start_orientation,
                ),
            ]
        )
        gate_contract = {
            "precontact_reachability": "native_end_effector_pose_readback",
            "base_collision_clearance": "native_robot_scene_contact_readback",
            "push_contact": "native_task_robot_contact_force_readback",
            "push_contact_maintained": "native_task_robot_contact_force_readback",
            "push_path": "native_task_root_pose_path_readback",
            "release": "native_gripper_and_task_contact_readback",
            "retreat": "native_grasp_frame_separation_readback",
            "support_contact": "native_task_scene_contact_force_readback",
            "support_stability": "native_settle_window_root_pose_readback",
            "destination_containment": "native_task_root_pose_volume_readback",
            "workspace_containment": "native_task_root_pose_workspace_bounds_readback",
            "recovery": "native_end_effector_pose_readback",
            "reset_readback": "native_robot_and_object_reset_replay",
        }
    else:
        phases = [
            _phase(
                "pregrasp",
                pregrasp,
                gripper_state="open",
                gate_ids=("pregrasp_reachability", "base_collision_clearance"),
                orientation_world_xyzw=start_gripper_orientation,
                expected_scoring_position_world_m=start,
                expected_scoring_orientation_world_xyzw=start_orientation,
            ),
            _phase(
                "grasp_contact",
                contact_start,
                gripper_state="closed",
                gate_ids=("grasp_contact",),
                orientation_world_xyzw=start_gripper_orientation,
                expected_scoring_position_world_m=start,
                expected_scoring_orientation_world_xyzw=start_orientation,
            ),
            _phase(
                "lift_clearance",
                lifted_start_contact,
                gripper_state="closed",
                gate_ids=("grasp_retention", "support_clearance"),
                orientation_world_xyzw=start_gripper_orientation,
                expected_scoring_position_world_m=[
                    start[index] + lift_world[index] * lift for index in range(3)
                ],
                expected_scoring_orientation_world_xyzw=start_orientation,
            ),
        ]
        for index in range(1, waypoint_count + 1):
            fraction = index / waypoint_count
            point = [
                start_value + (destination_value - start_value) * fraction
                for start_value, destination_value in zip(
                    lifted_start_contact, lifted_destination_contact, strict=True
                )
            ]
            expected_scoring = [
                start_value + (destination_value - start_value) * fraction
                + lift_world[axis] * lift
                for axis, (start_value, destination_value) in enumerate(
                    zip(start, destination, strict=True)
                )
            ]
            scoring_orientation = _slerp_xyzw(
                start_orientation, destination_orientation, fraction
            )
            phases.append(
                _phase(
                    f"relocate_{index:02d}",
                    point,
                    gripper_state="closed",
                    gate_ids=(
                        "relocation_path",
                        "grasp_retention",
                        "workspace_containment",
                    ),
                    orientation_world_xyzw=_quaternion_product_xyzw(
                        scoring_orientation,
                        affordance["gripper_orientation_scoring_frame_xyzw"],
                    ),
                    expected_scoring_position_world_m=expected_scoring,
                    expected_scoring_orientation_world_xyzw=scoring_orientation,
                )
            )
        phases.extend(
            [
                _phase(
                    "place",
                    contact_destination,
                    gripper_state="closed",
                    gate_ids=("destination_containment", "support_contact"),
                    orientation_world_xyzw=destination_gripper_orientation,
                    expected_scoring_position_world_m=destination,
                    expected_scoring_orientation_world_xyzw=destination_orientation,
                ),
                _phase(
                    "release",
                    contact_destination,
                    gripper_state="open",
                    gate_ids=("release",),
                    orientation_world_xyzw=destination_gripper_orientation,
                    expected_scoring_position_world_m=destination,
                    expected_scoring_orientation_world_xyzw=destination_orientation,
                ),
                _phase(
                    "settle_observe",
                    [
                        contact_destination[index]
                        + destination_approach_world[index] * pregrasp_clearance
                        for index in range(3)
                    ],
                    gripper_state="open",
                    gate_ids=("support_stability", "destination_containment"),
                    orientation_world_xyzw=destination_gripper_orientation,
                    expected_scoring_position_world_m=destination,
                    expected_scoring_orientation_world_xyzw=destination_orientation,
                ),
                _phase(
                    "retreat",
                    [
                        contact_destination[index]
                        + destination_approach_world[index] * pregrasp_clearance
                        for index in range(3)
                    ],
                    gripper_state="open",
                    gate_ids=("retreat",),
                    orientation_world_xyzw=destination_gripper_orientation,
                    expected_scoring_position_world_m=destination,
                    expected_scoring_orientation_world_xyzw=destination_orientation,
                ),
                _phase(
                    "recovery",
                    pregrasp,
                    gripper_state="open",
                    gate_ids=("recovery", "reset_readback"),
                    orientation_world_xyzw=start_gripper_orientation,
                    expected_scoring_position_world_m=start,
                    expected_scoring_orientation_world_xyzw=start_orientation,
                ),
            ]
        )
        gate_contract = {
        "pregrasp_reachability": "native_end_effector_pose_readback",
        "base_collision_clearance": "native_robot_scene_contact_readback",
        "grasp_contact": "native_task_robot_contact_force_readback",
        "grasp_retention": "native_task_root_pose_relative_motion_readback",
        "support_clearance": "native_task_root_height_readback",
        "relocation_path": "native_task_root_pose_path_readback",
        "release": "native_gripper_and_task_contact_readback",
        "retreat": "native_grasp_frame_separation_readback",
        "support_contact": "native_task_scene_contact_force_readback",
        "support_stability": "native_settle_window_root_pose_readback",
        "destination_containment": "native_task_root_pose_volume_readback",
        "workspace_containment": "native_task_root_pose_workspace_bounds_readback",
        "recovery": "native_end_effector_pose_readback",
        "reset_readback": "native_robot_and_object_reset_replay",
        }
    # The minimum qualified execution spends ``stable_samples`` on every phase
    # except settle_observe, which holds for the full settle window.  A
    # reserved budget below that cannot qualify any construction, so refuse at
    # plan time instead of paying to discover it.
    minimum_qualified_total_steps = (
        (len(phases) - 1) * stable_samples + settle_samples
    )
    if maximum_construction_total_steps < minimum_qualified_total_steps:
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_total_budget_infeasible"]
        )
    result: dict[str, Any] = {
        "schema_version": RIGID_SCHEMA_VERSION,
        "task_kind": "rigid_pick_place",
        "manipulation_strategy": manipulation_strategy,
        "scene_plan_digest": plan.get("plan_digest"),
        "subject_asset_id": subject.get("asset_id"),
        "interaction_affordance": affordance,
        "asset_root_reset_pose_world": [*root_position, *root_orientation],
        "asset_root_from_scoring_frame": scoring_transform,
        "start_scoring_pose_world": start_pose,
        "start_position_world_m": start,
        "destination_position_world_m": destination,
        "destination_orientation_xyzw": destination_orientation,
        "destination_position_bounds_world_m": destination_bounds,
        "support_height_interval_m": support_interval,
        "workspace_position_bounds_world_m": {
            "minimum": workspace_minimum,
            "maximum": workspace_maximum,
        },
        "settle_window_samples": settle_samples,
        "execution_parameters": {
            "arrival_tolerance_m": arrival_tolerance,
            "arrival_orientation_tolerance_rad": affordance[
                "arrival_orientation_tolerance_rad"
            ],
            "stable_samples": stable_samples,
            "maximum_steps_per_phase": maximum_steps_per_phase,
            "maximum_construction_total_steps": maximum_construction_total_steps,
            "relocation_waypoint_count": waypoint_count,
            **joint_command_limits(
                max_joint_delta_rad=max_joint_delta_rad,
                max_joint_setpoint_lead_rad=max_joint_setpoint_lead_rad,
                error="native_rigid_construction_joint_command_limits_invalid",
            ),
        },
        "thresholds": {
            "task_contact_minimum_force_n": contact_force,
            "collision_failure_minimum_force_n": collision_force,
            "reset_translation_tolerance_m": reset_translation,
            "reset_orientation_tolerance_rad": reset_orientation,
            "minimum_lift_m": lift,
            "minimum_translation_m": minimum_translation,
            "settle_position_tolerance_m": settle_position,
            "relocation_tracking_tolerance_m": relocation_tracking,
            "destination_orientation_tolerance_rad": destination_orientation_tolerance,
            "settle_orientation_tolerance_rad": settle_orientation,
        },
        "phases": phases,
        "phase_count": len(phases),
        "gate_contract": gate_contract,
        "required_gate_ids": sorted(gate_contract),
        "claim_boundary": {
            "phase_positions_are_native_ik_targets": True,
            "native_contact_and_motion_readback_required": True,
            "plan_is_not_construction_success": True,
            "plan_is_not_policy_or_physical_evidence": True,
        },
        "plan_digest": "",
    }
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    return result


def evaluate_rigid_construction_gates(
    *,
    phase_plan: Mapping[str, Any],
    phase_results: Sequence[Mapping[str, Any]],
    reset_replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate rigid construction solely from retained native readbacks."""

    if (
        phase_plan.get("schema_version") != RIGID_SCHEMA_VERSION
        or phase_plan.get("plan_digest")
        != canonical_digest(dict(phase_plan), digest_field="plan_digest")
    ):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_phase_plan_invalid"]
        )
    expected_ids = [row["phase_id"] for row in phase_plan["phases"]]
    observed = {
        str(row.get("phase_id") or ""): dict(row)
        for row in phase_results
        if isinstance(row, Mapping)
    }
    if set(observed) != set(expected_ids) or len(observed) != len(expected_ids):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_phase_results_invalid"]
        )

    def samples(phase_id: str) -> list[dict[str, Any]]:
        rows = list(observed[phase_id].get("task_samples") or [])
        terminal = observed[phase_id].get("task_sample")
        if isinstance(terminal, Mapping):
            rows.append(dict(terminal))
        if not rows or any(not isinstance(row, Mapping) for row in rows):
            raise NativeTaskConstructionPlanError(
                [f"native_rigid_construction_readback_missing:{phase_id}"]
            )
        return [dict(row) for row in rows]

    def pose(sample: Mapping[str, Any]) -> list[float]:
        value = _finite_vector(
            sample.get("task_scoring_pose_world"),
            length=7,
            error="native_rigid_construction_scoring_pose_readback_invalid",
        )
        _quaternion(
            value[3:],
            error="native_rigid_construction_scoring_pose_readback_invalid",
        )
        return value

    def position(sample: Mapping[str, Any]) -> list[float]:
        return pose(sample)[:3]

    thresholds = phase_plan["thresholds"]
    contact_threshold = float(thresholds["task_contact_minimum_force_n"])
    collision_threshold = float(thresholds["collision_failure_minimum_force_n"])
    start = list(phase_plan["start_position_world_m"])
    bounds = phase_plan["destination_position_bounds_world_m"]
    support = list(phase_plan["support_height_interval_m"])
    settle_count = int(phase_plan["settle_window_samples"])
    settle_tolerance = float(thresholds["settle_position_tolerance_m"])
    settle_orientation_tolerance = float(
        thresholds["settle_orientation_tolerance_rad"]
    )
    destination_orientation_tolerance = float(
        thresholds["destination_orientation_tolerance_rad"]
    )

    all_samples = [sample for phase_id in expected_ids for sample in samples(phase_id)]
    collision_clear = all(
        max(
            float(sample.get("robot_scene_contact_peak_force_n", float("inf"))),
            float(
                sample.get(
                    "robot_task_forbidden_collision_peak_force_n", float("inf")
                )
            ),
            float(sample.get("task_scene_collision_peak_force_n", float("inf"))),
        )
        < collision_threshold
        and sample.get("locked_joint_containment_violation") is False
        for sample in all_samples
    )
    strategy = str(phase_plan.get("manipulation_strategy") or "pick_and_place")
    push = strategy == "planar_push"
    contact_phase_id = "push_contact" if push else "grasp_contact"
    contact_rows = samples(contact_phase_id)
    initial_contact = max(
        float(row.get("task_robot_contact_peak_force_n", 0.0))
        for row in contact_rows
    ) >= contact_threshold
    if push:
        support_clearance = True
        relocation_ids = [
            phase_id for phase_id in expected_ids if phase_id.startswith("push_")
            and phase_id not in {"push_contact", "push_release"}
        ]
    else:
        lift_position = position(samples("lift_clearance")[-1])
        lift_delta = [lift_position[index] - start[index] for index in range(3)]
        support_clearance = sum(
            lift_delta[index]
            * float(phase_plan["interaction_affordance"]["lift_unit_world"][index])
            for index in range(3)
        ) + 1.0e-9 >= float(thresholds["minimum_lift_m"])
        relocation_ids = [
            phase_id
            for phase_id in expected_ids
            if phase_id.startswith("relocate_")
        ]
    relocation_terminal_samples = [samples(phase_id)[-1] for phase_id in relocation_ids]
    phase_by_id = {row["phase_id"]: row for row in phase_plan["phases"]}
    relocation_tracking = all(
        math.dist(
            position(sample),
            phase_by_id[phase_id]["expected_scoring_position_world_m"],
        )
        <= float(thresholds["relocation_tracking_tolerance_m"])
        and _quaternion_angle_xyzw(
            pose(sample)[3:],
            phase_by_id[phase_id]["expected_scoring_orientation_world_xyzw"],
        )
        <= destination_orientation_tolerance
        for phase_id, sample in zip(
            relocation_ids, relocation_terminal_samples, strict=True
        )
    )
    relocation_progress = (
        bool(relocation_terminal_samples)
        and math.dist(start, position(relocation_terminal_samples[-1]))
        >= float(thresholds["minimum_translation_m"])
    )
    relocation_path = relocation_tracking and relocation_progress
    closed_motion_ids = (
        [contact_phase_id, *relocation_ids]
        if push
        else ["lift_clearance", *relocation_ids, "place"]
    )
    closed_motion_samples = [
        sample for phase_id in closed_motion_ids for sample in samples(phase_id)
    ]
    contact_local = phase_plan["interaction_affordance"][
        "contact_point_scoring_frame_m"
    ]
    if push:
        contact_maintained = relocation_path and all(
            float(sample.get("task_robot_contact_peak_force_n", 0.0))
            >= contact_threshold
            and float(sample.get("task_support_contact_peak_force_n", 0.0))
            >= contact_threshold
            for sample in closed_motion_samples
        )
    else:
        contact_maintained = relocation_path and all(
            math.dist(
                [
                    pose(sample)[index]
                    + _quaternion_rotate_xyzw(pose(sample)[3:], contact_local)[index]
                    for index in range(3)
                ],
                _finite_vector(
                    sample.get("grasp_frame_position_world_m"),
                    length=3,
                    error="native_rigid_construction_grasp_frame_readback_invalid",
                ),
            )
            <= float(thresholds["relocation_tracking_tolerance_m"])
            and float(sample.get("task_robot_contact_peak_force_n", 0.0))
            >= contact_threshold
            for sample in closed_motion_samples
        )
    release_phase_id = "push_release" if push else "release"
    release_rows = samples(release_phase_id)
    release = (
        observed[release_phase_id].get("gripper_state") == "open"
        and release_rows[-1].get("finger_separation_m") is not None
        and float(release_rows[-1]["finger_separation_m"])
        > float(contact_rows[-1].get("finger_separation_m", float("inf")))
        and float(release_rows[-1].get("task_robot_contact_peak_force_n", float("inf")))
        < contact_threshold
    )
    settle_rows = samples("settle_observe")[-settle_count:]
    settle_positions = [position(row) for row in settle_rows]
    settle_poses = [pose(row) for row in settle_rows]
    final_position = settle_positions[-1]
    destination_containment = all(
        low <= value <= high
        for low, value, high in zip(
            bounds["minimum"], final_position, bounds["maximum"], strict=True
        )
    )
    destination_orientation = all(
        _quaternion_angle_xyzw(
            row[3:], phase_plan["destination_orientation_xyzw"]
        )
        <= destination_orientation_tolerance
        for row in settle_poses
    )
    support_contact = (
        len(settle_rows) >= settle_count
        and all(support[0] <= row[2] <= support[1] for row in settle_poses)
        and all(
            float(row.get("task_support_contact_peak_force_n", 0.0))
            >= contact_threshold
            for row in settle_rows
        )
    )
    support_stability = len(settle_rows) >= settle_count and all(
        math.dist(final_position, observed_position) <= settle_tolerance
        and _quaternion_angle_xyzw(settle_poses[-1][3:], observed_pose[3:])
        <= settle_orientation_tolerance
        for observed_position, observed_pose in zip(
            settle_positions, settle_poses, strict=True
        )
    )
    workspace = phase_plan["workspace_position_bounds_world_m"]
    workspace_containment = all(
        all(
            low <= value <= high
            for low, value, high in zip(
                workspace["minimum"],
                position(sample),
                workspace["maximum"],
                strict=True,
            )
        )
        for sample in all_samples
    )
    reachability = all(
        observed[phase_id].get("target_reached") is True
        for phase_id in expected_ids
    )
    gate_values = {
        "base_collision_clearance": collision_clear,
        "release": release,
        "retreat": observed["retreat"].get("target_reached") is True,
        "support_contact": support_contact,
        "support_stability": support_stability,
        "destination_containment": (
            destination_containment and destination_orientation
        ),
        "workspace_containment": workspace_containment,
        "recovery": observed["recovery"].get("target_reached") is True,
        "reset_readback": reset_replay.get("passed") is True,
        **(
            {
                "precontact_reachability": observed["precontact"].get(
                    "target_reached"
                )
                is True,
                "push_contact": initial_contact,
                "push_contact_maintained": contact_maintained,
                "push_path": relocation_path,
            }
            if push
            else {
                "pregrasp_reachability": observed["pregrasp"].get(
                    "target_reached"
                )
                is True,
                "grasp_contact": initial_contact,
                "grasp_retention": support_clearance and contact_maintained,
                "support_clearance": support_clearance,
                "relocation_path": relocation_path,
            }
        ),
    }
    gate_rows = [
        {
            "gate_id": gate_id,
            "measurement_authority": phase_plan["gate_contract"][gate_id],
            "passed": bool(gate_values[gate_id]),
        }
        for gate_id in phase_plan["required_gate_ids"]
    ]
    blockers = [
        f"native_rigid_construction_gate_failed:{row['gate_id']}"
        for row in gate_rows
        if not row["passed"]
    ]
    result = {
        "schema_version": "native_rigid_construction_gate_evaluation.v1",
        "phase_plan_digest": phase_plan["plan_digest"],
        "all_phase_targets_reached": reachability,
        "gates": gate_rows,
        "passed": not blockers and reachability,
        "blockers": sorted(blockers),
        "evaluation_digest": "",
    }
    result["evaluation_digest"] = canonical_digest(
        result, digest_field="evaluation_digest"
    )
    return result


def materialize_native_task_construction_phase_plan(
    scene_plan: Mapping[str, Any],
    *,
    articulated_clearance_m: float = 0.025,
    articulated_waypoint_count: int = 8,
    rigid_waypoint_count: int = 4,
    arrival_tolerance_m: float = 0.02,
    stable_samples: int = 2,
    maximum_steps_per_phase: int = 64,
    max_joint_delta_rad: float = MAX_JOINT_DELTA_RAD,
    max_joint_setpoint_lead_rad: float = MAX_JOINT_SETPOINT_LEAD_RAD,
    graph_articulated_prealign_clearance_m: float = (
        GRAPH_ARTICULATED_PREALIGN_CLEARANCE_M
    ),
) -> dict[str, Any]:
    """Dispatch one frozen scene plan without scene or object identities."""

    task_kind = str(scene_plan.get("task_kind") or "")
    if task_kind == "articulated_open_close":
        task_spec = scene_plan.get("task_spec") or {}
        if task_spec.get("schema_version") == "adp_task_spec.v2":
            if not isinstance(task_spec.get("interaction_affordance"), Mapping):
                raise NativeTaskConstructionPlanError(
                    [
                        "native_articulated_construction_general_interaction_"
                        "affordance_missing"
                    ]
                )
            return materialize_graph_articulated_construction_phase_plan(
                scene_plan,
                arrival_tolerance_m=arrival_tolerance_m,
                stable_samples=stable_samples,
                maximum_steps_per_phase=maximum_steps_per_phase,
                prealign_clearance_m=graph_articulated_prealign_clearance_m,
            )
        if task_spec.get("schema_version") != "adp_task_spec.v1":
            raise NativeTaskConstructionPlanError(
                ["native_articulated_graph_construction_task_spec_invalid"]
            )
        if (
            isinstance(stable_samples, bool)
            or not isinstance(stable_samples, int)
            or stable_samples <= 0
            or isinstance(maximum_steps_per_phase, bool)
            or not isinstance(maximum_steps_per_phase, int)
            or maximum_steps_per_phase <= 0
        ):
            raise NativeTaskConstructionPlanError(
                ["native_task_construction_execution_parameters_invalid"]
            )
        result = materialize_articulated_construction_phase_plan(
            scene_plan,
            clearance_m=articulated_clearance_m,
            waypoint_count=articulated_waypoint_count,
        )
        result["execution_parameters"] = {
            "arrival_tolerance_m": _positive(
                arrival_tolerance_m,
                error="native_task_construction_arrival_tolerance_invalid",
            ),
            "stable_samples": int(stable_samples),
            "maximum_steps_per_phase": int(maximum_steps_per_phase),
            "articulated_waypoint_count": int(articulated_waypoint_count),
            **joint_command_limits(
                max_joint_delta_rad=max_joint_delta_rad,
                max_joint_setpoint_lead_rad=max_joint_setpoint_lead_rad,
                error="native_task_construction_joint_command_limits_invalid",
            ),
        }
        result["plan_digest"] = canonical_digest(
            result, digest_field="plan_digest"
        )
        return result
    if task_kind == "rigid_pick_place":
        return materialize_rigid_construction_phase_plan(
            scene_plan,
            waypoint_count=rigid_waypoint_count,
            arrival_tolerance_m=arrival_tolerance_m,
            stable_samples=stable_samples,
            maximum_steps_per_phase=maximum_steps_per_phase,
            max_joint_delta_rad=max_joint_delta_rad,
            max_joint_setpoint_lead_rad=max_joint_setpoint_lead_rad,
        )
    raise NativeTaskConstructionPlanError(
        [f"native_task_construction_task_kind_unsupported:{task_kind or 'missing'}"]
    )


__all__ = [
    "GRAPH_ARTICULATED_AFFORDANCE_SCHEMA_VERSION",
    "GRAPH_ARTICULATED_SCHEMA_VERSION",
    "MAX_JOINT_DELTA_RAD",
    "MAX_JOINT_SETPOINT_LEAD_RAD",
    "VELOCITY_FEEDFORWARD_SCALE",
    "NativeTaskConstructionPlanError",
    "RIGID_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SUPPORTED_TASK_KINDS",
    "evaluate_graph_articulated_construction_gates",
    "evaluate_rigid_construction_gates",
    "joint_command_limits",
    "materialize_graph_articulated_construction_phase_plan",
    "materialize_native_task_construction_phase_plan",
    "materialize_rigid_construction_phase_plan",
]
