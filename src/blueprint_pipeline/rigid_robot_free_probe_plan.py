"""Robot-free probe states for a rigid pick/place success contract.

Beta journey step: L5 team policy evaluation. A team's policy may be scored on
a website scene only after a probe without any robot showed that the scene's
success contract can tell success from failure (owner decision 2026-10-12;
the articulated version drives a door joint). For a rigid object the robot is
replaced by a *virtual kinematic carrier*: the object's pose is set directly,
lifted, carried over the target, lowered and let go. Nothing else touches it.

This module is pure: it derives, from the exact task specification teams are
scored against, the states the probe sets and the scorer samples a virtual
carrier implies. Two uses share it:

- the contract states (``contract_cases``) are held still without physics and
  show the contract itself discriminates: the untouched start and wrong states
  (just outside the region, tipped, dropped, held too briefly, drifting) must
  fail and the authored goal must pass;
- the physics plan (``physics_plan``) is executed in a simulator by
  ``rigid_robot_free_probe_recorder``; physics alone decides where the released
  object rests, and the recorded trace is scored the same way.

Robot-side readback (gripper width, robot contact force, retries, regrasps,
withdrawal of the grasp frame) has no robot to measure; it is synthesized for
the virtual carrier, named in ``ROBOT_SIDE_CRITERIA`` and never claimed as
exercised. Contracts whose success is a robot state (holding the object at the
end) are refused by name. Destination-body tasks retain their qualified local
geometry and require a live destination-pose readback in every physics sample.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

from .adp_rigid_retreat_scoring import _oriented_bounds_projection
from .adp_rigid_task_scoring import _normalize_rigid_task_spec, _rotate_xyzw
from .decision_evidence_contracts import canonical_digest

PLAN_SCHEMA = "rigid_robot_free_probe_plan.v1"
# task_evaluation_rigid_relocation_native_adapter.NATIVE_PHYSICS_FREQUENCY_HZ;
# a test pins the two together so this module imports no adapter code.
PHYSICS_HZ = 120
RELEASE_CLEARANCE_M = 0.002
CARRY_CLEARANCE_M = 0.02
OUTSIDE_MARGIN_M = 0.002
DROP_EXTRA_FALL_M = 0.03
HOLD_EXTRA_SAMPLES = 10
CARRY_SECONDS = (0.5, 1.0, 0.5)
ZERO_ACTION_MINIMUM_SECONDS = 3.0
RELEASE_MINIMUM_SECONDS = 5.0
WITHDRAWAL_MARGIN_M = 0.05
GRAVITY_M_S2 = 9.81
ROBOT_SIDE_CRITERIA = ("gripper_state", "terminal_task_contact", "retreat", "maximum_task_contact_force",
                       "maximum_retries", "maximum_regrasps")


class RigidRobotFreeProbeUnsupported(ValueError):
    """The contract cannot be probed without a robot; the reason is typed."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__("rigid_robot_free_probe_unsupported:" + reason)


def _bounds(value: Any) -> tuple[list[float], list[float]] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        lower, upper = [float(v) for v in value["minimum"]], [float(v) for v in value["maximum"]]
    except (KeyError, TypeError, ValueError):
        return None
    return (lower, upper) if len(lower) == len(upper) == 3 else None


def probe_context(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the probe needs from the exact spec, or a typed refusal."""

    spec = _normalize_rigid_task_spec(task_spec)
    criteria = spec["task_success_contract"]["criteria"]
    if (criteria["gripper_state"]["mode"] == "closed_at_most"
            or criteria["terminal_task_contact"]["mode"] == "maintained"):
        # Success is the robot still holding the object; no carrier-free state shows that.
        raise RigidRobotFreeProbeUnsupported("terminal_grasp_success")
    if criteria["settling"]["mode"] != "required":
        # Without a hold window nothing says the released object stayed put.
        raise RigidRobotFreeProbeUnsupported("hold_window_not_required")
    if not spec["control_frequency_hz"]:
        # Teams are read once per control step; without the rate there is no episode to mirror.
        raise RigidRobotFreeProbeUnsupported("control_frequency_missing")
    raw_target = task_spec.get("target_position_world_m")
    lower = spec["destination_position_bounds_world_m"]["minimum"]
    upper = spec["destination_position_bounds_world_m"]["maximum"]
    target = ([float(v) for v in raw_target] if isinstance(raw_target, Sequence) and len(raw_target) == 3
              else [(a + b) / 2.0 for a, b in zip(lower, upper, strict=True)])
    workspace = _bounds(task_spec.get("workspace_position_bounds_world_m"))
    subject = _bounds(task_spec.get("subject_collision_bounds_scoring_frame_m"))
    surface = criteria.get("surface_target")
    stable = (math.ceil(float(surface["stable_seconds"]) * float(spec["control_frequency_hz"])) + 1
              if surface is not None and spec["control_frequency_hz"] else 0)
    window = int(criteria["settling"]["window_samples"])
    threshold = criteria["gripper_state"].get("threshold_m")
    lift = max(float(criteria["motion"]["minimum_lift_m"] or 0.0),
               float(spec["minimum_lift_m"]) if surface is not None else 0.0) + CARRY_CLEARANCE_M
    from .task_evaluation_surface_target import minimum_container_carry_height
    carry_height = minimum_container_carry_height(task_spec, clearance_m=CARRY_CLEARANCE_M)
    if carry_height is not None:
        # carry_waypoints uses one common flight height above the higher
        # endpoint; subtract that endpoint to avoid excess lift across levels.
        lift = max(lift, carry_height - max(float(spec["start_pose_world"][2]), target[2]))
    destination_pose = spec.get("destination_pose_world") if spec.get("destination_relation") else None
    return {
        "destination_pose_world": destination_pose,
        "spec": spec, "criteria": criteria, "start": list(spec["start_pose_world"]), "target": target,
        "criteria_contract_digest": str(spec["task_success_contract"]["contract_digest"]),
        "reference_xyzw": list(criteria["orientation"]["reference_xyzw"] or spec["destination_orientation_xyzw"]),
        "control_hz": float(spec["control_frequency_hz"] or 0.0),
        "hold_samples": max(window, stable) + HOLD_EXTRA_SAMPLES, "window_samples": window,
        "lift_m": lift,
        "open_width_m": float(threshold if threshold is not None else spec["release_gripper_width_min_m"]),
        "collision_threshold_n": float(task_spec.get("collision_failure_minimum_force_n") or 1.0),
        "workspace": workspace, "subject_bounds": subject, "surface": surface,
        "retreat": criteria.get("retreat"),
    }


def interpolate_pose(a: Sequence[float], b: Sequence[float], fraction: float) -> list[float]:
    position = [a[i] + (b[i] - a[i]) * fraction for i in range(3)]
    dot = sum(a[3 + i] * b[3 + i] for i in range(4))
    sign = -1.0 if dot < 0.0 else 1.0
    quaternion = [a[3 + i] + (sign * b[3 + i] - a[3 + i]) * fraction for i in range(4)]
    norm = math.sqrt(sum(v * v for v in quaternion))
    return position + [v / norm for v in quaternion]


def carry_waypoints(start: Sequence[float], goal: Sequence[float], lift_m: float) -> list[list[float]]:
    """Lift straight up, cross at a common height, lower onto the goal."""

    height = max(float(start[2]), float(goal[2])) + lift_m
    return [list(start), [start[0], start[1], height, *start[3:]], [goal[0], goal[1], height, *goal[3:]],
            list(goal)]


def _segments(waypoints: Sequence[Sequence[float]], steps: Sequence[int]) -> list[list[float]]:
    poses: list[list[float]] = []
    for (a, b), count in zip(zip(waypoints, waypoints[1:]), steps, strict=True):
        poses += [interpolate_pose(a, b, (k + 1) / count) for k in range(count)]
    return poses


def _row(pose: Sequence[float], phase: str, *, support: bool) -> dict[str, Any]:
    return {"phase": phase, "pose": [float(v) for v in pose], "support_contact_active": support,
            "scene_collision_force_n": 0.0}


def _placement(context: Mapping[str, Any], goal: Sequence[float], hold: int, *,
               drop_m: float = 0.0, drift_m: float = 0.0) -> list[dict[str, Any]]:
    """Untouched first reading, carry, then the released object held by nothing but the state set."""

    hz = context["control_hz"]
    steps = [max(1, round(seconds * hz)) for seconds in CARRY_SECONDS]
    release = [*goal[:2], goal[2] + drop_m, *goal[3:]]
    rows = [_row(context["start"], "zero_action", support=True)]
    rows += [_row(pose, "carry", support=False)
             for pose in _segments(carry_waypoints(context["start"], release, context["lift_m"]), steps)]
    elapsed = 0.0
    while drop_m > 0.0 and release[2] - 0.5 * GRAVITY_M_S2 * elapsed ** 2 > goal[2]:
        elapsed += 1.0 / hz
        z = max(goal[2], release[2] - 0.5 * GRAVITY_M_S2 * elapsed ** 2)
        rows.append(_row([*goal[:2], z, *goal[3:]], "release", support=z <= goal[2]))
    for k in range(hold):
        offset = drift_m * (1.0 - k / max(1, hold - 1))
        rows.append(_row([goal[0] - offset, *goal[1:]], "release", support=True))
    return rows


def _outside_offsets(context: Mapping[str, Any]) -> list[tuple[str, list[float]]]:
    """Just beyond every horizontal edge of the success region, including a marked disc."""

    spec, target = context["spec"], context["target"]
    lower = spec["destination_position_bounds_world_m"]["minimum"]
    upper = spec["destination_position_bounds_world_m"]["maximum"]
    destination = context.get("destination_pose_world")
    if destination is not None:
        local = spec["destination_interior_bounds_body_frame_m"]
        center = [(a + b) / 2 for a, b in zip(local["minimum"], local["maximum"], strict=True)]
        offsets = []
        for axis, sign, name in ((0, 1.0, "x_plus"), (0, -1.0, "x_minus"),
                                 (1, 1.0, "y_plus"), (1, -1.0, "y_minus")):
            point = list(center)
            point[axis] = local["maximum" if sign > 0 else "minimum"][axis] + sign * OUTSIDE_MARGIN_M
            rotated = _rotate_xyzw(point, destination[3:])
            offsets.append((name, [destination[i] + rotated[i] - target[i] for i in range(3)]))
        return offsets
    reach = 0.0
    if context["surface"] is not None:
        bounds = context["subject_bounds"]
        extent = (max(math.hypot(x, y) for x, y in product(*zip(bounds[0][:2], bounds[1][:2], strict=True)))
                  if bounds else 0.0)
        center = context["surface"]["surface_position_world_m"]
        reach = float(context["surface"]["radius_m"]) + extent + math.dist(center[:2], target[:2])
    offsets = []
    for axis, sign, name in ((0, 1.0, "x_plus"), (0, -1.0, "x_minus"), (1, 1.0, "y_plus"), (1, -1.0, "y_minus")):
        edge = upper[axis] - target[axis] if sign > 0 else target[axis] - lower[axis]
        shift = [0.0, 0.0, 0.0]
        shift[axis] = sign * (max(edge, reach) + OUTSIDE_MARGIN_M)
        offsets.append((name, shift))
    return offsets


def contract_cases(context: Mapping[str, Any]) -> dict[str, tuple[bool, list[dict[str, Any]]]]:
    """Named robot-free states and whether the contract must accept each."""

    goal = [*context["target"], *context["reference_xyzw"]]
    hold, window = context["hold_samples"], context["window_samples"]
    settle = context["criteria"]["settling"]
    placed = _placement(context, goal, hold)
    cases: dict[str, tuple[bool, list[dict[str, Any]]]] = {
        "untouched": (False, [_row(context["start"], "zero_action", support=True)] * len(placed)),
        "goal": (True, placed),
        "short_hold": (False, _placement(context, goal, window - 1)),
        # Moving by twice the settle tolerance inside the final hold window.
        "drift": (False, _placement(context, goal, hold, drift_m=(
            2.0 * float(settle["position_tolerance_m"]) + 0.001) * (hold - 1) / max(1, window - 1))),
    }
    for name, shift in _outside_offsets(context):
        cases["outside_" + name] = (False, _placement(
            context, [goal[i] + shift[i] for i in range(3)] + goal[3:], hold))
    if context["criteria"]["orientation"]["mode"] == "required" or context["surface"] is not None:
        half = math.sqrt(0.5)
        reference = context["reference_xyzw"]
        tipped = [half * reference[3] + half * reference[0], half * reference[1] + half * reference[2],
                  half * reference[2] - half * reference[1], half * reference[3] - half * reference[0]]
        cases["tipped"] = (False, _placement(context, goal[:3] + tipped, hold))
    no_drop = context["criteria"]["temporal_invariants"]["no_drop"]
    if no_drop["mode"] == "required":
        cases["dropped"] = (False, _placement(context, goal, hold,
                                              drop_m=float(no_drop["minimum_fall_m"]) + DROP_EXTRA_FALL_M))
    if context.get("destination_pose_world") is not None:
        for _expected, rows in cases.values():
            for row in rows:
                row["destination_pose_world"] = list(context["destination_pose_world"])
    return cases


def physics_plan(context: Mapping[str, Any]) -> dict[str, Any]:
    """What a simulator executes: untouched, then carried to the goal and let go."""

    hz = context["control_hz"]
    hold_seconds = context["hold_samples"] / hz if hz else 0.0
    plan = {
        "schema_version": PLAN_SCHEMA, "physics_hz": PHYSICS_HZ,
        "start_scoring_pose_world": list(context["start"]),
        "release_scoring_pose_world": [*context["target"][:2], context["target"][2] + RELEASE_CLEARANCE_M,
                                       *context["reference_xyzw"]],
        "carry_lift_m": context["lift_m"], "carry_seconds": list(CARRY_SECONDS),
        "zero_action_seconds": max(ZERO_ACTION_MINIMUM_SECONDS, hold_seconds),
        "release_seconds": max(RELEASE_MINIMUM_SECONDS, hold_seconds + 1.0),
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def scorer_samples(context: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Robot-free readings as scorer samples, with the virtual carrier's robot-side state."""

    workspace, bounds, retreat = context["workspace"], context["subject_bounds"], context["retreat"]
    threshold = context["collision_threshold_n"]
    surface = context["surface"]
    marker = ([*surface["surface_position_world_m"][:2], surface["surface_position_world_m"][2] + 0.001,
               0.0, 0.0, 0.0, 1.0] if surface is not None else None)
    samples = []
    for index, row in enumerate(rows):
        pose = [float(v) for v in row["pose"]]
        carrying = row["phase"] == "carry"
        scene_force = float(row["scene_collision_force_n"])
        corners = ([[pose[i] + c[i] for i in range(3)]
                    for c in (_rotate_xyzw(list(p), pose[3:]) for p in product(*zip(*bounds, strict=True)))]
                   if bounds else [pose[:3]])
        outside = (any(not all(lo <= c[i] <= hi for i, (lo, hi) in enumerate(zip(*workspace, strict=True)))
                       for c in corners) if workspace else False)
        sample: dict[str, Any] = {
            "step_index": index, "task_object_pose_world": pose,
            "task_contact_active": carrying, "support_contact_active": bool(row["support_contact_active"]),
            "gripper_width_m": 0.0 if carrying else context["open_width_m"],
            "task_robot_contact_peak_force_n": 0.0, "robot_scene_contact_peak_force_n": 0.0,
            "robot_task_forbidden_collision_peak_force_n": 0.0, "task_scene_collision_peak_force_n": scene_force,
            "collision_failure_minimum_force_n": threshold,
            "robot_collision_failure": False, "forbidden_robot_task_collision_failure": False,
            "locked_joint_containment_violation": False, "scene_collision_failure": scene_force >= threshold,
            "containment_violation": bool(workspace) and not all(
                lo <= pose[i] <= hi for i, (lo, hi) in enumerate(zip(*workspace, strict=True))),
            "workspace_excursion": outside,
            "contact_classes_active": ["object_background"] if scene_force >= threshold else [],
            "retry_count": 0, "regrasp_count": 0,
        }
        destination = row.get("destination_pose_world") if context.get("destination_pose_world") is not None else marker
        if destination is not None:
            sample["destination_pose_world"] = list(destination)
        if retreat is not None and destination is not None and bounds:
            direction = _rotate_xyzw(retreat["withdrawal_unit_destination_frame"], destination[3:])
            clear = (0.0 if carrying else _oriented_bounds_projection(
                {"minimum": bounds[0], "maximum": bounds[1]}, pose[3:], direction)
                + float(retreat["minimum_clearance_m"]) + WITHDRAWAL_MARGIN_M)
            sample["grasp_frame_position_world_m"] = [pose[i] + direction[i] * clear for i in range(3)]
        samples.append(sample)
    return samples


__all__ = [
    "PHYSICS_HZ",
    "PLAN_SCHEMA",
    "ROBOT_SIDE_CRITERIA",
    "RigidRobotFreeProbeUnsupported",
    "carry_waypoints",
    "contract_cases",
    "interpolate_pose",
    "physics_plan",
    "probe_context",
    "scorer_samples",
]
