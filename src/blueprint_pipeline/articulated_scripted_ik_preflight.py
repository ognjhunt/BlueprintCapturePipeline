"""Turn a planned handle path into the joint commands a control actually replays.

The control contract wants absolute joint positions, resolved before the run
and replayed open loop. That choice is deliberate - a scripted positive has to
be the same program every time or it cannot be a control - but it puts two
burdens here that a live servo would not have.

The first is continuity. Consecutive waypoints are solved independently, and a
seven-axis arm has many ways to reach the same pose, so an unseeded solver is
free to flip the elbow between neighbouring points. Replayed open loop that
reads as the arm snapping through its own workspace. Every solve is therefore
seeded from the previous solution.

The second is step size. Waypoints say where to go, not how finely; a step that
asks a joint to move further than the controller can track in one tick is a
teleport, and the arm will either miss the handle or hurl the door. Rather than
refuse such a plan, the trajectory is subdivided until every step is within
budget, which keeps the planner free to describe motion at whatever resolution
is natural for the geometry.

The solver itself is injected. Real IK needs a robot model and, in the lane
that matters, a GPU-adjacent runtime; everything around it is contract work
that should be provable on a laptop.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence


SCRIPTED_IK_PREFLIGHT_SCHEMA_VERSION = "articulated_scripted_ik_preflight.v1"
ARM_JOINT_COUNT = 7
ACTION_DIMENSION = 8
DEFAULT_MAX_JOINT_DELTA_RAD = 0.05
MAX_SUBDIVISIONS_PER_SEGMENT = 64


class ScriptedIkPreflightError(ValueError):
    """Stable, sorted scripted-IK preflight failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _joints(value: Any, error: str) -> list[float]:
    try:
        joints = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ScriptedIkPreflightError([error]) from exc
    if len(joints) != ARM_JOINT_COUNT or not all(
        math.isfinite(item) for item in joints
    ):
        raise ScriptedIkPreflightError([error])
    return joints


def resolve_scripted_joint_trajectory(
    *,
    poses: Sequence[dict[str, Any]],
    initial_joint_positions: Sequence[float],
    solve_pose: Callable[..., dict[str, Any]],
    max_joint_delta_rad: float = DEFAULT_MAX_JOINT_DELTA_RAD,
) -> dict[str, Any]:
    """Resolve planned poses to seeded, step-bounded 8D absolute-position actions."""

    if not poses:
        raise ScriptedIkPreflightError(["scripted_ik_preflight_poses_missing"])
    if (
        isinstance(max_joint_delta_rad, bool)
        or not isinstance(max_joint_delta_rad, (int, float))
        or not math.isfinite(float(max_joint_delta_rad))
        or float(max_joint_delta_rad) <= 0.0
    ):
        raise ScriptedIkPreflightError(
            ["scripted_ik_preflight_max_joint_delta_invalid"]
        )
    limit = float(max_joint_delta_rad)
    seed = _joints(
        initial_joint_positions, "scripted_ik_preflight_initial_joints_invalid"
    )

    errors: list[str] = []
    solved: list[dict[str, Any]] = []
    for index, raw in enumerate(poses):
        phase_id = str(raw.get("phase_id") or f"phase_{index:02d}")
        try:
            position = [float(value) for value in raw["position_world_m"]]
        except (KeyError, TypeError, ValueError):
            errors.append(f"scripted_ik_preflight_pose_invalid:{phase_id}")
            continue
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            errors.append(f"scripted_ik_preflight_pose_invalid:{phase_id}")
            continue
        orientation = raw.get("quaternion_world_xyzw")
        gripper_state = raw.get("gripper_state")
        has_gripper_command = "gripper_command" in raw
        if gripper_state is not None and has_gripper_command:
            errors.append(f"scripted_ik_preflight_gripper_ambiguous:{phase_id}")
            continue
        if gripper_state is not None:
            gripper_state = str(gripper_state)
            if gripper_state not in {"open", "closed"}:
                errors.append(f"scripted_ik_preflight_gripper_invalid:{phase_id}")
                continue
            gripper_command = None
        else:
            gripper = raw.get("gripper_command", 0.0)
            try:
                gripper_command = float(gripper)
            except (TypeError, ValueError):
                errors.append(f"scripted_ik_preflight_gripper_invalid:{phase_id}")
                continue
            if not math.isfinite(gripper_command):
                errors.append(f"scripted_ik_preflight_gripper_invalid:{phase_id}")
                continue

        result = solve_pose(
            position_world_m=position,
            quaternion_world_xyzw=(
                None
                if orientation is None
                else [float(value) for value in orientation]
            ),
            seed_joint_positions=list(seed),
        )
        if not isinstance(result, dict) or not result.get("solved"):
            # Naming the phase matters: a dropped waypoint otherwise reads as a
            # control that simply missed, rather than one that was never posed.
            reach = (
                result.get("position_error_m")
                if isinstance(result, dict)
                else None
            )
            errors.append(
                f"scripted_ik_preflight_pose_unreachable:{phase_id}:"
                f"error_m={reach}"
            )
            continue
        try:
            joints = _joints(
                result.get("joint_positions"),
                f"scripted_ik_preflight_solution_invalid:{phase_id}",
            )
        except ScriptedIkPreflightError as exc:
            errors.extend(exc.errors)
            continue
        solved.append(
            {
                "phase_id": phase_id,
                "joint_positions": joints,
                "gripper_command": gripper_command,
                "gripper_state": gripper_state,
                "position_error_m": float(result.get("position_error_m") or 0.0),
            }
        )
        seed = joints
    if errors:
        raise ScriptedIkPreflightError(errors)

    actions: list[dict[str, Any]] = []
    interpolated = 0
    previous = _joints(
        initial_joint_positions, "scripted_ik_preflight_initial_joints_invalid"
    )
    for row in solved:
        target = row["joint_positions"]
        span = max(abs(a - b) for a, b in zip(target, previous))
        steps = max(1, int(math.ceil(span / limit - 1e-9)))
        if steps > MAX_SUBDIVISIONS_PER_SEGMENT:
            raise ScriptedIkPreflightError(
                [
                    "scripted_ik_preflight_segment_exceeds_subdivision_cap:"
                    f"{row['phase_id']}:{steps}"
                ]
            )
        interpolated += steps - 1
        for step in range(1, steps + 1):
            fraction = step / steps
            joints = [
                previous[index] + (target[index] - previous[index]) * fraction
                for index in range(ARM_JOINT_COUNT)
            ]
            action = {
                "phase_id": row["phase_id"],
            }
            if row["gripper_state"] is not None:
                action.update(
                    {
                        "arm_joint_positions": joints,
                        "gripper_state": row["gripper_state"],
                    }
                )
            else:
                action["isaac_action"] = [*joints, row["gripper_command"]]
            actions.append(action)
        previous = target

    return {
        "schema_version": SCRIPTED_IK_PREFLIGHT_SCHEMA_VERSION,
        "actions": actions,
        "action_count": len(actions),
        "waypoint_count": len(solved),
        "interpolated_step_count": interpolated,
        "max_joint_delta_rad": limit,
        "worst_position_error_m": max(
            (row["position_error_m"] for row in solved), default=0.0
        ),
        "claim_boundary": {
            "solutions_are_seeded_so_the_elbow_cannot_flip": True,
            "replay_is_open_loop_not_servoed": True,
            "reachability_is_the_solver_s_claim_not_a_contact_proof": True,
            "semantic_gripper_states_require_native_convention_readback": any(
                row["gripper_state"] is not None for row in solved
            ),
        },
    }


__all__ = [
    "ACTION_DIMENSION",
    "ARM_JOINT_COUNT",
    "DEFAULT_MAX_JOINT_DELTA_RAD",
    "SCRIPTED_IK_PREFLIGHT_SCHEMA_VERSION",
    "ScriptedIkPreflightError",
    "resolve_scripted_joint_trajectory",
]
