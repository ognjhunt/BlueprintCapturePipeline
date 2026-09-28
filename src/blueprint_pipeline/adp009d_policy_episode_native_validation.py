"""Pure native joint and reset checks shared by policy episode runtimes."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from adp009d_droid_action_execution import ARM_JOINT_COUNT
except ModuleNotFoundError:
    from .adp009d_droid_action_execution import ARM_JOINT_COUNT

BLOCKER_PRESTART_READINESS = "policy_episode_prestart_readiness_failed"


class PolicyEpisodeError(ValueError):
    """Fail-closed episode contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))



class NativeJointStateBoundsError(PolicyEpisodeError):
    """Observed simulator state violated native limits; this is not a policy response."""

    def __init__(self, readback: Mapping[str, Any]):
        self.readback = dict(readback)
        first = self.readback['violations'][0]
        super().__init__([
            f"native_joint_state_bounds_invalid:phase={readback['phase']}:"
            f"count={len(readback['violations'])}:first_joint_index={first['joint_index']}:"
            f"value={first['observed_rad']!r}:bounds={first['limits_rad']!r}"
        ])



def validate_native_joint_state(joints, joint_limits, *, phase: str) -> None:
    """Use the same unexpanded native position interval as reset admission."""
    values = [float(value) for value in joints]
    limits = [[float(value) for value in row] for row in joint_limits]
    if (len(values) != ARM_JOINT_COUNT or len(limits) != ARM_JOINT_COUNT
            or any(len(row) != 2 or row[0] >= row[1] for row in limits)
            or not all(math.isfinite(value) for value in [*values, *(x for row in limits for x in row)])):
        raise PolicyEpisodeError(['native_joint_state_bounds_contract_invalid'])
    violations = [{'joint_index': index, 'observed_rad': value, 'limits_rad': limits[index]}
                  for index, value in enumerate(values) if not limits[index][0] <= value <= limits[index][1]]
    if violations:
        raise NativeJointStateBoundsError({'schema_version': 'native_joint_state_bounds_violation.v1',
            'phase': phase, 'observed_joint_positions_rad': values, 'joint_limits_rad': limits,
            'violations': violations, 'observed_state_clamped': False,
            'candidate_response_was_not_the_refusing_boundary': True})



def _validate_task_reset_restoration(
    initial: Mapping[str, Any], restored: Mapping[str, Any], task_spec: Mapping[str, Any]
) -> None:
    """Compare measured task reset fields, excluding episode bookkeeping.

    Only frozen reset tolerances permit numerical differences. This is a task
    readback check, not a claim that camera/physics configuration was measured.
    """
    fields = {
        "can_pose_world", "task_object_pose_world", "destination_pose_world",
        "joint_positions_rad", "joint_velocities_rad_s", "gripper_width_m",
        "task_contact_active", "support_contact_active", "finger_contact_forces_n",
        "robot_collision_failure", "scene_collision_failure", "containment_violation",
        "forbidden_robot_task_collision_failure", "locked_joint_containment_violation",
    }
    for field in fields & (set(initial) | set(restored)):
        left, right = initial.get(field), restored.get(field)
        matches = field in initial and field in restored and left == right
        if field.endswith("pose_world") and isinstance(left, list) and isinstance(right, list):
            if len(left) == len(right) == 7:
                prefix = "destination_" if field == "destination_pose_world" else ""
                translation_tolerance = float(task_spec.get(prefix + "reset_translation_tolerance_m", 0.0))
                rotation_key = "destination_reset_rotation_tolerance_rad" if prefix else "reset_orientation_tolerance_rad"
                rotation_tolerance = float(task_spec.get(rotation_key, 0.0))
                finite = all(math.isfinite(float(v)) for v in [*left, *right])
                # q and -q represent the same physical rotation.
                dot = abs(sum(float(a) * float(b) for a, b in zip(left[3:], right[3:], strict=True)))
                matches = finite and math.dist(left[:3], right[:3]) <= translation_tolerance and (
                    left[3:] == right[3:] or 2 * math.acos(min(1.0, dot)) <= rotation_tolerance
                )
        if not matches:
            raise PolicyEpisodeError([f"{BLOCKER_PRESTART_READINESS}:task_reset_state_mismatch:{field}"])

