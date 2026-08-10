"""Read an articulated task's joint state into the shape the scorer accepts.

This is the seam between a live physics view and the task-neutral scorer. It
exists as its own function because everything about it is easy to get subtly
wrong in ways that only show up mid-episode, on a GPU, after the money is spent.

The joint set is built from the binding rather than from whatever the runtime
happens to expose. The scorer rejects any sample whose joints differ at all
from the spec's, so a runtime that reports an extra DOF - or drops one because
a name did not resolve - fails the whole episode at scoring time rather than
at setup.

Readings are checked for plausibility, not just finiteness. A NaN out of a
diverged solver is obvious; a hinge reporting 45 is not, until you notice that
is 2578 degrees. This lane has already paid for one degree/radian confusion in
USD drive damping, and the same slip here would be recorded as a door angle and
scored.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence


ARTICULATED_TASK_SAMPLE_SCHEMA_VERSION = "articulated_task_sample.v1"
# No revolute joint on a household fitting turns more than a couple of turns;
# anything past this is a unit slip rather than a pose.
MAXIMUM_PLAUSIBLE_JOINT_RAD = 4.0 * math.pi
MAXIMUM_PLAUSIBLE_JOINT_RATE_RAD_S = 50.0
# Enough to name the fault and its frame, short of pasting a stack.
MAXIMUM_CAUSE_CHARACTERS = 220
# The scorer requires every one of these on every sample. They are supplied,
# never defaulted: False on a collision flag is not "we did not look", it is
# the positive claim that the robot did not collide, and this program must
# never make that claim by omission.
REQUIRED_OBSERVATIONS = (
    "task_contact_active",
    "containment_violation",
    "robot_collision_failure",
    "scene_collision_failure",
    "retreat_completed",
)
# Derived rather than observed: position against the authored limits needs no
# sensor, and asking for it would invite a caller to assert it.
DERIVED_OBSERVATIONS = ("joint_limit_violation",)


class ArticulatedTaskSampleError(ValueError):
    """Stable, sorted articulated task-sample failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def build_articulated_task_sample(
    *,
    joint_ids: Sequence[str],
    read_joint_state: Callable[[str], Any],
    joint_hard_limits_rad: Mapping[str, Sequence[float]],
    read_task_contact_active: Callable[[], Any] | None = None,
    read_containment_violation: Callable[[], Any] | None = None,
    read_robot_collision_failure: Callable[[], Any] | None = None,
    read_scene_collision_failure: Callable[[], Any] | None = None,
    read_retreat_completed: Callable[[], Any] | None = None,
    joint_limit_tolerance_rad: float = 0.0,
    step_index: int | None = None,
) -> dict[str, Any]:
    """Collect one articulated sample, refusing anything it cannot vouch for."""

    ids = [str(value) for value in joint_ids if str(value)]
    if not ids:
        raise ArticulatedTaskSampleError(["articulated_task_sample_joint_ids_missing"])

    errors: list[str] = []
    positions: dict[str, float] = {}
    velocities: dict[str, float] = {}
    for joint_id in sorted(set(ids)):
        try:
            reading = read_joint_state(joint_id)
        except Exception as exc:  # noqa: BLE001 - any failure to read is one fault
            # Carry the cause. rt23 returned this label for both joints and
            # nothing else, from inside a physics buffer read, and the launch
            # bought a symptom. Truncated because a physics traceback in an
            # error string is unreadable rather than helpful.
            detail = f"{type(exc).__name__}:{exc}"[:MAXIMUM_CAUSE_CHARACTERS]
            errors.append(
                f"articulated_task_sample_joint_unreadable:{joint_id}:{detail}"
            )
            continue
        try:
            position, velocity = (float(reading[0]), float(reading[1]))
        except (TypeError, ValueError, IndexError):
            errors.append(f"articulated_task_sample_joint_state_invalid:{joint_id}")
            continue
        if not math.isfinite(position) or not math.isfinite(velocity):
            errors.append(f"articulated_task_sample_joint_state_not_finite:{joint_id}")
            continue
        if abs(position) > MAXIMUM_PLAUSIBLE_JOINT_RAD:
            errors.append(
                f"articulated_task_sample_joint_position_implausible:{joint_id}:"
                f"{position:.4g}rad"
            )
            continue
        if abs(velocity) > MAXIMUM_PLAUSIBLE_JOINT_RATE_RAD_S:
            errors.append(
                f"articulated_task_sample_joint_velocity_implausible:{joint_id}:"
                f"{velocity:.4g}rad_s"
            )
            continue
        # Plain floats: numpy scalars out of a physics view break canonical
        # digesting downstream.
        positions[joint_id] = float(position)
        velocities[joint_id] = float(velocity)
    if errors:
        raise ArticulatedTaskSampleError(errors)

    limits = dict(joint_hard_limits_rad or {})
    tolerance = abs(float(joint_limit_tolerance_rad))
    limit_violation = False
    for joint_id in sorted(positions):
        bounds = limits.get(joint_id)
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            errors.append(
                f"articulated_task_sample_joint_hard_limits_missing:{joint_id}"
            )
            continue
        lower, upper = float(bounds[0]), float(bounds[1])
        position = positions[joint_id]
        # A hard stop settles a hair past itself under any solver; the
        # tolerance absorbs that and nothing wider.
        if position < lower - tolerance or position > upper + tolerance:
            limit_violation = True

    observed: dict[str, bool] = {}
    readers = {
        "task_contact_active": read_task_contact_active,
        "containment_violation": read_containment_violation,
        "robot_collision_failure": read_robot_collision_failure,
        "scene_collision_failure": read_scene_collision_failure,
        "retreat_completed": read_retreat_completed,
    }
    for field in REQUIRED_OBSERVATIONS:
        reader = readers.get(field)
        if reader is None:
            errors.append(f"articulated_task_sample_observation_missing:{field}")
            continue
        try:
            value = reader()
        except Exception as exc:  # noqa: BLE001 - any failure to observe is one
            # Same rule as the joint read twenty lines up, which was taught to
            # carry its cause one commit before this one was not.
            detail = f"{type(exc).__name__}:{exc}"[:MAXIMUM_CAUSE_CHARACTERS]
            errors.append(
                f"articulated_task_sample_observation_failed:{field}:{detail}"
            )
            continue
        if not isinstance(value, bool):
            errors.append(f"articulated_task_sample_observation_not_boolean:{field}")
            continue
        observed[field] = value
    if errors:
        raise ArticulatedTaskSampleError(errors)

    sample: dict[str, Any] = {
        "schema_version": ARTICULATED_TASK_SAMPLE_SCHEMA_VERSION,
        "joint_positions_rad": positions,
        "joint_velocities_rad_s": velocities,
        "joint_limit_violation": limit_violation,
        **observed,
    }
    if step_index is not None:
        sample["step_index"] = int(step_index)
    return sample


__all__ = [
    "ARTICULATED_TASK_SAMPLE_SCHEMA_VERSION",
    "DERIVED_OBSERVATIONS",
    "REQUIRED_OBSERVATIONS",
    "MAXIMUM_PLAUSIBLE_JOINT_RAD",
    "MAXIMUM_PLAUSIBLE_JOINT_RATE_RAD_S",
    "ArticulatedTaskSampleError",
    "build_articulated_task_sample",
]
