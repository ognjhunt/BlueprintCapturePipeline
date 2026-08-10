"""Measure which gripper command closes the fingers, rather than assuming it.

DROID encodes the gripper as a scalar in [0, 1] where above 0.5 means closed.
Arena's eighth action dimension has its own convention, and an inverted one
turns every commanded grasp into a release. That does not look like a harness
bug from the outside - it looks like a policy that reached the handle and
failed to hold it, which is the single most expensive misreading available on
this program.

So the convention is measured: command each candidate, let the fingers settle,
read the separation between the two finger bodies. The two widths that come
back are also exactly what the episode adapter needs, so measuring removes a
hardcoded pair of numbers at the same time.

Ambiguity stays ambiguous. If the two commands move the fingers by less than
the travel floor they are indistinguishable, and guessing a convention out of
noise would produce a confident wrong answer where a refusal costs one fix.

The rigid can lane grew this inline; the articulated scene worker needed the
same thing and had none, which would have failed at adapter construction after
a full Isaac boot and Arena provision. It is shared code so the next lane
inherits it rather than rediscovering it.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


GRIPPER_CONVENTION_PROBE_SCHEMA_VERSION = "adp009d_gripper_convention_probe.v1"
# Below this the two commands are the same command as far as the fingers are
# concerned.
DEFAULT_TRAVEL_FLOOR_M = 1.0e-3
DEFAULT_SETTLE_STEPS = 30
DEFAULT_FINGER_BODIES = ("left_inner_finger", "right_inner_finger")


class GripperConventionProbeError(ValueError):
    """Stable, sorted gripper-convention failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def measure_gripper_convention(
    *,
    apply_command: Callable[[float], None],
    read_finger_separation_m: Callable[[], float],
    body_names: Sequence[str],
    candidate_commands: Sequence[float] = (0.0, 1.0),
    required_finger_bodies: Sequence[str] = DEFAULT_FINGER_BODIES,
    travel_floor_m: float = DEFAULT_TRAVEL_FLOOR_M,
    settle_steps: int = DEFAULT_SETTLE_STEPS,
) -> dict[str, Any]:
    """Command each candidate and report which one closes the fingers."""

    commands = [float(value) for value in candidate_commands]
    if len(set(commands)) < 2:
        raise GripperConventionProbeError(
            ["gripper_convention_candidate_commands_insufficient"]
        )

    resolved = [name for name in required_finger_bodies if name in set(body_names)]
    if len(resolved) != 2:
        raise GripperConventionProbeError(
            [
                "gripper_convention_finger_bodies_not_resolved:"
                + ",".join(sorted(required_finger_bodies))
                + ":observed=" + ",".join(sorted(body_names))
            ]
        )

    separations: dict[str, float] = {}
    for command in commands:
        apply_command(command)
        separations[str(command)] = float(read_finger_separation_m())

    closed_command = min(commands, key=lambda value: separations[str(value)])
    open_command = max(commands, key=lambda value: separations[str(value)])
    travel = abs(separations[str(open_command)] - separations[str(closed_command)])
    if travel < float(travel_floor_m):
        raise GripperConventionProbeError(
            [
                "gripper_convention_travel_below_floor:"
                f"travel_m={travel!r}:floor_m={float(travel_floor_m)!r}"
            ]
        )

    return {
        "schema_version": GRIPPER_CONVENTION_PROBE_SCHEMA_VERSION,
        "candidate_commands": commands,
        "finger_bodies": list(resolved),
        "settle_steps": int(settle_steps),
        "finger_separation_m": separations,
        "separation_travel_m": travel,
        "closed_command": closed_command,
        "open_command": open_command,
        "gripper_closed_width_m": separations[str(closed_command)],
        "gripper_open_width_m": separations[str(open_command)],
        # DROID means "above 0.5 is closed". Reported, never corrected here:
        # silently flipping it would hide a runtime that changed under us.
        "convention_matches_droid": closed_command > open_command,
        "claim_boundary": {
            "separation_is_measured_not_declared": True,
            "convention_is_reported_not_corrected": True,
        },
    }


__all__ = [
    "DEFAULT_FINGER_BODIES",
    "DEFAULT_SETTLE_STEPS",
    "DEFAULT_TRAVEL_FLOOR_M",
    "GRIPPER_CONVENTION_PROBE_SCHEMA_VERSION",
    "GripperConventionProbeError",
    "measure_gripper_convention",
]
