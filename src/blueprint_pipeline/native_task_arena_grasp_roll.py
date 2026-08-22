"""Choose one grasp roll for the whole grasp, and put it in the plan.

C43 measured the arm at contact with ``panda_joint5`` on its hard lower stop
for a third of the phase while ``panda_joint6`` held full effort; off-sim IK
confirmed the authored contact orientation admits a best joint-limit margin of
0.0000 rad where the same position with the orientation free admits 0.8916 rad.
Rolling the grasp about its own approach axis is what buys that margin back.

The first attempt at this rolled the orientation *inside the solver* and sealed
the rolled quaternion in a receipt, while the control-plan row kept the
authored quaternion.  The live differential-IK controller drives the plan's
orientation, so the rolled pose was never commanded: it survived only as a
null-space posture preference, which by construction cannot move the primary
six-dimensional pose objective.  A run built that way could not have tested the
hypothesis in either direction.

So the roll is a property of the *plan*, decided once and written down:

* every candidate roll is evaluated across the entire grasp-holding family,
  not at contact entry alone;
* a roll is admissible only if its **worst** phase clears the required margin,
  so no phase in the family is left below the floor;
* among admissible rolls the **smallest** wins, and the authored orientation
  wins outright whenever it is itself admissible -- roll is a candidate
  generator, not something to maximise;
* the chosen roll is written into the target quaternions the controller and
  the arrival gate actually read, and the authored quaternion is preserved
  beside it.

What this does not do: it checks joint-limit margin and nothing else.  Pad
overlap, closure direction, approach collision and grasp-wrench validity are
not evaluated here and remain the construction stage's and the native contact
gate's to enforce.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


GRASP_ROLL_SCHEMA_VERSION = "native_task_arena_grasp_roll.v1"

#: Phases that hold the grasp.  Rolling only at contact and reverting would
#: twist the gripper against a rim it is already holding.
DEFAULT_GRASP_HOLDING_PHASE_IDS: tuple[str, ...] = (
    "contact_open",
    "contact_close",
    "joint_path_01",
    "joint_path_02",
    "joint_path_03",
    "joint_path_04",
    "release",
)

#: Rolls searched about the gripper's approach axis, smallest first.  These
#: stay inside the tilt an 85 mm jaw can afford on a 1.23 mm rim.
DEFAULT_GRASP_ROLL_CANDIDATES_RAD: tuple[float, ...] = (
    0.0, 0.175, -0.175, 0.349, -0.349, 0.524, -0.524,
)

#: Margin every holding phase must clear for a roll to be admissible.
DEFAULT_REQUIRED_MARGIN_RAD = 0.05


class GraspRollError(ValueError):
    """Fail-closed grasp-roll contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _unit(vector: Sequence[float]) -> list[float] | None:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return None
    return [value / norm for value in values]


def _multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    lx, ly, lz, lw = (float(value) for value in left)
    rx, ry, rz, rw = (float(value) for value in right)
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _rotate(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    x, y, z, w = (float(value) for value in quaternion)
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def rolled_grasp_quaternion(
    *,
    quaternion_xyzw: Sequence[float],
    approach_axis_body: Sequence[float],
    roll_rad: float,
) -> list[float]:
    """Roll a grasp orientation about its own approach axis.

    The axis is given in the grasp's own frame and carried into world by the
    grasp orientation itself, so the rotation is about the gripper's approach
    direction whatever direction that happens to point.
    """

    axis_body = _unit(approach_axis_body)
    base = [float(value) for value in quaternion_xyzw]
    if axis_body is None or len(base) != 4:
        raise GraspRollError(["grasp_roll_axis_or_quaternion_invalid"])
    if roll_rad == 0.0:
        return base
    axis_world = _rotate(base, axis_body)
    unit_world = _unit(axis_world)
    if unit_world is None:
        raise GraspRollError(["grasp_roll_axis_degenerate"])
    half = float(roll_rad) / 2.0
    sin_half = math.sin(half)
    roll = [
        unit_world[0] * sin_half,
        unit_world[1] * sin_half,
        unit_world[2] * sin_half,
        math.cos(half),
    ]
    return _multiply(roll, base)


def select_grasp_roll(
    *,
    holding_phases: Sequence[Mapping[str, Any]],
    approach_axis_body: Sequence[float],
    solve_phase: Callable[[Mapping[str, Any], Sequence[float]], Mapping[str, Any] | None],
    roll_candidates_rad: Sequence[float] = DEFAULT_GRASP_ROLL_CANDIDATES_RAD,
    required_margin_rad: float = DEFAULT_REQUIRED_MARGIN_RAD,
) -> dict[str, Any]:
    """Pick the smallest roll whose worst holding phase clears the floor.

    Every candidate is evaluated across the whole family.  Judging a roll by
    contact entry alone would let a later phase sit below the floor while the
    receipt reported a healthy margin for the one phase that was measured.
    """

    if not holding_phases:
        return {
            "schema_version": GRASP_ROLL_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "no_grasp_holding_phase",
            "selected_roll_rad": None,
        }
    candidates = [float(value) for value in roll_candidates_rad]
    if 0.0 not in candidates:
        candidates = [0.0, *candidates]
    candidates.sort(key=abs)

    attempts: list[dict[str, Any]] = []
    admissible: list[tuple[float, float]] = []
    for roll in candidates:
        worst: float | None = None
        worst_phase: str | None = None
        solved_all = True
        for phase in holding_phases:
            try:
                quaternion = rolled_grasp_quaternion(
                    quaternion_xyzw=phase["target_quaternion_world_xyzw"],
                    approach_axis_body=approach_axis_body,
                    roll_rad=roll,
                )
            except GraspRollError:
                solved_all = False
                break
            outcome = solve_phase(phase, quaternion)
            margin = (
                float(outcome.get("minimum_joint_limit_margin_rad") or 0.0)
                if isinstance(outcome, Mapping)
                else None
            )
            if margin is None:
                solved_all = False
                break
            if worst is None or margin < worst:
                worst, worst_phase = margin, str(phase.get("phase_id") or "")
        attempts.append(
            {
                "roll_rad": roll,
                "solved_every_holding_phase": solved_all,
                "worst_phase_id": worst_phase,
                "worst_minimum_joint_limit_margin_rad": worst,
            }
        )
        if solved_all and worst is not None and worst >= float(required_margin_rad):
            admissible.append((abs(roll), roll))

    if not admissible:
        return {
            "schema_version": GRASP_ROLL_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "no_roll_clears_required_margin_across_the_family",
            "selected_roll_rad": None,
            "required_margin_rad": float(required_margin_rad),
            "attempts": attempts,
            "claim_boundary": _CLAIM_BOUNDARY,
        }
    _, chosen = min(admissible)
    picked = next(row for row in attempts if row["roll_rad"] == chosen)
    return {
        "schema_version": GRASP_ROLL_SCHEMA_VERSION,
        "status": "selected",
        "selected_roll_rad": chosen,
        "authored_roll_is_admissible": any(row == 0.0 for _, row in admissible),
        "selected_worst_minimum_joint_limit_margin_rad": picked[
            "worst_minimum_joint_limit_margin_rad"
        ],
        "selected_worst_phase_id": picked["worst_phase_id"],
        "required_margin_rad": float(required_margin_rad),
        "attempts": attempts,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def derive_rolled_control_plan(
    *,
    control_plan: Mapping[str, Any],
    roll_rad: float,
    approach_axis_body: Sequence[float],
    holding_phase_ids: Sequence[str] = DEFAULT_GRASP_HOLDING_PHASE_IDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write the chosen roll into the quaternions the controller reads.

    The live differential-IK controller and the arrival gate both drive the
    plan's ``target_quaternion_world_xyzw``.  Sealing a rolled orientation
    anywhere else leaves it uncommanded, which is exactly how the first
    attempt at this produced a receipt full of rolls that never reached the
    robot.  The authored quaternion is preserved on each rewritten row.
    """

    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    plan = json.loads(json.dumps(dict(control_plan), allow_nan=False))
    actions = plan.get("scripted_positive_actions")
    if not isinstance(actions, list):
        raise GraspRollError(["grasp_roll_control_plan_invalid"])
    holding = set(holding_phase_ids)
    rewritten: list[dict[str, Any]] = []
    for row in actions:
        if not isinstance(row, Mapping):
            continue
        phase_id = str(row.get("phase_id") or "")
        quaternion = row.get("target_quaternion_world_xyzw")
        if phase_id not in holding or not isinstance(quaternion, list):
            continue
        rolled = rolled_grasp_quaternion(
            quaternion_xyzw=quaternion,
            approach_axis_body=approach_axis_body,
            roll_rad=roll_rad,
        )
        row["authored_target_quaternion_world_xyzw"] = list(quaternion)
        row["applied_grasp_roll_rad"] = float(roll_rad)
        row["target_quaternion_world_xyzw"] = rolled
        rewritten.append(
            {"phase_id": phase_id, "target_quaternion_world_xyzw": rolled}
        )
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    receipt = {
        "schema_version": GRASP_ROLL_SCHEMA_VERSION,
        "status": "applied" if rewritten else "not_applied",
        "applied_grasp_roll_rad": float(roll_rad),
        "rewritten_phase_ids": [row["phase_id"] for row in rewritten],
        "source_control_plan_digest": control_plan.get("plan_digest"),
        "derived_control_plan_digest": plan["plan_digest"],
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    return plan, receipt


_CLAIM_BOUNDARY = (
    "selects_and_applies_one_grasp_roll_about_the_approach_axis_for_every_"
    "grasp_holding_phase;admits_a_roll_only_if_its_worst_phase_clears_the_"
    "required_joint_limit_margin;does_not_check_pad_overlap_closure_direction_"
    "approach_collision_or_grasp_wrench;native_arrival_and_contact_gates_"
    "remain_the_authority"
)


__all__ = [
    "DEFAULT_GRASP_HOLDING_PHASE_IDS",
    "DEFAULT_GRASP_ROLL_CANDIDATES_RAD",
    "DEFAULT_REQUIRED_MARGIN_RAD",
    "GRASP_ROLL_SCHEMA_VERSION",
    "GraspRollError",
    "derive_rolled_control_plan",
    "rolled_grasp_quaternion",
    "select_grasp_roll",
]
