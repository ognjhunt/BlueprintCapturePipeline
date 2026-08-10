"""Decide whether a gripper can hold a handle hard enough to open its door.

Reach says the arm can get there. Force capability says the arm is strong
enough. Neither says the hand can hold on, and for a sealed door that is the
constraint that actually binds: published clamp forces for a parallel gripper
sit in the same tens-of-newtons band as the force needed to break a
refrigerator gasket, so the grasp is where the margin runs out first.

Two grasp modes, and which one is available is geometry, not choice. A bar with
space behind it can be hooked - the fingers wrap it, and pull-out is limited by
the handle breaking rather than by friction. A handle moulded flush to the
panel can only be pinched, and then everything depends on clamp force times
friction, both of which are uncertain to tens of percent.

That uncertainty is why a bare pass is not enough. A friction pinch that beats
its load by ten percent will hold most of the time and slip some of the time,
and every slip will be recorded as a policy failure rather than as a grasp that
was never viable. So the margin is reported and a thin one is a finding.

Bounds only. Clamp force is a datasheet figure, the friction coefficient is an
assumption, and neither accounts for the handle's surface, contact patch, or
the moment the pull applies about the wrist.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


HANDLE_GRASPABILITY_SCHEMA_VERSION = "handle_graspability.v1"
MINIMUM_PULL_OUT_MARGIN = 1.5
GRASP_FORM_CLOSURE = "form_closure_available"
GRASP_FRICTION_ONLY = "friction_pinch_only"


class HandleGraspabilityError(ValueError):
    """Stable, sorted handle-graspability failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _vector(value: Any, error: str) -> list[float]:
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise HandleGraspabilityError([error]) from exc
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise HandleGraspabilityError([error])
    return values


def _positive(value: Any, name: str, errors: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"handle_graspability_{name}_invalid")
        return 0.0
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        errors.append(f"handle_graspability_{name}_invalid")
        return 0.0
    return number


def evaluate_handle_graspability(
    *,
    handle_aabb_min_m: Sequence[float],
    handle_aabb_max_m: Sequence[float],
    panel_face_offset_m: float,
    outward_normal_world: Sequence[float],
    hinge_axis_world: Sequence[float],
    required_pull_force_n: float,
    gripper_clamp_force_n: float,
    gripper_stroke_m: float,
    gripper_finger_clearance_m: float,
    friction_coefficient: float,
    minimum_margin: float = MINIMUM_PULL_OUT_MARGIN,
) -> dict[str, Any]:
    """Report the grasp mode geometry allows and whether it holds the load."""

    low = _vector(handle_aabb_min_m, "handle_graspability_aabb_invalid")
    high = _vector(handle_aabb_max_m, "handle_graspability_aabb_invalid")
    normal = _vector(outward_normal_world, "handle_graspability_normal_invalid")
    axis = _vector(hinge_axis_world, "handle_graspability_axis_invalid")

    errors: list[str] = []
    required = _positive(required_pull_force_n, "required_pull_force", errors)
    clamp = _positive(gripper_clamp_force_n, "gripper_clamp_force", errors)
    stroke = _positive(gripper_stroke_m, "gripper_stroke", errors)
    friction = _positive(friction_coefficient, "friction_coefficient", errors)
    if any(high[index] < low[index] for index in range(3)):
        errors.append("handle_graspability_aabb_inverted")
    if errors:
        raise HandleGraspabilityError(errors)

    def _extent(direction: Sequence[float]) -> float:
        length = math.sqrt(sum(value * value for value in direction))
        if length <= 0.0:
            raise HandleGraspabilityError(["handle_graspability_direction_degenerate"])
        unit = [value / length for value in direction]
        return sum(
            abs(unit[index]) * (high[index] - low[index]) for index in range(3)
        )

    # Which span the jaws close on is not a free choice. The pull is along the
    # outward normal, so closing the jaws along that same direction just draws
    # the handle out from between them - clamp force does no work against the
    # load at all. Only a pinch across the pull resists it, which here means
    # across the hinge axis or across the handle's width.
    width = [
        axis[1] * normal[2] - axis[2] * normal[1],
        axis[2] * normal[0] - axis[0] * normal[2],
        axis[0] * normal[1] - axis[1] * normal[0],
    ]
    candidates = [("hinge_axis", _extent(axis)), ("handle_width", _extent(width))]
    viable = sorted(
        (span, name) for name, span in candidates if span <= stroke
    )
    if not viable:
        raise HandleGraspabilityError(
            [
                "handle_graspability_pinch_span_exceeds_gripper_stroke:"
                + ",".join(f"{name}={span:.4f}" for name, span in candidates)
                + f">{stroke:.4f}"
            ]
        )
    pinch_span, pinch_axis = viable[0]

    clearance = float(panel_face_offset_m)
    form_closure = clearance >= float(gripper_finger_clearance_m)
    # Hooked, the limit is the handle's own strength; pinched, it is friction
    # on two jaw faces.
    pull_out = (
        float("inf") if form_closure else 2.0 * friction * clamp
    )
    margin = pull_out / required if required > 0 else float("inf")

    # Absence of form closure is a property of the handle, already reported as
    # grasp_mode. It only becomes a finding through the margin it produces.
    findings: list[str] = []
    if margin < float(minimum_margin):
        findings.append(
            "handle_graspability_pull_out_margin_insufficient:"
            f"{margin:.2f}<{float(minimum_margin):.2f}"
        )

    return {
        "schema_version": HANDLE_GRASPABILITY_SCHEMA_VERSION,
        "grasp_mode": GRASP_FORM_CLOSURE if form_closure else GRASP_FRICTION_ONLY,
        "form_closure_available": form_closure,
        "panel_clearance_m": clearance,
        "required_finger_clearance_m": float(gripper_finger_clearance_m),
        "pinch_span_m": pinch_span,
        "pinch_axis": pinch_axis,
        "gripper_stroke_m": stroke,
        "pull_out_capacity_n": pull_out,
        "required_pull_force_n": required,
        "pull_out_margin": margin,
        "margin_sufficient": margin >= float(minimum_margin),
        "findings": sorted(findings),
        "claim_boundary": {
            "clamp_force_is_a_datasheet_figure_not_a_measurement": True,
            "friction_coefficient_is_an_assumption": True,
            "wrist_moment_and_contact_patch_not_modelled": True,
            "static_bound_not_a_grasp_simulation": True,
        },
    }


__all__ = [
    "GRASP_FORM_CLOSURE",
    "GRASP_FRICTION_ONLY",
    "HANDLE_GRASPABILITY_SCHEMA_VERSION",
    "MINIMUM_PULL_OUT_MARGIN",
    "HandleGraspabilityError",
    "evaluate_handle_graspability",
]
