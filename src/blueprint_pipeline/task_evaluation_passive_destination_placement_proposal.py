"""Propose where a passive destination rests on the subject's support.

The destination pose is authored before any policy outcome exists and is
computed, not picked: the tray is aligned with the support's long axis,
centered across its short axis, and placed beside the subject's observed
bounds with an authored clearance gap on whichever side has more free support
length.  The proposal is only a candidate: native placement qualification,
camera coverage, and robot reachability are later gates and are declared
unestablished here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_passive_destination_placement_proposal.v1"
NATIVE_PROBE_SCHEMA_VERSION = (
    "task_evaluation_rigid_destination_native_probe_configuration.v1"
)
SUPPORT_PLANE_SCHEMA_VERSION = "task_evaluation_support_plane_input.v1"
SUBJECT_SELECTION_SCHEMA_VERSION = "task_evaluation_source_object_selection.v1"
STATIC_QUALIFICATION_SCHEMA_VERSION = (
    "task_evaluation_rigid_replacement_static_qualification.v1"
)
DEFAULT_QUALIFICATION_LIMITS: dict[str, Any] = {
    "maximum_penetration_m": 0.001,
    "minimum_support_contact_force_n": 1.0,
    "maximum_forbidden_contact_force_n": 0.5,
    "settle_translation_tolerance_m": 0.002,
    "settle_rotation_tolerance_rad": 0.01,
    "reset_translation_tolerance_m": 0.002,
    "reset_rotation_tolerance_rad": 0.01,
    "minimum_camera_pixels": {"external": 400, "wrist": 200, "overview": 400},
}
DEFAULT_SETTLE_SAMPLE_COUNT = 3
DEFAULT_SETTLE_STEPS_PER_SAMPLE = 60
SUBJECT_REST_TOLERANCE_M = 0.005
_AXES = ("x", "y")


class PassiveDestinationPlacementProposalError(ValueError):
    """The destination cannot be placed without guessing."""


def _vector(value: Any, *, length: int, code: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PassiveDestinationPlacementProposalError(code)
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise PassiveDestinationPlacementProposalError(code) from exc
    if len(values) != length or not all(math.isfinite(item) for item in values):
        raise PassiveDestinationPlacementProposalError(code)
    return values


def _positive(value: Any, *, code: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PassiveDestinationPlacementProposalError(code)
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or (number == 0.0 and not allow_zero):
        raise PassiveDestinationPlacementProposalError(code)
    return number


def derive_passive_destination_placement_proposal(
    *,
    support_plane: Mapping[str, Any],
    subject_selection: Mapping[str, Any],
    destination_identity: Mapping[str, Any],
    destination_static_qualification: Mapping[str, Any],
    clearance_gap_m: float,
    support_edge_margin_m: float,
    qualification_limits: Mapping[str, Any] | None = None,
    settle_sample_count: int = DEFAULT_SETTLE_SAMPLE_COUNT,
    settle_steps_per_sample: int = DEFAULT_SETTLE_STEPS_PER_SAMPLE,
) -> dict[str, Any]:
    """Return the digest-bound destination placement proposal."""

    gap = _positive(clearance_gap_m, code="passive_destination_placement_gap_invalid")
    margin = _positive(
        support_edge_margin_m,
        code="passive_destination_placement_margin_invalid",
        allow_zero=True,
    )
    if (
        not isinstance(support_plane, Mapping)
        or support_plane.get("schema_version") != SUPPORT_PLANE_SCHEMA_VERSION
        or support_plane.get("status") != "frozen_candidate_pending_production_validation"
        or not str(support_plane.get("sage_prim_path") or "").startswith("/")
    ):
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_support_plane_invalid"
        )
    support_lower = _vector(
        support_plane.get("bounds_min_xyz_m"),
        length=3,
        code="passive_destination_placement_support_plane_invalid",
    )
    support_upper = _vector(
        support_plane.get("bounds_max_xyz_m"),
        length=3,
        code="passive_destination_placement_support_plane_invalid",
    )
    if any(low >= high for low, high in zip(support_lower, support_upper, strict=True)):
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_support_plane_invalid"
        )
    try:
        support_top = float(support_plane.get("top_z_m"))
    except (TypeError, ValueError) as exc:
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_support_top_invalid"
        ) from exc
    if not math.isfinite(support_top) or not math.isclose(
        support_top, support_upper[2], rel_tol=0.0, abs_tol=1e-6
    ):
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_support_top_invalid"
        )
    if (
        not isinstance(subject_selection, Mapping)
        or subject_selection.get("schema_version") != SUBJECT_SELECTION_SCHEMA_VERSION
        or subject_selection.get("status") != "frozen_before_scene_configuration_run"
    ):
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_subject_selection_invalid"
        )
    subject_lower = _vector(
        subject_selection.get("aabb_min_xyz_m"),
        length=3,
        code="passive_destination_placement_subject_selection_invalid",
    )
    subject_upper = _vector(
        subject_selection.get("aabb_max_xyz_m"),
        length=3,
        code="passive_destination_placement_subject_selection_invalid",
    )
    if any(low >= high for low, high in zip(subject_lower, subject_upper, strict=True)):
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_subject_selection_invalid"
        )
    if any(
        subject_lower[axis] < support_lower[axis] - 1e-6
        or subject_upper[axis] > support_upper[axis] + 1e-6
        for axis in range(2)
    ):
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_subject_outside_support"
        )
    if abs(subject_lower[2] - support_top) > SUBJECT_REST_TOLERANCE_M:
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_subject_not_on_support"
        )
    structure = (
        destination_static_qualification.get("observed_structure")
        if isinstance(destination_static_qualification, Mapping)
        else None
    )
    if (
        not isinstance(destination_static_qualification, Mapping)
        or destination_static_qualification.get("schema_version")
        != STATIC_QUALIFICATION_SCHEMA_VERSION
        or destination_static_qualification.get("status")
        != "authored_structure_statically_qualified"
        or not isinstance(structure, Mapping)
    ):
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_static_qualification_invalid"
        )
    if destination_static_qualification.get("replacement_identity") != dict(
        destination_identity
    ):
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_identity_mismatch"
        )
    bounds = structure.get("collision_bounds_body_frame_m")
    if not isinstance(bounds, Mapping):
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_static_qualification_invalid"
        )
    body_lower = _vector(
        bounds.get("minimum"),
        length=3,
        code="passive_destination_placement_static_qualification_invalid",
    )
    body_upper = _vector(
        bounds.get("maximum"),
        length=3,
        code="passive_destination_placement_static_qualification_invalid",
    )
    body_extents = [body_upper[axis] - body_lower[axis] for axis in range(3)]
    if any(extent <= 0.0 for extent in body_extents):
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_static_qualification_invalid"
        )
    if abs(body_lower[2]) > 1e-6:
        # The pose is the body origin; the base must sit at body z = 0 so the
        # tray rests exactly on the support top.
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_body_origin_not_at_base"
        )
    support_extents = [support_upper[axis] - support_lower[axis] for axis in range(2)]
    long_axis = 1 if support_extents[1] >= support_extents[0] else 0
    short_axis = 1 - long_axis
    body_long_axis = 1 if body_extents[1] >= body_extents[0] else 0
    yaw = 0.0 if body_long_axis == long_axis else math.pi / 2.0
    along = max(body_extents[0], body_extents[1])
    across = min(body_extents[0], body_extents[1])
    if across + 2.0 * margin > support_extents[short_axis] + 1e-9:
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_footprint_exceeds_support_width"
        )
    free_positive = support_upper[long_axis] - subject_upper[long_axis] - gap - margin
    free_negative = subject_lower[long_axis] - support_lower[long_axis] - gap - margin
    if free_positive >= along and free_positive >= free_negative:
        side, center_along = "positive", subject_upper[long_axis] + gap + along / 2.0
    elif free_negative >= along:
        side, center_along = "negative", subject_lower[long_axis] - gap - along / 2.0
    else:
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_no_free_support_length"
        )
    center_across = (support_lower[short_axis] + support_upper[short_axis]) / 2.0
    position = [0.0, 0.0, support_top]
    position[long_axis] = center_along
    position[short_axis] = center_across
    half = [0.0, 0.0]
    half[long_axis] = along / 2.0
    half[short_axis] = across / 2.0
    footprint = {
        "minimum": [position[0] - half[0], position[1] - half[1], support_top],
        "maximum": [position[0] + half[0], position[1] + half[1], support_top + body_extents[2]],
    }
    limits = dict(DEFAULT_QUALIFICATION_LIMITS)
    if qualification_limits is not None:
        if not isinstance(qualification_limits, Mapping) or set(qualification_limits) - set(
            DEFAULT_QUALIFICATION_LIMITS
        ):
            raise PassiveDestinationPlacementProposalError(
                "passive_destination_placement_probe_limits_invalid"
            )
        limits.update({key: qualification_limits[key] for key in qualification_limits})
    if (
        isinstance(settle_sample_count, bool)
        or not isinstance(settle_sample_count, int)
        or settle_sample_count < 3
        or isinstance(settle_steps_per_sample, bool)
        or not isinstance(settle_steps_per_sample, int)
        or settle_steps_per_sample < 1
    ):
        raise PassiveDestinationPlacementProposalError(
            "passive_destination_placement_probe_samples_invalid"
        )
    proposal: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "proposed_pending_native_placement_qualification",
        "scene_id": str(support_plane.get("scene_id") or ""),
        "destination_identity": dict(destination_identity),
        "subject_publisher_instance_id": str(
            subject_selection.get("publisher_instance_id") or ""
        ),
        "support_publisher_instance_id": str(
            support_plane.get("publisher_instance_id") or ""
        ),
        "support_sage_prim_path": str(support_plane["sage_prim_path"]),
        "pose_world": {
            "position_world_m": position,
            "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
        },
        "footprint_world_m": footprint,
        "derivation": {
            "rule": "align_with_support_long_axis_center_short_axis_beside_subject",
            "long_axis": _AXES[long_axis],
            "short_axis": _AXES[short_axis],
            "side": side,
            "yaw_rad": yaw,
            "clearance_gap_m": gap,
            "support_edge_margin_m": margin,
            "destination_extents_body_frame_m": body_extents,
            "free_support_length_positive_m": free_positive,
            "free_support_length_negative_m": free_negative,
            "subject_aabb_world_m": {"minimum": subject_lower, "maximum": subject_upper},
            "support_bounds_world_m": {"minimum": support_lower, "maximum": support_upper},
        },
        "native_probe": {
            "schema_version": NATIVE_PROBE_SCHEMA_VERSION,
            "placement_support_scene_prim_paths": [str(support_plane["sage_prim_path"])],
            "qualification_limits": limits,
            "settle_sample_count": settle_sample_count,
            "settle_steps_per_sample": settle_steps_per_sample,
        },
        "claim_boundary": {
            "native_placement_qualified": False,
            "camera_visibility_established": False,
            "robot_reachability_established": False,
            "policy_outcomes_consulted": False,
        },
        "proposal_digest": "",
    }
    proposal["proposal_digest"] = canonical_digest(proposal, digest_field="proposal_digest")
    return proposal


__all__ = [
    "DEFAULT_QUALIFICATION_LIMITS",
    "PassiveDestinationPlacementProposalError",
    "SCHEMA_VERSION",
    "derive_passive_destination_placement_proposal",
]
