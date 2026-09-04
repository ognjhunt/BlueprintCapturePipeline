"""Derive the qualified task geometry of one passive rigid destination.

The geometry record binds where the whole subject must come to rest relative
to a destination such as a tray.  It is computed, not authored: the subject's
scoring-frame collision bounds come from its exact static qualification and the
task's own scoring transform, the destination interior comes from the SimReady
result, and the containment volume is shrunk by the full oriented subject so a
center-point-only test can never pass.  The record refuses a destination the
subject cannot fit into instead of emitting an empty volume.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_native_arena_episode_compiler import (
    TaskEvaluationNativeArenaEpisodeCompilerError,
    _rotate_xyzw,
    _subject_bounds_in_scoring_frame,
)


SCHEMA_VERSION = "task_evaluation_rigid_destination_geometry.v1"
PRODUCER = "blueprint_pipeline.task_evaluation_rigid_destination_geometry"
STATIC_QUALIFICATION_SCHEMA_VERSION = (
    "task_evaluation_rigid_replacement_static_qualification.v1"
)
SIMREADY_SCHEMA_VERSION = "task_evaluation_passive_destination_simready.v1"
RELATIONS = frozenset({"inside", "on"})
IDENTITY_XYZW = [0.0, 0.0, 0.0, 1.0]
INSERTION_WITHDRAWAL_UNIT_DESTINATION_FRAME = [0.0, 0.0, 1.0]


class RigidDestinationGeometryError(ValueError):
    """The geometry cannot be derived without inventing a fit."""


def _finite_vector(value: Any, *, length: int, code: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RigidDestinationGeometryError(code)
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RigidDestinationGeometryError(code) from exc
    if len(values) != length or not all(math.isfinite(item) for item in values):
        raise RigidDestinationGeometryError(code)
    return values


def _bounds(value: Any, *, code: str) -> tuple[list[float], list[float]]:
    if not isinstance(value, Mapping):
        raise RigidDestinationGeometryError(code)
    lower = _finite_vector(value.get("minimum"), length=3, code=code)
    upper = _finite_vector(value.get("maximum"), length=3, code=code)
    if any(low >= high for low, high in zip(lower, upper, strict=True)):
        raise RigidDestinationGeometryError(code)
    return lower, upper


def _unit_quaternion(value: Any, *, code: str) -> list[float]:
    quaternion = _finite_vector(value, length=4, code=code)
    if not math.isclose(
        sum(item * item for item in quaternion), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RigidDestinationGeometryError(code)
    return quaternion


def _positive(value: Any, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RigidDestinationGeometryError(code)
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RigidDestinationGeometryError(code)
    return number


def _digest(value: Any, *, code: str) -> str:
    text = str(value or "")
    if not text.startswith("sha256:") or len(text) != 71:
        raise RigidDestinationGeometryError(code)
    return text


def _static_structure(
    receipt: Mapping[str, Any], *, identity: Mapping[str, Any], label: str
) -> dict[str, Any]:
    structure = receipt.get("observed_structure") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != STATIC_QUALIFICATION_SCHEMA_VERSION
        or receipt.get("status") != "authored_structure_statically_qualified"
        or not isinstance(structure, Mapping)
    ):
        raise RigidDestinationGeometryError(
            f"rigid_destination_geometry_static_qualification_invalid:{label}"
        )
    if receipt.get("replacement_identity") != dict(identity):
        raise RigidDestinationGeometryError(
            f"rigid_destination_geometry_identity_mismatch:{label}"
        )
    return dict(structure)


def derive_rigid_destination_geometry(
    *,
    subject_identity: Mapping[str, Any],
    destination_identity: Mapping[str, Any],
    relation: str,
    pose_world: Mapping[str, Any],
    subject_static_qualification: Mapping[str, Any],
    subject_static_qualification_digest: str,
    subject_scoring_transform: Mapping[str, Any],
    destination_static_qualification: Mapping[str, Any],
    destination_static_qualification_digest: str,
    destination_simready_result: Mapping[str, Any],
    qualification_limits: Mapping[str, Any],
    subject_orientation_destination_frame_xyzw: Sequence[float] = IDENTITY_XYZW,
) -> dict[str, Any]:
    """Return the digest-bound ``task_evaluation_rigid_destination_geometry.v1``."""

    if relation not in RELATIONS:
        raise RigidDestinationGeometryError("rigid_destination_geometry_relation_invalid")
    if dict(subject_identity) == dict(destination_identity):
        raise RigidDestinationGeometryError(
            "rigid_destination_geometry_identity_mismatch:destination"
        )
    if not isinstance(pose_world, Mapping):
        raise RigidDestinationGeometryError("rigid_destination_geometry_pose_invalid")
    position = _finite_vector(
        pose_world.get("position_world_m"),
        length=3,
        code="rigid_destination_geometry_pose_invalid",
    )
    orientation = _unit_quaternion(
        pose_world.get("orientation_xyzw"),
        code="rigid_destination_geometry_pose_invalid",
    )
    subject_orientation = _unit_quaternion(
        subject_orientation_destination_frame_xyzw,
        code="rigid_destination_geometry_subject_orientation_invalid",
    )
    if not isinstance(qualification_limits, Mapping):
        raise RigidDestinationGeometryError("rigid_destination_geometry_limits_invalid")
    floor_tolerance = _positive(
        qualification_limits.get("maximum_penetration_m"),
        code="rigid_destination_geometry_limits_invalid",
    )
    settle_tolerance = _positive(
        qualification_limits.get("settle_translation_tolerance_m"),
        code="rigid_destination_geometry_limits_invalid",
    )
    subject_structure = _static_structure(
        subject_static_qualification, identity=subject_identity, label="subject"
    )
    destination_structure = _static_structure(
        destination_static_qualification,
        identity=destination_identity,
        label="destination",
    )
    if (
        not isinstance(destination_simready_result, Mapping)
        or destination_simready_result.get("schema_version") != SIMREADY_SCHEMA_VERSION
        or destination_simready_result.get("destination_identity")
        != dict(destination_identity)
    ):
        raise RigidDestinationGeometryError(
            "rigid_destination_geometry_simready_result_invalid"
        )
    try:
        subject_lower, subject_upper = _subject_bounds_in_scoring_frame(
            bounds=subject_structure.get("collision_bounds_body_frame_m"),
            transform=subject_scoring_transform,
        )
    except TaskEvaluationNativeArenaEpisodeCompilerError as exc:
        raise RigidDestinationGeometryError(
            "rigid_destination_geometry_subject_bounds_invalid"
        ) from exc
    interior_lower, interior_upper = _bounds(
        destination_simready_result.get("interior_bounds_body_frame_m"),
        code="rigid_destination_geometry_interior_bounds_invalid",
    )
    destination_lower, destination_upper = _bounds(
        destination_structure.get("collision_bounds_body_frame_m"),
        code="rigid_destination_geometry_destination_bounds_invalid",
    )
    if any(
        interior_lower[axis] < destination_lower[axis] - 1e-9
        or interior_upper[axis] > destination_upper[axis] + 1e-9
        for axis in range(3)
    ):
        raise RigidDestinationGeometryError(
            "rigid_destination_geometry_interior_outside_collision_bounds"
        )
    rigid_paths = destination_structure.get("rigid_body_paths")
    collision_paths = destination_structure.get("collision_prim_paths")
    # The SimReady result names the exact support *rigid body* (contact routes
    # to bodies) and, separately, the exact bottom collision prims retained as
    # evidence.  Both must exist in the static qualification's observed structure.
    support_bodies = destination_simready_result.get("intended_support_prim_paths")
    support_colliders = destination_simready_result.get(
        "intended_support_collision_prim_paths"
    )
    if (
        not isinstance(rigid_paths, list)
        or len(rigid_paths) != 1
        or not isinstance(collision_paths, list)
        or not isinstance(support_bodies, list)
        or not support_bodies
        or not isinstance(support_colliders, list)
        or not support_colliders
    ):
        raise RigidDestinationGeometryError(
            "rigid_destination_geometry_support_structure_invalid"
        )
    if any(path not in rigid_paths for path in support_bodies):
        raise RigidDestinationGeometryError(
            "rigid_destination_geometry_support_body_unknown"
        )
    if any(path not in collision_paths for path in support_colliders):
        raise RigidDestinationGeometryError(
            "rigid_destination_geometry_support_prim_unknown"
        )
    # The tray floor is a contact surface, so containment tolerates the probe's
    # authored maximum penetration below it; the walls tolerate nothing.
    tolerant_interior_lower = [
        interior_lower[0],
        interior_lower[1],
        interior_lower[2] - floor_tolerance,
    ]
    corners = [
        _rotate_xyzw([x, y, z], subject_orientation)
        for x in (subject_lower[0], subject_upper[0])
        for y in (subject_lower[1], subject_upper[1])
        for z in (subject_lower[2], subject_upper[2])
    ]
    oriented_lower = [min(point[axis] for point in corners) for axis in range(3)]
    oriented_upper = [max(point[axis] for point in corners) for axis in range(3)]
    center_lower = [
        tolerant_interior_lower[axis] - oriented_lower[axis] for axis in range(3)
    ]
    center_upper = [interior_upper[axis] - oriented_upper[axis] for axis in range(3)]
    for axis, name in enumerate("xyz"):
        if center_lower[axis] >= center_upper[axis]:
            raise RigidDestinationGeometryError(
                f"rigid_destination_geometry_subject_does_not_fit:{name}"
            )
    # Resting on the floor: the scoring frame sits one oriented half-extent
    # above the interior floor, expressed in the world frame through the pose.
    rest_local = [
        0.0,
        0.0,
        interior_lower[2] - oriented_lower[2],
    ]
    rest_world_z = position[2] + _rotate_xyzw(rest_local, orientation)[2]
    support_tolerance = floor_tolerance + settle_tolerance
    geometry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified",
        "producer": PRODUCER,
        "subject_identity": dict(subject_identity),
        "destination_identity": dict(destination_identity),
        "relation": relation,
        "pose_world": {
            "position_world_m": position,
            "orientation_xyzw": orientation,
        },
        "subject_static_qualification_digest": _digest(
            subject_static_qualification_digest,
            code="rigid_destination_geometry_digest_invalid:subject",
        ),
        "destination_static_qualification_digest": _digest(
            destination_static_qualification_digest,
            code="rigid_destination_geometry_digest_invalid:destination",
        ),
        "subject_collision_bounds_scoring_frame_m": {
            "minimum": subject_lower,
            "maximum": subject_upper,
        },
        "destination_interior_bounds_body_frame_m": {
            "minimum": tolerant_interior_lower,
            "maximum": interior_upper,
        },
        "containment_floor_tolerance_m": floor_tolerance,
        "destination_position_bounds_destination_frame_m": {
            "minimum": center_lower,
            "maximum": center_upper,
        },
        "subject_orientation_destination_frame_xyzw": subject_orientation,
        "support_height_interval_m": [
            rest_world_z - support_tolerance,
            rest_world_z + support_tolerance,
        ],
        "support_height_tolerance_m": support_tolerance,
        "intended_support_prim_paths": [str(path) for path in support_bodies],
        "intended_support_collision_prim_paths": [str(path) for path in support_colliders],
        "insertion_withdrawal_unit_destination_frame": list(
            INSERTION_WITHDRAWAL_UNIT_DESTINATION_FRAME
        ),
        "whole_subject_containment_encoded_by_shrunk_bounds": True,
        "geometry_digest": "",
    }
    geometry["geometry_digest"] = canonical_digest(geometry, digest_field="geometry_digest")
    return geometry


__all__ = [
    "PRODUCER",
    "RigidDestinationGeometryError",
    "SCHEMA_VERSION",
    "derive_rigid_destination_geometry",
]
