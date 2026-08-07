"""Derive the frozen ADP-009D place destination from SAGE support triangles.

Goal prompt section 3 requires the destination to come from the support surface
itself and to be fixed *before* any policy outcome exists.  Choosing it later --
or choosing it by eye -- would let the target drift toward wherever a policy
happened to put the can, which is exactly the retrospective freedom the sealed
protocol exists to remove.

So the destination is computed, not picked: from the sealed support mesh's own
top-surface triangles, filtered by the constraints the task already fixes, and
resolved by a deterministic rule that cannot depend on run order.  The module
never reads a policy outcome and never touches the simulator.

Measured inputs, all from the sealed derivative and the runtime's own constants:
the support top plane sits at z = 0.526465000, matching ``SUPPORT_HEIGHT_M`` to
nine decimals; its top face spans x 3.2343..4.4682 and y -3.5690..-3.1677.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .adp009d_approach_capture import SUPPORT_HEIGHT_M
from .adp009d_task_scoring import (
    CAN_START_POSITION_M,
    DESTINATION_MIN_DISTANCE_FROM_START_M,
    FRANKA_REACH_M,
    PLACE_RADIUS_M,
    ROBOT_BASE_POSITION_M,
)
from .decision_evidence_contracts import canonical_digest

DESTINATION_SCHEMA_VERSION = "adp009d_task_destination.v1"

# A triangle counts as top surface when all three vertices sit on the support
# plane.  The tolerance is tight because the plane is exact in the sealed mesh.
TOP_SURFACE_PLANE_TOLERANCE_M = 1.0e-6

# Keep the whole place tolerance inside the arm's reach, not just its centre:
# a destination reachable at the centre but not at its edge would make the
# frozen 0.05 m tolerance partly unreachable and quietly bias every score.
REACH_SAFETY_MARGIN_M = 0.05

# The can must land fully on the support, so its footprint plus the place
# tolerance must clear the surface edge.  Radius measured from the sealed
# SimReady can's collider extent.
APPROVED_CAN_RADIUS_M = 0.033

BLOCKER_NO_TOP_SURFACE = "task_destination_no_top_surface_triangles"
BLOCKER_NO_ADMISSIBLE_CANDIDATE = "task_destination_no_admissible_candidate"
BLOCKER_MESH_INVALID = "task_destination_support_mesh_invalid"


class TaskDestinationError(ValueError):
    """Fail-closed destination derivation errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def _horizontal(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def top_surface_triangles(
    points: Sequence[Sequence[float]],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    *,
    plane_z: float = SUPPORT_HEIGHT_M,
    tolerance_m: float = TOP_SURFACE_PLANE_TOLERANCE_M,
) -> list[tuple[tuple[float, float, float], ...]]:
    """Triangles whose three vertices all lie on the support plane."""

    if not points or not face_vertex_counts or not face_vertex_indices:
        raise TaskDestinationError([BLOCKER_MESH_INVALID])
    triangles: list[tuple[tuple[float, float, float], ...]] = []
    cursor = 0
    for count in face_vertex_counts:
        count = int(count)
        if count < 3:
            cursor += count
            continue
        indices = face_vertex_indices[cursor : cursor + count]
        cursor += count
        if len(indices) != count:
            raise TaskDestinationError([BLOCKER_MESH_INVALID])
        try:
            vertices = [tuple(float(v) for v in points[int(i)]) for i in indices]
        except (IndexError, TypeError, ValueError) as exc:
            raise TaskDestinationError([BLOCKER_MESH_INVALID]) from exc
        if any(len(v) != 3 for v in vertices):
            raise TaskDestinationError([BLOCKER_MESH_INVALID])
        if all(abs(v[2] - float(plane_z)) <= float(tolerance_m) for v in vertices):
            # Fan-triangulate anything larger than a triangle.
            for offset in range(1, count - 1):
                triangles.append((vertices[0], vertices[offset], vertices[offset + 1]))
    return triangles


def derive_task_destination(
    points: Sequence[Sequence[float]],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    *,
    can_start_position_m: Sequence[float] = CAN_START_POSITION_M,
    robot_base_position_m: Sequence[float] = ROBOT_BASE_POSITION_M,
) -> dict[str, Any]:
    """Compute the frozen place destination, outcome-blind and deterministic.

    Candidates are the centroids of the support's own top-surface triangles.
    A candidate is admissible when it is far enough from the sealed can start
    to constitute a translation, close enough to the base that the entire place
    tolerance is reachable, and far enough from the surface edge that the placed
    can rests fully supported.

    Among admissible candidates the winner maximises the smallest of its
    margins.  Ties break on the exact coordinate ordering, so the result cannot
    depend on triangle order, and a rerun on the same mesh returns the same
    point.
    """

    triangles = top_surface_triangles(
        points, face_vertex_counts, face_vertex_indices
    )
    if not triangles:
        raise TaskDestinationError([BLOCKER_NO_TOP_SURFACE])

    # Surface extent, used for the edge-clearance margin.
    xs = [v[0] for tri in triangles for v in tri]
    ys = [v[1] for tri in triangles for v in tri]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)

    base_height = float(robot_base_position_m[2])
    vertical = SUPPORT_HEIGHT_M - base_height
    reach_squared = FRANKA_REACH_M**2 - vertical**2
    if reach_squared <= 0.0:
        raise TaskDestinationError([BLOCKER_NO_ADMISSIBLE_CANDIDATE])
    horizontal_reach = math.sqrt(reach_squared) - REACH_SAFETY_MARGIN_M - PLACE_RADIUS_M
    edge_clearance = APPROVED_CAN_RADIUS_M + PLACE_RADIUS_M

    admissible: list[dict[str, Any]] = []
    for triangle in triangles:
        centroid = (
            sum(v[0] for v in triangle) / 3.0,
            sum(v[1] for v in triangle) / 3.0,
            SUPPORT_HEIGHT_M,
        )
        start_distance = _horizontal(centroid, can_start_position_m)
        base_distance = _horizontal(centroid, robot_base_position_m)
        edge_margin = min(
            centroid[0] - x_min,
            x_max - centroid[0],
            centroid[1] - y_min,
            y_max - centroid[1],
        )
        if start_distance < DESTINATION_MIN_DISTANCE_FROM_START_M:
            continue
        if base_distance > horizontal_reach:
            continue
        if edge_margin < edge_clearance:
            continue
        admissible.append(
            {
                "position_world_m": [float(v) for v in centroid],
                "distance_from_can_start_m": start_distance,
                "distance_from_robot_base_m": base_distance,
                "edge_clearance_m": edge_margin,
                "limiting_margin_m": min(
                    start_distance - DESTINATION_MIN_DISTANCE_FROM_START_M,
                    horizontal_reach - base_distance,
                    edge_margin - edge_clearance,
                ),
            }
        )

    if not admissible:
        raise TaskDestinationError([BLOCKER_NO_ADMISSIBLE_CANDIDATE])

    admissible.sort(
        key=lambda row: (
            -row["limiting_margin_m"],
            row["position_world_m"][0],
            row["position_world_m"][1],
        )
    )
    winner = admissible[0]

    receipt: dict[str, Any] = {
        "schema_version": DESTINATION_SCHEMA_VERSION,
        "status": "frozen",
        "position_world_m": winner["position_world_m"],
        "derived_from": "sage_support_top_surface_triangle_centroids",
        "top_surface_triangle_count": len(triangles),
        "admissible_candidate_count": len(admissible),
        "distance_from_can_start_m": winner["distance_from_can_start_m"],
        "distance_from_robot_base_m": winner["distance_from_robot_base_m"],
        "edge_clearance_m": winner["edge_clearance_m"],
        "limiting_margin_m": winner["limiting_margin_m"],
        "constraints": {
            "support_plane_z_m": SUPPORT_HEIGHT_M,
            "minimum_distance_from_can_start_m": DESTINATION_MIN_DISTANCE_FROM_START_M,
            "maximum_horizontal_reach_m": horizontal_reach,
            "required_edge_clearance_m": edge_clearance,
            "place_radius_m": PLACE_RADIUS_M,
            "reach_safety_margin_m": REACH_SAFETY_MARGIN_M,
            "approved_can_radius_m": APPROVED_CAN_RADIUS_M,
        },
        "policy_outcome_consulted": False,
        "selection_rule": (
            "maximise the smallest constraint margin; ties break on x then y so "
            "the result cannot depend on triangle order"
        ),
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def validate_frozen_destination(receipt: Mapping[str, Any]) -> list[str]:
    """Check a retained destination receipt still satisfies its own constraints."""

    errors: list[str] = []
    if receipt.get("schema_version") != DESTINATION_SCHEMA_VERSION:
        errors.append("task_destination_schema_version_unexpected")
    if receipt.get("status") != "frozen":
        errors.append("task_destination_not_frozen")
    if receipt.get("policy_outcome_consulted") is not False:
        errors.append("task_destination_outcome_consulted")
    position = receipt.get("position_world_m")
    if not isinstance(position, Sequence) or len(position) != 3:
        errors.append("task_destination_position_invalid")
        return sorted(set(errors))
    if abs(float(position[2]) - SUPPORT_HEIGHT_M) > TOP_SURFACE_PLANE_TOLERANCE_M:
        errors.append("task_destination_not_on_support_plane")
    if (
        _horizontal(position, CAN_START_POSITION_M)
        < DESTINATION_MIN_DISTANCE_FROM_START_M
    ):
        errors.append("task_destination_too_close_to_can_start")
    expected = canonical_digest(dict(receipt), digest_field="receipt_digest")
    if receipt.get("receipt_digest") != expected:
        errors.append("task_destination_receipt_digest_mismatch")
    return sorted(set(errors))


__all__ = [
    "APPROVED_CAN_RADIUS_M",
    "DESTINATION_SCHEMA_VERSION",
    "REACH_SAFETY_MARGIN_M",
    "TOP_SURFACE_PLANE_TOLERANCE_M",
    "TaskDestinationError",
    "derive_task_destination",
    "top_surface_triangles",
    "validate_frozen_destination",
]
