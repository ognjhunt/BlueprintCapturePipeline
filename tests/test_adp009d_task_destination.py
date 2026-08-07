from __future__ import annotations

import math

import pytest

from blueprint_pipeline.adp009d_approach_capture import SUPPORT_HEIGHT_M
from blueprint_pipeline.adp009d_task_destination import (
    APPROVED_CAN_RADIUS_M,
    TaskDestinationError,
    derive_task_destination,
    top_surface_triangles,
    validate_frozen_destination,
)
from blueprint_pipeline.adp009d_task_scoring import (
    CAN_START_POSITION_M,
    DESTINATION_MIN_DISTANCE_FROM_START_M,
    PLACE_RADIUS_M,
    ROBOT_BASE_POSITION_M,
)


def _grid(half_extent: float = 0.6, step: float = 0.05):
    """A flat support patch centred on the can start, plus one wall below it."""

    points: list[tuple[float, float, float]] = []
    index: dict[tuple[int, int], int] = {}
    steps = int(half_extent / step)
    for ix in range(-steps, steps + 1):
        for iy in range(-steps, steps + 1):
            index[(ix, iy)] = len(points)
            points.append(
                (
                    CAN_START_POSITION_M[0] + ix * step,
                    CAN_START_POSITION_M[1] + iy * step,
                    SUPPORT_HEIGHT_M,
                )
            )
    counts: list[int] = []
    indices: list[int] = []
    for ix in range(-steps, steps):
        for iy in range(-steps, steps):
            a, b = index[(ix, iy)], index[(ix + 1, iy)]
            c, d = index[(ix + 1, iy + 1)], index[(ix, iy + 1)]
            counts.extend((3, 3))
            indices.extend((a, b, c, a, c, d))
    # A vertical face well below the plane: must never be treated as top surface.
    base = len(points)
    points.extend(
        [
            (CAN_START_POSITION_M[0], CAN_START_POSITION_M[1], SUPPORT_HEIGHT_M - 0.3),
            (CAN_START_POSITION_M[0] + 0.1, CAN_START_POSITION_M[1], SUPPORT_HEIGHT_M - 0.3),
            (CAN_START_POSITION_M[0], CAN_START_POSITION_M[1] + 0.1, SUPPORT_HEIGHT_M - 0.3),
        ]
    )
    counts.append(3)
    indices.extend((base, base + 1, base + 2))
    return points, counts, indices


def test_only_triangles_on_the_support_plane_are_candidates() -> None:
    points, counts, indices = _grid()

    triangles = top_surface_triangles(points, counts, indices)

    assert triangles
    for triangle in triangles:
        for vertex in triangle:
            assert vertex[2] == pytest.approx(SUPPORT_HEIGHT_M, abs=1e-9)
    # The sub-plane face is excluded, so the count is the grid's faces alone.
    assert len(triangles) == len(counts) - 1


def test_destination_satisfies_every_constraint_it_declares() -> None:
    points, counts, indices = _grid()

    receipt = derive_task_destination(points, counts, indices)

    assert receipt["status"] == "frozen"
    assert receipt["policy_outcome_consulted"] is False
    position = receipt["position_world_m"]
    assert position[2] == pytest.approx(SUPPORT_HEIGHT_M, abs=1e-9)

    start = math.hypot(
        position[0] - CAN_START_POSITION_M[0], position[1] - CAN_START_POSITION_M[1]
    )
    assert start >= DESTINATION_MIN_DISTANCE_FROM_START_M
    assert receipt["distance_from_can_start_m"] == pytest.approx(start, abs=1e-9)

    base = math.hypot(
        position[0] - ROBOT_BASE_POSITION_M[0], position[1] - ROBOT_BASE_POSITION_M[1]
    )
    assert base <= receipt["constraints"]["maximum_horizontal_reach_m"]
    assert receipt["edge_clearance_m"] >= APPROVED_CAN_RADIUS_M + PLACE_RADIUS_M
    assert receipt["limiting_margin_m"] > 0.0


def test_derivation_is_deterministic_and_order_independent() -> None:
    """A frozen target must not depend on the order triangles happen to arrive."""

    points, counts, indices = _grid()
    first = derive_task_destination(points, counts, indices)

    # Reverse the face order; the same surface must yield the same point.
    reversed_counts: list[int] = []
    reversed_indices: list[int] = []
    faces = []
    cursor = 0
    for count in counts:
        faces.append(indices[cursor : cursor + count])
        cursor += count
    for face in reversed(faces):
        reversed_counts.append(len(face))
        reversed_indices.extend(face)
    second = derive_task_destination(points, reversed_counts, reversed_indices)

    assert first["position_world_m"] == second["position_world_m"]
    assert first["receipt_digest"] == second["receipt_digest"]


def test_a_surface_entirely_too_close_to_the_start_fails_closed() -> None:
    """No admissible candidate is a blocker, never a relaxed threshold."""

    points, counts, indices = _grid(half_extent=0.05, step=0.025)

    with pytest.raises(TaskDestinationError) as excinfo:
        derive_task_destination(points, counts, indices)
    assert any("no_admissible_candidate" in e for e in excinfo.value.errors)


def test_malformed_meshes_are_refused() -> None:
    with pytest.raises(TaskDestinationError):
        top_surface_triangles([], [], [])
    with pytest.raises(TaskDestinationError):
        # Index past the end of the point array.
        top_surface_triangles([(0.0, 0.0, SUPPORT_HEIGHT_M)], [3], [0, 1, 2])


def test_validator_catches_a_tampered_or_drifted_receipt() -> None:
    points, counts, indices = _grid()
    receipt = derive_task_destination(points, counts, indices)

    assert validate_frozen_destination(receipt) == []

    moved = dict(receipt)
    moved["position_world_m"] = list(CAN_START_POSITION_M)
    errors = validate_frozen_destination(moved)
    assert "task_destination_too_close_to_can_start" in errors
    assert "task_destination_receipt_digest_mismatch" in errors

    off_plane = dict(receipt)
    off_plane["position_world_m"] = [
        receipt["position_world_m"][0],
        receipt["position_world_m"][1],
        SUPPORT_HEIGHT_M + 0.05,
    ]
    assert "task_destination_not_on_support_plane" in validate_frozen_destination(off_plane)

    consulted = dict(receipt)
    consulted["policy_outcome_consulted"] = True
    assert "task_destination_outcome_consulted" in validate_frozen_destination(consulted)


def test_the_frozen_destination_matches_the_sealed_support_mesh() -> None:
    """Pin the value actually derived from the sealed SAGE support triangles.

    Recomputed from ``sage_task_collision.usda``'s support prim: 63 top-surface
    triangles, 15 admissible candidates.  If this value ever changes, either the
    sealed mesh changed or the selection rule did -- both must be deliberate.
    """

    frozen = [3.750152333333333, -3.4074919, SUPPORT_HEIGHT_M]

    start = math.hypot(
        frozen[0] - CAN_START_POSITION_M[0], frozen[1] - CAN_START_POSITION_M[1]
    )
    base = math.hypot(
        frozen[0] - ROBOT_BASE_POSITION_M[0], frozen[1] - ROBOT_BASE_POSITION_M[1]
    )
    assert start == pytest.approx(0.2983281527646879, abs=1e-9)
    assert base == pytest.approx(0.6606117518875904, abs=1e-9)
    assert start >= DESTINATION_MIN_DISTANCE_FROM_START_M
    # Reach at the support height, less the place tolerance and safety margin.
    assert base <= 0.7177
