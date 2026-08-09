from __future__ import annotations

import math

import pytest

from blueprint_pipeline.articulated_franka_placement import (
    ArticulatedFrankaPlacementError,
    PLACEMENT_SEARCH_SCHEMA_VERSION,
    search_franka_base_placement,
)


HINGE = [1.617248144, 1.829218141, 1.2859256235]
HANDLE_MID = [2.11, 1.86, 1.03]
STATES = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
FRIDGE_KEEPOUT = {
    "label": "replacement_target_footprint",
    "aabb_min": [1.617248144, 1.129218059, 0.0],
    "aabb_max": [2.331180256, 1.829218141, 1.632],
}
SEARCH_REGION = {
    "aabb_min": [0.9, 1.85, 0.0],
    "aabb_max": [3.1, 3.6, 0.0],
}


def _search(**overrides):
    arguments = {
        "hinge_origin_world_m": list(HINGE),
        "joint_axis_world": [0.0, 0.0, 1.0],
        "handle_closed_midpoint_world_m": list(HANDLE_MID),
        "member_vertical_interval_m": [0.939981249, 1.631869998],
        "door_radius_m": 0.714,
        "door_state_angles_degrees": list(STATES),
        "obstacles": [],
        "keepout_aabbs": [dict(FRIDGE_KEEPOUT)],
        "search_region_aabb_m": dict(SEARCH_REGION),
        "grid_resolution_m": 0.1,
    }
    arguments.update(overrides)
    return search_franka_base_placement(**arguments)


def test_open_floor_produces_ranked_reachable_candidates() -> None:
    receipt = _search()

    assert receipt["schema_version"] == PLACEMENT_SEARCH_SCHEMA_VERSION
    assert receipt["status"] == "base_candidates_locally_admissible"
    candidates = receipt["candidates"]
    assert 0 < len(candidates) <= 16
    scores = [row["score"] for row in candidates]
    assert scores == sorted(scores, reverse=True)
    best = candidates[0]
    assert len(best["per_state_reach_m"]) == len(STATES)
    assert all(
        row["franka_base_class_box"]["aabb_min"][2] == 0.0 for row in candidates
    )
    assert receipt["claim_boundary"]["local_geometric_screen_only"] is True
    assert receipt["claim_boundary"]["native_ik_and_contact_required"] is True
    assert receipt["claim_boundary"]["floor_support_native_readback_required"] is True
    assert receipt["receipt_digest"].startswith("sha256:")


def test_high_door_lets_base_stand_under_the_arc() -> None:
    receipt = _search()

    assert receipt["door_corridor_applies_to_base_band"] is False
    for row in receipt["candidates"]:
        x, y = row["base_xy_world_m"]
        assert not (
            FRIDGE_KEEPOUT["aabb_min"][0] - 0.15 <= x <= FRIDGE_KEEPOUT["aabb_max"][0] + 0.15
            and FRIDGE_KEEPOUT["aabb_min"][1] - 0.15 <= y <= FRIDGE_KEEPOUT["aabb_max"][1] + 0.15
        )


def test_low_door_corridor_excludes_base_from_swing_sector() -> None:
    receipt = _search(
        member_vertical_interval_m=[0.0, 1.631869998],
        handle_closed_midpoint_world_m=[2.11, 1.86, 0.9],
    )

    assert receipt["door_corridor_applies_to_base_band"] is True
    closed_angle = math.degrees(
        math.atan2(1.86 - HINGE[1], 2.11 - HINGE[0])
    )
    for row in receipt["candidates"]:
        x, y = row["base_xy_world_m"]
        distance = math.hypot(x - HINGE[0], y - HINGE[1])
        relative = math.degrees(math.atan2(y - HINGE[1], x - HINGE[0])) - closed_angle
        if 0.0 <= relative <= 55.0:
            assert distance > 0.714


def test_obstacle_ring_yields_typed_infeasible_abstention() -> None:
    ring = [
        {
            "obstacle_id": f"ring_{index}",
            "world_aabb_min_m": [
                SEARCH_REGION["aabb_min"][0] - 0.2,
                SEARCH_REGION["aabb_min"][1] - 0.2 + 0.0,
                0.0,
            ],
            "world_aabb_max_m": [
                SEARCH_REGION["aabb_max"][0] + 0.2,
                SEARCH_REGION["aabb_max"][1] + 0.2,
                0.7,
            ],
        }
        for index in range(1)
    ]

    receipt = _search(obstacles=ring)

    assert receipt["status"] == "franka_base_placement_infeasible"
    assert receipt["candidates"] == []
    histogram = receipt["rejection_histogram"]
    assert histogram.get("base_footprint_obstacle", 0) > 0


def test_unreachable_handle_reports_reach_dominated_abstention() -> None:
    receipt = _search(
        search_region_aabb_m={
            "aabb_min": [1.0, 3.2, 0.0],
            "aabb_max": [1.4, 3.55, 0.0],
        }
    )

    assert receipt["status"] == "franka_base_placement_infeasible"
    assert receipt["rejection_histogram"].get("handle_arc_unreachable", 0) > 0


def test_invalid_axis_fails_closed() -> None:
    with pytest.raises(ArticulatedFrankaPlacementError) as excinfo:
        _search(joint_axis_world=[1.0, 0.0, 0.0])

    assert any(
        "joint_axis_not_vertical" in error for error in excinfo.value.errors
    )


def test_shell_wall_triangles_exclude_cells_at_base_height() -> None:
    wall = {
        "obstacle_id": "shell_wall",
        "triangles": [
            [[2.0, 2.5, 0.0], [2.0, 2.5, 2.2], [2.0, 3.6, 0.0]],
            [[2.0, 3.6, 0.0], [2.0, 2.5, 2.2], [2.0, 3.6, 2.2]],
        ],
    }
    ceiling = {
        "obstacle_id": "shell_ceiling",
        "triangles": [
            [[0.0, 0.0, 2.6], [4.0, 0.0, 2.6], [0.0, 5.0, 2.6]],
        ],
    }

    receipt = _search(triangle_shell_obstacles=[wall, ceiling])

    assert receipt["triangle_shell_obstacle_count"] == 2
    assert receipt["rejection_histogram"].get("base_footprint_shell_triangle", 0) > 0
    for row in receipt["candidates"]:
        x, y = row["base_xy_world_m"]
        if y >= 2.5:
            assert abs(x - 2.0) > 0.16


def test_degenerate_vertical_wall_triangle_does_not_reject_the_whole_plane() -> None:
    """A wall triangle projects to a 2D line; zero area is not 'contains all'."""

    wall = {
        "obstacle_id": "degenerate_wall",
        "triangles": [
            [[2.0, 3.55, 0.0], [2.4, 3.55, 0.0], [2.0, 3.55, 2.5]],
        ],
    }

    receipt = _search(triangle_shell_obstacles=[wall])

    assert receipt["status"] == "base_candidates_locally_admissible"
    for row in receipt["candidates"]:
        x, y = row["base_xy_world_m"]
        if 2.0 - 0.16 <= x <= 2.4 + 0.16:
            assert abs(y - 3.55) > 0.16 - 1e-9


def test_point_degenerate_wall_sliver_does_not_reject_the_whole_plane() -> None:
    """SAGE wall slivers project to a single point; zero area contains nothing."""

    sliver = {
        "obstacle_id": "point_degenerate_wall",
        "triangles": [
            [[3.49, 3.70, 2.70], [3.49, 3.70, 0.00], [3.49, 3.70, 0.90]],
        ],
    }

    receipt = _search(triangle_shell_obstacles=[sliver])

    assert receipt["status"] == "base_candidates_locally_admissible"
    assert receipt["rejection_histogram"].get("base_footprint_shell_triangle", 0) == 0
