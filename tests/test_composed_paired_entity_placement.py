from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import Any

from blueprint_pipeline.composed_paired_entity_placement import (
    NO_FIT_BLOCKER,
    SEARCH_SPACE_BLOCKER,
    plan_composed_paired_entity_placement,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _inputs(*, seed: int = 7) -> dict[str, Any]:
    return {
        "support_regions": [
            {
                "support_region_id": "observed_support",
                "aabb_min_m": [0.0, 0.0, 0.0],
                "aabb_max_m": [3.0, 2.0, 0.0],
                "supports_entities": True,
                "supports_robot_base": True,
            }
        ],
        "obstacle_aabbs": [],
        "entity_specs": [
            {
                "entity_id": "destination",
                "footprint_xy_m": [0.3, 0.3],
                "height_m": 0.4,
            },
            {
                "entity_id": "movable",
                "footprint_xy_m": [0.2, 0.2],
                "height_m": 0.05,
            },
        ],
        "canonical_task_centers_m": [[0.0, 0.0, 0.0]],
        "robot_spec": {
            "base_footprint_xy_m": [0.4, 0.4],
            "base_clearance_height_m": 0.25,
            "reach_annulus_m": [0.4, 1.6],
        },
        "minimum_separations_m": {
            "canonical_region": 0.6,
            "entity_entity": 0.1,
            "entity_obstacle": 0.05,
            "robot_entity": 0.05,
            "robot_obstacle": 0.05,
            "support_edge": 0.05,
        },
        "grid_spacing_m": 0.5,
        "frozen_seed": seed,
    }


def _aabb_distance_xy(first: dict[str, Any], second: dict[str, Any]) -> float:
    squared = 0.0
    for axis in range(2):
        gap = max(
            second["aabb_min_m"][axis] - first["aabb_max_m"][axis],
            first["aabb_min_m"][axis] - second["aabb_max_m"][axis],
            0.0,
        )
        squared += gap * gap
    return math.sqrt(squared)


def _point_aabb_distance_xy(point: list[float], placement: dict[str, Any]) -> float:
    squared = 0.0
    for axis in range(2):
        gap = max(
            placement["aabb_min_m"][axis] - point[axis],
            point[axis] - placement["aabb_max_m"][axis],
            0.0,
        )
        squared += gap * gap
    return math.sqrt(squared)


def test_planner_is_deterministic_and_canonicalises_entity_order() -> None:
    inputs = _inputs(seed=2026081001)

    first = plan_composed_paired_entity_placement(**inputs)
    reordered = deepcopy(inputs)
    reordered["entity_specs"].reverse()
    second = plan_composed_paired_entity_placement(**reordered)

    assert first == second
    assert first["receipt_digest"] == canonical_digest(
        first, digest_field="receipt_digest"
    )
    assert first["status"] == "geometry_plausibility_candidate_selected"
    assert first["claim_boundary"]["geometry_plausibility_only"] is True
    assert first["claim_boundary"]["native_ik_qualified"] is False
    assert first["pending_native_gates"] == [
        "native_full_phase_ik",
        "native_contact_and_stable_support",
        "native_policy_and_review_camera_visibility",
    ]


def test_selected_pair_excludes_canonical_region_and_meets_separation() -> None:
    inputs = _inputs(seed=11)
    receipt = plan_composed_paired_entity_placement(**inputs)
    placements = receipt["selection"]["entity_placements"]

    assert len(placements) == 2
    assert all(
        _point_aabb_distance_xy([0.0, 0.0], placement) >= 0.6 - 1e-9
        for placement in placements
    )
    assert _aabb_distance_xy(placements[0], placements[1]) >= 0.1 - 1e-9
    assert receipt["enumeration"]["entity_placements"]["destination"][
        "rejection_reason_counts"
    ]["entity_inside_canonical_exclusion"] > 0


def test_obstacle_collisions_are_rejected_and_never_selected() -> None:
    inputs = _inputs(seed=13)
    obstacle = {
        "obstacle_id": "registered_obstacle",
        "aabb_min_m": [1.35, 0.35, 0.0],
        "aabb_max_m": [1.65, 0.65, 0.8],
    }
    inputs["obstacle_aabbs"] = [obstacle]

    receipt = plan_composed_paired_entity_placement(**inputs)

    assert receipt["status"] == "geometry_plausibility_candidate_selected"
    entity_summaries = receipt["enumeration"]["entity_placements"]
    assert all(
        summary["rejection_reason_counts"]["entity_obstacle_clearance"] > 0
        for summary in entity_summaries.values()
    )
    assert receipt["enumeration"]["robot_base_placements"][
        "rejection_reason_counts"
    ]["robot_base_obstacle_clearance"] > 0
    for selected in [
        *receipt["selection"]["entity_placements"],
        receipt["selection"]["robot_base_placement"],
    ]:
        assert _aabb_distance_xy(selected, obstacle) > 0.0


def test_no_fit_returns_typed_digest_bound_blocker_with_rejection_counts() -> None:
    inputs = _inputs(seed=17)
    inputs["obstacle_aabbs"] = [
        {
            "obstacle_id": "fully_blocking_obstacle",
            "aabb_min_m": [-1.0, -1.0, -1.0],
            "aabb_max_m": [4.0, 3.0, 1.0],
        }
    ]

    receipt = plan_composed_paired_entity_placement(**inputs)

    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == [NO_FIT_BLOCKER]
    assert receipt["blocking_stage"] == "entity_placement:destination"
    summary = receipt["enumeration"]["entity_placements"]["destination"]
    assert summary["grid_points_considered"] > 0
    assert summary["admissible_count"] == 0
    assert summary["rejection_reason_counts"]["entity_obstacle_clearance"] == summary[
        "grid_points_considered"
    ]
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_frozen_seed_hash_resolves_one_canonical_selected_index() -> None:
    first = plan_composed_paired_entity_placement(**_inputs(seed=1))
    second = plan_composed_paired_entity_placement(**_inputs(seed=2))

    assert first["selection"]["admissible_candidate_count"] > 2
    for frozen_seed, receipt in ((1, first), (2, second)):
        count = receipt["selection"]["admissible_candidate_count"]
        request_digest = receipt["request"]["request_digest"]
        material = (
            "composed_paired_entity_placement_request.v1\n"
            f"{frozen_seed}\n{request_digest}\n{count}"
        ).encode("utf-8")
        digest = hashlib.sha256(material).hexdigest()
        assert receipt["selection"]["selection_seed_digest"] == f"sha256:{digest}"
        assert receipt["selection"]["selected_index"] == int(digest, 16) % count
    assert (
        first["selection"]["selection_digest"]
        != second["selection"]["selection_digest"]
    )
    assert (
        first["enumeration"]["paired_entity_robot_candidates"]["admissible_count"]
        == second["enumeration"]["paired_entity_robot_candidates"]["admissible_count"]
    )


def test_enumeration_cap_fails_closed_before_large_pair_product() -> None:
    inputs = _inputs(seed=19)
    inputs["maximum_combination_count"] = 10

    receipt = plan_composed_paired_entity_placement(**inputs)

    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == [SEARCH_SPACE_BLOCKER]
    assert receipt["blocking_stage"] == "bounded_entity_pair_enumeration"
    assert receipt["enumeration"]["entity_pairs"]["potential_count"] > 10
    assert receipt["enumeration"]["entity_pairs"]["pairs_considered"] == 0
