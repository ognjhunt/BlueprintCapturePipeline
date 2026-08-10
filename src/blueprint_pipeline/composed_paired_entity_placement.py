"""Deterministic local placement screen for one composed two-entity task.

The planner enumerates a global, origin-anchored metric grid over observed
support AABBs.  It accepts exactly two axis-aligned entity footprints and one
robot-base footprint, rejects canonical-task proximity, obstacle collisions,
pair collisions, implausible reach, and insufficient clearance, then selects a
seed-resolved candidate from the canonical-sorted admissible rows.

This module is deliberately simulator-independent.  An accepted row is only a
geometry-plausibility candidate; native IK, contact, and camera gates remain
mandatory before an evaluation cell can be admitted.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .decision_evidence_contracts import canonical_digest


REQUEST_SCHEMA_VERSION = "composed_paired_entity_placement_request.v1"
RECEIPT_SCHEMA_VERSION = "composed_paired_entity_placement_receipt.v1"
NO_FIT_BLOCKER = "typed_composed_relocation_placement_blocker"
SEARCH_SPACE_BLOCKER = "typed_composed_relocation_search_space_exceeded"
MAX_COMBINATIONS = 2_000_000
_PRECISION = 9
_TOLERANCE = 1e-9


class ComposedPairedEntityPlacementError(ValueError):
    """Stable, sorted planner input-contract failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


@dataclass(frozen=True)
class _Support:
    support_region_id: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    supports_entities: bool
    supports_robot_base: bool


@dataclass(frozen=True)
class _Obstacle:
    obstacle_id: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


@dataclass(frozen=True)
class _Entity:
    entity_id: str
    footprint: tuple[float, float]
    height: float


@dataclass(frozen=True)
class _Robot:
    footprint: tuple[float, float]
    clearance_height: float
    reach_minimum: float
    reach_maximum: float


@dataclass(frozen=True)
class _Placement:
    subject_id: str
    support_region_id: str
    center: tuple[float, float, float]
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


def _identifier(value: Any, error: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text or len(text) > 192:
        raise ComposedPairedEntityPlacementError([error])
    return text


def _number(value: Any, error: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ComposedPairedEntityPlacementError([error])
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ComposedPairedEntityPlacementError([error]) from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ComposedPairedEntityPlacementError([error])
    normalised = 0.0 if abs(number) < 0.5 * 10**-_PRECISION else round(number, _PRECISION)
    if positive and normalised <= 0.0:
        raise ComposedPairedEntityPlacementError([error])
    return normalised


def _nonnegative(value: Any, error: str) -> float:
    number = _number(value, error)
    if number < 0.0:
        raise ComposedPairedEntityPlacementError([error])
    return number


def _vector(
    value: Any,
    length: int,
    error: str,
    *,
    positive: bool = False,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ComposedPairedEntityPlacementError([error])
    return tuple(_number(item, error, positive=positive) for item in value)


def _aabb(value: Mapping[str, Any], prefix: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    minimum = _vector(value.get("aabb_min_m"), 3, f"{prefix}_aabb_invalid")
    maximum = _vector(value.get("aabb_max_m"), 3, f"{prefix}_aabb_invalid")
    if any(minimum[axis] > maximum[axis] for axis in range(3)):
        raise ComposedPairedEntityPlacementError([f"{prefix}_aabb_invalid"])
    return minimum, maximum


def _boolean(value: Any, error: str) -> bool:
    if not isinstance(value, bool):
        raise ComposedPairedEntityPlacementError([error])
    return value


def _normalise_request(
    *,
    support_regions: Sequence[Mapping[str, Any]],
    obstacle_aabbs: Sequence[Mapping[str, Any]],
    entity_specs: Sequence[Mapping[str, Any]],
    canonical_task_centers_m: Sequence[Sequence[float]],
    robot_spec: Mapping[str, Any],
    minimum_separations_m: Mapping[str, Any],
    grid_spacing_m: float,
    frozen_seed: int,
    maximum_combination_count: int,
) -> tuple[
    dict[str, Any],
    tuple[_Support, ...],
    tuple[_Obstacle, ...],
    tuple[_Entity, ...],
    tuple[tuple[float, float, float], ...],
    _Robot,
    dict[str, float],
]:
    if isinstance(support_regions, (str, bytes)) or not isinstance(support_regions, Sequence):
        raise ComposedPairedEntityPlacementError(["support_regions_invalid"])
    if isinstance(obstacle_aabbs, (str, bytes)) or not isinstance(obstacle_aabbs, Sequence):
        raise ComposedPairedEntityPlacementError(["obstacle_aabbs_invalid"])
    if isinstance(entity_specs, (str, bytes)) or not isinstance(entity_specs, Sequence):
        raise ComposedPairedEntityPlacementError(["entity_specs_invalid"])
    if len(entity_specs) != 2:
        raise ComposedPairedEntityPlacementError(["exactly_two_entity_specs_required"])
    if (
        isinstance(canonical_task_centers_m, (str, bytes))
        or not isinstance(canonical_task_centers_m, Sequence)
        or not canonical_task_centers_m
    ):
        raise ComposedPairedEntityPlacementError(["canonical_task_centers_invalid"])
    if not isinstance(robot_spec, Mapping):
        raise ComposedPairedEntityPlacementError(["robot_spec_invalid"])
    if not isinstance(minimum_separations_m, Mapping):
        raise ComposedPairedEntityPlacementError(["minimum_separations_invalid"])
    if isinstance(frozen_seed, bool) or not isinstance(frozen_seed, int) or not 0 <= frozen_seed < 2**64:
        raise ComposedPairedEntityPlacementError(["frozen_seed_not_uint64"])
    if (
        isinstance(maximum_combination_count, bool)
        or not isinstance(maximum_combination_count, int)
        or maximum_combination_count <= 0
        or maximum_combination_count > MAX_COMBINATIONS
    ):
        raise ComposedPairedEntityPlacementError(["maximum_combination_count_invalid"])

    supports: list[_Support] = []
    support_ids: set[str] = set()
    for row in support_regions:
        if not isinstance(row, Mapping):
            raise ComposedPairedEntityPlacementError(["support_region_invalid"])
        support_id = _identifier(row.get("support_region_id"), "support_region_id_invalid")
        if support_id in support_ids:
            raise ComposedPairedEntityPlacementError(["support_region_id_duplicate"])
        support_ids.add(support_id)
        minimum, maximum = _aabb(row, "support_region")
        if maximum[0] <= minimum[0] or maximum[1] <= minimum[1]:
            raise ComposedPairedEntityPlacementError(["support_region_planar_area_invalid"])
        supports.append(
            _Support(
                support_region_id=support_id,
                minimum=minimum,
                maximum=maximum,
                supports_entities=_boolean(
                    row.get("supports_entities"), "support_region_entity_role_invalid"
                ),
                supports_robot_base=_boolean(
                    row.get("supports_robot_base"), "support_region_robot_role_invalid"
                ),
            )
        )
    if not any(row.supports_entities for row in supports):
        raise ComposedPairedEntityPlacementError(["entity_support_region_missing"])
    if not any(row.supports_robot_base for row in supports):
        raise ComposedPairedEntityPlacementError(["robot_support_region_missing"])

    obstacles: list[_Obstacle] = []
    obstacle_ids: set[str] = set()
    for row in obstacle_aabbs:
        if not isinstance(row, Mapping):
            raise ComposedPairedEntityPlacementError(["obstacle_aabb_invalid"])
        obstacle_id = _identifier(row.get("obstacle_id"), "obstacle_id_invalid")
        if obstacle_id in obstacle_ids:
            raise ComposedPairedEntityPlacementError(["obstacle_id_duplicate"])
        obstacle_ids.add(obstacle_id)
        minimum, maximum = _aabb(row, "obstacle")
        obstacles.append(_Obstacle(obstacle_id, minimum, maximum))

    entities: list[_Entity] = []
    entity_ids: set[str] = set()
    for row in entity_specs:
        if not isinstance(row, Mapping):
            raise ComposedPairedEntityPlacementError(["entity_spec_invalid"])
        entity_id = _identifier(row.get("entity_id"), "entity_id_invalid")
        if entity_id in entity_ids:
            raise ComposedPairedEntityPlacementError(["entity_id_duplicate"])
        entity_ids.add(entity_id)
        footprint = _vector(
            row.get("footprint_xy_m"), 2, "entity_footprint_invalid", positive=True
        )
        entities.append(
            _Entity(
                entity_id=entity_id,
                footprint=(footprint[0], footprint[1]),
                height=_number(row.get("height_m"), "entity_height_invalid", positive=True),
            )
        )

    centers = tuple(
        sorted(
            {
                _vector(center, 3, "canonical_task_center_invalid")
                for center in canonical_task_centers_m
            }
        )
    )
    footprint = _vector(
        robot_spec.get("base_footprint_xy_m"),
        2,
        "robot_base_footprint_invalid",
        positive=True,
    )
    reach = _vector(robot_spec.get("reach_annulus_m"), 2, "robot_reach_annulus_invalid")
    if reach[0] < 0.0 or reach[1] <= reach[0]:
        raise ComposedPairedEntityPlacementError(["robot_reach_annulus_invalid"])
    robot = _Robot(
        footprint=(footprint[0], footprint[1]),
        clearance_height=_number(
            robot_spec.get("base_clearance_height_m"),
            "robot_base_clearance_height_invalid",
            positive=True,
        ),
        reach_minimum=reach[0],
        reach_maximum=reach[1],
    )
    required_separations = (
        "canonical_region",
        "entity_entity",
        "entity_obstacle",
        "robot_entity",
        "robot_obstacle",
        "support_edge",
    )
    separations = {
        name: _nonnegative(
            minimum_separations_m.get(name), f"minimum_separation_{name}_invalid"
        )
        for name in required_separations
    }
    spacing = _number(grid_spacing_m, "grid_spacing_invalid", positive=True)

    supports.sort(key=lambda row: row.support_region_id)
    obstacles.sort(key=lambda row: row.obstacle_id)
    entities.sort(key=lambda row: row.entity_id)
    request: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "support_regions": [
            {
                "support_region_id": row.support_region_id,
                "aabb_min_m": list(row.minimum),
                "aabb_max_m": list(row.maximum),
                "supports_entities": row.supports_entities,
                "supports_robot_base": row.supports_robot_base,
            }
            for row in supports
        ],
        "obstacle_aabbs": [
            {
                "obstacle_id": row.obstacle_id,
                "aabb_min_m": list(row.minimum),
                "aabb_max_m": list(row.maximum),
            }
            for row in obstacles
        ],
        "entity_specs": [
            {
                "entity_id": row.entity_id,
                "footprint_xy_m": list(row.footprint),
                "height_m": row.height,
            }
            for row in entities
        ],
        "canonical_task_centers_m": [list(center) for center in centers],
        "robot_spec": {
            "base_footprint_xy_m": list(robot.footprint),
            "base_clearance_height_m": robot.clearance_height,
            "reach_annulus_m": [robot.reach_minimum, robot.reach_maximum],
            "reach_distance_definition": "horizontal_base_center_to_entity_center_xy",
        },
        "minimum_separations_m": dict(sorted(separations.items())),
        "separation_distance_definitions": {
            "canonical_region": "horizontal_point_to_entity_footprint_aabb",
            "all_other_clearances": "euclidean_distance_between_3d_axis_aligned_aabbs",
        },
        "grid_spacing_m": spacing,
        "grid_origin_world_xy_m": [0.0, 0.0],
        "coordinate_precision_decimal_places": _PRECISION,
        "frozen_seed_uint64": frozen_seed,
        "maximum_combination_count": maximum_combination_count,
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return (
        request,
        tuple(supports),
        tuple(obstacles),
        tuple(entities),
        centers,
        robot,
        separations,
    )


def _grid_axis(minimum: float, maximum: float, spacing: float) -> tuple[float, ...]:
    first = math.ceil((minimum - _TOLERANCE) / spacing)
    last = math.floor((maximum + _TOLERANCE) / spacing)
    if first > last:
        return ()
    return tuple(round(index * spacing, _PRECISION) for index in range(first, last + 1))


def _placement_grid(
    *,
    subject_id: str,
    footprint: tuple[float, float],
    height: float,
    supports: Sequence[_Support],
    supports_subject: str,
    support_edge: float,
    spacing: float,
) -> tuple[list[_Placement], list[str]]:
    placements: list[_Placement] = []
    no_fit_supports: list[str] = []
    for support in supports:
        if supports_subject == "entity" and not support.supports_entities:
            continue
        if supports_subject == "robot_base" and not support.supports_robot_base:
            continue
        half_x = footprint[0] / 2.0
        half_y = footprint[1] / 2.0
        xs = _grid_axis(
            support.minimum[0] + half_x + support_edge,
            support.maximum[0] - half_x - support_edge,
            spacing,
        )
        ys = _grid_axis(
            support.minimum[1] + half_y + support_edge,
            support.maximum[1] - half_y - support_edge,
            spacing,
        )
        if not xs or not ys:
            no_fit_supports.append(support.support_region_id)
            continue
        bottom = support.maximum[2]
        for x in xs:
            for y in ys:
                placements.append(
                    _Placement(
                        subject_id=subject_id,
                        support_region_id=support.support_region_id,
                        center=(x, y, round(bottom + height / 2.0, _PRECISION)),
                        minimum=(
                            round(x - half_x, _PRECISION),
                            round(y - half_y, _PRECISION),
                            bottom,
                        ),
                        maximum=(
                            round(x + half_x, _PRECISION),
                            round(y + half_y, _PRECISION),
                            round(bottom + height, _PRECISION),
                        ),
                    )
                )
    placements.sort(
        key=lambda row: (row.support_region_id, row.center[0], row.center[1], row.center[2])
    )
    return placements, sorted(no_fit_supports)


def _aabb_distance(
    minimum_a: Sequence[float],
    maximum_a: Sequence[float],
    minimum_b: Sequence[float],
    maximum_b: Sequence[float],
    *,
    dimensions: int = 3,
) -> float:
    squared = 0.0
    for axis in range(dimensions):
        gap = max(minimum_b[axis] - maximum_a[axis], minimum_a[axis] - maximum_b[axis], 0.0)
        squared += gap * gap
    return math.sqrt(squared)


def _positive_volume_overlap(
    minimum_a: Sequence[float],
    maximum_a: Sequence[float],
    minimum_b: Sequence[float],
    maximum_b: Sequence[float],
) -> bool:
    return all(
        min(maximum_a[axis], maximum_b[axis])
        > max(minimum_a[axis], minimum_b[axis]) + _TOLERANCE
        for axis in range(3)
    )


def _violates_clearance(
    placement: _Placement,
    minimum: Sequence[float],
    maximum: Sequence[float],
    clearance: float,
) -> bool:
    if _positive_volume_overlap(placement.minimum, placement.maximum, minimum, maximum):
        return True
    return (
        _aabb_distance(placement.minimum, placement.maximum, minimum, maximum)
        < clearance - _TOLERANCE
    )


def _canonical_exclusion_hit(
    placement: _Placement,
    centers: Sequence[Sequence[float]],
    clearance: float,
) -> bool:
    for center in centers:
        point_minimum = (center[0], center[1], 0.0)
        if (
            _aabb_distance(
                placement.minimum,
                placement.maximum,
                point_minimum,
                point_minimum,
                dimensions=2,
            )
            < clearance - _TOLERANCE
        ):
            return True
    return False


def _record_reasons(
    histogram: dict[str, int],
    signature_histogram: dict[str, int],
    reasons: Sequence[str],
) -> None:
    signature = "+".join(sorted(reasons))
    signature_histogram[signature] = signature_histogram.get(signature, 0) + 1
    for reason in sorted(set(reasons)):
        histogram[reason] = histogram.get(reason, 0) + 1


def _placement_row(placement: _Placement) -> dict[str, Any]:
    return {
        "subject_id": placement.subject_id,
        "support_region_id": placement.support_region_id,
        "center_world_m": list(placement.center),
        "aabb_min_m": list(placement.minimum),
        "aabb_max_m": list(placement.maximum),
    }


def _seed_resolved_index(
    *, frozen_seed: int, request_digest: str, admissible_count: int
) -> tuple[int, str]:
    seed_material = (
        f"{REQUEST_SCHEMA_VERSION}\n{frozen_seed}\n{request_digest}\n"
        f"{admissible_count}"
    ).encode("utf-8")
    seed_digest = hashlib.sha256(seed_material).hexdigest()
    return int(seed_digest, 16) % admissible_count, f"sha256:{seed_digest}"


def _blocked_receipt(
    *,
    request: Mapping[str, Any],
    enumeration: Mapping[str, Any],
    blocking_stage: str,
    blocker: str = NO_FIT_BLOCKER,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "blocked",
        "request": dict(request),
        "blocking_stage": blocking_stage,
        "enumeration": dict(enumeration),
        "selection": None,
        "blockers": [blocker],
        "pending_native_gates": [
            "native_full_phase_ik",
            "native_contact_and_stable_support",
            "native_policy_and_review_camera_visibility",
        ],
        "claim_boundary": _claim_boundary(),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def _claim_boundary() -> dict[str, bool]:
    return {
        "geometry_plausibility_only": True,
        "native_ik_qualified": False,
        "native_contact_qualified": False,
        "native_camera_visibility_qualified": False,
        "simulator_execution_authorized": False,
        "physical_task_performance_proven": False,
    }


def plan_composed_paired_entity_placement(
    *,
    support_regions: Sequence[Mapping[str, Any]],
    obstacle_aabbs: Sequence[Mapping[str, Any]],
    entity_specs: Sequence[Mapping[str, Any]],
    canonical_task_centers_m: Sequence[Sequence[float]],
    robot_spec: Mapping[str, Any],
    minimum_separations_m: Mapping[str, Any],
    grid_spacing_m: float,
    frozen_seed: int,
    maximum_combination_count: int = MAX_COMBINATIONS,
) -> dict[str, Any]:
    """Return one seed-resolved geometry candidate or a typed blocked receipt."""

    (
        request,
        supports,
        obstacles,
        entities,
        canonical_centers,
        robot,
        separations,
    ) = _normalise_request(
        support_regions=support_regions,
        obstacle_aabbs=obstacle_aabbs,
        entity_specs=entity_specs,
        canonical_task_centers_m=canonical_task_centers_m,
        robot_spec=robot_spec,
        minimum_separations_m=minimum_separations_m,
        grid_spacing_m=grid_spacing_m,
        frozen_seed=frozen_seed,
        maximum_combination_count=maximum_combination_count,
    )
    spacing = request["grid_spacing_m"]
    enumeration: dict[str, Any] = {
        "canonical_order": (
            "entity_id, support_region_id, x, y, z; paired entity rows; "
            "robot support_region_id, x, y, z"
        ),
        "entity_placements": {},
        "entity_pairs": {},
        "robot_base_placements": {},
        "paired_entity_robot_candidates": {},
    }

    admissible_by_entity: dict[str, list[_Placement]] = {}
    for entity in entities:
        grid, no_fit_supports = _placement_grid(
            subject_id=entity.entity_id,
            footprint=entity.footprint,
            height=entity.height,
            supports=supports,
            supports_subject="entity",
            support_edge=separations["support_edge"],
            spacing=spacing,
        )
        admissible: list[_Placement] = []
        histogram: dict[str, int] = {}
        signatures: dict[str, int] = {}
        for placement in grid:
            reasons: list[str] = []
            if _canonical_exclusion_hit(
                placement, canonical_centers, separations["canonical_region"]
            ):
                reasons.append("entity_inside_canonical_exclusion")
            if any(
                _violates_clearance(
                    placement,
                    obstacle.minimum,
                    obstacle.maximum,
                    separations["entity_obstacle"],
                )
                for obstacle in obstacles
            ):
                reasons.append("entity_obstacle_clearance")
            if reasons:
                _record_reasons(histogram, signatures, reasons)
            else:
                admissible.append(placement)
        admissible_by_entity[entity.entity_id] = admissible
        enumeration["entity_placements"][entity.entity_id] = {
            "support_regions_without_grid_fit": no_fit_supports,
            "grid_points_considered": len(grid),
            "admissible_count": len(admissible),
            "rejected_count": len(grid) - len(admissible),
            "rejection_reason_counts": dict(sorted(histogram.items())),
            "rejection_signature_counts": dict(sorted(signatures.items())),
        }
        if not admissible:
            return _blocked_receipt(
                request=request,
                enumeration=enumeration,
                blocking_stage=f"entity_placement:{entity.entity_id}",
            )

    first_rows = admissible_by_entity[entities[0].entity_id]
    second_rows = admissible_by_entity[entities[1].entity_id]
    pair_potential_count = len(first_rows) * len(second_rows)
    if pair_potential_count > maximum_combination_count:
        enumeration["entity_pairs"] = {
            "potential_count": pair_potential_count,
            "pairs_considered": 0,
            "admissible_count": 0,
            "rejected_count": 0,
            "rejection_reason_counts": {},
            "rejection_signature_counts": {},
        }
        return _blocked_receipt(
            request=request,
            enumeration=enumeration,
            blocking_stage="bounded_entity_pair_enumeration",
            blocker=SEARCH_SPACE_BLOCKER,
        )
    entity_pairs: list[tuple[_Placement, _Placement]] = []
    pair_histogram: dict[str, int] = {}
    pair_signatures: dict[str, int] = {}
    for first in first_rows:
        for second in second_rows:
            reasons: list[str] = []
            if _violates_clearance(
                first,
                second.minimum,
                second.maximum,
                separations["entity_entity"],
            ):
                reasons.append("entity_pair_clearance")
            if reasons:
                _record_reasons(pair_histogram, pair_signatures, reasons)
            else:
                entity_pairs.append((first, second))
    enumeration["entity_pairs"] = {
        "potential_count": pair_potential_count,
        "pairs_considered": pair_potential_count,
        "admissible_count": len(entity_pairs),
        "rejected_count": pair_potential_count - len(entity_pairs),
        "rejection_reason_counts": dict(sorted(pair_histogram.items())),
        "rejection_signature_counts": dict(sorted(pair_signatures.items())),
    }
    if not entity_pairs:
        return _blocked_receipt(
            request=request,
            enumeration=enumeration,
            blocking_stage="entity_pair",
        )

    robot_grid, robot_no_fit_supports = _placement_grid(
        subject_id="robot_base",
        footprint=robot.footprint,
        height=robot.clearance_height,
        supports=supports,
        supports_subject="robot_base",
        support_edge=separations["support_edge"],
        spacing=spacing,
    )
    robot_rows: list[_Placement] = []
    robot_histogram: dict[str, int] = {}
    robot_signatures: dict[str, int] = {}
    for placement in robot_grid:
        reasons: list[str] = []
        if any(
            _violates_clearance(
                placement,
                obstacle.minimum,
                obstacle.maximum,
                separations["robot_obstacle"],
            )
            for obstacle in obstacles
        ):
            reasons.append("robot_base_obstacle_clearance")
        if reasons:
            _record_reasons(robot_histogram, robot_signatures, reasons)
        else:
            robot_rows.append(placement)
    enumeration["robot_base_placements"] = {
        "support_regions_without_grid_fit": robot_no_fit_supports,
        "grid_points_considered": len(robot_grid),
        "admissible_before_pair_count": len(robot_rows),
        "rejected_count": len(robot_grid) - len(robot_rows),
        "rejection_reason_counts": dict(sorted(robot_histogram.items())),
        "rejection_signature_counts": dict(sorted(robot_signatures.items())),
    }
    if not robot_rows:
        return _blocked_receipt(
            request=request,
            enumeration=enumeration,
            blocking_stage="robot_base_placement",
        )

    potential_count = len(entity_pairs) * len(robot_rows)
    enumeration["paired_entity_robot_candidates"] = {
        "potential_count": potential_count,
        "considered_count": 0,
        "admissible_count": 0,
        "rejected_count": 0,
        "rejection_reason_counts": {},
        "rejection_signature_counts": {},
    }
    if potential_count > maximum_combination_count:
        return _blocked_receipt(
            request=request,
            enumeration=enumeration,
            blocking_stage="bounded_combination_enumeration",
            blocker=SEARCH_SPACE_BLOCKER,
        )

    def combinations() -> Iterator[tuple[tuple[_Placement, _Placement], _Placement, list[str]]]:
        for pair in entity_pairs:
            for base in robot_rows:
                reasons: list[str] = []
                if any(
                    math.hypot(
                        entity.center[0] - base.center[0],
                        entity.center[1] - base.center[1],
                    )
                    < robot.reach_minimum - _TOLERANCE
                    for entity in pair
                ):
                    reasons.append("robot_reach_inner_radius")
                if any(
                    math.hypot(
                        entity.center[0] - base.center[0],
                        entity.center[1] - base.center[1],
                    )
                    > robot.reach_maximum + _TOLERANCE
                    for entity in pair
                ):
                    reasons.append("robot_reach_outer_radius")
                if any(
                    _violates_clearance(
                        base,
                        entity.minimum,
                        entity.maximum,
                        separations["robot_entity"],
                    )
                    for entity in pair
                ):
                    reasons.append("robot_base_entity_clearance")
                yield pair, base, reasons

    combination_histogram: dict[str, int] = {}
    combination_signatures: dict[str, int] = {}
    admissible_count = 0
    for _, _, reasons in combinations():
        if reasons:
            _record_reasons(combination_histogram, combination_signatures, reasons)
        else:
            admissible_count += 1
    combination_summary = enumeration["paired_entity_robot_candidates"]
    combination_summary.update(
        {
            "considered_count": potential_count,
            "admissible_count": admissible_count,
            "rejected_count": potential_count - admissible_count,
            "rejection_reason_counts": dict(sorted(combination_histogram.items())),
            "rejection_signature_counts": dict(sorted(combination_signatures.items())),
        }
    )
    if not admissible_count:
        return _blocked_receipt(
            request=request,
            enumeration=enumeration,
            blocking_stage="paired_entity_robot_geometry",
        )

    selected_index, selection_seed_digest = _seed_resolved_index(
        frozen_seed=frozen_seed,
        request_digest=request["request_digest"],
        admissible_count=admissible_count,
    )
    selected_pair: tuple[_Placement, _Placement] | None = None
    selected_base: _Placement | None = None
    seen_admissible = 0
    for pair, base, reasons in combinations():
        if reasons:
            continue
        if seen_admissible == selected_index:
            selected_pair = pair
            selected_base = base
            break
        seen_admissible += 1
    if selected_pair is None or selected_base is None:  # pragma: no cover - internal invariant.
        raise AssertionError("selected admissible candidate was not found")

    selection: dict[str, Any] = {
        "selection_algorithm": (
            "canonical-sort admissible candidates; sha256 schema version, frozen "
            "seed, request digest, and admissible count; interpret as a big-endian "
            "integer modulo admissible_candidate_count"
        ),
        "selection_seed_digest": selection_seed_digest,
        "selected_index": selected_index,
        "admissible_candidate_count": admissible_count,
        "entity_placements": [_placement_row(row) for row in selected_pair],
        "robot_base_placement": _placement_row(selected_base),
        "reach_distance_xy_m": {
            row.subject_id: round(
                math.hypot(
                    row.center[0] - selected_base.center[0],
                    row.center[1] - selected_base.center[1],
                ),
                _PRECISION,
            )
            for row in selected_pair
        },
        "geometry_checks": {
            "both_entities_supported": True,
            "canonical_region_excluded": True,
            "entity_pair_clearance_passed": True,
            "obstacle_clearance_passed": True,
            "robot_base_supported": True,
            "robot_reach_annulus_passed": True,
            "robot_base_clearance_passed": True,
        },
    }
    selection["selection_digest"] = canonical_digest(
        selection, digest_field="selection_digest"
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "geometry_plausibility_candidate_selected",
        "request": request,
        "blocking_stage": None,
        "enumeration": enumeration,
        "selection": selection,
        "blockers": [],
        "pending_native_gates": [
            "native_full_phase_ik",
            "native_contact_and_stable_support",
            "native_policy_and_review_camera_visibility",
        ],
        "claim_boundary": _claim_boundary(),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "ComposedPairedEntityPlacementError",
    "MAX_COMBINATIONS",
    "NO_FIT_BLOCKER",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "SEARCH_SPACE_BLOCKER",
    "plan_composed_paired_entity_placement",
]
