"""Independently cross-check semantic OBB candidates against collision evidence.

The semantic lifting and oriented-box stages intentionally stop before collision
or physics authority.  This module adds a deterministic, source-bound consistency
check against a separately produced and independently qualified collision scene.
It verifies support contact, target-volume overlap, non-target penetration,
verified-free-space conflicts, coverage, and generated-region intersections.

A passing result means only that a semantic OBB candidate is consistent with the
qualified collision evidence inside the declared envelope.  It does not turn the
box into collision geometry, establish material/contact dynamics, prove physical
task success, or establish safety/deployment readiness.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from .semantic_gaussian_lifting import canonical_json_digest


REQUEST_SCHEMA_VERSION = "semantic_collision_validation_request.v1"
RESULT_SCHEMA_VERSION = "semantic_collision_validation_result.v1"
COLLISION_SCENE_SCHEMA_VERSION = "independent_collision_scene.v1"
VALIDATION_METHOD = "independent_obb_collision_consistency.v1"
_EVIDENCE_SOURCE_CLASSES = {"observed", "verified_asset"}
_EPS = 1e-12
_MAX_PRIMITIVES = 100_000
_MAX_SUPPORT_SURFACES = 10_000
_MAX_COVERAGE_VOLUMES = 10_000
_REQUIRED_QUALIFIED_CHECKS = {
    "target_volume_overlap",
    "support_contact",
    "non_target_penetration",
    "verified_free_space_conflict",
    "coverage",
    "generated_region_intersection",
}


def _valid_digest(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _same_digest(left: Any, right: Any) -> bool:
    def normalized(value: Any) -> str:
        text = str(value or "").strip().lower()
        return text[7:] if text.startswith("sha256:") else text

    return normalized(left) == normalized(right)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    values = [_finite(item) for item in value]
    if any(item is None for item in values):
        return None
    return tuple(float(item) for item in values)  # type: ignore[arg-type]


def _result_digest_valid(payload: Mapping[str, Any], *, field: str) -> bool:
    claimed = payload.get(field)
    if not _valid_digest(claimed):
        return False
    unsigned = dict(payload)
    unsigned.pop(field, None)
    try:
        return _same_digest(claimed, canonical_json_digest(unsigned))
    except (TypeError, ValueError):
        return False


def _blocked(request: Mapping[str, Any], blockers: Sequence[str]) -> Dict[str, Any]:
    bindings = request.get("bindings")
    result: Dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "bindings": dict(bindings) if isinstance(bindings, Mapping) else {},
        "validation_method": VALIDATION_METHOD,
        "objects": [],
        "blockers": sorted(set(blockers)),
        "claim_ceiling": "none_invalid_or_unbound_input",
        "collision_consistency_candidate_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "prohibited_claims": [
            "collision_geometry_or_contact_truth",
            "mass_friction_articulation_or_dynamics",
            "task_or_physical_success",
            "safety_or_deployment_readiness",
        ],
    }
    result["result_digest"] = canonical_json_digest(result)
    return result


def _qualification(request: Mapping[str, Any], blockers: list[str]) -> Dict[str, float | bool]:
    raw = request.get("qualification")
    if not isinstance(raw, Mapping):
        blockers.append("qualification_missing")
        return {}
    ranges = {
        "min_scene_coverage": (0.0, 1.0),
        "max_spatial_uncertainty_m": (0.0, 100.0),
        "max_support_gap_m": (0.0, 100.0),
        "max_support_penetration_m": (0.0, 100.0),
        "min_support_overlap_fraction": (0.0, 1.0),
        "min_target_iou": (0.0, 1.0),
        "max_non_target_penetration_fraction": (0.0, 1.0),
        "max_free_space_conflict_fraction": (0.0, 1.0),
    }
    result: Dict[str, float | bool] = {}
    for field, (minimum, maximum) in ranges.items():
        value = _finite(raw.get(field))
        if value is None or value < minimum or value > maximum:
            blockers.append(f"qualification_invalid:{field}")
        else:
            result[field] = value
    require_full = raw.get("require_full_corner_coverage")
    if not isinstance(require_full, bool):
        blockers.append("qualification_invalid:require_full_corner_coverage")
    else:
        result["require_full_corner_coverage"] = require_full
    return result


def _validate_request(request: Mapping[str, Any], blockers: list[str]) -> None:
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("request_schema_version_unsupported")
    if request.get("validation_method") != VALIDATION_METHOD:
        blockers.append("validation_method_unsupported")
    bindings = request.get("bindings")
    if not isinstance(bindings, Mapping):
        blockers.append("bindings_missing")
    else:
        for field in (
            "capture_digest",
            "reconstruction_digest",
            "analysis_splat_digest",
            "semantic_oriented_box_result_digest",
            "collision_scene_digest",
            "collision_method_profile_digest",
        ):
            if not _valid_digest(bindings.get(field)):
                blockers.append(f"binding_digest_invalid:{field}")
    world = request.get("world")
    if not isinstance(world, Mapping):
        blockers.append("world_contract_missing")
    else:
        if str(world.get("up_axis") or "").strip().upper() != "Z":
            blockers.append("world_up_axis_must_be_z")
        if str(world.get("units") or "").strip().lower() != "meters":
            blockers.append("world_units_must_be_meters")
        if world.get("scale_verified") is not True:
            blockers.append("verified_metric_scale_required")
        if not str(world.get("coordinate_frame") or "").strip():
            blockers.append("world_coordinate_frame_missing")


def _validate_obb_result(
    request: Mapping[str, Any], result: Mapping[str, Any], blockers: list[str]
) -> None:
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    if result.get("schema_version") != "semantic_oriented_box_result.v1":
        blockers.append("semantic_oriented_box_schema_unsupported")
    if not _result_digest_valid(result, field="result_digest"):
        blockers.append("semantic_oriented_box_result_digest_invalid")
    elif not _same_digest(
        bindings.get("semantic_oriented_box_result_digest"), result.get("result_digest")
    ):
        blockers.append("semantic_oriented_box_result_digest_mismatch")
    if result.get("status") not in {"completed", "partially_completed", "abstained"}:
        blockers.append("semantic_oriented_box_not_terminal_valid")
    if result.get("collision_ready") is not False or result.get("physics_ready") is not False:
        blockers.append("semantic_oriented_box_authority_boundary_invalid")
    if result.get("generated_regions_can_upgrade_claims") is not False:
        blockers.append("semantic_generated_region_boundary_missing")
    result_world = result.get("world")
    request_world = request.get("world")
    if not isinstance(result_world, Mapping) or not isinstance(request_world, Mapping):
        blockers.append("semantic_oriented_box_world_contract_missing")
    elif (
        str(result_world.get("up_axis") or "").strip().upper()
        != str(request_world.get("up_axis") or "").strip().upper()
        or str(result_world.get("units") or "").strip().lower()
        != str(request_world.get("units") or "").strip().lower()
        or result_world.get("scale_verified") is not request_world.get("scale_verified")
    ):
        blockers.append("semantic_oriented_box_world_contract_mismatch")
    result_bindings = result.get("bindings")
    if not isinstance(result_bindings, Mapping):
        blockers.append("semantic_oriented_box_bindings_missing")
    else:
        for field in ("capture_digest", "reconstruction_digest", "analysis_splat_digest"):
            if not _same_digest(bindings.get(field), result_bindings.get(field)):
                blockers.append(f"semantic_oriented_box_binding_mismatch:{field}")


def _validate_method_profile(
    request: Mapping[str, Any], profile: Any, blockers: list[str]
) -> Mapping[str, Any]:
    if not isinstance(profile, Mapping):
        blockers.append("collision_method_profile_missing")
        return {}
    required_text = ("method_id", "method_version", "producer_identity", "validator_identity")
    for field in required_text:
        if not str(profile.get(field) or "").strip():
            blockers.append(f"collision_method_profile_missing:{field}")
    if not _valid_digest(profile.get("runtime_digest")):
        blockers.append("collision_method_runtime_digest_invalid")
    if profile.get("deterministic") is not True:
        blockers.append("collision_method_determinism_required")
    if profile.get("source_capture_bound") is not True:
        blockers.append("collision_method_source_capture_binding_required")
    if profile.get("metric_transform_verified") is not True:
        blockers.append("collision_method_metric_transform_required")
    if profile.get("qualification_status") != "qualified":
        blockers.append("collision_method_not_qualified")
    if profile.get("independent_from_semantic_geometry") is not True:
        blockers.append("collision_method_semantic_independence_required")
    raw_checks = profile.get("qualified_checks")
    checks = (
        {str(item).strip() for item in raw_checks if str(item).strip()}
        if isinstance(raw_checks, Sequence) and not isinstance(raw_checks, (str, bytes))
        else set()
    )
    missing_checks = sorted(_REQUIRED_QUALIFIED_CHECKS - checks)
    for check in missing_checks:
        blockers.append(f"collision_method_check_not_qualified:{check}")
    if (
        str(profile.get("producer_identity") or "").strip()
        == str(profile.get("validator_identity") or "").strip()
    ):
        blockers.append("collision_provider_cannot_self_qualify")
    claimed = profile.get("method_profile_digest")
    unsigned = dict(profile)
    unsigned.pop("method_profile_digest", None)
    if not _valid_digest(claimed) or not _same_digest(claimed, canonical_json_digest(unsigned)):
        blockers.append("collision_method_profile_digest_invalid")
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    if not _same_digest(bindings.get("collision_method_profile_digest"), claimed):
        blockers.append("collision_method_profile_digest_mismatch")
    return profile


def _aabb(
    row: Mapping[str, Any],
    blockers: list[str],
    prefix: str,
    *,
    allowed_source_classes: set[str] = _EVIDENCE_SOURCE_CLASSES,
) -> Dict[str, Any] | None:
    primitive_id = str(row.get("primitive_id") or row.get("region_id") or "").strip()
    minimum = _vector3(row.get("minimum_world_m"))
    maximum = _vector3(row.get("maximum_world_m"))
    source_class = str(row.get("source_class") or "").strip().lower()
    if not primitive_id:
        blockers.append(f"{prefix}_id_missing")
        return None
    if (
        minimum is None
        or maximum is None
        or any(minimum[axis] >= maximum[axis] for axis in range(3))
    ):
        blockers.append(f"{prefix}_bounds_invalid:{primitive_id}")
        return None
    if source_class not in allowed_source_classes:
        blockers.append(f"{prefix}_source_class_invalid:{primitive_id}")
        return None
    return {
        "primitive_id": primitive_id,
        "object_id": str(row.get("object_id") or "").strip(),
        "minimum": minimum,
        "maximum": maximum,
        "source_class": source_class,
    }


def _polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    return (
        abs(
            sum(
                polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
                - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
                for index in range(len(polygon))
            )
        )
        * 0.5
    )


def _signed_area(polygon: Sequence[tuple[float, float]]) -> float:
    return (
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
        * 0.5
    )


def _convex_polygon(value: Any) -> list[tuple[float, float]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    points: list[tuple[float, float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 2:
            return None
        x, y = _finite(row[0]), _finite(row[1])
        if x is None or y is None:
            return None
        points.append((x, y))
    if abs(_signed_area(points)) <= _EPS:
        return None
    signs: set[int] = set()
    for index in range(len(points)):
        first, second, third = points[index - 2], points[index - 1], points[index]
        cross = (second[0] - first[0]) * (third[1] - second[1]) - (second[1] - first[1]) * (
            third[0] - second[0]
        )
        if abs(cross) > _EPS:
            signs.add(1 if cross > 0 else -1)
    if len(signs) != 1:
        return None
    return points


def _inside_clip_edge(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    orientation: float,
) -> bool:
    cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    )
    return cross * orientation >= -_EPS


def _line_intersection(
    first: tuple[float, float],
    second: tuple[float, float],
    clip_start: tuple[float, float],
    clip_end: tuple[float, float],
) -> tuple[float, float]:
    dx, dy = second[0] - first[0], second[1] - first[1]
    ex, ey = clip_end[0] - clip_start[0], clip_end[1] - clip_start[1]
    denominator = dx * ey - dy * ex
    if abs(denominator) <= _EPS:
        return second
    t = ((clip_start[0] - first[0]) * ey - (clip_start[1] - first[1]) * ex) / denominator
    return first[0] + t * dx, first[1] + t * dy


def _clip_polygon(
    subject: Sequence[tuple[float, float]], clip: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    output = list(subject)
    orientation = 1.0 if _signed_area(clip) > 0 else -1.0
    for index, clip_start in enumerate(clip):
        clip_end = clip[(index + 1) % len(clip)]
        input_points, output = output, []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _inside_clip_edge(current, clip_start, clip_end, orientation)
            previous_inside = _inside_clip_edge(previous, clip_start, clip_end, orientation)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, clip_start, clip_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, clip_start, clip_end))
            previous = current
    return output


def _box_geometry(obj: Mapping[str, Any]) -> Dict[str, Any] | None:
    dimensions = _vector3(obj.get("dimensions_m"))
    center = _vector3(obj.get("center_world_m"))
    corners = obj.get("corners_world_m")
    if dimensions is None or center is None or any(value <= 0 for value in dimensions):
        return None
    if not isinstance(corners, Sequence) or isinstance(corners, (str, bytes)) or len(corners) != 8:
        return None
    normalized_corners = [_vector3(corner) for corner in corners]
    if any(corner is None for corner in normalized_corners):
        return None
    rows = [corner for corner in normalized_corners if corner is not None]
    bottom_z, top_z = min(row[2] for row in rows), max(row[2] for row in rows)
    footprint = [(rows[index][0], rows[index][1]) for index in range(4)]
    if _convex_polygon(footprint) is None or top_z - bottom_z <= _EPS:
        return None
    if any(
        abs(rows[index][0] - rows[index + 4][0]) > 1e-8
        or abs(rows[index][1] - rows[index + 4][1]) > 1e-8
        for index in range(4)
    ):
        return None
    footprint_area = _polygon_area(footprint)
    declared_volume = dimensions[0] * dimensions[1] * dimensions[2]
    corner_volume = footprint_area * (top_z - bottom_z)
    if abs(declared_volume - corner_volume) > max(1e-9, declared_volume * 1e-7):
        return None
    corner_center = tuple(sum(row[axis] for row in rows) / 8.0 for axis in range(3))
    if any(abs(center[axis] - corner_center[axis]) > 1e-8 for axis in range(3)):
        return None
    return {
        "center": center,
        "dimensions": dimensions,
        "corners": rows,
        "footprint": footprint,
        "bottom_z": bottom_z,
        "top_z": top_z,
        "volume": dimensions[0] * dimensions[1] * dimensions[2],
    }


def _aabb_polygon(primitive: Mapping[str, Any]) -> list[tuple[float, float]]:
    minimum, maximum = primitive["minimum"], primitive["maximum"]
    return [
        (minimum[0], minimum[1]),
        (maximum[0], minimum[1]),
        (maximum[0], maximum[1]),
        (minimum[0], maximum[1]),
    ]


def _intersection_volume(box: Mapping[str, Any], primitive: Mapping[str, Any]) -> float:
    polygon = _clip_polygon(box["footprint"], _aabb_polygon(primitive))
    area = _polygon_area(polygon) if len(polygon) >= 3 else 0.0
    overlap_z = max(
        0.0,
        min(float(box["top_z"]), primitive["maximum"][2])
        - max(float(box["bottom_z"]), primitive["minimum"][2]),
    )
    return area * overlap_z


def _aabb_volume(primitive: Mapping[str, Any]) -> float:
    return math.prod(primitive["maximum"][axis] - primitive["minimum"][axis] for axis in range(3))


def _point_in_aabb(point: Sequence[float], primitive: Mapping[str, Any]) -> bool:
    return all(
        primitive["minimum"][axis] - _EPS <= point[axis] <= primitive["maximum"][axis] + _EPS
        for axis in range(3)
    )


def _next_experiment(reasons: Sequence[str]) -> str:
    if "generated_region_intersection" in reasons:
        return "recapture_the_generated_or_unobserved_object_region"
    if (
        "collision_target_binding_missing" in reasons
        or "target_collision_primitive_missing" in reasons
    ):
        return "identify_and_independently_bind_the_exact_collision_object_instance"
    if "target_collision_iou_below_threshold" in reasons:
        return "capture_or_measure_the_object_for_independent_collision_geometry_alignment"
    if "support_contact_gap_too_large" in reasons:
        return "capture_the_object_support_contact_and_support_surface"
    if "support_surface_penetration_too_large" in reasons:
        return "verify_the_site_to_object_transform_and_support_plane_height"
    if "support_overlap_below_threshold" in reasons:
        return "capture_the_full_support_surface_and_object_footprint"
    if "non_target_penetration_exceeds_threshold" in reasons:
        return "verify_object_instance_separation_and_refit_collision_geometry"
    if "verified_free_space_conflict_exceeds_threshold" in reasons:
        return "reconcile_the_object_box_with_independently_observed_free_space"
    if "collision_coverage_incomplete" in reasons:
        return "capture_the_object_and_surrounding_collision_volume_from_more_views"
    if "semantic_oriented_box_not_qualified" in reasons:
        return "complete_semantic_support_and_metric_box_qualification_first"
    return "run_an_independent_collision_geometry_review_for_the_object"


def validate_semantic_boxes_against_collision(
    request: Mapping[str, Any],
    *,
    oriented_box_result: Mapping[str, Any],
    collision_scene: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return deterministic collision-consistency candidates or abstentions."""

    blockers: list[str] = []
    _validate_request(request, blockers)
    qualification = _qualification(request, blockers)
    _validate_obb_result(request, oriented_box_result, blockers)
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    world = request.get("world") if isinstance(request.get("world"), Mapping) else {}

    if collision_scene.get("schema_version") != COLLISION_SCENE_SCHEMA_VERSION:
        blockers.append("collision_scene_schema_unsupported")
    if not _result_digest_valid(collision_scene, field="collision_scene_digest"):
        blockers.append("collision_scene_digest_invalid")
    elif not _same_digest(
        bindings.get("collision_scene_digest"), collision_scene.get("collision_scene_digest")
    ):
        blockers.append("collision_scene_digest_mismatch")
    if not _same_digest(
        bindings.get("capture_digest"), collision_scene.get("source_capture_digest")
    ):
        blockers.append("collision_scene_source_capture_mismatch")
    if not _same_digest(
        bindings.get("reconstruction_digest"), collision_scene.get("reconstruction_digest")
    ):
        blockers.append("collision_scene_reconstruction_mismatch")
    if collision_scene.get("generated_geometry") is not False:
        blockers.append("generated_collision_geometry_forbidden")
    if collision_scene.get("scale_status") != "metric_verified":
        blockers.append("collision_scene_metric_scale_unverified")
    if str(collision_scene.get("up_axis") or "").strip().upper() != "Z":
        blockers.append("collision_scene_up_axis_must_be_z")
    if str(collision_scene.get("units") or "").strip().lower() != "meters":
        blockers.append("collision_scene_units_must_be_meters")
    if collision_scene.get("coordinate_frame") != world.get("coordinate_frame"):
        blockers.append("collision_scene_coordinate_frame_mismatch")
    profile = _validate_method_profile(request, collision_scene.get("method_profile"), blockers)

    validation = collision_scene.get("validation")
    if not isinstance(validation, Mapping):
        blockers.append("collision_scene_validation_missing")
        validation = {}
    else:
        if (
            validation.get("status") != "qualified"
            or validation.get("independent_validation") is not True
        ):
            blockers.append("collision_scene_independent_validation_missing")
        if (
            str(validation.get("validator_identity") or "").strip()
            != str(profile.get("validator_identity") or "").strip()
        ):
            blockers.append("collision_scene_validator_identity_mismatch")
    coverage = _finite(validation.get("coverage"))
    uncertainty = _finite(validation.get("maximum_spatial_uncertainty_m"))
    if coverage is None or not 0.0 <= coverage <= 1.0:
        blockers.append("collision_scene_coverage_invalid")
    elif qualification and coverage + _EPS < float(qualification["min_scene_coverage"]):
        blockers.append("collision_scene_coverage_below_qualification")
    if uncertainty is None or uncertainty < 0.0:
        blockers.append("collision_scene_spatial_uncertainty_invalid")
    elif qualification and uncertainty > float(qualification["max_spatial_uncertainty_m"]) + _EPS:
        blockers.append("collision_scene_spatial_uncertainty_exceeds_qualification")

    raw_occupied = collision_scene.get("occupied_primitives")
    raw_free = collision_scene.get("verified_free_space_primitives")
    raw_generated = collision_scene.get("generated_regions")
    raw_coverage = collision_scene.get("coverage_volumes")
    raw_surfaces = collision_scene.get("support_surfaces")
    for name, rows, limit in (
        ("occupied_primitives", raw_occupied, _MAX_PRIMITIVES),
        ("verified_free_space_primitives", raw_free, _MAX_PRIMITIVES),
        ("generated_regions", raw_generated, _MAX_PRIMITIVES),
        ("coverage_volumes", raw_coverage, _MAX_COVERAGE_VOLUMES),
        ("support_surfaces", raw_surfaces, _MAX_SUPPORT_SURFACES),
    ):
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            blockers.append(f"collision_scene_rows_invalid:{name}")
        elif len(rows) > limit:
            blockers.append(f"collision_scene_rows_exceed_limit:{name}")

    occupied: list[Dict[str, Any]] = []
    free_space: list[Dict[str, Any]] = []
    generated: list[Dict[str, Any]] = []
    coverage_volumes: list[Dict[str, Any]] = []
    for name, rows, destination, prefix in (
        ("occupied_primitives", raw_occupied, occupied, "occupied_primitive"),
        ("verified_free_space_primitives", raw_free, free_space, "free_space_primitive"),
        ("generated_regions", raw_generated, generated, "generated_region"),
        ("coverage_volumes", raw_coverage, coverage_volumes, "coverage_volume"),
    ):
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            for row in rows:
                if not isinstance(row, Mapping):
                    blockers.append(f"{prefix}_row_invalid")
                    continue
                normalized = _aabb(
                    row,
                    blockers,
                    prefix,
                    allowed_source_classes={"generated"}
                    if name == "generated_regions"
                    else _EVIDENCE_SOURCE_CLASSES,
                )
                if normalized is not None:
                    destination.append(normalized)
    if not occupied:
        blockers.append("collision_scene_occupied_primitives_missing")
    if not coverage_volumes:
        blockers.append("collision_scene_coverage_volumes_missing")

    surfaces: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_surfaces, Sequence) and not isinstance(raw_surfaces, (str, bytes)):
        for row in raw_surfaces:
            if not isinstance(row, Mapping):
                blockers.append("support_surface_row_invalid")
                continue
            surface_id = str(row.get("surface_id") or "").strip()
            z = _finite(row.get("z_world_m"))
            polygon = _convex_polygon(row.get("polygon_xy_world_m"))
            source_class = str(row.get("source_class") or "").strip().lower()
            if not surface_id or surface_id in surfaces:
                blockers.append("support_surface_id_missing_or_duplicate")
            elif z is None or polygon is None:
                blockers.append(f"support_surface_geometry_invalid:{surface_id}")
            elif source_class not in _EVIDENCE_SOURCE_CLASSES:
                blockers.append(f"support_surface_source_not_observed_or_verified:{surface_id}")
            else:
                surfaces[surface_id] = {
                    "surface_id": surface_id,
                    "z": z,
                    "polygon": polygon,
                    "source_class": source_class,
                }

    target_bindings = collision_scene.get("target_bindings")
    bindings_by_track: Dict[str, Mapping[str, Any]] = {}
    bound_primitive_ids: set[str] = set()
    if not isinstance(target_bindings, Sequence) or isinstance(target_bindings, (str, bytes)):
        blockers.append("collision_target_bindings_invalid")
    else:
        for row in target_bindings:
            if not isinstance(row, Mapping):
                blockers.append("collision_target_binding_row_invalid")
                continue
            track_id = str(row.get("track_id") or "").strip()
            if not track_id or track_id in bindings_by_track:
                blockers.append("collision_target_binding_track_missing_or_duplicate")
            elif row.get("identity_verified") is not True:
                blockers.append(f"collision_target_identity_not_verified:{track_id}")
            elif not _valid_digest(row.get("identity_evidence_digest")):
                blockers.append(f"collision_target_identity_evidence_invalid:{track_id}")
            else:
                primitive_id = str(row.get("primitive_id") or "").strip()
                if not primitive_id or primitive_id in bound_primitive_ids:
                    blockers.append("collision_target_primitive_missing_or_duplicate")
                    continue
                bindings_by_track[track_id] = row
                bound_primitive_ids.add(primitive_id)

    if blockers:
        return _blocked(request, blockers)

    assert uncertainty is not None and coverage is not None
    occupied_by_id = {row["primitive_id"]: row for row in occupied}
    object_rows: list[Dict[str, Any]] = []
    raw_objects = oriented_box_result.get("objects")
    if not isinstance(raw_objects, Sequence) or isinstance(raw_objects, (str, bytes)):
        return _blocked(request, ["semantic_oriented_box_objects_invalid"])
    for obj in sorted(raw_objects, key=lambda row: str(row.get("track_id") or "")):
        if not isinstance(obj, Mapping):
            return _blocked(request, ["semantic_oriented_box_object_row_invalid"])
        track_id = str(obj.get("track_id") or "").strip()
        if not track_id:
            return _blocked(request, ["semantic_object_track_id_missing"])
        reasons: list[str] = []
        if (
            obj.get("status") != "qualified_metric_obb_candidate"
            or obj.get("metric_obb_candidate_ready") is not True
        ):
            reasons.append("semantic_oriented_box_not_qualified")
        if obj.get("collision_ready") is not False or obj.get("physics_ready") is not False:
            return _blocked(request, [f"semantic_object_authority_boundary_invalid:{track_id}"])
        if reasons:
            object_rows.append(
                {
                    "track_id": track_id,
                    "label": str(obj.get("label") or "").strip(),
                    "status": "abstained",
                    "semantic_oriented_box_result_digest": oriented_box_result["result_digest"],
                    "collision_scene_digest": collision_scene["collision_scene_digest"],
                    "target_primitive_id": None,
                    "support_surface_id": None,
                    "metrics": None,
                    "non_target_penetrations": [],
                    "verified_free_space_conflicts": [],
                    "generated_region_intersection_ids": [],
                    "abstention_reasons": reasons,
                    "next_experiment": _next_experiment(reasons),
                    "collision_consistency_candidate_ready": False,
                    "collision_ready": False,
                    "physics_ready": False,
                    "claim_ceiling": "none_collision_consistency_abstained",
                    "provenance": {
                        "validation_method": VALIDATION_METHOD,
                        "collision_method_profile": dict(profile),
                        "target_identity_evidence_digest": None,
                        "generated_geometry_used": False,
                        "physical_robot_run_initiated": False,
                    },
                }
            )
            continue
        if obj.get("coordinate_frame") != world.get("coordinate_frame"):
            return _blocked(request, [f"semantic_object_coordinate_frame_mismatch:{track_id}"])
        box = _box_geometry(obj)
        if box is None:
            return _blocked(request, [f"semantic_object_box_geometry_invalid:{track_id}"])

        target_binding = bindings_by_track.get(track_id)
        target: Mapping[str, Any] | None = None
        support_surface: Mapping[str, Any] | None = None
        if target_binding is None:
            reasons.append("collision_target_binding_missing")
        else:
            primitive_id = str(target_binding.get("primitive_id") or "").strip()
            target = occupied_by_id.get(primitive_id)
            if (
                target is None
                or target.get("object_id") != str(target_binding.get("object_id") or "").strip()
            ):
                reasons.append("target_collision_primitive_missing")
                target = None
            support_surface = surfaces.get(str(target_binding.get("support_surface_id") or ""))
            if support_surface is None:
                reasons.append("support_surface_missing")

        target_iou = 0.0
        if target is not None:
            intersection = _intersection_volume(box, target)
            union = float(box["volume"]) + _aabb_volume(target) - intersection
            target_iou = intersection / union if union > _EPS else 0.0
            if target_iou + _EPS < float(qualification["min_target_iou"]):
                reasons.append("target_collision_iou_below_threshold")

        support_gap = None
        support_overlap = 0.0
        if support_surface is not None:
            support_gap = float(box["bottom_z"]) - float(support_surface["z"])
            overlap_polygon = _clip_polygon(box["footprint"], support_surface["polygon"])
            overlap_area = _polygon_area(overlap_polygon) if len(overlap_polygon) >= 3 else 0.0
            footprint_area = _polygon_area(box["footprint"])
            support_overlap = overlap_area / footprint_area if footprint_area > _EPS else 0.0
            if support_gap + uncertainty > float(qualification["max_support_gap_m"]) + _EPS:
                reasons.append("support_contact_gap_too_large")
            if (
                -support_gap + uncertainty
                > float(qualification["max_support_penetration_m"]) + _EPS
            ):
                reasons.append("support_surface_penetration_too_large")
            if support_overlap + _EPS < float(qualification["min_support_overlap_fraction"]):
                reasons.append("support_overlap_below_threshold")

        target_primitive_id = target.get("primitive_id") if target is not None else None
        penetrations = []
        for primitive in occupied:
            if primitive["primitive_id"] == target_primitive_id:
                continue
            fraction = _intersection_volume(box, primitive) / float(box["volume"])
            if fraction > _EPS:
                penetrations.append(
                    {
                        "primitive_id": primitive["primitive_id"],
                        "object_id": primitive["object_id"],
                        "volume_fraction": round(fraction, 12),
                    }
                )
        maximum_penetration = max((row["volume_fraction"] for row in penetrations), default=0.0)
        if maximum_penetration > float(qualification["max_non_target_penetration_fraction"]) + _EPS:
            reasons.append("non_target_penetration_exceeds_threshold")

        free_conflicts = []
        for primitive in free_space:
            fraction = _intersection_volume(box, primitive) / float(box["volume"])
            if fraction > _EPS:
                free_conflicts.append(
                    {
                        "primitive_id": primitive["primitive_id"],
                        "volume_fraction": round(fraction, 12),
                    }
                )
        maximum_free_conflict = max((row["volume_fraction"] for row in free_conflicts), default=0.0)
        if maximum_free_conflict > float(qualification["max_free_space_conflict_fraction"]) + _EPS:
            reasons.append("verified_free_space_conflict_exceeds_threshold")

        generated_intersections = sorted(
            primitive["primitive_id"]
            for primitive in generated
            if _intersection_volume(box, primitive) > _EPS
        )
        if generated_intersections:
            reasons.append("generated_region_intersection")

        covered_corner_count = sum(
            any(_point_in_aabb(corner, volume) for volume in coverage_volumes)
            for corner in box["corners"]
        )
        covered_corner_fraction = covered_corner_count / 8.0
        if qualification["require_full_corner_coverage"] is True and covered_corner_count != 8:
            reasons.append("collision_coverage_incomplete")

        reasons = sorted(set(reasons))
        row: Dict[str, Any] = {
            "track_id": track_id,
            "label": str(obj.get("label") or "").strip(),
            "status": "independent_collision_consistency_candidate" if not reasons else "abstained",
            "semantic_oriented_box_result_digest": oriented_box_result["result_digest"],
            "collision_scene_digest": collision_scene["collision_scene_digest"],
            "target_primitive_id": target_primitive_id,
            "support_surface_id": support_surface.get("surface_id") if support_surface else None,
            "metrics": {
                "target_collision_iou": round(target_iou, 12),
                "support_signed_gap_m": round(support_gap, 12) if support_gap is not None else None,
                "support_horizontal_overlap_fraction": round(support_overlap, 12),
                "maximum_non_target_penetration_fraction": round(maximum_penetration, 12),
                "maximum_verified_free_space_conflict_fraction": round(maximum_free_conflict, 12),
                "covered_corner_fraction": round(covered_corner_fraction, 12),
                "scene_coverage": round(coverage, 12),
                "maximum_spatial_uncertainty_m": round(uncertainty, 12),
            },
            "non_target_penetrations": sorted(penetrations, key=lambda item: item["primitive_id"]),
            "verified_free_space_conflicts": sorted(
                free_conflicts, key=lambda item: item["primitive_id"]
            ),
            "generated_region_intersection_ids": generated_intersections,
            "abstention_reasons": reasons,
            "collision_consistency_candidate_ready": not reasons,
            "collision_ready": False,
            "physics_ready": False,
            "claim_ceiling": (
                "semantic_obb_independently_consistent_with_qualified_collision_evidence"
                if not reasons
                else "none_collision_consistency_abstained"
            ),
            "provenance": {
                "validation_method": VALIDATION_METHOD,
                "collision_method_profile": dict(profile),
                "target_identity_evidence_digest": (
                    target_binding.get("identity_evidence_digest") if target_binding else None
                ),
                "generated_geometry_used": False,
                "physical_robot_run_initiated": False,
            },
        }
        if reasons:
            row["next_experiment"] = _next_experiment(reasons)
        object_rows.append(row)

    qualified_count = sum(
        row["status"] == "independent_collision_consistency_candidate" for row in object_rows
    )
    if object_rows and qualified_count == len(object_rows):
        status = "completed"
    elif qualified_count:
        status = "partially_completed"
    else:
        status = "abstained"
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "bindings": dict(bindings),
        "validation_method": VALIDATION_METHOD,
        "world": dict(world),
        "objects": object_rows,
        "qualified_object_count": qualified_count,
        "abstained_object_count": len(object_rows) - qualified_count,
        "blockers": [],
        "claim_ceiling": "independent_collision_consistency_candidates_only",
        "collision_consistency_candidate_ready": qualified_count > 0,
        "collision_ready": False,
        "physics_ready": False,
        "generated_regions_can_upgrade_claims": False,
        "prohibited_claims": [
            "collision_geometry_or_contact_truth",
            "mass_friction_articulation_or_dynamics",
            "task_or_physical_success",
            "safety_or_deployment_readiness",
        ],
    }
    result["result_digest"] = canonical_json_digest(result)
    return result
