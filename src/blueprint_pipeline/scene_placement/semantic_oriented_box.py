"""Fit deterministic metric Z-up object-box candidates from qualified support.

This module consumes the output of :mod:`semantic_gaussian_lifting`, the exact
Gaussian mapping used by that lift, and hash-bound 3D support points.  It removes
bounded statistical outliers, fits a minimum-area rectangle in the horizontal
plane, estimates vertical limits independently, and emits eight ordered corners.

The result is semantic metric geometry only.  It is never collision geometry,
physics truth, task success, physical evidence, or safety/deployment proof.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, Mapping, Sequence

from .semantic_gaussian_lifting import canonical_json_digest


REQUEST_SCHEMA_VERSION = "semantic_oriented_box_request.v1"
RESULT_SCHEMA_VERSION = "semantic_oriented_box_result.v1"
FIT_METHOD = "robust_z_up_minimum_area_rectangle.v1"
_POINT_SOURCES = {"observed_depth", "verified_mesh_surface", "gaussian_center"}
_MAX_POINTS = 250_000
_EPS = 1e-12


def _valid_digest(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


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


def _positive_int(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return value


def _result_digest_valid(payload: Mapping[str, Any]) -> bool:
    claimed = payload.get("result_digest")
    if not _valid_digest(claimed):
        return False
    unsigned = dict(payload)
    unsigned.pop("result_digest", None)
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
        "fit_method": FIT_METHOD,
        "objects": [],
        "blockers": sorted(set(blockers)),
        "claim_ceiling": "none_invalid_or_unbound_input",
        "metric_obb_candidate_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "prohibited_claims": [
            "collision_or_contact_truth",
            "task_or_physical_success",
            "safety_or_deployment_readiness",
        ],
    }
    result["result_digest"] = canonical_json_digest(result)
    return result


def _validate_request(request: Mapping[str, Any], blockers: list[str]) -> None:
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("request_schema_version_unsupported")
    if request.get("fit_method") != FIT_METHOD:
        blockers.append("fit_method_unsupported")
    bindings = request.get("bindings")
    if not isinstance(bindings, Mapping):
        blockers.append("bindings_missing")
    else:
        for field in (
            "capture_digest",
            "reconstruction_digest",
            "analysis_splat_digest",
            "gaussian_mapping_digest",
            "semantic_lifting_result_digest",
            "support_points_digest",
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
    support_profile = request.get("support_method_profile")
    if not isinstance(support_profile, Mapping):
        blockers.append("support_method_profile_missing")
    else:
        if not str(support_profile.get("method_id") or "").strip():
            blockers.append("support_method_id_missing")
        if not str(support_profile.get("method_version") or "").strip():
            blockers.append("support_method_version_missing")
        if not _valid_digest(support_profile.get("runtime_digest")):
            blockers.append("support_method_runtime_digest_invalid")
        if support_profile.get("deterministic") is not True:
            blockers.append("support_method_determinism_required")
        if support_profile.get("source_capture_bound") is not True:
            blockers.append("support_method_source_capture_binding_required")
        if support_profile.get("metric_transform_verified") is not True:
            blockers.append("support_method_metric_transform_required")


def _qualification(request: Mapping[str, Any], blockers: list[str]) -> Dict[str, float | int]:
    raw = request.get("qualification")
    if not isinstance(raw, Mapping):
        blockers.append("qualification_missing")
        return {}
    result: Dict[str, float | int] = {}
    for field in ("min_support_points", "min_distinct_gaussians"):
        value = _positive_int(raw.get(field))
        if value is None:
            blockers.append(f"qualification_invalid:{field}")
        else:
            result[field] = value
    ranges = {
        "outlier_mad_multiplier": (1.0, 100.0),
        "vertical_trim_fraction": (0.0, 0.2),
        "min_horizontal_extent_m": (0.0, 100.0),
        "min_vertical_extent_m": (0.0, 100.0),
        "max_dimension_m": (0.001, 10_000.0),
        "min_inlier_fraction": (0.0, 1.0),
    }
    for field, (minimum, maximum) in ranges.items():
        value = _finite(raw.get(field))
        if value is None or value < minimum or value > maximum:
            blockers.append(f"qualification_invalid:{field}")
        else:
            result[field] = value
    return result


def _validate_semantic_result(
    request: Mapping[str, Any],
    semantic_result: Mapping[str, Any],
    blockers: list[str],
) -> None:
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    if semantic_result.get("schema_version") != "semantic_gaussian_lifting_result.v1":
        blockers.append("semantic_lifting_schema_unsupported")
    if not _result_digest_valid(semantic_result):
        blockers.append("semantic_lifting_result_digest_invalid")
    elif not _same_digest(
        bindings.get("semantic_lifting_result_digest"), semantic_result.get("result_digest")
    ):
        blockers.append("semantic_lifting_result_digest_mismatch")
    if semantic_result.get("status") not in {"completed", "partially_completed", "abstained"}:
        blockers.append("semantic_lifting_not_terminal_valid")
    if semantic_result.get("generated_regions_can_upgrade_claims") is not False:
        blockers.append("semantic_generated_region_boundary_missing")
    semantic_bindings = semantic_result.get("bindings")
    if not isinstance(semantic_bindings, Mapping):
        blockers.append("semantic_bindings_missing")
    else:
        for field in (
            "capture_digest",
            "reconstruction_digest",
            "analysis_splat_digest",
            "gaussian_mapping_digest",
        ):
            if not _same_digest(bindings.get(field), semantic_bindings.get(field)):
                blockers.append(f"semantic_binding_mismatch:{field}")
    if semantic_result.get("world") != request.get("world"):
        blockers.append("semantic_world_contract_mismatch")


def _validate_mapping(
    request: Mapping[str, Any],
    gaussian_mapping: Sequence[Mapping[str, Any]],
    blockers: list[str],
) -> Dict[int, Mapping[str, Any]]:
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    try:
        digest = canonical_json_digest(gaussian_mapping)
    except (TypeError, ValueError):
        blockers.append("gaussian_mapping_not_canonical_json")
        return {}
    if not _same_digest(bindings.get("gaussian_mapping_digest"), digest):
        blockers.append("gaussian_mapping_digest_mismatch")
    gaussian_count = _positive_int(request.get("gaussian_count"))
    if gaussian_count is None:
        blockers.append("gaussian_count_invalid")
        return {}
    if len(gaussian_mapping) != gaussian_count:
        blockers.append("gaussian_mapping_count_mismatch")
    by_id: Dict[int, Mapping[str, Any]] = {}
    source_indices: set[int] = set()
    for row in gaussian_mapping:
        if not isinstance(row, Mapping):
            blockers.append("gaussian_mapping_row_invalid")
            continue
        gaussian_id = row.get("gaussian_id")
        source_index = row.get("source_index")
        source_class = str(row.get("source_class") or "").strip().lower()
        if (
            not isinstance(gaussian_id, int)
            or isinstance(gaussian_id, bool)
            or gaussian_id < 0
            or gaussian_id >= gaussian_count
            or gaussian_id in by_id
        ):
            blockers.append("gaussian_mapping_id_invalid_or_duplicate")
            continue
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or source_index < 0
            or source_index in source_indices
        ):
            blockers.append("gaussian_source_index_invalid_or_duplicate")
            continue
        if source_class not in {"observed", "generated", "unknown"}:
            blockers.append("gaussian_source_class_invalid")
            continue
        by_id[gaussian_id] = row
        source_indices.add(source_index)
    if set(by_id) != set(range(gaussian_count)):
        blockers.append("gaussian_mapping_ids_not_complete")
    return by_id


def _validate_support_points(
    request: Mapping[str, Any],
    support_points: Sequence[Mapping[str, Any]],
    mapping: Mapping[int, Mapping[str, Any]],
    blockers: list[str],
) -> list[Dict[str, Any]]:
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    try:
        digest = canonical_json_digest(support_points)
    except (TypeError, ValueError):
        blockers.append("support_points_not_canonical_json")
        return []
    if not _same_digest(bindings.get("support_points_digest"), digest):
        blockers.append("support_points_digest_mismatch")
    if len(support_points) > _MAX_POINTS:
        blockers.append("support_points_exceed_limit")
    normalized: list[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in support_points:
        if not isinstance(row, Mapping):
            blockers.append("support_point_row_invalid")
            continue
        point_id = str(row.get("point_id") or "").strip()
        gaussian_id = row.get("gaussian_id")
        point_source = str(row.get("point_source") or "").strip().lower()
        coordinates = row.get("point_world_m")
        if not point_id or point_id in seen_ids:
            blockers.append("support_point_id_missing_or_duplicate")
            continue
        if not isinstance(gaussian_id, int) or isinstance(gaussian_id, bool) or gaussian_id not in mapping:
            blockers.append(f"support_point_gaussian_id_invalid:{point_id}")
            continue
        if point_source not in _POINT_SOURCES:
            blockers.append(f"support_point_source_invalid:{point_id}")
            continue
        if (
            not isinstance(coordinates, Sequence)
            or isinstance(coordinates, (str, bytes))
            or len(coordinates) != 3
        ):
            blockers.append(f"support_point_coordinates_invalid:{point_id}")
            continue
        values = [_finite(value) for value in coordinates]
        if any(value is None for value in values):
            blockers.append(f"support_point_coordinates_invalid:{point_id}")
            continue
        seen_ids.add(point_id)
        normalized.append(
            {
                "point_id": point_id,
                "gaussian_id": gaussian_id,
                "point_source": point_source,
                "point": tuple(float(value) for value in values),
            }
        )
    return normalized


def _mad(values: Sequence[float]) -> tuple[float, float]:
    center = float(median(values))
    deviation = float(median(abs(value - center) for value in values))
    return center, deviation


def _robust_inliers(
    rows: Sequence[Mapping[str, Any]], multiplier: float
) -> tuple[list[Mapping[str, Any]], list[str]]:
    centers_and_mads = [_mad([float(row["point"][axis]) for row in rows]) for axis in range(3)]
    kept: list[Mapping[str, Any]] = []
    removed: list[str] = []
    for row in rows:
        accepted = True
        for axis, (center, axis_mad) in enumerate(centers_and_mads):
            if axis_mad > _EPS and abs(float(row["point"][axis]) - center) > multiplier * axis_mad:
                accepted = False
                break
        if accepted:
            kept.append(row)
        else:
            removed.append(str(row["point_id"]))
    return kept, sorted(removed)


def _cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (
        first[1] - origin[1]
    ) * (second[0] - origin[0])


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _normalize_yaw(yaw: float) -> float:
    while yaw < -0.5 * math.pi:
        yaw += math.pi
    while yaw >= 0.5 * math.pi:
        yaw -= math.pi
    return yaw


def _minimum_area_rectangle(
    points: Sequence[tuple[float, float]],
) -> Dict[str, float] | None:
    hull = _convex_hull(points)
    if len(hull) < 3:
        return None
    best: tuple[tuple[float, float, float], Dict[str, float]] | None = None
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        edge_x, edge_y = end[0] - start[0], end[1] - start[1]
        if abs(edge_x) <= _EPS and abs(edge_y) <= _EPS:
            continue
        angle = math.atan2(edge_y, edge_x)
        cosine, sine = math.cos(angle), math.sin(angle)
        local_x = [point[0] * cosine + point[1] * sine for point in hull]
        local_y = [-point[0] * sine + point[1] * cosine for point in hull]
        min_x, max_x = min(local_x), max(local_x)
        min_y, max_y = min(local_y), max(local_y)
        length, width = max_x - min_x, max_y - min_y
        original_center_x = 0.5 * (min_x + max_x)
        original_center_y = 0.5 * (min_y + max_y)
        center_x = original_center_x * cosine - original_center_y * sine
        center_y = original_center_x * sine + original_center_y * cosine
        if width > length:
            angle += 0.5 * math.pi
            length, width = width, length
        yaw = _normalize_yaw(angle)
        candidate = {
            "center_x": center_x,
            "center_y": center_y,
            "length": length,
            "width": width,
            "yaw": yaw,
        }
        key = (round(length * width, 12), round(abs(yaw), 12), round(yaw, 12))
        if best is None or key < best[0]:
            best = (key, candidate)
    return best[1] if best is not None else None


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _corners(box: Mapping[str, float], z_min: float, z_max: float) -> list[list[float]]:
    half_length, half_width = 0.5 * box["length"], 0.5 * box["width"]
    cosine, sine = math.cos(box["yaw"]), math.sin(box["yaw"])
    rows: list[list[float]] = []
    for z in (z_min, z_max):
        for local_x, local_y in (
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width),
        ):
            rows.append(
                [
                    round(box["center_x"] + local_x * cosine - local_y * sine, 9),
                    round(box["center_y"] + local_x * sine + local_y * cosine, 9),
                    round(z, 9),
                ]
            )
    return rows


def _next_experiment(reasons: Sequence[str]) -> str:
    if "semantic_track_not_qualified" in reasons:
        return "capture_or_render_additional_track_views_before_box_fitting"
    if "generated_or_unknown_gaussian_in_support" in reasons:
        return "recapture_the_object_region_with_direct_observed_geometry"
    if "insufficient_support_points" in reasons or "insufficient_distinct_gaussians" in reasons:
        return "capture_a_close_orbit_with_depth_around_the_object"
    if "insufficient_horizontal_extent" in reasons:
        return "capture_the_object_from_separated_horizontal_viewpoints"
    if "insufficient_vertical_extent" in reasons:
        return "capture_the_object_top_bottom_and_support_contact"
    if "dimension_exceeds_configured_limit" in reasons:
        return "verify_metric_scale_and_the_site_to_object_transform"
    if "inlier_fraction_below_threshold" in reasons:
        return "recapture_with_less_occlusion_and_remove_dynamic_objects"
    return "review_the_semantic_support_and_supply_verified_surface_points"


def fit_semantic_oriented_boxes(
    request: Mapping[str, Any],
    *,
    semantic_result: Mapping[str, Any],
    gaussian_mapping: Sequence[Mapping[str, Any]],
    support_points: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return deterministic OBB candidates or explicit per-track abstentions."""

    blockers: list[str] = []
    _validate_request(request, blockers)
    qualification = _qualification(request, blockers)
    _validate_semantic_result(request, semantic_result, blockers)
    mapping = _validate_mapping(request, gaussian_mapping, blockers)
    normalized_points = _validate_support_points(request, support_points, mapping, blockers)
    if blockers:
        return _blocked(request, blockers)

    points_by_gaussian: Dict[int, list[Mapping[str, Any]]] = {}
    for row in normalized_points:
        points_by_gaussian.setdefault(int(row["gaussian_id"]), []).append(row)

    object_rows: list[Dict[str, Any]] = []
    semantic_tracks = semantic_result.get("tracks")
    if not isinstance(semantic_tracks, Sequence) or isinstance(semantic_tracks, (str, bytes)):
        return _blocked(request, ["semantic_tracks_invalid"])
    for track in sorted(semantic_tracks, key=lambda row: str(row.get("track_id") or "")):
        if not isinstance(track, Mapping):
            return _blocked(request, ["semantic_track_row_invalid"])
        track_id = str(track.get("track_id") or "").strip()
        selected_ids = track.get("selected_gaussian_ids")
        if not track_id or not isinstance(selected_ids, Sequence) or isinstance(selected_ids, (str, bytes)):
            return _blocked(request, ["semantic_track_identity_or_support_invalid"])
        gaussian_ids: list[int] = []
        for gaussian_id in selected_ids:
            if not isinstance(gaussian_id, int) or isinstance(gaussian_id, bool) or gaussian_id not in mapping:
                return _blocked(request, [f"semantic_selected_gaussian_invalid:{track_id}"])
            gaussian_ids.append(gaussian_id)
        if len(set(gaussian_ids)) != len(gaussian_ids):
            return _blocked(request, [f"semantic_selected_gaussian_duplicate:{track_id}"])

        reasons: list[str] = []
        if track.get("status") != "qualified_semantic_support_candidate":
            reasons.append("semantic_track_not_qualified")
        if any(str(mapping[item].get("source_class") or "") != "observed" for item in gaussian_ids):
            reasons.append("generated_or_unknown_gaussian_in_support")
        rows = [row for gaussian_id in sorted(gaussian_ids) for row in points_by_gaussian.get(gaussian_id, [])]
        if len(rows) < int(qualification["min_support_points"]):
            reasons.append("insufficient_support_points")
        if len({int(row["gaussian_id"]) for row in rows}) < int(
            qualification["min_distinct_gaussians"]
        ):
            reasons.append("insufficient_distinct_gaussians")

        inliers: list[Mapping[str, Any]] = []
        removed_ids: list[str] = []
        box: Dict[str, float] | None = None
        z_min = z_max = 0.0
        if rows:
            inliers, removed_ids = _robust_inliers(
                rows, float(qualification["outlier_mad_multiplier"])
            )
            if len(inliers) / len(rows) + _EPS < float(qualification["min_inlier_fraction"]):
                reasons.append("inlier_fraction_below_threshold")
            if len(inliers) < int(qualification["min_support_points"]):
                reasons.append("insufficient_support_points_after_outlier_rejection")
            if len({int(row["gaussian_id"]) for row in inliers}) < int(
                qualification["min_distinct_gaussians"]
            ):
                reasons.append("insufficient_distinct_gaussians_after_outlier_rejection")
            box = _minimum_area_rectangle(
                [(float(row["point"][0]), float(row["point"][1])) for row in inliers]
            )
            if box is None or min(box["length"], box["width"]) + _EPS < float(
                qualification["min_horizontal_extent_m"]
            ):
                reasons.append("insufficient_horizontal_extent")
            verticals = [float(row["point"][2]) for row in inliers]
            if verticals:
                trim = float(qualification["vertical_trim_fraction"])
                z_min, z_max = _quantile(verticals, trim), _quantile(verticals, 1.0 - trim)
                if z_max - z_min + _EPS < float(qualification["min_vertical_extent_m"]):
                    reasons.append("insufficient_vertical_extent")
            if box is not None and max(box["length"], box["width"], z_max - z_min) > float(
                qualification["max_dimension_m"]
            ):
                reasons.append("dimension_exceeds_configured_limit")

        reasons = sorted(set(reasons))
        object_row: Dict[str, Any] = {
            "track_id": track_id,
            "label": str(track.get("label") or "").strip(),
            "status": "qualified_metric_obb_candidate" if not reasons else "abstained",
            "semantic_lifting_result_digest": semantic_result["result_digest"],
            "selected_gaussian_ids": sorted(gaussian_ids),
            "input_support_point_count": len(rows),
            "inlier_support_point_count": len(inliers),
            "removed_outlier_point_ids": removed_ids,
            "point_sources": sorted({str(row["point_source"]) for row in inliers}),
            "abstention_reasons": reasons,
            "metric_obb_candidate_ready": not reasons,
            "collision_ready": False,
            "physics_ready": False,
        }
        if not reasons and box is not None:
            point_sources = object_row["point_sources"]
            claim_ceiling = (
                "metric_obb_candidate_from_observed_surface_support"
                if set(point_sources) <= {"observed_depth", "verified_mesh_surface"}
                else "approximate_metric_obb_candidate_from_gaussian_centers"
            )
            object_row.update(
                {
                    "claim_ceiling": claim_ceiling,
                    "center_world_m": [
                        round(box["center_x"], 9),
                        round(box["center_y"], 9),
                        round(0.5 * (z_min + z_max), 9),
                    ],
                    "dimensions_m": [
                        round(box["length"], 9),
                        round(box["width"], 9),
                        round(z_max - z_min, 9),
                    ],
                    "yaw_rad": round(box["yaw"], 12),
                    "axes_world": [
                        [round(math.cos(box["yaw"]), 12), round(math.sin(box["yaw"]), 12), 0.0],
                        [round(-math.sin(box["yaw"]), 12), round(math.cos(box["yaw"]), 12), 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "corners_world_m": _corners(box, z_min, z_max),
                    "coordinate_frame": "analysis_splat_z_up_meters",
                    "units": "meters",
                    "fit_provenance": {
                        "method": FIT_METHOD,
                        "support_method_profile": dict(request["support_method_profile"]),
                        "horizontal_fit": "minimum_area_rectangle_over_robust_inliers",
                        "vertical_fit": "independent_trimmed_limits",
                        "outlier_policy": "coordinatewise_median_absolute_deviation",
                    },
                }
            )
        else:
            object_row["claim_ceiling"] = "none_object_box_abstained"
            object_row["next_experiment"] = _next_experiment(reasons)
        object_rows.append(object_row)

    qualified_count = sum(row["status"] == "qualified_metric_obb_candidate" for row in object_rows)
    if qualified_count == len(object_rows) and object_rows:
        status = "completed"
    elif qualified_count:
        status = "partially_completed"
    else:
        status = "abstained"
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "bindings": dict(request["bindings"]),
        "fit_method": FIT_METHOD,
        "world": dict(request["world"]),
        "objects": object_rows,
        "qualified_object_count": qualified_count,
        "abstained_object_count": len(object_rows) - qualified_count,
        "blockers": [],
        "claim_ceiling": "qualified_metric_obb_candidates_only",
        "metric_obb_candidate_ready": qualified_count > 0,
        "collision_ready": False,
        "physics_ready": False,
        "generated_regions_can_upgrade_claims": False,
        "prohibited_claims": [
            "collision_or_contact_truth",
            "mass_friction_articulation_or_support_truth",
            "task_or_physical_success",
            "safety_or_deployment_readiness",
        ],
    }
    result["result_digest"] = canonical_json_digest(result)
    return result
