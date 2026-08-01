"""Deterministic evaluation of metric semantic OBB candidates.

The benchmark consumes an exact semantic-oriented-box result and an independently
reviewed metric Z-up reference annotation.  It reports detection and geometry
quality plus stability under declared view-removal reruns.  Benchmark scores are
diagnostic evidence: they do not qualify collision geometry, physics, physical
task success, safety, deployment, or comparative policy ranking.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Dict, Mapping, Sequence

from .semantic_gaussian_lifting import canonical_json_digest


REQUEST_SCHEMA_VERSION = "semantic_geometry_benchmark_request.v1"
RESULT_SCHEMA_VERSION = "semantic_geometry_benchmark_result.v1"
GROUND_TRUTH_SCHEMA_VERSION = "semantic_geometry_ground_truth.v1"
BENCHMARK_METHOD = "independent_metric_obb_benchmark.v1"
_EPS = 1e-12
_MAX_OBJECTS = 512
_MAX_ABLATIONS = 128


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
    x, y, z = (_finite(item) for item in value)
    if x is None or y is None or z is None:
        return None
    return x, y, z


def _result_digest_valid(payload: Mapping[str, Any], field: str) -> bool:
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
        "benchmark_method": BENCHMARK_METHOD,
        "counts": {},
        "metrics": {},
        "matches": [],
        "view_ablation": [],
        "blockers": sorted(set(blockers)),
        "claim_ceiling": "none_invalid_or_unbound_benchmark_input",
        "benchmark_diagnostic_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "prohibited_claims": [
            "collision_or_contact_truth",
            "physical_task_success",
            "safety_or_deployment_readiness",
            "comparative_policy_ranking_support",
        ],
    }
    result["result_digest"] = canonical_json_digest(result)
    return result


def _validate_request(request: Mapping[str, Any], blockers: list[str]) -> Dict[str, float]:
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("request_schema_version_unsupported")
    if request.get("benchmark_method") != BENCHMARK_METHOD:
        blockers.append("benchmark_method_unsupported")
    bindings = request.get("bindings")
    if not isinstance(bindings, Mapping):
        blockers.append("bindings_missing")
        bindings = {}
    for field in (
        "capture_digest",
        "reconstruction_digest",
        "analysis_splat_digest",
        "prediction_result_digest",
        "ground_truth_digest",
        "benchmark_profile_digest",
        "prediction_input_manifest_digest",
        "evaluation_view_registry_digest",
        "view_ablation_manifest_digest",
    ):
        if not _valid_digest(bindings.get(field)):
            blockers.append(f"binding_digest_invalid:{field}")
    world = request.get("world")
    if not isinstance(world, Mapping):
        blockers.append("world_contract_missing")
    else:
        if str(world.get("up_axis") or "").upper() != "Z":
            blockers.append("world_up_axis_must_be_z")
        if str(world.get("units") or "").lower() != "meters":
            blockers.append("world_units_must_be_meters")
        if world.get("scale_verified") is not True:
            blockers.append("verified_metric_scale_required")
        if not str(world.get("coordinate_frame") or "").strip():
            blockers.append("world_coordinate_frame_missing")

    profile = request.get("benchmark_profile")
    if not isinstance(profile, Mapping):
        blockers.append("benchmark_profile_missing")
        return {}
    for field in ("evaluator_id", "evaluator_version", "evaluator_identity", "split_id"):
        if not str(profile.get(field) or "").strip():
            blockers.append(f"benchmark_profile_missing:{field}")
    if not _valid_digest(profile.get("runtime_digest")):
        blockers.append("benchmark_runtime_digest_invalid")
    if profile.get("deterministic") is not True:
        blockers.append("benchmark_determinism_required")
    if profile.get("prediction_input_manifest_complete") is not True:
        blockers.append("prediction_input_manifest_completeness_required")
    claimed = profile.get("benchmark_profile_digest")
    unsigned = dict(profile)
    unsigned.pop("benchmark_profile_digest", None)
    try:
        actual_profile_digest = canonical_json_digest(unsigned)
    except (TypeError, ValueError):
        actual_profile_digest = ""
    if not _valid_digest(claimed) or not _same_digest(claimed, actual_profile_digest):
        blockers.append("benchmark_profile_digest_invalid")
    if not _same_digest(bindings.get("benchmark_profile_digest"), claimed):
        blockers.append("benchmark_profile_digest_mismatch")
    ranges = {
        "max_match_center_distance_m": (0.001, 100.0),
        "min_match_3d_iou": (0.0, 1.0),
        "adjacent_same_label_distance_m": (0.001, 100.0),
        "square_yaw_ambiguity_ratio": (1.0, 2.0),
    }
    normalized: Dict[str, float] = {}
    for field, (minimum, maximum) in ranges.items():
        value = _finite(profile.get(field))
        if value is None or value < minimum or value > maximum:
            blockers.append(f"benchmark_profile_invalid:{field}")
        else:
            normalized[field] = value
    view_ids = request.get("evaluation_view_ids")
    if (
        not isinstance(view_ids, Sequence)
        or isinstance(view_ids, (str, bytes))
        or not view_ids
        or any(not str(item).strip() for item in view_ids)
        or len({str(item) for item in view_ids}) != len(view_ids)
    ):
        blockers.append("evaluation_view_registry_invalid")
    else:
        registry = sorted(str(item) for item in view_ids)
        if not _same_digest(
            bindings.get("evaluation_view_registry_digest"), canonical_json_digest(registry)
        ):
            blockers.append("evaluation_view_registry_digest_mismatch")
    prediction_inputs = request.get("prediction_input_digests")
    if (
        not isinstance(prediction_inputs, Sequence)
        or isinstance(prediction_inputs, (str, bytes))
        or not prediction_inputs
        or any(not _valid_digest(item) for item in prediction_inputs)
        or len({str(item).lower() for item in prediction_inputs}) != len(prediction_inputs)
    ):
        blockers.append("prediction_input_manifest_invalid")
    else:
        normalized_inputs = sorted(str(item).lower() for item in prediction_inputs)
        if not _same_digest(
            bindings.get("prediction_input_manifest_digest"),
            canonical_json_digest(normalized_inputs),
        ):
            blockers.append("prediction_input_manifest_digest_mismatch")
    return normalized


def _polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    )


def _signed_area(polygon: Sequence[tuple[float, float]]) -> float:
    return 0.5 * sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )


def _inside(
    point: tuple[float, float],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
    orientation: float,
) -> bool:
    cross = (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (
        edge_end[1] - edge_start[1]
    ) * (point[0] - edge_start[0])
    return cross * orientation >= -_EPS


def _line_intersection(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> tuple[float, float]:
    first_dx, first_dy = first_end[0] - first_start[0], first_end[1] - first_start[1]
    second_dx, second_dy = second_end[0] - second_start[0], second_end[1] - second_start[1]
    denominator = first_dx * second_dy - first_dy * second_dx
    if abs(denominator) <= _EPS:
        return first_end
    offset_x, offset_y = second_start[0] - first_start[0], second_start[1] - first_start[1]
    scale = (offset_x * second_dy - offset_y * second_dx) / denominator
    return first_start[0] + scale * first_dx, first_start[1] + scale * first_dy


def _clip_polygon(
    subject: Sequence[tuple[float, float]], clip: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    output = list(subject)
    orientation = 1.0 if _signed_area(clip) >= 0.0 else -1.0
    for index, edge_start in enumerate(clip):
        edge_end = clip[(index + 1) % len(clip)]
        incoming, output = output, []
        if not incoming:
            break
        previous = incoming[-1]
        for current in incoming:
            current_inside = _inside(current, edge_start, edge_end, orientation)
            previous_inside = _inside(previous, edge_start, edge_end, orientation)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            previous = current
    return output


def _corners(
    center: tuple[float, float, float], dimensions: tuple[float, float, float], yaw: float
) -> list[list[float]]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    half_x, half_y, half_z = (value * 0.5 for value in dimensions)
    rows: list[list[float]] = []
    for z_offset in (-half_z, half_z):
        for x_offset, y_offset in (
            (-half_x, -half_y),
            (half_x, -half_y),
            (half_x, half_y),
            (-half_x, half_y),
        ):
            rows.append(
                [
                    center[0] + x_offset * cosine - y_offset * sine,
                    center[1] + x_offset * sine + y_offset * cosine,
                    center[2] + z_offset,
                ]
            )
    return rows


def _box(row: Mapping[str, Any], *, identifier_field: str) -> Dict[str, Any] | None:
    identifier = str(row.get(identifier_field) or "").strip()
    label = str(row.get("label") or "").strip().casefold()
    center = _vector3(row.get("center_world_m"))
    dimensions = _vector3(row.get("dimensions_m"))
    yaw = _finite(row.get("yaw_rad"))
    corners = row.get("corners_world_m")
    if (
        not identifier
        or not label
        or center is None
        or dimensions is None
        or any(value <= 0.0 for value in dimensions)
        or yaw is None
        or not isinstance(corners, Sequence)
        or isinstance(corners, (str, bytes))
        or len(corners) != 8
    ):
        return None
    actual_corners = [_vector3(corner) for corner in corners]
    if any(corner is None for corner in actual_corners):
        return None
    expected_corners = _corners(center, dimensions, yaw)
    unmatched = [tuple(corner) for corner in actual_corners if corner is not None]
    for expected in expected_corners:
        best = min(range(len(unmatched)), key=lambda index: math.dist(expected, unmatched[index]))
        if math.dist(expected, unmatched[best]) > 1e-5:
            return None
        unmatched.pop(best)
    polygon = [(row[0], row[1]) for row in expected_corners[:4]]
    return {
        "id": identifier,
        "label": label,
        "center": center,
        "dimensions": dimensions,
        "yaw": yaw,
        "polygon": polygon,
        "z_min": center[2] - 0.5 * dimensions[2],
        "z_max": center[2] + 0.5 * dimensions[2],
        "volume": dimensions[0] * dimensions[1] * dimensions[2],
    }


def _iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    z_overlap = max(
        0.0,
        min(float(first["z_max"]), float(second["z_max"]))
        - max(float(first["z_min"]), float(second["z_min"])),
    )
    intersection = _polygon_area(_clip_polygon(first["polygon"], second["polygon"])) * z_overlap
    union = float(first["volume"]) + float(second["volume"]) - intersection
    return 0.0 if union <= _EPS else max(0.0, min(1.0, intersection / union))


def _yaw_delta(first: float, second: float) -> float:
    return abs((first - second + 0.5 * math.pi) % math.pi - 0.5 * math.pi)


def _geometry_errors(
    prediction: Mapping[str, Any], reference: Mapping[str, Any], square_ratio: float
) -> Dict[str, Any]:
    center_error = math.dist(prediction["center"], reference["center"])
    pred_dims = prediction["dimensions"]
    ref_dims = reference["dimensions"]
    variants = [
        (pred_dims, float(prediction["yaw"])),
        ((pred_dims[1], pred_dims[0], pred_dims[2]), float(prediction["yaw"]) + 0.5 * math.pi),
    ]
    chosen_dims, chosen_yaw = min(
        variants,
        key=lambda item: (
            sum(abs(item[0][axis] - ref_dims[axis]) for axis in range(3)),
            _yaw_delta(item[1], float(reference["yaw"])),
        ),
    )
    dimension_errors = [abs(chosen_dims[axis] - ref_dims[axis]) for axis in range(3)]
    horizontal_ratio = max(ref_dims[0], ref_dims[1]) / min(ref_dims[0], ref_dims[1])
    yaw_error = (
        None
        if horizontal_ratio <= square_ratio
        else math.degrees(_yaw_delta(chosen_yaw, float(reference["yaw"])))
    )
    return {
        "center_error_m": center_error,
        "center_error_cm": center_error * 100.0,
        "dimension_abs_error_m": dimension_errors,
        "dimension_mean_abs_error_m": mean(dimension_errors),
        "yaw_error_deg": yaw_error,
        "obb_3d_iou": _iou(prediction, reference),
    }


def _hungarian(cost: Sequence[Sequence[float]]) -> list[int]:
    """Return minimum-cost column index per row for a square matrix."""

    size = len(cost)
    potentials_rows = [0.0] * (size + 1)
    potentials_cols = [0.0] * (size + 1)
    matching = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row in range(1, size + 1):
        matching[0] = row
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            active_row = matching[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    cost[active_row - 1][candidate - 1]
                    - potentials_rows[active_row]
                    - potentials_cols[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    potentials_rows[matching[candidate]] += delta
                    potentials_cols[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matching[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matching[column] = matching[previous]
            column = previous
            if column == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        if matching[column]:
            assignment[matching[column] - 1] = column - 1
    return assignment


def _match(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    max_center: float,
    min_iou: float,
) -> list[tuple[int, int]]:
    if not predictions or not references:
        return []
    reference_count, prediction_count = len(references), len(predictions)
    size = reference_count + prediction_count
    cost = [[0.0] * size for _ in range(size)]
    for ref_index in range(reference_count):
        for pred_index in range(prediction_count):
            center = math.dist(references[ref_index]["center"], predictions[pred_index]["center"])
            overlap = _iou(references[ref_index], predictions[pred_index])
            allowed = (
                references[ref_index]["label"] == predictions[pred_index]["label"]
                and center <= max_center + _EPS
                and overlap + _EPS >= min_iou
            )
            cost[ref_index][pred_index] = (
                1.0 - overlap + 0.01 * min(center / max_center, 1.0)
                if allowed
                else 10.0
            )
        for dummy in range(reference_count):
            cost[ref_index][prediction_count + dummy] = 2.0
    for dummy_row in range(prediction_count):
        row = reference_count + dummy_row
        for pred_index in range(prediction_count):
            cost[row][pred_index] = 2.0
        for dummy in range(reference_count):
            cost[row][prediction_count + dummy] = 0.0
    assignment = _hungarian(cost)
    return [
        (ref_index, assignment[ref_index])
        for ref_index in range(reference_count)
        if 0 <= assignment[ref_index] < prediction_count
        and cost[ref_index][assignment[ref_index]] < 2.0
    ]


def _prediction_boxes(
    result: Mapping[str, Any], blockers: list[str], *, prefix: str
) -> list[Dict[str, Any]]:
    if result.get("schema_version") != "semantic_oriented_box_result.v1":
        blockers.append(f"{prefix}_schema_unsupported")
        return []
    if not _result_digest_valid(result, "result_digest"):
        blockers.append(f"{prefix}_result_digest_invalid")
    if result.get("status") not in {"completed", "partially_completed", "abstained"}:
        blockers.append(f"{prefix}_not_terminal")
    if result.get("collision_ready") is not False or result.get("physics_ready") is not False:
        blockers.append(f"{prefix}_authority_boundary_invalid")
    if result.get("generated_regions_can_upgrade_claims") is not False:
        blockers.append(f"{prefix}_generated_region_boundary_invalid")
    raw = result.get("objects")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) > _MAX_OBJECTS:
        blockers.append(f"{prefix}_objects_invalid")
        return []
    boxes: list[Dict[str, Any]] = []
    identifiers: set[str] = set()
    for row in raw:
        if not isinstance(row, Mapping):
            blockers.append(f"{prefix}_object_row_invalid")
            continue
        if row.get("status") != "qualified_metric_obb_candidate":
            continue
        normalized = _box(row, identifier_field="track_id")
        if normalized is None or normalized["id"] in identifiers:
            blockers.append(f"{prefix}_qualified_object_invalid_or_duplicate")
            continue
        identifiers.add(normalized["id"])
        boxes.append(normalized)
    return sorted(boxes, key=lambda row: row["id"])


def _validate_ground_truth(
    request: Mapping[str, Any], ground_truth: Mapping[str, Any], blockers: list[str]
) -> list[Dict[str, Any]]:
    if ground_truth.get("schema_version") != GROUND_TRUTH_SCHEMA_VERSION:
        blockers.append("ground_truth_schema_unsupported")
    if not _result_digest_valid(ground_truth, "annotation_digest"):
        blockers.append("ground_truth_digest_invalid")
    raw_bindings = request.get("bindings")
    bindings: Mapping[str, Any] = raw_bindings if isinstance(raw_bindings, Mapping) else {}
    if not _same_digest(bindings.get("ground_truth_digest"), ground_truth.get("annotation_digest")):
        blockers.append("ground_truth_digest_mismatch")
    gt_bindings = ground_truth.get("bindings")
    if not isinstance(gt_bindings, Mapping):
        blockers.append("ground_truth_bindings_missing")
    else:
        for field in ("capture_digest", "reconstruction_digest"):
            if not _same_digest(bindings.get(field), gt_bindings.get(field)):
                blockers.append(f"ground_truth_binding_mismatch:{field}")
    if ground_truth.get("world") != request.get("world"):
        blockers.append("ground_truth_world_mismatch")
    profile = ground_truth.get("annotation_profile")
    evaluator = request.get("benchmark_profile")
    if not isinstance(profile, Mapping):
        blockers.append("ground_truth_annotation_profile_missing")
    else:
        for field in ("source_type", "producer_identity", "reviewer_identity"):
            if not str(profile.get(field) or "").strip():
                blockers.append(f"ground_truth_annotation_profile_missing:{field}")
        for field in ("source_artifact_digest", "alignment_digest"):
            if not _valid_digest(profile.get(field)):
                blockers.append(f"ground_truth_annotation_profile_invalid:{field}")
        if profile.get("metric_authority_verified") is not True:
            blockers.append("ground_truth_metric_authority_required")
        if profile.get("review_status") != "accepted":
            blockers.append("ground_truth_review_acceptance_required")
        if profile.get("withheld_from_prediction") is not True:
            blockers.append("ground_truth_prediction_leakage_forbidden")
        if profile.get("independent_from_prediction") is not True:
            blockers.append("ground_truth_prediction_independence_required")
        if profile.get("rights_cleared_for_evaluation") is not True:
            blockers.append("ground_truth_evaluation_rights_required")
        if profile.get("producer_identity") == profile.get("reviewer_identity"):
            blockers.append("ground_truth_independent_review_required")
        if isinstance(evaluator, Mapping) and profile.get("producer_identity") == evaluator.get(
            "evaluator_identity"
        ):
            blockers.append("ground_truth_producer_cannot_self_evaluate")
        if isinstance(evaluator, Mapping) and profile.get("reviewer_identity") == evaluator.get(
            "evaluator_identity"
        ):
            blockers.append("ground_truth_reviewer_cannot_self_evaluate")
        prediction_inputs = request.get("prediction_input_digests")
        normalized_inputs = (
            {str(item).strip().lower() for item in prediction_inputs}
            if isinstance(prediction_inputs, Sequence)
            and not isinstance(prediction_inputs, (str, bytes))
            else set()
        )
        for field in ("source_artifact_digest", "alignment_digest"):
            if str(profile.get(field) or "").strip().lower() in normalized_inputs:
                blockers.append(f"ground_truth_leaked_into_prediction_inputs:{field}")
        if str(ground_truth.get("annotation_digest") or "").strip().lower() in normalized_inputs:
            blockers.append("ground_truth_annotation_leaked_into_prediction_inputs")
    raw = ground_truth.get("objects")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        blockers.append("ground_truth_objects_missing")
        return []
    if len(raw) > _MAX_OBJECTS:
        blockers.append("ground_truth_object_limit_exceeded")
        return []
    boxes: list[Dict[str, Any]] = []
    identifiers: set[str] = set()
    for row in raw:
        normalized = _box(row, identifier_field="reference_object_id") if isinstance(row, Mapping) else None
        if normalized is None or normalized["id"] in identifiers:
            blockers.append("ground_truth_object_invalid_or_duplicate")
            continue
        identifiers.add(normalized["id"])
        boxes.append(normalized)
    return sorted(boxes, key=lambda row: row["id"])


def _validate_prediction_bindings(
    request: Mapping[str, Any], result: Mapping[str, Any], blockers: list[str], *, prefix: str
) -> None:
    raw_bindings = request.get("bindings")
    bindings: Mapping[str, Any] = raw_bindings if isinstance(raw_bindings, Mapping) else {}
    result_bindings = result.get("bindings")
    if not isinstance(result_bindings, Mapping):
        blockers.append(f"{prefix}_bindings_missing")
        return
    for field in ("capture_digest", "reconstruction_digest", "analysis_splat_digest"):
        if not _same_digest(bindings.get(field), result_bindings.get(field)):
            blockers.append(f"{prefix}_binding_mismatch:{field}")
    result_world = result.get("world")
    request_world = request.get("world")
    if not isinstance(result_world, Mapping) or result_world != request_world:
        blockers.append(f"{prefix}_world_mismatch")


def _ablation_manifest(ablation_runs: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for row in sorted(ablation_runs, key=lambda item: str(item.get("ablation_id") or "")):
        removed = row.get("removed_view_ids")
        normalized_removed = (
            sorted(str(item) for item in removed)
            if isinstance(removed, Sequence) and not isinstance(removed, (str, bytes))
            else []
        )
        prediction = row.get("prediction_result")
        rows.append(
            {
                "ablation_id": str(row.get("ablation_id") or "").strip(),
                "removed_view_ids": normalized_removed,
                "prediction_result_digest": (
                    prediction.get("result_digest") if isinstance(prediction, Mapping) else None
                ),
            }
        )
    return rows


def benchmark_semantic_geometry(
    request: Mapping[str, Any],
    *,
    prediction_result: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    ablation_runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate exact semantic OBB candidates against independent reference boxes."""

    blockers: list[str] = []
    if not isinstance(prediction_result, Mapping):
        blockers.append("prediction_result_must_be_object")
        prediction_result = {}
    if not isinstance(ground_truth, Mapping):
        blockers.append("ground_truth_must_be_object")
        ground_truth = {}
    profile = _validate_request(request, blockers)
    _validate_prediction_bindings(request, prediction_result, blockers, prefix="prediction")
    predictions = _prediction_boxes(prediction_result, blockers, prefix="prediction")
    raw_bindings = request.get("bindings")
    bindings: Mapping[str, Any] = raw_bindings if isinstance(raw_bindings, Mapping) else {}
    if not _same_digest(bindings.get("prediction_result_digest"), prediction_result.get("result_digest")):
        blockers.append("prediction_result_digest_mismatch")
    references = _validate_ground_truth(request, ground_truth, blockers)
    if (
        not isinstance(ablation_runs, Sequence)
        or isinstance(ablation_runs, (str, bytes))
        or len(ablation_runs) > _MAX_ABLATIONS
    ):
        blockers.append("view_ablation_runs_invalid")
        ablation_runs = []
    elif any(not isinstance(row, Mapping) for row in ablation_runs):
        blockers.append("view_ablation_row_invalid")
        ablation_runs = []
    manifest = _ablation_manifest(ablation_runs)
    try:
        manifest_digest = canonical_json_digest(manifest)
    except (TypeError, ValueError):
        manifest_digest = ""
    if not _same_digest(bindings.get("view_ablation_manifest_digest"), manifest_digest):
        blockers.append("view_ablation_manifest_digest_mismatch")
    if blockers:
        return _blocked(request, blockers)

    matches = _match(
        predictions,
        references,
        max_center=profile["max_match_center_distance_m"],
        min_iou=profile["min_match_3d_iou"],
    )
    matched_references = {row[0] for row in matches}
    matched_predictions = {row[1] for row in matches}
    match_rows: list[Dict[str, Any]] = []
    errors: list[Dict[str, Any]] = []
    for ref_index, pred_index in matches:
        error = _geometry_errors(
            predictions[pred_index],
            references[ref_index],
            profile["square_yaw_ambiguity_ratio"],
        )
        errors.append(error)
        match_rows.append(
            {
                "reference_object_id": references[ref_index]["id"],
                "track_id": predictions[pred_index]["id"],
                "label": references[ref_index]["label"],
                **{key: round(value, 9) if isinstance(value, float) else value for key, value in error.items()},
            }
        )

    adjacent_pairs: list[Dict[str, Any]] = []
    match_by_reference = {ref_index: pred_index for ref_index, pred_index in matches}
    for left in range(len(references)):
        for right in range(left + 1, len(references)):
            distance = math.dist(references[left]["center"], references[right]["center"])
            if (
                references[left]["label"] == references[right]["label"]
                and distance <= profile["adjacent_same_label_distance_m"] + _EPS
            ):
                separated = left in match_by_reference and right in match_by_reference
                adjacent_pairs.append(
                    {
                        "reference_object_ids": [references[left]["id"], references[right]["id"]],
                        "center_distance_m": round(distance, 9),
                        "both_instances_recovered_separately": separated,
                    }
                )

    baseline_by_track = {row["id"]: row for row in predictions}
    evaluation_view_ids = {str(item) for item in request["evaluation_view_ids"]}
    ablation_rows: list[Dict[str, Any]] = []
    seen_ablation_ids: set[str] = set()
    for run in sorted(ablation_runs, key=lambda row: str(row.get("ablation_id") or "")):
        ablation_id = str(run.get("ablation_id") or "").strip()
        removed = run.get("removed_view_ids")
        result = run.get("prediction_result")
        run_blockers: list[str] = []
        if not ablation_id or ablation_id in seen_ablation_ids:
            run_blockers.append("ablation_id_missing_or_duplicate")
        seen_ablation_ids.add(ablation_id)
        if (
            not isinstance(removed, Sequence)
            or isinstance(removed, (str, bytes))
            or not removed
            or any(not str(item).strip() for item in removed)
            or len({str(item) for item in removed}) != len(removed)
        ):
            run_blockers.append("removed_view_ids_invalid")
            removed_ids: list[str] = []
        elif not {str(item) for item in removed} <= evaluation_view_ids:
            run_blockers.append("removed_view_id_not_in_evaluation_registry")
            removed_ids = sorted(str(item) for item in removed)
        else:
            removed_ids = sorted(str(item) for item in removed)
        if not isinstance(result, Mapping):
            run_blockers.append("ablation_prediction_result_missing")
            result = {}
        _validate_prediction_bindings(request, result, run_blockers, prefix="ablation")
        ablation_boxes = _prediction_boxes(result, run_blockers, prefix="ablation")
        if _same_digest(result.get("result_digest"), prediction_result.get("result_digest")):
            run_blockers.append("ablation_must_not_reuse_baseline_result")
        if run_blockers:
            return _blocked(request, [f"view_ablation_invalid:{ablation_id}:{item}" for item in run_blockers])
        by_track = {row["id"]: row for row in ablation_boxes}
        shared = sorted(set(baseline_by_track) & set(by_track))
        unexpected = sorted(set(by_track) - set(baseline_by_track))
        union_count = len(set(baseline_by_track) | set(by_track))
        drift = [
            _geometry_errors(
                by_track[track_id],
                baseline_by_track[track_id],
                profile["square_yaw_ambiguity_ratio"],
            )
            for track_id in shared
        ]
        ablation_rows.append(
            {
                "ablation_id": ablation_id,
                "removed_view_ids": removed_ids,
                "prediction_result_digest": result["result_digest"],
                "baseline_track_count": len(baseline_by_track),
                "retained_track_count": len(shared),
                "retained_track_fraction": (
                    round(len(shared) / len(baseline_by_track), 9) if baseline_by_track else 1.0
                ),
                "unexpected_track_ids": unexpected,
                "unexpected_track_count": len(unexpected),
                "track_set_jaccard": round(len(shared) / union_count, 9) if union_count else 1.0,
                "mean_center_drift_cm": (
                    round(mean(row["center_error_cm"] for row in drift), 9) if drift else None
                ),
                "mean_dimension_drift_m": (
                    round(mean(row["dimension_mean_abs_error_m"] for row in drift), 9)
                    if drift
                    else None
                ),
                "mean_obb_3d_iou_to_baseline": (
                    round(mean(row["obb_3d_iou"] for row in drift), 9) if drift else None
                ),
            }
        )

    true_positive_count = len(matches)
    false_positive_count = len(predictions) - true_positive_count
    false_negative_count = len(references) - true_positive_count
    yaw_values = [row["yaw_error_deg"] for row in errors if row["yaw_error_deg"] is not None]
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed",
        "bindings": dict(bindings),
        "benchmark_method": BENCHMARK_METHOD,
        "counts": {
            "reference_objects": len(references),
            "predicted_objects": len(predictions),
            "true_positives": true_positive_count,
            "false_positives": false_positive_count,
            "false_negatives": false_negative_count,
        },
        "metrics": {
            "object_recall": round(true_positive_count / len(references), 9),
            "false_positive_fraction_of_predictions": (
                round(false_positive_count / len(predictions), 9) if predictions else 0.0
            ),
            "mean_center_error_cm": (
                round(mean(row["center_error_cm"] for row in errors), 9) if errors else None
            ),
            "mean_dimension_abs_error_m": (
                round(mean(row["dimension_mean_abs_error_m"] for row in errors), 9)
                if errors
                else None
            ),
            "mean_yaw_error_deg": round(mean(yaw_values), 9) if yaw_values else None,
            "yaw_evaluable_match_count": len(yaw_values),
            "mean_obb_3d_iou": (
                round(mean(row["obb_3d_iou"] for row in errors), 9) if errors else None
            ),
            "adjacent_same_label_pair_recall": (
                round(
                    sum(row["both_instances_recovered_separately"] for row in adjacent_pairs)
                    / len(adjacent_pairs),
                    9,
                )
                if adjacent_pairs
                else None
            ),
        },
        "matches": match_rows,
        "unmatched_reference_object_ids": [
            references[index]["id"] for index in range(len(references)) if index not in matched_references
        ],
        "unmatched_track_ids": [
            predictions[index]["id"] for index in range(len(predictions)) if index not in matched_predictions
        ],
        "adjacent_same_label_pairs": adjacent_pairs,
        "view_ablation": ablation_rows,
        "blockers": [],
        "claim_ceiling": "independent_semantic_geometry_benchmark_diagnostic_only",
        "benchmark_diagnostic_ready": True,
        "collision_ready": False,
        "physics_ready": False,
        "prohibited_claims": [
            "collision_or_contact_truth",
            "physical_task_success",
            "safety_or_deployment_readiness",
            "comparative_policy_ranking_support",
        ],
    }
    result["result_digest"] = canonical_json_digest(result)
    return result
