"""Lift tracked 2D masks onto stable Gaussian IDs without inventing geometry.

This module is the provider-neutral seam between a Gaussian renderer and later
instance fusion / oriented-box fitting.  The renderer must supply its actual
front-to-back ``transmittance * alpha`` contribution for each pixel and Gaussian.
Blueprint validates the embedded payload digests, exact camera/source bindings,
full-frame coverage, stable Gaussian mapping, and observed/generated-region
labels before accumulating foreground and background evidence.

The output is per-Gaussian semantic support only.  It is not an object box,
collision surface, physics asset, task-success decision, or physical evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence


REQUEST_SCHEMA_VERSION = "semantic_gaussian_lifting_request.v1"
RESULT_SCHEMA_VERSION = "semantic_gaussian_lifting_result.v1"
CONTRIBUTION_SEMANTICS = "front_to_back_transmittance_times_alpha"
_SOURCE_CLASSES = {"observed", "generated", "unknown"}
_MAX_TRACKS = 256
_MAX_VIEWS = 128
_MAX_PIXELS_PER_VIEW = 20_000_000
_MAX_CONTRIBUTIONS_PER_PIXEL = 4096


def canonical_json_digest(value: Any) -> str:
    """Return a stable ``sha256:`` digest for a JSON-compatible value."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_int(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value if value > 0 else None


def _unit_direction(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    components = [_finite_number(item) for item in value]
    if any(item is None for item in components):
        return None
    x, y, z = (float(item) for item in components)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-12:
        return None
    return (x / norm, y / norm, z / norm)


def _angle_degrees(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    dot = sum(first[index] * second[index] for index in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _angular_diversity(
    view_ids: Sequence[str],
    directions: Mapping[str, tuple[float, float, float]],
) -> float:
    ordered = sorted(set(view_ids))
    if len(ordered) < 2:
        return 0.0
    return max(
        _angle_degrees(directions[ordered[left]], directions[ordered[right]])
        for left in range(len(ordered))
        for right in range(left + 1, len(ordered))
    )


def _blocked_result(request: Mapping[str, Any], blockers: Sequence[str]) -> Dict[str, Any]:
    bindings = request.get("bindings")
    result: Dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "bindings": dict(bindings) if isinstance(bindings, Mapping) else {},
        "tracks": [],
        "blockers": sorted(set(blockers)),
        "claim_ceiling": "none_invalid_or_unbound_input",
        "canonical_object_geometry": False,
        "metric_box_ready": False,
        "physics_ready": False,
        "prohibited_claims": [
            "object_metric_dimensions",
            "collision_or_contact_truth",
            "task_or_physical_success",
        ],
    }
    result["result_digest"] = canonical_json_digest(result)
    return result


def _validate_bindings(request: Mapping[str, Any], blockers: list[str]) -> None:
    bindings = request.get("bindings")
    if not isinstance(bindings, Mapping):
        blockers.append("bindings_missing")
        return
    for field in (
        "capture_digest",
        "retained_video_digest",
        "reconstruction_digest",
        "analysis_splat_digest",
        "camera_solution_digest",
        "gaussian_mapping_digest",
        "frame_registry_digest",
        "track_registry_digest",
        "views_digest",
    ):
        if not _valid_digest(bindings.get(field)):
            blockers.append(f"binding_digest_invalid:{field}")


def _validate_renderer_profile(request: Mapping[str, Any], blockers: list[str]) -> None:
    profile = request.get("renderer_profile")
    if not isinstance(profile, Mapping):
        blockers.append("renderer_profile_missing")
        return
    if not str(profile.get("method_id") or "").strip():
        blockers.append("renderer_method_id_missing")
    if not str(profile.get("method_version") or "").strip():
        blockers.append("renderer_method_version_missing")
    if not _valid_digest(profile.get("runtime_digest")):
        blockers.append("renderer_runtime_digest_invalid")
    if profile.get("contribution_semantics") != CONTRIBUTION_SEMANTICS:
        blockers.append("renderer_contribution_semantics_unsupported")
    if profile.get("exact_gaussian_ids") is not True:
        blockers.append("renderer_exact_gaussian_ids_required")
    if profile.get("deterministic") is not True:
        blockers.append("renderer_determinism_required")


def _validate_world(request: Mapping[str, Any], blockers: list[str]) -> None:
    world = request.get("world")
    if not isinstance(world, Mapping):
        blockers.append("world_contract_missing")
        return
    if str(world.get("up_axis") or "").strip().upper() != "Z":
        blockers.append("world_up_axis_must_be_z")
    if str(world.get("units") or "").strip().lower() != "meters":
        blockers.append("world_units_must_be_meters")
    if not isinstance(world.get("scale_verified"), bool):
        blockers.append("world_scale_verified_must_be_boolean")


def _validate_mapping(
    request: Mapping[str, Any],
    gaussian_mapping: Sequence[Mapping[str, Any]],
    blockers: list[str],
) -> Dict[int, str]:
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    try:
        actual_digest = canonical_json_digest(gaussian_mapping)
    except (TypeError, ValueError):
        blockers.append("gaussian_mapping_not_canonical_json")
        return {}
    if not _same_digest(bindings.get("gaussian_mapping_digest"), actual_digest):
        blockers.append("gaussian_mapping_digest_mismatch")

    gaussian_count = _positive_int(request.get("gaussian_count"))
    if gaussian_count is None:
        blockers.append("gaussian_count_invalid")
        return {}
    if len(gaussian_mapping) != gaussian_count:
        blockers.append("gaussian_mapping_count_mismatch")
    source_classes: Dict[int, str] = {}
    source_indices: set[int] = set()
    for row in gaussian_mapping:
        if not isinstance(row, Mapping):
            blockers.append("gaussian_mapping_row_invalid")
            continue
        gaussian_id = row.get("gaussian_id")
        source_index = row.get("source_index")
        if (
            not isinstance(gaussian_id, int)
            or isinstance(gaussian_id, bool)
            or gaussian_id < 0
            or gaussian_id >= gaussian_count
        ):
            blockers.append("gaussian_mapping_id_invalid")
            continue
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or source_index < 0
            or source_index in source_indices
        ):
            blockers.append("gaussian_source_index_invalid_or_duplicate")
            continue
        source_class = str(row.get("source_class") or "").strip().lower()
        if source_class not in _SOURCE_CLASSES:
            blockers.append("gaussian_source_class_invalid")
            continue
        if gaussian_id in source_classes:
            blockers.append("gaussian_mapping_id_duplicate")
            continue
        source_indices.add(source_index)
        source_classes[gaussian_id] = source_class
    if set(source_classes) != set(range(gaussian_count)):
        blockers.append("gaussian_mapping_ids_not_complete")
    return source_classes


def _validate_tracks(
    request: Mapping[str, Any],
    track_registry: Sequence[Mapping[str, Any]],
    blockers: list[str],
) -> Dict[str, Dict[str, Any]]:
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    try:
        actual_digest = canonical_json_digest(track_registry)
    except (TypeError, ValueError):
        blockers.append("track_registry_not_canonical_json")
        return {}
    if not _same_digest(bindings.get("track_registry_digest"), actual_digest):
        blockers.append("track_registry_digest_mismatch")
    tracks: Dict[str, Dict[str, Any]] = {}
    if len(track_registry) > _MAX_TRACKS:
        blockers.append("track_registry_exceeds_limit")
    for row in track_registry:
        if not isinstance(row, Mapping):
            blockers.append("track_registry_row_invalid")
            continue
        track_id = str(row.get("track_id") or "").strip()
        label = str(row.get("label") or "").strip()
        if not track_id or not label:
            blockers.append("track_id_or_label_missing")
            continue
        if track_id in tracks:
            blockers.append("track_id_duplicate")
            continue
        if not _valid_digest(row.get("mask_model_digest")):
            blockers.append(f"track_mask_model_digest_invalid:{track_id}")
        if not _valid_digest(row.get("track_evidence_digest")):
            blockers.append(f"track_evidence_digest_invalid:{track_id}")
        tracks[track_id] = dict(row)
    if not tracks:
        blockers.append("track_registry_empty")
    return tracks


def _validate_frame_registry(
    request: Mapping[str, Any], blockers: list[str]
) -> Dict[str, Dict[str, Any]]:
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    raw_registry = request.get("frame_registry")
    if not isinstance(raw_registry, list):
        blockers.append("frame_registry_missing")
        return {}
    try:
        actual_digest = canonical_json_digest(raw_registry)
    except (TypeError, ValueError):
        blockers.append("frame_registry_not_canonical_json")
        return {}
    if not _same_digest(bindings.get("frame_registry_digest"), actual_digest):
        blockers.append("frame_registry_digest_mismatch")
    frames: Dict[str, Dict[str, Any]] = {}
    for row in raw_registry:
        if not isinstance(row, Mapping):
            blockers.append("frame_registry_row_invalid")
            continue
        frame_id = str(row.get("source_frame_id") or "").strip()
        if not frame_id or frame_id in frames:
            blockers.append("frame_registry_id_missing_or_duplicate")
            continue
        for field in (
            "source_frame_digest",
            "retained_video_digest",
            "sync_map_row_digest",
            "camera_record_digest",
        ):
            if not _valid_digest(row.get(field)):
                blockers.append(f"frame_registry_digest_invalid:{frame_id}:{field}")
        if not _same_digest(
            row.get("retained_video_digest"), bindings.get("retained_video_digest")
        ):
            blockers.append(f"frame_registry_retained_video_mismatch:{frame_id}")
        decoded_pts = _finite_number(row.get("decoded_pts_seconds"))
        if decoded_pts is None or decoded_pts < 0.0:
            blockers.append(f"frame_registry_decoded_pts_invalid:{frame_id}")
        if row.get("encoder_retained") is not True:
            blockers.append(f"frame_registry_encoder_retention_not_proven:{frame_id}")
        frames[frame_id] = dict(row)
    if not frames:
        blockers.append("frame_registry_empty")
    return frames


def _validate_qualification(
    request: Mapping[str, Any], blockers: list[str]
) -> Dict[str, float | int]:
    raw = request.get("qualification")
    if not isinstance(raw, Mapping):
        blockers.append("qualification_contract_missing")
        return {}
    values: Dict[str, float | int] = {}
    for field in ("min_track_views", "min_gaussian_views"):
        value = _positive_int(raw.get(field))
        if value is None:
            blockers.append(f"qualification_invalid:{field}")
        else:
            values[field] = value
    for field in (
        "min_view_foreground_contribution",
        "min_gaussian_view_foreground_contribution",
        "min_gaussian_total_contribution",
        "foreground_probability_threshold",
        "min_angular_diversity_degrees",
    ):
        value = _finite_number(raw.get(field))
        if value is None or value < 0.0:
            blockers.append(f"qualification_invalid:{field}")
        else:
            values[field] = value
    threshold = values.get("foreground_probability_threshold")
    if isinstance(threshold, float) and not 0.5 < threshold <= 1.0:
        blockers.append("qualification_probability_threshold_out_of_range")
    angle = values.get("min_angular_diversity_degrees")
    if isinstance(angle, float) and angle > 180.0:
        blockers.append("qualification_angular_diversity_out_of_range")
    return values


def _validate_views(
    request: Mapping[str, Any],
    views: Sequence[Mapping[str, Any]],
    tracks: Mapping[str, Mapping[str, Any]],
    frames: Mapping[str, Mapping[str, Any]],
    source_classes: Mapping[int, str],
    blockers: list[str],
) -> Dict[str, tuple[float, float, float]]:
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    try:
        actual_digest = canonical_json_digest(views)
    except (TypeError, ValueError):
        blockers.append("views_not_canonical_json")
        return {}
    if not _same_digest(bindings.get("views_digest"), actual_digest):
        blockers.append("views_digest_mismatch")
    directions: Dict[str, tuple[float, float, float]] = {}
    seen_views: set[str] = set()
    if len(views) > _MAX_VIEWS:
        blockers.append("view_count_exceeds_limit")
    for view in views:
        if not isinstance(view, Mapping):
            blockers.append("view_record_invalid")
            continue
        view_id = str(view.get("view_id") or "").strip()
        if not view_id or view_id in seen_views:
            blockers.append("view_id_missing_or_duplicate")
            continue
        seen_views.add(view_id)
        if not _valid_digest(view.get("source_frame_digest")):
            blockers.append(f"view_digest_invalid:{view_id}:source_frame_digest")
        if not str(view.get("source_frame_id") or "").strip():
            blockers.append(f"view_source_frame_id_missing:{view_id}")
        source_frame_id = str(view.get("source_frame_id") or "").strip()
        frame = frames.get(source_frame_id)
        if frame is None:
            blockers.append(f"view_source_frame_not_in_registry:{view_id}")
        else:
            if not _same_digest(
                view.get("source_frame_digest"), frame.get("source_frame_digest")
            ):
                blockers.append(f"view_source_frame_digest_mismatch:{view_id}")
            if not _same_digest(
                view.get("camera_record_digest"), frame.get("camera_record_digest")
            ):
                blockers.append(f"view_frame_camera_digest_mismatch:{view_id}")
        decoded_pts = _finite_number(view.get("decoded_pts_seconds"))
        if decoded_pts is None or decoded_pts < 0.0:
            blockers.append(f"view_decoded_pts_invalid:{view_id}")
        elif frame is not None and decoded_pts != _finite_number(frame.get("decoded_pts_seconds")):
            blockers.append(f"view_decoded_pts_mismatch:{view_id}")
        width = _positive_int(view.get("width"))
        height = _positive_int(view.get("height"))
        if width is None or height is None:
            blockers.append(f"view_dimensions_invalid:{view_id}")
            continue
        if width * height > _MAX_PIXELS_PER_VIEW:
            blockers.append(f"view_pixel_count_exceeds_limit:{view_id}")
        if view.get("coverage_kind") != "full_frame":
            blockers.append(f"view_full_frame_contribution_coverage_required:{view_id}")
        camera_record = view.get("camera_record")
        if not isinstance(camera_record, Mapping):
            blockers.append(f"view_camera_record_invalid:{view_id}")
        else:
            intrinsics = camera_record.get("intrinsics")
            camera_to_world = camera_record.get("camera_to_world")
            intrinsics_values = (
                [_finite_number(item) for item in intrinsics]
                if isinstance(intrinsics, list) and len(intrinsics) == 4
                else []
            )
            transform_values = (
                [_finite_number(item) for item in camera_to_world]
                if isinstance(camera_to_world, list) and len(camera_to_world) == 16
                else []
            )
            if (
                len(intrinsics_values) != 4
                or any(item is None for item in intrinsics_values)
                or float(intrinsics_values[0]) <= 0.0
                or float(intrinsics_values[1]) <= 0.0
            ):
                blockers.append(f"view_camera_intrinsics_invalid:{view_id}")
            if len(transform_values) != 16 or any(item is None for item in transform_values):
                blockers.append(f"view_camera_to_world_invalid:{view_id}")
            if camera_record.get("coordinate_frame") != "analysis_splat_z_up_meters":
                blockers.append(f"view_camera_coordinate_frame_invalid:{view_id}")
            try:
                camera_digest = canonical_json_digest(camera_record)
            except (TypeError, ValueError):
                blockers.append(f"view_camera_record_not_canonical_json:{view_id}")
            else:
                if not _same_digest(view.get("camera_record_digest"), camera_digest):
                    blockers.append(f"view_camera_record_digest_mismatch:{view_id}")
        direction = _unit_direction(view.get("view_direction_world"))
        if direction is None:
            blockers.append(f"view_direction_invalid:{view_id}")
        else:
            directions[view_id] = direction
        pixels = view.get("pixels")
        if not isinstance(pixels, list) or len(pixels) != width * height:
            blockers.append(f"view_pixel_coverage_incomplete:{view_id}")
            continue
        mask_payload = [
            {
                "pixel_id": pixel.get("pixel_id"),
                "mask_probabilities": pixel.get("mask_probabilities"),
            }
            for pixel in pixels
            if isinstance(pixel, Mapping)
        ]
        contribution_payload = [
            {
                "pixel_id": pixel.get("pixel_id"),
                "contributions": pixel.get("contributions"),
            }
            for pixel in pixels
            if isinstance(pixel, Mapping)
        ]
        try:
            mask_digest = canonical_json_digest(mask_payload)
            contribution_digest = canonical_json_digest(contribution_payload)
        except (TypeError, ValueError):
            blockers.append(f"view_mask_or_contribution_not_canonical_json:{view_id}")
        else:
            if not _same_digest(view.get("mask_artifact_digest"), mask_digest):
                blockers.append(f"view_mask_artifact_digest_mismatch:{view_id}")
            if not _same_digest(
                view.get("contribution_artifact_digest"), contribution_digest
            ):
                blockers.append(f"view_contribution_artifact_digest_mismatch:{view_id}")
        seen_pixels: set[int] = set()
        for pixel in pixels:
            if not isinstance(pixel, Mapping):
                blockers.append(f"pixel_record_invalid:{view_id}")
                continue
            pixel_id = pixel.get("pixel_id")
            if (
                not isinstance(pixel_id, int)
                or isinstance(pixel_id, bool)
                or pixel_id < 0
                or pixel_id >= width * height
                or pixel_id in seen_pixels
            ):
                blockers.append(f"pixel_id_invalid_or_duplicate:{view_id}")
            else:
                seen_pixels.add(pixel_id)
            masks = pixel.get("mask_probabilities")
            if not isinstance(masks, Mapping):
                blockers.append(f"pixel_masks_invalid:{view_id}:{pixel_id}")
                masks = {}
            for track_id, probability in masks.items():
                value = _finite_number(probability)
                if track_id not in tracks or value is None or not 0.0 <= value <= 1.0:
                    blockers.append(f"pixel_mask_probability_invalid:{view_id}:{pixel_id}")
            contributions = pixel.get("contributions")
            if not isinstance(contributions, list):
                blockers.append(f"pixel_contributions_invalid:{view_id}:{pixel_id}")
                continue
            if len(contributions) > _MAX_CONTRIBUTIONS_PER_PIXEL:
                blockers.append(f"pixel_contribution_count_exceeds_limit:{view_id}:{pixel_id}")
            weight_sum = 0.0
            gaussian_ids: set[int] = set()
            for contribution in contributions:
                if not isinstance(contribution, Mapping):
                    blockers.append(f"pixel_contribution_row_invalid:{view_id}:{pixel_id}")
                    continue
                gaussian_id = contribution.get("gaussian_id")
                weight = _finite_number(contribution.get("weight"))
                if (
                    not isinstance(gaussian_id, int)
                    or isinstance(gaussian_id, bool)
                    or gaussian_id not in source_classes
                    or gaussian_id in gaussian_ids
                ):
                    blockers.append(f"pixel_gaussian_id_invalid_or_duplicate:{view_id}:{pixel_id}")
                    continue
                if weight is None or weight <= 0.0 or weight > 1.0:
                    blockers.append(f"pixel_contribution_weight_invalid:{view_id}:{pixel_id}")
                    continue
                gaussian_ids.add(gaussian_id)
                weight_sum += weight
            if weight_sum > 1.0 + 1e-6:
                blockers.append(f"pixel_contribution_sum_exceeds_one:{view_id}:{pixel_id}")
        if seen_pixels != set(range(width * height)):
            blockers.append(f"view_pixel_ids_not_complete:{view_id}")
    if not views:
        blockers.append("views_empty")
    return directions


def _next_experiment(reasons: Sequence[str], generated_support: int) -> str:
    if "insufficient_distinct_views" in reasons:
        return "render_or_capture_additional_overlapping_views_of_this_track"
    if "insufficient_angular_diversity" in reasons:
        return "add_oblique_views_with_a_wider_camera_baseline"
    if generated_support:
        return "recapture_the_generated_only_or_unobserved_object_region"
    return "capture_a_close_multi_view_orbit_and_rerun_qualified_contribution_rendering"


def lift_semantic_masks_to_gaussians(
    request: Mapping[str, Any],
    *,
    gaussian_mapping: Sequence[Mapping[str, Any]],
    track_registry: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Accumulate qualified foreground/background evidence for each track/Gaussian.

    Invalid bindings return ``status=blocked``.  Valid but insufficient evidence
    returns per-track abstentions and the next cheapest experiment.
    """

    blockers: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("request_schema_version_invalid")
    _validate_bindings(request, blockers)
    _validate_renderer_profile(request, blockers)
    _validate_world(request, blockers)
    source_classes = _validate_mapping(request, gaussian_mapping, blockers)
    tracks = _validate_tracks(request, track_registry, blockers)
    frames = _validate_frame_registry(request, blockers)
    qualification = _validate_qualification(request, blockers)
    directions = _validate_views(
        request,
        views,
        tracks,
        frames,
        source_classes,
        blockers,
    )
    if blockers:
        return _blocked_result(request, blockers)

    per_track: Dict[str, Dict[int, Dict[str, Any]]] = {
        track_id: defaultdict(lambda: {"foreground": 0.0, "background": 0.0, "views": {}})
        for track_id in tracks
    }
    view_foreground: Dict[str, Dict[str, float]] = {
        track_id: defaultdict(float) for track_id in tracks
    }
    for view in sorted(views, key=lambda item: str(item["view_id"])):
        view_id = str(view["view_id"])
        for pixel in sorted(view["pixels"], key=lambda item: int(item["pixel_id"])):
            masks = pixel["mask_probabilities"]
            for contribution in sorted(
                pixel["contributions"], key=lambda item: int(item["gaussian_id"])
            ):
                gaussian_id = int(contribution["gaussian_id"])
                weight = float(contribution["weight"])
                for track_id in sorted(tracks):
                    probability = float(masks.get(track_id, 0.0))
                    foreground = weight * probability
                    background = weight * (1.0 - probability)
                    row = per_track[track_id][gaussian_id]
                    row["foreground"] += foreground
                    row["background"] += background
                    view_row = row["views"].setdefault(
                        view_id,
                        {"foreground": 0.0, "background": 0.0},
                    )
                    view_row["foreground"] += foreground
                    view_row["background"] += background
                    view_foreground[track_id][view_id] += foreground

    track_results: list[Dict[str, Any]] = []
    for track_id in sorted(tracks):
        support_views = sorted(
            view_id
            for view_id, foreground in view_foreground[track_id].items()
            if foreground >= qualification["min_view_foreground_contribution"]
        )
        diversity = _angular_diversity(support_views, directions)
        selected: list[Dict[str, Any]] = []
        generated_support = 0
        gaussian_rows: list[Dict[str, Any]] = []
        for gaussian_id in sorted(per_track[track_id]):
            row = per_track[track_id][gaussian_id]
            foreground = float(row["foreground"])
            background = float(row["background"])
            total = foreground + background
            probability = foreground / total if total > 0.0 else 0.0
            gaussian_views = sorted(
                view_id
                for view_id, view_row in row["views"].items()
                if view_row["foreground"]
                >= qualification["min_gaussian_view_foreground_contribution"]
            )
            source_class = source_classes[gaussian_id]
            candidate = (
                total >= qualification["min_gaussian_total_contribution"]
                and probability >= qualification["foreground_probability_threshold"]
                and len(gaussian_views) >= qualification["min_gaussian_views"]
            )
            selected_for_semantics = candidate and source_class == "observed"
            if candidate and source_class == "generated":
                generated_support += 1
            evidence_row = {
                "gaussian_id": gaussian_id,
                "source_class": source_class,
                "foreground_contribution": round(foreground, 12),
                "background_contribution": round(background, 12),
                "foreground_probability": round(probability, 12),
                "supporting_view_ids": gaussian_views,
                "selected_for_semantic_support": selected_for_semantics,
            }
            gaussian_rows.append(evidence_row)
            if selected_for_semantics:
                selected.append(evidence_row)

        reasons: list[str] = []
        if len(support_views) < qualification["min_track_views"]:
            reasons.append("insufficient_distinct_views")
        if diversity + 1e-9 < qualification["min_angular_diversity_degrees"]:
            reasons.append("insufficient_angular_diversity")
        if not selected:
            reasons.append("no_observed_gaussians_passed_semantic_thresholds")
        status = "qualified_semantic_support_candidate" if not reasons else "abstained"
        track_result: Dict[str, Any] = {
            "track_id": track_id,
            "label": str(tracks[track_id]["label"]),
            "status": status,
            "supporting_view_ids": support_views,
            "supporting_view_count": len(support_views),
            "angular_diversity_degrees": round(diversity, 9),
            "selected_gaussian_ids": [row["gaussian_id"] for row in selected],
            "generated_candidate_gaussian_count": generated_support,
            "gaussian_evidence": gaussian_rows,
            "abstention_reasons": reasons,
            "claim_ceiling": "per_gaussian_semantic_support_candidate",
            "metric_box_ready": False,
            "physics_ready": False,
        }
        if reasons:
            track_result["next_experiment"] = _next_experiment(reasons, generated_support)
        track_results.append(track_result)

    qualified_count = sum(
        row["status"] == "qualified_semantic_support_candidate" for row in track_results
    )
    if qualified_count == len(track_results):
        status = "completed"
    elif qualified_count:
        status = "partially_completed"
    else:
        status = "abstained"
    world = request["world"]
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "bindings": dict(request["bindings"]),
        "renderer_profile": dict(request["renderer_profile"]),
        "world": dict(world),
        "tracks": track_results,
        "qualified_track_count": qualified_count,
        "abstained_track_count": len(track_results) - qualified_count,
        "blockers": [],
        "claim_ceiling": (
            "per_gaussian_semantic_support_candidate_metric_frame"
            if world["scale_verified"] is True
            else "per_gaussian_semantic_support_candidate_unverified_scale"
        ),
        "canonical_object_geometry": False,
        "metric_box_ready": False,
        "physics_ready": False,
        "generated_regions_can_upgrade_claims": False,
        "prohibited_claims": [
            "object_metric_dimensions_without_a_separate_qualified_box_fit",
            "collision_or_contact_truth",
            "task_or_physical_success",
            "safety_or_deployment_readiness",
        ],
    }
    result["result_digest"] = canonical_json_digest(result)
    return result
