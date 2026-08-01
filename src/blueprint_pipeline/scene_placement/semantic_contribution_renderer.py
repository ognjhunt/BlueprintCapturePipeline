"""Render source-frame mask evidence onto exact standard-3DGS Gaussian IDs.

This is a bounded, deterministic NumPy reference renderer for the semantic
lifting lane.  It projects anisotropic 3D Gaussians through calibrated OpenCV
pinhole cameras, sorts every pixel's contributors front-to-back, and emits the
actual ``transmittance * alpha`` weights consumed by
``semantic_gaussian_lifting``.  It is intended for hermetic fixtures, small
scenes, and GPU-renderer conformance checks.  Large customer scenes must use a
separately qualified accelerated adapter with the same output contract.

The result is semantic support evidence only.  Rendering a Gaussian does not
make it observed geometry, collision truth, physics truth, or physical success.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ..gaussian_splat_decode import SplatData
from .semantic_gaussian_lifting import (
    CONTRIBUTION_SEMANTICS,
    REQUEST_SCHEMA_VERSION as LIFTING_REQUEST_SCHEMA_VERSION,
    canonical_json_digest,
)


REQUEST_SCHEMA_VERSION = "semantic_contribution_render_request.v1"
RESULT_SCHEMA_VERSION = "semantic_contribution_render_result.v1"
METHOD_ID = "blueprint.numpy_standard_3dgs_contribution_renderer"
METHOD_VERSION = "1.0.0"
PROJECTION_CONVENTION = "opencv_pinhole_camera_z_forward"
MASK_ENCODING = "sparse_probability_rle.v1"

_MAX_VIEWS = 64
_MAX_PIXELS_PER_VIEW = 1_048_576
_MAX_GAUSSIANS = 250_000
_MAX_CONTRIBUTIONS_PER_PIXEL = 4096


def renderer_runtime_digest() -> str:
    """Return the pinned algorithm-profile digest expected by requests."""

    return canonical_json_digest(
        {
            "method_id": METHOD_ID,
            "method_version": METHOD_VERSION,
            "projection_convention": PROJECTION_CONVENTION,
            "quaternion_convention": "wxyz",
            "pixel_center_convention": "integer_plus_half",
            "covariance_projection": "jacobian_first_order",
            "contribution_semantics": CONTRIBUTION_SEMANTICS,
        }
    )


def _valid_digest(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _same_digest(left: Any, right: Any) -> bool:
    return str(left or "").strip().lower() == str(right or "").strip().lower()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _blocked(request: Mapping[str, Any], blockers: Sequence[str]) -> Dict[str, Any]:
    bindings = request.get("bindings")
    result: Dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "bindings": dict(bindings) if isinstance(bindings, Mapping) else {},
        "views": [],
        "blockers": sorted(set(blockers)),
        "claim_ceiling": "none_invalid_or_unbound_contribution_render",
        "reference_contribution_render_completed": False,
        "production_large_scene_ready": False,
        "metric_box_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "physical_task_success_established": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    result["result_digest"] = canonical_json_digest(result)
    return result


def blocked_semantic_contribution_render(
    request: Mapping[str, Any], blockers: Sequence[str]
) -> Dict[str, Any]:
    """Build a deterministic terminal result for file/admission failures."""

    return _blocked(request, blockers)


def _validate_request(request: Mapping[str, Any], blockers: list[str]) -> Dict[str, Any]:
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("request_schema_version_mismatch")
    bindings = request.get("bindings")
    if not isinstance(bindings, Mapping):
        blockers.append("bindings_missing")
        bindings = {}
    for field in (
        "capture_digest",
        "retained_video_digest",
        "reconstruction_digest",
        "analysis_splat_digest",
        "camera_solution_digest",
        "gaussian_mapping_digest",
        "frame_registry_digest",
        "source_track_result_digest",
        "track_registry_digest",
        "frame_masks_digest",
        "camera_records_digest",
    ):
        if not _valid_digest(bindings.get(field)):
            blockers.append(f"binding_digest_invalid:{field}")

    profile = request.get("renderer_profile")
    if not isinstance(profile, Mapping):
        blockers.append("renderer_profile_missing")
        return {}
    expected = {
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
        "runtime_digest": renderer_runtime_digest(),
        "contribution_semantics": CONTRIBUTION_SEMANTICS,
        "projection_convention": PROJECTION_CONVENTION,
        "exact_gaussian_ids": True,
        "deterministic": True,
    }
    for field, value in expected.items():
        if profile.get(field) != value:
            blockers.append(f"renderer_profile_mismatch:{field}")
    for field in (
        "minimum_alpha",
        "minimum_emitted_weight",
        "sigma_extent",
        "covariance_regularization_pixels_squared",
        "near_plane_meters",
    ):
        value = _finite(profile.get(field))
        if value is None or value <= 0.0:
            blockers.append(f"renderer_profile_positive_number_required:{field}")
    bounded_values = {
        "minimum_alpha": (1.0e-12, 1.0e-3),
        "minimum_emitted_weight": (1.0e-15, 1.0e-6),
        "sigma_extent": (3.0, 8.0),
        "covariance_regularization_pixels_squared": (1.0e-6, 4.0),
        "near_plane_meters": (1.0e-6, 0.25),
    }
    for field, (minimum, maximum) in bounded_values.items():
        value = _finite(profile.get(field))
        if value is not None and not minimum <= value <= maximum:
            blockers.append(f"renderer_profile_range_invalid:{field}")
    max_pairs = _positive_int(profile.get("max_projected_pixel_gaussian_pairs"))
    if max_pairs is None or max_pairs > 100_000_000:
        blockers.append("renderer_profile_max_pairs_invalid")
    max_contributions = _positive_int(profile.get("max_contributions_per_pixel"))
    if max_contributions is None or max_contributions > _MAX_CONTRIBUTIONS_PER_PIXEL:
        blockers.append("renderer_profile_max_contributions_invalid")

    world = request.get("world")
    if not isinstance(world, Mapping):
        blockers.append("world_contract_missing")
    else:
        if str(world.get("up_axis") or "").upper() != "Z":
            blockers.append("world_up_axis_must_be_z")
        if str(world.get("units") or "").lower() != "meters":
            blockers.append("world_units_must_be_meters")
        if not isinstance(world.get("scale_verified"), bool):
            blockers.append("world_scale_verified_must_be_boolean")
    if not isinstance(request.get("qualification"), Mapping):
        blockers.append("lifting_qualification_missing")
    return dict(profile)


def _validate_mapping(
    request: Mapping[str, Any],
    gaussian_mapping: Sequence[Mapping[str, Any]],
    splat: SplatData,
    blockers: list[str],
) -> tuple[np.ndarray, list[str]]:
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    try:
        digest = canonical_json_digest(gaussian_mapping)
    except (TypeError, ValueError):
        blockers.append("gaussian_mapping_not_canonical_json")
        return np.asarray([], dtype=np.int64), []
    if not _same_digest(bindings.get("gaussian_mapping_digest"), digest):
        blockers.append("gaussian_mapping_digest_mismatch")
    if not gaussian_mapping or len(gaussian_mapping) > _MAX_GAUSSIANS:
        blockers.append("gaussian_mapping_empty_or_too_large")
    if len(gaussian_mapping) != splat.count:
        blockers.append("gaussian_mapping_splat_count_mismatch")
    source_indices: list[int] = []
    source_classes: list[str] = []
    seen_sources: set[int] = set()
    for expected_id, row in enumerate(gaussian_mapping):
        if not isinstance(row, Mapping) or row.get("gaussian_id") != expected_id:
            blockers.append("gaussian_mapping_ids_not_dense_and_ordered")
            continue
        source_index = row.get("source_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or source_index >= splat.count
            or source_index in seen_sources
        ):
            blockers.append("gaussian_mapping_source_index_invalid_or_duplicate")
            continue
        source_class = str(row.get("source_class") or "").strip().lower()
        if source_class not in {"observed", "generated", "unknown"}:
            blockers.append("gaussian_mapping_source_class_invalid")
            continue
        seen_sources.add(source_index)
        source_indices.append(source_index)
        source_classes.append(source_class)
    if len(source_indices) != len(gaussian_mapping):
        return np.asarray([], dtype=np.int64), []
    return np.asarray(source_indices, dtype=np.int64), source_classes


def _validate_source_tracks(
    request: Mapping[str, Any], source_tracks: Mapping[str, Any], blockers: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    if source_tracks.get("schema_version") != "semantic_source_track_import_result.v1":
        blockers.append("source_track_result_schema_mismatch")
    if source_tracks.get("status") != "completed":
        blockers.append("source_track_result_not_completed")
    supplied_digest = source_tracks.get("result_digest")
    if not _valid_digest(supplied_digest) or supplied_digest != canonical_json_digest(
        {key: value for key, value in source_tracks.items() if key != "result_digest"}
    ):
        blockers.append("source_track_result_digest_invalid")
    if not _same_digest(bindings.get("source_track_result_digest"), supplied_digest):
        blockers.append("source_track_result_binding_mismatch")
    source_bindings = (
        source_tracks.get("bindings")
        if isinstance(source_tracks.get("bindings"), Mapping)
        else {}
    )
    for field in (
        "capture_digest",
        "retained_video_digest",
        "camera_solution_digest",
        "frame_registry_digest",
        "track_registry_digest",
        "frame_masks_digest",
    ):
        if not _same_digest(source_bindings.get(field), bindings.get(field)):
            blockers.append(f"source_track_binding_mismatch:{field}")
    tracks = source_tracks.get("track_registry")
    masks = source_tracks.get("frame_masks")
    if not isinstance(tracks, list) or not tracks:
        blockers.append("source_track_registry_missing_or_empty")
        tracks = []
    if not isinstance(masks, list) or not masks or len(masks) > _MAX_VIEWS:
        blockers.append("source_frame_masks_missing_empty_or_too_large")
        masks = []
    try:
        if canonical_json_digest(tracks) != source_bindings.get("track_registry_digest"):
            blockers.append("source_track_registry_digest_mismatch")
        if canonical_json_digest(masks) != source_bindings.get("frame_masks_digest"):
            blockers.append("source_frame_masks_digest_mismatch")
    except (TypeError, ValueError):
        blockers.append("source_track_payload_not_canonical_json")
    return [dict(row) for row in tracks if isinstance(row, Mapping)], [
        dict(row) for row in masks if isinstance(row, Mapping)
    ]


def _validate_frames_and_cameras(
    request: Mapping[str, Any],
    frame_masks: Sequence[Mapping[str, Any]],
    camera_records: Sequence[Mapping[str, Any]],
    blockers: list[str],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    bindings = request.get("bindings") if isinstance(request.get("bindings"), Mapping) else {}
    frames = request.get("frame_registry")
    if not isinstance(frames, list) or not frames:
        blockers.append("frame_registry_missing_or_empty")
        frames = []
    try:
        if canonical_json_digest(frames) != bindings.get("frame_registry_digest"):
            blockers.append("frame_registry_digest_mismatch")
        if canonical_json_digest(camera_records) != bindings.get("camera_records_digest"):
            blockers.append("camera_records_digest_mismatch")
    except (TypeError, ValueError):
        blockers.append("frame_or_camera_registry_not_canonical_json")
    frame_by_id: Dict[str, Dict[str, Any]] = {}
    for row in frames:
        if not isinstance(row, Mapping):
            blockers.append("frame_registry_row_invalid")
            continue
        frame_id = str(row.get("source_frame_id") or "").strip()
        if not frame_id or frame_id in frame_by_id:
            blockers.append("frame_registry_id_invalid_or_duplicate")
            continue
        if row.get("encoder_retained") is not True:
            blockers.append(f"frame_not_encoder_retained:{frame_id}")
        for field in (
            "source_frame_digest",
            "retained_video_digest",
            "sync_map_row_digest",
            "camera_record_digest",
        ):
            if not _valid_digest(row.get(field)):
                blockers.append(f"frame_registry_digest_invalid:{frame_id}:{field}")
        if not _same_digest(row.get("retained_video_digest"), bindings.get("retained_video_digest")):
            blockers.append(f"frame_registry_retained_video_mismatch:{frame_id}")
        pts = _finite(row.get("decoded_pts_seconds"))
        if pts is None or pts < 0.0:
            blockers.append(f"frame_registry_pts_invalid:{frame_id}")
        frame_by_id[frame_id] = dict(row)

    camera_by_id: Dict[str, Dict[str, Any]] = {}
    for row in camera_records:
        if not isinstance(row, Mapping):
            blockers.append("camera_registry_row_invalid")
            continue
        frame_id = str(row.get("source_frame_id") or "").strip()
        camera = row.get("camera_record")
        if not frame_id or frame_id in camera_by_id or not isinstance(camera, Mapping):
            blockers.append("camera_registry_identity_or_record_invalid")
            continue
        if camera.get("coordinate_frame") != "analysis_splat_z_up_meters":
            blockers.append(f"camera_coordinate_frame_invalid:{frame_id}")
        if camera.get("projection_convention") != PROJECTION_CONVENTION:
            blockers.append(f"camera_projection_convention_invalid:{frame_id}")
        if camera.get("distortion_status") != "rectified_none":
            blockers.append(f"camera_distortion_must_be_rectified:{frame_id}")
        try:
            camera_digest = canonical_json_digest(camera)
        except (TypeError, ValueError):
            blockers.append(f"camera_record_not_canonical_json:{frame_id}")
            continue
        if camera_digest != row.get("camera_record_digest"):
            blockers.append(f"camera_record_digest_mismatch:{frame_id}")
        frame = frame_by_id.get(frame_id)
        if frame is None:
            blockers.append(f"camera_frame_not_registered:{frame_id}")
        elif not _same_digest(camera_digest, frame.get("camera_record_digest")):
            blockers.append(f"camera_frame_digest_mismatch:{frame_id}")
        camera_by_id[frame_id] = dict(row)

    mask_ids: set[str] = set()
    for row in frame_masks:
        frame_id = str(row.get("source_frame_id") or "").strip()
        if not frame_id or frame_id in mask_ids:
            blockers.append("frame_mask_id_invalid_or_duplicate")
            continue
        mask_ids.add(frame_id)
        frame = frame_by_id.get(frame_id)
        if frame is None:
            blockers.append(f"frame_mask_frame_not_registered:{frame_id}")
            continue
        for field in ("source_frame_digest", "camera_record_digest"):
            if not _same_digest(row.get(field), frame.get(field)):
                blockers.append(f"frame_mask_binding_mismatch:{frame_id}:{field}")
        if _finite(row.get("decoded_pts_seconds")) != _finite(
            frame.get("decoded_pts_seconds")
        ):
            blockers.append(f"frame_mask_binding_mismatch:{frame_id}:decoded_pts_seconds")
    if mask_ids != set(camera_by_id):
        blockers.append("camera_and_mask_frame_sets_mismatch")
    return frame_by_id, camera_by_id


def _quaternion_rotation_wxyz(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("gaussian_quaternion_invalid")
    w, x, y, z = (float(value) / norm for value in quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _camera_arrays(camera: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = camera.get("intrinsics")
    transform = camera.get("camera_to_world")
    if not isinstance(intrinsics, list) or len(intrinsics) != 4:
        raise ValueError("camera_intrinsics_invalid")
    if not isinstance(transform, list) or len(transform) != 16:
        raise ValueError("camera_to_world_invalid")
    intrinsics_array = np.asarray(intrinsics, dtype=np.float64)
    c2w = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(intrinsics_array)) or not np.all(np.isfinite(c2w)):
        raise ValueError("camera_values_nonfinite")
    if intrinsics_array[0] <= 0.0 or intrinsics_array[1] <= 0.0:
        raise ValueError("camera_focal_length_invalid")
    if not np.allclose(c2w[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
        raise ValueError("camera_to_world_not_affine")
    rotation = c2w[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0):
        raise ValueError("camera_rotation_not_orthonormal")
    return intrinsics_array, c2w


def _expand_masks(
    frame_mask: Mapping[str, Any],
    track_ids: set[str],
    blockers: list[str],
) -> tuple[int, int, Dict[int, Dict[str, float]]]:
    width = _positive_int(frame_mask.get("width"))
    height = _positive_int(frame_mask.get("height"))
    frame_id = str(frame_mask.get("source_frame_id") or "")
    if width is None or height is None or width * height > _MAX_PIXELS_PER_VIEW:
        blockers.append(f"frame_mask_dimensions_invalid_or_too_large:{frame_id}")
        return 0, 0, {}
    if frame_mask.get("mask_encoding") != MASK_ENCODING:
        blockers.append(f"frame_mask_encoding_invalid:{frame_id}")
    sparse: Dict[int, Dict[str, float]] = defaultdict(dict)
    raw_tracks = frame_mask.get("track_masks")
    if not isinstance(raw_tracks, list):
        blockers.append(f"frame_track_masks_invalid:{frame_id}")
        return width, height, {}
    for row in raw_tracks:
        if not isinstance(row, Mapping):
            blockers.append(f"frame_track_mask_row_invalid:{frame_id}")
            continue
        track_id = str(row.get("track_id") or "")
        if track_id not in track_ids:
            blockers.append(f"frame_track_id_not_registered:{frame_id}:{track_id}")
            continue
        runs = row.get("runs")
        if not isinstance(runs, list):
            blockers.append(f"frame_track_runs_invalid:{frame_id}:{track_id}")
            continue
        previous_end = 0
        for run in runs:
            if not isinstance(run, Mapping):
                blockers.append(f"frame_mask_run_invalid:{frame_id}:{track_id}")
                continue
            start = run.get("start")
            length = run.get("length")
            probability = _finite(run.get("probability"))
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or start < previous_end
                or isinstance(length, bool)
                or not isinstance(length, int)
                or length <= 0
                or start + length > width * height
                or probability is None
                or not 0.0 < probability <= 1.0
            ):
                blockers.append(f"frame_mask_run_bounds_invalid:{frame_id}:{track_id}")
                continue
            for pixel_id in range(start, start + length):
                sparse[pixel_id][track_id] = probability
            previous_end = start + length
    return width, height, dict(sparse)


def _project_view(
    *,
    splat: SplatData,
    source_indices: np.ndarray,
    camera: Mapping[str, Any],
    width: int,
    height: int,
    profile: Mapping[str, Any],
) -> tuple[list[list[dict[str, Any]]], int]:
    intrinsics, c2w = _camera_arrays(camera)
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    world_to_camera_rotation = c2w[:3, :3].T
    camera_origin = c2w[:3, 3]
    xyz = np.asarray(splat.xyz[source_indices], dtype=np.float64)
    camera_xyz = (xyz - camera_origin) @ world_to_camera_rotation.T
    near = float(profile["near_plane_meters"])
    sigma_extent = float(profile["sigma_extent"])
    regularization = float(profile["covariance_regularization_pixels_squared"])
    minimum_alpha = float(profile["minimum_alpha"])
    max_pairs = int(profile["max_projected_pixel_gaussian_pairs"])
    per_pixel: list[list[tuple[float, int, float]]] = [
        [] for _ in range(width * height)
    ]
    pair_count = 0
    for gaussian_id in range(len(source_indices)):
        x, y, z = (float(value) for value in camera_xyz[gaussian_id])
        if not math.isfinite(z) or z <= near:
            continue
        mean_u = fx * x / z + cx
        mean_v = fy * y / z + cy
        if not math.isfinite(mean_u) or not math.isfinite(mean_v):
            continue
        rotation = _quaternion_rotation_wxyz(
            np.asarray(splat.quats[source_indices[gaussian_id]], dtype=np.float64)
        )
        scales = np.exp(
            np.clip(
                np.asarray(splat.scales[source_indices[gaussian_id]], dtype=np.float64),
                -20.0,
                20.0,
            )
        )
        covariance_world = rotation @ np.diag(scales * scales) @ rotation.T
        covariance_camera = (
            world_to_camera_rotation @ covariance_world @ world_to_camera_rotation.T
        )
        jacobian = np.asarray(
            [
                [fx / z, 0.0, -fx * x / (z * z)],
                [0.0, fy / z, -fy * y / (z * z)],
            ],
            dtype=np.float64,
        )
        covariance_2d = jacobian @ covariance_camera @ jacobian.T
        covariance_2d += np.eye(2, dtype=np.float64) * regularization
        if not np.all(np.isfinite(covariance_2d)):
            raise ValueError("projected_gaussian_covariance_nonfinite")
        eigenvalues = np.linalg.eigvalsh(covariance_2d)
        if eigenvalues[0] <= 0.0:
            raise ValueError("projected_gaussian_covariance_not_positive")
        radius = sigma_extent * math.sqrt(float(eigenvalues[-1]))
        min_x = max(0, int(math.floor(mean_u - radius)))
        max_x = min(width - 1, int(math.ceil(mean_u + radius)))
        min_y = max(0, int(math.floor(mean_v - radius)))
        max_y = min(height - 1, int(math.ceil(mean_v + radius)))
        if min_x > max_x or min_y > max_y:
            continue
        inverse = np.linalg.inv(covariance_2d)
        opacity = float(splat.opacity[source_indices[gaussian_id]])
        base_alpha = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, opacity))))
        for pixel_y in range(min_y, max_y + 1):
            for pixel_x in range(min_x, max_x + 1):
                delta = np.asarray(
                    [pixel_x + 0.5 - mean_u, pixel_y + 0.5 - mean_v],
                    dtype=np.float64,
                )
                exponent = -0.5 * float(delta @ inverse @ delta)
                alpha = min(0.999, base_alpha * math.exp(exponent))
                if alpha < minimum_alpha:
                    continue
                per_pixel[pixel_y * width + pixel_x].append((z, gaussian_id, alpha))
                pair_count += 1
                if pair_count > max_pairs:
                    raise OverflowError("projected_pixel_gaussian_pair_limit_exceeded")

    emitted: list[list[dict[str, Any]]] = []
    minimum_weight = float(profile["minimum_emitted_weight"])
    maximum_rows = int(profile["max_contributions_per_pixel"])
    for candidates in per_pixel:
        transmittance = 1.0
        rows: list[dict[str, Any]] = []
        for _depth, gaussian_id, alpha in sorted(candidates, key=lambda row: (row[0], row[1])):
            weight = transmittance * alpha
            transmittance *= 1.0 - alpha
            if weight >= minimum_weight:
                rows.append({"gaussian_id": gaussian_id, "weight": round(weight, 15)})
            if transmittance <= minimum_weight:
                break
        if len(rows) > maximum_rows:
            raise OverflowError("pixel_contribution_count_exceeds_profile_limit")
        emitted.append(rows)
    return emitted, pair_count


def render_semantic_contributions(
    request: Mapping[str, Any],
    *,
    splat: SplatData,
    gaussian_mapping: Sequence[Mapping[str, Any]],
    source_tracks: Mapping[str, Any],
    camera_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Render exact per-pixel contributions and a ready-to-run lifting request."""

    blockers: list[str] = []
    profile = _validate_request(request, blockers)
    source_indices, _source_classes = _validate_mapping(
        request, gaussian_mapping, splat, blockers
    )
    track_registry, frame_masks = _validate_source_tracks(request, source_tracks, blockers)
    frame_by_id, camera_by_id = _validate_frames_and_cameras(
        request, frame_masks, camera_records, blockers
    )
    track_ids = {str(row.get("track_id") or "") for row in track_registry}
    if blockers:
        return _blocked(request, blockers)

    views: list[dict[str, Any]] = []
    total_pairs = 0
    for frame_mask in sorted(frame_masks, key=lambda row: str(row.get("source_frame_id"))):
        frame_id = str(frame_mask["source_frame_id"])
        width, height, sparse_masks = _expand_masks(frame_mask, track_ids, blockers)
        camera_row = camera_by_id[frame_id]
        camera = camera_row["camera_record"]
        try:
            contributions, pair_count = _project_view(
                splat=splat,
                source_indices=source_indices,
                camera=camera,
                width=width,
                height=height,
                profile=profile,
            )
        except (OverflowError, ValueError, np.linalg.LinAlgError) as exc:
            blockers.append(f"contribution_render_failed:{frame_id}:{exc}")
            continue
        total_pairs += pair_count
        pixels = [
            {
                "pixel_id": pixel_id,
                "mask_probabilities": sparse_masks.get(pixel_id, {}),
                "contributions": contributions[pixel_id],
            }
            for pixel_id in range(width * height)
        ]
        mask_payload = [
            {
                "pixel_id": row["pixel_id"],
                "mask_probabilities": row["mask_probabilities"],
            }
            for row in pixels
        ]
        contribution_payload = [
            {"pixel_id": row["pixel_id"], "contributions": row["contributions"]}
            for row in pixels
        ]
        frame = frame_by_id[frame_id]
        c2w = np.asarray(camera["camera_to_world"], dtype=np.float64).reshape(4, 4)
        direction = c2w[:3, 2]
        direction /= np.linalg.norm(direction)
        views.append(
            {
                "view_id": frame_id,
                "source_frame_id": frame_id,
                "source_frame_digest": frame["source_frame_digest"],
                "decoded_pts_seconds": frame["decoded_pts_seconds"],
                "camera_record": dict(camera),
                "camera_record_digest": camera_row["camera_record_digest"],
                "mask_artifact_digest": canonical_json_digest(mask_payload),
                "contribution_artifact_digest": canonical_json_digest(
                    contribution_payload
                ),
                "width": width,
                "height": height,
                "coverage_kind": "full_frame",
                "view_direction_world": [round(float(value), 15) for value in direction],
                "pixels": pixels,
            }
        )
    if blockers:
        return _blocked(request, blockers)

    bindings = dict(request["bindings"])
    bindings["views_digest"] = canonical_json_digest(views)
    lifting_request = {
        "schema_version": LIFTING_REQUEST_SCHEMA_VERSION,
        "bindings": bindings,
        "frame_registry": list(request["frame_registry"]),
        "gaussian_count": len(gaussian_mapping),
        "renderer_profile": dict(profile),
        "world": dict(request["world"]),
        "qualification": dict(request["qualification"]),
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed",
        "bindings": bindings,
        "renderer_profile": dict(profile),
        "world": dict(request["world"]),
        "gaussian_mapping": [dict(row) for row in gaussian_mapping],
        "track_registry": track_registry,
        "views": views,
        "lifting_request": lifting_request,
        "render_statistics": {
            "view_count": len(views),
            "gaussian_count": len(gaussian_mapping),
            "pixel_count": sum(int(row["width"]) * int(row["height"]) for row in views),
            "projected_pixel_gaussian_pair_count": total_pairs,
        },
        "blockers": [],
        "claim_ceiling": "thresholded_reference_renderer_contribution_rows_for_semantic_lifting",
        "reference_contribution_render_completed": True,
        "production_large_scene_ready": False,
        "canonical_object_geometry": False,
        "metric_box_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "physical_task_success_established": False,
        "generated_regions_can_upgrade_claims": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
        "prohibited_claims": [
            "source_frame_semantics_are_directly_observed_facts",
            "gaussian_appearance_is_collision_or_physics_truth",
            "task_or_physical_success",
            "safety_or_deployment_readiness",
            "comparative_policy_ranking_support",
        ],
    }
    result["result_digest"] = canonical_json_digest(result)
    return result


__all__ = [
    "METHOD_ID",
    "METHOD_VERSION",
    "PROJECTION_CONVENTION",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "blocked_semantic_contribution_render",
    "render_semantic_contributions",
    "renderer_runtime_digest",
]
