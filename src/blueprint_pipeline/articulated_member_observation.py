"""Deterministic appearance observation for an articulated horizontal seam.

This selection-time helper projects a publisher object AABB into one retained
native-Spark view, detects the strongest room-wide horizontal luminance edge
inside a central object ROI, and back-projects that row onto a declared object
front plane.  It converts a visible refrigerator/drawer seam into reproducible
candidate geometry; it does not infer a joint, axis, physics, or physical truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "articulated_horizontal_seam_observation.v1"


class ArticulatedMemberObservationError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _vector(value: Any, *, length: int, error: str) -> np.ndarray:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ArticulatedMemberObservationError([error])
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ArticulatedMemberObservationError([error]) from exc
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ArticulatedMemberObservationError([error])
    return result


def _camera_basis(camera: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    position = _vector(camera.get("pos"), length=3, error="seam_camera_position_invalid")
    target = _vector(camera.get("target"), length=3, error="seam_camera_target_invalid")
    declared_up = _vector(camera.get("up"), length=3, error="seam_camera_up_invalid")
    forward = target - position
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-12:
        raise ArticulatedMemberObservationError(["seam_camera_direction_invalid"])
    forward /= forward_norm
    right = np.cross(forward, declared_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-12:
        raise ArticulatedMemberObservationError(["seam_camera_up_invalid"])
    right /= right_norm
    up = np.cross(right, forward)
    up /= float(np.linalg.norm(up))
    try:
        vertical_fov = float(camera["fov"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArticulatedMemberObservationError(["seam_camera_fov_invalid"]) from exc
    if not math.isfinite(vertical_fov) or not 0.0 < vertical_fov < 180.0:
        raise ArticulatedMemberObservationError(["seam_camera_fov_invalid"])
    return position, forward, right, up, vertical_fov


def _project(
    point: np.ndarray,
    *,
    position: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    vertical_fov_degrees: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    relative = point - position
    depth = float(np.dot(relative, forward))
    if depth <= 1e-9:
        raise ArticulatedMemberObservationError(["seam_target_behind_camera"])
    tangent = math.tan(math.radians(vertical_fov_degrees / 2.0))
    aspect = width / height
    ndc_x = float(np.dot(relative, right)) / (depth * tangent * aspect)
    ndc_y = float(np.dot(relative, up)) / (depth * tangent)
    return (
        (ndc_x + 1.0) * width / 2.0 - 0.5,
        (1.0 - ndc_y) * height / 2.0 - 0.5,
    )


def observe_horizontal_member_seam(
    *,
    image_path: str | Path,
    camera_spec: Mapping[str, Any],
    target_world_aabb_min_m: Sequence[float],
    target_world_aabb_max_m: Sequence[float],
    front_plane_axis: int,
    front_plane_value_m: float,
    minimum_peak_gradient_255: float = 20.0,
    minimum_peak_to_baseline_ratio: float = 8.0,
) -> dict[str, Any]:
    """Observe and back-project one dominant horizontal member seam."""

    image_source = Path(image_path).expanduser().resolve()
    if not image_source.is_file() or image_source.is_symlink():
        raise ArticulatedMemberObservationError(["seam_image_missing"])
    minimum = _vector(
        target_world_aabb_min_m, length=3, error="seam_target_aabb_invalid"
    )
    maximum = _vector(
        target_world_aabb_max_m, length=3, error="seam_target_aabb_invalid"
    )
    if np.any(minimum >= maximum):
        raise ArticulatedMemberObservationError(["seam_target_aabb_invalid"])
    if front_plane_axis not in (0, 1):
        raise ArticulatedMemberObservationError(["seam_front_plane_axis_invalid"])
    try:
        plane_value = float(front_plane_value_m)
        peak_threshold = float(minimum_peak_gradient_255)
        ratio_threshold = float(minimum_peak_to_baseline_ratio)
    except (TypeError, ValueError) as exc:
        raise ArticulatedMemberObservationError(["seam_threshold_invalid"]) from exc
    if not all(math.isfinite(value) for value in (plane_value, peak_threshold, ratio_threshold)) or peak_threshold <= 0.0 or ratio_threshold <= 1.0:
        raise ArticulatedMemberObservationError(["seam_threshold_invalid"])

    with Image.open(image_source) as source_image:
        rgb = source_image.convert("RGB")
        width, height = rgb.size
        pixels = np.asarray(rgb, dtype=np.float64)
    position, forward, right, up, vertical_fov = _camera_basis(camera_spec)
    corners = [
        np.asarray([x, y, z], dtype=np.float64)
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]
    projected = [
        _project(
            corner,
            position=position,
            forward=forward,
            right=right,
            up=up,
            vertical_fov_degrees=vertical_fov,
            width=width,
            height=height,
        )
        for corner in corners
    ]
    projected_min_u = max(0.0, min(point[0] for point in projected))
    projected_max_u = min(float(width - 1), max(point[0] for point in projected))
    projected_min_v = max(0.0, min(point[1] for point in projected))
    projected_max_v = min(float(height - 1), max(point[1] for point in projected))
    span_u = projected_max_u - projected_min_u
    span_v = projected_max_v - projected_min_v
    roi = {
        "min_u": int(math.floor(projected_min_u + 0.2 * span_u)),
        "max_u_exclusive": int(math.ceil(projected_max_u - 0.2 * span_u)),
        "min_v": int(math.floor(projected_min_v + 0.15 * span_v)),
        "max_v_exclusive": int(math.ceil(projected_max_v - 0.15 * span_v)),
    }
    if roi["max_u_exclusive"] - roi["min_u"] < 8 or roi["max_v_exclusive"] - roi["min_v"] < 8:
        raise ArticulatedMemberObservationError(["seam_projected_roi_too_small"])
    luminance = (
        0.2126 * pixels[:, :, 0] + 0.7152 * pixels[:, :, 1] + 0.0722 * pixels[:, :, 2]
    )
    vertical_gradient = np.abs(np.diff(luminance, axis=0))
    scores = np.median(
        vertical_gradient[
            roi["min_v"] : roi["max_v_exclusive"],
            roi["min_u"] : roi["max_u_exclusive"],
        ],
        axis=1,
    )
    peak_local = int(np.argmax(scores))
    seam_v = roi["min_v"] + peak_local
    peak = float(scores[peak_local])
    baseline_values = np.delete(
        scores,
        np.arange(max(0, peak_local - 3), min(len(scores), peak_local + 4)),
    )
    baseline = float(np.median(baseline_values)) if baseline_values.size else 0.0
    ratio = peak / max(baseline, 1e-6)
    if peak < peak_threshold or ratio < ratio_threshold:
        raise ArticulatedMemberObservationError(["seam_edge_quality_insufficient"])

    pixel_u = 0.5 * (roi["min_u"] + roi["max_u_exclusive"] - 1)
    ndc_x = 2.0 * (pixel_u + 0.5) / width - 1.0
    ndc_y = 1.0 - 2.0 * (seam_v + 0.5) / height
    tangent = math.tan(math.radians(vertical_fov / 2.0))
    ray = forward + right * ndc_x * tangent * (width / height) + up * ndc_y * tangent
    ray /= float(np.linalg.norm(ray))
    denominator = float(ray[front_plane_axis])
    if abs(denominator) <= 1e-12:
        raise ArticulatedMemberObservationError(["seam_ray_parallel_to_front_plane"])
    distance = (plane_value - float(position[front_plane_axis])) / denominator
    if distance <= 0.0:
        raise ArticulatedMemberObservationError(["seam_front_plane_behind_camera"])
    world_point = position + distance * ray

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "observed_candidate_geometry",
        "image": {
            "path": image_source.name,
            "size_bytes": image_source.stat().st_size,
            "sha256": _sha256(image_source),
            "width": width,
            "height": height,
        },
        "camera_spec": json.loads(json.dumps(camera_spec)),
        "target_world_aabb_min_m": minimum.tolist(),
        "target_world_aabb_max_m": maximum.tolist(),
        "front_plane": {"axis": front_plane_axis, "value_m": plane_value},
        "projected_target_bounds_pixels": {
            "min_u": projected_min_u,
            "max_u": projected_max_u,
            "min_v": projected_min_v,
            "max_v": projected_max_v,
        },
        "edge_scan": {
            "roi_pixels": roi,
            "method": "median_absolute_vertical_rec709_luminance_gradient",
            "excluded_peak_half_width_rows": 3,
            "peak_gradient_255": peak,
            "baseline_gradient_255": baseline,
            "peak_to_baseline_ratio": ratio,
            "minimum_peak_gradient_255": peak_threshold,
            "minimum_peak_to_baseline_ratio": ratio_threshold,
        },
        "seam_pixel": {"u": pixel_u, "v": seam_v},
        "seam_world_point_m": [float(value) for value in world_point],
        "claim_boundary": {
            "appearance_edge_is_candidate_member_partition": True,
            "joint_topology_or_axis_proven": False,
            "metric_surface_truth_proven": False,
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    return result


__all__ = [
    "ArticulatedMemberObservationError",
    "SCHEMA_VERSION",
    "observe_horizontal_member_seam",
]
