"""Shared fail-closed validation for calibrated camera geometry.

This module intentionally contains no runtime-specific defaults.  Callers may
choose whether a particular artifact must be projection-ready, but all camera
and SE(3) math is checked here so training, geometry, and task-grounding lanes
cannot disagree about what a valid calibration is.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


_INTRINSIC_KEYS = {
    "fx": ("fx", "focal_x"),
    "fy": ("fy", "focal_y"),
    "cx": ("cx", "principal_x"),
    "cy": ("cy", "principal_y"),
    "width": ("width", "image_width", "w"),
    "height": ("height", "image_height", "h"),
}
_CAMERA_FROM_REFERENCE_KEYS = (
    "camera_from_world",
    "T_camera_world",
    "camera_from_reference",
    "T_camera_reference",
)
_REFERENCE_FROM_CAMERA_KEYS = (
    "world_from_camera",
    "T_world_camera",
    "reference_from_camera",
    "T_reference_camera",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_number(payload: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in payload:
            number = _finite_number(payload.get(key))
            if number is not None:
                return number
    return None


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def matrix4(value: Any) -> np.ndarray | None:
    """Decode a finite 4x4 matrix without accepting ragged/defaulted rows."""

    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.ndim == 1 and matrix.size == 16:
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    return matrix


def validate_se3_matrix(
    value: Any,
    *,
    field: str,
    orthonormal_tolerance: float = 1e-5,
    determinant_tolerance: float = 1e-5,
    last_row_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Validate that ``value`` is a finite, right-handed rigid transform."""

    blockers: list[str] = []
    matrix = matrix4(value)
    determinant: float | None = None
    orthonormal_error: float | None = None
    last_row_error: float | None = None
    if matrix is None:
        blockers.append(f"{field}_missing_misshaped_or_nonfinite")
    else:
        rotation = matrix[:3, :3]
        determinant = float(np.linalg.det(rotation))
        orthonormal_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        last_row_error = float(
            np.max(np.abs(matrix[3, :] - np.asarray([0.0, 0.0, 0.0, 1.0])))
        )
        if last_row_error > last_row_tolerance:
            blockers.append(f"{field}_invalid_homogeneous_last_row")
        if orthonormal_error > orthonormal_tolerance:
            blockers.append(f"{field}_rotation_not_orthonormal")
        if abs(determinant - 1.0) > determinant_tolerance:
            blockers.append(f"{field}_rotation_not_right_handed")
    return {
        "field": field,
        "valid": not blockers,
        "matrix": matrix.tolist() if matrix is not None else None,
        "determinant": determinant,
        "orthonormal_error": orthonormal_error,
        "last_row_error": last_row_error,
        "blockers": blockers,
    }


def validate_camera_intrinsics(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate pinhole intrinsics, image shape, principal point, and FOV."""

    nested = _mapping(payload.get("intrinsics")) or _mapping(payload.get("camera_intrinsics"))
    intrinsics = nested or dict(payload)
    values = {
        name: _first_number(intrinsics, aliases)
        for name, aliases in _INTRINSIC_KEYS.items()
    }
    blockers: list[str] = []
    for name in ("fx", "fy", "width", "height", "cx", "cy"):
        if values[name] is None:
            blockers.append(f"camera_intrinsics_missing_{name}")
    for name in ("fx", "fy", "width", "height"):
        value = values[name]
        if value is not None and value <= 0:
            blockers.append(f"camera_intrinsics_nonpositive_{name}")
    width = values["width"]
    height = values["height"]
    cx = values["cx"]
    cy = values["cy"]
    fx = values["fx"]
    fy = values["fy"]
    if width is not None and (not float(width).is_integer() or width > 100_000):
        blockers.append("camera_intrinsics_invalid_width")
    if height is not None and (not float(height).is_integer() or height > 100_000):
        blockers.append("camera_intrinsics_invalid_height")
    if width is not None and cx is not None and not (0.0 <= cx < width):
        blockers.append("camera_principal_point_outside_image_x")
    if height is not None and cy is not None and not (0.0 <= cy < height):
        blockers.append("camera_principal_point_outside_image_y")
    horizontal_fov_deg = (
        math.degrees(2.0 * math.atan(width / (2.0 * fx)))
        if width is not None and fx is not None and width > 0 and fx > 0
        else None
    )
    vertical_fov_deg = (
        math.degrees(2.0 * math.atan(height / (2.0 * fy)))
        if height is not None and fy is not None and height > 0 and fy > 0
        else None
    )
    if horizontal_fov_deg is not None and not (10.0 <= horizontal_fov_deg <= 170.0):
        blockers.append("camera_horizontal_fov_implausible")
    if vertical_fov_deg is not None and not (10.0 <= vertical_fov_deg <= 170.0):
        blockers.append("camera_vertical_fov_implausible")
    normalized = None
    if not blockers:
        normalized = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": int(width),
            "height": int(height),
        }
    return {
        "valid": not blockers,
        "normalized": normalized,
        "horizontal_fov_deg": horizontal_fov_deg,
        "vertical_fov_deg": vertical_fov_deg,
        "blockers": blockers,
    }


def _matrix_candidate(payload: Mapping[str, Any], keys: Sequence[str]) -> tuple[Any, str]:
    extrinsics = _mapping(payload.get("extrinsics"))
    for source in (payload, extrinsics):
        for key in keys:
            if key in source:
                return source.get(key), key
    matrix_value = extrinsics.get("matrix")
    direction = _first_text(extrinsics.get("matrix_direction"), payload.get("matrix_direction"))
    if matrix_value is not None and direction in keys:
        return matrix_value, direction
    return None, ""


def validate_camera_calibration(
    payload: Mapping[str, Any],
    *,
    require_extrinsics: bool,
    require_frame_metadata: bool,
    require_translation_units: bool,
    require_reprojection_error: bool,
    expected_reference_frame: str | None = None,
    expected_camera_frame: str | None = None,
    max_reprojection_error_px: float = 5.0,
) -> dict[str, Any]:
    """Return a normalized calibration and explicit projection-ready blockers."""

    intrinsics_result = validate_camera_intrinsics(payload)
    blockers = list(intrinsics_result["blockers"])
    warnings: list[str] = []
    camera_raw, camera_key = _matrix_candidate(payload, _CAMERA_FROM_REFERENCE_KEYS)
    reference_raw, reference_key = _matrix_candidate(payload, _REFERENCE_FROM_CAMERA_KEYS)
    camera_result = (
        validate_se3_matrix(camera_raw, field="camera_from_reference")
        if camera_key
        else None
    )
    reference_result = (
        validate_se3_matrix(reference_raw, field="reference_from_camera")
        if reference_key
        else None
    )
    if camera_result is not None:
        blockers.extend(camera_result["blockers"])
    if reference_result is not None:
        blockers.extend(reference_result["blockers"])
    if require_extrinsics and camera_result is None and reference_result is None:
        blockers.append("camera_extrinsics_missing")

    camera_matrix = matrix4(camera_result.get("matrix")) if camera_result and camera_result["valid"] else None
    reference_matrix = matrix4(reference_result.get("matrix")) if reference_result and reference_result["valid"] else None
    inverse_consistency_error: float | None = None
    if camera_matrix is not None and reference_matrix is not None:
        inverse_consistency_error = float(
            np.max(np.abs(camera_matrix @ reference_matrix - np.eye(4)))
        )
        if inverse_consistency_error > 1e-5:
            blockers.append("camera_extrinsics_inverse_mismatch")
    elif camera_matrix is not None:
        reference_matrix = np.linalg.inv(camera_matrix)
    elif reference_matrix is not None:
        camera_matrix = np.linalg.inv(reference_matrix)

    extrinsics = _mapping(payload.get("extrinsics"))
    reference_frame = _first_text(
        extrinsics.get("reference_frame"),
        extrinsics.get("parent_frame"),
        payload.get("reference_frame"),
        payload.get("parent_frame"),
    )
    camera_frame = _first_text(
        extrinsics.get("child_frame"),
        extrinsics.get("camera_frame"),
        payload.get("child_frame"),
        payload.get("camera_frame"),
    )
    translation_unit = _first_text(
        extrinsics.get("translation_unit"),
        extrinsics.get("units"),
        payload.get("translation_unit"),
        payload.get("extrinsics_unit"),
    ).lower()
    if require_frame_metadata and not reference_frame:
        blockers.append("camera_reference_frame_missing")
    if require_frame_metadata and not camera_frame:
        blockers.append("camera_child_frame_missing")
    if reference_frame and camera_frame and reference_frame == camera_frame:
        blockers.append("camera_extrinsics_frame_cycle")
    if expected_reference_frame and reference_frame != expected_reference_frame:
        blockers.append("camera_reference_frame_mismatch")
    if expected_camera_frame and camera_frame != expected_camera_frame:
        blockers.append("camera_child_frame_mismatch")
    if require_translation_units and translation_unit not in {"m", "meter", "meters", "metre", "metres"}:
        blockers.append("camera_extrinsics_translation_unit_missing_or_not_meters")

    quality = _mapping(payload.get("quality"))
    reprojection_error = None
    for source in (payload, quality):
        reprojection_error = _first_number(
            source,
            ("reprojection_error_px", "mean_reprojection_error_px", "alignment_error_px"),
        )
        if reprojection_error is not None:
            break
    if require_reprojection_error and reprojection_error is None:
        blockers.append("camera_reprojection_error_missing")
    elif reprojection_error is None:
        warnings.append("camera_reprojection_error_missing")
    elif reprojection_error < 0:
        blockers.append("camera_reprojection_error_negative")
    elif reprojection_error > max_reprojection_error_px:
        blockers.append("camera_reprojection_error_too_high")

    normalized_intrinsics = intrinsics_result.get("normalized")
    projection_ready = bool(
        not blockers
        and normalized_intrinsics is not None
        and (not require_extrinsics or camera_matrix is not None)
    )
    return {
        "schema_version": "camera_calibration_validation.v1",
        "status": "passed" if projection_ready else "blocked",
        "projection_ready": projection_ready,
        "intrinsics": normalized_intrinsics,
        "horizontal_fov_deg": intrinsics_result.get("horizontal_fov_deg"),
        "vertical_fov_deg": intrinsics_result.get("vertical_fov_deg"),
        "extrinsics_present": camera_matrix is not None,
        "camera_from_reference": camera_matrix.tolist() if camera_matrix is not None else None,
        "reference_from_camera": reference_matrix.tolist() if reference_matrix is not None else None,
        "camera_from_reference_source": camera_key or reference_key or None,
        "reference_frame": reference_frame or None,
        "camera_frame": camera_frame or None,
        "translation_unit": translation_unit or None,
        "reprojection_error_px": reprojection_error,
        "inverse_consistency_error": inverse_consistency_error,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
    }


__all__ = [
    "matrix4",
    "validate_camera_calibration",
    "validate_camera_intrinsics",
    "validate_se3_matrix",
]
