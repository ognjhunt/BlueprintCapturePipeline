"""Fail-closed, scene-relative quality checks for learned Gaussian fields.

The checks in this module are intentionally representation-level rather than
scene-specific.  A retained pre-training field defines the admissible metric
envelope for a trained derivative; a standalone field is normalized by its own
robust occupied extent.  No learned tensor is clamped, filtered, or repaired.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


SCHEMA_VERSION = "gaussian_field_quality.v1"
DRIFT_SCHEMA_VERSION = "gaussian_field_source_relative_drift.v1"

ROBUST_LOWER_QUANTILE = 0.001
ROBUST_UPPER_QUANTILE = 0.999
MINIMUM_STATISTICAL_FIELD_COUNT = 1_000

# Dimensionless limits, calibrated with deliberately generous headroom above
# admitted room-scale fields.  The known divergent field measures roughly 57
# for scale/diagonal and >500 for center/diagonal; the known-good reference
# fields remain below 0.25 and 40 respectively.
MAX_SCALE_TO_ROBUST_DIAGONAL = 1.0
MAX_CENTER_DISTANCE_TO_ROBUST_DIAGONAL = 128.0

# A trained appearance derivative may not silently create new geometry far
# outside its immutable retained source or inflate learned kernels by orders of
# magnitude.  These are ratios, so the same contract applies to rooms, benches,
# tabletop captures, and smaller bounded workcells.
MAX_SOURCE_RELATIVE_SCALE_GROWTH = 8.0
MAX_SOURCE_RELATIVE_Q999_SCALE_GROWTH = 8.0
MAX_SOURCE_RELATIVE_POSITION_DRIFT = 64.0
MAX_SOURCE_RELATIVE_Q999_POSITION_DRIFT = 1.0
REFERENCE_AABB_MARGIN_DIAGONAL_FRACTION = 0.25
MAX_OUTSIDE_REFERENCE_AABB_FRACTION = 0.001


class GaussianFieldQualityError(ValueError):
    """Malformed tensor input that cannot receive a quality verdict."""


def _matrix(value: Any, *, name: str, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or array.shape[0] < 1:
        raise GaussianFieldQualityError(f"gaussian_field_{name}_shape_invalid")
    if not np.isfinite(array).all():
        raise GaussianFieldQualityError(f"gaussian_field_{name}_nonfinite")
    return array


def _vector(value: Any, *, name: str, count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise GaussianFieldQualityError(f"gaussian_field_{name}_invalid")
    return array


def _field_arrays(
    *, positions: Any, activated_scales: Any, opacities: Any | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    centers = _matrix(positions, name="positions", width=3)
    scales = _matrix(activated_scales, name="activated_scales", width=3)
    if scales.shape[0] != centers.shape[0] or np.any(scales <= 0.0):
        raise GaussianFieldQualityError("gaussian_field_activated_scales_invalid")
    alpha = (
        _vector(opacities, name="opacities", count=centers.shape[0])
        if opacities is not None
        else None
    )
    if alpha is not None and (np.any(alpha < 0.0) or np.any(alpha > 1.0)):
        raise GaussianFieldQualityError("gaussian_field_opacities_invalid")
    return centers, scales, alpha


def _geometry(positions: np.ndarray, scales: np.ndarray) -> dict[str, Any]:
    lower = np.quantile(positions, ROBUST_LOWER_QUANTILE, axis=0)
    upper = np.quantile(positions, ROBUST_UPPER_QUANTILE, axis=0)
    robust_extent = np.maximum(upper - lower, 0.0)
    max_scales = scales.max(axis=1)
    # A degenerate one-point field still receives a stable, scale-normalized
    # verdict instead of dividing by zero.
    robust_diagonal = max(
        float(np.linalg.norm(robust_extent)),
        float(np.median(max_scales)) * 2.0,
        float(np.finfo(np.float64).eps),
    )
    robust_center = (lower + upper) / 2.0
    center_distances = np.linalg.norm(positions - robust_center, axis=1)
    return {
        "count": int(positions.shape[0]),
        "robust_bounds_min": lower.tolist(),
        "robust_bounds_max": upper.tolist(),
        "robust_extent": robust_extent.tolist(),
        "robust_diagonal": robust_diagonal,
        "full_bounds_min": positions.min(axis=0).tolist(),
        "full_bounds_max": positions.max(axis=0).tolist(),
        "max_activated_scale": float(max_scales.max()),
        "q999_activated_scale": float(np.quantile(max_scales, 0.999)),
        "max_scale_to_robust_diagonal": float(max_scales.max() / robust_diagonal),
        "max_center_distance_to_robust_diagonal": float(center_distances.max() / robust_diagonal),
    }


def measure_gaussian_field_quality(
    *, positions: Any, activated_scales: Any, opacities: Any | None = None
) -> dict[str, Any]:
    """Measure whether one field is geometrically renderable on its own scale."""

    centers, scales, alpha = _field_arrays(
        positions=positions, activated_scales=activated_scales, opacities=opacities
    )
    metrics = _geometry(centers, scales)
    blockers: list[str] = []
    statistically_gated = metrics["count"] >= MINIMUM_STATISTICAL_FIELD_COUNT
    if (
        statistically_gated
        and metrics["max_scale_to_robust_diagonal"] > MAX_SCALE_TO_ROBUST_DIAGONAL
    ):
        blockers.append("gaussian_field_scale_to_robust_extent_above_ceiling")
    if (
        statistically_gated
        and metrics["max_center_distance_to_robust_diagonal"]
        > MAX_CENTER_DISTANCE_TO_ROBUST_DIAGONAL
    ):
        blockers.append("gaussian_field_center_outlier_above_ceiling")
    if alpha is not None:
        max_scales = scales.max(axis=1)
        large = max_scales > 0.05 * metrics["robust_diagonal"]
        metrics["large_opaque_gaussian_count"] = int(np.count_nonzero(large & (alpha >= 0.5)))
        metrics["large_opaque_gaussian_fraction"] = float(np.mean(large & (alpha >= 0.5)))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified" if not blockers else "blocked",
        "metrics": metrics,
        "thresholds": {
            "robust_lower_quantile": ROBUST_LOWER_QUANTILE,
            "robust_upper_quantile": ROBUST_UPPER_QUANTILE,
            "maximum_scale_to_robust_diagonal": MAX_SCALE_TO_ROBUST_DIAGONAL,
            "maximum_center_distance_to_robust_diagonal": (MAX_CENTER_DISTANCE_TO_ROBUST_DIAGONAL),
            "minimum_statistical_field_count": MINIMUM_STATISTICAL_FIELD_COUNT,
        },
        "statistically_gated": statistically_gated,
        "blockers": sorted(blockers),
        "learned_tensors_mutated": False,
        "measurement_authority": "exact_gaussian_tensor_arrays",
    }


def measure_source_relative_gaussian_drift(
    *,
    reference_positions: Any,
    reference_activated_scales: Any,
    candidate_positions: Any,
    candidate_activated_scales: Any,
    candidate_opacities: Any | None = None,
) -> dict[str, Any]:
    """Compare a trained derivative with its immutable retained source field."""

    reference, reference_scales, _ = _field_arrays(
        positions=reference_positions,
        activated_scales=reference_activated_scales,
        opacities=None,
    )
    candidate, candidate_scales, candidate_alpha = _field_arrays(
        positions=candidate_positions,
        activated_scales=candidate_activated_scales,
        opacities=candidate_opacities,
    )
    blockers: list[str] = []
    if reference.shape[0] != candidate.shape[0]:
        return {
            "schema_version": DRIFT_SCHEMA_VERSION,
            "status": "blocked",
            "reference_count": int(reference.shape[0]),
            "candidate_count": int(candidate.shape[0]),
            "blockers": ["gaussian_field_source_relative_count_mismatch"],
            "learned_tensors_mutated": False,
            "measurement_authority": "exact_gaussian_tensor_arrays",
        }

    reference_geometry = _geometry(reference, reference_scales)
    candidate_quality = measure_gaussian_field_quality(
        positions=candidate,
        activated_scales=candidate_scales,
        opacities=candidate_alpha,
    )
    blockers.extend(candidate_quality["blockers"])
    normalizer = reference_geometry["robust_diagonal"]
    position_drift = np.linalg.norm(candidate - reference, axis=1)
    reference_max_scales = reference_scales.max(axis=1)
    candidate_max_scales = candidate_scales.max(axis=1)
    max_scale_growth = float(
        candidate_max_scales.max()
        / max(float(reference_max_scales.max()), np.finfo(np.float64).eps)
    )
    q999_scale_growth = float(
        np.quantile(candidate_max_scales, 0.999)
        / max(
            float(np.quantile(reference_max_scales, 0.999)),
            np.finfo(np.float64).eps,
        )
    )
    max_position_drift = float(position_drift.max() / normalizer)
    q999_position_drift = float(np.quantile(position_drift, 0.999) / normalizer)
    margin = REFERENCE_AABB_MARGIN_DIAGONAL_FRACTION * normalizer
    outside = np.any(
        (candidate < reference.min(axis=0) - margin) | (candidate > reference.max(axis=0) + margin),
        axis=1,
    )
    outside_fraction = float(outside.mean())

    if max_scale_growth > MAX_SOURCE_RELATIVE_SCALE_GROWTH:
        blockers.append("gaussian_field_source_relative_max_scale_growth_above_ceiling")
    if q999_scale_growth > MAX_SOURCE_RELATIVE_Q999_SCALE_GROWTH:
        blockers.append("gaussian_field_source_relative_q999_scale_growth_above_ceiling")
    if max_position_drift > MAX_SOURCE_RELATIVE_POSITION_DRIFT:
        blockers.append("gaussian_field_source_relative_position_drift_above_ceiling")
    if q999_position_drift > MAX_SOURCE_RELATIVE_Q999_POSITION_DRIFT:
        blockers.append("gaussian_field_source_relative_q999_position_drift_above_ceiling")
    if outside_fraction > MAX_OUTSIDE_REFERENCE_AABB_FRACTION:
        blockers.append("gaussian_field_source_relative_aabb_escape_above_ceiling")

    return {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "status": "qualified" if not blockers else "blocked",
        "reference_count": int(reference.shape[0]),
        "candidate_count": int(candidate.shape[0]),
        "reference_geometry": reference_geometry,
        "candidate_quality": candidate_quality,
        "metrics": {
            "max_scale_growth": max_scale_growth,
            "q999_scale_growth": q999_scale_growth,
            "max_position_drift_to_reference_robust_diagonal": max_position_drift,
            "q999_position_drift_to_reference_robust_diagonal": q999_position_drift,
            "outside_inflated_reference_aabb_fraction": outside_fraction,
        },
        "thresholds": {
            "maximum_scale_growth": MAX_SOURCE_RELATIVE_SCALE_GROWTH,
            "maximum_q999_scale_growth": MAX_SOURCE_RELATIVE_Q999_SCALE_GROWTH,
            "maximum_position_drift_to_reference_robust_diagonal": (
                MAX_SOURCE_RELATIVE_POSITION_DRIFT
            ),
            "maximum_q999_position_drift_to_reference_robust_diagonal": (
                MAX_SOURCE_RELATIVE_Q999_POSITION_DRIFT
            ),
            "reference_aabb_margin_robust_diagonal_fraction": (
                REFERENCE_AABB_MARGIN_DIAGONAL_FRACTION
            ),
            "maximum_outside_reference_aabb_fraction": (MAX_OUTSIDE_REFERENCE_AABB_FRACTION),
        },
        "blockers": sorted(set(blockers)),
        "learned_tensors_mutated": False,
        "measurement_authority": "exact_retained_and_trained_gaussian_tensor_arrays",
    }


def gaussian_quality_is_qualified(value: Any) -> bool:
    """Strict structural check used at receipt and packet boundaries."""

    return bool(
        isinstance(value, Mapping)
        and value.get("schema_version") == SCHEMA_VERSION
        and value.get("status") == "qualified"
        and value.get("blockers") == []
        and value.get("learned_tensors_mutated") is False
    )


def gaussian_drift_is_qualified(value: Any) -> bool:
    """Strict source-relative receipt check used across provider boundaries."""

    return bool(
        isinstance(value, Mapping)
        and value.get("schema_version") == DRIFT_SCHEMA_VERSION
        and value.get("status") == "qualified"
        and value.get("blockers") == []
        and value.get("learned_tensors_mutated") is False
        and gaussian_quality_is_qualified(value.get("candidate_quality"))
    )


__all__ = [
    "DRIFT_SCHEMA_VERSION",
    "GaussianFieldQualityError",
    "SCHEMA_VERSION",
    "gaussian_drift_is_qualified",
    "gaussian_quality_is_qualified",
    "measure_gaussian_field_quality",
    "measure_source_relative_gaussian_drift",
]
