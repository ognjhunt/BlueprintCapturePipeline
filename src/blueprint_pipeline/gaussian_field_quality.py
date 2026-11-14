"""Fail-closed, scene-relative quality checks for learned Gaussian fields.

The checks in this module are intentionally representation-level rather than
scene-specific.  A retained pre-training field defines the admissible metric
envelope for a trained derivative; a standalone field is normalized by its own
robust occupied extent.  No learned tensor is clamped, filtered, or repaired.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
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


# A field is read in blocks of this many rows, so the working set does not grow
# with the field: a float32 array (USD's own, read without a copy) is never
# converted whole to float64.
CHUNK_ROWS = 1 << 18


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


def field_array_view(value: Any) -> np.ndarray:
    """The input as an array, without a copy where it already is one (a USD array included)."""
    array = np.asarray(value)
    return array if array.dtype.kind in "biuf" else np.asarray(value, dtype=np.float64)


def _blocks(array: np.ndarray, rows: int = CHUNK_ROWS) -> Iterator[np.ndarray]:
    """The rows of ``array`` as float64, one bounded block at a time."""
    for start in range(0, array.shape[0], rows):
        yield np.asarray(array[start : start + rows], dtype=np.float64)


def _finite(array: np.ndarray) -> bool:
    return all(bool(np.isfinite(block).all()) for block in _blocks(array))


def _matrix_view(value: Any, *, name: str, width: int) -> np.ndarray:
    array = field_array_view(value)
    if array.ndim != 2 or array.shape[1] != width or array.shape[0] < 1:
        raise GaussianFieldQualityError(f"gaussian_field_{name}_shape_invalid")
    if not _finite(array):
        raise GaussianFieldQualityError(f"gaussian_field_{name}_nonfinite")
    return array


def _field_views(
    *, positions: Any, activated_scales: Any, opacities: Any | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``_field_arrays``'s checks, in its order, read block by block."""
    centers = _matrix_view(positions, name="positions", width=3)
    scales = _matrix_view(activated_scales, name="activated_scales", width=3)
    if scales.shape[0] != centers.shape[0] or any(bool((block <= 0.0).any()) for block in _blocks(scales)):
        raise GaussianFieldQualityError("gaussian_field_activated_scales_invalid")
    alpha = None
    if opacities is not None:
        alpha = field_array_view(opacities).reshape(-1)
        if alpha.shape != (centers.shape[0],) or not _finite(alpha):
            raise GaussianFieldQualityError("gaussian_field_opacities_invalid")
        if any(bool((block < 0.0).any() or (block > 1.0).any()) for block in _blocks(alpha)):
            raise GaussianFieldQualityError("gaussian_field_opacities_invalid")
    return centers, scales, alpha


_SIGN_BIT = np.uint64(1 << 63)
_DIGIT = np.uint64(0xFFFF)


def _order_keys(values: np.ndarray) -> np.ndarray:
    """Unsigned keys whose order is the float64 values' order."""
    bits = np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)
    return np.where(bits & _SIGN_BIT, ~bits, bits | _SIGN_BIT)


def _key_value(key: int) -> np.float64:
    bits = key ^ (1 << 63) if key >> 63 else ~key & 0xFFFF_FFFF_FFFF_FFFF
    return np.array([bits], dtype=np.uint64).view(np.float64)[0]


def _order_statistics(blocks: Callable[[], Iterator[np.ndarray]], ranks: set[int]) -> dict[int, np.float64]:
    """The exact values at the given 0-based ranks of the sorted values, read in blocks.

    A radix selection over the float64 bit patterns, 16 bits a pass: four passes,
    each holding one block and a 65,536-bin count per distinct prefix.
    """
    state = {rank: (0, rank) for rank in ranks}
    for level in range(4):
        shift = np.uint64(48 - 16 * level)
        groups: dict[int, list[int]] = {}
        for rank, (prefix, _) in state.items():
            groups.setdefault(prefix, []).append(rank)
        counts = {prefix: np.zeros(1 << 16, dtype=np.int64) for prefix in groups}
        for block in blocks():
            keys = _order_keys(block)
            digits = (keys >> shift) & _DIGIT
            if level == 0:
                counts[0] += np.bincount(digits.astype(np.intp), minlength=1 << 16)
                continue
            high = keys >> (shift + np.uint64(16))
            for prefix in groups:
                selected = digits[high == np.uint64(prefix)]
                counts[prefix] += np.bincount(selected.astype(np.intp), minlength=1 << 16)
        for prefix, members in groups.items():
            cumulative = np.cumsum(counts[prefix])
            for rank in members:
                remaining = state[rank][1]
                digit = int(np.searchsorted(cumulative, remaining, side="right"))
                below = int(cumulative[digit - 1]) if digit else 0
                state[rank] = ((prefix << 16) | digit, remaining - below)
    return {rank: _key_value(key) for rank, (key, _) in state.items()}


def _quantile_ranks(count: int, quantile: float) -> tuple[int, int, np.float64]:
    """numpy's linear quantile: the two ranks it reads and its interpolation weight."""
    virtual = (count - 1) * np.asarray(quantile, dtype=np.float64)
    if virtual >= count - 1:
        # numpy reads the last value twice and measures the weight from index -1.
        return count - 1, count - 1, np.float64(virtual - (-1))
    previous = int(np.floor(virtual))
    return previous, previous + 1, np.float64(virtual - previous)


def _lerp(below: np.float64, above: np.float64, weight: np.float64) -> np.float64:
    """numpy's quantile interpolation, operation for operation."""
    difference = above - below
    if weight >= 0.5:
        return above - difference * (1 - weight)
    return below + difference * weight


def _quantiles(
    blocks: Callable[[], Iterator[np.ndarray]], count: int, quantiles: tuple[float, ...]
) -> list[np.float64]:
    """``np.quantile(values, q)`` for each q, exactly, without holding the values."""
    plans = [_quantile_ranks(count, quantile) for quantile in quantiles]
    values = _order_statistics(blocks, {rank for plan in plans for rank in plan[:2]})
    return [_lerp(values[below], values[above], weight) for below, above, weight in plans]


def _median(blocks: Callable[[], Iterator[np.ndarray]], count: int) -> np.float64:
    """``np.median(values)``, exactly, without holding the values."""
    middle = count // 2
    if count % 2:
        return _order_statistics(blocks, {middle})[middle]
    values = _order_statistics(blocks, {middle - 1, middle})
    return np.add(values[middle - 1], values[middle]) / 2


def _geometry(positions: Any, scales: Any) -> dict[str, Any]:
    positions, scales = field_array_view(positions), field_array_view(scales)
    count = int(positions.shape[0])

    def column(axis: int) -> Callable[[], Iterator[np.ndarray]]:
        return lambda: (block[:, axis] for block in _blocks(positions))

    def max_scales() -> Iterator[np.ndarray]:
        return (block.max(axis=1) for block in _blocks(scales))

    bounds = [
        _quantiles(column(axis), count, (ROBUST_LOWER_QUANTILE, ROBUST_UPPER_QUANTILE))
        for axis in range(3)
    ]
    lower = np.array([row[0] for row in bounds], dtype=np.float64)
    upper = np.array([row[1] for row in bounds], dtype=np.float64)
    robust_extent = np.maximum(upper - lower, 0.0)
    largest_scale = np.maximum.reduce([block.max() for block in max_scales()])
    # A degenerate one-point field still receives a stable, scale-normalized
    # verdict instead of dividing by zero.
    robust_diagonal = max(
        float(np.linalg.norm(robust_extent)),
        float(_median(max_scales, count)) * 2.0,
        float(np.finfo(np.float64).eps),
    )
    robust_center = (lower + upper) / 2.0
    largest_center_distance = np.maximum.reduce(
        [np.linalg.norm(block - robust_center, axis=1).max() for block in _blocks(positions)]
    )
    full_min = np.minimum.reduce([block.min(axis=0) for block in _blocks(positions)])
    full_max = np.maximum.reduce([block.max(axis=0) for block in _blocks(positions)])
    return {
        "count": count,
        "robust_bounds_min": lower.tolist(),
        "robust_bounds_max": upper.tolist(),
        "robust_extent": robust_extent.tolist(),
        "robust_diagonal": robust_diagonal,
        "full_bounds_min": full_min.tolist(),
        "full_bounds_max": full_max.tolist(),
        "max_activated_scale": float(largest_scale),
        "q999_activated_scale": float(_quantiles(max_scales, count, (0.999,))[0]),
        "max_scale_to_robust_diagonal": float(largest_scale / robust_diagonal),
        "max_center_distance_to_robust_diagonal": float(largest_center_distance / robust_diagonal),
    }


def measure_gaussian_field_quality(
    *, positions: Any, activated_scales: Any, opacities: Any | None = None
) -> dict[str, Any]:
    """Measure whether one field is geometrically renderable on its own scale.

    The arrays are read in blocks of ``CHUNK_ROWS`` rows (a USD array without a
    copy), so the measure holds a fixed working set whatever the field's size;
    every figure is the one the whole-array computation gives.
    """

    centers, scales, alpha = _field_views(
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
        limit = 0.05 * metrics["robust_diagonal"]
        large_opaque = sum(
            int(np.count_nonzero((block.max(axis=1) > limit) & (opacity >= 0.5)))
            for block, opacity in zip(_blocks(scales), _blocks(alpha))
        )
        metrics["large_opaque_gaussian_count"] = large_opaque
        # np.mean of a boolean array: its count over its length, in float64.
        metrics["large_opaque_gaussian_fraction"] = float(np.float64(large_opaque) / metrics["count"])
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
    "CHUNK_ROWS",
    "GaussianFieldQualityError",
    "SCHEMA_VERSION",
    "field_array_view",
    "gaussian_drift_is_qualified",
    "gaussian_quality_is_qualified",
    "measure_gaussian_field_quality",
    "measure_source_relative_gaussian_drift",
]
