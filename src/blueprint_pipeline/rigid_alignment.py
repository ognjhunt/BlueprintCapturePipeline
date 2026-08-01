"""Deterministic similarity-transform estimation shared by alignment lanes.

Umeyama closed-form estimation with the rotation constrained proper
(determinant +1).  Callers must fail closed when ``reflection_preferred`` is
true: a well-fitting improper transform means the two frames disagree in
handedness, which no silent correction may absorb.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class SimilarityAlignmentError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def estimate_similarity_transform(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, bool]:
    """Return (scale, rotation, translation, reflection_preferred).

    Maps ``source`` into ``target`` as ``target ~= scale * R @ source + t`` with
    ``det(R) = +1`` always.  ``reflection_preferred`` reports whether an
    improper transform would have fit better.
    """

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise SimilarityAlignmentError(["alignment_point_sets_shape_invalid"])
    if source.shape[0] < 3:
        raise SimilarityAlignmentError(["alignment_point_sets_too_small"])
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise SimilarityAlignmentError(["alignment_point_sets_nonfinite"])
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(np.square(source_centered), axis=1)))
    if source_variance <= 0.0:
        raise SimilarityAlignmentError(["alignment_source_degenerate"])
    covariance = (target_centered.T @ source_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    reflection_preferred = bool(np.linalg.det(u @ vt) < 0.0)
    correction = np.ones(3)
    if reflection_preferred:
        correction[2] = -1.0
    rotation = u @ np.diag(correction) @ vt
    scale = float(np.sum(singular_values * correction) / source_variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise SimilarityAlignmentError(["alignment_scale_invalid"])
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation, reflection_preferred


def similarity_residuals(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[float, float]:
    """Return (rms_residual, max_residual) of the fitted similarity transform."""

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    residual_vectors = target - (scale * (source @ np.asarray(rotation).T) + translation)
    residuals = np.linalg.norm(residual_vectors, axis=1)
    return float(np.sqrt(np.mean(np.square(residuals)))), float(np.max(residuals))


__all__ = [
    "SimilarityAlignmentError",
    "estimate_similarity_transform",
    "similarity_residuals",
]
