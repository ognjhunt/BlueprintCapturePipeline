"""Deterministic two-view epipolar consistency check for posed candidate images.

The check verifies that declared OpenCV-convention camera-to-world poses and
pinhole intrinsics actually explain the pixel geometry of adjacent candidate
image pairs.  It exists because a published trajectory whose camera-axis
convention is misread (for example OpenGL treated as OpenCV) still passes
orthonormality and digest checks while making every downstream reconstruction
input geometrically wrong.  The check reads candidate images only; it never
touches evaluator-hidden observations.

Verdicts:

* ``consistent`` — enough confident matches on well-conditioned pairs and the
  aggregate median epipolar distance is inside the frozen threshold.
* ``inconsistent`` — enough matches but the geometry does not explain them.
* ``inconclusive`` — not enough texture/baseline to decide; callers must treat
  this as a blocker for paid or scientific downstream use, not as a pass.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


CHECK_VERSION = "two_view_epipolar_consistency.v1"

MINIMUM_PAIR_BASELINE = 0.005
MAXIMUM_PAIR_BASELINE = 0.06
MAXIMUM_PAIRS = 3
CORNER_GRID_STRIDE = 42
CORNER_MINIMUM_STRENGTH = 900.0
CORNER_MINIMUM_PATCH_STD = 12.0
MAXIMUM_CORNERS = 120
PATCH_RADIUS = 9
SEARCH_STRIDE = 3
MINIMUM_MATCH_SCORE = 0.9
MINIMUM_MATCH_MARGIN = 0.03
MINIMUM_CONFIDENT_MATCHES_PER_PAIR = 8
MAXIMUM_MEDIAN_EPIPOLAR_PX = 1.5


class PoseImageConsistencyError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64)


def _camera(observation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    camera = observation.get("camera")
    camera = camera if isinstance(camera, Mapping) else observation
    matrix = np.asarray(camera["T_world_camera"], dtype=np.float64)
    intrinsics = camera["rgb_intrinsics"]
    k = np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ]
    )
    return matrix, k


def _fundamental(
    matrix_a: np.ndarray, matrix_b: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    rotation_a, translation_a = matrix_a[:3, :3], matrix_a[:3, 3]
    rotation_b, translation_b = matrix_b[:3, :3], matrix_b[:3, 3]
    relative_rotation = rotation_b.T @ rotation_a
    relative_translation = rotation_b.T @ (translation_a - translation_b)
    tx = np.array(
        [
            [0.0, -relative_translation[2], relative_translation[1]],
            [relative_translation[2], 0.0, -relative_translation[0]],
            [-relative_translation[1], relative_translation[0], 0.0],
        ]
    )
    inverse_intrinsic = np.linalg.inv(intrinsic)
    return inverse_intrinsic.T @ (tx @ relative_rotation) @ inverse_intrinsic


def _corners(image: np.ndarray) -> list[tuple[int, int]]:
    gradient_y, gradient_x = np.gradient(image)
    strength = gradient_x**2 + gradient_y**2
    height, width = image.shape
    margin = PATCH_RADIUS * 2 + 2
    points: list[tuple[int, int]] = []
    for row in range(margin, height - margin, CORNER_GRID_STRIDE):
        for column in range(margin, width - margin, CORNER_GRID_STRIDE):
            window = strength[row - 10 : row + 10, column - 10 : column + 10]
            offset = np.unravel_index(int(np.argmax(window)), window.shape)
            point_row, point_column = row - 10 + int(offset[0]), column - 10 + int(offset[1])
            if (
                point_row < margin
                or point_column < margin
                or point_row >= height - margin
                or point_column >= width - margin
            ):
                continue
            patch = image[
                point_row - PATCH_RADIUS : point_row + PATCH_RADIUS + 1,
                point_column - PATCH_RADIUS : point_column + PATCH_RADIUS + 1,
            ]
            if float(window[offset]) > CORNER_MINIMUM_STRENGTH and float(patch.std()) > (
                CORNER_MINIMUM_PATCH_STD
            ):
                points.append((point_column, point_row))
            if len(points) >= MAXIMUM_CORNERS:
                return points
    return points


def _match(
    image_a: np.ndarray,
    image_b: np.ndarray,
    column: int,
    row: int,
    *,
    search_radius: int,
) -> tuple[float, float, int, int] | None:
    height, width = image_b.shape
    patch = image_a[
        row - PATCH_RADIUS : row + PATCH_RADIUS + 1,
        column - PATCH_RADIUS : column + PATCH_RADIUS + 1,
    ]
    patch = patch - patch.mean()
    patch_norm = float(np.sqrt((patch * patch).sum())) + 1e-9
    top = max(0, row - search_radius)
    bottom = min(height, row + search_radius + 1)
    left = max(0, column - search_radius)
    right = min(width, column + search_radius + 1)
    region = image_b[top:bottom, left:right]
    size = PATCH_RADIUS * 2 + 1
    if region.shape[0] < size or region.shape[1] < size:
        return None
    all_windows = sliding_window_view(region, (size, size))
    windows = all_windows[::SEARCH_STRIDE, ::SEARCH_STRIDE]
    means = windows.mean(axis=(2, 3), keepdims=True)
    centered = windows - means
    norms = np.sqrt((centered * centered).sum(axis=(2, 3))) + 1e-9
    scores = (centered * patch).sum(axis=(2, 3)) / (norms * patch_norm)
    flat = scores.ravel()
    if flat.size < 2:
        return None
    order = np.argsort(flat)
    best_index, second_index = int(order[-1]), int(order[-2])
    best, second = float(flat[best_index]), float(flat[second_index])
    best_row, best_column = np.unravel_index(best_index, scores.shape)
    coarse_row = int(best_row) * SEARCH_STRIDE
    coarse_column = int(best_column) * SEARCH_STRIDE
    # Stride-1 local refinement around the coarse winner keeps the sweep cheap
    # while restoring integer-pixel precision for the epipolar distances.
    refined_row, refined_column = coarse_row, coarse_column
    for delta_row in range(-SEARCH_STRIDE + 1, SEARCH_STRIDE):
        for delta_column in range(-SEARCH_STRIDE + 1, SEARCH_STRIDE):
            row_index = coarse_row + delta_row
            column_index = coarse_column + delta_column
            if (
                (delta_row, delta_column) == (0, 0)
                or row_index < 0
                or column_index < 0
                or row_index >= all_windows.shape[0]
                or column_index >= all_windows.shape[1]
            ):
                continue
            window = all_windows[row_index, column_index]
            window_centered = window - window.mean()
            window_norm = float(np.sqrt((window_centered * window_centered).sum())) + 1e-9
            score = float((window_centered * patch).sum() / (window_norm * patch_norm))
            if score > best:
                best, refined_row, refined_column = score, row_index, column_index
    match_row = top + refined_row + PATCH_RADIUS
    match_column = left + refined_column + PATCH_RADIUS
    return best, second, match_column, match_row


def check_two_view_epipolar_consistency(
    *,
    observations: Sequence[Mapping[str, Any]],
    image_root: str | Path,
    maximum_median_epipolar_px: float = MAXIMUM_MEDIAN_EPIPOLAR_PX,
) -> dict[str, Any]:
    """Check adjacent candidate pairs against their declared epipolar geometry."""

    root = Path(image_root).resolve()
    rows = sorted(
        (dict(observation) for observation in observations),
        key=lambda observation: str(observation.get("observation_id") or ""),
    )
    if len(rows) < 2:
        return {
            "schema_version": CHECK_VERSION,
            "status": "inconclusive",
            "reason": "fewer_than_two_candidate_observations",
            "pairs": [],
            "maximum_median_epipolar_px": maximum_median_epipolar_px,
        }
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for left, right in zip(rows, rows[1:]):
        matrix_a, _ = _camera(left)
        matrix_b, _ = _camera(right)
        baseline = float(np.linalg.norm(matrix_a[:3, 3] - matrix_b[:3, 3]))
        if MINIMUM_PAIR_BASELINE <= baseline <= MAXIMUM_PAIR_BASELINE:
            candidates.append((baseline, left, right))
    candidates.sort(key=lambda item: item[0])
    pair_reports: list[dict[str, Any]] = []
    medians: list[float] = []
    for baseline, left, right in candidates[: MAXIMUM_PAIRS * 3]:
        if len([r for r in pair_reports if r["status"] == "measured"]) >= MAXIMUM_PAIRS:
            break
        matrix_a, intrinsic = _camera(left)
        matrix_b, _ = _camera(right)
        image_a = _grayscale(root / str(left["image_relative_path"]))
        image_b = _grayscale(root / str(right["image_relative_path"]))
        if image_a.shape != image_b.shape:
            pair_reports.append(
                {
                    "pair": [left["observation_id"], right["observation_id"]],
                    "status": "skipped_shape_mismatch",
                }
            )
            continue
        search_radius = max(60, int(round(image_a.shape[1] * 0.15)))
        fundamental = _fundamental(matrix_a, matrix_b, intrinsic)
        matches: list[tuple[int, int, int, int]] = []
        for column, row in _corners(image_a):
            matched = _match(image_a, image_b, column, row, search_radius=search_radius)
            if matched is None:
                continue
            best, second, match_column, match_row = matched
            if best > MINIMUM_MATCH_SCORE and best - second > MINIMUM_MATCH_MARGIN:
                matches.append((column, row, match_column, match_row))
        if len(matches) < MINIMUM_CONFIDENT_MATCHES_PER_PAIR:
            pair_reports.append(
                {
                    "pair": [left["observation_id"], right["observation_id"]],
                    "status": "insufficient_confident_matches",
                    "confident_matches": len(matches),
                }
            )
            continue
        distances = []
        for column, row, match_column, match_row in matches:
            line = fundamental @ np.array([float(column), float(row), 1.0])
            denominator = math.hypot(float(line[0]), float(line[1])) + 1e-12
            distances.append(
                abs(float(line @ np.array([float(match_column), float(match_row), 1.0])))
                / denominator
            )
        median = float(np.median(distances))
        medians.append(median)
        pair_reports.append(
            {
                "pair": [left["observation_id"], right["observation_id"]],
                "status": "measured",
                "baseline": round(baseline, 6),
                "confident_matches": len(matches),
                "median_epipolar_px": round(median, 4),
                "fraction_within_3px": round(
                    float(np.mean(np.asarray(distances) < 3.0)), 4
                ),
            }
        )
    if not medians:
        return {
            "schema_version": CHECK_VERSION,
            "status": "inconclusive",
            "reason": "no_pair_produced_enough_confident_matches",
            "pairs": pair_reports,
            "maximum_median_epipolar_px": maximum_median_epipolar_px,
        }
    aggregate_median = float(np.median(medians))
    status = (
        "consistent" if aggregate_median <= maximum_median_epipolar_px else "inconsistent"
    )
    return {
        "schema_version": CHECK_VERSION,
        "status": status,
        "aggregate_median_epipolar_px": round(aggregate_median, 4),
        "measured_pair_count": len(medians),
        "pairs": pair_reports,
        "maximum_median_epipolar_px": maximum_median_epipolar_px,
    }


__all__ = [
    "CHECK_VERSION",
    "PoseImageConsistencyError",
    "check_two_view_epipolar_consistency",
]
