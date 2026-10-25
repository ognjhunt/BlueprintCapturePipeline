"""Align two Atlas poses of the same views by the depth both returned.

ADP-040, day 28 (scene preparation). A background posed from cleaned views can
sit in a world whose cameras disagree with the source pose's: the editor worked
on an earlier crop, so Atlas estimated other intrinsics, and a short walk leaves
camera centres too close together to fix a similarity. Both poses still see one
room. Each shared view's two returned images are registered in 2-D; wherever
both show the same unedited pixel, both returned depths lift it, and one robust
similarity of those 3-D pairs takes the source world to the background world.
Edited pixels differ in colour and are never paired; a view whose depth
disagrees is outvoted. The result is an estimate, never a physical measurement;
a weak fit returns None.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .atlas_depth_splat import EDGE_RELATIVE_JUMP
from .atlas_scene_measurements import unproject

#: Both images are one photograph, perhaps re-cropped: enough feature inliers,
#: an upright turn and a plausible zoom, or the view is not registered.
MIN_VIEW_INLIERS = 30
MAX_VIEW_TURN_DEGREES = 10.0
VIEW_ZOOM_BOUNDS = (0.25, 4.0)
#: Mean RGB difference, after the view's median offset, beyond which a pixel was edited.
COLOUR_TOLERANCE = 24.0
SAMPLES_PER_VIEW = 20_000
#: A pair agrees within this fraction of the background's extent.
INLIER_FRACTION_OF_EXTENT = 0.01
MIN_INLIER_FRACTION = 0.3
MAX_MEDIAN_RESIDUAL_FRACTION = 0.02
#: A view agrees when this share of its pairs fits; one view alone is not enough.
MIN_VIEW_AGREEMENT = 0.2
MIN_AGREEING_VIEWS = 2
SCALE_BOUNDS = (0.1, 10.0)
RANSAC_ITERATIONS = 400
RANSAC_PROBE = 20_000


def _register(source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray | None:
    """The 2x3 map of source pixel centres onto target pixel centres for one view, or None."""
    import cv2

    (sh, sw), (th, tw) = source_rgb.shape[:2], target_rgb.shape[:2]
    if abs(sw / sh - tw / th) < 0.002 and np.abs(
            cv2.resize(source_rgb, (tw, th)).astype(np.float64) - target_rgb).mean() < 8:
        a, b = tw / sw, th / sh
        return np.array([[a, 0.0, 0.5 * a - 0.5], [0.0, b, 0.5 * b - 0.5]])
    if hasattr(cv2, "SIFT_create"):
        detector, norm = cv2.SIFT_create(nfeatures=4000), cv2.NORM_L2
    else:
        detector, norm = cv2.ORB_create(nfeatures=4000), cv2.NORM_HAMMING
    (ka, da), (kb, db) = (detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
                          for image in (source_rgb, target_rgb))
    if da is None or db is None or len(ka) < 2 or len(kb) < 2:
        return None
    matches = [pair[0] for pair in cv2.BFMatcher(norm).knnMatch(da, db, k=2)
               if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(matches) < MIN_VIEW_INLIERS:
        return None
    p = np.float32([ka[match.queryIdx].pt for match in matches])
    q = np.float32([kb[match.trainIdx].pt for match in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None or int(inliers.sum()) < MIN_VIEW_INLIERS:
        return None
    zoom = math.sqrt(abs(float(np.linalg.det(matrix[:, :2]))))
    turn = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    if not VIEW_ZOOM_BOUNDS[0] <= zoom <= VIEW_ZOOM_BOUNDS[1] or abs(turn) > MAX_VIEW_TURN_DEGREES:
        return None
    return matrix


def _continuous(frame: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Confident depth and the pixels away from depth discontinuities."""
    depth = np.where((frame["confidence"] > 0) & (frame["depth"] > 0), frame["depth"], np.nan).astype(np.float64)
    keep = np.zeros(depth.shape, dtype=bool)
    center = depth[1:-1, 1:-1]
    inner = np.isfinite(center)
    with np.errstate(invalid="ignore"):
        for neighbour in (depth[1:-1, 2:], depth[1:-1, :-2], depth[2:, 1:-1], depth[:-2, 1:-1]):
            inner &= np.abs(neighbour - center) <= EDGE_RELATIVE_JUMP * center
    keep[1:-1, 1:-1] = inner
    return depth, keep


def _view_pairs(source: Mapping[str, Any], target: Mapping[str, Any], matrix: np.ndarray,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """World points of the same unedited pixels: (source world, target world)."""
    import cv2

    source_depth, source_keep = _continuous(source)
    target_depth, target_keep = _continuous(target)
    ys, xs = np.nonzero(target_keep)
    if len(xs) > 4 * SAMPLES_PER_VIEW:
        chosen = rng.choice(len(xs), 4 * SAMPLES_PER_VIEW, replace=False)
        ys, xs = ys[chosen], xs[chosen]
    back = cv2.invertAffineTransform(matrix)
    sx, sy = np.rint(np.column_stack([xs, ys]) @ back[:, :2].T + back[:, 2]).astype(np.int64).T
    height, width = source_depth.shape
    keep = (sx >= 0) & (sy >= 0) & (sx < width) & (sy < height)
    xs, ys, sx, sy = xs[keep], ys[keep], sx[keep], sy[keep]
    keep = source_keep[sy, sx]
    xs, ys, sx, sy = xs[keep], ys[keep], sx[keep], sy[keep]
    difference = source["rgb"][sy, sx].astype(np.float64) - target["rgb"][ys, xs]
    if len(difference):
        difference -= np.median(difference, axis=0)
    keep = np.abs(difference).mean(axis=1) <= COLOUR_TOLERANCE
    xs, ys, sx, sy = xs[keep], ys[keep], sx[keep], sy[keep]
    if len(xs) > SAMPLES_PER_VIEW:
        chosen = rng.choice(len(xs), SAMPLES_PER_VIEW, replace=False)
        xs, ys, sx, sy = xs[chosen], ys[chosen], sx[chosen], sy[chosen]
    return (unproject(source, np.column_stack([sx, sy]).astype(np.float64), source_depth[sy, sx]),
            unproject(target, np.column_stack([xs, ys]).astype(np.float64), target_depth[ys, xs]))


def _similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares scale, rotation and translation taking points ``a`` onto ``b`` (Umeyama)."""
    a_mean, b_mean = a.mean(axis=0), b.mean(axis=0)
    u, singular, vt = np.linalg.svd((b - b_mean).T @ (a - a_mean) / len(a))
    sign = np.diag([1.0, 1.0, 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0])
    rotation = u @ sign @ vt
    variance = float(((a - a_mean) ** 2).sum() / len(a))
    scale = float((singular * np.diag(sign)).sum() / variance) if variance > 0 else 0.0
    return scale, rotation, b_mean - scale * rotation @ a_mean


def shared_view_alignment(source_frames: Sequence[Mapping[str, Any]],
                          target_frames: Sequence[Mapping[str, Any]], *,
                          estimated_depth_tolerance: bool = False) -> dict[str, Any] | None:
    """The similarity taking the source pose's world to the target pose's, from their shared views' depth.

    Frames are ``atlas_depth_splat.posed_frame`` records. Returns None unless at
    least two views agree on one well-supported fit.
    """
    if type(estimated_depth_tolerance) is not bool:
        raise ValueError('atlas_shared_alignment_tolerance_invalid')
    targets = {frame["frame_id"]: frame for frame in target_frames}
    shared = [frame for frame in source_frames if frame["frame_id"] in targets]
    rng = np.random.default_rng(607)
    lifted, registered = [], 0
    for frame in shared:
        matrix = _register(frame["rgb"], targets[frame["frame_id"]]["rgb"])
        if matrix is None:
            continue
        registered += 1
        pair = _view_pairs(frame, targets[frame["frame_id"]], matrix, rng)
        if len(pair[0]):
            lifted.append(pair)
    if len(lifted) < MIN_AGREEING_VIEWS:
        return None
    a, b = (np.concatenate([pair[side] for pair in lifted]) for side in (0, 1))
    view = np.concatenate([np.full(len(pair[0]), index) for index, pair in enumerate(lifted)])
    extent = float(np.linalg.norm(np.percentile(b, 95, axis=0) - np.percentile(b, 5, axis=0)))
    if not math.isfinite(extent) or extent <= 0:
        return None
    threshold = INLIER_FRACTION_OF_EXTENT * extent
    median_limit = MAX_MEDIAN_RESIDUAL_FRACTION * extent
    if estimated_depth_tolerance:
        # Development completion compares two independently estimated scenes.
        # Allow the sum of their existing relative fit allowances. This remains
        # a scene-scaled heuristic for candidate geometry, never calibrated
        # measurement accuracy. Source-priority/free-space checks still apply
        # downstream; the default registration rule remains unchanged.
        threshold *= 2
        median_limit *= 2
    probe = rng.choice(len(a), min(len(a), RANSAC_PROBE), replace=False)
    best = None
    for _ in range(RANSAC_ITERATIONS):
        sample = rng.choice(len(a), 3, replace=False)
        if np.linalg.norm(np.cross(b[sample[1]] - b[sample[0]], b[sample[2]] - b[sample[0]])) < (0.05 * extent) ** 2:
            continue
        scale, rotation, translation = _similarity(a[sample], b[sample])
        if not SCALE_BOUNDS[0] <= scale <= SCALE_BOUNDS[1]:
            continue
        count = int((np.linalg.norm(scale * a[probe] @ rotation.T + translation - b[probe], axis=1)
                     < threshold).sum())
        if best is None or count > best[0]:
            best = (count, scale, rotation, translation)
    if best is None:
        return None
    _, scale, rotation, translation = best
    for _ in range(3):
        inliers = np.linalg.norm(scale * a @ rotation.T + translation - b, axis=1) < threshold
        if inliers.sum() < 3:
            return None
        scale, rotation, translation = _similarity(a[inliers], b[inliers])
    residual = np.linalg.norm(scale * a @ rotation.T + translation - b, axis=1)
    inliers = residual < threshold
    agreeing = sum(1 for index in range(len(lifted)) if inliers[view == index].mean() >= MIN_VIEW_AGREEMENT)
    if (not SCALE_BOUNDS[0] <= scale <= SCALE_BOUNDS[1] or agreeing < MIN_AGREEING_VIEWS
            or inliers.mean() < MIN_INLIER_FRACTION
            or float(np.median(residual)) > median_limit):
        return None
    return {"scale": scale, "rotation": rotation, "translation": translation,
            "shared_frame_count": len(shared), "registered_frame_count": registered,
            "agreeing_frame_count": agreeing, "pair_count": int(len(a)),
            "inlier_fraction": float(inliers.mean()),
            "median_residual_fraction": float(np.median(residual)) / extent,
            **({'tolerance_basis': 'development_estimated_depth_cross_view_uncertainty',
                'inlier_distance_limit': threshold, 'median_distance_limit': median_limit,
                'metric_measurement_proven': False, 'claim_ceiling': 'development_only'}
               if estimated_depth_tolerance else {})}
