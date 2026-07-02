"""Metric depth maps straight from splat centers — no DA3, no GPU.

The labels-free bootstrap path needs a per-view depth map to unproject 2D
detections into world AABBs. For scenes where the ONLY input is the splat
itself, the splat centers ARE the scene geometry: projecting every sufficiently
opaque center into the camera and keeping the nearest hit per pixel gives a
point-based z-buffer that is exactly aligned with what the reference splat
renderer shows (same gaussians, same camera). That skips the DA3 model (and its
GPU) entirely for this path; DA3 remains the depth source for real-photo
captures where no splat exists.

Conventions match :mod:`scene_placement.perception_index` bit-for-bit — the
camera mapping is ``{eye, target, up, vfov, width, height}``, and depth is
metric Z-DEPTH ALONG THE OPTICAL AXIS (not Euclidean range), which is what
``unproject`` scales pixel rays by. The buffer is rendered at a reduced
resolution (splat centers are sparse; a full-res point z-buffer is mostly
holes) and sampled through :func:`scene_placement.perception_adapter.
depth_provider_from_map`, which already handles downsampled maps.

Truth boundary: this is geometry FROM the reconstruction, so it inherits the
reconstruction's errors (missing thin structures, floaters). It claims camera-
consistent depth for unprojection, not survey-grade ranging.
"""
from __future__ import annotations

import warnings
from typing import Mapping, Tuple

import numpy as np

from .gaussian_splat_decode import SplatData
from .scene_placement.perception_index import (
    camera_basis,
    resolve_extrinsics,
    resolve_intrinsics,
)

DEFAULT_DEPTH_SCALE = 4  # depth buffer at 1/4 the RGB resolution
DEFAULT_MIN_OPACITY = 0.30
DEFAULT_NEAR_M = 0.05
# Median-of-neighborhood hole fill: a point z-buffer leaves empty pixels between
# projected centers; one dilation pass with the median of valid neighbors closes
# single-pixel holes without smearing depth edges the way a mean would.
_FILL_PASSES = 2


def _project_points(
    xyz: np.ndarray,
    camera: Mapping[str, object],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into pixel coords; returns (px, py, z_depth) kept-only."""
    fx, fy, cx, cy = resolve_intrinsics(
        {"vfov": camera["vfov"], "width": width, "height": height}
    )
    eye, target, up = resolve_extrinsics(camera)
    right, up_cam, forward = camera_basis(eye, target, up)
    rel = xyz - np.asarray(eye, dtype=np.float64)
    z = rel @ np.asarray(forward, dtype=np.float64)
    keep = z > DEFAULT_NEAR_M
    rel = rel[keep]
    z = z[keep]
    x_c = rel @ np.asarray(right, dtype=np.float64)
    y_c = rel @ np.asarray(up_cam, dtype=np.float64)
    px = x_c / z * fx + cx
    py = cy - y_c / z * fy  # image y grows downward
    in_frame = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    return px[in_frame], py[in_frame], z[in_frame]


def _fill_holes_median(depth: np.ndarray) -> np.ndarray:
    """Fill invalid (inf) pixels with the median of their valid 8-neighbors."""
    filled = depth.copy()
    h, w = filled.shape
    for _ in range(_FILL_PASSES):
        invalid = ~np.isfinite(filled)
        if not invalid.any():
            break
        # Stack shifted copies; nanmedian over the neighborhood.
        stack = np.full((8, h, w), np.nan, dtype=np.float64)
        shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for k, (dy, dx) in enumerate(shifts):
            src = filled[
                max(0, -dy) : h - max(0, dy),
                max(0, -dx) : w - max(0, dx),
            ]
            stack[k][
                max(0, dy) : h - max(0, -dy),
                max(0, dx) : w - max(0, -dx),
            ] = np.where(np.isfinite(src), src, np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="All-NaN slice encountered", category=RuntimeWarning
            )
            neighborhood = np.nanmedian(stack, axis=0)
        take = invalid & np.isfinite(neighborhood)
        filled[take] = neighborhood[take]
    return filled


def render_pointsplat_depth(
    splat: SplatData,
    camera: Mapping[str, object],
    *,
    depth_scale: int = DEFAULT_DEPTH_SCALE,
    min_opacity: float = DEFAULT_MIN_OPACITY,
) -> np.ndarray:
    """Point z-buffer depth map for ``camera``, at ``1/depth_scale`` resolution.

    Returns an ``(H//s, W//s)`` float64 array of metric z-depths; pixels no splat
    center reached remain ``inf`` (the perception sampler drops non-finite
    readings). Nearest-hit per pixel, then a bounded median hole-fill.
    """
    width = int(camera["width"])  # type: ignore[arg-type]
    height = int(camera["height"])  # type: ignore[arg-type]
    scale = max(1, int(depth_scale))
    bw, bh = max(1, width // scale), max(1, height // scale)
    opacity = splat.opacity_sigmoid
    pts = splat.xyz[opacity >= float(min_opacity)].astype(np.float64)
    depth = np.full((bh, bw), np.inf, dtype=np.float64)
    if pts.shape[0] == 0:
        return depth
    px, py, z = _project_points(pts, camera, width, height)
    if px.size == 0:
        return depth
    ix = np.clip((px / scale).astype(np.int64), 0, bw - 1)
    iy = np.clip((py / scale).astype(np.int64), 0, bh - 1)
    np.minimum.at(depth, (iy, ix), z)
    return _fill_holes_median(depth)


def depth_provider_for_camera(
    splat: SplatData,
    camera: Mapping[str, object],
    *,
    depth_scale: int = DEFAULT_DEPTH_SCALE,
    min_opacity: float = DEFAULT_MIN_OPACITY,
):
    """Convenience: render the buffer and wrap it in the standard depth provider.

    The returned callable is ``(px, py) -> meters`` in FULL-resolution pixel
    coordinates (the adapter rescales into the reduced buffer), i.e. exactly what
    :class:`scene_placement.perception_index.PerceptionSceneSpatialIndex` takes.
    """
    from .scene_placement.perception_adapter import depth_provider_from_map

    depth = render_pointsplat_depth(
        splat, camera, depth_scale=depth_scale, min_opacity=min_opacity
    )
    return depth_provider_from_map(
        depth,
        cam_width=int(camera["width"]),  # type: ignore[arg-type]
        cam_height=int(camera["height"]),  # type: ignore[arg-type]
    )


__all__ = [
    "DEFAULT_DEPTH_SCALE",
    "DEFAULT_MIN_OPACITY",
    "depth_provider_for_camera",
    "render_pointsplat_depth",
]
