"""Depth-composite a robot-only Isaac render into a splat render — locally.

Isaac 6.0's ParticleField pass does not reliably depth-composite regular mesh
geometry with the gaussian scene (known splat-occlusion limitations), so a robot
referenced into the splat stage can render invisible even with all its mesh data
present. The two-pass fallback sidesteps the renderer: the GPU worker captures
(a) the splat scene and (b) a ROBOT-ONLY pass (splat hidden) with a
``distance_to_camera`` annotator, and this module composites them with correct
occlusion using the splat's own geometry for scene depth:

    robot pixel wins where  robot_z < splat_z + margin

Scene depth comes from :func:`splat_depth.render_pointsplat_depth` (the same
splat, the same camera — geometrically consistent with what the splat pass
drew). Robot depth converts the annotator's Euclidean range to optical-axis
z-depth so both sides use one convention.

Truth boundary: a visual composite with geometric occlusion — the robot was lit
by the mesh pass, not by the splat's radiance field. Good for POV/placement
review frames; not a claim of unified light transport.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np

from .gaussian_splat_decode import SplatData
from .splat_depth import render_pointsplat_depth

# Robot-only renders use Replicator's default empty background; range values
# beyond this are "no geometry on this ray" (sky/inf) rather than robot surface.
DEFAULT_MAX_ROBOT_RANGE_M = 500.0
DEFAULT_DEPTH_MARGIN_M = 0.10


def _range_to_z_depth(distance: np.ndarray, vfov: float, width: int, height: int) -> np.ndarray:
    """Euclidean range (Replicator ``distance_to_camera``) -> optical-axis z-depth."""
    fy = (height / 2.0) / math.tan(float(vfov) / 2.0)
    fx = fy
    cx, cy = width / 2.0, height / 2.0
    ys, xs = np.mgrid[0:height, 0:width]
    x_c = (xs + 0.5 - cx) / fx
    y_c = (cy - (ys + 0.5)) / fy
    ray_norm = np.sqrt(1.0 + x_c * x_c + y_c * y_c)
    return distance / ray_norm


def _upsample_to(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor upsample of a reduced depth buffer to full resolution."""
    reps_y = int(math.ceil(height / depth.shape[0]))
    reps_x = int(math.ceil(width / depth.shape[1]))
    return np.repeat(np.repeat(depth, reps_y, axis=0), reps_x, axis=1)[:height, :width]


def composite_robot_into_splat(
    splat_rgb_png: str | Path,
    robot_rgb_png: str | Path,
    robot_distance_npy: str | Path,
    camera: Mapping[str, object],
    splat: SplatData,
    out_png: str | Path,
    *,
    depth_margin_m: float = DEFAULT_DEPTH_MARGIN_M,
    max_robot_range_m: float = DEFAULT_MAX_ROBOT_RANGE_M,
    depth_scale: int = 2,
) -> dict:
    """Write the occlusion-correct composite; returns a report with pixel counts."""
    from PIL import Image

    scene = np.asarray(Image.open(str(splat_rgb_png)).convert("RGB"), dtype=np.uint8)
    robot = np.asarray(Image.open(str(robot_rgb_png)).convert("RGB"), dtype=np.uint8)
    if scene.shape != robot.shape:
        raise ValueError(f"frame shape mismatch: {scene.shape} vs {robot.shape}")
    height, width = scene.shape[:2]

    distance = np.load(str(robot_distance_npy)).astype(np.float64)
    if distance.ndim == 3:
        distance = distance[..., 0]
    if distance.shape != (height, width):
        raise ValueError(f"distance map shape {distance.shape} != frame {(height, width)}")
    robot_z = _range_to_z_depth(distance, float(camera["vfov"]), width, height)  # type: ignore[arg-type]
    robot_mask = np.isfinite(robot_z) & (robot_z > 0.0) & (robot_z < max_robot_range_m)

    splat_z_small = render_pointsplat_depth(
        splat,
        {**camera, "width": width, "height": height},
        depth_scale=depth_scale,
    )
    splat_z = _upsample_to(splat_z_small, height, width)
    # A splat hole (inf) cannot occlude anything: robot wins there.
    visible = robot_mask & (
        ~np.isfinite(splat_z) | (robot_z < splat_z + float(depth_margin_m))
    )

    out = scene.copy()
    out[visible] = robot[visible]
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(str(out_png))
    return {
        "out_png": str(out_png),
        "robot_pixels": int(robot_mask.sum()),
        "visible_robot_pixels": int(visible.sum()),
        "occluded_robot_pixels": int((robot_mask & ~visible).sum()),
        "depth_margin_m": float(depth_margin_m),
    }


__all__ = [
    "DEFAULT_DEPTH_MARGIN_M",
    "DEFAULT_MAX_ROBOT_RANGE_M",
    "composite_robot_into_splat",
]
