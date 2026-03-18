"""6-channel Plücker ray map computation.

Plücker coordinates represent a 3D ray as a (direction, moment) pair:
  d = unit direction vector in world/site frame (3 channels)
  m = O × d  where O is the camera centre in world frame (3 channels)

These are the camera-pose embeddings used by SWM (arXiv:2603.15583) and similar
retrieval-augmented world models. They give the generation model SE(3)-aware
knowledge of where each pixel's ray is in 3D space — no positional encoding needed.

Usage:
  plucker = compute_plucker_map(T_world_camera=T, intrinsics=K, height=H, width=W)
  # plucker: [6, H, W] float32

The map is computed at the requested output resolution (H, W). If H/W differ from
the intrinsic image size, the intrinsics are scaled accordingly.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np


def compute_plucker_map(
    *,
    T_world_camera: np.ndarray,      # [4, 4] SE(3), maps camera → world/site frame
    intrinsics: Dict[str, float],    # fx, fy, cx, cy, width, height
    height: int,
    width: int,
) -> np.ndarray:
    """
    Returns a [6, H, W] float32 Plücker ray map for the given camera pose.

    Channels [0:3] = ray direction d (unit vector in world frame)
    Channels [3:6] = ray moment m = O × d (world frame)

    If the requested (height, width) differs from intrinsics (width, height), the
    focal lengths and principal point are scaled proportionally.
    """
    T = np.asarray(T_world_camera, dtype=np.float64)
    R = T[:3, :3]   # rotation: camera → world
    O = T[:3, 3]    # camera centre in world frame

    # Scale intrinsics if rendering at different resolution than capture
    intr_w = float(intrinsics.get("width") or width)
    intr_h = float(intrinsics.get("height") or height)
    sx = width / intr_w if intr_w > 0 else 1.0
    sy = height / intr_h if intr_h > 0 else 1.0

    fx = float(intrinsics["fx"]) * sx
    fy = float(intrinsics["fy"]) * sy
    cx = float(intrinsics["cx"]) * sx
    cy = float(intrinsics["cy"]) * sy

    # Pixel grid (centre of each pixel)
    us = np.arange(width, dtype=np.float64)    # [W]
    vs = np.arange(height, dtype=np.float64)   # [H]
    us_grid, vs_grid = np.meshgrid(us, vs)      # [H, W]

    # Unproject: normalised image-plane coordinates
    x_norm = (us_grid - cx) / fx   # [H, W]
    y_norm = (vs_grid - cy) / fy   # [H, W]
    ones = np.ones_like(x_norm)    # [H, W]

    # Ray directions in camera frame: [H, W, 3]
    rays_cam = np.stack([x_norm, y_norm, ones], axis=-1)   # [H, W, 3]

    # Transform to world frame: d_world = R @ r_cam  (per-pixel)
    # Efficient: rays_cam @ R.T  →  [H, W, 3]
    rays_world = rays_cam @ R.T                            # [H, W, 3]

    # Normalise to unit vectors
    norms = np.linalg.norm(rays_world, axis=-1, keepdims=True)  # [H, W, 1]
    d = rays_world / np.maximum(norms, 1e-8)               # [H, W, 3]

    # Moment: m = O × d for each pixel
    # O: [3]  →  broadcast as [1, 1, 3]
    m = np.cross(O[np.newaxis, np.newaxis, :], d)          # [H, W, 3]

    # Stack and transpose to [6, H, W]
    plucker = np.concatenate([d, m], axis=-1)              # [H, W, 6]
    return plucker.transpose(2, 0, 1).astype(np.float32)   # [6, H, W]


def plucker_to_tensor(plucker_map: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
    """Convert [6, H, W] numpy map to a [1, 6, H, W] torch tensor for model input."""
    import torch
    return torch.from_numpy(plucker_map).unsqueeze(0)  # [1, 6, H, W]


def normalise_plucker(plucker_map: np.ndarray) -> np.ndarray:
    """
    Normalise Plücker maps to approximately [-1, 1] for stable model input.
    Scales d channels by 1 (already unit vectors) and m channels by 1/scene_scale.

    scene_scale is estimated as the 90th-percentile magnitude of moment vectors,
    which is proportional to the distance from world origin to camera centre.
    """
    d = plucker_map[:3]   # [3, H, W]
    m = plucker_map[3:]   # [3, H, W]

    # Moment magnitudes: [H, W]
    m_mag = np.linalg.norm(m, axis=0)
    scene_scale = float(np.percentile(m_mag, 90))
    if scene_scale < 1e-4:
        scene_scale = 1.0

    normalised = np.concatenate([d, m / scene_scale], axis=0)
    return normalised.astype(np.float32)
