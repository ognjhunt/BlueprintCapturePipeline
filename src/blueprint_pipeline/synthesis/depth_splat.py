"""Depth-based forward splatting: warp a reference frame into a target viewpoint.

This is the v1 synthesis primitive — no generative model required. Given a
reference frame, its depth map, and the source/target camera poses in site frame,
forward-splat the reference pixels into the target image plane.

Algorithm:
  For each reference pixel (u_r, v_r) with valid depth d_r:
    1. Unproject: P_cam_r = K_r⁻¹ @ [u_r, v_r, 1] * d_r
    2. World: P_world = T_world_ref @ P_cam_r
    3. Target camera: P_cam_t = T_world_target⁻¹ @ P_world
    4. Project: [u_t, v_t] = K_t @ P_cam_t[:3] / P_cam_t[2]
    5. Z-buffer: write pixel if no closer point already occupies (u_t, v_t)

Holes (target pixels with no reference coverage) are optionally filled using
nearest-neighbour propagation over a small kernel.

Expected depth format: float32 metres. If uint16 PNG (common iOS 16-bit depth),
pass depth_scale=0.001 to convert millimetres → metres.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def depth_splat(
    *,
    ref_image: np.ndarray,           # [H_r, W_r, 3] uint8 RGB
    ref_depth: np.ndarray,           # [H_r, W_r] float32, metres (after scale)
    T_world_ref: np.ndarray,         # [4, 4] reference pose in site frame
    K_ref: Dict[str, float],         # fx, fy, cx, cy
    T_world_target: np.ndarray,      # [4, 4] target pose in site frame
    K_target: Dict[str, float],      # fx, fy, cx, cy
    target_h: int,
    target_w: int,
    depth_scale: float = 1.0,        # multiply raw depth by this → metres
    min_depth_m: float = 0.1,        # discard points closer than this
    max_depth_m: float = 20.0,       # discard points farther than this
    fill_holes: bool = True,
    fill_radius: int = 3,            # neighbourhood for hole filling
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      warped_image: [target_h, target_w, 3] uint8 — splatted reference pixels
      coverage_mask: [target_h, target_w] bool — True where a reference pixel landed
    """
    depth = ref_depth.astype(np.float64) * depth_scale
    H_r, W_r = ref_image.shape[:2]

    # Pixel grid for reference image
    us_r = np.arange(W_r, dtype=np.float64)
    vs_r = np.arange(H_r, dtype=np.float64)
    us_grid, vs_grid = np.meshgrid(us_r, vs_r)  # [H_r, W_r]

    # Validity mask: only pixels with usable depth
    valid = (depth >= min_depth_m) & (depth <= max_depth_m)
    us_v = us_grid[valid].ravel()    # [N]
    vs_v = vs_grid[valid].ravel()    # [N]
    d_v = depth[valid].ravel()       # [N]
    colors_v = ref_image[valid]      # [N, 3]

    if us_v.size == 0:
        return (
            np.zeros((target_h, target_w, 3), dtype=np.uint8),
            np.zeros((target_h, target_w), dtype=bool),
        )

    # Unproject to reference camera frame: [N, 3]
    fx_r, fy_r = K_ref["fx"], K_ref["fy"]
    cx_r, cy_r = K_ref["cx"], K_ref["cy"]
    X_r = (us_v - cx_r) / fx_r * d_v
    Y_r = (vs_v - cy_r) / fy_r * d_v
    Z_r = d_v
    pts_cam_r = np.stack([X_r, Y_r, Z_r, np.ones_like(Z_r)], axis=1)  # [N, 4]

    # Transform: ref camera → world
    pts_world = (T_world_ref @ pts_cam_r.T).T  # [N, 4]

    # Transform: world → target camera
    T_cam_target_from_world = _mat_inv(T_world_target)
    pts_cam_t = (T_cam_target_from_world @ pts_world.T).T  # [N, 4]

    # Keep only points in front of target camera
    z_t = pts_cam_t[:, 2]
    front = z_t > min_depth_m
    pts_cam_t = pts_cam_t[front]
    colors_v = colors_v[front]
    z_t = z_t[front]

    # Project to target image plane
    fx_t, fy_t = K_target["fx"], K_target["fy"]
    cx_t, cy_t = K_target["cx"], K_target["cy"]
    u_t = pts_cam_t[:, 0] / z_t * fx_t + cx_t
    v_t = pts_cam_t[:, 1] / z_t * fy_t + cy_t

    # Round to pixel indices
    u_ti = np.round(u_t).astype(np.int32)
    v_ti = np.round(v_t).astype(np.int32)

    # Filter to in-bounds pixels
    in_bounds = (
        (u_ti >= 0) & (u_ti < target_w) &
        (v_ti >= 0) & (v_ti < target_h)
    )
    u_ti = u_ti[in_bounds]
    v_ti = v_ti[in_bounds]
    z_valid = z_t[in_bounds]
    colors_valid = colors_v[in_bounds]

    # Z-buffer: sort far-to-near so nearer points overwrite farther ones
    order = np.argsort(z_valid)[::-1]
    u_ti = u_ti[order]
    v_ti = v_ti[order]
    colors_valid = colors_valid[order]

    # Write to output
    warped = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    mask = np.zeros((target_h, target_w), dtype=bool)
    warped[v_ti, u_ti] = colors_valid
    mask[v_ti, u_ti] = True

    if fill_holes and not mask.all():
        warped, mask = _fill_holes(warped, mask, radius=fill_radius)

    return warped, mask


def load_depth_png(path: Path, depth_scale: float = 0.001) -> np.ndarray:
    """
    Load an ARKit depth PNG and return float32 metres.

    ARKit saves LiDAR depth as a 16-bit grayscale PNG where value = depth in
    millimetres (0 = invalid). depth_scale=0.001 converts mm → m.

    If the PNG is 8-bit (some fallback encodings), interpret as normalised [0, 1]
    scaled by a 10m range — this is a heuristic fallback only.
    """
    from PIL import Image
    img = Image.open(path)
    arr = np.array(img)
    if arr.dtype == np.uint16:
        depth = arr.astype(np.float32) * float(depth_scale)
        depth[arr == 0] = 0.0  # mark invalid as 0
        return depth
    if arr.dtype == np.uint8:
        # Heuristic: treat as 0-10m range normalised to 0-255
        depth = arr.astype(np.float32) / 255.0 * 10.0
        depth[arr == 0] = 0.0
        return depth
    # Float32 raw — assume already metres
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Hole filling
# ---------------------------------------------------------------------------


def _fill_holes(
    image: np.ndarray,   # [H, W, 3]
    mask: np.ndarray,    # [H, W] bool
    radius: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple nearest-neighbour hole filling.
    For each uncovered pixel, copy the colour of the nearest covered pixel within radius.
    Falls back to OpenCV inpaint if available; otherwise uses scipy distance transform.
    """
    try:
        import cv2
        inpaint_mask = (~mask).astype(np.uint8) * 255
        filled = cv2.inpaint(image, inpaint_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        new_mask = np.ones_like(mask)
        return filled, new_mask
    except ImportError:
        pass

    # Fallback: scipy distance transform nearest-neighbour
    try:
        from scipy.ndimage import distance_transform_edt
        # For each empty pixel, find the nearest filled pixel and copy its colour
        filled = image.copy()
        _, nearest_idx = distance_transform_edt(~mask, return_indices=True)
        for c in range(3):
            filled[:, :, c] = image[nearest_idx[0], nearest_idx[1], c]
        # Only fill where original was empty (don't overwrite valid pixels)
        filled[mask] = image[mask]
        return filled, np.ones_like(mask)
    except ImportError:
        pass

    # Final fallback: no hole filling
    return image, mask


# ---------------------------------------------------------------------------
# SE(3) utilities
# ---------------------------------------------------------------------------


def _mat_inv(T: np.ndarray) -> np.ndarray:
    """Efficient SE(3) inverse."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ t)
    return T_inv
