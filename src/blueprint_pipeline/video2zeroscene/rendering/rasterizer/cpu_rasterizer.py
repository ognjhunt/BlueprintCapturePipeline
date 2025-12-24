"""CPU fallback rasterizer using pure Python/NumPy.

This rasterizer provides a reference implementation that works on any system
without GPU or CUDA dependencies. It's slower but useful for:
- Testing and debugging
- Systems without GPU
- Validation against GPU implementations

Algorithm:
    1. Project 3D Gaussians to 2D screen space
    2. Compute 2D covariance ellipses
    3. Sort Gaussians by depth (front-to-back)
    4. Alpha-blend Gaussians in sorted order
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from .base import RasterizerBackend, RasterOutput


class CPURasterizer(RasterizerBackend):
    """Pure Python/NumPy CPU rasterizer for 3DGS.

    This is a reference implementation that works without GPU dependencies.
    It's significantly slower than GPU implementations but useful for:
    - Testing on systems without CUDA
    - Debugging rendering issues
    - Small-scale rendering tasks

    Note:
        For production use with large scenes, prefer CUDARasterizer or
        GsplatRasterizer for much better performance.
    """

    def __init__(self, tile_size: int = 16, max_gaussians_per_tile: int = 1000):
        """Initialize CPU rasterizer.

        Args:
            tile_size: Tile size for tiled rendering (for memory efficiency)
            max_gaussians_per_tile: Maximum Gaussians to render per tile
        """
        self.tile_size = tile_size
        self.max_gaussians_per_tile = max_gaussians_per_tile
        self._device = "cpu"

    @property
    def name(self) -> str:
        return "cpu-numpy"

    @property
    def supports_cuda(self) -> bool:
        return False

    @property
    def device(self) -> str:
        return "cpu"

    def rasterize(
        self,
        means3D: np.ndarray,
        opacities: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        sh_dc: np.ndarray,
        sh_rest: Optional[np.ndarray],
        sh_degree: int,
        viewpoint_camera: Dict[str, Any],
        bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scaling_modifier: float = 1.0,
        compute_depth: bool = False,
    ) -> RasterOutput:
        """Rasterize 3D Gaussians using CPU.

        See base class for full documentation.
        """
        # Extract camera parameters
        R = np.asarray(viewpoint_camera["R"], dtype=np.float32)
        T = np.asarray(viewpoint_camera["T"], dtype=np.float32)
        fx = viewpoint_camera["fx"]
        fy = viewpoint_camera["fy"]
        cx = viewpoint_camera["cx"]
        cy = viewpoint_camera["cy"]
        width = viewpoint_camera["width"]
        height = viewpoint_camera["height"]
        z_near = viewpoint_camera.get("z_near", 0.01)
        z_far = viewpoint_camera.get("z_far", 100.0)

        # Initialize output buffers
        color_buffer = np.zeros((height, width, 3), dtype=np.float32)
        alpha_buffer = np.zeros((height, width), dtype=np.float32)
        depth_buffer = np.full((height, width), np.inf, dtype=np.float32) if compute_depth else None

        # Apply scaling modifier
        scales = scales * scaling_modifier

        # Step 1: Transform points to camera space
        # World to camera: p_cam = R @ p_world + T
        means_cam = (R @ means3D.T).T + T  # (N, 3)

        # Filter points behind camera
        valid_mask = means_cam[:, 2] > z_near
        if not np.any(valid_mask):
            # All points behind camera
            color_buffer[:] = bg_color
            return RasterOutput(
                color=color_buffer,
                depth=depth_buffer,
                alpha=alpha_buffer,
                radii=np.zeros(means3D.shape[0]),
                n_rendered=0,
            )

        # Step 2: Project to 2D
        z = means_cam[:, 2]
        x_proj = (fx * means_cam[:, 0] / z) + cx
        y_proj = (fy * means_cam[:, 1] / z) + cy
        means2D = np.stack([x_proj, y_proj], axis=1)

        # Step 3: Compute 2D covariance from 3D covariance
        cov2D = self._compute_cov2d(
            means_cam, scales, rotations, fx, fy, width, height
        )

        # Step 4: Compute view directions and colors
        camera_center = -R.T @ T
        viewdirs = means3D - camera_center[None, :]
        viewdirs = viewdirs / (np.linalg.norm(viewdirs, axis=1, keepdims=True) + 1e-8)
        colors = self._eval_sh(sh_dc, sh_rest, sh_degree, viewdirs)

        # Step 5: Compute radii (approximate)
        # Radius = 3 sigma where sigma = sqrt(max eigenvalue of cov2D)
        det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0]
        trace = cov2D[:, 0, 0] + cov2D[:, 1, 1]
        # Eigenvalues: lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2
        discriminant = np.maximum(trace ** 2 - 4 * det, 0)
        lambda_max = (trace + np.sqrt(discriminant)) / 2
        radii = 3.0 * np.sqrt(np.maximum(lambda_max, 1e-8))

        # Step 6: Filter visible Gaussians
        # Must be in view frustum and have valid radius
        visible_mask = (
            valid_mask &
            (x_proj >= -radii) &
            (x_proj < width + radii) &
            (y_proj >= -radii) &
            (y_proj < height + radii) &
            (radii > 0.1) &
            (radii < max(width, height))
        )

        if not np.any(visible_mask):
            color_buffer[:] = bg_color
            return RasterOutput(
                color=color_buffer,
                depth=depth_buffer,
                alpha=alpha_buffer,
                radii=radii,
                n_rendered=0,
            )

        # Get visible Gaussians
        visible_indices = np.where(visible_mask)[0]
        n_visible = len(visible_indices)

        # Sort by depth (front to back for alpha blending)
        depth_order = np.argsort(z[visible_indices])
        sorted_indices = visible_indices[depth_order]

        # Step 7: Render each Gaussian (simplified splatting)
        n_rendered = 0
        for idx in sorted_indices:
            # Skip low opacity Gaussians
            if opacities[idx] < 0.01:
                continue

            # Get Gaussian parameters
            mean = means2D[idx]
            cov = cov2D[idx]
            color = colors[idx]
            opacity = opacities[idx]
            radius = int(np.ceil(radii[idx]))

            # Compute bounding box
            x_min = max(0, int(mean[0] - radius))
            x_max = min(width, int(mean[0] + radius) + 1)
            y_min = max(0, int(mean[1] - radius))
            y_max = min(height, int(mean[1] + radius) + 1)

            if x_max <= x_min or y_max <= y_min:
                continue

            # Create coordinate grid
            yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
            coords = np.stack([xx - mean[0], yy - mean[1]], axis=-1)  # (H, W, 2)

            # Compute Gaussian weights: exp(-0.5 * d^T @ inv(cov) @ d)
            # Use inverse of 2D covariance
            det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
            if det <= 1e-8:
                continue

            inv_cov = np.array([
                [cov[1, 1], -cov[0, 1]],
                [-cov[1, 0], cov[0, 0]]
            ]) / det

            # Mahalanobis distance squared
            # d = coords @ inv_cov @ coords^T (element-wise for each pixel)
            mahal = (
                coords[:, :, 0] ** 2 * inv_cov[0, 0] +
                2 * coords[:, :, 0] * coords[:, :, 1] * inv_cov[0, 1] +
                coords[:, :, 1] ** 2 * inv_cov[1, 1]
            )

            # Gaussian weight
            weight = np.exp(-0.5 * mahal)

            # Apply opacity
            alpha = weight * opacity

            # Alpha blending: C = alpha * C_new + (1 - alpha) * C_old
            # For accumulated alpha: A = alpha + (1 - alpha) * A_old
            existing_alpha = alpha_buffer[y_min:y_max, x_min:x_max]
            blend_alpha = alpha * (1 - existing_alpha)

            color_buffer[y_min:y_max, x_min:x_max] += (
                blend_alpha[:, :, None] * color[None, None, :]
            )
            alpha_buffer[y_min:y_max, x_min:x_max] += blend_alpha

            if compute_depth:
                depth_mask = blend_alpha > 0.01
                depth_buffer[y_min:y_max, x_min:x_max] = np.where(
                    depth_mask,
                    np.minimum(depth_buffer[y_min:y_max, x_min:x_max], z[idx]),
                    depth_buffer[y_min:y_max, x_min:x_max]
                )

            n_rendered += 1

        # Apply background color
        bg = np.array(bg_color, dtype=np.float32)
        color_buffer = color_buffer + (1 - alpha_buffer[:, :, None]) * bg[None, None, :]

        # Clamp colors
        color_buffer = np.clip(color_buffer, 0, 1)

        # Fill radii for non-visible Gaussians
        full_radii = np.zeros(means3D.shape[0], dtype=np.float32)
        full_radii[visible_mask] = radii[visible_mask]

        return RasterOutput(
            color=color_buffer,
            depth=depth_buffer,
            alpha=alpha_buffer,
            radii=full_radii,
            n_rendered=n_rendered,
        )

    def _compute_cov2d(
        self,
        means_cam: np.ndarray,  # (N, 3) in camera space
        scales: np.ndarray,  # (N, 3)
        rotations: np.ndarray,  # (N, 4) quaternions
        fx: float,
        fy: float,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Compute 2D covariance matrices from 3D Gaussians.

        Projects 3D covariance to 2D using the Jacobian of the projection.

        Args:
            means_cam: Gaussian centers in camera space
            scales: 3D scales
            rotations: Rotation quaternions
            fx, fy: Focal lengths

        Returns:
            2D covariance matrices (N, 2, 2)
        """
        N = means_cam.shape[0]

        # Build rotation matrices from quaternions
        R = self._quat_to_rotmat(rotations)  # (N, 3, 3)

        # Build scale matrices
        S = np.zeros((N, 3, 3), dtype=np.float32)
        S[:, 0, 0] = scales[:, 0]
        S[:, 1, 1] = scales[:, 1]
        S[:, 2, 2] = scales[:, 2]

        # 3D covariance: Sigma = R @ S @ S^T @ R^T
        RS = np.einsum("nij,njk->nik", R, S)
        cov3D = np.einsum("nij,nkj->nik", RS, RS)

        # Jacobian of perspective projection
        z = means_cam[:, 2]
        z2 = z ** 2
        J = np.zeros((N, 2, 3), dtype=np.float32)
        J[:, 0, 0] = fx / z
        J[:, 0, 2] = -fx * means_cam[:, 0] / z2
        J[:, 1, 1] = fy / z
        J[:, 1, 2] = -fy * means_cam[:, 1] / z2

        # 2D covariance: Sigma_2D = J @ Sigma_3D @ J^T
        JC = np.einsum("nij,njk->nik", J, cov3D)
        cov2D = np.einsum("nij,nkj->nik", JC, J)

        # Add small regularization for numerical stability
        cov2D[:, 0, 0] += 0.3
        cov2D[:, 1, 1] += 0.3

        return cov2D

    def _quat_to_rotmat(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternions to rotation matrices.

        Args:
            q: Quaternions (N, 4) in (w, x, y, z) format

        Returns:
            Rotation matrices (N, 3, 3)
        """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = np.zeros((q.shape[0], 3, 3), dtype=np.float32)

        R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

        return R

    def _eval_sh(
        self,
        sh_dc: np.ndarray,
        sh_rest: Optional[np.ndarray],
        sh_degree: int,
        viewdirs: np.ndarray,
    ) -> np.ndarray:
        """Evaluate spherical harmonics to get colors.

        Args:
            sh_dc: DC SH coefficients (N, 3)
            sh_rest: Higher-order SH coefficients (N, 3, num_coeffs) or None
            sh_degree: SH degree
            viewdirs: Normalized view directions (N, 3)

        Returns:
            RGB colors (N, 3) in [0, 1]
        """
        C0 = 0.28209479177387814
        C1 = 0.4886025119029199
        C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396,
        ]

        # DC color
        colors = sh_dc * C0 + 0.5

        if sh_rest is not None and sh_degree > 0:
            x, y, z = viewdirs[:, 0], viewdirs[:, 1], viewdirs[:, 2]
            idx = 0

            # Degree 1
            if sh_degree >= 1:
                colors += -C1 * y[:, None] * sh_rest[:, :, idx]
                idx += 1
                colors += C1 * z[:, None] * sh_rest[:, :, idx]
                idx += 1
                colors += -C1 * x[:, None] * sh_rest[:, :, idx]
                idx += 1

            # Degree 2
            if sh_degree >= 2 and sh_rest.shape[2] > idx:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z

                colors += C2[0] * xy[:, None] * sh_rest[:, :, idx]
                idx += 1
                colors += C2[1] * yz[:, None] * sh_rest[:, :, idx]
                idx += 1
                colors += C2[2] * (2.0 * zz - xx - yy)[:, None] * sh_rest[:, :, idx]
                idx += 1
                colors += C2[3] * xz[:, None] * sh_rest[:, :, idx]
                idx += 1
                colors += C2[4] * (xx - yy)[:, None] * sh_rest[:, :, idx]

        return np.clip(colors, 0.0, 1.0)


def is_available() -> bool:
    """Check if CPU rasterizer is available.

    Returns:
        Always True (no special requirements)
    """
    return True
