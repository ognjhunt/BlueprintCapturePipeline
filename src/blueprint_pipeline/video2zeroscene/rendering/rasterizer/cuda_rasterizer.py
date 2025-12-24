"""CUDA-accelerated 3DGS rasterizer using diff-gaussian-rasterization.

This module wraps the official INRIA diff-gaussian-rasterization CUDA kernels
for high-performance GPU rendering of 3D Gaussian Splatting scenes.

Installation:
    # Clone and install diff-gaussian-rasterization
    git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization
    cd diff-gaussian-rasterization
    pip install .
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import RasterizerBackend, RasterOutput

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Try to import diff-gaussian-rasterization
DIFF_GAUSSIAN_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        DIFF_GAUSSIAN_AVAILABLE = True
    except ImportError:
        pass


class CUDARasterizer(RasterizerBackend):
    """GPU rasterizer using the official INRIA diff-gaussian-rasterization.

    This is the fastest and most accurate rasterizer, using the same CUDA
    kernels as the original 3DGS paper implementation.

    Requires:
        - CUDA-capable GPU
        - PyTorch with CUDA support
        - diff-gaussian-rasterization package
    """

    def __init__(self, device: str = "cuda"):
        """Initialize CUDA rasterizer.

        Args:
            device: CUDA device to use (e.g., 'cuda', 'cuda:0')

        Raises:
            ImportError: If required dependencies are not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CUDA rasterization")
        if not DIFF_GAUSSIAN_AVAILABLE:
            raise ImportError(
                "diff-gaussian-rasterization is required for CUDA rasterization. "
                "Install from: https://github.com/graphdeco-inria/diff-gaussian-rasterization"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self._device = device

    @property
    def name(self) -> str:
        return "diff-gaussian-rasterization"

    @property
    def supports_cuda(self) -> bool:
        return True

    @property
    def device(self) -> str:
        return self._device

    def to(self, device: str) -> "CUDARasterizer":
        """Move rasterizer to specified device.

        Args:
            device: Target CUDA device

        Returns:
            self for method chaining
        """
        if not device.startswith("cuda"):
            raise ValueError(f"CUDARasterizer requires CUDA device, got: {device}")
        self._device = device
        return self

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
        """Rasterize 3D Gaussians using CUDA kernels.

        See base class for full documentation.
        """
        device = torch.device(self._device)

        # Extract camera parameters
        R = viewpoint_camera["R"]
        T = viewpoint_camera["T"]
        fx = viewpoint_camera["fx"]
        fy = viewpoint_camera["fy"]
        cx = viewpoint_camera["cx"]
        cy = viewpoint_camera["cy"]
        width = viewpoint_camera["width"]
        height = viewpoint_camera["height"]
        z_near = viewpoint_camera.get("z_near", 0.01)
        z_far = viewpoint_camera.get("z_far", 100.0)

        # Compute FOV from focal length
        fov_x = 2 * math.atan(width / (2 * fx))
        fov_y = 2 * math.atan(height / (2 * fy))

        # Compute view and projection matrices
        world_view_transform = self._get_world_view_transform(R, T).to(device)
        projection_matrix = self._get_projection_matrix(
            fov_x, fov_y, z_near, z_far
        ).to(device)
        full_proj_transform = (world_view_transform @ projection_matrix).to(device)

        # Compute camera center
        camera_center = torch.from_numpy(-R.T @ T).float().to(device)

        # Convert numpy arrays to tensors
        means3D_t = torch.from_numpy(means3D).float().to(device)
        opacities_t = torch.from_numpy(opacities[:, None]).float().to(device)
        scales_t = torch.from_numpy(scales).float().to(device) * scaling_modifier
        rotations_t = torch.from_numpy(rotations).float().to(device)

        # Prepare SH coefficients
        # DC coefficients: (N, 3) -> (N, 1, 3)
        sh_dc_t = torch.from_numpy(sh_dc).float().to(device).unsqueeze(1)

        # Rest coefficients: (N, 3, num_coeffs) -> (N, num_coeffs, 3)
        if sh_rest is not None and sh_degree > 0:
            sh_rest_t = torch.from_numpy(sh_rest).float().to(device)
            # Transpose from (N, 3, num_coeffs) to (N, num_coeffs, 3)
            sh_rest_t = sh_rest_t.permute(0, 2, 1)
            shs = torch.cat([sh_dc_t, sh_rest_t], dim=1)
        else:
            shs = sh_dc_t

        # Compute 2D covariance from scales and rotations
        cov3D_precomp = None  # Let rasterizer compute from scales/rotations

        # Background color
        bg_color_t = torch.tensor(bg_color, dtype=torch.float32, device=device)

        # Create rasterization settings
        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=math.tan(fov_x * 0.5),
            tanfovy=math.tan(fov_y * 0.5),
            bg=bg_color_t,
            scale_modifier=scaling_modifier,
            viewmatrix=world_view_transform.T,
            projmatrix=full_proj_transform.T,
            sh_degree=sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
        )

        # Create rasterizer
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Render
        rendered_image, radii = rasterizer(
            means3D=means3D_t,
            means2D=torch.zeros_like(means3D_t[:, :2]),  # Will be computed
            shs=shs,
            colors_precomp=None,
            opacities=opacities_t,
            scales=scales_t,
            rotations=rotations_t,
            cov3D_precomp=cov3D_precomp,
        )

        # Convert to numpy
        color = rendered_image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        radii_np = radii.cpu().numpy()

        # Count rendered Gaussians
        n_rendered = int((radii_np > 0).sum())

        return RasterOutput(
            color=color,
            depth=None,  # TODO: Add depth computation
            alpha=None,
            radii=radii_np,
            n_rendered=n_rendered,
        )

    def _get_world_view_transform(
        self,
        R: np.ndarray,
        T: np.ndarray,
    ) -> torch.Tensor:
        """Compute world-to-view transformation matrix.

        Args:
            R: 3x3 rotation matrix
            T: 3D translation vector

        Returns:
            4x4 transformation matrix as tensor
        """
        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = R
        Rt[:3, 3] = T
        return torch.from_numpy(Rt).float()

    def _get_projection_matrix(
        self,
        fov_x: float,
        fov_y: float,
        z_near: float,
        z_far: float,
    ) -> torch.Tensor:
        """Compute OpenGL-style projection matrix.

        Args:
            fov_x: Horizontal field of view in radians
            fov_y: Vertical field of view in radians
            z_near: Near clipping plane
            z_far: Far clipping plane

        Returns:
            4x4 projection matrix as tensor
        """
        tan_half_fov_y = math.tan(fov_y / 2.0)
        tan_half_fov_x = math.tan(fov_x / 2.0)

        top = tan_half_fov_y * z_near
        bottom = -top
        right = tan_half_fov_x * z_near
        left = -right

        P = torch.zeros((4, 4), dtype=torch.float32)

        P[0, 0] = 2.0 * z_near / (right - left)
        P[1, 1] = 2.0 * z_near / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[2, 2] = z_far / (z_far - z_near)
        P[2, 3] = -(z_far * z_near) / (z_far - z_near)
        P[3, 2] = 1.0

        return P


def is_available() -> bool:
    """Check if CUDA rasterizer is available.

    Returns:
        True if all requirements are met
    """
    return (
        TORCH_AVAILABLE
        and DIFF_GAUSSIAN_AVAILABLE
        and torch.cuda.is_available()
    )
