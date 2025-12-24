"""GPU rasterizer using the gsplat library.

gsplat is a PyTorch-native implementation of 3D Gaussian Splatting that
provides good performance without requiring custom CUDA compilation.

Installation:
    pip install gsplat

This is a good alternative when diff-gaussian-rasterization is not available.
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

# Try to import gsplat
GSPLAT_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        import gsplat
        from gsplat import rasterization
        GSPLAT_AVAILABLE = True
    except ImportError:
        pass


class GsplatRasterizer(RasterizerBackend):
    """GPU rasterizer using the gsplat library.

    gsplat provides PyTorch-native 3DGS rendering with good CUDA performance
    and easier installation than diff-gaussian-rasterization.

    Requires:
        - CUDA-capable GPU
        - PyTorch with CUDA support
        - gsplat package (pip install gsplat)
    """

    def __init__(self, device: str = "cuda"):
        """Initialize gsplat rasterizer.

        Args:
            device: CUDA device to use (e.g., 'cuda', 'cuda:0')

        Raises:
            ImportError: If required dependencies are not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for gsplat rasterization")
        if not GSPLAT_AVAILABLE:
            raise ImportError(
                "gsplat is required: pip install gsplat"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self._device = device

    @property
    def name(self) -> str:
        return "gsplat"

    @property
    def supports_cuda(self) -> bool:
        return True

    @property
    def device(self) -> str:
        return self._device

    def to(self, device: str) -> "GsplatRasterizer":
        """Move rasterizer to specified device."""
        if not device.startswith("cuda"):
            raise ValueError(f"GsplatRasterizer requires CUDA device, got: {device}")
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
        """Rasterize 3D Gaussians using gsplat.

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

        # Convert numpy arrays to tensors
        means = torch.from_numpy(means3D).float().to(device)
        quats = torch.from_numpy(rotations).float().to(device)  # (N, 4)
        scales_t = torch.from_numpy(scales).float().to(device) * scaling_modifier
        opacities_t = torch.from_numpy(opacities).float().to(device)

        # Compute colors from SH
        # First, we need the view direction for each Gaussian
        camera_center = -np.dot(R.T, T)
        viewdirs = means3D - camera_center[None, :]
        viewdirs = viewdirs / (np.linalg.norm(viewdirs, axis=1, keepdims=True) + 1e-8)

        # Evaluate SH to get colors
        colors = self._eval_sh(sh_dc, sh_rest, sh_degree, viewdirs)
        colors_t = torch.from_numpy(colors).float().to(device)

        # Camera matrices
        viewmat = self._get_viewmat(R, T).to(device)
        K = self._get_K(fx, fy, cx, cy, width, height).to(device)

        # Render using gsplat
        try:
            # gsplat API varies by version, try newer API first
            render_colors, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales_t,
                opacities=opacities_t,
                colors=colors_t,
                viewmats=viewmat[None],  # (1, 4, 4)
                Ks=K[None],  # (1, 3, 3)
                width=width,
                height=height,
                near_plane=z_near,
                far_plane=z_far,
                backgrounds=torch.tensor(bg_color, device=device)[None],
            )

            # Extract outputs
            color = render_colors[0].cpu().numpy()  # (H, W, 3)
            alpha = render_alphas[0].cpu().numpy() if render_alphas is not None else None

            # Get radii from info if available
            radii = info.get("radii", None)
            if radii is not None:
                radii = radii.cpu().numpy()
                n_rendered = int((radii > 0).sum())
            else:
                n_rendered = means3D.shape[0]

        except TypeError:
            # Fallback for older gsplat versions
            render_colors, render_alphas = gsplat.rasterize_gaussians(
                means3d=means,
                quats=quats,
                scales=scales_t,
                opacities=opacities_t,
                colors=colors_t,
                viewmat=viewmat,
                projmat=self._get_projmat(fx, fy, width, height, z_near, z_far).to(device),
                img_height=height,
                img_width=width,
            )

            color = render_colors.cpu().numpy()
            alpha = render_alphas.cpu().numpy() if render_alphas is not None else None
            radii = None
            n_rendered = means3D.shape[0]

        return RasterOutput(
            color=color,
            depth=None,
            alpha=alpha,
            radii=radii,
            n_rendered=n_rendered,
        )

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

        # DC color
        colors = sh_dc * C0 + 0.5

        if sh_rest is not None and sh_degree > 0:
            # Add view-dependent color from higher SH degrees
            C1 = 0.4886025119029199

            x, y, z = viewdirs[:, 0], viewdirs[:, 1], viewdirs[:, 2]
            idx = 0

            if sh_degree >= 1:
                colors += -C1 * y[:, None] * sh_rest[:, :, idx]
                idx += 1
                colors += C1 * z[:, None] * sh_rest[:, :, idx]
                idx += 1
                colors += -C1 * x[:, None] * sh_rest[:, :, idx]
                idx += 1

            # Add degree 2 and 3 if available (similar to gaussian_model.py)
            # Simplified for now - full implementation in SHCoefficients class

        return np.clip(colors, 0.0, 1.0)

    def _get_viewmat(self, R: np.ndarray, T: np.ndarray) -> torch.Tensor:
        """Create 4x4 view matrix from R and T."""
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = T
        return torch.from_numpy(viewmat).float()

    def _get_K(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Create 3x3 camera intrinsic matrix."""
        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=torch.float32)
        return K

    def _get_projmat(
        self,
        fx: float,
        fy: float,
        width: int,
        height: int,
        z_near: float,
        z_far: float,
    ) -> torch.Tensor:
        """Create 4x4 projection matrix."""
        fov_x = 2 * math.atan(width / (2 * fx))
        fov_y = 2 * math.atan(height / (2 * fy))

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
    """Check if gsplat rasterizer is available.

    Returns:
        True if all requirements are met
    """
    return (
        TORCH_AVAILABLE
        and GSPLAT_AVAILABLE
        and torch.cuda.is_available()
    )
