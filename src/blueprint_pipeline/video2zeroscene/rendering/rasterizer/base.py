"""Base interface for 3DGS rasterization backends.

This module defines the abstract interface that all rasterization backends
must implement, along with common data structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class RasterOutput:
    """Output from 3DGS rasterization.

    Attributes:
        color: Rendered color image (H, W, 3) in [0, 1]
        depth: Optional depth map (H, W)
        alpha: Optional alpha/accumulation map (H, W)
        radii: Optional 2D radii of projected Gaussians (N,)
        n_rendered: Number of Gaussians that contributed to the image
    """
    color: np.ndarray  # (H, W, 3)
    depth: Optional[np.ndarray] = None  # (H, W)
    alpha: Optional[np.ndarray] = None  # (H, W)
    radii: Optional[np.ndarray] = None  # (N,)
    n_rendered: int = 0

    def to_uint8(self) -> np.ndarray:
        """Convert color to uint8 RGB image.

        Returns:
            RGB image (H, W, 3) as uint8
        """
        return (np.clip(self.color, 0, 1) * 255).astype(np.uint8)


class RasterizerBackend(ABC):
    """Abstract base class for 3DGS rasterization backends.

    All rasterizers must implement the `rasterize` method which takes
    Gaussian parameters and camera settings to produce a rendered image.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this rasterizer backend."""
        pass

    @property
    @abstractmethod
    def supports_cuda(self) -> bool:
        """Whether this backend supports CUDA acceleration."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Current device ('cuda' or 'cpu')."""
        pass

    @abstractmethod
    def rasterize(
        self,
        # Gaussian parameters
        means3D: np.ndarray,  # (N, 3)
        opacities: np.ndarray,  # (N,)
        scales: np.ndarray,  # (N, 3)
        rotations: np.ndarray,  # (N, 4) quaternions (w, x, y, z)
        sh_dc: np.ndarray,  # (N, 3)
        sh_rest: Optional[np.ndarray],  # (N, 3, num_coeffs)
        sh_degree: int,
        # Camera parameters
        viewpoint_camera: Dict[str, Any],
        # Rendering options
        bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scaling_modifier: float = 1.0,
        compute_depth: bool = False,
    ) -> RasterOutput:
        """Rasterize 3D Gaussians to an image.

        Args:
            means3D: 3D positions of Gaussians (N, 3)
            opacities: Gaussian opacities in [0, 1] (N,)
            scales: 3D scales of Gaussians (N, 3)
            rotations: Rotation quaternions (N, 4) in (w, x, y, z) format
            sh_dc: DC spherical harmonics coefficients (N, 3)
            sh_rest: Higher-order SH coefficients (N, 3, num_coeffs) or None
            sh_degree: SH degree to use (0-3)
            viewpoint_camera: Camera parameters dict with:
                - R: 3x3 rotation matrix (world-to-camera)
                - T: 3D translation vector
                - fx, fy: Focal lengths
                - cx, cy: Principal point
                - width, height: Image dimensions
                - z_near, z_far: Clipping planes
            bg_color: Background color (R, G, B) in [0, 1]
            scaling_modifier: Multiplier for Gaussian scales
            compute_depth: Whether to compute depth map

        Returns:
            RasterOutput with rendered image and optional auxiliary outputs
        """
        pass

    def to(self, device: str) -> "RasterizerBackend":
        """Move rasterizer to specified device.

        Args:
            device: Target device ('cuda' or 'cpu')

        Returns:
            self for method chaining
        """
        return self
