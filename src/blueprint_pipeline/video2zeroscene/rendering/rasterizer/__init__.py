"""3D Gaussian Splatting rasterization backends.

This module provides different rasterization backends for rendering 3D Gaussians:

1. CUDARasterizer: GPU-accelerated rendering using diff-gaussian-rasterization
2. GsplatRasterizer: PyTorch-native GPU rendering using gsplat library
3. CPURasterizer: Pure Python/NumPy fallback for testing/debugging

Backend Selection:
    The renderer will automatically select the best available backend:
    - If CUDA available + diff-gaussian-rasterization installed -> CUDARasterizer
    - If CUDA available + gsplat installed -> GsplatRasterizer
    - Otherwise -> CPURasterizer (slow but always works)
"""

from .base import RasterizerBackend, RasterOutput
from .backend_selector import get_best_rasterizer, available_backends

__all__ = [
    "RasterizerBackend",
    "RasterOutput",
    "get_best_rasterizer",
    "available_backends",
]
