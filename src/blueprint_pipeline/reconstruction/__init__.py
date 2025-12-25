"""3D Reconstruction modules for BlueprintCapturePipeline.

This package provides:
- Standalone 3D Gaussian Splatting training
- Point cloud processing utilities
- Camera model handling
"""

from .gaussian_splatting import (
    GaussianModel,
    GaussianTrainer,
    train_gaussians,
)
from .point_cloud import (
    PointCloud,
    load_ply,
    save_ply,
    initialize_from_colmap,
)

__all__ = [
    "GaussianModel",
    "GaussianTrainer",
    "train_gaussians",
    "PointCloud",
    "load_ply",
    "save_ply",
    "initialize_from_colmap",
]
