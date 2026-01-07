"""3D Reconstruction modules for BlueprintCapturePipeline.

This package provides:
- Standalone 3D Gaussian Splatting training
- Point cloud processing utilities
- Camera model handling
- Difix3D+ scene inpainting and quality enhancement
"""

from .gaussian_splatting import (
    GaussianConfig,
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

# Difix3D+ scene inpainting (optional - requires diffusers)
try:
    from .difix_refinement import (
        DifixConfig,
        DifixPipeline,
        GapDetector,
        PoseInterpolator,
        refine_gaussians_with_difix,
    )
    DIFIX_AVAILABLE = True
except ImportError:
    DIFIX_AVAILABLE = False
    DifixConfig = None
    DifixPipeline = None
    GapDetector = None
    PoseInterpolator = None
    refine_gaussians_with_difix = None

__all__ = [
    # Gaussian Splatting
    "GaussianConfig",
    "GaussianModel",
    "GaussianTrainer",
    "train_gaussians",
    # Point Cloud
    "PointCloud",
    "load_ply",
    "save_ply",
    "initialize_from_colmap",
    # Difix3D+ (may be None if not installed)
    "DIFIX_AVAILABLE",
    "DifixConfig",
    "DifixPipeline",
    "GapDetector",
    "PoseInterpolator",
    "refine_gaussians_with_difix",
]
