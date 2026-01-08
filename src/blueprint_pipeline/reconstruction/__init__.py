"""3D Reconstruction modules for BlueprintCapturePipeline.

This package provides:
- Standalone 3D Gaussian Splatting training
- Point cloud processing utilities
- Camera model handling
- Difix3D+ scene inpainting and quality enhancement
- Metric scale recovery for RGB-only captures
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

# Metric Scale Recovery (optional - requires depth models)
try:
    from .metric_scale_recovery import (
        MetricScaleConfig,
        MetricScaleRecovery,
        MetricScaleResult,
        MetricDepthModel,
        recover_metric_scale,
        SEMANTIC_SCALE_ANCHORS,
    )
    METRIC_SCALE_AVAILABLE = True
except ImportError:
    METRIC_SCALE_AVAILABLE = False
    MetricScaleConfig = None
    MetricScaleRecovery = None
    MetricScaleResult = None
    MetricDepthModel = None
    recover_metric_scale = None
    SEMANTIC_SCALE_ANCHORS = None

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
    # Metric Scale Recovery (may be None if not installed)
    "METRIC_SCALE_AVAILABLE",
    "MetricScaleConfig",
    "MetricScaleRecovery",
    "MetricScaleResult",
    "MetricDepthModel",
    "recover_metric_scale",
    "SEMANTIC_SCALE_ANCHORS",
]
