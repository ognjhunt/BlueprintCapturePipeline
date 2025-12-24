"""3D Gaussian Splatting rendering module.

This module provides a complete 3DGS renderer implementation compatible with
the official INRIA 3D Gaussian Splatting format. It can render scenes from
ZeroScene bundles to produce static-scene videos for DWM (Dexterous World Models).

Main components:
- GaussianModel: Load and manage 3DGS PLY point clouds
- GaussianRenderer: GPU-accelerated rendering using CUDA rasterization
- CameraUtils: Camera pose and intrinsics handling

Usage:
    from blueprint_pipeline.video2zeroscene.rendering import GaussianRenderer

    renderer = GaussianRenderer.from_zeroscene("output/zeroscene")
    frames = renderer.render_trajectory()
    renderer.save_video(frames, "static_scene.mp4")
"""

from .camera_utils import (
    Camera,
    CameraTrajectory,
    focal_to_fov,
    fov_to_focal,
    quaternion_to_matrix,
    matrix_to_quaternion,
    get_projection_matrix,
    get_view_matrix,
)
from .gaussian_model import GaussianModel, SHCoefficients
from .renderer import GaussianRenderer, RenderSettings

__all__ = [
    # Camera utilities
    "Camera",
    "CameraTrajectory",
    "focal_to_fov",
    "fov_to_focal",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "get_projection_matrix",
    "get_view_matrix",
    # Gaussian model
    "GaussianModel",
    "SHCoefficients",
    # Renderer
    "GaussianRenderer",
    "RenderSettings",
]
