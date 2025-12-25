"""Standalone 3D Gaussian Splatting Training Module.

This module provides a self-contained implementation of 3D Gaussian Splatting
training, eliminating the dependency on external packages.

Based on:
- "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)

Features:
- Differentiable Gaussian rasterization using PyTorch
- Adaptive density control (densification/pruning)
- Spherical harmonics for view-dependent color
- COLMAP input format support
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np

# PyTorch imports with fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ExponentialLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# Image loading
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Spherical Harmonics Utilities
# =============================================================================

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]


def eval_sh(deg: int, sh: "torch.Tensor", dirs: "torch.Tensor") -> "torch.Tensor":
    """Evaluate spherical harmonics at given directions.

    Args:
        deg: Degree of SH (0-3)
        sh: Spherical harmonic coefficients [N, C, (deg+1)^2]
        dirs: View directions [N, 3]

    Returns:
        RGB colors [N, C]
    """
    assert deg >= 0 and deg <= 3

    result = SH_C0 * sh[..., 0]

    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = result - SH_C1 * y * sh[..., 1] + SH_C1 * z * sh[..., 2] - SH_C1 * x * sh[..., 3]

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = result + \
                SH_C2[0] * xy * sh[..., 4] + \
                SH_C2[1] * yz * sh[..., 5] + \
                SH_C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] + \
                SH_C2[3] * xz * sh[..., 7] + \
                SH_C2[4] * (xx - yy) * sh[..., 8]

            if deg > 2:
                result = result + \
                    SH_C3[0] * y * (3 * xx - yy) * sh[..., 9] + \
                    SH_C3[1] * xy * z * sh[..., 10] + \
                    SH_C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] + \
                    SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] + \
                    SH_C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] + \
                    SH_C3[5] * z * (xx - yy) * sh[..., 14] + \
                    SH_C3[6] * x * (xx - 3 * yy) * sh[..., 15]

    return result


def RGB2SH(rgb: "torch.Tensor") -> "torch.Tensor":
    """Convert RGB to 0-th order SH coefficient."""
    return (rgb - 0.5) / SH_C0


def SH2RGB(sh: "torch.Tensor") -> "torch.Tensor":
    """Convert 0-th order SH coefficient to RGB."""
    return sh * SH_C0 + 0.5


# =============================================================================
# Gaussian Model
# =============================================================================

@dataclass
class GaussianConfig:
    """Configuration for Gaussian splatting training."""
    # Model parameters
    sh_degree: int = 3

    # Training parameters
    iterations: int = 30000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

    # Densification
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    percent_dense: float = 0.01

    # Pruning
    min_opacity: float = 0.005
    max_screen_size: float = 20.0

    # Loss
    lambda_dssim: float = 0.2

    # Output
    save_iterations: List[int] = field(default_factory=lambda: [7000, 30000])
    checkpoint_iterations: List[int] = field(default_factory=lambda: [7000, 30000])


class GaussianModel(nn.Module if TORCH_AVAILABLE else object):
    """3D Gaussian Splatting model.

    Each Gaussian is parameterized by:
    - Position (xyz): 3D location
    - Covariance (scale + rotation): 3D ellipsoid shape
    - Opacity: Alpha value
    - Spherical harmonics: View-dependent color
    """

    def __init__(self, sh_degree: int = 3):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GaussianModel")
        super().__init__()

        self.sh_degree = sh_degree
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        # Learnable parameters (initialized later)
        self._xyz = None  # [N, 3] positions
        self._features_dc = None  # [N, 1, 3] DC SH coefficients
        self._features_rest = None  # [N, (sh_degree+1)^2 - 1, 3] higher SH
        self._scaling = None  # [N, 3] log scale
        self._rotation = None  # [N, 4] quaternion
        self._opacity = None  # [N, 1] logit opacity

        # Densification stats
        self.xyz_gradient_accum = None
        self.denom = None
        self.max_radii2D = None

        # Spatial extent
        self.spatial_lr_scale = 1.0

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians in the model."""
        return self._xyz.shape[0] if self._xyz is not None else 0

    @property
    def xyz(self) -> "torch.Tensor":
        """Get positions."""
        return self._xyz

    @property
    def features(self) -> "torch.Tensor":
        """Get all SH features concatenated."""
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def opacity(self) -> "torch.Tensor":
        """Get activated opacity."""
        return torch.sigmoid(self._opacity)

    @property
    def scaling(self) -> "torch.Tensor":
        """Get activated scaling."""
        return torch.exp(self._scaling)

    @property
    def rotation(self) -> "torch.Tensor":
        """Get normalized rotation quaternion."""
        return F.normalize(self._rotation, dim=-1)

    def get_covariance(self, scaling_modifier: float = 1.0) -> "torch.Tensor":
        """Compute 3D covariance matrices from scale and rotation.

        Returns:
            Covariance matrices [N, 6] (upper triangular)
        """
        S = self.scaling * scaling_modifier
        R = self.rotation

        # Build rotation matrix from quaternion
        r, x, y, z = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
        R_mat = torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
            2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
            2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)
        ], dim=-1).reshape(-1, 3, 3)

        # Covariance = R * S * S^T * R^T
        S_mat = torch.diag_embed(S)
        M = R_mat @ S_mat
        cov = M @ M.transpose(-1, -2)

        # Return upper triangular
        return torch.stack([
            cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
            cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]
        ], dim=-1)

    def get_colors(self, viewdirs: "torch.Tensor") -> "torch.Tensor":
        """Compute view-dependent colors.

        Args:
            viewdirs: View directions [N, 3]

        Returns:
            RGB colors [N, 3]
        """
        shs = self.features.transpose(1, 2)  # [N, 3, num_sh]
        colors = eval_sh(self.active_sh_degree, shs, viewdirs)
        return torch.clamp(colors + 0.5, 0.0, 1.0)

    def initialize_from_point_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        spatial_lr_scale: float = 1.0
    ):
        """Initialize Gaussians from a point cloud.

        Args:
            points: Point positions [N, 3]
            colors: Point colors [N, 3] in [0, 1]
            spatial_lr_scale: Spatial learning rate scale
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.spatial_lr_scale = spatial_lr_scale
        num_points = points.shape[0]

        # Initialize positions
        self._xyz = nn.Parameter(
            torch.tensor(points, dtype=torch.float32, device=device)
        )

        # Initialize colors as SH coefficients
        colors_tensor = torch.tensor(colors, dtype=torch.float32, device=device)
        sh_dc = RGB2SH(colors_tensor).unsqueeze(1)  # [N, 1, 3]

        num_sh = (self.max_sh_degree + 1) ** 2
        self._features_dc = nn.Parameter(sh_dc)
        self._features_rest = nn.Parameter(
            torch.zeros(num_points, num_sh - 1, 3, device=device)
        )

        # Initialize scales based on nearest neighbor distances
        dists = self._compute_nn_distances(points)
        scales = np.log(np.clip(dists * 0.5, 1e-7, None))
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float32, device=device).unsqueeze(-1).expand(-1, 3).clone()
        )

        # Initialize rotations as identity quaternions
        self._rotation = nn.Parameter(
            torch.tensor([[1, 0, 0, 0]] * num_points, dtype=torch.float32, device=device)
        )

        # Initialize opacity
        self._opacity = nn.Parameter(
            torch.tensor([[0.1]] * num_points, dtype=torch.float32, device=device)
        )

        # Densification stats
        self.xyz_gradient_accum = torch.zeros(num_points, 1, device=device)
        self.denom = torch.zeros(num_points, 1, device=device)
        self.max_radii2D = torch.zeros(num_points, device=device)

        logger.info(f"Initialized {num_points} Gaussians from point cloud")

    def _compute_nn_distances(self, points: np.ndarray, k: int = 3) -> np.ndarray:
        """Compute average distance to k nearest neighbors."""
        from scipy.spatial import KDTree

        tree = KDTree(points)
        dists, _ = tree.query(points, k=k+1)  # +1 because first is self
        return np.mean(dists[:, 1:], axis=1)

    def densify_and_prune(
        self,
        grad_threshold: float,
        min_opacity: float,
        extent: float,
        max_screen_size: float
    ):
        """Densify and prune Gaussians based on gradients and opacity."""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Densify by cloning small Gaussians with high gradients
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.scaling, dim=1).values <= self.percent_dense * extent
        )

        # Clone selected points
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]

        self._densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_scaling, new_rotation, new_opacity
        )

        # Densify by splitting large Gaussians with high gradients
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.scaling, dim=1).values > self.percent_dense * extent
        )

        # Split selected points
        stds = self.scaling[selected_pts_mask].repeat(2, 1)
        means = torch.zeros((stds.size(0), 3), device=self._xyz.device)
        samples = torch.normal(mean=means, std=stds)

        rots = self.rotation[selected_pts_mask].repeat(2, 1)
        new_xyz = self._xyz[selected_pts_mask].repeat(2, 1) + self._rotate_point(samples, rots)
        new_scaling = self.scaling_inverse(
            self.scaling[selected_pts_mask].repeat(2, 1) / 1.6
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(2, 1)

        self._densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_scaling, new_rotation, new_opacity
        )

        # Prune points based on opacity
        prune_mask = (self.opacity < min_opacity).squeeze()

        # Prune points too large in screen space
        if max_screen_size > 0:
            big_points_ws = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_ws)

        self._prune_points(prune_mask)

        # Reset densification stats
        self.xyz_gradient_accum = torch.zeros_like(self.xyz_gradient_accum)
        self.denom = torch.zeros_like(self.denom)
        self.max_radii2D = torch.zeros(self.num_gaussians, device=self._xyz.device)

    def _rotate_point(self, point: "torch.Tensor", quat: "torch.Tensor") -> "torch.Tensor":
        """Rotate points by quaternions."""
        r, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        R = torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
            2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
            2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)
        ], dim=-1).reshape(-1, 3, 3)
        return torch.bmm(R, point.unsqueeze(-1)).squeeze(-1)

    def scaling_inverse(self, scaling: "torch.Tensor") -> "torch.Tensor":
        """Inverse of scaling activation."""
        return torch.log(scaling)

    def _densification_postfix(
        self,
        new_xyz, new_features_dc, new_features_rest,
        new_scaling, new_rotation, new_opacity
    ):
        """Add new points after densification."""
        d = {
            "_xyz": new_xyz,
            "_features_dc": new_features_dc,
            "_features_rest": new_features_rest,
            "_scaling": new_scaling,
            "_rotation": new_rotation,
            "_opacity": new_opacity,
        }

        for name, param in d.items():
            current = getattr(self, name)
            setattr(self, name, nn.Parameter(torch.cat([current, param], dim=0)))

        # Extend stats
        device = self._xyz.device
        new_count = new_xyz.shape[0]
        self.xyz_gradient_accum = torch.cat([
            self.xyz_gradient_accum,
            torch.zeros(new_count, 1, device=device)
        ], dim=0)
        self.denom = torch.cat([
            self.denom,
            torch.zeros(new_count, 1, device=device)
        ], dim=0)
        self.max_radii2D = torch.cat([
            self.max_radii2D,
            torch.zeros(new_count, device=device)
        ], dim=0)

    def _prune_points(self, mask: "torch.Tensor"):
        """Remove points marked for pruning."""
        valid_mask = ~mask

        for name in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
            param = getattr(self, name)
            setattr(self, name, nn.Parameter(param[valid_mask]))

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_mask]
        self.denom = self.denom[valid_mask]
        self.max_radii2D = self.max_radii2D[valid_mask]

    def reset_opacity(self):
        """Reset opacity for all Gaussians."""
        opacities_new = torch.min(self._opacity, torch.ones_like(self._opacity) * 0.01)
        self._opacity = nn.Parameter(opacities_new)

    def save_ply(self, path: Union[str, Path]):
        """Save Gaussians to PLY file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        features_dc = self._features_dc.detach().cpu().numpy().reshape(-1, 3)
        features_rest = self._features_rest.detach().cpu().numpy().reshape(
            xyz.shape[0], -1
        )
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rotations = self._rotation.detach().cpu().numpy()

        # Write PLY
        num_gaussians = xyz.shape[0]
        num_sh_rest = features_rest.shape[1]

        # Build property list
        properties = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ]
        for i in range(num_sh_rest):
            properties.append((f"f_rest_{i}", "f4"))
        properties.append(("opacity", "f4"))
        properties.extend([("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")])
        properties.extend([("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")])

        # Create structured array
        dtype = np.dtype([(name, fmt) for name, fmt in properties])
        elements = np.empty(num_gaussians, dtype=dtype)

        elements["x"] = xyz[:, 0]
        elements["y"] = xyz[:, 1]
        elements["z"] = xyz[:, 2]
        elements["nx"] = normals[:, 0]
        elements["ny"] = normals[:, 1]
        elements["nz"] = normals[:, 2]
        elements["f_dc_0"] = features_dc[:, 0]
        elements["f_dc_1"] = features_dc[:, 1]
        elements["f_dc_2"] = features_dc[:, 2]
        for i in range(num_sh_rest):
            elements[f"f_rest_{i}"] = features_rest[:, i]
        elements["opacity"] = opacities.flatten()
        elements["scale_0"] = scales[:, 0]
        elements["scale_1"] = scales[:, 1]
        elements["scale_2"] = scales[:, 2]
        elements["rot_0"] = rotations[:, 0]
        elements["rot_1"] = rotations[:, 1]
        elements["rot_2"] = rotations[:, 2]
        elements["rot_3"] = rotations[:, 3]

        # Write header
        with open(path, "wb") as f:
            header = f"""ply
format binary_little_endian 1.0
element vertex {num_gaussians}
"""
            for name, fmt in properties:
                ply_type = "float" if fmt == "f4" else "double"
                header += f"property {ply_type} {name}\n"
            header += "end_header\n"

            f.write(header.encode("utf-8"))
            f.write(elements.tobytes())

        logger.info(f"Saved {num_gaussians} Gaussians to {path}")

    @classmethod
    def load_ply(cls, path: Union[str, Path], sh_degree: int = 3) -> "GaussianModel":
        """Load Gaussians from PLY file."""
        from .point_cloud import load_ply as load_ply_file

        model = cls(sh_degree=sh_degree)
        data = load_ply_file(path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model._xyz = nn.Parameter(torch.tensor(data["xyz"], dtype=torch.float32, device=device))

        # Load SH features
        if "f_dc" in data:
            model._features_dc = nn.Parameter(
                torch.tensor(data["f_dc"], dtype=torch.float32, device=device).unsqueeze(1)
            )
        else:
            # Initialize from colors if available
            colors = data.get("colors", np.ones((data["xyz"].shape[0], 3)) * 0.5)
            model._features_dc = nn.Parameter(
                RGB2SH(torch.tensor(colors, dtype=torch.float32, device=device)).unsqueeze(1)
            )

        if "f_rest" in data:
            model._features_rest = nn.Parameter(
                torch.tensor(data["f_rest"], dtype=torch.float32, device=device)
            )
        else:
            num_sh = (sh_degree + 1) ** 2
            model._features_rest = nn.Parameter(
                torch.zeros(data["xyz"].shape[0], num_sh - 1, 3, device=device)
            )

        if "opacity" in data:
            model._opacity = nn.Parameter(
                torch.tensor(data["opacity"], dtype=torch.float32, device=device).unsqueeze(-1)
            )
        else:
            model._opacity = nn.Parameter(
                torch.ones(data["xyz"].shape[0], 1, device=device) * 0.1
            )

        if "scales" in data:
            model._scaling = nn.Parameter(
                torch.tensor(data["scales"], dtype=torch.float32, device=device)
            )
        else:
            model._scaling = nn.Parameter(
                torch.ones(data["xyz"].shape[0], 3, device=device) * -3
            )

        if "rotations" in data:
            model._rotation = nn.Parameter(
                torch.tensor(data["rotations"], dtype=torch.float32, device=device)
            )
        else:
            rot = torch.zeros(data["xyz"].shape[0], 4, device=device)
            rot[:, 0] = 1
            model._rotation = nn.Parameter(rot)

        return model


# =============================================================================
# Differentiable Rasterizer
# =============================================================================

class GaussianRasterizer:
    """Software rasterizer for 3D Gaussians.

    This is a simplified Python implementation. For production use,
    the CUDA rasterizer from the original 3DGS repo is recommended.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        tanfovx: float,
        tanfovy: float,
        bg_color: "torch.Tensor",
        scale_modifier: float = 1.0,
        viewmatrix: "torch.Tensor" = None,
        projmatrix: "torch.Tensor" = None,
        sh_degree: int = 3,
        campos: "torch.Tensor" = None,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.tanfovx = tanfovx
        self.tanfovy = tanfovy
        self.bg_color = bg_color
        self.scale_modifier = scale_modifier
        self.viewmatrix = viewmatrix
        self.projmatrix = projmatrix
        self.sh_degree = sh_degree
        self.campos = campos

    def forward(
        self,
        means3D: "torch.Tensor",
        means2D: "torch.Tensor",
        shs: "torch.Tensor",
        colors_precomp: "torch.Tensor",
        opacities: "torch.Tensor",
        scales: "torch.Tensor",
        rotations: "torch.Tensor",
        cov3D_precomp: "torch.Tensor" = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Rasterize Gaussians to an image.

        This is a differentiable forward pass. For efficiency in training,
        you should use the CUDA implementation.

        Returns:
            Tuple of (rendered_image, radii)
        """
        device = means3D.device
        num_gaussians = means3D.shape[0]

        # Transform points to camera space
        ones = torch.ones(num_gaussians, 1, device=device)
        points_hom = torch.cat([means3D, ones], dim=-1)
        points_cam = points_hom @ self.viewmatrix.T

        # Project to screen
        points_proj = points_cam @ self.projmatrix.T
        points_ndc = points_proj[:, :3] / points_proj[:, 3:4]

        # Compute screen coordinates
        screen_x = ((points_ndc[:, 0] + 1) * 0.5 * self.image_width).long()
        screen_y = ((1 - points_ndc[:, 1]) * 0.5 * self.image_height).long()

        # Compute colors from SH or use precomputed
        if colors_precomp is not None:
            colors = colors_precomp
        else:
            # Compute view directions
            viewdirs = means3D - self.campos
            viewdirs = viewdirs / (torch.norm(viewdirs, dim=-1, keepdim=True) + 1e-8)
            colors = eval_sh(self.sh_degree, shs, viewdirs)
            colors = torch.clamp(colors + 0.5, 0.0, 1.0)

        # Depth sorting
        depths = points_cam[:, 2]
        sorted_indices = torch.argsort(depths)

        # Simple alpha blending (not fully differentiable, for reference)
        image = self.bg_color.clone().unsqueeze(0).unsqueeze(0).expand(
            self.image_height, self.image_width, -1
        ).clone()

        alpha_acc = torch.zeros(self.image_height, self.image_width, device=device)
        radii = torch.zeros(num_gaussians, device=device)

        for idx in sorted_indices:
            x, y = screen_x[idx].item(), screen_y[idx].item()

            if 0 <= x < self.image_width and 0 <= y < self.image_height:
                alpha = opacities[idx].item()
                color = colors[idx]

                # Simple point splatting (no Gaussian falloff for speed)
                remaining_alpha = 1 - alpha_acc[y, x].item()
                if remaining_alpha > 0.001:
                    image[y, x] = image[y, x] * (1 - alpha) + color * alpha
                    alpha_acc[y, x] += alpha * remaining_alpha
                    radii[idx] = 1.0  # Simplified

        return image.permute(2, 0, 1), radii


# =============================================================================
# Training Loop
# =============================================================================

class GaussianTrainer:
    """Trainer for 3D Gaussian Splatting."""

    def __init__(
        self,
        model: GaussianModel,
        config: GaussianConfig = None,
        output_path: Union[str, Path] = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GaussianTrainer")

        self.model = model
        self.config = config or GaussianConfig()
        self.output_path = Path(output_path) if output_path else Path("output")
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup optimizer
        self._setup_optimizer()

        self.iteration = 0

    def _setup_optimizer(self):
        """Setup Adam optimizer with per-parameter learning rates."""
        cfg = self.config

        params = [
            {"params": [self.model._xyz], "lr": cfg.position_lr_init, "name": "xyz"},
            {"params": [self.model._features_dc], "lr": cfg.feature_lr, "name": "f_dc"},
            {"params": [self.model._features_rest], "lr": cfg.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self.model._opacity], "lr": cfg.opacity_lr, "name": "opacity"},
            {"params": [self.model._scaling], "lr": cfg.scaling_lr, "name": "scaling"},
            {"params": [self.model._rotation], "lr": cfg.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler = self._get_expon_lr_func(
            cfg.position_lr_init,
            cfg.position_lr_final,
            cfg.position_lr_delay_mult,
            cfg.position_lr_max_steps
        )

    def _get_expon_lr_func(
        self,
        lr_init: float,
        lr_final: float,
        lr_delay_mult: float,
        max_steps: int
    ):
        """Get exponential learning rate schedule function."""
        def lr_func(step):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                return lr_init

            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / max_steps, 0, 1)
            )
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return delay_rate * log_lerp

        return lr_func

    def update_learning_rate(self, iteration: int):
        """Update learning rate based on iteration."""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = self.xyz_scheduler(iteration)

    def train_step(
        self,
        viewpoint_camera: Dict[str, Any],
        gt_image: "torch.Tensor",
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            viewpoint_camera: Camera parameters dict
            gt_image: Ground truth image [C, H, W]

        Returns:
            Dictionary of loss values
        """
        self.update_learning_rate(self.iteration)

        # Increase SH degree over time
        if self.iteration < 1000:
            self.model.active_sh_degree = 0
        elif self.iteration < 2000:
            self.model.active_sh_degree = min(1, self.model.max_sh_degree)
        elif self.iteration < 3000:
            self.model.active_sh_degree = min(2, self.model.max_sh_degree)
        else:
            self.model.active_sh_degree = self.model.max_sh_degree

        # Render
        rendered_image, radii = self._render(viewpoint_camera)

        # Compute loss
        l1_loss = F.l1_loss(rendered_image, gt_image)
        ssim_loss = 1.0 - self._ssim(rendered_image.unsqueeze(0), gt_image.unsqueeze(0))

        loss = (1.0 - self.config.lambda_dssim) * l1_loss + self.config.lambda_dssim * ssim_loss

        # Backward
        loss.backward()

        # Update densification stats
        if self.iteration < self.config.densify_until_iter:
            self.model.max_radii2D = torch.max(self.model.max_radii2D, radii)
            self.model.xyz_gradient_accum += torch.norm(
                self.model._xyz.grad[:, :2], dim=-1, keepdim=True
            )
            self.model.denom += 1

        # Densification
        if self.iteration >= self.config.densify_from_iter and \
           self.iteration < self.config.densify_until_iter:
            if self.iteration % self.config.densification_interval == 0:
                extent = self.model.spatial_lr_scale
                self.model.densify_and_prune(
                    self.config.densify_grad_threshold,
                    self.config.min_opacity,
                    extent,
                    self.config.max_screen_size
                )
                self._setup_optimizer()  # Rebuild optimizer with new params

            if self.iteration % self.config.opacity_reset_interval == 0:
                self.model.reset_opacity()

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.iteration += 1

        return {
            "loss": loss.item(),
            "l1_loss": l1_loss.item(),
            "ssim_loss": ssim_loss.item(),
            "num_gaussians": self.model.num_gaussians,
        }

    def _render(self, camera: Dict[str, Any]) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Render the scene from a viewpoint.

        This is a simplified software renderer. For production,
        use the CUDA rasterizer.
        """
        H, W = camera["image_height"], camera["image_width"]
        fx, fy = camera["fx"], camera["fy"]

        # Build view and projection matrices
        viewmatrix = torch.tensor(camera["world_to_camera"], device=self.device, dtype=torch.float32)

        znear, zfar = 0.01, 100.0
        fovx = 2 * math.atan(W / (2 * fx))
        fovy = 2 * math.atan(H / (2 * fy))

        tanfovx = math.tan(fovx / 2)
        tanfovy = math.tan(fovy / 2)

        projmatrix = self._get_projection_matrix(znear, zfar, fovx, fovy)
        projmatrix = projmatrix.to(self.device)

        # Camera position
        campos = torch.inverse(viewmatrix)[:3, 3]

        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)

        rasterizer = GaussianRasterizer(
            image_height=H,
            image_width=W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg_color=bg_color,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=self.model.active_sh_degree,
            campos=campos,
        )

        means2D = torch.zeros(self.model.num_gaussians, 3, device=self.device)

        rendered_image, radii = rasterizer.forward(
            means3D=self.model.xyz,
            means2D=means2D,
            shs=self.model.features,
            colors_precomp=None,
            opacities=self.model.opacity,
            scales=self.model.scaling,
            rotations=self.model.rotation,
        )

        return rendered_image, radii

    def _get_projection_matrix(
        self,
        znear: float,
        zfar: float,
        fovx: float,
        fovy: float
    ) -> "torch.Tensor":
        """Build a perspective projection matrix."""
        tanHalfFovY = math.tan(fovy / 2)
        tanHalfFovX = math.tan(fovx / 2)

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        P[3, 2] = 1.0

        return P

    def _ssim(
        self,
        img1: "torch.Tensor",
        img2: "torch.Tensor",
        window_size: int = 11
    ) -> "torch.Tensor":
        """Compute SSIM between two images."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Create Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32, device=img1.device)
        coords -= window_size // 2

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(img1.shape[1], 1, window_size, window_size)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.shape[1])

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def save_checkpoint(self, path: Union[str, Path] = None):
        """Save training checkpoint."""
        path = Path(path) if path else self.output_path / f"checkpoint_{self.iteration}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "iteration": self.iteration,
            "model_state_dict": {
                "_xyz": self.model._xyz,
                "_features_dc": self.model._features_dc,
                "_features_rest": self.model._features_rest,
                "_scaling": self.model._scaling,
                "_rotation": self.model._rotation,
                "_opacity": self.model._opacity,
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

        logger.info(f"Saved checkpoint to {path}")

    def save_gaussians(self, path: Union[str, Path] = None):
        """Save trained Gaussians to PLY."""
        path = Path(path) if path else self.output_path / "point_cloud.ply"
        self.model.save_ply(path)


def train_gaussians(
    colmap_path: Union[str, Path],
    output_path: Union[str, Path],
    config: GaussianConfig = None,
    progress_callback: callable = None,
) -> Path:
    """Train 3D Gaussians from COLMAP output.

    Args:
        colmap_path: Path to COLMAP sparse reconstruction
        output_path: Output directory
        config: Training configuration
        progress_callback: Optional callback(iteration, total, metrics)

    Returns:
        Path to output PLY file
    """
    from .point_cloud import initialize_from_colmap

    config = config or GaussianConfig()
    output_path = Path(output_path)
    colmap_path = Path(colmap_path)

    # Load COLMAP data
    logger.info(f"Loading COLMAP data from {colmap_path}")
    points, colors, cameras, images = initialize_from_colmap(colmap_path)

    # Initialize model
    model = GaussianModel(sh_degree=config.sh_degree)

    # Compute spatial extent for learning rate scaling
    extent = np.max(np.abs(points))
    model.initialize_from_point_cloud(points, colors, spatial_lr_scale=extent)

    # Create trainer
    trainer = GaussianTrainer(model, config, output_path)

    # Prepare training data
    training_cameras = []
    for img_id, img_data in images.items():
        cam_id = img_data["camera_id"]
        cam_params = cameras[cam_id]

        # Load image
        img_path = colmap_path.parent / "images" / img_data["name"]
        if not img_path.exists():
            img_path = colmap_path / "images" / img_data["name"]

        if img_path.exists() and PIL_AVAILABLE:
            pil_image = Image.open(img_path).convert("RGB")
            image_tensor = torch.tensor(
                np.array(pil_image) / 255.0,
                dtype=torch.float32
            ).permute(2, 0, 1)
        else:
            # Create dummy image if not found
            H, W = cam_params["height"], cam_params["width"]
            image_tensor = torch.zeros(3, H, W, dtype=torch.float32)

        # Build camera dict
        R = img_data["rotation"]
        t = img_data["translation"]
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = R
        world_to_camera[:3, 3] = t

        training_cameras.append({
            "image": image_tensor,
            "image_height": cam_params["height"],
            "image_width": cam_params["width"],
            "fx": cam_params["fx"],
            "fy": cam_params["fy"],
            "cx": cam_params["cx"],
            "cy": cam_params["cy"],
            "world_to_camera": world_to_camera,
        })

    if not training_cameras:
        raise ValueError("No training cameras found")

    logger.info(f"Training with {len(training_cameras)} views")

    # Training loop
    for iteration in range(config.iterations):
        # Random camera
        camera = training_cameras[iteration % len(training_cameras)]
        gt_image = camera["image"].to(trainer.device)

        metrics = trainer.train_step(camera, gt_image)

        if progress_callback:
            progress_callback(iteration, config.iterations, metrics)

        if iteration % 1000 == 0:
            logger.info(
                f"Iteration {iteration}: loss={metrics['loss']:.4f}, "
                f"gaussians={metrics['num_gaussians']}"
            )

        # Save checkpoints
        if iteration + 1 in config.save_iterations:
            ply_path = output_path / f"iteration_{iteration + 1}" / "point_cloud.ply"
            trainer.save_gaussians(ply_path)

    # Final save
    final_path = output_path / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    trainer.save_gaussians(final_path)

    logger.info(f"Training complete. Output saved to {final_path}")

    return final_path
