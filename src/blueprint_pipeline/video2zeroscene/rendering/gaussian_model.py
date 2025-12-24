"""3D Gaussian Splatting model for loading and managing point clouds.

This module provides the GaussianModel class that loads 3DGS PLY files in the
official INRIA format and prepares them for rendering.

PLY Format (3DGS Standard):
    - xyz: 3D position (float)
    - opacity: Gaussian opacity (float, stored as logit)
    - scale_0/1/2: Log-scale values for 3D covariance (float)
    - rot_0/1/2/3: Rotation quaternion (float)
    - f_dc_0/1/2: DC spherical harmonics coefficients (RGB)
    - f_rest_*: Higher-order SH coefficients (optional)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from plyfile import PlyData, PlyElement
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SHCoefficients:
    """Spherical Harmonics coefficients for view-dependent color.

    The 3DGS format uses SH for view-dependent appearance. The DC component
    (degree 0) provides base color, while higher degrees add view-dependency.

    Attributes:
        degree: SH degree (0-3 typically)
        dc: DC coefficients, shape (N, 3)
        rest: Higher-order coefficients, shape (N, 3, num_coeffs)
    """
    degree: int
    dc: np.ndarray  # (N, 3)
    rest: Optional[np.ndarray] = None  # (N, 3, num_coeffs)

    @property
    def num_coeffs_per_degree(self) -> List[int]:
        """Number of coefficients per SH degree."""
        return [1, 3, 5, 7]  # degrees 0, 1, 2, 3

    @property
    def total_coeffs(self) -> int:
        """Total number of SH coefficients."""
        return sum(self.num_coeffs_per_degree[: self.degree + 1])

    def get_colors(self, viewdirs: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute RGB colors from SH coefficients.

        Args:
            viewdirs: Optional view directions (N, 3). If None, returns DC color.

        Returns:
            RGB colors (N, 3) in [0, 1]
        """
        # C0 constant for SH
        C0 = 0.28209479177387814

        # DC color
        colors = self.dc * C0 + 0.5

        # Add view-dependent color from higher SH degrees (if available)
        if viewdirs is not None and self.rest is not None and self.degree > 0:
            colors = colors + self._eval_sh(viewdirs)

        # Clamp to valid range
        return np.clip(colors, 0.0, 1.0)

    def _eval_sh(self, viewdirs: np.ndarray) -> np.ndarray:
        """Evaluate spherical harmonics for view directions.

        Args:
            viewdirs: View directions (N, 3), normalized

        Returns:
            Color contribution from higher SH degrees (N, 3)
        """
        if self.rest is None:
            return np.zeros((viewdirs.shape[0], 3), dtype=np.float32)

        # SH basis function constants
        C1 = 0.4886025119029199
        C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396,
        ]
        C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435,
        ]

        x, y, z = viewdirs[:, 0], viewdirs[:, 1], viewdirs[:, 2]
        result = np.zeros((viewdirs.shape[0], 3), dtype=np.float32)

        idx = 0

        # Degree 1
        if self.degree >= 1:
            result += -C1 * y[:, None] * self.rest[:, :, idx]
            idx += 1
            result += C1 * z[:, None] * self.rest[:, :, idx]
            idx += 1
            result += -C1 * x[:, None] * self.rest[:, :, idx]
            idx += 1

        # Degree 2
        if self.degree >= 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z

            result += C2[0] * xy[:, None] * self.rest[:, :, idx]
            idx += 1
            result += C2[1] * yz[:, None] * self.rest[:, :, idx]
            idx += 1
            result += C2[2] * (2.0 * zz - xx - yy)[:, None] * self.rest[:, :, idx]
            idx += 1
            result += C2[3] * xz[:, None] * self.rest[:, :, idx]
            idx += 1
            result += C2[4] * (xx - yy)[:, None] * self.rest[:, :, idx]
            idx += 1

        # Degree 3
        if self.degree >= 3:
            result += C3[0] * y * (3 * xx - yy)[:, None] * self.rest[:, :, idx]
            idx += 1
            result += C3[1] * xy * z[:, None] * self.rest[:, :, idx]
            idx += 1
            result += C3[2] * y * (4 * zz - xx - yy)[:, None] * self.rest[:, :, idx]
            idx += 1
            result += C3[3] * z * (2 * zz - 3 * xx - 3 * yy)[:, None] * self.rest[:, :, idx]
            idx += 1
            result += C3[4] * x * (4 * zz - xx - yy)[:, None] * self.rest[:, :, idx]
            idx += 1
            result += C3[5] * z * (xx - yy)[:, None] * self.rest[:, :, idx]
            idx += 1
            result += C3[6] * x * (xx - 3 * yy)[:, None] * self.rest[:, :, idx]

        return result


class GaussianModel:
    """3D Gaussian Splatting model.

    This class loads and manages 3DGS point clouds in the official INRIA format.
    It provides efficient access to Gaussian parameters for rendering.

    Attributes:
        xyz: 3D positions (N, 3)
        opacities: Gaussian opacities in [0, 1] (N,)
        scales: 3D scales (N, 3)
        rotations: Rotation quaternions (N, 4) in (w, x, y, z) format
        sh_coeffs: Spherical harmonics coefficients
        num_gaussians: Number of Gaussians
        sh_degree: Spherical harmonics degree
    """

    def __init__(
        self,
        sh_degree: int = 3,
        device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
    ):
        """Initialize empty Gaussian model.

        Args:
            sh_degree: Maximum spherical harmonics degree (0-3)
            device: Device to use ('cuda' or 'cpu')
        """
        self.sh_degree = sh_degree
        self.device = device

        # Initialize empty arrays
        self._xyz: Optional[np.ndarray] = None
        self._opacities: Optional[np.ndarray] = None
        self._scales: Optional[np.ndarray] = None
        self._rotations: Optional[np.ndarray] = None
        self._sh_dc: Optional[np.ndarray] = None
        self._sh_rest: Optional[np.ndarray] = None

        # Cached torch tensors
        self._xyz_tensor: Optional["torch.Tensor"] = None
        self._opacities_tensor: Optional["torch.Tensor"] = None
        self._scales_tensor: Optional["torch.Tensor"] = None
        self._rotations_tensor: Optional["torch.Tensor"] = None
        self._sh_dc_tensor: Optional["torch.Tensor"] = None
        self._sh_rest_tensor: Optional["torch.Tensor"] = None

        # Computed properties
        self._covariances: Optional[np.ndarray] = None
        self._bounds_min: Optional[np.ndarray] = None
        self._bounds_max: Optional[np.ndarray] = None

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians in the model."""
        return self._xyz.shape[0] if self._xyz is not None else 0

    @property
    def xyz(self) -> np.ndarray:
        """3D positions (N, 3)."""
        return self._xyz

    @property
    def opacities(self) -> np.ndarray:
        """Gaussian opacities in [0, 1] (N,)."""
        return self._opacities

    @property
    def scales(self) -> np.ndarray:
        """3D scales (N, 3)."""
        return self._scales

    @property
    def rotations(self) -> np.ndarray:
        """Rotation quaternions (N, 4) in (w, x, y, z) format."""
        return self._rotations

    @property
    def sh_coeffs(self) -> SHCoefficients:
        """Spherical harmonics coefficients."""
        return SHCoefficients(
            degree=self.sh_degree,
            dc=self._sh_dc,
            rest=self._sh_rest,
        )

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Bounding box of Gaussian positions."""
        if self._bounds_min is None or self._bounds_max is None:
            if self._xyz is not None:
                self._bounds_min = self._xyz.min(axis=0)
                self._bounds_max = self._xyz.max(axis=0)
            else:
                self._bounds_min = np.zeros(3)
                self._bounds_max = np.zeros(3)
        return self._bounds_min, self._bounds_max

    @property
    def center(self) -> np.ndarray:
        """Center of the Gaussian cloud."""
        bounds_min, bounds_max = self.bounds
        return (bounds_min + bounds_max) / 2

    @property
    def extent(self) -> float:
        """Maximum extent of the Gaussian cloud."""
        bounds_min, bounds_max = self.bounds
        return np.max(bounds_max - bounds_min)

    def load_ply(self, ply_path: Union[str, Path]) -> "GaussianModel":
        """Load Gaussians from PLY file (official 3DGS format).

        Args:
            ply_path: Path to .ply file

        Returns:
            self for method chaining
        """
        if not PLYFILE_AVAILABLE:
            raise ImportError("plyfile is required for loading PLY files: pip install plyfile")

        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        print(f"Loading Gaussians from {ply_path}...")

        # Read PLY data
        plydata = PlyData.read(str(ply_path))
        vertex = plydata["vertex"]

        # Extract positions
        self._xyz = np.stack([
            np.asarray(vertex["x"]),
            np.asarray(vertex["y"]),
            np.asarray(vertex["z"]),
        ], axis=1).astype(np.float32)

        # Extract opacities (stored as logit, need sigmoid)
        opacities_raw = np.asarray(vertex["opacity"])
        self._opacities = self._sigmoid(opacities_raw).astype(np.float32)

        # Extract scales (stored as log, need exp)
        scales_raw = np.stack([
            np.asarray(vertex["scale_0"]),
            np.asarray(vertex["scale_1"]),
            np.asarray(vertex["scale_2"]),
        ], axis=1)
        self._scales = np.exp(scales_raw).astype(np.float32)

        # Extract rotations
        self._rotations = np.stack([
            np.asarray(vertex["rot_0"]),  # w
            np.asarray(vertex["rot_1"]),  # x
            np.asarray(vertex["rot_2"]),  # y
            np.asarray(vertex["rot_3"]),  # z
        ], axis=1).astype(np.float32)

        # Normalize rotations
        norms = np.linalg.norm(self._rotations, axis=1, keepdims=True)
        self._rotations = self._rotations / np.maximum(norms, 1e-8)

        # Extract SH coefficients
        self._sh_dc = np.stack([
            np.asarray(vertex["f_dc_0"]),
            np.asarray(vertex["f_dc_1"]),
            np.asarray(vertex["f_dc_2"]),
        ], axis=1).astype(np.float32)

        # Count SH rest coefficients
        sh_rest_names = [name for name in vertex.data.dtype.names if name.startswith("f_rest_")]
        if sh_rest_names:
            num_rest = len(sh_rest_names)
            sh_rest_flat = np.stack([
                np.asarray(vertex[name]) for name in sorted(sh_rest_names)
            ], axis=1).astype(np.float32)

            # Reshape: (N, num_rest) -> (N, 3, num_rest//3)
            # f_rest are stored interleaved: r0, g0, b0, r1, g1, b1, ...
            num_coeffs = num_rest // 3
            self._sh_rest = sh_rest_flat.reshape(-1, 3, num_coeffs)

            # Infer SH degree from number of coefficients
            # degree 1: 3 coeffs, degree 2: 3+5=8 coeffs, degree 3: 3+5+7=15 coeffs
            if num_coeffs >= 15:
                self.sh_degree = 3
            elif num_coeffs >= 8:
                self.sh_degree = 2
            elif num_coeffs >= 3:
                self.sh_degree = 1
            else:
                self.sh_degree = 0
        else:
            self._sh_rest = None
            self.sh_degree = 0

        # Reset cached values
        self._covariances = None
        self._bounds_min = None
        self._bounds_max = None
        self._clear_tensors()

        print(f"Loaded {self.num_gaussians:,} Gaussians (SH degree {self.sh_degree})")

        return self

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def _clear_tensors(self):
        """Clear cached PyTorch tensors."""
        self._xyz_tensor = None
        self._opacities_tensor = None
        self._scales_tensor = None
        self._rotations_tensor = None
        self._sh_dc_tensor = None
        self._sh_rest_tensor = None

    def to_torch(self) -> Dict[str, "torch.Tensor"]:
        """Convert all parameters to PyTorch tensors.

        Returns:
            Dictionary with tensor versions of all parameters
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for to_torch()")

        device = torch.device(self.device)

        if self._xyz_tensor is None:
            self._xyz_tensor = torch.from_numpy(self._xyz).to(device)
            self._opacities_tensor = torch.from_numpy(self._opacities).to(device)
            self._scales_tensor = torch.from_numpy(self._scales).to(device)
            self._rotations_tensor = torch.from_numpy(self._rotations).to(device)
            self._sh_dc_tensor = torch.from_numpy(self._sh_dc).to(device)
            if self._sh_rest is not None:
                self._sh_rest_tensor = torch.from_numpy(self._sh_rest).to(device)

        return {
            "xyz": self._xyz_tensor,
            "opacities": self._opacities_tensor,
            "scales": self._scales_tensor,
            "rotations": self._rotations_tensor,
            "sh_dc": self._sh_dc_tensor,
            "sh_rest": self._sh_rest_tensor,
            "sh_degree": self.sh_degree,
        }

    def get_covariance_3d(self) -> np.ndarray:
        """Compute 3D covariance matrices for all Gaussians.

        Returns:
            Covariance matrices (N, 3, 3)
        """
        if self._covariances is not None:
            return self._covariances

        # Build rotation matrices from quaternions
        R = self._quaternion_to_rotation_matrix(self._rotations)  # (N, 3, 3)

        # Build scale matrix
        S = np.zeros((self.num_gaussians, 3, 3), dtype=np.float32)
        S[:, 0, 0] = self._scales[:, 0]
        S[:, 1, 1] = self._scales[:, 1]
        S[:, 2, 2] = self._scales[:, 2]

        # Covariance = R @ S @ S^T @ R^T
        RS = np.einsum("nij,njk->nik", R, S)
        self._covariances = np.einsum("nij,nkj->nik", RS, RS)

        return self._covariances

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
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

    def get_colors(self, viewdirs: Optional[np.ndarray] = None) -> np.ndarray:
        """Get RGB colors for all Gaussians.

        Args:
            viewdirs: Optional view directions (N, 3). If None, returns DC color.

        Returns:
            RGB colors (N, 3) in [0, 1]
        """
        return self.sh_coeffs.get_colors(viewdirs)

    def filter_by_opacity(self, min_opacity: float = 0.01) -> "GaussianModel":
        """Create a new model with only Gaussians above opacity threshold.

        Args:
            min_opacity: Minimum opacity to keep

        Returns:
            New GaussianModel with filtered Gaussians
        """
        mask = self._opacities >= min_opacity

        new_model = GaussianModel(sh_degree=self.sh_degree, device=self.device)
        new_model._xyz = self._xyz[mask]
        new_model._opacities = self._opacities[mask]
        new_model._scales = self._scales[mask]
        new_model._rotations = self._rotations[mask]
        new_model._sh_dc = self._sh_dc[mask]
        if self._sh_rest is not None:
            new_model._sh_rest = self._sh_rest[mask]

        return new_model

    def filter_by_bounds(
        self,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
    ) -> "GaussianModel":
        """Create a new model with only Gaussians within bounds.

        Args:
            bounds_min: Minimum corner of bounding box
            bounds_max: Maximum corner of bounding box

        Returns:
            New GaussianModel with filtered Gaussians
        """
        mask = np.all(
            (self._xyz >= bounds_min) & (self._xyz <= bounds_max),
            axis=1
        )

        new_model = GaussianModel(sh_degree=self.sh_degree, device=self.device)
        new_model._xyz = self._xyz[mask]
        new_model._opacities = self._opacities[mask]
        new_model._scales = self._scales[mask]
        new_model._rotations = self._rotations[mask]
        new_model._sh_dc = self._sh_dc[mask]
        if self._sh_rest is not None:
            new_model._sh_rest = self._sh_rest[mask]

        return new_model

    def save_ply(self, ply_path: Union[str, Path]) -> None:
        """Save Gaussians to PLY file.

        Args:
            ply_path: Path to output .ply file
        """
        if not PLYFILE_AVAILABLE:
            raise ImportError("plyfile is required for saving PLY files: pip install plyfile")

        ply_path = Path(ply_path)
        ply_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare dtype
        dtype_list = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ]

        # Add SH rest coefficients
        if self._sh_rest is not None:
            num_rest = self._sh_rest.shape[2] * 3
            for i in range(num_rest):
                dtype_list.append((f"f_rest_{i}", "f4"))

        # Create structured array
        vertex_data = np.zeros(self.num_gaussians, dtype=dtype_list)

        # Fill positions
        vertex_data["x"] = self._xyz[:, 0]
        vertex_data["y"] = self._xyz[:, 1]
        vertex_data["z"] = self._xyz[:, 2]

        # Fill opacities (convert to logit)
        eps = 1e-5
        opacity_clamped = np.clip(self._opacities, eps, 1 - eps)
        vertex_data["opacity"] = np.log(opacity_clamped / (1 - opacity_clamped))

        # Fill scales (convert to log)
        vertex_data["scale_0"] = np.log(self._scales[:, 0])
        vertex_data["scale_1"] = np.log(self._scales[:, 1])
        vertex_data["scale_2"] = np.log(self._scales[:, 2])

        # Fill rotations
        vertex_data["rot_0"] = self._rotations[:, 0]
        vertex_data["rot_1"] = self._rotations[:, 1]
        vertex_data["rot_2"] = self._rotations[:, 2]
        vertex_data["rot_3"] = self._rotations[:, 3]

        # Fill SH DC
        vertex_data["f_dc_0"] = self._sh_dc[:, 0]
        vertex_data["f_dc_1"] = self._sh_dc[:, 1]
        vertex_data["f_dc_2"] = self._sh_dc[:, 2]

        # Fill SH rest (interleaved)
        if self._sh_rest is not None:
            for i in range(self._sh_rest.shape[2]):
                vertex_data[f"f_rest_{i*3}"] = self._sh_rest[:, 0, i]
                vertex_data[f"f_rest_{i*3+1}"] = self._sh_rest[:, 1, i]
                vertex_data[f"f_rest_{i*3+2}"] = self._sh_rest[:, 2, i]

        # Create PLY element and save
        vertex_element = PlyElement.describe(vertex_data, "vertex")
        PlyData([vertex_element], text=False).write(str(ply_path))

        print(f"Saved {self.num_gaussians:,} Gaussians to {ply_path}")

    def __repr__(self) -> str:
        return (
            f"GaussianModel("
            f"num_gaussians={self.num_gaussians:,}, "
            f"sh_degree={self.sh_degree}, "
            f"device={self.device})"
        )
