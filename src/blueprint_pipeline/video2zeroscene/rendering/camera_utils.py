"""Camera utilities for 3D Gaussian Splatting rendering.

This module provides camera handling utilities compatible with the official
INRIA 3DGS format and ZeroScene camera specifications.

Coordinate Conventions:
    - OpenCV/COLMAP: X-right, Y-down, Z-forward (optical axis)
    - World-to-camera transform convention
    - Quaternions: (w, x, y, z) format
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def quaternion_to_matrix(q: Union[List[float], np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix.

    Args:
        q: Quaternion in (w, x, y, z) format

    Returns:
        3x3 rotation matrix (numpy array)
    """
    if TORCH_AVAILABLE and isinstance(q, torch.Tensor):
        q = q.cpu().numpy()
    q = np.asarray(q, dtype=np.float64)

    # Normalize quaternion
    q = q / np.linalg.norm(q)

    w, x, y, z = q

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ], dtype=np.float64)

    return R


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion as numpy array (w, x, y, z)
    """
    R = np.asarray(R, dtype=np.float64)

    # Shepperd's method
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)

    # Ensure positive w (canonical form)
    if w < 0:
        q = -q

    return q / np.linalg.norm(q)


def focal_to_fov(focal: float, size: int) -> float:
    """Convert focal length in pixels to field of view in radians.

    Args:
        focal: Focal length in pixels
        size: Image dimension (width for FoVx, height for FoVy)

    Returns:
        Field of view in radians
    """
    return 2.0 * math.atan(size / (2.0 * focal))


def fov_to_focal(fov: float, size: int) -> float:
    """Convert field of view in radians to focal length in pixels.

    Args:
        fov: Field of view in radians
        size: Image dimension (width for FoVx, height for FoVy)

    Returns:
        Focal length in pixels
    """
    return size / (2.0 * math.tan(fov / 2.0))


def get_projection_matrix(
    fov_x: float,
    fov_y: float,
    z_near: float = 0.01,
    z_far: float = 100.0,
) -> np.ndarray:
    """Compute OpenGL-style projection matrix.

    Args:
        fov_x: Horizontal field of view in radians
        fov_y: Vertical field of view in radians
        z_near: Near clipping plane
        z_far: Far clipping plane

    Returns:
        4x4 projection matrix
    """
    tan_half_fov_y = math.tan(fov_y / 2.0)
    tan_half_fov_x = math.tan(fov_x / 2.0)

    top = tan_half_fov_y * z_near
    bottom = -top
    right = tan_half_fov_x * z_near
    left = -right

    P = np.zeros((4, 4), dtype=np.float32)

    P[0, 0] = 2.0 * z_near / (right - left)
    P[1, 1] = 2.0 * z_near / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = -(z_far + z_near) / (z_far - z_near)
    P[2, 3] = -2.0 * z_far * z_near / (z_far - z_near)
    P[3, 2] = -1.0

    return P


def get_view_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute 4x4 view matrix from rotation and translation.

    Args:
        R: 3x3 rotation matrix (world-to-camera)
        t: 3D translation vector (camera position in world or camera origin)

    Returns:
        4x4 view matrix
    """
    V = np.eye(4, dtype=np.float32)
    V[:3, :3] = R
    V[:3, 3] = t
    return V


@dataclass
class Camera:
    """Camera representation for 3DGS rendering.

    This class encapsulates camera intrinsics and extrinsics in a format
    compatible with the official 3DGS renderer.

    Attributes:
        uid: Unique identifier for this camera
        R: 3x3 rotation matrix (world-to-camera)
        T: 3D translation vector
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point
        width, height: Image dimensions
        fov_x, fov_y: Fields of view (computed from focal lengths)
        z_near, z_far: Clipping planes
    """
    uid: int
    R: np.ndarray
    T: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    z_near: float = 0.01
    z_far: float = 100.0
    timestamp: float = 0.0
    frame_id: str = ""

    def __post_init__(self):
        """Compute derived properties."""
        self.R = np.asarray(self.R, dtype=np.float32)
        self.T = np.asarray(self.T, dtype=np.float32)
        self.fov_x = focal_to_fov(self.fx, self.width)
        self.fov_y = focal_to_fov(self.fy, self.height)

    @property
    def projection_matrix(self) -> np.ndarray:
        """Get 4x4 projection matrix."""
        return get_projection_matrix(self.fov_x, self.fov_y, self.z_near, self.z_far)

    @property
    def view_matrix(self) -> np.ndarray:
        """Get 4x4 view matrix (world-to-camera transform)."""
        return get_view_matrix(self.R, self.T)

    @property
    def full_projection(self) -> np.ndarray:
        """Get combined projection * view matrix."""
        return self.projection_matrix @ self.view_matrix

    @property
    def camera_center(self) -> np.ndarray:
        """Get camera center in world coordinates."""
        # Camera center is -R^T * t
        return -self.R.T @ self.T

    def to_torch(self) -> Dict[str, "torch.Tensor"]:
        """Convert camera parameters to PyTorch tensors.

        Returns:
            Dictionary with tensor versions of all parameters
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for to_torch()")

        return {
            "R": torch.from_numpy(self.R).float(),
            "T": torch.from_numpy(self.T).float(),
            "projection_matrix": torch.from_numpy(self.projection_matrix).float(),
            "view_matrix": torch.from_numpy(self.view_matrix).float(),
            "full_projection": torch.from_numpy(self.full_projection).float(),
            "camera_center": torch.from_numpy(self.camera_center).float(),
            "fov_x": torch.tensor(self.fov_x),
            "fov_y": torch.tensor(self.fov_y),
        }

    @classmethod
    def from_pose_dict(
        cls,
        pose: Dict[str, Any],
        intrinsics: Dict[str, Any],
        uid: int = 0,
    ) -> "Camera":
        """Create Camera from ZeroScene pose and intrinsics dictionaries.

        Args:
            pose: Pose dict with 'rotation' (quat) and 'translation'
            intrinsics: Intrinsics dict with fx, fy, cx, cy, width, height
            uid: Unique identifier

        Returns:
            Camera instance
        """
        # Convert quaternion to rotation matrix
        quat = pose["rotation"]  # (w, x, y, z)
        R = quaternion_to_matrix(quat)

        # Translation
        T = np.array(pose["translation"], dtype=np.float32)

        return cls(
            uid=uid,
            R=R.astype(np.float32),
            T=T,
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
            width=intrinsics["width"],
            height=intrinsics["height"],
            timestamp=pose.get("timestamp", 0.0),
            frame_id=pose.get("frame_id", f"frame_{uid:04d}"),
        )


@dataclass
class CameraTrajectory:
    """A sequence of cameras representing a trajectory through a scene.

    This class manages a collection of Camera objects and provides utilities
    for loading from ZeroScene bundles and iterating over poses.
    """
    cameras: List[Camera] = field(default_factory=list)
    intrinsics: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, idx: int) -> Camera:
        return self.cameras[idx]

    def __iter__(self):
        return iter(self.cameras)

    @classmethod
    def from_zeroscene(cls, zeroscene_path: Union[str, Path]) -> "CameraTrajectory":
        """Load camera trajectory from ZeroScene bundle.

        Args:
            zeroscene_path: Path to zeroscene/ directory

        Returns:
            CameraTrajectory instance
        """
        zeroscene_path = Path(zeroscene_path)

        # Load intrinsics
        intrinsics_path = zeroscene_path / "camera" / "intrinsics.json"
        if not intrinsics_path.exists():
            raise FileNotFoundError(f"Camera intrinsics not found: {intrinsics_path}")
        intrinsics = json.loads(intrinsics_path.read_text())

        # Load trajectory
        trajectory_path = zeroscene_path / "camera" / "trajectory.json"
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Camera trajectory not found: {trajectory_path}")
        poses = json.loads(trajectory_path.read_text())

        # Create cameras
        cameras = [
            Camera.from_pose_dict(pose, intrinsics, uid=i)
            for i, pose in enumerate(poses)
        ]

        return cls(cameras=cameras, intrinsics=intrinsics)

    @classmethod
    def from_files(
        cls,
        intrinsics_path: Union[str, Path],
        trajectory_path: Union[str, Path],
    ) -> "CameraTrajectory":
        """Load camera trajectory from separate intrinsics and trajectory files.

        Args:
            intrinsics_path: Path to intrinsics.json
            trajectory_path: Path to trajectory.json

        Returns:
            CameraTrajectory instance
        """
        intrinsics = json.loads(Path(intrinsics_path).read_text())
        poses = json.loads(Path(trajectory_path).read_text())

        cameras = [
            Camera.from_pose_dict(pose, intrinsics, uid=i)
            for i, pose in enumerate(poses)
        ]

        return cls(cameras=cameras, intrinsics=intrinsics)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of camera positions.

        Returns:
            Tuple of (min_bounds, max_bounds) as 3D vectors
        """
        positions = np.array([cam.camera_center for cam in self.cameras])
        return positions.min(axis=0), positions.max(axis=0)

    def interpolate(self, t: float) -> Camera:
        """Interpolate camera pose at fractional time t.

        Args:
            t: Time parameter in [0, 1] spanning the trajectory

        Returns:
            Interpolated Camera
        """
        if len(self.cameras) < 2:
            return self.cameras[0] if self.cameras else None

        # Map t to camera index
        idx_float = t * (len(self.cameras) - 1)
        idx_low = int(idx_float)
        idx_high = min(idx_low + 1, len(self.cameras) - 1)
        alpha = idx_float - idx_low

        cam_low = self.cameras[idx_low]
        cam_high = self.cameras[idx_high]

        # Interpolate rotation (SLERP)
        q_low = matrix_to_quaternion(cam_low.R)
        q_high = matrix_to_quaternion(cam_high.R)
        q_interp = slerp(q_low, q_high, alpha)
        R_interp = quaternion_to_matrix(q_interp)

        # Interpolate translation (linear)
        T_interp = (1 - alpha) * cam_low.T + alpha * cam_high.T

        # Interpolate timestamp
        ts_interp = (1 - alpha) * cam_low.timestamp + alpha * cam_high.timestamp

        return Camera(
            uid=-1,  # Interpolated camera
            R=R_interp.astype(np.float32),
            T=T_interp.astype(np.float32),
            fx=cam_low.fx,
            fy=cam_low.fy,
            cx=cam_low.cx,
            cy=cam_low.cy,
            width=cam_low.width,
            height=cam_low.height,
            timestamp=ts_interp,
            frame_id=f"interp_{t:.4f}",
        )


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.

    Args:
        q0: Start quaternion (w, x, y, z)
        q1: End quaternion (w, x, y, z)
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated quaternion
    """
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)

    # Normalize
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    # Compute dot product
    dot = np.dot(q0, q1)

    # If quaternions are nearly parallel, use linear interpolation
    if abs(dot) > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    # If dot is negative, negate one quaternion to take shorter path
    if dot < 0:
        q1 = -q1
        dot = -dot

    # Clamp dot to valid range
    dot = np.clip(dot, -1.0, 1.0)

    # Compute interpolation
    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)

    return q0 * np.cos(theta) + q2 * np.sin(theta)
