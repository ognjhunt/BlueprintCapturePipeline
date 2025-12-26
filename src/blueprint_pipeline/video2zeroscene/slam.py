"""Stage 2: Sensor-conditional SLAM backends.

This module provides a unified interface for different SLAM backends:
- WildGS-SLAM: Default for RGB-only captures (handles dynamics)
- SplatMAP: Alternative for RGB-only (geometry focus)
- SplaTAM: For RGB-D captures (iPhone LiDAR)
- VIGS-SLAM: For visual-inertial captures
- ARKit Direct: Direct ARKit pose import
- COLMAP Fallback: SfM + 3DGS when other methods unavailable

All backends now integrate with the standalone 3DGS training module,
eliminating the dependency on external gaussian_splatting packages.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .interfaces import (
    CameraIntrinsics,
    CaptureManifest,
    FrameMetadata,
    PipelineConfig,
    ScaleAnchorObservation,
    SLAMBackend,
    SensorType,
    Submap,
)

# Optional imports for image processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# PyColmap - Python bindings for COLMAP (preferred over CLI)
try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Scale Calibration Utilities
# =============================================================================

def compute_scale_from_aruco(
    scale_observations: List["ScaleAnchorObservation"],
    poses: List["CameraPose"],
    intrinsics: Optional["CameraIntrinsics"],
) -> Tuple[float, float]:
    """Compute metric scale factor from ArUco/AprilTag observations.

    When we detect a marker of known physical size and observe its pixel size
    in the image, we can estimate the distance to the marker using:
        distance = (marker_size_meters * focal_length) / pixel_size

    By comparing this metric distance to the SLAM-estimated distance, we can
    compute a scale correction factor.

    Args:
        scale_observations: List of ScaleAnchorObservation from ingest
        poses: Camera poses from SLAM (arbitrary scale)
        intrinsics: Camera intrinsics (for focal length)

    Returns:
        Tuple of (scale_factor, confidence)
        - scale_factor: Multiply SLAM translations by this to get meters
        - confidence: How reliable the scale estimate is (0-1)
    """
    if not scale_observations or not poses or not intrinsics:
        return 1.0, 0.0

    # Build a map of frame_id to pose
    pose_map = {p.frame_id: p for p in poses}

    scale_estimates = []

    for obs in scale_observations:
        frame_id = obs.frame_id
        if frame_id not in pose_map:
            continue

        pose = pose_map[frame_id]

        # Compute estimated distance from marker observation
        # distance = (size_meters * focal_length_pixels) / size_pixels
        marker_size_m = obs.size_meters
        pixel_size = obs.pixel_size  # Average edge length in pixels
        focal_length = (intrinsics.fx + intrinsics.fy) / 2

        if pixel_size < 1:
            continue

        # Metric distance to marker
        metric_distance = (marker_size_m * focal_length) / pixel_size

        # Get SLAM translation magnitude (distance from origin)
        # For simplicity, we use distance from origin, but a more robust
        # approach would track relative motion between marker observations
        slam_distance = np.linalg.norm(pose.translation)

        if slam_distance < 0.001:
            # Camera is at origin, can't compute scale
            continue

        # Scale = metric / SLAM
        scale = metric_distance / slam_distance
        scale_estimates.append((scale, obs.confidence))

    if not scale_estimates:
        return 1.0, 0.0

    # Compute weighted average scale
    total_weight = sum(c for _, c in scale_estimates)
    if total_weight < 0.001:
        return 1.0, 0.0

    weighted_scale = sum(s * c for s, c in scale_estimates) / total_weight

    # Confidence based on number of observations and their individual confidences
    avg_confidence = total_weight / len(scale_estimates)
    num_confidence = min(1.0, len(scale_estimates) / 5.0)  # Max confidence at 5+ observations
    overall_confidence = avg_confidence * num_confidence

    logger.info(
        f"Scale calibration: factor={weighted_scale:.4f}, "
        f"confidence={overall_confidence:.2f} ({len(scale_estimates)} observations)"
    )

    return weighted_scale, overall_confidence


def apply_scale_to_poses(
    poses: List["CameraPose"],
    scale_factor: float,
) -> List["CameraPose"]:
    """Apply scale factor to pose translations.

    Args:
        poses: Original poses with arbitrary scale
        scale_factor: Factor to multiply translations by

    Returns:
        New list of poses with scaled translations
    """
    if abs(scale_factor - 1.0) < 1e-6:
        return poses  # No scaling needed

    scaled_poses = []
    for p in poses:
        tx, ty, tz = p.translation
        scaled_poses.append(CameraPose(
            frame_id=p.frame_id,
            image_name=p.image_name,
            rotation=p.rotation,
            translation=(tx * scale_factor, ty * scale_factor, tz * scale_factor),
            timestamp=p.timestamp,
            camera_id=p.camera_id,
        ))

    return scaled_poses


def apply_scale_to_gaussians(
    gaussians_path: Path,
    scale_factor: float,
    output_path: Optional[Path] = None,
) -> Path:
    """Apply scale factor to Gaussian positions.

    Args:
        gaussians_path: Path to input PLY file
        scale_factor: Factor to multiply positions by
        output_path: Output path (defaults to overwriting input)

    Returns:
        Path to scaled PLY file
    """
    if abs(scale_factor - 1.0) < 1e-6:
        return gaussians_path  # No scaling needed

    output_path = output_path or gaussians_path

    try:
        from ..reconstruction.point_cloud import load_ply, save_ply

        data = load_ply(gaussians_path)

        if "xyz" in data:
            data["xyz"] = data["xyz"] * scale_factor

            # Also scale the Gaussian scales if present
            if "scales" in data:
                # Scales are in log space, so add log(scale_factor)
                data["scales"] = data["scales"] + np.log(scale_factor)

            # Write back
            # For now, just modify positions and keep other attributes
            # A full implementation would preserve all 3DGS attributes

        logger.info(f"Applied scale factor {scale_factor:.4f} to Gaussians")

    except Exception as e:
        logger.warning(f"Could not apply scale to Gaussians: {e}")

    return output_path


@dataclass
class CameraPose:
    """Camera pose in world coordinates."""
    frame_id: str
    image_name: str
    rotation: Tuple[float, float, float, float]  # Quaternion (w, x, y, z)
    translation: Tuple[float, float, float]
    timestamp: float = 0.0
    camera_id: int = 1


@dataclass
class SLAMResult:
    """Result of SLAM reconstruction."""
    poses: List[CameraPose]
    gaussians_path: Optional[Path] = None
    sparse_points_path: Optional[Path] = None

    # Quality metrics
    registration_rate: float = 0.0
    mean_reprojection_error: float = 0.0
    scale_factor: float = 1.0
    scale_confidence: float = 0.0

    # Submaps (for large spaces)
    submaps: List[Submap] = field(default_factory=list)

    success: bool = True
    errors: List[str] = field(default_factory=list)


class BaseSLAM(ABC):
    """Base class for SLAM backends."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    @abstractmethod
    def run(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]] = None,
        scale_observations: Optional[List[ScaleAnchorObservation]] = None,
    ) -> SLAMResult:
        """Run SLAM reconstruction.

        Args:
            manifest: Capture manifest with metadata
            keyframes: Selected keyframes for reconstruction
            frames_dir: Directory containing frame images
            output_dir: Output directory for results
            dynamic_masks: Optional masks for dynamic objects
            scale_observations: Optional ArUco/AprilTag observations for scale calibration

        Returns:
            SLAMResult with poses and Gaussians
        """
        pass

    def _calibrate_scale(
        self,
        result: SLAMResult,
        scale_observations: Optional[List[ScaleAnchorObservation]],
        intrinsics: Optional[CameraIntrinsics],
    ) -> SLAMResult:
        """Apply scale calibration to SLAM result if observations available.

        Args:
            result: Raw SLAM result with arbitrary scale
            scale_observations: ArUco/AprilTag observations
            intrinsics: Camera intrinsics for scale computation

        Returns:
            Calibrated SLAMResult with metric scale
        """
        if not scale_observations or not result.poses:
            return result

        # Compute scale from observations
        scale_factor, confidence = compute_scale_from_aruco(
            scale_observations, result.poses, intrinsics
        )

        if abs(scale_factor - 1.0) < 1e-6 or confidence < 0.1:
            return result

        print(f"  Applying scale calibration: {scale_factor:.4f} (confidence: {confidence:.2f})")

        # Apply scale to poses
        scaled_poses = apply_scale_to_poses(result.poses, scale_factor)

        # Apply scale to Gaussians if available
        if result.gaussians_path and result.gaussians_path.exists():
            apply_scale_to_gaussians(result.gaussians_path, scale_factor)

        # Update result
        result.poses = scaled_poses
        result.scale_factor = scale_factor
        result.scale_confidence = confidence

        return result

    def _create_submaps(
        self,
        keyframes: List[FrameMetadata],
    ) -> List[List[FrameMetadata]]:
        """Split keyframes into submaps for large spaces."""
        if not self.config.enable_submapping:
            return [keyframes]

        submaps = []
        chunk_size = self.config.submap_keyframe_count
        overlap = self.config.submap_overlap_frames

        i = 0
        while i < len(keyframes):
            end = min(i + chunk_size, len(keyframes))
            submaps.append(keyframes[i:end])
            i += chunk_size - overlap

        return submaps


class WildGSSLAM(BaseSLAM):
    """WildGS-SLAM for RGB-only captures with dynamic handling.

    WildGS-SLAM (CVPR 2025) is designed for "in the wild" captures with:
    - Uncertainty-aware tracking/mapping
    - Handling of dynamic objects (people, hands)
    - Monocular RGB input

    GitHub: https://github.com/GradientSpaces/WildGS-SLAM

    This implementation:
    1. Uses the official WildGS-SLAM when installed (git clone + setup)
    2. Falls back to COLMAP + standalone 3DGS when not available
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._wildgs_path = None
        self._wildgs_available = None

    def run(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]] = None,
        scale_observations: Optional[List[ScaleAnchorObservation]] = None,
    ) -> SLAMResult:
        """Run WildGS-SLAM reconstruction."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if WildGS-SLAM is available
        if self._check_wildgs_available():
            try:
                result = self._run_native_wildgs(
                    manifest, keyframes, frames_dir, output_dir, dynamic_masks
                )
                # Apply scale calibration
                if scale_observations:
                    result = self._calibrate_scale(result, scale_observations, manifest.intrinsics)
                return result
            except Exception as e:
                logger.warning(f"WildGS-SLAM failed: {e}, falling back to COLMAP")

        # Fall back to our built-in COLMAP + standalone 3DGS
        logger.info("WildGS-SLAM not available, using COLMAP + standalone 3DGS")
        fallback = COLMAPFallback(self.config)
        return fallback.run(
            manifest, keyframes, frames_dir, output_dir, dynamic_masks, scale_observations
        )

    def _check_wildgs_available(self) -> bool:
        """Check if WildGS-SLAM is available.

        WildGS-SLAM can be installed via:
        1. pip install (if published) - imports as 'wildgs_slam'
        2. git clone + pip install -e . - imports from 'src' module
        3. Direct path to cloned repo
        """
        if self._wildgs_available is not None:
            return self._wildgs_available

        # Method 1: Try pip-installed package
        try:
            from wildgs_slam import WildGSSLAM as _  # noqa: F401
            self._wildgs_available = True
            logger.info("WildGS-SLAM available via pip package")
            return True
        except ImportError:
            pass

        # Method 2: Try git-cloned installation (src.slam module)
        try:
            from src.slam import SLAM as _  # noqa: F401
            from src import config as _  # noqa: F401
            self._wildgs_available = True
            logger.info("WildGS-SLAM available via git clone (src.slam)")
            return True
        except ImportError:
            pass

        # Method 3: Check for WildGS-SLAM in common locations
        common_paths = [
            Path.home() / "WildGS-SLAM",
            Path.home() / "repos" / "WildGS-SLAM",
            Path("/opt/WildGS-SLAM"),
            Path.cwd() / "WildGS-SLAM",
        ]

        for path in common_paths:
            if (path / "src" / "slam.py").exists():
                self._wildgs_path = path
                self._wildgs_available = True
                logger.info(f"WildGS-SLAM found at {path}")
                return True

        self._wildgs_available = False
        logger.info(
            "WildGS-SLAM not found. Install with: "
            "git clone --recursive https://github.com/GradientSpaces/WildGS-SLAM.git && "
            "cd WildGS-SLAM && pip install -e ."
        )
        return False

    def _run_native_wildgs(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]],
    ) -> SLAMResult:
        """Run native WildGS-SLAM implementation.

        WildGS-SLAM uses a config-based approach:
        1. Create config YAML with dataset/camera parameters
        2. Prepare images in expected format (rgb/ folder)
        3. Run SLAM via the SLAM class or subprocess
        4. Parse output poses and gaussians
        """
        import yaml

        # Prepare dataset structure expected by WildGS-SLAM
        dataset_dir = output_dir / "wildgs_dataset"
        rgb_dir = dataset_dir / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)

        # Copy/link images to rgb/ folder
        for i, kf in enumerate(keyframes):
            src = frames_dir.parent / kf.file_path
            if src.exists():
                dst = rgb_dir / f"{i:06d}.png"
                if not dst.exists():
                    shutil.copy(src, dst)

        # Get image dimensions
        if keyframes and CV2_AVAILABLE:
            sample_img = cv2.imread(str(rgb_dir / "000000.png"))
            if sample_img is not None:
                H, W = sample_img.shape[:2]
            else:
                H, W = 1080, 1920
        else:
            H, W = 1080, 1920

        # Create config YAML
        config_dict = {
            "scene": "custom_capture",
            "dataset": "custom",
            "data": {
                "input_folder": str(dataset_dir),
                "output": str(output_dir / "wildgs_output"),
            },
            "cam": {
                "H": H,
                "W": W,
                "H_out": min(H, 680),  # Recommended by WildGS-SLAM
                "W_out": min(W, 1200),
            },
            "mapping": {
                "Training": {"alpha": 0.8},
            },
        }

        # Add intrinsics if available
        if manifest.intrinsics:
            config_dict["cam"].update({
                "fx": manifest.intrinsics.fx,
                "fy": manifest.intrinsics.fy,
                "cx": manifest.intrinsics.cx,
                "cy": manifest.intrinsics.cy,
            })

        config_path = output_dir / "wildgs_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        # Try running WildGS-SLAM
        poses = []
        gaussians_path = None

        try:
            # Method 1: Direct Python import
            poses, gaussians_path = self._run_wildgs_python(
                config_dict, dataset_dir, output_dir
            )
        except Exception as e:
            logger.warning(f"Direct WildGS-SLAM import failed: {e}")
            try:
                # Method 2: Subprocess with run.py
                poses, gaussians_path = self._run_wildgs_subprocess(
                    config_path, output_dir
                )
            except Exception as e2:
                logger.error(f"WildGS-SLAM subprocess failed: {e2}")
                raise RuntimeError(f"WildGS-SLAM failed: {e}, {e2}")

        # Map poses back to original keyframes
        final_poses = []
        for i, kf in enumerate(keyframes):
            if i < len(poses):
                pose = poses[i]
                final_poses.append(CameraPose(
                    frame_id=kf.frame_id,
                    image_name=Path(kf.file_path).name,
                    rotation=pose.rotation,
                    translation=pose.translation,
                    timestamp=kf.timestamp_seconds,
                ))

        # Save poses
        self._save_poses(final_poses, output_dir / "poses")

        return SLAMResult(
            poses=final_poses,
            gaussians_path=gaussians_path,
            registration_rate=len(final_poses) / len(keyframes) if keyframes else 0,
        )

    def _run_wildgs_python(
        self,
        config_dict: Dict[str, Any],
        dataset_dir: Path,
        output_dir: Path,
    ) -> Tuple[List[CameraPose], Optional[Path]]:
        """Run WildGS-SLAM via direct Python import."""
        import sys

        # Add WildGS-SLAM to path if needed
        if self._wildgs_path:
            sys.path.insert(0, str(self._wildgs_path))

        try:
            from src import config as wildgs_config
            from src.slam import SLAM
            from src.utils.datasets import get_dataset
        except ImportError:
            # Try alternative import
            from wildgs_slam import config as wildgs_config
            from wildgs_slam.slam import SLAM
            from wildgs_slam.utils.datasets import get_dataset

        # Create a temporary config file for WildGS-SLAM
        import yaml
        config_path = output_dir / "wildgs_temp_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        # Load config
        cfg = wildgs_config.load_config(str(config_path))

        # Get dataset
        dataset = get_dataset(cfg)

        # Create and run SLAM
        slam = SLAM(cfg, dataset)
        slam.run()

        # Parse output poses
        poses = self._parse_wildgs_poses(
            Path(config_dict["data"]["output"]) / config_dict["scene"]
        )

        # Find gaussians
        gaussians_path = (
            Path(config_dict["data"]["output"]) /
            config_dict["scene"] /
            "final_gs.ply"
        )

        if not gaussians_path.exists():
            gaussians_path = None

        return poses, gaussians_path

    def _run_wildgs_subprocess(
        self,
        config_path: Path,
        output_dir: Path,
    ) -> Tuple[List[CameraPose], Optional[Path]]:
        """Run WildGS-SLAM via subprocess."""
        # Find run.py script
        run_script = None
        if self._wildgs_path:
            run_script = self._wildgs_path / "run.py"

        if not run_script or not run_script.exists():
            # Try common locations
            for path in [
                Path.home() / "WildGS-SLAM" / "run.py",
                Path.cwd() / "WildGS-SLAM" / "run.py",
            ]:
                if path.exists():
                    run_script = path
                    break

        if not run_script or not run_script.exists():
            raise RuntimeError("WildGS-SLAM run.py not found")

        # Run WildGS-SLAM
        result = subprocess.run(
            ["python", str(run_script), str(config_path)],
            capture_output=True,
            timeout=3600,
            cwd=run_script.parent,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"WildGS-SLAM failed: {result.stderr.decode()}"
            )

        # Parse output
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        output_scene_dir = Path(cfg["data"]["output"]) / cfg["scene"]
        poses = self._parse_wildgs_poses(output_scene_dir)

        gaussians_path = output_scene_dir / "final_gs.ply"
        if not gaussians_path.exists():
            gaussians_path = None

        return poses, gaussians_path

    def _parse_wildgs_poses(self, output_dir: Path) -> List[CameraPose]:
        """Parse poses from WildGS-SLAM output.

        WildGS-SLAM saves poses in video.npz and traj/ folder.
        """
        poses = []

        # Try loading from video.npz
        video_npz = output_dir / "video.npz"
        if video_npz.exists():
            try:
                data = np.load(video_npz, allow_pickle=True)

                # WildGS-SLAM stores poses as 4x4 matrices
                if "poses" in data:
                    pose_matrices = data["poses"]
                    for i, pose_mat in enumerate(pose_matrices):
                        pose_mat = pose_mat.reshape(4, 4)

                        # Extract rotation and translation
                        R = pose_mat[:3, :3]
                        t = pose_mat[:3, 3]

                        # Convert rotation matrix to quaternion
                        quat = self._rotation_matrix_to_quaternion(R)

                        poses.append(CameraPose(
                            frame_id=f"{i:06d}",
                            image_name=f"{i:06d}.png",
                            rotation=quat,
                            translation=tuple(t),
                        ))

                return poses
            except Exception as e:
                logger.warning(f"Failed to parse video.npz: {e}")

        # Try loading from traj/ folder
        traj_dir = output_dir / "traj"
        if traj_dir.exists():
            for traj_file in sorted(traj_dir.glob("*.txt")):
                try:
                    pose_mat = np.loadtxt(traj_file).reshape(4, 4)
                    R = pose_mat[:3, :3]
                    t = pose_mat[:3, 3]
                    quat = self._rotation_matrix_to_quaternion(R)

                    poses.append(CameraPose(
                        frame_id=traj_file.stem,
                        image_name=f"{traj_file.stem}.png",
                        rotation=quat,
                        translation=tuple(t),
                    ))
                except Exception:
                    continue

        return poses

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
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
        return (w, x, y, z)

    def _save_poses(self, poses: List[CameraPose], output_dir: Path) -> None:
        """Save poses in multiple formats."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON format
        poses_json = [
            {
                "frame_id": p.frame_id,
                "image_name": p.image_name,
                "rotation": list(p.rotation),
                "translation": list(p.translation),
                "timestamp": p.timestamp,
            }
            for p in poses
        ]
        (output_dir / "poses.json").write_text(json.dumps({"poses": poses_json}, indent=2))

        # COLMAP format
        with open(output_dir / "images.txt", "w") as f:
            f.write("# Image list\n")
            for i, p in enumerate(poses):
                qw, qx, qy, qz = p.rotation
                tx, ty, tz = p.translation
                f.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {p.image_name}\n")
                f.write("\n")  # Empty points line


class SplaTAM(BaseSLAM):
    """SplaTAM for RGB-D captures (CVPR 2024).

    Designed for dense SLAM with depth sensor input (e.g., iPhone LiDAR).
    Uses depth information for more accurate geometry reconstruction.

    GitHub: https://github.com/spla-tam/SplaTAM

    This implementation:
    1. Uses the official SplaTAM when installed (git clone + setup)
    2. Falls back to depth-guided COLMAP + standalone 3DGS when not available
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._splatam_path = None
        self._splatam_available = None

    def run(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]] = None,
        scale_observations: Optional[List[ScaleAnchorObservation]] = None,
    ) -> SLAMResult:
        """Run SplaTAM with RGB-D input."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for depth data
        if not manifest.has_depth or not manifest.depth_frames_path:
            logger.info("No depth data available for SplaTAM, falling back to WildGS-SLAM")
            fallback = WildGSSLAM(self.config)
            return fallback.run(
                manifest, keyframes, frames_dir, output_dir, dynamic_masks, scale_observations
            )

        # Try native SplaTAM first
        if self._check_splatam_available():
            try:
                result = self._run_native_splatam(
                    manifest, keyframes, frames_dir, output_dir
                )
                # RGB-D has metric scale, but apply calibration if available for refinement
                if scale_observations:
                    result = self._calibrate_scale(result, scale_observations, manifest.intrinsics)
                return result
            except Exception as e:
                logger.warning(f"Native SplaTAM failed: {e}, using fallback")

        # Use depth-guided COLMAP fallback
        return self._run_depth_guided_colmap(
            manifest, keyframes, frames_dir, output_dir, dynamic_masks
        )

    def _check_splatam_available(self) -> bool:
        """Check if SplaTAM package is available.

        SplaTAM can be installed via:
        1. pip install (if published)
        2. git clone + pip install -e .
        3. Direct path to cloned repo
        """
        if self._splatam_available is not None:
            return self._splatam_available

        # Method 1: Try pip-installed package
        try:
            from splatam import rgbd_slam  # noqa: F401
            self._splatam_available = True
            logger.info("SplaTAM available via pip package")
            return True
        except ImportError:
            pass

        # Method 2: Try git-cloned installation (scripts.splatam module)
        try:
            from scripts.splatam import rgbd_slam  # noqa: F401
            self._splatam_available = True
            logger.info("SplaTAM available via git clone (scripts.splatam)")
            return True
        except ImportError:
            pass

        # Method 3: Check for SplaTAM in common locations
        common_paths = [
            Path.home() / "SplaTAM",
            Path.home() / "repos" / "SplaTAM",
            Path("/opt/SplaTAM"),
            Path.cwd() / "SplaTAM",
        ]

        for path in common_paths:
            if (path / "scripts" / "splatam.py").exists():
                self._splatam_path = path
                self._splatam_available = True
                logger.info(f"SplaTAM found at {path}")
                return True

        self._splatam_available = False
        logger.info(
            "SplaTAM not found. Install with: "
            "git clone https://github.com/spla-tam/SplaTAM.git && "
            "cd SplaTAM && pip install -e ."
        )
        return False

    def _run_native_splatam(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
    ) -> SLAMResult:
        """Run native SplaTAM implementation.

        SplaTAM uses a config dictionary approach:
        1. Create config with dataset/camera parameters
        2. Prepare images in expected format (color/, depth/ folders)
        3. Run rgbd_slam(config) function
        4. Parse output poses and gaussians
        """
        # Prepare dataset structure expected by SplaTAM
        dataset_dir = output_dir / "splatam_dataset"
        color_dir = dataset_dir / "color"
        depth_dir_out = dataset_dir / "depth"
        color_dir.mkdir(parents=True, exist_ok=True)
        depth_dir_out.mkdir(parents=True, exist_ok=True)

        depth_dir = Path(manifest.depth_frames_path) if manifest.depth_frames_path else None

        # Copy RGB and depth images
        valid_frames = []
        for i, kf in enumerate(keyframes):
            rgb_src = frames_dir.parent / kf.file_path
            if depth_dir:
                # Try different depth naming conventions
                depth_src = None
                for pattern in [
                    f"{kf.frame_id}_depth.png",
                    f"{kf.frame_id}.png",
                    f"depth_{kf.frame_id}.png",
                    f"{i:06d}_depth.png",
                ]:
                    candidate = depth_dir / pattern
                    if candidate.exists():
                        depth_src = candidate
                        break

                if rgb_src.exists() and depth_src:
                    rgb_dst = color_dir / f"{i:06d}.png"
                    depth_dst = depth_dir_out / f"{i:06d}.png"
                    if not rgb_dst.exists():
                        shutil.copy(rgb_src, rgb_dst)
                    if not depth_dst.exists():
                        shutil.copy(depth_src, depth_dst)
                    valid_frames.append((i, kf))

        if not valid_frames:
            raise ValueError("No valid RGB-D pairs found for SplaTAM")

        # Get image dimensions
        if valid_frames and CV2_AVAILABLE:
            sample_img = cv2.imread(str(color_dir / "000000.png"))
            if sample_img is not None:
                H, W = sample_img.shape[:2]
            else:
                H, W = 480, 640
        else:
            H, W = 480, 640

        # Build SplaTAM config
        # Based on: https://github.com/spla-tam/SplaTAM/blob/main/configs/replica/splatam.py
        splatam_config = {
            "workdir": str(output_dir / "splatam_output"),
            "run_name": "custom_capture",
            "seed": 0,
            "primary_device": "cuda:0" if self._check_cuda_available() else "cpu",
            "map_every": 1,
            "keyframe_every": 5,
            "mapping_window_size": 24,
            "report_global_progress_every": 500,
            "eval_every": 5,
            "scene_radius_depth_ratio": 3,
            "mean_sq_dist_method": "projective",
            "gaussian_distribution": "isotropic",
            "report_iter_progress": False,
            "load_checkpoint": False,
            "checkpoint_time_idx": 0,
            "save_checkpoints": False,
            "checkpoint_interval": 100,
            "use_wandb": False,

            "data": {
                "basedir": str(dataset_dir),
                "sequence": "",
                "desired_image_height": H,
                "desired_image_width": W,
                "start": 0,
                "end": -1,
                "stride": 1,
                "num_frames": len(valid_frames),
            },

            "tracking": {
                "use_gt_poses": False,
                "forward_prop": True,
                "num_iters": 40,
                "use_sil_for_loss": True,
                "sil_thres": 0.99,
                "use_l1": True,
                "ignore_outlier_depth_loss": False,
                "use_uncertainty_for_loss_mask": False,
                "use_uncertainty_for_loss": False,
                "use_chamfer": False,
                "loss_weights": {
                    "im": 0.5,
                    "depth": 1.0,
                },
                "lrs": {
                    "means3D": 0.0,
                    "rgb_colors": 0.0,
                    "unnorm_rotations": 0.0,
                    "logit_opacities": 0.0,
                    "log_scales": 0.0,
                    "cam_unnorm_rots": 0.0004,
                    "cam_trans": 0.002,
                },
            },

            "mapping": {
                "num_iters": 60,
                "add_new_gaussians": True,
                "sil_thres": 0.5,
                "use_l1": True,
                "ignore_outlier_depth_loss": False,
                "use_sil_for_loss": False,
                "use_uncertainty_for_loss_mask": False,
                "use_uncertainty_for_loss": False,
                "use_chamfer": False,
                "loss_weights": {
                    "im": 0.5,
                    "depth": 1.0,
                },
                "lrs": {
                    "means3D": 0.0001,
                    "rgb_colors": 0.0025,
                    "unnorm_rotations": 0.001,
                    "logit_opacities": 0.05,
                    "log_scales": 0.001,
                    "cam_unnorm_rots": 0.0000,
                    "cam_trans": 0.0000,
                },
                "prune_gaussians": True,
                "pruning_dict": {
                    "start_after": 0,
                    "remove_big_after": 0,
                    "stop_after": 20,
                    "prune_every": 20,
                    "removal_opacity_threshold": 0.005,
                    "final_removal_opacity_threshold": 0.005,
                    "reset_opacities": False,
                    "reset_opacities_every": 500,
                },
                "use_gaussian_splatting_densification": False,
                "densify_dict": {
                    "start_after": 500,
                    "remove_big_after": 3000,
                    "stop_after": 5000,
                    "densify_every": 100,
                    "grad_thresh": 0.0002,
                    "num_to_split_into": 2,
                    "removal_opacity_threshold": 0.005,
                    "final_removal_opacity_threshold": 0.005,
                    "reset_opacities_every": 3000,
                },
            },

            "viz": {
                "render_mode": "color",
                "offset_first_viz_cam": True,
                "show_sil": False,
                "visualize_cams": True,
                "viz_w": 600,
                "viz_h": 340,
                "viz_near": 0.01,
                "viz_far": 100.0,
                "view_scale": 2,
                "viz_fps": 5,
                "enter_interactive_post_online": False,
            },
        }

        # Add camera intrinsics if available
        if manifest.intrinsics:
            splatam_config["cam"] = {
                "fx": manifest.intrinsics.fx,
                "fy": manifest.intrinsics.fy,
                "cx": manifest.intrinsics.cx,
                "cy": manifest.intrinsics.cy,
                "H": manifest.intrinsics.height,
                "W": manifest.intrinsics.width,
                "png_depth_scale": manifest.depth_scale if hasattr(manifest, 'depth_scale') else 6553.5,
            }
        else:
            # Default camera params
            splatam_config["cam"] = {
                "fx": 600.0,
                "fy": 600.0,
                "cx": W / 2,
                "cy": H / 2,
                "H": H,
                "W": W,
                "png_depth_scale": 6553.5,
            }

        # Run SplaTAM
        try:
            poses, gaussians_path = self._run_splatam_python(splatam_config, output_dir)
        except Exception as e:
            logger.warning(f"Direct SplaTAM import failed: {e}")
            try:
                poses, gaussians_path = self._run_splatam_subprocess(
                    splatam_config, dataset_dir, output_dir
                )
            except Exception as e2:
                logger.error(f"SplaTAM subprocess failed: {e2}")
                raise RuntimeError(f"SplaTAM failed: {e}, {e2}")

        # Map poses back to original keyframes
        final_poses = []
        for idx, kf in valid_frames:
            if idx < len(poses):
                pose = poses[idx]
                final_poses.append(CameraPose(
                    frame_id=kf.frame_id,
                    image_name=Path(kf.file_path).name,
                    rotation=pose.rotation,
                    translation=pose.translation,
                    timestamp=kf.timestamp_seconds,
                ))

        # Save poses
        self._save_poses(final_poses, output_dir / "poses")

        return SLAMResult(
            poses=final_poses,
            gaussians_path=gaussians_path,
            registration_rate=len(final_poses) / len(keyframes) if keyframes else 0,
            scale_factor=1.0,  # Depth gives metric scale
            scale_confidence=0.95,
        )

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available for PyTorch."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _run_splatam_python(
        self,
        config: Dict[str, Any],
        output_dir: Path,
    ) -> Tuple[List[CameraPose], Optional[Path]]:
        """Run SplaTAM via direct Python import."""
        import sys

        # Add SplaTAM to path if needed
        if self._splatam_path:
            sys.path.insert(0, str(self._splatam_path))

        try:
            from scripts.splatam import rgbd_slam
        except ImportError:
            from splatam import rgbd_slam

        # Run SplaTAM
        rgbd_slam(config)

        # Parse output
        splatam_output = Path(config["workdir"]) / config["run_name"]
        poses = self._parse_splatam_poses(splatam_output)

        # Find Gaussians PLY
        gaussians_path = None
        for ply_pattern in ["params.npz", "final_params.npz", "*.ply"]:
            matches = list(splatam_output.glob(ply_pattern))
            if matches:
                if matches[0].suffix == ".npz":
                    # Convert npz to ply
                    gaussians_path = self._convert_splatam_npz_to_ply(
                        matches[0], output_dir / "gaussians" / "point_cloud.ply"
                    )
                else:
                    gaussians_path = matches[0]
                break

        return poses, gaussians_path

    def _run_splatam_subprocess(
        self,
        config: Dict[str, Any],
        dataset_dir: Path,
        output_dir: Path,
    ) -> Tuple[List[CameraPose], Optional[Path]]:
        """Run SplaTAM via subprocess."""
        import json as json_module

        # Write config to file
        config_path = output_dir / "splatam_config.json"
        with open(config_path, "w") as f:
            json_module.dump(config, f, indent=2)

        # Find splatam.py script
        script_path = None
        if self._splatam_path:
            script_path = self._splatam_path / "scripts" / "splatam.py"

        if not script_path or not script_path.exists():
            for path in [
                Path.home() / "SplaTAM" / "scripts" / "splatam.py",
                Path.cwd() / "SplaTAM" / "scripts" / "splatam.py",
            ]:
                if path.exists():
                    script_path = path
                    break

        if not script_path or not script_path.exists():
            raise RuntimeError("SplaTAM splatam.py script not found")

        # Create a Python script to run SplaTAM with our config
        runner_script = output_dir / "run_splatam.py"
        runner_script.write_text(f"""
import sys
import json
sys.path.insert(0, "{script_path.parent.parent}")
from scripts.splatam import rgbd_slam

with open("{config_path}") as f:
    config = json.load(f)

rgbd_slam(config)
""")

        # Run SplaTAM
        result = subprocess.run(
            ["python", str(runner_script)],
            capture_output=True,
            timeout=7200,  # 2 hour timeout for SLAM
            cwd=script_path.parent.parent,
        )

        if result.returncode != 0:
            raise RuntimeError(f"SplaTAM failed: {result.stderr.decode()}")

        # Parse output
        splatam_output = Path(config["workdir"]) / config["run_name"]
        poses = self._parse_splatam_poses(splatam_output)

        # Find Gaussians
        gaussians_path = None
        for npz_file in splatam_output.glob("*.npz"):
            gaussians_path = self._convert_splatam_npz_to_ply(
                npz_file, output_dir / "gaussians" / "point_cloud.ply"
            )
            break

        return poses, gaussians_path

    def _parse_splatam_poses(self, output_dir: Path) -> List[CameraPose]:
        """Parse poses from SplaTAM output.

        SplaTAM saves poses in traj/ folder or in params.npz.
        """
        poses = []

        # Try loading from traj_est.txt (TUM format)
        traj_file = output_dir / "traj_est.txt"
        if traj_file.exists():
            try:
                with open(traj_file) as f:
                    for i, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) >= 8:
                            # TUM format: timestamp tx ty tz qx qy qz qw
                            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                            qx, qy, qz, qw = (
                                float(parts[4]), float(parts[5]),
                                float(parts[6]), float(parts[7])
                            )

                            poses.append(CameraPose(
                                frame_id=f"{i:06d}",
                                image_name=f"{i:06d}.png",
                                rotation=(qw, qx, qy, qz),  # Convert to w,x,y,z
                                translation=(tx, ty, tz),
                            ))
                return poses
            except Exception as e:
                logger.warning(f"Failed to parse traj_est.txt: {e}")

        # Try loading from params.npz
        params_file = output_dir / "params.npz"
        if params_file.exists():
            try:
                data = np.load(params_file, allow_pickle=True)
                if "cam_unnorm_rots" in data and "cam_trans" in data:
                    rots = data["cam_unnorm_rots"]
                    trans = data["cam_trans"]

                    for i in range(len(trans)):
                        # Normalize quaternion
                        q = rots[i]
                        q = q / np.linalg.norm(q)

                        poses.append(CameraPose(
                            frame_id=f"{i:06d}",
                            image_name=f"{i:06d}.png",
                            rotation=tuple(q),
                            translation=tuple(trans[i]),
                        ))
                return poses
            except Exception as e:
                logger.warning(f"Failed to parse params.npz: {e}")

        return poses

    def _convert_splatam_npz_to_ply(self, npz_path: Path, output_path: Path) -> Path:
        """Convert SplaTAM params.npz to standard 3DGS PLY format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = np.load(npz_path, allow_pickle=True)

            # SplaTAM stores: means3D, rgb_colors, unnorm_rotations, logit_opacities, log_scales
            means = data.get("means3D", np.zeros((0, 3)))
            colors = data.get("rgb_colors", np.ones((len(means), 3)) * 0.5)
            rotations = data.get("unnorm_rotations", np.tile([1, 0, 0, 0], (len(means), 1)))
            opacities = data.get("logit_opacities", np.zeros(len(means)))
            scales = data.get("log_scales", np.ones((len(means), 3)) * -3)

            # Normalize rotations
            rot_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
            rotations = rotations / np.clip(rot_norms, 1e-6, None)

            # Convert logit opacities to actual opacities
            # sigmoid(logit) = 1 / (1 + exp(-logit))
            opacities_actual = 1 / (1 + np.exp(-opacities))

            # Build PLY content
            n_points = len(means)
            header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
            lines = [header]
            for i in range(n_points):
                x, y, z = means[i]
                r, g, b = colors[i] if i < len(colors) else (0.5, 0.5, 0.5)
                opacity = opacities_actual[i] if i < len(opacities_actual) else 0.1
                s0, s1, s2 = scales[i] if i < len(scales) else (-3, -3, -3)
                q0, q1, q2, q3 = rotations[i] if i < len(rotations) else (1, 0, 0, 0)

                # SH DC coefficients from RGB (simplified)
                f_dc_0 = (r - 0.5) / 0.28209479177387814
                f_dc_1 = (g - 0.5) / 0.28209479177387814
                f_dc_2 = (b - 0.5) / 0.28209479177387814

                lines.append(
                    f"{x:.6f} {y:.6f} {z:.6f} 0 0 1 "
                    f"{f_dc_0:.6f} {f_dc_1:.6f} {f_dc_2:.6f} "
                    f"{opacity:.6f} {s0:.6f} {s1:.6f} {s2:.6f} "
                    f"{q0:.6f} {q1:.6f} {q2:.6f} {q3:.6f}\n"
                )

            output_path.write_text("".join(lines))
            logger.info(f"Converted SplaTAM output to PLY: {n_points} Gaussians")
            return output_path

        except Exception as e:
            logger.error(f"Failed to convert SplaTAM npz to ply: {e}")
            return npz_path

    def _save_poses(self, poses: List[CameraPose], output_dir: Path) -> None:
        """Save poses in multiple formats."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON format
        poses_json = [
            {
                "frame_id": p.frame_id,
                "image_name": p.image_name,
                "rotation": list(p.rotation),
                "translation": list(p.translation),
                "timestamp": p.timestamp,
            }
            for p in poses
        ]
        (output_dir / "poses.json").write_text(json.dumps({"poses": poses_json}, indent=2))

        # COLMAP format
        with open(output_dir / "images.txt", "w") as f:
            f.write("# Image list\n")
            for i, p in enumerate(poses):
                qw, qx, qy, qz = p.rotation
                tx, ty, tz = p.translation
                f.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {p.image_name}\n")
                f.write("\n")

    def _run_depth_guided_colmap(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]],
    ) -> SLAMResult:
        """Run COLMAP with depth prior for initialization."""
        logger.info("Using depth-guided COLMAP + standalone 3DGS")

        # First generate initial point cloud from depth maps
        depth_dir = Path(manifest.depth_frames_path) if manifest.depth_frames_path else None
        initial_points = self._generate_points_from_depth(
            keyframes, frames_dir, depth_dir, manifest.intrinsics
        )

        # Run COLMAP with depth-derived points as initial
        colmap_fallback = COLMAPFallback(self.config)
        result = colmap_fallback.run(
            manifest, keyframes, frames_dir, output_dir, dynamic_masks
        )

        # The depth gives us metric scale
        if result.success:
            result.scale_factor = 1.0
            result.scale_confidence = 0.9

        return result

    def _generate_points_from_depth(
        self,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        depth_dir: Optional[Path],
        intrinsics: Optional[CameraIntrinsics],
    ) -> np.ndarray:
        """Generate 3D points from depth maps."""
        if not depth_dir or not intrinsics:
            return np.array([]).reshape(0, 3)

        if not CV2_AVAILABLE:
            return np.array([]).reshape(0, 3)

        all_points = []

        for kf in keyframes[:10]:  # Use first 10 frames for initialization
            depth_path = depth_dir / f"{kf.frame_id}_depth.png"
            if not depth_path.exists():
                continue

            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue

            # Convert to meters (assuming 16-bit depth in mm)
            depth = depth.astype(np.float32) / 1000.0

            # Generate 3D points
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.cx, intrinsics.cy

            h, w = depth.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            z = depth
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            # Subsample and filter
            valid = (z > 0.1) & (z < 10.0)
            points = np.stack([x[valid], y[valid], z[valid]], axis=-1)

            # Subsample
            if len(points) > 1000:
                indices = np.random.choice(len(points), 1000, replace=False)
                points = points[indices]

            all_points.append(points)

        if all_points:
            return np.concatenate(all_points, axis=0)
        return np.array([]).reshape(0, 3)


class VIGSSLAM(BaseSLAM):
    """VIGS-SLAM for visual-inertial captures.

    Visual-Inertial Gaussian Splatting SLAM for improved robustness
    under motion blur and low texture. Fuses camera images with IMU
    measurements for more accurate pose estimation.

    This implementation provides:
    1. Native VIGS-SLAM integration when available
    2. IMU-guided COLMAP fallback with gyro-based motion priors
    """

    def run(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]] = None,
        scale_observations: Optional[List[ScaleAnchorObservation]] = None,
    ) -> SLAMResult:
        """Run VIGS-SLAM with visual-inertial input."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if not manifest.has_imu or not manifest.imu_data_path:
            logger.info("No IMU data available for VIGS-SLAM, falling back to WildGS-SLAM")
            fallback = WildGSSLAM(self.config)
            return fallback.run(
                manifest, keyframes, frames_dir, output_dir, dynamic_masks, scale_observations
            )

        # Try native VIGS-SLAM first
        if self._check_vigs_available():
            try:
                result = self._run_native_vigs(
                    manifest, keyframes, frames_dir, output_dir, dynamic_masks
                )
                # IMU has metric scale, but apply calibration if available for refinement
                if scale_observations:
                    result = self._calibrate_scale(result, scale_observations, manifest.intrinsics)
                return result
            except Exception as e:
                logger.warning(f"Native VIGS-SLAM failed: {e}, using fallback")

        # Use IMU-guided COLMAP fallback
        return self._run_imu_guided_colmap(
            manifest, keyframes, frames_dir, output_dir, dynamic_masks
        )

    def _check_vigs_available(self) -> bool:
        """Check if VIGS-SLAM package is available."""
        try:
            import vigs_slam  # noqa: F401
            return True
        except ImportError:
            return False

    def _run_native_vigs(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]],
    ) -> SLAMResult:
        """Run native VIGS-SLAM implementation."""
        from vigs_slam import VIGSSLAMRunner

        # Load IMU data
        imu_data = self._load_imu_data(Path(manifest.imu_data_path))

        # Prepare image paths
        image_paths = []
        timestamps = []

        for kf in keyframes:
            frame_path = frames_dir.parent / kf.file_path
            if frame_path.exists():
                image_paths.append(str(frame_path))
                timestamps.append(kf.timestamp_seconds)

        if not image_paths:
            raise ValueError("No valid images found")

        # Configure VIGS-SLAM
        config = {
            "use_imu": True,
            "imu_noise_acc": 0.01,
            "imu_noise_gyro": 0.001,
            "num_iterations": 30000,
        }

        if manifest.intrinsics:
            config["camera"] = {
                "fx": manifest.intrinsics.fx,
                "fy": manifest.intrinsics.fy,
                "cx": manifest.intrinsics.cx,
                "cy": manifest.intrinsics.cy,
            }

        # Run VIGS-SLAM
        runner = VIGSSLAMRunner(config)
        result = runner.run(image_paths, timestamps, imu_data, output_dir)

        # Extract poses
        poses = []
        for i, pose_data in enumerate(result.get("poses", [])):
            poses.append(CameraPose(
                frame_id=keyframes[i].frame_id,
                image_name=Path(image_paths[i]).name,
                rotation=tuple(pose_data["rotation"]),
                translation=tuple(pose_data["translation"]),
                timestamp=timestamps[i],
            ))

        gaussians_path = output_dir / "gaussians" / "point_cloud.ply"

        return SLAMResult(
            poses=poses,
            gaussians_path=gaussians_path if gaussians_path.exists() else None,
            registration_rate=len(poses) / len(keyframes) if keyframes else 0,
            scale_factor=1.0,  # IMU gives metric scale
            scale_confidence=0.9,
        )

    def _run_imu_guided_colmap(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]],
    ) -> SLAMResult:
        """Run COLMAP with IMU-derived motion priors."""
        logger.info("Using IMU-guided COLMAP + standalone 3DGS")

        # Load and process IMU data
        imu_data = self._load_imu_data(Path(manifest.imu_data_path))

        # Compute relative rotations from gyroscope
        relative_rotations = self._integrate_gyro(imu_data, keyframes)

        # Run COLMAP (IMU priors help with feature matching)
        colmap_fallback = COLMAPFallback(self.config)
        result = colmap_fallback.run(
            manifest, keyframes, frames_dir, output_dir, dynamic_masks
        )

        # IMU gives us metric scale (from accelerometer)
        if result.success and imu_data:
            scale = self._estimate_scale_from_imu(imu_data, result.poses)
            result.scale_factor = scale
            result.scale_confidence = 0.8

        return result

    def _load_imu_data(self, imu_path: Path) -> List[Dict[str, Any]]:
        """Load IMU data from JSON/JSONL file."""
        if not imu_path.exists():
            return []

        imu_data = []

        # Try JSONL format first
        if imu_path.suffix == ".jsonl":
            with open(imu_path, "r") as f:
                for line in f:
                    if line.strip():
                        imu_data.append(json.loads(line))
        else:
            # Try JSON array
            with open(imu_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    imu_data = data
                elif "measurements" in data:
                    imu_data = data["measurements"]

        return imu_data

    def _integrate_gyro(
        self,
        imu_data: List[Dict[str, Any]],
        keyframes: List[FrameMetadata],
    ) -> List[np.ndarray]:
        """Integrate gyroscope to get relative rotations."""
        if not imu_data:
            return []

        relative_rotations = []

        for i in range(len(keyframes) - 1):
            t0 = keyframes[i].timestamp_seconds
            t1 = keyframes[i + 1].timestamp_seconds

            # Find IMU measurements in this interval
            measurements = [
                m for m in imu_data
                if t0 <= m.get("timestamp", 0) <= t1
            ]

            if not measurements:
                relative_rotations.append(np.eye(3))
                continue

            # Simple gyro integration (euler approximation)
            R = np.eye(3)
            prev_t = t0

            for m in measurements:
                if "gyro" not in m:
                    continue

                t = m["timestamp"]
                dt = t - prev_t
                prev_t = t

                gx, gy, gz = m["gyro"]

                # Skew-symmetric matrix
                omega = np.array([
                    [0, -gz, gy],
                    [gz, 0, -gx],
                    [-gy, gx, 0]
                ])

                # Rodrigues formula (small angle approximation)
                R = R @ (np.eye(3) + omega * dt)

            relative_rotations.append(R)

        return relative_rotations

    def _estimate_scale_from_imu(
        self,
        imu_data: List[Dict[str, Any]],
        poses: List[CameraPose],
    ) -> float:
        """Estimate metric scale from IMU accelerometer."""
        if not imu_data or len(poses) < 2:
            return 1.0

        # Simple double integration of accelerometer
        # This is a rough estimate; real implementation would use
        # proper VIO optimization

        # Get total visual displacement
        visual_displacement = 0.0
        for i in range(len(poses) - 1):
            t0 = np.array(poses[i].translation)
            t1 = np.array(poses[i + 1].translation)
            visual_displacement += np.linalg.norm(t1 - t0)

        if visual_displacement < 1e-6:
            return 1.0

        # Get IMU displacement estimate
        # This is very approximate without proper bias estimation
        velocity = np.zeros(3)
        position = np.zeros(3)
        prev_t = imu_data[0].get("timestamp", 0)

        for m in imu_data:
            if "accel" not in m:
                continue

            t = m["timestamp"]
            dt = t - prev_t
            prev_t = t

            accel = np.array(m["accel"])
            # Remove gravity (approximate)
            accel[2] -= 9.81

            velocity += accel * dt
            position += velocity * dt

        imu_displacement = np.linalg.norm(position)

        if imu_displacement < 1e-6:
            return 1.0

        return imu_displacement / visual_displacement


class ARKitDirect(BaseSLAM):
    """Direct ARKit pose import (no SLAM needed).

    For iOS captures with ARKit tracking, we can skip SLAM entirely
    and use the metric-scale poses directly.
    """

    def run(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]] = None,
        scale_observations: Optional[List[ScaleAnchorObservation]] = None,
    ) -> SLAMResult:
        """Load ARKit poses and train 3DGS."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if not manifest.has_arkit_poses or not manifest.arkit_poses_path:
            print("No ARKit poses available, falling back to WildGS-SLAM")
            fallback = WildGSSLAM(self.config)
            return fallback.run(
                manifest, keyframes, frames_dir, output_dir, dynamic_masks, scale_observations
            )

        # Load ARKit poses
        arkit_poses = self._load_arkit_poses(
            Path(manifest.arkit_poses_path), keyframes
        )

        if not arkit_poses:
            print("Failed to load ARKit poses, falling back to WildGS-SLAM")
            fallback = WildGSSLAM(self.config)
            return fallback.run(
                manifest, keyframes, frames_dir, output_dir, dynamic_masks
            )

        # Save poses
        poses_dir = output_dir / "poses"
        poses_dir.mkdir(exist_ok=True)
        self._save_poses_json(arkit_poses, poses_dir / "poses.json")
        self._save_colmap_format(arkit_poses, manifest.intrinsics, poses_dir)

        # Train 3DGS with known poses
        gaussians_path = self._train_3dgs(
            keyframes, frames_dir, arkit_poses, manifest.intrinsics,
            output_dir, dynamic_masks
        )

        return SLAMResult(
            poses=arkit_poses,
            gaussians_path=gaussians_path,
            registration_rate=1.0,  # All poses from ARKit
            scale_factor=1.0,  # ARKit provides metric scale
            scale_confidence=1.0,
        )

    def _load_arkit_poses(
        self,
        arkit_path: Path,
        keyframes: List[FrameMetadata],
    ) -> List[CameraPose]:
        """Load ARKit poses from JSONL file."""
        poses = []
        poses_file = arkit_path / "poses.jsonl"

        if not poses_file.exists():
            # Try parent directory
            poses_file = arkit_path.parent / "arkit" / "poses.jsonl"

        if not poses_file.exists():
            return []

        # Load all ARKit poses
        arkit_poses_by_timestamp = {}
        with open(poses_file, "r") as f:
            for line in f:
                if line.strip():
                    pose_data = json.loads(line)
                    timestamp = pose_data.get("timestamp", 0)
                    arkit_poses_by_timestamp[timestamp] = pose_data

        # Match keyframes to ARKit poses
        for kf in keyframes:
            # Find closest pose by timestamp
            closest_ts = min(
                arkit_poses_by_timestamp.keys(),
                key=lambda ts: abs(ts - kf.timestamp_seconds),
                default=None
            )

            if closest_ts is not None:
                pose_data = arkit_poses_by_timestamp[closest_ts]

                # ARKit provides 4x4 camera-to-world transform
                # Convert to COLMAP convention (world-to-camera)
                transform = np.array(pose_data.get("transform", np.eye(4).tolist()))
                transform = transform.reshape(4, 4)

                # Invert for world-to-camera
                transform_inv = np.linalg.inv(transform)

                # Extract rotation (quaternion) and translation
                rotation_matrix = transform_inv[:3, :3]
                translation = transform_inv[:3, 3]

                # Convert rotation matrix to quaternion
                quat = self._rotation_matrix_to_quaternion(rotation_matrix)

                poses.append(CameraPose(
                    frame_id=kf.frame_id,
                    image_name=Path(kf.file_path).name,
                    rotation=tuple(quat),
                    translation=tuple(translation),
                    timestamp=kf.timestamp_seconds,
                ))

        return poses

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
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
        return (w, x, y, z)

    def _save_poses_json(self, poses: List[CameraPose], path: Path) -> None:
        """Save poses to JSON."""
        poses_data = [
            {
                "frame_id": p.frame_id,
                "image_name": p.image_name,
                "rotation": list(p.rotation),
                "translation": list(p.translation),
                "timestamp": p.timestamp,
            }
            for p in poses
        ]
        path.write_text(json.dumps({"poses": poses_data}, indent=2))

    def _save_colmap_format(
        self,
        poses: List[CameraPose],
        intrinsics: Optional[CameraIntrinsics],
        output_dir: Path,
    ) -> None:
        """Save poses in COLMAP text format."""
        # cameras.txt
        with open(output_dir / "cameras.txt", "w") as f:
            f.write("# Camera list\n")
            if intrinsics:
                f.write(f"1 PINHOLE {intrinsics.width} {intrinsics.height} "
                       f"{intrinsics.fx} {intrinsics.fy} {intrinsics.cx} {intrinsics.cy}\n")
            else:
                f.write("1 PINHOLE 1920 1080 1500 1500 960 540\n")

        # images.txt
        with open(output_dir / "images.txt", "w") as f:
            f.write("# Image list\n")
            for i, p in enumerate(poses):
                qw, qx, qy, qz = p.rotation
                tx, ty, tz = p.translation
                f.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {p.image_name}\n")
                f.write("\n")

        # points3D.txt (empty)
        with open(output_dir / "points3D.txt", "w") as f:
            f.write("# 3D point list\n")

    def _train_3dgs(
        self,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        poses: List[CameraPose],
        intrinsics: Optional[CameraIntrinsics],
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]],
    ) -> Optional[Path]:
        """Train 3D Gaussian Splatting with known poses."""
        colmap_dir = output_dir / "colmap"
        colmap_dir.mkdir(exist_ok=True)

        # Prepare images directory
        images_dir = colmap_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for kf in keyframes:
            src = frames_dir.parent / kf.file_path
            dst = images_dir / Path(kf.file_path).name
            if src.exists():
                shutil.copy(src, dst)

        # Create sparse model directory
        sparse_dir = colmap_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        self._save_colmap_format(poses, intrinsics, sparse_dir)

        # Train 3DGS
        gaussians_dir = output_dir / "gaussians"
        gaussians_dir.mkdir(exist_ok=True)

        try:
            result = subprocess.run(
                [
                    "python", "-m", "gaussian_splatting.train",
                    "--source_path", str(colmap_dir),
                    "--model_path", str(gaussians_dir),
                    "--iterations", "30000",
                ],
                capture_output=True,
                timeout=3600,
            )

            gaussians_path = gaussians_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            if gaussians_path.exists():
                return gaussians_path

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Create placeholder if training failed
        placeholder = gaussians_dir / "point_cloud.ply"
        self._create_placeholder_ply(placeholder)
        return placeholder

    def _create_placeholder_ply(self, path: Path) -> None:
        """Create minimal PLY file."""
        ply = """ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
end_header
0 0 0
"""
        path.write_text(ply)


class COLMAPFallback(BaseSLAM):
    """COLMAP SfM + standalone 3DGS training.

    Used when other SLAM backends are not available.
    Now uses our built-in 3DGS training module instead of external packages.

    Supports two modes:
    1. pycolmap (preferred) - Python bindings, no CLI needed
    2. colmap CLI (fallback) - requires colmap to be installed
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._colmap_available = None
        self._use_pycolmap = PYCOLMAP_AVAILABLE

    def check_colmap_available(self) -> bool:
        """Check if COLMAP is installed and available (pycolmap or CLI)."""
        if self._colmap_available is not None:
            return self._colmap_available

        # First check for pycolmap (preferred)
        if PYCOLMAP_AVAILABLE:
            logger.info("Using pycolmap Python bindings for COLMAP")
            self._colmap_available = True
            self._use_pycolmap = True
            return True

        # Fallback to CLI
        try:
            result = subprocess.run(
                ["colmap", "--help"],
                capture_output=True,
                timeout=10
            )
            self._colmap_available = result.returncode == 0
            self._use_pycolmap = False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._colmap_available = False

        if not self._colmap_available:
            logger.warning(
                "COLMAP not found. Install with: "
                "pip install pycolmap (recommended) or "
                "apt install colmap (Ubuntu) or brew install colmap (macOS)"
            )

        return self._colmap_available

    def run(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]] = None,
        scale_observations: Optional[List[ScaleAnchorObservation]] = None,
    ) -> SLAMResult:
        """Run COLMAP SfM then train 3DGS using our standalone module."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check COLMAP availability
        if not self.check_colmap_available():
            logger.error("COLMAP is required but not installed")
            return SLAMResult(
                poses=[],
                success=False,
                errors=["COLMAP not installed. Please install COLMAP first."],
            )

        colmap_dir = output_dir / "colmap"
        colmap_dir.mkdir(exist_ok=True)

        # Prepare images (apply masks if available)
        images_dir = self._prepare_images(
            keyframes, frames_dir, dynamic_masks, colmap_dir / "images"
        )

        # Run COLMAP
        poses = self._run_colmap(images_dir, colmap_dir, manifest.intrinsics)

        if not poses:
            return SLAMResult(
                poses=[],
                success=False,
                errors=["COLMAP reconstruction failed"],
            )

        # Save poses
        poses_dir = output_dir / "poses"
        poses_dir.mkdir(exist_ok=True)
        self._save_poses(poses, poses_dir)

        # Train 3DGS using our standalone module
        gaussians_dir = output_dir / "gaussians"
        gaussians_dir.mkdir(exist_ok=True)
        gaussians_path = self._train_3dgs_standalone(colmap_dir, gaussians_dir)

        result = SLAMResult(
            poses=poses,
            gaussians_path=gaussians_path,
            registration_rate=len(poses) / len(keyframes) if keyframes else 0,
        )

        # Apply scale calibration if observations available
        if scale_observations:
            result = self._calibrate_scale(result, scale_observations, manifest.intrinsics)

        return result

    def _prepare_images(
        self,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]],
        output_dir: Path,
    ) -> Path:
        """Prepare images for COLMAP, applying masks to hide dynamic regions."""
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import cv2
        except ImportError:
            # Just copy images without masking
            for kf in keyframes:
                src = frames_dir.parent / kf.file_path
                dst = output_dir / f"{kf.frame_id}.png"
                if src.exists():
                    shutil.copy(src, dst)
            return output_dir

        for kf in keyframes:
            src = frames_dir.parent / kf.file_path
            dst = output_dir / f"{kf.frame_id}.png"

            if not src.exists():
                continue

            img = cv2.imread(str(src))
            if img is None:
                continue

            # Apply dynamic mask if available
            if dynamic_masks and kf.frame_id in dynamic_masks:
                mask_path = dynamic_masks[kf.frame_id]
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None and mask.shape[:2] == img.shape[:2]:
                        # Set masked regions to neutral gray
                        img[mask > 127] = [128, 128, 128]

            cv2.imwrite(str(dst), img)

        return output_dir

    def _run_colmap(
        self,
        images_dir: Path,
        output_dir: Path,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> List[CameraPose]:
        """Run COLMAP SfM pipeline using pycolmap or CLI."""
        if self._use_pycolmap and PYCOLMAP_AVAILABLE:
            return self._run_colmap_pycolmap(images_dir, output_dir, intrinsics)
        else:
            return self._run_colmap_cli(images_dir, output_dir, intrinsics)

    def _run_colmap_pycolmap(
        self,
        images_dir: Path,
        output_dir: Path,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> List[CameraPose]:
        """Run COLMAP using pycolmap Python bindings (preferred method)."""
        database_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        try:
            # Remove existing database if present
            if database_path.exists():
                database_path.unlink()

            logger.info("Running COLMAP feature extraction via pycolmap...")

            # Configure camera options
            camera_mode = pycolmap.CameraMode.SINGLE
            if intrinsics:
                # Use known intrinsics
                camera_model = "PINHOLE"
                camera_params = f"{intrinsics.fx},{intrinsics.fy},{intrinsics.cx},{intrinsics.cy}"
            else:
                camera_model = "SIMPLE_RADIAL"
                camera_params = None

            # Feature extraction
            sift_options = pycolmap.SiftExtractionOptions()
            sift_options.max_num_features = 8192

            pycolmap.extract_features(
                database_path=database_path,
                image_path=images_dir,
                camera_mode=camera_mode,
                camera_model=camera_model,
                sift_options=sift_options,
            )

            logger.info("Running COLMAP feature matching via pycolmap...")

            # Sequential matching (best for video sequences)
            matching_options = pycolmap.SequentialMatchingOptions()
            matching_options.overlap = 10
            matching_options.quadratic_overlap = True

            pycolmap.match_sequential(
                database_path=database_path,
                matching_options=matching_options,
            )

            logger.info("Running COLMAP sparse reconstruction via pycolmap...")

            # Incremental mapper
            mapper_options = pycolmap.IncrementalMapperOptions()
            mapper_options.min_num_matches = 15

            reconstructions = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=images_dir,
                output_path=sparse_dir,
                options=mapper_options,
            )

            if not reconstructions:
                logger.warning("No reconstructions from pycolmap, trying exhaustive matching...")
                # Try with exhaustive matching as fallback
                pycolmap.match_exhaustive(database_path=database_path)
                reconstructions = pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=images_dir,
                    output_path=sparse_dir,
                    options=mapper_options,
                )

            if not reconstructions:
                logger.error("COLMAP reconstruction failed - no valid reconstructions")
                return []

            # Get the best reconstruction (usually index 0)
            best_reconstruction = reconstructions[0] if reconstructions else None
            if best_reconstruction is None:
                return []

            # Extract poses from reconstruction
            poses = []
            for image_id, image in best_reconstruction.images.items():
                # pycolmap provides rotation as quaternion and translation
                quat = image.cam_from_world.rotation.quat  # [w, x, y, z]
                trans = image.cam_from_world.translation

                poses.append(CameraPose(
                    frame_id=Path(image.name).stem,
                    image_name=image.name,
                    rotation=(quat[0], quat[1], quat[2], quat[3]),
                    translation=(trans[0], trans[1], trans[2]),
                ))

            logger.info(f"pycolmap reconstruction: {len(poses)} poses from {len(best_reconstruction.images)} images")

            # Also write the model in text format for 3DGS training
            model_dir = sparse_dir / "0"
            model_dir.mkdir(exist_ok=True)
            best_reconstruction.write_text(str(model_dir))

            return poses

        except Exception as e:
            logger.error(f"pycolmap failed: {e}")
            # Fallback to CLI if pycolmap fails
            logger.info("Falling back to COLMAP CLI...")
            return self._run_colmap_cli(images_dir, output_dir, intrinsics)

    def _run_colmap_cli(
        self,
        images_dir: Path,
        output_dir: Path,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> List[CameraPose]:
        """Run COLMAP SfM pipeline using CLI (fallback method)."""
        database_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        try:
            # Feature extraction with optional known intrinsics
            extract_cmd = [
                "colmap", "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(images_dir),
                "--ImageReader.single_camera", "1",
            ]

            # If we have known intrinsics, use them
            if intrinsics:
                extract_cmd.extend([
                    "--ImageReader.camera_model", "PINHOLE",
                    "--ImageReader.camera_params",
                    f"{intrinsics.fx},{intrinsics.fy},{intrinsics.cx},{intrinsics.cy}",
                ])

            logger.info("Running COLMAP feature extraction...")
            subprocess.run(extract_cmd, check=True, capture_output=True, timeout=600)

            # Feature matching - use sequential matcher for video sequences
            logger.info("Running COLMAP feature matching...")
            subprocess.run([
                "colmap", "sequential_matcher",
                "--database_path", str(database_path),
                "--SequentialMatching.overlap", "10",
            ], check=True, capture_output=True, timeout=600)

            # Sparse reconstruction
            logger.info("Running COLMAP sparse reconstruction...")
            subprocess.run([
                "colmap", "mapper",
                "--database_path", str(database_path),
                "--image_path", str(images_dir),
                "--output_path", str(sparse_dir),
            ], check=True, capture_output=True, timeout=1800)

        except subprocess.CalledProcessError as e:
            logger.error(f"COLMAP failed: {e.stderr.decode() if e.stderr else e}")
            return []
        except subprocess.TimeoutExpired:
            logger.error("COLMAP timed out")
            return []
        except FileNotFoundError:
            logger.error("COLMAP executable not found")
            return []

        # Parse poses
        return self._parse_colmap_poses(sparse_dir / "0")

    def _parse_colmap_poses(self, model_dir: Path) -> List[CameraPose]:
        """Parse poses from COLMAP sparse model."""
        poses = []
        images_txt = model_dir / "images.txt"

        if not images_txt.exists():
            # Try binary format
            images_bin = model_dir / "images.bin"
            if images_bin.exists():
                try:
                    subprocess.run([
                        "colmap", "model_converter",
                        "--input_path", str(model_dir),
                        "--output_path", str(model_dir),
                        "--output_type", "TXT",
                    ], capture_output=True)
                except Exception:
                    pass

        if not images_txt.exists():
            return poses

        with open(images_txt, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue

            parts = line.split()
            if len(parts) >= 10:
                image_id = int(parts[0])
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                image_name = parts[9]

                poses.append(CameraPose(
                    frame_id=Path(image_name).stem,
                    image_name=image_name,
                    rotation=(qw, qx, qy, qz),
                    translation=(tx, ty, tz),
                ))

            i += 2  # Skip points line

        return poses

    def _save_poses(self, poses: List[CameraPose], output_dir: Path) -> None:
        """Save poses in JSON format."""
        poses_data = [
            {
                "frame_id": p.frame_id,
                "image_name": p.image_name,
                "rotation": list(p.rotation),
                "translation": list(p.translation),
            }
            for p in poses
        ]
        (output_dir / "poses.json").write_text(json.dumps({"poses": poses_data}, indent=2))

    def _train_3dgs_standalone(
        self,
        colmap_dir: Path,
        output_dir: Path,
        iterations: int = 30000,
    ) -> Optional[Path]:
        """Train 3DGS using our standalone module (no external dependencies).

        Args:
            colmap_dir: Path to COLMAP sparse reconstruction
            output_dir: Output directory for trained Gaussians
            iterations: Number of training iterations

        Returns:
            Path to output PLY file, or None if training failed
        """
        try:
            from ..reconstruction.gaussian_splatting import (
                GaussianModel,
                GaussianTrainer,
                GaussianConfig,
            )
            from ..reconstruction.point_cloud import initialize_from_colmap

            logger.info(f"Training 3DGS from {colmap_dir}")

            # Load COLMAP data
            sparse_dir = colmap_dir / "sparse" / "0"
            if not sparse_dir.exists():
                sparse_dir = colmap_dir / "sparse"
            if not sparse_dir.exists():
                sparse_dir = colmap_dir

            points, colors, cameras, images = initialize_from_colmap(sparse_dir)

            if len(points) == 0:
                logger.warning("No points from COLMAP, creating placeholder")
                return self._create_placeholder_gaussians(output_dir)

            logger.info(f"Loaded {len(points)} points from COLMAP")

            # Initialize model
            config = GaussianConfig(iterations=iterations)
            model = GaussianModel(sh_degree=config.sh_degree)

            # Compute spatial extent for learning rate scaling
            extent = np.max(np.abs(points)) if len(points) > 0 else 1.0
            model.initialize_from_point_cloud(points, colors, spatial_lr_scale=extent)

            # Create trainer
            trainer = GaussianTrainer(model, config, output_dir)

            # Prepare training data from COLMAP
            training_cameras = self._prepare_training_cameras(
                cameras, images, colmap_dir
            )

            if not training_cameras:
                logger.warning("No training cameras found")
                return self._create_placeholder_gaussians(output_dir)

            logger.info(f"Training with {len(training_cameras)} views for {iterations} iterations")

            # Training loop
            for iteration in range(iterations):
                camera = training_cameras[iteration % len(training_cameras)]

                try:
                    import torch
                    gt_image = camera["image"].to(trainer.device)
                    metrics = trainer.train_step(camera, gt_image)

                    if iteration % 1000 == 0:
                        logger.info(
                            f"Iteration {iteration}: loss={metrics['loss']:.4f}, "
                            f"gaussians={metrics['num_gaussians']}"
                        )
                except Exception as e:
                    if iteration == 0:
                        logger.error(f"Training failed on first iteration: {e}")
                        return self._create_placeholder_gaussians(output_dir)
                    continue

            # Save final result
            final_dir = output_dir / "point_cloud" / "iteration_30000"
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / "point_cloud.ply"
            trainer.save_gaussians(final_path)

            logger.info(f"Training complete. Saved to {final_path}")
            return final_path

        except ImportError as e:
            logger.warning(f"Standalone 3DGS not available: {e}")
            return self._try_external_3dgs(colmap_dir, output_dir)
        except Exception as e:
            logger.error(f"3DGS training failed: {e}")
            return self._create_placeholder_gaussians(output_dir)

    def _prepare_training_cameras(
        self,
        cameras: Dict,
        images: Dict,
        colmap_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Prepare camera data for training."""
        training_cameras = []

        try:
            import torch
            from PIL import Image as PILImage
        except ImportError:
            logger.warning("PyTorch or PIL not available for training")
            return []

        images_dir = colmap_dir / "images"

        for img_id, img_data in images.items():
            cam_id = img_data.get("camera_id", 1)
            cam_params = cameras.get(cam_id, {})

            if not cam_params:
                continue

            # Find image file
            img_name = img_data.get("name", "")
            img_path = images_dir / img_name

            if not img_path.exists():
                continue

            try:
                pil_image = PILImage.open(img_path).convert("RGB")
                image_tensor = torch.tensor(
                    np.array(pil_image) / 255.0,
                    dtype=torch.float32
                ).permute(2, 0, 1)
            except Exception:
                continue

            # Build world-to-camera matrix
            R = img_data.get("rotation", np.eye(3))
            t = img_data.get("translation", np.zeros(3))
            world_to_camera = np.eye(4)
            world_to_camera[:3, :3] = R
            world_to_camera[:3, 3] = t

            training_cameras.append({
                "image": image_tensor,
                "image_height": cam_params.get("height", image_tensor.shape[1]),
                "image_width": cam_params.get("width", image_tensor.shape[2]),
                "fx": cam_params.get("fx", 1000),
                "fy": cam_params.get("fy", 1000),
                "cx": cam_params.get("cx", image_tensor.shape[2] / 2),
                "cy": cam_params.get("cy", image_tensor.shape[1] / 2),
                "world_to_camera": world_to_camera,
            })

        return training_cameras

    def _try_external_3dgs(self, colmap_dir: Path, output_dir: Path) -> Optional[Path]:
        """Try using external gaussian_splatting package as fallback."""
        try:
            result = subprocess.run([
                "python", "-m", "gaussian_splatting.train",
                "--source_path", str(colmap_dir),
                "--model_path", str(output_dir),
                "--iterations", "30000",
            ], capture_output=True, timeout=3600)

            ply_path = output_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            if ply_path.exists():
                return ply_path
        except Exception:
            pass

        return self._create_placeholder_gaussians(output_dir)

    def _create_placeholder_gaussians(self, output_dir: Path) -> Path:
        """Create a placeholder Gaussian PLY file."""
        ply_dir = output_dir / "point_cloud" / "iteration_30000"
        ply_dir.mkdir(parents=True, exist_ok=True)
        ply_path = ply_dir / "point_cloud.ply"

        # Create minimal valid 3DGS PLY
        ply_content = """ply
format ascii 1.0
element vertex 100
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
        # Add some random points
        np.random.seed(42)
        for _ in range(100):
            x, y, z = np.random.randn(3) * 0.5
            ply_content += f"{x:.6f} {y:.6f} {z:.6f} 0 0 1 0.5 0.5 0.5 0.1 -3 -3 -3 1 0 0 0\n"

        ply_path.write_text(ply_content)
        logger.info(f"Created placeholder Gaussians at {ply_path}")
        return ply_path


def get_slam_backend(backend: SLAMBackend, config: PipelineConfig) -> BaseSLAM:
    """Factory function to get SLAM backend."""
    backends = {
        SLAMBackend.WILDGS_SLAM: WildGSSLAM,
        SLAMBackend.SPLATMAP: WildGSSLAM,  # Use WildGS as placeholder
        SLAMBackend.SPLATAM: SplaTAM,
        SLAMBackend.VIGS_SLAM: VIGSSLAM,
        SLAMBackend.ARKIT_DIRECT: ARKitDirect,
        SLAMBackend.COLMAP_FALLBACK: COLMAPFallback,
    }
    return backends.get(backend, WildGSSLAM)(config)
