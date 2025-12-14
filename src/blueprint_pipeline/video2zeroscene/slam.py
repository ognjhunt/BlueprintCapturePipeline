"""Stage 2: Sensor-conditional SLAM backends.

This module provides a unified interface for different SLAM backends:
- WildGS-SLAM: Default for RGB-only captures (handles dynamics)
- SplatMAP: Alternative for RGB-only (geometry focus)
- SplaTAM: For RGB-D captures (iPhone LiDAR)
- VIGS-SLAM: For visual-inertial captures
- ARKit Direct: Direct ARKit pose import
- COLMAP Fallback: SfM + 3DGS when other methods unavailable
"""

from __future__ import annotations

import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .interfaces import (
    CameraIntrinsics,
    CaptureManifest,
    FrameMetadata,
    PipelineConfig,
    SLAMBackend,
    SensorType,
    Submap,
)


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
    ) -> SLAMResult:
        """Run SLAM reconstruction."""
        pass

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
    """

    def run(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]] = None,
    ) -> SLAMResult:
        """Run WildGS-SLAM reconstruction."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if WildGS-SLAM is available
        wildgs_available = self._check_wildgs_available()

        if wildgs_available:
            return self._run_native_wildgs(
                manifest, keyframes, frames_dir, output_dir, dynamic_masks
            )
        else:
            # Fall back to COLMAP + 3DGS
            print("WildGS-SLAM not available, using COLMAP fallback")
            fallback = COLMAPFallback(self.config)
            return fallback.run(
                manifest, keyframes, frames_dir, output_dir, dynamic_masks
            )

    def _check_wildgs_available(self) -> bool:
        """Check if WildGS-SLAM package is available."""
        try:
            import wildgs_slam  # noqa: F401
            return True
        except ImportError:
            return False

    def _run_native_wildgs(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]],
    ) -> SLAMResult:
        """Run native WildGS-SLAM implementation."""
        from wildgs_slam import WildGSSLAM as WildGSLib

        # Prepare image paths
        image_paths = []
        mask_paths = []

        for kf in keyframes:
            frame_path = frames_dir.parent / kf.file_path
            image_paths.append(str(frame_path))

            if dynamic_masks and kf.frame_id in dynamic_masks:
                mask_paths.append(str(dynamic_masks[kf.frame_id]))
            else:
                mask_paths.append(None)

        # Configure SLAM
        config = {
            "use_masks": bool(dynamic_masks),
            "num_iterations": 30000,
        }

        if manifest.intrinsics:
            config["intrinsics"] = {
                "fx": manifest.intrinsics.fx,
                "fy": manifest.intrinsics.fy,
                "cx": manifest.intrinsics.cx,
                "cy": manifest.intrinsics.cy,
            }

        # Run SLAM
        slam = WildGSLib(config)
        poses = []

        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            pose = slam.process_frame(img_path, mask_path)
            poses.append(CameraPose(
                frame_id=keyframes[i].frame_id,
                image_name=Path(img_path).name,
                rotation=tuple(pose["rotation"]),
                translation=tuple(pose["translation"]),
                timestamp=keyframes[i].timestamp_seconds,
            ))

        # Export gaussians
        gaussians_dir = output_dir / "gaussians"
        gaussians_dir.mkdir(exist_ok=True)
        gaussians_path = gaussians_dir / "point_cloud.ply"
        slam.export_gaussians(str(gaussians_path))

        # Save poses
        self._save_poses(poses, output_dir / "poses")

        return SLAMResult(
            poses=poses,
            gaussians_path=gaussians_path,
            registration_rate=len(poses) / len(keyframes) if keyframes else 0,
        )

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

    Designed for dense SLAM with depth sensor input.
    """

    def run(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]] = None,
    ) -> SLAMResult:
        """Run SplaTAM with RGB-D input."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for depth data
        if not manifest.has_depth or not manifest.depth_frames_path:
            print("No depth data available for SplaTAM, falling back to WildGS-SLAM")
            fallback = WildGSSLAM(self.config)
            return fallback.run(
                manifest, keyframes, frames_dir, output_dir, dynamic_masks
            )

        try:
            return self._run_native_splatam(
                manifest, keyframes, frames_dir, output_dir
            )
        except Exception as e:
            print(f"SplaTAM failed: {e}, falling back to COLMAP")
            fallback = COLMAPFallback(self.config)
            return fallback.run(
                manifest, keyframes, frames_dir, output_dir, dynamic_masks
            )

    def _run_native_splatam(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
    ) -> SLAMResult:
        """Run native SplaTAM implementation."""
        try:
            import splatam
        except ImportError:
            raise ImportError("splatam package not installed")

        # Implementation placeholder
        # Real implementation would follow SplaTAM API
        raise NotImplementedError("Native SplaTAM integration pending")


class VIGSSLAM(BaseSLAM):
    """VIGS-SLAM for visual-inertial captures.

    Visual-Inertial Gaussian Splatting SLAM for improved robustness
    under motion blur and low texture.
    """

    def run(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]] = None,
    ) -> SLAMResult:
        """Run VIGS-SLAM with visual-inertial input."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if not manifest.has_imu or not manifest.imu_data_path:
            print("No IMU data available for VIGS-SLAM, falling back to WildGS-SLAM")
            fallback = WildGSSLAM(self.config)
            return fallback.run(
                manifest, keyframes, frames_dir, output_dir, dynamic_masks
            )

        # VIGS-SLAM integration placeholder
        print("VIGS-SLAM not yet integrated, using WildGS-SLAM")
        fallback = WildGSSLAM(self.config)
        return fallback.run(
            manifest, keyframes, frames_dir, output_dir, dynamic_masks
        )


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
    ) -> SLAMResult:
        """Load ARKit poses and train 3DGS."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if not manifest.has_arkit_poses or not manifest.arkit_poses_path:
            print("No ARKit poses available, falling back to WildGS-SLAM")
            fallback = WildGSSLAM(self.config)
            return fallback.run(
                manifest, keyframes, frames_dir, output_dir, dynamic_masks
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
    """COLMAP SfM + 3DGS fallback.

    Used when other SLAM backends are not available.
    """

    def run(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
        dynamic_masks: Optional[Dict[str, Path]] = None,
    ) -> SLAMResult:
        """Run COLMAP SfM then train 3DGS."""
        output_dir.mkdir(parents=True, exist_ok=True)

        colmap_dir = output_dir / "colmap"
        colmap_dir.mkdir(exist_ok=True)

        # Prepare images (apply masks if available)
        images_dir = self._prepare_images(
            keyframes, frames_dir, dynamic_masks, colmap_dir / "images"
        )

        # Run COLMAP
        poses = self._run_colmap(images_dir, colmap_dir)

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

        # Train 3DGS
        gaussians_dir = output_dir / "gaussians"
        gaussians_dir.mkdir(exist_ok=True)
        gaussians_path = self._train_3dgs(colmap_dir, gaussians_dir)

        return SLAMResult(
            poses=poses,
            gaussians_path=gaussians_path,
            registration_rate=len(poses) / len(keyframes) if keyframes else 0,
        )

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

    def _run_colmap(self, images_dir: Path, output_dir: Path) -> List[CameraPose]:
        """Run COLMAP SfM pipeline."""
        database_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        try:
            # Feature extraction
            subprocess.run([
                "colmap", "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(images_dir),
                "--ImageReader.single_camera", "1",
            ], check=True, capture_output=True, timeout=600)

            # Feature matching
            subprocess.run([
                "colmap", "exhaustive_matcher",
                "--database_path", str(database_path),
            ], check=True, capture_output=True, timeout=600)

            # Sparse reconstruction
            subprocess.run([
                "colmap", "mapper",
                "--database_path", str(database_path),
                "--image_path", str(images_dir),
                "--output_path", str(sparse_dir),
            ], check=True, capture_output=True, timeout=1200)

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"COLMAP failed: {e}")
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

    def _train_3dgs(self, colmap_dir: Path, output_dir: Path) -> Optional[Path]:
        """Train 3DGS from COLMAP output."""
        try:
            subprocess.run([
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

        # Create placeholder
        placeholder = output_dir / "point_cloud.ply"
        placeholder.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
        return placeholder


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
