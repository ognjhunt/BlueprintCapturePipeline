"""Capture export for BlueprintPipeline/DWM handoff.

This module creates a simple output format containing:
- 3D Gaussian splat (point_cloud.ply)
- Camera trajectory and intrinsics
- Capture metadata

This output is ready for DWM (Dexterous World Models) processing
in the BlueprintPipeline repository.

Output structure:
    capture_output/
        gaussians.ply           # 3D Gaussian splatting point cloud
        camera/
            intrinsics.json     # Camera parameters
            trajectory.json     # Per-frame camera poses
        capture_info.json       # Metadata for handoff
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .interfaces import (
    CameraIntrinsics,
    CaptureManifest,
)
from .slam import CameraPose


@dataclass
class CaptureExportResult:
    """Result of capture export."""
    output_path: Path
    gaussians_path: Optional[Path] = None
    trajectory_path: Optional[Path] = None
    intrinsics_path: Optional[Path] = None

    success: bool = True
    errors: List[str] = field(default_factory=list)


class CaptureExporter:
    """Export capture results for BlueprintPipeline/DWM handoff.

    Creates a minimal output format containing:
    - 3D Gaussians (PLY format) from SLAM
    - Camera trajectory (poses)
    - Camera intrinsics
    - Capture metadata
    """

    def export(
        self,
        manifest: CaptureManifest,
        gaussians_path: Optional[Path],
        poses: List[CameraPose],
        intrinsics: Optional[CameraIntrinsics],
        output_dir: Path,
        scale_factor: float = 1.0,
        copy_frames: bool = False,
        frames_dir: Optional[Path] = None,
    ) -> CaptureExportResult:
        """Export capture for DWM processing.

        Args:
            manifest: Capture manifest with metadata
            gaussians_path: Path to 3D Gaussians PLY file
            poses: Camera poses from SLAM
            intrinsics: Camera intrinsics
            output_dir: Output directory
            scale_factor: Scale factor applied during reconstruction
            copy_frames: Whether to copy keyframes to output
            frames_dir: Source directory for frames (if copying)

        Returns:
            CaptureExportResult with paths to exported files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        result = CaptureExportResult(output_path=output_dir)

        # Export 3D Gaussians
        if gaussians_path and gaussians_path.exists():
            dst_gaussians = output_dir / "gaussians.ply"
            shutil.copy(gaussians_path, dst_gaussians)
            result.gaussians_path = dst_gaussians
        else:
            result.errors.append("No Gaussians file found")
            result.success = False

        # Export camera data
        camera_dir = output_dir / "camera"
        camera_dir.mkdir(exist_ok=True)

        # Intrinsics
        if intrinsics:
            intrinsics_data = {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
                "width": intrinsics.width,
                "height": intrinsics.height,
                "camera_model": getattr(intrinsics, 'camera_model', 'PINHOLE'),
            }
            intrinsics_path = camera_dir / "intrinsics.json"
            intrinsics_path.write_text(json.dumps(intrinsics_data, indent=2))
            result.intrinsics_path = intrinsics_path

        # Trajectory (camera poses)
        if poses:
            trajectory = [
                {
                    "frame_id": p.frame_id,
                    "image_name": p.image_name,
                    "rotation": list(p.rotation),  # Quaternion (w, x, y, z)
                    "translation": list(p.translation),
                    "timestamp": p.timestamp,
                }
                for p in poses
            ]
            trajectory_path = camera_dir / "trajectory.json"
            trajectory_path.write_text(json.dumps({
                "poses": trajectory,
                "coordinate_system": "colmap",  # World-to-camera convention
                "scale_factor": scale_factor,
            }, indent=2))
            result.trajectory_path = trajectory_path

        # Copy keyframes if requested
        if copy_frames and frames_dir and frames_dir.exists():
            frames_output = output_dir / "frames"
            frames_output.mkdir(exist_ok=True)
            for pose in poses:
                src = frames_dir / pose.image_name
                if src.exists():
                    shutil.copy(src, frames_output / pose.image_name)

        # Create capture info (metadata for handoff)
        capture_info = self._create_capture_info(
            manifest=manifest,
            pose_count=len(poses),
            has_gaussians=result.gaussians_path is not None,
            scale_factor=scale_factor,
        )
        info_path = output_dir / "capture_info.json"
        info_path.write_text(json.dumps(capture_info, indent=2))

        # Write completion marker
        (output_dir / ".complete").touch()

        return result

    def _create_capture_info(
        self,
        manifest: CaptureManifest,
        pose_count: int,
        has_gaussians: bool,
        scale_factor: float,
    ) -> Dict[str, Any]:
        """Create capture_info.json for BlueprintPipeline handoff."""
        return {
            # Identification
            "capture_id": manifest.capture_id,
            "capture_timestamp": manifest.capture_timestamp,

            # Source info
            "source": "BlueprintCapturePipeline",
            "version": "1.0",

            # Device metadata
            "device": {
                "platform": manifest.device_platform,
                "model": manifest.device_model,
            },

            # Sensor configuration
            "sensor": {
                "type": manifest.sensor_type.value,
                "has_depth": manifest.has_depth,
                "has_imu": manifest.has_imu,
                "has_arkit_poses": manifest.has_arkit_poses,
            },

            # Capture metrics
            "metrics": {
                "total_frames": manifest.total_frames,
                "pose_count": pose_count,
                "duration_seconds": manifest.estimated_duration_seconds,
                "resolution": list(manifest.resolution),
                "fps": manifest.fps,
            },

            # Reconstruction results
            "reconstruction": {
                "has_gaussians": has_gaussians,
                "gaussians_format": "3dgs_ply",
                "scale_factor": scale_factor,
                "coordinate_system": "colmap",
            },

            # DWM compatibility
            "dwm_ready": has_gaussians and pose_count > 0,

            # Handoff status
            "ready_for_pipeline": has_gaussians,
        }


def export_capture(
    manifest: CaptureManifest,
    gaussians_path: Optional[Path],
    poses: List[CameraPose],
    intrinsics: Optional[CameraIntrinsics],
    output_dir: Path,
    **kwargs,
) -> CaptureExportResult:
    """Convenience function to export capture.

    Args:
        manifest: Capture manifest
        gaussians_path: Path to Gaussians PLY
        poses: Camera poses
        intrinsics: Camera intrinsics
        output_dir: Output directory
        **kwargs: Additional arguments passed to CaptureExporter.export()

    Returns:
        CaptureExportResult
    """
    exporter = CaptureExporter()
    return exporter.export(
        manifest=manifest,
        gaussians_path=gaussians_path,
        poses=poses,
        intrinsics=intrinsics,
        output_dir=output_dir,
        **kwargs,
    )
