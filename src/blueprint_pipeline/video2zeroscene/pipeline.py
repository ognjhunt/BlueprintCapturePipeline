"""BlueprintCapture Pipeline - Video to Gaussian + DWM-ready output.

This module provides the main pipeline for Phase 3: Capture.

Pipeline stages:
    0. Ingest: Video → CaptureManifest + keyframes
    1. SLAM: Pose estimation + 3D Gaussian reconstruction
    2. Export: Gaussians + camera data for BlueprintPipeline/DWM handoff

The pipeline is designed to work with:
- RGB-only captures (Meta glasses, generic cameras)
- RGB-D captures (iPhone LiDAR)
- iOS ARKit captures (direct pose import)

Output is passed to BlueprintPipeline for DWM (Dexterous World Models) processing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .interfaces import (
    CaptureManifest,
    PipelineConfig,
)
from .ingest import VideoIngestor, IngestResult
from .slam import get_slam_backend, SLAMResult, CameraPose
from .export import CaptureExporter, CaptureExportResult


@dataclass
class CaptureResult:
    """Complete capture pipeline result."""
    capture_id: str
    output_path: Optional[Path] = None

    # Stage results
    ingest_result: Optional[IngestResult] = None
    slam_result: Optional[SLAMResult] = None
    export_result: Optional[CaptureExportResult] = None

    # Summary metrics
    total_frames: int = 0
    keyframe_count: int = 0
    registered_frames: int = 0
    registration_rate: float = 0.0

    # Status
    success: bool = True
    errors: List[str] = field(default_factory=list)

    # DWM readiness
    dwm_ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "capture_id": self.capture_id,
            "output_path": str(self.output_path) if self.output_path else None,
            "metrics": {
                "total_frames": self.total_frames,
                "keyframe_count": self.keyframe_count,
                "registered_frames": self.registered_frames,
                "registration_rate": self.registration_rate,
            },
            "success": self.success,
            "dwm_ready": self.dwm_ready,
            "errors": self.errors,
        }


class CapturePipeline:
    """Main pipeline for video → Gaussian + DWM-ready output.

    This is the core of Phase 3: Capture. It converts video walkthroughs
    into high-quality 3D Gaussian representations ready for DWM processing
    in BlueprintPipeline.

    Stages:
        0. Ingest - Video normalization, keyframe selection
        1. SLAM - Pose estimation + 3D Gaussian reconstruction
        2. Export - Package for BlueprintPipeline handoff
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize stage processors
        self.ingestor = VideoIngestor(self.config)
        self.exporter = CaptureExporter()

    def run(
        self,
        capture_id: str,
        video_paths: List[Path],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
        arkit_data_path: Optional[Path] = None,
        depth_path: Optional[Path] = None,
        imu_path: Optional[Path] = None,
        copy_frames: bool = False,
    ) -> CaptureResult:
        """Run the capture pipeline.

        Args:
            capture_id: Unique identifier for this capture
            video_paths: Paths to video files
            output_dir: Output directory for all artifacts
            metadata: Optional device/capture metadata
            arkit_data_path: Optional path to ARKit poses (iOS)
            depth_path: Optional path to depth frames
            imu_path: Optional path to IMU data
            copy_frames: Whether to include keyframes in export

        Returns:
            CaptureResult with Gaussian + camera data ready for DWM
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        result = CaptureResult(capture_id=capture_id)

        print(f"\n{'='*60}")
        print(f"BlueprintCapture Pipeline - {capture_id}")
        print(f"{'='*60}\n")

        # Stage 0: Ingest
        print("\n[Stage 0] Ingesting video capture...")
        ingest_dir = output_dir / "ingest"
        ingest_result = self.ingestor.ingest(
            capture_id=capture_id,
            video_paths=video_paths,
            output_dir=ingest_dir,
            metadata=metadata,
            arkit_data_path=arkit_data_path,
            depth_path=depth_path,
            imu_path=imu_path,
        )
        result.ingest_result = ingest_result
        result.total_frames = len(ingest_result.frames)
        result.keyframe_count = len(ingest_result.keyframes)

        if not ingest_result.success:
            result.success = False
            result.errors.append("Ingest failed")
            return result

        manifest = ingest_result.manifest
        keyframes = ingest_result.keyframes
        frames_dir = ingest_result.frames_dir

        print(f"  Extracted {len(ingest_result.frames)} frames")
        print(f"  Selected {len(keyframes)} keyframes")
        print(f"  Sensor type: {manifest.sensor_type.value}")

        # Stage 1: SLAM (Gaussian reconstruction)
        print("\n[Stage 1] Running SLAM reconstruction...")
        slam_backend = self.config.select_slam_backend(manifest)
        print(f"  Selected backend: {slam_backend.value}")

        slam_dir = output_dir / "slam"
        slam = get_slam_backend(slam_backend, self.config)
        slam_result = slam.run(
            manifest=manifest,
            keyframes=keyframes,
            frames_dir=frames_dir,
            output_dir=slam_dir,
            dynamic_masks=None,  # No object tracking in simplified pipeline
            scale_observations=ingest_result.scale_observations,  # Pass ArUco/AprilTag observations
        )
        result.slam_result = slam_result
        result.registered_frames = len(slam_result.poses)
        result.registration_rate = slam_result.registration_rate

        if not slam_result.success:
            result.success = False
            result.errors.extend(slam_result.errors)
            return result

        print(f"  Registered {len(slam_result.poses)}/{len(keyframes)} frames")
        print(f"  Registration rate: {slam_result.registration_rate:.1%}")
        if slam_result.gaussians_path:
            print(f"  Gaussians: {slam_result.gaussians_path}")

        # Stage 2: Export for DWM
        print("\n[Stage 2] Exporting for DWM processing...")
        export_dir = output_dir / "output"
        export_result = self.exporter.export(
            manifest=manifest,
            gaussians_path=slam_result.gaussians_path,
            poses=slam_result.poses,
            intrinsics=manifest.intrinsics,
            output_dir=export_dir,
            scale_factor=slam_result.scale_factor,
            copy_frames=copy_frames,
            frames_dir=frames_dir,
        )
        result.export_result = export_result
        result.output_path = export_result.output_path

        if not export_result.success:
            result.success = False
            result.errors.extend(export_result.errors)

        # Check DWM readiness
        result.dwm_ready = (
            export_result.gaussians_path is not None
            and export_result.trajectory_path is not None
        )

        # Save pipeline summary
        summary_path = output_dir / "pipeline_summary.json"
        summary_path.write_text(json.dumps(result.to_dict(), indent=2))

        print(f"\n{'='*60}")
        print(f"Pipeline complete!")
        print(f"  Output: {result.output_path}")
        print(f"  DWM ready: {result.dwm_ready}")
        print(f"  Success: {result.success}")
        print(f"{'='*60}\n")

        return result

    def run_from_manifest(
        self,
        manifest_path: Path,
        output_dir: Path,
    ) -> CaptureResult:
        """Run pipeline from an existing CaptureManifest.

        Args:
            manifest_path: Path to capture_manifest.json
            output_dir: Output directory

        Returns:
            CaptureResult
        """
        manifest_data = json.loads(manifest_path.read_text())
        manifest = CaptureManifest.from_dict(manifest_data)

        video_paths = [Path(c["uri"]) for c in manifest.clips]

        return self.run(
            capture_id=manifest.capture_id,
            video_paths=video_paths,
            output_dir=output_dir,
            metadata={
                "platform": manifest.device_platform,
                "model": manifest.device_model,
            },
            arkit_data_path=Path(manifest.arkit_poses_path) if manifest.arkit_poses_path else None,
            depth_path=Path(manifest.depth_frames_path) if manifest.depth_frames_path else None,
            imu_path=Path(manifest.imu_data_path) if manifest.imu_data_path else None,
        )


def run_capture_pipeline(
    video_paths: List[Path],
    output_dir: Path,
    capture_id: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
    **kwargs,
) -> CaptureResult:
    """Convenience function to run the capture pipeline.

    Args:
        video_paths: List of video file paths
        output_dir: Output directory
        capture_id: Optional capture identifier (auto-generated if not provided)
        config: Optional pipeline configuration
        **kwargs: Additional arguments passed to pipeline.run()

    Returns:
        CaptureResult with Gaussian + camera data for DWM
    """
    import uuid
    from datetime import datetime

    if capture_id is None:
        capture_id = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    pipeline = CapturePipeline(config)
    return pipeline.run(
        capture_id=capture_id,
        video_paths=video_paths,
        output_dir=output_dir,
        **kwargs,
    )


# Alias for backward compatibility
Video2ZeroScenePipeline = CapturePipeline
