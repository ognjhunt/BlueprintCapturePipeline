"""BlueprintCapture Pipeline - Video to Gaussian + DWM-ready output.

This module provides the main pipeline for Phase 3: Capture.

Pipeline stages:
    0. Ingest: Video → CaptureManifest + keyframes
    1. SLAM: Pose estimation + 3D Gaussian reconstruction
    1.5. Difix Refinement: Scene inpainting + quality enhancement (optional)
    2. Export: Gaussians + camera data for BlueprintPipeline/DWM handoff

The pipeline is designed to work with:
- RGB-only captures (Meta glasses, generic cameras)
- RGB-D captures (iPhone LiDAR)
- iOS ARKit captures (direct pose import)

Scene Inpainting (Difix3D+):
    When enabled, the pipeline uses NVIDIA's Difix3D+ (CVPR 2025) to:
    - Fill gaps in sparse captures
    - Remove artifacts from 3DGS reconstructions
    - Generate higher-quality renders through progressive refinement

    This is especially useful for:
    - Meta Glasses captures (RGB-only, no LiDAR)
    - Quick walk-through captures with incomplete coverage
    - Any capture where visual quality needs enhancement

Output is passed to BlueprintPipeline for DWM (Dexterous World Models) processing.
"""

from __future__ import annotations

import json
import logging
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

# Difix3D+ integration for scene inpainting
try:
    from ..reconstruction.difix_refinement import (
        DifixConfig,
        DifixPipeline,
        GapDetector,
    )
    DIFIX_AVAILABLE = True
except ImportError:
    DIFIX_AVAILABLE = False
    DifixConfig = None
    DifixPipeline = None

logger = logging.getLogger(__name__)


@dataclass
class DifixRefinementResult:
    """Result of Difix3D+ scene inpainting refinement."""
    success: bool = True
    enabled: bool = False

    # Refinement metrics
    rounds_completed: int = 0
    gaps_detected: int = 0
    pseudo_views_generated: int = 0

    # Output paths
    refined_gaussians_path: Optional[Path] = None

    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "enabled": self.enabled,
            "rounds_completed": self.rounds_completed,
            "gaps_detected": self.gaps_detected,
            "pseudo_views_generated": self.pseudo_views_generated,
            "refined_gaussians_path": str(self.refined_gaussians_path) if self.refined_gaussians_path else None,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class CaptureResult:
    """Complete capture pipeline result."""
    capture_id: str
    output_path: Optional[Path] = None

    # Stage results
    ingest_result: Optional[IngestResult] = None
    slam_result: Optional[SLAMResult] = None
    difix_result: Optional[DifixRefinementResult] = None  # Scene inpainting
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

    # Quality enhancement flags
    difix_refined: bool = False

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
            "difix_refinement": self.difix_result.to_dict() if self.difix_result else None,
            "success": self.success,
            "dwm_ready": self.dwm_ready,
            "difix_refined": self.difix_refined,
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
        1.5. Difix Refinement - Scene inpainting + quality enhancement (optional)
        2. Export - Package for BlueprintPipeline handoff

    Scene Inpainting (Difix3D+):
        When enable_difix_refinement=True, adds a refinement stage that uses
        NVIDIA's Difix3D+ to enhance the 3DGS reconstruction:

        - Detects gaps/artifacts in the initial reconstruction
        - Generates novel views via pose interpolation
        - Uses single-step diffusion to "fix" degraded renders
        - Distills enhanced views back into the 3DGS model
        - Progressively expands to harder viewpoints

        This significantly improves quality for:
        - RGB-only captures (no LiDAR/depth)
        - Sparse captures with incomplete coverage
        - Any capture where visual quality needs enhancement
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
        enable_difix_refinement: bool = True,
        difix_config: Optional["DifixConfig"] = None,
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
            enable_difix_refinement: Enable Difix3D+ scene inpainting (default True)
            difix_config: Optional DifixConfig for refinement settings

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

        # Stage 1.5: Difix3D+ Refinement (Scene Inpainting)
        gaussians_path = slam_result.gaussians_path
        difix_result = DifixRefinementResult(enabled=enable_difix_refinement)

        if enable_difix_refinement and slam_result.gaussians_path:
            print("\n[Stage 1.5] Running Difix3D+ scene inpainting...")

            if not DIFIX_AVAILABLE:
                print("  WARNING: Difix3D+ not available (missing dependencies)")
                print("  Install with: pip install diffusers transformers accelerate lpips")
                difix_result.warnings.append("Difix3D+ not available - skipped refinement")
            else:
                try:
                    difix_result = self._run_difix_refinement(
                        gaussians_path=slam_result.gaussians_path,
                        slam_result=slam_result,
                        manifest=manifest,
                        keyframes=keyframes,
                        frames_dir=frames_dir,
                        output_dir=output_dir / "difix",
                        difix_config=difix_config,
                    )

                    if difix_result.success and difix_result.refined_gaussians_path:
                        # Use refined Gaussians for export
                        gaussians_path = difix_result.refined_gaussians_path
                        result.difix_refined = True
                        print(f"  Refinement complete: {difix_result.rounds_completed} rounds")
                        print(f"  Gaps filled: {difix_result.gaps_detected}")
                        print(f"  Pseudo-views generated: {difix_result.pseudo_views_generated}")
                    else:
                        print(f"  Refinement had issues: {difix_result.errors}")

                except Exception as e:
                    logger.error(f"Difix refinement failed: {e}")
                    difix_result.success = False
                    difix_result.errors.append(str(e))
                    print(f"  ERROR: Difix refinement failed: {e}")
                    print("  Continuing with unrefined Gaussians...")

        result.difix_result = difix_result

        # Stage 2: Export for DWM
        print("\n[Stage 2] Exporting for DWM processing...")
        export_dir = output_dir / "output"

        # Determine scale source and metric status
        scale_source = "unknown"
        is_metric = False
        if manifest.has_arkit_poses:
            scale_source = "arkit"
            is_metric = True
        elif manifest.has_depth:
            scale_source = "lidar_depth"
            is_metric = True
        elif slam_result.scale_confidence > 0:
            scale_source = "metric_depth_recovery"
            is_metric = slam_result.scale_confidence >= 0.5

        export_result = self.exporter.export(
            manifest=manifest,
            gaussians_path=gaussians_path,  # Use refined if available
            poses=slam_result.poses,
            intrinsics=manifest.intrinsics,
            output_dir=export_dir,
            scale_factor=slam_result.scale_factor,
            scale_confidence=slam_result.scale_confidence,
            scale_source=scale_source,
            is_metric=is_metric,
            copy_frames=copy_frames,
            frames_dir=frames_dir,
        )
        result.export_result = export_result
        result.output_path = export_result.output_path

        if not export_result.success:
            result.success = False
            result.errors.extend(export_result.errors)

        # Check DWM readiness (now includes metric scale check)
        has_metric_scale = (
            is_metric or
            manifest.has_arkit_poses or
            manifest.has_depth or
            slam_result.scale_confidence >= 0.5
        )
        result.dwm_ready = (
            export_result.gaussians_path is not None
            and export_result.trajectory_path is not None
            and has_metric_scale
        )

        # Print scale status
        if is_metric:
            print(f"  ✓ Metric scale achieved via {scale_source}")
        elif slam_result.scale_confidence > 0:
            print(f"  ⚠ Scale confidence: {slam_result.scale_confidence:.2f} (source: {scale_source})")

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

    def _run_difix_refinement(
        self,
        gaussians_path: Path,
        slam_result: SLAMResult,
        manifest: CaptureManifest,
        keyframes: List[Any],
        frames_dir: Path,
        output_dir: Path,
        difix_config: Optional["DifixConfig"] = None,
    ) -> DifixRefinementResult:
        """Run Difix3D+ scene inpainting refinement.

        This is Stage 1.5 of the pipeline. It takes the initial 3DGS
        reconstruction and enhances it using progressive distillation
        with NVIDIA's Difix3D+ diffusion model.

        Args:
            gaussians_path: Path to initial Gaussians PLY
            slam_result: SLAM result with poses
            manifest: Capture manifest with intrinsics
            keyframes: List of keyframe metadata
            frames_dir: Directory containing frame images
            output_dir: Output directory for refined results
            difix_config: Optional configuration for refinement

        Returns:
            DifixRefinementResult with refined Gaussians path
        """
        from ..reconstruction.gaussian_splatting import GaussianModel

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = DifixRefinementResult(enabled=True)

        try:
            # Load the Gaussian model
            logger.info(f"Loading Gaussians from {gaussians_path}")
            gaussian_model = GaussianModel.load_ply(gaussians_path)

            # Build training views from keyframes + SLAM poses
            training_views = self._build_training_views(
                keyframes=keyframes,
                poses=slam_result.poses,
                frames_dir=frames_dir,
                manifest=manifest,
            )

            if not training_views:
                result.success = False
                result.errors.append("No valid training views found")
                return result

            # Build intrinsics dict
            intrinsics = {}
            if manifest.intrinsics:
                intrinsics = {
                    'fx': manifest.intrinsics.fx,
                    'fy': manifest.intrinsics.fy,
                    'cx': manifest.intrinsics.cx,
                    'cy': manifest.intrinsics.cy,
                    'width': manifest.intrinsics.width,
                    'height': manifest.intrinsics.height,
                }

            # Initialize Difix pipeline with config settings
            if difix_config is None:
                # Build DifixConfig from PipelineConfig settings (full parameter mapping)
                difix_config = DifixConfig(
                    # Model settings
                    model_name=self.config.difix_model_name,
                    device=self.config.difix_device,
                    dtype=self.config.difix_dtype,
                    # Progressive refinement
                    num_refinement_rounds=self.config.difix_num_rounds,
                    poses_per_round=self.config.difix_poses_per_round,
                    pose_interpolation_steps=self.config.difix_pose_interpolation_steps,
                    progressive_expansion_rate=self.config.difix_progressive_expansion_rate,
                    # Distillation
                    distillation_weight=self.config.difix_distillation_weight,
                    iterations_per_round=self.config.difix_iterations_per_round,
                    distillation_lr=self.config.difix_distillation_lr,
                    # Loss weights
                    l2_weight=self.config.difix_l2_weight,
                    lpips_weight=self.config.difix_lpips_weight,
                    gram_weight=self.config.difix_gram_weight,
                    ssim_weight=self.config.difix_ssim_weight,
                    # Quality thresholds
                    coverage_threshold=self.config.difix_coverage_threshold,
                    artifact_threshold=self.config.difix_artifact_threshold,
                    # Inference settings
                    difix_timestep=self.config.difix_timestep,
                    guidance_scale=self.config.difix_guidance_scale,
                    # Post-processing
                    enable_post_process=self.config.difix_enable_post_process,
                    post_process_strength=self.config.difix_post_process_strength,
                    # Output settings
                    save_intermediate=self.config.difix_save_intermediate,
                    output_resolution=self.config.difix_output_resolution,
                    # Prompt
                    difix_prompt=self.config.difix_prompt,
                )

            difix_pipeline = DifixPipeline(difix_config)

            # Track metrics during refinement
            metrics_tracker = {'gaps': 0, 'pseudo_views': 0}

            def progress_callback(round_idx, total_rounds, round_metrics):
                metrics_tracker['gaps'] += round_metrics.get('gaps_found', 0)
                metrics_tracker['pseudo_views'] += round_metrics.get('pseudo_views', 0)
                logger.info(
                    f"Difix round {round_idx + 1}/{total_rounds}: "
                    f"gaps={round_metrics.get('gaps_found', 0)}, "
                    f"pseudo_views={round_metrics.get('pseudo_views', 0)}"
                )

            # Run refinement
            refined_model = difix_pipeline.refine(
                gaussian_model=gaussian_model,
                training_views=training_views,
                intrinsics=intrinsics,
                output_dir=output_dir,
                progress_callback=progress_callback,
            )

            # Update result
            result.success = True
            result.rounds_completed = difix_config.num_refinement_rounds
            result.gaps_detected = metrics_tracker['gaps']
            result.pseudo_views_generated = metrics_tracker['pseudo_views']
            result.refined_gaussians_path = output_dir / "refined_gaussians.ply"

            logger.info(f"Difix refinement complete: {result.refined_gaussians_path}")

        except Exception as e:
            logger.error(f"Difix refinement failed: {e}")
            result.success = False
            result.errors.append(str(e))

        return result

    def _build_training_views(
        self,
        keyframes: List[Any],
        poses: List[CameraPose],
        frames_dir: Path,
        manifest: CaptureManifest,
    ) -> List[Dict[str, Any]]:
        """Build training view dicts from keyframes and SLAM poses.

        Args:
            keyframes: List of FrameMetadata
            poses: List of CameraPose from SLAM
            frames_dir: Directory containing frame images
            manifest: Capture manifest with intrinsics

        Returns:
            List of training view dicts with pose and image
        """
        import numpy as np

        try:
            from PIL import Image
            import torch
        except ImportError:
            logger.warning("PIL or torch not available for loading training views")
            return []

        # Build pose lookup by frame_id
        pose_by_frame = {p.frame_id: p for p in poses}

        training_views = []

        for kf in keyframes:
            frame_id = kf.frame_id
            if frame_id not in pose_by_frame:
                continue

            pose = pose_by_frame[frame_id]

            # Build 4x4 world-to-camera matrix from quaternion + translation
            qw, qx, qy, qz = pose.rotation
            tx, ty, tz = pose.translation

            # Quaternion to rotation matrix
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ])

            world_to_camera = np.eye(4)
            world_to_camera[:3, :3] = R
            world_to_camera[:3, 3] = [tx, ty, tz]

            # Load image
            img_path = frames_dir / kf.file_path if not Path(kf.file_path).is_absolute() else Path(kf.file_path)
            if not img_path.exists():
                img_path = frames_dir / Path(kf.file_path).name

            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = torch.tensor(np.array(img)).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
            except Exception as e:
                logger.warning(f"Could not load image {img_path}: {e}")
                continue

            # Build view dict
            view = {
                'frame_id': frame_id,
                'image': img_tensor,
                'pose': {
                    'world_to_camera': world_to_camera.tolist(),
                },
            }

            # Add intrinsics
            if manifest.intrinsics:
                view['fx'] = manifest.intrinsics.fx
                view['fy'] = manifest.intrinsics.fy
                view['cx'] = manifest.intrinsics.cx
                view['cy'] = manifest.intrinsics.cy
                view['image_height'] = manifest.intrinsics.height
                view['image_width'] = manifest.intrinsics.width

            training_views.append(view)

        logger.info(f"Built {len(training_views)} training views for Difix refinement")
        return training_views

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
