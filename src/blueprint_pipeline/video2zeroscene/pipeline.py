"""Main Video2ZeroScene pipeline orchestration.

This module provides the high-level pipeline that orchestrates all stages:
0. Ingest: Video → CaptureManifest + keyframes
1. Quality filtering and keyframe selection
2. SLAM: Sensor-conditional pose estimation + 3DGS
3. Mesh: SuGaR extraction + decimation
4. Tracks: SAM3 concept segmentation
5. Lift: 2D tracks → 3D proposals
6. Assetize: Tiered object asset generation
7. Export: ZeroScene bundle for BlueprintPipeline

Designed to:
- Work with Meta glasses (RGB-only)
- Scale from room to grocery store
- Output ZeroScene format for downstream processing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .interfaces import (
    CaptureManifest,
    PipelineConfig,
    SLAMBackend,
)
from .ingest import VideoIngestor, IngestResult
from .slam import get_slam_backend, SLAMResult
from .mesh import MeshExtractor, MeshResult
from .tracks import SAM3Tracker, TracksResult
from .lift import ObjectLifter, LiftResult
from .assetize import ObjectAssetizer, AssetizationResult
from .export import ZeroSceneExporter, ExportResult


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    capture_id: str
    zeroscene_path: Optional[Path] = None

    # Stage results
    ingest_result: Optional[IngestResult] = None
    slam_result: Optional[SLAMResult] = None
    mesh_result: Optional[MeshResult] = None
    tracks_result: Optional[TracksResult] = None
    lift_result: Optional[LiftResult] = None
    assetization_result: Optional[AssetizationResult] = None
    export_result: Optional[ExportResult] = None

    # Summary metrics
    total_frames: int = 0
    keyframe_count: int = 0
    object_count: int = 0
    registration_rate: float = 0.0

    success: bool = True
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "capture_id": self.capture_id,
            "zeroscene_path": str(self.zeroscene_path) if self.zeroscene_path else None,
            "metrics": {
                "total_frames": self.total_frames,
                "keyframe_count": self.keyframe_count,
                "object_count": self.object_count,
                "registration_rate": self.registration_rate,
            },
            "success": self.success,
            "errors": self.errors,
        }


class Video2ZeroScenePipeline:
    """Main pipeline for video → ZeroScene conversion.

    This orchestrates all stages of the pipeline with:
    - Sensor-conditional SLAM backend selection
    - Submap chunking for large spaces
    - Tiered object assetization
    - ZeroScene export for BlueprintPipeline
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize stage processors
        self.ingestor = VideoIngestor(self.config)
        self.mesh_extractor = MeshExtractor(self.config)
        self.tracker = SAM3Tracker(self.config)
        self.lifter = ObjectLifter(self.config)
        self.assetizer = ObjectAssetizer(self.config)
        self.exporter = ZeroSceneExporter(self.config)

    def run(
        self,
        capture_id: str,
        video_paths: List[Path],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
        arkit_data_path: Optional[Path] = None,
        depth_path: Optional[Path] = None,
        imu_path: Optional[Path] = None,
    ) -> PipelineResult:
        """Run the complete video → ZeroScene pipeline.

        Args:
            capture_id: Unique identifier for this capture
            video_paths: Paths to video files
            output_dir: Output directory for all artifacts
            metadata: Optional device/capture metadata
            arkit_data_path: Optional path to ARKit poses (iOS)
            depth_path: Optional path to depth frames
            imu_path: Optional path to IMU data

        Returns:
            PipelineResult with all stage results and ZeroScene path
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        result = PipelineResult(capture_id=capture_id)

        print(f"\n{'='*60}")
        print(f"Video2ZeroScene Pipeline - {capture_id}")
        print(f"{'='*60}\n")

        # Stage 0: Ingest
        print("\n[Stage 0] Ingesting video capture...")
        ingest_dir = output_dir / "stage0_ingest"
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

        # Stage 1: SAM3 Tracking (run before SLAM to get dynamic masks)
        print("\n[Stage 4-early] Running SAM3 tracking for dynamic masks...")
        tracks_dir = output_dir / "stage4_tracks"
        tracks_result = self.tracker.run(
            manifest=manifest,
            keyframes=keyframes,
            frames_dir=frames_dir,
            output_dir=tracks_dir,
        )
        result.tracks_result = tracks_result

        print(f"  Found {len(tracks_result.tracks)} object tracks")
        print(f"  Dynamic masks for {len(tracks_result.dynamic_mask_paths)} frames")

        # Stage 2: SLAM
        print("\n[Stage 2] Running SLAM reconstruction...")
        slam_backend = self.config.select_slam_backend(manifest)
        print(f"  Selected backend: {slam_backend.value}")

        slam_dir = output_dir / "stage2_slam"
        slam = get_slam_backend(slam_backend, self.config)
        slam_result = slam.run(
            manifest=manifest,
            keyframes=keyframes,
            frames_dir=frames_dir,
            output_dir=slam_dir,
            dynamic_masks=tracks_result.dynamic_mask_paths,
        )
        result.slam_result = slam_result
        result.registration_rate = slam_result.registration_rate

        if not slam_result.success:
            result.success = False
            result.errors.extend(slam_result.errors)
            return result

        print(f"  Registered {len(slam_result.poses)}/{len(keyframes)} frames")
        print(f"  Registration rate: {slam_result.registration_rate:.1%}")

        # Stage 3: Mesh extraction
        print("\n[Stage 3] Extracting environment mesh...")
        mesh_dir = output_dir / "stage3_mesh"

        mesh_result = None
        if slam_result.gaussians_path:
            mesh_result = self.mesh_extractor.run(
                gaussians_path=slam_result.gaussians_path,
                poses=slam_result.poses,
                intrinsics=manifest.intrinsics,
                frames_dir=frames_dir,
                output_dir=mesh_dir,
            )
            result.mesh_result = mesh_result

            if mesh_result.success:
                print(f"  Mesh vertices: {mesh_result.vertex_count}")
                print(f"  Mesh faces: {mesh_result.face_count}")
        else:
            print("  Skipping mesh extraction (no Gaussians)")

        # Stage 5: Lift 2D tracks to 3D
        print("\n[Stage 5] Lifting 2D tracks to 3D proposals...")
        lift_dir = output_dir / "stage5_lift"
        lift_result = self.lifter.run(
            tracks=tracks_result.tracks,
            poses=slam_result.poses,
            intrinsics=manifest.intrinsics,
            frames_dir=frames_dir,
            masks_dir=tracks_dir / "masks",
            mesh_path=mesh_result.render_mesh_path if mesh_result else None,
            output_dir=lift_dir,
        )
        result.lift_result = lift_result

        print(f"  Generated {len(lift_result.proposals)} 3D object proposals")

        # Stage 6: Assetization
        print("\n[Stage 6] Generating object assets...")
        assets_dir = output_dir / "stage6_assets"
        assetization_result = self.assetizer.run(
            proposals=lift_result.proposals,
            tracks=tracks_result.tracks,
            poses=slam_result.poses,
            intrinsics=manifest.intrinsics,
            frames_dir=frames_dir,
            masks_dir=tracks_dir / "masks",
            output_dir=assets_dir,
        )
        result.assetization_result = assetization_result
        result.object_count = len(assetization_result.assets)

        tier_counts = {}
        for asset in assetization_result.assets:
            tier_counts[asset.tier.value] = tier_counts.get(asset.tier.value, 0) + 1
        print(f"  Generated {len(assetization_result.assets)} object assets")
        for tier, count in tier_counts.items():
            print(f"    {tier}: {count}")

        # Stage 7: Export to ZeroScene
        print("\n[Stage 7] Exporting to ZeroScene format...")
        export_result = self.exporter.export(
            manifest=manifest,
            background_mesh_path=mesh_result.render_mesh_path if mesh_result else None,
            collision_mesh_path=mesh_result.collision_mesh_path if mesh_result else None,
            gaussians_path=slam_result.gaussians_path,
            objects=assetization_result.assets,
            poses=slam_result.poses,
            intrinsics=manifest.intrinsics,
            output_dir=output_dir,
            scale_factor=slam_result.scale_factor,
        )
        result.export_result = export_result
        result.zeroscene_path = export_result.bundle_path

        if export_result.success:
            print(f"  ZeroScene bundle: {export_result.bundle_path}")
            print(f"  Objects exported: {export_result.object_count}")
        else:
            result.errors.extend(export_result.errors or [])

        # Save pipeline summary
        summary_path = output_dir / "pipeline_summary.json"
        summary_path.write_text(json.dumps(result.to_dict(), indent=2))

        print(f"\n{'='*60}")
        print(f"Pipeline complete!")
        print(f"  ZeroScene: {result.zeroscene_path}")
        print(f"  Objects: {result.object_count}")
        print(f"  Success: {result.success}")
        print(f"{'='*60}\n")

        return result

    def run_from_manifest(
        self,
        manifest_path: Path,
        output_dir: Path,
    ) -> PipelineResult:
        """Run pipeline from an existing CaptureManifest.

        Args:
            manifest_path: Path to capture_manifest.json
            output_dir: Output directory

        Returns:
            PipelineResult
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


def run_video2zeroscene(
    video_paths: List[Path],
    output_dir: Path,
    capture_id: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
    **kwargs,
) -> PipelineResult:
    """Convenience function to run the video2zeroscene pipeline.

    Args:
        video_paths: List of video file paths
        output_dir: Output directory
        capture_id: Optional capture identifier (auto-generated if not provided)
        config: Optional pipeline configuration
        **kwargs: Additional arguments passed to pipeline.run()

    Returns:
        PipelineResult
    """
    import uuid
    from datetime import datetime

    if capture_id is None:
        capture_id = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    pipeline = Video2ZeroScenePipeline(config)
    return pipeline.run(
        capture_id=capture_id,
        video_paths=video_paths,
        output_dir=output_dir,
        **kwargs,
    )
