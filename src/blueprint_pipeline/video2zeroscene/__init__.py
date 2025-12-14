"""Video to ZeroScene conversion module.

This module implements the complete pipeline for converting walkthrough video
captures into ZeroScene-compatible outputs that can be handed off to BlueprintPipeline.

The pipeline is designed to work with:
- RGB-only captures (Meta glasses default)
- RGB-D captures (iPhone LiDAR)
- Visual-inertial captures (with synchronized IMU)

Key components:
- ingest: Video normalization and CaptureManifest creation
- slam: Sensor-conditional SLAM (WildGS-SLAM, SplaTAM, VIGS-SLAM)
- mesh: SuGaR mesh extraction and decimation
- tracks: SAM3 concept segmentation and video tracking
- lift: 2D-to-3D instance proposal generation
- assetize: Tiered object asset generation (reconstruction/proxy/replacement)
- export: ZeroScene bundle export for BlueprintPipeline handoff
"""

from .interfaces import (
    CaptureManifest,
    SensorType,
    PipelineConfig,
    ZeroSceneBundle,
    ObjectProposal,
    TrackInfo,
    Submap,
)

from .pipeline import Video2ZeroScenePipeline

__all__ = [
    "CaptureManifest",
    "SensorType",
    "PipelineConfig",
    "ZeroSceneBundle",
    "ObjectProposal",
    "TrackInfo",
    "Submap",
    "Video2ZeroScenePipeline",
]
