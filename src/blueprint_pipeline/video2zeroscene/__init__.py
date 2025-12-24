"""BlueprintCapture Pipeline - Video to Gaussian for DWM.

This module implements the Phase 3: Capture pipeline for converting
walkthrough video captures into high-quality 3D Gaussian representations
ready for DWM (Dexterous World Models) processing in BlueprintPipeline.

Pipeline stages:
    0. Ingest: Video → CaptureManifest + keyframes
    1. SLAM: Pose estimation + 3D Gaussian reconstruction
    2. Export: Gaussians + camera data for DWM handoff

Sensor support:
    - RGB-only: Meta glasses, generic cameras (WildGS-SLAM)
    - RGB-D: iPhone LiDAR, RealSense (SplaTAM)
    - iOS ARKit: Direct pose import (skips SLAM)
"""

from .interfaces import (
    CaptureManifest,
    SensorType,
    SLAMBackend,
    PipelineConfig,
    CameraIntrinsics,
    FrameMetadata,
    Submap,
)

from .pipeline import (
    CapturePipeline,
    CaptureResult,
    run_capture_pipeline,
    # Backward compatibility
    Video2ZeroScenePipeline,
)

from .export import (
    CaptureExporter,
    CaptureExportResult,
    export_capture,
)

from .slam import (
    CameraPose,
    SLAMResult,
)

__all__ = [
    # Main pipeline
    "CapturePipeline",
    "CaptureResult",
    "run_capture_pipeline",
    # Backward compatibility
    "Video2ZeroScenePipeline",
    # Data models
    "CaptureManifest",
    "SensorType",
    "SLAMBackend",
    "PipelineConfig",
    "CameraIntrinsics",
    "FrameMetadata",
    "Submap",
    # Export
    "CaptureExporter",
    "CaptureExportResult",
    "export_capture",
    # SLAM
    "CameraPose",
    "SLAMResult",
]
