"""BlueprintCapture Pipeline - Phase 3: Video to Gaussian for DWM.

A GPU-accelerated pipeline for converting walkthrough video captures
into high-quality 3D Gaussian representations for DWM (Dexterous World
Models) processing.

This is Phase 3 of the Blueprint system:
    Phase 1: BlueprintPipeline - Image → SimReady 3D reconstruction
    Phase 2: DWM data layer - Scene → egocentric rollouts
    Phase 3: BlueprintCapture (this repo) - Video → Gaussian capture
    Phase 4: AR platform - Digital twins → AR Cloud

Architecture:
    BlueprintCapturePipeline (this repo)
        -> CapturePipeline: Video → Gaussian + camera data
        -> Output: Ready for DWM processing in BlueprintPipeline

Pipeline stages:
    0. Ingest - Video normalization, keyframe selection
    1. SLAM - Pose estimation + 3D Gaussian reconstruction
    2. Export - Package for BlueprintPipeline/DWM handoff

Sensor support:
    - RGB-only: Meta glasses, generic cameras (WildGS-SLAM)
    - RGB-D: iPhone LiDAR, RealSense (SplaTAM)
    - iOS ARKit: Direct pose import (skips SLAM)
"""

from .models import (
    ArtifactPaths,
    Clip,
    JobPayload,
    PipelineConfig,
    ScaleAnchor,
    SensorType,
    SessionManifest,
    create_artifact_paths,
)
from .pipeline import (
    build_default_pipeline,
    default_artifact_paths,
)
from .orchestrator import (
    PipelineOrchestrator,
    PipelineResult,
    PipelineStage,
    StageResult,
    create_cloud_run_job_config,
)
from .jobs import (
    FrameExtractionJob,
    ReconstructionJob,
)
from .jobs.base import (
    BaseJob,
    GPUJob,
    JobResult,
    JobStatus,
)
from .ios_manifest import (
    IOSManifest,
    IOSUploadInfo,
    ExtendedSessionData,
    discover_ios_upload,
    convert_ios_to_session,
    load_extended_session,
)
from .arkit_loader import (
    ARKitData,
    ARKitIntrinsics,
    ARKitPose,
    load_arkit_data_from_directory,
    load_arkit_data_from_gcs,
    can_skip_slam,
)

# Capture pipeline (main interface)
from .video2zeroscene import (
    CapturePipeline,
    CaptureResult,
    run_capture_pipeline,
    CaptureManifest,
    CaptureExporter,
    CaptureExportResult,
    CameraPose,
    SLAMResult,
    SLAMBackend,
    Submap,
    # Backward compatibility
    Video2ZeroScenePipeline,
)
from .video2zeroscene.interfaces import (
    PipelineConfig as CaptureConfig,
)


__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Capture Pipeline (main interface)
    "CapturePipeline",
    "CaptureResult",
    "run_capture_pipeline",
    "CaptureManifest",
    "CaptureExporter",
    "CaptureExportResult",
    "CameraPose",
    "SLAMResult",
    "SLAMBackend",
    "CaptureConfig",
    "Submap",
    # Backward compatibility
    "Video2ZeroScenePipeline",
    # Legacy Models
    "ArtifactPaths",
    "Clip",
    "JobPayload",
    "PipelineConfig",
    "ScaleAnchor",
    "SensorType",
    "SessionManifest",
    "create_artifact_paths",
    # Legacy Pipeline
    "build_default_pipeline",
    "default_artifact_paths",
    # Orchestrator
    "PipelineOrchestrator",
    "PipelineResult",
    "PipelineStage",
    "StageResult",
    "create_cloud_run_job_config",
    # Jobs
    "BaseJob",
    "GPUJob",
    "JobResult",
    "JobStatus",
    "FrameExtractionJob",
    "ReconstructionJob",
    # iOS Capture Support
    "IOSManifest",
    "IOSUploadInfo",
    "ExtendedSessionData",
    "discover_ios_upload",
    "convert_ios_to_session",
    "load_extended_session",
    # ARKit Data
    "ARKitData",
    "ARKitIntrinsics",
    "ARKitPose",
    "load_arkit_data_from_directory",
    "load_arkit_data_from_gcs",
    "can_skip_slam",
]
