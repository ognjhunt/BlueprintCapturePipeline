"""Blueprint Capture Pipeline - Video to SimReady 3D Reconstruction.

A GPU-accelerated pipeline for converting walkthrough video captures
into realistic, physics-enabled 3D scenes for robotics simulation.

This pipeline is designed to work with BlueprintPipeline for downstream
SimReady USD assembly and Isaac Lab integration.

Architecture:
    BlueprintCapturePipeline (this repo)
        -> video2zeroscene: Converts video to ZeroScene format
        -> Jobs: GPU-accelerated processing stages

    BlueprintPipeline (downstream)
        -> zeroscene_adapter: Imports ZeroScene bundles
        -> simready: Physics property inference
        -> usd_assembly: Final USD packaging
        -> isaac_lab: Task scaffolding

Pipeline Stages (video2zeroscene):
    0. Ingest - Video normalization, keyframe selection
    1. SLAM - Sensor-conditional pose estimation (WildGS-SLAM/SplaTAM/VIGS-SLAM/ARKit)
    2. Mesh - SuGaR mesh extraction from Gaussian splats
    3. Tracks - SAM3 concept segmentation and tracking
    4. Lift - 2D tracks to 3D object proposals
    5. Assetize - Tiered object asset generation
    6. Export - ZeroScene bundle for BlueprintPipeline

Sensor Support:
    - RGB-only: Meta glasses, generic cameras (WildGS-SLAM)
    - RGB-D: iPhone LiDAR, RealSense (SplaTAM)
    - Visual-Inertial: RGB + IMU (VIGS-SLAM)
    - iOS ARKit: Direct pose import (skips SLAM)

Scale Support:
    - ArUco/AprilTag markers
    - Known object dimensions
    - Tape measure references
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
    MeshExtractionJob,
    ObjectAssetizationJob,
    USDAuthoringJob,
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

# Video2ZeroScene pipeline (new recommended interface)
from .video2zeroscene import (
    Video2ZeroScenePipeline,
    CaptureManifest,
    ZeroSceneBundle,
    ObjectProposal,
    TrackInfo,
    Submap,
)
from .video2zeroscene.interfaces import (
    SLAMBackend,
    AssetizationTier,
    PipelineConfig as V2ZConfig,
)


__version__ = "0.2.0"

__all__ = [
    # Version
    "__version__",
    # Models
    "ArtifactPaths",
    "Clip",
    "JobPayload",
    "PipelineConfig",
    "ScaleAnchor",
    "SensorType",
    "SessionManifest",
    "create_artifact_paths",
    # Video2ZeroScene (recommended)
    "Video2ZeroScenePipeline",
    "CaptureManifest",
    "ZeroSceneBundle",
    "ObjectProposal",
    "TrackInfo",
    "Submap",
    "SLAMBackend",
    "AssetizationTier",
    "V2ZConfig",
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
    "MeshExtractionJob",
    "ObjectAssetizationJob",
    "USDAuthoringJob",
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
