"""Blueprint Capture Pipeline - Video to SimReady 3D Reconstruction.

A GPU-accelerated pipeline for converting Meta smart glasses video captures
into realistic, physics-enabled 3D scenes for robotics simulation.

Pipeline Stages:
    1. Frame Extraction - Decode video, extract frames, run SAM3 segmentation
    2. Reconstruction - WildGS-SLAM with scale calibration (or ARKit poses if available)
    3. Mesh Extraction - SuGaR mesh extraction from Gaussian splats
    4. Object Assetization - Lift objects to 3D, Hunyuan3D fallback
    5. USD Authoring - Package with physics for Isaac Sim

iOS Capture Support:
    - Automatic detection of iOS uploads via Cloud Functions (see functions/)
    - Parses iOS manifest.json and ARKit data (poses, depth, intrinsics)
    - Can skip SLAM when ARKit poses are available for faster processing
"""

from .models import (
    ArtifactPaths,
    Clip,
    JobPayload,
    ScaleAnchor,
    SessionManifest,
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


__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Models
    "ArtifactPaths",
    "Clip",
    "JobPayload",
    "ScaleAnchor",
    "SessionManifest",
    # Pipeline
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
