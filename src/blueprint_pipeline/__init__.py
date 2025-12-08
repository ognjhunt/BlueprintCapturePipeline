"""Blueprint Capture Pipeline - Video to SimReady 3D Reconstruction.

A GPU-accelerated pipeline for converting Meta smart glasses video captures
into realistic, physics-enabled 3D scenes for robotics simulation.

Pipeline Stages:
    1. Frame Extraction - Decode video, extract frames, run SAM3 segmentation
    2. Reconstruction - WildGS-SLAM with scale calibration
    3. Mesh Extraction - SuGaR mesh extraction from Gaussian splats
    4. Object Assetization - Lift objects to 3D, Hunyuan3D fallback
    5. USD Authoring - Package with physics for Isaac Sim
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
]
