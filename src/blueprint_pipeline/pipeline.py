from __future__ import annotations

from typing import List

from .models import ArtifactPaths, JobPayload, SessionManifest
from .jobs.frame_extraction import FrameExtractionJob
from .jobs.reconstruction import ReconstructionJob


DEFAULT_ARTIFACT_TEMPLATE = "gs://<bucket>/sessions/{session_id}"


def default_artifact_paths(session_id: str, base: str = DEFAULT_ARTIFACT_TEMPLATE) -> ArtifactPaths:
    session_root = base.format(session_id=session_id)
    return ArtifactPaths(
        session_root=session_root,
        frames=f"{session_root}/frames",
        masks=f"{session_root}/masks",
        reconstruction=f"{session_root}/reconstruction",
        reports=f"{session_root}/reports",
    )


def build_default_pipeline(
    session: SessionManifest, artifacts: ArtifactPaths | None = None
) -> List[JobPayload]:
    """Return JobPayload stubs for each GPU-heavy stage.

    This provides the orchestrator with a deterministic fan-out order and a
    consistent payload shape for Cloud Run Jobs.

    Pipeline stages:
    1. Frame Extraction - Video → keyframes
    2. Reconstruction - SLAM → 3D Gaussians + camera poses
    """

    artifact_paths = artifacts or default_artifact_paths(session.session_id)

    frame_job = FrameExtractionJob()
    reconstruction_job = ReconstructionJob()

    payloads: List[JobPayload] = [
        frame_job.build_payload(session, artifact_paths),
        reconstruction_job.build_payload(session, artifact_paths),
    ]
    return payloads


__all__ = [
    "ArtifactPaths",
    "JobPayload",
    "SessionManifest",
    "build_default_pipeline",
    "default_artifact_paths",
]
