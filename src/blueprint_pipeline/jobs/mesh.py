"""Mesh extraction job (placeholder for future implementation)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..models import ArtifactPaths, JobPayload, SessionManifest
from .base import BaseJob, JobContext, JobResult, JobStatus, merge_parameters


@dataclass
class MeshExtractionJob(BaseJob):
    """Extract meshes from 3D Gaussian splats.

    This is a placeholder job for future mesh extraction functionality.
    The actual implementation would convert 3DGS to mesh using:
    - Marching cubes on density field
    - Poisson surface reconstruction
    - Or similar techniques
    """

    name: str = "mesh-extraction"
    description: str = "Extract meshes from 3D Gaussian splats (placeholder)"
    timeout_minutes: int = 30

    def _get_default_parameters(self) -> Dict[str, Any]:
        return {
            "mesh_resolution": 256,
            "export_format": "obj",
        }

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        params = merge_parameters(self._get_default_parameters(), parameters)
        return JobPayload(
            job_name=self.name,
            session_id=session.session_id,
            inputs={
                "gaussians": f"{artifacts.reconstruction}/gaussians",
            },
            outputs={
                "meshes": artifacts.meshes,
            },
            parameters=params,
        )

    def _execute(self, ctx: JobContext) -> JobResult:
        """Execute mesh extraction (placeholder)."""
        ctx.logger.info("MeshExtractionJob: Placeholder - not yet implemented")

        # Return success with empty outputs for now
        return JobResult(
            status=JobStatus.COMPLETED,
            outputs={
                "meshes": ctx.artifacts.meshes,
            },
            metrics={
                "placeholder": True,
            },
        )
