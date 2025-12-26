"""Object assetization job (placeholder for future implementation)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..models import ArtifactPaths, JobPayload, SessionManifest
from .base import BaseJob, JobContext, JobResult, JobStatus, merge_parameters


@dataclass
class ObjectAssetizationJob(BaseJob):
    """Convert detected objects to individual assets.

    This is a placeholder job for future object assetization functionality.
    The actual implementation would:
    - Segment individual objects from the scene
    - Create standalone 3D assets for each object
    - Generate PBR materials and textures
    """

    name: str = "object-assetization"
    description: str = "Convert objects to individual assets (placeholder)"
    timeout_minutes: int = 60

    def _get_default_parameters(self) -> Dict[str, Any]:
        return {
            "min_object_size": 0.05,  # Minimum object size in meters
            "max_objects": 100,
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
                "meshes": artifacts.meshes,
                "masks": artifacts.masks,
            },
            outputs={
                "objects": artifacts.objects,
            },
            parameters=params,
        )

    def _execute(self, ctx: JobContext) -> JobResult:
        """Execute object assetization (placeholder)."""
        ctx.logger.info("ObjectAssetizationJob: Placeholder - not yet implemented")

        return JobResult(
            status=JobStatus.COMPLETED,
            outputs={
                "objects": ctx.artifacts.objects,
            },
            metrics={
                "placeholder": True,
            },
        )
