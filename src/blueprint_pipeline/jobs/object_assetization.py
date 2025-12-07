from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models import ArtifactPaths, JobPayload, SessionManifest
from .base import GPUJob, merge_parameters


@dataclass
class ObjectAssetizationJob(GPUJob):
    name: str = "object-assetization"
    description: str = (
        "Lift SAM 3 tracks into 3D, reconstruct objects when coverage exists, "
        "and fall back to Hunyuan3D when needed."
    )
    timeout_minutes: int = 120
    coverage_threshold: float = 0.6
    hunyuan_enabled: bool = True

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        params = merge_parameters(
            self.base_parameters(),
            {
                "coverage_threshold": self.coverage_threshold,
                "hunyuan_enabled": self.hunyuan_enabled,
            },
        )
        params = merge_parameters(params, parameters)
        return JobPayload(
            job_name=self.name,
            session_id=session.session_id,
            inputs={
                "frames": artifacts.frames,
                "masks": artifacts.masks,
                "poses": f"{artifacts.reconstruction}/poses",
                "environment_mesh": f"{artifacts.meshes}/environment_mesh.usd",
            },
            outputs={
                "object_usds": f"{artifacts.objects}/",
                "object_reports": f"{artifacts.reports}/objects.json",
            },
            parameters=params,
        )
