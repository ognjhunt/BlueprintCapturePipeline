from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models import ArtifactPaths, JobPayload, SessionManifest
from .base import BaseJob, merge_parameters


@dataclass
class USDAuthoringJob(BaseJob):
    name: str = "usd-authoring"
    description: str = "Package environment and objects into USD with physics metadata."
    timeout_minutes: int = 30
    uses_gpu: bool = False
    meters_per_unit: float = 1.0
    convex_decomposition: bool = True

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        params = merge_parameters(
            {
                "meters_per_unit": self.meters_per_unit,
                "convex_decomposition": self.convex_decomposition,
            },
            parameters,
        )
        return JobPayload(
            job_name=self.name,
            session_id=session.session_id,
            inputs={
                "environment_mesh": f"{artifacts.meshes}/environment_mesh.usd",
                "collision_mesh": f"{artifacts.meshes}/environment_collision.usd",
                "object_usds": f"{artifacts.objects}/",
            },
            outputs={
                "scene_usd": f"{artifacts.session_root}/scene.usdc",
                "report": f"{artifacts.reports}/usd_authoring.json",
            },
            parameters=params,
        )
