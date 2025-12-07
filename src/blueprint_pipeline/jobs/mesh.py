from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models import ArtifactPaths, JobPayload, SessionManifest
from .base import GPUJob, merge_parameters


@dataclass
class MeshExtractionJob(GPUJob):
    name: str = "mesh-extraction"
    description: str = "SuGaR mesh extraction and texture baking from Gaussian splats."
    timeout_minutes: int = 60
    generate_collision_mesh: bool = True
    bake_textures: bool = True

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        params = merge_parameters(
            self.base_parameters(),
            {
                "generate_collision_mesh": self.generate_collision_mesh,
                "bake_textures": self.bake_textures,
            },
        )
        params = merge_parameters(params, parameters)
        return JobPayload(
            job_name=self.name,
            session_id=session.session_id,
            inputs={
                "gaussians": f"{artifacts.reconstruction}/gaussians",
            },
            outputs={
                "mesh": f"{artifacts.meshes}/environment_mesh.usd",
                "collision_mesh": f"{artifacts.meshes}/environment_collision.usd",
                "textures": f"{artifacts.meshes}/textures/",
            },
            parameters=params,
        )
