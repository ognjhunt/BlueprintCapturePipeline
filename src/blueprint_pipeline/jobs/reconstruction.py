from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models import ArtifactPaths, JobPayload, SessionManifest
from .base import GPUJob, merge_parameters


@dataclass
class ReconstructionJob(GPUJob):
    name: str = "reconstruction"
    description: str = "WildGS-SLAM with scale calibration and dynamic masking."
    timeout_minutes: int = 90
    use_dynamic_masks: bool = True
    enforce_scale_anchor: bool = True

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        params = merge_parameters(
            self.base_parameters(),
            {
                "use_dynamic_masks": self.use_dynamic_masks,
                "enforce_scale_anchor": self.enforce_scale_anchor,
                "scale_anchor_count": len(session.scale_anchors),
            },
        )
        params = merge_parameters(params, parameters)
        return JobPayload(
            job_name=self.name,
            session_id=session.session_id,
            inputs={
                "frames": artifacts.frames,
                "masks": artifacts.masks,
            },
            outputs={
                "poses": f"{artifacts.reconstruction}/poses",  # COLMAP-style outputs
                "gaussians": f"{artifacts.reconstruction}/gaussians",
                "reprojection_report": f"{artifacts.reconstruction}/reports/reprojection.json",
            },
            parameters=params,
        )
