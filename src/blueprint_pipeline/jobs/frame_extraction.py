from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models import ArtifactPaths, JobPayload, SessionManifest
from .base import GPUJob, merge_parameters


@dataclass
class FrameExtractionJob(GPUJob):
    name: str = "frame-extraction"
    description: str = "Decode video clips, extract frames, and run SAM 3 masking."
    timeout_minutes: int = 45
    target_fps: float = 4.0
    mask_model: str = "sam3-video"
    include_dynamic_masks: bool = True

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        params = merge_parameters(
            self.base_parameters(),
            {
                "target_fps": self.target_fps,
                "mask_model": self.mask_model,
                "include_dynamic_masks": self.include_dynamic_masks,
            },
        )
        params = merge_parameters(params, parameters)
        return JobPayload(
            job_name=self.name,
            session_id=session.session_id,
            inputs={
                "clips": ",".join([clip.uri for clip in session.clips]),
            },
            outputs={
                "frames": artifacts.frames,
                "masks": artifacts.masks,
            },
            parameters=params,
        )
