from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..models import ArtifactPaths, JobPayload, SessionManifest


@dataclass
class BaseJob:
    """Base class for pipeline jobs.

    Each job declares the inputs it consumes and outputs it produces. These
    declarations can be serialized into Cloud Run Job payloads or used by
    an orchestrator to wire Pub/Sub topics.
    """

    name: str
    description: str
    uses_gpu: bool = False
    timeout_minutes: int = 30

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        raise NotImplementedError


@dataclass
class GPUJob(BaseJob):
    uses_gpu: bool = True
    gpu_type: str = "L4"
    gpu_memory_gb: int = 24
    concurrency: int = 1

    def base_parameters(self) -> Dict[str, object]:
        return {
            "gpu_type": self.gpu_type,
            "gpu_memory_gb": self.gpu_memory_gb,
            "concurrency": self.concurrency,
        }


@dataclass
class FrameSet:
    """Common input/output references for frame-based stages."""

    frames_uri: str
    masks_uri: str


def merge_parameters(
    base: Dict[str, object], extra: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    if not extra:
        return base
    merged = dict(base)
    merged.update(extra)
    return merged
