from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ScaleAnchor:
    """Observations that allow us to recover metric scale from monocular capture."""

    anchor_type: str  # e.g., "aruco_board", "tape_measure", "known_object"
    size_meters: float
    notes: Optional[str] = None


@dataclass
class Clip:
    uri: str
    fps: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class SessionManifest:
    session_id: str
    capture_start: str
    device: Dict[str, str]
    scale_anchors: List[ScaleAnchor]
    clips: List[Clip]
    user_notes: Optional[str] = None


@dataclass
class ArtifactPaths:
    """Logical paths in GCS for artifacts produced by each stage."""

    session_root: str
    frames: str
    masks: str
    reconstruction: str
    meshes: str
    objects: str
    reports: str


@dataclass
class JobPayload:
    """A serializable payload to hand to Cloud Run Jobs."""

    job_name: str
    session_id: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    parameters: Dict[str, object] = field(default_factory=dict)

    def as_json(self) -> Dict[str, object]:
        return {
            "job_name": self.job_name,
            "session_id": self.session_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters,
        }
