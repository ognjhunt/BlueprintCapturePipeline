"""Contracts for agent review output artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from ..common import utc_now_iso


@dataclass(frozen=True)
class ReviewOutputFile:
    name: str
    path: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "path": self.path}


@dataclass(frozen=True)
class ReviewStepResult:
    skill_name: str
    output_path: str
    source: str
    provider_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "output_path": self.output_path,
            "source": self.source,
            "provider_metadata": dict(self.provider_metadata),
        }


@dataclass(frozen=True)
class AgentReviewBundle:
    scene_id: str
    capture_id: str
    provider: str
    readiness_state: str
    final_memo_path: str
    final_bundle_path: str
    human_actions_required_path: str
    outputs: List[ReviewOutputFile]
    steps: List[ReviewStepResult]
    runtime: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "v1",
            "scene_id": self.scene_id,
            "capture_id": self.capture_id,
            "generated_at": utc_now_iso(),
            "provider": self.provider,
            "readiness_state": self.readiness_state,
            "final_memo_path": self.final_memo_path,
            "final_bundle_path": self.final_bundle_path,
            "human_actions_required_path": self.human_actions_required_path,
            "outputs": [item.to_dict() for item in self.outputs],
            "steps": [item.to_dict() for item in self.steps],
            "runtime": dict(self.runtime),
        }


def ensure_mapping(value: Mapping[str, Any] | None) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
