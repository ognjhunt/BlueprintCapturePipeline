"""OpenAI provider metadata and optional overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional


SkillRunner = Callable[[str, Mapping[str, Any]], Optional[Mapping[str, Any]]]


@dataclass
class OpenAIAgentProvider:
    skill_runner: Optional[SkillRunner] = None
    repo_root: Optional[Path] = None

    name: str = "openai"

    def runtime_metadata(self) -> Dict[str, Any]:
        metadata = {
            "provider": self.name,
            "sdk_strategy": "local_skills_plus_shell",
            "preferred_tool": "shell",
            "skills_root": str((self.repo_root or Path.cwd()) / ".agents" / "skills"),
            "codex_compatible": True,
        }
        if self.skill_runner is not None and hasattr(self.skill_runner, "runtime_metadata"):
            runner_metadata = getattr(self.skill_runner, "runtime_metadata")()
            if isinstance(runner_metadata, Mapping):
                metadata.update({str(key): value for key, value in runner_metadata.items()})
        return metadata

    def skill_metadata(self, skill_name: str) -> Dict[str, Any]:
        metadata = {
            "provider": self.name,
            "skill_name": skill_name,
            "preferred_tool": "shell",
            "codex_compatible": True,
        }
        if self.skill_runner is not None and hasattr(self.skill_runner, "skill_metadata"):
            runner_metadata = getattr(self.skill_runner, "skill_metadata")(skill_name)
            if isinstance(runner_metadata, Mapping):
                metadata.update({str(key): value for key, value in runner_metadata.items()})
        return metadata

    def invoke_skill(self, skill_name: str, payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if self.skill_runner is None:
            return None
        result = self.skill_runner(skill_name, payload)
        return dict(result) if isinstance(result, Mapping) else None
