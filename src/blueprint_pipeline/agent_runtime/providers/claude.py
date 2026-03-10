"""Claude provider metadata and optional overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional


SkillRunner = Callable[[str, Mapping[str, Any]], Optional[Mapping[str, Any]]]


@dataclass
class ClaudeAgentProvider:
    skill_runner: Optional[SkillRunner] = None
    repo_root: Optional[Path] = None

    name: str = "claude"

    def runtime_metadata(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "sdk_strategy": "local_project_skills",
            "setting_sources": ["project"],
            "allowed_tools": ["Skill"],
            "skills_root": str((self.repo_root or Path.cwd()) / ".claude" / "skills"),
        }

    def skill_metadata(self, skill_name: str) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "skill_name": skill_name,
            "settings_sources": ["project"],
            "allowed_tools": ["Skill"],
        }

    def invoke_skill(self, skill_name: str, payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if self.skill_runner is None:
            return None
        result = self.skill_runner(skill_name, payload)
        return dict(result) if isinstance(result, Mapping) else None
