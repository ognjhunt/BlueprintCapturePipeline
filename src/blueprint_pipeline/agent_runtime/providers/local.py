"""Local deterministic provider metadata for no-LLM agent review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass
class LocalDeterministicAgentProvider:
    repo_root: Optional[Path] = None

    name: str = "local"

    def runtime_metadata(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "sdk_strategy": "local_deterministic_no_llm",
            "external_provider_calls_performed": False,
            "llm_provider_required": False,
            "setting_sources": ["project"],
            "proof_boundary": (
                "Local deterministic review writes contract artifacts and recapture "
                "guidance; it is not a live LLM/operator review."
            ),
        }

    def skill_metadata(self, skill_name: str) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "skill_name": skill_name,
            "source": "local_deterministic",
            "external_provider_calls_performed": False,
            "llm_provider_required": False,
        }

    def invoke_skill(
        self,
        skill_name: str,
        payload: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        return None
