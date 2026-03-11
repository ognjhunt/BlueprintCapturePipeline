"""OpenAI-backed Phase 2 skill runner using the local Codex runtime."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


_DEFAULT_TIMEOUT_SECONDS = 120
_DEFAULT_MODE = "disabled"
_DEFAULT_CODEX_BIN = "codex"
_DEFAULT_MODEL = "gpt-5.1"
_DEFAULT_REASONING_EFFORT = "high"


def _string_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _int_env(name: str, default: int) -> int:
    raw = _string_env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class OpenAIPhase2Config:
    mode: str = _DEFAULT_MODE
    model: str = _DEFAULT_MODEL
    codex_bin: str = _DEFAULT_CODEX_BIN
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    reasoning_effort: str = _DEFAULT_REASONING_EFFORT

    @classmethod
    def from_env(cls) -> "OpenAIPhase2Config":
        return cls(
            mode=_string_env("OPENAI_PHASE2_MODE", _DEFAULT_MODE).lower() or _DEFAULT_MODE,
            model=_string_env("OPENAI_PHASE2_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL,
            codex_bin=_string_env("OPENAI_PHASE2_CODEX_BIN", _DEFAULT_CODEX_BIN) or _DEFAULT_CODEX_BIN,
            timeout_seconds=_int_env("OPENAI_PHASE2_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS),
            reasoning_effort=_string_env("OPENAI_PHASE2_REASONING_EFFORT", _DEFAULT_REASONING_EFFORT) or _DEFAULT_REASONING_EFFORT,
        )

    def normalized_mode(self) -> str:
        return self.mode.strip().lower() or _DEFAULT_MODE

    def enabled(self) -> bool:
        return self.normalized_mode() != "disabled"


_COMMON_OBJECT = {"type": "object", "additionalProperties": True}
_COMMON_ARRAY = {"type": "array", "items": _COMMON_OBJECT}


def _skill_schema(skill_name: str) -> Dict[str, Any]:
    shared: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string"},
            "scene_id": {"type": "string"},
            "capture_id": {"type": "string"},
        },
        "required": ["schema_version", "scene_id", "capture_id"],
        "additionalProperties": True,
    }
    if skill_name == "intake_normalizer":
        shared["properties"].update(
            {
                "status": {"type": "string"},
                "capture_modality": {"type": "string"},
                "workflow": {"type": ["string", "array", "object", "null"]},
                "zone": {"type": ["object", "string", "null"]},
                "owner": {"type": ["string", "null"]},
                "success_criteria": {"type": ["array", "null"], "items": {"type": "string"}},
                "adjacent_systems": {"type": ["array", "null"], "items": {"type": "string"}},
                "non_routine_modes": {"type": ["array", "null"], "items": {"type": "string"}},
                "people_traffic_notes": {"type": ["array", "null"], "items": {"type": "string"}},
                "privacy_restrictions": {"type": ["array", "null"], "items": {"type": "string"}},
                "security_restrictions": {"type": ["array", "null"], "items": {"type": "string"}},
                "known_blockers": {"type": ["array", "null"], "items": {"type": "string"}},
                "missing_required_fields": {"type": "array", "items": {"type": "string"}},
            }
        )
        shared["required"].extend(["status", "capture_modality", "missing_required_fields"])
        return shared
    if skill_name == "evidence_auditor":
        shared["properties"].update(
            {
                "status": {"type": "string"},
                "evidence_gaps": {"type": "array", "items": _COMMON_OBJECT},
                "low_confidence_route_edges": {"type": "array", "items": _COMMON_OBJECT},
                "hidden_zone_bound": {"type": "number"},
                "metric_ready": {"type": "boolean"},
                "supplemental_geometry": {"type": "array", "items": {"type": "string"}},
            }
        )
        shared["required"].extend(
            [
                "status",
                "evidence_gaps",
                "low_confidence_route_edges",
                "hidden_zone_bound",
                "metric_ready",
                "supplemental_geometry",
            ]
        )
        return shared
    if skill_name == "blocker_taxonomist":
        shared["properties"]["entries"] = _COMMON_ARRAY
        shared["required"].append("entries")
        return shared
    if skill_name == "capability_envelope_writer":
        shared["properties"].update(
            {
                "metric_ready": {"type": "boolean"},
                "measurements": _COMMON_OBJECT,
                "bounded_claims": {"type": "array", "items": {"type": "string"}},
                "evidence_gaps": {"type": "array", "items": _COMMON_OBJECT},
            }
        )
        shared["required"].extend(["metric_ready", "measurements", "bounded_claims", "evidence_gaps"])
        return shared
    if skill_name == "standards_retriever":
        shared["properties"].update(
            {
                "source": {"type": "string"},
                "entries": {"type": "array", "items": _COMMON_OBJECT},
            }
        )
        shared["required"].extend(["source", "entries"])
        return shared
    if skill_name == "recapture_planner":
        shared["properties"].update(
            {
                "required": {"type": "boolean"},
                "steps": {"type": "array", "items": _COMMON_OBJECT},
            }
        )
        shared["required"].extend(["required", "steps"])
        return shared
    if skill_name in {
        "humanoid_site_readiness_reviewer",
        "humanoid_workcell_risk_reviewer",
        "humanoid_route_access_reviewer",
        "oem_handoff_writer",
    }:
        shared["properties"]["summary"] = {"type": "string"}
        shared["required"].append("summary")
        return shared
    if skill_name == "readiness_report_writer":
        return {
            "type": "object",
            "properties": {
                "memo_markdown": {"type": "string"},
            },
            "required": ["memo_markdown"],
            "additionalProperties": False,
        }
    return _COMMON_OBJECT


def _skill_instruction(skill_name: str) -> str:
    instructions = {
        "intake_normalizer": "Normalize the intake into a grounded, conservative site-intake record.",
        "evidence_auditor": "Identify evidence gaps and route-confidence issues without inventing geometry.",
        "blocker_taxonomist": "Merge evidence gaps into a blocker register, preserving grounded details only.",
        "capability_envelope_writer": "Summarize bounded capability claims from the given measurements and checks only.",
        "standards_retriever": "Return concise standards notes relevant to the blockers; do not fabricate citations.",
        "humanoid_site_readiness_reviewer": "Write a site-readiness summary grounded in the provided readiness decision and standards notes.",
        "humanoid_workcell_risk_reviewer": "Summarize top workcell risks from qualification artifacts only.",
        "humanoid_route_access_reviewer": "Summarize route-access constraints using only the provided route and measurement evidence.",
        "oem_handoff_writer": "Write a concise OEM/integrator handoff summary grounded in the opportunity handoff artifact.",
        "recapture_planner": "Produce an ordered recapture plan using only explicit evidence gaps.",
        "readiness_report_writer": "Write a human-readable markdown memo that stays conservative and does not overstate readiness.",
    }
    return instructions.get(skill_name, "Return a conservative grounded JSON response.")


def _prompt_for_skill(skill_name: str, payload: Mapping[str, Any]) -> str:
    return (
        "You are generating a Phase 2 review artifact for BlueprintCapturePipeline.\n"
        "Use ONLY the supplied JSON payload. Do not invent physical facts, measurements, or approvals.\n"
        "Stay conservative: if evidence is incomplete, say so explicitly.\n"
        "Do not run shell commands or inspect repository files.\n\n"
        f"Skill: {skill_name}\n"
        f"Instruction: {_skill_instruction(skill_name)}\n\n"
        "Return only a JSON object matching the provided schema.\n\n"
        "Payload:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n"
    )


class CodexOpenAIPhase2Runner:
    """Callable skill runner backed by `codex exec`."""

    def __init__(self, *, config: OpenAIPhase2Config, repo_root: Path) -> None:
        self._config = config
        self._repo_root = repo_root

    def runtime_metadata(self) -> Dict[str, Any]:
        return {
            "openai_phase2_mode": self._config.normalized_mode(),
            "openai_phase2_model": self._config.model,
            "openai_phase2_timeout_seconds": self._config.timeout_seconds,
            "openai_phase2_transport": "codex_exec",
            "openai_phase2_reasoning_effort": self._config.reasoning_effort,
        }

    def skill_metadata(self, skill_name: str) -> Dict[str, Any]:
        return {
            "skill_name": skill_name,
            "transport": "codex_exec",
            "mode": self._config.normalized_mode(),
            "model": self._config.model,
            "reasoning_effort": self._config.reasoning_effort,
        }

    def __call__(self, skill_name: str, payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        if self._config.normalized_mode() != "codex_cli":
            return None
        codex_path = shutil.which(self._config.codex_bin)
        if not codex_path:
            return None

        schema = _skill_schema(skill_name)
        prompt = _prompt_for_skill(skill_name, payload)
        with tempfile.TemporaryDirectory(prefix="blueprint-openai-phase2-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            schema_path = tmp_root / "schema.json"
            output_path = tmp_root / "output.json"
            schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
            command = [
                codex_path,
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--cd",
                str(self._repo_root),
            ]
            if self._config.model:
                command.extend(["--model", self._config.model])
            if self._config.reasoning_effort:
                command.extend(["-c", f"model_reasoning_effort={json.dumps(self._config.reasoning_effort)}"])
            command.append("-")
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=max(1, int(self._config.timeout_seconds)),
                    check=False,
                )
            except (OSError, subprocess.SubprocessError, ValueError):
                return None
            if completed.returncode != 0 or not output_path.is_file():
                return None
            try:
                response = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
            return response if isinstance(response, Mapping) else None


def build_openai_skill_runner(
    *,
    repo_root: Path,
    config: Optional[OpenAIPhase2Config] = None,
):
    resolved = config or OpenAIPhase2Config.from_env()
    if not resolved.enabled():
        return None
    if resolved.normalized_mode() == "codex_cli":
        return CodexOpenAIPhase2Runner(config=resolved, repo_root=repo_root)
    return None
