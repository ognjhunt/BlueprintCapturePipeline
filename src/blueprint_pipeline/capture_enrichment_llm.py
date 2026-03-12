"""Optional LLM enrichment for capture indexing and qualification."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


SkillRunner = Callable[[str, Mapping[str, Any]], Optional[Mapping[str, Any]]]


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
class CaptureEnrichmentConfig:
    provider: str = "disabled"
    mode: str = "auto"
    model: str = ""
    codex_bin: str = "codex"
    timeout_seconds: int = 120
    reasoning_effort: str = "medium"

    @classmethod
    def from_env(cls) -> "CaptureEnrichmentConfig":
        provider = (_string_env("CAPTURE_ENRICHMENT_LLM_PROVIDER", "disabled") or "disabled").lower()
        mode = (_string_env("CAPTURE_ENRICHMENT_LLM_MODE", "auto") or "auto").lower()
        default_model = "gpt-5.1-mini" if provider == "openai" else "claude-3-7-sonnet-latest"
        return cls(
            provider=provider,
            mode=mode,
            model=_string_env("CAPTURE_ENRICHMENT_LLM_MODEL", default_model) or default_model,
            codex_bin=_string_env("CAPTURE_ENRICHMENT_CODEX_BIN", "codex") or "codex",
            timeout_seconds=_int_env("CAPTURE_ENRICHMENT_LLM_TIMEOUT_SECONDS", 120),
            reasoning_effort=_string_env("CAPTURE_ENRICHMENT_LLM_REASONING_EFFORT", "medium") or "medium",
        )

    def enabled(self) -> bool:
        return self.provider in {"openai", "claude"}


def _skill_schema(skill_name: str) -> Dict[str, Any]:
    if skill_name == "prompt_bank_expander":
        return {
            "type": "object",
            "properties": {
                "additional_prompts": {"type": "array", "items": {"type": "string"}},
                "resolved_task_nouns": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
            "required": ["additional_prompts", "resolved_task_nouns", "notes"],
            "additionalProperties": True,
        }
    if skill_name == "task_relevance_ranker":
        return {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "object_id": {"type": "string"},
                            "score": {"type": "number"},
                            "matched_terms": {"type": "array", "items": {"type": "string"}},
                            "reason": {"type": "string"},
                        },
                        "required": ["object_id", "score"],
                        "additionalProperties": True,
                    },
                }
            },
            "required": ["scores"],
            "additionalProperties": True,
        }
    if skill_name == "workflow_target_resolver":
        return {
            "type": "object",
            "properties": {
                "manipulation_candidates": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                "articulation_hints": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                "tasks": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                "open_questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["manipulation_candidates", "articulation_hints", "tasks", "open_questions"],
            "additionalProperties": True,
        }
    if skill_name == "articulation_prior_writer":
        return {
            "type": "object",
            "properties": {
                "articulation_priors": {"type": "array", "items": {"type": "object", "additionalProperties": True}}
            },
            "required": ["articulation_priors"],
            "additionalProperties": True,
        }
    if skill_name == "qualification_weakness_summarizer":
        return {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "top_gaps": {"type": "array", "items": {"type": "string"}},
                "recommended_focus": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "top_gaps", "recommended_focus"],
            "additionalProperties": True,
        }
    if skill_name == "recapture_instruction_writer":
        return {
            "type": "object",
            "properties": {
                "instructions": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                "operator_brief": {"type": "string"},
            },
            "required": ["instructions", "operator_brief"],
            "additionalProperties": True,
        }
    return {"type": "object", "additionalProperties": True}


def _skill_instruction(skill_name: str) -> str:
    instructions = {
        "prompt_bank_expander": "Expand task-specific prompt terms conservatively from intake and capture context. Return concrete object nouns only.",
        "task_relevance_ranker": "Rank scene objects by task relevance. Be conservative and avoid inflating scores for generic scene clutter.",
        "workflow_target_resolver": "Resolve likely target objects, articulation-relevant objects, and likely tasks from the workflow text and object inventory.",
        "articulation_prior_writer": "Infer articulation priors only when the label strongly suggests an interactive object such as a drawer, cabinet, door, or refrigerator.",
        "qualification_weakness_summarizer": "Summarize why qualification is weak using only the provided blockers, evidence gaps, and readiness decision.",
        "recapture_instruction_writer": "Write clear recapture instructions that would reduce the specific evidence gaps in this capture.",
    }
    return instructions.get(skill_name, "Return conservative structured JSON only.")


def _prompt_for_skill(skill_name: str, payload: Mapping[str, Any]) -> str:
    return (
        "You are enriching BlueprintCapturePipeline artifacts.\n"
        "Use only the JSON payload. Do not invent measurements, IDs, or geometric facts.\n"
        "Be conservative. If the evidence is vague, say so.\n\n"
        f"Skill: {skill_name}\n"
        f"Instruction: {_skill_instruction(skill_name)}\n\n"
        "Return only a JSON object matching the requested schema.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}\n"
    )


def _extract_openai_text(response: Any) -> str:
    text = str(getattr(response, "output_text", "") or "").strip()
    if text:
        return text
    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for entry in content:
                    piece = getattr(entry, "text", None)
                    if piece:
                        chunks.append(str(piece))
        return "".join(chunks).strip()
    return ""


class _CodexRunner:
    def __init__(self, *, config: CaptureEnrichmentConfig, repo_root: Path) -> None:
        self._config = config
        self._repo_root = repo_root

    def __call__(self, skill_name: str, payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        codex_path = shutil.which(self._config.codex_bin)
        if not codex_path:
            return None
        schema = _skill_schema(skill_name)
        prompt = _prompt_for_skill(skill_name, payload)
        with tempfile.TemporaryDirectory(prefix="capture-enrichment-codex-") as tmp_dir:
            root = Path(tmp_dir)
            schema_path = root / "schema.json"
            output_path = root / "output.json"
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
                command.extend(["--reasoning-effort", self._config.reasoning_effort])
            try:
                subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=self._config.timeout_seconds,
                )
            except Exception:
                return None
            if not output_path.is_file():
                return None
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception:
                return None
            return payload if isinstance(payload, Mapping) else None


class _OpenAISDKRunner:
    def __init__(self, *, config: CaptureEnrichmentConfig) -> None:
        self._config = config

    def __call__(self, skill_name: str, payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        api_key = _string_env("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return None
        client = OpenAI(api_key=api_key)
        prompt = _prompt_for_skill(skill_name, payload)
        try:
            response = client.responses.create(
                model=self._config.model,
                input=prompt,
            )
        except Exception:
            return None
        text = _extract_openai_text(response)
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        return parsed if isinstance(parsed, Mapping) else None


class _ClaudeHTTPRunner:
    def __init__(self, *, config: CaptureEnrichmentConfig) -> None:
        self._config = config

    def __call__(self, skill_name: str, payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        api_key = _string_env("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        body = {
            "model": self._config.model,
            "max_tokens": 1800,
            "messages": [{"role": "user", "content": _prompt_for_skill(skill_name, payload)}],
        }
        request = urllib_request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self._config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError):
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        content = parsed.get("content") if isinstance(parsed, Mapping) else None
        if not isinstance(content, list):
            return None
        text_chunks = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                text_chunks.append(str(item.get("text") or ""))
        text = "".join(text_chunks).strip()
        if not text:
            return None
        try:
            result = json.loads(text)
        except Exception:
            return None
        return result if isinstance(result, Mapping) else None


def build_capture_enrichment_runner(
    *,
    repo_root: Path,
    config: Optional[CaptureEnrichmentConfig] = None,
) -> Optional[SkillRunner]:
    resolved = config or CaptureEnrichmentConfig.from_env()
    if not resolved.enabled():
        return None
    mode = resolved.mode
    if resolved.provider == "openai":
        if mode == "codex_cli":
            return _CodexRunner(config=resolved, repo_root=repo_root)
        if mode == "sdk":
            return _OpenAISDKRunner(config=resolved)
        if mode == "auto":
            codex_runner = _CodexRunner(config=resolved, repo_root=repo_root)
            if shutil.which(resolved.codex_bin):
                return codex_runner
            return _OpenAISDKRunner(config=resolved)
    if resolved.provider == "claude":
        return _ClaudeHTTPRunner(config=resolved)
    return None
