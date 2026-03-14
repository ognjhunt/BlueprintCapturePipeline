"""Grounding-first policy helpers for canonical and presentation world-model outputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from .common import parse_bool


def _string_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    text = value.strip()
    return text or default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return parse_bool(value, default=default)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value]
    else:
        items = []
    out: list[str] = []
    for item in items:
        text = item.strip()
        if text and text not in out:
            out.append(text)
    return out


@dataclass(frozen=True)
class WorldModelPolicy:
    output_policy: str = "grounding_first"
    emit_presentation: bool = True
    allow_generative_completion: str = "limited"
    provenance_required: bool = True
    canonical_incomplete_ok: bool = True
    validation_profile: str = "prototype"

    @classmethod
    def from_env(cls) -> "WorldModelPolicy":
        return cls(
            output_policy=_string_env("WORLD_MODEL_OUTPUT_POLICY", "grounding_first"),
            emit_presentation=_bool_env("WORLD_MODEL_EMIT_PRESENTATION", True),
            allow_generative_completion=_string_env(
                "WORLD_MODEL_ALLOW_GENERATIVE_COMPLETION",
                "limited",
            ),
            provenance_required=_bool_env("WORLD_MODEL_PROVENANCE_REQUIRED", True),
            canonical_incomplete_ok=_bool_env("WORLD_MODEL_CANONICAL_INCOMPLETE_OK", True),
            validation_profile=_string_env("WORLD_MODEL_VALIDATION_PROFILE", "prototype"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_model_output_policy": self.output_policy,
            "emit_presentation": self.emit_presentation,
            "allow_generative_completion": self.allow_generative_completion,
            "provenance_required": self.provenance_required,
            "canonical_incomplete_ok": self.canonical_incomplete_ok,
            "validation_profile": self.validation_profile,
        }


def build_output_linkage(
    *,
    policy: WorldModelPolicy,
    canonical_artifact_uri: Optional[str],
    presentation_artifact_uri: Optional[str],
    authoritative_record: bool,
    derivation_mode: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "canonical_artifact_uri": canonical_artifact_uri,
        "presentation_artifact_uri": presentation_artifact_uri,
        "derivation_mode": derivation_mode or policy.output_policy,
        "authoritative_record": authoritative_record,
        "output_policy": policy.to_dict(),
    }


def build_presentation_derivation_policy(
    *,
    policy: WorldModelPolicy,
    variance_policy: Optional[Mapping[str, Any]] = None,
    canonical_authority: str = "site_world_spec",
) -> Dict[str, Any]:
    variance = dict(variance_policy or {})
    return {
        "presentation_role": "non_authoritative_derivative",
        "allowed_completion_level": policy.allow_generative_completion,
        "editable_regions_source": "runtime_layer_policy.protected_regions_manifest",
        "allowed_editable_region_classes": _string_list(
            variance.get("allowed_editable_region_classes")
        ),
        "forbidden_changes": _string_list(variance.get("forbidden_changes")),
        "fallback_on_conflict": "canonical_only",
        "canonical_authority": canonical_authority,
        "world_model_output_policy": policy.output_policy,
        "provenance_required": policy.provenance_required,
    }


def build_provenance_record(
    *,
    grounding_level: str,
    evidence_sources: Iterable[Any],
    observation_coverage: Optional[Mapping[str, Any]] = None,
    confidence: Any = None,
    canonical_truth: bool = True,
    presentation_only: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_sources = []
    for item in evidence_sources:
        text = str(item or "").strip()
        if text and text not in normalized_sources:
            normalized_sources.append(text)

    confidence_value: Optional[float]
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None
    if confidence_value is not None:
        confidence_value = max(0.0, min(1.0, confidence_value))

    payload: Dict[str, Any] = {
        "grounding_level": grounding_level,
        "evidence_sources": normalized_sources,
        "observation_coverage": dict(observation_coverage or {}),
        "confidence": confidence_value,
        "canonical_truth": canonical_truth,
        "presentation_only": presentation_only,
    }
    if isinstance(extra, Mapping):
        payload.update({str(key): value for key, value in extra.items()})
    return payload
