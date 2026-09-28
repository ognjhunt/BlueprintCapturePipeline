"""Strict website-facing contract for preparing whole-splat object discovery."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import jsonschema

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "scene_object_discovery_request.v1"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "schemas"
    / "scene_object_discovery_request.v1.schema.json"
)


class SceneObjectDiscoveryContractError(ValueError):
    """The external request is structurally unsafe or semantically incoherent."""


@lru_cache(maxsize=1)
def scene_object_discovery_request_schema() -> dict[str, Any]:
    try:
        value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneObjectDiscoveryContractError(
            "scene_object_discovery_schema_unavailable"
        ) from exc
    if not isinstance(value, Mapping):
        raise SceneObjectDiscoveryContractError("scene_object_discovery_schema_invalid")
    jsonschema.Draft202012Validator.check_schema(value)
    return dict(value)


def validate_scene_object_discovery_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(value)
    validator = jsonschema.Draft202012Validator(
        scene_object_discovery_request_schema(),
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(validator.iter_errors(request), key=lambda row: list(row.path))
    if errors:
        path = ".".join(str(part) for part in errors[0].path) or "$"
        raise SceneObjectDiscoveryContractError(f"scene_object_discovery_request_invalid:{path}")
    task = request["task"]
    strategies = {
        "rigid_relocation": {"planar_push", "pick_and_place"},
        "articulated_manipulation": {"articulated_open_close"},
    }
    if task["strategy"] not in strategies[task["kind"]]:
        raise SceneObjectDiscoveryContractError(
            "scene_object_discovery_task_strategy_kind_mismatch"
        )
    analyzers = request["analysis"]["analyzers"]
    prompts = [str(prompt).casefold() for prompt in request["analysis"]["prompts"]]
    if len(analyzers) != len(set(analyzers)):
        raise SceneObjectDiscoveryContractError("scene_object_discovery_analyzers_duplicate")
    if len(prompts) != len(set(prompts)):
        raise SceneObjectDiscoveryContractError("scene_object_discovery_prompts_duplicate")
    execution = request["execution"]
    rights = request["rights"]
    if execution["mode"] == "provider_gpu_after_activation":
        if execution.get("selected_provider") != "vast":
            raise SceneObjectDiscoveryContractError("scene_object_discovery_provider_missing")
        if (
            rights["provider_disclosure_scope"] != "source_and_derived"
            or rights["source_bytes_redistributable"] is not True
        ):
            raise SceneObjectDiscoveryContractError(
                "scene_object_discovery_provider_source_disclosure_not_authorized"
            )
    elif "selected_provider" in execution:
        raise SceneObjectDiscoveryContractError(
            "scene_object_discovery_local_mode_provider_forbidden"
        )
    return request


def scene_object_discovery_request_digest(value: Mapping[str, Any]) -> str:
    return canonical_digest(validate_scene_object_discovery_request(value))


__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "SceneObjectDiscoveryContractError",
    "scene_object_discovery_request_digest",
    "scene_object_discovery_request_schema",
    "validate_scene_object_discovery_request",
]
