"""Read controls metadata without importing provisioning or execution workers.

ADP-050 Day 28 / ADP-009D: retain the worker's JSON, seal, and scene-intent
validation semantics for the read-only website context.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import task_evaluation_scene_intent_contracts as intake
from .decision_evidence_contracts import canonical_digest

CATALOG_SCHEMA = "task_evaluation_controls_robot_catalog.v1"
CONFIG_ENV = "BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG"
CONTENT_CATALOG_SCHEMA = "task_evaluation_controls_robot_content_catalog.v1"


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("controls_autoprovision_" + code)


def _json(path: Path) -> dict[str, Any]:
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink_refused")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "record_invalid")
    return value


def _sealed(path: Path, field: str) -> dict[str, Any]:
    value = _json(path)
    _require(value.get(field) == canonical_digest(value, digest_field=field), "digest_invalid")
    return value


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_digest(result, digest_field=field)
    return result


def _scene_intent(path: Path) -> dict[str, Any]:
    # Scene intents cross the Website/Pipeline boundary; let their canonical
    # contract own its number encoding rather than using controls-only hashing.
    _json(path)
    return intake._read(path, "intent_digest")
