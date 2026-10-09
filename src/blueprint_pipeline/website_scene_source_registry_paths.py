"""Read-only server-owned website source registration path contract.

ADP-010/day14: retained authority reads must not import the scene dispatch
worker merely to locate the immutable source record. No registration writes.
"""
import os
from pathlib import Path

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .task_evaluation_scene_progression_contracts import require, safe_path


def binding_root(config=None):
    config = config or {}
    configured = config.get("website_source_binding_root") or os.getenv("BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT")
    if configured:
        return safe_path(configured)
    intent_root = config.get("intent_root") or os.getenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT")
    require(bool(intent_root), "website_scene_intake_root_missing")
    return safe_path(Path(intent_root).parent / "website-source-bindings")


def _index_path(root, request):
    return root / (cross_runtime_canonical_digest(request)[7:] + ".json")


