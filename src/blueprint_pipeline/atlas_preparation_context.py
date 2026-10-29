"""Shared task-reading rules for website and owned-upload Atlas preparation.

These helpers carry task data and validate its retained identity. They issue no
provider disclosure, spending, scene admission, or execution authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .owned_capture_atlas_context import SCHEMA, validate_owned_capture_atlas_context


def validate_preparation_task_context(context: Mapping[str, Any]) -> None:
    """Dispatch existing source schemas without converting either source kind."""
    if context.get("schema_version") == SCHEMA:
        validate_owned_capture_atlas_context(context)
        return
    from .website_task_context_contracts import validate_website_task_context
    validate_website_task_context(context, request_id=str(context.get("request_id") or ""),
                                 scene_id=str(context.get("scene_id") or ""),
                                 capture_id=str(context.get("capture_id") or ""), purpose="scene_preparation")


def preparation_source_checkpoint(context: Mapping[str, Any]):
    """Owned provider adapters require the worker's fresh byte/source fence."""
    if context.get("schema_version") != SCHEMA:
        return None
    from .owned_capture_atlas_runtime import owned_atlas_source_checkpoint
    return owned_atlas_source_checkpoint(context)


def requires_task_semantic_analysis(context: Mapping[str, Any]) -> bool:
    """An owned task's declared family never bypasses reading its whole job."""
    return context.get("schema_version") == SCHEMA or context.get("task_family") == "general_task"


def preparation_context_digest_matches(context: Mapping[str, Any]) -> bool:
    """Keep each existing source contract's canonicalization and field checks."""
    try:
        if context.get("schema_version") == SCHEMA:
            validate_owned_capture_atlas_context(context)
            return True
        return context.get("context_digest") == canonical_digest(context, digest_field="context_digest")
    except (TypeError, ValueError):
        return False


def task_text_sources(context: Mapping[str, Any]) -> list[str]:
    """Quotes may come from any retained requirement, including ordered lists."""
    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Mapping):
            return [text for child in value.values() for text in strings(child)]
        if isinstance(value, (list, tuple)):
            return [text for child in value for text in strings(child)]
        return []

    keys = ["description", "success_criteria", "operator_answers", "operator_task_details"]
    if context.get("schema_version") == SCHEMA:
        keys.append("task_specification")
    return strings({key: context.get(key) for key in keys})
