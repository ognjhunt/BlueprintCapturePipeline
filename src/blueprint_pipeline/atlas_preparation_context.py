"""Shared task-reading rules for website and owned-upload Atlas preparation.

These helpers carry task data and validate its retained identity. They issue no
provider disclosure, spending, scene admission, or execution authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .owned_capture_atlas_context import SCHEMA, validate_owned_capture_atlas_context


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
