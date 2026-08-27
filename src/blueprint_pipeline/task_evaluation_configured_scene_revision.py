"""Validation for immutable production-configured Task Evaluation scenes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_configured_scene_revision.v1"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "schemas"
    / "task_evaluation_configured_scene_revision.v1.schema.json"
)


class TaskEvaluationConfiguredSceneRevisionError(ValueError):
    """A configured scene revision is incomplete or internally inconsistent."""


@lru_cache(maxsize=1)
def configured_scene_revision_schema() -> dict[str, Any]:
    try:
        value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationConfiguredSceneRevisionError(
            "configured_scene_revision_schema_unavailable"
        ) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationConfiguredSceneRevisionError(
            "configured_scene_revision_schema_invalid"
        )
    # Imported here, not at module scope. The scene-configuration provider
    # bundle copies this package into an Isaac Sim container that ships no
    # ``jsonschema``, and reaches this module only transitively: the provider's
    # stage adapters import the orchestrator for one string constant and never
    # validate anything against a JSON Schema. A module-scope import therefore
    # killed the provider runner with ``ModuleNotFoundError: No module named
    # 'jsonschema'`` before its first stage, on a GPU that was already rented.
    # Same reason ``rfc8785`` is imported inside ``cross_runtime_canonical_json``.
    import jsonschema

    jsonschema.Draft202012Validator.check_schema(value)
    return dict(value)


def validate_configured_scene_revision(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    import jsonschema

    revision = dict(value)
    validator = jsonschema.Draft202012Validator(
        configured_scene_revision_schema(),
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(validator.iter_errors(revision), key=lambda row: list(row.path))
    if errors:
        path = ".".join(str(part) for part in errors[0].path) or "$"
        raise TaskEvaluationConfiguredSceneRevisionError(
            f"configured_scene_revision_invalid:{path}"
        )
    if revision["revision_digest"] != canonical_digest(
        revision, digest_field="revision_digest"
    ):
        raise TaskEvaluationConfiguredSceneRevisionError(
            "configured_scene_revision_digest_invalid"
        )
    source = revision["source"]
    disclosure = source.get("provider_disclosure_decision")
    if disclosure is not None:
        from .task_evaluation_scene_configuration_disclosure import (
            SCHEMA_VERSION as DISCLOSURE_SCHEMA_VERSION,
            renders_on_provider,
        )

        if (
            not isinstance(disclosure, Mapping)
            or disclosure.get("schema_version") != DISCLOSURE_SCHEMA_VERSION
            or disclosure.get("decision_digest")
            != canonical_digest(disclosure, digest_field="decision_digest")
            or source["raw_source_sent_to_external_provider"]
            is not renders_on_provider(disclosure)
        ):
            raise TaskEvaluationConfiguredSceneRevisionError(
                "configured_scene_revision_disclosure_invalid"
            )
    task = revision["task_template"]
    if task["identity"]["id"] == revision["scene_identity"]["id"]:
        raise TaskEvaluationConfiguredSceneRevisionError(
            "configured_scene_revision_task_scene_identity_conflict"
        )
    return revision


__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "TaskEvaluationConfiguredSceneRevisionError",
    "configured_scene_revision_schema",
    "validate_configured_scene_revision",
]
