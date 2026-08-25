"""Exact coordinator release window for no-spend launch activation.

The window allows only profile publication, catalog synchronization, and one
standing authorization.  It never authorizes a paid request or a provider
allocation; those remain separate authenticated WebApp actions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_shared_mutation_window.v1"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "schemas"
    / "task_evaluation_shared_mutation_window.v1.schema.json"
)
ALLOWED_MUTATIONS = frozenset(
    {
        "profile_publication",
        "catalog_synchronization",
        "standing_authorization",
    }
)


class TaskEvaluationSharedMutationWindowError(ValueError):
    """A proposed shared mutation window is invalid or not currently usable."""


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise TaskEvaluationSharedMutationWindowError(
            f"shared_mutation_window_{field}_invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise TaskEvaluationSharedMutationWindowError(
            f"shared_mutation_window_{field}_invalid"
        )
    return parsed.astimezone(timezone.utc)


def validate_shared_mutation_window(
    value: Mapping[str, Any],
    *,
    activation_id: str,
    team_namespace: str,
    expected_production_commit: str,
    provider_allowlist: list[str],
    hard_cap_usd: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate one exact release window against the activation it releases."""

    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        ).validate(value)
    except (
        OSError,
        json.JSONDecodeError,
        jsonschema.SchemaError,
        jsonschema.ValidationError,
    ) as exc:
        raise TaskEvaluationSharedMutationWindowError(
            "shared_mutation_window_invalid"
        ) from exc
    window = dict(value)
    observed_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    issued = _parse_timestamp(window["issued_at"], field="issued_at")
    expires = _parse_timestamp(window["expires_at"], field="expires_at")
    if not issued <= observed_now < expires:
        raise TaskEvaluationSharedMutationWindowError(
            "shared_mutation_window_not_current"
        )
    if (
        window["activation_id"] != activation_id
        or window["team_namespace"] != team_namespace
        or window["expected_production_commit"] != expected_production_commit
        or set(window["allowed_mutations"]) != ALLOWED_MUTATIONS
        or window["provider_allowlist"] != provider_allowlist
        or float(window["maximum_hard_cap_usd"]) < float(hard_cap_usd)
        or window["window_digest"]
        != canonical_digest(window, digest_field="window_digest")
    ):
        raise TaskEvaluationSharedMutationWindowError(
            "shared_mutation_window_binding_mismatch"
        )
    return window


__all__ = [
    "ALLOWED_MUTATIONS",
    "SCHEMA_VERSION",
    "TaskEvaluationSharedMutationWindowError",
    "validate_shared_mutation_window",
]
