"""Exact coordinator release window for no-spend launch activation.

The window allows only profile publication, catalog synchronization, and one
standing authorization.  It never authorizes a paid request or a provider
allocation; those remain separate authenticated WebApp actions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import jsonschema

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_shared_mutation_window.v1"
TEMPLATE_SCHEMA_VERSION = (
    "task_evaluation_configured_controls_release_window_template.v1"
)
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
_PROVIDERS = frozenset({"vast", "runpod", "gcp", "aws", "azure"})
_TEMPLATE_FIELDS = {
    "schema_version",
    "status",
    "team_namespace",
    "expected_production_commit",
    "allowed_mutations",
    "provider_allowlist",
    "maximum_hard_cap_usd",
    "valid_for_seconds",
    "released_by",
    "release_reference",
    "provider_resource_allocation_allowed",
    "paid_request_allowed",
    "template_digest",
}


class TaskEvaluationSharedMutationWindowError(ValueError):
    """A proposed shared mutation window is invalid or not currently usable."""


def validate_shared_mutation_window_template(
    value: Mapping[str, Any],
    *,
    team_namespace: str | None = None,
    expected_production_commit: str | None = None,
) -> dict[str, Any]:
    """Validate coordinator authority without pretending a future intent exists."""

    try:
        template = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationSharedMutationWindowError(
            "shared_mutation_window_template_invalid"
        ) from exc
    providers = template.get("provider_allowlist")
    ttl = template.get("valid_for_seconds")
    cap = template.get("maximum_hard_cap_usd")
    if (
        set(template) != _TEMPLATE_FIELDS
        or template.get("schema_version") != TEMPLATE_SCHEMA_VERSION
        or template.get("status") != "authorized_for_dynamic_release"
        or not str(template.get("team_namespace") or "").strip()
        or not isinstance(template.get("expected_production_commit"), str)
        or len(template["expected_production_commit"]) != 40
        or any(character not in "0123456789abcdef" for character in template["expected_production_commit"])
        or set(template.get("allowed_mutations") or []) != ALLOWED_MUTATIONS
        or not isinstance(providers, list)
        or not providers
        or len(providers) != len(set(providers))
        or not set(providers) <= _PROVIDERS
        or not isinstance(cap, (int, float))
        or isinstance(cap, bool)
        or not 0.0 < float(cap) <= 50.0
        or not isinstance(ttl, int)
        or isinstance(ttl, bool)
        or not 60 <= ttl <= 604_800
        or not str(template.get("released_by") or "").strip()
        or not str(template.get("release_reference") or "").strip()
        or template.get("provider_resource_allocation_allowed") is not False
        or template.get("paid_request_allowed") is not False
        or template.get("template_digest")
        != canonical_digest(template, digest_field="template_digest")
        or (
            team_namespace is not None
            and template.get("team_namespace") != team_namespace
        )
        or (
            expected_production_commit is not None
            and template.get("expected_production_commit")
            != expected_production_commit
        )
    ):
        raise TaskEvaluationSharedMutationWindowError(
            "shared_mutation_window_template_invalid"
        )
    return template


def materialize_shared_mutation_window(
    template_value: Mapping[str, Any],
    *,
    activation_request: Mapping[str, Any],
    provider_allowlist: list[str],
    hard_cap_usd: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create the exact current window only after the activation is knowable."""

    from .task_evaluation_launch_activation_contract import (
        launch_activation_intent_digest,
        validate_launch_activation_request,
    )

    request = validate_launch_activation_request(activation_request)
    template = validate_shared_mutation_window_template(
        template_value,
        team_namespace=request["team_namespace"],
        expected_production_commit=request["expected_production_commit"],
    )
    if (
        template["provider_allowlist"] != provider_allowlist
        or float(template["maximum_hard_cap_usd"]) < float(hard_cap_usd)
    ):
        raise TaskEvaluationSharedMutationWindowError(
            "shared_mutation_window_template_spend_mismatch"
        )
    issued = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    intent_digest = launch_activation_intent_digest(request)
    identity = canonical_digest(
        {
            "activation_id": request["activation_id"],
            "activation_intent_digest": intent_digest,
            "template_digest": template["template_digest"],
            "issued_at": issued.isoformat(),
        }
    ).removeprefix("sha256:")[:32]
    window = {
        "schema_version": SCHEMA_VERSION,
        "status": "released",
        "window_id": f"window-{identity}",
        "activation_id": request["activation_id"],
        "activation_intent_digest": intent_digest,
        "team_namespace": request["team_namespace"],
        "expected_production_commit": request["expected_production_commit"],
        "allowed_mutations": sorted(ALLOWED_MUTATIONS),
        "provider_allowlist": list(provider_allowlist),
        "maximum_hard_cap_usd": float(template["maximum_hard_cap_usd"]),
        "issued_at": issued.isoformat(),
        "expires_at": (
            issued + timedelta(seconds=template["valid_for_seconds"])
        ).isoformat(),
        "released_by": template["released_by"],
        "release_reference": template["release_reference"],
        "provider_resource_allocation_allowed": False,
        "paid_request_allowed": False,
        "window_digest": "",
    }
    window["window_digest"] = canonical_digest(window, digest_field="window_digest")
    return validate_shared_mutation_window(
        window,
        activation_id=request["activation_id"],
        activation_intent_digest=intent_digest,
        team_namespace=request["team_namespace"],
        expected_production_commit=request["expected_production_commit"],
        provider_allowlist=provider_allowlist,
        hard_cap_usd=hard_cap_usd,
        now=issued,
    )


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
    activation_intent_digest: str,
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
        or window["activation_intent_digest"] != activation_intent_digest
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
    "TEMPLATE_SCHEMA_VERSION",
    "TaskEvaluationSharedMutationWindowError",
    "materialize_shared_mutation_window",
    "validate_shared_mutation_window",
    "validate_shared_mutation_window_template",
]
