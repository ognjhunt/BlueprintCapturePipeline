"""Typed authority boundary from verified preparation to launch publication.

The external request contains only immutable object-store references and human
intent.  Production paths, service identities, catalog destinations, provider
credentials, and allocator commands stay server-owned.  Validation performs no
publication, standing-authorization write, provider call, or paid execution.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonschema

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_launch_activation_request.v1"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "schemas"
    / "task_evaluation_launch_activation_request.v1.schema.json"
)


class TaskEvaluationLaunchActivationContractError(ValueError):
    """One activation request is unsafe, incomplete, or internally inconsistent."""


def activation_request_schema() -> dict[str, Any]:
    try:
        value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationLaunchActivationContractError(
            "launch_activation_schema_unavailable"
        ) from exc
    try:
        jsonschema.Draft202012Validator.check_schema(value)
    except jsonschema.SchemaError as exc:
        raise TaskEvaluationLaunchActivationContractError(
            "launch_activation_schema_invalid"
        ) from exc
    return value


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_launch_activation_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and copy one customer-facing activation request without mutation."""

    request = deepcopy(dict(value))
    validator = jsonschema.Draft202012Validator(
        activation_request_schema(),
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(validator.iter_errors(request), key=lambda row: list(row.path))
    if errors:
        path = ".".join(str(part) for part in errors[0].path) or "$"
        raise TaskEvaluationLaunchActivationContractError(
            f"launch_activation_request_invalid:{path}"
        )
    authorized_on = _parse_datetime(request["authorization"]["authorized_on"])
    expires_at = _parse_datetime(
        request["authorization"]["standing_authorization_expires_at"]
    )
    if authorized_on >= expires_at:
        raise TaskEvaluationLaunchActivationContractError(
            "launch_activation_authorization_window_invalid"
        )
    if request["activation_id"] == request["preparation"]["preparation_id"]:
        raise TaskEvaluationLaunchActivationContractError(
            "launch_activation_identity_not_independent"
        )
    return request


def launch_activation_request_digest(value: Mapping[str, Any]) -> str:
    """Return the immutable identity carried through WebApp and Pipeline."""

    return canonical_digest(validate_launch_activation_request(value))


__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "TaskEvaluationLaunchActivationContractError",
    "activation_request_schema",
    "launch_activation_request_digest",
    "validate_launch_activation_request",
]
