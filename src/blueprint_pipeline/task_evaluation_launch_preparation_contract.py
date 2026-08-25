"""Typed, scene-neutral intake contract for production launch preparation.

This boundary accepts only immutable customer/team references and bounded
runtime requirements.  Production-owned paths, credentials, catalog files,
commands, and provider resources are intentionally absent and are resolved by
the production preparation service after admission.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_launch_preparation_request.v1"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "schemas"
    / "task_evaluation_launch_preparation_request.v1.schema.json"
)


class TaskEvaluationLaunchPreparationContractError(ValueError):
    """The external preparation request is unsafe or incomplete."""


@lru_cache(maxsize=1)
def preparation_request_schema() -> dict[str, Any]:
    try:
        value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationLaunchPreparationContractError(
            "launch_preparation_schema_unavailable"
        ) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchPreparationContractError(
            "launch_preparation_schema_invalid"
        )
    jsonschema.Draft202012Validator.check_schema(value)
    return dict(value)


def validate_launch_preparation_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy one customer-facing preparation request.

    JSON Schema closes the external surface.  These semantic checks protect
    the provider-neutral invariants that are awkward to express structurally.
    No file or network operation occurs here.
    """

    request = dict(value)
    validator = jsonschema.Draft202012Validator(
        preparation_request_schema(),
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(validator.iter_errors(request), key=lambda row: list(row.path))
    if errors:
        path = ".".join(str(part) for part in errors[0].path) or "$"
        raise TaskEvaluationLaunchPreparationContractError(
            f"launch_preparation_request_invalid:{path}"
        )

    scene_rights = request["scene"]["rights"]
    if (
        scene_rights["source_bytes_redistributable"] is False
        and scene_rights["provider_disclosure_scope"] == "source_and_derived"
    ):
        raise TaskEvaluationLaunchPreparationContractError(
            "launch_preparation_scene_disclosure_conflicts_with_rights"
        )
    if request["runtime"]["requirements"]["gpu_count"] < 1:
        raise TaskEvaluationLaunchPreparationContractError(
            "launch_preparation_gpu_requirement_missing"
        )
    adapter = request["execution_adapter"]
    if (adapter["kind"], adapter["version"]) != (
        "native_task_arena",
        "v1",
    ):
        raise TaskEvaluationLaunchPreparationContractError(
            "launch_preparation_execution_adapter_unavailable"
        )
    output_mounts = [
        mount for mount in request["runtime"]["mounts"] if mount["mode"] == "output"
    ]
    if len(output_mounts) != 1:
        raise TaskEvaluationLaunchPreparationContractError(
            "launch_preparation_output_mount_count_invalid"
        )
    mount_paths = [mount["container_path"] for mount in request["runtime"]["mounts"]]
    if len(mount_paths) != len(set(mount_paths)):
        raise TaskEvaluationLaunchPreparationContractError(
            "launch_preparation_mount_path_duplicate"
        )
    return request


def launch_preparation_request_digest(value: Mapping[str, Any]) -> str:
    """Return the immutable identity used across WebApp, worker, and receipts."""

    return canonical_digest(validate_launch_preparation_request(value))


__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "TaskEvaluationLaunchPreparationContractError",
    "launch_preparation_request_digest",
    "preparation_request_schema",
    "validate_launch_preparation_request",
]
