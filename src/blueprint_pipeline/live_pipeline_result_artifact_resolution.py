"""Resolve authenticated Task Evaluation artifacts across durable run stores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core.security_controls import strict_identifier
from .decision_evidence_contracts import cross_runtime_canonical_digest
from .task_evaluation_result_delivery import (
    TaskEvaluationResultDeliveryError,
    resolve_task_evaluation_result_artifact,
)


TASK_EVALUATION_POLICY_CANARY_RESULT_ROOT_ENV = (
    "BLUEPRINT_TASK_EVALUATION_POLICY_CANARY_RESULT_ROOT"
)


def _policy_canary_run_root(
    *, configured_root: str | Path | None, run_id: str
) -> Path | None:
    configured = str(configured_root or "").strip()
    if not configured:
        return None
    root = Path(configured).expanduser()
    if not root.is_absolute() or root.is_symlink():
        raise TaskEvaluationResultDeliveryError("policy_canary_result_root_invalid")
    resolved_root = root.resolve()
    activation_id = strict_identifier(
        f"{run_id}-activation",
        field="policy_canary_activation_id",
        max_length=224,
    )
    unresolved = resolved_root / activation_id
    if unresolved.is_symlink():
        raise TaskEvaluationResultDeliveryError(
            "policy_canary_result_activation_symlink_forbidden"
        )
    resolved = unresolved.resolve(strict=False)
    if resolved.parent != resolved_root or resolved.name != activation_id:
        raise TaskEvaluationResultDeliveryError(
            "policy_canary_result_activation_mapping_invalid"
        )
    return resolved


def _registered_operator_run_root(*, activation_root: Path | None, run_id: str) -> Path | None:
    if activation_root is None:
        return None
    unresolved = activation_root.parent / run_id
    if unresolved.is_symlink():
        raise TaskEvaluationResultDeliveryError("operator_result_run_symlink_forbidden")
    root = unresolved.resolve(strict=False)
    if root.parent != activation_root.parent or root.name != run_id:
        raise TaskEvaluationResultDeliveryError("operator_result_run_mapping_invalid")
    registration_path = root / "website-operator-registration.json"
    if registration_path.is_symlink():
        raise TaskEvaluationResultDeliveryError("operator_result_registration_symlink_forbidden")
    if not registration_path.is_file():
        return None
    if registration_path.stat().st_size > 1024 * 1024:
        raise TaskEvaluationResultDeliveryError("operator_result_registration_invalid")
    try:
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        valid = (
            isinstance(registration, dict)
            and registration.get("schema_version") == "task_evaluation_operator_policy_canary_registration.v1"
            and registration.get("run_id") == run_id
            and registration.get("run_kind") == "internal_policy_canary"
            and registration.get("claim_ceiling") == "diagnostic_policy_execution"
            and registration.get("registration_digest")
            == cross_runtime_canonical_digest(registration, digest_field="registration_digest")
        )
    except (OSError, ValueError, TypeError):
        valid = False
    if not valid:
        raise TaskEvaluationResultDeliveryError("operator_result_registration_invalid")
    activation_registry = activation_root / "artifacts/result_delivery/artifact_registry.json"
    if activation_registry.is_file() or activation_registry.is_symlink():
        raise TaskEvaluationResultDeliveryError("policy_canary_result_run_mapping_ambiguous")
    return root


def resolve_live_pipeline_result_artifact(
    *,
    legacy_state_root: str | Path,
    policy_canary_result_root: str | Path | None,
    run_id: str,
    artifact_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve legacy, queued-canary, or registered operator run stores."""

    run = strict_identifier(run_id, field="run_id", max_length=192)
    legacy_root = Path(legacy_state_root).expanduser().resolve() / "runs" / run
    legacy_registry = (
        legacy_root / "artifacts" / "result_delivery" / "artifact_registry.json"
    )
    if legacy_registry.is_file() or legacy_registry.is_symlink():
        selected_root = legacy_root
    else:
        activation_root = _policy_canary_run_root(
            configured_root=policy_canary_result_root,
            run_id=run,
        )
        selected_root = _registered_operator_run_root(
            activation_root=activation_root, run_id=run
        ) or activation_root or legacy_root
    return resolve_task_evaluation_result_artifact(
        run_root=selected_root,
        run_id=run,
        artifact_id=artifact_id,
    )


__all__ = [
    "TASK_EVALUATION_POLICY_CANARY_RESULT_ROOT_ENV",
    "TaskEvaluationResultDeliveryError",
    "resolve_live_pipeline_result_artifact",
]
