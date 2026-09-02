"""Resolve authenticated Task Evaluation artifacts across durable run stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .core.security_controls import strict_identifier
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


def resolve_live_pipeline_result_artifact(
    *,
    legacy_state_root: str | Path,
    policy_canary_result_root: str | Path | None,
    run_id: str,
    artifact_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve legacy runs first, then the exact canary activation mapping."""

    run = strict_identifier(run_id, field="run_id", max_length=192)
    legacy_root = Path(legacy_state_root).expanduser().resolve() / "runs" / run
    legacy_registry = (
        legacy_root / "artifacts" / "result_delivery" / "artifact_registry.json"
    )
    if legacy_registry.is_file() or legacy_registry.is_symlink():
        selected_root = legacy_root
    else:
        selected_root = _policy_canary_run_root(
            configured_root=policy_canary_result_root,
            run_id=run,
        ) or legacy_root
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
