"""Pipeline adapter for the shared robot-eval job request contract.

BlueprintContracts owns the portable schema. This adapter keeps Pipeline on a
single local import path while the dependency pin rolls forward across repos.
The fallback constants match the published contract and exist only so older
installed blueprint-contracts pins do not break local development. CI and
production checks must call :func:`require_shared_robot_eval_job_request_contract`
or set ``BLUEPRINT_REQUIRE_SHARED_ROBOT_EVAL_CONTRACT=true`` so stale pins fail
closed instead of silently using Pipeline's fallback copy.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


_REQUIRE_SHARED_CONTRACT = _env_truthy(
    os.environ.get("BLUEPRINT_REQUIRE_SHARED_ROBOT_EVAL_CONTRACT")
)
_SHARED_CONTRACT_IMPORT_ERROR: BaseException | None = None


try:  # pragma: no cover - exercised when the newer BlueprintContracts pin lands.
    from blueprint_contracts.robot_eval_job_request_contract import (
        ROBOT_EVAL_JOB_REQUEST_INBOX_CONTRACT,
        ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION,
        robot_eval_job_request_inbox_schema,
        robot_eval_job_request_schema,
        validate_robot_eval_job_request_constants,
    )

    ROBOT_EVAL_JOB_REQUEST_CONTRACT_SOURCE = "blueprint_contracts"
except (ImportError, ModuleNotFoundError) as exc:
    _SHARED_CONTRACT_IMPORT_ERROR = exc
    if _REQUIRE_SHARED_CONTRACT:
        raise RuntimeError(
            "BlueprintContracts.robot_eval_job_request_contract is required. "
            "Install a BlueprintContracts revision that exports the shared "
            "robot_eval_job_request.v1 contract before running this strict gate."
        ) from exc
    ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION = "robot_eval_job_request.v1"
    ROBOT_EVAL_JOB_REQUEST_INBOX_CONTRACT = "robot_eval_job_request_inbox.v1"
    ROBOT_EVAL_JOB_REQUEST_CONTRACT_SOURCE = "pipeline_fallback"

    def robot_eval_job_request_schema() -> Dict[str, Any]:
        return {
            "schema_version": "pipeline_fallback_contract_pointer.v1",
            "canonical_source": "BlueprintContracts.robot_eval_job_request_contract",
            "job_request_schema_version": ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION,
        }

    def robot_eval_job_request_inbox_schema() -> Dict[str, Any]:
        return {
            "schema_version": "pipeline_fallback_contract_pointer.v1",
            "canonical_source": "BlueprintContracts.robot_eval_job_request_contract",
            "queue_contract": ROBOT_EVAL_JOB_REQUEST_INBOX_CONTRACT,
        }

    def validate_robot_eval_job_request_constants(payload: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        if payload.get("schema_version") != ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION:
            errors.append(
                f"schema_version must be {ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION}"
            )
        return errors


def require_shared_robot_eval_job_request_contract() -> None:
    """Raise if Pipeline is running on its local fallback contract copy."""

    if ROBOT_EVAL_JOB_REQUEST_CONTRACT_SOURCE == "blueprint_contracts":
        return
    detail = (
        f": {_SHARED_CONTRACT_IMPORT_ERROR}"
        if _SHARED_CONTRACT_IMPORT_ERROR is not None
        else ""
    )
    raise RuntimeError(
        "robot_eval_job_request.v1 must be loaded from BlueprintContracts for "
        f"strict launch/CI verification; current source is "
        f"{ROBOT_EVAL_JOB_REQUEST_CONTRACT_SOURCE}{detail}"
    )


__all__ = [
    "ROBOT_EVAL_JOB_REQUEST_CONTRACT_SOURCE",
    "ROBOT_EVAL_JOB_REQUEST_INBOX_CONTRACT",
    "ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION",
    "require_shared_robot_eval_job_request_contract",
    "robot_eval_job_request_inbox_schema",
    "robot_eval_job_request_schema",
    "validate_robot_eval_job_request_constants",
]
