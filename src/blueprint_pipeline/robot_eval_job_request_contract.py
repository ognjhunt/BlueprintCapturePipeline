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

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


_REQUIRE_SHARED_CONTRACT = _env_truthy(
    os.environ.get("BLUEPRINT_REQUIRE_SHARED_ROBOT_EVAL_CONTRACT")
)
_SHARED_CONTRACT_IMPORT_ERROR: BaseException | None = None


def _find_sibling_contracts_src_dir() -> Path | None:
    """Return a sibling BlueprintContracts/src checkout when one is available."""

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "BlueprintContracts" / "src"
        contract_file = (
            candidate / "blueprint_contracts" / "robot_eval_job_request_contract.py"
        )
        if contract_file.is_file():
            return candidate
    return None


def _schema_reader(schema_dir: Path, file_name: str) -> Dict[str, Any]:
    payload = json.loads((schema_dir / file_name).read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _load_sibling_contract_module(src_dir: Path) -> tuple[Any, str]:
    """Load the sibling shared contract even when an older wheel is installed."""

    module_path = src_dir / "blueprint_contracts" / "robot_eval_job_request_contract.py"
    schema_dir = src_dir / "blueprint_contracts" / "schemas"
    if not module_path.is_file() or not schema_dir.is_dir():
        raise ModuleNotFoundError(f"BlueprintContracts sibling is incomplete at {src_dir}")

    spec = importlib.util.spec_from_file_location(
        "_blueprint_contracts_robot_eval_job_request_contract_sibling",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Cannot load BlueprintContracts sibling at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def robot_eval_job_request_schema_from_sibling() -> Dict[str, Any]:
        return _schema_reader(schema_dir, module.ROBOT_EVAL_JOB_REQUEST_SCHEMA_FILE)

    def robot_eval_job_request_inbox_schema_from_sibling() -> Dict[str, Any]:
        return _schema_reader(schema_dir, module.ROBOT_EVAL_JOB_REQUEST_INBOX_SCHEMA_FILE)

    module.robot_eval_job_request_schema = robot_eval_job_request_schema_from_sibling
    module.robot_eval_job_request_inbox_schema = robot_eval_job_request_inbox_schema_from_sibling
    return module, f"blueprint_contracts_sibling:{src_dir}"


def _load_shared_contract_module() -> tuple[Any, str]:
    try:
        module = importlib.import_module(
            "blueprint_contracts.robot_eval_job_request_contract"
        )
        return module, "blueprint_contracts"
    except (ImportError, ModuleNotFoundError) as exc:
        sibling_src = _find_sibling_contracts_src_dir()
        if sibling_src is None:
            raise exc
        sibling_src_str = str(sibling_src)
        if sibling_src_str not in sys.path:
            sys.path.insert(0, sibling_src_str)
        return _load_sibling_contract_module(sibling_src)


try:  # pragma: no cover - exercised with installed or sibling BlueprintContracts.
    _SHARED_CONTRACT_MODULE, ROBOT_EVAL_JOB_REQUEST_CONTRACT_SOURCE = (
        _load_shared_contract_module()
    )
    ROBOT_EVAL_JOB_REQUEST_INBOX_CONTRACT = (
        _SHARED_CONTRACT_MODULE.ROBOT_EVAL_JOB_REQUEST_INBOX_CONTRACT
    )
    ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION = (
        _SHARED_CONTRACT_MODULE.ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION
    )
    robot_eval_job_request_inbox_schema = (
        _SHARED_CONTRACT_MODULE.robot_eval_job_request_inbox_schema
    )
    robot_eval_job_request_schema = (
        _SHARED_CONTRACT_MODULE.robot_eval_job_request_schema
    )
    validate_robot_eval_job_request_constants = (
        _SHARED_CONTRACT_MODULE.validate_robot_eval_job_request_constants
    )

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

    if ROBOT_EVAL_JOB_REQUEST_CONTRACT_SOURCE.startswith("blueprint_contracts"):
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
