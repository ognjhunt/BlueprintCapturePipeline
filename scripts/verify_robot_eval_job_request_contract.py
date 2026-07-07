#!/usr/bin/env python3
"""Verify Pipeline is using the shared robot-eval job request contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "robot_eval_job_request.v1"
INBOX_CONTRACT = "robot_eval_job_request_inbox.v1"


def _contracts_repo_candidates(explicit: str | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("BLUEPRINT_CONTRACTS_REPO")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "BlueprintContracts",
            REPO_ROOT.parent / "BlueprintContracts",
        ]
    )
    return candidates


def _add_contracts_repo_to_path(explicit: str | None) -> str | None:
    for candidate in _contracts_repo_candidates(explicit):
        src_dir = candidate / "src"
        if src_dir.is_dir():
            src = str(src_dir.resolve())
            if src not in sys.path:
                sys.path.insert(0, src)
            return str(candidate.resolve())
    return None


def _fixture_request() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "proof_boundary": {
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "robot_policy_execution_proven": False,
            "physics_contact_validated": False,
            "non_ranking_operational_claim_validated": False,
            "virtual_evaluation_proves_evaluation_readiness": False,
            "virtual_evaluation_proves_non_ranking_operational_claim": False,
            "public_claim_upgrade_allowed": False,
        },
        "execution_request": {
            "webapp_role": "queue_and_forward_only",
            "scheduler_owner": "BlueprintCapturePipeline",
        },
    }


def _schema_const(schema: Mapping[str, Any], field: str) -> Any:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return None
    field_schema = properties.get(field)
    if not isinstance(field_schema, Mapping):
        return None
    return field_schema.get("const")


def verify_contract(contracts_repo: str | None) -> dict[str, Any]:
    resolved_contracts_repo = _add_contracts_repo_to_path(contracts_repo)
    previous_strict = os.environ.get("BLUEPRINT_REQUIRE_SHARED_ROBOT_EVAL_CONTRACT")
    os.environ["BLUEPRINT_REQUIRE_SHARED_ROBOT_EVAL_CONTRACT"] = "true"
    try:
        from blueprint_pipeline import robot_eval_job_request_contract as contract
    finally:
        if previous_strict is None:
            os.environ.pop("BLUEPRINT_REQUIRE_SHARED_ROBOT_EVAL_CONTRACT", None)
        else:
            os.environ["BLUEPRINT_REQUIRE_SHARED_ROBOT_EVAL_CONTRACT"] = previous_strict

    errors: list[str] = []
    try:
        contract.require_shared_robot_eval_job_request_contract()
    except RuntimeError as exc:
        errors.append(str(exc))

    if contract.ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION != SCHEMA_VERSION:
        errors.append(
            "ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION drifted from "
            f"{SCHEMA_VERSION}"
        )
    if contract.ROBOT_EVAL_JOB_REQUEST_INBOX_CONTRACT != INBOX_CONTRACT:
        errors.append(
            "ROBOT_EVAL_JOB_REQUEST_INBOX_CONTRACT drifted from "
            f"{INBOX_CONTRACT}"
        )

    request_schema = contract.robot_eval_job_request_schema()
    inbox_schema = contract.robot_eval_job_request_inbox_schema()
    if request_schema.get("$id") != (
        "https://schemas.tryblueprint.io/robot_eval_job_request.v1.schema.json"
    ):
        errors.append("robot_eval_job_request schema $id is missing or unexpected")
    if _schema_const(request_schema, "schema_version") != SCHEMA_VERSION:
        errors.append("robot_eval_job_request schema_version const is missing")
    if _schema_const(inbox_schema, "queue_contract") != INBOX_CONTRACT:
        errors.append("robot_eval_job_request_inbox queue_contract const is missing")

    valid_errors = contract.validate_robot_eval_job_request_constants(_fixture_request())
    if valid_errors:
        errors.append(f"valid fixture failed shared constants guard: {valid_errors}")

    invalid = _fixture_request()
    invalid["proof_boundary"]["simulator_execution_proven"] = True
    invalid_errors = contract.validate_robot_eval_job_request_constants(invalid)
    if "proof_boundary.simulator_execution_proven must be false" not in invalid_errors:
        errors.append("shared constants guard did not reject upgraded proof boundary")

    return {
        "schema_version": "pipeline_robot_eval_job_request_contract_verification.v1",
        "status": "passed" if not errors else "blocked",
        "contract_source": contract.ROBOT_EVAL_JOB_REQUEST_CONTRACT_SOURCE,
        "contracts_repo": resolved_contracts_repo,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed unless Pipeline imports robot_eval_job_request.v1 from "
            "BlueprintContracts."
        )
    )
    parser.add_argument(
        "--contracts-repo",
        help="Optional path to a BlueprintContracts checkout.",
    )
    args = parser.parse_args(argv)
    try:
        report = verify_contract(args.contracts_repo)
    except Exception as exc:
        report = {
            "schema_version": "pipeline_robot_eval_job_request_contract_verification.v1",
            "status": "blocked",
            "contract_source": "import_failed",
            "contracts_repo": args.contracts_repo,
            "errors": [str(exc)],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
