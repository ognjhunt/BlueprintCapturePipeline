from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline import robot_eval_job_request_contract as contract
from scripts.verify_robot_eval_job_request_contract import verify_contract


def _find_contracts_repo() -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    for candidate in (repo_root / "BlueprintContracts", repo_root.parent / "BlueprintContracts"):
        if (candidate / "src" / "blueprint_contracts" / "robot_eval_job_request_contract.py").is_file():
            return candidate
    return None


def test_robot_eval_job_request_contract_adapter_exposes_shared_constants() -> None:
    assert contract.ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION == "robot_eval_job_request.v1"
    assert contract.ROBOT_EVAL_JOB_REQUEST_INBOX_CONTRACT == "robot_eval_job_request_inbox.v1"
    assert contract.ROBOT_EVAL_JOB_REQUEST_CONTRACT_SOURCE in {
        "blueprint_contracts",
        "pipeline_fallback",
    }
    if contract.ROBOT_EVAL_JOB_REQUEST_CONTRACT_SOURCE == "blueprint_contracts":
        assert (
            contract.robot_eval_job_request_schema().get("$id")
            == "https://schemas.tryblueprint.io/robot_eval_job_request.v1.schema.json"
        )
    else:
        assert (
            contract.robot_eval_job_request_schema().get("canonical_source")
            == "BlueprintContracts.robot_eval_job_request_contract"
        )


def test_robot_eval_job_request_contract_adapter_validates_schema_constant() -> None:
    valid = {
        "schema_version": contract.ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION,
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
    assert (
        contract.validate_robot_eval_job_request_constants(valid)
        == []
    )
    assert contract.validate_robot_eval_job_request_constants({"schema_version": "wrong"})[0] == (
        "schema_version must be robot_eval_job_request.v1"
    )


def test_robot_eval_job_request_contract_strict_requirement_passes_when_module_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag set + shared module installed -> adapter imports strict without raising."""

    import importlib

    monkeypatch.setenv("BLUEPRINT_REQUIRE_SHARED_ROBOT_EVAL_CONTRACT", "true")
    try:
        reloaded = importlib.reload(contract)
        if reloaded.ROBOT_EVAL_JOB_REQUEST_CONTRACT_SOURCE != "blueprint_contracts":
            pytest.skip("blueprint_contracts is not installed in this environment")
        # Must NOT raise: strict mode is satisfied because the shared module loaded.
        reloaded.require_shared_robot_eval_job_request_contract()
        assert (
            reloaded.ROBOT_EVAL_JOB_REQUEST_SCHEMA_VERSION == "robot_eval_job_request.v1"
        )
    finally:
        # Restore the module to its default (flag-unset) state for later tests.
        monkeypatch.delenv("BLUEPRINT_REQUIRE_SHARED_ROBOT_EVAL_CONTRACT", raising=False)
        importlib.reload(contract)


def test_robot_eval_job_request_contract_strict_requirement_rejects_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contract, "ROBOT_EVAL_JOB_REQUEST_CONTRACT_SOURCE", "pipeline_fallback"
    )
    monkeypatch.setattr(
        contract, "_SHARED_CONTRACT_IMPORT_ERROR", ModuleNotFoundError("missing")
    )
    with pytest.raises(RuntimeError, match="must be loaded from BlueprintContracts"):
        contract.require_shared_robot_eval_job_request_contract()


def test_robot_eval_job_request_contract_verifier_uses_blueprint_contracts() -> None:
    contracts_repo = _find_contracts_repo()
    if contracts_repo is None:
        pytest.skip("BlueprintContracts checkout with robot-eval contract is unavailable")
    report = verify_contract(str(contracts_repo))
    assert report["status"] == "passed"
    assert report["contract_source"] == "blueprint_contracts"
    assert report["errors"] == []
