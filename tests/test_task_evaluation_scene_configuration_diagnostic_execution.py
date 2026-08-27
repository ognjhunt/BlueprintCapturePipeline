from __future__ import annotations

from pathlib import Path

import pytest

import blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_execution as execution_module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_execution import (
    PARENT_LAUNCH_SCHEMA_VERSION,
    STATUS,
    TaskEvaluationSceneConfigurationDiagnosticExecutionError,
    execute_scene_configuration_diagnostic_retry,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_runtime import (
    RESULT_SCHEMA_VERSION as CHAIN_SCHEMA_VERSION,
    STATUS as CHAIN_STATUS,
)
from blueprint_pipeline.task_evaluation_scene_configuration_orchestrator import (
    CANONICAL_ALLOCATOR,
)


def _launch(checkpoint_digest: str) -> dict:
    chain = {
        "schema_version": CHAIN_SCHEMA_VERSION,
        "status": CHAIN_STATUS,
        "source_checkpoint_digest": checkpoint_digest,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "executed_inside_one_parent_provider_run": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "result_digest": "",
    }
    chain["result_digest"] = canonical_digest(chain, digest_field="result_digest")
    value = {
        "schema_version": PARENT_LAUNCH_SCHEMA_VERSION,
        "status": STATUS,
        "explicit_diagnostic_resume_requested": True,
        "normal_production_lane_used": False,
        "canonical_allocator": CANONICAL_ALLOCATOR,
        "provider_mutations_performed": 1,
        "paid_execution_requested": True,
        "retry_cap": 0,
        "watchdog_armed_before_allocation": True,
        "teardown_completed": True,
        "provider_zero_confirmed": True,
        "source_checkpoint_digest": checkpoint_digest,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "executed_inside_one_parent_provider_run": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "raw_secret_values_recorded": False,
        "remaining_spend_hard_cap_usd": 3.0,
        "remaining_runtime_hard_ttl_seconds": 18_000,
        "remaining_spend_authority_digest": "sha256:" + "1" * 64,
        "billing_reconciliation_digest": "sha256:" + "2" * 64,
        "watchdog_receipt_digest": "sha256:" + "3" * 64,
        "teardown_digest": "sha256:" + "4" * 64,
        "provider_zero_digest": "sha256:" + "5" * 64,
        "launch_receipt_digest": "sha256:" + "6" * 64,
        "diagnostic_stage_chain": chain,
        "launch_digest": "",
    }
    value["launch_digest"] = canonical_digest(value, digest_field="launch_digest")
    return value


def test_diagnostic_execution_requires_separate_spend_watchdog_and_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_digest = "sha256:" + "c" * 64
    monkeypatch.setattr(
        execution_module,
        "validate_scene_configuration_diagnostic_checkpoint",
        lambda **_kwargs: {"checkpoint_digest": checkpoint_digest},
    )
    result = execute_scene_configuration_diagnostic_retry(
        checkpoint_root=tmp_path / "checkpoint",
        output_root=tmp_path,
        diagnostic_parent_launch_executor=lambda **_kwargs: _launch(checkpoint_digest),
    )

    assert result["status"] == STATUS
    assert result["diagnostic_only"] is True
    assert result["qualification_eligible"] is False
    assert result["executed_inside_one_parent_provider_run"] is False
    assert result["configured_scene_revision"] is None
    assert result["configured_scene_offering"] is None
    assert result["terminal_e2e_status"] is None
    assert result["watchdog_armed_before_allocation"] is True
    assert result["provider_zero_confirmed"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("watchdog_armed_before_allocation", False),
        ("teardown_completed", False),
        ("provider_zero_confirmed", False),
        ("remaining_spend_hard_cap_usd", 0),
        ("remaining_runtime_hard_ttl_seconds", 0),
        ("configured_revision_publication_permitted", True),
    ],
)
def test_diagnostic_execution_fails_closed_without_closure_or_claim_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    checkpoint_digest = "sha256:" + "c" * 64
    monkeypatch.setattr(
        execution_module,
        "validate_scene_configuration_diagnostic_checkpoint",
        lambda **_kwargs: {"checkpoint_digest": checkpoint_digest},
    )
    launch = _launch(checkpoint_digest)
    launch[field] = value
    launch["launch_digest"] = canonical_digest(launch, digest_field="launch_digest")

    with pytest.raises(
        TaskEvaluationSceneConfigurationDiagnosticExecutionError,
        match="scene_configuration_diagnostic_parent_launch_invalid",
    ):
        execute_scene_configuration_diagnostic_retry(
            checkpoint_root=tmp_path / "checkpoint",
            output_root=tmp_path,
            diagnostic_parent_launch_executor=lambda **_kwargs: launch,
        )
