from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.adp_task_evaluation_abstention import collect_vast_provider_zero_receipt
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_prelaunch_abstention import (
    TaskEvaluationPrelaunchAbstentionError,
    materialize_external_asset_owner_processing_authority,
    materialize_task_evaluation_prelaunch_abstention_supersession,
    materialize_task_evaluation_prelaunch_external_input_abstention,
    materialize_task_evaluation_prelaunch_freeze,
)


def _write(path: Path, value: dict) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _freeze() -> dict:
    return materialize_task_evaluation_prelaunch_freeze(
        {
            "schema_version": "task_evaluation_prelaunch_freeze.v1",
            "program_id": "arm-decision-proof-v1",
            "run_id": "run-1",
            "scene_id": "scene-1",
            "task_id": "task-1",
            "task_kind": "deformable_transfer",
            "prompt": "Move the object into the destination and retreat.",
            "candidate_ids": ["candidate-a", "candidate-b"],
            "cell_ids": ["canonical", "held-out"],
            "external_asset_id": "asset-1",
            "external_asset_archive_sha256": "sha256:" + "a" * 64,
            "selection_file_sha256": "sha256:" + "b" * 64,
            "placement_manifest_digest": "sha256:" + "c" * 64,
            "observation_file_sha256": "sha256:" + "d" * 64,
            "repository_commit": "e" * 40,
            "claim_ceiling": "simulator rehearsal only",
            "freeze_digest": "",
        }
    )


def _rights() -> dict:
    value = {
        "schema_version": "external_asset_rights.v1",
        "asset_id": "asset-1",
        "source_asset": {"archive_sha256": "sha256:" + "a" * 64},
        "admission": {
            "private_upload_to_vast_permitted": False,
            "redistribution_permitted": False,
        },
        "status": "blocked_missing_generated_output_rights",
        "typed_blocker": "generated_output_rights_missing",
        "smallest_external_resolution": "Provide the exact output license.",
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def _zero() -> dict:
    return collect_vast_provider_zero_receipt(
        command_runner=lambda argv: subprocess.CompletedProcess(
            argv, returncode=0, stdout="[]\n", stderr=""
        )
    )


def _owner_authority(statement: Path) -> dict:
    return materialize_external_asset_owner_processing_authority(
        {
            "schema_version": "external_asset_owner_processing_authority.v1",
            "authority_id": "workspace-owner-1",
            "authority_kind": "direct_asset_owner_attestation",
            "authority_reference": "Direct statement retained by the current task.",
            "asset_id": "asset-1",
            "asset_archive_sha256": "sha256:" + "a" * 64,
            "commissioned_and_paid_by_authority": True,
            "authority_represents_asset_ownership_or_processing_control": True,
            "private_processing_authorized": True,
            "permitted_provider_ids": ["vast"],
            "public_redistribution_authorized": False,
            "statement_file_sha256": "",
            "statement_size_bytes": 0,
            "claim_ceiling": "private simulator development processing only",
            "receipt_digest": "",
        },
        owner_statement_path=statement,
    )


def test_seals_external_rights_blocker_without_provider_or_policy_claim(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    rights = tmp_path / "rights.json"
    _write(freeze, _freeze())
    _write(rights, _rights())
    receipt = materialize_task_evaluation_prelaunch_external_input_abstention(
        freeze_path=freeze,
        rights_receipt_path=rights,
        provider_zero_receipt=_zero(),
    )
    assert receipt["status"] == "typed_evidence_backed_abstention"
    assert receipt["candidate_ids"] == ["candidate-a", "candidate-b"]
    assert receipt["all_terminal_blockers"] == ["generated_output_rights_missing"]
    assert receipt["provider_upload_performed"] is False
    assert receipt["paid_gpu_cost_usd"] == 0.0
    assert receipt["controls_executed"] is False
    assert receipt["comparison_exists"] is False
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["admission"].__setitem__("private_upload_to_vast_permitted", True),
        lambda value: value["source_asset"].__setitem__("archive_sha256", "sha256:" + "f" * 64),
        lambda value: value.__setitem__("typed_blocker", 7),
    ],
)
def test_rejects_rights_receipt_that_does_not_prove_exact_upload_blocker(
    tmp_path: Path, mutation
) -> None:
    freeze = tmp_path / "freeze.json"
    rights = tmp_path / "rights.json"
    _write(freeze, _freeze())
    value = _rights()
    mutation(value)
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    _write(rights, value)
    with pytest.raises(TaskEvaluationPrelaunchAbstentionError):
        materialize_task_evaluation_prelaunch_external_input_abstention(
            freeze_path=freeze,
            rights_receipt_path=rights,
            provider_zero_receipt=_zero(),
        )


def test_requires_exact_two_candidate_freeze() -> None:
    value = _freeze()
    value["candidate_ids"] = ["candidate-a"]
    value["freeze_digest"] = ""
    with pytest.raises(
        TaskEvaluationPrelaunchAbstentionError,
        match="prelaunch_freeze_candidate_pair_invalid",
    ):
        materialize_task_evaluation_prelaunch_freeze(value)


def test_rejects_fabricated_provider_zero(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    rights = tmp_path / "rights.json"
    _write(freeze, _freeze())
    _write(rights, _rights())
    zero = _zero()
    zero["inventory"] = [{"id": 123}]
    zero["provider_zero_digest"] = canonical_digest(zero, digest_field="provider_zero_digest")
    with pytest.raises(TaskEvaluationPrelaunchAbstentionError, match="provider_zero_invalid"):
        materialize_task_evaluation_prelaunch_external_input_abstention(
            freeze_path=freeze,
            rights_receipt_path=rights,
            provider_zero_receipt=zero,
        )


def test_fifo_input_fails_fast(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO unavailable")
    fifo = tmp_path / "freeze.json"
    os.mkfifo(fifo)
    rights = tmp_path / "rights.json"
    _write(rights, _rights())
    with pytest.raises(TaskEvaluationPrelaunchAbstentionError, match="file_invalid"):
        materialize_task_evaluation_prelaunch_external_input_abstention(
            freeze_path=fifo,
            rights_receipt_path=rights,
            provider_zero_receipt=_zero(),
        )


def test_paid_owner_authority_supersedes_only_private_processing_blocker(
    tmp_path: Path,
) -> None:
    freeze = tmp_path / "freeze.json"
    rights = tmp_path / "rights.json"
    statement = tmp_path / "owner_statement.txt"
    abstention = tmp_path / "abstention.json"
    authority = tmp_path / "owner_authority.json"
    statement.write_text("I commissioned, paid for, and own this exact asset.\n")
    _write(freeze, _freeze())
    _write(rights, _rights())
    _write(
        abstention,
        materialize_task_evaluation_prelaunch_external_input_abstention(
            freeze_path=freeze,
            rights_receipt_path=rights,
            provider_zero_receipt=_zero(),
        ),
    )
    _write(authority, _owner_authority(statement))
    receipt = materialize_task_evaluation_prelaunch_abstention_supersession(
        freeze_path=freeze,
        abstention_path=abstention,
        owner_authority_path=authority,
    )
    assert receipt["rights_gate_passed_for_private_vast_canary"] is True
    assert receipt["private_upload_to_vast_permitted"] is True
    assert receipt["public_redistribution_permitted"] is False
    assert receipt["paid_launch_authorized_by_this_receipt"] is False
    assert receipt["native_asset_qualified"] is False


def test_owner_authority_is_exact_asset_and_provider_scoped(tmp_path: Path) -> None:
    statement = tmp_path / "owner_statement.txt"
    statement.write_text("I own this asset and authorize private processing.\n")
    value = {
        "schema_version": "external_asset_owner_processing_authority.v1",
        "authority_id": "workspace-owner-1",
        "authority_kind": "direct_asset_owner_attestation",
        "authority_reference": "Direct task statement.",
        "asset_id": "asset-1",
        "asset_archive_sha256": "sha256:" + "a" * 64,
        "commissioned_and_paid_by_authority": True,
        "authority_represents_asset_ownership_or_processing_control": True,
        "private_processing_authorized": True,
        "permitted_provider_ids": ["some-other-provider"],
        "public_redistribution_authorized": False,
        "statement_file_sha256": "",
        "statement_size_bytes": 0,
        "claim_ceiling": "private simulation only",
        "receipt_digest": "",
    }
    with pytest.raises(
        TaskEvaluationPrelaunchAbstentionError,
        match="owner_authority_provider_scope_invalid",
    ):
        materialize_external_asset_owner_processing_authority(value, owner_statement_path=statement)
