from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.adp_task_evaluation_abstention import collect_vast_provider_zero_receipt
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_prelaunch_abstention import (
    TaskEvaluationPrelaunchAbstentionError,
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
