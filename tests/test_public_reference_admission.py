from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_reference_admission import (
    PublicReferenceAdmissionError,
    build_public_reference_admission_receipt,
)


MANIFEST = (
    Path(__file__).parents[1]
    / "docs"
    / "arm_decision_proof_v1"
    / "manifests"
    / "simpler_google_robot_pick_coke_can.v1.json"
)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_simpler_manifest_retains_exact_source_and_outcome_joins() -> None:
    receipt = build_public_reference_admission_receipt(_manifest())

    assert receipt["status"] == "admitted"
    assert receipt["blockers"] == []
    assert receipt["phase_label"] == "retrospective_external_reference"
    assert receipt["claim_ceiling"] == "development_only"
    assert receipt["physical_reference_cell_count"] == 6
    assert receipt["exact_candidate_condition_join_available"] is True
    assert len(receipt["candidate_bindings"]) == 2
    assert len({row["checkpoint_identity_digest"] for row in receipt["candidate_bindings"]}) == 2


def test_paid_runtime_canary_cannot_override_zero_spend_blocker_when_tampered() -> None:
    value = _manifest()
    value.pop("manifest_digest")
    value["runtime"]["paid_runtime_canary"]["provider_zero_verified"] = False

    receipt = build_public_reference_admission_receipt(value)

    assert receipt["status"] == "blocked"
    assert "paid_runtime_canary_provider_zero_verified_not_true" in receipt["blockers"]
    assert (
        "zero_spend_feasibility_not_passed:blocked_host_incompatible_no_nvidia_cuda"
        in receipt["blockers"]
    )


def test_runtime_readiness_is_derived_not_caller_asserted() -> None:
    value = _manifest()
    value.pop("manifest_digest")
    value["runtime"]["environment_lock"] = {
        "status": "exact_immutable",
        "digest": "sha256:" + "a" * 64,
    }
    value["runtime"]["zero_spend_feasibility"]["status"] = "passed"

    receipt = build_public_reference_admission_receipt(value)

    assert receipt["status"] == "admitted"
    assert receipt["qualified_execution_ready"] is True
    assert receipt["blockers"] == []


def test_missing_candidate_condition_cell_fails_closed() -> None:
    value = _manifest()
    value.pop("manifest_digest")
    value["physical_reference"]["cell_bindings"].pop()

    with pytest.raises(PublicReferenceAdmissionError, match="missing_pairs"):
        build_public_reference_admission_receipt(value)


def test_duplicate_or_non_genuine_candidates_fail_closed() -> None:
    value = _manifest()
    value.pop("manifest_digest")
    value["candidates"][1] = copy.deepcopy(value["candidates"][0])
    value["candidates"][1]["genuine_public_checkpoint"] = False

    with pytest.raises(PublicReferenceAdmissionError) as caught:
        build_public_reference_admission_receipt(value)

    assert "candidates:candidate_id_duplicate" in caught.value.errors
    assert "candidates:checkpoint_identity_duplicate" in caught.value.errors
    assert "candidates[1].genuine_public_checkpoint:must_be_true" in caught.value.errors


def test_manifest_digest_rejects_late_source_change() -> None:
    value = _manifest()
    assert value["manifest_digest"] == canonical_digest(value, digest_field="manifest_digest")
    value["physical_reference"]["cell_bindings"][0]["trial_count"] = 24

    with pytest.raises(PublicReferenceAdmissionError, match="manifest_digest:mismatch"):
        build_public_reference_admission_receipt(value)
