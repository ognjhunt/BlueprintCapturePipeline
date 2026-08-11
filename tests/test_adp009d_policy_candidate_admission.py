from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_policy_candidate_admission import (
    Adp009dPolicyAdmissionError,
    freeze_policy_candidate_selection,
    validate_policy_candidate_inventory,
    validate_policy_runtime_admission,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    REPO_ROOT
    / "docs/arm_decision_proof_v1/manifests/adp009d_policy_candidate_inventory.v1.json"
)


def _inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def _candidate(inventory: dict, candidate_id: str) -> dict:
    return next(
        row for row in inventory["candidates"] if row["candidate_id"] == candidate_id
    )


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _runtime_admission(candidate: dict, character: str) -> dict:
    receipt = {
        "schema_version": "adp009d_policy_candidate_runtime_admission.v1",
        "candidate_id": candidate["candidate_id"],
        "candidate_digest": candidate["candidate_digest"],
        "checkpoint_materialization_digest": _sha(character),
        "checkpoint_tree_sha256": _sha(chr(ord(character) + 1)),
        "observation_adapter_digest": _sha(chr(ord(character) + 2)),
        "action_adapter_digest": _sha(chr(ord(character) + 3)),
        "camera_calibration_digest": _sha(chr(ord(character) + 4)),
        "immutable_smoke_input_digest": _sha(chr(ord(character) + 5)),
        "immutable_smoke_output_digest": _sha(chr(ord(character) + 6)),
        "runtime_environment_digest": _sha(chr(ord(character) + 7)),
        "checkpoint_all_files_verified": True,
        "live_policy_frames_used": True,
        "action_adapter_native_probe_passed": True,
        "immutable_smoke_passed": True,
        "task_outcomes_observed": False,
        "blockers": [],
        "admitted": True,
        "admission_digest": "",
    }
    receipt["admission_digest"] = canonical_digest(
        receipt, digest_field="admission_digest"
    )
    return receipt


def test_checked_in_inventory_binds_all_four_before_selection() -> None:
    inventory = validate_policy_candidate_inventory(_inventory())

    assert inventory["candidate_selection"]["frozen"] is False
    assert inventory["learned_task_outcomes_observed"] is False
    assert {row["candidate_id"] for row in inventory["candidates"]} == {
        "pi05_droid",
        "groot_n17_droid",
        "groot_n16_droid",
        "cosmos3_edge_policy_droid",
    }
    cosmos = _candidate(inventory, "cosmos3_edge_policy_droid")
    assert cosmos["checkpoint"]["revision"] == (
        "3ea407af3e156c0af3b4bb6edd85842cc9a58777"
    )
    assert cosmos["action_contract"]["card_canonical"]["shape"] == [16, 8]
    assert cosmos["action_contract"]["official_server_default"]["shape"] == [
        32,
        8,
    ]
    assert "cosmos3_edge_action_chunk_variant_not_frozen" in cosmos[
        "current_admission"
    ]["blockers"]


def test_inventory_rejects_checkpoint_or_outcome_tamper() -> None:
    checkpoint_tamper = _inventory()
    _candidate(checkpoint_tamper, "groot_n17_droid")["checkpoint"][
        "total_bytes"
    ] += 1
    with pytest.raises(
        Adp009dPolicyAdmissionError,
        match="policy_groot_n17_droid_checkpoint_total_bytes_invalid",
    ):
        validate_policy_candidate_inventory(checkpoint_tamper)

    outcome_tamper = _inventory()
    outcome_tamper["task_success"] = True
    outcome_tamper["inventory_digest"] = canonical_digest(
        outcome_tamper, digest_field="inventory_digest"
    )
    with pytest.raises(
        Adp009dPolicyAdmissionError,
        match="policy_inventory_caller_asserted_outcome_forbidden",
    ):
        validate_policy_candidate_inventory(outcome_tamper)


def test_runtime_admission_rejects_prepared_or_fake_smoke() -> None:
    inventory = _inventory()
    candidate = _candidate(inventory, "groot_n17_droid")
    receipt = _runtime_admission(candidate, "1")
    receipt["immutable_smoke_passed"] = False
    receipt["admission_digest"] = canonical_digest(
        receipt, digest_field="admission_digest"
    )

    with pytest.raises(
        Adp009dPolicyAdmissionError,
        match="policy_runtime_admission_smoke_missing",
    ):
        validate_policy_runtime_admission(receipt, candidate=candidate)


def test_selection_requires_exactly_two_real_runtime_admissions() -> None:
    inventory = _inventory()
    with pytest.raises(
        Adp009dPolicyAdmissionError,
        match="policy_selection_pi05_droid_runtime_admission_missing",
    ):
        freeze_policy_candidate_selection(
            inventory=inventory,
            selected_candidate_ids=["pi05_droid", "groot_n17_droid"],
            runtime_admissions={},
            protocol_request_digest=_sha("a"),
        )

    pi05 = _candidate(inventory, "pi05_droid")
    n17 = _candidate(inventory, "groot_n17_droid")
    admissions = {
        "pi05_droid": _runtime_admission(pi05, "1"),
        "groot_n17_droid": _runtime_admission(n17, "2"),
    }
    selection = freeze_policy_candidate_selection(
        inventory=inventory,
        selected_candidate_ids=["pi05_droid", "groot_n17_droid"],
        runtime_admissions=admissions,
        protocol_request_digest=_sha("a"),
    )

    assert selection["candidate_count"] == 2
    assert selection["selection_frozen_before_task_outcomes"] is True
    assert selection["selected_candidate_ids"] == [
        "pi05_droid",
        "groot_n17_droid",
    ]
    assert selection["selection_digest"] == canonical_digest(
        selection, digest_field="selection_digest"
    )

    with pytest.raises(
        Adp009dPolicyAdmissionError,
        match="policy_selection_exactly_two_distinct_required",
    ):
        freeze_policy_candidate_selection(
            inventory=inventory,
            selected_candidate_ids=[
                "pi05_droid",
                "groot_n17_droid",
                "cosmos3_edge_policy_droid",
            ],
            runtime_admissions=admissions,
            protocol_request_digest=_sha("a"),
        )


def test_candidate_digest_blocks_caller_relabeling() -> None:
    inventory = _inventory()
    tampered = copy.deepcopy(_candidate(inventory, "cosmos3_edge_policy_droid"))
    tampered["inventory_role"] = "third_scored_policy"
    inventory["candidates"][-1] = tampered
    inventory["inventory_digest"] = canonical_digest(
        inventory, digest_field="inventory_digest"
    )

    with pytest.raises(
        Adp009dPolicyAdmissionError,
        match="policy_cosmos3_edge_policy_droid_candidate_digest_mismatch",
    ):
        validate_policy_candidate_inventory(inventory)
