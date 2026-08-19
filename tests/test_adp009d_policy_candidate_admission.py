from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_policy_candidate_admission import (
    EXPECTED_CANDIDATES,
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


_FROZEN_INVENTORY = (
    Path(__file__).resolve().parents[1]
    / "docs/experiments/policy_ranking_thesis_20260726"
    / "openpi_polaris_checkpoint_inventory.json"
)


def test_ratified_pi05_checkpoint_is_derived_from_the_frozen_gcs_inventory() -> None:
    """Every ratified value must be readable from evidence, not asserted.

    The polaris checkpoint was ratified as the pi05 baseline. The committed
    inventory was built from the GCS JSON API and carries the real object
    generations, so the admission record is pinned to it here rather than
    restating digits that could drift into a second, silently wrong copy.
    """

    import hashlib
    import json

    from blueprint_pipeline.openpi_droid_policy_runtime import canonical_sha256

    raw = _FROZEN_INVENTORY.read_bytes()
    inventory = json.loads(raw)
    entry = next(
        row
        for row in inventory["entries"]
        if row["policy_id"] == "pi05_droid_jointpos_polaris"
    )

    # The inventory is self-consistent: its declared digest is its own content.
    recomputed = canonical_sha256(
        {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    )
    assert recomputed == inventory["inventory_sha256"]
    assert inventory["status"] == "frozen"
    assert inventory["blockers"] == []

    record = EXPECTED_CANDIDATES["pi05_droid"]
    assert record["checkpoint_repository"] == entry["checkpoint_uri"]
    assert record["checkpoint_total_bytes"] == entry["size_bytes"]
    assert record["checkpoint_inventory_digest"] == (
        "sha256:" + inventory["inventory_sha256"]
    )
    assert record["checkpoint_revision"] == (
        "gcs-generation-inventory:" + inventory["inventory_sha256"]
    )

    # The published manifest carries the same values, and its file digest is
    # the one the cohort froze.
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/arm_decision_proof_v1/manifests"
            / "adp009d_policy_candidate_inventory.v1.json"
        ).read_text(encoding="utf-8")
    )
    checkpoint = manifest["candidates"][0]["checkpoint"]
    assert checkpoint["repository"] == entry["checkpoint_uri"]
    assert checkpoint["total_bytes"] == entry["size_bytes"]
    assert checkpoint["file_count"] == entry["object_count"]
    assert checkpoint["publisher_inventory_file_sha256"] == (
        "sha256:" + hashlib.sha256(raw).hexdigest()
    )


def test_pi05_checkpoint_identity_satisfies_the_spec_that_refused_the_stock_one(
    tmp_path: Path,
) -> None:
    """The whole point of ratifying: the provisioned URI must now be describable.

    The stock checkpoint failed `checkpoint_uri_not_frozen_openpi_polaris`, so
    no execution spec could name what the lane fetched and still satisfy the
    episode client.
    """

    import json

    from blueprint_pipeline.openpi_droid_policy_runtime import (
        load_policy_spec_from_execution_spec,
    )

    record = EXPECTED_CANDIDATES["pi05_droid"]
    inventory = json.loads(_FROZEN_INVENTORY.read_text(encoding="utf-8"))
    entry = next(
        row
        for row in inventory["entries"]
        if row["policy_id"] == "pi05_droid_jointpos_polaris"
    )
    path = tmp_path / "execution_spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "native_task_arena_policy_execution_spec.v1",
                "candidate_id": "pi05_droid",
                "policy_spec": {
                    "policy_id": "pi05_droid",
                    "config_name": "pi05_droid",
                    # The URI this lane actually provisions.
                    "checkpoint_uri": record["checkpoint_repository"],
                    "checkpoint_object_manifest_sha256": entry[
                        "legacy_object_manifest_sha256"
                    ],
                    "checkpoint_generation_manifest_sha256": entry[
                        "generation_manifest_sha256"
                    ],
                    "checkpoint_inventory_sha256": inventory["inventory_sha256"],
                    "checkpoint_object_count": entry["object_count"],
                    "checkpoint_size_bytes": entry["size_bytes"],
                    "action_space": "joint_position",
                    "action_chunk_rows": 15,
                    "open_loop_horizon": 8,
                    "openpi_revision": inventory["openpi_revision"],
                },
            }
        ),
        encoding="utf-8",
    )

    spec = load_policy_spec_from_execution_spec(path)

    assert spec.checkpoint_uri == record["checkpoint_repository"]
    assert spec.checkpoint_size_bytes == record["checkpoint_total_bytes"]
    assert len(spec.server_metadata()) == 14


def test_the_pi05_candidates_openpi_revision_is_the_one_its_checkpoint_is_bound_to(
) -> None:
    """Two different pins shared a name; the candidate takes the policy one.

    `ARENA_OPENPI_REVISION` is the openpi source revision IsaacLab-Arena pins
    for its own tree, alongside arena_source and isaac_lab_source. The frozen
    inventory that binds these checkpoint bytes, the cohort that names them, the
    candidate admission record, and `OpenPIDroidPolicySpec.validate()` all name
    the other one -- so labelling the baseline with Arena's pin made the
    baseline undescribable.
    """

    import json

    from blueprint_pipeline.adp_founder_sim_protocol import (
        ARENA_OPENPI_REVISION,
        build_founder_sim_protocol,
    )
    from blueprint_pipeline.droid_policy_bridge import OPENPI_SOURCE_REVISION

    inventory = json.loads(_FROZEN_INVENTORY.read_text(encoding="utf-8"))
    assert inventory["openpi_revision"] == OPENPI_SOURCE_REVISION
    assert EXPECTED_CANDIDATES["pi05_droid"]["source_revision"] == (
        OPENPI_SOURCE_REVISION
    )
    assert ARENA_OPENPI_REVISION != OPENPI_SOURCE_REVISION

    protocol = build_founder_sim_protocol()
    baseline = next(
        row
        for row in protocol["candidates"]
        if row.get("role") == "baseline"
    )
    assert baseline["openpi_revision"] == OPENPI_SOURCE_REVISION
