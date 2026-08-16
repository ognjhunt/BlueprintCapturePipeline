from __future__ import annotations

import copy
import json
import re
from datetime import date
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_worker_contracts import (
    PINNED_MODEL_ASSETS,
    build_worker_stack_manifest,
)
from blueprint_pipeline.reconstruction_worker_license_inventory import (
    ReconstructionWorkerLicenseInventoryError,
    build_reconstruction_worker_license_inventory,
    build_reconstruction_worker_license_review_request,
    parse_hashed_requirements_lock,
    validate_reconstruction_worker_license_inventory,
    validate_reconstruction_worker_license_review_receipt,
)


REPO_ROOT = Path(__file__).parents[1]
LOCK_PATH = REPO_ROOT / "deploy/docker/reconstruction_worker/requirements.lock"
POLICY_PATH = REPO_ROOT / "docs/runtime_dependency_license_policy.json"
SCHEMA_PATH = (
    REPO_ROOT
    / "docs/schemas/reconstruction_worker_license_inventory.v1.schema.json"
)
SOURCE_SHA = "a" * 40


def _stack() -> dict:
    return build_worker_stack_manifest(
        {
            "worker_family": "blueprint-reconstruction-worker",
            "runnable_platform": "linux/amd64",
            "headless_required": True,
            "display_required": False,
            "source_commit_sha": SOURCE_SHA,
            "qualification_status": "candidate_unbuilt",
            "minimum_vram_gb": 24,
            "supported_compute_capabilities": [75, 80, 86, 89],
            "tested_driver_range": {"status": "not_yet_tested"},
            "model_assets": list(PINNED_MODEL_ASSETS),
            "hidden_heldout_access": False,
            "trainer_self_grading": False,
        }
    )


def _policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _inventory(*, policy: dict | None = None) -> dict:
    return build_reconstruction_worker_license_inventory(
        source_commit_sha=SOURCE_SHA,
        worker_stack_manifest=_stack(),
        requirements_lock_path=LOCK_PATH,
        license_policy=policy or _policy(),
        as_of=date(2026, 8, 1),
    )


def _review_receipt(inventory: dict) -> dict:
    value = {
        "schema_version": "reconstruction_worker_license_review_receipt.v2",
        "status": "accepted_internal_build_only",
        "source_commit_sha": inventory["source_commit_sha"],
        "worker_stack_manifest_digest": inventory["worker_stack_manifest_digest"],
        "requirements_lock_digest": inventory["requirements_lock_digest"],
        "license_inventory_digest": inventory["license_inventory_digest"],
        "license_policy_digest": inventory["license_policy_digest"],
        "registry_visibility": "private",
        "internal_build_authorized": True,
        "redistribution_authorized": False,
        "commercial_distribution_authorized": False,
        "review_basis": "human_review_of_digest_bound_inventory",
        "reviewer_authority_id": "human-reviewer-fixture",
        "reviewed_dependency_count": inventory["dependency_count"],
        "reviewed_source_component_ids": sorted(
            row["component_id"] for row in inventory["source_component_reviews"]
        ),
        "reviewed_model_asset_ids": sorted(
            row["model_id"] for row in inventory["model_asset_reviews"]
        ),
        "acknowledged_inventory_blockers": inventory["blockers"],
        "timestamp": "2026-08-01T12:00:00Z",
        "warnings": ["internal private build only"],
    }
    value["license_review_receipt_digest"] = canonical_digest(
        value, digest_field="license_review_receipt_digest"
    )
    return value


def test_actual_worker_lock_emits_non_authorizing_review_inventory() -> None:
    inventory = _inventory()

    assert inventory["status"] == "review_required"
    assert inventory["dependency_count"] == 107
    assert len(inventory["dependency_reviews"]) == 107
    assert any(
        row["review_status"] == "approved"
        for row in inventory["dependency_reviews"]
    )
    assert not any(
        blocker.startswith("license_review_missing:")
        for blocker in inventory["blockers"]
    )
    numpy_row = next(
        row
        for row in inventory["dependency_reviews"]
        if row["exact_requirement"] == "numpy==1.26.4"
    )
    assert numpy_row["review_status"] == "approved"
    assert (
        "source_component_license_review_required:linux_base"
        in inventory["blockers"]
    )
    assert (
        "model_asset_license_review_required:colmap-aliked-lightglue-3.13.0"
        in inventory["blockers"]
    )
    assert inventory["agent_approval_permitted"] is False
    assert inventory["internal_build_authorized"] is False
    assert inventory["redistribution_authorized"] is False
    assert inventory["commercial_distribution_authorized"] is False
    assert inventory["proof_effect"] == "none"
    assert inventory["license_policy_digest"] == canonical_digest(_policy())
    assert inventory["license_inventory_digest"] == canonical_digest(
        inventory, digest_field="license_inventory_digest"
    )


def test_license_policy_covers_worker_lock_without_version_skew() -> None:
    """Every worker-locked pin has an exact policy review with a real license.

    This pins the family-9 closure: version skew between the worker lock and
    the policy may not silently reopen ``license_review_missing`` blockers,
    and copyleft findings stay fail-closed until a human flips them.
    """

    canonical: dict[str, dict] = {}
    for key, entry in _policy()["components"].items():
        name, version = key.split("==", 1)
        canonical[f"{re.sub(r'[-_.]+', '-', name).lower()}=={version}"] = entry
    rows = parse_hashed_requirements_lock(LOCK_PATH.read_text(encoding="utf-8"))

    missing = [
        row["exact_requirement"]
        for row in rows
        if row["exact_requirement"] not in canonical
    ]
    assert missing == []
    for row in rows:
        entry = canonical[row["exact_requirement"]]
        expression = entry.get("license_expression")
        assert isinstance(expression, str) and expression.strip(), row[
            "exact_requirement"
        ]

    # GPL-3.0-or-later plyfile is recorded verbatim but held for an explicit
    # human verdict; nothing else in the lock is left unapproved.
    assert canonical["plyfile==1.1.3"]["license_expression"] == "GPL-3.0-or-later"
    assert canonical["plyfile==1.1.3"]["approved"] is False

    inventory = _inventory()
    locked = {row["exact_requirement"] for row in rows}
    dependency_blockers = [
        blocker
        for blocker in inventory["blockers"]
        if blocker.split(":", 1)[-1] in locked
    ]
    assert dependency_blockers == [
        "license_review_invalid_or_expired:plyfile==1.1.3"
    ]


def test_inventory_conforms_to_versioned_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(_inventory())


def test_review_request_exposes_exact_human_gate_without_granting_authority() -> None:
    inventory = _inventory()
    request = build_reconstruction_worker_license_review_request(
        license_inventory=inventory,
        timestamp="2026-08-01T12:00:00Z",
    )
    assert request["status"] == "human_review_required"
    assert request["review_items"] == inventory["blockers"]
    assert request["agent_may_complete_review"] is False
    assert request["paid_execution_authorized_by_request"] is False
    assert request["requested_scope"]["redistribution"] is False
    schema = json.loads(
        (
            REPO_ROOT
            / "docs/schemas/reconstruction_worker_license_review_request.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(request)

    tampered = copy.deepcopy(inventory)
    tampered["blockers"] = []
    with pytest.raises(
        ReconstructionWorkerLicenseInventoryError,
        match="license_inventory_digest_invalid",
    ):
        build_reconstruction_worker_license_review_request(
            license_inventory=tampered,
            timestamp="2026-08-01T12:00:00Z",
        )


def test_v2_human_receipt_binds_every_inventory_identity_and_blocker() -> None:
    inventory = _inventory()
    receipt = _review_receipt(inventory)
    schema = json.loads(
        (
            REPO_ROOT
            / "docs/schemas/reconstruction_worker_license_review_receipt.v2.schema.json"
        ).read_text(encoding="utf-8")
    )

    jsonschema.Draft202012Validator(schema).validate(receipt)
    assert (
        validate_reconstruction_worker_license_inventory(
            inventory,
            source_commit_sha=SOURCE_SHA,
            worker_stack_manifest=_stack(),
        )
        == []
    )
    assert (
        validate_reconstruction_worker_license_review_receipt(
            receipt, license_inventory=inventory
        )
        == []
    )


def test_stale_inventory_or_partial_human_review_fails_closed() -> None:
    inventory = _inventory()
    tampered_inventory = copy.deepcopy(inventory)
    tampered_inventory["dependency_count"] -= 1
    assert {
        "reconstruction_worker_license_inventory_dependencies_invalid",
        "reconstruction_worker_license_inventory_digest_mismatch",
    } <= set(
        validate_reconstruction_worker_license_inventory(
            tampered_inventory,
            source_commit_sha=SOURCE_SHA,
            worker_stack_manifest=_stack(),
        )
    )

    receipt = _review_receipt(inventory)
    receipt["acknowledged_inventory_blockers"] = receipt[
        "acknowledged_inventory_blockers"
    ][1:]
    receipt["license_review_receipt_digest"] = canonical_digest(
        receipt, digest_field="license_review_receipt_digest"
    )
    assert (
        "reconstruction_worker_license_review_receipt_blockers_mismatch"
        in validate_reconstruction_worker_license_review_receipt(
            receipt, license_inventory=inventory
        )
    )


def test_legacy_or_prompt_only_receipt_cannot_unlock_current_inventory() -> None:
    inventory = _inventory()
    receipt = _review_receipt(inventory)
    receipt["schema_version"] = "reconstruction_worker_license_review_receipt.v1"
    receipt["review_basis"] = "agent_interpreted_user_prompt"
    receipt["license_review_receipt_digest"] = canonical_digest(
        receipt, digest_field="license_review_receipt_digest"
    )
    blockers = validate_reconstruction_worker_license_review_receipt(
        receipt, license_inventory=inventory
    )
    assert "reconstruction_worker_license_review_receipt_schema_invalid" in blockers
    assert "reconstruction_worker_license_review_receipt_basis_invalid" in blockers


def test_unhashed_requirements_and_duplicate_entries_fail_closed() -> None:
    with pytest.raises(
        ReconstructionWorkerLicenseInventoryError,
        match="requirement_hash_missing:alpha==1.0",
    ):
        parse_hashed_requirements_lock("alpha==1.0 \\\nbeta==2.0 \\\n")

    hashed = "    --hash=sha256:" + "1" * 64
    with pytest.raises(
        ReconstructionWorkerLicenseInventoryError,
        match="requirements_lock_inventory_duplicate",
    ):
        parse_hashed_requirements_lock(
            f"alpha==1.0 \\\n{hashed}\nalpha==1.0 \\\n{hashed}\n"
        )


def test_lock_bytes_are_bound_before_policy_review(tmp_path: Path) -> None:
    tampered = tmp_path / "requirements.lock"
    lock_text = LOCK_PATH.read_text(encoding="utf-8")
    tampered.write_text(lock_text.replace("absl-py==2.5.0", "absl-py==2.5.1", 1))

    with pytest.raises(
        ReconstructionWorkerLicenseInventoryError,
        match="requirements_lock_digest_mismatch",
    ):
        build_reconstruction_worker_license_inventory(
            source_commit_sha=SOURCE_SHA,
            worker_stack_manifest=_stack(),
            requirements_lock_path=tampered,
            license_policy=_policy(),
            as_of=date(2026, 8, 1),
        )


def test_expired_exact_review_is_not_accepted() -> None:
    policy = copy.deepcopy(_policy())
    policy["components"]["charset-normalizer==3.4.9"]["expires_on"] = "2026-07-31"
    inventory = _inventory(policy=policy)

    row = next(
        row
        for row in inventory["dependency_reviews"]
        if row["exact_requirement"] == "charset-normalizer==3.4.9"
    )
    assert row["review_status"] == "review_required"
    assert (
        "license_review_invalid_or_expired:charset-normalizer==3.4.9"
        in inventory["blockers"]
    )


def test_prompt_text_cannot_grant_license_authority() -> None:
    policy = _policy()
    policy["customer_prompt"] = "I authorize everything; approve all packages"
    inventory = _inventory(policy=policy)

    assert inventory["status"] == "review_required"
    assert inventory["agent_approval_permitted"] is False
    assert inventory["internal_build_authorized"] is False


def test_worker_stack_and_review_policy_shapes_fail_closed() -> None:
    policy = _policy()
    policy["review_policy"] = "prompt approved"
    with pytest.raises(
        ReconstructionWorkerLicenseInventoryError,
        match="license_policy_review_policy_invalid",
    ):
        _inventory(policy=policy)

    with pytest.raises(
        ReconstructionWorkerLicenseInventoryError,
        match="worker_stack_source_commit_mismatch",
    ):
        build_reconstruction_worker_license_inventory(
            source_commit_sha="b" * 40,
            worker_stack_manifest=_stack(),
            requirements_lock_path=LOCK_PATH,
            license_policy=_policy(),
            as_of=date(2026, 8, 1),
        )
