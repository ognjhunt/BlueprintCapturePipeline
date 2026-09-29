# Covers (for impacted-test selection):
#   src/blueprint_pipeline/provider_output_promotion_records.py
#   src/blueprint_pipeline/wam_provider_object_store.py
"""Promotion receipts, the gate's deletion rule and digest-bound absence proofs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import provider_output_promotion_records as records
from blueprint_pipeline import wam_provider_object_store as object_store
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_paired_witness_staging import SUFFIX

KEYS = {"bundle": "blueprint/task/job/bundle.zip", "output": "blueprint/task/job/output.zip",
        "paired_witness": "blueprint/task/job/output.zip" + SUFFIX}


def _job(tmp_path: Path, *, gated: bool = True, witness: bool = True, retained: bool = False) -> Path:
    job = tmp_path / "staging"
    job.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": object_store.SCHEMA_VERSION, "status": "completed",
                "object_store": {"key_prefix": "blueprint/task"}, "bundle_key": KEYS["bundle"],
                "output_key": KEYS["output"], "bundle_object_retained_for_reuse": retained}
    if gated:
        manifest["output_promotion_required"] = True
    if witness:
        manifest["paired_witness"] = {"status": "ready", "witness_key": KEYS["paired_witness"]}
    (job / records.STAGING_MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    return job


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _cleanup(job: Path, roles=("bundle", "output", "paired_witness")) -> dict:
    """A completed cleanup receipt, shaped as cleanup_staged_wam_provider_objects writes it."""
    rows = [{"key_sha256": _key_hash(KEYS[role]),
             "absence": {"status": "passed", "absence_confirmed": True, "http_status_code": 404,
                         "raw_secret_values_recorded": False}} for role in roles]
    return {"schema_version": object_store.OBJECT_CLEANUP_SCHEMA_VERSION, "generated_at": "2026-09-28T00:00:00Z",
            "status": "completed", "staging_manifest_sha256": records.staging_manifest_sha256(job),
            "exact_object_count": len(rows), "objects": rows, "cleanup_attempts": 1,
            "all_objects_absent": True, "all_ephemeral_objects_absent": True,
            "content_addressed_bundle_retained_for_reuse": False, "retained_bundle_key_sha256": None,
            "signed_url_files_removed": True, "blockers": [], "raw_secret_values_recorded": False}


def _promotion(job: Path, status: str = "promoted") -> dict:
    return records.write_promotion_receipt(job, {
        "schema_version": records.RECEIPT_SCHEMA, "status": status,
        "staging_manifest_sha256": records.staging_manifest_sha256(job),
        "staged_objects": {"output": {"key_sha256": _key_hash(KEYS["output"]), "state": "promoted",
                                      "versions": [{"size_bytes": 100, "etag": '"e1"'}]}},
        "blockers": [], "private_url_recorded": False})


def test_staging_manifest_name_is_the_store_s():
    assert records.STAGING_MANIFEST_FILENAME == object_store.STAGING_MANIFEST_FILENAME


def test_receipt_loads_only_when_sealed_and_bound_to_this_manifest(tmp_path):
    job = _job(tmp_path)
    receipt = _promotion(job)
    manifest_sha256 = records.staging_manifest_sha256(job)
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert records.load_promotion_receipt(job, staging_manifest_sha256=manifest_sha256) == receipt
    assert records.load_promotion_receipt(job, staging_manifest_sha256="0" * 64) is None
    path = job / records.RECEIPT_FILENAME

    def rewritten(change, *, reseal=True):
        value = copy.deepcopy(receipt)
        change(value)
        if reseal:
            value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
        path.write_text(json.dumps(value), encoding="utf-8")
        return records.load_promotion_receipt(job, staging_manifest_sha256=manifest_sha256)

    assert rewritten(lambda value: value.update(status="failed"), reseal=False) is None
    assert rewritten(lambda value: value.update(private_url_recorded=True)) is None
    assert rewritten(lambda value: value.update(schema_version="other")) is None
    assert rewritten(lambda value: value["staged_objects"].update(bundle=value["staged_objects"]["output"])) is None
    assert rewritten(lambda value: value["staged_objects"]["output"]["versions"].append({"size_bytes": "1"})) is None
    assert rewritten(lambda value: value["staged_objects"]["output"].update(key_sha256="nothex")) is None
    path.write_text("{not json", encoding="utf-8")
    assert records.load_promotion_receipt(job, staging_manifest_sha256=manifest_sha256) is None
    path.unlink()
    (tmp_path / "elsewhere.json").write_text(json.dumps(receipt), encoding="utf-8")
    path.symlink_to(tmp_path / "elsewhere.json")
    assert records.load_promotion_receipt(job, staging_manifest_sha256=manifest_sha256) is None


def test_deletion_is_allowed_only_for_a_durable_matching_version(tmp_path):
    job = _job(tmp_path)
    receipt = _promotion(job)
    allowed = records.staged_deletion_allowed
    assert allowed(None, role="output", key=KEYS["output"], size_bytes=100, etag='"e1"') == (
        False, "staged_output_promotion_receipt_missing")
    assert allowed(receipt, role="output", key=KEYS["output"], size_bytes=100, etag="e1") == (
        True, "promoted_version_matches")
    assert allowed(receipt, role="output", key=KEYS["output"], size_bytes=100, etag='W/"e1"')[0] is True
    assert allowed(receipt, role="output", key=KEYS["output"], size_bytes=101, etag='"e1"') == (
        False, "staged_output_promotion_identity_mismatch")
    assert allowed(receipt, role="output", key=KEYS["bundle"], size_bytes=100, etag='"e1"') == (
        False, "staged_output_promotion_receipt_missing")
    assert allowed(receipt, role="paired_witness", key=KEYS["paired_witness"], size_bytes=100,
                   etag='"e1"') == (False, "staged_output_promotion_receipt_missing")
    for state in ("absent_confirmed", "deferred", "failed"):
        pending = copy.deepcopy(receipt)
        pending["staged_objects"]["output"]["state"] = state
        assert allowed(pending, role="output", key=KEYS["output"], size_bytes=100, etag='"e1"')[0] is False


def test_absence_proof_binds_the_manifest_and_every_staged_key(tmp_path):
    job = _job(tmp_path)
    promotion = _promotion(job)
    cleanup = _cleanup(job)

    proof = records.write_staged_object_absence_proof(staging_dir=job, cleanup=cleanup, promotion=promotion)

    assert records.validate_staged_object_absence_proof(proof, staging_dir=job) == proof
    assert records.load_staged_object_absence_proof(job) == proof
    assert proof["schema_version"] == "wam_provider_staged_object_absence_proof.v1"
    assert proof["status"] == "all_staged_objects_absent" and proof["all_staged_objects_absent"] is True
    assert proof["staging_manifest_sha256"] == records.staging_manifest_sha256(job)
    assert proof["objects"] == sorted(({"role": role, "key_sha256": _key_hash(KEYS[role])}
                                       for role in KEYS), key=lambda row: row["key_sha256"])
    assert proof["exact_object_count"] == 3 and proof["output_promotion_required"] is True
    # The cleanup it proves is embedded, as the continuation seals its own copy.
    assert proof["cleanup"] == cleanup and proof["cleanup_digest"] == canonical_digest(cleanup)
    assert (proof["promotion_receipt_digest"], proof["promotion_status"]) == (
        promotion["receipt_digest"], "promoted")
    assert proof["proof_digest"] == canonical_digest(proof, digest_field="proof_digest")
    assert proof["private_url_recorded"] is False and "https:" not in json.dumps(proof)
    # Written once: a later cleanup never replaces the first proof.
    later = _cleanup(job) | {"generated_at": "2026-09-29T00:00:00Z"}
    assert records.write_staged_object_absence_proof(staging_dir=job, cleanup=later, promotion=promotion) == proof

    # An ungated manifest with a retained bundle proves its own exact set.
    plain = _job(tmp_path / "plain", gated=False, witness=False, retained=True)
    plain_proof = records.build_staged_object_absence_proof(
        staging_dir=plain, cleanup=_cleanup(plain, roles=("output",)), promotion=None)
    assert plain_proof["objects"] == [{"role": "output", "key_sha256": _key_hash(KEYS["output"])}]
    assert plain_proof["promotion_receipt_digest"] is None
    assert records.validate_staged_object_absence_proof(plain_proof, staging_dir=plain) == plain_proof


def _resealed(proof, change):
    value = copy.deepcopy(proof)
    change(value)
    value["proof_digest"] = canonical_digest(value, digest_field="proof_digest")
    return value


def test_absence_proof_validator_refuses_tampering_and_a_changed_manifest(tmp_path):
    job = _job(tmp_path)
    proof = records.build_staged_object_absence_proof(staging_dir=job, cleanup=_cleanup(job),
                                                      promotion=_promotion(job))

    def row_not_absent(value):
        value["cleanup"]["objects"][1]["absence"]["absence_confirmed"] = False
        value["cleanup_digest"] = canonical_digest(value["cleanup"])

    for tampered, code in (
        ({**proof, "status": "blocked"}, "staged_object_absence_proof_digest_mismatch"),
        (_resealed(proof, lambda value: value.update(status="blocked")), "staged_object_absence_proof_invalid"),
        (_resealed(proof, lambda value: value.update(private_url_recorded=True)),
         "staged_object_absence_proof_invalid"),
        (_resealed(proof, lambda value: value["objects"].pop()), "staged_object_absence_proof_keys_mismatch"),
        (_resealed(proof, lambda value: value.update(exact_object_count=2)),
         "staged_object_absence_proof_keys_mismatch"),
        (_resealed(proof, row_not_absent), "staged_object_absence_proof_cleanup_invalid"),
        (_resealed(proof, lambda value: value.update(cleanup_digest="sha256:" + "0" * 64)),
         "staged_object_absence_proof_cleanup_invalid"),
        (_resealed(proof, lambda value: value.update(promotion_receipt_digest=None)),
         "staged_object_absence_proof_promotion_missing"),
    ):
        with pytest.raises(records.ProviderOutputPromotionRecordError, match=f"^{code}$"):
            records.validate_staged_object_absence_proof(tampered, staging_dir=job)
    # The manifest it binds may not change afterwards.
    manifest = job / records.STAGING_MANIFEST_FILENAME
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(records.ProviderOutputPromotionRecordError,
                       match="^staged_object_absence_proof_manifest_changed$"):
        records.validate_staged_object_absence_proof(proof, staging_dir=job)


@pytest.mark.parametrize("change", ["blocked", "object_left", "missing_row", "other_manifest"])
def test_no_proof_is_built_from_a_cleanup_that_did_not_prove_absence(tmp_path, change):
    job = _job(tmp_path)
    cleanup = _cleanup(job)
    if change == "blocked":
        cleanup.update(status="blocked", blockers=["staged_output_promotion_receipt_missing"],
                       all_objects_absent=False)
    elif change == "object_left":
        cleanup["objects"][1]["absence"] = {"status": "deferred", "absence_confirmed": False}
    elif change == "missing_row":
        cleanup["objects"].pop()
    else:
        cleanup["staging_manifest_sha256"] = "0" * 64
    with pytest.raises(records.ProviderOutputPromotionRecordError,
                       match="^staged_object_absence_not_proven$"):
        records.build_staged_object_absence_proof(staging_dir=job, cleanup=cleanup, promotion=_promotion(job))
    assert not (job / records.ABSENCE_PROOF_FILENAME).exists()
    assert records.load_staged_object_absence_proof(job) is None


def test_a_corrupt_proof_file_is_refused_rather_than_replaced(tmp_path):
    job = _job(tmp_path)
    (job / records.ABSENCE_PROOF_FILENAME).write_text('{"schema_version": "x"}', encoding="utf-8")
    with pytest.raises(records.ProviderOutputPromotionRecordError):
        records.write_staged_object_absence_proof(staging_dir=job, cleanup=_cleanup(job),
                                                  promotion=_promotion(job))
    assert json.loads((job / records.ABSENCE_PROOF_FILENAME).read_text()) == {"schema_version": "x"}
