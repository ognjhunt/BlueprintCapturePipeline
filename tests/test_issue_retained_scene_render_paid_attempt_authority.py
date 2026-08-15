"""A cap that forgets a prior attempt's spend is not a cap.

There is no automatic retry for this probe, so the attempt authority is
reissued for every run and each re-run must carry its predecessors. Doing that
by hand once per loop iteration is how the aggregate cap silently stops binding.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_retained_scene_render_vast import RESULT_SCHEMA
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "issue_retained_scene_render_paid_attempt_authority",
    REPO_ROOT / "scripts" / "issue_retained_scene_render_paid_attempt_authority.py",
)
issuer = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(issuer)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _job(tmp_path: Path, *, cap: float = 12.0, allowlist=(47373597, 47569249)) -> Path:
    job = tmp_path / "job"
    runtime = job / "provider_runtime"
    runtime.mkdir(parents=True)
    archive = job / "adp_retained_scene_gpu_render_bundle.zip"
    archive.write_bytes(b"bundle-archive-bytes")
    parent = {
        "schema_version": "third_scene_dual_task_execution_authority.v1",
        "paid_compute": {
            "provider": "vast",
            "external_instance_allowlist": list(allowlist),
        },
    }
    parent["authority_digest"] = canonical_digest(parent, digest_field="authority_digest")
    authority = runtime / "execution_authority.json"
    authority.write_text(json.dumps(parent), encoding="utf-8")
    receipt = {
        "schema_version": "adp009d_retained_scene_gpu_render_bundle.v1",
        "status": "ready",
        "blueprint_commit": "b" * 40,
        "hard_total_spend_cap_usd": cap,
        "bundle_path": "/Users/author/job/bundle.zip",
        "bundle_relative_path": "adp_retained_scene_gpu_render_bundle.zip",
        "bundle_sha256": _digest(archive.read_bytes()),
        "execution_authority": {
            "path": "/private/tmp/checkout/execution_authority.json",
            "relative_path": "provider_runtime/execution_authority.json",
            "sha256": _digest(authority.read_bytes()),
            "authority_digest": parent["authority_digest"],
        },
    }
    (job / "adp_retained_scene_gpu_render_bundle_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return job


def _receipt(job: Path) -> Path:
    return job / "adp_retained_scene_gpu_render_bundle_receipt.json"


def _prior_result(path: Path, *, cost: float) -> Path:
    authority_digest = "sha256:" + "a" * 64
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "blocked",
        "estimated_cost_usd": cost,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "bundle_sha256": "sha256:" + "b" * 64,
        "authorization_consumption": {"authorization_digest": authority_digest},
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def _record(path: Path, *, receipt_digest: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path.read_bytes()),
    }
    if receipt_digest is not None:
        value["receipt_digest"] = receipt_digest
    return value


def _prior_reconciliation(
    tmp_path: Path, result_path: Path, *, actual_cost: float
) -> Path:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    instance_id = 123
    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "status": "completed",
        "continuing_spend_from_this_run": False,
        "runner_gpu_teardown_completed": True,
        "vast_instance_ids": [instance_id],
    }
    teardown_path = tmp_path / "teardown.json"
    teardown_path.write_text(json.dumps(teardown), encoding="utf-8")
    zero = {
        "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
        "status": "provider_zero_confirmed",
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
    }
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    zero_path = tmp_path / "provider-zero.json"
    zero_path.write_text(json.dumps(zero), encoding="utf-8")
    billing = {
        "schema_version": "vast.official.charges.v1",
        "results": [
            {
                "source": f"instance-{instance_id}",
                "description": f"Instance {instance_id} Charges - 1 day",
                "amount": actual_cost,
                "metadata": {"label": "blueprint-retained-fixture"},
            }
        ],
    }
    billing_path = tmp_path / "official-billing.json"
    billing_path.write_text(json.dumps(billing), encoding="utf-8")
    billing_source = {
        "schema_version": "blueprint.provider_billing_source_receipt.v1",
        "status": "reconciled",
        "sources": [
            {
                "provider": "vast",
                "retained_path": str(billing_path.resolve()),
                "response_digest": _digest(billing_path.read_bytes()),
                "response_size_bytes": billing_path.stat().st_size,
            }
        ],
    }
    billing_source["receipt_digest"] = canonical_digest(
        billing_source, digest_field="receipt_digest"
    )
    billing_source_path = tmp_path / "billing-source.json"
    billing_source_path.write_text(json.dumps(billing_source), encoding="utf-8")
    source_receipts = [
        {
            "role": "terminal_result",
            "schema_version": result["schema_version"],
            "digest_field": "receipt_digest",
            "record": _record(result_path, receipt_digest=result["receipt_digest"]),
        },
        {
            "role": "teardown_manifest",
            "schema_version": teardown["schema_version"],
            "digest_field": None,
            "legacy_digest_gap": "exact_source_bytes_sha256_bound_no_canonical_digest",
            "record": _record(teardown_path),
        },
        {
            "role": "provider_zero",
            "schema_version": zero["schema_version"],
            "digest_field": "receipt_digest",
            "record": _record(zero_path, receipt_digest=zero["receipt_digest"]),
        },
        {
            "role": "official_billing_response",
            "schema_version": billing["schema_version"],
            "digest_field": None,
            "legacy_digest_gap": "exact_source_bytes_sha256_bound_no_canonical_digest",
            "record": _record(billing_path),
        },
        {
            "role": "provider_billing_source_receipt",
            "schema_version": billing_source["schema_version"],
            "digest_field": "receipt_digest",
            "record": _record(
                billing_source_path, receipt_digest=billing_source["receipt_digest"]
            ),
        },
    ]
    entry = {
        "schema_version": "adp_same_goal_spend_entry.v1",
        "goal_id": "arm-decision-proof-v1",
        "attempt_id": "retained-scene-prior-1",
        "lane": "retained_scene_render",
        "evidence_kind": "fully_bound_official_billing",
        "provider_instance_id": instance_id,
        "cost_usd": actual_cost,
        "authority_digest": result["authorization_consumption"]["authorization_digest"],
        "bundle_sha256": result["bundle_sha256"],
        "continuing_spend_from_this_run": False,
        "provider_zero_confirmed": True,
        "source_receipts": source_receipts,
        "bindings": [
            {
                "kind": "cost_usd",
                "source_role": "official_billing_response",
                "json_path": ["results", 0, "amount"],
                "expected_value": actual_cost,
            },
            {
                "kind": "continuing_spend",
                "source_role": "terminal_result",
                "json_path": ["continuing_spend_from_this_run"],
                "expected_value": False,
            },
            {
                "kind": "instance_id",
                "source_role": "official_billing_response",
                "json_path": ["results", 0, "source"],
                "expected_value": f"instance-{instance_id}",
            },
            {
                "kind": "authority_digest",
                "source_role": "terminal_result",
                "json_path": ["authorization_consumption", "authorization_digest"],
                "expected_value": result["authorization_consumption"]["authorization_digest"],
            },
            {
                "kind": "provider_zero",
                "source_role": "provider_zero",
                "json_path": ["provider_zero_verified"],
                "expected_value": True,
            },
            {
                "kind": "bundle_sha256",
                "source_role": "terminal_result",
                "json_path": ["bundle_sha256"],
                "expected_value": result["bundle_sha256"],
            },
        ],
        "entry_digest": "",
    }
    entry["entry_digest"] = canonical_digest(entry, digest_field="entry_digest")
    reconciliation = {
        "schema_version": "adp_same_goal_spend_reconciliation.v1",
        "status": "all_same_goal_paid_attempts_terminal_and_provider_zero",
        "goal_id": "arm-decision-proof-v1",
        "entries": [entry],
        "entry_count": 1,
        "total_cost_usd": actual_cost,
        "receipt_digest": "",
    }
    reconciliation["receipt_digest"] = canonical_digest(
        reconciliation, digest_field="receipt_digest"
    )
    path = tmp_path / "prior-spend-reconciliation.json"
    path.write_text(json.dumps(reconciliation), encoding="utf-8")
    return path


def test_an_attempt_authority_derives_its_bindings_from_the_bundle(tmp_path):
    job = _job(tmp_path)

    authority = issuer.issue_paid_attempt_authority(
        bundle_receipt_path=_receipt(job),
        authorized_by="operator",
        max_hourly_rate_usd=2.0,
        hard_ttl_seconds=10_800,
        authorized_on="2026-08-13",
    )

    receipt = json.loads(_receipt(job).read_text(encoding="utf-8"))
    assert authority["bundle_sha256"] == receipt["bundle_sha256"]
    assert authority["blueprint_commit"] == receipt["blueprint_commit"]
    assert authority["hard_attempt_spend_cap_usd"] == 12.0
    assert authority["maximum_paid_attempts"] == 1
    assert authority["automatic_paid_retry_authorized"] is False
    # Taken from the parent authority, never from this script's arguments.
    assert authority["external_active_instance_allowlist"] == [47373597, 47569249]
    assert authority["prior_terminal_attempts"] == []


def test_a_reissue_carries_prior_spend_against_the_same_cap(tmp_path):
    job = _job(tmp_path)
    first = _prior_result(tmp_path / "result-one.json", cost=0.12)
    reconciliation = _prior_reconciliation(tmp_path, first, actual_cost=0.25)

    authority = issuer.issue_paid_attempt_authority(
        bundle_receipt_path=_receipt(job),
        authorized_by="operator",
        max_hourly_rate_usd=2.0,
        hard_ttl_seconds=3_600,
        prior_result_paths=[first],
        prior_spend_reconciliation_path=reconciliation,
    )

    assert authority["manual_reissue_after_prior_terminal_attempt"] is True
    assert len(authority["prior_terminal_attempts"]) == 1
    row = authority["prior_terminal_attempts"][0]
    assert row["estimated_cost_usd"] == 0.12
    assert row["actual_provider_charge_usd"] == 0.25
    assert row["result_path"] == str(first.resolve())
    assert row["result_sha256"] == _digest(first.read_bytes())
    assert row["result"]["sha256"] == _digest(first.read_bytes())
    assert authority["prior_actual_provider_spend_usd"] == 0.25


def test_prior_spend_plus_this_attempt_cannot_exceed_the_cap(tmp_path):
    """The aggregate cap is what stops a loop of re-runs from spending past an
    approval, and it only binds if every prior attempt is carried."""

    job = _job(tmp_path, cap=1.0)
    spent = _prior_result(tmp_path / "result-one.json", cost=0.9)
    reconciliation = _prior_reconciliation(tmp_path, spent, actual_cost=0.9)

    with pytest.raises(ValueError, match="aggregate_spend_cap_exceeded"):
        issuer.issue_paid_attempt_authority(
            bundle_receipt_path=_receipt(job),
            authorized_by="operator",
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=3_600,
            prior_result_paths=[spent],
            prior_spend_reconciliation_path=reconciliation,
        )


def test_a_reissue_rejects_estimate_substitution_or_missing_reconciliation(tmp_path):
    job = _job(tmp_path)
    prior = _prior_result(tmp_path / "result-one.json", cost=0.12)

    with pytest.raises(
        issuer.AttemptAuthorityError, match="prior_spend_reconciliation_required"
    ):
        issuer.issue_paid_attempt_authority(
            bundle_receipt_path=_receipt(job),
            authorized_by="operator",
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=3_600,
            prior_result_paths=[prior],
        )

    reconciliation = _prior_reconciliation(tmp_path, prior, actual_cost=0.25)
    value = json.loads(reconciliation.read_text(encoding="utf-8"))
    value["entries"][0]["cost_usd"] = 0.12
    value["entries"][0]["entry_digest"] = canonical_digest(
        value["entries"][0], digest_field="entry_digest"
    )
    value["total_cost_usd"] = 0.12
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    reconciliation.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(issuer.AttemptAuthorityError, match="same_goal_spend_binding_invalid"):
        issuer.issue_paid_attempt_authority(
            bundle_receipt_path=_receipt(job),
            authorized_by="operator",
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=3_600,
            prior_result_paths=[prior],
            prior_spend_reconciliation_path=reconciliation,
        )


def test_a_reissue_rejects_noncanonical_reconciliation_evidence_kind(tmp_path):
    job = _job(tmp_path)
    prior = _prior_result(tmp_path / "result-one.json", cost=0.12)
    reconciliation = _prior_reconciliation(tmp_path, prior, actual_cost=0.25)
    value = json.loads(reconciliation.read_text(encoding="utf-8"))
    value["entries"][0]["evidence_kind"] = "fully_bound"
    value["entries"][0]["entry_digest"] = canonical_digest(
        value["entries"][0], digest_field="entry_digest"
    )
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    reconciliation.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(issuer.AttemptAuthorityError, match="same_goal_spend_entry_invalid"):
        issuer.issue_paid_attempt_authority(
            bundle_receipt_path=_receipt(job),
            authorized_by="operator",
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=3_600,
            prior_result_paths=[prior],
            prior_spend_reconciliation_path=reconciliation,
        )


def test_a_reissue_requires_teardown_to_name_the_billed_instance(tmp_path):
    job = _job(tmp_path)
    prior = _prior_result(tmp_path / "result-one.json", cost=0.12)
    reconciliation = _prior_reconciliation(tmp_path, prior, actual_cost=0.25)
    value = json.loads(reconciliation.read_text(encoding="utf-8"))
    entry = value["entries"][0]
    teardown_source = next(
        row
        for row in entry["source_receipts"]
        if row["role"] == "teardown_manifest"
    )
    teardown_path = Path(teardown_source["record"]["path"])
    teardown = json.loads(teardown_path.read_text(encoding="utf-8"))
    teardown.pop("vast_instance_ids")
    teardown_path.write_text(json.dumps(teardown), encoding="utf-8")
    teardown_source["record"] = _record(teardown_path)
    entry["entry_digest"] = canonical_digest(entry, digest_field="entry_digest")
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    reconciliation.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        issuer.AttemptAuthorityError,
        match="prior_terminal_billing_or_zero_invalid",
    ):
        issuer.issue_paid_attempt_authority(
            bundle_receipt_path=_receipt(job),
            authorized_by="operator",
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=3_600,
            prior_result_paths=[prior],
            prior_spend_reconciliation_path=reconciliation,
        )


def test_a_bundle_this_host_cannot_resolve_authorizes_nothing(tmp_path):
    job = _job(tmp_path)
    (job / "adp_retained_scene_gpu_render_bundle.zip").unlink()

    with pytest.raises(
        issuer.AttemptAuthorityError, match="attempt_authority_bundle_not_host_resident"
    ):
        issuer.issue_paid_attempt_authority(
            bundle_receipt_path=_receipt(job),
            authorized_by="operator",
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=3_600,
        )


def test_an_unreadable_prior_result_is_refused_rather_than_skipped(tmp_path):
    job = _job(tmp_path)

    with pytest.raises(issuer.AttemptAuthorityError, match="prior_terminal_attempt_missing"):
        issuer.issue_paid_attempt_authority(
            bundle_receipt_path=_receipt(job),
            authorized_by="operator",
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=3_600,
            prior_result_paths=[tmp_path / "never-written.json"],
        )


def test_an_unattributed_authorization_is_refused(tmp_path):
    job = _job(tmp_path)

    with pytest.raises(issuer.AttemptAuthorityError, match="authorized_by_required"):
        issuer.issue_paid_attempt_authority(
            bundle_receipt_path=_receipt(job),
            authorized_by="   ",
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=3_600,
        )


def test_the_issued_authority_is_the_one_the_allocator_will_accept(tmp_path):
    """Validated with the allocator's own function before it is written, so a
    document that would be refused at the paid boundary never exists."""

    from blueprint_pipeline.adp_retained_scene_render_vast import (
        validate_retained_scene_render_paid_attempt_authority,
    )

    job = _job(tmp_path)
    authority = issuer.issue_paid_attempt_authority(
        bundle_receipt_path=_receipt(job),
        authorized_by="operator",
        max_hourly_rate_usd=2.0,
        hard_ttl_seconds=10_800,
    )
    receipt = json.loads(_receipt(job).read_text(encoding="utf-8"))

    assert (
        validate_retained_scene_render_paid_attempt_authority(
            authority,
            prepared_bundle=receipt,
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=10_800,
            allowed_active_instance_ids=[47373597, 47569249],
        )
        == authority
    )
