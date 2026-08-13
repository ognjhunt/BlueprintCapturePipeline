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
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "blocked",
        "estimated_cost_usd": cost,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    path.write_text(json.dumps(result), encoding="utf-8")
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

    authority = issuer.issue_paid_attempt_authority(
        bundle_receipt_path=_receipt(job),
        authorized_by="operator",
        max_hourly_rate_usd=2.0,
        hard_ttl_seconds=3_600,
        prior_result_paths=[first],
    )

    assert authority["manual_reissue_after_prior_terminal_attempt"] is True
    assert len(authority["prior_terminal_attempts"]) == 1
    row = authority["prior_terminal_attempts"][0]
    assert row["estimated_cost_usd"] == 0.12
    assert row["result_sha256"] == _digest(first.read_bytes())


def test_prior_spend_plus_this_attempt_cannot_exceed_the_cap(tmp_path):
    """The aggregate cap is what stops a loop of re-runs from spending past an
    approval, and it only binds if every prior attempt is carried."""

    job = _job(tmp_path, cap=1.0)
    spent = _prior_result(tmp_path / "result-one.json", cost=0.9)

    with pytest.raises(ValueError, match="aggregate_spend_cap_exceeded"):
        issuer.issue_paid_attempt_authority(
            bundle_receipt_path=_receipt(job),
            authorized_by="operator",
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=3_600,
            prior_result_paths=[spent],
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
