"""The door re-issues a completed preparation only after the same review as a resume, and never runs it."""
import json
import time

import pytest

from blueprint_pipeline import pubsub_handoff_listener as native
from blueprint_pipeline import selected_atlas_preview_recovery as recovery
from blueprint_pipeline import website_preparation_renewal as renewal
from tests import test_selected_atlas_preview_recovery as atlas_tests
from tests.test_selected_atlas_preview_recovery import recover, write_job

site = atlas_tests.site
REISSUED = {"status": "reissued", "submission_id": "website-renewal-" + "1" * 64, "intent_id": "scene-" + "2" * 64,
            "request_digest": "sha256:" + "3" * 64, "preparation_digest": "sha256:" + "4" * 64,
            "authority_digest": "sha256:" + "5" * 64, "source_registration": {"path": "/private/registration.json"},
            "website_intake_outbox": {"id": "scene-outbox", "state": "forward_pending",
                                      "request_digest": "sha256:" + "3" * 64}}


def retained_native(site, status="completed"):
    key = "sha256:" + "d" * 64
    site.observation["producer_delivery"]["delivery_key"] = key
    ledger = {"schema_version": "v1", "status": status, "scene_id": "site-fixture",
              "capture_id": "walkthrough-fixture", "revision": 9, "attempt_count": 2,
              "source_payload_sha256": "a" * 64, "producer_delivery_key": key,
              "attempt_history": [{"attempt_number": n, "status": "completed" if n == 2 else "failed_retryable"}
                                  for n in (1, 2)],
              "lease_owner": None, "lease_token": None, "lease_expires_at": None, "recovered_expired_lease": False}
    (site.capture / native.JOB_LEDGER_FILENAME).write_text(json.dumps(ledger))
    return ledger


@pytest.fixture
def reissued(site, tmp_path, monkeypatch):
    calls, outcome = [], [REISSUED]
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(tmp_path / "intents"))
    monkeypatch.setattr(native, "process_handoff_payload",
                        lambda *a, **kw: pytest.fail("a reissue never enters the native handler"))
    monkeypatch.setattr(renewal, "reissue_renewed_preparation",
                        lambda **kw: calls.append(kw) or json.loads(json.dumps(outcome[0])))
    return calls, outcome


def reissue(site, reviewed):
    return recovery.recover_selected_atlas_preview(
        {**site.selection, "mode": "preparation-reissue", "atlas_status_digest": reviewed},
        observation=site.observation, membership=site.membership, capture_root=site.capture,
        storage_root=site.storage)


@pytest.mark.parametrize("age_seconds", [60, recovery.MAXIMUM_JOB_SECONDS + 1])
def test_a_reviewed_completed_preparation_is_handed_to_the_renewal(site, reissued, tmp_path, age_seconds):
    calls, _ = reissued
    retained_native(site)
    write_job(site, state="ready", enqueued_at_epoch=int(time.time()) - age_seconds)
    before = atlas_tests.tree_bytes(site)
    snapshot = recover(site)["atlas"]
    result = reissue(site, snapshot["atlas_status_digest"])
    assert result["status"] == "preparation_reissued" and result["blockers"] == []
    assert result["native_handler_invoked"] is False and result["atlas_retry_performed"] is False
    assert result["reissue"]["website_intake_outbox"]["id"] == "scene-outbox"
    assert "source_registration" not in result["reissue"]
    assert len(calls) == 1 and calls[0]["capture_root"] == site.capture
    assert calls[0]["queue_root"] == tmp_path / "intents"
    assert atlas_tests.tree_bytes(site) == before  # no Atlas or native state changed


@pytest.mark.parametrize("change", ["failed_retryable", "processing", "missing", "source", "owner", "not_ready",
                                    "digest"])
def test_only_a_reviewed_completed_native_job_is_reissued(site, reissued, change):
    calls, _ = reissued
    ledger = retained_native(site)
    write_job(site, state="ready")
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    if change in {"failed_retryable", "processing"}:
        ledger["status"] = change
        (site.capture / native.JOB_LEDGER_FILENAME).write_text(json.dumps(ledger))
        reviewed = recover(site)["atlas"]["atlas_status_digest"]
    elif change == "missing":
        (site.capture / native.JOB_LEDGER_FILENAME).unlink()
        reviewed = recover(site)["atlas"]["atlas_status_digest"]
    elif change == "source":
        ledger["producer_delivery_key"] = "sha256:" + "f" * 64
        (site.capture / native.JOB_LEDGER_FILENAME).write_text(json.dumps(ledger))
        reviewed = recover(site)["atlas"]["atlas_status_digest"]
    elif change == "owner":
        site.observation["producer_delivery"]["delivery_key"] = "sha256:" + "e" * 64
    elif change == "not_ready":
        write_job(site, state="processing")
        reviewed = recover(site)["atlas"]["atlas_status_digest"]
    else:
        ledger["revision"] += 1
        (site.capture / native.JOB_LEDGER_FILENAME).write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="selected_handoff_atlas_"):
        reissue(site, reviewed)
    assert not calls


def test_current_permission_refusals_still_block_a_reissue(site, reissued, monkeypatch):
    calls, _ = reissued
    retained_native(site)
    write_job(site, state="ready")

    def refuse(*_args, **_kwargs):
        raise ValueError("website_atlas_configuration:source_revoked")

    monkeypatch.setattr(recovery, "_configuration", refuse)
    snapshot = recover(site)["atlas"]
    with pytest.raises(ValueError, match="selected_handoff_atlas_preflight_blocked"):
        reissue(site, snapshot["atlas_status_digest"])
    assert not calls


def test_a_renewal_refusal_or_webapp_ceiling_is_a_typed_blocked_outcome(site, reissued, monkeypatch):
    calls, result = reissued
    retained_native(site)
    write_job(site, state="ready")
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    result[0] = {**REISSUED, "status": "enqueue_refused", "website_intake_outbox": None,
                 "webapp_refusal": {"code": "website_scene_preparation_budget_exhausted", "http_status": 409}}
    refused = reissue(site, reviewed)
    assert refused["status"] == "blocked"
    assert refused["blockers"] == ["selected_handoff_preparation_reissue_enqueue_refused"]
    assert refused["reissue"]["webapp_refusal"]["code"] == "website_scene_preparation_budget_exhausted"

    def renewal_refused(**_):
        raise ValueError("website_preparation_renewal_grant_unavailable:consent_expired")

    monkeypatch.setattr(renewal, "reissue_renewed_preparation", renewal_refused)
    blocked = reissue(site, reviewed)
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == ["website_preparation_renewal_grant_unavailable:consent_expired"]

    def unexpected(**_):
        raise ValueError("private detail /var/lib/blueprint/secret")

    monkeypatch.setattr(renewal, "reissue_renewed_preparation", unexpected)
    with pytest.raises(ValueError, match="private detail"):
        reissue(site, reviewed)  # the outer door handler redacts it as a generic failure


def test_a_reissue_needs_the_configured_intake_root(site, reissued, monkeypatch):
    calls, _ = reissued
    retained_native(site)
    write_job(site, state="ready")
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT")
    with pytest.raises(ValueError, match="selected_handoff_atlas_intake_root_missing"):
        reissue(site, reviewed)
    assert not calls
