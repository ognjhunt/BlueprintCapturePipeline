"""Resume retained website preparation without resetting delivery or spending history."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import pubsub_handoff_listener as native
from blueprint_pipeline import selected_atlas_preview_recovery as recovery
from blueprint_pipeline import website_reconstruction_resume as continuation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests import test_selected_atlas_preview_recovery as atlas_tests
from tests.test_selected_atlas_preview_recovery import recover, write_job

site = atlas_tests.site


def review(digest):
    return {"review_schema": continuation.REVIEW_SCHEMA, "ledger_digest": digest,
            "selected_payload_sha256": "b" * 64}


def retained_native(site):
    key = "sha256:" + "d" * 64
    site.observation["producer_delivery"]["delivery_key"] = key
    ledger = {"schema_version": "v1", "status": "failed_retryable", "scene_id": "site-fixture",
              "capture_id": "walkthrough-fixture", "revision": 12, "attempt_count": 5,
              "source_payload_sha256": "a" * 64, "producer_delivery_key": key,
              "last_error": "clean_plate: task_target_requires_clarification:cardboard_box_candidate",
              "attempt_history": [{"attempt_number": n, "status": "failed_retryable",
                                   "error": "retained original failure"} for n in range(1, 6)],
              "lease_owner": None, "lease_token": None, "lease_expires_at": None,
              "recovered_expired_lease": False}
    (site.capture / native.JOB_LEDGER_FILENAME).write_text(json.dumps(ledger))
    return ledger


def resume(site, reviewed, process):
    return recovery.recover_selected_atlas_preview(
        {**site.selection, "mode": "preparation-resume", "atlas_status_digest": reviewed},
        observation=site.observation, membership=site.membership, capture_root=site.capture,
        storage_root=site.storage, handoff_payload=b"original selected envelope",
        listener=SimpleNamespace(process_handoff_payload=process))


def test_resume_uses_same_payload_native_claim_and_keeps_all_prior_attempts(site, monkeypatch):
    before = retained_native(site)
    write_job(site, state="ready")
    snapshot = recover(site)["atlas"]
    assert snapshot["native_preparation"]["ledger_digest"] == canonical_digest(before)
    assert snapshot["native_preparation"]["attempt_count"] == 5
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")
    seen = []

    def process(raw, **kwargs):
        seen.append((raw, kwargs))
        status, ledger = native._claim_job_lease(site.capture, scene_id="site-fixture",
            capture_id="walkthrough-fixture", owner="reviewed-recovery", lease_seconds=60,
            payload_sha256=hashlib.sha256(raw).hexdigest(), producer_delivery_key=before["producer_delivery_key"],
            expected_reconstruction_resume=kwargs["expected_reconstruction_resume"])
        assert status == "claimed" and ledger["attempt_count"] == 6
        assert ledger["attempt_history"] == before["attempt_history"]
        return {"status": "processed", "queue_disposition": "terminal_success"}

    result = resume(site, snapshot["atlas_status_digest"], process)
    assert result["status"] == "native_preparation_resumed" and result["native_handler_invoked"]
    assert seen[0][0] == b"original selected envelope"
    assert seen[0][1]["expected_preparation_purpose"] == "scene_preparation"
    assert seen[0][1]["storage_root"] == site.storage
    assert json.loads(site.job_path.read_text())["state"] == "ready"  # no reconstruction repurchase
    with pytest.raises(ValueError, match="selected_handoff_atlas_status_changed"):
        resume(site, snapshot["atlas_status_digest"],
               lambda *_, **__: pytest.fail("same review started a second native attempt"))


@pytest.mark.parametrize("change", ["history", "revision", "source", "owner", "not_ready", "completed"])
def test_changed_review_or_ineligible_parent_never_enters_native_processing(site, monkeypatch, change):
    ledger = retained_native(site)
    write_job(site, state="ready")
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    if change == "history":
        ledger["attempt_history"].append({"status": "unknown provider outcome"})
    elif change == "revision":
        ledger["revision"] += 1
    elif change == "source":
        ledger["producer_delivery_key"] = "sha256:" + "f" * 64
    elif change == "owner":
        site.observation["producer_delivery"]["delivery_key"] = "sha256:" + "e" * 64
    elif change == "not_ready":
        write_job(site, state="processing")
    elif change == "completed":
        ledger["status"] = "completed"
    (site.capture / native.JOB_LEDGER_FILENAME).write_text(json.dumps(ledger))
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")
    with pytest.raises(ValueError, match="selected_handoff_atlas_"):
        resume(site, reviewed, lambda *_, **__: pytest.fail("unreviewed native processing"))


@pytest.mark.parametrize("state", ["processing", "ready", "completed", "authority_ended"])
def test_native_resume_claim_refuses_other_states_even_with_matching_digest(site, state):
    ledger = retained_native(site)
    ledger["status"] = state
    (site.capture / native.JOB_LEDGER_FILENAME).write_text(json.dumps(ledger))
    status, retained = native._claim_job_lease(site.capture, scene_id="site-fixture",
        capture_id="walkthrough-fixture", owner="operator", lease_seconds=60,
        payload_sha256="b" * 64, producer_delivery_key=ledger["producer_delivery_key"],
        expected_reconstruction_resume=review(canonical_digest(ledger)))
    assert status == "reconstruction_resume_changed" and retained == ledger


@pytest.mark.parametrize("change", ["revision", "lease_owner", "lease_token", "lease_expires_at", "recovered_expired_lease"])
def test_native_claim_checks_review_under_its_own_lock(site, change):
    ledger = retained_native(site)
    reviewed = canonical_digest(ledger)
    ledger[change] = 13 if change == "revision" else True if change == "recovered_expired_lease" else "live-or-unresolved"
    (site.capture / native.JOB_LEDGER_FILENAME).write_text(json.dumps(ledger))
    if change != "revision":
        reviewed = canonical_digest(ledger)  # reviewing an unresolved lease never authorizes it
    status, retained = native._claim_job_lease(site.capture, scene_id="site-fixture",
        capture_id="walkthrough-fixture", owner="operator", lease_seconds=60,
        payload_sha256="b" * 64, producer_delivery_key=ledger["producer_delivery_key"],
        expected_reconstruction_resume=review(reviewed))
    assert status == "reconstruction_resume_changed" and retained == ledger


def test_source_conflict_still_refuses_a_reviewed_resume(site):
    ledger = retained_native(site)
    status, _ = native._claim_job_lease(site.capture, scene_id="site-fixture",
        capture_id="walkthrough-fixture", owner="operator", lease_seconds=60,
        payload_sha256="b" * 64, producer_delivery_key="sha256:" + "e" * 64,
        expected_reconstruction_resume=review(canonical_digest(ledger)))
    assert status == "reconstruction_resume_changed"


@pytest.mark.parametrize("field,value", [
    ("selected_payload_sha256", "c" * 64), ("ledger_digest", "invalid"),
    ("review_schema", "unrecognized-review"), ("new_spend_authority", 999),
])
def test_review_cannot_authorize_another_envelope_or_expand_its_fields(site, field, value):
    ledger = retained_native(site)
    expected = {**review(canonical_digest(ledger)), field: value}
    status, after = native._claim_job_lease(site.capture, scene_id="site-fixture",
        capture_id="walkthrough-fixture", owner="operator", lease_seconds=60,
        payload_sha256="b" * 64, producer_delivery_key=ledger["producer_delivery_key"],
        expected_reconstruction_resume=expected)
    assert status == "reconstruction_resume_changed" and after == ledger


@pytest.mark.parametrize("atlas_only_limit", ["quota", "price_review"])
def test_finished_atlas_limits_do_not_become_a_second_native_budget(site, monkeypatch, atlas_only_limit):
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    retained_native(site)
    write_job(site, state="ready")
    if atlas_only_limit == "quota":
        site.config["authority"]["maximum_cost_usd"] = 1.5
        site.config["remaining_cost_usd"] = 0
    else:
        def old_price(_):
            raise ValueError("expired Atlas price review")
        monkeypatch.setattr(pipeline, "_require_current_price_review", old_price)
    snapshot = recover(site)["atlas"]
    assert snapshot["preflight_blockers"]
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")
    calls = []
    result = resume(site, snapshot["atlas_status_digest"],
        lambda *args, **kw: calls.append((args, kw)) or {
            "status": "processed", "queue_disposition": "terminal_success"})
    assert result["native_handler_invoked"] and len(calls) == 1


def test_global_paid_admission_still_blocks_native_resume(site, monkeypatch):
    from blueprint_pipeline import paid_resource_admission as budget
    retained_native(site)
    write_job(site, state="ready")
    def refuse(**_):
        raise ValueError("shared spending stop")
    monkeypatch.setattr(budget, "require_paid_resource_budget", refuse)
    snapshot = recover(site)["atlas"]
    with pytest.raises(ValueError, match="selected_handoff_atlas_preflight_blocked"):
        resume(site, snapshot["atlas_status_digest"], lambda *_, **__: pytest.fail("spending stop bypassed"))
