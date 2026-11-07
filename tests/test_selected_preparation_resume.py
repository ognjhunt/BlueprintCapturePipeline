"""Resume retained website preparation without resetting delivery or spending history."""
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
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


@pytest.mark.parametrize("age_seconds", [60, recovery.MAXIMUM_JOB_SECONDS + 1])
def test_resume_uses_same_payload_native_claim_and_keeps_all_prior_attempts(site, monkeypatch, age_seconds):
    before = retained_native(site)
    write_job(site, state="ready", enqueued_at_epoch=int(time.time()) - age_seconds)
    atlas_before, source_before, receipt_before = (path.read_bytes() for path in
        (site.job_path, site.video, site.retained))
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
    assert site.job_path.read_bytes() == atlas_before  # no reconstruction repurchase or deadline reset
    assert site.video.read_bytes() == source_before and site.retained.read_bytes() == receipt_before
    with pytest.raises(ValueError, match="selected_handoff_atlas_status_changed"):
        resume(site, snapshot["atlas_status_digest"],
               lambda *_, **__: pytest.fail("same review started a second native attempt"))


@pytest.mark.parametrize("state", ["processing", "failed", "needs_input", "withdrawn"])
def test_old_processing_deadline_does_not_make_unfinished_atlas_usable(site, monkeypatch, state):
    retained_native(site)
    write_job(site, state=state, enqueued_at_epoch=int(time.time()) - recovery.MAXIMUM_JOB_SECONDS - 1)
    snapshot = recover(site)["atlas"]
    assert "atlas_preview_expired" in snapshot["preflight_blockers"]
    before = atlas_tests.tree_bytes(site)
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")
    with pytest.raises(ValueError, match="selected_handoff_atlas_"):
        resume(site, snapshot["atlas_status_digest"], lambda *_, **__: pytest.fail("unfinished Atlas reused"))
    assert atlas_tests.tree_bytes(site) == before


@pytest.mark.parametrize("refusal", ["source_revoked", "website_atlas_source_owner_mismatch",
                                     "provider_terms_not_configured_or_changed", "website_atlas_consent_expired"])
def test_aged_ready_reuse_keeps_current_permission_refusals(site, monkeypatch, refusal):
    retained_native(site)
    write_job(site, state="ready", enqueued_at_epoch=int(time.time()) - recovery.MAXIMUM_JOB_SECONDS - 1)
    def refuse(*_args, **_kwargs):
        raise ValueError("website_atlas_configuration:" + refusal)
    monkeypatch.setattr(recovery, "_configuration", refuse)
    snapshot = recover(site)["atlas"]
    assert refusal in snapshot["preflight_blockers"]
    before = atlas_tests.tree_bytes(site)
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")
    with pytest.raises(ValueError, match="selected_handoff_atlas_preflight_blocked"):
        resume(site, snapshot["atlas_status_digest"], lambda *_, **__: pytest.fail("current permission bypassed"))
    assert atlas_tests.tree_bytes(site) == before


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


@pytest.mark.parametrize("age_seconds", [60, recovery.MAXIMUM_JOB_SECONDS + 1])
def test_global_paid_admission_still_blocks_native_resume(site, monkeypatch, age_seconds):
    from blueprint_pipeline import paid_resource_admission as budget
    retained_native(site)
    write_job(site, state="ready", enqueued_at_epoch=int(time.time()) - age_seconds)
    def refuse(**_):
        raise ValueError("shared spending stop")
    monkeypatch.setattr(budget, "require_paid_resource_budget", refuse)
    snapshot = recover(site)["atlas"]
    with pytest.raises(ValueError, match="selected_handoff_atlas_preflight_blocked"):
        resume(site, snapshot["atlas_status_digest"], lambda *_, **__: pytest.fail("spending stop bypassed"))


def _iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parsed(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def dead_native(site, *, expires_in_seconds=-60, **changes):
    """A reviewed resume claimed attempt 6, then the kernel killed its process: only the lease is left."""
    ledger = retained_native(site)
    now = datetime.now(timezone.utc)
    acquired = now + timedelta(seconds=expires_in_seconds - native.DEFAULT_JOB_LEASE_SECONDS)
    ledger.update(status="processing", revision=13, attempt_count=6, last_attempt_started_at=_iso(acquired),
                  lease_owner="control-plane:4242:" + "a" * 32, lease_token="c" * 32,
                  lease_acquired_at=_iso(acquired), lease_heartbeat_at=_iso(acquired),
                  lease_expires_at=_iso(now + timedelta(seconds=expires_in_seconds)),
                  recovered_expired_lease=False, previous_lease_owner=None)
    ledger.update(changes)
    (site.capture / native.JOB_LEDGER_FILENAME).write_text(json.dumps(ledger))
    return ledger


def claim(site, expected, *, payload_sha256="b" * 64, producer_delivery_key="sha256:" + "d" * 64, now=None):
    return native._claim_job_lease(site.capture, scene_id="site-fixture", capture_id="walkthrough-fixture",
        owner="operator", lease_seconds=60, payload_sha256=payload_sha256,
        producer_delivery_key=producer_delivery_key, expected_reconstruction_resume=expected,
        **({"now": now} if now is not None else {}))


def on_disk(site):
    return json.loads((site.capture / native.JOB_LEDGER_FILENAME).read_text())


def test_a_reviewed_dead_attempt_is_recorded_failed_then_claimed_under_one_lock(site, monkeypatch):
    dead = dead_native(site)
    commits = []
    commit = native._commit_job_ledger
    monkeypatch.setattr(native, "_commit_job_ledger",
                        lambda *args, **kwargs: commits.append(commit(*args, **kwargs)) or commits[-1])
    status, claimed = claim(site, review(canonical_digest(dead)))
    assert status == "claimed" and len(commits) == 2
    ended, row = commits[0], commits[0]["attempt_history"][-1]
    # The dead attempt keeps its number and gains one row; nothing before it changes.
    assert ended["attempt_history"][:-1] == dead["attempt_history"]
    assert row == {"attempt_number": 6, "status": "failed_retryable", "stage": continuation.LEASE_EXPIRED_STAGE,
                   "started_at": dead["last_attempt_started_at"], "failed_at": row["recovered_at"],
                   "error_type": "HandoffLeaseError", "error": continuation.LEASE_EXPIRED_ERROR,
                   "outcome": "unknown", "recovered_expired_lease": True,
                   "previous_lease_owner": dead["lease_owner"], "lease_acquired_at": dead["lease_acquired_at"],
                   "lease_heartbeat_at": dead["lease_heartbeat_at"], "lease_expires_at": dead["lease_expires_at"],
                   "recovered_at": row["recovered_at"], "recovered_by": continuation.REVIEW_SCHEMA,
                   "reviewed_ledger_digest": canonical_digest(dead)}
    assert _parsed(row["recovered_at"]) >= _parsed(dead["lease_expires_at"])
    assert {key: ended[key] for key in ("status", "revision", "attempt_count", "last_error",
                                        "lease_owner", "lease_token", "lease_expires_at")} == {
        "status": "failed_retryable", "revision": 14, "attempt_count": 6,
        "last_error": continuation.LEASE_EXPIRED_ERROR,
        "lease_owner": None, "lease_token": None, "lease_expires_at": None}
    # The claim on that failure is the ordinary reviewed retry: exactly one more attempt.
    assert claimed == commits[1] == on_disk(site)
    assert claimed["status"] == "processing" and claimed["revision"] == 15 and claimed["attempt_count"] == 7
    assert claimed["attempt_history"] == ended["attempt_history"]
    assert claimed["lease_owner"] == "operator" and claimed["lease_token"] != dead["lease_token"]
    assert claimed["recovered_expired_lease"] is False and claimed["previous_lease_owner"] is None
    # The unknown outcome is never fresh wait capacity for the new attempt.
    assert not continuation.safe_to_arm(claimed["attempt_count"], claimed["attempt_history"],
                                        claimed["recovered_expired_lease"])


def test_a_lease_that_has_not_expired_is_refused_as_active(site):
    live = dead_native(site, expires_in_seconds=600)
    status, after = claim(site, review(canonical_digest(live)))
    assert status == "active" and after == live == on_disk(site)


@pytest.mark.parametrize("change", ["digest", "payload", "delivery_key", "extra_field", "no_token",
                                    "unreadable_expiry", "unrecorded_takeover"])
def test_a_dead_attempt_takeover_keeps_every_review_check(site, change):
    dead = dead_native(site, **({"lease_token": None} if change == "no_token" else
                                {"lease_expires_at": "unresolved"} if change == "unreadable_expiry" else
                                {"recovered_expired_lease": True, "previous_lease_owner": "lost-worker"}
                                if change == "unrecorded_takeover" else {}))
    expected = review(canonical_digest({**dead, "revision": 12}) if change == "digest" else canonical_digest(dead))
    if change == "extra_field":
        expected["new_spend_authority"] = 999
    status, after = claim(site, expected, payload_sha256="c" * 64 if change == "payload" else "b" * 64,
        producer_delivery_key="sha256:" + ("e" if change == "delivery_key" else "d") * 64)
    assert status == "reconstruction_resume_changed" and after == dead == on_disk(site)


def test_one_review_takes_over_one_dead_attempt(site):
    dead = dead_native(site)
    expected = review(canonical_digest(dead))
    assert claim(site, expected)[0] == "claimed"
    taken = on_disk(site)
    assert taken["attempt_count"] == dead["attempt_count"] + 1
    assert len(taken["attempt_history"]) == taken["attempt_count"] - 1
    # The same review is spent: refused while the new lease runs and after it too has run out.
    assert claim(site, expected) == ("active", taken)
    later = datetime.now(timezone.utc) + timedelta(hours=1)
    assert claim(site, expected, now=later) == ("reconstruction_resume_changed", taken)
    assert on_disk(site) == taken


def test_operator_resume_of_a_dead_attempt_goes_through_the_native_claim(site, monkeypatch):
    dead = dead_native(site)
    write_job(site, state="ready")
    snapshot = recover(site)["atlas"]
    assert snapshot["native_preparation"]["status"] == "processing"
    assert snapshot["native_preparation"]["lease_expires_at"] == dead["lease_expires_at"]
    assert snapshot["native_preparation"]["ledger_digest"] == canonical_digest(dead)
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")
    seen = []

    def process(raw, **kwargs):
        seen.append(kwargs["expected_reconstruction_resume"])
        status, ledger = claim(site, kwargs["expected_reconstruction_resume"],
                               payload_sha256=hashlib.sha256(raw).hexdigest())
        assert status == "claimed" and ledger["attempt_count"] == 7
        assert ledger["attempt_history"][-1]["previous_lease_owner"] == dead["lease_owner"]
        return {"status": "processed", "queue_disposition": "terminal_success"}

    result = resume(site, snapshot["atlas_status_digest"], process)
    assert result["status"] == "native_preparation_resumed" and len(seen) == 1
    assert seen[0]["ledger_digest"] == canonical_digest(dead)


@pytest.mark.parametrize("lease,refusal", [(600, "native_lease_active"), (None, "native_resume_not_eligible")])
def test_operator_resume_refuses_a_live_or_missing_lease_before_the_native_handler(site, monkeypatch, lease, refusal):
    if lease is None:
        dead_native(site, lease_owner=None, lease_token=None, lease_expires_at=None)
    else:
        dead_native(site, expires_in_seconds=lease)
    write_job(site, state="ready")
    snapshot = recover(site)["atlas"]
    before = atlas_tests.tree_bytes(site)
    monkeypatch.setenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "openai")
    with pytest.raises(ValueError, match="selected_handoff_atlas_" + refusal):
        resume(site, snapshot["atlas_status_digest"], lambda *_, **__: pytest.fail("live lease taken over"))
    assert atlas_tests.tree_bytes(site) == before
