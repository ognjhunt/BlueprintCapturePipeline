"""Hermetic cloud recovery of one original beta capture's existing Atlas job."""

import copy
import fcntl
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from blueprint_pipeline import selected_atlas_preview_recovery as recovery
from blueprint_pipeline import website_atlas_pipeline as pipeline
from blueprint_pipeline import website_atlas_preview as preview
from blueprint_pipeline import website_atlas_preview_store as store
from blueprint_pipeline import website_task_context as context
from blueprint_pipeline.capture_delivery_staging import CaptureStagingError
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


@pytest.fixture
def site(tmp_path, monkeypatch):
    from blueprint_pipeline import capture_original_owner_observer as owner
    from blueprint_pipeline import paid_resource_admission as budget
    from blueprint_pipeline import task_evaluation_scene_retirement_generations as generations

    storage = tmp_path / "storage"
    capture = storage / "fixture-bucket/scenes/site-fixture/captures/walkthrough-fixture"
    (capture / "raw").mkdir(parents=True)
    atlas = capture / "pipeline/atlas"
    atlas.mkdir(parents=True)
    video = capture / "raw/walkthrough.mov"
    video.write_bytes(b"immutable original video")
    # A read must not be treated as a source edit when it updates access time.
    os.utime(video, (1, 1))
    digest = "sha256:" + hashlib.sha256(video.read_bytes()).hexdigest()
    os.utime(video, (1, 1))
    now = int(time.time())
    selection = {"kind": "selected-handoff", "mode": "atlas-status", "bucket": "fixture-bucket",
                 "scene_id": "site-fixture", "capture_id": "walkthrough-fixture", "marker_generation": "11"}
    original = {"object_name": "scenes/site-fixture/captures/walkthrough-fixture/raw/walkthrough.mov",
                "generation": "12", "size_bytes": video.stat().st_size, "crc32c": "AAAAAA=="}
    observation = {"request_id": "fixture", "producer_delivery": {"raw_video": original},
                   "source_projection_digest": "sha256:" + "a" * 64,
                   "capture_owner": {"email": "private-owner@example.invalid"},
                   "valid_until_epoch": now + 60}
    row = {**original, "relative_path": "raw/walkthrough.mov", "sha256": digest}
    membership = (b"immutable membership", {"raw": [row]}, [])
    job = {"schema_version": store.SCHEMA, "request_id": "fixture", "scene_id": "site-fixture",
           "capture_id": "walkthrough-fixture", "source_video": row["relative_path"],
           "state": "failed", "reason": "atlas_scene_planner_transients_invalid", "stage": "queued",
           "enqueued_at_epoch": now - 60, "revision": 4, "updated_at_epoch": now - 1,
           "revised_at_epoch": now - 1, "next_attempt_at_epoch": now - 1, "retry_at_epoch": None,
           "transient_failures": 2, "retries": [], "archive": None,
           "provider_cleanup": {"state": "complete", "attempts": 1, "deleted": ["old-asset"],
                                "remaining": [], "next_attempt_at_epoch": 0}}
    job_path, lock_path = atlas / store.JOB_FILE, atlas / store.LOCK_FILE
    job_path.write_text(json.dumps(job))
    lock_path.touch()
    retained = atlas / "retained-planner-receipt.json"
    retained.write_bytes(b"paid stage receipt must stay identical")
    config = {"schema_version": "website_atlas_configuration.v1", "source_video_digest": digest,
              "task_context_digest": "sha256:" + "b" * 64, "dispatch_authorized": False,
              "grant_creation_allowed": False,
              "authority": {"request_id": "fixture", "capture_id": "walkthrough-fixture",
                            "source_video_digest": digest, "maximum_cost_usd": 25,
                            "maximum_requests": 32, "expires_at_epoch": now + 3600,
                            "accepted_by": "private-owner@example.invalid", "secret": "private-api-secret"},
              "reserved_or_settled_cost_usd": 1.5, "remaining_cost_usd": 23.5, "remaining_requests": 30,
              "policy": {"reviewed_on": "2026-10-27", "maximum_review_cost_usd": 1,
                         "reviewed_by": "private-reviewer@example.invalid"}}
    common_accounting = {"external_capture_assessment_costs": "not_in_preparation_ledger"}
    reservations = [{"allocation_binding_digest": "sha256:" + key * 64, "provider": "world_labs_atlas",
                     "maximum_cost_usd": cost, "settled_cost_usd": cost if settled else None,
                     "accounted_cost_usd": cost, "request_count": 1, "status": "settled" if settled else "held",
                     "task_context_digest": config["task_context_digest"], "source_video_digest": digest,
                     "atlas_authorization_digest": "sha256:" + "e" * 64,
                     "settlement_digest": "sha256:" + "f" * 64 if settled else None}
                    for key, cost, settled in (("c", 1, True), ("d", 0.5, False))]
    config["allowance_accounting"] = {**common_accounting,
        "schema_version": "website_atlas_allowance_accounting.v1", "source_scope": "request_atlas_allowance",
        "settled_cost_usd": 1, "held_cost_usd": 0.5, "reserved_or_settled_cost_usd": 1.5,
        "request_count": 2, "reservations": reservations}
    config["native_preparation_accounting"] = {**common_accounting,
        "schema_version": "website_native_preparation_accounting.v1", "source_scope": "request_native_preparation_ledger",
        "settled_cost_usd": 0, "held_cost_usd": 0, "reserved_or_settled_cost_usd": 0,
        "request_count": 0, "reservations": []}
    state = SimpleNamespace(storage=storage, capture=capture, atlas=atlas, video=video,
        selection=selection, observation=observation, membership=membership, job=job, job_path=job_path,
        lock_path=lock_path, retained=retained, config=config, calls=[], owner_checks=[], admissions=[])

    @contextmanager
    def admission(path, *, required):
        state.admissions.append((path, required))
        yield

    def webapp(*, capture_id, operation, payload):
        state.calls.append((capture_id, operation, copy.deepcopy(payload)))
        assert operation == "atlas-configuration"
        assert payload.get("create") is False
        return copy.deepcopy(state.config)

    monkeypatch.setattr(generations, "_capture_admission", admission)
    monkeypatch.setattr(owner, "validate_observation", lambda value, **kw: state.owner_checks.append((value, kw)))
    monkeypatch.setattr(context, "load_current_website_task_context", lambda **kw: {
        **kw, "context_digest": "sha256:" + "b" * 64})
    monkeypatch.setattr(context, "website_webapp_request", webapp)
    monkeypatch.setattr(budget, "require_paid_resource_budget", lambda **kw: None)
    monkeypatch.setattr(pipeline, "_require_current_price_review", lambda policy: None)
    monkeypatch.setattr(store, "capture_withdrawn", lambda root: False)
    monkeypatch.setattr(preview, "_withdrawn", lambda root: False)

    def provider_call_forbidden(*args, **kwargs):
        raise AssertionError("recovery must not advance a provider stage")

    monkeypatch.setattr(pipeline, "run_atlas_pipeline", provider_call_forbidden)
    monkeypatch.setattr(preview, "advance_job", provider_call_forbidden)
    monkeypatch.setattr(preview, "_step", provider_call_forbidden)
    return state


def recover(site, *, retry_digest=None):
    selection = {**site.selection}
    if retry_digest is not None:
        selection.update(mode="atlas-retry", atlas_status_digest=retry_digest)
    return recovery.recover_selected_atlas_preview(selection, observation=site.observation,
        membership=site.membership, capture_root=site.capture, storage_root=site.storage)


def write_job(site, **updates):
    value = json.loads(site.job_path.read_text())
    value.update(updates)
    site.job_path.write_text(json.dumps(value))
    return value


def tree_bytes(site):
    return {str(path.relative_to(site.capture)): path.read_bytes()
            for path in site.capture.rglob("*") if path.is_file()}


def test_status_is_read_only_source_bound_and_uses_only_current_existing_grant(site):
    before = tree_bytes(site)
    result = recover(site)
    assert result["status"] == "atlas_status_returned" and result["atlas_retry_performed"] is False
    assert tree_bytes(site) == before
    assert site.admissions == [(site.capture, True)]
    assert len(site.owner_checks) == 1
    assert site.calls == [("walkthrough-fixture", "atlas-configuration", {
        "request_id": "fixture", "scene_id": "site-fixture", "task_context_digest": "sha256:" + "b" * 64,
        "source_video_digest": site.membership[1]["raw"][0]["sha256"], "create": False})]
    atlas = result["atlas"]
    assert atlas["job"]["expires_at_epoch"] == site.job["enqueued_at_epoch"] + store.MAXIMUM_JOB_SECONDS
    assert atlas["allowance"]["maximum_cost_usd"] == 25
    assert atlas["allowance"]["remaining_cost_usd"] == 23.5
    assert atlas["source"]["video_generation"] == "12" and atlas["preflight_blockers"] == []
    assert atlas["atlas_status_digest"] == canonical_digest(atlas, digest_field="atlas_status_digest")
    assert recover(site)["atlas"]["atlas_status_digest"] == atlas["atlas_status_digest"]
    encoded = json.dumps(result)
    assert "private-owner" not in encoded and "private-reviewer" not in encoded and "private-api-secret" not in encoded


@pytest.mark.parametrize("reason", ["atlas_scene_planner_transients_invalid", "atlas_scene_planner_abstained"])
def test_retry_calls_existing_controller_under_its_lock_preserving_source_expiry_and_receipts(site, monkeypatch, reason):
    site.job["reason"] = reason
    site.job["state"] = "needs_input" if reason == "atlas_scene_planner_abstained" else "failed"
    site.job_path.write_text(json.dumps(site.job))
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    original_retry = preview.retry_failed_job
    entered = []

    def retry(path, *, preflight):
        entered.append(path)

        def locked_preflight(job):
            with site.lock_path.open("rb") as competing_lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(competing_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return preflight(job)

        return original_retry(path, preflight=locked_preflight)

    monkeypatch.setattr(preview, "retry_failed_job", retry)
    before = tree_bytes(site)
    result = recover(site, retry_digest=reviewed)
    assert entered == [site.job_path] and result["status"] == "atlas_retry_queued"
    assert result["atlas_retry_performed"] is True and result["later_timer_may_make_paid_calls"] is True
    after = json.loads(site.job_path.read_text())
    assert after["state"] == "processing" and after["revision"] == site.job["revision"] + 1
    assert after["enqueued_at_epoch"] == site.job["enqueued_at_epoch"]
    assert after["source_video"] == site.job["source_video"]
    assert after["provider_cleanup"]["deleted"] == ["old-asset"]
    assert len(after["retries"]) == 1
    assert {key: value for key, value in tree_bytes(site).items() if key != "pipeline/atlas/preview_job.json"} == {
        key: value for key, value in before.items() if key != "pipeline/atlas/preview_job.json"}
    with pytest.raises(ValueError, match="atlas_preview_retry_requires_recoverable_failure"):
        recover(site, retry_digest=reviewed)
    assert len(json.loads(site.job_path.read_text())["retries"]) == 1


@pytest.mark.parametrize("change", ["source_bytes", "job_source", "capture_id", "request_id", "scene_id"])
def test_source_or_job_identity_change_refuses_before_webapp_read_and_never_writes(site, change):
    if change == "source_bytes":
        site.video.write_bytes(b"substituted original bytes")
    elif change == "job_source":
        write_job(site, source_video="raw/other.mov")
    else:
        write_job(site, **{change: "other-identity"})
    before = tree_bytes(site)
    with pytest.raises(ValueError, match="selected_handoff_atlas_(source_changed|source_binding_invalid|job_binding_invalid)"):
        recover(site)
    assert tree_bytes(site) == before and site.calls == []


@pytest.mark.parametrize("target", ["video", "job_path", "lock_path", "atlas"])
def test_symlink_or_ancestor_alias_refuses_before_webapp_read(site, tmp_path, target):
    path = getattr(site, target)
    destination = tmp_path / ("real-" + path.name)
    path.rename(destination)
    path.symlink_to(destination, target_is_directory=destination.is_dir())
    with pytest.raises(ValueError, match="selected_handoff_atlas_path_unsafe"):
        recover(site)
    assert site.calls == []


@pytest.mark.parametrize("change", ["job", "cap", "expiry", "usage", "policy"])
def test_reviewed_status_must_still_match_current_job_authority_and_allowance(site, change):
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    if change == "job":
        write_job(site, revision=site.job["revision"] + 1)
    elif change == "cap":
        site.config["authority"]["maximum_cost_usd"] += 1
        site.config["remaining_cost_usd"] += 1
    elif change == "expiry":
        site.config["authority"]["expires_at_epoch"] += 10
    elif change == "usage":
        site.config["reserved_or_settled_cost_usd"] += 1
        site.config["remaining_cost_usd"] -= 1
        accounting = site.config["allowance_accounting"]
        accounting["held_cost_usd"] += 1
        accounting["reserved_or_settled_cost_usd"] += 1
        accounting["reservations"][1]["maximum_cost_usd"] += 1
        accounting["reservations"][1]["accounted_cost_usd"] += 1
    else:
        site.config["policy"]["maximum_review_cost_usd"] += 1
    before = site.job_path.read_bytes()
    with pytest.raises(ValueError, match="selected_handoff_atlas_status_changed"):
        recover(site, retry_digest=reviewed)
    assert site.job_path.read_bytes() == before


@pytest.mark.parametrize("kind,expected", [
    ("expired_grant", "website_atlas_consent_expired"),
    ("exhausted", "website_scene_preparation_budget_exhausted"),
    ("price", "atlas_runtime_price_review_expired"),
    ("budget", "atlas_current_budget_admission_blocked"),
    ("config", "website_atlas_source_consent_required"),
    ("oversized_config", "website_atlas_accounting_response_too_large"),
])
def test_preflight_blockers_are_visible_and_cannot_requeue(site, monkeypatch, kind, expected):
    from blueprint_pipeline import paid_resource_admission as budget

    def refuse(*args, **kwargs):
        raise ValueError("private raw exception must not be returned")

    if kind == "expired_grant":
        site.config["authority"]["expires_at_epoch"] = int(time.time()) - 1
    elif kind == "exhausted":
        site.config["remaining_cost_usd"] = 0
        site.config["reserved_or_settled_cost_usd"] = 25
        accounting = site.config["allowance_accounting"]
        accounting.update(held_cost_usd=24, reserved_or_settled_cost_usd=25)
        accounting["reservations"][1].update(maximum_cost_usd=24, accounted_cost_usd=24)
    elif kind == "price":
        monkeypatch.setattr(pipeline, "_require_current_price_review", refuse)
    elif kind == "budget":
        monkeypatch.setattr(budget, "require_paid_resource_budget", refuse)
    else:
        def missing_grant(*args, **kwargs):
            raise ValueError("website_control_atlas-configuration_http_409:" + expected)
        monkeypatch.setattr(recovery, "_configuration", missing_grant)
    result = recover(site)
    assert expected in result["atlas"]["preflight_blockers"]
    assert "private raw" not in json.dumps(result)
    before = site.job_path.read_bytes()
    with pytest.raises(ValueError, match="selected_handoff_atlas_preflight_blocked"):
        recover(site, retry_digest=result["atlas"]["atlas_status_digest"])
    assert site.job_path.read_bytes() == before


def test_elapsed_original_job_expiry_is_observed_and_retry_refuses_without_reset(site):
    write_job(site, enqueued_at_epoch=int(time.time()) - store.MAXIMUM_JOB_SECONDS - 1)
    result = recover(site)
    assert "atlas_preview_expired" in result["atlas"]["preflight_blockers"]
    before = site.job_path.read_bytes()
    with pytest.raises(ValueError, match="atlas_preview_expired"):
        recover(site, retry_digest=result["atlas"]["atlas_status_digest"])
    assert site.job_path.read_bytes() == before


@pytest.mark.parametrize("mode", ["status", "retry"])
def test_busy_worker_lock_prevents_status_or_retry_without_changes(site, mode):
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    before = tree_bytes(site)
    with site.lock_path.open("r+") as worker:
        fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="atlas_job_busy|atlas_preview_job_busy"):
            recover(site, retry_digest=reviewed if mode == "retry" else None)
    assert tree_bytes(site) == before


def test_concurrent_retry_can_only_queue_once(site, monkeypatch):
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    entered, release = threading.Event(), threading.Event()
    original_config = recovery._configuration

    def paused_config(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_config(*args, **kwargs)

    monkeypatch.setattr(recovery, "_configuration", paused_config)
    results = []

    def first_retry():
        try:
            results.append(recover(site, retry_digest=reviewed))
        except BaseException as exc:
            results.append(exc)

    thread = threading.Thread(target=first_retry)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(ValueError, match="atlas_preview_job_busy"):
            recover(site, retry_digest=reviewed)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(results) == 1 and isinstance(results[0], dict), results
    assert results[0]["status"] == "atlas_retry_queued"
    assert len(json.loads(site.job_path.read_text())["retries"]) == 1


def test_stale_current_owner_observation_cannot_requeue(site, monkeypatch):
    from blueprint_pipeline import capture_original_owner_observer as owner

    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    before = site.job_path.read_bytes()

    def stale(*args, **kwargs):
        raise owner.CaptureOwnerObservationError("capture_owner_observation_stale")

    monkeypatch.setattr(owner, "validate_observation", stale)
    with pytest.raises(owner.CaptureOwnerObservationError, match="capture_owner_observation_stale"):
        recover(site, retry_digest=reviewed)
    assert site.job_path.read_bytes() == before


def test_current_withdrawal_refuses_before_configuration_or_mutation(site, monkeypatch):
    monkeypatch.setattr(store, "capture_withdrawn", lambda root: True)
    before = tree_bytes(site)
    with pytest.raises(ValueError, match="selected_handoff_atlas_capture_withdrawn"):
        recover(site)
    assert tree_bytes(site) == before and site.calls == []


def test_status_requires_existing_lock_and_never_creates_one(site):
    site.lock_path.unlink()
    with pytest.raises(FileNotFoundError):
        recover(site)
    assert not site.lock_path.exists() and site.calls == []


@pytest.mark.parametrize("change", ["top_level_secret", "row_secret", "too_many", "wrong_scope", "nonfinite", "boolean"])
def test_accounting_projection_rejects_unbounded_or_private_payloads(site, change):
    accounting = site.config["allowance_accounting"]
    if change == "top_level_secret":
        accounting["owner_email"] = "private-owner@example.invalid"
    elif change == "row_secret":
        accounting["reservations"][0]["provider_response"] = {"token": "private-api-secret"}
    elif change == "too_many":
        accounting["reservations"] *= 65
    elif change == "wrong_scope":
        accounting["source_scope"] = "new_combined_fifty_dollar_allowance"
    elif change == "nonfinite":
        accounting["held_cost_usd"] = float("nan")
    else:
        accounting["request_count"] = True
    before = tree_bytes(site)
    with pytest.raises(ValueError, match="selected_handoff_atlas_accounting_invalid") as failure:
        recover(site)
    assert "private" not in str(failure.value) and tree_bytes(site) == before


def test_reservation_order_is_not_a_change_to_reviewed_accounting(site):
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    site.config["allowance_accounting"]["reservations"].reverse()
    result = recover(site, retry_digest=reviewed)
    assert result["status"] == "atlas_retry_queued"


def test_review_digest_binds_exact_original_selector_and_membership_bytes(site):
    first = recover(site)["atlas"]["atlas_status_digest"]
    site.selection["handoff_generation"] = "991"
    changed = recover(site)["atlas"]["atlas_status_digest"]
    assert first != changed
    site.membership = (b"different selected membership", *site.membership[1:])
    assert recover(site)["atlas"]["atlas_status_digest"] != changed


@pytest.mark.parametrize("change", ["expiry", "withdrawal"])
def test_existing_retry_rechecks_expiry_and_withdrawal_after_slow_preflight(site, monkeypatch, change):
    clock = site.job["enqueued_at_epoch"] + store.MAXIMUM_JOB_SECONDS - 1
    withdrawn = False
    monkeypatch.setattr(preview, "_withdrawn", lambda root: withdrawn)

    def preflight(job):
        nonlocal clock, withdrawn
        if change == "expiry":
            clock += 2
        else:
            withdrawn = True

    before = site.job_path.read_bytes()
    with pytest.raises(ValueError, match="atlas_preview_expired|website_capture_withdrawn"):
        preview.retry_failed_job(site.job_path, now=lambda: clock, preflight=preflight)
    assert site.job_path.read_bytes() == before


@pytest.mark.parametrize("creation_allowed", [None, True])
def test_configuration_must_attest_read_only_existing_grant(site, creation_allowed):
    site.config["grant_creation_allowed"] = creation_allowed
    result = recover(site)
    assert result["atlas"]["preflight_blockers"] == ["atlas_current_configuration_unavailable"]
    before = site.job_path.read_bytes()
    with pytest.raises(ValueError, match="selected_handoff_atlas_preflight_blocked"):
        recover(site, retry_digest=result["atlas"]["atlas_status_digest"])
    assert site.job_path.read_bytes() == before
    assert all(call[2]["create"] is False for call in site.calls)


@pytest.mark.parametrize("state", ["processing", "ready", "withdrawn"])
def test_selected_retry_never_requeues_a_running_ready_or_withdrawn_job(site, state):
    write_job(site, state=state)
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    before = site.job_path.read_bytes()
    with pytest.raises(ValueError, match="retry_requires_recoverable_failure"):
        recover(site, retry_digest=reviewed)
    assert site.job_path.read_bytes() == before


@pytest.mark.parametrize("reason", ["atlas_task_and_scene_coverage_gap", "atlas_preview_expired",
                                  "atlas_preview_source_video_unavailable"])
def test_selected_retry_preserves_nonrecoverable_stopped_jobs(site, reason):
    write_job(site, state="needs_input", reason=reason)
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    before = tree_bytes(site)
    with pytest.raises(ValueError, match="retry_requires_recoverable_failure"):
        recover(site, retry_digest=reviewed)
    assert tree_bytes(site) == before


def test_raw_configuration_exception_is_not_exposed_in_status(site, monkeypatch):
    def private_failure(*args, **kwargs):
        raise ValueError("private-owner@example.invalid:provider-token=private-api-secret")

    monkeypatch.setattr(recovery, "_configuration", private_failure)
    result = recover(site)
    assert result["atlas"]["preflight_blockers"] == ["atlas_current_configuration_unavailable"]
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("ledger", ["allowance_accounting", "native_preparation_accounting"])
@pytest.mark.parametrize("total", ["settled_cost_usd", "held_cost_usd", "reserved_or_settled_cost_usd", "request_count"])
def test_ledger_totals_must_equal_reservation_rows_before_status_or_retry(site, ledger, total):
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    site.config[ledger][total] += 1
    before = tree_bytes(site)
    with pytest.raises(ValueError, match="selected_handoff_atlas_accounting_invalid"):
        recover(site)
    with pytest.raises(ValueError, match="selected_handoff_atlas_accounting_invalid"):
        recover(site, retry_digest=reviewed)
    assert tree_bytes(site) == before


@pytest.mark.parametrize("change", ["held_has_settlement", "settled_missing_cost", "held_wrong_accounted", "settled_wrong_accounted"])
def test_each_reservation_accounts_consistently_with_its_held_or_settled_state(site, change):
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    settled, held = site.config["allowance_accounting"]["reservations"]
    if change == "held_has_settlement":
        held["settled_cost_usd"] = held["maximum_cost_usd"]
    elif change == "settled_missing_cost":
        settled["settled_cost_usd"] = None
    elif change == "held_wrong_accounted":
        held["accounted_cost_usd"] += 0.1
    else:
        settled["accounted_cost_usd"] += 0.1
    before = site.job_path.read_bytes()
    with pytest.raises(ValueError, match="selected_handoff_atlas_accounting_invalid"):
        recover(site, retry_digest=reviewed)
    assert site.job_path.read_bytes() == before


def test_duplicate_allocation_identity_within_one_ledger_is_refused(site):
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    rows = site.config["allowance_accounting"]["reservations"]
    rows[1]["allocation_binding_digest"] = rows[0]["allocation_binding_digest"]
    before = site.job_path.read_bytes()
    with pytest.raises(ValueError, match="selected_handoff_atlas_accounting_invalid"):
        recover(site, retry_digest=reviewed)
    assert site.job_path.read_bytes() == before


def test_one_allocation_cannot_be_counted_in_both_native_and_atlas_ledgers(site):
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    duplicate = copy.deepcopy(site.config["allowance_accounting"]["reservations"][0])
    native = site.config["native_preparation_accounting"]
    native.update(settled_cost_usd=duplicate["settled_cost_usd"], held_cost_usd=0,
                  reserved_or_settled_cost_usd=duplicate["accounted_cost_usd"], request_count=1,
                  reservations=[duplicate])
    before = site.job_path.read_bytes()
    with pytest.raises(ValueError, match="selected_handoff_atlas_accounting_invalid"):
        recover(site, retry_digest=reviewed)
    assert site.job_path.read_bytes() == before


@pytest.mark.parametrize("change", ["usage", "remaining_cost", "remaining_requests", "fractional_requests"])
def test_current_configuration_allowance_must_agree_with_checked_ledger(site, change):
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    if change == "usage":
        # Keep this pair mutually consistent, while disagreeing with the ledger.
        site.config["reserved_or_settled_cost_usd"] += 1
        site.config["remaining_cost_usd"] -= 1
    elif change == "remaining_cost":
        site.config["remaining_cost_usd"] += 1
    elif change == "remaining_requests":
        site.config["remaining_requests"] += 1
    else:
        site.config["remaining_requests"] = 0.5
    before = site.job_path.read_bytes()
    with pytest.raises(ValueError, match="selected_handoff_atlas_(accounting_invalid|configuration_invalid)"):
        recover(site, retry_digest=reviewed)
    assert site.job_path.read_bytes() == before


def test_a_dollar_only_allowance_is_not_blocked_by_a_request_count(site):
    # Owner decision 2026-10-30: money is the hard stop; requests are only counted.
    site.config.pop("remaining_requests")
    result = recover(site)["atlas"]
    assert "remaining_requests" not in result["allowance"]
    assert "website_scene_preparation_budget_exhausted" not in result["preflight_blockers"]


def test_native_preparation_costs_remain_separate_from_the_original_atlas_cap(site):
    native_row = copy.deepcopy(site.config["allowance_accounting"]["reservations"][0])
    native_row.update(allocation_binding_digest="sha256:" + "9" * 64, provider="openai",
                      maximum_cost_usd=18, settled_cost_usd=18, accounted_cost_usd=18,
                      atlas_authorization_digest=None)
    site.config["native_preparation_accounting"].update(settled_cost_usd=18,
        reserved_or_settled_cost_usd=18, request_count=1, reservations=[native_row])
    result = recover(site)["atlas"]
    assert result["allowance"]["maximum_cost_usd"] == 25
    assert result["allowance"]["remaining_cost_usd"] == 23.5
    assert result["allowance_accounting"]["reserved_or_settled_cost_usd"] == 1.5
    assert result["native_preparation_accounting"]["reserved_or_settled_cost_usd"] == 18


def test_request_allowance_exhaustion_uses_current_original_cap_without_reset(site):
    site.config["authority"]["maximum_requests"] = 2
    site.config["remaining_requests"] = 0
    result = recover(site)
    assert result["atlas"]["preflight_blockers"] == ["website_scene_preparation_budget_exhausted"]
    before = site.job_path.read_bytes()
    with pytest.raises(ValueError, match="selected_handoff_atlas_preflight_blocked"):
        recover(site, retry_digest=result["atlas"]["atlas_status_digest"])
    assert site.job_path.read_bytes() == before


def test_a_long_native_ledger_is_read_not_refused(site):
    from blueprint_pipeline import pubsub_handoff_listener as listener

    ledger = {'scene_id': 'site-fixture', 'capture_id': 'walkthrough-fixture',
              'status': 'failed_retryable', 'revision': 13, 'attempt_count': 5,
              'history': [{'attempt': i, 'detail': 'x' * 200} for i in range(700)]}
    (site.capture / listener.JOB_LEDGER_FILENAME).write_text(json.dumps(ledger))
    assert (site.capture / listener.JOB_LEDGER_FILENAME).stat().st_size > 65536
    write_job(site, state='ready')
    assert recover(site)['atlas']['atlas_status_digest'].startswith('sha256:')


@pytest.mark.parametrize('field', ['staging_reason', 'owner_observation_reason'])
@pytest.mark.parametrize('reason', ['capture_owner_purpose_mismatch', 'capture_original_birth_refused',
                                  'private token=secret', None])
def test_native_resume_preserves_only_a_bounded_refusal_code(site, monkeypatch, field, reason):
    from blueprint_pipeline import pubsub_handoff_listener as listener

    delivery = 'sha256:' + '8' * 64
    site.observation['producer_delivery']['delivery_key'] = delivery
    ledger = {'scene_id': 'site-fixture', 'capture_id': 'walkthrough-fixture',
              'status': 'failed_retryable', 'revision': 13, 'attempt_count': 5,
              'source_payload_sha256': '7' * 64, 'producer_delivery_key': delivery}
    (site.capture / listener.JOB_LEDGER_FILENAME).write_text(json.dumps(ledger))
    write_job(site, state='ready')
    reviewed = recover(site)['atlas']['atlas_status_digest']
    calls = []

    def native(payload, **kwargs):
        calls.append((payload, kwargs))
        return {'status': 'capture_source_membership_unavailable_retryable',
                'queue_disposition': 'retryable',
                field: str(CaptureStagingError(reason)) if reason is not None else None}

    monkeypatch.setenv('BLUEPRINT_PUBSUB_HANDOFF_PROVIDER', 'openai')
    before = tree_bytes(site)
    result = recovery.recover_selected_atlas_preview(
        {**site.selection, 'mode': 'preparation-resume', 'atlas_status_digest': reviewed},
        observation=site.observation, membership=site.membership,
        capture_root=site.capture, storage_root=site.storage,
        handoff_payload=b'original selected bytes',
        listener=SimpleNamespace(process_handoff_payload=native))
    assert result['status'] == 'blocked' and result['native_handler_invoked'] is True
    safe = (reason == 'capture_owner_purpose_mismatch'
            or (field == 'staging_reason' and reason == 'capture_original_birth_refused'))
    assert result.get(field) == (reason if safe else None)
    assert 'private token=secret' not in json.dumps(result)
    assert len(calls) == 1 and calls[0][0] == b'original selected bytes'
    assert calls[0][1]['expected_preparation_purpose'] == 'scene_preparation'
    assert tree_bytes(site) == before
