# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_retry.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/website_reconstruction_resume.py
"""The host owns the retry of a failed website preparation; Pub/Sub's five deliveries are no longer the bound.

The actual listener, lease, ledger and claim run on a temporary storage root;
run_e2e stands in for the orchestrator, and every clock the retry reads is
passed in. No provider call, no network.
"""
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_preparation_retry as retry
from blueprint_pipeline import website_reconstruction_resume as resume
from blueprint_pipeline.common import PipelineError, write_json
from blueprint_pipeline.website_preparation_terminal import GRANT_FILE, _parse_utc
from tests.test_website_preparation_status import prepared_source  # noqa: F401
from tests.test_website_reconstruction_resume import _job, waiting  # noqa: F401

TRANSIENT = "website_control_preparation-spend_http_503"


def _fail(error=TRANSIENT):
    def run_e2e(**_):
        raise PipelineError(f"clean_plate: {error}")
    return run_e2e


@pytest.fixture
def failed(waiting):  # noqa: F811 - imported pytest fixture
    """A website preparation whose (only) delivery failed for a retryable reason after its Atlas job ended."""
    root, owner, context, raw, _ = waiting
    _job(root, "ready")
    # The grant its delivery ran under: its expiry is the job's deadline (the content is not read here).
    write_json(root / GRANT_FILE, {"schema_version": "website_scene_sponsorship.v1"})
    with pytest.raises(PipelineError):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=_fail())
    ledger = listener._read_job_ledger(root)
    assert ledger["status"] == "failed_retryable" and retry.armed(ledger) is not None
    return root, raw, context


def _tick(root, run_e2e, now):
    return retry.reconcile_preparation_retries(listener, storage_root=root.parents[4], now=now, process_args={
        "provider": "openai", "run_evaluation_prep": False, "run_e2e": run_e2e})


def _due(root):
    return _parse_utc(retry.armed(listener._read_job_ledger(root))["next_attempt_at"])


def test_five_failures_in_a_row_are_retried_by_the_tick_with_growing_gaps_and_the_sixth_succeeds(failed):
    root, _, _ = failed
    calls = []

    def run_e2e(**kwargs):
        calls.append(kwargs)
        if len(calls) < 5:
            raise PipelineError(f"clean_plate: {TRANSIENT}")
        return {"status": "completed"}
    gaps = []
    for _ in range(5):
        ledger = listener._read_job_ledger(root)
        due = _due(root)
        gaps.append((due - _parse_utc(ledger["last_failed_at"])).total_seconds())
        # Not before it is due: nothing runs, nothing changes.
        assert _tick(root, run_e2e, due - timedelta(seconds=1))["redriven"] == 0
        assert listener._read_job_ledger(root) == ledger
        assert _tick(root, run_e2e, due)["redriven"] == 1
    assert len(calls) == 5 and all(call["resume_completed_stages"] is True for call in calls)
    assert gaps == [120, 240, 480, 960, 1920]
    ledger = listener._read_job_ledger(root)
    # Atlas wait, then the delivery's failure, four retried failures and the success.
    assert ledger["status"] == "completed" and ledger["attempt_count"] == 7 and retry.armed(ledger) is None
    assert [row["status"] for row in ledger["attempt_history"]][1:] == ["failed_retryable"] * 5 + ["completed"]
    assert _tick(root, run_e2e, datetime.now(timezone.utc) + timedelta(days=1))["redriven"] == 0
    assert retry.recovery_failures(root.parents[4])["captures"] == {}


def test_one_retry_policy_classifies_and_schedules():
    # An outage or a money refusal backs off to an hour with no count; the deadline ends it.
    assert [retry.retry_delay(retry.OUTAGE, n) // 60 for n in range(1, 9)] == [2, 4, 8, 16, 32, 60, 60, 60]
    assert retry.retry_delay("money", 40) == 3600
    # Anything unclassified: six timed retries, then none.
    assert [retry.retry_delay(retry.REFUSAL, n) for n in range(1, 8)] == [900, 3600, 10800, 21600, 43200, 86400, None]
    assert retry.retry_delay("privacy_rights", 1) is None
    assert retry.classify(TRANSIENT) == retry.OUTAGE  # the operation's name ("preparation-spend") is no reason
    assert retry.classify("meta_sam_http_429") == retry.OUTAGE
    assert retry.classify("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted") == "money"
    assert retry.classify("capture_rights_not_granted", "meta_sam_http_503") == "privacy_rights"
    assert retry.classify("meta_sam_submission_requires_reconciliation") == retry.REFUSAL


def test_the_listener_tick_retries_without_any_delivery(failed, monkeypatch):
    import google.cloud.pubsub_v1
    root, _, _ = failed
    ledger = listener._read_job_ledger(root)
    ledger[retry.FIELD]["next_attempt_at"] = "2026-01-01T00:00:00Z"
    write_json(root / listener.JOB_LEDGER_FILENAME, ledger)
    from blueprint_pipeline import run_e2e
    calls = []
    monkeypatch.setattr(run_e2e, "run_end_to_end", lambda **kw: calls.append(kw) or {"status": "completed"})
    monkeypatch.setattr(google.cloud.pubsub_v1, "SubscriberClient",
                        lambda: SimpleNamespace(pull=lambda **_: SimpleNamespace(received_messages=[])))
    listener.pull_and_process(subscription="projects/synthetic/subscriptions/handoff", storage_root=root.parents[4],
                              provider="openai", max_messages=1, run_evaluation_prep=False)
    assert len(calls) == 1 and listener._read_job_ledger(root)["status"] == "completed"


def test_a_retry_refused_before_its_claim_still_backs_off(failed, monkeypatch):
    root, _, _ = failed
    due = _due(root)
    unavailable = {"schema_version": "v1", "status": "capture_owner_observation_unavailable_retryable",
                   "queue_disposition": "retryable", "blockers": ["capture_owner_observation_unavailable"]}
    calls = []
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *_, **kw: calls.append(kw) or unavailable)
    assert _tick(root, _fail(), due)["redriven"] == 1
    assert calls[0]["expected_revision"] == listener._read_job_ledger(root)["revision"]
    later = _due(root)
    assert later == due + timedelta(seconds=retry.retry_delay(retry.OUTAGE, 2))
    assert _tick(root, _fail(), later - timedelta(seconds=1))["redriven"] == 0 and len(calls) == 1
    count = retry.recovery_failures(root.parents[4])["captures"]
    assert [row["consecutive_failures"] for row in count.values()] == [1]


def test_a_retry_claims_only_the_revision_it_scheduled(failed):
    root, raw, _ = failed
    ledger = listener._read_job_ledger(root)
    result = listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
        expected_revision=ledger["revision"] - 1, run_e2e=lambda **_: pytest.fail("a stale retry ran"))
    assert result["status"] == "revision_changed" and listener._read_job_ledger(root) == ledger


@pytest.mark.parametrize("error", [
    "meta_sam_authorization_missing",
    "privacy_postprocess_gate_failed",
    "capture_rights_not_granted",
    "website_capture_consent_withdrawn",
    "dataset_license_revoked",
])
def test_a_privacy_rights_or_consent_failure_is_never_retried(failed, error):
    root, raw, _ = failed
    with pytest.raises(PipelineError):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=_fail(error))
    ledger = listener._read_job_ledger(root)
    assert ledger["status"] == "failed_retryable" and retry.armed(ledger) is None
    later = datetime.now(timezone.utc) + timedelta(days=2)
    assert _tick(root, lambda **_: pytest.fail("retried a hard stop"), later)["redriven"] == 0
    assert listener._read_job_ledger(root) == ledger


@pytest.mark.parametrize("error", [
    "website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted",
    "account_monthly_spend_limit_reached",
])
def test_a_money_refusal_waits_and_asks_again(failed, error):
    root, raw, _ = failed
    with pytest.raises(PipelineError):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=_fail(error))
    record = retry.armed(listener._read_job_ledger(root))
    assert record["kind"] == "money" and record["retry_count"] == 2


def test_an_unclassified_failure_gets_six_timed_retries_then_is_left_as_it_is(failed):
    root, raw, _ = failed
    error = "clean_plate: meta_sam_submission_requires_reconciliation"
    with pytest.raises(PipelineError):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
                                         run_e2e=_fail("meta_sam_submission_requires_reconciliation"))
    gaps = []
    while (record := retry.armed(listener._read_job_ledger(root))) is not None:
        ledger = listener._read_job_ledger(root)
        gaps.append((_due(root) - _parse_utc(ledger["last_failed_at"])).total_seconds())
        assert record["kind"] == retry.REFUSAL and ledger["last_error"] == error
        assert _tick(root, _fail("meta_sam_submission_requires_reconciliation"), _due(root))["redriven"] == 1
    # The delivery's failure followed the outage before it (count 2), then five more retries; then none.
    assert gaps == [3600, 10800, 21600, 43200, 86400]
    ledger = listener._read_job_ledger(root)
    assert ledger["status"] == "failed_retryable" and retry.armed(ledger) is None
    later = datetime.now(timezone.utc) + timedelta(days=30)
    assert _tick(root, lambda **_: pytest.fail("retried past the schedule"), later) == {
        "reclaimed": 0, "adopted": 0, "waiting": 0, "redriven": 0, "failed": 0}


def test_an_authority_ending_is_terminal_and_never_retried(failed):
    root, raw, _ = failed
    result = listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
        run_e2e=_fail("website_control_scene-sponsorship_http_409:consent_expired"))
    assert result["status"] == listener.TERMINAL_AUTHORITY_STATUS
    ledger = listener._read_job_ledger(root)
    assert _tick(root, lambda **_: pytest.fail("retried an ended job"),
                 datetime.now(timezone.utc) + timedelta(days=2))["redriven"] == 0
    assert listener._read_job_ledger(root) == ledger


def _dead_attempt(root, raw, *, owner, started_ago, lease_seconds=listener.DEFAULT_JOB_LEASE_SECONDS):
    failed = listener._read_job_ledger(root)
    status, dead = listener._claim_job_lease(root, scene_id=failed["scene_id"], capture_id=failed["capture_id"],
        owner=owner, lease_seconds=lease_seconds,
        now=datetime.now(timezone.utc) - started_ago, payload_sha256=listener.payload_sha256(raw),
        producer_delivery_key=failed["producer_delivery_key"])
    assert status == "claimed"
    return dead


def _gone_pid():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_an_attempt_whose_process_died_is_reclaimed_and_retried_within_30_minutes(failed):
    root, raw, _ = failed
    dead = _dead_attempt(root, raw, owner=f"{listener.socket.gethostname()}:{_gone_pid()}:killed",
                         started_ago=timedelta(minutes=20))
    now = datetime.now(timezone.utc)
    calls = []
    assert _tick(root, lambda **kw: calls.append(kw) or {"status": "completed"}, now)["reclaimed"] == 1
    ledger = listener._read_job_ledger(root)
    assert ledger["status"] == "failed_retryable" and ledger["lease_token"] is None and not calls
    row = ledger["attempt_history"][-1]
    assert (row["attempt_number"], row["stage"], row["outcome"]) == (dead["attempt_count"], "lease_expired", "unknown")
    assert row["previous_lease_owner"] == dead["lease_owner"]
    due = _due(root)
    # Lease expiry (15 min after the last heartbeat) + this tick + the delay is under 30 minutes.
    assert due - now <= timedelta(minutes=10)
    assert _tick(root, lambda **kw: calls.append(kw) or {"status": "completed"}, due)["redriven"] == 1
    assert len(calls) == 1 and listener._read_job_ledger(root)["status"] == "completed"


def _process_started():
    boot = retry._boot_epoch()
    ticks = float(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[19])
    return datetime.fromtimestamp(boot + ticks / listener.os.sysconf("SC_CLK_TCK"), timezone.utc)


@pytest.mark.parametrize("owner", ["live-current-process", "dead-process-live-lease", "other-host:1:x"])
def test_a_live_or_unproven_lease_is_never_touched(failed, owner):
    root, raw, _ = failed
    host = listener.socket.gethostname()
    if owner == "live-current-process":
        # Its lease ran out (a hung heartbeat), but the process that took it is still alive.
        _dead_attempt(root, raw, owner=f"{host}:{listener.os.getpid()}:live", lease_seconds=1,
                      started_ago=datetime.now(timezone.utc) - _process_started() - timedelta(seconds=1))
    elif owner == "dead-process-live-lease":
        _dead_attempt(root, raw, owner=f"{host}:{_gone_pid()}:dead", started_ago=timedelta(minutes=1))
    else:
        _dead_attempt(root, raw, owner=owner, started_ago=timedelta(minutes=20))
    ledger = listener._read_job_ledger(root)
    assert _tick(root, lambda **_: pytest.fail("took a live lease"), datetime.now(timezone.utc))["reclaimed"] == 0
    assert listener._read_job_ledger(root) == ledger


def test_a_reused_pid_is_not_the_lease_owner():
    # This process started long after a lease taken in 2001: whatever it is, it does not hold that lease.
    owner = f"{listener.socket.gethostname()}:{listener.os.getpid()}:old"
    assert retry.owner_gone(owner, datetime(2001, 1, 1, tzinfo=timezone.utc)) is True
    assert retry.owner_gone(owner, datetime.now(timezone.utc)) is False


def test_a_job_failed_before_this_release_is_adopted_from_its_kept_bytes(failed):
    root, _, _ = failed
    ledger = listener._read_job_ledger(root)
    ledger.pop(retry.FIELD)
    write_json(root / listener.JOB_LEDGER_FILENAME, ledger)
    (root / retry.PAYLOAD_FILE).unlink()  # Only the Atlas wait's own record kept the bytes then.
    calls = []
    counts = _tick(root, lambda **kw: calls.append(kw) or {"status": "completed"}, datetime.now(timezone.utc))
    assert counts["adopted"] == 1 and counts["redriven"] == 1 and len(calls) == 1


def test_a_job_failed_before_this_release_without_kept_bytes_is_counted_not_guessed(failed):
    root, _, _ = failed
    ledger = listener._read_job_ledger(root)
    ledger.pop(retry.FIELD)
    ledger["attempt_history"] = [{k: v for k, v in row.items() if k != resume.FIELD}
                                 for row in ledger["attempt_history"]]
    write_json(root / listener.JOB_LEDGER_FILENAME, ledger)
    (root / retry.PAYLOAD_FILE).unlink()
    counts = _tick(root, lambda **_: pytest.fail("ran without its bytes"), datetime.now(timezone.utc))
    assert counts["failed"] == 1 and counts["redriven"] == 0
    row, = retry.recovery_failures(root.parents[4])["captures"].values()
    assert row["capture_id"] == root.name and "payload_unavailable" in row["last_error"]


def test_a_replay_after_atlas_that_fails_is_retried_like_any_failure(waiting):  # noqa: F811
    root = waiting[0]
    _job(root, "ready")
    process_args = {"provider": "openai", "run_evaluation_prep": False, "run_e2e": _fail()}
    resume.reconcile_waiting_reconstructions(listener, storage_root=root.parents[4], process_args=process_args)
    ledger = listener._read_job_ledger(root)
    assert ledger["status"] == "failed_retryable" and not resume.armed(ledger)
    assert retry.armed(ledger)["retry_count"] == 1
    calls = []
    assert _tick(root, lambda **kw: calls.append(kw) or {"status": "completed"}, _due(root))["redriven"] == 1
    assert len(calls) == 1 and listener._read_job_ledger(root)["status"] == "completed"


def test_an_atlas_wait_is_left_to_its_own_replay(waiting):  # noqa: F811
    root = waiting[0]
    ledger = listener._read_job_ledger(root)
    assert resume.armed(ledger) and retry.armed(ledger) is None
    assert _tick(root, lambda **_: pytest.fail("retried a wait"),
                 datetime.now(timezone.utc) + timedelta(days=1))["redriven"] == 0


# --- a retry buys nothing twice -------------------------------------------------------------------

KINDS = ("gemini_plan", "sam_upload", "sam_prompt", "ranking")


def _paid_stages(root, context, monkeypatch, tmp_path):
    """run_e2e that makes each paid call through its real intent/receipt code, with counted providers."""
    from io import BytesIO

    from blueprint_pipeline import meta_sam31 as sam
    from blueprint_pipeline import website_gemini_receipts as receipts
    from blueprint_pipeline.paid_resource_admission import (
        PAID_LANE_ADMISSION_SCHEMA_VERSION, require_paid_resource_admission,
    )
    from tests.test_meta_sam31 import inputs, response

    bought = {kind: 0 for kind in KINDS}
    reserved = []

    def reserve(**kwargs):
        reserved.append(kwargs["binding_digest"])
        admission = {"schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION, "resource_class": kwargs["resource_class"],
                     "status": "admitted", "allocation_binding_digest": kwargs["binding_digest"], "blockers": []}
        return admission, require_paid_resource_admission(admission, resource_class=kwargs["resource_class"],
                                                          expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    monkeypatch.setattr(receipts, "reserve_website_preparation_spend", reserve)
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    sam_args = inputs(tmp_path)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"walkthrough")

    def gemini(kind):
        receipts.retained_gemini_call(output_root=root / "paid" / kind, binding={"kind": kind},
            task_context=context, maximum_cost_usd=0.5, preflight=lambda: None,
            invoke=lambda: bought.__setitem__(kind, bought[kind] + 1) or {"status": "completed"})

    def upload_opener(request, **_):
        bought["sam_upload"] += 1
        return BytesIO(json.dumps({"id": "file-retained"}).encode())

    def prompt_opener(request, **_):
        bought["sam_prompt"] += 1
        return BytesIO(json.dumps(response()).encode())

    steps = {"gemini_plan": lambda: gemini("gemini_plan"),
             "sam_upload": lambda: sam._upload_video(clip=clip, root=tmp_path, token="t", opener=upload_opener),
             "sam_prompt": lambda: sam.run_meta_sam31(**sam_args, opener=prompt_opener),
             "ranking": lambda: gemini("ranking")}
    runs = []

    def make(fail_after):
        def run_e2e(**_):
            runs.append(fail_after)
            for kind in KINDS:
                steps[kind]()
                if kind == fail_after and len(runs) == 1:
                    raise PipelineError(f"clean_plate: {TRANSIENT}")
            return {"status": "completed"}
        return run_e2e
    return make, bought, reserved


@pytest.mark.parametrize("fail_after", KINDS)
def test_a_retry_after_each_paid_call_buys_nothing_twice(failed, monkeypatch, tmp_path, fail_after):
    root, raw, context = failed
    make, bought, reserved = _paid_stages(root, context, monkeypatch, tmp_path)
    run_e2e = make(fail_after)
    with pytest.raises(PipelineError):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=run_e2e)
    first = dict(bought)
    assert all(first[kind] == 1 for kind in KINDS[:KINDS.index(fail_after) + 1])
    assert _tick(root, run_e2e, _due(root))["redriven"] == 1
    assert listener._read_job_ledger(root)["status"] == "completed"
    # Each paid call was bought exactly once across the failed attempt and its retry.
    assert bought == {kind: 1 for kind in KINDS}
    assert len(reserved) == len(set(reserved)) == 2  # one reservation per Gemini decision, never a second


# --- a failure before the lease -------------------------------------------------------------------

class _Subscriber:
    """One delivery, its last before Pub/Sub dead-letters it, then nothing."""

    def __init__(self, raw):
        self.messages = [SimpleNamespace(ack_id="ack-1", delivery_attempt=5, message=SimpleNamespace(
            data=raw, message_id="m-1", attributes={}))]
        self.acknowledged = []

    def pull(self, **_):
        messages, self.messages = self.messages, []
        return SimpleNamespace(received_messages=messages)

    def acknowledge(self, request):
        self.acknowledged.extend(request["ack_ids"])

    def modify_ack_deadline(self, **_):
        return None


@pytest.mark.parametrize("prepared_source", [None], indirect=True)
def test_a_failure_before_the_lease_is_retried_after_pubsub_has_given_up(waiting, monkeypatch):  # noqa: F811
    import google.cloud.pubsub_v1

    from blueprint_pipeline import capture_original_owner_observer as observer
    root, _, _, raw, _ = waiting
    _job(root, "ready")
    # The very first delivery: nothing has claimed this capture yet.
    (root / listener.JOB_LEDGER_FILENAME).unlink()
    (root / retry.PAYLOAD_FILE).unlink()
    observe = observer.load_original_owner_observation
    monkeypatch.setattr(observer, "load_original_owner_observation",
                        lambda **_: (_ for _ in ()).throw(TimeoutError("webapp down")))
    subscriber = _Subscriber(raw)
    monkeypatch.setattr(google.cloud.pubsub_v1, "SubscriberClient", lambda: subscriber)
    listener.pull_and_process(subscription="projects/synthetic/subscriptions/handoff", storage_root=root.parents[4],
                              provider="openai", max_messages=1, run_evaluation_prep=False)
    assert subscriber.acknowledged == [] and not (root / listener.JOB_LEDGER_FILENAME).exists()
    entry, = (root.parents[4] / retry.UNCLAIMED_DIR).glob("*.json")
    due = _parse_utc(json.loads(entry.read_text())["next_attempt_at"])
    # Pub/Sub has given up; the WebApp is back. Still down at the first retry, it backs off.
    assert _tick(root, lambda **_: pytest.fail("ran unclaimed"), due)["redriven"] == 1
    later = _parse_utc(json.loads(entry.read_text())["next_attempt_at"])
    assert later - due == timedelta(seconds=retry.retry_delay(retry.OUTAGE, 2))
    monkeypatch.setattr(observer, "load_original_owner_observation", observe)
    calls = []
    assert _tick(root, lambda **kw: calls.append(kw) or {"status": "completed"}, later)["redriven"] == 1
    assert len(calls) == 1 and listener._read_job_ledger(root)["status"] == "completed"
    assert not entry.exists()


def test_a_delivery_refused_for_rights_before_the_lease_is_not_kept(waiting, monkeypatch):  # noqa: F811
    root, _, _, raw, _ = waiting
    (root / listener.JOB_LEDGER_FILENAME).unlink()
    refused = {"status": "capture_owner_observation_unavailable_retryable", "queue_disposition": "retryable",
               "owner_observation_reason": "capture_owner_rights_not_admitted"}
    retry.spool_unclaimed(listener, storage_root=root.parents[4], payload=raw,
                          payload_digest=listener.payload_sha256(raw), result=refused)
    assert not (root.parents[4] / retry.UNCLAIMED_DIR).exists()


def test_a_failing_atlas_replay_is_logged_at_warning_and_counted(waiting, monkeypatch, caplog):  # noqa: F811
    from blueprint_pipeline import capture_original_owner_observer as observer
    root = waiting[0]
    _job(root, "ready")
    monkeypatch.setattr(observer, "load_original_owner_observation",
                        lambda **_: (_ for _ in ()).throw(TimeoutError("webapp down")))
    for _ in range(2):
        resume.reconcile_waiting_reconstructions(listener, storage_root=root.parents[4],
                                                 process_args={"provider": "openai"})
    row, = retry.recovery_failures(root.parents[4])["captures"].values()
    assert (row["capture_id"], row["recovery"], row["consecutive_failures"]) == (
        root.name, "reconstruction_resume", 2)
    assert any(record.levelname == "WARNING" and record.getMessage() == "pubsub_handoff.preparation_recovery_failed"
               for record in caplog.records)


# --- no fresh retry capacity for a call whose outcome is unknown ---------------------------------

@pytest.mark.parametrize("call", ["gemini", "sam"])
def test_an_unanswered_paid_call_is_sent_again_only_under_a_new_reservation(failed, monkeypatch, tmp_path, call):
    """An intent with no receipt is counted spent. Gemini: one resend per new reservation, until the ceiling
    refuses one (a money wait, nothing sent). SAM: one resend (owner rule 2026-11-07), then it stops."""
    from blueprint_pipeline import meta_sam31 as sam
    from blueprint_pipeline import website_gemini_receipts as receipts
    from tests.test_meta_sam31 import inputs
    root, raw, context = failed
    from blueprint_pipeline import website_task_context
    from blueprint_pipeline.paid_resource_admission import build_paid_lane_admission, require_paid_resource_admission
    _, _, reserved = _paid_stages(root, context, monkeypatch, tmp_path)
    sam_args = {**inputs(tmp_path), "admission": {}, "admission_grant": None, "task_context": {"capture_id": "c"}}

    def reserve_sam(**kwargs):
        reserved.append(kwargs["binding_digest"])
        value = {**build_paid_lane_admission(resource_class="evaluator_api"), "maximum_cost_usd": 1.0,
                 "allocation_binding_digest": kwargs["binding_digest"], "external_disclosure_allowed": True}
        return value, require_paid_resource_admission(value, resource_class="evaluator_api",
                                                      expected_schema_version="paid_lane_admission.v1")
    monkeypatch.setattr(website_task_context, "reserve_website_sam_spend", reserve_sam)
    dispatched = {"gemini": 0, "sam": 0}

    def unanswered_gemini():
        dispatched["gemini"] += 1
        raise TimeoutError("no answer: the provider may have charged")

    def gemini_run(**_):
        receipts.retained_gemini_call(output_root=root / "paid" / "gemini_plan", binding={"kind": "gemini_plan"},
            task_context=context, maximum_cost_usd=0.5, preflight=lambda: None, invoke=unanswered_gemini)
        return {"status": "completed"}

    def unanswered_sam(request, **_):
        dispatched["sam"] += 1
        raise TimeoutError("no answer: the provider may have charged")

    def sam_run(**_):
        sam.run_meta_sam31(**sam_args, opener=unanswered_sam)
        return {"status": "completed"}

    admit = receipts.reserve_website_preparation_spend

    def ceiling(**kwargs):
        if len(reserved) >= 2:
            reserved.append("refused")
            raise ValueError("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted")
        return admit(**kwargs)
    monkeypatch.setattr(receipts, "reserve_website_preparation_spend", ceiling)
    run_e2e = gemini_run if call == "gemini" else sam_run
    with pytest.raises(TimeoutError):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=run_e2e)
    errors = []
    for _ in range(4):
        _tick(root, run_e2e, _due(root))
        errors.append(listener._read_job_ledger(root)["last_error"])
    # Exactly one dispatch per admitted reservation, never a dispatch without one.
    assert dispatched[call] == 2 and len([row for row in reserved if row != "refused"]) == 2
    if call == "gemini":
        assert "budget_exhausted" in errors[-1]
        assert retry.armed(listener._read_job_ledger(root))["kind"] == "money"  # waits and asks again
    else:
        assert all("requires_reconciliation" in error for error in errors[-3:])


def test_a_job_with_no_grant_has_no_deadline_so_an_outage_takes_the_six_timed_retries(failed):
    root, raw, _ = failed
    (root / GRANT_FILE).unlink()
    with pytest.raises(PipelineError):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=_fail())
    ledger = listener._read_job_ledger(root)
    record = retry.armed(ledger)
    # Two outages in a row: 4 minutes with a deadline; the timed schedule's second gap without one.
    assert record["kind"] == retry.OUTAGE and record["retry_count"] == 2
    assert (_due(root) - _parse_utc(ledger["last_failed_at"])).total_seconds() == 3600
    assert retry.retry_delay(retry.OUTAGE, 7, deadline=False) is None
    assert retry.retry_delay(retry.OUTAGE, 7) == 3600


def _deterministic(code="atlas_mesh_not_registered_to_splat"):
    from blueprint_pipeline.deterministic_refusal import DeterministicRefusal

    def run_e2e(**_):
        refusal = DeterministicRefusal(code, step="atlas_mesh_registration",
                                       inputs={"source_glb_sha256": "sha256:" + "1" * 64})
        raise PipelineError(f"atlas: {code}") from refusal
    return run_e2e


def test_a_deterministic_refusal_is_not_retried_until_a_release_decides_its_step_differently(failed, monkeypatch):
    from blueprint_pipeline import atlas_mesh_registration
    root, raw, _ = failed
    with pytest.raises(PipelineError):
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=_deterministic())
    record = retry.armed(listener._read_job_ledger(root))
    assert record["kind"] == retry.DETERMINISTIC and record["deterministic_refusal"]["step"] == "atlas_mesh_registration"
    # Same inputs, same code: no run at any time, so nothing is bought or redone.
    later = datetime.now(timezone.utc) + timedelta(days=30)
    for days in (0, 1, 3, 30):
        counts = _tick(root, lambda **_: pytest.fail("a deterministic refusal was re-run"), later + timedelta(days=days))
        assert counts["redriven"] == 0 and counts["waiting"] == 1
    # A release that changes how the step decides runs it once; refused again, it waits again.
    monkeypatch.setattr(atlas_mesh_registration, "REGISTRATION_REVISION", "atlas_mesh_registration.r2")
    runs = []

    def again(**kwargs):
        runs.append(kwargs)
        return _deterministic()(**kwargs)
    assert _tick(root, again, later)["redriven"] == 1 and len(runs) == 1
    assert retry.armed(listener._read_job_ledger(root))["deterministic_refusal"]["revision"] == "atlas_mesh_registration.r2"
    assert _tick(root, again, later + timedelta(days=2))["redriven"] == 0 and len(runs) == 1
