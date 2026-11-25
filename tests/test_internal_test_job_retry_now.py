"""The door's ``test-job-retry-now``: one internal test job's scheduled retries brought forward to now. Fakes only.

The records are shaped as the live Austin test job's were on 2026-11-24 (read through the door): an Atlas preview
processing at ``source_pose`` with three refusal retries and its next attempt three hours out, and a preparation
``failed_retryable`` with its retry record's next attempt a day out.
"""
# Covers: src/blueprint_pipeline/internal_test_job_retry_now.py
# Covers: deploy/operator-door/operator_door/requests.py
# Covers: deploy/operator-door/operator_door/spool_runner.py
# Covers: deploy/operator-door/door-test-job-retry-now.sh
# Covers: scripts/operator_door.py
# Covers: scripts/test_jobs.py
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import internal_test_job_retry_now as retry
from tests.test_operator_door_scripts import (
    SETPRIV_STUB,
    _run,
    _write_stub,
    env,  # noqa: F401 - a fixture
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "operator-door"))
# Bind the door package now (see tests/test_internal_test_jobs.py): the client is also named ``operator_door``.
if not hasattr(sys.modules.get("operator_door"), "__path__"):
    sys.modules.pop("operator_door", None)
importlib.import_module("operator_door")

JOB = "itest-" + "c" * 40
CAPTURE = f"walkthrough-{JOB}"
TEST_JOB_REQUEST = "20261124T054046Z-test-job-6d0f23d7"
THIS = "20261124T184000Z-test-job-retry-now-0000abcd"
RELEASE, NEXT_RELEASE = "d" * 40, "e" * 40
NOW = 1_795_545_600  # 2026-11-24T18:40:00Z
BODY = {"kind": "test-job-retry-now", "job_id": JOB, "reason": "the fix for the scene review is live"}


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def _atlas(**change):
    return {"schema_version": "website_atlas_preview_job.v1", "request_id": JOB, "scene_id": f"site-{JOB}",
            "capture_id": CAPTURE, "source_video": "raw/walkthrough.mp4", "routing_mode": "consented",
            "enqueued_at_epoch": NOW - 46_663, "state": "processing", "reason": None, "stage": "source_pose",
            "revision": 3, "revised_at_epoch": NOW - 28_412, "updated_at_epoch": NOW - 25_722,
            "next_attempt_at_epoch": NOW + 8_840, "retry_at_epoch": None, "transient_failures": 0,
            "refusal_retries": 3, "refusal_retries_total": 3, "last_refusal_reason": "atlas_preview_step_interrupted",
            "retries": [{"reason": "release_reopened", "at_epoch": NOW - 28_412,
                         "stopped_on": "atlas_scene_planner_coverage_or_privacy_invalid"}],
            "release_reopened": "f" * 40, "archive": None, "delivered_revision": 3, "delivery_attempts": 0,
            "delivery_next_at_epoch": 0, "last_tick_epoch": NOW - 100,
            "provider_cleanup": {"state": "pending", "attempts": 0, "deleted": [], "remaining": [],
                                 "operations_running": [], "next_attempt_at_epoch": 0}, **change}


def _ledger(error="clean_plate: TimeoutExpired", record=None, **change):
    key, digest = "sha256:" + "b" * 64, "a" * 64
    return {"schema_version": "pipeline_job_ledger.v1", "revision": 42, "status": "failed_retryable",
            "scene_id": f"site-{JOB}", "capture_id": CAPTURE, "attempt_count": 6,
            "attempt_history": [{"attempt_number": 6, "status": "failed_retryable", "stage": "run_e2e",
                                 "failed_at": _iso(NOW - 42_000), "error": error}],
            "last_error": error, "last_failed_at": _iso(NOW - 42_000), "last_attempt_started_at": _iso(NOW - 42_050),
            "lease_owner": None, "lease_token": None, "source_payload_sha256": digest, "producer_delivery_key": key,
            "preparation_retry": {"schema_version": "website_preparation_retry.v1", "source_payload_sha256": digest,
                                  "producer_delivery_key": key, "error": error, "kind": "refusal", "retry_count": 6,
                                  "next_attempt_at": _iso(NOW + 44_400), **(record or {})}, **change}


class World:
    """One test job as the host keeps it: the door's ledger row and recorded result, its Atlas job and its ledger."""

    def __init__(self, tmp_path: Path, *, atlas=None, ledger=None, row=None, recorded=None):
        self.ledger_root, self.results = tmp_path / "internal-test-jobs", tmp_path / "results"
        self.handoffs = tmp_path / "handoffs"
        self.capture = self.handoffs / "bucket" / "scenes" / f"site-{JOB}" / "captures" / CAPTURE
        for directory in (self.ledger_root, self.results, self.capture / "pipeline" / "atlas"):
            directory.mkdir(parents=True)
        (self.ledger_root / ".lock").touch()
        (self.capture / "pipeline" / "atlas" / "preview.lock").touch()
        (self.capture / ".pipeline_job_ledger.json.lock").touch()
        self.atlas_path = self.capture / "pipeline" / "atlas" / "preview_job.json"
        self.ledger_path = self.capture / "pipeline_job_ledger.json"
        key = "stress-20261124b:austin-apartment-floors"
        self.row_path = self.ledger_root / (hashlib.sha256(key.encode()).hexdigest()[:32] + ".json")
        self.write(self.atlas_path, _atlas() if atlas is None else atlas)
        self.write(self.ledger_path, _ledger() if ledger is None else ledger)
        self.write(self.row_path, {"schema_version": "internal_test_job_door_row.v1", "label": "austin-apartment-floors",
                                   "footage_id": "owner-austin-apartment-20261123", "idempotency_key": key,
                                   "description_sha256": "sha256:" + "0" * 64, "door_request_id": TEST_JOB_REQUEST,
                                   "state": "submitted", "requested_at_epoch": NOW - 46_753, "job_id": JOB,
                                   "capture_id": CAPTURE, "submitted_at_epoch": NOW - 46_744, **(row or {})})
        self.write(self.results / f"{TEST_JOB_REQUEST}.test-job.json",
                   {"capture_id": CAPTURE, "door_outcome": "submitted", "job_id": JOB,
                    "label": "austin-apartment-floors", **(recorded or {})})

    @staticmethod
    def write(path: Path, value) -> None:
        path.write_text(json.dumps(value))

    @staticmethod
    def read(path: Path):
        return json.loads(path.read_text())

    def ask(self, *, release=RELEASE, now=NOW, request_id=THIS, body=None):
        return retry.run(body or BODY, door_request_id=request_id, requested_by="cloud-agent",
                         ledger_root=self.ledger_root, handoff_root=self.handoffs, door_results=self.results,
                         release=lambda: release, now=lambda: now)

    def records(self):
        return self.read(self.atlas_path), self.read(self.ledger_path)


# ------------------------------------------------------------------------------------------- the door's request

@pytest.mark.parametrize(("change", "code"), [
    ({}, None),
    ({"job_id": "site-capture-1234"}, "test_job_retry_now_job_invalid"),          # a site's job
    ({"job_id": "8d3f0c2a-9e7b-4c1d-a0f5-6b2e9d4c7a31"}, "test_job_retry_now_job_invalid"),
    ({"job_id": "itest-" + "C" * 40}, "test_job_retry_now_job_invalid"),
    ({"job_id": "itest-" + "c" * 39}, "test_job_retry_now_job_invalid"),
    ({"job_id": "itest-" + "c" * 40 + "/../x"}, "test_job_retry_now_job_invalid"),
    ({"job_id": None}, "test_job_retry_now_job_invalid"),
    ({"reason": ""}, "test_job_retry_now_reason_invalid"),
    ({"reason": " padded"}, "test_job_retry_now_reason_invalid"),
    ({"reason": "two\nlines"}, "test_job_retry_now_reason_invalid"),
    ({"reason": "x" * 501}, "test_job_retry_now_reason_invalid"),
    ({"reason": None}, "test_job_retry_now_reason_invalid"),
    ({"refusal_retries": 0}, "request_key_unknown:refusal_retries"),               # it carries no count, ...
    ({"spend_usd": 5}, "request_key_unknown:spend_usd"),                           # ... no money ...
    ({"next_attempt_at_epoch": 0}, "request_key_unknown:next_attempt_at_epoch"),   # ... and no time of its own
])
def test_the_door_admits_only_a_test_jobs_id_and_a_reason(change, code):
    from operator_door.requests import (
        RequestRefused,
        required_scope,
        validate_request,
        validate_request_id,
    )

    assert required_scope("test-job-retry-now") == "operate" and validate_request_id(THIS) == THIS
    body = {**BODY, **change}
    if code is None:
        assert validate_request(body) == BODY
        return
    with pytest.raises(RequestRefused, match=f"^{code}$"):
        validate_request(body)


def test_the_door_runs_it_as_the_records_owner_with_no_network_writing_only_the_jobs_scene(tmp_path, monkeypatch):
    from operator_door import spool_runner
    from operator_door.config import DoorConfig
    from operator_door.requests import RequestRefused, validate_request

    monkeypatch.setattr(spool_runner, "_PUBSUB_HANDOFFS", str(tmp_path))
    (tmp_path / "bucket" / "scenes" / f"site-{JOB}").mkdir(parents=True)
    (tmp_path / "bucket" / "scenes" / "site-other-job").mkdir(parents=True)
    config, request = DoorConfig(), validate_request(BODY)
    launch = spool_runner._LAUNCHES["test-job-retry-now"]
    properties = launch.properties(config, request)
    assert launch.script == "door-test-job-retry-now.sh" and JOB not in launch.label(request)
    assert {"PrivateNetwork=yes", "RestrictAddressFamilies=AF_UNIX", "ProtectSystem=strict"} <= set(properties)
    hidden = next(item for item in properties if item.startswith("InaccessiblePaths=")).split("=", 1)[1].split()
    assert {"-" + path for path in config.hidden_paths} == set(hidden)
    writable = next(item for item in properties if item.startswith("ReadWritePaths=")).split("=", 1)[1].split()
    assert writable == [f"-{tmp_path}/bucket/scenes/site-{JOB}",
                        "/var/lib/blueprint/pipeline-control-plane/internal-test-jobs",
                        "/var/lib/blueprint-operator-door/requests/results"]
    arguments = spool_runner._script_env(config, THIS, request, "cloud-agent")
    assert "--setenv=DOOR_REQUESTED_BY=cloud-agent" in arguments
    sent = next(item for item in arguments if item.startswith("--setenv=DOOR_TEST_JOB_RETRY_NOW_REQUEST="))
    assert json.loads(sent.split("=", 2)[2]) == BODY
    with pytest.raises(RequestRefused, match="^test_job_retry_now_caller_invalid$"):
        spool_runner._script_env(config, THIS, request, "")


# ------------------------------------------------------------------------------------------- timing only

def test_both_retries_come_due_now_and_nothing_else_on_either_record_changes(tmp_path):
    from blueprint_pipeline.website_preparation_retry import _due, armed

    world = World(tmp_path)
    atlas_before, ledger_before = world.records()
    answer = world.ask()
    atlas, ledger = world.records()
    assert answer["door_outcome"] == "brought_forward" and answer["records"]["atlas_preview"]["result"] == \
        answer["records"]["preparation_retry"]["result"] == "brought_forward"
    # The Atlas job: only its next attempt's time. Its refusal counts, reopen rows and revision are as they were.
    assert atlas == {**atlas_before, "next_attempt_at_epoch": NOW}
    assert max(atlas["next_attempt_at_epoch"], atlas.get("retry_at_epoch") or 0) <= NOW  # the tick's own gate
    # The ledger: only the retry record's time, committed at the next revision as the listener commits.
    assert ledger == {**ledger_before, "revision": 43,
                      "preparation_retry": {**ledger_before["preparation_retry"], "next_attempt_at": _iso(NOW)}}
    record = armed(ledger)  # the listener still reads the same record, with its count, as due
    assert record is not None and record["retry_count"] == 6 and _due(world.capture, ledger, record).timestamp() <= NOW
    assert answer["records"]["preparation_retry"]["listener_due_at"] == _iso(NOW)
    row = world.read(world.row_path)
    assert row["retry_now"] == [{"release": RELEASE, "door_request_id": THIS, "requested_by": "cloud-agent",
                                 "reason": BODY["reason"], "at_epoch": NOW, "outcome": "brought_forward",
                                 "records": {"atlas_preview": "brought_forward", "preparation_retry": "brought_forward"}}]
    assert (answer["requested_by"], answer["reason"], answer["release"]) == ("cloud-agent", BODY["reason"], RELEASE)


@pytest.mark.parametrize(("atlas", "ledger", "results"), [
    (_atlas(next_attempt_at_epoch=NOW - 5), None, ("already_due", "brought_forward")),
    (_atlas(state="needs_input", reason="atlas_some_stop"), None, ("not_waiting_on_a_retry", "brought_forward")),
    (_atlas(retry_at_epoch=NOW + 600), None, ("not_waiting_on_a_retry", "brought_forward")),
    (None, _ledger(record={"next_attempt_at": _iso(NOW - 5)}), ("brought_forward", "already_due")),
    (None, _ledger(record={"kind": "retries_spent", "release_commit": RELEASE}),
     ("brought_forward", "waits_for_a_release")),
    (None, _ledger(record={"kind": "deterministic"}), ("brought_forward", "waits_for_a_release")),
    (None, _ledger(status="completed"), ("brought_forward", "not_waiting_on_a_retry")),
    (_atlas(next_attempt_at_epoch=NOW), _ledger(record={"next_attempt_at": _iso(NOW)}), ("already_due", "already_due")),
])
def test_a_record_with_no_timed_retry_ahead_is_left_as_it_is(tmp_path, atlas, ledger, results):
    world = World(tmp_path, atlas=atlas, ledger=ledger)
    before = world.records()
    answer = world.ask()
    assert (answer["records"]["atlas_preview"]["result"], answer["records"]["preparation_retry"]["result"]) == results
    atlas_after, ledger_after = world.records()
    assert (atlas_after == before[0]) is (results[0] != "brought_forward")
    assert (ledger_after == before[1]) is (results[1] != "brought_forward")
    assert answer["door_outcome"] == ("brought_forward" if "brought_forward" in results else "unchanged")


# ------------------------------------------------------------------------------------------- only a test job

@pytest.mark.parametrize(("world", "body", "code"), [
    ({"row": {"state": "submitting"}}, None, "test_job_retry_now_not_in_door_ledger"),
    ({"row": {"job_id": "itest-" + "9" * 40}}, None, "test_job_retry_now_not_in_door_ledger"),  # no row names it
    ({"row": {"door_request_id": "20261124T054046Z-test-job-ffffffff"}}, None,
     "test_job_retry_now_not_the_doors_test_job"),                                  # no result of that request
    ({"row": {"door_request_id": "../../etc/x"}}, None, "test_job_retry_now_not_the_doors_test_job"),
    ({"recorded": {"job_id": "itest-" + "9" * 40}}, None, "test_job_retry_now_not_the_doors_test_job"),
    ({"recorded": {"capture_id": "walkthrough-other"}}, None, "test_job_retry_now_not_the_doors_test_job"),
    ({"recorded": {"door_outcome": "refused"}}, None, "test_job_retry_now_not_the_doors_test_job"),
    ({}, {**BODY, "job_id": "site-capture-1234"}, "test_job_retry_now_not_a_test_job"),
    ({}, {**BODY, "reason": ""}, "test_job_retry_now_reason_invalid"),
    ({}, {**BODY, "spend_usd": 5}, "test_job_retry_now_request_invalid"),
])
def test_only_a_test_job_the_door_itself_made_is_touched(tmp_path, world, body, code):
    world = World(tmp_path, **world)
    before = world.records()
    with pytest.raises(retry.Refused, match=f"^{code}$"):
        world.ask(body=body)
    assert world.records() == before and "retry_now" not in world.read(world.row_path)


def test_only_the_capture_the_doors_row_names_is_reached(tmp_path):
    """Other records under the same scene are not the test job's: the door's row names its capture."""
    world = World(tmp_path, row={"capture_id": "walkthrough-elsewhere"}, recorded={"capture_id": "walkthrough-elsewhere"})
    before = world.records()
    with pytest.raises(retry.Refused, match="^test_job_retry_now_capture_absent$"):
        world.ask()
    assert world.records() == before


def test_a_withdrawn_capture_is_refused(tmp_path):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    world = World(tmp_path)
    value = {"schema_version": "website_capture_withdrawal_tombstone.v1", "scene_id": f"site-{JOB}",
             "request_id": JOB, "consent_revoked_at": "2026-11-24T12:00:00Z"}
    tombstone = world.capture.parents[1] / "website_withdrawal" / "tombstone.json"
    tombstone.parent.mkdir()
    world.write(tombstone, {**value, "digest": canonical_digest(value, digest_field="digest")})
    before = world.records()
    with pytest.raises(retry.Refused, match="^test_job_retry_now_capture_withdrawn$"):
        world.ask()
    assert world.records() == before


# ------------------------------------------------------------------------------------------- walls

@pytest.mark.parametrize(("atlas", "ledger", "walled", "wall"), [
    # The live Austin preparation's own stop: the paid-resource chokepoint refused (a money wall by its type).
    (None, _ledger(error="clean_plate: PaidResourceAdmissionBlocked"), "preparation_retry", "money"),
    (None, _ledger(error="website_control_scene-sponsorship_http_409:website_run_closed_by_owner"),
     "preparation_retry", "privacy_rights"),
    (None, _ledger(error="website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted"),
     "preparation_retry", "money"),
    (_atlas(last_refusal_reason="account_monthly_spend_limit_reached"), None, "atlas_preview", "money"),
    (_atlas(last_transient_reason="website_atlas_source_consent_required"), None, "atlas_preview", "privacy_rights"),
    (_atlas(last_error={"type": "PaidResourceAdmissionBlocked", "message": "blocked"}), None, "atlas_preview", "money"),
    (_atlas(state="failed", reason="spending_cap_reached"), None, "atlas_preview", "money"),
])
def test_a_record_whose_last_stop_is_a_wall_keeps_its_own_schedule(tmp_path, atlas, ledger, walled, wall):
    world = World(tmp_path, atlas=atlas, ledger=ledger)
    before = dict(zip(("atlas_preview", "preparation_retry"), world.records()))
    answer = world.ask()
    after = dict(zip(("atlas_preview", "preparation_retry"), world.records()))
    other = ({"atlas_preview", "preparation_retry"} - {walled}).pop()
    assert answer["records"][walled]["result"] == "stop_is_a_wall" and answer["records"][walled]["wall"] == wall
    assert after[walled] == before[walled]
    assert answer["records"][other]["result"] == "brought_forward" and answer["door_outcome"] == "brought_forward"


def test_when_every_scheduled_record_stops_on_a_wall_the_ask_is_refused_and_counts_nothing(tmp_path):
    world = World(tmp_path, atlas=_atlas(last_refusal_reason="account_monthly_spend_limit_reached"),
                  ledger=_ledger(error="clean_plate: PaidResourceAdmissionBlocked"))
    before = world.records()
    answer = world.ask()
    assert (answer["door_outcome"], answer["code"]) == ("refused", "test_job_retry_now_last_stop_is_money")
    assert world.records() == before and world.read(world.row_path)["retry_now"] == []


def test_an_operations_name_is_no_wall(tmp_path):
    """``preparation-spend`` names the WebApp call, not the reason: its 503 is an outage, as the preparation retry reads it."""
    world = World(tmp_path, ledger=_ledger(error="website_control_preparation-spend_http_503"))
    assert world.ask()["records"]["preparation_retry"]["result"] == "brought_forward"


# ------------------------------------------------------------------------------------------- locks

@contextlib.contextmanager
def _held(path: Path):
    """Hold the lock at ``path`` as another process's step would."""
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def test_an_atlas_step_holding_the_ticks_lock_is_left_running(tmp_path):
    world = World(tmp_path)
    before = world.read(world.atlas_path)
    with _held(world.capture / "pipeline" / "atlas" / "preview.lock"):  # the tick's step holds it
        answer = world.ask()
    assert answer["records"]["atlas_preview"] == {"result": "step_running"}
    assert world.read(world.atlas_path) == before
    assert answer["records"]["preparation_retry"]["result"] == "brought_forward"


def test_a_step_mark_left_by_a_killed_atlas_step_is_left_for_the_tick(tmp_path):
    marked = _atlas(step_in_flight={"since_epoch": NOW - 600, "stage": "source_pose"})
    world = World(tmp_path, atlas=marked)
    assert world.ask()["records"]["atlas_preview"]["result"] == "step_interrupted"
    assert world.read(world.atlas_path) == marked


@pytest.mark.parametrize(("ledger", "result"), [
    (_ledger(status="processing", lease_owner="host:1:x", lease_expires_at=_iso(NOW + 600)), "step_running"),
    # The listener began this retry 5 minutes ago and has not claimed it yet: it may still claim.
    (_ledger(record={"last_redrive_at": _iso(NOW - 300), "retry_count": 7}), "step_running"),
    # Began 20 minutes ago and never claimed (refused before its claim): its time can come forward.
    (_ledger(record={"last_redrive_at": _iso(NOW - 1_200), "retry_count": 7}), "brought_forward"),
    # Began, then claimed and failed again since: an ordinary armed retry.
    (_ledger(record={"last_redrive_at": _iso(NOW - 300)}, last_attempt_started_at=_iso(NOW - 290)), "brought_forward"),
])
def test_a_preparation_step_running_or_starting_is_left_alone(tmp_path, ledger, result):
    world = World(tmp_path, ledger=ledger)
    answer = world.ask()
    assert answer["records"]["preparation_retry"]["result"] == result
    assert (world.read(world.ledger_path) == ledger) is (result == "step_running")


def test_the_preparation_write_waits_for_the_listeners_ledger_lock(tmp_path):
    world = World(tmp_path)
    before = world.read(world.ledger_path)
    answers = []
    worker = threading.Thread(target=lambda: answers.append(
        retry._preparation(world.capture, now=datetime.fromtimestamp(NOW, timezone.utc))))
    with _held(world.capture / ".pipeline_job_ledger.json.lock"):  # the listener reads or commits under it
        worker.start()
        worker.join(0.5)
        assert worker.is_alive() and world.read(world.ledger_path) == before
    worker.join(10)
    assert answers[0]["result"] == "brought_forward" and world.read(world.ledger_path)["revision"] == 43


def test_the_claim_window_is_the_listeners_lease():
    from blueprint_pipeline.pubsub_handoff_listener import DEFAULT_JOB_LEASE_SECONDS

    assert retry.REDRIVE_CLAIM_SECONDS == DEFAULT_JOB_LEASE_SECONDS


# ------------------------------------------------------------------------------------------- the bound

def _rearm(world: World) -> None:
    """The job failed again after the retry-now: both records have a time ahead once more."""
    world.write(world.atlas_path, {**world.read(world.atlas_path), "next_attempt_at_epoch": NOW + 3_600})
    ledger = world.read(world.ledger_path)
    world.write(world.ledger_path, {**ledger, "preparation_retry": {**ledger["preparation_retry"],
                                                                    "next_attempt_at": _iso(NOW + 3_600)}})


def test_once_per_job_per_release(tmp_path):
    world = World(tmp_path)
    assert world.ask()["door_outcome"] == "brought_forward"
    _rearm(world)
    before = world.records()
    with pytest.raises(retry.Refused, match="^test_job_retry_now_used_on_this_release$"):
        world.ask(request_id="20261124T190000Z-test-job-retry-now-0000abce")
    assert world.records() == before
    assert world.ask(release=NEXT_RELEASE, request_id="20261125T160000Z-test-job-retry-now-0000abcf",
                     now=NOW + 60)["door_outcome"] == "brought_forward"
    assert [entry["release"] for entry in world.read(world.row_path)["retry_now"]] == [RELEASE, NEXT_RELEASE]


def test_an_ask_that_changed_nothing_does_not_use_the_releases_one(tmp_path):
    world = World(tmp_path, atlas=_atlas(next_attempt_at_epoch=NOW), ledger=_ledger(status="processing"))
    assert world.ask()["door_outcome"] == "unchanged"
    assert world.read(world.row_path)["retry_now"] == []
    _rearm(world)
    world.write(world.ledger_path, {**world.read(world.ledger_path), "status": "failed_retryable"})
    assert world.ask(request_id="20261124T190000Z-test-job-retry-now-0000abce")["door_outcome"] == "brought_forward"


def test_an_ask_stopped_partway_still_uses_the_releases_one(tmp_path, monkeypatch):
    world = World(tmp_path)

    def killed(*_args, **_kwargs):
        raise OSError("killed")
    monkeypatch.setattr(retry, "_preparation", killed)
    with pytest.raises(OSError):
        world.ask()
    assert [entry["outcome"] for entry in world.read(world.row_path)["retry_now"]] == ["applying"]
    with pytest.raises(retry.Refused, match="^test_job_retry_now_used_on_this_release$"):
        world.ask(request_id="20261124T190000Z-test-job-retry-now-0000abce")


def test_an_ask_that_raised_before_any_record_changed_does_not_use_the_releases_one(tmp_path, monkeypatch):
    """Austin (2026-11-25): an OSError before either record changed left "applying", and the release's one was spent
    on nothing. Its records are unchanged and the next ask on the same release brings both forward."""
    world = World(tmp_path)
    before = world.records()
    real = retry._atlas

    def unreadable(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", "/scenes/x/pipeline/atlas/preview.lock")
    monkeypatch.setattr(retry, "_atlas", unreadable)
    with pytest.raises(PermissionError):
        world.ask()
    assert world.read(world.row_path)["retry_now"] == [] and world.records() == before
    monkeypatch.setattr(retry, "_atlas", real)
    assert world.ask(request_id="20261124T190000Z-test-job-retry-now-0000abce")["door_outcome"] == "brought_forward"


@pytest.mark.parametrize(("error", "code"), [
    (PermissionError(13, "Permission denied", "/var/lib/x/captures/c/pipeline/atlas/preview.lock"),
     "PermissionError:EACCES:atlas/preview.lock"),
    (OSError(30, "Read-only file system", "/var/lib/x/captures/c/.pipeline_job_ledger.json.lock"),
     "OSError:EROFS:c/.pipeline_job_ledger.json.lock"),
    (OSError("disk"), "OSError"),
])
def test_a_failed_ask_names_the_error_and_the_file_never_its_message(monkeypatch, capsys, error, code):
    def run(*_args, **_kwargs):
        raise error
    monkeypatch.setattr(retry, "run", run)
    assert retry.main(["--request-json", json.dumps(BODY), "--door-request-id", THIS, "--requested-by", "a"]) == 1
    assert json.loads(capsys.readouterr().out) == {"door_outcome": "failed", "code": code}


@pytest.mark.parametrize("release", [None, "", "not-a-commit"])
def test_with_no_release_to_bound_it_the_ask_is_refused(tmp_path, release):
    world = World(tmp_path)
    before = world.records()
    with pytest.raises(retry.Refused, match="^test_job_retry_now_release_unreadable$"):
        world.ask(release=release)
    assert world.records() == before


# ------------------------------------------------------------------------------------------- entry points

@pytest.mark.parametrize(("answer", "code"), [({"door_outcome": "brought_forward"}, 0),
                                              ({"door_outcome": "unchanged"}, 3),
                                              ({"door_outcome": "refused", "code": "x"}, 2),
                                              (retry.Refused("test_job_retry_now_used_on_this_release"), 2),
                                              (OSError("disk"), 1)])
def test_the_module_prints_its_answer_and_exits_by_it(monkeypatch, capsys, answer, code):
    def run(*_args, **_kwargs):
        if isinstance(answer, Exception):
            raise answer
        return answer
    monkeypatch.setattr(retry, "run", run)
    assert retry.main(["--request-json", json.dumps(BODY), "--door-request-id", THIS, "--requested-by", "a"]) == code
    printed = json.loads(capsys.readouterr().out)
    assert printed["door_outcome"] == (answer.get("door_outcome") if isinstance(answer, dict) else
                                       "refused" if isinstance(answer, retry.Refused) else "failed")


@pytest.fixture()
def retry_env(env, tmp_path):  # noqa: F811 - the scripts module's fixture
    repository = tmp_path / "repository"
    (repository / "src" / "blueprint_pipeline").mkdir(parents=True)
    (repository / "src" / "blueprint_pipeline" / "internal_test_job_retry_now.py").write_text("")
    _write_stub(tmp_path / "stubs" / "setpriv", SETPRIV_STUB)
    _write_stub(tmp_path / "job-python", "#!/bin/bash\nprintf 'job %s\\n' \"$*\" >> \"$STUB_LOG\"\n"
                                         "printf '%s\\n' \"${FAKE_JOB:-}\"\nexit \"${FAKE_TOOL_RC:-0}\"\n")
    return {**env, "DOOR_VENV_PYTHON": str(tmp_path / "job-python"), "DOOR_CONTROL_PLANE_REPO": str(repository),
            "DOOR_SERVICE_USER": "blueprint", "DOOR_REQUESTED_BY": "cloud-agent",
            "DOOR_TEST_JOB_RETRY_NOW_REQUEST": json.dumps(BODY)}


@pytest.mark.parametrize(("rc", "answer", "status", "code", "exit_code"), [
    (0, {"door_outcome": "brought_forward"}, "brought_forward", None, 0),
    (3, {"door_outcome": "unchanged"}, "unchanged", None, 0),
    (2, {"door_outcome": "refused", "code": "test_job_retry_now_used_on_this_release"}, "refused",
     "test_job_retry_now_used_on_this_release", 2),
    (1, {"door_outcome": "failed", "code": "OSError"}, "failed", "OSError", 1),
])
def test_the_door_script_runs_the_release_module_as_the_service_user(retry_env, rc, answer, status, code, exit_code):
    done, outcome, calls = _run("door-test-job-retry-now.sh", retry_env, DOOR_REQUEST_ID=THIS,
                                FAKE_JOB=json.dumps(answer), FAKE_TOOL_RC=str(rc))
    assert (done, outcome["status"], outcome["code"]) == (exit_code, status, code)
    assert calls[0] == "setpriv --reuid=blueprint --regid=blueprint --init-groups --inh-caps=-all --"
    assert calls[2].startswith("job -m blueprint_pipeline.internal_test_job_retry_now --request-json ")
    assert f"--door-request-id {THIS} --requested-by cloud-agent --door-results " in calls[2]
    assert json.loads(Path(outcome["result"]).read_text()) == answer


def test_the_door_script_says_when_the_release_lacks_the_module(retry_env, tmp_path):
    (tmp_path / "repository" / "src" / "blueprint_pipeline" / "internal_test_job_retry_now.py").unlink()
    done, outcome, _calls = _run("door-test-job-retry-now.sh", retry_env, DOOR_REQUEST_ID=THIS)
    assert (done, outcome["status"], outcome["code"]) == (2, "refused", "test_job_retry_now_release_lacks_module")


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_door_client_sends_the_job_and_the_reason(monkeypatch):
    client = _module(ROOT / "scripts" / "operator_door.py", "operator_door_client_retry_now")
    sent = []
    monkeypatch.setattr(client, "_submit", lambda body, args: sent.append(body) or 0)
    assert client.run(client.build_parser().parse_args(["test-job-retry-now", JOB, "--reason", BODY["reason"]])) == 0
    assert sent == [BODY]
    assert client._terminal({"request": BODY, "outcome": {"status": "brought_forward"}}) is True
    assert client._terminal({"request": BODY, "outcome": {"status": "unchanged"}}) is False


def test_the_runner_asks_for_a_jobs_retry_now_by_its_label(tmp_path, monkeypatch, capsys):
    runner = _module(ROOT / "scripts" / "test_jobs.py", "test_jobs_retry_now")
    state = tmp_path / "run.json"
    state.write_text(json.dumps({"jobs": [{"label": "austin-apartment-floors", "job_id": JOB, "capture_id": CAPTURE}]}))
    posts = []

    def door_json(method, route, query=None, body=None):
        if method == "POST":
            posts.append(body)
            return {"id": THIS}
        return {"request": BODY, "outcome": {"status": "brought_forward", "result": "/r.json"}}
    monkeypatch.setattr(runner.door, "_json", door_json)
    monkeypatch.setattr(runner, "cat_json", lambda path: {"door_outcome": "brought_forward", "job_id": JOB})
    argv = ["retry-now", "--state", str(state), "--label", "austin-apartment-floors", "--reason", BODY["reason"]]
    assert runner.main(argv) == 0 and posts == [BODY]
    assert json.loads(capsys.readouterr().out)["door_outcome"] == "brought_forward"
    assert runner.main(["retry-now", "--state", str(state), "--label", "unknown", "--reason", "r"]) == 2
    assert len(posts) == 1
