"""The site owner confirms a website job's thresholds on the job page; the team packet then proceeds with no operator."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from blueprint_pipeline import task_evaluation_owner_success_confirmation as confirmation
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import team_scene_packet_worker as worker
from blueprint_pipeline import website_owner_success_confirmation as website
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_configured_controls_deferred_inputs import _document_name
from tests.test_task_evaluation_owner_success_confirmation import OWNER, compiled_task_spec, confirm, rethreshold
from tests.test_task_evaluation_rigid_relocation_native_adapter import DEFINITION, SUCCESS
from tests.test_team_scene_packet_owner_confirmation import REFUSED, _compile, _job, _request


SCHEDULE_FIELDS = frozenset({"progression_digest", "attempt", "checked_at", "next_check_at"})
T0 = 1_793_700_000.0  # 2026-11-03T13:20:00Z


class Clock:
    """The exchange's clock. By default each pass is two days after the last, so every job is due."""

    def __init__(self) -> None:
        self.now, self.step = T0, 2 * 86400.0

    def __call__(self) -> float:
        self.now += self.step
        return self.now


@pytest.fixture(autouse=True)
def clock(monkeypatch) -> Clock:
    value = Clock()
    monkeypatch.setattr(website, "_now", value)
    return value


def _core(outcome: dict[str, Any]) -> dict[str, Any]:
    """An exchange outcome without its schedule."""

    return {key: value for key, value in outcome.items() if key not in SCHEDULE_FIELDS}


class FakeWebApp:
    """The WebApp side of the exchange: it keeps each proposal and answers with the owner's decision on it."""

    def __init__(self) -> None:
        self.proposals: dict[str, dict[str, Any]] = {}
        self.decisions: dict[str, dict[str, Any]] = {}
        self.calls = 0

    def __call__(self, notice: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        notice = json.loads(json.dumps(notice))  # as it crosses the wire
        assert notice["proposal_digest"] == cross_runtime_canonical_digest(notice, digest_field="proposal_digest")
        self.proposals[notice["proposal_digest"]] = notice
        decision = self.decisions.get(notice["proposal_digest"])
        return {"schema_version": website.RECEIPT_SCHEMA, "status": decision["decision"] if decision else "pending",
                "intent_id": notice["intent_id"], "proposal_digest": notice["proposal_digest"], "decision": decision}

    def decide(self, kind: str, reason: str | None = None, **changes: Any) -> dict[str, Any]:
        [notice] = list(self.proposals.values())[-1:]
        decision = {"schema_version": website.DECISION_SCHEMA, "intent_id": notice["intent_id"],
                    "intent_digest": notice["intent_digest"], "request_id": "job-one", "capture_id": "walkthrough-job-one",
                    "proposal_digest": notice["proposal_digest"],
                    "configured_scene_revision_digest": notice["configured_scene_revision_digest"],
                    "binding_digest": notice["binding_digest"], "decision": kind, "reason": reason,
                    "decided_via": "owner_job_link", "decided_at": "2026-11-03T18:00:00.000Z", **changes}
        decision["decision_digest"] = cross_runtime_canonical_digest(decision)
        self.decisions[notice["proposal_digest"]] = decision
        return decision


def _queued(tmp_path: Path, monkeypatch) -> tuple[Any, dict[str, Any], Path, str]:
    """A published proposal_only job whose team-packet request is blocked on the owner contract, as the host keeps it."""

    job, inputs = _job(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "current_release_commit", lambda: None)
    request = _request(job)
    progression = json.loads((job.directory / "progression.json").read_text())
    progression["state"]["scene_preparation_completion"]["team_scene_request"]["request_digest"] = (
        request["request_digest"])
    (job.directory / "progression.json").write_text(json.dumps(intake._seal(progression, "progression_digest")))
    queue = worker._queue(tmp_path / "queue")
    key = worker.request_key(request)
    (queue / "processed" / f"{key}.json").write_text(json.dumps(request))
    (queue / "status" / f"{key}.json").write_text(json.dumps({
        "schema_version": worker.STATUS_SCHEMA, "request_key": key, "status": "blocked", "blockers": [REFUSED]}))
    job.inputs = inputs
    return job, inputs, queue, key


def _status(queue: Path, key: str) -> dict[str, Any]:
    return json.loads((queue / "status" / f"{key}.json").read_text())


def _republish(job: Any, tmp_path: Path, monkeypatch, **values: Any) -> dict[str, Any]:
    """The same intent republished with other thresholds, as the host retains it."""

    from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as source

    configured, references = rethreshold(job.revision, job.references, tmp_path / "owner-case", **values)
    payload = json.dumps(configured, sort_keys=True).encode()
    (job.run / "configured_scene_revision.v1.json").write_bytes(payload)
    for name in (DEFINITION, SUCCESS):
        (job.documents / _document_name(name)).write_bytes(Path(references[name]["materialized_path"]).read_bytes())
    progression = json.loads((job.directory / "progression.json").read_text())
    progression["result_reference"] = {**progression["result_reference"],
                                       "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                                       "size_bytes": len(payload)}
    (job.directory / "progression.json").write_text(json.dumps(intake._seal(progression, "progression_digest")))
    monkeypatch.setattr(source, "_validate_source", lambda root: ({
        "configured_scene_revision_path": str(job.run / "configured_scene_revision.v1.json"),
        "configured_scene_revision_digest": configured["revision_digest"]}, {}, {}))
    return configured


def test_the_owner_confirms_on_the_website_and_the_team_packet_proceeds(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import _require_strict_owner_success_contract
    from blueprint_pipeline.team_scene_packet import read_team_scene_inputs

    job, inputs, queue, key = _queued(tmp_path, monkeypatch)
    webapp = FakeWebApp()
    # The first pass sends the thresholds exactly as built; the owner has not decided, so nothing moves.
    first = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    assert first["requeued"] == []
    [notice] = webapp.proposals.values()
    success = json.loads((job.documents / _document_name(SUCCESS)).read_text())
    assert notice["configured_success_criteria"] == {key: value for key, value in success.items()
                                                     if key not in {"schema_version", "status"}}
    assert notice["configured_scene_revision_digest"] == job.revision["revision_digest"]
    assert notice["intent_id"] == job.intent["intent_id"]
    assert _core(_status(queue, key)[website.STATUS_FIELD]) == {"status": "pending",
                                                                "proposal_digest": notice["proposal_digest"]}
    assert not (job.directory / confirmation.DIRECTORY).exists()
    # The owner confirms on the job page; the next pass writes the door's own record and requeues the request.
    decision = webapp.decide("confirmed")
    second = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    [path] = (job.directory / confirmation.DIRECTORY).glob("*.json")
    row = json.loads(path.read_text())
    assert second["requeued"] == [{"request_key": key, "reason": "owner_success_confirmation",
                                   "confirmation_digests": [row["confirmation_digest"]]}]
    assert (queue / "pending" / f"{key}.json").is_file()
    assert _core(second["website_owner_decisions"][key]) == {
        "status": "confirmed", "proposal_digest": notice["proposal_digest"],
        "decision_digest": decision["decision_digest"], "record": "confirmed",
        "confirmation_digest": row["confirmation_digest"]}
    # The very record the door writes, under the same validators, naming the website.
    assert set(row) == confirmation.RECORD_FIELDS
    assert row["schema_version"] == confirmation.SCHEMA and row["issuer_channel"] == "website"
    assert row["owner"] == OWNER and row["approved_by"] == OWNER["user_id"]
    assert decision["decision_digest"] in row["approval_evidence"]
    assert "channel=website" in row["authorization_reference"]
    assert row["thresholds_changed"] is False and row["provider_mutation_performed"] is False
    assert confirmation.confirmations(job.directory, job.intent) == [row]
    authority = confirmation.confirmed_owner_authority(task_spec=compiled_task_spec(job), revision=job.revision,
                                                       intent=job.intent, directory=job.directory)
    assert authority["basis"] == "owner_confirmed_built_thresholds_via_website"
    assert authority["confirmation_digest"] == row["confirmation_digest"]
    # The team packet compiles the owner contract as the owner's, and screening admits it.
    compiled = _compile(job, inputs, tmp_path / "confirmed")
    assert compiled["owner_success_confirmation"] == row["confirmation_digest"]
    contract = read_team_scene_inputs(compiled["path"])["task_success_contract"]
    assert contract["provenance"]["author_source"] == "task_owner"
    assert contract["provenance"]["confirmation_status"] == "confirmed"
    assert contract["provenance"]["author_id"] == "owner-success-confirmation:" + row["confirmation_digest"]
    packet = Path(compiled["path"]).parent / "native-task-packet"
    plan = json.loads((packet / "native_task_arena_scene_plan.v1.json").read_text())
    _require_strict_owner_success_contract(task_spec=plan["task_spec"], contract=contract)
    # A later pass sends nothing more for a confirmed job.
    (queue / "pending" / f"{key}.json").rename(queue / "processed" / f"{key}.json")
    calls = webapp.calls
    third = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    assert third["requeued"] == [] and webapp.calls == calls
    assert third["website_owner_decisions"][key]["status"] == "already_confirmed"
    assert third["website_owner_decisions"][key]["attempt"] == 1


def test_this_is_wrong_holds_the_job_with_its_reason_and_changes_no_number(tmp_path, monkeypatch) -> None:
    job, _inputs, queue, key = _queued(tmp_path, monkeypatch)
    before = {name: (job.documents / _document_name(name)).read_bytes() for name in (DEFINITION, SUCCESS)}
    webapp = FakeWebApp()
    worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    decision = webapp.decide("held", "too_strict")
    result = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    assert result["requeued"] == []
    [notice] = webapp.proposals.values()
    status = _status(queue, key)
    assert status["status"] == "blocked" and status["blockers"] == [REFUSED]
    assert _core(status[website.STATUS_FIELD]) == {"status": "held", "reason": "too_strict",
                                            "proposal_digest": notice["proposal_digest"],
                                            "decision_digest": decision["decision_digest"]}
    assert not (job.directory / confirmation.DIRECTORY).exists()
    assert {name: (job.documents / _document_name(name)).read_bytes() for name in (DEFINITION, SUCCESS)} == before


def test_a_decision_on_a_stale_proposal_is_never_applied(tmp_path, monkeypatch) -> None:
    job, _inputs, queue, key = _queued(tmp_path, monkeypatch)
    webapp = FakeWebApp()
    worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    stale = webapp.decide("confirmed")
    # The job is republished with another threshold: the owner never saw this one.
    _republish(job, tmp_path, monkeypatch, maximum_task_contact_force_n=50.0)
    # An honest WebApp asks again: the new proposal is pending, and nothing is written.
    result = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    fresh = result["website_owner_decisions"][key]
    assert fresh["status"] == "pending" and fresh["proposal_digest"] != stale["proposal_digest"]
    assert not (job.directory / confirmation.DIRECTORY).exists()
    # A WebApp answering the new proposal with the old decision is refused, and nothing is written.

    def replay(notice: dict[str, Any]) -> dict[str, Any]:
        return {"schema_version": website.RECEIPT_SCHEMA, "status": "confirmed", "intent_id": notice["intent_id"],
                "proposal_digest": notice["proposal_digest"], "decision": stale}

    replayed = worker.requeue_transient_blocked(queue_root=queue, website_transport=replay)
    assert _core(replayed["website_owner_decisions"][key]) == {
        "status": "failed", "error": "website_owner_success_decision_invalid", "proposal_digest": fresh["proposal_digest"]}
    assert _status(queue, key)[website.STATUS_FIELD]["status"] == "failed"
    # Nor can the old decision be recorded directly against the republished job.
    with pytest.raises(ValueError, match="website_owner_success_proposal_changed"):
        website.record_website_confirmation(intent_root=job.intents, intent=job.intent, decision=stale)
    assert not (job.directory / confirmation.DIRECTORY).exists()


@pytest.mark.parametrize("changes", [
    {"decided_via": "operator_console"},  # only the owner's job link decides
    {"intent_id": "scene-" + "b" * 64},  # another job
    {"binding_digest": "sha256:" + "c" * 64},  # another owner, document or proposal
    {"configured_scene_revision_digest": "sha256:" + "d" * 64},
    {"reason": "too_strict"},  # a confirmation carries no reason
    {"decision_digest": "sha256:" + "e" * 64},  # changed after sealing
])
def test_a_decision_that_is_not_the_owners_on_this_proposal_is_refused(tmp_path, monkeypatch, changes) -> None:
    job, _inputs, queue, key = _queued(tmp_path, monkeypatch)
    webapp = FakeWebApp()
    worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    decision = webapp.decide("confirmed")
    sealed = {**decision, **changes}
    if "decision_digest" not in changes:
        sealed["decision_digest"] = cross_runtime_canonical_digest(sealed, digest_field="decision_digest")
    webapp.decisions[decision["proposal_digest"]] = sealed
    result = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    assert result["requeued"] == []
    assert result["website_owner_decisions"][key]["status"] == "failed"
    assert not (job.directory / confirmation.DIRECTORY).exists()


def test_a_held_decision_needs_a_known_reason(tmp_path, monkeypatch) -> None:
    job, _inputs, queue, key = _queued(tmp_path, monkeypatch)
    webapp = FakeWebApp()
    worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    webapp.decide("held", "the numbers look off")
    result = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    assert _core(result["website_owner_decisions"][key]) == {
        "status": "failed", "error": "website_owner_success_decision_invalid",
        "proposal_digest": list(webapp.proposals)[-1]}


def test_the_door_path_is_unchanged_and_the_website_is_not_asked_again(tmp_path, monkeypatch) -> None:
    job, _inputs, queue, key = _queued(tmp_path, monkeypatch)
    applied = confirm(job)
    [path] = (job.directory / confirmation.DIRECTORY).glob("*.json")
    row = json.loads(path.read_text())
    assert row["issuer_channel"] == "operator_door" and "channel=operator_door" in row["authorization_reference"]
    authority = confirmation.confirmed_owner_authority(task_spec=compiled_task_spec(job), revision=job.revision,
                                                       intent=job.intent, directory=job.directory)
    assert authority["basis"] == "owner_confirmed_built_thresholds_via_operator_door"
    webapp = FakeWebApp()
    result = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    assert webapp.calls == 0
    assert _core(result["website_owner_decisions"][key]) == {
        "status": "already_confirmed", "proposal_digest": website.proposal_notice(
            confirmation.binding(intent=job.intent, revision=job.revision,
                                 criteria=confirmation.published_job(job.directory, job.intent)["criteria"],
                                 proposal=confirmation.published_job(job.directory, job.intent)["proposal"])
        )["proposal_digest"], "confirmation_digest": applied["confirmation_digest"], "issuer_channel": "operator_door"}
    assert result["requeued"] == [{"request_key": key, "reason": "owner_success_confirmation",
                                   "confirmation_digests": [applied["confirmation_digest"]]}]


def test_a_revoked_intent_sends_nothing_and_an_unknown_channel_is_no_confirmation(tmp_path, monkeypatch) -> None:
    job, _inputs, queue, key = _queued(tmp_path, monkeypatch)
    (job.directory / "revoked.json").write_text("{}")
    webapp = FakeWebApp()
    result = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    assert webapp.calls == 0 and _core(result["website_owner_decisions"][key]) == {"status": "intent_revoked"}
    with pytest.raises(ValueError, match="owner_success_confirmation_not_authorized"):
        confirmation.confirm_owner_success_contract(
            queue_root=job.intents, intent_id=job.intent["intent_id"], owner=OWNER, approved_by=OWNER["user_id"],
            approval_evidence="x", apply_ack="sha256:" + "0" * 64, authorization_reference="x",
            ack=confirmation.ACK, channel="email")


def test_an_unreachable_webapp_is_recorded_and_never_stops_the_pass(tmp_path, monkeypatch) -> None:
    job, _inputs, queue, key = _queued(tmp_path, monkeypatch)
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    result = worker.requeue_transient_blocked(queue_root=queue)
    assert result["status"] == "idle"
    assert _core(result["website_owner_decisions"][key])["status"] == "failed"
    assert result["website_owner_decisions"][key]["error"] == "website_owner_success_webapp_url_missing"
    assert _status(queue, key)[website.STATUS_FIELD] == result["website_owner_decisions"][key]


def test_each_pass_asks_about_a_bounded_rotating_set_of_jobs(tmp_path) -> None:
    queue = worker._queue(tmp_path / "queue")
    for index in range(6):
        (queue / "processed" / f"{index:032x}.json").write_text("{}")
        (queue / "status" / f"{index:032x}.json").write_text(json.dumps({"status": "blocked", "blockers": [REFUSED]}))
    (queue / "status" / f"{1:032x}.json").write_text(json.dumps({"status": "blocked", "blockers": ["other"]}))
    asked = website.exchange_owner_decisions(queue, limit=2, now=0)
    assert list(asked) == [f"{0:032x}", f"{2:032x}"]
    # Three minutes later the set starts three blocked jobs on, and wraps.
    assert list(website.exchange_owner_decisions(queue, limit=2, now=60 * 3)) == [f"{4:032x}", f"{5:032x}"]
    assert list(website.exchange_owner_decisions(queue, limit=3, now=60 * 4)) == [f"{5:032x}", f"{0:032x}", f"{2:032x}"]
    assert all(row["status"] == "failed" for row in asked.values())


def test_the_proposal_goes_over_the_signed_pipeline_sync_transport(monkeypatch) -> None:
    from blueprint_pipeline import safe_outbound_http, task_evaluation_launch_webapp_sync

    sent: dict[str, Any] = {}

    def fake_request(url, *, method, data, headers, timeout_seconds, policy, max_response_bytes):
        sent.update(url=url, method=method, data=data, headers=headers)
        return type("Response", (), {"body": json.dumps({"ok": True}).encode()})()

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://tryblueprint.io/api/internal/pipeline")
    monkeypatch.setattr(task_evaluation_launch_webapp_sync, "load_pipeline_sync_token", lambda: "test-token")
    monkeypatch.setattr(safe_outbound_http, "request", fake_request)
    notice = {"intent_id": "scene-" + "a" * 64, "proposal_digest": "sha256:" + "1" * 64}
    assert website.send_proposal(notice) == {"ok": True}
    assert sent["url"] == ("https://tryblueprint.io/api/internal/pipeline/task-evaluation-scene-intents/scene-"
                           + "a" * 64 + "/owner-success-proposal")
    assert sent["method"] == "POST" and json.loads(sent["data"]) == notice
    assert sent["headers"]["X-Blueprint-Pipeline-Signature"].startswith("sha256=")

    def refused(url, **_kwargs):
        raise HTTPError(url, 409, "conflict", {}, io.BytesIO(b'{"code":"website_owner_success_consent_revoked"}'))

    monkeypatch.setattr(safe_outbound_http, "request", refused)
    with pytest.raises(ValueError, match=r"\Awebsite_owner_success_http_409:website_owner_success_consent_revoked\Z"):
        website.send_proposal(notice)


def test_the_proposal_digest_is_the_cross_runtime_digest_the_webapp_checks(tmp_path, monkeypatch) -> None:
    job, _inputs, _queue, _key = _queued(tmp_path, monkeypatch)
    published = confirmation.published_job(job.directory, job.intent)
    bound = confirmation.binding(intent=job.intent, revision=published["revision"], criteria=published["criteria"],
                                 proposal=published["proposal"])
    notice = website.proposal_notice(bound)
    wire = json.loads(json.dumps(notice))
    assert wire["proposal_digest"] == cross_runtime_canonical_digest(wire, digest_field="proposal_digest")
    assert set(notice) == {"schema_version", "intent_id", "intent_digest", "task_id",
                           "configured_scene_revision_digest", "configured_success_criteria",
                           "configured_success_criteria_digest", "proposal_authority_digest", "forbidden_contacts",
                           "binding_digest", "proposal_digest"}
    # Each forbidden contact says what it counts and what it leaves out, as the native scorer defines it.
    assert notice["forbidden_contacts"] == [
        {"contact_class": "robot_background",
         "parts": ["robot_protected_bodies", "scene_including_destination_support"], "excluding": []},
        {"contact_class": "robot_object", "parts": ["robot_protected_bodies", "task_object"],
         "excluding": ["admitted_gripper_task_contact_bodies"]},
        {"contact_class": "object_background", "parts": ["task_object", "scene"], "excluding": ["destination_support"]}]
    # The private document locator never leaves the Pipeline.
    assert "s3://" not in json.dumps(notice) and bound["success_criteria_document"]["uri"].startswith("s3://")


def test_every_scored_contact_class_has_a_scope_and_an_unknown_one_is_sent_bare() -> None:
    from blueprint_pipeline.native_rigid_episode_telemetry import CONTACT_CHANNELS

    assert set(website.CONTACT_CLASS_SCOPE) == set(CONTACT_CHANNELS)
    assert website.forbidden_contacts({"forbidden_contact_classes": ["new_class"]}) == [
        {"contact_class": "new_class", "parts": [], "excluding": []}]
    assert website.forbidden_contacts({}) == []


def test_a_network_error_is_one_jobs_failed_exchange_and_never_stops_the_pass(tmp_path, monkeypatch) -> None:
    job, _inputs, queue, key = _queued(tmp_path, monkeypatch)

    def cut_off(_notice: dict[str, Any]) -> dict[str, Any]:
        raise http.client.IncompleteRead(b'{"schema_version"')

    result = worker.requeue_transient_blocked(queue_root=queue, website_transport=cut_off)
    assert result["status"] == "idle" and "website_owner_decisions_error" not in result
    assert result["website_owner_decisions"][key]["status"] == "failed"
    assert result["website_owner_decisions"][key]["error"] == "IncompleteRead"
    assert _status(queue, key)[website.STATUS_FIELD]["error"] == "IncompleteRead"
    assert not (job.directory / confirmation.DIRECTORY).exists()
    # An error outside any one job's exchange still leaves the requeue running: a door confirmation requeues.
    applied = confirm(job)

    def broken(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise http.client.BadStatusLine("")

    monkeypatch.setattr(website, "exchange_owner_decisions", broken)
    result = worker.requeue_transient_blocked(queue_root=queue)
    assert result["requeued"] == [{"request_key": key, "reason": "owner_success_confirmation",
                                   "confirmation_digests": [applied["confirmation_digest"]]}]
    assert result["website_owner_decisions"] == {}
    assert result["website_owner_decisions_error"]["type"] == "http.client.BadStatusLine"


def test_a_job_is_asked_again_only_when_due_and_each_wait_doubles(tmp_path, monkeypatch, clock) -> None:
    job, _inputs, queue, key = _queued(tmp_path, monkeypatch)
    webapp = FakeWebApp()
    clock.step = 0.0

    def run(at: float, transport: Any = webapp) -> dict[str, Any] | None:
        clock.now = T0 + at
        return worker.requeue_transient_blocked(queue_root=queue, website_transport=transport)["website_owner_decisions"].get(key)

    def kept() -> dict[str, Any]:
        return _status(queue, key)[website.STATUS_FIELD]

    first = run(0)
    assert webapp.calls == 1 and first["status"] == "pending" and first["attempt"] == 1
    assert first["checked_at"] == website._instant(T0) and first["next_check_at"] == website._instant(T0 + 120)
    # Not due: nothing is sent and the status is not rewritten.
    before = _status(queue, key)
    assert run(119) is None and webapp.calls == 1 and _status(queue, key) == before
    # Due: the same answer again doubles the wait (120, 240, 480 s ...).
    assert run(120)["attempt"] == 2 and webapp.calls == 2 and kept()["next_check_at"] == website._instant(T0 + 360)
    assert run(359) is None and webapp.calls == 2
    # The owner holds the job: a hold waits on its own, longer steps (30 min, doubling to a day).
    webapp.decide("held", "too_strict")
    held = run(360)
    assert held["status"] == "held" and held["attempt"] == 1 and held["next_check_at"] == website._instant(T0 + 360 + 1800)
    assert run(360 + 1799) is None and webapp.calls == 3
    assert run(360 + 1800)["attempt"] == 2 and kept()["next_check_at"] == website._instant(T0 + 360 + 1800 + 3600)
    # A WebApp refusal (4xx, a rate limit among them) waits 15 min first; a transport failure 2 min.

    def limited(_notice: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("website_owner_success_http_429:pipeline_sync_rate_limited")

    refused = run(10_000, limited)
    assert refused["error"] == "website_owner_success_http_429:pipeline_sync_rate_limited"
    assert refused["attempt"] == 1 and refused["next_check_at"] == website._instant(T0 + 10_000 + 900)
    assert run(10_000 + 900, limited)["next_check_at"] == website._instant(T0 + 10_900 + 1800)

    def reset(_notice: dict[str, Any]) -> dict[str, Any]:
        raise ConnectionResetError("reset by peer")

    dropped = run(20_000, reset)
    assert dropped["error"] == "ConnectionResetError" and dropped["next_check_at"] == website._instant(T0 + 20_120)
    # Every wait has a cap.
    capped = website._scheduled({"status": "held", "proposal_digest": "p"},
                                {"status": "held", "proposal_digest": "p", "progression_digest": "g", "attempt": 40},
                                "g", 0.0)
    assert capped["attempt"] == 41 and capped["next_check_at"] == website._instant(86400)
    # A progression that changed without changing the proposal sends nothing and keeps the schedule.
    schedule = kept()
    progression = json.loads((job.directory / "progression.json").read_text())
    (job.directory / "progression.json").write_text(json.dumps(intake._seal(
        {**progression, "blockers": ["unrelated_note"]}, "progression_digest")))
    calls = webapp.calls
    moved = run(20_001)
    assert webapp.calls == calls and moved["next_check_at"] == schedule["next_check_at"]
    assert moved["progression_digest"] != schedule["progression_digest"]
    assert run(20_002) is None
    # A republished job is a new proposal: it is sent at once, whatever the wait.
    _republish(job, tmp_path, monkeypatch, maximum_task_contact_force_n=50.0)
    fresh = run(20_003)
    assert webapp.calls == calls + 1 and fresh["status"] == "pending" and fresh["attempt"] == 1
    assert fresh["proposal_digest"] != schedule["proposal_digest"]
