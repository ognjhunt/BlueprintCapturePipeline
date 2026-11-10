"""Publication v3 carries what Blueprint's own checks knew about the scene, and no result waits on the website.

The site reads a count as an estimate of the robots. A run's camera-gate flags, a
humanoid row's spawn and floor flags and the free CPU scene check's verdict now
travel with that count (typed codes only), and a website that does not accept v3
yet is sent the same result as v2 in the same pass, in either deploy order.

``BLUEPRINT_WRITE_SCREENING_PUBLICATION_FIXTURES=1`` rewrites the v3 fixture. The
WebApp commits the same bytes (``server/tests/fixtures/
pipeline-beta-policy-screening-flagged-publication.v3.json``) and pins the same
sha256, so a change on either side fails a test there.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from urllib.error import HTTPError

from jsonschema import Draft202012Validator, FormatChecker
import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_worker as worker
from blueprint_pipeline import gpu_render_providers
from blueprint_pipeline.beta_policy_screening_plan import validate_screening_plan
from blueprint_pipeline.beta_policy_screening_publication_flags import (
    DOWNGRADE_FIELD, FLAG_CODE, PUBLICATION_SCHEMA_V2, PUBLICATION_SCHEMA_V3, UNTYPED, as_v2, publication_flags_valid,
    row_flags, scene_feasibility, typed_flags,
)
from blueprint_pipeline.beta_policy_screening_result import PUBLICATION_SCHEMA, screening_publication
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.website_candidate_feasibility import SCENE_CHECK_FLAGS, reach_rank, scene_check
from tests.test_beta_policy_screening_coordinator import Backend, native_result, plan as make_plan, seal
from tests.test_beta_policy_screening_partial_cells import Provider, _shape, _webapp_contract, run_campaign

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/beta_policy_screening_flagged_publication.v3.json"
#: The fixture's bytes. The WebApp's ``beta-policy-screening-flags.test.ts`` pins the same value for its copy.
FIXTURE_SHA256 = "1b95db35f7b61b89be966d2512707b3c11e7fa3c6f9f74b0cd5b34604957c8f2"
CARRY = "robot_rigid_pick_place_unreachable:carry"
SCRIPTED_CARRY = "scripted_carry_path_not_found"
G1_FLAGS = ["g1_goal_marker_not_in_initial_view", "g1_scene_floor_estimated"]


@pytest.fixture(autouse=True)
def _no_provider_io(monkeypatch):
    monkeypatch.setattr(gpu_render_providers, "VastRenderProvider", Provider)


def flagged_rank(*reasons, source="packet"):
    """A row's reach evidence as planning records it when its free CPU solve searched and did not reach."""
    if source == "packet":
        return reach_rank({"status": "feasible", "reach_source": "franka_chain_reach_bound",
                           "station_search": {"status": "blocked", "blockers": list(reasons), "cleared": False}})
    return reach_rank({"status": "feasible", "reach_source": "declared_reach_m",
                       "own_arm": {"status": "flagged", "advisories": list(reasons)}})


def flagged_plan(count=1, **options):
    """A plan whose first row's scripted carry found no path, and whose second (when planned) is a humanoid row's flags."""
    value = make_plan(count=count, **options)
    value["candidates"][0]["reach_rank"] = flagged_rank(CARRY)
    if count > 1:
        row = value["candidates"][1]
        row["execution_spec"]["setup_quality_flags"] = list(G1_FLAGS)
        row["admission_digest"] = canonical_digest(row["execution_spec"])
    return validate_screening_plan(seal(value, "plan_digest"))


def completed(root, value):
    """One campaign whose only candidate finished, as the coordinator records it; nothing delivered yet."""
    coordinator.enqueue_screening_campaign(value, queue_root=root)
    backend = Backend({"status": "completed", "provider_allocation_performed": True, "provider_zero_verified": True,
                       "result": native_result(value)})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=backend)
    return value["campaign_id"]


def state_of(root, campaign_id):
    return json.loads((root / coordinator.LEDGER_FILENAME).read_text())["campaigns"][campaign_id]


class Website:
    """The WebApp's publication endpoint, restated: the versions it parses and its replay rules.

    ``typed`` is the release that quarantines a payload it cannot parse and answers 422
    ``schema_not_accepted``; without it, today's release answers a bare 400 and keeps nothing.
    """

    def __init__(self, accepts=("v1", "v2", "v3"), typed=True):
        self.accepts, self.typed = accepts, typed
        self.held, self.received, self.quarantined = {}, [], []

    def __call__(self, publication):
        self.received.append(publication)
        assert publication["publication_digest"] == cross_runtime_canonical_digest(
            publication, digest_field="publication_digest")
        if publication["schema_version"].rsplit(".", 1)[-1] not in self.accepts:
            if not self.typed:
                raise worker.PublicationRefused(400, "screening_publication_invalid")
            self.quarantined.append(publication["publication_digest"])
            raise worker.PublicationRefused(422, worker.SCHEMA_NOT_ACCEPTED)
        previous = self.held.get(publication["campaign_id"])
        if previous is not None and previous["publication_digest"] != publication["publication_digest"]:
            if publication["revision"] == previous["revision"]:
                raise worker.PublicationRefused(409, worker.REVISION_CONFLICT)  # one revision, two digests
            assert publication["revision"] > previous["revision"]
        self.held[publication["campaign_id"]] = publication


def test_scene_check_reads_the_plans_reach_evidence_in_a_closed_vocabulary():
    from blueprint_pipeline.robot_arm_path_clearance import SCRIPTED_CARRY_FLAG, reach_insufficient_code

    assert SCRIPTED_CARRY == SCRIPTED_CARRY_FLAG  # the packet gate's own flag, restated without its imports
    # The packet arm's scripted carry found no path: the flag and the phase it stopped at.
    assert scene_check({"status": "selected", "reach_rank": flagged_rank(CARRY)}) == ("flagged", [CARRY, SCRIPTED_CARRY])
    # A struck link's name stays in the plan; an unlisted reason is not published at all.
    struck = flagged_rank("robot_rigid_pick_place_scene_strike:panda_link5:carry", "packet_station_search_note")
    assert scene_check({"reach_rank": struck}) == ("flagged", ["robot_rigid_pick_place_scene_strike", SCRIPTED_CARRY])
    own = flagged_rank("robot_arm_task_unreached:arm:place:no_solution:ik_not_converged",
                       "robot_grasp_orientation_unbound:position_only", source="own")
    assert scene_check({"reach_rank": own}) == ("flagged", ["arm_reach_solve_flagged", "robot_arm_task_unreached"])
    # A measured shortfall is published without its distances.
    refused = {"status": "infeasible", "blockers": [reach_insufficient_code("place", 1.086, 0.990)]}
    assert scene_check(refused) == ("flagged", ["robot_reach_insufficient"])
    assert scene_check({"status": "infeasible", "blockers": ["screening_gpu_memory_floor_unrentable"]}) == ("unknown", [])
    passed = reach_rank({"status": "feasible", "station_search": {"status": "clear", "blockers": []}})
    assert scene_check({"reach_rank": passed}) == ("clear", [])
    # Nothing measured, and a plan written before rows recorded their reach: unknown stays unknown.
    assert scene_check({"reach_rank": reach_rank(None)}) == scene_check({"status": "selected"}) == ("unknown", [])
    published = {code for row in (struck, own, flagged_rank(CARRY)) for code in scene_check({"reach_rank": row})[1]}
    assert published <= SCENE_CHECK_FLAGS and all(FLAG_CODE.fullmatch(code) for code in SCENE_CHECK_FLAGS)
    assert not any(character.isdigit() for code in SCENE_CHECK_FLAGS for character in code)

    rows = [{"reach_rank": passed}, {"status": "selected"}]
    assert scene_feasibility(rows) == {"status": "clear", "flags": []}
    assert scene_feasibility([*rows, refused, {"reach_rank": flagged_rank(CARRY)}]) == {
        "status": "flagged", "flags": ["robot_reach_insufficient", CARRY, SCRIPTED_CARRY]}
    assert scene_feasibility([{"status": "selected"}]) == {"status": "unknown", "flags": []}


def test_only_typed_codes_are_published_and_an_empty_list_is_left_out():
    assert typed_flags(["b_code", "a:b:c", "b_code", "Free text: /workspace/secret", 7]) == ["a:b:c", "b_code", UNTYPED]
    assert typed_flags(None) == typed_flags("a_code") == []
    configured = {"candidate_id": "g1", "execution_spec": {"setup_quality_flags": list(G1_FLAGS)},
                  "reach_rank": flagged_rank(CARRY)}
    assert row_flags(configured, {"observation_quality_flags": ["droid_wrist_task_pixels_below_threshold"]}) == {
        "observation_quality_flags": ["droid_wrist_task_pixels_below_threshold"], "setup_quality_flags": G1_FLAGS,
        "scene_check_flags": [CARRY, SCRIPTED_CARRY]}
    # A humanoid row's flags are in its plan before it runs; a row with nothing recorded carries no field.
    assert row_flags({"candidate_id": "g1", "execution_spec": {"setup_quality_flags": list(G1_FLAGS)}}, {}) == {
        "setup_quality_flags": G1_FLAGS}
    assert row_flags({"candidate_id": "plain"}, {"episodes_scored": 10}) == {}


def test_a_run_with_observation_flags_in_a_flagged_scene_publishes_v3_carrying_both(tmp_path):
    from tests.test_beta_screening_worker_control import REAL_GATE_BLOCKERS

    value = flagged_plan(count=2)
    first, second = (row["candidate_id"] for row in value["candidates"])
    delivered, rows, _ = run_campaign(tmp_path / "queue", value, {
        first: {"successes": 0, "flagged": True}, second: {"successes": 0}})
    final = delivered[-1]
    assert PUBLICATION_SCHEMA == PUBLICATION_SCHEMA_V3 == final["schema_version"]
    assert final["status"] == "completed"
    assert [(row["episodes_scored"], row["episodes_succeeded"]) for row in final["candidates"]] == [(10, 0), (10, 0)]
    # Zero successes, and everything Blueprint's own checks knew about the scene, beside them.
    assert final["scene_feasibility"] == {"status": "flagged", "flags": [CARRY, SCRIPTED_CARRY]}
    assert final["candidates"][0]["observation_quality_flags"] == sorted(REAL_GATE_BLOCKERS)
    assert final["candidates"][0]["scene_check_flags"] == [CARRY, SCRIPTED_CARRY]
    assert final["candidates"][1]["setup_quality_flags"] == G1_FLAGS
    assert "observation_quality_flags" not in final["candidates"][1] and "scene_check_flags" not in final["candidates"][1]
    assert rows[first]["summary"]["observation_quality_flags"] == final["candidates"][0]["observation_quality_flags"]
    # The plan's verdict is the same in every revision; a run's flags come with its result.
    assert all(publication["scene_feasibility"] == final["scene_feasibility"] for publication in delivered)
    assert "observation_quality_flags" not in delivered[0]["candidates"][0]
    schema = json.loads((ROOT / "docs/schemas/beta_policy_screening_publication.v3.schema.json").read_text())
    for publication in delivered:
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(publication)
        _webapp_contract(publication)
        assert publication_flags_valid(publication)
        # The same result without the flags is the v2 a website that does not accept v3 yet is sent.
        _webapp_contract(as_v2(publication))
        assert as_v2(publication)["schema_version"] == PUBLICATION_SCHEMA_V2
        assert "flags" not in json.dumps(as_v2(publication))
    facts = ("candidate_id", "status", "episodes_scored", "episodes_succeeded", "episodes_failed", "success_rate")
    assert [{key: row[key] for key in facts} for row in as_v2(final)["candidates"]] == [
        {key: row[key] for key in facts} for row in final["candidates"]]
    assert as_v2(final)["budget"] == final["budget"]

    if os.environ.get("BLUEPRINT_WRITE_SCREENING_PUBLICATION_FIXTURES") == "1":
        FIXTURE.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    retained = json.loads(FIXTURE.read_text())
    assert _shape(retained) == _shape(final)
    flag_fields = ("observation_quality_flags", "setup_quality_flags", "scene_check_flags")
    assert retained["scene_feasibility"] == final["scene_feasibility"]
    assert [{key: row.get(key) for key in (*facts, *flag_fields)} for row in retained["candidates"]] == [
        {key: row.get(key) for key in (*facts, *flag_fields)} for row in final["candidates"]]


def test_the_v3_fixture_shared_with_the_webapp_is_valid_and_unchanged():
    raw = FIXTURE.read_bytes()
    value = json.loads(raw)
    schema = json.loads((ROOT / "docs/schemas/beta_policy_screening_publication.v3.schema.json").read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
    _webapp_contract(value)
    assert value["publication_digest"] == cross_runtime_canonical_digest(value, digest_field="publication_digest")
    # Change the fixture in both repositories together, and this value in both tests.
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256


def test_a_ledger_written_before_v3_still_publishes_and_its_acknowledged_bytes_stand(tmp_path):
    # A plan without reach evidence and a summary without flags, acknowledged as v2 by the release before this one.
    value = make_plan()
    campaign_id = completed(tmp_path, value)
    ledger_path = tmp_path / coordinator.LEDGER_FILENAME
    ledger = json.loads(ledger_path.read_text())
    state = ledger["campaigns"][campaign_id]
    assert DOWNGRADE_FIELD not in state and "observation_quality_flags" not in state["candidates"]["candidate-000"]["summary"]
    budget = coordinator._published_budget(ledger, state)
    legacy = as_v2(screening_publication(plan=value, state=state, budget=budget))
    state.update(published_revision=state["revision"], published_digest=legacy["publication_digest"])
    ledger_path.write_text(json.dumps(seal(ledger, "ledger_digest")))

    # Nothing is sent again and the revision does not move: the website holds exactly these bytes.
    website = Website()
    assert worker.deliver_publications(queue_root=tmp_path, transport=website) == {"acknowledged": 0, "pending": 0}
    assert website.received == [] and state_of(tmp_path, campaign_id)["revision"] == state["revision"]
    current = state_of(tmp_path, campaign_id)
    assert screening_publication(plan=value, state=current, budget=budget) == legacy

    # Its next revision (here: the campaign changed) is published as v3, unknown where nothing was recorded.
    with coordinator._lock(tmp_path):
        ledger = coordinator._ledger(tmp_path)
        coordinator._changed(ledger["campaigns"][campaign_id])
        coordinator._save(tmp_path, ledger)
    website.held[campaign_id] = legacy
    assert worker.deliver_publications(queue_root=tmp_path, transport=website) == {"acknowledged": 1, "pending": 0}
    [sent] = website.received
    assert sent["schema_version"] == PUBLICATION_SCHEMA_V3 and sent["revision"] == legacy["revision"] + 1
    assert sent["scene_feasibility"] == {"status": "unknown", "flags": []}
    assert as_v2(sent)["candidates"] == legacy["candidates"]


@pytest.mark.parametrize("typed, reason", [(False, "http_400"), (True, "schema_not_accepted")])
def test_pipeline_deployed_first_delivers_every_result_as_v2_and_records_the_downgrade(tmp_path, typed, reason):
    # The website does not accept v3 yet: today's release (a bare 400), or one that already answers the typed refusal.
    website = Website(accepts=("v1", "v2"), typed=typed)
    value = flagged_plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    campaign_id = value["campaign_id"]
    assert worker.deliver_publications(queue_root=tmp_path, transport=website, refresh_startable=True) == {
        "acknowledged": 1, "pending": 0, "downgraded_to_v2": 1}
    offered, queued = website.received
    assert (offered["schema_version"], queued["schema_version"]) == (PUBLICATION_SCHEMA_V3, PUBLICATION_SCHEMA_V2)
    assert queued == as_v2(offered) and queued["revision"] == offered["revision"]
    record = state_of(tmp_path, campaign_id)[DOWNGRADE_FIELD]
    assert record == {"revision": offered["revision"], "from_schema_version": PUBLICATION_SCHEMA_V3,
                      "to_schema_version": PUBLICATION_SCHEMA_V2, "reason": reason,
                      "refused_publication_digest": offered["publication_digest"], "recorded_at": record["recorded_at"]}
    assert json.loads((tmp_path / "publications" / f"{campaign_id}.json").read_text()) == queued

    # The acknowledgment of the v2 bytes is what lets the paid run start; its result is delivered the same way.
    backend = Backend({"status": "completed", "provider_allocation_performed": True, "provider_zero_verified": True,
                       "result": native_result(value)})
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert backend.launches == 1
    assert worker.deliver_publications(queue_root=tmp_path, transport=website)["downgraded_to_v2"] == 1
    result = website.held[campaign_id]
    assert result["schema_version"] == PUBLICATION_SCHEMA_V2 and result["status"] == "completed"
    row = result["candidates"][0]
    assert (row["episodes_scored"], row["episodes_succeeded"], row["success_rate"]) == (10, 7, 0.7)
    final = state_of(tmp_path, campaign_id)
    assert final["published_revision"] == final["revision"] == result["revision"]
    assert final["published_digest"] == result["publication_digest"]
    assert final[DOWNGRADE_FIELD]["revision"] == result["revision"]  # each revision was offered as v3 first

    # Nothing is left to send, and a later pass sends nothing twice: no result waits on the website's release.
    before = len(website.received)
    assert worker.deliver_publications(queue_root=tmp_path, transport=website) == {"acknowledged": 0, "pending": 0}
    assert coordinator.pending_publications(queue_root=tmp_path) == [] and len(website.received) == before
    assert len(website.quarantined) == (2 if typed else 0)  # one refusal per v3 revision, never a resend loop

    # Once the website accepts v3, the campaign's next revision carries the flags.
    website.accepts = ("v1", "v2", "v3")
    with coordinator._lock(tmp_path):
        ledger = coordinator._ledger(tmp_path)
        coordinator._changed(ledger["campaigns"][campaign_id])
        coordinator._save(tmp_path, ledger)
    assert worker.deliver_publications(queue_root=tmp_path, transport=website) == {"acknowledged": 1, "pending": 0}
    assert website.held[campaign_id]["schema_version"] == PUBLICATION_SCHEMA_V3
    assert website.held[campaign_id]["scene_feasibility"]["status"] == "flagged"


def test_webapp_deployed_first_takes_v3_and_a_revision_it_holds_as_v2_is_never_rewritten(tmp_path):
    # The website accepts v3 while the Pipeline's earlier release is still sending v2.
    website = Website()
    value = flagged_plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    campaign_id = value["campaign_id"]
    [first] = coordinator.pending_publications(queue_root=tmp_path)
    earlier = as_v2(first)  # what the release before this one sent for this revision
    website(earlier)
    website.received.clear()

    # This release starts with that revision unacknowledged (a restart re-checks every startable campaign). Its v3
    # form is the same revision with another digest, which the website refuses: the v2 bytes are replayed instead.
    assert worker.deliver_publications(queue_root=tmp_path, transport=website, refresh_startable=True) == {
        "acknowledged": 1, "pending": 0, "downgraded_to_v2": 1}
    assert [row["schema_version"] for row in website.received] == [PUBLICATION_SCHEMA_V3, PUBLICATION_SCHEMA_V2]
    assert website.received[-1] == earlier == website.held[campaign_id]
    assert state_of(tmp_path, campaign_id)[DOWNGRADE_FIELD]["reason"] == worker.REVISION_CONFLICT
    # Every later pass re-checks the startable campaign with the same bytes, and the candidate can start.
    assert worker.deliver_publications(queue_root=tmp_path, transport=website, refresh_startable=True) == {
        "acknowledged": 1, "pending": 0}
    assert website.received[-1] == earlier

    backend = Backend({"status": "completed", "provider_allocation_performed": True, "provider_zero_verified": True,
                       "result": native_result(value)})
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert backend.launches == 1
    # The result is a new revision: published as v3, with the scene's flags, and accepted as such.
    assert worker.deliver_publications(queue_root=tmp_path, transport=website) == {"acknowledged": 1, "pending": 0}
    result = website.held[campaign_id]
    assert result["schema_version"] == PUBLICATION_SCHEMA_V3 and result["status"] == "completed"
    assert result["scene_feasibility"] == {"status": "flagged", "flags": [CARRY, SCRIPTED_CARRY]}
    assert result["candidates"][0]["scene_check_flags"] == [CARRY, SCRIPTED_CARRY]
    assert result["candidates"][0]["episodes_scored"] == 10
    assert website.quarantined == [] and coordinator.pending_publications(queue_root=tmp_path) == []


@pytest.mark.parametrize("refusal", [
    worker.PublicationRefused(409, "screening_scene_intake_unavailable"), worker.PublicationRefused(409),
    worker.PublicationRefused(401, "invalid_pipeline_sync_signature"), worker.PublicationRefused(422),
    worker.PublicationRefused(503, "screening_store_unavailable"), OSError("temporary network failure"),
    ValueError("beta_screening_publication_ack_invalid")])
def test_a_refusal_that_is_not_about_the_schema_never_downgrades(tmp_path, refusal):
    campaign_id = completed(tmp_path, flagged_plan())
    received = []

    def refuse(publication):
        received.append(publication)
        raise refusal
    summary = worker.deliver_publications(queue_root=tmp_path, transport=refuse)
    assert (summary["acknowledged"], summary["pending"]) == (0, 1) and "downgraded_to_v2" not in summary
    assert [row["schema_version"] for row in received] == [PUBLICATION_SCHEMA_V3]
    assert DOWNGRADE_FIELD not in state_of(tmp_path, campaign_id)
    # It stays the v3 it was, for the next pass.
    assert coordinator.pending_publications(queue_root=tmp_path) == received


def test_a_v2_publication_the_website_refuses_is_not_sent_a_third_time_in_the_pass(tmp_path):
    campaign_id = completed(tmp_path, flagged_plan())
    received = []

    def refuse(publication):
        received.append(publication)
        raise worker.PublicationRefused(422, worker.SCHEMA_NOT_ACCEPTED)
    summary = worker.deliver_publications(queue_root=tmp_path, transport=refuse)
    assert (summary["acknowledged"], summary["pending"]) == (0, 1)
    assert [row["schema_version"] for row in received] == [PUBLICATION_SCHEMA_V3, PUBLICATION_SCHEMA_V2]
    # The revision stays v2 and pending: the next pass offers the same v2 bytes, never the v3 form again.
    assert state_of(tmp_path, campaign_id)[DOWNGRADE_FIELD]["revision"] == received[0]["revision"]
    assert coordinator.pending_publications(queue_root=tmp_path) == [received[1]]


def test_signed_delivery_reads_the_typed_refusal_without_its_text(monkeypatch):
    from blueprint_pipeline import safe_outbound_http, task_evaluation_launch_webapp_sync

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://app.example.test/api/internal/pipeline/sync")
    monkeypatch.setattr(task_evaluation_launch_webapp_sync, "load_pipeline_sync_token", lambda: "unit-test-key")
    publication = {"schema_version": PUBLICATION_SCHEMA_V3, "campaign_id": "campaign1", "revision": 4,
                   "publication_digest": "sha256:" + "e" * 64}
    for status, body, code, reason in (
            (422, {"code": "schema_not_accepted", "error": "Schema not accepted: secret detail"},
             "schema_not_accepted", "schema_not_accepted"),
            (400, {"code": "screening_publication_invalid"}, "screening_publication_invalid", "http_400"),
            (409, {"code": "screening_revision_conflict"}, "screening_revision_conflict", "screening_revision_conflict"),
            (409, {"code": "screening_scene_source_mismatch"}, "screening_scene_source_mismatch", None),
            (502, "<html>Bad gateway</html>", None, None)):
        def send(url, _status=status, _body=body, **_):
            raw = (_body if isinstance(_body, str) else json.dumps(_body)).encode()
            raise HTTPError(url, _status, "refused", {}, io.BytesIO(raw))
        monkeypatch.setattr(safe_outbound_http, "request", send)
        with pytest.raises(worker.PublicationRefused) as refused:
            worker.publish_screening(publication)
        assert (refused.value.status, refused.value.code) == (status, code)
        assert str(refused.value) == f"beta_screening_publication_http_{status}" and "secret" not in str(refused.value)
        assert worker.v3_not_accepted(publication, refused.value) == reason
        # The v2 form has nothing to fall back to.
        assert worker.v3_not_accepted({**publication, "schema_version": PUBLICATION_SCHEMA_V2}, refused.value) is None
