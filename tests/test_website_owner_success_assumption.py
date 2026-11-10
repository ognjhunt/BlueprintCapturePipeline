# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_owner_success_assumption.py
#   src/blueprint_pipeline/team_scene_packet.py
#   src/blueprint_pipeline/team_scene_packet_worker.py
#   src/blueprint_pipeline/team_scene_packet_remote.py
#   src/blueprint_pipeline/website_owner_success_confirmation.py
#   src/blueprint_pipeline/task_evaluation_rigid_owner_contract.py
"""A website job is scored on Blueprint's standard thresholds at once, marked as assumed; the site can still confirm.

The job, its published revision and its configured bundle are the real proposal_only
fixture the owner-confirmation tests compile. The WebApp is a fake that keeps the
proposal and answers with the owner's decision, so nothing reaches the network.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as source
from blueprint_pipeline import task_evaluation_owner_success_confirmation as confirmation
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import team_scene_packet_remote as lane
from blueprint_pipeline import team_scene_packet_worker as worker
from blueprint_pipeline import website_owner_success_assumption as assumption
from blueprint_pipeline import website_owner_success_confirmation as website
from blueprint_pipeline.beta_policy_screening_scene_half import current_half
from blueprint_pipeline.team_candidate_quick10_registry import REGISTRY_ENV, profile_for_capture, scene_facts
from blueprint_pipeline.team_scene_packet import TeamScenePacketError, read_team_scene_inputs
from tests.test_task_evaluation_owner_success_confirmation import confirm
from tests.test_team_scene_packet import _scene
from tests.test_team_scene_packet import _compile as _compile_default
from tests.test_team_scene_packet_owner_confirmation import REFUSED, _compile, _job
from tests.test_team_scene_packet_worker import _registry, _sealed
from tests.test_website_owner_success_confirmation import FakeWebApp, _core

REQUEST_ID = "job-one"
SCENE_ID, CAPTURE_ID = f"site-{REQUEST_ID}", f"walkthrough-{REQUEST_ID}"
RECEIPT_DIGEST = "sha256:" + "e" * 64
RELEASE_1, RELEASE_2 = "1" * 40, "2" * 40
ASSUME = {"assume_unconfirmed_standard_thresholds": True}


def _record(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _plan(compiled: dict[str, Any]) -> dict[str, Any]:
    return _record(Path(compiled["path"]).parent / "native-task-packet" / "native_task_arena_scene_plan.v1.json")


def test_an_unconfirmed_website_job_compiles_at_once_on_the_same_thresholds_marked_as_assumed(tmp_path, monkeypatch):
    from blueprint_pipeline.adp_articulated_task_success_contract import validate_task_success_contract
    from blueprint_pipeline.beta_policy_screening_control_omission import omit_paused_controls
    from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import _require_strict_owner_success_contract

    job, inputs = _job(tmp_path, monkeypatch)
    compiled = _compile(job, {**inputs, **ASSUME}, tmp_path / "assumed")
    # Nothing was confirmed and nothing waited: the half is compiled, and it says what it assumed.
    assert not (job.directory / confirmation.DIRECTORY).exists() and "owner_success_confirmation" not in compiled
    marker = compiled[assumption.FIELD]
    assert marker["status"] == assumption.ASSUMED == "blueprint_standard_thresholds_assumed_not_confirmed_by_site"
    record = _record(compiled["path"])
    assert record[assumption.FIELD] == marker == assumption.half_assumption(compiled["path"])
    contract = read_team_scene_inputs(compiled["path"])["task_success_contract"]
    assert marker["task_success_contract_digest"] == contract["contract_digest"]
    # The contract names Blueprint, never the site, as who stands behind the thresholds.
    provenance = contract["provenance"]
    assert provenance["author_source"] == "agent_proposal" and provenance["author_id"] == assumption.AUTHOR_ID
    assert provenance["confirmed_by_team_id"] == assumption.ASSUMED_BY == "blueprint:assumed_not_confirmed_by_site"
    assert assumption.contract_assumed(contract)
    # The task keeps its proposal_only authority as built: nothing is recorded as the owner's acceptance.
    plan = _plan(compiled)
    assert plan["task_spec"]["configured_owner_authority"]["confirmation_status"] == "proposal_only"
    assert plan["task_spec"]["task_success_contract"] == contract
    # The digest a confirmation of these thresholds is bound to is the one the WebApp's proposal carries.
    published = confirmation.published_job(job.directory, job.intent)
    notice = website.proposal_notice(confirmation.binding(
        intent=job.intent, revision=published["revision"], criteria=published["criteria"],
        proposal=published["proposal"]))
    assert marker["binding_digest"] == notice["binding_digest"]
    assert marker["configured_success_criteria_digest"] == notice["configured_success_criteria_digest"]
    assert marker["proposal_authority_digest"] == notice["proposal_authority_digest"]
    # Every step after the packet admits it: the scorer's validator, the strict owner check, the scene's
    # facts as a team run stages them, and the screening contract under the standing controls pause.
    validate_task_success_contract(contract, task_kind="rigid_pick_place")
    _require_strict_owner_success_contract(task_spec=plan["task_spec"], contract=contract)
    facts = scene_facts({"team_scene_inputs_path": compiled["path"], "open_loop_horizon": 8})
    assert facts["inputs"]["task_success_contract"] == contract
    screened, _omission = omit_paused_controls(contract)
    assert assumption.contract_assumed(screened) and screened["provenance"] == provenance

    # The same job once its site confirmed: the very same criteria, the site's provenance, and no marker.
    applied = confirm(job)
    confirmed = _compile(job, {**inputs, **ASSUME}, tmp_path / "confirmed")
    assert assumption.FIELD not in confirmed and assumption.FIELD not in _record(confirmed["path"])
    assert assumption.half_assumption(confirmed["path"]) is None
    owners = read_team_scene_inputs(confirmed["path"])["task_success_contract"]
    assert owners["criteria"] == contract["criteria"] and owners["scope"] == contract["scope"]
    assert owners["provenance"]["author_source"] == "task_owner" and not assumption.contract_assumed(owners)
    assert owners["contract_digest"] != contract["contract_digest"]
    # A result scored under the assumption stays bound to the assumed contract: confirming never rewrites it.
    assert read_team_scene_inputs(compiled["path"])["task_success_contract"] == contract
    assert assumption.half_assumption(compiled["path"]) == marker
    assert applied["confirmation_digest"] == confirmed["owner_success_confirmation"]
    # A half of a revision that already assumed keeps that contract when compiled again: the confirmation
    # made since is recorded beside the assumption, and no digest changes.
    kept = _compile(job, {**inputs, **ASSUME, "keep_assumed_thresholds": True}, tmp_path / "kept")
    assert read_team_scene_inputs(kept["path"])["task_success_contract"] == contract
    assert "owner_success_confirmation" not in kept
    assert kept[assumption.FIELD] == {**marker, "site_answer": "confirmed", "confirmed_later": {
        "confirmation_digest": applied["confirmation_digest"], "confirmed_at": applied["confirmed_at"],
        "issuer_channel": "operator_door"}}


def test_a_job_its_site_already_confirmed_compiles_exactly_as_before_with_no_marker(tmp_path, monkeypatch):
    job, inputs = _job(tmp_path, monkeypatch)
    applied = confirm(job)
    before = _compile(job, inputs, tmp_path / "before")
    after = _compile(job, {**inputs, **ASSUME}, tmp_path / "after")
    assert before["owner_success_confirmation"] == after["owner_success_confirmation"] == applied["confirmation_digest"]
    assert assumption.FIELD not in after and assumption.FIELD not in _record(after["path"])
    halves = [read_team_scene_inputs(row["path"]) for row in (before, after)]
    assert halves[0]["task_success_contract"] == halves[1]["task_success_contract"]
    assert halves[0]["plan_digest"] == halves[1]["plan_digest"]
    assert set(_record(before["path"])) == set(_record(after["path"]))


def test_an_operator_or_arm_decision_proof_compile_is_exactly_as_before(tmp_path, monkeypatch):
    # Without the website worker's flag a proposal_only authority refuses, scoped or not, as it always has.
    for name in ("owner", "default"):
        (tmp_path / name).mkdir()
    job, inputs = _job(tmp_path / "owner", monkeypatch)
    for name, scope in (("unscoped", False), ("scoped", True)):
        with pytest.raises(TeamScenePacketError, match=f"\\A{REFUSED}\\Z"):
            _compile(job, inputs, tmp_path / name, scope=scope)
    # A task that requires no owner contract takes the compatibility default, with or without the flag.
    scene = _scene(tmp_path / "default")
    plain, _request = _compile_default(tmp_path / "plain", scene)
    flagged, _request = _compile_default(tmp_path / "flagged", {**scene, **ASSUME})
    halves = [read_team_scene_inputs(row["path"]) for row in (plain, flagged)]
    assert halves[0]["task_success_contract"] == halves[1]["task_success_contract"]
    assert halves[0]["task_success_contract"]["provenance"]["author_source"] == "compatibility_default"
    assert halves[0]["plan_digest"] == halves[1]["plan_digest"]
    assert assumption.FIELD not in flagged and assumption.FIELD not in _record(flagged["path"])
    assert not assumption.contract_assumed(halves[0]["task_success_contract"])


def test_a_confirmation_of_other_thresholds_confirms_nothing_about_these(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_intent_contracts import canonical_digest
    from tests.test_task_evaluation_owner_success_confirmation import rethreshold

    job, inputs = _job(tmp_path, monkeypatch)
    confirm(job)
    # The same intent republished with one threshold changed: the owner confirmed the old one, not this.
    configured, references = rethreshold(job.revision, job.references, tmp_path / "owner-case",
                                         maximum_task_contact_force_n=50.0)
    job.store.update({row["uri"]: Path(row["materialized_path"]).read_bytes() for row in references.values()})
    compiled = _compile(job, {**inputs, **ASSUME}, tmp_path / "changed", revision=configured)
    marker = compiled[assumption.FIELD]
    assert marker["status"] == assumption.ASSUMED and "owner_success_confirmation" not in compiled
    contract = read_team_scene_inputs(compiled["path"])["task_success_contract"]
    assert assumption.contract_assumed(contract)
    assert contract["criteria"]["temporal_invariants"]["maximum_task_contact_force_n"] == 50.0
    [row] = confirmation.confirmations(job.directory, job.intent)
    assert canonical_digest({field: row[field] for field in confirmation._BINDING_FIELDS}) != marker["binding_digest"]
    # So the half is not compiled again for it: the kept confirmation is of other thresholds.
    monkeypatch.setattr(worker, "owner_success_confirmation_scope",
                        lambda _request: ({"intent": job.intent, "directory": job.directory}, None))
    assert assumption.site_confirmation(compiled["path"], {}) is None
    status = {"status": "completed", "team_scene_inputs_path": compiled["path"], assumption.FIELD: marker}
    assert assumption.status_assumed(status) and assumption.noted_status(status, {}, exchange=None) is None


def test_thresholds_that_cannot_be_sealed_still_refuse_by_name(tmp_path, monkeypatch):
    from tests.test_task_evaluation_owner_success_confirmation import rethreshold

    job, inputs = _job(tmp_path, monkeypatch)
    # The success document and the task disagree on a limit the scorer reads: assumed or not, it refuses.
    configured, references = rethreshold(job.revision, job.references, tmp_path / "owner-case", minimum_lift_m=0.2)
    job.store.update({row["uri"]: Path(row["materialized_path"]).read_bytes() for row in references.values()})
    with pytest.raises(TeamScenePacketError, match="\\Ateam_scene_"):
        _compile(job, {**inputs, **ASSUME}, tmp_path / "changed", revision=configured)


def test_an_unconfirmed_drawer_job_compiles_on_its_thresholds_marked_as_assumed(tmp_path, monkeypatch):
    from blueprint_pipeline.adp_articulated_task_success_contract import validate_task_success_contract
    from tests import test_task_evaluation_articulated_open_close_native_adapter as drawer
    from tests.test_team_scene_packet import _articulated_scene, _sha

    for name in ("bare", "proposal", "owner"):
        (tmp_path / name).mkdir()
    # A task with no owner authority at all is not a proposal a site could confirm: refused, flag or not.
    bare = _articulated_scene(tmp_path / "bare", owner_confirmed=False)
    with pytest.raises(TeamScenePacketError, match="team_scene_owner_contract_invalid"):
        _compile_default(tmp_path / "bare-out", {**bare, **ASSUME})
    built = drawer._case

    def proposal_only(root: Path) -> dict[str, Any]:
        """The drawer job as the website publishes it: Blueprint's thresholds, proposal_only."""

        case = built(root)
        reference = case["references"][drawer.DEFINITION]
        template = json.loads(Path(reference["materialized_path"]).read_text())
        template["owner_success_contract_authority"] = {
            "confirmation_status": "proposal_only", "accepted_by": "owner-1",
            "authority_reference": "scene-intent:sha256:" + "f" * 64,
            "basis": "fixed_development_control_not_translated_owner_target", "scorer_translation_verified": False}
        payload = (json.dumps(template, sort_keys=True) + "\n").encode()
        Path(reference["materialized_path"]).write_bytes(payload)
        reference.update(digest=_sha(payload), size_bytes=len(payload))
        case["configured"]["task_template"]["definition"] = {key: reference[key]
                                                             for key in ("uri", "digest", "size_bytes")}
        return case

    monkeypatch.setattr(drawer, "_case", proposal_only)
    scene = _articulated_scene(tmp_path / "proposal", owner_confirmed=False)
    monkeypatch.setattr(drawer, "_case", built)
    with pytest.raises(TeamScenePacketError, match="team_scene_owner_contract_invalid"):
        _compile_default(tmp_path / "refused", scene)
    compiled, _request = _compile_default(tmp_path / "assumed", {**scene, **ASSUME})
    contract = validate_task_success_contract(read_team_scene_inputs(compiled["path"])["task_success_contract"],
                                              task_kind="articulated_open_close")
    assert assumption.contract_assumed(contract) and compiled[assumption.FIELD]["status"] == assumption.ASSUMED
    assert "binding_digest" not in compiled[assumption.FIELD]  # no intent was bound to look a confirmation up in
    # The same criteria the owner's confirmed drawer contract seals.
    owners, _request = _compile_default(tmp_path / "confirmed", _articulated_scene(tmp_path / "owner"))
    confirmed = read_team_scene_inputs(owners["path"])["task_success_contract"]
    assert confirmed["criteria"] == contract["criteria"] and assumption.FIELD not in owners


def _world(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    """The published proposal_only job as the host keeps it, with its capture, registry and queued request."""

    job, inputs = _job(tmp_path, monkeypatch)
    revision_path = job.run / "configured_scene_revision.v1.json"
    payload = revision_path.read_bytes()
    reference = _record(job.directory / "progression.json")["result_reference"]
    job.store[reference["uri"]] = payload
    shutil.copyfile(inputs["configured_scene_bundle_path"], job.run / worker.CONFIGURED_BUNDLE_FILENAME)
    result = {"configured_scene_revision_reference": reference,
              "configured_scene_revision_digest": job.revision["revision_digest"],
              "configured_scene_revision_path": str(revision_path)}
    monkeypatch.setattr(source, "_validate_source",
                        lambda root: (dict(result), {"receipt_digest": RECEIPT_DIGEST}, {}))
    partition = tmp_path / "partition"
    capture = partition / "scenes" / SCENE_ID / "captures" / CAPTURE_ID
    (capture / "pipeline").mkdir(parents=True)
    (capture / "capture_descriptor.json").write_text(json.dumps({
        "scene_id": SCENE_ID, "capture_id": CAPTURE_ID, "site_submission_id": REQUEST_ID,
        "metadata": {"capture_entry_source": "browser_self_capture", "site_submission_id": REQUEST_ID}}))
    context = _sealed({"schema_version": "website_site_task_context.v1", "request_id": REQUEST_ID,
                       "scene_id": SCENE_ID, "capture_id": CAPTURE_ID, "description": "Move the box",
                       "confirmed": False, "confirmed_at": None}, "context_digest")
    (capture / "pipeline" / "website_task_context.json").write_text(json.dumps(context))
    monkeypatch.setenv(worker.CAPTURE_PARTITION_ENV, str(partition))
    monkeypatch.setenv("BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE", str(tmp_path / "rigid-robot-free-probes"))
    operator = tmp_path / "operator"
    operator.mkdir()
    (operator / "salt.key").write_text("k" * 48)
    (operator / "salt.key").chmod(0o600)
    _registry(operator, packet_inputs={
        "droid_profile_path": str(inputs["droid_profile_path"]), "droid_profile_digest": inputs["droid_profile_digest"],
        "runtime_source_receipt_path": str(inputs["runtime_source_receipt_path"]),
        "output_root": str(tmp_path / "team-scene-halves")})
    monkeypatch.setenv(REGISTRY_ENV, str(operator / "registry.json"))
    monkeypatch.setattr(worker, "current_release_commit", lambda: RELEASE_1)
    request = worker.validated_request(worker._seal({
        "schema_version": worker.REQUEST_SCHEMA, "intent_id": job.intent["intent_id"],
        "intent_digest": job.intent["intent_digest"], "capture_id": CAPTURE_ID,
        "task_id": worker.WEBSITE_TASK_PREFIX + context["context_digest"][7:27], "source_launch_root": str(job.run),
        "launch_receipt_digest": RECEIPT_DIGEST, "scene_revision_digest": job.revision["revision_digest"],
        "robot_free_readiness_status": "refused"}, "request_digest"))
    progression = _record(job.directory / "progression.json")
    progression["state"]["scene_preparation_completion"]["team_scene_request"]["request_digest"] = (
        request["request_digest"])
    (job.directory / "progression.json").write_text(json.dumps(intake._seal(progression, "progression_digest")))
    queue = worker._queue(tmp_path / "queue")
    key = worker.request_key(request)
    (queue / "pending" / f"{key}.json").write_text(json.dumps(request))

    def process() -> dict[str, Any]:
        return worker.process_team_scene_queue(queue_root=queue, fetcher=lambda row: job.store[row["uri"]],
                                               offer_transport=lambda **_: {"status": "accepted"})

    def screening_source() -> dict[str, Any]:
        """The source screening plans this job under, checked as it is before every paid attempt."""

        from blueprint_pipeline import beta_policy_screening_worker as screening
        from blueprint_pipeline import website_task_context

        retained = _record(capture / "pipeline" / "website_task_context.json")
        current = {**retained, "capture_rights": {"consent_status": "granted", "consent_revoked": False,
                                                  "consent_scope": ["robot_evaluation"],
                                                  "derived_scene_generation_allowed": True}}
        monkeypatch.setattr(website_task_context, "load_current_website_task_context", lambda **_: current)
        queued = _record(queue / "processed" / f"{key}.json")
        value = {"scene_intent_id": queued["intent_id"], "scene_intent_digest": queued["intent_digest"],
                 "capture_id": queued["capture_id"], "task_id": queued["task_id"],
                 "task_context_digest": retained["context_digest"],
                 "scene_revision_digest": queued["scene_revision_digest"]}
        # Raises beta_screening_current_task_context_changed when the job's context no longer binds it.
        screening.require_current_source(value, site_id=SCENE_ID)
        return value

    return {"job": job, "queue": queue, "key": key, "request": request, "capture": capture.resolve(),
            "process": process, "context_path": capture / "pipeline" / "website_task_context.json",
            "screening_source": screening_source}


def _status(world: dict[str, Any]) -> dict[str, Any]:
    return _record(world["queue"] / "status" / f"{world['key']}.json")


@pytest.fixture
def clock(monkeypatch):
    """Each pass is two days after the last, so every exchange is due."""

    state = {"now": 1_793_700_000.0}

    def tick() -> float:
        state["now"] += 2 * 86400.0
        return state["now"]

    monkeypatch.setattr(website, "_now", tick)


def test_the_worker_compiles_an_unconfirmed_job_without_waiting_and_a_later_confirmation_is_only_recorded(
        tmp_path, monkeypatch, clock):
    from blueprint_pipeline import team_scene_packet
    from blueprint_pipeline.rigid_robot_free_probe_bundle import current_probe_state

    world = _world(tmp_path, monkeypatch)
    job, queue, key = world["job"], world["queue"], world["key"]
    context_before = world["context_path"].read_bytes()
    [first] = world["process"]()["results"]
    # No confirmation exists, and the request did not wait for one: its half is compiled and offered.
    assert not (job.directory / confirmation.DIRECTORY).exists()
    assert (first["status"], first["outcome"], first["blockers"]) == ("completed", "compiled", [])
    assert first[assumption.FIELD]["status"] == assumption.ASSUMED and "owner_success_confirmation" not in first
    assert "site_answer" not in first[assumption.FIELD]  # the site has not answered
    assert _status(world) == first and (queue / "processed" / f"{key}.json").is_file()
    assumed_path = first["team_scene_inputs_path"]
    assumed = read_team_scene_inputs(assumed_path)["task_success_contract"]
    assert assumption.contract_assumed(assumed)
    profile = profile_for_capture(world["capture"])
    revision = job.revision["revision_digest"]
    assumed_half = current_half(profile, scene_revision_digest=revision)
    probe_before = current_probe_state(world["capture"])["request"]["request_digest"]
    source_before = world["screening_source"]()
    record_before = Path(assumed_path).read_bytes()

    # The timer's pass sends the thresholds to the job page, as it did for a blocked job; nothing moves.
    webapp = FakeWebApp()
    asked = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    assert asked["requeued"] == [] and webapp.calls == 1
    [notice] = webapp.proposals.values()
    assert notice["binding_digest"] == first[assumption.FIELD]["binding_digest"]
    assert _core(_status(world)[website.STATUS_FIELD]) == {"status": "pending",
                                                         "proposal_digest": notice["proposal_digest"]}
    assert "site_answer" not in _status(world)[assumption.FIELD]

    # The owner confirms on the job page: the confirmation is sealed and RECORDED beside the assumption.
    webapp.decide("confirmed")
    monkeypatch.setattr(team_scene_packet, "prepare_team_scene", lambda **_: pytest.fail("compiled again"))
    confirmed = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    [row] = confirmation.confirmations(job.directory, job.intent)
    assert confirmed["requeued"] == [] and not list((queue / "pending").iterdir())
    status = _status(world)
    assert status["status"] == "completed" and status["team_scene_inputs_path"] == assumed_path
    assert status[assumption.FIELD] == {**first[assumption.FIELD], "site_answer": "confirmed", "confirmed_later": {
        "confirmation_digest": row["confirmation_digest"], "confirmed_at": row["confirmed_at"],
        "issuer_channel": "website"}}
    # Nothing is bought again: the same half, contract digest and probe request, so the same screening half
    # (no re-plan), and the screening source still binds the job's context (2026-11-07: no
    # task_context_changed). What was scored stays labelled as scored on assumed thresholds.
    assert Path(assumed_path).read_bytes() == record_before
    assert read_team_scene_inputs(assumed_path)["task_success_contract"] == assumed
    profile = profile_for_capture(world["capture"])
    assert profile["team_scene_inputs_path"] == assumed_path
    assert current_half(profile, scene_revision_digest=revision) == assumed_half
    assert current_probe_state(world["capture"])["request"]["request_digest"] == probe_before
    assert world["context_path"].read_bytes() == context_before
    assert world["screening_source"]() == source_before
    # Later passes ask the website nothing more, move nothing and leave the status as it is.
    calls = webapp.calls
    for _ in range(2):
        later = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
        assert later["requeued"] == [] and later["website_owner_decisions"] == {}
    assert webapp.calls == calls and world["process"]()["status"] == "idle" and _status(world) == status


def test_an_assumed_half_compiled_again_for_another_reason_keeps_its_contract_and_the_recorded_confirmation(
        tmp_path, monkeypatch, clock):
    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    [first] = world["process"]()["results"]
    assumed = read_team_scene_inputs(first["team_scene_inputs_path"])["task_success_contract"]
    applied = confirm(world["job"])
    assert worker.requeue_transient_blocked(queue_root=queue, website_transport=FakeWebApp())["requeued"] == []
    assert _status(world)[assumption.FIELD]["confirmed_later"]["issuer_channel"] == "operator_door"
    # Something else sends the half to be compiled again (here: it is found unscorable once).
    monkeypatch.setattr(worker, "unscorable_half", lambda _path: "start_spawn_mismatch")
    shutil.copyfile(queue / "processed" / f"{key}.json", queue / "pending" / f"{key}.json")
    [second] = world["process"]()["results"]
    assert (second["status"], second["outcome"]) == ("completed", "compiled")
    assert second["team_scene_inputs_path"] != first["team_scene_inputs_path"]
    # The same assumed contract, digest for digest: the confirmation is recorded in the new half, not applied.
    assert read_team_scene_inputs(second["team_scene_inputs_path"])["task_success_contract"] == assumed
    assert "owner_success_confirmation" not in second
    recorded = _record(second["team_scene_inputs_path"])[assumption.FIELD]
    assert recorded["confirmed_later"]["confirmation_digest"] == applied["confirmation_digest"]
    assert second[assumption.FIELD] == recorded and recorded["site_answer"] == "confirmed"


def test_a_request_whose_intent_cannot_be_read_is_assumed_too_and_matches_no_confirmation(tmp_path, monkeypatch, clock):
    from blueprint_pipeline.task_evaluation_scene_intent_contracts import ROOT_ENV

    world = _world(tmp_path, monkeypatch)
    intents = str(world["job"].intents)
    monkeypatch.delenv(ROOT_ENV)
    [first] = world["process"]()["results"]
    # Nothing can be looked up under the intent, and nothing waits for it: assumed, with no binding named.
    assert (first["status"], first["outcome"], first["blockers"]) == ("completed", "compiled", [])
    assert first["owner_success_confirmation"] == {"status": "intent_unbound",
                                                   "reason": "scene_intake_root_not_configured"}
    assert first[assumption.FIELD]["status"] == assumption.ASSUMED and "binding_digest" not in first[assumption.FIELD]
    webapp = FakeWebApp()
    unbound = worker.requeue_transient_blocked(queue_root=world["queue"], website_transport=webapp)
    assert unbound["requeued"] == [] and webapp.calls == 0
    assert unbound["website_owner_decisions"][world["key"]]["status"] == "intent_unbound"
    # A confirmation made later cannot be matched to a half that named no binding: unknown stays unknown.
    monkeypatch.setenv(ROOT_ENV, intents)
    confirm(world["job"])
    result = worker.requeue_transient_blocked(queue_root=world["queue"], website_transport=webapp)
    assert result["requeued"] == [] and "confirmed_later" not in _status(world)[assumption.FIELD]


def test_this_is_wrong_is_kept_with_its_reason_and_changes_nothing_else(tmp_path, monkeypatch, clock):
    world = _world(tmp_path, monkeypatch)
    [first] = world["process"]()["results"]
    webapp = FakeWebApp()
    worker.requeue_transient_blocked(queue_root=world["queue"], website_transport=webapp)
    decision = webapp.decide("held", "too_strict")
    result = worker.requeue_transient_blocked(queue_root=world["queue"], website_transport=webapp)
    status = _status(world)
    assert result["requeued"] == [] and not (world["job"].directory / confirmation.DIRECTORY).exists()
    assert _core(status[website.STATUS_FIELD]) == {
        "status": "held", "reason": "too_strict", "proposal_digest": decision["proposal_digest"],
        "decision_digest": decision["decision_digest"]}
    # The half stands; no number changed and nothing was stopped or compiled again. Its marker now says the
    # site disputed the thresholds, which a site that has not answered never carries.
    assert status[assumption.FIELD] == {**first[assumption.FIELD], "site_answer": "this_is_wrong", "this_is_wrong": {
        "reason": "too_strict", "decision_digest": decision["decision_digest"],
        "proposal_digest": decision["proposal_digest"]}}
    assert {key: value for key, value in status.items()
            if key not in (website.STATUS_FIELD, assumption.FIELD)} == {
        key: value for key, value in first.items() if key != assumption.FIELD}
    assert profile_for_capture(world["capture"])["team_scene_inputs_path"] == first["team_scene_inputs_path"]
    assert world["process"]()["status"] == "idle"
    # If the site then changes its mind and confirms, that is recorded too; still nothing moves.
    webapp.decide("confirmed")
    assert worker.requeue_transient_blocked(queue_root=world["queue"], website_transport=webapp)["requeued"] == []
    changed = _status(world)[assumption.FIELD]
    assert changed["site_answer"] == "confirmed" and changed["confirmed_later"]["issuer_channel"] == "website"
    assert changed["this_is_wrong"]["reason"] == "too_strict"


def test_a_request_an_older_release_left_waiting_for_the_owner_is_compiled_by_the_new_release(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    # As the release before this change left it: refused for want of the owner's confirmation, waiting.
    (queue / "pending" / f"{key}.json").rename(queue / "processed" / f"{key}.json")
    waiting = {"schema_version": worker.STATUS_SCHEMA, "request_key": key, "status": "blocked",
               "blockers": [REFUSED], "release_commit": RELEASE_1}
    (queue / "status" / f"{key}.json").write_text(json.dumps(waiting))
    assert worker.final_block(waiting, release=RELEASE_1) is True
    assert worker.requeue_transient_blocked(queue_root=queue, website_transport=FakeWebApp())["requeued"] == []
    monkeypatch.setattr(worker, "current_release_commit", lambda: RELEASE_2)
    # Not final under the new release: it returns the refusal once, and compiles it without the owner.
    assert worker.final_block(waiting, release=RELEASE_2) is False
    requeued = worker.requeue_transient_blocked(queue_root=queue, website_transport=FakeWebApp())["requeued"]
    assert requeued == [{"request_key": key, "reason": "new_release", "release_commit": RELEASE_2,
                         "blocked_release_commit": RELEASE_1}]
    [status] = world["process"]()["results"]
    assert (status["status"], status["outcome"], status["blockers"]) == ("completed", "compiled", [])
    assert status[assumption.FIELD]["status"] == assumption.ASSUMED
    assert status[worker.RELEASE_COMMIT_FIELD] == RELEASE_2


def test_halves_and_statuses_stored_before_the_marker_existed_still_read(tmp_path, monkeypatch, clock):
    world = _world(tmp_path, monkeypatch)
    job, queue, key = world["job"], world["queue"], world["key"]
    # A half its site confirmed is, field for field, what every release before this one wrote.
    confirm(job)
    [status] = world["process"]()["results"]
    record = _record(status["team_scene_inputs_path"])
    assert assumption.FIELD not in record and assumption.FIELD not in status
    assert read_team_scene_inputs(status["team_scene_inputs_path"])["task_success_contract"]["provenance"][
        "author_source"] == "task_owner"
    assert assumption.half_assumption(status["team_scene_inputs_path"]) is None
    assert not assumption.status_assumed(status)
    assert assumption.site_confirmation(status["team_scene_inputs_path"], world["request"]) is None
    # The passes leave it alone: no question for the website, no requeue, and a redelivery reuses the half.
    webapp = FakeWebApp()
    result = worker.requeue_transient_blocked(queue_root=queue, website_transport=webapp)
    assert result["requeued"] == [] and result["website_owner_decisions"] == {} and webapp.calls == 0
    shutil.copyfile(queue / "processed" / f"{key}.json", queue / "pending" / f"{key}.json")
    [again] = world["process"]()["results"]
    assert (again["status"], again["outcome"]) == ("completed", "already_compiled")
    assert again["team_scene_inputs_path"] == status["team_scene_inputs_path"] and assumption.FIELD not in again
    # Statuses of other shapes an older release wrote are read as having assumed nothing.
    for old in ({}, {"status": "completed"}, {"status": "completed", "team_scene_inputs_path": "/gone"},
                {"status": "blocked", "blockers": ["team_scene_task_invalid"]}, {"status": "skipped"}):
        assert not assumption.status_assumed(old) and not assumption.awaiting_site(old)
        assert assumption.noted_status(old, world["request"], exchange=None) is None
    assert assumption.half_assumption(tmp_path / "missing.json") is None and assumption.half_assumption(None) is None


def test_an_assumed_half_goes_on_to_its_probe_and_is_offered_once_the_probe_passes(tmp_path, monkeypatch):
    from blueprint_pipeline import robot_free_probe_gate as gate
    from blueprint_pipeline.rigid_robot_free_probe_bundle import current_probe_state
    from tests.test_team_scene_packet_worker import _passed_rigid_probe

    world = _world(tmp_path, monkeypatch)
    [first] = world["process"]()["results"]
    # The compile queued the half's robot-free probe: the request it plans is bound to the assumed contract.
    state = current_probe_state(world["capture"])
    assert state["status"] == "probe_required"
    [entry] = (tmp_path / "rigid-robot-free-probes" / "pending").glob("*.json")
    assert _record(entry)["probe_request_digest"] == state["request"]["request_digest"]
    assert first["offer"]["team_candidate_offer_refused"] == "team_evaluation_robot_free_probe_receipt_missing"
    # The probe lane passes it; the redelivered request reuses the half, still marked as assumed, and the
    # job's offer now lists the team evaluation: every gate after the packet admits the assumed contract.
    profile = profile_for_capture(world["capture"])
    (world["job"].run / gate.RECEIPT_FILENAME).write_text(json.dumps(_passed_rigid_probe(profile)))
    shutil.copyfile(world["queue"] / "processed" / f"{world['key']}.json",
                    world["queue"] / "pending" / f"{world['key']}.json")
    [again] = world["process"]()["results"]
    assert (again["status"], again["outcome"]) == ("completed", "already_compiled")
    assert again["team_scene_inputs_path"] == first["team_scene_inputs_path"]
    assert again[assumption.FIELD] == first[assumption.FIELD]
    assert again["offer"]["status"] == "published" and again["offer"]["team_evaluation_listed"] is True


def test_the_remote_compile_carries_the_website_flag_and_returns_the_marker(tmp_path, monkeypatch):
    fs = tmp_path / "fs"
    base = fs / "var/lib/blueprint/test"
    arguments = {"revision": {"revision_digest": "sha256:" + "1" * 64}, "droid_profile_digest": "sha256:" + "2" * 64,
                 "documents": {"task": base / "documents" / "task.json"},
                 **{name: base / name for name in lane.PATH_ARGUMENTS}}
    plain = lane._worker_arguments(arguments, fs)
    flagged = lane._worker_arguments({**arguments, **worker.WEBSITE_COMPILE}, fs)
    assert plain is not None and "assume_unconfirmed_standard_thresholds" not in plain
    assert flagged == {**plain, "assume_unconfirmed_standard_thresholds": True}

    class Roots:
        filesystem_root = fs

        @staticmethod
        def local(path: str) -> Path:
            return fs / str(path).lstrip("/")

    assert "assume_unconfirmed_standard_thresholds" not in lane._local_arguments(plain, Roots)
    assert lane._local_arguments(flagged, Roots)["assume_unconfirmed_standard_thresholds"] is True
