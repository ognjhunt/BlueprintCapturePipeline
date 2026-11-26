"""Reopening a job that ended on a hold the current rules no longer stop on: once, through the resume path."""
from __future__ import annotations

import time

import pytest

from blueprint_pipeline import website_held_handoff_replay as replay
from blueprint_pipeline import website_rules_reopen as reopen
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_task_preparation import native_task_preparation_gap
from tests.test_website_scene_handoff import retained_hold


def _held(tmp_path, *, gap=True, blockers=None, status="needs_input", ledger="completed", grant_in=86400.0,
          family="rigid_pick_and_place"):
    """A capture whose completed preparation retained a hold, as an earlier release wrote it."""
    capture = tmp_path / "bucket/scenes/site-x/captures/walkthrough-x"
    pipeline = capture / "pipeline"
    context = {"schema_version": "website_site_task_context.v1", "request_id": "req", "scene_id": "site-x",
               "capture_id": "walkthrough-x", "confirmed": True, "task_family": family,
               "description": "Move the box onto the shelf"}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    write_json(pipeline / "website_task_context.json", context)
    if gap:
        handoff = retained_hold(context, native_task_preparation_gap(context,
                                                                     inferred_profile_id="articulated_open_close"))
    else:
        handoff = {"schema_version": "website_scene_handoff.v1", "status": status, "blockers": blockers,
                   "scene_id": "site-x", "capture_id": "walkthrough-x", "simulator_ready": False}
        handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    write_json(pipeline / "website_scene_preparation/handoff.json", handoff)
    if grant_in is not None:
        write_json(pipeline / "website_scene_sponsorship.json", {"expires_at_epoch": time.time() + grant_in})
    write_json(capture / reopen.LEDGER, {"status": ledger, "revision": 3, "scene_id": "site-x",
                                         "capture_id": "walkthrough-x", "producer_delivery_key": "k"})
    return capture, pipeline


def test_a_held_mismatch_reopens_once_through_the_resume_path(tmp_path, monkeypatch):
    capture, pipeline = _held(tmp_path)
    verdict = reopen.reopen_verdict(pipeline)
    assert verdict == {"reopen": True, "reason": "held_gap_reopens_may_build_object_and_buy_authoring",
                       "stopped_on": ["website_task_profile_mechanism_mismatch"], "what_happens": reopen.WHAT_HAPPENS}
    assert "buys the object's authoring" in reopen.WHAT_HAPPENS
    fact = replay.arrived_fact(pipeline)
    assert fact["kind"] == reopen.KIND and fact["rules_revision"] == reopen.REVISION
    assert fact["fact_digest"] == canonical_digest({key: value for key, value in fact.items() if key != "fact_digest"})
    runs = []

    class Listener:
        def process_handoff_payload(self, payload, **kw):
            runs.append(kw)
            return {"status": "processed", "queue_disposition": "terminal_success"}
    monkeypatch.setattr(replay, "require_still_open", lambda *_, **__: None)
    replay.resume_held_preparation(Listener(), capture_root=capture, storage_root=tmp_path, handoff_payload=b"p",
                                   ledger_digest="sha256:" + "1" * 64, ledger_revision=3, provider="local")
    assert runs[0]["expected_held_replay"]["fact"] == fact
    # The same fact again after the run claimed the job: never a second reopen.
    with pytest.raises(ValueError, match="already_replayed"):
        replay.resume_held_preparation(Listener(), capture_root=capture, storage_root=tmp_path, handoff_payload=b"p",
                                       ledger_digest="sha256:" + "1" * 64, ledger_revision=4, provider="local")
    assert len(runs) == 1
    # The run claimed the job (its ledger moved on) and held again on another handoff: still never reopened.
    write_json(capture / reopen.LEDGER, {"status": "completed", "revision": 4, "scene_id": "site-x",
                                         "capture_id": "walkthrough-x", "producer_delivery_key": "k"})
    other = {"schema_version": "website_scene_handoff.v1", "status": "awaiting_inputs", "scene_id": "site-x",
             "capture_id": "walkthrough-x", "blockers": ["support_surface_not_found_under_subject"]}
    write_json(pipeline / "website_scene_preparation/handoff.json",
               {**other, "digest": canonical_digest(other, digest_field="digest")})
    assert reopen.reopen_verdict(pipeline)["reason"] == "already_reopened_under_these_rules"
    assert replay.arrived_fact(pipeline) is None
    with pytest.raises(ValueError, match="not_eligible"):
        replay.resume_held_preparation(Listener(), capture_root=capture, storage_root=tmp_path, handoff_payload=b"p",
                                       ledger_digest="sha256:" + "1" * 64, ledger_revision=4, provider="local")


def test_a_reopen_whose_run_never_claimed_the_job_stays_eligible(tmp_path, monkeypatch):
    capture, pipeline = _held(tmp_path)
    fact = replay.arrived_fact(pipeline)
    reopen.mark_reopened(pipeline, fact=fact, ledger_revision=3)  # marked, then the run died before its claim
    assert reopen.reopen_verdict(pipeline)["reopen"] is True


def test_a_reopen_refuses_a_site_that_closed_withdrew_or_refilmed(tmp_path, monkeypatch):
    capture, _ = _held(tmp_path)

    def closed(*_, **__):
        raise ValueError("website_run_closed_by_owner")
    monkeypatch.setattr(replay, "require_still_open", closed)
    with pytest.raises(ValueError, match="website_held_handoff_resume_not_eligible"):
        replay.resume_held_preparation(object(), capture_root=capture, storage_root=tmp_path, handoff_payload=b"p",
                                       ledger_digest="sha256:" + "1" * 64, ledger_revision=3, provider="local")


def test_a_blocker_the_current_rules_ship_at_once_reopens(tmp_path):
    _, pipeline = _held(tmp_path, gap=False, status="awaiting_inputs",
                        blockers=["support_surface_not_found_under_subject"])
    assert reopen.reopen_verdict(pipeline)["reason"] == "blocker_reopens_may_build_object_and_buy_authoring"


@pytest.mark.parametrize("change, reason", [
    (dict(grant_in=-60.0), "grant_expired_or_missing"),
    (dict(grant_in=None), "grant_expired_or_missing"),
    (dict(ledger="terminal_authority_ended"), "ledger_terminal_authority_ended"),
    (dict(ledger="failed_retryable"), "ledger_failed_retryable"),
    (dict(gap=False, status="awaiting_inputs", blockers=["website_gemini_provider_credit_depleted"]), "wall:money"),
    (dict(gap=False, status="awaiting_inputs",
          blockers=["website_control_preparation-spend_http_409:consent_revoked"]), "wall:privacy_rights"),
    (dict(gap=False, status="needs_input", blockers=["website_task_context_not_confirmed"]), "consent_hold"),
    (dict(gap=False, status="awaiting_inputs", blockers=["website_subject_origin_conflict"]),
     "current_rules_still_wait"),
    (dict(gap=False, status="room_package_written", blockers=[]), "not_stopped"),
])
def test_never_reopened(tmp_path, change, reason):
    _, pipeline = _held(tmp_path, **change)
    verdict = reopen.reopen_verdict(pipeline)
    assert (verdict["reopen"], verdict["reason"]) == (False, reason)
    assert reopen.rules_reopen(pipeline) is None


def test_a_withdrawn_capture_or_one_whose_prepared_scene_may_have_gone_is_never_reopened(tmp_path, monkeypatch):
    capture, pipeline = _held(tmp_path)
    write_json(pipeline / "website_scene_preparation/prepared_scene_outbox.json", {"state": "sending"})
    assert reopen.reopen_verdict(pipeline)["reason"] == "intake_may_have_been_sent"
    (pipeline / "website_scene_preparation/prepared_scene_outbox.json").unlink()
    monkeypatch.setattr("blueprint_pipeline.website_atlas_preview_store.capture_withdrawn", lambda _root: True)
    assert reopen.reopen_verdict(pipeline)["reason"] == "withdrawn"


def test_the_survey_lists_each_capture_and_why(tmp_path):
    _held(tmp_path)
    [row] = reopen.survey(tmp_path)
    assert (row["scene_id"], row["capture_id"], row["reopen"]) == ("site-x", "walkthrough-x", True)


# --- a room package shipped only for its destination (2026-11-25) -------------------------------------------------

#: Job move-the-lamp-2's room package record as the host wrote it on 24 Nov (``room_package_inputs.v1.json``): one
#: piece not prepared, the destination, with no cause (the search named none before #3735).
LAMP_ROOM_RECORD = {
    "schema_version": "blueprint.room_package_inputs.v1", "task_kind": "not_defined_by_blueprint", "task_profile": "none",
    "success_rule_defined": False, "success_rule_assumed": False,
    "job_objects": {"count": 1, "in_package": "taken_out_of_room_scan"},
    "flags": ["task_destination_surface_contact_required", "scene_completion_not_admitted"],
    "not_prepared": [{"piece": "destination", "code": "task_destination_surface_contact_required"}],
    "robot_free_probe": "not_queued", "robot_included": False, "simulator_run": False,
    "plan_digest": "sha256:" + "5" * 64}


def _shipped(tmp_path, *, record=None, grant_in=86400.0, **handoff_extra):
    """A completed job whose handoff shipped its room package, sealed as ``website_scene_handoff`` seals it."""
    capture, pipeline = _held(tmp_path, gap=False, status="room_package_written", blockers=[], grant_in=grant_in)
    record = dict(LAMP_ROOM_RECORD if record is None else record)
    record["task_context_digest"] = replay._retained_context(pipeline)["context_digest"]
    record["inputs_digest"] = canonical_digest(record, digest_field="inputs_digest")
    path = pipeline / "website_scene_preparation/room_package/half/room_package_inputs.v1.json"
    write_json(path, record)
    handoff = {"schema_version": "website_scene_handoff.v1", "status": "room_package_written", "blockers": [],
               "scene_id": "site-x", "capture_id": "walkthrough-x", "simulator_ready": False,
               "room_package": {"path": str(path), "inputs_digest": record["inputs_digest"], "task_profile": "none",
                                "success_rule_defined": False}, **handoff_extra}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    write_json(pipeline / "website_scene_preparation/handoff.json", handoff)
    return capture, pipeline, path


class _Listener:
    def __init__(self):
        self.runs = []

    def process_handoff_payload(self, payload, **kw):
        self.runs.append(kw)
        return {"status": "processed", "queue_disposition": "terminal_success"}


def _resume(listener, capture, tmp_path, revision):
    return replay.resume_held_preparation(listener, capture_root=capture, storage_root=tmp_path, handoff_payload=b"p",
                                          ledger_digest="sha256:" + "1" * 64, ledger_revision=revision,
                                          provider="local")


def test_lamp_2_s_room_package_reopens_once_through_the_operator_s_resume(tmp_path, monkeypatch):
    """Its room package shipped only for the destination; this release places the object where only the wall stood
    in its box. The operator's ``--resume-preparation`` runs it once, under this revision; never twice."""
    capture, pipeline, _ = _shipped(tmp_path)
    assert reopen.REVISION != reopen.PREVIOUS_REVISION
    verdict = reopen.reopen_verdict(pipeline)
    assert verdict == {"reopen": True, "reason": "room_package_reopens_may_build_object_and_buy_authoring",
                       "stopped_on": ["task_destination_surface_contact_required"], "what_happens": reopen.WHAT_HAPPENS}
    fact = replay.arrived_fact(pipeline)
    assert fact["kind"] == reopen.KIND and fact["rules_revision"] == reopen.REVISION
    listener = _Listener()
    monkeypatch.setattr(replay, "require_still_open", lambda *_, **__: None)
    _resume(listener, capture, tmp_path, 3)
    assert len(listener.runs) == 1 and listener.runs[0]["expected_held_replay"]["fact"] == fact
    # The capture stage's binding now names the fact, so only the website preparation runs again.
    assert replay.preparation_binding(pipeline, "sha256:" + "2" * 64) != "sha256:" + "2" * 64
    # A second resume under the same revision, after the run claimed the job: refused, nothing runs.
    with pytest.raises(ValueError, match="already_replayed"):
        _resume(listener, capture, tmp_path, 4)
    # The run shipped the room package once more (a cause it could not get past) and its ledger moved on: never
    # reopened again.
    _shipped(tmp_path)
    write_json(capture / reopen.LEDGER, {"status": "completed", "revision": 4, "scene_id": "site-x",
                                         "capture_id": "walkthrough-x", "producer_delivery_key": "k"})
    assert reopen.reopen_verdict(pipeline)["reason"] == "already_reopened_under_these_rules"
    with pytest.raises(ValueError, match="not_eligible"):
        _resume(listener, capture, tmp_path, 4)
    assert len(listener.runs) == 1


def test_a_room_package_whose_recorded_cause_this_release_fixes_reopens(tmp_path):
    record = {**LAMP_ROOM_RECORD, "not_prepared": [{"piece": "destination", "code": "task_destination_surface_contact_required",
                                                     "cause": "other_geometry_in_its_box_wherever_it_would_rest"}]}
    assert reopen.reopen_verdict(_shipped(tmp_path, record=record)[1])["reopen"] is True


@pytest.mark.parametrize("record, reason", [
    # No surface under the place, or none large enough: nothing in this release gets past those.
    ({**LAMP_ROOM_RECORD, "not_prepared": [{"piece": "destination", "code": "task_destination_surface_contact_required",
                                            "cause": "no_surface_under_its_centre"}]},
     "room_package_cause_not_fixed_in_this_release"),
    ({**LAMP_ROOM_RECORD, "not_prepared": [{"piece": "destination", "code": "task_destination_surface_contact_required",
                                            "cause": "no_spot_with_the_disc_and_half_of_it_on_the_surface"}]},
     "room_package_cause_not_fixed_in_this_release"),
    # Another cause, or the destination and another piece.
    ({**LAMP_ROOM_RECORD, "flags": ["support_surface_not_found_under_subject"],
      "not_prepared": [{"piece": "object", "code": "support_surface_not_found_under_subject"}]},
     "room_package_cause_not_fixed_in_this_release"),
    ({**LAMP_ROOM_RECORD, "not_prepared": [{"piece": "destination", "code": "task_destination_surface_contact_required"},
                                           {"piece": "object", "code": "website_object_spec_dimension_conflict"}]},
     "room_package_cause_not_fixed_in_this_release"),
    # A room package no record says anything of (the navigation and unknown-kind packages): as before.
    ({**LAMP_ROOM_RECORD, "flags": [], "not_prepared": []}, "room_package_cause_not_fixed_in_this_release"),
    # Money: the job's spending cap, alone or beside the destination; a website refusal that is money.
    ({**LAMP_ROOM_RECORD, "flags": ["spending_cap_reached"],
      "not_prepared": [{"piece": "object", "code": "spending_cap_reached"}]}, "wall:money"),
    ({**LAMP_ROOM_RECORD, "flags": [*LAMP_ROOM_RECORD["flags"], "spending_cap_reached"]}, "wall:money"),
    ({**LAMP_ROOM_RECORD, "flags": [*LAMP_ROOM_RECORD["flags"], "website_scene_preparation_budget_exhausted"]},
     "wall:money"),
    # Privacy or rights.
    ({**LAMP_ROOM_RECORD, "flags": [*LAMP_ROOM_RECORD["flags"],
                                    "website_control_preparation-spend_http_409:consent_revoked"]}, "wall:privacy_rights"),
])
def test_a_room_package_shipped_for_anything_else_is_never_reopened(tmp_path, record, reason):
    _, pipeline, _ = _shipped(tmp_path, record=record)
    verdict = reopen.reopen_verdict(pipeline)
    assert (verdict["reopen"], verdict["reason"]) == (False, reason)
    assert replay.arrived_fact(pipeline) is None


def test_a_room_package_whose_intake_may_have_gone_or_whose_grant_ended_is_never_reopened(tmp_path, monkeypatch):
    capture, pipeline, _ = _shipped(tmp_path, website_intake_outbox={"id": "o", "request_digest": "sha256:" + "3" * 64})
    assert reopen.reopen_verdict(pipeline)["reason"] == "intake_may_have_been_sent"
    _shipped(tmp_path)
    for state in ("sending", "sent"):
        write_json(pipeline / "website_scene_preparation/prepared_scene_outbox.json", {"state": state})
        assert reopen.reopen_verdict(pipeline)["reason"] == "intake_may_have_been_sent"
    (pipeline / "website_scene_preparation/prepared_scene_outbox.json").unlink()
    assert reopen.reopen_verdict(pipeline)["reopen"] is True
    _shipped(tmp_path, grant_in=-60.0)
    assert reopen.reopen_verdict(pipeline)["reason"] == "grant_expired_or_missing"
    with pytest.raises(ValueError, match="not_eligible"):
        _resume(_Listener(), capture, tmp_path, 3)
    _shipped(tmp_path)
    monkeypatch.setattr("blueprint_pipeline.website_atlas_preview_store.capture_withdrawn", lambda _root: True)
    assert reopen.reopen_verdict(pipeline)["reason"] == "withdrawn"


def test_a_room_package_record_that_no_longer_seals_or_is_not_the_one_named_is_never_reopened(tmp_path):
    _, pipeline, path = _shipped(tmp_path)
    tampered = {**LAMP_ROOM_RECORD, "not_prepared": [{"piece": "destination", "code": "task_destination_surface_contact_required"}],
                "flags": []}
    write_json(path, tampered)  # its digest no longer seals it
    assert reopen.reopen_verdict(pipeline)["reason"] == "not_stopped"
    _shipped(tmp_path)
    record = {**replay.json.loads(path.read_text()), "robot_free_probe": "queued"}
    record["inputs_digest"] = canonical_digest(record, digest_field="inputs_digest")
    write_json(path, record)  # seals itself, but is not the record the handoff names
    assert reopen.reopen_verdict(pipeline)["reason"] == "room_package_record_changed"


def test_a_hold_the_previous_revision_reopened_is_not_reopened_again(tmp_path):
    """The 2026-11-25 revision adds the room package only: a hold the 2026-11-17 revision reopened, still held after
    its run claimed the job, stays as it is."""
    _, pipeline = _held(tmp_path)
    write_json(pipeline / "website_scene_preparation" / reopen.REOPENED_FILE,
               {"schema_version": "website_rules_reopen.v1", "rules_revision": reopen.PREVIOUS_REVISION,
                "ledger_revision": 2})
    assert reopen.reopen_verdict(pipeline)["reason"] == "already_reopened_under_these_rules"
    # A room package that revision could not reopen is this one's.
    _, shipped, _ = _shipped(tmp_path)
    assert reopen.reopen_verdict(shipped)["reopen"] is True
