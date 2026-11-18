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
    assert verdict == {"reopen": True, "reason": "held_gap_now_ships_room_package",
                       "stopped_on": ["website_task_profile_mechanism_mismatch"]}
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
    assert reopen.reopen_verdict(pipeline)["reason"] == "blocker_now_goes_on"


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
