"""A registration recapture hold is looked at again once per release, by itself (capture-39ecdc97, 2026-11-12).

The fixture keeps 39ecdc97's handoff shape on the live release: ``needs_input`` on
``website_registration_ambiguous`` with ``recapture_required`` (next actor: the site). That stop came from a
pose search that counted the scene turned over as a candidate; the next release decides it differently.

The second half runs the real listener, ledger, status reader and drain (``tests.test_website_held_handoff_replay``)
and checks that a job the site can no longer go on with is never run again: a re-filmed capture (the website's
active capture is another one), a run the owner closed, and a withdrawn capture.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_held_handoff_replay.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_recapture_required.py
import json

import pytest

from blueprint_pipeline import website_held_handoff_replay as held
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_recapture_required import FIELD, record

CAPTURE = "walkthrough-capture-reference"
SCENE = "site-capture-reference"


def _job(tmp_path, code="website_registration_ambiguous"):
    pipeline = tmp_path / "scenes" / SCENE / "captures" / CAPTURE / "pipeline"
    (pipeline / "website_scene_preparation").mkdir(parents=True)
    context = {"schema_version": "website_site_task_context.v1", "purpose": "scene_preparation",
               "request_id": "capture-reference", "scene_id": SCENE, "capture_id": CAPTURE,
               "description": "Pick up the wooden tray from the dining table and place it at the other end of the same table.",
               "task_family": "rigid_pick_and_place", "confirmed": True}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    (pipeline / "website_task_context.json").write_text(json.dumps(context))
    handoff = {"schema_version": "website_scene_handoff.v1", "scene_id": SCENE, "capture_id": CAPTURE,
               "status": "needs_input", "blockers": [code],
               FIELD: record(code, task_context_digest=context["context_digest"]),
               "simulator_ready": False, "visual_reconstruction_ready": True, "claim_ceiling": "development_only"}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    (pipeline / "website_scene_preparation/handoff.json").write_text(json.dumps(handoff))
    import time
    (pipeline / "website_scene_sponsorship.json").write_text(json.dumps({"expires_at_epoch": time.time() + 86_400}))
    return pipeline


@pytest.mark.parametrize("release", ["a" * 40, "b" * 40])
def test_a_registration_hold_is_reevaluated_once_per_release(tmp_path, monkeypatch, release):
    from blueprint_pipeline import team_scene_packet_worker
    pipeline = _job(tmp_path)
    monkeypatch.setattr(team_scene_packet_worker, "current_release_commit", lambda: release)
    assert held.held_kind(pipeline) == "registration_recapture"
    fact = held.arrived_fact(pipeline)
    assert fact["kind"] == "release" and fact["release_commit"] == release
    assert fact["recapture_code"] == "website_registration_ambiguous"
    # The fact is keyed by the release: the drain spends it once, and the next release brings a new one.
    assert fact["fact_digest"] == canonical_digest({key: value for key, value in fact.items() if key != "fact_digest"})


@pytest.mark.parametrize("code", ["website_front_normal_undetermined", "website_task_object_not_observed",
                                  "website_capture_frames_unusable"])
def test_a_stop_decided_by_what_the_video_shows_is_not_reevaluated(tmp_path, monkeypatch, code):
    from blueprint_pipeline import team_scene_packet_worker
    pipeline = _job(tmp_path, code)
    monkeypatch.setattr(team_scene_packet_worker, "current_release_commit", lambda: "c" * 40)
    assert held.held_kind(pipeline) is None and held.arrived_fact(pipeline) is None


def test_without_a_release_there_is_nothing_to_look_at_again(tmp_path, monkeypatch):
    from blueprint_pipeline import team_scene_packet_worker
    pipeline = _job(tmp_path)
    monkeypatch.setattr(team_scene_packet_worker, "current_release_commit", lambda: None)
    assert held.arrived_fact(pipeline) is None


def test_a_job_whose_grant_has_ended_is_left_as_it_is(tmp_path, monkeypatch):
    from blueprint_pipeline import team_scene_packet_worker
    pipeline = _job(tmp_path)
    (pipeline / "website_scene_sponsorship.json").write_text(json.dumps({"expires_at_epoch": 1.0}))
    monkeypatch.setattr(team_scene_packet_worker, "current_release_commit", lambda: "d" * 40)
    assert held.arrived_fact(pipeline) is None


# --- The real drain and status reader on a held registration job ---------------------------------------------

from tests.test_website_held_handoff_replay import _confirm, _handoff, _resume, site  # noqa: E402,F401
from tests.test_website_preparation_status import prepared_source  # noqa: E402,F401

HELD_CODE = "website_registration_ambiguous"


def _held_registration(site):  # noqa: F811
    """The 39ecdc97 shape on the real listener: confirmed brief, live grant, held on registration."""
    import time

    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline.common import write_json

    _confirm(site)
    site.state["registration"] = HELD_CODE
    site.deliver()
    pipeline = site.root / "pipeline"
    write_json(pipeline / "website_scene_sponsorship.json", {"expires_at_epoch": time.time() + 86_400})
    assert _handoff(site)["blockers"] == [HELD_CODE]
    assert held.held_kind(pipeline) == "registration_recapture"
    assert listener._read_job_ledger(site.root)[held.FIELD]["schema_version"] == held.SCHEMA
    site.state["release"] = "f" * 40
    return pipeline


def _website_refuses(monkeypatch, code):
    """The website's typed refusal of the task-context read, as ``website_webapp_request`` raises it."""
    from blueprint_pipeline import website_task_context as context_reader

    def refuse(**_):
        raise ValueError(f"website_control_task-context_http_409:{code}")
    monkeypatch.setattr(context_reader, "load_current_website_task_context", refuse)


def _withdraw(site):  # noqa: F811
    from blueprint_pipeline.common import write_json
    scene = site.root.parent.parent
    tombstone = {"schema_version": "website_capture_withdrawal_tombstone.v1", "scene_id": scene.name,
                 "request_id": scene.name[5:], "consent_revoked_at": "2026-11-12T09:00:00Z"}
    tombstone["digest"] = canonical_digest(tombstone, digest_field="digest")
    write_json(scene / "website_withdrawal/tombstone.json", tombstone)


def test_an_open_held_registration_job_runs_once_on_the_new_release_and_continues(site):  # noqa: F811
    pipeline = _held_registration(site)
    # This release decides the registration: the stage no longer stops on it.
    site.state["registration"] = None
    assert site.reconcile()["replayed"] == 1 and len(site.runs) == 2
    assert held.FIELD not in _handoff(site) and HELD_CODE not in _handoff(site)["blockers"]
    # Nothing paid ran again: the stage read its retained inputs under the same context.
    assert len(set(site.state["media"])) == 1
    assert held.held_kind(pipeline) != "registration_recapture"


@pytest.mark.parametrize("ending", ["refilmed", "closed", "withdrawn"])
def test_a_job_the_site_cannot_go_on_with_is_never_run_again(site, monkeypatch, ending):  # noqa: F811
    from blueprint_pipeline import pubsub_handoff_listener as listener

    pipeline = _held_registration(site)
    site.state["registration"] = None
    before = listener._read_job_ledger(site.root)
    if ending == "refilmed":
        # The site sent a new video: the website's active capture is the new one (assertWebsiteRunOpen).
        _website_refuses(monkeypatch, "website_run_capture_superseded")
    elif ending == "closed":
        _website_refuses(monkeypatch, "website_run_closed_by_owner")
    else:
        _withdraw(site)
    assert held.arrived_fact(pipeline) is not None  # the release fact alone would rerun it
    assert site.reconcile()["replayed"] == 0 and len(site.runs) == 1
    # Refused before anything ran: no replay recorded, ledger and hold unchanged.
    assert not (site.work / held.STATE_FILE).exists()
    assert listener._read_job_ledger(site.root) == before and _handoff(site)["blockers"] == [HELD_CODE]
    # The operator's resume refuses it too, before any run.
    with pytest.raises(ValueError):
        _resume(site)
    assert len(site.runs) == 1 and not (site.work / held.STATE_FILE).exists()

