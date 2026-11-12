"""A job no catalog robot fits holds before spend, reads as waiting, and reruns when the robots change.

The actual listener, lease, ledger, capture preparation stage, status reader and drain run.
``run_e2e`` stands in for the orchestrator around the website preparation stage, and the scene
handoff around ``prepare_rigid_native_inputs`` (where the precheck decides) is cut down to what it
records before registration. Synthetic rooms and catalogs only; no provider call.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_robot_fit_hold.py
#   src/blueprint_pipeline/website_robot_fit_precheck.py
#   src/blueprint_pipeline/website_task_context_contracts.py
#   src/blueprint_pipeline/website_capture_preparation.py
#   src/blueprint_pipeline/website_preparation_outcome.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
#   src/blueprint_pipeline/pubsub_handoff_listener.py
#   src/blueprint_pipeline/run_e2e.py

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_robot_fit_hold as hold
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline.common import PipelineError, StageError, write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_retirement_generations import capture_birth_source_projection
from blueprint_pipeline.website_preparation_status import (
    DELIVERY_FILE, STATUS_V5, STATUS_V6, read_preparation_status, retain_preparation_wakeup,
)
from blueprint_pipeline.website_robot_fit_precheck import HOLD_PREFIX
from tests.test_website_assessment_resume import WIRE_FIXTURE
from tests.test_website_preparation_status import prepared_source  # noqa: F401
from tests.test_website_robot_fit_precheck import fixed_arm_catalog, shelf_scene


def _delivery(root: Path, owner: dict[str, Any], monkeypatch) -> bytes:
    """The capture's own handoff message, as the drains replay it."""
    birth = capture_birth_source_projection(root)
    payload = json.loads(WIRE_FIXTURE.read_text().replace("walkthrough-req-1", owner["capture_id"]))
    payload["source_finalize"] = {"bucket": owner["bucket"], "object_name": owner["completion_marker"]["object_name"],
        "generation": owner["completion_marker"]["generation"], "event_id": "original-event",
        "event_source": "original-source"}
    payload["source_membership_selector"] = birth["source_membership_selector"]
    payload["pipeline_status_event"]["source_finalize"] = payload["source_finalize"]
    payload["pipeline_handoff_uri"] = (f"gs://{owner['bucket']}/"
        f"{birth['source_membership_selector']['object_name'].rsplit('/', 1)[0]}/pipeline_handoff.json")
    payload["pipeline_status_event"]["pipeline_handoff_uri"] = payload["pipeline_handoff_uri"]
    (root / listener.JOB_LEDGER_FILENAME).unlink()
    write_json(root / "raw/manifest.json", {"capture_source": "browser_self_capture"})
    monkeypatch.setattr(listener, "stage_handoff_capture", lambda *_, **__: root)
    monkeypatch.setattr(context_reader, "load_website_scene_sponsorship", lambda **_: {})
    return json.dumps(payload, indent=2).encode()


@pytest.fixture
def site(prepared_source, monkeypatch, tmp_path):  # noqa: F811 - imported pytest fixture
    """A website job for the shelf above any reach, its catalog the fixed arms only."""
    from blueprint_pipeline import (
        beta_policy_candidate_catalog,
        website_native_background,
        website_object_observations,
        website_scene_handoff,
    )
    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    from blueprint_pipeline.website_scene_handoff import prepare_rigid_native_inputs

    root, owner, context, _, _, _ = prepared_source
    # No operator team registry here: a registry another test left in the environment would be
    # read as one, and the hold's recheck inputs would name it.
    monkeypatch.delenv("BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY", raising=False)
    raw = _delivery(root, owner, monkeypatch)
    job, collision = shelf_scene(tmp_path)
    work = root / "pipeline" / "website_scene_preparation"
    job = {**job, "schema_version": "website_scene_preparation.v1",
           "binding": {"task_context_digest": context["context_digest"]}}
    job["digest"] = canonical_digest(job, digest_field="digest")
    write_json(work / "preparation.json", job)
    catalogs = {"current": fixed_arm_catalog()}
    monkeypatch.setattr(beta_policy_candidate_catalog, "load_default_catalog", lambda: catalogs["current"])
    for name in ("prepare_collision_stage", "prepare_appearance_stage"):
        monkeypatch.setattr(website_native_background, name, lambda path: {"stage": "stub"})
    monkeypatch.setattr(website_native_background, "prepare_construction_stages", lambda **_: {"references": []})
    monkeypatch.setattr(website_native_background, "construction_rights_admission", lambda **_: {"status": "stub"})
    monkeypatch.setattr(website_object_observations, "_record", lambda path: {"path": str(path)})
    verdict = {"schema_version": "website_rigid_task_capability_verdict.v1", "preparation_digest": job["digest"],
               "collision_sha256": "sha256:" + hashlib.sha256(collision.read_bytes()).hexdigest(),
               "status": "unassessed", "blockers": ["robot_rigid_pick_place_unreachable:approach"],
               "embodiments": [{"robot_id": "franka_panda", "status": "blocked", "grasp": "top_down",
                                "blockers": ["robot_rigid_pick_place_unreachable:approach"]}]}
    verdict["verdict_digest"] = canonical_digest(verdict, digest_field="verdict_digest")
    monkeypatch.setattr("blueprint_pipeline.website_rigid_feasibility.rigid_task_verdict", lambda **_: verdict)
    monkeypatch.setattr("blueprint_pipeline.website_robot_fit.robot_fit_table",
                        lambda **_: {"table_digest": "sha256:" + "5" * 64, "counts": {}})

    state = {"legacy": True}

    def handoff(**_):
        # The handoff's own record around prepare_rigid_native_inputs; past it, registration and the
        # intake outbox are what the real handoff then writes.
        result = {"schema_version": "website_scene_handoff.v1", "status": job["status"], "blockers": [],
                  "claim_ceiling": "development_only", "simulator_ready": False,
                  "capture_id": owner["capture_id"], "scene_id": owner["scene_id"],
                  "visual_reconstruction_ready": True, "preparation_path": str(work / "preparation.json"),
                  "preparation_digest": job["digest"]}
        ready = prepare_rigid_native_inputs(result=result, preparation=job, preparation_path=work / "preparation.json",
                native_root=work / "native", task_context=context, now=0.0,
                runtime_inputs={"status": "ready", "digest": "sha256:" + "3" * 64,
                                "appearance": {"status": "native_appearance_authored"},
                                "collision": {"path": str(collision)}})
        precheck = result.get("robot_fit_precheck") or {}
        if state["legacy"] and precheck.get("status") == "no_fit":
            # A release before 2026-11-12 held a no-fit job here; jobs it held are still on the host.
            result.update(status="needs_input", blockers=[precheck["hold_reason"]])
            ready = False
        if ready:
            result["source_registration"] = {"status": "registered"}
            result["website_intake_outbox"] = {"state": "forward_pending", "submission_id": owner["capture_id"]}
        result["digest"] = canonical_digest(result, digest_field="digest")
        write_json(work / "handoff.json", result)
        return result

    monkeypatch.setattr(website_scene_handoff, "prepare_website_scene_handoff", handoff)
    runs = []

    def run_e2e(**kwargs):
        runs.append(kwargs)
        gate = SimpleNamespace(passed=True)
        try:
            run_website_capture_preparation(
                descriptor=SimpleNamespace(metadata={"site_task_context": context}, requested_outputs=[],
                                           to_dict=lambda: {"capture_id": owner["capture_id"],
                                                            "scene_id": owner["scene_id"], "metadata": {}}),
                capture_root=root, pipeline_dir=root / "pipeline", source_video=None,
                derived_scene_generation_allowed=False, gates=[],
                prepare_media=lambda **_: {"status": "noop", "privacy_verified": True},
                check_privacy=lambda **_: gate, make_gate=lambda *_: gate)
        except StageError as exc:
            raise PipelineError(str(exc)) from exc
        return {"status": "completed"}

    def deliver():
        return listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai",
                                                run_e2e=run_e2e)

    return SimpleNamespace(root=root, raw=raw, runs=runs, deliver=deliver, catalogs=catalogs, run_e2e=run_e2e,
                           selectors=None, state=state)


def _held(site) -> dict[str, Any]:
    site.deliver()
    retain_preparation_wakeup(site.root)
    site.selectors = json.loads((site.root / DELIVERY_FILE).read_text())["selectors"]
    return listener._read_job_ledger(site.root)


def test_a_no_fit_job_is_built_and_tested_with_its_paper_verdict_as_a_flag(site):
    from blueprint_pipeline.atlas_scene_estimates import preparation_flags

    site.state["legacy"] = False
    ledger = _held(site)
    work = site.root / "pipeline/website_scene_preparation"
    handoff = json.loads((work / "handoff.json").read_text())
    # The precheck is recorded as before, and the job goes on to registration and the tests.
    assert handoff["robot_fit_precheck"]["status"] == "no_fit"
    assert handoff["robot_fit_precheck"]["hold_reason"] == HOLD_PREFIX + "reach"
    assert HOLD_PREFIX + "reach" not in handoff["blockers"] and handoff["website_intake_outbox"]
    assert ledger["status"] == "completed" and hold.FIELD not in ledger
    assert read_preparation_status(capture_root=site.root, selectors=site.selectors,
                                   schema_version=STATUS_V6)["state"] == "handed_off"
    # The cause travels with the job to its published result.
    assert "scene_estimate:robot_fit_none:reach" in preparation_flags(work / "preparation.json")


def test_a_job_held_under_the_old_rule_reads_as_waiting(site):
    ledger = _held(site)
    handoff = json.loads((site.root / "pipeline/website_scene_preparation/handoff.json").read_text())
    # Held, not failed: the stage completed and nothing was registered or enqueued.
    assert ledger["status"] == "completed" and ledger["attempt_count"] == 1
    assert handoff["blockers"] == [HOLD_PREFIX + "reach"] and "website_intake_outbox" not in handoff
    assert base64.b64decode(ledger[hold.FIELD]["payload_base64"], validate=True) == site.raw
    assert ledger[hold.FIELD]["recheck_inputs"] == {"catalog_digest": fixed_arm_catalog()["catalog_digest"],
                                                    "team_registry_digest": None}
    v5 = read_preparation_status(capture_root=site.root, selectors=site.selectors, schema_version=STATUS_V5)
    # A v5 reader sees a job waiting, with no outcome its closed vocabulary does not have.
    assert (v5["state"], v5["code"], v5["outcome"]) == ("awaiting_inputs", "preparation_inputs_pending", None)
    v6 = read_preparation_status(capture_root=site.root, selectors=site.selectors, schema_version=STATUS_V6)
    assert v6["state"] == "awaiting_inputs"
    assert v6["outcome"] == {"category": "robot_incompatibility", "reason": "robot_fit_none",
                             "basis": "conservative_pre_spend_fit_check", "robot_id": None,
                             "evidence_digest": handoff["robot_fit_precheck"]["digest"]}
    assert v6["robot_fit_precheck"]["hold_reason"] == HOLD_PREFIX + "reach"
    assert len(json.dumps(v6)) < 8192


def test_a_redelivery_never_reruns_the_hold(site):
    _held(site)
    site.deliver()
    assert len(site.runs) == 1 and listener._read_job_ledger(site.root)["attempt_count"] == 1


def _new_release(site, monkeypatch, *, grant_ends_in: float = 86_400, release: str = "f" * 40):
    """The new release on the host, and the job's retained grant ending ``grant_ends_in`` seconds from now."""
    from blueprint_pipeline import team_scene_packet_worker

    monkeypatch.setattr(team_scene_packet_worker, "current_release_commit", lambda: release)
    write_json(site.root / "pipeline/website_scene_sponsorship.json", {"expires_at_epoch": time.time() + grant_ends_in})
    site.state["legacy"] = False  # the new release: the precheck never holds


def _reconcile(site):
    return hold.reconcile_robot_fit_holds(listener, storage_root=site.root.parents[4],
                                          process_args={"provider": "openai", "run_e2e": site.run_e2e})


def test_a_job_held_under_the_old_rule_is_rechecked_once_per_release_and_goes_on(site, monkeypatch):
    _held(site)
    _new_release(site, monkeypatch)
    pending = hold.pending_recheck(site.root / "pipeline")
    assert pending["robot_fit_rule"] == hold.ROBOT_FIT_RULE  # differs from the hold's inputs, robots unchanged
    assert _reconcile(site)["rechecked"] == 1 and len(site.runs) == 2
    handoff = json.loads((site.root / "pipeline/website_scene_preparation/handoff.json").read_text())
    assert handoff["robot_fit_precheck"]["status"] == "no_fit" and handoff["website_intake_outbox"]
    assert _reconcile(site) == {"rechecked": 0, "held": 0, "waiting": 0}


def test_an_old_hold_that_held_again_waits_for_the_next_release(site, monkeypatch):
    _held(site)
    _new_release(site, monkeypatch)
    site.state["legacy"] = True  # this release still held it (say, a second hold): once per release only
    assert _reconcile(site)["rechecked"] == 1
    assert _reconcile(site)["rechecked"] == 0 and len(site.runs) == 2
    _new_release(site, monkeypatch, release="e" * 40)
    assert _reconcile(site)["rechecked"] == 1 and len(site.runs) == 3


@pytest.mark.parametrize("ending", ["expired", "no_release", "refilmed", "closed", "withdrawn"])
def test_an_old_hold_the_site_cannot_go_on_with_is_left_exactly_as_it_is(site, monkeypatch, ending):
    from tests.test_registration_recapture_reevaluation import _website_refuses, _withdraw

    _held(site)
    _new_release(site, monkeypatch, grant_ends_in=-1 if ending == "expired" else 86_400,
                 release=None if ending == "no_release" else "f" * 40)
    if ending == "refilmed":
        _website_refuses(monkeypatch, "website_run_capture_superseded")
    elif ending == "closed":
        _website_refuses(monkeypatch, "website_run_closed_by_owner")
    elif ending == "withdrawn":
        _withdraw(site)
    before = listener._read_job_ledger(site.root)
    handoff = (site.root / "pipeline/website_scene_preparation/handoff.json").read_text()
    assert _reconcile(site)["rechecked"] == 0 and len(site.runs) == 1
    assert listener._read_job_ledger(site.root) == before
    assert (site.root / "pipeline/website_scene_preparation/handoff.json").read_text() == handoff


def test_a_catalog_change_reruns_the_held_job_once_and_it_goes_on(site):
    _held(site)
    site.catalogs["current"] = _shipped_catalog()  # the G1 lane on again
    pending = hold.pending_recheck(site.root / "pipeline")
    site.state["legacy"] = False
    assert pending == {"catalog_digest": site.catalogs["current"]["catalog_digest"], "team_registry_digest": None,
                       "robot_fit_rule": hold.ROBOT_FIT_RULE}
    # The preparation stage's input binding now names the changed inputs, so run_e2e reruns it.
    assert hold.preparation_binding_digest(site.root / "pipeline", None) == canonical_digest(
        {"corrections_binding": None, "robot_fit_recheck_inputs": pending})
    counts = hold.reconcile_robot_fit_holds(listener, storage_root=site.root.parents[4],
                                            process_args={"provider": "openai", "run_e2e": site.run_e2e})
    assert counts["rechecked"] == 1 and len(site.runs) == 2
    ledger = listener._read_job_ledger(site.root)
    handoff = json.loads((site.root / "pipeline/website_scene_preparation/handoff.json").read_text())
    # With the G1 lane on, the shelf is unknown: the job goes on to registration.
    assert ledger["status"] == "completed" and ledger["attempt_count"] == 2
    assert ledger["attempt_history"][-1][hold.RECHECK_MARK] == pending
    assert handoff["robot_fit_precheck"]["status"] == "unknown" and handoff["website_intake_outbox"]
    assert read_preparation_status(capture_root=site.root, selectors=site.selectors,
                                   schema_version=STATUS_V6)["state"] == "handed_off"
    assert hold.reconcile_robot_fit_holds(listener, storage_root=site.root.parents[4],
                                          process_args={"provider": "openai", "run_e2e": site.run_e2e}) == {
        "rechecked": 0, "held": 0, "waiting": 0}
    assert len(site.runs) == 2


def test_a_registry_change_is_a_recheck_input(monkeypatch):
    from blueprint_pipeline import team_candidate_quick10_registry as registry

    monkeypatch.setattr(registry, "load_registry", lambda: {"registry_digest": "sha256:" + "a" * 64})
    assert hold.recheck_inputs()["team_registry_digest"] == "sha256:" + "a" * 64

    def unreadable():
        raise ValueError("team_candidate_registry_not_operator_owned")
    monkeypatch.setattr(registry, "load_registry", unreadable)
    assert hold.team_registry_state() == (None, "unreadable")


def test_the_retain_check_accepts_only_the_hold_it_names():
    from blueprint_pipeline.website_task_context_contracts import retains_robot_fit_hold

    context = {"scene_id": "site-req1", "capture_id": "walkthrough-req1"}
    handoff = {"schema_version": "website_scene_handoff.v1", "status": "needs_input",
               "blockers": [HOLD_PREFIX + "reach"], "simulator_ready": False, "visual_reconstruction_ready": True,
               "runtime_inputs": {"status": "ready"}, **context,
               "robot_fit_precheck": {"status": "no_fit", "hold_reason": HOLD_PREFIX + "reach",
                                      "path": "/x/robot_fit_precheck.json", "digest": "sha256:" + "1" * 64}}

    def sealed(value):
        return {**value, "digest": canonical_digest(value, digest_field="digest")}

    assert retains_robot_fit_hold(sealed(handoff), context)
    for change in ({"blockers": [HOLD_PREFIX + "reach", "website_native_intake_pending"]},
                   {"website_intake_outbox": {"state": "forward_pending"}}, {"source_registration": {}},
                   {"robot_fit_precheck": {**handoff["robot_fit_precheck"], "status": "unknown"}},
                   {"robot_fit_precheck": {**handoff["robot_fit_precheck"], "hold_reason": "reach"}},
                   {"status": "intake_ready"}, {"capture_id": "walkthrough-other"}):
        assert not retains_robot_fit_hold(sealed({**handoff, **change}), context), change
    assert not retains_robot_fit_hold({**sealed(handoff), "status": "needs_input", "blockers": []}, context)
    # An abstained verdict beside it keeps its own blockers first.
    abstained = {**handoff, "capability_verdict": {"status": "abstained", "blockers": ["robot_x"]},
                 "blockers": ["robot_x", HOLD_PREFIX + "reach"]}
    assert retains_robot_fit_hold(sealed(abstained), context)


def _shipped_catalog() -> dict[str, Any]:
    from blueprint_pipeline.beta_policy_candidate_catalog import load_catalog

    return load_catalog(Path(__file__).resolve().parents[1] / "config" / "beta_policy_candidates.v1.json")


def test_a_recheck_that_stops_short_stays_owed_and_bounded(site):
    _held(site)
    site.catalogs["current"] = _shipped_catalog()
    failures = []

    def transient(**_):
        failures.append(1)
        raise PipelineError(str(StageError("website_scene_preparation", "synthetic_transient_refusal")))

    args = {"provider": "openai", "run_e2e": transient}
    now = 1_000_000.0
    counts = hold.reconcile_robot_fit_holds(listener, storage_root=site.root.parents[4], process_args=args, now=now)
    ledger = listener._read_job_ledger(site.root)
    assert counts["rechecked"] == 1 and ledger["status"] == "failed_retryable"
    assert ledger["attempt_history"][-1][hold.RECHECK_MARK]["catalog_digest"] == site.catalogs["current"]["catalog_digest"]
    # Backed off, then owed again until the bound; never a loop.
    assert hold.reconcile_robot_fit_holds(listener, storage_root=site.root.parents[4], process_args=args,
                                          now=now + 1)["waiting"] == 1
    for step in range(1, 6):
        hold.reconcile_robot_fit_holds(listener, storage_root=site.root.parents[4], process_args=args,
                                       now=now + 4000 * step)
    assert len(failures) == hold.MAX_RECHECKS


def test_a_recheck_whose_ledger_moved_before_its_claim_is_a_retryable_result(site, monkeypatch):
    """The ledger changes between the reconciler's read and its claim: retryable, no exception, no attempt."""
    _held(site)
    site.catalogs["current"] = _shipped_catalog()
    real = listener.process_handoff_payload
    results = []

    def moved_then_processed(payload, **kwargs):
        with listener._locked_job_ledger(site.root, create=False) as current:
            listener._commit_job_ledger(site.root, {**current, "updated_at": "2026-11-11T00:00:00Z"},
                                        previous_revision=int(current["revision"]))
        results.append(real(payload, **kwargs))
        return results[-1]

    monkeypatch.setattr(listener, "process_handoff_payload", moved_then_processed)
    before = listener._read_job_ledger(site.root)
    counts = hold.reconcile_robot_fit_holds(listener, storage_root=site.root.parents[4],
                                            process_args={"provider": "openai", "run_e2e": site.run_e2e})
    assert counts["rechecked"] == 1 and len(site.runs) == 1
    [result] = results
    assert result["status"] == "robot_fit_recheck_changed" and result["queue_disposition"] == "retryable"
    assert result["blockers"] == ["website_robot_fit_recheck_changed"]
    ledger = listener._read_job_ledger(site.root)
    # No attempt recorded and no lease taken: only the change made under the test.
    assert ledger["attempt_count"] == before["attempt_count"] and ledger["status"] == "completed"
    assert ledger["attempt_history"] == before["attempt_history"] and ledger["revision"] == before["revision"] + 1
    assert not ledger.get("lease_token")
