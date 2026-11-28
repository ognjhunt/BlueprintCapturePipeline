"""A website job's simulation is built only once an approved company asks for it (owner decision, 2026-11-28).

Until a request exists the job's preparation keeps what is free and shares the job (the normalized video, the task
context and evidence, the grant), ends waiting, and buys nothing; the request starts today's path through the
held-handoff replay. The orchestrator, listener, ledger, status reader, drains, retention plan, company clip and
signed route run for real; the website, the providers and the scene handoff are fakes. Synthetic records only; no
provider call, nothing paid.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_simulation_request.py
#   src/blueprint_pipeline/site_package_orchestrator.py
#   src/blueprint_pipeline/website_held_handoff_replay.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_atlas_preview_http.py
#   src/blueprint_pipeline/website_scene_workspace_retention.py
#   src/blueprint_pipeline/pubsub_handoff_scene_operations.py
#   src/blueprint_pipeline/internal_test_job_door.py
#   deploy/operator-door/operator_door/requests.py
#   scripts/operator_door.py
from __future__ import annotations

import json
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import Request

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import site_package_orchestrator as orchestrator
from blueprint_pipeline import website_held_handoff_replay as replay
from blueprint_pipeline import website_simulation_request as simulation
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline.common import PipelineError, write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_preparation_outcome import NEW_REASONS_ENV
from blueprint_pipeline.website_preparation_status import (
    DELIVERY_FILE, STATUS_V1, STATUS_V6, STATUS_V9, held_preparation, read_preparation_status,
    retain_preparation_wakeup,
)
from tests.test_qualification_coverage_edges import _descriptor
from tests.test_website_preparation_status import prepared_source  # noqa: F401
from tests.test_website_robot_fit_hold import _delivery

REQUEST_ID = "req1"
SCENE, CAPTURE = f"site-{REQUEST_ID}", f"walkthrough-{REQUEST_ID}"
DIGEST_A, DIGEST_B = "sha256:" + "a" * 64, "sha256:" + "b" * 64


def _seal(value: dict) -> dict:
    value = {key: item for key, item in value.items() if key != "context_digest"}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


def _body(request_id: str = REQUEST_ID, digest: str = DIGEST_A, **change) -> dict:
    return {"request_id": request_id, "scene_id": f"site-{request_id}", "request_digest": digest,
            "source": "company", "requested_at_epoch": 1_800_000_000, **change}


def _forbidden(name: str):
    def call(*_args, **_kwargs):
        pytest.fail(f"{name} ran for a job that waits for its simulation to be asked for")
    return call


# ------------------------------------------------------------------ the orchestrator's switch


@pytest.fixture
def job(tmp_path, monkeypatch):
    """A website upload as the orchestrator meets it: the hand-off kept its task context and its ledger."""
    from blueprint_pipeline import website_source_video

    storage = tmp_path / "storage"
    gcs = storage / "source-bucket"
    descriptor = _descriptor(
        scene_id=SCENE, capture_id=CAPTURE, capture_source="unknown", capture_modality="video_only",
        requested_outputs=[], raw_prefix_uri=f"gs://source-bucket/scenes/{SCENE}/captures/{CAPTURE}/raw",
        frames_index_uri=f"gs://source-bucket/scenes/{SCENE}/captures/{CAPTURE}/raw/frames.json",
        raw_video_uri=f"gs://source-bucket/scenes/{SCENE}/captures/{CAPTURE}/raw/job.mov",
        metadata={"capture_entry_source": "browser_self_capture"})
    capture = gcs / "scenes" / SCENE / "captures" / CAPTURE
    (capture / "raw").mkdir(parents=True)
    (capture / "raw/job.mov").write_bytes(b"retained-video-fixture")
    (capture / "capture_descriptor.json").write_text(json.dumps(descriptor.to_dict()))
    context = _seal({"schema_version": "website_site_task_context.v1", "request_id": REQUEST_ID,
                     "scene_id": SCENE, "capture_id": CAPTURE, "purpose": "scene_preparation", "confirmed": False,
                     "confirmed_at": None, "description": "Move the box to the shelf",
                     "task_family": "general_task", "capture_rights": {"derived_scene_generation_allowed": True},
                     simulation.CONTEXT_FIELD: True})
    write_json(capture / "pipeline/website_task_context.json", context)
    write_json(capture / "pipeline_job_ledger.json", {"producer_delivery_key": DIGEST_B, "status": "processing"})
    calls: list[str] = []
    website = {"context": context}

    def task_context(**_):
        calls.append("task_context")
        return deepcopy(website["context"])
    monkeypatch.setattr(orchestrator, "load_current_website_task_context", task_context)
    monkeypatch.setattr(orchestrator, "load_website_scene_sponsorship",
                        lambda **_: {"schema_version": "website_scene_sponsorship.v1", "purpose": "scene_preparation"})
    # The only other website reads on this path: the no-grant ending's question (offline here: no ending).
    monkeypatch.setattr(context_reader, "website_webapp_request", lambda **_: (_ for _ in ()).throw(ValueError("offline")))
    real_normalize = website_source_video.preparation_source_video

    def normalize(**kwargs):
        calls.append("normalize")
        return real_normalize(**kwargs)
    monkeypatch.setattr(website_source_video, "preparation_source_video", normalize)
    return SimpleNamespace(storage=storage, gcs=gcs, capture=capture, context=context, website=website, calls=calls,
                           uri=f"gs://source-bucket/scenes/{SCENE}/captures/{CAPTURE}/capture_descriptor.json",
                           config=SimpleNamespace(gcs_root=gcs))


def _forbid_every_paid_or_atlas_call(monkeypatch):
    from blueprint_pipeline import (
        paid_resource_admission, website_atlas_preview, website_scene_handoff, website_task_context,
    )
    monkeypatch.setattr(orchestrator, "run_clean_plate_stage", _forbidden("the clean plate (Gemini, SAM)"))
    monkeypatch.setattr(website_atlas_preview, "enqueue_preview", _forbidden("the Atlas job"))
    monkeypatch.setattr(website_scene_handoff, "prepare_website_scene_handoff", _forbidden("the scene handoff"))
    for name in ("reserve_website_preparation_spend", "reserve_website_sam_spend", "load_atlas_configuration",
                 "enqueue_website_prepared_scene"):
        monkeypatch.setattr(website_task_context, name, _forbidden(name))
    for name in ("require_paid_resource_admission", "require_paid_resource_budget"):
        monkeypatch.setattr(paid_resource_admission, name, _forbidden(name))


def _record_today(job, monkeypatch):
    from blueprint_pipeline import website_atlas_preview, website_scene_handoff

    monkeypatch.setattr(website_atlas_preview, "enqueue_preview", lambda **_: job.calls.append("atlas_job"))

    def clean_plate(**_):
        job.calls.append("clean_plate")
        return {"status": "noop", "mode": "task_aware_prepared_images",
                "privacy_status": "website_capture_admitted", "privacy_verified": True}

    def handoff(**_):
        job.calls.append("scene_handoff")
        return {"status": "intake_ready", "website_intake_outbox": {"state": "forward_pending", "id": "scene-1"}}
    monkeypatch.setattr(orchestrator, "run_clean_plate_stage", clean_plate)
    monkeypatch.setattr(website_scene_handoff, "prepare_website_scene_handoff", handoff)


def test_without_a_request_the_stage_waits_having_bought_nothing(job, monkeypatch):
    _forbid_every_paid_or_atlas_call(monkeypatch)
    result = orchestrator.run_qualification_pipeline(descriptor_gcs_uri=job.uri, config=job.config)
    assert result["status"] == "completed" and result["native_execution_complete"] is False
    handoff = json.loads((job.capture / "pipeline/website_scene_preparation/handoff.json").read_text())
    assert result["website_scene_preparation"] == handoff
    assert handoff["status"] == simulation.WAITING_STATUS and handoff["blockers"] == []
    assert handoff["digest"] == canonical_digest(handoff, digest_field="digest")
    assert simulation.retains_awaiting_simulation_request(handoff, job.context)
    # What stays as today: the video normalized once, the task context and its evidence kept, the grant kept.
    assert job.calls == ["normalize", "task_context"]
    assert json.loads((job.capture / "pipeline/website_task_context.json").read_text()) == job.context
    assert (job.capture / f"pipeline/website_task_evidence/{job.context['context_digest'][7:]}/task_evidence.json").is_file()
    assert json.loads((job.capture / "pipeline/website_scene_sponsorship.json").read_text())["purpose"] == "scene_preparation"
    assert not (job.capture / "pipeline/atlas").exists()
    manifest = json.loads((job.capture / "pipeline/provider_run_manifest.json").read_text())
    assert manifest["status"] == "not_requested" and manifest["website_scene_preparation"] == handoff
    # run_e2e accepts it as the website preparation stage's receipt (a completed stage, not a failure).
    from blueprint_pipeline.run_e2e import _completed_website_preparation_only
    assert _completed_website_preparation_only({"status": "completed", "lanes": ["qualification"],
                                                "results": [{**result, "lane": "qualification"}]})


def test_with_a_request_today_s_path_runs_in_today_s_order(job, monkeypatch):
    assert simulation.record_request(capture_id=CAPTURE, body=_body(), storage_root=job.storage) == "recorded"
    _record_today(job, monkeypatch)
    result = orchestrator.run_qualification_pipeline(descriptor_gcs_uri=job.uri, config=job.config)
    assert result["status"] == "completed"
    assert result["website_scene_preparation"]["website_intake_outbox"]["state"] == "forward_pending"
    assert job.calls == ["atlas_job", "task_context", "normalize", "clean_plate", "scene_handoff"]
    assert not (job.capture / "pipeline/website_scene_preparation/handoff.json").exists()


@pytest.mark.parametrize("case", ["website_does_not_ask", "no_delivery_key", "false_value"])
def test_a_job_whose_website_does_not_ask_for_the_wait_builds_exactly_as_today(job, monkeypatch, case):
    """An older website never sends the field: its jobs build at once, whichever repository deploys first."""
    if case == "no_delivery_key":
        write_json(job.capture / "pipeline_job_ledger.json", {"status": "processing"})
    else:
        context = _seal({key: value for key, value in job.context.items() if key != simulation.CONTEXT_FIELD}
                        if case == "website_does_not_ask" else {**job.context, simulation.CONTEXT_FIELD: "yes"})
        write_json(job.capture / "pipeline/website_task_context.json", context)
        job.website["context"] = context
    _record_today(job, monkeypatch)
    orchestrator.run_qualification_pipeline(descriptor_gcs_uri=job.uri, config=job.config)
    assert job.calls == ["atlas_job", "task_context", "normalize", "clean_plate", "scene_handoff"]


# ------------------------------------------------------------- the listener, status and drains


@pytest.fixture
def site(prepared_source, monkeypatch, tmp_path):  # noqa: F811 - imported pytest fixture
    """A browser job whose website builds the simulation only on request, delivered through the real listener.

    ``run_e2e`` stands in for the orchestrator around the website preparation stage: its entry keeps the
    website's context (as ``prepare_website_task_descriptor`` does), then it takes the orchestrator's own switch.
    The build after a request is cut to its outcome (``state["build"]``): it fails retryably or completes.
    """
    root, owner, context, _, _, _ = prepared_source
    raw = _delivery(root, owner, monkeypatch)
    flagged = _seal({**context, simulation.CONTEXT_FIELD: True})
    website = {"current": flagged}
    monkeypatch.setattr(context_reader, "load_current_website_task_context", lambda **_: deepcopy(website["current"]))
    write_json(root / "pipeline/website_task_context.json", flagged)
    monkeypatch.setattr("blueprint_pipeline.website_rules_reopen.rules_reopen", _forbidden("the rules reopen"))
    state = {"build": "fail", "built": 0, "waited": 0}
    runs = []

    def run_e2e(**_):
        runs.append(1)
        current = deepcopy(website["current"])
        write_json(root / "pipeline/website_task_context.json", current)
        if simulation.waits_for_request(root):
            state["waited"] += 1
            write_json(root / "pipeline/website_scene_sponsorship.json", {"expires_at_epoch": time.time() + 7 * 86400})
            simulation.retain_awaiting_simulation_request(capture_root=root, pipeline_dir=root / "pipeline",
                                                          task_context=current)
            return {"status": "completed"}
        state["built"] += 1
        if state["build"] == "fail":
            raise PipelineError("website_scene_preparation: website_native_intake_pending")
        return {"status": "completed"}

    def deliver():
        listener.process_handoff_payload(raw, storage_root=root.parents[4], provider="openai", run_e2e=run_e2e)
        retain_preparation_wakeup(root)
        return json.loads((root / DELIVERY_FILE).read_text())["selectors"]

    def reconcile():
        return replay.reconcile_held_handoffs(listener, storage_root=root.parents[4], limit=1,
                                              process_args={"provider": "openai", "run_e2e": run_e2e})

    def ask(digest=DIGEST_A):
        return simulation.record_request(capture_id=owner["capture_id"], body=_body(owner["request_id"], digest),
                                         storage_root=root.parents[4])
    return SimpleNamespace(root=root, owner=owner, context=flagged, state=state, runs=runs, deliver=deliver,
                           reconcile=reconcile, ask=ask, run_e2e=run_e2e)


def test_a_waiting_job_completes_keeps_its_bytes_and_is_held_on_the_request(site):
    site.deliver()
    ledger = listener._read_job_ledger(site.root)
    assert ledger["status"] == "completed" and site.state == {"build": "fail", "built": 0, "waited": 1}
    assert ledger[replay.FIELD]["schema_version"] == replay.SCHEMA
    assert replay.held_kind(site.root / "pipeline") == simulation.HOLD_KIND
    assert replay.arrived_fact(site.root / "pipeline") is None  # never the rules reopen (it fails the test)
    assert site.reconcile() == {"replayed": 0, "held": 1, "already_replayed": 0} and len(site.runs) == 1


def test_the_waiting_status_is_readable_and_names_the_wait_only_to_a_website_that_lists_it(site, monkeypatch):
    selectors = site.deliver()
    for version in (STATUS_V1, STATUS_V6):
        status = read_preparation_status(capture_root=site.root, selectors=selectors, schema_version=version)
        assert status["state"] == "awaiting_inputs" and status["code"] == "preparation_inputs_pending"
        assert status.get("outcome") is None  # v3 to v6 readers never see the new reason
    monkeypatch.setenv(NEW_REASONS_ENV, "false")
    status = read_preparation_status(capture_root=site.root, selectors=selectors, schema_version=STATUS_V9)
    assert status["state"] == "awaiting_inputs" and status["outcome"] is None
    monkeypatch.setenv(NEW_REASONS_ENV, "true")
    status = read_preparation_status(capture_root=site.root, selectors=selectors, schema_version=STATUS_V9)
    handoff = json.loads((site.root / "pipeline/website_scene_preparation/handoff.json").read_text())
    assert status["state"] == "awaiting_inputs" and status["terminal_reason"] is None
    assert status["outcome"] == {"category": "held", "reason": "awaiting_simulation_request",
                                 "basis": "simulation_on_request", "robot_id": None,
                                 "evidence_digest": handoff["digest"]}
    assert status["packet_facts"] is None and status["question"] is None
    assert held_preparation(site.root) == []


def test_a_request_replays_the_stage_once_per_digest_and_its_failure_retries_by_itself(site):
    from blueprint_pipeline.website_preparation_retry import FIELD as RETRY_FIELD, reconcile_preparation_retries
    site.deliver()
    assert site.ask() == "recorded"
    site.reconcile()  # the replay runs; its build raises, so the drain does not count it as replayed
    assert site.state["built"] == 1 and len(site.runs) == 2
    fact = json.loads((site.root / "pipeline/website_scene_preparation" / replay.STATE_FILE).read_text())["fact"]
    assert fact["kind"] == simulation.FACT_KIND and fact["request_digest"] == DIGEST_A
    assert not (site.root / "pipeline/website_scene_preparation/rules_reopen.json").exists()
    ledger = listener._read_job_ledger(site.root)
    assert ledger["status"] == "failed_retryable" and ledger["attempt_history"][-1][replay.MARK] == fact["fact_digest"]
    # The build the request started failed: the host's own retry runs it again, as any preparation's.
    assert isinstance(ledger.get(RETRY_FIELD), dict)
    site.state["build"] = "complete"
    later = datetime.now(timezone.utc) + timedelta(days=1)
    assert reconcile_preparation_retries(listener, storage_root=site.root.parents[4],
                                         process_args={"provider": "openai", "run_e2e": site.run_e2e},
                                         now=later)["redriven"] == 1
    assert site.state["built"] == 2 and listener._read_job_ledger(site.root)["status"] == "completed"
    # The same request, or a second company's, starts nothing again: one simulation serves every company.
    assert site.ask() == "already_recorded" and site.ask(DIGEST_B) == "already_recorded"
    assert simulation.read_request(site.root)["request_digest"] == DIGEST_A
    assert site.reconcile() == {"replayed": 0, "held": 0, "already_replayed": 1} and site.state["built"] == 2


def test_a_withdrawn_capture_refuses_the_request_and_the_replay(site):
    site.deliver()
    assert site.ask() == "recorded"
    scene = site.root.parents[1]
    value = {"schema_version": "website_capture_withdrawal_tombstone.v1", "scene_id": scene.name,
             "request_id": scene.name[5:], "consent_revoked_at": "2026-11-28T12:00:00Z"}
    write_json(scene / "website_withdrawal/tombstone.json", {**value, "digest": canonical_digest(value, digest_field="digest")})
    site.reconcile()
    assert site.state["built"] == 0 and len(site.runs) == 1
    simulation.request_path(site.root).unlink()
    assert site.ask() == "capture_withdrawn" and simulation.read_request(site.root) is None


def test_nothing_treats_a_waiting_job_as_a_failure(site, tmp_path):
    from blueprint_pipeline import control_plane_stall_alarm as stall
    from blueprint_pipeline import preparation_fault_alert as fault
    from blueprint_pipeline.website_preparation_retry import reconcile_preparation_retries
    from blueprint_pipeline.website_preparation_terminal import (
        end_expired_preparation, end_no_grant_preparation, record_unretried_preparation,
    )
    site.deliver()
    storage = site.root.parents[4]
    before = listener._read_job_ledger(site.root)
    long_ago, later = datetime.now(timezone.utc) - timedelta(days=30), datetime.now(timezone.utc) + timedelta(days=30)
    # The retry lane, the overdue observation and both silence endings leave it as it is.
    counts = reconcile_preparation_retries(listener, storage_root=storage,
                                           process_args={"provider": "openai", "run_e2e": site.run_e2e}, now=later)
    assert counts["redriven"] == 0 and counts["adopted"] == 0 and counts["failed"] == 0
    assert record_unretried_preparation(site.root, alive_since=long_ago, now=later) is None
    assert end_expired_preparation(site.root, alive_since=long_ago, listener=listener, now=later) is None
    assert end_no_grant_preparation(site.root, alive_since=long_ago, listener=listener, now=later) is None
    # The preparation fault page and the Atlas stall page are quiet: no Atlas job exists to stall.
    assert fault.check_preparation_faults(storage, now=time.time())[1] == []
    assert not list(storage.glob(stall.ATLAS_JOB_GLOB))
    config = stall.StallConfig(scene_intent_root=tmp_path / "intents", launch_queue_root=tmp_path / "launches",
                               launch_state_root=tmp_path / "runs", deploy_tool=None, atlas_storage_root=storage)
    budget = stall._Budget(deadline=float("inf"), events=stall.MAX_EVENT_READS, clock=lambda: 0.0)
    assert stall.check_atlas_previews(config, now=time.time() + 30 * 86400, budget=budget)[1] == []
    assert listener._read_job_ledger(site.root) == before and len(site.runs) == 1


def test_an_internal_test_job_frees_its_slot_while_it_waits_and_takes_it_again_after_the_request(site, tmp_path):
    from blueprint_pipeline.internal_test_job_door import finished
    row = {"state": "submitted", "job_id": site.owner["request_id"], "capture_id": site.owner["capture_id"]}
    site.deliver()
    assert finished(row, site.root.parents[4], tmp_path / "intents") is True
    site.ask()
    site.reconcile()  # the request moves the ledger: the build fails retryably and its retry is armed
    assert finished(row, site.root.parents[4], tmp_path / "intents") is False


# --------------------------------------------------------------------------- the signed route


SIGNED = {"x-blueprint-pipeline-signature": "sha256=x", "x-test-client": "blueprint-webapp"}


def _client(storage_root: Path | None, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from blueprint_pipeline import website_atlas_preview_http as http

    def admission(request: Request):
        request.state.intake_client_id = request.headers.get("x-test-client", "")
    app = FastAPI()
    http.register_website_atlas_preview_routes(app, require_admission=admission)
    if storage_root is None:
        monkeypatch.delenv(http.HANDOFF_STORAGE_ROOT_ENV, raising=False)
    else:
        monkeypatch.setenv(http.HANDOFF_STORAGE_ROOT_ENV, str(storage_root))
    return TestClient(app)


def _post(client, body, capture_id: str = CAPTURE, headers=SIGNED):
    return client.post(f"/api/live-pipeline/website-simulation-requests/{capture_id}", headers=headers,
                       content=json.dumps(body))


def _handed_off(tmp_path: Path, capture_id: str = CAPTURE) -> Path:
    capture = tmp_path / f"storage/source-bucket/scenes/{SCENE}/captures/{capture_id}"
    (capture / "pipeline").mkdir(parents=True)
    return capture


def test_the_route_is_signed_webapp_only_idempotent_and_refuses_unknown_and_withdrawn_captures(tmp_path, monkeypatch):
    client = _client(tmp_path / "storage", monkeypatch)
    for headers in ({"x-test-client": "blueprint-webapp"}, {**SIGNED, "x-test-client": "robot-team"}):
        response = _post(client, _body(), headers=headers)
        assert response.status_code == 403
        assert response.json()["detail"] == "website simulation request issuer not authorized"
    # Not handed off yet (no capture, then no pipeline folder): the website asks again.
    assert _post(client, _body()).json() == {"detail": "capture_unknown"}
    capture = tmp_path / f"storage/source-bucket/scenes/{SCENE}/captures/{CAPTURE}"
    capture.mkdir(parents=True)
    assert _post(client, _body()).status_code == 404
    (capture / "pipeline").mkdir()
    for body in (_body(name="Acme Robotics"), _body(digest="sha256:short"), _body(source="site"),
                 _body(scene_id="site-other"), _body(requested_at_epoch=True), _body("other"), [1]):
        response = _post(client, body)
        assert response.status_code == 400 and response.json() == {"detail": "website_simulation_request_invalid"}
    first = _post(client, _body())
    assert first.status_code == 202 and first.json() == {"status": "recorded"}
    assert first.headers["cache-control"] == "private, no-store"
    record = simulation.read_request(capture)
    assert record["request_digest"] == DIGEST_A and record["source"] == "company" and record["capture_id"] == CAPTURE
    for digest in (DIGEST_A, DIGEST_B):
        again = _post(client, _body(digest=digest))
        assert again.status_code == 202 and again.json() == {"status": "already_recorded"}
    assert simulation.read_request(capture) == record
    # Another capture name for the job, or a supplement nobody handed off: unknown.
    assert _post(client, _body(), capture_id="supplement-" + "c" * 40).json() == {"detail": "capture_unknown"}
    # No storage root configured: nothing can be found, so nothing is recorded.
    assert _post(_client(None, monkeypatch), _body()).json() == {"detail": "capture_unknown"}
    value = {"schema_version": "website_capture_withdrawal_tombstone.v1", "scene_id": SCENE,
             "request_id": REQUEST_ID, "consent_revoked_at": "2026-11-28T12:00:00Z"}
    write_json(capture.parents[1] / "website_withdrawal/tombstone.json",
               {**value, "digest": canonical_digest(value, digest_field="digest")})
    withdrawn = _post(_client(tmp_path / "storage", monkeypatch), _body())
    assert withdrawn.status_code == 404 and withdrawn.json() == {"detail": "capture_withdrawn"}


def test_the_record_is_written_without_following_a_link(tmp_path):
    capture = _handed_off(tmp_path)
    storage = tmp_path / "storage"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    folder = capture.joinpath(*simulation.DIRECTORY)
    folder.symlink_to(elsewhere)
    assert simulation.record_request(capture_id=CAPTURE, body=_body(), storage_root=storage) == "capture_unknown"
    assert list(elsewhere.iterdir()) == []
    folder.unlink()
    folder.mkdir()
    (folder / simulation.MARK_FILE).symlink_to(elsewhere / "planted.json")
    assert simulation.record_request(capture_id=CAPTURE, body=_body(), storage_root=storage) == "capture_unknown"
    assert list(elsewhere.iterdir()) == [] and simulation.read_request(capture) is None
    (folder / simulation.MARK_FILE).unlink()
    (folder / simulation.MARK_FILE).write_text("{not json")  # a garbled record names no request
    assert simulation.read_request(capture) is None
    assert simulation.record_request(capture_id=CAPTURE, body=_body(), storage_root=storage) == "recorded"
    assert [path.name for path in folder.iterdir()] == [simulation.MARK_FILE]
    record = simulation.read_request(capture)
    assert set(record) - set(_body()) == {"schema_version", "capture_id", "received_at_epoch", "digest"}
    tampered = {**record, "request_digest": DIGEST_B}
    (folder / simulation.MARK_FILE).write_text(json.dumps(tampered))
    assert simulation.read_request(capture) is None  # the seal no longer holds


# -------------------------------------------------------------------------------- the company clip


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs local ffmpeg and ffprobe")
def test_the_clip_publishes_for_a_waiting_job_and_is_not_rebuilt_after_the_request(tmp_path, monkeypatch):
    from blueprint_pipeline import website_company_clip as clip
    from tests.test_website_company_clip import NOW, World, _capture

    monkeypatch.setattr(clip, "_wall_clock", lambda: NOW + 60)
    capture = _capture(tmp_path)  # normalized at hand-off, as the waiting path does
    context = {"scene_id": SCENE, "capture_id": CAPTURE, "context_digest": "sha256:" + "4" * 64}
    simulation.retain_awaiting_simulation_request(capture_root=capture, pipeline_dir=capture / "pipeline",
                                                  task_context=context)
    world = World()
    first = world.publish(capture)
    assert first["status"] == "published" and len(world.delivered) == 1
    assert simulation.record_request(capture_id=CAPTURE, body=_body(), storage_root=tmp_path / "storage") == "recorded"
    assert world.publish(capture, now=lambda: NOW + 120)["status"] != "published"
    assert len(world.archived) == 1 and len(world.delivered) == 1


# ------------------------------------------------------------------------------------- retention


def test_retention_keeps_a_waiting_capture_while_it_is_listed_or_its_test_grant_is_live(tmp_path, monkeypatch):
    from blueprint_pipeline import website_scene_workspace_retention as retention
    from tests.test_website_scene_workspace_retention import CAPTURE as KEPT, SCENE as KEPT_SCENE
    from tests.test_website_scene_workspace_retention import _context, _plan, _scene

    scene, cloud = _scene(tmp_path)
    capture = scene / "captures" / KEPT
    simulation.retain_awaiting_simulation_request(
        capture_root=capture, pipeline_dir=capture / "pipeline",
        task_context={"scene_id": KEPT_SCENE, "capture_id": KEPT, "context_digest": "sha256:" + "4" * 64})
    reason = f"{retention.AWAITING_SIMULATION_REQUEST_REASON}:{KEPT}"
    # A site's job nobody can see (its clip never published): no company can ask, so it retires as before.
    plan = _plan(tmp_path, cloud)
    assert plan["status"] == "retirable" and plan["reasons"] == []
    # Listed for companies: kept, even by a plan that does not keep shared workspaces.
    write_json(capture / "pipeline/company_clip/company_clip.json", {"archive": {"uri": "s3://bucket/clip.mp4"}})
    plan = _plan(tmp_path, cloud)
    assert plan["status"] == "retained" and plan["reasons"] == [reason]
    # Blueprint's own test job: kept while its scene grant is live, then retired as any finished test job.
    monkeypatch.setattr(retention, "internal_test_job", lambda _root: True)
    timer = _context(tmp_path, keep_shared_workspaces=True, retire_internal_test_jobs=True)
    write_json(capture / "pipeline/website_scene_sponsorship.json", {"expires_at_epoch": time.time() + 5 * 86400})
    assert _plan(tmp_path, cloud, context=timer)["reasons"] == [reason]
    later = _plan(tmp_path, cloud, age=6 * 86400, context=timer)
    assert later["status"] == "retirable" and later["reasons"] == []


# --------------------------------------------------------------------------------- test jobs


def _door():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))
    if not hasattr(sys.modules.get("operator_door"), "__path__"):
        sys.modules.pop("operator_door", None)
    from operator_door.requests import RequestRefused, validate_request
    return RequestRefused, validate_request


TEST_JOB = {"kind": "test-job", "idempotency_key": "20261201:library", "label": "library-aisle",
            "footage_id": "pexels-854533", "description": "Move the blue book to the top shelf."}


@pytest.mark.parametrize(("simulation_choice", "valid"), [(None, True), ("now", True), ("on_request", True),
                                                          ("later", False), (True, False)])
def test_the_door_takes_the_test_job_s_simulation_choice_and_passes_it_on(tmp_path, simulation_choice, valid):
    from blueprint_pipeline import internal_test_job_door as door_job

    refused, validate = _door()
    body = {**TEST_JOB, **({} if simulation_choice is None else {"simulation": simulation_choice})}
    if not valid:
        with pytest.raises(refused, match="^test_job_fields_invalid$"):
            validate(body)
        return
    request = validate(body)
    sent = []

    def website(payload):
        sent.append(dict(payload))
        return {"job_id": "itest-" + "a" * 40, "capture_id": "walkthrough-itest-" + "a" * 40, "footage": {"status": 201}}
    door_job.run(request, door_request_id="r", ledger_root=tmp_path / "ledger", handoff_root=tmp_path / "handoffs",
                 intent_root=tmp_path / "intents", transport=website, month=lambda: None, now=lambda: 1_800_000_000)
    expected = {key: TEST_JOB[key] for key in ("idempotency_key", "label", "footage_id", "description")}
    assert sent == [{**expected, **({} if simulation_choice is None else {"simulation": simulation_choice})}]


def test_the_door_client_sends_the_choice_only_when_it_is_given():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "operator_door.py"
    spec = importlib.util.spec_from_file_location("operator_door_client_simulation", path)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    sent = []
    client._submit = lambda body, args: sent.append(body) or 0
    argv = ["test-job", "--idempotency-key", TEST_JOB["idempotency_key"], "--label", TEST_JOB["label"],
            "--footage-id", TEST_JOB["footage_id"], "--description", TEST_JOB["description"]]
    for extra in ([], ["--simulation", "on_request"]):
        assert client.run(client.build_parser().parse_args(argv + extra)) == 0
    assert sent == [TEST_JOB, {**TEST_JOB, "simulation": "on_request"}]
