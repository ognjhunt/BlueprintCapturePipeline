"""A held movable-furniture measurement is tried again, bounded, instead of stalling its job.

#3357 measures a completed half's due furniture once per binding and holds a measurement that could not run.
One WebApp 503, Gemini or SAM error, or timeout then held the job forever. A transient cause is now tried
again once under the release that held it, after a wait, and any held measurement once under each newly
deployed release; a refusal (the job's caps or window, a changed binding) waits for a new release. The queue,
the run and the status are the worker's own; only the host reads and the paid measurement are stubbed.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

from blueprint_pipeline import agent_execution_offer
from blueprint_pipeline import robot_access_measurement as measurement
from blueprint_pipeline import team_candidate_quick10_registry as registry
from blueprint_pipeline import team_scene_packet_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

RELEASE_1, RELEASE_2, RELEASE_3 = "a" * 40, "b" * 40, "c" * 40
BINDING = "sha256:" + "2" * 64
T0 = datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc)
UNAVAILABLE = ValueError("website_control_preparation-spend_http_503:unclassified")
CAPS = ValueError("website_control_preparation-spend_http_409:preparation_spend_cap_exceeded")


class ProviderError(Exception):
    """A provider client's own error, carrying the HTTP status it got (as Gemini's client errors do)."""

    def __init__(self, code: int) -> None:
        super().__init__(f"{code} provider error")
        self.code = code


@pytest.mark.parametrize("exc,transient", [
    (TimeoutError("timed out"), True),
    (URLError("connection reset"), True),
    (UNAVAILABLE, True),
    (ValueError("meta_sam_http_429"), True),
    (ValueError("gemini_clean_plate_file_processing_timeout"), True),
    (ProviderError(503), True),
    (ProviderError(400), False),
    (CAPS, False),
    (ValueError("robot_access_measurement_binding_changed"), False),
    (ValueError("website_gemini_requires_reconciliation"), False),
    (ValueError("meta_sam_submission_requires_reconciliation"), False),
    (ValueError("website_preparation_spend_receipt_invalid"), False),
    (ValueError("meta_sam_http_5030"), False),
])
def test_a_measurement_failure_is_transient_only_when_a_later_try_can_clear_it(exc, transient):
    assert worker.measurement_transient(exc) is transient


def _world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *failures: BaseException) -> dict[str, Any]:
    """A completed request whose half has a furniture measurement due; each measurement raises the next failure."""

    queue = worker._queue(tmp_path / "queue")
    request = worker._seal({
        "schema_version": worker.REQUEST_SCHEMA, "intent_id": "intent-1", "intent_digest": "sha256:" + "3" * 64,
        "capture_id": "walkthrough-1", "task_id": "website-" + "0" * 20, "source_launch_root": str(tmp_path / "launch"),
        "launch_receipt_digest": "sha256:" + "1" * 64, "scene_revision_digest": "sha256:" + "4" * 64,
        "robot_free_readiness_status": "refused"}, "request_digest")
    key = worker.request_key(request)
    (queue / "processed" / f"{key}.json").write_text(json.dumps(request))
    (queue / "status" / f"{key}.json").write_text(json.dumps({
        "schema_version": worker.STATUS_SCHEMA, "request_key": key, "status": "completed", "blockers": [],
        "team_scene_inputs_path": "/halves/first/team_scene_inputs.v1.json", "release_commit": RELEASE_1}))
    calls: list[str] = []
    remaining = list(failures)

    def run_measurement(scope: Any, *, binding_digest: str, revision_digest: str) -> dict[str, Any]:
        calls.append(binding_digest)
        if remaining:
            raise remaining.pop(0)
        return {"sidecar_digest": canonical_digest({"sidecar": len(calls)}),
                "analysis": {"reserved_cost_usd": 0.04}, "segmentation": {"reserved_cost_usd": 0.18}}

    compiled: list[str] = []

    def compile_half(root: Path, request: Any, **_: Any) -> dict[str, Any]:
        compiled.append(request["request_digest"])
        return {"outcome": "compiled", "team_scene_inputs_path": "/halves/second/team_scene_inputs.v1.json"}

    due = {"binding_digest": BINDING}
    monkeypatch.setattr(measurement, "measurement_due", lambda path: dict(due) if path and "first" in path else None)
    monkeypatch.setattr(measurement, "run_measurement", run_measurement)
    monkeypatch.setattr(registry, "load_registry", lambda: {"profiles": []})
    monkeypatch.setattr(registry, "_derived_settings", lambda _registry: {"root": tmp_path / "profiles"})
    monkeypatch.setattr(registry, "profile_for_capture", lambda capture: {})
    monkeypatch.setattr(worker, "packet_inputs", lambda _registry: {})
    monkeypatch.setattr(worker, "resolve_capture_root", lambda _request: tmp_path / "capture")
    monkeypatch.setattr(worker, "_current_half", lambda profile, revision: "/halves/first/team_scene_inputs.v1.json")
    monkeypatch.setattr(worker, "owner_success_confirmation_scope", lambda _request: ({"intent": {}, "directory": "/x"}, None))
    monkeypatch.setattr(worker, "registered_embodiments", lambda _registry: ([], []))
    monkeypatch.setattr(worker, "_compile", compile_half)
    monkeypatch.setattr(agent_execution_offer, "publish_agent_execution_offer",
                        lambda capture, transport=None: {"status": "published"})
    return {"queue": queue, "key": key, "calls": calls, "compiled": compiled, "due": due}


def _run(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, *, minutes: float, release: str | None
         ) -> dict[str, Any]:
    moment = T0 + timedelta(minutes=minutes)
    monkeypatch.setattr(worker, "_utc_now", lambda: moment)
    monkeypatch.setattr(worker, "current_release_commit", lambda: release)
    return worker.process_team_scene_queue(queue_root=world["queue"])


def _status(world: dict[str, Any]) -> dict[str, Any]:
    return json.loads((world["queue"] / "status" / f"{world['key']}.json").read_text())


def _reasons(run: dict[str, Any]) -> list[tuple[str, str | None]]:
    return [(row["reason"], row.get("trigger")) for row in run["automatic_requeue"]]


def test_a_transient_hold_is_tried_again_once_per_release_until_it_measures(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, UNAVAILABLE, TimeoutError("timed out"), UNAVAILABLE)
    # The due binding is measured once, and the WebApp is unavailable: a transient hold, the half kept.
    first = _run(world, monkeypatch, minutes=0, release=RELEASE_1)
    assert _reasons(first) == [("robot_access_measurement_due", None)]
    held = _status(world)["robot_access_measurement"]
    assert held["status"] == "held" and held["cause"] == "transient" and held["binding_digest"] == BINDING
    assert held["hold"] == "team_scene_robot_access_measurement_held:website_control_preparation-spend_http_503"
    assert _status(world)["status"] == "completed" and world["compiled"] == []
    # Not before its wait.
    assert _run(world, monkeypatch, minutes=10, release=RELEASE_1)["status"] == "idle"
    # After it, once under the release that held it: the timeout holds it again.
    again = _run(world, monkeypatch, minutes=31, release=RELEASE_1)
    assert _reasons(again) == [("robot_access_measurement_retry", "transient_wait_elapsed")]
    assert again["automatic_requeue"][0]["cause"] == "transient"
    assert _status(world)[worker.ROBOT_ACCESS_MEASUREMENT_RETRY_FIELD] == [RELEASE_1]
    assert _status(world)[worker.ROBOT_ACCESS_MEASUREMENT_REQUEUE_FIELD] == [BINDING]
    assert len(world["calls"]) == 2
    # This release has had its try: nothing more, however long it waits.
    assert _run(world, monkeypatch, minutes=600, release=RELEASE_1)["status"] == "idle"
    # A newly deployed release tries at once; held again, it then waits out its own wait under that release.
    newer = _run(world, monkeypatch, minutes=601, release=RELEASE_2)
    assert _reasons(newer) == [("robot_access_measurement_retry", "new_release")]
    assert _status(world)["robot_access_measurement"]["status"] == "held"
    assert _run(world, monkeypatch, minutes=700, release=RELEASE_2)["status"] == "idle"
    # The next release measures it, and only a measurement that ran compiles the half again.
    measured = _run(world, monkeypatch, minutes=701, release=RELEASE_3)
    assert _reasons(measured) == [("robot_access_measurement_retry", "new_release")]
    status = _status(world)
    assert status["robot_access_measurement"]["status"] == "measured" and status["outcome"] == "compiled"
    assert world["calls"] == [BINDING] * 4 and len(world["compiled"]) == 1
    assert status[worker.ROBOT_ACCESS_MEASUREMENT_RETRY_FIELD] == [RELEASE_1, RELEASE_2, RELEASE_3]


def test_a_refusal_stays_held_until_a_new_release_then_is_tried_once(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, CAPS, CAPS)
    _run(world, monkeypatch, minutes=0, release=RELEASE_1)
    held = _status(world)["robot_access_measurement"]
    assert held["cause"] == "refusal"
    assert held["hold"] == "team_scene_robot_access_measurement_held:website_control_preparation-spend_http_409"
    # The job's caps do not change by waiting.
    for minutes in (31, 600):
        assert _run(world, monkeypatch, minutes=minutes, release=RELEASE_1)["status"] == "idle"
    # Without a known running release nothing moves.
    assert _run(world, monkeypatch, minutes=601, release=None)["status"] == "idle"
    assert len(world["calls"]) == 1
    newer = _run(world, monkeypatch, minutes=602, release=RELEASE_2)
    assert _reasons(newer) == [("robot_access_measurement_retry", "new_release")]
    assert newer["automatic_requeue"][0]["cause"] == "refusal"
    for minutes in (640, 2000):
        assert _run(world, monkeypatch, minutes=minutes, release=RELEASE_2)["status"] == "idle"
    assert len(world["calls"]) == 2 and world["compiled"] == []


def test_a_hold_recorded_before_its_cause_was_kept_is_tried_once_by_the_next_release(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, CAPS)
    _run(world, monkeypatch, minutes=0, release=RELEASE_1)
    path = world["queue"] / "status" / f"{world['key']}.json"
    status = json.loads(path.read_text())
    # As #3357 wrote it: no cause and no time.
    status["robot_access_measurement"] = {key: value for key, value in status["robot_access_measurement"].items()
                                          if key not in {"cause", "held_at"}}
    path.write_text(json.dumps(status))
    assert _run(world, monkeypatch, minutes=600, release=RELEASE_1)["status"] == "idle"
    assert _reasons(_run(world, monkeypatch, minutes=601, release=RELEASE_2)) == [
        ("robot_access_measurement_retry", "new_release")]
    assert _status(world)["robot_access_measurement"]["status"] == "measured"


def test_a_held_measurement_of_a_binding_no_longer_due_is_not_tried_again(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, UNAVAILABLE)
    _run(world, monkeypatch, minutes=0, release=RELEASE_1)
    # The half's binding changed: the new binding is measured once as newly due, never as a retry of the old.
    world["due"]["binding_digest"] = "sha256:" + "5" * 64
    run = _run(world, monkeypatch, minutes=31, release=RELEASE_2)
    assert _reasons(run) == [("robot_access_measurement_due", None)]
    assert world["calls"] == [BINDING, "sha256:" + "5" * 64]
    assert worker.ROBOT_ACCESS_MEASUREMENT_RETRY_FIELD not in _status(world)
