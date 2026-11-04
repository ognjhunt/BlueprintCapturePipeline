"""A website job whose evaluation cannot start after scene publication is held by name, and released when it moves."""
import json

import pytest

from blueprint_pipeline import beta_policy_screening_holds as holds
from blueprint_pipeline import beta_policy_screening_worker as worker
from blueprint_pipeline import rigid_robot_free_probe_bundle as bundle
from blueprint_pipeline import rigid_robot_free_probe_runner as runner
from blueprint_pipeline import team_scene_packet_worker as packets
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from tests.test_beta_policy_screening_worker import prepared_source_queue

RELEASE, NEXT_RELEASE = "a" * 40, "b" * 40


def _probe_state(tmp_path, status="probe_required", **extra):
    request = {"request_digest": canonical_digest({"probe": 1})}
    return {"status": status, "capture_root": str(tmp_path / "capture"), "request": request, **extra}


def _seal_terminal(tmp_path, state, status, release):
    job = runner._job_dir(tmp_path / "probes", state)
    job.mkdir(parents=True)
    (job / runner.TERMINAL_FILENAME).write_text(json.dumps({"status": status, runner.RELEASE_COMMIT_FIELD: release}))


@pytest.mark.parametrize("state,terminal,expected", [
    ({"status": "receipt_decided", "receipt_status": "failed"}, None, ("robot_free_probe_failed", "not_passed")),
    ({"status": "receipt_decided", "receipt_status": "passed"}, None, ("robot_free_probe_not_completed", "not_passed")),
    ({"status": "unavailable_for_task", "task_kind": "rigid_pick_place"}, None,
     ("robot_free_probe_unavailable_for_task", "not_passed")),
    ({"status": "unavailable_for_task", "task_kind": "articulated_open_close"}, None,
     ("robot_free_probe_not_completed", "not_passed")),
    ({"status": "scene_revision_unbound"}, None, ("robot_free_probe_not_completed", "scene_revision_unbound")),
    # Sealed by the running release: nothing retries it.
    ({}, ("staging_refused", RELEASE), ("robot_free_probe_not_completed", "staging_refused")),
    ({}, ("attempts_exhausted", RELEASE), ("robot_free_probe_not_completed", "attempts_exhausted")),
    ({}, ("trace_conflict", NEXT_RELEASE), ("robot_free_probe_not_completed", "trace_conflict")),
    # A staging refusal the running release has not tried yet is retried by its next pass: no hold.
    ({}, ("staging_refused", NEXT_RELEASE), None),
    # Queued, waiting or running: still being evaluated.
    ({}, None, None),
    ({"status": "probe_root_unavailable"}, None, None),
])
def test_a_probe_refusal_is_held_only_once_nothing_will_decide_the_probe(tmp_path, monkeypatch, state, terminal,
                                                                         expected):
    current = {**_probe_state(tmp_path), **state}
    if terminal:
        _seal_terminal(tmp_path, current, *terminal)
    monkeypatch.setattr(bundle, "current_probe_state", lambda _root: dict(current))
    assert holds.probe_hold("team_evaluation_robot_free_probe_not_passed", tmp_path / "capture",
                            queue_root=tmp_path / "probes", release=RELEASE) == expected


def test_a_task_with_no_probe_is_held_as_a_capability_limit_without_reading_the_lane(tmp_path):
    assert holds.probe_hold("team_evaluation_robot_free_probe_unavailable_for_task:pour", tmp_path) == (
        "robot_free_probe_unavailable_for_task", "unavailable_for_task:pour")
    assert holds.probe_hold("beta_screening_scene_profile_unavailable", tmp_path) is None


@pytest.mark.parametrize("status,expected", [
    ({"status": "blocked", "blockers": ["team_scene_packet_bundle_invalid"], "release_commit": RELEASE},
     ("team_scene_packet_blocked", "team_scene_packet_bundle_invalid")),
    # Retried by this release already.
    ({"status": "blocked", "blockers": ["team_scene_packet_bundle_invalid"], "release_commit": NEXT_RELEASE,
      "release_requeued": [RELEASE]}, ("team_scene_packet_blocked", "team_scene_packet_bundle_invalid")),
    # Automatic retries spent.
    ({"status": "blocked", "blockers": ["team_scene_packet_interrupted"], "release_commit": RELEASE,
      "automatic_retry": {"attempts": len(packets.AUTOMATIC_RETRY_BACKOFF_SECONDS)}},
     ("team_scene_packet_blocked", "team_scene_packet_interrupted")),
    # Still due an automatic retry, a new release's try, or the owner's confirmation; or not blocked.
    ({"status": "blocked", "blockers": ["team_scene_packet_interrupted"], "release_commit": RELEASE,
      "automatic_retry": {"attempts": 1}}, None),
    ({"status": "blocked", "blockers": ["team_scene_packet_bundle_invalid"], "release_commit": NEXT_RELEASE}, None),
    ({"status": "blocked", "blockers": [packets.OWNER_CONTRACT_BLOCKER_PREFIX + "proposal_only"],
      "release_commit": RELEASE}, None),
    ({"status": "processing"}, None),
    # A withdrawn capture ends on the website itself, never as a hold.
    ({"status": "blocked", "blockers": ["team_scene_capture_withdrawn"], "release_commit": RELEASE}, None),
])
def test_a_team_half_block_is_held_only_once_nothing_returns_it_to_the_queue(status, expected):
    assert holds.packet_hold(status, release=RELEASE) == expected
    # Without a known running release nothing is final.
    assert holds.packet_hold(status, release=None) is None


def test_worker_holds_a_failed_probe_by_name_and_releases_it_when_the_probe_later_passes(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_capabilities as capabilities

    world = prepared_source_queue(tmp_path, monkeypatch)
    planning = capabilities.build_planning_inputs
    monkeypatch.setattr(runner, "enqueue_rigid_robot_free_probe", lambda _capture: {"status": "already_queued"})
    monkeypatch.setattr(runner, "source_commit", lambda: RELEASE)
    state = {**_probe_state(tmp_path), "status": "receipt_decided", "receipt_status": "failed"}
    monkeypatch.setattr(bundle, "current_probe_state", lambda _root: dict(state))

    def refused(**_kwargs):
        raise ValueError("team_evaluation_robot_free_probe_not_passed")

    monkeypatch.setattr(capabilities, "build_planning_inputs", refused)
    kwargs = {"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"]}
    [outcome] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert outcome["blockers"] == ["team_evaluation_robot_free_probe_not_passed"]
    assert outcome["evaluation_hold"] == {"status": "held", "reason_code": "robot_free_probe_failed", "revision": 1,
                                          "change": "recorded"}
    [held] = holds.pending_holds(queue_root=world["screening_queue"])
    request = world["requests"][0]
    assert held["hold_id"] == holds.hold_id(request["intent_id"]) and held["site_id"] == "site-job1"
    assert held["source"] == {"scene_intent_id": request["intent_id"], "scene_intent_digest": request["intent_digest"],
                              "capture_id": "walkthrough-job1", "task_id": request["task_id"],
                              "task_context_digest": world["context"]["context_digest"],
                              "scene_revision_digest": request["scene_revision_digest"]}
    assert (held["status"], held["reason_code"], held["detail_code"]) == ("held", "robot_free_probe_failed",
                                                                         "not_passed")
    assert held["hold_digest"] == cross_runtime_canonical_digest(held, digest_field="hold_digest")
    # The same decision on the next pass is the same bytes.
    [again] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert again["evaluation_hold"]["change"] == "unchanged"
    sent = []
    assert holds.deliver_holds(queue_root=world["screening_queue"], transport=sent.append) == {
        "acknowledged": 1, "pending": 0}
    assert sent == [held] and not holds.pending_holds(queue_root=world["screening_queue"])
    # A re-driven probe passes: the campaign is planned, and the same hold is published again as released.
    monkeypatch.setattr(capabilities, "build_planning_inputs", planning)
    [planned] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert planned["status"] == "enqueued"
    assert planned["evaluation_hold"] == {"status": "released", "reason_code": None, "revision": 2,
                                          "change": "recorded"}
    [released] = holds.pending_holds(queue_root=world["screening_queue"])
    assert (released["hold_id"], released["status"], released["revision"]) == (held["hold_id"], "released", 2)
    # Released once: later passes publish nothing more.
    assert holds.deliver_holds(queue_root=world["screening_queue"], transport=sent.append)["acknowledged"] == 1
    [later] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert "evaluation_hold" not in later and not holds.pending_holds(queue_root=world["screening_queue"])


def test_worker_holds_a_team_half_no_pass_will_retry_and_lets_a_new_release_try_it_first(tmp_path, monkeypatch):
    world = prepared_source_queue(tmp_path, monkeypatch)
    key = packets.request_key(world["requests"][0])
    (world["packet_queue"] / "status" / f"{key}.json").write_text(json.dumps(
        {"status": "blocked", "blockers": ["team_scene_packet_bundle_invalid"], "release_commit": RELEASE}))
    monkeypatch.setattr(packets, "current_release_commit", lambda: RELEASE)
    kwargs = {"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"]}
    [outcome] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert outcome["blockers"] == ["beta_screening_scene_packet_not_ready"]
    assert outcome["evaluation_hold"]["reason_code"] == "team_scene_packet_blocked"
    # A new release returns it to the queue on this pass, so the hold is released while it runs again.
    monkeypatch.setattr(packets, "current_release_commit", lambda: NEXT_RELEASE)
    result = worker.enqueue_prepared_scenes(**kwargs)
    assert result["team_scene_packet_requeue"]["requeued"][0]["reason"] == "new_release"
    [released] = holds.pending_holds(queue_root=world["screening_queue"])
    assert (released["status"], released["revision"]) == ("released", 2)


def test_scene_preparation_failed_receipt_is_kept_and_held_without_a_team_half(tmp_path, monkeypatch):
    from tests.test_beta_policy_screening_worker import source_world

    source, context = source_world(tmp_path / "partition", monkeypatch)
    queue = tmp_path / "packets"
    launch = tmp_path / "launch"
    launch.mkdir()
    intent = {"intent_id": "scene-intent-0", "intent_digest": canonical_digest({"intent": 0}),
              "request": {"submission_id": source["capture_id"], "task": {"task_id": source["task_id"]}}}
    revision = canonical_digest({"configured_scene_revision": 0})
    queued = packets.enqueue_team_scene_request(intent=intent, run_root=launch,
        receipt={"receipt_digest": canonical_digest({"r": 0})}, result={"configured_scene_revision_digest": revision},
        readiness={"status": "failed", "retained": True}, queue_root=queue)
    key = queued["hold_request_key"]
    assert queued == {"status": "not_enqueued", "reason": "robot_free_probe_failed", "hold_request_key": key}
    assert not list((queue / "pending").glob("*.json"))
    kwargs = {"queue_root": tmp_path / "screening", "packet_queue_root": queue}
    # The receipt is re-read: without a decided failed receipt beside the launch, nothing is held.
    [refused] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert refused["blockers"] == ["beta_screening_hold_request_invalid"]
    receipt = {"schema_version": "task_evaluation_configured_scene_robot_free_readiness.v1", "status": "failed",
               "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (launch / "robot_free_readiness.v1.json").write_text(json.dumps(receipt))
    [held] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert held["request_key"] == key and held["status"] == "held"
    assert held["evaluation_hold"]["reason_code"] == "robot_free_probe_failed"
    [publication] = holds.pending_holds(queue_root=tmp_path / "screening")
    assert publication["source"]["scene_revision_digest"] == revision
    assert publication["source"]["task_context_digest"] == context["context_digest"]


def test_a_hold_reaches_the_website_only_with_its_exact_acknowledgment(tmp_path, monkeypatch):
    from blueprint_pipeline import safe_outbound_http, task_evaluation_launch_webapp_sync

    publication = {"hold_id": "hold-" + "0" * 32, "revision": 3, "hold_digest": "sha256:" + "1" * 64}
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.example")
    monkeypatch.setattr(task_evaluation_launch_webapp_sync, "load_pipeline_sync_token", lambda: "unit-test-key")
    answers = [{"status": "created", **publication}, {"status": "created", **publication, "revision": 2}]
    seen = []

    def send(url, **kwargs):
        seen.append(url)
        return type("Response", (), {"body": json.dumps(answers.pop(0)).encode()})()

    monkeypatch.setattr(safe_outbound_http, "request", send)
    assert holds.publish_hold(publication)["status"] == "created"
    assert seen == ["https://webapp.example/api/internal/pipeline/beta-policy-screening-holds"]
    with pytest.raises(ValueError, match="beta_screening_hold_ack_invalid"):
        holds.publish_hold(publication)


def _failed_receipt(launch):
    launch.mkdir(parents=True, exist_ok=True)
    receipt = {"schema_version": "task_evaluation_configured_scene_robot_free_readiness.v1", "status": "failed",
               "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (launch / "robot_free_readiness.v1.json").write_text(json.dumps(receipt))


def _held_request(source, queue, launch, *, intent="scene-intent-0", revision=0, readiness="failed"):
    _failed_receipt(launch)
    request = {"intent_id": intent, "intent_digest": canonical_digest({"intent": intent}),
               "request": {"submission_id": source["capture_id"], "task": {"task_id": source["task_id"]}}}
    return packets.enqueue_team_scene_request(intent=request, run_root=launch,
        receipt={"receipt_digest": canonical_digest({"r": revision})},
        result={"configured_scene_revision_digest": canonical_digest({"configured_scene_revision": revision})},
        readiness={"status": readiness, "retained": True}, queue_root=queue)


def test_an_error_or_unreadable_state_keeps_a_hold_and_never_releases_it(tmp_path, monkeypatch):
    """A failed probe stays held through a pass that cannot read it: no release, no re-hold, no lost email."""
    from blueprint_pipeline import beta_policy_screening_capabilities as capabilities

    world = prepared_source_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "enqueue_rigid_robot_free_probe", lambda _capture: {"status": "already_queued"})
    failed = {**_probe_state(tmp_path), "status": "receipt_decided", "receipt_status": "failed"}
    monkeypatch.setattr(bundle, "current_probe_state", lambda _root: dict(failed))

    def refused(**_kwargs):
        raise ValueError("team_evaluation_robot_free_probe_not_passed")

    monkeypatch.setattr(capabilities, "build_planning_inputs", refused)
    kwargs = {"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"]}
    assert worker.enqueue_prepared_scenes(**kwargs)["sources"][0]["evaluation_hold"]["revision"] == 1

    def unreadable(_root):
        raise OSError(5, "Input/output error")

    def transient(**_kwargs):
        raise OSError(28, "No space left on device")

    # probe_hold itself fails, then planning fails for a reason that is not the probe.
    for patch in ((bundle, "current_probe_state", unreadable), (capabilities, "build_planning_inputs", transient)):
        monkeypatch.setattr(*patch)
        [outcome] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
        assert "evaluation_hold" not in outcome
        [kept] = holds.pending_holds(queue_root=world["screening_queue"])
        assert (kept["status"], kept["revision"]) == ("held", 1)
    monkeypatch.setattr(bundle, "current_probe_state", lambda _root: dict(failed))
    monkeypatch.setattr(capabilities, "build_planning_inputs", refused)
    assert worker.enqueue_prepared_scenes(**kwargs)["sources"][0]["evaluation_hold"]["change"] == "unchanged"
    [kept] = holds.pending_holds(queue_root=world["screening_queue"])
    assert (kept["status"], kept["revision"]) == ("held", 1)


def test_a_newer_scene_revision_of_the_intent_supersedes_a_held_older_one_without_churn(tmp_path, monkeypatch):
    import os

    from tests.test_beta_policy_screening_worker import source_world

    source, _context = source_world(tmp_path / "partition", monkeypatch)
    queue, screening = tmp_path / "packets", tmp_path / "screening"
    old = _held_request(source, queue, tmp_path / "launch-r1", revision=1)
    kwargs = {"queue_root": screening, "packet_queue_root": queue}
    worker.enqueue_prepared_scenes(**kwargs)
    [held] = holds.pending_holds(queue_root=screening)
    assert (held["status"], held["source"]["scene_revision_digest"]) == (
        "held", canonical_digest({"configured_scene_revision": 1}))
    # Revision 2 of the same intent is compiled later (its readiness passed); its team half is done.
    newer = _held_request(source, queue, tmp_path / "launch-r2", revision=2, readiness="passed")
    key = newer["request_key"]
    (queue / "pending" / f"{key}.json").rename(queue / "processed" / f"{key}.json")
    (queue / "status" / f"{key}.json").write_text(json.dumps({"status": "completed"}))
    stamp = (queue / "holds" / f"{old['hold_request_key']}.json").stat().st_mtime_ns
    os.utime(queue / "processed" / f"{key}.json", ns=(stamp + 10**9, stamp + 10**9))
    worker.enqueue_prepared_scenes(**kwargs)
    [released] = holds.pending_holds(queue_root=screening)
    assert (released["status"], released["revision"]) == ("released", 2)
    assert released["source"]["scene_revision_digest"] == canonical_digest({"configured_scene_revision": 2})
    assert (queue / "holds" / "superseded" / f"{old['hold_request_key']}.json").is_file()
    # Later passes add no revisions: no flipping between the two revisions.
    for _ in range(3):
        worker.enqueue_prepared_scenes(**kwargs)
    [again] = holds.pending_holds(queue_root=screening)
    assert again == released


def test_held_requests_rotate_under_a_cursor_and_are_set_aside_once_acknowledged(tmp_path, monkeypatch):
    from tests.test_beta_policy_screening_worker import source_world

    source, _context = source_world(tmp_path / "partition", monkeypatch)
    queue, screening = tmp_path / "packets", tmp_path / "screening"
    keys = sorted(_held_request(source, queue, tmp_path / f"launch-{index}", intent=f"scene-intent-{index}",
                                revision=index)["hold_request_key"] for index in range(3))
    seen = [worker.enqueue_prepared_scenes(queue_root=screening, packet_queue_root=queue, max_sources=1)
            ["sources"][0]["request_key"] for _ in range(3)]
    assert seen == keys
    assert holds.deliver_holds(queue_root=screening, transport=lambda _publication: None)["acknowledged"] == 3
    for _ in range(3):
        worker.enqueue_prepared_scenes(queue_root=screening, packet_queue_root=queue, max_sources=1)
    assert not list((queue / "holds").glob("*.json"))
    assert sorted(path.stem for path in (queue / "holds" / "settled").glob("*.json")) == keys
    assert not holds.pending_holds(queue_root=screening)


@pytest.mark.parametrize("answer,accepted", [
    ({"status": "replayed", "revision": 5, "hold_digest": "sha256:" + "2" * 64}, True),
    ({"status": "replayed", "revision": 3, "hold_digest": "sha256:" + "1" * 64}, True),
    ({"status": "replayed", "revision": 2, "hold_digest": "sha256:" + "2" * 64}, False),
    ({"status": "updated", "revision": 5, "hold_digest": "sha256:" + "2" * 64}, False),
])
def test_a_replayed_answer_for_this_or_a_newer_revision_acknowledges_the_hold(monkeypatch, answer, accepted):
    from blueprint_pipeline import safe_outbound_http, task_evaluation_launch_webapp_sync

    publication = {"hold_id": "hold-" + "0" * 32, "revision": 3, "hold_digest": "sha256:" + "1" * 64}
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.example")
    monkeypatch.setattr(task_evaluation_launch_webapp_sync, "load_pipeline_sync_token", lambda: "unit-test-key")
    body = json.dumps({"hold_id": publication["hold_id"], **answer}).encode()
    monkeypatch.setattr(safe_outbound_http, "request", lambda url, **_kwargs: type("Response", (), {"body": body})())
    if accepted:
        assert holds.publish_hold(publication)["status"] == answer["status"]
    else:
        with pytest.raises(ValueError, match="beta_screening_hold_ack_invalid"):
            holds.publish_hold(publication)
