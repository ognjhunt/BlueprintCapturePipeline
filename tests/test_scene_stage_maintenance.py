"""The scene stage's maintenance pass does the kept duties of the dropped trial's tick, and only those."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import beta_policy_screening_capabilities as capabilities
from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_worker as worker
from blueprint_pipeline import rigid_robot_free_probe_runner as runner
from blueprint_pipeline import scene_stage_holds as holds
from blueprint_pipeline import scene_stage_maintenance as maintenance
from blueprint_pipeline import team_scene_packet_worker as packets
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.scene_run_cold_storage import COLD_STORAGE
from tests.test_beta_policy_screening_worker import prepared_source_queue

RELEASE = "a" * 40
SPENT = {"refusal_retry": {"attempts": 6, "exhausted": True}}


@pytest.fixture
def no_trial(monkeypatch):
    """Every trial-side entry point fails the test if the maintenance pass reaches it."""
    def forbidden(name):
        def call(*_args, **_kwargs):
            raise AssertionError(f"the maintenance pass called the trial's {name}")
        return call

    for module, name in ((coordinator, "process_screening_queue"), (coordinator, "enqueue_screening_campaign"),
                         (coordinator, "supersede_campaign"), (worker, "close_queued_rows"),
                         (capabilities, "build_planning_inputs"), (worker, "enqueue_prepared_scenes"),
                         (worker, "deliver_publications"), (worker, "publish_screening")):
        monkeypatch.setattr(module, name, forbidden(name))


def _block(world, status):
    key = packets.request_key(world["requests"][0])
    (world["packet_queue"] / "status" / f"{key}.json").write_text(json.dumps(status))
    return key


def test_main_recovers_the_packet_queue_runs_the_confirmation_exchange_and_delivers_todays_hold(
        tmp_path, monkeypatch, capsys, no_trial):
    from blueprint_pipeline import robot_access_furniture, safe_outbound_http, task_evaluation_launch_webapp_sync
    from blueprint_pipeline import website_owner_success_confirmation as confirmation

    world = prepared_source_queue(tmp_path, monkeypatch)
    _block(world, {"status": "blocked", "blockers": ["team_scene_packet_bundle_invalid"], "release_commit": RELEASE,
                   **SPENT})
    monkeypatch.setattr(packets, "current_release_commit", lambda: RELEASE)
    exchanged = []
    monkeypatch.setattr(confirmation, "exchange_owner_decisions",
                        lambda root, transport=None: exchanged.append(("owner", root)) or {"asked": 1})
    monkeypatch.setattr(robot_access_furniture, "exchange_robot_access_decisions",
                        lambda root, transport=None: exchanged.append(("robot_access", root)) or {})
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.example")
    monkeypatch.setattr(task_evaluation_launch_webapp_sync, "load_pipeline_sync_token", lambda: "unit-test-key")
    posted = []

    def webapp(url, **kwargs):
        body = json.loads(kwargs["data"])
        posted.append((url, body))
        return SimpleNamespace(body=json.dumps({"status": "created", "hold_id": body["hold_id"],
                                                "revision": body["revision"],
                                                "hold_digest": body["hold_digest"]}).encode())

    monkeypatch.setattr(safe_outbound_http, "request", webapp)
    assert maintenance.main(["--queue-root", str(world["screening_queue"]),
                             "--packet-queue-root", str(world["packet_queue"])]) == 0
    result = json.loads(capsys.readouterr().out)
    # The site's success-rule confirmation exchange ran inside the packet stage's recovery.
    assert [name for name, _root in exchanged] == ["owner", "robot_access"]
    assert result["sources"]["team_scene_packet_requeue"]["website_owner_decisions"] == {"asked": 1}
    # The hold reached the WebApp's holds route with today's schema, hold id and digest, and was acknowledged.
    request = world["requests"][0]
    [(url, hold)] = posted
    assert url == "https://webapp.example/api/internal/pipeline/beta-policy-screening-holds"
    assert hold["schema_version"] == "beta_policy_screening_hold.v1"
    assert hold["hold_id"] == holds.hold_id(request["intent_id"]) and hold["site_id"] == "site-job1"
    assert (hold["status"], hold["reason_code"], hold["detail_code"], hold["revision"]) == (
        "held", "team_scene_packet_blocked", "team_scene_packet_bundle_invalid", 1)
    assert hold["source"]["scene_revision_digest"] == request["scene_revision_digest"]
    assert hold["hold_digest"] == cross_runtime_canonical_digest(hold, digest_field="hold_digest")
    assert result["holds"] == {"acknowledged": 1, "pending": 0}
    # The same store the trial's pass kept: the screening queue root's holds directory.
    assert (world["screening_queue"] / "holds" / f"{hold['hold_id']}.json").is_file()


def test_a_completed_packet_queues_its_probe_and_is_held_or_released_on_the_probe_gate(
        tmp_path, monkeypatch, no_trial):
    from blueprint_pipeline import rigid_robot_free_probe_bundle as bundle
    from blueprint_pipeline import website_task_context

    world = prepared_source_queue(tmp_path, monkeypatch)
    queued = []
    monkeypatch.setattr(runner, "enqueue_rigid_robot_free_probe",
                        lambda capture: queued.append(capture) or {"status": "queued"})
    monkeypatch.setattr(runner, "source_commit", lambda: RELEASE)
    monkeypatch.setattr(bundle, "current_probe_state",
                        lambda _root: {"status": "receipt_decided", "receipt_status": "failed"})

    def refused(_profile):
        raise ValueError("team_evaluation_robot_free_probe_not_passed")

    monkeypatch.setattr(maintenance, "probe_gate", refused)
    kwargs = {"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"]}
    [held] = maintenance.scene_stage_pass(**kwargs)["sources"]
    # A packet compiled before the probe lane gets its probe queued (unpaid, idempotent).
    capture = tmp_path / "partition/scenes/site-job1/captures/walkthrough-job1"
    assert queued == [capture] and held["robot_free_probe_status"] == "queued"
    assert held["evaluation_hold"]["reason_code"] == "robot_free_probe_failed"
    # The probe passes: the scene stage moves on and the hold is released.
    monkeypatch.setattr(maintenance, "probe_gate", lambda _profile: {})
    [ready] = maintenance.scene_stage_pass(**kwargs)["sources"]
    assert ready["status"] == "scene_stage_ready" and ready["evaluation_hold"]["status"] == "released"
    # Settled: later passes read nothing remote for this request and queue nothing again.
    monkeypatch.setattr(website_task_context, "load_current_website_task_context",
                        lambda **_: pytest.fail("a settled request re-read its source"))
    [later] = maintenance.scene_stage_pass(**kwargs)["sources"]
    assert later["scene_stage"] == maintenance.PROBE_PASSED and "evaluation_hold" not in later
    assert len(queued) == 2


def test_a_team_half_refused_for_cold_storage_brings_its_run_back_and_is_held_by_name(tmp_path, monkeypatch, no_trial):
    world = prepared_source_queue(tmp_path, monkeypatch)
    # Blocked under the running release: a release the block has not seen would send it back to pending once
    # (``release_requeued``), which is the packet stage's rule, not the hold this test is about.
    _block(world, {"status": "blocked", "blockers": [COLD_STORAGE], "release_commit": RELEASE})
    monkeypatch.setattr(packets, "current_release_commit", lambda: RELEASE)
    restored = []
    monkeypatch.setattr(maintenance, "_restore_source_run",
                        lambda request, **kwargs: restored.append((request["intent_id"], kwargs)) or {"status": "refused"})
    [outcome] = maintenance.scene_stage_pass(queue_root=world["screening_queue"],
                                             packet_queue_root=world["packet_queue"])["sources"]
    assert restored == [("scene-intent-0", {"packet_queue_root": world["packet_queue"]})]
    assert outcome["scene_run_restore"] == {"status": "refused"}
    [held] = holds.pending_holds(queue_root=world["screening_queue"])
    assert (held["reason_code"], held["detail_code"]) == ("team_scene_packet_blocked", "scene_run_in_cold_storage")


def test_the_requeue_goes_through_its_one_function(tmp_path, monkeypatch, no_trial):
    world = prepared_source_queue(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(packets, "requeue_transient_blocked", lambda *, queue_root: seen.append(queue_root) or {})
    assert maintenance.recover_scene_packets(world["packet_queue"]) == {} and seen == [world["packet_queue"]]
    calls = []
    monkeypatch.setattr(maintenance, "recover_scene_packets", lambda root: calls.append(root) or {"status": "idle"})
    result = maintenance.scene_stage_pass(queue_root=world["screening_queue"], packet_queue_root=world["packet_queue"])
    assert calls == [world["packet_queue"]] and result["team_scene_packet_requeue"] == {"status": "idle"}


def test_the_trial_worker_no_longer_does_the_scene_stage_duties():
    import inspect

    source = inspect.getsource(worker.main)
    for duty in ("enqueue_prepared_scenes", "deliver_holds", "requeue_transient_blocked",
                 "enqueue_rigid_robot_free_probe", "_restore_source_run"):
        assert duty not in source, duty
