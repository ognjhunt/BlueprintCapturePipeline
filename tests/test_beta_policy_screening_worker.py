"""Source consent and durable signed publication at the automatic worker seam."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import beta_policy_screening_worker as worker
from blueprint_pipeline import website_task_context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_scene_packet_worker import CAPTURE_PARTITION_ENV


def source_world(tmp_path, monkeypatch):
    context = {"schema_version": "website_site_task_context.v1", "request_id": "job1", "scene_id": "site-job1",
        "capture_id": "walkthrough-job1", "purpose": "scene_preparation", "confirmed": False,
        "confirmed_at": None, "description": "Move a box", "capture_rights": {
            "consent_status": "granted", "consent_revoked": False,
            "derived_scene_generation_allowed": True, "consent_scope": ["robot_evaluation"]}}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    capture = tmp_path / "scenes/site-job1/captures/walkthrough-job1"
    (capture / "pipeline").mkdir(parents=True)
    (capture / "capture_descriptor.json").write_text(json.dumps({"scene_id": context["scene_id"],
        "capture_id": context["capture_id"]}))
    (capture / "pipeline/website_task_context.json").write_text(json.dumps(context))
    monkeypatch.setenv(CAPTURE_PARTITION_ENV, str(tmp_path))
    source = {"capture_id": context["capture_id"], "task_context_digest": context["context_digest"],
              "task_id": "website-" + context["context_digest"][7:27]}
    monkeypatch.setattr(website_task_context, "load_current_website_task_context", lambda **_: deepcopy(context))
    return source, context


def test_current_source_preserves_proposed_diagnostic_task_and_stable_site(tmp_path, monkeypatch):
    source, context = source_world(tmp_path, monkeypatch)
    seen = []
    def load(**kwargs):
        seen.append(kwargs)
        return deepcopy(context)
    monkeypatch.setattr(website_task_context, "load_current_website_task_context", load)
    assert worker.require_current_source(source, site_id="site-job1") == context
    assert seen == [{"request_id": "job1", "scene_id": "site-job1", "capture_id": "walkthrough-job1",
                     "purpose": "scene_preparation"}]
    with pytest.raises(ValueError, match="source_context_mismatch"):
        worker.require_current_source(source, site_id="site-another-job")


@pytest.mark.parametrize("change", ["consent", "task"])
def test_current_source_refuses_withdrawn_consent_or_changed_task(tmp_path, monkeypatch, change):
    source, context = source_world(tmp_path, monkeypatch)
    if change == "consent":
        context["capture_rights"]["consent_status"] = "revoked"
    else:
        context["context_digest"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="current_capture_consent_required|current_task_context_changed"):
        worker.require_current_source(source, site_id="site-job1")


def test_signed_delivery_requires_exact_publication_ack(monkeypatch):
    from blueprint_pipeline import safe_outbound_http, task_evaluation_launch_webapp_sync
    publication = {"campaign_id": "campaign1", "revision": 4, "publication_digest": "sha256:" + "e" * 64}
    ack = {"status": "replayed", "run_id": "campaign1", "revision": 4,
           "publication_digest": publication["publication_digest"]}
    requests = []
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://app.example.test/api/internal/pipeline/sync")
    monkeypatch.setattr(task_evaluation_launch_webapp_sync, "load_pipeline_sync_token", lambda: "unit-test-key")
    def send(url, **kwargs):
        requests.append((url, kwargs))
        return SimpleNamespace(body=json.dumps(ack).encode())
    monkeypatch.setattr(safe_outbound_http, "request", send)
    assert worker.publish_screening(publication) == ack
    assert requests[0][0] == "https://app.example.test/api/internal/pipeline/beta-policy-screening-runs"
    assert json.loads(requests[0][1]["data"]) == publication
    assert requests[0][1]["headers"]
    ack["revision"] = 5
    with pytest.raises(ValueError, match="publication_ack_invalid"):
        worker.publish_screening(publication)


def test_failed_delivery_retains_durable_outbox_and_retry_acknowledges(tmp_path):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from tests.test_beta_policy_screening_coordinator import plan
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    def unavailable(_):
        raise OSError("temporary network failure")
    assert worker.deliver_publications(queue_root=tmp_path, transport=unavailable) == {"acknowledged": 0, "pending": 1}
    pending = coordinator.pending_publications(queue_root=tmp_path)
    received = []
    assert worker.deliver_publications(queue_root=tmp_path, transport=received.append) == {"acknowledged": 1, "pending": 0}
    assert received == pending
    assert coordinator.pending_publications(queue_root=tmp_path) == []


@pytest.mark.parametrize("error,category", [
    (ValueError("beta_screening_publication_http_403"), "auth"),
    (ValueError("beta_screening_publication_http_402"), "source"),
    (ValueError("beta_screening_publication_http_409"), "source"),
    (ValueError("beta_screening_publication_http_429"), "unavailable"),
    (ValueError("beta_screening_publication_http_307"), "unknown"),
    (ValueError("beta_screening_publication_http_503"), "unavailable"),
    (ValueError("beta_screening_publication_ack_invalid"), "ack"),
    (ValueError("beta_screening_publication_http_403 private token=do-not-expose"), "unknown"),
    (UnicodeError("private remote response token=do-not-expose"), "ack"),
    (json.JSONDecodeError("private remote response", "token=do-not-expose", 0), "ack"),
    (OSError("private remote URL token=do-not-expose"), "unavailable"),
    (RuntimeError("private customer content"), "unknown"),
])
def test_opt_in_publication_failure_details_are_only_safe_enums(tmp_path, error, category):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from tests.test_beta_policy_screening_coordinator import plan
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    def send(_publication):
        raise error
    result = worker.deliver_publications(queue_root=tmp_path, transport=send, include_failure_details=True)
    assert result == {"acknowledged": 0, "pending": 1, "last_failure_category": category, "next_retry_at_epoch": None}
    assert "private" not in json.dumps(result) and "do-not-expose" not in json.dumps(result)
    assert coordinator.pending_publications(queue_root=tmp_path)


def test_changed_ledger_ack_keeps_publication_pending_and_reports_safe_ack_failure(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from tests.test_beta_policy_screening_coordinator import plan
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    monkeypatch.setattr(coordinator, "acknowledge_publication", lambda *a, **k: False)
    assert worker.deliver_publications(queue_root=tmp_path, transport=lambda _: None, include_failure_details=True) == {
        "acknowledged": 0, "pending": 1, "last_failure_category": "ack", "next_retry_at_epoch": None}
    assert len(coordinator.pending_publications(queue_root=tmp_path)) == 1


@pytest.mark.parametrize("execution_error", [False, True])
def test_worker_writes_existing_status_before_execution_and_in_finally_without_blocking_work(
        tmp_path, monkeypatch, capsys, execution_error):
    import sys
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    calls = []
    def snapshot(root, **kwargs):
        assert root == tmp_path
        calls.append(("snapshot", kwargs))
        raise OSError("status filesystem unavailable")
    monkeypatch.setitem(sys.modules, "blueprint_pipeline.beta_policy_screening_status",
        SimpleNamespace(write_worker_status=snapshot))
    def delivery(**kwargs):
        calls.append(("publication", kwargs))
        return {"acknowledged": 0, "pending": 0, "last_failure_category": None, "next_retry_at_epoch": None}
    def execute(**kwargs):
        calls.append(("execution", kwargs))
        if execution_error:
            raise RuntimeError("canonical failure must still propagate")
        return {"status": "processed"}
    monkeypatch.setattr(worker, "deliver_publications", delivery)
    monkeypatch.setattr(coordinator, "process_screening_queue", execute)
    if execution_error:
        with pytest.raises(RuntimeError, match="canonical failure"):
            worker.main(["--queue-root", str(tmp_path), "--packet-queue-root", ""])
        assert [kind for kind, _ in calls] == ["publication", "snapshot", "execution", "snapshot"]
        assert capsys.readouterr().out == ""
    else:
        assert worker.main(["--queue-root", str(tmp_path), "--packet-queue-root", ""]) == 0
        assert [kind for kind, _ in calls] == ["publication", "snapshot", "execution", "publication", "snapshot"]
        result = json.loads(capsys.readouterr().out)
        assert result["worker_status"]["status"] == "unknown"
        assert result["worker_status"]["reason"] == "screening_status_unavailable"
        assert result["execution"]["status"] == "processed"
    assert calls[0][1]["include_failure_details"] is True


def test_restart_revalidates_accepted_source_and_refusal_blocks_new_spend(tmp_path):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from tests.test_beta_policy_screening_coordinator import Backend, plan
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    worker.deliver_publications(queue_root=tmp_path, transport=lambda _: None)
    assert coordinator.pending_publications(queue_root=tmp_path) == []
    def source_expired(_):
        raise ValueError("beta_screening_publication_http_409")
    assert worker.deliver_publications(queue_root=tmp_path, transport=source_expired,
        refresh_startable=True) == {"acknowledged": 0, "pending": 1}
    backend = Backend()
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert backend.preparations == backend.launches == 0
    assert coordinator.pending_publications(queue_root=tmp_path)[0]["budget"]["committed_upper_bound_usd"] == "0"
    worker.deliver_publications(queue_root=tmp_path, transport=lambda _: None, refresh_startable=True)
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert backend.preparations == backend.launches == 1


def prepared_source_queue(tmp_path, monkeypatch, *, revisions=1):
    """Real sealed completion requests and planner; stub only host scene/runtime reads."""
    from blueprint_pipeline import beta_policy_screening_capabilities as capabilities
    from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as completed_scene
    from blueprint_pipeline import team_candidate_quick10_registry as registry
    from blueprint_pipeline import team_scene_packet_worker as packets
    from tests.test_beta_policy_screening_plan import inputs, scene

    source, context = source_world(tmp_path / "partition", monkeypatch)
    capture = tmp_path / "partition/scenes/site-job1/captures/walkthrough-job1"
    queue = tmp_path / "packets"
    results = {}
    requests = []
    for index in range(revisions):
        revision = canonical_digest({"configured_scene_revision": index})
        receipt = {"receipt_digest": canonical_digest({"launch_receipt": index})}
        launch_root = tmp_path / "launches" / str(index)
        result = {"configured_scene_revision_digest": revision}
        intent = {"intent_id": f"scene-intent-{index}", "intent_digest": canonical_digest({"intent": index}),
                  "request": {"submission_id": source["capture_id"], "task": {"task_id": source["task_id"]}}}
        queued = packets.enqueue_team_scene_request(intent=intent, run_root=launch_root, receipt=receipt,
            result=result, readiness={"status": "passed", "retained": True}, queue_root=queue)
        assert queued["status"] == "enqueued"
        key = queued["request_key"]
        (queue / "pending" / f"{key}.json").rename(queue / "processed" / f"{key}.json")
        (queue / "status" / f"{key}.json").write_text(json.dumps({"status": "completed"}))
        results[launch_root] = (result, receipt, {"provider_zero_confirmed": True})
        requests.append(json.loads((queue / "processed" / f"{key}.json").read_text()))
    validated = []
    def validate(path):
        validated.append(path)
        return deepcopy(results[path])
    monkeypatch.setattr(completed_scene, "_validate_source", validate)
    profile = {"scene_id": context["scene_id"], "capture_root": str(capture)}
    monkeypatch.setattr(registry, "profile_for_capture", lambda path: profile if path == capture else None)
    planning_calls = []
    def planning_inputs(**kwargs):
        planning_calls.append(deepcopy(kwargs))
        catalog, admitted, quotes = inputs()
        assert kwargs["catalog"] == catalog
        assert kwargs["profile"] == profile and kwargs["capture_root"] == capture
        facts = {**scene(), **kwargs["source"], "site_id": context["scene_id"], "scene_id": context["scene_id"]}
        for quote in quotes.values():
            quote["scene_revision_digest"] = kwargs["source"]["scene_revision_digest"]
            quote["valid_until"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            quote["quote_digest"] = canonical_digest(quote, digest_field="quote_digest")
        return {"scene": facts, "capabilities": admitted, "cost_quotes": quotes}
    monkeypatch.setattr(capabilities, "build_planning_inputs", planning_inputs)
    return {"packet_queue": queue, "screening_queue": tmp_path / "screening", "requests": requests,
            "results": results, "validated": validated, "planning_calls": planning_calls, "context": context}


def test_completed_scene_is_planned_and_durably_queued_once_across_timer_ticks(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from blueprint_pipeline.beta_policy_screening_plan import validate_screening_plan

    world = prepared_source_queue(tmp_path, monkeypatch)
    kwargs = {"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"]}
    [adopted] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert adopted["status"] == "enqueued"
    plans = list((world["screening_queue"] / "plans").glob("*.json"))
    assert len(plans) == 1
    plan_bytes = plans[0].read_bytes()
    plan = validate_screening_plan(json.loads(plan_bytes))
    request = world["requests"][0]
    assert plan["source"] == {"scene_intent_id": request["intent_id"], "scene_intent_digest": request["intent_digest"],
        "capture_id": request["capture_id"], "task_id": request["task_id"],
        "task_context_digest": world["context"]["context_digest"], "scene_revision_digest": request["scene_revision_digest"]}
    assert plan["site_id"] == "site-job1" and plan["campaign_id"] == adopted["campaign_id"]
    assert len([row for row in plan["candidates"] if row["status"] == "selected"]) == 2
    [publication] = coordinator.pending_publications(queue_root=world["screening_queue"])
    assert publication["source"] == plan["source"] and publication["status"] == "queued"
    assert publication["budget"]["committed_upper_bound_usd"] == "0"
    ledger_bytes = (world["screening_queue"] / coordinator.LEDGER_FILENAME).read_bytes()
    for _ in range(3):
        [replay] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
        assert replay["status"] == "already_enqueued" and replay["campaign_id"] == adopted["campaign_id"]
    assert len(world["planning_calls"]) == len(world["validated"]) == 1
    assert list((world["screening_queue"] / "plans").glob("*.json")) == plans
    assert plans[0].read_bytes() == plan_bytes
    assert (world["screening_queue"] / coordinator.LEDGER_FILENAME).read_bytes() == ledger_bytes
    assert coordinator.pending_publications(queue_root=world["screening_queue"]) == [publication]
    assert not list((world["screening_queue"] / "attempts").iterdir())


def test_blocked_completed_source_does_not_starve_next_source_and_remains_retryable(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from blueprint_pipeline.team_scene_packet_worker import request_key

    world = prepared_source_queue(tmp_path, monkeypatch, revisions=2)
    first, second = sorted(world["requests"], key=request_key)
    first_launch = next(path for path in world["results"] if str(path) == first["source_launch_root"])
    good_receipt = world["results"][first_launch][1]["receipt_digest"]
    world["results"][first_launch][1]["receipt_digest"] = "sha256:" + "f" * 64
    kwargs = {"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"], "max_sources": 1}
    [blocked] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert blocked == {"request_key": request_key(first), "status": "blocked",
                       "blockers": ["beta_screening_completed_scene_binding_changed"]}
    assert not world["planning_calls"]
    [adopted] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert adopted["request_key"] == request_key(second) and adopted["status"] == "enqueued"
    assert len(coordinator.pending_publications(queue_root=world["screening_queue"])) == 1
    world["results"][first_launch][1]["receipt_digest"] = good_receipt
    [recovered] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert recovered["request_key"] == request_key(first) and recovered["status"] == "enqueued"
    assert len(coordinator.pending_publications(queue_root=world["screening_queue"])) == 2
    assert len(world["planning_calls"]) == 2


def test_existing_prepared_scene_queues_its_missing_probe_before_screening(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_capabilities as capabilities
    from blueprint_pipeline import rigid_robot_free_probe_runner as probes

    world = prepared_source_queue(tmp_path, monkeypatch)
    queued = []
    monkeypatch.setattr(probes, "enqueue_rigid_robot_free_probe",
                        lambda capture: queued.append(capture) or {"status": "queued"})

    def waiting(**_kwargs):
        raise ValueError("team_evaluation_robot_free_probe_receipt_missing")

    monkeypatch.setattr(capabilities, "build_planning_inputs", waiting)
    [outcome] = worker.enqueue_prepared_scenes(queue_root=world["screening_queue"],
        packet_queue_root=world["packet_queue"])["sources"]
    assert outcome["status"] == "blocked"
    assert outcome["robot_free_probe_status"] == "queued"
    assert outcome["blockers"] == ["team_evaluation_robot_free_probe_receipt_missing"]
    assert len(queued) == 1 and queued[0].name == "walkthrough-job1"
    assert not list((world["screening_queue"] / "plans").iterdir())
    assert not list((world["screening_queue"] / "attempts").iterdir())


@pytest.mark.parametrize("failure", ["packet_not_completed", "current_consent_revoked"])
def test_automatic_adoption_rejects_unready_packet_or_current_withdrawal_before_planning(tmp_path, monkeypatch, failure):
    from blueprint_pipeline.team_scene_packet_worker import request_key

    world = prepared_source_queue(tmp_path, monkeypatch)
    key = request_key(world["requests"][0])
    if failure == "packet_not_completed":
        (world["packet_queue"] / "status" / f"{key}.json").write_text(json.dumps({"status": "blocked"}))
        expected = "beta_screening_scene_packet_not_ready"
    else:
        world["context"]["capture_rights"]["consent_revoked"] = True
        expected = "beta_screening_current_capture_consent_required"
    [outcome] = worker.enqueue_prepared_scenes(queue_root=world["screening_queue"],
        packet_queue_root=world["packet_queue"])["sources"]
    assert outcome["status"] == "blocked" and outcome["blockers"] == [expected]
    assert not world["planning_calls"] and not world["validated"]
    assert not list((world["screening_queue"] / "plans").iterdir())


def test_fresh_deploy_provisions_screening_sandbox_before_timer_can_start(tmp_path):
    import os
    from pathlib import Path
    from tests.test_deploy_control_plane_commit import deploy

    host = tmp_path / "host"
    (host / "var/lib/blueprint").mkdir(parents=True)
    units = ("blueprint-beta-policy-screening.service", "blueprint-beta-policy-screening.timer")
    assert all(unit in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS for unit in units)
    assert units[1] in deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS
    receipt = deploy._install_unit_sandbox_paths(release_path=Path(__file__).resolve().parents[1],
        units=units, root_prefix=host, owner_ids=(os.getuid(), os.getgid()))
    target = host / "var/lib/blueprint/pipeline-control-plane/beta-policy-screening"
    assert target.is_dir() and target.stat().st_mode & 0o777 == 0o750
    assert receipt["created"] == [{"unit": units[0],
        "path": "/var/lib/blueprint/pipeline-control-plane/beta-policy-screening", "mode": "0750",
        "owner_uid": os.getuid(), "owner_gid": os.getgid()}]
    repeated = deploy._install_unit_sandbox_paths(release_path=Path(__file__).resolve().parents[1],
        units=units, root_prefix=host, owner_ids=(os.getuid(), os.getgid()))
    assert repeated["created"] == []


@pytest.mark.parametrize("before_ack,after_ack,expected", [(1, 0, 100), (0, 1, 200)])
def test_worker_status_publication_time_does_not_refresh_an_earlier_ack(
        tmp_path, monkeypatch, capsys, before_ack, after_ack, expected):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from blueprint_pipeline import beta_policy_screening_status as status

    ticks = iter((100, 200))
    monkeypatch.setattr(status, "time", SimpleNamespace(time=lambda: next(ticks)))
    deliveries = iter((before_ack, after_ack))
    monkeypatch.setattr(worker, "deliver_publications", lambda **_kwargs: {
        "acknowledged": next(deliveries), "pending": 0,
        "last_failure_category": None, "next_retry_at_epoch": None})
    monkeypatch.setattr(coordinator, "process_screening_queue", lambda **_kwargs: {"status": "processed"})
    assert worker.main(["--queue-root", str(tmp_path), "--packet-queue-root", ""]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["publication_before"]["acknowledged"] == before_ack
    assert result["publication_after"]["acknowledged"] == after_ack
    assert result["worker_status"]["last_successful_publication_at_epoch"] == expected
    retained = json.loads((tmp_path / "worker_status.json").read_text())
    assert retained["last_successful_publication_at_epoch"] == expected


def test_unprintable_publication_error_stays_unknown_and_pending(tmp_path):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from tests.test_beta_policy_screening_coordinator import plan

    class UnprintableError(RuntimeError):
        def __str__(self):
            raise ValueError("private token=do-not-expose")

    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    def unavailable(_publication):
        raise UnprintableError()
    result = worker.deliver_publications(queue_root=tmp_path, transport=unavailable, include_failure_details=True)
    assert result == {"acknowledged": 0, "pending": 1, "last_failure_category": "unknown", "next_retry_at_epoch": None}
    assert len(coordinator.pending_publications(queue_root=tmp_path)) == 1


def test_failed_initial_status_does_not_invent_a_later_publication_time(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from blueprint_pipeline import beta_policy_screening_status as status

    real_snapshot = status.write_worker_status
    calls = []
    def snapshot(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise OSError("initial status write unavailable")
        return real_snapshot(*args, **kwargs)
    monkeypatch.setattr(status, "write_worker_status", snapshot)
    deliveries = iter(((1, "ack"), (0, None)))
    def delivery(**_kwargs):
        count, category = next(deliveries)
        return {"acknowledged": count, "pending": int(category is not None),
            "last_failure_category": category, "next_retry_at_epoch": None}
    monkeypatch.setattr(worker, "deliver_publications", delivery)
    monkeypatch.setattr(coordinator, "process_screening_queue", lambda **_kwargs: {"status": "processed"})
    assert worker.main(["--queue-root", str(tmp_path), "--packet-queue-root", ""]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["publication_before"]["acknowledged"] == 1
    assert result["publication_before"]["last_failure_category"] == "ack"
    assert result["publication_before"]["next_retry_at_epoch"] is None
    assert result["worker_status"]["last_successful_publication_at_epoch"] is None
    assert result["worker_status"]["last_publication_failure"] is None


@pytest.mark.parametrize("after_failure,expected_category,expected_time", [(None, "auth", 100), ("ack", "ack", 200)])
def test_worker_status_retains_original_failure_time_until_a_new_attempt_fails(
        tmp_path, monkeypatch, capsys, after_failure, expected_category, expected_time):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from blueprint_pipeline import beta_policy_screening_status as status

    ticks = iter((100, 200))
    monkeypatch.setattr(status, "time", SimpleNamespace(time=lambda: next(ticks)))
    categories = iter(("auth", after_failure))
    def delivery(**_kwargs):
        category = next(categories)
        return {"acknowledged": 0, "pending": int(category is not None),
            "last_failure_category": category, "next_retry_at_epoch": None}
    monkeypatch.setattr(worker, "deliver_publications", delivery)
    monkeypatch.setattr(coordinator, "process_screening_queue", lambda **_kwargs: {"status": "processed"})
    assert worker.main(["--queue-root", str(tmp_path), "--packet-queue-root", ""]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["publication_before"]["last_failure_category"] == "auth"
    failure = result["worker_status"]["last_publication_failure"]
    assert failure["category"] == expected_category
    assert failure["observed_at_epoch"] == expected_time
    assert failure["next_retry_at_epoch"] is None
