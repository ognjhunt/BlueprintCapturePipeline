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


def test_worker_retains_operator_status_for_more_than_ten_candidates(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from blueprint_pipeline.team_scene_packet_worker import QUEUE_ROOT_ENV as PACKET_QUEUE_ENV
    from tests.test_beta_policy_screening_coordinator import plan

    coordinator.enqueue_screening_campaign(plan(count=12, cost="1.00"), queue_root=tmp_path)
    monkeypatch.delenv(PACKET_QUEUE_ENV, raising=False)
    monkeypatch.setattr(worker, "publish_screening", lambda publication: None)
    assert worker.main(["--queue-root", str(tmp_path), "--max-launches", "0"]) == 0
    status = json.loads((tmp_path / "worker_status.json").read_text())
    assert status["candidate_counts"]["pending"] == 12
    assert status["campaign_count"] == 1
    assert status["performs_live_actions"] is False
    result = json.loads(capsys.readouterr().out)
    assert result["worker_status"] == status
    assert result["execution"]["launches_started"] == 0


def test_failed_delivery_retains_durable_outbox_and_retry_acknowledges(tmp_path):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from tests.test_beta_policy_screening_coordinator import plan
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    def unavailable(_):
        raise OSError("temporary network failure")
    assert worker.deliver_publications(queue_root=tmp_path, transport=unavailable) == {
        "acknowledged": 0, "pending": 1, "last_failure_category": "unavailable",
        "last_error": {"type": "builtins.OSError", "message": "temporary network failure"}}
    pending = coordinator.pending_publications(queue_root=tmp_path)
    received = []
    assert worker.deliver_publications(queue_root=tmp_path, transport=received.append) == {"acknowledged": 1, "pending": 0}
    assert received == pending
    assert coordinator.pending_publications(queue_root=tmp_path) == []


def test_restart_revalidates_accepted_source_and_refusal_blocks_new_spend(tmp_path):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from tests.test_beta_policy_screening_coordinator import Backend, plan
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    worker.deliver_publications(queue_root=tmp_path, transport=lambda _: None)
    assert coordinator.pending_publications(queue_root=tmp_path) == []
    def source_expired(_):
        raise ValueError("beta_screening_publication_http_409")
    assert worker.deliver_publications(queue_root=tmp_path, transport=source_expired,
        refresh_startable=True) == {"acknowledged": 0, "pending": 1, "last_failure_category": "source",
        "last_error": {"type": "builtins.ValueError", "message": "beta_screening_publication_http_409"}}
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
    from tests.test_beta_policy_screening_plan import admitted as admit_candidate, inputs, scene

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
    # The deployed commit planning binds and preparation requires.
    deployed = {"commit": "a" * 40}
    monkeypatch.setattr(worker, "implementation_commit", lambda: deployed["commit"])
    def planning_inputs(**kwargs):
        planning_calls.append(deepcopy(kwargs))
        catalog, admitted, quotes = inputs()
        # This worker fixture supplies every enabled runtime, including Cosmos;
        # the shared pure-planner fixture deliberately supplies only two.
        for candidate in catalog["candidates"]:
            candidate_id = candidate["candidate_id"]
            if candidate["eligibility"]["enabled"] and candidate_id not in admitted["candidate_bindings"]:
                admitted["candidate_bindings"][candidate_id], quotes[candidate_id] = admit_candidate(candidate)
        for binding in admitted["candidate_bindings"].values():
            binding["execution_spec"]["implementation_commit"] = deployed["commit"]
            binding["admission_digest"] = canonical_digest(binding["execution_spec"])
        assert kwargs["catalog"] == catalog
        assert kwargs["profile"] == profile and kwargs["capture_root"] == capture
        facts = {**scene(), **kwargs["source"], "site_id": context["scene_id"], "scene_id": context["scene_id"]}
        for quote in quotes.values():
            quote["scene_revision_digest"] = kwargs["source"]["scene_revision_digest"]
            quote["valid_until"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            quote["quote_digest"] = canonical_digest(quote, digest_field="quote_digest")
        return {"scene": facts, "capabilities": admitted, "cost_quotes": quotes}
    monkeypatch.setattr(capabilities, "build_planning_inputs", planning_inputs)
    # The job's one dollar ceiling leaves more than the $15 site allowance.
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling
    from tests.test_beta_policy_screening_job_ceiling import ceiling_summary
    monkeypatch.setattr(job_ceiling, "read_job_ceiling_summary", lambda **_: ceiling_summary(40))
    return {"packet_queue": queue, "screening_queue": tmp_path / "screening", "requests": requests,
            "results": results, "validated": validated, "planning_calls": planning_calls, "context": context,
            "deployed": deployed}


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
    selected = [row for row in plan["candidates"] if row["status"] == "selected"]
    selected_ids = {row["candidate_id"] for row in selected}
    # Candidates are admitted in sorted order until the $15 site allowance runs
    # out, so the third pinned OpenPI PolaRiS variant is deferred for budget.
    assert selected_ids == {"pi05_droid", "groot_n17_droid", "cosmos3_edge_policy_droid",
                            "cosmos3_nano_policy_droid", "molmoact2_droid",
                            "pi0_droid_jointpos_100k_polaris", "pi0_droid_jointpos_polaris"}
    assert plan["maximum_planned_cost_usd"] == "14.00" and plan["budget_usd"] == "15.00"
    deferred = next(row for row in plan["candidates"] if row["candidate_id"] == "pi0_fast_droid_jointpos_polaris")
    assert deferred["status"] == "waitlisted" and deferred["blockers"] == ["site_screening_budget_insufficient"]
    assert all(row["max_cost_usd"] == row["cost_quote"]["maximum_cost_usd"] == "2.00" for row in selected)
    assert all(row["blockers"] for row in plan["candidates"] if row["status"] != "selected")
    [publication] = coordinator.pending_publications(queue_root=world["screening_queue"])
    assert publication["source"] == plan["source"] and publication["status"] == "queued"
    assert publication["budget"]["committed_upper_bound_usd"] == "0"
    published = {row["candidate_id"]: row for row in publication["candidates"]}
    assert set(published) == {row["candidate_id"] for row in plan["candidates"]}
    for row in plan["candidates"]:
        actual = published[row["candidate_id"]]
        assert actual["status"] == ("pending" if row["candidate_id"] in selected_ids else row["status"])
        if row["candidate_id"] in selected_ids:
            assert actual["blockers"] == []
        else:
            # Public delivery retains typed reasons while omitting private
            # details such as the name of an unavailable credential.
            assert actual["blockers"] and set(actual["blockers"]) <= set(row["blockers"])
        assert actual["episodes_scored"] == 0 and actual["success_rate"] is None
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
    # A typed refusal is its own cause.
    assert "error" not in outcome


def test_generic_source_refusal_keeps_its_bounded_cause(tmp_path, monkeypatch):
    from blueprint_pipeline import rigid_robot_free_probe_runner as probe

    world = prepared_source_queue(tmp_path, monkeypatch)

    def full(_capture):
        # A full work volume: the generic code alone cannot say so.
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(probe, "enqueue_rigid_robot_free_probe", full)
    [outcome] = worker.enqueue_prepared_scenes(queue_root=world["screening_queue"],
        packet_queue_root=world["packet_queue"])["sources"]
    assert outcome["status"] == "blocked" and outcome["blockers"] == ["beta_screening_source_unavailable"]
    assert outcome["error"] == {"type": "builtins.OSError", "message": "[Errno 28] No space left on device"}
    [state] = (world["screening_queue"] / "sources").iterdir()
    assert json.loads(state.read_text())["error"] == outcome["error"]


@pytest.mark.parametrize("raised,category", [
    (ValueError("beta_screening_publication_http_429"), "unavailable"),
    (ValueError("beta_screening_publication_http_503"), "unavailable"),
    (ValueError("beta_screening_publication_http_401"), "auth"),
    (ValueError("beta_screening_publication_ack_invalid"), "ack"),
    (TimeoutError("timed out"), "unavailable"),
    (RuntimeError("unexpected https://h/p?sig=abc"), "unknown"),
])
def test_failed_delivery_names_its_category_for_the_worker_status(tmp_path, raised, category):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from blueprint_pipeline.beta_policy_screening_status import build_worker_status
    from tests.test_beta_policy_screening_coordinator import plan
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)

    def refused(_):
        raise raised
    delivered = worker.deliver_publications(queue_root=tmp_path, transport=refused)
    assert delivered["pending"] == 1 and delivered["last_failure_category"] == category
    assert delivered["last_error"]["type"] == f"builtins.{type(raised).__name__}"
    assert "sig=abc" not in json.dumps(delivered)
    status = build_worker_status(tmp_path, publication_after=delivered)
    assert status["last_publication_failure"]["category"] == category


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


def test_the_timer_pass_returns_a_transient_scene_half_block_to_its_queue_and_names_a_refusal(
        tmp_path, monkeypatch):
    from blueprint_pipeline.team_scene_packet_worker import request_key

    world = prepared_source_queue(tmp_path, monkeypatch, revisions=2)
    transient, refused = sorted(world["requests"], key=request_key)
    packets = world["packet_queue"]
    (packets / "status" / f"{request_key(transient)}.json").write_text(json.dumps(
        {"status": "blocked", "blockers": ["team_scene_offer_publication_failed"]}))
    (packets / "status" / f"{request_key(refused)}.json").write_text(json.dumps(
        {"status": "blocked", "blockers": ["team_scene_capture_withdrawn"]}))
    adopted = worker.enqueue_prepared_scenes(queue_root=world["screening_queue"], packet_queue_root=packets)
    # Recorded before automatic retries existed, the WebApp outage is due at once; the path unit runs it again.
    assert adopted["team_scene_packet_requeue"]["requeued"] == [
        {"request_key": request_key(transient), "attempt": 1, "reason": "team_scene_offer_publication_failed"}]
    assert (packets / "pending" / f"{request_key(transient)}.json").is_file()
    # A deterministic refusal stays blocked, and the screening outcome says which.
    [outcome] = adopted["sources"]
    assert outcome["request_key"] == request_key(refused)
    assert outcome["blockers"] == ["beta_screening_scene_packet_not_ready"]
    assert outcome["scene_packet"] == {"status": "blocked", "blockers": ["team_scene_capture_withdrawn"]}
    assert not world["planning_calls"]


def test_deploy_before_preparation_replans_unprepared_candidates_once(tmp_path, monkeypatch):
    """A new commit while nothing is reserved re-plans; spent candidates keep their result."""
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator

    world = prepared_source_queue(tmp_path, monkeypatch)
    root = world["screening_queue"]
    kwargs = {"queue_root": root, "packet_queue_root": world["packet_queue"]}
    [adopted] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    first = adopted["campaign_id"]
    plan = json.loads((root / "plans" / f"{first}.json").read_text())
    selected = [row["candidate_id"] for row in plan["candidates"] if row["status"] == "selected"]

    class FirstCompletes:
        def prepare_candidate(self, **_):
            return {"sealed": "local"}

        def launch_candidate(self, **_):
            return {"status": "blocked", "provider_allocation_performed": True, "provider_zero_verified": True,
                    "blockers": ["beta_screening_native_result_missing"]}

    # One funded candidate allocates (its money is committed), then the code is deployed again.
    coordinator.process_screening_queue(queue_root=root, executor=FirstCompletes(), require_publication_ack=False)
    world["deployed"]["commit"] = "b" * 40
    [replanned] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    second = replanned["campaign_id"]
    assert replanned["status"] == "enqueued" and replanned["superseded_campaign_id"] == first and second != first
    ledger = coordinator._ledger(root)
    old = ledger["campaigns"][first]
    assert old["superseded_by"] == second and ledger["campaigns"][second]["supersedes"] == first
    assert old["candidates"][selected[0]]["status"] == "blocked"
    assert {row["status"] for key, row in old["candidates"].items() if key in selected[1:]} == {"blocked"}
    assert all(old["candidates"][key]["blockers"] == ["beta_screening_replanned_for_new_implementation"]
               for key in selected[1:])
    successor = json.loads((root / "plans" / f"{second}.json").read_text())
    ids = {row["candidate_id"] for row in successor["candidates"]}
    # The spent candidate is never planned again; the rest are, at the new commit.
    assert selected[0] not in ids and set(selected[1:]) <= ids
    assert {row["execution_spec"]["implementation_commit"] for row in successor["candidates"]
            if row["status"] in {"selected", "waitlisted"}} == {"b" * 40}
    # The site budget is shared: the committed money still counts.
    assert ledger["campaigns"][second]["site_budget"]["spent_upper_bound_usd"] == "2"
    for _ in range(2):
        [replay] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
        assert replay["status"] == "already_enqueued" and replay["campaign_id"] == second
    assert len(list((root / "plans").glob("*.json"))) == 2


def test_deploy_never_replans_while_a_reservation_is_held(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator

    world = prepared_source_queue(tmp_path, monkeypatch)
    root = world["screening_queue"]
    kwargs = {"queue_root": root, "packet_queue_root": world["packet_queue"]}
    [adopted] = worker.enqueue_prepared_scenes(**kwargs)["sources"]

    class Uncertain:
        def prepare_candidate(self, **_):
            return {"sealed": "local"}

        def launch_candidate(self, **_):
            raise RuntimeError("provider response lost")

    coordinator.process_screening_queue(queue_root=root, executor=Uncertain(), require_publication_ack=False)
    world["deployed"]["commit"] = "b" * 40
    [replay] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert replay["status"] == "already_enqueued" and replay["campaign_id"] == adopted["campaign_id"]
    assert len(list((root / "plans").glob("*.json"))) == 1
