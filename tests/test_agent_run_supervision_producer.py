"""Accepted-run producers own revision, failure, budget and cleanup progression."""

import hashlib
import json
from pathlib import Path
import time

import httpx
import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from blueprint_pipeline.agent_execution.production import ProductionAgentService
from blueprint_pipeline.agent_execution.supervision import progress_plan
from blueprint_pipeline.agent_execution.supervision_producer import (
    register_run_supervision,
    reserve_automatic_revision,
)
from blueprint_pipeline.agent_execution.failure_events import (
    register_preparation_failure_subscription,
    discover_retained_failures,
)
from blueprint_pipeline.agent_execution.controller_recovery import ControllerRecoveryBinding
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from tests.test_agent_production_service import fixture, write
from tests.test_agent_execution_sdk import response
from tests.test_task_evaluation_stage_replay import _queue


def setup(tmp_path):
    service, _, original, config_path = fixture(tmp_path)
    value = json.loads(original.read_text())
    value["autostart"] = False
    write(original, value)
    config = service.config.model_dump(mode="json")
    config.update(automatic_run_supervision=True, automatic_failure_investigation=True)
    write(config_path, config)
    service = ProductionAgentService(config_path, source_commit="a" * 40)
    intent = {
        "schema_version": "task_evaluation_scene_intent.v1",
        "intent_id": "scene-watch-fixture",
        "accepted_at_epoch": time.time(),
        "request": {
            "consent": {"spend_authorized": True, "task_confirmed": True},
            "execution": {
                "expires_at_epoch": time.time() + 1800,
                "max_total_spend_usd": 30.0,
                "max_retries": 1,
            },
        },
    }
    intent["intent_digest"] = cross_runtime_canonical_digest(intent)
    directory = tmp_path / "intents" / intent["intent_id"]
    write(directory / "intent.json", intent)
    progress = {
        "schema_version": "task_evaluation_scene_progression.v1",
        "intent_id": intent["intent_id"],
        "status": "needs_input",
        "phase": "intake",
        "state": {},
        "blockers": ["missing_fixture_measurement"],
    }
    write(directory / "progression.json", progress)
    plan = register_run_supervision(
        intent=intent, directory=directory, source_commit="a" * 40, service=service
    )
    return service, plan, intent, directory


def settle_sdk(service, record, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "1")
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            status = record.context["fresh_scene_preparation_status"]["status_digest"]
            return response(
                [
                    {
                        "type": "function_call",
                        "id": "fc1",
                        "call_id": "call1",
                        "name": "inspect_fresh_scene_preparation",
                        "arguments": json.dumps(
                            {
                                "context_revision": record.task.context_revision,
                                "arguments": {"status_digest": status},
                            }
                        ),
                        "status": "completed",
                    }
                ]
            )
        output = {
            "disposition": "no_action",
            "summary": "Controller revision inspected; no independent completion claim.",
            "evidence_references": [],
            "next_actions": [],
            "uncertainty": [],
        }
        return response(
            [
                {
                    "type": "message",
                    "id": "msg1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": json.dumps(output), "annotations": []}
                    ],
                }
            ],
            number=2,
        )

    runtime = service.runtime_for_task(record.task)
    runtime._hermetic_transport = httpx.MockTransport(handler)
    assert runtime.step(record.task.task_id)["state"] == "completed"
    assert len(calls) == 2
    return runtime


def test_accepted_run_registers_once_revisits_and_cleans_terminal_sdk(tmp_path, monkeypatch):
    service, plan, intent, directory = setup(tmp_path)
    assert (
        register_run_supervision(
            intent=intent, directory=directory, source_commit="a" * 40, service=service
        )
        == plan
    )
    from blueprint_pipeline.agent_execution.webapp_delivery import admission_payload
    assert admission_payload(service.record(plan.template_task_id)) is None
    state = progress_plan(service, plan)
    record = service.record(state["active_task_id"])
    assert record.autostart is False
    assert admission_payload(record)["autostart"] is True
    assert (
        record.task.run_id == intent["intent_id"] and record.supervision.watch_id == plan.watch_id
    )
    runtime = settle_sdk(service, record, monkeypatch)
    assert progress_plan(service, plan)["status"] == "waiting_for_new_evidence"
    progress = json.loads((directory / "progression.json").read_text())
    progress["updated_at_epoch"] = time.time()
    write(directory / "progression.json", progress)
    assert progress_plan(service, plan)["status"] == "waiting_for_new_evidence"
    progress.update(status="completed", phase="delivery", blockers=[])
    write(directory / "progression.json", progress)
    assert progress_plan(service, plan)["status"] == "settling_prior_scope"
    runtime.cleanup(record.task.task_id)
    next_state = progress_plan(service, plan)
    assert next_state["active_task_id"] != record.task.task_id
    final = service.record(next_state["active_task_id"])
    final_runtime = settle_sdk(service, final, monkeypatch)
    assert progress_plan(service, plan)["status"] == "final_cleanup_pending"
    final_runtime.cleanup(final.task.task_id)
    assert progress_plan(service, plan)["status"] == "completed"
    assert service.journal.task(record.task.task_id)["cleanup_state"] == "deleted"


def test_failed_child_and_recovery_share_the_run_owner(tmp_path):
    service, plan, intent, directory = setup(tmp_path)
    queue, _, job = _queue(tmp_path / "saved")
    link = {
        "schema_version": "task_evaluation_scene_preparation_link.v1",
        "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"],
        "preparation_id": job["parent_preparation_id"],
        "request_digest": job["parent_request_digest"],
    }
    link["link_digest"] = canonical_digest(link)
    link_path = directory / "preparation-link.json"
    write(link_path, link)
    controller = {
        "schema_version": "task_evaluation_scene_progression_config.v1",
        "intent_root": str(directory.parent),
    }
    controller["config_digest"] = canonical_digest(controller)
    controller_path = tmp_path / "controller.json"
    write(controller_path, controller)
    binding = ControllerRecoveryBinding(
        recovery_id="resume_scene",
        intent_id=intent["intent_id"],
        intent_digest=intent["intent_digest"],
        controller_config_path=str(controller_path),
        controller_config_sha256="sha256:"
        + hashlib.sha256(controller_path.read_bytes()).hexdigest(),
        required_replay_id="failed_boundary",
        parent_request_digest=job["parent_request_digest"],
        preparation_link_path=str(link_path),
        preparation_link_sha256="sha256:" + hashlib.sha256(link_path.read_bytes()).hexdigest(),
    )
    config = service.config.model_dump(mode="json")
    config["automatic_recovery_bindings"] = [binding.model_dump(mode="json")]
    write(service.config_path, config)
    service = ProductionAgentService(service.config_path, source_commit="a" * 40)
    register_preparation_failure_subscription(
        preparation_link=link,
        controller_config={
            "child_queue_root": str(queue),
            "preparation_queue_root": str(tmp_path / "parents"),
        },
        agent_config_path=service.config_path,
    )
    progress = json.loads((directory / "progression.json").read_text())
    progress.update(
        status="blocked",
        phase="preparation",
        state={
            "preparation_link": {"path": str(link_path), "sha256": binding.preparation_link_sha256}
        },
    )
    write(directory / "progression.json", progress)
    state = progress_plan(service, plan)
    record = service.record(state["active_task_id"])
    assert record.stage_replays[0].child_id == job["child_id"] and record.controller_recoveries == (
        binding,
    )
    assert record.task.run_id == plan.run_id
    assert record.supervision.watch_id == plan.watch_id
    assert discover_retained_failures(service)[0]["state"] == "owned_by_persistent_supervisor"
    service.validate_admission(record.task)
    config["automatic_recovery_bindings"] = []
    write(service.config_path, config)
    restarted = ProductionAgentService(service.config_path, source_commit="a" * 40)
    with pytest.raises(AgentExecutionError, match="automatic_supervision_scope_revoked"):
        restarted.validate_admission(record.task)


def test_lifetime_budget_does_not_reset_for_a_new_release(tmp_path):
    service, plan, _, _ = setup(tmp_path)
    assert reserve_automatic_revision(service, plan, "first", 1)
    assert reserve_automatic_revision(service, plan, "first", 1)
    next_plan = plan.model_copy(update={"source_commit": "b" * 40, "watch_id": "next-release"})
    assert reserve_automatic_revision(service, next_plan, "second", 1)
    assert reserve_automatic_revision(service, next_plan, "third", 1)
    assert not reserve_automatic_revision(service, next_plan, "fourth", 1)


def amend_allowance(service, intent, **updates):
    allowance = {"intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
        "maximum_revisions": 6, "maximum_reserved_inference_usd": 6,
        "expires_at": intent["request"]["execution"]["expires_at_epoch"],
        "authorization_reference": "owner-approved-three-dollar-reallocation", **updates}
    config = service.config.model_dump(mode="json")
    config["automatic_supervision_allowances"] = [allowance]
    write(service.config_path, config)
    return ProductionAgentService(service.config_path, source_commit=service.config.source_commit)


def test_explicit_allowance_preserves_prior_holds_and_one_owner_on_same_release(tmp_path):
    from blueprint_pipeline.agent_execution.supervision import ownership_event
    service, old_plan, intent, directory = setup(tmp_path)
    for task_id in ("first", "second", "third"):
        assert reserve_automatic_revision(service, old_plan, task_id, 1)
    with service.journal._connect() as connection:
        before = connection.execute("SELECT event_id, payload_json FROM events WHERE event_id LIKE 'automatic_supervision_budget_%'").fetchall()
    service = amend_allowance(service, intent)
    assert progress_plan(service, old_plan)["status"] == "revoked"
    plan = register_run_supervision(intent=intent, directory=directory, source_commit=service.config.source_commit, service=service)
    assert plan.watch_id != old_plan.watch_id
    state = progress_plan(service, plan)
    assert state["status"] == "executing_revision"
    assert ownership_event(service, plan.run_id)["watch_digest"] == plan.plan_digest
    assert service.journal.task(state["active_task_id"])["state"] == "queued"
    with service.journal._connect() as connection:
        for row in before:
            assert connection.execute("SELECT payload_json FROM events WHERE event_id=?", (row["event_id"],)).fetchone()[0] == row["payload_json"]
    next_plan = plan.model_copy(update={"source_commit": "b" * 40, "watch_id": "next-release"})
    assert reserve_automatic_revision(service, next_plan, "fifth", 1)
    assert reserve_automatic_revision(service, next_plan, "sixth", 1)
    assert not reserve_automatic_revision(service, next_plan, "seventh", 1)
    config = service.config.model_dump(mode="json")
    config["automatic_supervision_allowances"] = []
    write(service.config_path, config)
    revoked = ProductionAgentService(service.config_path, source_commit=service.config.source_commit)
    with pytest.raises(AgentExecutionError, match="allowance_revoked"):
        reserve_automatic_revision(revoked, next_plan, "seventh", 1)
    assert progress_plan(revoked, plan)["status"] == "revoked"
    assert revoked.journal.task(state["active_task_id"])["cancel_requested"]


@pytest.mark.parametrize("updates,code", [
    ({"intent_digest": "sha256:" + "f" * 64}, "allowance_scope_invalid"),
    ({"maximum_reserved_inference_usd": 31}, "allowance_exceeds_intent"),
    ({"expires_at": time.time() + 90000}, "allowance_exceeds_intent"),
])
def test_allowance_cannot_change_owner_or_exceed_accepted_envelope(tmp_path, updates, code):
    service, _, intent, directory = setup(tmp_path)
    service = amend_allowance(service, intent, **updates)
    with pytest.raises(AgentExecutionError, match=code):
        register_run_supervision(intent=intent, directory=directory, source_commit=service.config.source_commit, service=service)


def test_expired_allowance_and_unbound_plan_cannot_admit_inference(tmp_path):
    service, old_plan, intent, directory = setup(tmp_path)
    service = amend_allowance(service, intent, expires_at=time.time() - 1)
    assert register_run_supervision(intent=intent, directory=directory, source_commit=service.config.source_commit, service=service) is None
    with pytest.raises(AgentExecutionError, match="allowance_revoked"):
        reserve_automatic_revision(service, old_plan, "first", 1)


def test_revoked_automatic_configuration_cancels_the_existing_owner(tmp_path):
    service, plan, _, _ = setup(tmp_path)
    state = progress_plan(service, plan)
    config = service.config.model_dump(mode="json")
    config["automatic_run_supervision"] = False
    write(service.config_path, config)
    restarted = ProductionAgentService(service.config_path, source_commit="a" * 40)
    assert progress_plan(restarted, plan)["status"] == "revoked"
    assert restarted.journal.task(state["active_task_id"])["cancel_requested"]


@pytest.mark.parametrize("scope_change", [False, True])
def test_automatic_api_continuation_keeps_one_session_and_final_cleanup(tmp_path, scope_change):
    from tests.test_agent_execution_continuation import ContinuingAPI
    from blueprint_pipeline.agent_execution.contracts import digest

    service, plan, intent, directory = setup(tmp_path)
    guard = {
        "schema_version": "blueprint_agent_project_admission_observation.v1",
        "project_id": service.config.project_id,
        "credential_id": service.config.credential_id,
        "observed_at": time.time() - 1,
        "expires_at": time.time() + 1800,
        "dashboard_hard_limit_enabled": True,
        "disclosure_scope": "blueprint_sanitized_operational_records",
        "budget_policy": "project_guard_accepted_uncertainty",
        "session_retention": "until_deleted",
        "trace_retention": "provider_default",
        "provider_api_region": "us",
        "spend_limit": {
            "object": "project.spend_limit",
            "threshold_amount": 3000,
            "currency": "USD",
            "interval": "month",
        },
    }
    guard_path = tmp_path / "guard.json"
    write(guard_path, guard)
    config = service.config.model_dump(mode="json")
    config.update(
        managed_api_enabled=True,
        project_guard_receipt_file=str(guard_path),
        project_guard_receipt_digest=digest(guard),
        max_project_budget_usd=30,
    )
    write(service.config_path, config)
    template_path = Path(service.config.task_store_root) / (plan.template_task_id + ".json")
    template = json.loads(template_path.read_text())
    template["task"]["admission"].update(
        runtime="openai_agents_api",
        budget_policy="project_guard_accepted_uncertainty",
        session_retention="until_deleted",
        trace_retention="provider_default",
        region="us",
        project_guard_receipt_digest=digest(guard),
    )
    write(template_path, template)
    service = ProductionAgentService(service.config_path, source_commit="a" * 40)
    api = ContinuingAPI()

    def complete(state):
        task = service.record(state["active_task_id"]).task
        runtime = service.runtime_for_task(task)
        runtime.transport = api
        runtime.step(task.task_id)
        api.output = {
            "disposition": "no_action",
            "summary": "Retained controller revision.",
            "evidence_references": [],
            "next_actions": [],
            "uncertainty": [],
        }
        api.turn_status = "completed"
        assert runtime.step(task.task_id)["state"] == "completed"
        return task, runtime

    first, first_runtime = complete(progress_plan(service, plan))
    progress = json.loads((directory / "progression.json").read_text())
    if scope_change:
        queue, _, job = _queue(tmp_path / "new-failure")
        link = {
            "schema_version": "task_evaluation_scene_preparation_link.v1",
            "intent_id": plan.run_id,
            "intent_digest": intent["intent_digest"],
            "preparation_id": job["parent_preparation_id"],
            "request_digest": job["parent_request_digest"],
        }
        link["link_digest"] = canonical_digest(link)
        link_path = directory / "preparation-link.json"
        write(link_path, link)
        register_preparation_failure_subscription(
            preparation_link=link,
            controller_config={
                "child_queue_root": str(queue),
                "preparation_queue_root": str(tmp_path / "parents"),
            },
            agent_config_path=service.config_path,
        )
        progress.update(
            status="blocked",
            phase="preparation",
            blockers=["fixture_failure"],
            state={
                "preparation_link": {
                    "path": str(link_path),
                    "sha256": "sha256:" + hashlib.sha256(link_path.read_bytes()).hexdigest(),
                }
            },
        )
        write(directory / "progression.json", progress)
        assert progress_plan(service, plan)["status"] == "settling_prior_scope"
        first_runtime.cleanup(first.task_id)
        following = progress_plan(service, plan)
        successor = service.record(following["active_task_id"])
        assert successor.task.capability == "runtime_failure_recovery"
        assert successor.task.parent_task_id is None
        assert service.journal.task(first.task_id)["cleanup_state"] == "deleted"
        return
    progress.update(status="completed", phase="delivery", blockers=[])
    write(directory / "progression.json", progress)
    second, runtime = complete(progress_plan(service, plan))
    assert second.parent_task_id == first.task_id
    assert (
        sum(method == "POST" and path == "/agents/sessions" for method, path, *_ in api.calls) == 1
    )
    assert progress_plan(service, plan)["status"] == "final_cleanup_pending"
    runtime.cleanup(second.task_id)
    assert progress_plan(service, plan)["status"] == "completed"
    assert all(
        service.journal.task(task.task_id)["cleanup_state"] == "deleted" for task in (first, second)
    )
