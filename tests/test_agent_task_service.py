"""Queue admission, restart and signed events drive the real session adapter."""

import base64
import hashlib
import hmac
import json
import time
import threading

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from blueprint_pipeline.agent_execution.service import AgentTaskService
from tests.test_agent_execution_sessions import FakeAPI, make_task, runtime
from tests.test_agent_execution_continuation import completed_parent, successor


SECRET = "whsec_" + base64.b64encode(b"only-a-hermetic-webhook-signing-secret").decode()


def sign(event, *, secret=SECRET, timestamp=None):
    raw = json.dumps(event, separators=(",", ":")).encode()
    timestamp = str(int(time.time()) if timestamp is None else timestamp)
    message = event["id"] + "." + timestamp + "." + raw.decode()
    signature = base64.b64encode(hmac.new(base64.b64decode(secret[6:]), message.encode(), hashlib.sha256).digest()).decode()
    return raw, {"webhook-id": event["id"], "webhook-timestamp": timestamp,
                 "webhook-signature": "v1," + signature}


def service_fixture(tmp_path):
    clock = [1000.0]
    api = FakeAPI()
    run, _ = runtime(tmp_path, api=api, clock=lambda: clock[0])
    service = AgentTaskService(journal=run.journal, runtime_for_task=lambda _: run,
                               validate_admission=lambda _: None, clock=lambda: clock[0])
    return service, run, api, clock


def test_queue_then_signed_event_then_terminal_cleanup(tmp_path):
    service, run, api, _ = service_fixture(tmp_path)
    task = make_task()
    assert service.enqueue(task)["state"] == "queued"
    assert not api.calls
    assert service.tick()["state"] == "running"
    api.turn_status = "completed"
    raw, headers = sign({"id": "event_fixture", "type": "agent.session.idle", "data": {"id": "session_1"}})
    assert service.receive_webhook(raw, headers, signing_secret=SECRET)["woken"]
    assert service.tick()["state"] == "completed"
    assert service.receive_webhook(raw, headers, signing_secret=SECRET)["duplicate"]
    assert service.tick() is None
    service.request_cleanup(task.task_id)
    assert service.tick()["cleanup_state"] == "deleted"
    assert api.deleted
    assert run.inspect(task.task_id)["result"]["output"] == {"answer": 7}


def test_recover_after_process_restart_without_any_webhook(tmp_path):
    service, _, api, clock = service_fixture(tmp_path)
    task = make_task()
    service.enqueue(task)
    service.tick()
    api.turn_status = "completed"
    restarted_runtime, _ = runtime(tmp_path, api=api, clock=lambda: clock[0])
    restarted = AgentTaskService(journal=restarted_runtime.journal,
                                 runtime_for_task=lambda _: restarted_runtime,
                                 validate_admission=lambda _: None, clock=lambda: clock[0])
    assert restarted.recover() == 1
    assert restarted.tick()["state"] == "completed"
    assert sum(method == "POST" and path == "/agents/sessions" for method, path, *_ in api.calls) == 1


@pytest.mark.parametrize("kind", ["bad_signature", "old_timestamp", "altered_body"])
def test_webhook_authentication_precedes_state_mutation(tmp_path, kind):
    service, _, api, _ = service_fixture(tmp_path)
    service.enqueue(make_task())
    event = {"id": "event_fixture", "type": "agent.session.idle", "data": {"id": "session_1"}}
    raw, headers = sign(event, timestamp=1 if kind == "old_timestamp" else None)
    if kind == "bad_signature":
        headers["webhook-signature"] = "v1,invalid"
    elif kind == "altered_body":
        raw += b" "
    with pytest.raises(AgentExecutionError, match="signature_invalid"):
        service.receive_webhook(raw, headers, signing_secret=SECRET)
    assert not api.calls
    assert service.journal.event("webhook_event_fixture") is None


def test_signed_but_unowned_session_does_not_create_work(tmp_path):
    service, _, api, _ = service_fixture(tmp_path)
    raw, headers = sign({"id": "event_fixture", "type": "agent.session.created", "data": {"id": "foreign"}})
    assert service.receive_webhook(raw, headers, signing_secret=SECRET)["reason"] == "session_not_owned"
    assert service.tick() is None
    assert not api.calls


def test_cancel_queued_task_never_creates_remote_session(tmp_path):
    service, run, api, _ = service_fixture(tmp_path)
    task = make_task()
    service.enqueue(task)
    service.cancel(task.task_id)
    assert service.tick()["state"] == "cancelled"
    assert not api.calls
    assert run.inspect(task.task_id)["result"] is None


def test_webhook_cannot_claim_completion(tmp_path):
    service, _, api, _ = service_fixture(tmp_path)
    task = make_task()
    service.enqueue(task)
    service.tick()
    raw, headers = sign({"id": "event_fixture", "type": "agent.session.idle", "data": {"id": "session_1"}})
    service.receive_webhook(raw, headers, signing_secret=SECRET)
    assert service.tick()["state"] == "running"
    assert api.turn_status == "in_progress"


def test_invalid_continuation_cannot_reserve_or_poison_successor_slot(tmp_path):
    run, api, task = completed_parent(tmp_path)
    service = AgentTaskService(journal=run.journal, runtime_for_task=lambda _: run,
                               validate_admission=lambda _: None, clock=lambda: 1000)
    before = len(api.calls)
    with pytest.raises(AgentExecutionError, match="configuration_changed"):
        service.enqueue(successor(task, model="gpt-6-astra"))
    assert run.journal.successor(task.task_id) is None
    assert len(api.calls) == before
    service.request_cleanup(task.task_id)
    assert service.tick()["cleanup_state"] == "deleted"


def test_cleanup_admission_serializes_with_successor_admission(tmp_path, monkeypatch):
    run, _, task = completed_parent(tmp_path)
    service = AgentTaskService(journal=run.journal, runtime_for_task=lambda _: run,
                               validate_admission=lambda _: None, clock=lambda: 1000)
    entered, release = threading.Event(), threading.Event()
    original = run.journal.cleanup_state
    errors = []

    def paused_cleanup(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)

    def clean():
        try:
            service.request_cleanup(task.task_id)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(run.journal, "cleanup_state", paused_cleanup)
    worker = threading.Thread(target=clean)
    worker.start()
    try:
        assert entered.wait(2)
        with pytest.raises(AgentExecutionError, match="owned_by_another_worker"):
            service.enqueue(successor(task))
    finally:
        release.set()
        worker.join(3)
    assert not errors
    assert run.journal.successor(task.task_id) is None
    with pytest.raises(AgentExecutionError, match="parent_not_settled"):
        service.enqueue(successor(task))
