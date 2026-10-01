"""Future keyed sends use the actual SDK with an offline HTTP transport."""

import json

import httpx2
import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from blueprint_pipeline.agent_execution.openai_transport import AgentTransportError
from .event_transport import DurableEvents, sdk_client
from . import soft_pilot as base
from .test_soft_pilot import admitted, API  # noqa: F401 - fixture registration


@pytest.fixture
def future_transport(admitted):  # noqa: F811
    root, receipt, _, clock = admitted
    monitor = base.SoftMonitor(root, receipt, clock=clock)
    monitor.reserve()
    fallback = API("parallel_fast", clock)
    runtime, task = base.make_runtime(root, receipt, monitor, "parallel_fast", transport=fallback)
    runtime.start(task)
    return root, monitor, runtime, task, fallback, clock


def transport_fixture(state, handler):
    root, monitor, runtime, _, fallback, clock = state
    return DurableEvents(fallback=fallback, journal=runtime.journal, receipt_root=monitor.path / "future_events",
        clock=clock, timer=clock,
        client_factory=lambda: sdk_client("MOCK_SECRET_NEVER_NETWORK",base.PROJECT,
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler),follow_redirects=False)))


def message(text="Mock public task"):
    return {"events": [{"type": "agent.session.input.message", "input": [{"role": "user",
        "content": [{"type": "input_text", "text": text}]}]}]}


def test_future_key_and_exact_payload_exist_before_socket_and_survive_retry(future_transport):
    _, monitor, runtime, task, fallback, clock = future_transport
    path = "/agents/sessions/" + fallback.session["id"] + "/events"
    seen = []
    def handler(request):
        intent = json.loads(next((monitor.path / "future_events").glob("*/intent.json")).read_text())
        assert json.loads(intent["payload"]) == json.loads(request.content) == message()
        assert intent["idempotency_key"] == request.headers["Idempotency-Key"]
        assert runtime.journal.event("hosted_keyed_message_" + task.task_id)
        assert next((monitor.path / "future_events").glob("*/send_1.json")).exists()
        seen.append((request.headers["Idempotency-Key"], request.content))
        clock.sleep(0.25)
        if len(seen) == 1:
            raise httpx2.ReadTimeout("unsafe secret/url text is never recorded",request=request)
        return httpx2.Response(202,headers={"x-request-id":"req_second"},content=b"")
    transport = transport_fixture(future_transport, handler)
    with pytest.raises(AgentTransportError, match="connection_uncertain"):
        transport.request("POST",path,body=message())
    # A recreated wrapper reads the old durable slot/key, never a fresh key.
    again = transport_fixture(future_transport, handler)
    assert again.request("POST",path,body=message()) == {}
    assert seen[0] == seen[1]
    assert again.request("POST",path,body=message()) == {} and len(seen) == 2
    rows = [json.loads(p.read_text()) for p in sorted((monitor.path / "future_events").glob("*/reply_*.json"))]
    assert rows[0]["exception_type"] == "APITimeoutError" and rows[0]["cause_type"] == "ReadTimeout"
    assert rows[0]["http_status"] is None and rows[0]["latency_seconds"] == 0.25
    assert rows[1]["http_status"] == 202 and rows[1]["request_id"] == "req_second"
    assert "unsafe secret" not in json.dumps(rows) and "MOCK_SECRET" not in json.dumps(rows)


def test_changed_payload_and_tampered_sidecar_refuse_dispatch(future_transport):
    seen = []
    transport = transport_fixture(future_transport,lambda r: seen.append(r) or httpx2.Response(202,content=b""))
    _, monitor, _, task, fallback, _ = future_transport
    path = "/agents/sessions/" + fallback.session["id"] + "/events"
    slot, _ = transport.prepare(path,message(),task=task)
    with pytest.raises(AgentExecutionError, match="identity_conflict"):
        transport.request("POST",path,body=message("changed bytes"))
    (monitor.path / "future_events" / slot / "intent.json").write_text("{}")
    with pytest.raises((AgentExecutionError, ValueError)):
        transport.request("POST",path,body=message())
    assert seen == []


def test_persistent_unknown_has_only_two_transmissions(future_transport):
    seen = []
    def timeout(request):
        seen.append(request.headers["Idempotency-Key"])
        raise httpx2.ConnectTimeout("private provider detail",request=request)
    transport = transport_fixture(future_transport,timeout)
    fallback = future_transport[4]
    path = "/agents/sessions/" + fallback.session["id"] + "/events"
    for _ in range(2):
        with pytest.raises(AgentTransportError,match="connection_uncertain"):
            transport.request("POST",path,body=message())
    with pytest.raises(AgentExecutionError,match="two_transmissions_exhausted"):
        transport.request("POST",path,body=message())
    assert len(seen) == 2 and len(set(seen)) == 1


def test_retry_window_and_task_deadline_are_not_renewed(future_transport):
    _, _, _, _, fallback, clock = future_transport
    seen = []
    def timeout(request):
        seen.append(request)
        raise httpx2.ReadTimeout("private detail",request=request)
    transport = transport_fixture(future_transport,timeout)
    path = "/agents/sessions/" + fallback.session["id"] + "/events"
    with pytest.raises(AgentTransportError):
        transport.request("POST",path,body=message())
    clock.sleep(61)
    restarted = transport_fixture(future_transport,timeout)
    with pytest.raises(AgentExecutionError,match="retry_window_expired"):
        restarted.request("POST",path,body=message())
    clock.sleep(300)
    with pytest.raises(AgentExecutionError,match="cancelled_or_expired"):
        restarted.request("POST",path,body=message())
    assert len(seen) == 1


@pytest.mark.parametrize("status, rejected", [(400,True),(409,False),(500,False)])
def test_http_error_retains_only_safe_diagnostics(future_transport,status,rejected):
    seen = []
    def fail(request):
        seen.append(request)
        return httpx2.Response(status,headers={"x-request-id":"req_failed"},
            json={"error":{"message":"SECRET URL/BODY","code":"sensitive-body-ignored","type":"server_error"}})
    transport = transport_fixture(future_transport,fail)
    path = "/agents/sessions/" + future_transport[4].session["id"] + "/events"
    with pytest.raises(AgentTransportError) as exc:
        transport.request("POST",path,body=message())
    assert exc.value.status == status and exc.value.diagnostics["request_id"] == "req_failed"
    assert exc.value.definitively_rejected == rejected
    assert "SECRET" not in json.dumps(exc.value.diagnostics) and "sensitive-body" not in json.dumps(exc.value.diagnostics)
    if rejected:
        with pytest.raises(AgentTransportError):
            transport.request("POST",path,body=message())
        assert len(seen) == 1


def test_no_authorization_redirects(future_transport):
    seen = []
    def redirect(request):
        seen.append(request)
        return httpx2.Response(302,headers={"location":"https://unapproved.example/credential-sink"})
    transport = transport_fixture(future_transport,redirect)
    path = "/agents/sessions/" + future_transport[4].session["id"] + "/events"
    with pytest.raises(AgentTransportError):
        transport.request("POST",path,body=message())
    assert len(seen) == 1 and seen[0].url.host == "api.openai.com"


def test_sdk_endpoint_ignores_environment_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL","https://unapproved.example/v1")
    seen = []
    def handler(request):
        seen.append(request)
        return httpx2.Response(202,content=b"")
    with sdk_client("MOCK_NO_NETWORK",base.PROJECT,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler),follow_redirects=False)) as client:
        client.beta.agents.sessions.events.create("sess_mock",events=message()["events"],idempotency_key="mock-key")
    assert len(seen) == 1 and seen[0].url.host == "api.openai.com"
    assert seen[0].url.path == "/v1/agents/sessions/sess_mock/events"
    assert seen[0].headers["OpenAI-Project"] == base.PROJECT


def test_current_unkeyed_unknown_cannot_gain_a_retroactive_key(future_transport):
    _, monitor, runtime, task, fallback, _ = future_transport
    path = "/agents/sessions/" + fallback.session["id"] + "/events"
    runtime.journal.prepare_continuation(task.task_id,message(),["turn_original"])
    runtime.journal.continuation_delivery(task.task_id,"sent_unknown",clock=future_transport[-1])
    before = runtime.journal.continuation(task.task_id)
    seen = []
    transport = transport_fixture(future_transport,lambda r: seen.append(r) or httpx2.Response(202,content=b""))
    for payload in (message(), {"events":[{"type":"agent.session.input.cancel"}]}):
        with pytest.raises(AgentExecutionError,match="historical_unkeyed"):
            transport.request("POST",path,body=payload)
    assert runtime.journal.continuation(task.task_id) == before
    assert seen == [] and not list((monitor.path / "future_events").glob("**/intent.json"))


def test_cancellations_for_different_owned_tasks_do_not_share_keys(future_transport):
    _, _, runtime, task, fallback, _ = future_transport
    transport = transport_fixture(future_transport,lambda r: httpx2.Response(202,content=b""))
    path = "/agents/sessions/" + fallback.session["id"] + "/events"
    payload = {"events":[{"type":"agent.session.input.cancel"}]}
    first = transport.prepare(path,payload,task=task)[1]["idempotency_key"]
    # Separate immutable child scopes prevent a later cancellation from using
    # a prior turn's accepted cancellation key, even with the same event bytes.
    runtime.journal.set_state(task.task_id,"cancelled")
    data = task.model_dump(mode="json")
    data.update(task_id=task.task_id+"_child",parent_task_id=task.task_id)
    child = base.AgentTask.model_validate(data)
    runtime.journal.register(child)
    runtime.journal.bind_session(child.task_id,fallback.session["id"])
    second = transport.prepare(path,payload,task=child)[1]["idempotency_key"]
    assert first != second
