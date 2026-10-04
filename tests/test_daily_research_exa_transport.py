"""No live credentials/providers: protocol, cap, ACK and portable receipt boundaries."""
import base64
import copy
import hashlib
import json

import pytest

from tools.daily_research.exa_transport import ENDPOINT, ExaTransport, ExaTransportError

SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}, "runId": {"type": "string"},
    "budget": {"type": "object", "properties": {"maxCostDollars": {"type": "number", "maximum": 5}}}}}
REQUEST = {"query": "US regional laundry sites with towel handling evidence", "budget": {"maxCostDollars": 1.75}}
RECORD = {"id": "agent_run_synthetic", "status": "running", "outputReady": False, "usage": None}


class Wire:
    def __init__(self, *, schema=SCHEMA, record=RECORD, mode="json"):
        self.schema, self.record = copy.deepcopy(schema), copy.deepcopy(record)
        self.mode, self.calls, self.tool_result, self.response_override = mode, [], None, None

    def __call__(self, url, headers, body, timeout, max_bytes):
        request = json.loads(body)
        self.calls.append((url, dict(headers), body, timeout, max_bytes))
        if self.response_override:
            return self.response_override(request)
        if request["method"] == "notifications/initialized":
            return 202, {}, b""
        if request["method"] == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "fixture", "version": "1"}}
        elif request["method"] == "tools/list":
            result = {"tools": [{"name": "agent_run", "inputSchema": self.schema}]}
        else:
            result = self.tool_result if self.tool_result is not None else {"content": [{"type": "text", "text": json.dumps(self.record)}]}
        raw = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}, ensure_ascii=False).encode()
        if self.mode == "sse":
            notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}).encode()
            raw = b": heartbeat\r\n\r\nevent: message\r\ndata: " + notification + b"\r\n\r\ndata: " + raw + b"\r\n\r\n"
        return 200, {"Content-Type": "text/event-stream" if self.mode == "sse" else "application/json", "Mcp-Session-Id": "fixture-session"}, raw

    @property
    def native_calls(self):
        return [json.loads(call[2])["params"] for call in self.calls if json.loads(call[2])["method"] == "tools/call"]


@pytest.fixture(autouse=True)
def fake_authentication(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "synthetic-not-a-real-key")


@pytest.mark.parametrize("mode", ["json", "sse"])
def test_protocol_discovery_exact_capped_start_and_original_id_read(mode):
    wire = Wire(mode=mode)
    transport = ExaTransport(request_io=wire)
    schema = transport.discover()
    assert schema == SCHEMA
    schema.clear()  # Returned schema cannot mutate admission's cached contract.
    assert transport.start(REQUEST) == RECORD
    assert transport.read(RECORD["id"]) == RECORD
    assert wire.native_calls == [{"name": "agent_run", "arguments": REQUEST},
                                 {"name": "agent_run", "arguments": {"runId": RECORD["id"]}}]
    assert all(call[0] == ENDPOINT and 0 < call[3] <= 20 for call in wire.calls)
    assert wire.calls[2][1]["Mcp-Session-Id"] == "fixture-session"
    assert wire.calls[2][1]["MCP-Protocol-Version"] == "2025-03-26"
    receipts = transport.drain_receipts()
    assert len(receipts) == 5 and transport.drain_receipts() == []
    for receipt in receipts:
        raw = base64.b64decode(receipt["response_body_base64"])
        assert receipt["response_complete"]
        assert receipt["response_sha256"] == hashlib.sha256(raw).hexdigest()
        assert receipt["response_bytes"] == len(raw)
        assert "synthetic-not-a-real-key" not in json.dumps(receipt)


def test_full_unicode_provider_output_and_costs_are_unchanged():
    record = {**RECORD, "status": "completed", "output": {"report": "évidence 📦\n" * 9000},
              "costDollars": {"search": .1, "agentCompute": 1.2, "total": 1.3}, "extra_native_field": {"unknown": None}}
    wire = Wire(record=record)
    wire.tool_result = {"structuredContent": record, "content": [{"type": "text", "text": json.dumps(record)}]}
    transport = ExaTransport(request_io=wire)
    assert transport.start(REQUEST) == record
    raw = base64.b64decode(transport.last_receipt["response_body_base64"])
    assert json.loads(raw)["result"]["structuredContent"] == record


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"query": {"type": "string"}, "runId": {"type": "string"}}},
    {"properties": {"query": {"type": "string"}, "maxCostDollars": {"type": "number"}}},
    {**SCHEMA, "required": ["query", "effort"]},
    {"properties": {"query": {"type": "string"}, "budget": {"type": "object", "properties": {"maxCostDollars": {"type": "string"}}}}},
])
def test_no_guessed_or_unsupported_cost_cap_and_no_native_start(schema):
    wire = Wire(schema=schema)
    transport = ExaTransport(request_io=wire)
    assert transport.discover() == schema
    with pytest.raises(ExaTransportError, match="exa_mcp_supported_cost_cap_missing"):
        transport.start(REQUEST)
    assert wire.native_calls == []


@pytest.mark.parametrize("start_request", [
    {**REQUEST, "previousRunId": RECORD["id"]}, {**REQUEST, "runId": RECORD["id"]},
    {"query": "US", "maxCostDollars": 1}, {"query": "", "budget": {"maxCostDollars": 1}},
    {"query": "US", "budget": {"maxCostDollars": True}},
    {"query": "US", "budget": {"maxCostDollars": float("nan")}},
])
def test_bad_start_is_rejected_before_any_http(start_request):
    wire = Wire()
    with pytest.raises(ExaTransportError, match="exa_mcp_start_arguments_invalid"):
        ExaTransport(request_io=wire).start(start_request)
    assert not wire.calls


def test_receipt_sink_precedes_return_and_failed_sink_prevents_replay():
    wire, retained = Wire(), []

    def sink(receipt):
        retained.append(receipt)
        if json.loads(base64.b64decode(receipt["request_body_base64"]))["method"] == "tools/call":
            raise OSError("synthetic private detail must not appear")

    transport = ExaTransport(request_io=wire, receipt_sink=sink)
    with pytest.raises(ExaTransportError, match="^exa_mcp_receipt_retention_failed$") as error:
        transport.start(REQUEST)
    assert len(wire.native_calls) == 1 and len(retained) == 4
    assert json.loads(base64.b64decode(error.value.receipt["response_body_base64"]))["result"]["content"]
    with pytest.raises(ExaTransportError, match="exa_mcp_start_already_attempted"):
        transport.start(REQUEST)
    assert len(wire.native_calls) == 1


def test_uncertain_post_ack_retains_safe_error_and_never_retries():
    wire = Wire()
    transport = ExaTransport(request_io=wire)
    transport.discover()

    def unavailable(_):
        raise TimeoutError("synthetic private key in error")

    wire.response_override = unavailable
    with pytest.raises(ExaTransportError, match="^exa_mcp_request_timeout$") as error:
        transport.start(REQUEST)
    assert error.value.receipt["request_attempted"] and not error.value.receipt["response_complete"]
    assert "synthetic private key" not in json.dumps(error.value.receipt)
    with pytest.raises(ExaTransportError, match="exa_mcp_start_already_attempted"):
        transport.start(REQUEST)
    assert len(wire.native_calls) == 1


@pytest.mark.parametrize("change,code", [
    (lambda req: (401, {"content-type": "application/json"}, b'{"private":"upstream detail"}'), "exa_mcp_http_status_401"),
    (lambda req: (302, {"location": "https://another.example/"}, b"redirect"), "exa_mcp_http_status_302"),
    (lambda req: (200, {"content-type": "application/json"}, json.dumps({"jsonrpc": "2.0", "id": req["id"] + 1, "result": {}}).encode()), "exa_mcp_response_id_mismatch"),
    (lambda req: (200, {"content-type": "application/json"}, b'{"jsonrpc":"2.0","id":1,"id":2,"result":{}}'), "exa_mcp_json_invalid"),
    (lambda req: (200, {"content-type": "application/json"}, json.dumps({"jsonrpc": "2.0", "id": req["id"], "error": {"message": "private detail"}}).encode()), "exa_mcp_rpc_error"),
    (lambda req: (200, {"content-type": "text/html"}, b"html"), "exa_mcp_content_type_invalid"),
    (lambda req: (200, {"content-type": "text/event-stream"}, b'data: {"jsonrpc":"2.0","id":1,"result":{}}\n'), "exa_mcp_sse_incomplete"),
])
def test_protocol_failures_preserve_original_bytes_with_safe_codes(change, code):
    wire = Wire()
    wire.response_override = change
    transport = ExaTransport(request_io=wire)
    with pytest.raises(ExaTransportError, match="^" + code + "$") as error:
        transport.discover()
    assert error.value.receipt["response_complete"]
    assert base64.b64decode(error.value.receipt["response_body_base64"]) == change({"id": 1})[2]
    assert len(wire.calls) == 1


def test_sse_multiline_notifications_and_duplicate_responses_are_bound():
    wire = Wire()
    transport = ExaTransport(request_io=wire)
    transport.discover()

    def response(req):
        raw = json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": {"structuredContent": RECORD}}, indent=2)
        event = "".join("data: " + line + "\n" for line in raw.splitlines()) + "\n"
        return 200, {"content-type": "text/event-stream"}, (event + event).encode()

    wire.response_override = response
    with pytest.raises(ExaTransportError, match="exa_mcp_response_ambiguous"):
        transport.read(RECORD["id"])
    assert len(wire.native_calls) == 1


@pytest.mark.parametrize("result,code", [
    ({"isError": True, "content": [{"type": "text", "text": json.dumps(RECORD)}]}, "exa_mcp_tool_error"),
    ({"content": [{"type": "text", "text": "Results: " + json.dumps(RECORD)}]}, "exa_mcp_provider_record_ambiguous"),
    ({"structuredContent": RECORD, "content": [{"type": "text", "text": json.dumps({**RECORD, "status": "completed"})}]}, "exa_mcp_provider_record_ambiguous"),
    ({"structuredContent": {**RECORD, "id": "agent_run_other"}}, "exa_mcp_provider_id_mismatch"),
])
def test_tool_errors_ambiguous_text_and_wrong_original_id_are_not_success(result, code):
    wire = Wire()
    wire.tool_result = result
    transport = ExaTransport(request_io=wire)
    with pytest.raises(ExaTransportError, match=code) as error:
        transport.read(RECORD["id"])
    assert json.loads(base64.b64decode(error.value.receipt["response_body_base64"]))["result"] == result


def test_reply_size_failure_is_explicit_partial_evidence_not_silent_truncation():
    wire = Wire()
    wire.response_override = lambda req: (200, {"content-type": "application/json"}, b"x" * 11)
    transport = ExaTransport(request_io=wire, max_bytes=10)
    with pytest.raises(ExaTransportError, match="exa_mcp_reply_too_large_not_truncated") as error:
        transport.discover()
    assert error.value.receipt["response_complete"] is False
    assert base64.b64decode(error.value.receipt["response_body_base64"]) == b"x" * 11


def test_missing_runtime_key_uses_no_http_and_does_not_expose_environment(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY")
    wire = Wire()
    with pytest.raises(ExaTransportError, match="exa_mcp_authentication_missing"):
        ExaTransport(request_io=wire).discover()
    assert not wire.calls


@pytest.mark.parametrize("method", ["start", "read"])
def test_acknowledged_failed_native_record_retains_error_envelope_without_unknown_zero(method):
    failed = {**RECORD, "status": "failed", "success": False, "error": {"type": "synthetic"}, "costDollars": None}
    wire = Wire()
    wire.tool_result = {"isError": True, "content": [{"type": "text", "text": json.dumps(failed)},
        {"type": "text", "text": "The Agent run failed. Inspect the error above."}]}
    transport = ExaTransport(request_io=wire)
    result = transport.start(REQUEST) if method == "start" else transport.read(RECORD["id"])
    assert result == failed and result["costDollars"] is None
    assert json.loads(base64.b64decode(transport.last_receipt["response_body_base64"]))["result"]["isError"] is True
    assert len(wire.native_calls) == 1


def test_actual_http_adapter_is_fixed_host_bounded_and_closes_without_redirect_or_retry(monkeypatch):
    from tools.daily_research import exa_transport as module

    calls = []

    class Connection:
        def __init__(self, host, timeout):
            calls.append(("connect", host, timeout))
        def request(self, method, path, body, headers):
            calls.append(("post", method, path, body))
        def getresponse(self):
            return self
        def getheaders(self):
            return [("Content-Type", "application/json")]
        def read(self, maximum):
            calls.append(("read", maximum))
            return b"redirect body"
        def close(self):
            calls.append(("close",))
        status = 302

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    status, headers, raw = module._http_post(ENDPOINT, {}, b"request", 2, 15)
    assert status == 302 and raw == b"redirect body"
    assert calls == [("connect", "mcp.exa.ai", 2), ("post", "POST", "/mcp?tools=agent_run", b"request"),
                     ("read", 16), ("close",)]


def test_elapsed_operation_deadline_retains_returned_bytes_before_refusing_ack(monkeypatch):
    from tools.daily_research import exa_transport as module

    current = [0]
    monkeypatch.setattr(module.time, "monotonic", lambda: current[0])
    wire = Wire()
    original_io = wire.__call__

    def delayed(*args):
        result = original_io(*args)
        current[0] = 21
        return result

    transport = ExaTransport(request_io=delayed, timeout=20)
    with pytest.raises(ExaTransportError, match="exa_mcp_request_timeout") as error:
        transport.discover()
    assert len(wire.calls) == 1 and error.value.receipt["response_complete"]


@pytest.mark.parametrize("mode", ["json", "sse"])
def test_retained_ack_recovers_after_deadline_with_no_credential_http_or_schema(mode, monkeypatch):
    from tools.daily_research import exa_transport as module

    wire, saved = Wire(mode=mode), []
    transport = ExaTransport(request_io=wire, receipt_sink=saved.append)
    transport.discover()
    current = [0]
    monkeypatch.setattr(module.time, "monotonic", lambda: current[0])
    original_io = wire.__call__

    def delayed(*args):
        response = original_io(*args)
        current[0] = 30
        return response

    transport.request_io = delayed
    with pytest.raises(ExaTransportError, match="exa_mcp_request_timeout"):
        transport.start(REQUEST)
    ack = saved[-1]
    assert ack["operation"] == "tools/call" and ack["response_complete"]
    monkeypatch.delenv("EXA_API_KEY")
    monkeypatch.setattr(module, "_http_post", lambda *a, **k: pytest.fail("No recovery HTTP"))
    monkeypatch.setattr(module.ExaTransport, "discover", lambda *a: pytest.fail("No recovery schema query"))
    monkeypatch.setattr(module.time, "monotonic", lambda: pytest.fail("Recovery has no active deadline"))
    assert module.reconcile_start_ack(ack, REQUEST) == RECORD
    assert len(wire.native_calls) == 1


@pytest.fixture
def retained_ack():
    wire = Wire()
    transport = ExaTransport(request_io=wire)
    transport.start(REQUEST)
    return transport.last_receipt


@pytest.mark.parametrize("change,code", [
    ({"response_sha256": "0" * 64}, "exa_mcp_retained_ack_digest_mismatch"),
    ({"request_sha256": "0" * 64}, "exa_mcp_retained_ack_digest_mismatch"),
    ({"response_body_base64": "not base64"}, "exa_mcp_retained_ack_bytes_invalid"),
    ({"response_bytes": 1}, "exa_mcp_retained_ack_length_mismatch"),
    ({"response_complete": False}, "exa_mcp_retained_ack_envelope_invalid"),
    ({"http_status": 202}, "exa_mcp_retained_ack_envelope_invalid"),
    ({"operation": "tools/list"}, "exa_mcp_retained_ack_envelope_invalid"),
    ({"endpoint": "https://another.example/mcp"}, "exa_mcp_retained_ack_envelope_invalid"),
    ({"request_id": 7}, "exa_mcp_retained_ack_request_binding_mismatch"),
])
def test_retained_ack_hash_length_completeness_and_rpc_binding_are_required(retained_ack, change, code):
    from tools.daily_research.exa_transport import reconcile_start_ack

    retained_ack.update(change)
    with pytest.raises(ExaTransportError, match=code):
        reconcile_start_ack(retained_ack, REQUEST)


@pytest.mark.parametrize("expected", [
    {**REQUEST, "query": "Different frozen query"},
    {**REQUEST, "budget": {"maxCostDollars": 2}},
    {**REQUEST, "runId": RECORD["id"]},
    {**REQUEST, "previousRunId": RECORD["id"]},
])
def test_retained_ack_cannot_bind_another_start_or_read(retained_ack, expected):
    from tools.daily_research.exa_transport import reconcile_start_ack

    with pytest.raises(ExaTransportError, match="exa_mcp_retained_ack_request_binding_mismatch"):
        reconcile_start_ack(retained_ack, expected)


def test_retained_ack_known_failed_record_is_recovered_unchanged():
    from tools.daily_research.exa_transport import reconcile_start_ack

    failed = {**RECORD, "status": "failed", "error": "synthetic", "costDollars": None}
    wire = Wire()
    wire.tool_result = {"isError": True, "structuredContent": failed}
    transport = ExaTransport(request_io=wire)
    transport.start(REQUEST)
    assert reconcile_start_ack(transport.last_receipt, REQUEST) == failed


@pytest.mark.parametrize("target", ["request", "response", "status"])
def test_retained_ack_valid_digests_cannot_hide_changed_arguments_response_id_or_missing_status(retained_ack, target):
    from tools.daily_research.exa_transport import reconcile_start_ack

    side = "request" if target == "request" else "response"
    raw = json.loads(base64.b64decode(retained_ack[side + "_body_base64"]))
    if target == "request":
        raw["params"]["arguments"]["runId"] = RECORD["id"]
        code = "exa_mcp_retained_ack_request_binding_mismatch"
    elif target == "response":
        raw["id"] += 1
        code = "exa_mcp_response_id_mismatch"
    else:
        native = json.loads(raw["result"]["content"][0]["text"])
        native.pop("status")
        raw["result"]["content"][0]["text"] = json.dumps(native)
        code = "exa_mcp_retained_ack_status_invalid"
    encoded = json.dumps(raw).encode()
    retained_ack[side + "_body_base64"] = base64.b64encode(encoded).decode()
    retained_ack[side + "_sha256"] = hashlib.sha256(encoded).hexdigest()
    if side == "response":
        retained_ack["response_bytes"] = len(encoded)
    with pytest.raises(ExaTransportError, match=code):
        reconcile_start_ack(retained_ack, REQUEST)


def test_retained_ack_parser_never_constructs_transport(retained_ack, monkeypatch):
    from tools.daily_research.exa_transport import reconcile_start_ack

    monkeypatch.setattr(ExaTransport, "__init__", lambda *a, **k: pytest.fail("No transport construction"))
    assert reconcile_start_ack(retained_ack, REQUEST) == RECORD
