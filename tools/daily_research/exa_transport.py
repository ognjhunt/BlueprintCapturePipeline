"""Bounded, retry-free HTTP transport for the existing official Exa agent MCP.

Credentials are read from the worker's EXA_API_KEY environment only. Discovery
is read-only; callers own durable start claims and all-in spending admission.
Raw HTTP receipts are separate from unchanged provider records. A receipt_sink
can retain each response before a provider acknowledgement leaves this adapter.
"""
import base64
import copy
import hashlib
import http.client
import json
import math
import os
import re
import signal
import time
from contextlib import nullcontext
from datetime import datetime, timezone

ENDPOINT = "https://mcp.exa.ai/mcp?tools=agent_run"
PROTOCOLS = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}
PROTOCOL = "2025-03-26"
MAX_BYTES = 5_000_000


class ExaTransportError(ValueError):
    """Stable public code; private raw evidence is available in receipt."""

    def __init__(self, code, receipt=None):
        super().__init__(code)
        self.code, self.receipt = code, copy.deepcopy(receipt)


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("nonfinite_json")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _http_post(url, headers, body, timeout, max_bytes):
    """Exact host/path, no redirect/retry; existing Linux watchdog covers DNS."""
    from tools.daily_research.search import bounded_request

    if url != ENDPOINT:
        raise ExaTransportError("exa_mcp_endpoint_invalid")
    # An outer application-tool watchdog already bounds the original call.
    timer = signal.getitimer(signal.ITIMER_REAL)[0]
    guard = nullcontext() if timer > 0 else bounded_request(timeout)
    connection = http.client.HTTPSConnection("mcp.exa.ai", timeout=timeout)
    try:
        with guard:
            connection.request("POST", "/mcp?tools=agent_run", body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(max_bytes + 1)
            return response.status, dict(response.getheaders()), raw
    finally:
        connection.close()


class ExaTransport:
    """request_io(url, headers, body, timeout, max_bytes) returns status/headers/bytes.

    Each public operation has a bounded wall-time allowance (also bounded by the
    owner's outer watchdog/deadline). start is never automatically repeated.
    discover returns the actual schema, including schemas without a cost cap.
    """

    def __init__(self, *, request_io=None, receipt_sink=None, timeout=20, max_bytes=MAX_BYTES):
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 800
                or type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES):
            raise ExaTransportError("exa_mcp_bounds_invalid")
        self.request_io = request_io or _http_post
        self.receipt_sink = receipt_sink
        self.timeout, self.max_bytes = timeout, max_bytes
        self.receipts, self.last_receipt = [], None
        self._sequence, self._session, self._protocol, self._schema = 0, None, None, None
        self._start_attempted = False

    def drain_receipts(self):
        receipts, self.receipts = self.receipts, []
        return copy.deepcopy(receipts)

    def _fail(self, code):
        raise ExaTransportError(code, self.last_receipt)

    def _retain(self, record):
        self.last_receipt = copy.deepcopy(record)
        self.receipts.append(copy.deepcopy(record))
        if self.receipt_sink:
            try:
                self.receipt_sink(copy.deepcopy(record))
            except Exception:
                self._fail("exa_mcp_receipt_retention_failed")

    def _post(self, message, deadline, operation):
        remaining = min(self.timeout, deadline - time.monotonic())
        if remaining <= 0:
            self._fail("exa_mcp_request_timeout")
        key = os.environ.get("EXA_API_KEY")
        if not isinstance(key, str) or not key.strip() or any(ord(c) < 33 or ord(c) > 126 for c in key):
            self._fail("exa_mcp_authentication_missing")
        body = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "x-api-key": key}
        if self._session:
            headers["Mcp-Session-Id"] = self._session
        if self._protocol:
            headers["MCP-Protocol-Version"] = self._protocol
        record = {"endpoint": ENDPOINT, "operation": operation, "request_id": message.get("id"),
                  "observed_at": datetime.now(timezone.utc).isoformat(), "request_attempted": True,
                  "request_body_base64": base64.b64encode(body).decode(),
                  "request_sha256": hashlib.sha256(body).hexdigest(), "response_complete": False}
        try:
            status, response_headers, raw = self.request_io(ENDPOINT, headers, body, remaining, self.max_bytes)
        except Exception as error:
            code = "exa_mcp_request_timeout" if isinstance(error, TimeoutError) or str(error) == "research_tool_absolute_deadline" else "exa_mcp_transport_unavailable"
            self._retain({**record, "error_code": code})
            self._fail(code)
        if (type(status) is not int or not isinstance(response_headers, dict) or not isinstance(raw, bytes)):
            self._retain({**record, "error_code": "exa_mcp_io_contract_invalid"})
            self._fail("exa_mcp_io_contract_invalid")
        response_headers = {str(k).lower(): str(v) for k, v in response_headers.items()}
        complete = len(raw) <= self.max_bytes
        record.update(http_status=status, content_type=response_headers.get("content-type", ""),
                      response_complete=complete, response_bytes=len(raw),
                      response_sha256=hashlib.sha256(raw).hexdigest(),
                      response_body_base64=base64.b64encode(raw).decode())
        self._retain(record)  # Durable private bytes precede ACK/protocol parsing.
        if not complete:
            self._fail("exa_mcp_reply_too_large_not_truncated")
        if time.monotonic() >= deadline:
            self._fail("exa_mcp_request_timeout")
        if status not in (200, 202):
            self._fail("exa_mcp_http_status_" + str(status))
        return raw, response_headers, status

    def _messages(self, raw, content_type):
        try:
            text = raw.decode("utf-8-sig")
            media = content_type.split(";", 1)[0].strip().lower()
            if media == "application/json":
                return [_json(text)]
            if media != "text/event-stream":
                self._fail("exa_mcp_content_type_invalid")
            messages, data = [], []
            # SSE data dispatch requires a blank line; retain incomplete streams.
            for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
                if not line:
                    if data:
                        messages.append(_json("\n".join(data)))
                        data = []
                elif line.startswith("data:"):
                    value = line[5:]
                    data.append(value[1:] if value.startswith(" ") else value)
            if data:
                self._fail("exa_mcp_sse_incomplete")
            return messages
        except ExaTransportError:
            raise
        except (ValueError, UnicodeError, RecursionError):
            self._fail("exa_mcp_json_invalid")

    def _rpc(self, method, params, deadline):
        self._sequence += 1
        identifier = self._sequence
        message = {"jsonrpc": "2.0", "id": identifier, "method": method, "params": params}
        raw, headers, status = self._post(message, deadline, method)
        if status != 200:
            self._fail("exa_mcp_response_missing")
        matched = []
        for response in self._messages(raw, headers.get("content-type", "")):
            if not isinstance(response, dict) or response.get("jsonrpc") != "2.0":
                self._fail("exa_mcp_protocol_invalid")
            if "id" not in response and isinstance(response.get("method"), str):
                continue  # Notifications/progress are retained in the complete raw stream.
            if type(response.get("id")) is not int or response["id"] != identifier:
                self._fail("exa_mcp_response_id_mismatch")
            matched.append(response)
        if len(matched) != 1:
            self._fail("exa_mcp_response_ambiguous")
        response = matched[0]
        if "error" in response:
            self._fail("exa_mcp_rpc_error")
        if not isinstance(response.get("result"), dict):
            self._fail("exa_mcp_result_invalid")
        return response["result"], headers

    def _discover(self, deadline):
        if self._schema is not None:
            return copy.deepcopy(self._schema)
        if self._protocol is None:
            result, headers = self._rpc("initialize", {"protocolVersion": PROTOCOL,
                "capabilities": {}, "clientInfo": {"name": "blueprint-daily-research", "version": "1"}}, deadline)
            if result.get("protocolVersion") not in PROTOCOLS:
                self._fail("exa_mcp_protocol_version_unsupported")
            session = headers.get("mcp-session-id")
            if session is not None and (not session or any(ord(c) < 33 or ord(c) > 126 for c in session)):
                self._fail("exa_mcp_session_invalid")
            self._session, self._protocol = session, result["protocolVersion"]
            raw, headers, _ = self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, deadline, "notifications/initialized")
            if raw.strip():
                self._fail("exa_mcp_notification_response_invalid")
        result, _ = self._rpc("tools/list", {}, deadline)
        tools = result.get("tools")
        if not isinstance(tools, list) or result.get("nextCursor"):
            self._fail("exa_mcp_tool_catalog_incomplete")
        matches = [tool for tool in tools if isinstance(tool, dict) and tool.get("name") == "agent_run"]
        if len(matches) != 1 or not isinstance(matches[0].get("inputSchema"), dict):
            self._fail("exa_mcp_agent_run_schema_missing")
        self._schema = copy.deepcopy(matches[0]["inputSchema"])
        return copy.deepcopy(self._schema)

    def discover(self):
        return self._discover(time.monotonic() + self.timeout)

    def _record(self, result, expected_id=None):
        flagged_error = result.get("isError") is True
        if "isError" in result and type(result["isError"]) is not bool:
            self._fail("exa_mcp_tool_result_invalid")
        records = []
        if "structuredContent" in result:
            if not isinstance(result["structuredContent"], dict):
                self._fail("exa_mcp_provider_record_invalid")
            records.append(result["structuredContent"])
        content = result.get("content", [])
        if not isinstance(content, list):
            self._fail("exa_mcp_tool_result_invalid")
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                try:
                    candidate = _json(block["text"])
                except (ValueError, RecursionError):
                    continue
                if isinstance(candidate, dict):
                    records.append(candidate)
        if not records or any(record != records[0] for record in records[1:]):
            self._fail("exa_mcp_tool_error" if flagged_error else "exa_mcp_provider_record_ambiguous")
        record = records[0]
        if flagged_error and record.get("status") != "failed":
            self._fail("exa_mcp_tool_error")
        if not isinstance(record.get("id"), str) or not re.fullmatch(r"agent_run_[A-Za-z0-9_-]+", record["id"]):
            self._fail("exa_mcp_provider_id_missing")
        if expected_id is not None and record["id"] != expected_id:
            self._fail("exa_mcp_provider_id_mismatch")
        return copy.deepcopy(record)

    def start(self, request):
        from tools.daily_research.expansion import _cap_supported

        deadline = time.monotonic() + self.timeout
        if self._start_attempted:
            self._fail("exa_mcp_start_already_attempted")
        if (not isinstance(request, dict) or set(request) != {"query", "budget"}
                or not isinstance(request.get("query"), str) or not request["query"].strip()
                or not isinstance(request.get("budget"), dict) or set(request["budget"]) != {"maxCostDollars"}
                or type(request["budget"]["maxCostDollars"]) not in (int, float)
                or not math.isfinite(request["budget"]["maxCostDollars"]) or request["budget"]["maxCostDollars"] <= 0):
            self._fail("exa_mcp_start_arguments_invalid")
        if not _cap_supported(self._discover(deadline), request):
            self._fail("exa_mcp_supported_cost_cap_missing")
        self._start_attempted = True
        result, _ = self._rpc("tools/call", {"name": "agent_run", "arguments": copy.deepcopy(request)}, deadline)
        return self._record(result)

    def read(self, original_run_id):
        if not isinstance(original_run_id, str) or not re.fullmatch(r"agent_run_[A-Za-z0-9_-]+", original_run_id):
            self._fail("exa_mcp_original_run_id_invalid")
        deadline = time.monotonic() + self.timeout
        schema = self._discover(deadline)
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not isinstance(properties.get("runId"), dict) or properties["runId"].get("type") != "string":
            self._fail("exa_mcp_original_read_schema_missing")
        result, _ = self._rpc("tools/call", {"name": "agent_run", "arguments": {"runId": original_run_id}}, deadline)
        return self._record(result, original_run_id)
