"""Bounded, credential-preserving ADP authoring compatibility; no network calls."""
import io
import json
from types import SimpleNamespace
from urllib import request as urllib_request

import pytest
from pydantic import BaseModel

from blueprint_pipeline import claude_opus_authoring_invoker as authoring
from blueprint_pipeline import claude_opus_model_preflight as metadata
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import AgentsSDKAgentSpec
from blueprint_pipeline.task_evaluation_supervisor.inference_reservations import InferenceReservationAudit


class Response(io.BytesIO):
    def __init__(self, body=b'{"ok":true}', *, headers=None, url=authoring._API_URL,
                 status=200, chunk_bytes=3):
        super().__init__(body)
        self.headers = headers or {}
        self.url = url
        self.status = status
        self.chunk_bytes = chunk_bytes
        self.read_sizes = []

    def geturl(self):
        return self.url

    def getcode(self):
        return self.status

    def read(self, size=-1):
        assert size > 0, "the response must never be read without a byte bound"
        self.read_sizes.append(size)
        return super().read(min(size, self.chunk_bytes))


def _opener(monkeypatch, response=None, *, redirect=None):
    calls = []

    def build(handler):
        assert isinstance(handler, authoring._NoProviderRedirects)

        def open_request(request, *, timeout):
            calls.append((request, timeout))
            if redirect is not None:
                code, target = redirect
                handler.parent = SimpleNamespace(
                    open=lambda *_a, **_kw: pytest.fail("a redirected request was sent"))
                return getattr(handler, f"http_error_{code}")(request,
                    response if response is not None else io.BytesIO(), code, "redirect", {
                    "location": target,
                })
            return response

        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(urllib_request, "build_opener", build)
    monkeypatch.setattr(urllib_request, "urlopen",
                        lambda *_a, **_kw: pytest.fail("unrestricted urlopen was used"))
    return calls


@pytest.fixture(params=["message", "metadata"])
def adapter(request):
    if request.param == "message":
        return (authoring, "_API_URL", authoring._API_URL, "POST", 600,
                authoring._MAX_MESSAGE_RESPONSE_BYTES,
                lambda key: authoring._post_message({"model": authoring.MODEL}, key))
    return (metadata, "_MODEL_URL", metadata._MODEL_URL, "GET", 15,
            metadata._MAX_MODEL_RESPONSE_BYTES, metadata._get_model)


def test_original_request_has_exact_endpoint_timeout_and_unredirected_key(monkeypatch, adapter):
    _, _, endpoint, method, timeout, _, call = adapter
    response = Response(url=endpoint)
    calls = _opener(monkeypatch, response)
    assert call("test-only-key") == {"ok": True}
    [(request, actual_timeout)] = calls
    assert (request.full_url, request.get_method(), actual_timeout) == (endpoint, method, timeout)
    assert request.unredirected_hdrs == {"X-api-key": "test-only-key"}
    assert "X-api-key" not in request.headers
    assert request.headers == ({"Content-type": "application/json",
                                "Anthropic-version": "2023-06-01"} if method == "POST"
                               else {"Anthropic-version": "2023-06-01"})
    assert "test-only-key" not in (request.data or b"").decode()
    assert response.closed


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("target", ["https://api.anthropic.com/other", "https://example.invalid/",
                                    "ftp://example.invalid/", "invalid://example.invalid/"])
def test_redirect_is_refused_without_a_second_request(monkeypatch, adapter, code, target):
    *_, call = adapter
    response = Response()
    calls = _opener(monkeypatch, response, redirect=(code, target))
    with pytest.raises(authoring.ClaudeAuthoringBlocked,
                       match="^claude_provider_redirect_refused$") as blocked:
        call("test-only-key")
    assert len(calls) == 1
    assert target not in str(blocked.value)
    assert "test-only-key" not in str(blocked.value)
    assert response.read_sizes == []
    assert response.closed


@pytest.mark.parametrize("replacement", [
    "http://api.anthropic.com/v1/messages", "https://example.invalid/v1/messages",
    "https://api.anthropic.com/v1/messages?other=1",
    "https://api.anthropic.com:443/v1/messages", "https://api.anthropic.com/v1/messages#other",
])
def test_changed_adapter_endpoint_is_refused_before_open(monkeypatch, adapter, replacement):
    module, attribute, *_, call = adapter
    calls = _opener(monkeypatch)
    monkeypatch.setattr(module, attribute, replacement)
    with pytest.raises(authoring.ClaudeAuthoringBlocked, match="provider_endpoint_invalid"):
        call("test-only-key")
    assert calls == []


@pytest.mark.parametrize("method,url", [
    ("GET", authoring._API_URL), ("POST", metadata._MODEL_URL), ("DELETE", metadata._MODEL_URL),
])
def test_method_is_bound_to_its_exact_endpoint(monkeypatch, method, url):
    calls = _opener(monkeypatch)
    request = urllib_request.Request(url, method=method)
    with pytest.raises(authoring.ClaudeAuthoringBlocked, match="provider_endpoint_invalid"):
        authoring._request_json(request, timeout=15, maximum_bytes=64)
    assert calls == []


def test_declared_oversize_is_refused_before_read(monkeypatch, adapter):
    _, _, endpoint, _, _, maximum, call = adapter
    response = Response(headers={"Content-Length": str(maximum + 1)}, url=endpoint)
    calls = _opener(monkeypatch, response)
    with pytest.raises(authoring.ClaudeAuthoringBlocked, match="provider_response_bytes_exceeded"):
        call("test-only-key")
    assert len(calls) == 1
    assert response.read_sizes == []
    assert response.closed


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "2"}])
def test_streaming_oversize_stops_after_one_byte_beyond_bound(monkeypatch, headers):
    response = Response(b" " * 100, headers=headers, chunk_bytes=100)
    _opener(monkeypatch, response)
    with pytest.raises(authoring.ClaudeAuthoringBlocked, match="provider_response_bytes_exceeded"):
        authoring._request_json(urllib_request.Request(authoring._API_URL, method="POST"),
                                timeout=600, maximum_bytes=64)
    assert response.read_sizes == [65]
    assert response.closed


def test_exact_byte_bound_and_fragmented_json_are_accepted(monkeypatch):
    body = b'{"ok":true}' + b" " * 53
    assert len(body) == 64
    response = Response(body, headers={"Content-Length": "64"})
    _opener(monkeypatch, response)
    assert authoring._request_json(
        urllib_request.Request(authoring._API_URL, method="POST"), timeout=600,
        maximum_bytes=64) == {"ok": True}
    assert max(response.read_sizes) <= 65
    assert response.closed


@pytest.mark.parametrize("body,headers,reason", [
    (b'{"ok":true}', {"Content-Length": "20"}, "provider_response_truncated"),
    (b'{"ok":', {}, "provider_response_invalid"),
    (b"[]", {}, "provider_response_invalid"),
    (b"\xff", {}, "provider_response_invalid"),
    (b"{}", {"Content-Length": "invalid"}, "provider_response_invalid"),
])
def test_truncated_or_invalid_response_fails_closed(monkeypatch, adapter, body, headers, reason):
    _, _, endpoint, *_, call = adapter
    response = Response(body, headers=headers, url=endpoint)
    _opener(monkeypatch, response)
    with pytest.raises(authoring.ClaudeAuthoringBlocked, match=reason):
        call("test-only-key")
    assert response.closed


@pytest.mark.parametrize("change", [{"url": "https://example.invalid/"}, {"status": 302}])
def test_unexpected_response_endpoint_or_status_is_refused_before_read(monkeypatch, change):
    response = Response(**change)
    _opener(monkeypatch, response)
    with pytest.raises(authoring.ClaudeAuthoringBlocked, match="provider_response_invalid"):
        authoring._post_message({}, "test-only-key")
    assert response.read_sizes == []
    assert response.closed


@pytest.mark.parametrize("failure", ["redirect", "oversize", "truncated"])
def test_transport_refusal_keeps_full_unknown_reservation_without_replay(monkeypatch, tmp_path,
                                                                        failure):
    class Output(BaseModel):
        content: str

    response = Response(headers={"Content-Length": str(authoring._MAX_MESSAGE_RESPONSE_BYTES + 1)
                                 if failure == "oversize" else "20"})
    calls = _opener(monkeypatch, response, redirect=(307, "https://example.invalid/")
                    if failure == "redirect" else None)
    monkeypatch.setattr(authoring, "_scoped_key", lambda: "test-only-key")
    audit = InferenceReservationAudit(run_root=tmp_path, run_id="transport-fixture")
    invoker = authoring.ClaudeOpusAuthoringInvoker(authoring.ClaudeAuthoringConfig(
        run_id="transport-fixture", maximum_cost_usd=7, maximum_calls=2,
        allow_live_invocation=True), audit=audit, send=authoring._post_message,
        verify_authority=lambda run_id, _digest: {
            "run_id": run_id, "allowed_providers": ["anthropic"],
            "private_provider_processing_allowed": True, "provider_training_allowed": False,
            "authority_digest": "sha256:" + "a" * 64,
            "provider_terms_digest": "sha256:" + "a" * 64})
    spec = AgentsSDKAgentSpec(run_id="transport-fixture", capability="cad_brief", name="CAD",
        instructions="Draft a brief.", model=authoring.MODEL, max_turns=1,
        max_input_tokens=80_000, max_output_tokens=12_000, output_type=Output)
    with pytest.raises(authoring.ClaudeAuthoringBlocked, match="provider_outcome_unknown"):
        invoker.invoke(spec, "Draft")
    manifest = audit.manifest()
    assert manifest["reservation_count"] == manifest["in_flight_unknown_count"] == 1
    reserved_path = next((tmp_path / "inference_reservations/reserved").glob("*.json"))
    reservation = json.loads(reserved_path.read_text())
    assert reservation["projected_max_cost_usd"] == pytest.approx(4.664)
    assert "test-only-key" not in reserved_path.read_text()
    with pytest.raises(authoring.ClaudeAuthoringBlocked, match="spend_cap_exhausted"):
        invoker.invoke(spec, "Draft")
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["redirect", "oversize", "truncated"])
def test_metadata_transport_failure_stays_metadata_unavailable(monkeypatch, failure):
    response = Response(url=metadata._MODEL_URL,
        headers={"Content-Length": str(metadata._MAX_MODEL_RESPONSE_BYTES + 1)
                 if failure == "oversize" else "20"})
    calls = _opener(monkeypatch, response, redirect=(302, "https://example.invalid/")
                    if failure == "redirect" else None)
    monkeypatch.setattr(metadata, "_scoped_key", lambda: "test-only-key")
    with pytest.raises(authoring.ClaudeAuthoringBlocked,
                       match="^claude_model_metadata_unavailable$"):
        metadata.preflight(fetch=metadata._get_model)
    assert len(calls) == 1
