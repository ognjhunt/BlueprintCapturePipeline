"""Provider transport consolidation (C4): one boundary, classified retries."""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from typing import Any

import pytest

from blueprint_pipeline import safe_outbound_http
from blueprint_pipeline.provider_transport import (
    DEFAULT_PROVIDER_MAX_RESPONSE_BYTES,
    classify_operation,
    provider_json_request,
    provider_text_request,
)
from blueprint_pipeline.transport_retry_policy import (
    MutationRetryForbidden,
    bounded_read_retry,
)


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, url: str = "") -> None:
        self._body = body
        self.status = status
        self._url = url

    def read(self, amt: int | None = None) -> bytes:
        if amt is None:
            return self._body
        chunk, self._body = self._body[:amt], self._body[amt:]
        return chunk

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _install_boundary_fake(
    monkeypatch: pytest.MonkeyPatch,
    *,
    body: bytes = b"{}",
    status: int = 200,
    fail_first: type[BaseException] | None = None,
):
    calls: list[dict[str, Any]] = []
    state = {"failed": False}

    def fake(request: urllib.request.Request, timeout: float, policy):  # noqa: ANN001
        calls.append({"request": request, "timeout": timeout, "policy": policy})
        if fail_first is not None and not state["failed"]:
            state["failed"] = True
            raise fail_first("transient")
        return _FakeResponse(body, status=status, url=request.full_url)

    monkeypatch.setattr(safe_outbound_http, "_open_with_policy", fake)
    return calls


def _forbid_raw_urlopen(monkeypatch: pytest.MonkeyPatch, module) -> None:  # noqa: ANN001
    def raw_bypass(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("raw urllib bypass: transport must use safe_outbound_http")

    monkeypatch.setattr(module.urllib.request, "urlopen", raw_bypass)


def test_method_classification_is_read_vs_mutation() -> None:
    assert classify_operation("GET") == "read"
    assert classify_operation("head") == "read"
    for method in ("PUT", "POST", "DELETE", "PATCH"):
        assert classify_operation(method) == "mutation"


def test_json_request_routes_through_boundary_with_pinned_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_boundary_fake(monkeypatch, body=b'{"ok": true}')
    status, payload = provider_json_request(
        url="https://api.example.test/v1/instances",
        method="GET",
        headers={"Authorization": "Bearer k"},
        timeout_seconds=17,
    )
    assert status == 200
    assert payload == {"ok": True}
    (call,) = calls
    assert call["timeout"] == 17
    policy = call["policy"]
    assert policy.allowed_hosts == frozenset({"api.example.test"})
    assert policy.follow_same_origin_redirects is True
    assert policy.max_response_bytes == DEFAULT_PROVIDER_MAX_RESPONSE_BYTES
    assert call["request"].headers.get("Authorization") == "Bearer k"


def test_plain_http_is_refused_before_any_network(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_boundary_fake(monkeypatch)
    with pytest.raises(safe_outbound_http.SafeOutboundHttpError):
        provider_json_request(
            url="http://api.example.test/v1/instances",
            method="GET",
            headers={},
            timeout_seconds=5,
        )
    assert calls == []


def test_empty_body_returns_empty_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_boundary_fake(monkeypatch, body=b"   ")
    status, payload = provider_json_request(
        url="https://api.example.test/v1/x",
        method="GET",
        headers={},
        timeout_seconds=5,
    )
    assert status == 200
    assert payload == {}


def test_http_error_propagates_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(request, timeout, policy):  # noqa: ANN001
        raise urllib.error.HTTPError(
            request.full_url, 404, "not found", hdrs=None, fp=io.BytesIO(b"{}")
        )

    monkeypatch.setattr(safe_outbound_http, "_open_with_policy", fake)
    with pytest.raises(urllib.error.HTTPError):
        provider_json_request(
            url="https://api.example.test/v1/missing",
            method="GET",
            headers={},
            timeout_seconds=5,
        )


def test_mutations_refuse_read_retry_policies(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_boundary_fake(monkeypatch)
    retry = bounded_read_retry(
        operation="misapplied",
        exception_allowlist=(urllib.error.URLError,),
        max_attempts=3,
        max_delay_seconds=5.0,
        evidence_hook=lambda row: None,
        sleep=lambda _s: None,
    )
    with pytest.raises(MutationRetryForbidden):
        provider_json_request(
            url="https://api.example.test/v1/instances",
            method="PUT",
            headers={},
            timeout_seconds=5,
            read_retry=retry,
        )
    assert calls == []


def test_reads_may_use_bounded_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_boundary_fake(
        monkeypatch, body=b'{"ok": true}', fail_first=urllib.error.URLError
    )
    retry = bounded_read_retry(
        operation="vast_list",
        exception_allowlist=(urllib.error.URLError,),
        max_attempts=3,
        max_delay_seconds=5.0,
        evidence_hook=lambda row: None,
        sleep=lambda _s: None,
    )
    status, payload = provider_json_request(
        url="https://api.example.test/v1/instances",
        method="GET",
        headers={},
        timeout_seconds=5,
        read_retry=retry,
    )
    assert status == 200
    assert payload == {"ok": True}
    assert len(calls) == 2


def test_text_request_routes_through_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_boundary_fake(monkeypatch, body=b"plain text body")
    text = provider_text_request(
        url="https://downloads.example.test/notes.txt", timeout_seconds=9
    )
    assert text == "plain text body"
    (call,) = calls
    assert call["policy"].allowed_hosts == frozenset({"downloads.example.test"})


def test_vast_api_json_routes_through_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import vast_provider_adapter as vpa

    _forbid_raw_urlopen(monkeypatch, vpa)
    calls = _install_boundary_fake(monkeypatch, body=b'{"instances": []}')
    status, payload = vpa._api_json(
        method="GET", path="/instances/", api_key="vast-key"
    )
    assert status == 200
    assert payload == {"instances": []}
    (call,) = calls
    assert call["request"].full_url.startswith("https://console.vast.ai/api/v0")
    assert call["request"].headers.get("Authorization") == "Bearer vast-key"
    assert call["policy"].allowed_hosts == frozenset({"console.vast.ai"})


def test_vast_fetch_text_routes_through_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import vast_provider_adapter as vpa

    _forbid_raw_urlopen(monkeypatch, vpa)
    _install_boundary_fake(monkeypatch, body=b"template body")
    assert vpa._fetch_text("https://console.vast.ai/api/v0/thing") == "template body"


def test_lambda_http_json_routes_through_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import lambda_provider_adapter as lpa

    _forbid_raw_urlopen(monkeypatch, lpa)
    calls = _install_boundary_fake(monkeypatch, body=b'{"data": {"instances": []}}')
    status, payload = lpa._http_json(
        url="https://cloud.lambda.ai/api/v1/instances",
        payload=None,
        api_key="lambda-key",
        timeout_seconds=11,
        method="GET",
    )
    assert status == 200
    assert payload == {"data": {"instances": []}}
    (call,) = calls
    assert call["request"].headers.get("Authorization") == "Bearer lambda-key"
    assert call["policy"].allowed_hosts == frozenset({"cloud.lambda.ai"})
