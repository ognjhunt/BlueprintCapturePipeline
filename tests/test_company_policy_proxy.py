from __future__ import annotations

import json
from email.message import Message
from typing import Any

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import company_policy_proxy as proxy


class _Response:
    def __init__(self, payload: Any, *, content_type: str = "application/json"):
        self.body = json.dumps(payload).encode("utf-8")
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, limit: int) -> bytes:
        return self.body[:limit]


class _Opener:
    def __init__(self, response: _Response):
        self.response = response
        self.requests = []

    def open(self, request, *, timeout: float):
        self.requests.append((request, timeout))
        return self.response


def test_proxy_forwards_only_json_to_fixed_loopback_route(monkeypatch) -> None:
    opener = _Opener(_Response({"actions": [[0.0, 1.0]]}))
    monkeypatch.setattr(proxy.urllib.request, "build_opener", lambda *_args: opener)

    result = proxy.forward_action_json(
        payload={"observation": {"images": {}, "state": []}},
        upstream_port=8600,
        timeout_ms=2500,
        max_response_bytes=4096,
    )

    assert result == {"actions": [[0.0, 1.0]]}
    request, timeout = opener.requests[0]
    assert request.full_url == "http://127.0.0.1:8600/v1/actions"
    assert request.method == "POST"
    assert timeout == 2.5
    assert set(request.headers) == {"Content-type", "Accept"}
    assert b"authorization" not in request.data.lower()


def test_proxy_refuses_secret_carriers_and_non_object_responses(monkeypatch) -> None:
    with pytest.raises(ValueError, match="secret_carrier_forbidden"):
        proxy.forward_action_json(
            payload={"registry_secret": "do-not-forward"},
            upstream_port=8600,
            timeout_ms=100,
            max_response_bytes=4096,
        )
    opener = _Opener(_Response([1, 2, 3]))
    monkeypatch.setattr(proxy.urllib.request, "build_opener", lambda *_args: opener)
    with pytest.raises(ValueError, match="response_not_object"):
        proxy.forward_action_json(
            payload={"observation": {}},
            upstream_port=8600,
            timeout_ms=100,
            max_response_bytes=4096,
        )


def test_http_surface_is_bounded_no_store_and_has_no_docs(monkeypatch) -> None:
    opener = _Opener(_Response({"actions": [[0.0]]}))
    monkeypatch.setattr(proxy.urllib.request, "build_opener", lambda *_args: opener)
    monkeypatch.setenv(proxy.UPSTREAM_PORT_ENV, "8600")
    client = TestClient(proxy.create_app())

    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    health = client.get("/health")
    assert health.json()["external_network_required"] is False
    response = client.post("/v1/actions", json={"observation": {}})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"actions": [[0.0]]}


def test_invalid_upstream_port_and_oversize_body_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv(proxy.UPSTREAM_PORT_ENV, "80")
    monkeypatch.setenv(proxy.MAX_REQUEST_BYTES_ENV, "1024")
    client = TestClient(proxy.create_app())
    invalid = client.post("/v1/actions", json={"observation": {}})
    assert invalid.status_code == 502
    assert "upstream_port_invalid" in invalid.text

    monkeypatch.setenv(proxy.UPSTREAM_PORT_ENV, "8600")
    oversized = client.post(
        "/v1/actions",
        content=json.dumps({"observation": "x" * 2048}),
        headers={"content-type": "application/json"},
    )
    assert oversized.status_code == 413
