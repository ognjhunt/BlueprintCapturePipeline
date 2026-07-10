from __future__ import annotations

import io
import json
from http import HTTPStatus
from types import MethodType

from blueprint_pipeline import privacy_runner_service, video_to_world_runner_service


class _Headers(dict[str, str]):
    def get(self, key: str, default: str | None = None) -> str | None:
        return super().get(key, default)


def _handler(handler_cls, *, path: str, body: object = None, headers: dict[str, str] | None = None):
    handler = object.__new__(handler_cls)
    raw = b"" if body is None else json.dumps(body).encode("utf-8")
    response: dict[str, object] = {"headers": []}
    handler.path = path
    handler.headers = _Headers(headers or {})
    if body is not None and "Content-Length" not in handler.headers:
        handler.headers["Content-Length"] = str(len(raw))
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()

    def send_response(self, status: int) -> None:  # type: ignore[no-untyped-def]
        response["status"] = status

    def send_header(self, key: str, value: str) -> None:  # type: ignore[no-untyped-def]
        response["headers"].append((key, value))

    def end_headers(self) -> None:  # type: ignore[no-untyped-def]
        response["ended"] = True

    handler.send_response = MethodType(send_response, handler)
    handler.send_header = MethodType(send_header, handler)
    handler.end_headers = MethodType(end_headers, handler)
    return handler, response


def _payload(handler) -> dict[str, object]:
    return json.loads(handler.wfile.getvalue().decode("utf-8"))


def test_privacy_runner_service_auth_get_post_and_main(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("PRIVACY_RUNNER_KIND", "SAM3")
    monkeypatch.delenv("PRIVACY_RUNNER_TOKEN", raising=False)
    assert privacy_runner_service._service_kind() == "sam3"
    assert privacy_runner_service._auth_token() == ""

    handler, response = _handler(privacy_runner_service._Handler, path="/healthz")
    handler.do_GET()
    assert response["status"] == HTTPStatus.OK
    assert _payload(handler) == {"status": "ok", "runner_kind": "sam3"}

    handler, response = _handler(privacy_runner_service._Handler, path="/missing")
    handler.do_GET()
    assert response["status"] == HTTPStatus.NOT_FOUND

    monkeypatch.setattr(
        privacy_runner_service,
        "execute_privacy_service_request",
        lambda kind, body: {"status": "succeeded", "kind": kind, "body": body},
    )
    handler, response = _handler(privacy_runner_service._Handler, path="/run", body={"open": True})
    handler.do_POST()
    assert response["status"] == HTTPStatus.UNAUTHORIZED

    monkeypatch.setenv("PRIVACY_RUNNER_TOKEN", "secret")
    handler, response = _handler(
        privacy_runner_service._Handler,
        path="/canary",
        headers={"Authorization": "Bearer secret"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.OK
    assert _payload(handler) == {
        "status": "ok",
        "authentication": "verified",
        "runner_kind": "sam3",
        "model_execution_performed": False,
    }

    handler, response = _handler(
        privacy_runner_service._Handler,
        path="/canary",
        body={"forbidden": True},
        headers={"Authorization": "Bearer secret"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.BAD_REQUEST
    assert _payload(handler)["reason"] == "canary_body_forbidden"

    handler, response = _handler(privacy_runner_service._Handler, path="/run", body={})
    handler.do_POST()
    assert response["status"] == HTTPStatus.UNAUTHORIZED

    handler, response = _handler(
        privacy_runner_service._Handler,
        path="/run",
        body=None,
        headers={"Authorization": "Bearer secret", "Content-Length": "bad"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.BAD_REQUEST
    assert _payload(handler)["reason"] == "invalid_content_length"

    invalid_raw = b"{bad-json"
    handler, response = _handler(
        privacy_runner_service._Handler,
        path="/run",
        body=None,
        headers={"Authorization": "Bearer secret", "Content-Length": str(len(invalid_raw))},
    )
    handler.rfile = io.BytesIO(invalid_raw)
    handler.do_POST()
    assert response["status"] == HTTPStatus.BAD_REQUEST
    assert _payload(handler)["reason"] == "invalid_json"

    handler, response = _handler(
        privacy_runner_service._Handler,
        path="/run",
        body=["not", "mapping"],
        headers={"Authorization": "Bearer secret"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.BAD_REQUEST
    assert _payload(handler)["reason"] == "invalid_payload"

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_execute(kind: str, body: dict[str, object]) -> dict[str, object]:
        calls.append((kind, body))
        return {"status": "succeeded", "artifact": "ok"}

    monkeypatch.setattr(privacy_runner_service, "execute_privacy_service_request", fake_execute)
    handler, response = _handler(
        privacy_runner_service._Handler,
        path="/",
        body={"input": "video"},
        headers={"Authorization": "Bearer secret"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.OK
    assert calls == [("sam3", {"input": "video"})]

    monkeypatch.setattr(
        privacy_runner_service,
        "execute_privacy_service_request",
        lambda _kind, _body: {"status": "failed"},
    )
    handler, response = _handler(
        privacy_runner_service._Handler,
        path="/run",
        body={},
        headers={"Authorization": "Bearer secret"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.BAD_GATEWAY

    handler, response = _handler(
        privacy_runner_service._Handler,
        path="/elsewhere",
        body={},
        headers={"Authorization": "Bearer secret"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.NOT_FOUND

    started: dict[str, object] = {}

    class FakeServer:
        def __init__(self, address, handler_cls) -> None:  # type: ignore[no-untyped-def]
            started["address"] = address
            started["handler_cls"] = handler_cls

        def serve_forever(self) -> None:
            started["served"] = True

    monkeypatch.setenv("PORT", "not-an-int")
    monkeypatch.setattr(privacy_runner_service, "ThreadingHTTPServer", FakeServer)
    assert privacy_runner_service.main() == 0
    assert started == {
        "address": ("127.0.0.1", 8080),
        "handler_cls": privacy_runner_service._Handler,
        "served": True,
    }


def test_video_to_world_runner_service_auth_get_post_and_main(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("VIDEO_TO_WORLD_RUNNER_TOKEN", raising=False)
    monkeypatch.delenv("PRIVACY_RUNNER_TOKEN", raising=False)
    assert video_to_world_runner_service._auth_token() == ""

    monkeypatch.setattr(
        video_to_world_runner_service,
        "execute_video_to_world_request",
        lambda body: {"status": "succeeded", "body": body},
    )
    handler, response = _handler(video_to_world_runner_service._Handler, path="/run", body={"open": True})
    handler.do_POST()
    assert response["status"] == HTTPStatus.UNAUTHORIZED

    monkeypatch.setenv("PRIVACY_RUNNER_TOKEN", "fallback")
    assert video_to_world_runner_service._auth_token() == "fallback"

    handler, response = _handler(
        video_to_world_runner_service._Handler,
        path="/canary",
        headers={"Authorization": "Bearer fallback"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.OK
    assert _payload(handler) == {
        "status": "ok",
        "authentication": "verified",
        "runner": "video_to_world",
        "model_execution_performed": False,
    }

    handler, response = _handler(video_to_world_runner_service._Handler, path="/")
    handler.do_GET()
    assert response["status"] == HTTPStatus.OK
    assert _payload(handler) == {"status": "ok", "runner": "video_to_world"}

    handler, response = _handler(video_to_world_runner_service._Handler, path="/missing")
    handler.do_GET()
    assert response["status"] == HTTPStatus.NOT_FOUND

    handler, response = _handler(video_to_world_runner_service._Handler, path="/run", body={})
    handler.do_POST()
    assert response["status"] == HTTPStatus.UNAUTHORIZED

    handler, response = _handler(
        video_to_world_runner_service._Handler,
        path="/run",
        body=None,
        headers={"Authorization": "Bearer fallback", "Content-Length": "bad"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.BAD_REQUEST
    assert _payload(handler)["reason"] == "invalid_content_length"

    invalid_raw = b"{bad-json"
    handler, response = _handler(
        video_to_world_runner_service._Handler,
        path="/run",
        body=None,
        headers={"Authorization": "Bearer fallback", "Content-Length": str(len(invalid_raw))},
    )
    handler.rfile = io.BytesIO(invalid_raw)
    handler.do_POST()
    assert response["status"] == HTTPStatus.BAD_REQUEST
    assert _payload(handler)["reason"] == "invalid_json"

    handler, response = _handler(
        video_to_world_runner_service._Handler,
        path="/run",
        body=["not", "mapping"],
        headers={"Authorization": "Bearer fallback"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.BAD_REQUEST
    assert _payload(handler)["reason"] == "invalid_payload"

    monkeypatch.setattr(
        video_to_world_runner_service,
        "execute_video_to_world_request",
        lambda body: {"status": "succeeded", "body": body},
    )
    handler, response = _handler(
        video_to_world_runner_service._Handler,
        path="/",
        body={"input": "video"},
        headers={"Authorization": "Bearer fallback"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.OK
    assert _payload(handler)["body"] == {"input": "video"}

    monkeypatch.setattr(
        video_to_world_runner_service,
        "execute_video_to_world_request",
        lambda _body: {"status": "failed"},
    )
    handler, response = _handler(
        video_to_world_runner_service._Handler,
        path="/run",
        body={},
        headers={"Authorization": "Bearer fallback"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.BAD_GATEWAY

    handler, response = _handler(
        video_to_world_runner_service._Handler,
        path="/elsewhere",
        body={},
        headers={"Authorization": "Bearer fallback"},
    )
    handler.do_POST()
    assert response["status"] == HTTPStatus.NOT_FOUND

    started: dict[str, object] = {}

    class FakeServer:
        def __init__(self, address, handler_cls) -> None:  # type: ignore[no-untyped-def]
            started["address"] = address
            started["handler_cls"] = handler_cls

        def serve_forever(self) -> None:
            started["served"] = True

    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setattr(video_to_world_runner_service, "ThreadingHTTPServer", FakeServer)
    assert video_to_world_runner_service.main() == 0
    assert started == {
        "address": ("127.0.0.1", 9090),
        "handler_cls": video_to_world_runner_service._Handler,
        "served": True,
    }

    started.clear()
    monkeypatch.setenv("PORT", "not-an-int")
    assert video_to_world_runner_service.main() == 0
    assert started["address"] == ("127.0.0.1", 8080)
