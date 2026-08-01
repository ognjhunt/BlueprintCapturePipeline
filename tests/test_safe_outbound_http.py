"""Contract tests for the centralized safe outbound HTTP boundary (FABLE-007).

Every urllib call site in the pipeline routes through
``blueprint_pipeline.safe_outbound_http`` so scheme, host, redirect, timeout,
and response-size policy are enforced at one audited transport site.
"""

from __future__ import annotations

import urllib.error
import urllib.request
import hashlib
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from blueprint_pipeline import safe_outbound_http as soh


class _FakeResponse:
    """Stands in for http.client.HTTPResponse behind a monkeypatched urlopen."""

    def __init__(
        self,
        *,
        body: bytes = b"{}",
        status: int = 200,
        final_url: str | None = None,
        supports_amt: bool = True,
        has_geturl: bool = True,
    ) -> None:
        self._body = body
        self.status = status
        self._final_url = final_url
        self._supports_amt = supports_amt
        if not has_geturl:
            # Mimic minimal test doubles used elsewhere in the suite.
            self.geturl = None  # type: ignore[assignment]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, amt=None):
        if amt is None:
            return self._body
        if not self._supports_amt:
            raise TypeError("read() takes no arguments")
        return self._body[:amt]

    def geturl(self):  # noqa: F811 - deliberately shadowed when has_geturl=False
        return self._final_url


class _Transport:
    """Records the Request/timeout handed to the single urlopen site."""

    def __init__(self, response: _FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[urllib.request.Request, float]] = []
        self._explicit_final_url = (
            isinstance(response, _FakeResponse) and response._final_url is not None
        )

    def __call__(
        self,
        request: urllib.request.Request,
        timeout: float,
        policy: soh.OutboundHttpPolicy,
    ):
        del policy
        self.calls.append((request, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        if not self._explicit_final_url:
            self.response._final_url = request.full_url
        return self.response


class _StreamingResponse:
    def __init__(self, body: bytes, *, status: int = 200, final_url: str) -> None:
        self._body = body
        self._offset = 0
        self.status = status
        self._final_url = final_url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self._body) - self._offset
        chunk = self._body[self._offset : self._offset + amount]
        self._offset += len(chunk)
        return chunk

    def geturl(self) -> str:
        return self._final_url


def test_https_request_to_pinned_api_host_passes(monkeypatch) -> None:
    transport = _Transport(_FakeResponse(body=b'{"ok": true}', status=201))
    monkeypatch.setattr(soh, "_open_with_policy", transport)
    policy = soh.pinned_api_policy("https://rest.runpod.io/v1")

    response = soh.request(
        "https://rest.runpod.io/v1/pods",
        method="POST",
        data=b"{}",
        headers={"Authorization": "Bearer k"},
        timeout_seconds=90,
        policy=policy,
    )

    assert response.status == 201
    assert response.body == b'{"ok": true}'
    assert len(transport.calls) == 1
    sent, timeout = transport.calls[0]
    assert sent.full_url == "https://rest.runpod.io/v1/pods"
    assert sent.get_method() == "POST"
    assert sent.headers["Authorization"] == "Bearer k"
    assert timeout == 90.0


@pytest.mark.parametrize(
    "url",
    [
        "http://rest.runpod.io/v1/pods",
        "file:///etc/passwd",
        "ftp://rest.runpod.io/v1/pods",
        "gopher://rest.runpod.io/",
    ],
)
def test_non_https_schemes_are_rejected_before_any_network_io(monkeypatch, url) -> None:
    transport = _Transport(_FakeResponse())
    monkeypatch.setattr(soh, "_open_with_policy", transport)
    policy = soh.pinned_api_policy("https://rest.runpod.io/v1")

    with pytest.raises(soh.SafeOutboundHttpError, match="outbound_http_scheme_not_allowed"):
        soh.request(url, policy=policy)
    assert transport.calls == []


def test_host_outside_pinned_allowlist_is_rejected(monkeypatch) -> None:
    transport = _Transport(_FakeResponse())
    monkeypatch.setattr(soh, "_open_with_policy", transport)
    policy = soh.pinned_api_policy("https://api.digitalocean.com/v2")

    with pytest.raises(soh.SafeOutboundHttpError, match="outbound_http_host_not_allowed"):
        soh.request("https://attacker.example/v2/droplets", policy=policy)
    assert transport.calls == []


def test_loopback_policy_allows_http_only_for_loopback_hosts(monkeypatch) -> None:
    transport = _Transport(_FakeResponse(body=b"{}"))
    monkeypatch.setattr(soh, "_open_with_policy", transport)
    policy = soh.loopback_service_policy()

    ok = soh.request("http://127.0.0.1:8765/apply-and-measure", policy=policy)
    assert ok.status == 200
    ok_localhost = soh.request("http://localhost:8765/apply-and-measure", policy=policy)
    assert ok_localhost.status == 200
    with pytest.raises(soh.SafeOutboundHttpError, match="outbound_http_host_not_allowed"):
        soh.request("https://scorer.example/v1/score", policy=policy)

    with pytest.raises(soh.SafeOutboundHttpError, match="outbound_http_scheme_not_allowed"):
        soh.request("http://10.0.0.5:8765/apply-and-measure", policy=policy)


def test_configured_service_policy_pins_https_origin(monkeypatch) -> None:
    transport = _Transport(_FakeResponse(body=b"{}"))
    monkeypatch.setattr(soh, "_open_with_policy", transport)
    policy = soh.service_endpoint_policy("https://scorer.example/v1/score")

    assert soh.request("https://scorer.example/v1/score", policy=policy).status == 200
    with pytest.raises(soh.SafeOutboundHttpError, match="outbound_http_host_not_allowed"):
        soh.request("https://attacker.example/v1/score", policy=policy)


def test_credentials_embedded_in_url_are_rejected(monkeypatch) -> None:
    transport = _Transport(_FakeResponse())
    monkeypatch.setattr(soh, "_open_with_policy", transport)
    policy = soh.pinned_api_policy("https://rest.runpod.io/v1")

    with pytest.raises(
        soh.SafeOutboundHttpError, match="outbound_http_credentials_in_url_blocked"
    ):
        soh.request("https://user:secret@rest.runpod.io/v1/pods", policy=policy)
    assert transport.calls == []


def test_cross_host_redirect_escape_is_blocked(monkeypatch) -> None:
    response = _FakeResponse(body=b"{}", final_url="https://attacker.example/steal")
    monkeypatch.setattr(soh, "_open_with_policy", _Transport(response))
    policy = soh.pinned_api_policy("https://rest.runpod.io/v1")

    with pytest.raises(
        soh.SafeOutboundHttpError, match="outbound_http_redirect_escape_blocked"
    ):
        soh.request("https://rest.runpod.io/v1/pods", policy=policy)


def test_real_redirect_is_blocked_before_target_receives_headers() -> None:
    received: list[dict[str, str | None]] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            received.append(
                {
                    "authorization": self.headers.get("Authorization"),
                    "x_secret": self.headers.get("X-Secret"),
                }
            )
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            return

    class RedirectHandler(BaseHTTPRequestHandler):
        target_port = 0

        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header(
                "Location", f"http://127.0.0.1:{self.target_port}/credential-sink"
            )
            self.end_headers()

        def log_message(self, *_args):
            return

    target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    RedirectHandler.target_port = target.server_port
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (target, redirect)
    ]
    for thread in threads:
        thread.start()
    try:
        policy = soh.OutboundHttpPolicy(
            allowed_hosts=frozenset({"127.0.0.1"}), allow_loopback_http=True
        )
        with pytest.raises(
            soh.SafeOutboundHttpError, match="outbound_http_redirect_escape_blocked"
        ):
            soh.request(
                f"http://127.0.0.1:{redirect.server_port}/start",
                headers={"Authorization": "Bearer secret", "X-Secret": "secret"},
                policy=policy,
            )
        assert received == []
    finally:
        for server in (redirect, target):
            server.shutdown()
            server.server_close()


def test_cross_scheme_redirect_downgrade_is_blocked(monkeypatch) -> None:
    response = _FakeResponse(body=b"{}", final_url="http://rest.runpod.io/v1/pods")
    monkeypatch.setattr(soh, "_open_with_policy", _Transport(response))
    policy = soh.pinned_api_policy("https://rest.runpod.io/v1")

    with pytest.raises(
        soh.SafeOutboundHttpError, match="outbound_http_redirect_escape_blocked"
    ):
        soh.request("https://rest.runpod.io/v1/pods", policy=policy)


def test_same_origin_redirect_blocked_by_default_and_gated_by_policy(monkeypatch) -> None:
    blocked = _FakeResponse(body=b"{}", final_url="https://rest.runpod.io/v1/other")
    monkeypatch.setattr(soh, "_open_with_policy", _Transport(blocked))
    policy = soh.pinned_api_policy("https://rest.runpod.io/v1")
    with pytest.raises(soh.SafeOutboundHttpError, match="outbound_http_redirect_blocked"):
        soh.request("https://rest.runpod.io/v1/pods", policy=policy)

    allowed = _FakeResponse(body=b"{}", final_url="https://rest.runpod.io/v1/other")
    monkeypatch.setattr(soh, "_open_with_policy", _Transport(allowed))
    lenient = soh.OutboundHttpPolicy(
        allowed_hosts=frozenset({"rest.runpod.io"}),
        follow_same_origin_redirects=True,
    )
    response = soh.request("https://rest.runpod.io/v1/pods", policy=lenient)
    assert response.status == 200


def test_oversized_response_body_fails_closed(monkeypatch) -> None:
    response = _FakeResponse(body=b"x" * 33)
    monkeypatch.setattr(soh, "_open_with_policy", _Transport(response))
    policy = soh.OutboundHttpPolicy(
        allowed_hosts=frozenset({"rest.runpod.io"}), max_response_bytes=32
    )

    with pytest.raises(soh.SafeOutboundHttpError, match="outbound_http_response_too_large"):
        soh.request("https://rest.runpod.io/v1/pods", policy=policy)


def test_presigned_put_upload_is_supported_and_host_pinned(monkeypatch) -> None:
    transport = _Transport(_FakeResponse(status=200))
    monkeypatch.setattr(soh, "_open_with_policy", transport)
    put_url = "https://bucket.nyc3.digitaloceanspaces.com/out.zip?X-Amz-Signature=abc"
    policy = soh.presigned_transfer_policy(put_url)

    response = soh.request(
        put_url,
        method="PUT",
        data=b"zip-bytes",
        headers={"Content-Type": "application/zip"},
        timeout_seconds=180,
        policy=policy,
    )
    assert response.status == 200
    sent, timeout = transport.calls[0]
    assert sent.get_method() == "PUT"
    assert sent.data == b"zip-bytes"
    assert timeout == 180.0

    # The presigned policy pins the URL's own host: any other host is rejected.
    with pytest.raises(soh.SafeOutboundHttpError, match="outbound_http_host_not_allowed"):
        soh.request("https://other-bucket.example/out.zip", policy=policy)


def test_presigned_transfer_policy_rejects_non_https_presigned_url() -> None:
    with pytest.raises(soh.SafeOutboundHttpError, match="outbound_http_scheme_not_allowed"):
        soh.presigned_transfer_policy("http://bucket.example/out.zip")


def test_invalid_timeout_is_rejected(monkeypatch) -> None:
    transport = _Transport(_FakeResponse())
    monkeypatch.setattr(soh, "_open_with_policy", transport)
    policy = soh.pinned_api_policy("https://rest.runpod.io/v1")

    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(soh.SafeOutboundHttpError, match="outbound_http_timeout_invalid"):
            soh.request(
                "https://rest.runpod.io/v1/pods", policy=policy, timeout_seconds=bad
            )
    assert transport.calls == []


def test_http_error_propagates_to_callers(monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "https://rest.runpod.io/v1/pods", 500, "boom", hdrs=None, fp=None
    )
    monkeypatch.setattr(soh, "_open_with_policy", _Transport(error))
    policy = soh.pinned_api_policy("https://rest.runpod.io/v1")

    with pytest.raises(urllib.error.HTTPError):
        soh.request("https://rest.runpod.io/v1/pods", policy=policy)


def test_open_request_validates_prebuilt_request_objects(monkeypatch) -> None:
    transport = _Transport(_FakeResponse(body=b'{"status": "ok"}'))
    monkeypatch.setattr(soh, "_open_with_policy", transport)
    policy = soh.loopback_service_policy()

    prebuilt = urllib.request.Request(
        "http://127.0.0.1:8765/apply-and-measure",
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    response = soh.open_request(prebuilt, policy=policy, timeout_seconds=120.0)
    assert response.body == b'{"status": "ok"}'
    assert transport.calls[0][0] is prebuilt

    forbidden = urllib.request.Request("http://10.9.8.7/apply-and-measure", method="POST")
    with pytest.raises(soh.SafeOutboundHttpError, match="outbound_http_scheme_not_allowed"):
        soh.open_request(forbidden, policy=policy, timeout_seconds=120.0)


def test_transports_without_sized_read_or_geturl_still_work(monkeypatch) -> None:
    """Minimal test doubles (plain read(), no geturl) stay usable behind the boundary."""
    response = _FakeResponse(body=b'{"ok": 1}', supports_amt=False, has_geturl=False)
    monkeypatch.setattr(soh, "_open_with_policy", _Transport(response))
    policy = soh.loopback_service_policy()

    result = soh.request("http://127.0.0.1:8765/apply-and-measure", policy=policy)
    assert result.body == b'{"ok": 1}'
    assert result.status == 200


def test_digest_bound_file_download_is_streamed_atomic_and_secret_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"large-operation-output" * 1000
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    url = "https://objects.example/output.zip?X-Amz-Signature=do-not-record"
    response = _StreamingResponse(payload, final_url=url)
    monkeypatch.setattr(soh, "_open_with_policy", _Transport(response))
    destination = tmp_path / "retrieved.zip"
    receipt = soh.download_file(
        url,
        output_path=destination,
        expected_sha256=digest,
        max_bytes=len(payload),
        timeout_seconds=300,
        policy=soh.presigned_transfer_policy(url),
    )
    assert destination.read_bytes() == payload
    assert receipt.sha256 == digest
    assert receipt.transferred_bytes == len(payload)
    assert receipt.host == "objects.example"
    assert "do-not-record" not in repr(receipt)


def test_file_download_mismatch_or_oversize_preserves_previous_complete_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "https://objects.example/output.zip?signature=secret"
    destination = tmp_path / "retrieved.zip"
    destination.write_bytes(b"previous")
    wrong = "sha256:" + "0" * 64
    monkeypatch.setattr(
        soh,
        "_open_with_policy",
        _Transport(_StreamingResponse(b"new", final_url=url)),
    )
    with pytest.raises(soh.SafeOutboundHttpError, match="file_digest_mismatch"):
        soh.download_file(
            url,
            output_path=destination,
            expected_sha256=wrong,
            max_bytes=10,
            timeout_seconds=30,
            policy=soh.presigned_transfer_policy(url),
        )
    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".retrieved.zip.*.partial"))

    monkeypatch.setattr(
        soh,
        "_open_with_policy",
        _Transport(_StreamingResponse(b"too-large", final_url=url)),
    )
    with pytest.raises(soh.SafeOutboundHttpError, match="file_too_large"):
        soh.download_file(
            url,
            output_path=destination,
            expected_sha256=wrong,
            max_bytes=3,
            timeout_seconds=30,
            policy=soh.presigned_transfer_policy(url),
        )
    assert destination.read_bytes() == b"previous"


def test_observed_output_download_hashes_without_pretending_prior_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"new-provider-output"
    url = "https://objects.example/output.zip?signature=secret"
    monkeypatch.setattr(
        soh,
        "_open_with_policy",
        _Transport(_StreamingResponse(payload, final_url=url)),
    )
    destination = tmp_path / "output.zip"
    receipt = soh.download_file_observed(
        url,
        output_path=destination,
        max_bytes=1024,
        timeout_seconds=30,
        policy=soh.presigned_transfer_policy(url),
    )
    assert destination.read_bytes() == payload
    assert receipt.sha256 == "sha256:" + hashlib.sha256(payload).hexdigest()


def test_digest_bound_file_upload_streams_body_and_pins_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"checkpoint-and-splat" * 1000
    source = tmp_path / "output.zip"
    source.write_bytes(payload)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    url = "https://objects.example/output.zip?X-Amz-Signature=do-not-record"
    captured: dict[str, object] = {}

    def transport(request, timeout, policy):
        del policy
        captured["timeout"] = timeout
        captured["content_length"] = request.get_header("Content-length")
        captured["body_is_bytes"] = isinstance(request.data, bytes)
        body = bytearray()
        while True:
            chunk = request.data.read(4096)
            if not chunk:
                break
            body.extend(chunk)
        captured["body"] = bytes(body)
        return _StreamingResponse(b"", status=201, final_url=url)

    monkeypatch.setattr(soh, "_open_with_policy", transport)
    receipt = soh.upload_file(
        url,
        input_path=source,
        expected_sha256=digest,
        max_bytes=len(payload),
        timeout_seconds=300,
        policy=soh.presigned_transfer_policy(url, max_response_bytes=1024),
        content_type="application/zip",
    )
    assert captured["body"] == payload
    assert captured["body_is_bytes"] is False
    assert captured["content_length"] == str(len(payload))
    assert receipt.status == 201
    assert receipt.sha256 == digest


def test_blocked_presigned_redirect_does_not_expose_query_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_url = "https://objects.example/in.zip?signature=source-secret"
    final_url = "https://objects.example/other?signature=redirect-secret"
    monkeypatch.setattr(
        soh,
        "_open_with_policy",
        _Transport(_FakeResponse(final_url=final_url)),
    )
    with pytest.raises(soh.SafeOutboundHttpError) as raised:
        soh.request(
            source_url,
            policy=soh.presigned_transfer_policy(source_url),
        )
    message = str(raised.value)
    assert "outbound_http_redirect_blocked" in message
    assert "source-secret" not in message
    assert "redirect-secret" not in message
