"""Hermetic tests for fixed control origins, credentials, and bounded reads."""
from __future__ import annotations

import io
import http.client
import json
import socket
from types import SimpleNamespace
import urllib.request

import pytest

from blueprint_pipeline import controlled_http_json as transport
from blueprint_pipeline import policy_model_materialization as models
from blueprint_pipeline import company_policy_sandbox_manager as manager


ORIGIN = "https://compute.googleapis.com"
URL = ORIGIN + "/compute/v1/projects/test/global/firewalls/test"
METADATA = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"


class Response(io.BytesIO):
    def __init__(self, raw=b'{"ok":true}', *, url=URL, headers=None, status=200):
        super().__init__(raw)
        self.url, self.headers, self.status = url, headers or {}, status
        self.read_sizes = []
        self.timeouts = []
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=self.timeouts.append)))

    def geturl(self):
        return self.url

    def getcode(self):
        return self.status

    def isclosed(self):
        return self.closed

    def read1(self, count=-1):
        assert count > 0
        self.read_sizes.append(count)
        return super().read(min(count, 3))


def opener(monkeypatch, response, *, redirect=None):
    observed = {"calls": [], "handlers": []}

    def build(*handlers):
        observed["handlers"] = list(handlers)

        def send(request, *, timeout):
            observed["calls"].append((request, timeout))
            if redirect:
                handler = next(item for item in handlers if isinstance(item, transport._NoRedirects))
                handler.parent = SimpleNamespace(open=lambda *_a, **_k: pytest.fail("redirect followed"))
                code, destination = redirect
                return getattr(handler, f"http_error_{code}")(request, response, code, "redirect",
                                                               {"location": destination})
            return response

        return SimpleNamespace(open=send)

    monkeypatch.setattr(urllib.request, "build_opener", build)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: pytest.fail("unrestricted opener"))
    return observed


def read(request=None, **kwargs):
    values = {"origin": ORIGIN, "method": "GET", "timeout": 10, "maximum_bytes": 64}
    values.update(kwargs)
    return transport.read_control_json(request or urllib.request.Request(URL), **values)


def test_bounded_fragmented_object_and_response_close(monkeypatch):
    response = Response(headers={"Content-Length": "11"})
    observed = opener(monkeypatch, response)
    assert read() == {"ok": True}
    assert response.closed
    assert all(0 < size <= 65 for size in response.read_sizes)
    assert len(observed["calls"]) == 1


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("destination", [ORIGIN + "/other", "https://example.invalid/", "file:///tmp/probe"])
def test_redirect_never_forwards_credential_or_reads_response(monkeypatch, code, destination):
    response = Response()
    observed = opener(monkeypatch, response, redirect=(code, destination))
    request = urllib.request.Request(URL)
    request.add_unredirected_header("Authorization", "Bearer synthetic")
    with pytest.raises(ValueError, match="redirect_refused"):
        read(request)
    assert len(observed["calls"]) == 1
    assert response.closed and response.read_sizes == []


@pytest.mark.parametrize("url", ["file:///tmp/probe", "ftp://compute.googleapis.com/test",
                                "http://compute.googleapis.com/test", "https://example.invalid/",
                                "https://user@compute.googleapis.com/test", URL + "#fragment"])
def test_bad_origins_refuse_before_opener(monkeypatch, url):
    observed = opener(monkeypatch, Response())
    with pytest.raises(ValueError, match="request_invalid"):
        read(urllib.request.Request(url))
    assert observed["calls"] == [] and observed["handlers"] == []


@pytest.mark.parametrize("body,headers,reason", [
    (b"x" * 66, {}, "oversized"), (b"{}", {"Content-Length": "100"}, "oversized"),
    (b"{}", {"Content-Length": "10"}, "truncated"),
    (b"{}", {"Content-Length": "-1"}, "invalid"), (b"[]", {}, "invalid"),
    (b"bad-json", {}, "invalid"),
])
def test_bounded_or_invalid_json_refuses(monkeypatch, body, headers, reason):
    response = Response(body, headers=headers)
    opener(monkeypatch, response)
    with pytest.raises(ValueError, match=reason):
        read()
    assert response.closed


def test_metadata_uses_only_fixed_endpoint_and_bypasses_ambient_proxy(monkeypatch):
    body = json.dumps({"access_token": "synthetic-token"}).encode()
    response = Response(body, url=METADATA, headers={"Metadata-Flavor": "Google"})
    observed = opener(monkeypatch, response)
    request = urllib.request.Request(METADATA, headers={"Metadata-Flavor": "Google"})
    assert read(request, origin="http://metadata.google.internal") == {"access_token": "synthetic-token"}
    proxies = [handler for handler in observed["handlers"] if isinstance(handler, urllib.request.ProxyHandler)]
    assert len(proxies) == 1 and proxies[0].proxies == {}


@pytest.mark.parametrize("url", [METADATA + "?other", METADATA.replace("/token", "/identity"),
                                METADATA.replace("metadata.google.internal", "example.invalid")])
def test_metadata_endpoint_cannot_be_repurposed(monkeypatch, url):
    observed = opener(monkeypatch, Response())
    with pytest.raises(ValueError, match="request_invalid"):
        read(urllib.request.Request(url, headers={"Metadata-Flavor": "Google"}),
             origin="http://metadata.google.internal")
    assert observed["calls"] == []


def test_metadata_refuses_missing_response_binding(monkeypatch):
    response = Response(url=METADATA)
    opener(monkeypatch, response)
    with pytest.raises(ValueError, match="response_invalid"):
        read(urllib.request.Request(METADATA, headers={"Metadata-Flavor": "Google"}),
             origin="http://metadata.google.internal")
    assert response.read_sizes == []


def test_read_does_not_renew_deadline_between_fragments(monkeypatch):
    response = Response()
    observed = opener(monkeypatch, response)
    ticks = iter([0, 0, 1, 11])
    monkeypatch.setattr(transport.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="response_timeout"):
        read()
    assert len(observed["calls"]) == 1 and response.closed


def test_real_http_buffered_reader_observes_decreasing_socket_budget(monkeypatch):
    clock = [0.0]

    class SlowRaw(io.RawIOBase):
        def __init__(self):
            self._sock = self
            self.timeouts = []
            self.raw_reads = 0

        def readable(self):
            return True

        def settimeout(self, value):
            self.timeouts.append(value)

        def readinto(self, target):
            self.raw_reads += 1
            # A deterministic peer would need four seconds per fragment. The
            # real remaining socket timeout prevents its third fragment.
            if self.timeouts[-1] < 4:
                clock[0] += self.timeouts[-1]
                raise TimeoutError("synthetic remaining socket budget")
            clock[0] += 4
            target[0] = ord("x")
            return 1

    source = SlowRaw()
    response = http.client.HTTPResponse.__new__(http.client.HTTPResponse)
    response.fp = io.BufferedReader(source)
    response.chunked, response.length, response._method = False, None, "GET"
    monkeypatch.setattr(transport.time, "monotonic", lambda: clock[0])
    with pytest.raises(TimeoutError, match="remaining socket budget"):
        transport.read_bounded_response(response, deadline=10, maximum_bytes=64)
    assert source.raw_reads == 3 and source.timeouts == [10, 6, 2]
    assert clock[0] == 10
    response.close()


def test_unsupported_response_reader_refuses_without_unbounded_fallback(monkeypatch):
    response = Response()
    response.fp = SimpleNamespace(raw=SimpleNamespace())
    opener(monkeypatch, response)
    with pytest.raises(ValueError, match="reader_invalid"):
        read()
    assert response.read_sizes == [] and response.closed


@pytest.mark.parametrize("prefix,suffix", [
    (b"", b"HTTP/1.1 200 OK\r\n\r\n"),
    (b"HTTP/1.1 200 OK\r\n", b"X-Test: slow\r\n\r\n"),
    (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n", b"1;slow=metadata\r\nx\r\n0\r\n\r\n"),
    (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n", b"X-Trailer: slow\r\n\r\n"),
])
def test_actual_http_parser_bounds_dripping_headers_and_chunk_metadata(monkeypatch, prefix, suffix):
    clock = [0.0]

    class SyntheticSocket:
        def __init__(self):
            self.prefix, self.suffix = prefix, bytearray(suffix)
            self.timeout, self.timeouts = 0, []

        def settimeout(self, value):
            self.timeout = value
            self.timeouts.append(value)

        def recv_into(self, target):
            if self.prefix:
                count = len(self.prefix)
                target[:count] = self.prefix
                self.prefix = b""
                return count
            if self.timeout < 4:
                clock[0] += self.timeout
                raise TimeoutError("synthetic parser remaining budget")
            clock[0] += 4
            target[0] = self.suffix.pop(0)
            return 1

        def makefile(self, *_args, **_kwargs):
            return io.BufferedReader(socket.SocketIO(self, "r"))

        def _decref_socketios(self):
            pass

    peer = SyntheticSocket()
    monkeypatch.setattr(transport.time, "monotonic", lambda: clock[0])
    response = http.client.HTTPResponse(transport._DeadlineSocket(peer, 10))
    try:
        with pytest.raises(TimeoutError, match="parser remaining budget"):
            response.begin()
            transport.read_bounded_response(response, deadline=10, maximum_bytes=64)
        assert clock[0] == 10
        assert peer.timeouts[-3:] == [10, 6, 2]
    finally:
        response.close()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
def test_http_error_processor_closes_body_without_retry(status):
    response = Response(status=status)
    response.code, response.msg = status, "synthetic refusal"
    response.info = lambda: response.headers
    calls = []

    def fail(*args):
        calls.append(args)
        raise urllib.error.HTTPError(URL, status, "synthetic refusal", {}, response)

    processor = transport._ClosingHTTPErrorProcessor()
    processor.parent = SimpleNamespace(error=fail)
    with pytest.raises(urllib.error.HTTPError) as result:
        processor.https_response(urllib.request.Request(URL), response)
    assert result.value.code == status and response.closed
    assert len(calls) == 1 and response.read_sizes == []


def test_compute_request_uses_original_method_and_unredirected_bearer(monkeypatch):
    observed = opener(monkeypatch, Response())
    monkeypatch.setattr(manager, "_metadata_token", lambda: "synthetic")
    assert manager.FirewallLease._request("PATCH", URL, {"sourceRanges": ["192.0.2.1/32"]}) == {"ok": True}
    request, timeout = observed["calls"][0]
    assert request.get_method() == "PATCH" and timeout == 30
    assert request.unredirected_hdrs == {"Authorization": "Bearer synthetic"}
    assert "Authorization" not in request.headers
    assert json.loads(request.data) == {"sourceRanges": ["192.0.2.1/32"]}


def test_model_download_uses_bound_generation_and_refuses_redirect(monkeypatch):
    url = "https://storage.googleapis.com/storage/v1/b/test/o/object?alt=media&generation=7"
    response = Response(url=url)
    observed = opener(monkeypatch, response, redirect=(302, "https://example.invalid/"))
    monkeypatch.setattr(models, "_service_account_token", lambda: "synthetic")
    with pytest.raises(ValueError, match="redirect_refused"):
        models._download({"storage_generation": "7"}, bucket="test", object_name="object")
    request, timeout = observed["calls"][0]
    assert request.full_url == url and timeout == 45
    assert request.unredirected_hdrs == {"Authorization": "Bearer synthetic"}
    assert len(observed["calls"]) == 1 and response.read_sizes == []
