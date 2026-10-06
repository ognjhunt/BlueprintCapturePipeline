"""Bound fixed control-plane JSON requests without redirecting credentials."""
from __future__ import annotations

import json
import http.client
import re
import ssl
import time
from typing import Any
import urllib.parse
import urllib.request
import urllib.error


_METADATA_PATH = "/computeMetadata/v1/instance/service-accounts/default/token"
_METADATA_ORIGINS = {"http://metadata.google.internal", "http://169.254.169.254"}


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def http_error_302(self, req, fp, code, msg, headers):
        try:
            fp.close()
        finally:
            raise ValueError("controlled_http_redirect_refused")

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


class _DeadlineSocket:
    """Clamp every raw receive, including status/header/chunk line refills."""

    def __init__(self, socket, deadline: float):
        self._socket, self._deadline = socket, deadline

    def __getattr__(self, name):
        return getattr(self._socket, name)

    def _remaining(self):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("controlled_http_response_timeout")
        return remaining

    def recv_into(self, target, *args):
        self._socket.settimeout(self._remaining())
        result = self._socket.recv_into(target, *args)
        self._remaining()
        return result

    def sendall(self, data, *args):
        self._socket.settimeout(self._remaining())
        self._socket.sendall(data, *args)
        self._remaining()

    def makefile(self, *args, **kwargs):
        stream = self._socket.makefile(*args, **kwargs)
        raw = getattr(stream, "raw", None)
        if raw is None or getattr(raw, "_sock", None) is not self._socket:
            stream.close()
            raise ValueError("controlled_http_response_reader_invalid")
        raw._sock = self
        return stream


class _DeadlineHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args, deadline, **kwargs):
        self._deadline = deadline
        super().__init__(*args, **kwargs)

    def connect(self):
        super().connect()
        self.sock = _DeadlineSocket(self.sock, self._deadline)


class _DeadlineHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, deadline, **kwargs):
        self._deadline = deadline
        super().__init__(*args, **kwargs)

    def connect(self):
        super().connect()
        self.sock = _DeadlineSocket(self.sock, self._deadline)


class _DeadlineHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline

    def http_open(self, request):
        return self.do_open(lambda *args, **kwargs: _DeadlineHTTPConnection(
            *args, deadline=self.deadline, **kwargs), request)


class _DeadlineHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, deadline, context):
        super().__init__(context=context)
        self.deadline = deadline

    def https_open(self, request):
        return self.do_open(lambda *args, **kwargs: _DeadlineHTTPSConnection(
            *args, deadline=self.deadline, **kwargs), request, context=self._context)


class _ClosingHTTPErrorProcessor(urllib.request.HTTPErrorProcessor):
    def http_response(self, request, response):
        try:
            return super().http_response(request, response)
        except urllib.error.HTTPError as error:
            error.close()
            raise

    https_response = http_response


def fixed_response_opener(*, deadline: float, context=None, direct=False):
    handlers = [_NoRedirects(), _DeadlineHTTPHandler(deadline),
                _DeadlineHTTPSHandler(deadline, context), _ClosingHTTPErrorProcessor()]
    if direct:
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def read_bounded_response(response, *, deadline: float, maximum_bytes: int) -> bytes:
    """Read available HTTP fragments without renewing the overall deadline.

    CPython's HTTPResponse.read(n) can wait for all n bytes while a slow peer
    continually renews the socket timeout. read1 performs at most one raw read;
    its socket timeout is clamped to the remaining budget on every iteration.
    An unsupported response reader fails closed instead of losing that bound.
    """
    raw = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("controlled_http_response_timeout")
        if response.isclosed():
            break
        socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        if socket is None or not callable(getattr(socket, "settimeout", None)):
            raise ValueError("controlled_http_response_reader_invalid")
        socket.settimeout(remaining)
        reader = getattr(response, "read1", None)
        if not callable(reader):
            raise ValueError("controlled_http_response_reader_invalid")
        chunk = reader(min(64 * 1024, maximum_bytes + 1 - len(raw)))
        if time.monotonic() >= deadline:
            raise TimeoutError("controlled_http_response_timeout")
        if not chunk:
            break
        raw.extend(chunk)
        if len(raw) > maximum_bytes:
            raise ValueError("controlled_http_response_oversized")
    return bytes(raw)


def read_control_json(
    request: urllib.request.Request, *, origin: str, method: str,
    timeout: int, maximum_bytes: int, context: ssl.SSLContext | None = None,
    direct: bool = False,
) -> dict[str, Any]:
    """Read one bounded response from an explicitly selected origin and method.

    HTTP is only permitted for the two fixed Google metadata endpoints. Their
    requests always bypass proxies and require the metadata response header.
    The loopback bridge also passes ``direct`` to keep its pinned TLS peer local.
    This function never follows redirects or retries an uncertain mutation.
    """
    parsed = urllib.parse.urlsplit(request.full_url)
    expected = urllib.parse.urlsplit(origin)
    if (parsed.scheme not in {"http", "https"} or parsed.scheme != expected.scheme
            or parsed.netloc != expected.netloc or parsed.username is not None
            or parsed.password is not None or parsed.fragment or expected.path
            or not expected.hostname or expected.query or expected.fragment
            or expected.username is not None or expected.password is not None
            or request.get_method() != method
            or type(timeout) is not int or not 0 < timeout <= 120
            or type(maximum_bytes) is not int or not 0 < maximum_bytes <= 16 * 1024 * 1024):
        raise ValueError("controlled_http_request_invalid")
    metadata = parsed.scheme == "http"
    if metadata and (origin not in _METADATA_ORIGINS or parsed.path != _METADATA_PATH
                     or parsed.query or method != "GET"
                     or request.get_header("Metadata-flavor") != "Google"):
        raise ValueError("controlled_http_request_invalid")
    deadline = time.monotonic() + timeout
    opener = fixed_response_opener(deadline=deadline, context=context, direct=direct or metadata)
    with opener.open(request, timeout=timeout) as response:
        if (response.geturl() != request.full_url or not 200 <= response.getcode() < 300
                or (metadata and response.headers.get("Metadata-Flavor") != "Google")):
            raise ValueError("controlled_http_response_invalid")
        declared = response.headers.get("Content-Length")
        if declared is not None:
            if not re.fullmatch(r"[0-9]{1,10}", declared):
                raise ValueError("controlled_http_response_invalid")
            if int(declared) > maximum_bytes:
                raise ValueError("controlled_http_response_oversized")
        raw = read_bounded_response(response, deadline=deadline, maximum_bytes=maximum_bytes)
        if declared is not None and len(raw) != int(declared):
            raise ValueError("controlled_http_response_truncated")
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise ValueError("controlled_http_response_invalid") from None
        if not isinstance(value, dict):
            raise ValueError("controlled_http_response_invalid")
        return value
