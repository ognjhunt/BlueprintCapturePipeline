"""Bound fixed control-plane JSON requests without redirecting credentials."""
from __future__ import annotations

import functools
import http.client
import json
import os
import re
import selectors
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_METADATA_PATH = "/computeMetadata/v1/instance/service-accounts/default/token"
_METADATA_ORIGINS = {"http://metadata.google.internal", "http://169.254.169.254"}
_MAX_RESOLVER_BYTES = 64 * 1024
_MAX_RESOLVER_ADDRESSES = 256


_RESOLVER_SCRIPT = """
import json, socket, sys
host, port = json.loads(sys.stdin.buffer.read().decode("utf-8"))
try:
    addresses = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    result = ({"addresses": addresses} if len(addresses) <= 256 else
              {"refusal": "controlled_http_resolution_oversized"})
except socket.gaierror as error:
    result = {"error": [error.errno, error.strerror]}
sys.stdout.buffer.write(json.dumps(result).encode("utf-8"))
"""


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("controlled_http_response_timeout")
    return remaining


def _resolve_addresses(address, *, deadline):
    """Bound libc resolution without accumulating uncancellable threads.

    Socket timeouts do not cover getaddrinfo. The isolated child receives only
    the hostname and port, and is killed and reaped on timeout or interruption.
    It never receives the URL, headers, credentials, or request body.
    """
    _remaining(deadline)
    payload = json.dumps(address).encode("utf-8")
    if len(payload) > 4096:
        raise ValueError("controlled_http_resolution_oversized")
    _remaining(deadline)
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", _RESOLVER_SCRIPT],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env={},
    )
    try:
        raw = bytearray()
        written = 0
        with selectors.DefaultSelector() as selector:
            os.set_blocking(process.stdin.fileno(), False)
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE)
            selector.register(process.stdout, selectors.EVENT_READ)
            while not process.stdout.closed:
                events = selector.select(_remaining(deadline))
                if not events:
                    raise TimeoutError("controlled_http_response_timeout")
                for key, _event in events:
                    if key.fileobj is process.stdin:
                        written += os.write(process.stdin.fileno(), payload[written:written + 512])
                        if written == len(payload):
                            selector.unregister(process.stdin)
                            process.stdin.close()
                    else:
                        chunk = os.read(process.stdout.fileno(),
                                        min(8192, _MAX_RESOLVER_BYTES + 1 - len(raw)))
                        if not chunk:
                            selector.unregister(process.stdout)
                            process.stdout.close()
                        else:
                            raw.extend(chunk)
                            if len(raw) > _MAX_RESOLVER_BYTES:
                                raise ValueError("controlled_http_resolution_oversized")
                _remaining(deadline)
        try:
            process.wait(timeout=_remaining(deadline))
        except subprocess.TimeoutExpired:
            raise TimeoutError("controlled_http_response_timeout") from None
        _remaining(deadline)
        if process.returncode != 0:
            raise OSError("controlled_http_resolution_failed")
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise ValueError("controlled_http_resolution_invalid") from None
        _remaining(deadline)
        addresses = _validated_resolver_result(result)
        _remaining(deadline)
        return addresses
    finally:
        if process.poll() is None:
            process.kill()
        process.stdin.close()
        process.stdout.close()
        process.wait()


def _validated_resolver_result(result):
    if type(result) is not dict:
        raise ValueError("controlled_http_resolution_invalid")
    if set(result) == {"refusal"} and result["refusal"] == "controlled_http_resolution_oversized":
        raise ValueError("controlled_http_resolution_oversized")
    if set(result) == {"error"}:
        error = result["error"]
        if (type(error) is list and len(error) == 2
                and type(error[0]) is int and type(error[1]) is str):
            raise socket.gaierror(*error)
        raise ValueError("controlled_http_resolution_invalid")
    rows = result.get("addresses")
    if set(result) != {"addresses"} or type(rows) is not list:
        raise ValueError("controlled_http_resolution_invalid")
    if len(rows) > _MAX_RESOLVER_ADDRESSES:
        raise ValueError("controlled_http_resolution_oversized")
    addresses = []
    for row in rows:
        if type(row) is not list or len(row) != 5:
            raise ValueError("controlled_http_resolution_invalid")
        family, kind, protocol, canonical, sockaddr = row
        if (type(family) is not int or family not in {socket.AF_INET, socket.AF_INET6}
                or type(kind) is not int or kind != socket.SOCK_STREAM
                or type(protocol) is not int or protocol not in {0, socket.IPPROTO_TCP}
                or type(canonical) is not str or type(sockaddr) is not list
                or len(sockaddr) != (2 if family == socket.AF_INET else 4)
                or type(sockaddr[0]) is not str or type(sockaddr[1]) is not int
                or not 0 <= sockaddr[1] <= 65535
                or any(type(value) is not int or not 0 <= value <= 0xffffffff
                       for value in sockaddr[2:])):
            raise ValueError("controlled_http_resolution_invalid")
        try:
            socket.inet_pton(family, sockaddr[0])
        except (OSError, ValueError):
            raise ValueError("controlled_http_resolution_invalid") from None
        addresses.append((family, kind, protocol, canonical, tuple(sockaddr)))
    return addresses


def _create_deadline_connection(address, _timeout=None, source_address=None, *, deadline):
    addresses = _resolve_addresses(address, deadline=deadline)
    sources = (_resolve_addresses(source_address, deadline=deadline)
               if source_address else None)
    last_error = None
    for family, kind, protocol, _canonical, sockaddr in addresses:
        _remaining(deadline)
        peer = socket.socket(family, kind, protocol)
        try:
            peer.settimeout(_remaining(deadline))
            if sources:
                source = next((item[4] for item in sources if item[0] == family), None)
                if source is None:
                    raise OSError("controlled_http_source_address_unavailable")
                peer.bind(source)
            peer.connect(sockaddr)
            _remaining(deadline)
            # CONNECT status/headers must share the budget before TLS starts.
            return _DeadlineSocket(peer, deadline)
        except OSError as error:
            last_error = error
            peer.close()
        except BaseException:
            peer.close()
            raise
    _remaining(deadline)
    if last_error is not None:
        raise last_error
    raise OSError("getaddrinfo returns an empty list")


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
        return _remaining(self._deadline)

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
        self._create_connection = functools.partial(_create_deadline_connection, deadline=deadline)

    def connect(self):
        try:
            super().connect()
        except BaseException:
            self.close()
            raise


class _DeadlineHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, deadline, **kwargs):
        self._deadline = deadline
        super().__init__(*args, **kwargs)
        self._create_connection = functools.partial(_create_deadline_connection, deadline=deadline)

    def connect(self):
        try:
            http.client.HTTPConnection.connect(self)
            peer = self.sock._socket
            peer.settimeout(_remaining(self._deadline))
            hostname = self._tunnel_host or self.host
            self.sock = self._context.wrap_socket(peer, server_hostname=hostname,
                                                 do_handshake_on_connect=False)
            self.sock.settimeout(_remaining(self._deadline))
            self.sock.do_handshake()
            _remaining(self._deadline)
            self.sock = _DeadlineSocket(self.sock, self._deadline)
        except BaseException:
            self.close()
            raise


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
