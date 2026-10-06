"""Stream admitted immutable artifacts through anonymous HTTPS with one deadline.

Byte identity, size and consumption authority stay with each caller. This module
never forwards authorization headers or URL credentials to redirect targets.
"""
from __future__ import annotations

import functools
import json
import os
import selectors
import socket
import subprocess
import sys
import http.client
import math
import time
import urllib.error
import urllib.parse
import urllib.request

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


def _artifact_remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("artifact_transfer_deadline")
    return remaining


def _resolve_addresses(address, *, deadline):
    """Bound libc resolution without accumulating uncancellable threads.

    Socket timeouts do not cover getaddrinfo. The isolated child receives only
    the hostname and port, and is killed and reaped on timeout or interruption.
    It never receives the URL, headers, credentials, or request body.
    """
    _artifact_remaining(deadline)
    payload = json.dumps(address).encode("utf-8")
    if len(payload) > 4096:
        raise ValueError("controlled_http_resolution_oversized")
    _artifact_remaining(deadline)
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
                events = selector.select(_artifact_remaining(deadline))
                if not events:
                    raise TimeoutError("artifact_transfer_deadline")
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
                _artifact_remaining(deadline)
        try:
            process.wait(timeout=_artifact_remaining(deadline))
        except subprocess.TimeoutExpired:
            raise TimeoutError("artifact_transfer_deadline") from None
        _artifact_remaining(deadline)
        if process.returncode != 0:
            raise OSError("controlled_http_resolution_failed")
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise ValueError("controlled_http_resolution_invalid") from None
        _artifact_remaining(deadline)
        addresses = _validated_resolver_result(result)
        _artifact_remaining(deadline)
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


def _create_deadline_connection(address, _timeout=None, source_address=None, *, deadline, maximum_timeout):
    addresses = _resolve_addresses(address, deadline=deadline)
    sources = (_resolve_addresses(source_address, deadline=deadline)
               if source_address else None)
    last_error = None
    for family, kind, protocol, _canonical, sockaddr in addresses:
        _artifact_remaining(deadline)
        peer = socket.socket(family, kind, protocol)
        try:
            peer.settimeout(min(_artifact_remaining(deadline), maximum_timeout))
            if sources:
                source = next((item[4] for item in sources if item[0] == family), None)
                if source is None:
                    raise OSError("controlled_http_source_address_unavailable")
                peer.bind(source)
            peer.connect(sockaddr)
            _artifact_remaining(deadline)
            # CONNECT status/headers must share the budget before TLS starts.
            return _DeadlineSocket(peer, deadline, maximum_timeout)
        except OSError as error:
            last_error = error
            peer.close()
        except BaseException:
            peer.close()
            raise
    _artifact_remaining(deadline)
    if last_error is not None:
        raise last_error
    raise OSError("getaddrinfo returns an empty list")


class _DeadlineSocket:
    """Clamp every raw receive, including status/header/chunk line refills."""

    def __init__(self, socket, deadline: float, maximum_timeout: float):
        self._socket, self._deadline, self._maximum_timeout = socket, deadline, maximum_timeout

    def __getattr__(self, name):
        return getattr(self._socket, name)

    def _remaining(self):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("controlled_http_response_timeout")
        return min(remaining,self._maximum_timeout)

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

class _DeadlineHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, deadline, **kwargs):
        self._deadline = deadline
        super().__init__(*args, **kwargs)
        self._create_connection = functools.partial(
            _create_deadline_connection, deadline=deadline, maximum_timeout=self.timeout)

    def connect(self):
        # DNS, every address attempt and TLS consume the same absolute budget.
        # Keep the caller's per-operation timeout cap throughout setup and reads.
        self.timeout = min(self.timeout, _artifact_remaining(self._deadline))
        try:
            http.client.HTTPConnection.connect(self)
            peer = self.sock._socket
            peer.settimeout(min(self.timeout, _artifact_remaining(self._deadline)))
            hostname = self._tunnel_host or self.host
            self.sock = self._context.wrap_socket(peer, server_hostname=hostname,
                                                 do_handshake_on_connect=False)
            self.sock.settimeout(min(self.timeout, _artifact_remaining(self._deadline)))
            self.sock.do_handshake()
            _artifact_remaining(self._deadline)
            self.sock = _DeadlineSocket(self.sock, self._deadline, self.timeout)
        except BaseException:
            self.close()
            raise

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

def _artifact_https_url(url):
    if not isinstance(url,str) or len(url) > 65536 or any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise ValueError('artifact_https_url_invalid')
    try:
        parsed = urllib.parse.urlsplit(url)
        admitted = (parsed.scheme == 'https' and bool(parsed.hostname) and not parsed.username
                    and not parsed.password and parsed.port in {None,443} and not parsed.fragment)
    except ValueError:
        admitted = False
    if not admitted:
        raise ValueError('artifact_https_url_invalid')
    return url


class _AnonymousArtifactRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,response,code,message,headers,url):
        if request.get_method() != 'GET' or request.data is not None:
            raise ValueError('artifact_redirect_method_refused')
        _artifact_https_url(url)
        # Location is fresh anonymous HTTPS authority. Never copy source
        # headers, body, URL userinfo or the original query onto that target.
        return urllib.request.Request(url,headers={'Accept':'application/octet-stream'})


    def http_error_302(self,request,response,code,message,headers):
        try:
            location = headers.get('location') or headers.get('uri')
            if not isinstance(location,str) or len(location) > 65536:
                raise ValueError('artifact_redirect_location_invalid')
            target = urllib.parse.urljoin(request.full_url,location)
            redirected = self.redirect_request(request,response,code,message,headers,target)
            visited = getattr(request,'redirect_dict',{})
            if visited.get(target,0) >= self.max_repeats or len(visited) >= self.max_redirections:
                raise ValueError('artifact_redirect_limit')
            visited[target] = visited.get(target,0)+1
            redirected.redirect_dict = visited
        finally:
            # urllib's default handler drains an unbounded redirect body with
            # read(). A fresh GET needs no old body, so close without reading.
            response.close()
        return self.parent.open(redirected,timeout=request.timeout)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def artifact_deadline(maximum_seconds,*,deadline=None):
    now = time.monotonic()
    if (type(maximum_seconds) not in {int,float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0
            or deadline is not None and (type(deadline) not in {int,float} or not math.isfinite(deadline))):
        raise ValueError('artifact_deadline_invalid')
    result = min(now+maximum_seconds,deadline) if deadline is not None else now+maximum_seconds
    if result <= now:
        raise TimeoutError('artifact_transfer_deadline')
    return result


def open_artifact_response(url,*,deadline,socket_timeout,headers=None):
    _artifact_https_url(url)
    if (type(deadline) not in {int,float} or not math.isfinite(deadline)
            or type(socket_timeout) not in {int,float} or not math.isfinite(socket_timeout) or socket_timeout <= 0):
        raise ValueError('artifact_deadline_invalid')
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError('artifact_transfer_deadline')
    headers = {} if headers is None else dict(headers)
    if any(not isinstance(key,str) or key.lower() not in {'accept','user-agent'} or not isinstance(value,str) or len(value) > 4096
           or any(ord(c) < 32 or ord(c) == 127 for c in value) for key,value in headers.items()):
        raise ValueError('artifact_headers_invalid')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
        _DeadlineHTTPSHandler(deadline,None),_AnonymousArtifactRedirect(),_ClosingHTTPErrorProcessor())
    response = opener.open(urllib.request.Request(url,headers=headers),timeout=min(socket_timeout,remaining))
    try:
        if not 200 <= response.status < 300:
            raise ValueError('artifact_response_status_invalid')
        _artifact_https_url(response.url)
        if time.monotonic() >= deadline:
            raise TimeoutError('artifact_transfer_deadline')
    except BaseException:
        response.close()
        raise
    return response


def artifact_chunks(response,*,deadline,maximum_bytes,chunk_bytes=1024**2):
    if (type(maximum_bytes) is not int or maximum_bytes < 0 or type(chunk_bytes) is not int or chunk_bytes <= 0
            or type(deadline) not in {int,float} or not math.isfinite(deadline)):
        raise ValueError('artifact_stream_bound_invalid')
    count = 0
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError('artifact_transfer_deadline')
        # read1 makes at most one raw body read; the admitted socket also clamps
        # each status/header/chunk framing receive to this same deadline.
        chunk = response.read1(min(chunk_bytes,maximum_bytes+1-count))
        if time.monotonic() >= deadline:
            raise TimeoutError('artifact_transfer_deadline')
        if not chunk:
            break
        if len(chunk)+count > maximum_bytes:
            raise ValueError('artifact_stream_oversized')
        count += len(chunk)
        yield chunk
