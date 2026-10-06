"""Stream admitted immutable artifacts through anonymous HTTPS with one deadline.

Byte identity, size and consumption authority stay with each caller. This module
never forwards authorization headers or URL credentials to redirect targets.
"""
from __future__ import annotations

import http.client
import math
import time
import urllib.error
import urllib.parse
import urllib.request

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

    def connect(self):
        remaining = self._deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('artifact_transfer_deadline')
        # urllib reuses the initial timeout across redirect requests. Clamp
        # each fresh connection to the remaining budget before TCP/TLS setup.
        self.timeout = min(self.timeout,remaining)
        super().connect()
        self.sock = _DeadlineSocket(self.sock,self._deadline,self.timeout)

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
