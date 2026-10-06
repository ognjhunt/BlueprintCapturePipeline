"""ADP-050 Day28: offline bytes for the observed G1 VM system gaps.

Preparation and APT simulation never install packages, grant owner review,
allocate a provider, qualify CUDA, or enable paired policy dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time
import urllib.request


# Standalone artifact transport: definitions match artifact_http_transport.py.
import functools
import selectors
import socket
import sys
import http.client
import math
import urllib.error
import urllib.parse

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


IMAGE_REF = (
    "docker.io/vastai/kvm@sha256:28dc36f977d4a078ee410caf08f595d91f95185a00e0d4e7970c2d11f7358738"
)
GUEST_DISK_SHA = "sha256:5720729e5adfd15645e3c2035f54855a39b479728dca49d2ec405e7dd9e338de"
KERNEL = "5.15.0-1067-kvm"
BASE_PACKAGES = {
    "docker.io": "24.0.7-0ubuntu2~22.04.1",
    "linux-headers-5.15.0-1067-kvm": "5.15.0-1067.72",
    "build-essential": "12.9ubuntu3",
    "gcc-12": "12.3.0-1ubuntu1~22.04",
    "dkms": "2.8.7-2ubuntu2.2",
}
SYSTEM_PACKAGES = {
    "bubblewrap": {
        "package": "bubblewrap",
        "version": "0.6.1-1ubuntu0.3",
        "filename": "bubblewrap_0.6.1-1ubuntu0.3_amd64.deb",
        "size_bytes": 46314,
        "sha256": "sha256:706763714e5d19fc7ab2572a6418506e3a4a5d5960e99309707b58117e6b40cb",
        "url": "https://security.ubuntu.com/ubuntu/pool/main/b/bubblewrap/bubblewrap_0.6.1-1ubuntu0.3_amd64.deb",
    },
    "dkms": {
        "package": "dkms",
        "version": "1:3.2.1-1ubuntu2",
        "filename": "dkms_3.2.1-1ubuntu2_all.deb",
        "size_bytes": 53454,
        "sha256": "sha256:586d2c54880158da00ba387d1c224b3fd71986c44f36e9d4a7bd6d58c74030ec",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/dkms_3.2.1-1ubuntu2_all.deb",
    },
    "libnvidia-cfg1-580": {
        "package": "libnvidia-cfg1-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "libnvidia-cfg1-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 147274,
        "sha256": "sha256:7b84f78e2c33a88e44286308600fc9687acb417ca402553281bfb471f874c1b7",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-cfg1-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "libnvidia-common-580": {
        "package": "libnvidia-common-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "libnvidia-common-580_580.65.06-0ubuntu1_all.deb",
        "size_bytes": 16772,
        "sha256": "sha256:97026962f839f3b01de2dd8652c795f9836e2ba820b97d37677acd698b8ba875",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-common-580_580.65.06-0ubuntu1_all.deb",
    },
    "libnvidia-compute-580": {
        "package": "libnvidia-compute-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "libnvidia-compute-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 54080704,
        "sha256": "sha256:cab9ae0d40eb3960a2e2dd3894cd862f57dd4ee61205ed696e3568631194ba2d",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-compute-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "libnvidia-container-tools": {
        "package": "libnvidia-container-tools",
        "version": "1.19.0-1",
        "filename": "libnvidia-container-tools_1.19.0-1_amd64.deb",
        "size_bytes": 22148,
        "sha256": "sha256:7a345f3538683d587cd5a77ab5a580ee50f9a3132caed6687822d532f1fa6809",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-container-tools_1.19.0-1_amd64.deb",
    },
    "libnvidia-container1": {
        "package": "libnvidia-container1",
        "version": "1.19.0-1",
        "filename": "libnvidia-container1_1.19.0-1_amd64.deb",
        "size_bytes": 1192828,
        "sha256": "sha256:3bfd3726eb5430ace3b25db9a8830493c73a15b5050e71c2dbc666fc67ff7c34",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-container1_1.19.0-1_amd64.deb",
    },
    "libnvidia-decode-580": {
        "package": "libnvidia-decode-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "libnvidia-decode-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 2721862,
        "sha256": "sha256:3a461424e4f2ea220a2fd2b957028945cf8dc9246a2f52b2e218cdccfbd99604",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-decode-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "libnvidia-encode-580": {
        "package": "libnvidia-encode-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "libnvidia-encode-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 106674,
        "sha256": "sha256:74401ddb16527b5dff92077f268215ba08513e1148513b0aad6dfb79c33d8c58",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-encode-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "libnvidia-extra-580": {
        "package": "libnvidia-extra-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "libnvidia-extra-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 73112,
        "sha256": "sha256:720f13a4887d650b9783b12cfe14df80420179c83822f3d7717d5e580ba93102",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-extra-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "libnvidia-fbc1-580": {
        "package": "libnvidia-fbc1-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "libnvidia-fbc1-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 86860,
        "sha256": "sha256:f6edfa9fd31d7230f75fde8192ed5135e7903a1e8f6e107b9585fb1c99ec7d7e",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-fbc1-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "libnvidia-gl-580": {
        "package": "libnvidia-gl-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "libnvidia-gl-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 144104086,
        "sha256": "sha256:4d2632803dd46712e3421dd7aa9ed75b9b8b71085f799a7f360277d846fecac2",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-gl-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "libnvidia-gpucomp-580": {
        "package": "libnvidia-gpucomp-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "libnvidia-gpucomp-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 18483628,
        "sha256": "sha256:e72f28a0a95947ece76fcee03a659287ffe4bcb1f50107e330c67868811bcf46",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-gpucomp-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "nvidia-compute-utils-580": {
        "package": "nvidia-compute-utils-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "nvidia-compute-utils-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 44616,
        "sha256": "sha256:ed47fc6d7568f43e17dab6ed6e933820a87ef5dd5e62a69a6607f959f9d04d4b",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-compute-utils-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "nvidia-container-toolkit": {
        "package": "nvidia-container-toolkit",
        "version": "1.19.0-1",
        "filename": "nvidia-container-toolkit_1.19.0-1_amd64.deb",
        "size_bytes": 1333956,
        "sha256": "sha256:1b944df11c78fde8c75755e594c7c64ea1c8a8e8a01b0b99e247017bf12439ce",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-container-toolkit_1.19.0-1_amd64.deb",
    },
    "nvidia-container-toolkit-base": {
        "package": "nvidia-container-toolkit-base",
        "version": "1.19.0-1",
        "filename": "nvidia-container-toolkit-base_1.19.0-1_amd64.deb",
        "size_bytes": 5577536,
        "sha256": "sha256:471d6cb264c84b0b8b7023f39f0a8b6fca50b91eb70a8614b689e83b64b1c2d6",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-container-toolkit-base_1.19.0-1_amd64.deb",
    },
    "nvidia-dkms-580": {
        "package": "nvidia-dkms-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "nvidia-dkms-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 14746,
        "sha256": "sha256:f900a618860a8d4b46d7116d9628cca69613b49a944577116ca611e268ee8bbf",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-dkms-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "nvidia-driver-580": {
        "package": "nvidia-driver-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "nvidia-driver-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 496730,
        "sha256": "sha256:f6d066660225ad891b977395f2ad7da821a8d6beb211d22819641ce9c01bc4d1",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-driver-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "nvidia-firmware-580": {
        "package": "nvidia-firmware-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "nvidia-firmware-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 74553404,
        "sha256": "sha256:7ea3c67ad08f586da975d21321eef8b6700b19e0b585a5f33d25c96163c31fa5",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-firmware-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "nvidia-kernel-common-580": {
        "package": "nvidia-kernel-common-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "nvidia-kernel-common-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 700094,
        "sha256": "sha256:8fd19ed7f5fe089ebd70c1166f98ac2723d18b61ab95120fa9bb5aab12fd471c",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-kernel-common-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "nvidia-kernel-source-580": {
        "package": "nvidia-kernel-source-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "nvidia-kernel-source-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 82945374,
        "sha256": "sha256:26c57be2c9af09ac3f0d1785ae2a494a37daffa139c755f397015b321d1f172a",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-kernel-source-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "nvidia-modprobe": {
        "package": "nvidia-modprobe",
        "version": "580.65.06-0ubuntu1",
        "filename": "nvidia-modprobe_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 14958,
        "sha256": "sha256:0cba137a81adbe79f432e30ec0685265f2875fb7fee29dad3519384638788614",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-modprobe_580.65.06-0ubuntu1_amd64.deb",
    },
    "nvidia-persistenced": {
        "package": "nvidia-persistenced",
        "version": "580.65.06-0ubuntu1",
        "filename": "nvidia-persistenced_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 82356,
        "sha256": "sha256:77c95d97f90d0943ec6ef6b871e882e6b6c78154914187d6cf8cd12320ed324b",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-persistenced_580.65.06-0ubuntu1_amd64.deb",
    },
    "nvidia-utils-580": {
        "package": "nvidia-utils-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "nvidia-utils-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 480434,
        "sha256": "sha256:3e9c39d206196e4bb32d06efaf55cbd2f13e54b163928f26bd8d38b7cbba8fcc",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-utils-580_580.65.06-0ubuntu1_amd64.deb",
    },
    "xserver-xorg-video-nvidia-580": {
        "package": "xserver-xorg-video-nvidia-580",
        "version": "580.65.06-0ubuntu1",
        "filename": "xserver-xorg-video-nvidia-580_580.65.06-0ubuntu1_amd64.deb",
        "size_bytes": 1693946,
        "sha256": "sha256:b23df043e5d3fc049a8192dd30b82f9519c8212b319cddc4a4c569cb455ed928",
        "url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/xserver-xorg-video-nvidia-580_580.65.06-0ubuntu1_amd64.deb",
    },
}
SCHEMA = "native_g1_team_vm_system_packages.v2"
MANIFEST = "packages.json"
OBSERVATION = "observed.json"
EMPTY_LISTS = "empty-apt-lists"
FREE_FLOOR = 8_000_000_000


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value, field="receipt_digest"):
    return (
        "sha256:"
        + hashlib.sha256(
            _canonical({k: v for k, v in value.items() if k != field}).encode()
        ).hexdigest()
    )


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("g1_vm_system_duplicate_json_key")
        result[key] = value
    return result


def _regular(path):
    if not path.is_absolute() or path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("g1_vm_system_asset_path_invalid")


def _read(path):
    _regular(path)
    if path.stat().st_size > 131072:
        raise ValueError("g1_vm_system_json_limit")
    return json.loads(path.read_text(), object_pairs_hook=_object)


def verify_guest_observation(path):
    value = _read(path)
    if (
        value.get("receipt_digest") != _digest(value)
        or value.get("schema_version") != "g1_vm_system_cpu_preflight.v1"
        or value.get("status") != "guest_cpu_observed"
        or value.get("child_terminal") is not True
        or type(value.get("exit_code")) is not int
        or value["exit_code"] != 0
        or value.get("image_ref") != IMAGE_REF
        or value.get("guest_disk_sha256") != GUEST_DISK_SHA
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_commit"))) is None
    ):
        raise ValueError("g1_vm_system_guest_observation_invalid")
    guest = value.get("guest_observation")
    if (
        not isinstance(guest, dict)
        or guest.get("receipt_digest") != _digest(guest)
        or guest.get("schema_version") != "g1_vm_guest_system_cpu_observation.v1"
        or type(guest.get("uid")) is not int
        or guest["uid"] != 0
        or guest.get("platform") != "x86_64"
    ):
        raise ValueError("g1_vm_system_guest_receipt_invalid")
    for item in (value, guest):
        if (
            item.get("scope") != "local_tcg_cpu_only"
            or item.get("claim_ceiling") != "development_only"
            or any(
                item.get(key) is not False
                for key in (
                    "gpu_runtime_qualified",
                    "policy_inference_performed",
                    "provider_mutation_performed",
                )
            )
        ):
            raise ValueError("g1_vm_system_observation_scope_invalid")
    probes = guest.get("probes", {})
    kernel, packages = probes.get("kernel", {}), probes.get("packages", {})
    if (
        type(kernel.get("exit_code")) is not int
        or kernel["exit_code"] != 0
        or kernel.get("stdout", "").strip() != KERNEL
        or type(packages.get("exit_code")) is not int
        or packages["exit_code"] != 0
    ):
        raise ValueError("g1_vm_system_guest_kernel_invalid")
    rows = [row.split(" ", 1) for row in packages.get("stdout", "").splitlines()]
    if any(len(row) != 2 for row in rows) or len({row[0] for row in rows}) != len(rows):
        raise ValueError("g1_vm_system_guest_package_inventory_invalid")
    installed = dict(rows)
    if any(installed.get(name) != version for name, version in BASE_PACKAGES.items()):
        raise ValueError("g1_vm_system_guest_build_prerequisites_invalid")
    return value


def _verify_asset(path, row):
    _regular(path)
    if path.stat().st_size != row["size_bytes"]:
        raise ValueError("g1_vm_system_package_size_invalid")
    digest = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            digest.update(chunk)
    if "sha256:" + digest.hexdigest() != row["sha256"]:
        raise ValueError("g1_vm_system_package_hash_invalid")


def staged_filename(row):
    if re.fullmatch(r"sha256:[0-9a-f]{64}", row["sha256"]) is None:
        raise ValueError("g1_vm_system_asset_digest_invalid")
    return row["sha256"].split(":", 1)[1][:16] + ".deb"


def _transport_names():
    names = [staged_filename(row) for row in SYSTEM_PACKAGES.values()]
    if len(names) != len(set(names)):
        raise ValueError("g1_vm_system_transport_alias_collision")
    return names


def _fresh(root, additional_bytes):
    if (
        not root.is_absolute()
        or root.resolve() != root
        or root.exists()
        or not root.parent.is_dir()
    ):
        raise ValueError("g1_vm_system_fresh_root_required")
    if shutil.disk_usage(root.parent).free < FREE_FLOOR + additional_bytes:
        raise ValueError("g1_vm_system_capacity_insufficient")


def _manifest(commit, observed):
    value = {
        "schema_version": SCHEMA,
        "implementation_commit": commit,
        "image_ref": IMAGE_REF,
        "guest_disk_sha256": GUEST_DISK_SHA,
        "kernel": KERNEL,
        "guest_observation_digest": observed["receipt_digest"],
        "base_packages": BASE_PACKAGES,
        "system_packages": SYSTEM_PACKAGES,
        "owner_review_required": True,
        "runtime_installation_performed": False,
        "provider_mutation_performed": False,
        "gpu_runtime_qualified": False,
        "claim_ceiling": "development_only",
    }
    value["manifest_digest"] = _digest(value, "manifest_digest")
    return value


def prepare_system_packages(
    *,
    observation_path,
    asset_paths,
    output_root,
    implementation_commit,
    link_immutable_assets=False,
):
    if type(link_immutable_assets) is not bool:
        raise ValueError("g1_vm_system_asset_link_flag_invalid")
    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None or set(asset_paths) != set(
        SYSTEM_PACKAGES
    ):
        raise ValueError("g1_vm_system_package_inputs_invalid")
    observed = verify_guest_observation(observation_path)
    _transport_names()
    for role, row in SYSTEM_PACKAGES.items():
        _verify_asset(asset_paths[role], row)
        if link_immutable_assets and (
            asset_paths[role].stat().st_mode & 0o222
            or asset_paths[role].stat().st_dev != output_root.parent.stat().st_dev
        ):
            raise ValueError("g1_vm_system_asset_link_unavailable")
    asset_bytes = (
        0 if link_immutable_assets else sum(row["size_bytes"] for row in SYSTEM_PACKAGES.values())
    )
    _fresh(output_root, asset_bytes + 131072)
    output_root.mkdir(mode=0o700)
    (output_root / EMPTY_LISTS).mkdir(mode=0o700)
    for role, row in SYSTEM_PACKAGES.items():
        target = output_root / staged_filename(row)
        if link_immutable_assets:
            os.link(asset_paths[role], target, follow_symlinks=False)
        else:
            shutil.copyfile(asset_paths[role], target)
        _verify_asset(target, row)
        if not link_immutable_assets:
            target.chmod(0o444)
    shutil.copyfile(observation_path, output_root / OBSERVATION)
    if verify_guest_observation(output_root / OBSERVATION) != observed:
        raise ValueError("g1_vm_system_observation_changed")
    (output_root / OBSERVATION).chmod(0o444)
    value = _manifest(implementation_commit, observed)
    with (output_root / MANIFEST).open("x") as destination:
        destination.write(_canonical(value) + "\n")
    (output_root / MANIFEST).chmod(0o444)
    return value


def verify_system_packages(root, *, expected_implementation_commit):
    if (
        not root.is_absolute()
        or root.resolve() != root
        or not root.is_dir()
        or re.fullmatch(r"[0-9a-f]{40}", expected_implementation_commit) is None
    ):
        raise ValueError("g1_vm_system_package_root_invalid")
    expected_names = {
        MANIFEST,
        OBSERVATION,
        EMPTY_LISTS,
        *_transport_names(),
    }
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("g1_vm_system_package_inventory_invalid")
    lists = root / EMPTY_LISTS
    if not lists.is_dir() or lists.resolve() != lists or list(lists.iterdir()):
        raise ValueError("g1_vm_system_empty_apt_lists_invalid")
    observed = verify_guest_observation(root / OBSERVATION)
    value = _read(root / MANIFEST)
    if value != _manifest(expected_implementation_commit, observed):
        raise ValueError("g1_vm_system_package_manifest_invalid")
    for row in SYSTEM_PACKAGES.values():
        _verify_asset(root / staged_filename(row), row)
    return value


def offline_apt_simulation_command(root, *, expected_implementation_commit):
    verify_system_packages(root, expected_implementation_commit=expected_implementation_commit)
    return [
        "apt-get",
        "--simulate",
        "--no-download",
        "--no-install-recommends",
        "-o",
        "Dir::Etc::sourcelist=-",
        "-o",
        "Dir::Etc::sourceparts=-",
        "-o",
        "Dir::State::lists=" + str(root / EMPTY_LISTS),
        "install",
        *(str(root / staged_filename(row)) for row in SYSTEM_PACKAGES.values()),
    ]


def fetch_fixed_package_bytes(root, *, _deadline=None):
    """Read public pinned bytes; no Debian scripts or package install is run."""
    _fresh(root, sum(row["size_bytes"] for row in SYSTEM_PACKAGES.values()))
    root.mkdir(mode=0o700)
    for row in SYSTEM_PACKAGES.values():
        path, count = root / row['filename'], 0
        deadline = artifact_deadline(600,deadline=_deadline)
        try:
            with open_artifact_response(row['url'],deadline=deadline,socket_timeout=30) as response, path.open('xb') as destination:
                for chunk in artifact_chunks(response,deadline=deadline,maximum_bytes=row['size_bytes'],chunk_bytes=4*1024**2):
                    if shutil.disk_usage(root).free < FREE_FLOOR + len(chunk):
                        raise ValueError('g1_vm_system_capacity_insufficient')
                    destination.write(chunk)
                    count += len(chunk)
        except (TimeoutError,ValueError) as exc:
            if isinstance(exc,TimeoutError) or str(exc) == 'artifact_stream_oversized':
                raise ValueError('g1_vm_system_package_transfer_bound') from exc
            raise
        _verify_asset(path, row)
        path.chmod(0o444)
        print(
            _canonical(
                {"package": row["package"], "version": row["version"], "verified_bytes": count}
            ),
            flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--download-only-root", type=Path)
    modes.add_argument("--prepare-root", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--guest-observation", type=Path)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--link-immutable-assets", action="store_true")
    args = parser.parse_args(argv)
    if args.download_only_root is not None:
        if any(
            (
                args.asset_root,
                args.guest_observation,
                args.implementation_commit,
                args.link_immutable_assets,
            )
        ):
            parser.error("preparation options require --prepare-root")
        fetch_fixed_package_bytes(args.download_only_root)
        return
    if not all((args.asset_root, args.guest_observation, args.implementation_commit)):
        parser.error("prepare mode requires assets, observation and implementation commit")
    checkout = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=checkout, text=True
    ).strip()
    if head != args.implementation_commit or dirty:
        raise ValueError("g1_vm_system_immutable_source_required")
    value = prepare_system_packages(
        observation_path=args.guest_observation,
        asset_paths={
            role: args.asset_root / row["filename"] for role, row in SYSTEM_PACKAGES.items()
        },
        output_root=args.prepare_root,
        implementation_commit=head,
        link_immutable_assets=args.link_immutable_assets,
    )
    print(
        _canonical(
            {"manifest_digest": value["manifest_digest"], "output_root": str(args.prepare_root)}
        )
    )


if __name__ == "__main__":
    main()
