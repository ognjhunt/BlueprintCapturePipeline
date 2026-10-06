"""Standard-library bootstrap for an admitted website geometry worker.

Runs inside the allocator's exact base image. Install only the wheel and hashed
dependency files sealed into the input bundle, then enter the existing worker.
This file allocates no resources and accepts no caller-supplied shell commands.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import urllib.error
import urllib.request
import urllib.parse
import zipfile



# Standalone artifact transport: definitions match artifact_http_transport.py.
import http.client
import math
import urllib.error
import urllib.parse
import time

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


_PHASE = "start"


def _phase(name):
    global _PHASE
    _PHASE = name


def _sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _download(url,path,digest,limit,*,_deadline=None):
    # Admitted capability query remains on its HTTPS origin; redirect requests
    # are fresh anonymous HTTPS. Never print URLs or include them in receipts.
    deadline = artifact_deadline(180,deadline=_deadline)
    try:
        with open_artifact_response(url,deadline=deadline,socket_timeout=180) as response, path.open('xb') as target:
            for chunk in artifact_chunks(response,deadline=deadline,maximum_bytes=limit):
                target.write(chunk)
    except ValueError as exc:
        if str(exc) == 'artifact_stream_oversized':
            raise ValueError('website_worker_download_oversized') from exc
        raise
    if _sha(path) != digest:
        raise ValueError('website_worker_download_digest_mismatch')


def install_runtime(root):
    root.mkdir(parents=True, exist_ok=False)
    receipt_path, bundle_path = root / "receipt.json", root / "inputs.zip"
    _phase("receipt_download")
    _download(os.environ["BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_GET_URL"], receipt_path,
              os.environ["BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_FILE_DIGEST"], 8 * 1024**2)
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("operation") != "website_mapanything"
            or receipt.get("source_commit_sha") != os.environ["BLUEPRINT_SOURCE_COMMIT"]
            or receipt.get("worker_image_digest") != os.environ["BLUEPRINT_CONTAINER_IMAGE_DIGEST"]
            or receipt.get("operation_request_digest") != os.environ["BLUEPRINT_RECONSTRUCTION_OPERATION_REQUEST_DIGEST"]):
        raise ValueError("website_worker_receipt_binding_mismatch")
    size = receipt.get("bundle_bytes")
    if type(size) is not int or not 0 < size <= 512 * 1024**2:
        raise ValueError("website_worker_input_size_invalid")
    _phase("bundle_download")
    _download(os.environ["BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_GET_URL"], bundle_path,
              os.environ["BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_DIGEST"], size)
    _phase("runtime_verification")
    files = {}
    with zipfile.ZipFile(bundle_path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("website_worker_duplicate_bundle_member")
        for row in receipt["artifact_members"]:
            if row["role"] not in {"worker_wheel", "worker_dependencies"}:
                continue
            member = row["archive_path"]
            info = archive.getinfo(member)
            if info.file_size != row["bytes"] or info.file_size > 128 * 1024**2:
                raise ValueError("website_worker_runtime_size_invalid")
            destination = root / PurePosixPath(member).name
            if destination.exists():
                raise ValueError("website_worker_runtime_name_conflict")
            destination.write_bytes(archive.read(info))
            if _sha(destination) != row["digest"]:
                raise ValueError("website_worker_runtime_digest_mismatch")
            files.setdefault(row["role"], []).append(destination)
    wheels, dependencies = files.get("worker_wheel", []), files.get("worker_dependencies", [])
    if len(dependencies) != 1 or len(wheels) != 3 or any(path.suffix != ".whl" for path in wheels):
        raise ValueError("website_worker_runtime_missing")
    # The pinned PyTorch image uses Ubuntu's externally managed Python. This
    # disposable worker owns its interpreter; allow the sealed overlay there.
    _phase("dependency_install")
    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "--no-deps", "--require-hashes",
                    "--no-cache-dir", "-r", str(dependencies[0])], check=True)
    _phase("wheel_install")
    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "--no-deps", "--no-cache-dir", *map(str, wheels)], check=True)
    return root


def main():
    root = install_runtime(Path("/tmp/blueprint-website-worker"))
    from blueprint_pipeline.website_mapanything_operation import (
        MODEL_ROOT, MODEL_REVISION, MODEL_CHECKPOINT_DIGEST, MODEL_CONFIG_DIGEST,
    )
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    _phase("model_download")
    for name, digest, limit in (("model.safetensors", MODEL_CHECKPOINT_DIGEST, 5 * 1024**3),
                                ("config.json", MODEL_CONFIG_DIGEST, 1024**2)):
        target = MODEL_ROOT / name
        _download("https://huggingface.co/facebook/map-anything-apache/resolve/" + MODEL_REVISION + "/" + name,
                  target, digest, limit)
    # UniCeption calls DINOv2 through Torch Hub. Prepopulate its cache from an
    # exact source archive so that call cannot silently select newer code.
    revision = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    archive_path = root / "dinov2.zip"
    _phase("encoder_download")
    _download("https://codeload.github.com/facebookresearch/dinov2/zip/" + revision, archive_path,
              "sha256:04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48", 8 * 1024**2)
    os.environ["TORCH_HOME"] = str(root / "torch")
    cache = root / "torch" / "hub" / "facebookresearch_dinov2_main"
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if not parts or parts[0] != "dinov2-" + revision or ".." in parts or member.file_size > 8 * 1024**2:
                raise ValueError("website_worker_encoder_archive_invalid")
            target = cache.joinpath(*parts[1:])
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    os.environ["HF_HUB_OFFLINE"] = "1"
    # Model acquisition above is explicit and hash-verified; inference is local.
    _phase("geometry_inference")
    from blueprint_pipeline.reconstruction_gpu_operation_bootstrap import run_reconstruction_gpu_operation_bootstrap
    run_reconstruction_gpu_operation_bootstrap(environment=os.environ, work_root=root / "operation")


def _failure_code(exc):
    if isinstance(exc, ValueError) and str(exc).startswith("website_worker_"):
        return str(exc)[:100]
    if isinstance(exc, subprocess.CalledProcessError):
        return "subprocess_failed"
    if isinstance(exc, (TimeoutError, urllib.error.URLError)):
        return "network_or_timeout"
    return "worker_exception"


class _CapabilityNoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        raise ValueError("website_worker_capability_redirect")

    def http_error_302(self,request,response,code,message,headers):
        try:
            raise ValueError('website_worker_capability_redirect')
        finally:
            response.close()

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302



def _report_failure(exc):
    """Return a small bound failure artifact through the admitted output URL."""
    payload = {
        "schema_version": "website_mapanything_bootstrap_failure.v1",
        "status": "failed",
        "phase": _PHASE,
        "code": _failure_code(exc),
        "exception_type": type(exc).__name__,
        "operation_request_digest": os.environ["BLUEPRINT_RECONSTRUCTION_OPERATION_REQUEST_DIGEST"],
        "operation_input_bundle_digest": os.environ["BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_DIGEST"],
        "source_commit_sha": os.environ["BLUEPRINT_SOURCE_COMMIT"],
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    url = os.environ["BLUEPRINT_RECONSTRUCTION_OUTPUT_BUNDLE_PUT_URL"]
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.fragment or any(ord(char) < 32 or ord(char) == 127 for char in url)
            or len(raw) > 8192):
        raise ValueError("website_worker_failure_transport_invalid")
    request = urllib.request.Request(url,data=raw,method="PUT")
    # This admitted capability permits only the exact PUT. Explicit refusal
    # preserves that scope even if HTTP redirect behavior changes later.
    deadline = artifact_deadline(30)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),_CapabilityNoRedirect(),
        _DeadlineHTTPSHandler(deadline,None),_ClosingHTTPErrorProcessor())
    with opener.open(request,timeout=min(30,max(.001,deadline-time.monotonic()))) as response:
        if not 200 <= response.status < 300:
            raise ValueError("website_worker_failure_transport_invalid")
    print("BLUEPRINT_WEBSITE_MAPANYTHING_BOOTSTRAP_FAILURE:" + payload["phase"] + ":" + payload["code"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        try:
            _report_failure(error)
        except Exception:
            print("BLUEPRINT_WEBSITE_MAPANYTHING_BOOTSTRAP_FAILURE_REPORT_FAILED:" + _PHASE)
        sys.exit(1)
