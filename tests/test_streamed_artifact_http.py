"""Real HTTP framing and redirect dispatch, with in-memory sockets only."""

import ast
import http.client
import io
from pathlib import Path
import socket
from types import SimpleNamespace
import urllib.request
import urllib.error
import urllib.response

import pytest

from blueprint_pipeline import artifact_http_transport as transport
from blueprint_pipeline import native_g1_team_vm_system_packages as native
from blueprint_pipeline import production_blender_runtime as blender
from blueprint_pipeline import task_evaluation_artifixer_pretraining as artifixer
from blueprint_pipeline import website_mapanything_bootstrap as website
from tests.test_artifact_https_transports import registry


@pytest.mark.parametrize("module", [transport, native, website])
@pytest.mark.parametrize("url", [
    "http://artifact.example/a", "file:///a", "https://user:secret@artifact.example/a",
    "https://artifact.example:444/a", "https://artifact.example/a#fragment",
    "https://artifact.example/a\r\nAuthorization: secret",
])
def test_artifact_url_refuses_before_fetch(module, url, monkeypatch):
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: pytest.fail("no fetch before policy"))
    with pytest.raises(ValueError, match="artifact_https_url_invalid"):
        module.open_artifact_response(url, deadline=100, socket_timeout=30)


def test_initial_capability_query_is_preserved_and_headers_refuse_before_fetch(monkeypatch):
    clock = [0]
    monkeypatch.setattr(transport.time, "monotonic", lambda: clock[0])
    calls = []
    url = "https://origin.example/blob?generation=37&signature=synthetic"
    response = SimpleNamespace(status=200, url=url)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: SimpleNamespace(
        open=lambda request, **kw: calls.append((request, kw)) or response))
    assert transport.open_artifact_response(url, deadline=10, socket_timeout=30) is response
    assert calls[0][0].full_url == url and calls[0][1] == {"timeout": 10}
    with pytest.raises(ValueError, match="artifact_headers_invalid"):
        transport.open_artifact_response(url, deadline=10, socket_timeout=30,
                                         headers={"Authorization": "synthetic"})
    assert len(calls) == 1


@pytest.mark.parametrize("module,handler", [
    (transport, "_AnonymousArtifactRedirect"), (native, "_AnonymousArtifactRedirect"),
    (website, "_AnonymousArtifactRedirect"), (registry, "_AnonymousBlobRedirect"),
])
def test_real_redirect_dispatch_closes_without_draining_or_forwarding_credentials(module, handler):
    class UnboundedBody:
        closed = False
        def read(self, *args):
            pytest.fail("redirect body must never be drained")
        def close(self):
            self.closed = True
    old = UnboundedBody()
    request = urllib.request.Request("https://origin.example/a?source_secret=synthetic",
        headers={"Authorization": "Bearer synthetic", "Cookie": "synthetic", "X-Secret": "synthetic"})
    request.timeout = 7
    observed = []
    redirect = getattr(module, handler)()
    redirect.parent = SimpleNamespace(open=lambda request, **kw: observed.append((request, kw)) or "followed")
    assert redirect.http_error_302(request, old, 302, "redirect", {
        "location": "https://cdn.example/a?cdn_signature=synthetic"
    }) == "followed"
    assert old.closed and len(observed) == 1
    target, options = observed[0]
    assert target.full_url == "https://cdn.example/a?cdn_signature=synthetic"
    assert target.get_method() == "GET" and target.data is None
    assert not target.has_header("Authorization") and not target.has_header("Cookie")
    assert not target.has_header("X-Secret") and not target.unredirected_hdrs
    assert options == {"timeout": 7}
    refused = UnboundedBody()
    with pytest.raises(ValueError):
        redirect.http_error_307(target, refused, 307, "redirect", {"location": "http://cdn.example/a"})
    assert refused.closed and len(observed) == 1


@pytest.mark.parametrize("module,handler", [
    (registry, "_RegistryNoRedirect"), (website, "_CapabilityNoRedirect"),
])
def test_refused_control_redirect_closes_body_without_following(module, handler):
    closed = []
    redirect = getattr(module, handler)()
    redirect.parent = SimpleNamespace(open=lambda *a, **kw: pytest.fail("control redirect may not follow"))
    with pytest.raises(ValueError):
        redirect.http_error_302(None, SimpleNamespace(close=lambda: closed.append(True)),
                               302, "redirect", {"location": "https://other.example/"})
    assert closed == [True]


@pytest.mark.parametrize("module,handler,initial_method,status,target", [
    (transport, "_AnonymousArtifactRedirect", "GET", 302, "https://cdn.example/blob"),
    (transport, "_AnonymousArtifactRedirect", "GET", 302, "http://cdn.example/blob"),
    (registry, "_RegistryNoRedirect", "GET", 302, "https://cdn.example/blob"),
    (registry, "_AnonymousBlobRedirect", "GET", 302, "https://cdn.example/blob"),
    (website, "_CapabilityNoRedirect", "PUT", 307, "https://cdn.example/blob"),
    (transport, "_AnonymousArtifactRedirect", "GET", 401, None),
])
def test_actual_urllib_dispatch_keeps_refused_or_followed_responses_closed(
    module,handler,initial_method,status,target
):
    calls,responses = [],[]
    class UnreadBody(io.BytesIO):
        def read(self,*args):
            pytest.fail("redirect/error dispatch may not drain body")
    class SyntheticHTTPS(urllib.request.HTTPSHandler):
        def https_open(self,request):
            calls.append(request)
            code = status if len(calls) == 1 else 200
            headers = {"location":target} if code != 200 and target else {}
            result = urllib.response.addinfourl(UnreadBody(b"unbounded synthetic old body"),
                headers,request.full_url,code)
            result.msg = "synthetic response"
            responses.append(result)
            return result
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),SyntheticHTTPS(),
        getattr(module,handler)(),module._ClosingHTTPErrorProcessor())
    request = urllib.request.Request("https://origin.example/blob?private=synthetic",
        data=b"synthetic failure" if initial_method == "PUT" else None,method=initial_method)
    request.add_unredirected_header("Authorization","Bearer synthetic")
    allowed = handler in {"_AnonymousArtifactRedirect","_AnonymousBlobRedirect"} and target and target.startswith("https:")
    if allowed:
        with opener.open(request,timeout=7) as final:
            assert final.status == 200 and responses[0].closed
        assert len(calls) == 2 and not calls[1].has_header("Authorization")
    elif status == 401:
        with pytest.raises(urllib.error.HTTPError) as held:
            opener.open(request,timeout=7)
        assert held.value.fp.closed and responses[0].closed and len(calls) == 1
    else:
        with pytest.raises(ValueError):
            opener.open(request,timeout=7)
        assert responses[0].closed and len(calls) == 1


@pytest.mark.parametrize("framing", ["length", "chunked", "eof"])
def test_actual_http_response_full_transfer_eof_and_size_cap(framing):
    body = b"tiny admitted immutable bytes"
    if framing == "length":
        wire = b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    elif framing == "chunked":
        wire = b"Transfer-Encoding: chunked\r\n\r\n" + hex(len(body))[2:].encode() + b"\r\n" + body + b"\r\n0\r\n\r\n"
    else:
        wire = b"\r\n" + body
    class Peer:
        def makefile(self, *_args):
            return io.BytesIO(b"HTTP/1.1 200 OK\r\n" + wire)
    for cap, accepted in ((len(body), True), (len(body)-1, False)):
        response = http.client.HTTPResponse(Peer())
        response.begin()
        try:
            if accepted:
                assert b"".join(transport.artifact_chunks(response,
                    deadline=transport.time.monotonic()+10, maximum_bytes=cap, chunk_bytes=3)) == body
                assert response.fp is None
            else:
                with pytest.raises(ValueError, match="artifact_stream_oversized"):
                    list(transport.artifact_chunks(response,
                        deadline=transport.time.monotonic()+10, maximum_bytes=cap, chunk_bytes=3))
        finally:
            response.close()


@pytest.mark.parametrize("prefix,suffix", [
    (b"", b"HTTP/1.1 200 OK\r\n\r\n"),
    (b"HTTP/1.1 200 OK\r\n", b"X-Test: slow\r\n\r\n"),
    (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n", b"1;slow=metadata\r\nx\r\n0\r\n\r\n"),
    (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n", b"X-Trailer: slow\r\n\r\n"),
    (b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\n", b"xxx"),
])
@pytest.mark.parametrize("module", [transport, native, website, registry])
def test_real_http_parser_clamps_slow_headers_body_and_chunk_framing(module, prefix, suffix, monkeypatch):
    clock = [0.0]
    class Peer:
        def __init__(self):
            self.prefix, self.suffix = prefix, bytearray(suffix)
            self.timeouts = []
        def settimeout(self, value):
            self.timeouts.append(value)
        def recv_into(self, target):
            if self.prefix:
                count = len(self.prefix)
                target[:count], self.prefix = self.prefix, b""
                return count
            if self.timeouts[-1] < 4:
                clock[0] += self.timeouts[-1]
                raise TimeoutError("synthetic remaining budget")
            clock[0] += 4
            target[0] = self.suffix.pop(0)
            return 1
        def makefile(self, *_args, **_kwargs):
            return io.BufferedReader(socket.SocketIO(self, "r"))
        def _decref_socketios(self):
            pass
    peer = Peer()
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    response = http.client.HTTPResponse(module._DeadlineSocket(peer, 10, 30))
    try:
        with pytest.raises(TimeoutError, match="synthetic remaining budget"):
            response.begin()
            list(transport.artifact_chunks(response, deadline=10, maximum_bytes=64))
        assert clock[0] == 10 and peer.timeouts[-3:] == [10, 6, 2]
    finally:
        response.close()


def test_original_socket_timeout_and_earlier_absolute_deadline_are_preserved(monkeypatch):
    monkeypatch.setattr(transport.time, "monotonic", lambda: 100)
    assert transport.artifact_deadline(600, deadline=107) == 107
    assert transport.artifact_deadline(600, deadline=900) == 700
    for deadline in (100, 99):
        with pytest.raises(TimeoutError):
            transport.artifact_deadline(600, deadline=deadline)
    for deadline in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            transport.artifact_deadline(600, deadline=deadline)
    wrapped = transport._DeadlineSocket(SimpleNamespace(), 700, 30)
    assert wrapped._remaining() == 30
    monkeypatch.setattr(transport.time, "monotonic", lambda: 695)
    assert wrapped._remaining() == 5


@pytest.mark.parametrize("module", [transport,native,website,registry])
def test_each_late_redirect_connection_clamps_tcp_tls_timeout_before_connect(module,monkeypatch):
    clock,calls = [95],[]
    monkeypatch.setattr(module.time,"monotonic",lambda:clock[0])
    def connection(self):
        calls.append(self.timeout)
        self.sock = SimpleNamespace()
    monkeypatch.setattr(http.client.HTTPSConnection,"connect",connection)
    late = module._DeadlineHTTPSConnection("synthetic.invalid",deadline=100,timeout=30)
    late.connect()
    assert calls == [5] and late.timeout == 5 and late.sock._maximum_timeout == 5
    clock[0] = 100
    expired = module._DeadlineHTTPSConnection("synthetic.invalid",deadline=100,timeout=30)
    with pytest.raises(TimeoutError,match="artifact_transfer_deadline"):
        expired.connect()
    assert calls == [5]


def test_small_native_package_transfer_keeps_exact_size_sha_and_immutable_mode(
    tmp_path,monkeypatch
):
    body = b"tiny synthetic deb bytes"
    import hashlib
    row = {"filename":"fixture.deb","package":"fixture","version":"1",
           "size_bytes":len(body),"sha256":"sha256:"+hashlib.sha256(body).hexdigest(),
           "url":"https://origin.example/fixture.deb"}
    monkeypatch.setattr(native,"SYSTEM_PACKAGES",{"fixture":row})
    monkeypatch.setattr(native.shutil,"disk_usage",lambda *a:SimpleNamespace(free=10**12))
    monkeypatch.setattr(native,"open_artifact_response",lambda *a,**kw:io.BytesIO(body))
    native.fetch_fixed_package_bytes(tmp_path/"good")
    path = tmp_path/"good/fixture.deb"
    assert path.read_bytes() == body and path.stat().st_mode & 0o777 == 0o444
    monkeypatch.setattr(native,"open_artifact_response",lambda *a,**kw:io.BytesIO(body[:-1]))
    with pytest.raises(ValueError):
        native.fetch_fixed_package_bytes(tmp_path/"truncated")
    monkeypatch.setattr(native,"open_artifact_response",lambda *a,**kw:io.BytesIO(body+b"x"))
    with pytest.raises(ValueError,match="package_transfer_bound"):
        native.fetch_fixed_package_bytes(tmp_path/"oversized")


@pytest.mark.parametrize("fault,code", [
    ("truncated","download_truncated"), ("extra","download_exceeds_binding"),
    ("digest","archive_digest_invalid"),
])
def test_capsule_transfer_refuses_before_extraction_or_scientific_consumption(monkeypatch,fault,code):
    import hashlib
    body = b"tiny capsule transfer fixture"
    actual = body[:-1] if fault == "truncated" else body+b"x" if fault == "extra" else body
    monkeypatch.setattr(artifixer,"open_artifact_response",lambda *a,**kw:io.BytesIO(actual))
    monkeypatch.setattr(artifixer.zipfile,"ZipFile",lambda *a,**kw:pytest.fail("no extraction before binding"))
    environment = {artifixer.CAPSULE_URL_ENV:"https://objects.example/capsule?generation=12&signature=synthetic",
        artifixer.CAPSULE_SHA_ENV:"sha256:"+('0'*64 if fault == "digest" else hashlib.sha256(body).hexdigest()),
        artifixer.CAPSULE_BYTES_ENV:str(len(body))}
    with pytest.raises(ValueError,match="artifixer_pretraining_"+code):
        artifixer.consume_pretraining_capsule(environment=environment,stage_input={})


@pytest.mark.parametrize("caller", ["website","blender","capsule"])
def test_earlier_caller_deadline_refuses_before_network_or_file_creation(tmp_path,monkeypatch,caller):
    monkeypatch.setattr(transport.time,"monotonic",lambda:100)
    monkeypatch.setattr(urllib.request,"build_opener",lambda *a:pytest.fail("expired deadline cannot fetch"))
    path = tmp_path/"forbidden-download"
    with pytest.raises((TimeoutError,blender.BlenderRuntimeError)):
        if caller == "website":
            website._download("https://objects.example/a",path,"sha256:"+"0"*64,4,_deadline=99)
        elif caller == "blender":
            blender._download_archive(blender.ARCHIVE_URL,path,_deadline=99)
        else:
            artifixer.consume_pretraining_capsule(environment={
                artifixer.CAPSULE_URL_ENV:"https://objects.example/a",
                artifixer.CAPSULE_SHA_ENV:"sha256:"+"0"*64,
                artifixer.CAPSULE_BYTES_ENV:"4"},stage_input={},_deadline=99)
    assert not path.exists()


def test_standalone_source_cohorts_embed_the_exact_transport_contract():
    source = Path(transport.__file__).read_text()
    definitions = {node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for module in (native, website):
        actual = {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(Path(module.__file__).read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in definitions}
        assert actual == definitions
    classes = {name: value for name, value in definitions.items()
               if name in {"_DeadlineSocket", "_DeadlineHTTPSConnection", "_DeadlineHTTPSHandler", "_ClosingHTTPErrorProcessor"}}
    actual = {node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(Path(registry.__file__).read_text()).body
        if isinstance(node, ast.ClassDef) and node.name in classes}
    assert actual == classes
