"""Synthetic HTTP transports preserve immutable bytes and capability scope."""

import hashlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import time
import urllib.request

import pytest

from blueprint_pipeline import website_mapanything_bootstrap as website

SCRIPT = Path(__file__).parents[1] / "scripts/rehearse_g1_vm_system_cpu.py"
spec = importlib.util.spec_from_file_location("artifact_registry_probe", SCRIPT)
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)


def response(body):
    class Socket:
        def settimeout(self, value):
            assert 0 < value <= 45

        def makefile(self, mode):
            return io.BufferedReader(Raw(wire, self))

    class Raw(io.RawIOBase):
        def __init__(self, data, sock):
            self.body = io.BytesIO(data)
            self._sock = sock

        def readable(self):
            return True

        def readinto(self, target):
            data = self.body.read(len(target))
            target[: len(data)] = data
            return len(data)

    wire = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    result = http.client.HTTPResponse(Socket())
    result.begin()
    return result


def test_registry_bearer_is_initial_request_only_and_cdn_redirect_is_anonymous():
    initial = registry._registry_request("blobs/" + registry.LAYER_SHA, "synthetic-token")
    assert initial.has_header("Authorization") and "Authorization" not in initial.headers
    redirect = registry._AnonymousBlobRedirect().redirect_request(
        initial,
        None,
        307,
        "redirect",
        {},
        "https://production.cloudfront.docker.com/immutable/blob?signature=synthetic",
    )
    assert redirect.get_method() == "GET" and redirect.data is None
    assert not redirect.has_header("Authorization") and not redirect.unredirected_hdrs
    again = registry._AnonymousBlobRedirect().redirect_request(
        redirect,
        None,
        302,
        "redirect",
        {},
        "https://other-cdn.example/immutable/blob?signature=synthetic",
    )
    assert not again.has_header("Authorization")


@pytest.mark.parametrize(
    "url",
    [
        "http://cdn.example/blob",
        "file:///private",
        "https://user:password@cdn.example/blob",
        "https://cdn.example:444/blob",
        "https://cdn.example/blob#fragment",
    ],
)
def test_blob_redirect_refuses_downgrade_or_url_credentials_before_following(url):
    request = registry._registry_request("blobs/" + registry.LAYER_SHA, "synthetic-token")
    with pytest.raises(ValueError, match="blob_redirect_invalid"):
        registry._AnonymousBlobRedirect().redirect_request(request, None, 302, "redirect", {}, url)


def test_registry_control_requests_never_redirect():
    with pytest.raises(ValueError, match="registry_control_redirect"):
        registry._RegistryNoRedirect().redirect_request(
            None, None, 302, "redirect", {}, "https://other.example"
        )


def test_token_response_is_capped_before_json_parse_or_authenticated_registry_request(
    tmp_path, monkeypatch
):
    calls = []

    def open_request(request, timeout):
        calls.append(request)
        return response(b"x" * (128 * 1024 + 1))

    monkeypatch.setattr(registry, "require_capacity", lambda *a, **kw: None)
    monkeypatch.setattr(
        registry.urllib.request,
        "build_opener",
        lambda *a: type("Opener", (), {"open": staticmethod(open_request)})(),
    )
    with pytest.raises(ValueError, match="registry_data_oversized"):
        registry.download_layer(tmp_path / "layer")
    assert len(calls) == 1 and not (tmp_path / "layer").exists()


@pytest.mark.parametrize("fault", [None, "wrong-bytes", "too-many-bytes"])
def test_small_immutable_layer_keeps_size_hash_and_real_http_eof(tmp_path, monkeypatch, fault):
    blob = b"tiny synthetic layer"
    digest = "sha256:" + hashlib.sha256(blob).hexdigest()
    manifest = json.dumps(
        {"layers": [{"digest": digest, "size": len(blob)}]}, sort_keys=True
    ).encode()
    monkeypatch.setattr(registry, "LAYER_SHA", digest)
    monkeypatch.setattr(registry, "LAYER_BYTES", len(blob))
    monkeypatch.setattr(registry, "IMAGE_SHA", "sha256:" + hashlib.sha256(manifest).hexdigest())
    monkeypatch.setattr(registry, "require_capacity", lambda *a, **kw: None)
    requests = []

    def open_request(request, timeout):
        url = request if isinstance(request, str) else request.full_url
        requests.append(request)
        if "/token?" in url:
            return response(b'{"token":"synthetic-token"}')
        if "/manifests/" in url:
            return response(manifest)
        assert request.unredirected_hdrs["Authorization"] == "Bearer synthetic-token"
        return response(
            blob if fault is None else (b"X" + blob[1:] if fault == "wrong-bytes" else blob + b"x")
        )

    monkeypatch.setattr(
        registry.urllib.request,
        "build_opener",
        lambda *a: type("Opener", (), {"open": staticmethod(open_request)})(),
    )
    target = tmp_path / "layer"
    if fault is None:
        registry.download_layer(target)
        assert target.read_bytes() == blob and target.stat().st_mode & 0o777 == 0o400
    else:
        with pytest.raises(ValueError, match="g1_vm_cpu_download"):
            registry.download_layer(target)
    assert len(requests) == 3


def test_response_deadline_refuses_before_reading_any_asset_bytes():
    with pytest.raises(ValueError, match="download_bound_exceeded"):
        registry._response_chunk(response(b"bytes"), 5, time.monotonic() - 1)


def test_failure_put_refuses_redirect_without_forwarding_capability():
    request = urllib.request.Request(
        "https://objects.example/put?capability=synthetic", data=b"{}", method="PUT"
    )
    with pytest.raises(ValueError, match="website_worker_capability_redirect"):
        website._CapabilityNoRedirect().redirect_request(
            request, None, 307, "redirect", {}, "https://other.example"
        )


@pytest.mark.parametrize(
    "url",
    ["http://objects.example/put", "file:///private", "https://user:pass@objects.example/put"],
)
def test_failure_put_validates_admitted_https_before_any_call(monkeypatch, url):
    monkeypatch.setenv("BLUEPRINT_RECONSTRUCTION_OUTPUT_BUNDLE_PUT_URL", url)
    for name, value in [
        ("OPERATION_REQUEST_DIGEST", "sha256:" + "a" * 64),
        ("INPUT_BUNDLE_DIGEST", "sha256:" + "b" * 64),
        ("SOURCE_COMMIT", "c" * 40),
    ]:
        monkeypatch.setenv(
            "BLUEPRINT_RECONSTRUCTION_" + name
            if name != "SOURCE_COMMIT"
            else "BLUEPRINT_SOURCE_COMMIT",
            value,
        )
    monkeypatch.setattr(
        website.urllib.request,
        "build_opener",
        lambda *a: pytest.fail("invalid capability must refuse before transport"),
    )
    with pytest.raises(ValueError, match="website_worker_failure_transport_invalid"):
        website._report_failure(ValueError("synthetic"))
