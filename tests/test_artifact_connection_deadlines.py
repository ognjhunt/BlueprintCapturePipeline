"""Artifact setup bounds cover libc DNS, address fallback and verified TLS."""

import socket
import ssl
import subprocess
import time

import pytest

from blueprint_pipeline import artifact_http_transport as transport
from blueprint_pipeline import native_g1_team_vm_system_packages as native
from blueprint_pipeline import website_mapanything_bootstrap as website
from tests.test_artifact_https_transports import registry
from tests import test_controlled_http_connection_deadlines as control_fixtures

local_peer = control_fixtures.local_peer
tls_contexts = control_fixtures.tls_contexts

MODULES = [transport, native, website, registry]


@pytest.mark.parametrize("module", MODULES)
def test_artifact_dns_stall_kills_and_reaps_credential_free_resolver(module, monkeypatch):
    original_popen, original_lookup = subprocess.Popen, socket.getaddrinfo
    children = []

    def record(*args, **kwargs):
        assert kwargs["env"] == {} and args[0][1:4] == ["-I", "-S", "-c"]
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    def stalled(*args, **kwargs):
        time.sleep(0.6)
        return original_lookup(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", stalled)
    monkeypatch.setattr(subprocess, "Popen", record)
    if hasattr(module, "_RESOLVER_SCRIPT"):
        monkeypatch.setattr(module, "_RESOLVER_SCRIPT", "import time; time.sleep(30)")
    started = time.monotonic()
    connection = module._DeadlineHTTPSConnection(
        "localhost", deadline=started + 0.15, timeout=30)
    with pytest.raises(TimeoutError):
        connection.connect()
    assert time.monotonic() - started < 0.5
    assert connection.sock is None and len(children) == 1
    assert children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed


@pytest.mark.parametrize("module", MODULES)
def test_artifact_address_fallback_uses_remaining_budget_and_socket_cap(module, monkeypatch):
    clock, peers = [0.0], []

    class Peer:
        def __init__(self, *_args):
            self.timeouts, self.closed = [], False
            peers.append(self)

        def settimeout(self, value):
            self.timeouts.append(value)

        def connect(self, address):
            clock[0] += 0.6
            if address[0] == "192.0.2.1":
                raise OSError("synthetic first address refused")

        def close(self):
            self.closed = True

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(socket, "socket", Peer)
    monkeypatch.setattr(module, "_resolve_addresses", lambda *_args, **_kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.0.2.1", 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.0.2.2", 443)),
    ])
    with pytest.raises(TimeoutError, match="artifact_transfer_deadline"):
        module._create_deadline_connection(("synthetic.invalid", 443), deadline=1,
                                           maximum_timeout=0.8)
    assert [peer.timeouts for peer in peers] == [[0.8], [0.4]]
    assert all(peer.closed for peer in peers)


@pytest.mark.parametrize("module", MODULES)
def test_artifact_tls_stall_uses_budget_remaining_after_dns(module, monkeypatch, tls_contexts):
    _server, client = tls_contexts
    monkeypatch.setattr(module, "_RESOLVER_SCRIPT",
                        "import time; time.sleep(0.15)\n" + module._RESOLVER_SCRIPT)

    def stall(peer, stop, observed):
        observed["connected"] = True
        stop.wait(1)

    with local_peer(stall) as (port, observed):
        started = time.monotonic()
        connection = module._DeadlineHTTPSConnection(
            "localhost", port, context=client, deadline=started + 0.4, timeout=30)
        with pytest.raises(TimeoutError):
            connection.connect()
        assert 0.3 < time.monotonic() - started < 0.75
        assert connection.sock is None
    assert observed["connected"]


@pytest.mark.parametrize("module", MODULES)
def test_artifact_verified_tls_keeps_hostname_check_and_anonymous_get(module, tls_contexts):
    server, client = tls_contexts

    def respond(peer, _stop, observed):
        with server.wrap_socket(peer, server_side=True) as tls:
            raw = bytearray()
            while b"\r\n\r\n" not in raw:
                raw.extend(tls.recv(4096))
                assert len(raw) <= 8192
            observed["request"] = bytes(raw)
            tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nbyte")

    with local_peer(respond) as (port, observed):
        connection = module._DeadlineHTTPSConnection(
            "localhost", port, context=client, deadline=time.monotonic() + 2, timeout=1)
        try:
            connection.request("GET", "/artifact")
            with connection.getresponse() as response:
                assert response.status == 200 and response.read() == b"byte"
        finally:
            connection.close()
    assert observed["request"].startswith(b"GET /artifact HTTP/1.1\r\n")
    assert b"Authorization:" not in observed["request"]
    assert client.check_hostname and client.verify_mode == ssl.CERT_REQUIRED


@pytest.mark.parametrize("module", MODULES)
def test_artifact_wrong_tls_hostname_closes_connection(module, tls_contexts):
    server, client = tls_contexts

    def handshake(peer, _stop, observed):
        try:
            server.wrap_socket(peer, server_side=True).close()
        except ssl.SSLError:
            observed["rejected"] = True

    with local_peer(handshake) as (port, observed):
        connection = module._DeadlineHTTPSConnection(
            "127.0.0.1", port, context=client, deadline=time.monotonic() + 2, timeout=1)
        with pytest.raises(ssl.SSLCertVerificationError):
            connection.connect()
        assert connection.sock is None
    assert observed["rejected"]
