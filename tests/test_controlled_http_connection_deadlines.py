"""ADP-010 day-7 control calls share DNS, TCP, TLS, and parser deadlines."""
from __future__ import annotations

import http.client
import selectors
import socket
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest

from blueprint_pipeline import controlled_http_json as transport


@pytest.fixture(scope="module")
def tls_contexts(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-control-client-tls")
    certificate, key = root / "certificate.pem", root / "key.pem"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(certificate), "-days", "1",
        "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(certificate, key)
    client = ssl.create_default_context(cafile=str(certificate))
    return server, client


@contextmanager
def local_peer(handler):
    """Use actual loopback sockets; close and join every synthetic peer."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2)
    stop = threading.Event()
    observed = {}
    failures = []

    def serve():
        try:
            with listener.accept()[0] as peer:
                peer.settimeout(2)
                handler(peer, stop, observed)
        except (OSError, AssertionError) as error:
            failures.append(error)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        yield listener.getsockname()[1], observed
    finally:
        stop.set()
        listener.close()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not failures, failures


def slow_resolution(monkeypatch, seconds):
    # Fault injection occurs in a real, killable resolver subprocess. Also slow
    # the original stdlib resolver so this regression detects the pre-fix path.
    original = socket.getaddrinfo

    def delayed(*args, **kwargs):
        time.sleep(seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", delayed)
    if hasattr(transport, "_RESOLVER_SCRIPT"):
        monkeypatch.setattr(transport, "_RESOLVER_SCRIPT",
                            f"import time; time.sleep({seconds!r})\n" + transport._RESOLVER_SCRIPT)


def assert_transport_timeout(error):
    assert isinstance(error.value.reason, TimeoutError)


def read_complete_request(tls, *, headers_received=None):
    """Consume this fixture's entire request before responding or closing TLS."""
    deadline = time.monotonic() + 2
    raw = bytearray()

    def receive(count):
        remaining = deadline - time.monotonic()
        assert remaining > 0, "synthetic request deadline elapsed"
        tls.settimeout(remaining)
        chunk = tls.recv(count)
        assert chunk, "synthetic request was truncated"
        assert time.monotonic() < deadline, "synthetic request deadline elapsed"
        return chunk

    while b"\r\n\r\n" not in raw:
        raw.extend(receive(min(4096, 8193 - len(raw))))
        assert len(raw) <= 8192, "synthetic request headers exceeded bound"
    header, _separator, body = raw.partition(b"\r\n\r\n")
    lengths = [line.partition(b":")[2].strip() for line in header.split(b"\r\n")[1:]
               if line.partition(b":")[0].lower() == b"content-length"]
    assert len(lengths) == 1 and lengths[0].isdigit(), "synthetic request length invalid"
    length = int(lengths[0])
    assert length <= 64, "synthetic request body exceeded bound"
    if headers_received is not None:
        headers_received.set()
    while len(body) < length:
        body.extend(receive(length - len(body)))
    assert len(body) == length, "synthetic request body length mismatch"
    return bytes(header) + b"\r\n\r\n" + bytes(body)


def test_real_https_json_uses_verified_peer_and_original_method(tls_contexts):
    server, client = tls_contexts

    def respond(peer, _stop, observed):
        with server.wrap_socket(peer, server_side=True) as tls:
            observed["request"] = read_complete_request(tls)
            body = b'{"ok":true}'
            tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n" + body)

    with local_peer(respond) as (port, observed):
        origin = f"https://localhost:{port}"
        request = urllib.request.Request(origin + "/control", data=b"{}", method="PATCH")
        request.add_unredirected_header("Authorization", "Bearer synthetic")
        assert transport.read_control_json(request, origin=origin, method="PATCH", timeout=2,
                                           maximum_bytes=64, context=client, direct=True) == {"ok": True}
    assert observed["request"].startswith(b"PATCH /control HTTP/1.1\r\n")
    assert b"Authorization: Bearer synthetic\r\n" in observed["request"]
    assert observed["request"].partition(b"\r\n\r\n")[2] == b"{}"
    assert client.check_hostname and client.verify_mode == ssl.CERT_REQUIRED


def test_real_success_peer_waits_for_separately_delivered_tls_request_body(tls_contexts):
    server, client = tls_contexts
    headers_received = threading.Event()

    def respond(peer, _stop, observed):
        with server.wrap_socket(peer, server_side=True) as tls:
            observed["request"] = read_complete_request(tls, headers_received=headers_received)
            tls.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{"ok":true}')

    with local_peer(respond) as (port, observed):
        with socket.create_connection(("127.0.0.1", port), timeout=2) as peer:
            with client.wrap_socket(peer, server_hostname="localhost") as tls:
                header = (b"PATCH /control HTTP/1.1\r\nHost: localhost\r\n"
                          b"Authorization: Bearer synthetic\r\nContent-Length: 2\r\n\r\n")
                tls.sendall(header)
                assert headers_received.wait(1)
                tls.settimeout(0.1)
                with pytest.raises(TimeoutError):
                    tls.recv(1)
                tls.settimeout(2)
                tls.sendall(b"{")
                tls.sendall(b"}")
                response = http.client.HTTPResponse(tls)
                try:
                    response.begin()
                    assert response.status == 200 and response.read() == b'{"ok":true}'
                finally:
                    response.close()
    assert observed["request"] == header + b"{}"


def test_real_https_still_rejects_wrong_certificate_hostname(tls_contexts):
    server, client = tls_contexts

    def handshake(peer, _stop, observed):
        try:
            server.wrap_socket(peer, server_side=True).close()
        except ssl.SSLError:
            observed["rejected"] = True

    with local_peer(handshake) as (port, observed):
        origin = f"https://127.0.0.1:{port}"
        with pytest.raises(urllib.error.URLError) as error:
            transport.read_control_json(urllib.request.Request(origin + "/control"),
                origin=origin, method="GET", timeout=2, maximum_bytes=64, context=client, direct=True)
        assert isinstance(error.value.reason, ssl.SSLCertVerificationError)
    assert observed["rejected"]


def test_real_tls_stall_receives_only_budget_remaining_after_dns(monkeypatch, tls_contexts):
    _server, client = tls_contexts
    slow_resolution(monkeypatch, 0.65)

    def stall(peer, stop, observed):
        observed["client_hello"] = peer.recv(4096)
        stop.wait(2)

    with local_peer(stall) as (port, observed):
        origin = f"https://localhost:{port}"
        started = time.monotonic()
        with pytest.raises(urllib.error.URLError) as error:
            transport.read_control_json(urllib.request.Request(origin + "/control"),
                origin=origin, method="GET", timeout=1, maximum_bytes=64, context=client, direct=True)
        elapsed = time.monotonic() - started
        assert_transport_timeout(error)
        assert 0.9 <= elapsed < 1.4, elapsed
    assert observed["client_hello"].startswith(b"\x16\x03")


def test_real_http_setup_and_dripping_status_share_original_budget(monkeypatch):
    slow_resolution(monkeypatch, 0.65)

    def drip(peer, stop, observed):
        observed["request"] = peer.recv(4096)
        for value in b"HTTP/1.1 200 OK\r\n\r\n{}":
            if stop.wait(0.08):
                break
            try:
                peer.sendall(bytes([value]))
            except (BrokenPipeError, ConnectionResetError):
                break

    with local_peer(drip) as (port, observed):
        started = time.monotonic()
        connection = transport._DeadlineHTTPConnection("localhost", port,
                                                       deadline=started + 1, timeout=1)
        try:
            connection.request("GET", "/control")
            with pytest.raises(TimeoutError):
                connection.getresponse()
            elapsed = time.monotonic() - started
            assert 0.9 <= elapsed < 1.4, elapsed
        finally:
            connection.close()
    assert observed["request"].startswith(b"GET /control HTTP/1.1\r\n")


def test_real_proxy_connect_parser_is_bounded_before_tls(monkeypatch, tls_contexts):
    _server, client = tls_contexts
    slow_resolution(monkeypatch, 0.65)

    def drip(peer, stop, observed):
        observed["connect"] = peer.recv(4096)
        for value in b"HTTP/1.1 200 Connection established\r\n\r\n":
            if stop.wait(0.08):
                break
            try:
                peer.sendall(bytes([value]))
            except (BrokenPipeError, ConnectionResetError):
                break

    with local_peer(drip) as (port, observed):
        started = time.monotonic()
        connection = transport._DeadlineHTTPSConnection("localhost", port, context=client,
                                                        deadline=started + 1, timeout=1)
        connection.set_tunnel("localhost", 443)
        try:
            with pytest.raises(TimeoutError):
                connection.connect()
            elapsed = time.monotonic() - started
            assert 0.9 <= elapsed < 1.4, elapsed
            assert connection.sock is None
        finally:
            connection.close()
    assert observed["connect"].startswith(b"CONNECT localhost:443 HTTP/1.")


def test_repeated_resolver_timeouts_kill_and_reap_actual_children(monkeypatch):
    original = subprocess.Popen
    children = []

    def record(*args, **kwargs):
        assert kwargs["env"] == {}
        assert args[0][1:4] == ["-I", "-S", "-c"]
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", record)
    monkeypatch.setattr(transport, "_RESOLVER_SCRIPT", "import time; time.sleep(30)")
    for _ in range(4):
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="controlled_http_response_timeout"):
            transport._resolve_addresses(("localhost", 443), deadline=started + 0.15)
        assert time.monotonic() - started < 0.8
        assert all(child.poll() is not None for child in children)
        assert all(child.stdin.closed and child.stdout.closed for child in children)
    assert len(children) == 4


def test_expired_resolution_budget_never_starts_child(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("expired lookup"))
    with pytest.raises(TimeoutError, match="controlled_http_response_timeout"):
        transport._resolve_addresses(("localhost", 443), deadline=time.monotonic() - 1)


def test_resolver_interruption_kills_and_reaps_actual_child(monkeypatch):
    original = subprocess.Popen
    children = []

    def record(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    class InterruptSelector(selectors.DefaultSelector):
        def select(self, *_args, **_kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", record)
    monkeypatch.setattr(selectors, "DefaultSelector", InterruptSelector)
    monkeypatch.setattr(transport, "_RESOLVER_SCRIPT", "import time; time.sleep(30)")
    with pytest.raises(KeyboardInterrupt):
        transport._resolve_addresses(("localhost", 443), deadline=time.monotonic() + 1)
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed


def test_address_fallback_recomputes_budget_and_closes_failed_socket(monkeypatch):
    clock = [0.0]
    peers = []

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

    monkeypatch.setattr(transport.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(socket, "socket", Peer)
    monkeypatch.setattr(transport, "_resolve_addresses", lambda *_args, **_kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.0.2.1", 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.0.2.2", 443)),
    ])
    with pytest.raises(TimeoutError, match="controlled_http_response_timeout"):
        transport._create_deadline_connection(("synthetic.invalid", 443), 1, deadline=1)
    assert [peer.timeouts for peer in peers] == [[1], [0.4]]
    assert all(peer.closed for peer in peers)


def record_resolver_children(monkeypatch):
    original = subprocess.Popen
    children = []

    def record(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", record)
    return children


def test_actual_child_output_overflow_is_bounded_killed_and_reaped(monkeypatch):
    children = record_resolver_children(monkeypatch)
    monkeypatch.setattr(transport, "_RESOLVER_SCRIPT", """
import os
while True:
    os.write(1, b'x' * 8192)
""")
    started = time.monotonic()
    with pytest.raises(ValueError, match="controlled_http_resolution_oversized"):
        transport._resolve_addresses(("localhost", 443), deadline=started + 2)
    assert time.monotonic() - started < 1
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed


@pytest.mark.parametrize("count", [257, 20_000])
def test_actual_child_excess_addresses_refuse_before_tcp(monkeypatch, count):
    children = record_resolver_children(monkeypatch)
    script = f"""
import json, socket, sys
row = [socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ['127.0.0.1', 443]]
sys.stdout.write(json.dumps({{'addresses': [row] * {count}}}))
"""
    monkeypatch.setattr(transport, "_RESOLVER_SCRIPT", script)
    monkeypatch.setattr(socket, "socket", lambda *_a, **_k: pytest.fail("oversized result TCP"))
    with pytest.raises(ValueError, match="controlled_http_resolution_oversized"):
        transport._create_deadline_connection(("localhost", 443), deadline=time.monotonic() + 2)
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed


@pytest.mark.parametrize("result", [
    '[]', '{"addresses":{}}', '{"addresses":[[2,1,6,"",["hostname.invalid",443]]]}',
    '{"addresses":[[2,1,6,"",["127.0.0.1",true]]]}',
    '{"addresses":[[2,2,17,"",["127.0.0.1",443]]]}', '{"error":["bad",{}]}',
])
def test_actual_child_malformed_result_refuses_without_tcp(monkeypatch, result):
    children = record_resolver_children(monkeypatch)
    monkeypatch.setattr(transport, "_RESOLVER_SCRIPT", f"import sys; sys.stdout.write({result!r})")
    monkeypatch.setattr(socket, "socket", lambda *_a, **_k: pytest.fail("invalid result TCP"))
    with pytest.raises(ValueError, match="controlled_http_resolution_invalid"):
        transport._create_deadline_connection(("localhost", 443), deadline=time.monotonic() + 2)
    assert len(children) == 1 and children[0].poll() is not None


def test_actual_child_preserves_gaierror_type_without_stderr_read(monkeypatch):
    monkeypatch.setattr(transport, "_RESOLVER_SCRIPT",
        'import sys; sys.stdout.write(\'{"error":[-2,"synthetic DNS refusal"]}\'); '
        'sys.stderr.write("synthetic private diagnostic")')
    with pytest.raises(socket.gaierror, match="synthetic DNS refusal") as error:
        transport._resolve_addresses(("localhost", 443), deadline=time.monotonic() + 2)
    assert error.value.errno == -2


def test_resolution_checks_shared_deadline_after_actual_output_parsing(monkeypatch):
    original = transport.json.loads

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        time.sleep(0.15)
        return result

    monkeypatch.setattr(transport.json, "loads", delayed)
    monkeypatch.setattr(transport, "_RESOLVER_SCRIPT", 'print(\'{"addresses":[]}\')')
    with pytest.raises(TimeoutError, match="controlled_http_response_timeout"):
        transport._resolve_addresses(("localhost", 443), deadline=time.monotonic() + 0.1)
