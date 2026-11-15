"""ADP-010 day-7 control calls share DNS, TCP, TLS, and parser deadlines."""
from __future__ import annotations

import http.client
import socket
import ssl
import subprocess
import threading
import time
from contextlib import contextmanager

import pytest


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


def record_resolver_children(monkeypatch):
    original = subprocess.Popen
    children = []

    def record(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", record)
    return children


