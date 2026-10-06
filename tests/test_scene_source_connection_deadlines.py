"""Trusted installer bootstrap setup deadlines, using only synthetic loopback peers."""
from __future__ import annotations

import ast
import functools
import http.client
import json
import math
import os
import re
import selectors
import socket
import ssl
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
NAMES = {
    "_SourceDeadlineSocket", "_SourceDeadlineHTTPSConnection", "_SourceDeadlineHTTPSHandler",
    "_SourceClosingHTTPErrorProcessor", "_SourceNoRedirect", "_source_response_opener",
    "_source_connection_remaining", "_source_resolve_addresses", "_source_validated_resolver_result",
    "_source_create_deadline_connection",
}


@pytest.fixture(params=["deployer", "installer", "shell-bootstrap"])
def transport(request):
    if request.param == "shell-bootstrap":
        shell = (SCRIPTS / "install_live_pipeline_control_plane.sh").read_text()
        raw = shell.split("<<'PY_RUNTIME'\n", 1)[1].split("\nPY_RUNTIME", 1)[0]
    else:
        raw = (SCRIPTS / ("deploy_control_plane_commit.py" if request.param == "deployer"
                         else "install_scene_retirement_runtime.py")).read_text()
    nodes = [node for node in ast.parse(raw).body
             if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NAMES)
             or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                 and target.id.startswith("_SOURCE_") for target in node.targets))]
    module = types.ModuleType("synthetic_trusted_source_transport")
    module.__dict__.update({"functools": functools, "http": http, "json": json,
        "math": math, "os": os, "re": re, "selectors": selectors, "socket": socket,
        "subprocess": subprocess, "sys": sys, "time": time, "urllib": urllib})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), request.param, "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def tls_contexts(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-installer-tls")
    certificate, key = root / "certificate.pem", root / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key),
        "-out", str(certificate), "-days", "1", "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(certificate, key)
    return server, ssl.create_default_context(cafile=str(certificate))


@contextmanager
def local_peer(handler):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2)
    stop, observed, failures = threading.Event(), {}, []
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
        assert not thread.is_alive() and not failures, failures


def test_real_https_bootstrap_retains_verified_hostname_and_global_install_budget(transport, tls_contexts):
    server, client = tls_contexts
    def respond(peer, _stop, observed):
        with server.wrap_socket(peer, server_side=True) as tls:
            observed["request"] = tls.recv(4096)
            tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
    with local_peer(respond) as (port, observed):
        deadline = time.monotonic() + 900
        connection = transport._SourceDeadlineHTTPSConnection("localhost", port, deadline=deadline, timeout=2, context=client)
        try:
            connection.request("GET", "/synthetic-install-source")
            assert connection.sock._deadline == deadline and connection.sock._maximum_timeout == 2
            assert connection.getresponse().read() == b"{}"
        finally:
            connection.close()
    assert observed["request"].startswith(b"GET /synthetic-install-source ")
    assert client.check_hostname and client.verify_mode == ssl.CERT_REQUIRED


def test_real_https_bootstrap_rejects_wrong_hostname(transport, tls_contexts):
    server, client = tls_contexts
    def handshake(peer, _stop, observed):
        try:
            server.wrap_socket(peer, server_side=True).close()
        except ssl.SSLError:
            observed["rejected"] = True
    with local_peer(handshake) as (port, observed):
        connection = transport._SourceDeadlineHTTPSConnection("127.0.0.1", port, deadline=time.monotonic()+900, timeout=2, context=client)
        with pytest.raises(ssl.SSLCertVerificationError):
            connection.connect()
        assert connection.sock is None
    assert observed["rejected"]


def test_real_tls_stall_shares_setup_timeout_after_slow_dns_without_using_global900(transport, monkeypatch, tls_contexts):
    _server, client = tls_contexts
    monkeypatch.setattr(transport, "_SOURCE_RESOLVER_SCRIPT", "import time; time.sleep(0.65)\n" + transport._SOURCE_RESOLVER_SCRIPT)
    def stall(peer, stop, observed):
        observed["hello"] = peer.recv(4096)
        stop.wait(2)
    with local_peer(stall) as (port, observed):
        started = time.monotonic()
        connection = transport._SourceDeadlineHTTPSConnection("localhost", port, deadline=started+900, timeout=1, context=client)
        with pytest.raises(TimeoutError):
            connection.connect()
        elapsed = time.monotonic()-started
        assert 0.9 <= elapsed < 1.4, elapsed
        assert connection.sock is None
    assert observed["hello"].startswith(b"\x16\x03")


def test_real_proxy_connect_status_headers_are_bounded_before_tls(transport, monkeypatch, tls_contexts):
    _server, client = tls_contexts
    monkeypatch.setattr(transport, "_SOURCE_RESOLVER_SCRIPT", "import time; time.sleep(0.65)\n" + transport._SOURCE_RESOLVER_SCRIPT)
    def drip(peer, stop, observed):
        observed["request"] = peer.recv(4096)
        for value in b"HTTP/1.1 200 Connection established\r\n\r\n":
            if stop.wait(0.08):
                break
            try:
                peer.sendall(bytes([value]))
            except (BrokenPipeError, ConnectionResetError):
                break
    with local_peer(drip) as (port, observed):
        started = time.monotonic()
        connection = transport._SourceDeadlineHTTPSConnection("localhost", port, deadline=started+900, timeout=1, context=client)
        connection.set_tunnel("localhost",443)
        with pytest.raises(TimeoutError):
            connection.connect()
        assert 0.9 <= time.monotonic()-started < 1.4 and connection.sock is None
    assert observed["request"].startswith(b"CONNECT localhost:443 HTTP/1.")


def test_repeated_actual_resolver_timeouts_kill_reap_and_close_children(transport, monkeypatch):
    original, children = subprocess.Popen, []
    def record(*args, **kwargs):
        assert kwargs["env"] == {} and args[0][1:4] == ["-I","-S","-c"]
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, "Popen", record)
    monkeypatch.setattr(transport,"_SOURCE_RESOLVER_SCRIPT","import time; time.sleep(30)")
    for _ in range(2):
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            transport._source_resolve_addresses(("localhost",443),deadline=started+0.15)
        assert time.monotonic()-started < 0.8
        assert all(child.poll() is not None and child.stdin.closed and child.stdout.closed for child in children)
    assert len(children) == 2


def test_resolver_overflow_is_bounded_and_child_is_reaped(transport, monkeypatch):
    original, children = subprocess.Popen, []
    def record(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess,"Popen",record)
    monkeypatch.setattr(transport,"_SOURCE_RESOLVER_SCRIPT","import os\nwhile True: os.write(1,b'x'*8192)")
    with pytest.raises(ValueError,match="resolution_oversized"):
        transport._source_resolve_addresses(("localhost",443),deadline=time.monotonic()+2)
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed


def test_tcp_fallback_shares_deadline_and_closes_all_failed_sockets(transport,monkeypatch):
    clock, peers = [0.0], []
    class Peer:
        def __init__(self,*args):
            self.timeouts,self.closed = [],False
            peers.append(self)
        def settimeout(self,value):
            self.timeouts.append(value)
        def connect(self,address):
            clock[0] += 0.6
            if address[0] == "192.0.2.1":
                raise OSError("synthetic refusal")
        def close(self):
            self.closed = True
    monkeypatch.setattr(time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(socket,"socket",Peer)
    monkeypatch.setattr(transport,"_source_resolve_addresses",lambda *a,**kw:[
        (socket.AF_INET,socket.SOCK_STREAM,0,"",("192.0.2.1",443)),
        (socket.AF_INET,socket.SOCK_STREAM,0,"",("192.0.2.2",443))])
    with pytest.raises(TimeoutError):
        transport._source_create_deadline_connection(("synthetic.invalid",443),deadline=1)
    assert [peer.timeouts for peer in peers] == [[1],[0.4]] and all(peer.closed for peer in peers)


def test_expired_global_install_deadline_never_starts_resolver(transport,monkeypatch):
    monkeypatch.setattr(subprocess,"Popen",lambda *a,**kw:pytest.fail("expired installation started a resolver"))
    with pytest.raises(TimeoutError):
        transport._SourceDeadlineHTTPSConnection("synthetic.invalid",deadline=time.monotonic()-1,timeout=30).connect()
