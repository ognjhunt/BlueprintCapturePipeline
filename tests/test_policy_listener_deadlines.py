"""ADP release listeners bound pre-auth work without relaxing authority."""
from __future__ import annotations

import http.client
import json
import shutil
import socket
import ssl
import subprocess
import threading
import time

import pytest

from blueprint_pipeline import bounded_policy_https_server as bounded
from blueprint_pipeline import company_policy_sandbox_manager as manager
from blueprint_pipeline.controlled_policy_bridge_server import QualifiedPolicyBridge
from tests.test_controlled_policy_remote_sandbox import _fixture


@pytest.fixture(scope='module')
def tls_identity(tmp_path_factory):
    if shutil.which('openssl') is None:
        pytest.skip('openssl unavailable')
    root = tmp_path_factory.mktemp('synthetic-listener-tls')
    certificate, private_key = root / 'cert.pem', root / 'key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
        '-keyout', str(private_key), '-out', str(certificate), '-days', '1',
        '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost'],
        check=True, capture_output=True, timeout=15)
    return certificate, private_key


def _wait(predicate, seconds=2):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


@pytest.fixture(params=['bridge', 'manager'])
def listener(request, tls_identity, tmp_path, monkeypatch):
    certificate, private_key = tls_identity
    monkeypatch.setattr(bounded, 'HANDSHAKE_TIMEOUT_SECONDS', 1.0)
    monkeypatch.setattr(bounded, 'REQUEST_READ_TIMEOUT_SECONDS', 0.4)
    monkeypatch.setattr(bounded, 'MAXIMUM_CONNECTIONS', 2)
    calls = []
    if request.param == 'bridge':
        contract, job, _ = _fixture()
        bridge = QualifiedPolicyBridge(contract=contract, job_request=job,
            endpoint_url='https://localhost/v1/controlled-policy', bearer_token='a' * 40,
            manifest_path=tmp_path / 'manifest.json', bind_host='127.0.0.1', bind_port=0,
            tls_certificate=certificate, tls_private_key=private_key)
        server = bridge.server
        def target():
            server.serve_forever(poll_interval=0.02)
        route, body, success = '/v1/controlled-policy/ready', bridge.binding, 409
    else:
        class SyntheticManager:
            def __init__(self, *args, **kwargs):
                pass

            def prepare(self, body):
                calls.append(body)
                return {'status': 'synthetic_prepared'}

        captured = []
        class CapturedServer(bounded.BoundedPolicyHTTPSServer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured.append(self)

        monkeypatch.setattr(manager, 'SandboxManager', SyntheticManager)
        monkeypatch.setattr(manager, 'BoundedPolicyHTTPSServer', CapturedServer)
        def target():
            manager.serve_manager(settings={'bind_host': '127.0.0.1', 'bind_port': 0},
                token='a' * 40, certificate=certificate, private_key=private_key)
        route, body, success = '/v1/sessions', {}, 200
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    if request.param == 'manager':
        _wait(lambda: bool(captured))
        server = captured[0]
    context = ssl.create_default_context(cafile=str(certificate))
    value = {'kind': request.param, 'server': server, 'context': context,
        'port': server.server_address[1], 'route': route, 'body': body,
        'success': success, 'calls': calls}
    try:
        yield value
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()
        _wait(lambda: not server._connections)


def _post(listener, token='a' * 40):
    connection = http.client.HTTPSConnection('localhost', listener['port'],
        context=listener['context'], timeout=2)
    try:
        connection.request('POST', listener['route'], body=json.dumps(listener['body']),
            headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_stalled_tls_handshake_does_not_block_other_requests(listener):
    stalled = socket.create_connection(('127.0.0.1', listener['port']), timeout=2)
    try:
        _wait(lambda: len(listener['server']._connections) == 1)
        assert _post(listener, token='wrong')[0] == 401
        # The legitimate request was handled while the first TLS peer remains.
        assert listener['server']._connections
        _wait(lambda: not listener['server']._connections)
        assert _post(listener)[0] == listener['success']
    finally:
        stalled.close()


def test_connection_cap_refuses_extra_handshake_and_releases_after_timeout(listener):
    peers = [socket.create_connection(('127.0.0.1', listener['port']), timeout=2) for _ in range(2)]
    try:
        _wait(lambda: len(listener['server']._connections) == 2)
        extra = socket.create_connection(('127.0.0.1', listener['port']), timeout=2)
        try:
            extra.settimeout(0.4)
            assert extra.recv(1) == b''
        finally:
            extra.close()
        assert len(listener['server']._connections) == 2
        _wait(lambda: not listener['server']._connections)
        assert _post(listener)[0] == listener['success']
    finally:
        for peer in peers:
            peer.close()


@pytest.mark.parametrize('part', ['headers', 'body'])
def test_progressive_slow_http_reads_share_one_deadline(listener, part):
    raw = socket.create_connection(('127.0.0.1', listener['port']), timeout=2)
    connection = listener['context'].wrap_socket(raw, server_hostname='localhost')
    try:
        start = time.monotonic()
        prefix = f"POST {listener['route']} HTTP/1.1\r\nHost: localhost\r\n"
        if part == 'body':
            prefix += ('Authorization: Bearer ' + 'a' * 40 + '\r\n'
                'Content-Type: application/json\r\nContent-Length: 200\r\n\r\n{')
        connection.sendall(prefix.encode())
        for _ in range(40):
            try:
                connection.sendall(b' ')
            except OSError:
                break
            time.sleep(0.025)
            if not listener['server']._connections:
                break
        _wait(lambda: not listener['server']._connections)
        assert time.monotonic() - start < 0.9
        assert listener['calls'] == []  # No manager mutation from partial input.
        assert _post(listener)[0] == listener['success']
    finally:
        connection.close()


def test_listener_closure_cancels_incomplete_tls_and_returns_admission_slot(listener):
    stalled = socket.create_connection(('127.0.0.1', listener['port']), timeout=2)
    try:
        _wait(lambda: bool(listener['server']._connections))
        listener['server'].server_close()
        _wait(lambda: not listener['server']._connections)
        assert listener['server']._slots.acquire(blocking=False)
        listener['server']._slots.release()
    finally:
        stalled.close()


def test_deadline_reader_does_not_reset_for_progress(monkeypatch):
    now = [10.0]
    class Connection:
        def __init__(self):
            self.timeouts = []

        def settimeout(self, seconds):
            self.timeouts.append(seconds)

        def recv_into(self, buffer):
            buffer[0] = 32
            return 1

    monkeypatch.setattr(bounded.time, 'monotonic', lambda: now[0])
    connection = Connection()
    reader = bounded._DeadlineReader(connection, 10)
    assert reader.readinto(bytearray(2)) == 1
    now[0] = 15.0
    assert reader.readinto(bytearray(2)) == 1
    now[0] = 20.0
    with pytest.raises(TimeoutError, match='request_read_deadline'):
        reader.readinto(bytearray(2))
    assert connection.timeouts == [10.0, 5.0]
