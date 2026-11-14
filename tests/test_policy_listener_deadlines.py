"""ADP release listeners bound pre-auth work without relaxing authority."""
from __future__ import annotations

import shutil
import subprocess
import time

import pytest

from blueprint_pipeline import bounded_policy_https_server as bounded


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
