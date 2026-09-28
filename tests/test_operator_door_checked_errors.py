"""Checked HTTP failure bodies and error text stay bounded without live requests."""

# Covers (for impacted-test selection):
#   scripts/operator_door.py

from __future__ import annotations

import io
import urllib.error

import pytest

from scripts import operator_door as client


class ErrorBody(io.BytesIO):
    def __init__(self, data, fragment=None):
        super().__init__(data)
        self.fragment = fragment
        self.bounds = []
        self.accepted = 0

    def read(self, size=-1):
        self.bounds.append(size)
        if self.fragment is not None and size > 0:
            size = min(size, self.fragment)
        data = super().read(size)
        self.accepted += len(data)
        return data


def _error_request(monkeypatch, status, body, *, fragment=None):
    stream = ErrorBody(body, fragment)
    error = urllib.error.HTTPError('https://PRIVATE_TEST_MARKER/private/path', status,
                                  'PRIVATE_TEST_MARKER', {}, stream)
    def refused(*args, **kwargs):
        raise error
    monkeypatch.setattr(client.urllib.request, 'urlopen', refused)
    monkeypatch.setattr(client, 'auth_headers', lambda: {})
    return stream


@pytest.mark.parametrize('status,body,exit_code,message', [
    (401,b'{"error":"PRIVATE_TEST_MARKER"}',3,'checked_pull_unauthorized'),
    (403,b'{"error":"scope_missing:PRIVATE_TEST_MARKER"}',3,'checked_pull_unauthorized'),
    (403,b'{"error":"PRIVATE_TEST_MARKER"}',2,'checked_pull_remote_refused'),
    (403,b'{"error":"scope_missing:private"',2,'checked_pull_remote_refused'),
    (403,b'not json PRIVATE_TEST_MARKER',2,'checked_pull_remote_refused'),
    (403,b'[]',2,'checked_pull_remote_refused'),
    (403,b'{"error":123}',2,'checked_pull_remote_refused'),
    pytest.param(403,b'{"error":"scope_missing:private","x":NaN}',2,'checked_pull_remote_refused',id='nonfinite_json'),
    pytest.param(403,b'['*1500+b']'*1500,2,'checked_pull_remote_refused',id='deep_json'),
    (404,b'PRIVATE_TEST_MARKER',2,'checked_pull_remote_refused'),
    (500,b'{"error":"PRIVATE_TEST_MARKER"}',5,'checked_pull_server_failed')])
@pytest.mark.parametrize('fragment', [None, 3])
def test_checked_http_errors_use_complete_bounded_json_and_fixed_classified_text(monkeypatch, status, body, exit_code, message, fragment):
    stream = _error_request(monkeypatch, status, body, fragment=fragment)
    with pytest.raises(client.DoorError) as raised:
        client._request('GET','/fs/read',{'path':'/private/host'}, checked_error_max_bytes=4096)
    assert raised.value.exit_code == exit_code and str(raised.value) == message
    assert stream.bounds[0] == 4097
    assert all(0 < bound <= 4097 for bound in stream.bounds)
    assert stream.accepted == len(body)


@pytest.mark.parametrize('status,code', [(401,3), (403,2), (500,5)])
def test_oversized_error_body_is_not_used_for_scope_classification(monkeypatch, status, code):
    monkeypatch.setattr(client, 'MAX_CHECKED_HTTP_ERROR_BYTES', 32, raising=False)
    body = b'{"error":"scope_missing:private"}' + b' '*100
    stream = _error_request(monkeypatch, status, body, fragment=5)
    with pytest.raises(client.DoorError) as raised:
        client._request('GET','/fs/read', checked_error_max_bytes=32)
    assert raised.value.exit_code == code
    assert stream.bounds[0] == 33 and stream.accepted == 33
    assert 'private' not in str(raised.value)


@pytest.mark.parametrize('error', [urllib.error.URLError('PRIVATE_TEST_MARKER /private/url'),
                                  OSError('PRIVATE_TEST_MARKER /private/url')])
def test_checked_network_errors_are_sanitized_and_keep_exit_four(monkeypatch, error):
    def refused(*args, **kwargs):
        raise error
    monkeypatch.setattr(client.urllib.request, 'urlopen', refused)
    monkeypatch.setattr(client, 'auth_headers', lambda: {})
    with pytest.raises(client.DoorError) as raised:
        client._request('GET','/fs/read', checked_error_max_bytes=4096)
    assert raised.value.exit_code == 4 and str(raised.value) == 'checked_pull_network_failed'


def test_ordinary_http_and_network_request_messages_remain_compatible(monkeypatch):
    stream = _error_request(monkeypatch, 403, b'{"error":"scope_missing:read"}')
    with pytest.raises(client.DoorError, match='door refused authorization: scope_missing:read') as raised:
        client._request('GET','/fs/read')
    assert raised.value.exit_code == 3 and stream.bounds == [-1]
    def refused(*args, **kwargs):
        raise urllib.error.URLError('ordinary-marker')
    monkeypatch.setattr(client.urllib.request, 'urlopen', refused)
    with pytest.raises(client.DoorError, match='ordinary-marker') as raised:
        client._request('GET','/fs/read')
    assert raised.value.exit_code == 4
