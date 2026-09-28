"""Checked file pulls verify trusted bytes using tiny fake binary responses only."""

# Covers (for impacted-test selection):
#   scripts/operator_door.py

from __future__ import annotations

import hashlib
import io

import pytest

from scripts import operator_door as client


def _digest(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


class Response:
    def __init__(self, body, *, size, offset, length=None, eof=None, patch=None, fragment=2):
        self.body = io.BytesIO(body) if isinstance(body, bytes) else body
        self.headers = {'X-Door-Size': str(size), 'X-Door-Offset': str(offset),
                        'X-Door-Length': str(len(body) if length is None else length),
                        'X-Door-Eof': ('true' if offset + len(body) == size else 'false') if eof is None else eof}
        self.headers.update(patch or {})
        self.fragment = fragment
        self.bounds = []

    def read(self, size):
        assert type(size) is int and size > 0, 'unbounded body read'
        self.bounds.append(size)
        return self.body.read(min(size, self.fragment))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _requests(monkeypatch, payload, *, server_cap=3):
    calls = []
    responses = []

    def request(method, route, query, **kwargs):
        assert (method, route) == ('GET', '/fs/read')
        assert kwargs == {'checked_error_max_bytes': client.MAX_CHECKED_HTTP_ERROR_BYTES}
        calls.append(query.copy())
        offset, length = query['offset'], query['length']
        response = Response(payload[offset:offset+min(length, server_cap)], size=len(payload), offset=offset)
        responses.append(response)
        return response

    monkeypatch.setattr(client, '_request', request)
    return calls, responses


def _pull(tmp_path, payload, *, digest=None, size=None):
    destination = tmp_path / 'artifact.bin'
    report = client._pull_checked_file('/remote/census.json', destination,
        expected_sha256=_digest(payload) if digest is None else digest,
        expected_size=len(payload) if size is None else size)
    return destination, report


@pytest.mark.parametrize('payload,chunk,server_cap', [
    (b'', 4, 3), (b'\xff\x00a\nb', 8, 8), (b'\xff\x00abcdefgh', 4, 2), (b'abcd', 4, 4)])
def test_verified_binary_pull_handles_zero_single_and_server_limited_chunks(tmp_path, monkeypatch, payload, chunk, server_cap):
    monkeypatch.setattr(client, 'MAX_CHECKED_PULL_BYTES', 10, raising=False)
    monkeypatch.setattr(client, 'CHECKED_PULL_CHUNK_BYTES', chunk, raising=False)
    calls, responses = _requests(monkeypatch, payload, server_cap=server_cap)
    target, report = _pull(tmp_path, payload)
    assert target.read_bytes() == payload
    assert report == {'path': '/remote/census.json', 'saved': str(target), 'bytes': len(payload),
                      'verified_digest': _digest(payload), 'verified_bytes': len(payload)}
    assert calls[0]['offset'] == 0
    for call, response in zip(calls, responses):
        assert call['length'] == min(chunk, len(payload)-call['offset'])
        assert all(bound <= call['length']+1 for bound in response.bounds)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['artifact.bin']


@pytest.mark.parametrize('digest,size', [
    (None, 1), ('sha256:'+'a'*64, None), ('a'*64, 1), ('sha256:'+'A'*64, 1),
    ('sha256:'+'a'*63, 1), ('sha256:'+'g'*64, 1), ('sha256:'+'a'*64+'\n',1),
    ('sha256:'+'a'*64,True), ('sha256:'+'a'*64,1.0), ('sha256:'+'a'*64,-1),
    ('sha256:'+'a'*64,'01'), ('sha256:'+'a'*64,'+1'), ('sha256:'+'a'*64,' 1'),
    ('sha256:'+'a'*64,'1.0'), ('sha256:'+'a'*64,'١'), ('sha256:'+'a'*64,'9'*50),
    ('sha256:'+'a'*64,11)])
def test_bad_options_refuse_before_requests_or_local_creation(tmp_path, monkeypatch, digest, size):
    monkeypatch.setattr(client, 'MAX_CHECKED_PULL_BYTES', 10, raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('options must refuse before request or output')
    monkeypatch.setattr(client, '_request', forbidden)
    target = tmp_path / 'not-created' / 'artifact.bin'
    with pytest.raises(client.DoorError, match='^checked_pull_options_invalid$') as raised:
        client._pull_checked_file('/private/remote', target, expected_sha256=digest, expected_size=size)
    assert raised.value.exit_code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('patch,body,length,eof', [
    ({'X-Door-Size':None}, b'ab',2,'true'), ({'X-Door-Offset':None},b'ab',2,'true'),
    ({'X-Door-Length':None},b'ab',2,'true'), ({'X-Door-Eof':None},b'ab',2,'true'),
    ({'X-Door-Size':'02'},b'ab',2,'true'), ({'X-Door-Size':'+2'},b'ab',2,'true'),
    ({'X-Door-Size':' 2'},b'ab',2,'true'), ({'X-Door-Size':'2.0'},b'ab',2,'true'),
    ({'X-Door-Size':'٢'},b'ab',2,'true'), ({'X-Door-Size':'9'*100},b'ab',2,'true'),
    ({'X-Door-Size':'3'},b'ab',2,'true'), ({'X-Door-Offset':'1'},b'ab',2,'true'),
    ({},b'a',2,'true'), ({},b'abc',2,'true'), ({},b'ab',1,'true'),
    ({},b'ab',3,'true'), ({},b'',0,'false'), ({},b'a',1,'true'),
    ({},b'ab',2,'false'), ({},b'ab',2,'True')])
def test_inconsistent_transfer_refuses_preserves_destination_and_owned_temp_only(tmp_path, monkeypatch, patch, body, length, eof):
    target = tmp_path / 'artifact.bin'
    target.write_bytes(b'old')
    other = tmp_path / 'artifact.bin.partial'
    other.write_bytes(b'other operation')
    response = Response(body, size=2, offset=0, patch=patch, length=length, eof=eof)
    monkeypatch.setattr(client, '_request', lambda *args, **kwargs: response)
    with pytest.raises(client.DoorError, match='^checked_pull_response_invalid$') as raised:
        _pull(tmp_path, b'ab')
    assert raised.value.exit_code == 2
    assert target.read_bytes() == b'old' and other.read_bytes() == b'other operation'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['artifact.bin','artifact.bin.partial']


def test_request_count_bound_refuses_progress_loop_without_extra_request(tmp_path, monkeypatch):
    monkeypatch.setattr(client, 'MAX_CHECKED_PULL_REQUESTS', 2, raising=False)
    calls, _ = _requests(monkeypatch, b'abc', server_cap=1)
    with pytest.raises(client.DoorError, match='^checked_pull_request_limit$'):
        _pull(tmp_path, b'abc')
    assert len(calls) == 2 and list(tmp_path.iterdir()) == []


def test_same_size_mixed_version_refuses_trusted_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(client, 'CHECKED_PULL_CHUNK_BYTES', 2, raising=False)
    _requests(monkeypatch, b'abXY', server_cap=2)
    target = tmp_path / 'artifact.bin'
    target.write_bytes(b'old')
    with pytest.raises(client.DoorError, match='^checked_pull_identity_mismatch$'):
        _pull(tmp_path, b'abcd')
    assert target.read_bytes() == b'old' and len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize('where', ['request', 'body', 'publish', 'temp_create'])
def test_interruption_and_local_failure_preserve_old_destination(tmp_path, monkeypatch, where):
    target = tmp_path / 'artifact.bin'
    target.write_bytes(b'old')
    _requests(monkeypatch, b'ab')
    def failure(*args, **kwargs):
        raise OSError('PRIVATE_TEST_MARKER /private/url/path')
    if where == 'request':
        monkeypatch.setattr(client, '_request', failure)
    elif where == 'body':
        response = Response(b'ab', size=2, offset=0)
        response.read = failure
        monkeypatch.setattr(client, '_request', lambda *args, **kwargs: response)
    elif where == 'publish':
        monkeypatch.setattr(client.os, 'replace', failure)
    else:
        monkeypatch.setattr(client.tempfile, 'mkstemp', failure)
    with pytest.raises(client.DoorError) as raised:
        _pull(tmp_path, b'ab')
    assert str(raised.value) == ('checked_pull_network_failed' if where in ('request','body') else 'checked_pull_local_write_failed')
    assert raised.value.exit_code == (4 if where in ('request','body') else 2)
    assert target.read_bytes() == b'old' and len(list(tmp_path.iterdir())) == 1


def test_nonbinary_response_body_is_typed_refusal_without_decoding(tmp_path, monkeypatch):
    response = Response(b'ab', size=2, offset=0)
    response.body = io.StringIO('PRIVATE_TEST_MARKER')
    monkeypatch.setattr(client, '_request', lambda *args, **kwargs: response)
    target = tmp_path / 'artifact.bin'
    target.write_bytes(b'old')
    with pytest.raises(client.DoorError, match='^checked_pull_response_invalid$') as raised:
        _pull(tmp_path, b'ab')
    assert raised.value.exit_code == 2
    assert target.read_bytes() == b'old' and len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize('where', ['write', 'flush', 'close', 'fsync', 'fdopen'])
def test_temporary_stream_failure_is_local_preserves_destination_and_closes_descriptor(tmp_path, monkeypatch, where):
    import os
    _requests(monkeypatch, b'ab')
    target = tmp_path / 'artifact.bin'
    target.write_bytes(b'old')
    original_fdopen = client.os.fdopen
    descriptors = []

    def fail(*args, **kwargs):
        raise OSError('PRIVATE_TEST_MARKER /private/disk/path')

    class Stream:
        def __init__(self, real):
            self.real = real
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.real.close()
            if where == 'close':
                fail()
        def write(self, data):
            return fail() if where == 'write' else self.real.write(data)
        def flush(self):
            return fail() if where == 'flush' else self.real.flush()
        def fileno(self):
            return self.real.fileno()

    def fdopen(descriptor, *args, **kwargs):
        descriptors.append(descriptor)
        if where == 'fdopen':
            fail()
        return Stream(original_fdopen(descriptor, *args, **kwargs))

    monkeypatch.setattr(client.os, 'fdopen', fdopen)
    if where == 'fsync':
        monkeypatch.setattr(client.os, 'fsync', fail)
    with pytest.raises(client.DoorError, match='^checked_pull_local_write_failed$') as raised:
        _pull(tmp_path, b'ab')
    assert raised.value.exit_code == 2
    assert target.read_bytes() == b'old' and len(list(tmp_path.iterdir())) == 1
    try:
        for descriptor in descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


def test_successful_publication_ends_temp_ownership_and_preserves_recreated_name(tmp_path, monkeypatch):
    from pathlib import Path
    _requests(monkeypatch, b'ab')
    destination = tmp_path / 'artifact.bin'
    destination.write_bytes(b'old')
    real_replace = client.os.replace
    real_unlink = Path.unlink
    former = []
    unlink_attempts = []

    def publish(source, target):
        real_replace(source, target)
        source = Path(source)
        former.append(source)
        source.write_bytes(b'new unrelated operation')

    def unlink(path, *args, **kwargs):
        if path in former:
            unlink_attempts.append(path)
            raise OSError('former temporary name is no longer owned')
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(client.os, 'replace', publish)
        patch.setattr(Path, 'unlink', unlink)
        target, report = _pull(tmp_path, b'ab')
    assert target.read_bytes() == b'ab' and report['verified_bytes'] == 2
    assert unlink_attempts == []
    assert former[0].read_bytes() == b'new unrelated operation'
