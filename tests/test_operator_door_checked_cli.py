"""Checked CLI routing and bounded parse errors; all transport seams are fake."""

# Covers (for impacted-test selection):
#   scripts/operator_door.py

from __future__ import annotations

import hashlib
import io
import json

import pytest

from scripts import operator_door as client


def _flags(payload=b'ab'):
    return ['--expected-sha256', 'sha256:'+hashlib.sha256(payload).hexdigest(), '--expected-size', str(len(payload))]


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {'X-Door-Size': str(len(data)), 'X-Door-Offset':'0',
                        'X-Door-Length':str(len(data)), 'X-Door-Eof':'true'}
    def read(self, size):
        assert 0 < size <= 3
        return super().read(size)


def test_checked_cli_uses_file_read_without_listing_or_archive(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('checked pull must not list or choose ordinary/archive pull')
    for name in ('_json','_pull_file','_pull_directory'):
        monkeypatch.setattr(client, name, forbidden)
    calls = []
    def request(method, route, query, **kwargs):
        calls.append((method, route, query, kwargs))
        return Response(b'ab')
    monkeypatch.setattr(client, '_request', request)
    target = tmp_path/'artifact.bin'
    assert client.main(['pull','/private/census',str(target),*_flags()]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['verified_digest'] == _flags()[1] and result['verified_bytes'] == 2
    assert target.read_bytes() == b'ab'
    assert calls == [('GET','/fs/read',{'path':'/private/census','offset':0,'length':2},
                      {'checked_error_max_bytes':4096})]


@pytest.mark.parametrize('options', [
    [], ['--expected-sha256','bad'], ['--expected-size','1'],
    ['--expected-sha256','sha256:'+'a'*64,'--expected-size','01'],
    ['--expected-sha256','sha256:'+'a'*64,'--expected-size','PRIVATE_TEST_MARKER']])
def test_checked_pair_and_malformed_options_refuse_before_output_or_requests(tmp_path, monkeypatch, options, capsys):
    if not options:
        options = ['--expected-sha256','sha256:'+'a'*64]
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid checked options must not request or create output')
    monkeypatch.setattr(client,'_request',forbidden)
    monkeypatch.setattr(client,'_json',forbidden)
    target = tmp_path/'not-created'/'artifact'
    assert client.main(['pull','/private/census',str(target),*options]) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and captured.err == 'checked_pull_options_invalid\n'
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('argv', [
    ['pull','PRIVATE_TEST_MARKER','local','--expected-size'],
    ['pull','PRIVATE_TEST_MARKER','local','--expected-sha256'],
    ['pull','PRIVATE_TEST_MARKER','local','--expected-sh'],
    ['pull','PRIVATE_TEST_MARKER','local','--expected-si'],
    ['pull','PRIVATE_TEST_MARKER','local','--expected-s=1'],
    ['pull','PRIVATE_TEST_MARKER','local','--expected-size=1','--unexpected=PRIVATE_TEST_MARKER'],
    pytest.param(['pull','PRIVATE_TEST_MARKER','local','--expected-size=1','--unexpected='+'PRIVATE_TEST_MARKER'*400],id='long_unknown'),
    ['pull','PRIVATE_TEST_MARKER','--expected-sha256=bad'],
])
def test_checked_parse_errors_are_bounded_typed_and_never_echo_argv(argv, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('parse errors must not request or create output')
    monkeypatch.setattr(client,'run',forbidden)
    assert client.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and captured.err == 'checked_pull_options_invalid\n'
    assert len(captured.err.encode()) < 4096


def test_valid_abbreviations_and_equals_are_checked_options(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(client, '_request', lambda *args, **kwargs: Response(b'ab'))
    target = tmp_path/'artifact'
    assert client.main(['pull','/census',str(target),'--expected-sh='+_flags()[1],'--expected-si=2']) == 0
    assert json.loads(capsys.readouterr().out)['verified_bytes'] == 2
    assert target.read_bytes() == b'ab'


@pytest.mark.parametrize('kind', ['file','dir'])
def test_ordinary_pull_routing_and_result_shape_remain_compatible(tmp_path, monkeypatch, capsys, kind):
    monkeypatch.setattr(client,'_json',lambda *args, **kwargs: {'type':kind})
    def checked(*args, **kwargs):
        raise AssertionError('ordinary pull must not select checked path')
    monkeypatch.setattr(client,'_pull_checked_file',checked)
    target = tmp_path/'ordinary'
    if kind == 'file':
        monkeypatch.setattr(client,'_pull_file',lambda remote,local: local.write_bytes(b'plain'))
    else:
        monkeypatch.setattr(client,'_pull_directory',lambda remote,local: {'skipped':[], 'manifest':'ordinary'})
    assert client.main(['pull','/ordinary',str(target)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == ({'path':'/ordinary','saved':str(target),'bytes':5} if kind == 'file'
                      else {'skipped':[], 'manifest':'ordinary'})


def test_ordinary_parse_failure_remains_argparse_exit_and_message(capsys):
    with pytest.raises(SystemExit) as raised:
        client.main(['cat','/file','--offset','not-a-number'])
    assert raised.value.code == 2
    assert 'invalid int value' in capsys.readouterr().err
