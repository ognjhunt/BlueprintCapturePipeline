"""Actual door selectors and native bounded transport; no provider requests."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_cli.py
#   deploy/operator-door/operator_door/requests.py
#   deploy/operator-door/operator_door/spool_runner.py
#   src/blueprint_pipeline/control_plane_storage_gc.py

import hashlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_scene_retirement_engine import action_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy/operator-door'))


def bridge():
    from blueprint_pipeline import task_evaluation_scene_retirement_cli
    return task_evaluation_scene_retirement_cli


def request(kind='retire-scene', **extra):
    value = dict(kind=kind, intent_id='scene-1', consent_id='1'*32,
                 expected_sha256='sha256:'+'a'*64, expected_size_bytes=101)
    if kind == 'retire-scene':
        value['apply'] = False
    return dict(value, **extra)


@pytest.mark.parametrize('kind', ['retire-scene', 'restore-scene'])
def test_door_whole_scene_selectors_have_operate_scope_and_fixed_fields(kind):
    from operator_door.requests import validate_request, required_scope
    assert validate_request(request(kind)) == request(kind)
    assert required_scope(kind) == 'operate'


@pytest.mark.parametrize('change', [dict(consent_id='../x'), dict(expected_size_bytes=True),
    dict(expected_sha256='a'*64), dict(intent_id='../scene'), dict(plan_path='/tmp/plan'),
    dict(policy_path='/tmp/policy'), dict(bucket='other'), dict(apply=1)])
def test_door_never_accepts_path_policy_or_cloud_authority(change):
    from operator_door.requests import validate_request, RequestRefused
    with pytest.raises(RequestRefused):
        validate_request(request(**change))


def selected_fixture(tmp_path, monkeypatch):
    _, member, plan, consent = action_fixture(tmp_path, monkeypatch)
    root = tmp_path/'consents'
    root.mkdir(mode=0o700)
    selected = root/('1'*32+'.json')
    consent.rename(selected)
    module = bridge()
    monkeypatch.setattr(module, 'CONSENT_ROOT', root)
    raw = selected.read_bytes()
    values = dict(intent_id='scene-1', consent_id='1'*32,
        expected_sha256='sha256:'+hashlib.sha256(raw).hexdigest(), expected_size_bytes=len(raw))
    return module, member, plan, selected, values


def test_fixed_selector_calls_actual_engine_and_cannot_clear_its_reference_refusal(tmp_path, monkeypatch):
    module, member, _, _, values = selected_fixture(tmp_path, monkeypatch)
    calls = []
    class Transport:
        def close(self):
            calls.append('closed')
    result = module.run_selected_action('retire', **values, apply=True, now=lambda:100,
                                       transport_factory=lambda:Transport())
    assert result['status'] == 'kept'
    assert result['reason'] == 'scene_retirement_installed_context_unproven'
    assert member.exists() and calls == ['closed']


@pytest.mark.parametrize('case', ['digest', 'intent', 'symlink', 'writable'])
def test_bad_selected_consent_never_constructs_transport(tmp_path, monkeypatch, case):
    module, member, _, selected, values = selected_fixture(tmp_path, monkeypatch)
    if case == 'digest':
        values['expected_sha256'] = 'sha256:'+'e'*64
    elif case == 'intent':
        values['intent_id'] = 'other-scene'
    elif case == 'writable':
        selected.chmod(0o666)
    else:
        original = selected.with_suffix('.original')
        selected.rename(original)
        selected.symlink_to(original)
    result = module.run_selected_action('retire', **values, apply=True, now=lambda:100,
        transport_factory=lambda:pytest.fail('bad consent constructed remote client'))
    assert result['status'] == 'kept' and result['mutations'] == 0 and member.exists()


class Client:
    def __init__(self):
        self.meta = SimpleNamespace(config=SimpleNamespace(connect_timeout=5, read_timeout=30,
            retries={'total_max_attempts':1}, max_pool_connections=4))
        self.calls, self.parts, self.data = [], [], b''
        self.body = None
    def create_multipart_upload(self, **kw):
        self.calls.append('create')
        return {'UploadId':'owned-upload'}
    def upload_part(self, **kw):
        assert kw['UploadId'] == 'owned-upload'
        self.calls.append('part')
        self.parts.append(kw['Body'])
        return {'ETag':'part-'+str(kw['PartNumber'])}
    def complete_multipart_upload(self, **kw):
        self.calls.append('complete')
        self.data = b''.join(self.parts)
    def abort_multipart_upload(self, **kw):
        self.calls.append('abort')
    def get_object(self, **kw):
        self.calls.append('get')
        self.body = io.BytesIO(self.data)
        return {'Body':self.body, 'ContentLength':len(self.data)}
    def close(self):
        self.calls.append('close')


def allowance(**kw):
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    return ActionAllowance(expires_at=200, now=lambda:100, monotonic=lambda:0, **kw)


def test_native_stream_upload_and_full_readback_close_response_and_client(monkeypatch):
    module = bridge()
    client = Client()
    transport = module.SceneArchiveTransport(client=client, bucket='private-artifacts')
    transport.bind_allowance(allowance())
    ref = transport.put_archive('1'*32+'.tar', iter([b'abc', b'def']))
    assert ref['sha256'] == 'sha256:'+hashlib.sha256(b'abcdef').hexdigest()
    assert ref['size_bytes'] == 6
    assert b''.join(transport.read_archive(ref['uri'])) == b'abcdef'
    assert client.body.closed
    transport.close()
    assert client.calls == ['create','part','complete','get','close']


def test_native_stream_failure_aborts_only_owned_upload():
    module = bridge()
    client = Client()
    transport = module.SceneArchiveTransport(client=client, bucket='private-artifacts')
    transport.bind_allowance(allowance())
    def broken():
        yield b'abc'
        raise ValueError('injected producer failure')
    with pytest.raises(ValueError):
        transport.put_archive('1'*32+'.tar', broken())
    assert client.calls == ['create','abort'] and client.data == b''
    transport.close()


@pytest.mark.parametrize('case', ['timeouts','binding','foreign-uri','no-binding'])
def test_remote_transport_refuses_unknown_timeout_origin_and_archive_scope(case):
    module = bridge()
    client = Client()
    if case == 'timeouts':
        client.meta.config.read_timeout = 60
        with pytest.raises(ValueError):
            module.SceneArchiveTransport(client=client, bucket='private-artifacts')
        assert client.calls == []
        return
    transport = module.SceneArchiveTransport(client=client, bucket='private-artifacts')
    if case != 'no-binding':
        transport.bind_allowance(allowance())
    with pytest.raises(ValueError):
        if case == 'binding':
            transport.bind_allowance(allowance())
        elif case == 'foreign-uri':
            list(transport.read_archive('s3://other/private.tar'))
        else:
            transport.put_archive('1'*32+'.tar', iter([b'x']))
    assert client.calls == []


def test_timer_default_off_never_reads_selection_or_builds_client(monkeypatch):
    module = bridge()
    monkeypatch.delenv('BLUEPRINT_CONTROL_PLANE_SCENE_RETIREMENT', raising=False)
    monkeypatch.setattr(module, 'GC_SELECTION', Path('/definitely/absent'))
    result = module.run_gc_phase(apply=True, now=lambda:100)
    assert result['status'] == 'disabled' and result['removed_allocated_bytes'] == 0


def test_existing_tick_invokes_same_scene_phase_without_caller_flag_or_path(monkeypatch):
    module = bridge()
    calls = []
    def phase(*, apply, now):
        calls.append((apply, now()))
        return {'status':'disabled', 'removed_allocated_bytes':0}
    monkeypatch.setattr(module, 'run_gc_phase', phase)
    from blueprint_pipeline.control_plane_storage_gc import run_storage_gc
    result = run_storage_gc(content_store_roots=(), derived_roots=(), queue_roots=(),
                            pins_root='/absent', now=lambda:100)
    assert result['scene_lifecycle']['status'] == 'disabled' and calls == [(False,100)]
