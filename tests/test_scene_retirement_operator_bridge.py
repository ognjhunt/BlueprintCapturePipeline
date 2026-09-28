"""Actual door selectors and native bounded transport; no provider requests."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_cli.py
#   deploy/operator-door/operator_door/requests.py
#   deploy/operator-door/operator_door/spool_runner.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
#   scripts/operator_door.py
#   deploy/operator-door/door-scene-lifecycle.sh
#   deploy/operator-door/install.sh

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


@pytest.mark.parametrize('kind', ['retire-scene', 'restore-scene'])
def test_actual_spool_runner_launches_fixed_scene_script_with_no_public_paths(tmp_path, kind):
    from operator_door.config import DoorConfig
    from operator_door.requests import enqueue, validate_request
    from operator_door.spool_runner import process_spool
    from tests.test_operator_door_runner import FakeRunner
    for state in ('pending','processing','completed','results'):
        (tmp_path/'requests'/state).mkdir(parents=True)
    config = DoorConfig(state_root=str(tmp_path))
    body=request(kind)
    enqueue(config,validate_request(body),requested_by='operator')
    runner=FakeRunner()
    process_spool(config,runner=runner)
    argv=next(call for call in runner.calls if call[0]=='systemd-run')
    assert argv[-1] == config.install_root+'/door-scene-lifecycle.sh'
    assert '--setenv=DOOR_SCENE_ACTION='+('retire' if kind=='retire-scene' else 'restore') in argv
    assert '--setenv=DOOR_SCENE_INTENT_ID=scene-1' in argv
    assert '--setenv=DOOR_SCENE_CONSENT_ID='+'1'*32 in argv
    assert '--property=RuntimeMaxSec=2h' in argv
    assert '--property=ProtectSystem=strict' in argv
    assert any(x.startswith('--property=ReadWritePaths=') and '/var/lib/blueprint/scene-retirement' in x for x in argv)
    assert not any('DOOR_PLAN_PATH=' in x or 'DOOR_POLICY_PATH=' in x or 'DOOR_BUCKET=' in x for x in argv)


@pytest.mark.parametrize('action', ['retire-scene','restore-scene'])
def test_actual_client_submits_exact_selector_without_hidden_path_flags(monkeypatch,action):
    from tests.test_operator_door_client import client
    seen=[]
    monkeypatch.setattr(client,'_submit',lambda body,args:seen.append(body) or 0)
    argv=[action,'scene-1','--consent-id','1'*32,'--expected-sha256','sha256:'+'a'*64,
          '--expected-size-bytes','101']
    assert client.main(argv)==0
    assert seen==[request(action)]


def test_installed_environment_reads_only_selected_literal_settings(tmp_path,monkeypatch):
    module=bridge()
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    monkeypatch.setattr(access,'_POLICY_UID',__import__('os').getuid())
    path=tmp_path/'environment'
    path.write_text('LD_PRELOAD=/evil\nPATH=/evil\nBLUEPRINT_SCENE_RETIREMENT_POLICY_FILE="/etc/blueprint/policy.json"\n')
    path.chmod(0o640)
    monkeypatch.setattr(module,'ENVIRONMENT_FILE',path)
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE',raising=False)
    oldpath=__import__('os').environ['PATH']
    module.load_installed_environment()
    assert __import__('os').environ['PATH']==oldpath
    assert __import__('os').environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE']=='/etc/blueprint/policy.json'


def test_installed_environment_duplicate_selection_refuses_before_partial_export(tmp_path,monkeypatch):
    module=bridge()
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    monkeypatch.setattr(access,'_POLICY_UID',__import__('os').getuid())
    path=tmp_path/'environment'
    path.write_text('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE=/one\nBLUEPRINT_SCENE_RETIREMENT_POLICY_FILE=/two\n')
    path.chmod(0o640)
    monkeypatch.setattr(module,'ENVIRONMENT_FILE',path)
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE',raising=False)
    with pytest.raises(ValueError):
        module.load_installed_environment()
    assert 'BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE' not in __import__('os').environ


def test_installer_stages_the_actual_whole_scene_script():
    root=Path(__file__).resolve().parents[1]
    assert '"$source_dir"/door-scene-lifecycle.sh' in (root/'deploy/operator-door/install.sh').read_text()
    source=(root/'deploy/operator-door/door-scene-lifecycle.sh').read_text()
    assert 'blueprint_pipeline.task_evaluation_scene_retirement_cli' in source
    assert 'DOOR_PLAN_PATH' not in source and 'DOOR_POLICY_PATH' not in source


def test_native_charged_read_spends_same_allowance_before_body_read():
    module=bridge()
    client=Client()
    budget=allowance()
    client.data=b'abcdef'
    seen=[]
    class Body(io.BytesIO):
        def read(self,size=-1):
            if size!=1:
                seen.append(budget.counts['remote_bytes'])
            return super().read(size)
    client.get_object=lambda **kw:dict(Body=Body(client.data),ContentLength=6)
    transport=module.SceneArchiveTransport(client=client,bucket='private-artifacts')
    transport.bind_allowance(budget)
    uri='s3://private-artifacts/blueprint/arm-decision-proof-v1/scene-retirement/'+'1'*32+'.tar'
    assert b''.join(transport.read_archive_charged(uri,budget))==b'abcdef'
    assert seen==[6] and budget.counts['remote_bytes']==6
    with pytest.raises(ValueError):
        list(transport.read_archive_charged(uri,allowance()))


def test_deadline_after_create_aborts_retained_owned_upload_before_parts():
    module=bridge()
    client=Client()
    clock=[100]
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
    budget=ActionAllowance(expires_at=200,now=lambda:clock[0],monotonic=lambda:0)
    original=client.create_multipart_upload
    def create(**kw):
        result=original(**kw)
        clock[0]=201
        return result
    client.create_multipart_upload=create
    transport=module.SceneArchiveTransport(client=client,bucket='private-artifacts')
    transport.bind_allowance(budget)
    with pytest.raises(ValueError):
        transport.put_archive('1'*32+'.tar',iter([b'x']))
    assert client.calls==['create','abort']


def test_native_sdk_failure_is_typed_without_provider_or_secret_text():
    module=bridge()
    client=Client()
    def get(**kw):
        raise RuntimeError('secret=private-token /private/host/path')
    client.get_object=get
    transport=module.SceneArchiveTransport(client=client,bucket='private-artifacts')
    transport.bind_allowance(allowance())
    uri='s3://private-artifacts/blueprint/arm-decision-proof-v1/scene-retirement/'+'1'*32+'.tar'
    with pytest.raises(ValueError,match='^scene_retirement_transport_failure$'):
        list(transport.read_archive(uri))


@pytest.mark.slow
def test_actual_installed_script_invokes_exact_engine_cli_and_writes_sanitized_outcome(tmp_path):
    import os
    import shutil
    import subprocess
    root=Path(__file__).resolve().parents[1]
    scripts=tmp_path/'installed'
    scripts.mkdir()
    for name in ('door-common.sh','door-scene-lifecycle.sh'):
        shutil.copyfile(root/'deploy/operator-door'/name,scripts/name)
    executable=tmp_path/'python'
    executable.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
        'with open(os.environ["TEST_ARGS"],"w") as f: json.dump(sys.argv[1:],f)\n'
        'print(json.dumps({"status":"planned","mutations":0}))\n')
    executable.chmod(0o700)
    resultdir=tmp_path/'results'
    resultdir.mkdir()
    identity='20260928T180000Z-retire-scene-12345678'
    environment=dict(os.environ,DOOR_REQUEST_ID=identity,DOOR_RESULTS_DIR=str(resultdir),
        DOOR_SCENE_ACTION='retire',DOOR_SCENE_INTENT_ID='scene-1',DOOR_SCENE_CONSENT_ID='1'*32,
        DOOR_SCENE_CONSENT_SHA256='sha256:'+'a'*64,DOOR_SCENE_CONSENT_SIZE_BYTES='101',
        DOOR_SCENE_APPLY='0',DOOR_VENV_PYTHON=str(executable),DOOR_CONTROL_PLANE_REPO=str(root),
        TEST_ARGS=str(tmp_path/'args.json'))
    result=subprocess.run(['bash',str(scripts/'door-scene-lifecycle.sh')],env=environment,
                          capture_output=True,text=True,timeout=10)
    assert result.returncode==0, result.stderr+'\n'+(resultdir/(identity+'.log')).read_text()
    import json
    args=json.loads((tmp_path/'args.json').read_text())
    assert args[:3]==['-m','blueprint_pipeline.task_evaluation_scene_retirement_cli','retire']
    assert '--apply' not in args and '--intent-id' in args and '--consent-id' in args
    outcome=json.loads((resultdir/(identity+'.outcome.json')).read_text())
    assert outcome['status']=='planned' and outcome['intent_id']=='scene-1'


def test_client_close_failure_retains_completed_native_result_instead_of_zero_mutations(tmp_path,monkeypatch):
    module,_,_,_,values=selected_fixture(tmp_path,monkeypatch)
    # Isolate finalization; this is not a successful whole-scene action fixture.
    monkeypatch.setattr(module.engine,'retire_scene',lambda *a,**kw:
        dict(status='retired',intent_id='scene-1',removed_allocated_bytes=4096,logical_bytes=3))
    class Transport:
        def close(self):
            raise ValueError('secret=not-public')
    result=module.run_selected_action('retire',**values,apply=True,now=lambda:100,
                                     transport_factory=lambda:Transport())
    assert result['status']=='incomplete'
    assert result['removed_allocated_bytes']==4096 and result['logical_bytes']==3
    assert result.get('mutations')!=0 and 'not-public' not in str(result)


def test_native_response_close_fault_cannot_claim_success_and_is_typed():
    module=bridge()
    client=Client()
    class Body(io.BytesIO):
        def close(self):
            super().close()
            raise RuntimeError('secret=do-not-log')
    client.get_object=lambda **kw:dict(Body=Body(b'x'),ContentLength=1)
    transport=module.SceneArchiveTransport(client=client,bucket='private-artifacts')
    transport.bind_allowance(allowance())
    uri='s3://private-artifacts/blueprint/arm-decision-proof-v1/scene-retirement/'+'1'*32+'.tar'
    with pytest.raises(ValueError,match='^scene_retirement_remote_cleanup_unproven$'):
        list(transport.read_archive(uri))
