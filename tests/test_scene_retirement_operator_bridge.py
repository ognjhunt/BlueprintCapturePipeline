"""Actual door selectors and native bounded transport; no provider requests."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_cli.py
#   deploy/operator-door/operator_door/requests.py
#   deploy/operator-door/operator_door/spool_runner.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
#   scripts/operator_door.py
#   deploy/operator-door/door-scene-lifecycle.sh
#   deploy/operator-door/install.sh

from contextlib import contextmanager
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


@pytest.fixture
def door_path():
    import tempfile
    with tempfile.TemporaryDirectory(prefix='.scene-door-test-',dir=Path.home()) as name:
        yield Path(name).resolve()


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
    # delenv alone records no undo when absent; the loader publishes directly.
    monkeypatch.setenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', '')
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


def test_installed_environment_test_restores_absent_policy_before_other_consumers(tmp_path, monkeypatch):
    import os
    key = 'BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE'
    # Track even an initially absent key so this RED cannot pollute its own run.
    monkeypatch.setenv(key, '')
    monkeypatch.delenv(key)
    with pytest.MonkeyPatch.context() as scoped:
        test_installed_environment_reads_only_selected_literal_settings(tmp_path, scoped)
    present = key in os.environ
    assert not present


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
    assert seen==[6] and budget.counts['remote_bytes']==7
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
@pytest.mark.parametrize('bootstrap_exit',[0,7])
def test_actual_installed_script_invokes_exact_engine_cli_and_writes_sanitized_outcome(door_path,bootstrap_exit):
    import os
    import shutil
    import subprocess
    tmp_path=door_path
    root=Path(__file__).resolve().parents[1]
    scripts=tmp_path/'installed'
    scripts.mkdir()
    for name in ('door-common.sh','door-scene-lifecycle.sh'):
        shutil.copyfile(root/'deploy/operator-door'/name,scripts/name)
    executable=tmp_path/'python'
    executable.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
        'with open(os.environ["TEST_ARGS"],"w") as f: json.dump(dict(args=sys.argv[1:],pid=os.getpid()),f)\n'
        'if int(os.environ["TEST_BOOTSTRAP_EXIT"]): raise SystemExit(int(os.environ["TEST_BOOTSTRAP_EXIT"]))\n'
        'from blueprint_pipeline import task_evaluation_scene_retirement_cli as module\n'
        'module.access._POLICY_UID=os.getuid()\n'
        'module._DOOR_CONFIG=module.Path(os.environ["TEST_CONFIG"])\n'
        'module.load_installed_environment=lambda: None\n'
        'module.run_selected_action=lambda *a,**kw: dict(status="planned",mutations=0)\n'
        'raise SystemExit(module.main(sys.argv[3:]))\n')
    executable.chmod(0o700)
    # Substitute only the compiled interpreter+installed bootstrap in this
    # portable shell fixture. Production accepts no interpreter override; the
    # separate Linux fixture proves the real root/ordinary-UID boundary.
    script=scripts/'door-scene-lifecycle.sh'
    source=script.read_text()
    fixed='/usr/bin/python3 -I -S /usr/lib/blueprint/scene-retirement-runtime/continuous_bootstrap.py'
    assert source.count(fixed)==1
    script.write_text(source.replace(fixed,str(executable)))
    resultdir=tmp_path/'requests/results'
    resultdir.mkdir(parents=True)
    config=tmp_path/'door.json'
    config.write_text(__import__('json').dumps(dict(state_root=str(tmp_path))))
    identity='20260928T180000Z-retire-scene-12345678'
    environment=dict(os.environ,DOOR_REQUEST_ID=identity,DOOR_RESULTS_DIR=str(resultdir),
        DOOR_SCENE_ACTION='retire',DOOR_SCENE_INTENT_ID='scene-1',DOOR_SCENE_CONSENT_ID='1'*32,
        DOOR_SCENE_CONSENT_SHA256='sha256:'+'a'*64,DOOR_SCENE_CONSENT_SIZE_BYTES='101',
        DOOR_SCENE_APPLY='0',DOOR_VENV_PYTHON=str(executable),DOOR_CONTROL_PLANE_REPO=str(root),
        TEST_ARGS=str(tmp_path/'args.json'),TEST_CONFIG=str(config),PYTHONPATH=str(root/'src'),
        TEST_BOOTSTRAP_EXIT=str(bootstrap_exit))
    with subprocess.Popen(['bash',str(scripts/'door-scene-lifecycle.sh')],env=environment,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) as process:
        _, stderr=process.communicate(timeout=10)
        worker_pid=process.pid
        assert process.returncode==bootstrap_exit, stderr+'\n'+(resultdir/(identity+'.log')).read_text()
    import json
    invocation=json.loads((tmp_path/'args.json').read_text())
    assert invocation['pid']==worker_pid, 'installed action retained a waiting shell parent'
    args=invocation['args']
    assert args[:3]==['--action-module','blueprint_pipeline.task_evaluation_scene_retirement_cli','retire']
    assert '--apply' not in args and '--intent-id' in args and '--consent-id' in args
    assert '--operator-door' in args
    if bootstrap_exit:
        # No parent guesses whether an absent CLI outcome means zero mutation
        # or success. The real request projection leaves terminal proof unknown.
        from operator_door.config import DoorConfig
        from operator_door.requests import request_state
        state=request_state(DoorConfig(state_root=str(tmp_path)),identity)
        assert state['outcome'] is None and state['result'] is None
        assert not (resultdir/(identity+'.scene-lifecycle.json')).exists()
        return
    outcome=json.loads((resultdir/(identity+'.outcome.json')).read_text())
    assert outcome['status']=='planned' and outcome['intent_id']=='scene-1'


@pytest.mark.parametrize('status,expected,rc',[
    ('planned','planned',0),('retired','retired',0),('restored','restored',0),
    ('kept','retained',0),('incomplete','failed',1)])
def test_door_worker_publishes_durable_public_counters_then_typed_outcome(
        door_path,monkeypatch,status,expected,rc):
    import json
    import os
    module=bridge()
    monkeypatch.setattr(module.access,'_POLICY_UID',os.getuid())
    config=door_path/'door.json'
    config.write_text(json.dumps(dict(state_root=str(door_path))))
    monkeypatch.setattr(module,'_DOOR_CONFIG',config)
    directory=door_path/'requests/results'
    directory.mkdir(parents=True)
    identity='20260928T180000Z-retire-scene-12345678'
    monkeypatch.setenv('DOOR_REQUEST_ID',identity)
    monkeypatch.setenv('DOOR_RESULTS_DIR',str(directory))
    monkeypatch.setattr(module,'load_installed_environment',lambda:None)
    monkeypatch.setattr(module,'run_selected_action',lambda *a,**kw:
        dict(status=status,mutations=None,removed_allocated_bytes=4096,
             private_consent='do-not-publish',remote_uri='private-object'))
    synced=[]
    original=module.os.fsync
    monkeypatch.setattr(module.os,'fsync',lambda fd:(synced.append(os.fstat(fd).st_ino),original(fd))[1])
    options=['retire','--intent-id','scene-1','--consent-id','1'*32,
             '--expected-sha256','sha256:'+'a'*64,'--expected-size-bytes','101','--operator-door']
    assert module.main(options)==rc
    public=json.loads((directory/(identity+'.scene-lifecycle.json')).read_text())
    outcome=json.loads((directory/(identity+'.outcome.json')).read_text())
    assert public==dict(status=status,mutations=None,removed_allocated_bytes=4096)
    assert outcome['status']==expected and outcome['exit_code']==rc
    assert len(synced)==4 and synced[1]==directory.stat().st_ino
    assert all(path.stat().st_nlink==1 for path in directory.iterdir())
    assert not list(directory.glob('*.tmp.*'))


@pytest.mark.parametrize('change',['wrong-kind','foreign-directory','linked-directory',
                                  'writable-directory','prior-result','prior-outcome','result-alias'])
def test_door_context_refuses_without_entering_action(door_path,monkeypatch,change):
    import json
    import os
    module=bridge()
    monkeypatch.setattr(module.access,'_POLICY_UID',os.getuid())
    config=door_path/'door.json'
    config.write_text(json.dumps(dict(state_root=str(door_path))))
    monkeypatch.setattr(module,'_DOOR_CONFIG',config)
    directory=door_path/'requests/results'
    directory.mkdir(parents=True)
    identity='20260928T180000Z-retire-scene-12345678'
    selected=directory
    if change=='wrong-kind':
        identity=identity.replace('retire-scene','restore-scene')
    elif change=='foreign-directory':
        selected=door_path
    elif change=='linked-directory':
        directory.rename(directory.with_name('original'))
        directory.symlink_to(directory.with_name('original'))
    elif change=='writable-directory':
        directory.chmod(0o777)
    elif change=='result-alias':
        (directory/(identity+'.scene-lifecycle.json')).symlink_to(config)
    elif change in {'prior-result','prior-outcome'}:
        suffix='.scene-lifecycle.json' if change=='prior-result' else '.outcome.json'
        (directory/(identity+suffix)).write_text('preserve-original')
    monkeypatch.setenv('DOOR_REQUEST_ID',identity)
    monkeypatch.setenv('DOOR_RESULTS_DIR',str(selected))
    calls=[]
    monkeypatch.setattr(module,'load_installed_environment',lambda:calls.append('environment'))
    monkeypatch.setattr(module,'run_selected_action',lambda *a,**kw:calls.append('action'))
    options=['retire','--intent-id','scene-1','--consent-id','1'*32,
             '--expected-sha256','sha256:'+'a'*64,'--expected-size-bytes','101','--operator-door']
    assert module.main(options)==1 and calls==[]


@pytest.mark.parametrize('change',['outcome-write-failure','directory-replaced','outcome-fsync-failure',
                                  'directory-replaced-during-fsync','temp-replaced-during-fsync'])
def test_door_publication_failure_retains_partial_action_counters(
        door_path,monkeypatch,capsys,change):
    import json
    import os
    module=bridge()
    monkeypatch.setattr(module.access,'_POLICY_UID',os.getuid())
    config=door_path/'door.json'
    config.write_text(json.dumps(dict(state_root=str(door_path))))
    monkeypatch.setattr(module,'_DOOR_CONFIG',config)
    directory=door_path/'requests/results'
    directory.mkdir(parents=True)
    identity='20260928T180000Z-retire-scene-12345678'
    monkeypatch.setenv('DOOR_REQUEST_ID',identity)
    monkeypatch.setenv('DOOR_RESULTS_DIR',str(directory))
    monkeypatch.setattr(module,'load_installed_environment',lambda:None)
    def action(*a,**kw):
        if change=='directory-replaced':
            directory.rename(directory.with_name('original'))
            directory.mkdir()
        return dict(status='retired',mutations=None,removed_allocated_bytes=4096)
    monkeypatch.setattr(module,'run_selected_action',action)
    original=module.os.link
    def link(source,destination,**kw):
        if destination.endswith('.outcome.json'):
            raise OSError('private-provider-secret')
        return original(source,destination,**kw)
    if change=='outcome-write-failure':
        monkeypatch.setattr(module.os,'link',link)
    elif change=='outcome-fsync-failure':
        fsync=module.os.fsync
        count=[0]
        def sync(fd):
            count[0]+=1
            if count[0]==4:
                raise OSError('private-provider-secret')
            return fsync(fd)
        monkeypatch.setattr(module.os,'fsync',sync)
    elif change in {'directory-replaced-during-fsync','temp-replaced-during-fsync'}:
        fsync=module.os.fsync
        replaced=[False]
        def sync(fd):
            fsync(fd)
            if not replaced[0] and os.fstat(fd).st_ino!=directory.stat().st_ino:
                replaced[0]=True
                if change=='directory-replaced-during-fsync':
                    directory.rename(directory.with_name('original'))
                    directory.mkdir()
                else:
                    temporary=next(directory.glob('*.tmp.*'))
                    temporary.unlink()
                    temporary.write_text('preserve-foreign-inode')
        monkeypatch.setattr(module.os,'fsync',sync)
    options=['retire','--intent-id','scene-1','--consent-id','1'*32,
             '--expected-sha256','sha256:'+'a'*64,'--expected-size-bytes','101','--operator-door']
    assert module.main(options)==1
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='incomplete' and result['removed_allocated_bytes']==4096
    assert result['mutations'] is None and 'private-provider-secret' not in str(result)
    assert not (directory/(identity+'.outcome.json')).exists()
    if change=='temp-replaced-during-fsync':
        assert next(directory.glob('*.tmp.*')).read_text()=='preserve-foreign-inode'
        assert not (directory/(identity+'.scene-lifecycle.json')).exists()
    else:
        assert not list(directory.glob('*.tmp.*'))
    if change=='directory-replaced-during-fsync':
        assert not list(directory.with_name('original').iterdir())


def test_door_writer_never_closes_an_unproven_borrowed_creation_fd(door_path,monkeypatch):
    import os
    module=bridge()
    monkeypatch.setattr(module.access,'_POLICY_UID',os.getuid())
    borrowed_path=door_path/'borrowed'
    borrowed_path.write_text('preserve-original')
    original=os.open
    borrowed=original(borrowed_path,os.O_RDONLY)
    parent=original(door_path,os.O_RDONLY|os.O_DIRECTORY)
    created=[]
    def substituted(name,flags,*a,**kw):
        fd=original(name,flags,*a,**kw)
        if str(name).endswith('.tmp.'+str(os.getpid())):
            created.append(fd)
            return borrowed
        return fd
    monkeypatch.setattr(module.os,'open',substituted)
    try:
        with pytest.raises(ValueError):
            module._door_publish(door_path,'20260928T180000Z-retire-scene-12345678',
                                 parent,'scene-1',dict(status='kept',mutations=0))
        assert os.fstat(borrowed).st_ino==borrowed_path.stat().st_ino
        assert borrowed_path.read_text()=='preserve-original'
        assert not list(door_path.glob('*.outcome.json'))
    finally:
        for fd in created+[parent,borrowed]:
            try:
                os.close(fd)
            except OSError:
                pass


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


def test_installed_environment_cannot_rebind_after_private_mode_proof(tmp_path, monkeypatch):
    module = bridge()
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    import os
    monkeypatch.setattr(access, '_POLICY_UID', os.getuid())
    path = tmp_path / 'environment'
    path.write_text('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE=/original\n')
    path.chmod(0o640)
    monkeypatch.setattr(module, 'ENVIRONMENT_FILE', path)
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', raising=False)
    original = access._opened
    acquisitions = []
    @contextmanager
    def replaced(selected, **kwargs):
        with original(selected, **kwargs) as result:
            yield result
        if Path(selected) == path:
            acquisitions.append(1)
            if len(acquisitions) == 1:
                replacement = path.with_name('replacement')
                replacement.write_text('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE=/unapproved\n')
                replacement.chmod(0o644)
                replacement.replace(path)
    monkeypatch.setattr(access, '_opened', replaced)
    with pytest.raises(access.SceneRetirementAccessError):
        module.load_installed_environment()
    assert 'BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE' not in os.environ



def test_publication_readback_uses_separate_blueprint_store_and_same_charged_origin(monkeypatch):
    module = bridge()
    private, published = Client(), Client()
    published.data = b'published'
    selected = []
    bodies = []
    budget = allowance()
    class Body(io.BytesIO):
        def read(self, size=-1):
            if size != 1:
                assert budget.counts['remote_bytes'] >= size
            return super().read(size)
    def get(**kwargs):
        selected.append(kwargs)
        body = Body(published.data)
        bodies.append(body)
        return {'Body': body, 'ContentLength': len(published.data)}
    published.get_object = get
    monkeypatch.setattr(module, 'installed_publication_client', lambda: published, raising=False)
    transport = module.SceneArchiveTransport(client=private, bucket='private-artifacts')
    transport.bind_allowance(budget)
    uri = 's3://blueprint/task-evaluation/production-inputs/scene-1/bundle_manifest.v1.json'
    assert b''.join(transport.read_published_object_charged(
        uri, budget, expected_size_bytes=len(published.data))) == published.data
    assert selected == [{'Bucket': 'blueprint', 'Key': uri.split('s3://blueprint/', 1)[1]}]
    assert private.calls == [] and bodies[0].closed
    assert budget.counts['remote_bytes'] == len(published.data)+1
    transport.close()
    assert private.calls == ['close'] and published.calls == ['close']


@pytest.mark.parametrize('uri', [
    's3://other/task-evaluation/production-inputs/ns/derived/file',
    's3://blueprint/task-evaluation/host-only-owner-sources/ns/file',
    's3://blueprint/task-evaluation/production-inputs/ns/source/raw.mp4',
    's3://blueprint/task-evaluation/production-inputs/ns/../raw',
    's3://blueprint/task-evaluation/production-inputs/ns/%2fraw',
    's3://blueprint/task-evaluation/production-inputs/ns/file?token=secret',
    's3://blueprint/task-evaluation/production-inputs/ns/file#fragment',
])
def test_publication_uri_cannot_expand_bucket_raw_source_or_namespace_authority(monkeypatch, uri):
    module = bridge()
    calls = []
    monkeypatch.setattr(module, 'installed_publication_client', lambda: calls.append(1), raising=False)
    private = Client()
    transport = module.SceneArchiveTransport(client=private, bucket='private-artifacts')
    budget = allowance()
    transport.bind_allowance(budget)
    with pytest.raises(ValueError):
        list(transport.read_published_object_charged(uri, budget, expected_size_bytes=1))
    assert calls == [] and private.calls == [] and budget.counts['remote_bytes'] == 0


def test_publication_declared_size_is_checked_before_allocating_body(monkeypatch):
    module = bridge()
    private, published = Client(), Client()
    body = io.BytesIO(b'oversized')
    published.get_object = lambda **kw: {'Body': body, 'ContentLength': 9}
    monkeypatch.setattr(module, 'installed_publication_client', lambda: published, raising=False)
    transport = module.SceneArchiveTransport(client=private, bucket='private-artifacts')
    budget = allowance()
    transport.bind_allowance(budget)
    with pytest.raises(ValueError):
        list(transport.read_published_object_charged(
            's3://blueprint/task-evaluation/production-inputs/ns/derived/file', budget,
            expected_size_bytes=1))
    assert body.closed and budget.counts['remote_bytes'] == 0
    transport.close()


def test_publication_origin_and_invalid_sdk_caps_refuse_before_object_read(monkeypatch):
    module = bridge()
    private, published = Client(), Client()
    published.meta.config.read_timeout = 60
    monkeypatch.setattr(module, 'installed_publication_client', lambda: published, raising=False)
    transport = module.SceneArchiveTransport(client=private, bucket='private-artifacts')
    budget = allowance()
    transport.bind_allowance(budget)
    uri = 's3://blueprint/task-evaluation/production-inputs/ns/derived/file'
    with pytest.raises(ValueError):
        list(transport.read_published_object_charged(uri, allowance(), expected_size_bytes=1))
    assert published.calls == []
    with pytest.raises(ValueError):
        list(transport.read_published_object_charged(uri, budget, expected_size_bytes=1))
    assert 'get' not in published.calls and published.calls == ['close']
    transport.close()



@pytest.mark.parametrize('kind', ['archive-charged', 'archive-un-charged', 'publication'])
def test_full_original_remote_cap_refuses_before_trailing_eof_probe(monkeypatch, kind):
    module = bridge()
    private, published = Client(), Client()
    budget = allowance(remote_bytes=6)
    reads = []
    class Body(io.BytesIO):
        def read(self, size=-1):
            reads.append((size, budget.counts['remote_bytes']))
            return super().read(size)
    body = Body(b'abcdefZ')
    selected = published if kind == 'publication' else private
    selected.get_object = lambda **kw: {'Body': body, 'ContentLength': 6}
    monkeypatch.setattr(module, 'installed_publication_client', lambda: published)
    transport = module.SceneArchiveTransport(client=private, bucket='private-artifacts')
    transport.bind_allowance(budget)
    archive = 's3://private-artifacts/blueprint/arm-decision-proof-v1/scene-retirement/'+'1'*32+'.tar'
    if kind == 'publication':
        chunks = transport.read_published_object_charged(
            's3://blueprint/task-evaluation/production-inputs/ns/derived/file', budget,
            expected_size_bytes=6)
    elif kind == 'archive-charged':
        chunks = transport.read_archive_charged(archive, budget)
    else:
        chunks = transport.read_archive(archive)
    with pytest.raises(ValueError):
        for chunk in chunks:
            if kind == 'archive-un-charged':
                # The legacy engine charges the yielded body payload itself.
                budget.charge('remote_bytes', len(chunk))
    assert [count for count, _ in reads] == [6]
    assert body.closed and budget.counts['remote_bytes'] == 6
    transport.close()
