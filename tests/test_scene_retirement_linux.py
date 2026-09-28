# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_supervisor.py
#   scripts/scene_retirement_continuous_bootstrap.py
#   scripts/install_scene_retirement_runtime.py
"""Actual disposable Linux privilege boundary; a Mac skip is not proof.

This first case proves only default-off startup and the real UID/capability
drop. It issues no retirement grant and does not clear current readers.
"""
from __future__ import annotations

import grp
import ast
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

import pytest


_RUNTIME = Path('/mnt/blueprint-work/scene-retirement-runtime')
_BOOT = Path('/usr/lib/blueprint/scene-retirement-runtime')
_POLICY = Path('/etc/blueprint/scene-retirement-policy.json')


def _native(command, *, timeout=30):
    value = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
        text=True, timeout=timeout, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})
    assert len(value.stdout.encode()) <= 65536 and len(value.stderr.encode()) <= 4096
    assert value.returncode == 0, value.stderr
    return value.stdout


def _checkout_git(source, *arguments):
    # This hermetic job has already selected the exact immutable feature/lock
    # checkout. Root reads its data without a global or wildcard trust change.
    path = Path(source).resolve(strict=True)
    assert path.is_dir() and not Path(source).is_symlink()
    return ['/usr/bin/git', '--no-replace-objects', '-c', 'safe.directory='+str(path),
            '-C', str(path), *arguments]


def _copy_protected(source, target):
    shutil.copytree(source, target, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    for path in [target, *target.rglob('*')]:
        assert not path.is_symlink(), 'a source alias is not a protected runtime input'
        os.chown(path, 0, 0)
        path.chmod(0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644)


def _default_off_native_phase():
    assert sys.platform == 'linux' and os.getuid() == os.geteuid() == 0
    assert os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') == '1'
    assert Path('/run/systemd/system').is_dir()
    # Preserve every pre-existing installation. This case requires a clean
    # disposable VM and must not weaken the installed policy to gain admission.
    assert not _POLICY.exists() and not _POLICY.is_symlink()
    assert not _RUNTIME.exists() and not _RUNTIME.is_symlink()
    assert not _BOOT.exists() and not _BOOT.is_symlink()
    created_account = False
    try:
        account = pwd.getpwnam('blueprint')
    except KeyError:
        _native(['/usr/sbin/useradd', '--system', '--user-group', '--no-create-home',
                 '--shell', '/usr/sbin/nologin', 'blueprint'])
        created_account = True
        account = pwd.getpwnam('blueprint')
    assert account.pw_uid != 0 and account.pw_gid == grp.getgrnam('blueprint').gr_gid
    root = Path(tempfile.mkdtemp(prefix='blueprint-scene-native-', dir='/var/lib'))
    root.chmod(0o755)
    identity = (root.stat().st_dev, root.stat().st_ino)
    unit = 'blueprint-scene-native-default-' + uuid.uuid4().hex + '.service'
    unit_path = Path('/etc/systemd/system') / unit
    installed = {}
    try:
        source = Path(__file__).resolve().parents[1]
        sealed = root / 'source'
        sealed.mkdir(mode=0o755)
        for folder in ('src/blueprint_pipeline', 'scripts', 'deploy/systemd'):
            _copy_protected(source / folder, sealed / folder)
        dependencies = root / 'dependencies'
        dependencies.mkdir(mode=0o755)
        # Default-off startup never imports any dependency or daemon as root.
        # An empty real protected dependency namespace is sufficient here.
        receipt = json.loads(_native([
            '/usr/bin/python3', '-I', '-S', str(sealed / 'scripts/install_scene_retirement_runtime.py'),
            '--source', str(sealed), '--dependencies', str(dependencies),
        ], timeout=300))
        assert receipt['status'] == 'prepared'
        assert receipt['authority_issued'] is False and receipt['cleanup_enabled'] is False
        for path in (_RUNTIME, _BOOT):
            info = path.stat()
            installed[path] = (info.st_dev, info.st_ino)
            assert info.st_uid == 0 and not info.st_mode & 0o022
        assert not _POLICY.exists()
        probe = root / 'probe.py'
        probe.write_text('''import json, os, sys
status = dict(line.split(':', 1) for line in open('/proc/self/status') if ':' in line)
print(json.dumps({'uid': os.getuid(), 'euid': os.geteuid(), 'gid': os.getgid(),
 'groups': os.getgroups(), 'argv': sys.argv[1:],
 'kernel_uid': status['Uid'].split(), 'kernel_gid': status['Gid'].split(),
 'caps': {k: status[k].strip() for k in ('CapEff','CapPrm','CapInh','CapAmb')},
 'no_new_privs': status['NoNewPrivs'].strip()}, sort_keys=True), flush=True)
''')
        probe.chmod(0o644)
        executable = root / 'legacy-python'
        executable.write_text('#!/bin/sh\nexec /usr/bin/python3 -I -S ' + str(probe) + ' "$@"\n')
        executable.chmod(0o755)
        unit_path.write_text('[Unit]\nDescription=Disposable scene retirement default boundary\n'
            '[Service]\nType=oneshot\nUser=blueprint\nGroup=blueprint\n'
            'NoNewPrivileges=yes\nCapabilityBoundingSet=CAP_SETUID CAP_SETGID\n'
            'AmbientCapabilities=\nProtectHome=yes\nProtectSystem=strict\n'
            'Environment=BLUEPRINT_PIPELINE_REPO=' + str(root) + '\n'
            'Environment=BLUEPRINT_PIPELINE_PYTHON=' + str(executable) + '\n'
            'ExecStart=!/usr/bin/python3 -I -S ' + str(_BOOT / 'continuous_bootstrap.py')
            + ' --continuous-module blueprint_pipeline.live_pipeline_intake_service -- --native-probe\n'
            'StandardOutput=journal\nStandardError=journal\nTimeoutStartSec=60\n')
        unit_path.chmod(0o644)
        _native(['/usr/bin/systemctl', 'daemon-reload'])
        _native(['/usr/bin/systemctl', 'start', unit], timeout=90)
        facts = dict(line.split('=', 1) for line in _native([
            '/usr/bin/systemctl', 'show', unit, '--no-pager', '--all',
            '--property=Id,LoadState,ActiveState,SubState,Result,ExecMainStatus',
        ]).splitlines())
        assert facts == dict(Id=unit, LoadState='loaded', ActiveState='inactive',
                             SubState='dead', Result='success', ExecMainStatus='0')
        journal = _native(['/usr/bin/journalctl', '--unit=' + unit, '--no-pager',
                           '--lines=40', '--output=cat', '--quiet'])
        rows = [json.loads(line) for line in journal.splitlines() if line.startswith('{')]
        assert len(rows) == 1, journal
        observed = rows[0]
        assert observed['uid'] == observed['euid'] == account.pw_uid
        assert observed['gid'] == account.pw_gid and observed['groups'] == []
        assert observed['kernel_uid'] == [str(account.pw_uid)] * 4
        assert observed['kernel_gid'] == [str(account.pw_gid)] * 4
        assert all(int(value, 16) == 0 for value in observed['caps'].values())
        assert observed['no_new_privs'] == '1'
        assert observed['argv'] == ['-m', 'blueprint_pipeline.live_pipeline_intake_service', '--native-probe']
        assert not _POLICY.exists()
        print(json.dumps(dict(status='passed', actual_uid=account.pw_uid,
             actual_systemd=True, default_off=True, authority_issued=False,
             kernel_caps_zero=True, args_preserved=True), sort_keys=True), flush=True)
    finally:
        if unit_path.exists():
            _native(['/usr/bin/systemctl', 'stop', unit])
            unit_path.unlink()
            _native(['/usr/bin/systemctl', 'daemon-reload'])
        for path, original in installed.items():
            assert (path.stat().st_dev, path.stat().st_ino) == original
            shutil.rmtree(path)
        assert (root.stat().st_dev, root.stat().st_ino) == identity
        shutil.rmtree(root)
        if created_account:
            _native(['/usr/sbin/userdel', 'blueprint'])


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != 'linux' or os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') != '1',
                   reason='mandatory actual disposable Linux/systemd UID proof; Mac skip is unmet')
def test_actual_default_off_root_bootstrap_drops_blueprint_uid_and_all_caps():
    command = [sys.executable, str(Path(__file__).resolve()), '--native-default-off']
    if os.geteuid() != 0:
        command = ['sudo', '-n', 'env', 'BLUEPRINT_DISPOSABLE_LINUX_TEST=1',
                   'PYTHONDONTWRITEBYTECODE=1', *command]
    value = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=480, cwd=Path(__file__).parents[1])
    assert len(value.stdout.encode()) <= 65536 and len(value.stderr.encode()) <= 65536
    assert value.returncode == 0, value.stdout + value.stderr
    receipt = json.loads(value.stdout.strip().splitlines()[-1])
    assert receipt['status'] == 'passed' and receipt['actual_uid'] != 0
    assert receipt['actual_systemd'] and receipt['kernel_caps_zero'] and receipt['args_preserved']
    assert receipt['default_off'] and receipt['authority_issued'] is False


def _enabled_sdk_native_phase():
    """Actual ABI/locked artifacts, rolling selector and ordinary-UID imports."""
    assert sys.platform == 'linux' and os.getuid() == os.geteuid() == 0
    assert os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') == '1'
    assert Path('/run/systemd/system').is_dir()
    source = Path(os.environ['BLUEPRINT_NATIVE_SOURCE_ROOT'])
    commit = os.environ['BLUEPRINT_NATIVE_SOURCE_COMMIT']
    contracts_input = Path(os.environ['BLUEPRINT_NATIVE_CONTRACTS_ROOT'])
    assert _native(_checkout_git(source, 'rev-parse', 'HEAD')).strip() == commit
    assert _native(_checkout_git(contracts_input, 'rev-parse', 'HEAD')).strip() == '7708a4e4c5dedeeb39cc73d3f6869304de295b81'
    sdk_inputs = _RUNTIME.with_name(_RUNTIME.name+'-sdk-inputs')
    unit = 'blueprint-pipeline-intake.service'
    unit_path = Path('/etc/systemd/system') / unit
    environment = Path('/etc/blueprint/pipeline-control-plane.env')
    for path in (_RUNTIME, _BOOT, sdk_inputs, _POLICY, unit_path, environment):
        assert not path.exists() and not path.is_symlink(), 'preserve pre-existing native installations'
    created_account = False
    try:
        account = pwd.getpwnam('blueprint')
    except KeyError:
        _native(['/usr/sbin/useradd', '--system', '--user-group', '--no-create-home', '--shell', '/usr/sbin/nologin', 'blueprint'])
        account = pwd.getpwnam('blueprint')
        created_account = True
    assert account.pw_uid != 0
    base = Path('/var/lib/blueprint')
    base_created = not base.exists()
    base.mkdir(mode=0o755, exist_ok=True)
    assert base.stat().st_uid == 0 and not base.stat().st_mode & 0o022
    retirement_state=base/'scene-retirement'
    assert not retirement_state.exists() and not retirement_state.is_symlink()
    retirement_state.mkdir(mode=0o755)
    (retirement_state/'journals').mkdir(mode=0o700)
    (retirement_state/'journals/processes').mkdir(mode=0o700)
    retirement_identity=(retirement_state.stat().st_dev,retirement_state.stat().st_ino)
    root = Path(tempfile.mkdtemp(prefix='scene-native-enabled-', dir=base))
    root.chmod(0o755)
    original = (root.stat().st_dev,root.stat().st_ino)
    installed = {}
    try:
        # The private checkout token exists only in the preceding CI checkout
        # step. Root receives protected Git object data, with no token/env/key.
        contracts = root/'contracts'
        contracts.mkdir(mode=0o755)
        _copy_protected(contracts_input/'.git',contracts/'.git')
        installer = root/'installer.py'
        blob = subprocess.check_output(_checkout_git(source, 'cat-file', 'blob', commit+':scripts/install_scene_retirement_runtime.py'), env={'PATH':'/usr/bin:/bin','GIT_NO_LAZY_FETCH':'1','GIT_ALLOW_PROTOCOL':'','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'},timeout=10)
        assert 0 < len(blob) <= 1024*1024
        oid = _native(_checkout_git(source, 'rev-parse', commit+':scripts/install_scene_retirement_runtime.py')).strip()
        assert hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest() == oid
        installer.write_bytes(blob)
        installer.chmod(0o644)
        command = ['/usr/bin/python3','-I','-S',str(installer),'--source',str(source),
            '--source-commit',commit,'--locked-sdk','--contracts-checkout',str(contracts)]
        prepared = json.loads(_native(command,timeout=310))
        assert prepared['status'] == 'prepared' and prepared['source_commit'] == commit
        assert prepared['authority_issued'] is False and prepared['cleanup_enabled'] is False
        # A second genuine deployment invocation exercises the atomic rolling
        # source/SDK cohort. Each invocation retains its original 300s origin.
        refreshed = json.loads(_native(['/usr/bin/python3','-I','-S',str(_BOOT/'runtime_installer.py'),
            *command[4:]],timeout=310))
        assert refreshed['status'] == 'refreshed' and refreshed['source_commit'] == commit
        current = json.loads((_BOOT/'CURRENT.json').read_bytes())
        runtime = Path(current['runtime_root'])
        dependencies = Path(current['dependencies_root'])
        assert current['schema'] == 'scene-retirement-runtime-cohort.v1'
        abi = _native(['/usr/bin/python3','-I','-S','-c','import sys; print(str(sys.version_info.major)+"."+str(sys.version_info.minor))']).strip()
        assert prepared['system_python_abi'] == refreshed['system_python_abi'] == abi
        packages = {row['name'] for row in prepared['sdk_packages']}
        assert {'opencv-python-headless','build123d','trimesh','pycollada','blueprint-contracts','google-cloud-pubsub'} <= packages
        assert not {'ultralytics','torch','nvidia-cuda-runtime-cu12'} & packages
        for path in (_RUNTIME,_BOOT,sdk_inputs):
            info=path.stat()
            installed[path]=(info.st_dev,info.st_ino)
            assert info.st_uid == 0 and not info.st_mode & 0o022
        for name,mode in [('coordination',0o755),('generations',0o700),('payloads',0o755),('state',0o700)]:
            path=root/name
            path.mkdir(mode=mode)
            path.chmod(mode)
        os.chown(root/'state',account.pw_uid,account.pw_gid)
        engine = ast.parse((runtime/'src/blueprint_pipeline/task_evaluation_scene_retirement.py').read_bytes())
        catalogue = ast.literal_eval(next(node.value for node in engine.body if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id=='_COHORT_CALLS' for target in node.targets)))
        cohort=[]
        for module,names in sorted(catalogue.items()):
            raw=(runtime/'src/blueprint_pipeline'/ (module.replace('.','/')+'.py')).read_bytes()
            for function in names.split():
                cohort.append(dict(entrypoint='blueprint_pipeline.'+module+':'+function,
                    installed_source_sha='sha256:'+hashlib.sha256(raw).hexdigest(),lifetime_contract_version='scene_retirement_lifetime.v1'))
        policy=dict(schema_version='scene_retirement_policy.v1',enabled=True,policy_id='native-enabled-sdk',
            roots=[dict(root=str(root/'payloads'),storage_class='evidence',device=root.stat().st_dev)],
            coordinator_path=str(root/'coordination'),generation_store=str(root/'generations'),journal_store=str(retirement_state/'journals'),
            consumer_cohort=cohort,principals=[],private_archive_allowed_classes=[],limits={})
        policy['policy_digest']='sha256:'+hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        _POLICY.parent.mkdir(mode=0o755,exist_ok=True)
        _POLICY.write_text(json.dumps(policy))
        _POLICY.chmod(0o644)
        # ExecStartPre executes the real guard as blueprint with only the
        # protected selected ABI SDK/source. No mock successful guard exists.
        guard=root/'guard.py'
        guard.write_text('import runpy,sys\nassert sys.argv[1:]==["-m","blueprint_pipeline.production_runtime_env_guard"]\n'
            'sys.argv=["blueprint_pipeline.production_runtime_env_guard"]\nsys.path[:0]='+repr([str(runtime/'src'),str(dependencies),str(runtime)])
            +'\nrunpy.run_module("blueprint_pipeline.production_runtime_env_guard",run_name="__main__")\n')
        guard.chmod(0o644)
        executable=root/'sdk-python'
        executable.write_text('#!/bin/sh\nexec /usr/bin/python3 -I -S '+str(guard)+' "$@"\n')
        executable.chmod(0o755)
        environment.write_text('BLUEPRINT_PIPELINE_REPO='+str(runtime)+'\nBLUEPRINT_PIPELINE_PYTHON='+str(executable)
            +'\nBLUEPRINT_SPEND_AUTHORITY_ROOT='+str(root/'state/spend')+'\nBLUEPRINT_SPEND_AUTHORITY_LEGACY_ROOTS=\nPORT=18765\n')
        environment.chmod(0o644)
        shutil.copyfile(runtime/'deploy/systemd'/unit,unit_path)
        unit_path.chmod(0o644)
        _native(['/usr/bin/systemctl','daemon-reload'])
        _native(['/usr/bin/systemctl','start',unit],timeout=90)
        deadline=time.monotonic()+90
        health=None
        while time.monotonic()<deadline:
            try:
                with urllib.request.urlopen('http://127.0.0.1:18765/health',timeout=1) as response:
                    health=json.loads(response.read(65537))
                break
            except (OSError,ValueError):
                time.sleep(.1)
        journal=_native(['/usr/bin/journalctl','--unit='+unit,'--no-pager','--lines=80','--output=cat','--quiet'])
        assert health is not None, journal
        properties=dict(line.split('=',1) for line in _native(['/usr/bin/systemctl','show',unit,'--no-pager','--all',
            '--property=MainPID,ActiveState,SubState,ControlGroup,InvocationID']).splitlines())
        assert properties['ActiveState']=='active' and properties['SubState']=='running'
        pid=int(properties['MainPID'])
        status=dict(line.split(':',1) for line in Path('/proc/'+str(pid)+'/status').read_text().splitlines() if ':' in line)
        assert status['Uid'].split()==[str(account.pw_uid)]*4 and status['Groups'].split()==[]
        assert all(int(status[key].strip(),16)==0 for key in ('CapEff','CapPrm','CapInh','CapAmb'))
        boots=list((retirement_state/'journals/processes').glob('*.json'))
        assert len(boots)==1
        boot=json.loads(boots[0].read_bytes())
        assert boot['pid']==pid and boot['invocation_id']==properties['InvocationID'] and boot['cgroup']==properties['ControlGroup']
        assert boot['target_uid']==account.pw_uid and boot['policy_digest']==policy['policy_digest']
        assert boot['sources'] and all(row['path'].startswith(str(runtime/'src')+'/') for row in boot['sources'].values())
        assert '[production-runtime-env-guard] status=ready' in journal
        print(json.dumps(dict(status='passed',actual_systemd=True,actual_system_abi=abi,
            locked_sdk_packages=len(packages),current_selected=True,actual_enabled_imports=True,
            kernel_uid=account.pw_uid,kernel_caps_zero=True,retirement_action_executed=False)),flush=True)
    finally:
        if unit_path.exists():
            _native(['/usr/bin/systemctl','stop',unit])
            unit_path.unlink()
            _native(['/usr/bin/systemctl','daemon-reload'])
        for path in (_POLICY,environment):
            if path.exists():
                path.unlink()
        for path in (_RUNTIME,_BOOT,sdk_inputs):
            if path.exists():
                if path in installed:
                    assert (path.stat().st_dev,path.stat().st_ino)==installed[path]
                shutil.rmtree(path)
        assert (root.stat().st_dev,root.stat().st_ino)==original
        shutil.rmtree(root)
        assert (retirement_state.stat().st_dev,retirement_state.stat().st_ino)==retirement_identity
        shutil.rmtree(retirement_state)
        if base_created:
            base.rmdir()
        if created_account:
            _native(['/usr/sbin/userdel','blueprint'])


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != 'linux' or os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') != '1',
                   reason='mandatory actual Linux locked SDK/current/enabled imports; Mac skip is unmet')
def test_actual_locked_system_sdk_current_selection_and_enabled_service_imports():
    keys=('BLUEPRINT_NATIVE_SOURCE_ROOT','BLUEPRINT_NATIVE_SOURCE_COMMIT','BLUEPRINT_NATIVE_CONTRACTS_ROOT')
    command=[sys.executable,str(Path(__file__).resolve()),'--native-enabled-sdk']
    if os.geteuid()!=0:
        command=['sudo','-n','env','BLUEPRINT_DISPOSABLE_LINUX_TEST=1','PYTHONDONTWRITEBYTECODE=1',
            *(key+'='+os.environ[key] for key in keys),*command]
    value=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=900,cwd=Path(__file__).parents[1])
    assert len(value.stdout.encode())<=65536 and len(value.stderr.encode())<=65536
    assert value.returncode==0,value.stdout+value.stderr
    receipt=json.loads(value.stdout.strip().splitlines()[-1])
    assert receipt['status']=='passed' and receipt['kernel_uid']!=0 and receipt['locked_sdk_packages']>80
    assert receipt['current_selected'] and receipt['actual_enabled_imports'] and receipt['kernel_caps_zero']
    assert receipt['retirement_action_executed'] is False


if __name__ == '__main__':
    if sys.argv[1:] == ['--native-default-off']:
        _default_off_native_phase()
    else:
        assert sys.argv[1:]==['--native-enabled-sdk']
        _enabled_sdk_native_phase()


def test_native_fixture_git_access_is_scoped_to_exact_checkout(tmp_path):
    source = tmp_path / "checked-feature"
    source.mkdir()
    command = _checkout_git(source, "cat-file", "blob", "a" * 40 + ":installer.py")
    assert command == ["/usr/bin/git", "--no-replace-objects", "-c",
        "safe.directory=" + str(source.resolve()), "-C", str(source.resolve()),
        "cat-file", "blob", "a" * 40 + ":installer.py"]
    assert "safe.directory=*" not in command
