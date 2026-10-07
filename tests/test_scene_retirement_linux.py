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
import importlib.util
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
_SYSTEMD_PATH = Path('/etc/systemd/system')
_CONTINUOUS_UNIT_NAMES = ('blueprint-pipeline-intake.service', 'blueprint-agent-execution.service')


def _install_continuous_units(runtime, *, installed=None):
    # Both consumers must be genuinely loaded for the fixed native query. Keep
    # the second consumer inactive; never manufacture a successful show row.
    paths = [_SYSTEMD_PATH / name for name in _CONTINUOUS_UNIT_NAMES]
    assert all(not path.exists() and not path.is_symlink() for path in paths), \
        'preserve pre-existing native installations'
    sources = [(runtime / 'deploy/systemd' / path.name).read_bytes() for path in paths]
    assert all(0 < len(raw) <= 16384 for raw in sources)
    installed = {} if installed is None else installed
    for path, raw in zip(paths, sources, strict=True):
        with path.open('xb') as stream:
            info = os.fstat(stream.fileno())
            installed[path] = (info.st_dev, info.st_ino)
            stream.write(raw)
            os.fchmod(stream.fileno(), 0o644)
    return installed


def _native(command, *, timeout=30):
    value = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
        text=True, timeout=timeout, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})
    assert len(value.stdout.encode()) <= 65536 and len(value.stderr.encode()) <= 4096
    assert value.returncode == 0, value.stderr
    return value.stdout


def _create_native_configuration_parent():
    parent = _POLICY.parent
    try:
        parent.mkdir(mode=0o755)
    except FileExistsError:
        assert parent.is_dir() and not parent.is_symlink(), 'preserve pre-existing native configuration'
        return None
    info = parent.lstat()
    return parent, (info.st_dev, info.st_ino)


def _remove_native_configuration_parent(created):
    if created is None:
        return
    parent, identity = created
    info = parent.lstat()
    assert not parent.is_symlink() and (info.st_dev, info.st_ino) == identity
    # Empty-directory removal refuses any contents another installation owns.
    parent.rmdir()


def _checkout_git(source, *arguments):
    # This hermetic job has already selected the exact immutable feature/lock
    # checkout. Root reads its data without a global or wildcard trust change.
    path = Path(source).resolve(strict=True)
    assert path.is_dir() and not Path(source).is_symlink()
    return ['/usr/bin/git', '--no-replace-objects', '-c', 'safe.directory='+str(path),
            '-C', str(path), *arguments]


def _fixture_source_manifest(builder, source, contracts, commit, *, deadline):
    """Scope test-only builder Git commands to the two exact fixture roots."""
    run = builder._run_bounded
    roots = {str(Path(path).resolve(strict=True)) for path in (source,contracts)}
    def scoped(command, **kwargs):
        assert command[:3] == ['git','--no-replace-objects','-C']
        path = str(Path(command[3]).resolve(strict=True))
        assert path in roots
        return run(_checkout_git(path,*command[4:]), **kwargs)
    builder._run_bounded = scoped
    try:
        return builder.build_manifest(source,contracts,commit,deadline=deadline)
    finally:
        builder._run_bounded = run


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


def _native_fixture_source_driver():
    """Test-only source authority; no production CLI, env or crypto fallback."""
    return """import importlib.util,json,os,sys,time
from pathlib import Path
assert os.getuid()==os.geteuid()==0 and sys.flags.isolated and sys.flags.no_site
installer,source,commit,contracts,manifest=map(Path,sys.argv[1:])
commit=str(commit)
spec=importlib.util.spec_from_file_location('native_installer_under_test',installer)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
raw=manifest.read_bytes()
value=json.loads(raw)
assert raw==(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\\n').encode()
assert value['sources'][0]['commit']==commit
assert value['sources'][1]['commit']=='7708a4e4c5dedeeb39cc73d3f6869304de295b81'
def fixture_admission(manifest_path,bundle_path,verifier_path,expected_commit,deadline):
    assert expected_commit==commit and time.monotonic()<=deadline
    assert manifest.read_bytes()==raw
    return value
module._admitted_source_manifest=fixture_admission
result=module.prepare_deployment(source,source_commit=commit,contracts_checkout=contracts)
assert result['authority_issued'] is False and result['cleanup_enabled'] is False
print(json.dumps(result,sort_keys=True))
"""


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
    installed_units = {}
    configuration_parent = None
    try:
        # The private checkout token exists only in the preceding CI checkout
        # step. Root receives protected Git object data, with no token/env/key.
        contracts = root/'contracts'
        contracts.mkdir(mode=0o755)
        _copy_protected(contracts_input/'.git',contracts/'.git')
        installer = root/'installer.py'
        blob = subprocess.check_output(_checkout_git(source, 'cat-file', 'blob', commit+':scripts/install_scene_retirement_runtime.py'), env={'PATH':'/usr/bin:/bin','GIT_NO_LAZY_FETCH':'1','GIT_ALLOW_PROTOCOL':'','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'},timeout=10)
        assert 0 < len(blob) <= 1024*1024
        # The CI checkout is fixture authority, never a cryptographic main
        # release admission. Preserve the production refusal before any SDK.
        installer.write_bytes(blob)
        installer.chmod(0o644)
        command = ['/usr/bin/python3','-I','-S',str(installer),'--source',str(source),
            '--source-commit',commit,'--locked-sdk','--contracts-checkout',str(contracts)]
        refused = subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,
            timeout=30,env={'PATH':'/usr/bin:/bin','LC_ALL':'C'})
        assert refused.returncode == 2 and refused.stdout == b'' and len(refused.stderr) <= 4096
        assert b'scene_retirement_runtime_phase:source_attestation' in refused.stderr
        assert b'scene_retirement_runtime_failure:validation' in refused.stderr
        assert not any(path.exists() or path.is_symlink() for path in (_RUNTIME,_BOOT,sdk_inputs))
        # Generate canonical raw-byte SHA256 inventories from the two exact
        # checkout fixtures. Signing stays AFTER the required native gate.
        helper = root/'fixture_manifest_builder.py'
        helper.write_bytes(subprocess.check_output(_checkout_git(source,'cat-file','blob',
            commit+':scripts/release_source_manifest.py'),env={'PATH':'/usr/bin:/bin',
            'GIT_NO_LAZY_FETCH':'1','GIT_ALLOW_PROTOCOL':'','GIT_CONFIG_NOSYSTEM':'1',
            'GIT_CONFIG_GLOBAL':'/dev/null'},timeout=10))
        helper.chmod(0o644)
        spec = importlib.util.spec_from_file_location('native_fixture_manifest_builder',helper)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        raw_manifest = _fixture_source_manifest(builder,source,contracts,commit,deadline=time.monotonic()+120)
        admitted = builder.validate_manifest_bytes(raw_manifest,commit)
        row = builder.source_inventory(admitted,'ognjhunt/BlueprintCapturePipeline',commit)['scripts/install_scene_retirement_runtime.py']
        assert row['size'] == len(blob) and row['sha256'] == hashlib.sha256(blob).hexdigest()
        fixture_manifest = root/'fixture-source-manifest.json'
        fixture_manifest.write_bytes(raw_manifest)
        fixture_manifest.chmod(0o644)
        driver = root/'fixture-source-authority.py'
        driver.write_text(_native_fixture_source_driver())
        driver.chmod(0o644)
        command = ['/usr/bin/python3','-I','-S',str(driver),str(installer),str(source),commit,
                   str(contracts),str(fixture_manifest)]
        prepared = json.loads(_native(command,timeout=310))
        assert prepared['status'] == 'prepared' and prepared['source_commit'] == commit
        assert prepared['authority_issued'] is False and prepared['cleanup_enabled'] is False
        # These runtime reads must come from the existing authenticated source
        # roots, without requiring repository documentation in the generation.
        for name in ('task_evaluation_policy_canary_setup.v1.schema.json',
                     'rigid_task_success_contract.v1.schema.json',
                     'articulated_task_success_contract.v1.schema.json'):
            expected = subprocess.check_output(_checkout_git(source, 'cat-file', 'blob',
                commit + ':docs/schemas/' + name), timeout=10)
            assert (_RUNTIME / 'src/blueprint_pipeline/_catalog_schemas' / name).read_bytes() == expected
        # Only the source-admission boundary is intercepted by the test driver;
        # real locked SDK/copy/rolling/ABI/UID/systemd behavior remains exercised.
        # Each invocation retains its stricter 310s native fixture watchdog.
        refreshed = json.loads(_native([*command[:4],str(_BOOT/'runtime_installer.py'),
            *command[5:]],timeout=310))
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
        # ProtectHome remains enforced. Provider coordination belongs in the
        # disposable service state, where the real ordinary-UID guard creates
        # and verifies every canonical concurrency slot and deployment gate.
        lock_root = root/'state/provider-locks'
        lock_root.mkdir(mode=0o700)
        os.chown(lock_root,account.pw_uid,account.pw_gid)
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
        configuration_parent = _create_native_configuration_parent()
        _POLICY.write_text(json.dumps(policy))
        _POLICY.chmod(0o644)
        # ExecStartPre executes the real guard as blueprint with only the
        # protected selected ABI SDK/source. No mock successful guard exists.
        guard=root/'guard.py'
        guard.write_text('import runpy,sys\nassert sys.argv[1:]==["-m","blueprint_pipeline.production_runtime_env_guard"]\n'
            'sys.argv=["blueprint_pipeline.production_runtime_env_guard"]\nsys.path[:0]='+repr([str(runtime/'src'),str(dependencies),str(runtime)])
            +'\nrunpy.run_module("blueprint_pipeline.production_runtime_env_guard",run_name="__main__",alter_sys=True)\n')
        guard.chmod(0o644)
        executable=root/'sdk-python'
        executable.write_text('#!/bin/sh\nexec /usr/bin/python3 -I -S '+str(guard)+' "$@"\n')
        executable.chmod(0o755)
        environment.write_text('BLUEPRINT_PIPELINE_REPO='+str(runtime)+'\nBLUEPRINT_PIPELINE_PYTHON='+str(executable)
            +'\nBLUEPRINT_SPEND_AUTHORITY_ROOT='+str(root/'state/spend')
            +'\nVAST_LAUNCH_LOCK_FILE='+str(lock_root/'vast_paid_launch.lock')
            +'\nBLUEPRINT_SPEND_AUTHORITY_LEGACY_ROOTS=\nPORT=18765\n')
        environment.chmod(0o644)
        _install_continuous_units(runtime, installed=installed_units)
        _native(['/usr/bin/systemctl','daemon-reload'])
        try:
            _native(['/usr/bin/systemctl','start',unit],timeout=90)
        except AssertionError as error:
            # Retain the actual guard/bootstrap failure before the disposable
            # unit and installed SDK are removed by the fixture's finalizer.
            journal = _native(['/usr/bin/journalctl','--unit='+unit,'--no-pager',
                               '--lines=80','--output=cat','--quiet'])
            raise AssertionError(str(error) + '\n' + journal) from None
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
        # Observe the default three slots and deployment gate produced by the
        # genuine service guard; the privileged proof runner stays stdlib-only.
        locks = list(lock_root.iterdir())
        assert len(locks) == 4 and lock_root/'vast_paid_launch.lock' in locks
        for lock in locks:
            info = lock.lstat()
            assert info.st_uid == account.pw_uid and info.st_gid == account.pw_gid
            assert info.st_mode & 0o777 == 0o600 and info.st_nlink == 1
        assert 'ProtectHome=yes' in _native(['/usr/bin/systemctl','show',unit,'--property=ProtectHome'])
        print(json.dumps(dict(status='passed',actual_systemd=True,actual_system_abi=abi,
            locked_sdk_packages=len(packages),current_selected=True,actual_enabled_imports=True,
            kernel_uid=account.pw_uid,kernel_caps_zero=True,retirement_action_executed=False,
            source_authority='hermetic_fixture_sha256',cryptographic_main_release_admission_proven=False,
            production_missing_proof_refused=True)),flush=True)
    finally:
        if unit_path in installed_units:
            _native(['/usr/bin/systemctl','stop',unit])
        for path, identity in installed_units.items():
            assert (path.lstat().st_dev, path.lstat().st_ino) == identity
            path.unlink()
        if installed_units:
            _native(['/usr/bin/systemctl','daemon-reload'])
        for path in (_POLICY,environment):
            if path.exists():
                path.unlink()
        _remove_native_configuration_parent(configuration_parent)
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
    assert receipt['source_authority']=='hermetic_fixture_sha256'
    assert receipt['cryptographic_main_release_admission_proven'] is False and receipt['production_missing_proof_refused']


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


def test_native_fixture_manifest_git_access_is_scoped_and_restored(tmp_path):
    import types
    source,contracts = tmp_path/'source',tmp_path/'contracts'
    source.mkdir()
    contracts.mkdir()
    calls = []
    def original(command,**kwargs):
        calls.append((command,kwargs))
        return b'fixture raw Git data'
    builder = types.SimpleNamespace(_run_bounded=original)
    def build(*args,**kwargs):
        for path in (source,contracts):
            assert builder._run_bounded(['git','--no-replace-objects','-C',str(path),
                'cat-file','blob','a'*40],deadline=kwargs['deadline'],stdout_cap=1024) == b'fixture raw Git data'
        return b'canonical fixture manifest'
    builder.build_manifest = build
    deadline = time.monotonic()+10
    assert _fixture_source_manifest(builder,source,contracts,'a'*40,deadline=deadline) == b'canonical fixture manifest'
    assert builder._run_bounded is original and len(calls) == 2
    for path,(command,kwargs) in zip((source,contracts),calls,strict=True):
        assert command == _checkout_git(path,'cat-file','blob','a'*40)
        assert kwargs == {'deadline':deadline,'stdout_cap':1024}
    def failed(*args,**kwargs):
        raise ValueError('fixture failed')
    builder.build_manifest = failed
    with pytest.raises(ValueError,match='fixture failed'):
        _fixture_source_manifest(builder,source,contracts,'a'*40,deadline=deadline)
    assert builder._run_bounded is original


@pytest.mark.parametrize('preexisting', [False, True])
def test_native_configuration_cleanup_preserves_launch_residency(tmp_path, monkeypatch, preexisting):
    # ADP-009D/day28: installing the SDK must not turn later workstation
    # launches into control-plane launches by leaving /etc/blueprint behind.
    from blueprint_pipeline import host_resident_launch_inputs as residency
    from blueprint_pipeline.task_evaluation_launch_dispatcher import dispatch_launch_request
    from tests.test_task_evaluation_policy_canary_preparation_dispatch import _profile_and_request
    parent = tmp_path / 'etc-blueprint'
    monkeypatch.setattr(sys.modules[__name__], '_POLICY', parent / 'policy.json')
    monkeypatch.setattr(residency, 'PRODUCTION_LAUNCH_INPUT_ROOTS', (str(parent),))
    if preexisting:
        parent.mkdir()
    created = _create_native_configuration_parent()
    _POLICY.write_bytes(b'fixture policy')
    _POLICY.unlink()
    _remove_native_configuration_parent(created)
    assert parent.exists() is preexisting
    profile, request = _profile_and_request(tmp_path)
    profiles = tmp_path / 'profiles'
    profiles.mkdir()
    (profiles / (profile['profile_id'] + '.json')).write_text(json.dumps(profile))
    request_path = tmp_path / 'request.json'
    request_path.write_text(json.dumps(request))
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT', str(tmp_path / 'queue'))
    receipt = dispatch_launch_request(request_path=request_path, profile_dir=profiles,
        state_root=tmp_path / 'runs', execute=True,
        allocator_runner=lambda _: (_ for _ in ()).throw(AssertionError('provider forbidden')))
    assert receipt['status'] == ('blocked' if preexisting else 'queued_for_no_spend_preparation'), receipt
    if preexisting:
        assert 'launch_profile_input_not_host_resident:immutable_input:source_bundle_manifest' in receipt['blockers']


def test_native_configuration_cleanup_refuses_unowned_contents(tmp_path, monkeypatch):
    parent = tmp_path / 'etc-blueprint'
    monkeypatch.setattr(sys.modules[__name__], '_POLICY', parent / 'policy.json')
    created = _create_native_configuration_parent()
    other = parent / 'other-installation'
    other.write_bytes(b'preserve')
    with pytest.raises(OSError):
        _remove_native_configuration_parent(created)
    assert other.read_bytes() == b'preserve'


@pytest.mark.parametrize('replacement', ['directory', 'symlink'])
def test_native_configuration_cleanup_refuses_replaced_parent(tmp_path, monkeypatch, replacement):
    parent = tmp_path / 'etc-blueprint'
    monkeypatch.setattr(sys.modules[__name__], '_POLICY', parent / 'policy.json')
    created = _create_native_configuration_parent()
    original = tmp_path / 'original'
    parent.rename(original)
    if replacement == 'directory':
        parent.mkdir()
    else:
        parent.symlink_to(original, target_is_directory=True)
    with pytest.raises(AssertionError):
        _remove_native_configuration_parent(created)
    assert original.is_dir() and parent.exists()


@pytest.mark.parametrize('existing', [None, 'blueprint-agent-execution.service'])
def test_enabled_native_fixture_installs_entire_fixed_continuous_cohort(tmp_path, monkeypatch, existing):
    # ADP-009D/day28: the native startup query observes both installed consumers.
    # A missing second service is not an authenticated inactive consumer.
    import tests.test_scene_retirement_linux as fixture
    from blueprint_pipeline.task_evaluation_scene_retirement_supervisor import _CONTINUOUS
    native_units = tmp_path / 'systemd'
    native_units.mkdir()
    monkeypatch.setattr(fixture, '_SYSTEMD_PATH', native_units, raising=False)
    source = Path(__file__).resolve().parents[1]
    if existing:
        (native_units / existing).write_bytes(b'pre-existing native unit')
        with pytest.raises(AssertionError, match='pre-existing native installations'):
            fixture._install_continuous_units(source)
        assert list(native_units.iterdir()) == [native_units / existing]
        assert (native_units / existing).read_bytes() == b'pre-existing native unit'
        return
    installed = fixture._install_continuous_units(source)
    assert {path.name for path in installed} == set(_CONTINUOUS.values())
    for path, identity in installed.items():
        assert path.read_bytes() == (source / 'deploy/systemd' / path.name).read_bytes()
        assert path.stat().st_mode & 0o777 == 0o644
        assert (path.stat().st_dev, path.stat().st_ino) == identity


def test_native_source_driver_intercepts_only_fixture_admission_and_reports_no_signing_proof():
    driver = _native_fixture_source_driver()
    compile(driver,'native-fixture-source-authority','exec')
    patched = [node for node in ast.walk(ast.parse(driver)) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name)
                and target.value.id=='module' for target in node.targets)]
    assert len(patched)==1 and patched[0].targets[0].attr=='_admitted_source_manifest'
    assert 'module.prepare_deployment(' in driver and 'contracts_checkout=contracts' in driver
    source=Path(__file__).read_text()
    assert "cryptographic_main_release_admission_proven=False" in source
    assert "production_missing_proof_refused=True" in source
