# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_supervisor.py
"""New worker startup is fenced before import; this is not old-process clearance."""
import importlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_scene_retirement_real_participants import access_fixture


WORKER = 'blueprint_pipeline.task_evaluation_scene_progression'
UNITS = {
    'blueprint-pubsub-handoff-listener': 'pubsub_handoff_listener',
    'blueprint-task-evaluation-scene-progression': 'task_evaluation_scene_progression',
    'blueprint-task-evaluation-launch-preparation': 'task_evaluation_launch_preparation_worker',
    'blueprint-task-evaluation-launch-activation': 'task_evaluation_launch_activation_worker',
    'blueprint-task-evaluation-episode-compilation': 'task_evaluation_episode_compilation_worker',
    'blueprint-task-evaluation-sam31-preparation-execution': 'task_evaluation_sam31_preparation_execution',
    'blueprint-task-evaluation-launch-dispatcher': 'task_evaluation_launch_dispatcher',
    'blueprint-task-evaluation-launch-reconciler': 'task_evaluation_launch_reconciler',
    'blueprint-task-evaluation-launch-supervisor': 'task_evaluation_launch_supervisor',
    'blueprint-task-evaluation-policy-canary-dispatcher': 'task_evaluation_policy_canary_dispatcher',
    'blueprint-task-evaluation-terminal-resource-release': 'task_evaluation_terminal_resource_release',
    'blueprint-existing-policy-canary-watchdog': 'operator_policy_canary_continuation',
    'blueprint-existing-policy-canary-continuation': 'operator_policy_canary_continuation',
    'blueprint-task-evaluation-configured-controls-progression': 'task_evaluation_configured_controls_progression_worker',
}


def supervisor():
    return importlib.import_module('blueprint_pipeline.task_evaluation_scene_retirement_supervisor')


def test_exclusive_retirement_refuses_before_worker_import(tmp_path, monkeypatch):
    access, _, _ = access_fixture(tmp_path, monkeypatch)
    module = supervisor()
    monkeypatch.setattr(module, '_INSTALLED_POLICY', tmp_path / 'policy.json', raising=False)
    calls = []
    monkeypatch.setattr(module.runpy, 'run_module', lambda *a, **k: calls.append((a, k)))
    with access.exclusive_scene_access():
        with pytest.raises(access.SceneRetirementAccessError, match='generation_unavailable'):
            module.main(['--worker', WORKER, '--', '--config', 'unchanged'])
    assert calls == []


def test_worker_holds_shared_fence_through_body_and_preserves_argv(tmp_path, monkeypatch):
    access, _, _ = access_fixture(tmp_path, monkeypatch)
    module = supervisor()
    monkeypatch.setattr(module, '_INSTALLED_POLICY', tmp_path / 'policy.json', raising=False)
    original = sys.argv
    seen = []

    def worker(name, **kwargs):
        seen.append((name, kwargs, list(sys.argv)))
        with pytest.raises(access.SceneRetirementAccessError, match='reader_active'):
            with access.exclusive_scene_access():
                pytest.fail('live worker body must hold actual shared directory flock')
        return {'finished': True}

    monkeypatch.setattr(module.runpy, 'run_module', worker)
    assert module.main(['--worker', WORKER, '--', '--config', 'unchanged']) == 0
    assert seen == [(WORKER, {'run_name': '__main__', 'alter_sys': True},
                     [WORKER, '--config', 'unchanged'])]
    assert sys.argv is original
    with access.exclusive_scene_access():
        pass


def test_disabled_policy_preserves_worker_system_exit_and_argv(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    monkeypatch.setattr(access, '_POLICY_UID', os.getuid())
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', raising=False)
    module = supervisor()
    # Production is Linux/systemd. macOS /etc is a symlink and must not be
    # silently followed by the protected acquisition boundary.
    monkeypatch.setattr(module, '_INSTALLED_POLICY', tmp_path / 'absent-policy.json', raising=False)
    monkeypatch.setattr(access, '_INSTALLED_POLICY', tmp_path / 'absent-policy.json', raising=False)
    original = sys.argv

    def worker(name, **kwargs):
        assert name == WORKER and sys.argv == [WORKER, '--help']
        raise SystemExit(7)

    monkeypatch.setattr(module.runpy, 'run_module', worker)
    with pytest.raises(SystemExit) as exc:
        module.main(['--worker', WORKER, '--', '--help'])
    assert exc.value.code == 7 and sys.argv is original


@pytest.mark.parametrize('worker', ['os', 'blueprint_pipeline.control_plane_storage_gc',
                                  'blueprint_pipeline.task_evaluation_scene_retirement_cli',
                                  '../worker', WORKER + ':main'])
def test_arbitrary_or_action_worker_refused_before_import(monkeypatch, worker):
    module = supervisor()
    calls = []
    monkeypatch.setattr(module.runpy, 'run_module', lambda *a, **k: calls.append(a))
    with pytest.raises(SystemExit) as exc:
        module.main(['--worker', worker, '--'])
    assert exc.value.code == 2 and calls == []


def test_worker_failure_releases_only_owned_shared_lifetime(tmp_path, monkeypatch):
    access, _, _ = access_fixture(tmp_path, monkeypatch)
    module = supervisor()
    monkeypatch.setattr(module, '_INSTALLED_POLICY', tmp_path / 'policy.json', raising=False)
    original = sys.argv

    def worker(*args, **kwargs):
        raise RuntimeError('fixture failure')

    monkeypatch.setattr(module.runpy, 'run_module', worker)
    with pytest.raises(RuntimeError, match='fixture failure'):
        module.main(['--worker', WORKER, '--'])
    assert sys.argv is original
    with access.exclusive_scene_access():
        pass


@pytest.mark.slow
def test_real_cold_process_cannot_import_worker_while_retirement_holds_ex(tmp_path, monkeypatch):
    access, _, _ = access_fixture(tmp_path, monkeypatch)
    supervisor()  # RED remains a missing implementation, not a child import failure.
    code = '''
import os
from blueprint_pipeline import task_evaluation_scene_retirement_access as access
from blueprint_pipeline import task_evaluation_scene_retirement_supervisor as supervisor
access._POLICY_UID = os.getuid()
supervisor._INSTALLED_POLICY = os.environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE']
access._INSTALLED_POLICY = os.environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE']
supervisor.runpy.run_module = lambda *a, **k: print("WORKER_ENTERED")
try:
    supervisor.main(["--worker", "blueprint_pipeline.task_evaluation_scene_progression", "--"])
except access.SceneRetirementAccessError as error:
    assert str(error) == "scene_retirement_generation_unavailable", str(error)
    print("STARTUP_REFUSED")
'''
    with access.exclusive_scene_access():
        result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                text=True, timeout=20, check=True)
    assert result.stdout.strip() == 'STARTUP_REFUSED'
    assert 'WORKER_ENTERED' not in result.stdout


def test_installed_policy_admission_cannot_be_redirected_by_environment(tmp_path, monkeypatch):
    access, _, _ = access_fixture(tmp_path, monkeypatch)
    module = supervisor()
    fixed = tmp_path / 'policy.json'
    monkeypatch.setattr(module, '_INSTALLED_POLICY', fixed, raising=False)
    foreign = tmp_path / 'foreign-policy.json'
    foreign.write_bytes(fixed.read_bytes())
    foreign.chmod(0o644)
    monkeypatch.setenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', str(foreign))
    calls = []
    monkeypatch.setattr(module.runpy, 'run_module', lambda *a, **k: calls.append(a))
    with pytest.raises(access.SceneRetirementAccessError, match='policy_binding_unproven'):
        module.main(['--worker', WORKER, '--'])
    assert calls == []


def test_worker_uses_fixed_installed_policy_when_environment_omits_it(tmp_path, monkeypatch):
    access, _, _ = access_fixture(tmp_path, monkeypatch)
    module = supervisor()
    fixed = tmp_path / 'policy.json'
    monkeypatch.setattr(module, '_INSTALLED_POLICY', fixed, raising=False)
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE')
    calls = []
    monkeypatch.setattr(module.runpy, 'run_module', lambda *a, **k: calls.append(a))
    monkeypatch.setenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', str(fixed))
    with access.exclusive_scene_access():
        monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE')
        with pytest.raises(access.SceneRetirementAccessError, match='generation_unavailable'):
            module.main(['--worker', WORKER, '--'])
        assert 'BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE' not in os.environ
    assert calls == []


@pytest.mark.parametrize('unit,worker', UNITS.items())
def test_installed_known_worker_uses_startup_fence(unit, worker):
    root = Path(__file__).resolve().parents[1]
    text = (root / 'deploy' / 'systemd' / (unit + '.service')).read_text()
    exec_start = next(line for line in text.splitlines() if line.startswith('ExecStart='))
    assert ('-m blueprint_pipeline.task_evaluation_scene_retirement_supervisor --worker '
            'blueprint_pipeline.' + worker + ' --') in exec_start


def loaded_exec_start(source):
    text = next(line.split('=', 1)[1] for line in source.read_text().splitlines()
                if line.startswith('ExecStart='))
    executable, flag, script = shlex.split(text)
    # systemd's show representation, without executing a real installed unit.
    return ('{ path=' + executable + ' ; argv[]=' + executable + ' ' + flag + ' '
            + script.replace('$$', '$') + ' ; ignore_errors=no ; start_time= ; '
            'stop_time= ; pid=0 ; code=(null) ; status=0/0 }')


def loaded_unit_fixture(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    monkeypatch.setattr(access, '_POLICY_UID', os.getuid())
    module = supervisor()
    unit = 'blueprint-task-evaluation-scene-progression.service'
    source = Path(__file__).resolve().parents[1] / 'deploy' / 'systemd' / unit
    installed = tmp_path / 'installed'
    installed.mkdir()
    target = installed / unit
    target.write_bytes(source.read_bytes())
    target.chmod(0o644)
    monkeypatch.setattr(module, '_SYSTEMD_DIR', installed, raising=False)
    monkeypatch.setattr(module, '_KNOWN_UNITS', {unit: 'task_evaluation_scene_progression'}, raising=False)
    row = {
        'Id': unit, 'LoadState': 'loaded', 'ActiveState': 'inactive',
        'SubState': 'dead', 'MainPID': '0', 'ControlPID': '0', 'Job': '',
        'NeedDaemonReload': 'no', 'FragmentPath': str(target), 'DropInPaths': '',
        'ExecStart': loaded_exec_start(source),
    }
    def query(allowance):
        return ('\n'.join(key + '=' + value for key, value in row.items()) + '\n').encode()
    monkeypatch.setattr(module, '_query_systemd', query, raising=False)
    return module, access, row, target


def test_native_gate_observes_exact_loaded_idle_worker_without_global_clearance(tmp_path, monkeypatch):
    module, _, row, _ = loaded_unit_fixture(tmp_path, monkeypatch)
    result = module.require_inactive_known_workers(SimpleNamespace(tick=lambda: None))
    assert result['scope'] == 'installed_known_worker_quiescence'
    assert result['worker_units'] == [row['Id']]
    assert result['unknown_readers_cleared'] is False
    assert result['external_lifetimes_cleared'] is False


@pytest.mark.parametrize('key,value', [
    ('ActiveState', 'active'), ('ActiveState', 'activating'), ('SubState', 'running'),
    ('MainPID', '23'), ('ControlPID', '42'), ('Job', '8'),
    ('LoadState', 'not-found'), ('NeedDaemonReload', 'yes'),
    ('DropInPaths', '/etc/systemd/system/unit.service.d/override.conf'),
    ('ExecStart', '/usr/bin/python -m blueprint_pipeline.task_evaluation_scene_progression'),
])
def test_native_gate_keeps_old_active_or_unproven_loaded_worker(tmp_path, monkeypatch, key, value):
    module, access, row, _ = loaded_unit_fixture(tmp_path, monkeypatch)
    row[key] = value
    with pytest.raises(access.SceneRetirementAccessError, match='worker_cohort_unproven'):
        module.require_inactive_known_workers(SimpleNamespace(tick=lambda: None))


def test_native_gate_refuses_installed_unit_drift(tmp_path, monkeypatch):
    module, access, _, target = loaded_unit_fixture(tmp_path, monkeypatch)
    target.write_text(target.read_text().replace('NoNewPrivileges=true', 'NoNewPrivileges=false'))
    with pytest.raises(access.SceneRetirementAccessError, match='worker_cohort_unproven'):
        module.require_inactive_known_workers(SimpleNamespace(tick=lambda: None))


def test_native_gate_rechecks_loaded_state_after_fragment_proof(tmp_path, monkeypatch):
    module, access, row, _ = loaded_unit_fixture(tmp_path, monkeypatch)
    original = module._query_systemd
    calls = []
    def changed(allowance):
        calls.append(1)
        if len(calls) == 2:
            row['MainPID'] = '23'
        return original(allowance)
    monkeypatch.setattr(module, '_query_systemd', changed)
    with pytest.raises(access.SceneRetirementAccessError, match='worker_cohort_unproven'):
        module.require_inactive_known_workers(SimpleNamespace(tick=lambda: None))
    assert len(calls) == 2


def test_native_gate_duplicate_loaded_property_is_refused(tmp_path, monkeypatch):
    module, access, _, _ = loaded_unit_fixture(tmp_path, monkeypatch)
    original = module._query_systemd
    monkeypatch.setattr(module, '_query_systemd', lambda allowance: original(allowance) + b'MainPID=0\n')
    with pytest.raises(access.SceneRetirementAccessError, match='worker_cohort_unproven'):
        module.require_inactive_known_workers(SimpleNamespace(tick=lambda: None))


def test_native_gate_lost_original_deadline_stops_before_loaded_query(tmp_path, monkeypatch):
    module, access, _, _ = loaded_unit_fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(module, '_query_systemd', lambda allowance: calls.append(1))
    def expired():
        raise access.SceneRetirementAccessError('scene_retirement_deadline')
    with pytest.raises(access.SceneRetirementAccessError, match='scene_retirement_deadline'):
        module.require_inactive_known_workers(SimpleNamespace(tick=expired))
    assert calls == []


@pytest.mark.parametrize('change', ['echo', 'extra-command'])
def test_loaded_bootstrap_token_does_not_attest_different_argv(tmp_path, monkeypatch, change):
    module, access, row, _ = loaded_unit_fixture(tmp_path, monkeypatch)
    if change == 'echo':
        row['ExecStart'] = row['ExecStart'].replace('exec env PYTHONPATH=src', 'echo env PYTHONPATH=src')
    else:
        row['ExecStart'] = row['ExecStart'].replace(' ; ignore_errors=no', ' && /bin/true ; ignore_errors=no')
    with pytest.raises(access.SceneRetirementAccessError, match='worker_cohort_unproven'):
        module.require_inactive_known_workers(SimpleNamespace(tick=lambda: None))


def test_fragment_replacement_after_loaded_observation_stays_unproven(tmp_path, monkeypatch):
    module, access, _, target = loaded_unit_fixture(tmp_path, monkeypatch)
    original = module._query_systemd
    calls = []
    def changed(allowance):
        calls.append(1)
        if len(calls) == 2:
            replacement = target.with_name('replacement')
            replacement.write_bytes(target.read_bytes())
            replacement.chmod(0o644)
            replacement.replace(target)
        return original(allowance)
    monkeypatch.setattr(module, '_query_systemd', changed)
    with pytest.raises(access.SceneRetirementAccessError, match='worker_cohort_unproven'):
        module.require_inactive_known_workers(SimpleNamespace(tick=lambda: None))


@pytest.mark.slow
@pytest.mark.parametrize('failure', ['overflow', 'lost-deadline', 'nonzero'])
def test_real_query_child_is_reaped_on_output_deadline_or_command_failure(monkeypatch, failure):
    module = supervisor()
    original = subprocess.Popen
    children = []
    observed = []
    code = {'overflow': 'import os; os.write(1,b"x"*140000)',
            'lost-deadline': 'import time; time.sleep(30)',
            'nonzero': 'raise SystemExit(9)'}[failure]
    def child(command, **kwargs):
        observed.append((command, kwargs))
        process = original([sys.executable, '-c', code], **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(module.subprocess, 'Popen', child)
    calls = []
    def tick():
        calls.append(1)
        if failure == 'lost-deadline' and len(calls) > 1:
            from blueprint_pipeline.task_evaluation_scene_retirement_access import SceneRetirementAccessError
            raise SceneRetirementAccessError('scene_retirement_deadline')
    from blueprint_pipeline.task_evaluation_scene_retirement_access import SceneRetirementAccessError
    with pytest.raises(SceneRetirementAccessError):
        module._query_systemd(SimpleNamespace(tick=tick))
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdout.closed
    command, kwargs = observed[0]
    assert command[0] == '/usr/bin/systemctl' and 'Environment' not in ' '.join(command)
    assert kwargs['env'] == {'LC_ALL': 'C', 'PATH': '/usr/bin:/bin'}


def test_reader_closure_never_promotes_idle_unit_diagnostics_to_action_authority(tmp_path, monkeypatch):
    module, access, _, _ = loaded_unit_fixture(tmp_path, monkeypatch)
    (tmp_path / 'installation').mkdir()
    _, policy, _ = access_fixture(tmp_path / 'installation', monkeypatch)
    # The genuine public installed policy starts with no enrolled consumers.
    # Neither known-unit idleness nor caller-supplied diagnostic booleans fill
    # that missing boot/lifetime authority.
    observed = module.require_inactive_known_workers(SimpleNamespace(tick=lambda: None))
    assert observed['unknown_readers_cleared'] is False
    with pytest.raises(access.SceneRetirementAccessError, match='reader_closure_unproven'):
        module.require_current_reader_closure(policy, SimpleNamespace(tick=lambda: None))


def test_reader_closure_expired_original_allowance_refuses_before_kernel_scan(tmp_path, monkeypatch):
    module = supervisor()
    access, policy, _ = access_fixture(tmp_path, monkeypatch)
    def expired():
        raise access.SceneRetirementAccessError('scene_retirement_deadline')
    with pytest.raises(access.SceneRetirementAccessError, match='scene_retirement_deadline'):
        module.require_current_reader_closure(policy, SimpleNamespace(tick=expired))


def test_reader_closure_requires_actual_linux_kernel_not_caller_claims(tmp_path, monkeypatch):
    module = supervisor()
    access, policy, _ = access_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    with pytest.raises(access.SceneRetirementAccessError, match='reader_closure_unproven'):
        module.require_current_reader_closure(policy, SimpleNamespace(tick=lambda: None))


def continuous_unit_fixture(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    monkeypatch.setattr(access, '_POLICY_UID', os.getuid())
    module = supervisor()
    runtime = tmp_path / 'runtime'
    package = runtime / 'src/blueprint_pipeline'
    package.mkdir(parents=True)
    monkeypatch.setattr(module, '__file__', str(package / 'task_evaluation_scene_retirement_supervisor.py'))
    installed = tmp_path / 'installed'
    installed.mkdir()
    monkeypatch.setattr(module, '_SYSTEMD_DIR', installed)
    name = 'blueprint-pipeline-intake.service'
    source = Path(__file__).resolve().parents[1] / 'deploy/systemd' / name
    reference = runtime / 'deploy/systemd' / name
    reference.parent.mkdir(parents=True)
    reference.write_bytes(source.read_bytes())
    reference.chmod(0o644)
    selected = installed / name
    selected.write_bytes(source.read_bytes())
    selected.chmod(0o644)
    command = shlex.split(next(line.partition('=')[2] for line in source.read_text().splitlines()
                               if line.startswith('ExecStart=')))
    row = dict(Id=name, LoadState='loaded', FragmentPath=str(selected), DropInPaths='',
        NeedDaemonReload='no', ControlPID='0', Job='', User='blueprint', Group='blueprint',
        NoNewPrivileges='yes', AmbientCapabilities='', CapabilityBoundingSet='cap_setgid cap_setuid',
        ExecStart='{ path=/usr/bin/python3 ; argv[]=' + ' '.join(command).removeprefix('!')
        + ' ; ignore_errors=no ; start_time=n/a ; stop_time=n/a ; pid=118 ; code=(null) ; status=0 }')
    native = module._NativeObservation(SimpleNamespace(tick=lambda: None))
    return module, access, row, native, installed


def test_continuous_current_unit_accepts_exact_managed_identity_dropin(tmp_path, monkeypatch):
    module, _, row, native, installed = continuous_unit_fixture(tmp_path, monkeypatch)
    parent = installed / 'blueprint-pipeline-intake.service.d'
    parent.mkdir()
    selected = parent / '90-blueprint-deploy-identity.conf'
    environment = selected.with_suffix('.env')
    selected.write_text('# Managed by scripts/deploy_control_plane_commit.py.\n'
        '# Loaded after the base unit credential EnvironmentFile.\n[Service]\n'
        f'EnvironmentFile={environment}\nTimeoutStartSec=300s\n')
    environment.write_text('# Managed by scripts/deploy_control_plane_commit.py.\n'
        '# Contains deployment identity only; no credentials.\n'
        'BLUEPRINT_PIPELINE_REPO=/opt/blueprint/releases/' + 'a' * 40 + '\n'
        'BLUEPRINT_SOURCE_COMMIT=' + 'a' * 40 + '\n'
        'BLUEPRINT_PIPELINE_PYTHON=/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python\n'
        'PYTHONPATH=/opt/blueprint/releases/' + 'a' * 40 + '/src\n'
        'BLUEPRINT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT=/var/lib/blueprint/pipeline-control-plane/scene-object-discoveries\n')
    selected.chmod(0o644)
    environment.chmod(0o644)
    row['DropInPaths'] = str(selected)
    observed = module._exact_continuous_unit(row, 'blueprint_pipeline.live_pipeline_intake_service', native)
    assert observed['drop_in']['path'] == str(selected)
    assert observed['drop_in']['environment_path'] == str(environment)


@pytest.mark.parametrize('suffix', [
    ' ; path=/bin/false ; argv[]=/bin/false',
    ' } { path=/bin/false ; argv[]=/bin/false ; ignore_errors=no',
])
def test_continuous_loaded_command_rejects_extra_native_objects(tmp_path, monkeypatch, suffix):
    module, access, row, native, _ = continuous_unit_fixture(tmp_path, monkeypatch)
    row['ExecStart'] = row['ExecStart'][:-2] + suffix + ' }'
    with pytest.raises(access.SceneRetirementAccessError, match='reader_closure_unproven'):
        module._exact_continuous_unit(row, 'blueprint_pipeline.live_pipeline_intake_service', native)
