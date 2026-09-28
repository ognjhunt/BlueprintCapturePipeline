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
