# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_supervisor.py
"""New worker startup is fenced before import; this is not old-process clearance."""
import importlib
import subprocess
import sys
from pathlib import Path

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
}


def supervisor():
    return importlib.import_module('blueprint_pipeline.task_evaluation_scene_retirement_supervisor')


def test_exclusive_retirement_refuses_before_worker_import(tmp_path, monkeypatch):
    access, _, _ = access_fixture(tmp_path, monkeypatch)
    module = supervisor()
    calls = []
    monkeypatch.setattr(module.runpy, 'run_module', lambda *a, **k: calls.append((a, k)))
    with access.exclusive_scene_access():
        with pytest.raises(access.SceneRetirementAccessError, match='generation_unavailable'):
            module.main(['--worker', WORKER, '--', '--config', 'unchanged'])
    assert calls == []


def test_worker_holds_shared_fence_through_body_and_preserves_argv(tmp_path, monkeypatch):
    access, _, _ = access_fixture(tmp_path, monkeypatch)
    module = supervisor()
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


def test_disabled_policy_preserves_worker_system_exit_and_argv(monkeypatch):
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', raising=False)
    module = supervisor()
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
supervisor.runpy.run_module = lambda *a, **k: print("WORKER_ENTERED")
try:
    supervisor.main(["--worker", "blueprint_pipeline.task_evaluation_scene_progression", "--"])
except access.SceneRetirementAccessError:
    print("STARTUP_REFUSED")
'''
    with access.exclusive_scene_access():
        result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                text=True, timeout=20, check=True)
    assert result.stdout.strip() == 'STARTUP_REFUSED'
    assert 'WORKER_ENTERED' not in result.stdout


@pytest.mark.parametrize('unit,worker', UNITS.items())
def test_installed_known_worker_uses_startup_fence(unit, worker):
    root = Path(__file__).resolve().parents[1]
    text = (root / 'deploy' / 'systemd' / (unit + '.service')).read_text()
    exec_start = next(line for line in text.splitlines() if line.startswith('ExecStart='))
    assert ('-m blueprint_pipeline.task_evaluation_scene_retirement_supervisor --worker '
            'blueprint_pipeline.' + worker + ' --') in exec_start
