"""The root entrypoint must not import the mutable service checkout."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/scene_retirement_continuous_bootstrap.py'


def bootstrap():
    spec = importlib.util.spec_from_file_location('scene_bootstrap_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, monkeypatch):
    module = bootstrap()
    root = tmp_path.resolve() / 'runtime'
    package = root / 'src/blueprint_pipeline'
    package.mkdir(parents=True)
    for name in module._CORE:
        path = package / name
        path.write_text('value = "sealed-source"\n')
        path.chmod(0o644)
    monkeypatch.setattr(module, '_RUNTIME_ROOT', root)
    monkeypatch.setattr(module, '_OWNER', os.getuid())
    return module, package


def test_source_bundle_is_read_from_fixed_protected_core(tmp_path, monkeypatch):
    module, package = fixture(tmp_path, monkeypatch)
    foreign = tmp_path / 'untrusted'
    foreign.mkdir()
    (foreign / 'blueprint_pipeline.py').write_text('raise AssertionError("foreign import")')
    monkeypatch.setenv('PYTHONPATH', str(foreign))
    values = module._core_sources()
    assert len(values) == 6
    assert all(raw == b'value = "sealed-source"\n' and path.parent == package
               for path, raw in values.values())


def test_bootstrap_runtime_matches_enforcing_work_volume_snapshot():
    assert bootstrap()._RUNTIME_ROOT == Path('/mnt/blueprint-work/scene-retirement-runtime')


def test_loader_retains_original_source_identity_after_path_changes(tmp_path, monkeypatch):
    module, package = fixture(tmp_path, monkeypatch)
    values = module._core_sources()
    name = 'blueprint_pipeline.task_evaluation_scene_retirement_supervisor'
    path, raw = values[name]
    original = module._identity(path.stat())
    loader = module._SourceOnly(values)
    path.write_bytes(b'value = "new-protected-version"\n')
    assert loader.source_identities[name] == original
    assert loader.values[name] == (path, raw)
    assert loader.source_identities[name] != module._identity(path.stat())


@pytest.mark.parametrize('change', ['linked-source', 'writable-source', 'linked-parent', 'writable-parent'])
def test_root_import_refuses_mutable_or_linked_source(tmp_path, monkeypatch, change):
    module, package = fixture(tmp_path, monkeypatch)
    path = package / 'task_evaluation_scene_retirement_supervisor.py'
    if change == 'linked-source':
        target = tmp_path / 'foreign.py'
        target.write_text('raise AssertionError("foreign source")')
        path.unlink()
        path.symlink_to(target)
    elif change == 'writable-source':
        path.chmod(0o664)
    elif change == 'linked-parent':
        moved = package.with_name('foreign')
        package.rename(moved)
        package.symlink_to(moved)
    else:
        package.chmod(0o775)
    with pytest.raises(ValueError, match='scene_retirement_bootstrap_unproven'):
        module._core_sources()


@pytest.mark.parametrize('change', ['unisolated', 'site-enabled', 'nonroot', 'foreign-module'])
def test_root_bootstrap_refuses_before_any_application_import(monkeypatch, change):
    module = bootstrap()
    flags = SimpleNamespace(isolated=1, no_site=1)
    if change == 'unisolated':
        flags.isolated = 0
    if change == 'site-enabled':
        flags.no_site = 0
    monkeypatch.setattr(module.sys, 'flags', flags)
    monkeypatch.setattr(module.os, 'getuid', lambda: 1 if change == 'nonroot' else 0)
    monkeypatch.setattr(module.os, 'geteuid', lambda: 0)
    calls = []
    monkeypatch.setattr(module, '_core_sources', lambda: calls.append('application'))
    target = 'os' if change == 'foreign-module' else 'blueprint_pipeline.live_pipeline_intake_service'
    with pytest.raises(ValueError, match='scene_retirement_bootstrap_unproven'):
        module.main(['--continuous-module', target, '--'])
    assert calls == []


def test_bootstrap_rejects_preloaded_service_code_before_source_import(tmp_path, monkeypatch):
    module, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module.sys, 'flags', SimpleNamespace(isolated=1, no_site=1))
    monkeypatch.setattr(module.os, 'getuid', lambda: 0)
    monkeypatch.setattr(module.os, 'geteuid', lambda: 0)
    monkeypatch.setitem(sys.modules, 'blueprint_pipeline.already_mutable', SimpleNamespace())
    calls = []
    monkeypatch.setattr(module, '_core_sources', lambda: calls.append('application'))
    with pytest.raises(ValueError, match='scene_retirement_bootstrap_unproven'):
        module.main(['--continuous-module', 'blueprint_pipeline.live_pipeline_intake_service', '--'])
    assert calls == []


@pytest.mark.parametrize('unit,daemon', [
    ('blueprint-pipeline-intake.service', 'live_pipeline_intake_service'),
    ('blueprint-agent-execution.service', 'agent_execution.production'),
])
def test_continuous_unit_privileged_start_cannot_execute_mutable_checkout(unit, daemon):
    text = (SCRIPT.parents[1] / 'deploy/systemd' / unit).read_text()
    commands = [line for line in text.splitlines() if line.startswith('ExecStart=')]
    assert commands == [
        'ExecStart=!/usr/bin/python3 -I -S '
        '/usr/lib/blueprint/scene-retirement-runtime/continuous_bootstrap.py '
        '--continuous-module blueprint_pipeline.' + daemon + ' --'
    ]
    assert 'User=blueprint\n' in text and 'Group=blueprint\n' in text
    assert '\nCapabilityBoundingSet=CAP_SETUID CAP_SETGID\n' in text
    assert '\nAmbientCapabilities=\n' in text and '\nNoNewPrivileges=true\n' in text
    assert 'ReadWritePaths=/var/lib/blueprint/scene-retirement/journals/processes\n' in text
    assert 'ExecStart=+' not in text


@pytest.mark.slow
def test_isolated_compiler_uses_verified_source_and_ignores_stale_bytecode(tmp_path, monkeypatch):
    module, package = fixture(tmp_path, monkeypatch)
    path = package / 'task_evaluation_scene_retirement_supervisor.py'
    path.write_text('from .task_evaluation_scene_retirement_access import value\n'
                    'def main(args):\n    return value\n')
    cache = package / '__pycache__'
    cache.mkdir()
    # Bytecode is never an authority at the root boundary.
    (cache / f'{path.stem}.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc').write_bytes(b'foreign-cache')
    code = '''import importlib, importlib.util, os, pathlib, sys
spec=importlib.util.spec_from_file_location("boot",sys.argv[1])
boot=importlib.util.module_from_spec(spec); spec.loader.exec_module(boot)
boot._OWNER=os.getuid(); boot._RUNTIME_ROOT=pathlib.Path(sys.argv[2])
loader=boot._SourceOnly(boot._core_sources());sys.meta_path.insert(0,loader)
module=importlib.import_module("blueprint_pipeline.task_evaluation_scene_retirement_supervisor")
assert module.main([])=="sealed-source"
assert module.__file__.startswith(str(boot._RUNTIME_ROOT))
print("compiled-sealed-source")
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(SCRIPT),
                             str(module._RUNTIME_ROOT)], capture_output=True, text=True,
                            env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(tmp_path / 'foreign')},
                            timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == 'compiled-sealed-source\n'
