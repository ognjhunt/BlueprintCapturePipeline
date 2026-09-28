"""The root entrypoint must not import the mutable service checkout."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest


@pytest.fixture
def tmp_path():
    # Linux RUNNER_TEMP is intentionally shared/writable. Protected-input
    # positives require genuinely non-writable ancestry, not an ancestry waiver.
    # The directory and every tiny fixture are removed after each test.
    with tempfile.TemporaryDirectory(prefix='.blueprint-scene-fixture-', dir=Path.home()) as name:
        yield Path(name).resolve()


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


@pytest.mark.parametrize('action', ['blueprint_pipeline.control_plane_storage_gc',
                                  'blueprint_pipeline.task_evaluation_scene_retirement_cli'])
def test_root_action_only_compiles_fixed_protected_entrypoints(tmp_path, monkeypatch, action):
    module, package = fixture(tmp_path, monkeypatch)
    (package / (action.split('.')[-1] + '.py')).write_bytes(b'def main(args): return 0\n')
    values = module._action_sources(action)
    assert values[action][1] == b'def main(args): return 0\n'
    assert set(values) == set(module._core_sources()) | {action}


@pytest.mark.parametrize('change', ['foreign-action', 'linked-action', 'writable-action'])
def test_root_action_refuses_unknown_or_mutable_entrypoint(tmp_path, monkeypatch, change):
    module, package = fixture(tmp_path, monkeypatch)
    action = 'blueprint_pipeline.task_evaluation_scene_retirement_cli'
    path = package / 'task_evaluation_scene_retirement_cli.py'
    path.write_bytes(b'def main(args): return 0\n')
    if change == 'foreign-action':
        action = 'blueprint_pipeline.live_pipeline_intake_service'
    elif change == 'linked-action':
        foreign = tmp_path / 'foreign-action.py'
        path.rename(foreign)
        path.symlink_to(foreign)
    else:
        path.chmod(0o666)
    with pytest.raises(ValueError, match='scene_retirement_bootstrap_unproven'):
        module._action_sources(action)


def test_gc_and_door_actions_use_the_same_fixed_isolated_root_entrypoint():
    root = SCRIPT.parents[1]
    unit = (root / 'deploy/systemd/blueprint-control-plane-storage-gc.service').read_text()
    door = (root / 'deploy/operator-door/door-scene-lifecycle.sh').read_text()
    assert 'exec /usr/bin/python3 -I -S /usr/lib/blueprint/scene-retirement-runtime/continuous_bootstrap.py --action-module blueprint_pipeline.control_plane_storage_gc run' in unit
    assert '/usr/bin/python3 -I -S /usr/lib/blueprint/scene-retirement-runtime/continuous_bootstrap.py' in door
    assert '--action-module blueprint_pipeline.task_evaluation_scene_retirement_cli' in door
    assert 'PYTHONPATH=src' not in door


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


def _selected_cohort_fixture(tmp_path, monkeypatch):
    import hashlib
    import json
    module, _ = fixture(tmp_path, monkeypatch)
    source = module._RUNTIME_ROOT / "generations" / ("a" * 64)
    sdk = module._RUNTIME_ROOT / "sdk-generations" / ("b" * 64)
    (source / "src/blueprint_pipeline").mkdir(parents=True)
    (source / "scripts").mkdir()
    sdk.mkdir(parents=True)
    for name in module._CORE:
        (source / "src/blueprint_pipeline" / name).write_bytes(b'value = "selected-cohort"\n')
    boot = tmp_path.resolve() / "installed-bootstrap"
    boot.mkdir()
    executable = boot / "continuous_bootstrap.py"
    raw = SCRIPT.read_bytes()
    executable.write_bytes(raw)
    monkeypatch.setattr(module, "_BOOT_ROOT", boot, raising=False)
    monkeypatch.setattr(module, "__file__", str(executable))
    current = {"schema": "scene-retirement-runtime-cohort.v1", "runtime_root": str(source),
               "dependencies_root": str(sdk), "source_digest": "a" * 64,
               "dependency_digest": "b" * 64,
               "bootstrap": {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw), "mode": 0o644},
               "previous": {"sha256": "sha256:" + "c" * 64, "size_bytes": 1}}
    (boot / "CURRENT.json").write_text(json.dumps(current))
    return module, source, sdk, boot, current


def test_native_bootstrap_loads_one_exact_selected_source_sdk_cohort(tmp_path, monkeypatch):
    module, source, sdk, _, _ = _selected_cohort_fixture(tmp_path, monkeypatch)
    selected_source, selected_sdk = module._select_runtime()
    assert selected_source == source and selected_sdk == sdk
    values = module._core_sources()
    assert all(path.parent == source / "src/blueprint_pipeline" and raw == b'value = "selected-cohort"\n'
               for path, raw in values.values())
    assert values.runtime_root == source and values.dependencies_root == sdk


@pytest.mark.parametrize("change", ["sdk-tuple", "bootstrap-bytes", "linked-current"])
def test_mixed_or_replaced_current_cohort_refuses_before_source_import(tmp_path, monkeypatch, change):
    import json
    module, _, _, boot, current = _selected_cohort_fixture(tmp_path, monkeypatch)
    if change == "sdk-tuple":
        current["dependency_digest"] = "d" * 64
        (boot / "CURRENT.json").write_text(json.dumps(current))
    elif change == "bootstrap-bytes":
        (boot / "continuous_bootstrap.py").write_bytes(b"changed executable")
    else:
        (boot / "CURRENT.json").rename(boot / "foreign-current.json")
        (boot / "CURRENT.json").symlink_to(boot / "foreign-current.json")
    with pytest.raises(ValueError, match="scene_retirement_bootstrap_unproven"):
        module._core_sources()
