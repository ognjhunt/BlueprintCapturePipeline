"""The root entrypoint must not import the mutable service checkout."""
from pathlib import Path
import importlib.util
import os
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
    assert len(values) == 4
    assert all(raw == b'value = "sealed-source"\n' and path.parent == package
               for path, raw in values.values())


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
