"""Protected runtime preparation is separate from issuing deletion authority."""
from pathlib import Path
import importlib.util
import os

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/install_scene_retirement_runtime.py'


def fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('runtime_install_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path.resolve() / 'source'
    (source / 'src/blueprint_pipeline').mkdir(parents=True)
    (source / 'deploy/systemd').mkdir(parents=True)
    (source / 'scripts').mkdir()
    (source / 'src/blueprint_pipeline/__init__.py').write_bytes(b'# trusted package\n')
    (source / 'deploy/systemd/blueprint-pipeline-intake.service').write_bytes(b'[Service]\nUser=blueprint\n')
    (source / 'scripts/scene_retirement_continuous_bootstrap.py').write_bytes(b'# fixed bootstrap\n')
    deps = tmp_path.resolve() / 'dependencies'
    deps.mkdir()
    (deps / 'trusted_sdk.py').write_bytes(b'value = 1\n')
    monkeypatch.setattr(module, '_OWNER', os.getuid())
    monkeypatch.setattr(module, '_RUNTIME_ROOT', tmp_path.resolve() / 'installed')
    monkeypatch.setattr(module, '_BOOT_ROOT', tmp_path.resolve() / 'bootstrap')
    monkeypatch.setattr(module, '_FREE_FLOOR', 0)
    return module, source, deps


def test_preparation_installs_real_protected_source_dependencies_and_units(tmp_path, monkeypatch):
    module, source, deps = fixture(tmp_path, monkeypatch)
    result = module.prepare(source, deps)
    root = module._RUNTIME_ROOT
    assert result['status'] == 'prepared'
    assert (root / 'src/blueprint_pipeline/__init__.py').read_bytes() == b'# trusted package\n'
    assert (root / 'dependencies/trusted_sdk.py').read_bytes() == b'value = 1\n'
    assert (root / 'deploy/systemd/blueprint-pipeline-intake.service').read_bytes() == b'[Service]\nUser=blueprint\n'
    assert (module._BOOT_ROOT / 'continuous_bootstrap.py').read_bytes() == b'# fixed bootstrap\n'
    assert all(p.stat().st_uid == os.getuid() and p.stat().st_mode & 0o022 == 0
               for p in root.rglob('*'))
    assert not list(root.rglob('*.pyc'))
    assert result['authority_issued'] is False and result['cleanup_enabled'] is False


@pytest.mark.parametrize('change', ['source-link', 'source-writable', 'dependency-link', 'dependency-writable'])
def test_unprotected_input_refuses_before_creating_runtime(tmp_path, monkeypatch, change):
    module, source, deps = fixture(tmp_path, monkeypatch)
    path = (source / 'src/blueprint_pipeline/__init__.py') if change.startswith('source') else deps / 'trusted_sdk.py'
    if change.endswith('link'):
        foreign = tmp_path / 'foreign'
        foreign.write_bytes(b'foreign')
        path.unlink()
        path.symlink_to(foreign)
    else:
        path.chmod(0o666)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.prepare(source, deps)
    assert not module._RUNTIME_ROOT.exists()
    assert not module._BOOT_ROOT.exists()


def test_existing_identical_snapshot_is_reused_without_republication(tmp_path, monkeypatch):
    module, source, deps = fixture(tmp_path, monkeypatch)
    module.prepare(source, deps)
    path = module._RUNTIME_ROOT / 'dependencies/trusted_sdk.py'
    before = path.stat()
    assert module.prepare(source, deps)['status'] == 'already_prepared'
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_mtime_ns)


def test_unknown_or_changed_installed_runtime_is_preserved(tmp_path, monkeypatch):
    module, source, deps = fixture(tmp_path, monkeypatch)
    module.prepare(source, deps)
    (source / 'src/blueprint_pipeline/__init__.py').write_bytes(b'# changed source\n')
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.prepare(source, deps)
    assert (module._RUNTIME_ROOT / 'src/blueprint_pipeline/__init__.py').read_bytes() == b'# trusted package\n'


def test_space_floor_refuses_before_copying_any_runtime_bytes(tmp_path, monkeypatch):
    module, source, deps = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module, '_FREE_FLOOR', 2**63)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.prepare(source, deps)
    assert not module._RUNTIME_ROOT.exists()
