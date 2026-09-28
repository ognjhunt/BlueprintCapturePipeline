"""Protected runtime preparation is separate from issuing deletion authority."""
from pathlib import Path
import importlib.util
import os
import sys
import tempfile

import pytest


@pytest.fixture
def tmp_path():
    # Linux RUNNER_TEMP is intentionally shared/writable. Protected-input
    # positives require genuinely non-writable ancestry, not an ancestry waiver.
    # The directory and every tiny fixture are removed after each test.
    with tempfile.TemporaryDirectory(prefix='.blueprint-scene-fixture-', dir=Path.home()) as name:
        yield Path(name).resolve()


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


@pytest.mark.parametrize('change', ['abi', 'linked-config', 'writable-config', 'duplicate-version'])
def test_dependency_venv_must_match_actual_system_python_before_copy(tmp_path, monkeypatch, change):
    module, _, _ = fixture(tmp_path, monkeypatch)
    venv = tmp_path.resolve() / 'root-venv'
    sdk = venv / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    sdk.mkdir(parents=True)
    config = venv / 'pyvenv.cfg'
    config.write_text(f'version = {sys.version_info.major}.{sys.version_info.minor}.1\n')
    if change == 'abi':
        config.write_text('version = 2.7.1\n')
    elif change == 'linked-config':
        real = tmp_path / 'foreign-config'
        config.rename(real)
        config.symlink_to(real)
    elif change == 'writable-config':
        config.chmod(0o666)
    else:
        config.write_text(config.read_text() * 2)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.dependency_root(venv)
    assert not module._RUNTIME_ROOT.exists()


def test_dependency_location_is_derived_without_executing_venv(tmp_path, monkeypatch):
    module, _, _ = fixture(tmp_path, monkeypatch)
    venv = tmp_path.resolve() / 'root-venv'
    sdk = venv / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    sdk.mkdir(parents=True)
    (venv / 'pyvenv.cfg').write_text(f'version = {sys.version_info.major}.{sys.version_info.minor}.1\n')
    (venv / 'bin').mkdir()
    (venv / 'bin/python').write_text('raise AssertionError("must not execute service interpreter")')
    assert module.dependency_root(venv) == sdk


@pytest.mark.parametrize('partial', [False, True])
def test_interrupted_owned_copy_resumes_without_overwriting_prefix(tmp_path, monkeypatch, partial):
    module, source, deps = fixture(tmp_path, monkeypatch)
    original = module._copy
    failed = False
    def interrupt(path, destination, expected, deadline):
        nonlocal failed
        if not failed:
            failed = True
            if partial:
                module._mkdir(destination.parent)
                destination.write_bytes(path.read_bytes()[:3])
                destination.chmod(0o600)
            else:
                original(path, destination, expected, deadline)
            raise OSError('simulated interrupted copy')
        return original(path, destination, expected, deadline)
    monkeypatch.setattr(module, '_copy', interrupt)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.prepare(source, deps)
    assert not (module._BOOT_ROOT / 'continuous_bootstrap.py').exists()
    monkeypatch.setattr(module, '_copy', original)
    assert module.prepare(source, deps)['status'] == 'prepared'
    assert (module._RUNTIME_ROOT / 'src/blueprint_pipeline/__init__.py').read_bytes() == b'# trusted package\n'
    assert module.prepare(source, deps)['status'] == 'already_prepared'


@pytest.mark.parametrize('change', ['foreign-prefix', 'changed-input', 'extra-file'])
def test_interrupted_install_never_adopts_foreign_or_changed_bytes(tmp_path, monkeypatch, change):
    module, source, deps = fixture(tmp_path, monkeypatch)
    original = module._copy
    def interrupt(path, destination, expected, deadline):
        module._mkdir(destination.parent)
        destination.write_bytes(path.read_bytes()[:3])
        destination.chmod(0o600)
        raise OSError('simulated interrupted copy')
    monkeypatch.setattr(module, '_copy', interrupt)
    with pytest.raises(ValueError):
        module.prepare(source, deps)
    targets = [p for p in module._RUNTIME_ROOT.rglob('*') if p.is_file()]
    assert len(targets) == 1
    if change == 'foreign-prefix':
        targets[0].write_bytes(b'foreign')
    elif change == 'changed-input':
        (source / 'src/blueprint_pipeline/__init__.py').write_bytes(b'# changed source\n')
    else:
        (module._RUNTIME_ROOT / 'unowned').write_bytes(b'foreign')
    before = {str(p): p.read_bytes() for p in module._RUNTIME_ROOT.rglob('*') if p.is_file()}
    monkeypatch.setattr(module, '_copy', original)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.prepare(source, deps)
    assert {str(p): p.read_bytes() for p in module._RUNTIME_ROOT.rglob('*') if p.is_file()} == before
    assert not (module._BOOT_ROOT / 'continuous_bootstrap.py').exists()


def test_parallel_installer_refuses_before_touching_the_owned_copy(tmp_path, monkeypatch):
    module, source, deps = fixture(tmp_path, monkeypatch)
    original = module._copy
    entered = False
    def competing(path, destination, expected, deadline):
        nonlocal entered
        if not entered:
            entered = True
            with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
                module.prepare(source, deps)
            assert not destination.exists()
        return original(path, destination, expected, deadline)
    monkeypatch.setattr(module, '_copy', competing)
    assert module.prepare(source, deps)['status'] == 'prepared'
    assert entered
