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
    (source / 'scripts/install_scene_retirement_runtime.py').write_bytes(SCRIPT.read_bytes())
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


def _runtime_selector(path):
    import hashlib
    raw = path.read_bytes()
    return {"sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def test_explicit_refresh_publishes_one_verified_source_sdk_cohort(tmp_path, monkeypatch):
    module, source, deps = fixture(tmp_path, monkeypatch)
    module.prepare(source, deps)
    old_source = module._RUNTIME_ROOT / "src/blueprint_pipeline/__init__.py"
    old_dependency = module._RUNTIME_ROOT / "dependencies/trusted_sdk.py"
    prior = _runtime_selector(module._BOOT_ROOT / "installation.json")
    (source / "src/blueprint_pipeline/__init__.py").write_bytes(b"# new trusted package\n")
    (deps / "trusted_sdk.py").write_bytes(b"value = 2\n")
    result = module.refresh(source, deps, expected_current=prior)
    selected = Path(result["runtime_root"])
    assert result["status"] == "refreshed" and selected != module._RUNTIME_ROOT
    assert selected.is_relative_to(module._RUNTIME_ROOT / "generations")
    assert (selected / "src/blueprint_pipeline/__init__.py").read_bytes() == b"# new trusted package\n"
    assert (Path(result["dependencies_root"]) / "trusted_sdk.py").read_bytes() == b"value = 2\n"
    assert old_source.read_bytes() == b"# trusted package\n"
    assert old_dependency.read_bytes() == b"value = 1\n"
    assert result["current"] == _runtime_selector(module._BOOT_ROOT / "CURRENT.json")
    assert result["authority_issued"] is False and result["cleanup_enabled"] is False


def test_interrupted_refresh_keeps_old_current_and_resumes_exact_new_cohort(tmp_path, monkeypatch):
    module, source, deps = fixture(tmp_path, monkeypatch)
    module.prepare(source, deps)
    prior = _runtime_selector(module._BOOT_ROOT / "installation.json")
    (source / "src/blueprint_pipeline/__init__.py").write_bytes(b"# new trusted package\n")
    original = module._copy
    changed = False
    def interrupt(path, destination, expected, deadline):
        nonlocal changed
        if not changed:
            changed = True
            original(path, destination, expected, deadline)
            raise OSError("interrupted exact generation copy")
        return original(path, destination, expected, deadline)
    monkeypatch.setattr(module, "_copy", interrupt)
    with pytest.raises(ValueError, match="scene_retirement_runtime_unproven"):
        module.refresh(source, deps, expected_current=prior)
    assert not (module._BOOT_ROOT / "CURRENT.json").exists()
    assert _runtime_selector(module._BOOT_ROOT / "installation.json") == prior
    assert (module._RUNTIME_ROOT / "src/blueprint_pipeline/__init__.py").read_bytes() == b"# trusted package\n"
    monkeypatch.setattr(module, "_copy", original)
    result = module.refresh(source, deps, expected_current=prior)
    assert result["status"] == "refreshed"
    assert result["current"] == _runtime_selector(module._BOOT_ROOT / "CURRENT.json")


def test_refresh_stale_current_refuses_before_copying_new_generation(tmp_path, monkeypatch):
    module, source, deps = fixture(tmp_path, monkeypatch)
    module.prepare(source, deps)
    (source / "src/blueprint_pipeline/__init__.py").write_bytes(b"# new trusted package\n")
    with pytest.raises(ValueError, match="scene_retirement_runtime_unproven"):
        module.refresh(source, deps, expected_current={"sha256": "sha256:" + "0" * 64, "size_bytes": 1})
    assert not (module._RUNTIME_ROOT / "generations").exists()
    assert not (module._BOOT_ROOT / "CURRENT.json").exists()


def test_source_refresh_reuses_identical_protected_sdk_generation(tmp_path, monkeypatch):
    module, source, deps = fixture(tmp_path, monkeypatch)
    module.prepare(source, deps)
    previous = _runtime_selector(module._BOOT_ROOT / "installation.json")
    (source / "src/blueprint_pipeline/__init__.py").write_bytes(b"# protected source generation two\n")
    first = module.refresh(source, deps, expected_current=previous)
    sdk = Path(first["dependencies_root"]) / "trusted_sdk.py"
    before = sdk.stat()
    (source / "src/blueprint_pipeline/__init__.py").write_bytes(b"# protected source generation three\n")
    second = module.refresh(source, deps, expected_current=first["current"])
    assert second["runtime_root"] != first["runtime_root"]
    assert second["dependencies_root"] == first["dependencies_root"]
    after = sdk.stat()
    assert (before.st_dev, before.st_ino, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_mtime_ns)


def _sdk_wheel_fixture(source, directory):
    import hashlib
    import zipfile
    directory.mkdir()
    artifact = directory / 'fixture_sdk-1.0-py3-none-any.whl'
    with zipfile.ZipFile(artifact, 'w') as wheel:
        wheel.writestr('fixture_sdk/__init__.py', 'value = 1\n')
        wheel.writestr('fixture_sdk-1.0.dist-info/METADATA', 'Metadata-Version: 2.1\nName: fixture-sdk\nVersion: 1.0\n')
        wheel.writestr('fixture_sdk-1.0.dist-info/WHEEL', 'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (source / 'uv.lock').write_text(
        'version = 1\n[[package]]\nname = "blueprint-capture-pipeline"\nversion = "2.0.0"\n'
        'source = { editable = "." }\ndependencies = [{ name = "fixture-sdk" }]\n'
        '[[package]]\nname = "fixture-sdk"\nversion = "1.0"\nsource = { registry = "https://pypi.org/simple" }\n'
        'wheels = [{ url = "https://files.pythonhosted.org/' + artifact.name + '", hash = "sha256:' + digest + '", size = ' + str(artifact.stat().st_size) + ' }]\n')
    return artifact


def test_actual_locked_sdk_build_extracts_full_production_closure_without_executing(tmp_path, monkeypatch):
    module, source, _ = fixture(tmp_path, monkeypatch)
    wheel = _sdk_wheel_fixture(source, tmp_path / 'wheelhouse')
    selected = module.build_sdk(source, wheelhouse=wheel.parent)
    sdk = Path(selected['dependencies_root'])
    assert (sdk / 'fixture_sdk/__init__.py').read_bytes() == b'value = 1\n'
    assert selected['packages'] == [{'name': 'fixture-sdk', 'version': '1.0'}]
    assert selected['system_python_abi'] == f'{sys.version_info.major}.{sys.version_info.minor}'
    assert selected['authority_issued'] is False and selected['cleanup_enabled'] is False
    assert not list(sdk.rglob('*.pyc'))


@pytest.mark.parametrize('failure', ['tampered-wheel', 'missing-transitive'])
def test_locked_sdk_missing_or_foreign_artifacts_refuse_before_publication(tmp_path, monkeypatch, failure):
    module, source, _ = fixture(tmp_path, monkeypatch)
    wheel = _sdk_wheel_fixture(source, tmp_path / 'wheelhouse')
    if failure == 'tampered-wheel':
        wheel.write_bytes(wheel.read_bytes() + b'foreign')
    else:
        text = (source / 'uv.lock').read_text()
        (source / 'uv.lock').write_text(text.replace('name = "fixture-sdk"\nversion', 'dependencies = [{ name = "missing-sdk" }]\nname = "fixture-sdk"\nversion'))
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.build_sdk(source, wheelhouse=wheel.parent)
    assert not (module._BOOT_ROOT / 'CURRENT.json').exists()
    assert not (module._BOOT_ROOT / 'continuous_bootstrap.py').exists()


def _pinned_contracts_checkout(tmp_path):
    import subprocess
    root = tmp_path / 'contracts'
    (root / 'src/blueprint_contracts').mkdir(parents=True)
    (root / 'src/blueprint_contracts/__init__.py').write_bytes(b'raise RuntimeError("SDK preparation must not execute packages")\n')
    def git(*args):
        return subprocess.check_output(['/usr/bin/git', '-C', str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
    git('init', '-q')
    git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'add', 'src')
    git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'tiny pinned contract fixture')
    return root, git('rev-parse', 'HEAD')


def test_locked_sdk_includes_exact_pinned_git_contract_tree_without_build_hooks(tmp_path, monkeypatch):
    module, source, _ = fixture(tmp_path, monkeypatch)
    wheel = _sdk_wheel_fixture(source, tmp_path / 'wheelhouse')
    contracts, commit = _pinned_contracts_checkout(tmp_path)
    text = (source / 'uv.lock').read_text().replace(
        'dependencies = [{ name = "fixture-sdk" }]',
        'dependencies = [{ name = "fixture-sdk" }, { name = "blueprint-contracts" }]')
    text += '\n[[package]]\nname = "blueprint-contracts"\nversion = "0.1.0"\nsource = { git = "https://github.com/ognjhunt/BlueprintContracts.git?rev=' + commit + '#' + commit + '" }\n'
    (source / 'uv.lock').write_text(text)
    result = module.build_sdk(source, wheelhouse=wheel.parent, contracts_checkout=contracts)
    installed = Path(result['dependencies_root']) / 'blueprint_contracts/__init__.py'
    assert installed.read_bytes().startswith(b'raise RuntimeError(')
    assert result['packages'] == [{'name': 'blueprint-contracts', 'version': '0.1.0'}, {'name': 'fixture-sdk', 'version': '1.0'}]


def test_runtime_refresh_selects_only_exact_generation_after_many_prior_deploys(tmp_path, monkeypatch):
    import hashlib
    module, source, deps = fixture(tmp_path, monkeypatch)
    module.prepare(source, deps)
    raw = (module._BOOT_ROOT / 'installation.json').read_bytes()
    parent = module._RUNTIME_ROOT / 'generations'
    parent.mkdir()
    for index in range(33):
        (parent / f'{index:064x}').mkdir()
    (source / 'src/blueprint_pipeline/__init__.py').write_bytes(b'# next current source\n')
    result = module.refresh(source, deps, expected_current={'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)})
    assert result['status'] == 'refreshed'
    assert len(list(parent.iterdir())) == 34
    assert all((parent / f'{index:064x}').is_dir() for index in range(33))


def test_connected_deployment_prepares_signed_source_sdk_before_exposing_units(tmp_path, monkeypatch):
    import subprocess
    module, source, _ = fixture(tmp_path, monkeypatch)
    wheel = _sdk_wheel_fixture(source, tmp_path / 'wheelhouse')
    subprocess.run(['/usr/bin/git', '-C', str(source), 'init', '-q'], check=True)
    subprocess.run(['/usr/bin/git', '-C', str(source), 'add', 'src', 'scripts', 'deploy', 'uv.lock'], check=True)
    subprocess.run(['/usr/bin/git', '-C', str(source), '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'signed source-shaped fixture'], check=True)
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(source), 'rev-parse', 'HEAD']).decode().strip()
    # Service-owned checkout bytes may drift; installed root source must come
    # from the exact already-authorized Git object, not these mutable bytes.
    (source / 'src/blueprint_pipeline/__init__.py').write_bytes(b'raise RuntimeError("uncommitted mutable source")\n')
    result = module.prepare_deployment(source, source_commit=commit, wheelhouse=wheel.parent)
    assert result['status'] == 'prepared'
    assert result['source_commit'] == commit
    assert (module._RUNTIME_ROOT / 'src/blueprint_pipeline/__init__.py').read_bytes() == b'# trusted package\n'
    assert (module._RUNTIME_ROOT / 'dependencies/fixture_sdk/__init__.py').read_bytes() == b'value = 1\n'
    assert (module._BOOT_ROOT / 'continuous_bootstrap.py').is_file()
    assert result['authority_issued'] is False and result['cleanup_enabled'] is False


def test_live_installer_prepares_immutable_runtime_before_service_ownership_and_units():
    value = (SCRIPT.parent / 'install_live_pipeline_control_plane.sh').read_text()
    selected = value.index('install_scene_retirement_runtime.py')
    assert selected < value.index('run chown -R')
    assert selected < value.index('blueprint-pipeline-intake.service')
    assert '/usr/bin/python3 -I -S' in value[:selected]


def test_deployer_provisions_root_runtime_before_release_activation():
    value = (SCRIPT.parent / 'deploy_control_plane_commit.py').read_text()
    body = value[value.index('        staged_release = stage_task_evaluation_control_plane_release('):]
    assert body.index('_prepare_scene_retirement_runtime(') < body.index('activate=True')
    assert 'scene_retirement_runtime' in body[:body.index('activate=True')]


def test_raw_signed_release_git_cannot_execute_repository_promisor_helper(tmp_path, monkeypatch):
    import subprocess
    import time
    module, source, _ = fixture(tmp_path, monkeypatch)
    marker = tmp_path / 'mutable-helper-executed'
    helper = tmp_path / 'mutable-helper.sh'
    helper.write_text('#!/bin/sh\ntouch ' + str(marker) + '\nexit 1\n')
    helper.chmod(0o700)
    def git(*arguments):
        subprocess.run(['/usr/bin/git', '-C', str(source), *arguments], check=True, capture_output=True)
    git('init', '-q')
    git('config', 'remote.origin.url', 'ext::/bin/sh ' + str(helper))
    git('config', 'remote.origin.promisor', 'true')
    git('config', 'protocol.ext.allow', 'always')
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module._sdk_git_command(source, ['cat-file', 'blob', '1' * 40], time.monotonic() + 5, raw_checkout=True)
    assert not marker.exists(), 'root acquisition must not execute mutable repository remote helpers before authentication'


def test_locked_sdk_resumes_exact_owned_partial_extraction_without_recopying_completed_bytes(tmp_path, monkeypatch):
    module, source, _ = fixture(tmp_path, monkeypatch)
    wheel = _sdk_wheel_fixture(source, tmp_path / 'wheelhouse')
    write = module.os.write
    interrupted = False
    def fail_after_prefix(fd, value):
        nonlocal interrupted
        if not interrupted and bytes(value).startswith(b'value = 1'):
            interrupted = True
            write(fd, value[:1])
            raise OSError('actual partial SDK extraction interruption')
        return write(fd, value)
    monkeypatch.setattr(module.os, 'write', fail_after_prefix)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.build_sdk(source, wheelhouse=wheel.parent)
    partial = next(module._sdk_root().rglob('__init__.py.pending'))
    inode = partial.stat().st_ino
    assert partial.read_bytes() == b'v'
    monkeypatch.setattr(module.os, 'write', write)
    selected = module.build_sdk(source, wheelhouse=wheel.parent)
    final = Path(selected['dependencies_root']) / 'fixture_sdk/__init__.py'
    assert final.read_bytes() == b'value = 1\n'
    assert final.stat().st_ino == inode
    assert not partial.exists()


def test_locked_sdk_resumes_hash_bound_download_prefix_under_same_origin(tmp_path, monkeypatch):
    import hashlib
    import time
    module, source, _ = fixture(tmp_path, monkeypatch)
    wheel = _sdk_wheel_fixture(source, tmp_path / 'wheelhouse')
    raw = wheel.read_bytes()
    row = {'url': 'https://files.pythonhosted.org/' + wheel.name, 'size': len(raw),
           'hash': 'sha256:' + hashlib.sha256(raw).hexdigest()}
    requests = []
    class Response:
        status = 200
        url = row['url']
        def __init__(self, fail):
            self.offset = 0
            self.fail = fail
        def __enter__(self): return self
        def __exit__(self, *arguments): return False
        def read(self, amount):
            if self.fail and self.offset:
                raise OSError('actual interrupted native download')
            result = raw[self.offset:self.offset + (1 if self.fail else amount)]
            self.offset += len(result)
            return result
    def open_response(url, timeout):
        requests.append(url)
        return Response(len(requests) == 1)
    monkeypatch.setattr(module.urllib.request, 'urlopen', open_response)
    deadline = time.monotonic() + 10
    with pytest.raises(OSError, match='actual interrupted'):
        module._sdk_artifact(row, None, deadline)
    partial = next(module._sdk_root().rglob('*.whl.pending'))
    inode = partial.stat().st_ino
    result = module._sdk_artifact(row, None, deadline)
    assert result.read_bytes() == raw and result.stat().st_ino == inode
    assert not partial.exists() and len(requests) == 2
