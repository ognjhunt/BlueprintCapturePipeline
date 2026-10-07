"""Protected runtime preparation is separate from issuing deletion authority."""
from pathlib import Path
import fcntl
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
    # Each worker otherwise changes the same protected home directory's mtime
    # while another is checking its identity. Retain one shared lock inode;
    # unlinking it could let later workers lock a different file concurrently.
    fd = os.open(Path.home() / '.blueprint-scene-fixture.lock',
                 os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, 'a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix='.blueprint-scene-fixture-', dir=Path.home()) as name:
            yield Path(name).resolve()


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/install_scene_retirement_runtime.py'


def _synthetic_sha256_inventory(checkout, commit, wanted):
    """Fixture authority at the crypto boundary; production never builds its own trust."""
    import hashlib
    import subprocess
    rows = []
    raw = subprocess.check_output(['/usr/bin/git', '-C', str(checkout), 'ls-tree', '-rz', commit])
    for item in raw.split(b'\0'):
        if not item:
            continue
        head, name = item.split(b'\t', 1)
        mode, kind, oid = head.decode().split()
        name = name.decode()
        if not any(name == root or name.startswith(root+'/') for root in wanted):
            continue
        assert kind == 'blob'
        body = subprocess.check_output(['/usr/bin/git', '-C', str(checkout), 'cat-file', 'blob', oid])
        rows.append({'path':name, 'git_blob_oid':oid, 'mode':mode,
                     'size':len(body), 'sha256':hashlib.sha256(body).hexdigest()})
    return sorted(rows, key=lambda row:row['path'])


def fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('runtime_install_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path.resolve() / 'source'
    (source / 'src/blueprint_pipeline').mkdir(parents=True)
    (source / 'deploy/systemd').mkdir(parents=True)
    (source / 'scripts').mkdir()
    (source / 'docs/schemas').mkdir(parents=True)
    (source / 'docs/schemas/fixture.schema.json').write_bytes(b'{"type":"object"}\n')
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
    admitted = module._authenticated_git_entries
    def synthetic_entries(checkout, commit, wanted, deadline, **kwargs):
        if kwargs.get('source_manifest') is None:
            repository = kwargs.get('repository', 'ognjhunt/BlueprintCapturePipeline')
            kwargs['source_manifest'] = {'sources':[{'repository':repository, 'commit':commit,
                'files':_synthetic_sha256_inventory(checkout, commit, wanted)}]}
        return admitted(checkout, commit, wanted, deadline, **kwargs)
    monkeypatch.setattr(module, '_authenticated_git_entries', synthetic_entries)
    monkeypatch.setattr(module, '_admitted_source_manifest', lambda *args: None)
    return module, source, deps


@pytest.mark.parametrize('supplied', [0.0, float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('entry', ['prepare', 'refresh', 'build_sdk', 'prepare_deployment'])
def test_installation_rejects_expired_or_nonfinite_caller_deadline(tmp_path, monkeypatch, supplied, entry):
    module, source, deps = fixture(tmp_path, monkeypatch)
    arguments = {'_deadline': supplied}
    if entry == 'refresh':
        arguments['expected_current'] = module._selector(b'{}')
    if entry == 'prepare_deployment':
        arguments['source_commit'] = '0' * 40
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        getattr(module, entry)(source, *([deps] if entry in {'prepare', 'refresh'} else []), **arguments)
    assert not module._BOOT_ROOT.exists() and not module._RUNTIME_ROOT.exists()


def test_installation_deadline_is_finite_clamped_and_never_renewed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module, _, _ = fixture(tmp_path, monkeypatch)
    now = [100.0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    assert module._MAX_SECONDS == 900
    assert module._installation_deadline() == module._installation_deadline(2000.0) == 1000.0
    original = module._installation_deadline(175.0)
    now[0] = 150.0
    assert module._installation_deadline(original) == 175.0
    now[0] = 176.0
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module._installation_deadline(original)


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


def test_partial_sdk_directory_validation_scans_manifest_once(tmp_path, monkeypatch):
    import time
    module, _, deps = fixture(tmp_path, monkeypatch)
    names = [f'package_{index}' for index in range(48)]
    names += ['package-name', 'package.name', 'package[0]', 'space name', 'μpackage']
    for name in names:
        directory = deps / name / 'nested'
        directory.mkdir(parents=True)
        (directory / '__init__.py').write_bytes(b'# protected SDK\n')
    rows, sources = {}, {}
    deadline = time.monotonic() + 10
    module._tree(deps, Path('.'), rows, sources, deadline)
    class CountedRows(dict):
        scans = 0
        def __iter__(self):
            self.scans += 1
            return super().__iter__()
    counted = CountedRows(rows)
    module._partial_tree(deps, counted, sources, deadline)
    assert counted.scans == 1


def test_sdk_tree_reuses_protected_parent_descriptors(tmp_path, monkeypatch):
    import time
    module, _, deps = fixture(tmp_path, monkeypatch)
    nested = deps / 'package' / 'nested'
    nested.mkdir(parents=True)
    for index in range(64):
        (nested / f'leaf-{index}.py').write_bytes(b'# protected SDK\n')
    opened = []
    original = module._open
    def observed(path, **kwargs):
        opened.append(path)
        return original(path, **kwargs)
    monkeypatch.setattr(module, '_open', observed)
    rows, sources = {}, {}
    module._tree(deps, Path('.'), rows, sources, time.monotonic() + 10)
    assert len(rows) == 65 and all(sources[name].is_file() for name in rows)
    # Full ancestry is proven on entry and exit. Descendants are authenticated
    # through their retained parent, with every byte still read and hashed.
    assert opened == [deps, deps]


@pytest.mark.parametrize('change', ['directory-alias', 'ancestor-alias', 'writable-ancestor'])
def test_sdk_tree_rejects_path_or_ancestry_change_during_walk(tmp_path, monkeypatch, change):
    import time
    module, _, deps = fixture(tmp_path, monkeypatch)
    ancestor = tmp_path / 'ancestor'
    ancestor.mkdir()
    deps.rename(ancestor / deps.name)
    deps = ancestor / deps.name
    nested = deps / 'package'
    nested.mkdir()
    leaf = nested / 'leaf.py'
    leaf.write_bytes(b'# protected SDK\n')
    original = module._read_open_file
    def changed(fd, path, deadline, **kwargs):
        result = original(fd, path, deadline, **kwargs)
        if path == leaf:
            if change == 'directory-alias':
                retained = deps / 'renamed'
                nested.rename(retained)
                nested.symlink_to(retained, target_is_directory=True)
            elif change == 'ancestor-alias':
                retained = tmp_path / 'renamed-ancestor'
                ancestor.rename(retained)
                ancestor.symlink_to(retained, target_is_directory=True)
            else:
                tmp_path.chmod(0o777)
        return result
    monkeypatch.setattr(module, '_read_open_file', changed)
    try:
        with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
            module._tree(deps, Path('.'), {}, {}, time.monotonic() + 10)
    finally:
        tmp_path.chmod(0o700)
        if ancestor.is_symlink():
            ancestor.unlink()
            (tmp_path / 'renamed-ancestor').rename(ancestor)
    assert not module._BOOT_ROOT.exists()


@pytest.mark.parametrize('change', ['bytes', 'replacement'])
def test_sdk_tree_rejects_leaf_change_during_read(tmp_path, monkeypatch, change):
    import time
    module, _, deps = fixture(tmp_path, monkeypatch)
    leaf = deps / 'trusted_sdk.py'
    identity = leaf.stat()
    original = module.os.read
    mutated = []
    def changed(fd, amount):
        result = original(fd, amount)
        info = module.os.fstat(fd)
        if not mutated and (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            mutated.append(True)
            if change == 'bytes':
                leaf.write_bytes(b'value = 2\n')
            else:
                replacement = deps / 'replacement'
                replacement.write_bytes(b'value = 1\n')
                replacement.replace(leaf)
        return result
    monkeypatch.setattr(module.os, 'read', changed)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module._tree(deps, Path('.'), {}, {}, time.monotonic() + 10)
    assert mutated and not module._BOOT_ROOT.exists()


def test_sdk_tree_closes_child_descriptors_when_deadline_expires(tmp_path, monkeypatch):
    module, _, deps = fixture(tmp_path, monkeypatch)
    (deps / 'package').mkdir()
    (deps / 'package' / 'leaf.py').write_bytes(b'# protected SDK\n')
    opened, clock = set(), [1.0]
    real_open, real_close, real_child = module.os.open, module.os.close, module._open_tree_child
    def observed_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.add(fd)
        return fd
    def observed_close(fd):
        opened.discard(fd)
        return real_close(fd)
    def expired(*args, **kwargs):
        fd = real_child(*args, **kwargs)
        clock[0] = 11.0
        return fd
    monkeypatch.setattr(module.os, 'open', observed_open)
    monkeypatch.setattr(module.os, 'close', observed_close)
    monkeypatch.setattr(module, '_open_tree_child', expired)
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module._tree(deps, Path('.'), {}, {}, 10.0)
    assert not opened and not module._BOOT_ROOT.exists()


@pytest.mark.parametrize('kind', ['directory', 'leaf'])
@pytest.mark.parametrize('restore_mode', [False, True])
def test_sdk_tree_refuses_changed_identity_at_descriptor_handoff(tmp_path, monkeypatch, kind, restore_mode):
    import time
    module, _, deps = fixture(tmp_path, monkeypatch)
    nested = deps / 'package'
    nested.mkdir()
    leaf = nested / 'leaf.py'
    leaf.write_bytes(b'# protected SDK\n')
    target = nested if kind == 'directory' else leaf
    original = module._open_tree_child
    mutated = []
    def changed(parent, name, before, *, directory):
        fd = original(parent, name, before, directory=directory)
        if not mutated and name == target.name and directory == (kind == 'directory'):
            mutated.append(True)
            target.chmod(0o777 if directory else 0o666)
            if restore_mode:
                if directory:
                    (target / 'injected.py').write_bytes(b'# not admitted at open\n')
                else:
                    target.write_bytes(b'# changed after open\n')
                target.chmod(0o755 if directory else 0o644)
        return fd
    monkeypatch.setattr(module, '_open_tree_child', changed)
    rows = {}
    try:
        with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
            module._tree(deps, Path('.'), rows, {}, time.monotonic() + 10)
    finally:
        target.chmod(0o755 if kind == 'directory' else 0o644)
    assert mutated and 'package/leaf.py' not in rows
    assert not module._BOOT_ROOT.exists()


@pytest.mark.parametrize('change', ['unknown-directory', 'prefix-collision', 'symlink', 'changed-prefix'])
def test_partial_sdk_directory_index_preserves_refusals(tmp_path, monkeypatch, change):
    import time
    module, _, deps = fixture(tmp_path, monkeypatch)
    directory = deps / 'package' / 'nested'
    directory.mkdir(parents=True)
    leaf = directory / '__init__.py'
    leaf.write_bytes(b'# protected SDK\n')
    rows, sources = {}, {}
    deadline = time.monotonic() + 10
    module._tree(deps, Path('.'), rows, sources, deadline)
    if change == 'unknown-directory':
        (deps / 'foreign').mkdir()
    elif change == 'prefix-collision':
        (deps / 'pack').mkdir()
    elif change == 'symlink':
        (deps / 'foreign').symlink_to(directory, target_is_directory=True)
    else:
        leaf.write_bytes(b'# altered SDK\n')
    before = leaf.read_bytes()
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module._partial_tree(deps, rows, sources, deadline)
    assert leaf.read_bytes() == before
    assert not module._BOOT_ROOT.exists()


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
    # The real source tree contains an empty package marker. It is still an
    # authenticated Git blob and must survive the protected release copy.
    (source / 'src/blueprint_pipeline/_sam_parser_js').mkdir()
    (source / 'src/blueprint_pipeline/_sam_parser_js/__init__.py').write_bytes(b'')
    subprocess.run(['/usr/bin/git', '-C', str(source), 'init', '-q'], check=True)
    subprocess.run(['/usr/bin/git', '-C', str(source), 'add', 'src', 'scripts', 'deploy', 'docs', 'uv.lock'], check=True)
    subprocess.run(['/usr/bin/git', '-C', str(source), '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'signed source-shaped fixture'], check=True)
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(source), 'rev-parse', 'HEAD']).decode().strip()
    # Service-owned checkout bytes may drift; installed root source must come
    # from the exact already-authorized Git object, not these mutable bytes.
    (source / 'src/blueprint_pipeline/__init__.py').write_bytes(b'raise RuntimeError("uncommitted mutable source")\n')
    (source / 'docs/schemas/fixture.schema.json').write_bytes(b'{"uncommitted":true}\n')
    phases = []
    result = module.prepare_deployment(source, source_commit=commit, wheelhouse=wheel.parent, _progress=phases.append)
    assert phases == ['source_attestation', 'signed_release', 'build_sdk', 'prepare', 'publish_installer']
    assert result['status'] == 'prepared'
    assert result['source_commit'] == commit
    assert (module._RUNTIME_ROOT / 'src/blueprint_pipeline/__init__.py').read_bytes() == b'# trusted package\n'
    assert (module._RUNTIME_ROOT / 'src/blueprint_pipeline/_sam_parser_js/__init__.py').read_bytes() == b''
    assert (module._RUNTIME_ROOT / 'docs/schemas/fixture.schema.json').read_bytes() == b'{"type":"object"}\n'
    assert (module._RUNTIME_ROOT / 'dependencies/fixture_sdk/__init__.py').read_bytes() == b'value = 1\n'
    assert (module._BOOT_ROOT / 'continuous_bootstrap.py').is_file()
    assert result['authority_issued'] is False and result['cleanup_enabled'] is False
    phases.clear()
    repeated = module.prepare_deployment(source, source_commit=commit, wheelhouse=wheel.parent, _progress=phases.append)
    assert phases == ['source_attestation', 'signed_release', 'build_sdk', 'refresh', 'publish_installer']
    assert repeated['status'] == 'refreshed'
    assert (module._RUNTIME_ROOT / 'src/blueprint_pipeline/_sam_parser_js/__init__.py').read_bytes() == b''


def test_signed_release_copies_real_lockfile_size_with_exact_git_bytes(tmp_path, monkeypatch):
    import subprocess
    import time

    module, source, _ = fixture(tmp_path, monkeypatch)
    # The checked-in uv.lock is 1,165,102 bytes at this gate. Keep the test
    # tiny relative to the runtime budget while crossing the old 1 MiB cap.
    lock = b'lock\n' + b'x' * (1_165_102 - len(b'lock\n'))
    (source / 'uv.lock').write_bytes(lock)
    for index in range(33):
        (source / 'src/blueprint_pipeline' / f'binary_{index}.data').write_bytes(bytes([index]) + b'\0\n')
    subprocess.run(['/usr/bin/git', '-C', str(source), 'init', '-q'], check=True)
    subprocess.run(['/usr/bin/git', '-C', str(source), 'add', 'src', 'scripts', 'deploy', 'docs', 'uv.lock'], check=True)
    subprocess.run(['/usr/bin/git', '-C', str(source), '-c', 'user.name=fixture',
        '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'bounded signed lock'], check=True)
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(source), 'rev-parse', 'HEAD']).decode().strip()

    calls = []
    command = module._sdk_git_command
    def observed(checkout, arguments, deadline, **kwargs):
        calls.append(arguments)
        return command(checkout, arguments, deadline, **kwargs)
    monkeypatch.setattr(module, '_sdk_git_command', observed)
    copied = module._signed_release(source, commit, time.monotonic() + 30)
    assert (copied / 'uv.lock').read_bytes() == lock
    for index in range(33):
        assert (copied / 'src/blueprint_pipeline' / f'binary_{index}.data').read_bytes() == bytes([index]) + b'\0\n'
    batches = calls.count(['cat-file', '--batch'])
    assert batches == calls.count(['cat-file', '--batch-check']) == 3
    assert not any(arguments[:2] in (['cat-file', '-s'], ['cat-file', 'blob']) for arguments in calls)
    assert module._signed_release(source, commit, time.monotonic() + 30) == copied
    assert calls.count(['cat-file', '--batch']) == batches


@pytest.mark.parametrize('change', ['size-limit', 'wrong-type', 'wrong-id', 'missing', 'extra-header',
                                  'truncated', 'trailing', 'mutated-blob', 'body-header'])
def test_signed_blob_batch_refuses_unproven_headers_and_bytes(tmp_path, monkeypatch, change):
    import hashlib
    import time
    module, source, _ = fixture(tmp_path, monkeypatch)
    raw = b'protected\0\nbytes'
    digest = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    header = digest.encode() + b' blob ' + str(len(raw)).encode() + b'\n'
    checked, body = header, header + raw + b'\n'
    if change == 'size-limit':
        checked = digest.encode() + b' blob 1048577\n'
    elif change == 'wrong-type':
        checked = header.replace(b' blob ', b' tree ')
    elif change == 'wrong-id':
        checked = b'0' * 40 + header[40:]
    elif change == 'missing':
        checked = digest.encode() + b' missing\n'
    elif change == 'extra-header':
        checked += header
    elif change == 'truncated':
        body = body[:-1]
    elif change == 'trailing':
        body += b'foreign'
    elif change == 'mutated-blob':
        body = header + b'X' + raw[1:] + b'\n'
    else:
        body = header.replace(b' blob ', b' tree ') + raw + b'\n'
    calls = []
    def command(checkout, arguments, deadline, **kwargs):
        assert kwargs['input_data'] == digest.encode() + b'\n'
        assert kwargs['raw_checkout'] is True
        calls.append(arguments)
        return checked if arguments == ['cat-file', '--batch-check'] else body
    monkeypatch.setattr(module, '_sdk_git_command', command)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module._signed_release_blobs(source, [('100644', digest, 'src/blueprint_pipeline/fixture.py', len(raw), hashlib.sha256(raw).hexdigest())], time.monotonic()+5)
    if change in {'size-limit', 'wrong-type', 'wrong-id', 'missing', 'extra-header'}:
        assert calls == [['cat-file', '--batch-check']]
    assert not module._BOOT_ROOT.exists() and not module._RUNTIME_ROOT.exists()


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
        fp = type('FP', (), {'raw': type('Raw', (), {
            '_sock': type('Socket', (), {'settimeout': lambda *args: None})()})()})()
        def __init__(self, fail):
            self.offset = 0
            self.fail = fail
        def __enter__(self): return self
        def __exit__(self, *arguments): return False
        def read1(self, amount):
            if self.fail and self.offset:
                raise OSError('actual interrupted native download')
            result = raw[self.offset:self.offset + (1 if self.fail else amount)]
            self.offset += len(result)
            return result
    def open_response(url, timeout):
        requests.append(url)
        return Response(len(requests) == 1)
    monkeypatch.setattr(module.urllib.request, 'build_opener', lambda *args: type('Opener', (), {'open':staticmethod(open_response)})())
    deadline = time.monotonic() + 10
    with pytest.raises(OSError, match='actual interrupted'):
        module._sdk_artifact(row, None, deadline)
    partial = next(module._sdk_root().rglob('*.whl.pending'))
    inode = partial.stat().st_ino
    result = module._sdk_artifact(row, None, deadline)
    assert result.read_bytes() == raw and result.stat().st_ino == inode
    assert not partial.exists() and len(requests) == 2


def _deployer_runtime_fixture(monkeypatch, tmp_path):
    import ast
    import hashlib
    import json
    import re
    import stat
    import subprocess
    tree = ast.parse((SCRIPT.parent / 'deploy_control_plane_commit.py').read_bytes())
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and node.name in {'_prepare_scene_retirement_runtime', '_bootstrap_scene_retirement_installer', '_scene_runtime_diagnostic'}]
    namespace = {'Path': Path, 'Any': object, 'os': os, 'hashlib': hashlib, 'json': json, 're': re,
                 'stat': stat, 'subprocess': subprocess, 'time': __import__('time'),
                 'ControlPlaneDeployError': ValueError,
                 '_SCENE_RUNTIME_BOOT_ROOT': tmp_path / 'root-runtime', '_SCENE_RUNTIME_OWNER': os.getuid()}
    allowance = next(node for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == '_SCENE_RUNTIME_INSTALL_SECONDS'
                             for target in node.targets))
    namespace['_SCENE_RUNTIME_INSTALL_SECONDS'] = ast.literal_eval(allowance.value)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SCRIPT), 'exec'), namespace)
    def synthetic_attestation(commit, *, deadline):
        # These tests exercise protected publication/execution after crypto
        # admission, never substitute this fixture for production verification.
        source = active_source[0]
        files = _synthetic_sha256_inventory(source, commit, ('scripts',))
        installer = next(row for row in files if row['path'] == 'scripts/install_scene_retirement_runtime.py')
        if not any(row['path'] == 'scripts/release_source_manifest.py' for row in files):
            files.append(installer | {'path':'scripts/release_source_manifest.py'})
        manifest = {'sources':[{'repository':'ognjhunt/BlueprintCapturePipeline','commit':commit,'files':files}]}
        return manifest, json.dumps(manifest).encode(), b'{}'
    active_source = [None]
    bootstrap = namespace['_bootstrap_scene_retirement_installer']
    def admitted_bootstrap(source, commit, **kwargs):
        active_source[0] = source
        return bootstrap(source, commit, **kwargs)
    namespace['_bootstrap_scene_retirement_installer'] = admitted_bootstrap
    namespace['_scene_source_attestation'] = synthetic_attestation
    return namespace


def test_first_upgrade_authenticates_installer_data_from_service_owned_release_before_execution(tmp_path, monkeypatch):
    import subprocess
    import hashlib
    import json
    module, source, _ = fixture(tmp_path, monkeypatch)
    _sdk_wheel_fixture(source, tmp_path / 'wheelhouse')
    subprocess.run(['/usr/bin/git', '-C', str(source), 'init', '-q'], check=True)
    subprocess.run(['/usr/bin/git', '-C', str(source), 'add', 'src', 'scripts', 'deploy', 'docs', 'uv.lock'], check=True)
    subprocess.run(['/usr/bin/git', '-C', str(source), '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'authorized first upgrade'], check=True)
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(source), 'rev-parse', 'HEAD']).decode().strip()
    signed = (source / 'scripts/install_scene_retirement_runtime.py').read_bytes()
    (source / 'scripts/install_scene_retirement_runtime.py').write_bytes(b'raise RuntimeError("mutable checkout must never execute as root")\n')
    # This is a genuinely writable/service-owned source ancestry. The bootstrap
    # source must be authenticated Git DATA; root protection applies to output.
    source.chmod(0o777)
    namespace = _deployer_runtime_fixture(monkeypatch, tmp_path)
    from types import SimpleNamespace
    namespace['time'] = SimpleNamespace(monotonic=lambda: 100.0)
    assert namespace['_SCENE_RUNTIME_INSTALL_SECONDS'] == module._MAX_SECONDS == 900
    actual_run = subprocess.run
    executed = []
    def record_root_execution(command, **kwargs):
        if command[:3] == ['/usr/bin/python3', '-I', '-S']:
            assert float(command[command.index('--deadline-monotonic') + 1]) == 1000.0
            assert kwargs['timeout'] == 900.0
            fd = int(command[3].rsplit('/', 1)[1])
            raw = os.pread(fd, 1024 * 1024, 0)
            assert raw == signed
            protected = namespace['_SCENE_RUNTIME_BOOT_ROOT'] / 'installers' / commit / 'runtime_installer.py'
            assert protected.read_bytes() == signed
            assert not protected.stat().st_mode & 0o022
            receipt = json.loads((protected.parent / 'runtime-installer.json').read_bytes())
            assert receipt['sha256'] == 'sha256:' + hashlib.sha256(signed).hexdigest()
            executed.append(raw)
            value = {'status': 'prepared', 'source_commit': commit,
                     'authority_issued': False, 'cleanup_enabled': False}
            return subprocess.CompletedProcess(command, 0, json.dumps(value).encode(), b'')
        return actual_run(command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', record_root_execution)
    result = namespace['_prepare_scene_retirement_runtime'](source_repo=source, source_commit=commit)
    assert result['source_commit'] == commit and executed == [signed]
    assert not (namespace['_SCENE_RUNTIME_BOOT_ROOT'] / 'CURRENT.json').exists()


def test_first_upgrade_unknown_fixed_installer_refuses_without_overwriting_or_execution(tmp_path, monkeypatch):
    namespace = _deployer_runtime_fixture(monkeypatch, tmp_path)
    root = namespace['_SCENE_RUNTIME_BOOT_ROOT']
    root.mkdir()
    helper = root / 'runtime_installer.py'
    helper.write_bytes(b'unknown prior root helper')
    before = helper.stat()
    with pytest.raises(ValueError, match='deploy_scene_retirement_runtime_unproven'):
        namespace['_prepare_scene_retirement_runtime'](source_repo=tmp_path, source_commit='1' * 40)
    assert helper.read_bytes() == b'unknown prior root helper' and helper.stat().st_ino == before.st_ino


def test_public_contracts_fetch_uses_fixed_https_and_verifies_real_git_objects(tmp_path, monkeypatch):
    import subprocess
    import time
    module, source, _ = fixture(tmp_path, monkeypatch)
    repository = tmp_path / 'contracts'
    (repository / 'src/blueprint_contracts').mkdir(parents=True)
    (repository / 'src/blueprint_contracts/__init__.py').write_text('VALUE = 1\n')
    subprocess.run(['/usr/bin/git', '-C', str(repository), 'init', '-q'], check=True)
    subprocess.run(['/usr/bin/git', '-C', str(repository), 'add', '.'], check=True)
    subprocess.run(['/usr/bin/git', '-C', str(repository), '-c', 'user.name=fixture',
                    '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'contracts'], check=True)
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(repository), 'rev-parse', 'HEAD']).decode().strip()
    actual_popen = subprocess.Popen
    fetched = []

    def local_transport(command, **kwargs):
        if 'fetch' in command:
            expected = 'https://github.com/ognjhunt/BlueprintContracts.git'
            assert expected in command and command[-1] == commit
            assert 'GIT_SSH_COMMAND' not in kwargs['env']
            assert kwargs['env']['GIT_CONFIG_GLOBAL'] == '/dev/null'
            assert kwargs['env']['GIT_TERMINAL_PROMPT'] == '0'
            assert 'credential.helper=' in command
            fetched.append(command.copy())
            command = [str(repository) if arg == expected else arg for arg in command]
        return actual_popen(command, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', local_transport)
    package = {'name': 'blueprint-contracts', 'source': {
        'git': f'https://github.com/ognjhunt/BlueprintContracts.git?rev={commit}#{commit}'}}
    rows = module._sdk_git_rows(package, None, time.monotonic() + 30)
    assert len(fetched) == 1
    assert Path(rows['blueprint_contracts/__init__.py']['source']).read_text() == 'VALUE = 1\n'


def test_upgrade_executes_authenticated_candidate_instead_of_valid_obsolete_helper(tmp_path, monkeypatch):
    import hashlib
    import json
    import subprocess
    namespace = _deployer_runtime_fixture(monkeypatch, tmp_path)
    root = namespace['_SCENE_RUNTIME_BOOT_ROOT']
    root.mkdir()
    old = b'raise RuntimeError("obsolete fetch path")\n'
    (root / 'runtime_installer.py').write_bytes(old)
    (root / 'runtime-installer.json').write_text(json.dumps({
        'schema': 'scene-retirement-runtime-installer.v1',
        'sha256': 'sha256:' + hashlib.sha256(old).hexdigest(), 'size_bytes': len(old)}))
    source = tmp_path / 'candidate'
    (source / 'scripts').mkdir(parents=True)
    candidate = b'# reviewed new installer\n'
    (source / 'scripts/install_scene_retirement_runtime.py').write_bytes(candidate)
    subprocess.run(['/usr/bin/git', '-C', str(source), 'init', '-q'], check=True)
    subprocess.run(['/usr/bin/git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['/usr/bin/git', '-C', str(source), '-c', 'user.name=fixture',
                    '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'candidate'], check=True)
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(source), 'rev-parse', 'HEAD']).decode().strip()
    (source / 'scripts/install_scene_retirement_runtime.py').write_bytes(b'raise RuntimeError("mutable")\n')
    actual_run = subprocess.run
    executed = []

    def execute(command, **kwargs):
        if command[:3] == ['/usr/bin/python3', '-I', '-S']:
            raw = os.pread(int(command[3].rsplit('/', 1)[1]), 1024 * 1024, 0)
            assert raw == candidate
            assert (root / 'runtime_installer.py').read_bytes() == old
            executed.append(raw)
            return subprocess.CompletedProcess(command, 0, json.dumps({
                'status': 'refreshed', 'source_commit': commit,
                'authority_issued': False, 'cleanup_enabled': False}).encode(), b'')
        return actual_run(command, **kwargs)

    monkeypatch.setattr(subprocess, 'run', execute)
    for _ in range(2):
        namespace['_prepare_scene_retirement_runtime'](source_repo=source, source_commit=commit)
    assert executed == [candidate, candidate]
    selected = root / 'installers' / commit / 'runtime_installer.py'
    selected.write_bytes(b'changed candidate')
    with pytest.raises(ValueError, match='deploy_scene_retirement_runtime_unproven'):
        namespace['_prepare_scene_retirement_runtime'](source_repo=source, source_commit=commit)
    assert executed == [candidate, candidate]


@pytest.mark.parametrize('receipt', ['runtime-installer.json', 'runtime-installer-pending.json'])
def test_orphan_retained_installer_receipt_refuses_before_candidate_staging(tmp_path, monkeypatch, receipt):
    namespace = _deployer_runtime_fixture(monkeypatch, tmp_path)
    root = namespace['_SCENE_RUNTIME_BOOT_ROOT']
    root.mkdir()
    marker = root / receipt
    marker.write_bytes(b'{}')
    def unexpected(*args, **kwargs):
        pytest.fail('orphan installer receipt must refuse before candidate staging')
    namespace['_bootstrap_scene_retirement_installer'] = unexpected
    with pytest.raises(ValueError, match='deploy_scene_retirement_runtime_unproven'):
        namespace['_prepare_scene_retirement_runtime'](source_repo=tmp_path, source_commit='1' * 40)
    assert marker.read_bytes() == b'{}' and not (root / 'installers').exists()


@pytest.mark.parametrize('batch', [False, True])
def test_native_git_output_is_bounded_before_parent_buffer_growth(tmp_path, monkeypatch, batch):
    import stat
    import subprocess
    import time
    module, _, _ = fixture(tmp_path, monkeypatch)
    checkout, commit = _pinned_contracts_checkout(tmp_path)
    blob = subprocess.check_output(['/usr/bin/git', '-C', str(checkout), 'rev-parse',
        commit + ':src/blueprint_contracts/__init__.py']).decode().strip()
    real_read = module.os.read
    native_bytes = []
    def counted_read(fd, amount):
        fifo = stat.S_ISFIFO(os.fstat(fd).st_mode)
        raw = real_read(fd, amount)
        if fifo and raw:
            native_bytes.append(len(raw))
        return raw
    monkeypatch.setattr(module.os, 'read', counted_read)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module._sdk_git_command(checkout, ['cat-file', '--batch'] if batch else ['cat-file', 'blob', blob],
                                time.monotonic() + 5, cap=4,
                                input_data=blob.encode() + b'\n' if batch else None)
    assert sum(native_bytes) <= 5, 'native stdout cap must apply before an oversized read/append, not after communicate allocated output'


def test_native_git_zero_byte_blob_remains_supported_with_zero_output_allowance(tmp_path, monkeypatch):
    import subprocess
    import time
    module, _, _ = fixture(tmp_path, monkeypatch)
    checkout, _ = _pinned_contracts_checkout(tmp_path)
    blob = subprocess.check_output(['/usr/bin/git', '-C', str(checkout), 'hash-object', '-w', '--stdin'], input=b'').decode().strip()
    assert module._sdk_git_command(checkout, ['cat-file', 'blob', blob], time.monotonic() + 5, cap=0) == b''


def test_native_git_batch_deadline_kills_and_reaps_owned_child(tmp_path, monkeypatch):
    import subprocess
    import time
    module, _, _ = fixture(tmp_path, monkeypatch)
    checkout, _ = _pinned_contracts_checkout(tmp_path)
    popen, children = subprocess.Popen, []
    def stalled(command, **kwargs):
        assert command[-2:] == ['cat-file', '--batch']
        child = popen([sys.executable, '-I', '-S', '-c', 'import time; time.sleep(30)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, 'Popen', stalled)
    started = time.monotonic()
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module._sdk_git_command(checkout, ['cat-file', '--batch'], started + .05,
                                input_data=b'0' * 40 + b'\n')
    assert len(children) == 1 and children[0].poll() is not None
    assert time.monotonic() - started < 5


@pytest.mark.parametrize('phase', ['download', 'extraction'])
def test_sdk_preserves_protected_disk_floor_before_native_download_or_payload_write(tmp_path, monkeypatch, phase):
    import hashlib
    import time
    from types import SimpleNamespace
    module, source, _ = fixture(tmp_path, monkeypatch)
    wheel = _sdk_wheel_fixture(source, tmp_path / 'wheelhouse')
    monkeypatch.setattr(module, '_FREE_FLOOR', 1)
    monkeypatch.setattr(module.os, 'statvfs', lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
    if phase == 'download':
        raw = wheel.read_bytes()
        row = {'url': 'https://files.pythonhosted.org/' + wheel.name, 'size': len(raw),
               'hash': 'sha256:' + hashlib.sha256(raw).hexdigest()}
        monkeypatch.setattr(module.urllib.request, 'build_opener', lambda *a, **kw: pytest.fail('SDK floor refusal must precede native download'))
        with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
            module._sdk_artifact(row, None, time.monotonic()+5)
        assert not list(module._sdk_root().glob('wheel-artifacts/*/*.pending'))
    else:
        rows = module._wheel_entries(wheel, time.monotonic()+5)
        destination = module._sdk_root() / 'space-refused-sdk'
        with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
            module._sdk_extract(destination, rows, time.monotonic()+5)
        assert not destination.exists()


@pytest.mark.parametrize('tampered', [False, True])
def test_system_python_without_tomllib_bootstraps_only_exact_locked_protected_tomli(tmp_path, monkeypatch, tampered):
    import builtins
    import hashlib
    import importlib.util
    import re
    import zipfile
    module, source, _ = fixture(tmp_path, monkeypatch)
    wheel = _sdk_wheel_fixture(source, tmp_path / 'wheelhouse')
    parser_source = Path(importlib.util.find_spec('tomli').origin).parent
    init = (parser_source / '__init__.py').read_bytes()
    version = re.search(rb'__version__\s*=\s*"([^"]+)"', init)[1].decode()
    parser_wheel = wheel.parent / ('tomli-' + version + '-py3-none-any.whl')
    with zipfile.ZipFile(parser_wheel, 'w') as archive:
        for path in sorted(parser_source.glob('*.py')):
            archive.writestr('tomli/' + path.name, path.read_bytes())
        archive.writestr('tomli-' + version + '.dist-info/METADATA',
            'Metadata-Version: 2.1\nName: tomli\nVersion: ' + version + '\n')
        archive.writestr('tomli-' + version + '.dist-info/WHEEL',
            'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
    digest = hashlib.sha256(parser_wheel.read_bytes()).hexdigest()
    text = (source / 'uv.lock').read_text().replace('dependencies = [{ name = "fixture-sdk" }]',
        'dependencies = [{ name = "fixture-sdk" }, { name = "tomli" }]')
    text += '\n[[package]]\nname = "tomli"\nversion = "' + version + '"\nsource = { registry = "https://pypi.org/simple" }\n'
    text += 'wheels = [{ url = "https://files.pythonhosted.org/' + parser_wheel.name + '", hash = "sha256:' + digest + '", size = ' + str(parser_wheel.stat().st_size) + ' }]\n'
    (source / 'uv.lock').write_text(text)
    if tampered:
        parser_wheel.write_bytes(parser_wheel.read_bytes() + b'foreign parser code')
    original_import = builtins.__import__
    def no_stdlib_tomllib(name, *arguments, **kwargs):
        if name == 'tomllib':
            raise ImportError('actual pre-3.11 system parser availability')
        return original_import(name, *arguments, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_stdlib_tomllib)
    if tampered:
        with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
            module.build_sdk(source, wheelhouse=wheel.parent)
        assert not (module._BOOT_ROOT / 'CURRENT.json').exists()
    else:
        selected = module.build_sdk(source, wheelhouse=wheel.parent)
        assert {'name': 'tomli', 'version': version} in selected['packages']
        assert (Path(selected['dependencies_root']) / 'tomli/__init__.py').read_bytes() == init
        assert selected['authority_issued'] is False


def test_locked_sdk_preserves_actual_uv_multiple_extra_dependencies(tmp_path, monkeypatch):
    module, _, _ = fixture(tmp_path, monkeypatch)
    packages = [
        {'name': 'blueprint-capture-pipeline', 'version': '1', 'source': {'editable': '.'},
         'dependencies': [{'name': 'dependency', 'extra': ['grpc', 'requests']}]},
        {'name': 'dependency', 'version': '1', 'dependencies': [{'name': 'base'}],
         'optional-dependencies': {'grpc': [{'name': 'grpc-child'}], 'requests': [{'name': 'requests-child'}]}},
        *({'name': name, 'version': '1'} for name in ('base', 'grpc-child', 'requests-child')),
    ]
    selected = module._sdk_closure(packages, None)
    assert {row['name'] for row in selected} == {'dependency', 'base', 'grpc-child', 'requests-child'}


def test_actual_locked_production_sdk_dependency_closure_includes_pubsub_grpc(tmp_path, monkeypatch):
    import tomllib
    from packaging import markers
    module, _, _ = fixture(tmp_path, monkeypatch)
    lock = tomllib.loads((Path(__file__).parents[1] / 'uv.lock').read_text())
    selected = module._sdk_closure(lock['package'], (markers, None, None))
    names = {row['name'] for row in selected}
    assert {'google-cloud-pubsub', 'google-api-core', 'grpcio', 'grpcio-status', 'blueprint-contracts'} <= names


def test_actual_locked_sdk_includes_cpu_control_plane_runtime_imports_without_gpu_extra(tmp_path, monkeypatch):
    import tomllib
    from packaging import markers
    module, _, _ = fixture(tmp_path, monkeypatch)
    lock = tomllib.loads((Path(__file__).parents[1] / 'uv.lock').read_text())
    selected = module._sdk_closure(lock['package'], (markers, None, None))
    names = {row['name'] for row in selected}
    assert {'opencv-python-headless', 'build123d', 'pycollada', 'trimesh'} <= names
    # These are the root runtime's CPU import dependencies. No model/GPU
    # operator is invoked by this administrative protected-runtime install.
    assert not {'ultralytics', 'torch', 'nvidia-cuda-runtime-cu12'} & names


def test_many_member_wheel_parses_directory_once_for_consecutive_payloads(tmp_path, monkeypatch):
    import time
    import zipfile
    module, _, _ = fixture(tmp_path, monkeypatch)
    wheel = tmp_path / 'many.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for index in range(5620):
            archive.writestr(f'package/file{index:05}.py', f'value = {index}\n')
    rows = module._wheel_entries(wheel, time.monotonic()+30)
    # A bounded sample still carries the real, large central directory. Count
    # parsing work rather than asserting a machine-dependent wall clock.
    sample = dict(list(rows.items())[:128])
    real_zip = zipfile.ZipFile
    opened = []
    def observed(*args, **kwargs):
        archive = real_zip(*args, **kwargs)
        opened.append(archive)
        return archive
    monkeypatch.setattr(module.zipfile, 'ZipFile', observed)
    destination = tmp_path / 'extracted'
    module._sdk_extract(destination, sample, time.monotonic()+60)
    assert len(opened) == 1
    assert all(archive.fp is None for archive in opened)
    for index in range(128):
        assert (destination / f'package/file{index:05}.py').read_text() == f'value = {index}\n'
    module._sdk_extract(destination, sample, time.monotonic()+30)
    assert len(opened) == 1  # Complete retained files are validated, not re-extracted.


@pytest.mark.parametrize('change', ['replace', 'writable'])
def test_cached_archive_refuses_identity_change_between_members(tmp_path, monkeypatch, change):
    import time
    import zipfile
    module, _, _ = fixture(tmp_path, monkeypatch)
    wheel = tmp_path / 'source.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('a.py', 'a = 1\n')
        archive.writestr('b.py', 'b = 2\n')
    rows = module._wheel_entries(wheel, time.monotonic()+10)
    destination = tmp_path / 'extracted'
    real_read = module._read
    real_zip = zipfile.ZipFile
    opened = []
    def observed(*args, **kwargs):
        archive = real_zip(*args, **kwargs)
        opened.append(archive)
        return archive
    def mutate(path, deadline, **kwargs):
        result = real_read(path, deadline, **kwargs)
        if path == destination / 'a.py':
            if change == 'replace':
                replacement = tmp_path / 'replacement.whl'
                replacement.write_bytes(wheel.read_bytes())
                replacement.replace(wheel)
            else:
                wheel.chmod(0o666)
        return result
    monkeypatch.setattr(module, '_read', mutate)
    monkeypatch.setattr(module.zipfile, 'ZipFile', observed)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module._sdk_extract(destination, rows, time.monotonic()+10)
    assert (destination / 'a.py').read_text() == 'a = 1\n'
    assert not (destination / 'b.py').exists()
    assert len(opened) == 1 and opened[0].fp is None


def test_cached_archive_switch_and_interruption_resume_preserve_exact_bytes(tmp_path, monkeypatch):
    import time
    import zipfile
    module, _, _ = fixture(tmp_path, monkeypatch)
    rows = {}
    for index, names in enumerate([['a.py', 'c.py'], ['b.py']]):
        wheel = tmp_path / f'source{index}.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            for name in names:
                archive.writestr(name, name.encode()*100)
        rows.update(module._wheel_entries(wheel, time.monotonic()+10))
    destination = tmp_path / 'extracted'
    append = module._sdk_append_chunk
    def interrupt(output, original, raw, offset, deadline):
        append(output, original, raw[:10], offset, deadline)
        raise ValueError(module._ERROR)
    monkeypatch.setattr(module, '_sdk_append_chunk', interrupt)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module._sdk_extract(destination, rows, time.monotonic()+10)
    assert (destination / 'a.py.pending').read_bytes() == (b'a.py'*100)[:10]
    monkeypatch.setattr(module, '_sdk_append_chunk', append)
    module._sdk_extract(destination, rows, time.monotonic()+10)
    for name in rows:
        assert (destination / name).read_bytes() == name.encode()*100
    assert not list(destination.glob('*.pending'))


def _interrupted_connected_install(tmp_path, monkeypatch):
    import subprocess
    module, source, _ = fixture(tmp_path, monkeypatch)
    wheel = _sdk_wheel_fixture(source, tmp_path / 'wheelhouse')
    def git(*args):
        return subprocess.check_output(['/usr/bin/git', '-C', str(source), *args], stderr=subprocess.DEVNULL).decode().strip()
    git('init', '-q')
    git('add', 'src', 'scripts', 'deploy', 'docs', 'uv.lock')
    git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'initial')
    first = git('rev-parse', 'HEAD')
    copy = module._copy
    def interrupt(path, destination, expected, deadline):
        if destination == module._RUNTIME_ROOT / 'src/blueprint_pipeline/__init__.py':
            copy(path, destination, expected, deadline)
            raise ValueError(module._ERROR)
        return copy(path, destination, expected, deadline)
    monkeypatch.setattr(module, '_copy', interrupt)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.prepare_deployment(source, source_commit=first, wheelhouse=wheel.parent)
    monkeypatch.setattr(module, '_copy', copy)
    return module, source, wheel, first, git


@pytest.mark.parametrize('next_commit', [False, True])
def test_connected_deployment_resumes_initial_intent_before_refresh(tmp_path, monkeypatch, next_commit):
    module, source, wheel, first, git = _interrupted_connected_install(tmp_path, monkeypatch)
    intent = module._BOOT_ROOT / 'installation.json'
    original = intent.read_bytes()
    if next_commit:
        (source / 'scripts/install_scene_retirement_runtime.py').write_bytes(b'# new authenticated helper data\n')
        (source / 'src/blueprint_pipeline/__init__.py').write_bytes(b'# new release\n')
        git('add', 'scripts', 'src')
        git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'next')
    target = git('rev-parse', 'HEAD')
    result = module.prepare_deployment(source, source_commit=target, wheelhouse=wheel.parent)
    assert result['status'] == 'refreshed' and result['source_commit'] == target
    assert result['authority_issued'] is False and result['cleanup_enabled'] is False
    assert intent.read_bytes() == original
    assert (module._RUNTIME_ROOT / 'src/blueprint_pipeline/__init__.py').read_bytes() == b'# trusted package\n'
    import json
    selected = json.loads((module._BOOT_ROOT / 'CURRENT.json').read_bytes())
    assert (Path(selected['runtime_root']) / 'src/blueprint_pipeline/__init__.py').read_bytes() == (b'# new release\n' if next_commit else b'# trusted package\n')


@pytest.mark.parametrize('change', ['source', 'sdk', 'partial', 'foreign', 'missing'])
def test_initial_resume_preserves_unmatched_or_tampered_inputs(tmp_path, monkeypatch, change):
    module, source, wheel, first, git = _interrupted_connected_install(tmp_path, monkeypatch)
    intent = module._BOOT_ROOT / 'installation.json'
    original = intent.read_bytes()
    cache = module._sdk_root() / 'release-inputs' / first
    if change == 'source':
        (cache / 'src/blueprint_pipeline/__init__.py').write_bytes(b'# changed retained source\n')
    elif change == 'missing':
        (cache / 'scripts/scene_retirement_continuous_bootstrap.py').unlink()
    elif change == 'sdk':
        next((module._sdk_root() / 'sdk-inputs').glob('*/fixture_sdk/__init__.py')).write_bytes(b'# changed retained sdk\n')
    elif change == 'foreign':
        (module._RUNTIME_ROOT / 'foreign.py').write_bytes(b'# unknown\n')
    else:
        (module._RUNTIME_ROOT / 'src/blueprint_pipeline/__init__.py').write_bytes(b'# wrong partial\n')
    (source / 'scripts/install_scene_retirement_runtime.py').write_bytes(b'# next helper\n')
    git('add', 'scripts')
    git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'next')
    with pytest.raises((ValueError, FileNotFoundError), match='scene_retirement_runtime_unproven|No such file'):
        module.prepare_deployment(source, source_commit=git('rev-parse', 'HEAD'), wheelhouse=wheel.parent)
    assert intent.read_bytes() == original
    assert not (module._BOOT_ROOT / 'CURRENT.json').exists()
    assert not (module._BOOT_ROOT / 'continuous_bootstrap.py').exists()


def test_initial_resume_survives_interruption_before_refresh(tmp_path, monkeypatch):
    module, source, wheel, first, _ = _interrupted_connected_install(tmp_path, monkeypatch)
    original = (module._BOOT_ROOT / 'installation.json').read_bytes()
    refresh = module.refresh
    monkeypatch.setattr(module, 'refresh', lambda *args, **kwargs: (_ for _ in ()).throw(ValueError(module._ERROR)))
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.prepare_deployment(source, source_commit=first, wheelhouse=wheel.parent)
    assert (module._BOOT_ROOT / 'continuous_bootstrap.py').is_file()
    assert not (module._BOOT_ROOT / 'CURRENT.json').exists()
    monkeypatch.setattr(module, 'refresh', refresh)
    assert module.prepare_deployment(source, source_commit=first, wheelhouse=wheel.parent)['status'] == 'refreshed'
    assert (module._BOOT_ROOT / 'installation.json').read_bytes() == original


def test_signed_release_retry_authenticates_retained_git_blob_without_reacquiring(tmp_path, monkeypatch):
    import time
    module, source, _, first, _ = _interrupted_connected_install(tmp_path, monkeypatch)
    git = module._sdk_git_command
    def existing(checkout, arguments, deadline, **kwargs):
        assert arguments[:2] not in (['cat-file', 'blob'], ['cat-file', '-s'],
                                     ['cat-file', '--batch'], ['cat-file', '--batch-check'])
        return git(checkout, arguments, deadline, **kwargs)
    monkeypatch.setattr(module, '_sdk_git_command', existing)
    root = module._signed_release(source, first, time.monotonic()+10)
    assert (root / 'src/blueprint_pipeline/__init__.py').read_bytes() == b'# trusted package\n'
    (root / 'src/blueprint_pipeline/__init__.py').write_bytes(b'# changed retained source\n')
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module._signed_release(source, first, time.monotonic()+10)


def test_initial_resume_uses_retained_sdk_when_new_release_changes_dependencies(tmp_path, monkeypatch):
    import hashlib
    import json
    import zipfile
    module, source, wheel, first, git = _interrupted_connected_install(tmp_path, monkeypatch)
    original = (module._BOOT_ROOT / 'installation.json').read_bytes()
    old_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    old_size = wheel.stat().st_size
    replacement = tmp_path / 'next-wheelhouse' / wheel.name
    replacement.parent.mkdir()
    with zipfile.ZipFile(wheel) as before, zipfile.ZipFile(replacement, 'w') as after:
        for name in before.namelist():
            after.writestr(name, b'value = 2\n' if name.endswith('/__init__.py') else before.read(name))
    lock = source / 'uv.lock'
    lock.write_text(lock.read_text().replace(old_hash, hashlib.sha256(replacement.read_bytes()).hexdigest())
                    .replace('size = ' + str(old_size), 'size = ' + str(replacement.stat().st_size)))
    (source / 'src/blueprint_pipeline/__init__.py').write_bytes(b'# changed release\n')
    git('add', 'uv.lock', 'src')
    git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'new SDK')
    module.prepare_deployment(source, source_commit=git('rev-parse', 'HEAD'), wheelhouse=replacement.parent)
    assert (module._BOOT_ROOT / 'installation.json').read_bytes() == original
    assert (module._RUNTIME_ROOT / 'dependencies/fixture_sdk/__init__.py').read_bytes() == b'value = 1\n'
    selected = json.loads((module._BOOT_ROOT / 'CURRENT.json').read_bytes())
    assert (Path(selected['dependencies_root']) / 'fixture_sdk/__init__.py').read_bytes() == b'value = 2\n'


def test_initial_resume_retries_interrupted_original_dependency_copy(tmp_path, monkeypatch):
    module, source, wheel, first, _ = _interrupted_connected_install(tmp_path, monkeypatch)
    original = (module._BOOT_ROOT / 'installation.json').read_bytes()
    copy = module._copy
    def interrupt(path, destination, expected, deadline):
        result = copy(path, destination, expected, deadline)
        if destination == module._RUNTIME_ROOT / 'dependencies/fixture_sdk/__init__.py':
            raise ValueError(module._ERROR)
        return result
    monkeypatch.setattr(module, '_copy', interrupt)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.prepare_deployment(source, source_commit=first, wheelhouse=wheel.parent)
    assert not (module._BOOT_ROOT / 'continuous_bootstrap.py').exists()
    monkeypatch.setattr(module, '_copy', copy)
    assert module.prepare_deployment(source, source_commit=first, wheelhouse=wheel.parent)['status'] == 'refreshed'
    assert (module._BOOT_ROOT / 'installation.json').read_bytes() == original


@pytest.mark.parametrize('timed_out', [False, True])
def test_deploy_installer_failure_preserves_blocker_and_only_fixed_diagnostics(tmp_path, monkeypatch, timed_out):
    import json
    import hashlib
    import subprocess
    namespace = _deployer_runtime_fixture(monkeypatch, tmp_path)
    commit = 'a' * 40
    candidate = namespace['_SCENE_RUNTIME_BOOT_ROOT'] / 'installers' / commit
    candidate.mkdir(parents=True)
    raw = b'# authenticated installer fixture\n'
    (candidate / 'runtime_installer.py').write_bytes(raw)
    (candidate / 'runtime-installer.json').write_text(json.dumps({
        'schema': 'scene-retirement-runtime-installer.v1',
        'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}))
    namespace['_bootstrap_scene_retirement_installer'] = lambda *args, **kwargs: {
        'source_manifest':candidate/'source-sha256-manifest.json',
        'source_attestation':candidate/'source-provenance.sigstore.json',
        'manifest_verifier':candidate/'release_source_manifest.py'}
    private = b'private-canary-path?token=private-canary-credential'
    stderr = (private + b'\nscene_retirement_runtime_phase:build_sdk\n'
              b'scene_retirement_runtime_failure:validation\n'
              b'scene_retirement_runtime_phase:' + private + b'\n')
    def failed(command, **kwargs):
        if timed_out:
            raise subprocess.TimeoutExpired(command, kwargs['timeout'], output=private, stderr=stderr)
        return subprocess.CompletedProcess(command, 2, private, stderr)
    monkeypatch.setattr(subprocess, 'run', failed)
    with pytest.raises(ValueError, match='^deploy_scene_retirement_runtime_unproven$') as caught:
        namespace['_prepare_scene_retirement_runtime'](source_repo=tmp_path, source_commit=commit)
    assert caught.value.runtime_diagnostic == {'phase': 'build_sdk', 'reason': 'deadline' if timed_out else 'validation'}
    assert private.decode() not in str(caught.value.runtime_diagnostic)
    assert not (namespace['_SCENE_RUNTIME_BOOT_ROOT'] / 'CURRENT.json').exists()


@pytest.mark.parametrize('stderr', [b'private-canary', b'x' * 65537,
    b'scene_retirement_runtime_phase:build_sdk/secret\nscene_retirement_runtime_failure:unknown',
    b'\xffscene_retirement_runtime_phase:build_sdk'])
def test_deploy_diagnostic_rejects_unrecognized_or_unbounded_markers(tmp_path, monkeypatch, stderr):
    namespace = _deployer_runtime_fixture(monkeypatch, tmp_path)
    assert namespace['_scene_runtime_diagnostic'](stderr, phase='execute_installer', reason='validation') == {
        'phase': 'execute_installer', 'reason': 'validation'}


@pytest.mark.parametrize('reason', ['deadline', 'validation', 'io', 'unexpected'])
def test_installer_cli_failure_emits_no_private_exception_or_traceback(tmp_path, monkeypatch, capsys, reason):
    from types import SimpleNamespace
    module, source, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module, 'os', SimpleNamespace(getuid=lambda: 0, geteuid=lambda: 0))
    monkeypatch.setattr(module, 'sys', SimpleNamespace(flags=SimpleNamespace(isolated=True, no_site=True),
        stderr=sys.stderr))
    def failed(*args, **kwargs):
        kwargs['_progress']('resume_initial_intent')
        error = OSError if reason == 'io' else RuntimeError if reason == 'unexpected' else ValueError
        raise error('private-canary-path?token=private-canary-credential')
    monkeypatch.setattr(module, 'prepare_deployment', failed)
    options = ['--source', str(source), '--source-commit', 'a'*40, '--locked-sdk']
    if reason == 'deadline':
        options += ['--deadline-monotonic', '0']
    assert module.main(options) == 2
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == (('' if reason == 'deadline' else 'scene_retirement_runtime_phase:resume_initial_intent\n')
                            + 'scene_retirement_runtime_failure:' + reason + '\n')


@pytest.mark.parametrize('supplied', [None, 250.0])
def test_connected_installation_keeps_one_deadline_through_resume_and_refresh(tmp_path, monkeypatch, supplied):
    from types import SimpleNamespace
    module, source, wheel, commit, _ = _interrupted_connected_install(tmp_path, monkeypatch)
    original_intent = (module._BOOT_ROOT / 'installation.json').read_bytes()
    now = [100.0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    deadline = 1000.0 if supplied is None else supplied
    phases = []
    def phase(name):
        phases.append(name)
        now[0] = {'source_attestation': 100.0, 'signed_release': 100.0, 'build_sdk': 100.0+(deadline-100.0)/3,
                  'resume_initial_intent': 100.0+2*(deadline-100.0)/3,
                  'refresh': deadline+1}[name]
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.prepare_deployment(source, source_commit=commit, wheelhouse=wheel.parent,
                                  _deadline=supplied, _progress=phase)
    assert phases == ['source_attestation', 'signed_release', 'build_sdk', 'resume_initial_intent', 'refresh']
    assert (module._BOOT_ROOT / 'continuous_bootstrap.py').is_file()
    assert (module._BOOT_ROOT / 'installation.json').read_bytes() == original_intent
    assert not (module._BOOT_ROOT / 'CURRENT.json').exists()


@pytest.mark.parametrize('selection', ['--dependencies', '--venv', '--locked-sdk'])
def test_installer_cli_retains_earlier_caller_deadline_for_every_mode(tmp_path, monkeypatch, selection):
    from types import SimpleNamespace
    module, source, deps = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module, 'os', SimpleNamespace(getuid=lambda: 0, geteuid=lambda: 0))
    monkeypatch.setattr(module, 'sys', SimpleNamespace(flags=SimpleNamespace(isolated=True, no_site=True), stderr=sys.stderr))
    now, observed = [100.0], []
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    def resolve(venv):
        now[0] = 120.0  # Dependency discovery consumes the original allowance.
        return deps
    monkeypatch.setattr(module, 'dependency_root', resolve)
    def prepare(*args, **kwargs):
        observed.append(kwargs['_deadline'])
        return {'status': 'prepared'}
    monkeypatch.setattr(module, 'prepare', prepare)
    monkeypatch.setattr(module, 'prepare_deployment', prepare)
    options = ['--source', str(source), '--deadline-monotonic', '150', selection]
    options += ['--source-commit', 'a'*40] if selection == '--locked-sdk' else [str(deps)]
    assert module.main(options) == 0
    assert observed == [150.0]


def test_schema_consumer_reads_each_immutable_runtime_generation(tmp_path, monkeypatch):
    import json

    module, source, deps = fixture(tmp_path, monkeypatch)
    repository = SCRIPT.parent.parent
    name = 'rigid_task_success_contract_schema.py'
    schema_name = 'rigid_task_success_contract.v1.schema.json'
    (source / 'src/blueprint_pipeline' / name).write_bytes(
        (repository / 'src/blueprint_pipeline' / name).read_bytes())
    schema = source / 'docs/schemas' / schema_name
    original = (repository / 'docs/schemas' / schema_name).read_bytes()
    schema.write_bytes(original)

    def consume(root):
        spec = importlib.util.spec_from_file_location('installed_schema_consumer', root / 'src/blueprint_pipeline' / name)
        consumer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(consumer)
        assert consumer.SCHEMA_PATH == root / 'docs/schemas' / schema_name
        return consumer.rigid_task_success_contract_schema()

    module.prepare(source, deps)
    assert consume(module._RUNTIME_ROOT) == json.loads(original)
    import hashlib
    prior_raw = (module._BOOT_ROOT / 'installation.json').read_bytes()
    prior = {'sha256': 'sha256:' + hashlib.sha256(prior_raw).hexdigest(), 'size_bytes': len(prior_raw)}
    changed = json.loads(original)
    changed['$comment'] = 'fixture second authenticated schema generation'
    schema.write_text(json.dumps(changed))
    result = module.refresh(source, deps, expected_current=prior)
    selected = Path(result['runtime_root'])
    assert consume(selected) == changed
    assert consume(module._RUNTIME_ROOT) == json.loads(original)
    assert (module._RUNTIME_ROOT / 'docs/schemas' / schema_name).read_bytes() == original


def test_schema_resource_symlink_refuses_runtime_preparation(tmp_path, monkeypatch):
    module, source, deps = fixture(tmp_path, monkeypatch)
    schema = source / 'docs/schemas/fixture.schema.json'
    outside = tmp_path / 'foreign-schema.json'
    outside.write_bytes(schema.read_bytes())
    schema.unlink()
    schema.symlink_to(outside)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module.prepare(source, deps)
    assert not (module._BOOT_ROOT / 'installation.json').exists()
