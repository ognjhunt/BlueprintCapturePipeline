"""ADP-009D/day28: fixed installed historical entry isolates privileged imports."""
# Covers: deploy/operator-door/historical-generation-entry.py
# Covers: deploy/operator-door/stage-historical-runtime.py

import importlib.util
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata, protected_root_tmp_path  # noqa: F401


def entry():
    path = Path(__file__).parents[1] / 'deploy/operator-door/historical-generation-entry.py'
    spec = importlib.util.spec_from_file_location('historical_entry_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('arguments', [[], ['a' * 32, '/tmp/path'], ['../escape'], ['a' * 31], ['--config']])
def test_installed_entry_accepts_only_exact_action_id_before_application_import(arguments):
    module = entry()
    with pytest.raises(ValueError, match='historical_generation_entry_unproven'):
        module.main(arguments)


@pytest.mark.parametrize('kind', ['writable', 'symlink', 'hardlink'])
def test_root_loader_rejects_unprotected_import_bytes(protected_root_tmp_path, root_metadata, kind):  # noqa: F811
    import os
    module = entry()
    source = protected_root_tmp_path / 'module.py'
    source.write_bytes(b'value = 1\n')
    if kind == 'writable':
        source.chmod(0o666)
    elif kind == 'symlink':
        original = source.with_name('original.py')
        source.rename(original)
        source.symlink_to(original)
    else:
        os.link(source, source.with_name('alias.py'))
    with pytest.raises(ValueError, match='historical_generation_entry_unproven'):
        module._read_source(source)


def test_staged_runtime_uses_actual_bounded_production_import_closure(tmp_path):
    path = Path(__file__).parents[1] / 'deploy/operator-door/stage-historical-runtime.py'
    spec = importlib.util.spec_from_file_location('historical_stage_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(__file__).parents[1]
    staged = tmp_path / 'historical-python'
    module.stage(root / 'src/blueprint_pipeline', staged)
    package = staged / 'blueprint_pipeline'
    assert (package / 'control_plane_lane_historical_action.py').read_bytes() == (
        root / 'src/blueprint_pipeline/control_plane_lane_historical_action.py').read_bytes()
    assert (package / 'control_plane_lane_historical_restore_worker.py').is_file()
    assert (package / 'control_plane_lane_historical_gc.py').is_file()
    assert not (package / 'live_pipeline_intake_service.py').exists()
    assert sum(path.stat().st_size for path in package.iterdir()) <= 8 * 1024**2


def test_staged_runtime_rejects_aggregate_bytes_before_writing_overbudget_module(tmp_path):
    path = Path(__file__).parents[1] / 'deploy/operator-door/stage-historical-runtime.py'
    spec = importlib.util.spec_from_file_location('historical_stage_budget_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    origin = tmp_path / 'source'
    origin.mkdir()
    for name in ('__init__', 'control_plane_lane_historical_action'):
        (origin / (name + '.py')).write_text('pass\n')
    names = ['control_plane_lane_historical_gc', *(f'part_{index}' for index in range(8))]
    for index, name in enumerate(names):
        following = f'from . import {names[index + 1]}\n' if index + 1 < len(names) else 'pass\n'
        raw = b'#' + b'x' * 950_000 + b'\n' + following.encode()
        assert len(raw) < 1024**2  # Every source independently meets its per-file bound.
        (origin / (name + '.py')).write_bytes(raw)
    staged = tmp_path / 'historical-python'
    with pytest.raises(ValueError, match='historical_generation_stage_unproven'):
        module.stage(origin, staged)
    package = staged / 'blueprint_pipeline'
    assert (package / (names[-2] + '.py')).is_file()
    assert not (package / (names[-1] + '.py')).exists()
    assert sum(path.stat().st_size for path in package.iterdir()) <= 8 * 1024**2


@pytest.mark.parametrize('archive', [False, True])
def test_native_fixture_stages_production_entry_and_rebinds_only_disposable_namespaces(tmp_path, archive):
    from tests.historical_generation_native_acceptance import _installed_entry
    root = tmp_path.resolve()
    (root / 'work').mkdir()
    (root / 'operator').mkdir()
    (root / 'work/adjacent-unselected.log').write_bytes(b'adjacent original bytes\n')
    if archive:
        (root / 'cloud-fixture.json').write_bytes(b'{"corrupt":false,"objects":{},"metadata":{}}\n')
    entry_path = root / 'action-entry'
    from blueprint_pipeline import control_plane_lane_historical_dispatch as dispatch
    original = dispatch._ACTION_EXECUTABLE
    try:
        _installed_entry(root, entry_path)
        installed = root / 'operator'
        boot = (installed / 'historical-generation-entry.py').read_text()
        compile(boot, 'installed-disposable-entry', 'exec')
        assert 'from blueprint_pipeline.control_plane_lane_historical_action import run_historical_action' in boot
        assert 'sys.flags.isolated == 1 and sys.flags.no_site == 1' in boot
        assert 'fixture_controller_startup_not_complete' in boot
        assert 'adjacent denial must be actual EROFS' in boot
        assert 'exec /usr/bin/python3 -I -S' in entry_path.read_text()
        if archive:
            assert 'installed_fixture_archive_transport' in boot
            assert 'import boto3' in boot
            assert 'sys.path.insert' not in boot
        assert dispatch._ACTION_EXECUTABLE == str(entry_path)
        assert (installed / 'historical-python/blueprint_pipeline/control_plane_lane_historical_action.py').read_bytes() == (
            Path(__file__).parents[1] / 'src/blueprint_pipeline/control_plane_lane_historical_action.py').read_bytes()
    finally:
        dispatch._ACTION_EXECUTABLE = original
