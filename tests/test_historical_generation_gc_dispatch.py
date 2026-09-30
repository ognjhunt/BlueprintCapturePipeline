"""ADP-009D/day28: GC launches one ID-only unit and never credits launch as removal."""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_gc.py
# Covers: src/blueprint_pipeline/control_plane_storage_gc.py
# Covers: src/blueprint_pipeline/control_plane_lane_historical_dispatch.py

import json
from types import SimpleNamespace

import pytest

from tests.test_historical_generation_authority import (
    decision, historical_installation, packet)  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401

# ruff: noqa: F811


def enable(installed):
    config = installed[0]
    value = json.loads(config.read_bytes())
    value['historical_generation_actions_enabled'] = True
    config.write_text(json.dumps(value))


def tick(installed, *, apply=True):
    from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, run_storage_gc
    return run_storage_gc(content_store_roots=(), derived_roots=(), queue_roots=(),
        pins_root=installed[0].parent / 'pins', apply=apply, ack=RUN_ACK if apply else '',
        _experiment_config_path=installed[0], now=lambda: 1030)['historical_generations']


@pytest.mark.parametrize('mode', ['flag_off', 'dry_run', 'owner_review'])
def test_real_gc_phase_never_launches_without_current_action_opt_in(historical_installation, monkeypatch, mode):
    from blueprint_pipeline import control_plane_lane_historical_gc as gc
    if mode != 'flag_off':
        enable(historical_installation)
    approved = decision(historical_installation, packet(historical_installation),
                        action='owner_review' if mode == 'owner_review' else 'delete')
    monkeypatch.setattr(gc, 'dispatch_historical_action', lambda **kw: pytest.fail('unauthorized launch'))
    before = (historical_installation[1] / 'one.log').read_bytes()
    result = tick(historical_installation, apply=mode != 'dry_run')
    assert result['units_started'] == result['removed_bytes'] == result['mutations'] == 0
    assert (historical_installation[1] / 'one.log').read_bytes() == before
    assert not (historical_installation[3].parent / 'historical-generation-journals' / approved['action_id']).exists()


def test_real_gc_phase_dispatches_only_one_immutable_id_and_reports_pending(historical_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_gc as gc
    enable(historical_installation)
    selected = packet(historical_installation)
    decisions = [decision(historical_installation, selected) for _ in range(2)]
    launched = []

    def launch(**kw):
        launched.append(kw)
        return dict(status='submitted', action_id=kw['action_id'], action_unit_started=True,
                    execution_authorized=False, removed_bytes=0, mutations=0)

    monkeypatch.setattr(gc, 'dispatch_historical_action', launch)
    result = tick(historical_installation)
    assert len(launched) == result['units_started'] == 1
    assert launched[0]['action_id'] == min(row['action_id'] for row in decisions)
    assert set(launched[0]) == {'installed_config_path', 'action_id', 'now', 'monotonic'}
    assert result['removed_bytes'] == result['mutations'] == 0
    assert result['outcomes'][0]['status'] == 'submitted'
    assert (historical_installation[1] / 'one.log').is_file()


def test_dispatch_is_default_off_before_generation_hash_or_process_start(historical_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_dispatch as dispatch
    approved = decision(historical_installation, packet(historical_installation))
    monkeypatch.setattr(dispatch.generation, 'inventory_historical_generation',
                        lambda *a, **kw: pytest.fail('disabled inventory'))
    monkeypatch.setattr(dispatch, '_start_unit', lambda *a, **kw: pytest.fail('disabled process'))
    with pytest.raises(ValueError, match='disabled'):
        dispatch.dispatch_historical_action(installed_config_path=historical_installation[0],
            action_id=approved['action_id'], now=1030, monotonic=lambda: 0)


def test_dispatch_uses_fixed_systemd_command_no_shell_and_exact_selected_mounts(historical_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_dispatch as dispatch
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    enable(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    executable = owners.INSTALLED_PACKAGE_ROOT / 'bin/blueprint-historical-generation-action'
    executable.parent.mkdir()
    executable.write_bytes(b'#!/bin/sh\nexit 1\n')
    executable.chmod(0o755)
    monkeypatch.setattr(dispatch, '_ACTION_EXECUTABLE', str(executable))
    calls = []

    def process(argv, **kw):
        calls.append((argv, kw))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dispatch.subprocess, 'run', process)
    result = dispatch.dispatch_historical_action(installed_config_path=historical_installation[0],
        action_id=approved['action_id'], now=1030, monotonic=lambda: 0)
    assert len(calls) == 1
    argv, options = calls[0]
    assert argv[:4] == ['/usr/bin/systemd-run', '--system', '--no-block', '--collect']
    assert argv[-2:] == [str(executable), approved['action_id']]
    assert '--property=ReadWritePaths=' + str(historical_installation[1]) + ' ' + str(
        historical_installation[3].parent / 'historical-generation-journals') in argv
    assert not options.get('shell') and options['cwd'] == '/'
    assert options['timeout'] <= 5 and 'PYTHONPATH' not in options['env']
    assert options['env']['DBUS_SYSTEM_BUS_ADDRESS'] == 'unix:path=/run/dbus/system_bus_socket'
    assert result['status'] == 'submitted' and result['action_unit_started'] is True
    assert result['execution_authorized'] is False
    assert result['removed_bytes'] == result['mutations'] == 0


@pytest.mark.parametrize('unsafe', ['writable', 'symlink', 'hardlink', 'not_executable', 'missing'])
def test_unprotected_installed_entry_never_starts_root_unit(historical_installation, monkeypatch, unsafe):
    from blueprint_pipeline import control_plane_lane_historical_dispatch as dispatch
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    import os
    enable(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    executable = owners.INSTALLED_PACKAGE_ROOT / 'bin/blueprint-historical-generation-action'
    executable.parent.mkdir()
    if unsafe != 'missing':
        executable.write_bytes(b'#!/bin/sh\nexit 1\n')
        executable.chmod(0o755)
    if unsafe == 'writable':
        executable.chmod(0o777)
    elif unsafe == 'symlink':
        original = executable.with_name('original')
        executable.rename(original)
        executable.symlink_to(original)
    elif unsafe == 'hardlink':
        os.link(executable, executable.with_name('alias'))
    elif unsafe == 'not_executable':
        executable.chmod(0o644)
    monkeypatch.setattr(dispatch, '_ACTION_EXECUTABLE', str(executable))
    monkeypatch.setattr(dispatch, '_start_unit', lambda *a, **kw: pytest.fail('unsafe root entry'))
    with pytest.raises(ValueError):
        dispatch.dispatch_historical_action(installed_config_path=historical_installation[0],
            action_id=approved['action_id'], now=1030, monotonic=lambda: 0)
    assert (historical_installation[1] / 'one.log').is_file()


@pytest.mark.parametrize('changed', ['policy', 'config', 'decision', 'payload'])
def test_drift_during_preflight_refuses_before_launch(historical_installation, monkeypatch, changed):
    from blueprint_pipeline import control_plane_lane_historical_dispatch as dispatch
    enable(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    actual = dispatch.generation.inventory_historical_generation
    config, target, policy, store, _ = historical_installation
    path = {'policy': policy, 'config': config, 'decision': store / (approved['action_id'] + '.json'),
            'payload': target / 'one.log'}[changed]

    def changed_inventory(*args, **kw):
        value = actual(*args, **kw)
        path.write_bytes(path.read_bytes() + b'changed')
        return value

    monkeypatch.setattr(dispatch.generation, 'inventory_historical_generation', changed_inventory)
    monkeypatch.setattr(dispatch, '_start_unit', lambda *a, **kw: pytest.fail('stale dispatch'))
    with pytest.raises(ValueError):
        dispatch.dispatch_historical_action(installed_config_path=config,
            action_id=approved['action_id'], now=1030, monotonic=lambda: 0)


def begin_journal(installed, action_id):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline import control_plane_lane_historical_dispatch as dispatch
    from blueprint_pipeline.control_plane_lane_historical_journal import HistoricalActionJournal
    operation = authority._Operation(1030, lambda: 0)
    with authority._session(installed[0], operation) as (files, config, store):
        selected = dispatch._selection(files, config, store, installed[0], action_id, 1030)
        return HistoricalActionJournal(files, config, selected, operation).head


def test_journaled_retry_derives_original_target_without_requiring_pristine_bytes(historical_installation):
    from blueprint_pipeline.control_plane_lane_historical_dispatch import select_historical_dispatch
    enable(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    intent = begin_journal(historical_installation, approved['action_id'])
    # This preflight supplies no recovery authority. The actual worker must
    # reject an unjournaled transition; the dispatcher only restores its scope.
    historical_installation[1].chmod(0o700)
    selected = select_historical_dispatch(installed_config_path=historical_installation[0],
        action_id=approved['action_id'], now=1031, monotonic=lambda: 1)
    assert selected['journal_head_digest'] == intent['event_digest']
    assert selected['read_write_paths'] == [str(historical_installation[1]),
        str(historical_installation[3].parent / 'historical-generation-journals')]
    assert selected['execution_authorized'] is selected['action_unit_started'] is False
    assert (historical_installation[1] / 'one.log').read_bytes() == b'original owner diagnostics\n'


def test_retry_cannot_freshen_original_four_hour_clock(historical_installation):
    from blueprint_pipeline.control_plane_lane_historical_dispatch import select_historical_dispatch
    enable(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    begin_journal(historical_installation, approved['action_id'])
    with pytest.raises(ValueError, match='deadline'):
        select_historical_dispatch(installed_config_path=historical_installation[0],
            action_id=approved['action_id'], now=1031, monotonic=lambda: 14401)


@pytest.mark.parametrize('failure', ['timeout', 'nonzero'])
def test_ambiguous_or_failed_manager_submission_never_reports_completion(historical_installation, monkeypatch, failure):
    from blueprint_pipeline import control_plane_lane_historical_dispatch as dispatch
    from blueprint_pipeline import control_plane_lane_historical_gc as gc
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    enable(historical_installation)
    selected = packet(historical_installation)
    decisions = [decision(historical_installation, selected) for _ in range(2)]
    executable = owners.INSTALLED_PACKAGE_ROOT / 'bin/blueprint-historical-generation-action'
    executable.parent.mkdir()
    executable.write_bytes(b'#!/bin/sh\nexit 1\n')
    executable.chmod(0o755)
    monkeypatch.setattr(dispatch, '_ACTION_EXECUTABLE', str(executable))
    calls = []

    def process(argv, **kw):
        calls.append(argv)
        if failure == 'timeout':
            raise dispatch.subprocess.TimeoutExpired(argv, 5)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(dispatch.subprocess, 'run', process)
    result = gc.gc_historical_actions(installed_config_path=historical_installation[0], apply=True,
                                      now=lambda: 1030, monotonic=lambda: 0)
    assert len(calls) == 1
    assert calls[0][-1] == min(row['action_id'] for row in decisions)
    assert result['units_started'] == result['mutations'] == result['removed_bytes'] == 0
    assert result['outcomes'][0]['status'] == 'kept'
    assert (historical_installation[1] / 'one.log').is_file()
