"""Select one original historical generation for a separate hardened action.

This root preflight derives write mounts only from protected owner decisions.
It neither starts a unit nor closes readers or mutates historical payloads.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
import time
from pathlib import Path

from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_journal import HistoricalJournalObservation, journal_root
from . import control_plane_lane_legacy_owner as legacy
from . import control_plane_lane_owner_consents as owners
from .decision_evidence_contracts import canonical_digest
from .control_plane_disk_ledger import DEFAULT_RESERVATION_ROOT

_DECISION_FIELDS = frozenset({'schema_version', 'action_id', 'packet_id', 'packet_digest',
    'generation_digest', 'manifest', 'principal', 'owner', 'action', 'finished_run_ref',
    'no_future_writers', 'no_future_readers', 'issued_at_epoch', 'expires_at_epoch',
    'policy', 'decommission_approved', 'execution_authorized', 'decision_digest'})
_ACTION_EXECUTABLE = '/opt/blueprint/operator-door/bin/blueprint-historical-generation-action'
_SYSTEM_ENV = {'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LANG': 'C',
               'DBUS_SYSTEM_BUS_ADDRESS': 'unix:path=/run/dbus/system_bus_socket'}


def _unit_properties(target, private_store, *, restore=False):
    """No parent write mount, private PID view, arbitrary command or privilege gain."""
    _require(type(restore) is bool, 'unit_path_unsupported')
    return dict(Type='oneshot', User='root', Group='root', UMask='0077',
        NoNewPrivileges=True, PrivateTmp=True, PrivateDevices=True, PrivateUsers=False,
        ProtectSystem='strict', ProtectHome=True, ProtectHostname=True, ProtectClock=True,
        ProtectKernelTunables=True, ProtectKernelModules=True, ProtectKernelLogs=True,
        ProtectControlGroups=True, RestrictSUIDSGID=True, LockPersonality=True,
        ProtectProc='default', ProcSubset='all',
        CapabilityBoundingSet=['CAP_DAC_OVERRIDE', 'CAP_DAC_READ_SEARCH', 'CAP_CHOWN',
                              'CAP_FOWNER', 'CAP_SYS_PTRACE'], AmbientCapabilities=[],
        RestrictAddressFamilies=['AF_UNIX', 'AF_INET', 'AF_INET6'],
        SystemCallArchitectures='native',
        SystemCallFilter=['@system-service seccomp landlock_create_ruleset landlock_add_rule landlock_restrict_self',
                          '~ptrace process_vm_readv process_vm_writev'],
        ReadWritePaths=[str(target), private_store] + ([str(DEFAULT_RESERVATION_ROOT)] if restore else []),
        TasksMax=64, LimitNOFILE=512,
        MemoryMax=512 * 1024**2, TimeoutStartSec=generation.MAX_SECONDS, Restart='no',
        WorkingDirectory='/')


def _require(value, code):
    generation._require(value, 'dispatch_' + code)


def _unit_property_assignments(target, private_store, *, restore=False):
    """Finite first scope refuses unit specifiers/escaping instead of expanding them."""
    _require(type(restore) is bool, 'unit_path_unsupported')
    paths = (target, private_store) + ((DEFAULT_RESERVATION_ROOT,) if restore else ())
    for path in paths:
        _require(isinstance(path, (str, Path)) and re.fullmatch(r'/[A-Za-z0-9_./-]+', str(path))
                 and len(os.fsencode(path)) <= 4096, 'unit_path_unsupported')
        legacy._absolute(Path(path))
    _require(all(Path(left) != Path(right) and not Path(left).is_relative_to(right)
                 and not Path(right).is_relative_to(left)
                 for index, left in enumerate(paths) for right in paths[index + 1:]), 'unit_path_unsupported')
    assignments = []
    for key, value in _unit_properties(target, str(private_store), restore=restore).items():
        if key == 'SystemCallFilter':
            assignments.extend(key + '=' + part for part in value)
        else:
            if isinstance(value, bool):
                value = 'yes' if value else 'no'
            elif isinstance(value, list):
                value = ' '.join(value)
            assignments.append(key + '=' + str(value))
    return tuple(assignments)


def _original_selection(store, action_id):
    """Authenticate original protected facts; this grants no current authority."""
    decision, decision_raw = store.read(action_id)
    _require(set(decision) == _DECISION_FIELDS
        and decision['schema_version'] == 'control_plane_historical_decommission.v1'
        and decision['action_id'] == action_id and decision['decommission_approved'] is True
        and decision['execution_authorized'] is False
        and decision['decision_digest'] == canonical_digest(decision, digest_field='decision_digest')
        and decision['no_future_writers'] is True and decision['no_future_readers'] is True,
        'decision_invalid')
    _require(decision['action'] != 'owner_review', 'owner_review')
    _require(decision['action'] in ('delete', 'offload'), 'decision_invalid')
    packet, packet_raw = store.read(decision['packet_id'])
    _require(packet.get('schema_version') == 'control_plane_historical_generation_packet.v1'
        and packet.get('packet_id') == decision['packet_id']
        and packet.get('packet_digest') == decision['packet_digest']
        == canonical_digest(packet, digest_field='packet_digest')
        and packet.get('execution_authorized') is False
        and all(packet[key] == decision[key] for key in ('owner', 'principal', 'generation_digest', 'manifest')),
        'packet_invalid')
    manifest, raw = store.read(decision['packet_id'], manifest=True)
    _require(authority._selector(raw) == decision['manifest']
        and manifest.get('schema_version') == 'control_plane_historical_generation.v1'
        and manifest.get('generation_digest') == decision['generation_digest']
        == canonical_digest(manifest, digest_field='generation_digest')
        and manifest.get('target_path') == packet['selected_path'], 'manifest_changed')
    return packet, decision, manifest, dict(packet=authority._selector(packet_raw),
                                           decision=authority._selector(decision_raw))


def _selection(files, config, store, config_path, action_id, moment):
    value, _ = store.read(action_id)
    if value.get('schema_version') == 'control_plane_historical_restore_decision.v1':
        from .control_plane_lane_historical_restore_authority import select_restore
        return select_restore(files, config, store, config_path, action_id, moment)
    selected = _original_selection(store, action_id)
    packet, decision, _, _ = selected
    _require(packet['observed_at_epoch'] <= decision['issued_at_epoch'] <= moment
        < decision['expires_at_epoch'] <= packet['expires_at_epoch'], 'expired')
    old, configured, policy_selector, policy = authority._authority(
        files, config, config_path, packet['old_consent'], moment)
    _require(configured == packet['installed_config'] and policy_selector == packet['policy']
        == decision['policy'] and old['principal'] == decision['principal']
        and old['consent_digest'] == packet['old_consent']['consent_digest'], 'authority_changed')
    approved_policy = owners._policy(policy, decision['principal'], files.budget)
    owners._authorize(dict(owner=decision['owner'], action=decision['action'], ttl_seconds=900),
                      approved_policy, decision['expires_at_epoch'], moment)
    return selected


def select_historical_action(*, installed_config_path, action_id, now, monotonic=time.monotonic):
    """Authenticate a fixed ID and hash the original generation before dispatch."""
    operation = authority._Operation(now, monotonic)
    with authority._session(installed_config_path, operation) as (files, config, store):
        packet, decision, manifest, records = _selection(
            files, config, store, installed_config_path, action_id, operation.moment())
        roots = owners._roots(config, files.budget)
        target = Path(packet['selected_path'])
        parent, name = files.parent(target, protected=True)
        parent_info = os.fstat(parent)
        _require(stat.S_ISDIR(parent_info.st_mode) and parent_info.st_uid == 0
            and not parent_info.st_mode & 0o022
            and legacy._directory_identity(parent_info) == manifest['root_identity'], 'parent_unsafe')
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        _require(legacy._directory_identity(named) == manifest['target_identity'], 'target_changed')
        private_store = str(journal_root(config))
        private, _ = files.parent(Path(private_store) / '.journal-probe', protected=True)
        owners._protected(os.fstat(private), directory=True, mode=0o700)
    observed = generation.inventory_historical_generation(target, allowed_roots=roots,
        max_seconds=operation.remaining(), monotonic=monotonic)
    _require(observed == manifest, 'generation_changed')
    with authority._session(installed_config_path, operation) as (files, config, store):
        current = _selection(files, config, store, installed_config_path, action_id, operation.moment())
        _require(current == (packet, decision, manifest, records), 'authority_changed')
        parent, name = files.parent(target, protected=True)
        _require(legacy._directory_identity(os.fstat(parent)) == manifest['root_identity']
            and legacy._directory_identity(os.stat(name, dir_fd=parent, follow_symlinks=False))
            == manifest['target_identity'] and str(journal_root(config)) == private_store, 'target_changed')
        private, _ = files.parent(Path(private_store) / '.journal-probe', protected=True)
        owners._protected(os.fstat(private), directory=True, mode=0o700)
        generation.verify_historical_member_versions(manifest, tick=files.budget.tick)
        assignments = _unit_property_assignments(target, private_store)
        value = dict(schema_version='control_plane_historical_action_selection.v1', action_id=action_id,
            action=decision['action'], owner=decision['owner'], records=records,
            packet_id=packet['packet_id'], generation_digest=manifest['generation_digest'],
            manifest=decision['manifest'], target_path=str(target),
            read_write_paths=[str(target), private_store],
            unit_name='blueprint-historical-generation-' + action_id + '.service',
            exec_start=[_ACTION_EXECUTABLE, action_id],
            service_properties=_unit_properties(target, private_store),
            unit_property_assignments=list(assignments),
            observed_at_epoch=operation.moment(), expires_at_epoch=decision['expires_at_epoch'],
            execution_authorized=False, action_unit_started=False)
        value['selection_digest'] = canonical_digest(value, digest_field='selection_digest')
        return value


def _start_unit(selected, *, timeout):
    """Start only the fixed ID-only action; launch is never a removal receipt."""
    command = ['/usr/bin/systemd-run', '--system', '--no-block', '--collect',
        '--unit=' + selected['unit_name'],
        *('--property=' + value for value in selected['unit_property_assignments']),
        '--', *selected['exec_start']]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, cwd='/', env=dict(_SYSTEM_ENV), timeout=timeout,
            check=False)
    except (OSError, subprocess.TimeoutExpired):
        # A timeout can follow manager admission. Never relaunch with a fresh
        # ID/deadline, report completion, or claim an unobserved unit is absent.
        raise generation.HistoricalGenerationError('historical_generation_dispatch_start_unknown') from None
    _require(result.returncode == 0, 'start_refused')


def _prior_journal(files, config, selected, operation):
    """Read original scope/clock only; never create or append dispatcher records."""
    action_id = selected[1]['action_id']
    parent, name = files.parent(journal_root(config) / action_id, protected=True)
    owners._protected(os.fstat(parent), directory=True, mode=0o700)
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None
    journal = HistoricalJournalObservation(files, config, selected, operation)
    head, seed = journal.head, journal._read(0)['body']
    from .control_plane_lane_experiment_work import _controller_boot_id
    _require(seed['boot_id'] == _controller_boot_id(files)
        and seed['started_monotonic'] <= operation.last
        < seed['started_monotonic'] + generation.MAX_SECONDS, 'deadline')
    operation.resume_original(seed['started_monotonic'])
    return head['event_digest']


def _dispatch_scope(files, config, selected):
    packet, decision, manifest, records = selected
    target = Path(manifest['target_path'])
    parent, name = files.parent(target, protected=True)
    parent_info = os.fstat(parent)
    _require(stat.S_ISDIR(parent_info.st_mode) and parent_info.st_uid == 0
        and not parent_info.st_mode & 0o022
        and legacy._directory_identity(parent_info) == manifest['root_identity'], 'parent_unsafe')
    named = os.stat(name, dir_fd=parent, follow_symlinks=False)
    _require(stat.S_ISDIR(named.st_mode)
        and (named.st_dev, named.st_ino) == (manifest['target_identity']['dev'],
                                           manifest['target_identity']['ino']), 'target_changed')
    private_store = str(journal_root(config))
    private, _ = files.parent(Path(private_store) / '.journal-probe', protected=True)
    owners._protected(os.fstat(private), directory=True, mode=0o700)
    restore = decision['action'] == 'restore'
    properties = _unit_properties(target, private_store, restore=restore)
    return dict(schema_version='control_plane_historical_action_selection.v1',
        action_id=decision['action_id'], action=decision['action'], owner=decision['owner'], records=records,
        packet_id=packet['packet_id'], generation_digest=manifest['generation_digest'], manifest=decision['manifest'],
        target_path=str(target), read_write_paths=properties['ReadWritePaths'],
        unit_name='blueprint-historical-generation-' + decision['action_id'] + '.service',
        exec_start=[_ACTION_EXECUTABLE, decision['action_id']], service_properties=properties,
        unit_property_assignments=list(_unit_property_assignments(target, private_store, restore=restore)),
        expires_at_epoch=decision['expires_at_epoch'], execution_authorized=False, action_unit_started=False)


def select_historical_dispatch(*, installed_config_path, action_id, now, monotonic=time.monotonic):
    """Select a new action, original interrupted scope, or separately approved restore.

    Journal presence supplies the original scope and deadline, never acceptance
    of changed bytes or recovery authority. The worker replays the full chain
    and authenticates every current member before any effect.
    """
    operation = authority._Operation(now, monotonic)
    with authority._session(installed_config_path, operation) as (files, config, store):
        _require(config.historical_generation_actions_enabled is True, 'disabled')
        original = _selection(files, config, store, installed_config_path, action_id, operation.moment())
        head = _prior_journal(files, config, original, operation)
        pristine = head is None and original[1]['action'] != 'restore'
        selected = _dispatch_scope(files, config, original)
    if pristine:
        selected = select_historical_action(installed_config_path=installed_config_path,
            action_id=action_id, now=operation.moment(), monotonic=monotonic)
    with authority._session(installed_config_path, operation) as (files, config, store):
        _require(config.historical_generation_actions_enabled is True, 'disabled')
        current = _selection(files, config, store, installed_config_path, action_id, operation.moment())
        _require(current == original and _prior_journal(files, config, current, operation) == head,
                 'authority_changed')
        scope = _dispatch_scope(files, config, current)
        _require(all(selected[key] == value for key, value in scope.items()), 'authority_changed')
        if pristine:
            generation.verify_historical_member_versions(current[2], tick=files.budget.tick)
        selected.update(journal_head_digest=head, pristine_generation_checked=pristine,
                        observed_at_epoch=operation.moment())
        selected['selection_digest'] = canonical_digest(selected, digest_field='selection_digest')
        return selected


def dispatch_historical_action(*, installed_config_path, action_id, now,
                               monotonic=time.monotonic):
    """Fresh protected selection and one fixed target-only transient unit.

    The actual worker independently authenticates original journal, clock,
    native sandbox and current readers before each effect. This return records
    manager submission only; it cannot credit freed bytes or action completion.
    """
    operation = authority._Operation(now, monotonic)
    with authority._session(installed_config_path, operation) as (_, config, _):
        _require(config.historical_generation_actions_enabled is True, 'disabled')
    selected = select_historical_dispatch(installed_config_path=installed_config_path,
        action_id=action_id, now=operation.moment(), monotonic=monotonic)
    with authority._session(installed_config_path, operation) as (files, config, store):
        _require(config.historical_generation_actions_enabled is True, 'disabled')
        packet, decision, manifest, records = _selection(files, config, store,
            installed_config_path, action_id, operation.moment())
        _require(records == selected['records'] and manifest['generation_digest'] == selected['generation_digest']
            and packet['selected_path'] == selected['target_path'] and decision['action'] == selected['action'],
            'authority_changed')
        _require(_prior_journal(files, config, (packet, decision, manifest, records), operation)
            == selected['journal_head_digest'], 'journal_changed')
        scope = _dispatch_scope(files, config, (packet, decision, manifest, records))
        _require(all(selected[key] == value for key, value in scope.items()), 'authority_changed')
        if selected['pristine_generation_checked']:
            generation.verify_historical_member_versions(manifest, tick=files.budget.tick)
        _, executable = files.read(_ACTION_EXECUTABLE, cap=65536, protected=True)
        _require(stat.S_IMODE(executable.info.st_mode) == 0o755, 'executable_unsafe')
        files.verify()
        _start_unit(selected, timeout=min(5, operation.remaining()))
        operation.remaining()
    return dict(schema_version='control_plane_historical_dispatch_submission.v1',
        status='submitted', action_id=action_id, action=selected['action'],
        unit_name=selected['unit_name'], selection_digest=selected['selection_digest'],
        action_unit_started=True, execution_authorized=False, mutations=0, removed_bytes=0)
