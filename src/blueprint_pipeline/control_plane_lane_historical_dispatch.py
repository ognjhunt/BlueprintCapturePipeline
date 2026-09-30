"""Select one original historical generation for a separate hardened action.

This root preflight derives write mounts only from protected owner decisions.
It neither starts a unit nor closes readers or mutates historical payloads.
"""
from __future__ import annotations

import os
import re
import stat
import time
from pathlib import Path

from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_legacy_owner as legacy
from . import control_plane_lane_owner_consents as owners
from .decision_evidence_contracts import canonical_digest

_DECISION_FIELDS = frozenset({'schema_version', 'action_id', 'packet_id', 'packet_digest',
    'generation_digest', 'manifest', 'principal', 'owner', 'action', 'finished_run_ref',
    'no_future_writers', 'no_future_readers', 'issued_at_epoch', 'expires_at_epoch',
    'policy', 'decommission_approved', 'execution_authorized', 'decision_digest'})
_ACTION_EXECUTABLE = '/opt/blueprint/operator-door/bin/blueprint-historical-generation-action'


def _unit_properties(target, private_store):
    """No parent write mount, private PID view, arbitrary command or privilege gain."""
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
        SystemCallFilter=['@system-service', '~ptrace process_vm_readv process_vm_writev'],
        ReadWritePaths=[str(target), private_store], TasksMax=64, LimitNOFILE=512,
        MemoryMax=512 * 1024**2, TimeoutStartSec=generation.MAX_SECONDS, Restart='no',
        WorkingDirectory='/')


def _require(value, code):
    generation._require(value, 'dispatch_' + code)


def _unit_property_assignments(target, private_store):
    """Finite first scope refuses unit specifiers/escaping instead of expanding them."""
    for path in (target, private_store):
        _require(isinstance(path, (str, Path)) and re.fullmatch(r'/[A-Za-z0-9_./-]+', str(path))
                 and len(os.fsencode(path)) <= 4096, 'unit_path_unsupported')
        legacy._absolute(Path(path))
    _require(Path(target) != Path(private_store)
             and not Path(target).is_relative_to(private_store)
             and not Path(private_store).is_relative_to(target), 'unit_path_unsupported')
    assignments = []
    for key, value in _unit_properties(target, str(private_store)).items():
        if key == 'SystemCallFilter':
            assignments.extend(key + '=' + part for part in value)
        else:
            if isinstance(value, bool):
                value = 'yes' if value else 'no'
            elif isinstance(value, list):
                value = ' '.join(value)
            assignments.append(key + '=' + str(value))
    return tuple(assignments)


def _selection(files, config, store, config_path, action_id, moment):
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
    manifest, raw = store.read(decision['packet_id'], manifest=True)
    _require(authority._selector(raw) == decision['manifest']
        and manifest.get('schema_version') == 'control_plane_historical_generation.v1'
        and manifest.get('generation_digest') == decision['generation_digest']
        == canonical_digest(manifest, digest_field='generation_digest')
        and manifest.get('target_path') == packet['selected_path'], 'manifest_changed')
    return packet, decision, manifest, dict(packet=authority._selector(packet_raw),
                                           decision=authority._selector(decision_raw))


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
        private_store = str(store.root)
    observed = generation.inventory_historical_generation(target, allowed_roots=roots,
        max_seconds=operation.remaining(), monotonic=monotonic)
    _require(observed == manifest, 'generation_changed')
    with authority._session(installed_config_path, operation) as (files, config, store):
        current = _selection(files, config, store, installed_config_path, action_id, operation.moment())
        _require(current == (packet, decision, manifest, records), 'authority_changed')
        parent, name = files.parent(target, protected=True)
        _require(legacy._directory_identity(os.fstat(parent)) == manifest['root_identity']
            and legacy._directory_identity(os.stat(name, dir_fd=parent, follow_symlinks=False))
            == manifest['target_identity'] and str(store.root) == private_store, 'target_changed')
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
