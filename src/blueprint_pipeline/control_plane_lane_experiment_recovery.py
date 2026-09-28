"""Finite exact-operation reconciliation; no absence-based removal authority."""
from __future__ import annotations

import fcntl
import os
import stat
from pathlib import Path

from . import control_plane_lane_experiment_retirement as issuance
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_publication import _publish
from .control_plane_lane_owner_target_versions import _require
from .decision_evidence_contracts import canonical_digest

_FIELDS = frozenset({'schema_version', 'event_id', 'operation_id', 'event_kind', 'intent_id',
                     'generation', 'sequence', 'previous_event', 'issued_at_epoch', 'body', 'event_digest'})
_STAT = ('st_mode', 'st_uid', 'st_gid', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')


def _read_event(files, directory, action, index, previous):
    name = f'e-{index:05d}.json'
    files.location(directory)
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    raw, _ = files.read(Path(files._operation_path) / name, cap=32768, protected=True, mode=0o600)
    value = retained._document(raw, 32768, _work_budget=files.budget)
    _require(set(value) == _FIELDS and value['schema_version'] == 'control_plane_lane_experiment_event.v1'
             and value['event_digest'] == canonical_digest(value, digest_field='event_digest')
             and owners._matches(value['event_id'], owners._CONSENT_ID)
             and all(value[key] == action[key] for key in ('intent_id', 'generation'))
             and value['operation_id'] == action['action_id'] and type(value['sequence']) is int
             and value['sequence'] == index and value['previous_event'] == previous,
             'experiment_operation_invalid')
    return value, issuance._selector(raw, files.budget)


def _once(files, parent, name, payload, *, kind):
    """Already durable bytes are selected, never reissued or overwritten."""
    files.location(parent)
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return _publish(files, parent, name, payload, kind=kind)
    raw, _ = files.read(Path(files._store_path) / name, cap=32768, protected=True, mode=0o600)
    _require(raw == payload, 'experiment_operation_invalid')
    return issuance._selector(raw, files.budget)


def begin(files, config, action, expected, entry, current, refreshed, public, store,
          target, rows, reference, issued, gid):
    from . import control_plane_lane_experiment_actions as code
    action_id = action['action_id']
    files._store_path = config.experiment_record_store
    files._operation_path = str(Path(config.experiment_record_store) / 'operations' / action_id)
    files.location(store)
    try:
        os.stat('operations', dir_fd=store, follow_symlinks=False)
    except FileNotFoundError:
        operations = code._directory(files, store, 'operations', create=True)
    else:
        operations = code._directory(files, store, 'operations')
    files.location(operations)
    try:
        os.stat(action_id, dir_fd=operations, follow_symlinks=False)
    except FileNotFoundError:
        _require(entry['state'] == 'active', 'experiment_operation_missing')
        operation = code._directory(files, operations, action_id, create=True)
    else:
        operation = code._directory(files, operations, action_id)
    files.proof(operation)
    fcntl.flock(operation, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected = _read_event(files, operation, action, 0, None)
    if selected is None:
        _require(entry['state'] == 'active', 'experiment_operation_missing')
        previous = code._event(files, operation, action, 'started', dict(action=expected,
            birth=entry['birth'], initial_authority=current[0]['record'], manifest=action['manifest'],
            process_identity={'pid': os.getpid()}, controller_origin_epoch=issued,
            deadline_epoch=min(issued + 4 * 3600, action['expires_at_epoch']), reference_authority=reference),
            0, None, issued)
    else:
        started, previous = selected
        body = started['body']
        _require(started['event_kind'] == 'started' and set(body) == {'action', 'birth', 'initial_authority',
                 'manifest', 'process_identity', 'controller_origin_epoch', 'deadline_epoch', 'reference_authority'}
                 and body['action'] == expected and body['birth'] == entry['birth']
                 and body['manifest'] == action['manifest'], 'experiment_operation_invalid')
        _require(body['reference_authority'] == reference, 'experiment_reference_authority_changed')
        _require(type(body['deadline_epoch']) in (int, float) and issued < body['deadline_epoch']
                 <= action['expires_at_epoch'], 'experiment_action_expired')
        if entry['state'] == 'active':
            _require(body['initial_authority'] == current[0]['record'], 'experiment_action_current_changed')
    if entry['state'] == 'active':
        prepared, old_head = code._version(files, public, refreshed, entry | {'state': 'retiring'},
                                          gid, action['policy'], issued)
        _once(files, store, action_id + '.retiring-head.json', prepared, kind='private')
        code._install_head(files, public, prepared, gid, old_head)
        retiring = code._current(files, public, gid)
    else:
        _require(entry['state'] == 'retiring', 'experiment_action_current_changed')
        prepared, _ = files.read(Path(config.experiment_record_store) / (action_id + '.retiring-head.json'),
                                cap=32768, protected=True, mode=0o600)
        _require(issuance._selector(prepared, files.budget) == issuance._selector(
            code._encoded(refreshed[0] | {}, 'head_digest', 4096), files.budget),
            'experiment_action_current_changed')
        retiring = refreshed
    return (operation, retiring, *progress(files, operation, action, expected, target, rows, previous))


def progress(files, operation, action, expected, target, rows, previous):
    logical, allocated, changed, count = 0, 0, {}, 0
    for index, row in enumerate(rows, 1):
        selected = _read_event(files, operation, action, index, previous)
        if selected is None:
            break
        event, selected = selected
        body = event['body']
        _require(event['event_kind'] == 'member_removed' and set(body) == {'preservation', 'action', 'manifest',
                 'index', 'path', 'original_identity', 'logical_bytes', 'eligible_allocated_bytes', 'parent_after'}
                 and body['preservation'] is None and body['action'] == expected and body['manifest'] == action['manifest']
                 and body['index'] == index - 1 and body['path'] == row[0], 'experiment_operation_invalid')
        identity = row[2].split(':')
        tokens = row[3].split(':')
        _require(body['original_identity'] == dict(dev=int(identity[0]), ino=int(identity[1]), type=row[1])
                 and type(body['logical_bytes']) is int and body['logical_bytes'] == (int(tokens[4]) if row[1] == 'file' else 0)
                 and type(body['eligible_allocated_bytes']) is int and 0 <= body['eligible_allocated_bytes'] <= 128 * 1024**3,
                 'experiment_operation_invalid')
        member = target / row[0]
        try:
            os.stat(member, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _require(False, 'experiment_removed_name_reappeared')
        parent_after = body['parent_after']
        _require(isinstance(parent_after, dict) and set(parent_after) == {'path', 'identity', 'stat_token'}
                 and parent_after['path'] == str(Path(row[0]).parent), 'experiment_operation_invalid')
        parts = parent_after['stat_token'].split(':')
        _require(len(parts) == 7 and all(part.isdecimal() and len(part) <= 20 for part in parts),
                 'experiment_operation_invalid')
        changed[parent_after['path']] = (parent_after['identity'], tuple(map(int, parts)))
        logical += body['logical_bytes']
        allocated += body['eligible_allocated_bytes']
        count, previous = index, selected
    # Only last durable exact own transitions may explain surviving directories.
    for path, (identity, metadata) in changed.items():
        destination = target / path
        try:
            named = os.stat(destination, follow_symlinks=False)
        except FileNotFoundError:
            _require(any(row[0] == path for row in rows[:count]), 'experiment_operation_invalid')
            continue
        _require(identity == dict(dev=named.st_dev, ino=named.st_ino, type='directory')
                 and stat.S_ISDIR(named.st_mode) and tuple(getattr(named, key) for key in _STAT) == metadata,
                 'experiment_directory_transition_changed')
    completed = _read_event(files, operation, action, len(rows) + 1, previous) if count == len(rows) else None
    if completed is not None:
        event, selected = completed
        body = event['body']
        _require(event['event_kind'] == 'retired' and set(body) == {'preservation', 'manifest', 'removed_event_count',
                 'removed_logical_bytes', 'eligible_allocated_bytes', 'remaining_metadata', 'partial'}
                 and body['preservation'] is None and body['manifest'] == action['manifest']
                 and type(body['partial']) is bool and body['partial'] is False
                 and body['removed_event_count'] == count and body['removed_logical_bytes'] == logical
                 and body['eligible_allocated_bytes'] == allocated, 'experiment_operation_invalid')
        completed = selected
    return previous, logical, allocated, {p: v[1] for p, v in changed.items()}, count, completed


def retired(files, config, action, expected, target, rows, entry, marker):
    from . import control_plane_lane_experiment_actions as code
    files._operation_path = str(Path(config.experiment_record_store) / 'operations' / action['action_id'])
    store = issuance._store(files, config.experiment_record_store)
    operations = code._directory(files, store, 'operations')
    operation = code._directory(files, operations, action['action_id'])
    files.proof(operation)
    fcntl.flock(operation, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = _read_event(files, operation, action, 0, None)
    _require(started is not None, 'experiment_operation_invalid')
    event, previous = started
    body = event['body']
    _require(event['event_kind'] == 'started' and set(body) == {'action', 'birth', 'initial_authority', 'manifest',
             'process_identity', 'controller_origin_epoch', 'deadline_epoch', 'reference_authority'}
             and body['action'] == expected and body['birth'] == entry['birth'] and body['manifest'] == action['manifest'],
             'experiment_operation_invalid')
    _, logical, allocated, _, count, receipt = progress(files, operation, action, expected, target, rows, previous)
    _require(count == len(rows) and receipt is not None, 'experiment_operation_invalid')
    raw, _ = files.read(Path(files._operation_path) / f'e-{len(rows) + 1:05d}.json', cap=32768, protected=True, mode=0o600)
    final = retained._document(raw, 32768, _work_budget=files.budget)
    _require(final['body']['remaining_metadata'] == [entry['lease'], marker], 'experiment_operation_invalid')
    return receipt, logical, allocated
