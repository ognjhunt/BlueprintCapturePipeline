"""Durable per-intent retirement projection, never deletion or restore authority.

The connected engine retains EX and validates action authority. This publisher
keeps immutable versions beside the actual intent and only advances the current
projection from the exact selected pending bytes. It never mutates payloads.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path

from . import task_evaluation_scene_retirement_access as access
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_retirement_access import _canonical, _close_owned, _document, _identity, _opened, _require
from .task_evaluation_scene_retirement_authority import ID, SHA, TOKEN, raw_reference, selected_document
from .task_evaluation_scene_retirement_generations import _guard, _named, _new_file

MAX_BYTES = 65536
NAME = 'scene-retired.v1.json'


def _encode(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    _require(0 < len(raw) <= MAX_BYTES, 'scene_retirement_receipt_limit')
    return raw


def _ref(path, raw):
    return {'path': str(path), 'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def _version(info):
    return (*_identity(info), info.st_uid, info.st_gid, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


def _parent(path, parent, expected):
    _guard(parent, _identity(expected))
    with _opened(path, directory=True) as (_, observed):
        _require(_identity(observed) == _identity(expected)
                 and (observed.st_uid, observed.st_gid) == (expected.st_uid, expected.st_gid),
                 'scene_retirement_receipt_parent_changed')
    _guard(parent, _identity(expected))


def _read(path, allowance, *, selected=None, projection=False):
    with _opened(path) as (fd, info):
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                 and not info.st_mode & 0o022 and 0 < info.st_size <= MAX_BYTES,
                 'scene_retirement_receipt_permissions')
        if projection:
            _require(info.st_uid == access._POLICY_UID and stat.S_IMODE(info.st_mode) == 0o644,
                     'scene_retirement_receipt_permissions')
        raw, remaining = [], info.st_size
        while remaining:
            allowance.tick()
            _guard(fd, _identity(info))
            part = os.read(fd, min(65536, remaining))
            _require(part, 'scene_retirement_receipt_changed')
            raw.append(part)
            remaining -= len(part)
        allowance.tick()
        _guard(fd, _identity(info))
        _require(_version(os.fstat(fd)) == _version(info), 'scene_retirement_receipt_changed')
    data = b''.join(raw)
    reference = _ref(path, data)
    _require(selected is None or reference == selected, 'scene_retirement_receipt_changed')
    return _document(data), reference, info


def _location(policy, consent, allowance):
    allowance.tick()
    _require(access._policy() == policy, 'scene_retirement_policy_changed')
    intent_id = consent.get('intent_id')
    _require(type(intent_id) is str and ID.fullmatch(intent_id), 'scene_retirement_receipt_intent_invalid')
    context = policy.get('reference_context')
    _require(type(context) is dict and type(context.get('roots')) is dict,
             'scene_retirement_installed_context_unproven')
    root = _canonical(context['roots']['intent_root'])
    directory = root / intent_id
    intent_path = directory / 'intent.json'
    selected = raw_reference(consent['intent_raw_ref'])
    _require(selected['path'] == str(intent_path), 'scene_retirement_receipt_intent_invalid')
    value, _, intent_info = _read(intent_path, allowance, selected=selected)
    _require(value.get('intent_id') == intent_id and value.get('schema_version') == 'task_evaluation_scene_intent.v1'
             and value.get('intent_digest') == canonical_digest(value, digest_field='intent_digest'),
             'scene_retirement_receipt_intent_invalid')
    with _opened(directory, directory=True) as (_, info):
        _require(info.st_uid == intent_info.st_uid and info.st_gid == intent_info.st_gid
                 and not info.st_mode & 0o022 and stat.S_IMODE(info.st_mode) in {0o700, 0o750},
                 'scene_retirement_receipt_permissions')
    return directory


def _publish(directory, name, raw, allowance, *, prior=None):
    """Publish bounded bytes; CAS validates the prior named token and full stat."""
    _require(type(name) is str and '/' not in name and name not in {'', '.', '..'})
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES, 'scene_retirement_receipt_limit')
    temporary = '.' + secrets.token_hex(16) + '.pending'
    with _opened(directory, directory=True) as (parent, parent_info):
        _require(not parent_info.st_mode & 0o022, 'scene_retirement_receipt_permissions')
        allowance.tick()
        _parent(directory, parent, parent_info)
        fd, identity = _new_file(parent, temporary, parent_identity=_identity(parent_info))
        placed = False
        try:
            allowance.tick()
            _parent(directory, parent, parent_info)
            _named(parent, _identity(parent_info), temporary, fd, identity)
            os.fchmod(fd, 0o644)
            identity = (*identity[:2], (identity[2] & ~0o7777) | 0o644)
            _named(parent, _identity(parent_info), temporary, fd, identity)
            _require(os.fstat(fd).st_uid == access._POLICY_UID, 'scene_retirement_receipt_permissions')
            view = memoryview(raw)
            while view:
                allowance.tick()
                _parent(directory, parent, parent_info)
                _named(parent, _identity(parent_info), temporary, fd, identity)
                written = os.write(fd, view)
                _require(written > 0)
                view = view[written:]
            allowance.tick()
            _parent(directory, parent, parent_info)
            _named(parent, _identity(parent_info), temporary, fd, identity)
            os.fsync(fd)
            allowance.tick()
            _parent(directory, parent, parent_info)
            _named(parent, _identity(parent_info), temporary, fd, identity)
            if prior is None:
                os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
                _named(parent, _identity(parent_info), temporary, fd, identity)
                os.unlink(temporary, dir_fd=parent)
            else:
                prior_fd, prior_info = prior
                _guard(prior_fd, _identity(prior_info))
                _require(_version(os.fstat(prior_fd)) == _version(prior_info)
                         == _version(os.stat(name, dir_fd=parent, follow_symlinks=False)),
                         'scene_retirement_receipt_changed')
                os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            placed = True
            allowance.tick()
            _parent(directory, parent, parent_info)
            _guard(fd, identity)
            _require(_identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) == identity)
            os.fsync(parent)
            allowance.tick()
        finally:
            incoming = sys.exc_info()[1]
            cleanup_failed = False
            if not placed:
                try:
                    _parent(directory, parent, parent_info)
                    _named(parent, _identity(parent_info), temporary, fd, identity)
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
                except (ValueError, OSError):
                    cleanup_failed = True
            failure = _close_owned(fd, identity)
            if failure or cleanup_failed:
                if incoming is None:
                    raise access.SceneRetirementAccessError('scene_retirement_descriptor_cleanup_failed')
                incoming.add_note('scene_retirement_descriptor_cleanup_failed')
    reference = _ref(directory / name, raw)
    _, observed, _ = _read(directory / name, allowance, selected=reference, projection=True)
    return observed


def _members(consent, preserved, allowance):
    rows = consent['members']
    _require(type(rows) is list and 0 < len(rows) <= 256
             and len(rows) == len(preserved['members']), 'scene_retirement_receipt_members_invalid')
    result = []
    for index, row in enumerate(rows):
        allowance.tick()
        path = _canonical(row['canonical_path'])
        _require(str(path) == preserved['members'][index]['path']
                 and type(row['generation_id']) is str and TOKEN.fullmatch(row['generation_id'])
                 and type(row['class']) is str and 0 < len(row['class']) <= 128
                 and type(row['inventory_sha256']) is str and SHA.fullmatch(row['inventory_sha256']),
                 'scene_retirement_receipt_members_invalid')
        result.append({'canonical_path': str(path), 'class': row['class'],
                       'generation_id': row['generation_id'], 'inventory_sha256': row['inventory_sha256'],
                       'action': 'pending'})
    return result


def publish_pending_receipt(policy, consent, journal, preserved, allowance):
    """Publish/read back the intent receipt before any retirement transition."""
    directory = _location(policy, consent, allowance)
    _require(type(journal.token) is str and TOKEN.fullmatch(journal.token))
    _require(Path(journal.initial_ref['path']).parent == Path(policy['journal_store']))
    allowance.tick()
    initial = selected_document(journal.initial_ref, maximum=16*1024*1024, protected=True)
    _require(initial.get('token') == journal.token and initial.get('status') == 'pending'
             and initial.get('intent_id') == consent['intent_id']
             and initial.get('intent_raw_ref') == consent['intent_raw_ref']
             and initial.get('plan_raw_ref') == consent['plan_raw_ref']
             and initial.get('members') == consent['members'] and initial.get('preserved') == preserved,
             'scene_retirement_receipt_journal_changed')
    members = _members(consent, preserved, allowance)
    _require(not any(directory.is_relative_to(Path(row['canonical_path'])) for row in members),
             'scene_retirement_receipt_inside_payload')
    archive = preserved['archive']
    _require(type(archive) is dict and type(archive.get('uri')) is str and len(archive['uri']) <= 4096
             and type(archive.get('sha256')) is str and SHA.fullmatch(archive['sha256'])
             and archive.get('fresh_readback_sha256') == archive.get('sha256')
             and type(archive.get('size_bytes')) is int
             and archive.get('fresh_readback_size_bytes') == archive['size_bytes'],
             'scene_retirement_readback_unproven')
    _require(type(preserved['unique_allocated_bytes']) is int and preserved['unique_allocated_bytes'] >= 0,
             'scene_retirement_receipt_members_invalid')
    value = {'schema_version': 'scene_lifecycle_retirement_receipt.v1', 'status': 'pending',
             'intent_id': consent['intent_id'], 'intent_raw_ref': consent['intent_raw_ref'],
             'plan_raw_ref': consent['plan_raw_ref'], 'retiring_token': journal.token,
             'journal_initial_raw_ref': journal.initial_ref, 'members': members, 'archive': archive,
             'planned_unique_allocated_bytes': preserved['unique_allocated_bytes']}
    value['receipt_digest'] = canonical_digest(value, digest_field='receipt_digest')
    raw = _encode(value)  # Refuse oversize before history or projection mutation.
    _publish(directory, 'scene-retired.' + journal.token + '.pending.json', raw, allowance)
    return _publish(directory, NAME, raw, allowance)


def publish_terminal_receipt(policy, consent, pending_raw_ref, receipt, allowance):
    """Advance exact pending projection only after the immutable retired snapshot."""
    directory = _location(policy, consent, allowance)
    expected = raw_reference(pending_raw_ref)
    _require(expected['path'] == str(directory / NAME), 'scene_retirement_receipt_changed')
    pending, _, before = _read(directory / NAME, allowance, selected=expected, projection=True)
    _require(pending.get('receipt_digest') == canonical_digest(pending, digest_field='receipt_digest')
             and pending.get('status') == 'pending' and pending.get('intent_id') == consent['intent_id']
             and pending.get('intent_raw_ref') == consent['intent_raw_ref']
             and type(pending.get('retiring_token')) is str and TOKEN.fullmatch(pending['retiring_token'])
             and pending.get('retiring_token') == receipt.get('token')
             and receipt.get('status') == 'retired' and receipt.get('intent_id') == consent['intent_id'],
             'scene_retirement_receipt_changed')
    history_ref = dict(expected, path=str(directory / ('scene-retired.' + pending['retiring_token'] + '.pending.json')))
    history, _, _ = _read(Path(history_ref['path']), allowance, selected=history_ref, projection=True)
    _require(history == pending, 'scene_retirement_receipt_changed')
    snapshot_ref = raw_reference(receipt['retired_journal_raw_ref'])
    _require(Path(snapshot_ref['path']).parent == Path(policy['journal_store']) / 'retired'
             and Path(snapshot_ref['path']).name == snapshot_ref['sha256'][7:] + '.json',
             'scene_retirement_restore_snapshot_invalid')
    allowance.tick()
    snapshot = selected_document(snapshot_ref, maximum=16*1024*1024, protected=True)
    _require(snapshot.get('status') == 'retired' and snapshot.get('token') == receipt['token']
             and snapshot.get('intent_id') == consent['intent_id'] and snapshot.get('members') == consent['members'],
             'scene_retirement_restore_snapshot_invalid')
    outcomes = receipt['members']
    _require(type(outcomes) is list and len(outcomes) == len(pending['members']))
    expected_paths = {row['canonical_path'] for row in pending['members']}
    _require({row['canonical_path'] for row in outcomes} == expected_paths
             and all(row.get('outcome') == 'removed' for row in outcomes), 'scene_retirement_receipt_members_invalid')
    value = {key: item for key, item in pending.items() if key != 'receipt_digest'}
    value.update(status='retired', pending_receipt_raw_ref=history_ref, retired_journal_raw_ref=snapshot_ref,
                 members=[dict(row, action='offloaded', outcome='removed') for row in pending['members']])
    value['receipt_digest'] = canonical_digest(value, digest_field='receipt_digest')
    raw = _encode(value)
    with _opened(directory / NAME) as (prior_fd, current):
        _require(_version(current) == _version(before), 'scene_retirement_receipt_changed')
        _publish(directory, 'scene-retired.' + receipt['token'] + '.terminal.json', raw, allowance)
        return _publish(directory, NAME, raw, allowance, prior=(prior_fd, current))
