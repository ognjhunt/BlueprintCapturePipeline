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
from .task_evaluation_scene_retirement_journal import SceneJournal

MAX_BYTES = 1024 * 1024
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


def _read(path, allowance, *, selected=None, projection=False, maximum=MAX_BYTES):
    with _opened(path) as (fd, info):
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                 and not info.st_mode & 0o022 and 0 < info.st_size <= maximum,
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
    value, _, intent_info = _read(intent_path, allowance, selected=selected, maximum=65536)
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
        fd, identity = _new_file(parent, temporary, parent_identity=_identity(parent_info),action_guard=allowance.tick)
        placed = False
        try:
            allowance.tick()
            _parent(directory, parent, parent_info)
            _named(parent, _identity(parent_info), temporary, fd, identity)
            allowance.tick()
            os.fchmod(fd, 0o644)
            identity = (*identity[:2], (identity[2] & ~0o7777) | 0o644)
            _named(parent, _identity(parent_info), temporary, fd, identity)
            _require(os.fstat(fd).st_uid == access._POLICY_UID, 'scene_retirement_receipt_permissions')
            view = memoryview(raw)
            while view:
                allowance.tick()
                _parent(directory, parent, parent_info)
                _named(parent, _identity(parent_info), temporary, fd, identity)
                allowance.tick()
                written = os.write(fd, view)
                _require(written > 0)
                view = view[written:]
            allowance.tick()
            _parent(directory, parent, parent_info)
            _named(parent, _identity(parent_info), temporary, fd, identity)
            allowance.tick()
            os.fsync(fd)
            allowance.tick()
            _parent(directory, parent, parent_info)
            _named(parent, _identity(parent_info), temporary, fd, identity)
            if prior is None:
                allowance.tick()
                os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
                _named(parent, _identity(parent_info), temporary, fd, identity)
                allowance.tick()
                os.unlink(temporary, dir_fd=parent)
            else:
                prior_fd, prior_info = prior
                _guard(prior_fd, _identity(prior_info))
                _require(_version(os.fstat(prior_fd)) == _version(prior_info)
                         == _version(os.stat(name, dir_fd=parent, follow_symlinks=False)),
                         'scene_retirement_receipt_changed')
                allowance.tick()
                os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            placed = True
            allowance.tick()
            _parent(directory, parent, parent_info)
            _guard(fd, identity)
            _require(_identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) == identity)
            allowance.tick()
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


def _cache_members(objects,preserved,allowance):
    aliases=preserved.get('cache_aliases',[])
    _require(type(objects) is list and type(aliases) is list and len(objects)==len(aliases)<=256,
             'scene_retirement_receipt_members_invalid')
    result=[]
    for target,alias in zip(objects,aliases):
        allowance.tick()
        _require(type(target) is dict and target['canonical_path']==alias['path']
            and target['digest']==alias['digest'] and target['size_bytes']==alias['size_bytes']
            and type(target['generation_id']) is str and TOKEN.fullmatch(target['generation_id']),
            'scene_retirement_receipt_members_invalid')
        row={key:target[key] for key in ('canonical_path','digest','size_bytes','generation_id')}
        row.update(action='pending',**{'class':'cache'})
        _require(sum(len(_encode(item)) for item in result)+len(_encode(row))<=MAX_BYTES,
                 'scene_retirement_receipt_limit')
        result.append(row)
    return result


def _measured_cache(policy,pending,journal,allowance,*,snapshot=None,outcomes=None,restoring=False,status=None):
    """Project only exact protected native cache events, never caller counters."""
    reference=pending['journal_initial_raw_ref']
    allowance.charge('local_bytes',reference['size_bytes'])
    initial=selected_document(reference,maximum=16*1024*1024,protected=True)
    objects=initial.get('cache_objects',[])
    result=_cache_members(objects,initial['preserved'],allowance)
    _require(result==pending.get('cache_members',[]),'scene_retirement_receipt_members_invalid')
    aliases=initial['preserved'].get('cache_aliases',[])
    events=[event for event in journal.events if event['event']==('cache_alias_restored' if restoring else 'cache_unlinked')]
    _require(len(events)<=len(result),'scene_retirement_receipt_members_invalid')
    recorded=[]
    for index,event in enumerate(events):
        allowance.tick()
        evidence=event['evidence']
        _require(event['member_key']=='cache-'+str(index) and evidence['canonical_path']==result[index]['canonical_path']
            and evidence['digest']==result[index]['digest'],'scene_retirement_receipt_event_invalid')
        if restoring:
            _require(type(evidence['restore_identity']) is list and len(evidence['restore_identity'])==3,
                     'scene_retirement_receipt_event_invalid')
            activated=any(row['event']=='restored-active' and row['member_key']==event['member_key']
                and row['evidence']==evidence for row in journal.events)
            if not activated:
                continue
            key=hashlib.sha256(result[index]['canonical_path'].encode()).hexdigest()+'.json'
            allowance.charge('local_bytes',65536)
            generation,_,info=_read(Path(policy['generation_store'])/key,allowance,maximum=65536)
            service=access._service_identity()
            _require(stat.S_IMODE(info.st_mode)==0o600 and (info.st_uid,info.st_gid)==service
                and generation.get('schema_version')=='scene_content_generation.v1'
                and generation.get('state_digest')==canonical_digest(generation,digest_field='state_digest')
                and generation.get('state')=='restored-active'
                and generation.get('retirement_token')==pending['retiring_token']
                and all(generation.get(field)==result[index][field] for field in
                    ('canonical_path','digest','size_bytes','generation_id')),
                'scene_retirement_receipt_members_invalid')
            with _opened(result[index]['canonical_path']) as (_,observed):
                _require(list(_identity(observed))==evidence['restore_identity']
                    ==[generation['dev'],generation['ino'],generation['mode']]
                    and observed.st_size==result[index]['size_bytes']
                    and (observed.st_uid,observed.st_gid)==(aliases[index]['uid'],aliases[index]['gid'])
                    and stat.S_IMODE(observed.st_mode)==aliases[index]['mode'],
                    'scene_retirement_receipt_members_invalid')
            result[index].update(action='restored',restore_identity=evidence['restore_identity'],
                                 restore_event_raw_ref=event['raw_ref'])
        else:
            original=next(row for row in initial['preserved']['files'] if
                row['member_index']==aliases[index]['member_index'] and row['relative_path']==aliases[index]['relative_path'])
            _require(evidence.get('outcome')=='removed' and evidence.get('size_bytes')==result[index]['size_bytes']
                and type(evidence.get('removed_allocated_bytes')) is int
                and 0<=evidence['removed_allocated_bytes']<=original['allocated_bytes']
                and evidence.get('allocation_method')=='observed_file_st_blocks_512_last_union_link_unlinked',
                'scene_retirement_receipt_members_invalid')
            recorded.append(dict(evidence,event_raw_ref=event['raw_ref']))
            result[index].update(action='offloaded',outcome='removed',
                removed_allocated_bytes=evidence['removed_allocated_bytes'],
                allocation_method=evidence['allocation_method'],event_raw_ref=event['raw_ref'])
    if not restoring:
        _require(outcomes is None or outcomes==recorded,'scene_retirement_receipt_members_invalid')
        if snapshot is not None:
            _require(snapshot.get('cache_outcomes',[])==recorded and len(recorded)==len(result),
                     'scene_retirement_receipt_members_invalid')
            generations=snapshot.get('cache_generations',[])
            _require(len(generations)==len(result) and all(generation.get('state')=='retired'
                and all(generation.get(field)==row[field] for field in
                    ('canonical_path','digest','size_bytes','generation_id'))
                for generation,row in zip(generations,result)),'scene_retirement_receipt_members_invalid')
    else:
        for row in result:
            if row['action']!='restored':
                row['action']='restoring'
        _require(status!='restored' or all(row['action']=='restored' for row in result),
                 'scene_retirement_receipt_members_invalid')
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
    shared_keeps=initial.get('unselected_shared_content_keeps',[])
    _require(type(shared_keeps) is list and len(shared_keeps)<=10000 and
             all(type(row) is dict and set(row)=={
                     'canonical_path','action','observation_status','reasons'}
                 and row['action']=='KEEP'
                 and (row['observation_status'],row['reasons']) in (
                     ('observed_scoped_metadata',['shared_content_object_not_exclusive']),
                     ('incomplete_scoped_metadata',[
                         'member_or_child_unavailable_or_changed','shared_content_object_not_exclusive']),
                     ('incomplete_scoped_metadata',[
                         'shared_content_object_not_exclusive','member_or_child_unavailable_or_changed']))
                 and type(row['canonical_path']) is str
                 and str(_canonical(row['canonical_path']))==row['canonical_path']
                 and not any(Path(row['canonical_path']).is_relative_to(Path(member['canonical_path']))
                             for member in consent['members']) for row in shared_keeps)
             and len({row['canonical_path'] for row in shared_keeps})==len(shared_keeps),
             'scene_retirement_receipt_members_invalid')
    value['unselected_shared_content_keeps']=shared_keeps
    _require(initial.get('cache_objects',[])==consent.get('cache_objects',[]),'scene_retirement_receipt_members_invalid')
    cache_members=_cache_members(initial.get('cache_objects',[]),preserved,allowance)
    if cache_members:
        value['cache_members']=cache_members
    value['receipt_digest'] = canonical_digest(value, digest_field='receipt_digest')
    raw = _encode(value)  # Refuse oversize before history or projection mutation.
    _publish(directory, 'scene-retired.' + journal.token + '.pending.json', raw, allowance)
    return _publish(directory, NAME, raw, allowance)


def _measured_members(policy, pending, snapshot, outcomes, allowance, *, partial=False):
    _require(type(outcomes) is list and len(outcomes) <= len(pending['members'])
             and (partial or len(outcomes) == len(pending['members']))
             and (snapshot is None or snapshot.get('outcomes') == outcomes),
             'scene_retirement_receipt_members_invalid')
    measured = []
    fields = ('logical_bytes', 'apparent_bytes', 'unique_allocated_bytes',
              'removed_allocated_bytes', 'removed_file_count')
    for index, (selected, outcome) in enumerate(zip(pending['members'], outcomes)):
        allowance.tick()
        _require(type(outcome) is dict and outcome.get('outcome') == 'removed'
                 and all(outcome.get(key) == selected[key]
                         for key in ('canonical_path', 'generation_id', 'inventory_sha256')),
                 'scene_retirement_receipt_members_invalid')
        _require(all(type(outcome.get(key)) is int and 0 <= outcome[key] < 2**63 for key in fields)
                 and outcome['removed_allocated_bytes'] <= outcome['unique_allocated_bytes']
                 and outcome['apparent_bytes'] == outcome['logical_bytes']
                 and outcome.get('allocation_method') == 'observed_file_st_blocks_512_last_union_link_unlinked',
                 'scene_retirement_receipt_members_invalid')
        event_ref = raw_reference(outcome['event_raw_ref'])
        _require(Path(event_ref['path']).parent == Path(policy['journal_store'])
                 and event_ref['size_bytes'] <= 65536, 'scene_retirement_receipt_event_invalid')
        allowance.tick()
        event = selected_document(event_ref, maximum=65536, protected=True)
        sequence = event.get('sequence')
        _require(type(sequence) is int and 1 <= sequence <= 10000
                 and Path(event_ref['path']).name == pending['retiring_token'] + '.' + str(sequence) + '.json'
                 and event.get('schema_version') == 'scene_retirement_journal_event.v1'
                 and event.get('event_digest') == canonical_digest(event, digest_field='event_digest')
                 and event.get('token') == pending['retiring_token']
                 and event.get('event') == 'member_removed' and event.get('member_key') == str(index)
                 and event.get('evidence') == {key: value for key, value in outcome.items() if key != 'event_raw_ref'},
                 'scene_retirement_receipt_event_invalid')
        measured.append(dict(selected, action='offloaded', outcome='removed',
                             **{key: outcome[key] for key in fields},
                             allocation_method=outcome['allocation_method'], event_raw_ref=event_ref))
    return measured


def _projection(policy, consent, reference, allowance, *, restoring=False):
    directory = _location(policy, consent, allowance)
    expected = raw_reference(reference)
    _require(expected['path'] == str(directory / NAME), 'scene_retirement_receipt_changed')
    current, _, before = _read(directory / NAME, allowance, selected=expected, projection=True)
    token = current.get('retiring_token')
    _require(current.get('schema_version') == 'scene_lifecycle_retirement_receipt.v1'
             and current.get('receipt_digest') == canonical_digest(current, digest_field='receipt_digest')
             and current.get('intent_id') == consent['intent_id']
             and current.get('intent_raw_ref') == consent['intent_raw_ref']
             and type(token) is str and TOKEN.fullmatch(token), 'scene_retirement_receipt_changed')
    status = current.get('status')
    if restoring and 'restore_journal_initial_raw_ref' in current:
        restore_token = current.get('restore_token')
        sequence = current.get('journal_sequence')
        _require(type(restore_token) is str and TOKEN.fullmatch(restore_token)
                 and type(sequence) is int and 0 <= sequence <= 10000
                 and status in {'restoring', 'restored', 'incomplete'}, 'scene_retirement_receipt_changed')
        name = ('scene-retired.' + token + '.restore.' + restore_token + '.' + str(sequence) + '.'
                + status + '.' + expected['sha256'][7:] + '.json')
    elif restoring and status == 'retired':
        name = 'scene-retired.' + token + '.terminal.json'
    elif status == 'pending':
        name = 'scene-retired.' + token + '.pending.json'
    elif status in {'retiring', 'incomplete'}:
        sequence = current.get('journal_sequence')
        _require(type(sequence) is int and 0 <= sequence <= 10000, 'scene_retirement_receipt_changed')
        name = 'scene-retired.' + token + '.' + str(sequence) + '.' + status + '.' + expected['sha256'][7:] + '.json'
    else:
        _require(False, 'scene_retirement_receipt_changed')
    history_ref = dict(expected, path=str(directory / name))
    history, _, _ = _read(Path(history_ref['path']), allowance, selected=history_ref, projection=True)
    _require(history == current, 'scene_retirement_receipt_changed')
    pending_ref = history_ref if status == 'pending' else raw_reference(current['pending_receipt_raw_ref'])
    _require(pending_ref['path'] == str(directory / ('scene-retired.' + token + '.pending.json')),
             'scene_retirement_receipt_changed')
    pending, _, _ = _read(Path(pending_ref['path']), allowance, selected=pending_ref, projection=True)
    _require(pending.get('status') == 'pending' and pending.get('retiring_token') == token
             and pending.get('receipt_digest') == canonical_digest(pending, digest_field='receipt_digest')
             and (restoring or pending.get('plan_raw_ref') == consent['plan_raw_ref'])
             and pending.get('intent_raw_ref') == consent['intent_raw_ref']
             and pending.get('intent_id') == consent['intent_id'], 'scene_retirement_receipt_changed')
    return directory, current, before, history_ref, pending, pending_ref


def _resume(policy, pending, allowance):
    reference = raw_reference(pending['journal_initial_raw_ref'])
    _require(reference['path'] == str(Path(policy['journal_store']) / (pending['retiring_token'] + '.initial.json')),
             'scene_retirement_receipt_journal_changed')
    return SceneJournal.resume(reference, allowance=allowance)


def _immutable(directory, name, raw, allowance):
    try:
        return _publish(directory, name, raw, allowance)
    except FileExistsError:
        # A crash may have persisted this exact version before current CAS.
        # Re-select it; never overwrite or remove an orphan or foreign version.
        selected = _ref(directory / name, raw)
        _, observed, _ = _read(directory / name, allowance, selected=selected, projection=True)
        return observed


def _restored_generation(policy, selected, outcome, original_token, allowance):
    allowance.tick()
    service = access._service_identity()
    with _opened(policy['generation_store'], directory=True) as (_, info):
        _require(stat.S_IMODE(info.st_mode) == 0o700 and (info.st_uid, info.st_gid) == service,
                 'scene_retirement_service_identity_unproven')
    key = hashlib.sha256(selected['canonical_path'].encode()).hexdigest() + '.json'
    state, _, info = _read(Path(policy['generation_store']) / key, allowance, maximum=65536)
    capture = 'capture_owner_user_id' in selected
    owner_fields = ('capture_owner_user_id', 'owner_observation_raw_ref', 'birth_delivery_raw_ref') if capture else (
        'owner_intent_id', 'owner_raw_ref')
    _require(stat.S_IMODE(info.st_mode) == 0o600 and (info.st_uid, info.st_gid) == service
             and state.get('schema_version') == (
                 'scene_capture_generation.v1' if capture else 'scene_member_generation.v1')
             and state.get('state_digest') == canonical_digest(state, digest_field='state_digest')
             and state.get('state') == 'restored-active' and state.get('retirement_token') == original_token
             and all(state.get(field) == selected[field] for field in
                     ('canonical_path', 'generation_id', *owner_fields)),
             'scene_retirement_generation_unavailable')
    identity = outcome.get('restore_identity')
    _require(type(identity) is list and len(identity) == 3
             and all(type(value) is int and 0 <= value < 2**63 for value in identity)
             and identity == [state.get(field) for field in ('dev', 'ino', 'mode')],
             'scene_retirement_generation_unavailable')
    with _opened(selected['canonical_path'], directory=True) as (_, current):
        _require(list(_identity(current)) == identity, 'scene_retirement_generation_unavailable')


def _restore_progress(policy, consent, current_raw_ref, progress, allowance):
    directory, current, before, history_ref, pending, pending_ref = _projection(
        policy, consent, current_raw_ref, allowance, restoring=True)
    token = progress.get('token')
    _require(progress.get('status') in {'restoring', 'restored', 'incomplete'}
             and type(token) is str and TOKEN.fullmatch(token) and token != pending['retiring_token']
             and progress.get('original_retirement_token') == pending['retiring_token']
             and progress.get('intent_id') == consent['intent_id'], 'scene_retirement_receipt_changed')
    initial_ref = raw_reference(progress['restore_journal_initial_raw_ref'])
    _require(initial_ref['path'] == str(Path(policy['journal_store']) / (token + '.initial.json')),
             'scene_retirement_receipt_journal_changed')
    allowance.tick()
    initial = selected_document(initial_ref, maximum=16*1024*1024, protected=True)
    _require(initial.get('schema_version') == 'scene_restore_journal.v1' and initial.get('status') == 'restoring'
             and initial.get('intent_id') == consent['intent_id'] and initial.get('intent_raw_ref') == consent['intent_raw_ref']
             and initial.get('members') == consent['members']
             and initial.get('original_retirement_token') == pending['retiring_token']
             and initial.get('retired_journal_raw_ref') == consent['retired_journal_raw_ref'],
             'scene_retirement_receipt_journal_changed')
    allowance.tick()
    _require(selected_document(initial['consent_raw_ref'], maximum=512*1024, protected=True) == consent,
             'scene_retirement_receipt_journal_changed')
    snapshot_ref = raw_reference(consent['retired_journal_raw_ref'])
    _require(Path(snapshot_ref['path']).parent == Path(policy['journal_store']) / 'retired'
             and Path(snapshot_ref['path']).name == snapshot_ref['sha256'][7:] + '.json',
             'scene_retirement_restore_snapshot_invalid')
    allowance.tick()
    snapshot = selected_document(snapshot_ref, maximum=16*1024*1024, protected=True)
    _require(bool(snapshot.get('cache_objects'))==bool(pending.get('cache_members')),
             'scene_retirement_receipt_members_invalid')
    _require(snapshot.get('schema_version') == 'scene_retirement_journal.v1' and snapshot.get('status') == 'retired'
             and snapshot.get('journal_digest') == canonical_digest(snapshot, digest_field='journal_digest')
             and snapshot.get('token') == pending['retiring_token']
             and snapshot.get('intent_id') == consent['intent_id'] and snapshot.get('members') == consent['members']
             and snapshot.get('unselected_shared_content_keeps',[]) == pending.get('unselected_shared_content_keeps',[]),
             'scene_retirement_restore_snapshot_invalid')
    if 'restore_journal_initial_raw_ref' in current:
        _require(current['restore_journal_initial_raw_ref'] == initial_ref and current['restore_token'] == token
                 and current.get('retired_journal_raw_ref') == snapshot_ref
                 and (current['status'] != 'restored' or progress['status'] == 'restored'),
                 'scene_retirement_receipt_changed')
    elif 'retired_journal_raw_ref' in current:
        _require(current['retired_journal_raw_ref'] == snapshot_ref, 'scene_retirement_receipt_changed')
    journal = SceneJournal.resume(initial_ref, allowance=allowance)
    _require(raw_reference(progress['last_event_raw_ref']) == journal.prior_ref,
             'scene_retirement_receipt_event_invalid')
    outcomes = progress['members']
    recorded = [event for event in journal.events if event['event'] == 'member_restored']
    previous_count = sum('restore_event_raw_ref' in row for row in current['members'])
    _require(type(outcomes) is list and previous_count <= len(outcomes) <= len(consent['members'])
             and len(outcomes) <= len(recorded)
             and (progress['status'] != 'restored' or len(outcomes) == len(consent['members'])),
             'scene_retirement_receipt_members_invalid')
    members = _measured_members(policy, pending, snapshot, snapshot['outcomes'], allowance)
    cache_members=_measured_cache(policy,pending,journal,allowance,restoring=True,status=progress['status']) if pending.get('cache_members') else []
    for index, (selected, outcome) in enumerate(zip(consent['members'], outcomes)):
        event = recorded[index]
        _require(type(outcome) is dict and outcome.get('outcome') == 'restored'
                 and outcome.get('canonical_path') == selected['canonical_path']
                 and event['member_key'] == str(index) and event['evidence'] == outcome,
                 'scene_retirement_receipt_members_invalid')
        _restored_generation(policy, selected, outcome, pending['retiring_token'], allowance)
        members[index].update(action='restored', outcome='restored', restore_identity=outcome['restore_identity'],
                              restore_event_raw_ref=event['raw_ref'])
    for member in members[len(outcomes):]:
        member.update(action='restoring')
    if (current.get('status'), current.get('journal_sequence'), current.get('last_event_raw_ref'), current['members'],current.get('cache_members',[])) == (
            progress['status'], journal.sequence, journal.prior_ref, members,cache_members):
        return raw_reference(current_raw_ref)
    value = {key: item for key, item in pending.items() if key != 'receipt_digest'}
    value.update(status=progress['status'], members=members, journal_sequence=journal.sequence,
                 last_event_raw_ref=journal.prior_ref, pending_receipt_raw_ref=pending_ref,
                 prior_receipt_raw_ref=history_ref, retired_journal_raw_ref=snapshot_ref,
                 restore_token=token, restore_journal_initial_raw_ref=initial_ref)
    if cache_members:
        value['cache_members']=cache_members
    value['receipt_digest'] = canonical_digest(value, digest_field='receipt_digest')
    raw = _encode(value)
    name = ('scene-retired.' + pending['retiring_token'] + '.restore.' + token + '.' + str(journal.sequence) + '.'
            + progress['status'] + '.' + _ref(directory / NAME, raw)['sha256'][7:] + '.json')
    with _opened(directory / NAME) as (prior_fd, info):
        _require(_version(info) == _version(before), 'scene_retirement_receipt_changed')
        _immutable(directory, name, raw, allowance)
        return _publish(directory, NAME, raw, allowance, prior=(prior_fd, info))


def publish_progress_receipt(policy, consent, current_raw_ref, progress, allowance):
    """Record actual retirement progress without clearing or extending authority."""
    _require(type(progress) is dict, 'scene_retirement_receipt_changed')
    if 'restore_journal_initial_raw_ref' in progress or progress.get('status') in {'restoring', 'restored'}:
        return _restore_progress(policy, consent, current_raw_ref, progress, allowance)
    directory, current, before, history_ref, pending, pending_ref = _projection(
        policy, consent, current_raw_ref, allowance)
    _require(type(progress) is dict and progress.get('status') in {'retiring', 'incomplete'}
             and progress.get('token') == pending['retiring_token']
             and progress.get('intent_id') == consent['intent_id'], 'scene_retirement_receipt_changed')
    journal = _resume(policy, pending, allowance)
    last = raw_reference(progress['last_event_raw_ref'])
    _require(last == journal.prior_ref and journal.sequence >= current.get('journal_sequence', 0),
             'scene_retirement_receipt_event_invalid')
    outcomes = progress['members']
    recorded = [event for event in journal.events if event['event'] == 'member_removed']
    _require(type(outcomes) is list and len(recorded) == len(outcomes)
             and all(event['raw_ref'] == row.get('event_raw_ref') for event, row in zip(recorded, outcomes)),
             'scene_retirement_receipt_members_invalid')
    members = _measured_members(policy, pending, None, outcomes, allowance, partial=True)
    cache_members=_measured_cache(policy,pending,journal,allowance) if pending.get('cache_members') else []
    for index, selected in enumerate(pending['members'][len(members):], start=len(members)):
        started = any(event['member_key'] == str(index) and event['event'] != 'kept' for event in journal.events)
        members.append(dict(selected, action='retiring' if started else 'pending'))
    # A replay of the exact recorded progress needs no new version or mutation.
    if (current.get('status'), current.get('journal_sequence'), current.get('last_event_raw_ref'), current['members'],current.get('cache_members',[])) == (
            progress['status'], journal.sequence, last, members,cache_members):
        return raw_reference(current_raw_ref)
    value = {key: item for key, item in pending.items() if key != 'receipt_digest'}
    value.update(status=progress['status'], members=members, journal_sequence=journal.sequence,
                 last_event_raw_ref=last, pending_receipt_raw_ref=pending_ref, prior_receipt_raw_ref=history_ref)
    if cache_members:
        value['cache_members']=cache_members
    value['receipt_digest'] = canonical_digest(value, digest_field='receipt_digest')
    raw = _encode(value)
    name = ('scene-retired.' + pending['retiring_token'] + '.' + str(journal.sequence) + '.'
            + progress['status'] + '.' + _ref(directory / NAME, raw)['sha256'][7:] + '.json')
    with _opened(directory / NAME) as (prior_fd, info):
        _require(_version(info) == _version(before), 'scene_retirement_receipt_changed')
        _immutable(directory, name, raw, allowance)
        return _publish(directory, NAME, raw, allowance, prior=(prior_fd, info))


def publish_terminal_receipt(policy, consent, pending_raw_ref, receipt, allowance):
    """Advance exact pending/progress only after the immutable retired snapshot."""
    directory, _, before, history_ref, pending, pending_ref = _projection(
        policy, consent, pending_raw_ref, allowance)
    _require(pending.get('retiring_token') == receipt.get('token')
             and receipt.get('status') == 'retired' and receipt.get('intent_id') == consent['intent_id'],
             'scene_retirement_receipt_changed')
    snapshot_ref = raw_reference(receipt['retired_journal_raw_ref'])
    _require(Path(snapshot_ref['path']).parent == Path(policy['journal_store']) / 'retired'
             and Path(snapshot_ref['path']).name == snapshot_ref['sha256'][7:] + '.json',
             'scene_retirement_restore_snapshot_invalid')
    allowance.tick()
    snapshot = selected_document(snapshot_ref, maximum=16*1024*1024, protected=True)
    _require(bool(snapshot.get('cache_objects'))==bool(pending.get('cache_members')),
             'scene_retirement_receipt_members_invalid')
    _require(snapshot.get('status') == 'retired' and snapshot.get('token') == receipt['token']
             and snapshot.get('journal_digest') == canonical_digest(snapshot, digest_field='journal_digest')
             and snapshot.get('intent_id') == consent['intent_id'] and snapshot.get('members') == consent['members']
             and snapshot.get('unselected_shared_content_keeps',[]) == pending.get('unselected_shared_content_keeps',[])
             == receipt.get('unselected_shared_content_keeps',[]),
             'scene_retirement_restore_snapshot_invalid')
    journal = _resume(policy, pending, allowance)
    _require(snapshot.get('sequence') == journal.sequence
             and snapshot.get('prior_event_sha256') == journal.prior_ref['sha256'],
             'scene_retirement_receipt_event_invalid')
    members = _measured_members(policy, pending, snapshot, receipt['members'], allowance)
    cache_members=_measured_cache(policy,pending,journal,allowance,snapshot=snapshot,
        outcomes=receipt.get('cache_outcomes',[])) if pending.get('cache_members') else []
    value = {key: item for key, item in pending.items() if key != 'receipt_digest'}
    value.update(status='retired', pending_receipt_raw_ref=pending_ref, prior_receipt_raw_ref=history_ref,
                 retired_journal_raw_ref=snapshot_ref, members=members,
                 journal_sequence=journal.sequence, last_event_raw_ref=journal.prior_ref)
    if cache_members:
        value['cache_members']=cache_members
    value['receipt_digest'] = canonical_digest(value, digest_field='receipt_digest')
    raw = _encode(value)
    with _opened(directory / NAME) as (prior_fd, current):
        _require(_version(current) == _version(before), 'scene_retirement_receipt_changed')
        _immutable(directory, 'scene-retired.' + receipt['token'] + '.terminal.json', raw, allowance)
        return _publish(directory, NAME, raw, allowance, prior=(prior_fd, current))
