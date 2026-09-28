"""Explicit root preservation restoration; archive presence is not permission."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import stat
from pathlib import Path

from . import control_plane_lane_experiment_actions as actions
from . import control_plane_lane_experiment_archive as archive
from . import control_plane_lane_experiment_birth as birth
from . import control_plane_lane_experiment_recovery as recovery
from . import control_plane_lane_experiment_retirement as issuance
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch as scratch
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_disk_budget import reserve_control_plane_disk
from .control_plane_lane_experiment_authority import _current
from .control_plane_lane_experiment_publication import _BirthFiles, _publish
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _epoch, _require
from .control_plane_reference_budget import ReferenceCollectionBudget
from .decision_evidence_contracts import canonical_digest

RESTORE_SCHEMA = 'control_plane_lane_experiment_restore_intent.v1'
_RESTORE_FIELDS = frozenset({'schema_version', 'intent_id', 'action_id', 'issuer_uid', 'principal', 'owner',
    'generation', 'birth', 'target_identity', 'lease', 'preservation', 'action', 'issued_at_epoch',
    'expires_at_epoch', 'policy', 'new_lease_ttl_seconds', 'new_lease_expires_at_epoch', 'action_digest'})


def _policy(files, config, principal, owner, issued, deadline, new_expiry):
    raw, _ = files.read(config.lane_owner_policy_file, cap=owners.MAX_POLICY_BYTES, protected=True, mode=0o600)
    policy = owners._policy(raw, principal, files.budget)
    # Existing REGISTER grants the new live lease; OFFLOAD grants preserved
    # evidence handling. No new shared owner-policy verb is fabricated.
    for action, expiry in (('register', new_expiry), ('offload', deadline)):
        decision = dict(owner=owner, action=action, expires_at_epoch=expiry)
        if action == 'register':
            decision['ttl_seconds'] = new_expiry - issued
        owners._authorize(decision, policy, expiry, issued)
    return issuance._selector(raw, files.budget)


def _document(files, path, cap, *, selector=None):
    raw, _ = files.read(path, cap=cap, protected=True, mode=0o600)
    if selector is not None:
        _require(issuance._selector(raw, files.budget) == selector, 'experiment_restore_record_changed')
    return raw, retained._document(raw, cap, _work_budget=files.budget)


def issue_restore(intent_id, *, principal, owner, lease_ttl_seconds, expires_at_epoch, installed_config_path, now):
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        issued = now()
        config, gid = actions._context(files, installed_config_path, issued)
        _require(config.experiment_retirement_enabled is True and type(lease_ttl_seconds) is int
                 and 0 < lease_ttl_seconds <= 1209600 and _epoch(expires_at_epoch) and issued < expires_at_epoch,
                 'experiment_restore_intent_invalid')
        public, current, entry = actions._selected(files, config, intent_id, issued, gid)
        _require(entry['state'] == 'retired' and entry['owner'] == owner and entry['operation_id'] is not None,
                 'experiment_restore_ineligible')
        target, _ = actions._target(files, config, entry)
        actions._lease(files, target, entry)
        original = actions._birth(files, public, entry, gid)
        public = birth._authority_lock(files, config.experiment_authority_root, gid)
        refreshed = _current(files, public, gid)
        _require(refreshed[0] == current[0], 'experiment_action_current_changed')
        store = issuance._store(files, config.experiment_record_store)
        store_path = Path(config.experiment_record_store)
        old_operation = entry['operation_id']
        prior_raw, prior = _document(files, store_path / (old_operation + '.action.json'), 32768)
        _require(set(prior) == actions._ACTION_FIELDS and prior['schema_version'] == actions.ACTION_SCHEMA
                 and prior['action_digest'] == canonical_digest(prior, digest_field='action_digest')
                 and prior['action_id'] == old_operation and prior['action'] == 'offload'
                 and all(prior[key] == entry[key] for key in ('intent_id', 'owner', 'generation', 'birth', 'lease', 'target_identity', 'completion')),
                 'experiment_restore_ineligible')
        prior_selector = issuance._selector(prior_raw, files.budget)
        manifest_raw, manifest = _document(files, store_path / (old_operation + '.manifest.json'), 1048576,
                                            selector=prior['manifest'])
        rows = sorted(manifest['rows'], key=lambda row: (len(Path(row[0]).parts), row[0]), reverse=True)
        retired, _, _ = recovery.retired(files, config, prior, prior_selector, target, rows, entry, original['marker'], _retained_store=store)
        ready_path = store_path / 'operations' / old_operation / 'e-00001.json'
        ready_raw, ready = _document(files, ready_path, 32768)
        _require(ready['event_kind'] == 'preservation_ready' and ready['body']['action'] == prior_selector,
                 'experiment_restore_ineligible')
        preservation = issuance._selector(ready_raw, files.budget)
        new_expiry = issued + lease_ttl_seconds
        policy = _policy(files, config, principal, owner, issued, expires_at_epoch, new_expiry)
        _require(policy == current[1]['policy'] and new_expiry <= current[1]['expires_at_epoch'],
                 'experiment_restore_authority_expiry')
        action_id = secrets.token_hex(16)
        _require(owners._matches(action_id, owners._CONSENT_ID) and action_id not in (intent_id, old_operation),
                 'experiment_restore_intent_invalid')
        value = dict(schema_version=RESTORE_SCHEMA, intent_id=intent_id, action_id=action_id, issuer_uid=0,
            principal=principal, owner=owner, generation=entry['generation'], birth=entry['birth'],
            target_identity=entry['target_identity'], lease=entry['lease'], preservation=preservation,
            action='restore', issued_at_epoch=issued, expires_at_epoch=expires_at_epoch, policy=policy,
            new_lease_ttl_seconds=lease_ttl_seconds, new_lease_expires_at_epoch=new_expiry)
        payload = actions._encoded(value, 'action_digest', 32768)
        occupied = issuance._capacity(files, store, adding_registration=False)
        _require(occupied + 256 * 1024 <= issuance.MAX_EXPERIMENT_STORE_BYTES, 'experiment_store_full')
        intent = _publish(files, store, action_id + '.restore-intent.json', payload, kind='private')
        selection = dict(schema_version='control_plane_lane_experiment_restore_selection.v1',
            restore_intent=intent, original_operation_id=old_operation, original_action=prior_selector,
            manifest=prior['manifest'], retired=retired, preservation=preservation)
        _publish(files, store, action_id + '.restore-selection.json', actions._encoded(selection, 'selection_digest', 32768), kind='private')
        prepared, old_head = actions._version(files, public, refreshed, entry | {'operation_id': action_id},
                                              gid, policy, issued)
        _publish(files, store, action_id + '.restore-pending-head.json', prepared, kind='private')
        files.verify()
        actions._install_head(files, public, prepared, gid, old_head)
        return dict(action_id=action_id, restore_intent=intent)
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()


def _restore_intent(files, config, action_id, expected, issued):
    _require(owners._matches(action_id, owners._CONSENT_ID), 'experiment_restore_intent_invalid')
    _, value = _document(files, Path(config.experiment_record_store) / (action_id + '.restore-intent.json'), 32768,
                         selector=expected)
    _require(set(value) == _RESTORE_FIELDS and value['schema_version'] == RESTORE_SCHEMA
             and value['action_digest'] == canonical_digest(value, digest_field='action_digest')
             and value['action_id'] == action_id and value['action'] == 'restore'
             and type(value['issuer_uid']) is int and value['issuer_uid'] == 0
             and value['issued_at_epoch'] <= issued < value['expires_at_epoch']
             and issued < value['new_lease_expires_at_epoch'], 'experiment_restore_intent_invalid')
    policy = _policy(files, config, value['principal'], value['owner'], value['issued_at_epoch'],
                     value['expires_at_epoch'], value['new_lease_expires_at_epoch'])
    _require(policy == value['policy'], 'experiment_policy_changed')
    return value


def _owner_mode(files, fd, uid, gid, mode):
    _require(type(uid) is int and type(gid) is int and type(mode) is int and 0 <= mode <= 0o777
             and not mode & 0o022, 'experiment_restore_member_mode')
    files.location(fd)
    original = files.proof(fd)
    os.fchown(fd, uid, gid)
    parent, name, _ = files.bindings[fd]
    named = os.stat(name, dir_fd=parent, follow_symlinks=False)
    _require(files.proof(fd) == original and owners._metadata(named) == owners._metadata(os.fstat(fd)),
             'experiment_restore_member_changed')
    files.bindings[fd] = (parent, name, owners._security(named))
    files.location(fd)
    os.fchmod(fd, mode)
    named = os.stat(name, dir_fd=parent, follow_symlinks=False)
    _require(files.proof(fd) == original and owners._metadata(named) == owners._metadata(os.fstat(fd)),
             'experiment_restore_member_changed')
    files.bindings[fd] = (parent, name, owners._security(named))
    files.location(fd)
    os.fsync(fd)


def _new_directory(files, parent, name):
    files.location(parent)
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise OwnerTargetVersionError('experiment_restore_destination_exists')
    os.mkdir(name, 0o700, dir_fd=parent)
    files.location(parent)
    os.fsync(parent)
    fd = files.open(name, os.O_RDONLY | os.O_DIRECTORY, parent=parent)
    info = files.acquired[fd]
    _require(info.st_uid == info.st_gid == 0 and stat.S_IMODE(info.st_mode) == 0o700,
             'experiment_restore_stage_unsafe')
    return fd


def _lease_cas(files, target, parent, previous, payload):
    """Fixed original lease inode, no truncate before independently named proof."""
    fd = files.open(scratch.LEASE_FILE, os.O_WRONLY | os.O_NONBLOCK, parent=parent)
    original = files.acquired[fd]
    _require(owners._metadata(original) == owners._metadata(previous.info), 'experiment_lease_changed')
    aliases = [r for r in files.records if r.parent == parent and r.name == scratch.LEASE_FILE]
    for record in aliases:
        files.verify_record(record)
    expected = owners._metadata(original)
    def check():
        files.location(parent)
        files.proof(fd)
        _require(expected == owners._metadata(os.fstat(fd))
                 == owners._metadata(os.stat(scratch.LEASE_FILE, dir_fd=parent, follow_symlinks=False)),
                 'experiment_restore_lease_changed')
    check()
    os.ftruncate(fd, 0)
    expected = owners._metadata(os.fstat(fd))
    count = 0
    while count < len(payload):
        check()
        written = os.write(fd, memoryview(payload)[count:])
        _require(type(written) is int and 0 < written <= len(payload) - count, 'experiment_restore_lease_changed')
        count += written
        expected = owners._metadata(os.fstat(fd))
    check()
    os.fsync(fd)
    check()
    os.fsync(parent)
    check()
    for record in aliases:
        files.records.remove(record)
        files.close(record.fd)
    files.close(fd)
    raw, _ = files.read(target / scratch.LEASE_FILE, cap=scratch.MAX_LEASE_BYTES)
    _require(raw == payload, 'experiment_restore_lease_changed')
    return issuance._selector(raw, files.budget)


def _stage_archive(files, config, target, target_fd, action, selection, manifest_raw, rows, guard):
    """Bounded stream parser into a fresh owned stage, never an archive pathname."""
    from urllib.parse import urlsplit
    ready_path = Path(config.experiment_record_store) / 'operations' / selection['original_operation_id'] / 'e-00001.json'
    _, ready = _document(files, ready_path, 32768, selector=action['preservation'])
    preserved = ready['body']['archive']
    archive.verify_preservation(files, config, preserved, guard)
    controller = archive._Controller(guard, preserved['size_bytes'])
    client, bucket = archive._client(files, config)
    uri = urlsplit(preserved['uri'])
    _require(uri.scheme == 's3' and uri.netloc == bucket, 'experiment_archive_target_changed')
    guarded = archive._LaneArchiveClient(client, bucket, uri.path[1:], controller)
    response = None
    stage_name = '.restore-' + action['action_id']
    stage = _new_directory(files, target_fd, stage_name)
    stage_path = target / stage_name
    directory_modes, staged = {}, []
    digest = hashlib.sha256()
    buffer, read_count = bytearray(), 0
    def exact(count):
        nonlocal read_count
        _require(type(count) is int and 0 <= count <= archive.QUANTUM, 'experiment_restore_decode_limit')
        while len(buffer) < count:
            controller.check('readback')
            block = response.read(archive.QUANTUM)
            _require(block and read_count + len(block) <= preserved['size_bytes'], 'experiment_restore_archive_short')
            read_count += len(block)
            digest.update(block)
            buffer.extend(block)
            _require(len(buffer) <= 2 * archive.QUANTUM, 'experiment_restore_decode_limit')
        result = bytes(buffer[:count])
        del buffer[:count]
        return result
    try:
        response = guarded.get_object(Bucket=bucket, Key=uri.path[1:])['Body']
        _require(exact(len(archive.MAGIC)) == archive.MAGIC, 'experiment_restore_archive_format')
        length = int.from_bytes(exact(4), 'big')
        _require(0 < length <= archive.QUANTUM and exact(length) == manifest_raw, 'experiment_restore_manifest_changed')
        for row in sorted(rows, key=lambda item: item[0]):
            relative, kind, _, token, sha = row
            path = Path(relative)
            _require(not path.is_absolute() and path.parts and all(p not in ('.', '..') for p in path.parts)
                     and path.parts[0] not in actions._METADATA, 'experiment_restore_manifest_invalid')
            original = tuple(map(int, token.split(':')))
            _require(len(original) == 7 and kind in ('file', 'directory'), 'experiment_restore_manifest_invalid')
            parent, name = files.parent(stage_path / path)
            if kind == 'directory':
                fd = _new_directory(files, parent, name)
                directory_modes[relative] = original
                files.close(fd)
                continue
            temporary = '.target-version-' + secrets.token_hex(16) + '.tmp'
            fd = files.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, parent=parent)
            initial = files.proof(fd)
            written, content = 0, hashlib.sha256()
            try:
                while written < original[4]:
                    controller.check('source')
                    block = exact(min(archive.QUANTUM, original[4] - written))
                    content.update(block)
                    offset = 0
                    while offset < len(block):
                        files.location(parent)
                        _require(files.proof(fd) == initial and os.fstat(fd).st_size == written,
                                 'experiment_restore_stage_changed')
                        count = os.write(fd, memoryview(block)[offset:])
                        _require(type(count) is int and 0 < count <= len(block) - offset, 'experiment_restore_write_failed')
                        written += count
                        offset += count
                        controller.check('source')
                _require('sha256:' + content.hexdigest() == sha, 'experiment_restore_member_digest')
                files.location(fd)
                os.fsync(fd)
                _owner_mode(files, fd, original[1], original[2], stat.S_IMODE(original[0]))
                files.location(parent)
                try:
                    os.stat(name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise OwnerTargetVersionError('experiment_restore_stage_exists')
                files.location(fd)
                os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
                files.proof(fd)
                _require(owners._metadata(os.fstat(fd)) == owners._metadata(os.stat(name, dir_fd=parent, follow_symlinks=False))
                         == owners._metadata(os.stat(temporary, dir_fd=parent, follow_symlinks=False)),
                         'experiment_restore_stage_changed')
                files.location(parent)
                os.unlink(temporary, dir_fd=parent)
                files.location(parent)
                os.fsync(parent)
                info = os.fstat(fd)
                _require(files.proof(fd) == initial and info.st_nlink == 1 and info.st_size == original[4],
                         'experiment_restore_stage_changed')
                staged.append((relative, info, sha))
            finally:
                files.close(fd)
        _require(not buffer and read_count == preserved['size_bytes'] and response.read(1) == b''
                 and 'sha256:' + digest.hexdigest() == preserved['sha256'], 'experiment_restore_archive_digest')
        return stage, stage_name, staged, directory_modes
    finally:
        if response is not None:
            response.close()
        guarded.close()


def _activated(files, config, gid, action, expected, public, current, entry, issued, pins_root):
    """Reconcile durable current activation; no transfer or lease is repeated."""
    from .control_plane_lane_experiment_consumer import _restoration
    _require(entry['operation_id'] == action['action_id'] and current[1]['policy'] == action['policy']
             and all(entry[key] == action[key] for key in ('intent_id', 'generation', 'birth', 'owner', 'target_identity'))
             and entry['expires_at_epoch'] == action['new_lease_expires_at_epoch'], 'experiment_restore_current_changed')
    target, _ = actions._target(files, config, entry)
    actions._lease(files, target, entry)
    actions._birth(files, public, entry, gid)
    _restoration(files, public, entry, gid, issued)
    actions._pin_fence(files, config, pins_root, target, issued)
    public = birth._authority_lock(files, config.experiment_authority_root, gid)
    refreshed = _current(files, public, gid)
    _require(refreshed[0] == current[0], 'experiment_restore_current_changed')
    store = issuance._store(files, config.experiment_record_store)
    operations = actions._directory(files, store, 'operations')
    operation = actions._directory(files, operations, action['action_id'])
    files.proof(operation)
    fcntl.flock(operation, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files._operation_path = str(Path(config.experiment_record_store) / 'operations' / action['action_id'])
    previous, events = None, []
    for index in range(4104):
        value = recovery._read_event(files, operation, action, index, previous)
        if value is None:
            break
        event, selected = value
        events.append((event, selected))
        previous = selected
    else:
        raise OwnerTargetVersionError('experiment_restore_event_limit')
    _require(len(events) >= 5 and events[0][0]['event_kind'] == 'restore_started'
             and events[0][0]['body']['restore_intent'] == expected, 'experiment_restore_operation_invalid')
    tail = 1 if events[-1][0]['event_kind'] == 'activation_complete' else 0
    ready, correspondence, restored = events[-3-tail:len(events)-tail]
    _require([value[0]['event_kind'] for value in (ready, correspondence, restored)]
             == ['restored_payload_ready', 'restoration_correspondence', 'restored'],
             'experiment_restore_operation_invalid')
    _require(set(ready[0]['body']) == {'restore_started', 'restored_manifest', 'target_identity', 'new_lease'}
             and ready[0]['body']['restore_started'] == events[0][1]
             and ready[0]['body']['target_identity'] == entry['target_identity']
             and ready[0]['body']['new_lease'] == entry['lease']
             and set(correspondence[0]['body']) == {'restored_payload_ready', 'public_certificate', 'restore_intent', 'policy', 'principal'}
             and correspondence[0]['body'] == dict(restored_payload_ready=ready[1], public_certificate=entry['restoration'],
                restore_intent=expected, policy=action['policy'], principal=action['principal'])
             and set(restored[0]['body']) == {'restore_started', 'restored_payload_ready', 'public_certificate',
                                            'restoration_correspondence', 'prepared_authority'},
             'experiment_restore_operation_invalid')
    for index, (event, _) in enumerate(events[1:-3-tail]):
        body = event['body']
        _require(event['event_kind'] == 'restore_member'
                 and set(body) == {'restore_started', 'index', 'path', 'sha256', 'size_bytes'}
                 and body['restore_started'] == events[0][1] and body['index'] == index,
                 'experiment_restore_operation_invalid')
    manifest_selector = ready[0]['body']['restored_manifest']
    raw, _ = _document(files, Path(config.experiment_record_store) / (action['action_id'] + '.manifest.json'),
                       1048576, selector=manifest_selector)
    measured = actions._manifest(files, target, files.parents[target])
    _require(raw == actions._encoded(measured, 'manifest_digest', 1048576), 'experiment_restore_payload_changed')
    prepared, _ = _document(files, Path(config.experiment_record_store) / (action['action_id'] + '.restored-head.json'),
                            4096, selector=restored[0]['body']['prepared_authority'])
    _require(prepared == (json.dumps(current[0], sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()
             and restored[0]['body']['restore_started'] == events[0][1]
             and restored[0]['body']['restored_payload_ready'] == ready[1]
             and restored[0]['body']['public_certificate'] == entry['restoration']
             and restored[0]['body']['restoration_correspondence'] == correspondence[1],
             'experiment_restore_operation_invalid')
    head_selector = issuance._selector(prepared, files.budget)
    body = dict(restored=restored[1], active_head=head_selector)
    if tail:
        _require(events[-1][0]['body'] == body, 'experiment_restore_operation_invalid')
        receipt = events[-1][1]
    else:
        files.verify()
        receipt = actions._event(files, operation, action, 'activation_complete', body, len(events), restored[1], issued)
    return dict(action_id=action['action_id'], intent_id=entry['intent_id'], decision='restored', receipt=receipt,
                generation=entry['generation'], removed_logical_bytes=0, removed_allocated_bytes=0)


def restore(action_id, *, expected_restore_intent, installed_config_path, now, pins_root):
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    reservation = None
    try:
        issued = now()
        config, gid = actions._context(files, installed_config_path, issued)
        _require(config.experiment_retirement_enabled is True, 'experiment_retirement_disabled')
        action = _restore_intent(files, config, action_id, expected_restore_intent, issued)
        public, current, entry = actions._selected(files, config, action['intent_id'], issued, gid)
        if entry['state'] == 'active' and entry['restoration'] is not None:
            return _activated(files, config, gid, action, expected_restore_intent, public, current, entry, issued, pins_root)
        _require(entry['state'] == 'retired' and entry['operation_id'] == action_id and current[1]['policy'] == action['policy']
                 and all(entry[key] == action[key] for key in ('generation', 'birth', 'lease', 'target_identity', 'owner')),
                 'experiment_restore_current_changed')
        target, target_fd = actions._target(files, config, entry)
        lease, lease_record = actions._lease(files, target, entry)
        _require(issued >= lease['expires_at_epoch'] and lease['released_at_epoch'] is None, 'experiment_restore_lease_changed')
        origin = actions._birth(files, public, entry, gid)
        reference, reference_fd = actions._pin_fence(files, config, pins_root, target, issued)
        _, selection = _document(files, Path(config.experiment_record_store) / (action_id + '.restore-selection.json'), 32768)
        _require(set(selection) == {'schema_version', 'restore_intent', 'original_operation_id', 'original_action',
                 'manifest', 'retired', 'preservation', 'selection_digest'}
                 and selection['schema_version'] == 'control_plane_lane_experiment_restore_selection.v1'
                 and selection['selection_digest'] == canonical_digest(selection, digest_field='selection_digest')
                 and selection['restore_intent'] == expected_restore_intent and selection['preservation'] == action['preservation']
                 and owners._matches(selection['original_operation_id'], owners._CONSENT_ID), 'experiment_restore_selection_invalid')
        store_path = Path(config.experiment_record_store)
        prior_raw, prior = _document(files, store_path / (selection['original_operation_id'] + '.action.json'), 32768,
                                     selector=selection['original_action'])
        _require(prior['action'] == 'offload' and all(prior[key] == entry[key] for key in ('intent_id', 'owner', 'generation', 'birth', 'lease', 'target_identity', 'completion')),
                 'experiment_restore_selection_invalid')
        manifest_raw, manifest = _document(files, store_path / (selection['original_operation_id'] + '.manifest.json'), 1048576,
                                            selector=selection['manifest'])
        rows = sorted(manifest['rows'], key=lambda row: (len(Path(row[0]).parts), row[0]), reverse=True)
        public = birth._authority_lock(files, config.experiment_authority_root, gid)
        refreshed = _current(files, public, gid)
        _require(refreshed[0] == current[0], 'experiment_restore_current_changed')
        store = issuance._store(files, config.experiment_record_store)
        old_entry = entry | {'operation_id': selection['original_operation_id']}
        final, _, _ = recovery.retired(files, config, prior, issuance._selector(prior_raw, files.budget),
                                      target, rows, old_entry, origin['marker'], _retained_store=store)
        _require(final == selection['retired'], 'experiment_restore_selection_invalid')
        # The exact old operation/store lock and pins EX remain held.
        occupied = issuance._capacity(files, store, adding_registration=False)
        reserved = 2 * len(rows) * 4096 + 8 * 32768
        _require(occupied + reserved + 32768 <= issuance.MAX_EXPERIMENT_STORE_BYTES, 'experiment_store_full')
        reservation_raw = actions._encoded(dict(schema_version='control_plane_lane_experiment_reservation.v1',
            operation_id=action_id, reserved_bytes=reserved), 'reservation_digest', 4096)
        _publish(files, store, action_id + '.reservation.json', reservation_raw, kind='private')
        operations = actions._directory(files, store, 'operations')
        operation = actions._directory(files, operations, action_id, create=True)
        stage_name = '.restore-' + action_id
        started = actions._event(files, operation, action, 'restore_started', dict(restore_intent=expected_restore_intent,
            preservation=action['preservation'], retired=selection['retired'], stage_name=stage_name,
            new_lease_expiry=action['new_lease_expires_at_epoch'], reference_authority=reference), 0, None, issued)
        prepared, old_head = actions._version(files, public, refreshed, entry | {'state': 'restoring'},
                                              gid, action['policy'], issued)
        _publish(files, store, action_id + '.restore-head.json', prepared, kind='private')
        actions._install_head(files, public, prepared, gid, old_head)
        restoring = _current(files, public, gid)
        def guard():
            _require(now() < action['expires_at_epoch'], 'experiment_restore_expired')
            files.verify()
            files.location(target_fd)
            files.location(reference_fd)
            files.verify_record(restoring[2])
        need = manifest['logical_bytes'] + len(rows) * 8192 + 8192
        reservation = reserve_control_plane_disk('experiment_restore', target_root=target, expected_bytes=need,
            minimum_bytes=need, workspace=target / stage_name, fresh=True, evictor=None)
        stage, stage_name, staged, directory_modes = _stage_archive(files, config, target, target_fd,
            action, selection, manifest_raw, rows, guard)
        previous, index = started, 1
        for relative, before in sorted(directory_modes.items(), key=lambda item: (len(Path(item[0]).parts), item[0])):
            guard()
            parent, name = files.parent(target / relative)
            fd = _new_directory(files, parent, name)
            _owner_mode(files, fd, before[1], before[2], stat.S_IMODE(before[0]))
            files.close(fd)
        for relative, before, sha in sorted(staged, key=lambda item: item[0]):
            guard()
            source_parent, source_name = files.parent(target / stage_name / relative)
            destination_parent, name = files.parent(target / relative)
            fd = files.open(source_name, os.O_RDONLY | os.O_NONBLOCK, parent=source_parent)
            try:
                _require(owners._metadata(os.fstat(fd)) == owners._metadata(before), 'experiment_restore_stage_changed')
                files.location(destination_parent)
                try:
                    os.stat(name, dir_fd=destination_parent, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise OwnerTargetVersionError('experiment_restore_destination_exists')
                files.location(fd)
                os.link(source_name, name, src_dir_fd=source_parent, dst_dir_fd=destination_parent, follow_symlinks=False)
                files.proof(fd)
                _require(owners._metadata(os.fstat(fd)) == owners._metadata(os.stat(name, dir_fd=destination_parent, follow_symlinks=False)),
                         'experiment_restore_stage_changed')
                files.location(destination_parent)
                os.fsync(destination_parent)
                previous = actions._event(files, operation, action, 'restore_member', dict(restore_started=started,
                    index=index - 1, path=relative, sha256=sha, size_bytes=before.st_size), index, previous, issued)
                index += 1
                files.location(source_parent)
                files.proof(fd)
                _require(owners._metadata(os.stat(source_name, dir_fd=source_parent, follow_symlinks=False)) == owners._metadata(os.fstat(fd)),
                         'experiment_restore_stage_changed')
                os.unlink(source_name, dir_fd=source_parent)
                files.location(source_parent)
                os.fsync(source_parent)
            finally:
                files.close(fd)
        for relative in sorted(directory_modes, key=lambda value: (len(Path(value).parts), value), reverse=True):
            parent, name = files.parent(target / stage_name / relative)
            child = files.open(name, os.O_RDONLY | os.O_DIRECTORY, parent=parent)
            files.location(child)
            files.location(parent)
            os.rmdir(name, dir_fd=parent)
            files.close(child)
            files.location(parent)
            os.fsync(parent)
        files.location(stage)
        files.location(target_fd)
        os.rmdir(stage_name, dir_fd=target_fd)
        files.close(stage)
        files.location(target_fd)
        os.fsync(target_fd)
        new_lease = lease | {'renewed_at_epoch': issued, 'expires_at_epoch': action['new_lease_expires_at_epoch']}
        payload = actions._encoded(new_lease, 'lease_digest', scratch.MAX_LEASE_BYTES)
        _require(scratch._lease_fields_valid(json.loads(payload)), 'experiment_restore_lease_invalid')
        new_selector = _lease_cas(files, target, target_fd, lease_record, payload)
        # Root measured publication bytes bind restored identities, not a new execution.
        restored_manifest = actions._manifest(files, target, target_fd)
        restored_raw = actions._encoded(restored_manifest, 'manifest_digest', 1048576)
        manifest_selector = _publish(files, store, action_id + '.manifest.json', restored_raw, kind='manifest')
        ready = actions._event(files, operation, action, 'restored_payload_ready', dict(restore_started=started,
            restored_manifest=manifest_selector, target_identity=entry['target_identity'], new_lease=new_selector),
            index, previous, issued)
        index += 1
        certificate = dict(schema_version='control_plane_lane_experiment_restoration_certificate.v1',
            restoration_id=action_id, intent_id=entry['intent_id'], generation=entry['generation'], birth=entry['birth'],
            target_identity=entry['target_identity'], new_lease=new_selector, manifest=manifest_selector,
            restored_at_epoch=issued, lease_expires_at_epoch=action['new_lease_expires_at_epoch'])
        certificate_raw = actions._encoded(certificate, 'certificate_digest', 8192)
        certificate_name = 'restoration-' + hashlib.sha256(certificate_raw).hexdigest() + '.json'
        certificate_ref = _publish(files, public, certificate_name, certificate_raw, kind='certificate', blueprint_gid=gid)
        correspondence = actions._event(files, operation, action, 'restoration_correspondence', dict(restored_payload_ready=ready,
            public_certificate=certificate_ref, restore_intent=expected_restore_intent, policy=action['policy'], principal=action['principal']),
            index, ready, issued)
        index += 1
        active_entry = entry | {'state': 'active', 'lease': new_selector, 'restoration': certificate_ref,
                                'expires_at_epoch': action['new_lease_expires_at_epoch']}
        prepared, old_head = actions._version(files, public, restoring, active_entry, gid, action['policy'], issued)
        prepared_ref = _publish(files, store, action_id + '.restored-head.json', prepared, kind='private')
        restored = actions._event(files, operation, action, 'restored', dict(restore_started=started,
            restored_payload_ready=ready, public_certificate=certificate_ref, restoration_correspondence=correspondence,
            prepared_authority=prepared_ref), index, correspondence, issued)
        index += 1
        files.verify()
        active_head = actions._install_head(files, public, prepared, gid, old_head)
        receipt = actions._event(files, operation, action, 'activation_complete', dict(restored=restored,
            active_head=active_head), index, restored, issued)
        return dict(action_id=action_id, intent_id=entry['intent_id'], decision='restored', receipt=receipt,
                    generation=entry['generation'], removed_logical_bytes=0, removed_allocated_bytes=0)
    except OSError:
        raise OwnerTargetVersionError('experiment_restore_io_failed') from None
    finally:
        try:
            files.finish()
        finally:
            try:
                if reservation is not None:
                    reservation.release(outcome='completed' if 'receipt' in locals() else 'failed')
            finally:
                files.budget.close()
