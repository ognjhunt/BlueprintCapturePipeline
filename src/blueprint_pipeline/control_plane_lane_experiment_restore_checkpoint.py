"""Root-sealed published payload and original lease CAS recovery.

A partial lease grants no reader permission. Only the existing restoring action,
original target/operation EX, sealed payload inventory and original lease inode
allow the fixed root finalizer to continue its own prepared write.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

from . import control_plane_lane_experiment_actions as actions
from . import control_plane_lane_experiment_acquisition as acquisition
from . import control_plane_lane_experiment_recovery as recovery
from . import control_plane_lane_experiment_retirement as issuance
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch as scratch
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_owner_target_versions import _require
from .decision_evidence_contracts import canonical_digest

_SCHEMA = 'control_plane_lane_experiment_lease_transition.v1'
_FIELDS = {'schema_version', 'action_id', 'restore_intent', 'generation', 'birth', 'target_identity',
           'old_lease', 'new_lease', 'lease_identity', 'payload_manifest', 'target_metadata',
           'restore_started', 'previous', 'sequence', 'transition_digest'}


def read(files, config, action, expected, target, entry):
    """Select a fixed private checkpoint; unknown bytes/inodes remain refused."""
    path = Path(config.experiment_record_store) / (action['action_id'] + '.lease-transition.json')
    parent, name = files.parent(path, protected=True)
    files.location(parent)
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None
    raw, _ = files.read(path, cap=32768, protected=True, mode=0o600)
    value = retained._document(raw, 32768, _work_budget=files.budget)
    _require(entry['state'] == 'restoring' and set(value) == _FIELDS and value['schema_version'] == _SCHEMA
             and value['transition_digest'] == canonical_digest(value, digest_field='transition_digest')
             and value['action_id'] == action['action_id'] and value['restore_intent'] == expected
             and all(value[key] == entry[key] for key in ('generation', 'birth', 'target_identity'))
             and type(value['sequence']) is int and 2 <= value['sequence'] <= 4098
             and type(value['lease_identity']) is list and len(value['lease_identity']) == 9
             and all(type(item) is int and item >= 0 for item in value['lease_identity'])
             and type(value['target_metadata']) is list and len(value['target_metadata']) == 7
             and all(type(item) is int and item >= 0 for item in value['target_metadata']),
             'experiment_restore_transition_invalid')
    old, new = value['old_lease'], value['new_lease']
    _require(type(old) is str and type(new) is str and 0 < len(old.encode()) <= scratch.MAX_LEASE_BYTES
             and 0 < len(new.encode()) <= scratch.MAX_LEASE_BYTES
             and issuance._selector(old.encode(), files.budget) == entry['lease'],
             'experiment_restore_transition_invalid')
    original = retained._document(old.encode(), scratch.MAX_LEASE_BYTES, _work_budget=files.budget)
    replacement = retained._document(new.encode(), scratch.MAX_LEASE_BYTES, _work_budget=files.budget)
    _require(scratch._lease_fields_valid(original) and scratch._lease_fields_valid(replacement)
             and original['lease_digest'] == canonical_digest(original, digest_field='lease_digest')
             and replacement['lease_digest'] == canonical_digest(replacement, digest_field='lease_digest')
             and all(original[key] == entry[key] for key in ('owner', 'lane', 'name'))
             and replacement == original | {'renewed_at_epoch': replacement['renewed_at_epoch'],
                 'expires_at_epoch': action['new_lease_expires_at_epoch'], 'lease_digest':replacement['lease_digest']},
             'experiment_restore_transition_invalid')
    actual, record = files.read(target / scratch.LEASE_FILE, cap=scratch.MAX_LEASE_BYTES)
    metadata = owners._metadata(record.info)
    before = value['lease_identity']
    _require(stat.S_ISREG(record.info.st_mode) and tuple(before[:6]) == metadata[:6]
             and before[5] == 1 and before[6] == len(old.encode())
             and (actual == old.encode() or new.encode().startswith(actual)),
             'experiment_restore_lease_changed')
    return value, original, record


def published_union(files, config, action, entry, checkpoint, target, target_fd, operation, initial):
    value = checkpoint[0]
    _require(value['restore_started'] == initial[1], 'experiment_restore_transition_invalid')
    body = initial[0]['body']
    _require(body['controller_origin_epoch'] <= files.now() < body['deadline_epoch']
             <= min(body['controller_origin_epoch'] + 4*3600, action['expires_at_epoch']),
             'experiment_restore_operation_invalid')
    files.validate_controller(body['controller'])
    files.bind_deadline(body['deadline_epoch'])
    previous = initial[1]
    for index in range(1, value['sequence']):
        if (index - 1) % 16 == 0:
            files.phase('restore_recovery_batch')
        event = recovery._read_event(files, operation, action, index, previous)
        _require(event is not None and event[0]['event_kind'] in ('restore_stage_ready', 'restore_directory', 'restore_member'),
                 'experiment_restore_transition_invalid')
        previous = event[1]
    _require(previous == value['previous'] and recovery._read_event(files, operation, action, value['sequence'], previous) is None,
             'experiment_restore_transition_invalid')
    files.phase('restore_checkpoint_manifest')
    raw, _ = files.read(Path(config.experiment_record_store) / (action['action_id'] + '.payload-manifest.json'),
                        cap=1048576, protected=True, mode=0o600)
    _require(issuance._selector(raw, files.budget) == value['payload_manifest'], 'experiment_restore_transition_invalid')
    saved = actions._manifest_record(files, raw, entry)
    store = files.bindings[files.bindings[operation][0]][0]
    acquisition.begin(files, config, store, action['action_id'], entry, role='restore_checkpoint_compare', operation=operation)
    current = actions._manifest(files, target, target_fd, binding=entry, hash_payload=False)
    _require(len(current['members']) == len(saved['members']) and all(
        current_row[:4] == saved_row[:4] for current_row, saved_row in zip(current['members'], saved['members'], strict=True))
        and [getattr(os.fstat(target_fd), key) for key in recovery._STAT] == value['target_metadata'],
        'experiment_restore_published_payload_changed')
    actions._hash_manifest(files, target, target_fd, current, role='restore_checkpoint_validate')
    _require(current == {key:item for key,item in saved.items() if key != 'manifest_digest'},
             'experiment_restore_published_payload_changed')
    mapped = {}
    for path, kind, identity, token, _ in saved['members']:
        ids, data = identity.split(':'), tuple(map(int, token.split(':')))
        mapped[path] = (int(ids[0]), int(ids[1]), (stat.S_IFDIR if kind == 'directory' else stat.S_IFREG) | data[0], *data[1:])
    files.phase('restore_union_finalize')
    acquisition.completed(files, value['payload_manifest'], len(current['members']))
    return dict(stage=None, staged=[], directory_modes={}, mapped=mapped, created={}, linked={},
                previous=previous, index=value['sequence'], target_transition=value['target_metadata'])


def prepare(files, store, target, target_fd, action, expected, entry, lease_record, payload, started, previous, index, *, config, operation):
    acquisition.begin(files, config, store, action['action_id'], entry, role='restore_checkpoint_compare', operation=operation)
    manifest = actions._manifest(files, target, target_fd, binding=entry, hash_payload=False)
    actions._hash_manifest(files, target, target_fd, manifest, role='restore_checkpoint_validate')
    files.phase('restore_checkpoint_manifest')
    selected = actions._publish(files, store, action['action_id'] + '.payload-manifest.json',
                                actions._encoded(manifest, 'manifest_digest', 1048576), kind='manifest')
    acquisition.completed(files, selected, len(manifest['members']))
    files.phase('restore_checkpoint_record')
    raw, original_record = files.read(target / scratch.LEASE_FILE, cap=scratch.MAX_LEASE_BYTES)
    _require(issuance._selector(raw, files.budget) == entry['lease']
             and owners._metadata(original_record.info) == owners._metadata(lease_record.info),
             'experiment_restore_lease_changed')
    value = dict(schema_version=_SCHEMA, action_id=action['action_id'], restore_intent=expected,
        generation=entry['generation'], birth=entry['birth'], target_identity=entry['target_identity'],
        old_lease=raw.decode(), new_lease=payload.decode(), lease_identity=list(owners._metadata(lease_record.info)),
        payload_manifest=selected, target_metadata=[getattr(os.fstat(target_fd), key) for key in recovery._STAT],
        restore_started=started, previous=previous, sequence=index)
    actions._publish(files, store, action['action_id'] + '.lease-transition.json',
                     actions._encoded(value, 'transition_digest', 32768), kind='private')
