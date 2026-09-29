"""Finite durable metadata passes under one authenticated action lifetime.

No partial rows survive a retry. A reserved uncertain pass stays consumed; the
second and last pass starts fresh against the same retained target authority.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

from . import control_plane_lane_experiment_retirement as issuance
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_work import _ActionFiles
from .control_plane_lane_owner_target_versions import _require
from .decision_evidence_contracts import canonical_digest

_ROLES = frozenset({'issue', 'restore_stage', 'restore_final', 'restore_activation', 'restore_checkpoint_compare'})
_RESERVED = 'control_plane_lane_experiment_scan_reserved.v1'
_COMPLETED = 'control_plane_lane_experiment_scan_completed.v1'
_SCAN_RESERVATION = 20 * 4096


def _record(files, path):
    raw, record = files.read(path, cap=4096, protected=True, mode=0o600)
    value = retained._document(raw, 4096, _work_budget=files.budget)
    _require(type(value) is dict and value.get('work_digest') == canonical_digest(value, digest_field='work_digest'),
             'experiment_scan_record_invalid')
    selected = issuance._selector(raw, files.budget)
    files.verify_record(record)
    files.records.remove(record)
    files.close(record.fd)
    _require(record.fd not in files.owned, 'experiment_scan_cleanup_failed')
    return value, selected


def begin(files, config, store, operation_id, binding, *, role, operation=None):
    from . import control_plane_lane_experiment_actions as actions
    _require(type(files) is _ActionFiles and role in _ROLES
             and owners._matches(operation_id, owners._CONSENT_ID), 'experiment_scan_scope_invalid')
    files.phase('scan_admission_' + role)
    # Fixed conservative reservation for all ten possible reserved/completed
    # records. Directory accounting requires this durable proof before mkdir.
    name = operation_id + '.scan-reservation.json'
    value = dict(schema_version='control_plane_lane_experiment_reservation.v1', operation_id=operation_id,
                 reserved_bytes=_SCAN_RESERVATION)
    raw = actions._encoded(value, 'reservation_digest', 4096)
    files.location(store)
    try:
        os.stat(name, dir_fd=store, follow_symlinks=False)
    except FileNotFoundError:
        occupied = issuance._capacity(files, store, adding_registration=False)
        _require(occupied + _SCAN_RESERVATION + len(raw) <= issuance.MAX_EXPERIMENT_STORE_BYTES,
                 'experiment_store_full')
        actions._publish(files, store, name, raw, kind='private')
    else:
        actual, record = files.read(Path(config.experiment_record_store) / name, cap=4096, protected=True, mode=0o600)
        _require(actual == raw, 'experiment_scan_record_invalid')
        files.verify_record(record)
    if operation is None:
        files.location(store)
        try:
            os.stat('operations', dir_fd=store, follow_symlinks=False)
        except FileNotFoundError:
            operations = actions._directory(files, store, 'operations', create=True)
        else:
            operations = actions._directory(files, store, 'operations')
        try:
            os.stat(operation_id, dir_fd=operations, follow_symlinks=False)
        except FileNotFoundError:
            operation = actions._directory(files, operations, operation_id, create=True)
        else:
            operation = actions._directory(files, operations, operation_id)
    else:
        files.proof(operation)
        _require(files.bindings[operation][1] == operation_id
                 and owners._metadata(os.fstat(operation)) == owners._metadata(os.stat(
                     Path(config.experiment_record_store) / 'operations' / operation_id,
                     follow_symlinks=False)), 'experiment_scan_scope_invalid')
    files.location(operation)
    files.proof(operation)
    fcntl.flock(operation, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = Path(config.experiment_record_store) / 'operations' / operation_id
    expected = dict(schema_version=_RESERVED, operation_id=operation_id, intent_id=binding['intent_id'],
        generation=binding['generation'], **{key: binding[key] for key in ('birth', 'target_identity', 'lease')},
        role=role, reserved_member_operations=4096, reserved_batches=256)
    previous = None
    for index in range(2):
        name = f'scan-{role}-{index}-reserved.json'
        files.location(operation)
        try:
            os.stat(name, dir_fd=operation, follow_symlinks=False)
        except FileNotFoundError:
            reserved = expected | dict(**{'pass': index}, controller=files.controller(),
                controller_origin_epoch=files.controller_epoch, deadline_epoch=files.deadline_epoch,
                aggregate_before=dict(files.conserved), previous_reservation=previous)
            selected = actions._publish(files, operation, name,
                                       actions._encoded(reserved, 'work_digest', 4096), kind='scan_work')
            files._scan_scope = dict(role=role, operation=operation, path=path, index=index,
                                     reservation=selected, member_operations=0, batches=0)
            return
        value, selected = _record(files, path / name)
        _require(set(value) == set(expected) | {'pass', 'controller', 'controller_origin_epoch',
            'deadline_epoch', 'aggregate_before', 'previous_reservation', 'work_digest'}
            and all(value[key] == item for key, item in expected.items()) and value['pass'] == index
            and value['previous_reservation'] == previous, 'experiment_scan_record_invalid')
        # Original invocation controller was selected independently from the
        # current action/issue locator. Stored phase controller must agree.
        files.validate_controller(value['controller'])
        counts = value['aggregate_before']
        _require(type(counts) is dict and set(counts) == {'raw_bytes', 'output_bytes'}
                 and all(type(amount) is int and 0 <= amount <= 20*1024*1024 for amount in counts.values()),
                 'experiment_scan_record_invalid')
        # Uncertain traversal reserves a full bounded manifest output even if
        # killed before durable completion; it cannot replay those bytes free.
        for key in counts:
            files.conserved[key] = max(files.conserved[key], counts[key] + (1048576 if key == 'output_bytes' else 0))
        files.location(operation)
        completed_name = f'scan-{role}-{index}-completed.json'
        try:
            os.stat(completed_name, dir_fd=operation, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            completion, _ = _record(files, path / completed_name)
            _require(set(completion) == {'schema_version', 'reservation', 'manifest', 'member_count',
                'batches_used', 'aggregate_after', 'controller', 'completed_at_epoch', 'work_digest'}
                and completion['schema_version'] == _COMPLETED and completion['reservation'] == selected
                and type(completion['member_count']) is int and 0 <= completion['member_count'] <= 4096
                and type(completion['batches_used']) is int
                and completion['batches_used'] == (completion['member_count'] + 15) // 16,
                'experiment_scan_record_invalid')
            files.validate_controller(completion['controller'])
            files.bind_aggregate(completion['aggregate_after'])
        files.check_long()
        previous = selected
    _require(False, 'experiment_scan_pass_exhausted')


def member(files):
    scope = files._scan_scope
    _require(scope['member_operations'] < 4096, 'experiment_manifest_limit')
    if scope['member_operations'] % 16 == 0:
        _require(scope['batches'] < 256, 'experiment_scan_batch_limit')
        files.phase('scan_batch_' + scope['role'])
        scope['batches'] += 1
    scope['member_operations'] += 1
    files.check_long()


def finish_metadata(files):
    files.phase('scan_proof_' + files._scan_scope['role'])


def completed(files, manifest, member_count):
    from . import control_plane_lane_experiment_actions as actions
    scope = files._scan_scope
    _require(type(member_count) is int and member_count == scope['member_operations'], 'experiment_scan_record_invalid')
    counts = {key: files.conserved[key] + files.budget.counts[key] for key in files.conserved}
    _require(all(amount <= 20*1024*1024 for amount in counts.values()), 'experiment_work_aggregate_limit')
    value = dict(schema_version=_COMPLETED, reservation=scope['reservation'], manifest=manifest,
                 member_count=member_count, batches_used=scope['batches'], aggregate_after=counts,
                 controller=files.controller(), completed_at_epoch=files.now())
    actions._publish(files, scope['operation'], f"scan-{scope['role']}-{scope['index']}-completed.json",
                     actions._encoded(value, 'work_digest', 4096), kind='scan_work')
