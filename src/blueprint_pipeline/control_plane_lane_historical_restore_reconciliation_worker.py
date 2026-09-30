"""Discard only an exact separately approved unfinished original restore row.

Preserve every known birth and all original records. Current DELETE/restore
authority, original clocks, full readback, reservation, native rights and live
readers are rechecked. An absent row is uncertainty, never freed-byte credit.
"""
from __future__ import annotations

import hashlib
import os

from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_fence import _version
from .control_plane_lane_historical_restore_reconciliation_authority import decision_id
from .control_plane_lane_historical_restore_reconciliation_scope import reconciled_parent, unknown_creation_scope


def _remove(held, worker, scope):
    relative, parent_path = scope['remove_member']['path'], scope['parent_path']
    row = scope['remove_member']
    with held._opened(relative) as (fd, guard):
        if row['kind'] == 'file':
            digest, size = hashlib.sha256(), 0
            while True:
                with worker.mutation_authority(readers=True):
                    guard()
                    block = os.read(fd, 1024**2)
                    guard()
                if not block:
                    break
                size += len(block)
                generation._require(size <= row['size_bytes'], 'restore_reconciliation_scope_invalid')
                digest.update(block)
            generation._require(size == row['size_bytes'] and 'sha256:' + digest.hexdigest() == row['sha256'],
                                'restore_reconciliation_scope_invalid')
        with held._opened(parent_path) as (parent, parent_guard):
            with worker.mutation_authority(readers=True):
                held.verify()
                guard()
                parent_guard()
                name = relative.rpartition('/')[2]
                if row['kind'] == 'directory':
                    generation._require(not held.children[relative], 'restore_reconciliation_scope_invalid')
                    os.rmdir(name, dir_fd=parent)
                else:
                    os.unlink(name, dir_fd=parent)
                held.removed.add(relative)
                held.children[parent_path].remove(name)
                after = _version(os.fstat(fd))
                generation._require(after[:5] == row['version'][:5] and after[5] == 0,
                                    'restore_reconciliation_scope_invalid')
                held.versions[relative] = after
                old, current = held.versions[parent_path], _version(os.fstat(parent))
                generation._require(current[:5] == old[:5]
                    and current[5] == old[5] - int(row['kind'] == 'directory')
                    and current[7] >= old[7] and current[8] >= old[8], 'restore_reconciliation_scope_invalid')
                held.versions[parent_path] = current
                if not parent_path:
                    ancestor, leaf, root, _ = held.chain[-1]
                    held.chain[-1] = (ancestor, leaf, root, current)
                guard()
                parent_guard()
                os.fsync(parent)
                guard()
                parent_guard()
    return held.versions[parent_path].copy()


def reconcile_unlogged_creation(worker, events, observed, roots, monotonic):
    from .control_plane_lane_historical_restore_worker import _readback, _resources
    pending_cleanup = events[-1]['kind'] == 'restore_intent' and events[-1]['body'].get('phase') == 'reconcile_intent'
    if pending_cleanup:
        generation._require(worker.reconciliations, 'restore_reconciliation_approval_invalid')
        prefix, identifier, selected, _ = worker.reconciliations[-1]
    else:
        prefix, selected = events, None
        # This proves exactly one unknown pending row before approval lookup;
        # other drift cannot be converted into a broader owner discard request.
        unknown_creation_scope(worker.selected[2], observed, worker.selected[1], prefix,
            worker.action_id, tick=worker.operation.remaining)
        identifier = decision_id(worker.action_id, prefix[-1]['event_digest'])
        generation._require(len(worker.reconciliations) < 8, 'restore_reconciliation_limit')
        worker.reconciliations.append((prefix, identifier, selected, None))
    try:
        with worker.checkpoint(journal=True):
            approval, approved, selected = worker.reconciliation_selected[-1]
    except FileNotFoundError:
        raise generation.HistoricalGenerationError('historical_generation_restore_reconciliation_approval_missing') from None
    worker.reconciliations[-1] = (prefix, identifier, selected, None)
    scope = approval['packet']['scope']
    present = observed == approved
    if not present:
        generation._require(pending_cleanup, 'restore_reconciliation_scope_invalid')
        reconciled_parent(approved, observed, scope, tick=worker.operation.remaining)
    _readback(worker, observed, roots, monotonic)
    binding = dict(decision_id=identifier, decision=selected,
                   original_head_event_digest=prefix[-1]['event_digest'])
    with _resources(worker, observed) as (held, _):
        if not pending_cleanup:
            worker.record('restore_intent', dict(phase='reconcile_intent', **binding))
        if present:
            parent = _remove(held, worker, scope)
        else:
            with worker.mutation_authority(readers=True):
                held.verify()
                held.sync_directory(scope['parent_path'])
            parent = held.versions[scope['parent_path']].copy()
        worker.record('restore_intent', dict(phase='reconciled', **binding,
            parent_path=scope['parent_path'], parent_version=parent, uncertain=True,
            credited_removed_allocated_bytes=0))
    # Fresh inventory and whole replay observe the effects; no original packet,
    # decision, birth, archive, journal prefix or clock is rewritten.
    current = generation.inventory_historical_generation(worker.selected[2]['target_path'], allowed_roots=roots,
        max_seconds=worker.operation.remaining(), monotonic=monotonic)
    return worker.replay(), current
