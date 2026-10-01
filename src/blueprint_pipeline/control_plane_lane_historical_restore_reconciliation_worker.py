"""Discard only an exact separately approved unfinished original restore row.

Preserve every known birth and all original records. Current DELETE/restore
authority, original clocks, full readback, reservation, native rights and live
readers are rechecked. An absent row is uncertainty, never freed-byte credit.
"""
from __future__ import annotations

from .control_plane_lane_historical_restore_limits import selected_restore_bounds

import hashlib
import os

from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_fence import _version
from .control_plane_lane_historical_restore_reconciliation_authority import decision_id
from .control_plane_lane_historical_restore_reconciliation_scope import reconciled_parent, unknown_creation_scope


def _effect_pin(worker, selected):
    """The held syscall cannot silently switch to a newer owner grant."""
    effect = worker.effect_selected[-1]
    generation._require(selected == dict(decision_id=effect[0]['decision_id'], decision=effect[2]),
                        'restore_reconciliation_approval_invalid')
    generation._require(worker.operation.moment() < effect[0]['expires_at_epoch']
        <= worker.selected[1]['expires_at_epoch'], 'restore_reconciliation_approval_invalid')
    extra = worker.resume_selected[-1]
    if extra is not None:
        recorded = worker.reconciliations[-1][3]
        generation._require(extra[0]['action'] == 'discard_unfinished_restore_row'
            and recorded and recorded[-1]['body'].get('phase') == 'reconcile_delete_resume'
            and recorded[-1]['body'].get('delete_resume') == selected,
            'restore_reconciliation_approval_invalid')


def _resume_effect_scope(present, pending, recorded, selected):
    """Absence cannot acquire a new DELETE intent; exact prior effect may finish."""
    generation._require(pending and (present or recorded
        and recorded[-1]['body'].get('phase') == 'reconcile_delete_resume'
        and recorded[-1]['body'].get('delete_resume') == selected), 'restore_reconciliation_scope_invalid')


def _sync_absence(held, worker, relative, grant, *, observe_only):
    """Sync only the guarded parent under the exact current selected grant."""
    generation._require(relative in held.rows and held.rows[relative]['kind'] == 'directory', 'fence_changed')
    with held._opened(relative) as (fd, guard):
        guard()
        current = worker.resume_selected[-1] if observe_only else worker.effect_selected[-1]
        generation._require(current is not None
            and current[0]['decision_id'] == grant[0]['decision_id'] and current[2] == grant[2]
            and current[0]['action'] == ('observe_unfinished_restore_row_absence' if observe_only
                else 'discard_unfinished_restore_row')
            and worker.operation.moment() < current[0]['expires_at_epoch']
                <= worker.selected[1]['expires_at_epoch'], 'restore_reconciliation_approval_invalid')
        os.fsync(fd)
        guard()


def _remove(held, worker, scope, selected):
    relative, parent_path = scope['remove_member']['path'], scope['parent_path']
    row = scope['remove_member']
    with held._opened(relative) as (fd, guard):
        if row['kind'] == 'file':
            digest, size = hashlib.sha256(), 0
            while True:
                with worker.mutation_authority(readers=True):
                    guard()
                    _effect_pin(worker, selected)
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
                    _effect_pin(worker, selected)
                    os.rmdir(name, dir_fd=parent)
                else:
                    _effect_pin(worker, selected)
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
                _effect_pin(worker, selected)
                os.fsync(parent)
                guard()
                parent_guard()
    return held.versions[parent_path].copy()


def reconcile_unlogged_creation(worker, events, observed, roots, monotonic):
    from .control_plane_lane_historical_restore_worker import _readback, _resources
    pending_cleanup = events[-1]['kind'] == 'restore_intent' and events[-1]['body'].get('phase') in ('reconcile_intent', 'reconcile_delete_resume')
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
    with worker.checkpoint(journal=True):
        approval, approved, selected = worker.reconciliation_selected[-1]
        observation = worker.resume_selected[-1]
    if not pending_cleanup:
        identifier = approval['decision_id']
    recorded = worker.reconciliations[-1][3]
    worker.reconciliations[-1] = (prefix, identifier, selected, recorded)
    observe_only = observation is not None and observation[0]['action'] == 'observe_unfinished_restore_row_absence'
    effect = worker.effect_selected[-1]
    scope = effect[0]['packet']['scope']
    approved = effect[1]
    present = observed == approved
    if not present:
        generation._require(pending_cleanup, 'restore_reconciliation_scope_invalid')
        reconciled_parent(approved, observed, scope, tick=worker.operation.remaining,
            restore_bounds=selected_restore_bounds(worker))
    if observe_only:
        # Observation approval can never reach removal, even if old DELETE is
        # still current. The whole current absence observation must be exact.
        generation._require(not present and pending_cleanup and observed == observation[1],
                            'restore_absent_observation_approval_invalid')
    if observation is not None and not observe_only:
        _resume_effect_scope(present, pending_cleanup, recorded,
            dict(decision_id=observation[0]['decision_id'], decision=observation[2]))
    from .control_plane_lane_historical_restore_metadata import preflight_restore_metadata
    preflight_restore_metadata(worker)
    _readback(worker, observed, roots, monotonic)
    binding = dict(decision_id=identifier, decision=selected,
                   original_head_event_digest=prefix[-1]['event_digest'])
    with _resources(worker, observed) as (held, _):
        if not pending_cleanup:
            intent = worker.record('restore_intent', dict(phase='reconcile_intent', **binding))
            recorded = (intent,)
            worker.reconciliations[-1] = (prefix, identifier, selected, recorded)
        if observation is not None and not observe_only:
            grant = dict(decision_id=observation[0]['decision_id'], decision=observation[2])
            if recorded[-1]['body'].get('delete_resume') != grant:
                resumed = worker.record('restore_intent', dict(phase='reconcile_delete_resume', **binding, delete_resume=grant))
                recorded = (*recorded, resumed)
                worker.reconciliations[-1] = (prefix, identifier, selected, recorded)
        if present:
            parent = _remove(held, worker, scope,
                dict(decision_id=effect[0]['decision_id'], decision=effect[2]))
        else:
            with worker.mutation_authority(readers=True):
                held.verify()
                _sync_absence(held, worker, scope['parent_path'],
                    observation if observe_only else effect, observe_only=observe_only)
            parent = held.versions[scope['parent_path']].copy()
        extra = ({('observation_resume' if observe_only else 'delete_resume'):
                  dict(decision_id=observation[0]['decision_id'], decision=observation[2])}
                 if observation is not None else {})
        completed = worker.record('restore_intent', dict(phase='reconciled', **binding, **extra,
            parent_path=scope['parent_path'], parent_version=parent, uncertain=True,
            credited_removed_allocated_bytes=0))
        worker.reconciliations[-1] = (prefix, identifier, selected, (*recorded, completed))
    # Fresh inventory and whole replay observe the effects; no original packet,
    # decision, birth, archive, journal prefix or clock is rewritten.
    current = generation.inventory_historical_generation(worker.selected[2]['target_path'], allowed_roots=roots,
        max_seconds=worker.operation.remaining(), monotonic=monotonic, _restore_bounds=selected_restore_bounds(worker))
    return worker.replay(), current
