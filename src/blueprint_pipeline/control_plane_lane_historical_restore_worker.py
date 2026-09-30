"""Restore exact preserved bytes under a distinct current native action.

The original generation stays protected through full stage/readback, no-replace
publication and durable final. Unknown partial recovery remains a typed KEEP.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from . import control_plane_disk_ledger as ledger
from . import control_plane_lane_experiment_archive as transport
from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_historical_archive as archive
from . import control_plane_lane_scratch_decisions as encoding
from .control_plane_disk_budget import reserve_control_plane_disk
from .control_plane_lane_historical_fence import _HistoricalGenerationFence
from .control_plane_lane_historical_restore_archive import extract_preserved_members
from .control_plane_lane_historical_restore_tree import RestoreTree
from .control_plane_lane_historical_restore_snapshot import after_reopen, validate_private
from .control_plane_lane_historical_restore_publication import validate_publication
from .control_plane_lane_historical_restore_staging import validate_complete_stage
from .control_plane_lane_historical_restore_receipts import validate_restore_final, validate_pending_owner_access
from .control_plane_lane_historical_sandbox import HistoricalNativeSandbox


def _restored_snapshot(worker, events):
    """Authenticate actual durable final and protected physical observations."""
    finals = [event for event in events if event['kind'] == 'restore_final']
    generation._require(len(finals) == 1, 'restore_incomplete')
    receipt = finals[0]['body']
    with worker.checkpoint(journal=True) as (_, _, journal):
        snapshot = journal.read_restore_snapshot(receipt.get('restored_snapshot'))
    return validate_restore_final(worker.selected, events, snapshot), snapshot


def _verify_snapshot(worker, events, snapshot):
    """Original bytes plus actual journaled new file inodes; no synthetic birth."""
    manifest, decision = worker.selected[2], worker.selected[1]
    validate_private(manifest, snapshot)
    files = [row for row in manifest['members'] if row['kind'] == 'file']
    members = [event['body'] for event in events if event['kind'] == 'restore_member']
    by_path = {row['path']: row for row in members}
    generation._require(len(members) == len(files)
        and set(by_path) == {row['path'] for row in files}
        and all(by_path[row['path']]['sha256'] == row['sha256']
                    and by_path[row['path']]['size_bytes'] == row['size_bytes'] for row in files),
        'restore_final_invalid')
    generation._require(snapshot['root_version'] == decision['parent_version']
        and all(row['version'][:2] == by_path[row['path']]['version'][:2]
                for row in snapshot['members'] if row['kind'] == 'file'), 'restore_snapshot_changed')


def _readback(worker, expected, roots, monotonic):
    """Fresh full original cloud/local bytes under current protected authority."""
    manifest, decision = worker.selected[2], worker.selected[1]
    def guard():
        with worker.checkpoint(journal=True):
            generation.verify_historical_member_versions(expected, tick=worker.operation.remaining)
    guard()
    with worker.checkpoint(journal=True) as (files, config, _):
        client, bucket = transport._client(files, config)
    archive.verify_preservation(decision['archive'], client, bucket, guard, origin=worker.operation.started)
    observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
        max_seconds=worker.operation.remaining(), monotonic=monotonic)
    generation._require(observed == expected, 'restore_snapshot_changed')
    guard()


def _completed_restore(worker, events, roots, monotonic):
    """Fresh full readback; no publication, reservation or incremental credit."""
    manifest, decision = worker.selected[2], worker.selected[1]
    accesses = [event for event in events if event['kind'] == 'access_reopened']
    generation._require(len(accesses) == 1 and accesses[0] == events[-1], 'restore_incomplete')
    final, snapshot = _restored_snapshot(worker, events)
    access = accesses[0]['body']
    generation._require(final['sequence'] < accesses[0]['sequence']
        and access.get('phase') == 'owner_rights_observed' and access.get('path') == '', 'restore_final_invalid')
    _readback(worker, after_reopen(manifest, snapshot, access.get('version')), roots, monotonic)
    return dict(status='completed', action='restore', action_id=worker.action_id, owner=decision['owner'],
        generation_digest=manifest['generation_digest'], idempotent=True, owner_access_reopened=True,
        original_restore_final_event_digest=final['event_digest'], root_version=access['version'],
        restored_files=0, restored_logical_bytes=0, root_directory_retained=True)


@contextmanager
def _resources(worker, observed):
    """Real exclusive generation fence, fresh reservation and fixed sandbox."""
    operation, manifest = worker.operation, worker.selected[2]
    outcome = 'failed'
    with _HistoricalGenerationFence(observed, tick=operation.remaining) as held:
        quantum = os.fstatvfs(held.root).f_frsize
        generation._require(type(quantum) is int and 0 < quantum <= 1024**2, 'restore_reservation_unknown')
        need = 8 * 1024**2 + sum(((row['size_bytes'] + quantum - 1) // quantum) * quantum
            if row['kind'] == 'file' else 2 * quantum for row in manifest['members'])
        with worker.checkpoint(journal=True):
            held.verify()
            reservation = reserve_control_plane_disk('experiment_restore', target_root=held.target,
                expected_bytes=need, minimum_bytes=need, reservation_root=ledger.DEFAULT_RESERVATION_ROOT,
                fresh=True, evictor=None, lock_nonblocking=True)
        worker.reservation = reservation
        private = resource = None
        try:
            with worker.checkpoint(journal=True) as (_, _, journal):
                private = os.dup(journal.directory)
            resource = os.open(ledger.DEFAULT_RESERVATION_ROOT,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            with HistoricalNativeSandbox(held.root, private, observed, tick=operation.remaining,
                                         reservation=resource) as sandbox:
                worker.sandbox = sandbox
                yield held, reservation
                outcome = 'completed'
        finally:
            worker.reservation = None
            for descriptor in (private, resource):
                if descriptor is not None:
                    os.close(descriptor)
            reservation.release(outcome=outcome)


def _recover_access(worker, events, roots, monotonic):
    """Only a complete untouched private tree may finish its owner transition."""
    final, snapshot = _restored_snapshot(worker, events)
    validate_pending_owner_access(worker.selected, events, snapshot, final)
    _readback(worker, snapshot, roots, monotonic)
    with _resources(worker, snapshot) as (held, reservation):
        with worker.mutation_authority(readers=True):
            held.verify()
            reservation.renew()
        tree = RestoreTree(held, worker, worker.selected[2])
        tree.verify_bytes(staged=False)
        tree.reopen()
        return dict(status='completed', action='restore', action_id=worker.action_id,
            owner=worker.selected[1]['owner'], generation_digest=worker.selected[2]['generation_digest'],
            recovered_access=True, owner_access_reopened=True, fresh_disk_reservation=True,
            original_restore_final_event_digest=final['event_digest'], root_version=held.versions[''],
            restored_files=0, restored_logical_bytes=0, root_directory_retained=True)


def _recover_before_final(worker, events, roots, monotonic):
    """Complete only authenticated physical stages or privately snapshotted trees."""
    if events[-1]['kind'] == 'restore_intent' and events[-1]['body'].get('phase') == 'stage_complete':
        return _recover_complete_stage(worker, events, roots, monotonic)
    if all(event['kind'] in ('intent', 'restore_directory', 'restore_member')
           or event['kind'] == 'restore_intent' and event['body'].get('phase') in
               ('reservation', 'directory', 'member') for event in events):
        from .control_plane_lane_historical_restore_prefix import recover_private_prefix
        return recover_private_prefix(worker, events, roots, monotonic)
    generation._require(events[-1]['kind'] == 'restore_intent'
        and events[-1]['body'].get('phase') in ('owner_rights', 'stage_removed')
        and not any(event['kind'] in ('restore_final', 'access_reopened') for event in events),
        'restore_recovery_required')
    with worker.checkpoint(journal=True) as (_, _, journal):
        try:
            os.stat('restore.snapshot.json', dir_fd=journal.directory, follow_symlinks=False)
        except FileNotFoundError:
            missing_snapshot = True
        else:
            missing_snapshot = False
        if not missing_snapshot:
            selector, snapshot = journal.select_restore_snapshot()
    if missing_snapshot:
        return _recover_publication(worker, events, roots, monotonic)
    _verify_snapshot(worker, events, snapshot)
    _readback(worker, snapshot, roots, monotonic)
    manifest, decision = worker.selected[2], worker.selected[1]
    with _resources(worker, snapshot) as (held, reservation):
        with worker.mutation_authority(readers=True):
            held.verify()
            reservation.renew()
        tree = RestoreTree(held, worker, manifest)
        tree.verify_bytes(staged=False)
        files = [row for row in manifest['members'] if row['kind'] == 'file']
        receipt = dict(status='completed', action='restore', action_id=worker.action_id,
            owner=decision['owner'], generation_digest=manifest['generation_digest'],
            original_manifest=decision['manifest'], original_final_event_digest=decision['final_event_digest'],
            archive_sha256=decision['archive']['sha256'], archive_size_bytes=decision['archive']['size_bytes'],
            restored_files=len(files), restored_logical_bytes=sum(row['size_bytes'] for row in files),
            fresh_disk_reservation=True, root_directory_retained=True,
            protected_root_version=held.versions[''], restored_snapshot=selector, owner_access_reopened=False)
        with worker.checkpoint(journal=True):
            held.verify()
            generation.verify_historical_member_versions(snapshot, tick=worker.operation.remaining)
        final = worker.record('restore_final', receipt)
        tree.reopen()
        return dict(receipt, recovered_before_final=True, owner_access_reopened=True,
            root_version=held.versions[''], original_restore_final_event_digest=final['event_digest'],
            restored_files=0, restored_logical_bytes=0)


def _finish_restore(worker, tree, held, roots, monotonic, extracted):
    """Finish verified publication under the retained original resource gates."""
    manifest, decision = worker.selected[2], worker.selected[1]
    tree.owner_rights()
    tree.verify_bytes(staged=False)
    with worker.checkpoint(journal=True):
        held.verify()
        worker.reservation.renew()
    snapshot = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
        max_seconds=worker.operation.remaining(), monotonic=monotonic)
    generation._require(all(row['version'] == held.versions[row['path']]
        for row in snapshot['members']) and len(snapshot['members']) == len(manifest['members']),
        'restore_snapshot_changed')
    validate_private(manifest, snapshot)
    with worker.checkpoint(journal=True) as (_, _, journal):
        held.verify()
        generation.verify_historical_member_versions(snapshot, tick=worker.operation.remaining)
        selected_snapshot = journal.publish_restore_snapshot(snapshot)
    receipt = dict(status='completed', action='restore', action_id=worker.action_id,
        owner=decision['owner'], generation_digest=manifest['generation_digest'],
        original_manifest=decision['manifest'], original_final_event_digest=decision['final_event_digest'],
        fresh_disk_reservation=True, **extracted, root_directory_retained=True,
        protected_root_version=held.versions[''], restored_snapshot=selected_snapshot,
        owner_access_reopened=False)
    worker.record('restore_final', receipt)
    tree.reopen()
    return dict(receipt, owner_access_reopened=True, root_version=held.versions[''])


def _recover_publication(worker, events, roots, monotonic):
    """Actual published births and planned member rights behind a private root."""
    manifest, decision = worker.selected[2], worker.selected[1]
    observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
        max_seconds=worker.operation.remaining(), monotonic=monotonic)
    validate_publication(manifest, observed, decision, events, worker.action_id,
                         tick=worker.operation.remaining, pending_owner_rights=True)
    _readback(worker, observed, roots, monotonic)
    with _resources(worker, observed) as (held, reservation):
        with worker.mutation_authority(readers=True):
            held.verify()
            reservation.renew()
        tree = RestoreTree(held, worker, manifest)
        tree.verify_bytes(staged=False)
        files = [row for row in manifest['members'] if row['kind'] == 'file']
        extracted = dict(archive_sha256=decision['archive']['sha256'],
            archive_size_bytes=decision['archive']['size_bytes'], restored_files=len(files),
            restored_logical_bytes=sum(row['size_bytes'] for row in files))
        receipt = _finish_restore(worker, tree, held, roots, monotonic, extracted)
        return dict(receipt, recovered_publication=True, restored_files=0, restored_logical_bytes=0)


def _recover_complete_stage(worker, events, roots, monotonic):
    """Resume fully journaled private bytes; never create replacement births."""
    manifest, decision = worker.selected[2], worker.selected[1]
    observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
        max_seconds=worker.operation.remaining(), monotonic=monotonic)
    validate_complete_stage(manifest, observed, decision, events, worker.action_id,
                            tick=worker.operation.remaining)
    _readback(worker, observed, roots, monotonic)
    with _resources(worker, observed) as (held, reservation):
        with worker.mutation_authority(readers=True):
            held.verify()
            reservation.renew()
        tree = RestoreTree(held, worker, manifest)
        tree.verify_bytes(staged=True)
        tree.publish()
        extracted = {key: events[-1]['body'][key] for key in
            ('archive_sha256', 'archive_size_bytes', 'restored_files', 'restored_logical_bytes')}
        receipt = _finish_restore(worker, tree, held, roots, monotonic, extracted)
        return dict(receipt, recovered_stage=True, restored_files=0, restored_logical_bytes=0)


def _unwritten_restore_attempt(events, action_id):
    """Only classify pre-write steps; this proves no filesystem fact or authority."""
    if not events or events[0]['kind'] != 'intent':
        return False
    stage = '.historical-restore-' + action_id
    return all(event['kind'] == 'restore_intent' and
        (event['body'].get('phase') == 'reservation' or
         event['body'].get('phase') == 'directory' and event['body'].get('path') == ''
         and event['body'].get('stage_path') == stage) for event in events[1:])


def _unchanged_tombstone(observed, decision):
    return observed['member_count'] == 1 and observed['members'][0]['path'] == '' \
        and observed['members'][0]['kind'] == 'directory' \
        and observed['members'][0]['version'] == decision['tombstone_version'] \
        and observed['root_version'] == decision['parent_version']


def run_restore(worker, roots, monotonic):
    operation = worker.operation
    manifest, decision = worker.selected[2], worker.selected[1]
    events = worker.replay()
    if any(event['kind'] == 'restore_final' for event in events):
        if not any(event['kind'] == 'access_reopened' for event in events):
            return _recover_access(worker, events, roots, monotonic)
        return _completed_restore(worker, events, roots, monotonic)
    observed, restarted = None, False
    if len(events) != 1 or events[0]['kind'] != 'intent':
        if _unwritten_restore_attempt(events, worker.action_id):
            observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
                max_seconds=operation.remaining(), monotonic=monotonic)
            if _unchanged_tombstone(observed, decision):
                with worker.checkpoint(journal=True) as (_, _, journal):
                    try:
                        journal.select_restore_snapshot()
                    except FileNotFoundError:
                        restarted = True
        if not restarted:
            return _recover_before_final(worker, events, roots, monotonic)
    if observed is None:
        observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
            max_seconds=operation.remaining(), monotonic=monotonic)
    generation._require(_unchanged_tombstone(observed, decision), 'restore_tombstone_changed')
    raw = encoding.encode_validation_report(manifest)
    generation._require(authority._selector(raw) == decision['manifest'], 'restore_manifest_changed')
    with _resources(worker, observed) as (held, reservation):
        tree = RestoreTree(held, worker, manifest)
        def guard():
            with worker.checkpoint(journal=True):
                held.verify()
                reservation.renew()
        guard()
        with worker.mutation_authority(readers=True):
            held.verify()
        worker.record('restore_intent', dict(phase='reservation', role='experiment_restore',
            token=reservation.token, expected_bytes=reservation.expected_bytes,
            device=reservation.device, owner_pid=os.getpid()))
        with worker.checkpoint(journal=True) as (files, config, _):
            client, bucket = transport._client(files, config)
        extracted = extract_preserved_members(manifest, raw, decision['archive'], client, bucket,
            tree, guard, origin=operation.started)
        tree.verify_bytes(staged=True)
        guard()
        worker.record('restore_intent', dict(phase='stage_complete', **extracted))
        tree.publish()
        receipt = _finish_restore(worker, tree, held, roots, monotonic, extracted)
        return dict(receipt, restarted_unwritten=True) if restarted else receipt
