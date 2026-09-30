"""Restore exact preserved bytes under a distinct current native action.

The original generation stays protected through full stage/readback, no-replace
publication and durable final. Unknown partial recovery remains a typed KEEP.
"""
from __future__ import annotations

import os

from . import control_plane_disk_ledger as ledger
from . import control_plane_lane_experiment_archive as transport
from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_scratch_decisions as encoding
from .control_plane_disk_budget import reserve_control_plane_disk
from .control_plane_lane_historical_fence import _HistoricalGenerationFence
from .control_plane_lane_historical_restore_archive import extract_preserved_members
from .control_plane_lane_historical_restore_tree import RestoreTree
from .control_plane_lane_historical_sandbox import HistoricalNativeSandbox


def run_restore(worker, roots, monotonic):
    operation = worker.operation
    manifest, decision = worker.selected[2], worker.selected[1]
    events = worker.replay()
    generation._require(len(events) == 1 and events[0]['kind'] == 'intent', 'restore_recovery_required')
    observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
        max_seconds=operation.remaining(), monotonic=monotonic)
    generation._require(observed['member_count'] == 1 and observed['members'][0]['path'] == ''
        and observed['members'][0]['version'] == decision['tombstone_version']
        and observed['root_version'] == decision['parent_version'], 'restore_tombstone_changed')
    raw = encoding.encode_validation_report(manifest)
    generation._require(authority._selector(raw) == decision['manifest'], 'restore_manifest_changed')
    outcome = 'failed'
    with _HistoricalGenerationFence(observed, tick=operation.remaining) as held:
        # Include filesystem block rounding and finite directory/stage overhead.
        quantum = os.fstatvfs(held.root).f_frsize
        generation._require(type(quantum) is int and 0 < quantum <= 1024**2, 'restore_reservation_unknown')
        need = 8 * 1024**2 + sum(((row['size_bytes'] + quantum - 1) // quantum) * quantum
            if row['kind'] == 'file' else 2 * quantum for row in manifest['members'])
        with worker.checkpoint(journal=True):
            held.verify()
            reservation = reserve_control_plane_disk('experiment_restore', target_root=held.target,
                expected_bytes=need, minimum_bytes=need, reservation_root=ledger.DEFAULT_RESERVATION_ROOT,
                fresh=True, evictor=None, lock_nonblocking=True)
        private = resource = None
        try:
            with worker.checkpoint(journal=True) as (_, _, journal):
                private = os.dup(journal.directory)
            resource = os.open(ledger.DEFAULT_RESERVATION_ROOT,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            with HistoricalNativeSandbox(held.root, private, manifest, tick=operation.remaining,
                                         reservation=resource) as sandbox:
                worker.sandbox = sandbox
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
                tree.owner_rights()
                tree.verify_bytes(staged=False)
                guard()
                receipt = dict(status='completed', action='restore', action_id=worker.action_id,
                    owner=decision['owner'], generation_digest=manifest['generation_digest'],
                    original_manifest=decision['manifest'], original_final_event_digest=decision['final_event_digest'],
                    fresh_disk_reservation=True, **extracted, root_directory_retained=True,
                    protected_root_version=held.versions[''], owner_access_reopened=False)
                worker.record('restore_final', receipt)
                tree.reopen()
                outcome = 'completed'
                return dict(receipt, owner_access_reopened=True, root_version=held.versions[''])
        finally:
            for descriptor in (private, resource):
                if descriptor is not None:
                    os.close(descriptor)
            reservation.release(outcome=outcome)
