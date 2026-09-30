"""Resume original publication without replacement births or fresh clocks."""
from __future__ import annotations

import os

from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_restore_split import validate_split_stage
from .control_plane_lane_historical_restore_tree import RestoreTree


def recover_split_stage(worker, events, roots, monotonic):
    from .control_plane_lane_historical_restore_worker import _readback, _resources, _finish_restore
    manifest, decision = worker.selected[2], worker.selected[1]
    observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
        max_seconds=worker.operation.remaining(), monotonic=monotonic)
    selection = validate_split_stage(manifest, observed, decision, events, worker.action_id,
                                     tick=worker.operation.remaining)
    _readback(worker, observed, roots, monotonic)
    with _resources(worker, observed) as (held, reservation):
        with worker.mutation_authority(readers=True):
            held.verify()
            reservation.renew()
        tree = RestoreTree(held, worker, manifest)
        tree.verify_bytes(staged=True, published=selection['published'])
        if selection['uncertain']:
            # This records an uncertain observed rename, never an inode birth
            # or newly restored byte. Both original parents are durable first.
            with held._opened(tree.name) as (stage, guard):
                with worker.mutation_authority(readers=True):
                    guard()
                    os.fsync(stage)
                    os.fsync(held.root)
                    guard()
            tree.publication_observed(selection['pending'], uncertain=True)
        if selection['stage_absent']:
            with worker.mutation_authority(readers=True):
                held.verify()
                os.fsync(held.root)
                held.verify()
            worker.record('restore_intent', dict(phase='stage_removed',
                target_version=held.versions[''], uncertain=True))
        else:
            tree.publish(published=selection['published'], pending=selection['pending'],
                         stage_removal_pending=selection['stage_removal_pending'])
        extracted = {key: selection['stage_complete'][key] for key in
            ('archive_sha256', 'archive_size_bytes', 'restored_files', 'restored_logical_bytes')}
        receipt = _finish_restore(worker, tree, held, roots, monotonic, extracted)
        return dict(receipt, recovered_split=True, restored_files=0, restored_logical_bytes=0)
