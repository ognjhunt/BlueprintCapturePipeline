"""Resume only authenticated private births under original restore authority.

Unrecorded creation or partial writes remain KEEP. Reused members are compared
against every original archive byte through retained no-follow descriptors.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from . import control_plane_lane_experiment_archive as transport
from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_scratch_decisions as encoding
from .control_plane_lane_historical_restore_archive import extract_preserved_members
from .control_plane_lane_historical_restore_staging import validate_private_prefix
from .control_plane_lane_historical_restore_tree import RestoreTree


class _PrefixTree:
    """No replacement writes or birth records for authenticated existing rows."""
    def __init__(self, tree, births):
        self.tree, self.births = tree, births

    def directory(self, row):
        if row['path'] not in self.births:
            return self.tree.directory(row)
        relative = self.tree.name + ('/' + row['path'] if row['path'] else '')
        with self.tree.held._opened(relative) as (_, guard):
            with self.tree.worker.mutation_authority(readers=True):
                guard()

    @contextmanager
    def member(self, row):
        if row['path'] not in self.births:
            with self.tree.member(row) as output:
                yield output
            return
        tree = self.tree
        with tree.held._opened(tree.name + '/' + row['path']) as (fd, guard):
            read = 0
            class Compare:
                def write(self, payload):
                    nonlocal read
                    generation._require(isinstance(payload, bytes) and 0 < len(payload) <= 1024**2
                        and read + len(payload) <= row['size_bytes'], 'restore_payload_changed')
                    offset = 0
                    while offset < len(payload):
                        with tree.worker.mutation_authority(readers=True):
                            guard()
                            block = os.read(fd, len(payload) - offset)
                            guard()
                        generation._require(block and block == payload[offset:offset + len(block)],
                                            'restore_payload_changed')
                        offset += len(block)
                        read += len(block)
                    return len(payload)
            yield Compare()
            with tree.worker.mutation_authority(readers=True):
                guard()
                generation._require(read == row['size_bytes'] and not os.read(fd, 1), 'restore_payload_changed')
                guard()


def recover_private_prefix(worker, events, roots, monotonic):
    from .control_plane_lane_historical_restore_worker import _readback, _resources, _finish_restore
    manifest, decision = worker.selected[2], worker.selected[1]
    observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
        max_seconds=worker.operation.remaining(), monotonic=monotonic)
    try:
        births = validate_private_prefix(manifest, observed, decision, events, worker.action_id,
                                         tick=worker.operation.remaining)
    except generation.HistoricalGenerationError:
        from .control_plane_lane_historical_restore_reconciliation_worker import reconcile_unlogged_creation
        events, observed = reconcile_unlogged_creation(worker, events, observed, roots, monotonic)
        births = validate_private_prefix(manifest, observed, decision, events, worker.action_id,
                                         tick=worker.operation.remaining)
    raw = encoding.encode_validation_report(manifest)
    generation._require(authority._selector(raw) == decision['manifest'], 'restore_manifest_changed')
    _readback(worker, observed, roots, monotonic)
    with _resources(worker, observed) as (held, reservation):
        tree = RestoreTree(held, worker, manifest)
        def guard():
            with worker.checkpoint(journal=True):
                held.verify()
                reservation.renew()
        guard()
        with worker.mutation_authority(readers=True):
            held.verify()
        with worker.checkpoint(journal=True) as (files, config, _):
            client, bucket = transport._client(files, config)
        extracted = extract_preserved_members(manifest, raw, decision['archive'], client, bucket,
            _PrefixTree(tree, births), guard, origin=worker.operation.started)
        tree.verify_bytes(staged=True)
        guard()
        worker.record('restore_intent', dict(phase='stage_complete', **extracted))
        tree.publish()
        receipt = _finish_restore(worker, tree, held, roots, monotonic, extracted)
        reused = [row for row in manifest['members'] if row['path'] in births and row['kind'] == 'file']
        count, size = len(reused), sum(row['size_bytes'] for row in reused)
        return dict(receipt, recovered_prefix=True, reused_files=count, reused_logical_bytes=size,
            restored_files=extracted['restored_files'] - count,
            restored_logical_bytes=extracted['restored_logical_bytes'] - size)
