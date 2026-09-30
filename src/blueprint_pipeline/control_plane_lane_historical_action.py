"""Fixed historical action worker; fresh authority is held around revocation.

Owner decisions alone never clear readers. The worker currently refuses payload
removal until its connected reference, preservation and recovery gates exist.
No entrypoint launches this partial worker on an installed host.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_dispatch as dispatch
from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_fence import _HistoricalGenerationFence
from .control_plane_lane_historical_journal import HistoricalActionJournal, journal_root
from .control_plane_lane_historical_unit import prove_historical_unit
from .control_plane_lane_historical_references import historical_reference_fence
from .control_plane_lane_historical_sandbox import HistoricalNativeSandbox


def _require(value, code):
    generation._require(value, 'action_' + code)


class _Worker:
    def __init__(self, config_path, action_id, operation):
        self.config_path, self.action_id, self.operation = config_path, action_id, operation
        self.selected = None
        self.head = None
        self.sandbox = None

    @contextmanager
    def checkpoint(self, *, journal=False):
        with authority._session(self.config_path, self.operation) as (files, config, store):
            _require(config.historical_generation_actions_enabled is True, 'disabled')
            current = dispatch._selection(files, config, store, self.config_path,
                                           self.action_id, self.operation.moment())
            if self.selected is None:
                self.selected = current
            else:
                _require(current == self.selected, 'authority_changed')
            target = current[2]['target_path']
            prove_historical_unit(self.action_id, target, str(journal_root(config)))
            selected_journal = HistoricalActionJournal(files, config, current, self.operation) if journal else None
            if selected_journal is not None:
                head = selected_journal.head
                _require(self.head is None or head['event_digest'] == self.head, 'journal_changed')
            yield files, config, selected_journal
            # Keep the actual EX lock and protected acquired records through the
            # caller's syscall and final verification, rather than returning a
            # stale authorization boolean before a mutation.
            again = dispatch._selection(files, config, store, self.config_path,
                                         self.action_id, self.operation.moment())
            _require(again == current, 'authority_changed')
            prove_historical_unit(self.action_id, target, str(journal_root(config)))
            self.operation.remaining()

    @contextmanager
    def mutation_authority(self, *, readers=False):
        with self.checkpoint(journal=True) as (files, config, _):
            if readers:
                with historical_reference_fence(files, config,
                        Path(self.selected[2]['target_path']), observed_at=self.operation.moment()) as guard:
                    _require(self.sandbox is not None, 'sandbox_unknown')
                    self.sandbox.refuse_references()
                    guard()
                    yield
                    guard()
            else:
                yield

    def record(self, kind, body):
        with self.checkpoint(journal=True) as (_, _, journal):
            head = journal.head
            event = journal.append(kind, body, previous=head['event_digest'])
            self.head = event['event_digest']


def run_historical_action(*, installed_config_path, action_id, now, monotonic=time.monotonic):
    """Internal worker seam: authenticate exact ID, native unit, journal and tree.

    The root dispatcher will use a fixed config and executable; this function
    accepts config only for the installed Python seam and disposable tests.
    """
    operation = authority._Operation(now, monotonic)
    # Disabled configurations refuse before native probes or journal creation.
    with authority._session(installed_config_path, operation) as (_, config, _):
        _require(config.historical_generation_actions_enabled is True, 'disabled')
    dispatch.select_historical_action(installed_config_path=installed_config_path,
                                     action_id=action_id, now=operation.moment(), monotonic=monotonic)
    worker = _Worker(installed_config_path, action_id, operation)
    with worker.checkpoint() as (files, config, _):
        roots = authority.owners._roots(config, files.budget)
    manifest = worker.selected[2]
    observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
        max_seconds=operation.remaining(), monotonic=monotonic)
    _require(observed == manifest, 'generation_changed')
    with worker.checkpoint(journal=True) as (_, _, journal):
        head = journal.head
        _require(head['kind'] == 'intent', 'recovery_required')
        worker.head = head['event_digest']
    with _HistoricalGenerationFence(manifest, tick=operation.remaining) as held:
        with worker.checkpoint(journal=True) as (_, _, journal):
            private = os.dup(journal.directory)
        try:
            with HistoricalNativeSandbox(held.root, private, manifest, tick=operation.remaining) as sandbox:
                worker.sandbox = sandbox
                held.revoke(before_change=worker.mutation_authority, record=worker.record)
                with worker.mutation_authority(readers=True):
                    held.verify()
                _require(worker.selected[1]['action'] == 'delete', 'preservation_required')
                def removal_authority():
                    return worker.mutation_authority(readers=True)
                outcome = held.remove_members(before_change=removal_authority, record=worker.record)
                receipt = dict(status='completed', action='delete', action_id=action_id,
                    owner=worker.selected[1]['owner'], generation_digest=manifest['generation_digest'],
                    original_manifest=worker.selected[1]['manifest'], **outcome)
                worker.record('final', receipt)
                return receipt
        finally:
            os.close(private)
