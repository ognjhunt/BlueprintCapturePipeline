"""Fixed historical action worker; fresh authority is held around revocation.

Owner decisions alone never clear readers. Delete holds current authority and
native reference gates through each original-member removal. Offload and restore
remain gated; no entrypoint launches this partial worker on an installed host.
"""
from __future__ import annotations

import os
import stat
import time
from contextlib import contextmanager
from pathlib import Path

from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_dispatch as dispatch
from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_historical_archive as archive
from . import control_plane_lane_experiment_archive as transport
from . import control_plane_lane_scratch_decisions as encoding
from .decision_evidence_contracts import canonical_digest
from .control_plane_lane_historical_fence import _HistoricalGenerationFence
from .control_plane_lane_historical_journal import HistoricalActionJournal, journal_root
from .control_plane_lane_historical_unit import prove_historical_unit
from .control_plane_lane_historical_references import historical_reference_fence
from .control_plane_lane_historical_sandbox import HistoricalNativeSandbox
from .control_plane_lane_historical_recovery import recover_action


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
            restore = current[1]['action'] == 'restore'
            prove_historical_unit(self.action_id, target, str(journal_root(config)), restore=restore)
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
            prove_historical_unit(self.action_id, target, str(journal_root(config)), restore=restore)
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
            return event

    def replay(self):
        """Authenticate every original journal link across bounded acquisitions."""
        start, previous, observed_at, events = 0, None, None, []
        while True:
            with self.checkpoint(journal=True) as (_, _, journal):
                if self.head is None:
                    self.head = journal.head['event_digest']
                batch = journal.replay_batch(start, previous=previous,
                    observed_at=observed_at, expected_head=self.head)
                events.extend(batch['events'])
                start, previous, observed_at = batch['next_start'], batch['previous'], batch['observed_at']
                if batch['complete']:
                    return events


def _preservation_record(worker, event):
    body = event['body']
    _require(worker.selected[1]['action'] == 'offload' and event['kind'] == 'preservation'
        and type(body) is dict and set(body) == {'schema_version', 'action_id', 'generation_digest',
            'manifest', 'archive', 'pointer_digest'}
        and body['schema_version'] == 'control_plane_historical_preservation.v1'
        and body['action_id'] == worker.action_id
        and body['generation_digest'] == worker.selected[2]['generation_digest']
        and body['manifest'] == worker.selected[1]['manifest']
        and body['pointer_digest'] == canonical_digest(body, digest_field='pointer_digest'), 'preservation_invalid')
    return body['archive']


def _archive_guard(worker, held=None):
    with worker.checkpoint(journal=True):
        if held is not None:
            held.verify()


def _readback_preservation(worker, event, held=None):
    pointer = _preservation_record(worker, event)
    with worker.checkpoint(journal=True) as (files, config, _):
        client, bucket = transport._client(files, config)
    archive.verify_preservation(pointer, client, bucket, lambda: _archive_guard(worker, held),
                                origin=worker.operation.started)
    return pointer


def _preserve(worker, held, recovered):
    previous = recovered['preservation'] if recovered else None
    if previous is not None:
        return _readback_preservation(worker, previous, held), previous['event_digest']
    _require(not recovered or not (recovered['removed'] or recovered['uncertain']), 'preservation_required')
    raw = encoding.encode_validation_report(worker.selected[2])
    _require(authority._selector(raw) == worker.selected[1]['manifest'], 'manifest_changed')
    with worker.checkpoint(journal=True) as (files, config, _):
        client, bucket = transport._client(files, config)
    pointer = archive.preserve(held, worker.selected[2], raw, client, bucket,
        lambda: _archive_guard(worker, held), origin=worker.operation.started)
    body = dict(schema_version='control_plane_historical_preservation.v1', action_id=worker.action_id,
        generation_digest=worker.selected[2]['generation_digest'], manifest=worker.selected[1]['manifest'],
        archive=pointer)
    body['pointer_digest'] = canonical_digest(body, digest_field='pointer_digest')
    event = worker.record('preservation', body)
    return pointer, event['event_digest']


def _completed_replay(worker, events, roots, monotonic):
    """Read an exact completed tombstone; never credit its removals twice."""
    manifest, decision = worker.selected[2], worker.selected[1]
    final = events[-1]
    receipt = final['body']
    _require(final['kind'] == 'final' and receipt.get('status') == 'completed'
        and receipt.get('action') == decision['action'] and decision['action'] in ('delete', 'offload')
        and receipt.get('action_id') == worker.action_id and receipt.get('owner') == decision['owner']
        and receipt.get('generation_digest') == manifest['generation_digest']
        and receipt.get('original_manifest') == decision['manifest']
        and receipt.get('root_directory_retained') is True
        and receipt.get('uncertain_removed_allocated_bytes') == 0, 'final_invalid')
    observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
        max_seconds=worker.operation.remaining(), monotonic=monotonic)
    _require(observed['member_count'] == 1 and observed['members'][0]['path'] == ''
        and observed['members'][0]['kind'] == 'directory'
        and observed['members'][0]['version'] == receipt.get('tombstone_version')
        and observed['members'][0]['version'][:2] == manifest['members'][0]['version'][:2]
        and observed['members'][0]['version'][3:5] == [0, 0]
        and observed['members'][0]['version'][2] == stat.S_IFDIR | 0o700
        and observed['root_version'] == manifest['root_version'], 'tombstone_changed')
    with worker.checkpoint(journal=True):
        generation.verify_historical_member_versions(observed, tick=worker.operation.remaining)
    if decision['action'] == 'offload':
        preservation = [event for event in events if event['kind'] == 'preservation']
        _require(len(preservation) == 1 and receipt.get('preservation_event_digest') == preservation[0]['event_digest']
            and receipt.get('preservation') == _readback_preservation(worker, preservation[0]), 'preservation_invalid')
        with worker.checkpoint(journal=True):
            generation.verify_historical_member_versions(observed, tick=worker.operation.remaining)
    return dict(status='completed', action=decision['action'], action_id=worker.action_id, owner=decision['owner'],
        generation_digest=manifest['generation_digest'], idempotent=True,
        original_final_event_digest=final['event_digest'], removed_files=0, removed_directories=0,
        logical_bytes=0, observed_removed_allocated_bytes=0, uncertain_removed_allocated_bytes=0,
        tombstone_version=receipt['tombstone_version'], root_directory_retained=True)


def run_historical_action(*, installed_config_path, action_id, now, monotonic=time.monotonic):
    """Internal worker seam: authenticate exact ID, native unit, journal and tree.

    The root dispatcher will use a fixed config and executable; this function
    accepts config only for the installed Python seam and disposable tests.
    """
    operation = authority._Operation(now, monotonic)
    # Disabled configurations refuse before native probes or journal creation.
    with authority._session(installed_config_path, operation) as (_, config, _):
        _require(config.historical_generation_actions_enabled is True, 'disabled')
    worker = _Worker(installed_config_path, action_id, operation)
    prior = None
    with worker.checkpoint() as (files, config, _):
        roots = authority.owners._roots(config, files.budget)
        parent, name = files.parent(journal_root(config) / action_id, protected=True)
        try:
            os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            # Only an already protected original operation can bypass the
            # new-generation hash preflight. Complete chain and exact current
            # tombstone verification are still required before a cached result.
            prior = True
    if worker.selected[1]['action'] == 'restore':
        from .control_plane_lane_historical_restore_worker import run_restore
        return run_restore(worker, roots, monotonic)
    recovered, events = None, None
    if prior:
        events = worker.replay()
        if events[-1]['kind'] == 'final':
            return _completed_replay(worker, events, roots, monotonic)
    else:
        dispatch.select_historical_action(installed_config_path=installed_config_path,
                                     action_id=action_id, now=operation.moment(), monotonic=monotonic)
    manifest = worker.selected[2]
    observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
        max_seconds=operation.remaining(), monotonic=monotonic)
    if events is not None:
        recovered = recover_action(manifest, events, observed)
    else:
        _require(observed == manifest, 'generation_changed')
    with worker.checkpoint(journal=True) as (_, _, journal):
        head = journal.head
        _require(recovered is not None or head['kind'] == 'intent', 'recovery_required')
        worker.head = head['event_digest']
    with _HistoricalGenerationFence(observed, tick=operation.remaining) as held:
        with worker.checkpoint(journal=True) as (_, _, journal):
            private = os.dup(journal.directory)
        try:
            with HistoricalNativeSandbox(held.root, private, manifest, tick=operation.remaining) as sandbox:
                worker.sandbox = sandbox
                held.revoke(before_change=worker.mutation_authority, record=worker.record,
                    completed=recovered['completed'] if recovered else frozenset(),
                    pending=recovered['pending'] if recovered else None)
                with worker.mutation_authority(readers=True):
                    held.verify()
                if recovered and recovered['reconcile'] is not None:
                    # The exact last intent names an absent original member.
                    # Record that uncertainty durably under fresh gates without
                    # claiming freed bytes or performing another payload effect.
                    with worker.mutation_authority(readers=True):
                        held.verify()
                        held.sync_directory(recovered['reconcile']['parent_path'])
                    worker.record('removal_uncertain', recovered['reconcile'])
                preservation, preservation_event = None, None
                if worker.selected[1]['action'] == 'offload':
                    preservation, preservation_event = _preserve(worker, held, recovered)
                else:
                    _require(not recovered or recovered['preservation'] is None, 'preservation_invalid')
                def removal_authority():
                    return worker.mutation_authority(readers=True)
                outcome = held.remove_members(before_change=removal_authority, record=worker.record,
                    pending=recovered['pending_removal'] if recovered else None)
                if recovered:
                    original = {row['path']: row for row in manifest['members']}
                    for name in recovered['removed']:
                        row = original[name]
                        outcome['removed_files'] += int(row['kind'] == 'file')
                        outcome['removed_directories'] += int(row['kind'] == 'directory')
                        outcome['logical_bytes'] += row['size_bytes']
                    outcome['observed_removed_allocated_bytes'] += recovered['prior_observed_removed_allocated_bytes']
                    outcome['uncertain_removed_members'] = len(recovered['uncertain'])
                receipt = dict(status='completed', action=worker.selected[1]['action'], action_id=action_id,
                    owner=worker.selected[1]['owner'], generation_digest=manifest['generation_digest'],
                    original_manifest=worker.selected[1]['manifest'], **outcome)
                if preservation is not None:
                    receipt.update(preservation=preservation, preservation_event_digest=preservation_event)
                worker.record('final', receipt)
                return receipt
        finally:
            os.close(private)
