"""Observe authenticated historical terminal facts without renewing write authority.

ADP-009D/day28. A full past journal plus the exact current protected tombstone
can prove an already completed action. Observation never launches a unit,
appends an event, adopts an expired mutation clock or credits removals again.
"""
from __future__ import annotations

import os
import stat
import time

from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_dispatch as dispatch
from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_owner_consents as owners
from .control_plane_lane_historical_journal import HistoricalJournalObservation, MAX_EVENTS, journal_root
from .control_plane_lane_historical_recovery import recover_action
from .decision_evidence_contracts import canonical_digest


def _require(value, code):
    generation._require(value, 'receipt_' + code)


def validate_historical_final(selected, events):
    """Pure recorded-transition validation; no label, clock or absence authority."""
    _, decision, manifest, _ = selected
    action = decision['action']
    _require(action in ('delete', 'offload') and type(events) is list
        and 0 < len(events) <= MAX_EVENTS and events[-1]['kind'] == 'final', 'invalid')
    final, receipt = events[-1], events[-1]['body']
    _require(receipt.get('status') == 'completed' and receipt.get('action') == action
        and receipt.get('action_id') == decision['action_id'] and receipt.get('owner') == decision['owner']
        and receipt.get('generation_digest') == manifest['generation_digest']
        and receipt.get('original_manifest') == decision['manifest']
        and receipt.get('root_directory_retained') is True
        and receipt.get('uncertain_removed_allocated_bytes') == 0
        and all(type(receipt.get(key)) is int and receipt[key] >= 0 for key in
                ('removed_files', 'removed_directories', 'logical_bytes',
                 'observed_removed_allocated_bytes', 'uncertain_removed_allocated_bytes'))
        and all(decision['issued_at_epoch'] <= event['observed_at_epoch'] < decision['expires_at_epoch']
                for event in events), 'invalid')
    # A projection of recorded facts, not an observation of filesystem absence.
    tombstone = dict(manifest, member_count=1,
        members=[dict(manifest['members'][0], version=receipt.get('tombstone_version'))])
    _require(tombstone['members'][0]['path'] == '', 'invalid')
    recovered = recover_action(manifest, events[:-1], tombstone)
    known, uncertain = recovered['removed'], recovered['uncertain']
    rows = {row['path']: row for row in manifest['members']}
    _require(known | uncertain == set(rows) - {''} and not recovered['pending_removal']
        and recovered['reconcile'] is None
        and receipt['removed_files'] == sum(rows[name]['kind'] == 'file' for name in known)
        and receipt['removed_directories'] == sum(rows[name]['kind'] == 'directory' for name in known)
        and receipt['logical_bytes'] == sum(rows[name]['size_bytes'] for name in known)
        and type(receipt.get('uncertain_removed_members', 0)) is int
        and receipt.get('uncertain_removed_members', 0) == len(uncertain)
        and receipt['observed_removed_allocated_bytes'] == recovered['prior_observed_removed_allocated_bytes'],
        'invalid')
    event = recovered['preservation']
    if action == 'delete':
        _require(event is None and 'preservation' not in receipt
            and 'preservation_event_digest' not in receipt, 'invalid')
    else:
        _require(event is not None, 'preservation_invalid')
        pointer = event['body']
        _require(set(pointer) == {'schema_version', 'action_id', 'generation_digest', 'manifest',
                'archive', 'pointer_digest'}
            and pointer['schema_version'] == 'control_plane_historical_preservation.v1'
            and pointer['action_id'] == decision['action_id']
            and pointer['generation_digest'] == manifest['generation_digest']
            and pointer['manifest'] == decision['manifest']
            and pointer['pointer_digest'] == canonical_digest(pointer, digest_field='pointer_digest')
            and receipt.get('preservation_event_digest') == event['event_digest']
            and receipt.get('preservation') == pointer['archive'], 'preservation_invalid')
    return dict(final=final, preservation=event)


def _original(files, config, store, action_id, operation):
    _require(config.historical_generation_actions_enabled is True, 'disabled')
    value, _ = store.read(action_id)
    if value.get('schema_version') == 'control_plane_historical_restore_decision.v1':
        # Restore observation needs its separately authenticated snapshot/access
        # chain. Never mistake restore_final alone for reopened owner access.
        return None
    selected = dispatch._original_selection(store, action_id)
    packet, decision = selected[:2]
    _require(all(owners._number(value) for value in (packet.get('observed_at_epoch'),
        packet.get('expires_at_epoch'), decision.get('issued_at_epoch'), decision.get('expires_at_epoch')))
        and packet['observed_at_epoch'] <= decision['issued_at_epoch']
        < decision['expires_at_epoch'] <= packet['expires_at_epoch']
        and decision['issued_at_epoch'] <= operation.moment(), 'invalid')
    return selected


def observe_historical_action(*, installed_config_path, action_id, now, monotonic=time.monotonic):
    """Full immutable chain and fresh tombstone; an absent/incomplete journal is pending."""
    operation = authority._Operation(now, monotonic)
    original, start, previous, observed_at, head, events = None, 0, None, None, None, []
    while True:
        with authority._session(installed_config_path, operation) as (files, config, store):
            selected = _original(files, config, store, action_id, operation)
            if selected is None:
                return None
            if original is None:
                original = selected
                parent, name = files.parent(journal_root(config) / action_id, protected=True)
                owners._protected(os.fstat(parent), directory=True, mode=0o700)
                try:
                    os.stat(name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    return None
            _require(original == selected, 'changed')
            journal = HistoricalJournalObservation(files, config, selected, operation)
            if head is None:
                if journal.head['kind'] != 'final':
                    return None
                head = journal.head['event_digest']
            batch = journal.replay_batch(start, previous=previous, observed_at=observed_at, expected_head=head)
            events.extend(batch['events'])
            start, previous, observed_at = batch['next_start'], batch['previous'], batch['observed_at']
            if batch['complete']:
                roots = owners._roots(config, files.budget)
                break
    completed = validate_historical_final(original, events)
    manifest, receipt = original[2], completed['final']['body']
    observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
        max_seconds=operation.remaining(), monotonic=monotonic)
    _require(observed['member_count'] == 1 and observed['members'][0]['path'] == ''
        and observed['members'][0]['kind'] == 'directory'
        and observed['members'][0]['version'] == receipt.get('tombstone_version')
        and observed['members'][0]['version'][:2] == manifest['members'][0]['version'][:2]
        and observed['members'][0]['version'][2:5] == [stat.S_IFDIR | 0o700, 0, 0]
        and observed['root_version'] == manifest['root_version'], 'tombstone_changed')
    with authority._session(installed_config_path, operation) as (files, config, store):
        _require(_original(files, config, store, action_id, operation) == original
            and HistoricalJournalObservation(files, config, original, operation).head['event_digest'] == head,
            'changed')
        generation.verify_historical_member_versions(observed, tick=files.budget.tick)
        result = dict(status='completed', action=original[1]['action'], action_id=action_id,
            owner=original[1]['owner'], generation_digest=manifest['generation_digest'],
            original_final_event_digest=head, idempotent=True, observation_only=True,
            execution_authorized=False, action_unit_started=False, mutations=0, removed_bytes=0,
            removed_files=0, removed_directories=0, logical_bytes=0,
            observed_removed_allocated_bytes=0, uncertain_removed_allocated_bytes=0,
            observed_at_epoch=operation.moment())
        return result
