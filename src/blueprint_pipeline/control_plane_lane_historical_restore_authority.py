"""Distinct current owner approval over authenticated past offload facts.

ADP-009D/day28. Observation and approval do not download, publish, reopen access
or grant execution. The restore worker must independently reserve and fence.
"""
from __future__ import annotations

import math
import secrets
import stat
import time

from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_dispatch as dispatch
from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_legacy_owner as legacy
from . import control_plane_lane_owner_consents as owners
from .control_plane_lane_historical_journal import HistoricalJournalObservation
from .control_plane_lane_historical_recovery import recover_action
from .decision_evidence_contracts import canonical_digest

SCHEMA = 'control_plane_historical_restore_decision.v1'
_FIELDS = frozenset({'schema_version', 'action_id', 'action', 'principal', 'owner', 'offload_action_id',
    'source_records', 'packet_id', 'generation_digest', 'manifest', 'target_path', 'final_event_digest',
    'preservation_event_digest', 'archive', 'tombstone_version', 'parent_version', 'installed_config',
    'policy', 'issued_at_epoch', 'expires_at_epoch', 'restore_approved', 'execution_authorized', 'decision_digest'})


def _require(value, code):
    generation._require(value, 'restore_' + code)


def _completed(selected, events):
    """Replay actual recorded transitions; a final label alone is insufficient."""
    _, decision, manifest, _ = selected
    _require(events[-1]['kind'] == 'final', 'offload_incomplete')
    final, receipt = events[-1], events[-1]['body']
    _require(receipt.get('status') == 'completed' and receipt.get('action') == 'offload'
        and receipt.get('action_id') == decision['action_id'] and receipt.get('owner') == decision['owner']
        and receipt.get('generation_digest') == manifest['generation_digest']
        and receipt.get('original_manifest') == decision['manifest']
        and receipt.get('root_directory_retained') is True
        and receipt.get('uncertain_removed_allocated_bytes') == 0, 'offload_invalid')
    # This is a projection of journal facts, not a filesystem observation.
    # Its resulting root must later match a separate current full inventory.
    tombstone = dict(manifest, member_count=1,
        members=[dict(manifest['members'][0], version=receipt.get('tombstone_version'))])
    _require(tombstone['members'][0]['path'] == '', 'offload_invalid')
    recovered = recover_action(manifest, events[:-1], tombstone)
    known, uncertain = recovered['removed'], recovered['uncertain']
    rows = {row['path']: row for row in manifest['members']}
    _require(known | uncertain == set(rows) - {''} and not recovered['pending_removal']
        and recovered['reconcile'] is None
        and receipt.get('removed_files') == sum(rows[name]['kind'] == 'file' for name in known)
        and receipt.get('removed_directories') == sum(rows[name]['kind'] == 'directory' for name in known)
        and receipt.get('logical_bytes') == sum(rows[name]['size_bytes'] for name in known)
        and receipt.get('uncertain_removed_members', 0) == len(uncertain)
        and receipt.get('observed_removed_allocated_bytes') == recovered['prior_observed_removed_allocated_bytes'],
        'offload_invalid')
    event = recovered['preservation']
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


def observe_preserved_generation(*, installed_config_path, offload_action_id, operation):
    """Whole immutable past chain, bounded batches, no adoption of old clocks."""
    original = None
    start, previous, observed_at, head, events = 0, None, None, None, []
    while True:
        with authority._session(installed_config_path, operation) as (files, config, store):
            selected = dispatch._original_selection(store, offload_action_id)
            _require(selected[1]['action'] == 'offload'
                and selected[0]['observed_at_epoch'] <= selected[1]['issued_at_epoch']
                < selected[1]['expires_at_epoch'] <= selected[0]['expires_at_epoch']
                and selected[1]['issued_at_epoch'] <= operation.moment(), 'offload_invalid')
            if original is None:
                original = selected
            _require(original == selected, 'source_changed')
            journal = HistoricalJournalObservation(files, config, selected, operation)
            if head is None:
                head = journal.head['event_digest']
            batch = journal.replay_batch(start, previous=previous, observed_at=observed_at, expected_head=head)
            events.extend(batch['events'])
            start, previous, observed_at = batch['next_start'], batch['previous'], batch['observed_at']
            if batch['complete']:
                break
    return original, _completed(original, events)


def _current_authority(files, config, config_path, principal, owner, now, expiry):
    _require(type(expiry) in (int, float) and math.isfinite(expiry)
        and now < expiry <= now + 900, 'approval_expired')
    raw = legacy._policy_bytes(files, config)
    policy = owners._policy(raw, principal, files.budget)
    # REGISTER grants new owner access, OFFLOAD grants preserved-data handling.
    # Neither is inferred from the old principal, policy or offload approval.
    for action in ('register', 'offload'):
        owners._authorize(dict(owner=owner, action=action, ttl_seconds=expiry - now), policy, expiry, now)
    config_raw, _ = files.read(config_path, cap=owners.MAX_POLICY_BYTES, protected=True)
    return authority._selector(config_raw), authority._selector(raw)


def select_restore(files, config, store, config_path, action_id, moment):
    """Fresh protected decision, current owner/policy and exact past pointers.

    This selection does not adopt the old journal's mutation timer or clear
    target writers. The worker still verifies its new operation and native unit.
    """
    decision, raw = store.read(action_id)
    _require(set(decision) == _FIELDS and decision['schema_version'] == SCHEMA
        and decision['action_id'] == action_id and decision['action'] == 'restore'
        and decision['restore_approved'] is True and decision['execution_authorized'] is False
        and decision['decision_digest'] == canonical_digest(decision, digest_field='decision_digest')
        and decision['issued_at_epoch'] <= moment < decision['expires_at_epoch']
        <= decision['issued_at_epoch'] + 900, 'approval_invalid')
    _require(_current_authority(files, config, config_path, decision['principal'], decision['owner'],
        moment, decision['expires_at_epoch']) == (decision['installed_config'], decision['policy']),
        'authority_changed')
    original = dispatch._original_selection(store, decision['offload_action_id'])
    packet, old, manifest, records = original
    _require(old['action'] == 'offload' and old['owner'] == decision['owner']
        and decision['source_records'] == records and decision['packet_id'] == packet['packet_id']
        and all(decision[key] == old[key] for key in ('generation_digest', 'manifest'))
        and decision['target_path'] == manifest['target_path'], 'source_changed')
    observer = authority._Operation(moment, time.monotonic)
    final = HistoricalJournalObservation(files, config, original, observer).head
    _require(final['kind'] == 'final' and final['event_digest'] == decision['final_event_digest']
        and final['body'].get('status') == 'completed'
        and final['body'].get('preservation') == decision['archive']
        and final['body'].get('preservation_event_digest') == decision['preservation_event_digest']
        and final['body'].get('tombstone_version') == decision['tombstone_version'], 'source_changed')
    return packet, decision, manifest, dict(packet=records['packet'], decision=authority._selector(raw))


def approve_historical_restore(*, installed_config_path, offload_action_id, ack_final_event_digest,
        principal, owner, expires_at_epoch, now, monotonic=time.monotonic):
    operation = authority._Operation(now, monotonic)
    original, proof = observe_preserved_generation(installed_config_path=installed_config_path,
        offload_action_id=offload_action_id, operation=operation)
    packet, old_decision, manifest, records = original
    final = proof['final']
    _require(old_decision['owner'] == owner and final['event_digest'] == ack_final_event_digest,
             'approval_unacknowledged')
    with authority._session(installed_config_path, operation) as (files, config, _):
        configured, policy = _current_authority(files, config, installed_config_path,
            principal, owner, operation.moment(), expires_at_epoch)
        roots = owners._roots(config, files.budget)
    observed = generation.inventory_historical_generation(manifest['target_path'], allowed_roots=roots,
        max_seconds=operation.remaining(), monotonic=monotonic)
    root = observed['members'][0]
    _require(observed['member_count'] == 1 and root['path'] == '' and root['kind'] == 'directory'
        and root['version'] == final['body']['tombstone_version']
        and root['version'][:2] == manifest['members'][0]['version'][:2]
        and root['version'][2:5] == [stat.S_IFDIR | 0o700, 0, 0]
        and observed['root_identity'] == manifest['root_identity']
        and observed['root_version'][3] == 0 and not observed['root_version'][2] & 0o022,
        'tombstone_changed')
    with authority._session(installed_config_path, operation) as (files, config, store):
        _require(_current_authority(files, config, installed_config_path, principal, owner,
            operation.moment(), expires_at_epoch) == (configured, policy), 'authority_changed')
        _require(dispatch._original_selection(store, offload_action_id) == original, 'source_changed')
        journal = HistoricalJournalObservation(files, config, original, operation)
        _require(journal.head == final, 'source_changed')
        generation.verify_historical_member_versions(observed, tick=files.budget.tick)
        identifier = secrets.token_hex(16)
        _require(identifier not in (offload_action_id, packet['packet_id']), 'approval_invalid')
        value = dict(schema_version=SCHEMA, action_id=identifier, action='restore', principal=principal,
            owner=owner, offload_action_id=offload_action_id, source_records=records,
            packet_id=packet['packet_id'], generation_digest=manifest['generation_digest'],
            manifest=old_decision['manifest'], target_path=manifest['target_path'],
            final_event_digest=final['event_digest'],
            preservation_event_digest=proof['preservation']['event_digest'],
            archive=proof['preservation']['body']['archive'], tombstone_version=root['version'],
            parent_version=observed['root_version'], installed_config=configured, policy=policy,
            issued_at_epoch=operation.moment(), expires_at_epoch=expires_at_epoch,
            restore_approved=True, execution_authorized=False)
        value['decision_digest'] = canonical_digest(value, digest_field='decision_digest')
        store.publish(identifier, value)
        return value
