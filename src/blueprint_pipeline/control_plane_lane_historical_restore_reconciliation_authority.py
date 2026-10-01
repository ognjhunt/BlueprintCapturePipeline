"""Distinct owner DELETE decision for one unfinished original restore row.

Review and approval do not remove bytes, adopt a creation, clear references,
renew the original restore permission or start a new operation clock.
"""
from __future__ import annotations

import hashlib
import math
import time

from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_historical_restore_authority as restore
from . import control_plane_lane_legacy_owner as legacy
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_experiment_work as work
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_historical_journal import HistoricalJournalObservation, journal_root, MAX_EVENT_BYTES
from .control_plane_lane_historical_restore_reconciliation_scope import unknown_creation_scope
from .decision_evidence_contracts import canonical_digest

SCHEMA = 'control_plane_historical_restore_reconciliation_decision.v1'
_FIELDS = frozenset({'schema_version', 'decision_id', 'action', 'principal', 'owner', 'packet',
    'packet_digest', 'observed_manifest', 'installed_config', 'policy', 'issued_at_epoch',
    'expires_at_epoch', 'discard_unfinished_row_approved', 'no_future_writers', 'no_future_readers',
    'execution_authorized', 'decision_digest'})


def _require(value):
    generation._require(value, 'restore_reconciliation_approval_invalid')


def decision_id(action_id, original_head):
    """Fixed lookup from the original operation and exact interrupted head."""
    _require(type(action_id) is str and authority._ID.fullmatch(action_id)
        and type(original_head) is str and len(original_head) == 71 and original_head.startswith('sha256:'))
    return hashlib.sha256(('restore-reconcile.v1:' + action_id + ':' + original_head).encode()).hexdigest()[:32]


def _current_delete(files, config, config_path, principal, owner, moment, expiry, original_expiry):
    _require(type(expiry) in (int, float) and math.isfinite(expiry)
        and moment < expiry <= min(moment + 900, original_expiry))
    raw = legacy._policy_bytes(files, config)
    policy = owners._policy(raw, principal, files.budget)
    owners._authorize(dict(owner=owner, action='delete', ttl_seconds=expiry - moment), policy, expiry, moment)
    configured, _ = files.read(config_path, cap=owners.MAX_POLICY_BYTES, protected=True)
    return authority._selector(configured), authority._selector(raw)


def _event_selectors(files, config, action_id, events):
    selected = []
    for event in (events[0], events[-1]):
        raw, record = files.read(journal_root(config) / action_id / f"e-{event['sequence']:05d}.json",
                                cap=MAX_EVENT_BYTES, protected=True, mode=0o600)
        _require(retained._document(raw, MAX_EVENT_BYTES, _work_budget=files.budget) == event)
        files.verify_record(record)
        selected.append(authority._selector(raw))
    return selected


def _packet(selected, events, observed, action_id, byte_selectors, *, tick):
    scope = unknown_creation_scope(selected[2], observed, selected[1], events, action_id, tick=tick)
    value = dict(schema_version='control_plane_historical_restore_reconciliation_packet.v1',
        action_id=action_id, owner=selected[1]['owner'], source_records=selected[3],
        original_intent=events[0], original_head_event_digest=events[-1]['event_digest'],
        original_intent_bytes=byte_selectors[0], original_head_bytes=byte_selectors[1],
        original_expires_at_epoch=selected[1]['expires_at_epoch'], scope=scope,
        execution_authorized=False)
    value['packet_digest'] = canonical_digest(value, digest_field='packet_digest')
    return value


def _observe(config_path, action_id, operation):
    start, previous, moment, head, events, selected = 0, None, None, None, [], None
    while True:
        with authority._session(config_path, operation) as (files, config, store):
            current = restore.select_restore(files, config, store, config_path, action_id, operation.moment())
            if selected is None:
                selected = current
            _require(selected == current)
            observer = HistoricalJournalObservation(files, config, selected, operation)
            if head is None:
                head = observer.head['event_digest']
            batch = observer.replay_batch(start, previous=previous, observed_at=moment, expected_head=head)
            events.extend(batch['events'])
            start, previous, moment = batch['next_start'], batch['previous'], batch['observed_at']
            if batch['complete']:
                seed = events[0]['body']
                _require(seed['boot_id'] == work._controller_boot_id(files)
                    and seed['started_monotonic'] <= operation.monotonic()
                    < seed['started_monotonic'] + generation.MAX_SECONDS)
                roots = owners._roots(config, files.budget)
                byte_selectors = _event_selectors(files, config, action_id, events)
                break
    observed = generation.inventory_historical_generation(selected[2]['target_path'], allowed_roots=roots,
        max_seconds=operation.remaining(), monotonic=operation.monotonic)
    packet = _packet(selected, events, observed, action_id, byte_selectors, tick=operation.remaining)
    with authority._session(config_path, operation) as (files, config, store):
        _require(restore.select_restore(files, config, store, config_path, action_id, operation.moment()) == selected
            and HistoricalJournalObservation(files, config, selected, operation).head['event_digest'] == head)
        _require(_event_selectors(files, config, action_id, events) == byte_selectors)
        generation.verify_historical_member_versions(observed, tick=files.budget.tick)
    return selected, events, observed, packet


def observe_historical_restore_reconciliation(*, installed_config_path, action_id, now,
        monotonic=time.monotonic):
    """Read-only exact owner packet; no original journal creation or append."""
    return _observe(installed_config_path, action_id, authority._Operation(now, monotonic))[3]


def select_reconciliation(files, config, store, config_path, selected, events, identifier, selector, moment, *, consumed=None):
    """Fresh protected DELETE approval; current physical gates remain separate."""
    try:
        value, raw = store.read(identifier)
    except FileNotFoundError:
        # Classify absence while still inside the protected session. Its
        # OSError boundary must continue to refuse every other IO failure.
        raise generation.HistoricalGenerationError(
            'historical_generation_restore_reconciliation_approval_missing') from None
    _require((selector is None or authority._selector(raw) == selector)
        and set(value) == _FIELDS and value['schema_version'] == SCHEMA
        and value['decision_id'] == identifier and value['action'] == 'discard_unfinished_restore_row'
        and value['discard_unfinished_row_approved'] is True and value['execution_authorized'] is False
        and value['no_future_writers'] is True and value['no_future_readers'] is True
        and value['decision_digest'] == canonical_digest(value, digest_field='decision_digest')
        and all(type(value[key]) in (int, float) and math.isfinite(value[key])
            for key in ('issued_at_epoch', 'expires_at_epoch'))
        and value['issued_at_epoch'] < value['expires_at_epoch']
            <= min(value['issued_at_epoch'] + 900, selected[1]['expires_at_epoch'])
        and value['issued_at_epoch'] <= moment)
    observed, manifest_raw = store.read(identifier, manifest=True)
    _require(authority._selector(manifest_raw) == value['observed_manifest'])
    action_id = selected[1]['action_id']
    packet = _packet(selected, events, observed, action_id,
        _event_selectors(files, config, action_id, events), tick=files.budget.tick)
    _require(value['packet'] == packet and value['packet_digest'] == packet['packet_digest']
        and identifier == decision_id(packet['action_id'], packet['original_head_event_digest'])
        and value['owner'] == selected[1]['owner']
        and (value['installed_config'], value['policy']) ==
            (selected[1]['installed_config'], selected[1]['policy']))
    if consumed is None:
        _require(_current_delete(files, config, config_path, value['principal'], value['owner'], moment,
            value['expires_at_epoch'], selected[1]['expires_at_epoch']) ==
            (value['installed_config'], value['policy']))
    else:
        intent, completed = consumed
        expected = dict(decision_id=identifier, decision=authority._selector(raw),
            original_head_event_digest=events[-1]['event_digest'])
        _require(intent['body'] == dict(phase='reconcile_intent', **expected)
            and completed['kind'] == intent['kind'] == 'restore_intent'
            and completed['body']['phase'] == 'reconciled'
            and all(completed['body'][key] == item for key, item in expected.items())
            and completed['previous_event_digest'] == intent['event_digest']
            and value['issued_at_epoch'] <= intent['observed_at_epoch']
                <= completed['observed_at_epoch'] < value['expires_at_epoch'])
        # The exact immutable policy still belongs to the current original
        # restore. Prove this principal's DELETE grant at issuance; only its
        # continued TTL is unnecessary after an authenticated consumed effect.
        _require(_current_delete(files, config, config_path, value['principal'], value['owner'],
            value['issued_at_epoch'], value['expires_at_epoch'], selected[1]['expires_at_epoch'])
            == (value['installed_config'], value['policy']))
    return value, observed, authority._selector(raw)


def approve_historical_restore_reconciliation(*, installed_config_path, action_id, ack_packet_digest,
        principal, owner, discard_unfinished_row, no_future_writers, no_future_readers,
        expires_at_epoch, now, monotonic=time.monotonic):
    _require(discard_unfinished_row is True and no_future_writers is True and no_future_readers is True)
    operation = authority._Operation(now, monotonic)
    selected, events, observed, packet = _observe(installed_config_path, action_id, operation)
    _require(packet['packet_digest'] == ack_packet_digest and owner == packet['owner'])
    identifier = decision_id(action_id, packet['original_head_event_digest'])
    with authority._session(installed_config_path, operation) as (files, config, store):
        _require(restore.select_restore(files, config, store, installed_config_path, action_id,
            operation.moment()) == selected
            and HistoricalJournalObservation(files, config, selected, operation).head['event_digest']
                == packet['original_head_event_digest'])
        configured, policy = _current_delete(files, config, installed_config_path, principal, owner,
            operation.moment(), expires_at_epoch, selected[1]['expires_at_epoch'])
        generation.verify_historical_member_versions(observed, tick=files.budget.tick)
        try:
            previous, _ = store.read(identifier)
        except FileNotFoundError:
            previous = None
        if previous is not None:
            value, _, _ = select_reconciliation(files, config, store, installed_config_path,
                selected, events, identifier, None, operation.moment())
            _require(value['principal'] == principal and value['expires_at_epoch'] == expires_at_epoch)
            return value
        try:
            existing, raw = store.read(identifier, manifest=True)
        except FileNotFoundError:
            manifest_selector = store.publish(identifier, observed, manifest=True)
        else:
            _require(existing == observed)
            manifest_selector = authority._selector(raw)
        value = dict(schema_version=SCHEMA, decision_id=identifier, action='discard_unfinished_restore_row',
            principal=principal, owner=owner, packet=packet, packet_digest=ack_packet_digest,
            observed_manifest=manifest_selector, installed_config=configured, policy=policy,
            issued_at_epoch=operation.moment(), expires_at_epoch=expires_at_epoch,
            discard_unfinished_row_approved=True, no_future_writers=True, no_future_readers=True,
            execution_authorized=False)
        value['decision_digest'] = canonical_digest(value, digest_field='decision_digest')
        store.publish(identifier, value)
        return value
