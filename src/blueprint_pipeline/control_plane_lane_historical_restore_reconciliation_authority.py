"""Distinct owner DELETE decision for one unfinished original restore row.

Review and approval do not remove bytes, adopt a creation, clear references,
renew the original restore permission or start a new operation clock.
"""
from __future__ import annotations

from .control_plane_lane_historical_restore_limits import RestoreObservationBounds

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
from .control_plane_lane_historical_journal import HistoricalJournalObservation, journal_root, MAX_SUPPORTED_EVENT_BYTES as MAX_EVENT_BYTES
from .control_plane_lane_historical_restore_reconciliation_scope import unknown_creation_scope
from .decision_evidence_contracts import canonical_digest

SCHEMA = 'control_plane_historical_restore_reconciliation_decision.v2'
_LEGACY_SCHEMA = 'control_plane_historical_restore_reconciliation_decision.v1'
_FIELDS = frozenset({'schema_version', 'decision_id', 'action', 'principal', 'owner', 'packet',
    'packet_digest', 'observed_manifest', 'installed_config', 'policy', 'issued_at_epoch',
    'expires_at_epoch', 'discard_unfinished_row_approved', 'no_future_writers', 'no_future_readers',
    'execution_authorized', 'decision_digest', 'attempt'})
_LEGACY_FIELDS = _FIELDS - {'attempt'}


def _require(value):
    generation._require(value, 'restore_reconciliation_approval_invalid')


def decision_id(action_id, original_head, attempt=0):
    """Fixed lookup from the original operation and exact interrupted head."""
    _require(type(action_id) is str and authority._ID.fullmatch(action_id)
        and type(original_head) is str and len(original_head) == 71 and original_head.startswith('sha256:')
        and type(attempt) is int and 0 <= attempt < 8)
    suffix = '' if attempt == 0 else ':' + str(attempt)
    return hashlib.sha256(('restore-reconcile.v1:' + action_id + ':' + original_head + suffix).encode()).hexdigest()[:32]


def _attempt_proposal(series, moment):
    """Fresh explicit approval only after the latest immutable grant expires."""
    index = len(series) - int(bool(series) and moment < series[-1][0]['expires_at_epoch'])
    _require(index < 8)
    return index, [dict(decision_id=value['decision_id'], decision=raw) for value, raw in series[:index]]


def _attempt_series(files, config, store, config_path, selected, events, moment):
    """Authenticate all finite previous grants without granting current DELETE."""
    series, gap = [], False
    for index in range(8):
        identifier = decision_id(selected[1]['action_id'], events[-1]['event_digest'], index)
        try:
            value, raw = store.read(identifier)
        except FileNotFoundError:
            gap = True
            continue
        legacy_value = value.get('schema_version') == _LEGACY_SCHEMA
        _require(not gap and set(value) == (_LEGACY_FIELDS if legacy_value else _FIELDS)
            and (legacy_value and index == 0 or value.get('schema_version') == SCHEMA
                and type(value.get('attempt')) is int and value['attempt'] == index)
            and value['decision_id'] == identifier and value['owner'] == selected[1]['owner']
            and value['action'] == 'discard_unfinished_restore_row'
            and value['discard_unfinished_row_approved'] is True
            and value['no_future_writers'] is value['no_future_readers'] is True
            and value['execution_authorized'] is False
            and value['decision_digest'] == canonical_digest(value, digest_field='decision_digest')
            and all(type(value[key]) in (int, float) and math.isfinite(value[key])
                for key in ('issued_at_epoch', 'expires_at_epoch'))
            and events[-1]['observed_at_epoch'] <= value['issued_at_epoch'] <= moment
            and value['issued_at_epoch'] < value['expires_at_epoch']
                <= min(value['issued_at_epoch'] + 900, selected[1]['expires_at_epoch'])
            and (not series or series[-1][0]['expires_at_epoch'] <= value['issued_at_epoch']))
        _require(_current_delete(files, config, config_path, value['principal'], value['owner'],
            value['issued_at_epoch'], value['expires_at_epoch'], selected[1]['expires_at_epoch']) ==
            (value['installed_config'], value['policy']) == (selected[1]['installed_config'], selected[1]['policy']))
        packet = value['packet']
        _require(type(packet) is dict and packet['packet_digest'] == value['packet_digest']
            == canonical_digest(packet, digest_field='packet_digest')
            and packet['action_id'] == selected[1]['action_id'] and packet['owner'] == selected[1]['owner']
            and packet['source_records'] == selected[3]
            and packet['original_head_event_digest'] == events[-1]['event_digest']
            and packet['original_expires_at_epoch'] == selected[1]['expires_at_epoch']
            and packet['execution_authorized'] is False)
        if not legacy_value:
            _require(packet['attempt'] == index and packet['prior_decisions'] == [
                dict(decision_id=previous['decision_id'], decision=selector) for previous, selector in series])
        series.append((value, authority._selector(raw)))
    return series


def _attempt_packet(packet, attempt, prior, resume_from=None):
    value = dict(packet, attempt=attempt, prior_decisions=prior, resume_from=resume_from)
    value['schema_version'] = 'control_plane_historical_restore_reconciliation_packet.v2'
    value['packet_digest'] = canonical_digest(value, digest_field='packet_digest')
    return value


def _pending_prefix(events):
    if events[-1]['kind'] == 'restore_intent' and events[-1]['body'].get('phase') in (
            'reconcile_intent', 'reconcile_delete_resume'):
        from .control_plane_lane_historical_restore_reconciliation_replay import reconciliation_bindings
        values = reconciliation_bindings(events)
        _require(values and values[-1][3][-1]['body']['phase'] != 'reconciled')
        return values[-1][0]
    return events


def _resume_selector(files, config, action_id, event):
    raw, record = files.read(journal_root(config) / action_id / f"e-{event['sequence']:05d}.json",
                            cap=MAX_EVENT_BYTES, protected=True, mode=0o600)
    _require(retained._document(raw, MAX_EVENT_BYTES, _work_budget=files.budget) == event)
    files.verify_record(record)
    return dict(sequence=event['sequence'], event_digest=event['event_digest'], event_bytes=authority._selector(raw))


def _build_packet(files, config, store, config_path, selected, events, observed, byte_selectors, moment):
    prefix = _pending_prefix(events)
    if prefix is not events:
        from .control_plane_lane_historical_restore_reconciliation_replay import reconciliation_bindings
        pending = reconciliation_bindings(events)[-1]
        historical_effect_grant(files, config, store, config_path, selected, *pending, moment)
    attempt, prior = _attempt_proposal(_attempt_series(files, config, store, config_path, selected, prefix, moment), moment)
    base = _packet(selected, prefix, observed, selected[1]['action_id'],
                   _event_selectors(files, config, selected[1]['action_id'], prefix), tick=files.budget.tick)
    return _attempt_packet(base, attempt, prior,
        _resume_selector(files, config, selected[1]['action_id'], events[-1]) if prefix is not events else None)


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


def _observe(config_path, action_id, operation, *, build_packet=None):
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
        max_seconds=operation.remaining(), monotonic=operation.monotonic,
        _restore_bounds=RestoreObservationBounds(selected[2], action_id))
    packet = None
    with authority._session(config_path, operation) as (files, config, store):
        _require(restore.select_restore(files, config, store, config_path, action_id, operation.moment()) == selected
            and HistoricalJournalObservation(files, config, selected, operation).head['event_digest'] == head)
        _require(_event_selectors(files, config, action_id, events) == byte_selectors)
        generation.verify_historical_member_versions(observed, tick=files.budget.tick,
            _restore_bounds=RestoreObservationBounds(selected[2], selected[1]['action_id']))
        if build_packet is None:
            build_packet = _build_packet
        if build_packet is not None:
            packet = build_packet(files, config, store, config_path, selected, events, observed,
                                  byte_selectors, operation.moment())
    return selected, events, observed, packet


def observe_historical_restore_reconciliation(*, installed_config_path, action_id, now,
        monotonic=time.monotonic):
    """Read-only exact owner packet; no original journal creation or append."""
    return _observe(installed_config_path, action_id, authority._Operation(now, monotonic))[3]


def select_reconciliation(files, config, store, config_path, selected, events, identifier, selector, moment,
        *, consumed=None, historical_intent=None):
    """Fresh protected DELETE approval; current physical gates remain separate."""
    try:
        value, raw = store.read(identifier)
    except FileNotFoundError:
        # Classify absence while still inside the protected session. Its
        # OSError boundary must continue to refuse every other IO failure.
        raise generation.HistoricalGenerationError(
            'historical_generation_restore_reconciliation_approval_missing') from None
    _require((selector is None or authority._selector(raw) == selector)
        and ((set(value) == _FIELDS and value['schema_version'] == SCHEMA)
            or (set(value) == _LEGACY_FIELDS and value['schema_version'] == _LEGACY_SCHEMA))
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
    if value['schema_version'] == SCHEMA:
        series = _attempt_series(files, config, store, config_path, selected, events, moment)
        index = value['attempt']
        _require(type(index) is int and 0 <= index < len(series) and series[index][1] == authority._selector(raw))
        resume = value['packet']['resume_from']
        if resume is not None:
            _require(type(resume) is dict and set(resume) == {'sequence', 'event_digest', 'event_bytes'}
                and type(resume['sequence']) is int and events[-1]['sequence'] < resume['sequence'])
            resume_raw, _ = files.read(journal_root(config) / action_id / f"e-{resume['sequence']:05d}.json",
                cap=MAX_EVENT_BYTES, protected=True, mode=0o600)
            resume_event = retained._document(resume_raw, MAX_EVENT_BYTES, _work_budget=files.budget)
            _require(_resume_selector(files, config, action_id, resume_event) == resume
                and resume_event['kind'] == 'restore_intent'
                and resume_event['body'].get('phase') in ('reconcile_intent', 'reconcile_delete_resume')
                and resume_event['action_id'] == action_id
                and resume_event['observed_at_epoch'] <= value['issued_at_epoch'])
        packet = _attempt_packet(packet, index, [dict(decision_id=previous['decision_id'], decision=selector)
            for previous, selector in series[:index]], resume)
    else:
        index = 0
    _require(value['packet'] == packet and value['packet_digest'] == packet['packet_digest']
        and identifier == decision_id(packet['action_id'], packet['original_head_event_digest'], index)
        and value['owner'] == selected[1]['owner']
        and (value['installed_config'], value['policy']) ==
            (selected[1]['installed_config'], selected[1]['policy']))
    _require(consumed is None or historical_intent is None)
    if consumed is None and historical_intent is None:
        _require(_current_delete(files, config, config_path, value['principal'], value['owner'], moment,
            value['expires_at_epoch'], selected[1]['expires_at_epoch']) ==
            (value['installed_config'], value['policy']))
    elif consumed is not None:
        intent, completed = consumed
        intent_binding = _grant_event(value, authority._selector(raw), events, intent)
        _require(intent_binding
            and completed['kind'] == intent['kind'] == 'restore_intent'
            and completed['body']['phase'] == 'reconciled'
            and all(completed['body'][key] == item for key, item in intent_binding.items())
            and completed['body'].get('delete_resume') == intent['body'].get('delete_resume')
            and completed['previous_event_digest'] == intent['event_digest']
            and value['issued_at_epoch'] <= intent['observed_at_epoch']
                <= completed['observed_at_epoch'] < value['expires_at_epoch'])
        # The exact immutable policy still belongs to the current original
        # restore. Prove this principal's DELETE grant at issuance; only its
        # continued TTL is unnecessary after an authenticated consumed effect.
        _require(_current_delete(files, config, config_path, value['principal'], value['owner'],
            value['issued_at_epoch'], value['expires_at_epoch'], selected[1]['expires_at_epoch'])
            == (value['installed_config'], value['policy']))
    else:
        # Historical fact selection supplies no current removal authority.
        # Only a separately protected, current observation decision can use
        # this mode on the worker, and only for an already absent row.
        _grant_event(value, authority._selector(raw), events, historical_intent)
        _require(value['issued_at_epoch'] <= historical_intent['observed_at_epoch'] < value['expires_at_epoch'])
        _require(_current_delete(files, config, config_path, value['principal'], value['owner'],
            value['issued_at_epoch'], value['expires_at_epoch'], selected[1]['expires_at_epoch'])
            == (value['installed_config'], value['policy']))
    return value, observed, authority._selector(raw)


def _grant_event(value, raw_selector, prefix, event):
    """A real issued intent or resume selects one exact grant, never latest."""
    from .control_plane_lane_historical_restore_reconciliation_replay import binding, observation_binding
    _require(event['kind'] == 'restore_intent')
    base = binding(event['body'])
    _require(base['original_head_event_digest'] == prefix[-1]['event_digest'])
    if event['body'].get('phase') == 'reconcile_intent':
        _require(event['body'] == dict(phase='reconcile_intent', **base)
            and base['decision_id'] == value['decision_id'] and base['decision'] == raw_selector
            and event['previous_event_digest'] == prefix[-1]['event_digest']
            and value['packet'].get('resume_from') is None)
    else:
        grant = observation_binding(event['body'].get('delete_resume'))
        _require(event['body'] == dict(phase='reconcile_delete_resume', **base, delete_resume=grant)
            and grant == dict(decision_id=value['decision_id'], decision=raw_selector)
            and value['packet']['resume_from'] is not None
            and event['previous_event_digest'] == value['packet']['resume_from']['event_digest']
            and event['sequence'] == value['packet']['resume_from']['sequence'] + 1)
    return base


def latest_reconciliation(files, config, store, config_path, selected, events, moment):
    series = _attempt_series(files, config, store, config_path, selected, events, moment)
    if not series:
        raise generation.HistoricalGenerationError('historical_generation_restore_reconciliation_approval_missing')
    value, raw = series[-1]
    return select_reconciliation(files, config, store, config_path, selected, events,
        value['decision_id'], raw, moment)


def historical_effect_grant(files, config, store, config_path, selected, prefix, identifier, selector, recorded, moment):
    """Authenticate the original intent and EACH immutable real resume event."""
    _require(type(recorded) is tuple and 1 <= len(recorded) <= 9)
    original = select_reconciliation(files, config, store, config_path, selected, prefix,
        identifier, selector, moment, historical_intent=recorded[0])
    effect = original
    from .control_plane_lane_historical_restore_reconciliation_replay import binding, observation_binding
    base = binding(recorded[0]['body'])
    previous = recorded[0]
    resumes = recorded[1:-1] if recorded[-1]['body'].get('phase') == 'reconciled' else recorded[1:]
    for event in resumes:
        _require(binding(event['body']) == base and event['previous_event_digest'] == previous['event_digest'])
        grant = observation_binding(event['body'].get('delete_resume'))
        effect = select_reconciliation(files, config, store, config_path, selected, prefix,
            grant['decision_id'], grant['decision'], moment, historical_intent=event)
        previous = event
    return original, effect


def approve_historical_restore_reconciliation(*, installed_config_path, action_id, ack_packet_digest,
        principal, owner, discard_unfinished_row, no_future_writers, no_future_readers,
        expires_at_epoch, now, monotonic=time.monotonic):
    _require(discard_unfinished_row is True and no_future_writers is True and no_future_readers is True)
    operation = authority._Operation(now, monotonic)
    selected, events, observed, packet = _observe(installed_config_path, action_id, operation)
    _require(packet['packet_digest'] == ack_packet_digest and owner == packet['owner'])
    identifier = decision_id(action_id, packet['original_head_event_digest'], packet['attempt'])
    with authority._session(installed_config_path, operation) as (files, config, store):
        _require(restore.select_restore(files, config, store, installed_config_path, action_id,
            operation.moment()) == selected
            and HistoricalJournalObservation(files, config, selected, operation).head['event_digest']
                == (packet['resume_from']['event_digest'] if packet['resume_from'] is not None
                    else packet['original_head_event_digest']))
        prefix = _pending_prefix(events)
        _require(_attempt_proposal(_attempt_series(files, config, store, installed_config_path, selected, prefix,
            operation.moment()), operation.moment()) == (packet['attempt'], packet['prior_decisions']))
        configured, policy = _current_delete(files, config, installed_config_path, principal, owner,
            operation.moment(), expires_at_epoch, selected[1]['expires_at_epoch'])
        generation.verify_historical_member_versions(observed, tick=files.budget.tick,
            _restore_bounds=RestoreObservationBounds(selected[2], selected[1]['action_id']))
        try:
            previous, _ = store.read(identifier)
        except FileNotFoundError:
            previous = None
        if previous is not None:
            value, _, _ = select_reconciliation(files, config, store, installed_config_path,
                selected, prefix, identifier, None, operation.moment())
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
            execution_authorized=False, attempt=packet['attempt'])
        value['decision_digest'] = canonical_digest(value, digest_field='decision_digest')
        store.publish(identifier, value)
        return value
