"""Current owner observation of an absent unfinished row; never renewed DELETE.

The old immutable discard approval and authentic pending intent remain
historical facts. This separate decision permits only durability observation
under the original restore, with no replacement birth or freed-byte credit.
"""
from __future__ import annotations

from .control_plane_lane_historical_restore_limits import RestoreObservationBounds

import hashlib
import math
import time

from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_historical_restore_authority as restore
from . import control_plane_lane_historical_restore_reconciliation_authority as discard
from .control_plane_lane_historical_journal import HistoricalJournalObservation
from .control_plane_lane_historical_restore_absent_observation import absent_observation_scope
from .control_plane_lane_historical_restore_reconciliation_replay import binding, reconciliation_bindings, observation_binding
from .decision_evidence_contracts import canonical_digest

SCHEMA = 'control_plane_historical_restore_absent_decision.v1'
_FIELDS = frozenset({'schema_version', 'decision_id', 'action', 'principal', 'owner', 'packet',
    'packet_digest', 'observed_manifest', 'installed_config', 'policy', 'issued_at_epoch',
    'expires_at_epoch', 'absence_observation_approved', 'no_future_writers', 'no_future_readers',
    'execution_authorized', 'decision_digest', 'attempt'})


def _require(value):
    generation._require(value, 'restore_absent_observation_approval_invalid')


def decision_id(action_id, intent_digest, attempt=0):
    _require(type(action_id) is str and authority._ID.fullmatch(action_id)
        and type(intent_digest) is str and len(intent_digest) == 71 and intent_digest.startswith('sha256:')
        and type(attempt) is int and 0 <= attempt < 8)
    return hashlib.sha256(('restore-observe-absence.v1:' + action_id + ':' + intent_digest + ':' + str(attempt)).encode()).hexdigest()[:32]


def _proposal(series, moment):
    """Only an expired latest observation permits an explicit new attempt."""
    index = len(series) - int(bool(series) and moment < series[-1][0]['expires_at_epoch'])
    _require(index < 8)
    return index, [dict(decision_id=value['decision_id'], decision=selector) for value, selector in series[:index]]


def _series(files, config, store, config_path, selected, intent, moment):
    """Finite full lookup; gaps, changed roots and unauthorized records refuse."""
    values, gap = [], False
    for index in range(8):
        identifier = decision_id(selected[1]['action_id'], intent['event_digest'], index)
        try:
            value, raw = store.read(identifier)
        except FileNotFoundError:
            gap = True
            continue
        _require(not gap and set(value) == _FIELDS and value['schema_version'] == SCHEMA
            and value['decision_id'] == identifier and type(value['attempt']) is int and value['attempt'] == index
            and value['decision_digest'] == canonical_digest(value, digest_field='decision_digest')
            and value['owner'] == selected[1]['owner'] and value['execution_authorized'] is False
            and value['action'] == 'observe_unfinished_restore_row_absence'
            and value['absence_observation_approved'] is True
            and value['no_future_writers'] is value['no_future_readers'] is True
            and all(type(value[key]) in (int, float) and math.isfinite(value[key])
                for key in ('issued_at_epoch', 'expires_at_epoch'))
            and intent['observed_at_epoch'] <= value['issued_at_epoch'] <= moment
            and value['issued_at_epoch'] < value['expires_at_epoch']
                <= min(value['issued_at_epoch'] + 900, selected[1]['expires_at_epoch'])
            and (not values or values[-1][0]['expires_at_epoch'] <= value['issued_at_epoch']))
        _require(_current_observation(files, config, config_path, value['principal'], value['owner'],
            value['issued_at_epoch'], value['expires_at_epoch'], selected[1]['expires_at_epoch']) ==
            (value['installed_config'], value['policy']) == (selected[1]['installed_config'], selected[1]['policy']))
        packet = value['packet']
        _require(type(packet) is dict and packet['packet_digest'] == value['packet_digest']
            == canonical_digest(packet, digest_field='packet_digest')
            and packet['action_id'] == selected[1]['action_id'] and packet['owner'] == selected[1]['owner']
            and packet['source_records'] == selected[3]
            and packet['original_head_event_digest'] == intent['event_digest']
            and packet['original_expires_at_epoch'] == selected[1]['expires_at_epoch']
            and packet['execution_authorized'] is packet['permits_removal'] is False
            and packet['observation_only'] is True)
        prior = [dict(decision_id=previous['decision_id'], decision=selector) for previous, selector in values]
        _require(value['packet']['attempt'] == index and value['packet']['prior_observations'] == prior)
        values.append((value, authority._selector(raw)))
    return values


def _current_observation(files, config, config_path, principal, owner, moment, expiry, original_expiry):
    _require(type(expiry) in (int, float) and math.isfinite(expiry) and expiry <= original_expiry)
    return restore._current_authority(files, config, config_path, principal, owner, moment, expiry)


def _packet(selected, events, observed, old, byte_selectors, *, tick, attempt=0, prior=()):
    scope = absent_observation_scope(selected[2], old[1], observed, selected[1], events,
                                     selected[1]['action_id'], tick=tick)
    value = dict(schema_version='control_plane_historical_restore_absent_packet.v1',
        action_id=selected[1]['action_id'], owner=selected[1]['owner'], source_records=selected[3],
        original_intent=events[0], original_head_event_digest=events[-1]['event_digest'],
        original_intent_bytes=byte_selectors[0], original_head_bytes=byte_selectors[1],
        original_expires_at_epoch=selected[1]['expires_at_epoch'], scope=scope,
        execution_authorized=False, observation_only=True, permits_removal=False,
        attempt=attempt, prior_observations=list(prior))
    value['packet_digest'] = canonical_digest(value, digest_field='packet_digest')
    return value


def _build_packet(files, config, store, config_path, selected, events, observed, byte_selectors, moment):
    _require(len(events) >= 2 and events[-1]['kind'] == 'restore_intent'
        and events[-1]['body'].get('phase') in ('reconcile_intent', 'reconcile_delete_resume'))
    pending = reconciliation_bindings(events)[-1]
    _, old = discard.historical_effect_grant(files, config, store, config_path, selected,
        *pending, moment)
    attempt, prior = _proposal(_series(files, config, store, config_path, selected, events[-1], moment), moment)
    return _packet(selected, events, observed, old, byte_selectors, tick=files.budget.tick, attempt=attempt, prior=prior)


def observe_historical_restore_absence(*, installed_config_path, action_id, now, monotonic=time.monotonic):
    """Read-only full current absence packet with original raw e0/head bindings."""
    return discard._observe(installed_config_path, action_id, authority._Operation(now, monotonic),
                            build_packet=_build_packet)[3]


def _observation_index(series, completed):
    _require(bool(series))
    if completed is None:
        return len(series) - 1
    resume = observation_binding(completed['body'].get('observation_resume'))
    indices = [index for index, (value, raw) in enumerate(series)
        if resume == dict(decision_id=value['decision_id'], decision=raw)]
    _require(len(indices) == 1)
    return indices[0]


def select_absent_observation(files, config, store, config_path, selected, events, old, moment,
        *, selector=None, completed=None):
    """Authenticate observation only; caller must prohibit every discard syscall."""
    series = _series(files, config, store, config_path, selected, events[-1], moment)
    if not series:
        raise generation.HistoricalGenerationError(
            'historical_generation_restore_absent_observation_approval_missing') from None
    index = _observation_index(series, completed)
    value, selected_raw = series[index]
    identifier = value['decision_id']
    value, raw = store.read(identifier)
    _require(authority._selector(raw) == selected_raw)
    _require((selector is None or authority._selector(raw) == selector)
        and set(value) == _FIELDS and value['schema_version'] == SCHEMA
        and value['decision_id'] == identifier and value['action'] == 'observe_unfinished_restore_row_absence'
        and value['absence_observation_approved'] is True and value['execution_authorized'] is False
        and value['no_future_writers'] is True and value['no_future_readers'] is True
        and value['decision_digest'] == canonical_digest(value, digest_field='decision_digest')
        and all(type(value[key]) in (int, float) and math.isfinite(value[key])
            for key in ('issued_at_epoch', 'expires_at_epoch'))
        and events[-1]['observed_at_epoch'] <= value['issued_at_epoch'] <= moment
        and value['issued_at_epoch'] < value['expires_at_epoch']
            <= min(value['issued_at_epoch'] + 900, selected[1]['expires_at_epoch']))
    observed, manifest_raw = store.read(identifier, manifest=True)
    _require(authority._selector(manifest_raw) == value['observed_manifest'])
    packet = _packet(selected, events, observed, old,
        discard._event_selectors(files, config, selected[1]['action_id'], events), tick=files.budget.tick,
        attempt=value['attempt'], prior=[dict(decision_id=previous['decision_id'], decision=selector)
                                       for previous, selector in series[:index]])
    _require(value['packet'] == packet and value['packet_digest'] == packet['packet_digest']
        and value['owner'] == selected[1]['owner']
        and (value['installed_config'], value['policy']) ==
            (selected[1]['installed_config'], selected[1]['policy']))
    if completed is None:
        moment_for_rights = moment
        generation.verify_historical_member_versions(observed, tick=files.budget.tick,
            _restore_bounds=RestoreObservationBounds(selected[2], selected[1]['action_id']))
    else:
        expected = dict(decision_id=identifier, decision=authority._selector(raw))
        _require(completed['kind'] == 'restore_intent' and completed['body'].get('phase') == 'reconciled'
            and completed['body'].get('observation_resume') == expected
            and binding(completed['body']) == binding(events[-1]['body'])
            and completed['previous_event_digest'] == events[-1]['event_digest']
            and events[-1]['observed_at_epoch'] <= value['issued_at_epoch']
                <= completed['observed_at_epoch'] < value['expires_at_epoch']
            and completed['body']['parent_path'] == packet['scope']['parent_path']
            and completed['body']['parent_version'] == packet['scope']['parent_version']
            and completed['body']['uncertain'] is True
            and type(completed['body']['credited_removed_allocated_bytes']) is int
            and completed['body']['credited_removed_allocated_bytes'] == 0)
        moment_for_rights = value['issued_at_epoch']
    _require(_current_observation(files, config, config_path, value['principal'], value['owner'],
        moment_for_rights, value['expires_at_epoch'], selected[1]['expires_at_epoch']) ==
        (value['installed_config'], value['policy']))
    return value, observed, authority._selector(raw)


def approve_historical_restore_absence(*, installed_config_path, action_id, ack_packet_digest,
        principal, owner, observe_absence_only, no_future_writers, no_future_readers,
        expires_at_epoch, now, monotonic=time.monotonic):
    _require(observe_absence_only is True and no_future_writers is True and no_future_readers is True)
    operation = authority._Operation(now, monotonic)
    selected, events, observed, packet = discard._observe(installed_config_path, action_id, operation,
                                                         build_packet=_build_packet)
    _require(packet['packet_digest'] == ack_packet_digest and owner == packet['owner'])
    identifier = decision_id(action_id, packet['original_head_event_digest'], packet['attempt'])
    with authority._session(installed_config_path, operation) as (files, config, store):
        _require(restore.select_restore(files, config, store, installed_config_path, action_id,
            operation.moment()) == selected
            and HistoricalJournalObservation(files, config, selected, operation).head['event_digest']
                == packet['original_head_event_digest'])
        _require(_proposal(_series(files, config, store, installed_config_path, selected, events[-1],
            operation.moment()), operation.moment()) == (packet['attempt'], packet['prior_observations']))
        configured, policy = _current_observation(files, config, installed_config_path, principal, owner,
            operation.moment(), expires_at_epoch, selected[1]['expires_at_epoch'])
        generation.verify_historical_member_versions(observed, tick=files.budget.tick,
            _restore_bounds=RestoreObservationBounds(selected[2], selected[1]['action_id']))
        pending = reconciliation_bindings(events)[-1]
        _, old = discard.historical_effect_grant(files, config, store, installed_config_path, selected,
            *pending, operation.moment())
        try:
            previous, _ = store.read(identifier)
        except FileNotFoundError:
            previous = None
        if previous is not None:
            value, _, _ = select_absent_observation(files, config, store, installed_config_path,
                selected, events, old, operation.moment())
            _require(value['principal'] == principal and value['expires_at_epoch'] == expires_at_epoch)
            return value
        try:
            existing, raw = store.read(identifier, manifest=True)
        except FileNotFoundError:
            manifest_selector = store.publish(identifier, observed, manifest=True)
        else:
            _require(existing == observed)
            manifest_selector = authority._selector(raw)
        value = dict(schema_version=SCHEMA, decision_id=identifier, action='observe_unfinished_restore_row_absence',
            principal=principal, owner=owner, packet=packet, packet_digest=ack_packet_digest,
            observed_manifest=manifest_selector, installed_config=configured, policy=policy,
            issued_at_epoch=operation.moment(), expires_at_epoch=expires_at_epoch,
            absence_observation_approved=True, no_future_writers=True, no_future_readers=True,
            execution_authorized=False, attempt=packet['attempt'])
        value['decision_digest'] = canonical_digest(value, digest_field='decision_digest')
        store.publish(identifier, value)
        return value


def select_worker_reconciliation(files, config, store, config_path, selected, events, identifier,
        selector, moment, *, recorded=None):
    """Fresh effects never reinterpret the original intent as a later grant."""
    if recorded is None:
        value = (discard.latest_reconciliation(files, config, store, config_path, selected, events, moment)
            if selector is None else discard.select_reconciliation(files, config, store, config_path,
                selected, events, identifier, selector, moment))
        return value, None, value
    original, effect = discard.historical_effect_grant(files, config, store, config_path, selected,
        events, identifier, selector, recorded, moment)
    complete = recorded[-1] if recorded[-1]['body'].get('phase') == 'reconciled' else None
    pending = list(recorded[:-1] if complete is not None else recorded)
    if complete is not None and 'observation_resume' not in complete['body']:
        consumed = discard.select_reconciliation(files, config, store, config_path, selected, events,
            effect[0]['decision_id'], effect[2], moment, consumed=(pending[-1], complete))
        return original, consumed if len(pending) > 1 else None, consumed
    resume = complete['body']['observation_resume'] if complete is not None else None
    try:
        observation = select_absent_observation(files, config, store, config_path, selected,
            events + pending, effect, moment, selector=resume['decision'] if resume is not None else None,
            completed=complete)
    except generation.HistoricalGenerationError as error:
        if str(error) != 'historical_generation_restore_absent_observation_approval_missing' or complete is not None:
            raise
        series = discard._attempt_series(files, config, store, config_path, selected, events, moment)
        if moment >= effect[0]['expires_at_epoch'] and series[-1][0]['decision_id'] == effect[0]['decision_id']:
            raise
        latest = discard.latest_reconciliation(files, config, store, config_path, selected, events, moment)
        return original, latest if latest[0]['decision_id'] != original[0]['decision_id'] else None, latest
    if resume is not None:
        _require(resume == dict(decision_id=observation[0]['decision_id'], decision=observation[2]))
    return original, observation, effect
