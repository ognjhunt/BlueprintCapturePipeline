"""Non-authorizing observation after an exact pending discard became absent.

This is no new DELETE grant, creation receipt or freed-byte measurement. An
explicit new current owner observation decision and all original restore gates
are required before durability observation and same-operation continuation.
"""
from __future__ import annotations

import copy
import re

from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_restore_reconciliation_replay import binding, reconciliation_bindings
from .control_plane_lane_historical_restore_reconciliation_scope import unknown_creation_scope, reconciled_parent
from .control_plane_lane_historical_restore_limits import RestoreObservationBounds


def absent_observation_scope(original, approved, observed, decision, events, action_id, *, tick=lambda: None):
    generation._require(type(events) is list and len(events) >= 2
        and events[-1]['kind'] == 'restore_intent', 'restore_absent_observation_invalid')
    event = events[-1]
    selected = binding(event['body'])
    pending = reconciliation_bindings(events)
    generation._require(pending and pending[-1][3][-1] == event
        and event['body'].get('phase') in ('reconcile_intent', 'reconcile_delete_resume'), 'restore_absent_observation_invalid')
    prefix = pending[-1][0]
    generation._require(selected['original_head_event_digest'] == prefix[-1]['event_digest']
        and type(event.get('event_digest')) is str
        and re.fullmatch('sha256:[0-9a-f]{64}', event['event_digest']), 'restore_absent_observation_invalid')
    scope = unknown_creation_scope(original, approved, decision, prefix, action_id, tick=tick)
    parent = reconciled_parent(approved, observed, scope, tick=tick,
        restore_bounds=RestoreObservationBounds(original, action_id))
    return copy.deepcopy(dict(execution_authorized=False, observation_only=True, permits_removal=False,
        action_id=action_id, original_generation_digest=original['generation_digest'],
        original_reconciliation_intent_digest=event['event_digest'],
        original_reconciliation=selected, original_approved_generation_digest=approved['generation_digest'],
        observed_generation_digest=observed['generation_digest'], parent_path=scope['parent_path'],
        parent_version=parent, credited_removed_allocated_bytes=0))
