"""Structural reconciliation replay; never authenticates approval or births."""
from __future__ import annotations

import re

from . import control_plane_lane_historical_generation as generation


def _require(value):
    generation._require(value, 'restore_stage_changed')


def binding(body):
    _require(type(body) is dict and {'decision_id', 'decision', 'original_head_event_digest'} <= body.keys())
    selected = {key: body[key] for key in ('decision_id', 'decision', 'original_head_event_digest')}
    _require(type(selected['decision_id']) is str and re.fullmatch('[0-9a-f]{32}', selected['decision_id'])
        and type(selected['decision']) is dict and set(selected['decision']) == {'sha256', 'size_bytes'}
        and type(selected['decision']['size_bytes']) is int and 0 < selected['decision']['size_bytes'] <= 32768
        and all(type(digest) is str and re.fullmatch('sha256:[0-9a-f]{64}', digest)
            for digest in (selected['decision']['sha256'], selected['original_head_event_digest'])))
    return selected


def observation_binding(value):
    _require(type(value) is dict and set(value) == {'decision_id', 'decision'}
        and type(value['decision_id']) is str and re.fullmatch('[0-9a-f]{32}', value['decision_id'])
        and type(value['decision']) is dict and set(value['decision']) == {'sha256', 'size_bytes'}
        and type(value['decision']['size_bytes']) is int and 0 < value['decision']['size_bytes'] <= 32768
        and type(value['decision']['sha256']) is str
        and re.fullmatch('sha256:[0-9a-f]{64}', value['decision']['sha256']))
    return value.copy()


def parent_observation(body, pending, selected, stage, versions):
    parent = pending['stage_path'].rpartition('/')[0]
    extra = ({'observation_resume': observation_binding(body['observation_resume'])}
             if 'observation_resume' in body else {})
    if 'delete_resume' in body:
        _require(not extra)
        extra = dict(delete_resume=observation_binding(body['delete_resume']))
    _require(parent in versions and 'parent_version' in body
        and type(body.get('credited_removed_allocated_bytes')) is int
        and body == dict(phase='reconciled', **selected, **extra,
        parent_path=parent, parent_version=body['parent_version'], uncertain=True,
        credited_removed_allocated_bytes=0))
    before, after = versions[parent], body['parent_version']
    _require(type(after) is list and len(after) == 10
        and all(type(value) is int and value >= 0 for value in after)
        and after[:6] == before[:6] and after[7] >= before[7] and after[8] >= before[8])
    return parent, after.copy()


def reconciliation_bindings(events):
    """Each exact original prefix must separately authenticate its decision."""
    values, pending = [], None
    for index, event in enumerate(events):
        if event['kind'] != 'restore_intent':
            continue
        body = event['body']
        if body.get('phase') == 'reconcile_intent':
            _require(pending is None and index > 0 and len(values) < 8)
            selected = binding(body)
            _require(body == dict(phase='reconcile_intent', **selected)
                and selected['original_head_event_digest'] == events[index-1]['event_digest'])
            values.append((events[:index], selected['decision_id'], selected['decision'], (event,)))
            pending = selected
        elif body.get('phase') == 'reconcile_delete_resume':
            _require(pending is not None and binding(body) == pending and len(values[-1][3]) < 8)
            resume = observation_binding(body.get('delete_resume'))
            _require(body == dict(phase='reconcile_delete_resume', **pending, delete_resume=resume)
                and resume['decision_id'] not in [values[-1][1]] + [
                    old['body']['delete_resume']['decision_id'] for old in values[-1][3][1:]])
            values[-1] = (*values[-1][:3], (*values[-1][3], event))
        elif body.get('phase') == 'reconciled':
            _require(pending is not None and binding(body) == pending)
            resume = values[-1][3][-1]['body'].get('delete_resume')
            # Observation-only completion after a resumed DELETE is separately
            # authenticated and can contain no renewed removal authority.
            _require(body.get('delete_resume') == resume if 'observation_resume' not in body
                     else 'delete_resume' not in body)
            values[-1] = (*values[-1][:3], (*values[-1][3], event))
            pending = None
    return values
