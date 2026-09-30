"""Authenticate private/published partitions of the same original restore births.

An uncertain rename observation is never a birth. Current owner, archive,
reservation, reader and native fences remain the worker's independent gates.
"""
from __future__ import annotations

from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_fence import _members
from .control_plane_lane_historical_restore_staging import stage_birth_versions
from .decision_evidence_contracts import canonical_digest


def _require(value):
    generation._require(value, 'restore_split_changed')


def _version(value):
    return type(value) is list and len(value) == 10 and all(type(v) is int and v >= 0 for v in value)


def _parent(before, after, delta):
    _require(_version(after) and after[:5] == before[:5] and after[5] == before[5] + delta
        and after[7] >= before[7] and after[8] >= before[8])


def _rename(versions, originals, stage, path, body):
    old = stage + '/' + path
    before, after = versions[old], body['member_version']
    _require(_version(after) and after[:8] == before[:8] and after[9] == before[9]
        and after[8] >= before[8])
    delta = int(originals[path]['kind'] == 'directory')
    _parent(versions[''], body['target_version'], delta)
    _parent(versions[stage], body['stage_version'], -delta)
    for name in list(versions):
        if name == old or name.startswith(old + '/'):
            versions[name[len(stage) + 1:]] = versions.pop(name)
    versions[path] = after
    versions[''], versions[stage] = body['target_version'], body['stage_version']


def _replay(original, decision, events, action_id, tick):
    originals = _members(original)
    stage = '.historical-restore-' + action_id
    complete = [i for i, event in enumerate(events) if event['kind'] == 'restore_intent'
                and event['body'].get('phase') == 'stage_complete']
    _require(len(complete) == 1)
    boundary = complete[0]
    versions, _ = stage_birth_versions(original, decision, events[:boundary + 1], action_id,
                                       complete=True, tick=tick)
    tops = sorted(path for path in originals if path and '/' not in path)
    published, pending, removal = set(), None, False
    for event in events[boundary + 1:]:
        tick()
        body = event['body']
        _require(event['kind'] == 'restore_intent')
        if body.get('phase') == 'stage_remove':
            _require(event is events[-1] and pending is None and published == set(tops)
                and body == dict(phase='stage_remove', stage_version=versions[stage], target_version=versions['']))
            removal = True
            continue
        path = body.get('path')
        _require(type(path) is str and path in tops and path not in published)
        if body.get('phase') == 'publish':
            expected = dict(phase='publish', path=path, stage_version=versions[stage],
                            target_version=versions[''], member_version=versions[stage + '/' + path])
            _require(body == expected and pending is None)
            pending = path
        else:
            _require(body.keys() == {'phase', 'path', 'member_version', 'stage_version', 'target_version',
                                     'uncertain'} and body['phase'] == 'publish_observed'
                and type(body['uncertain']) is bool and pending == path)
            _rename(versions, originals, stage, path, body)
            published.add(path)
            pending = None
    return versions, published, pending, events[boundary]['body'], removal


def validate_split_stage(original, observed, decision, events, action_id, *, tick=lambda: None):
    """Exactly one side of every top-level subtree, with only journaled renames."""
    originals, rows = _members(original), _members(observed)
    stage = '.historical-restore-' + action_id
    _require(set(observed) == set(original)
        and observed['schema_version'] == original['schema_version']
        and observed['execution_authorized'] is False
        and observed['generation_digest'] == canonical_digest(observed, digest_field='generation_digest')
        and all(observed[key] == original[key] for key in ('target_path', 'parent_path', 'logical_payload_bytes'))
        and observed['root_version'] == decision['parent_version']
        and rows['']['version'] == observed['target_version']
        and observed['member_count'] == original['member_count'] + int(stage in rows))
    versions, published, pending, complete, removal = _replay(original, decision, events, action_id, tick)
    stage_absent = stage not in rows
    if stage_absent:
        _require(removal and pending is None)
        _parent(versions[''], rows['']['version'], -1)
        versions[''] = rows['']['version']
        versions.pop(stage)
    uncertain = False
    if pending is not None:
        old = stage + '/' + pending
        _require((old in rows) != (pending in rows))
        if pending in rows:
            _rename(versions, originals, stage, pending, dict(member_version=rows[pending]['version'],
                stage_version=rows[stage]['version'], target_version=rows['']['version']))
            published.add(pending)
            uncertain = True
    _require(set(rows) == set(versions))
    for name, version in versions.items():
        tick()
        row = rows[name]
        _require(row['version'] == version)
        if name in ('', stage):
            _require(row['kind'] == 'directory')
            continue
        path = name[len(stage) + 1:] if name.startswith(stage + '/') else name
        old = originals[path]
        _require(row.keys() == old.keys() and all(row[key] == old[key]
            for key in old if key not in ('path', 'version', 'allocated_bytes')))
    return dict(published=published, pending=pending, uncertain=uncertain,
                stage_complete=complete, stage_removal_pending=removal, stage_absent=stage_absent)


def validate_publication_events(original, decision, events, action_id, *, tick=lambda: None):
    """Validate new publication observations and the planned stage removal."""
    removed = [i for i, event in enumerate(events) if event['kind'] == 'restore_intent'
               and event['body'].get('phase') == 'stage_removed']
    _require(len(removed) == 1)
    boundary = removed[0]
    versions, published, pending, _, removal = _replay(original, decision, events[:boundary], action_id, tick)
    _require(removal and pending is None)
    body = events[boundary]['body']
    _require(body.keys() in ({'phase', 'target_version'}, {'phase', 'target_version', 'uncertain'})
        and body['phase'] == 'stage_removed' and ('uncertain' not in body or body['uncertain'] is True))
    _parent(versions[''], body['target_version'], -1)
    return versions
