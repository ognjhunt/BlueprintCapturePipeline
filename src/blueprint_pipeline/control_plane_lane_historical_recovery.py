"""Reconcile authenticated original fence events with fully reread owner bytes.

This projection grants no authority. The worker still holds current owner and
unit checks, the original deadline, native ancestry and every reader gate.
"""
from __future__ import annotations

import stat

from .control_plane_lane_historical_fence import _members
from .control_plane_lane_historical_generation import _require


def _valid_transition(original, observed, kind, *, complete):
    if not (isinstance(observed, list) and len(observed) == 10
            and all(type(value) is int for value in observed)):
        return False
    mode = stat.S_IFMT(original[2]) | (0o700 if kind == 'directory' else 0o600)
    if complete:
        rights = observed[2:5] == [mode, 0, 0]
    else:
        rights = observed == original or observed[3:5] == [0, 0] and observed[2] in (original[2], mode)
    return rights and observed[:2] == original[:2] and observed[5:8] == original[5:8] \
        and observed[9] == original[9] and observed[8] >= original[8]


def require(value):
    _require(value, 'recovery_changed')


def _fence_state(manifest, events):
    rows = _members(manifest)
    require(isinstance(events, list) and 0 < len(events) <= 4 * len(rows) + 64
            and events[0]['kind'] == 'intent')
    order = sorted(rows, key=lambda name: (name.count('/') + bool(name), name))
    completed, versions, pending = set(), {}, None
    index = 1
    while index < len(events) and events[index]['kind'] in ('fence_intent', 'fenced'):
        event = events[index]
        kind, body = event.get('kind'), event.get('body')
        require(type(body) is dict)
        if kind == 'fence_intent':
            require(pending is None and len(completed) < len(order))
            pending = order[len(completed)]
            row = rows[pending]
            require(body == dict(path=pending, version=row['version'], uid=0, gid=0,
                mode=0o700 if row['kind'] == 'directory' else 0o600))
        elif kind == 'fenced':
            require(pending is not None and set(body) == {'path', 'version'} and body['path'] == pending)
            require(_valid_transition(rows[pending]['version'], body['version'], rows[pending]['kind'], complete=True))
            completed.add(pending)
            versions[pending] = body['version']
            pending = None
        else:
            require(False)
        index += 1
    return rows, completed, versions, pending, events[index:]


def _current(manifest, observed, rows, remaining):
    require(observed.get('target_path') == manifest['target_path']
        and observed.get('parent_path') == manifest['parent_path']
        and observed.get('root_version') == manifest['root_version']
        and observed.get('member_count') == len(remaining)
        and isinstance(observed.get('members'), list) and len(observed['members']) == len(remaining))
    current = {row['path']: row for row in observed['members']}
    require(set(current) == remaining)
    for name in remaining:
        original = rows[name]
        row = current[name]
        require(set(row) == set(original) and all(row[key] == value for key, value in original.items() if key != 'version'))
    return current


def recover_fence(manifest, events, observed):
    """Sequential journaled fence prefix, exact bytes, no missing members."""
    rows, completed, versions, pending, rest = _fence_state(manifest, events)
    require(not rest)
    current = _current(manifest, observed, rows, set(rows))
    for name, original in rows.items():
        row = current[name]
        if name in completed:
            require(row['version'] == versions[name])
        elif name == pending:
            require(_valid_transition(original['version'], row['version'], original['kind'], complete=False))
        else:
            require(row['version'] == original['version'])
    return dict(manifest=observed, completed=completed, pending=pending)


def _parent_transition(before, after, kind):
    return isinstance(after, list) and len(after) == 10 and all(type(value) is int and value >= 0 for value in after) \
        and after[:5] == before[:5] and after[5] == before[5] - int(kind == 'directory') \
        and after[7] >= before[7] and after[8] >= before[8]


def recover_action(manifest, events, observed):
    """Resume original removal intent; unobserved absence earns zero bytes.

    Only the last syscall can have an unlogged outcome. Remaining original
    bytes are fully reread, and every removed name must follow the fixed order.
    """
    rows, completed, versions, pending_fence, rest = _fence_state(manifest, events)
    if not rest:
        result = recover_fence(manifest, events, observed)
        return dict(**result, removed=set(), uncertain=set(), pending_removal=None, reconcile=None,
                    prior_observed_removed_allocated_bytes=0)
    require(len(completed) == len(rows) and pending_fence is None)
    order = sorted((name for name in rows if name), key=lambda name: (name.count('/'), name), reverse=True)
    removed, uncertain, pending, allocated = set(), set(), None, 0
    for event in rest:
        kind, body = event.get('kind'), event.get('body')
        require(type(body) is dict)
        if kind == 'removal_intent':
            require(pending is None and len(removed | uncertain) < len(order))
            pending = order[len(removed | uncertain)]
            row = rows[pending]
            require(body == dict(path=pending, kind=row['kind'], version=versions[pending],
                sha256=row['sha256'], size_bytes=row['size_bytes']))
        elif kind in ('removed', 'removal_uncertain'):
            require(pending is not None)
            row, before = rows[pending], versions[pending]
            parent = pending.rpartition('/')[0]
            require(set(body) == {'path', 'kind', 'physical_identity', 'logical_bytes',
                'observed_removed_allocated_bytes', 'parent_path', 'parent_version'}
                and body['path'] == pending and body['kind'] == row['kind']
                and body['physical_identity'] == before[:2] and body['parent_path'] == parent
                and body['logical_bytes'] == (row['size_bytes'] if kind == 'removed' else 0)
                and body['observed_removed_allocated_bytes'] == (before[9] * 512 if kind == 'removed' else 0)
                and _parent_transition(versions[parent], body['parent_version'], row['kind']))
            versions[parent] = body['parent_version']
            allocated += body['observed_removed_allocated_bytes']
            (removed if kind == 'removed' else uncertain).add(pending)
            pending = None
        else:
            require(False)
    present = {row['path'] for row in observed['members']}
    reconcile = None
    if pending is not None and pending not in present:
        row, before = rows[pending], versions[pending]
        parent = pending.rpartition('/')[0]
        parent_row = next((row for row in observed['members'] if row['path'] == parent), None)
        require(parent_row is not None and _parent_transition(versions[parent], parent_row['version'], row['kind']))
        versions[parent] = parent_row['version']
        reconcile = dict(path=pending, kind=row['kind'], physical_identity=before[:2], logical_bytes=0,
            observed_removed_allocated_bytes=0, parent_path=parent, parent_version=versions[parent])
        uncertain.add(pending)
        pending = None
    current = _current(manifest, observed, rows, set(rows) - removed - uncertain)
    require(all(row['version'] == versions[name] for name, row in current.items()))
    return dict(manifest=observed, completed=completed, pending=None, removed=removed,
                uncertain=uncertain, pending_removal=pending, reconcile=reconcile,
                prior_observed_removed_allocated_bytes=allocated)
