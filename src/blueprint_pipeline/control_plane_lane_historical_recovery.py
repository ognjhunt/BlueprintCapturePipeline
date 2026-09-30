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


def recover_fence(manifest, events, observed):
    """Only a sequential journaled fence prefix; no missing member adoption.

    The last intent can precede chown, chmod, or their completion record. All
    original payload hashes and namespace versions must still match exactly.
    """
    def require(value):
        _require(value, 'recovery_changed')
    rows = _members(manifest)
    require(isinstance(events, list) and 0 < len(events) <= 2 * len(rows) + 1
            and events[0]['kind'] == 'intent')
    order = sorted(rows, key=lambda name: (name.count('/') + bool(name), name))
    completed, versions, pending = set(), {}, None
    for event in events[1:]:
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
    require(observed.get('target_path') == manifest['target_path']
        and observed.get('parent_path') == manifest['parent_path']
        and observed.get('root_version') == manifest['root_version']
        and observed.get('member_count') == len(rows)
        and isinstance(observed.get('members'), list) and len(observed['members']) == len(rows))
    current = {row['path']: row for row in observed['members']}
    require(set(current) == set(rows))
    for name, original in rows.items():
        row = current[name]
        require(set(row) == set(original) and all(row[key] == value for key, value in original.items() if key != 'version'))
        if name in completed:
            require(row['version'] == versions[name])
        elif name == pending:
            require(_valid_transition(original['version'], row['version'], original['kind'], complete=False))
        else:
            require(row['version'] == original['version'])
    return dict(manifest=observed, completed=completed, pending=pending)
