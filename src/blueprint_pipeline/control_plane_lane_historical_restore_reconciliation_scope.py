"""Non-authorizing exact scope for one unlogged private creation or write.

Unknown current observations never become birth receipts. A distinct protected
owner decision, current native gates and real removal observations are required
before any discard or same-operation retry can use this reviewable scope.
"""
from __future__ import annotations

import copy
import re
import stat

from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_fence import _members
from .control_plane_lane_historical_restore_staging import stage_birth_versions
from .decision_evidence_contracts import canonical_digest


def _require(value):
    generation._require(value, 'restore_reconciliation_scope_invalid')


def unknown_creation_scope(original, observed, decision, events, action_id, *, tick=lambda: None):
    """Exactly one pending unlogged row; no selected or published inode discard."""
    originals, rows = _members(original), _members(observed)
    stage = '.historical-restore-' + action_id
    _require(type(events) is list and events and events[-1]['kind'] == 'restore_intent')
    pending = events[-1]['body']
    _require(pending.get('phase') in ('directory', 'member'))
    versions, births = stage_birth_versions(original, decision, events, action_id, complete=False, tick=tick)
    path = pending['path']
    _require(path not in births and pending['stage_path'] == stage + ('/' + path if path else ''))
    relative = pending['stage_path']
    parent = relative.rpartition('/')[0]
    _require(parent in versions and relative in rows and set(rows) == set(versions) | {relative})
    _require(set(observed) == set(original) and observed['schema_version'] == original['schema_version']
        and observed['execution_authorized'] is False
        and observed['generation_digest'] == canonical_digest(observed, digest_field='generation_digest')
        and all(observed[key] == original[key] for key in ('target_path', 'parent_path', 'root_identity'))
        and observed['root_version'] == decision['parent_version']
        and observed['target_version'] == rows['']['version']
        and observed['member_count'] == len(versions) + 1)
    unknown = rows[relative]
    old = originals[path]
    version = unknown['version']
    directory = pending['phase'] == 'directory'
    _require(unknown.keys() == old.keys() and unknown['kind'] == old['kind']
        and version[0] == versions[''][0] and all(value >= 0 for value in version)
        and version[:2] not in [born[:2] for born in versions.values()]
        and version[2:5] == [stat.S_IFDIR | 0o700 if directory else stat.S_IFREG | 0o600, 0, 0])
    _require(unknown['size_bytes'] == 0 and unknown['sha256'] is None and version[5] == 2 if directory else
        type(unknown['size_bytes']) is int and 0 <= unknown['size_bytes'] <= old['size_bytes']
        and version[5] == 1 and version[6] == unknown['size_bytes']
        and type(unknown['sha256']) is str and re.fullmatch('sha256:[0-9a-f]{64}', unknown['sha256']))
    before, after = versions[parent], rows[parent]['version']
    _require(after[:5] == before[:5] and after[5] == before[5] + int(directory)
        and after[7] >= before[7] and after[8] >= before[8])
    for name, born in versions.items():
        tick()
        row = rows[name]
        _require(row['version'] == (after if name == parent else born))
        if not name:
            _require(row['kind'] == 'directory')
            continue
        original_path = name[len(stage):].lstrip('/')
        old_row = originals[original_path]
        _require(row.keys() == old_row.keys() and all(row[key] == old_row[key]
            for key in old_row if key not in ('path', 'version', 'allocated_bytes')))
    _require(observed['logical_payload_bytes'] == unknown['size_bytes'] + sum(
        originals[path]['size_bytes'] for path in births if originals[path]['kind'] == 'file'))
    return copy.deepcopy(dict(execution_authorized=False, owner_reconciliation_required=True,
        original_generation_digest=original['generation_digest'],
        observed_generation_digest=observed['generation_digest'], pending_intent=pending,
        remove_member=unknown, parent_path=parent, parent_before=before, parent_after=after))


def reconciled_parent(approved, observed, scope, *, tick=lambda: None):
    """One exact absent dentry and its parent transition; never freed credit."""
    before, rows = _members(approved), _members(observed)
    relative, parent = scope['remove_member']['path'], scope['parent_path']
    _require(before.get(relative) == scope['remove_member']
        and before[parent]['version'] == scope['parent_after']
        and set(rows) == set(before) - {relative}
        and set(observed) == set(approved)
        and observed['generation_digest'] == canonical_digest(observed, digest_field='generation_digest')
        and observed['execution_authorized'] is False
        and all(observed[key] == approved[key] for key in
            ('schema_version', 'target_path', 'parent_path', 'root_identity', 'root_version'))
        and observed['target_version'] == rows['']['version']
        and observed['member_count'] == approved['member_count'] - 1
        and observed['logical_payload_bytes'] == approved['logical_payload_bytes'] - scope['remove_member']['size_bytes'])
    old, current = before[parent]['version'], rows[parent]['version']
    _require(current[:5] == old[:5]
        and current[5] == old[5] - int(scope['remove_member']['kind'] == 'directory')
        and current[7] >= old[7] and current[8] >= old[8])
    for name, row in rows.items():
        tick()
        original = before[name]
        _require(row.keys() == original.keys() and (row == original if name != parent else
            all(row[key] == original[key] for key in original if key not in ('version', 'allocated_bytes'))))
    return current.copy()
