"""Authenticate a complete private publication against actual restore births.

This pure validator grants no authority and fabricates no filesystem metadata.
The worker supplies the original authenticated chain and a fresh full inventory,
then still needs current owner, archive, reservation, reference and native gates.
"""
from __future__ import annotations

import stat

from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_fence import _members
from .decision_evidence_contracts import canonical_digest


def _require(value):
    generation._require(value, 'restore_publication_changed')


def validate_publication(original, observed, decision, events, action_id, *, tick=lambda: None):
    """Only the fully published, still root-private phase is supported here."""
    originals = _members(original)
    _require(type(observed) is dict and set(observed) == set(original)
        and observed['schema_version'] == original['schema_version']
        and observed['execution_authorized'] is False
        and observed['generation_digest'] == canonical_digest(observed, digest_field='generation_digest')
        and all(observed[key] == original[key] for key in ('target_path', 'parent_path',
                                                         'member_count', 'logical_payload_bytes'))
        and observed['root_version'] == decision['parent_version'])
    rows = _members(observed)
    _require(set(rows) == set(originals) and observed['target_version'] == rows['']['version']
        and rows['']['version'][:5] == decision['tombstone_version'][:5]
        and rows['']['version'][:2] == originals['']['version'][:2])
    for path, row in rows.items():
        tick()
        old = originals[path]
        _require(row.keys() == old.keys()
            and all(row[key] == old[key] for key in old if key not in ('version', 'allocated_bytes'))
            and row['version'][0] == rows['']['version'][0]
            and row['version'][2:5] == [stat.S_IFDIR | 0o700 if row['kind'] == 'directory'
                                      else stat.S_IFREG | 0o600, 0, 0])
    _require(type(events) is list and events and events[-1]['kind'] == 'restore_intent'
        and events[-1]['body'].get('phase') == 'stage_removed'
        and events[-1]['body'].get('target_version') == rows['']['version'])
    stage = '.historical-restore-' + action_id
    versions, births, publications = {}, set(), set()
    complete = False
    for event in events:
        tick()
        kind, body = event['kind'], event['body']
        _require(kind not in ('restore_final', 'access_reopened')
            and not (kind == 'restore_intent' and body.get('phase') == 'owner_rights'))
        if kind in ('restore_directory', 'restore_member'):
            path = body['path']
            _require(not complete and path in originals and path not in births
                and body['stage_path'] == stage + ('/' + path if path else '')
                and originals[path]['kind'] == ('directory' if kind == 'restore_directory' else 'file'))
            version = body['version']
            _require(type(version) is list and len(version) == 10
                and all(type(value) is int and value >= 0 for value in version)
                and version[0] == rows['']['version'][0]
                and version[2:5] == [stat.S_IFDIR | 0o700 if kind == 'restore_directory'
                                     else stat.S_IFREG | 0o600, 0, 0])
            if kind == 'restore_member':
                _require(body['sha256'] == originals[path]['sha256']
                    and body['size_bytes'] == originals[path]['size_bytes']
                    and version[5] == 1 and version[6] == body['size_bytes'])
            births.add(path)
            versions[path] = version
            parent = body['parent_path']
            if parent:
                _require(parent == stage or parent.startswith(stage + '/'))
                relative = parent[len(stage):].lstrip('/')
                _require(relative in versions and body['parent_version'][:5] == versions[relative][:5])
                versions[relative] = body['parent_version']
        elif kind == 'restore_intent' and body.get('phase') == 'stage_complete':
            _require(not complete and births == set(originals))
            complete = True
        elif kind == 'restore_intent' and body.get('phase') == 'publish':
            path = body['path']
            _require(complete and path in versions and path and '/' not in path
                and path not in publications and body['member_version'] == versions[path])
            publications.add(path)
    _require(complete and publications == {path for path in originals if path and '/' not in path})
    for path, row in rows.items():
        tick()
        if not path:
            continue
        before, after = versions[path], row['version']
        _require(after == before if path not in publications else
            after[:8] == before[:8] and after[9] == before[9] and after[8] >= before[8])
