"""Authenticate actual complete private stage births before resuming publication.

Pure observation validation only. The worker still needs full archive/local
readback, current original authority, reservation, references and native rights.
"""
from __future__ import annotations

import stat

from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_fence import _members
from .decision_evidence_contracts import canonical_digest


def _require(value):
    generation._require(value, 'restore_stage_changed')


def validate_complete_stage(original, observed, decision, events, action_id, *, tick=lambda: None):
    originals, rows = _members(original), _members(observed)
    stage = '.historical-restore-' + action_id
    _require(type(observed) is dict and set(observed) == set(original)
        and observed['schema_version'] == original['schema_version']
        and observed['execution_authorized'] is False
        and observed['generation_digest'] == canonical_digest(observed, digest_field='generation_digest')
        and all(observed[key] == original[key] for key in ('target_path', 'parent_path', 'logical_payload_bytes'))
        and observed['root_version'] == decision['parent_version']
        and observed['member_count'] == original['member_count'] + 1
        and set(rows) == {''} | {stage + ('/' + path if path else '') for path in originals}
        and rows['']['version'] == observed['target_version'])
    _require(type(events) is list and events and events[-1]['kind'] == 'restore_intent')
    files = [row for row in originals.values() if row['kind'] == 'file']
    _require(events[-1]['body'] == dict(phase='stage_complete',
        archive_sha256=decision['archive']['sha256'], archive_size_bytes=decision['archive']['size_bytes'],
        restored_files=len(files), restored_logical_bytes=original['logical_payload_bytes']))
    versions = {'': decision['tombstone_version']}
    births = set()
    for event in events[:-1]:
        tick()
        kind, body = event['kind'], event['body']
        if kind == 'intent':
            _require(event is events[0])
            continue
        if kind == 'restore_intent':
            _require(body.get('phase') in ('reservation', 'directory', 'member'))
            continue
        _require(kind in ('restore_directory', 'restore_member'))
        path = body.get('path')
        _require(type(path) is str and path in originals and path not in births)
        relative = stage + ('/' + path if path else '')
        old = originals[path]
        _require(body['stage_path'] == relative
            and old['kind'] == ('directory' if kind == 'restore_directory' else 'file'))
        version = body['version']
        _require(type(version) is list and len(version) == 10
            and all(type(value) is int and value >= 0 for value in version)
            and version[0] == versions[''][0]
            and version[2:5] == [stat.S_IFDIR | 0o700 if kind == 'restore_directory'
                                 else stat.S_IFREG | 0o600, 0, 0])
        if kind == 'restore_member':
            _require(body['sha256'] == old['sha256'] and body['size_bytes'] == old['size_bytes']
                and version[5] == 1 and version[6] == old['size_bytes'])
        parent = relative.rpartition('/')[0]
        _require(body['parent_path'] == parent and parent in versions)
        before, after = versions[parent], body['parent_version']
        _require(type(after) is list and len(after) == 10
            and all(type(value) is int and value >= 0 for value in after)
            and after[:5] == before[:5]
            and after[5] == before[5] + int(kind == 'restore_directory')
            and after[7] >= before[7] and after[8] >= before[8])
        versions[parent], versions[relative] = after, version
        births.add(path)
    _require(births == set(originals) and rows['']['version'] == versions[''])
    for path, old in originals.items():
        tick()
        relative = stage + ('/' + path if path else '')
        row = rows[relative]
        _require(row.keys() == old.keys() and row['version'] == versions[relative]
            and all(row[key] == old[key] for key in old if key not in ('path', 'version', 'allocated_bytes')))
