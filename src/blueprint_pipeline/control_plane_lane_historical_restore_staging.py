"""Authenticate actual complete private stage births before resuming publication.

Pure observation validation only. The worker still needs full archive/local
readback, current original authority, reservation, references and native rights.
"""
from __future__ import annotations

import stat

from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_fence import _members
from .control_plane_lane_historical_restore_reconciliation_replay import binding, parent_observation, observation_binding
from .decision_evidence_contracts import canonical_digest


def _require(value):
    generation._require(value, 'restore_stage_changed')


def stage_birth_versions(original, decision, events, action_id, *, complete, tick):
    """Replay original durable births only; this supplies no physical observation."""
    originals = _members(original)
    stage = '.historical-restore-' + action_id
    _require(type(events) is list and events)
    files = [row for row in originals.values() if row['kind'] == 'file']
    if complete:
        _require(events[-1]['kind'] == 'restore_intent'
            and events[-1]['body'] == dict(phase='stage_complete',
                archive_sha256=decision['archive']['sha256'], archive_size_bytes=decision['archive']['size_bytes'],
                restored_files=len(files), restored_logical_bytes=original['logical_payload_bytes']))
    pending, reconciliation = None, None
    versions = {'': decision['tombstone_version']}
    births = set()
    for event in events[:-1] if complete else events:
        tick()
        kind, body = event['kind'], event['body']
        if kind == 'intent':
            _require(event is events[0])
            continue
        if kind == 'restore_intent':
            phase = body.get('phase')
            _require(phase in ('reservation', 'directory', 'member', 'reconcile_intent', 'reconcile_delete_resume', 'reconciled'))
            if phase == 'reconcile_intent':
                _require(pending is not None and reconciliation is None)
                reconciliation = binding(body)
                _require(body == dict(phase=phase, **reconciliation))
                continue
            if phase == 'reconcile_delete_resume':
                _require(pending is not None and reconciliation is not None and binding(body) == reconciliation)
                _require(body == dict(phase=phase, **reconciliation,
                    delete_resume=observation_binding(body.get('delete_resume'))))
                continue
            if phase == 'reconciled':
                _require(pending is not None and reconciliation is not None)
                parent, version = parent_observation(body, pending, reconciliation, stage, versions)
                versions[parent] = version
                pending, reconciliation = None, None
                continue
            _require(reconciliation is None)
            if not complete or phase in ('directory', 'member'):
                if phase == 'reservation':
                    _require(not births and (pending is None or pending == dict(
                        phase='directory', path='', stage_path=stage)))
                else:
                    path = body.get('path')
                    _require(type(path) is str and path in originals and path not in births)
                    expected = dict(phase=phase, path=path,
                        stage_path=stage + ('/' + path if path else ''))
                    _require(originals[path]['kind'] == ('directory' if phase == 'directory' else 'file'))
                    if phase == 'member':
                        expected.update(sha256=originals[path]['sha256'], size_bytes=originals[path]['size_bytes'])
                    _require(body == expected and (pending is None or pending == expected))
                    pending = expected
            continue
        _require(kind in ('restore_directory', 'restore_member') and reconciliation is None)
        path = body.get('path')
        _require(type(path) is str and path in originals and path not in births)
        if not complete and pending is not None:
            _require(pending['path'] == path
                and pending['phase'] == ('directory' if kind == 'restore_directory' else 'member'))
        pending = None
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
    if complete:
        _require(births == set(originals))
    return versions, births


def _validate_stage(original, observed, decision, events, action_id, *, complete, tick):
    originals, rows = _members(original), _members(observed)
    stage = '.historical-restore-' + action_id
    versions, births = stage_birth_versions(original, decision, events, action_id, complete=complete, tick=tick)
    _require(type(observed) is dict and set(observed) == set(original)
        and observed['schema_version'] == original['schema_version']
        and observed['execution_authorized'] is False
        and observed['generation_digest'] == canonical_digest(observed, digest_field='generation_digest')
        and all(observed[key] == original[key] for key in ('target_path', 'parent_path'))
        and observed['root_version'] == decision['parent_version']
        and rows['']['version'] == observed['target_version'])
    _require(('' in births or not births and any(event['kind'] == 'restore_intent'
        and event['body'].get('phase') == 'reconciled' for event in events))
        and rows['']['version'] == versions['']
        and set(rows) == {''} | {stage + ('/' + path if path else '') for path in births}
        and observed['member_count'] == len(births) + 1
        and observed['logical_payload_bytes'] == sum(originals[path]['size_bytes']
            for path in births if originals[path]['kind'] == 'file'))
    for path in births:
        old = originals[path]
        tick()
        relative = stage + ('/' + path if path else '')
        row = rows[relative]
        _require(row.keys() == old.keys() and row['version'] == versions[relative]
            and all(row[key] == old[key] for key in old if key not in ('path', 'version', 'allocated_bytes')))
    return births


def validate_complete_stage(original, observed, decision, events, action_id, *, tick=lambda: None):
    _validate_stage(original, observed, decision, events, action_id, complete=True, tick=tick)


def validate_private_prefix(original, observed, decision, events, action_id, *, tick=lambda: None):
    """Known private births only; an unlogged syscall or payload always refuses."""
    return _validate_stage(original, observed, decision, events, action_id, complete=False, tick=tick)
