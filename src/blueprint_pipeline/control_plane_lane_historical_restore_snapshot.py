"""Bound restored physical observations for read-only retry, never authority."""
from __future__ import annotations

import copy
import stat

from . import control_plane_lane_historical_generation as generation
from .decision_evidence_contracts import canonical_digest


def _require(value):
    generation._require(value, 'restore_snapshot_changed')


def after_reopen(original, private, reopened):
    """Project only the journaled final root permission transition.

    The caller separately authenticates the complete protected snapshot and
    access event, then compares this projection with a fresh full inventory.
    Non-root inode versions and every original byte remain exact. This function
    observes no filesystem and cannot grant execution or reopen access.
    """
    _require(type(private) is dict and set(private) == set(original)
        and private.get('schema_version') == 'control_plane_historical_generation.v1'
        and private.get('execution_authorized') is False
        and private.get('generation_digest') == canonical_digest(private, digest_field='generation_digest')
        and all(private[key] == original[key] for key in ('target_path', 'parent_path'))
        and type(private.get('root_identity')) is dict
        and set(private['root_identity']) == set(original['root_identity'])
        and all(private['root_identity'][key] == original['root_identity'][key]
                for key in original['root_identity'] if key != 'ctime_ns')
        and private.get('member_count') == original['member_count']
        and private.get('logical_payload_bytes') == original['logical_payload_bytes'])
    rows, originals = private.get('members'), original['members']
    _require(type(rows) is list and len(rows) == len(originals) <= generation.MAX_MEMBERS)
    for row, old in zip(rows, originals, strict=True):
        _require(type(row) is dict and set(row) == set(old)
            and all(row[key] == old[key] for key in old if key not in ('version', 'allocated_bytes'))
            and type(row['version']) is list and len(row['version']) == 10
            and all(type(value) is int and value >= 0 for value in row['version'])
            and row['version'][0] == originals[0]['version'][0]
            and (row['version'][2:5] == old['version'][2:5] if row['path'] else
                 row['version'][2:5] == [stat.S_IFDIR | 0o700, 0, 0]))
    before, owner = rows[0]['version'], originals[0]['version']
    _require(rows[0]['path'] == '' and rows[0]['kind'] == 'directory'
        and before[:2] == owner[:2] and private['target_version'] == before
        and type(reopened) is list and len(reopened) == 10
        and all(type(value) is int and value >= 0 for value in reopened)
        and reopened[:2] == before[:2] and reopened[2:5] == owner[2:5]
        and reopened[5:8] == before[5:8] and reopened[9] == before[9]
        and reopened[8] >= before[8])
    projected = copy.deepcopy(private)
    projected['members'][0]['version'] = reopened.copy()
    projected['target_version'] = reopened.copy()
    projected['target_identity'] = dict(dev=reopened[0], ino=reopened[1], type='directory',
        mode=stat.S_IMODE(reopened[2]), uid=reopened[3], gid=reopened[4], ctime_ns=reopened[8])
    projected['generation_digest'] = canonical_digest(projected, digest_field='generation_digest')
    return projected
