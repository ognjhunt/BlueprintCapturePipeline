"""ADP-009D/day28: restore labels cannot replace births, final and owner access.

These parser observations grant no native authority. Positive installed action
and read-only GC observation are exercised by the disposable Linux selector.
"""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_restore_receipts.py
from copy import deepcopy

import pytest

from tests.test_historical_generation_restore_snapshot import observations
from tests.test_historical_generation_authority import historical_installation  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401

# ruff: noqa: F811


def recorded_observations(installed):
    original, private, reopened = observations(installed)
    action_id = 'a' * 32
    decision = dict(action_id=action_id, owner='owner', manifest='original-selector',
        final_event_digest='offload-final', archive={'sha256': 'archive', 'size_bytes': 42},
        parent_version=private['root_version'], issued_at_epoch=100, expires_at_epoch=200)
    files = [row for row in original['members'] if row['kind'] == 'file']
    receipt = dict(status='completed', action='restore', action_id=action_id, owner='owner',
        generation_digest=original['generation_digest'], original_manifest=decision['manifest'],
        original_final_event_digest=decision['final_event_digest'], archive_sha256='archive',
        archive_size_bytes=42, restored_files=len(files),
        restored_logical_bytes=sum(row['size_bytes'] for row in files), fresh_disk_reservation=True,
        root_directory_retained=True, owner_access_reopened=False,
        protected_root_version=private['members'][0]['version'])
    events = [dict(kind='restore_member', body=deepcopy(row), sequence=index,
                   observed_at_epoch=110) for index, row in enumerate(files)]
    events.extend([dict(kind='restore_final', body=receipt, sequence=len(events),
                        observed_at_epoch=120, event_digest='restore-final'),
                   dict(kind='access_reopened', body=dict(phase='owner_rights_observed', path='',
                        version=reopened), sequence=len(events)+1, observed_at_epoch=121)])
    return ({}, decision, original, {}), events, private


def test_exact_recorded_projection_preserves_every_byte_and_inode_selector(historical_installation):
    from blueprint_pipeline.control_plane_lane_historical_restore_receipts import validate_restored_receipt
    selected, events, snapshot = recorded_observations(historical_installation)
    before = deepcopy(snapshot)
    final, expected = validate_restored_receipt(selected, events, snapshot)
    assert final == events[-2] and snapshot == before
    assert expected['execution_authorized'] is False
    assert expected['members'][1:] == snapshot['members'][1:]
    assert expected['members'][0]['version'] == events[-1]['body']['version']


@pytest.mark.parametrize('change', ['missing_final', 'missing_access', 'access_before_final',
    'duplicate_birth', 'wrong_birth', 'changed_byte', 'wrong_owner', 'late_final', 'false_count'])
def test_invalid_restore_history_cannot_be_observed_as_completed(historical_installation, change):
    from blueprint_pipeline.control_plane_lane_historical_restore_receipts import validate_restored_receipt
    selected, events, snapshot = recorded_observations(historical_installation)
    if change == 'missing_final':
        events = [event for event in events if event['kind'] != 'restore_final']
    elif change == 'missing_access':
        events.pop()
    elif change == 'access_before_final':
        events[-1]['sequence'] = 0
    elif change == 'duplicate_birth':
        events.insert(0, deepcopy(events[0]))
    elif change == 'wrong_birth':
        events[0]['body']['version'][1] += 1
    elif change == 'changed_byte':
        events[0]['body']['sha256'] = 'sha256:' + 'f' * 64
    elif change == 'wrong_owner':
        events[-2]['body']['owner'] = 'other-owner'
    elif change == 'late_final':
        events[-2]['observed_at_epoch'] = 200
    else:
        events[-2]['body']['restored_files'] += 1
    with pytest.raises(ValueError):
        validate_restored_receipt(selected, events, snapshot)


@pytest.mark.parametrize('change', [None, 'path', 'version', 'owner', 'mode', 'phase', 'kind'])
def test_pending_access_only_selects_the_exact_still_private_root(historical_installation, change):
    import stat
    from blueprint_pipeline.control_plane_lane_historical_restore_receipts import validate_pending_owner_access
    selected, events, snapshot = recorded_observations(historical_installation)
    final = events[-2]
    events.pop()  # No completed access event is fabricated.
    owner = selected[2]['members'][0]['version']
    body = dict(phase='owner_rights', path='', version=deepcopy(snapshot['members'][0]['version']),
                uid=owner[3], gid=owner[4], mode=stat.S_IMODE(owner[2]))
    pending = dict(kind='restore_intent', body=body, sequence=final['sequence'] + 1)
    events.append(pending)
    if change == 'kind':
        pending['kind'] = 'restore_member'
    elif change == 'version':
        body['version'][1] += 1
    elif change == 'owner':
        body['uid'] += 1
    elif change == 'mode':
        body['mode'] ^= 0o020
    elif change == 'path':
        body['path'] = 'nested'
    elif change == 'phase':
        body['phase'] = 'publish'
    if change is None:
        assert validate_pending_owner_access(selected, events, snapshot, final) is None
    else:
        with pytest.raises(ValueError):
            validate_pending_owner_access(selected, events, snapshot, final)
