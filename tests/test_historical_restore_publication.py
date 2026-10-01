"""ADP-009D/day28: published bytes require original journaled inode births.

Pure metadata cases exercise validation only; actual syscall provenance and
recovery are separately required by the disposable Linux connected fixture.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_restore_publication.py
import copy
import stat

import pytest

from tests.test_historical_generation_authority import historical_installation  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401

# ruff: noqa: F811


def projections(installed):
    from blueprint_pipeline.control_plane_lane_historical_generation import inventory_historical_generation
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    original = inventory_historical_generation(installed[1], allowed_roots=(installed[1].parent,))
    observed = copy.deepcopy(original)
    for row in observed['members']:
        row['version'][2:5] = [stat.S_IFDIR | 0o700 if row['kind'] == 'directory'
                               else stat.S_IFREG | 0o600, 0, 0]
        if row['path']:
            row['version'][1] += 100
    observed['target_version'] = observed['members'][0]['version'].copy()
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    action_id = 'a' * 32
    stage = '.historical-restore-' + action_id
    root, file = observed['members']
    staged_root = root['version'].copy()
    staged_root[1] += 200
    events = [dict(kind='restore_directory', body=dict(path='', stage_path=stage,
        version=staged_root, parent_path='', parent_version=root['version'])),
        dict(kind='restore_member', body=dict(path=file['path'], stage_path=stage + '/' + file['path'],
            version=file['version'].copy(), sha256=file['sha256'], size_bytes=file['size_bytes'],
            parent_path=stage, parent_version=staged_root)),
        dict(kind='restore_intent', body=dict(phase='stage_complete')),
        dict(kind='restore_intent', body=dict(phase='publish', path=file['path'],
            member_version=file['version'].copy())),
        dict(kind='restore_intent', body=dict(phase='stage_removed', target_version=root['version'].copy()))]
    decision = dict(parent_version=observed['root_version'].copy(),
                    tombstone_version=root['version'].copy())
    return original, observed, decision, events, action_id


def validate(values):
    from blueprint_pipeline.control_plane_lane_historical_restore_publication import validate_publication
    return validate_publication(*values)


def test_complete_private_publication_validates_without_creating_metadata(historical_installation):
    values = projections(historical_installation)
    saved = copy.deepcopy(values)
    assert validate(values) is None
    assert values == saved


@pytest.mark.parametrize('change', ['extra', 'missing_birth', 'duplicate_birth', 'inode', 'bytes',
                                   'parent', 'root', 'owner', 'mode', 'size', 'mtime',
                                   'incomplete_publish', 'publication_inode', 'root_event', 'rights'])
def test_unproven_publication_is_not_recovery_authority(historical_installation, change):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = projections(historical_installation)
    _, observed, _, events, _ = values
    row = observed['members'][1]
    if change == 'extra':
        observed['members'].append(dict(row, path='unaccounted'))
        observed['member_count'] += 1
    elif change == 'missing_birth':
        events.pop(1)
    elif change == 'duplicate_birth':
        events.insert(1, copy.deepcopy(events[1]))
    elif change == 'inode':
        row['version'][1] += 1
    elif change == 'bytes':
        row['sha256'] = 'sha256:' + 'f' * 64
    elif change == 'parent':
        observed['root_version'][1] += 1
    elif change == 'root':
        observed['members'][0]['version'][1] += 1
        observed['target_version'] = observed['members'][0]['version'].copy()
    elif change in ('owner', 'mode', 'size', 'mtime'):
        row['version'][{'owner': 3, 'mode': 2, 'size': 6, 'mtime': 7}[change]] += 1
    elif change == 'incomplete_publish':
        events.pop(-2)
    elif change == 'publication_inode':
        events[-2]['body']['member_version'][1] += 1
    elif change == 'root_event':
        events[-1]['body']['target_version'][8] += 1
    else:
        events.append(dict(kind='restore_intent', body=dict(phase='owner_rights', path=row['path'])))
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    with pytest.raises(ValueError, match='restore_publication_changed'):
        validate(values)


def rights_projections(installed):
    """Pure projections only: actual chown fault belongs in Linux acceptance."""
    values = projections(installed)
    original, observed, _, events, _ = values
    original['members'][1]['version'][3:5] = [123, 456]
    original['members'][1]['version'][2] = stat.S_IFREG | 0o640
    before = observed['members'][1]['version'].copy()
    events.append(dict(kind='restore_intent', body=dict(phase='owner_rights',
        path=observed['members'][1]['path'], version=before,
        uid=123, gid=456, mode=stat.S_IMODE(original['members'][1]['version'][2]))))
    return values


@pytest.mark.parametrize('phase', ['intent', 'chown', 'chmod', 'repeat'])
def test_private_publication_resumes_only_exact_planned_owner_syscalls(historical_installation, phase):
    from blueprint_pipeline.control_plane_lane_historical_restore_publication import validate_publication
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = rights_projections(historical_installation)
    original, observed, _, events, _ = values
    row = observed['members'][1]
    if phase != 'intent':
        row['version'][3:5] = original['members'][1]['version'][3:5]
        row['version'][8] += 1
    if phase in ('chmod', 'repeat'):
        row['version'][2] = original['members'][1]['version'][2]
    if phase == 'repeat':
        repeated = copy.deepcopy(events[-1])
        repeated['body']['version'] = row['version'].copy()
        events.append(repeated)
        row['version'][8] += 1
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    saved = copy.deepcopy(values)
    assert validate_publication(*values, pending_owner_rights=True) is None
    assert values == saved


@pytest.mark.parametrize('change', ['root_access', 'foreign_uid', 'foreign_mode', 'bytes',
    'inode', 'mtime', 'ctime_backwards', 'intent_uid', 'intent_mode', 'intent_inode',
    'intent_extra', 'unknown_trailer', 'missing_stage_removed', 'missing_birth'])
def test_private_permission_recovery_never_adopts_unplanned_change(historical_installation, change):
    from blueprint_pipeline.control_plane_lane_historical_restore_publication import validate_publication
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = rights_projections(historical_installation)
    _, observed, _, events, _ = values
    row, body = observed['members'][1], events[-1]['body']
    if change == 'root_access':
        observed['members'][0]['version'][3] = 123
        observed['target_version'] = observed['members'][0]['version'].copy()
    elif change in ('foreign_uid', 'foreign_mode', 'inode', 'mtime', 'ctime_backwards'):
        slot = {'foreign_uid': 3, 'foreign_mode': 2, 'inode': 1, 'mtime': 7, 'ctime_backwards': 8}[change]
        row['version'][slot] += -1 if change == 'ctime_backwards' else 1
    elif change == 'bytes':
        row['sha256'] = 'sha256:' + 'f' * 64
    elif change in ('intent_uid', 'intent_mode'):
        body[change.removeprefix('intent_')] += 1
    elif change == 'intent_inode':
        body['version'][1] += 1
    elif change == 'intent_extra':
        body['unapproved'] = True
    elif change == 'unknown_trailer':
        events.append(dict(kind='restore_intent', body=dict(phase='unknown')))
    elif change == 'missing_stage_removed':
        events.pop(-2)
    else:
        events.pop(1)
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    with pytest.raises(ValueError, match='restore_publication_changed'):
        validate_publication(*values, pending_owner_rights=True)
