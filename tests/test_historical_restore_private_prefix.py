"""ADP-009D/day28: only journaled private births can resume archive extraction.

These are pure parser projections. Native extraction and retained inode proofs
are required separately; no projection supplies filesystem authority.
"""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_restore_staging.py
import copy

import pytest

from tests.test_historical_generation_authority import historical_installation  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_historical_restore_complete_stage import staged

# ruff: noqa: F811


def prefix(installed, *, member=False, pending=False):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = staged(installed)
    original, observed, _, events, action_id = values
    events.pop()
    if not member:
        events.pop()
        observed['members'].pop()
        observed['member_count'] -= 1
        observed['logical_payload_bytes'] = 0
    if pending:
        row = original['members'][1]
        events.append(dict(kind='restore_intent', body=dict(phase='member', path=row['path'],
            stage_path='.historical-restore-' + action_id + '/' + row['path'],
            sha256=row['sha256'], size_bytes=row['size_bytes'])))
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    return values


def validate(values):
    from blueprint_pipeline.control_plane_lane_historical_restore_staging import validate_private_prefix
    return validate_private_prefix(*values)


@pytest.mark.parametrize('member,pending', [(False, False), (False, True), (True, False)])
def test_known_private_prefix_requires_no_new_birth_or_replacement(historical_installation, member, pending):
    values = prefix(historical_installation, member=member, pending=pending)
    before = copy.deepcopy(values)
    result = validate(values)
    assert result == {row['body']['path'] for row in values[3] if row['kind'].startswith('restore_')
                      and row['kind'] != 'restore_intent'}
    assert values == before


@pytest.mark.parametrize('change', ['unknown_file', 'missing_birth', 'inode', 'owner', 'mode',
    'bytes', 'parent', 'unknown_phase', 'pending_existing', 'pending_hash', 'pending_path',
    'duplicate_birth', 'payload_total', 'root_version'])
def test_partial_or_unlogged_kernel_effects_cannot_be_adopted(historical_installation, change):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = prefix(historical_installation, member=True)
    original, observed, _, events, action_id = values
    row = observed['members'][-1]
    if change == 'unknown_file':
        observed['members'].append(dict(row, path=row['path'] + '.unknown'))
        observed['member_count'] += 1
    elif change == 'missing_birth':
        events.pop()
    elif change in ('inode', 'owner', 'mode'):
        row['version'][{'inode': 1, 'owner': 3, 'mode': 2}[change]] += 1
    elif change == 'bytes':
        row['sha256'] = 'sha256:' + 'f' * 64
    elif change == 'parent':
        observed['root_version'][1] += 1
    elif change == 'unknown_phase':
        events.append(dict(kind='restore_intent', body=dict(phase='publish')))
    elif change.startswith('pending'):
        pending = dict(phase='member', path=original['members'][1]['path'],
            stage_path='.historical-restore-' + action_id + '/' + original['members'][1]['path'],
            sha256=original['members'][1]['sha256'], size_bytes=original['members'][1]['size_bytes'])
        if change != 'pending_existing':
            events.pop()
            observed['members'].pop()
            observed['member_count'] -= 1
            observed['logical_payload_bytes'] = 0
            pending['sha256' if change == 'pending_hash' else 'stage_path'] = 'unknown'
        events.append(dict(kind='restore_intent', body=pending))
    elif change == 'duplicate_birth':
        events.append(copy.deepcopy(events[-1]))
    elif change == 'payload_total':
        observed['logical_payload_bytes'] += 1
    else:
        observed['members'][0]['version'][8] += 1
        observed['target_version'] = observed['members'][0]['version'].copy()
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    with pytest.raises(ValueError, match='restore_stage_changed'):
        validate(values)
