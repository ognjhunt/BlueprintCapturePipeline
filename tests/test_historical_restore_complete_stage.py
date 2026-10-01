"""ADP-009D/day28: full private stage observations are not publication authority.

Pure projections only; actual stage syscall and same-ID recovery have a separate
Linux acceptance case. No filesystem versions are created by the validator.
"""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_restore_staging.py
import copy

import pytest

from tests.test_historical_generation_authority import historical_installation  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_historical_restore_publication import projections

# ruff: noqa: F811


def staged(installed):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    original, published, decision, events, action_id = projections(installed)
    stage = '.historical-restore-' + action_id
    observed = copy.deepcopy(published)
    root, file = observed['members']
    stage_row = copy.deepcopy(root)
    stage_row.update(path=stage, version=events[0]['body']['version'].copy())
    root['version'][5] += 1
    events[0]['body']['parent_version'] = root['version'].copy()
    file['path'] = stage + '/' + file['path']
    observed['members'].insert(1, stage_row)
    observed['member_count'] += 1
    observed['target_version'] = root['version'].copy()
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    events = events[:3]
    events[-1]['body'].update(archive_sha256='sha256:' + '1' * 64,
        archive_size_bytes=123, restored_files=1, restored_logical_bytes=original['logical_payload_bytes'])
    decision['archive'] = dict(sha256='sha256:' + '1' * 64, size_bytes=123)
    return original, observed, decision, events, action_id


def validate(values):
    from blueprint_pipeline.control_plane_lane_historical_restore_staging import validate_complete_stage
    return validate_complete_stage(*values)


def test_complete_private_stage_requires_exact_original_births_and_full_bytes(historical_installation):
    values = staged(historical_installation)
    before = copy.deepcopy(values)
    assert validate(values) is None
    assert values == before


@pytest.mark.parametrize('change', ['missing_birth', 'duplicate_birth', 'inode', 'bytes', 'parent',
    'root', 'owner', 'mode', 'size', 'mtime', 'stage_inode', 'stage_owner', 'stage_links',
    'unknown_member', 'stage_complete_missing', 'archive', 'count', 'unknown_trailer'])
def test_complete_stage_cannot_adopt_an_unproven_or_changed_member(historical_installation, change):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = staged(historical_installation)
    _, observed, decision, events, _ = values
    root, stage, file = observed['members']
    if change == 'missing_birth':
        events.pop(1)
    elif change == 'duplicate_birth':
        events.insert(1, copy.deepcopy(events[1]))
    elif change in ('inode', 'owner', 'mode', 'size', 'mtime'):
        file['version'][{'inode': 1, 'owner': 3, 'mode': 2, 'size': 6, 'mtime': 7}[change]] += 1
    elif change == 'bytes':
        file['sha256'] = 'sha256:' + 'f' * 64
    elif change == 'parent':
        observed['root_version'][1] += 1
    elif change == 'root':
        root['version'][8] += 1
        observed['target_version'] = root['version'].copy()
    elif change.startswith('stage_') and change != 'stage_complete_missing':
        stage['version'][{'stage_inode': 1, 'stage_owner': 3, 'stage_links': 5}[change]] += 1
    elif change == 'unknown_member':
        observed['members'].append(dict(file, path='.unaccounted'))
        observed['member_count'] += 1
    elif change == 'stage_complete_missing':
        events.pop()
    elif change == 'archive':
        decision['archive']['sha256'] = 'sha256:' + 'f' * 64
    elif change == 'count':
        events[-1]['body']['restored_files'] += 1
    else:
        events.append(dict(kind='restore_intent', body=dict(phase='unknown')))
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    with pytest.raises(ValueError, match='restore_stage_changed'):
        validate(values)
