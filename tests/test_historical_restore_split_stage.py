"""ADP-009D/day28: a publish intent only reconciles its original born inode.

These are non-authorizing metadata projections. Actual rename interruption and
recovery must also pass the installed disposable Linux acceptance case.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_restore_split.py
import copy

import pytest

from tests.test_historical_generation_authority import historical_installation  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_historical_restore_complete_stage import staged

# ruff: noqa: F811


def split(installed, *, moved):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = staged(installed)
    _, observed, _, events, _ = values
    root, stage, member = observed['members']
    intent = dict(phase='publish', path=member['path'].split('/')[-1],
        stage_version=stage['version'].copy(), target_version=root['version'].copy(),
        member_version=member['version'].copy())
    events.append(dict(kind='restore_intent', body=intent))
    if moved:
        member['path'] = intent['path']
        member['version'][8] += 1
        for row in (root, stage):
            row['version'][7] += 1
            row['version'][8] += 1
        observed['target_version'] = root['version'].copy()
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    return values


def validate(values):
    from blueprint_pipeline.control_plane_lane_historical_restore_split import validate_split_stage
    return validate_split_stage(*values)


@pytest.mark.parametrize('moved', [False, True])
def test_same_intent_reconciles_only_absent_source_or_absent_destination(historical_installation, moved):
    values = split(historical_installation, moved=moved)
    before = copy.deepcopy(values)
    result = validate(values)
    name = values[3][-1]['body']['path']
    assert result['published'] == ({name} if moved else set())
    assert result['pending'] == name
    assert result['uncertain'] is moved
    assert values == before


@pytest.mark.parametrize('change', ['both', 'neither', 'inode', 'bytes', 'size', 'mode', 'owner',
    'mtime', 'blocks', 'root_inode', 'root_links', 'root_owner', 'stage_inode', 'stage_links',
    'stage_owner', 'parent', 'intent_member', 'intent_root', 'intent_stage', 'intent_extra',
    'missing_complete', 'missing_birth', 'extra', 'unknown_trailer'])
def test_split_never_adopts_unknown_or_changed_bytes(historical_installation, change):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = split(historical_installation, moved=True)
    _, observed, _, events, action_id = values
    root, stage, member = observed['members']
    if change == 'both':
        observed['members'].append(dict(member, path='.historical-restore-' + action_id + '/' + member['path']))
    elif change == 'neither':
        observed['members'].pop()
    elif change in ('inode', 'size', 'mode', 'owner', 'mtime', 'blocks'):
        member['version'][{'inode': 1, 'size': 6, 'mode': 2, 'owner': 3, 'mtime': 7, 'blocks': 9}[change]] += 1
    elif change == 'bytes':
        member['sha256'] = 'sha256:' + 'f' * 64
    elif change.startswith('root_') or change.startswith('stage_'):
        row = root if change.startswith('root_') else stage
        row['version'][{'inode': 1, 'links': 5, 'owner': 3}[change.split('_')[1]]] += 1
    elif change == 'parent':
        observed['root_version'][1] += 1
    elif change in ('intent_member', 'intent_root', 'intent_stage'):
        events[-1]['body'][{'intent_member': 'member_version', 'intent_root': 'target_version',
                            'intent_stage': 'stage_version'}[change]][1] += 1
    elif change == 'intent_extra':
        events[-1]['body']['adopt'] = True
    elif change == 'missing_complete':
        events.pop(-2)
    elif change == 'missing_birth':
        events.pop(1)
    elif change == 'extra':
        observed['members'].append(dict(member, path='unknown'))
    else:
        events.append(dict(kind='restore_intent', body=dict(phase='unknown')))
    observed['member_count'] = len(observed['members'])
    observed['target_version'] = root['version'].copy()
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    with pytest.raises(ValueError):
        validate(values)
