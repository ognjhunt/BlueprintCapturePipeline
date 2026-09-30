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


def observed_rename(values, *, uncertain):
    _, observed, _, events, _ = values
    root, stage, member = observed['members']
    events.append(dict(kind='restore_intent', body=dict(phase='publish_observed', path=member['path'],
        stage_version=stage['version'].copy(), target_version=root['version'].copy(),
        member_version=member['version'].copy(), uncertain=uncertain)))


@pytest.mark.parametrize('uncertain', [False, True])
def test_observed_rename_remains_an_observation_never_a_new_birth(historical_installation, uncertain):
    values = split(historical_installation, moved=True)
    observed_rename(values, uncertain=uncertain)
    before = copy.deepcopy(values)
    result = validate(values)
    assert result['pending'] is None and result['uncertain'] is False
    assert result['published'] == {values[1]['members'][-1]['path']}
    assert values == before


@pytest.mark.parametrize('change', ['duplicate', 'missing_intent', 'extra', 'wrong_path', 'not_bool',
    'inode', 'member_time', 'stage_inode', 'stage_links', 'root_inode', 'root_links'])
def test_observation_is_bound_to_exact_intent_and_parent_effects(historical_installation, change):
    values = split(historical_installation, moved=True)
    observed_rename(values, uncertain=True)
    events = values[3]
    body = events[-1]['body']
    if change == 'duplicate':
        events.append(copy.deepcopy(events[-1]))
    elif change == 'missing_intent':
        events.pop(-2)
    elif change == 'extra':
        body['birth'] = True
    elif change == 'wrong_path':
        body['path'] = 'unknown'
    elif change == 'not_bool':
        body['uncertain'] = 1
    else:
        field, index = {'inode': ('member_version', 1), 'member_time': ('member_version', 7),
            'stage_inode': ('stage_version', 1), 'stage_links': ('stage_version', 5),
            'root_inode': ('target_version', 1), 'root_links': ('target_version', 5)}[change]
        body[field][index] += 1
    with pytest.raises(ValueError):
        validate(values)


def removed_stage(installed):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = split(installed, moved=True)
    observed_rename(values, uncertain=False)
    _, observed, _, events, _ = values
    root, stage, _ = observed['members']
    events.append(dict(kind='restore_intent', body=dict(phase='stage_remove',
        stage_version=stage['version'].copy(), target_version=root['version'].copy())))
    observed['members'].pop(1)
    root['version'][5] -= 1
    root['version'][7] += 1
    root['version'][8] += 1
    observed['member_count'] -= 1
    observed['target_version'] = root['version'].copy()
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    return values


def test_unobserved_stage_removal_requires_original_complete_publish_and_removal_intent(historical_installation):
    values = removed_stage(historical_installation)
    before = copy.deepcopy(values)
    result = validate(values)
    assert result['stage_absent'] is True and result['stage_removal_pending'] is True
    assert result['published'] == {values[1]['members'][-1]['path']}
    assert values == before


@pytest.mark.parametrize('change', ['missing_intent', 'missing_observation', 'root_inode', 'root_links',
    'root_owner', 'root_mode', 'root_mtime', 'root_ctime', 'intent_root', 'intent_stage', 'extra_intent'])
def test_unobserved_stage_removal_cannot_adopt_unknown_namespace_effect(historical_installation, change):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = removed_stage(historical_installation)
    _, observed, _, events, _ = values
    root = observed['members'][0]
    if change == 'missing_intent':
        events.pop()
    elif change == 'missing_observation':
        events.pop(-2)
    elif change.startswith('root_'):
        index = {'inode': 1, 'links': 5, 'owner': 3, 'mode': 2, 'mtime': 7, 'ctime': 8}[change[5:]]
        root['version'][index] += -3 if change in ('root_mtime', 'root_ctime') else 1
    elif change in ('intent_root', 'intent_stage'):
        events[-1]['body']['target_version' if change == 'intent_root' else 'stage_version'][1] += 1
    else:
        events[-1]['body']['uncertain'] = True
    observed['target_version'] = root['version'].copy()
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    with pytest.raises(ValueError):
        validate(values)


@pytest.mark.parametrize('change', ['none', 'ctime', 'rights_before', 'removal_uncertain'])
def test_completed_publication_retains_exact_observed_post_rename_version(historical_installation, change):
    from blueprint_pipeline.control_plane_lane_historical_restore_publication import validate_publication
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = removed_stage(historical_installation)
    original, observed, _, events, _ = values
    body = dict(phase='stage_removed', target_version=observed['target_version'].copy())
    if change == 'removal_uncertain':
        body['uncertain'] = True
    events.append(dict(kind='restore_intent', body=body))
    if change == 'ctime':
        observed['members'][-1]['version'][8] += 1
    if change == 'rights_before':
        before = observed['members'][-1]['version'].copy()
        before[8] += 1
        owner = original['members'][-1]['version']
        events.append(dict(kind='restore_intent', body=dict(phase='owner_rights',
            path=observed['members'][-1]['path'], version=before, uid=owner[3], gid=owner[4], mode=owner[2] & 0o777)))
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    if change in ('ctime', 'rights_before'):
        with pytest.raises(ValueError, match='restore_publication_changed'):
            validate_publication(*values, pending_owner_rights=True)
    else:
        assert validate_publication(*values, pending_owner_rights=True) is None
