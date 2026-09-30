"""ADP-009D/day28: reviewable unlogged state is never deletion authority.

Pure projections isolate one pending unlogged syscall. Exact owner approval and
actual installed cleanup/restore remain separate required acceptance gates.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_restore_reconciliation_scope.py
import copy

import pytest

from tests.test_historical_generation_authority import historical_installation  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_historical_restore_private_prefix import prefix

# ruff: noqa: F811


def uncertain_member(installed):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = prefix(installed, member=True)
    original, observed, _, events, action_id = values
    birth = events.pop()['body']
    member = observed['members'][-1]
    member['size_bytes'] = 1
    member['version'][6] = 1
    member['sha256'] = 'sha256:' + 'f' * 64
    old = original['members'][-1]
    events.append(dict(kind='restore_intent', body=dict(phase='member', path=old['path'],
        stage_path=birth['stage_path'], sha256=old['sha256'], size_bytes=old['size_bytes'])))
    observed['logical_payload_bytes'] = 1
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    return values


def scope(values):
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_scope import unknown_creation_scope
    return unknown_creation_scope(*values)


def test_only_exact_unlogged_pending_member_is_reviewable_not_authorized(historical_installation):
    values = uncertain_member(historical_installation)
    before = copy.deepcopy(values)
    result = scope(values)
    assert result['execution_authorized'] is False
    assert result['owner_reconciliation_required'] is True
    assert result['remove_member'] == values[1]['members'][-1]
    assert result['parent_before'] == values[3][0]['body']['version']
    assert result['observed_generation_digest'] == values[1]['generation_digest']
    assert values == before


@pytest.mark.parametrize('change', ['missing_intent', 'born_member', 'extra_row', 'wrong_path', 'wrong_kind',
    'owner', 'mode', 'inode_parent', 'parent_links', 'parent_owner', 'root_inode', 'parent_identity',
    'size', 'wrong_total', 'wrong_count', 'complete', 'publication', 'wrong_digest', 'known_alias'])
def test_unknown_scope_cannot_include_known_birth_or_unselected_namespace(historical_installation, change):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = uncertain_member(historical_installation)
    original, observed, _, events, _ = values
    root, parent, member = observed['members']
    if change == 'missing_intent':
        events.pop()
    elif change == 'born_member':
        values = prefix(historical_installation, member=True)
    elif change == 'extra_row':
        observed['members'].append(dict(member, path=member['path'] + '.extra'))
        observed['member_count'] += 1
    elif change == 'wrong_path':
        events[-1]['body']['path'] = 'not-selected'
    elif change == 'wrong_kind':
        events[-1]['body']['phase'] = 'directory'
    elif change == 'known_alias':
        member['version'][:2] = parent['version'][:2]
    elif change in ('owner', 'mode'):
        member['version'][3 if change == 'owner' else 2] += 1
    elif change.startswith('parent_') and change != 'parent_identity':
        parent['version'][{'parent_links': 5, 'parent_owner': 3}[change]] += 1
    elif change == 'inode_parent':
        parent['version'][1] += 1
    elif change == 'root_inode':
        root['version'][1] += 1
    elif change == 'parent_identity':
        observed['root_version'][1] += 1
    elif change == 'size':
        member['size_bytes'] = original['members'][-1]['size_bytes'] + 1
        member['version'][6] = member['size_bytes']
        observed['logical_payload_bytes'] = member['size_bytes']
    elif change == 'wrong_total':
        observed['logical_payload_bytes'] += 1
    elif change == 'wrong_count':
        observed['member_count'] -= 1
    elif change in ('complete', 'publication'):
        events.append(dict(kind='restore_intent', body=dict(phase='stage_complete' if change == 'complete' else 'publish')))
    if change != 'born_member':
        observed['target_version'] = root['version'].copy()
        observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    if change == 'wrong_digest':
        observed['generation_digest'] = 'sha256:' + 'a' * 64
    with pytest.raises(ValueError):
        scope(values)


def uncertain_root(installed):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = prefix(installed)
    _, observed, _, events, action_id = values
    root_birth = events.pop()['body']
    events.append(dict(kind='restore_intent', body=dict(phase='directory', path='',
        stage_path='.historical-restore-' + action_id)))
    observed['members'][1]['version'] = root_birth['version'].copy()
    # Explicit parser-only Linux empty-directory projection. This is neither
    # an observed inode nor a birth receipt; real native proof remains separate.
    observed['members'][1]['version'][5] = 2
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    return values


def test_only_empty_unlogged_root_can_be_proposed_for_distinct_owner_discard(historical_installation):
    values = uncertain_root(historical_installation)
    result = scope(values)
    assert result['remove_member']['kind'] == 'directory'
    assert result['execution_authorized'] is False
    assert result['parent_path'] == '' and result['parent_before'] == values[2]['tombstone_version']


def test_unlogged_root_with_extra_descendant_is_not_remove_only_scope(historical_installation):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = uncertain_root(historical_installation)
    values[1]['members'].append(copy.deepcopy(values[1]['members'][1]))
    values[1]['members'][-1]['path'] += '/unknown-child'
    values[1]['member_count'] += 1
    values[1]['generation_digest'] = canonical_digest(values[1], digest_field='generation_digest')
    with pytest.raises(ValueError):
        scope(values)


def removed_projection(values):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    approved = scope(values)
    observed = copy.deepcopy(values[1])
    observed['members'] = [row for row in observed['members'] if row['path'] != approved['remove_member']['path']]
    parent = next(row for row in observed['members'] if row['path'] == approved['parent_path'])
    parent['version'][5] -= int(approved['remove_member']['kind'] == 'directory')
    parent['version'][7] += 1
    parent['version'][8] += 1
    observed['member_count'] -= 1
    observed['logical_payload_bytes'] -= approved['remove_member']['size_bytes']
    observed['target_version'] = observed['members'][0]['version'].copy()
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    return approved, observed


@pytest.mark.parametrize('root', [False, True])
def test_exact_absence_is_uncertain_observation_never_birth_or_freed_credit(historical_installation, root):
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_scope import reconciled_parent
    values = uncertain_root(historical_installation) if root else uncertain_member(historical_installation)
    selected, observed = removed_projection(values)
    before = copy.deepcopy((values, selected, observed))
    version = reconciled_parent(values[1], observed, selected)
    assert version == next(row['version'] for row in observed['members'] if row['path'] == selected['parent_path'])
    assert (values, selected, observed) == before


@pytest.mark.parametrize('change', ['extra', 'still_present', 'lost_known', 'known_inode', 'known_bytes',
    'parent_inode', 'parent_owner', 'parent_links', 'parent_time', 'ancestor', 'total', 'count', 'digest'])
def test_uncertain_absence_cannot_adopt_other_namespace_changes(historical_installation, change):
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_scope import reconciled_parent
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = uncertain_member(historical_installation)
    selected, observed = removed_projection(values)
    parent = observed['members'][1]
    if change == 'extra':
        observed['members'].append(dict(parent, path=parent['path'] + '/extra'))
        observed['member_count'] += 1
    elif change == 'still_present':
        observed = copy.deepcopy(values[1])
    elif change == 'lost_known':
        observed['members'].pop()
        observed['member_count'] -= 1
    elif change == 'known_inode':
        observed['members'][0]['version'][1] += 1
    elif change == 'known_bytes':
        observed['members'][0]['size_bytes'] = 1
    elif change.startswith('parent_'):
        index = {'parent_inode': 1, 'parent_owner': 3, 'parent_links': 5, 'parent_time': 8}[change]
        parent['version'][index] += -10 if change == 'parent_time' else 1
    elif change == 'ancestor':
        observed['root_version'][1] += 1
    elif change == 'total':
        observed['logical_payload_bytes'] += 1
    elif change == 'count':
        observed['member_count'] += 1
    observed['target_version'] = observed['members'][0]['version'].copy()
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    if change == 'digest':
        observed['generation_digest'] = 'sha256:' + 'f' * 64
    with pytest.raises(ValueError):
        reconciled_parent(values[1], observed, selected)


@pytest.mark.parametrize('root', [False, True])
def test_reconciled_parent_is_not_replacement_birth_and_known_births_survive(historical_installation, root):
    from blueprint_pipeline.control_plane_lane_historical_restore_staging import stage_birth_versions, validate_private_prefix
    values = uncertain_root(historical_installation) if root else uncertain_member(historical_installation)
    selected, observed = removed_projection(values)
    parent_version = next(row['version'] for row in observed['members'] if row['path'] == selected['parent_path'])
    # Explicit parser-only event projection, never a protected decision or
    # actual removal receipt. The worker must authenticate both independently.
    binding = dict(decision_id='d' * 32, decision=dict(sha256='sha256:' + 'e' * 64, size_bytes=100),
                   original_head_event_digest='sha256:' + 'f' * 64)
    values[3].extend([dict(kind='restore_intent', body=dict(phase='reconcile_intent', **binding)),
        dict(kind='restore_intent', body=dict(phase='reconciled', **binding,
            parent_path=selected['parent_path'], parent_version=parent_version, uncertain=True,
            credited_removed_allocated_bytes=0))])
    versions, births = stage_birth_versions(values[0], values[2], values[3], values[4], complete=False, tick=lambda: None)
    assert versions[selected['parent_path']] == parent_version
    assert births == (set() if root else {''})
    assert validate_private_prefix(values[0], observed, values[2], values[3], values[4]) == births
    assert not any(event['kind'] == 'restore_member' for event in values[3])


@pytest.mark.parametrize('change', ['missing_binding', 'bad_selector', 'credit', 'links', 'inode',
    'owner', 'parent', 'time', 'missing_intent', 'unknown_phase'])
def test_reconciliation_projection_cannot_add_birth_authority_or_unplanned_changes(historical_installation, change):
    from blueprint_pipeline.control_plane_lane_historical_restore_staging import stage_birth_versions
    values = uncertain_member(historical_installation)
    selected, observed = removed_projection(values)
    binding = dict(decision_id='d' * 32, decision=dict(sha256='sha256:' + 'e' * 64, size_bytes=100),
                   original_head_event_digest='sha256:' + 'f' * 64)
    intent = dict(phase='reconcile_intent', **copy.deepcopy(binding))
    body = dict(phase='reconciled', **binding, parent_path=selected['parent_path'],
        parent_version=observed['members'][1]['version'].copy(), uncertain=True, credited_removed_allocated_bytes=0)
    if change == 'missing_binding':
        intent.pop('decision')
    elif change == 'bad_selector':
        intent['decision']['sha256'] = 'unselected'
    elif change == 'credit':
        body['credited_removed_allocated_bytes'] = 1
    elif change in ('links', 'inode', 'owner', 'time'):
        body['parent_version'][{'links': 5, 'inode': 1, 'owner': 3, 'time': 8}[change]] += -10 if change == 'time' else 1
    elif change == 'parent':
        body['parent_path'] += '/other'
    elif change == 'unknown_phase':
        body['phase'] = 'birth'
    if change != 'missing_intent':
        values[3].append(dict(kind='restore_intent', body=intent))
    values[3].append(dict(kind='restore_intent', body=body))
    with pytest.raises(ValueError):
        stage_birth_versions(values[0], values[2], values[3], values[4], complete=False, tick=lambda: None)
