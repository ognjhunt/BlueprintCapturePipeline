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
    'size', 'wrong_total', 'wrong_count', 'complete', 'publication', 'wrong_digest'])
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
