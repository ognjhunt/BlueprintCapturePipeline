"""ADP-009D/day28: exact absence is observation-only, never renewed DELETE.

Parser projections supply no protected owner decision, birth, native syscall,
reader clearance or execution authority. Connected Linux proof is separate.
"""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_restore_absent_observation.py
# Covers: src/blueprint_pipeline/control_plane_lane_historical_restore_absent_authority.py
# Covers: src/blueprint_pipeline/control_plane_lane_historical_restore_reconciliation_replay.py
import copy

import pytest

from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_historical_generation_authority import historical_installation  # noqa: F401
from tests.test_historical_restore_reconciliation_scope import uncertain_member, removed_projection

# ruff: noqa: F811


def absent(installed):
    values = uncertain_member(installed)
    _, current = removed_projection(values)
    events = values[3]
    events[-1]['event_digest'] = 'sha256:' + 'd' * 64
    events.append(dict(kind='restore_intent', event_digest='sha256:' + 'e' * 64,
        body=dict(phase='reconcile_intent', decision_id='c' * 32,
            decision=dict(sha256='sha256:' + 'a' * 64, size_bytes=100),
            original_head_event_digest=events[-1]['event_digest'])))
    return values[0], values[1], current, values[2], events, values[4]


def observe(values):
    from blueprint_pipeline.control_plane_lane_historical_restore_absent_observation import absent_observation_scope
    return absent_observation_scope(*values)


def test_absent_scope_cannot_authorize_removal_or_birth(historical_installation):
    values = absent(historical_installation)
    before = copy.deepcopy(values)
    result = observe(values)
    assert result['execution_authorized'] is False
    assert result['observation_only'] is True
    assert result['permits_removal'] is False
    assert result['original_reconciliation_intent_digest'] == values[4][-1]['event_digest']
    assert result['observed_generation_digest'] == values[2]['generation_digest']
    assert result['parent_version'] == values[2]['members'][1]['version']
    assert result['credited_removed_allocated_bytes'] == 0
    assert values == before


def test_absence_after_real_resume_keeps_original_intent_distinct(historical_installation):
    # Explicit structural projection; no protected/native receipt is implied.
    values = absent(historical_installation)
    events = values[4]
    base = {key: events[-1]['body'][key] for key in ('decision_id', 'decision', 'original_head_event_digest')}
    events.append(dict(kind='restore_intent', event_digest='sha256:' + 'f' * 64,
        body=dict(phase='reconcile_delete_resume', **base,
                  delete_resume=dict(decision_id='9' * 32, decision=dict(sha256='sha256:' + '8' * 64, size_bytes=200)))))
    result = observe(values)
    assert result['permits_removal'] is False and result['credited_removed_allocated_bytes'] == 0
    assert result['original_reconciliation'] == base
    assert result['original_reconciliation_intent_digest'] == events[-1]['event_digest']


@pytest.mark.parametrize('change', ['present', 'extra', 'lost_known', 'inode', 'mode', 'links', 'time',
    'parent', 'total', 'digest', 'missing_intent', 'phase', 'wrong_original_head', 'unselected_decision'])
def test_changed_or_present_namespace_never_becomes_observation_resume_authority(historical_installation, change):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    values = list(absent(historical_installation))
    observed, events = values[2], values[4]
    if change == 'present':
        observed = values[2] = copy.deepcopy(values[1])
    elif change == 'extra':
        observed['members'].append(dict(observed['members'][1], path=observed['members'][1]['path'] + '/extra'))
        observed['member_count'] += 1
    elif change == 'lost_known':
        observed['members'].pop()
        observed['member_count'] -= 1
    elif change in ('inode', 'mode', 'links', 'time'):
        observed['members'][1]['version'][{'inode': 1, 'mode': 2, 'links': 5, 'time': 8}[change]] += -10 if change == 'time' else 1
    elif change == 'parent':
        observed['root_version'][1] += 1
    elif change == 'total':
        observed['logical_payload_bytes'] += 1
    elif change == 'missing_intent':
        events.pop()
    elif change == 'phase':
        events[-1]['body']['phase'] = 'reconciled'
    elif change == 'wrong_original_head':
        events[-1]['body']['original_head_event_digest'] = 'sha256:' + 'f' * 64
    elif change == 'unselected_decision':
        events[-1]['body']['decision']['sha256'] = 'unselected'
    observed['target_version'] = observed['members'][0]['version'].copy()
    observed['generation_digest'] = canonical_digest(observed, digest_field='generation_digest')
    if change == 'digest':
        observed['generation_digest'] = 'sha256:' + 'f' * 64
    with pytest.raises(ValueError):
        observe(values)


def observation_rights(installed, **changes):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline.control_plane_lane_historical_restore_absent_authority import _current_observation
    options = dict(principal='operator', owner='owner', moment=1030, expiry=1090, original_expiry=1100)
    with authority._session(installed[0], authority._Operation(1030, lambda: 0)) as (files, config, _):
        return _current_observation(files, config, installed[0], **(options | changes))


def test_observation_rights_never_grant_or_require_delete(historical_installation):
    import json
    from tests.test_historical_generation_authority import selector
    policy = historical_installation[2]
    value = json.loads(policy.read_bytes())
    value['principals'][0]['allowed_actions'].remove('delete')
    policy.write_text(json.dumps(value))
    before = {path.name: path.read_bytes() for path in historical_installation[3].iterdir()}
    assert observation_rights(historical_installation) == (
        selector(historical_installation[0].read_bytes()), selector(policy.read_bytes()))
    assert {path.name: path.read_bytes() for path in historical_installation[3].iterdir()} == before


@pytest.mark.parametrize('change', [dict(owner='other'), dict(principal='unknown'),
    dict(expiry=True), dict(expiry=float('nan')), dict(expiry=1030), dict(expiry=1101)])
def test_observation_rights_cannot_change_owner_or_original_expiry(historical_installation, change):
    with pytest.raises(ValueError):
        observation_rights(historical_installation, **change)


@pytest.mark.parametrize('action', ['register', 'offload'])
def test_delete_alone_cannot_authorize_observation_resume(historical_installation, action):
    import json
    policy = historical_installation[2]
    value = json.loads(policy.read_bytes())
    value['principals'][0]['allowed_actions'].remove(action)
    policy.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        observation_rights(historical_installation)


def observation_projection(installed):
    values = list(absent(installed))
    packet = observe(values)
    values[4].append(dict(kind='restore_intent', event_digest='sha256:' + 'f' * 64,
        body=dict(values[4][-1]['body'], phase='reconciled',
            parent_path=packet['parent_path'], parent_version=packet['parent_version'], uncertain=True,
            credited_removed_allocated_bytes=0, observation_resume=dict(decision_id='b' * 32,
                decision=dict(sha256='sha256:' + 'c' * 64, size_bytes=300)))))
    return values


def test_observation_projection_updates_parent_without_inventing_birth(historical_installation):
    from blueprint_pipeline.control_plane_lane_historical_restore_staging import stage_birth_versions
    values = observation_projection(historical_installation)
    versions, births = stage_birth_versions(values[0], values[3], values[4], values[5],
                                             complete=False, tick=lambda: None)
    assert births == {''}
    assert versions[values[4][-1]['body']['parent_path']] == values[4][-1]['body']['parent_version']
    assert not any(event['kind'] == 'restore_member' for event in values[4])


@pytest.mark.parametrize('change', ['id', 'sha', 'size', 'bool_size', 'extra', 'effect_credit', 'parent'])
def test_observation_projection_rejects_unselected_resume_or_effect(historical_installation, change):
    from blueprint_pipeline.control_plane_lane_historical_restore_staging import stage_birth_versions
    values = observation_projection(historical_installation)
    body = values[4][-1]['body']
    resume = body['observation_resume']
    if change == 'id':
        resume['decision_id'] = 'wrong'
    elif change == 'sha':
        resume['decision']['sha256'] = 'unselected'
    elif change == 'size':
        resume['decision']['size_bytes'] = 32769
    elif change == 'bool_size':
        resume['decision']['size_bytes'] = True
    elif change == 'extra':
        resume['permits_removal'] = True
    elif change == 'effect_credit':
        body['credited_removed_allocated_bytes'] = 1
    else:
        body['parent_version'][1] += 1
    with pytest.raises(ValueError):
        stage_birth_versions(values[0], values[3], values[4], values[5], complete=False, tick=lambda: None)


def test_expired_observation_requires_a_new_bounded_identity():
    from blueprint_pipeline.control_plane_lane_historical_restore_absent_authority import decision_id, _proposal
    first = dict(decision_id='b' * 32, expires_at_epoch=1040)
    previous = dict(sha256='sha256:' + 'c' * 64, size_bytes=100)
    assert _proposal([(first, previous)], 1039) == (0, [])
    assert _proposal([(first, previous)], 1040) == (1, [dict(decision_id=first['decision_id'], decision=previous)])
    assert decision_id('a' * 32, 'sha256:' + 'f' * 64, 0) != decision_id('a' * 32, 'sha256:' + 'f' * 64, 1)
    with pytest.raises(ValueError):
        _proposal([(first, previous)] * 8, 1040)


@pytest.mark.parametrize('attempt', [True, -1, 8, '1'])
def test_observation_attempt_identity_cannot_expand_unboundedly(attempt):
    from blueprint_pipeline.control_plane_lane_historical_restore_absent_authority import decision_id
    with pytest.raises(ValueError):
        decision_id('a' * 32, 'sha256:' + 'f' * 64, attempt)


@pytest.mark.parametrize('changed', [None, 'id', 'raw', 'missing'])
def test_consumed_observation_selects_exact_historical_attempt(changed):
    from blueprint_pipeline.control_plane_lane_historical_restore_absent_authority import _observation_index
    first = (dict(decision_id='a' * 32), dict(sha256='sha256:' + 'b' * 64, size_bytes=100))
    second = (dict(decision_id='c' * 32), dict(sha256='sha256:' + 'd' * 64, size_bytes=200))
    completed = dict(body=dict(observation_resume=dict(decision_id=first[0]['decision_id'], decision=first[1])))
    if changed == 'id':
        completed['body']['observation_resume']['decision_id'] = 'e' * 32
    elif changed == 'raw':
        completed['body']['observation_resume']['decision'] = second[1]
    elif changed == 'missing':
        completed['body'].clear()
    assert _observation_index([first, second], None) == 1
    if changed is None:
        assert _observation_index([first, second], completed) == 0
    else:
        with pytest.raises(ValueError):
            _observation_index([first, second], completed)
