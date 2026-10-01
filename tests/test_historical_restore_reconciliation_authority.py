"""ADP-009D/day28: unfinished restore needs separate current DELETE rights.

These protected-policy tests prove authorization predicates only. They supply
no original restore, creation receipt, reader clearance or native execution.
"""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_restore_reconciliation_authority.py
import json

import pytest

from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_historical_generation_authority import historical_installation  # noqa: F401

# ruff: noqa: F811


def authorize(installed, *, principal='operator', owner='owner', expiry=1090):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_authority import _current_delete
    with authority._session(installed[0], authority._Operation(1030, lambda: 0)) as (files, config, _):
        return _current_delete(files, config, installed[0], principal, owner, 1030, expiry, 1100)


def test_current_restore_handling_never_implies_partial_file_delete(historical_installation):
    policy = historical_installation[2]
    value = json.loads(policy.read_bytes())
    value['principals'][0]['allowed_actions'].remove('delete')
    policy.write_text(json.dumps(value))
    before = {path.name: path.read_bytes() for path in historical_installation[3].iterdir()}
    with pytest.raises(ValueError):
        authorize(historical_installation)
    assert {path.name: path.read_bytes() for path in historical_installation[3].iterdir()} == before


@pytest.mark.parametrize('change', [dict(principal='unknown'), dict(owner='other'), dict(expiry=1030),
    dict(expiry=1101), dict(expiry=True), dict(expiry=float('nan'))])
def test_exact_owner_and_original_expiry_cannot_be_renewed(historical_installation, change):
    with pytest.raises(ValueError):
        authorize(historical_installation, **change)


def test_current_explicit_delete_returns_only_config_and_policy_selectors(historical_installation):
    from tests.test_historical_generation_authority import selector
    before = (historical_installation[1] / 'one.log').read_bytes()
    assert authorize(historical_installation) == (
        selector(historical_installation[0].read_bytes()), selector(historical_installation[2].read_bytes()))
    assert (historical_installation[1] / 'one.log').read_bytes() == before


def test_missing_reconciliation_approval_survives_authority_session(historical_installation):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_authority import select_reconciliation
    before = {path.name: path.read_bytes() for path in historical_installation[3].iterdir()}
    with pytest.raises(ValueError, match='historical_generation_restore_reconciliation_approval_missing'):
        with authority._session(historical_installation[0], authority._Operation(1030, lambda: 0)) as (files, config, store):
            # The missing protected decision must refuse before any selector,
            # packet, native fact or effect can be accepted.
            select_reconciliation(files, config, store, historical_installation[0],
                None, [], 'a' * 32, None, 1030)
    assert {path.name: path.read_bytes() for path in historical_installation[3].iterdir()} == before


def test_unreadable_reconciliation_stays_authority_io_refusal(historical_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_authority import select_reconciliation
    def denied(*args, **kwargs):
        raise PermissionError('unreadable protected decision')
    monkeypatch.setattr(authority._Store, 'read', denied)
    with pytest.raises(ValueError, match='historical_generation_authority_io_unavailable'):
        with authority._session(historical_installation[0], authority._Operation(1030, lambda: 0)) as (files, config, store):
            select_reconciliation(files, config, store, historical_installation[0],
                None, [], 'a' * 32, None, 1030)


def test_expired_discard_requires_distinct_bounded_explicit_attempt():
    # Protocol projection only: no native authority, receipt or owner decision.
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_authority import (
        decision_id, _attempt_proposal)
    action, head = 'a' * 32, 'sha256:' + 'b' * 64
    first = dict(decision_id=decision_id(action, head), expires_at_epoch=1040)
    raw = dict(sha256='sha256:' + 'c' * 64, size_bytes=123)
    assert decision_id(action, head, 1) != first['decision_id']
    assert _attempt_proposal([(first, raw)], 1039) == (0, [])
    assert _attempt_proposal([(first, raw)], 1040) == (1, [dict(decision_id=first['decision_id'], decision=raw)])
    with pytest.raises(ValueError):
        _attempt_proposal([(first, raw)] * 8, 1040)


@pytest.mark.parametrize('attempt', [True, -1, 8, '1'])
def test_discard_attempt_identity_stays_finite(attempt):
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_authority import decision_id
    with pytest.raises(ValueError):
        decision_id('a' * 32, 'sha256:' + 'b' * 64, attempt)


def resume_projection():
    # Structural projections never supply authenticated journal or authority.
    link = dict(decision_id='a' * 32, decision=dict(sha256='sha256:' + 'b' * 64, size_bytes=123),
                original_head_event_digest='sha256:' + 'c' * 64)
    grant = dict(decision_id='d' * 32, decision=dict(sha256='sha256:' + 'e' * 64, size_bytes=234))
    head = dict(kind='restore_intent', event_digest=link['original_head_event_digest'], body=dict(phase='member'))
    intent = dict(kind='restore_intent', event_digest='sha256:' + 'f' * 64,
                  body=dict(phase='reconcile_intent', **link))
    resume = dict(kind='restore_intent', event_digest='sha256:' + '1' * 64,
                  body=dict(phase='reconcile_delete_resume', **link, delete_resume=grant))
    return [head, intent, resume], link, grant


def test_pending_resume_is_retained_separately_from_original_intent():
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_replay import reconciliation_bindings
    events, link, grant = resume_projection()
    values = reconciliation_bindings(events)
    assert values == [(events[:1], link['decision_id'], link['decision'], tuple(events[1:]))]
    completed = dict(kind='restore_intent', body=dict(phase='reconciled', **link, delete_resume=grant))
    values = reconciliation_bindings(events + [completed])
    assert values[0][3] == tuple(events[1:] + [completed])


@pytest.mark.parametrize('change', ['missing', 'selector', 'wrong_original', 'changed_completed', 'unbound_completed'])
def test_resume_projection_cannot_silently_select_a_different_grant(change):
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_replay import reconciliation_bindings
    events, link, grant = resume_projection()
    if change == 'missing':
        events[2]['body'].pop('delete_resume')
    elif change == 'selector':
        events[2]['body']['delete_resume']['decision']['size_bytes'] = True
    elif change == 'wrong_original':
        events[2]['body']['decision_id'] = '9' * 32
    else:
        body = dict(phase='reconciled', **link)
        if change == 'changed_completed':
            body['delete_resume'] = dict(grant, decision_id='9' * 32)
        events.append(dict(kind='restore_intent', body=body))
    with pytest.raises(ValueError):
        reconciliation_bindings(events)


@pytest.mark.parametrize('change', [None, 'rotated', 'missing_resume', 'other_resume', 'observation_only'])
def test_effect_pin_cannot_rotate_to_an_unrecorded_or_observation_grant(change):
    from types import SimpleNamespace
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_worker import _effect_pin
    events, _, grant = resume_projection()
    effect = (dict(decision_id=grant['decision_id'], action='discard_unfinished_restore_row'), {}, grant['decision'])
    worker = SimpleNamespace(effect_selected=[effect], resume_selected=[effect],
        reconciliations=[([], 'a' * 32, {}, tuple(events[1:]))])
    selected = dict(grant)
    if change == 'rotated':
        worker.effect_selected = [(dict(effect[0], decision_id='8' * 32), {}, effect[2])]
    elif change == 'missing_resume':
        worker.reconciliations[0] = ([], 'a' * 32, {}, (events[1],))
    elif change == 'other_resume':
        events[2]['body']['delete_resume'] = dict(grant, decision_id='8' * 32)
    elif change == 'observation_only':
        worker.resume_selected = [(dict(effect[0], action='observe_unfinished_restore_row_absence'), {}, effect[2])]
    if change is None:
        _effect_pin(worker, selected)
    else:
        with pytest.raises(ValueError):
            _effect_pin(worker, selected)


@pytest.mark.parametrize('present,pending,recorded_grant,allowed', [
    (True, True, False, True), (False, True, True, True),
    (False, True, False, False), (False, False, True, False), (True, False, False, False)])
def test_absent_delete_completion_requires_an_already_recorded_exact_resume(present, pending, recorded_grant, allowed):
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_worker import _resume_effect_scope
    events, _, grant = resume_projection()
    recorded = tuple(events[1:] if recorded_grant else events[1:2])
    if allowed:
        _resume_effect_scope(present, pending, recorded, grant)
    else:
        with pytest.raises(ValueError):
            _resume_effect_scope(present, pending, recorded, grant)
