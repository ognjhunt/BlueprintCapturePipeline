"""Restore control flow only; doubles provide no native or owner authority."""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_restore_prefix.py
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def worker_and_events():
    action_id = 'a' * 32
    events = [dict(kind='intent', action_id=action_id, event_digest='sha256:' + '1' * 64),
              dict(kind='restore_intent', action_id=action_id, event_digest='sha256:' + '2' * 64,
                   body=dict(phase='reconciled', credited_removed_allocated_bytes=0))]
    manifest = dict(target_path='/private/selected', generation_digest='sha256:' + '3' * 64)
    decision = dict(action='restore', action_id=action_id, owner='original owner',
                    manifest=dict(sha256='sha256:' + '4' * 64, size_bytes=100))
    journal = SimpleNamespace(head=dict(event_digest=events[-1]['event_digest']))
    calls = []
    @contextmanager
    def checkpoint(**kwargs):
        assert kwargs == dict(journal=True)
        calls.append('checkpoint')
        yield None, None, journal
        calls.append('current-grants-rechecked')
    worker = SimpleNamespace(action_id=action_id, selected=(None, decision, manifest, None),
        operation=SimpleNamespace(remaining=lambda: calls.append('original-clock')), checkpoint=checkpoint)
    return worker, events, journal, calls


def test_reconciled_prefix_stops_before_readback_or_second_sandbox(monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_restore_prefix as prefix
    from blueprint_pipeline import control_plane_lane_historical_restore_worker as restore
    from blueprint_pipeline import control_plane_lane_historical_restore_reconciliation_worker as reconciliation
    worker, events, _, calls = worker_and_events()
    before = events[:1]
    observed = dict(current_original_tree='control flow double')
    monkeypatch.setattr(prefix.generation, 'inventory_historical_generation', lambda *args, **kwargs: observed)
    monkeypatch.setattr(prefix, 'selected_restore_bounds', lambda selected: None)
    count = 0
    def validate(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            raise prefix.generation.HistoricalGenerationError('historical_generation_restore_stage_changed')
        assert args[3] == events
        return set()
    monkeypatch.setattr(prefix, 'validate_private_prefix', validate)
    def reconciled(*args):
        calls.append('durable-reconciled')
        return events, observed
    monkeypatch.setattr(reconciliation, 'reconcile_unlogged_creation', reconciled)
    def forbidden(*args, **kwargs):
        raise AssertionError('restricted worker attempted another stage')
    monkeypatch.setattr(restore, '_readback', forbidden)
    monkeypatch.setattr(restore, '_resources', forbidden)
    receipt = prefix.recover_private_prefix(worker, before, (), lambda: 0)
    assert receipt['status'] == 'pending' and receipt['action_id'] == worker.action_id
    assert receipt['continuation_event_digest'] == events[-1]['event_digest']
    assert receipt['original_intent_event_digest'] == events[0]['event_digest']
    assert receipt['restored_files'] == receipt['restored_logical_bytes'] == receipt['credited_removed_allocated_bytes'] == 0
    assert receipt['owner_access_reopened'] is False
    assert count == 2 and calls == ['original-clock', 'durable-reconciled', 'checkpoint',
                                  'current-grants-rechecked', 'original-clock']


@pytest.mark.parametrize('change', ['head', 'phase', 'action', 'credit', 'expired'])
def test_pending_continuation_cannot_substitute_unknown_head_or_expired_authority(change):
    from blueprint_pipeline.control_plane_lane_historical_restore_prefix import _pending_continuation
    worker, events, journal, _ = worker_and_events()
    if change == 'head':
        journal.head['event_digest'] = 'sha256:' + 'f' * 64
    elif change == 'phase':
        events[-1]['body']['phase'] = 'restore_final'
    elif change == 'action':
        events[-1]['action_id'] = 'b' * 32
    elif change == 'credit':
        events[-1]['body']['credited_removed_allocated_bytes'] = 1
    else:
        @contextmanager
        def expired(**kwargs):
            yield None, None, journal
            raise ValueError('original approval expired')
        worker.checkpoint = expired
    with pytest.raises(ValueError):
        _pending_continuation(worker, events)


def pending_projection():
    from blueprint_pipeline.control_plane_lane_historical_restore_prefix import _pending_continuation
    worker, events, _, _ = worker_and_events()
    return worker.action_id, events, _pending_continuation(worker, events)


@pytest.mark.parametrize('change', ['head', 'phase', 'action', 'intent', 'credit', 'access'])
def test_controller_does_not_start_a_new_unit_for_unmatched_pending_head(tmp_path, monkeypatch, change):
    from tests import historical_generation_native_acceptance as native
    action_id, events, receipt = pending_projection()
    if change == 'head':
        receipt['continuation_event_digest'] = 'sha256:' + 'f' * 64
    elif change == 'phase':
        events[-1]['body']['phase'] = 'member'
    elif change == 'action':
        receipt['action_id'] = 'b' * 32
    elif change == 'intent':
        receipt['original_intent_event_digest'] = 'sha256:' + 'f' * 64
    elif change == 'credit':
        receipt['restored_files'] = 1
    else:
        receipt['owner_access_reopened'] = True
    monkeypatch.setattr(native, '_capture_action_events', lambda *args: events)
    calls = []
    def unit():
        calls.append('terminal unit projection')
        return receipt
    with pytest.raises(AssertionError):
        native._continued_restore_units(unit, tmp_path, action_id, [])
    assert len(calls) == 1


def test_controller_cannot_repeat_the_same_reconciled_head(tmp_path, monkeypatch):
    from tests import historical_generation_native_acceptance as native
    action_id, events, receipt = pending_projection()
    monkeypatch.setattr(native, '_capture_action_events', lambda *args: events)
    calls, continuations = [], []
    def unit():
        calls.append('terminal unit projection')
        return receipt
    with pytest.raises(AssertionError):
        native._continued_restore_units(unit, tmp_path, action_id, continuations)
    assert len(calls) == 2 and len(continuations) == 1


def test_routine_continuation_never_refunds_original_unknown_cadence(tmp_path, monkeypatch):
    import json
    from tests import historical_generation_native_acceptance as native
    action_id, events, receipt = pending_projection()
    journal = tmp_path / action_id
    journal.mkdir()
    original = json.dumps(events[0]).encode()
    (journal / 'e-00000.json').write_bytes(original)
    monkeypatch.setattr(native, '_capture_action_events', lambda *args: events)
    calls, continuations, observations = [], [], []
    unknown = dict(status='failed', code='historical_generation_process_unknown')
    def unit():
        calls.append('terminal unit projection')
        return receipt if len(calls) == 1 else unknown
    result = native._later_reference_attempts(lambda: native._continued_restore_units(
        unit, tmp_path, action_id, continuations), tmp_path, action_id, observations=observations)
    assert result == unknown and len(calls) == 4
    assert len(continuations) == 1 and [row['attempt'] for row in observations] == [1, 2, 3]
    assert (journal / 'e-00000.json').read_bytes() == original


def test_boundary_recovery_retains_durable_continuation_separate_from_unknowns():
    import json
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from tests import historical_generation_native_acceptance as native
    action_id, events, _ = pending_projection()
    events[0]['event_digest'] = canonical_digest(events[0], digest_field='event_digest')
    original = json.dumps(events[0]).encode()
    continuation = dict(action_id=action_id, original_intent_event_digest=events[0]['event_digest'],
                        continuation_event_digest=events[-1]['event_digest'])
    receipt = dict(status='completed', recovered_prefix=True, restored_files=0, restored_logical_bytes=0,
                   _fixture_prior_member_births=[], _fixture_unit_continuations=[continuation])
    native._assert_boundary_recovery(receipt, 'restarted_unwritten', [], action_id, original)
    assert '_fixture_unit_continuations' not in native._durable_receipt(receipt)
    continuation['original_intent_event_digest'] = 'sha256:' + 'f' * 64
    with pytest.raises(AssertionError):
        native._assert_boundary_recovery(receipt, 'restarted_unwritten', [], action_id, original)


@pytest.mark.parametrize('matched', [True, False])
def test_death_helper_continues_matching_pending_then_requires_genuine_death(tmp_path, monkeypatch, matched):
    from tests import historical_generation_native_acceptance as native
    action_id, events, receipt = pending_projection()
    if not matched:
        receipt['continuation_event_digest'] = 'sha256:' + 'f' * 64
    monkeypatch.setattr(native, '_capture_action_events', lambda *args: events)
    calls = []
    def unit(*args, **kwargs):
        assert kwargs == dict(restore=True, process_death=True)
        calls.append('terminal death-unit projection')
        # None represents the existing helper's independent actual SIGKILL
        # checks. This projection supplies no death or native proof itself.
        return receipt if len(calls) == 1 else None
    monkeypatch.setattr(native, '_launch_worker_once', unit)
    if matched:
        native._launch_death_worker(tmp_path / 'entry', action_id, tmp_path / 'target', tmp_path)
        assert len(calls) == 2
    else:
        with pytest.raises(AssertionError):
            native._launch_death_worker(tmp_path / 'entry', action_id, tmp_path / 'target', tmp_path)
        assert len(calls) == 1


@pytest.mark.parametrize('pending_count', [8, 9])
def test_routine_continuation_keeps_original_eight_reconciliation_limit(tmp_path, monkeypatch, pending_count):
    from tests import historical_generation_native_acceptance as native
    action_id, events, receipt = pending_projection()
    monkeypatch.setattr(native, '_capture_action_events', lambda *args: events)
    calls, continuations = [], []
    final = dict(status='completed', action_id=action_id)
    def unit():
        calls.append('terminal unit projection')
        if len(calls) > pending_count:
            return final
        digest = 'sha256:' + format(len(calls), '064x')
        events[-1]['event_digest'] = digest
        return dict(receipt, continuation_event_digest=digest)
    if pending_count == 8:
        assert native._continued_restore_units(unit, tmp_path, action_id, continuations) is final
    else:
        with pytest.raises(AssertionError):
            native._continued_restore_units(unit, tmp_path, action_id, continuations)
    assert len(calls) == 9 and len(continuations) == 8
