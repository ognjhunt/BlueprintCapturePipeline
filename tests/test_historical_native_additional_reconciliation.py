# Covers: tests/historical_generation_native_acceptance.py
"""Controller transcript projections, never native or owner proof."""
from contextlib import contextmanager
from copy import deepcopy

import pytest

from tests import historical_generation_native_acceptance as native


@pytest.mark.parametrize('change', [None, 'missing', 'duplicate', 'wrong_record',
    'unconsumed', 'expired', 'credited', 'uncertain', 'wrong_head', 'wrong_expiry'])
def test_every_additional_row_requires_its_exact_consumed_protected_grant(tmp_path, monkeypatch, change):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline import control_plane_lane_historical_restore_authority as restore_authority
    from blueprint_pipeline import control_plane_lane_historical_restore_reconciliation_authority as grants
    from blueprint_pipeline import control_plane_lane_historical_restore_reconciliation_replay as replay
    identifier, head, born = 'a' * 32, 'sha256:' + 'b' * 64, 'sha256:' + 'c' * 64
    approval = dict(decision_id=identifier, issued_at_epoch=10, expires_at_epoch=20,
        packet=dict(original_head_event_digest=head, original_expires_at_epoch=100))
    prefix = [dict(event_digest=head)]
    intent = dict(observed_at_epoch=11, event_digest=born)
    completed = dict(observed_at_epoch=12, previous_event_digest=born, event_digest='sha256:' + 'd' * 64,
        body=dict(phase='reconciled', uncertain=True, credited_removed_allocated_bytes=0))
    binding = (prefix, identifier, {'projected': 'exact raw selector'}, (intent, completed))
    stored = deepcopy(approval)
    if change == 'wrong_record':
        stored['packet']['original_head_event_digest'] = 'changed'
    elif change == 'unconsumed':
        completed['body']['phase'] = 'reconcile_intent'
    elif change == 'expired':
        completed['observed_at_epoch'] = approval['expires_at_epoch']
    elif change == 'credited':
        completed['body']['credited_removed_allocated_bytes'] = 1
    elif change == 'uncertain':
        completed['body']['uncertain'] = False
    elif change == 'wrong_head':
        prefix[0]['event_digest'] = 'different current head'
    elif change == 'wrong_expiry':
        approval['packet']['original_expires_at_epoch'] = 101
        stored = deepcopy(approval)
    calls = []
    class Store:
        def read(self, selected):
            assert selected == identifier
            return stored, b'projected protected raw'
    @contextmanager
    def session(*args):
        yield object(), object(), Store()
    monkeypatch.setattr(authority, '_session', session)
    selected = object()
    monkeypatch.setattr(restore_authority, 'select_restore', lambda *args: selected)
    def select_effect(files, config, store, config_path, current, actual_prefix, actual_id, selector, recorded, moment):
        assert current is selected and actual_prefix is prefix and actual_id == identifier
        assert recorded is binding[3] and selector is binding[2]
        calls.append(identifier)
        return ((stored,), (stored,))
    monkeypatch.setattr(grants, 'historical_effect_grant', select_effect)
    monkeypatch.setattr(replay, 'reconciliation_bindings', lambda _: [] if change == 'missing' else
        [binding, binding] if change == 'duplicate' else [binding])
    def verify():
        native._authenticate_additional_reconciliations(tmp_path, tmp_path / 'config',
            dict(action_id='e' * 32, expires_at_epoch=100), [], [approval])
    if change is None:
        verify()
        assert calls == [identifier]
    else:
        with pytest.raises(AssertionError):
            verify()
        assert len(calls) <= 1
