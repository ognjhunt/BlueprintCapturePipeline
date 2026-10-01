"""ADP-009D/day28: resumed history must reach the full private-prefix gate.

Dispatch boundary only. The child refuses absent actual native/owner proof;
these phase projections never claim an authorized effect or completion.
"""
import pytest


@pytest.mark.parametrize('consumed', [False, True])
def test_resumed_delete_history_reaches_full_prefix_authorization(monkeypatch, consumed):
    from blueprint_pipeline.control_plane_lane_historical_restore_worker import _recover_before_final
    from blueprint_pipeline import control_plane_lane_historical_restore_prefix as prefix
    worker, roots, clock = object(), ('bounded-root',), lambda: 0
    events = [dict(kind='intent', body={})]
    for phase in ('reservation', 'directory', 'member', 'reconcile_intent', 'reconcile_delete_resume'):
        events.append(dict(kind='restore_intent', body=dict(phase=phase)))
    if consumed:
        events.append(dict(kind='restore_intent', body=dict(phase='reconciled')))
    def full_gate(observed_worker, observed_events, observed_roots, observed_clock):
        assert observed_worker is worker and observed_events is events
        assert observed_roots is roots and observed_clock is clock
        raise ValueError('full_prefix_native_and_owner_proof_missing')
    monkeypatch.setattr(prefix, 'recover_private_prefix', full_gate)
    with pytest.raises(ValueError, match='full_prefix_native_and_owner_proof_missing'):
        _recover_before_final(worker, events, roots, clock)


def test_unknown_resume_phase_never_bypasses_prefix_authentication(monkeypatch):
    from blueprint_pipeline.control_plane_lane_historical_restore_worker import _recover_before_final
    from blueprint_pipeline import control_plane_lane_historical_restore_prefix as prefix
    monkeypatch.setattr(prefix, 'recover_private_prefix', lambda *args: pytest.fail('unknown phase delegated'))
    events = [dict(kind='intent', body={}),
              dict(kind='restore_intent', body=dict(phase='unrecognized_delete_resume'))]
    with pytest.raises(ValueError, match='restore_recovery_required'):
        _recover_before_final(object(), events, (), lambda: 0)
