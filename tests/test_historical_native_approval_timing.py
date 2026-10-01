# Covers (for impacted-test selection):
#   tests/historical_generation_native_acceptance.py
#   src/blueprint_pipeline/control_plane_lane_historical_restore_reconciliation_authority.py
#   src/blueprint_pipeline/control_plane_lane_historical_restore_absent_authority.py
"""Controller timing projections; no native observation or owner authority."""
import hashlib
import json

import pytest

from tests import historical_generation_native_acceptance as native


class _ApprovalObserved(Exception):
    def __init__(self, options):
        self.options = options


@pytest.mark.parametrize('kind', ['unlogged', 'fresh_discard', 'absence'])
def test_negative_validations_do_not_backdate_later_fixture_approval(tmp_path, monkeypatch, kind):
    from blueprint_pipeline import control_plane_lane_historical_restore_reconciliation_authority as reconcile
    from blueprint_pipeline import control_plane_lane_historical_restore_absent_authority as absent
    from blueprint_pipeline import control_plane_lane_historical_generation as generation

    root = tmp_path
    target = root / 'work' / 'target'
    target.mkdir(parents=True)
    (target / 'one.log').write_bytes(b'original')
    action_id, old_id = 'a' * 32, 'b' * 32
    journals = root / 'journals'
    journal = journals / action_id
    journal.mkdir(parents=True)
    original = json.dumps({'kind': 'restore_directory', 'body': {'phase': 'original'}}).encode()
    (journal / 'e-00000.json').write_bytes(original)
    store = root / 'state/requests/historical-generation-actions'
    store.mkdir(parents=True)
    old_raw = b'earlier decision projection'
    (store / (old_id + '.json')).write_bytes(old_raw)
    def selector(raw):
        return {'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
    packet = {'packet_digest': 'sha256:' + 'c' * 64,
              'original_intent_bytes': selector(original), 'execution_authorized': False,
              'original_expires_at_epoch': 1000,
              'scope': {'remove_member': {'path': 'one.log'}}, 'resume_from': None,
              'attempt': 1, 'prior_decisions': [{'decision_id': old_id, 'decision': selector(old_raw)}],
              'permits_removal': False, 'observation_only': True}
    restore = {'action_id': action_id, 'expires_at_epoch': 1000}
    old = {'decision_id': old_id, 'expires_at_epoch': 99, 'attempt': 0}
    clock = [100]
    monkeypatch.setattr(native.time, 'time', lambda: clock[0])
    monkeypatch.setattr(reconcile, 'observe_historical_restore_reconciliation', lambda **_: packet)
    monkeypatch.setattr(absent, 'observe_historical_restore_absence', lambda **_: packet)
    monkeypatch.setattr(generation, 'inventory_historical_generation', lambda *a, **k: {'projected': True})
    code = {'unlogged': 'historical_generation_restore_reconciliation_approval_missing',
            'fresh_discard': 'historical_generation_restore_reconciliation_approval_invalid',
            'absence': 'historical_generation_restore_absent_observation_approval_missing'}[kind]
    monkeypatch.setattr(native, '_launch_worker', lambda *a, **k: {'status': 'failed', 'code': code})
    calls = []

    def approve(**options):
        # Each negative validation consumes real controller time in the native
        # case. The projection advances that time, without granting anything.
        assert options['now'] == clock[0], 'approval_observation_was_backdated'
        calls.append(options)
        valid = (options['principal'] == ('restore-operator' if kind == 'absence' else 'operator')
                 and options['owner'] == 'owner' and options['ack_packet_digest'] == packet['packet_digest']
                 and options['no_future_writers'] is options['no_future_readers'] is True
                 and options.get('observe_absence_only', options.get('discard_unfinished_row')) is True
                 and options['expires_at_epoch'] <= restore['expires_at_epoch'])
        if valid:
            raise _ApprovalObserved(options)
        clock[0] += 3
        raise ValueError('projected negative refusal')

    monkeypatch.setattr(reconcile, 'approve_historical_restore_reconciliation', approve)
    monkeypatch.setattr(absent, 'approve_historical_restore_absence', approve)
    with pytest.raises(_ApprovalObserved) as captured:
        if kind == 'unlogged':
            native._approve_unlogged_fixture(root, root / 'config', root / 'entry', restore,
                                             target, journals, short_expiry=True)
        elif kind == 'fresh_discard':
            native._approve_fresh_discard_fixture(root, root / 'config', root / 'entry', restore,
                                                  target, journals, old, short_expiry=True)
        else:
            native._approve_absent_fixture(root, root / 'config', root / 'entry', restore,
                                           target, journals, old, repeat_expiry=True)
    assert len(calls) == (7 if kind == 'absence' else 6)
    actual = captured.value.options
    assert actual['expires_at_epoch'] - actual['now'] == (4 if kind == 'absence' else 12)
    assert (journal / 'e-00000.json').read_bytes() == original
    assert (store / (old_id + '.json')).read_bytes() == old_raw
