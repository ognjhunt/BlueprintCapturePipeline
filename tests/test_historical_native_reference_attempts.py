"""Fixture orchestration negatives: a refused unit is never terminal success.

This supplies no native observations or successful worker receipt. Actual
subsequent units and original clocks are proved by disposable Linux acceptance.
"""
# Covers: tests/historical_generation_native_acceptance.py
import json

import pytest

from tests import historical_generation_native_acceptance as native


ACTION = 'a' * 32
REFUSED = dict(status='kept', code='historical_generation_process_unknown')


def original_journal(tmp_path):
    path = tmp_path / ACTION
    path.mkdir()
    raw = json.dumps(dict(action_id=ACTION, kind='intent',
        body=dict(started_at_epoch=100, started_monotonic=20, boot_id='original'))).encode()
    (path / 'e-00000.json').write_bytes(raw)
    return path, raw


def test_repeated_unknowns_remain_refused_under_the_same_original_intent(tmp_path):
    path, raw = original_journal(tmp_path)
    calls = []
    def invoke():
        calls.append(ACTION)
        return dict(REFUSED)
    result = native._later_reference_attempts(invoke, tmp_path, ACTION)
    assert result == REFUSED and calls == [ACTION] * 3
    assert (path / 'e-00000.json').read_bytes() == raw


@pytest.mark.parametrize('code', ['historical_generation_restore_recovery_required',
    'historical_generation_restore_approval_invalid', 'historical_generation_dispatch_deadline'])
def test_non_reference_refusals_never_trigger_another_attempt(tmp_path, code):
    calls = []
    def invoke():
        calls.append(ACTION)
        return dict(status='kept', code=code)
    assert native._later_reference_attempts(invoke, tmp_path, ACTION)['code'] == code
    assert calls == [ACTION]


@pytest.mark.parametrize('change', ['missing', 'wrong_id', 'rewrite'])
def test_missing_or_changed_original_intent_never_triggers_a_second_attempt(tmp_path, change):
    path, _ = original_journal(tmp_path)
    if change == 'missing':
        (path / 'e-00000.json').unlink()
    elif change == 'wrong_id':
        (path / 'e-00000.json').write_text(json.dumps(dict(action_id='b' * 32, kind='intent')))
    calls = []
    def invoke():
        calls.append(ACTION)
        if change == 'rewrite':
            (path / 'e-00000.json').write_bytes(b'changed original clock')
        return dict(REFUSED)
    with pytest.raises((AssertionError, ValueError)):
        native._later_reference_attempts(invoke, tmp_path, ACTION)
    assert calls == [ACTION]


@pytest.mark.parametrize('phase', [None, 'recovered_publication', 'recovered_before_final',
                                  'recovered_access', 'restarted_unwritten', 'recovered_stage'])
def test_restore_increment_is_bound_to_actual_recovery_phase(phase):
    original = {'one.log': b'a', 'nested/two.log': b'bc'}
    receipt = dict(restored_files=2, restored_logical_bytes=3)
    if phase:
        receipt[phase] = True
    if phase and phase != 'restarted_unwritten':
        receipt.update(restored_files=0, restored_logical_bytes=0)
    native._assert_restore_increment(receipt, original)


@pytest.mark.parametrize('receipt', [dict(restored_files=0, restored_logical_bytes=0),
    dict(recovered_publication=True, restored_files=2, restored_logical_bytes=3),
    dict(recovered_access=True, recovered_before_final=True, restored_files=0, restored_logical_bytes=0),
    dict(recovered_publication='true', restored_files=0, restored_logical_bytes=0)])
def test_restore_increment_never_counts_recovered_bytes_twice_or_adopts_unproved_zero(receipt):
    with pytest.raises(AssertionError):
        native._assert_restore_increment(receipt, {'one.log': b'a', 'nested/two.log': b'bc'})
