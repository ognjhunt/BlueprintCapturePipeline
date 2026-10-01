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
                                  'recovered_access', 'restarted_unwritten', 'recovered_stage',
                                  'recovered_split', 'idempotent'])
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


@pytest.mark.parametrize('reused_files,reused_bytes', [(0, 0), (1, 2), (2, 3)])
def test_partial_prefix_counts_only_missing_bytes(reused_files, reused_bytes):
    native._assert_restore_increment(dict(recovered_prefix=True, reused_files=reused_files,
        reused_logical_bytes=reused_bytes, restored_files=2-reused_files,
        restored_logical_bytes=3-reused_bytes), {'one.log': b'a', 'nested/two.log': b'bc'})


@pytest.mark.parametrize('change', ['double_count', 'negative', 'bool', 'unknown_subset', 'missing'])
def test_prefix_accounting_refuses_unproven_or_repeated_credit(change):
    receipt = dict(recovered_prefix=True, reused_files=1, reused_logical_bytes=2,
                   restored_files=1, restored_logical_bytes=1)
    if change == 'double_count':
        receipt['restored_logical_bytes'] = 3
    elif change == 'negative':
        receipt['reused_files'] = -1
    elif change == 'bool':
        receipt['reused_files'] = True
    elif change == 'unknown_subset':
        receipt.update(reused_files=2, reused_logical_bytes=2, restored_files=0)
    else:
        receipt.pop('reused_files')
    with pytest.raises(AssertionError):
        native._assert_restore_increment(receipt, {'one.log': b'a', 'nested/two.log': b'bc'})


def test_later_recovery_phase_needs_retained_actual_same_intent_refusal(tmp_path):
    import hashlib
    path, raw = original_journal(tmp_path)
    receipts = [dict(REFUSED), dict(status='completed', recovered_access=True,
                                 restored_files=0, restored_logical_bytes=0)]
    observations = []
    result = native._later_reference_attempts(lambda: receipts.pop(0), tmp_path, ACTION,
                                              observations=observations)
    assert observations == [dict(action_id=ACTION, code=REFUSED['code'],
        original_intent_sha256=hashlib.sha256(raw).hexdigest(), attempt=1)]
    native._assert_boundary_recovery(result, 'recovered_split', observations, ACTION, raw)
    assert (path / 'e-00000.json').read_bytes() == raw


@pytest.mark.parametrize('change', ['none', 'wrong_id', 'wrong_intent', 'wrong_code', 'credit', 'flag'])
def test_later_phase_cannot_replace_a_missing_boundary_proof(tmp_path, change):
    import hashlib
    _, raw = original_journal(tmp_path)
    result = dict(status='completed', recovered_access=True, restored_files=0, restored_logical_bytes=0)
    observations = [dict(action_id=ACTION, code=REFUSED['code'],
        original_intent_sha256=hashlib.sha256(raw).hexdigest(), attempt=1)]
    if change == 'none':
        observations.clear()
    elif change == 'wrong_id':
        observations[0]['action_id'] = 'b' * 32
    elif change == 'wrong_intent':
        observations[0]['original_intent_sha256'] = 'f' * 64
    elif change == 'wrong_code':
        observations[0]['code'] = 'restore_recovery_required'
    elif change == 'credit':
        result['restored_files'] = 1
    else:
        result['recovered_access'] = 'true'
    with pytest.raises(AssertionError):
        native._assert_boundary_recovery(result, 'recovered_split', observations, ACTION, raw)


def test_controller_refusals_are_separate_from_actual_durable_receipt():
    receipt = dict(status='completed', action='delete', removed_files=2)
    decorated = dict(receipt, _fixture_reference_refusals=[dict(REFUSED)])
    assert native._durable_receipt(decorated) == receipt
    assert decorated['_fixture_reference_refusals'] == [REFUSED]
    assert native._durable_receipt(dict(decorated, other='unexpected')) != receipt


@pytest.mark.parametrize('expected', ['recovered_stage', 'recovered_split', 'recovered_prefix'])
def test_actual_later_restore_observation_may_follow_refused_boundary(tmp_path, expected):
    import hashlib
    _, raw = original_journal(tmp_path)
    result = dict(status='completed', idempotent=True, restored_files=0, restored_logical_bytes=0)
    observations = [dict(action_id=ACTION, code=REFUSED['code'],
        original_intent_sha256=hashlib.sha256(raw).hexdigest(), attempt=1)]
    native._assert_boundary_recovery(result, expected, observations, ACTION, raw)


@pytest.mark.parametrize('phase', ['restarted_unwritten', 'recovered_prefix', 'recovered_stage'])
def test_stage_boundary_cannot_move_backwards_after_refusal(tmp_path, phase):
    import hashlib
    _, raw = original_journal(tmp_path)
    result = dict(status='completed', **{phase: True}, restored_files=0, restored_logical_bytes=0)
    observations = [dict(action_id=ACTION, code=REFUSED['code'],
        original_intent_sha256=hashlib.sha256(raw).hexdigest(), attempt=1)]
    with pytest.raises(AssertionError):
        native._assert_boundary_recovery(result, 'recovered_split', observations, ACTION, raw)


def test_exact_prefix_phase_keeps_actual_incremental_accounting():
    result = dict(status='completed', recovered_prefix=True, restored_files=1, restored_logical_bytes=2,
                  reused_files=1, reused_logical_bytes=1)
    native._assert_boundary_recovery(result, 'recovered_prefix', [], ACTION, b'')
    native._assert_restore_increment(result, {'one': b'a', 'two': b'bc'})


@pytest.mark.parametrize('change', ['swapped_count', 'wrong_hash', 'wrong_path', 'duplicate'])
def test_native_prefix_credit_requires_the_exact_preceding_birth_set(change):
    import hashlib
    original = {'one': b'a', 'two': b'bc'}
    body = dict(path='one', sha256='sha256:' + hashlib.sha256(b'a').hexdigest(), size_bytes=1)
    receipt = dict(status='completed', recovered_prefix=True, reused_files=1, reused_logical_bytes=1,
        restored_files=1, restored_logical_bytes=2, _fixture_prior_member_births=[dict(body=body)])
    if change == 'swapped_count':
        receipt.update(reused_logical_bytes=2, restored_logical_bytes=1)
    elif change == 'wrong_hash':
        body['sha256'] = 'sha256:' + 'f' * 64
    elif change == 'wrong_path':
        body['path'] = 'unselected'
    else:
        receipt['_fixture_prior_member_births'].append(dict(body=body))
        receipt.update(reused_files=2, reused_logical_bytes=2, restored_files=0, restored_logical_bytes=1)
    with pytest.raises(AssertionError):
        native._assert_restore_increment(receipt, original)
