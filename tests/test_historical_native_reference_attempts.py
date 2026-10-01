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


def test_same_unit_journal_observation_uses_actual_prelaunch_cursor(monkeypatch):
    from types import SimpleNamespace
    calls = []
    cursor = 's=abc;i=123;b=def;m=456;t=789;x=abc'
    def run(argv, **kwargs):
        calls.append(argv)
        if '--lines=1' in argv:
            return SimpleNamespace(returncode=0, stdout=json.dumps(dict(__CURSOR=cursor)))
        assert '--after-cursor=' + cursor in argv
        return SimpleNamespace(returncode=0, stdout=json.dumps(dict(MESSAGE=json.dumps(dict(status='failed', code='actual_refusal')))))
    monkeypatch.setattr(native.subprocess, 'run', run)
    observed = native._unit_cursor('actual-unit')
    records = [json.loads(line) for line in native._unit_output('actual-unit', observed).splitlines()]
    assert observed == cursor and records == [dict(status='failed', code='actual_refusal')]
    assert all('--unit=actual-unit' in argv for argv in calls)


@pytest.mark.parametrize('value', [None, '', 'invented cursor', 's=bad\ncommand'])
def test_unknown_journald_cursor_cannot_select_a_receipt_window(monkeypatch, value):
    from types import SimpleNamespace
    monkeypatch.setattr(native.subprocess, 'run', lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout=json.dumps(dict(__CURSOR=value))))
    with pytest.raises(AssertionError):
        native._unit_cursor('actual-unit')


def test_journal_delta_refuses_overflow_instead_of_losing_new_receipts(monkeypatch):
    from types import SimpleNamespace
    def run(argv, **kwargs):
        assert '--lines=65' in argv
        return SimpleNamespace(returncode=0, stdout='\n'.join(
            json.dumps(dict(MESSAGE='manager line')) for _ in range(65)))
    monkeypatch.setattr(native.subprocess, 'run', run)
    with pytest.raises(AssertionError, match='journal_delta_overflow'):
        native._unit_output('actual-unit', 's=abc')


def test_journal_delta_requests_complete_long_messages_with_existing_byte_bound(monkeypatch):
    from types import SimpleNamespace
    message = json.dumps(dict(status='completed', evidence='a' * 5000))
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        # journalctl's documented JSON default emits null above4096 bytes.
        value = message if '--all' in argv else None
        return SimpleNamespace(returncode=0, stdout=json.dumps(dict(MESSAGE=value)))
    monkeypatch.setattr(native.subprocess, 'run', run)
    assert native._unit_output('actual-unit', 's=abc') == message
    assert '--output-fields=MESSAGE' in calls[0]


def test_reconciliation_fault_observes_held_pin_without_a_new_journal_event():
    from types import SimpleNamespace
    scope = dict(remove_member=dict(path='actual pending row'))
    worker = SimpleNamespace(effect_selected=[(dict(packet=dict(scope=scope)), None, None)])
    calls = []
    binding = dict(decision_id='actual selected ID')
    def pin(current, selected):
        assert current is worker and selected is binding
        calls.append(selected)
    assert native._observe_reconciliation_pin(worker, binding, pin) is scope
    assert calls == [binding]
    def refused(*args):
        raise ValueError('real grant refused')
    with pytest.raises(ValueError, match='real grant refused'):
        native._observe_reconciliation_pin(worker, binding, refused)


@pytest.mark.parametrize('new_record', [None, dict(status='failed', code='current_real_refusal'),
    dict(unknown_json='must refuse'), dict(status='failed', code='historical_generation_process_unknown'),
    dict(status='failed', code='historical_generation_restore_reconciliation_approval_missing')])
def test_old_receipt_eviction_cannot_make_a_current_killed_unit_emit_a_receipt(tmp_path, monkeypatch, new_record):
    # Controller-flow projection only: no native worker, death or cgroup proof.
    # The old receipt falls out of the last64 lines after new manager messages.
    import os
    from types import SimpleNamespace
    from blueprint_pipeline import control_plane_lane_historical_dispatch as dispatch
    monkeypatch.setattr(dispatch, '_unit_property_assignments', lambda *args, **kwargs: ())
    entry = tmp_path / 'entry'
    entry.write_bytes(b'')
    group = tmp_path / 'simulated-cgroup-events'
    group.write_bytes(b'populated 1\n')
    opened, reads, journal_reads, startup_order = [], [], [], []
    real_open, real_pread = os.open, os.pread
    real_write = os.write
    def open_file(path, *args, **kwargs):
        if str(path).startswith('/sys/fs/cgroup/'):
            fd = real_open(group, *args, **kwargs)
            opened.append(fd)
            startup_order.append('held-cgroup')
            return fd
        return real_open(path, *args, **kwargs)
    def pread(fd, *args):
        if fd in opened:
            reads.append(fd)
            return b'populated 1\n' if len(reads) == 1 else b'populated 0\n'
        return real_pread(fd, *args)
    def run(argv, **kwargs):
        if argv[0].endswith('systemd-run'):
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        assert argv[0].endswith('journalctl')
        if '--sync' in argv:
            assert kwargs['timeout'] == 5
            startup_order.append('reaped-sync')
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        if '--lines=1' in argv:
            return SimpleNamespace(returncode=0, stdout=json.dumps(dict(__CURSOR='s=abc;i=123')))
        journal_reads.append(argv)
        # Earlier genuine fixture transport receipts can exceed one query's
        # bound in aggregate. They cannot enter this attempt's cursor window.
        uncompleted = new_record is not None and new_record.get('code') in (
            'historical_generation_process_unknown', 'historical_generation_restore_reconciliation_approval_missing')
        text = (json.dumps(dict(status='failed', code='prior_real_refusal',
                                transport='a' * 40000))
                if '--after-cursor=s=abc;i=123' not in argv else
                'code=exited, status=1/FAILURE\n' if uncompleted else 'code=killed, status=9/KILL\n')
        if '--after-cursor=s=abc;i=123' in argv and new_record is not None:
            text += json.dumps(new_record) + '\n'
        return SimpleNamespace(returncode=0, stdout='\n'.join(json.dumps(dict(MESSAGE=line)) for line in text.splitlines()))
    monkeypatch.setattr(native.os, 'open', open_file)
    def write(fd, raw):
        if raw == b'ready\n':
            startup_order.append('release')
        return real_write(fd, raw)
    monkeypatch.setattr(native.os, 'write', write)
    monkeypatch.setattr(native.os, 'pread', pread)
    monkeypatch.setattr(native.subprocess, 'run', run)
    if new_record is None:
        assert native._launch_worker_once(entry, ACTION, tmp_path / 'target', tmp_path / 'journals',
            process_death=True) is None
    elif new_record.get('code') in ('historical_generation_process_unknown',
                                   'historical_generation_restore_reconciliation_approval_missing'):
        # Actual typed refusal can reach exact owner handling, but it can
        # never be converted to the None reserved for observed SIGKILL.
        assert native._launch_worker_once(entry, ACTION, tmp_path / 'target', tmp_path / 'journals',
            process_death=True) == new_record
    else:
        with pytest.raises(AssertionError, match='killed worker must not emit a terminal receipt'):
            native._launch_worker_once(entry, ACTION, tmp_path / 'target', tmp_path / 'journals', process_death=True)
    assert len(reads) == 2
    assert all('--after-cursor=s=abc;i=123' in argv for argv in journal_reads)
    assert startup_order == ['held-cgroup', 'reaped-sync', 'release']


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


@pytest.mark.parametrize('refusals', [0, 1, 2])
def test_observed_death_ends_bounded_same_intent_cadence_without_a_success_receipt(tmp_path, refusals):
    path, raw = original_journal(tmp_path)
    calls = []
    def invoke():
        calls.append(ACTION)
        return dict(REFUSED) if len(calls) <= refusals else None
    assert native._later_reference_attempts(invoke, tmp_path, ACTION) is None
    assert len(calls) == refusals + 1
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


def test_exact_unfinished_owner_handling_uses_existing_three_unit_cadence(tmp_path):
    path, raw = original_journal(tmp_path)
    missing = dict(status='failed', code='historical_generation_restore_reconciliation_approval_missing')
    results = iter([dict(REFUSED), missing, None])
    calls, approvals = [], []
    def invoke():
        calls.append(ACTION)
        return next(results)
    def handle(receipt):
        # Controller projection only: the actual fixture callback must use
        # the protected exact-head owner API; this creates no approval.
        assert receipt is missing
        approvals.append(receipt)
        assert (path / 'e-00000.json').read_bytes() == raw
    assert native._later_reference_attempts(invoke, tmp_path, ACTION,
        on_unfinished=handle) is None
    assert len(calls) == 3 and approvals == [missing]
    assert (path / 'e-00000.json').read_bytes() == raw


def test_last_failed_unit_cannot_issue_an_unused_recovery_grant(tmp_path):
    original_journal(tmp_path)
    missing = dict(status='failed', code='historical_generation_restore_reconciliation_approval_missing')
    calls, approvals = [], []
    def invoke():
        calls.append(ACTION)
        return dict(REFUSED) if len(calls) < 3 else missing
    assert native._later_reference_attempts(invoke, tmp_path, ACTION,
        on_unfinished=approvals.append) is missing
    assert len(calls) == 3 and approvals == []


@pytest.mark.parametrize('last_code', ['historical_generation_restore_reconciliation_approval_missing',
                                      'historical_generation_process_unknown', 'arbitrary_failure'])
def test_one_extra_unit_requires_a_real_missing_decision_receipt(tmp_path, last_code):
    path, raw = original_journal(tmp_path)
    calls, approvals = [], []
    last = dict(status='failed', code=last_code)
    def invoke():
        calls.append(ACTION)
        return dict(REFUSED) if len(calls) < 3 else last if len(calls) == 3 else None
    def approve(receipt):
        approvals.append(receipt)
        return dict(decision_id='b' * 32, discard_unfinished_row_approved=True,
                    execution_authorized=False, packet=dict(action_id=ACTION,
                    original_intent_bytes=dict(sha256='sha256:' + __import__('hashlib').sha256(raw).hexdigest(),
                                               size_bytes=len(raw))))
    result = native._reconciled_reference_attempts(invoke, tmp_path, ACTION, on_unfinished=approve)
    assert len(calls) == (4 if last_code.endswith('approval_missing') else 3)
    assert approvals == ([last] if last_code.endswith('approval_missing') else [])
    assert result is (None if approvals else last)
    assert (path / 'e-00000.json').read_bytes() == raw


def test_extra_owner_unit_preserves_all_original_journal_bytes(tmp_path):
    path, raw = original_journal(tmp_path)
    calls = []
    def invoke():
        calls.append(ACTION)
        return dict(REFUSED) if len(calls) < 3 else dict(status='failed',
            code='historical_generation_restore_reconciliation_approval_missing')
    def approve(receipt):
        (path / 'e-00000.json').write_bytes(b'replaced original intent')
    with pytest.raises(AssertionError, match='original_journal_changed'):
        native._reconciled_reference_attempts(invoke, tmp_path, ACTION, on_unfinished=approve)
    assert len(calls) == 3 and (path / 'e-00000.json').read_bytes() != raw


def test_owner_handling_cannot_rewrite_the_original_journal(tmp_path):
    path, _ = original_journal(tmp_path)
    missing = dict(status='failed', code='historical_generation_restore_reconciliation_approval_missing')
    def handle(receipt):
        (path / 'e-00000.json').write_bytes(b'replaced operation')
    with pytest.raises(AssertionError, match='original_journal_changed'):
        native._later_reference_attempts(lambda: missing, tmp_path, ACTION, on_unfinished=handle)


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


@pytest.mark.parametrize('phase', ['recovered_access', 'recovered_prefix'])
def test_unwritten_restart_can_advance_before_a_later_real_reference_refusal(tmp_path, phase):
    import hashlib
    _, raw = original_journal(tmp_path)
    result = dict(status='completed', **{phase: True}, restored_files=0, restored_logical_bytes=0)
    if phase == 'recovered_prefix':
        result.update(restored_files=1, restored_logical_bytes=2, reused_files=1,
                      reused_logical_bytes=1, _fixture_prior_member_births=[dict(body=dict(
                          path='one', size_bytes=1, sha256='sha256:' + hashlib.sha256(b'a').hexdigest()))])
    observations = [dict(action_id=ACTION, code=REFUSED['code'],
        original_intent_sha256=hashlib.sha256(raw).hexdigest(), attempt=1)]
    native._assert_boundary_recovery(result, 'restarted_unwritten', observations, ACTION, raw)
    native._assert_restore_increment(result, {'one': b'a', 'two': b'bc'})


def test_unwritten_later_prefix_cannot_claim_reuse_without_original_births(tmp_path):
    import hashlib
    _, raw = original_journal(tmp_path)
    result = dict(status='completed', recovered_prefix=True, restored_files=1,
                  restored_logical_bytes=2, reused_files=1, reused_logical_bytes=1)
    observations = [dict(action_id=ACTION, code=REFUSED['code'],
        original_intent_sha256=hashlib.sha256(raw).hexdigest(), attempt=1)]
    with pytest.raises(AssertionError):
        native._assert_boundary_recovery(result, 'restarted_unwritten', observations, ACTION, raw)


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
