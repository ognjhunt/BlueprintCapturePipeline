"""Failure evidence preserves the real refusal; these are observer unit tests."""
import errno
import json

import pytest

from tests import registered_disk_diagnostic_native_acceptance as native


def test_suppressed_context_retains_typed_cause_without_private_content():
    try:
        try:
            raise OSError(errno.ENOENT, 'private contents', '/private/owner/secret')
        except OSError:
            raise ValueError('experiment_diagnostic_references_unknown') from None
    except ValueError as error:
        value = native.failure_evidence(error)
    raw = json.dumps(value)
    assert 'private contents' not in raw and '/private/owner/secret' not in raw
    assert [row['class'] for row in value['exceptions']] == ['ValueError', 'FileNotFoundError']
    assert value['exceptions'][0]['code'] == 'experiment_diagnostic_references_unknown'
    assert value['exceptions'][1]['errno'] == errno.ENOENT
    assert value['observation_phase'] == 'post_unwind'


def test_exception_cycles_and_large_chains_are_bounded():
    error = ValueError('secret-not-a-code')
    cursor = error
    for _ in range(20):
        cursor.__context__ = ValueError('arbitrary private message')
        cursor = cursor.__context__
    cursor.__context__ = error
    value = native.failure_evidence(error)
    assert len(value['exceptions']) == 8 and value['truncated']
    assert len(json.dumps(value).encode()) <= 8192
    assert 'secret-not-a-code' not in json.dumps(value)


def test_traceback_frames_are_bounded_without_locals():
    def recurse(depth):
        private_input = 'owner-private-payload'
        if depth:
            return recurse(depth - 1)
        raise ValueError(private_input)
    try:
        recurse(40)
    except ValueError as error:
        value = native.failure_evidence(error)
    assert len(value['exceptions'][0]['frames']) == 16 and value['truncated']
    assert 'owner-private-payload' not in json.dumps(value)


def test_actual_cycle_terminates_and_large_frame_chains_fit_byte_limit():
    first, second = ValueError('private-one'), ValueError('private-two')
    first.__context__, second.__context__ = second, first
    cyclic = native.failure_evidence(first)
    assert len(cyclic['exceptions']) == 2 and cyclic['truncated']
    def recurse(depth):
        if depth:
            return recurse(depth - 1)
        raise ValueError('private bytes')
    errors = []
    for _ in range(8):
        try:
            recurse(40)
        except ValueError as error:
            errors.append(error)
    for first, second in zip(errors, errors[1:]):
        first.__context__ = second
    evidence = native.failure_evidence(errors[0])
    assert len(evidence['exceptions']) == 8 and evidence['truncated']
    assert len(json.dumps(evidence).encode()) <= 8192
    assert all(error.__traceback__ is not None for error in errors)
    assert 'private bytes' not in json.dumps(evidence)


def test_success_is_unchanged_and_emits_nothing(monkeypatch, capsys):
    expected = object()
    monkeypatch.setattr(native, 'run', lambda root: expected)
    assert native.run_observed('fixture') is expected
    assert capsys.readouterr() == ('', '')


@pytest.mark.parametrize('stderr_fails', [False, True])
def test_original_exception_is_reraised_even_if_observer_output_fails(monkeypatch, capsys, stderr_fails):
    original = ValueError('experiment_diagnostic_references_unknown')
    def fail(root):
        raise original
    monkeypatch.setattr(native, 'run', fail)
    if stderr_fails:
        def broken_output(*args, **kwargs):
            raise OSError('observer output failed')
        monkeypatch.setattr(native, 'print', broken_output, raising=False)
    with pytest.raises(ValueError) as refused:
        native.run_observed('fixture')
    assert refused.value is original
    output = capsys.readouterr()
    if not stderr_fails:
        assert json.loads(output.err)['exceptions'][0]['code'] == str(original)
    assert not output.out
