"""Failure evidence preserves the real refusal; these are observer unit tests."""
import errno
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize('gc_raises,packet_fails', [(False, False), (True, False), (True, True)])
def test_actual_gc_child_entry_retains_pid_open_observation_in_native_result(
    tmp_path, monkeypatch, gc_raises, packet_fails,
):
    """Hermetic child-hook regression, not Linux syscall or clearance proof."""
    from tests import test_registered_feature_linux as sandbox
    from blueprint_pipeline import control_plane_lane_historical_processes as processes
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    from blueprint_pipeline import control_plane_lane_experiment_archive as archive
    from blueprint_pipeline import control_plane_storage_gc as gc
    from blueprint_pipeline.control_plane_lane_owner_target_io import _TargetFiles
    from blueprint_pipeline.control_plane_lane_disk_diagnostic_references import (
        DiagnosticReferences, _ReferenceFiles,
    )
    # The existing archive-service fixture imports unrelated model tests.
    # This hook-only regression needs its empty object-service state, no model.
    monkeypatch.setitem(sys.modules, 'tests.test_registered_experiment_offload',
        SimpleNamespace(Cloud=lambda: SimpleNamespace(objects={}, bodies=[], metadata={})))

    # Save attributes that the real child entry decorates, so this unit fixture
    # cannot leave observers or installation paths active for another test.
    for target, names in ((owners, ('INSTALLED_PACKAGE_ROOT',)), (archive, ('_client',)),
                          (_TargetFiles, ('slot', 'read_bytes')), (_ReferenceFiles, ('read_bytes',)),
                          (DiagnosticReferences, ('__init__', 'guard'))):
        for name in names:
            monkeypatch.setattr(target, name, getattr(target, name))
    logical_root = Path('/var/lib/blueprint-adp-contained-child-hook')
    root = tmp_path / logical_root.relative_to('/')
    selected = root / 'work/lanes/diagnostics/action/selected.json'
    selected.parent.mkdir(parents=True)
    selected.write_text(json.dumps({'config': str(root / 'door.json'), 'pins': str(root / 'pins'),
        'now': 1.0, 'action_id': 'a' * 32, 'realtime': True}))
    (root / 'door.json').write_text(json.dumps({
        'lane_scratch_work_root': str(root / 'work/lanes'),
        'lane_scratch_inputs_root': str(root / 'inputs/lanes')}))
    status = tmp_path / 'status'
    status.write_text('CapEff: 0000000000080003\nNoNewPrivs: 1\n')
    def fixture_path(value):
        path = Path(value)
        if path == Path('/proc/self/status'):
            return status
        if path == Path('/var/lib') or logical_root == path or logical_root in path.parents:
            return tmp_path / path.relative_to('/')
        return path
    monkeypatch.setattr(sandbox, 'Path', fixture_path)
    monkeypatch.setattr(os, 'geteuid', lambda: 0)
    original_open = os.open
    def opened(name, flags, *args, **kwargs):
        if name == '4242' and kwargs.get('dir_fd') == 77:
            raise FileNotFoundError(errno.ENOENT, 'private PID-open error')
        if name == root / 'outside-gc-rw-probe':
            raise OSError(errno.EROFS, 'unit fixture write fence')
        return original_open(name, flags, *args, **kwargs)
    monkeypatch.setattr(os, 'open', opened)
    def scanner():
        pid, proc, _attempt = '4242', 77, 1
        scan = SimpleNamespace(started=1.0, last=1.1)
        # These locals deliberately match the scanner's observer seam.
        assert scan and _attempt == 1
        os.open(pid, os.O_RDONLY, dir_fd=proc)
    monkeypatch.setattr(processes, 'refuse_historical_process_references', scanner)
    original_error = ValueError('experiment_diagnostic_references_unknown')
    def refused_gc(**kwargs):
        assert kwargs['apply'] is True and kwargs['lane_scratch_enabled'] is True
        try:
            scanner()
        except FileNotFoundError:
            if gc_raises:
                raise original_error
            return {'registered_experiments': {'outcomes': [{
                'action_id': 'a' * 32, 'decision': 'kept', 'removed_logical_bytes': 0,
                'removed_allocated_bytes': 0}]}}
        raise AssertionError('fixture syscall unexpectedly succeeded')
    monkeypatch.setattr(gc, 'run_storage_gc', refused_gc)
    if packet_fails:
        def broken_packet(self):
            raise OSError('unit fixture projection failure')
        monkeypatch.setattr(native.PidOpenEvidence, 'packet', broken_packet)
    if gc_raises:
        with pytest.raises(ValueError) as refused:
            sandbox._gc_sandbox_main(str(logical_root / selected.relative_to(root)))
        assert refused.value is original_error and os.open is opened
        assert not (selected.parent / 'native-result.json').exists()
        return
    sandbox._gc_sandbox_main(str(logical_root / selected.relative_to(root)))
    result = json.loads((selected.parent / 'native-result.json').read_bytes())
    observed = result['native_reference_failures']['pid_open_evidence']
    assert observed['observations'] == [{'pid_token': 1, 'attempt': 2, 'elapsed_ms': 100.0,
        'ownership': 'UNKNOWN', 'child_role': None, 'previous_fixture_child_reaped': False}]
    assert observed['pid_tokens_are_numeric_slots_not_start_identities'] is True
    assert observed['truncated'] is False
    assert result['report']['registered_experiments']['outcomes'][0]['decision'] == 'kept'
    assert '4242' not in json.dumps(result) and 'private PID-open' not in json.dumps(result)
