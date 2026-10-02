"""Failure evidence preserves original refusals; no native clearance is supplied."""
# Covers: tests/registered_disk_diagnostic_native_acceptance.py
# Covers: tests/test_registered_feature_linux.py
import errno
import json

import pytest

from blueprint_pipeline import control_plane_lane_disk_diagnostic_references as references
from blueprint_pipeline import control_plane_lane_historical_processes as processes
from tests import registered_disk_diagnostic_native_acceptance as observer


def install(monkeypatch, cls):
    evidence = {'records': [], 'truncated': False}
    monkeypatch.setattr(cls, '__init__', cls.__init__)
    monkeypatch.setattr(cls, 'guard', cls.guard)
    originals = observer.observe_reference_failures(cls, evidence)
    assert cls.__init__.__wrapped__ is originals[0]
    assert cls.guard.__wrapped__ is originals[1]
    return evidence


def unknown():
    return references.OwnerTargetVersionError('experiment_diagnostic_references_unknown')


def test_observation_keeps_class_original_arguments_and_success(monkeypatch):
    calls, token = [], object()
    class Reference:
        def __init__(self, value, *, selected):
            calls.append(('constructor', self, value, selected))
        def guard(self, *, processes=True):
            calls.append(('guard', self, processes))
            return token
    evidence = install(monkeypatch, Reference)
    value = Reference(token, selected=token)
    assert type(value) is Reference
    assert value.guard(processes=False) is token
    assert calls == [('constructor', value, token, token), ('guard', value, False)]
    assert evidence == {'records': [], 'truncated': False}


def test_actual_unknown_is_same_exception_with_honest_duplicate_boundaries(monkeypatch):
    error, calls = unknown(), []
    class Reference:
        def __init__(self):
            calls.append('constructor')
            self.guard()
        def guard(self):
            calls.append('guard')
            raise error
    evidence = install(monkeypatch, Reference)
    with pytest.raises(references.OwnerTargetVersionError) as raised:
        Reference()
    assert raised.value is error and calls == ['constructor', 'guard']
    assert [row['boundary'] for row in evidence['records']] == ['guard', 'constructor']
    assert all(row['observation_phase'] == 'caught_unknown_before_gc_consumption'
               and row['final_process_identity_verified'] is False for row in evidence['records'])


@pytest.mark.parametrize('error', [OSError(errno.ENOENT, 'private input'),
    references.OwnerTargetVersionError('experiment_diagnostic_active_reference')])
def test_other_refusals_are_not_reclassified(monkeypatch, error):
    class Reference:
        def __init__(self):
            pass
        def guard(self):
            raise error
    evidence = install(monkeypatch, Reference)
    with pytest.raises(type(error)) as raised:
        Reference().guard()
    assert raised.value is error and not evidence['records']


def test_observer_failure_cannot_replace_original_unknown(monkeypatch):
    error = unknown()
    class Reference:
        def __init__(self):
            pass
        def guard(self):
            raise error
    evidence = install(monkeypatch, Reference)
    def failed_projection(*args):
        raise RuntimeError('private observer input')
    monkeypatch.setattr(observer, 'failure_evidence', failed_projection)
    with pytest.raises(references.OwnerTargetVersionError) as raised:
        Reference().guard()
    assert raised.value is error and not evidence['records']
    assert error.__context__ is None and error.__cause__ is None
    assert evidence['truncated'] is True


def test_real_inspector_frame_projects_only_already_read_initial_identity(monkeypatch):
    reads, private_comm = [], 'PRIVATE_PROCESS_CONTENT'
    raw = ('777 (' + private_comm + ') S 12 ' + '0 ' * 17 + '12345 0\n').encode()
    class Scan:
        def read(self, directory, name, cap):
            reads.append(name)
            if len(reads) == 1:
                return raw
            raise FileNotFoundError(errno.ENOENT, 'PRIVATE_INPUT_PATH')
    class Reference:
        def __init__(self):
            pass
        def guard(self):
            try:
                processes._inspect_process(Scan(), -1, '777', '/never-read', set(), (0, 0, 0), 0, (0, 0))
            except OSError:
                raise unknown() from None
    evidence = install(monkeypatch, Reference)
    with pytest.raises(references.OwnerTargetVersionError):
        Reference().guard()
    assert len(reads) == 2
    views = evidence['records'][0]['same_failed_views']
    assert views == [{'initial_process': {'pid': 777, 'parent_pid': 12, 'start_tick': 12345},
                      'final_process_identity_verified': False}]
    encoded = json.dumps(evidence)
    assert private_comm not in encoded and 'PRIVATE_INPUT_PATH' not in encoded
    assert '/never-read' not in encoded and 'fd_names' not in encoded


def test_same_named_foreign_frame_cannot_supply_process_identity(monkeypatch):
    def _inspect_process():
        initial_stat, pid, started, names, after_names = b'PRIVATE_STAT', '777', 12345, ['1'], ['2']
        assert all((initial_stat, pid, started, names, after_names))
        raise ValueError('PRIVATE_EXCEPTION_CONTENT')
    class Reference:
        def __init__(self):
            pass
        def guard(self):
            try:
                _inspect_process()
            except ValueError:
                raise unknown() from None
    evidence = install(monkeypatch, Reference)
    with pytest.raises(references.OwnerTargetVersionError):
        Reference().guard()
    assert evidence['records'][0]['same_failed_views'] == []
    assert 'PRIVATE_' not in json.dumps(evidence)


def test_event_and_serialized_bounds_do_not_limit_original_execution(monkeypatch):
    error, calls = unknown(), []
    class Reference:
        def __init__(self):
            pass
        def guard(self):
            calls.append(True)
            raise error
    evidence = install(monkeypatch, Reference)
    value = Reference()
    for _ in range(20):
        with pytest.raises(references.OwnerTargetVersionError) as raised:
            value.guard()
        assert raised.value is error
    assert len(calls) == 20 and len(evidence['records']) <= 4 and evidence['truncated']
    assert len(json.dumps(evidence).encode()) <= 16384
    evidence['records'].clear()
    monkeypatch.setattr(observer, 'failure_evidence', lambda error: {'large': 'x' * 20000})
    with pytest.raises(references.OwnerTargetVersionError) as raised:
        value.guard()
    assert raised.value is error and not evidence['records']
    assert len(json.dumps(evidence).encode()) <= 16384
