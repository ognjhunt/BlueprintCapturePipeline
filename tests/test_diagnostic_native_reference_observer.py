"""Failure evidence preserves original refusals; no native clearance is supplied."""
# Covers: tests/registered_disk_diagnostic_native_acceptance.py
# Covers: tests/test_registered_feature_linux.py
import errno
import json

import pytest

from blueprint_pipeline import control_plane_lane_disk_diagnostic_references as references
from blueprint_pipeline import control_plane_lane_historical_processes as processes
from tests import registered_disk_diagnostic_native_acceptance as observer
from tests import test_historical_process_census as census_helpers


@pytest.fixture
def census(tmp_path, monkeypatch):
    return census_helpers.census.__wrapped__(tmp_path, monkeypatch)


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


def test_exhausted_census_counts_survive_retained_cause_without_new_reads():
    error = processes.HistoricalProcessError('historical_generation_process_unknown')
    rows = tuple({'pid_open_disappeared': 0, 'descriptor_membership_changed': 1,
                  'corroborated_process_exit': 0, 'pid_census_changed': 0} for _ in range(3))
    error.census_attempt_reasons = rows
    selected = unknown()
    selected.__context__ = error
    value = observer.failure_evidence(selected)
    assert value['exceptions'][1]['census_attempt_reasons'] == list(rows)
    assert value['observation_phase'] == 'post_unwind'
    assert value['exceptions'][1]['code'] == 'historical_generation_process_unknown'


@pytest.mark.parametrize('invalid', ['foreign_type', 'foreign_code', 'private_key', 'bool',
                                    'negative', 'oversize_count', 'oversize_rows', 'zero_reasons'])
def test_exhausted_census_projection_refuses_untrusted_or_unbounded_metadata(invalid):
    error = processes.HistoricalProcessError('historical_generation_process_unknown')
    rows = tuple({'pid_open_disappeared': 0, 'descriptor_membership_changed': 1,
                  'corroborated_process_exit': 0, 'pid_census_changed': 0} for _ in range(3))
    if invalid == 'foreign_type':
        error = ValueError('historical_generation_process_unknown')
    if invalid == 'foreign_code':
        error = processes.HistoricalProcessError('historical_generation_process_reference')
    if invalid == 'private_key':
        rows[0]['private_path'] = 'PRIVATE-PATH-SENTINEL'
    if invalid in {'bool', 'negative', 'oversize_count', 'zero_reasons'}:
        for row in rows:
            row['descriptor_membership_changed'] = {'bool': True, 'negative': -1,
                'oversize_count': 4097, 'zero_reasons': 0}[invalid]
    if invalid == 'oversize_rows':
        rows = rows * 2
    error.census_attempt_reasons = rows
    result = observer.failure_evidence(error)
    assert 'census_attempt_reasons' not in result['exceptions'][0]
    assert 'PRIVATE-PATH-SENTINEL' not in json.dumps(result)


def test_exhausted_count_projection_failure_preserves_original_evidence(monkeypatch):
    def unavailable(error):
        raise ValueError('projection unavailable')
    monkeypatch.setattr(observer, '_census_reason_evidence', unavailable)
    error = processes.HistoricalProcessError('historical_generation_process_unknown')
    result = observer.failure_evidence(error)
    assert result['exceptions'][0]['code'] == 'historical_generation_process_unknown'
    assert 'census_attempt_reasons' not in result['exceptions'][0]



def test_pid_open_observer_keeps_original_refusal_and_private_tokens(census, monkeypatch):
    census.failure = lambda pid: (_ for _ in ()).throw(FileNotFoundError(errno.ENOENT, 'PRIVATE_PATH'))
    evidence = observer.PidOpenEvidence()
    evidence.child_started(2, 'foreign_report_reader')
    evidence.child_started(3, 'ordinary_access_probe')
    evidence.child_reaped(3)
    with observer.observe_pid_opens(evidence):
        with pytest.raises(processes.HistoricalProcessError, match='^historical_generation_process_unknown$'):
            processes.refuse_historical_process_references(census.manifest, tick=lambda: None)
    packet = evidence.packet()
    assert len(packet['observations']) == 6
    assert [row['attempt'] for row in packet['observations']] == [1, 1, 2, 2, 3, 3]
    assert [row['pid_token'] for row in packet['observations']] == [1, 2] * 3
    assert packet['observations'][0]['ownership'] == 'fixture_owned_unreaped_child'
    assert packet['observations'][0]['child_role'] == 'foreign_report_reader'
    assert packet['observations'][1]['ownership'] == 'UNKNOWN'
    assert packet['observations'][1]['previous_fixture_child_reaped'] is True
    assert all(0 <= row['elapsed_ms'] < 5000 for row in packet['observations'])
    assert not any(secret in json.dumps(packet) for secret in ['PRIVATE_PATH', 'cmdline', 'environ'])


def test_pid_open_observer_failure_keeps_exception_and_restores_open(census, monkeypatch):
    error = FileNotFoundError(errno.ENOENT, 'PRIVATE_PATH')
    census.failure = lambda pid: (_ for _ in ()).throw(error)
    evidence = observer.PidOpenEvidence()
    monkeypatch.setattr(evidence, 'record', lambda *args: (_ for _ in ()).throw(RuntimeError('SECRET')))
    original = processes.os.open
    with observer.observe_pid_opens(evidence):
        with pytest.raises(processes.HistoricalProcessError, match='^historical_generation_process_unknown$') as raised:
            processes.refuse_historical_process_references(census.manifest, tick=lambda: None)
    assert processes.os.open is original
    assert raised.value.census_attempt_reasons[0]['pid_open_disappeared'] == 2
    assert evidence.packet()['truncated'] is True
    assert not evidence.packet()['observations']


def test_pid_open_evidence_is_capped_and_never_promotes_reaped_pid():
    from types import SimpleNamespace
    evidence = observer.PidOpenEvidence()
    evidence.child_started(777, 'foreign_report_reader')
    evidence.child_reaped(777)
    for _ in range(10):
        evidence.record('777', 2, SimpleNamespace(started=4.0, last=4.001))
    packet = evidence.packet()
    assert len(packet['observations']) == 8 and packet['truncated'] is True
    assert all(row['ownership'] == 'UNKNOWN' and row['child_role'] is None for row in packet['observations'])
    assert '777' not in json.dumps(packet)


def test_unrelated_open_error_is_same_object_and_unobserved(monkeypatch):
    error = FileNotFoundError(errno.ENOENT, 'SECRET_PATH')
    def original(*args, **kwargs):
        raise error
    monkeypatch.setattr(observer.os, 'open', original)
    evidence = observer.PidOpenEvidence()
    with observer.observe_pid_opens(evidence):
        with pytest.raises(FileNotFoundError) as raised:
            observer.os.open('777', 0, dir_fd=123)
    assert raised.value is error and evidence.packet()['observations'] == []
    assert observer.os.open is original


def test_combined_failure_projection_stays_bounded_and_keeps_original_error(monkeypatch, capsys):
    from types import SimpleNamespace
    evidence = observer.PidOpenEvidence()
    for index in range(8):
        evidence.record(str(700 + index), 2, SimpleNamespace(started=0.0, last=.001))
    name = 'retained_frame_' + 'f' * 48
    namespace = {'HistoricalProcessError': processes.HistoricalProcessError}
    code = f"def {name}(depth):\n    if depth: return {name}(depth-1)\n    raise HistoricalProcessError('historical_generation_process_unknown')\n"
    exec(compile(code, 'retained_source_' + 's' * 96 + '.py', 'exec'), namespace)
    previous = None
    for _ in range(8):
        try:
            namespace[name](16)
        except processes.HistoricalProcessError as caught:
            caught.__context__ = previous
            previous = caught
    original_error = previous
    assert len(json.dumps(observer.failure_evidence(original_error)).encode()) <= 8192
    def failed_run(root):
        raise original_error
    monkeypatch.setattr(observer, 'run', failed_run)
    monkeypatch.setattr(observer, 'PidOpenEvidence', lambda: evidence)
    original_open, original_global = observer.os.open, observer._PID_OPEN_EVIDENCE
    with pytest.raises(processes.HistoricalProcessError) as raised:
        observer.run_observed(None)
    raw = capsys.readouterr().err.strip()
    packet = json.loads(raw)
    assert len(raw.encode()) <= 8192
    assert raised.value is original_error
    assert observer.os.open is original_open and observer._PID_OPEN_EVIDENCE is original_global
    assert packet['exceptions'] and any(row.get('code') == 'historical_generation_process_unknown' for row in packet['exceptions'])
    assert packet.get('pid_open_evidence', {}).get('truncated', packet['truncated']) is True
