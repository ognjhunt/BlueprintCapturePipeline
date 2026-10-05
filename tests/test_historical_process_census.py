"""ADP-081/day-42: bounded census retries never waive a process-reference guard.

These syscall fixtures prove retry boundaries, not native process clearance.
The disposable systemd acceptance lane remains required for native evidence.
"""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_processes.py
import errno
import os
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_lane_historical_processes as processes
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


@pytest.fixture
def census(tmp_path, monkeypatch):
    proc = tmp_path / 'proc'
    proc.mkdir()
    for pid in ('1', '2', '3', '99'):
        (proc / pid).mkdir()
        (proc / pid / 'root').symlink_to(tmp_path)
        (proc / pid / 'observed').write_bytes(b'observed')
    real_open, real_names = os.open, processes._Scan.names
    state = SimpleNamespace(snapshots=[], inspections=[], scans=[], opens=[], failure=None)

    def open_process(path, flags, *args, **kwargs):
        if path == '/proc':
            path = proc
        elif path in ('2', '3'):
            state.opens.append(path)
            if state.failure is not None:
                state.failure(path)
        return real_open(path, flags, *args, **kwargs)

    def names(scan, directory, limit):
        # Charge the actual directory enumeration on every pass.
        actual = real_names(scan, directory, limit)
        state.scans.append(scan)
        return state.snapshots.pop(0) if state.snapshots else actual

    def inspect(scan, directory, pid, *args):
        state.inspections.append(pid)
        assert scan.read(directory, 'observed') == b'observed'
        return set()

    monkeypatch.setattr(processes.os, 'open', open_process)
    monkeypatch.setattr(processes.os, 'getpid', lambda: 99)
    monkeypatch.setattr(processes.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(processes.sys, 'platform', 'linux')
    monkeypatch.setattr(processes, '_namespace', lambda *args: ('pid:[1]', 'user:[1]', 'mnt:[1]'))
    monkeypatch.setattr(processes._Scan, 'names', names)
    monkeypatch.setattr(processes, '_inspect_process', inspect)
    state.manifest = dict(target_path='/fixture/selected', members=[dict(version=[1, 1])])
    state.proc = proc
    return state


def run(census, **kwargs):
    return processes.refuse_historical_process_references(census.manifest, tick=lambda: None, **kwargs)


def test_missing_pid_open_requires_a_complete_new_census(census):
    # The initial census names a PID that has already exited by its open.
    census.snapshots = [['1', '2', '3', '99'], ['1', '3', '99'], ['1', '3', '99']]
    def exited(pid):
        if pid == '2':
            raise FileNotFoundError(errno.ENOENT, 'exited')
    census.failure = exited
    budget = ReferenceCollectionBudget()
    run(census, budget=budget)
    assert census.inspections == ['1', '1', '3']
    assert census.opens == ['2', '3']
    assert len(census.scans) == 3 and len({id(scan) for scan in census.scans}) == 1
    scan = census.scans[0]
    assert budget.counts['raw_bytes'] == scan.raw_bytes == 3 * len(b'observed')
    assert budget.counts['entries'] == scan.entries == 3 * 4


@pytest.mark.parametrize('changed', [['1', '3', '99'], ['1', '2', '3', '99']])
def test_changed_census_requires_every_surviving_and_new_pid_to_be_reinspected(census, changed):
    before = ['1', '2', '99']
    census.snapshots = [before, changed, changed, changed]
    run(census)
    assert census.inspections == ['1', '2'] + [pid for pid in changed if pid != '99']
    assert len(census.scans) == 4


def test_perpetual_census_churn_refuses_after_three_complete_passes(census):
    census.snapshots = [['1', '2', '99'], ['1', '3', '99']] * 3
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(census)
    assert census.inspections == ['1', '2'] * 3
    assert len(census.scans) == 6


def test_perpetually_missing_pid_refuses_after_three_incomplete_passes(census):
    def missing(pid):
        raise FileNotFoundError(errno.ENOENT, 'exited')
    census.failure = missing
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(census)
    assert census.inspections == ['1'] * 3 and census.opens == ['2'] * 3
    assert len(census.scans) == 3


@pytest.mark.parametrize('error', [PermissionError(errno.EACCES, 'hidden'),
                                 OSError(errno.EIO, 'unreadable'),
                                 ProcessLookupError(errno.ESRCH, 'unknown')])
def test_other_pid_open_errors_are_not_retried_or_skipped(census, error):
    def refused(pid):
        raise error
    census.failure = refused
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(census)
    assert census.opens == ['2'] and census.inspections == ['1']
    assert len(census.scans) == 1


@pytest.mark.parametrize('error', [FileNotFoundError(errno.ENOENT, 'missing channel'),
                                 PermissionError(errno.EACCES, 'unreadable channel'),
                                 processes.HistoricalProcessError('historical_generation_process_unknown'),
                                 processes.HistoricalProcessError('historical_generation_process_view_unknown')])
def test_process_channel_unknowns_refuse_without_retry(census, monkeypatch, error):
    def inspect(*args):
        raise error
    monkeypatch.setattr(processes, '_inspect_process', inspect)
    with pytest.raises(processes.HistoricalProcessError, match='process_(view_)?unknown'):
        run(census)
    assert len(census.scans) == 1


def test_observed_reference_refuses_immediately_even_if_process_census_would_change(census, monkeypatch):
    census.snapshots = [['1', '2', '99'], ['1', '99']]
    monkeypatch.setattr(processes, '_inspect_process', lambda *args: {'fd'})
    with pytest.raises(processes.HistoricalProcessError, match='process_reference'):
        run(census)
    assert len(census.scans) == 1 and census.snapshots == [['1', '99']]


def test_retry_cannot_renew_original_five_second_scan_clock(census, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(processes.time, 'monotonic', lambda: now[0])
    def exited(pid):
        now[0] = 5.0
        raise FileNotFoundError(errno.ENOENT, 'exited')
    census.failure = exited
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(census)
    assert census.opens == ['2'] and len(census.scans) == 1


@pytest.mark.parametrize('kind', ['entries', 'raw_bytes'])
def test_retry_cannot_renew_shared_entry_or_byte_allowance(census, kind):
    budget = ReferenceCollectionBudget()
    def exited(pid):
        budget.charge(kind, budget.limits[kind] - budget.counts[kind])
        raise FileNotFoundError(errno.ENOENT, 'exited')
    census.failure = exited
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(census, budget=budget)
    assert budget.failure == 'reference_' + kind + '_limit'
    assert census.opens == ['2']
