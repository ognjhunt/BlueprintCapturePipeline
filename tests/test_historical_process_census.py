"""ADP-081/day-42: bounded census retries never waive a process-reference guard.

These syscall fixtures prove retry boundaries, not native process clearance.
The disposable systemd acceptance lane remains required for native evidence.
"""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_processes.py
import errno
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_lane_historical_processes as processes
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


inspect_process = processes._inspect_process


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
    monkeypatch.setattr(processes, '_namespace', lambda *args, **kwargs:
                        ('pid:[1]', 'user:[1]', 'mnt:[1]'))
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


@pytest.fixture
def descriptor_census(census, monkeypatch):
    """Real parser and descriptor symlinks; fixture bytes supply no native proof."""
    process = census.proc / '2'
    fields = ['S', '0', '0', '0', '0', '0', str(0x00200000), *(['0'] * 12), '123']
    for name, data in dict(
        stat=b'2 (kernel worker) ' + ' '.join(fields).encode(),
        status=b'Name:\tkworker\nKthread:\t1\nUid:\t0 0 0 0\nGid:\t0 0 0 0\n',
        cmdline=b'', environ=b'', maps=b'',
    ).items():
        (process / name).write_bytes(data)
    (process / 'cwd').symlink_to(census.proc.parent)
    descriptors = process / 'fd'
    descriptors.mkdir()
    (descriptors / '0').symlink_to(census.proc.parent)
    (descriptors / '2').symlink_to(census.proc.parent)
    descriptor_identity = descriptors.stat().st_ino
    original_names, original_inspect = processes._Scan.names, processes._inspect_process
    census.fd_censuses, census.churn_forever = 0, False
    census.new_fd_target = census.proc.parent

    def names(scan, directory, limit):
        if os.fstat(directory).st_ino != descriptor_identity:
            return original_names(scan, directory, limit)
        census.fd_censuses += 1
        if census.fd_censuses == 2 or census.churn_forever and census.fd_censuses % 2 == 0:
            previous, current = ('2', '3') if (descriptors / '2').exists() else ('3', '2')
            (descriptors / previous).unlink()
            (descriptors / current).symlink_to(census.new_fd_target)
        return original_names(scan, directory, limit)

    def inspect(scan, directory, pid, *args):
        if pid != '2':
            return original_inspect(scan, directory, pid, *args)
        census.inspections.append(pid)
        return inspect_process(scan, directory, pid, *args)

    monkeypatch.setattr(processes._Scan, 'names', names)
    monkeypatch.setattr(processes, '_inspect_process', inspect)
    return census


def test_fd_membership_change_restarts_full_census_and_reinspects_same_process(descriptor_census):
    run(descriptor_census)
    assert descriptor_census.inspections == ['1', '2', '1', '2', '3']
    assert descriptor_census.fd_censuses == 4
    assert len({id(scan) for scan in descriptor_census.scans}) == 1


def test_continual_fd_membership_churn_remains_unknown_after_three_passes(descriptor_census):
    descriptor_census.churn_forever = True
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(descriptor_census)
    assert descriptor_census.inspections == ['1', '2'] * 3
    assert descriptor_census.fd_censuses == 6


@pytest.mark.parametrize('channel', ['fd', 'cwd'])
def test_fd_churn_never_discards_an_observed_reference(descriptor_census, channel):
    process = descriptor_census.proc / '2'
    selected = descriptor_census.proc.parent / 'selected'
    selected.mkdir()
    descriptor_census.manifest['target_path'] = str(selected)
    link = process / ('fd/0' if channel == 'fd' else 'cwd')
    link.unlink()
    link.symlink_to(selected)
    with pytest.raises(processes.HistoricalProcessError, match='process_reference'):
        run(descriptor_census)
    assert descriptor_census.inspections == ['1', '2']
    assert descriptor_census.fd_censuses == 2


def test_new_reference_in_changed_fd_census_is_inspected_on_next_pass(descriptor_census):
    selected = descriptor_census.proc.parent / 'selected'
    selected.mkdir()
    descriptor_census.manifest['target_path'] = str(selected)
    descriptor_census.new_fd_target = selected
    with pytest.raises(processes.HistoricalProcessError, match='process_reference'):
        run(descriptor_census)
    assert descriptor_census.inspections == ['1', '2', '1', '2']
    assert descriptor_census.fd_censuses == 4


@pytest.mark.parametrize('change', ['start_identity', 'namespace', 'missing_stat'])
def test_fd_churn_does_not_skip_final_process_identity_or_channel_checks(
    descriptor_census, monkeypatch, change,
):
    read, namespace = processes._Scan.read, processes._namespace

    def current_read(scan, directory, name, cap=1024**2):
        if name == 'stat' and descriptor_census.fd_censuses == 2:
            if change == 'missing_stat':
                raise FileNotFoundError(errno.ENOENT, 'process ended')
            if change == 'start_identity':
                return read(scan, directory, name, cap).rsplit(b' ', 1)[0] + b' 124'
        return read(scan, directory, name, cap)

    def current_namespace(*args, **kwargs):
        if change == 'namespace' and descriptor_census.fd_censuses == 2:
            return ('pid:[1]', 'user:[1]', 'mnt:[2]')
        return namespace(*args, **kwargs)

    monkeypatch.setattr(processes._Scan, 'read', current_read)
    monkeypatch.setattr(processes, '_namespace', current_namespace)
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(descriptor_census)
    assert descriptor_census.inspections == ['1', '2']
    assert descriptor_census.fd_censuses == 2


@pytest.fixture
def exiting_process(descriptor_census, monkeypatch):
    """Exercise the real parser's user-process channel and retained-stat reads."""
    state = descriptor_census
    process = state.proc / '2'
    (process / 'mountinfo').write_bytes(b'fixture mount view')
    monkeypatch.setattr(processes, 'kernel_has_no_user_memory', lambda *args: False)

    def filesystem_view(scan, directory, view, *args):
        scan.views[view] = (scan.read(directory, 'mountinfo'), ())

    monkeypatch.setattr(processes, '_known_filesystem_view', filesystem_view)
    state.channel = 'environ'
    state.channel_error = ProcessLookupError(errno.ESRCH, 'exited')
    state.final_error = ProcessLookupError(errno.ESRCH, 'exited')
    state.final_value = None
    state.channel_failed = False
    state.final_reads = 0
    read = processes._Scan.read

    def current_read(scan, directory, name, cap=1024**2):
        if name == state.channel:
            state.channel_failed = True
            raise state.channel_error
        if name == 'stat' and state.channel_failed:
            state.channel_failed = False
            state.final_reads += 1
            if state.final_error is not None:
                raise state.final_error
            if state.final_value is not None:
                return state.final_value
        return read(scan, directory, name, cap)

    monkeypatch.setattr(processes._Scan, 'read', current_read)
    return state


@pytest.mark.parametrize('channel', ['environ', 'maps'])
@pytest.mark.parametrize('final_errno', [errno.ENOENT, errno.ESRCH])
def test_corroborated_exit_requires_complete_new_census(exiting_process, channel, final_errno):
    state = exiting_process
    state.channel = channel
    state.final_error = OSError(final_errno, 'exited')
    state.snapshots = [['1', '2', '3', '99'], ['1', '3', '99'], ['1', '3', '99']]
    budget = ReferenceCollectionBudget()
    run(state, budget=budget)
    assert state.inspections == ['1', '2', '1', '3']
    assert state.final_reads == 1
    assert len(state.scans) == 3 and len({id(scan) for scan in state.scans}) == 1
    assert budget.counts['entries'] == state.scans[0].entries == 12
    assert budget.counts['raw_bytes'] == state.scans[0].raw_bytes > 3 * len(b'observed')


def test_perpetual_corroborated_exit_exhausts_original_three_passes(exiting_process):
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(exiting_process)
    assert exiting_process.inspections == ['1', '2'] * 3
    assert exiting_process.final_reads == 3


@pytest.mark.parametrize('final', ['same_identity', 'changed_identity', 'malformed', 'permission', 'io'])
def test_uncorroborated_exit_cannot_retry_or_skip_identity(exiting_process, final):
    state = exiting_process
    state.final_error = None
    if final == 'changed_identity':
        raw = (state.proc / '2/stat').read_bytes()
        state.final_value = raw.rsplit(b' ', 1)[0] + b' 124'
    elif final == 'malformed':
        state.final_value = b'unknown'
    elif final in ('permission', 'io'):
        state.final_error = OSError(errno.EACCES if final == 'permission' else errno.EIO, 'unknown')
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(state)
    assert state.inspections == ['1', '2'] and state.final_reads == 1
    assert len(state.scans) == 1


@pytest.mark.parametrize('error', [PermissionError(errno.EACCES, 'hidden'),
                                 OSError(errno.EIO, 'unknown'),
                                 ProcessLookupError(errno.EIO, 'not ESRCH')])
def test_non_esrch_user_channel_failure_does_not_probe_or_retry(exiting_process, error):
    exiting_process.channel_error = error
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(exiting_process)
    assert exiting_process.inspections == ['1', '2']
    assert exiting_process.final_reads == 0 and len(exiting_process.scans) == 1


def test_exit_never_discards_already_observed_reference(exiting_process):
    selected = exiting_process.proc.parent / 'selected'
    selected.mkdir()
    exiting_process.manifest['target_path'] = str(selected)
    cwd = exiting_process.proc / '2/cwd'
    cwd.unlink()
    cwd.symlink_to(selected)
    with pytest.raises(processes.HistoricalProcessError, match='process_reference'):
        run(exiting_process)
    assert exiting_process.inspections == ['1', '2']
    assert exiting_process.final_reads == 0 and len(exiting_process.scans) == 1


@pytest.mark.parametrize('kind', ['entries', 'raw_bytes', 'clock'])
def test_corroborated_exit_cannot_renew_original_scan_budget(exiting_process, monkeypatch, kind):
    budget = ReferenceCollectionBudget()
    now = [0.0]
    monkeypatch.setattr(processes.time, 'monotonic', lambda: now[0])
    read = processes._Scan.read

    def current_read(scan, directory, name, cap=1024**2):
        if name == 'stat' and exiting_process.channel_failed:
            if kind == 'clock':
                now[0] = 5.0
            else:
                budget.charge(kind, budget.limits[kind] - budget.counts[kind])
        return read(scan, directory, name, cap)

    monkeypatch.setattr(processes._Scan, 'read', current_read)
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(exiting_process, budget=budget)
    assert exiting_process.inspections == (['1', '2', '1'] if kind == 'raw_bytes' else ['1', '2'])
    assert exiting_process.final_reads == 1
    if kind != 'clock':
        assert budget.failure == 'reference_' + kind + '_limit'


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux retained proc descriptor semantics')
def test_actual_exited_user_process_requires_whole_census_restart(monkeypatch):
    """Real ESRCH evidence; this does not authorize a generation deletion."""
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    directory = os.open('/proc/' + str(process.pid), os.O_RDONLY | os.O_DIRECTORY)
    scan = processes._Scan(lambda: None)
    read = scan.read
    observed = []

    def current_read(directory, name, cap=1024**2):
        if name == 'environ':
            process.terminate()
            process.wait(timeout=5)
        try:
            return read(directory, name, cap)
        except ProcessLookupError as error:
            observed.append((name, error.errno))
            raise

    monkeypatch.setattr(scan, 'read', current_read)
    monkeypatch.setattr(processes, '_known_filesystem_view', lambda *args: None)
    root = os.stat('/')
    try:
        namespaces = processes._namespace(directory)
        with pytest.raises(processes._ProcessExited):
            inspect_process(scan, directory, str(process.pid), '/fixture/selected', {(0, 0)},
                            namespaces, namespaces[2], (root.st_dev, root.st_ino))
        assert observed == [('environ', errno.ESRCH), ('stat', errno.ESRCH)]
    finally:
        os.close(directory)
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
