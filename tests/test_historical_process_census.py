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
def user_memory_exit(descriptor_census, monkeypatch):
    """Real parser/held directory seam; no assertion of native clearance."""
    state = descriptor_census
    process = state.proc / '2'
    (process / 'mountinfo').write_bytes(b'fixture mount view')
    monkeypatch.setattr(processes, 'kernel_has_no_user_memory', lambda *args: False)

    def known_view(scan, directory, view, *args):
        scan.views[view] = (b'fixture mount view', ())

    monkeypatch.setattr(processes, '_known_filesystem_view', known_view)
    read = processes._Scan.read
    state.exit_mode, state.failed_channel, state.failures = 'gone', 'environ', 0
    state.read_error = ProcessLookupError(errno.ESRCH, 'process ended during read')
    state.before_failure = lambda scan: None
    process_identity = process.stat().st_ino

    def current_read(scan, directory, name, cap=1024**2):
        if os.fstat(directory).st_ino == process_identity and name == state.failed_channel:
            scan.tick()
            state.failures += 1
            state.before_failure(scan)
            if state.exit_mode != 'live':
                process.rename(state.proc.parent / 'exited-process')
            if state.exit_mode == 'reused':
                process.mkdir()
            raise state.read_error
        return read(scan, directory, name, cap)

    monkeypatch.setattr(processes._Scan, 'read', current_read)
    return state


@pytest.mark.parametrize('channel', ['environ', 'maps'])
def test_proven_user_memory_exit_restarts_every_pid_with_same_budget(user_memory_exit, channel):
    state = user_memory_exit
    state.failed_channel = channel
    budget = ReferenceCollectionBudget()
    run(state, budget=budget)
    assert state.inspections == ['1', '2', '1', '3']
    assert state.opens == ['2', '2', '3']  # second open proves real pathname absence
    assert state.failures == 1 and len(state.scans) == 3
    assert len({id(scan) for scan in state.scans}) == 1
    scan = state.scans[0]
    assert budget.counts['entries'] == scan.entries > 4
    assert budget.counts['raw_bytes'] == scan.raw_bytes > 3 * len(b'observed')


@pytest.mark.parametrize('mode', ['live', 'reused'])
def test_esrch_with_live_or_reused_pid_remains_unknown(user_memory_exit, mode):
    state = user_memory_exit
    state.exit_mode = mode
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(state)
    assert state.inspections == ['1', '2'] and len(state.scans) == 1
    assert state.opens == ['2', '2'] and state.failures == 1


@pytest.mark.parametrize('channel', ['cwd', 'cmdline'])
def test_proven_exit_cannot_discard_a_previously_observed_reference(user_memory_exit, channel):
    state = user_memory_exit
    process = state.proc / '2'
    selected = state.proc.parent / 'selected'
    selected.mkdir()
    state.manifest['target_path'] = str(selected)
    if channel == 'cwd':
        (process / channel).unlink()
        (process / channel).symlink_to(selected)
    else:
        (process / channel).write_bytes(os.fsencode(selected))
    with pytest.raises(processes.HistoricalProcessError, match='process_reference'):
        run(state)
    assert state.inspections == ['1', '2'] and state.opens == ['2']
    assert state.failures == 1 and len(state.scans) == 1


@pytest.mark.parametrize('error', [PermissionError(errno.EACCES, 'hidden'),
                                 OSError(errno.EIO, 'unreadable'),
                                 FileNotFoundError(errno.ENOENT, 'missing channel')])
def test_non_esrch_memory_errors_remain_unknown_even_when_pid_disappears(user_memory_exit, error):
    state = user_memory_exit
    state.read_error = error
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(state)
    assert state.inspections == ['1', '2'] and state.opens == ['2']
    assert state.failures == 1 and len(state.scans) == 1


def test_cmdline_esrch_remains_unknown_even_when_pid_disappears(user_memory_exit):
    state = user_memory_exit
    state.failed_channel = 'cmdline'
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(state)
    assert state.inspections == ['1', '2'] and state.opens == ['2']
    assert state.failures == 1 and len(state.scans) == 1


def test_proven_exit_cannot_renew_original_scan_clock(user_memory_exit, monkeypatch):
    state = user_memory_exit
    now = [0.0]
    monkeypatch.setattr(processes.time, 'monotonic', lambda: now[0])
    state.before_failure = lambda scan: now.__setitem__(0, 5.0)
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(state)
    assert state.inspections == ['1', '2'] and state.opens == ['2']
    assert state.failures == 1 and len(state.scans) == 1


@pytest.mark.parametrize('kind', ['entries', 'raw_bytes'])
def test_proven_exit_cannot_renew_shared_reference_allowance(user_memory_exit, kind):
    state = user_memory_exit
    budget = ReferenceCollectionBudget()
    state.before_failure = lambda scan: budget.charge(kind, budget.limits[kind] - budget.counts[kind])
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(state, budget=budget)
    assert budget.failure == 'reference_' + kind + '_limit'
    assert state.inspections == (['1', '2'] if kind == 'entries' else ['1', '2', '1'])
    assert state.opens == ['2', '2']
    assert state.failures == 1


def test_proven_exit_requires_reinspection_of_surviving_references(user_memory_exit, monkeypatch):
    state = user_memory_exit
    inspect = processes._inspect_process

    def current_inspect(scan, directory, pid, *args):
        channels = inspect(scan, directory, pid, *args)
        return {'fd'} if pid == '3' else channels

    monkeypatch.setattr(processes, '_inspect_process', current_inspect)
    with pytest.raises(processes.HistoricalProcessError, match='process_reference'):
        run(state)
    assert state.inspections == ['1', '2', '1', '3']
    assert state.failures == 1 and len(state.scans) == 2


def test_repeated_proven_exits_refuse_after_original_three_passes(census, monkeypatch):
    names = processes._Scan.names

    def current_names(scan, directory, limit):
        process = census.proc / '2'
        if not process.exists():
            process.mkdir()
        return names(scan, directory, limit)

    def ended(scan, directory, pid, *args):
        census.inspections.append(pid)
        if pid == '2':
            (census.proc / pid).rename(census.proc.parent / ('exited-' + str(len(census.scans))))
            raise processes._UserMemoryReadUnavailable('historical_generation_process_unknown')
        return set()

    monkeypatch.setattr(processes._Scan, 'names', current_names)
    monkeypatch.setattr(processes, '_inspect_process', ended)
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(census)
    assert census.inspections == ['1', '2'] * 3
    assert census.opens == ['2', '2'] * 3
    assert len(census.scans) == 3 and len({id(scan) for scan in census.scans}) == 1
