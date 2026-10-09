"""ADP-081/day-42: bounded census retries never waive a process-reference guard.

These syscall fixtures prove retry boundaries, not native process clearance.
The disposable systemd acceptance lane remains required for native evidence.
"""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_processes.py
import errno
import os
import select
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
    assert census.inspections == ['1', '3', '1', '3']
    assert census.opens == ['2', '3', '3']
    assert len(census.scans) == 3 and len({id(scan) for scan in census.scans}) == 1
    scan = census.scans[0]
    assert budget.counts['raw_bytes'] == scan.raw_bytes == 4 * len(b'observed')
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
    assert census.inspections == ['1'] * 3 and census.opens == ['2', '3'] * 3
    assert len(census.scans) == 3


@pytest.mark.parametrize('transient', ['missing_pid', 'descriptor_churn', 'process_exit'])
def test_incomplete_pass_cannot_clear_even_when_later_process_is_clear(census, monkeypatch, transient):
    actual = processes._inspect_process
    def inspect(scan, directory, pid, *args):
        if pid == '2' and transient != 'missing_pid':
            census.inspections.append(pid)
            cls = processes._DescriptorCensusChanged if transient == 'descriptor_churn' else processes._ProcessExited
            raise cls('historical_generation_process_unknown')
        return actual(scan, directory, pid, *args)
    if transient == 'missing_pid':
        def missing(pid):
            if pid == '2':
                raise FileNotFoundError(errno.ENOENT, 'exited')
        census.failure = missing
    monkeypatch.setattr(processes, '_inspect_process', inspect)
    with pytest.raises(processes.HistoricalProcessError, match='^historical_generation_process_unknown$'):
        run(census)
    assert census.inspections.count('3') == 3
    assert len(census.scans) == 3 and len({id(scan) for scan in census.scans}) == 1


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


@pytest.mark.parametrize('transient', ['missing_pid', 'descriptor_churn', 'process_exit'])
def test_earlier_transient_cannot_starve_later_conclusive_reference(census, monkeypatch, transient):
    """Synthetic ordering counterexample; not the retained CI PID/FD cause."""
    census.snapshots = [['1', '2', '3', '99']] * 3
    actual = processes._inspect_process

    def inspect(scan, directory, pid, *args):
        if pid == '2':
            census.inspections.append(pid)
            if transient == 'descriptor_churn':
                raise processes._DescriptorCensusChanged('historical_generation_process_unknown')
            if transient == 'process_exit':
                raise processes._ProcessExited('historical_generation_process_unknown')
        if pid == '3':
            census.inspections.append(pid)
            return {'fd'}
        return actual(scan, directory, pid, *args)

    if transient == 'missing_pid':
        def missing(pid):
            if pid == '2':
                raise FileNotFoundError(errno.ENOENT, 'exited')
        census.failure = missing
    monkeypatch.setattr(processes, '_inspect_process', inspect)
    with pytest.raises(processes.HistoricalProcessError, match='^historical_generation_process_reference$'):
        run(census)
    assert census.inspections[-1] == '3'
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
    assert census.opens == ['2', '3']


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
    census.churn_at_second_census = True
    census.new_fd_target = census.proc.parent

    def names(scan, directory, limit):
        if os.fstat(directory).st_ino != descriptor_identity:
            return original_names(scan, directory, limit)
        census.fd_censuses += 1
        if (census.churn_at_second_census and census.fd_censuses == 2
                or census.churn_forever and census.fd_censuses % 2 == 0):
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
    assert descriptor_census.inspections == ['1', '2', '3', '1', '2', '3']
    assert descriptor_census.fd_censuses == 4
    assert len({id(scan) for scan in descriptor_census.scans}) == 1


def test_continual_fd_membership_churn_remains_unknown_after_three_passes(descriptor_census):
    descriptor_census.churn_forever = True
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(descriptor_census)
    assert descriptor_census.inspections == ['1', '2', '3'] * 3
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
    assert descriptor_census.inspections == ['1', '2', '3', '1', '2']
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
def disappearing_descriptor(descriptor_census, monkeypatch):
    """An enumerated FD closes before stat, or between stat and readlink."""
    state = descriptor_census
    state.churn_at_second_census = False
    descriptors = state.proc / '2/fd'
    identity = descriptors.stat().st_ino
    state.operation, state.error_errno = 'stat', errno.ENOENT
    state.remove, state.repeat, state.replacement = True, False, None
    state.failures = []
    original_stat, original_readlink = os.stat, os.readlink
    original_names = processes._Scan.names

    def names(scan, directory, limit):
        result = original_names(scan, directory, limit)
        return sorted(result, key=int) if os.fstat(directory).st_ino == identity else result

    def disappeared(operation, path, kwargs):
        directory = kwargs.get('dir_fd')
        if operation != state.operation or path not in ('2', '3') or directory is None \
                or os.fstat(directory).st_ino != identity \
                or state.fd_censuses % 2 != 1 \
                or state.fd_censuses in state.failures \
                or state.failures and not state.repeat:
            return
        state.failures.append(state.fd_censuses)
        if state.remove:
            (descriptors / path).unlink()
            if state.repeat or state.replacement is not None:
                (descriptors / ('3' if path == '2' else '2')).symlink_to(
                    state.replacement or state.proc.parent)
        raise OSError(state.error_errno, 'fixture descriptor observation unavailable')

    def current_stat(path, *args, **kwargs):
        disappeared('stat', path, kwargs)
        return original_stat(path, *args, **kwargs)

    def current_readlink(path, *args, **kwargs):
        disappeared('readlink', path, kwargs)
        return original_readlink(path, *args, **kwargs)

    monkeypatch.setattr(processes.os, 'stat', current_stat)
    monkeypatch.setattr(processes.os, 'readlink', current_readlink)
    monkeypatch.setattr(processes._Scan, 'names', names)
    return state


@pytest.mark.parametrize('operation', ['stat', 'readlink'])
def test_disappearing_fd_requires_complete_fresh_census_and_same_shared_budget(
    disappearing_descriptor, operation,
):
    state = disappearing_descriptor
    state.operation = operation
    budget = ReferenceCollectionBudget()
    run(state, budget=budget)
    assert state.inspections == ['1', '2', '3', '1', '2', '3']
    assert state.fd_censuses == 4 and state.failures == [1]
    assert len({id(scan) for scan in state.scans}) == 1
    assert budget.counts['entries'] == state.scans[0].entries > 12
    assert budget.counts['raw_bytes'] == state.scans[0].raw_bytes > 3 * len(b'observed')


@pytest.mark.parametrize('operation', ['stat', 'readlink'])
def test_continually_disappearing_fd_exhausts_original_three_complete_attempts(
    disappearing_descriptor, operation,
):
    state = disappearing_descriptor
    state.operation, state.repeat = operation, True
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(state)
    assert state.inspections == ['1', '2', '3'] * 3
    assert state.fd_censuses == 6 and state.failures == [1, 3, 5]


@pytest.mark.parametrize('operation', ['stat', 'readlink'])
def test_enoent_without_missing_fd_in_second_census_cannot_retry(
    disappearing_descriptor, operation,
):
    state = disappearing_descriptor
    state.operation, state.remove = operation, False
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(state)
    assert state.inspections == ['1', '2'] and state.fd_censuses == 2


@pytest.mark.parametrize('operation', ['stat', 'readlink'])
@pytest.mark.parametrize('error_errno', [errno.EACCES, errno.EIO, errno.ESRCH])
def test_other_fd_errors_cannot_retry_or_skip_process(disappearing_descriptor, operation, error_errno):
    state = disappearing_descriptor
    state.operation, state.error_errno = operation, error_errno
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(state)
    assert state.inspections == ['1', '2'] and state.fd_censuses == 1


@pytest.mark.parametrize('reference', ['previous_fd', 'cwd', 'same_stat_inode'])
def test_disappearing_fd_keeps_every_already_observed_reference(disappearing_descriptor, reference):
    state = disappearing_descriptor
    selected = state.proc.parent / 'selected'
    selected.mkdir()
    state.manifest['target_path'] = str(selected)
    link = state.proc / '2' / dict(previous_fd='fd/0', cwd='cwd', same_stat_inode='fd/2')[reference]
    link.unlink()
    link.symlink_to(selected)
    if reference == 'same_stat_inode':
        identity = selected.stat()
        state.manifest['members'] = [dict(version=[identity.st_dev, identity.st_ino])]
        state.operation = 'readlink'
    with pytest.raises(processes.HistoricalProcessError, match='process_reference'):
        run(state)
    assert state.inspections == ['1', '2'] and state.fd_censuses == 1


def test_new_reference_after_fd_disappearance_is_seen_on_fresh_pass(disappearing_descriptor):
    state = disappearing_descriptor
    selected = state.proc.parent / 'selected'
    selected.mkdir()
    state.manifest['target_path'] = str(selected)
    state.replacement = selected
    with pytest.raises(processes.HistoricalProcessError, match='process_reference'):
        run(state)
    assert state.inspections == ['1', '2', '3', '1', '2'] and state.fd_censuses == 4


@pytest.mark.parametrize('change', ['start_identity', 'namespace', 'missing_stat', 'kernel_proof'])
def test_disappearing_fd_cannot_skip_final_process_identity(disappearing_descriptor, monkeypatch, change):
    state = disappearing_descriptor
    read, namespace = processes._Scan.read, processes._namespace

    def current_read(scan, directory, name, cap=1024**2):
        if name == 'stat' and state.fd_censuses == 2:
            if change == 'missing_stat':
                raise FileNotFoundError(errno.ENOENT, 'fixture process ended')
            if change == 'start_identity':
                return read(scan, directory, name, cap).rsplit(b' ', 1)[0] + b' 124'
        return read(scan, directory, name, cap)

    def current_namespace(*args, **kwargs):
        if change == 'namespace' and state.fd_censuses == 2:
            return ('pid:[1]', 'user:[1]', 'mnt:[2]')
        return namespace(*args, **kwargs)

    monkeypatch.setattr(processes._Scan, 'read', current_read)
    monkeypatch.setattr(processes, '_namespace', current_namespace)
    if change == 'kernel_proof':
        proof = processes.kernel_has_no_user_memory
        monkeypatch.setattr(processes, 'kernel_has_no_user_memory',
            lambda *args: state.fd_censuses != 2 and proof(*args))
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(state)
    assert state.inspections == ['1', '2'] and state.fd_censuses == 2


@pytest.mark.parametrize('change', ['mount_bytes', 'root_identity'])
def test_disappearing_fd_cannot_skip_final_user_filesystem_view(
    disappearing_descriptor, monkeypatch, change,
):
    state = disappearing_descriptor
    process = state.proc / '2'
    (process / 'mountinfo').write_bytes(b'fixture mount view')
    monkeypatch.setattr(processes, 'kernel_has_no_user_memory', lambda *args: False)

    def filesystem_view(scan, directory, view, *args):
        scan.views[view] = (scan.read(directory, 'mountinfo'), ())

    monkeypatch.setattr(processes, '_known_filesystem_view', filesystem_view)
    read, current_stat = processes._Scan.read, processes.os.stat

    def final_read(scan, directory, name, cap=1024**2):
        if name == 'mountinfo' and state.fd_censuses == 2 and change == 'mount_bytes':
            return b'different fixture mount view'
        return read(scan, directory, name, cap)

    def final_stat(path, *args, **kwargs):
        result = current_stat(path, *args, **kwargs)
        if path == 'root' and state.fd_censuses == 2 and change == 'root_identity':
            return SimpleNamespace(st_dev=result.st_dev, st_ino=result.st_ino + 1)
        return result

    monkeypatch.setattr(processes._Scan, 'read', final_read)
    monkeypatch.setattr(processes.os, 'stat', final_stat)
    with pytest.raises(processes.HistoricalProcessError, match='process_view_unknown'):
        run(state)
    assert state.inspections == ['1', '2'] and state.fd_censuses == 2


@pytest.mark.parametrize('limit', ['entries', 'raw_bytes', 'clock'])
def test_fd_disappearance_cannot_renew_original_scan_budget(disappearing_descriptor, monkeypatch, limit):
    state = disappearing_descriptor
    budget, now = ReferenceCollectionBudget(), [0.0]
    monkeypatch.setattr(processes.time, 'monotonic', lambda: now[0])
    names = processes._Scan.names

    def final_names(scan, directory, maximum):
        result = names(scan, directory, maximum)
        if state.fd_censuses == 2:
            if limit == 'clock':
                now[0] = 5.0
            else:
                budget.charge(limit, budget.limits[limit] - budget.counts[limit])
        return result

    monkeypatch.setattr(processes._Scan, 'names', final_names)
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(state, budget=budget)
    assert state.inspections == (['1', '2', '3'] if limit == 'entries' else ['1', '2']) and state.fd_censuses == 2
    assert len({id(scan) for scan in state.scans}) == 1


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
    assert state.inspections == ['1', '2', '3', '1', '3']
    assert state.final_reads == 1
    assert len(state.scans) == 3 and len({id(scan) for scan in state.scans}) == 1
    assert budget.counts['entries'] == state.scans[0].entries == 12
    assert budget.counts['raw_bytes'] == state.scans[0].raw_bytes > 3 * len(b'observed')


def test_perpetual_corroborated_exit_exhausts_original_three_passes(exiting_process):
    with pytest.raises(processes.HistoricalProcessError, match='process_unknown'):
        run(exiting_process)
    assert exiting_process.inspections == ['1', '2', '3'] * 3
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
    assert exiting_process.inspections == (['1', '2'] if kind == 'clock' else ['1', '2', '3'])
    assert exiting_process.final_reads == 1
    if kind != 'clock':
        assert budget.failure == 'reference_' + kind + '_limit'


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux retained proc descriptor semantics')
def test_actual_closed_descriptor_requires_reinspection_of_retained_process(monkeypatch):
    """Real FD ENOENT and stable follow-up; not native retirement authority."""
    child = subprocess.Popen([sys.executable, '-c',
        'import os, sys; fd = os.open("/dev/null", os.O_RDONLY); '
        'print(fd, flush=True); sys.stdin.readline(); os.close(fd); '
        'print("closed", flush=True); sys.stdin.readline()'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    directory = None
    try:
        assert child.stdout is not None and child.stdin is not None
        def line():
            assert select.select([child.stdout], [], [], 5)[0], 'fixture child response timed out'
            return child.stdout.readline().strip()
        slot = line()
        assert slot.isdecimal()
        directory = os.open('/proc/' + str(child.pid), os.O_RDONLY | os.O_DIRECTORY)
        descriptor_identity = os.stat('fd', dir_fd=directory).st_ino
        scan = processes._Scan(lambda: None)
        original_names = scan.names
        closed, fd_censuses = False, 0

        def current_names(directory, limit):
            nonlocal closed, fd_censuses
            result = original_names(directory, limit)
            if os.fstat(directory).st_ino == descriptor_identity:
                fd_censuses += 1
                if not closed:
                    assert slot in result
                    child.stdin.write('close\n')
                    child.stdin.flush()
                    assert line() == 'closed'
                    closed = True
            return result

        def filesystem_view(scan, directory, view, *args):
            scan.views[view] = (scan.read(directory, 'mountinfo'), ())

        monkeypatch.setattr(scan, 'names', current_names)
        monkeypatch.setattr(processes, '_known_filesystem_view', filesystem_view)
        root = os.stat('/')
        namespaces = processes._namespace(directory)
        args = (scan, directory, str(child.pid), '/fixture/selected', {(0, 0)},
                namespaces, namespaces[2], (root.st_dev, root.st_ino))
        with pytest.raises(processes._DescriptorCensusChanged):
            inspect_process(*args)
        assert fd_censuses == 2 and closed and child.poll() is None
        assert inspect_process(*args) == set()
        assert fd_censuses == 4
    finally:
        if directory is not None:
            os.close(directory)
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
        if child.stdin is not None:
            child.stdin.close()
        if child.stdout is not None:
            child.stdout.close()


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


def test_exhausted_three_passes_retains_only_fixed_reason_counts(census):
    def missing(pid):
        if pid == '2':
            raise FileNotFoundError(errno.ENOENT, 'exited')
    census.failure = missing
    with pytest.raises(processes.HistoricalProcessError, match='^historical_generation_process_unknown$') as raised:
        run(census)
    assert type(raised.value) is processes.HistoricalProcessError
    assert raised.value.census_attempt_reasons == tuple({
        'pid_open_disappeared': 1, 'descriptor_membership_changed': 0,
        'corroborated_process_exit': 0, 'pid_census_changed': 0,
    } for _ in range(3))
    assert len(census.scans) == 3


def test_exhaustion_evidence_attachment_failure_preserves_original_refusal(census, monkeypatch):
    def missing(pid):
        if pid == '2':
            raise FileNotFoundError(errno.ENOENT, 'exited')
    def refused_attribute(error, name, value):
        raise ValueError('projection unavailable')
    census.failure = missing
    monkeypatch.setattr(processes.HistoricalProcessError, '__setattr__', refused_attribute)
    with pytest.raises(processes.HistoricalProcessError, match='^historical_generation_process_unknown$'):
        run(census)
    assert len(census.scans) == 3
