"""ADP release coordination cannot escape a retargeted workspace ancestry."""
import errno
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_workspace_lock as locks


def _layout(tmp_path):
    parent = tmp_path / 'workspaces'
    parent.mkdir(mode=0o770)
    directory = parent / '.workspace-locks'
    directory.mkdir(mode=0o770)
    directory.chmod(0o770)
    outside = tmp_path / 'outside'
    outside.mkdir()
    return parent / 'victim', directory, outside


def _track_opens(monkeypatch, before=None):
    original = locks.os.open
    descriptors = []

    def observed(path, flags, mode=0o777, *, dir_fd=None):
        if before:
            before(path, flags, dir_fd)
        descriptor = original(path, flags, mode, dir_fd=dir_fd)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(locks.os, 'open', observed)
    return descriptors


def _all_closed(descriptors):
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


def test_canonical_lock_modes_and_consumer_exception_preserve_producer_proof(tmp_path, monkeypatch):
    workspace = tmp_path / 'new-parent' / 'job'
    descriptors = _track_opens(monkeypatch)
    with pytest.raises(RuntimeError, match='consumer failed'):
        with locks.workspace_lock(workspace) as acquired:
            assert acquired
            raise RuntimeError('consumer failed')
    directory = workspace.parent / '.workspace-locks'
    path = directory / 'job.lock'
    assert path.is_file()
    assert stat.S_IMODE(directory.stat().st_mode) == 0o770
    assert stat.S_IMODE(path.stat().st_mode) == 0o660
    assert path.stat().st_gid == workspace.parent.stat().st_gid
    _all_closed(descriptors)


def test_reclaim_never_creates_missing_producer_proof(tmp_path):
    workspace = tmp_path / 'absent' / 'nested' / 'job'
    with locks.workspace_lock(workspace, reclaim=True) as acquired:
        assert not acquired
    assert not (tmp_path / 'absent').exists()
    workspace.parent.mkdir(parents=True)
    with locks.workspace_lock(workspace, reclaim=True) as acquired:
        assert not acquired
    assert not (workspace.parent / '.workspace-locks').exists()
    (workspace.parent / '.workspace-locks').mkdir(mode=0o770)
    with locks.workspace_lock(workspace, reclaim=True) as acquired:
        assert not acquired
    assert list((workspace.parent / '.workspace-locks').iterdir()) == []


def test_shared_producers_block_nonblocking_exclusive_reclamation(tmp_path):
    workspace = tmp_path / 'job'
    with locks.workspace_lock(workspace) as first:
        with locks.workspace_lock(workspace) as second:
            assert first and second
            with locks.workspace_lock(workspace, reclaim=True) as acquired:
                assert not acquired
    with locks.workspace_lock(workspace, reclaim=True) as acquired:
        assert acquired


@pytest.mark.parametrize('target', ['ancestor', 'directory', 'leaf', 'hardlink', 'fifo'])
def test_symlinks_hardlinks_and_nonregular_locks_are_refused(tmp_path, target):
    workspace, directory, outside = _layout(tmp_path)
    sentinel = outside / 'sentinel'
    sentinel.write_text('unchanged')
    if target == 'ancestor':
        alias = tmp_path / 'alias'
        alias.symlink_to(workspace.parent, target_is_directory=True)
        workspace = alias / workspace.name
    elif target == 'directory':
        directory.rmdir()
        directory.symlink_to(outside, target_is_directory=True)
    elif target == 'leaf':
        (directory / 'victim.lock').symlink_to(sentinel)
    elif target == 'hardlink':
        sentinel.chmod(0o660)
        os.link(sentinel, directory / 'victim.lock')
    else:
        os.mkfifo(directory / 'victim.lock', 0o660)
    with pytest.raises((OSError, ValueError)):
        with locks.workspace_lock(workspace):
            pytest.fail('unsafe lock was admitted')
    assert sentinel.read_text() == 'unchanged'
    assert not (outside / 'victim.lock').exists()


@pytest.mark.parametrize('target', ['ancestor', 'directory'])
def test_directory_retarget_during_relative_creation_leaves_no_escaped_lock(tmp_path, monkeypatch, target):
    workspace, directory, outside = _layout(tmp_path)
    if target == 'ancestor':
        (outside / '.workspace-locks').mkdir(mode=0o770)
    moved = tmp_path / 'original'
    changed = False

    def retarget(path, flags, dir_fd):
        nonlocal changed
        if Path(path).name == 'victim.lock' and flags & os.O_CREAT and not changed:
            changed = True
            victim = workspace.parent if target == 'ancestor' else directory
            victim.rename(moved)
            victim.symlink_to(outside, target_is_directory=True)

    descriptors = _track_opens(monkeypatch, retarget)
    with pytest.raises(ValueError, match='directory_retargeted'):
        with locks.workspace_lock(workspace):
            pytest.fail('retargeted ancestry was admitted')
    assert changed
    redirected_directory = outside / '.workspace-locks' if target == 'ancestor' else outside
    assert list(redirected_directory.iterdir()) == []
    original_directory = moved / '.workspace-locks' if target == 'ancestor' else moved
    assert (original_directory / 'victim.lock').is_file()
    _all_closed(descriptors)


def test_retarget_after_flock_is_refused_before_handing_lock_to_consumer(tmp_path, monkeypatch):
    workspace, directory, outside = _layout(tmp_path)
    moved = tmp_path / 'original'
    original = locks.fcntl.flock
    descriptors = _track_opens(monkeypatch)

    def retarget(descriptor, operation):
        original(descriptor, operation)
        directory.rename(moved)
        directory.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(locks.fcntl, 'flock', retarget)
    with pytest.raises(ValueError, match='directory_retargeted'):
        with locks.workspace_lock(workspace):
            pytest.fail('retargeted lock was handed to consumer')
    assert list(outside.iterdir()) == []
    assert (moved / 'victim.lock').is_file()
    _all_closed(descriptors)


def test_replaced_existing_lock_is_refused_without_unlinking_either_inode(tmp_path, monkeypatch):
    workspace, directory, _ = _layout(tmp_path)
    path = directory / 'victim.lock'
    path.touch(mode=0o660)
    original = locks.fcntl.flock

    def replace(descriptor, operation):
        original(descriptor, operation)
        path.rename(directory / 'original.lock')
        path.write_text('replacement')
        path.chmod(0o660)

    monkeypatch.setattr(locks.fcntl, 'flock', replace)
    with pytest.raises(ValueError, match='file_retargeted'):
        with locks.workspace_lock(workspace):
            pytest.fail('replacement lock was admitted')
    assert (directory / 'original.lock').exists()
    assert path.read_text() == 'replacement'


@pytest.mark.parametrize('target', ['directory', 'file', 'ancestor'])
def test_unsafe_installed_permissions_are_refused_without_repair(tmp_path, target):
    workspace, directory, _ = _layout(tmp_path)
    path = directory / 'victim.lock'
    path.touch(mode=0o660)
    victim = directory if target == 'directory' else path if target == 'file' else workspace.parent
    mode = 0o777 if target != 'file' else 0o666
    victim.chmod(mode)
    with pytest.raises(ValueError, match='unsafe'):
        with locks.workspace_lock(workspace):
            pytest.fail('unsafe permissions were admitted')
    assert stat.S_IMODE(victim.stat().st_mode) == mode


def test_acquisition_failure_preserves_coordination_inode_and_closes_descriptors(tmp_path, monkeypatch):
    workspace, directory, _ = _layout(tmp_path)
    descriptors = _track_opens(monkeypatch)

    def fail(descriptor, mode):
        raise PermissionError('fixture refuses mode repair')

    monkeypatch.setattr(locks.os, 'fchmod', fail)
    with pytest.raises(PermissionError, match='fixture refuses'):
        with locks.workspace_lock(workspace):
            pytest.fail('failed acquisition was admitted')
    assert (directory / 'victim.lock').is_file()
    _all_closed(descriptors)


def test_failed_creator_cannot_unlink_a_concurrent_consumers_lock(tmp_path, monkeypatch):
    workspace, directory, _ = _layout(tmp_path)
    consumer = locks.workspace_lock(workspace)
    original = locks.os.fchmod
    entered = False

    def fail_after_consumer_entered(descriptor, mode):
        nonlocal entered
        assert consumer.__enter__()
        entered = True
        raise PermissionError('creator failed after another consumer acquired')

    monkeypatch.setattr(locks.os, 'fchmod', fail_after_consumer_entered)
    try:
        with pytest.raises(PermissionError, match='creator failed'):
            with locks.workspace_lock(workspace):
                pytest.fail('failed creator was admitted')
        monkeypatch.setattr(locks.os, 'fchmod', original)
        assert (directory / 'victim.lock').is_file()
        # A later writer must use that same inode; it must not create a second
        # inode that an exclusive reclaimer can acquire while this consumer lives.
        with locks.workspace_lock(workspace):
            pass
        with locks.workspace_lock(workspace, reclaim=True) as acquired:
            assert not acquired
    finally:
        if entered:
            consumer.__exit__(None, None, None)
    with locks.workspace_lock(workspace, reclaim=True) as acquired:
        assert acquired


def test_permissions_changed_after_flock_refuse_producer_proof(tmp_path, monkeypatch):
    workspace, directory, _ = _layout(tmp_path)
    original = locks.fcntl.flock

    def change_mode(descriptor, operation):
        original(descriptor, operation)
        os.fchmod(descriptor, 0o666)

    monkeypatch.setattr(locks.fcntl, 'flock', change_mode)
    with pytest.raises(ValueError, match='object_unsafe'):
        with locks.workspace_lock(workspace):
            pytest.fail('unsafe post-flock mode was admitted')
    assert stat.S_IMODE((directory / 'victim.lock').stat().st_mode) == 0o666


def test_hardlink_inserted_between_post_flock_descriptor_and_name_checks_is_refused(tmp_path, monkeypatch):
    workspace, directory, _ = _layout(tmp_path)
    original_flock, original_stat = locks.fcntl.flock, locks.os.stat
    armed = False

    def arm(descriptor, operation):
        nonlocal armed
        original_flock(descriptor, operation)
        armed = True

    def insert_link(path, *args, **kwargs):
        nonlocal armed
        if armed and path == 'victim.lock' and kwargs.get('dir_fd') is not None:
            armed = False
            os.link(directory / 'victim.lock', directory / 'another.lock')
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(locks.fcntl, 'flock', arm)
    monkeypatch.setattr(locks.os, 'stat', insert_link)
    with pytest.raises(ValueError, match='file_retargeted'):
        with locks.workspace_lock(workspace):
            pytest.fail('post-flock hardlink was admitted')
    assert (directory / 'victim.lock').stat().st_nlink == 2


def test_runtime_group_owner_remains_admitted_for_root_reclamation(monkeypatch):
    parent = SimpleNamespace(st_uid=0, st_gid=123)
    monkeypatch.setattr(locks.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(locks.pwd, 'getpwuid', lambda uid: SimpleNamespace(pw_name='runtime', pw_gid=123))
    assert locks._shared_owner(456, parent)
    monkeypatch.setattr(locks.pwd, 'getpwuid', lambda uid: SimpleNamespace(pw_name='runtime', pw_gid=789))
    monkeypatch.setattr(locks.grp, 'getgrgid', lambda gid: SimpleNamespace(gr_mem=['runtime']))
    assert locks._shared_owner(456, parent)
    monkeypatch.setattr(locks.grp, 'getgrgid', lambda gid: SimpleNamespace(gr_mem=[]))
    assert not locks._shared_owner(456, parent)
