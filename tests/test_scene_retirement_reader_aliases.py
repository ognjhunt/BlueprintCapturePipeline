"""ADP-009D/day28: physical reader aliases cannot rely on path spelling.

Portable proc metadata projections with actual payload inodes and symlinks.
These cases do not prove installed Linux mount or process clearance.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_supervisor.py
import os
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_retirement_supervisor as supervisor
from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance


@pytest.mark.parametrize('channel', ['cwd', 'fd', 'maps'])
def test_reader_alias_of_selected_physical_inode_is_kept(tmp_path, monkeypatch, channel):
    selected = tmp_path / 'selected'
    selected.mkdir()
    payload = selected / 'payload.bin'
    payload.write_bytes(b'actual-owned-development-only-payload')
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    proc = tmp_path / 'proc-projection'
    proc.mkdir()
    (proc / 'fd').mkdir()
    (proc / 'cwd').symlink_to(selected if channel == 'cwd' else unrelated, target_is_directory=True)
    if channel == 'fd':
        (proc / 'fd' / '3').symlink_to(payload)
    info = payload.stat()
    mapping = (f'1000-2000 r--p 00000000 {os.major(info.st_dev):x}:{os.minor(info.st_dev):x} '
               f'{info.st_ino} /unrelated-mounted-alias/payload.bin\n') if channel == 'maps' else ''
    (proc / 'maps').write_text(mapping)
    original_opened, original_readlink = supervisor._opened, os.readlink
    def opened(path, **kwargs):
        path = Path(path)
        if path.is_relative_to('/proc/54321'):
            path = proc / path.relative_to('/proc/54321')
        return original_opened(path, **kwargs)
    def readlink(name, **kwargs):
        actual = original_readlink(name, **kwargs)
        if str(name) == '3':
            # Explicit mount-path projection; actual stat follows the retained
            # fixture symlink to the selected real payload inode.
            return '/unrelated-mounted-alias/payload.bin'
        return actual
    monkeypatch.setattr(supervisor, '_opened', opened)
    monkeypatch.setattr(os, 'readlink', readlink)
    monkeypatch.setattr(supervisor, '_reader_view', lambda *args: None, raising=False)
    native = supervisor._NativeObservation(ActionAllowance(expires_at=3600, now=lambda: 100,
                                                           monotonic=lambda: 0))
    native.inodes = {(path.stat().st_dev, path.stat().st_ino) for path in (selected, payload)}
    row = dict(kernel_thread=False, cwd='/unrelated-mounted-alias')
    with pytest.raises(ValueError, match='scene_retirement_reader_closure_unproven'):
        supervisor._exposure(54321, row, [selected], native)
    assert payload.read_bytes() == b'actual-owned-development-only-payload'


def test_bounded_reader_inventory_records_actual_physical_members(tmp_path):
    root = tmp_path / 'selected'
    (root / 'nested').mkdir(parents=True)
    payload = root / 'nested/payload.bin'
    payload.write_bytes(b'owned')
    native = supervisor._NativeObservation(ActionAllowance(expires_at=3600, now=lambda: 100,
                                                           monotonic=lambda: 0))
    rows = supervisor._reader_inodes([root], native)
    assert set(rows) == {str(root), str(root / 'nested'), str(payload)}
    assert all(tuple(rows[str(path)][:2]) == (path.stat().st_dev, path.stat().st_ino)
               for path in (root, root / 'nested', payload))
    payload.write_bytes(b'changed')
    assert supervisor._reader_inodes([root], native) != rows
    assert native.entries > 0 and native.allowance.last_wall == 100


def test_reader_inventory_unknown_symlink_keeps(tmp_path):
    root = tmp_path / 'selected'
    root.mkdir()
    (root / 'unknown-alias').symlink_to(tmp_path)
    native = supervisor._NativeObservation(ActionAllowance(expires_at=3600, now=lambda: 100,
                                                           monotonic=lambda: 0))
    with pytest.raises(ValueError, match='scene_retirement_reader_closure_unproven'):
        supervisor._reader_inodes([root], native)


@pytest.mark.parametrize('view_kind', ['known', 'private_mount', 'subtree', 'hidden_device',
                                      'remapped_root', 'pid', 'user', 'changed_mount'])
def test_mount_route_projection_requires_actual_root_inode_and_complete_view(
        tmp_path, monkeypatch, view_kind):
    selected = tmp_path / 'selected'
    (selected / 'nested').mkdir(parents=True)
    payload = selected / 'nested/payload.bin'
    payload.write_bytes(b'actual-member')
    own, subject = tmp_path / 'own-proc', tmp_path / 'subject-proc'
    for path in (own, subject):
        path.mkdir()
        (path / 'root').symlink_to('/')
    info = selected.stat()
    device = f'{os.major(info.st_dev)}:{os.minor(info.st_dev)}'
    host = f'1 0 {device} / / rw - fixture fixture rw\n'
    (own / 'mountinfo').write_text(host)
    view = host
    if view_kind == 'subtree':
        view += f'2 1 {device} {selected / "nested"} /unknown-alias rw - fixture fixture rw\n'
    elif view_kind == 'hidden_device':
        view = f'1 0 {os.major(info.st_dev)}:{os.minor(info.st_dev) + 1} / / rw - fixture fixture rw\n'
    elif view_kind == 'remapped_root':
        (subject / 'root').unlink()
        (subject / 'root').symlink_to(tmp_path)
    (subject / 'mountinfo').write_text(view)
    original_opened = supervisor._opened
    def opened(path, **kwargs):
        path = Path(path)
        for pid, directory in ((os.getpid(), own), (54321, subject)):
            prefix = Path('/proc') / str(pid)
            if path.is_relative_to(prefix):
                path = directory / path.relative_to(prefix)
                break
        return original_opened(path, **kwargs)
    monkeypatch.setattr(supervisor, '_opened', opened)
    subject_ino = subject.stat().st_ino
    def namespace_projection(parent, native):
        is_subject = os.fstat(parent).st_ino == subject_ino
        values = [(f'{kind}:[{index}]', 999, index) for index, kind in enumerate(('pid', 'user', 'mnt'), 1)]
        if is_subject:
            index = {'pid': 0, 'user': 1, 'private_mount': 2}.get(view_kind)
            if index is not None:
                kind = ('pid', 'user', 'mnt')[index]
                values[index] = (f'{kind}:[99]', 999, 99)
        return tuple(values)
    monkeypatch.setattr(supervisor, '_reader_namespaces', namespace_projection)
    original_bytes = supervisor._native_bytes
    subject_reads = []
    def changed_bytes(path, native, **kwargs):
        value, identity = original_bytes(path, native, **kwargs)
        if Path(path) == Path('/proc/54321/mountinfo'):
            subject_reads.append(value)
            if view_kind == 'changed_mount' and len(subject_reads) > 1:
                value += b'\n'
        return value, identity
    monkeypatch.setattr(supervisor, '_native_bytes', changed_bytes)
    native = supervisor._NativeObservation(ActionAllowance(expires_at=3600, now=lambda: 100,
                                                           monotonic=lambda: 0))
    native.inventory = supervisor._reader_inodes([selected], native)
    native.inodes = {tuple(value[:2]) for value in native.inventory.values()}
    if view_kind in ('known', 'private_mount'):
        assert supervisor._reader_view(54321, [selected], native) == supervisor._reader_view(54321, [selected], native)
    else:
        with pytest.raises(ValueError, match='scene_retirement_reader_closure_unproven'):
            supervisor._reader_view(54321, [selected], native)
    assert payload.read_bytes() == b'actual-member'


@pytest.mark.parametrize('access_group', ['primary', 'effective', 'saved', 'filesystem', 'supplementary'])
def test_manual_foreign_uid_with_service_group_has_unclosed_access_lifetime(access_group):
    service_uid, service_gid = os.getuid(), os.getgid()
    foreign = {'uid': [service_uid + 10000] * 4, 'gid': [service_gid + 10000] * 4, 'groups': [],
               'status': dict(CapEff='0', CapPrm='0', CapInh='0', CapAmb='0', CapBnd='0', NoNewPrivs='1')}
    if access_group == 'supplementary':
        foreign['groups'] = [service_gid]
    else:
        foreign['gid'][('primary', 'effective', 'saved', 'filesystem').index(access_group)] = service_gid
    # There need not be any current FD or map: this identity can traverse750
    # members after the scan without an enrolled SH participant.
    foreign.update(kernel_thread=False, ppid=1)
    with pytest.raises(ValueError, match='scene_retirement_reader_closure_unproven'):
        supervisor._require_classified_readers({os.getpid() + 100000: foreign}, {}, set())


@pytest.mark.parametrize('groups', ['20 21', '', 'unknown', '4294967296', ' '.join(['20'] * 65), None])
@pytest.mark.parametrize('cap_field,cap_value', [(None, None), ('CapEff', 'xyz'),
    ('CapPrm', '-1'), ('CapInh', '0x4'), ('CapAmb', '1' * 17), ('CapBnd', '４'),
    ('NoNewPrivs', 'unknown')])
def test_proc_group_projection_is_complete_bounded_and_retained(tmp_path, monkeypatch, groups, cap_field, cap_value):
    proc = tmp_path / 'proc-group-projection'
    proc.mkdir()
    fields = ['S', '1', *(['0'] * 18)]
    fields[19] = '12345'
    (proc / 'stat').write_text('54321 (group-fixture) ' + ' '.join(fields) + '\n')
    status = dict(Uid='1000 1000 1000 1000', Gid='1001 1001 1001 1001', Threads='1',
                  CapEff='0', CapPrm='0', CapInh='0', CapAmb='0', CapBnd='0', NoNewPrivs='1')
    if cap_field is not None:
        status[cap_field] = cap_value
    if groups is not None:
        status['Groups'] = groups
    (proc / 'status').write_text(''.join(key + ': ' + value + '\n' for key, value in status.items()))
    (proc / 'cgroup').write_text('0::/fixture\n')
    (proc / 'cmdline').write_bytes(b'fixture\0')
    for kind in ('exe', 'cwd'):
        (proc / kind).symlink_to('/unrelated-projection')
    original_opened = supervisor._opened
    def opened(path, **kwargs):
        path = Path(path)
        if path.is_relative_to('/proc/54321'):
            path = proc / path.relative_to('/proc/54321')
        return original_opened(path, **kwargs)
    monkeypatch.setattr(supervisor, '_opened', opened)
    native = supervisor._NativeObservation(ActionAllowance(expires_at=3600, now=lambda: 100,
                                                           monotonic=lambda: 0))
    if groups in ('20 21', '') and cap_field is None:
        row = supervisor._proc_fields(54321, native)
        assert row['groups'] == ([20, 21] if groups else [])
        assert row['status']['Groups'] == groups
        assert native.entries == len(row['groups'])
    else:
        with pytest.raises(ValueError, match='scene_retirement_reader_closure_unproven'):
            supervisor._proc_fields(54321, native)


@pytest.mark.parametrize('field', ['CapEff', 'CapPrm', 'CapInh', 'CapAmb'])
@pytest.mark.parametrize('capability', [1, 2, 4, 64, 128, 256, 2**21])
def test_unknown_capable_reader_has_unclosed_access_lifetime(field, capability):
    status = dict(CapEff='0', CapPrm='0', CapInh='0', CapAmb='0', CapBnd='0', NoNewPrivs='1')
    status[field] = format(capability, '016x')
    row = dict(uid=[10001] * 4, gid=[10002] * 4, groups=[], status=status)
    # Actual capability-enabled access need not already have a current FD.
    # This projection pins admission logic; it is not native reader clearance.
    row.update(kernel_thread=False, ppid=1)
    with pytest.raises(ValueError, match='scene_retirement_reader_closure_unproven'):
        supervisor._require_classified_readers({os.getpid() + 100000: row}, {}, set())


@pytest.mark.parametrize('no_new_privs,bound', [('0', '4'), ('1', '4'), ('0', '0'), ('1', '0')])
def test_unknown_reader_cannot_acquire_executable_capabilities(no_new_privs, bound):
    status = dict(CapEff='0', CapPrm='0', CapInh='0', CapAmb='0', CapBnd=bound, NoNewPrivs=no_new_privs)
    row = dict(uid=[10001] * 4, gid=[10002] * 4, groups=[], status=status)
    row.update(kernel_thread=False, ppid=1)
    with pytest.raises(ValueError, match='scene_retirement_reader_closure_unproven'):
        supervisor._require_classified_readers({os.getpid() + 100000: row}, {}, set())


@pytest.mark.parametrize('route_mode,payload_mode,foreign_group', [(0o755, 0o644, False), (0o750, 0o640, True)])
def test_readable_actual_payload_does_not_close_unclassified_reader_lifetime(tmp_path, route_mode, payload_mode, foreign_group):
    route = tmp_path / 'selected-cache'
    route.mkdir(mode=route_mode)
    route.chmod(route_mode)
    payload = route / 'actual.payload'
    payload.write_bytes(b'actual-reproducible-cache')
    payload.chmod(payload_mode)
    birth = payload.stat()
    service_uid, service_gid = os.getuid(), os.getgid() + 10000
    # Explicit foreign identity projection; permission bytes are real and are
    # preserved. Absence of a current FD/map never proves future admission.
    row = dict(uid=[service_uid + 10000] * 4, gid=[birth.st_gid if foreign_group else service_gid + 1] * 4,
               groups=[], status=dict(CapEff='0', CapPrm='0', CapInh='0', CapAmb='0', CapBnd='0', NoNewPrivs='1'),
               kernel_thread=False, ppid=1)
    with pytest.raises(ValueError, match='scene_retirement_reader_closure_unproven'):
        supervisor._require_classified_readers({os.getpid() + 100000: row}, {}, set())
    assert payload.read_bytes() == b'actual-reproducible-cache'
    assert (payload.stat().st_ino, payload.stat().st_mode, payload.stat().st_gid) == (birth.st_ino, birth.st_mode, birth.st_gid)


@pytest.mark.parametrize('classification', ['caller', 'kernel', 'continuous', 'platform'])
def test_only_previously_authenticated_reader_classes_retain_exceptions(classification):
    # Classification sets are projections of the independently authenticated
    # upstream records. This only checks that the existing exceptions remain.
    pid = os.getpid() if classification == 'caller' else os.getpid() + 100000
    row = dict(kernel_thread=classification == 'kernel')
    supervisor._require_classified_readers({pid: row}, {pid} if classification == 'continuous' else {},
                                          {pid} if classification == 'platform' else set())
