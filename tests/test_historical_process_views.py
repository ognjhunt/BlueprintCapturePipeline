"""A namespace label alone never proves the selected access fence."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_processes.py
import os
from pathlib import Path

import pytest


def mounts(extra=b''):
    return b'31 1 8:1 / / ro - ext4 /dev/test ro\n' + extra


def test_native_target_bind_derives_original_physical_path_and_every_alias():
    from blueprint_pipeline.control_plane_lane_historical_processes import _physical_target, _view_routes
    device = os.makedev(8, 1)
    target = Path('/var/lib/selected')
    own = mounts(b'32 31 8:1 /var/lib/selected /var/lib/selected rw - ext4 /dev/test rw\n')
    physical = _physical_target(own, target, device)
    assert physical == target
    view = mounts(b'33 31 8:1 /var/lib /alternative ro - ext4 /dev/test ro\n')
    assert _view_routes(view, physical, device) == (target, Path('/alternative/selected'))


@pytest.mark.parametrize('extra', [
    b'33 31 8:1 /var/lib/selected/nested /other rw - ext4 /dev/test rw\n',
    b'33 31 8:1 /var/lib/selected /other rw - ext4 /dev/test rw\n',
    b'not a mount observation\n',
    b'33 31 8:1 /var/lib/../selected /other rw - ext4 /dev/test rw\n',
])
def test_unknown_or_target_subtree_alias_is_not_a_known_fenced_view(extra):
    from blueprint_pipeline.control_plane_lane_historical_processes import _view_routes
    with pytest.raises(ValueError, match='process_view_unknown'):
        _view_routes(mounts(extra), Path('/var/lib/selected'), os.makedev(8, 1))


def test_isolated_kernel_view_requires_complete_device_disjointness():
    from blueprint_pipeline.control_plane_lane_historical_processes import _kernel_view_disjoint
    selected = {os.makedev(8, 1)}
    devtmpfs = b'45 1 0:7 / / rw - devtmpfs devtmpfs rw\n'
    assert _kernel_view_disjoint(devtmpfs, selected) is True
    assert _kernel_view_disjoint(devtmpfs + mounts(), selected) is False
    with pytest.raises(ValueError, match='process_view_unknown'):
        _kernel_view_disjoint(b'not mountinfo\n', selected)


def test_disjoint_kernel_mounts_still_detect_an_actual_retained_fd(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_processes as processes
    selected = tmp_path / 'selected'
    selected.mkdir()
    payload = selected / 'original.bin'
    payload.write_bytes(b'original')
    root = tmp_path / 'isolated-root'
    root.mkdir()
    process = tmp_path / 'process'
    (process / 'fd').mkdir(parents=True)
    fields = ['S', '0', '0', '0', '0', '0', str(0x00200000), *(['0'] * 12), '123']
    values = dict(stat=b'2 (kernel worker) ' + ' '.join(fields).encode(),
        status=b'Name:\tkworker\nKthread:\t1\nUid:\t0 0 0 0\nGid:\t0 0 0 0\n',
        cmdline=b'', maps=b'')
    for name, raw in values.items():
        (process / name).write_bytes(raw)
    (process / 'environ').write_bytes(b'')
    (process / 'mountinfo').write_bytes(b'45 1 0:7 / / rw - devtmpfs devtmpfs rw\n')
    (process / 'root').symlink_to(root)
    (process / 'cwd').symlink_to(root)
    (process / 'fd/3').symlink_to(payload)
    monkeypatch.setattr(processes, '_namespace', lambda directory, kernel=False:
                        ('pid:[1]', 'user:[1]', 'mnt:[7]'))
    info = payload.stat()
    descriptor = os.open(process, os.O_RDONLY | os.O_DIRECTORY)
    try:
        channels = processes._inspect_process(processes._Scan(lambda: None), descriptor, '2',
            str(selected), {(info.st_dev, info.st_ino)}, ('pid:[1]', 'user:[1]', 'mnt:[1]'),
            'mnt:[1]', (0, 1))
    finally:
        os.close(descriptor)
    assert channels == {'fd'}
