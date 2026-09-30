"""Known no-mm kernel tasks are distinct from unreadable user processes."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_kernel_process.py
import pytest
import time
from pathlib import Path


def records():
    fields = ['S', '0', '0', '0', '0', '0', str(0x00200000), *(['0'] * 12), '123']
    return dict(stat=b'2 (kernel worker) ' + ' '.join(fields).encode(),
                status=b'Name:\tkworker\nKthread:\t1\nUid:\t0 0 0 0\nGid:\t0 0 0 0\n',
                cmdline=b'', maps=b'')


def test_known_kernel_without_user_memory_requires_observed_empty_channels():
    from blueprint_pipeline.control_plane_kernel_process import kernel_has_no_user_memory
    values = records()
    assert kernel_has_no_user_memory(lambda name, cap: values[name], '2') is True


@pytest.mark.parametrize('change', ['user_task', 'cmdline', 'maps', 'memory', 'identity', 'unreadable'])
def test_flag_alone_or_missing_observations_cannot_classify_kernel_task(change):
    from blueprint_pipeline.control_plane_kernel_process import kernel_has_no_user_memory
    values = records()
    if change == 'user_task':
        values['stat'] = values['stat'].replace(str(0x00200000).encode(), b'0')
    elif change in ('cmdline', 'maps'):
        values[change] = b'user bytes'
    elif change == 'memory':
        values['status'] += b'VmSize:\t10 kB\n'
    calls = [0]
    def read(name, cap):
        if change == 'unreadable' and name == 'maps':
            raise PermissionError()
        if name == 'stat':
            calls[0] += 1
            if change == 'identity' and calls[0] > 1:
                return values[name].replace(b'123', b'124')
        return values[name]
    assert kernel_has_no_user_memory(read, '2') is False


@pytest.mark.parametrize('kernel', [True, False])
def test_census_kernel_classification_preserves_fd_references_and_unknown_users(tmp_path, monkeypatch, kernel):
    from blueprint_pipeline.control_plane_lane_scratch_census import _process_references
    target = tmp_path / 'selected'
    target.mkdir()
    (target / 'input.bin').write_bytes(b'original')
    proc = tmp_path / 'proc'
    (proc / '2/fd').mkdir(parents=True)
    values = records()
    if not kernel:
        values['stat'] = values['stat'].replace(str(0x00200000).encode(), b'0')
    for name, raw in values.items():
        (proc / '2' / name).write_bytes(raw)
    (proc / '2/environ').write_bytes(b'')
    (proc / '2/fd/3').symlink_to(target / 'input.bin')
    real = Path.open
    def open_channel(self, *args, **kwargs):
        if self == proc / '2/environ':
            raise ProcessLookupError(3, 'kernel no mm')
        return real(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_channel)
    errors = []
    assert _process_references([target], proc, errors, time.monotonic() + 5) == {target}
    assert errors == ([] if kernel else ['process_inventory_unreadable'])
