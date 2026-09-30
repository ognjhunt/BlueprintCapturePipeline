"""ADP-009D/day28: original archive extraction into a private no-replace stage."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_restore_archive.py
import os
from contextlib import contextmanager

import pytest

from tests.historical_generation_fake_cloud import Cloud
from tests.test_historical_generation_archive import source  # noqa: F401


class PrivateStage:
    """Actual tiny files; no native authority, publication or access claim."""
    def __init__(self, root):
        self.root = root
        root.mkdir(mode=0o700)

    def directory(self, row):
        if row['path']:
            (self.root / row['path']).mkdir(mode=0o700)

    @contextmanager
    def member(self, row):
        fd = os.open(self.root / row['path'], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb', buffering=0) as output:
            yield output


def prepared(source):  # noqa: F811
    from blueprint_pipeline.control_plane_lane_historical_archive import preserve
    held, manifest, raw, original = source
    cloud = Cloud()
    pointer = preserve(held, manifest, raw, cloud, 'development-only', lambda: None)
    current = Cloud()
    current.objects, current.metadata = dict(cloud.objects), dict(cloud.metadata)
    return manifest, raw, original, pointer, current


def test_extracts_every_original_byte_without_republishing_cloud(source, tmp_path):  # noqa: F811
    from blueprint_pipeline.control_plane_lane_historical_restore_archive import extract_preserved_members
    manifest, raw, original, pointer, cloud = prepared(source)
    stage = PrivateStage(tmp_path / 'stage')
    result = extract_preserved_members(manifest, raw, pointer, cloud, 'development-only', stage, lambda: None)
    assert result == dict(archive_sha256=pointer['sha256'], archive_size_bytes=pointer['size_bytes'],
                         restored_files=2, restored_logical_bytes=sum(map(len, original.values())))
    assert all((stage.root / name).read_bytes() == value for name, value in original.items())
    assert cloud.calls == ['head', 'readback', 'head', 'client_closed']
    assert all(body.closed for body in cloud.bodies)


@pytest.mark.parametrize('change', ['member', 'prefix', 'suffix', 'missing', 'owner_refused'])
def test_changed_archive_or_owner_never_reports_complete_stage(source, tmp_path, change):  # noqa: F811
    from blueprint_pipeline.control_plane_lane_historical_restore_archive import extract_preserved_members
    manifest, raw, _, pointer, cloud = prepared(source)
    key = next(iter(cloud.objects))
    if change == 'member':
        cloud.objects[key] = cloud.objects[key][:-1] + b'x'
    elif change == 'prefix':
        cloud.objects[key] = b'x' + cloud.objects[key][1:]
    elif change == 'suffix':
        cloud.corrupt = True
    elif change == 'missing':
        cloud.objects.clear()
    def guard():
        if change == 'owner_refused':
            raise ValueError('current_owner_refused')
    with pytest.raises(ValueError):
        extract_preserved_members(manifest, raw, pointer, cloud, 'development-only',
                                 PrivateStage(tmp_path / 'stage'), guard)
    assert cloud.calls[-1] == 'client_closed' and all(body.closed for body in cloud.bodies)


def test_existing_private_member_is_never_overwritten(source, tmp_path):  # noqa: F811
    from blueprint_pipeline.control_plane_lane_historical_restore_archive import extract_preserved_members
    manifest, raw, _, pointer, cloud = prepared(source)
    stage = PrivateStage(tmp_path / 'stage')
    stage.directory(dict(path='nested'))
    sentinel = stage.root / 'nested/two'
    sentinel.write_bytes(b'keep existing destination')
    with pytest.raises((ValueError, OSError)):
        extract_preserved_members(manifest, raw, pointer, cloud, 'development-only', stage, lambda: None)
    assert sentinel.read_bytes() == b'keep existing destination' and cloud.calls[-1] == 'client_closed'
