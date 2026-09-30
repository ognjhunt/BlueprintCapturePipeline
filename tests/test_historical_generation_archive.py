"""Original historical archive bytes use bounded native multipart readback."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_archive.py
import hashlib
import io
import json
import os
from contextlib import contextmanager

import pytest

from tests.historical_generation_fake_cloud import Cloud


class ReadOnlyMembers:
    """Byte-format adapter; this supplies no native fence or delete authority."""
    def __init__(self, target):
        self.target = target

    @contextmanager
    def _opened(self, relative):
        fd = os.open(self.target / relative, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            yield fd, lambda: None
        finally:
            os.close(fd)


@pytest.fixture
def source(tmp_path):
    from blueprint_pipeline.control_plane_lane_historical_generation import inventory_historical_generation
    target = tmp_path / 'original'
    (target / 'nested').mkdir(parents=True)
    original = {'one': b'original diagnostics', 'nested/two': b'original nested bytes'}
    for name, raw in original.items():
        (target / name).write_bytes(raw)
    manifest = inventory_historical_generation(target, allowed_roots=(tmp_path,))
    raw = (json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()
    return ReadOnlyMembers(target), manifest, raw, original


def test_original_manifest_and_members_are_in_deterministic_stream(source):
    from blueprint_pipeline.control_plane_lane_historical_archive import MAGIC, _write_original
    held, manifest, raw, original = source
    output = io.BytesIO()
    _write_original(held, manifest, raw, output, lambda: None)
    expected = MAGIC + len(raw).to_bytes(4, 'big') + raw + b''.join(original[name] for name in sorted(original))
    assert output.getvalue() == expected


def test_native_transport_reads_back_every_original_byte_before_pointer(source):
    from blueprint_pipeline.control_plane_lane_historical_archive import preserve
    held, manifest, raw, original = source
    cloud = Cloud()
    pointer = preserve(held, manifest, raw, cloud, 'development-only', lambda: None)
    payload = next(iter(cloud.objects.values()))
    assert pointer['sha256'] == pointer['readback_sha256'] == 'sha256:' + hashlib.sha256(payload).hexdigest()
    assert pointer['size_bytes'] == pointer['readback_size_bytes'] == len(payload)
    assert pointer['full_byte_service_account_readback_passed'] is True
    assert cloud.calls.count('readback') == 1 and cloud.calls[-1] == 'client_closed'
    assert all((held.target / name).read_bytes() == value for name, value in original.items())


def test_corrupt_fake_readback_never_returns_a_preservation_pointer(source):
    from blueprint_pipeline.control_plane_lane_historical_archive import preserve
    held, manifest, raw, original = source
    cloud = Cloud(corrupt=True)
    with pytest.raises(ValueError, match='preservation_failed'):
        preserve(held, manifest, raw, cloud, 'development-only', lambda: None)
    assert cloud.calls[-1] == 'client_closed'
    assert all((held.target / name).read_bytes() == value for name, value in original.items())


def test_rewritten_member_refuses_stream_even_with_unchanged_size(source):
    from blueprint_pipeline.control_plane_lane_historical_archive import _write_original
    held, manifest, raw, original = source
    (held.target / 'one').write_bytes(b'x' * len(original['one']))
    with pytest.raises(ValueError, match='archive_payload_changed'):
        _write_original(held, manifest, raw, io.BytesIO(), lambda: None)


def test_authority_refusal_closes_transport_and_preserves_local_members(source):
    from blueprint_pipeline.control_plane_lane_historical_archive import preserve
    held, manifest, raw, original = source
    cloud = Cloud()
    def refused():
        raise ValueError('current_owner_expired')
    with pytest.raises(ValueError, match='preservation_failed'):
        preserve(held, manifest, raw, cloud, 'development-only', refused)
    assert cloud.calls == ['client_closed']
    assert all((held.target / name).read_bytes() == value for name, value in original.items())
