"""Lossless fixture transport only; no provider, archive or native authority."""
# Covers: tests/historical_generation_fake_cloud.py
import base64
import hashlib
import json
import zlib

import pytest

from tests import historical_generation_fake_cloud as transport


def test_escaped_archive_wire_preserves_actual_bytes_inside_existing_log_bound():
    cloud = transport.Cloud()
    original = b'\\u0001' * 10000
    cloud.objects['actual-original-key'] = original
    cloud.metadata['actual-original-key'] = {'digest': 'original metadata'}
    cloud.calls = ['readback', 'client_closed']
    wire = transport.wire_state(cloud)
    restored = transport.decode_wire_state(wire)
    assert base64.b64decode(restored['objects']['actual-original-key'], validate=True) == original
    assert restored['metadata'] == cloud.metadata and restored['calls'] == cloud.calls


def test_actual_escaped_member_archive_round_trips_through_bounded_fixture_wire(tmp_path):
    from blueprint_pipeline.control_plane_lane_historical_archive import preserve
    from blueprint_pipeline.control_plane_lane_historical_generation import inventory_historical_generation
    from blueprint_pipeline.control_plane_lane_scratch_decisions import encode_validation_report
    from tests.test_historical_generation_archive import ReadOnlyMembers

    parent = tmp_path / 'work'
    target = parent / 'owned'
    # Keep the absolute local path below macOS PATH_MAX; the actual longest
    # four-component relative path remains required in disposable Linux.
    relative = '/'.join('\x01' * amount for amount in (255, 255, 255))
    nested = target / relative
    nested.mkdir(parents=True)
    (nested / 'f').write_bytes(b'actual original one')
    (nested / 'g').write_bytes(b'actual original two')
    (nested / 'h').write_bytes(b'actual original three')
    manifest = inventory_historical_generation(target, allowed_roots=(parent,))
    cloud = transport.Cloud()
    pointer = preserve(ReadOnlyMembers(target), manifest, encode_validation_report(manifest),
                       cloud, 'development-only', lambda: None)
    legacy = dict(objects={key: base64.b64encode(raw).decode() for key, raw in cloud.objects.items()},
                  metadata=cloud.metadata, calls=cloud.calls, corrupt=cloud.corrupt)
    assert len(json.dumps(legacy).encode()) > 32768
    restored = transport.decode_wire_state(transport.wire_state(cloud))
    key, original = next(iter(cloud.objects.items()))
    actual = base64.b64decode(restored['objects'][key], validate=True)
    assert actual == original and len(actual) == pointer['size_bytes']
    assert 'sha256:' + hashlib.sha256(actual).hexdigest() == pointer['readback_sha256']


@pytest.mark.parametrize('compressed', [
    zlib.compress(b'a' * 65537),
    zlib.compress(b'original') + zlib.compress(b'foreign'),
    zlib.compress(b'original')[:-1],
], ids=['expanded_overflow', 'foreign_suffix', 'truncated'])
def test_unknown_or_oversized_wire_never_replaces_fixture_object_bytes(compressed):
    wire = dict(corrupt=False, objects_zlib={'key': base64.b64encode(compressed).decode()},
                metadata={}, calls=[])
    with pytest.raises(AssertionError):
        transport.decode_wire_state(wire)
