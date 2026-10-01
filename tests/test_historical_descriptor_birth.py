"""ADP-009D/day28: pure bounded Linux dirent64 framing negatives.

These byte projections are not kernel procfs, descriptor birth or action proof.
"""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_descriptor_birth.py
import sys

import pytest

from blueprint_pipeline.control_plane_lane_historical_descriptor_birth import _names


def entry(name, kind=10):
    raw = name + b'\0'
    size = (19 + len(raw) + 7) // 8 * 8
    return (b'\x01' + b'\0' * 15 + size.to_bytes(2, sys.byteorder)
            + bytes([kind]) + raw + b'\0' * (size - 19 - len(raw)))


def test_bounded_actual_abi_layout_supports_stdio_and_retained_inventory_slot():
    assert _names(entry(b'.', 4) + entry(b'..', 4) + entry(b'0') + entry(b'7')) == [0, 7]


@pytest.mark.parametrize('raw', [b'', b'a' * 8193, b'a' * 23, entry(b'7')[:-1],
    entry(b'07'), entry(b'-1'), entry(b'x'), entry(b'21474836480'), entry(b'7', 4),
    entry(b'7')[:16] + b'\0\0' + entry(b'7')[18:],
    entry(b'7')[:19] + b'a' * (len(entry(b'7')) - 19)])
def test_unknown_or_truncated_kernel_frames_never_prove_a_slot(raw):
    with pytest.raises(ValueError, match='descriptor_namespace_unknown'):
        _names(raw)
