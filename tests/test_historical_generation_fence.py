"""Historical action fence refusals; actual root effects are tested on Linux."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_fence.py
import os
import sys

import pytest


def test_ordinary_process_cannot_revoke_historical_access(tmp_path):
    from blueprint_pipeline.control_plane_lane_historical_fence import _HistoricalGenerationFence
    target = tmp_path / 'original'
    target.mkdir()
    payload = target / 'keep.bin'
    payload.write_bytes(b'owner bytes')
    with pytest.raises(ValueError, match='native_unavailable'):
        _HistoricalGenerationFence({'target_path': str(target)}, tick=lambda: None)
    assert payload.read_bytes() == b'owner bytes'
    assert payload.stat().st_uid == os.geteuid()


@pytest.mark.parametrize('relative', ['../outside', '/outside', 'nested//file', 'nested/./file'])
def test_fence_rejects_unsupported_manifest_paths_before_open(relative):
    from blueprint_pipeline.control_plane_lane_historical_fence import _members
    with pytest.raises(ValueError, match='manifest_invalid'):
        _members({'member_count': 1, 'members': [dict(path=relative, kind='file', version=[0] * 10)]})


def test_fence_requires_bounded_original_tree_not_path_only():
    from blueprint_pipeline.control_plane_lane_historical_fence import _members
    with pytest.raises(ValueError, match='manifest_invalid'):
        _members({'member_count': 0, 'members': []})


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux-only ACL descriptor observation')
def test_unknown_acl_is_a_refusal(tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_lane_historical_fence import _acl
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(os, 'listxattr', lambda selected: ['system.posix_acl_default'])
        with pytest.raises(ValueError, match='acl_unknown'):
            _acl(fd)
    finally:
        os.close(fd)
