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
