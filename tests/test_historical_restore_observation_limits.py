"""ADP-009D/day28: original admission must fit its bounded restore partition.

These are actual tiny filesystem observations, not owner/native authority.
"""
from pathlib import Path

import pytest

from blueprint_pipeline.control_plane_lane_historical_generation import inventory_historical_generation
from blueprint_pipeline.control_plane_lane_historical_fence import _members


def staged(tmp_path, *, depth):
    target = tmp_path / 'original'
    target.mkdir()
    leaf = target
    for number in range(depth):
        leaf = leaf / str(number)
        leaf.mkdir()
    (leaf / 'last.bin').write_bytes(b'original bytes')
    original = inventory_historical_generation(target, allowed_roots=(tmp_path,))
    action_id = 'a' * 32
    stage = target / ('.historical-restore-' + action_id)
    stage.mkdir()
    for child in list(target.iterdir()):
        if child != stage:
            child.rename(stage / child.name)
    return target, stage, original, action_id


@pytest.mark.parametrize('depth', [0, 16])
def test_stage_observation_preserves_admitted_rows_and_maximum_depth(tmp_path, depth):
    from blueprint_pipeline.control_plane_lane_historical_restore_limits import RestoreObservationBounds
    target, stage, original, action_id = staged(tmp_path, depth=depth)
    bounds = RestoreObservationBounds(original, action_id)
    observed = inventory_historical_generation(target, allowed_roots=(tmp_path,), _restore_bounds=bounds)
    assert observed['member_count'] == original['member_count'] + 1
    assert observed['logical_payload_bytes'] == original['logical_payload_bytes']
    assert (stage / Path(*map(str, range(depth))) / 'last.bin').read_bytes() == b'original bytes'
    assert len(_members(observed, restore_bounds=bounds)) == original['member_count'] + 1


@pytest.mark.parametrize('change', ['unknown', 'wrong_stage', 'wrong_kind'])
def test_internal_allowance_never_admits_an_unrelated_row(tmp_path, change):
    from blueprint_pipeline.control_plane_lane_historical_restore_limits import RestoreObservationBounds
    target, stage, original, action_id = staged(tmp_path, depth=0)
    bounds = RestoreObservationBounds(original, action_id)
    if change == 'unknown':
        (stage / 'unrelated').write_bytes(b'')
    elif change == 'wrong_stage':
        stage.rename(target / ('.historical-restore-' + 'b' * 32))
    else:
        (stage / 'last.bin').unlink()
        (stage / 'last.bin').mkdir()
    with pytest.raises(ValueError):
        inventory_historical_generation(target, allowed_roots=(tmp_path,), _restore_bounds=bounds)


def test_private_allowance_cannot_expand_public_original_admission(tmp_path):
    target, _, original, action_id = staged(tmp_path, depth=16)
    with pytest.raises(ValueError, match='historical_generation_member_unsupported'):
        inventory_historical_generation(target, allowed_roots=(tmp_path,))
    from blueprint_pipeline.control_plane_lane_historical_restore_limits import RestoreObservationBounds
    with pytest.raises(ValueError):
        RestoreObservationBounds(original, 'invalid-action')
