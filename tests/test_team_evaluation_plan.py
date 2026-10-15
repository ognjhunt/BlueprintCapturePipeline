"""The scored matrix: 200 hidden cells inside the task's variation bounds, the same for every team on a job."""

from __future__ import annotations

import pytest

from blueprint_pipeline.team_evaluation_plan import (
    TeamEvaluationPlanError,
    scored_cells,
    shard_ttl_seconds,
    shards,
    variation_bounds,
)

SPEC = {"task_kind": "rigid_pick_place", "start_pose_world": [0.5, 0.0, 0.8, 0, 0, 0, 1],
        "destination_position_bounds_world_m": {"minimum": [0.5, 0.25, 0.79], "maximum": [0.6, 0.35, 0.81]},
        "destination_orientation_tolerance_rad": 0.5}
REVISION = "sha256:" + "1" * 64


def _cells(salt: str = "a" * 64) -> list[dict]:
    return scored_cells(scene_revision_digest=REVISION, scene_id="site-req1", salt=salt, task_spec=SPEC,
                        task_kind="rigid_pick_place", nominal_dynamic_friction=0.6)


def test_scored_cells_cover_every_family_inside_the_task_bounds() -> None:
    cells = _cells()
    bounds = variation_bounds(SPEC, task_kind="rigid_pick_place", nominal_dynamic_friction=0.6)
    assert len(cells) == 200 and len({cell["cell_id"] for cell in cells}) == 200
    assert len({cell["seed"] for cell in cells}) == 200
    assert {cell["family"] for cell in cells} == {
        "canonical_anchor", "placement_approach", "illumination", "camera_sensor", "bounded_physics",
        "pairwise_stress", "held_out_composition"}
    for cell in cells:
        for name, value in cell["resolved_scenario"]["parameters"].items():
            low, high = bounds[name]
            assert low - 1e-9 <= value <= high + 1e-9, (name, value)
    held_out = [cell for cell in cells if cell["partition"] == "held_out"]
    assert len(held_out) >= 40 and all(set(cell["resolved_scenario"]["parameters"]) == set(bounds)
                                       for cell in held_out)
    # Placement varies both table axes and yaw, spread across each range rather than bunched.
    xs = sorted(cell["resolved_scenario"]["parameters"]["object_start_x_delta_m"] for cell in cells
                if cell["family"] == "placement_approach")
    assert xs[0] < bounds["object_start_x_delta_m"][0] / 2 and xs[-1] > bounds["object_start_x_delta_m"][1] / 2


def test_every_team_on_a_job_faces_the_same_hidden_draws_and_the_salt_hides_them() -> None:
    assert _cells() == _cells()
    assert [cell["resolved_scenario"] for cell in _cells()] != [cell["resolved_scenario"]
                                                               for cell in _cells("b" * 64)]
    with pytest.raises(TeamEvaluationPlanError):
        _cells("short")


def test_a_scored_run_is_eight_shards_that_each_fit_their_time_limit() -> None:
    groups = shards(_cells())
    assert [len(group) for group in groups] == [25] * 8
    assert shard_ttl_seconds(25) <= 4 * 3600
