"""The scored matrix: 200 hidden cells inside the task's variation bounds, the same for every team on a job."""

from __future__ import annotations

import pytest

from blueprint_pipeline.team_evaluation_plan import (
    TeamEvaluationPlanError,
    scored_cells,
    shard_ttl_seconds,
    shards,
    variation_bounds,
    campaign_ttl_seconds,
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


@pytest.mark.parametrize("control_hz", [15, 30, 60, 120])
def test_full_campaign_lease_covers_the_planned_work_at_the_selected_rate(control_hz):
    assert campaign_ttl_seconds(control_hz=control_hz) == shard_ttl_seconds(200, control_hz=control_hz)


def test_no_varied_start_lies_inside_the_scored_area_and_no_try_is_dropped():
    """Stage 8 2d: a large area ("the other end of the table") reaching back toward the start."""
    from blueprint_pipeline import team_evaluation_plan as plan
    from blueprint_pipeline.varied_start_clearance import START_CLEARANCE_M, start_clearance_m, uncleared_cells

    # 447's start and its sealed 10 cm box: every start is about 0.4 m clear today.
    spec = {"start_pose_world": [-0.4326, -0.3031, 0.8143, 0, 0, 0, 1],
            "destination_position_bounds_world_m": {"minimum": [-0.2777, 0.0498, 0.8093],
                                                    "maximum": [-0.1777, 0.1498, 0.8193]},
            "destination_orientation_tolerance_rad": 0.08}
    cells = plan.scored_cells(scene_revision_digest="sha256:" + "1" * 64, scene_id="s", salt="a" * 64,
                              task_spec=spec, task_kind="rigid_pick_place", count=200)
    assert len(cells) == 200 and uncleared_cells(spec, cells) == []
    # An area that ends 2 cm past the start's x: without the re-draw some shifted starts fall inside it.
    near = {**spec, "destination_position_bounds_world_m": {"minimum": [-0.41, -0.33, 0.80], "maximum": [0.40, 0.60, 0.83]}}
    cells = plan.scored_cells(scene_revision_digest="sha256:" + "1" * 64, scene_id="s", salt="a" * 64,
                              task_spec=near, task_kind="rigid_pick_place", count=200)
    assert len(cells) == 200  # never dropped
    assert all(start_clearance_m(near, cell["resolved_scenario"]["parameters"]) >= START_CLEARANCE_M for cell in cells)


def test_an_area_reaching_over_the_start_is_shrunk_about_its_target_and_the_plan_is_flagged_not_refused():
    """Stage 8 2d, owner 2026-11-12: never a new stop. Shrink (never below the 5 cm disc), record, flag, plan."""
    from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
    from blueprint_pipeline.varied_start_clearance import (
        DESTINATION_NEAR_START, MINIMUM_HALF_WIDTH_M, START_CLEARANCE_M, shrink_area_off_start, start_clearance_m,
    )
    from tests.test_beta_policy_screening_plan import NOW, inputs, scene

    spec = {"start_pose_world": [0.50, 0.00, 0.8, 0, 0, 0, 1], "target_position_world_m": [0.75, 0.0, 0.8],
            "destination_position_bounds_world_m": {"minimum": [0.45, -0.3, 0.79], "maximum": [1.05, 0.3, 0.81]}}
    shrunk, record = shrink_area_off_start(spec)
    assert record["start_clear"] is True and start_clearance_m(shrunk, {}) >= START_CLEARANCE_M
    assert min(record["half_width_m"]) >= MINIMUM_HALF_WIDTH_M and record["flag"] == DESTINATION_NEAR_START
    # A target 3 cm from the start: the 5 cm disc is kept, the start is still not clear, and it is said.
    near = {**spec, "target_position_world_m": [0.53, 0.0, 0.8]}
    kept, still = shrink_area_off_start(near)
    assert still["half_width_m"] == [MINIMUM_HALF_WIDTH_M, MINIMUM_HALF_WIDTH_M] and still["start_clear"] is False
    # The screening plan is built and carries the flag (before: beta_screening_varied_start_inside_scored_area).
    catalog, capabilities, quotes = inputs()
    value = scene()
    value["task_spec"] = {**value["task_spec"], "start_pose_world": [0.55, 0.30, 0.8, 0, 0, 0, 1]}
    plan = build_screening_plan(catalog, value, campaign_id="campaign1", capabilities=capabilities,
                                cost_quotes=quotes, now=NOW)
    assert DESTINATION_NEAR_START in plan["scene_estimate_flags"]


def test_the_area_guard_never_grows_an_area_smaller_than_the_disc():
    """Review 36 probe_shrink: a 4 cm box used to become 10 cm and contain the start."""
    from blueprint_pipeline.varied_start_clearance import shrink_area_off_start

    spec = {"start_pose_world": [0.0, 0.0, 0.8], "target_position_world_m": [0.035, 0.0, 0.8],
            "destination_position_bounds_world_m": {"minimum": [0.015, -0.02, 0.78], "maximum": [0.055, 0.02, 0.85]}}
    out, record = shrink_area_off_start(spec)
    assert out["destination_position_bounds_world_m"] == spec["destination_position_bounds_world_m"]
    assert record["start_clear"] is False and record["flag"] == "scene_estimate:destination_near_start"
