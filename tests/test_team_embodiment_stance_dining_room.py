"""A fixed arm's pedestal at a dining table: as close as the room lets it, and chairs named, never removed.

capture-447ae4f4 (a tray carried 0.45 m across a dining table) placed the Franka's pedestal 1.03 m from the
pick and 1.11 m from the place, past the Panda's 0.99 m reach bound. Its record shows why
(``team_scene_inputs.v1.json``): of the 3,209 stances in the search disc, the stand validator's humanoid
standoff (0.4-1.2 m from the tray) refused 1,009, every one within 0.75 m of the pick; the room mesh refused
1,471 (1,373 occupied with its 8 cm clearance, 98 with no floor seen), among them all 244 that passed the
standoff nearer than 0.985 m; and the eight stances the carry was tried from were the eight closest to the
pick, on one arc on one side of the table, 1.02-1.29 m from the place. These rooms are synthetic: a scanned
oval table on four legs, chairs around it, all in one mesh as a website capture's collider is.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pxr")

from blueprint_pipeline.team_embodiment_stance import (  # noqa: E402
    PEDESTAL_CLEARANCE_M,
    PEDESTAL_SEARCH,
    RANK_BY_SUBJECT,
    RANK_BY_TASK_POINTS,
    ground_stance,
    spread_ranked,
)
from tests.test_team_embodiment_stance_room_mesh import (  # noqa: E402
    SUPPORT_TOP,
    TABLE_TOP,
    _box,
    _in_oval,
    _legged_table,
    footprint_samples,
    write_scanned_room,
)

TRAY_HALF = (0.155, 0.26, 0.02)
PICK = [-0.43, -0.30, SUPPORT_TOP + TRAY_HALF[2]]
PLACE = [-0.23, 0.10, SUPPORT_TOP + TRAY_HALF[2]]
SEAT_TOP, BACK_TOP = 0.45, 0.90


def _chair(x0: float, y0: float, *, back: str) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    """A 45 cm chair: a seat box from the floor and a 5 cm back on side ``back``; its mesh and its named bounds."""

    x1, y1 = x0 + 0.45, y0 + 0.45
    seat = _box((x0, y0, 0.0), (x1, y1, SEAT_TOP), bottom=False)
    bx0, by0, bx1, by1 = {"-x": (x0, y0, x0 + 0.05, y1), "+x": (x1 - 0.05, y0, x1, y1),
                          "-y": (x0, y0, x1, y0 + 0.05), "+y": (x0, y1 - 0.05, x1, y1)}[back]
    rest = _box((bx0, by0, SEAT_TOP), (bx1, by1, BACK_TOP), bottom=False)
    points = np.vstack([seat[0], rest[0]])
    faces = np.vstack([seat[1], rest[1] + len(seat[0])])
    return (points, faces), {"obstacle_id": f"chair_{back}_{x0:+.2f}_{y0:+.2f}", "label": "dining chair",
                             "bounds_world_m": {"minimum": [x0 - 0.01, y0 - 0.01, -0.01],
                                                "maximum": [x1 + 0.01, y1 + 0.01, BACK_TOP + 0.01]},
                             "basis": "synthetic_test_bounds"}


# Two chairs on each long side of the table, one at each end, pulled out as a dining room leaves them.
CHAIRS = [_chair(-1.55, -0.55, back="-x"), _chair(-1.55, 0.00, back="-x"),
          _chair(0.40, -0.55, back="+x"), _chair(0.40, 0.00, back="+x"),
          _chair(-0.60, -1.60, back="-y"), _chair(-0.60, 0.65, back="+y")]


def _room(tmp_path: Path, *, chairs: bool) -> Path:
    return write_scanned_room(tmp_path / ("chairs.usda" if chairs else "table.usda"), table=_legged_table(),
                              parts=[mesh for mesh, _ in CHAIRS] if chairs else [])


def _search(room: Path, *, pick: list[float] = PICK, **overrides: Any) -> Any:
    return ground_stance(**{"collision_path": room, "collision_sha256": "sha256:" + "0" * 64,
                            "offset": [0.0, 0.0, 0.0], "subject_center": pick,
                            "subject_minimum": [pick[i] - TRAY_HALF[i] for i in range(3)],
                            "subject_maximum": [pick[i] + TRAY_HALF[i] for i in range(3)],
                            "footprint_half_extent_m": [0.15, 0.15], "root_world_z_m": SUPPORT_TOP, **overrides})


def _gap_to_tray(stance: dict[str, Any], pick: list[float]) -> float:
    """The stand validator's standoff: the plan gap from the footprint's world box to the tray's."""

    x, y, _ = stance["position_world_m"]
    c, s = abs(math.cos(stance["yaw_rad"])), abs(math.sin(stance["yaw_rad"]))
    half = 0.15 * (c + s)
    dx = max(0.0, abs(x - pick[0]) - half - TRAY_HALF[0])
    dy = max(0.0, abs(y - pick[1]) - half - TRAY_HALF[1])
    return math.hypot(dx, dy)


def _farthest(stance: dict[str, Any], points: list[list[float]]) -> float:
    return max(math.dist(stance["position_world_m"], point) for point in points)


def _overlaps(stance: dict[str, Any], bounds: dict[str, Any]) -> bool:
    low, high = bounds["minimum"], bounds["maximum"]
    xy = footprint_samples(stance, grow_m=PEDESTAL_CLEARANCE_M)
    return bool(((xy[:, 0] >= low[0]) & (xy[:, 0] <= high[0]) & (xy[:, 1] >= low[1]) & (xy[:, 1] <= high[1])).any())


def test_a_pedestal_stands_at_the_tables_edge_not_a_humanoids_standoff_from_the_tray(tmp_path) -> None:
    # A tray 20 cm in from the table's end: the validator's humanoid standoff kept the pedestal 0.4 m from
    # the tray's box, 35 cm past where the table itself ends: 20 cm farther than the table makes it stand.
    room = _room(tmp_path, chairs=False)
    pick = [-0.65, -0.27, PICK[2]]
    default = _search(room, pick=pick)
    pedestal = _search(room, pick=pick, **PEDESTAL_SEARCH)
    assert _gap_to_tray(default, pick) >= 0.4 - 1e-9
    assert default["search"]["standoff_range_m"] is None and default["search"]["obstacle_clearance_m"] == 0.08
    assert default["search"]["nearer_refusals"]["by_reason"]["standoff_out"] > 0
    # The pedestal stands at the table's end: off the table by its own clearance, and nearer the tray.
    assert pedestal["search"]["standoff_range_m"] == [PEDESTAL_CLEARANCE_M, 1.6]
    assert pedestal["room_mesh_clearance"]["obstacle_clearance_m"] == PEDESTAL_CLEARANCE_M
    assert not _in_oval(footprint_samples(pedestal, grow_m=PEDESTAL_CLEARANCE_M)).any()
    near = math.dist(pedestal["position_world_m"][:2], pick[:2])
    assert near <= math.dist(default["position_world_m"][:2], pick[:2]) - 0.2 + 1e-9
    assert _gap_to_tray(pedestal, pick) < 0.4
    # What was nearer still and refused is the table itself (and the tray on it), measured at its own top.
    refusals = pedestal["search"]["nearer_refusals"]
    assert set(refusals["by_reason"]) <= {"occupied", "standoff_out"} and refusals["by_reason"]["occupied"] > 0
    tops = refusals["blocker_tops"]
    assert tops["at_mount"] == refusals["by_reason"]["occupied"] and tops["above_mount"] == tops["below_mount"] == 0
    blocked = [row for row in refusals["frontier"] if row["reason"] == "occupied"]
    assert blocked and all(row["blockers"]["top_above_floor_m"] == pytest.approx(TABLE_TOP, abs=0.01)
                           for row in blocked)


def test_a_carry_ranks_its_stances_by_the_farther_of_pick_and_place_a_footprint_apart(tmp_path) -> None:
    room = _room(tmp_path, chairs=False)
    points = [PICK, PLACE]
    by_pick = _search(room, ranked=8, **PEDESTAL_SEARCH)
    carry = _search(room, ranked=8, task_points_world_m=points, **PEDESTAL_SEARCH)
    assert by_pick[0]["search"]["ranking"] == {"rule": RANK_BY_SUBJECT}
    assert carry[0]["search"]["ranking"]["rule"] == RANK_BY_TASK_POINTS
    # The best stance for the carry is no farther from either end than any stance the pick ranking tried.
    assert _farthest(carry[0], points) < min(_farthest(stance, points) for stance in by_pick) - 0.02
    # Ranked by the pick alone, the eight tries crowd one spot; the carry's stand a footprint apart.
    def spacing(stances: list[dict[str, Any]]) -> float:
        return min(math.dist(a["position_world_m"][:2], b["position_world_m"][:2])
                   for i, a in enumerate(stances) for b in stances[i + 1:])
    assert spacing(by_pick) < 0.3 <= spacing(carry) + 1e-9
    assert [_farthest(stance, points) for stance in carry][0] == min(_farthest(stance, points) for stance in carry)
    assert all(not _in_oval(footprint_samples(stance, grow_m=PEDESTAL_CLEARANCE_M)).any() for stance in carry)


def test_chairs_around_the_table_are_named_as_movable_never_removed(tmp_path) -> None:
    room = _room(tmp_path, chairs=True)
    named = [row for _, row in CHAIRS]
    plain = _search(room, task_points_world_m=[PICK, PLACE], **PEDESTAL_SEARCH)
    reported = _search(room, task_points_world_m=[PICK, PLACE], movable_obstacles=named, **PEDESTAL_SEARCH)
    # The station is the same: the chairs are still in the room the robot runs in.
    assert reported["position_world_m"] == plain["position_world_m"]
    assert not any(_overlaps(reported, row["bounds_world_m"]) for row in named)
    assert "movable_obstacle_counterfactual" not in plain["search"]
    counterfactual = reported["search"]["movable_obstacle_counterfactual"]
    assert counterfactual["chosen"] is False and counterfactual["ignored_obstacles"] == 0
    assert counterfactual["obstacles"] == named
    assert counterfactual["ahead_of_chosen_stances"] > 0 and counterfactual["stances"]
    by_id = {row["obstacle_id"]: row["bounds_world_m"] for row in named}
    best = counterfactual["stances"][0]
    # Nearer the carry than the chosen station, and clear only with the chair it names moved.
    assert best["farthest_task_point_m"] < _farthest(reported, [PICK, PLACE]) - 0.05
    for row in counterfactual["stances"]:
        stance = {**reported, "position_world_m": [*row["position_world_m"], reported["position_world_m"][2]],
                  "yaw_rad": math.atan2(PICK[1] - row["position_world_m"][1], PICK[0] - row["position_world_m"][0])}
        assert row["moved"] and any(_overlaps(stance, by_id[name]) for name in row["moved"])
        assert not _in_oval(footprint_samples(stance, grow_m=PEDESTAL_CLEARANCE_M)).any()
    # The refusals ahead of the station say what was in the way: the table's top at the mount's height, and
    # chairs' seats below it, with the bounds of each just ahead of the chosen stance.
    refusals = reported["search"]["nearer_refusals"]
    assert refusals["blocker_tops"]["below_mount"] > 0 and refusals["blocker_tops"]["at_mount"] > 0
    seats = [row for row in refusals["frontier"] if row["reason"] == "occupied"
             and row["blockers"]["top_above_floor_m"] == pytest.approx(SEAT_TOP, abs=0.01)]
    assert seats and all(any(all(low - 0.02 <= v <= high + 0.02 for v, low, high in zip(
        point, by_id[name]["minimum"], by_id[name]["maximum"], strict=True)) for name in by_id)
        for row in seats for point in (row["blockers"]["bounds_world_m"]["minimum"],
                                       row["blockers"]["bounds_world_m"]["maximum"]))


def test_a_named_obstacle_without_usable_bounds_is_counted_and_never_refuses_the_search(tmp_path) -> None:
    room = _room(tmp_path, chairs=True)
    broken = [{"obstacle_id": "", "bounds_world_m": {"minimum": [0, 0, 0], "maximum": [1, 1, 1]}},
              {"obstacle_id": "chair", "bounds_world_m": {"minimum": [0, 0, 0], "maximum": [0, 1, 1]}},
              {"obstacle_id": "chair", "bounds_world_m": {"minimum": [0, 0, math.nan], "maximum": [1, 1, 1]}},
              {"label": "chair"}]
    stance = _search(room, movable_obstacles=broken, **PEDESTAL_SEARCH)
    record = stance["search"]["movable_obstacle_counterfactual"]
    assert record["ignored_obstacles"] == 4 and record["obstacles"] == [] and record["stances"] == []


def test_a_pedestal_never_stands_in_a_long_counter_the_box_index_would_drop(tmp_path) -> None:
    # A per-object scene: an 8 m counter is past the box index's 6 m cap. Only the humanoid standoff kept
    # the pedestal out of it; the pedestal's search keeps the long fixture instead.
    from tests.test_team_scene_packet import _box as usd_box
    from tests.test_team_scene_packet import _stage

    stage = _stage(tmp_path / "counter.usda", "/Root")
    usd_box(stage, "/Root/Floor", (-2.0, -14.0, 0.0), (9.0, 4.0, 0.0))
    usd_box(stage, "/Root/Counter", (2.5, -9.5, 0.0), (4.5, -1.4, 0.75))
    stage.GetRootLayer().Save()
    pick = [2.97, -6.76, 0.80]
    found = ground_stance(collision_path=tmp_path / "counter.usda", collision_sha256="sha256:" + "0" * 64,
                          offset=[0.0, 0.0, 0.0], subject_center=pick,
                          subject_minimum=[pick[0] - 0.06, pick[1] - 0.06, 0.75],
                          subject_maximum=[pick[0] + 0.06, pick[1] + 0.06, 0.88], footprint_half_extent_m=[0.15, 0.15],
                          root_world_z_m=0.75, task_points_world_m=[pick, [3.09, -6.76, 0.80]], **PEDESTAL_SEARCH)
    x, y, _ = found["position_world_m"]
    c, s = abs(math.cos(found["yaw_rad"])), abs(math.sin(found["yaw_rad"]))
    half = 0.15 * (c + s) + PEDESTAL_CLEARANCE_M
    assert x + half <= 2.5 + 1e-9 or x - half >= 4.5 - 1e-9 or y - half >= -1.4 - 1e-9 or y + half <= -9.5 + 1e-9


def test_a_free_base_and_the_g1_keep_the_validators_own_search(tmp_path) -> None:
    room = _room(tmp_path, chairs=False)
    stance = _search(room, root_world_z_m=None, root_height_m=0.35)
    assert stance["search"]["obstacle_clearance_m"] == 0.08 and stance["search"]["standoff_range_m"] is None
    assert stance["search"]["ranking"] == {"rule": RANK_BY_SUBJECT}
    assert _gap_to_tray(stance, PICK) >= 0.4 - 1e-9


def test_the_pedestal_keeps_the_arm_paths_own_scene_margin() -> None:
    from blueprint_pipeline.robot_arm_path_clearance import DEFAULT_MARGIN_M

    assert PEDESTAL_CLEARANCE_M == DEFAULT_MARGIN_M


def test_spread_ranked_takes_the_best_first_then_a_footprint_apart() -> None:
    rows = [((0.10,), 0.00, 0.0, 0.0), ((0.11,), 0.05, 0.0, 0.0), ((0.12,), 0.10, 0.0, 0.0),
            ((0.13,), 0.40, 0.0, 0.0), ((0.14,), 0.45, 0.0, 0.0), ((0.15,), 0.90, 0.0, 0.0)]
    assert [row[1] for row in spread_ranked(rows, separation=0.3, count=3)] == [0.0, 0.4, 0.9, 0.05, 0.1, 0.45]
    assert [row[1] for row in spread_ranked(rows, separation=0.3, count=1)] == [0.0, 0.05, 0.1, 0.4, 0.45, 0.9]
    assert spread_ranked(rows, separation=0.0, count=6) == rows
