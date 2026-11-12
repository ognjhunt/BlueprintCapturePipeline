"""The first look never says more than its own evidence: no false fit, no condition already false.

Real USD rooms where the stance search matters; fixture catalogs and a stand-in floor search elsewhere.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_robot_fit.py
#   src/blueprint_pipeline/website_robot_fit_floor.py
#   src/blueprint_pipeline/website_robot_fit_projection.py

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import pytest

from blueprint_pipeline import website_robot_fit as fit
from blueprint_pipeline import website_robot_fit_floor as floor_module
from blueprint_pipeline import website_robot_fit_projection as projection
from blueprint_pipeline.robot_spec_catalog import load_robot_spec_catalog
from blueprint_pipeline.team_embodiment import TeamEmbodimentError
from tests.test_robot_spec_catalog import _figure, _model, sealed
from tests.test_website_rigid_feasibility import _box, _room, _stage
from tests.test_website_robot_fit import _check, _collision, _job, _payload, _reach, _table, _verdict

HANDS = {"max_hand_height_m": _figure(1.5, "m", "Hand height up to 1.5 m"),
         "min_hand_height_m": _figure(0, "m", "Hand height from 0 m")}
DRIVES = {"max_base_speed_m_s": _figure(1.0, "m/s", "Base speed 1 m/s"), "base_footprint_diameter_m": None}
BAND = (0.75, 1.0 / 0.75)


def _floor(monkeypatch, **fields):
    """The floor search as the table would get it, with ``fields`` over a wide, clear floor at z = 0."""
    value = {"floor_z_m": 0.0, "lower_half_m": 2.0, "upper_half_m": None, "upper_hole_possible": False,
             "budget_exhausted": False,
             "searches": [{"half_extent_m": 2.0, "clear": True, "position_world_m": [0.0, 0.0, 0.0]}], **fields}
    monkeypatch.setattr(floor_module, "widest_clear_square", lambda **_: value)


def _square(side):
    return _figure([side, side], "m", f"{side * 1000:g} x {side * 1000:g} mm", f"{side * 1000:g} mm = {side:g} m")


# --- Must fix 1: stance and reach are decided at the same place -----------------------------------------


def test_a_large_base_is_never_passed_on_reach_measured_from_a_station_it_does_not_fit(tmp_path):
    # The review's corridor: 0.5 m wide around the task, holding the 0.3 m station by the task; a 0.6 m
    # base stands only where it opens, 1.4 m away, beyond the arm's 0.8 m reach.
    room = _stage(tmp_path / "collision.usda", "/Root")
    _box(room, "/Root/Floor", (-3.0, -3.0, 0.0), (3.0, 3.0, 0.0))
    index = 0
    for x0 in [-0.7 + 0.4 * k for k in range(8)]:
        for y0, y1 in [(0.25, 0.65), (0.65, 1.05), (1.05, 1.45), (1.45, 1.85), (-0.65, -0.25), (-1.05, -0.65),
                       (-1.45, -1.05), (-1.85, -1.45)]:
            _box(room, f"/Root/Block{index}", (x0, y0, 0.0), (x0 + 0.4, y1, 1.0))
            index += 1
    room.GetRootLayer().Save()
    collision = tmp_path / "collision.usda"
    job = _job(destination={"relation": "on", "position_world_m": [0.8, 0.0, 0.0]})
    job["authoring_inputs"]["metric_envelope"]["maximum_dimension_relative_error"] = 0.0
    catalog = sealed(_model("big_base_short_arm", base_footprint_diameter_m=None, base_footprint_l_w_m=_square(0.6),
                            max_reach_m=_reach(0.8), rated_payload_kg=_payload(30)))
    table = fit.robot_fit_table(preparation=job, collision_path=collision, verdict=_verdict(job, collision, stations=()),
                                catalog=catalog)
    row = table["rows"][0]
    stance, reach = _check(row, "stance"), _check(row, "reach")
    # The base stands where the corridor opens, not at the 0.3 m station by the task.
    assert stance["result"] == "pass"
    stood = stance["values"]["own_stance_positions_world_m"]
    assert stood and all(math.dist(position[:2], (0.5, 0.0)) > 1.0 for position in stood)
    assert table["inputs"]["stations"]["positions_world_m"][0][:2] == [0.25, 0.0]
    # Reach from there misses; others were not searched, so it is not proven either way.
    assert (reach["result"], reach["cause"]) == ("unknown", "own_stance_reach_unproven")
    assert reach["values"]["measured_from"] == "own_stance"
    assert row["first_look_verdict"] == "unknown"


def test_a_large_base_reaches_from_where_it_stands_and_otherwise_proves_nothing(tmp_path, monkeypatch):
    catalog = sealed(_model("wide", base_footprint_diameter_m=None, base_footprint_l_w_m=_square(0.5),
                            max_reach_m=_reach(1.5), rated_payload_kg=_payload(30)))
    # Stood next to the task: reach is measured from there and passes.
    _floor(monkeypatch)
    row = _table(tmp_path, catalog)["rows"][0]
    assert _check(row, "reach")["values"]["station_world_m"] == [0.0, 0.0, 0.0]
    assert row["first_look_verdict"] == "fits_spec"
    # The floor search said nothing (a non-answer): reach from the standard stations is not a pass.
    monkeypatch.setattr(floor_module, "widest_clear_square", lambda **_: (_ for _ in ()).throw(
        floor_module.FloorNotSearched("team_embodiment_scene_floor_not_found")))
    row = _table(tmp_path, catalog)["rows"][0]
    assert _check(row, "stance")["cause"] == "footprint_not_searched"
    assert (_check(row, "reach")["result"], _check(row, "reach")["cause"]) == ("unknown", "own_stance_reach_unproven")
    assert row["first_look_verdict"] == "unknown"


def test_no_robot_that_drives_reads_a_plain_fit_from_hand_heights_alone(tmp_path, monkeypatch):
    _floor(monkeypatch)
    rng = random.Random(3)
    models = [_model(f"drive_{index}", rng.choice(["mobile_manipulator", "humanoid", "wheeled_humanoid"]), **DRIVES,
                     **HANDS, rated_payload_kg=_payload(rng.choice([5, 20, 40])),
                     base_footprint_l_w_m=_square(rng.choice([0.2, 0.5, 1.0])))
              for index in range(12)]
    for weight in (None, "1 kg", "30 kg"):
        for row in _table(tmp_path, sealed(*models), job=_job(weight=weight))["rows"]:
            assert _check(row, "reach")["values"]["decided_by"] == "published_hand_height_range"
            assert row["first_look_verdict"] not in ("fits", "fits_spec")
            if row["first_look_verdict"] == "fits_if":
                assert "route_clear" in [entry["quantity"] for entry in row["conditions"]]


# --- Must fix 2: no condition the job's own estimate already makes false -------------------------------


def test_a_floor_narrower_than_the_base_at_every_scale_is_not_a_condition(tmp_path, monkeypatch):
    # The review's P3 and P3b: the floor seems to offer 0.3 to 0.6 m; the base is 1 m long.
    for size, hole in ((_figure([1.0, 0.5], "m", "1000 x 500 mm", "1000 mm = 1.0 m; 500 mm = 0.5 m"), False),
                       (_square(1.0), True)):
        _floor(monkeypatch, lower_half_m=0.2, upper_half_m=0.225, upper_hole_possible=hole,
               searches=[{"half_extent_m": 0.2, "clear": True, "position_world_m": [0.0, 0.0, 0.0]}])
        row = _table(tmp_path, sealed(_model("base", base_footprint_diameter_m=None, base_footprint_l_w_m=size,
                                             max_reach_m=_reach(2.0), rated_payload_kg=_payload(30))))["rows"][0]
        stance = _check(row, "stance")
        assert (stance["result"], stance["cause"], "condition" in stance) == ("unknown", "base_stance_not_found", False)
        assert row["first_look_verdict"] == "unknown"
        assert row["missing"][0] == {"check": "stance", "cause": "base_stance_not_found"}


def test_a_stated_weight_above_what_passes_with_the_default_tool_does_not_fit(tmp_path):
    # The review's P4: the site says 3.5 kg (3.15 to 3.85 kg); rated 3.6 kg less the 0.9 kg tool is 2.7 kg.
    row = _table(tmp_path, sealed(_model("arm36", rated_payload_kg=_payload(3.6), max_reach_m=_reach(2.0))),
                 job=_job(weight="3.5 kg"))["rows"][0]
    assert row["first_look_verdict"] == "doesnt_fit" and row["verdict"] == "doesnt_fit"
    assert [check["check"] for check in row["checks"] if check["result"] == "fail"] == ["payload"]
    assert row["conditions"] == []


def test_a_tool_that_alone_uses_the_whole_rating_leaves_no_weight_condition(tmp_path):
    job = _job()
    job["physics"] = {"basis": "estimated", "bounds": {"mass_kg": [0.0, 1.0]}}
    row = _table(tmp_path, sealed(_model("weak", rated_payload_kg=_payload(0.9))), job=job)["rows"][0]
    payload = _check(row, "payload")
    # Nothing heavier than 0 kg passes with the tool: no "fits if under 0 kg".
    assert (payload["result"], "condition" in payload) == ("unknown", False)
    assert row["first_look_verdict"] == "unknown"


def _false_by_own_estimate(condition):
    if condition["need_min"] is None or condition["quantity"] == "route_clear":
        return False
    if condition["quantity"] == "floor_width_m":
        return condition["need_max"] is not None and condition["need_max"] < condition["limit"]
    return condition["need_min"] > condition["limit"]


def test_random_tables_validate_and_carry_no_condition_already_false(tmp_path, monkeypatch):
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(json.loads((Path(__file__).resolve().parents[1]
                                                            / "docs/schemas/website_robot_fit_projection.v2.schema.json")
                                                           .read_text()))
    rng = random.Random(7)

    def maybe(chance, value):
        return value if rng.random() < chance else None

    def model(index):
        body = rng.choice(["fixed_arm", "mobile_manipulator", "humanoid", "wheeled_humanoid"])
        figures = {"max_reach_m": maybe(0.9, _reach(round(rng.uniform(0.2, 2.5), 3))),
                   "rated_payload_kg": maybe(0.9, _payload(round(rng.uniform(0.5, 40), 2))),
                   "max_tcp_speed_m_s": maybe(0.7, _figure(round(rng.uniform(0.2, 3), 2), "m/s", "Speed 1 m/s"))}
        shape = rng.random()
        figures["base_footprint_diameter_m"] = (_figure(round(rng.uniform(0.1, 1.5), 3), "m", "Footprint 1 mm",
                                                        "a = b") if shape < 0.4 else None)
        figures["base_footprint_l_w_m"] = (_figure(sorted([round(rng.uniform(0.08, 1.5), 3) for _ in range(2)],
                                                          reverse=True), "m", "1 x 1 mm", "a = b")
                                           if 0.4 <= shape < 0.8 else None)
        if body != "fixed_arm":
            top = round(rng.uniform(0.05, 2.5), 3)
            figures.update(max_base_speed_m_s=_figure(1.0, "m/s", "Base speed 1 m/s"),
                           max_hand_height_m=maybe(0.7, _figure(top, "m", "up to 1 m")),
                           min_hand_height_m=maybe(0.7, _figure(round(rng.uniform(0, top * 0.9), 3), "m", "from 0 m")))
        return _model(f"m{index}", body, **figures)

    def floor_result():
        if rng.random() < 0.15:
            return None
        lower = rng.uniform(0.15, 1.0)
        upper = None if rng.random() < 0.2 else lower + rng.uniform(0.001, 0.3)
        return {"floor_z_m": rng.choice([0.0, -0.05, 0.05, 0.2, None]), "lower_half_m": lower, "upper_half_m": upper,
                "upper_hole_possible": rng.random() < 0.5, "budget_exhausted": rng.random() < 0.1,
                "searches": [{"half_extent_m": lower, "clear": True, "position_world_m": [0.0, 0.0, 0.0]}]}

    conditions = 0
    for _ in range(150):
        catalog = sealed(*[model(index) for index in range(rng.randint(1, 10))])
        job = _job(weight=rng.choice([None, None, "3 kg", "12 lb", "0.5 kg", "25 kg"]),
                   cycle=rng.choice([None, 5, 15, 60]))
        job["authoring_inputs"]["metric_envelope"]["maximum_dimension_relative_error"] = rng.choice([0.0, 0.1, 0.25, 0.6])
        if rng.random() < 0.3:
            job["physics"] = {"basis": "estimated", "bounds": {"mass_kg": [0.0, rng.uniform(0, 50)]}}
        found = floor_result()
        monkeypatch.setattr(floor_module, "widest_clear_square", lambda found=found, **_: found)
        table = _table(tmp_path, catalog, job=job,
                       stations=rng.choice([([0.0, 0.0, 0.0],), ([0.0, 0.0, 0.0], [0.9, 0.2, 0.0]), ()]))
        models = [projection._v2_model(row) for row in sorted(table["rows"], key=projection.row_order)]
        value = projection._fit_bytes(projection._projection(
            "full", schema_version=projection.SCHEMA_VERSION_V2, status="completed",
            evidence_label="spec_bounded_estimate", counts=projection._first_look_counts(table),
            geometry_basis="estimated_from_video", weight_basis=table["inputs"]["mass_basis"], unknown_causes={},
            models=models, table_digest=table["table_digest"]))
        assert not list(validator.iter_errors(value))
        for row in table["rows"]:
            assert row["first_look_verdict"] not in ("fits", "fits_spec") or row["body_type"] == "fixed_arm"
            for condition in row["conditions"]:
                conditions += 1
                assert not _false_by_own_estimate(condition), (row["model_id"], condition)
    assert conditions > 50  # the run exercised conditions, not only rows without any


# --- Should fix -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("where, value", [("destination_y", float("nan")), ("subject_top", float("nan")),
                                          ("destination_z", float("inf"))])
def test_job_geometry_that_is_not_finite_is_refused_with_one_cause(tmp_path, where, value):
    job = copy.deepcopy(_job())  # the fixture's boxes are shared lists: never edit them in place
    if where == "subject_top":
        job["subject"]["aabb_max_xyz"][2] = value
    else:
        job["destination"]["position_world_m"][{"destination_y": 1, "destination_z": 2}[where]] = value
    catalog = sealed(_model("a"), _model("short", max_reach_m=_reach(0.1)),
                     _model("walker", "humanoid", **DRIVES, **HANDS))
    collision = _collision(tmp_path)
    table = fit.robot_fit_table(preparation=job, collision_path=collision, verdict=_verdict(job, collision),
                                catalog=catalog)
    assert {row["first_look_verdict"] for row in table["rows"]} == {"unknown"}
    assert {entry["cause"] for row in table["rows"] for entry in row["missing"]} == {"job_geometry_not_finite"}
    assert table["inputs"]["refused"] == "job_geometry_not_finite" and table["table_digest"].startswith("sha256:")


def test_a_task_point_below_the_found_floor_is_unknown_not_floor_level():
    model = sealed(_model("walker", "humanoid", **DRIVES, **HANDS))["models"][0]
    reach = fit._hand_height_reach(model, ([0.5, 0.0, 0.1], [0.5, 0.3, 0.1]), lambda: 0.9, BAND)
    assert (reach["result"], reach["cause"]) == ("unknown", "point_below_floor")
    assert reach["values"]["point_heights_above_floor_m"] == [-0.8, -0.8]
    reach = fit._hand_height_reach(model, ([0.5, 0.0, 0.1], [0.5, 0.3, 0.1]), lambda: None, BAND)
    assert reach["cause"] == "floor_height_not_found"


def test_hand_heights_use_the_floor_the_capability_verdict_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(floor_module, "widest_clear_square", lambda **_: pytest.fail("no search is needed"))
    job, collision = _job(), _collision(tmp_path)
    verdict = _verdict(job, collision)
    verdict["station"] = {"floor_z_m": 0.05}
    table = fit.robot_fit_table(preparation=job, collision_path=collision, verdict=verdict,
                                catalog=sealed(_model("walker", "humanoid", **DRIVES, **HANDS)))
    reach = _check(table["rows"][0], "reach")
    # The box top is 0.1 m: 0.05 m above that floor, 0.0375 to 0.0667 m across the band.
    assert reach["values"]["floor_z_m"] == 0.05 and reach["values"]["heights_above_floor_m"][0] == [0.0375, 0.066667]
    assert table["inputs"]["floor_height"] == {"floor_z_m": 0.05, "source": "capability_verdict_station"}


def test_the_floor_search_stops_at_its_wall_clock_budget(tmp_path):
    ticks = iter(range(100))
    asked = []

    def search(*, footprint_half_extent_m, **_):
        asked.append(footprint_half_extent_m[0])
        raise TeamEmbodimentError(["team_embodiment_free_base_no_clear_stance"])

    # Each reading of the clock moves it on 10 s: the budget of 20 s allows only the first searches.
    found = floor_module.widest_clear_square(preparation=_job(), collision_path=Path("unused"),
                                             collision_sha256="sha256:x", known_clear_half_m=0.15, needed_half_m=1.0,
                                             search=search, clock=lambda: 10.0 * next(ticks))
    assert found["budget_exhausted"] is True and len(asked) < 3
    # A table whose search ran out of time leaves the larger base unsearched, never decided.
    stance = fit._stance(sealed(_model("wide", base_footprint_diameter_m=None, base_footprint_l_w_m=_square(0.5)))
                         ["models"][0], [[0.0, 0.0, 0.0]], "test", (0.15, 0.15), path_checked=False,
                         floor=lambda: found, band=BAND)
    assert stance["cause"] == "footprint_not_searched"


def test_the_bisection_is_bounded_and_finds_the_floor_height(tmp_path):
    asked = []

    def search(*, footprint_half_extent_m, **_):
        half = footprint_half_extent_m[0]
        asked.append(half)
        if half > 0.16:
            raise TeamEmbodimentError(["team_embodiment_free_base_no_clear_stance"])
        return {"floor_z_m": 0.02, "position_world_m": [0.0, 0.0, 0.0]}

    found = floor_module.widest_clear_square(preparation=_job(), collision_path=Path("unused"),
                                             collision_sha256="sha256:x", known_clear_half_m=0.15, needed_half_m=100.0,
                                             search=search)
    # Seven searches toward a bound 0.01 m wide would take a dozen; then one at the station for the floor.
    assert len(asked) == 8 and asked[-1] == 0.15 and found["floor_z_m"] == 0.02
    assert found["upper_half_m"] is not None and found["upper_half_m"] > 0.2


def test_the_table_digest_is_the_same_for_the_same_inputs(tmp_path):
    room, job = _room(tmp_path), _job()
    verdict, catalog = _verdict(job, room), load_robot_spec_catalog()
    first = fit.robot_fit_table(preparation=job, collision_path=room, verdict=verdict, catalog=catalog)
    second = fit.robot_fit_table(preparation=job, collision_path=room, verdict=verdict, catalog=catalog)
    assert first["inputs"]["floor_search"]["result"]["searches"]
    assert first["table_digest"] == second["table_digest"]


def test_a_base_just_over_the_station_reads_like_one_just_under_it(tmp_path, monkeypatch):
    # At 0.75x the 0.3 m station is 0.225 m: both bases go to the floor search, which holds them alike.
    _floor(monkeypatch)
    rows = {row["model_id"]: row for row in _table(tmp_path, sealed(
        _model("at_030", base_footprint_diameter_m=None, base_footprint_l_w_m=_square(0.30)),
        _model("at_031", base_footprint_diameter_m=None, base_footprint_l_w_m=_square(0.31)),
        _model("small", base_footprint_diameter_m=_figure(0.2, "m", "Footprint 200 mm", "200 mm = 0.2 m"))))["rows"]}
    assert rows["at_030"]["first_look_verdict"] == rows["at_031"]["first_look_verdict"]
    assert _check(rows["at_030"], "stance")["values"]["floor_search"] == "widest_clear_square_base"
    assert "floor_search" not in _check(rows["small"], "stance")["values"]


def test_floor_verdicts_against_their_own_bounds():
    floor = {"lower_half_m": 0.38, "upper_half_m": 0.4, "upper_hole_possible": False,
             "searches": [{"half_extent_m": 0.38, "clear": True, "position_world_m": [1.0, 0.0, 0.0]}]}
    # The floor seems to offer 0.57 to 1.07 m.
    assert floor_module.floor_stance([[0.5, 0.4]], floor, BAND)["result"] == "pass"
    # A long, narrow base passes only on its longest side, never on its shortest.
    assert floor_module.floor_stance([[0.7, 0.3]], floor, BAND)["result"] == "conditional"
    assert floor_module.floor_stance([[0.7, 0.3]], floor, BAND)["positions"] == [[1.0, 0.0, 0.0]]
    # Within one grid step (0.067 m at 1.33x) of the widest found: not proven too wide.
    assert floor_module.floor_stance([[1.12, 1.12]], floor, BAND)["result"] == "not_found"
    assert floor_module.floor_stance([[1.2, 1.2]], floor, BAND)["result"] == "fail"


def test_no_reach_condition_after_the_arms_own_search_missed(tmp_path, monkeypatch):
    from blueprint_pipeline.robot_arm_chains import load_robot_arm_chains
    from tests.test_website_robot_fit import UR5E, _chained

    # The place needs 0.49 to 0.83 m; a 0.6 m published reach straddles it; the solver finds no pose.
    monkeypatch.setattr(fit, "_solve_station", lambda *_: (2, {"point": "place", "scale": 1.0, "failure": "x",
                                                               "detail": "y"}))
    row = _table(tmp_path, _chained(_model("ur5e", recipe_id=UR5E, max_reach_m=_reach(0.6))),
                 arm_chains=load_robot_arm_chains())["rows"][0]
    reach = _check(row, "reach")
    assert reach["values"]["published_reach"]["result"] == "unknown"
    assert (reach["result"], reach["cause"], "condition" in reach) == ("unknown", "arm_pose_search_inconclusive", False)
    assert row["first_look_verdict"] == "unknown"


def test_the_byte_cut_keeps_the_rows_most_likely_to_hold(tmp_path, monkeypatch):
    from tests.test_website_robot_fit_projection import CONTEXT, capture_root, write_fit_table

    rows = []
    for index in range(40):
        conditions = 1 + index % 3
        rows.append({"model_id": f"m{index}", "recipe_id": None, "name": f"Robot {index:02d}", "manufacturer": "Maker",
                     "body_type": "fixed_arm", "verdict": "unknown", "first_look_verdict": "fits_if",
                     "verdict_if_conditions_hold": "fits_spec", "rated_payload_kg": index + 1.0,
                     "payload_limit_kg": float(index), "missing": [], "reasons": [],
                     "conditions": [{"check": "payload", "quantity": "item_mass_kg", "unit": "kg", "need_min": 0.0,
                                     "need_max": 100.0, "limit": float(index), "need_basis": "estimated_from_video",
                                     "limit_basis": "catalog"}] * conditions,
                     "checks": [{"check": "payload", "result": "unknown", "reason": "", "values": {}}]})
    root = capture_root(tmp_path)
    write_fit_table(root, rows=rows)
    value = projection.read_robot_fit_v2(capture_root=root, task_context_digest=CONTEXT)
    kept = [(len(model["conditions"]), model["payload_limit_kg"]) for model in value["models"]]
    assert value["models_truncated"] is True and kept == sorted(kept, key=lambda row: (row[0], -row[1]))
    assert kept[0] == (1, 39.0)
    # A reader derives how many rows were cut from the whole table's counts.
    assert sum(value["counts"].values()) - len(value["models"]) == 40 - len(kept) > 0


def test_the_v2_schema_lists_every_cause_and_quantity_the_code_writes():
    schema = json.loads((Path(__file__).resolve().parents[1]
                         / "docs/schemas/website_robot_fit_projection.v2.schema.json").read_text())
    model = schema["properties"]["models"]["items"]["properties"]
    assert set(model["missing"]["items"]["properties"]["cause"]["enum"]) == set(projection.UNKNOWN_CAUSES)
    assert set(model["conditions"]["items"]["properties"]["quantity"]["enum"]) == set(projection.QUANTITY_UNITS)


def test_a_base_needing_a_large_square_never_passes_from_where_only_small_squares_stand(tmp_path, monkeypatch):
    # The re-check's case C: squares up to 0.45 m half stand next to the task, larger ones only 1.7 m away,
    # nothing over 0.6 m half. A 0.75 m base needs a 0.5 m half square at 0.75x: only the far place holds it.
    # A 1.05 m base sets the search's top (0.7 m half), so the bisection also stands a near, smaller square.
    near, far = [0.2, 0.0, 0.0], [-1.5, 0.0, 0.0]

    def search(*, footprint_half_extent_m, **_):
        half = footprint_half_extent_m[0]
        if half > 0.6:
            raise TeamEmbodimentError(["team_embodiment_free_base_no_clear_stance"])
        return {"floor_z_m": 0.0, "position_world_m": near if half <= 0.45 else far}

    real = floor_module.widest_clear_square
    monkeypatch.setattr(floor_module, "widest_clear_square", lambda **kwargs: real(search=search, **kwargs))
    def rows(*sides):
        catalog = sealed(*[_model(f"base_{side}", base_footprint_diameter_m=None, base_footprint_l_w_m=_square(side),
                                  max_reach_m=_reach(0.8), rated_payload_kg=_payload(30)) for side in sides])
        collision, job = _collision(tmp_path), _job()
        table = fit.robot_fit_table(preparation=job, collision_path=collision, verdict=_verdict(job, collision),
                                    catalog=catalog)
        return {row["model_id"]: row for row in table["rows"]}, table["inputs"]["floor_search"]["result"]["searches"]

    found, searches = rows(0.75, 1.05)
    assert near in [row.get("position_world_m") for row in searches]  # a near square was found, too small
    large = _check(found["base_0.75"], "stance")
    assert large["result"] == "pass" and large["values"]["own_stance_positions_world_m"]
    # Every place it is measured from held a square at least as wide as the base needs at 0.75x (0.5 m half).
    held = [row["position_world_m"] for row in searches if row["clear"] and row["half_extent_m"] >= 0.5]
    assert all(place in held for place in large["values"]["own_stance_positions_world_m"])
    assert near not in large["values"]["own_stance_positions_world_m"]
    # Reach is measured from the far place only, where it misses: never a pass from the near one.
    assert _check(found["base_0.75"], "reach")["cause"] == "own_stance_reach_unproven"
    assert found["base_0.75"]["first_look_verdict"] == "unknown"
    # Alone, the smaller base is searched at its own size, held next to the task, and reaches from there.
    found, _ = rows(0.45)
    assert _check(found["base_0.45"], "stance")["values"]["own_stance_positions_world_m"] == [near]
    assert found["base_0.45"]["first_look_verdict"] == "fits_spec"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_stations_and_places_are_dropped_never_measured_from(tmp_path, monkeypatch, bad):
    # A station from the capability verdict: a 0.1 m reach arm must not fit from a NaN or infinite station.
    job, collision = _job(), _collision(tmp_path)
    table = fit.robot_fit_table(preparation=job, collision_path=collision,
                                verdict=_verdict(job, collision, stations=([bad, 0.0, 0.0],)),
                                catalog=sealed(_model("short", max_reach_m=_reach(0.1))))
    row = table["rows"][0]
    assert table["inputs"]["stations"]["positions_world_m"] == []
    assert row["first_look_verdict"] != "fits_spec" and _check(row, "stance")["cause"] == "no_clear_station"
    json.dumps(table, allow_nan=False)  # nothing non-finite is written into the digested table

    # A place or floor height the floor search returns that is not finite is dropped too.
    def search(*, footprint_half_extent_m, **_):
        return {"floor_z_m": bad, "position_world_m": [0.0, bad, 0.0]}

    found = floor_module.widest_clear_square(preparation=job, collision_path=Path("unused"),
                                             collision_sha256="sha256:x", known_clear_half_m=0.15,
                                             needed_half_m=1.0, search=search)
    assert found["floor_z_m"] is None and all("position_world_m" not in row for row in found["searches"])
    json.dumps(found, allow_nan=False)
