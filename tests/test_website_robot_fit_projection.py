"""The site reads its job's robot fit table through preparation status v5, never an unbound one."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_robot_fit_projection.py
#   src/blueprint_pipeline/website_preparation_status.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import website_robot_fit_projection as projection
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

CONTEXT = "sha256:" + "c" * 64


def _row(name, verdict, results):
    return {"model_id": name.lower().replace(" ", "_"), "recipe_id": None, "name": name, "manufacturer": "Maker",
            "body_type": "fixed_arm", "verdict": verdict, "reasons": [],
            "checks": [{"check": check, "result": result, "reason": "", "values": {}}
                       for check, result in results.items()]}


ROWS = [
    _row("Arm Unknown", "unknown", {"stance": "pass", "reach": "unknown", "payload": "unknown",
                                    "cycle_time": "not_requested"}),
    _row("Arm Short", "doesnt_fit", {"stance": "pass", "reach": "fail", "payload": "pass"}),
    _row("Arm Spec", "fits_spec", {"stance": "pass", "reach": "pass", "payload": "pass", "access": "unknown"}),
    _row("Arm Path", "fits", {"stance": "pass", "reach": "pass", "payload": "pass"}),
]


def capture_root(tmp_path: Path, scene_id: str = "site-req1", capture_id: str = "walkthrough-req1") -> Path:
    root = tmp_path / "storage" / "scenes" / scene_id / "captures" / capture_id
    root.mkdir(parents=True)
    return root


def write_fit_table(root: Path, *, route: str = "full", rows=ROWS, context: str = CONTEXT,
                    handoff: dict | None = None) -> dict:
    """A handoff with its preparation and fit table, as the scene handoff leaves them."""
    work = root / "pipeline" / "website_scene_preparation"
    preparation = {"schema_version": "website_scene_preparation.v1", "binding": {"task_context_digest": context}}
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    write_json(work / "preparation.json", preparation)
    table = {"schema_version": "website_robot_fit_table.v1", "preparation_digest": preparation["digest"],
             "evidence_label": "spec_bounded_estimate", "claim_ceiling": "development_only",
             "physical_world_truth_claimed": False, "rows": rows,
             "counts": {verdict: sum(row["verdict"] == verdict for row in rows) for verdict in projection.VERDICTS}}
    table["table_digest"] = canonical_digest(table, digest_field="table_digest")
    table_path = work / ("fit_check" if route == "geometry_only" else "native") / "robot_fit_table.json"
    write_json(table_path, table)
    reference = {"path": str(table_path), "digest": table["table_digest"], "counts": table["counts"]}
    value = {"schema_version": "website_scene_handoff.v1", "status": "intake_ready", "blockers": [],
             "scene_id": root.parent.parent.name, "capture_id": root.name,
             "preparation_path": str(work / "preparation.json"), "preparation_digest": preparation["digest"],
             **({"atlas_route": "geometry_only", "fit_check": {"status": "completed", "robot_fit_table": reference}}
                if route == "geometry_only" else {"robot_fit_table": reference}),
             **(handoff or {})}
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(work / "handoff.json", value)
    return value


def test_the_site_reads_the_full_routes_fit_table_with_the_robots_that_fit_first(tmp_path):
    root = capture_root(tmp_path)
    handoff = write_fit_table(root)
    value = projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT)
    table = json.loads(Path(handoff["robot_fit_table"]["path"]).read_text())
    assert value == {
        "schema_version": "website_robot_fit_projection.v1", "route": "full", "status": "completed", "reason": None,
        "evidence_label": "spec_bounded_estimate",
        "counts": {"fits": 1, "fits_spec": 1, "doesnt_fit": 1, "unknown": 1},
        # Only what the spec sheets and the scene could not settle, or ruled out, is named.
        "models": [{"name": "Arm Path", "manufacturer": "Maker", "verdict": "fits", "open_checks": []},
                   {"name": "Arm Spec", "manufacturer": "Maker", "verdict": "fits_spec", "open_checks": ["access"]},
                   {"name": "Arm Short", "manufacturer": "Maker", "verdict": "doesnt_fit", "open_checks": ["reach"]},
                   {"name": "Arm Unknown", "manufacturer": "Maker", "verdict": "unknown",
                    "open_checks": ["reach", "payload"]}],
        "models_truncated": False, "table_digest": table["table_digest"]}


def test_a_geometry_only_job_reads_its_fit_check(tmp_path):
    root = capture_root(tmp_path)
    write_fit_table(root, route="geometry_only")
    value = projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT)
    assert value["route"] == "geometry_only" and value["status"] == "completed" and len(value["models"]) == 4


@pytest.mark.parametrize("reason, expected", [("task_family_not_modelled", True), ("subject_not_excised", True),
                                              ("arbitrary_error", False)])
def test_a_geometry_only_check_that_did_not_run_says_why(tmp_path, reason, expected):
    root = capture_root(tmp_path)
    write_fit_table(root, route="geometry_only", handoff={"fit_check": {"status": "not_run", "reason": reason}})
    value = projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT)
    if not expected:
        assert value is None
        return
    assert value == {"schema_version": "website_robot_fit_projection.v1", "route": "geometry_only",
                     "status": "not_run", "reason": reason, "evidence_label": None, "counts": None, "models": [],
                     "models_truncated": False, "table_digest": None}


@pytest.mark.parametrize("fault", ["edited_table", "reference_digest", "other_context", "other_preparation",
                                   "outside_capture", "symlink", "physical_truth", "unknown_verdict",
                                   "other_capture", "unknown_route", "no_table"])
def test_a_fit_table_that_does_not_bind_to_this_job_is_left_out(tmp_path, fault):
    root = capture_root(tmp_path)
    if fault == "other_context":
        write_fit_table(root, context="sha256:" + "d" * 64)
    elif fault == "unknown_verdict":
        write_fit_table(root, rows=[_row("Arm", "probably", {"reach": "pass"})])
    elif fault == "other_capture":
        write_fit_table(root, handoff={"capture_id": "walkthrough-other"})
    elif fault == "unknown_route":
        write_fit_table(root, handoff={"atlas_route": "cheapest"})
    elif fault == "no_table":
        write_fit_table(root, handoff={"robot_fit_table": {"status": "failed", "error": "KeyError"}})
    else:
        handoff = write_fit_table(root)
        work = root / "pipeline" / "website_scene_preparation"
        table_path = Path(handoff["robot_fit_table"]["path"])
        table = json.loads(table_path.read_text())
        if fault == "edited_table":
            table["rows"][0]["verdict"] = "fits"
            write_json(table_path, table)
        elif fault == "physical_truth":
            table["physical_world_truth_claimed"] = True
            table["table_digest"] = canonical_digest(table, digest_field="table_digest")
            write_json(table_path, table)
        elif fault == "other_preparation":
            table["preparation_digest"] = "sha256:" + "e" * 64
            table["table_digest"] = canonical_digest(table, digest_field="table_digest")
            write_json(table_path, table)
        else:
            reference = dict(handoff["robot_fit_table"])
            if fault == "reference_digest":
                reference["digest"] = "sha256:" + "f" * 64
            elif fault == "outside_capture":
                outside = tmp_path / "elsewhere.json"
                write_json(outside, table)
                reference["path"] = str(outside)
            elif fault == "symlink":
                link = work / "linked.json"
                link.symlink_to(table_path)
                reference["path"] = str(link)
            handoff["robot_fit_table"] = reference
            handoff["digest"] = canonical_digest(handoff, digest_field="digest")
            write_json(work / "handoff.json", handoff)
    assert projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT) is None


def test_no_handoff_yet_projects_nothing(tmp_path):
    assert projection.read_robot_fit(capture_root=capture_root(tmp_path), task_context_digest=CONTEXT) is None


def test_a_long_table_is_cut_to_its_budget_keeping_the_robots_that_fit(tmp_path):
    rows = ([_row(f"Unknown Arm {index:02d}", "unknown", {"reach": "unknown", "payload": "unknown",
                                                          "access": "unknown"}) for index in range(70)]
            + [_row(f"Fitting Arm {index:02d}", "fits", {"reach": "pass"}) for index in range(10)])
    root = capture_root(tmp_path)
    write_fit_table(root, rows=rows)
    value = projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT)
    assert value["models_truncated"] is True
    assert len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()) <= projection.MAX_BYTES
    assert [row["verdict"] for row in value["models"][:10]] == ["fits"] * 10
    # The counts stay those of the whole table.
    assert value["counts"] == {"fits": 10, "fits_spec": 0, "doesnt_fit": 0, "unknown": 70}


def test_every_projection_matches_the_published_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).resolve().parents[1]
                         / "docs/schemas/website_robot_fit_projection.v1.schema.json").read_text())
    completed = capture_root(tmp_path / "completed")
    write_fit_table(completed)
    not_run = capture_root(tmp_path / "not_run")
    write_fit_table(not_run, route="geometry_only",
                    handoff={"fit_check": {"status": "not_run", "reason": "task_family_not_modelled"}})
    for root in (completed, not_run):
        jsonschema.validate(projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT), schema)


def test_the_screening_planner_reads_each_models_verdict_in_the_tables_own_order(tmp_path):
    root = capture_root(tmp_path)
    write_fit_table(root, rows=[{**row, "recipe_id": row["model_id"] if row["verdict"] == "fits" else None}
                                for row in ROWS])
    assert projection.robot_fit_rows(capture_root=root, task_context_digest=CONTEXT) == [
        {"model_id": "arm_unknown", "recipe_id": None, "verdict": "unknown"},
        {"model_id": "arm_short", "recipe_id": None, "verdict": "doesnt_fit"},
        {"model_id": "arm_spec", "recipe_id": None, "verdict": "fits_spec"},
        {"model_id": "arm_path", "recipe_id": "arm_path", "verdict": "fits"}]
    # Bound to another task context, or a check that did not run: nothing, never a guess.
    assert projection.robot_fit_rows(capture_root=root, task_context_digest="sha256:" + "d" * 64) is None
    other = capture_root(tmp_path, capture_id="walkthrough-req2")
    write_fit_table(other, route="geometry_only", handoff={"fit_check": {"status": "not_run",
                                                                         "reason": "preparation_blocked"}})
    assert projection.robot_fit_rows(capture_root=other, task_context_digest=CONTEXT) is None
    assert projection.read_robot_fit(capture_root=other, task_context_digest=CONTEXT)["reason"] == "preparation_blocked"


def _condition(quantity, need, limit, basis="estimated_from_video"):
    check = {"item_mass_kg": "payload", "reach_m": "reach", "cycle_time_s": "cycle_time"}[quantity]
    unit = {"item_mass_kg": "kg", "reach_m": "m", "cycle_time_s": "s"}[quantity]
    return {"check": check, "quantity": quantity, "unit": unit, "need_min": need and need[0],
            "need_max": need and need[1], "limit": limit, "need_basis": basis, "limit_basis": "catalog"}


def _first_look_row(name, legacy, first_look, results, *, limit=None, conditions=(), missing=(), hold=None):
    """A table row as ``website_robot_fit`` now writes it: the legacy verdict beside the first look's."""
    return {**_row(name, legacy, results), "first_look_verdict": first_look, "verdict_if_conditions_hold": hold,
            "rated_payload_kg": None if limit is None else limit + 0.9, "payload_limit_kg": limit,
            "conditions": list(conditions), "missing": list(missing)}


FIRST_LOOK_ROWS = [
    _first_look_row("Arm Spec", "fits_spec", "fits_spec", {"stance": "pass", "reach": "pass", "payload": "pass"},
                    limit=19.1),
    _first_look_row("Arm Big", "unknown", "fits_if", {"stance": "pass", "reach": "pass", "payload": "unknown"},
                    limit=11.6, hold="fits_spec", conditions=[_condition("item_mass_kg", [0.024, 33.647], 11.6)]),
    _first_look_row("Arm Small", "unknown", "fits_if", {"stance": "pass", "reach": "unknown", "payload": "unknown"},
                    limit=2.1, hold="fits", conditions=[_condition("reach_m", [0.701, 1.219], 0.85),
                                                        _condition("item_mass_kg", [0.024, 33.647], 2.1)]),
    _first_look_row("Base Roaming", "unknown", "fits_if", {"stance": "pass", "reach": "pass", "payload": "pass",
                                                            "access": "unknown"},
                    limit=30.0, hold="fits_spec", conditions=[{
                        "check": "access", "quantity": "route_clear", "unit": None, "need_min": None,
                        "need_max": None, "limit": None, "need_basis": "not_checked", "limit_basis": "not_checked"}]),
    _first_look_row("Arm Short", "doesnt_fit", "doesnt_fit", {"stance": "pass", "reach": "fail", "payload": "unknown"},
                    limit=2.6),
    _first_look_row("Arm Open", "unknown", "unknown", {"stance": "unknown", "reach": "pass", "payload": "unknown"},
                    limit=4.1, missing=[{"check": "stance", "cause": "catalog_figure_missing"}]),
]


def test_v1_reads_a_first_look_table_exactly_as_before(tmp_path):
    root = capture_root(tmp_path)
    write_fit_table(root, rows=FIRST_LOOK_ROWS)
    value = projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT)
    # The WebApp on main parses v1 strictly: the same keys, the four verdicts, a fits_if row read as unknown.
    assert value["schema_version"] == "website_robot_fit_projection.v1"
    assert value["counts"] == {"fits": 0, "fits_spec": 1, "doesnt_fit": 1, "unknown": 4}
    assert [(row["name"], row["verdict"]) for row in value["models"]] == [
        ("Arm Spec", "fits_spec"), ("Arm Short", "doesnt_fit"), ("Arm Big", "unknown"), ("Arm Small", "unknown"),
        ("Base Roaming", "unknown"), ("Arm Open", "unknown")]
    assert all(set(row) == {"name", "manufacturer", "verdict", "open_checks"} for row in value["models"])
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(value, json.loads((Path(__file__).resolve().parents[1]
                                           / "docs/schemas/website_robot_fit_projection.v1.schema.json").read_text()))


def test_v2_says_what_each_robot_can_do_most_useful_first(tmp_path):
    root = capture_root(tmp_path)
    handoff = write_fit_table(root, rows=FIRST_LOOK_ROWS)
    value = projection.read_robot_fit_v2(capture_root=root, task_context_digest=CONTEXT)
    table = json.loads(Path(handoff["robot_fit_table"]["path"]).read_text())
    assert {key: value[key] for key in ("schema_version", "status", "counts", "geometry_basis", "unknown_causes",
                                        "table_digest")} == {
        "schema_version": "website_robot_fit_projection.v2", "status": "completed",
        "counts": {"fits": 0, "fits_spec": 1, "fits_if": 3, "doesnt_fit": 1, "unknown": 1},
        "geometry_basis": "estimated_from_video", "unknown_causes": {"stance:catalog_figure_missing": 1},
        "table_digest": table["table_digest"]}
    # fits_spec, then fits_if by ascending payload limit, then doesnt_fit, then unknown.
    assert [row["name"] for row in value["models"]] == ["Arm Spec", "Arm Small", "Arm Big", "Base Roaming",
                                                        "Arm Short", "Arm Open"]
    small = value["models"][1]
    assert small == {"name": "Arm Small", "manufacturer": "Maker", "verdict": "fits_if",
                     "verdict_if_conditions_hold": "fits", "rated_payload_kg": 3.0, "payload_limit_kg": 2.1,
                     "failed_checks": [],
                     "conditions": [{"quantity": "reach_m", "need_min": 0.701, "need_max": 1.219, "limit": 0.85,
                                     "need_basis": "estimated_from_video", "limit_basis": "catalog"},
                                    {"quantity": "item_mass_kg", "need_min": 0.024, "need_max": 33.647, "limit": 2.1,
                                     "need_basis": "estimated_from_video", "limit_basis": "catalog"}],
                     "missing": []}
    assert value["models"][3]["conditions"] == [{"quantity": "route_clear", "need_min": None, "need_max": None,
                                                  "limit": None, "need_basis": "not_checked",
                                                  "limit_basis": "not_checked"}]
    assert value["models"][4]["failed_checks"] == ["reach"] and value["models"][4]["conditions"] == []
    assert value["models"][5]["missing"] == [{"check": "stance", "cause": "catalog_figure_missing"}]
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).resolve().parents[1]
                         / "docs/schemas/website_robot_fit_projection.v2.schema.json").read_text())
    jsonschema.validate(value, schema)
    not_run = capture_root(tmp_path / "not_run")
    write_fit_table(not_run, route="geometry_only",
                    handoff={"fit_check": {"status": "not_run", "reason": "task_family_not_modelled"}})
    jsonschema.validate(projection.read_robot_fit_v2(capture_root=not_run, task_context_digest=CONTEXT), schema)


def test_v2_reads_a_table_written_before_the_first_look(tmp_path):
    root = capture_root(tmp_path)
    write_fit_table(root)
    value = projection.read_robot_fit_v2(capture_root=root, task_context_digest=CONTEXT)
    assert value["counts"] == {"fits": 1, "fits_spec": 1, "fits_if": 0, "doesnt_fit": 1, "unknown": 1}
    assert [row["verdict"] for row in value["models"]] == ["fits", "fits_spec", "doesnt_fit", "unknown"]
    assert all(row["payload_limit_kg"] is None and row["conditions"] == [] for row in value["models"])


@pytest.mark.parametrize("fault", ["fits_if_legacy_fits", "no_conditions", "unknown_quantity", "unknown_cause",
                                   "unknown_first_look", "route_with_numbers", "route_on_payload"])
def test_a_first_look_row_a_reader_could_not_word_is_left_out(tmp_path, fault):
    rows = [dict(row) for row in FIRST_LOOK_ROWS]
    big = rows[1]
    if fault == "fits_if_legacy_fits":
        big["verdict"] = "fits_spec"
    elif fault == "no_conditions":
        big["conditions"] = []
    elif fault == "unknown_quantity":
        big["conditions"] = [{**big["conditions"][0], "quantity": "item_volume_l"}]
    elif fault == "unknown_cause":
        rows[5]["missing"] = [{"check": "stance", "cause": "felt_wrong"}]
    elif fault == "route_with_numbers":
        rows[3]["conditions"] = [{**rows[3]["conditions"][0], "limit": 1.0}]
    elif fault == "route_on_payload":
        rows[3]["conditions"] = [{**rows[3]["conditions"][0], "check": "payload"}]
    else:
        big["first_look_verdict"] = "probably"
    root = capture_root(tmp_path)
    write_fit_table(root, rows=rows)
    assert projection.read_robot_fit_v2(capture_root=root, task_context_digest=CONTEXT) is None


@pytest.mark.parametrize("quantity, check", [("floor_width_m", "stance"), ("hand_height_m", "reach")])
def test_v2_publishes_the_floor_and_hand_height_conditions_as_values_only(tmp_path, quantity, check):
    rows = [dict(row) for row in FIRST_LOOK_ROWS]
    rows[1]["conditions"] = [{"check": check, "quantity": quantity, "unit": "m", "need_min": 0.6, "need_max": 1.06,
                              "limit": 0.9, "need_basis": "estimated_from_video", "limit_basis": "catalog"}]
    root = capture_root(tmp_path)
    write_fit_table(root, rows=rows)
    value = projection.read_robot_fit_v2(capture_root=root, task_context_digest=CONTEXT)
    assert value["models"][2]["conditions"] == [{"quantity": quantity, "need_min": 0.6, "need_max": 1.06,
                                                 "limit": 0.9, "need_basis": "estimated_from_video",
                                                 "limit_basis": "catalog"}]
    # The same keys at every level as revision 2: a new value, never a new field.
    assert set(value["models"][2]) == set(value["models"][0])
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(value, json.loads((Path(__file__).resolve().parents[1]
                                           / "docs/schemas/website_robot_fit_projection.v2.schema.json").read_text()))
    # A condition on the wrong check is not one a reader could word.
    rows[1]["conditions"] = [{**rows[1]["conditions"][0], "check": "payload"}]
    write_fit_table(root, rows=rows)
    assert projection.read_robot_fit_v2(capture_root=root, task_context_digest=CONTEXT) is None
