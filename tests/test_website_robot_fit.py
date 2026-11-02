"""Per robot model, whether it fits a website job by spec sheet, decided across the whole uncertainty band.

Fixture spec catalogs only. Every case but one carries a recorded capability verdict, so no carry is
solved; the clear case runs the real verdict on the toy room its own tests use.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import team_embodiment_stance
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.robot_spec_catalog import validate_robot_spec_catalog
from blueprint_pipeline.website_rigid_feasibility import rigid_task_verdict
from blueprint_pipeline.website_robot_fit import PROFILE_RECIPES, owner_stated_weight_kg, robot_fit_table
from blueprint_pipeline.website_scene_handoff import prepare_rigid_native_inputs
from tests.test_robot_spec_catalog import _catalog, _figure, _model, sealed
from tests.test_website_rigid_feasibility import DROID, _preparation, _room

# A 10 cm box on the floor, its top at (0.5, 0, 0.1), set down 0.3 m away: its top again at 0.1 m.
SUBJECT = ([0.45, -0.05, 0.0], [0.55, 0.05, 0.1])
ACROSS = {"relation": "on", "position_world_m": [0.5, 0.3, 0.0]}
FRANKA_RECIPE = "droid_franka_robotiq_2f85"
LIFT_BLOCKED = "robot_rigid_pick_place_unreachable:lift"
UNMODELED = "robot_profile_unmodeled:team_mobile_manipulator_v1"


def _job(*, weight=None, cycle=None, destination=ACROSS, subject=SUBJECT):
    """A prepared rigid job as ``compile_website_scene_preparation`` records the fields the table reads."""
    job = _preparation(dict(destination), subject=subject)
    job["authoring_inputs"] = {"metric_envelope": {"maximum_dimension_relative_error": 0.25},
                               "configuration": {"construction_constraints": {"operator_task_details": (
                                   {} if weight is None else {"item_weight": weight})}}}
    job["physics"] = {"basis": "estimated", "bounds": {"mass_kg": [0.2, 1.0]}}
    job["owner_success_criteria"] = {"status": "not_supplied" if cycle is None else "unsupported", "targets": (
        None if cycle is None else {"successDefinition": None, "successRate": None, "cycleTimeSeconds": cycle,
                                    "unknown": False})}
    return job


def _collision(tmp_path):
    """Collision bytes a recorded verdict is bound to; with its stations recorded, nothing reads the scene."""
    path = tmp_path / "recorded_collision.usda"
    path.write_bytes(b"#usda 1.0\n")
    return path


def _verdict(job, collision, status="abstained", *, stations=([0.0, 0.0, 0.0],), carry=None, carry_blockers=(),
             blockers=()):
    """A capability verdict as ``rigid_task_verdict`` records one, every station tried unless clear."""
    value = {"schema_version": "website_rigid_task_capability_verdict.v1", "preparation_digest": job["digest"],
             "collision_sha256": "sha256:" + hashlib.sha256(Path(collision).read_bytes()).hexdigest(),
             "status": status, "blockers": list(blockers), "embodiments": [] if carry is None else [
                 {"robot_id": "franka_panda", "status": carry, "blockers": list(carry_blockers), "grasp": "top_down"}]}
    if stations:
        value["station_results"] = [{"position_world_m": list(row), "yaw_rad": 0.0, "grasp": "top_down",
                                     "status": "blocked", "blockers": []} for row in stations]
    value["verdict_digest"] = canonical_digest(value, digest_field="verdict_digest")
    return value


def _table(tmp_path, catalog, job=None, **verdict):
    job = job or _job()
    collision = _collision(tmp_path)
    return robot_fit_table(preparation=job, collision_path=collision, verdict=_verdict(job, collision, **verdict),
                           catalog=catalog)


def _rows(table):
    return {row["model_id"]: row for row in table["rows"]}


def _check(row, name):
    return next(check for check in row["checks"] if check["check"] == name)


def _reach(metres):
    return _figure(metres, "m", f"Reach {metres * 1000:g} mm", f"{metres * 1000:g} mm = {metres:g} m")


def test_reach_fails_only_when_even_the_optimistic_scale_is_out_of_reach(tmp_path):
    catalog = sealed(_model("long_arm", max_reach_m=_reach(1.0)), _model("short_arm", max_reach_m=_reach(0.4)),
                     _model("middle_arm", max_reach_m=_reach(0.6)))
    table = _table(tmp_path, catalog)
    rows = _rows(table)
    # From the station at the origin with the 0.15 m tool, at 0.75x and 1.33x scale, the place point
    # (0.5, 0.3, 0.1 + tool) needs 0.4918 m at best and 0.8275 m at worst; the grasp point
    # (0.5, 0, 0.1 + tool) needs less, 0.4373 and 0.7244 m.
    assert _check(rows["long_arm"], "reach")["values"]["required_m"] == pytest.approx(
        {"optimistic": 0.491808, "pessimistic": 0.827479}, abs=1e-6)
    assert [_check(rows[name], "reach")["result"] for name in ("long_arm", "short_arm", "middle_arm")] == [
        "pass", "fail", "unknown"]
    assert [rows[name]["verdict"] for name in ("long_arm", "short_arm", "middle_arm")] == [
        "fits_spec", "doesnt_fit", "unknown"]
    assert rows["short_arm"]["reasons"] == [
        "reach: the grasp or place point is beyond reach from every station even at the smaller scale"]
    assert rows["middle_arm"]["reasons"] == ["reach: reach depends on the scene's scale"]
    assert table["counts"] == {"fits": 0, "fits_spec": 1, "doesnt_fit": 1, "unknown": 1}
    inputs = table["inputs"]
    # Both points are the object's top plus the tool: where it starts and where it is set down.
    assert inputs["grasp_point_world_m"] == [0.5, 0.0, 0.25] and inputs["place_point_world_m"] == [0.5, 0.3, 0.25]
    assert inputs["scale_uncertainty_fraction"] == 0.25 and inputs["scale_band"] == [0.75, 1.333333]
    assert inputs["stations"] == {"source": "capability_verdict_stations", "positions_world_m": [[0.0, 0.0, 0.0]]}
    # A fixed arm is assumed on the existing station's pedestal; it has no access check.
    assert _check(rows["long_arm"], "stance")["values"]["mount"] == "station_pedestal_level_with_task_support"
    assert [check["check"] for check in rows["long_arm"]["checks"]] == ["stance", "reach", "payload", "cycle_time"]


@pytest.mark.parametrize("text, kilograms", [("5 kg", 5.0), ("12 lb", 5.44310844), ("about 3kg", 3.0),
                                              ("~2.5 KG", 2.5), ("500 g", 0.5), ("approx. 16 oz", 0.45359237),
                                              ("7 kilograms.", 7.0)])
def test_a_cleanly_stated_weight_parses(text, kilograms):
    assert owner_stated_weight_kg(text) == pytest.approx(kilograms)


@pytest.mark.parametrize("text", ["2-3 kg", "5", "heavy", "5 kg or 11 lb", "under 5 kg", "1,500 g", "5 kg each",
                                  "0 kg", "5 stone", "", None])
def test_an_ambiguous_weight_is_ignored(text):
    assert owner_stated_weight_kg(text) is None


def _payload(kilograms):
    return _figure(kilograms, "kg", f"Payload {kilograms:g} kg")


def test_payload_takes_the_owners_weight_with_a_band_else_the_estimate(tmp_path):
    stated = _table(tmp_path, sealed(*[_model(name, rated_payload_kg=_payload(kg))
                                       for name, kg in (("strong", 7), ("weak", 5), ("close", 6))]),
                    job=_job(weight="5 kg"))
    # 5 kg +-10% with the 0.9 kg tool needs 5.4 to 6.4 kg.
    assert stated["inputs"]["mass_basis"] == "owner_stated" and stated["inputs"]["mass_kg_range"] == [4.5, 5.5]
    assert stated["inputs"]["owner_item_weight"] == {"text": "5 kg", "parsed_kg": 5.0, "used": True}
    assert {name: _check(row, "payload")["result"] for name, row in _rows(stated).items()} == {
        "strong": "pass", "weak": "fail", "close": "unknown"}
    assert _rows(stated)["weak"]["verdict"] == "doesnt_fit"
    # Without a clean statement: the preparation's estimate, 0.2-1.0 kg at the measured volume. Mass
    # scales with volume, so across the 0.75-1.33x band it is 0.084-2.370 kg: 0.98-3.27 kg with the tool.
    catalog = sealed(*[_model(name, rated_payload_kg=_payload(kg)) for name, kg in (("strong", 4), ("weak", 0.9),
                                                                                    ("close", 2))])
    for job in (_job(), _job(weight="2-3 kg")):
        estimated = _table(tmp_path, catalog, job=job)
        assert estimated["inputs"]["mass_basis"] == "estimated"
        assert estimated["inputs"]["mass_kg_range"] == [0.084375, 2.37037]
        assert estimated["inputs"]["mass_estimate_kg"] == [0.2, 1.0]
        assert estimated["inputs"]["mass_widened_by"] == "scale_band_cubed"
        assert {name: _check(row, "payload")["result"] for name, row in _rows(estimated).items()} == {
            "strong": "pass", "weak": "fail", "close": "unknown"}
    assert estimated["inputs"]["owner_item_weight"] == {"text": "2-3 kg", "parsed_kg": None, "used": False}


@pytest.mark.parametrize("target, result", [(None, "not_requested"), (0.3, "fail"), (2.0, "unknown"),
                                            (20.0, "pass")])
def test_cycle_time_is_bounded_from_both_sides(tmp_path, target, result):
    table = _table(tmp_path, sealed(), job=_job(cycle=target))
    check = _check(table["rows"][0], "cycle_time")
    assert check["result"] == result and table["inputs"]["cycle_time_target_s"] == target
    if target is not None:
        # Two straight 0.225 m trips at 1 m/s take 0.45 s at best. At worst each 0.56 m carry (up 8 cm,
        # 0.4 m across, down 8 cm) runs at a tenth of the 1 m/s peak, plus 4 s: 15.2 s.
        assert check["values"]["lower_bound_s"] == pytest.approx(0.45, abs=1e-6)
        assert check["values"]["pessimistic_s"] == pytest.approx(15.2, abs=1e-6)
    assert table["rows"][0]["verdict"] == {"not_requested": "fits_spec", "pass": "fits_spec", "fail": "doesnt_fit",
                                           "unknown": "unknown"}[result]


def test_every_path_checked_profile_names_its_recipe():
    from blueprint_pipeline.robot_arm_path_clearance import ROBOT_PROFILE_ARMS
    from blueprint_pipeline.robot_spec_catalog import known_recipe_ids

    assert PROFILE_RECIPES == {DROID: FRANKA_RECIPE}
    assert set(PROFILE_RECIPES) == set(ROBOT_PROFILE_ARMS) and set(PROFILE_RECIPES.values()) <= known_recipe_ids()


def test_the_franka_takes_the_capability_verdicts_carry(tmp_path, monkeypatch):
    # A spec reach of 10 cm would fail: the verdict's carry decides for the arm it path-checks.
    catalog = validate_robot_spec_catalog(_catalog(_model("droid_franka", recipe_id=FRANKA_RECIPE,
                                                          max_reach_m=_reach(0.1))), recipe_ids={FRANKA_RECIPE})
    room = _room(tmp_path)
    job = _job(destination={"relation": "on", "position_world_m": [0.3, 0.35, 0.0]},
               subject=([0.2675, -0.0325, 0.0], [0.3325, 0.0325, 0.065]))
    clear = rigid_task_verdict(preparation=job, collision_path=room, robot_profiles=[DROID])
    assert clear["status"] == "clear" and len(clear["station_results"]) == 1
    table = robot_fit_table(preparation=job, collision_path=room, verdict=clear, catalog=catalog)
    row = table["rows"][0]
    assert row["verdict"] == "fits" and row["reasons"] == []
    assert _check(row, "reach")["result"] == "pass"
    assert _check(row, "reach")["values"]["decided_by"] == "capability_verdict_path_check"
    # The clear verdict stopped at its first station; the table searched them all once, the same way.
    stations = table["inputs"]["stations"]
    assert stations["source"] == "ground_stance_search" and len(stations["positions_world_m"]) > 1

    blocked = _rows(_table(tmp_path, catalog, carry="blocked", carry_blockers=[LIFT_BLOCKED], blockers=[LIFT_BLOCKED]))
    assert blocked["droid_franka"]["verdict"] == "doesnt_fit"
    assert _check(blocked["droid_franka"], "reach")["values"]["blockers"] == [LIFT_BLOCKED]
    assert blocked["droid_franka"]["reasons"] == [
        "reach: the capability verdict's carry was refused from every station tried: " + LIFT_BLOCKED]

    # No arm the verdict models was enabled: it never searched, so the table does, and the carry is unknown.
    monkeypatch.setattr(team_embodiment_stance, "ground_stance", lambda **_: [{"position_world_m": [0.0, 0.0, 0.0]}])
    unassessed = _rows(_table(tmp_path, catalog, status="unassessed", stations=(), blockers=[UNMODELED]))
    assert _check(unassessed["droid_franka"], "reach")["result"] == "unknown"
    assert unassessed["droid_franka"]["verdict"] == "unknown"
    # Unassessed only for another robot: the Franka's own carry was refused, so it does not fit.
    mixed = _rows(_table(tmp_path, catalog, status="unassessed", carry="blocked", carry_blockers=[LIFT_BLOCKED],
                         blockers=[LIFT_BLOCKED, UNMODELED]))
    assert mixed["droid_franka"]["verdict"] == "doesnt_fit"


def test_no_clear_station_is_unknown_never_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(team_embodiment_stance, "ground_stance",
                        lambda **_: pytest.fail("the verdict's failed station search is not repeated"))
    catalog = validate_robot_spec_catalog(_catalog(_model("tiny_arm", max_reach_m=_reach(0.1)),
                                                   _model("droid_franka", recipe_id=FRANKA_RECIPE)),
                                          recipe_ids={FRANKA_RECIPE})
    table = _table(tmp_path, catalog, stations=(), blockers=["robot_station_team_embodiment_free_base_no_clear_stance"])
    for row in table["rows"]:
        assert (_check(row, "stance")["result"], _check(row, "reach")["result"], row["verdict"]) == (
            "unknown", "unknown", "unknown")


def test_mobile_bases_and_humanoids_are_never_cleared_for_access(tmp_path):
    base_speed = _figure(1.0, "m/s", "Base speed 1 m/s")
    catalog = sealed(
        _model("fixed"),
        _model("small_mobile", "mobile_manipulator", max_reach_m=_reach(1.0), max_base_speed_m_s=base_speed,
               base_footprint_diameter_m=_figure(0.25, "m", "Footprint 250 mm", "250 mm = 0.25 m")),
        _model("large_mobile", "mobile_manipulator", max_reach_m=_reach(1.0), max_base_speed_m_s=base_speed,
               base_footprint_diameter_m=None,
               base_footprint_l_w_m=_figure([0.6, 0.5], "m", "600 x 500 mm", "600 mm = 0.6 m; 500 mm = 0.5 m")),
        _model("humanoid", "humanoid", max_base_speed_m_s=base_speed, base_footprint_diameter_m=None))
    rows = _rows(_table(tmp_path, catalog))
    for name in ("small_mobile", "large_mobile", "humanoid"):
        assert _check(rows[name], "access") == {"check": "access", "result": "unknown",
                                                "reason": "path from the entrance not checked", "values": {}}
        assert rows[name]["verdict"] == "unknown"
    assert "access" not in {check["check"] for check in rows["fixed"]["checks"]}
    assert rows["fixed"]["verdict"] == "fits_spec"
    # They stand at the same stations: a base inside the cleared 0.3 m square passes, a larger one is unknown.
    assert _check(rows["small_mobile"], "stance")["result"] == "pass"
    assert _check(rows["small_mobile"], "reach")["result"] == "pass"
    assert _check(rows["large_mobile"], "stance")["result"] == "unknown"
    assert _check(rows["humanoid"], "reach")["result"] == "unknown"


def test_a_fixed_arm_passes_stance_only_when_its_published_base_fits_the_station(tmp_path):
    rows = _rows(_table(tmp_path, sealed(
        _model("cobot"),
        _model("industrial", base_footprint_diameter_m=None,
               base_footprint_l_w_m=_figure([0.6, 0.6], "m", "600 x 600 mm", "600 mm = 0.6 m")),
        _model("unpublished", base_footprint_diameter_m=None))))
    # The standard pedestal clears a 0.3 m square: a 0.2 m base fits; a 0.6 m one or an unknown one is not known to.
    assert {name: _check(row, "stance")["result"] for name, row in rows.items()} == {
        "cobot": "pass", "industrial": "unknown", "unpublished": "unknown"}
    assert {name: row["verdict"] for name, row in rows.items()} == {
        "cobot": "fits_spec", "industrial": "unknown", "unpublished": "unknown"}
    assert _check(rows["unpublished"], "stance")["reason"] == "no base footprint figure to compare with the cleared station"


def test_the_table_is_bound_to_its_inputs_and_its_digest_is_stable(tmp_path):
    job, collision = _job(weight="2 kg", cycle=5.0), _collision(tmp_path)
    catalog, verdict = sealed(_model("arm_a"), _model("arm_b", max_reach_m=_reach(0.6))), _verdict(job, collision)
    table = robot_fit_table(preparation=job, collision_path=collision, verdict=verdict, catalog=catalog)
    again = robot_fit_table(preparation=copy.deepcopy(job), collision_path=collision, verdict=copy.deepcopy(verdict),
                            catalog=copy.deepcopy(catalog))
    assert again == table and json.loads(json.dumps(table, allow_nan=False)) == table
    assert table["table_digest"] == canonical_digest(table, digest_field="table_digest")
    assert {key: table[key] for key in ("schema_version", "preparation_digest", "collision_sha256", "catalog_digest",
                                        "verdict_digest", "evidence_label", "claim_ceiling",
                                        "physical_world_truth_claimed")} == {
        "schema_version": "website_robot_fit_table.v1", "preparation_digest": job["digest"],
        "collision_sha256": verdict["collision_sha256"], "catalog_digest": catalog["catalog_digest"],
        "verdict_digest": verdict["verdict_digest"], "evidence_label": "spec_bounded_estimate",
        "claim_ceiling": "development_only", "physical_world_truth_claimed": False}
    changed = sealed(_model("arm_a"), _model("arm_b", max_reach_m=_reach(0.7)))
    assert robot_fit_table(preparation=job, collision_path=collision, verdict=verdict,
                           catalog=changed)["table_digest"] != table["table_digest"]
    # A verdict for other inputs, or a catalog changed after it was validated, is refused, not mixed in.
    with pytest.raises(ValueError, match="website_robot_fit_verdict_binding_mismatch"):
        robot_fit_table(preparation=job, collision_path=collision, catalog=catalog,
                        verdict={**verdict, "preparation_digest": "sha256:" + "2" * 64})
    tampered = copy.deepcopy(catalog)
    tampered["models"][0]["figures"]["rated_payload_kg"]["value"] = 50
    with pytest.raises(ValueError, match="website_robot_fit_catalog_unsealed"):
        robot_fit_table(preparation=job, collision_path=collision, verdict=verdict, catalog=tampered)


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    """``prepare_rigid_native_inputs`` for an intake-ready rigid job: stage builders stubbed, a recorded
    verdict and a fixture catalog in place of the shipped one."""
    from blueprint_pipeline import robot_spec_catalog, website_native_background, website_object_observations

    for name in ("prepare_collision_stage", "prepare_appearance_stage"):
        monkeypatch.setattr(website_native_background, name, lambda path: {"stage": "stub"})
    monkeypatch.setattr(website_native_background, "prepare_construction_stages", lambda **_: {"references": []})
    monkeypatch.setattr(website_native_background, "construction_rights_admission", lambda **_: {"status": "stub"})
    monkeypatch.setattr(website_object_observations, "_record", lambda path: {"path": str(path)})
    monkeypatch.setattr(team_embodiment_stance, "ground_stance", lambda **_: [{"position_world_m": [0.0, 0.0, 0.0]}])
    catalog_path = tmp_path / "robot_spec_catalog.json"
    catalog_path.write_text(json.dumps(_catalog(_model("arm_a"))))
    monkeypatch.setattr(robot_spec_catalog, "CATALOG_PATH", catalog_path)
    collision, job = _collision(tmp_path), {**_job(), "status": "intake_ready"}

    def run(status):
        verdict = (_verdict(job, collision, "clear", carry="clear") if status == "clear" else
                   _verdict(job, collision, carry="blocked", carry_blockers=[LIFT_BLOCKED], blockers=[LIFT_BLOCKED]))
        monkeypatch.setattr("blueprint_pipeline.website_rigid_feasibility.rigid_task_verdict", lambda **_: verdict)
        result, native = {}, tmp_path / status / "native"
        admissible = prepare_rigid_native_inputs(
            result=result, preparation=job, preparation_path=tmp_path / "preparation.json", native_root=native,
            task_context={}, now=0.0, runtime_inputs={"status": "ready", "digest": "sha256:" + "3" * 64,
                                                      "appearance": {"status": "native_appearance_authored"},
                                                      "collision": {"path": str(collision)}})
        return admissible, result, native
    return run


def test_the_handoff_writes_the_table_beside_the_verdict(handoff):
    admissible, result, native = handoff("clear")
    table = json.loads((native / "robot_fit_table.json").read_text())
    assert admissible is True and (native / "capability_verdict.json").is_file()
    assert result["robot_fit_table"] == {"path": str(native / "robot_fit_table.json"),
                                         "digest": table["table_digest"], "counts": table["counts"]}
    assert table["table_digest"] == canonical_digest(table, digest_field="table_digest")
    assert table["counts"] == {"fits": 0, "fits_spec": 1, "doesnt_fit": 0, "unknown": 0}


@pytest.mark.parametrize("status", ["clear", "abstained"])
def test_a_failing_table_never_blocks_the_handoff_or_changes_its_gating(handoff, monkeypatch, status):
    def broken(**_):
        raise RuntimeError("table bug")

    with monkeypatch.context() as patch:
        patch.setattr("blueprint_pipeline.website_robot_fit.robot_fit_table", broken)
        failed_admissible, failed, native = handoff(status)
    assert failed["robot_fit_table"] == {"status": "failed", "error": "RuntimeError"}
    assert not (native / "robot_fit_table.json").exists()
    admissible, result, _ = handoff(status)
    assert "path" in result["robot_fit_table"]
    assert failed_admissible == admissible == (status == "clear")
    assert {key: value for key, value in failed.items() if key != "robot_fit_table"} == {
        key: value for key, value in result.items() if key != "robot_fit_table"}
    if status == "abstained":
        assert (result["status"], result["blockers"]) == ("needs_input", [LIFT_BLOCKED])


def test_a_broken_catalog_file_is_recorded_and_the_handoff_continues(handoff, monkeypatch, tmp_path):
    from blueprint_pipeline import robot_spec_catalog

    (tmp_path / "broken.json").write_text("{")
    monkeypatch.setattr(robot_spec_catalog, "CATALOG_PATH", tmp_path / "broken.json")
    admissible, result, _ = handoff("clear")
    assert admissible is True and result["robot_fit_table"] == {"status": "failed", "error": "RobotSpecCatalogError"}


def test_a_lost_handoff_lease_still_stops_the_handoff(handoff, monkeypatch):
    def lost(**_):
        raise HandoffLeaseError("lost", "shared_artifact_promotion")

    monkeypatch.setattr("blueprint_pipeline.website_robot_fit.robot_fit_table", lost)
    with pytest.raises(HandoffLeaseError):
        handoff("clear")
