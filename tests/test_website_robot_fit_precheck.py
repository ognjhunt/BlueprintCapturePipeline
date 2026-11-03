"""Before any spend, whether a robot Blueprint runs can do a website job in its room: fit, no fit or unknown.

Synthetic rooms only (a floor, a table, a tall shelf), with the shipped screening catalog or one derived
from it. The consistency of this early check with the late checks it must never be stricter than is in
``test_website_robot_fit_precheck_consistency``.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_robot_fit_precheck.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/robot_arm_path_clearance.py
#   src/blueprint_pipeline/beta_policy_screening_plan.py
#   src/blueprint_pipeline/beta_policy_screening_gpu.py
#   src/blueprint_pipeline/website_preparation_renewal.py

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline import website_robot_fit_precheck as precheck
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog, seal_catalog
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_robot_fit_precheck import HOLD_PREFIX, robot_fit_precheck
from tests.test_team_scene_packet import _box, _stage

FRANKA = precheck.FRANKA_PROFILE
G1 = precheck.G1_PROFILE
FLOOR = {"description": "floor", "aabb_min_xyz": [-3.0, -3.0, 0.0], "aabb_max_xyz": [3.0, 3.0, 0.0]}
# A dining table, 0.8 x 0.6 m and 0.75 m tall; a long one, 3.2 x 0.8 m; a shelf whose top is 2.4 m up.
TABLE = {"description": "table top", "aabb_min_xyz": [0.3, -0.3, 0.75], "aabb_max_xyz": [1.1, 0.3, 0.75]}
LONG_TABLE = {"description": "long table top", "aabb_min_xyz": [-1.6, -0.4, 0.75], "aabb_max_xyz": [1.6, 0.4, 0.75]}
SHELF = {"description": "shelf top", "aabb_min_xyz": [-1.5, 1.2, 2.4], "aabb_max_xyz": [-0.9, 1.6, 2.4]}
# A 20 x 15 x 3 cm wooden tray.
TRAY = ([0.42, -0.075, 0.75], [0.62, 0.075, 0.78])


def room(directory: Path, *furniture: tuple[str, tuple[float, ...], tuple[float, ...]]) -> Path:
    """A synthetic room: a floor and each named box standing on it."""
    directory.mkdir(parents=True, exist_ok=True)
    stage = _stage(directory / "collision.usda", "/Root")
    _box(stage, "/Root/Floor", (-3.0, -3.0, 0.0), (3.0, 3.0, 0.0))
    for name, low, high in furniture:
        _box(stage, "/Root/" + name, low, high)
    stage.GetRootLayer().Save()
    return directory / "collision.usda"


def job(subject, destination, support, *, strategy="pick_and_place", envelope=0.25) -> dict[str, Any]:
    """A prepared website job as ``compile_website_scene_preparation`` records the fields the precheck reads."""
    value = {"digest": "sha256:" + "1" * 64, "status": "intake_ready",
             "intake_request": {"task": {"strategy": strategy, "support": support,
                                         "success": {"pregrasp_clearance_m": 0.1}}},
             "subject": {"aabb_min_xyz": list(subject[0]), "aabb_max_xyz": list(subject[1])},
             "destination": destination,
             "authoring_inputs": {"metric_envelope": {"maximum_dimension_relative_error": envelope}}}
    if envelope is None:
        del value["authoring_inputs"]
    return value


def tray_scene(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """Pick up a wooden tray on a dining table and place it 30 cm along, at the table's other end."""
    collision = room(tmp_path / "tray", ("Table", (0.3, -0.3, 0.0), (1.1, 0.3, 0.75)))
    return job(TRAY, {"relation": "on", "position_world_m": [0.85, 0.0, 0.75], "support": TABLE}, TABLE), collision


def shelf_scene(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """Pick a box up off the floor and place it on a shelf 2.4 m up: taller than any arm on a floor mount reaches."""
    collision = room(tmp_path / "shelf", ("Shelf", (-1.5, 1.2, 0.0), (-0.9, 1.6, 2.4)))
    return job(([-0.1, -0.05, 0.0], [0.0, 0.05, 0.1]),
               {"relation": "on", "position_world_m": [-1.2, 1.4, 2.4], "support": SHELF}, FLOOR), collision


def long_table_scene(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """Carry the tray 2.7 m along a 3.2 m table: farther apart than any one station's arm spans."""
    collision = room(tmp_path / "long", ("Table", (-1.6, -0.4, 0.0), (1.6, 0.4, 0.75)))
    return job(([-1.45, -0.075, 0.75], [-1.25, 0.075, 0.78]),
               {"relation": "on", "position_world_m": [1.35, 0.0, 0.75], "support": LONG_TABLE}, LONG_TABLE), collision


def fixed_arm_catalog() -> dict[str, Any]:
    """The shipped catalog with the G1 lane switched off: only the DROID Franka's candidates stay enabled."""
    catalog = copy.deepcopy(load_default_catalog())
    for row in catalog["candidates"]:
        if (row.get("interface") or {}).get("robot_profile_id") == G1:
            row["eligibility"].update(enabled=False, blockers=["synthetic_g1_lane_off"])
    return seal_catalog(catalog)


def by_embodiment(value: dict[str, Any], key: str) -> dict[str, Any]:
    return value["embodiments"][key]


def test_the_franka_reach_bound_holds_at_every_sampled_joint_value():
    from blueprint_pipeline.robot_arm_path_clearance import FRANKA_JOINT_LIMITS_RAD, _arm_model, _frames, grasp_reach_bound

    shoulder, bound = grasp_reach_bound("franka_panda")
    _, grasp = _arm_model("franka_panda")
    low, high = (np.array([row[i] for row in FRANKA_JOINT_LIMITS_RAD]) for i in (0, 1))
    farthest = 0.0
    for joints in np.random.default_rng(7).uniform(low, high, size=(5000, 7)):
        frame = _frames(joints)[8]
        farthest = max(farthest, float(np.linalg.norm(frame[:3, 3] + frame[:3, :3] @ grasp - [0.0, 0.0, shoulder])))
    assert shoulder == pytest.approx(0.333) and farthest <= bound
    # Not a loose bound: within 10 cm of the farthest pose sampled.
    assert bound - farthest < 0.1


def test_the_reach_lower_bound_needs_no_collider():
    boxes = {"pick": ([0.0, 0.0, 0.0], [0.1, 0.1, 0.1]), "place": ([2.0, 0.0, 0.0], [2.1, 0.1, 0.1])}
    # Half the 1.9 m gap, at any height in the band.
    assert precheck.mount_reach_lower_bound(boxes, (0.0, 0.0)) == pytest.approx(0.95)
    # A box 2 m above the band is 2 m from any point in it.
    boxes["place"] = ([0.0, 0.0, 2.3], [0.1, 0.1, 2.4])
    assert precheck.mount_reach_lower_bound(boxes, (0.2, 0.3)) == pytest.approx(2.0)


def test_a_reachable_tray_on_a_table_fits(tmp_path):
    from blueprint_pipeline.website_rigid_feasibility import rigid_task_verdict

    preparation, collision = tray_scene(tmp_path)
    verdict = rigid_task_verdict(preparation=preparation, collision_path=collision)
    value = robot_fit_precheck(preparation=preparation, collision_path=collision, verdict=verdict)
    assert value["status"] == "fit" and value["hold_reason"] is None and value["task_kind"] == "rigid_pick_place"
    franka = by_embodiment(value, FRANKA)
    assert franka["verdict"] == "fit" and franka["carry"]["status"] == "clear"
    assert franka["reach"]["result"] == "not_ruled_out" and franka["reach"]["mount_z_m"] == 0.75
    # Every enabled DROID candidate whose GPU floor the lane can rent fits on the Franka.
    rows = {row["candidate_id"]: row for row in value["candidates"]}
    assert set(value["fit_candidates"]) == {key for key, row in rows.items() if row["embodiment"] == FRANKA
                                            and row["verdict"] == "fit"}
    assert "pi05_droid" in value["fit_candidates"]
    assert rows["cosmos3_nano_policy_droid"]["verdict"] == "no_fit"
    assert rows["cosmos3_nano_policy_droid"]["causes"] == ["gpu_memory"]
    # The G1 stands at the table; its nominal arm span (0.45 m) misses, which its lane never refuses.
    g1 = by_embodiment(value, G1)
    assert g1["verdict"] == "unknown" and g1["reasons"] == ["nominal_arm_span_missed_body_motion_may_reach"]
    assert {rows[key]["verdict"] for key in rows if rows[key]["embodiment"] == G1} == {"unknown"}
    assert value["counts"]["unavailable"] == sum(not row["eligibility"]["enabled"]
                                                 for row in load_default_catalog()["candidates"])
    assert value["precheck_digest"] == canonical_digest(value, digest_field="precheck_digest")
    assert value["physical_world_truth_claimed"] is False


def test_a_shelf_taller_than_any_reach_holds_a_fixed_arm_job_before_spend(tmp_path):
    preparation, collision = shelf_scene(tmp_path)
    value = robot_fit_precheck(preparation=preparation, collision_path=collision, catalog=fixed_arm_catalog())
    assert value["status"] == "no_fit" and value["hold_reason"] == HOLD_PREFIX + "reach"
    # Seven DROID candidates out of reach; the eighth's 80 GB floor no GPU the lane rents meets.
    assert value["no_fit_causes"] == {"gpu_memory": 1, "reach": 7}
    franka = by_embodiment(value, FRANKA)
    # From the floor mount, the shelf top's box is 2 m above the shoulder: beyond the arm from any station.
    assert franka["verdict"] == "no_fit" and franka["reach"]["result"] == "beyond_reach"
    assert franka["reach"]["lower_bound_m"] > franka["reach"]["reach_bound_m"] + precheck.REACH_MARGIN_M
    assert value["fit_candidates"] == [] and value["counts"]["unknown"] == 0
    assert G1 not in value["embodiments"]  # its candidates are off, so it is never measured


def test_the_same_shelf_with_the_g1_lane_on_is_unknown_and_goes_on(tmp_path):
    preparation, collision = shelf_scene(tmp_path)
    value = robot_fit_precheck(preparation=preparation, collision_path=collision)
    # The Franka is refused, but the G1 may walk, lean or crouch: its lane refuses no reach, so neither does this.
    assert value["status"] == "unknown" and value["hold_reason"] is None
    assert by_embodiment(value, FRANKA)["verdict"] == "no_fit"
    assert by_embodiment(value, G1)["verdict"] == "unknown" and by_embodiment(value, G1)["reach_refuses"] is False
    assert "unknown:nominal_arm_span_missed_body_motion_may_reach" in value["flags"]


def test_a_carry_longer_than_any_station_spans_is_refused_for_the_franka(tmp_path):
    preparation, collision = long_table_scene(tmp_path)
    value = robot_fit_precheck(preparation=preparation, collision_path=collision, catalog=fixed_arm_catalog())
    reach = by_embodiment(value, FRANKA)["reach"]
    assert reach["result"] == "beyond_reach" and value["hold_reason"] == HOLD_PREFIX + "reach"


@pytest.mark.parametrize("missing", ["metric_envelope", "destination_position", "subject_bounds"])
def test_a_missing_input_is_unknown_and_never_a_hold(tmp_path, missing):
    preparation, collision = shelf_scene(tmp_path)
    if missing == "metric_envelope":
        del preparation["authoring_inputs"]
    elif missing == "destination_position":
        del preparation["destination"]["position_world_m"]
    else:
        preparation["subject"]["aabb_max_xyz"] = None
    value = robot_fit_precheck(preparation=preparation, collision_path=collision, catalog=fixed_arm_catalog())
    # The same job with every input refuses the Franka; with one missing nothing is decided against it.
    assert value["status"] == "unknown" and value["hold_reason"] is None
    assert "input_missing:" + missing in value["flags"]
    assert by_embodiment(value, FRANKA) == {"embodiment": FRANKA, "verdict": "unknown",
                                            "reasons": ["task_inputs_missing"]}


def test_a_missing_collider_leaves_the_g1_unknown_and_the_reach_bound_still_decides(tmp_path):
    preparation, _ = shelf_scene(tmp_path)
    value = robot_fit_precheck(preparation=preparation, collision_path=tmp_path / "absent.usda")
    assert "input_missing:collider" in value["flags"] and value["collision_sha256"] is None
    assert by_embodiment(value, G1)["reasons"] == ["collider_unavailable"]
    assert by_embodiment(value, FRANKA)["verdict"] == "no_fit" and value["status"] == "unknown"


def test_an_articulated_job_is_read_by_task_kind_only(tmp_path):
    preparation, collision = tray_scene(tmp_path)
    preparation["intake_request"]["task"]["strategy"] = "articulated_open_close"
    value = robot_fit_precheck(preparation=preparation, collision_path=collision)
    rows = {row["candidate_id"]: row for row in value["candidates"]}
    assert value["task_kind"] == "articulated_open_close" and value["status"] == "unknown"
    # The G1 candidates and MolmoAct2 run rigid tasks only: the screening plan's own refusal.
    assert {rows[key]["blockers"][0] for key in rows if rows[key]["embodiment"] == G1} == {
        "task_kind_not_supported_by_candidate"}
    assert rows["molmoact2_droid"]["verdict"] == "no_fit" and rows["molmoact2_droid"]["causes"] == ["task_kind"]
    assert by_embodiment(value, FRANKA)["reasons"] == ["task_geometry_not_modelled_before_authoring"]
    assert G1 not in value["embodiments"]


def test_a_job_no_catalog_candidate_can_run_holds_with_its_cause(tmp_path):
    preparation, collision = tray_scene(tmp_path)
    catalog = copy.deepcopy(load_default_catalog())
    for row in catalog["candidates"]:
        row["eligibility"].update(enabled=False, blockers=row["eligibility"]["blockers"] or ["synthetic_off"])
    value = robot_fit_precheck(preparation=preparation, collision_path=collision, catalog=seal_catalog(catalog))
    assert value["status"] == "no_fit" and value["hold_reason"] == HOLD_PREFIX + "no_candidate_available"
    assert value["embodiments"] == {} and value["counts"]["unavailable"] == len(catalog["candidates"])


def test_a_verdict_for_another_preparation_or_collider_is_refused(tmp_path):
    preparation, collision = tray_scene(tmp_path)
    verdict = {"preparation_digest": "sha256:" + "9" * 64, "collision_sha256": None}
    with pytest.raises(ValueError, match="verdict_binding_mismatch"):
        robot_fit_precheck(preparation=preparation, collision_path=collision, verdict=verdict)


# The handoff: before registration, so before ARTISAN authoring and the GPU scene build.


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    """``prepare_rigid_native_inputs`` for an intake-ready job: stage builders stubbed, the verdict recorded."""
    from blueprint_pipeline import website_native_background, website_object_observations

    for name in ("prepare_collision_stage", "prepare_appearance_stage"):
        monkeypatch.setattr(website_native_background, name, lambda path: {"stage": "stub"})
    monkeypatch.setattr(website_native_background, "prepare_construction_stages", lambda **_: {"references": []})
    monkeypatch.setattr(website_native_background, "construction_rights_admission", lambda **_: {"status": "stub"})
    monkeypatch.setattr(website_object_observations, "_record", lambda path: {"path": str(path)})

    def run(preparation, collision, *, catalog=None, verdict_status="unassessed"):
        from blueprint_pipeline import beta_policy_candidate_catalog
        from blueprint_pipeline.website_scene_handoff import prepare_rigid_native_inputs

        if catalog is not None:
            monkeypatch.setattr(beta_policy_candidate_catalog, "load_default_catalog", lambda: catalog)
        verdict = {"schema_version": "website_rigid_task_capability_verdict.v1",
                   "preparation_digest": preparation["digest"],
                   "collision_sha256": "sha256:" + hashlib.sha256(collision.read_bytes()).hexdigest(),
                   "status": verdict_status, "blockers": ["robot_rigid_pick_place_unreachable:approach"],
                   "embodiments": [{"robot_id": "franka_panda", "status": "blocked", "grasp": "top_down",
                                    "blockers": ["robot_rigid_pick_place_unreachable:approach"]}]}
        verdict["verdict_digest"] = canonical_digest(verdict, digest_field="verdict_digest")
        monkeypatch.setattr("blueprint_pipeline.website_rigid_feasibility.rigid_task_verdict", lambda **_: verdict)
        monkeypatch.setattr("blueprint_pipeline.website_robot_fit.robot_fit_table", lambda **_: {
            "table_digest": "sha256:" + "5" * 64, "counts": {}})
        result, native = {"status": "intake_ready", "blockers": []}, collision.parent / "native"
        admissible = prepare_rigid_native_inputs(
            result=result, preparation=preparation, preparation_path=collision.parent / "preparation.json",
            native_root=native, task_context={}, now=0.0,
            runtime_inputs={"status": "ready", "digest": "sha256:" + "3" * 64,
                            "appearance": {"status": "native_appearance_authored"},
                            "collision": {"path": str(collision)}})
        return admissible, result, native
    return run


def test_the_handoff_holds_a_no_fit_job_before_it_is_registered(tmp_path, handoff):
    preparation, collision = shelf_scene(tmp_path)
    admissible, result, native = handoff(preparation, collision, catalog=fixed_arm_catalog())
    written = json.loads((native / precheck.FILENAME).read_text())
    # Not admissible: nothing is registered or enqueued, so no ARTISAN authoring and no GPU follow. A hold.
    assert admissible is False and result["status"] == "needs_input"
    assert result["blockers"] == [HOLD_PREFIX + "reach"]
    assert result["robot_fit_precheck"] == {"path": str(native / precheck.FILENAME),
                                            "digest": written["precheck_digest"], "status": "no_fit",
                                            "hold_reason": HOLD_PREFIX + "reach", "counts": written["counts"]}


def test_the_handoff_goes_on_when_the_precheck_is_unknown(tmp_path, handoff):
    preparation, collision = shelf_scene(tmp_path)
    admissible, result, _ = handoff(preparation, collision)
    assert admissible is True and result["status"] == "intake_ready" and result["blockers"] == []
    assert result["robot_fit_precheck"]["status"] == "unknown"


def test_a_missing_input_goes_on_through_the_handoff(tmp_path, handoff):
    preparation, collision = shelf_scene(tmp_path)
    del preparation["authoring_inputs"]
    admissible, result, _ = handoff(preparation, collision, catalog=fixed_arm_catalog())
    assert admissible is True and result["robot_fit_precheck"]["status"] == "unknown"


def test_a_failing_precheck_is_unknown_and_never_holds(tmp_path, handoff, monkeypatch):
    def broken(**_):
        raise RuntimeError("precheck bug")

    monkeypatch.setattr(precheck, "robot_fit_precheck", broken)
    preparation, collision = shelf_scene(tmp_path)
    admissible, result, _ = handoff(preparation, collision, catalog=fixed_arm_catalog())
    assert admissible is True and result["robot_fit_precheck"] == {"status": "unknown", "hold_reason": None,
                                                                   "error": "RuntimeError"}


def test_an_abstained_verdict_keeps_its_own_blockers_beside_the_hold(tmp_path, handoff):
    preparation, collision = shelf_scene(tmp_path)
    admissible, result, _ = handoff(preparation, collision, catalog=fixed_arm_catalog(), verdict_status="abstained")
    assert admissible is False
    assert result["blockers"] == ["robot_rigid_pick_place_unreachable:approach", HOLD_PREFIX + "reach"]


def test_a_lost_handoff_lease_stops_the_precheck(tmp_path, handoff, monkeypatch):
    from blueprint_pipeline.core.handoff_lease import HandoffLeaseError

    def lost(**_):
        raise HandoffLeaseError("lost", "shared_artifact_promotion")

    monkeypatch.setattr(precheck, "robot_fit_precheck", lost)
    preparation, collision = shelf_scene(tmp_path)
    with pytest.raises(HandoffLeaseError):
        handoff(preparation, collision)


# What the site reads: preparation status v6.

CONTEXT = "sha256:" + "c" * 64


def write_precheck(root: Path, value: dict[str, Any], *, context: str = CONTEXT) -> dict[str, Any]:
    """A handoff with its preparation and precheck, as the scene handoff leaves them."""
    from blueprint_pipeline.common import write_json

    work = root / "pipeline" / "website_scene_preparation"
    preparation = {"schema_version": "website_scene_preparation.v1", "binding": {"task_context_digest": context}}
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    write_json(work / "preparation.json", preparation)
    value = {**value, "preparation_digest": preparation["digest"], "precheck_digest": ""}
    value["precheck_digest"] = canonical_digest(value, digest_field="precheck_digest")
    path = work / "native" / precheck.FILENAME
    write_json(path, value)
    handoff = {"schema_version": "website_scene_handoff.v1", "status": "needs_input",
               "blockers": [value["hold_reason"]] if value["hold_reason"] else [],
               "scene_id": root.parent.parent.name, "capture_id": root.name,
               "preparation_path": str(work / "preparation.json"), "preparation_digest": preparation["digest"],
               "robot_fit_precheck": {"path": str(path), "digest": value["precheck_digest"],
                                      "status": value["status"], "hold_reason": value["hold_reason"]}}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    write_json(work / "handoff.json", handoff)
    return handoff


def capture(tmp_path: Path) -> Path:
    root = tmp_path / "storage" / "scenes" / "site-req1" / "captures" / "walkthrough-req1"
    root.mkdir(parents=True)
    return root


@pytest.fixture(scope="module")
def held(tmp_path_factory) -> dict[str, Any]:
    preparation, collision = shelf_scene(tmp_path_factory.mktemp("held"))
    return robot_fit_precheck(preparation=preparation, collision_path=collision, catalog=fixed_arm_catalog())


def test_the_site_reads_why_its_job_is_held(tmp_path, held):
    root = capture(tmp_path)
    write_precheck(root, held)
    value = precheck.read_robot_fit_precheck(capture_root=root, task_context_digest=CONTEXT)
    assert value["status"] == "no_fit" and value["hold_reason"] == HOLD_PREFIX + "reach"
    assert value["embodiments"] == [{"embodiment": FRANKA, "verdict": "no_fit", "reasons": ["reach"],
                                     "fit_candidates": 0}]
    assert value["counts"] == held["counts"] and value["evidence_label"] == "pre_spend_estimate"
    assert len(json.dumps(value, separators=(",", ":")).encode()) <= precheck.MAX_PROJECTION_BYTES


def test_every_precheck_projection_matches_the_published_schema(tmp_path, held):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).resolve().parents[1]
                         / "docs/schemas/website_robot_fit_precheck_projection.v1.schema.json").read_text())
    preparation, collision = tray_scene(tmp_path)
    fit = robot_fit_precheck(preparation=preparation, collision_path=collision)
    for name, value in (("held", held), ("fit", fit)):
        root = capture(tmp_path / name)
        write_precheck(root, value)
        jsonschema.validate(precheck.read_robot_fit_precheck(capture_root=root, task_context_digest=CONTEXT), schema)


@pytest.mark.parametrize("fault", ["other_context", "edited", "reference_digest", "no_reference", "other_preparation"])
def test_a_precheck_that_does_not_bind_to_this_job_is_left_out(tmp_path, held, fault):
    from blueprint_pipeline.common import write_json

    root = capture(tmp_path)
    handoff = write_precheck(root, held, context="sha256:" + "d" * 64 if fault == "other_context" else CONTEXT)
    work = root / "pipeline" / "website_scene_preparation"
    path = Path(handoff["robot_fit_precheck"]["path"])
    if fault == "edited":
        value = json.loads(path.read_text())
        value["status"] = "fit"
        write_json(path, value)
    elif fault == "other_preparation":
        value = json.loads(path.read_text())
        value["preparation_digest"] = "sha256:" + "e" * 64
        value["precheck_digest"] = canonical_digest(value, digest_field="precheck_digest")
        write_json(path, value)
    elif fault in {"reference_digest", "no_reference"}:
        if fault == "no_reference":
            handoff.pop("robot_fit_precheck")
        else:
            handoff["robot_fit_precheck"]["digest"] = "sha256:" + "f" * 64
        handoff["digest"] = canonical_digest(handoff, digest_field="digest")
        write_json(work / "handoff.json", handoff)
    assert precheck.read_robot_fit_precheck(capture_root=root, task_context_digest=CONTEXT) is None
