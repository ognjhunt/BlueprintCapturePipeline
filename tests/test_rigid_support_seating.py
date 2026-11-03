"""A rigid task starts where its object is spawned, and its goal height is read on the scene's own collider.

Beta journey step: L5 team policy evaluation. Capture-447ae4f4 (wooden tray, "the
other end of the same table"): its sealed start was the source estimate's centre,
0.8244 m; the tray spawned with its scoring frame at 0.8391 m (14.7 mm high, the
scorer reads the first sample within 2 mm), because it was aligned by the estimate's
bottom instead of its own qualified collision bottom; and its goal disc sat 12.7 mm
below the registered support top. These fixtures reproduce each offset with the
job's own numbers and show the fix; no scorer tolerance changes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline import robot_free_readiness_retention as retention
from blueprint_pipeline import task_evaluation_scene_rigid_robot_free_readiness as rigid
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.rigid_robot_free_probe_plan import (
    RELEASE_CLEARANCE_M,
    physics_plan,
    probe_context,
)
from blueprint_pipeline.rigid_robot_free_probe_recorder import record_rigid_robot_free_probe
from blueprint_pipeline.rigid_support_seating import (
    footprint_top_z,
    start_spawn_disagreement_m,
    start_spawn_mismatch,
    triangle_support_surface,
)
from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import (
    _goal_support_reading,
    _runtime_geometry,
)
from tests.test_rigid_robot_free_probe import FlatSupport, binding, rigid_case

# Capture-447ae4f4's sealed geometry, as its probe input carried it on 2026-11-03.
SUPPORT_TOP = 0.8166057543686343  # the registered support plane's top: the highest face of the whole table
START = [-0.4326211349137302, -0.3030958365897039, 0.8243587329963455]  # the source estimate's centre
SOURCE_Z = (0.8018946653229508, 0.8468228006697404)
HALF_XY = (0.15668499469757083, 0.260563999414444)
COM = [0.0, 0.0, 0.010372125543653965]  # the replacement's scoring frame above its root
COLLISION_Z = (0.0, 0.04492800179123879)  # the replacement's qualified collision bounds, root frame
TARGET = [-0.22765085348616906, 0.09984717917168634, 0.8142599161339986]
DISC_Z = 0.8038877905903447
IDENTITY = {"id": "website-subject-53e1ef5c8bf71f6d840a", "version": "v1"}


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value = {**value, field: ""}
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _geometry(*, qualified_bounds: bool = True, support_surface: Any = None) -> dict[str, Any]:
    observed: dict[str, Any] = {"center_of_mass_m": COM, "rigid_body_paths": ["/Asset/wooden_tray"]}
    if qualified_bounds:
        observed["collision_bounds_body_frame_m"] = {"minimum": [-HALF_XY[0], -HALF_XY[1], COLLISION_Z[0]],
                                                     "maximum": [HALF_XY[0], HALF_XY[1], COLLISION_Z[1]]}
    return _runtime_geometry(
        replacement_identity=IDENTITY,
        support={"status": "frozen_candidate_pending_production_validation", "sage_prim_path": "/Root",
                 "top_z_m": SUPPORT_TOP, "bounds_min_xyz_m": [-0.91, -0.47, 0.7657],
                 "bounds_max_xyz_m": [0.29, 1.06, SUPPORT_TOP]},
        source_object={"status": "frozen_before_scene_configuration_run", "center_xyz_m": START,
                       "aabb_min_xyz_m": [START[0] - HALF_XY[0], START[1] - HALF_XY[1], SOURCE_Z[0]],
                       "aabb_max_xyz_m": [START[0] + HALF_XY[0], START[1] + HALF_XY[1], SOURCE_Z[1]]},
        static=_sealed({"status": "authored_structure_statically_qualified", "replacement_identity": IDENTITY,
                        "observed_structure": observed}, "result_digest"),
        native_import=_sealed({"status": "qualified", "replacement_identity": IDENTITY,
                               "native_simulator_import_qualified": True, "blockers": []}, "result_digest"),
        start=START, target=TARGET, strategy="pick_and_place", support_surface=support_surface,
        search_below_m=0.07203071739327313)


def _plan(start: list[float], root_z: float) -> dict[str, Any]:
    """The two plan fields the scorer's reset check joins: the task's start and the object it spawns."""

    return {"task_spec": {"task_kind": "rigid_pick_place", "start_pose_world": [*start, 0, 0, 0, 1],
                          "reset_translation_tolerance_m": 0.002,
                          "interaction_affordance": {"asset_root_from_scoring_frame": {
                              "position_m": COM, "orientation_xyzw": [0, 0, 0, 1]}}},
            "objects": [{"task_subject": False, "pose_world": {}},
                        {"task_subject": True, "reset_state": {"root_pose_world": {
                            "position_world_m": [START[0], START[1], root_z], "orientation_xyzw": [0, 0, 0, 1]}}}]}


def _slab(z: float, *, low=(-1.0, -1.0), high=(1.0, 1.0)) -> np.ndarray:
    (x0, y0), (x1, y1) = low, high
    return np.array([[[x0, y0, z], [x1, y0, z], [x1, y1, z]], [[x0, y0, z], [x1, y1, z], [x0, y1, z]]])


def test_the_sealed_alignment_spawned_capture_447s_tray_14_7_mm_from_its_start() -> None:
    # Before: the root was aligned by the estimate's bottom, 22.5 mm under its centre, not the replacement's own.
    legacy_root = SUPPORT_TOP - (COM[2] + SOURCE_Z[0] - START[2])
    assert legacy_root == pytest.approx(0.828697696498375)
    assert start_spawn_disagreement_m(_plan(START, legacy_root)) == pytest.approx(0.0147, abs=1e-4)
    assert start_spawn_mismatch(_plan(START, legacy_root)) == "rigid_task_start_pose_not_spawn_pose:14.7mm>2.0mm"
    # After: its qualified collision bottom (root z 0) sits on the registered top, and the start is that spawn.
    geometry = _geometry()
    assert geometry["root_position_world_m"][2] == pytest.approx(SUPPORT_TOP)
    assert geometry["start_scoring_position_world_m"] == pytest.approx([*START[:2], SUPPORT_TOP + COM[2]])
    assert geometry["start_scoring_position_world_m"][2] - START[2] == pytest.approx(0.0026, abs=1e-4)
    assert geometry["support_alignment"]["replacement_minimum_basis"] == "qualified_collision_bounds"
    seated = _plan(geometry["start_scoring_position_world_m"], geometry["root_position_world_m"][2])
    assert start_spawn_disagreement_m(seated) == pytest.approx(0.0, abs=1e-12)
    assert start_spawn_mismatch(seated) is None
    # A replacement without qualified bounds keeps the estimate's bottom, and still starts where it spawns.
    estimate = _geometry(qualified_bounds=False)
    assert estimate["root_position_world_m"][2] == pytest.approx(legacy_root)
    assert estimate["start_scoring_position_world_m"][2] == pytest.approx(legacy_root + COM[2])
    assert estimate["support_alignment"]["replacement_minimum_basis"] == "source_bounds_estimate"


def test_read_on_the_collider_the_tray_starts_where_it_rests_under_its_own_footprint() -> None:
    # The table under the tray: its fitted surface (0.8019) with one bump at 0.8102 under the footprint,
    # a taller one just outside it, and a shelf above the tray that never supports it.
    root_xy = START[:2]
    bump = np.array([[[root_xy[0] + 0.05, root_xy[1], 0.8019], [root_xy[0] + 0.08, root_xy[1], 0.8019],
                      [root_xy[0] + 0.06, root_xy[1] + 0.03, 0.8102]]])
    outside = bump + np.array([0.4, 0.0, 0.004])
    shelf = _slab(1.2)
    surface = triangle_support_surface(np.concatenate([_slab(0.8019), bump, outside, shelf]))
    geometry = _geometry(support_surface=surface)
    reading = geometry["support_alignment"]["start_support"]
    assert reading["collider_top_z_m"] == pytest.approx(0.8102)
    assert reading["basis"] == "scene_collider_highest_point_under_collision_footprint"
    assert geometry["root_position_world_m"][2] == pytest.approx(0.8102)
    assert geometry["start_scoring_position_world_m"][2] == pytest.approx(0.8102 + COM[2])
    assert start_spawn_mismatch(_plan(geometry["start_scoring_position_world_m"],
                                      geometry["root_position_world_m"][2])) is None
    # Nothing under the footprint in the window: the registered top, by name.
    empty = _geometry(support_surface=triangle_support_surface(shelf))
    assert empty["support_alignment"]["start_support"]["reason"] == "no_collider_point_under_footprint_in_window"
    assert empty["root_position_world_m"][2] == pytest.approx(SUPPORT_TOP)


def test_the_highest_collider_point_over_a_footprint_is_exact_for_its_triangles() -> None:
    # A ramp crossing the rectangle's edge: its highest point over the rectangle is on that edge.
    ramp = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 2.0, 0.0]]])
    assert footprint_top_z(ramp, (0.1, 0.1), (0.5, 0.5), floor_z=-1, ceiling_z=5) == pytest.approx(0.5)
    # A rectangle wholly inside one tilted triangle: its highest corner.
    assert footprint_top_z(ramp, (0.1, 0.1), (0.3, 0.2), floor_z=-1, ceiling_z=5) == pytest.approx(0.3)
    # A spike vertex inside the rectangle.
    spike = np.array([[[0.2, 0.2, 0.9], [3.0, 3.0, 0.0], [3.0, -3.0, 0.0]]])
    assert footprint_top_z(spike, (0.1, 0.1), (0.5, 0.5), floor_z=-1, ceiling_z=5) == pytest.approx(0.9)
    # Above the window (clear of the object) or below it (another surface) is not read.
    assert footprint_top_z(spike, (0.1, 0.1), (0.5, 0.5), floor_z=-1, ceiling_z=0.5) is None
    assert footprint_top_z(_slab(0.1), (0.1, 0.1), (0.5, 0.5), floor_z=0.2, ceiling_z=5) is None
    assert footprint_top_z(spike, (5.0, 5.0), (6.0, 6.0), floor_z=-1, ceiling_z=5) is None


@pytest.mark.parametrize(("goal_surface_z", "within"), [(DISC_Z, True), (SUPPORT_TOP, False)])
def test_capture_447s_goal_is_read_against_its_sealed_window_which_is_never_widened(goal_surface_z, within) -> None:
    spec = {"target_position_world_m": TARGET, "destination_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "support_height_interval_m": [TARGET[2] - 0.005, TARGET[2] + 0.005],
            "destination_position_bounds_world_m": {"minimum": [TARGET[0] - 0.05, TARGET[1] - 0.05, TARGET[2] - 0.005],
                                                    "maximum": [TARGET[0] + 0.05, TARGET[1] + 0.05, TARGET[2] + 0.005]},
            "subject_collision_bounds_scoring_frame_m": {
                "minimum": [-HALF_XY[0], -HALF_XY[1], COLLISION_Z[0] - COM[2]],
                "maximum": [HALF_XY[0], HALF_XY[1], COLLISION_Z[1] - COM[2]]}}
    before = copy.deepcopy(spec)
    reading = _goal_support_reading(triangle_support_surface(_slab(goal_surface_z)), task_spec=spec,
                                    collision=None, center=COM, support_top=SUPPORT_TOP, below_m=0.072)
    assert reading["rest_scoring_z_m"] == pytest.approx(goal_surface_z + COM[2])
    assert reading["success_window_scoring_z_m"] == pytest.approx([TARGET[2] - 0.005, TARGET[2] + 0.005])
    assert reading["within_success_window"] is within
    assert spec == before  # recorded, never applied: the sealed window stays the contract's
    if not within:
        # On the registered top the tray rests 12.7 mm above the disc, 7.7 mm past the window.
        assert reading["rest_scoring_z_m"] - (TARGET[2] + 0.005) == pytest.approx(0.0077, abs=1e-4)


class _OffsetSupport(FlatSupport):
    """The scene rests the object ``offset`` from the task's start (where it is untouched), and at the goal."""

    def __init__(self, start: list[float], *, start_offset: float = 0.0, goal: list[float] | None = None,
                 goal_offset: float = 0.0):
        super().__init__(start)
        self.start_z, self.start_offset = start[2], start_offset
        self.goal, self.goal_offset = goal, goal_offset

    def _ground(self) -> float:
        if self.goal is not None and abs(self.pose[0] - self.goal[0]) < 0.03:
            return self.start_z + self.goal_offset
        return self.start_z + self.start_offset


def test_an_episode_the_scorer_cannot_read_is_no_verdict_and_stays_replaceable(tmp_path) -> None:
    configured, _references, _docs, spec = rigid_case(tmp_path)
    plan = physics_plan(probe_context(spec))
    # The object rests 14.7 mm below the task's start: no episode can be read from it.
    scene = _OffsetSupport(spec["start_pose_world"], start_offset=-0.0147)
    trace = record_rigid_robot_free_probe(scene=scene, plan=plan, binding=binding(spec, configured["revision_digest"]))
    receipt = rigid.score_rigid_robot_free_readiness(task_spec=spec, physics_probe=trace,
                                                     configured_scene_revision_digest=configured["revision_digest"])
    assert receipt["status"] == "probe_unavailable", receipt.get("checks")
    assert receipt["probe_error"] == "physics_episode_unscorable:rigid_task_reset_readback_mismatch"
    assert receipt["learned_policy_evaluation_admitted"] is False
    assert spec["reset_translation_tolerance_m"] == 0.002  # unchanged
    # Undecided: a corrected scene's passing receipt replaces it; a decided one never would.
    root = tmp_path / "launch"
    root.mkdir()
    assert retention.retain_receipt(root, receipt) == "written"
    passing = record_rigid_robot_free_probe(scene=FlatSupport(spec["start_pose_world"]), plan=plan,
                                            binding=binding(spec, configured["revision_digest"]))
    passed = rigid.score_rigid_robot_free_readiness(task_spec=spec, physics_probe=passing,
                                                    configured_scene_revision_digest=configured["revision_digest"])
    assert passed["status"] == "passed"
    assert retention.retain_receipt(root, passed) == "replaced_unavailable"


def test_a_correct_placement_outside_the_sealed_goal_window_is_a_decided_failure(tmp_path) -> None:
    configured, _references, _docs, spec = rigid_case(tmp_path)
    window = spec["support_height_interval_m"]
    goal = list(spec["target_position_world_m"])
    plan = physics_plan(probe_context(spec))
    # The goal's surface stands 12.7 mm above the region the success document sealed.
    scene = _OffsetSupport(spec["start_pose_world"], goal=goal, goal_offset=goal[2] + 0.0127 - spec["start_pose_world"][2])
    trace = record_rigid_robot_free_probe(scene=scene, plan=plan, binding=binding(spec, configured["revision_digest"]))
    receipt = rigid.score_rigid_robot_free_readiness(task_spec=spec, physics_probe=trace,
                                                     configured_scene_revision_digest=configured["revision_digest"])
    assert receipt["status"] == "failed" and receipt["checks"]["direct_placement_success"] is False
    assert "support_height" in receipt["physics"]["direct_placement"]["failed_criteria"]
    assert spec["support_height_interval_m"] == window  # the window is the contract's, never widened


def test_the_virtual_carrier_lowers_onto_the_measured_goal_surface(tmp_path) -> None:
    _configured, _references, _docs, spec = rigid_case(tmp_path)
    target = spec["target_position_world_m"][2]
    assert physics_plan(probe_context(spec))["release_scoring_pose_world"][2] == pytest.approx(
        target + RELEASE_CLEARANCE_M)
    measured = copy.deepcopy(spec)
    measured["interaction_affordance"]["support_alignment"]["goal_support"] = {"rest_scoring_z_m": target + 0.004}
    assert physics_plan(probe_context(measured))["release_scoring_pose_world"][2] == pytest.approx(
        target + 0.004 + RELEASE_CLEARANCE_M)


def test_a_compiled_plan_that_spawns_its_object_off_its_start_is_named(tmp_path) -> None:
    plan = json.loads(json.dumps(_plan(START, SUPPORT_TOP - (COM[2] + SOURCE_Z[0] - START[2]))))
    assert start_spawn_mismatch(plan).startswith("rigid_task_start_pose_not_spawn_pose:14.7mm")
    plan["task_spec"]["task_kind"] = "articulated_open_close"
    assert start_spawn_mismatch(plan) is None
    assert start_spawn_mismatch({"task_spec": {}}) is None
    assert start_spawn_disagreement_m({"task_spec": {"task_kind": "rigid_pick_place"}, "objects": []}) is None
    del tmp_path


# --- end to end: the team packet reads the room's own collider --------------------------------------

BUMP_TOP = 0.7575  # 3 mm above the fixture room's registered support (0.7545), under the start only


def _bumped_bundle(tmp_path: Path) -> Path:
    """The fixture room (tests/test_team_scene_packet.py) with one bump on its support under the start."""

    import zipfile

    from pxr import UsdPhysics

    from tests.test_team_scene_packet import SUPPORT_TOP_Z, _box, _sha, _stage

    room = _stage(tmp_path / "room.usda", "/Root")
    _box(room, "/Root/Floor", (-2.0, -14.0, 0.0), (9.0, 4.0, 0.0))
    _box(room, "/Root/Support", (2.5, -9.5, 0.0), (4.5, -1.4, SUPPORT_TOP_Z))
    _box(room, "/Root/Bump", (2.95, -6.77, SUPPORT_TOP_Z), (2.97, -6.75, BUMP_TOP))
    room.GetRootLayer().Save()
    subject = _stage(tmp_path / "replacement.usda", "/Asset")
    UsdPhysics.RigidBodyAPI.Apply(subject.GetDefaultPrim())
    UsdPhysics.MassAPI.Apply(subject.GetDefaultPrim()).CreateMassAttr(0.3)
    _box(subject, "/Asset/body", (-0.0638749, -0.0658936, 0.0), (0.0638749, 0.0658936, 0.127638))
    subject.GetRootLayer().Save()
    payloads = {"appearance.usda": (tmp_path / "room.usda").read_bytes(),
                "collision.usda": (tmp_path / "room.usda").read_bytes(),
                "replacement.usda": (tmp_path / "replacement.usda").read_bytes()}
    rows = [{"role": name.split(".")[0], "relative_path": name, "digest": _sha(data), "size_bytes": len(data)}
            for name, data in payloads.items()]
    manifest = _sealed({"schema_version": "task_evaluation_configured_scene_bundle_candidate.v1",
                        "status": "assembled_pending_control_plane_publication", "robot_neutral": True,
                        "robot_specific_base_registration_included": False, "assets": rows}, "manifest_digest")
    path = tmp_path / "configured-scene.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr("configured_scene_bundle_candidate.v1.json", json.dumps(manifest))
    return path


def test_a_team_packet_seats_its_object_on_the_rooms_collider_and_reads_its_goal(tmp_path, monkeypatch) -> None:
    pytest.importorskip("pxr")
    import tests.test_team_scene_packet as packet_tests
    from blueprint_pipeline.adp_articulated_task_success_contract import seal_task_success_contract
    from tests.test_task_evaluation_rigid_relocation_native_adapter import STATIC, _rewrite

    original_case = packet_tests.rigid_case

    def with_qualified_bounds(path: Path, **kwargs: Any) -> tuple[dict, dict, dict, dict]:
        configured, references, docs, spec = original_case(path, **kwargs)
        static = copy.deepcopy(docs[STATIC])
        static["observed_structure"]["collision_bounds_body_frame_m"] = {
            "minimum": [-0.0638749, -0.0658936, 0.0], "maximum": [0.0638749, 0.0658936, 0.127638]}
        static["result_digest"] = canonical_digest({**static, "result_digest": ""}, digest_field="result_digest")
        _rewrite(tmp_path=path, configured=configured, references=references, contract_path=STATIC, document=static)
        return configured, references, {**docs, STATIC: static}, spec

    monkeypatch.setattr(packet_tests, "rigid_case", with_qualified_bounds)
    monkeypatch.setattr(packet_tests, "_bundle", _bumped_bundle)
    compiled, request = packet_tests._compile(tmp_path, packet_tests._scene(tmp_path))
    spec = request["task_spec"]
    (task,) = [row for row in request["assets"] if row["semantic_role"] == "task_object"]
    center_z = spec["interaction_affordance"]["asset_root_from_scoring_frame"]["position_m"][2]
    # Spawned resting on the bump under its footprint, and the episode starts exactly there.
    assert task["pose_world"]["position_world_m"][2] == pytest.approx(BUMP_TOP)
    assert spec["start_pose_world"][2] == pytest.approx(BUMP_TOP + center_z)
    plan = json.loads((Path(compiled["path"]).parent / "native-task-packet" /
                       "native_task_arena_scene_plan.v1.json").read_text())
    assert start_spawn_disagreement_m(plan) == pytest.approx(0.0, abs=1e-9)
    record = json.loads(Path(compiled["path"]).read_text())["task_support_seating"]
    assert record["status"] == "measured_on_scene_collider" and "flag" not in record and record["applied"] is True
    assert record["start"]["collider_top_z_m"] == pytest.approx(BUMP_TOP)
    # The goal footprint is clear of the bump: a correct push rests inside the sealed window.
    assert record["goal"]["collider_top_z_m"] == pytest.approx(packet_tests.SUPPORT_TOP_Z)
    assert record["goal"]["within_success_window"] is True
    # Seating moved the start only: the success contract is the one the unseated start gives.
    unseated = copy.deepcopy(plan["task_spec"])
    unseated["start_pose_world"][2] = 0.818319
    contract = json.loads(Path(compiled["path"]).read_text())["task_success_contract"]
    assert contract["contract_digest"] == seal_task_success_contract(
        task_kind="rigid_pick_place", task_spec=unseated, site_id=str(plan["scene_id"]), task_id=str(plan["task_id"]),
        author_source="compatibility_default", author_id="blueprint:manipulation_strategy_defaults.v1",
        confirmation_status="confirmed")["contract_digest"]
