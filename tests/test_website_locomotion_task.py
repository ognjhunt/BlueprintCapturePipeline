from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_profile_contracts import task_profile_for
from blueprint_pipeline.task_spec_compiler import requirements
from blueprint_pipeline.task_evaluation_embodiment_registry import EMBODIMENT_REGISTRY
from blueprint_pipeline.website_locomotion_task import (
    BUILT_NAVIGATION_PIECES,
    HOLD_CODE,
    compile_locomotion_spec,
    obstacle_distance,
    policy_requirements_served,
    runnable_options,
    runnable_policy_gap,
    scene_facts,
)
from blueprint_pipeline.website_task_context_contracts import holds_native_task_preparation

trimesh = pytest.importorskip("trimesh")

WALK = requirements(locomote=True, manipulate=False)


def _corridor(tmp_path):
    """A levelled Y-up room: a floor, two long obstacles either side of a lane and one across its end."""
    floor = trimesh.creation.box(extents=[8.0, 0.02, 4.0])
    floor.apply_translation([2.5, -0.01, 0.0])
    left = trimesh.creation.box(extents=[5.5, 2.0, 0.3])
    left.apply_translation([2.75, 1.0, 0.8])
    right = left.copy()
    right.apply_translation([0.0, 0.0, -1.6])
    end = trimesh.creation.box(extents=[0.3, 2.0, 1.9])
    end.apply_translation([5.65, 1.0, 0.0])
    glb = tmp_path / "collider.glb"
    glb.write_bytes(trimesh.Scene([floor, left, right, end]).export(file_type="glb"))
    base = {"collision_mesh_path": str(glb), "collision_mesh_digest": "sha256:" + hashlib.sha256(glb.read_bytes()).hexdigest(),
            "meters_per_unit": 1.0, "scale_authority": "atlas_reference_estimate", "levelling": {"tilt_degrees": 0.4}}
    # The camera walked the lane at eye height and stopped 0.2 m short of the end obstacle.
    path = [[x, 1.5, 0.0] for x in np.arange(0.0, 5.26, 0.25)]
    return base, path


def _context(family: str = "navigation") -> dict:
    context = {"schema_version": "website_site_task_context.v1", "scene_id": "site-a", "capture_id": "walkthrough-a",
               "description": "Walk down the lane to its far end without touching either side.", "task_family": family}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    return context


def test_navigation_is_a_profile_that_needs_locomotion_and_no_manipulation() -> None:
    profile = task_profile_for("navigation")
    assert profile.profile_id == "goal_region_navigation"
    assert profile.requires_locomotion and not profile.requires_manipulation
    assert not task_profile_for("rigid_pick_place").requires_locomotion


def test_scene_facts_read_floor_obstacles_and_route_from_the_collider(tmp_path) -> None:
    base, path = _corridor(tmp_path)
    facts = scene_facts(base_scene=base, camera_path_levelled_m=path)
    assert facts["floor_height_world_m"] == pytest.approx(0.0, abs=0.02)
    # Stage is Z up: the walked lane runs along +x at height 1.5.
    assert facts["filmed_path_world_m"][0] == pytest.approx([0.0, 0.0, 1.5])
    assert facts["obstacle_triangle_count"] > 0
    assert facts["metric_scale"]["status"] == "provider_declared_not_independently_validated"
    obstacles = facts["_obstacles_xy"]
    assert obstacle_distance([2.0, 0.0], obstacles) == pytest.approx(0.65, abs=0.01)
    assert obstacle_distance([5.25, 0.0], obstacles) == pytest.approx(0.25, abs=0.01)
    base["collision_mesh_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="website_locomotion_collider_changed"):
        scene_facts(base_scene=base, camera_path_levelled_m=path)


def test_a_vertical_face_is_a_segment_not_its_whole_line() -> None:
    # A wall's triangle projects onto a line segment; points on that line beyond it are not inside.
    wall = np.array([[[0.0, 10.0], [0.0, 11.0], [0.0, 10.0]]])
    assert obstacle_distance([0.0, 0.0], wall) == pytest.approx(10.0)
    assert obstacle_distance([0.0, 10.5], wall) == pytest.approx(0.0)
    assert obstacle_distance([0.3, 10.5], wall) == pytest.approx(0.3)
    floor_tile = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    assert obstacle_distance([0.2, 0.2], floor_tile) == 0.0


def test_the_goal_is_pulled_back_until_a_robot_can_stand_clear_and_every_guess_is_flagged(tmp_path) -> None:
    base, path = _corridor(tmp_path)
    spec = compile_locomotion_spec(task_context=_context(), facts=scene_facts(base_scene=base, camera_path_levelled_m=path),
                                   requirements=WALK)
    goal = spec["success"]["predicates"][0]
    assert goal["kind"] == "region_reached" and goal["subject"] == {"kind": "robot_root"}
    assert goal["region"]["center_world_m"][0] == pytest.approx(5.0)
    assert goal["region"]["center_world_m"][2] == pytest.approx(0.0, abs=0.02)
    assert spec["goal_source"] == "filmed_path_cleared_of_obstacles"
    assert {"goal_from_filmed_path", "goal_pulled_back_for_obstacle_clearance", "start_from_filmed_path",
            "metric_scale_provider_declared_not_independently_validated"} <= set(spec["flags"])
    assert [row["kind"] for row in spec["success"]["predicates"]] == ["region_reached", "height_stable", "no_contact"]
    assert spec["task_id"] == "website-" + _context()["context_digest"][7:27]


def test_bodies_and_policies_are_chosen_by_requirement_from_data() -> None:
    options = runnable_options(WALK)
    assert {row["embodiment_id"] for row in options["embodiments"]} == {
        row.embodiment_id for row in EMBODIMENT_REGISTRY if row.mobility != "fixed"}
    served = policy_requirements_served()
    walking = sorted(key for key, value in served.items() if value["locomote"])
    assert sorted(row["candidate_id"] for row in options["policies"]) == walking
    assert walking  # the catalog declares at least one policy that moves its robot
    assert all(not row["runnable"] and "pipeline:arena_navigation_scene_runtime" in row["missing"]
               and "pipeline:navigation_task_spec_from_compiled_spec" not in row["missing"]
               for row in options["policies"])
    # A fixed-arm policy never serves a locomotion requirement, and serves manipulation.
    arm = runnable_options(requirements(locomote=False, manipulate=True))
    assert arm["policies"] and all(served[row["candidate_id"]]["manipulate"] for row in arm["policies"])
    # An undeclared policy serves nothing.
    assert runnable_options(WALK, served={})["policies"] == []


def test_the_handoff_turns_a_compiled_corridor_walk_into_a_g1_navigation_task(tmp_path) -> None:
    # What website_scene_handoff writes beside compiled_task_spec.json (navigation_task_spec.json).
    from blueprint_pipeline.native_g1_navigation_goal import NAVIGATION_TASK_KIND
    from blueprint_pipeline.website_navigation_packet import navigation_task_spec

    base, path = _corridor(tmp_path)
    context = _context()
    spec = compile_locomotion_spec(task_context=context, facts=scene_facts(base_scene=base, camera_path_levelled_m=path),
                                   requirements=WALK)
    task = navigation_task_spec(spec, task_context=context)
    assert task["task_kind"] == NAVIGATION_TASK_KIND and task["compiled_task_spec_digest"] == spec["spec_digest"]
    assert task["g1_navigation_goal"]["center_world_m"][0] == pytest.approx(5.0)
    assert task["start_pose_world"][2] == pytest.approx(0.0, abs=0.02)  # on the floor
    assert "compiled:goal_pulled_back_for_obstacle_clearance" in task["navigation_goal_basis"]["disclosed_assumptions"]


def test_the_hold_names_each_missing_piece_and_is_re_derived(tmp_path) -> None:
    context = _context()
    gap = runnable_policy_gap(task_context=context, requirements=WALK, spec_digest="sha256:" + "c" * 64)
    assert gap["code"] == HOLD_CODE and gap["job_disposition"] == "held_for_preparation"
    assert {"pipeline:arena_navigation_scene_runtime", "pipeline:g1_navigation_preparation"} <= set(gap["missing"])
    assert "pipeline:navigation_scene_assets" not in gap["missing"]
    assert gap["runnable_options"]["navigation_pieces_built"] == sorted(BUILT_NAVIGATION_PIECES)
    # A job whose scene files could not be authored names that piece itself.
    failed = runnable_options(WALK, built_pieces=BUILT_NAVIGATION_PIECES - {"navigation_scene_assets"})
    assert all("pipeline:navigation_scene_assets" in row["missing"] for row in failed["policies"])
    assert "Unitree G1" in gap["reason"]
    handoff = {"schema_version": "website_scene_handoff.v1", "scene_id": "site-a", "capture_id": "walkthrough-a",
               "status": "needs_input", "blockers": [HOLD_CODE], "simulator_ready": False,
               "visual_reconstruction_ready": True, "native_task_preparation_gap": gap,
               "compiled_task_spec_digest": gap["compiled_task_spec_digest"]}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    assert holds_native_task_preparation(handoff, context)
    stale = copy.deepcopy(handoff)
    stale["native_task_preparation_gap"]["runnable_options"]["policies"] = []
    stale["digest"] = canonical_digest(stale, digest_field="digest")
    assert not holds_native_task_preparation(stale, context)  # a changed catalog releases the hold
    pick = _context("rigid_pick_place")
    assert not holds_native_task_preparation({**handoff, "scene_id": "site-a"}, pick)


def test_a_runnable_policy_is_never_held() -> None:
    options = runnable_options(WALK)
    options = json.loads(json.dumps(options))
    options["policies"][0].update(runnable=True, missing=[])
    assert runnable_policy_gap(task_context=_context(), requirements=WALK, spec_digest="sha256:" + "c" * 64,
                               options=options) is None
