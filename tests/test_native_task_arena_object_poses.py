"""Each rigid try records every movable body's pose; the single-object fields stay as they were."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_predicates as tp
from blueprint_pipeline.native_task_arena_readback import (
    NativeRigidTaskArenaReadback, read_native_task_arena_object_poses,
)
from tests.test_native_task_arena_readback import _built


def _rigid_two_box_scene(*, scored_subjects: int = 2):
    """The readback fixture as a rigid pick-and-place with a second box and a crate co-present.

    With ``scored_subjects=2`` both boxes are scored subjects (a several-object plan); with 1 only the
    first is, as on every plan today.
    """
    built = _built()
    built.plan["task_kind"] = "rigid_pick_place"
    built.plan["task_sample_binding"] = {"joint_ids": []}
    built.plan["task_spec"] = {"task_kind": "rigid_pick_place", "interaction_affordance": {
        "asset_root_from_scoring_frame": {"position_m": [0.0, 0.0, 0.05], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}}}
    built.plan["articulation"]["non_support_scene_contact_body_paths"] = []
    built.contact_sensor_names["task_support_contact"] = ("task_scene_contact",)
    del built.contact_sensor_names["task_scene_contact"]
    built.plan["objects"][0].update(asset_id="box_a", object_type="RIGID")
    for role, name, asset_id, pose in (("replacement", "replacement__box_b", "box_b", [1.15, 0.0, 0.03]),
                                       ("task_support", "task_support", "crate", [1.0, 0.0, 0.0])):
        built.plan["objects"].append({"semantic_role": role, "asset_id": asset_id, "name": name,
                                      "task_subject": role == "replacement" and scored_subjects == 2,
                                      "object_type": "RIGID",
                                      "pose_world": {"position_world_m": pose, "orientation_xyzw": [0, 0, 0, 1]}})
        built.scene_asset_names[name] = name
        built.env.unwrapped.scene[name] = SimpleNamespace(data=SimpleNamespace(root_pose_w=[[*pose, 0, 0, 0, 1]]))
    built.contact_sensor_names["destination_scene_support_contact"] = ("robot_scene_contact",)
    built.contact_sensor_names["destination_scene_forbidden_contact"] = ("robot_scene_contact",)
    return built


def test_a_one_object_try_writes_no_extra_fields() -> None:
    sample = NativeRigidTaskArenaReadback(_rigid_two_box_scene(scored_subjects=1)).read_task_sample()
    assert "object_poses_world" not in sample and "object_poses_unreadable" not in sample


def test_the_rigid_sample_carries_every_movable_body_and_keeps_the_single_object_fields() -> None:
    built = _rigid_two_box_scene()
    sample = NativeRigidTaskArenaReadback(built).read_task_sample()
    assert sample["object_poses_world"] == {
        "box_a": pytest.approx([1.9742142, 1.4792181, 0.0, 0.0, 0.0, 0.0, 1.0]),
        "box_b": [1.15, 0.0, 0.03, 0.0, 0.0, 0.0, 1.0],
        "crate": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    # Unchanged: the scoring-frame pose and the destination are the single-object fields.
    assert sample["task_object_pose_world"] == pytest.approx([1.9742142, 1.4792181, 0.05, 0.0, 0.0, 0.0, 1.0])
    assert sample["asset_root_pose_world"] == sample["object_poses_world"]["box_a"]
    assert sample["destination_pose_world"] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_the_recorded_poses_feed_the_several_object_predicates() -> None:
    built = _rigid_two_box_scene()
    box = {"minimum": [-0.10, -0.075, 0.0], "maximum": [0.10, 0.075, 0.10]}
    inside = {"id": "b_in_crate", "kind": "inside", "subject": {"kind": "object", "object_id": "box_b"},
              "subject_extent": box, "container": {"kind": "object", "object_id": "crate"},
              "container_interior": {"minimum": [-0.28, -0.18, 0.03], "maximum": [0.28, 0.18, 0.30]},
              "tolerance_m": 0.005, "settle_motion_m": 0.005, "hold_samples": 2, "when": "terminal"}
    samples = []
    for index in range(3):
        sample = NativeRigidTaskArenaReadback(built).read_task_sample()
        samples.append({"step_index": index, "object_poses_world": sample["object_poses_world"],
                        "object_robot_contact_active": {"box_b": False}})
    assert tp.score_episode(predicates=tp.seal_predicates([inside]), samples=samples)["outcome"] == "success"


def test_an_unreadable_or_nan_extra_body_never_ends_the_try() -> None:
    built = _rigid_two_box_scene()
    plain = NativeRigidTaskArenaReadback(built).read_task_sample()
    # Mid-try, the crate's pose reads NaN and the second box disappears from the scene.
    built.env.unwrapped.scene["task_support"].data.root_pose_w = [[float("nan"), 0, 0, 0, 0, 0, 1]]
    del built.env.unwrapped.scene["replacement__box_b"]
    sample = NativeRigidTaskArenaReadback(built).read_task_sample()
    assert sample["object_poses_unreadable"] == ["box_b", "crate"]
    assert set(sample["object_poses_world"]) == {"box_a"}
    # The task object's own readback is exactly as before.
    assert sample["task_object_pose_world"] == plain["task_object_pose_world"]
    assert read_native_task_arena_object_poses(built, built.env.unwrapped.scene)["unreadable"] == ["box_b", "crate"]


def test_a_rule_that_needs_an_unreadable_body_is_unscored_not_failed() -> None:
    box = {"minimum": [-0.10, -0.075, 0.0], "maximum": [0.10, 0.075, 0.10]}
    inside = {"id": "b_in_crate", "kind": "inside", "subject": {"kind": "object", "object_id": "box_b"},
              "subject_extent": box, "container": {"kind": "object", "object_id": "crate"},
              "container_interior": {"minimum": [-0.28, -0.18, 0.03], "maximum": [0.28, 0.18, 0.30]},
              "tolerance_m": 0.005, "settle_motion_m": 0.005, "hold_samples": 2, "when": "terminal"}
    samples = [{"step_index": index, "object_poses_world": {"box_a": [0, 0, 0, 0, 0, 0, 1]},
                "object_poses_unreadable": ["box_b", "crate"]} for index in range(3)]
    assert tp.score_episode(predicates=tp.seal_predicates([inside]), samples=samples)["status"] == "unscored"


def test_the_field_reaches_the_scoring_sample() -> None:
    from blueprint_pipeline.native_task_episode_environment import NativeRigidScoringEnvironment
    from tests.test_native_task_episode_environment import (
        _RigidEpisodeEnvironment, _RigidNativeReadback, _rigid_scoring_task_spec,
    )

    poses = {"box_a": [1.1, 2.1, 0.8, 0, 0, 0, 1], "crate": [1.0, 2.0, 0.7, 0, 0, 0, 1]}
    environment = NativeRigidScoringEnvironment(
        environment=_RigidEpisodeEnvironment(),
        task_readback=_RigidNativeReadback(object_poses_world=poses, object_poses_unreadable=["box_b"]),
        task_spec=_rigid_scoring_task_spec())
    sample = environment.read_object_sample()
    assert sample["object_poses_world"] == poses and sample["object_poses_unreadable"] == ["box_b"]
