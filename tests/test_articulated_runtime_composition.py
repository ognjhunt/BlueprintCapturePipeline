from __future__ import annotations

import pytest

from blueprint_pipeline.articulated_runtime_composition import (
    RUNTIME_COMPOSITION_SCHEMA_VERSION,
    ArticulatedRuntimeCompositionError,
    plan_articulated_runtime_composition,
)


def _task_spec(**overrides) -> dict:
    spec = {
        "task_kind": "articulated_open_close",
        "articulated_joints": [
            {
                "joint_id": "upper_door_hinge",
                "joint_prim_path": "/Asset/joints/upper_door_hinge",
                "role": "task_joint",
            },
            {
                "joint_id": "lower_door_hinge",
                "joint_prim_path": "/Asset/joints/lower_door_hinge",
                "role": "locked_joint",
            },
        ],
    }
    spec.update(overrides)
    return spec


def _plan(**overrides):
    arguments = {
        "task_spec": _task_spec(),
        "twin_usd_filename": "twin.usda",
        "scene_collision_filename": "scene_collision.usda",
        "appearance_filename": "appearance.usdz",
        "twin_position_world_m": [1.974, 1.479, 0.0],
    }
    arguments.update(overrides)
    return plan_articulated_runtime_composition(**arguments)


def test_the_twin_is_spawned_as_an_articulation_not_a_rigid_body() -> None:
    """A rigid refrigerator has no door, and would silently pass every check.

    Nothing downstream inspects the spawn type; the joints would simply be
    frozen and the task would read as impossible rather than misconfigured.
    """

    plan = _plan()

    twin = next(row for row in plan["objects"] if row["semantic_role"] == "task_object")
    assert twin["object_type"] == "ARTICULATION"
    assert twin["usd_filename"] == "twin.usda"
    assert plan["schema_version"] == RUNTIME_COMPOSITION_SCHEMA_VERSION


def test_collision_is_invisible_and_appearance_is_visible() -> None:
    """The splat renders; the collision proxy must never appear in a camera."""

    plan = _plan()

    by_role = {row["semantic_role"]: row for row in plan["objects"]}
    assert by_role["scene_collision"]["visible"] is False
    assert by_role["scene_collision"]["object_type"] == "BASE"
    assert by_role["scene_appearance"]["visible"] is True
    assert by_role["scene_appearance"]["object_type"] == "BASE"


def test_the_sample_binding_names_every_joint_the_scorer_demands() -> None:
    """The scorer rejects a sample whose joint set differs from the spec's.

    Discovering that inside a paid run costs the run, so the binding is derived
    from the same spec the scorer will check against.
    """

    plan = _plan()

    binding = plan["task_sample_binding"]
    assert binding["joint_ids"] == ["lower_door_hinge", "upper_door_hinge"]
    assert (
        binding["joint_prim_paths"]["upper_door_hinge"]
        == "/Asset/joints/upper_door_hinge"
    )


def test_a_rigid_task_never_produces_an_articulation() -> None:
    plan = _plan(task_spec=_task_spec(task_kind="rigid_pick_place", articulated_joints=[]))

    assert all(row["object_type"] != "ARTICULATION" for row in plan["objects"])
    assert plan["task_sample_binding"]["joint_ids"] == []


def test_an_articulated_task_with_no_joints_fails_closed() -> None:
    """An articulated task whose spec lists no joints cannot be scored at all."""

    with pytest.raises(ArticulatedRuntimeCompositionError) as excinfo:
        _plan(task_spec=_task_spec(articulated_joints=[]))

    assert any("joints_missing" in error for error in excinfo.value.errors)


def test_duplicate_joint_ids_fail_closed() -> None:
    """Two joints sharing an id would collapse into one sample entry."""

    with pytest.raises(ArticulatedRuntimeCompositionError) as excinfo:
        _plan(
            task_spec=_task_spec(
                articulated_joints=[
                    {"joint_id": "hinge", "joint_prim_path": "/Asset/joints/a"},
                    {"joint_id": "hinge", "joint_prim_path": "/Asset/joints/b"},
                ]
            )
        )

    assert any("joint_id_duplicated" in error for error in excinfo.value.errors)


def test_an_absent_appearance_leaves_the_scene_collidable_but_unrendered() -> None:
    """Appearance is optional; collision is not."""

    plan = _plan(appearance_filename=None)

    roles = {row["semantic_role"] for row in plan["objects"]}
    assert "scene_collision" in roles
    assert "scene_appearance" not in roles
    assert plan["claim_boundary"]["cameras_see_no_scene_background"] is True


def test_planning_is_deterministic() -> None:
    assert _plan() == _plan()


def test_composition_carries_bundle_aliases_for_each_asset():
    """The provider renames assets; the spec must say what to also look for.

    Bindings ship the task object as ``approved_can.usda`` and the scene
    collision as ``sage_collision.usd`` regardless of authoring names, so a
    composition that only records the authoring name cannot be resolved on the
    provider.
    """

    plan = plan_articulated_runtime_composition(
        task_spec=_task_spec(),
        twin_usd_filename="twin.usda",
        scene_collision_filename="scene.usda",
        asset_filename_aliases={
            "task_object": ["approved_can.usda"],
            "scene_collision": ["sage_collision.usd"],
        },
    )

    by_role = {row["semantic_role"]: row for row in plan["objects"]}
    assert by_role["task_object"]["usd_filename_aliases"] == ["approved_can.usda"]
    assert by_role["scene_collision"]["usd_filename_aliases"] == ["sage_collision.usd"]


def test_composition_defaults_to_no_aliases():
    plan = plan_articulated_runtime_composition(
        task_spec=_task_spec(),
        twin_usd_filename="twin.usda",
        scene_collision_filename="scene.usda",
    )

    for row in plan["objects"]:
        assert row["usd_filename_aliases"] == []


def test_composition_refuses_an_alias_for_an_unknown_role():
    """A typo in a role name must not silently leave the asset unaliased."""

    with pytest.raises(ArticulatedRuntimeCompositionError) as excinfo:
        plan_articulated_runtime_composition(
            task_spec=_task_spec(),
            twin_usd_filename="twin.usda",
            scene_collision_filename="scene.usda",
            asset_filename_aliases={"taskobject": ["approved_can.usda"]},
        )

    assert any("alias_role_unknown" in error for error in excinfo.value.errors)


def test_a_policy_bound_composition_must_carry_an_appearance():
    """A vision policy cannot be evaluated against an invisible room.

    The 840796 scene shipped with a collision-only background: the room's
    geometry present for physics, nothing for the cameras. That is fine for
    scripted controls, which are pure physics - and silently wrong for
    pi05_droid or groot_n17_droid, which consume the camera images. They would
    be looking at a floating appliance in a void, and a failure there says
    nothing about the policy.

    Nobody noticed for several runs because nothing asked. Now something asks.
    """

    with pytest.raises(ArticulatedRuntimeCompositionError) as excinfo:
        plan_articulated_runtime_composition(
            task_spec=_task_spec(),
            twin_usd_filename="twin.usda",
            scene_collision_filename="scene.usda",
            appearance_filename=None,
            intended_for_policy_execution=True,
        )

    assert any("appearance_missing" in error for error in excinfo.value.errors)


def test_a_policy_bound_composition_with_an_appearance_is_admitted():
    plan = plan_articulated_runtime_composition(
        task_spec=_task_spec(),
        twin_usd_filename="twin.usda",
        scene_collision_filename="scene.usda",
        appearance_filename="scene_appearance.usdz",
        intended_for_policy_execution=True,
    )

    roles = {row["semantic_role"] for row in plan["objects"]}
    assert "scene_appearance" in roles
    assert plan["intended_for_policy_execution"] is True


def test_scripted_controls_do_not_require_an_appearance():
    """Physics does not need to be lit; refusing here would block real work."""

    plan = plan_articulated_runtime_composition(
        task_spec=_task_spec(),
        twin_usd_filename="twin.usda",
        scene_collision_filename="scene.usda",
    )

    assert plan["intended_for_policy_execution"] is False
    assert plan["appearance_present"] is False
    # And it says so, so a reader cannot mistake it for a policy-ready scene.
    assert plan["claim_boundary"]["cameras_see_no_scene_background"] is True
