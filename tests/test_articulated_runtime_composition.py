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
        "target_joint_id": "upper_door_hinge",
        "joint_reset_positions_rad": {
            "upper_door_hinge": 0.0,
            "lower_door_hinge": 0.0,
        },
    }
    spec.update(overrides)
    return spec


def _joint_bindings() -> list[dict]:
    return [
        {
            "joint_id": "upper_door_hinge",
            "joint_prim_path": "/Asset/joints/upper_door_hinge",
            "native_joint_name": "upper_door_hinge",
            "role": "task_joint",
        },
        {
            "joint_id": "lower_door_hinge",
            "joint_prim_path": "/Asset/joints/lower_door_hinge",
            "native_joint_name": "lower_door_hinge",
            "role": "locked_joint",
        },
    ]


def _plan(**overrides):
    arguments = {
        "task_spec": _task_spec(),
        "task_joint_bindings": _joint_bindings(),
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
    assert binding["binding_source"] == "runtime_contract"
    assert binding["joint_ids"] == ["lower_door_hinge", "upper_door_hinge"]
    assert (
        binding["joint_prim_paths"]["upper_door_hinge"]
        == "/Asset/joints/upper_door_hinge"
    )


def test_a_rigid_task_never_produces_an_articulation() -> None:
    plan = _plan(
        task_spec={"task_kind": "rigid_pick_place"}, task_joint_bindings=[]
    )

    assert all(row["object_type"] != "ARTICULATION" for row in plan["objects"])
    assert plan["task_sample_binding"]["joint_ids"] == []


def test_an_articulated_task_with_no_joints_fails_closed() -> None:
    """An articulated task whose spec lists no joints cannot be scored at all."""

    with pytest.raises(ArticulatedRuntimeCompositionError) as excinfo:
        _plan(task_joint_bindings=[])

    assert any("joints_missing" in error for error in excinfo.value.errors)


def test_duplicate_joint_ids_fail_closed() -> None:
    """Two joints sharing an id would collapse into one sample entry."""

    with pytest.raises(ArticulatedRuntimeCompositionError) as excinfo:
        _plan(
            task_spec=_task_spec(
                target_joint_id="hinge", joint_reset_positions_rad={"hinge": 0.0}
            ),
            task_joint_bindings=[
                {
                    "joint_id": "hinge",
                    "joint_prim_path": "/Asset/joints/a",
                    "native_joint_name": "a",
                },
                {
                    "joint_id": "hinge",
                    "joint_prim_path": "/Asset/joints/b",
                    "native_joint_name": "b",
                },
            ],
        )

    assert any("joint_id_duplicated" in error for error in excinfo.value.errors)


def test_duplicate_native_joint_names_fail_closed() -> None:
    bindings = _joint_bindings()
    bindings[1]["native_joint_name"] = bindings[0]["native_joint_name"]

    with pytest.raises(ArticulatedRuntimeCompositionError) as excinfo:
        _plan(task_joint_bindings=bindings)

    assert excinfo.value.errors == (
        "articulated_runtime_composition_joint_binding_missing:lower_door_hinge",
        "articulated_runtime_composition_native_joint_name_duplicated:upper_door_hinge",
    )


def test_an_absent_appearance_leaves_the_scene_collidable_but_unrendered() -> None:
    """Appearance is optional; collision is not."""

    plan = _plan(appearance_filename=None)

    roles = {row["semantic_role"] for row in plan["objects"]}
    assert "scene_collision" in roles
    assert "scene_appearance" not in roles
    assert plan["claim_boundary"]["cameras_see_no_scene_background"] is True


def test_scorer_and_runtime_joint_sets_must_match_exactly() -> None:
    with pytest.raises(ArticulatedRuntimeCompositionError) as excinfo:
        _plan(
            task_joint_bindings=[
                _joint_bindings()[0],
                {
                    "joint_id": "unscored_hinge",
                    "joint_prim_path": "/Asset/joints/unscored_hinge",
                    "native_joint_name": "unscored_hinge",
                },
            ]
        )

    assert excinfo.value.errors == (
        "articulated_runtime_composition_joint_binding_missing:lower_door_hinge",
        "articulated_runtime_composition_joint_binding_unexpected:unscored_hinge",
    )


def test_a_rigid_task_rejects_an_unscored_joint_binding() -> None:
    with pytest.raises(ArticulatedRuntimeCompositionError) as excinfo:
        _plan(
            task_spec={"task_kind": "rigid_pick_place"},
            task_joint_bindings=[_joint_bindings()[0]],
        )

    assert excinfo.value.errors == (
        "articulated_runtime_composition_joint_binding_unexpected:upper_door_hinge",
    )


def test_planning_is_deterministic() -> None:
    assert _plan() == _plan()
