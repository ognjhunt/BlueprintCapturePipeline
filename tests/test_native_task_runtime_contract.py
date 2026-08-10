from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.native_task_runtime_contract import (
    DROID_FRANKA_RESET_JOINT_NAMES,
    FROZEN_CANDIDATES,
    NativeTaskRuntimeContractError,
    load_native_task_runtime_contract,
    materialize_native_task_runtime_contract,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> dict:
    return {
        "position_world_m": [x, y, z],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _camera(role: str) -> dict:
    wrist = role == "wrist"
    return {
        "role": role,
        "policy_input": role in {"external", "wrist"},
        "scoring_input": False,
        "pose_frame": "panda_hand" if wrist else "world",
        "frame_from_camera_matrix": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "intrinsics": {
            "fx": 300.0,
            "fy": 300.0,
            "cx": 159.5,
            "cy": 119.5,
            "width": 320,
            "height": 240,
        },
    }


def _common(task_spec: dict, *, scene_id: str, task_id: str) -> dict:
    return {
        "scene_id": scene_id,
        "task_id": task_id,
        "task_spec": task_spec,
        "assets": [
            {
                "semantic_role": "scene_collision",
                "filename": "scene_collision.usd",
                "sha256": _sha("a"),
                "pose_world": _pose(),
            },
            {
                "semantic_role": "scene_appearance",
                "filename": "scene_appearance.usdz",
                "sha256": _sha("b"),
                "pose_world": _pose(),
            },
            {
                "semantic_role": "task_object",
                "filename": "task_object.usda",
                "sha256": _sha("c"),
                "pose_world": _pose(1.0, 2.0, 0.0),
            },
        ],
        "robot_base_pose_world": _pose(1.75, 1.99, 0.0),
        "robot_joint_reset_positions_rad": {
            name: float(index) / 100.0
            for index, name in enumerate(DROID_FRANKA_RESET_JOINT_NAMES)
        },
        "cameras": [_camera(role) for role in ("external", "wrist", "overview")],
        "scenario_cell_id": "canonical_seed_17",
        "scenario_instance_digest": _sha("d"),
        "seed": 17,
    }


def _rigid_fixture() -> dict:
    fixture = _common(
        {
            "task_kind": "rigid_pick_place",
            "prompt": "Pick up the can and place it in the destination.",
        },
        scene_id="840313",
        task_id="840313_canned_beverage_pick_place_v1",
    )
    fixture["task_joint_bindings"] = []
    return fixture


def _articulated_fixture() -> dict:
    freeze = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/arm_decision_proof_v1/manifests"
            / "second_scene_840796_scene_task_freeze.v1.json"
        ).read_text(encoding="utf-8")
    )
    fixture = _common(
        freeze["task_spec"],
        scene_id="840796",
        task_id="840796_refrigerator_upper_door_open_v1",
    )
    fixture["task_joint_bindings"] = [
        {
            "joint_id": "refrigerator_upper_door_hinge",
            "joint_prim_path": "/Asset/joints/upper_door_hinge",
            "native_joint_name": "upper_door_hinge",
            "role": "task_joint",
        },
        {
            "joint_id": "refrigerator_lower_door_hinge",
            "joint_prim_path": "/Asset/joints/lower_door_hinge",
            "native_joint_name": "lower_door_hinge",
            "role": "locked_joint",
        },
    ]
    fixture["task_state_binding"] = {
        "moving_link_prim_path": "/Asset/upper_door",
        "handle_prim_paths": [
            "/Asset/upper_door/component_004",
            "/Asset/upper_door/handle_post_a",
            "/Asset/upper_door/handle_post_b",
        ],
        "handle_grasp_point_link_m": [0.119962, 0.327634, 1.022997],
        "robot_gripper_contact_prim_pattern": (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/.*"
        ),
        "robot_collision_prim_pattern": "{ENV_REGEX_NS}/Robot/.*",
        "task_contact_minimum_force_n": 0.5,
        "collision_failure_minimum_force_n": 1.0,
        "retreat_minimum_separation_m": 0.10,
        "root_translation_tolerance_m": 0.002,
        "root_orientation_tolerance_rad": 0.01,
    }
    return fixture


@pytest.mark.parametrize(
    ("fixture", "expected_type", "expected_joints"),
    [
        (_rigid_fixture, "RIGID", []),
        (
            _articulated_fixture,
            "ARTICULATION",
            [
                "refrigerator_lower_door_hinge",
                "refrigerator_upper_door_hinge",
            ],
        ),
    ],
)
def test_original_and_second_scene_share_one_runtime_contract(
    fixture, expected_type: str, expected_joints: list[str]
) -> None:
    contract = materialize_native_task_runtime_contract(**fixture())

    objects = {row["semantic_role"]: row for row in contract["objects"]}
    assert objects["task_object"]["object_type"] == expected_type
    assert objects["scene_collision"]["visible"] is False
    assert objects["scene_appearance"]["visible"] is True
    assert contract["task_sample_binding"]["joint_ids"] == expected_joints
    assert contract["candidate_ids"] == list(FROZEN_CANDIDATES)
    assert contract["robot"]["action_seam"]["action_dimension"] == 8
    assert set(contract["robot"]["joint_reset_positions_rad"]) == set(
        DROID_FRANKA_RESET_JOINT_NAMES
    )
    if expected_type == "ARTICULATION":
        assert contract["task_state_binding"]["measurement_authority"][
            "caller_asserted_booleans_forbidden"
        ] is True
    else:
        assert contract["task_state_binding"] is None


def test_policy_and_review_camera_roles_cannot_be_swapped() -> None:
    fixture = _articulated_fixture()
    fixture["cameras"][2]["policy_input"] = True

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_camera_policy_role_invalid:overview" in excinfo.value.errors


def test_runtime_contract_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    path = tmp_path / "native_task_runtime_contract.json"
    created = materialize_native_task_runtime_contract(
        **_articulated_fixture(), destination=path
    )

    assert load_native_task_runtime_contract(path) == created
    tampered = json.loads(path.read_text())
    tampered["robot"]["base_pose_world"]["position_world_m"][0] += 0.1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        load_native_task_runtime_contract(path)
    assert excinfo.value.errors == ("native_task_runtime_contract_digest_invalid",)


def test_articulated_task_without_joint_binding_fails_before_gpu() -> None:
    fixture = _articulated_fixture()
    fixture["task_joint_bindings"] = []

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert any("composition_invalid" in error for error in excinfo.value.errors)


def test_articulated_task_without_native_state_binding_fails_before_gpu() -> None:
    fixture = _articulated_fixture()
    fixture["task_state_binding"] = None

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_state_binding_missing" in excinfo.value.errors


def test_state_binding_rejects_handle_outside_the_moving_link() -> None:
    fixture = _articulated_fixture()
    fixture["task_state_binding"]["handle_prim_paths"] = [
        "/Asset/lower_door/component_005"
    ]

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_handle_prim_invalid:0" in excinfo.value.errors


def test_task_asset_path_cannot_escape_the_provider_asset_directory() -> None:
    fixture = _articulated_fixture()
    fixture["assets"][2]["filename"] = "../task_object.usda"

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_asset_filename_invalid:task_object" in excinfo.value.errors


def test_robot_reset_joint_map_rejects_missing_and_extra_names() -> None:
    fixture = _articulated_fixture()
    fixture["robot_joint_reset_positions_rad"].pop("panda_joint7")
    fixture["robot_joint_reset_positions_rad"]["scene_specific_joint"] = 0.0

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_robot_reset_joint_missing:panda_joint7" in excinfo.value.errors
    assert (
        "native_task_runtime_robot_reset_joint_unexpected:scene_specific_joint"
        in excinfo.value.errors
    )
