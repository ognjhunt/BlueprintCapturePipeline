from __future__ import annotations

import json
import math

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_stance_variant import (
    DROID_ARENA_DEFAULT_RESET,
    DROID_ARENA_DEFAULT_RESET_SOURCE,
    FRANKA_ROBOTIQ_READY_RESET,
    FRANKA_ROBOTIQ_READY_RESET_SOURCE,
    FRONT_ENTRY_BASE_LATERAL_OFFSET_M,
    FRONT_ENTRY_GRASP_ORIENTATION_VARIANT,
    NativeTaskArenaStanceVariantError,
    RETREAT_STRATEGY_ID,
    materialize_native_task_arena_stance_variant_request,
)


def _request() -> dict:
    camera = {
        "policy_input": True,
        "review_only": False,
        "pose_frame": "world",
        "parent_prim_path": "{ENV_REGEX_NS}",
        "optical_convention": "opencv",
        "frame_from_camera_matrix": [1.0, 0.0, 0.0, 0.0] * 4,
        "intrinsics": {
            "fx": 172.0,
            "fy": 172.0,
            "cx": 159.5,
            "cy": 89.5,
            "width": 320,
            "height": 180,
        },
    }
    request = {
        "schema_version": "native_task_arena_packet_request.v1",
        "scene_id": "840920",
        "task_id": "task_a_washer_door_open",
        "robot_base_pose_world": {
            "position_world_m": [3.5154863, 9.208716, 0.090782],
            "orientation_xyzw": [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
        },
        "robot_joint_reset_positions_rad": dict(FRANKA_ROBOTIQ_READY_RESET),
        "assets": [
            {
                "asset_id": "washer",
                "pose_world": {
                    "position_world_m": [3.5154863, 9.758716, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ],
        "task_spec": {
            "subject_asset_id": "washer",
            "articulation_graph_digest": "sha256:" + "a" * 64,
            "interaction_affordance": {
                "approach_unit_asset_root": [1.0, 0.0, 0.0],
                "retreat_unit_asset_root": [0.0, -1.0, 0.0],
                "gripper_orientation_contact_xyzw": [
                    -math.sqrt(0.5),
                    0.0,
                    math.sqrt(0.5),
                    0.0,
                ],
                "precontact_clearance_m": 0.12,
                "retreat_clearance_m": 0.12,
                "affordance_digest": "",
                "joint_contact_path": [
                    {
                        "contact_pose_asset_root": {
                            "position_m": [0.248, -0.302052, 0.405],
                            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        }
                    },
                    {
                        "contact_pose_asset_root": {
                            "position_m": [0.08949, -0.664167, 0.405],
                            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        }
                    },
                ],
            },
        },
        "task_state_binding": {
            "schema_version": "native_articulated_graph_task_state_binding.v1",
            "articulation_graph_digest": "sha256:" + "a" * 64,
            "interaction_affordance_digest": "",
        },
        "cameras": [
            {**camera, "role": "external"},
            {
                **camera,
                "role": "wrist",
                "pose_frame": "robot_body",
                "parent_prim_path": "{ENV_REGEX_NS}/Robot/Gripper/base_link",
            },
            {**camera, "role": "overview", "policy_input": False, "review_only": True},
        ],
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    affordance = request["task_spec"]["interaction_affordance"]
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    request["task_state_binding"]["interaction_affordance_digest"] = affordance[
        "affordance_digest"
    ]
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    return request


def test_stance_keeps_front_base_while_tool_approaches_free_edge(tmp_path) -> None:
    request = _request()
    source = tmp_path / "request.json"
    source.write_text(json.dumps(request), encoding="utf-8")
    output = tmp_path / "stance.json"

    result = materialize_native_task_arena_stance_variant_request(
        base_request_path=source, output_path=output
    )

    base = result["robot_base_pose_world"]
    assert base["position_world_m"] == pytest.approx(
        [3.7634863, 8.906664, 0.090782]
    )
    assert base["orientation_xyzw"] == pytest.approx(
        [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
    )
    assert result["robot_joint_reset_positions_rad"] == FRANKA_ROBOTIQ_READY_RESET
    stance = result["stance_variant"]
    assert stance["door_standoff_m"] == 0.55
    assert stance["approach_outward_world"] == pytest.approx([1.0, 0.0, 0.0])
    assert stance["base_outward_world"] == pytest.approx([0.0, -1.0, 0.0])
    assert stance["base_outward_source"] == "authored_gripper_positive_jaw_axis"
    assert stance["maximum_door_sweep_bearing_deviation_rad"] < math.pi / 4.0
    assert stance["native_ik_qualified"] is False
    assert stance["retreat_strategy_id"] == RETREAT_STRATEGY_ID
    assert stance["reset_source"] == FRANKA_ROBOTIQ_READY_RESET_SOURCE
    assert stance["authored_retreat_enters_base_dead_zone"] is True
    closed = stance["closed_contact_world_m"]
    expected_front_staging = [closed[0], closed[1] - 0.12, closed[2]]
    assert stance["resolved_retreat_target_world_m"] == pytest.approx(
        expected_front_staging
    )
    assert stance["resolved_retreat_clearance_m"] > 0.1
    affordance = result["task_spec"]["interaction_affordance"]
    final_contact = [
        3.5154863 + 0.08949,
        9.758716 - 0.664167,
        0.405,
    ]
    resolved_target = [
        final_contact[index]
        + affordance["retreat_unit_asset_root"][index]
        * affordance["retreat_clearance_m"]
        for index in range(3)
    ]
    assert resolved_target == pytest.approx(
        expected_front_staging
    )
    assert affordance["affordance_digest"] == canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    assert result["task_state_binding"]["interaction_affordance_digest"] == (
        affordance["affordance_digest"]
    )
    assert result["request_digest"] == canonical_digest(
        result, digest_field="request_digest"
    )
    assert json.loads(output.read_text()) == result

    external = next(row for row in result["cameras"] if row["role"] == "external")
    overview = next(row for row in result["cameras"] if row["role"] == "overview")
    assert external["frame_from_camera_matrix"] == pytest.approx(
        [
            0.7295372041,
            0.3878550832,
            -0.5633328523,
            4.5134863,
            0.6839411289,
            -0.4137120886,
            0.6008883755,
            8.456664,
            0.0,
            -0.8236569316,
            -0.5670884043,
            1.255,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        abs=1.0e-7,
    )
    assert overview["frame_from_camera_matrix"][7] < 9.456664


def test_front_entry_patch_places_base_on_outward_approach_axis(tmp_path) -> None:
    request = _request()
    affordance = request["task_spec"]["interaction_affordance"]
    affordance.update(
        {
            "approach_unit_asset_root": [0.0, -1.0, 0.0],
            "gripper_orientation_contact_xyzw": [
                0.0,
                -math.sqrt(0.5),
                -math.sqrt(0.5),
                0.0,
            ],
            "contact_outward_standoff_m": 0.01,
            "grasp_swept_volume_receipt_digest": "sha256:" + "f" * 64,
        }
    )
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    request["task_state_binding"]["interaction_affordance_digest"] = affordance[
        "affordance_digest"
    ]
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    source = tmp_path / "front-entry.json"
    source.write_text(json.dumps(request), encoding="utf-8")

    result = materialize_native_task_arena_stance_variant_request(
        base_request_path=source,
        output_path=tmp_path / "front-entry-stance.json",
    )

    assert result["robot_base_pose_world"]["position_world_m"] == pytest.approx(
        [3.7884863, 8.906664, 0.090782]
    )
    stance = result["stance_variant"]
    assert stance["approach_outward_world"] == pytest.approx([0.0, -1.0, 0.0])
    assert stance["base_outward_world"] == pytest.approx([0.0, -1.0, 0.0])
    assert stance["base_outward_source"] == (
        "measured_front_entry_approach_outward_axis"
    )
    assert stance["derivation"] == (
        "door_contact_plus_front_entry_approach_standoff"
    )
    assert stance["base_lateral_offset_m"] == FRONT_ENTRY_BASE_LATERAL_OFFSET_M
    assert stance["base_lateral_world"] == pytest.approx([1.0, 0.0, 0.0])
    assert stance["base_lateral_source"] == (
        "world_up_cross_front_entry_outward_axis"
    )
    assert result["robot_joint_reset_positions_rad"] == DROID_ARENA_DEFAULT_RESET
    assert stance["reset_source"] == DROID_ARENA_DEFAULT_RESET_SOURCE
    assert stance["grasp_orientation_variant"] == (
        FRONT_ENTRY_GRASP_ORIENTATION_VARIANT
    )
    assert result["robot_joint_reset_positions_rad"]["panda_joint7"] == 0.0
    assert result["robot_joint_reset_positions_rad"]["panda_joint6"] == pytest.approx(
        3.0 * math.pi / 5.0
    )
    flipped = result["task_spec"]["interaction_affordance"][
        "gripper_orientation_contact_xyzw"
    ]
    assert flipped == pytest.approx(
        [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
    )
    # The local +Z tool approach remains inward +Y while local +Y changes from
    # +Z to -Z. The affordance's outward approach vector is the opposite, -Y.
    assert _rotate(flipped, [0.0, 0.0, 1.0]) == pytest.approx(
        [0.0, 1.0, 0.0]
    )
    assert _rotate(flipped, [0.0, 1.0, 0.0]) == pytest.approx(
        [0.0, 0.0, -1.0]
    )


def _rotate(quaternion, vector):
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def test_front_entry_jaw_sign_variant_is_idempotent(tmp_path) -> None:
    request = _request()
    affordance = request["task_spec"]["interaction_affordance"]
    affordance.update(
        {
            "approach_unit_asset_root": [0.0, -1.0, 0.0],
            "gripper_orientation_contact_xyzw": [
                0.0,
                -math.sqrt(0.5),
                -math.sqrt(0.5),
                0.0,
            ],
            "contact_outward_standoff_m": 0.01,
            "grasp_swept_volume_receipt_digest": "sha256:" + "f" * 64,
        }
    )
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    request["task_state_binding"]["interaction_affordance_digest"] = affordance[
        "affordance_digest"
    ]
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    source = tmp_path / "source.json"
    source.write_text(json.dumps(request), encoding="utf-8")
    first_path = tmp_path / "first.json"
    first = materialize_native_task_arena_stance_variant_request(
        base_request_path=source,
        output_path=first_path,
        door_standoff_m=0.65,
    )

    second = materialize_native_task_arena_stance_variant_request(
        base_request_path=first_path,
        output_path=tmp_path / "second.json",
        door_standoff_m=0.65,
    )

    assert second["task_spec"]["interaction_affordance"][
        "gripper_orientation_contact_xyzw"
    ] == pytest.approx(
        first["task_spec"]["interaction_affordance"][
            "gripper_orientation_contact_xyzw"
        ]
    )


def test_stance_variant_reapplication_preserves_original_retreat_geometry(
    tmp_path,
) -> None:
    request = _request()
    source = tmp_path / "request.json"
    source.write_text(json.dumps(request), encoding="utf-8")
    first_path = tmp_path / "first.json"
    first = materialize_native_task_arena_stance_variant_request(
        base_request_path=source,
        output_path=first_path,
        door_standoff_m=0.65,
    )

    second = materialize_native_task_arena_stance_variant_request(
        base_request_path=first_path,
        output_path=tmp_path / "second.json",
        door_standoff_m=0.65,
    )

    first_stance = first["stance_variant"]
    second_stance = second["stance_variant"]
    assert second_stance["authored_retreat_unit_asset_root"] == pytest.approx(
        [0.0, -1.0, 0.0]
    )
    assert second_stance["resolved_retreat_target_world_m"] == pytest.approx(
        first_stance["resolved_retreat_target_world_m"]
    )
    assert second["task_spec"]["interaction_affordance"][
        "retreat_unit_asset_root"
    ] == pytest.approx(
        first["task_spec"]["interaction_affordance"][
            "retreat_unit_asset_root"
        ]
    )
    assert second["task_spec"]["interaction_affordance"][
        "retreat_clearance_m"
    ] == pytest.approx(
        first["task_spec"]["interaction_affordance"][
            "retreat_clearance_m"
        ]
    )


def test_stance_refuses_a_base_too_close_to_the_door(tmp_path) -> None:
    request = _request()
    source = tmp_path / "request.json"
    source.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(
        NativeTaskArenaStanceVariantError,
        match="native_task_arena_stance_standoff_invalid",
    ):
        materialize_native_task_arena_stance_variant_request(
            base_request_path=source,
            output_path=tmp_path / "stance.json",
            door_standoff_m=0.25,
        )
