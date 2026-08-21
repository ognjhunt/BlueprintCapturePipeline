from __future__ import annotations

import json
import math

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_stance_variant import (
    FRANKA_ROBOTIQ_READY_RESET,
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
            "interaction_affordance": {
                "approach_unit_asset_root": [0.0, -1.0, 0.0],
                "retreat_unit_asset_root": [0.0, -1.0, 0.0],
                "retreat_clearance_m": 0.12,
                "affordance_digest": "",
                "joint_contact_path": [
                    {
                        "contact_pose_asset_root": {
                            "position_m": [0.248, -0.302052, 0.405]
                        }
                    },
                    {
                        "contact_pose_asset_root": {
                            "position_m": [0.08949, -0.664167, 0.405]
                        }
                    },
                ],
            },
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
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    return request


def test_stance_centers_floor_base_on_door_normal_and_replaces_reset(tmp_path) -> None:
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
    assert stance["maximum_door_sweep_bearing_deviation_rad"] < math.pi / 4.0
    assert stance["native_ik_qualified"] is False
    assert stance["retreat_strategy_id"] == RETREAT_STRATEGY_ID
    assert stance["authored_retreat_enters_base_dead_zone"] is True
    assert stance["resolved_retreat_unit_world"] == [0.0, 0.0, 1.0]
    affordance = result["task_spec"]["interaction_affordance"]
    assert affordance["retreat_unit_asset_root"] == pytest.approx(
        [0.0, 0.0, 1.0]
    )
    assert affordance["affordance_digest"] == canonical_digest(
        affordance, digest_field="affordance_digest"
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
