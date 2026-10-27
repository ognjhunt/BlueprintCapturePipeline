"""DROID's right exterior: Arena's mount first, nearby DROID-like poses next, none when nothing sees the task."""

import math

import numpy as np
import pytest

from blueprint_pipeline import droid_exterior_cameras as dec
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

IDENTITY = {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
INTRINSICS = {"fx": 500.0, "fy": 500.0, "cx": 639.5, "cy": 359.5, "width": 1280, "height": 720}


def _matrix(row):
    return np.asarray(row["frame_from_camera_matrix"]).reshape(4, 4)


def _hits_plane(matrix, z):
    origin, axis = matrix[:3, 3], matrix[:3, 2]
    return origin + (z - origin[2]) / axis[2] * axis


def test_arena_mount_stands_over_the_right_shoulder_looking_at_the_workspace():
    matrix = dec.arena_mount_world(IDENTITY)
    assert matrix[:3, 3] == pytest.approx(dec.ARENA_MOUNT_POSITION_M)
    # OpenCV optics: forward meets the base plane ahead of the arm, image down points down, right points right.
    assert _hits_plane(matrix, 0.0)[:2] == pytest.approx([0.73, -0.04], abs=0.01)
    assert matrix[2, 1] < -0.7 and matrix[:3, :3] @ matrix[:3, :3].T == pytest.approx(np.eye(3), abs=1e-9)
    # The station's yaw carries it: a base turned 90 degrees at (2, 1) puts it on the base's right.
    turned = dec.arena_mount_world({"position_world_m": [2.0, 1.0, 0.75],
                                    "orientation_xyzw": [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]})
    assert turned[:3, 3] == pytest.approx([2.0 + 0.57, 1.0 + 0.05, 0.75 + 0.66])


def test_alternatives_stay_beside_the_shoulder_and_aim_at_the_task():
    aim = [0.6, 0.1, 0.2]
    poses = dec.candidate_poses(IDENTITY, aim)
    assert poses[0][0] == "arena_mount" and len(poses) == 27 and len({name for name, _ in poses}) == 27
    distances = []
    for name, matrix in poses[1:]:
        offset = matrix[:3, 3] - np.asarray(dec.ARENA_MOUNT_POSITION_M)
        assert -0.10 - 1e-9 <= offset[0] <= 0.15 + 1e-9 and -0.15 - 1e-9 <= offset[1] <= 0.15 + 1e-9
        assert -0.15 - 1e-9 <= offset[2] <= 0.25 + 1e-9
        forward = matrix[:3, 2]
        towards = (np.asarray(aim) - matrix[:3, 3]) / np.linalg.norm(np.asarray(aim) - matrix[:3, 3])
        assert forward == pytest.approx(towards) and matrix[2, 1] < 0  # upright
        distances.append(float(np.linalg.norm(offset)))
    assert distances == sorted(distances)


def test_the_first_clear_pose_is_placed_and_every_attempt_is_receipted():
    seen = []

    def evaluate(row):
        seen.append(row)
        return ["scene_camera_contact_occluded_by_scene:exterior_right:40%"] if len(seen) < 3 else []

    row, receipt = dec.place_exterior_right(base_pose_world=IDENTITY, aim_world=[0.6, 0.0, 0.2],
                                            intrinsics=INTRINSICS, evaluate=evaluate)
    assert row is seen[2] and row["role"] == "exterior_right" and row["policy_input"] is True
    assert row["pose_frame"] == "world" and row["optical_convention"] == "opencv" and row["intrinsics"] == INTRINSICS
    assert receipt["status"] == "placed" and receipt["placement"] == receipt["attempts"][2]["candidate"]
    assert [bool(attempt["problems"]) for attempt in receipt["attempts"]] == [True, True, False]
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")


def test_the_placed_row_converts_to_the_runtime_camera_with_arenas_optics():
    from blueprint_pipeline.native_task_arena_runtime import camera_runtime_parameters

    row, _ = dec.place_exterior_right(base_pose_world=IDENTITY, aim_world=[0.6, 0.0, 0.2],
                                      intrinsics=dec.ARENA_EXTERIOR_INTRINSICS, evaluate=lambda row: [])
    # The runtime contract seals review_only beside policy_input; the compiled plan is covered in
    # test_team_scene_packet.
    parameters = camera_runtime_parameters({**row, "review_only": False}, robot_id="franka_panda")
    assert parameters["runtime_name"] == "exterior_right_camera" and parameters["policy_input"] is True
    assert (parameters["width"], parameters["height"]) == (1280, 720)
    across = 2 * math.degrees(math.atan(parameters["horizontal_aperture_mm"] / 2 / parameters["focal_length_mm"]))
    assert across == pytest.approx(2 * math.degrees(math.atan(5.376 / 2 / 2.1)), abs=0.01)  # Arena's 104 degrees


def test_no_clear_pose_leaves_the_scene_without_a_right_exterior_and_says_why():
    row, receipt = dec.place_exterior_right(base_pose_world=IDENTITY, aim_world=[0.6, 0.0, 0.2],
                                            intrinsics=INTRINSICS,
                                            evaluate=lambda row: ["scene_camera_contact_outside_view:exterior_right"])
    assert row is None and receipt["status"] == "unavailable"
    assert receipt["reason"] == dec.UNAVAILABLE_NO_CLEAR_VIEW and len(receipt["attempts"]) == 27
    unmeasured = dec.unmeasured_receipt("scene_camera_appearance_not_a_gaussian_splat")
    assert unmeasured["status"] == "unavailable" and unmeasured["attempts"] == []
    assert unmeasured["reason"] == "droid_exterior_right_unmeasured:scene_camera_appearance_not_a_gaussian_splat"


def test_a_candidate_is_fresh_on_the_views_it_reads_only():
    from blueprint_pipeline.adp009d_policy_episode import _policy_views_fresh

    observed, stale = {"status": "observed"}, {"status": "stale"}
    three = {"external": observed, "wrist": observed, "exterior_right": stale}
    assert _policy_views_fresh(three, "pi05_droid") is True  # it never reads the right exterior
    assert _policy_views_fresh(three, "cosmos3_edge_policy_droid") is False
    assert _policy_views_fresh({"external": observed, "wrist": observed}, "cosmos3_edge_policy_droid") is False
    assert _policy_views_fresh({**three, "exterior_right": observed}, "cosmos3_edge_policy_droid") is True


def test_the_runtime_adds_the_slot_as_a_rig_field_arena_spawns_like_its_own(monkeypatch):
    import dataclasses
    import sys
    import types

    from blueprint_pipeline.native_task_robot_registry import (
        EXTERIOR_RIGHT_RUNTIME_NAME,
        add_droid_exterior_right_slot,
        native_robot_adapter,
    )

    class CameraCfg:
        def __init__(self, prim_path):
            self.prim_path = prim_path

    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = dataclasses.dataclass
    camera_cfg = types.ModuleType("isaaclab.sensors.camera.camera_cfg")
    camera_cfg.CameraCfg = CameraCfg
    for name, module in (("isaaclab", types.ModuleType("isaaclab")), ("isaaclab.utils", utils),
                         ("isaaclab.sensors", types.ModuleType("isaaclab.sensors")),
                         ("isaaclab.sensors.camera", types.ModuleType("isaaclab.sensors.camera")),
                         ("isaaclab.sensors.camera.camera_cfg", camera_cfg)):
        monkeypatch.setitem(sys.modules, name, module)

    @dataclasses.dataclass
    class DroidCameraCfg:  # Arena's rig: three camera fields and a tiled flag
        external_camera: CameraCfg = CameraCfg("{ENV_REGEX_NS}/Robot/panda_link0/external_camera")
        external_camera_2: CameraCfg = CameraCfg("{ENV_REGEX_NS}/Robot/panda_link0/external_camera_2")
        wrist_camera: CameraCfg = CameraCfg("{ENV_REGEX_NS}/Robot/Gripper/wrist_camera")
        use_tiled_camera = False

        def set_use_tiled_camera(self, value):
            self.use_tiled_camera = value

    embodiment = types.SimpleNamespace(camera_config=DroidCameraCfg())
    original = embodiment.camera_config
    add_droid_exterior_right_slot(embodiment, [{"role": "external"}, {"role": "wrist"}, {"role": "overview"}])
    assert embodiment.camera_config is original  # a two-view plan keeps Arena's rig untouched
    add_droid_exterior_right_slot(embodiment, [{"role": "external"}, {"role": "exterior_right"}])
    rig = embodiment.camera_config
    assert isinstance(rig, DroidCameraCfg) and [field.name for field in dataclasses.fields(rig)] == [
        "external_camera", "external_camera_2", "wrist_camera", EXTERIOR_RIGHT_RUNTIME_NAME]
    assert rig.external_camera is original.external_camera and rig.use_tiled_camera is False
    slot = getattr(rig, EXTERIOR_RIGHT_RUNTIME_NAME)
    assert slot is not original.external_camera_2 and slot.prim_path == "{ENV_REGEX_NS}/exterior_right_camera"
    # The DROID adapter declares it as an optional policy camera, never a required one.
    adapter = native_robot_adapter("franka_panda")
    assert ("exterior_right", EXTERIOR_RIGHT_RUNTIME_NAME, "world") in adapter.camera_roles
    assert "exterior_right" in adapter.policy_camera_roles
    assert "exterior_right" not in adapter.required_camera_roles
