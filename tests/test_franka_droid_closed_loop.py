import numpy as np
import pytest

from blueprint_pipeline.franka_droid_closed_loop import (
    ScriptedDroidOracleClient,
    ScriptedDroidJointPositionOracleClient,
    StationaryDroidJointPositionClient,
    ZeroDroidPolicyClient,
    _camera_from_spec,
    _composite_mujoco_interaction,
    _enable_panda_gravity_compensation,
    _extract_action_chunk,
)
from blueprint_pipeline.droid_policy_bridge import validate_droid_action_chunk


def _observation(joints: np.ndarray | None = None) -> dict:
    return {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": (
            np.zeros(7, dtype=float) if joints is None else joints
        ),
        "observation/gripper_position": np.asarray([1.0]),
        "prompt": "Pick up the can and place it inside the marked tray.",
    }


def test_zero_control_is_exact_ten_by_eight_chunk() -> None:
    actions = ZeroDroidPolicyClient().infer(_observation())
    assert validate_droid_action_chunk(actions) == []
    assert np.count_nonzero(actions) == 0


def test_scripted_control_uses_same_chunk_contract_and_advances() -> None:
    targets = {
        "approach": np.full(7, 0.1),
        "grasp": np.full(7, 0.1),
        "lift": np.full(7, 0.2),
        "transport": np.full(7, 0.3),
        "release": np.full(7, 0.3),
        "retreat": np.full(7, 0.4),
        "hold": np.full(7, 0.4),
    }
    client = ScriptedDroidOracleClient(targets)
    first = client.infer(_observation())
    second = client.infer(_observation(np.full(7, 0.08)))
    assert validate_droid_action_chunk(first) == []
    assert validate_droid_action_chunk(second) == []
    assert first.shape == (10, 8)
    assert client.total_action_steps == 168
    assert np.all(first[:, 7] == 0.0)
    assert not np.array_equal(first, second)


def test_joint_position_controls_hold_or_emit_absolute_targets() -> None:
    observation = _observation(np.full(7, 0.08))
    stationary = StationaryDroidJointPositionClient().infer(observation)
    assert stationary.shape == (10, 8)
    assert np.all(stationary[:, :7] == 0.08)
    targets = {
        key: np.full(7, value)
        for key, value in {
            "approach": 0.1,
            "grasp": 0.1,
            "lift": 0.2,
            "transport": 0.3,
            "release": 0.3,
            "retreat": 0.4,
            "hold": 0.4,
        }.items()
    }
    scripted = ScriptedDroidJointPositionOracleClient(targets).infer(observation)
    assert scripted.shape == (10, 8)
    assert np.all(scripted[:, 7] == 0.0)
    assert np.all(np.isfinite(scripted))


def test_extract_action_chunk_accepts_openpi_mapping_and_fails_closed() -> None:
    expected = np.zeros((10, 8))
    assert _extract_action_chunk({"actions": expected}) is expected
    with pytest.raises(ValueError, match="missing_action_chunk"):
        _extract_action_chunk({"metadata": {}})


def test_hybrid_compositor_excludes_floor_and_preserves_live_geoms() -> None:
    background = np.full((224, 224, 3), 10, dtype=np.uint8)
    interaction = np.full((224, 224, 3), 200, dtype=np.uint8)
    segmentation = np.full((224, 224, 2), -1, dtype=np.int32)
    segmentation[0, 0] = [0, 5]
    segmentation[1, 1] = [82, 5]
    composite, pixel_count = _composite_mujoco_interaction(
        background=background,
        interaction_rgb=interaction,
        segmentation=segmentation,
        geom_object_type=5,
        np=np,
    )
    assert pixel_count == 1
    assert np.all(composite[0, 0] == 10)
    assert np.all(composite[1, 1] == 200)


def test_free_camera_conversion_has_finite_pose() -> None:
    mujoco = pytest.importorskip("mujoco")
    camera = _camera_from_spec(
        {"pos": [1.0, -1.0, 1.0], "target": [0.0, 0.0, 0.0]},
        mujoco,
        np,
    )
    assert camera.distance == pytest.approx(np.sqrt(3.0))
    assert np.isfinite(camera.azimuth)
    assert np.isfinite(camera.elevation)


def test_gravity_compensation_is_scoped_to_panda_bodies() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body name="link0"><body name="link1"/></body>
        <body name="spraycan"><freejoint/><geom type="sphere" size=".01"/></body>
        </worldbody></mujoco>"""
    )
    names = _enable_panda_gravity_compensation(model, mujoco)
    link0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    link1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link1")
    can = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spraycan")
    assert names == ["link0", "link1"]
    assert model.body_gravcomp[link0] == 1.0
    assert model.body_gravcomp[link1] == 1.0
    assert model.body_gravcomp[can] == 0.0
