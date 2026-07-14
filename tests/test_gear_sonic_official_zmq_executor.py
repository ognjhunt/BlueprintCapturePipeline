from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline import gear_sonic_joint_order_contract as contract
from blueprint_pipeline import gear_sonic_official_zmq_executor as executor

FIXTURE_MODEL = (
    Path(__file__).parent / "fixtures" / "gear_sonic_g1_min" / "g1_29dof_with_hand_min.xml"
)


def _request(action: dict) -> dict:
    return {
        "step_index": 3,
        "action": action,
        "source_action_sha256": hashlib.sha256(
            json.dumps(action, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _controller_state(motion_token, left_hand=None, right_hand=None, **overrides) -> dict:
    state = {
        "token_state": motion_token,
        "body_q_target": [0.1] * 29,
        "body_q_measured": [0.0] * 29,
        "last_left_hand_action": list(left_hand) if left_hand is not None else [0.0] * 7,
        "last_right_hand_action": list(right_hand) if right_hand is not None else [0.0] * 7,
        "base_quat_measured": [1.0, 0.0, 0.0, 0.0],
        "ros_timestamp": 123,
    }
    state.update(overrides)
    return state


def _echoing_transport(calls=None):
    def transport(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return _controller_state(
            kwargs["motion_token"],
            left_hand=kwargs["left_hand"],
            right_hand=kwargs["right_hand"],
        )

    return transport


def _fake_fk(**kwargs):
    names = list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
    positions = list(kwargs["body_positions"]) + list(kwargs["left_hand"]) + list(
        kwargs["right_hand"]
    )
    applied = [
        {"joint_name": name, "protocol_index": index, "model_qpos_address": index + 7}
        for index, name in enumerate(names)
    ]
    return names, positions, [
        {"name": "right_wrist_yaw_link", "x": 0.1, "y": 0.2, "z": 1.0}
    ], applied


def _install_model(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "wbc"
    model = root / "gear_sonic_deploy" / "g1" / "g1_29dof_with_hand.xml"
    model.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE_MODEL, model)
    return root, model


def _execute(request, **kwargs):
    return executor.execute(
        request,
        controller_revision_resolver=lambda _root: contract.PINNED_WBC_SOURCE_REVISION,
        **kwargs,
    )


def test_pinned_controller_revision_accepts_runtime_marker_without_git(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wbc"
    root.mkdir()
    (root / ".blueprint-source-revision").write_text(
        contract.PINNED_WBC_SOURCE_REVISION + "\n", encoding="utf-8"
    )
    assert executor._pinned_controller_revision(root) == (
        contract.PINNED_WBC_SOURCE_REVISION
    )


def test_pinned_controller_revision_rejects_wrong_runtime_marker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wbc"
    root.mkdir()
    (root / ".blueprint-source-revision").write_text("wrong\n", encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="official_gear_sonic_controller_revision_mismatch"
    ):
        executor._pinned_controller_revision(root)


@pytest.fixture()
def wbc_env(tmp_path, monkeypatch):
    root, model = _install_model(tmp_path)
    monkeypatch.setenv(executor.ROOT_ENV, str(root))
    monkeypatch.setenv(executor.MODEL_ENV, str(model))
    return root, model


def test_executor_sends_78d_sonic_action_to_official_protocol_and_uses_fk(
    wbc_env,
) -> None:
    _, model = wbc_env
    action = {"sonic_action_chunk": [float(index) / 100 for index in range(78)]}
    calls = []
    transport = _echoing_transport(calls)

    def fk_solver(**kwargs):
        assert kwargs["model_path"] == model
        assert kwargs["body_positions"] == [0.1] * 29
        return _fake_fk(**kwargs)

    result = _execute(_request(action), transport=transport, fk_solver=fk_solver)

    assert len(calls) == 1
    assert len(calls[0]["motion_token"]) == 64
    assert calls[0]["left_hand"] == action["sonic_action_chunk"][64:71]
    assert calls[0]["right_hand"] == action["sonic_action_chunk"][71:78]
    assert result["status"] == "completed"
    assert result["proprioceptive_state"]["official_controller_protocol"] == 4
    assert result["joint_order_schema_version"] == contract.JOINT_ORDER_SCHEMA_VERSION
    assert result["mapping_digest"] == contract.PROTOCOL_V4_MAPPING_DIGEST
    assert result["controller_revision"] == contract.PINNED_WBC_SOURCE_REVISION
    assert result["robot_model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert [row["joint_name"] for row in result["applied_dof_mapping"]] == list(
        contract.PROTOCOL_V4_FULL_JOINT_ORDER
    )
    assert result["joint_names"] == list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)


def test_executor_rejects_shape_only_or_nonfinite_action(wbc_env) -> None:
    with pytest.raises(ValueError, match="dimension_or_value_invalid"):
        executor.execute(_request({"action_chunk": [0.0] * 77}))
    with pytest.raises(ValueError, match="dimension_or_value_invalid"):
        executor.execute(_request({"action_chunk": [0.0] * 77 + [float("nan")]}))


def test_executor_accepts_official_positional_controller_state(wbc_env) -> None:
    action = {"sonic_action_chunk": [0.0] * 78}

    def transport(**kwargs):
        return _controller_state(kwargs["motion_token"])

    result = _execute(_request(action), transport=transport, fk_solver=_fake_fk)
    assert result["mapping_source"] == "pinned_wbc_mujoco_order"


def test_executor_rejects_unpinned_installed_controller_revision(wbc_env) -> None:
    action = {"sonic_action_chunk": [0.0] * 78}

    def transport(**kwargs):
        return _controller_state(kwargs["motion_token"])

    with pytest.raises(ValueError, match="controller_revision_mismatch"):
        executor.execute(
            _request(action),
            transport=transport,
            fk_solver=_fake_fk,
            controller_revision_resolver=lambda _root: "unreviewed-revision",
        )


def test_executor_validates_live_isaac_articulation_joints(wbc_env) -> None:
    action = {"sonic_action_chunk": [0.0] * 78}

    def transport(**kwargs):
        return _controller_state(kwargs["motion_token"])

    live = list(reversed(contract.PROTOCOL_V4_FULL_JOINT_ORDER))
    result = _execute(
        _request(action),
        transport=transport,
        fk_solver=_fake_fk,
        isaac_joint_names=live,
    )
    assert len(result["isaac_dof_mapping"]) == 43
    for row in result["isaac_dof_mapping"]:
        assert live[row["articulation_dof_index"]] == row["joint_name"]

    with pytest.raises(ValueError, match="isaac_articulation_joint_names_missing"):
        _execute(
            _request(action),
            transport=transport,
            fk_solver=_fake_fk,
            isaac_joint_names=live[:-1],
        )


def _mujoco_or_skip():
    return pytest.importorskip("mujoco", reason="mujoco_not_installed_in_venv")


def test_real_fk_maps_values_by_joint_name_not_position() -> None:
    _mujoco_or_skip()
    body = [0.0] * 29
    body[contract.PROTOCOL_V4_BODY_JOINT_NAMES.index("right_wrist_yaw_joint")] = 0.5
    names, positions, _, applied = executor._official_mujoco_fk(
        model_path=FIXTURE_MODEL,
        body_positions=body,
        left_hand=[0.0] * 7,
        right_hand=[0.0] * 7,
    )
    assert names == list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
    assert len(positions) == 43
    by_name = {row["joint_name"]: row for row in applied}
    assert by_name["right_wrist_yaw_joint"]["applied_value"] == 0.5
    assert all(
        row["applied_value"] == 0.0
        for row in applied
        if row["joint_name"] != "right_wrist_yaw_joint"
    )
    # qpos addresses must be unique: no positional collapsing.
    addresses = [row["model_qpos_address"] for row in applied]
    assert len(set(addresses)) == 43


def _landmarks_by_name(landmarks) -> dict[str, tuple[float, float, float]]:
    return {row["name"]: (row["x"], row["y"], row["z"]) for row in landmarks}


def test_real_fk_two_asymmetric_perturbations_move_distinct_joints() -> None:
    _mujoco_or_skip()
    neutral = [0.0] * 29
    left = list(neutral)
    left[contract.PROTOCOL_V4_BODY_JOINT_NAMES.index("left_elbow_joint")] = 0.6
    right = list(neutral)
    right[contract.PROTOCOL_V4_BODY_JOINT_NAMES.index("right_elbow_joint")] = 0.6
    hands = {"left_hand": [0.0] * 7, "right_hand": [0.0] * 7}
    _, _, base_marks, _ = executor._official_mujoco_fk(
        model_path=FIXTURE_MODEL, body_positions=neutral, **hands
    )
    _, _, left_marks, _ = executor._official_mujoco_fk(
        model_path=FIXTURE_MODEL, body_positions=left, **hands
    )
    _, _, right_marks, _ = executor._official_mujoco_fk(
        model_path=FIXTURE_MODEL, body_positions=right, **hands
    )
    base = _landmarks_by_name(base_marks)
    after_left = _landmarks_by_name(left_marks)
    after_right = _landmarks_by_name(right_marks)
    assert after_left != after_right
    # The left perturbation moves the left wrist and leaves the right wrist alone.
    assert after_left["left_wrist_yaw_link"] != base["left_wrist_yaw_link"]
    assert after_left["right_wrist_yaw_link"] == base["right_wrist_yaw_link"]
    assert after_right["right_wrist_yaw_link"] != base["right_wrist_yaw_link"]
    assert after_right["left_wrist_yaw_link"] == base["left_wrist_yaw_link"]


def test_real_fk_rejects_model_with_wrong_joint_set(tmp_path) -> None:
    _mujoco_or_skip()
    mutated = tmp_path / "mutated.xml"
    mutated.write_text(
        FIXTURE_MODEL.read_text(encoding="utf-8").replace(
            "left_elbow_joint", "left_elbow_mystery_joint"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mujoco_model_joint_names_unknown"):
        executor._official_mujoco_fk(
            model_path=mutated,
            body_positions=[0.0] * 29,
            left_hand=[0.0] * 7,
            right_hand=[0.0] * 7,
        )


def test_execute_end_to_end_with_real_fk_carries_sha_and_digest(wbc_env) -> None:
    _mujoco_or_skip()
    action = {"sonic_action_chunk": [0.01] * 78}
    request = _request(action)

    result = _execute(request, transport=_echoing_transport())
    assert result["source_action_sha256"] == request["source_action_sha256"]
    assert result["mapping_digest"] == contract.PROTOCOL_V4_MAPPING_DIGEST
    assert result["joint_names"] == list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
    assert len(result["joint_positions"]) == 43
    assert result["landmarks"]


def test_executor_rejects_stale_hand_echo_before_fk(wbc_env) -> None:
    action = {"sonic_action_chunk": [float(index) / 100 for index in range(78)]}

    def stale_left(**kwargs):
        return _controller_state(
            kwargs["motion_token"],
            last_left_hand_action=[9.9] * 7,
            last_right_hand_action=list(kwargs["right_hand"]),
        )

    with pytest.raises(
        RuntimeError, match="official_gear_sonic_controller_hand_echo_mismatch:left"
    ):
        _execute(_request(action), transport=stale_left, fk_solver=_fake_fk)

    def stale_right(**kwargs):
        return _controller_state(
            kwargs["motion_token"],
            last_left_hand_action=list(kwargs["left_hand"]),
            last_right_hand_action=[9.9] * 7,
        )

    with pytest.raises(
        RuntimeError, match="official_gear_sonic_controller_hand_echo_mismatch:right"
    ):
        _execute(_request(action), transport=stale_right, fk_solver=_fake_fk)


def test_executor_accepts_matching_hand_echo(wbc_env) -> None:
    action = {"sonic_action_chunk": [float(index) / 100 for index in range(78)]}

    def matching(**kwargs):
        return _controller_state(
            kwargs["motion_token"],
            last_left_hand_action=list(kwargs["left_hand"]),
            last_right_hand_action=list(kwargs["right_hand"]),
        )

    result = _execute(_request(action), transport=matching, fk_solver=_fake_fk)
    assert result["status"] == "completed"
    assert result["joint_positions"][29:36] == action["sonic_action_chunk"][64:71]
    assert result["joint_positions"][36:43] == action["sonic_action_chunk"][71:78]
