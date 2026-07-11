"""One-step client for the official GEAR-SONIC protocol-v4 controller stack.

The official C++ deployment process remains persistent. This client publishes
one UNITREE_G1_SONIC latent action to its documented ``pose`` endpoint, waits
for the corresponding ``g1_debug`` controller result, and derives FK landmarks
with the official G1 MuJoCo model. It never decodes the latent action itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT_ENV = "BLUEPRINT_GEAR_SONIC_ROOT"
MODEL_ENV = "BLUEPRINT_GEAR_SONIC_ROBOT_MODEL"
INPUT_ENV = "BLUEPRINT_GEAR_SONIC_INPUT"
OUTPUT_ENV = "BLUEPRINT_GEAR_SONIC_OUTPUT"
ACTION_HOST_ENV = "BLUEPRINT_GEAR_SONIC_ACTION_HOST"
ACTION_PORT_ENV = "BLUEPRINT_GEAR_SONIC_ACTION_PORT"
STATE_HOST_ENV = "BLUEPRINT_GEAR_SONIC_STATE_HOST"
STATE_PORT_ENV = "BLUEPRINT_GEAR_SONIC_STATE_PORT"
DEFAULT_ROOT = "/opt/wbc"
DEFAULT_MODEL = "/opt/wbc/gear_sonic_deploy/g1/g1_29dof_with_hand.xml"
MOTION_TOKEN_DIM = 64
HAND_DIM = 7
ACTION_DIM = MOTION_TOKEN_DIM + 2 * HAND_DIM


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _finite_vector(value: Any, *, size: int, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name}_missing")
    result = [float(item) for item in value]
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name}_dimension_or_value_invalid")
    return result


def _zmq_roundtrip(
    *, motion_token: Sequence[float], left_hand: Sequence[float],
    right_hand: Sequence[float], frame_index: int, timeout_seconds: float
) -> dict[str, Any]:
    import msgpack  # type: ignore
    import numpy as np  # type: ignore
    import zmq  # type: ignore
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # type: ignore
        build_command_message,
        pack_pose_message,
    )

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    subscriber = context.socket(zmq.SUB)
    publisher.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "g1_debug")
    publisher.bind(
        f"tcp://{os.getenv(ACTION_HOST_ENV, '127.0.0.1')}:{int(os.getenv(ACTION_PORT_ENV, '5556'))}"
    )
    subscriber.connect(
        f"tcp://{os.getenv(STATE_HOST_ENV, '127.0.0.1')}:{int(os.getenv(STATE_PORT_ENV, '5557'))}"
    )
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    token = np.asarray(motion_token, dtype=np.float32)
    pose = pack_pose_message(
        {
            "token_state": token.reshape(1, -1),
            "frame_index": np.asarray([frame_index], dtype=np.int64),
            "left_hand_joints": np.asarray(left_hand, dtype=np.float32).reshape(1, -1),
            "right_hand_joints": np.asarray(right_hand, dtype=np.float32).reshape(1, -1),
        },
        topic="pose",
        version=4,
    )
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    try:
        time.sleep(0.3)  # PUB/SUB slow-joiner boundary.
        publisher.send(build_command_message(start=True, stop=False, planner=False))
        while time.monotonic() < deadline:
            publisher.send(pose)
            events = dict(poller.poll(100))
            if subscriber not in events:
                continue
            raw = subscriber.recv()
            state = msgpack.unpackb(raw[len(b"g1_debug") :], raw=False)
            observed = state.get("token_state")
            if isinstance(observed, Sequence) and np.allclose(
                np.asarray(observed, dtype=np.float32).reshape(-1), token, atol=1e-6
            ):
                return dict(state)
        raise TimeoutError("official_gear_sonic_matching_controller_state_timeout")
    finally:
        publisher.close()
        subscriber.close()
        context.term()


def _official_mujoco_fk(
    *, model_path: Path, body_positions: Sequence[float],
    left_hand: Sequence[float], right_hand: Sequence[float]
) -> tuple[list[str], list[float], list[dict[str, Any]]]:
    import mujoco  # type: ignore

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_rows: list[tuple[int, str]] = []
    for index in range(model.njnt):
        if int(model.jnt_type[index]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if name:
            joint_rows.append((int(model.jnt_qposadr[index]), str(name)))
    joint_rows.sort()
    positions = list(body_positions) + list(left_hand) + list(right_hand)
    if len(joint_rows) < len(positions):
        raise RuntimeError("official_gear_sonic_model_joint_dimension_too_small")
    selected = joint_rows[: len(positions)]
    for (qpos_address, _), value in zip(selected, positions):
        data.qpos[qpos_address] = float(value)
    mujoco.mj_forward(model, data)
    landmarks: list[dict[str, Any]] = []
    for body_index in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_index) or ""
        lower = name.lower()
        if not any(term in lower for term in ("shoulder", "elbow", "wrist", "hand")):
            continue
        xyz = data.xpos[body_index]
        landmarks.append(
            {"name": name, "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}
        )
    if not landmarks:
        raise RuntimeError("official_gear_sonic_fk_landmarks_missing")
    return [name for _, name in selected], positions, landmarks


def execute(
    request: Mapping[str, Any], *,
    transport: Callable[..., Mapping[str, Any]] = _zmq_roundtrip,
    fk_solver: Callable[..., tuple[list[str], list[float], list[dict[str, Any]]]] = _official_mujoco_fk,
) -> dict[str, Any]:
    action = dict(request.get("action") or {})
    expected_sha = str(request.get("source_action_sha256") or "")
    if _canonical(action) != expected_sha:
        raise ValueError("official_gear_sonic_request_action_sha256_mismatch")
    vector = _finite_vector(
        action.get("sonic_action_chunk") or action.get("action_chunk"),
        size=ACTION_DIM,
        name="unitree_g1_sonic_action",
    )
    motion, left, right = vector[:MOTION_TOKEN_DIM], vector[64:71], vector[71:78]
    state = dict(
        transport(
            motion_token=motion,
            left_hand=left,
            right_hand=right,
            frame_index=int(request.get("step_index") or 0),
            timeout_seconds=120.0,
        )
    )
    body_target = _finite_vector(
        state.get("body_q_target"), size=29, name="official_body_q_target"
    )
    root = Path(os.getenv(ROOT_ENV, DEFAULT_ROOT)).resolve()
    model = Path(os.getenv(MODEL_ENV, DEFAULT_MODEL)).resolve()
    if root.name != "wbc" or not (root / "gear_sonic_deploy").is_dir():
        raise RuntimeError("official_gear_sonic_repository_missing")
    if not model.is_file() or root not in model.parents:
        raise RuntimeError("official_gear_sonic_robot_model_missing_or_outside_repository")
    names, positions, landmarks = fk_solver(
        model_path=model,
        body_positions=body_target,
        left_hand=left,
        right_hand=right,
    )
    return {
        "status": "completed",
        "runtime_result_id": f"gear-sonic-zmq-{uuid.uuid4().hex}",
        "source_action_sha256": expected_sha,
        "landmarks": landmarks,
        "joint_positions": positions,
        "joint_names": names,
        "proprioceptive_state": {
            "body_q_measured": state.get("body_q_measured"),
            "base_quat_measured": state.get("base_quat_measured"),
            "official_controller_protocol": 4,
        },
        "state_timestamp": str(state.get("ros_timestamp") or time.time_ns()),
    }


def main() -> int:
    request = json.loads(Path(os.environ[INPUT_ENV]).read_text(encoding="utf-8"))
    result = execute(request)
    Path(os.environ[OUTPUT_ENV]).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
