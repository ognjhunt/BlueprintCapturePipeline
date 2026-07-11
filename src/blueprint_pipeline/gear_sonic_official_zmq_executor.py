"""One-step client for the official GEAR-SONIC protocol-v4 controller stack.

The official C++ deployment process remains persistent. This client publishes
one UNITREE_G1_SONIC latent action to its documented ``pose`` endpoint, waits
for the corresponding ``g1_debug`` controller result, and derives FK landmarks
with the official G1 MuJoCo model. It never decodes the latent action itself.

Controller results must carry the pinned protocol-v4 joint-order schema
(:mod:`blueprint_pipeline.gear_sonic_joint_order_contract`); positional-only
results are rejected fail-closed and FK targets are applied by joint name.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .gear_sonic_joint_order_contract import (
    PROTOCOL_V4_BODY_JOINT_NAMES,
    PROTOCOL_V4_FULL_JOINT_ORDER,
    PINNED_WBC_SOURCE_REVISION,
    build_isaac_dof_mapping,
    validate_model_joint_names,
    pinned_controller_joint_order,
)

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
BODY_DIM = len(PROTOCOL_V4_BODY_JOINT_NAMES)
ACTION_DIM = MOTION_TOKEN_DIM + 2 * HAND_DIM
STATE_TOPIC = b"g1_debug"


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_controller_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    revision = completed.stdout.strip().lower() if completed.returncode == 0 else ""
    if revision != PINNED_WBC_SOURCE_REVISION:
        raise RuntimeError("official_gear_sonic_controller_revision_mismatch")
    return revision


def _finite_vector(value: Any, *, size: int, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name}_missing")
    result = [float(item) for item in value]
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name}_dimension_or_value_invalid")
    return result


def _token_matches(observed: Any, motion_token: Sequence[float]) -> bool:
    if isinstance(observed, (str, bytes)) or not isinstance(observed, Sequence):
        return False
    try:
        values = [float(item) for item in observed]
    except (TypeError, ValueError):
        return False
    if len(values) != len(motion_token):
        return False
    return all(
        math.isfinite(value) and abs(value - float(expected)) <= 1e-6
        for value, expected in zip(values, motion_token)
    )


def _zmq_pubsub_roundtrip(
    *,
    pose_message: bytes,
    command_message: bytes,
    motion_token: Sequence[float],
    timeout_seconds: float,
    action_endpoint: str | None = None,
    state_endpoint: str | None = None,
    state_topic: bytes = STATE_TOPIC,
    slow_joiner_grace_seconds: float = 0.3,
) -> dict[str, Any]:
    """Real PUB/SUB roundtrip: bind the action PUB, connect the state SUB.

    Stale controller states whose ``token_state`` does not match this
    attempt's motion token are discarded; only the matching reply returns.
    """

    import msgpack  # type: ignore
    import zmq  # type: ignore

    action = action_endpoint or (
        f"tcp://{os.getenv(ACTION_HOST_ENV, '127.0.0.1')}:"
        f"{int(os.getenv(ACTION_PORT_ENV, '5556'))}"
    )
    state_address = state_endpoint or (
        f"tcp://{os.getenv(STATE_HOST_ENV, '127.0.0.1')}:"
        f"{int(os.getenv(STATE_PORT_ENV, '5557'))}"
    )
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    subscriber = context.socket(zmq.SUB)
    publisher.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.SUBSCRIBE, state_topic)
    publisher.bind(action)
    subscriber.connect(state_address)
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    try:
        time.sleep(max(0.0, float(slow_joiner_grace_seconds)))  # PUB/SUB slow joiner.
        publisher.send(command_message)
        while time.monotonic() < deadline:
            publisher.send(pose_message)
            events = dict(poller.poll(100))
            if subscriber not in events:
                continue
            raw = subscriber.recv()
            if not raw.startswith(state_topic):
                continue
            state = msgpack.unpackb(raw[len(state_topic):], raw=False)
            if not isinstance(state, Mapping):
                continue
            if _token_matches(state.get("token_state"), motion_token):
                return dict(state)
        raise TimeoutError("official_gear_sonic_matching_controller_state_timeout")
    finally:
        publisher.close()
        subscriber.close()
        context.term()


def _zmq_roundtrip(
    *, motion_token: Sequence[float], left_hand: Sequence[float],
    right_hand: Sequence[float], frame_index: int, timeout_seconds: float
) -> dict[str, Any]:
    import numpy as np  # type: ignore
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # type: ignore
        build_command_message,
        pack_pose_message,
    )

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
    return _zmq_pubsub_roundtrip(
        pose_message=pose,
        command_message=build_command_message(start=True, stop=False, planner=False),
        motion_token=[float(item) for item in token.reshape(-1)],
        timeout_seconds=timeout_seconds,
    )


def _official_mujoco_fk(
    *, model_path: Path, body_positions: Sequence[float],
    left_hand: Sequence[float], right_hand: Sequence[float]
) -> tuple[list[str], list[float], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply protocol-v4 targets to the pinned model by joint name.

    The model must expose exactly the pinned 43-joint set; each value is
    written to the qpos address of its named joint, never positionally.
    """

    import mujoco  # type: ignore

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    model_joint_names: list[str] = []
    for index in range(model.njnt):
        if int(model.jnt_type[index]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        model_joint_names.append(str(name) if name else "")
    validate_model_joint_names(model_joint_names)
    body = _finite_vector(body_positions, size=BODY_DIM, name="official_body_q_target")
    left = _finite_vector(left_hand, size=HAND_DIM, name="official_left_hand_target")
    right = _finite_vector(right_hand, size=HAND_DIM, name="official_right_hand_target")
    names = list(PROTOCOL_V4_FULL_JOINT_ORDER)
    positions = body + left + right
    applied: list[dict[str, Any]] = []
    for protocol_index, (joint_name, value) in enumerate(zip(names, positions)):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError("official_gear_sonic_mujoco_model_joint_names_missing")
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_address] = float(value)
        applied.append(
            {
                "joint_name": joint_name,
                "protocol_index": protocol_index,
                "model_qpos_address": qpos_address,
                "applied_value": float(value),
            }
        )
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
    return names, positions, landmarks, applied


def validate_live_isaac_articulation(joint_names: Sequence[str]) -> list[dict[str, Any]]:
    """Hook for validating a live Isaac articulation joint list before use."""

    return build_isaac_dof_mapping(joint_names)


def execute(
    request: Mapping[str, Any], *,
    transport: Callable[..., Mapping[str, Any]] = _zmq_roundtrip,
    fk_solver: Callable[
        ..., tuple[list[str], list[float], list[dict[str, Any]], list[dict[str, Any]]]
    ] = _official_mujoco_fk,
    isaac_joint_names: Sequence[str] | None = None,
    controller_revision_resolver: Callable[[Path], str] = _pinned_controller_revision,
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
        state.get("body_q_target"), size=BODY_DIM, name="official_body_q_target"
    )
    root = Path(os.getenv(ROOT_ENV, DEFAULT_ROOT)).resolve()
    model = Path(os.getenv(MODEL_ENV, DEFAULT_MODEL)).resolve()
    if root.name != "wbc" or not (root / "gear_sonic_deploy").is_dir():
        raise RuntimeError("official_gear_sonic_repository_missing")
    if not model.is_file() or root not in model.parents:
        raise RuntimeError("official_gear_sonic_robot_model_missing_or_outside_repository")
    controller_revision = controller_revision_resolver(root)
    joint_order = pinned_controller_joint_order(controller_revision)
    controller_left = _finite_vector(
        state.get("last_left_hand_action"),
        size=HAND_DIM,
        name="official_left_hand_target",
    )
    controller_right = _finite_vector(
        state.get("last_right_hand_action"),
        size=HAND_DIM,
        name="official_right_hand_target",
    )
    names, positions, landmarks, applied_dof_mapping = fk_solver(
        model_path=model,
        body_positions=body_target,
        left_hand=controller_left,
        right_hand=controller_right,
    )
    isaac_dof_mapping = (
        build_isaac_dof_mapping(isaac_joint_names)
        if isaac_joint_names is not None
        else None
    )
    return {
        "status": "completed",
        "runtime_result_id": f"gear-sonic-zmq-{uuid.uuid4().hex}",
        "source_action_sha256": expected_sha,
        "landmarks": landmarks,
        "joint_positions": positions,
        "joint_names": names,
        "joint_order_schema_version": joint_order["schema_version"],
        "mapping_digest": joint_order["mapping_digest"],
        "controller_revision": controller_revision,
        "mapping_source": joint_order["mapping_source"],
        "robot_model_sha256": _sha256_file(model),
        "applied_dof_mapping": applied_dof_mapping,
        "isaac_dof_mapping": isaac_dof_mapping,
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
