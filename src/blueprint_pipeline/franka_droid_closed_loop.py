"""Policy-server-neutral DROID closed loop for the Franka can-to-tray task.

The runner is deliberately simulator-only. It renders exact 224x224 external
and live hand-mounted observations, validates the OpenPI DROID contract, asks an
injected policy client for a 10x8 or 15x8 action chunk, executes the first eight
actions at 15 Hz in MuJoCo, and scores only deterministic object-state predicates.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .common import write_json
from .droid_policy_bridge import (
    DROID_ACTION_CHUNK_SHAPE,
    DROID_CONTROL_HZ,
    DROID_INNER_CONTROL_HZ,
    DROID_OPEN_LOOP_HORIZON,
    droid_action_to_mujoco_targets,
    droid_joint_position_action_to_mujoco_targets,
    validate_droid_action_chunk,
    validate_droid_observation,
)
from .franka_can_tray_feasibility import (
    _ARM_SEED,
    _CAN_INITIAL,
    _TRAY_CENTER,
    _solve_ik,
    _stage_model,
)
from .policy_ranking_thesis import canonical_sha256
from .scene_placement.stance_cameras import link_mounted_camera_spec


SCHEMA_VERSION = "franka_droid_closed_loop.v1"
DEFAULT_PROMPT = "Pick up the can and place it inside the marked tray."
DEFAULT_LEARNED_MAX_ACTION_STEPS = 600
_EXTERNAL_CAMERA = {
    "pos": [1.25, -0.85, 1.10],
    "target": [0.43, 0.16, 0.17],
    "fov": 52.0,
}
_PHASE_SPECS = (
    ("approach", (0.5, 0.075, 0.12), 24, 0.0),
    ("grasp", (0.5, 0.075, 0.12), 24, 1.0),
    ("lift", (0.5, 0.075, 0.25), 24, 1.0),
    ("transport", (0.45, 0.32, 0.30), 32, 1.0),
    ("release", (0.45, 0.32, 0.30), 24, 0.0),
    ("retreat", (0.45, 0.32, 0.40), 24, 0.0),
    ("hold", (0.45, 0.32, 0.40), 16, 0.0),
)


class DroidPolicyClient(Protocol):
    """Smallest interface shared by local controls and a remote OpenPI client."""

    policy_id: str

    def infer(self, observation: Mapping[str, Any]) -> Any: ...


class ZeroDroidPolicyClient:
    """Preregistered stationary negative control.

    The gripper command is also zero, matching a literal all-zero 10x8 chunk.
    Under the public DROID convention that leaves the gripper open and cannot
    move the arm toward the can.
    """

    policy_id = "zero_action_negative_control"
    action_space = "joint_velocity"
    action_chunk_rows = 10

    def infer(self, observation: Mapping[str, Any]) -> Any:
        import numpy as np

        del observation
        return np.zeros(DROID_ACTION_CHUNK_SHAPE, dtype=float)


class ScriptedDroidOracleClient:
    """Positive scene-feasibility control expressed through the DROID contract."""

    policy_id = "scripted_ik_positive_control"
    action_space = "joint_velocity"
    action_chunk_rows = 10

    def __init__(self, joint_targets: Mapping[str, Sequence[float]]) -> None:
        import numpy as np

        self._targets = {
            str(key): np.asarray(value, dtype=float) for key, value in joint_targets.items()
        }
        self._cursor = 0
        self._initial_joints: Any | None = None
        self.total_action_steps = sum(int(row[2]) for row in _PHASE_SPECS)

    def _phase_at(self, step: int) -> tuple[str, str, int, int, float]:
        cursor = 0
        previous = "pregrasp"
        for phase, _target, length, gripper in _PHASE_SPECS:
            if step < cursor + int(length):
                return phase, previous, step - cursor, int(length), float(gripper)
            cursor += int(length)
            previous = phase
        phase, _target, _length, gripper = _PHASE_SPECS[-1]
        return phase, "retreat", 0, 1, float(gripper)

    def infer(self, observation: Mapping[str, Any]) -> Any:
        import numpy as np

        planned = np.asarray(observation["observation/joint_position"], dtype=float).copy()
        if self._initial_joints is None:
            self._initial_joints = planned.copy()
        rows: list[Any] = []
        for offset in range(DROID_ACTION_CHUNK_SHAPE[0]):
            phase, previous, within, length, gripper = self._phase_at(
                self._cursor + offset
            )
            ratio = min(1.0, (within + 1) / max(1, length))
            blend = ratio * ratio * (3.0 - 2.0 * ratio)
            phase_start = (
                self._initial_joints if previous == "pregrasp" else self._targets[previous]
            )
            desired = (1.0 - blend) * phase_start + blend * self._targets[phase]
            joint_action = np.clip((desired - planned) / 0.2, -1.0, 1.0)
            action = np.concatenate((joint_action, np.asarray([gripper], dtype=float)))
            rows.append(action)
            planned = desired
        self._cursor += DROID_OPEN_LOOP_HORIZON
        return np.asarray(rows, dtype=float)


class StationaryDroidJointPositionClient:
    """Stationary negative control for absolute joint-position policies."""

    policy_id = "stationary_joint_position_negative_control"
    action_space = "joint_position"
    action_chunk_rows = 10

    def infer(self, observation: Mapping[str, Any]) -> Any:
        import numpy as np

        joints = np.asarray(observation["observation/joint_position"], dtype=float)
        gripper = np.asarray(observation["observation/gripper_position"], dtype=float)
        row = np.concatenate((joints, gripper))
        return np.repeat(row[None, :], self.action_chunk_rows, axis=0)


class ScriptedDroidJointPositionOracleClient:
    """Positive control using OpenPI's absolute DROID joint-position output."""

    policy_id = "scripted_ik_joint_position_positive_control"
    action_space = "joint_position"
    action_chunk_rows = 10

    def __init__(
        self,
        joint_targets: Mapping[str, Sequence[float]],
        *,
        initial_joint_target: Sequence[float] | None = None,
    ) -> None:
        import numpy as np

        self._targets = {
            str(key): np.asarray(value, dtype=float) for key, value in joint_targets.items()
        }
        self._cursor = 0
        self._initial_joints: Any | None = (
            None
            if initial_joint_target is None
            else np.asarray(initial_joint_target, dtype=float).copy()
        )
        self.total_action_steps = sum(int(row[2]) for row in _PHASE_SPECS)

    def infer(self, observation: Mapping[str, Any]) -> Any:
        import numpy as np

        if self._initial_joints is None:
            self._initial_joints = np.asarray(
                observation["observation/joint_position"], dtype=float
            ).copy()
        rows: list[Any] = []
        for offset in range(self.action_chunk_rows):
            phase, previous, within, length, gripper = ScriptedDroidOracleClient._phase_at(
                self, self._cursor + offset
            )
            ratio = min(1.0, (within + 1) / max(1, length))
            blend = ratio * ratio * (3.0 - 2.0 * ratio)
            phase_start = (
                self._initial_joints if previous == "pregrasp" else self._targets[previous]
            )
            desired = (1.0 - blend) * phase_start + blend * self._targets[phase]
            rows.append(np.concatenate((desired, np.asarray([gripper], dtype=float))))
        self._cursor += DROID_OPEN_LOOP_HORIZON
        return np.asarray(rows, dtype=float)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def _extract_action_chunk(response: Any) -> Any:
    if isinstance(response, Mapping):
        for key in ("actions", "action", "action_chunk"):
            if key in response:
                return response[key]
        raise ValueError("policy_response_missing_action_chunk")
    return response


def _camera_from_spec(spec: Mapping[str, Any], mujoco: Any, np: Any) -> Any:
    eye = np.asarray(spec["pos"], dtype=float)
    target = np.asarray(spec["target"], dtype=float)
    offset = eye - target
    distance = float(np.linalg.norm(offset))
    if distance <= 1e-6 or not math.isfinite(distance):
        raise ValueError("camera_eye_target_invalid")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    camera.distance = distance
    camera.azimuth = math.degrees(math.atan2(float(offset[1]), float(offset[0])))
    camera.elevation = -math.degrees(math.asin(float(offset[2]) / distance))
    return camera


def _render_observation(renderer: Any, model: Any, data: Any, spec: Mapping[str, Any], mujoco: Any, np: Any) -> Any:
    model.vis.global_.fovy = float(spec.get("fov", 60.0))
    renderer.update_scene(data, camera=_camera_from_spec(spec, mujoco, np))
    image = renderer.render().copy()
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise RuntimeError("mujoco_renderer_did_not_produce_uint8_224_square")
    return image


def _render_hybrid_external_observation(
    renderer: Any,
    model: Any,
    data: Any,
    spec: Mapping[str, Any],
    background: Any,
    mujoco: Any,
    np: Any,
) -> tuple[Any, int]:
    """Composite live robot/task geoms over a frozen captured-site background."""
    camera = _camera_from_spec(spec, mujoco, np)
    model.vis.global_.fovy = float(spec.get("fov", 60.0))
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    interaction_rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    return _composite_mujoco_interaction(
        background=background,
        interaction_rgb=interaction_rgb,
        segmentation=segmentation,
        geom_object_type=int(mujoco.mjtObj.mjOBJ_GEOM),
        np=np,
    )


def _composite_mujoco_interaction(
    *,
    background: Any,
    interaction_rgb: Any,
    segmentation: Any,
    geom_object_type: int,
    np: Any,
) -> tuple[Any, int]:
    if interaction_rgb.shape != (224, 224, 3) or background.shape != (224, 224, 3):
        raise RuntimeError("hybrid_external_observation_shape_invalid")
    if segmentation.shape != (224, 224, 2):
        raise RuntimeError("mujoco_segmentation_shape_invalid")
    # Channel 0 is the MuJoCo object id and channel 1 is the object type. The
    # generated workcell's only excluded geom is floor (geom id 0); tray,
    # spraycan, and all Panda visual geoms therefore remain action-conditioned.
    mask = (segmentation[:, :, 1] == int(geom_object_type)) & (
        segmentation[:, :, 0] != 0
    )
    composite = np.asarray(background, dtype=np.uint8).copy()
    composite[mask] = interaction_rgb[mask]
    return composite, int(mask.sum())


def _joint_targets(model: Any, data: Any, mujoco: Any, np: Any, site_id: int) -> dict[str, Any]:
    seed = np.asarray(_ARM_SEED, dtype=float)
    data.qpos[:7] = seed
    data.qpos[7:9] = 0.04
    data.qpos[9:16] = (*_CAN_INITIAL, 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    target_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    q = seed
    q = _solve_ik(
        model,
        data,
        mujoco,
        np,
        site_id,
        q,
        (0.5, 0.075, 0.25),
        target_rotation,
    )
    targets: dict[str, Any] = {"pregrasp": q.copy()}
    for phase, target, _length, _gripper in _PHASE_SPECS:
        stationary_source = {
            "grasp": "approach",
            "release": "transport",
            "hold": "retreat",
        }.get(phase)
        if stationary_source is not None:
            targets[phase] = targets[stationary_source].copy()
            q = targets[phase].copy()
            continue
        q = _solve_ik(model, data, mujoco, np, site_id, q, target, target_rotation)
        targets[phase] = q.copy()
    return targets


def _enable_panda_gravity_compensation(model: Any, mujoco: Any) -> list[str]:
    """Approximate DROID's always-on Polymetis impedance-controller support.

    The public DROID controller starts Cartesian impedance before accepting
    non-blocking desired-joint updates.  The stock Menagerie position actuators
    otherwise let a zero DROID command sag under link gravity, which is not the
    upstream controller behavior.  MuJoCo body gravity compensation is applied
    only to descendants of the Panda link0 body; task objects retain gravity.
    """
    link0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    if link0_id < 0:
        raise RuntimeError("panda_link0_body_missing")
    compensated: list[str] = []
    for body_id in range(1, model.nbody):
        ancestor = body_id
        is_panda = False
        while ancestor > 0:
            if ancestor == link0_id:
                is_panda = True
                break
            ancestor = int(model.body_parentid[ancestor])
        if not is_panda:
            continue
        model.body_gravcomp[body_id] = 1.0
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        compensated.append(str(name or body_id))
    if not compensated:
        raise RuntimeError("no_panda_bodies_gravity_compensated")
    return compensated


def prepare_franka_droid_runtime(*, menagerie_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Stage the pinned model and return model/data IDs plus scripted targets."""
    import mujoco
    import numpy as np

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_path = _stage_model(Path(menagerie_root).expanduser().resolve(), output)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = 1.0 / DROID_INNER_CONTROL_HZ
    compensated_bodies = _enable_panda_gravity_compensation(model, mujoco)
    data = mujoco.MjData(model)
    ids = {
        "site": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper"),
        "hand": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand"),
        "can": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spraycan"),
    }
    if min(ids.values()) < 0:
        raise RuntimeError("required_franka_or_can_body_missing")
    targets = _joint_targets(model, data, mujoco, np, ids["site"])
    return {
        "mujoco": mujoco,
        "np": np,
        "model": model,
        "data": data,
        "ids": ids,
        "targets": targets,
        "scene_path": scene_path,
        "gravity_compensated_bodies": compensated_bodies,
    }


def run_franka_droid_closed_loop(
    *,
    runtime: Mapping[str, Any],
    policy_client: DroidPolicyClient,
    output_dir: str | Path,
    prompt: str = DEFAULT_PROMPT,
    max_action_steps: int | None = None,
    settle_seconds: float = 2.0,
    captured_site_background_path: str | Path | None = None,
    external_background_kind: str = "captured_3dgs",
    external_background_scene_id: str | None = None,
    initial_can_position_m: Sequence[float] = _CAN_INITIAL,
) -> dict[str, Any]:
    """Execute one simulator episode through the exact DROID observation seam."""
    if captured_site_background_path is not None and external_background_kind not in {
        "captured_3dgs",
        "controlled_nvidia_usd",
    }:
        raise ValueError("external_background_kind_invalid")
    if captured_site_background_path is None and external_background_scene_id is not None:
        raise ValueError("external_background_scene_id_without_background")
    mujoco = runtime["mujoco"]
    np = runtime["np"]
    model = runtime["model"]
    data = runtime["data"]
    ids = runtime["ids"]
    targets = runtime["targets"]
    gravity_compensated_bodies = list(runtime["gravity_compensated_bodies"])
    output = Path(output_dir).expanduser().resolve()
    frames_dir = output / "observations"
    frames_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=224, width=224)
    initial_can_position = np.asarray(initial_can_position_m, dtype=float)
    if initial_can_position.shape != (3,) or not np.all(np.isfinite(initial_can_position)):
        raise ValueError("initial_can_position_m_invalid")
    if float(np.linalg.norm(initial_can_position - np.asarray(_CAN_INITIAL))) > 0.05:
        raise ValueError("initial_can_position_m_outside_frozen_perturbation_envelope")
    captured_site_background = None
    captured_site_background_sha256 = None
    if captured_site_background_path is not None:
        from PIL import Image

        background_path = Path(captured_site_background_path).expanduser().resolve()
        if not background_path.is_file() or background_path.is_symlink():
            raise FileNotFoundError("captured_site_background_missing_or_unsafe")
        with Image.open(background_path) as image:
            captured_site_background = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if captured_site_background.shape != (224, 224, 3):
            raise ValueError("captured_site_background_must_be_224_square_rgb")
        captured_site_background_sha256 = _sha256_bytes(background_path.read_bytes())

    mujoco.mj_resetData(model, data)
    data.qpos[:7] = targets["pregrasp"]
    data.qpos[7:9] = 0.04
    data.qpos[9:16] = (*initial_can_position, 1.0, 0.0, 0.0, 0.0)
    data.ctrl[:7] = targets["pregrasp"]
    data.ctrl[7] = 0.04
    mujoco.mj_forward(model, data)
    for _ in range(int(0.8 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    initial_can_z = float(data.xpos[ids["can"], 2])
    max_can_z = initial_can_z
    trace: list[dict[str, Any]] = []
    blockers: list[str] = []
    action_step = 0
    episode_limit = int(
        max_action_steps
        if max_action_steps is not None
        else getattr(policy_client, "total_action_steps", 160)
    )
    joint_limits = np.asarray(model.jnt_range[:7], dtype=float)
    action_space = str(getattr(policy_client, "action_space", "joint_velocity"))
    action_chunk_rows = int(getattr(policy_client, "action_chunk_rows", 10))
    if action_space not in {"joint_velocity", "joint_position"}:
        raise ValueError(f"unsupported_droid_action_space:{action_space}")
    if action_chunk_rows not in {10, 15}:
        raise ValueError(f"unsupported_droid_action_chunk_rows:{action_chunk_rows}")
    learned_policy = bool(getattr(policy_client, "learned_policy", False))
    client_evidence_factory = getattr(policy_client, "evidence_summary", None)
    policy_client_evidence = (
        client_evidence_factory() if callable(client_evidence_factory) else None
    )

    try:
        while action_step < episode_limit:
            wrist_spec = link_mounted_camera_spec(
                parent_translation=data.xpos[ids["hand"]],
                parent_rotation_row_major=data.xmat[ids["hand"]],
                mount_translation=(0.0, 0.10, 0.03),
                mount_forward=(0.0, 0.0, 1.0),
                mount_up=(0.0, 1.0, 0.0),
                look_distance_m=0.5,
                fov_deg=82.0,
            )
            interaction_pixel_count = None
            if captured_site_background is None:
                external = _render_observation(
                    renderer, model, data, _EXTERNAL_CAMERA, mujoco, np
                )
            else:
                external, interaction_pixel_count = _render_hybrid_external_observation(
                    renderer,
                    model,
                    data,
                    _EXTERNAL_CAMERA,
                    captured_site_background,
                    mujoco,
                    np,
                )
            wrist = _render_observation(renderer, model, data, wrist_spec, mujoco, np)
            observation = {
                "observation/exterior_image_1_left": external,
                "observation/wrist_image_left": wrist,
                "observation/joint_position": np.asarray(data.qpos[:7], dtype=float).copy(),
                # DROID reports 0=open and 1=closed, while this MuJoCo model's
                # finger coordinate is 0=closed and 0.04=open.
                "observation/gripper_position": np.asarray(
                    [float(1.0 - np.clip(data.qpos[7] / 0.04, 0.0, 1.0))],
                    dtype=float,
                ),
                "prompt": str(prompt),
            }
            observation_blockers = validate_droid_observation(observation)
            if observation_blockers:
                blockers.extend(observation_blockers)
                break

            query_index = len(trace)
            from PIL import Image

            external_path = frames_dir / f"query_{query_index:03d}_external.png"
            wrist_path = frames_dir / f"query_{query_index:03d}_wrist.png"
            Image.fromarray(external).save(external_path)
            Image.fromarray(wrist).save(wrist_path)
            started = time.monotonic()
            try:
                response = policy_client.infer(observation)
                actions = np.asarray(_extract_action_chunk(response), dtype=float)
            except Exception as exc:  # noqa: BLE001 - policy errors are evidence blockers
                blockers.append(f"policy_inference_failed:{type(exc).__name__}:{exc}")
                break
            latency = time.monotonic() - started
            action_blockers = validate_droid_action_chunk(
                actions, expected_rows=action_chunk_rows
            )
            if action_blockers:
                blockers.extend(action_blockers)
                break

            executed: list[dict[str, Any]] = []
            for row in actions[:DROID_OPEN_LOOP_HORIZON]:
                if action_step >= episode_limit:
                    break
                if action_space == "joint_velocity":
                    mapped = droid_action_to_mujoco_targets(
                        row,
                        current_joint_position=data.qpos[:7],
                        joint_limits=joint_limits,
                    )
                else:
                    mapped = droid_joint_position_action_to_mujoco_targets(
                        row,
                        joint_limits=joint_limits,
                    )
                target_arm_control = np.asarray(
                    mapped["joint_position_target_rad"], dtype=float
                )
                target_gripper_control = float(mapped["gripper_position_target_m"])
                # DROID's non-blocking controller immediately updates the
                # desired joint positions, then the outer loop waits for the
                # remainder of the 15 Hz interval.  The robot-side impedance
                # controller itself runs at 1 kHz.
                data.ctrl[:7] = target_arm_control
                data.ctrl[7] = target_gripper_control
                end_time = float(data.time) + 1.0 / DROID_CONTROL_HZ
                while float(data.time) + 0.5 * float(model.opt.timestep) < end_time:
                    mujoco.mj_step(model, data)
                    max_can_z = max(max_can_z, float(data.xpos[ids["can"], 2]))
                executed.append(
                    {
                        "action_step": action_step,
                        "mapped": _jsonable(mapped),
                        "joint_position_rad": [float(value) for value in data.qpos[:7]],
                        "gripper_position_m": float(data.qpos[7]),
                        "spraycan_center_m": [
                            float(value) for value in data.xpos[ids["can"]]
                        ],
                        "gripper_site_m": [
                            float(value) for value in data.site_xpos[ids["site"]]
                        ],
                    }
                )
                action_step += 1
            trace.append(
                {
                    "query_index": query_index,
                    "action_step_start": executed[0]["action_step"] if executed else action_step,
                    "policy_latency_seconds": latency,
                    "external_png": str(external_path),
                    "external_png_sha256": _sha256_bytes(external_path.read_bytes()),
                    "wrist_png": str(wrist_path),
                    "wrist_png_sha256": _sha256_bytes(wrist_path.read_bytes()),
                    "wrist_camera": wrist_spec,
                    "captured_site_interaction_pixel_count": interaction_pixel_count,
                    "action_chunk_sha256": canonical_sha256(actions.tolist()),
                    "executed_actions": executed,
                }
            )
    finally:
        renderer.close()

    for _ in range(max(0, int(float(settle_seconds) / model.opt.timestep))):
        mujoco.mj_step(model, data)
        max_can_z = max(max_can_z, float(data.xpos[ids["can"], 2]))

    final_position = np.asarray(data.xpos[ids["can"]], dtype=float).copy()
    final_speed = float(np.linalg.norm(data.cvel[ids["can"], 3:]))
    contained = bool(
        abs(float(final_position[0]) - _TRAY_CENTER[0]) <= 0.15
        and abs(float(final_position[1]) - _TRAY_CENTER[1]) <= 0.11
        and 0.10 <= float(final_position[2]) <= 0.14
    )
    lift_delta = float(max_can_z - initial_can_z)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blockers else "completed",
        "policy_id": str(policy_client.policy_id),
        "policy_client_evidence": policy_client_evidence,
        "captured_site_observation": {
            "external_background_used": captured_site_background is not None,
            "external_background_kind": (
                external_background_kind if captured_site_background is not None else None
            ),
            "external_background_scene_id": (
                str(external_background_scene_id)
                if captured_site_background is not None and external_background_scene_id
                else None
            ),
            "external_background_path": (
                str(Path(captured_site_background_path).expanduser().resolve())
                if captured_site_background_path is not None
                else None
            ),
            "external_background_sha256": captured_site_background_sha256,
            "composition": (
                (
                    "frozen_3dgs_background_plus_live_mujoco_robot_task_segmentation"
                    if external_background_kind == "captured_3dgs"
                    else "frozen_nvidia_usd_background_plus_live_mujoco_robot_task_segmentation"
                )
                if captured_site_background is not None
                else "mujoco_only"
            ),
            "dynamic_external_interaction_layer": captured_site_background is not None,
            "dynamic_wrist_view": True,
            "full_site_physics": False,
        },
        "prompt": str(prompt),
        "initial_can_position_m": [float(value) for value in initial_can_position],
        "simulator": {
            "name": "MuJoCo",
            "version": mujoco.__version__,
            "timestep_seconds": float(model.opt.timestep),
        },
        "droid_contract": {
            "observation_resolution": [224, 224],
            "control_hz": DROID_CONTROL_HZ,
            "inner_control_hz": DROID_INNER_CONTROL_HZ,
            "action_space": action_space,
            "action_chunk_shape": [action_chunk_rows, 8],
            "open_loop_horizon": DROID_OPEN_LOOP_HORIZON,
            "desired_joint_update": "immediate_nonblocking",
            "inner_impedance_approximation": {
                "panda_link_gravity_compensation": True,
                "compensated_bodies": gravity_compensated_bodies,
                "task_object_gravity_preserved": True,
            },
        },
        "action_steps_executed": action_step,
        "policy_query_count": len(trace),
        "blockers": blockers,
        "metrics": {
            "initial_spraycan_z_m": initial_can_z,
            "max_spraycan_z_m": max_can_z,
            "lift_delta_m": lift_delta,
            "final_spraycan_center_m": [float(value) for value in final_position],
            "final_linear_speed_m_s": final_speed,
            "contained_in_tray_interior": contained,
            "policy_latency_seconds_total": sum(
                float(row["policy_latency_seconds"]) for row in trace
            ),
        },
        "gates": {
            "contract_valid": not blockers,
            "lift_at_least_0_05m": lift_delta >= 0.05,
            "final_containment": contained,
            "final_stability_below_0_02m_s": final_speed < 0.02,
        },
        "trace": trace,
        "claim_boundary": {
            "simulator_policy_execution": not blockers,
            "learned_policy_execution": bool(
                learned_policy and not blockers and action_step > 0
            ),
            "wam_executed": False,
            "nvidia_warehouse_executed": bool(
                captured_site_background is not None
                and external_background_kind == "controlled_nvidia_usd"
                and not blockers
            ),
            "captured_3dgs_composited": bool(
                captured_site_background is not None
                and external_background_kind == "captured_3dgs"
            ),
            "nvidia_warehouse_is_physical_answer_key": False,
            "isaac_physics_executed": False,
            "physical_success_proven": False,
        },
    }
    result["task_success"] = bool(
        result["gates"]["contract_valid"]
        and result["gates"]["lift_at_least_0_05m"]
        and result["gates"]["final_containment"]
        and result["gates"]["final_stability_below_0_02m_s"]
    )
    result["manifest_sha256"] = canonical_sha256(result)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "franka_droid_closed_loop.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menagerie-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--control",
        choices=(
            "zero",
            "scripted",
            "stationary_joint_position",
            "scripted_joint_position",
            "learned_joint_position",
        ),
        required=True,
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-action-steps", type=int)
    parser.add_argument("--cohort")
    parser.add_argument("--policy-id")
    parser.add_argument("--policy-host")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--policy-api-key-file")
    parser.add_argument("--captured-site-background")
    parser.add_argument(
        "--external-background-kind",
        choices=("captured_3dgs", "controlled_nvidia_usd"),
        default="captured_3dgs",
    )
    parser.add_argument("--external-background-scene-id")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    runtime = prepare_franka_droid_runtime(
        menagerie_root=args.menagerie_root,
        output_dir=output,
    )
    client: DroidPolicyClient
    if args.control == "zero":
        client = ZeroDroidPolicyClient()
    elif args.control == "scripted":
        client = ScriptedDroidOracleClient(runtime["targets"])
    elif args.control == "stationary_joint_position":
        client = StationaryDroidJointPositionClient()
    elif args.control == "scripted_joint_position":
        client = ScriptedDroidJointPositionOracleClient(
            runtime["targets"],
            initial_joint_target=runtime["targets"]["pregrasp"],
        )
    else:
        if not args.cohort or not args.policy_id or not args.policy_host:
            parser.error(
                "learned_joint_position requires --cohort, --policy-id, and --policy-host"
            )
        from .openpi_droid_policy_runtime import (
            OpenPIWebsocketDroidPolicyClient,
            load_policy_spec,
        )

        api_key = None
        if args.policy_api_key_file:
            api_key = (
                Path(args.policy_api_key_file)
                .expanduser()
                .read_text(encoding="utf-8")
                .strip()
            )
            if not api_key:
                parser.error("--policy-api-key-file is empty")
        client = OpenPIWebsocketDroidPolicyClient(
            spec=load_policy_spec(args.cohort, policy_id=args.policy_id),
            host=args.policy_host,
            port=args.policy_port,
            api_key=api_key,
        )
    max_action_steps = args.max_action_steps
    if args.control == "learned_joint_position" and max_action_steps is None:
        max_action_steps = DEFAULT_LEARNED_MAX_ACTION_STEPS
    result = run_franka_droid_closed_loop(
        runtime=runtime,
        policy_client=client,
        output_dir=output,
        prompt=args.prompt,
        max_action_steps=max_action_steps,
        captured_site_background_path=args.captured_site_background,
        external_background_kind=args.external_background_kind,
        external_background_scene_id=args.external_background_scene_id,
    )
    if result["status"] == "blocked":
        return 2
    if args.control.startswith("scripted") and not result["task_success"]:
        return 3
    if args.control in {"zero", "stationary_joint_position"} and result["task_success"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
