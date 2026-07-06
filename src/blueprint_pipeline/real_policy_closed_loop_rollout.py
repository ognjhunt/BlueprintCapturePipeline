"""Real closed-loop MuJoCo rollout with a checkpoint-loaded policy in the loop.

This is a simulator command for ``robot_eval_job_orchestrator`` (honoring the
``BLUEPRINT_SIMULATOR_OUTPUT`` / ``BLUEPRINT_SCENARIO_EVAL_MATRIX`` /
``BLUEPRINT_CAPTURE_ROOT`` env contract) that, unlike the route-following
preview harness, queries a policy for actions every control chunk and lets
MuJoCo physics decide what happens.

Loop shape follows SC3-Eval's receding horizon: the policy returns an action
chunk per query, the runner executes the first ``executed_horizon`` actions,
then requeries with a fresh observation. Every query is recorded in a
``policy_requery_trace`` and every control step in a per-attempt control
stream, so scoring reads measured simulator state — never policy claims.

Substrate honesty: results are labeled ``classical_sim_mujoco`` with a
``cartesian_ee_proxy_v1`` embodiment (actuated x/y/z/yaw + parallel gripper;
roll/pitch action dims recorded but unactuated). Nothing here is Unitree G1
locomotion, generated video, or physical-robot proof.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .lerobot_policy_family import (
    LoadedPolicyFamily,
    load_lerobot_policy_checkpoint,
)

SIMULATOR_OUTPUT_SCHEMA_VERSION = "real_policy_closed_loop_simulator_output.v1"
REQUERY_TRACE_SCHEMA_VERSION = "real_policy_requery_trace.v1"
EMBODIMENT_ID = "cartesian_ee_proxy_v1"
SUBSTRATE = "classical_sim_mujoco"

# Scene constants (meters). Table top surface and object rest pose drive the
# lifting / placing thresholds, so they live in one place.
TABLE_TOP_Z = 0.42
OBJECT_HALF_HEIGHT = 0.03
OBJECT_HALF_WIDTH = 0.02
OBJECT_REST_Z = TABLE_TOP_Z + OBJECT_HALF_HEIGHT
FINGER_SLIDE_RANGE = 0.026
FINGER_INNER_OPEN_HALF_GAP = 0.030
GRASP_TIP_OFFSET = 0.03  # ee_site sits this far above object center at grasp.

LIFT_HEIGHT_THRESHOLD_M = 0.08
LIFT_SUSTAIN_STEPS = 5
GRASP_SUSTAIN_STEPS = 3
PLACE_SETTLE_SPEED_MPS = 0.08
PLACE_Z_TOLERANCE_M = 0.025

DEFAULT_CONTROL_HZ = 20.0
DEFAULT_CHUNK_SIZE = 25
DEFAULT_EXECUTED_HORIZON = 16
DEFAULT_MAX_SECONDS = 20.0
PHYSICS_TIMESTEP = 0.005

SC3_CRITERIA = ("language_following", "object_lifting", "object_placing")

PHYSICALLY_MODELED_VARIATIONS = {
    "normal",
    "object_rotation",
    "cart_shifted",
    "blocked_path",
    "edge_case",
    "wrong_object_nearby",
    "narrow_approach_angle",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def tabletop_scene_xml(
    *,
    target_start_xy: Sequence[float],
    target_yaw_rad: float,
    distractor_start_xy: Sequence[float],
    goal_center_xy: Sequence[float],
    obstacle_xy: Sequence[float] | None = None,
) -> str:
    """Self-contained MJCF tabletop: two graspable items, goal zone, EE proxy."""
    half_yaw = target_yaw_rad / 2.0
    target_quat = f"{math.cos(half_yaw):.6f} 0 0 {math.sin(half_yaw):.6f}"
    obstacle_body = ""
    if obstacle_xy is not None:
        obstacle_body = f"""
    <body name="path_obstacle" pos="{obstacle_xy[0]:.4f} {obstacle_xy[1]:.4f} {TABLE_TOP_Z + 0.04:.4f}">
      <geom name="path_obstacle_geom" type="box" size="0.03 0.03 0.04" rgba="0.4 0.4 0.4 1"/>
    </body>"""
    return f"""
<mujoco model="blueprint_real_policy_tabletop">
  <option timestep="{PHYSICS_TIMESTEP}" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="table" pos="0 0 {TABLE_TOP_Z - 0.02:.4f}">
      <geom name="table_top" type="box" size="0.5 0.5 0.02" friction="1.0 0.005 0.0001" rgba="0.6 0.45 0.3 1"/>
    </body>
    <body name="target_item" pos="{target_start_xy[0]:.4f} {target_start_xy[1]:.4f} {OBJECT_REST_Z:.4f}" quat="{target_quat}">
      <freejoint name="target_item_free"/>
      <geom name="target_item_geom" type="box" size="{OBJECT_HALF_WIDTH} {OBJECT_HALF_WIDTH} {OBJECT_HALF_HEIGHT}"
            density="300" friction="1.8 0.02 0.0002" rgba="0.85 0.2 0.2 1"/>
    </body>
    <body name="distractor_item" pos="{distractor_start_xy[0]:.4f} {distractor_start_xy[1]:.4f} {OBJECT_REST_Z:.4f}">
      <freejoint name="distractor_item_free"/>
      <geom name="distractor_item_geom" type="box" size="{OBJECT_HALF_WIDTH} {OBJECT_HALF_WIDTH} {OBJECT_HALF_HEIGHT}"
            density="300" friction="1.8 0.02 0.0002" rgba="0.2 0.3 0.85 1"/>
    </body>{obstacle_body}
    <site name="goal_zone_site" pos="{goal_center_xy[0]:.4f} {goal_center_xy[1]:.4f} {TABLE_TOP_Z + 0.001:.4f}"
          size="0.06 0.06 0.001" type="box" rgba="0.2 0.8 0.2 0.35"/>
    <camera name="obs_cam" pos="0.55 -0.55 0.95" mode="targetbody" target="table"/>
    <body name="ee_base" pos="0 0 0.72">
      <joint name="ee_x" type="slide" axis="1 0 0" range="-0.45 0.45"/>
      <joint name="ee_y" type="slide" axis="0 1 0" range="-0.45 0.45"/>
      <joint name="ee_z" type="slide" axis="0 0 1" range="-0.30 0.15"/>
      <joint name="ee_yaw" type="hinge" axis="0 0 1" range="-3.1416 3.1416"/>
      <geom name="palm" type="box" size="0.035 0.012 0.01" density="800" rgba="0.3 0.3 0.3 1"/>
      <body name="finger_left" pos="0 0.034 -0.045">
        <joint name="finger_left_slide" type="slide" axis="0 -1 0" range="0 {FINGER_SLIDE_RANGE}"/>
        <geom name="finger_left_geom" type="box" size="0.010 0.004 0.035"
              density="800" friction="2.0 0.05 0.001" rgba="0.2 0.2 0.2 1"/>
      </body>
      <body name="finger_right" pos="0 -0.034 -0.045">
        <joint name="finger_right_slide" type="slide" axis="0 1 0" range="0 {FINGER_SLIDE_RANGE}"/>
        <geom name="finger_right_geom" type="box" size="0.010 0.004 0.035"
              density="800" friction="2.0 0.05 0.001" rgba="0.2 0.2 0.2 1"/>
      </body>
      <site name="ee_site" pos="0 0 -0.055" size="0.004" rgba="1 1 0 0.5"/>
    </body>
  </worldbody>
  <actuator>
    <position name="act_ee_x" joint="ee_x" kp="600" kv="80" forcerange="-120 120" ctrlrange="-0.45 0.45"/>
    <position name="act_ee_y" joint="ee_y" kp="600" kv="80" forcerange="-120 120" ctrlrange="-0.45 0.45"/>
    <position name="act_ee_z" joint="ee_z" kp="600" kv="80" forcerange="-160 160" ctrlrange="-0.30 0.15"/>
    <position name="act_ee_yaw" joint="ee_yaw" kp="40" kv="4" forcerange="-20 20" ctrlrange="-3.1416 3.1416"/>
    <position name="act_finger_left" joint="finger_left_slide" kp="120" kv="6" forcerange="-30 30" ctrlrange="0 {FINGER_SLIDE_RANGE}"/>
    <position name="act_finger_right" joint="finger_right_slide" kp="120" kv="6" forcerange="-30 30" ctrlrange="0 {FINGER_SLIDE_RANGE}"/>
  </actuator>
</mujoco>
"""


def _stable_seed(run_id: str, base_seed: int) -> int:
    digest = 0
    for char in run_id:
        digest = (digest * 131 + ord(char)) % 2_147_483_647
    return (digest + int(base_seed)) % 2_147_483_647


def _variation_layout(
    *, variation_name: str, rng: random.Random
) -> dict[str, Any]:
    """Deterministic physical layout per variation; honest about coverage."""
    jitter = lambda scale: (rng.random() * 2.0 - 1.0) * scale  # noqa: E731
    target_xy = [0.10 + jitter(0.02), -0.12 + jitter(0.02)]
    distractor_xy = [-0.14 + jitter(0.02), -0.10 + jitter(0.02)]
    goal_xy = [0.02 + jitter(0.01), 0.16 + jitter(0.01)]
    target_yaw = 0.0
    obstacle_xy: list[float] | None = None
    physically_modeled = variation_name in PHYSICALLY_MODELED_VARIATIONS
    if variation_name == "object_rotation":
        target_yaw = rng.choice([0.35, -0.35, 0.6, -0.6])
    elif variation_name == "cart_shifted":
        target_xy = [target_xy[0] + 0.05, target_xy[1] - 0.04]
    elif variation_name in {"blocked_path", "edge_case"}:
        obstacle_xy = [
            (target_xy[0] + goal_xy[0]) / 2.0,
            (target_xy[1] + goal_xy[1]) / 2.0,
        ]
    elif variation_name == "wrong_object_nearby":
        distractor_xy = [target_xy[0] - 0.07, target_xy[1] + 0.01]
    elif variation_name == "narrow_approach_angle":
        obstacle_xy = [target_xy[0] - 0.08, target_xy[1] - 0.02]
    return {
        "variation_name": variation_name or "normal",
        "variation_physically_modeled": physically_modeled,
        "variation_note": (
            "physical layout perturbation applied"
            if physically_modeled
            else "no physical analog on proprioceptive tabletop substrate; "
            "base layout with variation-specific deterministic seed"
        ),
        "target_start_xy": [round(v, 6) for v in target_xy],
        "target_yaw_rad": round(target_yaw, 6),
        "distractor_start_xy": [round(v, 6) for v in distractor_xy],
        "goal_center_xy": [round(v, 6) for v in goal_xy],
        "goal_radius_m": 0.06,
        "obstacle_xy": [round(v, 6) for v in obstacle_xy] if obstacle_xy else None,
    }


class _PolicyTransport:
    """Query the policy in-process, via one-shot subprocess, or via a
    persistent stdio adapter (required for torch models that must load once)."""

    def __init__(
        self,
        *,
        loaded: LoadedPolicyFamily | None,
        adapter_command: str | None,
        chunk_size: int,
        work_dir: Path,
        adapter_mode: str = "subprocess",
    ) -> None:
        self.loaded = loaded
        self.adapter_command = _string(adapter_command) or None
        self.chunk_size = chunk_size
        self.work_dir = work_dir
        self._server: subprocess.Popen[str] | None = None
        if self.adapter_command and adapter_mode == "persistent":
            self.transport = "persistent_stdio_adapter"
        elif self.adapter_command:
            self.transport = "subprocess_adapter_command"
        elif loaded is not None and loaded.policy is not None:
            self.transport = "in_process_checkpoint_policy"
        else:
            raise RuntimeError(
                "no executable policy: checkpoint not cpu-loadable and no adapter command"
            )

    def close(self) -> None:
        if self._server is not None:
            try:
                if self._server.stdin:
                    self._server.stdin.close()
                self._server.terminate()
                self._server.wait(timeout=15)
            except Exception:
                self._server.kill()
            self._server = None

    def _persistent_query(
        self, observation: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        if self._server is None or self._server.poll() is not None:
            self._server = subprocess.Popen(
                shlex.split(self.adapter_command or ""),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        assert self._server.stdin and self._server.stdout
        self._server.stdin.write(
            json.dumps({"observation": dict(observation)}) + "\n"
        )
        self._server.stdin.flush()
        line = self._server.stdout.readline().strip()
        if not line:
            raise RuntimeError("persistent policy adapter closed its stdout")
        payload = _mapping(json.loads(line))
        if _string(payload.get("status")) != "completed":
            raise RuntimeError(
                f"persistent policy adapter returned status={payload.get('status')} "
                f"error={payload.get('error')}"
            )
        chunk = [
            _mapping(item)
            for item in payload.get("action_chunk") or []
            if isinstance(item, Mapping)
        ]
        if not chunk:
            raise RuntimeError("persistent policy adapter returned no actions")
        return chunk

    @property
    def policy_id(self) -> str:
        if self.loaded is not None and self.loaded.policy is not None:
            return self.loaded.policy.policy_id
        return "adapter_command_policy"

    def query(self, observation: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self.transport == "in_process_checkpoint_policy":
            assert self.loaded is not None and self.loaded.policy is not None
            return self.loaded.policy.select_action_chunk(
                observation, chunk_size=self.chunk_size
            )
        if self.transport == "persistent_stdio_adapter":
            return self._persistent_query(observation)
        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmp:
            obs_path = Path(tmp) / "observation.json"
            out_path = Path(tmp) / "action.json"
            obs_path.write_text(
                json.dumps({"observation": dict(observation)}), encoding="utf-8"
            )
            env = os.environ.copy()
            env["BLUEPRINT_POLICY_ACTION_INPUT"] = str(obs_path)
            env["BLUEPRINT_POLICY_ACTION_OUTPUT"] = str(out_path)
            completed = subprocess.run(
                shlex.split(self.adapter_command or ""),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                env=env,
            )
            if completed.returncode != 0 or not out_path.is_file():
                raise RuntimeError(
                    f"policy adapter command failed (exit {completed.returncode}): "
                    f"{completed.stderr[-500:]}"
                )
            payload = _mapping(json.loads(out_path.read_text(encoding="utf-8")))
            chunk = [
                _mapping(item)
                for item in payload.get("action_chunk") or []
                if isinstance(item, Mapping)
            ]
            if not chunk and isinstance(payload.get("action"), Mapping):
                chunk = [_mapping(payload["action"])]
            if not chunk:
                raise RuntimeError("policy adapter returned no actions")
            return chunk


def _action_7d(action: Mapping[str, Any]) -> list[float]:
    vector = action.get("action_7d")
    if isinstance(vector, Sequence) and len(vector) >= 7:
        return [_number(v, 0.0) for v in vector[:7]]
    delta = [
        _number(v, 0.0)
        for v in (action.get("delta_xyz_m") or [0.0, 0.0, 0.0])
    ][:3]
    rpy = [
        _number(v, 0.0)
        for v in (action.get("delta_rpy_rad") or [0.0, 0.0, 0.0])
    ][:3]
    grip = _number(action.get("gripper_command"), 1.0)
    return delta + rpy + [grip]


def run_real_policy_closed_loop_rollout(
    *,
    checkpoint_dir: str | Path | None,
    scenario_eval_matrix_path: str | Path,
    output_path: str | Path,
    capture_root: str | Path | None = None,
    adapter_command: str | None = None,
    adapter_mode: str = "subprocess",
    render_obs_frames: bool = False,
    render_width: int = 640,
    render_height: int = 480,
    control_hz: float = DEFAULT_CONTROL_HZ,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    executed_horizon: int = DEFAULT_EXECUTED_HORIZON,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    base_seed: int = 20260705,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    generated_at = utc_now_iso()
    output_file = Path(output_path).resolve()
    output_root = output_file.parent / "real_policy_rollout"
    ensure_dir(output_root)

    matrix_payload = _mapping(
        json.loads(Path(scenario_eval_matrix_path).read_text(encoding="utf-8"))
    )
    matrix_runs = [
        _mapping(run)
        for run in matrix_payload.get("runs") or []
        if isinstance(run, Mapping) and _string(run.get("scenario_eval_run_id"))
    ]
    if max_attempts is not None:
        matrix_runs = matrix_runs[: max(1, int(max_attempts))]
    if not matrix_runs:
        raise RuntimeError("scenario_eval_matrix contains no executable runs")

    loaded: LoadedPolicyFamily | None = None
    if checkpoint_dir:
        loaded = load_lerobot_policy_checkpoint(checkpoint_dir)
        if loaded.blockers and not adapter_command:
            payload = _blocked_output(
                generated_at=generated_at,
                blockers=loaded.blockers,
                loaded=loaded,
            )
            write_json(output_file, payload)
            return payload
    transport = _PolicyTransport(
        loaded=loaded,
        adapter_command=adapter_command,
        chunk_size=chunk_size,
        work_dir=output_root,
        adapter_mode=adapter_mode,
    )

    try:
        import mujoco  # type: ignore[import-not-found]
    except Exception as exc:
        payload = _blocked_output(
            generated_at=generated_at,
            blockers=[f"mujoco_import_failed:{exc.__class__.__name__}"],
            loaded=loaded,
        )
        write_json(output_file, payload)
        return payload

    control_dt = 1.0 / max(1e-6, float(control_hz))
    physics_substeps = max(1, int(round(control_dt / PHYSICS_TIMESTEP)))
    max_control_steps = max(1, int(round(float(max_seconds) / control_dt)))
    executed_horizon = max(1, min(int(executed_horizon), int(chunk_size)))

    attempts: list[dict[str, Any]] = []
    requery_rows: list[dict[str, Any]] = []
    obs_frames_rendered = 0
    frames_dir = output_root / "obs_frames"
    if render_obs_frames:
        ensure_dir(frames_dir)

    for run_index, run in enumerate(matrix_runs):
        run_id = _string(run.get("scenario_eval_run_id"))
        variation_name = _string(run.get("variation_name")) or "normal"
        seed = _stable_seed(run_id, base_seed)
        rng = random.Random(seed)
        layout = _variation_layout(variation_name=variation_name, rng=rng)

        model = mujoco.MjModel.from_xml_string(
            tabletop_scene_xml(
                target_start_xy=layout["target_start_xy"],
                target_yaw_rad=layout["target_yaw_rad"],
                distractor_start_xy=layout["distractor_start_xy"],
                goal_center_xy=layout["goal_center_xy"],
                obstacle_xy=layout["obstacle_xy"],
            )
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        episode_renderer = None
        if render_obs_frames:
            episode_renderer = mujoco.Renderer(
                model, height=int(render_height), width=int(render_width)
            )
        latest_frame: dict[str, Any] = {"path": None}

        name2geom = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid): gid
            for gid in range(model.ngeom)
        }
        finger_geoms = {name2geom["finger_left_geom"], name2geom["finger_right_geom"]}
        item_geoms = {
            name2geom["target_item_geom"]: "target_item",
            name2geom["distractor_item_geom"]: "distractor_item",
        }
        target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_item")
        distractor_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "distractor_item"
        )
        ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        joint_addr = {
            name: int(
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in ("ee_x", "ee_y", "ee_z", "ee_yaw", "finger_left_slide")
        }
        target_dofadr = int(
            model.jnt_dofadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_item_free")
            ]
        )

        # ctrl targets start at current joint positions; fingers open.
        ctrl = {
            "x": float(data.qpos[joint_addr["ee_x"]]),
            "y": float(data.qpos[joint_addr["ee_y"]]),
            "z": float(data.qpos[joint_addr["ee_z"]]),
            "yaw": float(data.qpos[joint_addr["ee_yaw"]]),
            "finger": 0.0,
        }

        def _apply_ctrl() -> None:
            data.ctrl[0] = _clamp(ctrl["x"], -0.45, 0.45)
            data.ctrl[1] = _clamp(ctrl["y"], -0.45, 0.45)
            data.ctrl[2] = _clamp(ctrl["z"], -0.30, 0.15)
            data.ctrl[3] = _clamp(ctrl["yaw"], -3.1416, 3.1416)
            data.ctrl[4] = _clamp(ctrl["finger"], 0.0, FINGER_SLIDE_RANGE)
            data.ctrl[5] = _clamp(ctrl["finger"], 0.0, FINGER_SLIDE_RANGE)

        def _contact_summary() -> dict[str, Any]:
            finger_contact_items: set[str] = set()
            contact_count = int(data.ncon)
            for index in range(contact_count):
                contact = data.contact[index]
                pair = {int(contact.geom1), int(contact.geom2)}
                touched = pair & set(item_geoms)
                if pair & finger_geoms and touched:
                    for gid in touched:
                        finger_contact_items.add(item_geoms[gid])
            return {
                "contact_count": contact_count,
                "finger_contact_items": sorted(finger_contact_items),
            }

        def _observation(step_index: int) -> dict[str, Any]:
            ee_pos = [float(v) for v in data.site_xpos[ee_site]]
            opening = 2.0 * (
                FINGER_INNER_OPEN_HALF_GAP
                - float(data.qpos[joint_addr["finger_left_slide"]])
            )
            target_pos = [float(v) for v in data.xpos[target_body]]
            distractor_pos = [float(v) for v in data.xpos[distractor_body]]
            contact = _contact_summary()
            grasped = (
                "target_item" in contact["finger_contact_items"]
                and target_pos[2] > OBJECT_REST_Z + 0.01
            )
            return {
                "schema_version": "blueprint_real_policy_closed_loop_observation.v1",
                "task_id": _string(run.get("task_id")),
                "task_statement": _string(
                    run.get("task_statement") or run.get("statement")
                )
                or "Pick up the commanded target item and place it in the goal zone.",
                "scenario_eval_run_id": run_id,
                "step_index": step_index,
                "sim_time_s": round(float(data.time), 6),
                "control_dt_s": round(control_dt, 6),
                "end_effector": {
                    "position_xyz": [round(v, 6) for v in ee_pos],
                    "yaw_rad": round(float(data.qpos[joint_addr["ee_yaw"]]), 6),
                    "gripper_opening_m": round(opening, 6),
                },
                "objects": [
                    {
                        "object_id": "target_item",
                        "role": "commanded_target",
                        "position_xyz": [round(v, 6) for v in target_pos],
                        "grasped": bool(grasped),
                    },
                    {
                        "object_id": "distractor_item",
                        "role": "distractor",
                        "position_xyz": [round(v, 6) for v in distractor_pos],
                        "grasped": False,
                    },
                ],
                "goal_zone": {
                    "zone_id": "goal_zone_site",
                    "center_xyz": [
                        layout["goal_center_xy"][0],
                        layout["goal_center_xy"][1],
                        OBJECT_REST_Z,
                    ],
                    "radius_m": layout["goal_radius_m"],
                },
                "visual_observation": {
                    "camera_frame_path": latest_frame["path"],
                    "camera_name": "obs_cam" if latest_frame["path"] else None,
                    "frame_source": "mujoco_rendered_scene_camera"
                    if latest_frame["path"]
                    else None,
                },
                "proprio_state_source": "measured_simulator_state",
            }

        attempt_id = f"real_policy_attempt_{run_index + 1:04d}"
        control_rows: list[dict[str, Any]] = []
        contact_trace: list[dict[str, Any]] = []
        first_sustained_contact_item: str | None = None
        contact_streak: dict[str, int] = {}
        lift_streak = 0
        lifted_sustained = False
        step_index = 0
        query_index = 0
        attempt_physics_steps = 0
        policy_error: str | None = None

        while step_index < max_control_steps:
            if episode_renderer is not None:
                from PIL import Image

                episode_renderer.update_scene(data, camera="obs_cam")
                frame_path = frames_dir / f"{attempt_id}_q{query_index:04d}.png"
                Image.fromarray(episode_renderer.render()).save(frame_path)
                latest_frame["path"] = str(frame_path)
                obs_frames_rendered += 1
            observation = _observation(step_index)
            try:
                chunk = transport.query(observation)
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                policy_error = f"{exc.__class__.__name__}: {exc}"
                break
            requery_rows.append(
                {
                    "attempt_id": attempt_id,
                    "scenario_eval_run_id": run_id,
                    "query_index": query_index,
                    "step_index": step_index,
                    "chunk_size": len(chunk),
                    "executed_horizon": min(executed_horizon, len(chunk)),
                    "transport": transport.transport,
                    "policy_id": transport.policy_id,
                    "sim_time_s": observation["sim_time_s"],
                }
            )
            query_index += 1
            for action in chunk[:executed_horizon]:
                if step_index >= max_control_steps:
                    break
                vector = _action_7d(action)
                ctrl["x"] += vector[0]
                ctrl["y"] += vector[1]
                ctrl["z"] += vector[2]
                ctrl["yaw"] += vector[5]
                grip = _clamp(vector[6], 0.0, 1.0)
                ctrl["finger"] = (1.0 - grip) * FINGER_SLIDE_RANGE
                _apply_ctrl()
                for _ in range(physics_substeps):
                    mujoco.mj_step(model, data)
                    attempt_physics_steps += 1
                contact = _contact_summary()
                for item in contact["finger_contact_items"]:
                    contact_streak[item] = contact_streak.get(item, 0) + 1
                    if (
                        contact_streak[item] >= GRASP_SUSTAIN_STEPS
                        and first_sustained_contact_item is None
                    ):
                        first_sustained_contact_item = item
                for item in list(contact_streak):
                    if item not in contact["finger_contact_items"]:
                        contact_streak[item] = 0
                target_z = float(data.xpos[target_body][2])
                if (
                    target_z - OBJECT_REST_Z >= LIFT_HEIGHT_THRESHOLD_M
                    and "target_item" in contact["finger_contact_items"]
                ):
                    lift_streak += 1
                    if lift_streak >= LIFT_SUSTAIN_STEPS:
                        lifted_sustained = True
                else:
                    lift_streak = 0
                control_rows.append(
                    {
                        "step_index": step_index,
                        "sim_time_s": round(float(data.time), 6),
                        "action_7d": [round(v, 6) for v in vector],
                        "ee_xyz": [
                            round(float(v), 6) for v in data.site_xpos[ee_site]
                        ],
                        "target_item_xyz": [
                            round(float(v), 6) for v in data.xpos[target_body]
                        ],
                        "contact_count": contact["contact_count"],
                        "finger_contact_items": contact["finger_contact_items"],
                    }
                )
                if contact["finger_contact_items"]:
                    contact_trace.append(
                        {
                            "step_index": step_index,
                            "sim_time_s": round(float(data.time), 6),
                            "finger_contact_items": contact["finger_contact_items"],
                        }
                    )
                step_index += 1

            # Early stop once placed: object in goal zone, settled, released.
            placed_now, _ = _placing_measurement(
                data=data,
                target_body=target_body,
                target_dofadr=target_dofadr,
                layout=layout,
                contact=_contact_summary(),
            )
            if placed_now:
                break

        final_contact = _contact_summary()
        placed, placing_detail = _placing_measurement(
            data=data,
            target_body=target_body,
            target_dofadr=target_dofadr,
            layout=layout,
            contact=final_contact,
        )
        language_following = first_sustained_contact_item == "target_item"
        criteria = {
            "language_following": {
                "passed": bool(language_following),
                "measured_from": "first_sustained_finger_contact_body",
                "first_sustained_contact_item": first_sustained_contact_item,
            },
            "object_lifting": {
                "passed": bool(lifted_sustained),
                "measured_from": "target_item_height_above_rest_with_grasp_contact",
                "lift_threshold_m": LIFT_HEIGHT_THRESHOLD_M,
                "sustain_steps": LIFT_SUSTAIN_STEPS,
            },
            "object_placing": {
                "passed": bool(placed),
                "measured_from": "final_object_pose_velocity_and_release_state",
                **placing_detail,
            },
        }
        task_success = all(criteria[name]["passed"] for name in SC3_CRITERIA)
        timeout = step_index >= max_control_steps and not task_success
        failure_mode_ids = [
            f"{name}_failure" for name in SC3_CRITERIA if not criteria[name]["passed"]
        ]
        if timeout:
            failure_mode_ids.append("rollout_timeout")
        if policy_error:
            failure_mode_ids.append("policy_query_failed")

        trace_path = output_root / f"{attempt_id}_control_stream.jsonl"
        with trace_path.open("w", encoding="utf-8") as handle:
            for row in control_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        final_ee = [round(float(v), 6) for v in data.site_xpos[ee_site]]
        final_target = [round(float(v), 6) for v in data.xpos[target_body]]
        goal_center = [
            layout["goal_center_xy"][0],
            layout["goal_center_xy"][1],
            OBJECT_REST_Z,
        ]
        final_target_error_m = math.dist(final_target, goal_center)
        attempt = {
            "attempt_id": attempt_id,
            "episode_id": f"real_policy_episode_{run_index + 1:04d}",
            "scenario_eval_run_id": run_id,
            "scenario_variation_instance_id": _string(
                run.get("scenario_variation_instance_id")
            )
            or None,
            "variation_name": layout["variation_name"],
            "task_id": _string(run.get("task_id")),
            "scenario_id": _string(run.get("scenario_id")),
            "policy_id": transport.policy_id,
            "status": "completed" if policy_error is None else "failed",
            "success": bool(task_success),
            "task_success": bool(task_success),
            "failure_mode_ids": failure_mode_ids,
            "failure_reason": policy_error,
            "deterministic_seed": seed,
            "spawn_pose": list(layout["target_start_xy"]) + [OBJECT_REST_Z],
            "target_pose": goal_center,
            "final_pose": final_target,
            "pose_frame": "tabletop_local",
            "metrics": {
                "control_step_count": step_index,
                "physics_step_count": attempt_physics_steps,
                "policy_query_count": query_index,
                "cycle_time_seconds": round(float(data.time), 6),
                "final_target_error_m": round(final_target_error_m, 6),
                "contact_event_count": len(contact_trace),
                "intervention_count": 0,
                "safety_event_count": 0,
                "deterministic_seed": seed,
            },
            "task_outcome": {
                "task_success": bool(task_success),
                "success_criteria": criteria,
                "goal_reached": bool(placed),
                "endpoint_clean": bool(placed),
                "spawn_clean": True,
                "timeout": bool(timeout),
                "fall_detected": False,
                "stuck_detected": bool(
                    policy_error is None and step_index >= max_control_steps and not lifted_sustained
                ),
                "policy_instability_detected": bool(policy_error is not None),
                "final_target_error_m": round(final_target_error_m, 6),
                "goal_tolerance_m": layout["goal_radius_m"],
                "robot_scene_contact_event_count": len(contact_trace),
                "near_miss_event_count": 0,
                "success_criteria_source": "measured_simulator_state",
            },
            "actions": [
                {
                    "step_index": row["step_index"],
                    "action_7d": row["action_7d"],
                }
                for row in control_rows[:200]
            ],
            "contact_trace": contact_trace[:100],
            "safety_events": [],
            "variation_physically_modeled": layout["variation_physically_modeled"],
            "variation_note": layout["variation_note"],
            "artifact_paths": {
                "policy_trace": str(trace_path),
                "control_stream": str(trace_path),
            },
            "final_ee_xyz": final_ee,
        }
        attempts.append(attempt)
        if episode_renderer is not None:
            episode_renderer.close()

    transport.close()
    if obs_frames_rendered:
        _encode_attempt_videos_and_coverage(
            attempts=attempts,
            frames_dir=frames_dir,
            output_root=output_root,
            job_dir=output_file.parent,
            control_hz=control_hz,
            generated_at=generated_at,
        )
    requery_path = output_root / "policy_requery_trace.jsonl"
    with requery_path.open("w", encoding="utf-8") as handle:
        for row in requery_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    executed = sum(
        int(_mapping(a.get("metrics")).get("physics_step_count") or 0) for a in attempts
    )
    payload = {
        "schema_version": SIMULATOR_OUTPUT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed",
        "substrate": SUBSTRATE,
        "capture_root": str(capture_root) if capture_root else None,
        "policy_in_the_loop": True,
        "policy_transport": transport.transport,
        "policy_id": transport.policy_id,
        "policy_family_id": loaded.family_id if loaded else None,
        "policy_type": loaded.policy_type if loaded else None,
        "checkpoint_sha256": loaded.checkpoint_sha256 if loaded else None,
        "checkpoint_dir": loaded.checkpoint_dir if loaded else None,
        "simulator_execution_proven": executed > 0,
        "physics_step_count": executed,
        "obs_frames_rendered": obs_frames_rendered,
        "obs_frame_source": "mujoco_rendered_scene_camera" if obs_frames_rendered else None,
        "control_hz": control_hz,
        "chunk_size": chunk_size,
        "executed_horizon": executed_horizon,
        "max_rollout_seconds": max_seconds,
        "embodiment_contract": {
            "embodiment_id": EMBODIMENT_ID,
            "actuated_action_dims": ["dx", "dy", "dz", "dyaw", "gripper"],
            "recorded_unactuated_action_dims": ["droll", "dpitch"],
            "not_a_humanoid_or_unitree_g1_claim": True,
        },
        "sc3_alignment": {
            "action_representation": "7d_delta_end_effector_pose",
            "receding_horizon": {
                "policy_action_chunk_count": chunk_size,
                "executed_receding_horizon_action_count": executed_horizon,
            },
            "success_criteria": list(SC3_CRITERIA),
            "success_criteria_source": "measured_simulator_state_not_generated_video",
        },
        "policy_requery_trace_path": str(requery_path),
        "attempts": attempts,
        "claim_boundary": {
            "classical_sim_rollout_is_not_physical_robot_proof": True,
            "scripted_baseline_is_not_a_learned_policy": bool(
                loaded and loaded.policy_type == "blueprint_scripted_pick_place"
            ),
            "task_success_labels_are_simulator_measured_only": True,
            "deployment_approval_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    write_json(output_file, payload)
    return payload


def _encode_attempt_videos_and_coverage(
    *,
    attempts: list[dict[str, Any]],
    frames_dir: Path,
    output_root: Path,
    job_dir: Path,
    control_hz: float,
    generated_at: str,
) -> None:
    """Encode per-attempt episode videos from the real rendered observation
    frames and write the visual-media coverage manifest the buyer-report
    media_validity layer reads. Fail-closed: no ffmpeg/ffprobe or an
    undecodable file marks coverage false — never claims media it can't prove.
    """
    import shutil
    import subprocess as sp

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    if not ffmpeg:
        blockers.append("ffmpeg_missing_video_encoding_skipped")
    total_frames = 0
    for attempt in attempts:
        attempt_id = _string(attempt.get("attempt_id"))
        frame_paths = sorted(frames_dir.glob(f"{attempt_id}_q*.png"))
        total_frames += len(frame_paths)
        video_path = output_root / f"{attempt_id}_third_person.mp4"
        encoded = False
        decodable = False
        if ffmpeg and len(frame_paths) >= 2:
            list_file = output_root / f"{attempt_id}_frames.txt"
            list_file.write_text(
                "".join(
                    f"file '{p}'\nduration {1.0 / max(1.0, control_hz / 16.0):.4f}\n"
                    for p in frame_paths
                ),
                encoding="utf-8",
            )
            result = sp.run(
                [
                    ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt", "yuv420p", str(video_path),
                ],
                capture_output=True, text=True, timeout=300, check=False,
            )
            encoded = result.returncode == 0 and video_path.is_file()
            if encoded and ffprobe:
                probe = sp.run(
                    [
                        ffprobe, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name,nb_read_frames",
                        "-count_frames", "-of", "json", str(video_path),
                    ],
                    capture_output=True, text=True, timeout=120, check=False,
                )
                try:
                    streams = _mapping(json.loads(probe.stdout)).get("streams") or []
                    decodable = bool(
                        streams and int(_mapping(streams[0]).get("nb_read_frames") or 0) >= 2
                    )
                except (json.JSONDecodeError, ValueError):
                    decodable = False
        if encoded:
            attempt["video_path"] = str(video_path)
            attempt.setdefault("artifact_paths", {})["third_person_video"] = str(video_path)
        rows.append(
            {
                "attempt_id": attempt_id,
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "frame_count": len(frame_paths),
                "third_person_video": str(video_path) if encoded else None,
                "video_decodable": decodable,
                "camera": "obs_cam_third_person_scene_camera",
            }
        )
    all_recorded = bool(rows) and all(row["third_person_video"] for row in rows)
    all_decodable = bool(rows) and all(row["video_decodable"] for row in rows)
    if rows and not all_recorded:
        blockers.append("some_attempts_missing_episode_video")
    if all_recorded and not all_decodable:
        blockers.append("some_episode_videos_not_decodable")
    write_json(
        job_dir / "simulator_command_batch_visual_media_coverage.json",
        {
            "schema_version": "real_policy_closed_loop_visual_media_coverage.v1",
            "generated_at": generated_at,
            "all_required_runs_have_visual_recording": all_recorded,
            "all_required_runs_have_third_person_video": all_recorded,
            "all_required_runs_have_robot_pov_video": False,
            "all_required_videos_decodable": all_recorded and all_decodable,
            "frame_count": total_frames,
            "runs": rows,
            "blockers": blockers,
            "claim_boundary": (
                "Episode videos are real renders of the classical-sim rollout for "
                "review evidence; they are not physical-robot or generated-world proof."
            ),
        },
    )


def _placing_measurement(
    *,
    data: Any,
    target_body: int,
    target_dofadr: int,
    layout: Mapping[str, Any],
    contact: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    target_pos = [float(v) for v in data.xpos[target_body]]
    goal_xy = layout["goal_center_xy"]
    xy_error = math.hypot(target_pos[0] - goal_xy[0], target_pos[1] - goal_xy[1])
    speed = math.sqrt(
        sum(float(data.qvel[target_dofadr + axis]) ** 2 for axis in range(3))
    )
    in_zone = xy_error <= float(layout["goal_radius_m"])
    at_rest_height = abs(target_pos[2] - OBJECT_REST_Z) <= PLACE_Z_TOLERANCE_M
    released = "target_item" not in (contact.get("finger_contact_items") or [])
    settled = speed <= PLACE_SETTLE_SPEED_MPS
    placed = in_zone and at_rest_height and released and settled
    return placed, {
        "goal_xy_error_m": round(xy_error, 6),
        "in_goal_zone": in_zone,
        "at_rest_height": at_rest_height,
        "released": released,
        "settled": settled,
        "object_speed_mps": round(speed, 6),
    }


def _blocked_output(
    *,
    generated_at: str,
    blockers: Sequence[str],
    loaded: LoadedPolicyFamily | None,
) -> dict[str, Any]:
    return {
        "schema_version": SIMULATOR_OUTPUT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "substrate": SUBSTRATE,
        "policy_in_the_loop": False,
        "policy_family_id": loaded.family_id if loaded else None,
        "simulator_execution_proven": False,
        "attempts": [],
        "blockers": sorted(set(_string(b) for b in blockers if _string(b))),
        "claim_boundary": {
            "classical_sim_rollout_is_not_physical_robot_proof": True,
            "task_success_labels_are_simulator_measured_only": True,
            "public_claim_upgrade_allowed": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="LeRobot checkpoint dir")
    parser.add_argument(
        "--adapter-command",
        default=None,
        help="Policy adapter subprocess command (GPU/learned policy swap path)",
    )
    parser.add_argument(
        "--adapter-mode",
        choices=("subprocess", "persistent"),
        default="subprocess",
        help="persistent keeps one adapter process alive (torch models load once)",
    )
    parser.add_argument("--render-obs-frames", action="store_true")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--scenario-eval-matrix", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--executed-horizon", type=int, default=DEFAULT_EXECUTED_HORIZON
    )
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=20260705)
    args = parser.parse_args(argv)

    matrix_path = args.scenario_eval_matrix or os.getenv(
        "BLUEPRINT_SCENARIO_EVAL_MATRIX", ""
    )
    output_path = args.output or os.getenv("BLUEPRINT_SIMULATOR_OUTPUT", "")
    if not matrix_path or not output_path:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": ["scenario_eval_matrix_or_output_path_missing"],
                }
            )
        )
        return 1
    payload = run_real_policy_closed_loop_rollout(
        checkpoint_dir=args.checkpoint,
        adapter_command=args.adapter_command,
        adapter_mode=args.adapter_mode,
        render_obs_frames=args.render_obs_frames,
        render_width=args.render_width,
        render_height=args.render_height,
        scenario_eval_matrix_path=matrix_path,
        output_path=output_path,
        capture_root=os.getenv("BLUEPRINT_CAPTURE_ROOT") or None,
        control_hz=args.control_hz,
        chunk_size=args.chunk_size,
        executed_horizon=args.executed_horizon,
        max_seconds=args.max_seconds,
        base_seed=args.base_seed,
        max_attempts=args.max_attempts,
    )
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "attempt_count": len(payload.get("attempts") or []),
                "simulator_execution_proven": payload.get(
                    "simulator_execution_proven"
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if payload.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
