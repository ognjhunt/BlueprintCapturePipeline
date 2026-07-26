"""Local articulated Franka can-to-tray feasibility oracle.

This lane is deliberately a scripted control, not a candidate policy and not a
physical answer key.  It verifies that the pinned Menagerie Franka, rigid can,
tray predicate, and live hand-mounted camera transform can execute together in
MuJoCo before expensive policy/WAM work is admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .common import write_json
from .scene_placement.stance_cameras import link_mounted_camera_spec


SCHEMA_VERSION = "franka_can_tray_feasibility.v1"
_ARM_SEED = (0.2897, 0.50732, -0.140016, -2.176, -0.0310497, 2.51592, -0.49251)
_CAN_INITIAL = (0.5, 0.075, 0.09)
_TRAY_CENTER = (0.45, 0.32, 0.03)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_xml() -> str:
    return """<mujoco model="franka can tray feasibility">
  <include file="mjx_scene.xml"/>
  <worldbody>
    <body name="spraycan" pos="0.5 0.075 0.09">
      <freejoint/>
      <geom name="spraycan_geom" type="cylinder" size="0.03 0.09" mass="0.25"
        friction="1 .01 .001" rgba=".9 .1 .05 1"/>
    </body>
    <geom name="tray_base" type="box" pos="0.45 0.32 0.015" size="0.18 0.14 0.015"
      rgba=".05 .1 .9 1"/>
  </worldbody>
</mujoco>
"""


def _stage_model(menagerie_root: Path, output_dir: Path) -> Path:
    required = ("mjx_scene.xml", "mjx_panda.xml", "assets")
    missing = [name for name in required if not (menagerie_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"menagerie_missing:{','.join(missing)}")
    stage = output_dir / "runtime_model"
    stage.mkdir(parents=True, exist_ok=True)
    for child in menagerie_root.iterdir():
        destination = stage / child.name
        if destination.exists() or destination.is_symlink():
            continue
        os.symlink(child, destination, target_is_directory=child.is_dir())
    scene = stage / "blueprint_can_tray.xml"
    scene.write_text(_scene_xml(), encoding="utf-8")
    return scene


def _rotation_error(target: Any, current: Any, np: Any) -> Any:
    residual = target @ current.T
    return 0.5 * np.asarray(
        (
            residual[2, 1] - residual[1, 2],
            residual[0, 2] - residual[2, 0],
            residual[1, 0] - residual[0, 1],
        )
    )


def _solve_ik(model: Any, data: Any, mujoco: Any, np: Any, site_id: int, q: Any, target: Any, target_rotation: Any) -> Any:
    low = model.jnt_range[:7, 0]
    high = model.jnt_range[:7, 1]
    for _ in range(300):
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        position_error = np.asarray(target) - data.site_xpos[site_id]
        rotation_error = _rotation_error(
            target_rotation, data.site_xmat[site_id].reshape(3, 3), np
        )
        if np.linalg.norm(position_error) < 1e-4 and np.linalg.norm(rotation_error) < 1e-3:
            break
        jac_position = np.zeros((3, model.nv))
        jac_rotation = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac_position, jac_rotation, site_id)
        jacobian = np.vstack((jac_position[:, :7], 0.25 * jac_rotation[:, :7]))
        error = np.concatenate((position_error, 0.25 * rotation_error))
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1e-4 * np.eye(6), error
        )
        q = np.clip(q + np.clip(delta, -0.08, 0.08), low + 0.01, high - 0.01)
    return q.copy()


def run_franka_can_tray_feasibility(
    *, menagerie_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    import mujoco
    import numpy as np

    root = Path(menagerie_root).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene_path = _stage_model(root, out)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    can_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spraycan")
    if min(site_id, hand_id, can_id) < 0:
        raise RuntimeError("required_franka_or_can_body_missing")

    seed = np.asarray(_ARM_SEED, dtype=float)
    data.qpos[:7] = seed
    data.qpos[7:9] = 0.04
    data.qpos[9:16] = (*_CAN_INITIAL, 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    target_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    waypoint_specs = (
        ("pregrasp", (0.5, 0.075, 0.25), 0.04, 1.0),
        ("approach", (0.5, 0.075, 0.12), 0.04, 1.5),
        ("grasp", (0.5, 0.075, 0.12), 0.0, 1.5),
        ("lift", (0.5, 0.075, 0.25), 0.0, 1.5),
        ("transport", (0.45, 0.32, 0.30), 0.0, 2.0),
        ("release", (0.45, 0.32, 0.30), 0.04, 1.2),
        ("retreat", (0.45, 0.32, 0.40), 0.04, 1.5),
    )
    q = seed
    joint_targets: dict[str, Any] = {}
    for phase, target, _gripper, _seconds in waypoint_specs:
        if phase not in joint_targets:
            q = _solve_ik(model, data, mujoco, np, site_id, q, target, target_rotation)
            joint_targets[phase] = q.copy()

    mujoco.mj_resetData(model, data)
    data.qpos[:7] = joint_targets["pregrasp"]
    data.qpos[7:9] = 0.04
    data.qpos[9:16] = (*_CAN_INITIAL, 1.0, 0.0, 0.0, 0.0)
    data.ctrl[:7] = joint_targets["pregrasp"]
    data.ctrl[7] = 0.04
    mujoco.mj_forward(model, data)
    initial_can_z = float(data.xpos[can_id, 2])
    max_can_z = initial_can_z
    trace: list[dict[str, Any]] = []

    def snapshot(phase: str) -> None:
        camera = link_mounted_camera_spec(
            parent_translation=data.xpos[hand_id],
            parent_rotation_row_major=data.xmat[hand_id],
            mount_translation=(0.0, 0.10, 0.03),
            mount_forward=(0.0, 0.0, 1.0),
            mount_up=(0.0, 1.0, 0.0),
            look_distance_m=0.5,
            fov_deg=82.0,
        )
        trace.append(
            {
                "phase": phase,
                "simulation_time_s": float(data.time),
                "gripper_site_m": [float(value) for value in data.site_xpos[site_id]],
                "spraycan_center_m": [float(value) for value in data.xpos[can_id]],
                "hand_world_rotation_row_major": [float(value) for value in data.xmat[hand_id]],
                "wrist_camera": camera,
            }
        )

    def execute(phase: str, seconds: float, gripper: float) -> None:
        nonlocal max_can_z
        start = data.ctrl[:7].copy()
        target = joint_targets[phase]
        steps = int(seconds / model.opt.timestep)
        for index in range(steps):
            ratio = (index + 1) / steps
            blend = ratio * ratio * (3.0 - 2.0 * ratio)
            data.ctrl[:7] = (1.0 - blend) * start + blend * target
            data.ctrl[7] = gripper
            mujoco.mj_step(model, data)
            max_can_z = max(max_can_z, float(data.xpos[can_id, 2]))
        snapshot(phase)

    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    snapshot("settled")
    for phase, _target, gripper, seconds in waypoint_specs[1:]:
        execute(phase, seconds, gripper)
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        max_can_z = max(max_can_z, float(data.xpos[can_id, 2]))
    snapshot("final_stable")

    final_position = data.xpos[can_id].copy()
    final_speed = float(np.linalg.norm(data.cvel[can_id, 3:]))
    contained = bool(
        abs(float(final_position[0]) - _TRAY_CENTER[0]) <= 0.15
        and abs(float(final_position[1]) - _TRAY_CENTER[1]) <= 0.11
        and 0.10 <= float(final_position[2]) <= 0.14
    )
    lift_delta = float(max_can_z - initial_can_z)
    unique_camera_positions = {
        tuple(round(value, 6) for value in row["wrist_camera"]["pos"]) for row in trace
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "role": "scripted_scene_feasibility_oracle_not_candidate_policy",
        "simulator": {"name": "MuJoCo", "version": mujoco.__version__, "timestep_s": float(model.opt.timestep)},
        "source": {
            "menagerie_root": str(root),
            "mjx_panda_sha256": _sha256(root / "mjx_panda.xml"),
            "generated_scene_sha256": _sha256(scene_path),
        },
        "metrics": {
            "initial_spraycan_z_m": initial_can_z,
            "max_spraycan_z_m": max_can_z,
            "lift_delta_m": lift_delta,
            "final_spraycan_center_m": [float(value) for value in final_position],
            "final_linear_speed_m_s": final_speed,
            "contained_in_tray_interior": contained,
        },
        "gates": {
            "lift_at_least_0_05m": lift_delta >= 0.05,
            "final_containment": contained,
            "final_stability_below_0_02m_s": final_speed < 0.02,
            "live_wrist_camera_updated_across_phases": len(unique_camera_positions) >= 4,
        },
        "trace": trace,
        "claim_boundary": {
            "learned_policy_executed": False,
            "wam_executed": False,
            "nvidia_warehouse_executed": False,
            "captured_3dgs_composited_in_mujoco": False,
            "physical_success_proven": False,
        },
    }
    result["oracle_pass"] = all(result["gates"].values())
    result["manifest_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(out / "franka_can_tray_feasibility.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menagerie-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-output")
    args = parser.parse_args(argv)
    result = run_franka_can_tray_feasibility(
        menagerie_root=args.menagerie_root, output_dir=args.output_dir
    )
    if args.manifest_output:
        write_json(Path(args.manifest_output), result)
    return 0 if result["oracle_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
