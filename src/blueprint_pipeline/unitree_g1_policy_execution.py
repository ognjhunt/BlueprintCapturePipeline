from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from .common import utc_now_iso, write_json

UNITREE_G1_POLICY_EXECUTION_SCHEMA_VERSION = "official_unitree_g1_policy_execution.v1"
DEFAULT_OUTPUT_SUBDIR = (
    "pipeline/g1_controlled_proof_setup/official_unitree_g1_policy_execution"
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML payload is not an object: {path}")
    return payload


def _gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def _pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def _repo_head(repo_root: Path) -> str | None:
    head = repo_root / ".git" / "HEAD"
    if not head.is_file():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = repo_root / ".git" / text.split(" ", 1)[1]
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else text
    return text


def build_unitree_g1_policy_execution(
    *,
    capture_root: str | Path,
    unitree_rl_gym_root: str | Path,
    job_id: str,
    duration_seconds: float = 4.0,
    max_steps: int | None = None,
    output_dir: str | Path | None = None,
    command_xyz: Sequence[float] | None = None,
) -> dict[str, Any]:
    # Local macOS Python environments often load duplicate OpenMP runtimes via
    # torch/mujoco. Set before importing torch so the execution can be audited.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import mujoco
    import torch

    root = Path(capture_root).expanduser().resolve()
    repo_root = Path(unitree_rl_gym_root).expanduser().resolve()
    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root / DEFAULT_OUTPUT_SUBDIR
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = repo_root / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"
    config = _read_yaml(config_path)
    policy_path = Path(
        str(config["policy_path"]).replace("{LEGGED_GYM_ROOT_DIR}", str(repo_root))
    )
    xml_path = Path(str(config["xml_path"]).replace("{LEGGED_GYM_ROOT_DIR}", str(repo_root)))
    policy = torch.jit.load(str(policy_path), map_location="cpu")

    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    requested_steps = int(round(duration_seconds / simulation_dt))
    total_steps = max(1, min(requested_steps, max_steps or requested_steps))

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    if command_xyz is not None:
        if len(command_xyz) != 3:
            raise ValueError("command_xyz must contain [vx_mps, vy_mps, yaw_rate_rad_s]")
        cmd = np.array([float(value) for value in command_xyz], dtype=np.float32)
        command_source = "caller_override"
    else:
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        command_source = "unitree_rl_gym_config_cmd_init"
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions = int(config["num_actions"])
    num_obs = int(config["num_obs"])
    action_scale = float(config["action_scale"])
    dof_pos_scale = float(config["dof_pos_scale"])
    dof_vel_scale = float(config["dof_vel_scale"])
    ang_vel_scale = float(config["ang_vel_scale"])

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    trace_path = out_dir / "policy_execution_trace.jsonl"
    update_rows: list[dict[str, Any]] = []

    with trace_path.open("w", encoding="utf-8") as trace_handle:
        for step in range(total_steps):
            tau = _pd_control(
                target_dof_pos,
                data.qpos[7:],
                kps,
                np.zeros_like(kds),
                data.qvel[6:],
                kds,
            )
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            if step % control_decimation == 0:
                qj = data.qpos[7:]
                dqj = data.qvel[6:]
                quat = data.qpos[3:7]
                omega = data.qvel[3:6]

                qj_scaled = (qj - default_angles) * dof_pos_scale
                dqj_scaled = dqj * dof_vel_scale
                gravity_orientation = _gravity_orientation(quat)
                omega_scaled = omega * ang_vel_scale

                period = 0.8
                sim_time = step * simulation_dt
                phase = sim_time % period / period
                obs[:3] = omega_scaled
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj_scaled
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj_scaled
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array(
                    [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
                    dtype=np.float32,
                )

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze().astype(np.float32)
                target_dof_pos = action * action_scale + default_angles
                row = {
                    "step": step,
                    "sim_time_s": sim_time,
                    "base_position_xyz": data.qpos[:3].astype(float).tolist(),
                    "command_xyz": cmd.astype(float).tolist(),
                    "action": action.astype(float).tolist(),
                    "target_dof_pos": target_dof_pos.astype(float).tolist(),
                }
                update_rows.append(row)
                trace_handle.write(json.dumps(row, sort_keys=True) + "\n")

    actions = np.array([row["action"] for row in update_rows], dtype=np.float32)
    finite_state = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    finite_actions = bool(actions.size and np.isfinite(actions).all())
    metrics = {
        "status": "completed" if finite_state and finite_actions else "blocked",
        "duration_seconds_requested": duration_seconds,
        "sim_time_s": total_steps * simulation_dt,
        "simulation_dt": simulation_dt,
        "steps": total_steps,
        "control_updates": len(update_rows),
        "command_xyz": cmd.astype(float).tolist(),
        "command_source": command_source,
        "final_base_position_xyz": data.qpos[:3].astype(float).tolist(),
        "final_base_yaw_quat_wxyz": data.qpos[3:7].astype(float).tolist(),
        "finite_state": finite_state,
        "finite_actions": finite_actions,
        "mean_abs_action": float(np.mean(np.abs(actions))) if actions.size else None,
        "max_abs_action": float(np.max(np.abs(actions))) if actions.size else None,
    }
    metrics_path = out_dir / "policy_metrics.json"
    write_json(metrics_path, metrics)

    manifest = {
        "schema_version": UNITREE_G1_POLICY_EXECUTION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": metrics["status"],
        "job_id": job_id,
        "policy_id": "unitree_rl_gym_g1_pretrain_motion",
        "robot_profile_id": "unitree_g1_humanoid",
        "robot_make_model": "Unitree G1",
        "source_repository": {
            "name": "unitree_rl_gym",
            "url": "https://github.com/unitreerobotics/unitree_rl_gym",
            "pinned_commit": _repo_head(repo_root),
            "local_inspection_root": str(repo_root),
        },
        "official_artifacts": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "policy_path": str(policy_path),
            "policy_sha256": _sha256(policy_path),
            "xml_path": str(xml_path),
            "xml_sha256": _sha256(xml_path),
        },
        "execution": {
            "runner": "headless_mujoco_torchscript",
            "trace_path": str(trace_path),
            "metrics_path": str(metrics_path),
            "kmp_duplicate_lib_ok_set": os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE",
            "command_source": command_source,
        },
        "metrics": metrics,
        "proof_boundary": {
            "non_default_policy_execution_trace_proven": metrics["status"] == "completed",
            "policy_metrics_tied_to_scenario_variation": metrics["status"] == "completed",
            "robot_team_owner_acceptance_or_review_proven": False,
            "robot_team_policy_performance_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "real_robot_pov_evidence_proven": False,
            "non_ranking_operational_claim_validated": False,
            "public_claim_upgrade_allowed": False,
        },
        "blockers": []
        if metrics["status"] == "completed"
        else ["official_unitree_g1_policy_execution_failed"],
    }
    manifest_path = out_dir / "official_unitree_g1_policy_execution_manifest.json"
    write_json(manifest_path, manifest)
    manifest["output_path"] = str(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--unitree-rl-gym-root", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--duration-seconds", type=float, default=4.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--command-xyz",
        nargs=3,
        type=float,
        metavar=("VX_MPS", "VY_MPS", "YAW_RATE_RAD_S"),
    )
    args = parser.parse_args(argv)
    manifest = build_unitree_g1_policy_execution(
        capture_root=args.capture_root,
        unitree_rl_gym_root=args.unitree_rl_gym_root,
        job_id=args.job_id,
        duration_seconds=args.duration_seconds,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        command_xyz=args.command_xyz,
    )
    print(manifest["output_path"])
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
