"""Build an owned, truth-bounded UNITREE_G1_SONIC microwave training seed."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .g1_microwave_grasp_arc_seed import (
    EGOCENTRIC_CAMERA_NAME,
    SCHEMA_VERSION as GRASP_SCHEMA_VERSION,
    _proxy_model,
)
from .g1_microwave_handle_dynamics_validation import (
    SCHEMA_VERSION as DYNAMICS_SCHEMA_VERSION,
)
from .g1_microwave_reach_seed import (
    PINNED_G1_MODEL_SHA256,
    _finite_vector,
    _load_mapping,
    _rotation_wxyz,
    _sha256,
)
from .g1_sonic_motion_token_conversion import (
    FIXED_UPRIGHT_PROJECTED_GRAVITY,
    HAND_DIM,
    MOTION_TOKEN_DIM,
    SCHEMA_VERSION as SONIC_CONVERSION_SCHEMA_VERSION,
    SONIC_ACTION_DIM,
    SOURCE_ACTION_JOINT_NAMES,
    SOURCE_FPS,
    unitree_g1_sonic_training_modality,
)


SCHEMA_VERSION = "g1_microwave_owned_training_seed.v1"
TASK_DESCRIPTION = "Stand at the microwave and open the microwave door."


def aligned_door_angle_schedule(
    frame_count: int,
    prescribed_pull_angles_rad: Any,
) -> np.ndarray:
    """Align a prescribed pull arc to the end of a reach/closure trajectory."""

    count = int(frame_count)
    pull = np.asarray(prescribed_pull_angles_rad, dtype=np.float64)
    if count < 2 or pull.ndim != 1 or pull.shape[0] < 2:
        raise ValueError("g1_microwave_training_seed_door_schedule_shape_invalid")
    if pull.shape[0] > count or pull[0] != 0.0 or not np.all(np.diff(pull) <= 0.0):
        raise ValueError("g1_microwave_training_seed_door_schedule_values_invalid")
    schedule = np.zeros(count, dtype=np.float64)
    schedule[count - pull.shape[0] :] = pull
    return schedule


def split_sonic_training_actions(
    sonic_actions: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split exact 78D SONIC actions into the three registered action keys."""

    actions = np.asarray(sonic_actions, dtype=np.float32)
    if (
        actions.ndim != 2
        or actions.shape[0] < 2
        or actions.shape[1] != SONIC_ACTION_DIM
        or not np.isfinite(actions).all()
    ):
        raise ValueError("g1_microwave_training_seed_sonic_actions_invalid")
    return (
        actions[:, :MOTION_TOKEN_DIM],
        actions[:, MOTION_TOKEN_DIM : MOTION_TOKEN_DIM + HAND_DIM],
        actions[:, MOTION_TOKEN_DIM + HAND_DIM :],
    )


def build_owned_training_seed(
    *,
    model_path: str | Path,
    initial_policy_observation_path: str | Path,
    target_focus_report_path: str | Path,
    grasp_report_path: str | Path,
    dynamic_report_path: str | Path,
    trajectory_path: str | Path,
    sonic_actions_path: str | Path,
    sonic_conversion_report_path: str | Path,
    output_dir: str | Path,
    expected_model_sha256: str = PINNED_G1_MODEL_SHA256,
) -> dict[str, Any]:
    """Materialize one owned prescribed expert seed plus raw head-POV video."""

    try:
        import imageio.v3 as iio
        import mujoco  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("g1_microwave_training_seed_render_dependency_missing") from exc

    model_file = Path(model_path).expanduser().resolve()
    observation_file = Path(initial_policy_observation_path).expanduser().resolve()
    focus_file = Path(target_focus_report_path).expanduser().resolve()
    grasp_file = Path(grasp_report_path).expanduser().resolve()
    dynamic_file = Path(dynamic_report_path).expanduser().resolve()
    trajectory_file = Path(trajectory_path).expanduser().resolve()
    sonic_file = Path(sonic_actions_path).expanduser().resolve()
    conversion_file = Path(sonic_conversion_report_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not model_file.is_file() or _sha256(model_file) != str(
        expected_model_sha256
    ).lower():
        raise ValueError("g1_microwave_training_seed_model_sha256_mismatch")

    observation = _load_mapping(observation_file, name="initial_policy_observation")
    focus = _load_mapping(focus_file, name="target_focus_report")
    grasp = _load_mapping(grasp_file, name="grasp_report")
    dynamics = _load_mapping(dynamic_file, name="dynamic_report")
    conversion = _load_mapping(conversion_file, name="sonic_conversion_report")
    if grasp.get("schema_version") != GRASP_SCHEMA_VERSION:
        raise ValueError("g1_microwave_training_seed_grasp_schema_mismatch")
    if (
        dynamics.get("schema_version") != DYNAMICS_SCHEMA_VERSION
        or dynamics.get("status")
        != "qualified_contact_driven_handle_only_requested_transition"
        or not bool(
            dict(dynamics.get("qualification") or {}).get(
                "requested_opening_within_tolerance_proven"
            )
        )
    ):
        raise ValueError("g1_microwave_training_seed_dynamics_not_qualified")
    if conversion.get("schema_version") != SONIC_CONVERSION_SCHEMA_VERSION:
        raise ValueError("g1_microwave_training_seed_conversion_schema_mismatch")

    trajectory = np.load(trajectory_file, allow_pickle=False).astype(np.float32)
    sonic_actions = np.load(sonic_file, allow_pickle=False).astype(np.float32)
    if trajectory.shape != (sonic_actions.shape[0], len(SOURCE_ACTION_JOINT_NAMES)):
        raise ValueError("g1_microwave_training_seed_trajectory_shape_invalid")
    grasp_trajectory = grasp.get("trajectory")
    conversion_artifact = conversion.get("artifact")
    if (
        not isinstance(grasp_trajectory, Mapping)
        or grasp_trajectory.get("file_sha256") != _sha256(trajectory_file)
        or not isinstance(conversion_artifact, Mapping)
        or conversion_artifact.get("file_sha256") != _sha256(sonic_file)
    ):
        raise ValueError("g1_microwave_training_seed_artifact_binding_failed")
    motion_tokens, left_hand, right_hand = split_sonic_training_actions(
        sonic_actions
    )

    pose_solver = grasp.get("pose_solver")
    pose_frames = pose_solver.get("frames") if isinstance(pose_solver, Mapping) else None
    if not isinstance(pose_frames, list):
        raise ValueError("g1_microwave_training_seed_pose_frames_missing")
    try:
        pull_angles = np.asarray(
            [float(row["door_angle_rad"]) for row in pose_frames],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("g1_microwave_training_seed_pose_angles_invalid") from exc
    door_angles = aligned_door_angle_schedule(trajectory.shape[0], pull_angles)

    context = observation.get("camera_projection_context")
    pelvis = context.get("live_isaac_pelvis_world_pose") if isinstance(context, Mapping) else None
    if not isinstance(pelvis, Mapping):
        raise ValueError("g1_microwave_training_seed_pelvis_pose_missing")
    model, _ = _proxy_model(
        mujoco=mujoco,
        model_path=model_file,
        pelvis_world=_finite_vector(
            pelvis.get("position_xyz"), size=3, name="live_pelvis_world"
        ),
        pelvis_rotation=_rotation_wxyz(pelvis.get("quaternion_wxyz")),
        handle_world=_finite_vector(
            focus.get("target_world_xyz_m"), size=3, name="handle_world"
        ),
        hinge_world=_finite_vector(
            focus.get("hinge_world_xyz_m"), size=3, name="hinge_world"
        ),
        hinge_axis_world=_finite_vector(
            focus.get("joint_world_axis_xyz"), size=3, name="hinge_axis_world"
        ),
        focus=focus,
    )
    data = mujoco.MjData(model)
    qpos_addresses: list[int] = []
    for name in SOURCE_ACTION_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
    door_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "microwave_door_hinge"
    )
    door_address = int(model.jnt_qposadr[door_joint])
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames: list[np.ndarray] = []
    for action, door_angle in zip(trajectory, door_angles):
        for address, value in zip(qpos_addresses, action):
            data.qpos[address] = float(value)
        data.qpos[door_address] = float(door_angle)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=EGOCENTRIC_CAMERA_NAME)
        frames.append(renderer.render().copy())
    renderer.close()

    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "observation_state_43d": destination / "observation_state_43d.npy",
        "observation_projected_gravity": (
            destination / "observation_projected_gravity.npy"
        ),
        "action_motion_token": destination / "action_motion_token_64d.npy",
        "teleop_left_hand_joints": destination / "teleop_left_hand_joints_7d.npy",
        "teleop_right_hand_joints": destination / "teleop_right_hand_joints_7d.npy",
        "prescribed_door_angle": destination / "prescribed_door_angle_rad.npy",
        "ego_view": destination / "ego_view.mp4",
        "task_description": destination / "task_description.txt",
    }
    np.save(paths["observation_state_43d"], trajectory, allow_pickle=False)
    np.save(
        paths["observation_projected_gravity"],
        np.tile(
            np.asarray(FIXED_UPRIGHT_PROJECTED_GRAVITY, dtype=np.float32),
            (trajectory.shape[0], 1),
        ),
        allow_pickle=False,
    )
    np.save(paths["action_motion_token"], motion_tokens, allow_pickle=False)
    np.save(paths["teleop_left_hand_joints"], left_hand, allow_pickle=False)
    np.save(paths["teleop_right_hand_joints"], right_hand, allow_pickle=False)
    np.save(paths["prescribed_door_angle"], door_angles, allow_pickle=False)
    iio.imwrite(
        paths["ego_view"],
        np.asarray(frames),
        fps=SOURCE_FPS,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
    )
    paths["task_description"].write_text(TASK_DESCRIPTION + "\n", encoding="utf-8")
    decoded_shapes: list[list[int]] = []
    decoded_sharpness: list[float] = []
    decoded_standard_deviation: list[float] = []
    for decoded in iio.imiter(paths["ego_view"]):
        frame = np.asarray(decoded)
        decoded_shapes.append(list(frame.shape))
        gray = np.mean(frame.astype(np.float64), axis=2)
        decoded_sharpness.append(
            float(np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1)))
        )
        decoded_standard_deviation.append(float(np.std(gray)))
    decode_integrity_passed = bool(
        len(decoded_shapes) == trajectory.shape[0]
        and all(shape == [480, 640, 3] for shape in decoded_shapes)
        and min(decoded_sharpness[1:]) > 1.0
        and min(decoded_standard_deviation[1:]) > 1.0
    )
    if not decode_integrity_passed:
        raise RuntimeError("g1_microwave_training_seed_video_integrity_failed")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified_owned_prescribed_expert_training_seed",
        "task_description": TASK_DESCRIPTION,
        "embodiment_tag": "UNITREE_G1_SONIC",
        "frame_count": int(trajectory.shape[0]),
        "fps": SOURCE_FPS,
        "duration_seconds": float(trajectory.shape[0] / SOURCE_FPS),
        "modality": unitree_g1_sonic_training_modality(),
        "camera": {
            "name": EGOCENTRIC_CAMERA_NAME,
            "width": 640,
            "height": 480,
            "mount": "robot_head_mesh_eye_height_task_directed",
            "third_person_observer_camera": False,
        },
        "video_integrity": {
            "all_encoded_frames_decoded": True,
            "decoded_frame_count": len(decoded_shapes),
            "all_decoded_frames_640x480_rgb": True,
            "frame_1_gradient_variance": decoded_sharpness[0],
            "post_frame_1_gradient_variance_min": min(decoded_sharpness[1:]),
            "post_frame_1_gradient_variance_median": float(
                np.median(decoded_sharpness[1:])
            ),
            "post_frame_1_nonempty_std_min": min(
                decoded_standard_deviation[1:]
            ),
            "garbage_after_frame_1_detected": False,
        },
        "rights": {
            "source": "Blueprint-owned generated simulation trajectory",
            "external_public_dataset_rows_included": False,
            "external_checkpoint_outputs_included": False,
        },
        "source_evidence": {
            "model": {"path": str(model_file), "sha256": _sha256(model_file)},
            "grasp_report": {"path": str(grasp_file), "sha256": _sha256(grasp_file)},
            "dynamic_report": {
                "path": str(dynamic_file),
                "sha256": _sha256(dynamic_file),
            },
            "sonic_conversion_report": {
                "path": str(conversion_file),
                "sha256": _sha256(conversion_file),
            },
        },
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "claim_boundary": {
            "door_angle_in_training_video_is_prescribed": True,
            "separate_handle_only_contact_dynamics_transition_is_proven": True,
            "package_is_not_native_lerobot_materialization": True,
            "package_is_not_a_trained_checkpoint": True,
            "package_is_not_isaac_usd_qualification": True,
            "package_is_not_semantic_episode_success": True,
        },
        "blockers": [
            "native_lerobot_dataset_materialization_not_completed",
            "groot_n1_7_sonic_fine_tune_not_run",
            "trained_checkpoint_qualification_not_run",
            "semantic_episode_success_not_proven",
        ],
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an owned UNITREE_G1_SONIC microwave training seed."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--initial-policy-observation", required=True)
    parser.add_argument("--target-focus-report", required=True)
    parser.add_argument("--grasp-report", required=True)
    parser.add_argument("--dynamic-report", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--sonic-actions", required=True)
    parser.add_argument("--sonic-conversion-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        build_owned_training_seed(
            model_path=args.model,
            initial_policy_observation_path=args.initial_policy_observation,
            target_focus_report_path=args.target_focus_report,
            grasp_report_path=args.grasp_report,
            dynamic_report_path=args.dynamic_report,
            trajectory_path=args.trajectory,
            sonic_actions_path=args.sonic_actions,
            sonic_conversion_report_path=args.sonic_conversion_report,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
