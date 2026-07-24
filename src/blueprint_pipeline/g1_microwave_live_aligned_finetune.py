"""Rebuild the microwave fine-tune episode from the live Isaac start state.

The original owned demonstration was rendered in a lightweight MuJoCo proxy and
started from the generic GEAR-SONIC standing pose.  The qualification runtime
starts from a task-facing manipulation-ready pose in the textured Isaac kitchen.
Training directly on the proxy episode therefore leaves both the visual
observation and the first controller horizon out of distribution.

This module is embedded into the retained-worker fine-tune component.  It:

1. solves a new owned grasp trajectory from the exact live Isaac proprioception;
2. encodes that trajectory with the pinned official GEAR-SONIC encoder;
3. renders the trajectory from the same rigid head camera in the exact kitchen;
4. replaces the one-episode LeRobot rows and recomputes their numeric statistics.

It does not claim controller execution, contact, articulation transition, or
semantic task success.  Those remain later fail-closed qualification gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "g1_microwave_live_aligned_finetune.v1"
TASK_DESCRIPTION = "Stand at the microwave and open the microwave door."
FRAME_COUNT = 176
FPS = 50
TARGET_PRIM_PATH = "/root/Microwave017/Microwave017_Door"
# The same-session manipulation-ready pose uses the palm-down grasp convention
# seen by the rigid head camera.  The older proxy demonstration used the
# opposite wrist axis plus a 45-degree yaw; carrying that transform into this
# pose leaves the first handle target 49 mm out of reach.  These values were
# qualified against the exact live G1 model and all 51 pull/contact frames.
LIVE_ALIGNED_HAND_AXIS_POLARITY = -1.0
LIVE_ALIGNED_GRASP_YAW_RAD = 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label}_missing_or_unsafe")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label}_not_object")
    return value


def _canonical_joint_positions(initial_state: dict[str, Any]) -> list[float]:
    from blueprint_pipeline.gear_sonic_joint_order_contract import (
        PROTOCOL_V4_FULL_JOINT_ORDER,
    )

    mapping = dict(initial_state.get("proprioception_mapping") or {})
    inventory = list(mapping.get("observed_dof_inventory") or [])
    positions = {
        str(row.get("normalized_name") or row.get("observed_name") or ""): float(
            row["position"]
        )
        for row in inventory
        if isinstance(row, dict)
    }
    if set(positions) != set(PROTOCOL_V4_FULL_JOINT_ORDER):
        raise ValueError("live_aligned_initial_joint_inventory_mismatch")
    result = [positions[name] for name in PROTOCOL_V4_FULL_JOINT_ORDER]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("live_aligned_initial_joint_positions_nonfinite")
    return result


def prepare_actions(
    *,
    initial_state_path: str | Path,
    standing_report_path: str | Path,
    initial_observation_path: str | Path,
    robot_model_path: str | Path,
    encoder_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Solve and encode one trajectory from the exact live start state."""

    from blueprint_pipeline.g1_microwave_grasp_arc_seed import solve_grasp_arc_seed
    from blueprint_pipeline.g1_sonic_motion_token_conversion import (
        FIXED_UPRIGHT_PROJECTED_GRAVITY,
        SOURCE_ACTION_JOINT_NAMES,
        convert_to_sonic_actions,
        fixed_upright_root_anchor_rotations,
        load_onnx_encoder,
    )

    initial_state_file = Path(initial_state_path).expanduser().resolve()
    standing_file = Path(standing_report_path).expanduser().resolve()
    observation_file = Path(initial_observation_path).expanduser().resolve()
    model_file = Path(robot_model_path).expanduser().resolve()
    encoder_file = Path(encoder_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    initial_state = _load_object(initial_state_file, label="live_aligned_initial_state")
    standing = _load_object(standing_file, label="live_aligned_standing_report")
    observation = _load_object(
        observation_file, label="live_aligned_initial_observation"
    )
    live_positions = _canonical_joint_positions(initial_state)
    aligned_standing = dict(standing)
    aligned_standing.update(
        {
            "measured_full_joint_positions": live_positions,
            "source": "same_session_live_isaac_manipulation_ready_proprioception",
            "claim_boundary": (
                "This derived seed binds the owned IK start to the same-session "
                "live Isaac manipulation-ready state. It is not controller or "
                "task-success proof."
            ),
        }
    )
    camera_context = dict(observation.get("camera_projection_context") or {})
    camera_contract = dict(camera_context.get("camera_contract") or {})
    focus = dict(camera_contract.get("task_target_focus") or {})
    if (
        focus.get("target_prim_path") != observation.get("target_prim_path")
        or focus.get("target_prim_path") != TARGET_PRIM_PATH
    ):
        raise ValueError("live_aligned_task_focus_binding_invalid")

    destination.mkdir(parents=True, exist_ok=True)
    aligned_standing_path = destination / "aligned_standing_initialization.json"
    focus_path = destination / "live_task_focus_report.json"
    aligned_standing_path.write_text(
        json.dumps(aligned_standing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    focus_path.write_text(
        json.dumps(focus, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trajectory, grasp_report = solve_grasp_arc_seed(
        model_path=model_file,
        standing_initialization_path=aligned_standing_path,
        initial_policy_observation_path=observation_file,
        target_focus_report_path=focus_path,
        reach_frame_count=101,
        closure_frame_count=26,
        pull_frame_count=51,
        door_open_angle_rad=0.35,
        hand_axis_polarity=LIVE_ALIGNED_HAND_AXIS_POLARITY,
        grasp_yaw_rad=LIVE_ALIGNED_GRASP_YAW_RAD,
    )
    if trajectory.shape != (FRAME_COUNT, len(SOURCE_ACTION_JOINT_NAMES)):
        raise RuntimeError("live_aligned_trajectory_shape_invalid")
    anchors = fixed_upright_root_anchor_rotations(
        FRAME_COUNT,
        fixed_base_upright_attested=True,
    )
    encoder = load_onnx_encoder(encoder_file)
    sonic_actions, conversion_report = convert_to_sonic_actions(
        trajectory,
        action_joint_names=SOURCE_ACTION_JOINT_NAMES,
        root_anchor_rotations_6d=anchors,
        encoder=encoder,
        source_provenance={
            "source_type": "same_session_live_aligned_owned_isaac_seed",
            "initial_state_sha256": _sha256(initial_state_file),
            "initial_observation_sha256": _sha256(observation_file),
            "robot_model_sha256": _sha256(model_file),
        },
        fps=float(FPS),
    )
    paths = {
        "observation_state": destination / "observation_state_43d.npy",
        "projected_gravity": destination / "observation_projected_gravity.npy",
        "motion_token": destination / "action_motion_token_64d.npy",
        "left_hand": destination / "teleop_left_hand_joints_7d.npy",
        "right_hand": destination / "teleop_right_hand_joints_7d.npy",
    }
    np.save(paths["observation_state"], trajectory.astype(np.float32), allow_pickle=False)
    np.save(
        paths["projected_gravity"],
        np.tile(
            np.asarray(
                initial_state.get(
                    "projected_gravity", FIXED_UPRIGHT_PROJECTED_GRAVITY
                ),
                dtype=np.float32,
            ),
            (FRAME_COUNT, 1),
        ),
        allow_pickle=False,
    )
    np.save(paths["motion_token"], sonic_actions[:, :64], allow_pickle=False)
    np.save(paths["left_hand"], sonic_actions[:, 64:71], allow_pickle=False)
    np.save(paths["right_hand"], sonic_actions[:, 71:78], allow_pickle=False)
    grasp_path = destination / "live_aligned_grasp_report.json"
    conversion_path = destination / "live_aligned_sonic_conversion_report.json"
    grasp_path.write_text(
        json.dumps(grasp_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    conversion_path.write_text(
        json.dumps(conversion_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "live_start_aligned_actions_prepared",
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "task_target_prim_path": TARGET_PRIM_PATH,
        "initial_state_sha256": _sha256(initial_state_file),
        "initial_observation_sha256": _sha256(observation_file),
        "robot_model_sha256": _sha256(model_file),
        "encoder_sha256": _sha256(encoder_file),
        "grasp_geometry": dict(grasp_report.get("geometry") or {}),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "blockers": [],
        "claim_boundary": {
            "same_session_live_start_bound": True,
            "owned_kinematic_reach_only": True,
            "controller_execution_not_proven": True,
            "isaac_contact_not_proven": True,
            "semantic_success_not_proven": True,
        },
    }
    report_path = destination / "live_aligned_action_preparation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _load_runtime_backend_overlay() -> Any:
    """Load the hash-verified task backend materialized by the prior episode."""

    import blueprint_pipeline  # noqa: F401

    source = Path("/workspace/runtime_overlay/isaac_runtime_task_backend.py")
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("live_aligned_isaac_backend_overlay_missing")
    name = "blueprint_pipeline.isaac_runtime_task_backend"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("live_aligned_isaac_backend_overlay_spec_missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def render_isaac(
    *,
    seed_dir: str | Path,
    stage_path: str | Path,
    g1_usd_path: str | Path,
    route_file: str | Path,
    evidence_dir: str | Path,
) -> dict[str, Any]:
    """Render the aligned expert states from the exact rigid head camera."""

    seed = Path(seed_dir).expanduser().resolve()
    trajectory = np.load(seed / "observation_state_43d.npy", allow_pickle=False)
    if trajectory.shape != (FRAME_COUNT, 43) or not np.isfinite(trajectory).all():
        raise ValueError("live_aligned_isaac_trajectory_invalid")
    from blueprint_pipeline.g1_sonic_motion_token_conversion import (
        SOURCE_ACTION_JOINT_NAMES,
    )

    backend_module = _load_runtime_backend_overlay()
    backend = backend_module.create_backend(
        stage_path=str(Path(stage_path).expanduser().resolve()),
        robot_prim_path="/World/G1",
        evidence_dir=str(Path(evidence_dir).expanduser().resolve()),
        g1_usd_path=str(Path(g1_usd_path).expanduser().resolve()),
        route_file=str(Path(route_file).expanduser().resolve()),
    )
    frames_dir = seed / "isaac_head_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=False)
    try:
        registration = backend._live_robot_registration_link_poses()
        backend.review_renderer.set_initial_robot_pov_calibration_landmarks(
            registration["landmarks"]
        )
        indices = np.asarray(
            [int(backend.robot.get_dof_index(name)) for name in SOURCE_ACTION_JOINT_NAMES],
            dtype=np.int64,
        )
        if np.any(indices < 0) or len(set(indices.tolist())) != 43:
            raise RuntimeError("live_aligned_isaac_joint_mapping_invalid")
        zeros = np.zeros(43, dtype=np.float32)
        for frame_index, positions in enumerate(trajectory):
            backend.robot.set_joint_positions(
                np.asarray(positions, dtype=np.float32),
                joint_indices=indices,
            )
            backend.robot.set_joint_velocities(zeros, joint_indices=indices)
            set_targets = getattr(backend.robot, "set_joint_position_targets", None)
            if callable(set_targets):
                set_targets(
                    np.asarray(positions, dtype=np.float32),
                    joint_indices=indices,
                )
            rows = list(
                backend.review_renderer.render(
                    step_index=frame_index,
                    target_prim_path=TARGET_PRIM_PATH,
                )
            )
            robot_rows = [
                dict(row)
                for row in rows
                if isinstance(row, dict) and row.get("camera_role") == "robot_pov"
            ]
            if len(robot_rows) != 1:
                raise RuntimeError("live_aligned_isaac_robot_pov_not_unique")
            source = Path(str(robot_rows[0].get("path") or "")).resolve()
            if not source.is_file() or _sha256(source) != robot_rows[0].get("sha256"):
                raise RuntimeError("live_aligned_isaac_robot_pov_digest_invalid")
            shutil.copyfile(source, frames_dir / f"frame_{frame_index:06d}.png")
    finally:
        backend.close()

    video = seed / "ego_view.mp4"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
            str(video),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not video.is_file() or video.stat().st_size <= 0:
        raise RuntimeError("live_aligned_isaac_video_encode_failed")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "exact_isaac_rigid_head_episode_rendered",
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "video_path": str(video),
        "video_sha256": _sha256(video),
        "stage_path": str(Path(stage_path).expanduser().resolve()),
        "stage_sha256": _sha256(Path(stage_path).expanduser().resolve()),
        "camera_role": "robot_pov",
        "camera_motion_model": "rigid_head_local_transform",
        "third_person_used_for_policy": False,
        "door_motion_in_training_render": "closed_visual_state_only",
        "blockers": [],
        "claim_boundary": {
            "exact_isaac_visual_domain_rendered": True,
            "render_is_owned_training_support": True,
            "contact_not_proven": True,
            "articulation_transition_not_proven": True,
            "semantic_success_not_proven": True,
        },
    }
    report_path = seed / "live_aligned_isaac_render_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _numeric_stats(array: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(array, dtype=np.float64)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def patch_dataset(*, seed_dir: str | Path, dataset_dir: str | Path) -> dict[str, Any]:
    """Replace the one LeRobot episode and all action/state statistics."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    seed = Path(seed_dir).expanduser().resolve()
    dataset = Path(dataset_dir).expanduser().resolve()
    arrays = {
        "observation.state": np.load(
            seed / "observation_state_43d.npy", allow_pickle=False
        ).astype(np.float64),
        "observation.projected_gravity": np.load(
            seed / "observation_projected_gravity.npy", allow_pickle=False
        ).astype(np.float64),
        "action.motion_token": np.load(
            seed / "action_motion_token_64d.npy", allow_pickle=False
        ).astype(np.float64),
        "teleop.left_hand_joints": np.load(
            seed / "teleop_left_hand_joints_7d.npy", allow_pickle=False
        ).astype(np.float32),
        "teleop.right_hand_joints": np.load(
            seed / "teleop_right_hand_joints_7d.npy", allow_pickle=False
        ).astype(np.float32),
    }
    if any(value.shape[0] != FRAME_COUNT for value in arrays.values()):
        raise ValueError("live_aligned_dataset_array_horizon_invalid")
    parquet_path = dataset / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(parquet_path)
    for name, values in arrays.items():
        index = table.schema.get_field_index(name)
        if index < 0:
            raise ValueError(f"live_aligned_dataset_column_missing:{name}")
        table = table.set_column(
            index,
            name,
            pa.array(values.tolist(), type=table.schema.field(index).type),
        )
    pq.write_table(table, parquet_path, compression="snappy")
    video_target = (
        dataset
        / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
    )
    shutil.copyfile(seed / "ego_view.mp4", video_target)

    timestamps = np.arange(FRAME_COUNT, dtype=np.float64) / float(FPS)
    stats = {name: _numeric_stats(values) for name, values in arrays.items()}
    stats["timestamp"] = _numeric_stats(timestamps[:, None])
    fingerprints = {
        name: "sha256:" + hashlib.sha256(
            np.ascontiguousarray(values).tobytes()
        ).hexdigest()
        for name, values in {**arrays, "timestamp": timestamps[:, None]}.items()
    }
    stats["__fingerprints__"] = fingerprints
    stats_path = dataset / "meta/stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    episode_stats = {key: value for key, value in stats.items() if key != "__fingerprints__"}
    (dataset / "meta/episodes_stats.jsonl").write_text(
        json.dumps({"episode_index": 0, "stats": episode_stats}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    materialization_path = dataset / "materialization_report.json"
    materialization = _load_object(
        materialization_path, label="live_aligned_materialization_report"
    )
    materialization["live_alignment"] = {
        "schema_version": SCHEMA_VERSION,
        "status": "same_session_live_start_and_exact_isaac_head_view_bound",
        "preparation_report_sha256": _sha256(
            seed / "live_aligned_action_preparation.json"
        ),
        "render_report_sha256": _sha256(seed / "live_aligned_isaac_render_report.json"),
        "stats_sha256": _sha256(stats_path),
        "parquet_sha256": _sha256(parquet_path),
        "video_sha256": _sha256(video_target),
    }
    materialization_path.write_text(
        json.dumps(materialization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preflight_path = dataset / "groot_n17_finetune_preflight.json"
    preflight = _load_object(
        preflight_path, label="live_aligned_groot_finetune_preflight"
    )
    loader = dict(preflight.get("training_loader") or {})
    loader["stats_sha256"] = _sha256(stats_path)
    loader["live_aligned_isaac_head_view"] = True
    preflight["training_loader"] = loader
    preflight["live_alignment"] = dict(materialization["live_alignment"])
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "live_aligned_lerobot_episode_patched",
        "dataset_dir": str(dataset),
        "parquet_sha256": _sha256(parquet_path),
        "video_sha256": _sha256(video_target),
        "stats_sha256": _sha256(stats_path),
        "frame_count": FRAME_COUNT,
        "blockers": [],
    }
    report_path = seed / "live_aligned_dataset_patch_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-actions")
    prepare.add_argument("--initial-state", required=True)
    prepare.add_argument("--standing-report", required=True)
    prepare.add_argument("--initial-observation", required=True)
    prepare.add_argument("--robot-model", required=True)
    prepare.add_argument("--encoder", required=True)
    prepare.add_argument("--output-dir", required=True)
    render = subparsers.add_parser("render-isaac")
    render.add_argument("--seed-dir", required=True)
    render.add_argument("--stage", required=True)
    render.add_argument("--g1-usd", required=True)
    render.add_argument("--route-file", required=True)
    render.add_argument("--evidence-dir", required=True)
    patch = subparsers.add_parser("patch-dataset")
    patch.add_argument("--seed-dir", required=True)
    patch.add_argument("--dataset-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "prepare-actions":
        prepare_actions(
            initial_state_path=args.initial_state,
            standing_report_path=args.standing_report,
            initial_observation_path=args.initial_observation,
            robot_model_path=args.robot_model,
            encoder_path=args.encoder,
            output_dir=args.output_dir,
        )
    elif args.command == "render-isaac":
        render_isaac(
            seed_dir=args.seed_dir,
            stage_path=args.stage,
            g1_usd_path=args.g1_usd,
            route_file=args.route_file,
            evidence_dir=args.evidence_dir,
        )
    else:
        patch_dataset(seed_dir=args.seed_dir, dataset_dir=args.dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
