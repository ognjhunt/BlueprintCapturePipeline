"""Execute one sealed, task-neutral native Franka construction canary.

The worker consumes only ``native_task_packet`` and its provider manifest.  It
does not know a scene id, object class, task coordinate, or candidate outcome.
It verifies the complete dependency matrix before scene construction, applies
the exact Arena plan, measures reset/contact/camera state, and drives the
Franka finger midpoint through the contact-clear phase plan using the same 8-D
absolute action seam later used by controls and learned policies.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = "native_task_arena_construction_result.v1"
RESULT_FILENAME = "native_task_arena_construction_result.v1.json"
DEPENDENCY_IMPORTS = (
    "torch",
    "numpy",
    "PIL.Image",
    "gymnasium",
    "lazy_loader",
    "cloudpickle",
    "farama_notifications",
    "packaging",
    "prettytable",
    "typing_extensions",
    "wcwidth",
    "h5py",
    "yaml",
    "toml",
    "pxr.Usd",
    "pxr.UsdPhysics",
    "pxr.UsdVol",
    "isaaclab",
    "isaaclab_contrib",
    "isaaclab.controllers",
    "isaaclab.utils.math",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaaclab_teleop",
    "isaaclab_physx",
    "isaaclab_physx.physics",
    "isaaclab_arena",
    "isaaclab_arena.environments.arena_env_builder",
)
CAMERA_THRESHOLDS = {
    "external": {"minimum_pixels": 200, "minimum_pixel_fraction": 0.003},
    "wrist": {"minimum_pixels": 120, "minimum_pixel_fraction": 0.002},
    "overview": {"minimum_pixels": 200, "minimum_pixel_fraction": 0.003},
}


def _announce(phase: str, status: str = "started") -> None:
    print(
        f"BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena:{phase}:{status}",
        flush=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any], *, field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def preflight_native_dependency_matrix() -> dict[str, Any]:
    """Probe all worker imports and media tools in one retained receipt."""

    imports = []
    blockers = []
    for name in DEPENDENCY_IMPORTS:
        try:
            module = importlib.import_module(name)
            imports.append(
                {
                    "module": name,
                    "available": True,
                    "version": str(getattr(module, "__version__", "unreported")),
                }
            )
        except Exception as exc:  # noqa: BLE001 - exact missing matrix is evidence
            imports.append(
                {
                    "module": name,
                    "available": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            blockers.append(f"native_task_dependency_missing:{name}")
    tools = []
    for executable in ("ffmpeg", "ffprobe"):
        try:
            completed = subprocess.run(
                [executable, "-version"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            tools.append(
                {
                    "executable": executable,
                    "available": False,
                    "returncode": None,
                    "version_line": "",
                }
            )
            blockers.append(f"native_task_dependency_missing:{executable}")
            continue
        tools.append(
            {
                "executable": executable,
                "available": completed.returncode == 0,
                "returncode": completed.returncode,
                "version_line": (
                    (completed.stdout or completed.stderr).splitlines() or [""]
                )[0],
            }
        )
        if completed.returncode != 0:
            blockers.append(f"native_task_dependency_missing:{executable}")
    return {
        "schema_version": "native_task_dependency_matrix.v1",
        "imports": imports,
        "tools": tools,
        "all_required_available": not blockers,
        "blockers": sorted(set(blockers)),
    }


def _persist(output: Path, result: dict[str, Any]) -> None:
    result["result_digest"] = _canonical_digest(result, field="result_digest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _load_and_verify_manifest(runtime: Path) -> dict[str, Any]:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    path = runtime / "adp_arena_provider_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "native_task_arena_provider_bundle.v1"
        or manifest.get("execution_mode") != "construction_canary"
        or manifest.get("input_digest")
        != canonical_digest(manifest, digest_field="input_digest")
    ):
        raise RuntimeError("native_task_construction_manifest_invalid")
    return manifest


def _finger_separation(robot: Any, *, torch: Any) -> float:
    names = list(robot.data.body_names)
    indices = [names.index(name) for name in ("left_inner_finger", "right_inner_finger")]
    positions = torch.as_tensor(robot.data.body_pose_w)[0, indices, :3]
    return float(torch.linalg.vector_norm(positions[0] - positions[1]))


def _requested_arm_reset(
    *, plan: Mapping[str, Any], servo_binding: Mapping[str, Any]
) -> list[float]:
    resets = plan["robot"]["joint_reset_positions_rad"]
    return [float(resets[name]) for name in servo_binding["arm_joint_names"]]


def _gripper_convention_probe(*, env: Any, robot: Any, seed: int, torch: Any) -> dict[str, Any]:
    separations: dict[str, float] = {}
    for command in (0.0, 1.0):
        env.reset(seed=seed)
        for _ in range(30):
            current = torch.as_tensor(robot.data.joint_pos)[0, :7]
            action = torch.tensor(
                [[*[float(value) for value in current], command]],
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            env.step(action)
        separations[str(command)] = _finger_separation(robot, torch=torch)
    travel = abs(separations["0.0"] - separations["1.0"])
    if travel < 1.0e-3:
        return {
            "status": "ambiguous",
            "finger_separation_m": separations,
            "separation_travel_m": travel,
            "blockers": ["native_task_gripper_convention_travel_below_floor"],
        }
    closed = 1.0 if separations["1.0"] < separations["0.0"] else 0.0
    return {
        "status": "measured",
        "finger_separation_m": separations,
        "separation_travel_m": travel,
        "closed_command": closed,
        "open_command": 1.0 - closed,
        "blockers": [],
    }


def _camera_snapshot(
    *,
    env: Any,
    camera_scene_names: Mapping[str, str],
    output_root: Path,
    snapshot_id: str,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    from blueprint_pipeline.native_task_camera_observability import (
        measure_native_task_camera_observability,
    )

    rows = []
    for role, scene_name in camera_scene_names.items():
        camera = env.unwrapped.scene[scene_name]
        outputs = camera.data.output
        rgb = _jsonable(outputs["rgb"])[0]
        rgb_array = np.asarray(rgb)
        if rgb_array.shape[-1] == 4:
            rgb_array = rgb_array[..., :3]
        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
        semantic = np.asarray(_jsonable(outputs["semantic_segmentation"])[0])
        if semantic.ndim == 3 and semantic.shape[-1] == 1:
            semantic = semantic[..., 0]
        info = _jsonable((camera.data.info or {}).get("semantic_segmentation") or {})
        labels = info.get("idToLabels") or {}
        thresholds = CAMERA_THRESHOLDS[role]
        observability = measure_native_task_camera_observability(
            semantic_ids=semantic,
            id_to_labels=labels,
            target_label="task_object",
            minimum_pixels=thresholds["minimum_pixels"],
            minimum_pixel_fraction=thresholds["minimum_pixel_fraction"],
        )
        frame_dir = output_root / "construction_frames" / role
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"{snapshot_id}.png"
        Image.fromarray(rgb_array, mode="RGB").save(
            frame_path, format="PNG", compress_level=9
        )
        rows.append(
            {
                "role": role,
                "scene_name": scene_name,
                "snapshot_id": snapshot_id,
                "rgb_png": {
                    "path": str(frame_path.relative_to(output_root)),
                    "sha256": _sha256(frame_path),
                },
                "rgb_min": int(rgb_array.min()),
                "rgb_max": int(rgb_array.max()),
                "rgb_mean": float(rgb_array.mean()),
                "intrinsic_matrix": _jsonable(camera.data.intrinsic_matrices)[0],
                "position_world_m": _jsonable(camera.data.pos_w)[0],
                "quaternion_world_opengl_xyzw": _jsonable(
                    camera.data.quat_w_opengl
                )[0],
                "observability": observability,
                "semantic_id_to_labels": labels,
                "native_sensor_timestamp": _jsonable(
                    getattr(camera.data, "frame", None)
                ),
            }
        )
    return {"snapshot_id": snapshot_id, "cameras": rows}


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    runtime = Path(__file__).resolve().parent
    output_root = Path(
        os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR")
        or runtime.parent / "runtime_output"
    ).resolve()
    output = output_root / RESULT_FILENAME
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [],
        "phase_reached": "start",
        "native_isaac_executed": False,
        "construction_gate_qualified": False,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
        "simulator_execution_is_not_physical_truth": True,
    }
    simulation_app = None
    try:
        _announce("packet_verification")
        manifest = _load_and_verify_manifest(runtime)
        result["manifest_input_digest"] = manifest["input_digest"]
        result["implementation_commit"] = manifest["implementation_commit"]
        packet = runtime / "native_task_packet"
        receipt_path = packet / "native_task_arena_packet_receipt.v1.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("receipt_digest") != manifest.get("packet_receipt_digest"):
            raise RuntimeError("native_task_construction_packet_binding_mismatch")
        plan_path = packet / "native_task_arena_scene_plan.v1.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("plan_digest") != manifest.get("arena_scene_plan_digest"):
            raise RuntimeError("native_task_construction_plan_binding_mismatch")
        result["packet_receipt_digest"] = receipt["receipt_digest"]
        result["scene_plan_digest"] = plan["plan_digest"]
        result["scenario"] = plan["scenario"]
        result["phase_reached"] = "packet_verified"
        _announce("packet_verification", "completed")

        _announce("simulation_app")
        from isaacsim.simulation_app import SimulationApp

        simulation_app = SimulationApp(
            {"headless": True, "renderer": "RayTracedLighting"}
        )
        _announce("simulation_app", "completed")
        _announce("dependency_matrix")
        dependency_matrix = preflight_native_dependency_matrix()
        result["dependency_matrix"] = dependency_matrix
        if not dependency_matrix["all_required_available"]:
            result["blockers"].extend(dependency_matrix["blockers"])
            raise RuntimeError("native_task_construction_dependency_preflight_failed")
        result["phase_reached"] = "dependencies_qualified"
        _announce("dependency_matrix", "completed")

        import torch

        from blueprint_pipeline.native_articulated_construction_plan import (
            materialize_articulated_construction_phase_plan,
        )
        from blueprint_pipeline.native_franka_pose_servo import (
            NativeFrankaDifferentialIkServo,
        )
        from blueprint_pipeline.native_task_arena_readback import (
            NativeArticulatedTaskArenaReadback,
        )
        from blueprint_pipeline.native_task_arena_runtime import (
            build_native_task_arena_environment,
        )

        _announce("environment_build")
        built = build_native_task_arena_environment(
            plan, device="cuda:0", bundle_root=packet
        )
        env = built.env
        seed = int(plan["scenario"]["seed"])
        env.reset(seed=seed)
        scene = env.unwrapped.scene
        robot = scene["robot"]
        task_object = scene[built.scene_asset_names["task_object"]]
        readback = NativeArticulatedTaskArenaReadback(built)
        result["native_isaac_executed"] = True
        result["phase_reached"] = "environment_built"
        _announce("environment_build", "completed")

        initial_sample = readback.read_task_sample()
        result["initial_readback"] = {
            "robot_root_pose_world": _jsonable(robot.data.root_pose_w)[0],
            "robot_joint_names": list(robot.joint_names),
            "robot_joint_positions_rad": _jsonable(robot.data.joint_pos)[0],
            "robot_body_names": list(robot.data.body_names),
            "task_joint_names": list(task_object.joint_names),
            "task_sample": initial_sample,
            "scene_asset_names": dict(built.scene_asset_names),
            "contact_sensor_names": dict(built.contact_sensor_names),
            "camera_scene_names": dict(built.camera_scene_names),
        }
        initial_native = initial_sample["native_readback"]
        collision_threshold = float(
            plan["articulation"]["state_thresholds"][
                "collision_failure_minimum_force_n"
            ]
        )
        if (
            initial_native["task_robot_contact_peak_force_n"] >= collision_threshold
            or initial_native["task_scene_contact_peak_force_n"] >= collision_threshold
            or initial_native["robot_scene_contact_peak_force_n"] >= collision_threshold
        ):
            result["blockers"].append("native_task_initial_penetration_or_contact")

        _announce("gripper_convention")
        gripper = _gripper_convention_probe(
            env=env, robot=robot, seed=seed, torch=torch
        )
        result["gripper_convention"] = gripper
        result["blockers"].extend(gripper["blockers"])
        if gripper["status"] != "measured":
            raise RuntimeError("native_task_construction_gripper_convention_unresolved")
        env.reset(seed=seed)
        result["phase_reached"] = "gripper_convention_measured"
        _announce("gripper_convention", "completed")

        servo = NativeFrankaDifferentialIkServo(env=env, robot=robot)
        result["franka_pose_binding"] = servo.binding
        reset_body_pose = servo.current_body_pose_world()
        phase_plan = materialize_articulated_construction_phase_plan(
            plan, clearance_m=0.025, waypoint_count=8
        )
        result["construction_phase_plan"] = phase_plan
        snapshots = []
        for _ in range(8):
            current = servo.read_arm_joint_positions()
            env.step(
                torch.tensor(
                    [[*current, float(gripper["open_command"])]],
                    device=env.unwrapped.device,
                    dtype=torch.float32,
                )
            )
        snapshots.append(
            _camera_snapshot(
                env=env,
                camera_scene_names=built.camera_scene_names,
                output_root=output_root,
                snapshot_id="reset",
            )
        )

        phase_results = []
        total_steps = 0
        max_total_steps = int(plan["cadence"]["maximum_action_steps"])
        for phase in phase_plan["phases"]:
            _announce(f"phase_{phase['phase_id']}")
            servo.reset_command_state()
            stable = 0
            diagnostics = []
            start_position = servo.current_grasp_frame_position_world()
            while total_steps < max_total_steps and len(diagnostics) < 35:
                action, diagnostic = servo.action_for_grasp_target(
                    target_position_world_m=phase["position_world_m"],
                    target_body_quaternion_world_xyzw=reset_body_pose[3:7],
                    gripper_command=float(gripper["open_command"]),
                )
                env.step(
                    torch.tensor(
                        [action],
                        device=env.unwrapped.device,
                        dtype=torch.float32,
                    )
                )
                total_steps += 1
                achieved = servo.current_grasp_frame_position_world()
                error = math.dist(achieved, phase["position_world_m"])
                stable = stable + 1 if error <= 0.02 else 0
                diagnostic["step_index"] = total_steps
                diagnostic["position_error_m"] = error
                diagnostics.append(diagnostic)
                if stable >= 2:
                    break
            terminal = servo.current_grasp_frame_position_world()
            terminal_error = math.dist(terminal, phase["position_world_m"])
            sample = readback.read_task_sample()
            row = {
                "phase_id": phase["phase_id"],
                "target_position_world_m": phase["position_world_m"],
                "start_position_world_m": start_position,
                "terminal_position_world_m": terminal,
                "terminal_position_error_m": terminal_error,
                "arrival_tolerance_m": 0.02,
                "target_reached": terminal_error <= 0.02 and stable >= 2,
                "steps": len(diagnostics),
                "diagnostics": diagnostics[:4] + diagnostics[-2:],
                "task_sample": sample,
            }
            phase_results.append(row)
            snapshots.append(
                _camera_snapshot(
                    env=env,
                    camera_scene_names=built.camera_scene_names,
                    output_root=output_root,
                    snapshot_id=phase["phase_id"],
                )
            )
            _announce(
                f"phase_{phase['phase_id']}",
                "completed" if row["target_reached"] else "blocked",
            )
        result["phase_results"] = phase_results
        result["total_action_steps"] = total_steps
        failed_phases = [
            row["phase_id"] for row in phase_results if not row["target_reached"]
        ]
        result["blockers"].extend(
            f"native_task_phase_ik_unreached:{phase_id}"
            for phase_id in failed_phases
        )

        camera_gates = {}
        for role in built.camera_scene_names:
            observations = [
                next(row for row in snapshot["cameras"] if row["role"] == role)
                for snapshot in snapshots
            ]
            best = max(
                observations,
                key=lambda row: row["observability"]["pixel_count"],
            )
            camera_gates[role] = {
                "passed": any(
                    row["observability"]["passed"] for row in observations
                ),
                "best_snapshot_id": best["snapshot_id"],
                "best_observability": best["observability"],
            }
            if not camera_gates[role]["passed"]:
                result["blockers"].append(
                    f"native_task_camera_observability_failed:{role}"
                )
        result["camera_snapshots"] = snapshots
        result["camera_gates"] = camera_gates

        _announce("reset_replay")
        env.reset(seed=seed)
        reset_sample = readback.read_task_sample()
        reset_arm = servo.read_arm_joint_positions()
        requested_reset = _requested_arm_reset(
            plan=plan, servo_binding=servo.binding
        )
        reset_errors = [
            abs(actual - expected)
            for actual, expected in zip(reset_arm, requested_reset, strict=True)
        ]
        task_reset_errors = {
            joint_id: abs(
                float(reset_sample["joint_positions_rad"][joint_id])
                - float(expected)
            )
            for joint_id, expected in plan["task_spec"][
                "joint_reset_positions_rad"
            ].items()
        }
        reset_passed = max(reset_errors, default=0.0) <= 1.0e-4 and max(
            task_reset_errors.values(), default=0.0
        ) <= float(plan["task_spec"]["reset_tolerance_rad"])
        result["reset_replay"] = {
            "passed": reset_passed,
            "robot_joint_absolute_errors_rad": reset_errors,
            "task_joint_absolute_errors_rad": task_reset_errors,
            "task_sample": reset_sample,
        }
        if not reset_passed:
            result["blockers"].append("native_task_reset_replay_mismatch")
        _announce("reset_replay", "completed" if reset_passed else "blocked")

        result["blockers"] = sorted(set(result["blockers"]))
        result["construction_gate_qualified"] = not result["blockers"]
        result["status"] = (
            "completed" if result["construction_gate_qualified"] else "blocked"
        )
        result["phase_reached"] = "construction_gate_complete"
        _announce(
            "construction_gate",
            "completed" if result["construction_gate_qualified"] else "blocked",
        )
    except BaseException as exc:  # noqa: BLE001 - one paid launch retains every failure
        result["blockers"].append(
            f"native_task_construction_failed_at_{result['phase_reached']}:"
            f"{type(exc).__name__}:{exc}"
        )
        result["blockers"] = sorted(set(result["blockers"]))
        result["status"] = "blocked"
        _announce(str(result["phase_reached"]), "blocked")
    finally:
        result["completed_at_unix_ns"] = time.time_ns()
        _persist(output, result)
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:  # noqa: BLE001
                pass
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
