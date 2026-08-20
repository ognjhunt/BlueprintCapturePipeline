"""Paid GPU preflight for the exact native Arena runtime, without task motion."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any


RESULT_FILENAME = "native_task_arena_runtime_preflight.v1.json"
RESULT_SCHEMA_VERSION = "native_task_arena_runtime_preflight.v1"


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _announce(phase: str, status: str = "started") -> None:
    print(
        f"BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena_preflight:{phase}:{status}",
        flush=True,
    )


def main() -> int:
    runtime = Path(__file__).resolve().parent
    output_root = Path(
        os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", runtime / "runtime_output")
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "preflight_only": True,
        "task_motion_executed": False,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
        "phase_reached": "not_started",
        "blockers": [],
    }
    simulation_app = None
    env = None
    try:
        from blueprint_pipeline.native_task_arena_construction_worker import (
            _articulation_device_binding,
            _camera_snapshot,
            _load_and_verify_manifest,
            preflight_native_dependency_matrix,
        )

        _announce("packet_verification")
        manifest = _load_and_verify_manifest(runtime)
        packet = runtime / "native_task_packet"
        plan = json.loads(
            (packet / "native_task_arena_scene_plan.v1.json").read_text(
                encoding="utf-8"
            )
        )
        result["implementation_commit"] = manifest["implementation_commit"]
        result["manifest_input_digest"] = manifest["input_digest"]
        result["packet_receipt_digest"] = manifest["packet_receipt_digest"]
        result["scene_plan_digest"] = plan["plan_digest"]
        result["phase_reached"] = "packet_verified"
        _announce("packet_verification", "completed")

        _announce("simulation_app")
        from blueprint_pipeline.native_task_isaaclab_launch import (
            NATIVE_TASK_ARENA_DEVICE,
            launch_native_task_isaaclab,
        )

        simulation_app, launch = launch_native_task_isaaclab(
            output_root / "native_task_runtime_source_provisioning.v1.json",
            device=NATIVE_TASK_ARENA_DEVICE,
        )
        result["isaaclab_launch"] = launch
        result["phase_reached"] = "simulation_app_qualified"
        _announce("simulation_app", "completed")

        _announce("dependency_matrix")
        dependencies = preflight_native_dependency_matrix(
            robot_id=str(plan["robot"]["robot_id"])
        )
        result["dependency_matrix"] = dependencies
        if not dependencies["all_required_available"]:
            result["blockers"].extend(dependencies["blockers"])
            raise RuntimeError("native_task_arena_preflight_dependencies_failed")
        _announce("dependency_matrix", "completed")

        from blueprint_pipeline.native_task_arena_preconstruction import (
            prepare_native_task_arena_preconstruction,
        )
        from blueprint_pipeline.native_task_arena_runtime import (
            build_native_task_arena_environment,
        )
        from blueprint_pipeline.native_task_arena_device_readback import (
            read_native_task_arena_device_binding,
        )

        _announce("environment_build")
        preconstruction = prepare_native_task_arena_preconstruction(
            expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        result["preconstruction_device_binding"] = preconstruction
        if not preconstruction["passed"]:
            result["blockers"].extend(preconstruction["blockers"])
            raise RuntimeError("native_task_arena_preflight_preconstruction_failed")
        built = build_native_task_arena_environment(
            plan,
            device=NATIVE_TASK_ARENA_DEVICE,
            bundle_root=packet,
            preconstruction_receipt=preconstruction,
        )
        device = read_native_task_arena_device_binding(
            built, expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        result["device_readback"] = device
        if not device["passed"]:
            result["blockers"].extend(device["blockers"])
            raise RuntimeError("native_task_arena_preflight_device_binding_failed")
        env = built.env
        seed = int(plan["scenario"]["seed"])
        env.reset(seed=seed)
        result["articulation_device_binding"] = _articulation_device_binding(
            built, expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        articulation_rows = result["articulation_device_binding"].get(
            "articulations"
        ) or {}
        if not articulation_rows or not all(
            row.get("on_expected_device") is True and "unavailable" not in row
            for row in articulation_rows.values()
        ):
            result["blockers"].append(
                "native_task_arena_preflight_articulation_device_mismatch"
            )
            raise RuntimeError("native_task_arena_preflight_articulation_device_failed")
        result["phase_reached"] = "environment_built"
        _announce("environment_build", "completed")

        import torch

        from blueprint_pipeline.native_franka_pose_servo import (
            NativeFrankaDifferentialIkServo,
        )

        robot = env.unwrapped.scene["robot"]
        servo = NativeFrankaDifferentialIkServo(env=env, robot=robot)
        result["gripper_frame_axis_readback"] = (
            servo.current_gripper_frame_axis_readback()
        )
        for _ in range(8):
            current = servo.read_arm_joint_positions()
            env.step(
                torch.tensor(
                    [[*current, 0.0]],
                    device=env.unwrapped.device,
                    dtype=torch.float32,
                )
            )
        result["camera_snapshot"] = _camera_snapshot(
            env=env,
            camera_scene_names=built.camera_scene_names,
            output_root=output_root,
            snapshot_id="runtime_preflight",
        )
        camera_rows = result["camera_snapshot"]["cameras"]
        if not camera_rows or not all(
            row["observability"]["passed"] for row in camera_rows
        ):
            result["blockers"].append(
                "native_task_arena_preflight_camera_observability_failed"
            )
            raise RuntimeError("native_task_arena_preflight_camera_failed")
        result["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "environment_device": str(env.unwrapped.device),
        }
        result["phase_reached"] = "runtime_preflight_completed"
        result["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - retained as typed preflight evidence
        result["blockers"].append(
            f"native_task_arena_runtime_preflight_failed_at_{result['phase_reached']}:"
            f"{type(exc).__name__}:{exc}"[:500]
        )
        result["traceback"] = traceback.format_exc()[-6000:]
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                result["blockers"].append(
                    f"native_task_arena_preflight_env_close_failed:{type(exc).__name__}"
                )
                result["status"] = "blocked"
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception as exc:  # noqa: BLE001
                result["blockers"].append(
                    f"native_task_arena_preflight_app_close_failed:{type(exc).__name__}"
                )
                result["status"] = "blocked"
        result["blockers"] = sorted(set(result["blockers"]))
        (output_root / RESULT_FILENAME).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "BLUEPRINT_NATIVE_TASK_ARENA_RUNTIME_PREFLIGHT_"
        + ("OK" if result["status"] == "completed" else "BLOCKED"),
        flush=True,
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
