"""GPU child for the one174-command native replay; never constructs a policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from blueprint_pipeline.native_task_composition_diagnostic import seal, verify_record
from blueprint_pipeline.native_task_retained_command_replay import (
    RetainedReplayAdapters,
    run_retained_command_replay,
    validate_retained_replay_request,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    args = parser.parse_args(argv)
    runtime = Path(args.runtime_root).resolve()
    output = Path(os.environ["BLUEPRINT_ADP_ARENA_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    app = env = None
    result = {
        "schema_version": "native_task_retained_command_child.v1",
        "status": "blocked",
        "candidate_policy_queried": False,
        "policy_queries": 0,
        "model_calls": 0,
        "task_motion_executed": False,
        "blockers": [],
    }
    try:
        from blueprint_pipeline.native_task_arena_construction_worker import (
            _load_and_verify_manifest,
            preflight_native_dependency_matrix,
        )
        from blueprint_pipeline.native_task_arena_preconstruction import (
            prepare_native_task_arena_preconstruction,
        )
        from blueprint_pipeline.native_task_arena_runtime import build_native_task_arena_environment
        from blueprint_pipeline.native_task_isaaclab_launch import (
            launch_native_task_isaaclab,
            NATIVE_TASK_ARENA_DEVICE,
        )
        from blueprint_pipeline.native_task_nurec_render_setup import (
            setup_and_warm_native_nurec_renderer,
        )
        from blueprint_pipeline.native_franka_pose_servo import NativeFrankaDifferentialIkServo
        from blueprint_pipeline.native_task_episode_environment import (
            build_native_task_episode_environment,
        )
        from blueprint_pipeline.native_task_arena_readback import NativeRigidTaskArenaReadback
        from blueprint_pipeline.policy_scientific_reset import (
            read_native_reset_channels,
            seal_reset_readback,
        )
        from blueprint_pipeline.native_task_composition_worker import make_native_adapters, _plain

        manifest = _load_and_verify_manifest(runtime, expected_execution_mode="runtime_preflight")
        for row in [*manifest["bound_runtime_inputs"], *manifest["runtime_modules"]]:
            verify_record(runtime / row["relative_path"], row)
        for row in manifest["packet_files"]:
            verify_record(runtime / "native_task_packet" / row["relative_path"], row)
        actual_module = runtime / "blueprint_pipeline" / Path(__file__).name
        if actual_module.resolve() != Path(__file__).resolve():
            raise ValueError("retained_replay_worker_module_not_from_sealed_bundle")
        request = validate_retained_replay_request(
            json.loads((runtime / "runtime_inputs/replay_request.json").read_text())
        )
        source_path = runtime / "runtime_inputs/retained_cell_result.json"
        verify_record(source_path, request["source_result"])
        source = json.loads(source_path.read_text())
        source_index = int(request["source_episode_json_pointer"].rsplit("/", 1)[1])
        source_episode = source["episodes"][source_index]["episode"]
        if (
            source_episode["commanded_actions"][:174] != request["commands"]
            or source_episode["scientific_reset"] != request["expected_reset"]
        ):
            raise ValueError("retained_replay_tape_or_reset_not_from_source_bytes")
        verify_record(
            runtime / "runtime_inputs/retained_adapter_reset.json",
            request["retained_adapter_reset"],
        )
        plan = json.loads((runtime / "runtime_inputs/replay_scene_plan.json").read_text())
        if plan["plan_digest"] != request["scene_plan_digest"]:
            raise ValueError("retained_replay_plan_digest_mismatch")
        result.update(
            implementation_commit=manifest["implementation_commit"],
            container_image=manifest["container_image"],
            bundle_input_digest=manifest["input_digest"],
            request_digest=request["request_digest"],
        )
        app, launch = launch_native_task_isaaclab(
            output / "native_task_runtime_source_provisioning.v1.json",
            device=NATIVE_TASK_ARENA_DEVICE,
            appearance_render_path="particlefield_3d_gaussian_splat",
        )
        result["isaaclab_launch"] = launch
        if not preflight_native_dependency_matrix(robot_id=plan["robot"]["robot_id"])[
            "all_required_available"
        ]:
            raise ValueError("retained_replay_dependencies_missing")
        preconstruction = prepare_native_task_arena_preconstruction(
            expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        if not preconstruction["passed"]:
            raise ValueError("retained_replay_device_unqualified")
        built = build_native_task_arena_environment(
            plan,
            device=NATIVE_TASK_ARENA_DEVICE,
            bundle_root=runtime / "native_task_packet",
            preconstruction_receipt=preconstruction,
        )
        env = built.env
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        warmup = setup_and_warm_native_nurec_renderer(
            app, stage, require_display_referred_particlefield=True
        )
        result["renderer_warmup"] = warmup
        if not warmup["passed"]:
            raise ValueError("retained_replay_renderer_unqualified")
        env.reset(seed=request["seed"])
        robot = env.unwrapped.scene["robot"]
        # This calibration only supports metadata/width readback. All actual
        # gripper commands are the already-native tape values, never remapped.
        calibration = json.loads(
            (runtime / "runtime_inputs/retained_adapter_reset.json").read_text()
        )["adapter_binding"]["gripper_command_mapping"]
        convention = {
            "closed_command": calibration["closed_command"],
            "open_command": calibration["open_command"],
            "finger_separation_m": {
                str(float(calibration["closed_command"])): calibration[
                    "closed_finger_separation_m"
                ],
                str(float(calibration["open_command"])): calibration["open_finger_separation_m"],
            },
        }
        servo = NativeFrankaDifferentialIkServo(env=env, robot=robot)
        adapter, binding = build_native_task_episode_environment(
            built=built,
            gripper_convention=convention,
            servo=servo,
            task_readback=NativeRigidTaskArenaReadback(built),
            to_tensor=lambda value: getattr(value, "torch", value),
        )
        result["adapter_binding"] = binding
        result["retained_calibration_reused_without_new_gripper_probe"] = True
        camera_request = {"camera_role": "external", "render_refresh_count": 8}
        camera = make_native_adapters(built=built, stage=stage, request=camera_request)
        names = list(getattr(robot.data, "joint_names", None) or robot.joint_names)
        indices = list(adapter._arm_joint_indices)

        def state():
            data = robot.data
            return {
                "joint_names": names,
                "arm_joint_indices": indices,
                "arm_joint_names": [names[i] for i in indices],
                "joint_position_rad": adapter.read_arm_joint_positions(),
                "joint_limits_rad": adapter.joint_limits(),
                "joint_velocity_rad_s": adapter._arm_vector("joint_vel"),
                "joint_position_target_rad": adapter._arm_vector("joint_pos_target"),
                "computed_torque_nm": adapter._arm_vector("computed_torque"),
                "applied_torque_nm": adapter._arm_vector("applied_torque"),
                "joint_effort_limits_nm": adapter._arm_vector("joint_effort_limits"),
                "all_native_joint_positions": _plain(data.joint_pos),
                "all_native_joint_limits": _plain(data.joint_limits),
                "native_body_pose_world": _plain(data.body_pose_w),
                "physics_time_seconds": float(env.unwrapped.sim.current_time),
                "physics_step_index": int(env.unwrapped.sim.current_time_step_index),
                "native_storage_type": type(data.joint_pos).__module__
                + "."
                + type(data.joint_pos).__name__,
                "torque_authority": "native_simulator_buffers_not_physical_measurement",
            }

        def reset():
            adapter.reset()
            channels = read_native_reset_channels(built, adapter)
            return seal_reset_readback(binding=request["expected_reset"]["binding"], **channels)

        def snapshot(label, root):
            before = camera.read_fixed_state()
            generation = camera.sensor_generation()
            row = camera.render_and_capture(label, root / label)
            if camera.read_fixed_state() != before or camera.sensor_generation() <= generation:
                raise ValueError("retained_replay_snapshot_changed_state_or_stale_aov")
            return {"label": label, "camera": row, "root_relative_path": "frames/" + label}

        replay = run_retained_command_replay(
            request,
            output_root=output / "replay",
            adapters=RetainedReplayAdapters(reset, state, adapter.step, snapshot),
        )
        result.update(
            status=replay["status"],
            task_motion_executed=replay["task_motion_executed"],
            replay=replay,
            blockers=replay["blockers"],
        )
    except Exception as exc:
        result["blockers"].append(type(exc).__name__ + ":" + str(exc))
    finally:
        seal(result)
        (output / "native_task_retained_command_child.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
