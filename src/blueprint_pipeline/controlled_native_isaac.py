"""ADP-050/day 28: one sealed Franka task, controlled inputs and native scoring.

Only the operator's digest-bound runtime configuration selects the scene and
robot. Policy code is never loaded into this simulator interpreter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .controlled_policy_outcome import NativeOutcomeRecorder
from .controlled_simulator_adapter import ControlledSimulatorAdapter
from .decision_evidence_contracts import canonical_digest


def validate_native_controller_interface(contract: Mapping[str, Any]) -> None:
    """Bind action tensor positions to the offered arm and gripper semantics."""
    robot = contract["robot"]
    action = contract["action_schema"]
    joints = robot["joint_names"]
    channels = action["channels"]
    if (action["adapter_id"] != "absolute_joint_position_gripper_v1"
            or len(joints) != 7 or len(channels) != 8
            or [row["name"] for row in channels] != [*joints, robot["gripper"]["name"]]
            or any(channel["unit"] != limit["unit"] or channel["unit"] != "radian"
                   or channel["executed_semantics"] != "absolute_joint_position"
                   for channel, limit in zip(channels[:7], robot["joint_limits"], strict=True))
            or channels[7]["unit"] != robot["gripper"]["unit"]
            or channels[7]["executed_semantics"] != robot["gripper"]["executed_semantics"]):
        raise ValueError("controlled_native_controller_interface_mismatch")


def build_controlled_native_environment(*, runtime_root: Path, configuration: Mapping[str, Any],
                                       job_request: Mapping[str, Any], observation: Mapping[str, Any],
                                       evidence_root: Path) -> ControlledSimulatorAdapter:
    from .native_task_isaaclab_launch import launch_native_task_isaaclab, NATIVE_TASK_ARENA_DEVICE
    from .native_task_arena_runtime import build_native_task_arena_environment
    from .native_task_arena_preconstruction import prepare_native_task_arena_preconstruction
    from .native_task_arena_construction_worker import _gripper_convention_probe, preflight_native_dependency_matrix
    from .native_task_arena_device_readback import read_native_task_arena_device_binding
    from .native_franka_pose_servo import NativeFrankaDifferentialIkServo
    from .native_task_arena_readback import NativeRigidTaskArenaReadback
    from .native_task_episode_environment import build_native_task_episode_environment, NativeRigidScoringEnvironment
    from .native_task_nurec_render_setup import prepare_site_appearance_renderer, appearance_render_path_from_plan

    packet = runtime_root / "native_task_packet"
    plan = json.loads((packet / "native_task_arena_scene_plan.v1.json").read_text())
    contract = configuration["contract"]
    validate_native_controller_interface(contract)
    from .native_task_runtime_source_packet import ISAACLAB_COMMIT, ARENA_COMMIT
    robot_definition = {"robot": plan["robot"], "isaaclab_commit": ISAACLAB_COMMIT, "arena_commit": ARENA_COMMIT}
    if contract["robot"]["definition_digest"] != canonical_digest(robot_definition):
        raise ValueError("controlled_native_robot_definition_binding_mismatch")
    declared_cameras = {row["name"]: row for row in contract["observation_schema"]["cameras"]}
    plan_cameras = {row["role"]: row for row in plan["cameras"] if row.get("policy_input") is True}
    for name, role in configuration["camera_roles"].items():
        native_camera = plan_cameras[role]
        declared_camera = declared_cameras[name]
        if (declared_camera["calibration_digest"] != canonical_digest(native_camera)
                or declared_camera["width"] != native_camera["intrinsics"]["width"]
                or declared_camera["height"] != native_camera["intrinsics"]["height"]):
            raise ValueError("controlled_native_camera_binding_mismatch")
    if (plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
            or plan.get("plan_digest") != configuration["scene_plan_digest"]
            or plan.get("task_kind") != "rigid_pick_place"
            or plan.get("task_id") != observation.get("task_id")
            or job_request["requested_tasks"][0]["task_id"] != plan["task_id"]
            or (plan.get("scenario") or {}).get("cell_id") != configuration["native_cell_id"]
            or observation.get("scenario_id") != configuration["scenario_id"]
            or plan["cadence"]["control_frequency_hz"] != contract["observation_schema"]["control_frequency_hz"]):
        raise ValueError("controlled_native_frozen_task_binding_mismatch")
    simulation_app = None
    built = None
    recorder = None
    try:
        simulation_app, launch = launch_native_task_isaaclab(
            evidence_root / "native_task_runtime_source_provisioning.v1.json",
            device=NATIVE_TASK_ARENA_DEVICE, appearance_render_path=appearance_render_path_from_plan(plan))
        deps = preflight_native_dependency_matrix(robot_id=str(plan["robot"]["robot_id"]))
        preconstruction = prepare_native_task_arena_preconstruction(expected_device=NATIVE_TASK_ARENA_DEVICE)
        if deps.get("all_required_available") is not True or preconstruction.get("passed") is not True:
            raise ValueError("controlled_native_runtime_preflight_failed")
        built = build_native_task_arena_environment(plan, device=NATIVE_TASK_ARENA_DEVICE,
            bundle_root=packet, preconstruction_receipt=preconstruction)
        appearance = prepare_site_appearance_renderer(simulation_app=simulation_app, plan=plan)
        device = read_native_task_arena_device_binding(built, expected_device=NATIVE_TASK_ARENA_DEVICE)
        if appearance.get("passed") is not True or device.get("passed") is not True:
            raise ValueError("controlled_native_scene_or_device_binding_failed")
        import torch
        env = built.env
        robot = env.unwrapped.scene["robot"]
        seed = int(plan["scenario"]["seed"])
        gripper = _gripper_convention_probe(env=env, robot=robot, seed=seed, torch=torch)
        if gripper.get("status") != "measured":
            raise ValueError("controlled_native_gripper_not_measured")
        env.reset(seed=seed)
        servo = NativeFrankaDifferentialIkServo(env=env, robot=robot, gripper_convention=gripper)
        readback = NativeRigidTaskArenaReadback(built)
        def to_tensor(value):
            if hasattr(value, "detach"):
                return value
            import warp as wp
            return wp.to_torch(value)
        native, native_receipt = build_native_task_episode_environment(built=built, gripper_convention=gripper,
            servo=servo, task_readback=readback, to_tensor=to_tensor)
        native = NativeRigidScoringEnvironment(environment=native, task_readback=readback, task_spec=plan["task_spec"])
        native.begin_episode()
        from .adp009d_isaac_episode_adapter import DROID_ARM_JOINT_NAMES
        if list(contract["robot"]["joint_names"]) != list(DROID_ARM_JOINT_NAMES):
            raise ValueError("controlled_native_joint_order_mismatch")
        measured_limits = native.joint_limits()
        declared = contract["robot"]["joint_limits"]
        if (len(measured_limits) != 7 or len(declared) != 7
                or any(abs(float(row[side]) - limits[index]) > 1e-4
                       for row, limits in zip(declared, measured_limits, strict=True)
                       for index, side in enumerate(("lower", "upper")))):
            raise ValueError("controlled_native_measured_joint_limits_mismatch")
        recorder = NativeOutcomeRecorder(task_spec=plan["task_spec"], read_sample=native.read_object_sample,
            evidence_path=evidence_root / "native_task_samples.jsonl")
        initial_joints = list(native.read_arm_joint_positions())
        def translate(row, environment, schema):
            if len(row) != 8:
                raise ValueError("controlled_native_action_width_mismatch")
            grip = float(gripper["open_command"]) + (float(gripper["closed_command"])
                - float(gripper["open_command"])) * row[7]
            return environment.bounded_joint_action(target_joint_positions_rad=row[:7], gripper_command=grip,
                max_joint_delta_rad=float(configuration["max_joint_delta_rad"]),
                max_joint_setpoint_lead_rad=float(configuration["max_joint_setpoint_lead_rad"]))
        stopped = {"value": False}
        adapter = ControlledSimulatorAdapter(native_environment=native, contract=contract,
            camera_bindings=configuration["camera_bindings"], state_bindings=configuration["state_bindings"],
            state_units={row["name"]: row["unit"] for row in contract["observation_schema"]["state_fields"]},
            control_frequency_hz=float(native_receipt["control_frequency_hz"]), prompt=plan["task_spec"]["prompt"],
            translate_action=translate, terminal=lambda: stopped["value"] or recorder.steps >= int(plan["cadence"]["maximum_action_steps"]),
            stop_controller=lambda: stopped.update(value=True), outcome_recorder=recorder)
        adapter.native_runtime_receipt = {"launch": launch, "device": device, "appearance": appearance,
            "environment": native_receipt, "gripper": gripper, "initial_joint_positions_rad": initial_joints}
        adapter.native_simulation_app = simulation_app
        adapter.native_built = built
        return adapter
    except BaseException:
        if recorder is not None:
            recorder.close()
        if built is not None:
            built.env.close()
        if simulation_app is not None:
            simulation_app.close()
        raise


def read_controlled_native_outcome(*, environment: ControlledSimulatorAdapter,
                                  episode_receipt: Mapping[str, Any], **_context: Any) -> Mapping[str, Any]:
    return environment.outcome_recorder.finish(executed_motor_steps=episode_receipt["executed_motor_steps"])
