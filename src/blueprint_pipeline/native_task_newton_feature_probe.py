"""Separate, opt-in Newton/RTX compatibility smoke on a retained workcell.

The construction transport is reused; this never qualifies construction,
controls, policy outcomes, a runtime improvement, or physical evidence.
"""
from __future__ import annotations

import json
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest

REQUEST_FILENAME = "native_task_newton_feature_request.v1.json"
RESULT_FILENAME = "native_task_newton_feature_result.v1.json"
BACKEND = "newton_mjwarp"
STEPS = 64


def validate_feature_source(source: Mapping[str, Any]) -> None:
    from .native_task_isaaclab_ga import IMAGE, COMMIT, TREE, ARENA_COMMIT, ARENA_TREE
    if (source.get("runtime_profile") != "isaaclab_3_ga" or source.get("container_image") != IMAGE
            or [(row.get("commit"), row.get("tree")) for row in source.get("repositories", [])]
            != [(COMMIT, TREE), (ARENA_COMMIT, ARENA_TREE)]):
        raise ValueError("native_task_newton_feature_ga_sources_required")


def build_feature_request(comparison: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "schema_version": "native_task_newton_feature_request.v1",
        "experiment_kind": "separate_newton_rtx_compatibility_smoke",
        "reference_construction_request_digest": comparison["request_digest"],
        "scene_plan_digest": plan["plan_digest"],
        "runtime_profile": "isaaclab_3_ga", "physics_backend": BACKEND,
        "renderer": "isaac_rtx", "action_mode": "hold_initial_joint_positions",
        "control_steps": STEPS, "seed": int(plan["scenario"]["seed"]),
        "qualification_effect": "none", "candidate_policy_queried": False,
        "runtime_promotion_authorized": False,
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return validate_feature_request(value, comparison=comparison, plan=plan)


def validate_feature_request(value: Mapping[str, Any], *, comparison: Mapping[str, Any],
                             plan: Mapping[str, Any]) -> dict[str, Any]:
    from .native_task_isaaclab_ga import IMAGE
    expected = {
        "schema_version": "native_task_newton_feature_request.v1",
        "experiment_kind": "separate_newton_rtx_compatibility_smoke",
        "reference_construction_request_digest": comparison["request_digest"],
        "scene_plan_digest": plan["plan_digest"], "runtime_profile": "isaaclab_3_ga",
        "physics_backend": BACKEND, "renderer": "isaac_rtx",
        "action_mode": "hold_initial_joint_positions", "control_steps": STEPS,
        "seed": int(plan["scenario"]["seed"]), "qualification_effect": "none",
        "candidate_policy_queried": False, "runtime_promotion_authorized": False,
    }
    expected["request_digest"] = canonical_digest(expected, digest_field="request_digest")
    if (dict(value) != expected or comparison.get("variant") != "isaaclab_3_ga"
            or comparison.get("runtime_identity", {}).get("container_image") != IMAGE
            or comparison.get("request_digest") != canonical_digest(comparison, digest_field="request_digest")):
        raise ValueError("native_task_newton_feature_request_invalid")
    return dict(value)


def prepare_newton_preconstruction(*, expected_device: str) -> dict[str, Any]:
    """Bind CUDA without assigning PhysxManager ownership to a Newton run."""
    import torch
    import warp as wp
    from .native_task_arena_preconstruction import SCHEMA_VERSION
    index = torch.device(expected_device).index
    if index is None or not expected_device.startswith("cuda:"):
        raise ValueError("newton_feature_cuda_device_invalid")
    torch.cuda.set_device(index)
    warp_device = wp.get_device(expected_device)
    observed = {"torch_device": f"cuda:{torch.cuda.current_device()}",
                "warp_device": str(warp_device), "warp_is_cuda": bool(warp_device.is_cuda)}
    passed = (observed["torch_device"] == expected_device
              and observed["warp_device"] == expected_device and observed["warp_is_cuda"])
    value = {"schema_version": SCHEMA_VERSION, "physics_backend": BACKEND,
             "expected_device": expected_device, "observed": observed, "passed": passed,
             "blockers": [] if passed else ["newton_feature_device_binding_failed"]}
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def verify_native_backend(env: Any) -> dict[str, str]:
    """Read the live manager, solver and device, never infer them from flags."""
    sim = env.unwrapped.sim
    manager = sim.physics_manager
    qualified = f"{manager.__module__}.{manager.__name__}"
    solver = getattr(manager, "_solver", None)
    solver_type = f"{type(solver).__module__}.{type(solver).__name__}"
    device = str(manager.get_device())
    if (qualified != "isaaclab_newton.physics.mjwarp_manager.NewtonMJWarpManager"
            or "SolverMuJoCo" not in solver_type or device != "cuda:0"):
        raise ValueError(f"newton_feature_native_backend_mismatch:{qualified}:{solver_type}:{device}")
    return {"manager": qualified, "solver": solver_type, "device": device}


def run_hold_probe(*, built: Any, request: Mapping[str, Any], torch: Any,
                   snapshot: Any, output_root: Path) -> dict[str, Any]:
    """Exercise the real environment interface; injectable for lifecycle rehearsal."""
    env = built.env
    backend = verify_native_backend(env)
    robot = env.unwrapped.scene["robot"]

    def tensor(value: Any) -> Any:
        return torch.as_tensor(getattr(value, "torch", value), device="cuda:0")

    def state() -> Any:
        value = tensor(robot.data.joint_pos)
        if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(tensor(robot.data.joint_vel)).all()):
            raise ValueError("newton_feature_nonfinite_robot_state")
        return value.clone()

    env.reset(seed=request["seed"])
    initial = state()
    action = initial.clone()
    # The retained DROID action contract is seven arm joints plus one gripper
    # command, rather than all thirteen mimic-joint positions.
    names = list(robot.joint_names)
    arm_ids = [names.index(f"panda_joint{i}") for i in range(1, 8)]
    finger_id = names.index("finger_joint")
    action = action[:, [*arm_ids, finger_id]]
    snapshots = [snapshot(env=env, camera_scene_names=built.camera_scene_names,
                          output_root=output_root, snapshot_id="newton_reset")]
    timings = []
    max_drift = 0.0
    for _ in range(request["control_steps"]):
        torch.cuda.synchronize()
        before = time.perf_counter()
        env.step(action.clone())
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - before)
        max_drift = max(max_drift, float((state() - initial).abs().max()))
    snapshots.append(snapshot(env=env, camera_scene_names=built.camera_scene_names,
                              output_root=output_root, snapshot_id="newton_after_hold"))
    env.reset(seed=request["seed"])
    reset_error = float((state() - initial).abs().max())
    frames = [row for item in snapshots for row in item["cameras"]]
    if len(frames) != 6 or any(not row["observability"]["render_passed"] for row in frames):
        raise ValueError("newton_feature_render_frames_unqualified")
    if reset_error > 1e-6:
        raise ValueError("newton_feature_reset_not_repeatable")
    return {"native_backend": backend, "control_steps_completed": len(timings),
            "step_seconds": timings, "max_joint_drift_rad": max_drift,
            "reset_max_joint_error_rad": reset_error, "camera_snapshots": snapshots,
            "passed": True}


def run_feature_probe(*, request: Mapping[str, Any], plan: Mapping[str, Any],
                      runtime: Path, output_root: Path, snapshot: Any,
                      implementation_commit: str) -> tuple[dict[str, Any], Any]:
    """Return measurements and the app; the worker seals its envelope before close."""
    from .native_task_isaaclab_ga import ga_app_launcher
    from .native_task_isaaclab_launch import launch_native_task_isaaclab
    from .native_task_nurec_render_setup import appearance_render_path_from_plan, prepare_site_appearance_renderer
    from .native_task_arena_runtime import build_native_task_arena_environment

    output_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": "native_task_newton_feature_result.v1", "status": "running",
        "implementation_commit": implementation_commit, "scene_plan_digest": plan["plan_digest"],
        "request_digest": request["request_digest"], "physics_backend": BACKEND,
        "renderer": "isaac_rtx", "qualification_effect": "none", "development_only": True,
        "runtime_promotion_authorized": False, "candidate_policy_queried": False,
        "construction_gate_qualified": False, "physical_evidence": False,
        "native_isaac_executed": False, "first_observation_reached": False,
        "blockers": [], "phase": "launch",
        "limitations": ["Separate compatibility smoke; no policy or task-success score.",
                        "Newton may interpret the retained PhysX USD properties differently.",
                        "Timing is not comparable to the frozen PhysX construction replay."],
    }
    app = built = None
    def checkpoint() -> None:
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        (output_root / RESULT_FILENAME).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    try:
        checkpoint()
        before = time.perf_counter()
        app, launch = launch_native_task_isaaclab(
            output_root / "native_task_runtime_source_provisioning.v1.json",
            device="cuda:0", appearance_render_path=appearance_render_path_from_plan(plan),
            app_launcher_factory=partial(ga_app_launcher, physics_backend=BACKEND))
        result.update(startup_seconds=time.perf_counter() - before, launch_receipt=launch, phase="build")
        from .native_task_runtime_comparison import gpu_observation
        from importlib.metadata import version
        result["gpu_observation"] = gpu_observation()
        result["runtime_distributions"] = {name: version(name) for name in
                                            ("newton", "mujoco", "mujoco-warp", "torch", "warp-lang")}
        checkpoint()
        built = build_native_task_arena_environment(plan, bundle_root=runtime / "native_task_packet",
                                                    physics_backend=BACKEND)
        result["native_isaac_executed"] = True
        result["native_backend"] = verify_native_backend(built.env)
        result["native_configuration_readback"] = built.native_configuration_readback
        result["phase"] = "render_warmup"
        checkpoint()
        built.env.reset(seed=request["seed"])
        warmup = prepare_site_appearance_renderer(simulation_app=app, plan=plan,
            retained_packet_request=json.loads((runtime / "native_task_packet/native_task_arena_packet_request.v1.json").read_text()))
        result["render_warmup"] = warmup
        if not warmup["passed"]:
            raise ValueError("newton_feature_appearance_warmup_failed")
        import torch
        result["phase"] = "hold_step_reset"
        checkpoint()
        measured = run_hold_probe(built=built, request=request, torch=torch, output_root=output_root,
                                  snapshot=snapshot)
        result.update(measured, status="completed", phase="complete", first_observation_reached=True)
    except BaseException as exc:
        result["status"] = "blocked"
        result["blockers"] = [f"newton_feature_failed_at_{result['phase']}:{type(exc).__name__}:{exc}"]
        result["traceback"] = traceback.format_exc()
    finally:
        retained = sorted(output_root.glob("construction_frames/*/newton_*.png"))
        result["retained_png_count"] = len(retained)
        result["first_observation_reached"] = bool(retained)
        result["media_gap"] = (None if result["first_observation_reached"] else
                               {"type": "before_first_observation", "reason": result["blockers"][0] if result["blockers"] else "unavailable"})
        checkpoint()
    return result, app
