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
ARENA_NEWTON_MANAGER = "isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg.NewtonArenaMJWarpManager"
_RELEASED_BACKEND_SOURCE_SHA256 = {
    "environment": "f53e566eb74f72cb0581e942781c55c7acab008f12f2b8919656d6ac317d9408",
    "builder": "fcf7c088a1c08f998fff4ca6deddc835318ade0d35b38e8db0df460932b9fa79",
    "backend": "369cd8bc4ead94cfcec58deed87e439ef53dbaea4e5f169b9691a9b356a2f054",
    "physics": "97196afd442adc0f8ca26dba23097d6dbeaadfea769acd97375dfca99fac93c1",
}


def arena_backend_kwargs(physics_backend: str) -> dict[str, str]:
    # Arena selects and adapts its embodiment before invoking env_cfg_callback.
    # Changing only cfg.sim.physics in that callback violates the released API.
    return {"default_physics_backend": "newton"} if physics_backend == BACKEND else {}


def preserve_arena_newton_physics(physics: Any) -> Any:
    manager = getattr(getattr(physics, "solver_cfg", None), "class_type", None)
    identity = f"{getattr(manager, '__module__', '')}.{getattr(manager, '__qualname__', '')}"
    if identity != ARENA_NEWTON_MANAGER:
        raise ValueError("newton_feature_arena_physics_preset_missing")
    return physics


def newton_embodiment_configuration(embodiment: Any) -> dict[str, Any]:
    """Record released Newton adaptations separately from the PhysX reference."""
    import math
    robot = embodiment.scene_config.robot
    spawn = robot.spawn
    def normal(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else repr(value)
        if isinstance(value, Mapping):
            return {str(k): normal(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normal(v) for v in value]
        return str(value)
    func = getattr(spawn, "func", None)
    material = getattr(spawn, "physics_material", None)
    fields = ("stiffness", "damping", "armature", "joint_effort_limit", "effort_limit", "effort_limit_sim")
    return {"scope": "released_newton_backend_configuration_not_physx_equivalence",
            "usd_path": str(getattr(spawn, "usd_path", "")),
            "spawn_function": None if func is None else f"{func.__module__}.{func.__qualname__}",
            "disable_gravity": getattr(getattr(spawn, "rigid_props", None), "disable_gravity", None),
            "material": {name: normal(getattr(material, name, None)) for name in
                         ("static_friction", "dynamic_friction", "restitution")},
            "actuators": {name: {field: normal(getattr(cfg, field, None)) for field in fields}
                          for name, cfg in getattr(robot, "actuators", {}).items()},
            "configuration_equivalent_to_physx": False}


def replay_arena_backend_binding(source: Mapping[str, Any], *, legacy_default: bool = False) -> dict[str, Any]:
    """Replay the real released Arena selector and callback guard without Kit.

    Only the retained constructor, selector property and three guard statements
    are compiled. Physics config objects are type witnesses; no simulator,
    geometry, network, or provider is constructed.
    """
    import ast
    import hashlib
    from enum import Enum
    from types import SimpleNamespace
    import zipfile

    prefix = "runtime_sources/arena/isaaclab_arena/"
    names = {"environment": "environments/isaaclab_arena_environment.py",
             "builder": "environments/arena_env_builder.py", "backend": "utils/physics_backend.py",
             "physics": "environments/isaaclab_arena_manager_based_env_cfg.py"}
    with zipfile.ZipFile(source["packet_path"]) as archive:
        files = {key: archive.read(prefix + name) for key, name in names.items()}
    hashes = {key: hashlib.sha256(data).hexdigest() for key, data in files.items()}
    if hashes != _RELEASED_BACKEND_SOURCE_SHA256:
        raise ValueError("newton_arena_backend_source_digest_mismatch")
    trees = {key: ast.parse(data) for key, data in files.items()}
    def class_node(key: str, name: str) -> Any:
        return next(node for node in trees[key].body if isinstance(node, ast.ClassDef) and node.name == name)
    def execute(nodes: list[Any], namespace: dict[str, Any]) -> None:
        future = ast.parse("from __future__ import annotations").body[0]
        module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
        exec(compile(module, "<retained-arena-backend-contract>", "exec"), namespace)

    namespace: dict[str, Any] = {"Enum": Enum, "__name__": "retained_arena_backend_replay"}
    execute([class_node("backend", "PhysicsBackend")], namespace)
    execute([class_node("environment", "IsaacLabArenaEnvironment")], namespace)
    backend = namespace["PhysicsBackend"]
    builder = class_node("builder", "ArenaEnvBuilder")
    selector = next(node for node in builder.body if isinstance(node, ast.FunctionDef)
                    and node.name == "resolved_physics_backend")
    replay_class = ast.parse("class ArenaBackendReplay: pass").body[0]
    replay_class.body = [selector]
    execute([replay_class], namespace)
    starts = [(function, index) for function in builder.body if isinstance(function, ast.FunctionDef)
              for index, node in enumerate(function.body) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "arena_physics" for target in node.targets)]
    if len(starts) != 1:
        raise ValueError("newton_arena_backend_source_contract_changed")
    function, index = starts[0]
    guard = function.body[index:index + 3]
    if (len(guard) != 3 or not isinstance(guard[1], ast.If) or not isinstance(guard[2], ast.If)
            or "changed the physics backend away from PhysX" not in ast.unparse(guard[2])):
        raise ValueError("newton_arena_backend_source_contract_changed")
    physx_type = type("PhysxCfg", (), {})
    class NewtonCfg:
        def __init__(self, **kwargs: Any):
            self.__dict__.update(kwargs)
    newton_type = NewtonCfg
    arena_manager = type("NewtonArenaMJWarpManager", (), {})
    arena_manager.__module__ = ARENA_NEWTON_MANAGER.rsplit(".", 1)[0]
    namespace.update(PhysxCfg=physx_type, NewtonCfg=newton_type, PresetCfg=object,
                     configclass=lambda cls: cls, MJWarpSolverCfg=lambda **kwargs: SimpleNamespace(**kwargs),
                     NewtonArenaMJWarpManager=arena_manager)
    execute([class_node("physics", "ArenaPhysicsCfg")], namespace)
    def configure(cfg: Any) -> Any:
        if legacy_default:
            cfg.sim.physics = newton_type()
        else:
            cfg.sim.physics = preserve_arena_newton_physics(cfg.sim.physics)
        return cfg
    environment = namespace["IsaacLabArenaEnvironment"](
        name="retained_backend_preflight", scene=SimpleNamespace(), env_cfg_callback=configure,
        **({} if legacy_default else arena_backend_kwargs(BACKEND)))
    instance = namespace["ArenaBackendReplay"]()
    instance.cfg = SimpleNamespace(presets=None)
    instance.arena_env = environment
    cfg = SimpleNamespace(sim=SimpleNamespace(physics=None), scene=SimpleNamespace(replicate_physics=False))
    namespace.update(self=instance, env_cfg=cfg, resolved_physics_backend=instance.resolved_physics_backend,
        PhysxCfg=physx_type, NewtonCfg=newton_type)
    failure = None
    try:
        execute(guard, namespace)
    except AssertionError as exc:
        failure = str(exc)
    result = {"schema_version": "native_task_newton_arena_backend_preflight.v1",
              "status": "refused" if failure else "passed", "legacy_default_replayed": legacy_default,
              "selected_backend": str(instance.resolved_physics_backend), "callback_backend": BACKEND,
              "replicate_physics": cfg.scene.replicate_physics, "refusing_predicate": failure,
              "retained_arena_solver": ({name: getattr(cfg.sim.physics.solver_cfg, name) for name in
                  ("njmax", "nconmax", "integrator", "iterations", "use_mujoco_contacts")}
                  if not failure else None),
              "vendor_source_sha256": hashes,
              "native_execution": False, "qualification_effect": "none"}
    if not legacy_default and (failure or instance.resolved_physics_backend is not backend.NEWTON):
        raise ValueError("native_task_newton_arena_backend_preflight_failed:" + str(failure))
    return result


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
    if (qualified != ARENA_NEWTON_MANAGER
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
