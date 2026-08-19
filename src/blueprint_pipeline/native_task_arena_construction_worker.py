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
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = "native_task_arena_construction_result.v1"
RESULT_FILENAME = "native_task_arena_construction_result.v1.json"
DEPENDENCY_IMPORTS = (
    "warp",
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
    "antlr4",
    "omegaconf",
    "hydra",
    "hydra.core",
    "msgpack",
    "zmq",
    "tensordict",
    "importlib_metadata",
    "zipp",
    "orjson",
    "pyvers",
    "git",
    "gitdb",
    "smmap",
    "lightwheel_sdk",
    "lightwheel_sdk.loader",
    "requests",
    "charset_normalizer",
    "idna",
    "urllib3",
    "certifi",
    "tqdm",
    "termcolor",
    "click",
    "rsl_rl",
    "rsl_rl.runners",
    "pxr.Usd",
    "pxr.UsdPhysics",
    "pxr.UsdVol",
    "isaaclab",
    "isaaclab.controllers",
    "isaaclab.utils.math",
    "isaaclab_assets",
    "isaaclab_contrib",
    "isaaclab_experimental",
    "isaaclab_mimic",
    "isaaclab_newton",
    "isaaclab_ov",
    "isaaclab_physx",
    "isaaclab_physx.physics",
    "isaaclab_rl",
    "isaaclab_tasks",
    "isaaclab_tasks_experimental",
    "isaaclab_teleop",
    "isaaclab_visualizers",
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


def _quaternion_angle_xyzw(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        raise RuntimeError("native_task_construction_quaternion_invalid")
    qa = [float(value) for value in a]
    qb = [float(value) for value in b]
    if not all(math.isfinite(value) for value in [*qa, *qb]):
        raise RuntimeError("native_task_construction_quaternion_invalid")
    norm_a = math.sqrt(sum(value * value for value in qa))
    norm_b = math.sqrt(sum(value * value for value in qb))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise RuntimeError("native_task_construction_quaternion_invalid")
    dot = abs(
        sum(left * right for left, right in zip(qa, qb, strict=True))
        / (norm_a * norm_b)
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _pose_arrival_readback(
    *,
    position_world_m: Sequence[float],
    target_position_world_m: Sequence[float],
    orientation_world_xyzw: Sequence[float],
    target_orientation_world_xyzw: Sequence[float],
    position_tolerance_m: float,
    orientation_tolerance_rad: float | None,
) -> dict[str, Any]:
    position_error = math.dist(position_world_m, target_position_world_m)
    orientation_error = _quaternion_angle_xyzw(
        orientation_world_xyzw, target_orientation_world_xyzw
    )
    reached = position_error <= float(position_tolerance_m) and (
        orientation_tolerance_rad is None
        or orientation_error <= float(orientation_tolerance_rad)
    )
    return {
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "reached": reached,
    }


def _retain_task_path_samples(*, task_kind: str, task_spec: Mapping[str, Any]) -> bool:
    return task_kind == "rigid_pick_place" or (
        task_kind == "articulated_open_close"
        and task_spec.get("schema_version") == "adp_task_spec.v2"
    )


def _evaluate_task_construction_gates(
    *,
    phase_plan: Mapping[str, Any],
    phase_results: Sequence[Mapping[str, Any]],
    reset_replay: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    from blueprint_pipeline.native_task_construction_plan import (
        evaluate_graph_articulated_construction_gates,
        evaluate_rigid_construction_gates,
    )

    schema = phase_plan.get("schema_version")
    if schema == "native_rigid_construction_phase_plan.v1":
        return (
            "rigid_construction_gates",
            evaluate_rigid_construction_gates(
                phase_plan=phase_plan,
                phase_results=phase_results,
                reset_replay=reset_replay,
            ),
        )
    if schema == "native_articulated_graph_construction_phase_plan.v1":
        return (
            "articulated_graph_construction_gates",
            evaluate_graph_articulated_construction_gates(
                phase_plan=phase_plan,
                phase_results=phase_results,
                reset_replay=reset_replay,
            ),
        )
    return None


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


def preflight_native_dependency_matrix(*, robot_id: str) -> dict[str, Any]:
    """Probe all worker imports and media tools in one retained receipt."""

    imports = []
    blockers = []
    try:
        from blueprint_pipeline.native_task_arena_import_scope import (
            install_scoped_arena_embodiment,
        )

        embodiment_scope = install_scoped_arena_embodiment(robot_id)
    except Exception as exc:  # noqa: BLE001 - exact scope failure is evidence
        embodiment_scope = {
            "schema_version": "native_task_arena_embodiment_scope.v1",
            "robot_id": str(robot_id),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        blockers.append(f"native_task_arena_embodiment_scope_failed:{robot_id}")
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
                    "traceback": traceback.format_exc(),
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
        "embodiment_scope": embodiment_scope,
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


def _verified_construction_phase_plan_path(
    runtime: Path, manifest: Mapping[str, Any]
) -> Path:
    rows = manifest.get("bound_runtime_inputs")
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError("native_task_construction_runtime_inputs_invalid")
    row = rows[0]
    if not isinstance(row, Mapping):
        raise RuntimeError("native_task_construction_runtime_inputs_invalid")
    relative = str(row.get("relative_path") or "")
    path = runtime / relative
    if (
        relative != "runtime_inputs/native_task_construction_phase_plan.v1.json"
        or not path.is_file()
        or path.stat().st_size != row.get("size_bytes")
        or _sha256(path) != row.get("sha256")
    ):
        raise RuntimeError("native_task_construction_phase_plan_identity_mismatch")
    return path


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


def _task_joint_reset_passed(
    *, absolute_errors_rad: Mapping[str, float], task_spec: Mapping[str, Any]
) -> bool:
    """Apply a joint tolerance only when the task scorer declares joint resets."""

    if not absolute_errors_rad:
        return True
    try:
        tolerance = float(task_spec["reset_tolerance_rad"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("native_task_joint_reset_tolerance_missing") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise RuntimeError("native_task_joint_reset_tolerance_invalid")
    return max(float(value) for value in absolute_errors_rad.values()) <= tolerance


def _initial_contact_blocked(
    *, task_kind: str, sample: Mapping[str, Any], collision_threshold_n: float
) -> bool:
    channels = [
        float(sample["task_robot_contact_peak_force_n"]),
        float(sample["robot_scene_contact_peak_force_n"]),
    ]
    if task_kind == "rigid_pick_place":
        channels.append(
            float(sample["robot_task_forbidden_collision_peak_force_n"])
        )
    channels.append(
        float(
            sample[
                "task_scene_contact_peak_force_n"
                if task_kind == "articulated_open_close"
                else "task_scene_collision_peak_force_n"
            ]
        )
    )
    return max(channels) >= float(collision_threshold_n)


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


def expected_articulation_prim_paths(plan: Mapping[str, Any]) -> list[str]:
    """Concrete env-0 prim paths for every articulation the plan declares.

    The plan carries `{ENV_REGEX_NS}` templates; the tensor views want a real
    path. Deriving them here means the device report does not depend on a stage
    traversal, which is exactly what was unavailable when it was needed.
    """

    env_ns = "/World/envs/env_0"
    paths = [f"{env_ns}/Robot"]
    for entry in plan.get("objects", []) or []:
        if str(entry.get("object_type", "")).upper() != "ARTICULATION":
            continue
        prim_path = str(entry.get("prim_path", "")).replace("{ENV_REGEX_NS}", env_ns)
        if prim_path and prim_path not in paths:
            paths.append(prim_path)
    return paths


# Ordered stage accessors, most-certain first. `isaacsim.core.utils` is NOT
# present in this runtime -- r11 spent $0.056 collecting nothing because the
# diagnostic imported it and gave up. Its appearance in a shipped isaaclab
# source file proved only that upstream references it, not that this image
# ships it. Both entries below are used by isaaclab's own runtime code paths,
# and a single missing module can no longer blind the whole diagnostic.
_STAGE_ACCESSORS: tuple[tuple[str, str], ...] = (
    ("omni.usd", "omni.usd.get_context().get_stage()"),
    ("isaaclab.sim.utils.stage", "isaaclab.sim.utils.stage.get_current_stage()"),
    ("isaacsim.core.utils.stage", "isaacsim.core.utils.stage.get_current_stage()"),
)


def _current_stage() -> tuple[Any, dict[str, Any]]:
    """Return the live USD stage and a note of how it was reached."""

    attempts: dict[str, Any] = {}
    for module_name, _description in _STAGE_ACCESSORS:
        try:
            if module_name == "omni.usd":
                import omni.usd

                stage = omni.usd.get_context().get_stage()
            elif module_name == "isaaclab.sim.utils.stage":
                from isaaclab.sim.utils import stage as stage_utils

                stage = stage_utils.get_current_stage()
            else:
                import isaacsim.core.utils.stage as stage_utils

                stage = stage_utils.get_current_stage()
        except Exception as exc:  # noqa: BLE001
            attempts[module_name] = f"{type(exc).__name__}:{exc}"[:160]
            continue
        if stage is None:
            attempts[module_name] = "returned_none"
            continue
        attempts[module_name] = "ok"
        return stage, {"stage_source": module_name, "stage_attempts": attempts}
    return None, {"stage_source": None, "stage_attempts": attempts}


def physics_scene_device_evidence(
    articulation_prim_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report what PhysX was actually configured with.

    Articulation initialisation happens inside the environment build, so a
    failure there leaves no built object to interrogate. What does survive is
    the PhysX manager (the traceback runs through it, so it is certainly
    importable) and the USD stage carrying the `physxScene:*` attributes Isaac
    Lab authored. That is the only place the GPU-dynamics decision is
    observable: PhysX resolves an unsupported scene in C++ without logging it,
    and the first visible symptom is a CPU-backed tensor view.

    Nothing here may raise, and no single import may make the whole report
    empty -- each fact is collected independently.
    """

    evidence: dict[str, Any] = {}
    # The PhysX manager first: it needs no stage, and it holds the device the
    # whole decision hangs on.
    try:
        from isaaclab_physx.physics.physx_manager import PhysxManager

        evidence["physics_manager_device"] = str(PhysxManager.get_device())
        evidence["articulation_view_devices"] = _articulation_view_devices(
            PhysxManager, articulation_prim_paths
        )
    except Exception as exc:  # noqa: BLE001
        evidence["physics_manager_unavailable"] = f"{type(exc).__name__}:{exc}"[:200]
    try:
        from isaaclab.sim import SimulationContext

        instance = SimulationContext.instance()
        evidence["simulation_context_device"] = str(getattr(instance, "device", None))
        for setting in (
            "/physics/suppressReadback",
            "/physics/cudaDevice",
            "/physics/physxDispatcher",
        ):
            try:
                evidence.setdefault("settings", {})[setting] = str(
                    instance.get_setting(setting)
                )
            except Exception:  # noqa: BLE001
                evidence.setdefault("settings", {})[setting] = "unreadable"
    except Exception as exc:  # noqa: BLE001
        evidence["simulation_context_unavailable"] = f"{type(exc).__name__}:{exc}"[:200]

    stage, stage_note = _current_stage()
    evidence.update(stage_note)
    if stage is None:
        return evidence
    scenes: dict[str, Any] = {}
    try:
        for prim in stage.Traverse():
            # GetTypeName alone identifies the scene; an IsA(SchemaBase) guard
            # would only add a way for this to raise.
            if prim.GetTypeName() != "PhysicsScene":
                continue
            attributes: dict[str, Any] = {}
            for attribute in prim.GetAttributes():
                name = attribute.GetName()
                if not name.startswith(("physxScene:", "physics:")):
                    continue
                try:
                    attributes[name] = str(attribute.Get())
                except Exception:  # noqa: BLE001
                    attributes[name] = "unreadable"
            scenes[str(prim.GetPath())] = attributes
    except Exception as exc:  # noqa: BLE001
        evidence["traverse_failed"] = f"{type(exc).__name__}:{exc}"[:200]
    evidence["physics_scenes"] = scenes
    return evidence


def _articulation_view_devices(
    physx_manager: Any, prim_paths: Sequence[str] | None
) -> dict[str, Any]:
    """Report the backing device of every articulation view.

    This is the question a device-mismatch traceback cannot answer. If every
    articulation is CPU-backed the scene never got GPU dynamics at all, and the
    cause is scene-level. If exactly one is CPU-backed the cause is that asset.
    Those two answers point at completely different fixes.

    Paths come from the scene plan rather than a stage traversal, so this still
    reports when no stage accessor resolves.
    """

    view = getattr(physx_manager, "_view", None)
    if view is None:
        return {"unavailable": "simulation_view_is_none"}
    candidates = list(prim_paths or ())
    if not candidates:
        return {"unavailable": "no_articulation_prim_paths_supplied"}
    rows: dict[str, Any] = {}
    for path in candidates:
        try:
            articulation = view.create_articulation_view(path)
            velocities = articulation.get_dof_velocities()
            rows[path] = {
                "device": str(getattr(velocities, "device", None)),
                "backend_present": getattr(articulation, "_backend", None) is not None,
            }
        except Exception as exc:  # noqa: BLE001
            rows[path] = {"unavailable": f"{type(exc).__name__}:{exc}"[:200]}
    return rows


def _articulation_device_binding(
    built: Any, *, expected_device: str
) -> dict[str, Any]:
    """Report the device backing each articulation's joint-state arrays.

    Isaac Lab raises the mismatch from inside a Warp kernel launch, naming the
    kernel argument, so the message cannot say which asset is on the wrong
    device. This reads each articulation directly and says so.
    """

    rows: dict[str, Any] = {"expected_device": expected_device, "articulations": {}}
    try:
        scene = built.env.unwrapped.scene
    except Exception as exc:  # the scene may not be reachable on some failures
        rows["unavailable"] = f"{type(exc).__name__}:{exc}"[:200]
        return rows
    for name in sorted(set(built.scene_asset_names.values()) | {"robot"}):
        entry: dict[str, Any] = {}
        try:
            asset = scene[name]
            data = asset.data
            for field in ("joint_pos", "joint_vel"):
                value = getattr(data, field, None)
                entry[field] = str(getattr(value, "device", None))
            entry["num_joints"] = len(getattr(asset, "joint_names", []) or [])
            entry["num_actuators"] = len(getattr(asset, "actuators", {}) or {})
            entry["data_device"] = str(getattr(data, "device", None))
            entry["on_expected_device"] = entry.get("joint_vel") == expected_device
        except Exception as exc:
            entry["unavailable"] = f"{type(exc).__name__}:{exc}"[:200]
        rows["articulations"][name] = entry
    return rows


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
        from blueprint_pipeline.native_task_construction_plan import (
            materialize_native_task_construction_phase_plan,
        )

        frozen_phase_path = _verified_construction_phase_plan_path(runtime, manifest)
        frozen_phase_plan = json.loads(
            frozen_phase_path.read_text(encoding="utf-8")
        )
        recomputed_phase_plan = materialize_native_task_construction_phase_plan(plan)
        if frozen_phase_plan != recomputed_phase_plan:
            raise RuntimeError("native_task_construction_phase_plan_binding_mismatch")
        phase_plan = frozen_phase_plan
        result["construction_phase_plan"] = phase_plan
        result["phase_reached"] = "packet_verified"
        _announce("packet_verification", "completed")

        _announce("simulation_app")
        from blueprint_pipeline.native_task_isaaclab_launch import (
            NATIVE_TASK_ARENA_DEVICE,
            launch_native_task_isaaclab,
        )

        simulation_app, launch_receipt = launch_native_task_isaaclab(
            output_root / "native_task_runtime_source_provisioning.v1.json",
            device=NATIVE_TASK_ARENA_DEVICE,
        )
        result["isaaclab_launch"] = launch_receipt
        _announce("simulation_app", "completed")
        _announce("dependency_matrix")
        dependency_matrix = preflight_native_dependency_matrix(
            robot_id=str(plan["robot"]["robot_id"])
        )
        result["dependency_matrix"] = dependency_matrix
        if not dependency_matrix["all_required_available"]:
            result["blockers"].extend(dependency_matrix["blockers"])
            raise RuntimeError("native_task_construction_dependency_preflight_failed")
        result["phase_reached"] = "dependencies_qualified"
        _announce("dependency_matrix", "completed")

        import torch

        from blueprint_pipeline.native_franka_pose_servo import (
            NativeFrankaDifferentialIkServo,
        )
        from blueprint_pipeline.native_task_arena_readback import (
            NativeArticulatedTaskArenaReadback,
            NativeRigidTaskArenaReadback,
            read_native_task_arena_object_reset_state,
            read_native_task_arena_scenario_parameters,
        )
        from blueprint_pipeline.native_task_arena_device_readback import (
            read_native_task_arena_device_binding,
        )
        from blueprint_pipeline.native_task_arena_preconstruction import (
            prepare_native_task_arena_preconstruction,
        )
        from blueprint_pipeline.native_task_arena_runtime import (
            build_native_task_arena_environment,
        )

        _announce("preconstruction_device_binding")
        preconstruction = prepare_native_task_arena_preconstruction(
            expected_device="cuda:0"
        )
        result["preconstruction_device_binding"] = preconstruction
        if not preconstruction["passed"]:
            result["blockers"].extend(preconstruction["blockers"])
            raise RuntimeError("native_task_arena_preconstruction_failed")
        _announce("preconstruction_device_binding", "completed")

        _announce("environment_build")
        try:
            built = build_native_task_arena_environment(
                plan,
                device="cuda:0",
                bundle_root=packet,
                preconstruction_receipt=preconstruction,
            )
        except Exception as exc:
            # Articulation views are created here, inside sim.reset(). Attempts
            # r6-r10 all died in this call with a cuda/cpu mismatch naming a
            # Warp kernel argument rather than the asset or the setting that
            # demoted the scene. Record what PhysX was configured with before
            # re-raising, so the next failure is diagnosable from the receipt.
            result["environment_build_failure"] = {
                "error": f"{type(exc).__name__}:{exc}"[:400],
                "traceback": traceback.format_exc()[-4000:],
                "physics_scene_device_evidence": physics_scene_device_evidence(
                    expected_articulation_prim_paths(plan)
                ),
            }
            raise
        device_readback = read_native_task_arena_device_binding(
            built, expected_device="cuda:0"
        )
        result["device_readback"] = device_readback
        if not device_readback["passed"]:
            result["blockers"].extend(device_readback["blockers"])
            raise RuntimeError("native_task_arena_device_binding_failed")
        env = built.env
        seed = int(plan["scenario"]["seed"])
        # env.reset() computes observations, which touches every articulation's
        # joint state. Attempts r6-r9 all died inside it with a cuda/cpu array
        # mismatch that names a kernel argument and not the asset, so record
        # which articulation is backed by which device before re-raising: one
        # run of certainty instead of another round of hypotheses.
        result["articulation_device_binding"] = _articulation_device_binding(
            built, expected_device=str(preconstruction["expected_device"])
        )
        try:
            env.reset(seed=seed)
        except Exception as exc:
            result["reset_failure"] = {
                "error": f"{type(exc).__name__}:{exc}"[:400],
                "traceback": traceback.format_exc()[-4000:],
                "articulation_device_binding": result["articulation_device_binding"],
            }
            raise
        scene = env.unwrapped.scene
        robot = scene["robot"]
        task_object = scene[built.scene_asset_names["task_object"]]
        task_kind = str(plan["task_kind"])
        readback = (
            NativeArticulatedTaskArenaReadback(built)
            if task_kind == "articulated_open_close"
            else NativeRigidTaskArenaReadback(built)
        )
        result["native_isaac_executed"] = True
        result["phase_reached"] = "environment_built"
        _announce("environment_build", "completed")

        initial_sample = readback.read_task_sample()
        scenario_parameter_readback = read_native_task_arena_scenario_parameters(built)
        result["scenario_parameter_readback"] = scenario_parameter_readback
        if not scenario_parameter_readback["passed"]:
            result["blockers"].append(
                "native_task_scenario_parameter_readback_mismatch"
            )
        result["initial_readback"] = {
            "robot_root_pose_world": _jsonable(robot.data.root_pose_w)[0],
            "robot_joint_names": list(robot.joint_names),
            "robot_joint_positions_rad": _jsonable(robot.data.joint_pos)[0],
            "robot_body_names": list(robot.data.body_names),
            "task_joint_names": list(getattr(task_object, "joint_names", ()) or ()),
            "task_sample": initial_sample,
            "scene_asset_names": dict(built.scene_asset_names),
            "contact_sensor_names": dict(built.contact_sensor_names),
            "camera_scene_names": dict(built.camera_scene_names),
        }
        initial_native = initial_sample.get("native_readback") or initial_sample
        collision_threshold = float(
            (
                plan["articulation"]["state_thresholds"]
                if task_kind == "articulated_open_close"
                else phase_plan["thresholds"]
            )["collision_failure_minimum_force_n"]
        )
        if _initial_contact_blocked(
            task_kind=task_kind,
            sample=initial_native,
            collision_threshold_n=collision_threshold,
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
        execution_parameters = phase_plan["execution_parameters"]
        arrival_tolerance = float(execution_parameters["arrival_tolerance_m"])
        default_orientation_tolerance = execution_parameters.get(
            "arrival_orientation_tolerance_rad"
        )
        stable_samples = int(execution_parameters["stable_samples"])
        maximum_steps_per_phase = int(
            execution_parameters["maximum_steps_per_phase"]
        )
        for phase in phase_plan["phases"]:
            _announce(f"phase_{phase['phase_id']}")
            servo.reset_command_state()
            stable = 0
            diagnostics = []
            start_position = servo.current_grasp_frame_position_world()
            start_body_pose = servo.current_body_pose_world()
            target_orientation = phase.get(
                "orientation_world_xyzw", reset_body_pose[3:7]
            )
            orientation_tolerance = phase.get(
                "arrival_orientation_tolerance_rad",
                default_orientation_tolerance,
            )
            gripper_command = float(
                gripper[
                    "closed_command"
                    if phase.get("gripper_state") == "closed"
                    else "open_command"
                ]
            )
            task_samples = []
            while (
                total_steps < max_total_steps
                and len(diagnostics) < maximum_steps_per_phase
            ):
                action, diagnostic = servo.action_for_grasp_target(
                    target_position_world_m=phase["position_world_m"],
                    target_body_quaternion_world_xyzw=phase.get(
                        "orientation_world_xyzw", reset_body_pose[3:7]
                    ),
                    gripper_command=gripper_command,
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
                achieved_body_pose = servo.current_body_pose_world()
                arrival = _pose_arrival_readback(
                    position_world_m=achieved,
                    target_position_world_m=phase["position_world_m"],
                    orientation_world_xyzw=achieved_body_pose[3:7],
                    target_orientation_world_xyzw=target_orientation,
                    position_tolerance_m=arrival_tolerance,
                    orientation_tolerance_rad=(
                        None
                        if orientation_tolerance is None
                        else float(orientation_tolerance)
                    ),
                )
                orientation_error = arrival["orientation_error_rad"]
                stable = stable + 1 if arrival["reached"] else 0
                diagnostic["step_index"] = total_steps
                diagnostic["position_error_m"] = error
                diagnostic["orientation_error_rad"] = orientation_error
                diagnostics.append(diagnostic)
                if _retain_task_path_samples(
                    task_kind=task_kind, task_spec=plan["task_spec"]
                ):
                    task_samples.append(readback.read_task_sample())
                required_stable = (
                    int(phase_plan["settle_window_samples"])
                    if phase.get("phase_id") == "settle_observe"
                    else stable_samples
                )
                if stable >= required_stable:
                    break
            terminal = servo.current_grasp_frame_position_world()
            terminal_body_pose = servo.current_body_pose_world()
            terminal_error = math.dist(terminal, phase["position_world_m"])
            terminal_arrival = _pose_arrival_readback(
                position_world_m=terminal,
                target_position_world_m=phase["position_world_m"],
                orientation_world_xyzw=terminal_body_pose[3:7],
                target_orientation_world_xyzw=target_orientation,
                position_tolerance_m=arrival_tolerance,
                orientation_tolerance_rad=(
                    None
                    if orientation_tolerance is None
                    else float(orientation_tolerance)
                ),
            )
            terminal_orientation_error = terminal_arrival["orientation_error_rad"]
            sample = readback.read_task_sample()
            row = {
                "phase_id": phase["phase_id"],
                "target_position_world_m": phase["position_world_m"],
                "start_position_world_m": start_position,
                "start_body_orientation_world_xyzw": start_body_pose[3:7],
                "terminal_position_world_m": terminal,
                "terminal_position_error_m": terminal_error,
                "target_orientation_world_xyzw": target_orientation,
                "terminal_body_orientation_world_xyzw": terminal_body_pose[3:7],
                "terminal_orientation_error_rad": terminal_orientation_error,
                "arrival_orientation_tolerance_rad": orientation_tolerance,
                "arrival_tolerance_m": arrival_tolerance,
                "target_reached": (
                    terminal_arrival["reached"]
                    and stable >= required_stable
                ),
                "gripper_state": phase.get("gripper_state", "open"),
                "gripper_command": gripper_command,
                "gate_ids": list(phase.get("gate_ids") or []),
                "steps": len(diagnostics),
                "diagnostics": diagnostics[:4] + diagnostics[-2:],
                "task_sample": sample,
                "task_samples": task_samples,
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
        task_joint_resets = dict(
            phase_plan.get("joint_reset_positions")
            if phase_plan.get("schema_version")
            == "native_articulated_graph_construction_phase_plan.v1"
            else plan["task_spec"].get("joint_reset_positions_rad", {})
        )
        reset_joint_positions = reset_sample.get("joint_positions")
        if reset_joint_positions is None:
            reset_joint_positions = reset_sample.get("joint_positions_rad", {})
        task_reset_errors = {
            joint_id: abs(
                float(reset_joint_positions[joint_id])
                - float(expected)
            )
            for joint_id, expected in task_joint_resets.items()
        }
        object_reset_readback = read_native_task_arena_object_reset_state(built)
        if phase_plan.get("schema_version") == (
            "native_articulated_graph_construction_phase_plan.v1"
        ):
            task_joint_reset_passed = all(
                error
                <= float(phase_plan["joint_reset_tolerances"][joint_id])
                for joint_id, error in task_reset_errors.items()
            )
        else:
            task_joint_reset_passed = _task_joint_reset_passed(
                absolute_errors_rad=task_reset_errors,
                task_spec=plan["task_spec"],
            )
        reset_passed = (
            max(reset_errors, default=0.0) <= 1.0e-4
            and task_joint_reset_passed
            and object_reset_readback["passed"]
        )
        result["reset_replay"] = {
            "passed": reset_passed,
            "robot_joint_absolute_errors_rad": reset_errors,
            "task_joint_absolute_errors_rad": task_reset_errors,
            "object_reset_readback": object_reset_readback,
            "task_sample": reset_sample,
        }
        if not reset_passed:
            result["blockers"].append("native_task_reset_replay_mismatch")
        if not object_reset_readback["passed"]:
            result["blockers"].append(
                "native_task_object_reset_replay_mismatch"
            )
        task_gate_evaluation = _evaluate_task_construction_gates(
            phase_plan=phase_plan,
            phase_results=phase_results,
            reset_replay=result["reset_replay"],
        )
        if task_gate_evaluation is not None:
            gate_key, gate_result = task_gate_evaluation
            result[gate_key] = gate_result
            result["blockers"].extend(gate_result["blockers"])
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
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": result["phase_reached"],
            "traceback": traceback.format_exc(),
        }
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
