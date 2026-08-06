#!/usr/bin/env python3
"""Run the frozen ADP-009B exact-SimReady probes in native Isaac Sim 6.0.1.

This worker is intentionally self-contained for the canonical provider bundle.
It loads the exact SAGE collision composition plus the approved replacement,
executes drop, slide, tip, and bounded two-finger proxy probes, and persists a
terminal digest-bound result before SimulationApp shutdown. The result is
simulation-only and cannot establish physical or robot-task success.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


RESULT_SCHEMA_VERSION = "adp009b_simready_isaac_result.v1"
ISAAC_SIM_VERSION = "6.0.1"


def _canonical_digest(value: Mapping[str, Any], *, field: str) -> str:
    normalized = dict(value)
    normalized.pop(field, None)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("simready_isaac_json_not_object")
    return value


def _persist(path: Path, value: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def _import_simulation_app() -> Any:
    try:
        root_package = importlib.import_module("isaacsim")
        package_path = getattr(root_package, "__path__", None)
        extension = Path("/isaac-sim/exts/isaacsim.simulation_app")
        namespace = extension / "isaacsim"
        if namespace.is_dir() and package_path is not None and hasattr(package_path, "append"):
            if str(namespace) not in package_path:
                package_path.append(str(namespace))
            if str(extension) not in sys.path:
                sys.path.insert(0, str(extension))
            importlib.invalidate_caches()
    except Exception:
        pass
    for module_name in ("isaacsim.simulation_app", "isaacsim", "omni.isaac.kit"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        simulation_app = getattr(module, "SimulationApp", None)
        if callable(simulation_app):
            return simulation_app
    raise RuntimeError("simready_isaac_simulation_app_unavailable")


def _live_transform(omni_physx: Any, prim_path: str) -> dict[str, list[float]]:
    state = omni_physx.get_physx_interface().get_rigidbody_transformation(prim_path)
    if not hasattr(state, "get") or state.get("ret_val") is not True:
        raise RuntimeError("simready_isaac_live_rigid_transform_unavailable")
    position = state.get("position")
    rotation = state.get("rotation")
    return {
        "position": [float(position[index]) for index in range(3)],
        "rotation_xyzw": [float(rotation[index]) for index in range(4)],
    }


def _contact_count(
    omni_physx: Any,
    physics_schema_tools: Any,
    roots: Sequence[str],
) -> int:
    report = omni_physx.get_physx_simulation_interface().get_contact_report()
    headers = report[0] if report else []
    count = 0
    for header in headers:
        paths: list[str] = []
        for name in ("actor0", "actor1", "collider0", "collider1"):
            value = getattr(header, name, "")
            try:
                paths.append(str(physics_schema_tools.intToSdfPath(int(value))))
            except (TypeError, ValueError):
                paths.append(str(value))
        if any(
            path == root or path.startswith(root + "/")
            for path in paths
            for root in roots
        ):
            count += 1
    return count


def _horizontal_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))


def _settle_motion(positions: Sequence[Sequence[float]], window: int = 30) -> float:
    rows = positions[-window:]
    if len(rows) < 2:
        return math.inf
    return max(
        math.dist([float(value) for value in rows[index - 1]], [float(value) for value in rows[index]])
        for index in range(1, len(rows))
    )


def _wait_for_stage(simulation_app: Any, usd_context: Any) -> None:
    for _ in range(6000):
        if not usd_context.is_stage_loading():
            return
        simulation_app.update()
    raise RuntimeError("simready_isaac_stage_load_timeout")


def _inventory(stage: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    from pxr import UsdGeom, UsdPhysics  # type: ignore

    if abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) > 1.0e-12:
        raise RuntimeError("simready_isaac_stage_units_invalid")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError("simready_isaac_stage_up_axis_invalid")
    replacement_path = str(spec["replacement_prim_path"])
    source_path = str(spec["source_target_collider_path"])
    support_path = str(spec["support_collider_path"])
    replacements = [
        prim for prim in stage.TraverseAll() if str(prim.GetPath()) == replacement_path
    ]
    source = stage.GetPrimAtPath(source_path)
    support = stage.GetPrimAtPath(support_path)
    replacement = stage.GetPrimAtPath(replacement_path)
    colliders = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if (
            str(prim.GetPath()).startswith(replacement_path + "/")
            or str(prim.GetPath()) == support_path
        )
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if (
        len(replacements) != 1
        or not replacement.IsValid()
        or not replacement.HasAPI(UsdPhysics.RigidBodyAPI)
        or not support.IsValid()
        or not support.HasAPI(UsdPhysics.CollisionAPI)
        or (source.IsValid() and source.IsActive())
        or support_path not in colliders
        or not any(path.startswith(replacement_path + "/") for path in colliders)
    ):
        raise RuntimeError("simready_isaac_scene_inventory_invalid")
    return {
        "replacement_count": len(replacements),
        "replacement_path": replacement_path,
        "source_target_collider_active": bool(source.IsValid() and source.IsActive()),
        "support_collider_path": support_path,
        "colliders": sorted(colliders),
    }


def _set_kinematic_translation(stage: Any, path: str, xyz: Sequence[float]) -> None:
    from pxr import Gf  # type: ignore

    attribute = stage.GetPrimAtPath(path).GetAttribute("xformOp:translate")
    if not attribute.IsValid():
        raise RuntimeError("simready_isaac_gripper_translation_missing")
    attribute.Set(Gf.Vec3d(*[float(value) for value in xyz]))


def _run_probe(
    *,
    simulation_app: Any,
    stage_path: Path,
    probe_name: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    import omni.physx as omni_physx  # type: ignore
    import omni.usd  # type: ignore
    from isaacsim.core.api import SimulationContext  # type: ignore
    from pxr import PhysicsSchemaTools  # type: ignore

    clear_instance = getattr(SimulationContext, "clear_instance", None)
    if callable(clear_instance):
        clear_instance()
    usd_context = omni.usd.get_context()
    if not usd_context.open_stage(str(stage_path)):
        raise RuntimeError("simready_isaac_stage_open_failed")
    _wait_for_stage(simulation_app, usd_context)
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("simready_isaac_stage_unavailable")
    inventory = _inventory(stage, spec)
    timestep = float(spec["fixed_step_seconds"])
    context = SimulationContext(
        physics_dt=timestep,
        rendering_dt=timestep,
        stage_units_in_meters=1.0,
    )
    physics_context = context.get_physics_context()
    for name, argument in (
        ("set_solver_type", "TGS"),
        ("set_broadphase_type", "SAP"),
        ("enable_gpu_dynamics", False),
        ("enable_enhanced_determinism", True),
    ):
        method = getattr(physics_context, name, None)
        if callable(method):
            method(argument)
    context.initialize_physics()
    context.play()
    replacement_path = str(spec["replacement_prim_path"])
    initial = _live_transform(omni_physx, replacement_path)
    positions: list[list[float]] = []
    contact_events = 0
    finger_contact_events = 0
    left_path = "/World/BlueprintProbeGripper/left_finger"
    right_path = "/World/BlueprintProbeGripper/right_finger"
    placement = initial["position"]
    for step in range(360):
        if probe_name == "gripper":
            phase = min(step, 239)
            if phase < 60:
                ratio = phase / 59.0
                half_gap = 0.056 - 0.021 * ratio
                lift = 0.0
            elif phase < 120:
                half_gap = 0.035
                lift = 0.0
            elif phase < 180:
                half_gap = 0.035
                lift = 0.03 * ((phase - 60) / 119.0)
            else:
                half_gap = 0.035 + 0.021 * ((phase - 180) / 59.0)
                lift = 0.03
            finger_z = float(spec["expected_support_z_m"]) + 0.084713995 + lift
            _set_kinematic_translation(
                stage,
                left_path,
                (placement[0] - half_gap, placement[1], finger_z),
            )
            _set_kinematic_translation(
                stage,
                right_path,
                (placement[0] + half_gap, placement[1], finger_z),
            )
        try:
            context.step(render=False)
        except TypeError:
            context.step()
        transform = _live_transform(omni_physx, replacement_path)
        positions.append(transform["position"])
        contact_events += _contact_count(
            omni_physx,
            PhysicsSchemaTools,
            (replacement_path, str(spec["support_collider_path"])),
        )
        if probe_name == "gripper":
            finger_contact_events += _contact_count(
                omni_physx,
                PhysicsSchemaTools,
                (left_path, right_path),
            )
    context.stop()
    final = _live_transform(omni_physx, replacement_path)
    result = {
        "probe": probe_name,
        "stage_sha256": _sha256(stage_path),
        "inventory": inventory,
        "initial_transform": initial,
        "final_transform": final,
        "contact_report_event_count": contact_events,
        "finger_contact_report_event_count": finger_contact_events,
        "horizontal_motion_m": _horizontal_distance(initial["position"], final["position"]),
        "minimum_z_m": min(row[2] for row in positions),
        "maximum_z_m": max(row[2] for row in positions),
        "settle_motion_m": _settle_motion(positions),
        "step_count": len(positions),
    }
    acceptance = spec["acceptance"][probe_name]
    support_z = float(spec["expected_support_z_m"])
    checks: dict[str, bool]
    if probe_name == "drop":
        checks = {
            "minimum_drop": initial["position"][2] - final["position"][2]
            >= float(acceptance["minimum_observed_drop_m"]),
            "contact": contact_events >= int(acceptance["minimum_contact_events"]),
            "support_height": abs(final["position"][2] - support_z)
            <= float(acceptance["maximum_support_height_error_m"]),
            "settled": result["settle_motion_m"]
            <= float(acceptance["maximum_settle_motion_m"]),
        }
    elif probe_name == "slide":
        checks = {
            "minimum_motion": result["horizontal_motion_m"]
            >= float(acceptance["minimum_horizontal_motion_m"]),
            "bounded_motion": result["horizontal_motion_m"]
            <= float(acceptance["maximum_horizontal_motion_m"]),
            "support_height": abs(final["position"][2] - support_z)
            <= float(acceptance["maximum_support_height_error_m"]),
        }
    elif probe_name == "tip":
        checks = {
            "perturbation_authored": float(acceptance["minimum_perturbation_degrees"])
            <= 6.0,
            "center_drop_bounded": initial["position"][2] - result["minimum_z_m"]
            <= float(acceptance["maximum_center_drop_m"]),
            "support_height": abs(final["position"][2] - support_z)
            <= float(acceptance["maximum_support_height_error_m"]),
        }
    else:
        lift = result["maximum_z_m"] - initial["position"][2]
        checks = {
            "finger_contact": finger_contact_events
            >= int(acceptance["minimum_finger_contact_events"]),
            "lift": lift >= float(acceptance["minimum_lift_m"]),
            "release": abs(final["position"][2] - support_z) <= 0.012,
        }
        result["observed_lift_m"] = lift
    result["checks"] = checks
    result["passed"] = all(checks.values())
    result["trace_digest"] = _canonical_digest(result, field="trace_digest")
    if callable(clear_instance):
        clear_instance()
    return result


def run(spec_path: Path, output_path: Path) -> dict[str, Any]:
    spec_source = spec_path.expanduser().resolve()
    spec = _read_json(spec_source)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [],
        "isaac_sim_version": ISAAC_SIM_VERSION,
        "probe_spec_sha256": _sha256(spec_source),
        "probe_results": [],
        "source_target_collider_active": None,
        "replacement_count": None,
        "native_isaac_executed": False,
        "physical_success_established": False,
        "robot_task_success_established": False,
        "provider_zero_required_after_return": True,
        "claim_ceiling": "native_isaac_exact_scene_simulation_only",
    }
    simulation_app: Any | None = None
    try:
        if (
            spec.get("schema_version") != "adp009b_simready_isaac_probe_spec.v1"
            or spec.get("status") != "frozen_before_execution"
            or spec.get("isaac_sim_version") != ISAAC_SIM_VERSION
        ):
            raise ValueError("simready_isaac_probe_spec_invalid")
        stages = spec.get("stages")
        if not isinstance(stages, Mapping) or set(stages) != {"drop", "slide", "tip", "gripper"}:
            raise ValueError("simready_isaac_probe_stage_set_invalid")
        stage_paths: dict[str, Path] = {}
        for name, record in stages.items():
            if not isinstance(record, Mapping):
                raise ValueError("simready_isaac_probe_stage_record_invalid")
            path = (spec_source.parent / str(record.get("relative_path") or "")).resolve()
            if spec_source.parent not in path.parents or not path.is_file():
                raise ValueError("simready_isaac_probe_stage_path_invalid")
            if _sha256(path) != record.get("sha256"):
                raise ValueError("simready_isaac_probe_stage_digest_mismatch")
            stage_paths[str(name)] = path
        SimulationApp = _import_simulation_app()
        simulation_app = SimulationApp({"headless": True, "fast_shutdown": True})
        from isaacsim.core.version import get_version  # type: ignore

        observed_version = str(get_version()[0])
        result["observed_isaac_sim_version"] = observed_version
        if observed_version != ISAAC_SIM_VERSION:
            raise RuntimeError("simready_isaac_runtime_version_mismatch")
        result["native_isaac_executed"] = True
        probe_results = [
            _run_probe(
                simulation_app=simulation_app,
                stage_path=stage_paths[name],
                probe_name=name,
                spec=spec,
            )
            for name in ("drop", "slide", "tip", "gripper")
        ]
        result["probe_results"] = probe_results
        inventories = [row["inventory"] for row in probe_results]
        result["source_target_collider_active"] = any(
            row["source_target_collider_active"] for row in inventories
        )
        result["replacement_count"] = max(row["replacement_count"] for row in inventories)
        failed = [row["probe"] for row in probe_results if row["passed"] is not True]
        result["blockers"] = [f"simready_isaac_{name}_probe_failed" for name in failed]
        result["status"] = "completed" if not failed else "blocked"
    except Exception as exc:
        result["blockers"] = [f"simready_isaac_execution_failed:{type(exc).__name__}"]
        result["exception"] = repr(exc)[:1000]
        result["status"] = "blocked"
    finally:
        result["result_written_before_simulation_app_close"] = True
        result["result_digest"] = _canonical_digest(result, field="result_digest")
        _persist(output_path, result)
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception as exc:
                result["shutdown_warning"] = repr(exc)[:500]
        result["result_digest"] = _canonical_digest(result, field="result_digest")
        _persist(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.spec, args.output)
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
