#!/usr/bin/env python3
"""Isolated, headless ovphysx smoke-test worker.

Run this only in a pinned external environment. The worker performs real USD
ingest and fixed-step PhysX execution, then emits a Blueprint-normalizable JSON
report. It never claims Isaac parity or task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any


def _sha256_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", "details": details}


def _gpu_diagnostic() -> dict[str, Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "name": str(pynvml.nvmlDeviceGetName(handle)),
            "uuid": str(pynvml.nvmlDeviceGetUUID(handle)),
            "memory_total_bytes": int(memory.total),
            "memory_used_bytes": int(memory.used),
        }
    except Exception as exc:  # noqa: BLE001 - optional diagnostic boundary
        return {"query_error": type(exc).__name__}


def _scene_inventory(config: dict[str, Any], input_path: Path) -> dict[str, Any]:
    """Read the digest-bound inventory derived by the OpenUSD environment.

    ``ovphysx`` and ``usd-exchange`` each ship USD libraries. Importing both in
    one process is unsupported, so native dynamics ingests the USD through
    ``PhysX.add_usd`` while schema inspection is bound into the frozen config.
    """

    inventory = config.get("usd_scene_inventory")
    if not isinstance(inventory, dict):
        raise ValueError("usd_scene_inventory is required")
    required_lists = ("rigid_bodies", "colliders", "joints", "masses", "materials")
    if any(not isinstance(inventory.get(name), list) for name in required_lists):
        raise ValueError("usd_scene_inventory is malformed")
    if not str(inventory.get("source_sha256") or "").startswith("sha256:"):
        raise ValueError("usd_scene_inventory source digest is missing")
    if inventory["source_sha256"] != _sha256_file(input_path):
        raise ValueError("usd_scene_inventory source digest changed")
    return {
        **inventory,
        "expected_joint_min_count": int(config.get("expected_joint_min_count", 0)),
    }


def _values_in_bounds(values: list[dict[str, Any]], field: str, low: float, high: float) -> bool:
    found = [item.get(field) for item in values if item.get(field) is not None]
    return bool(found) and all(low <= float(value) <= high for value in found)


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    import numpy as np
    import ovphysx
    from ovphysx import PhysX
    from ovphysx.types import TensorType

    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    inventory = _scene_inventory(config, args.input)
    checks: list[dict[str, Any]] = [
        _check("usd_scene_load", True, inventory=inventory),
    ]
    device = str(config.get("device", "cpu"))
    gpu_samples = [_gpu_diagnostic()]
    physx = PhysX(device=device)
    bindings: list[Any] = []
    contact_binding = None
    snapshots: list[dict[str, Any]] = []
    pose_history: list[list[Any]] = []
    try:
        usd_handle, _ = physx.add_usd(str(args.input))
        physx.wait_all()
        gpu_samples.append(_gpu_diagnostic())
        patterns = [str(value) for value in config.get("rigid_body_patterns", [])]
        if not patterns:
            raise ValueError("rigid_body_patterns must identify at least one dynamic body")
        for pattern in patterns:
            binding = physx.create_tensor_binding(
                pattern=pattern, tensor_type=TensorType.RIGID_BODY_POSE
            )
            bindings.append(binding)
        initial_arrays: list[Any] = []
        for binding in bindings:
            array = np.zeros(binding.shape, dtype=np.float32)
            binding.read(array)
            initial_arrays.append(array)

        sensor_patterns = [str(value) for value in config.get("penetration_sensor_patterns", [])]
        filter_patterns = [str(value) for value in config.get("penetration_filter_patterns", [])]
        if sensor_patterns:
            kwargs: dict[str, Any] = {"sensor_patterns": sensor_patterns}
            if filter_patterns:
                kwargs.update(
                    {
                        "filter_patterns": filter_patterns,
                        "filters_per_sensor": int(
                            config.get("filters_per_sensor", len(filter_patterns))
                        ),
                    }
                )
            contact_binding = physx.create_contact_binding(**kwargs)

        dt = float(config.get("fixed_step_seconds", 1.0 / 60.0))
        steps = int(config.get("steps", 60))
        snapshot_steps = set(int(value) for value in config.get("snapshot_steps", [0, steps - 1]))
        initial_contact_norm = 0.0
        maximum_contact_norm = 0.0
        contact_step_count = 0
        for step_index in range(steps):
            physx.step(dt, step_index * dt)
            physx.wait_all()
            if contact_binding is not None:
                forces = np.zeros((contact_binding.sensor_count, 3), dtype=np.float32)
                contact_binding.read_net_forces(forces)
                contact_norm = (
                    float(np.max(np.linalg.norm(forces, axis=-1))) if forces.size else 0.0
                )
                if step_index == 0:
                    initial_contact_norm = contact_norm
                maximum_contact_norm = max(maximum_contact_norm, contact_norm)
                if contact_norm > float(config.get("contact_force_threshold", 1.0e-3)):
                    contact_step_count += 1
            history_state: list[Any] = []
            for binding in bindings:
                array = np.zeros(binding.shape, dtype=np.float32)
                binding.read(array)
                history_state.append(array.copy())
            pose_history.append(history_state)
            if step_index in snapshot_steps:
                state = [array.round(decimals=7).tolist() for array in history_state]
                snapshots.append(
                    {"step": step_index, "time_seconds": step_index * dt, "rigid_body_poses": state}
                )
        gpu_samples.append(_gpu_diagnostic())

        final_arrays: list[Any] = []
        finite = True
        changed = False
        for binding, initial in zip(bindings, initial_arrays):
            array = np.zeros(binding.shape, dtype=np.float32)
            binding.read(array)
            final_arrays.append(array)
            finite = finite and bool(np.all(np.isfinite(array)))
            changed = changed or bool(not np.allclose(array, initial, atol=1.0e-7, rtol=0.0))
        checks.append(
            _check(
                "gravity_and_rigid_body_integration",
                finite and changed,
                finite=finite,
                state_changed=changed,
            )
        )

        max_initial_force = float(config.get("maximum_initial_contact_force", 1.0e-3))
        collider_ok = bool(inventory["colliders"])
        penetration_measured = contact_binding is not None
        checks.append(
            _check(
                "collider_presence_and_penetration",
                collider_ok and penetration_measured and initial_contact_norm <= max_initial_force,
                collider_count=len(inventory["colliders"]),
                penetration_measured=penetration_measured,
                initial_contact_force_norm=initial_contact_norm,
                maximum_initial_contact_force=max_initial_force,
            )
        )

        settle = config.get("drop_contact_settle")
        if settle is not None:
            if not isinstance(settle, dict) or len(bindings) != 1:
                raise ValueError(
                    "drop_contact_settle requires one rigid body binding and an object config"
                )
            if not pose_history or not pose_history[0] or not pose_history[0][0].size:
                raise ValueError("drop_contact_settle pose history is empty")
            rows = [np.asarray(sample[0], dtype=np.float64).reshape(-1, 7) for sample in pose_history]
            if any(row.shape[0] != 1 for row in rows):
                raise ValueError("drop_contact_settle pattern must resolve exactly one rigid body")
            positions = np.stack([row[0, :3] for row in rows], axis=0)
            quaternions = np.stack([row[0, 3:7] for row in rows], axis=0)
            settle_window = int(settle.get("settle_window_steps", 30))
            if settle_window < 2 or settle_window > len(rows):
                raise ValueError("drop_contact_settle settle_window_steps is invalid")
            expected_support_z = float(settle["expected_support_z_m"])
            initial_drop_height = float(settle["initial_drop_height_m"])
            support_tolerance = float(settle.get("support_height_tolerance_m", 0.005))
            maximum_settle_motion = float(settle.get("maximum_settle_motion_m", 0.002))
            maximum_rotation_degrees = float(
                settle.get("maximum_rotation_from_initial_degrees", 5.0)
            )
            minimum_drop = float(
                settle.get("minimum_observed_drop_m", initial_drop_height * 0.5)
            )
            final_position = positions[-1]
            window_positions = positions[-settle_window:]
            maximum_window_motion = float(
                np.max(np.linalg.norm(window_positions - final_position, axis=1))
            )
            initial_quaternion = quaternions[0]
            final_quaternion = quaternions[-1]
            initial_quaternion /= max(float(np.linalg.norm(initial_quaternion)), 1.0e-12)
            final_quaternion /= max(float(np.linalg.norm(final_quaternion)), 1.0e-12)
            quaternion_dot = float(
                np.clip(abs(np.dot(initial_quaternion, final_quaternion)), 0.0, 1.0)
            )
            rotation_degrees = float(np.degrees(2.0 * np.arccos(quaternion_dot)))
            observed_drop = float(positions[0, 2] - np.min(positions[:, 2]))
            final_support_error = abs(float(final_position[2]) - expected_support_z)
            required_contact_steps = int(settle.get("minimum_contact_steps", 1))
            checks.extend(
                [
                    _check(
                        "drop_height_observed",
                        observed_drop >= minimum_drop,
                        observed_drop_m=observed_drop,
                        minimum_observed_drop_m=minimum_drop,
                        configured_initial_drop_height_m=initial_drop_height,
                    ),
                    _check(
                        "support_contact_observed",
                        contact_binding is not None
                        and contact_step_count >= required_contact_steps,
                        maximum_contact_force_norm=maximum_contact_norm,
                        contact_step_count=contact_step_count,
                        minimum_contact_steps=required_contact_steps,
                    ),
                    _check(
                        "settled_on_expected_support",
                        final_support_error <= support_tolerance
                        and maximum_window_motion <= maximum_settle_motion,
                        final_position_m=final_position.round(decimals=7).tolist(),
                        expected_support_z_m=expected_support_z,
                        final_support_error_m=final_support_error,
                        support_height_tolerance_m=support_tolerance,
                        settle_window_steps=settle_window,
                        maximum_settle_window_motion_m=maximum_window_motion,
                        maximum_allowed_settle_motion_m=maximum_settle_motion,
                    ),
                    _check(
                        "upright_after_settle",
                        rotation_degrees <= maximum_rotation_degrees,
                        rotation_from_initial_degrees=rotation_degrees,
                        maximum_rotation_from_initial_degrees=maximum_rotation_degrees,
                    ),
                ]
            )

        expected_joints = inventory["expected_joint_min_count"]
        limits_valid = all(
            item["lower"] is None
            or item["upper"] is None
            or float(item["lower"]) <= float(item["upper"])
            for item in inventory["joints"]
        )
        checks.append(
            _check(
                "joint_and_limit_inspection",
                len(inventory["joints"]) >= expected_joints and limits_valid,
                joint_count=len(inventory["joints"]),
                expected_minimum=expected_joints,
                limits_valid=limits_valid,
            )
        )

        mass_bounds = config.get("mass_bounds_kg", [1.0e-6, 1.0e6])
        friction_bounds = config.get("friction_bounds", [0.0, 10.0])
        mass_ok = _values_in_bounds(
            inventory["masses"], "mass", float(mass_bounds[0]), float(mass_bounds[1])
        ) or _values_in_bounds(
            inventory["masses"], "density", float(mass_bounds[0]), float(mass_bounds[1])
        )
        friction_ok = _values_in_bounds(
            inventory["materials"],
            "static_friction",
            float(friction_bounds[0]),
            float(friction_bounds[1]),
        ) and _values_in_bounds(
            inventory["materials"],
            "dynamic_friction",
            float(friction_bounds[0]),
            float(friction_bounds[1]),
        )
        checks.append(
            _check(
                "mass_and_friction_bounds",
                mass_ok and friction_ok,
                mass_ok=mass_ok,
                friction_ok=friction_ok,
            )
        )
        checks.append(
            _check(
                "fixed_step_state_snapshot", bool(snapshots), snapshot_count=len(snapshots), dt=dt
            )
        )
        if config.get("articulation_prim_path"):
            articulation_changed = changed and len(inventory["joints"]) > 0
            checks.append(_check("simple_articulation_motion", articulation_changed))

        snapshot_path = args.output_dir / "state_snapshots.json"
        _write_json(
            snapshot_path, {"schema_version": "ovphysx_state_snapshots.v1", "snapshots": snapshots}
        )
        outputs = [
            {
                "kind": "state_snapshots",
                "path": snapshot_path.name,
                "metadata": {
                    "fixed_step_seconds": dt,
                    "steps": steps,
                    "snapshot_count": len(snapshots),
                },
            }
        ]
        physx.remove_usd(usd_handle)
    finally:
        if contact_binding is not None:
            contact_binding.destroy()
        for binding in bindings:
            binding.destroy()
        physx.release()

    passed = all(item["status"] == "passed" for item in checks)
    runtime = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_version": os.environ.get("CUDA_VERSION"),
        "driver_version": os.environ.get("NVIDIA_DRIVER_VERSION"),
        "gpu_identity": {"requested_device": device, **gpu_samples[-1]},
        "library_versions": {"ovphysx": ovphysx.__version__, "numpy": np.__version__},
    }
    return (
        {
            "component_name": "ovphysx",
            "component_version": ovphysx.__version__,
            "source_revision": args.source_revision,
            "configuration_sha256": _sha256_json(config),
            "runtime": runtime,
            "checks": checks,
            "outputs": outputs,
            "metrics": {
                "worker_wall_seconds": time.monotonic() - started,
                "mode": args.mode,
                "device": device,
                "gpu_memory_baseline_bytes": next(
                    (
                        int(row["memory_used_bytes"])
                        for row in gpu_samples
                        if row.get("memory_used_bytes") is not None
                    ),
                    None,
                ),
                "gpu_memory_peak_observed_bytes": max(
                    (
                        int(row["memory_used_bytes"])
                        for row in gpu_samples
                        if row.get("memory_used_bytes") is not None
                    ),
                    default=None,
                ),
            },
            "failure_classes_checked": [
                "usd_physics_scene_load",
                "nonfinite_or_static_rigid_body_state",
                "gross_initial_penetration",
                "invalid_joint_limits",
                "missing_or_out_of_bounds_physical_properties",
            ],
        },
        0 if passed else 2,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("cold", "warm"), required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    try:
        report, status = _run(args)
    except Exception as exc:  # noqa: BLE001 - preserve machine-readable failure evidence
        config = json.loads(args.config.read_text(encoding="utf-8"))
        report = {
            "component_name": "ovphysx",
            "component_version": "unavailable",
            "source_revision": args.source_revision,
            "configuration_sha256": _sha256_json(config),
            "runtime": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            "checks": [
                {
                    "name": "worker_execution",
                    "status": "failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
            "outputs": [],
            "failure_classes_checked": [],
        }
        status = 2
    _write_json(args.output, report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
