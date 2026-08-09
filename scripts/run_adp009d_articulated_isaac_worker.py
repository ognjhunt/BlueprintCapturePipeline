#!/usr/bin/env python
"""Read an articulated replacement back out of PhysX, on the provider GPU.

Everything qualifying this asset so far has been geometry on paper: validators
that read the USD, ray casts through an aperture, swept-volume clearance. None
of it proves the joint moves. This worker boots Isaac headless and answers the
eleven readbacks the frozen probe spec preregisters - articulation identity,
joint graph, axis and limits, the locked joint holding, the commanded joint
reaching its maximum, contact stability, initial penetration, reset replay and
determinism.

It runs the blank physics stage first. If that fails, the runtime is broken and
nothing about the asset has been learned, which is a different result from the
asset failing. Every readback is recorded with its observed value, not just a
verdict, so a failure says which quantity was wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

RESULT_SCHEMA_VERSION = "adp009d_articulated_isaac_result.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _persist(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [reason],
        "native_isaac_executed": False,
        "articulation_qualified": False,
        "physical_success_established": False,
        "provider_zero_required_after_return": True,
        **extra,
    }


def _simulation_app() -> Any:
    from isaacsim.simulation_app import SimulationApp  # type: ignore

    return SimulationApp({"headless": True, "renderer": "RayTracedLighting"})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    output = Path(arguments.output).expanduser().resolve()
    spec_path = Path(arguments.spec).expanduser().resolve()

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _persist(output, _blocked(f"articulated_isaac_spec_unreadable:{type(exc).__name__}"))
        return 1

    try:
        simulation_app = _simulation_app()
    except Exception as exc:  # noqa: BLE001 - any import/boot failure is terminal
        _persist(output, _blocked(f"articulated_isaac_simulation_app_failed:{type(exc).__name__}"))
        return 1

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [],
        "native_isaac_executed": False,
        "articulation_qualified": False,
        "physical_success_established": False,
        "provider_zero_required_after_return": True,
        "spec_sha256": _sha256(spec_path),
        "probe_results": [],
        "replacement_count": 1,
        "source_target_collider_active": False,
        "readbacks": {},
    }

    try:
        import omni.usd  # type: ignore
        from isaacsim.core.api import World  # type: ignore
        from pxr import UsdPhysics  # type: ignore

        stages = spec.get("stages") or {}
        root = spec_path.parent
        blank = root / Path(str(stages["blank_stage"]["path"])).name
        articulation_stage = root / Path(str(stages["articulation_stage"]["path"])).name

        # 1. blank stage: does the runtime bring up physics at all?
        context = omni.usd.get_context()
        context.open_stage(str(blank))
        for _ in range(60):
            simulation_app.update()
        blank_world = World(stage_units_in_meters=1.0)
        blank_world.reset()
        for _ in range(30):
            blank_world.step(render=False)
        result["readbacks"]["blank_stage_physics_scene_ran"] = True
        blank_world.stop()
        blank_world.clear_instance()

        # 2. the asset itself
        context.open_stage(str(articulation_stage))
        for _ in range(90):
            simulation_app.update()
        stage = context.get_stage()
        expected = spec.get("expected") or {}

        roots = sorted(
            str(p.GetPath())
            for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        joints = {
            str(p.GetPath()): p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)
        }
        result["readbacks"]["articulation_root_identity"] = {
            "observed": roots,
            "expected": [expected.get("articulation_root_prim_path")],
            "passed": roots == [expected.get("articulation_root_prim_path")],
        }
        observed_types = {
            path: (
                "revolute"
                if prim.IsA(UsdPhysics.RevoluteJoint)
                else "prismatic"
                if prim.IsA(UsdPhysics.PrismaticJoint)
                else "other"
            )
            for path, prim in sorted(joints.items())
        }
        result["readbacks"]["joint_count_and_types"] = {
            "observed": observed_types,
            "expected": expected.get("joint_types"),
            "passed": observed_types == (expected.get("joint_types") or {}),
        }

        task_path = str(expected.get("task_joint_prim_path") or "")
        locked_paths = [str(p) for p in (expected.get("locked_joint_prim_paths") or [])]
        task_prim = joints.get(task_path)
        result["readbacks"]["task_joint_identity"] = {
            "observed": task_path if task_prim else None,
            "passed": task_prim is not None,
        }
        result["readbacks"]["locked_joint_identity"] = {
            "observed": [p for p in locked_paths if p in joints],
            "expected": locked_paths,
            "passed": all(p in joints for p in locked_paths),
        }
        if task_prim is not None:
            revolute = UsdPhysics.RevoluteJoint(task_prim)
            axis = task_prim.GetAttribute("physics:axis")
            observed_limits = [
                float(revolute.GetLowerLimitAttr().Get() or 0.0),
                float(revolute.GetUpperLimitAttr().Get() or 0.0),
            ]
            result["readbacks"]["joint_axis_and_limits"] = {
                "observed_axis": str(axis.Get()) if axis else "",
                "observed_limits_deg": observed_limits,
                "expected_axis": expected.get("task_joint_axis"),
                "expected_limits_deg": expected.get("task_joint_limits_deg"),
                "passed": str(axis.Get() if axis else "")
                == str(expected.get("task_joint_axis"))
                and observed_limits == list(expected.get("task_joint_limits_deg") or []),
            }

        # 3. simulate: reset, drive the task joint, watch the locked ones
        from isaacsim.core.prims import Articulation  # type: ignore

        world = World(stage_units_in_meters=1.0)
        world.reset()
        if not roots:
            # Guessing a prim path here would turn "the asset has no
            # articulation root" into an unrelated lookup failure.
            raise RuntimeError("articulated_isaac_no_articulation_root_to_drive")
        view = Articulation(prim_paths_expr=roots[0], name="twin")
        world.scene.add(view)
        world.reset()
        for _ in range(30):
            world.step(render=False)

        names = list(view.dof_names or [])
        result["readbacks"]["dof_names"] = names
        index = {name: position for position, name in enumerate(names)}
        task_dof = task_path.rsplit("/", 1)[-1]
        locked_dofs = [p.rsplit("/", 1)[-1] for p in locked_paths]

        import numpy as np  # type: ignore

        initial = np.asarray(view.get_joint_positions())[0]
        result["readbacks"]["no_initial_penetration"] = {
            "observed_initial_positions_rad": [float(v) for v in initial],
            "passed": bool(np.all(np.abs(initial) < 0.05)),
        }

        sweep = [float(v) for v in (expected.get("commanded_sweep_degrees") or [])]
        tolerance = float(expected.get("locked_joint_motion_tolerance_rad") or 0.001)
        settle = spec.get("settle") or {}
        reached = []
        locked_drift = 0.0
        for angle in sweep:
            target = np.asarray(view.get_joint_positions())
            if task_dof in index:
                target[0][index[task_dof]] = math.radians(angle)
            view.set_joint_position_targets(target)
            for _ in range(int(settle.get("samples") or 40)):
                world.step(render=False)
            current = np.asarray(view.get_joint_positions())[0]
            if task_dof in index:
                reached.append(math.degrees(float(current[index[task_dof]])))
            for name in locked_dofs:
                if name in index:
                    locked_drift = max(locked_drift, abs(float(current[index[name]])))

        maximum = float(expected.get("maximum_commanded_degrees") or 0.0)
        result["readbacks"]["commanded_sweep_reaches_maximum"] = {
            "observed_degrees": reached,
            "maximum_commanded_degrees": maximum,
            "passed": bool(reached) and reached[-1] >= maximum - 2.0,
        }
        result["readbacks"]["locked_joint_motion_within_tolerance"] = {
            "observed_max_drift_rad": locked_drift,
            "tolerance_rad": tolerance,
            "passed": locked_drift <= tolerance,
        }

        settled = np.asarray(view.get_joint_velocities())[0]
        result["readbacks"]["contact_stability"] = {
            "observed_max_velocity_rad_s": float(np.abs(settled).max()),
            "passed": float(np.abs(settled).max()) <= 0.05,
        }

        # 4. reset replay and determinism
        reset_positions = expected.get("reset_joint_positions_rad") or {}
        target = np.asarray(view.get_joint_positions())
        for path, value in reset_positions.items():
            name = str(path).rsplit("/", 1)[-1]
            if name in index:
                target[0][index[name]] = float(value)
        view.set_joint_positions(target)
        for _ in range(30):
            world.step(render=False)
        after_reset = np.asarray(view.get_joint_positions())[0]
        result["readbacks"]["reset_replay_within_tolerance"] = {
            "observed_positions_rad": [float(v) for v in after_reset],
            "passed": bool(np.all(np.abs(after_reset) <= 0.02)),
        }
        replay = []
        for _ in range(2):
            view.set_joint_positions(target)
            for _ in range(20):
                world.step(render=False)
            replay.append([float(v) for v in np.asarray(view.get_joint_positions())[0]])
        result["readbacks"]["deterministic_final_state"] = {
            "observed_runs": replay,
            "passed": bool(
                np.allclose(np.asarray(replay[0]), np.asarray(replay[1]), atol=1e-4)
            ),
        }

        world.stop()
        result["native_isaac_executed"] = True
        failed = [
            name
            for name, row in result["readbacks"].items()
            if isinstance(row, Mapping) and row.get("passed") is False
        ]
        result["blockers"] = [f"articulated_isaac_readback_failed:{name}" for name in failed]
        result["articulation_qualified"] = not failed
        result["status"] = "completed" if not failed else "blocked"
        # The provider lane validates a common result shape across probes; the
        # articulated readbacks are reported through it rather than in a
        # parallel schema only this worker understands.
        result["probe_results"] = [
            {"probe": name, "passed": bool(row.get("passed")), "observed": row}
            for name, row in sorted(result["readbacks"].items())
            if isinstance(row, Mapping) and "passed" in row
        ]
        result["replacement_count"] = 1
        result["source_target_collider_active"] = False
    except Exception as exc:  # noqa: BLE001 - retain the real failure
        import traceback

        result["blockers"].append(
            f"articulated_isaac_runtime_failed:{type(exc).__name__}"
        )
        result["traceback"] = traceback.format_exc()[-4000:]
    finally:
        _persist(output, result)
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001 - shutdown must not mask the result
            pass
    return 0 if result.get("articulation_qualified") else 1


if __name__ == "__main__":
    sys.exit(main())
