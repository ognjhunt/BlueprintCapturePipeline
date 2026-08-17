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
COMMANDED_ARTICULATION_MODE = "commanded_articulation"
LOCKED_HINGE_RIGID_MODE = "locked_hinge_rigid_validation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any], *, field: str) -> str:
    """Match blueprint_pipeline.decision_evidence_contracts.canonical_digest.

    The lane refuses a result that does not attest to itself, so a hand-edited
    result is detectable. This worker runs inside Isaac's interpreter with no
    access to the package, so the definition is mirrored rather than imported -
    it must stay byte-compatible with the in-repo one.
    """

    normalized = dict(value)
    normalized.pop(field, None)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _persist(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload["result_digest"] = _canonical_digest(payload, field="result_digest")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _command_schedule(spec: Mapping[str, Any]) -> list[float]:
    """Return the only joint-command schedule this frozen spec permits."""

    mode = str(spec.get("validation_mode") or COMMANDED_ARTICULATION_MODE)
    expected = spec.get("expected") or {}
    sweep = [float(value) for value in expected.get("commanded_sweep_degrees") or []]
    if mode == LOCKED_HINGE_RIGID_MODE:
        if expected.get("task_joint_prim_path") or sweep or spec.get("probe_drive"):
            raise ValueError("locked_hinge_probe_contains_joint_command")
        return []
    if mode != COMMANDED_ARTICULATION_MODE:
        raise ValueError("articulated_isaac_validation_mode_invalid")
    if not expected.get("task_joint_prim_path") or not sweep:
        raise ValueError("commanded_articulation_schedule_missing")
    return sweep


def _simulation_app() -> Any:
    from isaacsim.simulation_app import SimulationApp  # type: ignore

    return SimulationApp({"headless": True, "renderer": "RayTracedLighting"})


def _authored_articulation_joints(
    stage: Any, articulation_root_path: str, usd_physics: Any
) -> dict[str, Any]:
    """Return only joints in the candidate's authored articulation subtree.

    The frozen runtime overlay may add one world FixedJoint to anchor a dynamic
    physics representation.  That anchor is intentionally outside the asset
    root; counting it as candidate topology would either hide a topology drift
    or make the temporary physics representation look like authored geometry.
    """

    prefix = articulation_root_path.rstrip("/") + "/"
    return {
        str(prim.GetPath()): prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(prefix) and prim.IsA(usd_physics.Joint)
    }


def _physx_homogeneous_articulation_readback(
    stage: Any,
    representation: Mapping[str, Any],
    usd_physics: Any,
) -> dict[str, Any]:
    """Verify the frozen dynamic-base/world-anchor runtime representation.

    PhysX tensor articulations cannot contain kinematic bodies.  The candidate
    bytes remain immutable and retain their truthful kinematic fixed-base
    authoring; the stage opened by this worker must instead show a digest-bound
    dynamic override plus a world FixedJoint.  This deliberately rejects a
    missing override or an arbitrary second anchor before creating an Isaac
    tensor view, which would otherwise collapse to an unhelpful ``NoneType``
    backend exception.
    """

    expected_schema = "adp009d_physx_articulation_representation.v1"
    body_paths = representation.get("articulation_body_prim_paths")
    authored_kinematic = representation.get(
        "authored_kinematic_articulation_body_prim_paths"
    )
    overrides = representation.get("runtime_dynamic_override_body_prim_paths")
    mode = str(representation.get("mode") or "")
    fixed_base = representation.get("fixed_base_body_prim_path")
    anchor_path = representation.get("fixed_base_anchor_prim_path")
    schema_valid = (
        representation.get("schema_version") == expected_schema
        and representation.get("candidate_bytes_modified") is False
        and representation.get("candidate_structural_topology_preserved") is True
        and representation.get("physx_homogeneous_articulation_required") is True
        and isinstance(body_paths, list)
        and isinstance(authored_kinematic, list)
        and isinstance(overrides, list)
        and all(isinstance(path, str) and path.startswith("/") for path in body_paths)
        and all(
            isinstance(path, str) and path.startswith("/")
            for path in authored_kinematic
        )
        and all(isinstance(path, str) and path.startswith("/") for path in overrides)
        and len(set(body_paths)) == len(body_paths)
        and len(set(authored_kinematic)) == len(authored_kinematic)
        and len(set(overrides)) == len(overrides)
        and set(authored_kinematic).issubset(body_paths)
        and overrides == authored_kinematic
    )
    observed_kinematic: dict[str, bool | None] = {}
    authored_values: dict[str, bool] = {}
    override_authored: dict[str, bool] = {}
    for body_path in body_paths if isinstance(body_paths, list) else []:
        prim = stage.GetPrimAtPath(body_path)
        if not prim.IsValid() or not prim.HasAPI(usd_physics.RigidBodyAPI):
            observed_kinematic[body_path] = None
            continue
        attribute = usd_physics.RigidBodyAPI(prim).GetKinematicEnabledAttr()
        observed_kinematic[body_path] = bool(attribute.Get())
        authored_values[body_path] = bool(attribute.HasAuthoredValue())
        if body_path in (overrides if isinstance(overrides, list) else []):
            override_authored[body_path] = bool(attribute.HasAuthoredValue())

    all_dynamic = (
        bool(body_paths)
        and all(observed_kinematic.get(path) is False for path in body_paths)
    )
    anchor_observed: dict[str, Any] | None = None
    if authored_kinematic:
        anchor = stage.GetPrimAtPath(str(anchor_path or ""))
        is_fixed_joint = bool(anchor.IsValid() and anchor.IsA(usd_physics.FixedJoint))
        body0 = []
        body1: list[str] = []
        if is_fixed_joint:
            fixed_joint = usd_physics.FixedJoint(anchor)
            body0 = [str(path) for path in fixed_joint.GetBody0Rel().GetTargets()]
            body1 = [str(path) for path in fixed_joint.GetBody1Rel().GetTargets()]
        anchor_observed = {
            "path": str(anchor_path or ""),
            "is_fixed_joint": is_fixed_joint,
            "body0": body0,
            "body1": body1,
        }
        anchor_valid = (
            mode == "dynamic_articulation_with_world_fixed_base_anchor"
            and isinstance(fixed_base, str)
            and fixed_base in authored_kinematic
            and str(anchor_path) == "/BlueprintProbeRuntime/fixed_base_anchor"
            and body0 == []
            and body1 == [fixed_base]
        )
    else:
        anchor_valid = (
            mode == "candidate_authored_dynamic_articulation"
            and fixed_base is None
            and anchor_path is None
            and overrides == []
        )

    passed = (
        schema_valid
        and all_dynamic
        and anchor_valid
        and all(
            override_authored.get(path) is True
            for path in overrides if isinstance(overrides, list)
        )
    )
    return {
        "expected_authored_kinematic_body_prim_paths": authored_kinematic,
        "expected_runtime_dynamic_override_body_prim_paths": overrides,
        "observed_runtime_kinematic_enabled": observed_kinematic,
        "observed_runtime_kinematic_attribute_authored": authored_values,
        "world_fixed_base_anchor": anchor_observed,
        "passed": passed,
        "claim_boundary": {
            "candidate_bytes_modified": False,
            "runtime_overlay_is_not_authored_asset_topology": True,
            "kinematic_articulation_is_not_accepted_as_physx_valid": True,
        },
    }


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
        validation_mode = str(
            spec.get("validation_mode") or COMMANDED_ARTICULATION_MODE
        )
        sweep = _command_schedule(spec)
        result["validation_mode"] = validation_mode
        result["joint_command_issued"] = bool(sweep)

        roots = sorted(
            str(p.GetPath())
            for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        joints = (
            _authored_articulation_joints(stage, roots[0], UsdPhysics)
            if len(roots) == 1
            else {}
        )
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
        representation = spec.get("runtime_representation")
        if not isinstance(representation, Mapping):
            raise RuntimeError("articulated_isaac_runtime_representation_missing")
        result["readbacks"]["physx_homogeneous_articulation_representation"] = (
            _physx_homogeneous_articulation_readback(
                stage, representation, UsdPhysics
            )
        )
        if (
            result["readbacks"]["physx_homogeneous_articulation_representation"]
            .get("passed")
            is not True
        ):
            raise RuntimeError("articulated_isaac_runtime_representation_invalid")

        task_path = str(expected.get("task_joint_prim_path") or "")
        locked_paths = [str(p) for p in (expected.get("locked_joint_prim_paths") or [])]
        task_prim = joints.get(task_path)
        if validation_mode == COMMANDED_ARTICULATION_MODE:
            result["readbacks"]["task_joint_identity"] = {
                "observed": task_path if task_prim else None,
                "passed": task_prim is not None,
            }
        result["readbacks"]["locked_joint_identity"] = {
            "observed": [p for p in locked_paths if p in joints],
            "expected": locked_paths,
            "passed": all(p in joints for p in locked_paths),
        }
        if validation_mode == COMMANDED_ARTICULATION_MODE and task_prim is not None:
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
        if validation_mode == LOCKED_HINGE_RIGID_MODE:
            observed_locked: dict[str, dict[str, Any]] = {}
            for path in locked_paths:
                prim = joints.get(path)
                if prim is None:
                    continue
                axis = prim.GetAttribute("physics:axis")
                row: dict[str, Any] = {
                    "type": observed_types[path],
                    "axis": str(axis.Get()) if axis and axis.HasAuthoredValue() else "",
                }
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    joint = UsdPhysics.RevoluteJoint(prim)
                    row["limits"] = [
                        joint.GetLowerLimitAttr().Get(),
                        joint.GetUpperLimitAttr().Get(),
                    ]
                elif prim.IsA(UsdPhysics.PrismaticJoint):
                    joint = UsdPhysics.PrismaticJoint(prim)
                    row["limits"] = [
                        joint.GetLowerLimitAttr().Get(),
                        joint.GetUpperLimitAttr().Get(),
                    ]
                observed_locked[path] = row
            result["readbacks"]["locked_joint_axes_and_limits"] = {
                "observed": observed_locked,
                "expected": expected.get("locked_joint_axes_and_limits"),
                "passed": observed_locked
                == (expected.get("locked_joint_axes_and_limits") or {}),
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
        reset_positions = expected.get("reset_joint_positions_rad") or {}
        initial_errors = [
            abs(float(initial[index[name]]) - float(value))
            for path, value in reset_positions.items()
            if (name := str(path).rsplit("/", 1)[-1]) in index
        ]
        initial_matches_reset = (
            len(initial_errors) == len(reset_positions)
            and max(initial_errors, default=0.0) <= 0.02
        )
        initial_key = (
            "initial_state_matches_frozen_reset"
            if validation_mode == LOCKED_HINGE_RIGID_MODE
            else "no_initial_penetration"
        )
        result["readbacks"][initial_key] = {
            "observed_initial_positions_rad": [float(v) for v in initial],
            "maximum_reset_error_rad": max(initial_errors, default=None),
            "passed": initial_matches_reset,
            "claim_boundary": (
                "joint_reset_readback_only_not_geometry_penetration_truth"
            ),
        }

        tolerance = float(expected.get("locked_joint_motion_tolerance_rad") or 0.001)
        settle = spec.get("settle") or {}
        reached = []
        locked_drift = 0.0
        initial_by_dof = {
            name: float(initial[position]) for name, position in index.items()
        }
        for angle in sweep:
            if task_dof in index:
                # Only the task door receives a command.  The other four
                # joints remain governed by their authored limits/drives, so
                # their readback is evidence of a lock rather than an echo of
                # a command we sent them every tick.
                view.set_joint_position_targets(
                    np.asarray([math.radians(angle)], dtype=np.float32),
                    joint_indices=np.asarray([index[task_dof]], dtype=np.int64),
                )
            for _ in range(int(settle.get("samples") or 40)):
                world.step(render=False)
            current = np.asarray(view.get_joint_positions())[0]
            if task_dof in index:
                reached.append(math.degrees(float(current[index[task_dof]])))
            for name in locked_dofs:
                if name in index:
                    locked_drift = max(
                        locked_drift,
                        abs(float(current[index[name]]) - initial_by_dof[name]),
                    )
        if not sweep:
            for _ in range(int(settle.get("samples") or 40)):
                world.step(render=False)
            current = np.asarray(view.get_joint_positions())[0]
            for name in locked_dofs:
                if name in index:
                    locked_drift = max(
                        locked_drift,
                        abs(float(current[index[name]]) - initial_by_dof[name]),
                    )
            result["readbacks"]["no_joint_command_issued"] = {
                "command_schedule": [],
                "position_target_command_count": 0,
                "passed": validation_mode == LOCKED_HINGE_RIGID_MODE,
            }
        else:
            maximum = float(expected.get("maximum_commanded_degrees") or 0.0)
            result["readbacks"]["commanded_sweep_reaches_maximum"] = {
                "observed_degrees": reached,
                "maximum_commanded_degrees": maximum,
                "commanded_dof_names": [task_dof],
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
        target = np.asarray(view.get_joint_positions())
        for path, value in reset_positions.items():
            name = str(path).rsplit("/", 1)[-1]
            if name in index:
                target[0][index[name]] = float(value)
        view.set_joint_positions(target)
        for _ in range(30):
            world.step(render=False)
        after_reset = np.asarray(view.get_joint_positions())[0]
        reset_errors = [
            abs(float(after_reset[index[name]]) - float(value))
            for path, value in reset_positions.items()
            if (name := str(path).rsplit("/", 1)[-1]) in index
        ]
        result["readbacks"]["reset_replay_within_tolerance"] = {
            "observed_positions_rad": [float(v) for v in after_reset],
            "maximum_reset_error_rad": max(reset_errors, default=None),
            "passed": len(reset_errors) == len(reset_positions)
            and max(reset_errors, default=0.0) <= 0.02,
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
        required_readbacks = [str(name) for name in spec.get("required_readbacks") or []]
        failed = [
            name
            for name in required_readbacks
            if not isinstance(result["readbacks"].get(name), Mapping)
            or result["readbacks"][name].get("passed") is not True
        ]
        result["blockers"] = [f"articulated_isaac_readback_failed:{name}" for name in failed]
        result["articulation_qualified"] = not failed
        result["status"] = "completed" if not failed else "blocked"
        # The provider lane validates a common result shape across probes; the
        # articulated readbacks are reported through it rather than in a
        # parallel schema only this worker understands.
        result["probe_results"] = [
            {
                "probe": name,
                "passed": bool((result["readbacks"].get(name) or {}).get("passed")),
                "observed": result["readbacks"].get(name),
            }
            for name in sorted(required_readbacks)
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
