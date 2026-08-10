"""Run the required task-neutral controls on one qualified native Arena cell.

The worker consumes a sealed scene packet plus two digest-bound runtime inputs:
the successful construction receipt and the control plan compiled from it.  It
does not name a scene, object class, joint, or policy.  Both controls execute
through the same native eight-dimensional action seam and deterministic scorer
that later policy episodes use.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = "native_task_arena_control_result.v1"
RESULT_FILENAME = "native_task_arena_control_result.v1.json"


def _announce(phase: str, status: str = "started") -> None:
    print(
        f"BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena_controls:{phase}:{status}",
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


def _persist(output: Path, result: dict[str, Any]) -> None:
    result["result_digest"] = _canonical_digest(result, field="result_digest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _load_and_verify_manifest(runtime: Path) -> dict[str, Any]:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    manifest = json.loads(
        (runtime / "adp_arena_provider_manifest.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "native_task_arena_provider_bundle.v1"
        or manifest.get("execution_mode") != "controls"
        or manifest.get("policy_candidate_id") is not None
        or manifest.get("candidate_policy_queried") is not False
        or manifest.get("input_digest")
        != canonical_digest(manifest, digest_field="input_digest")
    ):
        raise RuntimeError("native_task_controls_manifest_invalid")
    return manifest


def _verified_runtime_inputs(
    runtime: Path, manifest: Mapping[str, Any]
) -> dict[str, Path]:
    rows = manifest.get("bound_runtime_inputs")
    if not isinstance(rows, list):
        raise RuntimeError("native_task_controls_runtime_inputs_invalid")
    verified: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("native_task_controls_runtime_inputs_invalid")
        relative = str(row.get("relative_path") or "")
        path = runtime / relative
        if (
            not relative.startswith("runtime_inputs/")
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise RuntimeError(
                f"native_task_controls_runtime_input_identity_mismatch:{relative}"
            )
        verified[Path(relative).name] = path
    required = {
        "native_task_arena_construction_result.v1.json",
        "adp_task_control_plan.v1.json",
    }
    if set(verified) != required:
        raise RuntimeError("native_task_controls_runtime_inputs_incomplete")
    return verified


def _to_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    value_module = type(value).__module__
    if value_module == "warp" or value_module.startswith("warp."):
        import warp as wp

        return wp.to_torch(value)
    raise TypeError(f"unsupported_sim_array:{value_module}.{type(value).__name__}")


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
        "controls_qualified": False,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
        "simulator_execution_is_not_physical_truth": True,
    }
    simulation_app = None
    try:
        _announce("input_verification")
        manifest = _load_and_verify_manifest(runtime)
        inputs = _verified_runtime_inputs(runtime, manifest)
        packet = runtime / "native_task_packet"
        packet_receipt = json.loads(
            (packet / "native_task_arena_packet_receipt.v1.json").read_text(
                encoding="utf-8"
            )
        )
        scene_plan = json.loads(
            (packet / "native_task_arena_scene_plan.v1.json").read_text(
                encoding="utf-8"
            )
        )
        construction = json.loads(
            inputs["native_task_arena_construction_result.v1.json"].read_text(
                encoding="utf-8"
            )
        )
        control_plan = json.loads(
            inputs["adp_task_control_plan.v1.json"].read_text(encoding="utf-8")
        )
        if (
            packet_receipt.get("receipt_digest")
            != manifest.get("packet_receipt_digest")
            or scene_plan.get("plan_digest")
            != manifest.get("arena_scene_plan_digest")
            or construction.get("result_digest")
            != control_plan.get("planner_receipt_digest")
            or control_plan.get("construction_scene_plan_digest")
            != scene_plan.get("plan_digest")
        ):
            raise RuntimeError("native_task_controls_input_binding_mismatch")
        result["manifest_input_digest"] = manifest["input_digest"]
        result["implementation_commit"] = manifest["implementation_commit"]
        result["packet_receipt_digest"] = packet_receipt["receipt_digest"]
        result["scene_plan_digest"] = scene_plan["plan_digest"]
        result["construction_result_digest"] = construction["result_digest"]
        result["control_plan_digest"] = control_plan["plan_digest"]
        result["phase_reached"] = "inputs_verified"
        _announce("input_verification", "completed")

        _announce("simulation_app")
        from blueprint_pipeline.native_task_isaaclab_launch import (
            launch_native_task_isaaclab,
        )

        simulation_app, launch_receipt = launch_native_task_isaaclab(
            output_root / "native_task_runtime_source_provisioning.v1.json"
        )
        result["isaaclab_launch"] = launch_receipt
        _announce("simulation_app", "completed")

        from blueprint_pipeline.native_task_arena_construction_worker import (
            _gripper_convention_probe,
            preflight_native_dependency_matrix,
        )

        _announce("dependency_matrix")
        dependency_matrix = preflight_native_dependency_matrix(
            robot_id=str(scene_plan["robot"]["robot_id"])
        )
        result["dependency_matrix"] = dependency_matrix
        if not dependency_matrix["all_required_available"]:
            result["blockers"].extend(dependency_matrix["blockers"])
            raise RuntimeError("native_task_controls_dependency_preflight_failed")
        result["phase_reached"] = "dependencies_qualified"
        _announce("dependency_matrix", "completed")

        import torch

        from blueprint_pipeline.adp009d_control_episode import (
            run_task_neutral_controls,
        )
        from blueprint_pipeline.native_franka_pose_servo import (
            NativeFrankaDifferentialIkServo,
        )
        from blueprint_pipeline.native_task_arena_readback import (
            NativeArticulatedTaskArenaReadback,
        )
        from blueprint_pipeline.native_task_arena_device_readback import (
            read_native_task_arena_device_binding,
        )
        from blueprint_pipeline.native_task_arena_runtime import (
            build_native_task_arena_environment,
        )
        from blueprint_pipeline.native_task_episode_environment import (
            build_native_task_episode_environment,
        )

        _announce("environment_build")
        built = build_native_task_arena_environment(
            scene_plan, device="cuda:0", bundle_root=packet
        )
        device_readback = read_native_task_arena_device_binding(
            built, expected_device="cuda:0"
        )
        result["device_readback"] = device_readback
        if not device_readback["passed"]:
            result["blockers"].extend(device_readback["blockers"])
            raise RuntimeError("native_task_arena_device_binding_failed")
        env = built.env
        seed = int(scene_plan["scenario"]["seed"])
        env.reset(seed=seed)
        robot = env.unwrapped.scene["robot"]
        readback = NativeArticulatedTaskArenaReadback(built)
        result["native_isaac_executed"] = True
        result["phase_reached"] = "environment_built"
        _announce("environment_build", "completed")

        _announce("gripper_convention")
        gripper = _gripper_convention_probe(
            env=env, robot=robot, seed=seed, torch=torch
        )
        result["gripper_convention"] = gripper
        result["blockers"].extend(gripper["blockers"])
        if gripper["status"] != "measured":
            raise RuntimeError("native_task_controls_gripper_convention_unresolved")
        env.reset(seed=seed)
        servo = NativeFrankaDifferentialIkServo(env=env, robot=robot)
        episode_environment, environment_receipt = (
            build_native_task_episode_environment(
                built=built,
                gripper_convention=gripper,
                servo=servo,
                task_readback=readback,
                to_tensor=_to_tensor,
            )
        )
        result["episode_environment"] = environment_receipt
        result["phase_reached"] = "episode_environment_bound"
        _announce("gripper_convention", "completed")

        _announce("required_controls")
        pair = run_task_neutral_controls(
            environment=episode_environment,
            task_spec=scene_plan["task_spec"],
            control_plan=control_plan,
            gripper_open_command=float(gripper["open_command"]),
            gripper_closed_command=float(gripper["closed_command"]),
            output_dir=output_root / "controls",
        )
        result["control_pair"] = pair
        result["controls_qualified"] = pair["cell_admitted_for_policy_execution"]
        result["blockers"].extend(pair["policy_execution_blockers"])
        result["blockers"] = sorted(set(result["blockers"]))
        result["status"] = "completed" if not result["blockers"] else "blocked"
        result["phase_reached"] = "required_controls_complete"
        _announce(
            "required_controls",
            "completed" if result["controls_qualified"] else "blocked",
        )
    except BaseException as exc:  # noqa: BLE001 - retain every paid failure
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": result["phase_reached"],
            "traceback": traceback.format_exc(),
        }
        result["blockers"].append(
            f"native_task_controls_failed_at_{result['phase_reached']}:"
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
