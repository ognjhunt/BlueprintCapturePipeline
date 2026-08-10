"""Execute one frozen learned candidate after native construction and controls."""

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


RESULT_SCHEMA_VERSION = "native_task_arena_policy_result.v1"
RESULT_FILENAME = "native_task_arena_policy_result.v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _persist(path: Path, result: dict[str, Any]) -> None:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    module = type(value).__module__
    if module == "warp" or module.startswith("warp."):
        import warp as wp

        return wp.to_torch(value)
    raise TypeError(f"unsupported_sim_array:{module}.{type(value).__name__}")


def _inputs(runtime: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    verified: dict[str, Path] = {}
    for row in manifest.get("bound_runtime_inputs") or []:
        relative = str(row.get("relative_path") or "")
        path = runtime / relative
        if (
            not relative.startswith("runtime_inputs/")
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise RuntimeError(f"native_task_policy_input_identity_mismatch:{relative}")
        verified[Path(relative).name] = path
    required = {
        "native_task_arena_construction_result.v1.json",
        "native_task_arena_control_result.v1.json",
        "native_task_arena_policy_execution_spec.v1.json",
    }
    if set(verified) != required:
        raise RuntimeError("native_task_policy_inputs_incomplete")
    return verified


def _policy_client(spec: Mapping[str, Any]) -> Any:
    endpoint = spec["policy_endpoint"]
    secret = os.environ.get(str(endpoint["credential_env"]))
    if spec["candidate_id"] == "pi05_droid":
        from blueprint_pipeline.openpi_droid_policy_runtime import (
            OpenPIDroidPolicySpec,
            OpenPIWebsocketDroidPolicyClient,
        )

        return OpenPIWebsocketDroidPolicyClient(
            spec=OpenPIDroidPolicySpec(**spec["policy_spec"]),
            host=str(endpoint["host"]),
            port=int(endpoint["port"]),
            api_key=secret,
        )
    from blueprint_pipeline.groot_n17_droid_policy_runtime import (
        GrootN17DroidPolicyClient,
        GrootN17DroidPolicySpec,
    )

    return GrootN17DroidPolicyClient(
        spec=GrootN17DroidPolicySpec(**spec["policy_spec"]),
        worker_identity_receipt=spec["policy_identity_receipt"],
        host=str(endpoint["host"]),
        port=int(endpoint["port"]),
        api_token=secret,
    )


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
        "candidate_policy_queried": False,
        "policy_outcome_interpretable": False,
        "scientific_outcome_admitted": False,
        "ranking_eligible": False,
        "provider_zero_required_after_return": True,
        "simulator_execution_is_not_physical_truth": True,
    }
    simulation_app = None
    try:
        from blueprint_pipeline.decision_evidence_contracts import canonical_digest
        manifest = json.loads(
            (runtime / "adp_arena_provider_manifest.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("schema_version") != "native_task_arena_provider_bundle.v1"
            or manifest.get("execution_mode") != "policy"
            or manifest.get("policy_candidate_id") not in {
                "pi05_droid",
                "groot_n17_droid",
            }
            or manifest.get("candidate_policy_queried") is not False
            or manifest.get("input_digest")
            != canonical_digest(manifest, digest_field="input_digest")
        ):
            raise RuntimeError("native_task_policy_manifest_invalid")
        inputs = _inputs(runtime, manifest)
        spec = json.loads(
            inputs["native_task_arena_policy_execution_spec.v1.json"].read_text()
        )
        if (
            spec.get("schema_version")
            != "native_task_arena_policy_execution_spec.v1"
            or spec.get("candidate_id") not in {"pi05_droid", "groot_n17_droid"}
            or spec.get("execution_spec_digest")
            != canonical_digest(spec, digest_field="execution_spec_digest")
        ):
            raise RuntimeError("native_task_policy_execution_spec_invalid")
        construction = json.loads(
            inputs["native_task_arena_construction_result.v1.json"].read_text()
        )
        controls = json.loads(
            inputs["native_task_arena_control_result.v1.json"].read_text()
        )
        packet = runtime / "native_task_packet"
        scene_plan = json.loads(
            (packet / "native_task_arena_scene_plan.v1.json").read_text()
        )
        pair = controls.get("control_pair") or {}
        if (
            spec["candidate_id"] != manifest["policy_candidate_id"]
            or construction.get("result_digest")
            != spec["construction_result_digest"]
            or construction.get("construction_gate_qualified") is not True
            or controls.get("result_digest") != spec["control_result_digest"]
            or controls.get("controls_qualified") is not True
            or pair.get("pair_digest") != spec["control_pair_digest"]
            or pair.get("cell_admitted_for_policy_execution") is not True
            or scene_plan.get("plan_digest") != spec["scene_plan_digest"]
        ):
            raise RuntimeError("native_task_policy_admission_binding_mismatch")
        result["phase_reached"] = "inputs_verified"

        from blueprint_pipeline.native_task_isaaclab_launch import (
            launch_native_task_isaaclab,
        )

        simulation_app, launch = launch_native_task_isaaclab(
            output_root / "native_task_runtime_source_provisioning.v1.json"
        )
        result["isaaclab_launch"] = launch
        import torch

        from blueprint_pipeline.adp009d_droid_action_execution import (
            GripperConvention,
        )
        from blueprint_pipeline.adp009d_policy_episode import run_policy_episode
        from blueprint_pipeline.native_franka_pose_servo import (
            NativeFrankaDifferentialIkServo,
        )
        from blueprint_pipeline.native_task_arena_construction_worker import (
            _gripper_convention_probe,
            preflight_native_dependency_matrix,
        )
        from blueprint_pipeline.native_task_arena_device_readback import (
            read_native_task_arena_device_binding,
        )
        from blueprint_pipeline.native_task_arena_preconstruction import (
            prepare_native_task_arena_preconstruction,
        )
        from blueprint_pipeline.native_task_arena_readback import (
            NativeArticulatedTaskArenaReadback,
        )
        from blueprint_pipeline.native_task_arena_runtime import (
            build_native_task_arena_environment,
        )
        from blueprint_pipeline.native_task_episode_environment import (
            build_native_task_episode_environment,
        )

        dependencies = preflight_native_dependency_matrix(
            robot_id=str(scene_plan["robot"]["robot_id"])
        )
        if not dependencies["all_required_available"]:
            result["blockers"].extend(dependencies["blockers"])
            raise RuntimeError("native_task_policy_dependency_preflight_failed")
        preconstruction = prepare_native_task_arena_preconstruction(
            expected_device="cuda:0"
        )
        if not preconstruction["passed"]:
            result["blockers"].extend(preconstruction["blockers"])
            raise RuntimeError("native_task_policy_preconstruction_failed")
        built = build_native_task_arena_environment(
            scene_plan,
            device="cuda:0",
            bundle_root=packet,
            preconstruction_receipt=preconstruction,
        )
        device = read_native_task_arena_device_binding(built, expected_device="cuda:0")
        if not device["passed"]:
            result["blockers"].extend(device["blockers"])
            raise RuntimeError("native_task_policy_device_binding_failed")
        env = built.env
        seed = int(scene_plan["scenario"]["seed"])
        env.reset(seed=seed)
        robot = env.unwrapped.scene["robot"]
        gripper = _gripper_convention_probe(env=env, robot=robot, seed=seed, torch=torch)
        if gripper["status"] != "measured":
            result["blockers"].extend(gripper["blockers"])
            raise RuntimeError("native_task_policy_gripper_unresolved")
        env.reset(seed=seed)
        servo = NativeFrankaDifferentialIkServo(env=env, robot=robot)
        task_readback = (
            NativeArticulatedTaskArenaReadback(built)
            if scene_plan["task_kind"] == "articulated_open_close"
            else None
        )
        episode_environment, environment_receipt = build_native_task_episode_environment(
            built=built,
            gripper_convention=gripper,
            servo=servo,
            task_readback=task_readback,
            to_tensor=_to_tensor,
        )
        result["episode_environment"] = environment_receipt
        policy = _policy_client(spec)
        result["phase_reached"] = "policy_client_verified"
        episode_id = f"{scene_plan['task_id']}--{spec['cell_id']}--{spec['candidate_id']}"
        episode = run_policy_episode(
            environment=episode_environment,
            policy=policy,
            candidate_id=spec["candidate_id"],
            prompt=spec["prompt"],
            task_spec=scene_plan["task_spec"],
            max_policy_queries=spec["max_policy_queries"],
            settle_window_samples=int(scene_plan["task_spec"]["settle_window_samples"]),
            open_loop_horizon=spec["open_loop_horizon"],
            gripper=GripperConvention(
                closed_command=float(gripper["closed_command"]),
                open_command=float(gripper["open_command"]),
                measured_by_probe=True,
            ),
            media_output_dir=output_root / "episodes",
            episode_id=episode_id,
        )
        result["candidate_policy_queried"] = True
        result["episode"] = episode
        result["policy_outcome_interpretable"] = bool(
            episode["motion_evidence"]["policy_outcome_interpretable"]
        )
        result["scientific_outcome_admitted"] = bool(
            result["policy_outcome_interpretable"]
            and episode["score"].get("status") == "scored"
        )
        result["ranking_eligible"] = result["scientific_outcome_admitted"]
        result["status"] = "completed"
        result["phase_reached"] = "episode_complete"
    except BaseException as exc:  # noqa: BLE001 - retain every paid failure
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": result["phase_reached"],
            "traceback": traceback.format_exc(),
        }
        result["blockers"].append(
            f"native_task_policy_failed_at_{result['phase_reached']}:"
            f"{type(exc).__name__}:{exc}"
        )
    finally:
        result["blockers"] = sorted(set(result["blockers"]))
        result["completed_at_unix_ns"] = time.time_ns()
        _persist(output, result)
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:  # noqa: BLE001
                pass
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
