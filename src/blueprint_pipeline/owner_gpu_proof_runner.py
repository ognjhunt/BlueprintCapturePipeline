"""Run an owner GPU simulator command and emit Pipeline-validated proof."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context
from .simulation_automation import SIMULATOR_FRAMEWORKS, validate_owner_gpu_system_proof


OWNER_GPU_PROOF_SCHEMA_VERSION = "gpu_owner_system_proof.v1"
OWNER_DEFAULT_SMOKE_POLICY_SCHEMA_VERSION = "owner_default_smoke_policy.v1"
OWNER_SIM_ROBOT_POV_SCHEMA_VERSION = "owner_sim_robot_pov_evidence_manifest.v1"
DEFAULT_ISAAC_ROBOT_ASSET_NAME = "Unitree G1"
DEFAULT_ISAAC_ROBOT_ASSET_URI_OR_PATH = "Robots/Unitree/G1/g1.usd"
DEFAULT_ISAAC_ROBOT_ASSET_SOURCE = "isaac_sim_robot_assets"
DEFAULT_ISAAC_ROBOT_ASSET_CLASS = "humanoid"
DEFAULT_MUJOCO_ROBOT_ASSET_NAME = "Unitree G1"
DEFAULT_MUJOCO_ROBOT_ASSET_URI_OR_PATH = (
    "output/external_assets/mujoco_menagerie/unitree_g1/g1.xml"
)
DEFAULT_MUJOCO_ROBOT_ASSET_SOURCE = "google_deepmind_mujoco_menagerie"
DEFAULT_MUJOCO_ROBOT_ASSET_CLASS = "humanoid_mjcf"
ISAAC_SIMULATOR_BACKENDS = {"isaac_sim", "isaac_lab_arena"}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _default_proof_dir(capture_root: Path) -> Path:
    return capture_root / "pipeline" / "simulation_automation" / "owner_gpu_proof"


def _relative_or_absolute(path: Path, *, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _command_list(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("--command must not be empty")
    return parts


def _owner_attestation(*, operator_id: str, statement: str) -> Dict[str, str]:
    return {
        "operator_id": operator_id,
        "attested_by": operator_id,
        "statement": statement,
    }


def _default_smoke_policy(
    *,
    scene_id: str,
    capture_id: str,
    target: str,
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": OWNER_DEFAULT_SMOKE_POLICY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "policy_id": "blueprint_default_walk_to_target_smoke_policy",
        "policy_kind": "walk_to_target",
        "target": target,
        "success_criteria": [
            "simulator command loads the scene",
            "robot spawn pose is valid",
            "policy trace records at least one walk_to_target action",
            "simulator robot POV evidence manifest records camera/video/frame evidence",
        ],
        "required_owner_command_env": {
            "policy_trace": "BLUEPRINT_POLICY_EXECUTION_TRACE",
            "sim_robot_pov_evidence": "BLUEPRINT_SIM_ROBOT_POV_EVIDENCE",
            "policy_target": "BLUEPRINT_DEFAULT_SMOKE_POLICY_TARGET",
            "robot_asset_name": "BLUEPRINT_ROBOT_ASSET_NAME",
            "robot_asset_uri_or_path": "BLUEPRINT_ROBOT_ASSET_URI_OR_PATH",
            "robot_asset_source": "BLUEPRINT_ROBOT_ASSET_SOURCE",
        },
        "claim_boundary": {
            "default_policy_execution_contract": True,
            "robot_team_policy_quality_proven": False,
            "real_robot_pov_evidence_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _default_robot_asset(
    *,
    simulator_backend: str,
    robot_asset_name: str = "",
    robot_asset_uri_or_path: str = "",
    robot_asset_source: str = "",
    robot_asset_class: str = "",
) -> Dict[str, Any]:
    backend = _string(simulator_backend)
    default_to_isaac_g1 = backend in ISAAC_SIMULATOR_BACKENDS
    default_to_mujoco_g1 = backend == "mujoco"
    name = _string(robot_asset_name) or (
        DEFAULT_ISAAC_ROBOT_ASSET_NAME
        if default_to_isaac_g1
        else DEFAULT_MUJOCO_ROBOT_ASSET_NAME
        if default_to_mujoco_g1
        else ""
    )
    uri_or_path = _string(robot_asset_uri_or_path) or (
        DEFAULT_ISAAC_ROBOT_ASSET_URI_OR_PATH
        if default_to_isaac_g1
        else DEFAULT_MUJOCO_ROBOT_ASSET_URI_OR_PATH
        if default_to_mujoco_g1
        else ""
    )
    source = _string(robot_asset_source) or (
        DEFAULT_ISAAC_ROBOT_ASSET_SOURCE
        if default_to_isaac_g1
        else DEFAULT_MUJOCO_ROBOT_ASSET_SOURCE
        if default_to_mujoco_g1
        else "owner_command"
    )
    asset_class = _string(robot_asset_class) or (
        DEFAULT_ISAAC_ROBOT_ASSET_CLASS
        if default_to_isaac_g1
        else DEFAULT_MUJOCO_ROBOT_ASSET_CLASS
        if default_to_mujoco_g1
        else "robot"
    )
    asset = {
        "name": name,
        "uri_or_path": uri_or_path,
        "source": source,
        "asset_class": asset_class,
        "isaac_robot_asset_required": default_to_isaac_g1,
        "default_isaac_asset_target": default_to_isaac_g1,
    }
    if default_to_isaac_g1:
        asset["catalog_reference"] = "Isaac Sim Robot Assets: Robots/Unitree/G1/g1.usd"
        asset["expected_usd_path_suffix"] = DEFAULT_ISAAC_ROBOT_ASSET_URI_OR_PATH
    return asset


def run_owner_gpu_proof(
    *,
    capture_root: str | Path,
    command: str,
    owner_system_id: str,
    simulator_backend: str,
    simulator_version: str,
    gpu_model: str,
    operator_id: str,
    operator_attestation: str,
    proof_dir: str | Path | None = None,
    timeout_seconds: int = 1800,
    extra_env: Mapping[str, str] | None = None,
    default_policy_target: str = "walk_to_target_pose",
    robot_asset_name: str = "",
    robot_asset_uri_or_path: str = "",
    robot_asset_source: str = "",
    robot_asset_class: str = "",
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    resolved_proof_dir = Path(proof_dir).expanduser().resolve() if proof_dir else _default_proof_dir(context.capture_root)
    ensure_dir(resolved_proof_dir)

    stdout_path = resolved_proof_dir / "owner_simulator_stdout.log"
    stderr_path = resolved_proof_dir / "owner_simulator_stderr.log"
    scene_load_trace_path = resolved_proof_dir / "owner_scene_load_trace.json"
    spawn_trace_path = resolved_proof_dir / "owner_spawn_pose_trace.json"
    action_trace_path = resolved_proof_dir / "owner_action_or_policy_trace.json"
    default_policy_path = resolved_proof_dir / "owner_default_smoke_policy.json"
    sim_robot_pov_path = resolved_proof_dir / "owner_sim_robot_pov_evidence_manifest.json"
    artifact_manifest_path = resolved_proof_dir / "owner_artifact_manifest.json"
    proof_path = context.capture_root / "pipeline" / "simulation_automation" / "gpu_owner_system_proof.json"
    validation_path = (
        context.capture_root
        / "pipeline"
        / "simulation_automation"
        / "owner_gpu_simulator_execution_proof_manifest.json"
    )

    started_at = utc_now_iso()
    policy_target = _string(default_policy_target) or "walk_to_target_pose"
    robot_asset = _default_robot_asset(
        simulator_backend=simulator_backend,
        robot_asset_name=robot_asset_name,
        robot_asset_uri_or_path=robot_asset_uri_or_path,
        robot_asset_source=robot_asset_source,
        robot_asset_class=robot_asset_class,
    )
    write_json(
        default_policy_path,
        _default_smoke_policy(
            scene_id=context.scene_id,
            capture_id=context.capture_id,
            target=policy_target,
            generated_at=started_at,
        ),
    )

    env = os.environ.copy()
    env.update(extra_env or {})
    env.update(
        {
            "BLUEPRINT_CAPTURE_ROOT": str(context.capture_root),
            "BLUEPRINT_GPU_PROOF_DIR": str(resolved_proof_dir),
            "BLUEPRINT_SCENE_LOAD_TRACE": str(scene_load_trace_path),
            "BLUEPRINT_SPAWN_TRACE": str(spawn_trace_path),
            "BLUEPRINT_ACTION_OR_POLICY_TRACE": str(action_trace_path),
            "BLUEPRINT_DEFAULT_SMOKE_POLICY": str(default_policy_path),
            "BLUEPRINT_DEFAULT_SMOKE_POLICY_TARGET": policy_target,
            "BLUEPRINT_POLICY_EXECUTION_TRACE": str(action_trace_path),
            "BLUEPRINT_SIM_ROBOT_POV_EVIDENCE": str(sim_robot_pov_path),
            "BLUEPRINT_ARTIFACT_MANIFEST": str(artifact_manifest_path),
            "BLUEPRINT_OWNER_STDOUT": str(stdout_path),
            "BLUEPRINT_OWNER_STDERR": str(stderr_path),
            "BLUEPRINT_ROBOT_ASSET_NAME": _string(robot_asset.get("name")),
            "BLUEPRINT_ROBOT_ASSET_URI_OR_PATH": _string(robot_asset.get("uri_or_path")),
            "BLUEPRINT_ROBOT_ASSET_SOURCE": _string(robot_asset.get("source")),
            "BLUEPRINT_ROBOT_ASSET_CLASS": _string(robot_asset.get("asset_class")),
        }
    )

    try:
        completed = subprocess.run(
            _command_list(command),
            cwd=str(context.capture_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_seconds),
            check=False,
        )
        exit_code = completed.returncode
        stdout_text = completed.stdout
        stderr_text = completed.stderr
        execution_error = None
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout_text = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr_text = exc.stderr if isinstance(exc.stderr, str) else ""
        execution_error = f"timeout_after_{timeout_seconds}_seconds"
    except (OSError, ValueError) as exc:
        exit_code = 127
        stdout_text = ""
        stderr_text = str(exc)
        execution_error = exc.__class__.__name__
    completed_at = utc_now_iso()

    stdout_path.write_text(stdout_text or "", encoding="utf-8")
    stderr_path.write_text(stderr_text or "", encoding="utf-8")

    pass_fail_criteria: Dict[str, Any] = {
        "status": "passed" if exit_code == 0 and execution_error is None else "failed",
        "passed": exit_code == 0 and execution_error is None,
        "exit_code": exit_code,
    }
    if execution_error:
        pass_fail_criteria["execution_error"] = execution_error

    proof = {
        "schema_version": OWNER_GPU_PROOF_SCHEMA_VERSION,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "owner_system_id": owner_system_id,
        "simulator_backend": simulator_backend,
        "simulator_version": simulator_version,
        "gpu_model": gpu_model,
        "robot_asset": robot_asset,
        "command": command,
        "started_at": started_at,
        "completed_at": completed_at,
        "exit_code": exit_code,
        "stdout_uri_or_path": _relative_or_absolute(stdout_path, base=proof_path.parent),
        "stderr_uri_or_path": _relative_or_absolute(stderr_path, base=proof_path.parent),
        "scene_load_trace_uri_or_path": _relative_or_absolute(
            scene_load_trace_path,
            base=proof_path.parent,
        ),
        "spawn_pose_validation_uri_or_path": _relative_or_absolute(
            spawn_trace_path,
            base=proof_path.parent,
        ),
        "default_smoke_policy_uri_or_path": _relative_or_absolute(
            default_policy_path,
            base=proof_path.parent,
        ),
        "action_or_policy_trace_uri_or_path": _relative_or_absolute(
            action_trace_path,
            base=proof_path.parent,
        ),
        "policy_execution_trace_uri_or_path": _relative_or_absolute(
            action_trace_path,
            base=proof_path.parent,
        ),
        "sim_robot_pov_evidence_uri_or_path": _relative_or_absolute(
            sim_robot_pov_path,
            base=proof_path.parent,
        ),
        "artifact_manifest_uri_or_path": _relative_or_absolute(
            artifact_manifest_path,
            base=proof_path.parent,
        ),
        "default_policy_target": policy_target,
        "default_policy_execution_scope": "owner_gpu_default_walk_to_target_smoke_policy",
        "pass_fail_criteria": pass_fail_criteria,
        "operator_attestation": _owner_attestation(
            operator_id=operator_id,
            statement=operator_attestation,
        ),
        "robot_readiness_proven": False,
        "robot_policy_execution_proven": False,
        "owner_gpu_default_policy_execution_proven": False,
        "owner_gpu_sim_robot_pov_evidence_proven": False,
        "isaac_robot_asset_execution_proven": False,
        "isaac_sim_execution_proven": False,
        "real_robot_pov_evidence_proven": False,
        "physics_contact_validated": False,
        "safety_validated": False,
        "public_claim_upgrade_allowed": False,
    }
    write_json(proof_path, proof)
    validation = validate_owner_gpu_system_proof(
        proof_path=proof_path,
        capture_root=context.capture_root,
        output_path=validation_path,
    )
    return {
        "schema_version": "owner_gpu_proof_runner_result.v1",
        "capture_root": str(context.capture_root),
        "proof_dir": str(resolved_proof_dir),
        "proof_path": str(proof_path),
        "validation_manifest_path": str(validation_path),
        "command_exit_code": exit_code,
        "validation_status": validation.get("status"),
        "owner_gpu_simulator_execution_proven": bool(
            validation.get("owner_gpu_simulator_execution_proven")
        ),
        "validation_blockers": validation.get("blockers") or [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an owner GPU simulator command and emit gpu_owner_system_proof.json"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--command", required=True, help="Owner simulator command to execute")
    parser.add_argument("--proof-dir", default=None)
    parser.add_argument("--owner-system-id", required=True)
    parser.add_argument("--simulator-backend", choices=SIMULATOR_FRAMEWORKS, required=True)
    parser.add_argument("--simulator-version", required=True)
    parser.add_argument("--gpu-model", required=True)
    parser.add_argument("--operator-id", required=True)
    parser.add_argument("--operator-attestation", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--default-policy-target",
        default="walk_to_target_pose",
        help="Target label or pose id for the built-in walk-to-target smoke policy.",
    )
    parser.add_argument(
        "--robot-asset-name",
        default="",
        help="Robot asset display name; defaults to Unitree G1 for Isaac backends.",
    )
    parser.add_argument(
        "--robot-asset-uri-or-path",
        default="",
        help=(
            "Robot asset URI or content-browser path; defaults to "
            "Robots/Unitree/G1/g1.usd for Isaac backends."
        ),
    )
    parser.add_argument(
        "--robot-asset-source",
        default="",
        help="Robot asset catalog/source; defaults to isaac_sim_robot_assets for Isaac backends.",
    )
    parser.add_argument(
        "--robot-asset-class",
        default="",
        help="Robot asset class; defaults to humanoid for Isaac backends.",
    )
    args = parser.parse_args(argv)

    result = run_owner_gpu_proof(
        capture_root=args.capture_root,
        command=args.command,
        proof_dir=args.proof_dir,
        owner_system_id=args.owner_system_id,
        simulator_backend=args.simulator_backend,
        simulator_version=args.simulator_version,
        gpu_model=args.gpu_model,
        operator_id=args.operator_id,
        operator_attestation=args.operator_attestation,
        timeout_seconds=args.timeout_seconds,
        default_policy_target=args.default_policy_target,
        robot_asset_name=args.robot_asset_name,
        robot_asset_uri_or_path=args.robot_asset_uri_or_path,
        robot_asset_source=args.robot_asset_source,
        robot_asset_class=args.robot_asset_class,
    )
    print(f"[owner-gpu-proof] validation_status={result['validation_status']}")
    print(f"[owner-gpu-proof] proof_path={result['proof_path']}")
    print(f"[owner-gpu-proof] validation_manifest={result['validation_manifest_path']}")
    if result["validation_blockers"]:
        print("[owner-gpu-proof] blockers=" + ",".join(result["validation_blockers"]))
    return 0 if result["owner_gpu_simulator_execution_proven"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
