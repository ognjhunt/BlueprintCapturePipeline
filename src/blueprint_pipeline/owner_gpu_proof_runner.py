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
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    resolved_proof_dir = Path(proof_dir).expanduser().resolve() if proof_dir else _default_proof_dir(context.capture_root)
    ensure_dir(resolved_proof_dir)

    stdout_path = resolved_proof_dir / "owner_simulator_stdout.log"
    stderr_path = resolved_proof_dir / "owner_simulator_stderr.log"
    scene_load_trace_path = resolved_proof_dir / "owner_scene_load_trace.json"
    spawn_trace_path = resolved_proof_dir / "owner_spawn_pose_trace.json"
    action_trace_path = resolved_proof_dir / "owner_action_or_policy_trace.json"
    artifact_manifest_path = resolved_proof_dir / "owner_artifact_manifest.json"
    proof_path = context.capture_root / "pipeline" / "simulation_automation" / "gpu_owner_system_proof.json"
    validation_path = (
        context.capture_root
        / "pipeline"
        / "simulation_automation"
        / "owner_gpu_simulator_execution_proof_manifest.json"
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
            "BLUEPRINT_ARTIFACT_MANIFEST": str(artifact_manifest_path),
            "BLUEPRINT_OWNER_STDOUT": str(stdout_path),
            "BLUEPRINT_OWNER_STDERR": str(stderr_path),
        }
    )

    started_at = utc_now_iso()
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
        "action_or_policy_trace_uri_or_path": _relative_or_absolute(
            action_trace_path,
            base=proof_path.parent,
        ),
        "artifact_manifest_uri_or_path": _relative_or_absolute(
            artifact_manifest_path,
            base=proof_path.parent,
        ),
        "pass_fail_criteria": pass_fail_criteria,
        "operator_attestation": _owner_attestation(
            operator_id=operator_id,
            statement=operator_attestation,
        ),
        "robot_readiness_proven": False,
        "robot_policy_execution_proven": False,
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
    )
    print(f"[owner-gpu-proof] validation_status={result['validation_status']}")
    print(f"[owner-gpu-proof] proof_path={result['proof_path']}")
    print(f"[owner-gpu-proof] validation_manifest={result['validation_manifest_path']}")
    if result["validation_blockers"]:
        print("[owner-gpu-proof] blockers=" + ",".join(result["validation_blockers"]))
    return 0 if result["owner_gpu_simulator_execution_proven"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
