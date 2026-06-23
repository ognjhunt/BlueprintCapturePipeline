"""Build or import a bounded GR00T N1.7 + UNITREE_G1_SONIC provider smoke.

This provider bundle proves only packaging and, after a real provider run,
whether a GR00T/SONIC action command returned a Blueprint-compatible action. A
dry run, replay, or imported provider output is not a fresh local policy proof.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .unitree_groot_n17_sonic_policy_runtime import (
    GROOT_ROOT_ENV,
    N17_CHECKPOINT_ENV,
    POLICY_COMMAND_ENV,
    POLICY_ID,
    POLICY_SERVER_URL_ENV,
    SIM2SIM_COMMAND_ENV,
    SONIC_CHECKPOINT_ENV,
    WBC_ROOT_ENV,
)


SCHEMA_VERSION = "unitree_groot_n17_sonic_policy_provider_smoke.v1"
OUTPUT_SCHEMA_VERSION = "unitree_groot_n17_sonic_policy_provider_output.v1"
DEFAULT_TASK_ID = "contact_or_push_light_object"
DEFAULT_TASK_PROMPT = "move the Unitree G1 SONIC hand toward the light object"
DEFAULT_BUNDLE_FILENAME = "unitree_groot_n17_sonic_policy_provider_runtime_bundle.zip"
PROVIDER_OUTPUT_FILENAME = "unitree_groot_n17_sonic_policy_provider_output.json"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_executable(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _copy_frame(frame_path: Path, output_path: Path) -> None:
    if not frame_path.is_file():
        raise FileNotFoundError(f"unitree_groot_n17_sonic_input_frame_missing:{frame_path}")
    ensure_dir(output_path.parent)
    shutil.copy2(frame_path, output_path)


PROVIDER_RUNNER = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

OUTPUT_SCHEMA_VERSION = "unitree_groot_n17_sonic_policy_provider_output.v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase(name: str, **fields: Any) -> None:
    print(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_PHASE:"
        + json.dumps(
            {
                "phase": name,
                "observed_at_epoch": round(time.time(), 3),
                "raw_secret_values_recorded": False,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    runtime_dir = Path(__file__).resolve().parent
    payload_path = Path(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_INPUT", runtime_dir / "policy_input.json"))
    output_path = Path(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT", runtime_dir / "unitree_groot_n17_sonic_policy_provider_output.json"))
    command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND", "")
    n17_checkpoint = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT", "")
    sonic_checkpoint = os.environ.get("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT", "")
    groot_root = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT", "")
    wbc_root = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT", "")
    policy_server_url = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL", "")
    sim2sim_command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND", "")
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        _phase("invoke_unitree_groot_n17_sonic_adapter", command_configured=bool(command), sonic_checkpoint_configured=bool(sonic_checkpoint))
        from blueprint_pipeline.unitree_groot_n17_sonic_policy_command_adapter import run_unitree_groot_n17_sonic_policy

        response, exit_code = run_unitree_groot_n17_sonic_policy(
            payload=payload,
            command=command,
            n17_checkpoint=n17_checkpoint,
            sonic_checkpoint=sonic_checkpoint,
            groot_root=groot_root,
            wbc_root=wbc_root,
            policy_server_url=policy_server_url,
            sim2sim_command=sim2sim_command,
            timeout_seconds=float(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_TIMEOUT_SECONDS", "240")),
        )
        action = _mapping(response.get("action"))
        completed = exit_code == 0 and response.get("status") == "completed" and bool(action)
        output = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "completed" if completed else "blocked",
            "policy_id": "unitree_groot_n17_sonic_policy",
            "unitree_groot_n17_sonic_model_executed": bool(response.get("model_ran")),
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(response.get("unitree_groot_n17_sonic_policy_action_command_ran")),
            "policy_action_model_command_ran": bool(response.get("unitree_groot_n17_sonic_policy_action_command_ran")),
            "action": action or None,
            "adapter_response": response,
            "endpoint_closed_loop_policy_proven": False,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "blockers": [] if completed else list(response.get("blockers", []) or ["unitree_groot_n17_sonic_provider_smoke_blocked"]),
        }
        _write_json(output_path, output)
        return 0 if completed else 2
    except Exception as exc:
        _write_json(
            output_path,
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "status": "failed",
                "policy_id": "unitree_groot_n17_sonic_policy",
                "unitree_groot_n17_sonic_model_executed": False,
                "unitree_groot_n17_sonic_policy_action_command_ran": False,
                "policy_action_model_command_ran": False,
                "action": None,
                "traceback_tail": traceback.format_exc()[-4000:],
                "blockers": [f"unitree_groot_n17_sonic_provider_runner_failed:{type(exc).__name__}"],
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


RUN_SCRIPT = """#!/usr/bin/env bash
set +e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python3 unitree_groot_n17_sonic_provider_runner.py
runner_rc=$?
if [ $runner_rc -ne 0 ] && [ ! -f unitree_groot_n17_sonic_policy_provider_output.json ]; then
python3 - <<'PY'
import json
from pathlib import Path
Path("unitree_groot_n17_sonic_policy_provider_output.json").write_text(json.dumps({
    "schema_version": "unitree_groot_n17_sonic_policy_provider_output.v1",
    "status": "failed",
    "policy_id": "unitree_groot_n17_sonic_policy",
    "unitree_groot_n17_sonic_model_executed": False,
    "unitree_groot_n17_sonic_policy_action_command_ran": False,
    "policy_action_model_command_ran": False,
    "action": None,
    "blockers": [
        "unitree_groot_n17_sonic_provider_runner_failed_without_runtime_result",
        "blocked_unitree_groot_n17_sonic_process_exited_without_result"
    ],
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False
}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
fi
exit $runner_rc
"""


def _write_minimal_blueprint_runtime(runtime_dir: Path) -> list[str]:
    package_dir = runtime_dir / "blueprint_pipeline"
    ensure_dir(package_dir)
    copied: list[str] = []
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    copied.append("provider_runtime/blueprint_pipeline/__init__.py")
    source_dir = Path(__file__).resolve().parent
    for filename in (
        "unitree_groot_n17_sonic_policy_command_adapter.py",
        "unitree_groot_n17_sonic_policy_runtime.py",
    ):
        destination = package_dir / filename
        shutil.copy2(source_dir / filename, destination)
        copied.append(f"provider_runtime/blueprint_pipeline/{filename}")
    common_path = package_dir / "common.py"
    shutil.copy2(source_dir / "common.py", common_path)
    copied.append("provider_runtime/blueprint_pipeline/common.py")
    return copied


def _policy_input(
    *,
    frame_path: Path,
    task_id: str,
    task_prompt: str,
) -> dict[str, Any]:
    return {
        "observation": {
            "schema_version": "blueprint_policy_observation.v1",
            "task_id": task_id,
            "task_prompt": task_prompt,
            "visual_observation": {"camera_frame_path": str(frame_path)},
            "object_state": {
                "object_id": "blueprint_light_object",
                "position": [0.36, -0.65, 0.27],
            },
            "route_task_state": {
                "target_pose": [0.54, -0.65, 0.79],
                "target_error_m": 0.8,
            },
        }
    }


def build_unitree_groot_n17_sonic_policy_provider_bundle(
    *,
    job_dir: str | Path,
    frame_path: str | Path,
    task_id: str = DEFAULT_TASK_ID,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    policy_command: str | None = None,
    n17_checkpoint: str | None = None,
    sonic_checkpoint: str | None = None,
    groot_root: str | None = None,
    wbc_root: str | None = None,
    policy_server_url: str | None = None,
    sim2sim_command: str | None = None,
) -> dict[str, Any]:
    job = Path(job_dir)
    ensure_dir(job)
    runtime_dir = job / "provider_runtime"
    ensure_dir(runtime_dir)
    frame_copy = runtime_dir / "input_frame.png"
    _copy_frame(Path(frame_path).expanduser(), frame_copy)
    payload = _policy_input(frame_path=frame_copy, task_id=task_id, task_prompt=task_prompt)
    policy_input_path = runtime_dir / "policy_input.json"
    write_json(policy_input_path, payload)
    runner_path = runtime_dir / "unitree_groot_n17_sonic_provider_runner.py"
    _write_executable(runner_path, PROVIDER_RUNNER)
    run_script_path = runtime_dir / "run_unitree_groot_n17_sonic_provider_runtime.sh"
    _write_executable(run_script_path, RUN_SCRIPT)
    bundled_blueprint_modules = _write_minimal_blueprint_runtime(runtime_dir)
    runtime_execution_blockers: list[str] = []
    if not policy_command:
        runtime_execution_blockers.append("blocked_missing_unitree_groot_n17_sonic_policy_command")
    if not n17_checkpoint:
        runtime_execution_blockers.append("blocked_missing_unitree_groot_n17_checkpoint")
    if not sonic_checkpoint:
        runtime_execution_blockers.append("blocked_missing_unitree_g1_sonic_checkpoint")
    manifest = {
        "schema_version": "unitree_groot_n17_sonic_policy_provider_bundle.v1",
        "generated_at": utc_now_iso(),
        "status": "bundle_ready",
        "policy_id": POLICY_ID,
        "runtime_entrypoint": "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh",
        "runner_path": str(runner_path),
        "policy_input_path": str(policy_input_path),
        "input_frame_path": str(frame_copy),
        "bundled_blueprint_modules": bundled_blueprint_modules,
        "policy_command_configured": bool(policy_command),
        "n17_checkpoint_configured": bool(n17_checkpoint),
        "g1_sonic_checkpoint_configured": bool(sonic_checkpoint),
        "groot_root_configured": bool(groot_root),
        "wbc_root_configured": bool(wbc_root),
        "policy_server_url_configured": bool(policy_server_url),
        "sim2sim_command_configured": bool(sim2sim_command),
        "local_bundle_ready_for_remote_staging": not runtime_execution_blockers,
        "ready_for_fresh_model_execution": not runtime_execution_blockers,
        "runtime_execution_blockers": runtime_execution_blockers,
        "expected_output_filename": PROVIDER_OUTPUT_FILENAME,
        "env_contract": {
            POLICY_COMMAND_ENV: "<configured>" if policy_command else None,
            N17_CHECKPOINT_ENV: "<configured>" if n17_checkpoint else None,
            SONIC_CHECKPOINT_ENV: "<configured>" if sonic_checkpoint else None,
            GROOT_ROOT_ENV: "<configured>" if groot_root else None,
            WBC_ROOT_ENV: "<configured>" if wbc_root else None,
            POLICY_SERVER_URL_ENV: "<configured>" if policy_server_url else None,
            SIM2SIM_COMMAND_ENV: "<configured>" if sim2sim_command else None,
        },
        "truth_boundary": {
            "provider_bundle_is_not_model_execution": True,
            "unitree_groot_n17_sonic_policy_action_command_ran": False,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    manifest_path = runtime_dir / "unitree_groot_n17_sonic_policy_provider_manifest.json"
    write_json(manifest_path, manifest)
    bundle_path = job / DEFAULT_BUNDLE_FILENAME
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(runtime_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(job).as_posix())
    with zipfile.ZipFile(bundle_path) as archive:
        bundle_file_count = len(archive.namelist())
    manifest.update(
        {
            "bundle_path": str(bundle_path),
            "manifest_path": str(manifest_path),
            "bundle_file_count": bundle_file_count,
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def import_unitree_groot_n17_sonic_provider_output(
    *,
    provider_output_zip: str | Path,
    extraction_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    source_zip = Path(provider_output_zip).expanduser()
    extract_dir = Path(extraction_dir)
    ensure_dir(extract_dir)
    with zipfile.ZipFile(source_zip) as archive:
        archive.extractall(extract_dir)
    candidates = [
        extract_dir / PROVIDER_OUTPUT_FILENAME,
        extract_dir / "provider_runtime" / PROVIDER_OUTPUT_FILENAME,
    ]
    payload_path = next((path for path in candidates if path.is_file()), None)
    if payload_path is None:
        payload = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "blocked",
            "unitree_groot_n17_sonic_model_executed": False,
            "unitree_groot_n17_sonic_policy_action_command_ran": False,
            "policy_action_model_command_ran": False,
            "action": None,
            "blockers": ["unitree_groot_n17_sonic_provider_output_json_missing"],
        }
    else:
        value = json.loads(payload_path.read_text(encoding="utf-8"))
        payload = dict(value) if isinstance(value, Mapping) else {}
    completed = bool(
        payload.get("status") == "completed"
        and payload.get("unitree_groot_n17_sonic_model_executed") is True
        and payload.get("unitree_groot_n17_sonic_policy_action_command_ran") is True
        and isinstance(payload.get("action"), Mapping)
    )
    imported = {
        "schema_version": "unitree_groot_n17_sonic_policy_provider_import.v1",
        "status": "completed" if completed else "blocked",
        "provider_output_zip": str(source_zip),
        "provider_output_path": str(payload_path) if payload_path else None,
        "unitree_groot_n17_sonic_model_executed": bool(
            payload.get("unitree_groot_n17_sonic_model_executed")
        ),
        "unitree_groot_n17_sonic_policy_action_command_ran": bool(
            payload.get("unitree_groot_n17_sonic_policy_action_command_ran")
        ),
        "policy_action_model_command_ran": bool(payload.get("policy_action_model_command_ran")),
        "action": dict(payload["action"]) if isinstance(payload.get("action"), Mapping) else None,
        "truth_boundary": {
            "provider_output_import_is_not_fresh_local_policy_execution": True,
            "provider_output_import_is_not_closed_loop_endpoint_control": True,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "blockers": list(
            payload.get("blockers", [])
            or ([] if completed else ["unitree_groot_n17_sonic_provider_output_not_completed"])
        ),
    }
    write_json(Path(output_path), imported)
    return imported


def run_unitree_groot_n17_sonic_policy_provider_smoke(
    *,
    job_dir: str | Path,
    frame_path: str | Path,
    provider_output_zip: str | Path | None = None,
    dry_run: bool = True,
    policy_command: str | None = None,
    n17_checkpoint: str | None = None,
    sonic_checkpoint: str | None = None,
    groot_root: str | None = None,
    wbc_root: str | None = None,
    policy_server_url: str | None = None,
    sim2sim_command: str | None = None,
) -> dict[str, Any]:
    job = Path(job_dir)
    ensure_dir(job)
    if provider_output_zip:
        imported = import_unitree_groot_n17_sonic_provider_output(
            provider_output_zip=provider_output_zip,
            extraction_dir=job / "provider_output",
            output_path=job / "unitree_groot_n17_sonic_policy_provider_import.json",
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": imported["status"],
            "job_dir": str(job),
            "policy_id": POLICY_ID,
            "unitree_groot_n17_sonic_model_executed": bool(
                imported.get("unitree_groot_n17_sonic_model_executed")
            ),
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                imported.get("unitree_groot_n17_sonic_policy_action_command_ran")
            ),
            "policy_action_model_command_ran": bool(imported.get("policy_action_model_command_ran")),
            "action": imported.get("action"),
            "blockers": imported.get("blockers", []),
            "truth_boundary": imported.get("truth_boundary", {}),
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        write_json(job / "unitree_groot_n17_sonic_policy_provider_smoke_summary.json", summary)
        return summary

    bundle = build_unitree_groot_n17_sonic_policy_provider_bundle(
        job_dir=job,
        frame_path=frame_path,
        policy_command=policy_command,
        n17_checkpoint=n17_checkpoint,
        sonic_checkpoint=sonic_checkpoint,
        groot_root=groot_root,
        wbc_root=wbc_root,
        policy_server_url=policy_server_url,
        sim2sim_command=sim2sim_command,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "dry_run_ready" if dry_run else "blocked_provider_launch_not_implemented",
        "job_dir": str(job),
        "policy_id": POLICY_ID,
        "bundle_manifest_path": bundle["manifest_path"],
        "bundle_path": bundle["bundle_path"],
        "ready_for_fresh_model_execution": bool(bundle.get("ready_for_fresh_model_execution")),
        "runtime_execution_blockers": list(bundle.get("runtime_execution_blockers", [])),
        "unitree_groot_n17_sonic_model_executed": False,
        "unitree_groot_n17_sonic_policy_action_command_ran": False,
        "policy_action_model_command_ran": False,
        "action": None,
        "blockers": [] if dry_run else ["blocked_provider_launch_not_implemented"],
        "truth_boundary": {
            "dry_run_is_not_model_execution": True,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job / "unitree_groot_n17_sonic_policy_provider_smoke_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--frame-path", type=Path, required=True)
    parser.add_argument("--provider-output-zip", type=Path)
    parser.add_argument("--policy-command")
    parser.add_argument("--n17-checkpoint")
    parser.add_argument("--sonic-checkpoint")
    parser.add_argument("--groot-root")
    parser.add_argument("--wbc-root")
    parser.add_argument("--policy-server-url")
    parser.add_argument("--sim2sim-command")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args(argv)
    summary = run_unitree_groot_n17_sonic_policy_provider_smoke(
        job_dir=args.job_dir,
        frame_path=args.frame_path,
        provider_output_zip=args.provider_output_zip,
        dry_run=args.dry_run,
        policy_command=args.policy_command or os.getenv(POLICY_COMMAND_ENV),
        n17_checkpoint=args.n17_checkpoint or os.getenv(N17_CHECKPOINT_ENV),
        sonic_checkpoint=args.sonic_checkpoint or os.getenv(SONIC_CHECKPOINT_ENV),
        groot_root=args.groot_root or os.getenv(GROOT_ROOT_ENV),
        wbc_root=args.wbc_root or os.getenv(WBC_ROOT_ENV),
        policy_server_url=args.policy_server_url or os.getenv(POLICY_SERVER_URL_ENV),
        sim2sim_command=args.sim2sim_command or os.getenv(SIM2SIM_COMMAND_ENV),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary.get("status") in {"completed", "dry_run_ready"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
