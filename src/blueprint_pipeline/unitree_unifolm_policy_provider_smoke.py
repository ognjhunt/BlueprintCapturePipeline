"""Run or import a bounded Unitree UnifoLM policy provider smoke.

This is the Unitree-native counterpart to the OpenVLA provider smoke. It proves
only that a Unitree UnifoLM VLA/WMA command produced a Blueprint-compatible
policy action from a Blueprint observation packet. It does not prove episode
success, dexterous manipulation, deployment readiness, or physical-robot
readiness.
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


SCHEMA_VERSION = "unitree_unifolm_policy_provider_smoke.v1"
OUTPUT_SCHEMA_VERSION = "unitree_unifolm_policy_provider_output.v1"
DEFAULT_TASK_ID = "contact_or_push_light_object"
DEFAULT_TASK_PROMPT = "move the Unitree G1 hand toward the light object and make controlled contact"
DEFAULT_BUNDLE_FILENAME = "unitree_unifolm_policy_provider_runtime_bundle.zip"
PROVIDER_OUTPUT_FILENAME = "unitree_unifolm_policy_provider_output.json"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _write_executable(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _copy_frame(frame_path: Path, output_path: Path) -> None:
    if not frame_path.is_file():
        raise FileNotFoundError(f"unitree_unifolm_input_frame_missing:{frame_path}")
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

OUTPUT_SCHEMA_VERSION = "unitree_unifolm_policy_provider_output.v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase(name: str, **fields: Any) -> None:
    print(
        "BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_PHASE:"
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
    payload_path = Path(os.environ.get("BLUEPRINT_UNITREE_UNIFOLM_POLICY_INPUT", runtime_dir / "policy_input.json"))
    output_path = Path(os.environ.get("BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_OUTPUT", runtime_dir / "unitree_unifolm_policy_provider_output.json"))
    mode = os.environ.get("BLUEPRINT_UNITREE_UNIFOLM_MODE", "vla")
    command = os.environ.get("BLUEPRINT_UNITREE_UNIFOLM_COMMAND", "")
    checkpoint = os.environ.get("BLUEPRINT_UNITREE_UNIFOLM_CHECKPOINT", "")
    vlm_checkpoint = os.environ.get("BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT", "")
    source_root = os.environ.get("BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT", "")
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        observation = _mapping(payload.get("observation"))
        visual_observation = _mapping(observation.get("visual_observation"))
        bundled_frame = runtime_dir / "input_frame.png"
        if bundled_frame.is_file():
            visual_observation["camera_frame_path"] = str(bundled_frame)
            observation["visual_observation"] = visual_observation
            payload["observation"] = observation
        _phase("invoke_unitree_unifolm_adapter", mode=mode, command_configured=bool(command), checkpoint_configured=bool(checkpoint))
        from blueprint_pipeline.unitree_unifolm_policy_command_adapter import run_unitree_unifolm_policy

        response, exit_code = run_unitree_unifolm_policy(
            payload=payload,
            mode=mode,
            command=command,
            checkpoint=checkpoint,
            vlm_checkpoint=vlm_checkpoint,
            source_root=source_root,
            timeout_seconds=float(os.environ.get("BLUEPRINT_UNITREE_UNIFOLM_TIMEOUT_SECONDS", "240")),
        )
        action = _mapping(response.get("action"))
        completed = exit_code == 0 and response.get("status") == "completed" and bool(action)
        output = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "completed" if completed else "blocked",
            "mode": mode,
            "unitree_unifolm_model_executed": bool(response.get("model_ran")),
            "unitree_unifolm_policy_action_command_ran": bool(response.get("unitree_unifolm_policy_action_command_ran")),
            "policy_action_model_command_ran": bool(response.get("unitree_unifolm_policy_action_command_ran")),
            "action": action or None,
            "adapter_response": response,
            "endpoint_closed_loop_policy_proven": False,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "blockers": [] if completed else list(response.get("blockers", []) or ["unitree_unifolm_provider_smoke_blocked"]),
        }
        _write_json(output_path, output)
        return 0 if completed else 2
    except Exception as exc:
        _write_json(
            output_path,
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "status": "failed",
                "mode": mode,
                "unitree_unifolm_model_executed": False,
                "unitree_unifolm_policy_action_command_ran": False,
                "policy_action_model_command_ran": False,
                "action": None,
                "traceback_tail": traceback.format_exc()[-4000:],
                "blockers": [f"unitree_unifolm_provider_runner_failed:{type(exc).__name__}"],
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
python3 unitree_unifolm_provider_runner.py
runner_rc=$?
if [ $runner_rc -ne 0 ] && [ ! -f unitree_unifolm_policy_provider_output.json ]; then
python3 - <<'PY'
import json
from pathlib import Path
Path("unitree_unifolm_policy_provider_output.json").write_text(json.dumps({
    "schema_version": "unitree_unifolm_policy_provider_output.v1",
    "status": "failed",
    "unitree_unifolm_model_executed": False,
    "unitree_unifolm_policy_action_command_ran": False,
    "policy_action_model_command_ran": False,
    "action": None,
    "blockers": [
        "unitree_unifolm_provider_runner_failed_without_runtime_result",
        "blocked_unitree_unifolm_process_exited_without_result"
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
        "unitree_unifolm_policy_command_adapter.py",
        "unitree_unifolm_vla_server_bridge.py",
    ):
        destination = package_dir / filename
        shutil.copy2(source_dir / filename, destination)
        copied.append(f"provider_runtime/blueprint_pipeline/{filename}")
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


def build_unitree_unifolm_policy_provider_bundle(
    *,
    job_dir: str | Path,
    frame_path: str | Path,
    mode: str = "vla",
    task_id: str = DEFAULT_TASK_ID,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    policy_command: str | None = None,
    checkpoint: str | None = None,
    vlm_checkpoint: str | None = None,
    source_root: str | None = None,
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
    runner_path = runtime_dir / "unitree_unifolm_provider_runner.py"
    _write_executable(runner_path, PROVIDER_RUNNER)
    run_script_path = runtime_dir / "run_unitree_unifolm_provider_runtime.sh"
    _write_executable(run_script_path, RUN_SCRIPT)
    bundled_blueprint_modules = _write_minimal_blueprint_runtime(runtime_dir)
    runtime_execution_blockers: list[str] = []
    if not policy_command:
        runtime_execution_blockers.append("blocked_missing_unitree_unifolm_policy_command")
    if not checkpoint:
        runtime_execution_blockers.append("blocked_missing_unitree_unifolm_policy_checkpoint")
    if mode == "vla" and not vlm_checkpoint:
        runtime_execution_blockers.append("blocked_missing_unitree_unifolm_vlm_checkpoint")
    manifest = {
        "schema_version": "unitree_unifolm_policy_provider_bundle.v1",
        "generated_at": utc_now_iso(),
        "status": "bundle_ready",
        "mode": mode,
        "policy_id": f"unitree_unifolm_{mode}_policy",
        "runtime_entrypoint": "provider_runtime/run_unitree_unifolm_provider_runtime.sh",
        "runner_path": str(runner_path),
        "policy_input_path": str(policy_input_path),
        "input_frame_path": str(frame_copy),
        "bundled_blueprint_modules": bundled_blueprint_modules,
        "policy_command_configured": bool(policy_command),
        "checkpoint_configured": bool(checkpoint),
        "vlm_checkpoint_configured": bool(vlm_checkpoint),
        "source_root_configured": bool(source_root),
        "local_bundle_ready_for_remote_staging": not runtime_execution_blockers,
        "ready_for_fresh_model_execution": not runtime_execution_blockers,
        "runtime_execution_blockers": runtime_execution_blockers,
        "expected_output_filename": PROVIDER_OUTPUT_FILENAME,
        "env_contract": {
            "BLUEPRINT_UNITREE_UNIFOLM_MODE": mode,
            "BLUEPRINT_UNITREE_UNIFOLM_COMMAND": "<configured>" if policy_command else None,
            "BLUEPRINT_UNITREE_UNIFOLM_CHECKPOINT": "<configured>" if checkpoint else None,
            "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT": "<configured>" if vlm_checkpoint else None,
            "BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT": "<configured>" if source_root else None,
        },
        "truth_boundary": {
            "provider_bundle_is_not_model_execution": True,
            "unitree_unifolm_policy_action_command_ran": False,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    manifest_path = runtime_dir / "unitree_unifolm_policy_provider_manifest.json"
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


def import_unitree_unifolm_provider_output(
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
            "unitree_unifolm_model_executed": False,
            "unitree_unifolm_policy_action_command_ran": False,
            "policy_action_model_command_ran": False,
            "action": None,
            "blockers": ["unitree_unifolm_provider_output_json_missing"],
        }
    else:
        value = json.loads(payload_path.read_text(encoding="utf-8"))
        payload = dict(value) if isinstance(value, Mapping) else {}
    completed = bool(
        payload.get("status") == "completed"
        and payload.get("unitree_unifolm_model_executed") is True
        and payload.get("unitree_unifolm_policy_action_command_ran") is True
        and isinstance(payload.get("action"), Mapping)
    )
    imported = {
        "schema_version": "unitree_unifolm_policy_provider_import.v1",
        "status": "completed" if completed else "blocked",
        "provider_output_zip": str(source_zip),
        "provider_output_path": str(payload_path) if payload_path else None,
        "unitree_unifolm_model_executed": bool(payload.get("unitree_unifolm_model_executed")),
        "unitree_unifolm_policy_action_command_ran": bool(
            payload.get("unitree_unifolm_policy_action_command_ran")
        ),
        "policy_action_model_command_ran": bool(payload.get("policy_action_model_command_ran")),
        "action": dict(payload["action"]) if isinstance(payload.get("action"), Mapping) else None,
        "mode": payload.get("mode"),
        "truth_boundary": {
            "provider_output_import_is_not_closed_loop_endpoint_control": True,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "blockers": list(payload.get("blockers", []) or ([] if completed else ["unitree_unifolm_provider_output_not_completed"])),
    }
    write_json(Path(output_path), imported)
    return imported


def run_unitree_unifolm_policy_provider_smoke(
    *,
    job_dir: str | Path,
    frame_path: str | Path,
    mode: str = "vla",
    provider_output_zip: str | Path | None = None,
    dry_run: bool = True,
    policy_command: str | None = None,
    checkpoint: str | None = None,
    vlm_checkpoint: str | None = None,
    source_root: str | None = None,
) -> dict[str, Any]:
    job = Path(job_dir)
    ensure_dir(job)
    if provider_output_zip:
        imported = import_unitree_unifolm_provider_output(
            provider_output_zip=provider_output_zip,
            extraction_dir=job / "provider_output",
            output_path=job / "unitree_unifolm_policy_provider_import.json",
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": imported["status"],
            "job_dir": str(job),
            "mode": imported.get("mode") or mode,
            "unitree_unifolm_model_executed": bool(imported.get("unitree_unifolm_model_executed")),
            "unitree_unifolm_policy_action_command_ran": bool(
                imported.get("unitree_unifolm_policy_action_command_ran")
            ),
            "policy_action_model_command_ran": bool(imported.get("policy_action_model_command_ran")),
            "action": imported.get("action"),
            "blockers": imported.get("blockers", []),
            "truth_boundary": imported.get("truth_boundary", {}),
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        write_json(job / "unitree_unifolm_policy_provider_smoke_summary.json", summary)
        return summary

    bundle = build_unitree_unifolm_policy_provider_bundle(
        job_dir=job,
        frame_path=frame_path,
        mode=mode,
        policy_command=policy_command,
        checkpoint=checkpoint,
        vlm_checkpoint=vlm_checkpoint,
        source_root=source_root,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "dry_run_ready" if dry_run else "blocked_provider_launch_not_implemented",
        "job_dir": str(job),
        "mode": mode,
        "bundle_manifest_path": bundle["manifest_path"],
        "bundle_path": bundle["bundle_path"],
        "ready_for_fresh_model_execution": bool(bundle.get("ready_for_fresh_model_execution")),
        "runtime_execution_blockers": list(bundle.get("runtime_execution_blockers", [])),
        "unitree_unifolm_model_executed": False,
        "unitree_unifolm_policy_action_command_ran": False,
        "policy_action_model_command_ran": False,
        "action": None,
        "blockers": [] if dry_run else ["blocked_provider_launch_not_implemented"],
        "truth_boundary": {
            "dry_run_is_not_model_execution": True,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job / "unitree_unifolm_policy_provider_smoke_summary.json", summary)
    return summary


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--frame-path", type=Path, required=True)
    parser.add_argument("--mode", choices=("vla", "wma"), default="vla")
    parser.add_argument("--provider-output-zip", type=Path)
    parser.add_argument("--policy-command")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vlm-checkpoint")
    parser.add_argument("--source-root")
    parser.add_argument("--dry-run", action="store_true", default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    summary = run_unitree_unifolm_policy_provider_smoke(
        job_dir=args.job_dir,
        frame_path=args.frame_path,
        mode=args.mode,
        provider_output_zip=args.provider_output_zip,
        dry_run=args.dry_run,
        policy_command=args.policy_command or os.getenv("BLUEPRINT_UNITREE_UNIFOLM_COMMAND"),
        checkpoint=args.checkpoint or os.getenv("BLUEPRINT_UNITREE_UNIFOLM_CHECKPOINT"),
        vlm_checkpoint=args.vlm_checkpoint or os.getenv("BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT"),
        source_root=args.source_root or os.getenv("BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT"),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary.get("status") in {"completed", "dry_run_ready"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
