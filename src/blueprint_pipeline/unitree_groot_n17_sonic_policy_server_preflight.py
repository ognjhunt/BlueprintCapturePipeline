"""Preflight the GR00T N1.7 + UNITREE_G1_SONIC PolicyServer lane.

The preflight is deliberately non-destructive: it verifies local source,
Python environment, embodiment schema, checkpoint sizing, and local execution
constraints without downloading model weights or launching robot hardware.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, Callable, Mapping, NamedTuple, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .unitree_groot_n17_sonic_policy_runtime import (
    DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT,
    GROOT_ROOT_ENV,
    N17_CHECKPOINT_ENV,
    POLICY_ID,
    POLICY_SERVER_URL_ENV,
    SIM2SIM_COMMAND_ENV,
    SONIC_CHECKPOINT_ENV,
    WBC_ROOT_ENV,
    configured_checkpoint_reference,
    probe_unitree_groot_n17_sonic_runtime,
    select_unitree_g1_sonic_policy_checkpoint,
    unitree_g1_sonic_checkpoint_provenance,
)


SCHEMA_VERSION = "unitree_groot_n17_sonic_policy_server_preflight.v1"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 5550
DEFAULT_MINIMUM_N17_DOWNLOAD_BYTES = 8 * 1024**3
SERVER_HELP_TIMEOUT_SECONDS = 20
PYTHON_IMPORT_TIMEOUT_SECONDS = 45


class CommandProbe(NamedTuple):
    args: Sequence[str]
    cwd: Path | None = None
    timeout_seconds: int = 30


CommandRunner = Callable[[CommandProbe], dict[str, Any]]
DiskUsageProvider = Callable[[Path], Any]


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _path_env(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else None


def _tail_text(value: str, *, max_chars: int = 4000) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _run_command(probe: CommandProbe) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(probe.args),
            cwd=probe.cwd,
            capture_output=True,
            check=False,
            text=True,
            timeout=probe.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ran": True,
            "timed_out": True,
            "returncode": None,
            "stdout_tail": _tail_text(exc.stdout or ""),
            "stderr_tail": _tail_text(exc.stderr or ""),
        }
    except Exception as exc:  # pragma: no cover - host dependent
        return {
            "ran": False,
            "timed_out": False,
            "returncode": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return {
        "ran": True,
        "timed_out": False,
        "returncode": completed.returncode,
        "stdout_tail": _tail_text(completed.stdout),
        "stderr_tail": _tail_text(completed.stderr),
    }


def _json_stdout(probe_result: Mapping[str, Any]) -> dict[str, Any] | None:
    if probe_result.get("returncode") != 0:
        return None
    text = _string(probe_result.get("stdout_tail"))
    if not text:
        return None
    try:
        value = json.loads(text.splitlines()[-1])
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _runtime_setup_audit_path(
    *,
    explicit_path: Path | None,
    groot_root: Path | None,
) -> Path | None:
    if explicit_path is not None:
        return explicit_path.expanduser()
    if groot_root is None:
        return None
    root = groot_root.expanduser()
    try:
        job_dir = root.parent.parent
    except IndexError:
        return None
    candidate = job_dir / "unitree_groot_n17_sonic_runtime_setup_audit.json"
    return candidate if candidate.is_file() else None


def _read_json_object(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _model_size_bytes(
    *,
    runtime_setup_audit: Mapping[str, Any],
    checkpoint_reference: str,
) -> tuple[int, dict[str, Any] | None]:
    models = runtime_setup_audit.get("huggingface_models")
    if isinstance(models, Mapping):
        model = models.get(checkpoint_reference)
        if isinstance(model, Mapping):
            total_size_gib = model.get("total_size_gib")
            if total_size_gib is not None:
                try:
                    size = int(float(total_size_gib) * 1024**3)
                    return size, dict(model)
                except (TypeError, ValueError):
                    pass
    return DEFAULT_MINIMUM_N17_DOWNLOAD_BYTES, None


def _server_venv_python(groot_root: Path | None) -> Path | None:
    if groot_root is None:
        return None
    return groot_root.expanduser() / ".venv" / "bin" / "python"


def _git_commit(root: Path | None, *, run_command: CommandRunner) -> str | None:
    if root is None or not root.is_dir():
        return None
    result = run_command(
        CommandProbe(
            args=("git", "-C", str(root), "rev-parse", "HEAD"),
            timeout_seconds=10,
        )
    )
    if result.get("returncode") == 0:
        commit = _string(result.get("stdout_tail")).splitlines()[-1]
        return commit or None
    return None


def _python_version_probe(
    python_path: Path | None,
    *,
    run_command: CommandRunner,
) -> dict[str, Any]:
    if python_path is None or not python_path.is_file():
        return {"status": "missing", "path": str(python_path) if python_path else None}
    result = run_command(
        CommandProbe(args=(str(python_path), "--version"), timeout_seconds=10)
    )
    return {
        "status": "ok" if result.get("returncode") == 0 else "failed",
        "path": str(python_path),
        "result": result,
    }


def _import_probe(
    python_path: Path | None,
    *,
    groot_root: Path | None,
    run_command: CommandRunner,
) -> dict[str, Any]:
    if python_path is None or not python_path.is_file():
        return {"status": "missing", "modules": {}}
    script = r"""
import importlib
import json

modules = {}
for name in [
    "torch",
    "transformers",
    "huggingface_hub",
    "gr00t",
    "zmq",
    "msgpack",
    "msgpack_numpy",
    "tyro",
]:
    try:
        module = importlib.import_module(name)
        modules[name] = {
            "imported": True,
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        modules[name] = {
            "imported": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
try:
    import torch

    cuda_available = bool(torch.cuda.is_available())
except Exception:
    cuda_available = False
print(json.dumps({"modules": modules, "torch_cuda_available": cuda_available}))
"""
    result = run_command(
        CommandProbe(
            args=(str(python_path), "-c", script),
            cwd=groot_root,
            timeout_seconds=PYTHON_IMPORT_TIMEOUT_SECONDS,
        )
    )
    parsed = _json_stdout(result)
    modules = dict(parsed.get("modules", {})) if parsed else {}
    missing = [
        name
        for name, value in modules.items()
        if isinstance(value, Mapping) and not value.get("imported")
    ]
    return {
        "status": "ok" if parsed and not missing else "failed",
        "result": result,
        "modules": modules,
        "missing_modules": missing,
        "torch_cuda_available": bool(parsed.get("torch_cuda_available")) if parsed else False,
    }


def _server_help_probe(
    python_path: Path | None,
    *,
    groot_root: Path | None,
    run_command: CommandRunner,
) -> dict[str, Any]:
    if python_path is None or not python_path.is_file() or groot_root is None:
        return {"status": "missing"}
    server_script = groot_root / "gr00t" / "eval" / "run_gr00t_server.py"
    if not server_script.is_file():
        return {"status": "missing", "script_path": str(server_script)}
    result = run_command(
        CommandProbe(
            args=(str(python_path), str(server_script), "--help"),
            cwd=groot_root,
            timeout_seconds=SERVER_HELP_TIMEOUT_SECONDS,
        )
    )
    stdout = _string(result.get("stdout_tail"))
    return {
        "status": "ok" if result.get("returncode") == 0 else "failed",
        "script_path": str(server_script),
        "result": result,
        "has_model_path_option": "--model-path" in stdout,
        "has_dataset_path_option": "--dataset-path" in stdout,
        "has_embodiment_tag_option": "--embodiment-tag" in stdout,
        "has_device_option": "--device" in stdout,
        "has_port_option": "--port" in stdout,
        "has_no_strict_option": "--no-strict" in stdout,
    }


def _embodiment_contract_probe(
    python_path: Path | None,
    *,
    groot_root: Path | None,
    run_command: CommandRunner,
) -> dict[str, Any]:
    if python_path is None or not python_path.is_file():
        return {"status": "missing"}
    script = r"""
import json
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag, POSTTRAIN_TAGS, PRETRAIN_TAGS

tag = EmbodimentTag.resolve("UNITREE_G1_SONIC")
config = MODALITY_CONFIGS.get(tag.value)

def keys(section):
    if config is None or section not in config:
        return []
    return list(getattr(config[section], "modality_keys", []))

def delta_indices(section):
    if config is None or section not in config:
        return []
    return list(getattr(config[section], "delta_indices", []))

print(json.dumps({
    "tag_name": tag.name,
    "tag_value": tag.value,
    "is_pretrain_tag": tag in PRETRAIN_TAGS,
    "is_posttrain_tag": tag in POSTTRAIN_TAGS,
    "modality_config_present": config is not None,
    "video_keys": keys("video"),
    "state_keys": keys("state"),
    "action_keys": keys("action"),
    "action_delta_indices": delta_indices("action"),
}))
"""
    result = run_command(
        CommandProbe(
            args=(str(python_path), "-c", script),
            cwd=groot_root,
            timeout_seconds=PYTHON_IMPORT_TIMEOUT_SECONDS,
        )
    )
    parsed = _json_stdout(result)
    expected_video = ["ego_view"]
    expected_state = [
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "projected_gravity",
    ]
    expected_action = ["motion_token", "left_hand_joints", "right_hand_joints"]
    matches = bool(
        parsed
        and parsed.get("tag_name") == "UNITREE_G1_SONIC"
        and parsed.get("tag_value") == "unitree_g1_sonic"
        and parsed.get("video_keys") == expected_video
        and parsed.get("state_keys") == expected_state
        and parsed.get("action_keys") == expected_action
    )
    return {
        "status": "ok" if matches else "failed",
        "result": result,
        "contract": parsed,
        "expected": {
            "video_keys": expected_video,
            "state_keys": expected_state,
            "action_keys": expected_action,
            "action_horizon": 40,
        },
    }


def _disk_probe(
    *,
    path: Path,
    required_bytes: int,
    disk_usage_provider: DiskUsageProvider,
) -> dict[str, Any]:
    usage = disk_usage_provider(path)
    free_bytes = int(getattr(usage, "free", 0))
    return {
        "path": str(path),
        "free_bytes": free_bytes,
        "minimum_required_bytes": required_bytes,
        "recommended_free_bytes": int(required_bytes * 1.5),
        "free_gib": round(free_bytes / 1024**3, 3),
        "minimum_required_gib": round(required_bytes / 1024**3, 3),
        "sufficient_for_minimum_download": free_bytes >= required_bytes,
    }


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _start_command(
    *,
    groot_root: Path | None,
    host: str,
    port: int,
    model_path: str | None = None,
) -> str:
    root = str(groot_root) if groot_root else f"${GROOT_ROOT_ENV}"
    model = model_path or f"${N17_CHECKPOINT_ENV}"
    return (
        f"cd {_shell_join([root])} && "
        ".venv/bin/python gr00t/eval/run_gr00t_server.py "
        f"--model-path {shlex.quote(model)} "
        "--embodiment-tag UNITREE_G1_SONIC "
        "--device cuda "
        f"--host {shlex.quote(host)} "
        f"--port {port} "
        "--no-strict"
    )


def run_unitree_groot_n17_sonic_policy_server_preflight(
    *,
    job_dir: str | Path,
    generated_at: str | None = None,
    runtime_setup_audit_path: str | Path | None = None,
    run_command: CommandRunner = _run_command,
    disk_usage_provider: DiskUsageProvider = shutil.disk_usage,
) -> dict[str, Any]:
    generated_at = generated_at or utc_now_iso()
    job = Path(job_dir)
    ensure_dir(job)

    groot_root = _path_env(GROOT_ROOT_ENV)
    wbc_root = _path_env(WBC_ROOT_ENV)
    n17_checkpoint = os.getenv(N17_CHECKPOINT_ENV, "").strip()
    (
        effective_n17_checkpoint,
        n17_checkpoint_selection_source,
        n17_default_experimental_checkpoint_applied,
    ) = select_unitree_g1_sonic_policy_checkpoint(n17_checkpoint)
    sonic_checkpoint = os.getenv(SONIC_CHECKPOINT_ENV, "").strip()
    policy_server_url = (
        os.getenv(POLICY_SERVER_URL_ENV, "").strip()
        or f"tcp://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT}"
    )
    sim2sim_command = os.getenv(SIM2SIM_COMMAND_ENV, "").strip()

    audit_path = _runtime_setup_audit_path(
        explicit_path=Path(runtime_setup_audit_path).expanduser()
        if runtime_setup_audit_path
        else None,
        groot_root=groot_root,
    )
    runtime_setup_audit = _read_json_object(audit_path)
    runtime_audit = probe_unitree_groot_n17_sonic_runtime(generated_at=generated_at)

    n17_configured, n17_reference, n17_exists, n17_kind = configured_checkpoint_reference(
        effective_n17_checkpoint
    )
    sonic_configured, sonic_reference, sonic_exists, sonic_kind = (
        configured_checkpoint_reference(sonic_checkpoint)
    )
    required_n17_bytes, model_metadata = _model_size_bytes(
        runtime_setup_audit=runtime_setup_audit,
        checkpoint_reference=effective_n17_checkpoint,
    )
    disk_path = job if job.exists() else Path.cwd()
    disk = _disk_probe(
        path=disk_path,
        required_bytes=required_n17_bytes,
        disk_usage_provider=disk_usage_provider,
    )

    python_path = _server_venv_python(groot_root)
    python_version = _python_version_probe(python_path, run_command=run_command)
    imports = _import_probe(python_path, groot_root=groot_root, run_command=run_command)
    server_help = _server_help_probe(
        python_path,
        groot_root=groot_root,
        run_command=run_command,
    )
    embodiment = _embodiment_contract_probe(
        python_path,
        groot_root=groot_root,
        run_command=run_command,
    )

    blockers = set(runtime_audit.get("blockers", []))
    if python_path is None or not python_path.is_file():
        blockers.add("blocked_missing_isaac_groot_server_venv_python")
    if python_version.get("status") != "ok":
        blockers.add("blocked_isaac_groot_server_python_unavailable")
    if imports.get("status") != "ok":
        blockers.add("blocked_isaac_groot_server_python_imports_failed")
    if server_help.get("status") != "ok":
        blockers.add("blocked_isaac_groot_run_gr00t_server_help_failed")
    if embodiment.get("status") != "ok":
        blockers.add("blocked_unitree_g1_sonic_embodiment_contract_not_verified")
    if not n17_configured:
        blockers.add(f"blocked_missing_{N17_CHECKPOINT_ENV}")
    elif n17_kind == "repo_id" and not disk["sufficient_for_minimum_download"]:
        blockers.add("blocked_insufficient_local_disk_for_gr00t_n17_checkpoint_download")
    elif n17_kind == "missing_path":
        blockers.add(f"blocked_missing_path_for_{N17_CHECKPOINT_ENV}")
    if not sonic_configured:
        blockers.add(f"blocked_missing_{SONIC_CHECKPOINT_ENV}")
    elif sonic_kind == "missing_path":
        blockers.add(f"blocked_missing_path_for_{SONIC_CHECKPOINT_ENV}")
    if not imports.get("torch_cuda_available"):
        blockers.add("blocked_no_local_cuda_gpu_for_gr00t_n17_policy_server")
    if not sim2sim_command:
        blockers.add(f"blocked_missing_{SIM2SIM_COMMAND_ENV}")

    policy_server_start_ready = not blockers
    status = "ready_to_start" if policy_server_start_ready else "blocked"
    artifact_path = job / "unitree_groot_n17_sonic_policy_server_preflight.json"
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_dir": str(job),
        "artifact_path": str(artifact_path),
        "status": status,
        "policy_id": POLICY_ID,
        "selected_candidate_id": POLICY_ID,
        "unitree_groot_n17_sonic_policy_action_command_ran": False,
        "unitree_policy_action_command_ran": False,
        "openvla_policy_action_command_ran": False,
        "policy_server_start_ready": policy_server_start_ready,
        "policy_server_url_for_blueprint_adapter": policy_server_url,
        "source_repositories": {
            "isaac_groot": {
                "path": str(groot_root) if groot_root else None,
                "exists": bool(groot_root and groot_root.is_dir()),
                "commit": _git_commit(groot_root, run_command=run_command),
            },
            "groot_wholebodycontrol": {
                "path": str(wbc_root) if wbc_root else None,
                "exists": bool(wbc_root and wbc_root.is_dir()),
                "commit": _git_commit(wbc_root, run_command=run_command),
            },
        },
        "server_python": python_version,
        "server_imports": imports,
        "run_gr00t_server_help": server_help,
        "unitree_g1_sonic_embodiment_contract": embodiment,
        "n17_checkpoint": {
            "env": N17_CHECKPOINT_ENV,
            "configured_reference": n17_checkpoint or None,
            "effective_reference": effective_n17_checkpoint,
            "selection_source": n17_checkpoint_selection_source,
            "default_experimental_checkpoint_applied": (
                n17_default_experimental_checkpoint_applied
            ),
            "resolved_reference": n17_reference,
            "configured": n17_configured,
            "exists_locally": n17_exists,
            "reference_kind": n17_kind,
            "huggingface_model_metadata": model_metadata,
            "distinguishes_repo_id_from_local_checkpoint": True,
            "provenance": unitree_g1_sonic_checkpoint_provenance(
                effective_n17_checkpoint
            ),
            "trusted_for_production": False,
            "task_specific_sink_handle_training_proven": False,
            "task_specific_finetuning_required_for_admission": False,
        },
        "sonic_checkpoint": {
            "env": SONIC_CHECKPOINT_ENV,
            "configured_reference": sonic_checkpoint or None,
            "resolved_reference": sonic_reference,
            "configured": sonic_configured,
            "exists_locally": sonic_exists,
            "reference_kind": sonic_kind,
        },
        "local_disk": disk,
        "runtime_setup_audit_path": str(audit_path) if audit_path else None,
        "runtime_setup_audit_used": bool(runtime_setup_audit),
        "runtime_installation_audit": runtime_audit,
        "exact_commands_once_blockers_are_resolved": {
            "start_gr00t_n17_sonic_policy_server": _start_command(
                groot_root=groot_root,
                host=DEFAULT_SERVER_HOST,
                port=DEFAULT_SERVER_PORT,
                model_path=effective_n17_checkpoint,
            ),
            "select_default_experimental_checkpoint": (
                f"export {N17_CHECKPOINT_ENV}="
                f"{DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT}"
            ),
            "run_blueprint_action_command_probe": (
                f"{POLICY_SERVER_URL_ENV}=tcp://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT} "
                "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND="
                "\"python -m blueprint_pipeline."
                "unitree_groot_n17_sonic_policy_server_command\" "
                "blueprint-unitree-groot-n17-sonic-policy-command-adapter"
            ),
            "run_bounded_closed_loop_eval": (
                "python -m blueprint_pipeline.mujoco_g1_wam_vla_policy_endpoint_eval "
                "--policy-lane unitree_groot_n17_sonic_policy "
                "--task-filter contact_or_push_light_object "
                "--max-tasks 1 --max-spawns 1 --steps-per-episode 24 "
                "--policy-interval-steps 12 --allow-policy-action-model-command-run"
            ),
        },
        "blockers": sorted(blockers),
        "claim_boundary": {
            "simulator_only": True,
            "preflight_is_not_policy_execution": True,
            "gr00t_n17_source_and_schema_probes_are_not_task_success": True,
            "policy_server_must_run_and_return_action_for_model_proof": True,
            "wam_evaluator_is_not_robot_policy": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "real_world_manipulation_success_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(artifact_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--runtime-setup-audit-path", type=Path)
    args = parser.parse_args(argv)
    report = run_unitree_groot_n17_sonic_policy_server_preflight(
        job_dir=args.job_dir,
        runtime_setup_audit_path=args.runtime_setup_audit_path,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "ready_to_start" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
