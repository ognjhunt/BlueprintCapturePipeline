"""GR00T N1.7 + UNITREE_G1_SONIC runtime discovery.

This is a simulator-only Blueprint boundary for the Unitree G1 SONIC lane. It
does not launch physical robot commands. A configured runtime means the source
roots and checkpoint references are present; a policy proof still requires the
separate action-command adapter to run and return real action payloads.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json


POLICY_ID = "unitree_groot_n17_sonic_policy"
SCHEMA_VERSION = "unitree_groot_n17_sonic_policy_runtime.v1"
AUDIT_SCHEMA_VERSION = "unitree_groot_n17_sonic_installation_audit.v1"
TRUTH_SCHEMA_VERSION = "unitree_groot_n17_sonic_policy_runtime_truth_boundary.v1"
OFFICIAL_SIM2SIM_SCHEMA_VERSION = "unitree_groot_n17_sonic_official_sim2sim_audit.v1"
OFFICIAL_LAUNCHER_PREFLIGHT_SCHEMA_VERSION = (
    "unitree_groot_n17_sonic_official_launcher_preflight.v1"
)
KNOWN_BASE_N17_MODEL_REPO = "nvidia/GR00T-N1.7-3B"
DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT = "LucaFrat/groot-bs16"
BASE_N17_WITHOUT_SONIC_SUPPORT_BLOCKER = (
    "blocked_BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT_points_to_base_GR00T_N17_without_UNITREE_G1_SONIC_support"
)

GROOT_ROOT_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT"
WBC_ROOT_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT"
N17_CHECKPOINT_ENV = "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT"
SONIC_CHECKPOINT_ENV = "BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT"
POLICY_COMMAND_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"
POLICY_SERVER_URL_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL"
POLICY_SERVER_HOST_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_HOST"
POLICY_SERVER_PORT_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_PORT"
POLICY_SERVER_TOKEN_FILE_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_TOKEN_FILE"
HF_TOKEN_FILE_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_HF_TOKEN_FILE"
SIM2SIM_COMMAND_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND"

ENV_VAR_NAMES = (
    GROOT_ROOT_ENV,
    WBC_ROOT_ENV,
    N17_CHECKPOINT_ENV,
    SONIC_CHECKPOINT_ENV,
    POLICY_COMMAND_ENV,
    POLICY_SERVER_URL_ENV,
    POLICY_SERVER_HOST_ENV,
    POLICY_SERVER_PORT_ENV,
    POLICY_SERVER_TOKEN_FILE_ENV,
    HF_TOKEN_FILE_ENV,
    SIM2SIM_COMMAND_ENV,
)

EXPECTED_GROOT_FILES = (
    "gr00t/eval/run_gr00t_server.py",
    "gr00t/eval/open_loop_eval.py",
    "scripts/deployment/standalone_inference_script.py",
)
EXPECTED_WBC_FILES = (
    "gear_sonic/scripts/launch_inference.py",
    "gear_sonic/scripts/launch_data_collection.py",
    "download_from_hf.py",
)
EXPECTED_OFFICIAL_SONIC_SIM_FILES = (
    "gear_sonic/scripts/launch_inference.py",
    "gear_sonic/scripts/run_vla_inference.py",
    "gear_sonic/scripts/run_sim_loop.py",
    "gear_sonic_deploy/deploy.sh",
    "install_scripts/install_inference.sh",
)
OFFICIAL_SONIC_REQUIRED_SIM_VENVS = (
    ".venv_inference",
    ".venv_sim",
)
OFFICIAL_SONIC_OPTIONAL_SIM_VENVS = (
    ".venv_data_collection",
)
OFFICIAL_SONIC_INFERENCE_INSTALL_MIN_FREE_GIB = 8.0
OFFICIAL_SONIC_INFERENCE_REQUIRED_IMPORTS = (
    "tyro",
    "gear_sonic",
    "gr00t",
)
OFFICIAL_SONIC_DEPLOY_ASSET_RELATIVE_PATHS = {
    "deploy_checkpoint_prefix": "policy/release/model",
    "deploy_decoder_onnx": "policy/release/model_decoder.onnx",
    "deploy_encoder_onnx": "policy/release/model_encoder.onnx",
    "deploy_observation_config": "policy/release/observation_config.yaml",
    "deploy_planner_onnx": "planner/target_vel/V2/planner_sonic.onnx",
}
OFFICIAL_SONIC_DEFAULT_MOTION_DATA_RELATIVE_PATH = "gear_sonic_deploy/reference/example"
OFFICIAL_SONIC_LINUX_DEPLOY_TOOLS = (
    "cmake",
    "clang",
    "clang++",
    "just",
    "git-lfs",
    "pkg-config",
    "nvcc",
    "ros2",
)
OFFICIAL_SONIC_RUNTIME_ENV_VARS = (
    "TensorRT_ROOT",
    "CUDAToolkit_ROOT",
    "CUDA_HOME",
    "onnxruntime_DIR",
)
OFFICIAL_SONIC_ONNXRUNTIME_COMMON_PATHS = (
    "/opt/onnxruntime",
    "/usr/local/onnxruntime",
    "/usr/lib/onnxruntime",
    "~/.local/onnxruntime",
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _path_env(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else None


def _command_available(command: str | None) -> bool:
    text = _string(command)
    if not text:
        return False
    try:
        parts = shlex.split(text)
    except ValueError:
        return False
    if not parts:
        return False
    executable = parts[0]
    return bool(shutil.which(executable) or Path(executable).expanduser().is_file())


def _is_repo_id_reference(value: str) -> bool:
    text = value.strip()
    if not text or text.startswith(("/", "./", "../", "~")):
        return False
    parts = text.split("/")
    return (
        len(parts) >= 2
        and all(part.strip() for part in parts[:2])
        and not any(part in {".", ".."} for part in parts[:2])
        and " " not in text
    )


def configured_checkpoint_reference(value: str | None) -> tuple[bool, str | None, bool, str | None]:
    text = _string(value)
    if not text:
        return False, None, False, None
    path = Path(text).expanduser()
    if path.exists():
        return True, str(path), True, "local_path"
    if _is_repo_id_reference(text):
        return True, text, False, "repo_id"
    return False, str(path), False, "missing_path"


def is_known_base_n17_without_unitree_g1_sonic_support(value: str | None) -> bool:
    return _string(value).rstrip("/") == KNOWN_BASE_N17_MODEL_REPO


def is_default_experimental_unitree_g1_sonic_checkpoint(value: str | None) -> bool:
    return (
        _string(value).rstrip("/")
        == DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT
    )


def select_unitree_g1_sonic_policy_checkpoint(
    value: str | None,
) -> tuple[str, str, bool]:
    """Return the checkpoint to use for simulator evaluation admission.

    A sink-task-specific policy is not an admission requirement. The checkpoint
    only needs to match the Unitree G1 SONIC embodiment/action interface. If the
    env is missing or points at the incompatible base GR00T model, use the
    documented third-party UNITREE_G1_SONIC fine-tune as an experimental
    simulator candidate and record that it is not production-trusted.
    """

    text = _string(value)
    if not text or is_known_base_n17_without_unitree_g1_sonic_support(text):
        return (
            DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT,
            "default_experimental_unitree_g1_sonic_checkpoint",
            True,
        )
    return text, "configured_checkpoint", False


def unitree_g1_sonic_checkpoint_provenance(value: str | None) -> dict[str, Any]:
    text = _string(value)
    if is_default_experimental_unitree_g1_sonic_checkpoint(text):
        return {
            "checkpoint_id": DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT,
            "source": "huggingface_user_model",
            "uploader": "LucaFrat",
            "base_model": KNOWN_BASE_N17_MODEL_REPO,
            "embodiment_tag": "UNITREE_G1_SONIC",
            "dataset": "LucaFrat/dataset_100",
            "license": "other",
            "allowed_for_experimental_simulator_evaluation": True,
            "trusted_for_production": False,
            "official_nvidia_or_unitree_checkpoint": False,
            "task_specific_sink_handle_training_proven": False,
            "task_specific_finetuning_required_for_admission": False,
            "action_interface_compatibility_is_admission_requirement": True,
            "model_card_url": (
                "https://huggingface.co/"
                f"{DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT}"
            ),
        }
    return {
        "checkpoint_id": text or None,
        "source": "configured_checkpoint",
        "allowed_for_experimental_simulator_evaluation": bool(text),
        "trusted_for_production": False,
        "official_nvidia_or_unitree_checkpoint": False,
        "task_specific_sink_handle_training_proven": False,
        "task_specific_finetuning_required_for_admission": False,
        "action_interface_compatibility_is_admission_requirement": True,
    }


def _root_probe(path: Path | None, expected_files: Sequence[str]) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "configured": False,
            "exists": False,
            "expected_files": list(expected_files),
            "expected_files_present": False,
            "missing_expected_files": list(expected_files),
        }
    root = path.expanduser()
    missing = [relative for relative in expected_files if not (root / relative).is_file()]
    return {
        "path": str(root),
        "configured": True,
        "exists": root.exists(),
        "expected_files": [str(root / relative) for relative in expected_files],
        "expected_files_present": bool(root.exists() and not missing),
        "missing_expected_files": missing,
    }


def _relative_file_status(root: Path, relative_paths: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": relative,
            "path": str(root / relative),
            "exists": (root / relative).is_file(),
        }
        for relative in relative_paths
    ]


def _venv_import_status(venv_python: Path, module_names: Sequence[str]) -> list[dict[str, Any]]:
    if not venv_python.is_file():
        return [
            {
                "module": module_name,
                "importable": False,
                "error_type": "missing_venv_python",
            }
            for module_name in module_names
        ]
    statuses: list[dict[str, Any]] = []
    for module_name in module_names:
        probe = (
            "import importlib.util, sys; "
            f"sys.exit(0 if importlib.util.find_spec({module_name!r}) else 1)"
        )
        try:
            result = subprocess.run(
                [str(venv_python), "-c", probe],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            statuses.append(
                {
                    "module": module_name,
                    "importable": False,
                    "check": "importlib_find_spec",
                    "error_type": "TimeoutExpired",
                    "error": "module spec check timed out after 10 seconds",
                }
            )
            continue
        status: dict[str, Any] = {
            "module": module_name,
            "importable": result.returncode == 0,
            "check": "importlib_find_spec",
        }
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout).strip().splitlines()
            status["error"] = stderr[-1][:500] if stderr else "import failed"
        statuses.append(status)
    return statuses


def _venv_status(
    root: Path,
    names: Sequence[str],
    *,
    import_checks: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    import_checks = import_checks or {}
    statuses: list[dict[str, Any]] = []
    for name in names:
        venv_root = root / name
        venv_python = venv_root / "bin" / "python"
        import_status = _venv_import_status(venv_python, import_checks[name]) if name in import_checks else []
        statuses.append(
            {
                "name": name,
                "path": str(venv_root),
                "exists": venv_root.is_dir(),
                "python_exists": venv_python.is_file(),
                "activate_exists": (venv_root / "bin" / "activate").is_file(),
                "required_imports": import_status,
                "required_imports_available": all(
                    bool(item.get("importable")) for item in import_status
                )
                if import_status
                else None,
            }
        )
    return statuses


def _bounded_command_probe(
    args: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    timeout_seconds: int = 20,
    max_output_chars: int = 2000,
) -> dict[str, Any]:
    command = [str(arg) for arg in args]
    executable = Path(command[0]).expanduser() if command else Path()
    executable_available = bool(
        command
        and (
            shutil.which(command[0])
            or executable.is_file()
            or command[0] in {"bash", "sh", "python"}
        )
    )
    if not command or not executable_available:
        return {
            "command": " ".join(shlex.quote(part) for part in command),
            "cwd": str(cwd) if cwd is not None else None,
            "ran": False,
            "available": False,
            "returncode": None,
            "timed_out": False,
            "stdout_excerpt": "",
            "stderr_excerpt": "",
        }
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(shlex.quote(part) for part in command),
            "cwd": str(cwd) if cwd is not None else None,
            "ran": True,
            "available": True,
            "returncode": None,
            "timed_out": True,
            "stdout_excerpt": (exc.stdout or "")[:max_output_chars],
            "stderr_excerpt": (exc.stderr or "")[:max_output_chars],
        }
    return {
        "command": " ".join(shlex.quote(part) for part in command),
        "cwd": str(cwd) if cwd is not None else None,
        "ran": True,
        "available": True,
        "returncode": result.returncode,
        "timed_out": False,
        "stdout_excerpt": (result.stdout or "")[:max_output_chars],
        "stderr_excerpt": (result.stderr or "")[:max_output_chars],
    }


def _toolchain_status(command_names: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "command": command_name,
            "available": bool(shutil.which(command_name)),
            "path": shutil.which(command_name),
        }
        for command_name in command_names
    ]


def _runtime_env_status(names: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "configured": bool(os.getenv(name, "").strip()),
            "value_redacted": "<configured>" if os.getenv(name, "").strip() else None,
        }
        for name in names
    ]


def _path_status(path: Path, *, kind: str, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "kind": kind,
        "path": str(path),
        "exists": path.is_dir() if kind == "directory" else path.is_file(),
    }


def _docker_daemon_probe(*, enabled: bool) -> dict[str, Any]:
    docker_path = shutil.which("docker")
    if not enabled or not docker_path:
        return {
            "checked": enabled,
            "docker_available": bool(docker_path),
            "docker_daemon_available": None,
        }
    try:
        result = subprocess.run(
            [docker_path, "info"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "checked": True,
            "docker_available": True,
            "docker_daemon_available": False,
            "error_type": type(exc).__name__,
        }
    return {
        "checked": True,
        "docker_available": True,
        "docker_daemon_available": result.returncode == 0,
    }


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _official_sonic_asset_status(
    *,
    sonic_checkpoint: Path | None,
    wbc_root: Path | None,
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    if sonic_checkpoint is not None:
        for role, relative in OFFICIAL_SONIC_DEPLOY_ASSET_RELATIVE_PATHS.items():
            path = sonic_checkpoint / relative
            if role == "deploy_checkpoint_prefix":
                statuses.append(
                    {
                        "role": role,
                        "kind": "checkpoint_prefix",
                        "path": str(path),
                        "exists": (path.with_name(path.name + "_decoder.onnx").is_file())
                        and (path.with_name(path.name + "_encoder.onnx").is_file()),
                    }
                )
            else:
                statuses.append(_path_status(path, kind="file", role=role))
    if wbc_root is not None:
        statuses.append(
            _path_status(
                wbc_root / OFFICIAL_SONIC_DEFAULT_MOTION_DATA_RELATIVE_PATH,
                kind="directory",
                role="deploy_motion_data_default",
            )
        )
    return statuses


def probe_unitree_groot_n17_sonic_official_sim2sim_runtime(
    *,
    generated_at: str | None = None,
    wbc_root: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect the official GR00T-WholeBodyControl SONIC sim launcher surface.

    This probe records local readiness only. It does not start tmux, MuJoCo,
    the C++ deploy loop, a policy server, or physical robot commands.
    """

    generated_at = generated_at or utc_now_iso()
    root = Path(wbc_root).expanduser() if wbc_root is not None else _path_env(WBC_ROOT_ENV)
    configured = root is not None
    root_exists = bool(root and root.exists())
    root_path = root if root is not None else Path()
    file_status = _relative_file_status(root_path, EXPECTED_OFFICIAL_SONIC_SIM_FILES) if root else []
    required_venv_status = (
        _venv_status(
            root_path,
            OFFICIAL_SONIC_REQUIRED_SIM_VENVS,
            import_checks={
                ".venv_inference": OFFICIAL_SONIC_INFERENCE_REQUIRED_IMPORTS,
            },
        )
        if root
        else []
    )
    optional_venv_status = _venv_status(root_path, OFFICIAL_SONIC_OPTIONAL_SIM_VENVS) if root else []
    missing_files = [item["relative_path"] for item in file_status if not item["exists"]]
    missing_required_venvs = [
        item["name"]
        for item in required_venv_status
        if not item["exists"] or not item["activate_exists"]
    ]
    missing_required_imports = [
        f"{item['name']}:{import_status['module']}"
        for item in required_venv_status
        for import_status in item.get("required_imports", [])
        if not import_status.get("importable")
    ]
    tmux_available = bool(shutil.which("tmux"))
    disk_free_gib = None
    if root and root.exists():
        usage = shutil.disk_usage(root)
        disk_free_gib = round(float(usage.free) / (1024.0**3), 3)

    blockers: list[str] = []
    if not configured:
        blockers.append(f"blocked_missing_{WBC_ROOT_ENV}")
    elif not root_exists:
        blockers.append(f"blocked_missing_path_for_{WBC_ROOT_ENV}")
    if missing_files:
        blockers.append("blocked_official_sonic_sim2sim_expected_files_missing")
    for name in missing_required_venvs:
        blockers.append(f"blocked_missing_groot_wholebodycontrol_{name.lstrip('.')}")
    for import_name in missing_required_imports:
        venv_name, module_name = import_name.split(":", maxsplit=1)
        blockers.append(
            f"blocked_groot_wholebodycontrol_{venv_name.lstrip('.')}_missing_import_{module_name}"
        )
    if (
        ".venv_inference" in missing_required_venvs
        and disk_free_gib is not None
        and disk_free_gib < OFFICIAL_SONIC_INFERENCE_INSTALL_MIN_FREE_GIB
    ):
        blockers.append(
            "blocked_local_disk_free_space_below_8g_for_official_sonic_inference_venv_install"
        )
    if not tmux_available:
        blockers.append("blocked_missing_tmux_for_official_sonic_launcher")

    ready = bool(configured and root_exists and not missing_files and not missing_required_venvs and tmux_available)
    ready = bool(ready and not missing_required_imports)
    return {
        "schema_version": OFFICIAL_SIM2SIM_SCHEMA_VERSION,
        "generated_at": generated_at,
        "policy_id": POLICY_ID,
        "status": "configured" if ready else "not_configured",
        "wbc_root_env": WBC_ROOT_ENV,
        "wbc_root_configured": configured,
        "wbc_root": str(root) if root is not None else None,
        "wbc_root_exists": root_exists,
        "expected_files": file_status,
        "missing_expected_files": missing_files,
        "required_virtualenvs": required_venv_status,
        "optional_virtualenvs": optional_venv_status,
        "missing_required_virtualenvs": missing_required_venvs,
        "missing_required_imports": missing_required_imports,
        "tmux_available": tmux_available,
        "disk_free_gib": disk_free_gib,
        "inference_venv_install_min_free_gib": OFFICIAL_SONIC_INFERENCE_INSTALL_MIN_FREE_GIB,
        "official_groot_wholebodycontrol_sim2sim_configured": ready,
        "official_groot_wholebodycontrol_sim2sim_used": False,
        "official_sonic_wbc_mapping_proven": False,
        "current_blueprint_direct_action_bridge_is_official_sonic_wbc": False,
        "safe_sim_only_launcher_command": (
            "python gear_sonic/scripts/launch_inference.py --sim --no-data-exporter "
            "--policy-host 127.0.0.1 --policy-port 5550 "
            "--embodiment-tag unitree_g1_sonic --prompt '<task prompt>'"
        ),
        "install_missing_inference_venv_command": "bash install_scripts/install_inference.sh",
        "physical_robot_launcher_intentionally_not_run": True,
        "simulator_only": True,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "simulator_only": True,
            "official_wbc_probe_is_not_policy_execution": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def probe_unitree_groot_n17_sonic_official_launcher_preflight(
    *,
    generated_at: str | None = None,
    wbc_root: str | Path | None = None,
    sonic_checkpoint: str | Path | None = None,
    run_help_commands: bool = True,
    check_docker_daemon: bool = False,
) -> dict[str, Any]:
    """Bounded preflight for the official SONIC launcher.

    This records whether the official launcher surface is locally invocable and
    which exact commands should be retried on a Linux/GPU runtime once a trusted
    UNITREE_G1_SONIC GR00T checkpoint is supplied. It never starts tmux,
    MuJoCo, the deploy loop, a policy server, or physical robot commands.
    """

    generated_at = generated_at or utc_now_iso()
    root = Path(wbc_root).expanduser() if wbc_root is not None else _path_env(WBC_ROOT_ENV)
    checkpoint_env_value = (
        str(sonic_checkpoint)
        if sonic_checkpoint is not None
        else os.getenv(SONIC_CHECKPOINT_ENV, "").strip()
    )
    checkpoint_path = Path(checkpoint_env_value).expanduser() if checkpoint_env_value else None
    local_checkpoint = checkpoint_path if checkpoint_path and checkpoint_path.exists() else None
    n17_checkpoint = os.getenv(N17_CHECKPOINT_ENV, "").strip()
    (
        effective_n17_checkpoint,
        n17_checkpoint_selection_source,
        n17_default_experimental_checkpoint_applied,
    ) = select_unitree_g1_sonic_policy_checkpoint(n17_checkpoint)
    known_base_n17 = is_known_base_n17_without_unitree_g1_sonic_support(n17_checkpoint)
    official_sim2sim_probe = probe_unitree_groot_n17_sonic_official_sim2sim_runtime(
        generated_at=generated_at,
        wbc_root=root,
    )

    deploy_dir = root / "gear_sonic_deploy" if root else None
    launch_script = root / "gear_sonic" / "scripts" / "launch_inference.py" if root else None
    vla_script = root / "gear_sonic" / "scripts" / "run_vla_inference.py" if root else None
    sim_script = root / "gear_sonic" / "scripts" / "run_sim_loop.py" if root else None
    deploy_script = deploy_dir / "deploy.sh" if deploy_dir else None
    inference_python = root / ".venv_inference" / "bin" / "python" if root else None
    sim_python = root / ".venv_sim" / "bin" / "python" if root else None
    asset_status = _official_sonic_asset_status(
        sonic_checkpoint=local_checkpoint,
        wbc_root=root,
    )
    missing_assets = [item["role"] for item in asset_status if not item["exists"]]
    help_probes: dict[str, Any] = {}
    if run_help_commands and root is not None:
        help_probes = {
            "launch_inference_help": _bounded_command_probe(
                [inference_python or sys.executable, launch_script or "", "--help"],
                cwd=root,
            ),
            "run_vla_inference_help": _bounded_command_probe(
                [inference_python or sys.executable, vla_script or "", "--help"],
                cwd=root,
            ),
            "run_sim_loop_help": _bounded_command_probe(
                [sim_python or sys.executable, sim_script or "", "--help"],
                cwd=root,
            ),
            "deploy_help": _bounded_command_probe(
                ["bash", deploy_script or "", "--help"],
                cwd=deploy_dir,
            ),
            "just_list": _bounded_command_probe(["just", "--list"], cwd=deploy_dir),
        }
    elif not run_help_commands:
        help_probes = {
            "skipped": True,
            "reason": "run_help_commands_false",
        }

    toolchain = _toolchain_status(OFFICIAL_SONIC_LINUX_DEPLOY_TOOLS)
    runtime_env = _runtime_env_status(OFFICIAL_SONIC_RUNTIME_ENV_VARS)
    onnxruntime_paths = [
        {
            "path": str(Path(path).expanduser()),
            "exists": Path(path).expanduser().is_dir(),
        }
        for path in OFFICIAL_SONIC_ONNXRUNTIME_COMMON_PATHS
    ]
    docker_probe = _docker_daemon_probe(enabled=check_docker_daemon)
    missing_tools = [item["command"] for item in toolchain if not item["available"]]
    missing_env = [item["name"] for item in runtime_env if not item["configured"]]
    onnxruntime_configured = any(item["exists"] for item in onnxruntime_paths) or bool(
        os.getenv("onnxruntime_DIR", "").strip()
    )
    launch_help_ok = all(
        bool(item.get("ran") and item.get("returncode") == 0 and not item.get("timed_out"))
        for item in help_probes.values()
        if isinstance(item, dict) and item.get("ran") is not None
    )
    checkpoint_prefix = (
        local_checkpoint / OFFICIAL_SONIC_DEPLOY_ASSET_RELATIVE_PATHS["deploy_checkpoint_prefix"]
        if local_checkpoint
        else Path("<BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT>") / "policy" / "release" / "model"
    )
    obs_config = (
        local_checkpoint
        / OFFICIAL_SONIC_DEPLOY_ASSET_RELATIVE_PATHS["deploy_observation_config"]
        if local_checkpoint
        else Path("<BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT>")
        / "policy"
        / "release"
        / "observation_config.yaml"
    )
    planner = (
        local_checkpoint / OFFICIAL_SONIC_DEPLOY_ASSET_RELATIVE_PATHS["deploy_planner_onnx"]
        if local_checkpoint
        else Path("<BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT>")
        / "planner"
        / "target_vel"
        / "V2"
        / "planner_sonic.onnx"
    )
    motion_data = (
        root / OFFICIAL_SONIC_DEFAULT_MOTION_DATA_RELATIVE_PATH
        if root
        else Path("<GR00T-WholeBodyControl>") / OFFICIAL_SONIC_DEFAULT_MOTION_DATA_RELATIVE_PATH
    )
    launcher_command = (
        "python gear_sonic/scripts/launch_inference.py --sim --no-data-exporter "
        "--policy-host 127.0.0.1 --policy-port 5550 "
        "--embodiment-tag unitree_g1_sonic --prompt '<task prompt>' "
        f"--deploy-checkpoint {shlex.quote(str(checkpoint_prefix))} "
        f"--deploy-obs-config {shlex.quote(str(obs_config))} "
        f"--deploy-planner {shlex.quote(str(planner))} "
        f"--deploy-motion-data {shlex.quote(str(motion_data))} "
        "--deploy-output-type all"
    )
    deploy_command = (
        "./deploy.sh --input-type zmq_manager --zmq-host localhost "
        f"--cp {shlex.quote(str(checkpoint_prefix))} "
        f"--obs-config {shlex.quote(str(obs_config))} "
        f"--planner {shlex.quote(str(planner))} "
        f"--motion-data {shlex.quote(str(motion_data))} "
        "--output-type all sim"
    )

    blockers: list[str] = []
    if root is None:
        blockers.append(f"blocked_missing_{WBC_ROOT_ENV}")
    elif not root.exists():
        blockers.append(f"blocked_missing_path_for_{WBC_ROOT_ENV}")
    if local_checkpoint is None:
        blockers.append(f"blocked_missing_local_path_for_{SONIC_CHECKPOINT_ENV}")
    if missing_assets:
        blockers.append("blocked_official_sonic_deploy_assets_missing")
    n17_effective_configured, _, _, _ = configured_checkpoint_reference(
        effective_n17_checkpoint
    )
    if not n17_effective_configured:
        blockers.append(
            "blocked_missing_embodiment_compatible_unitree_g1_sonic_gr00t_n17_policy_checkpoint"
        )
    if sys.platform == "darwin":
        blockers.append(
            "blocked_local_official_sonic_deploy_runtime_requires_linux_gpu_or_jetson_not_macos"
        )
    for command_name in missing_tools:
        if command_name in {"nvcc", "ros2", "pkg-config"}:
            blockers.append(f"blocked_local_official_sonic_deploy_missing_{command_name}")
    if "TensorRT_ROOT" in missing_env:
        blockers.append("blocked_local_official_sonic_deploy_missing_TensorRT_ROOT")
    if "CUDAToolkit_ROOT" in missing_env and "CUDA_HOME" in missing_env:
        blockers.append("blocked_local_official_sonic_deploy_missing_cuda_toolkit")
    if not onnxruntime_configured:
        blockers.append("blocked_local_official_sonic_deploy_missing_onnxruntime")

    return {
        "schema_version": OFFICIAL_LAUNCHER_PREFLIGHT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "policy_id": POLICY_ID,
        "status": "preflight_complete",
        "wbc_root_env": WBC_ROOT_ENV,
        "wbc_root": str(root) if root is not None else None,
        "wbc_root_exists": bool(root and root.exists()),
        "sonic_checkpoint_env": SONIC_CHECKPOINT_ENV,
        "sonic_checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "sonic_checkpoint_local_path_exists": bool(local_checkpoint),
        "n17_checkpoint_env": N17_CHECKPOINT_ENV,
        "n17_checkpoint_value_redacted": "<configured>" if n17_checkpoint else None,
        "n17_checkpoint_effective_reference": effective_n17_checkpoint,
        "n17_checkpoint_selection_source": n17_checkpoint_selection_source,
        "default_experimental_checkpoint_applied": n17_default_experimental_checkpoint_applied,
        "n17_checkpoint_known_base_model_without_unitree_g1_sonic_support": known_base_n17,
        "unitree_g1_sonic_policy_checkpoint_provenance": (
            unitree_g1_sonic_checkpoint_provenance(effective_n17_checkpoint)
        ),
        "embodiment_compatible_experimental_policy_checkpoint_available": (
            n17_effective_configured
        ),
        "trusted_task_finetuned_unitree_g1_sonic_policy_checkpoint_available": False,
        "task_specific_finetuning_required_for_admission": False,
        "policy_admission_requires_unitree_g1_sonic_action_interface": True,
        "official_sim2sim_probe": official_sim2sim_probe,
        "official_launcher_help_probes": help_probes,
        "official_launcher_help_probes_passed": launch_help_ok,
        "official_sonic_deploy_assets": asset_status,
        "missing_official_sonic_deploy_assets": missing_assets,
        "local_platform": sys.platform,
        "local_toolchain": toolchain,
        "local_runtime_env": runtime_env,
        "local_onnxruntime_common_paths": onnxruntime_paths,
        "docker_probe": docker_probe,
        "official_sonic_wbc_launcher_preflight_completed": True,
        "official_sonic_wbc_launcher_executed": False,
        "official_sonic_wbc_deploy_loop_executed": False,
        "official_sonic_wbc_tmux_session_started": False,
        "physical_robot_launcher_intentionally_not_run": True,
        "sim_launcher_intentionally_not_run": True,
        "launcher_not_executed_reason": (
            "local deploy substrate is macOS without CUDA/TensorRT/ONNX Runtime/ROS2; "
            "policy checkpoint trust and task competence remain evaluator outputs, not "
            "admission requirements"
        ),
        "exact_retry_commands_once_checkpoint_and_linux_gpu_runtime_are_available": {
            "start_groot_sonic_policy_server": (
                "cd $BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT && "
                "uv run python gr00t/eval/run_gr00t_server.py "
                f"--model-path {DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT} "
                "--embodiment-tag UNITREE_G1_SONIC --device cuda:0 --port 5550"
            ),
            "select_default_experimental_checkpoint": (
                f"export {N17_CHECKPOINT_ENV}="
                f"{DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT}"
            ),
            "run_official_sonic_wbc_launcher_sim_only": (
                "cd $BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT && " + launcher_command
            ),
            "run_official_deploy_direct_sim_only": (
                "cd $BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT/gear_sonic_deploy && "
                + deploy_command
            ),
            "run_blueprint_action_command_after_policy_server_is_up": (
                "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL=tcp://127.0.0.1:5550 "
                "blueprint-unitree-groot-n17-sonic-policy-command-adapter"
            ),
        },
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "simulator_only": True,
            "preflight_is_not_policy_execution": True,
            "preflight_is_not_sim2sim_execution": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "real_world_manipulation_success_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def probe_unitree_groot_n17_sonic_runtime(*, generated_at: str | None = None) -> dict[str, Any]:
    generated_at = generated_at or utc_now_iso()
    groot_root = _path_env(GROOT_ROOT_ENV)
    wbc_root = _path_env(WBC_ROOT_ENV)
    n17_checkpoint = os.getenv(N17_CHECKPOINT_ENV, "").strip()
    (
        effective_n17_checkpoint,
        n17_checkpoint_selection_source,
        n17_default_experimental_checkpoint_applied,
    ) = select_unitree_g1_sonic_policy_checkpoint(n17_checkpoint)
    sonic_checkpoint = os.getenv(SONIC_CHECKPOINT_ENV, "").strip()
    policy_command = os.getenv(POLICY_COMMAND_ENV, "").strip()
    policy_server_url = os.getenv(POLICY_SERVER_URL_ENV, "").strip()
    policy_server_host = os.getenv(POLICY_SERVER_HOST_ENV, "").strip()
    policy_server_port = os.getenv(POLICY_SERVER_PORT_ENV, "").strip()
    policy_server_token_file = os.getenv(POLICY_SERVER_TOKEN_FILE_ENV, "").strip()
    hf_token_file = os.getenv(HF_TOKEN_FILE_ENV, "").strip()
    sim2sim_command = os.getenv(SIM2SIM_COMMAND_ENV, "").strip()

    n17_configured, n17_reference, n17_exists, n17_kind = configured_checkpoint_reference(
        effective_n17_checkpoint
    )
    n17_known_base_without_sonic_support = (
        is_known_base_n17_without_unitree_g1_sonic_support(n17_checkpoint)
    )
    sonic_configured, sonic_reference, sonic_exists, sonic_kind = configured_checkpoint_reference(
        sonic_checkpoint
    )
    groot_probe = _root_probe(groot_root, EXPECTED_GROOT_FILES)
    wbc_probe = _root_probe(wbc_root, EXPECTED_WBC_FILES)
    policy_command_available = _command_available(policy_command)
    sim2sim_command_available = _command_available(sim2sim_command)
    official_sim2sim_probe = probe_unitree_groot_n17_sonic_official_sim2sim_runtime(
        generated_at=generated_at,
        wbc_root=wbc_root,
    )

    blockers: list[str] = []
    if not groot_probe["configured"]:
        blockers.append(f"blocked_missing_{GROOT_ROOT_ENV}")
    elif not groot_probe["exists"]:
        blockers.append(f"blocked_missing_path_for_{GROOT_ROOT_ENV}")
    elif not groot_probe["expected_files_present"]:
        blockers.append("blocked_isaac_groot_expected_runtime_files_missing")
    if not wbc_probe["configured"]:
        blockers.append(f"blocked_missing_{WBC_ROOT_ENV}")
    elif not wbc_probe["exists"]:
        blockers.append(f"blocked_missing_path_for_{WBC_ROOT_ENV}")
    elif not wbc_probe["expected_files_present"]:
        blockers.append("blocked_groot_wholebodycontrol_expected_runtime_files_missing")
    if not n17_configured:
        blockers.append(f"blocked_missing_path_for_{N17_CHECKPOINT_ENV}")
    if not sonic_checkpoint:
        blockers.append(f"blocked_missing_{SONIC_CHECKPOINT_ENV}")
    elif not sonic_configured:
        blockers.append(f"blocked_missing_path_for_{SONIC_CHECKPOINT_ENV}")

    runtime_configured = bool(
        groot_probe["exists"]
        and wbc_probe["exists"]
        and n17_configured
        and sonic_configured
        and not any("expected_runtime_files_missing" in item for item in blockers)
    )
    policy_server_command_selected = (
        "policy-server-command" in policy_command
        or "policy_server_command" in policy_command
    )
    policy_command_readiness_blockers: list[str] = []
    if policy_server_command_selected and not policy_server_url:
        policy_command_readiness_blockers.append(f"blocked_missing_{POLICY_SERVER_URL_ENV}")
    sim2sim_command_readiness_blockers: list[str] = []
    if runtime_configured and not sim2sim_command:
        sim2sim_command_readiness_blockers.append(f"blocked_missing_{SIM2SIM_COMMAND_ENV}")
    elif sim2sim_command and not sim2sim_command_available:
        sim2sim_command_readiness_blockers.append(
            f"blocked_unavailable_{SIM2SIM_COMMAND_ENV}"
        )
    ready_for_policy_action_command = bool(
        runtime_configured
        and policy_command
        and policy_command_available
        and not policy_command_readiness_blockers
    )
    ready_for_sim2sim = bool(runtime_configured and sim2sim_command and sim2sim_command_available)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "policy_id": POLICY_ID,
        "status": "configured" if runtime_configured else "not_configured",
        "runtime_configured": runtime_configured,
        "unitree_groot_n17_sonic_policy_configured": runtime_configured,
        "ready_for_policy_action_command": ready_for_policy_action_command,
        "ready_for_sim2sim": ready_for_sim2sim,
        "groot_root_env": GROOT_ROOT_ENV,
        "groot_root": groot_probe,
        "wbc_root_env": WBC_ROOT_ENV,
        "wbc_root": wbc_probe,
        "n17_checkpoint_env": N17_CHECKPOINT_ENV,
        "n17_checkpoint_original_env_reference": n17_checkpoint or None,
        "n17_checkpoint_effective_reference": effective_n17_checkpoint,
        "n17_checkpoint_selection_source": n17_checkpoint_selection_source,
        "default_experimental_checkpoint_applied": n17_default_experimental_checkpoint_applied,
        "n17_checkpoint_configured": n17_configured,
        "n17_checkpoint_path": n17_reference,
        "n17_checkpoint_exists": n17_exists,
        "n17_checkpoint_reference_kind": n17_kind,
        "n17_checkpoint_known_base_model_without_unitree_g1_sonic_support": (
            n17_known_base_without_sonic_support
        ),
        "n17_checkpoint_supports_unitree_g1_sonic_proven": bool(
            n17_configured and not n17_known_base_without_sonic_support
        ),
        "unitree_g1_sonic_policy_checkpoint_provenance": (
            unitree_g1_sonic_checkpoint_provenance(effective_n17_checkpoint)
        ),
        "embodiment_compatible_experimental_policy_checkpoint_available": (
            n17_configured
        ),
        "trusted_for_production": False,
        "task_specific_finetuning_required_for_admission": False,
        "policy_admission_requires_unitree_g1_sonic_action_interface": True,
        "unitree_g1_sonic_requires_finetuned_gr00t_checkpoint": True,
        "g1_sonic_checkpoint_env": SONIC_CHECKPOINT_ENV,
        "g1_sonic_checkpoint_configured": sonic_configured,
        "g1_sonic_checkpoint_path": sonic_reference,
        "g1_sonic_checkpoint_exists": sonic_exists,
        "g1_sonic_checkpoint_reference_kind": sonic_kind,
        "policy_command_env": POLICY_COMMAND_ENV,
        "policy_command_configured": bool(policy_command),
        "policy_command_available": policy_command_available,
        "policy_server_command_selected": policy_server_command_selected,
        "policy_command_readiness_blockers": policy_command_readiness_blockers,
        "policy_command_value_redacted": "<configured>" if policy_command else None,
        "policy_server_url_env": POLICY_SERVER_URL_ENV,
        "policy_server_url_configured": bool(policy_server_url),
        "policy_server_url_redacted": "<configured>" if policy_server_url else None,
        "policy_server_host_env": POLICY_SERVER_HOST_ENV,
        "policy_server_host_configured": bool(policy_server_host),
        "policy_server_host_redacted": "<configured>" if policy_server_host else None,
        "policy_server_port_env": POLICY_SERVER_PORT_ENV,
        "policy_server_port_configured": bool(policy_server_port),
        "policy_server_port_redacted": "<configured>" if policy_server_port else None,
        "policy_server_token_file_env": POLICY_SERVER_TOKEN_FILE_ENV,
        "policy_server_token_file_configured": bool(policy_server_token_file),
        "policy_server_token_file_path": policy_server_token_file or None,
        "policy_server_token_value_written_to_artifacts": False,
        "hf_token_file_env": HF_TOKEN_FILE_ENV,
        "hf_token_file_configured": bool(hf_token_file),
        "hf_token_file_path": hf_token_file or None,
        "hf_token_file_exists": bool(hf_token_file and Path(hf_token_file).expanduser().is_file()),
        "hf_token_value_written_to_artifacts": False,
        "sim2sim_command_env": SIM2SIM_COMMAND_ENV,
        "sim2sim_command_configured": bool(sim2sim_command),
        "sim2sim_command_available": sim2sim_command_available,
        "sim2sim_command_readiness_blockers": sim2sim_command_readiness_blockers,
        "sim2sim_command_value_redacted": "<configured>" if sim2sim_command else None,
        "official_sonic_sim2sim_runtime": official_sim2sim_probe,
        "official_groot_wholebodycontrol_sim2sim_configured": bool(
            official_sim2sim_probe.get("official_groot_wholebodycontrol_sim2sim_configured")
        ),
        "official_groot_wholebodycontrol_sim2sim_used": False,
        "official_sonic_wbc_mapping_proven": False,
        "official_sources": {
            "isaac_groot_repo": "https://github.com/NVIDIA/Isaac-GR00T",
            "groot_wholebodycontrol_repo": "https://github.com/NVlabs/GR00T-WholeBodyControl",
            "n17_base_model": "nvidia/GR00T-N1.7-3B",
            "sonic_embodiment_tag": "UNITREE_G1_SONIC",
        },
        "retry_commands_once_access_is_supplied": {
            "clone_isaac_groot": (
                "git clone https://github.com/NVIDIA/Isaac-GR00T.git "
                "$BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT"
            ),
            "clone_groot_wholebodycontrol": (
                "git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "
                "$BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT"
            ),
            "start_groot_sonic_policy_server": (
                "HF_HOME=<directory-containing-token-file> "
                "uv run python gr00t/eval/run_gr00t_server.py "
                f"--model-path {effective_n17_checkpoint} "
                "--embodiment-tag UNITREE_G1_SONIC --device cuda:0 --port 5550"
            ),
            "select_default_experimental_checkpoint": (
                f"export {N17_CHECKPOINT_ENV}={effective_n17_checkpoint}"
            ),
            "run_blueprint_action_command": (
                "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND="
                "blueprint-unitree-groot-n17-sonic-policy-server-command "
                "blueprint-unitree-groot-n17-sonic-policy-command-adapter"
            ),
            "run_blueprint_sim2sim_command": (
                "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND="
                "'python -m blueprint_pipeline.unitree_groot_n17_sonic_sim2sim_command' "
                "python -m blueprint_pipeline.unitree_groot_n17_sonic_sim2sim_command "
                "--job-dir robot_eval_jobs/<job_id>"
            ),
            "run_official_sonic_sim2sim_launcher": (
                "python gear_sonic/scripts/launch_inference.py --sim --no-data-exporter "
                "--policy-host 127.0.0.1 --policy-port 5550 "
                "--embodiment-tag unitree_g1_sonic --prompt '<task prompt>'"
            ),
            "run_blueprint_probe": (
                "python -m blueprint_pipeline.mujoco_g1_wam_vla_policy_endpoint_eval "
                "--allow-policy-action-model-command-run"
            ),
        },
        "distinguishes_demo_assets_from_policy_checkpoints": True,
        "blockers": sorted(
            set(
                blockers
                + policy_command_readiness_blockers
                + sim2sim_command_readiness_blockers
            )
        ),
        "claim_boundary": {
            "simulator_only": True,
            "runtime_configuration_is_not_policy_execution": True,
            "unitree_g1_sonic_requires_finetuned_checkpoint_for_action_interface": True,
            "sink_task_specific_finetuning_required_for_policy_attempt": False,
            "third_party_checkpoint_trusted_for_production": False,
            "policy_action_command_required_for_backend_proof": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def build_unitree_groot_n17_sonic_runtime_truth_boundary(
    *,
    audit: Mapping[str, Any],
    policy_action_command_result: Mapping[str, Any] | None = None,
    sim2sim_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(policy_action_command_result or {})
    sim2sim = dict(sim2sim_result or {})
    action_ran = bool(result.get("unitree_groot_n17_sonic_policy_action_command_ran"))
    sim2sim_ran = bool(
        result.get("unitree_groot_n17_sonic_sim2sim_command_ran")
        or result.get("sim2sim_command_ran")
        or sim2sim.get("unitree_groot_n17_sonic_sim2sim_command_ran")
        or sim2sim.get("sim2sim_command_ran")
    )
    sim2sim_consumed_action = bool(
        sim2sim.get("unitree_groot_n17_sonic_action_chunk_consumed")
        or result.get("unitree_groot_n17_sonic_action_chunk_consumed")
    )
    return {
        "schema_version": TRUTH_SCHEMA_VERSION,
        "generated_at": audit.get("generated_at") or utc_now_iso(),
        "policy_id": POLICY_ID,
        "selected_candidate_id": POLICY_ID
        if action_ran or audit.get("unitree_groot_n17_sonic_policy_configured")
        else None,
        "unitree_groot_n17_sonic_policy_configured": bool(
            audit.get("unitree_groot_n17_sonic_policy_configured")
        ),
        "unitree_groot_n17_sonic_policy_action_command_ran": action_ran,
        "unitree_policy_action_command_ran": action_ran,
        "unitree_specific_manipulation_candidate_ran": action_ran,
        "openvla_policy_action_command_ran": False,
        "policy_server_url_configured": bool(audit.get("policy_server_url_configured")),
        "sim2sim_command_configured": bool(audit.get("sim2sim_command_configured")),
        "sim2sim_command_ran": sim2sim_ran,
        "unitree_groot_n17_sonic_sim2sim_command_ran": sim2sim_ran,
        "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim": (
            sim2sim_consumed_action
        ),
        "real_checkpoint_runtime_action_required": True,
        "provider_output_replay_used": bool(result.get("provider_output_replay_used")),
        "provider_output_replay_is_not_fresh_model_proof": bool(
            result.get("provider_output_replay_used")
        ),
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "safety_validation_proven": False,
        "real_world_manipulation_success_proven": False,
        "claim_boundary": {
            "simulator_only": True,
            "gr00t_sonic_action_command_is_single_step_not_task_success": True,
            "wam_evaluator_is_not_robot_policy": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
        "blockers": list(audit.get("blockers", [])) if not action_ran else [],
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def run_unitree_groot_n17_sonic_policy_runtime(
    *,
    job_dir: str | Path,
    generated_at: str | None = None,
    run_launcher_help_commands: bool = False,
) -> dict[str, Any]:
    job = Path(job_dir)
    ensure_dir(job)
    generated_at = generated_at or utc_now_iso()
    audit = probe_unitree_groot_n17_sonic_runtime(generated_at=generated_at)
    launcher_preflight = probe_unitree_groot_n17_sonic_official_launcher_preflight(
        generated_at=generated_at,
        run_help_commands=run_launcher_help_commands,
    )
    policy_action_result = _read_json_mapping(job / "policy_action_model_command_output.json")
    sim2sim_result = _read_json_mapping(job / "unitree_groot_n17_sonic_sim2sim_execution.json")
    truth = build_unitree_groot_n17_sonic_runtime_truth_boundary(
        audit=audit,
        policy_action_command_result=policy_action_result,
        sim2sim_result=sim2sim_result,
    )
    action_ran = bool(truth["unitree_groot_n17_sonic_policy_action_command_ran"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_dir": str(job),
        "status": "action_command_evidence_present" if action_ran else audit["status"],
        "policy_id": POLICY_ID,
        "selected_candidate_id": truth["selected_candidate_id"],
        "unitree_groot_n17_sonic_policy_configured": truth[
            "unitree_groot_n17_sonic_policy_configured"
        ],
        "unitree_groot_n17_sonic_policy_action_command_ran": truth[
            "unitree_groot_n17_sonic_policy_action_command_ran"
        ],
        "unitree_policy_action_command_ran": truth["unitree_policy_action_command_ran"],
        "unitree_specific_manipulation_candidate_ran": truth[
            "unitree_specific_manipulation_candidate_ran"
        ],
        "openvla_policy_action_command_ran": truth["openvla_policy_action_command_ran"],
        "unitree_groot_n17_sonic_sim2sim_command_ran": truth[
            "unitree_groot_n17_sonic_sim2sim_command_ran"
        ],
        "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim": truth[
            "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim"
        ],
        "ready_for_policy_action_command": audit["ready_for_policy_action_command"],
        "ready_for_sim2sim": audit["ready_for_sim2sim"],
        "installation_audit_path": str(job / "unitree_groot_n17_sonic_installation_audit.json"),
        "truth_boundary_path": str(
            job / "unitree_groot_n17_sonic_policy_runtime_truth_boundary.json"
        ),
        "official_launcher_preflight_path": str(
            job / "unitree_groot_n17_sonic_official_launcher_preflight.json"
        ),
        "blockers": audit["blockers"],
        "runtime_configuration_blockers": audit["blockers"],
        "claim_boundary": truth["claim_boundary"],
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job / "unitree_groot_n17_sonic_installation_audit.json", audit)
    write_json(
        job / "unitree_groot_n17_sonic_official_launcher_preflight.json",
        launcher_preflight,
    )
    write_json(job / "unitree_groot_n17_sonic_policy_runtime_truth_boundary.json", truth)
    write_json(job / "unitree_groot_n17_sonic_policy_runtime_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument(
        "--run-official-launcher-help-probes",
        action="store_true",
        help=(
            "Run bounded --help probes for the official SONIC launcher scripts. "
            "This does not launch tmux, MuJoCo, deploy, policy server, or robot commands."
        ),
    )
    args = parser.parse_args(argv)
    summary = run_unitree_groot_n17_sonic_policy_runtime(
        job_dir=args.job_dir,
        run_launcher_help_commands=args.run_official_launcher_help_probes,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary.get("status") == "configured" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
