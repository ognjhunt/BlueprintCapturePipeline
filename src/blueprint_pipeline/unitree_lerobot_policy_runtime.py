"""Unitree G1 LeRobot sim policy runtime lane.

This module is the job-level boundary for Unitree LeRobot G1 policy evaluation.
It probes local configuration, builds the Unitree LeRobot sim-eval command, and
optionally runs that command while keeping LeRobot/VLA/WAM claims separate from
the already-proven official Unitree RL Gym locomotion lane.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, parse_bool, utc_now_iso, write_json
from .unitree_groot_n17_sonic_policy_runtime import (
    GROOT_ROOT_ENV,
    N17_CHECKPOINT_ENV,
    POLICY_COMMAND_ENV as GROOT_POLICY_COMMAND_ENV,
    POLICY_ID as GROOT_POLICY_ID,
    POLICY_SERVER_URL_ENV,
    SIM2SIM_COMMAND_ENV,
    SONIC_CHECKPOINT_ENV,
    WBC_ROOT_ENV,
    probe_unitree_groot_n17_sonic_runtime,
)


SCHEMA_VERSION = "unitree_lerobot_g1_policy_runtime.v1"
TRUTH_SCHEMA_VERSION = "unitree_lerobot_g1_policy_runtime_truth_boundary.v1"
HANDOFF_SCHEMA_VERSION = "unitree_lerobot_g1_policy_handoff_manifest.v1"
PROBE_SCHEMA_VERSION = "unitree_lerobot_g1_policy_runtime_probe.v1"
DEFAULT_DATASET_REPO_ID = "unitreerobotics/G1_Dex3_ToastedBread_Dataset"
DEFAULT_POLICY_FAMILY = "unitree_lerobot"
DEFAULT_ARM = "G1_29"
DEFAULT_EE = "dex3"
DEFAULT_FREQUENCY = 30
DEFAULT_EPISODES = 0
DEFAULT_MAX_EPISODES = 1200
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_RUNTIME_SMOKE_TIMEOUT_SECONDS = 60.0
RUNTIME_MODES = ("probe", "dry_run", "sim_eval", "not_configured")
POLICY_FAMILIES = (
    "unitree_lerobot",
    "pi05",
    "groot",
    "unitree_groot_n17_sonic",
    "unitree_vla",
    "unifolm_vla",
    "unifolm_wma",
    "openvla_endpoint",
    "unknown",
)
VLA_POLICY_FAMILIES = {
    "pi05",
    "groot",
    "unitree_vla",
    "unifolm_vla",
    "openvla_endpoint",
}
HAND_EES = {"dex3", "dex1", "inspire1", "brainco"}
ENV_VAR_NAMES = (
    "BLUEPRINT_UNITREE_LEROBOT_ROOT",
    "BLUEPRINT_UNITREE_LEROBOT_PYTHON",
    "BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH",
    "BLUEPRINT_UNITREE_LEROBOT_DATASET_REPO_ID",
    "BLUEPRINT_UNITREE_LEROBOT_DATASET_ROOT",
    "BLUEPRINT_UNITREE_LEROBOT_TASK",
    "BLUEPRINT_UNITREE_POLICY_FAMILY",
    "BLUEPRINT_UNITREE_G1_ARM",
    "BLUEPRINT_UNITREE_G1_EE",
    "BLUEPRINT_UNITREE_LEROBOT_FREQUENCY",
    "BLUEPRINT_UNITREE_LEROBOT_EPISODES",
    "BLUEPRINT_UNITREE_LEROBOT_MAX_EPISODES",
    "BLUEPRINT_UNITREE_LEROBOT_VISUALIZATION",
    "BLUEPRINT_UNITREE_LEROBOT_SAVE_DATA",
    "BLUEPRINT_UNITREE_LEROBOT_TASK_DIR",
    "BLUEPRINT_UNITREE_LEROBOT_SMOKE_TIMEOUT_SECONDS",
    "BLUEPRINT_UNITREE_LEROBOT_SEND_REAL_ROBOT",
    "BLUEPRINT_UNITREE_ALLOW_REAL_ROBOT_COMMANDS",
    "BLUEPRINT_UNITREE_ALLOW_DOWNLOADS",
    "BLUEPRINT_UNITREE_OPENVLA_ENDPOINT_URL",
    "BLUEPRINT_UNITREE_WAM_ENDPOINT_URL",
    GROOT_ROOT_ENV,
    WBC_ROOT_ENV,
    N17_CHECKPOINT_ENV,
    SONIC_CHECKPOINT_ENV,
    GROOT_POLICY_COMMAND_ENV,
    POLICY_SERVER_URL_ENV,
    SIM2SIM_COMMAND_ENV,
)
UNITREE_ACTION_COMMAND_CANDIDATES = (
    {
        "candidate_id": "unitree_g1_policy",
        "command_envs": (
            "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
            "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        ),
        "checkpoint_envs": ("BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT",),
        "runtime_role": "unitree_g1_locomotion_action_command",
    },
    {
        "candidate_id": "unitree_lerobot_policy",
        "command_envs": ("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",),
        "checkpoint_envs": ("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",),
        "runtime_role": "unitree_g1_manipulation_action_command",
        "source_root_env": "BLUEPRINT_UNITREE_LEROBOT_ROOT",
    },
    {
        "candidate_id": GROOT_POLICY_ID,
        "command_envs": (GROOT_POLICY_COMMAND_ENV,),
        "checkpoint_envs": (SONIC_CHECKPOINT_ENV,),
        "extra_required_checkpoint_envs": (N17_CHECKPOINT_ENV,),
        "runtime_role": "unitree_groot_n17_sonic_manipulation_action_command",
        "source_root_env": GROOT_ROOT_ENV,
        "extra_required_root_envs": (WBC_ROOT_ENV,),
    },
    {
        "candidate_id": "unitree_unifolm_vla_policy",
        "command_envs": ("BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",),
        "checkpoint_envs": ("BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT",),
        "extra_required_checkpoint_envs": ("BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT",),
        "runtime_role": "unitree_native_vla_action_command",
        "source_root_env": "BLUEPRINT_UNITREE_UNIFOLM_VLA_SOURCE_ROOT",
    },
    {
        "candidate_id": "unitree_unifolm_wma_policy",
        "command_envs": ("BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",),
        "checkpoint_envs": ("BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT",),
        "runtime_role": "unitree_native_wma_action_command",
        "source_root_env": "BLUEPRINT_UNITREE_UNIFOLM_WMA_SOURCE_ROOT",
    },
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _path_env(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else None


def _json_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _is_hf_or_repo_id(value: str) -> bool:
    text = _string(value)
    if not text:
        return False
    if text.startswith(("/", "./", "../", "~")):
        return False
    return "/" in text


def _policy_path_configured(value: str) -> bool:
    text = _string(value)
    if not text:
        return False
    path = Path(text).expanduser()
    return path.exists() or _is_hf_or_repo_id(text)


def _policy_or_endpoint_configured(config: "UnitreeLeRobotPolicyRuntimeConfig") -> bool:
    endpoint_based = config.normalized_policy_family == "openvla_endpoint"
    endpoint_configured = bool(config.openvla_endpoint_url)
    return _policy_path_configured(config.policy_path) or (endpoint_based and endpoint_configured)


def _lerobot_configuration_stage(
    *,
    source_runtime_configured: bool,
    policy_or_endpoint_configured: bool,
    safety_blocks: Sequence[str],
) -> str:
    if safety_blocks:
        return "blocked_by_safety"
    if source_runtime_configured and policy_or_endpoint_configured:
        return "ready_for_sim_eval"
    if source_runtime_configured:
        return "source_runtime_ready_policy_missing"
    if policy_or_endpoint_configured:
        return "policy_configured_source_runtime_missing"
    return "not_configured"


def _command_available(command: str) -> bool:
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


def _configured_path_or_repo(value: str) -> tuple[bool, str | None, bool]:
    text = _string(value)
    if not text:
        return False, None, False
    path = Path(text).expanduser()
    exists = path.exists()
    return exists or _is_hf_or_repo_id(text), str(path) if exists else text, exists


def _repo_head(path: Path | None) -> str | None:
    if path is None or not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _python_executable_value(value: str | None) -> str:
    text = _string(value) or sys.executable
    if text.startswith(("~", "/", "./", "../")):
        return str(Path(text).expanduser())
    return text


def _python_executable_available(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("~", "/", "./", "../")):
        return Path(value).expanduser().is_file()
    return shutil.which(value) is not None


def _prepend_env_path(existing: str, entries: Sequence[str]) -> str:
    seen: set[str] = set()
    merged: list[str] = []
    for entry in [*entries, *existing.split(os.pathsep)]:
        text = entry.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return os.pathsep.join(merged)


def _unitree_lerobot_pythonpath_entries(
    config: UnitreeLeRobotPolicyRuntimeConfig,
) -> list[str]:
    if config.source_root is None:
        return []
    source_root = config.source_root
    candidates = [
        source_root,
        source_root / "unitree_lerobot" / "lerobot" / "src",
    ]
    return [str(path) for path in candidates if path.exists()]


def _unitree_lerobot_eval_script_smoke_probe(
    *,
    config: UnitreeLeRobotPolicyRuntimeConfig,
    script: Path | None = None,
    timeout_seconds: float = DEFAULT_RUNTIME_SMOKE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    script = script or config.preferred_sim_eval_script()
    if script is None:
        return {
            "status": "not_run",
            "passed": False,
            "reason": "unitree_lerobot_eval_g1_sim_script_missing",
            "command": [],
            "return_code": None,
            "stdout_size_bytes": 0,
            "stderr_size_bytes": 0,
            "runtime_error_summary": "unitree_lerobot_eval_g1_sim_script_missing",
            "python_executable": config.python_executable,
            "python_executable_available": _python_executable_available(
                config.python_executable
            ),
            "pythonpath_entries": [],
            "timeout_seconds": timeout_seconds,
        }
    command = [config.python_executable, str(script), "--help"]
    cwd = config.source_root or script.parent
    pythonpath_entries = _unitree_lerobot_pythonpath_entries(config)
    env = dict(os.environ)
    if pythonpath_entries:
        env["PYTHONPATH"] = _prepend_env_path(
            env.get("PYTHONPATH", ""),
            pythonpath_entries,
        )
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "passed": False,
            "reason": "timeout",
            "command": command,
            "cwd": str(cwd),
            "return_code": None,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout_size_bytes": 0,
            "stderr_size_bytes": 0,
            "runtime_error_summary": "unitree_lerobot_eval_g1_sim_help_timeout",
            "python_executable": config.python_executable,
            "python_executable_available": _python_executable_available(
                config.python_executable
            ),
            "pythonpath_entries": pythonpath_entries,
            "timeout_seconds": timeout_seconds,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "passed": False,
            "reason": type(exc).__name__,
            "command": command,
            "cwd": str(cwd),
            "return_code": None,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout_size_bytes": 0,
            "stderr_size_bytes": 0,
            "runtime_error_summary": f"{type(exc).__name__}: {str(exc)[:300]}",
            "python_executable": config.python_executable,
            "python_executable_available": _python_executable_available(
                config.python_executable
            ),
            "pythonpath_entries": pythonpath_entries,
            "timeout_seconds": timeout_seconds,
        }
    stderr = result.stderr or ""
    stdout = result.stdout or ""
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "passed": result.returncode == 0,
        "reason": None if result.returncode == 0 else "nonzero_exit",
        "command": command,
        "cwd": str(cwd),
        "return_code": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stdout_size_bytes": len(stdout),
        "stderr_size_bytes": len(stderr),
        "stderr_tail": stderr[-1000:] if stderr else "",
        "python_executable": config.python_executable,
        "python_executable_available": _python_executable_available(
            config.python_executable
        ),
        "pythonpath_entries": pythonpath_entries,
        "pythonpath_local_source_enabled": bool(pythonpath_entries),
        "timeout_seconds": timeout_seconds,
        "runtime_error_summary": None
        if result.returncode == 0
        else f"unitree_lerobot_eval_g1_sim_help_exited_{result.returncode}",
    }


def _collect_artifacts(task_dir: Path) -> dict[str, Any]:
    videos: list[str] = []
    traces: list[str] = []
    metrics: list[str] = []
    other: list[str] = []
    if not task_dir.exists():
        return {
            "task_dir_exists": False,
            "videos": videos,
            "traces": traces,
            "metrics": metrics,
            "other": other,
        }
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        resolved = _json_path(path)
        if suffix in {".mp4", ".mov", ".mkv", ".gif", ".webm"}:
            videos.append(resolved)
        elif suffix in {".jsonl", ".npz", ".npy", ".csv"} or "trace" in path.name.lower():
            traces.append(resolved)
        elif path.name.lower() in {"metrics.json", "policy_metrics.json"}:
            metrics.append(resolved)
        elif suffix == ".json" and "metric" in path.name.lower():
            metrics.append(resolved)
        else:
            other.append(resolved)
    return {
        "task_dir_exists": True,
        "videos": videos,
        "traces": traces,
        "metrics": metrics,
        "other": other,
    }


@dataclass(frozen=True)
class UnitreeLeRobotPolicyRuntimeConfig:
    root: Path | None
    python_executable: str = sys.executable
    policy_path: str = ""
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID
    dataset_root: str = ""
    task: str = ""
    policy_family: str = DEFAULT_POLICY_FAMILY
    arm: str = DEFAULT_ARM
    ee: str = DEFAULT_EE
    frequency: int = DEFAULT_FREQUENCY
    episodes: int = DEFAULT_EPISODES
    max_episodes: int = DEFAULT_MAX_EPISODES
    visualization: bool = True
    save_data: bool = False
    task_dir: Path | None = None
    send_real_robot: bool = False
    allow_real_robot_commands: bool = False
    allow_downloads: bool = False
    mode: str = "probe"
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    runtime_smoke_timeout_seconds: float = DEFAULT_RUNTIME_SMOKE_TIMEOUT_SECONDS
    openvla_endpoint_url: str = ""
    wam_endpoint_url: str = ""

    @classmethod
    def from_env(
        cls,
        *,
        job_dir: Path | None = None,
        mode: str | None = None,
        timeout_seconds: float | None = None,
    ) -> "UnitreeLeRobotPolicyRuntimeConfig":
        resolved_mode = mode or os.getenv("BLUEPRINT_UNITREE_LEROBOT_MODE", "probe")
        save_data_default = resolved_mode == "sim_eval"
        task_dir = _path_env("BLUEPRINT_UNITREE_LEROBOT_TASK_DIR")
        if task_dir is None and job_dir is not None:
            task_dir = Path(job_dir) / "unitree_lerobot_g1_policy_handoff" / "task_data"
        policy_path = os.getenv("BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH", "").strip()
        if not policy_path:
            policy_path = os.getenv("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT", "").strip()
        return cls(
            root=_path_env("BLUEPRINT_UNITREE_LEROBOT_ROOT"),
            python_executable=_python_executable_value(
                os.getenv("BLUEPRINT_UNITREE_LEROBOT_PYTHON")
            ),
            policy_path=policy_path,
            dataset_repo_id=os.getenv(
                "BLUEPRINT_UNITREE_LEROBOT_DATASET_REPO_ID",
                DEFAULT_DATASET_REPO_ID,
            ).strip()
            or DEFAULT_DATASET_REPO_ID,
            dataset_root=os.getenv("BLUEPRINT_UNITREE_LEROBOT_DATASET_ROOT", "").strip(),
            task=os.getenv("BLUEPRINT_UNITREE_LEROBOT_TASK", "").strip(),
            policy_family=os.getenv(
                "BLUEPRINT_UNITREE_POLICY_FAMILY",
                DEFAULT_POLICY_FAMILY,
            ).strip()
            or DEFAULT_POLICY_FAMILY,
            arm=os.getenv("BLUEPRINT_UNITREE_G1_ARM", DEFAULT_ARM).strip() or DEFAULT_ARM,
            ee=os.getenv("BLUEPRINT_UNITREE_G1_EE", DEFAULT_EE).strip() or DEFAULT_EE,
            frequency=_int_env("BLUEPRINT_UNITREE_LEROBOT_FREQUENCY", DEFAULT_FREQUENCY),
            episodes=_int_env("BLUEPRINT_UNITREE_LEROBOT_EPISODES", DEFAULT_EPISODES),
            max_episodes=_int_env(
                "BLUEPRINT_UNITREE_LEROBOT_MAX_EPISODES",
                DEFAULT_MAX_EPISODES,
            ),
            visualization=parse_bool(
                os.getenv("BLUEPRINT_UNITREE_LEROBOT_VISUALIZATION"),
                default=True,
            ),
            save_data=parse_bool(
                os.getenv("BLUEPRINT_UNITREE_LEROBOT_SAVE_DATA"),
                default=save_data_default,
            ),
            task_dir=task_dir,
            send_real_robot=parse_bool(
                os.getenv("BLUEPRINT_UNITREE_LEROBOT_SEND_REAL_ROBOT"),
                default=False,
            ),
            allow_real_robot_commands=parse_bool(
                os.getenv("BLUEPRINT_UNITREE_ALLOW_REAL_ROBOT_COMMANDS"),
                default=False,
            ),
            allow_downloads=parse_bool(
                os.getenv("BLUEPRINT_UNITREE_ALLOW_DOWNLOADS"),
                default=False,
            ),
            mode=resolved_mode,
            timeout_seconds=timeout_seconds
            if timeout_seconds is not None
            else _float_env("BLUEPRINT_UNITREE_LEROBOT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            runtime_smoke_timeout_seconds=_float_env(
                "BLUEPRINT_UNITREE_LEROBOT_SMOKE_TIMEOUT_SECONDS",
                DEFAULT_RUNTIME_SMOKE_TIMEOUT_SECONDS,
            ),
            openvla_endpoint_url=os.getenv("BLUEPRINT_UNITREE_OPENVLA_ENDPOINT_URL", "").strip(),
            wam_endpoint_url=os.getenv("BLUEPRINT_UNITREE_WAM_ENDPOINT_URL", "").strip(),
        )

    @property
    def normalized_mode(self) -> str:
        return self.mode if self.mode in RUNTIME_MODES else "probe"

    @property
    def normalized_policy_family(self) -> str:
        return self.policy_family if self.policy_family in POLICY_FAMILIES else "unknown"

    @property
    def root_exists(self) -> bool:
        return bool(self.root and self.root.exists())

    @property
    def source_root(self) -> Path | None:
        return self.root.expanduser().resolve() if self.root_exists and self.root else None

    def eval_script_candidates(self) -> list[dict[str, Any]]:
        if self.root is None:
            return []
        root = self.root.expanduser()
        specs = (
            ("unitree_lerobot/eval_robot/eval_g1_sim.py", True),
            ("eval_robot/eval_g1_sim.py", True),
            ("unitree_lerobot/eval_robot/eval_g1.py", False),
            ("eval_robot/eval_g1.py", False),
        )
        return [
            {
                "relative_path": relative,
                "path": str(root / relative),
                "exists": (root / relative).is_file(),
                "sim_script": is_sim,
                "preferred_for_blueprint_sim_eval": is_sim,
            }
            for relative, is_sim in specs
        ]

    def preferred_sim_eval_script(self) -> Path | None:
        for row in self.eval_script_candidates():
            if row["sim_script"] and row["exists"]:
                return Path(str(row["path"]))
        return None

    def missing_requirements(self) -> list[str]:
        missing: list[str] = []
        mode = self.normalized_mode
        script = self.preferred_sim_eval_script()
        if mode in {"probe", "dry_run", "sim_eval", "not_configured"}:
            if self.root is None:
                missing.append("BLUEPRINT_UNITREE_LEROBOT_ROOT")
            elif not self.root_exists:
                missing.append("unitree_lerobot_root_missing")
            if script is None:
                missing.append("unitree_lerobot_eval_g1_sim_script_missing")
        if mode in {"probe", "sim_eval", "not_configured"}:
            endpoint_based = self.normalized_policy_family == "openvla_endpoint"
            endpoint_configured = bool(self.openvla_endpoint_url)
            if not _policy_path_configured(self.policy_path) and not (
                endpoint_based and endpoint_configured
            ):
                missing.append("BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH")
        if self.normalized_policy_family == "openvla_endpoint" and not self.openvla_endpoint_url:
            missing.append("BLUEPRINT_UNITREE_OPENVLA_ENDPOINT_URL")
        if self.normalized_policy_family == "unifolm_wma" and not self.wam_endpoint_url:
            missing.append("BLUEPRINT_UNITREE_WAM_ENDPOINT_URL")
        return sorted(set(missing))

    def is_configured_for_probe(self) -> bool:
        return not self.safety_errors()

    def is_configured_for_dry_run(self) -> bool:
        if self.safety_errors():
            return False
        if self.root is None or not self.root_exists:
            return False
        return self.preferred_sim_eval_script() is not None

    def is_configured_for_sim_eval(self) -> bool:
        if not self.is_configured_for_dry_run():
            return False
        return _policy_or_endpoint_configured(self)

    def safety_errors(self) -> list[str]:
        errors: list[str] = []
        if self.send_real_robot and not self.allow_real_robot_commands:
            errors.append(
                "blocked_real_robot_command_requires_BLUEPRINT_UNITREE_ALLOW_REAL_ROBOT_COMMANDS"
            )
        return errors

    def task_dir_or_default(self, job_dir: Path) -> Path:
        return (
            self.task_dir.expanduser()
            if self.task_dir is not None
            else job_dir / "unitree_lerobot_g1_policy_handoff" / "task_data"
        )

    def to_safe_dict(self, *, job_dir: Path | None = None) -> dict[str, Any]:
        task_dir = self.task_dir or (
            job_dir / "unitree_lerobot_g1_policy_handoff" / "task_data"
            if job_dir is not None
            else None
        )
        return {
            "root": str(self.root.expanduser()) if self.root else None,
            "root_exists": self.root_exists,
            "python_executable": self.python_executable,
            "python_executable_available": _python_executable_available(
                self.python_executable
            ),
            "policy_path": self.policy_path or None,
            "policy_path_configured": _policy_path_configured(self.policy_path),
            "policy_or_endpoint_configured": _policy_or_endpoint_configured(self),
            "dataset_repo_id": self.dataset_repo_id,
            "dataset_root": self.dataset_root,
            "task": self.task,
            "policy_family": self.normalized_policy_family,
            "arm": self.arm,
            "ee": self.ee,
            "frequency": self.frequency,
            "episodes": self.episodes,
            "max_episodes": self.max_episodes,
            "visualization": self.visualization,
            "save_data": self.save_data,
            "task_dir": str(task_dir) if task_dir else None,
            "send_real_robot": self.send_real_robot,
            "allow_real_robot_commands": self.allow_real_robot_commands,
            "allow_downloads": self.allow_downloads,
            "mode": self.normalized_mode,
            "timeout_seconds": self.timeout_seconds,
            "runtime_smoke_timeout_seconds": self.runtime_smoke_timeout_seconds,
            "openvla_endpoint_url_configured": bool(self.openvla_endpoint_url),
            "wam_endpoint_url_configured": bool(self.wam_endpoint_url),
        }


def build_unitree_lerobot_g1_sim_command(
    config: UnitreeLeRobotPolicyRuntimeConfig,
    *,
    job_dir: Path,
) -> dict[str, Any]:
    task_dir = config.task_dir_or_default(job_dir)
    script = config.preferred_sim_eval_script()
    safety_errors = config.safety_errors()
    missing = config.missing_requirements()
    if script is None:
        return {
            "status": "not_configured",
            "command_built": False,
            "missing_requirements": missing,
            "safety_errors": safety_errors,
        }
    if safety_errors:
        return {
            "status": "blocked",
            "command_built": False,
            "missing_requirements": missing,
            "safety_errors": safety_errors,
        }
    command = [
        config.python_executable,
        str(script),
        f"--policy.path={config.policy_path}",
        f"--repo_id={config.dataset_repo_id}",
        f"--root={config.dataset_root}",
        f"--episodes={config.episodes}",
        f"--frequency={config.frequency}",
        f"--arm={config.arm}",
        f"--ee={config.ee}",
        f"--visualization={_bool_text(config.visualization)}",
        f"--save_data={_bool_text(config.save_data)}",
        f"--task_dir={task_dir}",
        f"--max_episodes={config.max_episodes}",
        f"--send_real_robot={_bool_text(config.send_real_robot)}",
    ]
    if config.task:
        command.append(f"--task={config.task}")
    return {
        "schema_version": "unitree_lerobot_g1_policy_command.v1",
        "status": "ready",
        "command_built": True,
        "command": command,
        "cwd": str(config.source_root or script.parent),
        "eval_script": str(script),
        "task_dir": str(task_dir),
        "missing_requirements": missing,
        "safety_errors": safety_errors,
        "env_overlay_names": [
            "BLUEPRINT_UNITREE_ALLOW_DOWNLOADS",
            "BLUEPRINT_UNITREE_ALLOW_REAL_ROBOT_COMMANDS",
            "HF_HUB_OFFLINE",
            "PYTHONPATH",
        ],
        "python_executable": config.python_executable,
        "python_executable_available": _python_executable_available(
            config.python_executable
        ),
        "pythonpath_entries": _unitree_lerobot_pythonpath_entries(config),
        "claim_boundary": {
            "command_build_is_not_execution_proof": True,
            "send_real_robot": config.send_real_robot,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
    }


def probe_unitree_lerobot_g1_runtime(
    *,
    config: UnitreeLeRobotPolicyRuntimeConfig,
    job_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    script = config.preferred_sim_eval_script()
    policy_path_configured = _policy_path_configured(config.policy_path)
    policy_or_endpoint_configured = _policy_or_endpoint_configured(config)
    runtime_configured = bool(config.root_exists and script is not None)
    missing = config.missing_requirements()
    safety = config.safety_errors()
    configuration_stage = _lerobot_configuration_stage(
        source_runtime_configured=runtime_configured,
        policy_or_endpoint_configured=policy_or_endpoint_configured,
        safety_blocks=safety,
    )
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "configured"
        if runtime_configured and policy_or_endpoint_configured
        else "not_configured",
        "configuration_stage": configuration_stage,
        "mode": config.normalized_mode,
        "job_dir": str(job_dir),
        "unitree_lerobot_root_exists": config.root_exists,
        "unitree_lerobot_root": str(config.root.expanduser()) if config.root else None,
        "unitree_lerobot_python_executable": config.python_executable,
        "unitree_lerobot_python_executable_available": _python_executable_available(
            config.python_executable
        ),
        "unitree_lerobot_pythonpath_entries": _unitree_lerobot_pythonpath_entries(config),
        "eval_scripts": config.eval_script_candidates(),
        "unitree_lerobot_eval_script_found": script is not None,
        "preferred_eval_script": str(script) if script else None,
        "unitree_lerobot_source_runtime_configured": runtime_configured,
        "unitree_lerobot_policy_path_configured": policy_path_configured,
        "unitree_lerobot_policy_or_endpoint_configured": policy_or_endpoint_configured,
        "dataset_repo_id_configured": bool(config.dataset_repo_id),
        "dataset_root_configured": bool(config.dataset_root),
        "chosen_arm": config.arm,
        "chosen_end_effector": config.ee,
        "frequency": config.frequency,
        "episodes": config.episodes,
        "max_episodes": config.max_episodes,
        "simulation_mode": True,
        "visualization": config.visualization,
        "save_data": config.save_data,
        "task_dir": str(config.task_dir_or_default(job_dir)),
        "runtime_smoke_timeout_seconds": config.runtime_smoke_timeout_seconds,
        "send_real_robot": config.send_real_robot,
        "allow_real_robot_commands": config.allow_real_robot_commands,
        "allow_downloads": config.allow_downloads,
        "missing_requirements": missing,
        "safety_blocks": safety,
        "source_commit": _repo_head(config.source_root),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def _run_subprocess(
    *,
    command: Sequence[str],
    cwd: Path,
    env_overlay: Mapping[str, str],
    timeout_seconds: float,
    stdout_log: Path,
    stderr_log: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    env = {**os.environ, **dict(env_overlay)}
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        stdout_log.write_text(completed.stdout or "", encoding="utf-8")
        stderr_log.write_text(completed.stderr or "", encoding="utf-8")
        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "subprocess_attempted": True,
            "return_code": completed.returncode,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "stdout_size_bytes": len(completed.stdout or ""),
            "stderr_size_bytes": len(completed.stderr or ""),
            "runtime_error_summary": None
            if completed.returncode == 0
            else f"unitree_lerobot_sim_eval_exited_{completed.returncode}",
        }
    except Exception as exc:
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text(str(exc), encoding="utf-8")
        return {
            "status": "failed",
            "subprocess_attempted": True,
            "return_code": None,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "stdout_size_bytes": 0,
            "stderr_size_bytes": len(str(exc)),
            "runtime_error_summary": f"{type(exc).__name__}: {str(exc)[:500]}",
        }


def _truth_boundary(
    *,
    config: UnitreeLeRobotPolicyRuntimeConfig,
    probe: Mapping[str, Any],
    command_result: Mapping[str, Any],
    subprocess_result: Mapping[str, Any] | None,
    artifacts: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    attempted = bool(subprocess_result and subprocess_result.get("subprocess_attempted"))
    command_built = bool(command_result.get("command_built"))
    proven = bool(
        attempted
        and subprocess_result
        and subprocess_result.get("return_code") == 0
        and command_built
        and Path(str(subprocess_result.get("stdout_log") or "")).is_file()
        and Path(str(subprocess_result.get("stderr_log") or "")).is_file()
    )
    family = config.normalized_policy_family
    hand_policy_used = bool(proven and family != "openvla_endpoint" and config.ee in HAND_EES)
    vla_used = bool(proven and family in VLA_POLICY_FAMILIES)
    missing = list(probe.get("missing_requirements") or [])
    safety = list(probe.get("safety_blocks") or [])
    not_configured_reason = None
    if missing:
        not_configured_reason = "missing_requirements"
    if safety:
        not_configured_reason = "safety_blocked"
    return {
        "schema_version": TRUTH_SCHEMA_VERSION,
        "generated_at": generated_at,
        "official_unitree_rl_gym_controller_used": False,
        "official_unitree_rl_gym_policy_execution_proven": False,
        "official_unitree_controller_used": False,
        "official_policy_execution_proven": False,
        "unitree_lerobot_runtime_configured": bool(
            probe.get("unitree_lerobot_root_exists")
            and probe.get("unitree_lerobot_eval_script_found")
        ),
        "unitree_lerobot_eval_script_found": bool(probe.get("unitree_lerobot_eval_script_found")),
        "unitree_lerobot_policy_path_configured": _policy_path_configured(config.policy_path),
        "unitree_lerobot_policy_loaded": proven,
        "unitree_lerobot_sim_inference_attempted": attempted,
        "unitree_lerobot_sim_inference_proven": proven,
        "unitree_lerobot_command_built": command_built,
        "unitree_hand_manipulation_policy_used": hand_policy_used,
        "g1_manipulation_policy_used": hand_policy_used,
        "g1_locomotion_policy_used": False,
        "vla_policy_family": family,
        "vla_policy_used": vla_used,
        "wam_world_model_used": False,
        "openvla_endpoint_used": bool(proven and family == "openvla_endpoint"),
        "openvla_g1_action_adapter_configured": False,
        "unifolm_vla_used": bool(proven and family == "unifolm_vla"),
        "unifolm_wma_used": bool(proven and family == "unifolm_wma"),
        "freejoint_proxy_used": False,
        "physical_robot_command_attempted": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "not_configured_reason": not_configured_reason,
        "missing_requirements": missing,
        "safety_blocks": safety,
        "runtime_error_summary": subprocess_result.get("runtime_error_summary")
        if subprocess_result
        else None,
        "source_root": str(config.source_root) if config.source_root else None,
        "source_commit": _repo_head(config.source_root),
        "policy_path": config.policy_path or None,
        "dataset_repo_id": config.dataset_repo_id,
        "dataset_root": config.dataset_root,
        "arm": config.arm,
        "ee": config.ee,
        "frequency": config.frequency,
        "task_dir": str(config.task_dir_or_default(Path(str(probe.get("job_dir"))))),
        "artifacts": dict(artifacts),
        "claim_boundary": {
            "lerobot_vla_execution_requires_successful_sim_eval_subprocess": True,
            "generic_openvla_is_not_g1_control_without_explicit_adapter": True,
            "wam_world_model_used_only_when_invoked": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
    }


def _handoff_manifest(
    *,
    config: UnitreeLeRobotPolicyRuntimeConfig,
    job_dir: Path,
    generated_at: str,
    probe: Mapping[str, Any],
    command_result: Mapping[str, Any],
    subprocess_result: Mapping[str, Any] | None,
    artifacts: Mapping[str, Any],
    truth_path: Path,
) -> dict[str, Any]:
    claims_proven: list[str] = []
    claims_not_proven = [
        "physical_robot_readiness",
        "deployment_readiness",
        "official_unitree_rl_gym_locomotion",
        "openvla_g1_control",
        "wam_world_model_control",
    ]
    if subprocess_result and subprocess_result.get("return_code") == 0:
        claims_proven.append("unitree_lerobot_g1_sim_eval_subprocess_completed")
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "generated_at": generated_at,
        "lane_name": "unitree_lerobot_g1_policy_eval",
        "provider_name": "UnitreeLeRobotG1Provider",
        "runtime_mode": config.normalized_mode,
        "command": list(command_result.get("command") or []),
        "cwd": command_result.get("cwd"),
        "env_var_names_used": list(ENV_VAR_NAMES),
        "source_root": str(config.source_root) if config.source_root else None,
        "source_commit": _repo_head(config.source_root),
        "python_executable": config.python_executable,
        "python_executable_available": _python_executable_available(
            config.python_executable
        ),
        "pythonpath_entries": _unitree_lerobot_pythonpath_entries(config),
        "policy_path": config.policy_path or None,
        "policy_family": config.normalized_policy_family,
        "dataset_repo_id": config.dataset_repo_id,
        "dataset_root": config.dataset_root,
        "task": config.task,
        "arm": config.arm,
        "ee": config.ee,
        "frequency": config.frequency,
        "episodes": config.episodes,
        "max_episodes": config.max_episodes,
        "visualization": config.visualization,
        "save_data": config.save_data,
        "task_dir": str(config.task_dir_or_default(job_dir)),
        "stdout_log": subprocess_result.get("stdout_log") if subprocess_result else None,
        "stderr_log": subprocess_result.get("stderr_log") if subprocess_result else None,
        "return_code": subprocess_result.get("return_code") if subprocess_result else None,
        "videos": list(artifacts.get("videos") or []),
        "traces": list(artifacts.get("traces") or []),
        "metrics": list(artifacts.get("metrics") or []),
        "truth_boundary_path": str(truth_path),
        "missing_requirements": list(probe.get("missing_requirements") or []),
        "safety_blocks": list(probe.get("safety_blocks") or []),
        "claims_proven": claims_proven,
        "claims_not_proven": claims_not_proven,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def run_unitree_lerobot_g1_policy_eval(
    *,
    job_dir: Path,
    config: UnitreeLeRobotPolicyRuntimeConfig | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or utc_now_iso()
    job_dir = Path(job_dir).expanduser().resolve()
    ensure_dir(job_dir)
    handoff_dir = job_dir / "unitree_lerobot_g1_policy_handoff"
    ensure_dir(handoff_dir)
    config = config or UnitreeLeRobotPolicyRuntimeConfig.from_env(job_dir=job_dir)
    task_dir = config.task_dir_or_default(job_dir)
    probe = probe_unitree_lerobot_g1_runtime(
        config=config,
        job_dir=job_dir,
        generated_at=generated_at,
    )
    write_json(job_dir / "unitree_lerobot_g1_runtime_probe.json", probe)
    command_result = build_unitree_lerobot_g1_sim_command(config, job_dir=job_dir)
    write_json(handoff_dir / "command.json", command_result)

    subprocess_result: dict[str, Any] | None = None
    mode = config.normalized_mode
    should_run = mode == "sim_eval" and config.is_configured_for_sim_eval()
    if should_run:
        ensure_dir(task_dir)
        stdout_log = handoff_dir / "stdout.log"
        stderr_log = handoff_dir / "stderr.log"
        env_overlay = {
            "BLUEPRINT_UNITREE_ALLOW_DOWNLOADS": _bool_text(config.allow_downloads),
            "BLUEPRINT_UNITREE_ALLOW_REAL_ROBOT_COMMANDS": _bool_text(
                config.allow_real_robot_commands
            ),
            "HF_HUB_OFFLINE": "0" if config.allow_downloads else "1",
        }
        pythonpath_entries = _unitree_lerobot_pythonpath_entries(config)
        if pythonpath_entries:
            env_overlay["PYTHONPATH"] = _prepend_env_path(
                os.environ.get("PYTHONPATH", ""),
                pythonpath_entries,
            )
        subprocess_result = _run_subprocess(
            command=command_result["command"],
            cwd=Path(str(command_result["cwd"])),
            env_overlay=env_overlay,
            timeout_seconds=config.timeout_seconds,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )
        write_json(handoff_dir / "subprocess_result.json", subprocess_result)
    artifacts = _collect_artifacts(task_dir)
    truth_path = job_dir / "unitree_lerobot_g1_policy_runtime_truth_boundary.json"
    truth = _truth_boundary(
        config=config,
        probe=probe,
        command_result=command_result,
        subprocess_result=subprocess_result,
        artifacts=artifacts,
        generated_at=generated_at,
    )
    write_json(truth_path, truth)
    handoff = _handoff_manifest(
        config=config,
        job_dir=job_dir,
        generated_at=generated_at,
        probe=probe,
        command_result=command_result,
        subprocess_result=subprocess_result,
        artifacts=artifacts,
        truth_path=truth_path,
    )
    write_json(handoff_dir / "robot_team_handoff_manifest.json", handoff)
    stack_installation_audit = build_unitree_policy_stack_installation_audit(
        job_dir=job_dir,
        generated_at=generated_at,
        config=config,
        lerobot_probe=probe,
    )
    stack_installation_audit_path = job_dir / "unitree_policy_stack_installation_audit.json"
    write_json(stack_installation_audit_path, stack_installation_audit)
    provider_registry_path = job_dir / "unitree_policy_provider_registry_probe.json"
    provider_registry = _build_policy_provider_registry_payload(
        generated_at=generated_at,
        providers=[
            OfficialUnitreeRLGymProvider().probe(),
            UnitreeLeRobotG1Provider(config).probe(job_dir=job_dir, generated_at=generated_at),
            OpenVLAEndpointProvider(config.openvla_endpoint_url).probe(),
            UnitreeGrootN17SonicProvider().probe(),
            UnifoLMVLAProvider().probe(),
            UnifoLMWMAProvider().probe(),
        ],
        installation_audit=stack_installation_audit,
    )
    write_json(provider_registry_path, provider_registry)
    manipulation_candidates = (
        stack_installation_audit.get("component_checks", {})
        .get("unitree_manipulation_runtime", {})
        .get("candidates", [])
    )
    lerobot_installation_candidate = next(
        (
            dict(candidate)
            for candidate in manipulation_candidates
            if isinstance(candidate, Mapping)
            and candidate.get("candidate_id") == "unitree_lerobot_g1"
        ),
        {},
    )
    if config.safety_errors():
        status = "blocked"
    elif mode == "sim_eval" and subprocess_result is not None:
        status = "completed" if subprocess_result.get("return_code") == 0 else "failed"
    elif mode == "dry_run" and command_result.get("command_built"):
        status = "dry_run"
    elif probe.get("status") == "configured":
        status = "configured"
    else:
        status = "not_configured"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "job_dir": str(job_dir),
        "mode": mode,
        "provider_name": "UnitreeLeRobotG1Provider",
        "runtime_probe_path": str(job_dir / "unitree_lerobot_g1_runtime_probe.json"),
        "truth_boundary_path": str(truth_path),
        "handoff_manifest_path": str(handoff_dir / "robot_team_handoff_manifest.json"),
        "command_path": str(handoff_dir / "command.json"),
        "subprocess_result_path": str(handoff_dir / "subprocess_result.json")
        if subprocess_result
        else None,
        "unitree_policy_stack_installation_audit_path": str(stack_installation_audit_path),
        "unitree_policy_provider_registry_probe_path": str(provider_registry_path),
        "selected_locomotion_provider": provider_registry["selected_locomotion_provider"],
        "selected_unitree_manipulation_runtime": provider_registry[
            "selected_unitree_manipulation_runtime"
        ],
        "selected_unitree_action_command": provider_registry["selected_unitree_action_command"],
        "selected_unitree_hand_policy": provider_registry["selected_unitree_hand_policy"],
        "unitree_hand_manipulation_policy_in_place": provider_registry[
            "unitree_hand_manipulation_policy_in_place"
        ],
        "openvla_selected_for_g1_policy": provider_registry["openvla_selected_for_g1_policy"],
        "wam_selected_for_g1_policy": provider_registry["wam_selected_for_g1_policy"],
        "whole_unitree_policy_stack_installed": stack_installation_audit[
            "whole_unitree_policy_stack_installed"
        ],
        "unitree_policy_stack_installation_status": stack_installation_audit["status"],
        "unitree_policy_stack_installation_blockers": stack_installation_audit["blockers"],
        "unitree_lerobot_python_executable": config.python_executable,
        "unitree_lerobot_python_executable_available": _python_executable_available(
            config.python_executable
        ),
        "unitree_lerobot_pythonpath_entries": _unitree_lerobot_pythonpath_entries(config),
        "unitree_lerobot_runtime_smoke_timeout_seconds": config.runtime_smoke_timeout_seconds,
        "unitree_lerobot_runtime_configured": truth["unitree_lerobot_runtime_configured"],
        "unitree_lerobot_configuration_stage": lerobot_installation_candidate.get(
            "configuration_stage"
        ),
        "unitree_lerobot_source_runtime_files_configured": lerobot_installation_candidate.get(
            "source_runtime_files_configured"
        ),
        "unitree_lerobot_source_runtime_execution_ready": lerobot_installation_candidate.get(
            "source_runtime_execution_ready"
        ),
        "unitree_lerobot_source_runtime_dependency_smoke_passed": (
            lerobot_installation_candidate.get("source_runtime_dependency_smoke_passed")
        ),
        "unitree_lerobot_source_runtime_blockers": lerobot_installation_candidate.get(
            "blockers",
            [],
        ),
        "unitree_lerobot_command_built": truth["unitree_lerobot_command_built"],
        "unitree_lerobot_sim_inference_attempted": truth["unitree_lerobot_sim_inference_attempted"],
        "unitree_lerobot_sim_inference_proven": truth["unitree_lerobot_sim_inference_proven"],
        "missing_requirements": truth["missing_requirements"],
        "safety_blocks": truth["safety_blocks"],
        "runtime_error_summary": truth["runtime_error_summary"],
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
    }
    write_json(job_dir / "unitree_lerobot_g1_policy_runtime_summary.json", summary)
    return summary


class OfficialUnitreeRLGymProvider:
    name = "OfficialUnitreeRLGymProvider"
    lane_name = "official_unitree_rl_gym"

    def probe(self) -> dict[str, Any]:
        root_value = os.getenv("BLUEPRINT_UNITREE_RL_GYM_ROOT", "").strip()
        root = Path(root_value).expanduser() if root_value else None
        checkpoint = root / "deploy" / "pre_train" / "g1" / "motion.pt" if root else None
        configured = bool(root and root.exists() and checkpoint and checkpoint.is_file())
        return {
            "lane_name": self.lane_name,
            "provider_name": self.name,
            "status": "configured" if configured else "not_configured",
            "root": str(root) if root else None,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "official_unitree_rl_gym_policy_execution_proven": False,
            "claim_boundary": "probe_only_existing_lane_proof_lives_in_official_run_artifacts",
        }


class UnitreeLeRobotG1Provider:
    name = "UnitreeLeRobotG1Provider"
    lane_name = "unitree_lerobot_g1"

    def __init__(self, config: UnitreeLeRobotPolicyRuntimeConfig) -> None:
        self.config = config

    def probe(self, *, job_dir: Path, generated_at: str) -> dict[str, Any]:
        probe = probe_unitree_lerobot_g1_runtime(
            config=self.config,
            job_dir=job_dir,
            generated_at=generated_at,
        )
        return {
            "lane_name": self.lane_name,
            "provider_name": self.name,
            "status": probe["status"],
            "configuration_stage": probe["configuration_stage"],
            "runtime_configured": bool(
                probe["unitree_lerobot_root_exists"] and probe["unitree_lerobot_eval_script_found"]
            ),
            "source_runtime_configured": probe["unitree_lerobot_source_runtime_configured"],
            "policy_path_configured": probe["unitree_lerobot_policy_path_configured"],
            "policy_or_endpoint_configured": probe[
                "unitree_lerobot_policy_or_endpoint_configured"
            ],
            "missing_requirements": probe["missing_requirements"],
            "safety_blocks": probe["safety_blocks"],
        }


class OpenVLAEndpointProvider:
    name = "OpenVLAEndpointProvider"
    lane_name = "openvla_endpoint"

    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url

    def probe(self) -> dict[str, Any]:
        configured = bool(self.endpoint_url)
        return {
            "lane_name": self.lane_name,
            "provider_name": self.name,
            "status": "configured" if configured else "not_configured",
            "endpoint_url_configured": configured,
            "g1_action_adapter_configured": False,
            "openvla_endpoint_used": False,
            "claim_boundary": (
                "generic_openvla_endpoint_is_not_g1_control_without_explicit_g1_action_adapter"
            ),
        }


class UnifoLMVLAProvider:
    name = "UnifoLMVLAProvider"
    lane_name = "unifolm_vla"

    def probe(self) -> dict[str, Any]:
        command = os.getenv("BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND", "").strip()
        checkpoint = os.getenv("BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT", "").strip()
        vlm = os.getenv("BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT", "").strip()
        configured = bool(command and checkpoint and vlm)
        return {
            "lane_name": self.lane_name,
            "provider_name": self.name,
            "status": "configured" if configured else "not_configured",
            "command_configured": bool(command),
            "checkpoint_configured": bool(checkpoint),
            "vlm_checkpoint_configured": bool(vlm),
            "unifolm_vla_used": False,
        }


class UnifoLMWMAProvider:
    name = "UnifoLMWMAProvider"
    lane_name = "unifolm_wma"

    def probe(self) -> dict[str, Any]:
        command = os.getenv("BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND", "").strip()
        checkpoint = os.getenv("BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT", "").strip()
        configured = bool(command and checkpoint)
        return {
            "lane_name": self.lane_name,
            "provider_name": self.name,
            "status": "configured" if configured else "not_configured",
            "command_configured": bool(command),
            "checkpoint_configured": bool(checkpoint),
            "unifolm_wma_used": False,
            "wam_world_model_used": False,
            "claim_boundary": "wam_world_model_used_stays_false_until_wma_runtime_is_invoked",
        }


class UnitreeGrootN17SonicProvider:
    name = "UnitreeGrootN17SonicProvider"
    lane_name = GROOT_POLICY_ID

    def probe(self) -> dict[str, Any]:
        audit = probe_unitree_groot_n17_sonic_runtime(generated_at=utc_now_iso())
        return {
            "lane_name": self.lane_name,
            "provider_name": self.name,
            "status": audit["status"],
            "runtime_configured": bool(audit["runtime_configured"]),
            "policy_command_configured": bool(audit["policy_command_configured"]),
            "policy_command_available": bool(audit["policy_command_available"]),
            "ready_for_policy_action_command": bool(
                audit["ready_for_policy_action_command"]
            ),
            "ready_for_sim2sim": bool(audit["ready_for_sim2sim"]),
            "blockers": list(audit.get("blockers", [])),
            "claim_boundary": audit["claim_boundary"],
        }


def _env_command_candidate_row(spec: Mapping[str, Any]) -> dict[str, Any]:
    command_env = ""
    command_value = ""
    for env_name in spec["command_envs"]:
        value = os.getenv(str(env_name), "").strip()
        if value:
            command_env = str(env_name)
            command_value = value
            break
    checkpoint_env = ""
    checkpoint_value = ""
    for env_name in spec["checkpoint_envs"]:
        value = os.getenv(str(env_name), "").strip()
        if value:
            checkpoint_env = str(env_name)
            checkpoint_value = value
            break
    checkpoint_configured, checkpoint_path, checkpoint_exists = _configured_path_or_repo(
        checkpoint_value
    )
    command_available = _command_available(command_value)
    blockers: list[str] = []
    if not command_value:
        blockers.append("blocked_missing_unitree_action_command")
    elif not command_available:
        blockers.append("blocked_unitree_action_command_unavailable")
    if not checkpoint_value:
        blockers.append("blocked_missing_unitree_action_checkpoint")
    elif not checkpoint_configured:
        blockers.append("blocked_unitree_action_checkpoint_missing")

    extra_required_checkpoints = []
    for env_name in spec.get("extra_required_checkpoint_envs", ()):
        value = os.getenv(str(env_name), "").strip()
        configured, path_text, exists = _configured_path_or_repo(value)
        extra_required_checkpoints.append(
            {
                "checkpoint_env": str(env_name),
                "checkpoint_configured": bool(value),
                "checkpoint_path": path_text,
                "checkpoint_exists": exists,
            }
        )
        if not value:
            blockers.append(f"blocked_missing_{env_name}")
        elif not configured:
            blockers.append(f"blocked_missing_path_for_{env_name}")

    source_root_env = _string(spec.get("source_root_env"))
    source_root_value = os.getenv(source_root_env, "").strip() if source_root_env else ""
    source_root_path = Path(source_root_value).expanduser() if source_root_value else None
    source_root_exists = bool(source_root_path and source_root_path.exists())
    if source_root_value and not source_root_exists:
        blockers.append(f"blocked_missing_path_for_{source_root_env}")
    extra_required_roots = []
    for env_name in spec.get("extra_required_root_envs", ()):
        value = os.getenv(str(env_name), "").strip()
        path = Path(value).expanduser() if value else None
        exists = bool(path and path.exists())
        extra_required_roots.append(
            {
                "root_env": str(env_name),
                "root_configured": bool(value),
                "root_path": str(path) if path else None,
                "root_exists": exists,
            }
        )
        if value and not exists:
            blockers.append(f"blocked_missing_path_for_{env_name}")

    return {
        "candidate_id": spec["candidate_id"],
        "runtime_role": spec["runtime_role"],
        "command_env": command_env or list(spec["command_envs"])[0],
        "command_configured": bool(command_value),
        "command_available": command_available,
        "command_value_redacted": "<configured>" if command_value else None,
        "checkpoint_env": checkpoint_env or list(spec["checkpoint_envs"])[0],
        "checkpoint_configured": bool(checkpoint_value),
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": checkpoint_exists,
        "extra_required_checkpoints": extra_required_checkpoints,
        "source_root_env": source_root_env or None,
        "source_root_configured": bool(source_root_value),
        "source_root_path": str(source_root_path) if source_root_path else None,
        "source_root_exists": source_root_exists,
        "extra_required_roots": extra_required_roots,
        "ready_for_policy_action_command": bool(command_value and command_available and not blockers),
        "blockers": sorted(set(blockers)),
    }


def _unitree_action_command_installation_probe() -> dict[str, Any]:
    candidates = [_env_command_candidate_row(spec) for spec in UNITREE_ACTION_COMMAND_CANDIDATES]
    ready = [row for row in candidates if row["ready_for_policy_action_command"]]
    return {
        "status": "configured" if ready else "not_configured",
        "ready_candidate_count": len(ready),
        "selected_candidate_id": ready[0]["candidate_id"] if ready else None,
        "candidates": candidates,
        "blockers": []
        if ready
        else ["unitree_specific_action_command_not_configured"],
    }


def _unifolm_candidate_configured(mode: str) -> dict[str, Any]:
    if mode == "vla":
        command = os.getenv("BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND", "").strip()
        checkpoint = os.getenv("BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT", "").strip()
        vlm = os.getenv("BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT", "").strip()
        checkpoint_configured, checkpoint_path, checkpoint_exists = _configured_path_or_repo(
            checkpoint
        )
        vlm_configured, vlm_path, vlm_exists = _configured_path_or_repo(vlm)
        blockers = []
        if not command:
            blockers.append("blocked_missing_BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND")
        elif not _command_available(command):
            blockers.append("blocked_unifolm_vla_command_unavailable")
        if not checkpoint_configured:
            blockers.append("blocked_missing_BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT")
        if not vlm_configured:
            blockers.append("blocked_missing_BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT")
        return {
            "candidate_id": "unitree_unifolm_vla_policy",
            "runtime_role": "unitree_native_vla_manipulation_runtime",
            "configured": bool(command and _command_available(command) and not blockers),
            "command_configured": bool(command),
            "command_available": _command_available(command),
            "checkpoint_path": checkpoint_path,
            "checkpoint_exists": checkpoint_exists,
            "vlm_checkpoint_path": vlm_path,
            "vlm_checkpoint_exists": vlm_exists,
            "blockers": blockers,
        }
    command = os.getenv("BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND", "").strip()
    checkpoint = os.getenv("BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT", "").strip()
    checkpoint_configured, checkpoint_path, checkpoint_exists = _configured_path_or_repo(checkpoint)
    blockers = []
    if not command:
        blockers.append("blocked_missing_BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND")
    elif not _command_available(command):
        blockers.append("blocked_unifolm_wma_command_unavailable")
    if not checkpoint_configured:
        blockers.append("blocked_missing_BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT")
    return {
        "candidate_id": "unitree_unifolm_wma_policy",
        "runtime_role": "unitree_native_wma_manipulation_runtime",
        "configured": bool(command and _command_available(command) and not blockers),
        "command_configured": bool(command),
        "command_available": _command_available(command),
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": checkpoint_exists,
        "blockers": blockers,
    }


def _groot_n17_sonic_candidate_configured() -> dict[str, Any]:
    audit = probe_unitree_groot_n17_sonic_runtime(generated_at=utc_now_iso())
    configured = bool(audit.get("runtime_configured"))
    command_configured = bool(audit.get("policy_command_configured"))
    command_available = bool(audit.get("policy_command_available"))
    sim2sim_command_configured = bool(audit.get("sim2sim_command_configured"))
    sim2sim_command_available = bool(audit.get("sim2sim_command_available"))
    action_command_blockers: list[str] = []
    if not command_configured:
        action_command_blockers.append(
            "blocked_missing_BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"
        )
    elif not command_available:
        action_command_blockers.append("blocked_groot_n17_sonic_policy_command_unavailable")
    sim2sim_blockers: list[str] = []
    if sim2sim_command_configured and not sim2sim_command_available:
        sim2sim_blockers.append("blocked_groot_n17_sonic_sim2sim_command_unavailable")
    n17_checkpoint_original_reference = audit.get("n17_checkpoint_original_env_reference")
    partial_configuration = bool(
        audit.get("policy_command_configured")
        or n17_checkpoint_original_reference
        or audit.get("g1_sonic_checkpoint_configured")
        or audit.get("groot_root", {}).get("configured")
        or audit.get("wbc_root", {}).get("configured")
        or audit.get("sim2sim_command_configured")
        or audit.get("policy_server_url_configured")
        or audit.get("policy_server_host_configured")
        or audit.get("policy_server_port_configured")
    ) and (
        not configured
        or bool(action_command_blockers)
        or bool(sim2sim_blockers)
        or bool(audit.get("blockers"))
    )
    return {
        "candidate_id": "unitree_groot_n17_sonic_policy",
        "runtime_role": "unitree_groot_n17_sonic_manipulation_runtime",
        "configured": configured,
        "partial_configuration": partial_configuration,
        "ready_for_policy_action_command": bool(
            audit.get("ready_for_policy_action_command")
        ),
        "ready_for_sim2sim": bool(audit.get("ready_for_sim2sim")),
        "command_configured": command_configured,
        "command_available": command_available,
        "groot_checkpoint_path": audit.get("n17_checkpoint_path"),
        "groot_checkpoint_original_reference": n17_checkpoint_original_reference,
        "groot_checkpoint_effective_reference": audit.get("n17_checkpoint_effective_reference"),
        "groot_checkpoint_selection_source": audit.get("n17_checkpoint_selection_source"),
        "groot_default_experimental_checkpoint_applied": bool(
            audit.get("default_experimental_checkpoint_applied")
        ),
        "groot_checkpoint_exists": bool(audit.get("n17_checkpoint_exists")),
        "groot_checkpoint_reference_kind": audit.get("n17_checkpoint_reference_kind"),
        "sonic_checkpoint_path": audit.get("g1_sonic_checkpoint_path"),
        "sonic_checkpoint_exists": bool(audit.get("g1_sonic_checkpoint_exists")),
        "sonic_checkpoint_reference_kind": audit.get("g1_sonic_checkpoint_reference_kind"),
        "groot_source_root_path": audit.get("groot_root", {}).get("path"),
        "groot_source_root_exists": bool(audit.get("groot_root", {}).get("exists")),
        "wbc_source_root_path": audit.get("wbc_root", {}).get("path"),
        "wbc_source_root_exists": bool(audit.get("wbc_root", {}).get("exists")),
        "policy_server_url_configured": bool(audit.get("policy_server_url_configured")),
        "policy_server_host_configured": bool(audit.get("policy_server_host_configured")),
        "policy_server_port_configured": bool(audit.get("policy_server_port_configured")),
        "sim2sim_command_configured": sim2sim_command_configured,
        "sim2sim_command_available": sim2sim_command_available,
        "embodiment_tag": "UNITREE_G1_SONIC",
        "expected_action_dimension": 78,
        "action_command_blockers": sorted(set(action_command_blockers)),
        "sim2sim_blockers": sorted(set(sim2sim_blockers)),
        "runtime_audit": audit,
        "blockers": sorted(set(str(item) for item in audit.get("blockers", []))),
    }


def _unitree_manipulation_runtime_installation_probe(
    *,
    job_dir: Path,
    config: UnitreeLeRobotPolicyRuntimeConfig,
    lerobot_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    probe = dict(lerobot_probe) if lerobot_probe is not None else {}
    if not probe:
        probe = probe_unitree_lerobot_g1_runtime(
            config=config,
            job_dir=job_dir,
            generated_at=utc_now_iso(),
        )
    lerobot_source_runtime_configured = bool(
        probe.get("unitree_lerobot_source_runtime_configured")
        if "unitree_lerobot_source_runtime_configured" in probe
        else probe.get("unitree_lerobot_root_exists")
        and probe.get("unitree_lerobot_eval_script_found")
    )
    lerobot_policy_or_endpoint_configured = bool(
        probe.get("unitree_lerobot_policy_or_endpoint_configured")
        if "unitree_lerobot_policy_or_endpoint_configured" in probe
        else probe.get("unitree_lerobot_policy_path_configured")
    )
    script = config.preferred_sim_eval_script()
    eval_script_smoke = _unitree_lerobot_eval_script_smoke_probe(
        config=config,
        script=script,
        timeout_seconds=config.runtime_smoke_timeout_seconds,
    )
    lerobot_source_runtime_execution_ready = bool(
        lerobot_source_runtime_configured and eval_script_smoke.get("passed")
    )
    lerobot_ready_for_sim_eval = bool(
        config.is_configured_for_sim_eval() and lerobot_source_runtime_execution_ready
    )
    lerobot_candidate_blockers: list[str] = []
    if not lerobot_source_runtime_configured:
        lerobot_candidate_blockers.append("blocked_unitree_lerobot_source_runtime_not_configured")
    elif not eval_script_smoke.get("passed"):
        lerobot_candidate_blockers.append("blocked_unitree_lerobot_eval_script_smoke_failed")
    if not lerobot_policy_or_endpoint_configured:
        lerobot_candidate_blockers.append("blocked_unitree_lerobot_policy_or_endpoint_not_configured")
    lerobot_candidate_blockers.extend(str(item) for item in probe.get("safety_blocks") or [])
    lerobot_candidate_stage = probe.get("configuration_stage") or _lerobot_configuration_stage(
        source_runtime_configured=lerobot_source_runtime_configured,
        policy_or_endpoint_configured=lerobot_policy_or_endpoint_configured,
        safety_blocks=list(probe.get("safety_blocks") or []),
    )
    if lerobot_source_runtime_configured and not eval_script_smoke.get("passed"):
        lerobot_candidate_stage = "source_runtime_files_ready_dependency_smoke_failed"
    candidates = [
        _groot_n17_sonic_candidate_configured(),
        {
            "candidate_id": "unitree_lerobot_g1",
            "runtime_role": "unitree_lerobot_g1_sim_manipulation_runtime",
            "configured": lerobot_ready_for_sim_eval,
            "configuration_stage": lerobot_candidate_stage,
            "runtime_configured": bool(
                probe.get("unitree_lerobot_root_exists")
                and probe.get("unitree_lerobot_eval_script_found")
            ),
            "source_runtime_configured": lerobot_source_runtime_configured,
            "source_runtime_files_configured": lerobot_source_runtime_configured,
            "source_runtime_execution_ready": lerobot_source_runtime_execution_ready,
            "source_runtime_dependency_smoke_passed": bool(eval_script_smoke.get("passed")),
            "source_runtime_dependency_smoke": eval_script_smoke,
            "policy_path_configured": bool(
                probe.get("unitree_lerobot_policy_path_configured")
            ),
            "policy_or_endpoint_configured": lerobot_policy_or_endpoint_configured,
            "source_runtime_ready_without_policy": bool(
                lerobot_source_runtime_execution_ready
                and not lerobot_policy_or_endpoint_configured
            ),
            "source_files_ready_without_policy": bool(
                lerobot_source_runtime_configured and not lerobot_policy_or_endpoint_configured
            ),
            "partial_configuration": bool(
                (lerobot_source_runtime_configured or lerobot_policy_or_endpoint_configured)
                and not lerobot_ready_for_sim_eval
            ),
            "missing_requirements": list(probe.get("missing_requirements") or []),
            "safety_blocks": list(probe.get("safety_blocks") or []),
            "blockers": sorted(set(lerobot_candidate_blockers)),
        },
        _unifolm_candidate_configured("vla"),
        _unifolm_candidate_configured("wma"),
    ]
    ready = [
        row
        for row in candidates
        if row.get("configured") and not row.get("blockers")
    ]
    partial = [row for row in candidates if row.get("partial_configuration")]
    blockers = []
    if not ready:
        blockers.append("unitree_manipulation_runtime_not_configured")
        for row in candidates:
            if row.get("configured") or row.get("partial_configuration"):
                blockers.extend(str(item) for item in row.get("blockers", []))
    return {
        "status": "configured" if ready else "not_configured",
        "ready_candidate_count": len(ready),
        "partial_candidate_count": len(partial),
        "selected_candidate_id": ready[0]["candidate_id"] if ready else None,
        "partial_candidate_ids": [str(row["candidate_id"]) for row in partial],
        "candidates": candidates,
        "blockers": sorted(set(blockers)),
    }


def build_unitree_policy_stack_installation_audit(
    *,
    job_dir: Path,
    generated_at: str,
    config: UnitreeLeRobotPolicyRuntimeConfig | None = None,
    lerobot_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or UnitreeLeRobotPolicyRuntimeConfig.from_env(job_dir=job_dir)
    official_probe = OfficialUnitreeRLGymProvider().probe()
    manipulation_probe = _unitree_manipulation_runtime_installation_probe(
        job_dir=job_dir,
        config=config,
        lerobot_probe=lerobot_probe,
    )
    action_command_probe = _unitree_action_command_installation_probe()
    component_checks = {
        "official_rl_gym_locomotion": {
            "status": official_probe["status"],
            "configured": official_probe["status"] == "configured",
            "provider_probe": official_probe,
            "blockers": []
            if official_probe["status"] == "configured"
            else ["official_unitree_rl_gym_locomotion_not_configured"],
        },
        "unitree_manipulation_runtime": {
            "status": manipulation_probe["status"],
            "configured": manipulation_probe["status"] == "configured",
            **manipulation_probe,
        },
        "unitree_action_command": {
            "status": action_command_probe["status"],
            "configured": action_command_probe["status"] == "configured",
            **action_command_probe,
        },
    }
    blockers: list[str] = []
    for component in component_checks.values():
        if not component.get("configured"):
            blockers.extend(str(item) for item in component.get("blockers", []))
    partial_component_ids = [
        component_id
        for component_id, component in component_checks.items()
        if any(row.get("partial_configuration") for row in component.get("candidates", []))
    ]
    installed = not blockers
    return {
        "schema_version": "unitree_policy_stack_installation_audit.v1",
        "generated_at": generated_at,
        "job_dir": str(job_dir),
        "status": "installed" if installed else "not_installed",
        "whole_unitree_policy_stack_installed": installed,
        "partial_component_ids": partial_component_ids,
        "component_checks": component_checks,
        "required_components": [
            "official_rl_gym_locomotion",
            "unitree_manipulation_runtime",
            "unitree_action_command",
        ],
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "selected_locomotion_provider_is_not_whole_stack_installation": True,
            "whole_stack_requires_unitree_manipulation_runtime_and_action_command": True,
            "stack_installation_is_not_task_success_or_physical_readiness": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def _build_policy_provider_registry_payload(
    *,
    generated_at: str,
    providers: Sequence[Mapping[str, Any]],
    installation_audit: Mapping[str, Any],
) -> dict[str, Any]:
    component_checks = installation_audit["component_checks"]
    official_locomotion = component_checks["official_rl_gym_locomotion"]
    manipulation_runtime = component_checks["unitree_manipulation_runtime"]
    action_command = component_checks["unitree_action_command"]
    selected_locomotion_provider = (
        "official_unitree_rl_gym" if official_locomotion.get("configured") else None
    )
    selected_manipulation_runtime = manipulation_runtime.get("selected_candidate_id")
    selected_action_command = action_command.get("selected_candidate_id")
    hand_policy_in_place = bool(
        manipulation_runtime.get("configured") and action_command.get("configured")
    )
    legacy_first_configured = next(
        (row["lane_name"] for row in providers if row.get("status") == "configured"),
        None,
    )
    return {
        "schema_version": "unitree_policy_provider_registry_probe.v1",
        "generated_at": generated_at,
        "providers": [dict(row) for row in providers],
        "selected_provider_legacy_first_configured": legacy_first_configured,
        "selected_provider": legacy_first_configured,
        "selected_locomotion_provider": selected_locomotion_provider,
        "selected_unitree_manipulation_runtime": selected_manipulation_runtime,
        "selected_unitree_action_command": selected_action_command,
        "selected_unitree_hand_policy": selected_action_command if hand_policy_in_place else None,
        "unitree_hand_manipulation_policy_in_place": hand_policy_in_place,
        "unitree_hand_manipulation_policy_used": False,
        "openvla_selected_for_g1_policy": False,
        "wam_selected_for_g1_policy": False,
        "g1_robot_policy_family_decision": {
            "locomotion_provider": selected_locomotion_provider,
            "manipulation_runtime": selected_manipulation_runtime,
            "action_command": selected_action_command,
            "unitree_native_policy_required_for_g1_hand_claims": True,
            "openvla_is_comparison_only_for_g1": True,
            "wam_is_evaluator_not_robot_policy": True,
            "hand_policy_in_place": hand_policy_in_place,
        },
        "whole_unitree_policy_stack_installed": installation_audit[
            "whole_unitree_policy_stack_installed"
        ],
        "installation_status": installation_audit["status"],
        "installation_blockers": installation_audit["blockers"],
        "installation_audit": dict(installation_audit),
        "claim_boundary": {
            "provider_probe_is_not_execution_proof": True,
            "official_rl_gym_and_lerobot_lanes_are_separate": True,
            "selected_provider_is_not_whole_stack_installation": True,
            "selected_provider_legacy_field_may_be_locomotion_only": True,
            "unitree_hand_policy_requires_manipulation_runtime_and_action_command": True,
            "openvla_is_not_default_g1_robot_policy": True,
            "wam_is_not_g1_robot_policy": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
    }


def build_policy_provider_registry_probe(
    *,
    job_dir: Path,
    generated_at: str,
    config: UnitreeLeRobotPolicyRuntimeConfig | None = None,
) -> dict[str, Any]:
    config = config or UnitreeLeRobotPolicyRuntimeConfig.from_env(job_dir=job_dir)
    providers = [
        OfficialUnitreeRLGymProvider().probe(),
        UnitreeLeRobotG1Provider(config).probe(job_dir=job_dir, generated_at=generated_at),
        OpenVLAEndpointProvider(config.openvla_endpoint_url).probe(),
        UnitreeGrootN17SonicProvider().probe(),
        UnifoLMVLAProvider().probe(),
        UnifoLMWMAProvider().probe(),
    ]
    installation_audit = build_unitree_policy_stack_installation_audit(
        job_dir=job_dir,
        generated_at=generated_at,
        config=config,
    )
    return _build_policy_provider_registry_payload(
        generated_at=generated_at,
        providers=providers,
        installation_audit=installation_audit,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=RUNTIME_MODES, default=None)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--smoke-timeout-seconds", type=float)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--python-executable")
    parser.add_argument("--policy-path")
    parser.add_argument("--dataset-repo-id")
    parser.add_argument("--dataset-root")
    parser.add_argument("--task")
    parser.add_argument("--policy-family", choices=POLICY_FAMILIES)
    parser.add_argument("--arm")
    parser.add_argument("--ee")
    parser.add_argument("--frequency", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--visualization", choices=("true", "false"))
    parser.add_argument("--save-data", choices=("true", "false"))
    parser.add_argument("--task-dir", type=Path)
    args = parser.parse_args(argv)

    config = UnitreeLeRobotPolicyRuntimeConfig.from_env(
        job_dir=args.job_dir,
        mode=args.mode,
        timeout_seconds=args.timeout_seconds,
    )
    config = UnitreeLeRobotPolicyRuntimeConfig(
        root=args.root if args.root is not None else config.root,
        python_executable=args.python_executable or config.python_executable,
        policy_path=args.policy_path if args.policy_path is not None else config.policy_path,
        dataset_repo_id=args.dataset_repo_id or config.dataset_repo_id,
        dataset_root=args.dataset_root if args.dataset_root is not None else config.dataset_root,
        task=args.task if args.task is not None else config.task,
        policy_family=args.policy_family or config.policy_family,
        arm=args.arm or config.arm,
        ee=args.ee or config.ee,
        frequency=args.frequency if args.frequency is not None else config.frequency,
        episodes=args.episodes if args.episodes is not None else config.episodes,
        max_episodes=args.max_episodes if args.max_episodes is not None else config.max_episodes,
        visualization=parse_bool(args.visualization, default=config.visualization),
        save_data=parse_bool(args.save_data, default=config.save_data),
        task_dir=args.task_dir if args.task_dir is not None else config.task_dir,
        send_real_robot=config.send_real_robot,
        allow_real_robot_commands=config.allow_real_robot_commands,
        allow_downloads=config.allow_downloads,
        mode=config.normalized_mode,
        timeout_seconds=config.timeout_seconds,
        runtime_smoke_timeout_seconds=args.smoke_timeout_seconds
        if args.smoke_timeout_seconds is not None
        else config.runtime_smoke_timeout_seconds,
        openvla_endpoint_url=config.openvla_endpoint_url,
        wam_endpoint_url=config.wam_endpoint_url,
    )
    summary = run_unitree_lerobot_g1_policy_eval(job_dir=args.job_dir, config=config)
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] in {"completed", "dry_run", "configured", "not_configured"}:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
