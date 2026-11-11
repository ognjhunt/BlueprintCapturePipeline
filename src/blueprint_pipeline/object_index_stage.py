"""Object-index backend preflight helpers (the setup check reads them; the device object index was removed 2026-11-11)."""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List


def _command_from_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if value:
        return value
    repo_root = Path(__file__).resolve().parents[2]
    defaults = {
        "OBJECT_INDEX_YOLO_WORLD_COMMAND": repo_root / "scripts" / "object_index_yolo_world_runner.py",
        "OBJECT_INDEX_GROUNDING_DINO_COMMAND": repo_root / "scripts" / "object_index_grounding_dino_runner.py",
        "OBJECT_INDEX_SAM3_COMMAND": repo_root / "scripts" / "object_index_sam3_runner.py",
        "OBJECT_INDEX_SPLAT_ANALYZER_COMMAND": repo_root / "scripts" / "object_index_splat_analyzer_runner.py",
    }
    script_path = defaults.get(name)
    if script_path is not None and script_path.is_file():
        return f"{shlex.quote(sys.executable)} {shlex.quote(str(script_path))} {{INPUT_JSON}} {{OUTPUT_JSON}}"
    return ""


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _default_sam3_weights_path() -> Path:
    return Path(os.getenv("SAM3_WEIGHTS_PATH") or "/opt/sam3_weights/sam3.pt")


def _backend_runtime_requirements(backend_name: str) -> Dict[str, Any]:
    support_level = "required"
    required_modules: List[str]
    required_paths: List[str] = []
    if backend_name in {"yolo_world", "grounding_dino"}:
        required_modules = ["torch", "ultralytics"]
    elif backend_name == "sam3":
        support_level = "optional"
        required_modules = ["torch", "sam3"]
        required_paths = [str(_default_sam3_weights_path())]
    elif backend_name == "splat_analyzer":
        support_level = "optional"
        required_modules = []
    else:
        required_modules = []
    missing_modules = [name for name in required_modules if not _module_available(name)]
    missing_paths = [path for path in required_paths if not Path(path).is_file()]
    return {
        "support_level": support_level,
        "required_modules": required_modules,
        "missing_modules": missing_modules,
        "required_paths": required_paths,
        "missing_paths": missing_paths,
    }


def _backend_preflight_status(*, backend_name: str, command_template: str) -> Dict[str, Any]:
    requirements = _backend_runtime_requirements(backend_name)
    if not command_template:
        return {
            "configured": False,
            "status": "missing",
            "reason": "command_not_configured",
            **requirements,
        }
    try:
        rendered = (
            command_template.replace("{INPUT_JSON}", "/tmp/input.json")
            .replace("{OUTPUT_JSON}", "/tmp/output.json")
            .replace("{OUTPUT_DIR}", "/tmp")
        )
        command = shlex.split(rendered)
    except ValueError as exc:
        return {
            "configured": True,
            "status": "invalid",
            "reason": f"invalid_command_template:{exc}",
            **requirements,
        }
    executable = command[0] if command else ""
    executable_path = shutil.which(executable) or executable
    status = "ready" if executable else "invalid"
    reason = ""
    if not executable:
        reason = "empty_command"
    elif executable_path and str(executable_path).startswith("/"):
        if not Path(str(executable_path)).exists():
            status = "missing"
            reason = f"missing_executable:{executable_path}"
    if not reason and requirements["missing_modules"]:
        if requirements["support_level"] == "optional":
            status = "optional_unavailable"
        else:
            status = "runtime_missing"
        reason = "missing_modules:" + ",".join(requirements["missing_modules"])
    if not reason and requirements["missing_paths"]:
        if requirements["support_level"] == "optional":
            status = "optional_unavailable"
        else:
            status = "runtime_missing"
        reason = "missing_paths:" + ",".join(requirements["missing_paths"])
    return {
        "configured": True,
        "status": status,
        "reason": reason,
        "command": command,
        **requirements,
    }
