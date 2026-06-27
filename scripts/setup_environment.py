#!/usr/bin/env python3
"""Validate the supported single-VM site-world runtime environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


CheckResult = Tuple[bool, str]


def print_status(message: str, status: str = "info") -> None:
    colors = {
        "ok": Colors.GREEN + "OK" + Colors.RESET,
        "warn": Colors.YELLOW + "WARN" + Colors.RESET,
        "error": Colors.RED + "FAIL" + Colors.RESET,
        "info": Colors.BLUE + "INFO" + Colors.RESET,
    }
    print(f"  {colors.get(status, colors['info'])} {message}")


def print_header(title: str) -> None:
    print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
    print("=" * len(title))


def run_command(args: Iterable[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(list(args), text=True, capture_output=True, check=False)
    except (OSError, ValueError):
        return None


def check_python() -> CheckResult:
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 10
    return ok, f"Python {version.major}.{version.minor}.{version.micro}"


def check_pip() -> CheckResult:
    result = run_command([sys.executable, "-m", "pip", "--version"])
    if result and result.returncode == 0:
        return True, result.stdout.strip()
    return False, "pip is not available in this interpreter"


def check_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def check_repo_package() -> CheckResult:
    ok = check_module("blueprint_pipeline")
    return ok, "blueprint_pipeline importable" if ok else "blueprint_pipeline is not installed"


def check_torch() -> CheckResult:
    probe = (
        "import json\n"
        "import torch\n"
        "payload = {'version': torch.__version__, 'cuda': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), 'device_name': ''}\n"
        "if payload['cuda_available']:\n"
        "    payload['device_name'] = torch.cuda.get_device_name(0)\n"
        "print(json.dumps(payload))\n"
    )
    result = run_command([sys.executable, "-c", probe])
    if result is None:
        return False, "torch probe could not be launched"
    if result.returncode != 0:
        stderr_tail = result.stderr.strip().splitlines()[-1:] if result.stderr.strip() else []
        reason = stderr_tail[0] if stderr_tail else f"probe exited {result.returncode}"
        return False, f"torch unavailable: {reason}"
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return False, f"torch probe returned invalid output: {exc}"
    if payload.get("cuda_available"):
        return True, f"torch {payload.get('version')} CUDA {payload.get('cuda')} ({payload.get('device_name')})"
    return False, f"torch {payload.get('version')} installed but CUDA is not available"


def check_nvidia_smi() -> CheckResult:
    result = run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if result and result.returncode == 0:
        gpu_name = result.stdout.strip().splitlines()[0]
        return True, gpu_name
    return False, "nvidia-smi not available"


def check_binary(name: str) -> CheckResult:
    path = shutil.which(name)
    if path:
        return True, path
    return False, f"{name} not found in PATH"


def check_optional_sam3() -> CheckResult:
    if not check_module("sam3"):
        return False, "sam3 package not installed (optional)"
    weights_path = Path(os.getenv("SAM3_WEIGHTS_PATH", "/opt/sam3_weights/sam3.pt"))
    if weights_path.is_file():
        return True, f"sam3 installed, weights at {weights_path}"
    return False, f"sam3 installed but weights missing at {weights_path}"


def check_privacy_command(name: str) -> CheckResult:
    value = str(os.getenv(name) or "").strip()
    if value:
        return True, f"{name} configured"
    return False, f"{name} not configured"


def check_site_world_runtime() -> CheckResult:
    service_url = str(os.getenv("SITE_WORLD_RUNTIME_SERVICE_URL") or "").strip()
    if service_url:
        return True, service_url
    return False, "SITE_WORLD_RUNTIME_SERVICE_URL is not configured"


def backend_checks() -> Dict[str, Mapping[str, object]]:
    from blueprint_pipeline.object_index_stage import _backend_preflight_status, _command_from_env

    commands = {
        "yolo_world": _command_from_env("OBJECT_INDEX_YOLO_WORLD_COMMAND"),
        "grounding_dino": _command_from_env("OBJECT_INDEX_GROUNDING_DINO_COMMAND"),
        "sam3": _command_from_env("OBJECT_INDEX_SAM3_COMMAND"),
        "splat_analyzer": _command_from_env("OBJECT_INDEX_SPLAT_ANALYZER_COMMAND"),
    }
    return {
        name: _backend_preflight_status(backend_name=name, command_template=template)
        for name, template in commands.items()
    }


def run_checks() -> Dict[str, CheckResult]:
    checks: Dict[str, CheckResult] = {}

    print_header("Base Environment")
    checks["python"] = check_python()
    print_status(checks["python"][1], "ok" if checks["python"][0] else "error")

    checks["pip"] = check_pip()
    print_status(checks["pip"][1], "ok" if checks["pip"][0] else "error")

    checks["repo"] = check_repo_package()
    print_status(checks["repo"][1], "ok" if checks["repo"][0] else "error")

    print_header("GPU Runtime")
    checks["torch"] = check_torch()
    print_status(checks["torch"][1], "ok" if checks["torch"][0] else "warn")

    checks["nvidia_smi"] = check_nvidia_smi()
    print_status(checks["nvidia_smi"][1], "ok" if checks["nvidia_smi"][0] else "warn")

    print_header("Pipeline Runtime")
    for module_name in ("ultralytics", "trimesh"):
        ok = check_module(module_name)
        checks[module_name] = (ok, f"{module_name} importable" if ok else f"{module_name} missing")
        print_status(checks[module_name][1], "ok" if ok else "error")

    for binary in ("ffmpeg", "ffprobe"):
        checks[binary] = check_binary(binary)
        print_status(checks[binary][1], "ok" if checks[binary][0] else "error")

    print_header("Optional Runtime")
    checks["sam3"] = check_optional_sam3()
    print_status(checks["sam3"][1], "ok" if checks["sam3"][0] else "info")
    for name in ("PRIVACY_SAM3_COMMAND", "VIP_COMMAND", "DEEPPRIVACY2_COMMAND"):
        checks[name.lower()] = check_privacy_command(name)
        print_status(checks[name.lower()][1], "ok" if checks[name.lower()][0] else "info")

    checks["site_world_runtime"] = check_site_world_runtime()
    print_status(checks["site_world_runtime"][1], "ok" if checks["site_world_runtime"][0] else "warn")

    print_header("Object Index Backends")
    for backend, payload in backend_checks().items():
        status = str(payload.get("status") or "unknown")
        reason = str(payload.get("reason") or "")
        message = f"{backend}: status={status}"
        if reason:
            message += f" reason={reason}"
        severity = "ok" if status == "ready" else "warn" if backend in {"sam3", "splat_analyzer"} else "error"
        print_status(message, severity)

    return checks


def print_summary(checks: Mapping[str, CheckResult]) -> None:
    print_header("Summary")
    required = ("python", "pip", "repo", "ultralytics", "trimesh", "ffmpeg", "ffprobe")
    gpu_runtime = checks.get("torch", (False, ""))[0] or checks.get("nvidia_smi", (False, ""))[0]
    missing = [name for name in required if not checks.get(name, (False, ""))[0]]
    if not missing and gpu_runtime:
        print_status("Supported single-VM site-world path is ready.", "ok")
    elif not missing:
        print_status("Core pipeline packages are present, but GPU runtime is not fully ready.", "warn")
    else:
        print_status(f"Missing required runtime components: {', '.join(missing)}", "error")

    print("\nRecommended next step:")
    print("  ./scripts/install_ml_stack.sh")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the supported single-VM site-world environment")
    parser.add_argument("--check", action="store_true", help="Run checks only (default behavior)")
    parser.add_argument("--json-output", help="Optional path to write the check results as JSON")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}BlueprintCapturePipeline Environment Check{Colors.RESET}")
    print("=" * 42)
    checks = run_checks()
    print_summary(checks)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({key: {"passed": value[0], "detail": value[1]} for key, value in checks.items()}, indent=2), encoding="utf-8")

    required = ("python", "pip", "repo", "ultralytics", "trimesh", "ffmpeg", "ffprobe")
    gpu_runtime = checks.get("torch", (False, ""))[0] or checks.get("nvidia_smi", (False, ""))[0]
    return 0 if all(checks.get(name, (False, ""))[0] for name in required) and gpu_runtime else 1


if __name__ == "__main__":
    raise SystemExit(main())
