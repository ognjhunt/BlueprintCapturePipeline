"""No-GPU environment doctor for CPU/dry-render validation.

The doctor is intentionally import-only: it does not launch simulators, call cloud
providers, or touch live model APIs. It exists to fail fast when the canonical
local interpreter is missing CPU dependencies that would otherwise turn important
USD/MuJoCo/dry-render tests into quiet skips.
"""
from __future__ import annotations

import importlib
import json
import sys
from typing import Iterable


CPU_ENV_MODULES: tuple[str, ...] = (
    "PIL",
    "pxr",
    "mujoco",
    "trimesh",
    "boto3",
    "botocore",
    "numpy",
    "yaml",
    "jsonschema",
    "blueprint_pipeline",
    "blueprint_contracts",
)


def check_cpu_env(modules: Iterable[str] = CPU_ENV_MODULES) -> dict:
    """Return structured import status for the no-GPU validation stack."""
    checked: dict[str, dict] = {}
    missing: list[str] = []
    for module in modules:
        try:
            imported = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report every import failure shape.
            checked[module] = {
                "present": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
            }
            missing.append(module)
        else:
            checked[module] = {
                "present": True,
                "version": str(getattr(imported, "__version__", "")),
            }
    return {
        "schema_version": "cpu_env_doctor.v1",
        "sys_executable": sys.executable,
        "sys_version": sys.version,
        "modules": checked,
        "missing": missing,
        "ok": not missing,
    }


def main(argv: list[str] | None = None) -> int:
    """Console entry point; prints JSON and exits nonzero when required modules are missing."""
    _ = argv
    report = check_cpu_env()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
