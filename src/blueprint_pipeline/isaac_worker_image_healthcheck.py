"""Fail-closed build/runtime contract checks for the Isaac review worker image."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import utc_now_iso, write_json

SCHEMA_VERSION = "isaac_worker_image_healthcheck.v1"


def run_image_healthcheck(
    *,
    build_time: bool,
    env: Mapping[str, str] | None = None,
    exists: Callable[[Path], bool] | None = None,
    importer: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    runtime_env = dict(env or os.environ)
    path_exists = exists or Path.exists
    import_name = importer or importlib.import_module
    isaac_python = Path(runtime_env.get("BLUEPRINT_ISAAC_PYTHON") or "/isaac-sim/python.sh")
    g1_usd = Path(
        runtime_env.get("BLUEPRINT_ISAAC_UNITREE_G1_USD")
        or "/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd"
    )
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    for name, path, blocker in (
        ("isaac_python", isaac_python, "isaac_python_missing"),
        ("unitree_g1_usd", g1_usd, "unitree_g1_usd_missing"),
    ):
        passed = bool(path_exists(path))
        checks.append({"name": name, "status": "passed" if passed else "blocked", "path": str(path)})
        if not passed:
            blockers.append(blocker)
    try:
        import_name("blueprint_pipeline")
        checks.append({"name": "blueprint_pipeline_import", "status": "passed"})
    except Exception as exc:  # noqa: BLE001
        checks.append(
            {
                "name": "blueprint_pipeline_import",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        )
        blockers.append("blueprint_pipeline_import_failed")
    family = str(runtime_env.get("BLUEPRINT_WORKER_IMAGE_FAMILY") or "").strip()
    family_passed = family == "isaac-eval-worker"
    checks.append(
        {
            "name": "worker_image_family",
            "status": "passed" if family_passed else "blocked",
            "value": family or None,
        }
    )
    if not family_passed:
        blockers.append("isaac_worker_image_family_invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "passed" if not blockers else "blocked",
        "mode": "build_time" if build_time else "runtime_static",
        "checks": checks,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "Static image-content healthcheck only. GPU, CUDA, Isaac startup, RTX rendering, "
            "scene loading, policy execution, and task success require the runtime preflight."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-time", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_image_healthcheck(build_time=args.build_time)
    if args.output:
        write_json(args.output, result)
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
