"""Fail-closed build/runtime contract checks for the Isaac review worker image."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from .common import utc_now_iso, write_json

SCHEMA_VERSION = "isaac_worker_image_healthcheck.v1"


def _safe_exists(path_exists: Callable[[Path], bool], path: Path) -> bool:
    try:
        return bool(path_exists(path))
    except OSError:
        return False


def _asset_binding_check(
    *,
    runtime_env: Mapping[str, str],
    path_exists: Callable[[Path], bool],
) -> tuple[dict[str, Any], bool]:
    asset_root = str(runtime_env.get("ISAACSIM_ASSET_ROOT") or "").strip()
    g1_binding = str(runtime_env.get("BLUEPRINT_ISAAC_UNITREE_G1_USD") or "").strip()
    binding_path = Path(g1_binding) if g1_binding else None
    binding_is_absolute = bool(binding_path and binding_path.is_absolute())
    binding_is_uri = bool(urlparse(g1_binding).scheme in {"http", "https", "omniverse"})
    binding_is_relative = bool(
        g1_binding
        and not binding_is_absolute
        and not binding_is_uri
        and g1_binding.endswith("Isaac/Robots/Unitree/G1/g1.usd")
    )
    root_is_uri = urlparse(asset_root).scheme in {"http", "https", "omniverse"}
    root_is_local = bool(asset_root and Path(asset_root).is_absolute())
    local_exists = bool(
        binding_is_absolute and binding_path and _safe_exists(path_exists, binding_path)
    )
    passed = local_exists or binding_is_uri or (
        binding_is_relative and (root_is_uri or root_is_local)
    )
    return (
        {
            "name": "unitree_g1_asset_binding",
            "status": "passed" if passed else "blocked",
            "binding": g1_binding or None,
            "asset_root": asset_root or None,
            "local_file_exists": local_exists,
            "resolution_deferred_to_runtime": passed and not local_exists,
        },
        passed,
    )


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
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    for name, path, blocker in (("isaac_python", isaac_python, "isaac_python_missing"),):
        passed = _safe_exists(path_exists, path)
        checks.append({"name": name, "status": "passed" if passed else "blocked", "path": str(path)})
        if not passed:
            blockers.append(blocker)
    asset_check, asset_binding_passed = _asset_binding_check(
        runtime_env=runtime_env,
        path_exists=path_exists,
    )
    checks.append(asset_check)
    if not asset_binding_passed:
        blockers.append("unitree_g1_asset_binding_invalid")
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
    simulator_family = str(
        runtime_env.get("BLUEPRINT_SIMULATOR_FRAMEWORK") or ""
    ).strip()
    simulator_family_passed = simulator_family == "isaac_sim"
    checks.append(
        {
            "name": "simulator_family",
            "status": "passed" if simulator_family_passed else "blocked",
            "value": simulator_family or None,
        }
    )
    if not simulator_family_passed:
        blockers.append("isaac_worker_simulator_family_invalid")
    try:
        simulator_major = int(
            runtime_env.get("BLUEPRINT_ISAAC_SIM_MAJOR_VERSION") or 0
        )
    except ValueError:
        simulator_major = 0
    if simulator_major != 6:
        blockers.append("isaac_worker_simulator_major_version_invalid")
    source_commit = str(runtime_env.get("BLUEPRINT_SOURCE_COMMIT") or "").lower()
    dirty_patch = str(
        runtime_env.get("BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256") or ""
    ).lower()
    if not (7 <= len(source_commit) <= 64) or any(
        char not in "0123456789abcdef" for char in source_commit
    ):
        blockers.append("isaac_worker_source_commit_invalid")
    if len(dirty_patch) != 64 or any(
        char not in "0123456789abcdef" for char in dirty_patch
    ):
        blockers.append("isaac_worker_source_dirty_patch_sha256_invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "passed" if not blockers else "blocked",
        "mode": "build_time" if build_time else "runtime_static",
        "runtime_metadata": {
            "image_family": family or None,
            "simulator_family": simulator_family or None,
            "simulator_major_version": simulator_major or None,
            "source_commit": source_commit or None,
            "source_dirty_patch_sha256": dirty_patch or None,
            "blueprint_pipeline_imported": "blueprint_pipeline_import_failed"
            not in blockers,
            "configured_g1_asset_binding_valid": asset_binding_passed,
            "configured_g1_usd_exists": asset_check["local_file_exists"],
            "g1_asset_resolution_deferred_to_runtime": asset_check[
                "resolution_deferred_to_runtime"
            ],
        },
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
