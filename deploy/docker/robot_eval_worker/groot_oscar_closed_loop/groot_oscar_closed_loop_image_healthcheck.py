#!/usr/bin/env python3
"""Build/runtime healthcheck for the sealed ``blueprint-groot-oscar-eval`` image.

Fail-closed: proves the two Python environments and the baked assets are present
before a paid pod trusts the image. Runs in the MAIN env; the GR00T env is
checked out-of-process against ``/opt/gr00t-venv``.

Claim boundary: this proves build/runtime readiness only — not provider startup,
GR00T inference, WAM quality, or task success.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


OSCAR_REPO = os.environ.get("BLUEPRINT_GROOT_OSCAR_OSCAR_REPO", "/opt/OSCAR")
OSCAR_CHECKPOINT = os.environ.get(
    "BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT", "/opt/blueprint/ckpts/oscar"
)
GROOT_ROOT = os.environ.get("BLUEPRINT_GROOT_OSCAR_GROOT_ROOT", "/opt/gr00t")
GROOT_VENV_PYTHON = os.environ.get(
    "BLUEPRINT_GROOT_OSCAR_GROOT_VENV_PYTHON", "/opt/gr00t-venv/bin/python"
)
SONIC_CHECKPOINT = os.environ.get(
    "BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT", "/opt/blueprint/ckpts/sonic"
)

# Main-env modules the closed loop imports (import name : label).
MAIN_ENV_MODULES = {
    "blueprint_pipeline": "blueprint_pipeline",
    "mujoco": "mujoco",
    "zmq": "pyzmq",
    "msgpack_numpy": "msgpack_numpy",
    "imageio": "imageio",
    "PIL": "pillow",
    "huggingface_hub": "huggingface_hub",
}


def _dir_has_files(path: Path) -> bool:
    return path.is_dir() and any(path.rglob("*"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-time", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    blockers: list[str] = []
    payload: dict[str, Any] = {
        "schema_version": "groot_oscar_closed_loop_image_healthcheck.v1",
        "python": sys.version.split()[0],
        "build_time": args.build_time,
        "raw_secret_values_recorded": False,
    }

    # --- main env: torch (from base) ---
    try:
        import torch

        payload["torch_version"] = torch.__version__
        if not torch.__version__.startswith("2.10.0"):
            blockers.append("torch_version_not_2_10_0")
        if "+cu128" not in torch.__version__:
            blockers.append("torch_not_built_for_cu128")
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        if args.require_cuda and not torch.cuda.is_available():
            blockers.append("torch_cuda_unavailable")
    except Exception as exc:  # pragma: no cover - image-only path
        payload["torch_error_type"] = type(exc).__name__
        blockers.append("torch_import_failed")

    # --- main env: closed-loop imports ---
    payload["main_env_imports"] = {}
    for module, label in MAIN_ENV_MODULES.items():
        spec = importlib.util.find_spec(module)
        payload["main_env_imports"][label] = spec is not None
        if spec is None:
            blockers.append(f"{label}_not_importable")

    # --- OSCAR source on PYTHONPATH ---
    oscar_entrypoint = Path(OSCAR_REPO) / "inference" / "inference_oscar.py"
    payload["oscar_repo"] = OSCAR_REPO
    payload["oscar_inference_entrypoint_present"] = oscar_entrypoint.is_file()
    if not oscar_entrypoint.is_file():
        blockers.append("oscar_inference_entrypoint_missing")

    # --- GR00T env (out-of-process) ---
    payload["groot_venv_python"] = GROOT_VENV_PYTHON
    if Path(GROOT_VENV_PYTHON).exists():
        proc = subprocess.run(
            [GROOT_VENV_PYTHON, "-c", "from gr00t.policy.gr00t_policy import Gr00tPolicy"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        payload["groot_policy_importable"] = proc.returncode == 0
        if proc.returncode != 0:
            payload["groot_import_stderr_tail"] = proc.stderr[-500:]
            blockers.append("groot_policy_not_importable")
    else:
        payload["groot_policy_importable"] = False
        blockers.append("groot_venv_python_missing")

    # --- baked checkpoints ---
    for label, path in (("sonic", SONIC_CHECKPOINT), ("oscar", OSCAR_CHECKPOINT)):
        present = _dir_has_files(Path(path))
        payload[f"{label}_checkpoint_present"] = present
        payload[f"{label}_checkpoint_path"] = path
        if not present:
            blockers.append(f"{label}_checkpoint_missing")

    payload["status"] = "passed" if not blockers else "blocked"
    payload["blockers"] = blockers
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
