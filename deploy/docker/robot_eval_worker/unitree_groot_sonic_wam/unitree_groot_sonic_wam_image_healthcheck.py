from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


REQUIRED_IMPORTS = (
    "albumentations",
    "diffusers",
    "huggingface_hub",
    "msgpack",
    "msgpack_numpy",
    "numpy",
    "pydantic",
    "pydantic_core",
    "scipy",
    "torch",
    "transformers",
    "zmq",
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return _string(getattr(module, "__version__", "")) or None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-time", action="store_true")
    args = parser.parse_args()

    groot_root = Path(
        os.environ.get(
            "BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT",
            "/opt/blueprint/groot_runtime/Isaac-GR00T",
        )
    )
    checkpoint_root = Path(
        os.environ.get(
            "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT",
            "/opt/blueprint/groot_runtime/model_snapshots/LucaFrat__groot-bs16",
        )
    )
    missing_imports = [
        module for module in REQUIRED_IMPORTS if importlib.util.find_spec(module) is None
    ]
    expected_files = [
        groot_root / "gr00t" / "eval" / "run_gr00t_server.py",
        groot_root / "gr00t" / "policy" / "gr00t_policy.py",
        checkpoint_root / "config.json",
        checkpoint_root / "model.safetensors.index.json",
        checkpoint_root / "processor" / "processor_config.json",
    ]
    missing_files = [str(path) for path in expected_files if not path.is_file()]
    pydantic_version = _version("pydantic")
    pydantic_core_version = _version("pydantic_core")
    blockers: list[str] = []
    if missing_imports:
        blockers.append("unitree_groot_sonic_image_missing_python_imports")
    if missing_files:
        blockers.append("unitree_groot_sonic_image_missing_baked_runtime_files")
    if pydantic_version != "2.13.4" or pydantic_core_version != "2.46.4":
        blockers.append("unitree_groot_sonic_image_pydantic_pin_mismatch")

    payload = {
        "schema_version": "unitree_groot_sonic_wam_image_healthcheck.v1",
        "status": "completed" if not blockers else "blocked",
        "build_time": bool(args.build_time),
        "groot_root": str(groot_root),
        "checkpoint_root": str(checkpoint_root),
        "missing_imports": missing_imports,
        "missing_files": missing_files,
        "pydantic_version": pydantic_version,
        "pydantic_core_version": pydantic_core_version,
        "torch_version": _version("torch"),
        "blockers": blockers,
        "claim_boundary": {
            "image_healthcheck_is_not_policy_inference": True,
            "image_healthcheck_is_not_task_success": True,
            "gpu_server_startup_requires_provider_canary": True,
            "raw_secret_values_recorded": False,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
