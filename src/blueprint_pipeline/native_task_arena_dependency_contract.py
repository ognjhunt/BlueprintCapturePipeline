"""Pinned import closure required by the native Task Arena provider runtime."""

from __future__ import annotations

import traceback
from typing import Any


NATIVE_TASK_ARENA_DEPENDENCY_IMPORTS = (
    "warp",
    "torch",
    "triton",
    "cuda",
    "filelock",
    "fsspec",
    "jinja2",
    "markupsafe",
    "mpmath",
    "networkx",
    "setuptools",
    "sympy",
    "numpy",
    "PIL.Image",
    "gymnasium",
    "lazy_loader",
    "cloudpickle",
    "farama_notifications",
    "packaging",
    "prettytable",
    "typing_extensions",
    "wcwidth",
    "h5py",
    "yaml",
    "toml",
    "antlr4",
    "omegaconf",
    "hydra",
    "hydra.core",
    "msgpack",
    "zmq",
    "tensordict",
    "importlib_metadata",
    "zipp",
    "orjson",
    "pyvers",
    "git",
    "gitdb",
    "smmap",
    "lightwheel_sdk",
    "lightwheel_sdk.loader",
    "requests",
    "charset_normalizer",
    "idna",
    "urllib3",
    "certifi",
    "tqdm",
    "termcolor",
    "click",
    "rsl_rl",
    "rsl_rl.runners",
    "pxr.Usd",
    "pxr.UsdPhysics",
    "pxr.UsdVol",
    "isaaclab",
    "isaaclab.controllers",
    "isaaclab.utils.math",
    "isaaclab_assets",
    "isaaclab_contrib",
    "isaaclab_experimental",
    "isaaclab_mimic",
    "isaaclab_newton",
    "isaaclab_ov",
    "isaaclab_physx",
    "isaaclab_physx.physics",
    "isaaclab_rl",
    "isaaclab_tasks",
    "isaaclab_tasks_experimental",
    "isaaclab_teleop",
    "isaaclab_visualizers",
    "isaaclab_arena",
    "isaaclab_arena_g1",
    "coloredlogs",
    "humanfriendly",
    "flatbuffers",
    "isaaclab_arena.environments.arena_env_builder",
    "isaacsim.robot_motion.experimental.motion_generation",
    "isaacsim.robot_motion.pink",
    "pinocchio",
    "pink",
    "qpsolvers",
    "osqp",
    "onnxruntime",
    "google.protobuf",
)

# These imports arrived with the G1 embodiment and its ONNX policy runtime.
# Retained Franka packets predate those wheels and do not use that embodiment.
G1_ONLY_DEPENDENCY_IMPORTS = frozenset({
    "isaaclab_arena_g1", "coloredlogs", "humanfriendly", "flatbuffers",
    "onnxruntime", "google.protobuf",
})


def probe_native_dependency_matrix(
    *, robot_id: str, dependency_imports, import_module, run
) -> dict[str, Any]:
    """Probe all worker imports and media tools in one retained receipt."""
    imports, blockers = [], []
    try:
        from blueprint_pipeline.native_task_arena_import_scope import (
            install_scoped_arena_embodiment,
        )
        embodiment_scope = install_scoped_arena_embodiment(robot_id)
    except Exception as exc:  # noqa: BLE001 - exact scope failure is evidence
        embodiment_scope = {
            "schema_version": "native_task_arena_embodiment_scope.v1",
            "robot_id": str(robot_id),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        blockers.append(f"native_task_arena_embodiment_scope_failed:{robot_id}")
    for name in dependency_imports:
        if name in G1_ONLY_DEPENDENCY_IMPORTS and robot_id != "unitree_g1":
            continue
        try:
            module = import_module(name)
            imports.append(
                {
                    "module": name,
                    "available": True,
                    "version": str(getattr(module, "__version__", "unreported")),
                }
            )
        except Exception as exc:  # noqa: BLE001 - exact missing matrix is evidence
            imports.append(
                {
                    "module": name,
                    "available": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            blockers.append(f"native_task_dependency_missing:{name}")
    tools = []
    for executable in ("ffmpeg", "ffprobe"):
        try:
            completed = run(
                [executable, "-version"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            tools.append(
                {
                    "executable": executable,
                    "available": False,
                    "returncode": None,
                    "version_line": "",
                }
            )
            blockers.append(f"native_task_dependency_missing:{executable}")
            continue
        tools.append(
            {
                "executable": executable,
                "available": completed.returncode == 0,
                "returncode": completed.returncode,
                "version_line": (
                    (completed.stdout or completed.stderr).splitlines() or [""]
                )[0],
            }
        )
        if completed.returncode != 0:
            blockers.append(f"native_task_dependency_missing:{executable}")
    return {
        "schema_version": "native_task_dependency_matrix.v1",
        "embodiment_scope": embodiment_scope,
        "imports": imports,
        "tools": tools,
        "all_required_available": not blockers,
        "blockers": sorted(set(blockers)),
    }
