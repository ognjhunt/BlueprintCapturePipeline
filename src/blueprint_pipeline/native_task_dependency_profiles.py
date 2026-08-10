"""Versioned dependency profiles for native task execution phases.

Arena's released package metadata describes its complete monorepo, including
embodiments, UI, remote providers, data conversion, and analysis tools that are
not reachable from a sealed Franka construction or control run.  Treating that
union as one runtime requirement makes an unrelated optional package a paid-run
blocker and encourages unsafe overlays on top of the paired Isaac image.

Profiles in this module are scene and task neutral.  They name the selected
execution graph and retain every deliberately unselected optional import so a
receipt never turns "not probed" into "available".
"""

from __future__ import annotations

from typing import Any


CONSTRUCTION_CONTROLS_DEPENDENCY_PROFILE = "native_task_construction_controls.v2"
CONSTRUCTION_CONTROLS_EXECUTION_MODES = ("construction_canary", "controls")

# Every third-party module reachable from the selected construction/control
# graph.  Before SimulationApp starts these names are *discovered* without
# executing module code.  The native workers import this entire tuple only
# after SimulationApp has established Kit/Carbonite/USD ownership.
CONSTRUCTION_CONTROLS_REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "numpy",
    "onnx",
    "prettytable",
    "toml",
    "hid",
    "gymnasium",
    "trimesh",
    "pyglet",
    "transformers",
    "einops",
    "warp",
    "matplotlib",
    "PIL",
    "botocore",
    "starlette",
    "debugpy",
    "flatdict",
    "flaky",
    "packaging",
    "psutil",
    "filelock",
    "h5py",
    "typing_extensions",
    "pydantic",
    "lazy_loader",
    "pinocchio",
    "pink",
    "daqp",
    "pxr.Usd",
    "usdex",
    "pytetwild",
    "hf_xet",
    "google.protobuf",
    "tensorboard",
    "scipy",
    "cloudpickle",
    "farama_notifications",
    "antlr4",
    "omegaconf",
    "hydra",
    "msgpack",
    "tensordict",
    "importlib_metadata",
    "zipp",
    "orjson",
    "pyvers",
    "git",
    "gitdb",
    "smmap",
    "requests",
    "charset_normalizer",
    "idna",
    "urllib3",
    "certifi",
    "tqdm",
    "termcolor",
    "yaml",
    "click",
    "rsl_rl",
)

# Importing any of these namespaces before SimulationApp is unsupported.  The
# v25 paid canary proved that a pre-app ``pxr.Usd`` import can make Kit abort in
# native code before a Python result can be written.
SIMULATION_APP_OWNED_MODULE_ROOTS = frozenset(("carb", "isaacsim", "omni", "pxr"))

# (module, capability owner, stable reason code).  A submodule gets its own row
# when an older worker explicitly imported it, even if the root package already
# appears, so regression tests cover the exact paid failure that was removed.
_DEFERRED_OPTIONAL_DEPENDENCIES = (
    (
        "lightwheel_sdk",
        "remote_asset_provider",
        "sealed_scene_and_asset_bytes_supplied",
    ),
    (
        "lightwheel_sdk.loader",
        "remote_asset_provider",
        "sealed_scene_and_asset_bytes_supplied",
    ),
    (
        "onnxruntime",
        "unselected_onnx_embodiment",
        "franka_panda_embodiment_selected",
    ),
    (
        "openai",
        "agentic_environment_generation",
        "frozen_scene_plan_supplied",
    ),
    (
        "pandas",
        "dataset_conversion",
        "no_dataset_conversion_in_execution_graph",
    ),
    (
        "sbi",
        "sensitivity_analysis",
        "no_sensitivity_analysis_in_execution_graph",
    ),
    (
        "vuer",
        "teleoperation_ui",
        "no_teleoperation_ui_in_execution_graph",
    ),
    (
        "zmq",
        "remote_policy_transport",
        "native_action_seam_selected",
    ),
)


def construction_controls_deferred_dependencies() -> list[dict[str, Any]]:
    """Return a fresh, deterministic receipt representation of deferred imports."""

    return [
        {
            "module": module,
            "capability_owner": owner,
            "reason_code": reason,
            "selection_state": "deferred_unselected",
            "required_for_profile": False,
        }
        for module, owner, reason in _DEFERRED_OPTIONAL_DEPENDENCIES
    ]


CONSTRUCTION_CONTROLS_DEFERRED_MODULES = frozenset(
    module for module, _owner, _reason in _DEFERRED_OPTIONAL_DEPENDENCIES
)


__all__ = [
    "CONSTRUCTION_CONTROLS_DEFERRED_MODULES",
    "CONSTRUCTION_CONTROLS_DEPENDENCY_PROFILE",
    "CONSTRUCTION_CONTROLS_EXECUTION_MODES",
    "CONSTRUCTION_CONTROLS_REQUIRED_MODULES",
    "SIMULATION_APP_OWNED_MODULE_ROOTS",
    "construction_controls_deferred_dependencies",
]
