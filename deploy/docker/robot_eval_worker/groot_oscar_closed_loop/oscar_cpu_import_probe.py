#!/usr/bin/env python3
"""Import the pinned OSCAR CLI on a CPU-only image builder.

The pinned upstream source evaluates ``torch.cuda.current_device()`` in a
default argument while the module is imported. A CPU image builder has no
NVIDIA driver, so patch only that discovery call for the duration of the
import. Runtime GPU admission and model execution remain separate and must
still prove a real CUDA device on the qualification provider.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from unittest import mock

import torch


def main() -> int:
    original_current_device = torch.cuda.current_device
    with mock.patch.object(torch.cuda, "current_device", return_value=0):
        inference_module = importlib.import_module("inference.inference_oscar")
        config_module = importlib.import_module(
            "worldsim._src.configs.agibot_control.config"
        )
        config = config_module.make_config()
    if torch.cuda.current_device is not original_current_device:
        raise RuntimeError("oscar_cpu_import_probe_cuda_patch_not_restored")
    if not getattr(inference_module, "__file__", None):
        raise RuntimeError("oscar_cpu_import_probe_module_identity_missing")
    if importlib.metadata.version("pytest") != "9.1.1":
        raise RuntimeError("oscar_cpu_import_probe_pytest_version_mismatch")
    if config is None:
        raise RuntimeError("oscar_cpu_import_probe_dynamic_config_missing")
    print("BLUEPRINT_OSCAR_CPU_IMPORT_PROBE_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
