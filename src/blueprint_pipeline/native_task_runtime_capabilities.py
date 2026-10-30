"""Explicit controller requirements and post-Kit identities for native tasks."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

CONTROLLERS = frozenset({"joint", "g1_joint", "pink_franka"})
PINK_IMPORTS = frozenset({"isaacsim.robot_motion.experimental.motion_generation",
    "isaacsim.robot_motion.pink", "pinocchio", "pink", "qpsolvers", "osqp"})
EXCLUDED_ARCHIVES = ("omni.isaac.ml_archive", "isaacsim.pip.newton", "omni.warp.core")


def require_controller(value: str) -> str:
    if value not in CONTROLLERS:
        raise ValueError("native_task_controller_backend_invalid")
    return value


def policy_controller(*, team_binding=None, robot_id: str) -> str:
    if robot_id == "unitree_g1":
        return "g1_joint"
    return "joint" if team_binding is not None else "pink_franka"


def controller_imports(imports, g1_imports, *, robot_id, controller_backend):
    require_controller(controller_backend)
    pure_pink = {"pinocchio", "pink", "qpsolvers", "osqp"}
    return tuple(name for name in imports
                 if (name not in PINK_IMPORTS or controller_backend == "pink_franka"
                     or (robot_id == "unitree_g1" and name in pure_pink))
                 and (name not in g1_imports or robot_id == "unitree_g1"))


def prepare_g1_python_dependencies(sdk_root=Path("/isaac-sim")):
    """Arena imports pure Pink even for SONIC; do not activate its SDK controller.

    Append only the immutable image's Python bundles. The retained source
    packet's Pinocchio/CMeel and Torch/Warp paths keep priority.
    """
    import sys
    paths = [sdk_root / "exts/isaacsim.robot_motion.pink/pip_prebundle",
             sdk_root / "exts/omni.isaac.core_archive/pip_prebundle"]
    for path in paths:
        if not path.is_dir():
            raise ValueError("g1_image_python_dependency_bundle_missing")
        if str(path) not in sys.path:
            sys.path.append(str(path))
    return [str(path) for path in paths]


def post_start_identity(provisioning_path: Path, *, controller_backend: str) -> dict:
    """Read the actually loaded packages and GPU interop after Kit changes paths."""
    import omni.kit.app
    import torch
    import warp

    provision = json.loads(Path(provisioning_path).read_text())
    target = Path(provision["runtime_dependency_target"]).resolve(strict=True)
    manager = omni.kit.app.get_app().get_extension_manager()
    modules = {}
    for name, version in (("torch", "2.10.0+cu128"), ("warp", "1.13.0")):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve(strict=True)
        expected = target / name / "__init__.py"
        modules[name] = {"version": str(module.__version__), "path": str(path),
                         "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                         "retained_dependency_target": str(target), "origin_matches": path == expected,
                         "version_matches": str(module.__version__) == version}
    tensor = torch.arange(3, dtype=torch.float32, device="cuda:0")
    view = warp.from_torch(tensor)
    restored = warp.to_torch(view)
    gpu = {"torch_device": str(restored.device), "warp_device": str(view.device),
           "cuda_version": str(torch.version.cuda),
           "shared_pointer": restored.data_ptr() == tensor.data_ptr(),
           "sum": float((restored + 1).sum().item())}
    result = {"modules": modules, "gpu_interop": gpu,
              "excluded_extensions": {name: manager.is_extension_enabled(name) for name in EXCLUDED_ARCHIVES},
              "controller_backend": controller_backend}
    if controller_backend == "pink_franka":
        from .native_task_pink_compatibility import readback_pink_runtime
        result["pink"] = readback_pink_runtime(manager)
    elif controller_backend == "g1_joint":
        paths = prepare_g1_python_dependencies()
        python_modules = {}
        for name in ("pink", "pinocchio", "qpsolvers", "osqp"):
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            roots = [target] if name == "pinocchio" else [*(Path(value).resolve() for value in paths), target]
            python_modules[name] = {"path": str(path), "version": str(getattr(module, "__version__", "unreported")),
                                    "origin_bound": any(path.is_relative_to(root) for root in roots)}
        result["g1_python_dependencies"] = python_modules
        result["sdk_pink_extension_enabled"] = manager.is_extension_enabled("isaacsim.robot_motion.pink")
    return result


def validate_post_start_identity(readback: dict, *, controller_backend: str) -> None:
    for name, expected in (("torch", "2.10.0+cu128"), ("warp", "1.13.0")):
        row = readback.get("modules", {}).get(name, {})
        if row.get("version") != expected or row.get("origin_matches") is not True or not row.get("sha256"):
            raise ValueError("native_task_post_start_package_identity_invalid:" + name)
    if any(readback.get("excluded_extensions", {}).get(name) is not False for name in EXCLUDED_ARCHIVES):
        raise ValueError("native_task_post_start_conflicting_extension_loaded")
    gpu = readback.get("gpu_interop", {})
    if (gpu.get("torch_device") != "cuda:0" or gpu.get("warp_device") != "cuda:0"
            or gpu.get("cuda_version") != "12.8" or gpu.get("shared_pointer") is not True or gpu.get("sum") != 6.):
        raise ValueError("native_task_post_start_gpu_interop_failed")
    if controller_backend == "pink_franka" and readback.get("pink", {}).get("controller_constructed") is not True:
        raise ValueError("native_task_required_pink_controller_unavailable")
    if controller_backend == "g1_joint" and (readback.get("sdk_pink_extension_enabled") is not False
            or any(readback.get("g1_python_dependencies", {}).get(name, {}).get("origin_bound") is not True
                   for name in ("pink", "pinocchio", "qpsolvers", "osqp"))):
        raise ValueError("native_task_g1_python_dependency_identity_invalid")
