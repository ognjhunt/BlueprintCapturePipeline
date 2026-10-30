"""Asset-only Isaac Lab startup; official converters, no site appearance graph.

Isaac Sim 6.0.1 importers need its Newton and Kit pip archives. Isaac Lab's
pinned headless experience excludes both to isolate simulation dependencies.
Permit those two archives only in the disposable converter process. The
separate controller diagnostic retains the original upstream exclusions.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback

from .native_task_isaaclab_launch import verify_native_task_isaaclab_launch_contract

HEADLESS_SHA256 = "sha256:e5d1f4e581f177526a1b90414d1139242c89234dc4eaef8c16fee5c931c788c6"
UPSTREAM_EXCLUSIONS = ["omni.kit.pip_archive", "omni.isaac.ml_archive", "isaacsim.pip.newton", "omni.warp.core"]
IMPORT_EXCLUSIONS = ["omni.isaac.ml_archive", "omni.warp.core"]
IMPORTERS = {"urdf": "isaacsim.asset.importer.urdf", "mjcf": "isaacsim.asset.importer.mjcf"}


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def prepare_experience(contract: dict, output: Path, source_format: str | None) -> dict:
    """Retain the exact upstream bytes; derive an import-only experience beside output."""
    if source_format is not None and source_format not in IMPORTERS:
        raise ValueError("embodiment_isaac_source_format_invalid")
    upstream = Path(contract["experience"]["path"]).parent / "isaaclab.python.headless.kit"
    original = upstream.read_bytes()
    if _sha(original) != HEADLESS_SHA256:
        raise ValueError("embodiment_isaac_headless_source_changed")
    exclusions = UPSTREAM_EXCLUSIONS if source_format is None else IMPORT_EXCLUSIONS
    experience = upstream
    if source_format is not None:
        text = original.decode("utf-8")
        before = "app.extensions.excluded = " + json.dumps(UPSTREAM_EXCLUSIONS)
        if text.count(before) != 1:
            raise ValueError("embodiment_isaac_headless_exclusions_changed")
        text = text.replace(before, "app.extensions.excluded = " + json.dumps(exclusions))
        # The copied experience lives outside Isaac Lab's apps directory.
        # Keep the upstream extension search paths pointing at verified source.
        text = text.replace("${app}", json.dumps(str(upstream.parent))[1:-1])
        output.mkdir(parents=True, exist_ok=True)
        # Kit resolves the experience filename as an extension identifier.
        experience = output / ("blueprint.embodiment.import." + source_format + ".kit")
        experience.write_text(text, encoding="utf-8")
    return {"profile": "controller_smoke" if source_format is None else "official_" + source_format + "_import",
            "upstream_path": str(upstream), "upstream_sha256": HEADLESS_SHA256,
            "path": str(experience), "sha256": _sha(experience.read_bytes()),
            "excluded_extensions": list(exclusions), "source_format": source_format}


def _runtime_readback(importer: str | None) -> dict:
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    extensions = {name: manager.is_extension_enabled(name) for name in UPSTREAM_EXCLUSIONS}
    imported = None
    if importer:
        extensions[importer] = manager.is_extension_enabled(importer)
        module = importlib.import_module(importer)
        imported = {"module": importer, "path": str(module.__file__),
                    "sha256": _sha(Path(module.__file__).read_bytes())}
    modules = {}
    for name in ("torch", "warp"):
        module = importlib.import_module(name)
        modules[name] = {"version": str(module.__version__), "path": str(module.__file__)}
    return {"extensions": extensions, "modules": modules, "importer": imported}


def launch_embodiment_isaaclab(provisioning_receipt_path: Path, *, output: Path,
                              source_format: str | None = None,
                              app_launcher_factory=None, runtime_probe=None) -> tuple:
    """Launch a converter or a separate no-camera controller diagnostic on CUDA."""
    contract = verify_native_task_isaaclab_launch_contract(provisioning_receipt_path)
    experience = prepare_experience(contract, output, source_format)
    importer = IMPORTERS.get(source_format)
    kit_args = "--/renderer/multiGpu/enabled=false"
    if importer:
        kit_args += " --enable " + importer
    if app_launcher_factory is None:
        from isaaclab.app import AppLauncher
        app_launcher_factory = AppLauncher
    launcher = app_launcher_factory(headless=True, device="cuda:0", enable_cameras=False,
                                    experience=experience["path"], kit_args=kit_args)
    app = getattr(launcher, "app", launcher)
    try:
        readback = (runtime_probe or _runtime_readback)(importer)
        if any(readback["extensions"].get(name) is not False for name in experience["excluded_extensions"]):
            raise ValueError("embodiment_isaac_excluded_extension_loaded")
        if importer and any(readback["extensions"].get(name) is not True
                            for name in (importer, "omni.kit.pip_archive", "isaacsim.pip.newton")):
            raise ValueError("embodiment_isaac_official_importer_not_enabled")
        if {name: value["version"] for name, value in readback["modules"].items()} != {
                "warp": "1.13.0", "torch": "2.10.0+cu128"}:
            raise ValueError("embodiment_isaac_runtime_dependency_changed")
        receipt = {"schema_version": "embodiment_isaac_launch.v1", "profile": experience,
                   "verified_runtime_source": contract, "kit_args": kit_args.split(),
                   "runtime_readback": readback, "device": "cuda:0", "enable_cameras": False,
                   "site_appearance_requested": False, "native_operation_still_required": True}
        output.mkdir(parents=True, exist_ok=True)
        (output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return app, receipt
    except Exception:
        # Isaac fast shutdown can exit the process. Print the actual bootstrap
        # error before close so the parent still retains the reason per body.
        traceback.print_exc()
        sys.stderr.flush()
        app.close()
        raise
