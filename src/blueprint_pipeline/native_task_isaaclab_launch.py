"""Launch native tasks with one digest-bound upstream runtime stack.

Arena, Isaac Lab, its Kit experience, the simulator base image, and external
Warp are a compatibility unit.  Mixing revisions can make a CUDA PhysX view
return CPU arrays only after an expensive environment build.  This adapter
reverifies that unit and records live extension state after Kit starts.

No scene, robot, object, or policy decision is made here.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import sys
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from .decision_evidence_contracts import canonical_digest
from .native_task_dependency_profiles import (
    CONSTRUCTION_CONTROLS_DEFERRED_MODULES,
    CONSTRUCTION_CONTROLS_DEPENDENCY_PROFILE,
    CONSTRUCTION_CONTROLS_EXECUTION_MODES,
    construction_controls_deferred_dependencies,
)
from .native_task_runtime_source_packet import (
    ARENA_COMMIT,
    ARENA_ISAACLAB_SUBMODULE_PATH,
    ARENA_REPOSITORY,
    ISAAC_SIM_BASE_IMAGE,
    ISAAC_SIM_RUNTIME_IMAGE,
    ISAACLAB_REPOSITORY,
    ISAACLAB_RUNTIME_COMPATIBILITY_COMMIT,
    ISAACLAB_RUNTIME_COMPATIBILITY_TREE,
    ISAACLAB_RUNTIME_COMPATIBILITY_UPSTREAM_FIXES,
    RUNTIME_EXPERIENCE_RELATIVE_PATH,
)


SCHEMA_VERSION = "native_task_isaaclab_launch.v1"
PRE_APP_DEPENDENCY_SCHEMA_VERSION = "native_task_pre_app_dependency_matrix.v1"
PRE_APP_DEPENDENCY_FILENAME = "native_task_pre_app_dependency_matrix.v1.json"
PROVISIONING_SCHEMA_VERSION = "native_task_runtime_source_provisioning.v1"
REQUIRED_EXPERIENCE_FILES = (
    "isaaclab.python.kit",
    "isaaclab.python.headless.kit",
    "isaaclab.python.headless.rendering.kit",
)
ISAAC_SIM_DEFAULT_CALLBACKS_SETTING = (
    "/exts/isaacsim.core.simulation_manager/enable_default_callbacks"
)
ISAAC_SIM_DEFAULT_CALLBACKS_KIT_ARG = f"--{ISAAC_SIM_DEFAULT_CALLBACKS_SETTING}=false"
ISAAC_SIM_DEFAULT_CALLBACKS_UPSTREAM_FIX = "d81d2160220a4401be1d94f871c8f0b62e217acb"
BUNDLED_WARP_EXTENSION = "omni.warp.core"

# This is the pre-SimulationApp-safe portion of the selected Franka
# construction/control graph.  Arena's complete setup.py union also names UI,
# remote-provider, analysis, conversion, and unselected-embodiment packages;
# those are retained explicitly as deferred profile evidence below rather than
# allowed to block this execution graph.
PRE_APP_DEPENDENCY_IMPORTS = (
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
PRE_APP_VERSION_CONSTRAINTS = {
    "torch": ">=2.10",
    "torchvision": ">=0.25.0",
    "numpy": ">=2",
    "prettytable": "==3.3.0",
    "gymnasium": "==1.2.1",
    "transformers": "==4.57.6",
    "warp": "==1.13.0",
    "PIL": "==12.2.0",
    "typing_extensions": "==4.12.2",
    "h5py": ">=3.16.0",
    "tqdm": "==4.67.1",
}
PRE_APP_DISTRIBUTION_NAMES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "prettytable": "prettytable",
    "gymnasium": "gymnasium",
    "transformers": "transformers",
    "warp": "warp-lang",
    "PIL": "Pillow",
    "typing_extensions": "typing_extensions",
    "h5py": "h5py",
    "packaging": "packaging",
    "tqdm": "tqdm",
}

if set(PRE_APP_DEPENDENCY_IMPORTS) & CONSTRUCTION_CONTROLS_DEFERRED_MODULES:
    raise RuntimeError("native_task_dependency_profile_required_deferred_overlap")


class NativeTaskIsaacLabLaunchError(ValueError):
    """Stable fail-closed launch-contract errors."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(sorted(set(errors)))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _has_symlink_component(path: Path, *, root: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _module_version(
    name: str,
    module: Any,
    *,
    distribution_version_reader: Callable[[str], str],
) -> str:
    distribution = PRE_APP_DISTRIBUTION_NAMES.get(name)
    if distribution:
        try:
            return str(distribution_version_reader(distribution))
        except importlib.metadata.PackageNotFoundError:
            pass
    return str(
        getattr(module, "__version__", None) or getattr(module, "VERSION", None) or "unreported"
    )


def preflight_native_task_pre_app_dependencies(
    *,
    module_importer: Callable[[str], Any] = importlib.import_module,
    distribution_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, Any]:
    """Inventory every pre-app import in the selected execution profile."""

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    imported: dict[str, Any] = {}
    for name in PRE_APP_DEPENDENCY_IMPORTS:
        try:
            module = module_importer(name)
            imported[name] = module
            observed = _module_version(
                name,
                module,
                distribution_version_reader=distribution_version_reader,
            )
            constraint = PRE_APP_VERSION_CONSTRAINTS.get(name)
            version_matches: bool | None = None
            if constraint:
                try:
                    version_matches = Version(observed) in SpecifierSet(constraint)
                except InvalidVersion:
                    version_matches = False
                if not version_matches:
                    blockers.append(f"native_task_pre_app_version_mismatch:{name}")
            rows.append(
                {
                    "module": name,
                    "available": True,
                    "observed_version": observed,
                    "version_constraint": constraint,
                    "version_matches": version_matches,
                }
            )
        except Exception as exc:  # noqa: BLE001 - retain every missing import
            rows.append(
                {
                    "module": name,
                    "available": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            blockers.append(f"native_task_pre_app_dependency_missing:{name}")

    torch = imported.get("torch")
    torch_cuda: dict[str, Any] = {
        "available": False,
        "runtime_version": None,
        "device_count": 0,
        "device_name": None,
    }
    if torch is not None:
        try:
            cuda = torch.cuda
            available = bool(cuda.is_available())
            device_count = int(cuda.device_count())
            runtime_version = str(getattr(torch.version, "cuda", None) or "")
            torch_cuda = {
                "available": available,
                "runtime_version": runtime_version,
                "device_count": device_count,
                "device_name": (
                    str(cuda.get_device_name(0)) if available and device_count > 0 else None
                ),
            }
            if not available or device_count < 1:
                blockers.append("native_task_pre_app_torch_cuda_unavailable")
            if runtime_version != "12.8":
                blockers.append("native_task_pre_app_torch_cuda_version_mismatch")
        except Exception as exc:  # noqa: BLE001 - exact CUDA probe is evidence
            torch_cuda["error_type"] = type(exc).__name__
            torch_cuda["error"] = str(exc)
            torch_cuda["traceback"] = traceback.format_exc()
            blockers.append("native_task_pre_app_torch_cuda_probe_failed")

    result: dict[str, Any] = {
        "schema_version": PRE_APP_DEPENDENCY_SCHEMA_VERSION,
        "dependency_profile": CONSTRUCTION_CONTROLS_DEPENDENCY_PROFILE,
        "execution_modes": list(CONSTRUCTION_CONTROLS_EXECUTION_MODES),
        "status": "qualified" if not blockers else "blocked",
        "imports": rows,
        "import_count": len(rows),
        "all_profile_imports_attempted": len(rows) == len(PRE_APP_DEPENDENCY_IMPORTS),
        "all_declared_imports_attempted": len(rows) == len(PRE_APP_DEPENDENCY_IMPORTS),
        "deferred_optional_dependencies": construction_controls_deferred_dependencies(),
        "torch_cuda": torch_cuda,
        "simulation_app_started": False,
        "candidate_policy_queried": False,
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
        "matrix_digest": "",
    }
    result["matrix_digest"] = canonical_digest(result, digest_field="matrix_digest")
    return result


def _persist_pre_app_dependency_matrix(
    provisioning_receipt_path: str | Path,
    result: Mapping[str, Any],
) -> Path:
    path = (
        Path(provisioning_receipt_path).expanduser().resolve().parent / PRE_APP_DEPENDENCY_FILENAME
    )
    path.write_text(
        json.dumps(dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def verify_native_task_isaaclab_launch_contract(
    provisioning_receipt_path: str | Path,
) -> dict[str, Any]:
    """Reverify source identity and the complete experience inheritance chain."""

    path = Path(provisioning_receipt_path).expanduser().resolve()
    try:
        provisioning = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeTaskIsaacLabLaunchError(
            ["native_task_isaaclab_provisioning_receipt_invalid"]
        ) from exc
    errors: list[str] = []
    if (
        not isinstance(provisioning, Mapping)
        or provisioning.get("schema_version") != PROVISIONING_SCHEMA_VERSION
        or provisioning.get("status") != "completed"
        or provisioning.get("receipt_digest")
        != canonical_digest(provisioning, digest_field="receipt_digest")
    ):
        errors.append("native_task_isaaclab_provisioning_receipt_invalid")
    if (
        provisioning.get("python_executable") != "/isaac-sim/python.sh"
        or provisioning.get("python_executable_source") != "simulator_python_launcher"
        or provisioning.get("python_probe_flag") != "-P"
        or provisioning.get("python_probe_mode") != "simulator_wrapper_safe_path"
    ):
        errors.append("native_task_isaaclab_runtime_launcher_invalid")

    experience = dict(provisioning.get("runtime_experience") or {})
    expected_identity = {
        "relative_path": RUNTIME_EXPERIENCE_RELATIVE_PATH,
        "repository": ISAACLAB_REPOSITORY,
        "source_revision": ISAACLAB_RUNTIME_COMPATIBILITY_COMMIT,
        "source_tree": ISAACLAB_RUNTIME_COMPATIBILITY_TREE,
        "upstream_fix_revisions": list(ISAACLAB_RUNTIME_COMPATIBILITY_UPSTREAM_FIXES),
    }
    if any(experience.get(key) != value for key, value in expected_identity.items()):
        errors.append("native_task_isaaclab_experience_revision_mismatch")
    expected_pair = {
        "arena_repository": ARENA_REPOSITORY,
        "arena_revision": ARENA_COMMIT,
        "isaaclab_repository": ISAACLAB_REPOSITORY,
        "isaaclab_revision": ISAACLAB_RUNTIME_COMPATIBILITY_COMMIT,
        "isaaclab_submodule_path": ARENA_ISAACLAB_SUBMODULE_PATH,
        "simulator_base_image": ISAAC_SIM_BASE_IMAGE,
        "simulator_runtime_image": ISAAC_SIM_RUNTIME_IMAGE,
    }
    paired_stack = dict(provisioning.get("paired_stack") or {})
    if any(paired_stack.get(key) != value for key, value in expected_pair.items()):
        errors.append("native_task_isaaclab_paired_stack_mismatch")

    raw_root = Path(str(provisioning.get("extraction_dir") or "")).expanduser()
    raw_experience = Path(str(experience.get("path") or "")).expanduser()
    try:
        root = raw_root.resolve(strict=True)
        experience_path = raw_experience.resolve(strict=True)
    except OSError:
        errors.append("native_task_isaaclab_experience_missing")
        root = raw_root.resolve()
        experience_path = raw_experience.resolve()
    outside = experience_path != root and root not in experience_path.parents
    if (
        not root.is_dir()
        or outside
        or not experience_path.is_file()
        or (not outside and _has_symlink_component(experience_path, root=root))
        or _sha256(experience_path) != experience.get("sha256")
    ):
        errors.append("native_task_isaaclab_experience_identity_mismatch")

    apps_root = experience_path.parent
    file_rows: list[dict[str, Any]] = []
    texts: dict[str, str] = {}
    for filename in REQUIRED_EXPERIENCE_FILES:
        candidate = apps_root / filename
        if not candidate.is_file() or _has_symlink_component(candidate, root=root):
            errors.append(f"native_task_isaaclab_experience_dependency_missing:{filename}")
            continue
        data = candidate.read_text(encoding="utf-8")
        texts[filename] = data
        file_rows.append(
            {
                "filename": filename,
                "size_bytes": candidate.stat().st_size,
                "sha256": _sha256(candidate),
            }
        )
    base = texts.get("isaaclab.python.kit", "")
    headless = texts.get("isaaclab.python.headless.kit", "")
    if (
        '"isaacsim.core.simulation_manager" = {}' not in base
        or '"omni.warp.core" = {}' in base
        or '"omni.physx.bundle" = {}' not in base
        or '"omni.physx" = {}' not in headless
        or BUNDLED_WARP_EXTENSION not in base
        or BUNDLED_WARP_EXTENSION not in headless
    ):
        errors.append("native_task_isaaclab_experience_warp_contract_invalid")
    if (
        provisioning.get("runtime_dependency_owner") != "official_isaac_lab_complete_runtime"
        or provisioning.get("runtime_dependency_overlay_required") is not False
        or provisioning.get("runtime_dependencies_installed") != []
    ):
        errors.append("native_task_isaaclab_runtime_dependency_ownership_invalid")
    if (
        provisioning.get("runtime_import_probe_returncode") != 0
        or provisioning.get("runtime_import_probes") != []
    ):
        errors.append("native_task_isaaclab_early_dependency_probe_invalid")
    if errors:
        raise NativeTaskIsaacLabLaunchError(errors)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified",
        "experience": {
            **expected_identity,
            "path": str(experience_path),
            "sha256": experience["sha256"],
        },
        "experience_files": file_rows,
        "paired_stack": paired_stack,
        # Static experience inspection proves exclusion was requested.  Only
        # the live extension-manager readback after SimulationApp starts can
        # prove whether the bundled extension actually loaded.
        "bundled_isaac_sim_warp_extension_loaded": None,
        "external_warp": {
            "package": "warp-lang",
            "version_requirement": "==1.13.0",
            "runtime_owner": "official_isaac_lab_complete_runtime",
            "runtime_image": ISAAC_SIM_RUNTIME_IMAGE,
            "import_module": "warp",
            "import_qualified_before_simulation_app": False,
        },
        "direct_physx_registration_required": True,
        "device_coherence_still_requires_native_readback": True,
    }


def launch_native_task_isaaclab(
    provisioning_receipt_path: str | Path,
    *,
    simulation_app_factory: Callable[..., Any] | None = None,
    settings_reader: Callable[[str], Any] | None = None,
    extension_enabled_reader: Callable[[str], bool] | None = None,
    pre_app_dependency_probe: Callable[[], Mapping[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Launch SimulationApp with one verified PhysX lifecycle owner.

    Isaac Lab upstream now disables Isaac Sim's default simulation-manager
    callbacks *before* extension startup.  Those callbacks otherwise race the
    PhysxManager lifecycle and can invalidate or rebind its tensor view.  This
    harness launches ``SimulationApp`` directly, so it must carry the same
    upstream Kit setting explicitly and read it back after startup.
    """

    receipt = verify_native_task_isaaclab_launch_contract(provisioning_receipt_path)
    setting_prefix = f"--{ISAAC_SIM_DEFAULT_CALLBACKS_SETTING}="
    existing = [arg for arg in sys.argv if arg.startswith(setting_prefix)]
    if existing and existing != [ISAAC_SIM_DEFAULT_CALLBACKS_KIT_ARG]:
        raise NativeTaskIsaacLabLaunchError(
            ["native_task_isaaclab_default_callbacks_setting_conflict"]
        )
    try:
        pre_app = dict((pre_app_dependency_probe or preflight_native_task_pre_app_dependencies)())
    except Exception as exc:  # noqa: BLE001 - retain a probe implementation gap
        pre_app = {
            "schema_version": PRE_APP_DEPENDENCY_SCHEMA_VERSION,
            "dependency_profile": CONSTRUCTION_CONTROLS_DEPENDENCY_PROFILE,
            "execution_modes": list(CONSTRUCTION_CONTROLS_EXECUTION_MODES),
            "status": "blocked",
            "imports": [],
            "import_count": 0,
            "all_profile_imports_attempted": False,
            "all_declared_imports_attempted": False,
            "deferred_optional_dependencies": construction_controls_deferred_dependencies(),
            "torch_cuda": {},
            "simulation_app_started": False,
            "candidate_policy_queried": False,
            "blockers": [f"native_task_pre_app_dependency_probe_failed:{type(exc).__name__}"],
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "raw_secret_values_recorded": False,
            "matrix_digest": "",
        }
        pre_app["matrix_digest"] = canonical_digest(pre_app, digest_field="matrix_digest")
    pre_app_path = _persist_pre_app_dependency_matrix(
        provisioning_receipt_path,
        pre_app,
    )
    pre_app_imports = [row for row in pre_app.get("imports") or [] if isinstance(row, Mapping)]
    pre_app_modules = [str(row.get("module") or "") for row in pre_app_imports]
    pre_app_valid = bool(
        pre_app.get("schema_version") == PRE_APP_DEPENDENCY_SCHEMA_VERSION
        and pre_app.get("dependency_profile")
        == CONSTRUCTION_CONTROLS_DEPENDENCY_PROFILE
        and pre_app.get("execution_modes")
        == list(CONSTRUCTION_CONTROLS_EXECUTION_MODES)
        and pre_app.get("matrix_digest") == canonical_digest(pre_app, digest_field="matrix_digest")
        and pre_app.get("all_profile_imports_attempted") is True
        and pre_app.get("all_declared_imports_attempted") is True
        and pre_app.get("deferred_optional_dependencies")
        == construction_controls_deferred_dependencies()
        and pre_app.get("import_count") == len(PRE_APP_DEPENDENCY_IMPORTS)
        and len(pre_app_imports) == len(PRE_APP_DEPENDENCY_IMPORTS)
        and len(set(pre_app_modules)) == len(PRE_APP_DEPENDENCY_IMPORTS)
        and set(pre_app_modules) == set(PRE_APP_DEPENDENCY_IMPORTS)
        and pre_app.get("simulation_app_started") is False
        and pre_app.get("candidate_policy_queried") is False
    )
    if not pre_app_valid or pre_app.get("status") != "qualified":
        errors = list(pre_app.get("blockers") or [])
        if not pre_app_valid:
            errors.append("native_task_pre_app_dependency_matrix_invalid")
        raise NativeTaskIsaacLabLaunchError(errors)
    receipt["pre_app_dependency_matrix"] = {
        **pre_app,
        "path": str(pre_app_path),
    }
    warp_rows = [row for row in pre_app_imports if row.get("module") == "warp"]
    if (
        len(warp_rows) != 1
        or warp_rows[0].get("available") is not True
        or warp_rows[0].get("observed_version") != "1.13.0"
        or warp_rows[0].get("version_constraint") != "==1.13.0"
        or warp_rows[0].get("version_matches") is not True
    ):
        raise NativeTaskIsaacLabLaunchError(
            ["native_task_isaaclab_external_warp_import_unqualified"]
        )
    receipt["external_warp"].update(
        {
            "observed_version": warp_rows[0]["observed_version"],
            "import_qualified_before_simulation_app": True,
            "qualification_matrix_digest": pre_app["matrix_digest"],
        }
    )
    if simulation_app_factory is None:
        from isaacsim.simulation_app import SimulationApp

        simulation_app_factory = SimulationApp
    inserted = not existing
    if inserted:
        sys.argv.append(ISAAC_SIM_DEFAULT_CALLBACKS_KIT_ARG)
    try:
        app = simulation_app_factory(
            {"headless": True, "renderer": "RayTracedLighting"},
            experience=receipt["experience"]["path"],
        )
    finally:
        if inserted:
            sys.argv.remove(ISAAC_SIM_DEFAULT_CALLBACKS_KIT_ARG)

    if settings_reader is None:
        import carb

        settings_reader = carb.settings.get_settings().get
    if extension_enabled_reader is None:
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_enabled_reader = extension_manager.is_extension_enabled
    try:
        observed = settings_reader(ISAAC_SIM_DEFAULT_CALLBACKS_SETTING)
        bundled_warp_loaded = bool(extension_enabled_reader(BUNDLED_WARP_EXTENSION))
    except Exception as exc:
        close = getattr(app, "close", None)
        if callable(close):
            close()
        raise NativeTaskIsaacLabLaunchError(
            ["native_task_isaaclab_live_extension_readback_failed"]
        ) from exc
    if observed is not False or bundled_warp_loaded:
        close = getattr(app, "close", None)
        if callable(close):
            close()
        errors = []
        if observed is not False:
            errors.append("native_task_isaaclab_default_callbacks_readback_failed")
        if bundled_warp_loaded:
            errors.append("native_task_isaaclab_bundled_warp_extension_live")
        raise NativeTaskIsaacLabLaunchError(errors)
    receipt["bundled_isaac_sim_warp_extension_loaded"] = bundled_warp_loaded
    receipt["bundled_warp_extension_readback"] = {
        "extension_id": BUNDLED_WARP_EXTENSION,
        "observed_enabled": bundled_warp_loaded,
        "observed_after_simulation_app_startup": True,
    }
    receipt["simulation_manager_lifecycle"] = {
        "owner": "isaaclab_physx.PhysxManager",
        "setting": ISAAC_SIM_DEFAULT_CALLBACKS_SETTING,
        "requested_value": False,
        "observed_value": observed,
        "applied_before_extension_startup": True,
        "upstream_fix_revision": ISAAC_SIM_DEFAULT_CALLBACKS_UPSTREAM_FIX,
    }
    receipt["launch_receipt_digest"] = canonical_digest(
        receipt, digest_field="launch_receipt_digest"
    )
    return app, receipt


__all__ = [
    "BUNDLED_WARP_EXTENSION",
    "NativeTaskIsaacLabLaunchError",
    "ISAAC_SIM_DEFAULT_CALLBACKS_KIT_ARG",
    "ISAAC_SIM_DEFAULT_CALLBACKS_SETTING",
    "ISAAC_SIM_DEFAULT_CALLBACKS_UPSTREAM_FIX",
    "PRE_APP_DEPENDENCY_FILENAME",
    "PRE_APP_DEPENDENCY_IMPORTS",
    "PRE_APP_DEPENDENCY_SCHEMA_VERSION",
    "PRE_APP_VERSION_CONSTRAINTS",
    "CONSTRUCTION_CONTROLS_DEPENDENCY_PROFILE",
    "REQUIRED_EXPERIENCE_FILES",
    "SCHEMA_VERSION",
    "launch_native_task_isaaclab",
    "preflight_native_task_pre_app_dependencies",
    "verify_native_task_isaaclab_launch_contract",
]
