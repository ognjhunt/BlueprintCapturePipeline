"""Launch native tasks with a digest-bound, Warp-compatible Isaac Lab experience.

The native Arena lane binds the official Isaac Lab 3.0 Beta 2 Patch 1 source
and Kit experience released for Isaac Sim 6.0.1.  Its experience excludes
Isaac Sim's bundled Warp extension so it cannot mix with the pinned external
Warp 1.13 runtime and make a CUDA PhysX view return CPU arrays.  The failure
only appears after an expensive native environment build, so this adapter
reverifies those exact experience bytes and makes them an explicit launch
input.

No scene, robot, object, or policy decision is made here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_runtime_source_packet import (
    ISAACLAB_REPOSITORY,
    ISAACLAB_RUNTIME_COMPATIBILITY_COMMIT,
    ISAACLAB_RUNTIME_COMPATIBILITY_TREE,
    ISAACLAB_RUNTIME_COMPATIBILITY_UPSTREAM_FIXES,
    RUNTIME_EXPERIENCE_RELATIVE_PATH,
)


# Every arena link (construction, controls, policy) runs on one device. Keeping
# it here means the three workers cannot drift apart, and the launcher, the
# preconstruction probe and the post-build readback all name the same string.
NATIVE_TASK_ARENA_DEVICE = "cuda:0"
NATIVE_TASK_ARENA_IMAGE = (
    "nvcr.io/nvidia/isaac-sim:6.0.1@"
    "sha256:b1c542b2ecc549b3d1ebb78c25664aa3bacba1709e6ad8e0a68e09426d57dedb"
)
NATIVE_TASK_ARENA_NUREC_EXTENSION = "omni.rtx.spg"
NATIVE_TASK_ARENA_NUREC_UTILS_EXTENSION = "isaacsim.replicator.nurec_utils"
NATIVE_TASK_ARENA_NUREC_SCHEMA = "OmniNuRecFieldAsset"
NATIVE_TASK_ARENA_NUREC_RENDER_PATH = "plain_nurec_volume"
NATIVE_TASK_ARENA_PARTICLEFIELD_RENDER_PATH = "particlefield_3d_gaussian_splat"
#: Every appearance backend the arena launcher may be told to expect.  The
#: launcher records the caller's choice; it never picks one.  Scene 839873's
#: render-only probe was launched with the ParticleField path while the policy
#: canary worker launched with no path at all and inherited the legacy NuRec
#: default, so the two receipts disagreed about which renderer a Website run
#: would use.  A default argument must not select the scientific representation.
#: A lane that composes no site appearance at all (control sweeps run with
#: ``appearance_mode: omitted`` and cameras disabled) says so explicitly.
NATIVE_TASK_ARENA_NO_APPEARANCE_RENDER_PATH = "no_site_appearance"
NATIVE_TASK_ARENA_APPEARANCE_RENDER_PATHS = frozenset(
    {
        NATIVE_TASK_ARENA_NUREC_RENDER_PATH,
        NATIVE_TASK_ARENA_PARTICLEFIELD_RENDER_PATH,
        NATIVE_TASK_ARENA_NO_APPEARANCE_RENDER_PATH,
    }
)
NATIVE_TASK_ARENA_UJITSO_GEOMETRY_SETTING = "/UJITSO/geometry"
# Omniverse RTX composites ParticleField prims "as-is": their light fields are
# display-referred sRGB and the tonemapping pipeline is skipped for them
# (Omniverse Materials and Rendering, "Gaussian Splats (Particle Fields)").
# NVIDIA's shipped ``nurec_config.yaml`` forces this flag off only for PPISP
# (``info:spg:sourceAsset``) stages and leaves plain gaussians at the engine
# default.  This lane forced it off at launch from 2026-08-20, and every world
# camera then rendered the sealed splat as linear radiance up to 60x display
# white (scene-839873 r13 construction receipt: 22-24 percent of world-camera
# pixels above 1.0, p99 = 20).  The LDR annotator clamps those per channel
# into white blobs with chromatic fringes, and that clamp is the exact frame
# the policies were fed.  The flag is therefore never forced here, and the
# launch receipt refuses a runtime that reports it forced off.
NATIVE_TASK_ARENA_GAUSSIAN_SKIP_TONEMAPPING_SETTING = (
    "/rtx/rtpt/gaussian/skipTonemapping/enabled"
)
NATIVE_TASK_ARENA_KIT_ARGS = (
    "--enable isaacsim.replicator.nurec_utils "
    "--enable omni.rtx.spg "
    "--enable isaacsim.robot_motion.pink "
    "--/UJITSO/geometry=true "
    "--/renderer/multiGpu/enabled=false"
)

SCHEMA_VERSION = "native_task_isaaclab_launch.v1"
PROVISIONING_SCHEMA_VERSION = "native_task_runtime_source_provisioning.v1"
REQUIRED_EXPERIENCE_FILES = (
    "isaaclab.python.kit",
    "isaaclab.python.headless.kit",
    "isaaclab.python.headless.rendering.kit",
)


class NativeTaskIsaacLabLaunchError(ValueError):
    """Stable fail-closed launch-contract errors."""

    def __init__(
        self,
        errors: list[str],
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ):
        self.errors = tuple(sorted(set(errors)))
        self.diagnostics = dict(diagnostics or {})
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
        or provisioning.get("python_executable_source")
        != "simulator_python_launcher"
        or provisioning.get("python_probe_flag") != "-P"
        or provisioning.get("python_probe_mode")
        != "simulator_wrapper_safe_path"
    ):
        errors.append("native_task_isaaclab_runtime_launcher_invalid")

    experience = dict(provisioning.get("runtime_experience") or {})
    expected_identity = {
        "relative_path": RUNTIME_EXPERIENCE_RELATIVE_PATH,
        "repository": ISAACLAB_REPOSITORY,
        "source_revision": ISAACLAB_RUNTIME_COMPATIBILITY_COMMIT,
        "source_tree": ISAACLAB_RUNTIME_COMPATIBILITY_TREE,
        "upstream_fix_revisions": list(
            ISAACLAB_RUNTIME_COMPATIBILITY_UPSTREAM_FIXES
        ),
    }
    if any(experience.get(key) != value for key, value in expected_identity.items()):
        errors.append("native_task_isaaclab_experience_revision_mismatch")

    raw_root = Path(str(provisioning.get("extraction_dir") or "")).expanduser()
    raw_experience = Path(str(experience.get("path") or "")).expanduser()
    try:
        root = raw_root.resolve(strict=True)
        experience_path = raw_experience.resolve(strict=True)
    except OSError:
        errors.append("native_task_isaaclab_experience_missing")
        root = raw_root.resolve()
        experience_path = raw_experience.resolve()
    expected_experience_path = (root / RUNTIME_EXPERIENCE_RELATIVE_PATH).resolve()
    outside = experience_path != root and root not in experience_path.parents
    if (
        not root.is_dir()
        or outside
        or experience_path != expected_experience_path
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
            {"filename": filename, "size_bytes": candidate.stat().st_size, "sha256": _sha256(candidate)}
        )
    base = texts.get("isaaclab.python.kit", "")
    headless = texts.get("isaaclab.python.headless.kit", "")
    if (
        '"isaacsim.core.simulation_manager" = {}' not in base
        or '"isaacsim.core.simulation_manager" = {}' not in headless
        or '"omni.warp.core" = {}' in base
        or '"omni.warp.core" = {}' in headless
        or "omni.warp.core" not in base
        or "omni.warp.core" not in headless
    ):
        errors.append("native_task_isaaclab_experience_warp_contract_invalid")
    installed_warp = [
        row
        for row in provisioning.get("runtime_dependencies_installed") or []
        if isinstance(row, Mapping) and row.get("package") == "warp-lang"
    ]
    import_warp = [
        row
        for row in provisioning.get("runtime_import_probes") or []
        if isinstance(row, Mapping) and row.get("module") == "warp"
    ]
    installed_torch = [
        row
        for row in provisioning.get("runtime_dependencies_installed") or []
        if isinstance(row, Mapping) and row.get("package") == "torch"
    ]
    import_torch = [
        row
        for row in provisioning.get("runtime_import_probes") or []
        if isinstance(row, Mapping) and row.get("module") == "torch"
    ]
    if (
        len(installed_warp) != 1
        or installed_warp[0].get("version") != "1.13.0"
        or installed_warp[0].get("pure_python") is not False
        or installed_warp[0].get("wheel_tag")
        != "py3-none-manylinux_2_28_x86_64"
    ):
        errors.append("native_task_isaaclab_external_warp_identity_invalid")
    if (
        provisioning.get("runtime_import_probe_returncode") != 0
        or len(import_warp) != 1
        or import_warp[0].get("available") is not True
        or import_warp[0].get("expected_version") != "1.13.0"
        or import_warp[0].get("observed_version") != "1.13.0"
        or import_warp[0].get("version_matches") is not True
    ):
        errors.append("native_task_isaaclab_external_warp_import_unqualified")
    if (
        len(installed_torch) != 1
        or installed_torch[0].get("version") != "2.10.0+cu128"
        or installed_torch[0].get("pure_python") is not False
        or installed_torch[0].get("wheel_tag")
        != "cp312-cp312-manylinux_2_28_x86_64"
    ):
        errors.append("native_task_isaaclab_torch_identity_invalid")
    if (
        provisioning.get("runtime_import_probe_returncode") != 0
        or len(import_torch) != 1
        or import_torch[0].get("available") is not True
        or import_torch[0].get("expected_version") != "2.10.0+cu128"
        or import_torch[0].get("observed_version") != "2.10.0+cu128"
        or import_torch[0].get("version_matches") is not True
        or import_torch[0].get("cuda_available") is not True
        or import_torch[0].get("expected_cuda_version") != "12.8"
        or import_torch[0].get("observed_cuda_version") != "12.8"
        or import_torch[0].get("cuda_version_matches") is not True
        or import_torch[0].get("cuda_tensor_device") != "cuda:0"
        or import_torch[0].get("cuda_tensor_operation_passed") is not True
    ):
        errors.append("native_task_isaaclab_torch_cuda_import_unqualified")
    if errors:
        raise NativeTaskIsaacLabLaunchError(errors)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified",
        "experience": {**expected_identity, "path": str(experience_path), "sha256": experience["sha256"]},
        "experience_files": file_rows,
        "bundled_isaac_sim_warp_extension_loaded": False,
        "external_warp": {
            "package": "warp-lang",
            "version": "1.13.0",
            "wheel_tag": "py3-none-manylinux_2_28_x86_64",
            "import_module": "warp",
            "import_qualified_before_simulation_app": True,
        },
        "torch": {
            "package": "torch",
            "version": "2.10.0+cu128",
            "wheel_tag": "cp312-cp312-manylinux_2_28_x86_64",
            "cuda_version": "12.8",
            "cuda_tensor_device": "cuda:0",
            "import_and_cuda_operation_qualified_before_simulation_app": True,
        },
        "isaac_simulation_manager_required": True,
        "device_coherence_still_requires_native_readback": True,
    }


def launch_native_task_isaaclab(
    provisioning_receipt_path: str | Path,
    *,
    device: str,
    appearance_render_path: str,
    enable_cameras: bool = True,
    app_launcher_factory: Callable[..., Any] | None = None,
    nurec_renderer_probe_factory: Callable[[], Mapping[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Launch Isaac Lab on ``device`` with the verified compatibility experience.

    ``appearance_render_path`` is required and must name the backend the scene
    plan actually composes (see
    :func:`blueprint_pipeline.native_task_nurec_render_setup.appearance_render_path_from_plan`).
    The launcher only records it; it does not switch renderers, so a wrong
    value here is a false receipt rather than a different picture, and that
    is exactly why it cannot default.

    The launch must go through Isaac Lab's ``AppLauncher``, not a bare
    ``SimulationApp``. ``AppLauncher`` is what resolves the requested device and
    applies Isaac Lab's GPU-pipeline settings before any physics exists.
    Launching bare leaves that configuration undone, and the consequence is
    invisible until much later: ``SimulationContext`` and ``PhysicsManager``
    both report ``cuda:0`` while the PhysX tensor views hand back CPU-backed
    arrays, and PhysX logs nothing because from its own side nothing is wrong.
    The first symptom is a Warp kernel refusing a CPU ``joint_vel`` an
    environment build later.

    Proven by a controlled A/B on one machine (2026-08-19): the same sealed
    plan, assets and GPU built cleanly with every articulation on ``cuda:0``
    under ``AppLauncher``, and reproduced the production failure exactly under
    a bare ``SimulationApp``.
    """

    if (
        not isinstance(appearance_render_path, str)
        or appearance_render_path not in NATIVE_TASK_ARENA_APPEARANCE_RENDER_PATHS
    ):
        raise NativeTaskIsaacLabLaunchError(
            ["native_task_isaaclab_appearance_render_path_invalid"],
            diagnostics={"appearance_render_path": appearance_render_path},
        )
    receipt = verify_native_task_isaaclab_launch_contract(
        provisioning_receipt_path
    )
    requested_device = str(device).strip()
    if not requested_device:
        raise NativeTaskIsaacLabLaunchError(["native_task_isaaclab_device_missing"])
    if app_launcher_factory is None:
        from isaaclab.app import AppLauncher

        app_launcher_factory = AppLauncher
    launcher = app_launcher_factory(
        headless=True,
        device=requested_device,
        enable_cameras=enable_cameras,
        experience=receipt["experience"]["path"],
        kit_args=NATIVE_TASK_ARENA_KIT_ARGS,
    )
    app = getattr(launcher, "app", launcher)
    if nurec_renderer_probe_factory is None:

        def nurec_renderer_probe_factory() -> Mapping[str, Any]:
            import carb
            import omni.kit.app
            from pxr import Usd

            extension_manager = omni.kit.app.get_app().get_extension_manager()
            extension_enabled = extension_manager.is_extension_enabled(
                NATIVE_TASK_ARENA_NUREC_EXTENSION
            )
            settings = carb.settings.get_settings()
            return {
                "render_path": appearance_render_path,
                "activation_method": "kit_prelaunch_extensions_no_dynamic_enable",
                # NVIDIA launches the SPG extension globally even when stage
                # classification later selects the plain ParticleField render
                # product. Loading the extension and using the SPG graph are
                # distinct claims.
                "extension_required": True,
                "extension_was_enabled_before_probe": extension_enabled,
                "extension_enabled": extension_enabled,
                "spg_dynamic_enable_attempted": False,
                "nurec_utils_extension_enabled": (
                    extension_manager.is_extension_enabled(
                        NATIVE_TASK_ARENA_NUREC_UTILS_EXTENSION
                    )
                ),
                "renderer_hints": settings.get(
                    "/omni/rtx/nre/compositing/rendererHints"
                ),
                "ujitso_geometry_enabled": settings.get(
                    NATIVE_TASK_ARENA_UJITSO_GEOMETRY_SETTING
                ),
                "multi_gpu_enabled": settings.get(
                    "/renderer/multiGpu/enabled"
                ),
                "gaussian_skip_tonemapping_enabled": settings.get(
                    NATIVE_TASK_ARENA_GAUSSIAN_SKIP_TONEMAPPING_SETTING
                ),
                "schema_registered": (
                    Usd.SchemaRegistry().FindConcretePrimDefinition(
                        NATIVE_TASK_ARENA_NUREC_SCHEMA
                    )
                    is not None
                ),
            }

    try:
        raw_nurec = dict(nurec_renderer_probe_factory())
    except Exception as exc:
        raise NativeTaskIsaacLabLaunchError(
            ["native_task_isaaclab_nurec_runtime_readback_failed"],
            diagnostics={"probe_error_type": type(exc).__name__},
        ) from exc
    nurec = {
        "extension_id": NATIVE_TASK_ARENA_NUREC_EXTENSION,
        "render_path": raw_nurec.get("render_path")
        or appearance_render_path,
        "activation_method": raw_nurec.get("activation_method"),
        "extension_required": raw_nurec.get("extension_required") is True,
        "extension_was_enabled_before_probe": raw_nurec.get(
            "extension_was_enabled_before_probe"
        ),
        "extension_enabled": raw_nurec.get("extension_enabled") is True,
        "spg_dynamic_enable_attempted": raw_nurec.get(
            "spg_dynamic_enable_attempted"
        )
        is True,
        "nurec_utils_extension_id": NATIVE_TASK_ARENA_NUREC_UTILS_EXTENSION,
        "nurec_utils_extension_enabled": raw_nurec.get(
            "nurec_utils_extension_enabled"
        )
        is True,
        "renderer_hints": raw_nurec.get("renderer_hints"),
        "renderer_hints_expected": 3,
        "ujitso_geometry_required": True,
        "ujitso_geometry_enabled": raw_nurec.get("ujitso_geometry_enabled")
        is True,
        "multi_gpu_enabled": raw_nurec.get("multi_gpu_enabled"),
        "gaussian_skip_tonemapping_setting": (
            NATIVE_TASK_ARENA_GAUSSIAN_SKIP_TONEMAPPING_SETTING
        ),
        # ``None`` is the engine default, which skips tonemapping for
        # ParticleField prims; only an explicit ``False`` is a forced-off flag.
        "gaussian_skip_tonemapping_enabled": raw_nurec.get(
            "gaussian_skip_tonemapping_enabled"
        ),
        "schema_type_name": NATIVE_TASK_ARENA_NUREC_SCHEMA,
        "schema_registration_required": raw_nurec.get(
            "schema_registration_required"
        ) is True,
        "schema_registered": raw_nurec.get("schema_registered") is True,
    }
    nurec_errors = []
    if nurec["extension_required"] and not nurec["extension_enabled"]:
        nurec_errors.append("native_task_isaaclab_nurec_extension_not_enabled")
    if not nurec["nurec_utils_extension_enabled"]:
        nurec_errors.append(
            "native_task_isaaclab_nurec_utils_extension_not_enabled"
        )
    if nurec["renderer_hints"] != 3:
        nurec_errors.append("native_task_isaaclab_nurec_renderer_hints_invalid")
    if not nurec["ujitso_geometry_enabled"]:
        nurec_errors.append("native_task_isaaclab_ujitso_geometry_not_enabled")
    if nurec["multi_gpu_enabled"] is not False:
        nurec_errors.append("native_task_isaaclab_nurec_multi_gpu_not_disabled")
    if nurec["gaussian_skip_tonemapping_enabled"] is False:
        nurec_errors.append(
            "native_task_isaaclab_gaussian_tonemapping_forced_off"
        )
    if nurec["schema_registration_required"] and not nurec["schema_registered"]:
        nurec_errors.append("native_task_isaaclab_nurec_schema_not_registered")
    if nurec_errors:
        # Do not call ``SimulationApp.close()`` here. In this runtime Kit may
        # terminate the process from close instead of returning, which erases
        # the worker's sealed failure receipt. The worker records these raw
        # diagnostics and process exit/atexit owns resource closure.
        raise NativeTaskIsaacLabLaunchError(nurec_errors, diagnostics=nurec)
    nurec["status"] = "qualified"
    receipt["launch"] = {
        "launcher": "isaaclab.app.AppLauncher",
        "requested_device": requested_device,
        "enable_cameras": bool(enable_cameras),
        "device_configured_by_launcher": True,
        "kit_args": NATIVE_TASK_ARENA_KIT_ARGS.split(),
    }
    receipt["nurec_renderer"] = nurec
    return app, receipt


__all__ = [
    "NATIVE_TASK_ARENA_APPEARANCE_RENDER_PATHS",
    "NATIVE_TASK_ARENA_DEVICE",
    "NATIVE_TASK_ARENA_GAUSSIAN_SKIP_TONEMAPPING_SETTING",
    "NATIVE_TASK_ARENA_IMAGE",
    "NATIVE_TASK_ARENA_KIT_ARGS",
    "NATIVE_TASK_ARENA_NO_APPEARANCE_RENDER_PATH",
    "NATIVE_TASK_ARENA_NUREC_EXTENSION",
    "NATIVE_TASK_ARENA_PARTICLEFIELD_RENDER_PATH",
    "NATIVE_TASK_ARENA_NUREC_SCHEMA",
    "NATIVE_TASK_ARENA_NUREC_UTILS_EXTENSION",
    "NATIVE_TASK_ARENA_UJITSO_GEOMETRY_SETTING",
    "NativeTaskIsaacLabLaunchError",
    "REQUIRED_EXPERIENCE_FILES",
    "SCHEMA_VERSION",
    "launch_native_task_isaaclab",
    "verify_native_task_isaaclab_launch_contract",
]
