"""Exact Isaac 6.0.1 PINK metadata adaptation for external Warp 1.13.

SDK Python, plugins and robot assets are unchanged. Only the four dependency
edges naming bundled Warp are redirected to a verified external-Warp provider.
Explicit Kit extension paths must be read back before controller use.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
import tempfile

MANIFEST = Path(__file__).parent / "config/isaacsim_pink_601.json"
MANIFEST_SHA256 = "a53b881e108e90152af2d6d824e6f8c0c7c4fbba7918b32a0986a1ec41510730"
PROVIDER = "blueprint.external_warp"
_PROFILE = None


def _sha(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _manifest():
    data = MANIFEST.read_bytes()
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise ValueError("pink_source_manifest_changed")
    return json.loads(data)


def _verify_blob(path, row):
    data = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if len(data) != row["size_bytes"] or blob != row["git_blob_sha1"]:
        raise ValueError("pink_sdk_source_changed:" + row["path"])


def _kit_experience(apps, output, enable_cameras):
    # Both parent and child declare exclusions. Give the derived parent its own
    # name so no search-order accident can select the unmodified parent instead.
    original_excluded = '["omni.kit.pip_archive", "omni.isaac.ml_archive", "isaacsim.pip.newton", "omni.warp.core"]'
    new_excluded = '["omni.isaac.ml_archive", "isaacsim.pip.newton", "omni.warp.core"]'
    records = []
    for source_name, target_name in (("isaaclab.python.headless.kit", "blueprint.native.headless.kit"),
                                      ("isaaclab.python.headless.rendering.kit", "blueprint.native.camera.kit")):
        original = (apps / source_name).read_bytes().decode("utf-8")
        expected = {"isaaclab.python.headless.kit": "sha256:e5d1f4e581f177526a1b90414d1139242c89234dc4eaef8c16fee5c931c788c6",
                    "isaaclab.python.headless.rendering.kit": "sha256:97c33b410d81c4e95ccea680ca892f10179a1f26d74c4513ddfe4688cff6a88a"}
        if _sha(original.encode()) != expected[source_name]:
            raise ValueError("pink_experience_source_changed")
        if original.count("app.extensions.excluded = " + original_excluded) != 1:
            raise ValueError("pink_experience_exclusions_changed")
        text = original.replace(original_excluded, new_excluded)
        text = text.replace('"isaaclab.python.headless" = {}', '"blueprint.native.headless" = {}')
        text = text.replace("${app}", json.dumps(str(apps))[1:-1])
        target = output / target_name
        target.write_text(text)
        records.append({"source": str(apps / source_name), "source_sha256": _sha(original.encode()),
                        "path": str(target), "sha256": _sha(text.encode())})
    return str(output / ("blueprint.native.camera.kit" if enable_cameras else "blueprint.native.headless.kit")), records


def prepare_pink_runtime(provisioning_path: Path, *, enable_cameras: bool, sdk_root=Path("/isaac-sim")):
    global _PROFILE
    from .native_task_isaaclab_launch import verify_native_task_isaaclab_launch_contract
    contract = verify_native_task_isaaclab_launch_contract(provisioning_path)
    provision = json.loads(provisioning_path.read_text())
    extraction = Path(provision["extraction_dir"]).resolve(strict=True)
    dependency_target = Path(provision["runtime_dependency_target"]).resolve(strict=True)
    if dependency_target != extraction / "runtime_python_dependencies":
        raise ValueError("pink_external_warp_target_unbound")
    source = _manifest()
    overlay_root = Path(tempfile.mkdtemp(prefix="blueprint-pink-", dir=extraction))
    profiles = []
    for name, row in source["extensions"].items():
        original = sdk_root / "exts" / name
        config = original / "config/extension.toml"
        raw = config.read_bytes().decode("utf-8")
        if raw != row["config"]:
            raise ValueError("pink_sdk_extension_config_changed:" + name)
        for record in row["files"]:
            _verify_blob(original / record["path"], record)
        # Links point only at this immutable image's exact extension, never a
        # client path. They are runtime scaffolding, not published model assets.
        overlay = overlay_root / name
        overlay.mkdir()
        for child in original.iterdir():
            if child.name != "config":
                (overlay / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
        (overlay / "config").mkdir()
        for child in (original / "config").iterdir():
            if child.name != "extension.toml":
                (overlay / "config" / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
        needle = '"omni.warp.core" = {}'
        substituted = raw.count(needle)
        if substituted not in (0, 1):
            raise ValueError("pink_warp_dependency_ambiguous")
        modified = raw.replace(needle, '"' + PROVIDER + '" = {}')
        (overlay / "config/extension.toml").write_text(modified)
        profiles.append({"extension_id": name, "source_path": str(original), "overlay_path": str(overlay),
                         "source_config_sha256": _sha(raw.encode()), "overlay_config_sha256": _sha(modified.encode()),
                         "external_warp_substitution": bool(substituted), "verified_python_file_count": len(row["files"])})
    if sum(row["external_warp_substitution"] for row in profiles) != 4:
        raise ValueError("pink_warp_dependency_set_changed")
    _verify_blob(sdk_root / "exts/isaacsim.robot_motion.pink" / source["franka_urdf"]["path"], source["franka_urdf"])
    provider = overlay_root / PROVIDER
    (provider / "config").mkdir(parents=True)
    (provider / "config/extension.toml").write_text(
        '[package]\nversion="1.0.0"\n[[python.module]]\nname="blueprint_external_warp"\n')
    (provider / "blueprint_external_warp.py").write_text(
        'from pathlib import Path\nimport warp\n'
        f'if warp.__version__ != "1.13.0" or Path(warp.__file__).resolve() != Path({str(dependency_target / "warp/__init__.py")!r}):\n'
        '    raise RuntimeError("pink_external_warp_provider_identity_changed")\n')
    apps = Path(contract["experience"]["path"]).parent
    experience, experience_records = _kit_experience(apps, overlay_root, enable_cameras)
    paths = [provider, *(Path(row["overlay_path"]) for row in profiles)]
    if any(any(character.isspace() for character in str(path)) for path in [overlay_root, *paths]):
        raise ValueError("pink_kit_extension_path_contains_whitespace")
    kit_args = "--ext-folder " + str(overlay_root) + " " + " ".join("--ext-path " + str(path) for path in paths)
    kit_args += " --enable isaacsim.robot_motion.pink"
    _PROFILE = {"source_revision": source["source_revision"], "source_manifest_sha256": "sha256:" + MANIFEST_SHA256,
                "dependency_target": str(dependency_target),
                "extensions": profiles, "external_warp_provider": str(provider), "experience_path": experience,
                "experiences": experience_records, "kit_args": kit_args, "native_readback_required": True}
    return _PROFILE


def readback_pink_runtime(manager):
    if _PROFILE is None:
        raise ValueError("pink_compatibility_profile_missing")
    identities = []
    for row in _PROFILE["extensions"]:
        enabled_id = manager.get_enabled_extension_id(row["extension_id"])
        resolved = manager.get_extension_path(enabled_id) if enabled_id else None
        if not resolved or Path(resolved).resolve() != Path(row["overlay_path"]):
            raise ValueError("pink_runtime_extension_precedence_mismatch:" + row["extension_id"])
        if _sha((Path(resolved) / "config/extension.toml").read_bytes()) != row["overlay_config_sha256"]:
            raise ValueError("pink_runtime_overlay_changed")
        module = importlib.import_module(row["extension_id"])
        expected_python = Path(row["source_path"]) / row["extension_id"].replace(".", "/") / "__init__.py"
        if Path(module.__file__).resolve() != expected_python.resolve():
            raise ValueError("pink_runtime_python_origin_changed:" + row["extension_id"])
        identities.append({"extension_id": row["extension_id"], "enabled_id": enabled_id, "path": str(resolved)})
    if not manager.is_extension_enabled(PROVIDER):
        raise ValueError("pink_external_warp_provider_not_loaded")
    from isaacsim.robot_motion.pink import PinkIKController, load_pink_supported_robot
    pink_source = Path(next(row["source_path"] for row in _PROFILE["extensions"]
                            if row["extension_id"] == "isaacsim.robot_motion.pink"))
    if Path(inspect.getfile(PinkIKController)).resolve() != pink_source / "isaacsim/robot_motion/pink/impl/pink_ik_controller.py":
        raise ValueError("pink_controller_python_origin_changed")
    robot = load_pink_supported_robot("franka")
    controller = PinkIKController(pink_robot=robot, robot_joint_space=list(robot.controlled_joint_names),
        robot_site_space=["panda_hand"], tool_frame="panda_hand", solver="osqp", dt=1. / 60.)
    if controller.get_frame_task() is None:
        raise ValueError("pink_controller_construction_failed")
    modules = {}
    for name in ("isaacsim.robot_motion.pink", "pinocchio", "pink", "qpsolvers", "osqp"):
        module = importlib.import_module(name)
        modules[name] = {"path": str(Path(module.__file__).resolve()),
                         "version": str(getattr(module, "__version__", "unreported"))}
        roots = (pink_source, Path(_PROFILE["dependency_target"]),
                 pink_source.parent / "omni.isaac.core_archive/pip_prebundle")
        if not any(Path(module.__file__).resolve().is_relative_to(root.resolve()) for root in roots):
            raise ValueError("pink_python_dependency_origin_changed:" + name)
    return {"extensions": identities, "modules": modules, "controller_constructed": True,
            "controller_class": PinkIKController.__module__ + "." + PinkIKController.__name__,
            "solver_step_executed": False, "task_executed": False}
