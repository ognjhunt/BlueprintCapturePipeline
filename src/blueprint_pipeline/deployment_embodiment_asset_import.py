"""Convert a retained publisher source with the installed official Isaac importer.

This prepares source geometry, not a complete mounted robot or an admitted run.
Call inside the existing initialized Isaac worker; this module never starts an
application, installs a runtime, allocates resources or downloads model bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .deployment_embodiment_assets import AssetPreparationError, recipe_for, verify_source_bundle
from .team_embodiment import read_model_structure
from .immutable_directory_publication import (
    ImmutableDirectoryPublicationError,
    publish_staged_immutable_directory,
)

RECEIPT = "asset-import.json"
SCHEMA = "embodiment_asset_import.v1"
_FIND = re.compile(r"\$\(find\s+([A-Za-z0-9_]+)\)")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _package_roots(source: Path, mappings: dict[str, str]) -> dict[str, Path]:
    roots = {}
    for name, relative in mappings.items():
        path = source / relative
        if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or path.is_symlink()
                or not path.resolve().is_relative_to(source.resolve()) or not path.is_dir()):
            raise AssetPreparationError("asset_import_package_root_invalid")
        roots[name] = path.resolve()
    return roots


def _expand_xacro(entry: Path, *, source: Path, work: Path, arguments: dict[str, str],
                  roots: dict[str, Path]) -> Path:
    try:
        xacro = importlib.import_module("xacro")
    except ImportError as exc:
        raise AssetPreparationError("asset_import_xacro_runtime_unavailable") from exc
    # Work only on a disposable source copy. The retained publisher bytes and
    # receipt remain unchanged. ROS aliases resolve solely to this exact copy.
    shutil.copytree(source, work)
    aliases = {name: work / path.relative_to(source.resolve()) for name, path in roots.items()}
    for path in work.rglob("*"):
        if not path.is_file() or path.suffix not in {".xacro", ".urdf", ".xml"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "$(env " in text or "$(optenv " in text:
            raise AssetPreparationError("asset_import_xacro_environment_substitution_refused")

        def resolve(match):
            if match[1] not in aliases:
                raise AssetPreparationError("asset_import_xacro_package_alias_missing:" + match[1])
            return str(aliases[match[1]])

        path.write_text(_FIND.sub(resolve, text), encoding="utf-8")
    expanded = work / entry.relative_to(source)
    try:
        document = xacro.process_file(str(expanded), mappings=dict(arguments))
    except Exception as exc:
        raise AssetPreparationError("asset_import_xacro_expansion_failed:" + type(exc).__name__) from exc
    output = expanded.with_suffix(".urdf")
    output.write_text(document.toxml(), encoding="utf-8")
    return output


def _converter(format_name: str, entry: Path, output: Path, *, fix_base: bool,
               packages: dict[str, Path]):
    try:
        converters = importlib.import_module("isaaclab.sim.converters")
    except ImportError as exc:
        raise AssetPreparationError("asset_import_initialized_isaac_runtime_unavailable") from exc
    common = {"asset_path": str(entry), "usd_dir": str(output), "force_usd_conversion": True,
              "make_instanceable": False, "fix_base": fix_base, "link_density": 0.0,
              "collision_from_visuals": False, "collision_type": "Convex Hull", "merge_mesh": False,
              "run_asset_transformer": False, "run_multi_physics_conversion": True}
    if format_name == "urdf":
        options = {**common, "merge_fixed_joints": False, "joint_drive": None,
                   "ros_package_paths": [{"name": name, "path": str(path)} for name, path in packages.items()]}
        cfg_type, converter_type = converters.UrdfConverterCfg, converters.UrdfConverter
    else:
        options = {**common, "import_physics_scene": False,
                   "override_gain_type": None, "override_bias_type": None,
                   "override_gain_prm": None, "override_bias_prm": None}
        cfg_type, converter_type = converters.MjcfConverterCfg, converters.MjcfConverter
    try:
        converted = converter_type(cfg_type(**options))
    except Exception as exc:
        raise AssetPreparationError("asset_import_official_converter_failed:" + type(exc).__name__) from exc
    versions = {}
    for package in ("isaaclab", "isaacsim", "isaacsim-core"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    # Bind the loaded wrapper itself even when an editable runtime has no wheel
    # version. Do not claim a source commit just because the repo expects it.
    module = importlib.import_module(converter_type.__module__)
    module_file = getattr(module, "__file__", None)
    identity = {"class": converter_type.__module__ + "." + converter_type.__name__,
                "module_sha256": _sha(Path(module_file)) if module_file and Path(module_file).is_file() else None,
                "package_versions": versions}
    return Path(converted.usd_path), options, identity


def _dependencies(entry: Path, *, staged: Path, source: Path, source_receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Inspect local layers before stage composition; never resolve an unbound URL/file."""
    from pxr import Sdf

    allowed_source = {(source / row["path"]).resolve() for row in source_receipt["files"]}
    found: dict[str, dict[str, Any]] = {}
    visited = set()

    def inspect(path: Path):
        resolved = path.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        if (path.is_symlink() or not path.is_file()
                or not (resolved.is_relative_to(staged.resolve()) or resolved in allowed_source)):
            raise AssetPreparationError("asset_import_unbound_dependency")
        internal = resolved.is_relative_to(staged.resolve())
        relative = str(resolved.relative_to(staged.resolve() if internal else source.resolve()))
        found[str(resolved)] = {"location": "prepared" if internal else "source_bundle", "path": relative,
                                "sha256": _sha(resolved), "size_bytes": resolved.stat().st_size}
        if resolved.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            return
        layer = Sdf.Layer.FindOrOpen(str(resolved))
        if layer is None:
            raise AssetPreparationError("asset_import_dependency_layer_unreadable")
        references = set(layer.GetExternalReferences()) | set(layer.subLayerPaths)

        def attributes(spec_path):
            spec = layer.GetObjectAtPath(spec_path)
            if isinstance(spec, Sdf.AttributeSpec) and str(spec.typeName) in {"asset", "asset[]"}:
                values = [spec.default, *(layer.QueryTimeSample(spec_path, time)
                                        for time in layer.ListTimeSamplesForPath(spec_path))]
                for value in values:
                    for asset in ([value] if isinstance(value, Sdf.AssetPath) else value or []):
                        if isinstance(asset, Sdf.AssetPath) and asset.path:
                            references.add(asset.path)

        layer.Traverse("/", attributes)
        for value in references:
            if "://" in value or "[" in value or "\x00" in value:
                raise AssetPreparationError("asset_import_unbound_dependency")
            dependency = Path(value)
            inspect(dependency if dependency.is_absolute() else resolved.parent / dependency)

    inspect(entry)
    return sorted(found.values(), key=lambda row: (row["location"], row["path"]))


def _existing(destination: Path, source: dict[str, Any], fix_base: bool, *, source_root: Path) -> dict[str, Any]:
    if destination.is_symlink() or (destination / RECEIPT).is_symlink():
        raise AssetPreparationError("asset_import_destination_symlink_refused")
    try:
        receipt = json.loads((destination / RECEIPT).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssetPreparationError("asset_import_destination_incomplete") from exc
    if (receipt.get("schema_version") != SCHEMA or receipt.get("source_bundle_digest") != source["bundle_digest"]
            or receipt.get("fix_base") is not fix_base
            or receipt.get("import_digest") != cross_runtime_canonical_digest(receipt, digest_field="import_digest")):
        raise AssetPreparationError("asset_import_receipt_changed")
    recorded = set()
    for row in receipt["files"]:
        path = destination / row["path"]
        if (not path.resolve().is_relative_to(destination.resolve()) or path.is_symlink() or not path.is_file()
                or path.stat().st_size != row["size_bytes"] or _sha(path) != row["sha256"]):
            raise AssetPreparationError("asset_import_output_changed")
        recorded.add(row["path"])
    actual = {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()}
    if actual != recorded | {RECEIPT}:
        raise AssetPreparationError("asset_import_output_changed")
    dependencies = _dependencies(destination / receipt["model"]["path"], staged=destination,
                                 source=source_root, source_receipt=source)
    if dependencies != receipt["dependencies"]:
        raise AssetPreparationError("asset_import_dependency_changed")
    return receipt


def import_source_asset(source_bundle_root: str | Path, output_dir: str | Path, *, fix_base: bool,
                        embodiment_id: str | None = None) -> dict[str, Any]:
    """Import one exact source, preserving its tools/arm count and unresolved facts.

    The caller explicitly chooses fixed/free base. Named source arm count never
    becomes an automatically assembled bimanual configuration. Imported collider
    approximations, unknown source inertias and missing tools stay unqualified.
    """
    if type(fix_base) is not bool:
        raise AssetPreparationError("asset_import_base_mode_required")
    bundle = Path(source_bundle_root)
    source_receipt = verify_source_bundle(bundle, embodiment_id=embodiment_id)
    bundle = bundle.resolve()
    recipe = recipe_for(source_receipt["embodiment_id"])
    description = recipe["source"]
    destination = Path(output_dir)
    if any(path.is_symlink() for path in [destination, *destination.parents]):
        raise AssetPreparationError("asset_import_destination_symlink_refused")
    if destination.exists():
        return _existing(destination, source_receipt, fix_base, source_root=bundle / "source")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise AssetPreparationError("asset_import_destination_parent_invalid")
    source = bundle / "source"
    entry = source / description["entry_path"]
    packages = _package_roots(source, description["package_roots"])
    with tempfile.TemporaryDirectory(prefix=".blueprint-asset-import-", dir=destination.parent) as temporary:
        staged = Path(temporary)
        work = staged / "expanded_source"
        format_name = description["format"]
        if format_name == "xacro":
            entry = _expand_xacro(entry, source=source, work=work,
                                  arguments=description["import_arguments"], roots=packages)
            packages = {name: work / path.relative_to(source.resolve()) for name, path in packages.items()}
            format_name = "urdf"
        imported, options, runtime = _converter(format_name, entry, staged / "converted",
                                                 fix_base=fix_base, packages=packages)
        if not imported.resolve().is_relative_to(staged.resolve()) or not imported.is_file():
            raise AssetPreparationError("asset_import_converter_output_missing")
        from pxr import Usd, UsdUtils

        _dependencies(imported, staged=staged, source=source, source_receipt=source_receipt)
        stage = Usd.Stage.Open(str(imported))
        if stage is None:
            raise AssetPreparationError("asset_import_usd_unreadable")
        layer = stage.Flatten()
        # Keep assets internal to the prepared folder relative when publishing
        # the directory. Source-bundle references remain bound to retained files.
        def relative_asset(path):
            candidate = Path(path)
            if candidate.is_absolute() and candidate.is_relative_to(staged):
                return str(candidate.relative_to(staged))
            return path

        UsdUtils.ModifyAssetPaths(layer, relative_asset)
        output = staged / "robot.usda"
        layer.Export(str(output))
        dependencies = _dependencies(output, staged=staged, source=source, source_receipt=source_receipt)
        structure = read_model_structure(output)
        if not structure["joint_names"] or not structure["body_paths"]:
            raise AssetPreparationError("asset_import_robot_structure_missing")
        if any(path.is_symlink() for path in staged.rglob("*")):
            raise AssetPreparationError("asset_import_output_symlink_refused")
        files = [{"path": str(path.relative_to(staged)), "size_bytes": path.stat().st_size, "sha256": _sha(path)}
                 for path in sorted(staged.rglob("*")) if path.is_file()]
        # Absolute staging paths are invocation details, not repeatable settings.
        settings = {key: value for key, value in options.items() if key not in {"asset_path", "usd_dir", "ros_package_paths"}}
        receipt = {"schema_version": SCHEMA, "claim_ceiling": "source_asset_not_qualified",
                   "embodiment_id": source_receipt["embodiment_id"], "source_bundle_digest": source_receipt["bundle_digest"],
                   "source_arm_count": source_receipt["source_arm_count"], "fix_base": fix_base,
                   "importer": runtime, "settings": settings, "files": files,
                   "model": {"path": "robot.usda", "sha256": _sha(output), "size_bytes": output.stat().st_size},
                   "structure": structure, "dependencies": dependencies,
                   "source_bundle_required_for_dependency_resolution": any(
                       row["location"] == "source_bundle" for row in dependencies),
                   "composition_required": list(source_receipt["composition_required"]),
                   "missing": list(source_receipt["missing"]),
                   "limitations": ["source_only_not_complete_mounted_robot", "collision_meshes_use_recorded_importer_approximation",
                                   "missing_source_inertia_is_not_measured_physics", "no_native_episode_or_method_qualification"],
                   "execution_authorized": False}
        receipt["import_digest"] = cross_runtime_canonical_digest(receipt, digest_field="import_digest")
        (staged / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        # Recheck retained source after the importer read it and publish under
        # an exclusively reserved name; an existing empty directory is owned too.
        verify_source_bundle(bundle, embodiment_id=source_receipt["embodiment_id"])
        try:
            publish_staged_immutable_directory(staging=staged, destination=destination, manifest_name=RECEIPT,
                                               output_exists_code="asset_import_destination_appeared")
        except ImmutableDirectoryPublicationError as exc:
            raise AssetPreparationError(str(exc)) from exc
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--embodiment")
    parser.add_argument("--initialize-isaac", action="store_true",
                        help="Boot local headless Isaac on an already admitted/installed worker; no allocation or qualification")
    base = parser.add_mutually_exclusive_group(required=True)
    base.add_argument("--fixed-base", action="store_true")
    base.add_argument("--free-base", action="store_true")
    args = parser.parse_args(argv)
    app = None
    try:
        if args.initialize_isaac:
            try:
                from isaaclab.app import AppLauncher
            except ImportError as exc:
                raise AssetPreparationError("asset_import_initialized_isaac_runtime_unavailable") from exc
            app = AppLauncher(headless=True).app
        receipt = import_source_asset(args.source_bundle, args.output, fix_base=args.fixed_base,
                                      embodiment_id=args.embodiment)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Robot source import refused: {exc}\n")
    finally:
        if app is not None:
            app.close()
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["import_source_asset", "main"]
