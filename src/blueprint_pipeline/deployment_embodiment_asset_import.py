"""Convert a retained publisher source with the installed official Isaac importer.

This prepares source geometry, not a complete mounted robot or an admitted run.
The library expects the existing initialized Isaac worker; the CLI boots local
Isaac only with --initialize-isaac. Neither path installs a runtime, allocates
resources or downloads model bytes.
"""

from __future__ import annotations

import argparse
import builtins
import fnmatch
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import struct
import tempfile
import threading
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from defusedxml import ElementTree as ET

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .deployment_embodiment_assets import AssetPreparationError, recipe_for, verify_source_bundle, source_description
from .team_embodiment import read_model_structure
from .immutable_directory_publication import (
    ImmutableDirectoryPublicationError,
    publish_staged_immutable_directory,
)

RECEIPT = "asset-import.json"
SCHEMA = "embodiment_asset_import.v1"
MAX_PREPARED_FILES = 10_000
MAX_PREPARED_BYTES = 2 * 1024**3
MAX_PREPARED_FILE_BYTES = 512 * 1024**2
MAX_IMPORT_RECEIPT_BYTES = 8 * 1024**2
_FIND = re.compile(r"\$\(find\s+([A-Za-z0-9_]+)\)")
_XACRO_LOCK = threading.RLock()


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


def _input_path(value: str, *, anchor: Path, allowed: set[Path], packages: dict[str, Path],
                directory: bool = False) -> Path:
    """Resolve one actual importer read against the retained file inventory."""
    if not isinstance(value, str) or not value or any(char in value for char in ("\x00", "\\", "$")):
        raise AssetPreparationError("asset_import_input_reference_unresolved")
    if value.startswith("package://"):
        uri = urlsplit(value)
        if uri.netloc not in packages or uri.query or uri.fragment:
            raise AssetPreparationError("asset_import_input_package_unbound")
        path = packages[uri.netloc] / unquote(uri.path).lstrip("/")
    else:
        if ":" in value or value.startswith("//"):
            raise AssetPreparationError("asset_import_input_external_reference_refused")
        path = Path(value)
        if not path.is_absolute():
            path = anchor / path
    resolved = path.resolve()
    if (path.is_symlink() or (not directory and resolved not in allowed)
            or (directory and not any(item.is_relative_to(resolved) for item in allowed))):
        raise AssetPreparationError("asset_import_input_reference_unbound")
    return resolved


def _precheck_input_closure(entry: Path, *, allowed: set[Path], packages: dict[str, Path],
                            relative_references: bool = False) -> set[Path]:
    """Bind XML includes, geometry and textures before a third-party importer reads them."""
    pending, seen = [entry], set()
    mesh_dirs: dict[str, Path] = {}

    def reference(value: str, *, anchor: Path, directory: bool = False) -> Path:
        if relative_references and Path(value).is_absolute():
            raise AssetPreparationError("asset_import_dependency_not_relocatable")
        return _input_path(value, anchor=anchor, allowed=allowed, packages=packages, directory=directory)

    while pending:
        path = _input_path(str(pending.pop()), anchor=entry.parent, allowed=allowed, packages=packages)
        if path in seen:
            continue
        seen.add(path)
        suffix = path.suffix.lower()
        if suffix in {".stl", ".ply", ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff", ".exr", ".hdr", ".webp", ".ktx", ".ktx2", ".dds", ".gif", ".bin"}:
            continue  # These formats have no separate referenced file closure.
        if suffix in {".gltf", ".glb"}:
            try:
                if suffix == ".glb":
                    with path.open("rb") as stream:
                        header = stream.read(20)
                        if len(header) != 20:
                            raise ValueError("GLB header missing")
                        magic, version, length, json_length, kind = struct.unpack("<4sIII4s", header)
                        if (magic != b"glTF" or version != 2 or length != path.stat().st_size
                                or kind != b"JSON" or json_length > length - 20):
                            raise ValueError("GLB JSON chunk invalid")
                        model = json.loads(stream.read(json_length))
                else:
                    model = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise AssetPreparationError("asset_import_input_gltf_invalid") from exc
            values = [model]
            while values:
                value = values.pop()
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key == "uri":
                            if not isinstance(item, str):
                                raise AssetPreparationError("asset_import_input_gltf_invalid")
                            if not item.startswith("data:"):
                                pending.append(reference(unquote(item), anchor=path.parent))
                        else:
                            values.append(item)
                elif isinstance(value, list):
                    values.extend(value)
            continue
        if suffix not in {".xml", ".urdf", ".dae", ".obj", ".mtl"}:
            raise AssetPreparationError("asset_import_input_format_dependency_unsupported")
        text = path.read_text(encoding="utf-8")
        if suffix in {".obj", ".mtl"}:
            for line in text.splitlines():
                parts = line.strip().split(maxsplit=1)
                kind, value = parts[0] if parts else "", parts[1] if len(parts) > 1 else ""
                if kind == "mtllib" or kind.lower().startswith("map_") or kind in {"bump", "disp", "decal", "refl"}:
                    if not value or value.startswith("-"):
                        raise AssetPreparationError("asset_import_input_texture_options_unsupported")
                    pending.append(reference(value.strip().strip('"'), anchor=path.parent))
            continue
        if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
            raise AssetPreparationError("asset_import_input_xml_entity_refused")
        try:
            document = ET.fromstring(text)
        except ET.ParseError as exc:
            raise AssetPreparationError("asset_import_input_xml_invalid") from exc
        # A URDF can carry a <mujoco><compiler .../></mujoco> extension for
        # MuJoCo conversion. Its URDF mesh filenames still resolve relative to
        # the URDF, not that extension's meshdir (e.g. G1's meshes/pelvis.STL).
        is_urdf = document.tag.rsplit("}", 1)[-1] == "robot"
        # This path targets the Isaac URDF importer. Gazebo extensions and
        # their plugin filenames are not geometry or libraries it loads.
        ignored = {node for extension in document.findall("gazebo") for node in extension.iter()} if is_urdf else set()
        # MuJoCo mesh/texture roots are explicit; never search alternative paths.
        for node in document.iter():
            if node in ignored:
                continue
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "compiler" and not is_urdf:
                for key in ("assetdir", "meshdir", "texturedir"):
                    if key in node.attrib:
                        directory = reference(node.attrib[key], anchor=entry.parent, directory=True)
                        if key in mesh_dirs and mesh_dirs[key] != directory:
                            raise AssetPreparationError("asset_import_input_compiler_roots_conflict")
                        mesh_dirs[key] = directory
        for node in document.iter():
            if node in ignored:
                continue
            tag = node.tag.rsplit("}", 1)[-1]
            for key in ("file", "filename"):
                if key not in node.attrib:
                    continue
                anchor = path.parent
                if tag in {"mesh", "hfield", "skin"} and not is_urdf:
                    anchor = mesh_dirs.get("meshdir", mesh_dirs.get("assetdir", anchor))
                elif tag == "texture" and not is_urdf:
                    anchor = mesh_dirs.get("texturedir", mesh_dirs.get("assetdir", anchor))
                pending.append(reference(node.attrib[key], anchor=anchor))
            if tag == "image":  # COLLADA images can reference a separate texture.
                for child in node:
                    if child.tag.rsplit("}", 1)[-1] == "init_from":
                        pending.append(reference("".join(child.itertext()).strip(), anchor=path.parent))
    return seen


def _expand_xacro(entry: Path, *, source: Path, work: Path, arguments: dict[str, str],
                  roots: dict[str, Path], source_files: set[Path]) -> Path:
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
        if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
            raise AssetPreparationError("asset_import_input_xml_entity_refused")
        # Old publisher examples can name inactive packages in comments.
        # Remove comments only from this disposable expansion copy; retained
        # publisher bytes and license notices remain untouched in the bundle.
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

        def resolve(match):
            if match[1] not in aliases:
                raise AssetPreparationError("asset_import_xacro_package_alias_missing:" + match[1])
            return str(aliases[match[1]])

        path.write_text(_FIND.sub(resolve, text), encoding="utf-8")
    expanded = work / entry.relative_to(source)
    allowed = {(work / path.relative_to(source.resolve())).resolve() for path in source_files}
    # Xacro resolves argument-dependent include/YAML paths itself. Check the
    # resolved path at its read boundary rather than guessing macro variables.
    # The module-global hook is scoped and serialized, never builtins.open.
    with _XACRO_LOCK:
        missing = object()
        old_open = xacro.__dict__.get("open", missing)
        old_extension = getattr(xacro, "eval_extension", None)
        old_include = getattr(xacro, "get_include_files", None)
        if (not callable(old_extension) or not callable(old_include)
                or any(getattr(getattr(xacro, name, None), "__globals__", None) is not xacro.__dict__
                       for name in ("parse", "load_yaml"))):
            raise AssetPreparationError("asset_import_xacro_read_boundary_unsupported")

        def guarded_open(filename, mode="r", *args, **kwargs):
            if mode not in {"r", "rt", "rb"}:
                raise AssetPreparationError("asset_import_xacro_write_refused")
            path = _input_path(filename, anchor=work, allowed=allowed, packages=aliases)
            return builtins.open(path, mode, *args, **kwargs)

        def guarded_extension(value):
            match = _FIND.fullmatch(value)
            if match:
                if match[1] not in aliases:
                    raise AssetPreparationError("asset_import_xacro_package_alias_missing")
                return str(aliases[match[1]])
            if not (value.startswith("$(arg ") or value == "$(dirname)"):
                raise AssetPreparationError("asset_import_xacro_environment_or_eval_substitution_refused")
            return old_extension(value)

        def guarded_include(value, symbols):
            resolved = xacro.abs_filename_spec(xacro.eval_text(value, symbols))
            if any(char in resolved for char in "*?["):
                matches = [path for path in sorted(allowed) if fnmatch.fnmatch(str(path), resolved)]
                if not matches:
                    raise AssetPreparationError("asset_import_input_reference_unbound")
                return iter(map(str, matches))
            return iter([str(_input_path(resolved, anchor=work, allowed=allowed, packages=aliases))])

        xacro.open, xacro.eval_extension, xacro.get_include_files = guarded_open, guarded_extension, guarded_include
        try:
            document = xacro.process_file(str(expanded), mappings=dict(arguments))
        except AssetPreparationError:
            raise
        except Exception as exc:
            raise AssetPreparationError("asset_import_xacro_expansion_failed:" + type(exc).__name__) from exc
        finally:
            if old_open is missing:
                del xacro.open
            else:
                xacro.open = old_open
            xacro.eval_extension, xacro.get_include_files = old_extension, old_include
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


def _dependencies(entry: Path, *, staged: Path, source: Path | None = None,
                  source_receipt: dict[str, Any] | None = None,
                  relative_references: bool = False) -> list[dict[str, Any]]:
    """Inspect local layers before stage composition; never resolve an unbound URL/file."""
    from pxr import Sdf

    allowed_source = ({(source / row["path"]).resolve() for row in source_receipt["files"]}
                      if source is not None and source_receipt is not None else set())
    allowed = allowed_source | {path.resolve() for path in staged.rglob("*") if path.is_file()}
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
            # Runtime content can itself name buffers, materials or images.
            # Preserve that closure too, rather than treating a DAE/OBJ as an
            # opaque leaf just because USD references it as an asset value.
            for dependency in _precheck_input_closure(resolved, allowed=allowed, packages={},
                                                       relative_references=relative_references):
                if dependency != resolved:
                    inspect(dependency)
            return
        # Read saved bytes afresh; an earlier Sdf cache entry must not hide a
        # changed reference after digest verification.
        layer = Sdf.Layer.OpenAsAnonymous(str(resolved))
        if layer is None:
            raise AssetPreparationError("asset_import_dependency_layer_unreadable")
        references = (set(layer.GetExternalReferences()) | set(layer.subLayerPaths)
                      | set(layer.GetExternalAssetDependencies()))

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
            if relative_references and dependency.is_absolute():
                raise AssetPreparationError("asset_import_dependency_not_relocatable")
            inspect(dependency if dependency.is_absolute() else resolved.parent / dependency)

    inspect(entry)
    return sorted(found.values(), key=lambda row: (row["location"], row["path"]))


def _retain_flattened_content(layer, *, imported: Path, staged: Path, source: Path,
                              source_receipt: dict[str, Any]) -> None:
    """Make the flattened model relocatable without changing robot geometry.

    Flatten composes layers, but asset-valued textures can still point into the
    original source or a disposable converter directory. Keep their publisher
    path hierarchy (including sidecars and notices) and author relative paths
    rooted in the published preparation directory. No network resolver is used.
    """
    from pxr import UsdUtils

    source_files = {(source / row["path"]).resolve() for row in source_receipt["files"]}
    retained = staged / "retained_source"

    def relative_asset(value):
        if not value:
            return value
        if "://" in value or "[" in value or "\x00" in value:
            raise AssetPreparationError("asset_import_unbound_dependency")
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else imported.parent / candidate
        resolved = candidate.resolve()
        if candidate.is_symlink() or not candidate.is_file():
            raise AssetPreparationError("asset_import_unbound_dependency")
        if resolved.is_relative_to(staged.resolve()):
            target = resolved
        elif resolved in source_files:
            if not retained.exists():
                # The complete bounded, hash-verified inventory retains local
                # sidecar paths and source notices together. The source bundle
                # still supplies provenance, but is no longer needed to load.
                shutil.copytree(source, retained)
                for row in source_receipt["files"]:
                    copied = retained / row["path"]
                    if copied.stat().st_size != row["size_bytes"] or _sha(copied) != row["sha256"]:
                        raise AssetPreparationError("asset_import_retained_source_changed")
            target = retained / resolved.relative_to(source.resolve())
        else:
            raise AssetPreparationError("asset_import_unbound_dependency")
        return Path(os.path.relpath(target, staged)).as_posix()

    UsdUtils.ModifyAssetPaths(layer, relative_asset)


def _closure(dependencies: list[dict[str, Any]]) -> dict[str, Any]:
    if any(row["location"] != "prepared" for row in dependencies):
        raise AssetPreparationError("asset_import_prepared_dependency_external")
    value = {"schema_version": "embodiment_asset_dependency_closure.v1", "model_path": "robot.usda",
             "files": [{key: row[key] for key in ("path", "sha256", "size_bytes")} for row in dependencies]}
    value["closure_digest"] = cross_runtime_canonical_digest(value, digest_field="closure_digest")
    return value


def verify_prepared_asset_directory(directory: str | Path) -> dict[str, Any]:
    """Verify a retained preparation and closed runtime content without resolving URLs.

    The directory must be selected by the operator's existing private staging
    path. This verifies actual files/paths/digests and preserves source-only
    claims; the receipt by itself grants no source rights or execution authority.
    """
    destination = Path(directory)
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise AssetPreparationError("asset_import_destination_symlink_refused")
    receipt_path = destination / RECEIPT
    try:
        if receipt_path.is_symlink() or receipt_path.stat().st_size > MAX_IMPORT_RECEIPT_BYTES:
            raise AssetPreparationError("asset_import_receipt_invalid")
        with receipt_path.open("rb") as stream:
            raw = stream.read(MAX_IMPORT_RECEIPT_BYTES + 1)
        if len(raw) > MAX_IMPORT_RECEIPT_BYTES:
            raise AssetPreparationError("asset_import_receipt_invalid")
        receipt = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise AssetPreparationError("asset_import_destination_incomplete") from exc
    if (not isinstance(receipt, dict) or receipt.get("schema_version") != SCHEMA
            or receipt.get("claim_ceiling") != "source_asset_not_qualified"
            or receipt.get("execution_authorized") is not False or type(receipt.get("fix_base")) is not bool
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt.get("source_bundle_digest")))
            or receipt.get("import_digest") != cross_runtime_canonical_digest(receipt, digest_field="import_digest")):
        raise AssetPreparationError("asset_import_receipt_changed")
    inventory = receipt.get("files")
    if not isinstance(inventory, list) or not 0 < len(inventory) <= MAX_PREPARED_FILES:
        raise AssetPreparationError("asset_import_file_inventory_invalid")
    recorded = set()
    total_bytes = 0
    for row in inventory:
        if (not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}
                or not isinstance(row["path"], str) or not row["path"] or "\\" in row["path"]
                or "\x00" in row["path"] or PurePosixPath(row["path"]).is_absolute()
                or ".." in PurePosixPath(row["path"]).parts
                or str(PurePosixPath(row["path"])) != row["path"]
                or row["path"] in recorded or row["path"] == RECEIPT
                or type(row["size_bytes"]) is not int or not 0 <= row["size_bytes"] <= MAX_PREPARED_FILE_BYTES
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(row["sha256"]))):
            raise AssetPreparationError("asset_import_file_inventory_invalid")
        total_bytes += row["size_bytes"]
        if total_bytes > MAX_PREPARED_BYTES:
            raise AssetPreparationError("asset_import_file_inventory_bound_exceeded")
        path = destination / row["path"]
        if (not path.resolve().is_relative_to(destination.resolve())
                or any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_file()
                or path.stat().st_size != row["size_bytes"] or _sha(path) != row["sha256"]):
            raise AssetPreparationError("asset_import_output_changed")
        recorded.add(row["path"])
    actual = set()
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise AssetPreparationError("asset_import_destination_symlink_refused")
        if path.is_file():
            actual.add(path.relative_to(destination).as_posix())
        if len(actual) > MAX_PREPARED_FILES + 1:
            raise AssetPreparationError("asset_import_file_inventory_bound_exceeded")
    if actual != recorded | {RECEIPT}:
        raise AssetPreparationError("asset_import_output_changed")
    model = receipt.get("model")
    expected_model = next((row for row in inventory if row["path"] == "robot.usda"), None)
    if model != expected_model or expected_model is None:
        raise AssetPreparationError("asset_import_model_binding_changed")
    dependencies = _dependencies(destination / model["path"], staged=destination,
                                 relative_references=True)
    if (dependencies != receipt.get("dependencies") or receipt.get("dependency_closure") != _closure(dependencies)
            or receipt.get("prepared_directory_self_contained") is not True
            or receipt.get("single_file_self_contained") is not (len(dependencies) == 1)
            or receipt.get("source_bundle_required_for_dependency_resolution") is not False):
        raise AssetPreparationError("asset_import_dependency_changed")
    from .deployment_embodiment_simulation_models import for_identity
    gofa = for_identity(receipt["embodiment_id"])
    if gofa is not None:
        report = receipt.get("simulation_model", {})
        gofa.verify_report(destination, report, identity=receipt["embodiment_id"])
        if gofa.usd_values(destination / model["path"], report) != receipt.get("simulation_model_readback"):
            raise AssetPreparationError("asset_import_simulation_model_readback_changed")
    return receipt


def _existing(destination: Path, source: dict[str, Any], fix_base: bool,
              station_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    receipt = verify_prepared_asset_directory(destination)
    if receipt["source_bundle_digest"] != source["bundle_digest"] or receipt["fix_base"] is not fix_base:
        raise AssetPreparationError("asset_import_receipt_changed")
    station = receipt.get("station_composition")
    observed_parameters = station.get("composition", {}).get("parameters") if isinstance(station, dict) else None
    if observed_parameters != station_parameters:
        raise AssetPreparationError("asset_import_station_parameters_changed")
    return receipt


def import_source_asset(source_bundle_root: str | Path, output_dir: str | Path, *, fix_base: bool,
                        embodiment_id: str | None = None,
                        station_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
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
    description = source_description(recipe, source_receipt)
    destination = Path(output_dir)
    if any(path.is_symlink() for path in [destination, *destination.parents]):
        raise AssetPreparationError("asset_import_destination_symlink_refused")
    if destination.exists():
        return _existing(destination, source_receipt, fix_base, station_parameters)
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise AssetPreparationError("asset_import_destination_parent_invalid")
    source = bundle / "source"
    entry = source / description["entry_path"]
    packages = _package_roots(source, description["package_roots"])
    source_files = {(source / row["path"]).resolve() for row in source_receipt["files"]}
    with tempfile.TemporaryDirectory(prefix=".blueprint-asset-import-", dir=destination.parent) as temporary:
        staged = Path(temporary)
        work = staged / "expanded_source"
        format_name = description["format"]
        station_composition = None
        if station_parameters is not None:
            from .deployment_embodiment_station import prepare_station_source

            station_root = staged / "station"
            station_composition = prepare_station_source(bundle, station_root, station_parameters)
            entry = station_root / station_composition["model_path"]
            source_files = {(station_root / row["path"]).resolve() for row in station_composition["files"]}
            packages, format_name = {}, "urdf"
        elif format_name == "xacro":
            entry = _expand_xacro(entry, source=source, work=work,
                                  arguments=description["import_arguments"], roots=packages, source_files=source_files)
            packages = {name: work / path.relative_to(source.resolve()) for name, path in packages.items()}
            format_name = "urdf"
            source_files = {(work / path.relative_to(source.resolve())).resolve() for path in source_files} | {entry.resolve()}
        _precheck_input_closure(entry, allowed=source_files, packages=packages)
        from .deployment_embodiment_simulation_models import for_identity
        gofa = for_identity(source_receipt["embodiment_id"])
        simulation_model = None
        if gofa is not None:
            entry, simulation_model = gofa.prepare_model(entry, identity=source_receipt["embodiment_id"],
                                                        staged=staged, packages=packages, allowed=source_files)
            _precheck_input_closure(entry, allowed={p.resolve() for p in staged.rglob("*") if p.is_file()}, packages={})
        imported, options, runtime = _converter(format_name, entry, staged / "converted",
                                                 fix_base=fix_base, packages=packages)
        if not imported.resolve().is_relative_to(staged.resolve()) or not imported.is_file():
            raise AssetPreparationError("asset_import_converter_output_missing")
        from pxr import Usd

        _dependencies(imported, staged=staged, source=source, source_receipt=source_receipt)
        stage = Usd.Stage.Open(str(imported))
        if stage is None:
            raise AssetPreparationError("asset_import_usd_unreadable")
        layer = stage.Flatten(addSourceFileComment=False)
        _retain_flattened_content(layer, imported=imported, staged=staged,
                                  source=source, source_receipt=source_receipt)
        output = staged / "robot.usda"
        layer.Export(str(output))
        simulation_readback = gofa.usd_values(output, simulation_model, apply=True) if simulation_model is not None else None
        dependencies = _dependencies(output, staged=staged, relative_references=True)
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
                   "composed_arm_count": 2 if station_composition is not None else source_receipt["source_arm_count"],
                   "station_composition": station_composition,
                   "importer": runtime, "settings": settings, "files": files,
                   "model": {"path": "robot.usda", "sha256": _sha(output), "size_bytes": output.stat().st_size},
                   "structure": structure, "dependencies": dependencies,
                   "dependency_closure": _closure(dependencies),
                   "prepared_directory_self_contained": True,
                   "single_file_self_contained": len(dependencies) == 1,
                   "source_bundle_required_for_dependency_resolution": False,
                   "composition_required": list(source_receipt["composition_required"]),
                   "missing": list(source_receipt["missing"]),
                   "limitations": ["source_only_not_complete_mounted_robot", "collision_meshes_use_recorded_importer_approximation",
                                   "missing_source_inertia_is_not_measured_physics", "no_native_episode_or_method_qualification"],
                   "execution_authorized": False}
        if simulation_model is not None:
            receipt["simulation_model"] = simulation_model
            receipt["simulation_model_readback"] = simulation_readback
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
    # Check the actual published location. A disposable absolute path must not
    # survive simply because it was readable before the directory rename.
    _existing(destination, source_receipt, fix_base, station_parameters)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--embodiment")
    parser.add_argument("--station-parameters", type=Path,
                        help="Explicit two-arm station parameters; no mount defaults or generated robot geometry")
    parser.add_argument("--initialize-isaac", action="store_true",
                        help="Boot local headless Isaac on an already admitted/installed worker; no allocation or qualification")
    base = parser.add_mutually_exclusive_group(required=True)
    base.add_argument("--fixed-base", action="store_true")
    base.add_argument("--free-base", action="store_true")
    args = parser.parse_args(argv)
    app = None
    try:
        verify_source_bundle(args.source_bundle, embodiment_id=args.embodiment)
        station_parameters = json.loads(args.station_parameters.read_text()) if args.station_parameters else None
        if args.initialize_isaac:
            try:
                from isaaclab.app import AppLauncher
            except ImportError as exc:
                raise AssetPreparationError("asset_import_initialized_isaac_runtime_unavailable") from exc
            app = AppLauncher(headless=True).app
        receipt = import_source_asset(args.source_bundle, args.output, fix_base=args.fixed_base,
                                      embodiment_id=args.embodiment, station_parameters=station_parameters)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Robot source import refused: {exc}\n")
    finally:
        if app is not None:
            app.close()
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["import_source_asset", "verify_prepared_asset_directory", "main"]
