"""Compose exact publisher URDFs into an explicitly mounted simulation station.

This is a constrained kinematic assembly, not geometry authoring or physical
calibration. Units are metres, kilograms, radians and unit WXYZ quaternions.
The source supplies every link, axis, collision, inertia and tool. Parameters
supply only names and simulation-authored mounting transforms; no defaults are
provided. Generated URDF, generator, parameters and source inventory stay together.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping
import xml.etree.ElementTree as XML

from defusedxml import ElementTree as SafeXML

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .deployment_embodiment_assets import AssetPreparationError, recipe_for, verify_source_bundle
from .immutable_directory_publication import publish_staged_immutable_directory

SCHEMA = "embodiment_station_source.v1"
PARAMETERS_SCHEMA = "embodiment_station_parameters.v1"
RECEIPT = "station-source.json"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")


def _require(condition, reason):
    if not condition:
        raise AssetPreparationError("station_composition_" + reason)


def _sha(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _numbers(value, length):
    _require(isinstance(value, list) and len(value) == length
             and all(type(x) in (int, float) and math.isfinite(x) for x in value), "transform_invalid")
    return [float(x) for x in value]


def _pose(value):
    _require(isinstance(value, Mapping) and set(value) == {"translation_m", "quaternion_wxyz"}, "transform_required")
    xyz, q = _numbers(value["translation_m"], 3), _numbers(value["quaternion_wxyz"], 4)
    _require(abs(sum(x * x for x in q) - 1.0) < 1e-6, "quaternion_not_unit")
    norm = math.sqrt(sum(x * x for x in q))
    w, x, y, z = [x / norm for x in q]
    return [[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), xyz[0]],
            [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w), xyz[1]],
            [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y), xyz[2]], [0.0, 0.0, 0.0, 1.0]]


def _multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _inverse(a):
    result = [[a[j][i] for j in range(3)] + [-sum(a[k][i]*a[k][3] for k in range(3))] for i in range(3)]
    return [*result, [0.0, 0.0, 0.0, 1.0]]


def _origin_matrix(joint):
    origin = joint.find("origin")
    xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0., 0., 0.]
    rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()] if origin is not None else [0., 0., 0.]
    _require(len(xyz) == len(rpy) == 3 and all(math.isfinite(x) for x in xyz+rpy), "source_origin_invalid")
    cr, cp, cy = [math.cos(x) for x in rpy]
    sr, sp, sy = [math.sin(x) for x in rpy]
    return [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, xyz[0]],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, xyz[1]],
            [-sp, cp*sr, cp*cr, xyz[2]], [0., 0., 0., 1.]]


def _set_origin(joint, matrix):
    pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    if abs(math.cos(pitch)) > 1e-9:
        roll, yaw = math.atan2(matrix[2][1], matrix[2][2]), math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll, yaw = math.atan2(-matrix[1][2], matrix[1][1]), 0.0
    origin = joint.find("origin")
    if origin is None:
        origin = XML.SubElement(joint, "origin")
    origin.set("xyz", " ".join(format(matrix[i][3], ".17g") for i in range(3)))
    origin.set("rpy", " ".join(format(x, ".17g") for x in (roll, pitch, yaw)))


def _tree(root):
    links, joints = root.findall("link"), root.findall("joint")
    names, joint_names = [x.get("name") for x in links], [x.get("name") for x in joints]
    _require(root.tag == "robot" and names and len(set(names)) == len(names)
             and len(set(joint_names)) == len(joint_names)
             and all(isinstance(x, str) and _NAME.fullmatch(x) for x in names+joint_names), "source_names_invalid")
    incoming = {}
    for joint in joints:
        parent, child = joint.find("parent"), joint.find("child")
        _require(parent is not None and child is not None and parent.get("link") in names
                 and child.get("link") in names and child.get("link") not in incoming
                 and joint.get("type") in {"fixed", "revolute", "continuous", "prismatic"}, "source_tree_invalid")
        incoming[child.get("link")] = joint
        mimic = joint.find("mimic")
        _require(mimic is None or mimic.get("joint") in joint_names and mimic.get("joint") != joint.get("name"),
                 "source_mimic_reference_invalid")
    for transmission in root.findall("transmission"):
        _require(all(j.get("name") in joint_names for j in transmission.findall("joint")),
                 "source_transmission_reference_invalid")
    roots = set(names) - set(incoming)
    _require(len(roots) == 1, "source_tree_invalid")
    root_link = next(iter(roots))
    for name in names:
        seen, current = set(), name
        while current != root_link:
            _require(current not in seen and current in incoming, "source_tree_invalid")
            seen.add(current)
            current = incoming[current].find("parent").get("link")
    return root_link, incoming


def _prefix(root, prefix):
    links = {node.get("name") for node in root.findall("link")}
    joints = {node.get("name") for node in root.findall("joint")}
    materials = {node.get("name") for node in root.iter("material") if node.get("name")}
    transmissions = {node.get("name") for node in root.iter("transmission") if node.get("name")}
    actuators = {node.get("name") for node in root.iter("actuator") if node.get("name")}
    for node in root.iter():
        name = node.get("name")
        if ((node.tag == "link" and name in links) or (node.tag == "joint" and name in joints)
                or (node.tag == "material" and name in materials)
                or (node.tag == "transmission" and name in transmissions)
                or node.tag in {"actuator", "ros2_control"} and name):
            node.set("name", prefix + name)
        for attribute, values in (("link", links), ("joint", joints), ("reference", links | joints)):
            if node.get(attribute) in values:
                node.set(attribute, prefix + node.get(attribute))
        # ROS-control/Gazebo metadata can use element text for an exact named
        # joint/link reference. Do not rewrite substrings, paths or class names.
        if node.text and node.text.strip() in links | joints | materials | transmissions | actuators:
            node.text = prefix + node.text.strip()


def _mesh_paths(root, *, entry, output_anchor, packages, allowed):
    for mesh in root.iter("mesh"):
        value = mesh.get("filename")
        if value is None:
            continue
        if value.startswith("package://"):
            package, _, relative = value.removeprefix("package://").partition("/")
            _require(package in packages, "mesh_package_unbound")
            path = packages[package] / relative
        else:
            _require(":" not in value and "\\" not in value and "\x00" not in value, "mesh_reference_invalid")
            path = Path(value) if Path(value).is_absolute() else entry.parent / value
        _require(not path.is_symlink() and path.resolve() in allowed and path.is_file(), "mesh_reference_unbound")
        _require(path.resolve().is_relative_to(output_anchor.resolve()), "mesh_not_retained")
        mesh.set("filename", path.resolve().relative_to(output_anchor.resolve()).as_posix())


def gen_urdf(source_path: Path, parameters: Mapping[str, Any], *, output_anchor: Path,
             packages: Mapping[str, Path], allowed_files: set[Path]) -> tuple[str, dict[str, Any]]:
    """Generate one explicit station from unchanged source geometry and mounts."""
    _require(isinstance(parameters, Mapping), "parameters_invalid")
    parameters = deepcopy(dict(parameters))
    mode = parameters.get("composition_mode")
    keys = {"schema_version", "embodiment_id", "composition_mode", "station_root_link", "arms"}
    if mode == "remount_existing_pair":
        keys.add("station_from_source_root")
    _require(set(parameters) == keys and parameters.get("schema_version") == PARAMETERS_SCHEMA
             and mode in {"two_instances", "remount_existing_pair"}, "parameters_invalid")
    station_name = parameters["station_root_link"]
    _require(isinstance(station_name, str) and _NAME.fullmatch(station_name), "station_name_invalid")
    arms = parameters["arms"]
    _require(isinstance(arms, list) and len(arms) == 2, "two_explicit_mounts_required")
    for arm in arms:
        _require(isinstance(arm, dict) and set(arm) == {
            "name", "prefix", "source_base_link", "source_mount_joint", "station_from_base"}, "arm_parameters_invalid")
        _require(all(isinstance(arm[key], str) and _NAME.fullmatch(arm[key]) for key in ("name", "source_base_link")),
                 "arm_names_invalid")
        _pose(arm["station_from_base"])
    _require(len({arm["name"] for arm in arms}) == 2, "arm_names_duplicated")
    source = SafeXML.fromstring(source_path.read_bytes())
    source_root, incoming = _tree(source)
    _mesh_paths(source, entry=source_path, output_anchor=output_anchor, packages=packages, allowed=allowed_files)
    station = XML.Element("robot", {"name": station_name})
    XML.SubElement(station, "link", {"name": station_name})  # Frame-only fixed station root.
    bindings = []
    if mode == "two_instances":
        prefixes = [arm["prefix"] for arm in arms]
        _require(len(set(prefixes)) == 2 and all(isinstance(x, str) and _NAME.fullmatch(x) for x in prefixes),
                 "independent_prefixes_required")
        for arm in arms:
            _require(arm["source_base_link"] == source_root and arm["source_mount_joint"] is None,
                     "single_arm_root_binding_invalid")
            copy = deepcopy(source)
            _prefix(copy, arm["prefix"])
            station.extend(copy)
            mounted = arm["prefix"] + source_root
            joint = XML.SubElement(station, "joint", {"name": arm["name"] + "_station_mount", "type": "fixed"})
            XML.SubElement(joint, "parent", {"link": station_name})
            XML.SubElement(joint, "child", {"link": mounted})
            _set_origin(joint, _pose(arm["station_from_base"]))
            bindings.append({"name": arm["name"], "base_link": mounted,
                             "joint_name_mapping": {j.get("name"): arm["prefix"] + j.get("name")
                                                    for j in source.findall("joint")}})
    else:
        _require(len({arm["source_base_link"] for arm in arms}) == 2
                 and all(arm["prefix"] is None for arm in arms), "paired_source_names_must_be_preserved")
        root_pose = _pose(parameters["station_from_source_root"])
        desired = {arm["source_base_link"]: _pose(arm["station_from_base"]) for arm in arms}
        if source_root in desired:
            _require(max(abs(root_pose[i][j]-desired[source_root][i][j]) for i in range(4) for j in range(4)) < 1e-9,
                     "source_root_mount_conflict")

        def parent_pose(name, seen=None):
            if name in desired:
                return desired[name]
            if name == source_root:
                return root_pose
            seen = set() if seen is None else seen
            _require(name not in seen and name in incoming and incoming[name].get("type") == "fixed",
                     "mount_ancestry_not_fixed")
            seen.add(name)
            joint = incoming[name]
            return _multiply(parent_pose(joint.find("parent").get("link"), seen), _origin_matrix(joint))

        for arm in arms:
            base = arm["source_base_link"]
            if base == source_root:
                _require(arm["source_mount_joint"] is None, "root_mount_joint_must_be_null")
            else:
                _require(base in incoming and incoming[base].get("type") == "fixed"
                         and incoming[base].get("name") == arm["source_mount_joint"], "actual_fixed_mount_required")
                joint = incoming[base]
                parent = joint.find("parent").get("link")
                _set_origin(joint, _multiply(_inverse(parent_pose(parent)), desired[base]))
            bindings.append({"name": arm["name"], "base_link": base, "joint_name_mapping": {}})
        station.extend(source)
        joint = XML.SubElement(station, "joint", {"name": "source_root_station_mount", "type": "fixed"})
        XML.SubElement(joint, "parent", {"link": station_name})
        XML.SubElement(joint, "child", {"link": source_root})
        _set_origin(joint, root_pose)
    _tree(station)  # Reject name collisions, duplicate children and disconnected graphs.
    text = XML.tostring(station, encoding="unicode", xml_declaration=True)
    return text, {"parameters": parameters, "input_urdf_sha256": _sha(source_path), "arm_bindings": bindings,
                  "mount_parameter_authority": "sim_authored", "manufacturer_geometry_changed": False}


def prepare_station_source(source_bundle_root: str | Path, output_dir: str | Path,
                           parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Retain an inspectable generated URDF and all exact source inputs, no Isaac."""
    from .deployment_embodiment_asset_import import _expand_xacro, _package_roots, _precheck_input_closure

    bundle, destination = Path(source_bundle_root), Path(output_dir)
    _require(isinstance(parameters, Mapping), "parameters_invalid")
    source_receipt = verify_source_bundle(bundle)
    recipe = recipe_for(source_receipt["embodiment_id"])
    description = recipe["source"]
    _require(parameters.get("embodiment_id") == source_receipt["embodiment_id"], "embodiment_identity_mismatch")
    expected_count = 1 if parameters.get("composition_mode") == "two_instances" else 2
    _require(source_receipt["source_arm_count"] == expected_count and recipe["arm_count"] == 2,
             "source_arm_count_mismatch")
    _require(description["format"] in {"urdf", "xacro"}, "urdf_source_required")
    _require(not destination.exists() and not any(x.is_symlink() for x in (destination, *destination.parents))
             and destination.parent.is_dir(), "new_destination_required")
    with tempfile.TemporaryDirectory(prefix=".blueprint-station-", dir=destination.parent) as raw:
        staged = Path(raw)
        source = staged / "source"
        shutil.copytree(bundle / "source", source)
        for row in source_receipt["files"]:
            copied = source / row["path"]
            _require(copied.stat().st_size == row["size_bytes"] and _sha(copied) == row["sha256"],
                     "retained_source_changed")
        files = {(source / row["path"]).resolve() for row in source_receipt["files"]}
        packages = _package_roots(source, description["package_roots"])
        entry = source / description["entry_path"]
        if description["format"] == "xacro":
            work = staged / "expanded_source"
            entry = _expand_xacro(entry, source=source, work=work, arguments=description["import_arguments"],
                                  roots=packages, source_files=files)
            packages = {name: work / path.relative_to(source.resolve()) for name, path in packages.items()}
            files = {(work / path.relative_to(source.resolve())).resolve() for path in files} | {entry.resolve()}
        text, composition = gen_urdf(entry, parameters, output_anchor=staged, packages=packages, allowed_files=files)
        (staged / "robot.urdf").write_text(text, encoding="utf-8")
        (staged / "parameters.json").write_text(json.dumps(dict(parameters), indent=2) + "\n", encoding="utf-8")
        generator = ('from pathlib import Path\nimport json\n'
                     'from blueprint_pipeline.deployment_embodiment_station import gen_urdf as compose\n'
                     'def gen_urdf():\n'
                     '    root = Path(__file__).resolve().parent\n'
                     f'    source = root / {entry.relative_to(staged).as_posix()!r}\n'
                     '    parameters = json.loads((root / "parameters.json").read_text())\n'
                     '    allowed = {path.resolve() for path in root.rglob("*") if path.is_file()}\n'
                     '    return compose(source, parameters, output_anchor=root, packages={}, allowed_files=allowed)[0]\n')
        # For raw URDF package URIs, preserve the exact alias roots in the
        # generator, rather than depending on a host ROS installation.
        aliases = {name: path.relative_to(staged).as_posix() for name, path in packages.items()}
        generator = generator.replace('packages={}', f'packages={{name: root / value for name, value in {aliases!r}.items()}}')
        (staged / "station.py").write_text(generator, encoding="utf-8")
        _precheck_input_closure(staged / "robot.urdf", allowed={path.resolve() for path in staged.rglob("*") if path.is_file()},
                                packages={})
        inventory = [{"path": path.relative_to(staged).as_posix(), "sha256": _sha(path), "size_bytes": path.stat().st_size}
                     for path in sorted(staged.rglob("*")) if path.is_file()]
        receipt = {"schema_version": SCHEMA, "claim_ceiling": "source_asset_not_qualified",
                   "embodiment_id": source_receipt["embodiment_id"], "source_bundle_digest": source_receipt["bundle_digest"],
                   "source_arm_count": source_receipt["source_arm_count"], "composed_arm_count": 2,
                   "model_path": "robot.urdf", "model_sha256": _sha(staged / "robot.urdf"),
                   "composition": composition, "generator_module_sha256": _sha(Path(__file__)),
                   "kernel": "python_standard_library_xml_and_rigid_transforms", "files": inventory,
                   "execution_authorized": False, "missing": source_receipt["missing"]}
        receipt["station_digest"] = cross_runtime_canonical_digest(receipt, digest_field="station_digest")
        (staged / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        verify_source_bundle(bundle)
        publish_staged_immutable_directory(staging=staged, destination=destination, manifest_name=RECEIPT,
                                           output_exists_code="station_composition_destination_appeared")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        parameters = json.loads(args.parameters.read_text(encoding="utf-8"))
        result = prepare_station_source(args.source_bundle, args.output, parameters)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Station source composition refused: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
