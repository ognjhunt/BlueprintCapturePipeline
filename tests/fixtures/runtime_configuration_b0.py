"""Turn retained robot sources into explicit generic-runtime configurations.

Source joint/link names and limits are read from the pinned robot. Controller
gains, reset poses, camera rigs and duplicate-arm mounts are simulation choices,
recorded separately from source facts. A final manifest requires an actual
verified Isaac import; a source configuration never invents a USD digest.
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
from typing import Any
from urllib.parse import urlsplit
import xml.etree.ElementTree as XML

from defusedxml import ElementTree as SafeXML

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .deployment_embodiment_assets import AssetPreparationError, asset_recipes, recipe_for, verify_source_bundle, source_description
from .deployment_embodiment_asset_import import (
    _expand_xacro, _package_roots, _precheck_input_closure,
    import_source_asset, verify_prepared_asset_directory,
)
from .deployment_embodiment_station import _inverse, _mesh_paths, _multiply, _origin_matrix, _tree, gen_urdf
from .immutable_directory_publication import publish_staged_immutable_directory
from .team_embodiment import model_structure_blockers, read_model_structure, validate_embodiment_manifest

SCHEMA = "embodiment_runtime_configuration.v1"
RECEIPT = "runtime-configuration.json"


def _arm(base, tool, dof, *, tcp=None, fingers=None, contacts=None, width=None):
    return {"base_link": base, "tool_link": tool, "tcp_link": tcp or tool, "dof": dof,
            "fingers": fingers or {}, "contact_links": contacts or [], "max_width_m": width}


_PROFILES = {
    "trlc_dk1_bimanual": [_arm("base_link", "link6-7", 6, tcp="tool0",
        fingers={"gripper_left": [0.0, -.045], "gripper_right": [0.0, -.045]},
        contacts=["finger_left", "finger_right"], width=.09)],
    "agilex_piper_bimanual": [_arm("base_link", "gripper_base", 6,
        fingers={"gripper_joint1": [.05, 0.0], "gripper_joint2": [-.05, 0.0]},
        contacts=["gripper_link1", "gripper_link2"], width=.1)],
    "franka_research_3_bimanual": [_arm("fr3_link0", "fr3_hand", 7, tcp="fr3_hand_tcp",
        fingers={"fr3_finger_joint1": [.04, 0.0]}, contacts=["fr3_leftfinger", "fr3_rightfinger"], width=.08)],
    "universal_robots_ur5e_bimanual": [_arm("base_link_inertia", "robotiq_85_base_link", 6, tcp="grasp_link",
        fingers={"robotiq_85_left_knuckle_joint": [0.0, .7929]},
        contacts=["robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link"], width=.085)],
    "kinova_gen3": [_arm("base_link", "robotiq_arg2f_base_link", 7, tcp="tool_frame",
        fingers={"finger_joint": [0.0, .8]}, contacts=["left_inner_finger_pad", "right_inner_finger_pad"], width=.085)],
    "ufactory_xarm7": [_arm("link_base", "xarm_gripper_base_link", 7, tcp="link_tcp",
        fingers={"drive_joint": [0.0, .85]}, contacts=["left_finger", "right_finger"], width=.085)],
    "i2rt_yam_bimanual": [_arm(f"{side}_base", f"{side}_gripper", 6,
        fingers={f"{side}_joint7": [.04695, 0.0], f"{side}_joint8": [.04695, 0.0]},
        contacts=[f"{side}_tip_left", f"{side}_tip_right"], width=.0939) for side in ("left", "right")],
    "almond_axol": [_arm("s1", f"{side}_gripper", 7, tcp=f"{side}_fingers",
        fingers={f"{side}_finger_1_joint": [-.044, 0.0]},
        contacts=[f"{side}_finger_1", f"{side}_finger_2"], width=.088) for side in ("left", "right")],
}
_PROFILES["franka_research_3_robotiq_2f85_sim"] = [_arm("fr3_link0", "robotiq_85_base_link", 7, tcp="grasp_link",
    fingers={"robotiq_85_left_knuckle_joint": [0.0, .7929]},
    contacts=["robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link"], width=.085)]
_PROFILES["kinova_gen3_bimanual"] = deepcopy(_PROFILES["kinova_gen3"])
_PROFILES["ufactory_xarm7_bimanual"] = [
    {**deepcopy(_PROFILES["ufactory_xarm7"][0]), "source_prefix": prefix} for prefix in ("L_", "R_")]
for _identity in ("universal_robots_ur5e", "universal_robots_ur10e"):
    _PROFILES[_identity] = [_arm("base_link_inertia", "wrist_3_link", 6)]
for _identity in ("fanuc_crx_10ia", "fanuc_crx_10ia_l", "fanuc_m710ic_45m"):
    _PROFILES[_identity] = [_arm("base_link", "J6_link", 6)]
for _identity in ("yaskawa_gp12", "yaskawa_gp20"):
    _PROFILES[_identity] = [_arm("base_link", "link_6_t", 6)]
_PROFILES["kuka_lbr_iiwa_14"] = [_arm("base", "link7", 7)]


def runtime_profiles() -> dict[str, Any]:
    """Registered source bindings; this is deliberately not runnable-policy data."""
    return {"schema_version": "embodiment_runtime_profiles.v1", "claim_ceiling": "configuration_only",
            "profiles": [{"embodiment_id": name, "source_bindings": deepcopy(arms)}
                         for name, arms in sorted(_PROFILES.items())]}


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise AssetPreparationError("runtime_configuration_" + reason)


def _name(value: str) -> str:
    # Isaac/USD identifiers normalize punctuation; final USD readback checks
    # this mapping, rather than assuming that a successful import used it.
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _prefix_arm(arm, prefix):
    value = deepcopy(arm)
    for key in ("base_link", "tool_link", "tcp_link"):
        value[key] = prefix + value[key]
    value["fingers"] = {prefix + name: positions for name, positions in value["fingers"].items()}
    value["contact_links"] = [prefix + name for name in value["contact_links"]]
    return value


def _chain(incoming, base, target):
    result, current = [], target
    while current != base:
        _require(current in incoming, "arm_chain_unbound:" + target)
        joint = incoming[current]
        result.append(joint)
        current = joint.find("parent").get("link")
    return list(reversed(result))


def _fixed_transform(incoming, source_root, origin, target):
    def pose(link):
        result = [[float(i == j) for j in range(4)] for i in range(4)]
        for joint in _chain(incoming, source_root, link):
            # Only relative fixed branches below the common ancestor matter;
            # shared arm joints cancel from the two zero-pose transforms.
            result = _multiply(result, _origin_matrix(joint))
        return result
    left = _chain(incoming, source_root, origin)
    right = _chain(incoming, source_root, target)
    common = 0
    while common < min(len(left), len(right)) and left[common] is right[common]:
        common += 1
    _require(all(j.get("type") == "fixed" for j in left[common:] + right[common:]), "tcp_frame_not_rigid")
    return _multiply(_inverse(pose(origin)), pose(target))


def _quaternion(matrix):
    # The runtime uses XYZW. The stable branch form also handles 180-degree
    # authored tool frames where the trace-only formula loses precision.
    import numpy as np
    from .team_arm_kinematics import quaternion_xyzw
    return [float(v) for v in quaternion_xyzw(np.asarray(matrix, dtype=float)[:3, :3])]


def _configuration(root, embodiment_id, arms, source_receipt, station_parameters):
    source_root, incoming = _tree(root)
    joints = {j.get("name"): j for j in root.findall("joint")}
    links = {link.get("name") for link in root.findall("link")}
    moving_links = [joint.find("child").get("link") for joint in joints.values() if joint.get("type") != "fixed"]
    link_elements = {link.get("name"): link for link in root.findall("link")}
    missing_inertials = []
    for name in moving_links:
        inertial = link_elements[name].find("inertial")
        mass = inertial.find("mass") if inertial is not None else None
        inertia = inertial.find("inertia") if inertial is not None else None
        try:
            values = [float(mass.get("value")), *(float(inertia.get(key)) for key in ("ixx", "iyy", "izz"))]
            positive = all(math.isfinite(value) and value > 0 for value in values)
        except (AttributeError, TypeError, ValueError):
            positive = False
        if not positive:
            missing_inertials.append(name)
    dynamics = {"moving_link_count": len(moving_links),
                "moving_links_without_positive_authored_mass_and_inertia": sorted(missing_inertials),
                "estimated_inertias_added": False, "simulator_dynamics_validated": False}
    names = list(joints) + list(links)
    _require(len({_name(x) for x in joints}) == len(joints)
             and len({_name(x) for x in links}) == len(links), "usd_names_collide")
    defaults = {"stiffness": 100.0, "damping": 20.0}
    declared, channels, groups, reset, limitations = [], [], [], {}, []
    for index, arm in enumerate(arms):
        side = ("left", "right")[index] if len(arms) == 2 else "arm"
        chain = [j for j in _chain(incoming, arm["base_link"], arm["tool_link"]) if j.get("type") != "fixed"]
        _require(len(chain) == arm["dof"], "arm_dof_mismatch")
        ordered = []
        for joint in chain:
            name, limit = joint.get("name"), joint.find("limit")
            _require(limit is not None and joint.get("type") in {"revolute", "continuous", "prismatic"}, "joint_limits_missing")
            if joint.get("type") == "continuous":
                lower, upper = -math.pi, math.pi
                limitations.append("simulation_command_window_for_continuous_joint:" + name)
            else:
                lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            velocity = min(float(limit.get("velocity")), 1.0)
            _require(all(math.isfinite(x) for x in (lower, upper, velocity)) and lower < upper and velocity > 0,
                     "joint_limits_invalid")
            item = {"name": _name(name), "control": "position", "lower": lower, "upper": upper, "max_velocity": velocity}
            drive = joint.find("drive")
            if drive is not None and all(float(drive.get(k, "0")) > 0 for k in ("stiffness", "damping")):
                item.update({key: float(drive.get(key)) for key in ("stiffness", "damping")})
            declared.append(item)
            ordered.append(item["name"])
            # A reproducible interior start, explicitly simulation-authored.
            margin = (upper-lower) * .1
            reset[item["name"]] = max(lower+margin, min(0.0, upper-margin))
        finger_positions = deepcopy(arm["fingers"])
        # Resolve actual mimic chains, including negative multipliers. Do not
        # independently invent or clamp a slave's geometry.
        while True:
            additions = {}
            for name, joint in joints.items():
                mimic = joint.find("mimic")
                if name not in finger_positions and mimic is not None and mimic.get("joint") in finger_positions:
                    scale, offset = float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))
                    additions[name] = [scale*q + offset for q in finger_positions[mimic.get("joint")]]
            if not additions:
                break
            finger_positions.update(additions)
        fingers = []
        for name, positions in finger_positions.items():
            _require(name in joints, "finger_missing:" + name)
            joint = joints[name]
            limit = joint.find("limit")
            _require(limit is not None and all(float(limit.get("lower"))-1e-10 <= q <= float(limit.get("upper"))+1e-10
                                              for q in positions), "finger_limits_mismatch:" + name)
            fingers.append({"name": _name(name), "open": float(positions[0]), "closed": float(positions[1])})
            reset[_name(name)] = float(positions[0])
        _require(all(link in links for link in arm["contact_links"]), "tool_source_incomplete")
        channel_name = side + "_gripper"
        if fingers:
            channels.append({"name": channel_name, "tool_link": _name(arm["tool_link"]), "joints": fingers,
                             "max_width_m": arm["max_width_m"], "contact_links": [_name(link) for link in arm["contact_links"]]})
        else:
            limitations.append("bare_arm_without_application_tool_or_grasp_capability")
        transform = _fixed_transform(incoming, source_root, arm["tool_link"], arm["tcp_link"])
        groups.append({"name": side, "joint_names": ordered, "base_link": _name(arm["base_link"]),
                       "tool_link": _name(arm["tool_link"]), "gripper_name": channel_name,
                       "tool_from_tcp": {"position_m": [transform[i][3] for i in range(3)],
                                         "orientation_xyzw": _quaternion(transform)}})
    declared_names = {item["name"] for item in declared} | {j["name"] for c in channels for j in c["joints"]}
    _require(declared_names == {_name(name) for name, j in joints.items() if j.get("type") != "fixed"},
             "uncontrolled_source_joint")
    cameras = [{"name": "overhead", "mount_link": None, "position_m": [.5, 0.0, 1.5],
                "orientation_xyzw": [1.0, 0.0, 0.0, 0.0], "width": 320, "height": 176, "horizontal_fov_deg": 90.0}]
    cameras += [{"name": group["name"] + "_wrist", "mount_link": group["tool_link"],
                 "position_m": [.0, .08, .0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                 "width": 320, "height": 176, "horizontal_fov_deg": 90.0} for group in groups]
    # Sum of authored link translations is a conservative kinematic upper
    # bound, not a collision-free workspace or measured payload capability.
    reach = max(sum(math.sqrt(sum(_origin_matrix(j)[i][3]**2 for i in range(3)))
                    for j in _chain(incoming, source_root, arm["tcp_link"])) for arm in arms)
    fields = {"schema_version": "blueprint.robot_embodiment.v2", "embodiment_id": embodiment_id,
              "form": "bimanual_arm" if len(arms) == 2 else "fixed_base_arm", "base": {"mode": "fixed"},
              "joints": declared, "grippers": channels, "tool_link": groups[0]["tool_link"],
              "reset_joint_positions": reset, "actuator": defaults, "cameras": cameras,
              "control_hz": 60, "action_chunk_rows": 48, "reach_m": min(3.0, max(.1, reach))}
    if channels:
        fields["arms"] = groups
    return {"schema_version": SCHEMA, "claim_ceiling": "configuration_only", "embodiment_id": embodiment_id,
            "source_bundle_digest": source_receipt["bundle_digest"], "station_parameters": station_parameters,
            "manifest_fields": fields, "source_name_mapping": {name: _name(name) for name in sorted(set(names))},
            "simulation_assumptions": ["interior_zero_reset_pose", "default_PD_gains_unless_source_drive_present",
                "arm_velocity_capped_at_1_SI_unit_per_second", "90_degree_simulated_camera_rig_not_site_calibration",
                "parallel_gripper_width_is_nominal", "reach_is_source_chain_upper_bound",
                *(["duplicate_arms_separated_by_0.7_m_simulation_mounts"] if station_parameters else []), *limitations],
            "source_dynamics": dynamics,
            "source_limitations": [*source_receipt["missing"],
                *(["missing_positive_authored_moving_link_inertias:" + ",".join(sorted(missing_inertials))]
                  if missing_inertials else [])], "execution_authorized": False,
            "policy_compatibility_established": False}


def prepare_runtime_configuration(source_bundle: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Retain complete source plus executable joint/tool/camera/reset bindings."""
    bundle, destination = Path(source_bundle).resolve(), Path(output_dir)
    source_receipt = verify_source_bundle(bundle)
    identity = source_receipt["embodiment_id"]
    _require(identity in _PROFILES, "profile_unavailable:" + identity)
    if destination.exists():
        retained = verify_runtime_configuration(destination)
        _require(retained["source_bundle_digest"] == source_receipt["bundle_digest"], "destination_source_mismatch")
        return retained
    _require(destination.parent.is_dir() and not any(p.is_symlink() for p in (destination, *destination.parents)),
             "destination_invalid")
    recipe = recipe_for(identity)
    description = source_description(recipe, source_receipt)
    with tempfile.TemporaryDirectory(prefix=".runtime-configuration-", dir=destination.parent) as temporary:
        staged = Path(temporary)
        shutil.copytree(bundle / "source", staged / "source")
        shutil.copyfile(bundle / "source-bundle.json", staged / "source-bundle.json")
        verify_source_bundle(staged)
        source = staged / "source"
        files = {(source / row["path"]).resolve() for row in source_receipt["files"]}
        packages = _package_roots(source, description["package_roots"])
        entry = source / description["entry_path"]
        if description["format"] == "xacro":
            work = staged / "expanded_source"
            entry = _expand_xacro(entry, source=source, work=work, arguments=description["import_arguments"],
                                  roots=packages, source_files=files)
            packages = {name: work / path.relative_to(source) for name, path in packages.items()}
            files = {(work / path.relative_to(source)).resolve() for path in files} | {entry.resolve()}
        if description["format"] == "mjcf":
            from . import deployment_embodiment_mjcf_configuration as mjcf
            _require(recipe["arm_count"] == source_receipt["source_arm_count"] == 1
                     and len(_PROFILES[identity]) == 1, "mjcf_single_arm_required")
            _precheck_input_closure(entry, allowed=files, packages=packages)
            result = mjcf.configuration_from_mjcf(entry, identity, _PROFILES[identity][0], source_receipt)
            result["source_model"] = {"format": "mjcf", "path": entry.relative_to(staged).as_posix(), "sha256": _sha(entry)}
            result["metadata_generator_module_sha256"] = _sha(Path(mjcf.__file__))
        else:
            _require(description["format"] in {"xacro", "urdf"}, "urdf_source_required")
            root = SafeXML.fromstring(entry.read_bytes())
            source_root, _ = _tree(root)
            arms = [_prefix_arm(arm, arm.get("source_prefix", "")) for arm in _PROFILES[identity]]
            parameters = None
            if recipe["arm_count"] == 2 and source_receipt["source_arm_count"] == 1:
                parameters = {"schema_version": "embodiment_station_parameters.v1", "embodiment_id": identity,
                              "composition_mode": "two_instances", "station_root_link": "blueprint_station",
                              "arms": [{"name": side, "prefix": side + "_", "source_base_link": source_root,
                                        "source_mount_joint": None,
                                        "station_from_base": {"translation_m": [0.0, y, 0.0], "quaternion_wxyz": [1, 0, 0, 0]}}
                                       for side, y in (("left", .35), ("right", -.35))]}
                text, _ = gen_urdf(entry, parameters, output_anchor=staged, packages=packages, allowed_files=files)
                root = SafeXML.fromstring(text)
                arms = [_prefix_arm(arms[0], side + "_") for side in ("left", "right")]
            else:
                _mesh_paths(root, entry=entry, output_anchor=staged, packages=packages, allowed=files)
            model = staged / "robot.urdf"
            model.write_text(XML.tostring(root, encoding="unicode"), encoding="utf-8")
            _precheck_input_closure(model, allowed={p.resolve() for p in staged.rglob("*") if p.is_file()}, packages={})
            result = _configuration(root, identity, arms, source_receipt, parameters)
            result["source_urdf_sha256"] = _sha(model)
        result["generator_module_sha256"] = _sha(Path(__file__))
        result["generated_files"] = [{"path": path.relative_to(staged).as_posix(), "sha256": _sha(path),
                                      "size_bytes": path.stat().st_size}
                                     for path in sorted(staged.rglob("*")) if path.is_file()
                                     and path != staged / "source-bundle.json" and not path.is_relative_to(source)]
        result["configuration_digest"] = cross_runtime_canonical_digest(result, digest_field="configuration_digest")
        (staged / RECEIPT).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        publish_staged_immutable_directory(staging=staged, destination=destination, manifest_name=RECEIPT,
                                           output_exists_code="runtime_configuration_destination_appeared")
    return result


def verify_runtime_configuration(directory: str | Path) -> dict[str, Any]:
    from .deployment_embodiment_runtime_compatibility import is_exact_predecessor, verify_predecessor_metadata
    root = Path(directory)
    _require(not any(p.is_symlink() for p in (root, *root.parents)), "symlink_refused")
    source = verify_source_bundle(root)
    _require((root / RECEIPT).stat().st_size <= 8 * 1024 * 1024, "receipt_bound_exceeded")
    record = json.loads((root / RECEIPT).read_text())
    if source["format"] == "mjcf":
        from . import deployment_embodiment_mjcf_configuration as mjcf
        model_identity_matches = (record.get("source_model") == {
            "format": "mjcf", "path": "source/" + source["entry_path"],
            "sha256": _sha(root / "source" / source["entry_path"])}
            and record.get("metadata_generator_module_sha256") == _sha(Path(mjcf.__file__))
            and "source_urdf_sha256" not in record)
    else:
        model_identity_matches = record.get("source_urdf_sha256") == _sha(root / "robot.urdf")
    _require(record.get("schema_version") == SCHEMA and record.get("claim_ceiling") == "configuration_only"
             and record.get("execution_authorized") is False
             and record.get("source_bundle_digest") == source["bundle_digest"]
             and record.get("embodiment_id") == source["embodiment_id"]
             and model_identity_matches
             and (record.get("generator_module_sha256") == _sha(Path(__file__)) or is_exact_predecessor(record))
             and record.get("configuration_digest") == cross_runtime_canonical_digest(record, digest_field="configuration_digest"),
             "receipt_changed")
    seen = set()
    for row in record["generated_files"]:
        relative = Path(row["path"])
        _require(not relative.is_absolute() and ".." not in relative.parts and str(relative) not in seen,
                 "generated_path_invalid")
        path = root / relative
        _require(not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file()
                 and path.stat().st_size == row["size_bytes"] and _sha(path) == row["sha256"], "generated_file_changed")
        seen.add(str(relative))
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
              and not p.is_relative_to(root / "source") and p not in {root / RECEIPT, root / "source-bundle.json"}}
    _require(seen == actual, "generated_inventory_changed")
    if is_exact_predecessor(record):
        verify_predecessor_metadata(root, source, record)
    return record


def prepare_available_runtime_configurations(source_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Process the whole source catalog with an individual outcome for each body."""
    destination = Path(output_root)
    _require(not any(p.is_symlink() for p in (destination, *destination.parents)), "symlink_refused")
    destination.mkdir(exist_ok=True)
    results = []
    for recipe in asset_recipes()["recipes"]:
        identity = recipe["embodiment_id"]
        row = {"embodiment_id": identity}
        if identity not in _PROFILES:
            row.update(status="existing_runtime" if recipe["status"] == "existing_runtime" else "configuration_unavailable",
                       reason=(recipe["reason"] if recipe["source"] is None
                       else "No registered complete arm/tool binding for this exact source configuration."))
        else:
            try:
                config = prepare_runtime_configuration(Path(source_root) / identity, destination / identity)
                row.update(status="source_configured", configuration_digest=config["configuration_digest"],
                           joint_count=len(config["manifest_fields"]["joints"]),
                           tool_count=len(config["manifest_fields"]["grippers"]),
                           source_dynamics=config["source_dynamics"])
            except (ValueError, OSError, KeyError) as exc:
                row.update(status="configuration_failed", reason=str(exc)[:500])
        results.append(row)
    return {"schema_version": "embodiment_runtime_configuration_batch.v1", "claim_ceiling": "configuration_only",
            "execution_authorized": False, "results": results}


def bind_prepared_configuration(configuration_dir: str | Path, prepared_dir: str | Path,
                                model_url: str, *, runtime_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Produce the real manifest after verifying the imported USD and closure."""
    config = verify_runtime_configuration(configuration_dir)
    prepared = verify_prepared_asset_directory(prepared_dir)
    _require(prepared["source_bundle_digest"] == config["source_bundle_digest"] and prepared["fix_base"] is True,
             "import_source_mismatch")
    actual_parameters = (prepared.get("station_composition") or {}).get("composition", {}).get("parameters")
    _require(actual_parameters == config["station_parameters"], "station_parameters_mismatch")
    model = prepared["model"]
    settings = runtime_settings or {}
    _require(isinstance(settings, dict) and not set(settings) - {"control_hz", "action_chunk_rows"},
             "runtime_settings_invalid")
    manifest = validate_embodiment_manifest({**config["manifest_fields"], **settings,
        "model": {"format": "usd", "url": model_url, "sha256": model["sha256"]}})
    structure = read_model_structure(Path(prepared_dir) / model["path"])
    blockers = model_structure_blockers(manifest, structure)
    _require(not blockers, "import_structure_mismatch:" + ";".join(blockers))
    return {"schema_version": "embodiment_runtime_binding_inputs.v1", "claim_ceiling": "configured_not_executed",
            "robot_embodiment": manifest, "model_record": {"sha256": model["sha256"], "size_bytes": model["size_bytes"],
                "dependency_closure": prepared["dependency_closure"], "source_import_provenance": prepared},
            "structure": structure, "configuration_digest": config["configuration_digest"],
            "simulation_assumptions": config["simulation_assumptions"],
            "source_dynamics": config["source_dynamics"], "source_limitations": config["source_limitations"],
            "runtime_settings": {key: manifest[key] for key in ("control_hz", "action_chunk_rows")},
            "execution_authorized": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all-available", action="store_true", help="Treat --source-bundle and --output as catalog roots")
    parser.add_argument("--import-prepared", type=Path, help="Import using the already initialized Isaac runtime")
    parser.add_argument("--model-url", help="Actual operator publication URL for the prepared USD")
    parser.add_argument("--initialize-isaac", action="store_true", help="Initialize Isaac locally; allocates no provider")
    parser.add_argument("--runtime-settings", type=Path,
                        help="JSON with control_hz/action_chunk_rows matching the policy being bound")
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(runtime_profiles(), indent=2))
        return 0
    if not args.source_bundle or not args.output:
        parser.error("--source-bundle and --output are required")
    if args.initialize_isaac and not args.import_prepared:
        parser.error("--initialize-isaac requires --import-prepared")
    if args.runtime_settings and not args.import_prepared:
        parser.error("--runtime-settings requires --import-prepared")
    if args.import_prepared:
        url = urlsplit(args.model_url or "")
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            parser.error("--import-prepared requires a credential-free HTTPS --model-url")
    if args.all_available:
        if args.import_prepared or args.initialize_isaac:
            parser.error("batch source configuration and a single Isaac import are separate operations")
        result = prepare_available_runtime_configurations(args.source_bundle, args.output)
        print(json.dumps(result, indent=2))
        return int(any(row["status"] == "configuration_failed" for row in result["results"]))
    config = prepare_runtime_configuration(args.source_bundle, args.output)
    if args.import_prepared:
        app = None
        try:
            if args.initialize_isaac:
                from isaaclab.app import AppLauncher
                app = AppLauncher(headless=True).app
            import_source_asset(args.output, args.import_prepared, fix_base=True,
                                station_parameters=config["station_parameters"])
            settings = json.loads(args.runtime_settings.read_text()) if args.runtime_settings else None
            config = bind_prepared_configuration(args.output, args.import_prepared, args.model_url,
                                                 runtime_settings=settings)
        finally:
            if app is not None:
                app.close()
    print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
