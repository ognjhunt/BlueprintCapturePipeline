"""Source iiwa14/tool dynamics with explicit, finite simulation drive estimates."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from defusedxml import ElementTree as SafeXML

from . import deployment_embodiment_kuka_bridge as bridge
from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .deployment_embodiment_station import _origin_matrix

IDENTITY = bridge.IDENTITY
SOURCE_MODEL, MODELED_URDF = "simulation-source.urdf", "simulation-model.urdf"
require, sha = bridge.require, bridge.sha


def source_metadata(anchor):
    work = anchor / "expanded_source"
    receipt = json.loads((work / "arm/source-bundle.json").read_text())
    generated, expected = bridge.generate(work / "arm/source", receipt)
    require((work / bridge.ARM_ENTRY).read_bytes() == generated
            and json.loads((work / bridge.REPORT).read_text()) == expected, "retained_bridge_changed")
    return expected


def body_dynamics(root):
    rows = []
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").get("value"))
        values = inertial.find("inertia").attrib
        matrix = np.array([[float(values["i" + min(a, b) + max(a, b)]) for b in "xyz"] for a in "xyz"])
        transform = np.array(_origin_matrix(inertial))
        tensor = transform[:3, :3] @ matrix @ transform[:3, :3].T
        eigen = np.linalg.eigvalsh(tensor)
        require(np.isfinite(tensor).all() and np.isfinite(mass) and mass > 0 and min(eigen) > 0
                and max(eigen) <= sum(eigen)/2+1e-10, "authored_inertia_invalid")
        rows.append({"link": link.get("name"), "mass_kg": mass,
                     "center_of_mass_m": transform[:3, 3].tolist(), "inertia_body_frame_kg_m2": tensor.tolist()})
    require(len(rows) == 17, "arm_or_tool_inertial_missing")
    return rows


def estimate_drives(root, dynamics):
    """Orientation-independent lever bound; assumptions remain part of identity."""
    incoming = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
    def radius(ancestor, child, com):
        value = float(np.linalg.norm(com))
        while child != ancestor:
            if child not in incoming:
                return None
            joint = incoming[child]
            value += float(np.linalg.norm(np.array(_origin_matrix(joint))[:3, 3]))
            child = joint.find("parent").get("link")
        return value
    result = []
    for index in range(1, 8):
        name, anchor = "joint"+str(index), "link"+str(index)
        joint = root.find(f"joint[@name='{name}']")
        require(joint is not None and float(joint.find("limit").get("effort")) == 0,
                "source_force_placeholder_changed")
        gravity, inertia = 0., 0.
        for row in dynamics:
            lever = radius(anchor, row["link"], row["center_of_mass_m"])
            if lever is not None:
                gravity += 9.81 * row["mass_kg"] * lever
                inertia += row["mass_kg"] * lever**2 + float(max(np.linalg.eigvalsh(row["inertia_body_frame_kg_m2"])))
        payload_lever = radius(anchor, "grasp_link", [0, 0, 0])
        require(payload_lever is not None, "grasp_chain_missing")
        gravity += 9.81 * payload_lever  # Explicit hypothetical 1 kg payload.
        inertia += payload_lever**2
        effort = max(1., 2 * gravity + inertia)  # 2x gravity + 1 rad/s² allowance.
        require(np.isfinite(effort) and 0 < effort < 1000, "estimated_drive_out_of_bounds")
        joint.find("limit").set("effort", str(effort))
        joint.find("limit").set("velocity", "1.0")
        result.append({"joint": name, "effort_limit_nm": effort, "velocity_limit_rad_s": 1.,
                       "gravity_moment_bound_nm": gravity, "inertia_bound_kg_m2": inertia,
                       "basis": "simulation_estimate_not_manufacturer_limit"})
    for joint in root.findall("joint"):
        if joint.get("type") == "revolute" and joint.get("name", "").startswith("robotiq_85_"):
            limit = joint.find("limit")
            result.append({"joint": joint.get("name"), "effort_limit_nm": float(limit.get("effort")),
                           "velocity_limit_rad_s": float(limit.get("velocity")), "basis": "authored_tool_URDF_limit"})
    require(len(result) == 13 and all(0 < row["effort_limit_nm"] < 1000 and 0 < row["velocity_limit_rad_s"] <= 1
                                     for row in result), "drive_structure_invalid")
    return result


def model_urdf(root, *, identity, anchor):
    require(identity == IDENTITY, "model_identity_invalid")
    metadata = source_metadata(anchor)
    dynamics = body_dynamics(root)
    report = {"schema_version": "iiwa14_robotiq_simulation_model.v1", "embodiment_id": identity,
              "claim_ceiling": "source_dynamics_with_unvalidated_simulation_drive_estimates",
              "manufacturer_motor_limits": False, "estimated_inertias_added": False,
              "source_bridge": metadata, "links": dynamics, "joints": estimate_drives(root, dynamics),
              "combined_mass_total_kg": sum(row["mass_kg"] for row in dynamics),
              "drive_estimates": {"hypothetical_payload_kg": 1., "payload_frame": "grasp_link",
                                  "gravity_margin": 2., "acceleration_allowance_rad_s2": 1.,
                                  "method": "triangle_inequality_descendant_COM_radius_bound"},
              "runtime_self_collisions_enabled": True,
              "native_fixed_base_gravity_disabled": True,
              "limitations": ["ideal_fixed_tool_mount_without_adapter_geometry_or_hardware_fit",
                  "native_fixed_arm_gravity_off_is_not_the_source_MuJoCo_gravity_model",
                  "finite_drive_caps_are_simulation_estimates_not_manufacturer_motor_limits",
                  "native_joint_adjacent_collision_filtering_is_a_separate_engine_default",
                  "source_spheres_authored_exactly_tool_mesh_colliders_use_recorded_importer_approximation",
                  "task_and_scene_contact_readback_does_not_measure_arm_arm_contact_forces",
                  *metadata["appearance_limitations"]],
              "generator_sha256": sha(Path(__file__)),
              "physics_application_sha256": sha(Path(__file__).with_name("deployment_embodiment_kuka_usd.py"))}
    report["model_digest"] = digest(report, digest_field="model_digest")
    return report


def prepare_model(entry, *, identity, staged, packages, allowed):
    from .deployment_embodiment_station import _mesh_paths
    root = SafeXML.fromstring(entry.read_bytes())
    _mesh_paths(root, entry=entry, output_anchor=staged, packages=packages, allowed=allowed)
    (staged / SOURCE_MODEL).write_text(ET.tostring(root, encoding="unicode"))
    report = model_urdf(root, identity=identity, anchor=staged)
    target = staged / MODELED_URDF
    target.write_text(ET.tostring(root, encoding="unicode"))
    return target, report


def verify_report(root, report, *, identity):
    from .deployment_embodiment_asset_import import _precheck_input_closure
    require(isinstance(report, dict) and report.get("embodiment_id") == identity, "model_identity_changed")
    source = root / SOURCE_MODEL
    _precheck_input_closure(source, allowed={p.resolve() for p in root.rglob("*") if p.is_file()}, packages={},
                            relative_references=True)
    expected = model_urdf(SafeXML.fromstring(source.read_bytes()), identity=identity, anchor=root)
    require(expected == report, "model_report_changed")


def usd_values(path, report, *, apply=False):
    from .deployment_embodiment_kuka_usd import source_physics
    return source_physics(path, report, apply=apply)


def annotate_configuration(result, original, report):
    result["source_dynamics"] = original["source_dynamics"]
    result["source_limitations"] = original["source_limitations"]
    result["simulation_model"] = report
    result["simulation_assumptions"] = [value for value in result["simulation_assumptions"]
                                        if value != "interior_zero_reset_pose"] + report["limitations"]
    reset = report["source_bridge"]["source_kinematics"]["reset"]
    result["manifest_fields"]["reset_joint_positions"].update(dict(zip(reset["joint_order"], reset["qpos_rad"], strict=True)))
    result["simulation_assumptions"].append("source_home_keyframe_arm_reset_with_source_open_tool")


def verify_configuration(root, source, record):
    from . import deployment_embodiment_runtime_configuration as runtime
    verify_report(root, record.get("simulation_model", {}), identity=source["embodiment_id"])
    unmodeled = SafeXML.fromstring((root / SOURCE_MODEL).read_bytes())
    profile = runtime._PROFILES[IDENTITY]
    original = runtime._configuration(unmodeled, IDENTITY, profile, source, None)
    report = model_urdf(unmodeled, identity=IDENTITY, anchor=root)
    model = root / "robot.urdf"
    require(ET.tostring(unmodeled, encoding="unicode") == model.read_text(), "generated_urdf_changed")
    expected = runtime._configuration(unmodeled, IDENTITY, profile, source, None)
    expected["source_urdf_sha256"] = runtime._sha(model)
    annotate_configuration(expected, original, report)
    retained = {key: value for key, value in record.items()
                if key not in {"generator_module_sha256", "configuration_digest", "generated_files"}}
    require(expected == retained, "configuration_metadata_changed")
