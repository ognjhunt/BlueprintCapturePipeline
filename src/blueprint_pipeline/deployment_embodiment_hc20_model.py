"""Declared Yaskawa HC20 arm estimates plus untouched source Robotiq inertias/limits.

The retained ROS-Industrial HC20 macro has no inertial elements. As for GP12,
a whole-arm nominal mass is a public anchor, never a measured link
distribution. The only primary mass document found is the HC20DTP flyer, a
variant with its own wrist links, so the exact source variant is explicitly
unestablished. Only separately generated simulation XML/USD receives these
estimates; retained source bytes keep their missing inertias.
"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as XML

from . import deployment_embodiment_gofa_model as geometry
from . import deployment_embodiment_gofa_tool_model as tool_model
from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .deployment_embodiment_assets import AssetPreparationError
from .deployment_embodiment_yaskawa_model import _length, _tool_lever

VARIANTS = {"yaskawa_hc20_robotiq_2f85_sim": "yaskawa_hc20"}
NOMINAL_ROBOT_KG = {"yaskawa_hc20": 140.0}
PAYLOAD_KG = {"yaskawa_hc20": 20.0}
MASS_REFERENCES = {
    "yaskawa_hc20": {
        "document_revision": "B-11-2023", "article_number": "220876", "page": 2,
        "document_model": "HC20DTP YR-1-06VXCP20-B00",
        "url": "https://yaskawa.de/_downloads/download_d17096",
        "download_sha256": "sha256:89bb45594f5a4a1f4c602a6efe5a86c331f7934deaaf3356b299eb1c3ae2658f",
        "nominal_mass_kg": 140.0, "exact_source_variant_established": False,
        "basis": "published_HC20DTP_mass_assumed_applicable_to_retained_HC20_DT_XP_geometry",
        "uncertainties": ["source_package_models_HC20_DT_XP_while_mass_document_is_HC20DTP_with_own_link_5_6_wrist",
            "primary_flyer_S_range_210_degrees_differs_from_retained_source_180_degrees",
            "deployed_HC20_variant_unconfirmed",
            "nominal_total_mass_does_not_identify_link_mass_distribution"],
    },
}
SOURCE_MODEL, MODELED_URDF = geometry.SOURCE_MODEL, geometry.MODELED_URDF
_sha = geometry._sha


def _require(value, reason):
    if not value:
        raise AssetPreparationError("hc20_simulation_" + reason)


def model_urdf(root, *, identity, anchor):
    _require(identity in VARIANTS, "unsupported_identity")
    base = VARIANTS[identity]
    tool = tool_model._tool_dynamics(root)
    links = {link.get("name"): link for link in root.findall("link")}
    names = ["link_1_s", "link_2_l", "link_3_u", "link_4_r", "link_5_b", "link_6_t"]
    joints = [root.find(f"joint[@name='joint_{n}_{axis}']") for n, axis in enumerate("slurbt", 1)]
    for index, joint in enumerate(joints):
        _require(joint is not None and joint.get("type") == "revolute"
                 and joint.find("parent").get("link") == (["base_link"] + names)[index]
                 and joint.find("child").get("link") == names[index], "arm_chain_invalid")
        _require(joint.find("drive") is None, "source_drive_changed")
    ordered = [links[name] for name in ["base_link", *names]]
    shapes = [geometry._bounds(link, anchor) for link in ordered]
    density = NOMINAL_ROBOT_KG[base] / sum(math.prod(size) for size, _, _ in shapes)
    _require(math.isfinite(density) and 1 < density < 100_000, "density_out_of_bounds")
    modeled = []
    for link, (size, center, meshes) in zip(ordered, shapes, strict=True):
        _require(link.find("inertial") is None, "source_inertias_changed")
        mass = density * math.prod(size)
        inertia = [mass * (size[(i+1)%3]**2 + size[(i+2)%3]**2) / 12 for i in range(3)]
        _require(.001 < mass < NOMINAL_ROBOT_KG[base] and all(math.isfinite(v) and 0 < v < 1000 for v in inertia),
                 "mass_or_inertia_out_of_bounds")
        inertial = XML.SubElement(link, "inertial")
        XML.SubElement(inertial, "origin", xyz=" ".join(map(str, center)), rpy="0 0 0")
        XML.SubElement(inertial, "mass", value=str(mass))
        XML.SubElement(inertial, "inertia", ixx=str(inertia[0]), iyy=str(inertia[1]), izz=str(inertia[2]),
                       ixy="0", ixz="0", iyz="0")
        modeled.append({"link": link.get("name"), "mass_kg": mass, "center_of_mass_m": center,
                        "diagonal_inertia_kg_m2": inertia, "bounds_m": size, "collision_meshes": meshes})
    moving, drives = modeled[1:], []
    lengths = [_length(j) for j in joints]
    tool_lever = _tool_lever(root, tool)
    for index, joint in enumerate(joints):
        limit = joint.find("limit")
        values = {key: float(limit.get(key)) for key in ("effort", "velocity", "lower", "upper")}
        _require(all(math.isfinite(v) for v in values.values()) and values["effort"] > 0
                 and values["velocity"] > 0 and values["lower"] < values["upper"], "source_limits_invalid")
        radii = [sum(lengths[index+1:k+1]) + math.sqrt(sum(v*v for v in moving[k]["center_of_mass_m"]))
                 + .5*math.sqrt(sum(v*v for v in moving[k]["bounds_m"])) for k in range(index, 6)]
        payload_radius = sum(lengths[index+1:]) + max(tool_lever, .3)
        # The 0.3m payload lever and nominal payload are conservative simulation
        # tuning assumptions, not an observed object or manufacturer motor model.
        load = PAYLOAD_KG[base] + tool["mass_total_kg"]
        effective_inertia = sum(row["mass_kg"]*r*r + max(row["diagonal_inertia_kg_m2"])
                                for row, r in zip(moving[index:], radii, strict=True)) + load*payload_radius**2
        stiffness = values["effort"] / .1
        damping = 2 * math.sqrt(stiffness * effective_inertia)
        _require(all(math.isfinite(v) and 0 < v < 1_000_000 for v in (stiffness, damping)), "gains_invalid")
        XML.SubElement(joint, "drive", stiffness=str(stiffness), damping=str(damping))
        # Authored limit element, including effort, remains byte-for-byte equivalent.
        drives.append({"joint": joint.get("name"), "effort_limit_nm": values["effort"],
                       "source_limits": dict(limit.attrib), "stiffness_nm_per_rad": stiffness,
                       "damping_nm_s_per_rad": damping, "effective_inertia_estimate_kg_m2": effective_inertia})
    arm = {"embodiment_id": base, "method": "uniform_collision_aabb_box_in_link_frame",
           "effective_density_kg_m3": density, "nominal_robot_mass_kg": NOMINAL_ROBOT_KG[base],
           "mass_reference": MASS_REFERENCES[base], "modeled_robot_mass_total_kg": sum(r["mass_kg"] for r in modeled),
           "mass_distribution_assumption": "uniform_density_over_base_and_six_link_collision_aabbs",
           "links": modeled, "joints": drives, "nominal_payload_kg": PAYLOAD_KG[base],
           "minimum_assumed_payload_lever_from_last_joint_m": .3, "source_tool_lever_bound_m": tool_lever,
           "drive_effort_basis": "source_URDF_values_retained_not_manufacturer_verified_torques",
           "gain_basis": "source_effort_saturation_at_0.1_rad_critical_damping_for_estimated_load_inertia"}
    report = {"schema_version": "hc20_robotiq_simulation_model.v1", "embodiment_id": identity,
              "claim_ceiling": "unvalidated_simulation_estimate", "manufacturer_dynamics": False,
              "arm_model": arm, "tool_source_dynamics": tool,
              "combined_mass_total_kg": arm["modeled_robot_mass_total_kg"] + tool["mass_total_kg"],
              "limitations": ["hollow_structure_and_actuator_mass_distribution_unknown",
                  "nominal_total_mass_does_not_establish_per_link_distribution",
                  "source_effort_limits_not_manufacturer_verified_drive_torques",
                  "PD_gains_and_payload_lever_are_simulation_assumptions",
                  "tool_source_inertias_added_separately_to_bare_arm_mass",
                  "robotiq_2f85_is_simulation_stand_in_not_deployed_tool",
                  "ideal_fixed_mount_without_adapter_geometry_or_hardware_fit_validation",
                  "not_physical_feasibility_or_policy_ranking_evidence", *MASS_REFERENCES[base]["uncertainties"]],
              "generator_sha256": _sha(Path(__file__))}
    report["model_digest"] = digest(report, digest_field="model_digest")
    return report


def prepare_model(entry, *, identity, staged, packages, allowed):
    from defusedxml import ElementTree
    from .deployment_embodiment_station import _mesh_paths
    root = ElementTree.fromstring(entry.read_bytes())
    _mesh_paths(root, entry=entry, output_anchor=staged, packages=packages, allowed=allowed)
    (staged / SOURCE_MODEL).write_text(XML.tostring(root, encoding="unicode"))
    report = model_urdf(root, identity=identity, anchor=staged)
    target = staged / MODELED_URDF
    target.write_text(XML.tostring(root, encoding="unicode"))
    return target, report


def verify_report(root, report, *, identity):
    from defusedxml import ElementTree
    from .deployment_embodiment_asset_import import _precheck_input_closure
    _require(isinstance(report, dict) and report.get("embodiment_id") == identity, "model_identity_changed")
    source = root / SOURCE_MODEL
    _precheck_input_closure(source, allowed={p.resolve() for p in root.rglob("*") if p.is_file()}, packages={},
                            relative_references=True)
    expected = model_urdf(ElementTree.fromstring(source.read_bytes()), identity=identity, anchor=root)
    _require(expected == report, "model_report_changed")


def usd_values(path, report, *, apply=False):
    # The shared readback reconstructs full tool tensors and checks arm COM,
    # diagonal tensors, identity principal axes, SI units and force-drive caps.
    return tool_model.usd_values(path, report, apply=apply)


def annotate_configuration(result, original, report):
    result["source_dynamics"] = {**original["source_dynamics"], "estimated_inertias_added": True}
    result["source_limitations"] = original["source_limitations"]
    result["simulation_model"] = report
    result["simulation_assumptions"] += report["limitations"] + ["hc20_geometry_based_dynamics_and_PD_estimates"]


def verify_configuration(root, source, record):
    from . import deployment_embodiment_runtime_configuration as runtime
    verify_report(root, record.get("simulation_model", {}), identity=source["embodiment_id"])
    unmodeled = runtime.SafeXML.fromstring((root / SOURCE_MODEL).read_bytes())
    profile = runtime._PROFILES[source["embodiment_id"]]
    original = runtime._configuration(unmodeled, source["embodiment_id"], profile, source, None)
    report = model_urdf(unmodeled, identity=source["embodiment_id"], anchor=root)
    model = root / "robot.urdf"
    _require(XML.tostring(unmodeled, encoding="unicode") == model.read_text(), "generated_urdf_changed")
    expected = runtime._configuration(unmodeled, source["embodiment_id"], profile, source, None)
    expected["source_urdf_sha256"] = runtime._sha(model)
    annotate_configuration(expected, original, report)
    retained = {key: value for key, value in record.items()
                if key not in {"generator_module_sha256", "configuration_digest", "generated_files"}}
    _require(expected == retained, "configuration_metadata_changed")
