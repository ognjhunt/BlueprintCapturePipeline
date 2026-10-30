"""Exact pinned iiwa14 MJCF subset to URDF, retaining publisher geometry/dynamics.

This is not a general MJCF converter. Source encodings/bytes stay retained;
only the generated simulation entry normalizes quaternion encodings and
expresses full inertia tensors in each body frame. Force caps are added later
from an explicit composite simulation model, never attributed to the publisher.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from defusedxml import ElementTree as SafeXML

from . import deployment_embodiment_mjcf_configuration as metadata
from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .deployment_embodiment_assets import AssetPreparationError
from .team_arm_kinematics import pose_matrix

IDENTITY = "kuka_lbr_iiwa_14_robotiq_2f85_sim"
SOURCE_ENTRY = "kuka_iiwa_14/iiwa14.xml"
SOURCE_SHA = "sha256:ee7e6d796a9f8642de6097f95077281d15f43678dc5285beafeda99b84cf9ef5"
ARM_ENTRY, REPORT = "iiwa14.generated.urdf", "iiwa14.bridge.json"


def require(value, reason):
    if not value:
        raise AssetPreparationError("kuka_source_bridge_" + reason)


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(q):
    return pose_matrix([0, 0, 0], [*q[1:], q[0]])[:3, :3]


def rpy(q):
    matrix = rotation(q)
    return [math.atan2(matrix[2, 1], matrix[2, 2]),
            math.atan2(-matrix[2, 0], math.hypot(matrix[0, 0], matrix[1, 0])),
            math.atan2(matrix[1, 0], matrix[0, 0])]


def numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


def origin(parent, position, quaternion=(1, 0, 0, 0)):
    ET.SubElement(parent, "origin", xyz=numbers(position), rpy=numbers(rpy(quaternion)))


def generate(arm_root: Path, source_receipt: dict) -> tuple[bytes, dict]:
    entry = arm_root / SOURCE_ENTRY
    require(sha(entry) == SOURCE_SHA, "unreviewed_source_entry")
    source = SafeXML.fromstring(entry.read_bytes())
    profile = {"base_link": "base", "tool_link": "link7", "dof": 7, "fingers": {}}
    fact = metadata.configuration_from_mjcf(entry, "kuka_lbr_iiwa_14", profile, source_receipt)
    kin = fact["source_kinematics"]
    require(len(kin["bodies"]) == 8 and len(kin["joints"]) == 7, "structure_changed")
    root = ET.Element("robot", name="blueprint_iiwa14_source_bridge")
    # Original assets remain at their exact relative paths in this two-source bundle.
    meshes = {Path(node.get("file")).stem: node.get("file") for node in source.findall("asset/mesh")}
    materials = {node.get("name"): node.get("rgba") for node in source.findall("asset/material")}
    for name, rgba in materials.items():
        ET.SubElement(ET.SubElement(root, "material", name=name), "color", rgba=rgba)
    bodies = {node.get("name"): node for node in source.findall(".//worldbody//body")}
    joints = {row["body"]: row for row in kin["joints"]}
    collisions, visuals = [], []
    for body in kin["bodies"]:
        link = ET.SubElement(root, "link", name=body["name"])
        authored = body["inertial"]
        require(authored is not None, "source_inertia_missing")
        inertial = ET.SubElement(link, "inertial")
        origin(inertial, authored["position_m"])
        ET.SubElement(inertial, "mass", value=str(authored["mass_kg"]))
        axes = rotation(authored["orientation_wxyz"])
        tensor = axes @ np.diag(authored["principal_inertia_kg_m2"]) @ axes.T
        ET.SubElement(inertial, "inertia", **{
            "i" + a + b: format(float(tensor[i, j]), ".17g")
            for i, a in enumerate("xyz") for j, b in enumerate("xyz") if i <= j})
        if body["parent"] is not None:
            row = joints[body["name"]]
            require(row["position_m"] == [0, 0, 0], "nonzero_joint_anchor_unsupported")
            joint = ET.SubElement(root, "joint", name=row["name"], type="revolute")
            ET.SubElement(joint, "parent", link=body["parent"])
            ET.SubElement(joint, "child", link=body["name"])
            origin(joint, body["position_m"], body["orientation_wxyz"])
            ET.SubElement(joint, "axis", xyz=numbers(row["axis"]))
            # Zero effort records missing source force limits. The composite
            # model supplies explicit nonzero simulation caps before import.
            ET.SubElement(joint, "limit", lower=str(row["range_rad"][0]), upper=str(row["range_rad"][1]),
                          velocity="1", effort="0")
            servo = kin["actuators"][row["name"]]
            ET.SubElement(joint, "drive", stiffness=str(servo["stiffness"]), damping=str(servo["damping"]))
        for index, geom in enumerate(bodies[body["name"]].findall("geom")):
            kind = geom.get("class")
            require(kind in {"visual", "collision"}, "geometry_class_unsupported")
            name = f'{body["name"]}_{kind}_{index}'
            out = ET.SubElement(link, kind, name=name)
            position = [float(value) for value in geom.get("pos", "0 0 0").split()]
            origin(out, position)
            geometry = ET.SubElement(out, "geometry")
            if kind == "visual":
                filename = "kuka_iiwa_14/assets/" + meshes[geom.get("mesh")]
                ET.SubElement(geometry, "mesh", filename="arm/source/" + filename, scale="1 1 1")
                material = geom.get("material", "gray")
                ET.SubElement(out, "material", name=material)
                visuals.append({"link": body["name"], "name": name, "mesh": filename,
                                "source_sha256": sha(arm_root / filename), "position_m": position,
                                "rgba": [float(value) for value in materials[material].split()]})
            else:
                radius = float(geom.get("size"))
                ET.SubElement(geometry, "sphere", radius=str(radius))
                collisions.append({"link": body["name"], "name": name, "position_m": position, "radius_m": radius})
    attachment = next(site for site in kin["sites"] if site["name"] == "attachment_site")
    require(attachment["body"] == "link7" and attachment["position_m"] == [0, 0, .045], "attachment_changed")
    ET.SubElement(root, "link", name="attachment_site")
    joint = ET.SubElement(root, "joint", name="blueprint_attachment_site", type="fixed")
    ET.SubElement(joint, "parent", link="link7")
    ET.SubElement(joint, "child", link="attachment_site")
    origin(joint, attachment["position_m"], attachment["orientation_wxyz"])
    require(len(visuals) == 15 and len(collisions) == 46 and len(meshes) == 13, "geometry_count_changed")
    report = {"schema_version": "iiwa14_source_bridge.v1", "source_entry_sha256": SOURCE_SHA,
              "source_revision": source_receipt["revision"], "source_kinematics": kin,
              "source_arm_mass_kg": sum(body["inertial"]["mass_kg"] for body in kin["bodies"]),
              "visuals": visuals, "collision_spheres": collisions,
              "source_contact_exclusions": [dict(row.attrib) for row in source.findall("contact/exclude")],
              "source_force_limits_present": False, "source_velocity_limits_present": False,
              "simulation_velocity_cap_rad_s": 1.0,
              "appearance_limitations": ["MJCF_specular_0.5_and_shininess_0.25_have_no_exact_standard_URDF_field"],
              "generator_sha256": sha(Path(__file__))}
    report["bridge_digest"] = digest(report, digest_field="bridge_digest")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), report
