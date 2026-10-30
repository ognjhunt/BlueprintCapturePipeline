"""GoFa arm estimates plus separately retained, source-authored Robotiq dynamics.

The unchanged arm model is anchored to the bare robot's 28/48 kg. Tool mass is
added, never included in that normalization. A fixed ideal mount has no adapter
mass or geometry. This is simulation provenance, not manufacturer validation.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as XML

import numpy as np

from . import deployment_embodiment_gofa_model as arm_model
from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .deployment_embodiment_station import _origin_matrix

VARIANTS = {name + "_robotiq_2f85_sim": name for name in arm_model.PAYLOAD_KG}
PAYLOAD_KG = {name: arm_model.PAYLOAD_KG[base] for name, base in VARIANTS.items()}
SOURCE_MODEL, MODELED_URDF = arm_model.SOURCE_MODEL, arm_model.MODELED_URDF
_require, _sha = arm_model._require, arm_model._sha
TOOL_PIN = "a74d007d8f2f06dc6a503ad21038ba869d4999a6"


def _tool_dynamics(root):
    links = []
    for link in root.findall("link"):
        if not link.get("name", "").startswith("robotiq_85_"):
            continue
        inertial = link.find("inertial")
        _require(inertial is not None, "tool_inertial_missing")
        mass = float(inertial.find("mass").get("value"))
        values = inertial.find("inertia").attrib
        matrix = np.array([[float(values["i" + min(a, b) + max(a, b)]) for b in "xyz"] for a in "xyz"])
        transform = np.array(_origin_matrix(inertial))
        tensor = transform[:3, :3] @ matrix @ transform[:3, :3].T
        eigenvalues = np.linalg.eigvalsh(tensor)
        _require(np.isfinite(tensor).all() and np.isfinite(mass) and mass > 0 and min(eigenvalues) > 0
                 and max(eigenvalues) <= sum(eigenvalues) / 2 + 1e-10, "tool_inertial_invalid")
        links.append({"link": link.get("name"), "mass_kg": mass,
                      "center_of_mass_m": transform[:3, 3].tolist(), "inertia_body_frame_kg_m2": tensor.tolist()})
    joints = []
    for joint in root.findall("joint"):
        if joint.get("name", "").startswith("robotiq_85_") and joint.get("type") == "revolute":
            limit = joint.find("limit")
            joints.append({"joint": joint.get("name"), "effort_limit_nm": float(limit.get("effort")),
                           "velocity_limit_rad_s": float(limit.get("velocity"))})
    _require(len(links) == 9 and len(joints) == 6
             and all(0 < row["effort_limit_nm"] <= 100 and 0 < row["velocity_limit_rad_s"] <= 1 for row in joints),
             "tool_structure_changed")
    return {"source_revision": TOOL_PIN, "mass_total_kg": sum(row["mass_kg"] for row in links),
            "links": links, "joints": joints, "basis": "authored_urdf_inertias_and_limits_not_measured_hardware"}


def model_urdf(root, *, identity, anchor):
    _require(identity in VARIANTS, "unsupported_composite_identity")
    tool = _tool_dynamics(root)
    arm = arm_model.model_urdf(root, identity=VARIANTS[identity], anchor=anchor)
    report = {"schema_version": "gofa_robotiq_simulation_model.v1", "embodiment_id": identity,
              "claim_ceiling": "unvalidated_simulation_estimate", "manufacturer_dynamics": False,
              "arm_model": arm, "tool_source_dynamics": tool,
              "combined_mass_total_kg": arm["modeled_robot_mass_total_kg"] + tool["mass_total_kg"],
              "mount": {"parent_link": "tool0", "translation_m": [0, 0, 0], "rpy_rad": [0, 0, 0],
                        "adapter_geometry_present": False, "mount_kind": "ideal_fixed_simulation_mount"},
              "limitations": [v for v in arm["limitations"] if v != "no_application_tool_modeled"] + [
                  "arm_estimates_retain_nominal_payload_allowance_not_tool_specific_identified_actuator_dynamics",
                  "tool_source_inertias_added_separately_to_bare_arm_mass",
                  "ideal_fixed_mount_without_adapter_geometry_or_hardware_fit_validation"],
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
    """Read back arm estimates and all nine unmerged source tool rigid bodies.

    Full reconstructed body-frame inertia is compared, not an arbitrary
    eigenvector sign/order. Applying only reproduces recorded source values.
    """
    from pxr import Gf, Sdf, Usd, UsdPhysics
    from .team_arm_kinematics import pose_matrix, quaternion_xyzw
    arm = arm_model.usd_values(path, report["arm_model"], apply=apply)
    stage = Usd.Stage.Open(str(path) if apply else Sdf.Layer.OpenAsAnonymous(str(path)))
    prims = list(stage.Traverse())

    def named(name, schema):
        matches = [p for p in prims if p.GetName() == name and (
            p.HasAPI(schema) if schema == UsdPhysics.RigidBodyAPI else p.IsA(schema))]
        _require(len(matches) == 1, "usd_tool_binding_missing:" + name)
        return matches[0]

    tool = {"links": [], "joints": []}
    for row in report["tool_source_dynamics"]["links"]:
        prim = named(row["link"], UsdPhysics.RigidBodyAPI)
        mass = UsdPhysics.MassAPI.Apply(prim) if apply else UsdPhysics.MassAPI(prim)
        if apply:
            values, axes = np.linalg.eigh(np.array(row["inertia_body_frame_kg_m2"]))
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            q = quaternion_xyzw(axes)
            mass.CreateMassAttr(row["mass_kg"])
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*row["center_of_mass_m"]))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(q[3], *q[:3]))
        diagonal, quaternion = mass.GetDiagonalInertiaAttr().Get(), mass.GetPrincipalAxesAttr().Get()
        _require(diagonal is not None and quaternion is not None, "usd_tool_inertial_missing")
        rotation = pose_matrix([0, 0, 0], [*quaternion.GetImaginary(), quaternion.GetReal()])[:3, :3]
        actual = {"link": row["link"], "mass_kg": mass.GetMassAttr().Get(),
                  "center_of_mass_m": list(mass.GetCenterOfMassAttr().Get()),
                  "inertia_body_frame_kg_m2": (rotation @ np.diag(diagonal) @ rotation.T).tolist()}
        for key in ("mass_kg", "center_of_mass_m", "inertia_body_frame_kg_m2"):
            _require(np.allclose(actual[key], row[key], rtol=2e-6, atol=1e-10), "usd_tool_mass_inertia_changed")
        tool["links"].append(actual)
    for row in report["tool_source_dynamics"]["joints"]:
        prim = named(row["joint"], UsdPhysics.RevoluteJoint)
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular") if apply else UsdPhysics.DriveAPI(prim, "angular")
        if apply:
            drive.CreateTypeAttr("force")
            drive.CreateMaxForceAttr(row["effort_limit_nm"])
        value = drive.GetMaxForceAttr().Get()
        _require(drive.GetTypeAttr().Get() == "force" and value is not None
                 and np.isclose(value, row["effort_limit_nm"], rtol=1e-6), "usd_tool_effort_changed")
        tool["joints"].append({"joint": row["joint"], "effort_limit_nm": value, "drive_type": "force"})
    if apply:
        stage.GetRootLayer().Save()
        return usd_values(path, report)
    return {"arm_model": arm, "tool_source_dynamics": tool}


def annotate_configuration(result, original, report):
    result["source_dynamics"] = {**original["source_dynamics"], "estimated_inertias_added": True}
    result["source_limitations"] = original["source_limitations"]
    result["simulation_model"] = report
    result["simulation_assumptions"] += report["limitations"] + ["gofa_geometry_based_dynamics_and_drive_estimates"]


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
