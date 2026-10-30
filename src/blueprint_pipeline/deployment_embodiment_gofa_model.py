"""Explicit geometry-based GoFa simulation estimates; never ABB physical data.

The publisher supplies no inertias and zero effort placeholders. Preserve those
bytes. Only a generated simulation URDF/USD receives these declared estimates.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as XML

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .deployment_embodiment_assets import AssetPreparationError
from .deployment_embodiment_station import _origin_matrix

PAYLOAD_KG = {"abb_gofa_crb15000_5_95": 5.0, "abb_gofa_crb15000_12_127": 12.0}
NOMINAL_ROBOT_KG = {"abb_gofa_crb15000_5_95": 28.0, "abb_gofa_crb15000_12_127": 48.0}
MASS_REFERENCE = {
    "document_id": "3HAC077389-001", "revision": "Y", "section": "2.2 Technical data", "page": 42,
    "url": "https://search.abb.com/library/Download.aspx?DocumentID=3HAC077389-001&LanguageCode=en&DocumentPartId=&Action=Launch",
    "download_sha256": "sha256:b502ddd2e7cdf12066c708f5e590b6768651771cbb738fe35f27f1561e407fa5",
    "observed_date": "2026-10-30", "scope": "nominal_whole_robot_without_additional_options_tools_equipment",
}
SOURCE_MODEL = "simulation-source.urdf"
MODELED_URDF = "simulation-model.urdf"


def _require(value, reason):
    if not value:
        raise AssetPreparationError("gofa_simulation_" + reason)


def _sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _numbers(text):
    values = [float(v) for v in text.split()]
    _require(len(values) == 3 and all(math.isfinite(v) for v in values), "vector_invalid")
    return values


def _bounds(link, anchor):
    """Actual binary-STL vertices, with authored mesh scale and collision pose."""
    points, meshes = [], []
    for collision in link.findall("collision"):
        mesh = collision.find("geometry/mesh")
        _require(mesh is not None, "collision_mesh_required")
        path = anchor / mesh.get("filename")
        raw = path.read_bytes()
        _require(len(raw) >= 84, "stl_invalid")
        count = struct.unpack_from("<I", raw, 80)[0]
        _require(0 < count <= 1_000_000 and len(raw) == 84 + 50 * count, "binary_stl_invalid")
        scale = _numbers(mesh.get("scale", "1 1 1"))
        _require(all(0 < v <= 1000 for v in scale), "mesh_scale_invalid")
        transform = _origin_matrix(collision)
        for offset in range(84, len(raw), 50):
            for vertex in range(3):
                xyz = struct.unpack_from("<fff", raw, offset + 12 + 12 * vertex)
                local = [xyz[i] * scale[i] for i in range(3)]
                points.append([sum(transform[i][j] * local[j] for j in range(3)) + transform[i][3]
                               for i in range(3)])
        meshes.append({"sha256": _sha(path), "scale": scale, "link_from_collision": transform})
    _require(points and all(math.isfinite(v) for p in points for v in p), "mesh_vertices_invalid")
    low, high = ([min(p[i] for p in points) for i in range(3)], [max(p[i] for p in points) for i in range(3)])
    dimensions = [high[i] - low[i] for i in range(3)]
    _require(all(.0001 < v < 3 for v in dimensions), "mesh_units_or_dimensions_invalid")
    return dimensions, [(a+b)/2 for a, b in zip(low, high)], meshes


def model_urdf(root, *, identity, anchor):
    """Modify generated XML only; return deterministic assumptions and values."""
    _require(identity in PAYLOAD_KG, "unsupported_identity")
    links = {link.get("name"): link for link in root.findall("link")}
    joints = [root.find(f"joint[@name='joint_{n}']") for n in range(1, 7)]
    _require(all(j is not None and j.get("type") == "revolute" for j in joints), "joint_chain_invalid")
    for index, joint in enumerate(joints, 1):
        _require(joint.find("parent").get("link") == (f"link_{index-1}" if index > 1 else "base_link")
                 and joint.find("child").get("link") == f"link_{index}", "joint_chain_invalid")
        _require(float(joint.find("limit").get("effort")) == 0, "upstream_dynamics_changed")
    ordered = [links["base_link"], *[links[f"link_{n}"] for n in range(1, 7)]]
    shapes = [_bounds(link, anchor) for link in ordered]
    density = NOMINAL_ROBOT_KG[identity] / sum(math.prod(size) for size, _, _ in shapes)
    _require(1 < density < 20_000, "density_out_of_bounds")
    modeled = []
    for link, (size, center, meshes) in zip(ordered, shapes):
        _require(link.find("inertial") is None, "upstream_dynamics_changed")
        mass = density * math.prod(size)
        inertia = [mass * (size[(i+1)%3]**2 + size[(i+2)%3]**2) / 12 for i in range(3)]
        _require(.001 < mass < 100 and all(0 < v < 100 for v in inertia), "mass_or_inertia_out_of_bounds")
        inertial = XML.SubElement(link, "inertial")
        XML.SubElement(inertial, "origin", xyz=" ".join(map(str, center)), rpy="0 0 0")
        XML.SubElement(inertial, "mass", value=str(mass))
        XML.SubElement(inertial, "inertia", ixx=str(inertia[0]), iyy=str(inertia[1]), izz=str(inertia[2]),
                       ixy="0", ixz="0", iyz="0")
        modeled.append({"link": link.get("name"), "mass_kg": mass, "center_of_mass_m": center,
                        "diagonal_inertia_kg_m2": inertia, "bounds_m": size, "collision_meshes": meshes})
    moving = modeled[1:]
    drives = []
    lengths = [math.sqrt(sum(_origin_matrix(j)[i][3]**2 for i in range(3))) for j in joints]
    for index, joint in enumerate(joints):
        # Triangle-inequality lever bounds are orientation-independent. The
        # payload is nominal variant capacity, not measured motor capability.
        radii = [sum(lengths[index+1:k+1]) + math.sqrt(sum(v*v for v in moving[k]["center_of_mass_m"]))
                 + .5 * math.sqrt(sum(v*v for v in moving[k]["bounds_m"])) for k in range(index, 6)]
        payload_radius = sum(lengths[index+1:]) + .15  # Explicit hypothetical tool/payload lever, no tool geometry.
        gravity_moment = 9.81 * (sum(row["mass_kg"] * radius for row, radius in zip(moving[index:], radii))
                                 + PAYLOAD_KG[identity] * payload_radius)
        effort = max(1.0, 2.0 * gravity_moment)
        effective_inertia = sum(row["mass_kg"]*radius**2 + max(row["diagonal_inertia_kg_m2"])
                                for row, radius in zip(moving[index:], radii)) + PAYLOAD_KG[identity]*payload_radius**2
        stiffness = effort / .1  # Nominal saturation at 0.1 rad position error.
        damping = 2 * math.sqrt(stiffness * effective_inertia)
        _require(all(math.isfinite(v) and 0 < v < 100_000 for v in (effort, stiffness, damping)), "drive_out_of_bounds")
        joint.find("limit").set("effort", str(effort))
        XML.SubElement(joint, "drive", stiffness=str(stiffness), damping=str(damping))
        drives.append({"joint": joint.get("name"), "effort_limit_nm": effort, "stiffness_nm_per_rad": stiffness,
                       "damping_nm_s_per_rad": damping, "gravity_moment_bound_nm": gravity_moment})
    report = {"schema_version": "gofa_simulation_model.v1", "embodiment_id": identity,
              "claim_ceiling": "unvalidated_simulation_estimate", "manufacturer_dynamics": False,
              "method": "uniform_collision_aabb_box_in_link_frame", "effective_density_kg_m3": density,
              "nominal_robot_mass_kg": NOMINAL_ROBOT_KG[identity], "mass_reference": MASS_REFERENCE,
              "mass_distribution_assumption": "uniform_density_over_base_and_six_link_collision_aabbs",
              "modeled_robot_mass_total_kg": sum(row["mass_kg"] for row in modeled),
              "nominal_payload_kg": PAYLOAD_KG[identity], "hypothetical_payload_lever_m": .15,
              "effort_gravity_margin": 2.0, "links": modeled, "joints": drives,
              "moving_link_mass_total_kg": sum(row["mass_kg"] for row in moving),
              "limitations": ["hollow_structure_and_actuator_mass_distribution_unknown",
                              "nominal_total_mass_does_not_establish_per_link_distribution", "not_manufacturer_motor_limits",
                              "no_application_tool_modeled", "not_physical_feasibility_or_policy_ranking_evidence"],
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
    expected = model_urdf(ElementTree.fromstring(source.read_bytes()), identity=report["embodiment_id"], anchor=root)
    _require(expected == report, "model_report_changed")


def usd_values(path, report, *, apply=False):
    """Author/read back local body-frame SI estimates on the actual saved USD."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(path) if apply else Sdf.Layer.OpenAsAnonymous(str(path)))
    _require(stage is not None and math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0)
             and math.isclose(UsdPhysics.GetStageKilogramsPerUnit(stage), 1.0), "usd_units_invalid")
    prims = list(stage.Traverse())
    def named(name, api):
        matches = [p for p in prims if p.GetName() == name and (p.HasAPI(api) if api == UsdPhysics.RigidBodyAPI else p.IsA(api))]
        _require(len(matches) == 1, "usd_binding_missing:" + name)
        return matches[0]
    observations = {"links": [], "joints": []}
    for row in report["links"]:
        prim = named(row["link"], UsdPhysics.RigidBodyAPI)
        mass = UsdPhysics.MassAPI.Apply(prim) if apply else UsdPhysics.MassAPI(prim)
        if apply:
            mass.CreateMassAttr(row["mass_kg"])
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*row["center_of_mass_m"]))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*row["diagonal_inertia_kg_m2"]))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
        actual = {"link": row["link"], "mass_kg": mass.GetMassAttr().Get(),
                  "center_of_mass_m": list(mass.GetCenterOfMassAttr().Get()),
                  "diagonal_inertia_kg_m2": list(mass.GetDiagonalInertiaAttr().Get())}
        _require(mass.GetPrincipalAxesAttr().Get() == Gf.Quatf(1, 0, 0, 0), "usd_inertia_frame_changed")
        for key in ("mass_kg", "center_of_mass_m", "diagonal_inertia_kg_m2"):
            observed, expected = actual[key], row[key]
            _require(all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)
                         for a, b in zip(observed if isinstance(observed, list) else [observed],
                                         expected if isinstance(expected, list) else [expected])), "usd_mass_inertia_changed")
        observations["links"].append(actual)
    for row in report["joints"]:
        prim = named(row["joint"], UsdPhysics.RevoluteJoint)
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular") if apply else UsdPhysics.DriveAPI(prim, "angular")
        if apply:
            drive.CreateTypeAttr("force")
            drive.CreateMaxForceAttr(row["effort_limit_nm"])
        _require(drive.GetTypeAttr().Get() == "force", "usd_drive_type_changed")
        value = drive.GetMaxForceAttr().Get()
        _require(value is not None and math.isclose(value, row["effort_limit_nm"], rel_tol=1e-6), "usd_effort_changed")
        observations["joints"].append({"joint": row["joint"], "effort_limit_nm": value, "drive_type": drive.GetTypeAttr().Get()})
    if apply:
        stage.GetRootLayer().Save()
        return usd_values(path, report)
    return observations


def annotate_configuration(result, original, report):
    result["source_dynamics"] = {**original["source_dynamics"], "estimated_inertias_added": True}
    result["source_limitations"] = original["source_limitations"]
    result["simulation_model"] = report
    result["simulation_assumptions"] += report["limitations"] + ["gofa_geometry_based_dynamics_and_drive_estimates"]


def verify_configuration(root, source, record):
    from . import deployment_embodiment_runtime_configuration as runtime
    verify_report(root, record.get("simulation_model", {}), identity=source["embodiment_id"])
    unmodeled = runtime.SafeXML.fromstring((root / SOURCE_MODEL).read_bytes())
    original = runtime._configuration(unmodeled, source["embodiment_id"], runtime._PROFILES[source["embodiment_id"]], source, None)
    report = model_urdf(unmodeled, identity=source["embodiment_id"], anchor=root)
    model = root / "robot.urdf"
    _require(XML.tostring(unmodeled, encoding="unicode") == model.read_text(), "generated_urdf_changed")
    expected = runtime._configuration(unmodeled, source["embodiment_id"], runtime._PROFILES[source["embodiment_id"]], source, None)
    expected["source_urdf_sha256"] = runtime._sha(model)
    annotate_configuration(expected, original, report)
    retained = {key: value for key, value in record.items()
                if key not in {"generator_module_sha256", "configuration_digest", "generated_files"}}
    _require(expected == retained, "configuration_metadata_changed")
