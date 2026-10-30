"""Apply/read back the fixed iiwa14 source physics, without a generic converter.

Angular PhysxJointAPI.maxJointVelocity is degrees/s (IsaacSim6.0.1,
045ca8b, experimental/prims/articulation.py:3023). Public contracts use rad/s.
Source sphere collisions replace importer approximations on the eight arm
bodies only. The tool retains the recorded official importer's mesh treatment.
"""
from __future__ import annotations

import math
import numpy as np

from .deployment_embodiment_kuka_bridge import require
from .team_arm_kinematics import pose_matrix, quaternion_xyzw


def source_physics(path, report, *, apply=False):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(path) if apply else Sdf.Layer.OpenAsAnonymous(str(path)))
    require(stage is not None and UsdGeom.GetStageMetersPerUnit(stage) == 1
            and UsdPhysics.GetStageKilogramsPerUnit(stage) == 1, "usd_physics_units_changed")
    prims = list(stage.Traverse())

    def named(name, schema):
        rows = [p for p in prims if p.GetName() == name and (
            p.HasAPI(schema) if schema == UsdPhysics.RigidBodyAPI else p.IsA(schema))]
        require(len(rows) == 1, "usd_binding_missing:" + name)
        return rows[0]

    bodies = {row["link"]: named(row["link"], UsdPhysics.RigidBodyAPI) for row in report["links"]}
    result = {"links": [], "joints": []}
    for row in report["links"]:
        prim = bodies[row["link"]]
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
        diagonal, q, com = (mass.GetDiagonalInertiaAttr().Get(), mass.GetPrincipalAxesAttr().Get(),
                            mass.GetCenterOfMassAttr().Get())
        require(diagonal is not None and q is not None and com is not None, "usd_inertial_missing")
        rotation = pose_matrix([0, 0, 0], [*q.GetImaginary(), q.GetReal()])[:3, :3]
        actual = {"link": row["link"], "mass_kg": mass.GetMassAttr().Get(), "center_of_mass_m": list(com),
                  "inertia_body_frame_kg_m2": (rotation @ np.diag(diagonal) @ rotation.T).tolist()}
        for key in ("mass_kg", "center_of_mass_m", "inertia_body_frame_kg_m2"):
            require(actual[key] is not None and np.allclose(actual[key], row[key], rtol=2e-6, atol=(max(np.linalg.eigvalsh(row[key]))*2e-6
                    if key == "inertia_body_frame_kg_m2" else 1e-10)),
                    "usd_mass_inertia_changed")
        result["links"].append(actual)
    if apply:
        from pxr import PhysxSchema
    for row in report["joints"]:
        prim = named(row["joint"], UsdPhysics.RevoluteJoint)
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular") if apply else UsdPhysics.DriveAPI(prim, "angular")
        if apply:
            drive.CreateTypeAttr("force")
            drive.CreateMaxForceAttr(row["effort_limit_nm"])
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(math.degrees(row["velocity_limit_rad_s"]))
        effort = drive.GetMaxForceAttr().Get()
        velocity = prim.GetAttribute("physxJoint:maxJointVelocity").Get()
        require(drive.GetTypeAttr().Get() == "force" and effort is not None
                and np.isclose(effort, row["effort_limit_nm"], rtol=1e-6), "usd_effort_changed")
        require("PhysxJointAPI" in prim.GetMetadata("apiSchemas").GetAppliedItems() and velocity is not None and np.isfinite(velocity)
                and np.isclose(math.radians(velocity), row["velocity_limit_rad_s"], rtol=1e-6), "usd_velocity_changed")
        result["joints"].append({"joint": row["joint"], "effort_limit_nm": effort, "drive_type": "force",
                                 "velocity_limit_rad_s": math.radians(velocity), "usd_velocity_degrees_s": velocity})
    roots = [p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    require(len(roots) == 1, "usd_articulation_root_ambiguous")
    articulation = roots[0]
    if apply:
        PhysxSchema.PhysxArticulationAPI.Apply(articulation).CreateEnabledSelfCollisionsAttr(True)
    require("PhysxArticulationAPI" in articulation.GetMetadata("apiSchemas").GetAppliedItems()
            and articulation.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True,
            "usd_self_collisions_disabled")
    result["enabled_self_collisions"] = True
    result["source_collision_spheres"] = _spheres(stage, bodies, report, apply)
    result["source_contact_exclusions"] = _filters(stage, bodies, report, apply)
    if apply:
        stage.GetRootLayer().Save()
        return source_physics(path, report)
    return result


def _spheres(stage, bodies, report, apply):
    from pxr import Gf, UsdGeom, UsdPhysics
    arm = {row["name"] for row in report["source_bridge"]["source_kinematics"]["bodies"]}
    specs = report["source_bridge"]["collision_spheres"]
    expected = {str(bodies[row["link"]].GetPath().AppendChild("blueprint_source_"+row["name"])): row for row in specs}

    def owner(prim):
        while prim and not prim.IsPseudoRoot():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim.GetName()
            prim = prim.GetParent()
        return None

    if apply:
        for prim in list(stage.Traverse()):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and owner(prim) in arm:
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        for path, row in expected.items():
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.CreateRadiusAttr(row["radius_m"])
            sphere.CreatePurposeAttr("guide")
            xform = UsdGeom.Xformable(sphere)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*row["position_m"]))
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim()).CreateCollisionEnabledAttr(True)
    active = {str(p.GetPath()): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
              and owner(p) in arm and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()}
    require(set(active) == set(expected), "usd_arm_collision_inventory_changed")
    rows = []
    for path, row in expected.items():
        prim = active[path]
        require(prim.IsA(UsdGeom.Sphere), "usd_sphere_type_changed")
        radius = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        xform = UsdGeom.Xformable(prim)
        transform = np.array(xform.GetLocalTransformation()).T
        expected_transform = np.eye(4)
        expected_transform[:3, 3] = row["position_m"]
        require(not xform.GetResetXformStack() and radius is not None
                and np.isclose(radius, row["radius_m"], rtol=1e-9)
                and np.allclose(transform, expected_transform, rtol=0, atol=1e-10), "usd_sphere_geometry_changed")
        rows.append({**row, "radius_m": radius, "position_m": transform[:3, 3].tolist()})
    return rows


def _filters(stage, bodies, report, apply):
    from pxr import UsdPhysics
    source = report["source_bridge"]["source_contact_exclusions"]
    expected = {tuple(sorted((row["body1"], row["body2"]))) for row in source}
    arm = {row["name"] for row in report["source_bridge"]["source_kinematics"]["bodies"]}
    paths = {str(prim.GetPath()): name for name, prim in bodies.items()}
    arm_paths = {bodies[name].GetPath() for name in arm}
    def touches_arm(prim):
        owner = prim
        while owner and not owner.IsPseudoRoot():
            if owner.HasAPI(UsdPhysics.RigidBodyAPI):
                return owner.GetPath() in arm_paths
            owner = owner.GetParent()
        return False
    if apply:
        # Clear importer-authored suppression involving this arm before writing
        # the seven pinned exclusions. Tool-only filters remain unchanged.
        for prim in stage.Traverse():
            relation = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
            targets = relation.GetTargets()
            if targets and (touches_arm(prim) or any(touches_arm(stage.GetPrimAtPath(p)) for p in targets)):
                relation.ClearTargets(True)
        for name in arm:
            api = UsdPhysics.FilteredPairsAPI.Apply(bodies[name])
            api.CreateFilteredPairsRel().SetTargets([bodies[b].GetPath() for a, b in sorted(expected) if a == name])
    actual = set()
    for name in arm:
        for path in UsdPhysics.FilteredPairsAPI(bodies[name]).GetFilteredPairsRel().GetTargets():
            require(str(path) in paths, "usd_filter_target_changed")
            actual.add(tuple(sorted((name, paths[str(path)]))))
    for prim in stage.Traverse():
        targets = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
        if targets and (touches_arm(prim) or any(touches_arm(stage.GetPrimAtPath(p)) for p in targets)):
            require(prim.GetPath() in arm_paths and all(str(p) in paths and paths[str(p)] in arm for p in targets),
                    "usd_unexpected_arm_filter")
    require(actual == expected and len(actual) == 7, "usd_source_collision_filters_changed")
    return [{"body1": a, "body2": b} for a, b in sorted(actual)]
