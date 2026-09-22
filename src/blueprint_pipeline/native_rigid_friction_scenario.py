"""Apply a rigid task's friction scenario before import; verify native shape values."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile

KIND = "task_subject_rigid_dynamic_friction"


def add_application(parameters, subject, applications, coverage_gaps):
    if "dynamic_friction" not in parameters:
        return
    if subject.get("object_type") == "ARTICULATION":
        coverage_gaps.append({"family": "bounded_physics",
            "reason": "runtime_material_link_binding_unavailable", "fallback": "canonical_task_material"})
        return
    applications.append({"parameter_id": "dynamic_friction", "unit": "coefficient",
        "readback_kind": KIND, "expected_native_value": float(parameters["dynamic_friction"]),
        "application_tolerance": 1.0e-6})


def prepare(runtime_objects, applications):
    selected = [row for row in applications if row.get("readback_kind") == KIND]
    if not selected:
        return {}
    if len(selected) != 1:
        raise ValueError("rigid_friction_scenario_not_unique")
    application = selected[0]
    subjects = [row for row in runtime_objects if row.get("task_subject") is True]
    if len(subjects) != 1:
        raise ValueError("rigid_friction_subject_not_unique")
    subject = subjects[0]
    value = float(application["expected_native_value"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("rigid_friction_value_invalid")
    from pxr import Usd, UsdPhysics, UsdShade

    source = Path(subject["usd_path"]).resolve()
    stage = Usd.Stage.Open(str(source))
    materials = set()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError("rigid_friction_requires_rigid_subject")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            if not material or not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                raise ValueError("rigid_friction_collision_material_missing")
            materials.add(str(material.GetPath()))
    if len(materials) != 1:
        raise ValueError("rigid_friction_material_not_unique")
    material_path = materials.pop()
    before = float(UsdPhysics.MaterialAPI(stage.GetPrimAtPath(material_path)).GetDynamicFrictionAttr().Get())
    fd, name = tempfile.mkstemp(prefix="blueprint-rigid-friction-", suffix=".usda")
    os.close(fd)
    destination = Path(name)
    destination.unlink()
    derived = Usd.Stage.CreateNew(str(destination))
    derived.GetRootLayer().subLayerPaths = [str(source)]
    material = derived.OverridePrim(material_path)
    UsdPhysics.MaterialAPI(material).GetDynamicFrictionAttr().Set(value)
    derived.SetDefaultPrim(derived.GetPrimAtPath(stage.GetDefaultPrim().GetPath()))
    derived.GetRootLayer().Save()
    subject["usd_path"] = str(destination)
    return {application["parameter_id"]: {
        "runtime_name": application["runtime_name"], "source_usd": str(source),
        "source_material_prim_path": material_path, "source_dynamic_friction": before,
        "expected_dynamic_friction": value, "derived_usd": str(destination),
        "derived_usd_sha256": "sha256:" + hashlib.sha256(destination.read_bytes()).hexdigest(),
    }}


def verify(env, overrides):
    """Read PhysX values, not just a USD edit that might not reach simulation."""
    results = {}
    for parameter_id, record in overrides.items():
        subject = env.unwrapped.scene[record["runtime_name"]]
        values = subject.root_physx_view.get_material_properties().tolist()
        dynamic = [float(shape[1]) for instance in values for shape in instance]
        expected = record["expected_dynamic_friction"]
        if not dynamic or any(not math.isclose(value, expected, rel_tol=0, abs_tol=1e-6) for value in dynamic):
            raise ValueError("rigid_friction_native_readback_mismatch")
        results[parameter_id] = {**record, "observed_dynamic_friction": dynamic[0],
            "native_shape_dynamic_friction": dynamic, "source": "physx_material_properties"}
    return results
