"""Apply a rigid task's friction scenario before import; verify native shape values."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile

KIND = "task_subject_rigid_dynamic_friction"


class UnboundRigidMaterial(ValueError):
    """Every rigid collider uses the simulator's unauthored material."""


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


def _source_material(source):
    from pxr import Usd, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(source))
    materials = set()
    unbound = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError("rigid_friction_requires_rigid_subject")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            if not material:
                unbound += 1
                continue
            if not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                raise ValueError("rigid_friction_collision_material_missing")
            materials.add(str(material.GetPath()))
    if unbound and not materials:
        raise UnboundRigidMaterial("rigid_friction_collision_material_unbound")
    if unbound:
        raise ValueError("rigid_friction_collision_material_missing")
    if len(materials) != 1:
        raise ValueError("rigid_friction_material_not_unique")
    material_path = materials.pop()
    before = float(UsdPhysics.MaterialAPI(stage.GetPrimAtPath(material_path)).GetDynamicFrictionAttr().Get())
    return stage, material_path, before


def scenario_application(binding, subject, source):
    if source is None:
        raise ValueError("rigid_friction_material_asset_missing")
    _, material_path, before = _source_material(source)
    if not math.isclose(before, binding["nominal_value"], rel_tol=0, abs_tol=binding["application_tolerance"]):
        raise ValueError("rigid_friction_nominal_mismatch")
    return {**binding, "readback_kind": KIND, "expected_native_value": binding["resolved_value"],
            "runtime_name": subject["name"], "source_material_prim_path": material_path}


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
    from pxr import Usd, UsdPhysics

    source = Path(subject["usd_path"]).resolve()
    try:
        stage, material_path, before = _source_material(source)
    except UnboundRigidMaterial:
        # The importer's defaults are not source evidence. Measure them after
        # construction and change only the requested property before reset.
        return {application["parameter_id"]: {
            "application_mode": "native_material_properties_before_reset",
            "runtime_name": application["runtime_name"], "source_usd": str(source),
            "source_material_prim_path": None, "source_dynamic_friction": None,
            "expected_dynamic_friction": value,
        }}
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


def _material_view(env, record):
    subject = env.unwrapped.scene[record["runtime_name"]]
    return subject.root_view if hasattr(subject, "root_view") else subject.root_physx_view


def _material_array(view):
    import numpy as np

    values = view.get_material_properties()
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    elif hasattr(values, "numpy"):
        values = values.numpy()
    result = np.asarray(values)
    if (result.ndim != 3 or result.shape[-1] != 3 or not result.size
            or not np.isfinite(result).all()):
        raise ValueError("rigid_friction_native_material_properties_invalid")
    return result.copy()


def apply_native_overrides(env, overrides):
    """Use the pinned SDK's native setter for wholly unbound rigid shapes."""
    import numpy as np

    applied = {}
    for parameter_id, record in overrides.items():
        if record.get("application_mode") != "native_material_properties_before_reset":
            applied[parameter_id] = record
            continue
        view = _material_view(env, record)
        before = _material_array(view)
        expected = float(record["expected_dynamic_friction"])
        if not math.isfinite(expected) or expected < 0:
            raise ValueError("rigid_friction_value_invalid")
        updated = before.copy()
        updated[..., 1] = expected
        indices = np.arange(len(updated), dtype=np.int32)
        subject = env.unwrapped.scene[record["runtime_name"]]
        if hasattr(subject, "root_view"):
            # Isaac Lab 6.0.1 root_view uses Warp arrays, including CPU indices.
            import torch
            import warp as wp

            view.set_material_properties(
                wp.from_torch(torch.from_numpy(updated), dtype=wp.float32),
                wp.from_torch(torch.from_numpy(indices), dtype=wp.int32),
            )
        else:
            native = view.get_material_properties()
            if hasattr(native, "detach"):
                import torch

                view.set_material_properties(torch.as_tensor(updated, device=native.device),
                                             torch.as_tensor(indices, device="cpu"))
            else:
                view.set_material_properties(updated, indices)
        after = _material_array(view)
        if (after.shape != before.shape
                or not np.array_equal(after[..., (0, 2)], before[..., (0, 2)])):
            raise ValueError("rigid_friction_unrequested_material_property_changed")
        if not np.allclose(after[..., 1], expected, rtol=0, atol=1e-6):
            raise ValueError("rigid_friction_native_readback_mismatch")
        baseline = before[..., 1].reshape(-1)
        applied[parameter_id] = {**record,
            "source_native_material_properties": before.tolist(),
            "applied_native_material_properties": after.tolist(),
            "source_dynamic_friction": float(baseline[0]) if np.all(baseline == baseline[0]) else None,
            "unrequested_material_properties_preserved": True,
        }
    return applied


def verify(env, overrides):
    """Read PhysX values, not just a USD edit that might not reach simulation."""
    results = {}
    for parameter_id, record in overrides.items():
        values = _material_array(_material_view(env, record)).tolist()
        dynamic = [float(shape[1]) for instance in values for shape in instance]
        expected = record["expected_dynamic_friction"]
        if not dynamic or any(not math.isclose(value, expected, rel_tol=0, abs_tol=1e-6) for value in dynamic):
            raise ValueError("rigid_friction_native_readback_mismatch")
        results[parameter_id] = {**record, "observed_dynamic_friction": dynamic[0],
            "native_shape_dynamic_friction": dynamic, "source": "physx_material_properties"}
    return results
