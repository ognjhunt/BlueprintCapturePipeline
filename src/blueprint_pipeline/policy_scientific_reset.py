"""Measured reset evidence and paired comparison, independent of simulator labels.

Missing native channels remain typed gaps. A diagnostic can retain a gap; a
qualified comparison cannot treat it as equality. No whole episode artifact
hash is compared across candidates.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

try:
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:
    from .decision_evidence_contracts import canonical_digest

SCHEMA = "policy_scientific_reset.v1"
REQUIRED_CHANNELS = frozenset({"robot", "objects", "scene_assets", "cameras", "physics", "lighting", "colliders", "contacts"})


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _differences(left: Any, right: Any, tolerances: Mapping[str, float], path: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        errors = [path + "/" + str(key) for key in set(left) ^ set(right)]
        for key in sorted(set(left) & set(right)):
            errors.extend(_differences(left[key], right[key], tolerances, path + "/" + str(key)))
        return errors
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        return [error for index, (a, b) in enumerate(zip(left, right, strict=True))
                for error in _differences(a, b, tolerances, path + "/" + str(index))]
    tolerance = tolerances.get(path, 0.0)
    numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (left, right))
    equal = (math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance) if numeric else type(left) is type(right) and left == right
    return [] if equal else [path]


def seal_reset_readback(*, binding: Mapping[str, Any], observed: Mapping[str, Any],
                        sources: Mapping[str, str], gaps: list[str],
                        tolerances: Mapping[str, float] | None = None) -> dict[str, Any]:
    binding = _json(binding)
    observed = _json(observed)
    tolerances = _json(tolerances or {})
    if not all(binding.get(key) for key in ("cell_id", "candidate_id", "task_spec_digest", "resolved_scenario_digest")) or type(binding.get("seed")) is not int:
        raise ValueError("scientific_reset_binding_invalid")
    if any(not isinstance(key, str) or not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value < 0 for key, value in tolerances.items()):
        raise ValueError("scientific_reset_tolerance_invalid")
    missing = sorted(REQUIRED_CHANNELS - set(observed))
    gaps = sorted(set([*gaps, *("missing_native_channel:" + key for key in missing)]))
    if any(not observed.get(key) or not sources.get(key) for key in REQUIRED_CHANNELS & set(observed)):
        raise ValueError("scientific_reset_empty_native_channel")
    value = {"schema_version": SCHEMA, "binding": binding, "observed": observed,
             "sources": dict(sources), "tolerances": tolerances, "gaps": gaps,
             "complete": not gaps, "scientific_state_digest": canonical_digest(observed)}
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def validate_reset_readback(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(value)
    if receipt.get("schema_version") != SCHEMA or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest"):
        raise ValueError("scientific_reset_receipt_digest_invalid")
    expected = seal_reset_readback(binding=receipt["binding"], observed=receipt["observed"],
        sources=receipt["sources"], gaps=receipt["gaps"], tolerances=receipt["tolerances"])
    if receipt != expected:
        raise ValueError("scientific_reset_receipt_invalid")
    return receipt


def compare_reset_readbacks(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left, right = validate_reset_readback(left), validate_reset_readback(right)
    bindings = (left["binding"], right["binding"])
    # Candidate id is metadata; all scientific execution identities must agree.
    identity_fields = ("cell_id", "seed", "task_spec_digest", "resolved_scenario_digest")
    mismatches = ["binding/" + field for field in identity_fields if bindings[0][field] != bindings[1][field]]
    mismatches.extend("binding/" + field for field in ("matrix_cell_digest", "matrix_reset_digest")
                      if bindings[0].get(field) != bindings[1].get(field))
    if left["tolerances"] != right["tolerances"]:
        mismatches.append("tolerances")
    mismatches.extend(_differences(left["observed"], right["observed"], left["tolerances"]))
    gaps = sorted(set(left["gaps"] + right["gaps"]))
    result = {"schema_version": "policy_scientific_reset_parity.v1",
              "left_receipt_digest": left["receipt_digest"], "right_receipt_digest": right["receipt_digest"],
              "status": "mismatch" if mismatches else "unverified" if gaps else "matched",
              "mismatches": sorted(set(mismatches)), "gaps": gaps,
              "comparison_eligible": not mismatches and not gaps}
    result["parity_digest"] = canonical_digest(result, digest_field="parity_digest")
    return result


def _native(value: Any) -> Any:
    if value is None:
        raise ValueError("native_value_missing")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        value = {str(key): _native(item) for key, item in value.items()}
    elif not isinstance(value, (str, int, float, bool, list)) and hasattr(value, "__iter__"):
        value = [_native(item) for item in value]
    return _json(value)


def read_native_reset_channels(built: Any, episode_environment: Any) -> dict[str, Any]:
    """Read current Isaac tensors/sensors and the loaded stage, never plan values.

    This runs after reset in the provider process. Missing API surfaces are
    gaps, not copied expected values. The same seam is exercised with spies.
    """
    observed: dict[str, Any] = {}
    sources: dict[str, str] = {}
    gaps: list[str] = []
    scene = getattr(getattr(built.env, "unwrapped", built.env), "scene", {})

    def channel(name, reader, source):
        try:
            observed[name] = _json(reader())
            sources[name] = source
        except (AttributeError, KeyError, TypeError, ValueError, IndexError, ImportError, RuntimeError) as exc:
            gaps.append(name + ":" + type(exc).__name__)

    def robot():
        data = scene["robot"].data
        return {name: _native(getattr(data, name)) for name in ("joint_pos", "joint_vel", "root_pose_w", "root_vel_w", "joint_limits")}

    def objects():
        result = {}
        dynamic_names = {row["name"] for row in built.plan["objects"]
                         if row.get("object_type") in {"RIGID", "ARTICULATION"} or row.get("task_subject") is True}
        for name in sorted(dynamic_names):
            scene_name = built.scene_asset_names[name]
            asset = scene[scene_name]
            data, view = asset.data, asset.root_physx_view
            result[name] = {"root_pose_w": _native(data.root_pose_w), "root_vel_w": _native(data.root_vel_w),
                "masses": _native(view.get_masses()), "inertias": _native(view.get_inertias()),
                "material_properties": _native(view.get_material_properties())}
        if not result:
            raise ValueError("native_objects_missing")
        return result

    def contacts():
        names = sorted({name for names in built.contact_sensor_names.values() for name in names})
        if not names:
            raise ValueError("native_contact_sensors_missing")
        return {name: {"net_forces_w": _native(scene[name].data.net_forces_w),
                       "force_matrix_w": _native(scene[name].data.force_matrix_w)} for name in names}

    def physics():
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        values = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                values[str(prim.GetPath())] = {attr.GetName(): _native(attr.Get()) for attr in prim.GetAttributes()
                    if attr.GetName().startswith(("physics:", "physxScene:")) and attr.Get() is not None}
        if not values:
            raise ValueError("native_physics_scene_missing")
        return {"scene_attributes": values, "physics_dt_seconds": float(built.env.unwrapped.sim.get_physics_dt()),
                "control_step_dt_seconds": float(built.env.unwrapped.step_dt)}

    def colliders():
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        values = {}
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                properties = {attr.GetName(): _native(attr.Get()) for attr in prim.GetAttributes()
                    if attr.GetName().startswith(("physics:", "physxCollision:")) and attr.Get() is not None}
                geometry = {attr.GetName(): canonical_digest({"value": _native(attr.Get())}) for attr in prim.GetAttributes()
                    if (attr.GetName() in {"points", "faceVertexCounts", "faceVertexIndices", "radius", "height", "size", "axis"}) and attr.Get() is not None}
                values[str(prim.GetPath())] = {"properties": properties, "geometry_attribute_digests": geometry,
                    "world_transform": _native(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))}
        if not values:
            raise ValueError("native_colliders_missing")
        return values

    def lighting():
        import omni.usd
        from pxr import Usd, UsdGeom, UsdLux
        stage = omni.usd.get_context().get_stage()
        values = {}
        for prim in stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                values[str(prim.GetPath())] = {"inputs": {attr.GetName(): _native(attr.Get()) for attr in prim.GetAttributes()
                    if attr.GetName() in {"inputs:intensity", "inputs:exposure", "inputs:color", "inputs:colorTemperature", "inputs:enableColorTemperature"} and attr.Get() is not None},
                    "world_transform": _native(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))}
        if not values:
            raise ValueError("native_lighting_readback_missing")
        return values

    def scene_assets():
        import omni.usd
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        prims = list(stage.Traverse())
        values = {}
        for name in built.scene_asset_names:
            matches = [prim for prim in prims if str(prim.GetPath()).endswith("/env_0/" + name)]
            if len(matches) != 1:
                raise ValueError("native_scene_asset_prim_missing_or_ambiguous:" + name)
            prim = matches[0]
            values[name] = {"prim_path": str(prim.GetPath()),
                "world_transform": _native(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())),
                "visibility": str(UsdGeom.Imageable(prim).ComputeVisibility())}
        if not values:
            raise ValueError("native_scene_assets_missing")
        return values

    channel("robot", robot, "live_isaac_robot_tensors")
    channel("objects", objects, "live_isaac_object_tensors_and_physx_view")
    channel("scene_assets", scene_assets, "loaded_usd_registered_asset_transforms")
    channel("contacts", contacts, "live_isaac_filtered_contact_sensors")
    channel("cameras", lambda: episode_environment.read_control_observation_metadata()["calibrations"], "live_isaac_camera_calibration")
    channel("physics", physics, "loaded_usd_physics_scene_and_simulation_context")
    channel("lighting", lighting, "loaded_usd_light_api")
    channel("colliders", colliders, "loaded_usd_collision_api")
    applications = (built.plan.get("scenario") or {}).get("parameter_applications") or []
    if applications:
        from .native_task_arena_readback import read_native_task_arena_scenario_parameters
        channel("scenario_parameters", lambda: read_native_task_arena_scenario_parameters(built), "live_native_scenario_parameter_readback")
        if observed.get("scenario_parameters", {}).get("passed") is False:
            raise ValueError("scientific_reset_scenario_application_mismatch")
    gaps.extend("unapplied_scenario:" + str(row.get("family")) for row in
                (built.plan.get("scenario") or {}).get("runtime_coverage_gaps") or [])
    return {"observed": observed, "sources": sources, "gaps": gaps}
