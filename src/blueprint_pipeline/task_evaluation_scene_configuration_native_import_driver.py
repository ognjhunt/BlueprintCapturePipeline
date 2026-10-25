"""Native Isaac driver for one Website scene-configuration component stage."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .articulation_graph_contract import ArticulationGraphContractError, validate_articulation_graph
from .decision_evidence_contracts import canonical_digest, canonical_json
from .measurement_isaac_physx_rigid_adapter import (
    ISAAC_VERSION,
    _bind_isaac_runtime_environment,
    _import_simulation_app,
    _observe_isaac_runtime_identity,
)
from .task_evaluation_scene_configuration_stage_tool import (
    COMPONENT_RESULT_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_stage_configuration import (
    native_import_checks_valid,
)


ADAPTER_ID = "simready_native_import_qualification"
RUNTIME_RESULT_SCHEMA_VERSION = "task_evaluation_replacement_native_import_result.v1"
ROBOT_FREE_PROBE_SCHEMA = "scene_configuration_robot_free_success_probe.v1"
_PROBE_DRIVE_SECONDS = 1.0
# The rate robot teams' episodes are simulated and scored at, so the probe's
# physics matches theirs and decimates exactly to any admitted control rate.
_PROBE_PHYSICS_HZ = 120
_PROBE_RELEASE_SECONDS = 5.0
DESTINATION_ASSET_DEPENDENCY_ROLE = "statically_qualified_destination_asset"
DESTINATION_STATIC_DEPENDENCY_ROLE = "destination_static_qualification_receipt"
DESTINATION_RUNTIME_RESULT_ROLE = "destination_native_import_runtime_result"
_STAGE_INPUT_SCHEMA = "task_evaluation_scene_configuration_stage_production_input.v1"
_COMPONENT_RESULT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT"
_STAGE_INPUT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"
_DEPENDENCIES_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"
_OUTPUT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"
NativeObservationConsumer = Callable[[Mapping[str, Any]], Mapping[str, Any]]
NativeRunner = Callable[..., Mapping[str, Any]]


class TaskEvaluationSceneConfigurationNativeImportDriverError(RuntimeError):
    """The exact asset could not be qualified in the native runtime."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(code) from exc
    if path.is_symlink():
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(code)
    return value


def _required_path(environment: Mapping[str, str], name: str) -> Path:
    unresolved = str(environment.get(name) or "").strip()
    if not unresolved:
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            f"scene_configuration_native_import_environment_missing:{name}"
        )
    return Path(unresolved).expanduser().resolve()


def _artifact(dependencies: Any, *, role: str) -> tuple[dict[str, Any], Path]:
    if not isinstance(dependencies, list):
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            "scene_configuration_native_import_dependencies_invalid"
        )
    matches = [
        row
        for result in dependencies
        if isinstance(result, Mapping)
        for row in result.get("output_artifacts") or []
        if isinstance(row, Mapping) and row.get("role") == role
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            f"scene_configuration_native_import_dependency_missing:{role}"
        )
    record = dict(matches[0])
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("digest")
    ):
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            f"scene_configuration_native_import_dependency_invalid:{role}"
        )
    return record, path


def _declared_supplemental_destination(
    production_input: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the recipe's supplemental destination binding from the stage input."""

    envelope = production_input.get("construction_envelope")
    recipe = envelope.get("recipe") if isinstance(envelope, Mapping) else None
    destination = recipe.get("supplemental_destination") if isinstance(recipe, Mapping) else None
    if destination is None:
        return None
    identity = destination.get("identity") if isinstance(destination, Mapping) else None
    if (
        not isinstance(identity, Mapping)
        or not str(identity.get("id") or "")
        or not str(identity.get("version") or "")
    ):
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            "scene_configuration_native_import_destination_binding_invalid"
        )
    return dict(destination)


def _angle_between_xyzw(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(left, right, strict=True)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _live_pose(omni_physx: Any, prim_path: str) -> tuple[list[float], list[float]]:
    state = omni_physx.get_physx_interface().get_rigidbody_transformation(prim_path)
    if not hasattr(state, "get") or state.get("ret_val") is not True:
        raise RuntimeError("scene_configuration_native_import_live_pose_unavailable")
    position = [float(state["position"][index]) for index in range(3)]
    rotation = [float(state["rotation"][index]) for index in range(4)]
    if not all(math.isfinite(value) for value in position + rotation):
        raise RuntimeError("scene_configuration_native_import_live_pose_nonfinite")
    return position, rotation


#: A passive drawer may creep a little under gravity while its slide friction
#: settles; more than this at the closed reset is a start state a policy would
#: begin from without the part actually being shut. A hinged part uses its own
#: graph reset tolerance (radians), declared by the qualified receipt.
TASK_JOINT_RESET_TOLERANCE = 0.005
_TASK_JOINT_TYPES = frozenset({"prismatic", "revolute"})
_IMPORTED_ROOT = "/World/Placement/Replacement"


def _receipt_mechanism(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The one mechanism the static receipt admitted, in the family's own units.

    Links, root, task joint type, USD axis token and reset tolerance all come
    from the qualified graph, so a drawer cabinet (carcass, prismatic, metres)
    and a hinged appliance (body, revolute, radians) import through one path.
    """
    try:
        graph = validate_articulation_graph(receipt["articulation_graph"])
        links = set(receipt["observed_structure"]["links"])
        declared = receipt["task_joint"]
    except (KeyError, TypeError, ArticulationGraphContractError) as exc:
        raise RuntimeError("scene_configuration_native_import_mechanism_invalid") from exc
    roots = [row["link_id"] for row in graph["links"] if row["is_root"]]
    targets = [row for row in graph["joints"] if row["role"] == "target"]
    target = targets[0] if len(targets) == 1 else {}
    # Passive joints (dishwasher racks) move, are declared by the graph, hang
    # from the root and are never the task joint.
    passive = [row for row in graph["joints"] if row["role"] == "passive"]
    artisan = receipt.get("backend") == "artisan_v1"
    axis = [float(value) for value in target.get("axis") or [0.0, 0.0, 0.0]]
    index = max(range(3), key=lambda i: abs(axis[i]))
    if (len(roots) != 1 or len(targets) != 1 or len(links) < 2
            or links != {row["link_id"] for row in graph["links"]}
            or target["joint_type"] not in _TASK_JOINT_TYPES
            or (not artisan and target["parent_link_id"] != roots[0])
            or declared.get("joint_id") != target["joint_id"]
            or declared.get("joint_type") != target["joint_type"]
            or any(row["joint_type"] != "fixed" or (not artisan and row["parent_link_id"] != roots[0])
                   for row in graph["joints"] if row["role"] not in {"target", "passive"})
            or any(row["joint_type"] not in _TASK_JOINT_TYPES or (not artisan and row["parent_link_id"] != roots[0])
                   for row in passive)
            or (not artisan and (not math.isclose(abs(axis[index]), 1.0, abs_tol=1e-6)
                    or any(abs(axis[i]) > 1e-6 for i in range(3) if i != index)))):
        raise RuntimeError("scene_configuration_native_import_mechanism_invalid")
    return {
        "links": frozenset(links),
        "root_link_id": roots[0],
        "task_joint_type": target["joint_type"],
        "task_child_link_id": target["child_link_id"],
        # Joint frames rotate only by 180 degrees about X, so the USD token is
        # the axis-aligned direction of the graph axis.
        "task_joint_usd_axis": str(declared.get("usd_axis") or "XYZ"[index]),
        "fixed_joint_count": len(graph["joints"]) - 1 - len(passive),
        "passive_joint_ids": frozenset(row["joint_id"] for row in passive),
        "reset_tolerance": float(target["reset_tolerance"]),
    }


def _imported_path(asset_path: str) -> str:
    if not asset_path.startswith("/Asset/"):
        raise RuntimeError("scene_configuration_native_import_static_path_invalid")
    return _IMPORTED_ROOT + asset_path[len("/Asset"):]


def _finite_vector(value: Any, count: int, *, positive: bool = False) -> list[float]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != count:
        raise RuntimeError("scene_configuration_native_import_numeric_readback_invalid")
    numbers = [float(item) for item in value]
    if not all(math.isfinite(item) and (not positive or item > 0.0) for item in numbers):
        raise RuntimeError("scene_configuration_native_import_numeric_readback_invalid")
    return numbers


def _articulated_structure_observation(
    *, stage: Any, usd_physics: Any, static_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare the imported composition with the exact static receipt's parts."""

    expected = static_receipt["observed_structure"]
    expected_links = expected["links"]
    mechanism = _receipt_mechanism(static_receipt)
    link_ids = mechanism["links"]
    prims = list(stage.Traverse())
    roots = [str(prim.GetPath()) for prim in prims if prim.HasAPI(usd_physics.ArticulationRootAPI)]
    bodies = [prim for prim in prims if prim.HasAPI(usd_physics.RigidBodyAPI)]
    actual_links = {prim.GetName(): prim for prim in bodies}
    if roots != [_IMPORTED_ROOT] or set(actual_links) != link_ids or len(bodies) != len(link_ids):
        raise RuntimeError("scene_configuration_native_import_link_set_invalid")
    collisions = [prim for prim in prims if prim.HasAPI(usd_physics.CollisionAPI)]
    collision_paths = {str(prim.GetPath()) for prim in collisions}
    link_readback: dict[str, Any] = {}
    for link_id in sorted(link_ids):
        prim = actual_links[link_id]
        expected_link = expected_links[link_id]
        path = str(prim.GetPath())
        if path != _imported_path(expected_link["prim_path"]):
            raise RuntimeError("scene_configuration_native_import_link_path_mismatch")
        rigid = usd_physics.RigidBodyAPI(prim)
        if rigid.GetRigidBodyEnabledAttr().Get() is False or rigid.GetKinematicEnabledAttr().Get() is True:
            raise RuntimeError("scene_configuration_native_import_link_not_dynamic")
        if not prim.HasAPI(usd_physics.MassAPI):
            raise RuntimeError("scene_configuration_native_import_link_mass_missing")
        mass_api = usd_physics.MassAPI(prim)
        mass = float(mass_api.GetMassAttr().Get())
        center = _finite_vector(mass_api.GetCenterOfMassAttr().Get(), 3)
        inertia = _finite_vector(mass_api.GetDiagonalInertiaAttr().Get(), 3, positive=True)
        if not math.isfinite(mass) or mass <= 0.0 or not math.isclose(
            mass, float(expected_link["mass_kg"]), rel_tol=1e-5, abs_tol=1e-7
        ) or any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-7) for a, b in zip(
            center + inertia,
            _finite_vector(expected_link["center_of_mass_m"], 3)
            + _finite_vector(expected_link["diagonal_inertia_kg_m2"], 3, positive=True),
            strict=True,
        )):
            raise RuntimeError("scene_configuration_native_import_link_physics_mismatch")
        if static_receipt.get("backend") == "artisan_v1":
            from .task_object_artisan_exports import _owner
            own = sorted(str(c.GetPath()) for c in collisions if _owner(c, usd_physics) == prim)
        else:
            own = sorted(path for path in collision_paths if path.startswith(str(prim.GetPath()) + "/"))
        declared = sorted(_imported_path(path) for path in expected_link["collision_prim_paths"])
        if not own or own != declared:
            raise RuntimeError("scene_configuration_native_import_link_collision_mismatch")
        link_readback[link_id] = {
            "prim_path": path,
            "mass_kg": mass,
            "center_of_mass_m": center,
            "diagonal_inertia_kg_m2": inertia,
            "collision_prim_paths": own,
        }
        if static_receipt.get("backend") == "artisan_v1":
            from pxr import UsdShade
            principal = mass_api.GetPrincipalAxesAttr().Get()
            axes = [float(principal.GetReal()), *map(float, principal.GetImaginary())]
            if any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(axes, expected_link["principal_axes_wxyz"], strict=True)):
                raise RuntimeError("scene_configuration_native_import_principal_axes_mismatch")
            materials = {}
            for path in own:
                material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial(materialPurpose="physics")
                if not material or not material.GetPrim().HasAPI(usd_physics.MaterialAPI):
                    raise RuntimeError("scene_configuration_native_import_material_unbound")
                api = usd_physics.MaterialAPI(material.GetPrim())
                materials[str(material.GetPath())] = {"path": str(material.GetPath()),
                    "static_friction": float(api.GetStaticFrictionAttr().Get()),
                    "dynamic_friction": float(api.GetDynamicFrictionAttr().Get()),
                    "restitution": float(api.GetRestitutionAttr().Get())}
            expected_materials = {_imported_path(row["path"]): row for row in expected_link["physics_materials"]}
            if set(materials) != set(expected_materials) or any(not math.isclose(materials[path][name], expected_materials[path][name], abs_tol=1e-6)
                    for path in materials for name in ("static_friction", "dynamic_friction", "restitution")):
                raise RuntimeError("scene_configuration_native_import_material_mismatch")
            link_readback[link_id].update(principal_axes_wxyz=axes, physics_materials=list(materials.values()))
    joints = [prim for prim in prims if prim.IsA(usd_physics.Joint)]
    expected_joint_paths = sorted(_imported_path(path) for path in expected["joint_prim_paths"])
    if (sorted(str(prim.GetPath()) for prim in joints) != expected_joint_paths
            or len(joints) != mechanism["fixed_joint_count"] + 1 + len(mechanism["passive_joint_ids"])):
        raise RuntimeError("scene_configuration_native_import_joint_set_invalid")
    task = _articulated_joint_observation(stage=stage, usd_physics=usd_physics, root_path=_IMPORTED_ROOT)
    declared_task = static_receipt["task_joint"]
    task_path = task["task_joint_prim_path"]
    # Limits read back in SI: metres for a slide, radians for a hinge.
    if (task["task_joint_type"] != mechanism["task_joint_type"]
            or task_path != _imported_path(declared_task["prim_path"])
            or task["task_joint_name"] != declared_task["joint_id"]
            or any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-7) for a, b in zip(
                task["task_joint_limits"], _finite_vector(declared_task["limits"], 2), strict=True))
            or not math.isclose(task["task_joint_reset_position"], float(declared_task["reset_position"]), abs_tol=1e-7)
            or len(task["fixed_joint_prim_paths"]) != mechanism["fixed_joint_count"]):
        raise RuntimeError("scene_configuration_native_import_task_joint_mismatch")
    task_prim = stage.GetPrimAtPath(task_path)
    typed_joint = (usd_physics.RevoluteJoint if mechanism["task_joint_type"] == "revolute"
                   else usd_physics.PrismaticJoint)
    axis = str(typed_joint(task_prim).GetAxisAttr().Get())
    graph_target = next(row for row in static_receipt["articulation_graph"]["joints"] if row["role"] == "target")
    graph_axis = _finite_vector(graph_target["axis"], 3)
    authored_axis = task_prim.GetCustomDataByKey("blueprint:graphAxis")
    if (axis != mechanism["task_joint_usd_axis"] or authored_axis is None
            or any(not math.isclose(float(authored_axis[i]), graph_axis[i], abs_tol=1e-6) for i in range(3))):
        raise RuntimeError("scene_configuration_native_import_task_axis_mismatch")
    if static_receipt.get("backend") == "artisan_v1":
        from pxr import Gf, UsdGeom, Usd
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        joint = usd_physics.Joint(task_prim)
        parent = list(joint.GetBody0Rel().GetTargets())
        if len(parent) != 1:
            raise RuntimeError("scene_configuration_native_import_task_joint_body_invalid")
        matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent[0])) * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(_IMPORTED_ROOT)).GetInverse()
        direction = Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get())).TransformDir(Gf.Vec3d(*{
            "X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[axis]))
        actual_axis = list(matrix.TransformDir(direction))
        if any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(actual_axis, graph_axis, strict=True)):
            raise RuntimeError("scene_configuration_native_import_physical_joint_axis_mismatch")
    if graph_target["child_link_id"] not in link_ids:
        raise RuntimeError("scene_configuration_native_import_task_joint_body_invalid")
    moving_link_path = link_readback[graph_target["child_link_id"]]["prim_path"]
    root_link_path = link_readback[mechanism["root_link_id"]]["prim_path"]
    task_joint = usd_physics.Joint(task_prim)
    parent_link_path = link_readback[graph_target["parent_link_id"]]["prim_path"]
    if ([str(path) for path in task_joint.GetBody0Rel().GetTargets()] != [parent_link_path]
            or [str(path) for path in task_joint.GetBody1Rel().GetTargets()] != [moving_link_path]):
        raise RuntimeError("scene_configuration_native_import_task_joint_body_invalid")
    for row in static_receipt["articulation_graph"]["joints"]:
        if row["role"] == "target":
            continue
        if row["role"] == "passive":
            if _imported_path("/Asset/joints/" + row["joint_id"]) not in task["passive_joint_prim_paths"]:
                raise RuntimeError("scene_configuration_native_import_passive_joint_invalid")
            continue
        fixed_path = _imported_path("/Asset/joints/" + row["joint_id"])
        fixed_prim = stage.GetPrimAtPath(fixed_path)
        if not fixed_prim.IsA(usd_physics.FixedJoint):
            raise RuntimeError("scene_configuration_native_import_fixed_joint_invalid")
        fixed = usd_physics.Joint(fixed_prim)
        if ([str(path) for path in fixed.GetBody0Rel().GetTargets()] != [link_readback[row["parent_link_id"]]["prim_path"]]
                or [str(path) for path in fixed.GetBody1Rel().GetTargets()] != [link_readback[row["child_link_id"]]["prim_path"]]):
            raise RuntimeError("scene_configuration_native_import_fixed_joint_body_invalid")
    task["moving_link_prim_path"] = moving_link_path
    handle_paths = sorted(_imported_path(path) for path in static_receipt["task_contact"]["handle_prim_paths"])
    # A graspable feature may be its own rigid part bolted to the moving link.
    # Use the same fixed assembly the static authoring contract admitted;
    # crossing another movable joint must never admit an unrelated handle.
    moving_assembly = {graph_target["child_link_id"]}
    if static_receipt.get("backend") == "artisan_v1":
        from .task_object_artisan_exports import _fixed_descendants
        moving_assembly = _fixed_descendants(graph_target["child_link_id"], static_receipt["articulation_graph"]["joints"])
    moving_collisions = {path for link_id in moving_assembly for path in link_readback[link_id]["collision_prim_paths"]}
    if not handle_paths or any(path not in moving_collisions for path in handle_paths):
        raise RuntimeError("scene_configuration_native_import_handle_mismatch")
    task["task_joint_axis"] = axis
    return {
        "articulation_root_paths": roots,
        "rigid_body_paths": sorted(str(prim.GetPath()) for prim in bodies),
        "collision_paths": sorted(collision_paths),
        "fixed_joint_prim_paths": task["fixed_joint_prim_paths"],
        "task_joint": task,
        "link_physics_readback": link_readback,
        "handle_prim_paths": handle_paths,
        "settle_measured_body_prim_path": root_link_path,
    }


def _validated_articulated_static_receipt(
    *, path: Path, asset_record: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _read(path, code="scene_configuration_native_import_static_receipt_invalid")
    if not isinstance(receipt, Mapping):
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            "scene_configuration_native_import_static_receipt_invalid"
        )
    asset = receipt.get("replacement_usd") or {}
    if (
        receipt.get("schema_version")
        != "task_evaluation_articulated_replacement_static_qualification.v1"
        or receipt.get("status") != "authored_structure_statically_qualified"
        or receipt.get("asset_kind") != "articulated_assembly"
        or receipt.get("replacement_identity") != identity
        or receipt.get("structural_findings") != []
        or receipt.get("result_digest") != canonical_digest(receipt, digest_field="result_digest")
        or not isinstance(asset, Mapping)
        or asset.get("sha256") != asset_record["digest"]
        or asset.get("size_bytes") != asset_record["size_bytes"]
        or not isinstance(receipt.get("observed_structure"), Mapping)
        or not isinstance(receipt.get("articulation_graph"), Mapping)
        or not isinstance(receipt.get("task_contact"), Mapping)
        or not isinstance(receipt.get("task_joint"), Mapping)
    ):
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            "scene_configuration_native_import_static_receipt_invalid"
        )
    try:
        _receipt_mechanism(receipt)
    except RuntimeError as exc:
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            "scene_configuration_native_import_static_receipt_invalid"
        ) from exc
    return dict(receipt)


def _articulated_repeat_matches_static(
    row: Mapping[str, Any], receipt: Mapping[str, Any]
) -> bool:
    """Refuse a runner observation that diverges from the sealed imported parts."""

    try:
        expected_links = receipt["observed_structure"]["links"]
        observed_links = row["link_physics_readback"]
        if set(observed_links) != set(expected_links):
            return False
        for link_id, expected in expected_links.items():
            observed = observed_links[link_id]
            if (observed["prim_path"] != _imported_path(expected["prim_path"])
                    or sorted(observed["collision_prim_paths"])
                    != sorted(_imported_path(path) for path in expected["collision_prim_paths"])):
                return False
            numbers = (
                [float(observed["mass_kg"])]
                + _finite_vector(observed["center_of_mass_m"], 3)
                + _finite_vector(observed["diagonal_inertia_kg_m2"], 3, positive=True)
            )
            declared = (
                [float(expected["mass_kg"])]
                + _finite_vector(expected["center_of_mass_m"], 3)
                + _finite_vector(expected["diagonal_inertia_kg_m2"], 3, positive=True)
            )
            if any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-7)
                   for a, b in zip(numbers, declared, strict=True)):
                return False
        joint = row["task_joint"]
        task = receipt["task_joint"]
        passive_ids = _receipt_mechanism(receipt)["passive_joint_ids"]
        expected_fixed_paths = sorted(
            _imported_path(path)
            for path in receipt["observed_structure"]["joint_prim_paths"]
            if path != task["prim_path"] and PurePosixPath(path).name not in passive_ids
        )
        if (joint["task_joint_name"] != task["joint_id"]
                or joint["task_joint_reset_position"] != task["reset_position"]
                or any(not math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=1e-7)
                       for a, b in zip(joint["task_joint_limits"], task["limits"], strict=True))
                or len(expected_fixed_paths) != _receipt_mechanism(receipt)["fixed_joint_count"]
                or sorted(joint["fixed_joint_prim_paths"]) != expected_fixed_paths
                or sorted(row["fixed_joint_prim_paths"]) != expected_fixed_paths):
            return False
        return True
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        return False


def _reset_tolerance(receipt: Mapping[str, Any]) -> float:
    """Metres for a slide (unchanged), the graph's radians for a hinge."""
    mechanism = _receipt_mechanism(receipt)
    return (TASK_JOINT_RESET_TOLERANCE if mechanism["task_joint_type"] == "prismatic"
            else mechanism["reset_tolerance"])


def _live_joint_position(articulation: Any, joint_name: str) -> float:
    """Read one DOF from Isaac's initialized articulation handle."""

    names = [str(name) for name in (articulation.dof_names or [])]
    if names.count(joint_name) != 1:
        raise RuntimeError("scene_configuration_native_import_joint_dof_unresolved")
    index = int(articulation.get_dof_index(joint_name))
    positions = articulation.get_joint_positions()
    if positions is None or index < 0 or index >= len(positions):
        raise RuntimeError("scene_configuration_native_import_joint_readback_unavailable")
    coordinate = float(positions[index])
    if not math.isfinite(coordinate):
        raise RuntimeError("scene_configuration_native_import_joint_readback_nonfinite")
    return coordinate


def _physics_step(simulation: Any) -> None:
    try:
        simulation.step(render=False)
    except TypeError:
        simulation.step()


def robot_free_probe_plan(static_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Where the robot-free probe drives the task joint: the middle of its goal interval."""

    graph = static_receipt["articulation_graph"]
    target = next(row for row in graph["joints"] if row["role"] == "target")
    lower, upper = (float(value) for value in graph["success_predicate"]["joint_intervals"][target["joint_id"]])
    return {"target_joint_id": str(target["joint_id"]), "reset_position": float(target["reset_position"]),
            "success_interval": [lower, upper], "drive_target_position": (lower + upper) / 2.0}


def _dof_sample(articulation: Any, names: list[str], *, step_index: int, phase: str) -> dict[str, Any]:
    positions, velocities = articulation.get_joint_positions(), articulation.get_joint_velocities()
    if positions is None or velocities is None or len(positions) != len(names) or len(velocities) != len(names):
        raise RuntimeError("robot_free_probe_readback_unavailable")
    values = {name: [round(float(positions[i]), 7), round(float(velocities[i]), 7)] for i, name in enumerate(names)}
    if not all(math.isfinite(value) for pair in values.values() for value in pair):
        raise RuntimeError("robot_free_probe_readback_nonfinite")
    return {"step_index": step_index, "phase": phase, "dof": values}


def _robot_free_drive_and_release(
    *, simulation: Any, articulation: Any, joint_name: str, plan: Mapping[str, Any],
    timestep_seconds: float, first_step_index: int,
) -> list[dict[str, Any]]:
    """Ramp the task joint into its goal interval, then let go; physics alone decides where it rests.

    No robot is involved. The ramp phase is recorded as contact, so only the
    released phase can ever count as success when the trace is scored.
    """

    import numpy as np  # type: ignore

    names = [str(name) for name in (articulation.dof_names or [])]
    index = np.array([names.index(joint_name)])
    start = _live_joint_position(articulation, joint_name)
    drive_steps = max(1, int(round(_PROBE_DRIVE_SECONDS / timestep_seconds)))
    release_steps = max(1, int(round(_PROBE_RELEASE_SECONDS / timestep_seconds)))
    samples: list[dict[str, Any]] = []
    for step in range(drive_steps + release_steps):
        if step < drive_steps:
            target = start + (float(plan["drive_target_position"]) - start) * (step + 1) / drive_steps
            articulation.set_joint_positions(np.array([target]), joint_indices=index)
            articulation.set_joint_velocities(np.array([0.0]), joint_indices=index)
        _physics_step(simulation)
        samples.append(_dof_sample(articulation, names, step_index=first_step_index + step,
                                   phase="drive" if step < drive_steps else "release"))
    return samples


def record_robot_free_probe(*, settle: Callable[..., Mapping[str, Any]], asset_path: Path,
                            duration_seconds: float, static_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """One extra robot-free run after the qualifying settles; a failure is recorded, never raised."""

    try:
        observed = settle(asset_path=asset_path, duration_seconds=duration_seconds,
                          timestep_seconds=1.0 / _PROBE_PHYSICS_HZ,
                          articulated=True, static_receipt=static_receipt,
                          robot_free_probe=robot_free_probe_plan(static_receipt))
        probe = dict(observed["robot_free_success_probe"])
        if probe.get("schema_version") != ROBOT_FREE_PROBE_SCHEMA or probe.get("status") != "recorded":
            raise RuntimeError("robot_free_probe_observation_invalid")
        return probe
    except Exception as exc:  # noqa: BLE001 - the probe never decides import qualification
        return {"schema_version": ROBOT_FREE_PROBE_SCHEMA, "status": "failed",
                "error": f"{type(exc).__name__}:{str(exc)[:200]}"}


def _subscribe_body_contact_reports(
    *,
    omni_physx: Any,
    physics_schema_tools: Any,
    body_path: str,
    support_path: str | None = None,
    event_count: list[int],
) -> Any:
    """Count body contacts through PhysX's supported simulation callback."""

    def _on_contact_report(contact_headers: Any, _contact_data: Any) -> None:
        try:
            for header in contact_headers:
                paths: list[str] = []
                for name in ("actor0", "actor1", "collider0", "collider1"):
                    encoded = getattr(header, name, 0)
                    try:
                        paths.append(str(physics_schema_tools.intToSdfPath(int(encoded))))
                    except (TypeError, ValueError):
                        paths.append(str(encoded))
                body_contact = body_path in paths or any(path.startswith(body_path + "/") for path in paths)
                support_contact = support_path is None or support_path in paths or any(
                    path.startswith(support_path + "/") for path in paths
                )
                if body_contact and support_contact:
                    event_count[0] += 1
        except Exception:  # noqa: BLE001 - absence of proof fails qualification
            return

    interface = omni_physx.get_physx_simulation_interface()
    return interface.subscribe_contact_report_events(_on_contact_report)


def _articulated_joint_observation(
    *, stage: Any, usd_physics: Any, root_path: str
) -> dict[str, Any]:
    """Read the one task joint from the staged bytes, before physics runs.

    The qualifier that admitted this asset proved there is exactly one movable
    joint with finite limits and a closed reset. Re-derive that from the
    imported stage so the native receipt names the joint the runtime will read,
    and refuse a drive that could open the part without the robot.
    """

    joints = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(root_path)
        and prim.IsA(usd_physics.Joint)
    ]
    movable = [prim for prim in joints if not prim.IsA(usd_physics.FixedJoint)]
    targets = [prim for prim in movable if prim.GetCustomDataByKey("blueprint:jointRole") in {None, "target"}]
    passive = [prim for prim in movable if prim.GetCustomDataByKey("blueprint:jointRole") == "passive"]
    if len(targets) != 1 or len(targets) + len(passive) != len(movable):
        raise RuntimeError("scene_configuration_native_import_single_task_joint_required")
    for prim in passive:
        # A passive joint may move with the robot, never by itself.
        typed_passive = (usd_physics.PrismaticJoint(prim) if prim.IsA(usd_physics.PrismaticJoint)
                         else usd_physics.RevoluteJoint(prim) if prim.IsA(usd_physics.RevoluteJoint) else None)
        bounds = (typed_passive.GetLowerLimitAttr().Get(), typed_passive.GetUpperLimitAttr().Get()) if typed_passive else ()
        if (len(bounds) != 2 or any(value is None or not math.isfinite(float(value)) for value in bounds)
                or float(bounds[0]) >= float(bounds[1])
                or any(prim.HasAPI(usd_physics.DriveAPI, name)
                       and float(usd_physics.DriveAPI(prim, name).GetStiffnessAttr().Get() or 0.0) > 0.0
                       for name in ("linear", "angular"))):
            raise RuntimeError("scene_configuration_native_import_passive_joint_invalid")
    joint = targets[0]
    if joint.IsA(usd_physics.PrismaticJoint):
        typed, joint_type = usd_physics.PrismaticJoint(joint), "prismatic"
    elif joint.IsA(usd_physics.RevoluteJoint):
        typed, joint_type = usd_physics.RevoluteJoint(joint), "revolute"
    else:
        raise RuntimeError("scene_configuration_native_import_task_joint_type_unsupported")
    lower, upper = typed.GetLowerLimitAttr().Get(), typed.GetUpperLimitAttr().Get()
    if joint_type == "revolute" and lower is not None and upper is not None:
        lower, upper = math.radians(float(lower)), math.radians(float(upper))
    if (
        lower is None
        or upper is None
        or not math.isfinite(float(lower))
        or not math.isfinite(float(upper))
        or float(lower) >= float(upper)
    ):
        raise RuntimeError("scene_configuration_native_import_task_joint_limits_invalid")
    for name in ("linear", "angular"):
        if joint.HasAPI(usd_physics.DriveAPI, name):
            stiffness = usd_physics.DriveAPI(joint, name).GetStiffnessAttr().Get()
            if stiffness is not None and float(stiffness) > 0.0:
                raise RuntimeError(
                    "scene_configuration_native_import_task_joint_drive_forbidden"
                )
    reset = joint.GetCustomDataByKey("blueprint:resetPosition")
    if (
        not isinstance(reset, (int, float))
        or isinstance(reset, bool)
        or not float(lower) <= float(reset) <= float(upper)
    ):
        raise RuntimeError("scene_configuration_native_import_task_joint_reset_invalid")
    return {
        "task_joint_prim_path": str(joint.GetPath()),
        "task_joint_name": PurePosixPath(str(joint.GetPath())).name,
        "task_joint_type": joint_type,
        "task_joint_limits": [float(lower), float(upper)],
        "task_joint_reset_position": float(reset),
        "fixed_joint_prim_paths": sorted(
            str(prim.GetPath()) for prim in joints if prim.IsA(usd_physics.FixedJoint)
        ),
        "passive_joint_prim_paths": sorted(str(prim.GetPath()) for prim in passive),
        "task_joint_drive_forbidden_verified": True,
    }


def _one_native_settle(
    *,
    asset_path: Path,
    duration_seconds: float,
    timestep_seconds: float,
    articulated: bool = False,
    static_receipt: Mapping[str, Any] | None = None,
    robot_free_probe: Mapping[str, Any] | None = None,
    required_motion_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import omni.physx as omni_physx  # type: ignore
    import omni.usd  # type: ignore
    from isaacsim.core.api import SimulationContext  # type: ignore
    from pxr import (  # type: ignore
        Gf,
        PhysicsSchemaTools,
        PhysxSchema,
        Sdf,
        Usd,
        UsdGeom,
        UsdPhysics,
    )

    clear_instance = getattr(SimulationContext, "clear_instance", None)
    if callable(clear_instance):
        clear_instance()
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("scene_configuration_native_import_stage_creation_failed")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    scene_api.CreateEnableEnhancedDeterminismAttr().Set(True)
    scene_api.CreateEnableGPUDynamicsAttr().Set(False)
    scene_api.CreateBroadphaseTypeAttr().Set("SAP")
    ground = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Ground"))
    ground.CreateSizeAttr(2.0)
    ground_xform = UsdGeom.Xformable(ground.GetPrim())
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))
    ground_xform.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.025))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    placement = UsdGeom.Xform.Define(stage, Sdf.Path("/World/Placement"))
    replacement = stage.DefinePrim("/World/Placement/Replacement", "Xform")
    replacement.GetReferences().AddReference(str(asset_path), "/Asset")
    stage.Load()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    bounds = cache.ComputeWorldBound(replacement).ComputeAlignedRange()
    lower = bounds.GetMin()
    if not all(math.isfinite(float(lower[index])) for index in range(3)):
        raise RuntimeError("scene_configuration_native_import_bounds_invalid")
    placement.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.002 - float(lower[2])))
    rigid_paths = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    collision_paths = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    articulation_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    joint_observation: dict[str, Any] = {}
    structure_observation: dict[str, Any] = {}
    if articulated:
        if static_receipt is None:
            raise RuntimeError("scene_configuration_native_import_static_receipt_missing")
        structure_observation = _articulated_structure_observation(
            stage=stage, usd_physics=UsdPhysics, static_receipt=static_receipt
        )
        joint_observation = structure_observation["task_joint"]
        body_path = structure_observation["settle_measured_body_prim_path"]
    elif len(rigid_paths) != 1 or not collision_paths:
        raise RuntimeError("scene_configuration_native_import_structure_invalid")
    else:
        body_path = rigid_paths[0]
    if required_motion_probe is not None:
        from .articulated_required_motion_probe import DIAGNOSTIC_DRIVE
        from .articulated_native_diagnostic_runtime import _author_fixed_base_anchor, _configure_required_motion_drives
        assert static_receipt is not None
        base_id = _receipt_mechanism(static_receipt)["root_link_id"]
        _author_fixed_base_anchor(stage, _imported_path(static_receipt["link_prim_paths"][base_id]), UsdPhysics)
        _configure_required_motion_drives(stage, required_motion_probe, UsdPhysics, DIAGNOSTIC_DRIVE)
    body = stage.GetPrimAtPath(body_path)
    contact_api = PhysxSchema.PhysxContactReportAPI.Apply(body)
    contact_api.CreateThresholdAttr().Set(0.0)
    contact_event_count = [0]
    contact_subscription = _subscribe_body_contact_reports(
        omni_physx=omni_physx,
        physics_schema_tools=PhysicsSchemaTools,
        body_path=body_path,
        support_path="/World/Ground",
        event_count=contact_event_count,
    )
    simulation = SimulationContext(
        physics_dt=timestep_seconds,
        rendering_dt=timestep_seconds,
        stage_units_in_meters=1.0,
    )
    physics_context = simulation.get_physics_context()
    for name, argument in (
        ("set_solver_type", "TGS"),
        ("set_broadphase_type", "SAP"),
        ("enable_gpu_dynamics", False),
        ("enable_enhanced_determinism", True),
    ):
        method = getattr(physics_context, name, None)
        if callable(method):
            method(argument)
    simulation.initialize_physics()
    simulation.play()
    articulation = None
    if articulated:
        from isaacsim.core.prims import SingleArticulation  # type: ignore

        articulation = SingleArticulation(
            prim_path=_IMPORTED_ROOT, name="blueprint_replacement_native_import"
        )
        articulation.initialize()
        if not bool(getattr(articulation, "handles_initialized", False)):
            raise RuntimeError("scene_configuration_native_import_articulation_handle_invalid")
    initial_position, initial_rotation = _live_pose(omni_physx, body_path)
    initial_joint = (
        _live_joint_position(articulation, joint_observation["task_joint_name"])
        if articulated else None
    )
    trace: list[list[float]] = []
    joint_trace: list[float] = []
    probe_samples: list[dict[str, Any]] = []
    dof_names = [str(name) for name in (articulation.dof_names or [])] if robot_free_probe is not None else []
    step_count = int(math.ceil(duration_seconds / timestep_seconds))
    for _step in range(step_count):
        _physics_step(simulation)
        position, _rotation = _live_pose(omni_physx, body_path)
        trace.append(position)
        if articulated:
            coordinate = _live_joint_position(articulation, joint_observation["task_joint_name"])
            joint_trace.append(coordinate)
        if robot_free_probe is not None:
            probe_samples.append(_dof_sample(articulation, dof_names, step_index=_step, phase="zero_action"))
    final_position, final_rotation = _live_pose(omni_physx, body_path)
    observed_joint = (
        _live_joint_position(articulation, joint_observation["task_joint_name"])
        if articulated else None
    )
    if robot_free_probe is not None:
        probe_samples += _robot_free_drive_and_release(
            simulation=simulation, articulation=articulation, joint_name=joint_observation["task_joint_name"],
            plan=robot_free_probe, timestep_seconds=timestep_seconds, first_step_index=step_count)
    required_motion_result = None
    if required_motion_probe is not None:
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore
        from .articulated_native_diagnostic_runtime import _execute_native_required_motions
        required_motion_result = _execute_native_required_motions(
            required_motion_probe, articulation=articulation, action_type=ArticulationAction,
            simulation=simulation, settle_steps=max(1, int(round(2.0 / timestep_seconds))))
    simulation.stop()
    del contact_subscription
    if callable(clear_instance):
        clear_instance()
    translation = math.dist(initial_position, final_position)
    rotation = _angle_between_xyzw(initial_rotation, final_rotation)
    state = {
        "position_m": [round(value, 7) for value in final_position],
        "orientation_xyzw": [round(value, 7) for value in final_rotation],
    }
    if articulated:
        assert static_receipt is not None
        tolerance = _reset_tolerance(static_receipt)
        # The key keeps its historical name; a revolute coordinate is radians.
        state["task_joint_position_m"] = round(float(observed_joint), 7)
        joint_observation["initial_task_joint_position"] = initial_joint
        joint_observation["settled_task_joint_position"] = observed_joint
        joint_observation["task_joint_returned_to_reset"] = (
            abs(float(initial_joint) - joint_observation["task_joint_reset_position"]) <= tolerance
            and abs(float(observed_joint) - joint_observation["task_joint_reset_position"]) <= tolerance
        )
    return {
        "asset_imported": True,
        "rigid_body_paths": rigid_paths,
        "articulation_root_paths": articulation_paths,
        "settle_measured_body_prim_path": body_path,
        "collision_paths": collision_paths,
        **({**structure_observation, "task_joint": joint_observation} if articulated else {}),
        "support_contact_observed": contact_event_count[0] > 0,
        "contact_report_event_count": contact_event_count[0],
        "settle_translation_m": translation,
        "settle_rotation_rad": rotation,
        "step_count": step_count,
        "final_state": state,
        "final_state_digest": canonical_digest(state),
        "position_trace_digest": canonical_digest(
            {"position_trace_m": [[round(v, 7) for v in row] for row in trace]}
        ),
        **({"required_motion_probe": required_motion_result} if required_motion_result is not None else {}),
        **({"task_joint_trace_digest": canonical_digest(
            {"task_joint_position_m": [round(value, 7) for value in joint_trace]}
        )} if articulated else {}),
        **({"robot_free_success_probe": {
            "schema_version": ROBOT_FREE_PROBE_SCHEMA, "status": "recorded", "plan": dict(robot_free_probe),
            "task_joint_name": joint_observation["task_joint_name"], "timestep_seconds": timestep_seconds,
            "dof_names": dof_names, "samples": probe_samples,
            "proof_boundary": {"robot_embodiment_used": False, "scene_context": "replacement_asset_on_ground_plane",
                               "drive": "kinematic_joint_ramp_then_release", "scored_here": False},
        }} if robot_free_probe is not None else {}),
    }


def _run_native_import(
    *,
    asset_path: Path,
    required_checks: Mapping[str, Any],
    observation_consumer: NativeObservationConsumer,
    destination_asset_path: Path | None = None,
    articulated: bool = False,
    static_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Isaac allows one SimulationApp per process, so a supplemental destination
    # is settled inside the same session right after the subject replacement.
    app = None
    exit_code = 1
    try:
        _bind_isaac_runtime_environment()
        SimulationApp = _import_simulation_app()
        print("BLUEPRINT_NATIVE_IMPORT_SIMULATION_APP_STARTING", flush=True)
        app = SimulationApp({"headless": True, "fast_shutdown": True})
        print("BLUEPRINT_NATIVE_IMPORT_SIMULATION_APP_READY", flush=True)
        runtime_identity = _observe_isaac_runtime_identity(app)
        repeat_count = int(required_checks["state_digest_repeat_count"])
        duration = float(required_checks["gravity_settle_seconds"])
        repeats = [
            _one_native_settle(
                asset_path=asset_path,
                duration_seconds=duration,
                timestep_seconds=1.0 / 60.0,
                articulated=articulated,
                static_receipt=static_receipt,
            )
            for _ in range(repeat_count)
        ]
        observation: dict[str, Any] = {
            "runtime_identity": runtime_identity,
            "repeats": repeats,
        }
        if destination_asset_path is not None:
            observation["destination_repeats"] = [
                _one_native_settle(
                    asset_path=destination_asset_path,
                    duration_seconds=duration,
                    timestep_seconds=1.0 / 60.0,
                )
                for _ in range(repeat_count)
            ]
        if articulated:
            assert static_receipt is not None
            observation["robot_free_success_probe"] = record_robot_free_probe(
                settle=_one_native_settle, asset_path=asset_path, duration_seconds=duration,
                static_receipt=static_receipt)
            from .articulated_required_motion_probe import required_motion_plan_from_static_receipt
            required_plan = required_motion_plan_from_static_receipt(static_receipt)
            if required_plan is not None:
                extra = _one_native_settle(asset_path=asset_path, duration_seconds=duration,
                    timestep_seconds=1.0 / 60.0, articulated=True, static_receipt=static_receipt,
                    required_motion_probe=required_plan)
                observation["required_motion_probe"] = extra["required_motion_probe"]
        result = dict(observation_consumer(observation))
        exit_code = 0
        return result
    except BaseException as error:
        # Fast Kit shutdown can exit Python with code 0 before an exception
        # reaches the process entrypoint. Retain its real predicate first.
        if isinstance(error, SystemExit) and isinstance(error.code, int) and error.code != 0:
            exit_code = error.code
        elif isinstance(error, KeyboardInterrupt):
            exit_code = 130
        traceback.print_exc()
        raise
    finally:
        if app is not None:
            # Isaac 6.0.1 can otherwise replace a validation failure with a
            # successful process exit during fast shutdown.
            if "exit_code" in inspect.signature(app.close).parameters:
                app.close(exit_code=exit_code)
            else:
                app.close()


def execute_native_import_component(
    *,
    environment: Mapping[str, str] | None = None,
    native_runner: NativeRunner = _run_native_import,
) -> dict[str, Any]:
    """Execute and seal three deterministic native import/reset observations."""

    values = dict(os.environ if environment is None else environment)
    stage_input_path = _required_path(values, _STAGE_INPUT_ENV)
    dependencies_path = _required_path(values, _DEPENDENCIES_ENV)
    output_root = _required_path(values, _OUTPUT_ENV)
    component_result_path = _required_path(values, _COMPONENT_RESULT_ENV)
    production_input = _read(
        stage_input_path, code="scene_configuration_native_import_input_invalid"
    )
    dependencies = _read(
        dependencies_path,
        code="scene_configuration_native_import_dependencies_invalid",
    )
    stage = production_input.get("stage") if isinstance(production_input, Mapping) else None
    configuration = (
        production_input.get("configuration") if isinstance(production_input, Mapping) else None
    )
    checks = configuration.get("required_checks") if isinstance(configuration, Mapping) else None
    if (
        not isinstance(production_input, Mapping)
        or production_input.get("schema_version") != _STAGE_INPUT_SCHEMA
        or not isinstance(stage, Mapping)
        or stage.get("adapter", {}).get("id") != ADAPTER_ID
        or not isinstance(configuration, Mapping)
        or configuration.get("schema_version")
        != "replacement_native_import_qualification_configuration.v1"
        or not native_import_checks_valid(checks)
        or component_result_path.exists()
        or component_result_path.parent != output_root
    ):
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            "scene_configuration_native_import_input_invalid"
        )
    # An articulated assembly declares its kind on the stage-5 configuration
    # that stage_five_configuration emitted; the rigid path is unchanged.
    articulated = configuration.get("asset_kind") == "articulated_assembly"
    asset_record, asset_path = _artifact(
        dependencies, role="statically_qualified_replacement_asset"
    )
    static_record, static_path = _artifact(dependencies, role="static_qualification_receipt")
    static_receipt = (
        _validated_articulated_static_receipt(
            path=static_path,
            asset_record=asset_record,
            identity=configuration["replacement_identity"],
        )
        if articulated else None
    )
    destination = _declared_supplemental_destination(production_input)
    destination_asset_record: Mapping[str, Any] | None = None
    destination_static_record: Mapping[str, Any] | None = None
    destination_asset_path: Path | None = None
    if destination is not None:
        try:
            destination_asset_record, destination_asset_path = _artifact(
                dependencies, role=DESTINATION_ASSET_DEPENDENCY_ROLE
            )
            destination_static_record, _destination_static_path = _artifact(
                dependencies, role=DESTINATION_STATIC_DEPENDENCY_ROLE
            )
        except TaskEvaluationSceneConfigurationNativeImportDriverError as exc:
            raise TaskEvaluationSceneConfigurationNativeImportDriverError(
                "scene_configuration_native_import_destination_dependency_invalid"
            ) from exc
    elif any(
        artifact.get("role") in {DESTINATION_ASSET_DEPENDENCY_ROLE, DESTINATION_STATIC_DEPENDENCY_ROLE}
        for result in dependencies
        if isinstance(result, Mapping)
        for artifact in result.get("output_artifacts") or []
        if isinstance(artifact, Mapping)
    ):
        raise TaskEvaluationSceneConfigurationNativeImportDriverError(
            "scene_configuration_native_import_destination_dependency_invalid"
        )

    def _qualified_repeats(
        repeats: Any, *, code: str, articulated: bool = False
    ) -> tuple[list[str], float, float]:
        if not isinstance(repeats, list) or len(repeats) != 3:
            raise TaskEvaluationSceneConfigurationNativeImportDriverError(code)
        state_digests = [str(row.get("final_state_digest") or "") for row in repeats]
        translations = [float(row.get("settle_translation_m", math.inf)) for row in repeats]
        rotations = [float(row.get("settle_rotation_rad", math.inf)) for row in repeats]
        maximum_translation = max(translations)
        maximum_rotation = max(rotations)
        if articulated:
            assert static_receipt is not None
            expected_links = static_receipt["observed_structure"]["links"]
            expected_paths = {
                link_id: _imported_path(row["prim_path"])
                for link_id, row in expected_links.items()
            }
            expected_task = static_receipt["task_joint"]
            mechanism = _receipt_mechanism(static_receipt)
            tolerance = _reset_tolerance(static_receipt)
            expected_handle = sorted(
                _imported_path(path)
                for path in static_receipt["task_contact"]["handle_prim_paths"]
            )
            joints = [row.get("task_joint") for row in repeats]
            links = mechanism["links"]
            structure_ok = (
                set(expected_links) == links
                and all(set(row.get("rigid_body_paths") or []) == set(expected_paths.values()) for row in repeats)
                and all(row.get("articulation_root_paths") == [_IMPORTED_ROOT] for row in repeats)
                and all(row.get("settle_measured_body_prim_path") == expected_paths[mechanism["root_link_id"]]
                        for row in repeats)
                and all(row.get("handle_prim_paths") == expected_handle for row in repeats)
                and all(len(row.get("fixed_joint_prim_paths") or []) == mechanism["fixed_joint_count"]
                        for row in repeats)
                and all(set((row.get("link_physics_readback") or {})) == links for row in repeats)
                and all(_articulated_repeat_matches_static(row, static_receipt) for row in repeats)
                and all(isinstance(joint, Mapping) for joint in joints)
                and all((joint or {}).get("task_joint_prim_path") == _imported_path(expected_task["prim_path"]) for joint in joints)
                and all((joint or {}).get("task_joint_type") == mechanism["task_joint_type"] for joint in joints)
                and all((joint or {}).get("task_joint_axis") == mechanism["task_joint_usd_axis"] for joint in joints)
                and all((joint or {}).get("moving_link_prim_path")
                        == expected_paths[mechanism["task_child_link_id"]] for joint in joints)
                and all(
                    (joint or {}).get("task_joint_drive_forbidden_verified") is True
                    for joint in joints
                )
                and all((joint or {}).get("task_joint_returned_to_reset") is True for joint in joints)
                and all(
                    len((joint or {}).get("task_joint_limits") or []) == 2
                    and float((joint or {}).get("task_joint_limits")[0])
                    < float((joint or {}).get("task_joint_limits")[1])
                    for joint in joints
                )
                and all(
                    isinstance((joint or {}).get("settled_task_joint_position"), (int, float))
                    and not isinstance((joint or {}).get("settled_task_joint_position"), bool)
                    and math.isfinite(float((joint or {}).get("settled_task_joint_position")))
                    and abs(float((joint or {}).get("settled_task_joint_position")) - float(expected_task["reset_position"]))
                    <= tolerance
                    and abs(float((joint or {}).get("initial_task_joint_position")) - float(expected_task["reset_position"]))
                    <= tolerance
                    for joint in joints
                )
                and all(
                    isinstance(row.get("final_state"), Mapping)
                    and row["final_state"].get("task_joint_position_m") == round(float(row["task_joint"]["settled_task_joint_position"]), 7)
                    and row.get("final_state_digest") == canonical_digest(row["final_state"])
                    and isinstance(row.get("task_joint_trace_digest"), str)
                    and row["task_joint_trace_digest"].startswith("sha256:")
                    for row in repeats
                )
            )
        else:
            structure_ok = all(len(row.get("rigid_body_paths") or []) == 1 for row in repeats)
        predicates = {
            "deterministic_reset": len(set(state_digests)) == 1,
            "asset_imported": all(row.get("asset_imported") is True for row in repeats),
            "support_contact": all(row.get("support_contact_observed") is True for row in repeats),
            "structure_matches_static": structure_ok,
            "colliders_present": all(bool(row.get("collision_paths")) for row in repeats),
            "settle_translation": all(math.isfinite(value) for value in translations)
                and maximum_translation <= float(checks["maximum_settle_translation_m"]),
            "settle_rotation": all(math.isfinite(value) for value in rotations)
                and maximum_rotation <= float(checks["maximum_settle_rotation_rad"]),
        }
        failed = [name for name, passed in predicates.items() if not passed]
        if failed:
            raise TaskEvaluationSceneConfigurationNativeImportDriverError(
                "scene_configuration_native_import_qualification_failed:" + ",".join(failed)
            )
        return state_digests, maximum_translation, maximum_rotation

    def _runtime_result(
        *,
        articulated_result: bool,
        identity: Mapping[str, Any],
        asset_digest: str,
        static_digest: str,
        runtime_identity: Mapping[str, Any],
        repeats: list[Any],
        state_digests: list[str],
        maximum_translation: float,
        maximum_rotation: float,
        robot_free_probe: Any = None,
        required_motion_probe: Any = None,
    ) -> dict[str, Any]:
        runtime_result: dict[str, Any] = {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "qualified",
            "replacement_identity": dict(identity),
            "asset_digest": asset_digest,
            "static_qualification_digest": static_digest,
            "native_isaac_executed": True,
            "native_simulator_import_qualified": True,
            "support_contact_observed": True,
            "deterministic_reset_state_digest_repeat_count": 3,
            "deterministic_reset_state_digest": state_digests[0],
            "maximum_observed_settle_translation_m": maximum_translation,
            "maximum_observed_settle_rotation_rad": maximum_rotation,
            "qualification_limits": {
                "gravity_settle_seconds": float(checks["gravity_settle_seconds"]),
                "maximum_settle_translation_m": float(
                    checks["maximum_settle_translation_m"]
                ),
                "maximum_settle_rotation_rad": float(
                    checks["maximum_settle_rotation_rad"]
                ),
                "state_digest_repeat_count": int(
                    checks["state_digest_repeat_count"]
                ),
            },
            "runtime_identity": dict(runtime_identity),
            "repeats": repeats,
            "physical_equivalence_claimed": False,
            "evaluation_episode_executed": False,
            "blockers": [],
            "result_digest": "",
        }
        if articulated_result and repeats and isinstance(repeats[0], Mapping):
            joint = dict(repeats[0].get("task_joint") or {})
            runtime_result.update(
                asset_kind="articulated_assembly",
                link_physics_readback=repeats[0]["link_physics_readback"],
                handle_prim_paths=repeats[0]["handle_prim_paths"],
                task_joint_reset_numeric_readbacks=[
                    row["task_joint"]["settled_task_joint_position"] for row in repeats
                ],
                task_joint_readback={
                    key: joint.get(key)
                    for key in (
                        "task_joint_prim_path", "task_joint_name", "task_joint_type",
                        "task_joint_limits", "task_joint_reset_position",
                        "moving_link_prim_path", "fixed_joint_prim_paths", "task_joint_axis",
                    )
                },
                task_joint_drive_forbidden_verified=True,
                task_joint_closed_at_reset_verified=True,
                task_joint_travel_is_measured=False,
            )
            if joint.get("task_joint_type") == "revolute":
                runtime_result["task_joint_coordinate_units"] = "rad"
            # Unscored raw evidence for the robot-free readiness checks; it is
            # never part of import qualification.
            if isinstance(robot_free_probe, Mapping) and robot_free_probe.get("schema_version") == ROBOT_FREE_PROBE_SCHEMA:
                runtime_result["robot_free_success_probe"] = dict(robot_free_probe)
            if required_motion_probe is not None:
                runtime_result["required_motion_probe"] = dict(required_motion_probe)
        runtime_result["result_digest"] = canonical_digest(
            runtime_result, digest_field="result_digest"
        )
        return runtime_result

    def _write_artifact(*, role: str, filename: str, value: Mapping[str, Any]) -> dict[str, Any]:
        artifact_path = output_root / filename
        artifact_path.write_text(canonical_json(value) + "\n", encoding="utf-8")
        return {
            "role": role,
            "path": str(artifact_path),
            "digest": _sha256(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        }

    def _seal_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
        observed = dict(observation)
        # A failed gate must retain the actual simulator data for offline
        # replay. This is deliberately ungraded, never a qualified artifact.
        raw = {"schema_version": "scene_configuration_native_import_observation.v1", "status": "ungraded",
            "stage_input_digest": _sha256(stage_input_path), "asset_digest": asset_record["digest"],
            "static_qualification_digest": static_record["digest"], "observation": observed,
            "claim_ceiling": "development_only"}
        try:
            raw_text = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            # Nonfinite or malformed simulator output remains inspectable,
            # while the normal validator still refuses its original values.
            raw_text = canonical_json({**{k: v for k, v in raw.items() if k != "observation"},
                "status": "unserializable_observation", "serialization_error_type": type(error).__name__,
                "observation_repr": repr(observed)[:2_000_000]})
        (output_root / "native_import_observation.v1.json").write_text(raw_text + "\n", encoding="utf-8")
        runtime_identity = observed.get("runtime_identity")
        if (
            not isinstance(runtime_identity, Mapping)
            or runtime_identity.get("engine_version") != ISAAC_VERSION
        ):
            raise TaskEvaluationSceneConfigurationNativeImportDriverError(
                "scene_configuration_native_import_runtime_identity_invalid"
            )
        repeats = observed.get("repeats")
        state_digests, maximum_translation, maximum_rotation = _qualified_repeats(
            repeats, code="scene_configuration_native_import_execution_invalid",
            articulated=articulated,
        )
        required_result = None
        if articulated and static_receipt is not None:
            from .articulated_required_motion_probe import (
                required_motion_plan_from_static_receipt, validate_required_motion_probe_result,
            )
            required_plan = required_motion_plan_from_static_receipt(static_receipt)
            if required_plan is not None:
                try:
                    required_result = validate_required_motion_probe_result(observed.get("required_motion_probe"), required_plan)
                except (ValueError, TypeError, KeyError) as exc:
                    raise TaskEvaluationSceneConfigurationNativeImportDriverError(
                        "scene_configuration_native_import_required_motion_probe_failed") from exc
        artifacts = [
            _write_artifact(
                role="native_import_runtime_result",
                filename=RUNTIME_RESULT_SCHEMA_VERSION,
                value=_runtime_result(
                    articulated_result=articulated,
                    identity=configuration["replacement_identity"],
                    asset_digest=asset_record["digest"],
                    static_digest=static_record["digest"],
                    runtime_identity=runtime_identity,
                    repeats=repeats,
                    state_digests=state_digests,
                    maximum_translation=maximum_translation,
                    maximum_rotation=maximum_rotation,
                    robot_free_probe=observed.get("robot_free_success_probe"),
                    required_motion_probe=required_result,
                ),
            )
        ]
        if destination is not None:
            assert destination_asset_record is not None
            assert destination_static_record is not None
            destination_repeats = observed.get("destination_repeats")
            destination_digests, destination_translation, destination_rotation = (
                _qualified_repeats(
                    destination_repeats,
                    code="scene_configuration_native_import_destination_execution_invalid",
                )
            )
            artifacts.append(
                _write_artifact(
                    role=DESTINATION_RUNTIME_RESULT_ROLE,
                    filename=f"destination_{RUNTIME_RESULT_SCHEMA_VERSION}",
                    value=_runtime_result(
                        articulated_result=False,
                        identity=destination["identity"],
                        asset_digest=destination_asset_record["digest"],
                        static_digest=destination_static_record["digest"],
                        runtime_identity=runtime_identity,
                        repeats=destination_repeats,
                        state_digests=destination_digests,
                        maximum_translation=destination_translation,
                        maximum_rotation=destination_rotation,
                    ),
                )
            )
        elif observed.get("destination_repeats") is not None:
            raise TaskEvaluationSceneConfigurationNativeImportDriverError(
                "scene_configuration_native_import_destination_execution_invalid"
            )
        component: dict[str, Any] = {
            "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "adapter_id": ADAPTER_ID,
            "stage_id": stage["stage_id"],
            "provider_mutations_performed": 0,
            "nested_paid_execution_requested": False,
            "artifacts": artifacts,
            "result_digest": "",
        }
        component["result_digest"] = canonical_digest(component, digest_field="result_digest")
        component_result_path.write_text(canonical_json(component) + "\n", encoding="utf-8")
        return component

    runner_arguments: dict[str, Any] = {
        "asset_path": asset_path,
        "required_checks": checks,
        "observation_consumer": _seal_observation,
    }
    if articulated:
        # Only pass the flag for an articulated asset, so an existing rigid
        # runner double without the parameter keeps working unchanged.
        runner_arguments["articulated"] = True
        runner_arguments["static_receipt"] = static_receipt
    if destination_asset_path is not None:
        runner_arguments["destination_asset_path"] = destination_asset_path
    return dict(native_runner(**runner_arguments))


def main() -> int:
    result = execute_native_import_component()
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ADAPTER_ID",
    "RUNTIME_RESULT_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationNativeImportDriverError",
    "execute_native_import_component",
    "main",
]
