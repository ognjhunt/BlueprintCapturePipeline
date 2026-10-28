"""Author bounded button tasks from registered layout and admitted model bytes.

Beta capture-to-native preparation only. The protected, existing engine catalog
may carry ``authoring_assets``; these are reusable engineering model records,
never a job's finished native profile. No simulator, policy or provider is run.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from trimesh.transformations import quaternion_from_matrix

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .launch_immutable_input_writer import write_exclusive_private_bytes
from .task_family_authoring import _operations, _QUESTIONS
from .task_family_native_contracts import _pose, family_source_blockers
from .task_family_scoring import seal_task_family_contract
from .task_object_authoring_contracts import file_record, verified_file

SCHEMA = "website_button_family_authoring.v1"
LAYOUT_SCHEMA = "website_registered_task_layout.v1"
FACTS_SCHEMA = "website_family_task_facts.v1"
MAX_BUTTONS = 8
_CRITERIA = ("minimum_button_travel_m", "maximum_contact_force_n", "minimum_hold_seconds")


class _EngineeringDependency(ValueError):
    def __init__(self, quantity: str, question: str, category: str = "source_measurement"):
        self.quantity, self.question, self.category = quantity, question, category
        super().__init__(quantity)


def _require(condition: Any, reason: str) -> None:
    if not condition:
        raise ValueError("website_button_authoring_" + reason)


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _retain(root: Path, kind: str, value: Mapping[str, Any]) -> dict[str, Any]:
    value = {"schema_version": "website_family_authoring_stage.v1", "stage": kind,
             "inputs_and_result": dict(value), "claim_ceiling": "development_only"}
    value["stage_digest"] = canonical_digest(value, digest_field="stage_digest")
    path = root / "family_authoring" / "stages" / value["stage_digest"][7:] / (kind + ".json")
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    write_exclusive_private_bytes(path, data)
    return file_record(path)


def _pose_from_matrix(matrix: Any) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=float)
    _require(matrix.shape == (4, 4) and np.isfinite(matrix).all()
             and np.allclose(matrix[3], [0, 0, 0, 1])
             and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8)
             and np.linalg.det(matrix[:3, :3]) > 0, "layout_pose_invalid")
    quaternion = quaternion_from_matrix(matrix)
    result = {"position_world_m": matrix[:3, 3].tolist(),
              "orientation_xyzw": [float(quaternion[i]) for i in (1, 2, 3, 0)]}
    _pose(result)
    return result


def _registered_pose(pose_source: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
    position, rotation = _pose(pose_source)
    transform = np.asarray(frame["source_to_runtime"], dtype=float)
    linear = transform[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    _require(scale > 0 and np.allclose(linear.T @ linear, np.eye(3) * scale**2, atol=1e-8),
             "layout_registration_invalid")
    value = np.eye(4)
    value[:3, :3] = linear @ rotation / scale
    value[:3, 3] = (linear @ position + transform[:3, 3]) * frame["runtime_meters_per_unit"]
    return _pose_from_matrix(value)


def _layout(task_masks, geometry, scene):
    value = task_masks.get("registered_layout")
    if value is None:
        return None
    _require(isinstance(value, Mapping) and value.get("schema_version") == LAYOUT_SCHEMA
             and value.get("layout_digest") == canonical_digest(value, digest_field="layout_digest")
             and value.get("source_geometry_digest") == geometry["digest"]
             and value.get("collision_mesh_digest") == scene["collision_mesh_digest"]
             and value.get("source_video_digest") == (geometry.get("binding") or {}).get("source_video_digest"),
             "layout_source_changed")
    # Neither estimated visible boxes nor a model-estimated camera frame can
    # supply metric placement. Their original evidence remains unchanged.
    if geometry.get("metric_measurement_proven") is not True or value.get("metric_registration_verified") is not True:
        return None
    evidence = value.get("registration_evidence")
    _require(isinstance(evidence, Mapping), "registration_evidence_required")
    evidence_path = verified_file(evidence)
    frame = value.get("frame_registration")
    _require(isinstance(frame, Mapping), "layout_registration_required")
    registration = json.loads(evidence_path.read_text())
    _require(isinstance(registration, Mapping) and registration.get("frame_registration") == frame
             and registration.get("source_video_digest") == value["source_video_digest"],
             "registration_evidence_binding_changed")
    # Reuse the existing frame/source checks, before any geometry derivation.
    bindings = {"task_context_digest": "unused", "source_geometry_digest": geometry["digest"],
                "task_masks_digest": task_masks["digest"], "splat_digest": scene["splat_digest"],
                "collision_mesh_digest": scene["collision_mesh_digest"]}
    fake = {"task_state_binding": {"source_scene_binding": {**bindings, "frame_registration": frame}},
            "assets": [{"semantic_role": role, "content_digest": scene[key]} for role, key in
                       (("scene_appearance", "splat_digest"), ("scene_collision", "collision_mesh_digest"))]}
    _require(not family_source_blockers(native=fake, source_bindings=bindings, source_scene=scene),
             "layout_registration_required")
    return value


def _facts(context, video):
    value = (context.get("operator_answers") or {}).get("family_task_facts")
    if value is None:
        return {}
    _require(isinstance(value, Mapping), "owner_facts_source_changed")
    _require("canonicalization" not in value or value["canonicalization"] == "rfc8785",
             "owner_facts_canonicalization_invalid")
    digest = cross_runtime_canonical_digest if value.get("canonicalization") == "rfc8785" else canonical_digest
    _require(isinstance(value, Mapping) and value.get("schema_version") == FACTS_SCHEMA
             and value.get("facts_digest") == digest(value, digest_field="facts_digest")
             and value.get("capture_id") == context["capture_id"]
             and value.get("source_video_digest") == video
             and value.get("task_description_digest") == digest({"description": context["description"]})
             and value.get("task_success_criteria_digest") == digest({"success_criteria": context.get("success_criteria")}),
             "owner_facts_source_changed")
    if context.get("confirmed") is not True or value.get("confirmed") is not True:
        return {}
    _require(isinstance(value.get("confirmation_reference"), str) and bool(value["confirmation_reference"].strip()),
             "owner_confirmation_reference_required")
    rows = value.get("operations")
    _require(isinstance(rows, list) and len(rows) <= MAX_BUTTONS
             and all(isinstance(row, dict) and isinstance(row.get("operation_id"), str) for row in rows)
             and len({row["operation_id"] for row in rows}) == len(rows), "owner_facts_invalid")
    return {row["operation_id"]: row for row in rows}


def _catalog(layout):
    # Only existing operator configuration may name this path. Never load a
    # catalog path, executable or finished native packet from capture metadata.
    configured = os.environ.get("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH")
    if not configured or layout is None:
        return None
    from .task_family_native_runtime import CATALOG_SCHEMA, _read
    path = Path(configured)
    _require(path.is_absolute(), "operator_catalog_path_invalid")
    value = _read(path, code="catalog_required")
    _require(value.get("schema_version") == CATALOG_SCHEMA
             and value.get("catalog_digest") == canonical_digest(value, digest_field="catalog_digest"),
             "operator_catalog_changed")
    robot = layout.get("robot") or {}
    rows = [row for row in value.get("runtimes", []) if isinstance(row, dict)
            and "button_press" in row.get("task_families", [])
            and robot.get("configuration") in row.get("robot_interfaces", [])]
    if len(rows) != 1:
        return None
    return path, value, rows[0]


def _asset(catalog, asset_id):
    path, _value, runtime = catalog
    rows = [row for row in runtime.get("authoring_assets", []) if row.get("asset_id") == asset_id]
    if len(rows) != 1:
        return None
    value = rows[0]
    _require(value.get("task_family") == "button_press"
             and isinstance(value.get("rights_reference"), str) and value["rights_reference"].strip(),
             "asset_rights_or_family_missing")
    source = verified_file(value["source"], root=path.parent)
    contact = value.get("contact_material")
    if not isinstance(contact, Mapping) or any(contact.get(key) is None for key in
            ("static_friction", "dynamic_friction", "restitution")):
        raise _EngineeringDependency("button_contact_material",
            "Engineering evidence for this button's static/dynamic friction and restitution is missing; retain source values before authoring.")
    _require(all(_number(contact[key]) and contact[key] >= 0 for key in contact)
             and contact["dynamic_friction"] <= contact["static_friction"] and contact["restitution"] <= 1,
             "contact_material_invalid")
    # This bounded first producer admits one self-contained prismatic button
    # model. It extracts actual joint limits/axis from pinned URDF bytes; no
    # hidden geometry, mesh dependency, material or joint default is invented.
    raw = source.read_bytes()
    _require(len(raw) <= 1024 * 1024 and b"<!DOCTYPE" not in raw.upper(), "asset_model_invalid")
    model = ET.fromstring(raw)
    permitted = {"robot", "link", "inertial", "origin", "mass", "inertia", "collision", "geometry",
                 "box", "cylinder", "sphere", "visual", "material", "color", "joint", "parent", "child",
                 "axis", "limit", "dynamics"}
    _require(model.tag == "robot", "button_model_root_invalid")
    if any(node.tag not in permitted for node in model.iter()):
        raise _EngineeringDependency("button_model_import_support",
            "Prepare and qualify this model's exact dependencies/import path; the bounded producer currently supports self-contained primitive URDF buttons.",
            "runtime_integration")
    links = model.findall("link")
    _require(len(links) == 2 and all(link.get("name") for link in links)
             and len({link.get("name") for link in links}) == 2, "button_model_links_invalid")
    for link in links:
        mass, inertia = link.find("inertial/mass"), link.find("inertial/inertia")
        if mass is None or inertia is None or link.find("collision/geometry") is None:
            raise _EngineeringDependency("button_geometry_and_inertia",
                "Retain the button model's complete collision geometry, mass and inertia from engineering/source evidence.")
        mass_value = float(mass.get("value", "nan"))
        inertia_values = {key: float(inertia.get(key, "nan")) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
        inertia_matrix = np.array([[inertia_values["ixx"], inertia_values["ixy"], inertia_values["ixz"]],
            [inertia_values["ixy"], inertia_values["iyy"], inertia_values["iyz"]],
            [inertia_values["ixz"], inertia_values["iyz"], inertia_values["izz"]]])
        _require(_number(mass_value) and mass_value > 0 and np.isfinite(inertia_matrix).all()
                 and np.linalg.eigvalsh(inertia_matrix).min() > 0, "button_inertia_invalid")
    for geometry in model.iter("geometry"):
        shapes = list(geometry)
        _require(len(shapes) == 1, "button_model_geometry_invalid")
        shape = shapes[0]
        fields = {"box": ("size",), "cylinder": ("radius", "length"), "sphere": ("radius",)}
        _require(shape.tag in fields, "button_model_geometry_invalid")
        values = [float(number) for field in fields[shape.tag] for number in shape.get(field, "nan").split()]
        _require(len(values) == (3 if shape.tag == "box" else len(fields[shape.tag]))
                 and all(_number(number) and number > 0 for number in values), "button_model_geometry_invalid")
    for origin in model.iter("origin"):
        for field in ("xyz", "rpy"):
            if field in origin.attrib:
                values = [float(number) for number in origin.get(field).split()]
                _require(len(values) == 3 and all(_number(number) for number in values), "button_model_origin_invalid")
    joints = model.findall("joint")
    if len(joints) != 1 or joints[0].get("type") != "prismatic":
        raise _EngineeringDependency("button_model_import_support",
            "Prepare a qualified state/reset adapter for this button assembly; the bounded producer currently supports one prismatic joint.",
            "runtime_integration")
    joint = joints[0]
    parent, child = joint.find("parent"), joint.find("child")
    _require(parent is not None and child is not None and parent.get("link") != child.get("link")
             and {parent.get("link"), child.get("link")} == {link.get("name") for link in links},
             "button_model_link_binding_invalid")
    _require(isinstance(joint.get("name"), str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", joint.get("name")),
             "button_joint_identity_invalid")
    axis, limit = joint.find("axis"), joint.find("limit")
    _require(axis is not None and limit is not None, "joint_engineering_facts_required")
    direction = [float(number) for number in axis.get("xyz", "").split()]
    lower, upper = float(limit.get("lower", "nan")), float(limit.get("upper", "nan"))
    effort, velocity = float(limit.get("effort", "nan")), float(limit.get("velocity", "nan"))
    _require(_number(effort) and effort > 0 and _number(velocity) and velocity > 0, "button_model_limits_invalid")
    dynamics = joint.find("dynamics")
    if dynamics is not None:
        _require(all(_number(float(number)) and float(number) >= 0 for number in dynamics.attrib.values()),
                 "button_model_dynamics_invalid")
    _require(len(direction) == 3 and all(_number(number) for number in direction)
             and math.isclose(sum(number**2 for number in direction), 1, abs_tol=1e-8)
             and _number(lower) and _number(upper) and lower < upper, "joint_engineering_facts_invalid")
    mechanics = value.get("button_mechanics") or {}
    for field in ("rest_position_m", "reset_position_m", "press_direction", "reset_sensor_active",
                  "reset_contact_force_n", "activation_sensor_id"):
        if mechanics.get(field) is None:
            raise _EngineeringDependency("button_" + field,
                "Retain engineering/source evidence for this button's " + field.replace("_", " ") + "; it cannot be inferred from visible bounds.")
    rest, reset, sign = (mechanics.get(key) for key in ("rest_position_m", "reset_position_m", "press_direction"))
    _require(_number(rest) and _number(reset) and lower <= rest <= upper and lower <= reset <= upper
             and type(sign) is int and sign in (-1, 1)
             and type(mechanics.get("reset_sensor_active")) is bool
             and _number(mechanics.get("reset_contact_force_n")) and mechanics["reset_contact_force_n"] >= 0
             and isinstance(mechanics.get("activation_sensor_id"), str) and mechanics["activation_sensor_id"]
             and isinstance(value.get("engineering_evidence"), Mapping), "button_engineering_facts_required")
    verified_file(value["engineering_evidence"], root=path.parent)
    return {"record": value, "joint_id": joint.get("name"), "axis_local": direction,
            "joint_limits_m": [lower, upper], "mechanics": mechanics}


def author_button_family_inputs(*, family, task_context, task_masks, source_geometry, base_scene,
                               source_bindings, output_root, task_semantics=None, now=None):
    """Build native inputs, or retain exact facts/dependencies without effects."""
    video = (source_geometry.get("binding") or {}).get("source_video_digest")
    operations, reading = _operations(family=family, task_context=task_context, task_semantics=task_semantics,
                                      source_video_digest=video, contract=None)
    if not operations or any(row["task_family"] != "button_press" for row in operations):
        return None
    if reading is not None and (not reading["complete"] or reading["unresolved"] or not reading["matches_declared_scope"]):
        return None
    requirements, retained = [], []

    def missing(stage, quantity, category, question):
        requirements.append({"requirement_id": stage + ":" + quantity, "stage_id": stage,
            "quantity": quantity, "category": category, "owner": {"task_fact": "task_owner",
                "runtime_integration": "runtime_operator"}.get(category, "scene_preparation"),
            "status": "required", "question": question, "unit": _QUESTIONS.get(quantity, (None,))[0],
            "authority_created": False})

    bounded = len(operations) <= MAX_BUTTONS
    if not bounded:
        missing("job", "button_sequence_support", "runtime_integration",
                "Prepare support for this retained sequence; the bounded producer currently composes at most eight distinct buttons without release/repress steps.")
    facts = _facts(task_context, video) if bounded else {}
    _require(set(facts) <= {row["stage_id"] for row in operations}, "owner_operation_changed")
    layout = _layout(task_masks, source_geometry, base_scene)
    source_files = []
    if layout is not None:
        for frame in source_geometry.get("frames", []):
            records = {"frame_id": frame["frame_id"]}
            for prefix in ("image", "geometry"):
                reference = file_record(Path(frame[prefix + "_path"]))
                _require(reference["sha256"] == frame[prefix + "_digest"], "source_frame_bytes_changed")
                records[prefix] = reference
            source_files.append(records)
    catalog = _catalog(layout)
    targets = {row["target_id"]: row for row in task_masks.get("targets", [])}
    layouts = (layout or {}).get("targets", {})
    stages, resets, bindings, assets, asset_sources, used = [], {}, {}, [], {}, set()
    engineering_assets = {}
    for operation in operations if bounded else ():
        key = operation["stage_id"]
        answer = facts.get(key, {})
        criteria = {}
        for quantity in _CRITERIA:
            number = answer.get(quantity)
            if number is None:
                if quantity == "minimum_hold_seconds":
                    missing(key, quantity, "task_fact", _QUESTIONS[quantity][1])
                else:
                    missing(key, quantity, "source_measurement",
                        "Retain engineering/source evidence for " + quantity.replace("_", " ")
                        + " for this button; do not infer joint or force limits from visible bounds.")
            else:
                _require(_number(number) and number >= 0 and (quantity != "minimum_button_travel_m" or number > 0),
                         "owner_threshold_invalid")
                criteria[quantity] = number
        quote = (operation.get("source_operation") or {}).get("task_quote") or task_context["description"]
        matches = [target for target in targets.values() if target.get("semantic_label") and re.search(
            r"(?<!\w)" + re.escape(target["semantic_label"]) + r"(?!\w)", quote, re.IGNORECASE)]
        selected = answer.get("target_id")
        if selected is not None:
            _require(selected in targets, "owner_target_changed")
            _require(not matches or selected in {row["target_id"] for row in matches}, "owner_target_source_conflict")
            target = targets[selected]
        elif len(matches) == 1:
            target = matches[0]
        else:
            missing(key, "target_id", "task_fact", "Which captured button does this operation press?")
            continue
        target_id = target["target_id"]
        if target_id in used:
            missing(key, "repeated_button_reset", "runtime_integration",
                    "Prepare a source-bound release/repress reset for this repeated button operation.")
            continue
        used.add(target_id)
        placed = layouts.get(target_id)
        if layout is None or not isinstance(placed, Mapping):
            missing(key, "registered_target_layout", "source_measurement",
                    "Register the observed button pose and frame against metric source evidence; visible bounds are insufficient.")
            continue
        frames = {row["frame_id"]: row for row in source_geometry.get("frames", [])}
        observed = {row.get("source_frame_id") for row in (target.get("track") or {}).get("observations", [])}
        _require(placed.get("source_frame_ids") and set(placed["source_frame_ids"]) <= set(frames) & observed,
                 "target_source_frames_changed")
        if catalog is None:
            missing(key, "admitted_button_asset_and_runtime", "runtime_integration",
                    "Bind the engineering button model and compatible robot to the existing protected runtime catalog.")
            continue
        try:
            asset = _asset(catalog, placed.get("asset_id"))
        except _EngineeringDependency as dependency:
            missing(key, dependency.quantity, dependency.category, dependency.question)
            continue
        if asset is None:
            missing(key, "button_engineering_asset", "source_measurement",
                    "Retain the licensed button model, joint/activation/reset evidence and exact dependency bytes.")
            continue
        engineering_assets[asset["record"]["asset_id"]] = asset["record"]
        pose = _registered_pose(placed["pose_source"], layout["frame_registration"])
        mechanics = asset["mechanics"]
        asset_id = "button-" + hashlib.sha256(target_id.encode()).hexdigest()[:20]
        native_asset = {"asset_id": asset_id, "semantic_role": "button",
            "content_digest": asset["record"]["source"]["sha256"], "bundle_relative_path": "assets/" + asset_id + ".urdf",
            "reset_state": {"pose_world": pose, "joint_positions_m": {asset["joint_id"]: mechanics["reset_position_m"]}},
            "native_joint_id": asset["joint_id"], "activation_sensor_id": mechanics["activation_sensor_id"],
            "joint_axis_local": asset["axis_local"], "joint_limits_m": asset["joint_limits_m"],
            "contact_material": asset["record"]["contact_material"],
            "source_geometry_basis": "admitted_engineering_model", "physical_measurement_claimed": False,
            "engineering_record_digest": canonical_digest(asset["record"]), "source_target_id": target_id}
        retained.append(_retain(output_root, "source-" + asset_id, {"layout_digest": layout["layout_digest"],
            "source_frame_bindings": [frames[frame] for frame in placed["source_frame_ids"]], "asset": native_asset}))
        assets.append(native_asset)
        asset_sources[asset_id] = asset["record"]["source"]
        if len(criteria) != len(_CRITERIA):
            continue
        travel = max(0, (mechanics["reset_position_m"] - mechanics["rest_position_m"]) * mechanics["press_direction"])
        resets[key] = {"button_travel_m": travel, "button_activated": mechanics["reset_sensor_active"],
                       "contact_force_n": mechanics["reset_contact_force_n"]}
        stages.append({"stage_id": key, "task_family": "button_press", "minimum_hold_seconds": criteria.pop("minimum_hold_seconds"),
                       "criteria": criteria})
        bindings[key] = {"native_readback_kind": "button_press", "asset_ids": [asset_id], "asset_roles": {"button": asset_id},
                        "measurement_geometry": {"joint_rest_position_m": mechanics["rest_position_m"],
                                                 "press_direction": mechanics["press_direction"]}}
        retained.append(_retain(output_root, "operation-" + str(operation["order"]),
            {"source_operation": operation, "owner_facts": answer, "target_id": target_id,
             "stage": stages[-1], "reset": resets[key], "binding": bindings[key]}))
    if layout is None:
        missing("job", "registered_target_layout", "source_measurement",
                "Register the observed button pose/frame from metric evidence; estimated visible bounds cannot supply it.")
    if catalog is None:
        missing("job", "native_runtime_inputs", "runtime_integration",
                "Bind an admitted button model/robot in the existing protected runtime catalog; no customer engine data is required.")
    result = {"schema_version": SCHEMA, "status": "needs_input", "operations": operations,
              "requirements": requirements, "stages": retained, "claim_ceiling": "development_only",
              "provider_mutation_performed": False, "automatic_paid_execution_authorized": False}
    if not requirements:
        from .task_family_authoring_evidence import INPUT_SCHEMA
        evidence = _retain(output_root, "evidence-inputs", {"schema_version": INPUT_SCHEMA,
            "binding": dict(source_bindings), "source_video_digest": video, "layout_digest": layout["layout_digest"],
            "registration_evidence": layout["registration_evidence"], "runtime_catalog": file_record(catalog[0]),
            "source_frames": source_files, "engineering_assets": list(engineering_assets.values())})
        retained.append(evidence)
        result["outputs"] = _compile(family, task_context, source_bindings, source_geometry, base_scene,
            layout, catalog, stages, resets, bindings, assets, asset_sources, evidence=evidence, now=now)
        result["status"] = "authored"
        retained.append(_retain(output_root, "native-inputs", result["outputs"]))
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result["reference"] = _retain(output_root, "authoring-result", result)
    return result


def _compile(family, context, source_bindings, geometry, scene, layout, catalog, stages, resets,
             bindings, assets, asset_sources, *, evidence, now):
    path, registry, runtime = catalog
    scope = {"scene_id": context["scene_id"], "task_id": context.get("task_id") or "website-" + context["context_digest"][7:27],
             "task_context_digest": context["context_digest"]}
    contract = seal_task_family_contract(task_family=family, scope=scope, stages=stages, reset=resets,
        runtime_identity_digest=runtime["runtime_identity_digest"])
    frame = layout["frame_registration"]
    robot = layout["robot"]
    cameras = []
    by_id = {row["frame_id"]: row for row in geometry["frames"]}
    _require(isinstance(layout.get("camera_frame_ids"), list) and bool(layout["camera_frame_ids"]),
             "source_cameras_required")
    for frame_id in layout["camera_frame_ids"]:
        _require(frame_id in by_id, "camera_source_changed")
        source = by_id[frame_id]
        source_pose = _pose_from_matrix(source["world_from_camera"])
        cameras.append({"camera_id": frame_id, "role": "source_reference", "pose_world": _registered_pose(source_pose, frame),
                        "intrinsics": source["intrinsics"], "resolution": [source["width"], source["height"]]})
    for role, prefix in (("scene_appearance", "splat"), ("scene_collision", "collision_mesh")):
        reference = file_record(Path(scene[prefix + "_path"]))
        assets.append({"asset_id": role, "semantic_role": role, "content_digest": reference["sha256"],
                       "bundle_relative_path": "assets/" + role + Path(reference["path"]).suffix,
                       "reset_state": {"fixed": True}})
        asset_sources[role] = reference
    scenario = {"source_geometry_digest": geometry["digest"], "layout_digest": layout["layout_digest"],
                "task_context_digest": context["context_digest"], "reset": resets}
    inputs = {"assets": assets, "task_state_binding": {"source": "native_simulator_state", "engine_id": runtime["engine_id"],
        "runtime_identity_digest": runtime["runtime_identity_digest"], "stage_bindings": bindings,
        "source_scene_binding": {**source_bindings, "frame_registration": frame},
        "source_authoring_evidence": evidence},
        "robot_base_pose_world": _registered_pose(robot["base_pose_source"], frame),
        "robot_joint_reset_positions_rad": robot["joint_reset_positions_rad"], "robot_configuration": robot["configuration"],
        "cameras": cameras, "scenario_cell_id": "capture-authored-0", "scenario_instance_digest": canonical_digest(scenario),
        "seed": 0, "scenario_context_kind": "construction_canary"}
    from .native_task_runtime_contract import materialize_native_task_runtime_contract
    native = materialize_native_task_runtime_contract(scene_id=scope["scene_id"], task_id=scope["task_id"],
        task_spec={"task_kind": family, "task_success_contract": contract}, **inputs)
    _require(not family_source_blockers(native=native, source_bindings=source_bindings, source_scene=scene), "source_binding_invalid")
    selection = {"runtime_id": runtime["runtime_id"], "runtime_catalog_digest": registry["catalog_digest"]}
    from .task_family_native_runtime import _qualified_registration
    dispatch_blockers = []
    try:
        _qualified_registration(native, selection, path, now=time.time() if now is None else now)
    except ValueError as error:
        # AUT-13 runtime qualification is not a prerequisite for retaining
        # static native inputs. It remains mandatory before an intake exists.
        if not str(error).startswith("task_family_native_runtime_"):
            raise
        dispatch_blockers.append(str(error))
    return {"task_success_contract": contract, "native_runtime_inputs": inputs,
            "task_family_runtime_selection": selection, "asset_sources": asset_sources,
            "authoring_dispatch_blockers": dispatch_blockers}
