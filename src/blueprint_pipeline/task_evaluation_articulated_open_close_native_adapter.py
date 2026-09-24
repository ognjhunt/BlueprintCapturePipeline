"""One digest-bound native view of a configured articulated open/close task.

Sibling of the rigid relocation adapter. It reads the same seven exact
configured documents, but the task it compiles keeps its moving part inside its
assembly: there is no destination pose and no lift. The success predicate is a
joint interval on the one target joint, taken from the articulation graph the
static qualifier already checked against the exact asset bytes, so the opening
threshold a policy is scored against is frozen from qualified geometry rather
than from the authoring proposal.

Nothing here actuates the task joint. The drive stays passive; opening the
drawer is the policy's job.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .articulation_graph_contract import ArticulationGraphContractError, validate_articulation_graph
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_configured_scene_revision import (
    TaskEvaluationConfiguredSceneRevisionError, validate_configured_scene_revision,
)
from .task_evaluation_launch_preparation_contract import (
    TaskEvaluationLaunchPreparationContractError, validate_launch_preparation_request,
)
from .task_evaluation_rigid_relocation_native_adapter import (
    EXECUTION_CONTRACT_PATH, NATIVE_PHYSICS_FREQUENCY_HZ,
    NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH, SOURCE_OBJECT_CONTRACT_PATH,
    STATIC_QUALIFICATION_CONTRACT_PATH, SUCCESS_CONTRACT_PATH,
    SUPPORT_PLANE_CONTRACT_PATH, DEFINITION_CONTRACT_PATH,
    _source_document, _vector,
)

SCHEMA_VERSION = "task_evaluation_articulated_open_close_native_adapter.v1"
STATIC_QUALIFICATION_SCHEMA = "task_evaluation_articulated_replacement_static_qualification.v1"
TEMPLATE_SCHEMA = "task_evaluation_articulated_open_close_template.v1"
SUCCESS_SCHEMA = "task_evaluation_articulated_open_close_success_criteria.v1"
EXECUTION_SCHEMA = "task_evaluation_articulated_open_close_execution_spec.v1"
#: The exact schema each configured document must carry for this task kind.
#: Only the three task documents differ from the rigid lane; support, source
#: object and native import are shared, and the static qualification is the
#: articulated sibling.
SOURCE_SCHEMAS = {
    DEFINITION_CONTRACT_PATH: TEMPLATE_SCHEMA,
    SUCCESS_CONTRACT_PATH: SUCCESS_SCHEMA,
    EXECUTION_CONTRACT_PATH: EXECUTION_SCHEMA,
    SUPPORT_PLANE_CONTRACT_PATH: "task_evaluation_support_plane_input.v1",
    SOURCE_OBJECT_CONTRACT_PATH: "task_evaluation_source_object_selection.v1",
    STATIC_QUALIFICATION_CONTRACT_PATH: STATIC_QUALIFICATION_SCHEMA,
    NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH: "task_evaluation_replacement_native_import_result.v1",
}
AFFORDANCE_SCHEMA = "native_articulated_graph_interaction_affordance.v1"
STATE_BINDING_SCHEMA = "native_articulated_graph_task_state_binding.v1"
#: Contact and collision force floors, shared with the rigid lane so one scene's
#: sensors do not silently use a different threshold from another's.
TASK_CONTACT_MINIMUM_FORCE_N = 0.5
COLLISION_FAILURE_MINIMUM_FORCE_N = 1.0
#: The assembly is expected to stay put while its drawer moves. These are the
#: root-pose tolerances that turn "the robot dragged the cabinet" into a failure.
ROOT_TRANSLATION_TOLERANCE_M = 0.02
ROOT_ORIENTATION_TOLERANCE_RAD = 0.05
RETREAT_MINIMUM_SEPARATION_M = 0.10
PRECONTACT_CLEARANCE_M = 0.12


class TaskEvaluationArticulatedOpenCloseNativeAdapterError(ValueError):
    """The configured articulated task is not executable as declared."""


def _error(code: str) -> TaskEvaluationArticulatedOpenCloseNativeAdapterError:
    return TaskEvaluationArticulatedOpenCloseNativeAdapterError(
        "articulated_open_close_native_adapter_" + code
    )


def _number(value: Any, *, positive: bool = True) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _error("numeric_invalid") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise _error("numeric_invalid")
    return number


def _revision_bound_task(
    *, revision: Mapping[str, Any], materialized_references: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    template, _binding = _source_document(
        materialized_references,
        contract_path=DEFINITION_CONTRACT_PATH,
        expected_reference=revision["task_template"]["definition"],
        expected_schema=SOURCE_SCHEMAS[DEFINITION_CONTRACT_PATH],
    )
    if (
        template.get("strategy") != "articulated_open_close"
        or template.get("task_identity") != revision["task_template"]["identity"]
        or template.get("object_identity") != revision["replacement"]["identity"]
    ):
        raise _error("request_binding_mismatch")
    return {
        "identity": dict(revision["task_template"]["identity"]),
        "subject": {"identity": dict(revision["replacement"]["identity"])},
        "strategy": "articulated_open_close",
        "binding_mode": "reuse_configured_template",
        "kind": "articulated_manipulation",
        "configured_scene_revision_digest": revision["revision_digest"],
    }


def _jaw_across_handle_bar(contact: Mapping[str, Any], *, revolute: bool) -> list[float]:
    """Close the jaw across the qualified handle bar: Z for a Y bar, Y for a Z bar.

    A drawer keeps its historical Z jaw when no bounds were sealed; a hinged
    door's bar runs along Y or Z depending on its hinge edge, so it must be
    read from the exact handle geometry.
    """
    bounds = contact.get("handle_bounds_link_frame_m")
    try:
        extents = [float(bounds["maximum"][i]) - float(bounds["minimum"][i]) for i in range(3)]
    except (KeyError, IndexError, TypeError, ValueError):
        if revolute:
            raise _error("handle_bar_axis_unresolved") from None
        return [0.0, 0.0, 1.0]
    if not all(math.isfinite(value) and value > 0.0 for value in extents) or extents[1] == extents[2]:
        raise _error("handle_bar_axis_unresolved")
    return [0.0, 0.0, 1.0] if extents[1] > extents[2] else [0.0, 1.0, 0.0]


def _qualified_mechanism(
    *, static: Mapping[str, Any], native_import: Mapping[str, Any],
    replacement_identity: Mapping[str, Any], template: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the executable joint, links and contact from the qualified receipts only."""

    if (
        static.get("schema_version") != STATIC_QUALIFICATION_SCHEMA
        or static.get("status") != "authored_structure_statically_qualified"
        or static.get("asset_kind") != "articulated_assembly"
        or static.get("replacement_identity") != replacement_identity
        or static.get("structural_findings") not in ([], ())
        or static.get("result_digest") != canonical_digest(static, digest_field="result_digest")
        or native_import.get("status") != "qualified"
        or native_import.get("replacement_identity") != replacement_identity
        or native_import.get("native_simulator_import_qualified") is not True
        or native_import.get("blockers") not in ([], ())
        or native_import.get("result_digest")
        != canonical_digest(native_import, digest_field="result_digest")
    ):
        raise _error("executable_geometry_missing")
    try:
        graph = validate_articulation_graph(static["articulation_graph"])
    except (KeyError, ArticulationGraphContractError) as exc:
        raise _error("articulation_graph_invalid") from exc
    targets = [row for row in graph["joints"] if row["role"] == "target"]
    if len(targets) != 1:
        raise _error("single_target_joint_required")
    target = targets[0]
    joint = static.get("task_joint") or {}
    contact = static.get("task_contact") or {}
    link_paths = static.get("link_prim_paths") or {}
    if (
        joint.get("joint_id") != target["joint_id"]
        or joint.get("joint_type") != target["joint_type"]
        or target["joint_type"] not in {"prismatic", "revolute"}
        or not isinstance(link_paths, Mapping)
        or set(link_paths) != {row["link_id"] for row in graph["links"]}
        or any(not str(path).startswith("/") for path in link_paths.values())
        or contact.get("contact_link_id") != target["child_link_id"]
        or not contact.get("handle_prim_paths")
    ):
        raise _error("qualified_mechanism_binding_mismatch")
    limits = [float(value) for value in target["limits"]]
    contact_point = _vector(contact.get("contact_point_link_m"), field="task_contact.contact_point_link_m")
    interval = graph["success_predicate"]["joint_intervals"].get(target["joint_id"])
    if not interval or float(interval[0]) <= float(target["reset_position"]):
        raise _error("success_interval_invalid")
    # The template only declares the intended opening fraction; the executable
    # threshold comes from the qualified joint limit, and the two must agree.
    declared = (template.get("success") or {}).get("minimum_opening_fraction_of_usable_travel")
    if declared is not None:
        expected = float(declared) * limits[1]
        if abs(expected - float(interval[0])) > max(1e-4, 1e-3 * abs(limits[1])):
            raise _error("success_threshold_disagrees_with_qualified_limit")
    return {
        "graph": graph,
        "jaw_unit_asset_root": _jaw_across_handle_bar(contact, revolute=target["joint_type"] == "revolute"),
        "target_joint_axis": [float(value) for value in target["axis"]],
        "target_joint_id": str(target["joint_id"]),
        "target_joint_type": str(target["joint_type"]),
        "target_joint_prim_path": str(joint.get("prim_path") or ""),
        "target_joint_limits": limits,
        "target_success_interval": [float(interval[0]), float(interval[1])],
        "link_prim_paths": {str(k): str(v) for k, v in link_paths.items()},
        "contact_link_id": str(target["child_link_id"]),
        "contact_body_prim_paths": [str(link_paths[target["child_link_id"]])],
        "contact_point_link_m": contact_point,
        "handle_prim_paths": [str(path) for path in contact["handle_prim_paths"]],
    }


def adapt_articulated_open_close_task_template(
    *,
    request: Mapping[str, Any] | None = None,
    configured_revision: Mapping[str, Any] | None = None,
    materialized_references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return one digest-bound native view of the exact configured articulated task."""

    try:
        revision = validate_configured_scene_revision(configured_revision or {})
        validated_request = (
            validate_launch_preparation_request(request) if request is not None else None
        )
    except (
        TaskEvaluationLaunchPreparationContractError,
        TaskEvaluationConfiguredSceneRevisionError,
    ) as exc:
        raise _error("authority_invalid") from exc
    if validated_request is None:
        task = _revision_bound_task(
            revision=revision, materialized_references=materialized_references
        )
    else:
        task = validated_request["task"]
        if (
            validated_request["run_mode"] != "episode_evaluation"
            or task["binding_mode"] != "reuse_configured_template"
            or task["kind"] != "articulated_manipulation"
            or task["strategy"] != "articulated_open_close"
            or task["identity"] != revision["task_template"]["identity"]
            or task["subject"]["identity"] != revision["replacement"]["identity"]
            or task["configured_scene_revision_digest"] != revision["revision_digest"]
        ):
            raise _error("request_binding_mismatch")
    documents: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    for contract_path, expected_reference in (
        (DEFINITION_CONTRACT_PATH, revision["task_template"]["definition"]),
        (SUCCESS_CONTRACT_PATH, revision["task_template"]["success_criteria"]),
        (EXECUTION_CONTRACT_PATH, revision["task_template"]["execution"]),
        (SUPPORT_PLANE_CONTRACT_PATH, revision["registration"]["support_plane"]),
        (SOURCE_OBJECT_CONTRACT_PATH, revision["replacement"]["source_object"]),
        (STATIC_QUALIFICATION_CONTRACT_PATH, revision["replacement"]["static_qualification"]),
        (
            NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH,
            revision["replacement"]["native_import_qualification"],
        ),
    ):
        document, binding = _source_document(
            materialized_references,
            contract_path=contract_path,
            expected_reference=expected_reference,
            expected_schema=SOURCE_SCHEMAS[contract_path],
        )
        documents[contract_path] = document
        bindings.append(binding)
    template = documents[DEFINITION_CONTRACT_PATH]
    success = documents[SUCCESS_CONTRACT_PATH]
    execution = documents[EXECUTION_CONTRACT_PATH]
    if (
        template.get("schema_version") != TEMPLATE_SCHEMA
        or success.get("schema_version") != SUCCESS_SCHEMA
        or execution.get("schema_version") != EXECUTION_SCHEMA
        or template.get("status") != "preregistered_candidate_pending_configured_scene_revision"
        or success.get("status") != "preregistered_before_any_episode"
        or execution.get("status") != "preregistered_before_any_episode"
        or template.get("strategy") != "articulated_open_close"
        or execution.get("strategy") != "articulated_open_close"
        or template.get("task_identity") != task["identity"]
        or template.get("object_identity") != task["subject"]["identity"]
        or template.get("success") != {
            key: value for key, value in success.items()
            if key not in {"schema_version", "status"}
        }
    ):
        raise _error("configured_task_documents_invalid")
    for field in ("control_frequency_hz", "maximum_step_count", "maximum_episode_seconds", "resolved_seed"):
        if template.get(field) != execution.get(field):
            raise _error("configured_task_timing_mismatch")
    if success.get("task_joint_drive_forbidden") is not True:
        raise _error("task_joint_drive_not_forbidden")
    revolute = success.get("joint_type") == "revolute"
    # A hinge is scored in radians: its settle speed and locked tolerance must
    # be declared angular, and a slide must never carry angular units.
    units = (success.get("joint_coordinate_units"), success.get("settled_target_speed_units"))
    if units != (("rad", "rad_per_s") if revolute else (None, None)):
        raise _error("success_units_invalid")
    control_frequency = _number(execution["control_frequency_hz"])
    maximum_steps = int(execution["maximum_step_count"])
    maximum_seconds = _number(execution["maximum_episode_seconds"])
    if (
        maximum_steps <= 0
        or abs(maximum_steps / control_frequency - maximum_seconds) > 1e-9
        or abs(NATIVE_PHYSICS_FREQUENCY_HZ / control_frequency
               - round(NATIVE_PHYSICS_FREQUENCY_HZ / control_frequency)) > 1e-9
    ):
        raise _error("episode_timing_invalid")
    decimation = round(NATIVE_PHYSICS_FREQUENCY_HZ / control_frequency)
    seed = int(execution["resolved_seed"])
    mechanism = _qualified_mechanism(
        static=documents[STATIC_QUALIFICATION_CONTRACT_PATH],
        native_import=documents[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH],
        replacement_identity=revision["replacement"]["identity"],
        template=template,
    )
    if mechanism["target_joint_type"] != success.get("joint_type"):
        raise _error("success_joint_type_mismatch")
    hold_seconds = _number(success["minimum_hold_seconds"])
    settle_window_samples = max(1, int(round(hold_seconds * control_frequency)))
    if settle_window_samples > maximum_steps:
        raise _error("hold_window_exceeds_episode")
    source_documents = {
        "source_documents_digest": canonical_digest(
            {"bindings": bindings, "documents": documents}
        ),
        "bindings": bindings,
        "documents": documents,
    }
    graph = mechanism["graph"]
    graph_digest = canonical_digest(dict(graph))
    # The runtime identifies assets by a name a simulator prim can carry, while
    # the configured identity keeps its own spelling. Bind both, exactly as the
    # native packet boundary expects, instead of letting one shadow the other.
    source_subject_id = str(task["subject"]["identity"]["id"])
    runtime_subject_id = re.sub(r"[^A-Za-z0-9_]", "_", source_subject_id)
    if not runtime_subject_id.replace("_", "a").isalnum():
        raise _error("subject_identity_unrepresentable")
    affordance: dict[str, Any] = {
        "schema_version": AFFORDANCE_SCHEMA,
        "subject_asset_id": runtime_subject_id,
        "articulation_graph_digest": graph_digest,
        "contact_link_id": mechanism["contact_link_id"],
        "contact_body_prim_paths": list(mechanism["contact_body_prim_paths"]),
        "contact_point_link_m": list(mechanism["contact_point_link_m"]),
        "handle_prim_paths": list(mechanism["handle_prim_paths"]),
        # The drawer is pulled along its own opening axis; the gripper closes on
        # the handle bar across that axis. Both are asset-frame, matching the
        # joint axis the qualifier read back from the bytes. A hinged door's
        # free edge also leaves along +X from closed, then follows the arc
        # about its hinge axis; the contact point rides the door link.
        "approach_unit_asset_root": [-1.0, 0.0, 0.0],
        "retreat_unit_asset_root": [-1.0, 0.0, 0.0],
        "pull_unit_asset_root": [1.0, 0.0, 0.0],
        "jaw_unit_asset_root": mechanism["jaw_unit_asset_root"],
        **({"pull_follows_arc_about_axis_asset_root": mechanism["target_joint_axis"]}
           if mechanism["target_joint_type"] == "revolute" else {}),
        "precontact_clearance_m": PRECONTACT_CLEARANCE_M,
        "retreat_clearance_m": PRECONTACT_CLEARANCE_M,
        "task_joint_drive_forbidden": True,
        "affordance_digest": "",
    }
    affordance["affordance_digest"] = canonical_digest(affordance, digest_field="affordance_digest")
    native_task_spec = {
        "schema_version": "adp_task_spec.v2",
        "task_kind": "articulated_open_close",
        "subject_asset_id": runtime_subject_id,
        "source_subject_identity": source_subject_id,
        "manipulation_strategy": "articulated_open_close",
        "prompt": str(template.get("instruction") or "").strip()
        or f"Open the {template['mechanism']['task_part_label']}.",
        "instruction_subject_label": str(template.get("instruction_subject_label") or ""),
        "visible_target_label": str(template.get("visible_target_label") or ""),
        "articulation_graph": graph,
        "articulation_graph_digest": graph_digest,
        "interaction_affordance": affordance,
        "settle_window_samples": settle_window_samples,
        "maximum_settled_target_speed": _number(success["maximum_settled_target_speed"]),
        "locked_joint_motion_tolerance": _number(success["locked_joint_motion_tolerance"]),
        "movement_epsilon": max(1e-4, mechanism["target_joint_limits"][1] / 100.0),
        "control_frequency_hz": int(control_frequency),
        "maximum_action_steps": maximum_steps,
        "configured_success_criteria": {
            key: value for key, value in success.items()
            if key not in {"schema_version", "status"}
        },
        "configured_task_source_documents_digest": source_documents["source_documents_digest"],
        "executable_opening_threshold": {
            "target_joint_id": mechanism["target_joint_id"],
            "joint_type": mechanism["target_joint_type"],
            "success_interval": mechanism["target_success_interval"],
            "qualified_joint_limits": mechanism["target_joint_limits"],
            "reset_position": 0.0,
            "hold_seconds": hold_seconds,
            "hold_window_samples": settle_window_samples,
            "authority": "frozen_from_static_qualification_of_exact_asset_bytes",
            "travel_is_measured": False,
            **({"coordinate_units": "rad", "opening_fraction_of_swing": round(
                mechanism["target_success_interval"][0] / mechanism["target_joint_limits"][1], 6)}
               if mechanism["target_joint_type"] == "revolute" else {}),
        },
    }
    owner_authority = template.get("owner_success_contract_authority")
    if isinstance(owner_authority, Mapping):
        native_task_spec["configured_owner_authority"] = dict(owner_authority)
    joint_by_id = {str(row["joint_id"]): row for row in graph["joints"]}
    static_digest = documents[STATIC_QUALIFICATION_CONTRACT_PATH]["result_digest"]
    task_joint_bindings: list[dict[str, Any]] = []
    for joint_id in sorted(joint_by_id):
        row = joint_by_id[joint_id]
        prim_path = (
            mechanism["target_joint_prim_path"]
            if joint_id == mechanism["target_joint_id"]
            else f"/Asset/joints/{joint_id}"
        )
        if row["joint_type"] == "fixed":
            task_joint_bindings.append({
                "joint_id": joint_id, "joint_prim_path": prim_path,
                "readback_kind": "fixed_joint_static",
                "static_qualification_digest": static_digest, "role": row["role"],
            })
        else:
            task_joint_bindings.append({
                "joint_id": joint_id, "joint_prim_path": prim_path,
                "native_joint_name": PurePosixPath(prim_path).name,
                "readback_kind": "native_coordinate", "role": row["role"],
            })
    task_state_binding = {
        "schema_version": STATE_BINDING_SCHEMA,
        "articulation_graph_digest": graph_digest,
        "interaction_affordance_digest": affordance["affordance_digest"],
        "link_native_body_names": {
            link_id: PurePosixPath(path).name
            for link_id, path in mechanism["link_prim_paths"].items()
        },
        "task_contact_minimum_force_n": TASK_CONTACT_MINIMUM_FORCE_N,
        "collision_failure_minimum_force_n": COLLISION_FAILURE_MINIMUM_FORCE_N,
        "retreat_minimum_separation_m": RETREAT_MINIMUM_SEPARATION_M,
        "root_translation_tolerance_m": ROOT_TRANSLATION_TOLERANCE_M,
        "root_orientation_tolerance_rad": ROOT_ORIENTATION_TOLERANCE_RAD,
    }
    support = documents[SUPPORT_PLANE_CONTRACT_PATH]
    if support.get("status") != "frozen_candidate_pending_production_validation" or not str(
        support.get("sage_prim_path") or ""
    ).startswith("/"):
        raise _error("support_plane_invalid")
    start_center = _vector(template.get("start_center_xyz_m"), field="template.start_center_xyz_m")
    assembly = template.get("assembly_bounds_xyz_m") or {}
    lower = _vector(assembly.get("minimum"), field="template.assembly_bounds.minimum")
    support_top = _number(support.get("top_z_m"), positive=False)
    # Place the assembly root so its own base sits on the registered support
    # top; the authored frame's origin is the carcass bottom.
    root_position = [float(start_center[0]), float(start_center[1]), support_top - float(lower[2])]
    scenario_document = {
        "schema_version": "adp009d_scenario_instance.v1",
        "context_kind": "evaluation_cell",
        "cell_id": f"{task['identity']['id']}-nominal",
        "seed": seed,
        "parameter_bindings": [],
        "instance_digest": "",
    }
    scenario_document["instance_digest"] = canonical_digest(
        scenario_document, digest_field="instance_digest"
    )
    native_definition = {
        "schema_version": "task_evaluation_native_task_definition.v1",
        "identity": dict(task["identity"]),
        "task_spec": native_task_spec,
        "task_object_pose_world": {
            "position_world_m": root_position,
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "task_joint_bindings": task_joint_bindings,
        "task_state_binding": task_state_binding,
        "task_object_reset_joint_positions": {
            PurePosixPath(
                mechanism["target_joint_prim_path"]
                if joint_id == mechanism["target_joint_id"]
                else f"/Asset/joints/{joint_id}"
            ).name: 0.0
            for joint_id in sorted(joint_by_id)
            if joint_by_id[joint_id]["joint_type"] != "fixed"
        },
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "adapted",
        "external_task_kind": "articulated_manipulation",
        "native_task_kind": "articulated_open_close",
        "manipulation_strategy": "articulated_open_close",
        "configured_scene_revision_digest": revision["revision_digest"],
        "source_documents": source_documents,
        "native_task_definition": native_definition,
        "native_success_criteria": {
            "schema_version": "task_evaluation_native_success_criteria.v1",
            "identity": dict(task["identity"]),
            "criteria": {
                key: value for key, value in success.items()
                if key not in {"schema_version", "status"}
            },
        },
        "native_episode_execution": {
            "schema_version": "task_evaluation_native_episode_execution.v1",
            "identity": dict(task["identity"]),
            "physics_frequency_hz": NATIVE_PHYSICS_FREQUENCY_HZ,
            "control_frequency_hz": int(control_frequency),
            "control_decimation": int(decimation),
            "maximum_step_count": maximum_steps,
            "maximum_episode_seconds": maximum_seconds,
            "scenario": {
                "context_kind": "evaluation_cell",
                "cell_id": scenario_document["cell_id"],
                "instance_digest": scenario_document["instance_digest"],
                "seed": seed,
                "context_document": scenario_document,
            },
        },
        "claim_boundary": {
            "task_joint_is_passive": True,
            "policy_must_open_the_part": True,
            "joint_travel_is_measured": False,
            "simulator_execution_is_not_physical_truth": True,
        },
        "adapter_digest": "",
    }
    result["adapter_digest"] = canonical_digest(result, digest_field="adapter_digest")
    return result


__all__ = [
    "SCHEMA_VERSION",
    "TaskEvaluationArticulatedOpenCloseNativeAdapterError",
    "adapt_articulated_open_close_task_template",
]
