"""Reusable fail-closed contracts for the ADP-009D Isaac Lab evaluation harness.

The module owns deterministic validation and scenario materialization.  Isaac
Lab, Arena, appearance renderers, ovphysx, and learned policies remain runtime adapters behind
the existing :mod:`evaluation_run_contract` seams.  Nothing in this module can
turn a prepared manifest or caller-authored result into evaluation evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.articulated_workspace_clearance import (
    ArticulatedWorkspaceClearanceError,
    validate_door_state_clearance,
    validate_sage_mesh_sweep,
)


HARNESS_SCHEMA_VERSION = "adp009d_franka_eval_harness_manifest.v1"
COUSIN_SCHEMA_VERSION = "adp009d_cousin_manifest.v1"
COUSIN_PACKAGE_RECEIPT_SCHEMA_VERSION = "adp009d_cousin_package_receipt.v1"
COUSIN_STATIC_VALIDATION_SCHEMA_VERSION = (
    "adp009d_cousin_static_validation_receipt.v1"
)
SCENARIO_SUITE_SCHEMA_VERSION = "adp009d_scenario_suite.v1"
SCENARIO_INSTANCE_SCHEMA_VERSION = "adp009d_scenario_instance.v1"
SCENARIO_MATERIALIZATION_SCHEMA_VERSION = "adp009d_scenario_materialization.v1"
TASK_CONSTRUCTION_ADMISSION_SCHEMA_VERSION = "adp_task_construction_admission.v1"
PROGRAM_ID = "arm-decision-proof-v1"
SCENARIO_FREEZE_STATUS = (
    "frozen_after_canonical_canary_before_scenario_evaluation"
)
TASK_DESTINATION_RECEIPT_DIGEST = (
    "sha256:d9f4a32dbc58adfb5e8e1112e30b8b490f7a928b5ade2c3fd1ae8b84bc7aaf79"
)
TASK_DESTINATION_POSITION_M = [
    3.750152333333333,
    -3.4074919,
    0.5264650138348479,
]
PRIOR_CANARY_RECEIPT_DIGESTS = {
    "sha256:fc35d64d3ba255bfb086d74ea8cab327e11664b6d0bfeb6fd468bee686c0b253",
    "sha256:ddac03ff73648f61fee28bc0093f5d4a02a83a9f51916ba50a1265d388a8a7f9",
    "sha256:15385d341dbedf49f75e1b2bd52e52290b1ad841f93ebf9723b2aea14b8e24fc",
}

REQUIRED_ASSET_DIGESTS = {
    "agent_skill_audit": "sha256:aa42ad117ce9885f1d741ee897bfefae2f9d3529ec76d4be9bbc5b161bbe33ec",
    "approved_can": "sha256:61c2a03bef425803d82cc5ef24ced5b2ccb4160923c53bb10c6ad0e3f52532ec",
    "aura_appearance": "sha256:cbb05fc8e6da6ecdb72464f3b115f63e8747e2b67e97c309b4e40952b33000bd",
    "hybrid_seal_receipt": "sha256:dbb19cd7ce3229d58e2a1fafee6ddd042b5f3002d1ab223783382171373e4b1b",
    "original_interiorgs": "sha256:57c71edcb450f2323a5b8ad290b5672b437fc73b9283a7485804ce607da12254",
    "sage_collision": "sha256:b265706c24f6a8ace3ee6743fd138583c4e21d83f61b99a06fd435e6ac2d6b41",
    "sage_usdz": "sha256:bcdc8d36ed88c1a5c4e7cd333479e24c67cde64ed3b3ea135f37028f70d1ebb8",
}

REQUIRED_MANAGERS = {
    "ActionManager",
    "EventManager",
    "ObservationManager",
    "RecorderManager",
    "TerminationManager",
}
EVENT_MODES = {"prestartup", "startup", "reset", "interval"}
REQUIRED_TERMINALS = {
    "success",
    "timeout",
    "drop",
    "support_loss",
    "collision_violation",
    "joint_limit_violation",
    "camera_failure",
    "invalid_observation",
    "policy_runtime_failure",
}
REQUIRED_FAMILIES = {
    "canonical",
    "placement_approach",
    "illumination",
    "camera_sensor",
    "physics",
    "visual_material_cousin",
    "geometric_cousin",
    "held_out_composed",
}
REQUIRED_CONTROLS = {"zero_action_negative", "deterministic_scripted_positive"}
ALLOWED_PARTITIONS = {"development", "qualification", "held_out"}
ALLOWED_SAMPLING = {"fixed", "discrete", "uniform"}
ALLOWED_METRIC_DEPTH_AOVS = {
    "DistanceToCameraSD",
    "DistanceToImagePlaneSD",
    "aurafusion360_expected_camera_depth_m",
    "distance_to_camera",
    "distance_to_image_plane",
}
FORBIDDEN_OUTCOME_KEYS = {
    "episode_success",
    "learned_outcome",
    "policy_result",
    "result_success",
    "task_success",
}
ARTICULATED_TASK_KIND = "articulated_open_close"
REQUIRED_ARTICULATED_CONSTRUCTION_GATES = {
    "source_link_partition",
    "source_visual_removal",
    "replacement_asset",
    # Static admission and a clear sweep describe the asset on paper. The
    # native readback is what proves the joint graph, limits, locked joints,
    # commanded motion, contact, and reset actually behave in the simulator,
    # so it is a required gate rather than a later nicety.
    "native_articulation_readback",
    # A twin can satisfy every part-level check and still open onto a sealed
    # carcass. This gate asks what the open door actually reveals.
    "interior_exposure",
    "native_robot_placement",
    "native_phase_ik",
    "policy_camera_observability",
    "review_camera_observability",
}


class Adp009dHarnessError(ValueError):
    """Stable errors raised when a harness artifact cannot be admitted."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class ScenarioMaterialization:
    receipt: Mapping[str, Any]
    instances: tuple[Mapping[str, Any], ...]


def _clone(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise Adp009dHarnessError([error]) from exc
    if not isinstance(cloned, dict):
        raise Adp009dHarnessError([error])
    return cloned


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _strings(value: Any, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list):
        return []
    result = [str(item).strip() for item in value if str(item).strip()]
    return result if result or not nonempty else []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _under(path: Path, root: Path, *, error: str) -> Path:
    resolved = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise Adp009dHarnessError([error])
    return resolved


def _forbidden_outcome_paths(value: Any, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_OUTCOME_KEYS:
                found.append(path)
            found.extend(_forbidden_outcome_paths(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_outcome_paths(child, prefix=f"{prefix}[{index}]"))
    return found


def _resolve_file_record(
    record: Mapping[str, Any],
    *,
    repo_root: Path,
    evidence_root: Path,
    error_prefix: str,
) -> Path:
    root_id = record.get("root")
    root = repo_root if root_id == "repo" else evidence_root if root_id == "evidence" else None
    if root is None:
        raise Adp009dHarnessError([f"{error_prefix}_root_invalid"])
    relative_path = str(record.get("relative_path") or "")
    if not relative_path:
        raise Adp009dHarnessError([f"{error_prefix}_relative_path_missing"])
    path = _under(root / relative_path, root, error=f"{error_prefix}_outside_root")
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise Adp009dHarnessError([f"{error_prefix}_file_missing"])
    expected_size = record.get("size_bytes")
    if not isinstance(expected_size, int) or expected_size != path.stat().st_size:
        raise Adp009dHarnessError([f"{error_prefix}_size_mismatch"])
    if record.get("sha256") != _sha256(path):
        raise Adp009dHarnessError([f"{error_prefix}_digest_mismatch"])
    return path


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def admit_task_construction(
    *,
    task_contract: Mapping[str, Any],
    member_sweep_clearance: Mapping[str, Any],
    construction_gate_receipts: Sequence[Mapping[str, Any]] = (),
    door_state_clearance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether an articulated task may reach scenario materialization.

    This is the harness admission seam, not an analysis helper.  A blocked
    member sweep rejects the task before placement or spend.  A clear sweep is
    necessary but insufficient: every construction, placement, IK, and camera
    gate must also provide a digest-bound passing receipt.
    """

    contract = _clone(task_contract, error="task_construction_contract_invalid")
    errors: list[str] = []
    if contract.get("schema_version") != "adp_task_spec.v1":
        errors.append("task_construction_contract_schema_invalid")
    if contract.get("task_kind") != ARTICULATED_TASK_KIND:
        errors.append("task_construction_kind_unsupported")
    if not str(contract.get("target_joint_id") or ""):
        errors.append("task_construction_target_joint_missing")
    try:
        clearance = validate_sage_mesh_sweep(member_sweep_clearance)
    except ArticulatedWorkspaceClearanceError as exc:
        errors.extend(exc.errors)
        clearance = {}
    if errors:
        raise Adp009dHarnessError(errors)

    blockers: list[str] = []
    gate_bindings: list[dict[str, Any]] = []
    door_states: dict[str, Any] = {}
    if clearance.get("status") == "blocked_by_exact_sage_mesh_contact":
        obstacle_ids = _strings(clearance.get("collision_prim_paths"))
        blockers.extend(
            f"articulated_member_sweep_obstructed:{obstacle_id}"
            for obstacle_id in obstacle_ids
        )
        if not obstacle_ids:
            blockers.append("articulated_member_sweep_obstructed")
    else:
        # A clear continuous sweep is necessary but not sufficient: the frozen
        # discrete door-state matrix must also be bound with the replacement
        # body, locked lower door, and Franka base classes before any
        # scenario cell may materialize. An unbound class is a blocker, never
        # an implicit clear.
        if door_state_clearance is None:
            blockers.append("articulated_door_state_matrix_missing")
        else:
            try:
                door_states = validate_door_state_clearance(door_state_clearance)
            except ArticulatedWorkspaceClearanceError as exc:
                raise Adp009dHarnessError(list(exc.errors)) from exc
            if door_states.get("status") == "blocked_by_door_state_contact":
                contact = _mapping(door_states.get("first_contact"))
                blockers.append(
                    "articulated_door_state_contact:"
                    f"{contact.get('obstacle_class')}:{contact.get('source')}"
                )
            missing_classes = sorted(
                {"replacement_body", "replacement_lower_door", "franka_base"}
                - {
                    str(item)
                    for item in door_states.get("static_obstacle_classes_bound")
                    or []
                }
            )
            if missing_classes:
                blockers.append(
                    "articulated_door_state_obstacle_classes_unbound:"
                    + ",".join(missing_classes)
                )
        seen_gate_ids: set[str] = set()
        for index, raw in enumerate(construction_gate_receipts):
            if not isinstance(raw, Mapping):
                raise Adp009dHarnessError(
                    [f"task_construction_gate_{index}_invalid"]
                )
            gate = _clone(raw, error=f"task_construction_gate_{index}_invalid")
            gate_id = str(gate.get("gate_id") or "")
            receipt_digest = str(gate.get("receipt_digest") or "")
            if (
                gate_id not in REQUIRED_ARTICULATED_CONSTRUCTION_GATES
                or gate_id in seen_gate_ids
                or gate.get("status") != "passed"
                or not receipt_digest.startswith("sha256:")
                or len(receipt_digest) != 71
            ):
                raise Adp009dHarnessError(
                    [f"task_construction_gate_{index}_invalid"]
                )
            seen_gate_ids.add(gate_id)
            gate_bindings.append(
                {
                    "gate_id": gate_id,
                    "status": "passed",
                    "receipt_digest": receipt_digest,
                }
            )
        blockers.extend(
            f"articulated_construction_gate_missing:{gate_id}"
            for gate_id in sorted(
                REQUIRED_ARTICULATED_CONSTRUCTION_GATES - seen_gate_ids
            )
        )
    authorized = not blockers
    receipt: dict[str, Any] = {
        "schema_version": TASK_CONSTRUCTION_ADMISSION_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "task_contract_digest": canonical_digest(contract),
        "task_kind": contract["task_kind"],
        "target_joint_id": contract["target_joint_id"],
        "member_sweep_clearance_receipt_digest": clearance["receipt_digest"],
        "door_state_clearance_receipt_digest": door_states.get("receipt_digest"),
        "construction_gate_bindings": sorted(
            gate_bindings, key=lambda row: row["gate_id"]
        ),
        "status": "admitted" if authorized else "rejected_or_blocked",
        "scenario_materialization_authorized": authorized,
        "placement_search_authorized": clearance.get("status")
        == "exact_sage_mesh_clearance_candidate_only",
        "blockers": sorted(blockers),
        "learned_policy_outcomes_consulted": False,
        "caller_asserted_success_accepted": False,
        "admission_digest": "",
    }
    receipt["admission_digest"] = canonical_digest(
        receipt, digest_field="admission_digest"
    )
    return receipt


def validate_task_construction_admission(
    value: Mapping[str, Any], *, task_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a retained harness admission receipt before materialization."""

    receipt = _clone(value, error="task_construction_admission_invalid")
    errors: list[str] = []
    if receipt.get("schema_version") != TASK_CONSTRUCTION_ADMISSION_SCHEMA_VERSION:
        errors.append("task_construction_admission_schema_invalid")
    if receipt.get("program_id") != PROGRAM_ID:
        errors.append("task_construction_admission_program_invalid")
    if receipt.get("task_contract_digest") != canonical_digest(task_contract):
        errors.append("task_construction_admission_contract_mismatch")
    if receipt.get("learned_policy_outcomes_consulted") is not False:
        errors.append("task_construction_admission_policy_outcome_leakage")
    if receipt.get("caller_asserted_success_accepted") is not False:
        errors.append("task_construction_admission_caller_success_accepted")
    if receipt.get("admission_digest") != canonical_digest(
        receipt, digest_field="admission_digest"
    ):
        errors.append("task_construction_admission_digest_invalid")
    authorized = receipt.get("scenario_materialization_authorized") is True
    if authorized != (receipt.get("status") == "admitted"):
        errors.append("task_construction_admission_status_inconsistent")
    if authorized and receipt.get("blockers") != []:
        errors.append("task_construction_admission_authorized_with_blockers")
    if errors:
        raise Adp009dHarnessError(errors)
    return receipt


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def validate_harness_manifest(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
    evidence_root: str | Path,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate the reusable manager-based harness manifest."""

    manifest = _clone(value, error="harness_not_json_mapping")
    repo = Path(repo_root).expanduser().resolve()
    evidence = Path(evidence_root).expanduser().resolve()
    errors: list[str] = []
    if manifest.get("schema_version") != HARNESS_SCHEMA_VERSION:
        errors.append("harness_schema_invalid")
    if manifest.get("program_id") != PROGRAM_ID:
        errors.append("harness_program_invalid")
    if manifest.get("reusable_harness") is not True:
        errors.append("harness_reusable_flag_missing")
    if manifest.get("workflow") != "isaac_lab_manager_based":
        errors.append("harness_manager_based_workflow_required")
    if _forbidden_outcome_paths(manifest):
        errors.append("harness_caller_asserted_outcome_forbidden")

    managers = _mapping(manifest.get("managers"))
    if set(managers) != REQUIRED_MANAGERS:
        errors.append("harness_manager_set_invalid")
    events = _rows(managers.get("EventManager", {}).get("terms"))
    event_modes = {str(row.get("mode") or "") for row in events}
    if event_modes != EVENT_MODES:
        errors.append("harness_event_modes_invalid")
    if any(not str(row.get("term_id") or "") for row in events):
        errors.append("harness_event_term_id_missing")

    layers = _mapping(manifest.get("layers"))
    if set(layers) != {"scene", "embodiment", "task", "policy_adapter", "evaluation"}:
        errors.append("harness_layer_set_invalid")
    if _mapping(layers.get("policy_adapter")).get("owns_grading") is not False:
        errors.append("harness_policy_adapter_grading_authority_invalid")
    if _mapping(layers.get("evaluation")).get("orchestration") != (
        "blueprint_evaluation_run_contract"
    ):
        errors.append("harness_competing_orchestration_forbidden")

    assets = _rows(manifest.get("asset_bindings"))
    asset_by_role = {str(row.get("role") or ""): row for row in assets}
    if len(asset_by_role) != len(assets) or set(asset_by_role) != set(REQUIRED_ASSET_DIGESTS):
        errors.append("harness_asset_role_set_invalid")
    for role, expected_digest in REQUIRED_ASSET_DIGESTS.items():
        record = asset_by_role.get(role)
        if record is None:
            continue
        if record.get("sha256") != expected_digest:
            errors.append(f"harness_asset_{role}_identity_invalid")
        if verify_files:
            try:
                _resolve_file_record(
                    record,
                    repo_root=repo,
                    evidence_root=evidence,
                    error_prefix=f"harness_asset_{role}",
                )
            except Adp009dHarnessError as exc:
                errors.extend(exc.errors)

    scene = _mapping(manifest.get("scene"))
    if scene.get("sealed_scene_receipt_digest") != (
        "sha256:b259532be614098a3830aa9945770a96371968f9c68e8087eb21a2ca00e3c3e3"
    ):
        errors.append("harness_sealed_scene_receipt_invalid")
    if scene.get("source_target_collider_prim") != "/Root/ZHQYGJJVAJYEYPTUKY888888":
        errors.append("harness_source_target_collider_invalid")
    if scene.get("source_target_collider_active") is not False:
        errors.append("harness_source_target_collider_not_disabled")
    if scene.get("support_collider_prim") != "/Root/_LTFTHJVAZ3VMPTUJU888888":
        errors.append("harness_support_collider_invalid")
    if scene.get("support_collider_active") is not True:
        errors.append("harness_support_collider_not_active")

    canonical = _mapping(manifest.get("canonical_condition"))
    start = canonical.get("object_start_position_m")
    target = canonical.get("target_position_m")
    if not (
        isinstance(start, list)
        and len(start) == 3
        and all(_number(item) is not None for item in start)
        and isinstance(target, list)
        and len(target) == 3
        and all(_number(item) is not None for item in target)
    ):
        errors.append("harness_canonical_positions_invalid")
    else:
        displacement = math.dist([float(item) for item in start], [float(item) for item in target])
        if displacement < 0.15:
            errors.append("harness_canonical_translation_below_threshold")
        if abs(float(start[2]) - float(target[2])) > 1e-9:
            errors.append("harness_canonical_target_not_same_support_height")
    if canonical.get("immutable") is not True:
        errors.append("harness_canonical_immutability_missing")
    if target != TASK_DESTINATION_POSITION_M:
        errors.append("harness_canonical_destination_identity_invalid")
    target_selection = _mapping(canonical.get("target_selection"))
    if target_selection.get("receipt_digest") != TASK_DESTINATION_RECEIPT_DIGEST:
        errors.append("harness_canonical_destination_receipt_invalid")
    if target_selection.get("policy_outcome_consulted") is not False:
        errors.append("harness_canonical_destination_outcome_blindness_invalid")
    parameters = _mapping(canonical.get("parameters"))
    if [
        parameters.get("target_x_m"),
        parameters.get("target_y_m"),
        parameters.get("target_z_m"),
    ] != TASK_DESTINATION_POSITION_M:
        errors.append("harness_canonical_parameter_destination_invalid")
    if canonical.get("camera_calibration_status") != (
        "v85_dual_view_admissible_exact_can_visible_lossless_media_retained"
    ):
        errors.append("harness_canonical_camera_calibration_status_invalid")

    task = _mapping(manifest.get("task"))
    thresholds = _mapping(task.get("success_thresholds"))
    required_thresholds = {
        "minimum_lift_m": 0.08,
        "minimum_translation_m": 0.15,
        "maximum_center_error_m": 0.05,
        "maximum_tilt_degrees": 15.0,
    }
    for field, expected in required_thresholds.items():
        if _number(thresholds.get(field)) != expected:
            errors.append(f"harness_task_{field}_invalid")
    if thresholds.get("same_support_surface_required") is not True:
        errors.append("harness_task_same_support_required")
    if task.get("grader_authority") != "deterministic_simulator_state":
        errors.append("harness_task_grader_authority_invalid")
    terminals = _mapping(task.get("terminal_definitions"))
    if set(terminals) != REQUIRED_TERMINALS:
        errors.append("harness_terminal_set_invalid")
    for terminal_id, definition in terminals.items():
        if not isinstance(definition, Mapping) or not str(definition.get("condition") or ""):
            errors.append(f"harness_terminal_{terminal_id}_definition_missing")

    embodiment = _mapping(manifest.get("embodiment"))
    if embodiment.get("robot") != "Franka Panda":
        errors.append("harness_robot_identity_invalid")
    if embodiment.get("gripper") != "Robotiq 2F-85":
        errors.append("harness_gripper_identity_invalid")
    asset_cfg = _mapping(embodiment.get("isaac_lab_asset_config"))
    if asset_cfg.get("symbol") != "FRANKA_ROBOTIQ_GRIPPER_CFG":
        errors.append("harness_franka_robotiq_asset_symbol_invalid")
    if asset_cfg.get("source_commit") != "e57379c634b42db5a0fe9f754341be6e2a7c7c43":
        errors.append("harness_franka_robotiq_source_commit_invalid")
    if asset_cfg.get("source_blob") != "f03d21760f903b09438cdccb75d11434abd5a2aa":
        errors.append("harness_franka_robotiq_source_blob_invalid")
    if asset_cfg.get("gripper_variant") != "Robotiq_2F_85":
        errors.append("harness_franka_robotiq_variant_invalid")

    physics = _mapping(manifest.get("physics"))
    if physics.get("centralized") is not True or not isinstance(physics.get("settings"), Mapping):
        errors.append("harness_physx_configuration_not_centralized")
    if physics.get("backend") != "physx":
        errors.append("harness_physics_backend_invalid")
    physics_settings = _mapping(physics.get("settings"))
    required_cooking_settings = {
        "collision_cooking_profile": "legacy_cooker_after_ujitso_stall.v1",
        "collision_cooking_backend": "legacy",
        "ujitso_collision_cooking": False,
        "collision_cooking_decision": (
            "measured v14 UJITSO environment-construction stall; use NVIDIA's "
            "documented legacy-cooker diagnostic without changing collider geometry "
            "or parameters"
        ),
        "collision_cooking_reference": (
            "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/"
            "dev_guide/rigid_bodies_articulations/collision.html"
            "#generate-mesh-colliders-cooking"
        ),
        "collision_geometry_or_parameters_changed": False,
    }
    if any(
        physics_settings.get(key) != value
        for key, value in required_cooking_settings.items()
    ):
        errors.append("harness_physx_collision_cooking_configuration_invalid")
    sage_override = _mapping(_mapping(physics.get("entity_overrides")).get("sealed_sage_static_collision"))
    required_sage_override = {
        "source_sha256": REQUIRED_ASSET_DIGESTS["sage_collision"],
        "source_mesh_count": 165,
        "source_point_count": 509268,
        "source_face_count": 993678,
        "source_rigid_body_count": 0,
        "source_convex_decomposition_count": 164,
        "runtime_task_roi_min_m": [2.4681748, -4.3100837, -0.1],
        "runtime_task_roi_max_m": [4.4681748, -1.9100837, 1.8],
        "runtime_candidate_source_mesh_count": 16,
        "runtime_active_triangle_mesh_count": 15,
        "runtime_source_face_count": 47359,
        "runtime_clipped_source_face_count": 24248,
        "runtime_derived_face_count": 26828,
        "runtime_derived_point_count": 80484,
        "runtime_selected_source_surface_area_m2": 774.974830891,
        "runtime_clipped_source_surface_area_m2": 26.28230143,
        "runtime_maximum_edge_m": 0.5,
        "runtime_approximation": "none",
        "runtime_approximation_semantics": "static_triangle_mesh",
        "runtime_surface_operation": (
            "aabb_clip_then_coplanar_longest_edge_midpoint_retriangulation"
        ),
        "runtime_surface_area_relative_error_maximum": 1.0e-6,
        "runtime_derivative_claim_ceiling": (
            "preregistered_franka_task_envelope_only"
        ),
        "source_target_collider_active": False,
        "support_collider_active": True,
        "geometry_mutation_allowed": False,
        "sealed_source_bytes_retained": True,
        "runtime_surface_preserving_derivative_allowed": True,
        "out_of_envelope_source_colliders_active": False,
        "required_blocker_on_mismatch": "sage_runtime_collision_profile_mismatch",
        "physx_triangle_stability_warning_allowed": False,
        "required_blocker_on_triangle_stability_warning": (
            "physx_collision_stability_warning_detected"
        ),
        "cold_cooking_is_startup_evidence_only": True,
        "warm_start_cache_requirements": [
            "exact sealed SAGE source digest",
            "exact task collision derivative digest",
            "exact Isaac Sim image digest",
            "exact PhysX version and settings",
            "exact live topology revalidation",
            "no PhysX collision fallback or triangle-size stability warning",
        ],
    }
    if sage_override != required_sage_override:
        errors.append("harness_sage_static_triangle_override_invalid")

    renderer = _mapping(manifest.get("observation_renderer"))
    if renderer.get("composition") != "live_depth_correct_aura_dynamic_geometry":
        errors.append("harness_renderer_composition_invalid")
    if renderer.get("missing_blocker") != (
        "sealed_aura_hybrid_policy_observation_renderer_missing"
    ):
        errors.append("harness_renderer_missing_blocker_invalid")
    metric_depth = renderer.get("metric_depth_aov")
    metric_depth_valid = (
        isinstance(metric_depth, Mapping)
        and metric_depth.get("aura") in ALLOWED_METRIC_DEPTH_AOVS
        and metric_depth.get("dynamic") in ALLOWED_METRIC_DEPTH_AOVS
    )
    if not metric_depth_valid:
        errors.append("harness_renderer_metric_depth_invalid")
    if renderer.get("unitless_depth_sd_allowed") is not False:
        errors.append("harness_renderer_unitless_depth_not_forbidden")
    if set(_strings(renderer.get("lossless_policy_cameras"))) != {"external", "wrist"}:
        errors.append("harness_renderer_camera_set_invalid")
    if set(_strings(renderer.get("evaluation_review_cameras"))) != {"overview"}:
        errors.append("harness_renderer_review_camera_set_invalid")
    overview = _mapping(renderer.get("overview_camera_contract"))
    if overview != {
        "camera_id": "overview",
        "required_for_every_new_simulator_evaluation_episode": True,
        "review_only": True,
        "policy_input": False,
        "grader_input": False,
        "lossless_frames_calibration_timestamps_manifest_and_video_required": True,
    }:
        errors.append("harness_renderer_overview_camera_contract_invalid")
    if set(_strings(renderer.get("forbidden_fallbacks"))) != {
        "browser_render",
        "old_2d_overlay",
        "plain_sage_mesh_appearance",
        "unrelated_frames",
    }:
        errors.append("harness_renderer_fallback_set_invalid")

    controls = _mapping(manifest.get("controls"))
    if set(_strings(controls.get("required"))) != REQUIRED_CONTROLS:
        errors.append("harness_controls_required_set_invalid")
    if controls.get("execution_order") != [
        "zero_action_negative",
        "deterministic_scripted_positive",
    ]:
        errors.append("harness_controls_execution_order_invalid")
    if controls.get("same_instance_digest_required") is not True:
        errors.append("harness_controls_instance_binding_missing")
    if controls.get("positive_failure_effect") != "block_cell_before_policy_execution":
        errors.append("harness_controls_positive_failure_effect_invalid")
    if controls.get("media_contract") != (
        "lossless_external_and_wrist_frames_manifest_and_review_video"
    ):
        errors.append("harness_controls_media_contract_invalid")
    if controls.get("grader_authority") != "deterministic_simulator_state":
        errors.append("harness_controls_grader_authority_invalid")

    candidate_pair = _mapping(manifest.get("candidate_pair"))
    if candidate_pair.get("candidate_ids") != ["pi05_droid", "groot_n17_droid"]:
        errors.append("harness_candidate_pair_invalid")
    if candidate_pair.get("exactly_two") is not True:
        errors.append("harness_candidate_pair_cardinality_invalid")
    if candidate_pair.get("same_resolved_cells_and_seeds") is not True:
        errors.append("harness_candidate_pairing_rule_invalid")
    if candidate_pair.get("frozen_before_scenario_evaluation") is not True:
        errors.append("harness_candidate_pair_freeze_invalid")

    timing_receipt = _mapping(manifest.get("runtime_timing_receipt"))
    required_timing_contract = {
        "required": True,
        "fields_seconds": [
            "environment_build",
            "reset_0",
            "reset_1",
            "zero_action_step",
            "camera_warmup_40_frames",
            "camera_retention",
        ],
        "warm_start_claim_requires_measured_result": True,
        "cold_install_time_reported_separately": True,
    }
    if any(
        timing_receipt.get(key) != expected
        for key, expected in required_timing_contract.items()
    ):
        errors.append("harness_runtime_timing_receipt_invalid")
    latest_timing = _mapping(timing_receipt.get("latest_measurement"))
    required_measured_seconds = set(required_timing_contract["fields_seconds"])
    measured_seconds_valid = all(
        isinstance(latest_timing.get(field), (int, float))
        and math.isfinite(float(latest_timing[field]))
        and float(latest_timing[field]) >= 0.0
        for field in required_measured_seconds
    )
    if not (
        latest_timing.get("run_id")
        == "native_microcheck_v22_contact_clear_base"
        and latest_timing.get("result_sha256")
        == "sha256:1c01933c7fdfb4fc6177ef8db5fe18507db5304ea3777cc67d3852488c62e1c9"
        and latest_timing.get("status")
        == "completed_contact_clear_reset_step_and_camera_tensor_microcheck"
        and latest_timing.get("provider_zero_observed") is True
        and latest_timing.get("policy_queried") is False
        and measured_seconds_valid
    ):
        errors.append("harness_runtime_timing_measurement_invalid")

    runtime = _mapping(manifest.get("runtime_pins"))
    if runtime.get("isaac_lab_arena") != "8b4a3a47fc53de23e8205089d71109a2e2348acd":
        errors.append("harness_arena_pin_invalid")
    if runtime.get("isaac_lab") != "e57379c634b42db5a0fe9f754341be6e2a7c7c43":
        errors.append("harness_isaac_lab_pin_invalid")
    if runtime.get("isaac_sim_family") != "6.0.x":
        errors.append("harness_isaac_sim_pin_invalid")
    if runtime.get("arena_claim_ceiling") != "alpha_internal_rehearsal_only":
        errors.append("harness_arena_claim_ceiling_invalid")

    if manifest.get("harness_digest") != canonical_digest(
        manifest, digest_field="harness_digest"
    ):
        errors.append("harness_digest_mismatch")
    if errors:
        raise Adp009dHarnessError(errors)
    return manifest


def validate_cousin_manifest(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate one separate digest-bound can-family cousin."""

    manifest = _clone(value, error="cousin_not_json_mapping")
    repo = Path(repo_root).expanduser().resolve()
    errors: list[str] = []
    if manifest.get("schema_version") != COUSIN_SCHEMA_VERSION:
        errors.append("cousin_schema_invalid")
    if manifest.get("program_id") != PROGRAM_ID:
        errors.append("cousin_program_invalid")
    if manifest.get("cousin_type") not in {"visual_material", "geometric"}:
        errors.append("cousin_type_invalid")
    if manifest.get("canonical_anchor") is not False:
        errors.append("cousin_canonical_anchor_must_be_false")
    if manifest.get("base_asset_sha256") != REQUIRED_ASSET_DIGESTS["approved_can"]:
        errors.append("cousin_base_asset_invalid")
    if manifest.get("admission_status") not in {
        "static_candidate",
        "admitted_for_control_execution",
        "control_invalidated",
    }:
        errors.append("cousin_admission_status_invalid")
    for field in ("dimensions_m", "physics", "collider", "material_provenance", "rights"):
        if not isinstance(manifest.get(field), Mapping):
            errors.append(f"cousin_{field}_missing")
    if not _strings(manifest.get("claim_ceiling"), nonempty=True):
        errors.append("cousin_claim_ceiling_missing")
    base = _mapping(manifest.get("base_asset"))
    overlay = _mapping(manifest.get("overlay_asset"))
    if verify_files:
        for role, record in (("base", base), ("overlay", overlay)):
            try:
                _resolve_file_record(
                    record,
                    repo_root=repo,
                    evidence_root=repo,
                    error_prefix=f"cousin_{role}",
                )
            except Adp009dHarnessError as exc:
                errors.extend(exc.errors)
    if base.get("sha256") != REQUIRED_ASSET_DIGESTS["approved_can"]:
        errors.append("cousin_base_file_identity_invalid")
    if overlay.get("sha256") != manifest.get("usd_sha256"):
        errors.append("cousin_overlay_identity_invalid")
    materialization = _mapping(manifest.get("materialization"))
    expected_usd = _mapping(materialization.get("expected_self_contained_usd"))
    if materialization.get("method") != "usd_flatten_and_bake_v1":
        errors.append("cousin_materialization_method_invalid")
    if expected_usd.get("filename") != f"{manifest.get('cousin_id')}.usda":
        errors.append("cousin_materialized_filename_invalid")
    if (
        not isinstance(expected_usd.get("size_bytes"), int)
        or expected_usd.get("size_bytes", 0) <= 0
        or not str(expected_usd.get("sha256") or "").startswith("sha256:")
    ):
        errors.append("cousin_materialized_identity_invalid")
    static_validation = _mapping(manifest.get("static_validation"))
    if static_validation.get("status") != "external_observed_receipt_required":
        errors.append("cousin_static_validation_status_invalid")
    if static_validation.get("profile") != "Prop-Robotics-Physx":
        errors.append("cousin_static_validation_profile_invalid")
    if static_validation.get("profile_version") != "2.0.0":
        errors.append("cousin_static_validation_profile_version_invalid")
    if "profile_passed" in static_validation:
        errors.append("cousin_manifest_caller_asserted_static_pass_forbidden")
    if manifest.get("cousin_digest") != canonical_digest(
        manifest, digest_field="cousin_digest"
    ):
        errors.append("cousin_digest_mismatch")
    if errors:
        raise Adp009dHarnessError(errors)
    return manifest


def materialize_cousin_package(
    *,
    cousin_manifest: Mapping[str, Any],
    repo_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create one self-contained USD for validation and runtime staging.

    The checked-in cousin layer is a small, reviewable authoring recipe.  The
    admitted runtime object must not depend on that recipe or on the sealed can
    through a second USD file: the SimReady package profile admits exactly one
    USD file below the asset directory.  Geometric cousin scale is baked into
    point, normal, extent, and curve-width data so the resulting rigid body has
    an identity root transform.
    """

    repo = Path(repo_root).expanduser().resolve()
    manifest = validate_cousin_manifest(cousin_manifest, repo_root=repo, verify_files=True)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise Adp009dHarnessError(["cousin_package_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    base_record = _mapping(manifest["base_asset"])
    overlay_record = _mapping(manifest["overlay_asset"])
    base_path = _resolve_file_record(
        base_record,
        repo_root=repo,
        evidence_root=repo,
        error_prefix="cousin_base",
    )
    overlay_path = _resolve_file_record(
        overlay_record,
        repo_root=repo,
        evidence_root=repo,
        error_prefix="cousin_overlay",
    )
    authored_name = f"{manifest['cousin_id']}.usda"
    authored_path = output / authored_name
    authoring = _author_flattened_cousin(
        base_path=base_path,
        overlay_path=overlay_path,
        output_path=authored_path,
        cousin_type=str(manifest["cousin_type"]),
        dimensions=_mapping(manifest.get("dimensions_m")),
    )
    expected_usd = _mapping(
        _mapping(manifest.get("materialization")).get("expected_self_contained_usd")
    )
    authored_sha256 = _sha256(authored_path)
    if (
        authored_path.name != expected_usd.get("filename")
        or authored_path.stat().st_size != expected_usd.get("size_bytes")
        or authored_sha256 != expected_usd.get("sha256")
    ):
        raise Adp009dHarnessError(["cousin_materialized_usd_identity_mismatch"])
    receipt = {
        "schema_version": COUSIN_PACKAGE_RECEIPT_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "cousin_id": manifest["cousin_id"],
        "cousin_digest": manifest["cousin_digest"],
        "package_root": str(output),
        "root_layer": authored_path.name,
        "authoring": authoring,
        "source_recipe_files": [
            {
                "role": "sealed_base",
                "relative_path": str(base_record["relative_path"]),
                "size_bytes": base_path.stat().st_size,
                "sha256": _sha256(base_path),
            },
            {
                "role": "cousin_overlay_recipe",
                "relative_path": str(overlay_record["relative_path"]),
                "size_bytes": overlay_path.stat().st_size,
                "sha256": _sha256(overlay_path),
            },
        ],
        "files": [
            {
                "relative_path": authored_path.name,
                "size_bytes": authored_path.stat().st_size,
                "sha256": authored_sha256,
            },
        ],
        "caller_asserted_validation": False,
        "package_receipt_digest": "",
    }
    receipt["package_receipt_digest"] = canonical_digest(
        receipt, digest_field="package_receipt_digest"
    )
    write_json(output / "adp009d_cousin_package_receipt.v1.json", receipt)
    return receipt


def _author_flattened_cousin(
    *,
    base_path: Path,
    overlay_path: Path,
    output_path: Path,
    cousin_type: str,
    dimensions: Mapping[str, Any],
) -> dict[str, Any]:
    """Flatten a cousin recipe and bake any geometric scale into geometry."""

    try:
        from pxr import Gf, Usd, UsdGeom, UsdUtils
    except ImportError as exc:  # pragma: no cover - exercised in runtime image admission
        raise Adp009dHarnessError(["cousin_authoring_pxr_unavailable"]) from exc

    stage = Usd.Stage.Open(str(overlay_path))
    if stage is None:
        raise Adp009dHarnessError(["cousin_overlay_usd_open_failed"])
    dependencies = {
        str(layer.realPath or layer.identifier)
        for layer in stage.GetLayerStack()
        if not layer.anonymous
    }
    if str(base_path) not in dependencies or str(overlay_path) not in dependencies:
        raise Adp009dHarnessError(["cousin_overlay_dependency_identity_invalid"])
    flattened = stage.Flatten()
    if flattened is None or not flattened.Export(str(output_path)):
        raise Adp009dHarnessError(["cousin_flatten_export_failed"])

    authored = Usd.Stage.Open(str(output_path))
    if authored is None:
        raise Adp009dHarnessError(["cousin_flattened_usd_open_failed"])
    scale = tuple(float(value) for value in dimensions.get("scale_xyz", (1, 1, 1)))
    if len(scale) != 3 or any(not math.isfinite(value) or value <= 0 for value in scale):
        raise Adp009dHarnessError(["cousin_geometry_scale_invalid"])
    if cousin_type == "visual_material" and scale != (1.0, 1.0, 1.0):
        raise Adp009dHarnessError(["cousin_visual_geometry_must_be_unchanged"])

    points_baked = 0
    normals_baked = 0
    extents_baked = 0
    widths_baked = 0
    if cousin_type == "geometric":
        for prim in authored.Traverse():
            points = prim.GetAttribute("points")
            if points and points.HasAuthoredValueOpinion():
                values = points.Get() or []
                points.Set(
                    [
                        Gf.Vec3f(
                            value[0] * scale[0],
                            value[1] * scale[1],
                            value[2] * scale[2],
                        )
                        for value in values
                    ]
                )
                points_baked += len(values)
            normals = prim.GetAttribute("normals")
            if normals and normals.HasAuthoredValueOpinion():
                values = normals.Get() or []
                baked_normals = []
                for value in values:
                    normal = Gf.Vec3f(
                        value[0] / scale[0],
                        value[1] / scale[1],
                        value[2] / scale[2],
                    )
                    normal.Normalize()
                    baked_normals.append(normal)
                normals.Set(baked_normals)
                normals_baked += len(values)
            extent = prim.GetAttribute("extent")
            if extent and extent.HasAuthoredValueOpinion():
                values = extent.Get() or []
                extent.Set(
                    [
                        Gf.Vec3f(
                            value[0] * scale[0],
                            value[1] * scale[1],
                            value[2] * scale[2],
                        )
                        for value in values
                    ]
                )
                extents_baked += len(values)
            widths = prim.GetAttribute("widths")
            if widths and widths.HasAuthoredValueOpinion():
                values = widths.Get() or []
                radial_scale = math.sqrt(scale[0] * scale[1])
                widths.Set([float(value) * radial_scale for value in values])
                widths_baked += len(values)

        root_prim = authored.GetDefaultPrim()
        if not root_prim:
            raise Adp009dHarnessError(["cousin_default_prim_missing"])
        UsdGeom.Xformable(root_prim).ClearXformOpOrder()
        root_prim.RemoveProperty("xformOp:scale")

    metadata = dict(authored.GetRootLayer().customLayerData or {})
    legacy_metadata = metadata.pop("simready_metadata", None)
    if "SimReady_Metadata" not in metadata and isinstance(legacy_metadata, Mapping):
        metadata["SimReady_Metadata"] = dict(legacy_metadata)
    if "SimReady_Metadata" not in metadata:
        raise Adp009dHarnessError(["cousin_simready_metadata_missing"])
    authored.GetRootLayer().customLayerData = metadata
    # Flatten writes the composed root layer's absolute path into the output
    # doc string, which would bind the materialization digest to the checkout
    # directory: verified only where it was authored, failed everywhere else.
    # Identity must be a property of the sealed inputs alone.
    authored.GetRootLayer().documentation = (
        "adp009d cousin flattened from the sealed base and recipe overlay; "
        "source identity is carried by the manifest, not by paths"
    )
    authored.GetRootLayer().Save()
    authored_text = output_path.read_bytes()
    for source_root in {str(base_path.parent), str(overlay_path.parent), str(output_path.parent)}:
        if source_root.encode("utf-8") in authored_text:
            raise Adp009dHarnessError(
                [f"cousin_authored_usd_embeds_absolute_path:{source_root}"]
            )

    verify = Usd.Stage.Open(str(output_path))
    if verify is None:
        raise Adp009dHarnessError(["cousin_authored_usd_reopen_failed"])
    dependency_layers, dependency_assets, unresolved = UsdUtils.ComputeAllDependencies(
        str(output_path)
    )
    if (
        len(dependency_layers) != 1
        or dependency_assets
        or unresolved
        or list(verify.GetRootLayer().subLayerPaths)
    ):
        raise Adp009dHarnessError(["cousin_authored_usd_not_self_contained"])
    root_prim = verify.GetDefaultPrim()
    root_transform = UsdGeom.Xformable(root_prim).GetLocalTransformation()
    root_transform_identity = root_transform == Gf.Matrix4d(1.0)
    if cousin_type == "geometric" and not root_transform_identity:
        raise Adp009dHarnessError(["cousin_geometric_root_transform_not_identity"])
    return {
        "method": "usd_flatten_and_bake_v1",
        "self_contained_usd": True,
        "root_transform_identity": root_transform_identity,
        "scale_baked_xyz": list(scale),
        "points_baked": points_baked,
        "normals_baked": normals_baked,
        "extent_values_baked": extents_baked,
        "curve_width_values_baked": widths_baked,
    }


def admit_cousin_static_validation(
    *,
    cousin_manifest: Mapping[str, Any],
    package_receipt: Mapping[str, Any],
    repo_root: str | Path,
    validator_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Admit an observed SimReady report; caller-authored pass flags are ignored."""

    repo = Path(repo_root).expanduser().resolve()
    manifest = validate_cousin_manifest(
        cousin_manifest, repo_root=repo, verify_files=True
    )
    package = _clone(package_receipt, error="cousin_package_receipt_not_mapping")
    errors: list[str] = []
    if package.get("schema_version") != COUSIN_PACKAGE_RECEIPT_SCHEMA_VERSION:
        errors.append("cousin_package_receipt_schema_invalid")
    if package.get("program_id") != PROGRAM_ID:
        errors.append("cousin_package_receipt_program_invalid")
    if package.get("cousin_id") != manifest.get("cousin_id"):
        errors.append("cousin_package_receipt_cousin_mismatch")
    if package.get("cousin_digest") != manifest.get("cousin_digest"):
        errors.append("cousin_package_receipt_manifest_digest_mismatch")
    if package.get("caller_asserted_validation") is not False:
        errors.append("cousin_package_caller_asserted_validation_forbidden")
    if package.get("package_receipt_digest") != canonical_digest(
        package, digest_field="package_receipt_digest"
    ):
        errors.append("cousin_package_receipt_digest_mismatch")

    package_root = Path(str(package.get("package_root") or "")).expanduser().resolve()
    authored_path = package_root / str(package.get("root_layer") or "")
    package_receipt_path = package_root / "adp009d_cousin_package_receipt.v1.json"
    package_files = _rows(package.get("files"))
    if (
        len(package_files) != 1
        or package_files[0].get("relative_path") != authored_path.name
        or not authored_path.is_file()
    ):
        errors.append("cousin_package_authored_usd_missing")
    elif (
        package_files[0].get("size_bytes") != authored_path.stat().st_size
        or package_files[0].get("sha256") != _sha256(authored_path)
    ):
        errors.append("cousin_package_authored_usd_identity_mismatch")
    try:
        persisted_package = json.loads(package_receipt_path.read_text(encoding="utf-8"))
        if persisted_package != package:
            errors.append("cousin_package_persisted_receipt_mismatch")
    except (OSError, json.JSONDecodeError):
        errors.append("cousin_package_persisted_receipt_missing")

    report_path = Path(validator_report_path).expanduser().resolve()
    try:
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise Adp009dHarnessError(
            sorted(set(errors + ["cousin_static_validator_report_invalid"]))
        ) from exc
    if not isinstance(report, Mapping) or len(report) != 1:
        errors.append("cousin_static_validator_report_shape_invalid")
        result: dict[str, Any] = {}
        report_asset = ""
    else:
        report_asset, raw_result = next(iter(report.items()))
        result = _mapping(raw_result)
    try:
        if Path(str(report_asset)).expanduser().resolve() != authored_path:
            errors.append("cousin_static_validator_asset_path_mismatch")
    except (OSError, RuntimeError, ValueError):
        errors.append("cousin_static_validator_asset_path_invalid")
    if result.get("profile_id") != "Prop-Robotics-Physx":
        errors.append("cousin_static_validator_profile_invalid")
    if result.get("profile_version") != "2.0.0":
        errors.append("cousin_static_validator_profile_version_invalid")
    features = _mapping(result.get("features_summary"))
    if not features:
        errors.append("cousin_static_validator_features_missing")
    failing_features = sorted(
        feature_id
        for feature_id, feature in features.items()
        if _mapping(feature).get("passed") is not True
    )
    if failing_features:
        errors.append("cousin_static_validator_profile_failed")
    if errors:
        raise Adp009dHarnessError(errors)

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise Adp009dHarnessError(["cousin_static_validation_output_exists"])
    receipt = {
        "schema_version": COUSIN_STATIC_VALIDATION_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "cousin_id": manifest["cousin_id"],
        "cousin_digest": manifest["cousin_digest"],
        "package_receipt_digest": package["package_receipt_digest"],
        "package_receipt": {
            "path": str(package_receipt_path),
            "size_bytes": package_receipt_path.stat().st_size,
            "sha256": _sha256(package_receipt_path),
        },
        "authored_usd": {
            "path": str(authored_path),
            "size_bytes": authored_path.stat().st_size,
            "sha256": _sha256(authored_path),
        },
        "validator": {
            "repository_url": "https://github.com/NVIDIA/simready-foundation",
            "commit": "a1e9dd68ee2d107f74dc6cd6da875b54ad3f8fd3",
            "tree": "3794fa2eeb96e3e48c3f4874f638f6ab5a9636d5",
            "cli": "simready-validate",
            "cli_version": "2026.6.4",
            "profile": "Prop-Robotics-Physx",
            "profile_version": "2.0.0",
            "report_path": str(report_path),
            "report_size_bytes": len(report_bytes),
            "report_sha256": "sha256:" + hashlib.sha256(report_bytes).hexdigest(),
        },
        "validation_logs": [
            {
                "path": str(log_path),
                "size_bytes": log_path.stat().st_size,
                "sha256": _sha256(log_path),
            }
            for log_path in (
                report_path.parent / "simready_validate.stdout.log",
                report_path.parent / "simready_validate.stderr.log",
            )
            if log_path.is_file()
        ],
        "observed_feature_results": {
            feature_id: {
                "passed": True,
                "version": _mapping(feature).get("version"),
            }
            for feature_id, feature in sorted(features.items())
        },
        "profile_passed": True,
        "admission_effect": "static_profile_only_native_controls_still_required",
        "caller_asserted_success_accepted": False,
        "validation_receipt_digest": "",
    }
    receipt["validation_receipt_digest"] = canonical_digest(
        receipt, digest_field="validation_receipt_digest"
    )
    write_json(output, receipt)
    return receipt


def validate_cousin_static_validation_receipt(
    value: Mapping[str, Any],
    *,
    cousin_manifest: Mapping[str, Any],
    verify_files: bool = True,
) -> dict[str, Any]:
    """Re-derive a static cousin gate from retained USD and validator bytes."""

    receipt = _clone(value, error="cousin_static_validation_receipt_not_mapping")
    errors: list[str] = []
    if receipt.get("schema_version") != COUSIN_STATIC_VALIDATION_SCHEMA_VERSION:
        errors.append("cousin_static_validation_receipt_schema_invalid")
    if receipt.get("program_id") != PROGRAM_ID:
        errors.append("cousin_static_validation_receipt_program_invalid")
    if receipt.get("cousin_id") != cousin_manifest.get("cousin_id"):
        errors.append("cousin_static_validation_receipt_cousin_mismatch")
    if receipt.get("cousin_digest") != cousin_manifest.get("cousin_digest"):
        errors.append("cousin_static_validation_receipt_manifest_digest_mismatch")
    if receipt.get("profile_passed") is not True:
        errors.append("cousin_static_validation_receipt_profile_not_passed")
    if receipt.get("caller_asserted_success_accepted") is not False:
        errors.append("cousin_static_validation_receipt_caller_success_invalid")
    validator = _mapping(receipt.get("validator"))
    if (
        validator.get("repository_url")
        != "https://github.com/NVIDIA/simready-foundation"
        or validator.get("commit")
        != "a1e9dd68ee2d107f74dc6cd6da875b54ad3f8fd3"
        or validator.get("tree")
        != "3794fa2eeb96e3e48c3f4874f638f6ab5a9636d5"
        or validator.get("cli_version") != "2026.6.4"
        or validator.get("profile") != "Prop-Robotics-Physx"
        or validator.get("profile_version") != "2.0.0"
    ):
        errors.append("cousin_static_validation_receipt_validator_identity_invalid")
    observed = _mapping(receipt.get("observed_feature_results"))
    required_features = {
        "FET000_CORE",
        "FET001_BASE_NEUTRAL",
        "FET003_BASE_NEUTRAL",
        "FET003_BASE_PHYSX",
        "FET004_BASE_NEUTRAL",
        "FET004_BASE_PHYSX",
        "FET005_BASE_NEUTRAL",
        "FET006_BASE_MDL",
    }
    if set(observed) != required_features or any(
        _mapping(result).get("passed") is not True for result in observed.values()
    ):
        errors.append("cousin_static_validation_receipt_feature_results_invalid")
    if receipt.get("validation_receipt_digest") != canonical_digest(
        receipt, digest_field="validation_receipt_digest"
    ):
        errors.append("cousin_static_validation_receipt_digest_mismatch")

    if verify_files:
        authored = _mapping(receipt.get("authored_usd"))
        package_record = _mapping(receipt.get("package_receipt"))
        report_path = Path(str(validator.get("report_path") or "")).expanduser().resolve()
        for label, record in (
            ("authored_usd", authored),
            ("package_receipt", package_record),
            (
                "validator_report",
                {
                    "path": str(report_path),
                    "size_bytes": validator.get("report_size_bytes"),
                    "sha256": validator.get("report_sha256"),
                },
            ),
        ):
            path = Path(str(record.get("path") or "")).expanduser().resolve()
            if (
                not path.is_file()
                or record.get("size_bytes") != path.stat().st_size
                or record.get("sha256") != _sha256(path)
            ):
                errors.append(f"cousin_static_validation_{label}_identity_invalid")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report, Mapping) or len(report) != 1:
                raise ValueError("report shape")
            report_asset, result = next(iter(report.items()))
            result = _mapping(result)
            if Path(str(report_asset)).expanduser().resolve() != Path(
                str(authored.get("path") or "")
            ).expanduser().resolve():
                errors.append("cousin_static_validation_report_asset_mismatch")
            report_features = _mapping(result.get("features_summary"))
            if (
                result.get("profile_id") != "Prop-Robotics-Physx"
                or result.get("profile_version") != "2.0.0"
                or set(report_features) != required_features
                or any(
                    _mapping(feature).get("passed") is not True
                    for feature in report_features.values()
                )
            ):
                errors.append("cousin_static_validation_report_result_invalid")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            errors.append("cousin_static_validation_report_invalid")

    if errors:
        raise Adp009dHarnessError(errors)
    return receipt


def _expanded_cells(suite: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for template in _rows(suite.get("cell_templates")):
        template_id = str(template.get("template_id") or "")
        seeds = template.get("seeds")
        if not isinstance(seeds, list):
            continue
        for seed in seeds:
            cells.append(
                {
                    "cell_id": f"{template_id}.seed_{seed}",
                    "template_id": template_id,
                    "family": template.get("family"),
                    "partition": template.get("partition"),
                    "scored": template.get("scored"),
                    "seed": seed,
                    "factor_ids": list(template.get("factor_ids") or []),
                    "cousin_id": template.get("cousin_id"),
                }
            )
    return cells


def _required_paired_n(
    *, alpha: float, power: float, effect: float, discordance: float
) -> int:
    if not (0 < alpha < 1 and 0 < power < 1 and 0 < effect < discordance <= 1):
        raise Adp009dHarnessError(["scenario_power_inputs_invalid"])
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    variance_under_alternative = discordance - effect * effect
    if variance_under_alternative <= 0:
        raise Adp009dHarnessError(["scenario_power_variance_invalid"])
    numerator = (
        z_alpha * math.sqrt(discordance)
        + z_power * math.sqrt(variance_under_alternative)
    ) ** 2
    return math.ceil(numerator / (effect * effect))


def _wilson_worst_half_width(n: int, *, alpha: float) -> float:
    if n <= 0 or not 0 < alpha < 1:
        return float("inf")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    return z * math.sqrt(0.25 / n + z * z / (4 * n * n)) / (1 + z * z / n)


def _factor_allowed(factor: Mapping[str, Any], value: Any) -> bool:
    allowed = _mapping(factor.get("allowed"))
    if "values" in allowed:
        return isinstance(allowed["values"], list) and value in allowed["values"]
    number = _number(value)
    lower = _number(allowed.get("minimum"))
    upper = _number(allowed.get("maximum"))
    return number is not None and lower is not None and upper is not None and lower <= number <= upper


def _sample_factor(
    factor: Mapping[str, Any], *, suite_digest: str, cell_id: str, seed: int
) -> tuple[Any, str]:
    parameter_id = str(factor.get("parameter_id") or "")
    resolved_seed_digest = canonical_digest(
        {
            "suite_digest": suite_digest,
            "cell_id": cell_id,
            "factor_id": parameter_id,
            "seed": seed,
        }
    )
    integer_seed = int(resolved_seed_digest.removeprefix("sha256:")[:16], 16)
    rng = random.Random(integer_seed)
    sampling = _mapping(factor.get("sampling"))
    kind = sampling.get("kind")
    if kind == "fixed":
        value = factor.get("nominal_value")
    elif kind == "discrete":
        values = list(
            sampling.get("values")
            or _mapping(factor.get("allowed")).get("values")
            or []
        )
        if not values:
            raise Adp009dHarnessError([f"scenario_factor_{parameter_id}_discrete_empty"])
        value = values[rng.randrange(len(values))]
    elif kind == "uniform":
        allowed = _mapping(factor.get("allowed"))
        lower = _number(allowed.get("minimum"))
        upper = _number(allowed.get("maximum"))
        if lower is None or upper is None or lower > upper:
            raise Adp009dHarnessError([f"scenario_factor_{parameter_id}_range_invalid"])
        decimals = sampling.get("decimals", 8)
        if not isinstance(decimals, int) or not 0 <= decimals <= 12:
            raise Adp009dHarnessError([f"scenario_factor_{parameter_id}_decimals_invalid"])
        value = round(lower + (upper - lower) * rng.random(), decimals)
    else:
        raise Adp009dHarnessError([f"scenario_factor_{parameter_id}_sampling_invalid"])
    if not _factor_allowed(factor, value):
        raise Adp009dHarnessError([f"scenario_factor_{parameter_id}_resolved_out_of_bounds"])
    return value, resolved_seed_digest


def validate_scenario_suite(
    value: Mapping[str, Any],
    *,
    harness_manifest: Mapping[str, Any],
    cousin_manifests: Sequence[Mapping[str, Any]],
    cousin_static_validation_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a bounded, explicit, pre-outcome scenario suite."""

    suite = _clone(value, error="scenario_suite_not_json_mapping")
    errors: list[str] = []
    if suite.get("schema_version") != SCENARIO_SUITE_SCHEMA_VERSION:
        errors.append("scenario_suite_schema_invalid")
    if suite.get("program_id") != PROGRAM_ID:
        errors.append("scenario_suite_program_invalid")
    if suite.get("freeze_status") != SCENARIO_FREEZE_STATUS:
        errors.append("scenario_suite_not_frozen_before_scenario_evaluation")
    disclosure = _mapping(suite.get("prior_canary_disclosure"))
    if disclosure.get("scope") != "canonical_smoke_canaries_only":
        errors.append("scenario_suite_prior_canary_scope_invalid")
    if set(_strings(disclosure.get("retained_receipt_digests"))) != (
        PRIOR_CANARY_RECEIPT_DIGESTS
    ):
        errors.append("scenario_suite_prior_canary_receipts_invalid")
    if disclosure.get("prior_outcomes_used_to_select_parameters") is not False:
        errors.append("scenario_suite_prior_outcome_parameter_selection_invalid")
    if disclosure.get("scenario_family_results_observed") is not False:
        errors.append("scenario_suite_prior_scenario_results_invalid")
    if disclosure.get("next_learned_run_requires_frozen_suite_digest") is not True:
        errors.append("scenario_suite_next_run_freeze_binding_missing")
    if disclosure.get("claim_ceiling") != (
        "post_canary_preregistered_scenario_evaluation_not_prospective_from_first_learned_contact"
    ):
        errors.append("scenario_suite_prior_canary_claim_ceiling_invalid")
    if suite.get("harness_digest") != harness_manifest.get("harness_digest"):
        errors.append("scenario_suite_harness_digest_mismatch")
    if _forbidden_outcome_paths(suite):
        errors.append("scenario_suite_caller_asserted_outcome_forbidden")
    if set(_strings(suite.get("required_controls"))) != REQUIRED_CONTROLS:
        errors.append("scenario_suite_controls_invalid")
    if suite.get("cartesian_product_allowed") is not False:
        errors.append("scenario_suite_cartesian_product_not_forbidden")

    cousin_by_id = {
        str(row.get("cousin_id") or ""): row for row in cousin_manifests
    }
    if set(cousin_by_id) != {
        "adp009d_visual_material_cousin",
        "adp009d_geometric_cousin",
    }:
        errors.append("scenario_suite_cousin_set_invalid")
    for cousin_id, cousin in cousin_by_id.items():
        if cousin.get("admission_status") not in {
            "admitted_for_control_execution",
            "static_candidate",
        }:
            errors.append(f"scenario_suite_cousin_{cousin_id}_not_control_admissible")
    static_receipt_by_id = {
        str(row.get("cousin_id") or ""): row
        for row in cousin_static_validation_receipts
        if isinstance(row, Mapping)
    }
    if set(static_receipt_by_id) != set(cousin_by_id):
        errors.append("scenario_suite_cousin_static_receipt_set_invalid")
    else:
        for cousin_id, cousin in cousin_by_id.items():
            try:
                validate_cousin_static_validation_receipt(
                    static_receipt_by_id[cousin_id],
                    cousin_manifest=cousin,
                    verify_files=True,
                )
            except Adp009dHarnessError as exc:
                errors.extend(
                    f"scenario_suite_{cousin_id}_{error}" for error in exc.errors
                )

    factor_rows = _rows(suite.get("factors"))
    factors = {str(row.get("parameter_id") or ""): row for row in factor_rows}
    if len(factors) != len(factor_rows) or "" in factors:
        errors.append("scenario_suite_factor_id_duplicate_or_missing")
    required_factor_fields = {
        "semantic_meaning",
        "unit",
        "nominal_value",
        "allowed",
        "sampling",
        "source",
        "reason",
        "runtime_target",
        "affects",
        "validity",
    }
    for factor_id, factor in factors.items():
        if not required_factor_fields.issubset(factor):
            errors.append(f"scenario_factor_{factor_id}_fields_missing")
        if _mapping(factor.get("sampling")).get("kind") not in ALLOWED_SAMPLING:
            errors.append(f"scenario_factor_{factor_id}_sampling_invalid")
        sampling = _mapping(factor.get("sampling"))
        if sampling.get("kind") == "discrete" and "values" in sampling:
            values = sampling.get("values")
            if not isinstance(values, list) or not values or any(
                not _factor_allowed(factor, item) for item in (values or [])
            ):
                errors.append(f"scenario_factor_{factor_id}_discrete_values_invalid")
        if not _factor_allowed(factor, factor.get("nominal_value")):
            errors.append(f"scenario_factor_{factor_id}_nominal_out_of_bounds")
        if not _strings(factor.get("affects"), nonempty=True):
            errors.append(f"scenario_factor_{factor_id}_affects_missing")
        validity = _mapping(factor.get("validity"))
        if validity.get("invalid_behavior") != "reject_instance_fail_closed":
            errors.append(f"scenario_factor_{factor_id}_invalid_behavior_invalid")

    templates = _rows(suite.get("cell_templates"))
    template_ids = [str(row.get("template_id") or "") for row in templates]
    if not templates or len(set(template_ids)) != len(template_ids) or "" in template_ids:
        errors.append("scenario_suite_template_ids_invalid")
    for template in templates:
        template_id = str(template.get("template_id") or "")
        family = template.get("family")
        partition = template.get("partition")
        factor_ids = template.get("factor_ids")
        seeds = template.get("seeds")
        if family not in REQUIRED_FAMILIES:
            errors.append(f"scenario_template_{template_id}_family_invalid")
        if partition not in ALLOWED_PARTITIONS:
            errors.append(f"scenario_template_{template_id}_partition_invalid")
        if not isinstance(template.get("scored"), bool):
            errors.append(f"scenario_template_{template_id}_scored_invalid")
        if not isinstance(seeds, list) or not seeds or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in (seeds or [])
        ):
            errors.append(f"scenario_template_{template_id}_seeds_invalid")
        elif len(set(seeds)) != len(seeds):
            errors.append(f"scenario_template_{template_id}_seeds_duplicate")
        if not isinstance(factor_ids, list) or any(item not in factors for item in (factor_ids or [])):
            errors.append(f"scenario_template_{template_id}_factors_invalid")
        if family not in {"canonical", "held_out_composed"} and family not in {
            "visual_material_cousin",
            "geometric_cousin",
        } and len(factor_ids or []) != 1:
            errors.append(f"scenario_template_{template_id}_not_one_factor")
        if family == "held_out_composed" and len(factor_ids or []) < 2:
            errors.append(f"scenario_template_{template_id}_heldout_not_composed")
        expected_cousin = {
            "visual_material_cousin": "adp009d_visual_material_cousin",
            "geometric_cousin": "adp009d_geometric_cousin",
        }.get(str(family), "approved_can")
        if template.get("cousin_id") != expected_cousin:
            errors.append(f"scenario_template_{template_id}_cousin_invalid")

    if {str(row.get("family") or "") for row in templates} != REQUIRED_FAMILIES:
        errors.append("scenario_suite_family_coverage_invalid")
    canonical_templates = [row for row in templates if row.get("family") == "canonical"]
    if len(canonical_templates) != 1:
        errors.append("scenario_suite_canonical_template_count_invalid")
    elif canonical_templates[0].get("factor_ids") != []:
        errors.append("scenario_suite_canonical_not_immutable")

    expanded = _expanded_cells(suite)
    cell_ids = [row["cell_id"] for row in expanded]
    if len(set(cell_ids)) != len(cell_ids):
        errors.append("scenario_suite_cell_ids_duplicate")
    scored = [row for row in expanded if row.get("scored") is True]
    if any(row.get("partition") == "development" for row in scored):
        errors.append("scenario_suite_development_cell_scored")

    analysis = _mapping(suite.get("power_cost_analysis"))
    try:
        alpha = float(analysis["two_sided_alpha"])
        power = float(analysis["target_power"])
        effect = float(analysis["minimum_paired_difference"])
        discordance = float(analysis["anticipated_discordance"])
        required_pairs = _required_paired_n(
            alpha=alpha,
            power=power,
            effect=effect,
            discordance=discordance,
        )
        planned_pairs = len(scored)
        if analysis.get("computed_required_paired_cells") != required_pairs:
            errors.append("scenario_power_required_pairs_mismatch")
        if analysis.get("planned_paired_cells") != planned_pairs:
            errors.append("scenario_power_planned_pairs_mismatch")
        if planned_pairs < required_pairs:
            errors.append("scenario_power_paired_design_underpowered")
        canonical_n = sum(row.get("family") == "canonical" for row in scored)
        canonical_half_width = _wilson_worst_half_width(canonical_n, alpha=alpha)
        if not math.isclose(
            float(analysis.get("canonical_worst_case_wilson_half_width")),
            canonical_half_width,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            errors.append("scenario_power_canonical_precision_mismatch")
        if canonical_half_width > float(analysis["canonical_half_width_max"]):
            errors.append("scenario_power_canonical_precision_insufficient")
        per_family_max = float(analysis["per_family_half_width_max"])
        for family in REQUIRED_FAMILIES - {"canonical"}:
            count = sum(row.get("family") == family for row in scored)
            if _wilson_worst_half_width(count, alpha=alpha) > per_family_max:
                errors.append(f"scenario_power_family_{family}_precision_insufficient")
        estimated_episode_seconds = float(analysis["estimated_episode_wall_seconds"])
        planned_episodes = planned_pairs * 4
        planned_gpu_hours = planned_episodes * estimated_episode_seconds / 3600
        if analysis.get("planned_episode_count") != planned_episodes:
            errors.append("scenario_cost_episode_count_mismatch")
        if not math.isclose(
            float(analysis.get("estimated_total_gpu_hours")),
            planned_gpu_hours,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            errors.append("scenario_cost_gpu_hours_mismatch")
        if planned_gpu_hours > float(analysis["maximum_total_gpu_hours"]):
            errors.append("scenario_cost_cap_exceeded")
        if analysis.get("analysis_frozen_before_learned_outcomes") is not False:
            errors.append("scenario_power_prior_learned_outcomes_not_disclosed")
        if analysis.get("analysis_frozen_before_scenario_evaluation_outcomes") is not True:
            errors.append("scenario_power_scenario_freeze_missing")
    except (KeyError, TypeError, ValueError, Adp009dHarnessError):
        errors.append("scenario_power_cost_analysis_invalid")

    invalid_rows = _rows(suite.get("invalid_combinations"))
    if not invalid_rows:
        errors.append("scenario_invalid_combinations_missing")
    for row in invalid_rows:
        if row.get("behavior") != "reject_instance_fail_closed":
            errors.append("scenario_invalid_combination_behavior_invalid")
        if not _strings(row.get("when_all_non_nominal"), nonempty=True):
            errors.append("scenario_invalid_combination_terms_missing")
        elif any(
            factor_id not in factors
            for factor_id in _strings(row.get("when_all_non_nominal"))
        ):
            errors.append("scenario_invalid_combination_factor_unknown")

    if suite.get("suite_digest") != canonical_digest(suite, digest_field="suite_digest"):
        errors.append("scenario_suite_digest_mismatch")
    if errors:
        raise Adp009dHarnessError(errors)
    return suite


def _check_instance_constraints(
    *,
    harness: Mapping[str, Any],
    suite: Mapping[str, Any],
    factor_records: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
) -> None:
    errors: list[str] = []
    factor_by_id = {str(row.get("parameter_id")): row for row in factor_records}
    for invalid in _rows(suite.get("invalid_combinations")):
        ids = _strings(invalid.get("when_all_non_nominal"))
        if ids and all(
            factor_id in factor_by_id
            and factor_by_id[factor_id].get("resolved_value")
            != factor_by_id[factor_id].get("nominal_value")
            for factor_id in ids
        ):
            errors.append(f"scenario_invalid_combination:{invalid.get('constraint_id')}")

    required_position_keys = (
        "object_start_x_m",
        "object_start_y_m",
        "object_start_z_m",
        "target_x_m",
        "target_y_m",
        "target_z_m",
    )
    if any(_number(parameters.get(key)) is None for key in required_position_keys):
        errors.append("scenario_resolved_positions_missing")
    else:
        start = [float(parameters[key]) for key in required_position_keys[:3]]
        target = [float(parameters[key]) for key in required_position_keys[3:]]
        if math.dist(start, target) < 0.15:
            errors.append("scenario_resolved_translation_below_threshold")
        if abs(start[2] - target[2]) > 1e-9:
            errors.append("scenario_resolved_target_support_height_mismatch")
        support = _mapping(_mapping(harness.get("canonical_condition")).get("support_bounds"))
        radius = float(parameters.get("object_radius_m", 0))
        margin = float(support.get("edge_margin_m", 0))
        for prefix, point in (("start", start), ("target", target)):
            if not (
                float(support.get("x_min_m")) + radius + margin <= point[0]
                <= float(support.get("x_max_m")) - radius - margin
                and float(support.get("y_min_m")) + radius + margin <= point[1]
                <= float(support.get("y_max_m")) - radius - margin
            ):
                errors.append(f"scenario_resolved_{prefix}_outside_support")
    if errors:
        raise Adp009dHarnessError(errors)


def materialize_scenario_suite(
    *,
    harness_manifest: Mapping[str, Any],
    scenario_suite: Mapping[str, Any],
    cousin_manifests: Sequence[Mapping[str, Any]],
    cousin_static_validation_receipts: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    task_construction_admission: Mapping[str, Any] | None = None,
) -> ScenarioMaterialization:
    """Resolve every explicit suite cell from its frozen digest and seed."""

    task_contract = _mapping(harness_manifest.get("task_contract"))
    if task_contract.get("task_kind") == ARTICULATED_TASK_KIND:
        if task_construction_admission is None:
            raise Adp009dHarnessError(
                ["scenario_task_construction_admission_missing"]
            )
        admission = validate_task_construction_admission(
            task_construction_admission, task_contract=task_contract
        )
        if admission.get("scenario_materialization_authorized") is not True:
            raise Adp009dHarnessError(
                [
                    f"scenario_task_construction_not_admitted:{blocker}"
                    for blocker in _strings(admission.get("blockers"), nonempty=True)
                ]
                or ["scenario_task_construction_not_admitted"]
            )

    suite = validate_scenario_suite(
        scenario_suite,
        harness_manifest=harness_manifest,
        cousin_manifests=cousin_manifests,
        cousin_static_validation_receipts=cousin_static_validation_receipts,
    )
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise Adp009dHarnessError(["scenario_materialization_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    instance_dir = output / "instances"
    instance_dir.mkdir()
    factor_by_id = {
        str(row["parameter_id"]): row for row in _rows(suite.get("factors"))
    }
    cousin_by_id = {
        str(row["cousin_id"]): row for row in cousin_manifests
    }
    static_receipt_by_id = {
        str(row["cousin_id"]): row for row in cousin_static_validation_receipts
    }
    canonical_parameters = _mapping(
        _mapping(harness_manifest.get("canonical_condition")).get("parameters")
    )
    instances: list[dict[str, Any]] = []
    for cell in _expanded_cells(suite):
        parameters = json.loads(json.dumps(canonical_parameters))
        factor_records: list[dict[str, Any]] = []
        factor_seed_digests: list[str] = []
        for factor_id in cell["factor_ids"]:
            factor = factor_by_id[factor_id]
            resolved_value, seed_digest = _sample_factor(
                factor,
                suite_digest=str(suite["suite_digest"]),
                cell_id=str(cell["cell_id"]),
                seed=int(cell["seed"]),
            )
            parameters[factor_id] = resolved_value
            factor_seed_digests.append(seed_digest)
            factor_records.append(
                {
                    "parameter_id": factor_id,
                    "semantic_meaning": factor["semantic_meaning"],
                    "unit": factor["unit"],
                    "nominal_value": factor["nominal_value"],
                    "allowed": factor["allowed"],
                    "sampling": factor["sampling"],
                    "seed": cell["seed"],
                    "resolved_seed_digest": seed_digest,
                    "resolved_value": resolved_value,
                    "source": factor["source"],
                    "reason": factor["reason"],
                    "runtime_target": factor["runtime_target"],
                    "affects": factor["affects"],
                    "validity": factor["validity"],
                }
            )
        cousin_id = str(cell["cousin_id"])
        cousin_digest = (
            REQUIRED_ASSET_DIGESTS["approved_can"]
            if cousin_id == "approved_can"
            else str(cousin_by_id[cousin_id]["cousin_digest"])
        )
        cousin_static_validation_receipt_digest = (
            None
            if cousin_id == "approved_can"
            else static_receipt_by_id[cousin_id]["validation_receipt_digest"]
        )
        if cousin_id != "approved_can":
            dimensions = _mapping(cousin_by_id[cousin_id].get("dimensions_m"))
            parameters["object_radius_m"] = float(dimensions["diameter"]) / 2
            parameters["object_height_m"] = dimensions["height"]
            physics = _mapping(cousin_by_id[cousin_id].get("physics"))
            parameters["object_mass_kg"] = physics["mass_kg"]
        _check_instance_constraints(
            harness=harness_manifest,
            suite=suite,
            factor_records=factor_records,
            parameters=parameters,
        )
        instance = {
            "schema_version": SCENARIO_INSTANCE_SCHEMA_VERSION,
            "program_id": PROGRAM_ID,
            "suite_digest": suite["suite_digest"],
            "harness_digest": harness_manifest["harness_digest"],
            "cell_id": cell["cell_id"],
            "template_id": cell["template_id"],
            "family": cell["family"],
            "partition": cell["partition"],
            "scored": cell["scored"],
            "seed": cell["seed"],
            "cell_seed_digest": canonical_digest(
                {
                    "suite_digest": suite["suite_digest"],
                    "cell_id": cell["cell_id"],
                    "seed": cell["seed"],
                    "factor_seed_digests": factor_seed_digests,
                }
            ),
            "cousin_id": cousin_id,
            "cousin_digest": cousin_digest,
            "cousin_static_validation_receipt_digest": (
                cousin_static_validation_receipt_digest
            ),
            "resolved_parameters": parameters,
            "factor_records": factor_records,
            "required_controls": sorted(REQUIRED_CONTROLS),
            "policy_neutral": True,
            "caller_asserted_success": False,
            "instance_digest": "",
        }
        instance["instance_digest"] = canonical_digest(
            instance, digest_field="instance_digest"
        )
        write_json(instance_dir / f"{cell['cell_id']}.json", instance)
        instances.append(instance)
    receipt = {
        "schema_version": SCENARIO_MATERIALIZATION_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "suite_digest": suite["suite_digest"],
        "harness_digest": harness_manifest["harness_digest"],
        "cousin_static_validation_bindings": [
            {
                "cousin_id": cousin_id,
                "cousin_digest": cousin_by_id[cousin_id]["cousin_digest"],
                "validation_receipt_digest": static_receipt_by_id[cousin_id][
                    "validation_receipt_digest"
                ],
            }
            for cousin_id in sorted(cousin_by_id)
        ],
        "instance_count": len(instances),
        "instance_bindings": [
            {
                "cell_id": row["cell_id"],
                "instance_digest": row["instance_digest"],
                "relative_path": f"instances/{row['cell_id']}.json",
            }
            for row in instances
        ],
        "candidate_pairing_rule": "all_candidates_and_controls_reference_the_same_instance_digest",
        "caller_asserted_results_accepted": False,
        "materialization_digest": "",
    }
    receipt["materialization_digest"] = canonical_digest(
        receipt, digest_field="materialization_digest"
    )
    write_json(output / "adp009d_scenario_materialization.v1.json", receipt)
    return ScenarioMaterialization(receipt=receipt, instances=tuple(instances))
