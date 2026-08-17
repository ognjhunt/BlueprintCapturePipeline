"""Compile 1-5 paired replacement tasks into production Arena requests.

The compiler joins only digest-bound task, registered-USD, interaction,
camera, scenario, and support/kinematic records.  It extracts the registered
USD root transform as the single Arena spawn authority, so a frame-registration
rotation cannot be lost or applied twice.  It authors no geometry and performs
no native reach, contact, control, policy, or physical execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .articulation_graph_contract import validate_articulation_graph
from .decision_evidence_contracts import canonical_digest
from .dual_task_rehearsal_contract import (
    FROZEN_CANDIDATES,
    MAX_REPLACEMENT_OBJECTS,
    validate_task_freeze,
)
from .dual_task_scenario_suite import validate_dual_task_scenario_suite
from .native_task_runtime_contract import DROID_FRANKA_RESET_JOINT_NAMES
from .paired_target_native_construction_bindings import (
    validate_materialized_paired_target_native_construction_bindings,
)


SCHEMA_VERSION = "paired_target_native_arena_requests.v1"
REQUEST_SCHEMA = "native_task_arena_packet_request.v1"


class PairedTargetNativeArenaRequestError(ValueError):
    """Stable fail-closed request compilation error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: str | Path, code: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedTargetNativeArenaRequestError(code) from exc
    if source.is_symlink() or not isinstance(value, dict):
        raise PairedTargetNativeArenaRequestError(code)
    return source, value


def _record(path: Path, **extra: Any) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise PairedTargetNativeArenaRequestError(
            "paired_target_arena_request_source_invalid"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        **extra,
    }


def _bound_json(
    path: str | Path,
    *,
    schema: str,
    digest_field: str,
    code: str,
) -> tuple[Path, dict[str, Any]]:
    source, value = _read(path, code)
    if (
        value.get("schema_version") != schema
        or value.get(digest_field)
        != canonical_digest(value, digest_field=digest_field)
    ):
        raise PairedTargetNativeArenaRequestError(code)
    return source, value


def _relative_source(path: Path, *, evidence_root: Path) -> dict[str, Any]:
    try:
        relative = path.relative_to(evidence_root)
    except ValueError as exc:
        raise PairedTargetNativeArenaRequestError(
            "paired_target_arena_request_asset_outside_evidence_root"
        ) from exc
    if path.is_symlink() or not path.is_file() or ".." in relative.parts:
        raise PairedTargetNativeArenaRequestError(
            "paired_target_arena_request_asset_invalid"
        )
    return {
        "root": "evidence",
        "relative_path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _quat_from_rotation(rotation: np.ndarray) -> list[float]:
    # Stable matrix-to-quaternion conversion, output xyzw.
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = [
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        ]
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            q = [0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale]
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            q = [(rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale]
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            q = [(rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale]
    norm = math.sqrt(sum(item * item for item in q))
    result = [float(item / norm) for item in q]
    if result[3] < 0.0:
        result = [-item for item in result]
    return result


def _matrix_from_pose(position: Sequence[float], orientation: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(item) for item in orientation)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def _pose_from_matrix(matrix: np.ndarray) -> dict[str, list[float]]:
    return {
        "position_world_m": [float(item) for item in matrix[:3, 3]],
        "orientation_xyzw": _quat_from_rotation(matrix[:3, :3]),
    }


def _registered_root_pose(usd_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover
        raise PairedTargetNativeArenaRequestError(
            "paired_target_arena_request_openusd_missing"
        ) from exc
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim() if stage is not None else None
    if root is None or not root.IsValid():
        raise PairedTargetNativeArenaRequestError(
            "paired_target_arena_request_registered_usd_invalid"
        )
    # Gf matrices use row-vector storage; transpose to column convention.
    matrix = np.asarray(
        UsdGeom.XformCache().GetLocalToWorldTransform(root), dtype=np.float64
    ).T
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6
    ):
        raise PairedTargetNativeArenaRequestError(
            "paired_target_arena_request_registered_root_not_rigid"
        )
    return _pose_from_matrix(matrix), matrix


def _joint_bindings(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for joint in graph["joints"]:
        fixed = joint["joint_type"] == "fixed"
        rows.append(
            {
                "joint_id": joint["joint_id"],
                "joint_prim_path": f"/Asset/joints/{joint['joint_id']}",
                "native_joint_name": None if fixed else joint["joint_id"],
                "readback_kind": "fixed_joint_static" if fixed else "native_coordinate",
                "static_qualification_digest": (
                    canonical_digest(joint) if fixed else None
                ),
                "role": joint["role"],
            }
        )
    return rows


def _articulated_task_spec(
    freeze: Mapping[str, Any],
    affordance: Mapping[str, Any],
    path_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = validate_articulation_graph(freeze["articulation_graph"])
    graph_digest = canonical_digest(graph)
    candidate = affordance["candidate"]
    if (
        path_receipt.get("task_id") != freeze["task_id"]
        or path_receipt.get("interaction_affordance", {}).get("receipt_digest")
        != affordance["receipt_digest"]
        or path_receipt.get("articulation_graph_digest") != graph_digest
    ):
        raise PairedTargetNativeArenaRequestError(
            "paired_target_arena_request_kinematic_binding_mismatch"
        )
    interaction = {
        "schema_version": "native_articulated_graph_interaction_affordance.v1",
        "subject_asset_id": affordance["asset_id"],
        "articulation_graph_digest": graph_digest,
        "kinematic_path_receipt_digest": path_receipt["receipt_digest"],
        "contact_link_id": candidate["link_id"],
        "contact_body_prim_paths": list(candidate["contact_body_prim_paths"]),
        "contact_point_link_m": list(candidate["contact_point_link_m"]),
        "approach_unit_asset_root": list(path_receipt["joint_contact_path"][0]["clearance_unit_asset_root"]),
        "retreat_unit_asset_root": list(path_receipt["joint_contact_path"][-1]["clearance_unit_asset_root"]),
        "gripper_orientation_contact_xyzw": [0.0, 0.0, 0.0, 1.0],
        "precontact_clearance_m": 0.12,
        "sweep_clearance_m": 0.025,
        "retreat_clearance_m": 0.12,
        "arrival_tolerance_m": 0.02,
        "arrival_orientation_tolerance_rad": 0.08,
        "arrival_stability_steps": 2,
        "motion_minimum_steps": 1,
        "motion_maximum_steps": 64,
        "gripper_dwell_minimum_steps": 5,
        "gripper_dwell_maximum_steps": 12,
        "max_joint_delta_rad": 0.03,
        "max_joint_setpoint_lead_rad": 0.2,
        "joint_contact_path": path_receipt["joint_contact_path"],
        "affordance_digest": "",
    }
    interaction["affordance_digest"] = canonical_digest(
        interaction, digest_field="affordance_digest"
    )
    execution = freeze["execution_contract"]
    spec = {
        "schema_version": "adp_task_spec.v2",
        "task_kind": "articulated_open_close",
        "subject_asset_id": affordance["asset_id"],
        "prompt": freeze["prompt"],
        "articulation_graph": graph,
        "articulation_graph_digest": graph_digest,
        "interaction_affordance": interaction,
        "settle_window_samples": int(execution["settle_window_steps"]),
        "maximum_settled_target_speed": 0.03,
        "locked_joint_motion_tolerance": 0.01,
        "movement_epsilon": 0.01,
        "control_frequency_hz": int(execution["control_frequency_hz"]),
        "maximum_action_steps": int(execution["maximum_steps"]),
    }
    state = {
        "schema_version": "native_articulated_graph_task_state_binding.v1",
        "articulation_graph_digest": graph_digest,
        "interaction_affordance_digest": interaction["affordance_digest"],
        "link_native_body_names": {
            link["link_id"]: link["link_id"] for link in graph["links"]
        },
        "task_contact_minimum_force_n": 0.5,
        "collision_failure_minimum_force_n": 1.0,
        "retreat_minimum_separation_m": 0.10,
        "root_translation_tolerance_m": 0.002,
        "root_orientation_tolerance_rad": 0.01,
    }
    return spec, state


def _rigid_task_spec(
    freeze: Mapping[str, Any],
    affordance: Mapping[str, Any],
    support: Mapping[str, Any],
    *,
    root_matrix: np.ndarray,
) -> tuple[dict[str, Any], None]:
    graph = validate_articulation_graph(
        freeze["articulation_graph"], require_target_joint=False
    )
    if any(joint["role"] != "locked" for joint in graph["joints"]):
        raise PairedTargetNativeArenaRequestError(
            "paired_target_arena_request_rigid_graph_not_locked"
        )
    matches = support.get("whole_object_matches")
    if (
        support.get("receipt_digest")
        != canonical_digest(support, digest_field="receipt_digest")
        or support.get("whole_object_collision_identity_passed") is not True
        or not isinstance(matches, list)
        or len(matches) != 1
        or matches[0].get("prim_path") in (None, "")
        or freeze["source_object"].get("support_receipt_digest")
        != support["receipt_digest"]
    ):
        raise PairedTargetNativeArenaRequestError(
            "paired_target_arena_request_support_invalid"
        )
    observed = freeze["source_object"]["observed_pose_world"]
    scoring_world = _matrix_from_pose(
        observed["position_world_m"], observed["orientation_xyzw"]
    )
    root_from_scoring = np.linalg.inv(root_matrix) @ scoring_world
    candidate = affordance["candidate"]
    contact_world = np.asarray([*candidate["contact_point_registered_stage_m"], 1.0])
    contact_scoring = np.linalg.inv(scoring_world) @ contact_world
    approach_scoring = np.linalg.inv(scoring_world[:3, :3]) @ np.asarray(
        candidate["approach_unit_registered_stage"]
    )
    target = freeze["target_configuration"]
    bounds = target["position_bounds_world_m"]
    destination = [
        (float(low) + float(high)) / 2.0
        for low, high in zip(bounds["minimum"], bounds["maximum"], strict=True)
    ]
    start = [float(item) for item in observed["position_world_m"]]
    distance = math.dist(start[:2], destination[:2])
    pad = 0.5
    interaction = {
        "schema_version": "native_rigid_interaction_affordance.v1",
        "subject_asset_id": affordance["asset_id"],
        "scoring_frame_id": "task_scoring_frame",
        "asset_root_from_scoring_frame": {
            "position_m": [float(item) for item in root_from_scoring[:3, 3]],
            "orientation_xyzw": _quat_from_rotation(root_from_scoring[:3, :3]),
        },
        "contact_point_scoring_frame_m": [float(item) for item in contact_scoring[:3]],
        "approach_unit_scoring_frame": [float(item) for item in approach_scoring],
        "lift_unit_world": [0.0, 0.0, 1.0],
        "gripper_orientation_scoring_frame_xyzw": [0.0, 0.0, 0.0, 1.0],
        "pregrasp_clearance_m": 0.12,
        "arrival_orientation_tolerance_rad": 0.08,
        "allowed_contact_prim_paths": list(candidate["contact_body_prim_paths"]),
        "intended_support_prim_paths": [str(matches[0]["prim_path"])],
        "affordance_digest": "",
    }
    interaction["affordance_digest"] = canonical_digest(
        interaction, digest_field="affordance_digest"
    )
    spec = {
        "schema_version": "adp_task_spec.v2",
        "task_kind": "rigid_pick_place",
        "subject_asset_id": affordance["asset_id"],
        "prompt": freeze["prompt"],
        "articulation_graph": graph,
        "articulation_graph_digest": canonical_digest(graph),
        "start_pose_world": [*start, *observed["orientation_xyzw"]],
        "destination_position_bounds_world_m": bounds,
        "support_height_interval_m": [float(bounds["minimum"][2]), float(bounds["maximum"][2])],
        "destination_orientation_xyzw": list(target["orientation_reference_xyzw"]),
        "destination_orientation_tolerance_rad": float(target["maximum_orientation_error_rad"]),
        "minimum_lift_m": 0.05,
        "minimum_translation_m": max(0.01, distance * 0.8),
        "movement_epsilon_m": 0.005,
        "settle_window_samples": int(freeze["execution_contract"]["settle_window_steps"]),
        "control_frequency_hz": int(
            freeze["execution_contract"]["control_frequency_hz"]
        ),
        "maximum_action_steps": int(freeze["execution_contract"]["maximum_steps"]),
        "release_required": True,
        "release_gripper_width_min_m": 0.06,
        "task_contact_minimum_force_n": 0.5,
        "collision_failure_minimum_force_n": 1.0,
        "reset_translation_tolerance_m": 0.002,
        "reset_orientation_tolerance_rad": 0.01,
        "settle_position_tolerance_m": 0.005,
        "settle_orientation_tolerance_rad": 0.03,
        "relocation_tracking_tolerance_m": 0.03,
        "workspace_position_bounds_world_m": {
            "minimum": [min(start[i], destination[i]) - pad for i in range(3)],
            "maximum": [max(start[i], destination[i]) + pad for i in range(3)],
        },
        "interaction_affordance": interaction,
    }
    return spec, None


def materialize_paired_target_native_arena_requests(
    *,
    construction_bindings_path: str | Path,
    task_inputs: Sequence[Mapping[str, Any]],
    evidence_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Compile one canonical Arena request per selected task (1-5)."""

    evidence = Path(evidence_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if (
        not evidence.is_dir()
        or evidence.is_symlink()
        or destination.exists()
        or isinstance(task_inputs, (str, bytes))
        or not 1 <= len(task_inputs) <= MAX_REPLACEMENT_OBJECTS
    ):
        raise PairedTargetNativeArenaRequestError(
            "paired_target_arena_request_inputs_invalid"
        )
    construction_path, raw_construction = _read(
        construction_bindings_path,
        "paired_target_arena_request_construction_invalid",
    )
    construction = validate_materialized_paired_target_native_construction_bindings(
        raw_construction
    )
    bindings = {row["task_id"]: row for row in construction["bindings"]}
    if len(bindings) != len(task_inputs):
        raise PairedTargetNativeArenaRequestError(
            "paired_target_arena_request_task_set_mismatch"
        )

    opened: dict[str, dict[str, Any]] = {}
    for raw in task_inputs:
        if not isinstance(raw, Mapping):
            raise PairedTargetNativeArenaRequestError(
                "paired_target_arena_request_task_input_invalid"
            )
        freeze_path, raw_freeze = _read(
            raw.get("task_freeze_path"), "paired_target_arena_request_freeze_invalid"
        )
        freeze = validate_task_freeze(raw_freeze)
        task_id = freeze["task_id"]
        if task_id in opened or task_id not in bindings:
            raise PairedTargetNativeArenaRequestError(
                "paired_target_arena_request_task_set_mismatch"
            )
        registered_path, registered = _bound_json(
            raw.get("registered_asset_receipt_path"),
            schema="registered_replacement_asset.v1",
            digest_field="receipt_digest",
            code="paired_target_arena_request_registered_asset_invalid",
        )
        affordance_path, affordance = _bound_json(
            raw.get("interaction_affordance_path"),
            schema="paired_target_interaction_affordance_candidate.v1",
            digest_field="receipt_digest",
            code="paired_target_arena_request_affordance_invalid",
        )
        camera_path, camera = _bound_json(
            raw.get("camera_rig_path"),
            schema="paired_target_native_camera_rig_candidate.v1",
            digest_field="receipt_digest",
            code="paired_target_arena_request_camera_invalid",
        )
        scenario_path, raw_scenario = _read(
            raw.get("scenario_suite_path"),
            "paired_target_arena_request_scenario_invalid",
        )
        scenario = validate_dual_task_scenario_suite(raw_scenario)
        appearance_path = Path(raw.get("appearance_path") or "").expanduser().resolve()
        usd_record = registered.get("output_usd") or {}
        usd_path = Path(usd_record.get("path") or "").expanduser().resolve()
        if (
            registered.get("task_id") != task_id
            or registered.get("asset_id") != affordance.get("asset_id")
            or registered.get("task_freeze_digest") != freeze["task_freeze_digest"]
            or affordance.get("task_id") != task_id
            or affordance.get("registered_asset", {}).get("receipt_digest") != registered["receipt_digest"]
            or affordance.get("candidate", {}).get("pinch_span_within_stroke") is not True
            or affordance.get("candidate", {}).get("contact_body_prim_paths") != [affordance.get("candidate", {}).get("link_prim_path")]
            or camera.get("task_id") != task_id
            or camera.get("interaction_affordance_candidate", {}).get("receipt_digest") != affordance["receipt_digest"]
            or scenario.get("task_id") != task_id
            or scenario.get("task_freeze_digest") != freeze["task_freeze_digest"]
            or bindings[task_id]["asset_id"] != registered["asset_id"]
            or bindings[task_id]["replacement_asset_sha256"] != usd_record.get("sha256")
            or not usd_path.is_file()
            or usd_path.stat().st_size != usd_record.get("size_bytes")
            or _sha256(usd_path) != usd_record.get("sha256")
            or appearance_path.is_symlink()
            or not appearance_path.is_file()
        ):
            raise PairedTargetNativeArenaRequestError(
                "paired_target_arena_request_task_binding_mismatch"
            )
        pose, root_matrix = _registered_root_pose(usd_path)
        opened[task_id] = {
            "input": dict(raw),
            "freeze_path": freeze_path,
            "freeze": freeze,
            "registered_path": registered_path,
            "registered": registered,
            "usd_path": usd_path,
            "pose": pose,
            "root_matrix": root_matrix,
            "affordance_path": affordance_path,
            "affordance": affordance,
            "camera_path": camera_path,
            "camera": camera,
            "scenario_path": scenario_path,
            "scenario": scenario,
            "appearance_path": appearance_path,
        }

    collision_path = Path(construction["collision_scene"]["path"]).resolve()
    if (
        not collision_path.is_file()
        or collision_path.stat().st_size != construction["collision_scene"]["size_bytes"]
        or _sha256(collision_path) != construction["collision_scene"]["sha256"]
    ):
        raise PairedTargetNativeArenaRequestError(
            "paired_target_arena_request_collision_invalid"
        )
    destination.mkdir(parents=True)
    request_records = []
    try:
        for task_id in sorted(opened):
            row = opened[task_id]
            freeze = row["freeze"]
            affordance = row["affordance"]
            graph = validate_articulation_graph(
                freeze["articulation_graph"],
                require_target_joint=freeze["task_kind"] == "articulated_interaction",
            )
            if freeze["task_kind"] == "articulated_interaction":
                path_path, path_receipt = _bound_json(
                    row["input"].get("kinematic_path_receipt_path"),
                    schema="paired_target_articulated_kinematic_path.v1",
                    digest_field="receipt_digest",
                    code="paired_target_arena_request_kinematic_invalid",
                )
                task_spec, state = _articulated_task_spec(
                    freeze, affordance, path_receipt
                )
                support_record = None
                kinematic_record = _record(
                    path_path, receipt_digest=path_receipt["receipt_digest"]
                )
            else:
                support_path, support = _bound_json(
                    row["input"].get("support_receipt_path"),
                    schema="interiorgs_sage_collision_identity.v1",
                    digest_field="receipt_digest",
                    code="paired_target_arena_request_support_invalid",
                )
                task_spec, state = _rigid_task_spec(
                    freeze, affordance, support, root_matrix=row["root_matrix"]
                )
                support_record = _record(
                    support_path, receipt_digest=support["receipt_digest"]
                )
                kinematic_record = None
            assets: list[dict[str, Any]] = [
                {
                    "semantic_role": "scene_collision",
                    "filename": "scene_collision.usda",
                    "source": _relative_source(collision_path, evidence_root=evidence),
                    "pose_world": {
                        "position_world_m": [0.0, 0.0, 0.0],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "visible": False,
                },
                {
                    "semantic_role": "scene_appearance",
                    "filename": "scene_appearance.usdz",
                    "source": _relative_source(row["appearance_path"], evidence_root=evidence),
                    "pose_world": {
                        "position_world_m": [0.0, 0.0, 0.0],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "visible": True,
                },
            ]
            for replacement_task_id in sorted(opened):
                replacement = opened[replacement_task_id]
                replacement_graph = validate_articulation_graph(
                    replacement["freeze"]["articulation_graph"],
                    require_target_joint=(
                        replacement["freeze"]["task_kind"]
                        == "articulated_interaction"
                    ),
                )
                joint_resets = {
                    joint["joint_id"]: float(joint["reset_position"])
                    for joint in replacement_graph["joints"]
                    if joint["joint_type"] != "fixed"
                }
                assets.append(
                    {
                        "semantic_role": "replacement",
                        "asset_id": replacement["registered"]["asset_id"],
                        "filename": f"replacement__{replacement['registered']['asset_id']}.usda",
                        "source": _relative_source(replacement["usd_path"], evidence_root=evidence),
                        "pose_world": replacement["pose"],
                        "object_type": "ARTICULATION",
                        "reset_state": {"joint_positions": joint_resets},
                        "visible": True,
                    }
                )
            canonical = next(
                cell for cell in row["scenario"]["cells"] if cell["family"] == "canonical"
            )
            context = {
                "schema_version": "native_task_construction_canary.v1",
                "program_id": "arm-decision-proof-v1",
                "scene_id": construction["scene_id"],
                "task_id": task_id,
                "cell_id": canonical["cell_id"],
                "seed": canonical["seed"],
                "scenario_suite_digest": row["scenario"]["suite_digest"],
                "resolved_parameters": dict(canonical["resolved_parameters"]),
                "factor_records": list(canonical["factor_records"]),
                "policy_neutral": True,
                "caller_asserted_success": False,
                "learned_policy_outcomes_consulted": False,
                "context_digest": "",
            }
            context["context_digest"] = canonical_digest(
                context, digest_field="context_digest"
            )
            request = {
                "schema_version": REQUEST_SCHEMA,
                "scene_id": construction["scene_id"],
                "task_id": task_id,
                "task_freeze_digest": freeze["task_freeze_digest"],
                "construction_bindings": construction,
                "task_spec": task_spec,
                "task_joint_bindings": _joint_bindings(graph),
                "task_state_binding": state,
                "assets": assets,
                "robot_base_pose_world": row["camera"]["robot_base_pose_world"],
                "robot_joint_reset_positions_rad": row["camera"]["robot_joint_reset_positions_rad"],
                "cameras": row["camera"]["cameras"],
                "scenario": {
                    "context_kind": "construction_canary",
                    "cell_id": canonical["cell_id"],
                    "instance_digest": context["context_digest"],
                    "seed": canonical["seed"],
                    "context_document": context,
                },
                "physics_frequency_hz": 120,
                "request_digest": "",
            }
            if set(request["robot_joint_reset_positions_rad"]) != set(
                DROID_FRANKA_RESET_JOINT_NAMES
            ):
                raise PairedTargetNativeArenaRequestError(
                    "paired_target_arena_request_robot_reset_invalid"
                )
            request["request_digest"] = canonical_digest(
                request, digest_field="request_digest"
            )
            task_dir = destination / task_id
            task_dir.mkdir()
            request_path = task_dir / "native_task_arena_packet_request.v1.json"
            request_path.write_text(
                json.dumps(request, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            request_records.append(
                {
                    "task_id": task_id,
                    "asset_id": row["registered"]["asset_id"],
                    "task_kind": request["task_spec"]["task_kind"],
                    "request": _record(
                        request_path, request_digest=request["request_digest"]
                    ),
                    "task_freeze": _record(
                        row["freeze_path"],
                        task_freeze_digest=freeze["task_freeze_digest"],
                    ),
                    "registered_asset": _record(
                        row["registered_path"],
                        receipt_digest=row["registered"]["receipt_digest"],
                    ),
                    "interaction_affordance": _record(
                        row["affordance_path"],
                        receipt_digest=affordance["receipt_digest"],
                    ),
                    "camera_rig": _record(
                        row["camera_path"],
                        receipt_digest=row["camera"]["receipt_digest"],
                    ),
                    "scenario_suite": _record(
                        row["scenario_path"],
                        suite_digest=row["scenario"]["suite_digest"],
                    ),
                    "support_receipt": support_record,
                    "kinematic_path": kinematic_record,
                    "registered_root_pose_world": row["pose"],
                }
            )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "canonical_native_arena_requests_compiled_pending_packet_and_native_execution",
            "scene_id": construction["scene_id"],
            "replacement_object_count": len(opened),
            "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
            "construction_bindings": _record(
                construction_path,
                construction_digest=construction["construction_digest"],
            ),
            "tasks": request_records,
            "candidate_ids": list(FROZEN_CANDIDATES),
            "required_controls": ["zero_action_negative", "scripted_positive"],
            "native_execution_performed": False,
            "learned_policies_executed": False,
            "claim_boundary": (
                "deterministic_request_compilation_only;not_native_spawn_pose_camera_"
                "reach_contact_controls_policy_task_success_or_physical_evidence"
            ),
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        (destination / f"{SCHEMA_VERSION}.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return json.loads(json.dumps(receipt))
    except Exception:
        import shutil

        shutil.rmtree(destination)
        raise


__all__ = [
    "PairedTargetNativeArenaRequestError",
    "SCHEMA_VERSION",
    "materialize_paired_target_native_arena_requests",
]
