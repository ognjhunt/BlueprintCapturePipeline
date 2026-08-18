"""Join paired-target appearance/import proof to native manipulation inputs.

The paired-target appearance lane intentionally stops after calibrated review
camera materialization and native replacement import.  A robot construction
canary needs more: the frozen task, the registered Franka placement, and the
policy camera and interaction inputs.  The explicit ``pre_arena`` phase seals
those inputs for construction bindings before Arena requests exist; the
``arena_packet`` phase then validates each compiled request containing the
camera rig, robot reset, task-state bindings, and all co-present replacements.

This module is the task-neutral 1--5 object join between those seams.  It never
fills missing manipulation data from object names or scene constants.  A
In the pre-arena phase, a missing Arena request is a typed pending requirement,
not a blocker.  It becomes a blocker only in the final arena-packet phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adp009d_sage_franka_placement import PLACEMENT_PACKET_SCHEMA_VERSION
from .freeze_amendment_carry_forward import (
    FreezeAmendmentCarryForwardError,
    validate_freeze_amendment_carry_forward_content,
)
from .decision_evidence_contracts import canonical_digest
from .dual_task_rehearsal_contract import (
    MAX_REPLACEMENT_OBJECTS,
    DualTaskRehearsalContractError,
    validate_task_freeze,
    validate_task_freeze_set,
)
from .native_task_arena_packet import REQUEST_SCHEMA_VERSION as ARENA_REQUEST_SCHEMA
from .paired_target_interaction_affordance_candidate import (
    SCHEMA_VERSION as INTERACTION_AFFORDANCE_SCHEMA,
)
from .paired_target_native_camera_rig_candidate import (
    SCHEMA_VERSION as CAMERA_RIG_SCHEMA,
)


SCHEMA_VERSION = "paired_target_native_manipulation_preflight.v1"
PAIRED_PREFLIGHT_SCHEMA = "paired_target_native_preflight.v1"
NATIVE_IMPORT_RESULT_SCHEMA = "paired_target_native_import_runtime_result.v1"
SCENARIO_SCHEMA = "third_scene_task_scenario_suite.v1"
REGISTERED_ASSET_SCHEMA = "registered_replacement_asset.v1"
FROZEN_CANDIDATES = ("pi05_droid", "groot_n17_droid")
PREFLIGHT_PHASES = ("pre_arena", "arena_packet")


class PairedTargetNativeManipulationPreflightError(ValueError):
    """Stable fail-closed manipulation preflight failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: str | Path | None, code: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(str(path or "")).expanduser().resolve()
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedTargetNativeManipulationPreflightError(code) from exc
    if candidate.is_symlink() or not isinstance(value, dict):
        raise PairedTargetNativeManipulationPreflightError(code)
    return candidate, value


def _record(path: Path, **extra: Any) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise PairedTargetNativeManipulationPreflightError(
            "paired_target_manipulation_file_invalid"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        **extra,
    }


def _bound_record(value: Any, code: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise PairedTargetNativeManipulationPreflightError(code)
    path = Path(str(value.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise PairedTargetNativeManipulationPreflightError(code)
    return path, dict(value)


def _pose_from_xyzyaw(value: Any) -> dict[str, list[float]]:
    try:
        x, y, z, yaw = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise PairedTargetNativeManipulationPreflightError(
            "paired_target_manipulation_franka_pose_invalid"
        ) from exc
    if not all(math.isfinite(item) for item in (x, y, z, yaw)):
        raise PairedTargetNativeManipulationPreflightError(
            "paired_target_manipulation_franka_pose_invalid"
        )
    return {
        "position_world_m": [x, y, z],
        "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
    }


def _same_vector(left: Any, right: Any, *, tolerance: float = 1.0e-9) -> bool:
    try:
        left_values = [float(item) for item in left]
        right_values = [float(item) for item in right]
    except (TypeError, ValueError):
        return False
    return len(left_values) == len(right_values) and all(
        abs(a - b) <= tolerance
        for a, b in zip(left_values, right_values, strict=True)
    )


def _validate_arena_request(
    *,
    path: Path,
    request: Mapping[str, Any],
    scene_id: str,
    task_id: str,
    task_freeze_digest: str,
    expected_asset_ids: set[str],
    robot_base_pose_world: Mapping[str, Any],
    expected_cameras: Sequence[Mapping[str, Any]],
    expected_robot_reset: Mapping[str, Any],
) -> dict[str, Any]:
    assets = request.get("assets")
    cameras = request.get("cameras")
    if (
        request.get("schema_version") != ARENA_REQUEST_SCHEMA
        or request.get("request_digest")
        != canonical_digest(request, digest_field="request_digest")
        or request.get("scene_id") != scene_id
        or request.get("task_id") != task_id
        or request.get("task_freeze_digest") != task_freeze_digest
        or not isinstance(assets, list)
        or not isinstance(cameras, list)
    ):
        raise PairedTargetNativeManipulationPreflightError(
            f"paired_target_manipulation_arena_request_invalid:{task_id}"
        )
    roles = [str(row.get("role") or "") for row in cameras if isinstance(row, Mapping)]
    replacement_ids = {
        str(row.get("asset_id") or "")
        for row in assets
        if isinstance(row, Mapping) and row.get("semantic_role") == "replacement"
    }
    base = request.get("robot_base_pose_world")
    if (
        sorted(roles) != ["external", "overview", "wrist"]
        or replacement_ids != expected_asset_ids
        or not isinstance(base, Mapping)
        or not _same_vector(
            base.get("position_world_m"), robot_base_pose_world.get("position_world_m")
        )
        or not _same_vector(
            base.get("orientation_xyzw"), robot_base_pose_world.get("orientation_xyzw")
        )
        or cameras != list(expected_cameras)
        or request.get("robot_joint_reset_positions_rad") != dict(expected_robot_reset)
    ):
        raise PairedTargetNativeManipulationPreflightError(
            f"paired_target_manipulation_arena_request_mismatch:{task_id}"
        )
    return _record(
        path,
        request_digest=request["request_digest"],
        camera_roles=roles,
        replacement_asset_ids=sorted(replacement_ids),
    )


def _validate_camera_rig_candidate(
    *,
    path: Path,
    candidate: Mapping[str, Any],
    scene_id: str,
    task_id: str,
    affordance_record: Mapping[str, Any],
    placement_digest: str,
    robot_base_pose_world: Mapping[str, Any],
) -> dict[str, Any]:
    cameras = candidate.get("cameras")
    reset = candidate.get("robot_joint_reset_positions_rad")
    roles = [
        str(row.get("role") or "")
        for row in cameras or []
        if isinstance(row, Mapping)
    ]
    base = candidate.get("robot_base_pose_world")
    if (
        candidate.get("schema_version") != CAMERA_RIG_SCHEMA
        or candidate.get("receipt_digest")
        != canonical_digest(candidate, digest_field="receipt_digest")
        or candidate.get("status")
        != "native_camera_rig_requested_requires_readback_and_observability"
        or candidate.get("scene_id") != scene_id
        or candidate.get("task_id") != task_id
        or candidate.get("interaction_affordance_candidate", {}).get(
            "receipt_digest"
        )
        != affordance_record.get("receipt_digest")
        or candidate.get("franka_placement_packet", {}).get("packet_digest")
        != placement_digest
        or not isinstance(cameras, list)
        or roles != ["external", "wrist", "overview"]
        or not isinstance(reset, Mapping)
        or not isinstance(base, Mapping)
        or not _same_vector(
            base.get("position_world_m"), robot_base_pose_world.get("position_world_m")
        )
        or not _same_vector(
            base.get("orientation_xyzw"), robot_base_pose_world.get("orientation_xyzw")
        )
        or candidate.get("native_camera_readback_qualified") is not False
        or candidate.get("native_semantic_observability_qualified") is not False
        or candidate.get("overview_review_only") is not True
    ):
        raise PairedTargetNativeManipulationPreflightError(
            f"paired_target_manipulation_camera_rig_invalid:{task_id}"
        )
    return _record(
        path,
        receipt_digest=candidate["receipt_digest"],
        camera_roles=roles,
        policy_input_roles=list(candidate.get("policy_input_roles") or []),
        requested_camera_readback_qualified=False,
    )


def _validate_interaction_affordance_candidate(
    *,
    path: Path,
    candidate: Mapping[str, Any],
    scene_id: str,
    task_id: str,
    asset_id: str,
    task_freeze_digest: str,
    registered_asset: Mapping[str, Any],
    robot_base_pose_world: Mapping[str, Any],
) -> dict[str, Any]:
    registered_record = candidate.get("registered_asset")
    if (
        candidate.get("schema_version") != INTERACTION_AFFORDANCE_SCHEMA
        or candidate.get("receipt_digest")
        != canonical_digest(candidate, digest_field="receipt_digest")
        or candidate.get("status")
        != "candidate_geometry_materialized_requires_native_contact"
        or candidate.get("scene_id") != scene_id
        or candidate.get("task_id") != task_id
        or candidate.get("asset_id") != asset_id
        or candidate.get("native_contact_execution_authorized") is not False
        or candidate.get("native_contact_executed") is not False
        or not isinstance(registered_record, Mapping)
        or registered_record.get("receipt_digest")
        != registered_asset.get("receipt_digest")
        or candidate.get("task_freeze", {}).get("task_freeze_digest")
        != task_freeze_digest
        or not _same_vector(
            candidate.get("robot_base_position_world_m"),
            robot_base_pose_world.get("position_world_m"),
        )
        or candidate.get("selection_contract", {}).get(
            "object_label_or_task_id_geometry_shortcut_used"
        )
        is not False
        or candidate.get("selection_contract", {}).get(
            "candidate_geometry_authored_or_modified"
        )
        is not False
        or candidate.get("candidate", {}).get("pinch_span_within_stroke") is not True
    ):
        raise PairedTargetNativeManipulationPreflightError(
            f"paired_target_manipulation_interaction_affordance_invalid:{task_id}"
        )
    return _record(
        path,
        receipt_digest=candidate["receipt_digest"],
        selection_method=candidate["selection_contract"]["method"],
        candidate_link_id=candidate["candidate"]["link_id"],
        pinch_span_m=candidate["candidate"]["pinch_span_m"],
    )


def _scenario_carry_forward_accepted(
    *,
    proof: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    superseded_digest: str,
    amended_digest: str,
) -> bool:
    """One amendment-shaped divergence, accepted only with its exact proof."""

    if proof is None or not superseded_digest or not amended_digest:
        return False
    candidates = [proof] if isinstance(proof, Mapping) else list(proof)
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("sealed_schema") != SCENARIO_SCHEMA:
            continue
        try:
            validate_freeze_amendment_carry_forward_content(
                candidate,
                sealed_schema=SCENARIO_SCHEMA,
                superseded_digest=superseded_digest,
                amended_digest=amended_digest,
            )
        except FreezeAmendmentCarryForwardError:
            return False
        return True
    return False


def materialize_paired_target_native_manipulation_preflight(
    *,
    paired_target_preflight_path: str | Path,
    native_import_result_path: str | Path,
    task_records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    phase: str = "arena_packet",
    freeze_carry_forward: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bind available manipulation inputs and expose exact remaining blockers."""

    if phase not in PREFLIGHT_PHASES:
        raise PairedTargetNativeManipulationPreflightError(
            "paired_target_manipulation_phase_invalid"
        )
    if not 1 <= len(task_records) <= MAX_REPLACEMENT_OBJECTS:
        raise PairedTargetNativeManipulationPreflightError(
            "paired_target_manipulation_task_count_invalid"
        )
    preflight_path, preflight = _read(
        paired_target_preflight_path, "paired_target_manipulation_preflight_invalid"
    )
    preflight_tasks = preflight.get("tasks")
    if (
        preflight.get("schema_version") != PAIRED_PREFLIGHT_SCHEMA
        or preflight.get("receipt_digest")
        != canonical_digest(preflight, digest_field="receipt_digest")
        or tuple(preflight.get("candidate_ids") or ()) != FROZEN_CANDIDATES
        or not isinstance(preflight_tasks, list)
        or not 1 <= len(preflight_tasks) <= MAX_REPLACEMENT_OBJECTS
        or preflight.get("replacement_object_count") != len(preflight_tasks)
    ):
        raise PairedTargetNativeManipulationPreflightError(
            "paired_target_manipulation_preflight_invalid"
        )
    scene_id = str(preflight.get("scene_id") or "")
    preflight_by_task = {
        str(row.get("task_id") or ""): row
        for row in preflight_tasks
        if isinstance(row, Mapping)
    }
    if len(preflight_by_task) != len(preflight_tasks):
        raise PairedTargetNativeManipulationPreflightError(
            "paired_target_manipulation_preflight_tasks_invalid"
        )

    import_path, native_import = _read(
        native_import_result_path, "paired_target_manipulation_native_import_invalid"
    )
    import_rows = native_import.get("replacements")
    if (
        native_import.get("schema_version") != NATIVE_IMPORT_RESULT_SCHEMA
        or native_import.get("result_digest")
        != canonical_digest(native_import, digest_field="result_digest")
        or native_import.get("status") != "completed"
        or native_import.get("scene_id") != scene_id
        or native_import.get("native_isaac_executed") is not True
        or native_import.get("all_replacements_import_qualified") is not True
        or native_import.get("candidate_policy_queried") is not False
        or not isinstance(import_rows, list)
    ):
        raise PairedTargetNativeManipulationPreflightError(
            "paired_target_manipulation_native_import_invalid"
        )
    import_by_task = {
        str(row.get("task_id") or ""): row
        for row in import_rows
        if isinstance(row, Mapping)
    }
    expected_asset_ids = {
        str(row.get("asset_id") or "") for row in preflight_tasks
    }
    if (
        set(import_by_task) != set(preflight_by_task)
        or len(import_rows) != len(preflight_tasks)
        or {
            str(row.get("asset_id") or "")
            for row in import_rows
            if isinstance(row, Mapping)
        }
        != expected_asset_ids
        or any(
            row.get("native_simulator_import_qualified") is not True
            or row.get("blockers") != []
            for row in import_rows
            if isinstance(row, Mapping)
        )
    ):
        raise PairedTargetNativeManipulationPreflightError(
            "paired_target_manipulation_native_import_mismatch"
        )

    raw_task_ids = [str(row.get("task_id") or "") for row in task_records]
    if len(set(raw_task_ids)) != len(raw_task_ids) or set(raw_task_ids) != set(
        preflight_by_task
    ):
        raise PairedTargetNativeManipulationPreflightError(
            "paired_target_manipulation_task_set_mismatch"
        )

    opened: list[tuple[dict[str, Any], dict[str, Any]]] = []
    frozen_tasks: list[dict[str, Any]] = []
    for raw in task_records:
        task_id = str(raw.get("task_id") or "")
        freeze_path, raw_freeze = _read(
            raw.get("task_freeze_path"),
            f"paired_target_manipulation_task_freeze_invalid:{task_id}",
        )
        try:
            freeze = validate_task_freeze(raw_freeze)
        except DualTaskRehearsalContractError as exc:
            raise PairedTargetNativeManipulationPreflightError(
                f"paired_target_manipulation_task_freeze_invalid:{task_id}"
            ) from exc
        if freeze.get("task_id") != task_id:
            raise PairedTargetNativeManipulationPreflightError(
                f"paired_target_manipulation_task_freeze_mismatch:{task_id}"
            )
        placement_path, placement = _read(
            raw.get("franka_placement_packet_path"),
            f"paired_target_manipulation_franka_placement_invalid:{task_id}",
        )
        source = freeze["source_object"]
        selected = placement.get("target_analysis", {}).get("selected_target", {})
        if (
            placement.get("schema_version") != PLACEMENT_PACKET_SCHEMA_VERSION
            or placement.get("packet_digest")
            != canonical_digest(placement, digest_field="packet_digest")
            or placement.get("status") != "placement_candidate_materialized"
            or placement.get("native_contact_reachability_qualified") is not False
            or placement.get("policy_execution_authorized") is not False
            or placement.get("packet_digest")
            != source.get("franka_placement_packet_digest")
            or selected.get("target_label") != source.get("semantic_label")
            or not _same_vector(
                selected.get("position_m"),
                source.get("observed_pose_world", {}).get("position_world_m"),
            )
        ):
            raise PairedTargetNativeManipulationPreflightError(
                f"paired_target_manipulation_franka_placement_mismatch:{task_id}"
            )
        preflight_task = preflight_by_task[task_id]
        scenario_path, scenario_record = _bound_record(
            preflight_task.get("scenario_suite"),
            f"paired_target_manipulation_scenario_invalid:{task_id}",
        )
        _, scenario = _read(
            scenario_path, f"paired_target_manipulation_scenario_invalid:{task_id}"
        )
        registered_path, _ = _bound_record(
            preflight_task.get("registered_replacement_asset_receipt"),
            f"paired_target_manipulation_registered_asset_invalid:{task_id}",
        )
        _, registered = _read(
            registered_path,
            f"paired_target_manipulation_registered_asset_invalid:{task_id}",
        )
        # The scenario suite was prospectively sealed while the superseded
        # freeze was current: it carries policy candidates, seeds, and
        # controls, and reads only task identity from the freeze -- nothing a
        # joint-axis amendment touches. Re-sealing it would rewrite history;
        # a carry-forward proof pinned to this exact amendment bridges it,
        # the same acceptance the CAD receipts and visual binding already use.
        suite_freeze_ok = scenario.get("task_freeze_digest") == freeze[
            "task_freeze_digest"
        ] or _scenario_carry_forward_accepted(
            proof=freeze_carry_forward,
            superseded_digest=str(scenario.get("task_freeze_digest") or ""),
            amended_digest=str(freeze["task_freeze_digest"]),
        )
        if (
            scenario.get("schema_version") != SCENARIO_SCHEMA
            or scenario.get("suite_digest") != scenario_record.get("suite_digest")
            or scenario.get("suite_digest")
            != canonical_digest(scenario, digest_field="suite_digest")
            or scenario.get("scene_id") != scene_id
            or scenario.get("task_id") != task_id
            or not suite_freeze_ok
            or registered.get("schema_version") != REGISTERED_ASSET_SCHEMA
            or registered.get("receipt_digest")
            != canonical_digest(registered, digest_field="receipt_digest")
            or registered.get("task_freeze_digest") != freeze["task_freeze_digest"]
            or registered.get("task_id") != task_id
            or registered.get("asset_id") != preflight_task.get("asset_id")
        ):
            raise PairedTargetNativeManipulationPreflightError(
                f"paired_target_manipulation_task_binding_mismatch:{task_id}"
            )
        frozen_tasks.append(freeze)
        opened.append(
            (
                dict(raw),
                {
                    "task_id": task_id,
                    "asset_id": str(preflight_task["asset_id"]),
                    "registered_asset": registered,
                    "task_freeze": _record(
                        freeze_path, task_freeze_digest=freeze["task_freeze_digest"]
                    ),
                    "scenario_suite": _record(
                        scenario_path, suite_digest=scenario["suite_digest"]
                    ),
                    "franka_placement_packet": _record(
                        placement_path, packet_digest=placement["packet_digest"]
                    ),
                    "robot_base_pose_world": _pose_from_xyzyaw(
                        placement.get("placement", {}).get(
                            "robot_pose_xyzyaw_collision_stage"
                        )
                    ),
                    "native_import": dict(import_by_task[task_id]),
                    "review_camera_ids": list(
                        preflight_task.get("camera_index", {}).get("camera_ids") or []
                    ),
                },
            )
        )
    try:
        task_set = validate_task_freeze_set(frozen_tasks)
    except DualTaskRehearsalContractError as exc:
        raise PairedTargetNativeManipulationPreflightError(
            "paired_target_manipulation_task_freeze_set_invalid"
        ) from exc

    tasks: list[dict[str, Any]] = []
    blockers: list[str] = []
    pending_requirements: list[str] = []
    for raw, row in opened:
        affordance_path_value = raw.get("interaction_affordance_candidate_path")
        camera_rig_path_value = raw.get("native_camera_rig_candidate_path")
        arena_path_value = raw.get("native_task_arena_request_path")
        task_blockers: list[str] = []
        affordance_record: dict[str, Any] | None = None
        camera_rig_record: dict[str, Any] | None = None
        camera_rig: dict[str, Any] | None = None
        arena_record: dict[str, Any] | None = None
        if affordance_path_value in (None, ""):
            task_blockers.append("interaction_affordance_candidate_missing")
        else:
            affordance_path, affordance = _read(
                affordance_path_value,
                f"paired_target_manipulation_interaction_affordance_invalid:{row['task_id']}",
            )
            affordance_record = _validate_interaction_affordance_candidate(
                path=affordance_path,
                candidate=affordance,
                scene_id=scene_id,
                task_id=row["task_id"],
                asset_id=row["asset_id"],
                task_freeze_digest=row["task_freeze"]["task_freeze_digest"],
                registered_asset=row["registered_asset"],
                robot_base_pose_world=row["robot_base_pose_world"],
            )
        if camera_rig_path_value in (None, ""):
            task_blockers.append("native_camera_rig_candidate_missing")
        elif affordance_record is None:
            raise PairedTargetNativeManipulationPreflightError(
                f"paired_target_manipulation_camera_rig_without_affordance:{row['task_id']}"
            )
        else:
            camera_rig_path, camera_rig = _read(
                camera_rig_path_value,
                f"paired_target_manipulation_camera_rig_invalid:{row['task_id']}",
            )
            camera_rig_record = _validate_camera_rig_candidate(
                path=camera_rig_path,
                candidate=camera_rig,
                scene_id=scene_id,
                task_id=row["task_id"],
                affordance_record=affordance_record,
                placement_digest=row["franka_placement_packet"]["packet_digest"],
                robot_base_pose_world=row["robot_base_pose_world"],
            )
        construction_ready = (
            affordance_record is not None and camera_rig_record is not None
        )
        arena_pending = phase == "pre_arena"
        if arena_pending:
            if arena_path_value not in (None, ""):
                raise PairedTargetNativeManipulationPreflightError(
                    f"paired_target_manipulation_pre_arena_request_unexpected:{row['task_id']}"
                )
            pending_requirements.append(
                f"{row['task_id']}:native_task_arena_packet_request_missing"
            )
        elif arena_path_value in (None, ""):
            task_blockers.append("native_task_arena_packet_request_missing")
        else:
            arena_path, arena = _read(
                arena_path_value,
                f"paired_target_manipulation_arena_request_invalid:{row['task_id']}",
            )
            arena_record = _validate_arena_request(
                path=arena_path,
                request=arena,
                scene_id=scene_id,
                task_id=row["task_id"],
                task_freeze_digest=row["task_freeze"]["task_freeze_digest"],
                expected_asset_ids=expected_asset_ids,
                robot_base_pose_world=row["robot_base_pose_world"],
                expected_cameras=(camera_rig or {}).get("cameras") or [],
                expected_robot_reset=(camera_rig or {}).get(
                    "robot_joint_reset_positions_rad"
                )
                or {},
            )
        qualified = arena_record is not None and construction_ready
        task_row = {
            **{key: value for key, value in row.items() if key != "registered_asset"},
            "review_camera_count": len(row["review_camera_ids"]),
            "calibrated_review_camera_set_bound": len(row["review_camera_ids"]) == 8,
            "interaction_affordance_candidate": affordance_record,
            "native_camera_rig_candidate": camera_rig_record,
            "native_task_arena_request": arena_record,
            "policy_camera_and_interaction_contract_bound": (
                construction_ready if arena_pending else qualified
            ),
            "native_construction_binding_ready": construction_ready,
            "native_arena_packet_materialization_ready": qualified,
            "native_reachability_execution_authorized": False,
            "native_reachability_executed": False,
            "blockers": task_blockers,
            "pending_requirements": (
                ["native_task_arena_packet_request_missing"] if arena_pending else []
            ),
        }
        tasks.append(task_row)
        blockers.extend(f"{row['task_id']}:{value}" for value in task_blockers)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "ready_for_native_construction_bindings"
            if phase == "pre_arena" and not blockers
            else (
                "ready_for_native_arena_packet_materialization"
                if not blockers
                else "blocked_pending_native_manipulation_inputs"
            )
        ),
        "preflight_phase": phase,
        "program_id": "arm-decision-proof-v1",
        "scene_id": scene_id,
        "paired_target_preflight": _record(
            preflight_path, receipt_digest=preflight["receipt_digest"]
        ),
        "native_import_result": _record(
            import_path, result_digest=native_import["result_digest"]
        ),
        "replacement_object_count": len(tasks),
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
        "task_freeze_set_digest": task_set["set_digest"],
        "tasks": sorted(tasks, key=lambda row: row["task_id"]),
        "candidate_ids": list(FROZEN_CANDIDATES),
        "required_controls": ["zero_action_negative", "scripted_positive"],
        "native_import_qualified": True,
        "native_construction_bindings_ready": not blockers
        and all(row["native_construction_binding_ready"] for row in tasks),
        "calibrated_review_camera_requests_bound": all(
            row["calibrated_review_camera_set_bound"] for row in tasks
        ),
        "native_reachability_executed": False,
        "controls_executed": False,
        "learned_policies_executed": False,
        "blockers": sorted(blockers),
        "pending_requirements": sorted(pending_requirements),
        "generated_output_is_capture_or_physical_evidence": False,
        "claim_boundary": (
            "file_backed_native_manipulation_admission_only;native_import_is_not_"
            "camera_contact_reachability_controls_policy_or_physical_evidence"
        ),
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(
        result, digest_field="receipt_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise PairedTargetNativeManipulationPreflightError(
            "paired_target_manipulation_destination_exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return json.loads(json.dumps(result))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _, request = _read(args.request, "paired_target_manipulation_request_invalid")
    result = materialize_paired_target_native_manipulation_preflight(
        paired_target_preflight_path=request["paired_target_preflight_path"],
        native_import_result_path=request["native_import_result_path"],
        task_records=request["tasks"],
        output_path=args.output,
        phase=str(request.get("phase") or "arena_packet"),
    )
    print(
        json.dumps(
            {"status": result["status"], "receipt_digest": result["receipt_digest"]},
            sort_keys=True,
        )
    )
    return 0 if not result["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PairedTargetNativeManipulationPreflightError",
    "SCHEMA_VERSION",
    "materialize_paired_target_native_manipulation_preflight",
]
