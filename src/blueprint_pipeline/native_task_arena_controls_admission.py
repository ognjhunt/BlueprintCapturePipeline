"""Shared fail-closed admission for native-Arena construction and controls."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .adp009d_control_episode import REQUIRED_CONTROLS
from .decision_evidence_contracts import canonical_digest
from .native_task_camera_observability import (
    NativeTaskCameraObservabilityError,
    validate_native_task_policy_start_camera_observability,
)
from .native_task_construction_result_validation import (
    NativeTaskConstructionResultError,
    validate_qualified_rigid_construction_result,
)


def validate_native_task_controls_admission(
    *,
    scene_plan: Mapping[str, Any],
    construction_result: Mapping[str, Any],
    control_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Reprove the exact controls-before-policy boundary.

    This validator is shared by website readiness promotion and learned-policy
    bundle construction.  A website status therefore cannot become more
    permissive than the paid policy path it claims to unlock.
    """

    scene = json.loads(json.dumps(dict(scene_plan), allow_nan=False))
    construction = json.loads(
        json.dumps(dict(construction_result), allow_nan=False)
    )
    controls = json.loads(json.dumps(dict(control_result), allow_nan=False))
    pair = controls.get("control_pair")
    task_spec = scene.get("task_spec")
    scene_digest = scene.get("plan_digest")
    construction_digest = construction.get("result_digest")
    control_digest = controls.get("result_digest")
    cell_id = (scene.get("scenario") or {}).get("cell_id")
    errors: list[str] = []

    if (
        scene.get("schema_version") != "native_task_arena_scene_plan.v1"
        or scene_digest != canonical_digest(scene, digest_field="plan_digest")
        or not isinstance(task_spec, Mapping)
        or not str(scene.get("task_id") or "")
        or not str(cell_id or "")
        or not str(task_spec.get("prompt") or "")
    ):
        errors.append("native_task_policy_scene_plan_invalid")

    if (
        construction.get("schema_version")
        != "native_task_arena_construction_result.v1"
        or construction.get("status") != "completed"
        or construction.get("construction_gate_qualified") is not True
        or construction.get("blockers") != []
        or construction.get("candidate_policy_queried") is not False
        or construction.get("scene_plan_digest") != scene_digest
        or construction_digest
        != canonical_digest(construction, digest_field="result_digest")
    ):
        errors.append("native_task_policy_construction_not_qualified")
    if scene.get("task_kind") == "rigid_pick_place":
        try:
            validate_qualified_rigid_construction_result(
                scene_plan=scene,
                construction_result=construction,
            )
        except NativeTaskConstructionResultError as exc:
            errors.extend(exc.errors)
    else:
        try:
            validate_native_task_policy_start_camera_observability(construction)
        except NativeTaskCameraObservabilityError as exc:
            errors.extend(exc.errors)

    pair_valid = isinstance(pair, Mapping)
    if pair_valid:
        pair_controls = pair.get("controls")
        pair_valid = (
            pair.get("schema_version") == "adp_task_control_pair.v1"
            and pair.get("cell_id") == cell_id
            and pair.get("task_spec_digest") == canonical_digest(task_spec)
            and pair.get("execution_order") == list(REQUIRED_CONTROLS)
            and isinstance(pair_controls, list)
            and len(pair_controls) == len(REQUIRED_CONTROLS)
            and all(
                isinstance(row, Mapping)
                and row.get("control_id") == control_id
                and row.get("control_passed") is True
                and isinstance(row.get("receipt_digest"), str)
                and str(row["receipt_digest"]).startswith("sha256:")
                and len(str(row["receipt_digest"])) == 71
                for row, control_id in zip(
                    pair_controls, REQUIRED_CONTROLS, strict=True
                )
            )
            and pair.get("cell_admitted_for_policy_execution") is True
            and pair.get("policy_execution_blockers") == []
            and pair.get("candidate_policy_queried") is False
            and pair.get("pair_digest")
            == canonical_digest(pair, digest_field="pair_digest")
        )
    if (
        controls.get("schema_version") != "native_task_arena_control_result.v1"
        or controls.get("status") != "completed"
        or controls.get("controls_qualified") is not True
        or controls.get("blockers") != []
        or controls.get("candidate_policy_queried") is not False
        or controls.get("scene_plan_digest") != scene_digest
        or controls.get("construction_result_digest") != construction_digest
        or control_digest
        != canonical_digest(controls, digest_field="result_digest")
        or not pair_valid
    ):
        errors.append("native_task_policy_controls_not_qualified")

    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    return {
        "task_id": scene["task_id"],
        "cell_id": cell_id,
        "scene_plan_digest": scene_digest,
        "construction_result_digest": construction_digest,
        "control_result_digest": control_digest,
        "control_pair_digest": pair["pair_digest"],
        "packet_receipt_digest": controls.get("packet_receipt_digest"),
        "controls": [
            {
                "control_id": row["control_id"],
                "control_passed": row["control_passed"],
                "receipt_digest": row["receipt_digest"],
            }
            for row in pair["controls"]
        ],
    }


__all__ = ["validate_native_task_controls_admission"]
