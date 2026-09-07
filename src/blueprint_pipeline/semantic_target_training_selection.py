"""Admit a bounded training subset while retaining every final-review camera.

Development-only coverage policy: retain at least eight and 75% of the original
views; each excluded view needs two approved optical axes within 30 degrees.
This is a training admission heuristic, never a geometry or appearance proof.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .decision_evidence_contracts import canonical_digest

SCHEMA = "semantic_target_training_selection.v1"
MINIMUM_VIEWS = 8
MINIMUM_FRACTION = 0.75
MAX_NEIGHBOR_ANGLE_DEGREES = 30.0


def _require(ok, code):
    if not ok:
        raise ValueError("semantic_target_selection_" + code)


def build_selection(*, review_input, review_execution, transforms, minimum_views):
    _require(
        review_input.get("review_phase") == "pre_training_semantic_targets"
        and review_execution.get("review_phase") == "pre_training_semantic_targets",
        "phase_invalid",
    )
    _require(
        review_input.get("receipt_digest")
        == canonical_digest(review_input, digest_field="receipt_digest")
        and review_execution.get("execution_digest")
        == canonical_digest(review_execution, digest_field="execution_digest")
        and review_execution.get("final_composite_receipt_digest") == review_input["receipt_digest"]
        and review_execution.get("provider_called") is True,
        "review_binding_invalid",
    )
    _require(
        review_execution.get("schema_version")
        == "task_evaluation_artifixer_ai_visual_review_execution.v1"
        and review_execution.get("status") == "completed"
        and review_execution.get("all_frames_upright") is True
        and review_execution.get("response_store") is False
        and review_execution.get("tracing_disabled") is True
        and review_execution.get("raw_secret_values_recorded") is False,
        "review_execution_invalid",
    )
    frames = review_input["tasks"][0]["frames"]
    rows = review_execution["frames"]
    inventory = {r["camera_id"]: r for r in frames}
    decisions = {r["camera_id"]: r for r in rows}
    poses = {r["camera_id"]: r for r in transforms["frames"]}
    _require(
        len(inventory) == len(frames) == len(rows) == len(decisions)
        and set(inventory) == set(decisions) == set(poses),
        "inventory_invalid",
    )
    approved, excluded = [], []
    for camera, row in decisions.items():
        _require(
            row.get("frame_sha256") == inventory[camera]["final_frame"]["sha256"]
            and row.get("orientation_is_upright") is True,
            "frame_binding_or_orientation_invalid",
        )
        good = row.get("decision") == "accepted" and all(
            row.get(k) is True
            for k in (
                "source_object_absent",
                "repair_is_locally_plausible",
                "preserves_non_target_content",
            )
        )
        (approved if good else excluded).append(camera)
    _require(
        bool(excluded) or review_execution.get("decision") == "accepted",
        "unattributed_set_rejection",
    )
    required = max(MINIMUM_VIEWS, int(minimum_views), math.ceil(len(frames) * MINIMUM_FRACTION))
    _require(len(approved) >= required, "insufficient_approved_views")
    _require(
        len({canonical_digest({"matrix": poses[k]["transform_matrix"]}) for k in approved})
        >= required,
        "insufficient_distinct_approved_poses",
    )
    axes = {}
    for camera, row in poses.items():
        matrix = row["transform_matrix"]
        axis = [float(matrix[i][2]) for i in range(3)]
        length = math.sqrt(sum(x * x for x in axis))
        _require(math.isfinite(length) and abs(length - 1) < 0.01, "camera_rotation_invalid")
        axes[camera] = [x / length for x in axis]
    neighbors = {}
    for camera in excluded:
        adjacent = []
        for other in approved:
            cosine = sum(x * y for x, y in zip(axes[camera], axes[other]))
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            if angle <= MAX_NEIGHBOR_ANGLE_DEGREES:
                adjacent.append({"camera_id": other, "angle_degrees": angle})
        _require(len(adjacent) >= 2, "excluded_view_uncovered")
        neighbors[camera] = adjacent
    value = {
        "schema_version": SCHEMA,
        "status": "admitted_for_training_only",
        "review_input": review_input,
        "review_execution": review_execution,
        "review_input_digest": review_input["receipt_digest"],
        "review_execution_digest": review_execution["execution_digest"],
        "source_transforms": transforms,
        "transforms_digest": canonical_digest(transforms),
        "minimum_approved_views": required,
        "approved_camera_ids": sorted(approved),
        "excluded_camera_ids": sorted(excluded),
        "coverage_neighbors": neighbors,
        "reviewed_frame_digests": {k: v["final_frame"]["sha256"] for k, v in inventory.items()},
        "exclusion_reasons": {k: decisions[k]["rationale"] for k in excluded},
        "final_review_camera_ids": sorted(inventory),
        "appearance_repair_qualified": False,
        "rejected_teacher_slot": "original_outside_anchor_with_same_exclusion_mask",
    }
    value["selection_digest"] = canonical_digest(value, digest_field="selection_digest")
    return value


def validate_selection(selection, *, transforms, teacher_frames):
    _require(
        isinstance(selection, Mapping)
        and selection.get("schema_version") == SCHEMA
        and selection.get("status") == "admitted_for_training_only"
        and selection.get("selection_digest")
        == canonical_digest(selection, digest_field="selection_digest")
        and selection.get("transforms_digest") == canonical_digest(transforms)
        and selection.get("appearance_repair_qualified") is False,
        "seal_invalid",
    )
    rebuilt = build_selection(
        review_input=selection["review_input"],
        review_execution=selection["review_execution"],
        transforms=transforms,
        minimum_views=selection["minimum_approved_views"],
    )
    _require(rebuilt == selection, "review_selection_mismatch")
    cameras = {f["camera_id"] for f in teacher_frames}
    approved, excluded = (
        set(selection["approved_camera_ids"]),
        set(selection["excluded_camera_ids"]),
    )
    _require(
        not approved & excluded
        and approved | excluded == cameras
        and set(selection["final_review_camera_ids"]) == cameras
        and len(approved)
        >= max(
            MINIMUM_VIEWS,
            math.ceil(len(cameras) * MINIMUM_FRACTION),
            selection["minimum_approved_views"],
        ),
        "coverage_invalid",
    )
    for frame in teacher_frames:
        _require(
            selection["reviewed_frame_digests"][frame["camera_id"]]
            == frame["whole_frame_semantic_teacher"]["sha256"],
            "teacher_digest_mismatch",
        )
    # Recompute the geometric coverage rather than accepting a claimed neighbor list.
    axes = {
        r["camera_id"]: [float(r["transform_matrix"][i][2]) for i in range(3)]
        for r in transforms["frames"]
    }
    for camera in excluded:
        count = sum(
            sum(x * y for x, y in zip(axes[camera], axes[other]))
            >= math.cos(math.radians(MAX_NEIGHBOR_ANGLE_DEGREES))
            for other in approved
        )
        _require(count >= 2, "excluded_view_uncovered")
    return excluded


def validate_training_partition(task):
    """Verify that excluded teacher slots contain only masked original anchors."""
    frames = task.get("frames") or []
    selection = task.get("training_view_selection")
    excluded = (
        set(selection.get("excluded_camera_ids", [])) if isinstance(selection, Mapping) else set()
    )
    if selection is not None:
        _require(
            selection.get("selection_digest")
            == canonical_digest(selection, digest_field="selection_digest")
            and selection.get("schema_version") == SCHEMA
            and set(selection.get("final_review_camera_ids", []))
            == {f["camera_id"] for f in frames},
            "partition_selection_invalid",
        )
    anchors, teachers = [], []
    for row in frames:
        anchors.append(row["anchor_training_index"])
        is_excluded = row["camera_id"] in excluded
        _require(
            row.get("semantic_teacher_excluded_from_training", False) is is_excluded,
            "partition_flag_invalid",
        )
        index = row["semantic_teacher_training_index"]
        if is_excluded:
            _require(
                row["semantic_teacher_rgb"]["sha256"]
                == row["anchor_rgb"]["sha256"]
                == row["semantic_teacher_override_rgb"]["sha256"]
                and row.get("excluded_teacher_anchor_mask", {}).get("sha256")
                == row["anchor_loss_mask"]["sha256"],
                "excluded_teacher_pixels_or_mask_invalid",
            )
            anchors.append(index)
        else:
            teachers.append(index)
    _require(
        task.get("selected_anchor_indices") == anchors
        and task.get("semantic_teacher_indices") == teachers,
        "partition_indices_invalid",
    )
    return anchors, teachers
