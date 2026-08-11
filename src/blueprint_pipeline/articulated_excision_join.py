"""Digest-bound join seam between Gaussian excision and the articulated USD.

The excision branch owns source-Gaussian ownership (owned/ambiguous/retained
index sets plus the held-out residual audit). This branch owns the replacement
USD and its bounded motion/pose coverage. Neither may re-derive the other's
result: the join consumes both sides' immutable receipts, verifies the shared
world transform, binds a task-specific clearance matrix, checks the complete
camera x state-cell coverage grid, and resolves the final inpainting policy:

- ``inpainting_not_required`` when the posed replacement hides every measured
  residual at every camera/state cell;
- ``narrow_mask_contained_seam_repair_only`` when residuals exist but every
  one stays inside the target-core mask and below the frozen component
  threshold;
- otherwise the join fails closed.

Untouched scene pixels must remain byte-identical: any measured change outside
the target mask blocks the join regardless of replacement quality.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


JOIN_SCHEMA_VERSION = "articulated_excision_join.v1"
# How the source object stops being visible. "deletion" is the sealed path: a
# forked scene file with those rows removed. The suppression modes leave the
# canonical scan untouched and apply the same index set at render or package
# time, so the object can be restored by dropping one small receipt. The mode
# changes only where the removal is applied - every coverage, collider,
# replacement, and door-state gate below is identical in all three.
SUPPRESSION_MODES = ("deletion", "render_time", "package_time")
COVERAGE_SCHEMA_VERSION = "articulated_excision_coverage.v1"
COVERAGE_CONDITIONED_CUTOUT_SCHEMA_VERSION = (
    "adp009b_coverage_conditioned_cutout.v1"
)
_REQUIRED_LEGACY_DOOR_CLASSES = frozenset(
    {"replacement_body", "replacement_lower_door", "franka_base"}
)
GENERIC_STATE_CLEARANCE_SCHEMA_VERSION = "articulated_state_cell_clearance.v1"


class ArticulatedExcisionJoinError(ValueError):
    """Stable, sorted join failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _clone(value: Any, *, error: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(error)
        return {}
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        errors.append(error)
        return {}
    if not isinstance(cloned, dict):
        errors.append(error)
        return {}
    return cloned


def _canonical(
    value: Any, *, digest_field: str, error: str, errors: list[str]
) -> dict[str, Any]:
    cloned = _clone(value, error=error, errors=errors)
    if not cloned:
        return {}
    if cloned.get(digest_field) != canonical_digest(cloned, digest_field=digest_field):
        errors.append(error)
        return {}
    return cloned


def _sha256_field(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _matrix4(value: Any) -> list[list[float]] | None:
    if not isinstance(value, Sequence) or len(value) != 4:
        return None
    rows: list[list[float]] = []
    for row in value:
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != 4
        ):
            return None
        values: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            number = float(item)
            if not math.isfinite(number):
                return None
            values.append(number)
        rows.append(values)
    return rows


def _state_ids(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result = [str(item) for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        return []
    return result


def compile_coverage_conditioned_cutout_receipt(
    *,
    bound_cutout_candidate: Mapping[str, Any],
    coverage_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit byte-exact deletion only when the posed replacement covers the target.

    This receipt intentionally does not rename a conservative deletion set as
    factual Gaussian ownership.  It binds an immutable cutout, exact retained
    rows, and an independently computed all-camera/all-state USD coverage
    audit.  Any uncovered pixels remain explicit seam candidates.
    """

    errors: list[str] = []
    cutout = _canonical(
        bound_cutout_candidate,
        digest_field="receipt_digest",
        error="coverage_conditioned_cutout_candidate_digest_invalid",
        errors=errors,
    )
    coverage = _canonical(
        coverage_receipt,
        digest_field="receipt_digest",
        error="coverage_conditioned_cutout_coverage_digest_invalid",
        errors=errors,
    )
    if cutout:
        if cutout.get("schema_version") != "adp009b_bound_index_union_candidate.v1":
            errors.append("coverage_conditioned_cutout_candidate_schema_invalid")
        if cutout.get("status") != (
            "bound_cutout_materialized_pending_coverage_and_seam_gates"
        ):
            errors.append("coverage_conditioned_cutout_candidate_status_invalid")
        counts = cutout.get("counts")
        preservation = cutout.get("preservation")
        outputs = cutout.get("outputs")
        if not isinstance(counts, Mapping):
            errors.append("coverage_conditioned_cutout_counts_invalid")
            counts = {}
        if not isinstance(preservation, Mapping):
            errors.append("coverage_conditioned_cutout_preservation_invalid")
            preservation = {}
        if not isinstance(outputs, Mapping):
            errors.append("coverage_conditioned_cutout_outputs_invalid")
            outputs = {}
        source = counts.get("source")
        deleted = counts.get("deleted_total")
        retained = counts.get("retained_total")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (source, deleted, retained)
            )
            or int(deleted or 0) + int(retained or 0) != int(source or -1)
        ):
            errors.append("coverage_conditioned_cutout_counts_invalid")
        if (
            preservation.get("retained_rows_byte_exact") is not True
            or preservation.get("retained_order_matches_source") is not True
            or preservation.get("retained_vertex_count") != retained
            or preservation.get("source_vertex_count") != source
        ):
            errors.append("coverage_conditioned_cutout_preservation_invalid")
        for name in (
            "deleted_source_indices",
            "retained_source_indices",
            "retained_scene_gaussians",
        ):
            record = outputs.get(name)
            if not isinstance(record, Mapping) or not _sha256_field(
                record.get("sha256")
            ):
                errors.append(f"coverage_conditioned_cutout_output_invalid:{name}")
        selection = cutout.get("selection")
        if (
            not isinstance(selection, Mapping)
            or selection.get("caller_asserted_coverage") is not False
            or selection.get("learned_policy_outcomes_used") is not False
            or selection.get("heldout_pixels_used_to_select_indices") is not False
        ):
            errors.append("coverage_conditioned_cutout_selection_invalid")
    if coverage:
        if coverage.get("schema_version") != COVERAGE_SCHEMA_VERSION:
            errors.append("coverage_conditioned_cutout_coverage_schema_invalid")
        if coverage.get("coverage_qualified") is not True:
            errors.append("coverage_conditioned_cutout_coverage_not_qualified")
        if coverage.get("caller_asserted_coverage_accepted") is not False:
            errors.append("coverage_conditioned_cutout_caller_assertion_forbidden")
        if coverage.get("rendered_pixels_changed_by_audit") is not False:
            errors.append("coverage_conditioned_cutout_pixel_mutation_forbidden")
    if errors:
        raise ArticulatedExcisionJoinError(errors)

    counts = cutout["counts"]
    outputs = cutout["outputs"]
    receipt: dict[str, Any] = {
        "schema_version": COVERAGE_CONDITIONED_CUTOUT_SCHEMA_VERSION,
        "status": "coverage_conditioned_cutout_admitted",
        "cutout_method": "byte_exact_deletion_plus_actual_usd_coverage",
        "bound_cutout_candidate_digest": cutout["receipt_digest"],
        "source_gaussian_count": counts["source"],
        "deleted_gaussian_count": counts["deleted_total"],
        "retained_scene_gaussian_count": counts["retained_total"],
        "deleted_index_set_sha256": outputs["deleted_source_indices"]["sha256"],
        "retained_index_set_sha256": outputs["retained_source_indices"]["sha256"],
        "retained_scene_ply_sha256": outputs["retained_scene_gaussians"]["sha256"],
        "retained_rows_byte_exact": True,
        "retained_order_matches_source": True,
        "coverage_receipt_digest": coverage["receipt_digest"],
        "coverage_qualified": True,
        "uncovered_pixels_are_explicit_seam_candidates": True,
        "broad_inpainting_authorized": False,
        "learned_policy_outcomes_used": False,
        "factual_gaussian_ownership_claimed": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def _normalize_suppression(
    *,
    mode: str,
    receipts: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    """Validate how the source object is hidden, without relaxing any gate."""

    rows = list(receipts)
    if mode not in SUPPRESSION_MODES:
        errors.append(f"articulated_excision_join_suppression_mode_unsupported:{mode}")
        return {"summary": {"mode": mode}, "digests": []}
    if mode == "deletion":
        if rows:
            errors.append(
                "articulated_excision_join_suppression_receipts_unexpected_for_deletion"
            )
        return {
            "summary": {
                "mode": "deletion",
                "canonical_scan_modified": True,
                "reversible": False,
                "task_ids": [],
            },
            "digests": [],
        }
    if not rows:
        errors.append("articulated_excision_join_suppression_receipts_missing")
        return {"summary": {"mode": mode}, "digests": []}

    scans: set[str] = set()
    task_ids: list[str] = []
    digests: list[str] = []
    for index, receipt in enumerate(rows):
        volume = _canonical(
            receipt,
            digest_field="receipt_digest",
            error=f"articulated_excision_join_suppression_receipt_{index}_digest_invalid",
            errors=errors,
        )
        if not volume:
            continue
        if volume.get("schema_version") != "gaussian_suppression_volume.v1":
            errors.append(
                f"articulated_excision_join_suppression_receipt_{index}_schema_invalid"
            )
        if volume.get("canonical_scan_modified") is not False:
            errors.append(
                "articulated_excision_join_suppression_canonical_scan_modified"
            )
        for field in ("canonical_scan_sha256", "suppressed_index_digest"):
            if not _sha256_field(volume.get(field)):
                errors.append(
                    f"articulated_excision_join_suppression_field_invalid:{field}"
                )
        scans.add(str(volume.get("canonical_scan_sha256")))
        task_ids.append(str(volume.get("task_id")))
        digests.append(str(volume.get("receipt_digest")))
    if len(scans) > 1:
        errors.append("articulated_excision_join_suppression_canonical_scan_mismatch")
    if len(set(task_ids)) != len(task_ids):
        errors.append("articulated_excision_join_suppression_task_ids_duplicated")
    return {
        "summary": {
            "mode": mode,
            "canonical_scan_modified": False,
            "reversible": True,
            "task_ids": sorted(task_ids),
            "canonical_scan_sha256": next(iter(scans)) if len(scans) == 1 else None,
            "volume_count": len(digests),
        },
        "digests": digests,
    }


def compile_articulated_excision_join(
    *,
    ownership_receipt: Mapping[str, Any],
    collider_removal_receipt: Mapping[str, Any],
    replacement_binding: Mapping[str, Any],
    door_state_receipt: Mapping[str, Any],
    coverage_receipt: Mapping[str, Any],
    expected_T_world_asset: Sequence[Sequence[float]],
    expected_camera_ids: Sequence[str],
    expected_door_state_angles_degrees: Sequence[float] = (),
    expected_state_cell_ids: Sequence[str] = (),
    transform_tolerance_m: float = 1e-6,
    suppression_mode: str = "deletion",
    suppression_receipts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Fail-closed join of the excision and articulated-replacement branches."""

    errors: list[str] = []

    suppression = _normalize_suppression(
        mode=suppression_mode, receipts=suppression_receipts, errors=errors
    )

    expected_transform = _matrix4(expected_T_world_asset)
    if expected_transform is None:
        errors.append("articulated_excision_join_expected_transform_invalid")
    expected_cameras = [str(item) for item in expected_camera_ids]
    if (
        not expected_cameras
        or any(not camera for camera in expected_cameras)
        or len(set(expected_cameras)) != len(expected_cameras)
    ):
        errors.append("articulated_excision_join_expected_camera_ids_invalid")
    expected_states: list[float] = []
    for value in expected_door_state_angles_degrees:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            errors.append("articulated_excision_join_expected_door_states_invalid")
            expected_states = []
            break
        expected_states.append(float(value))
    if expected_states and (
        any(
            late <= early for early, late in zip(expected_states, expected_states[1:])
        )
    ):
        errors.append("articulated_excision_join_expected_door_states_invalid")
    expected_state_ids = _state_ids(expected_state_cell_ids)
    if expected_state_cell_ids and not expected_state_ids:
        errors.append("articulated_excision_join_expected_state_cell_ids_invalid")
    if bool(expected_states) == bool(expected_state_ids):
        errors.append("articulated_excision_join_expected_state_binding_invalid")

    ownership = _canonical(
        ownership_receipt,
        digest_field="receipt_digest",
        error="articulated_excision_join_ownership_receipt_digest_invalid",
        errors=errors,
    )
    coverage_conditioned_ownership = bool(
        ownership.get("schema_version")
        == COVERAGE_CONDITIONED_CUTOUT_SCHEMA_VERSION
    )
    # A suppression mode makes a visibility claim, not a factual Gaussian-
    # ownership claim. Deletion still needs either a passed ownership audit or
    # the separately admitted byte-exact, coverage-conditioned cutout.
    ownership_required = suppression["summary"].get("mode") == "deletion"
    if ownership and ownership_required:
        required_digest_fields = (
            ("deleted_index_set_sha256", "retained_scene_ply_sha256")
            if coverage_conditioned_ownership
            else (
                "owned_index_set_sha256",
                "ambiguous_index_set_sha256",
                "retained_scene_ply_sha256",
            )
        )
        for field in required_digest_fields:
            if not _sha256_field(ownership.get(field)):
                errors.append(
                    f"articulated_excision_join_ownership_field_invalid:{field}"
                )
        source_count = ownership.get("source_gaussian_count")
        retained_count = ownership.get("retained_scene_gaussian_count")
        if (
            isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or source_count <= 0
            or isinstance(retained_count, bool)
            or not isinstance(retained_count, int)
            or not 0 < retained_count <= source_count
        ):
            errors.append("articulated_excision_join_ownership_counts_invalid")
        if coverage_conditioned_ownership:
            if (
                ownership.get("status") != "coverage_conditioned_cutout_admitted"
                or ownership.get("coverage_qualified") is not True
                or ownership.get("retained_rows_byte_exact") is not True
                or ownership.get("learned_policy_outcomes_used") is not False
                or ownership.get("factual_gaussian_ownership_claimed") is not False
            ):
                errors.append(
                    "articulated_excision_join_coverage_conditioned_cutout_invalid"
                )
        elif ownership.get("heldout_audit_passed") is not True:
            errors.append("articulated_excision_join_heldout_audit_not_passed")
    elif ownership and not ownership_required:
        # Recorded as disclosed context, never as an established claim.
        pass

    removal = _canonical(
        collider_removal_receipt,
        digest_field="receipt_digest",
        error="articulated_excision_join_collider_removal_receipt_digest_invalid",
        errors=errors,
    )
    if removal:
        if not str(removal.get("removed_prim_path") or ""):
            errors.append("articulated_excision_join_collider_removal_prim_missing")
        if removal.get("remaining_target_collision_prim_count") != 0:
            errors.append(
                "articulated_excision_join_source_collider_not_fully_removed"
            )
        if not _sha256_field(removal.get("removed_scene_usd_sha256")):
            errors.append(
                "articulated_excision_join_collider_removal_output_digest_missing"
            )

    replacement = _clone(
        replacement_binding,
        error="articulated_excision_join_replacement_binding_invalid",
        errors=errors,
    )
    if replacement:
        for field in (
            "replacement_usd_sha256",
            "topology_receipt_digest",
            "physics_receipt_digest",
        ):
            if not _sha256_field(replacement.get(field)):
                errors.append(
                    f"articulated_excision_join_replacement_field_invalid:{field}"
                )
        observed_transform = _matrix4(replacement.get("T_world_asset"))
        if observed_transform is None:
            errors.append("articulated_excision_join_replacement_transform_invalid")
        elif expected_transform is not None:
            deviation = max(
                abs(observed_transform[row][column] - expected_transform[row][column])
                for row in range(4)
                for column in range(4)
            )
            if deviation > float(transform_tolerance_m):
                errors.append(
                    "articulated_excision_join_replacement_world_transform_mismatch:"
                    f"deviation={deviation!r}"
                )

    state_clearance = _canonical(
        door_state_receipt,
        digest_field="receipt_digest",
        error="articulated_excision_join_door_state_receipt_digest_invalid",
        errors=errors,
    )
    if state_clearance:
        schema_version = state_clearance.get("schema_version")
        if schema_version == "articulated_door_state_clearance.v1":
            bound = set(
                str(item)
                for item in state_clearance.get("static_obstacle_classes_bound") or []
            )
            if not _REQUIRED_LEGACY_DOOR_CLASSES.issubset(bound):
                errors.append(
                    "articulated_excision_join_door_state_obstacle_classes_incomplete:"
                    f"missing={sorted(_REQUIRED_LEGACY_DOOR_CLASSES - bound)}"
                )
            if (
                state_clearance.get("status")
                != "door_state_matrix_clearance_candidate_only"
            ):
                errors.append("articulated_excision_join_door_state_matrix_not_clear")
            observed_angles = [
                row.get("angle_degrees")
                for row in state_clearance.get("door_state_rows") or []
                if isinstance(row, Mapping)
            ]
            if expected_states and observed_angles != expected_states:
                errors.append("articulated_excision_join_door_state_angles_mismatch")
            if expected_state_ids:
                errors.append(
                    "articulated_excision_join_door_state_schema_for_state_cells"
                )
        elif schema_version == GENERIC_STATE_CLEARANCE_SCHEMA_VERSION:
            bound = set(
                str(item)
                for item in state_clearance.get("static_obstacle_classes_bound") or []
            )
            required = set(
                str(item)
                for item in state_clearance.get("required_obstacle_classes_bound")
                or []
            )
            rows = [
                row
                for row in state_clearance.get("state_rows") or []
                if isinstance(row, Mapping)
            ]
            observed_state_ids = [str(row.get("cell_id") or "") for row in rows]
            if not required or not required.issubset(bound):
                errors.append(
                    "articulated_excision_join_state_obstacle_classes_incomplete"
                )
            if (
                state_clearance.get("status")
                != "state_cell_matrix_clearance_candidate_only"
            ):
                errors.append("articulated_excision_join_state_matrix_not_clear")
            if expected_state_ids and observed_state_ids != expected_state_ids:
                errors.append("articulated_excision_join_state_cell_ids_mismatch")
            if expected_states:
                errors.append(
                    "articulated_excision_join_state_schema_for_door_angles"
                )
            if any(row.get("clear") is not True for row in rows):
                errors.append("articulated_excision_join_state_matrix_not_clear")
        else:
            errors.append("articulated_excision_join_door_state_schema_invalid")

    coverage = _canonical(
        coverage_receipt,
        digest_field="receipt_digest",
        error="articulated_excision_join_coverage_receipt_digest_invalid",
        errors=errors,
    )
    inpainting_policy = None
    if coverage:
        if coverage.get("schema_version") != COVERAGE_SCHEMA_VERSION:
            errors.append("articulated_excision_join_coverage_schema_invalid")
        cameras = [str(item) for item in coverage.get("camera_ids") or []]
        if expected_cameras and sorted(cameras) != sorted(expected_cameras):
            errors.append("articulated_excision_join_coverage_camera_ids_mismatch")
        states: list[float] = []
        state_ids: list[str] = []
        if expected_states:
            states = [
                float(item)
                for item in coverage.get("door_state_angles_degrees") or []
                if isinstance(item, (int, float)) and not isinstance(item, bool)
            ]
            if states != expected_states:
                errors.append("articulated_excision_join_coverage_door_states_mismatch")
        else:
            state_ids = _state_ids(coverage.get("state_cell_ids") or [])
            if state_ids != expected_state_ids:
                errors.append("articulated_excision_join_coverage_state_cell_ids_mismatch")
        component_threshold = coverage.get(
            "maximum_residual_connected_component_pixels"
        )
        protected_threshold = coverage.get("maximum_protected_changed_pixels")
        if (
            isinstance(component_threshold, bool)
            or not isinstance(component_threshold, int)
            or component_threshold < 0
            or isinstance(protected_threshold, bool)
            or not isinstance(protected_threshold, int)
            or protected_threshold != 0
        ):
            errors.append("articulated_excision_join_coverage_thresholds_invalid")
            component_threshold = -1
        cells = [
            cell for cell in coverage.get("cells") or [] if isinstance(cell, Mapping)
        ]
        expected_cells = (
            {(camera, angle) for camera in cameras for angle in states}
            if expected_states
            else {(camera, state_id) for camera in cameras for state_id in state_ids}
        )
        observed_cells = set()
        residual_total = 0
        for cell in cells:
            camera = str(cell.get("camera_id") or "")
            if expected_states:
                angle = cell.get("door_state_angle_degrees")
                if (
                    isinstance(angle, bool)
                    or not isinstance(angle, (int, float))
                    or not math.isfinite(float(angle))
                ):
                    errors.append("articulated_excision_join_coverage_cell_invalid")
                    continue
                state_key: float | str = float(angle)
            else:
                state_key = str(cell.get("state_cell_id") or "")
                if not state_key:
                    errors.append("articulated_excision_join_coverage_cell_invalid")
                    continue
            observed_cells.add((camera, state_key))
            residual = cell.get("residual_significant_pixels")
            component = cell.get("residual_max_connected_component_pixels")
            outside = cell.get("outside_mask_changed_pixels")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (residual, component, outside)
            ):
                errors.append("articulated_excision_join_coverage_cell_invalid")
                continue
            if outside > 0:
                errors.append(
                    "articulated_excision_join_untouched_scene_pixels_changed:"
                    f"{camera}:{state_key!r}:pixels={outside}"
                )
            if residual > 0:
                residual_total += residual
                if cell.get("residual_inside_target_core_mask") is not True:
                    errors.append(
                        "articulated_excision_join_residual_outside_target_core_mask:"
                        f"{camera}:{state_key!r}"
                    )
                if component_threshold >= 0 and component > component_threshold:
                    errors.append(
                        "articulated_excision_join_residual_component_above_threshold:"
                        f"{camera}:{state_key!r}:{component}>{component_threshold}"
                    )
        if expected_cells and observed_cells != expected_cells:
            errors.append(
                "articulated_excision_join_coverage_grid_incomplete:"
                f"missing={len(expected_cells - observed_cells)}"
            )
        if not errors:
            inpainting_policy = (
                "inpainting_not_required"
                if residual_total == 0
                else "narrow_mask_contained_seam_repair_only"
            )
        if ownership_required and coverage_conditioned_ownership and ownership.get(
            "coverage_receipt_digest"
        ) != coverage.get("receipt_digest"):
            errors.append(
                "articulated_excision_join_cutout_coverage_binding_mismatch"
            )

    if errors:
        raise ArticulatedExcisionJoinError(errors)

    decision: dict[str, Any] = {
        "schema_version": JOIN_SCHEMA_VERSION,
        "status": "join_admitted",
        "inpainting_policy": inpainting_policy,
        "suppression": suppression["summary"],
        "bindings": {
            "ownership_receipt_digest": ownership.get("receipt_digest"),
            "cutout_method": ownership.get("cutout_method", "three_way_ownership"),
            "owned_index_set_sha256": ownership.get("owned_index_set_sha256"),
            "ambiguous_index_set_sha256": ownership.get("ambiguous_index_set_sha256"),
            "deleted_index_set_sha256": ownership.get("deleted_index_set_sha256"),
            "retained_scene_ply_sha256": ownership.get("retained_scene_ply_sha256"),
            "collider_removal_receipt_digest": removal.get("receipt_digest"),
            "removed_prim_path": removal.get("removed_prim_path"),
            "replacement_usd_sha256": replacement.get("replacement_usd_sha256"),
            "replacement_topology_receipt_digest": replacement.get(
                "topology_receipt_digest"
            ),
            "replacement_physics_receipt_digest": replacement.get(
                "physics_receipt_digest"
            ),
            "door_state_receipt_digest": state_clearance.get("receipt_digest"),
            "state_clearance_receipt_digest": state_clearance.get("receipt_digest"),
            "state_binding": (
                {
                    "kind": "door_angles",
                    "door_state_angles_degrees": expected_states,
                }
                if expected_states
                else {
                    "kind": "state_cell_ids",
                    "state_cell_ids": expected_state_ids,
                }
            ),
            "coverage_receipt_digest": coverage.get("receipt_digest"),
            "suppression_receipt_digests": suppression["digests"],
            "ownership_audit_status": str(ownership.get("status") or "unreported"),
            "ownership_audit_passed": ownership.get("heldout_audit_passed"),
            "T_world_asset": expected_transform,
        },
        "claim_boundary": {
            "gaussian_ownership_authored_here": False,
            "gaussian_ownership_established": ownership.get("heldout_audit_passed")
            is True,
            "visibility_after_replacement_is_the_criterion": not ownership_required,
            "hidden_interior_is_observed_truth": False,
            "native_simulator_qualified": False,
            "physical_equivalence_proven": False,
            "join_admission_is_not_policy_evaluation_authority": True,
        },
        "receipt_digest": "",
    }
    decision["receipt_digest"] = canonical_digest(
        decision, digest_field="receipt_digest"
    )
    return decision


__all__ = [
    "ArticulatedExcisionJoinError",
    "COVERAGE_CONDITIONED_CUTOUT_SCHEMA_VERSION",
    "COVERAGE_SCHEMA_VERSION",
    "JOIN_SCHEMA_VERSION",
    "compile_articulated_excision_join",
    "compile_coverage_conditioned_cutout_receipt",
]
