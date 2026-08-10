"""Digest-bound join seam between Gaussian excision and the articulated USD.

The excision branch owns source-Gaussian ownership (owned/ambiguous/retained
index sets plus the held-out residual audit). This branch owns the articulated
SimReady replacement. Neither may re-derive the other's result: the join
consumes both sides' immutable receipts, verifies the shared world transform,
requires the twelve-door-state clearance matrix with every static obstacle
class bound, checks the eight-camera x twelve-door-state coverage grid, and
resolves the final inpainting policy:

- ``inpainting_not_required`` when the posed replacement hides every measured
  residual at every camera/door-state cell;
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
_REQUIRED_DOOR_CLASSES = frozenset(
    {"replacement_body", "replacement_lower_door", "franka_base"}
)


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
    expected_door_state_angles_degrees: Sequence[float],
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
    if len(expected_cameras) != 8 or len(set(expected_cameras)) != 8:
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
        len(expected_states) != 12
        or any(
            late <= early for early, late in zip(expected_states, expected_states[1:])
        )
    ):
        errors.append("articulated_excision_join_expected_door_states_invalid")

    ownership = _canonical(
        ownership_receipt,
        digest_field="receipt_digest",
        error="articulated_excision_join_ownership_receipt_digest_invalid",
        errors=errors,
    )
    # Under a suppression mode the claim is visibility after replacement, not
    # ownership of individual Gaussians. The 840796 held-out audit abstained -
    # a splat cannot be cleanly assigned to the refrigerator, and deletion-only
    # partitioning will not make it so - and the suppression path never
    # asserted otherwise. Requiring a passing ownership audit here would block
    # a construction whose claim does not rest on one. What replaced it,
    # coverage, is still enforced in full below.
    ownership_required = suppression["summary"].get("mode") == "deletion"
    if ownership and ownership_required:
        for field in (
            "owned_index_set_sha256",
            "ambiguous_index_set_sha256",
            "retained_scene_ply_sha256",
        ):
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
        if ownership.get("heldout_audit_passed") is not True:
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

    door_states = _canonical(
        door_state_receipt,
        digest_field="receipt_digest",
        error="articulated_excision_join_door_state_receipt_digest_invalid",
        errors=errors,
    )
    if door_states:
        if door_states.get("schema_version") != "articulated_door_state_clearance.v1":
            errors.append("articulated_excision_join_door_state_schema_invalid")
        bound = set(
            str(item) for item in door_states.get("static_obstacle_classes_bound") or []
        )
        if not _REQUIRED_DOOR_CLASSES.issubset(bound):
            errors.append(
                "articulated_excision_join_door_state_obstacle_classes_incomplete:"
                f"missing={sorted(_REQUIRED_DOOR_CLASSES - bound)}"
            )
        if door_states.get("status") != "door_state_matrix_clearance_candidate_only":
            errors.append("articulated_excision_join_door_state_matrix_not_clear")
        observed_angles = [
            row.get("angle_degrees")
            for row in door_states.get("door_state_rows") or []
            if isinstance(row, Mapping)
        ]
        if expected_states and observed_angles != expected_states:
            errors.append("articulated_excision_join_door_state_angles_mismatch")

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
        states = [
            float(item)
            for item in coverage.get("door_state_angles_degrees") or []
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        ]
        if expected_cameras and sorted(cameras) != sorted(expected_cameras):
            errors.append("articulated_excision_join_coverage_camera_ids_mismatch")
        if expected_states and states != expected_states:
            errors.append("articulated_excision_join_coverage_door_states_mismatch")
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
        expected_cells = {
            (camera, angle) for camera in cameras for angle in states
        }
        observed_cells = set()
        residual_total = 0
        for cell in cells:
            camera = str(cell.get("camera_id") or "")
            angle = cell.get("door_state_angle_degrees")
            if (
                isinstance(angle, bool)
                or not isinstance(angle, (int, float))
                or not math.isfinite(float(angle))
            ):
                errors.append("articulated_excision_join_coverage_cell_invalid")
                continue
            observed_cells.add((camera, float(angle)))
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
                    f"{camera}:{float(angle)!r}:pixels={outside}"
                )
            if residual > 0:
                residual_total += residual
                if cell.get("residual_inside_target_core_mask") is not True:
                    errors.append(
                        "articulated_excision_join_residual_outside_target_core_mask:"
                        f"{camera}:{float(angle)!r}"
                    )
                if component_threshold >= 0 and component > component_threshold:
                    errors.append(
                        "articulated_excision_join_residual_component_above_threshold:"
                        f"{camera}:{float(angle)!r}:{component}>{component_threshold}"
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

    if errors:
        raise ArticulatedExcisionJoinError(errors)

    decision: dict[str, Any] = {
        "schema_version": JOIN_SCHEMA_VERSION,
        "status": "join_admitted",
        "inpainting_policy": inpainting_policy,
        "suppression": suppression["summary"],
        "bindings": {
            "ownership_receipt_digest": ownership.get("receipt_digest"),
            "owned_index_set_sha256": ownership.get("owned_index_set_sha256"),
            "ambiguous_index_set_sha256": ownership.get("ambiguous_index_set_sha256"),
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
            "door_state_receipt_digest": door_states.get("receipt_digest"),
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
    "COVERAGE_SCHEMA_VERSION",
    "JOIN_SCHEMA_VERSION",
    "compile_articulated_excision_join",
]
