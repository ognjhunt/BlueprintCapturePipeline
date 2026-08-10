"""Scene-neutral preregistration contracts for a two-task ADP rehearsal.

These contracts freeze selection and task definitions before learned-policy
execution.  They bind plans and identities, not claimed outcomes: removal,
asset, native-import, control, and episode qualifications remain separate
evidence receipts joined later by the construction/evaluation pipeline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .articulation_graph_contract import (
    ArticulationGraphContractError,
    validate_articulation_graph,
)
from .decision_evidence_contracts import canonical_digest


SELECTION_SCHEMA_VERSION = "dual_task_scene_selection_preregistration.v1"
SCENE_FREEZE_SCHEMA_VERSION = "dual_task_scene_freeze.v1"
TASK_FREEZE_SCHEMA_VERSION = "dual_task_task_freeze.v1"
JOIN_SCHEMA_VERSION = "dual_task_freeze_join.v1"
FROZEN_CANDIDATES = ("pi05_droid", "groot_n17_droid")
TASK_KINDS = frozenset({"articulated_interaction", "rigid_object_manipulation"})
REQUIRED_SELECTION_CRITERIA = frozenset(
    {
        "complete_topology_around_two_distinct_task_regions",
        "two_distinct_observed_source_objects",
        "one_mechanically_distinct_articulated_task",
        "one_rigid_object_manipulation_task",
        "support_or_destination_geometry",
        "per_task_franka_reachability_and_collision_free_base_space",
        "external_wrist_and_review_overview_camera_coverage",
        "interiorgs_appearance_availability",
        "sage_collision_availability",
        "shared_frame_metric_scale_axes_handedness_or_typed_abstention",
        "admissible_rights_and_provider_disclosure",
        "independent_masks_colliders_replacements_and_task_assumptions",
    }
)
FORBIDDEN_SELECTION_SIGNALS = frozenset(
    {
        "learned_policy_outcomes",
        "candidate_self_grading",
        "inpainting_quality_outcomes",
        "scene_id_specific_thresholds",
        "moving_unrelated_geometry_for_robot_fit",
    }
)


class DualTaskRehearsalContractError(ValueError):
    """Stable, sorted dual-task contract failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _digest(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71


def _identifier(value: Any) -> str:
    return str(value or "").strip()


def validate_selection_preregistration(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _clone(value)
    errors: list[str] = []
    if payload.get("schema_version") != SELECTION_SCHEMA_VERSION:
        errors.append("dual_task_selection_schema_invalid")
    if payload.get("program_id") != "arm-decision-proof-v1":
        errors.append("dual_task_selection_program_invalid")
    if payload.get("adp_item") != "ADP-009D" or payload.get("day_gate") != "public_scene_day_28":
        errors.append("dual_task_selection_gate_invalid")
    if payload.get("frozen_before_learned_policy_execution") is not True:
        errors.append("dual_task_selection_not_preregistered")
    if payload.get("learned_policy_outcomes_accessed") is not False:
        errors.append("dual_task_selection_outcome_leakage")
    if tuple(payload.get("candidate_ids") or []) != FROZEN_CANDIDATES:
        errors.append("dual_task_selection_candidates_invalid")
    criteria = _rows(payload.get("criteria"))
    criterion_ids = [_identifier(row.get("criterion_id")) for row in criteria]
    if set(criterion_ids) != REQUIRED_SELECTION_CRITERIA or len(criterion_ids) != len(
        REQUIRED_SELECTION_CRITERIA
    ):
        errors.append("dual_task_selection_criteria_invalid")
    if any(row.get("required") is not True or not _identifier(row.get("evidence_rule")) for row in criteria):
        errors.append("dual_task_selection_criterion_binding_invalid")
    if set(payload.get("forbidden_selection_signals") or []) != FORBIDDEN_SELECTION_SIGNALS:
        errors.append("dual_task_selection_forbidden_signals_invalid")
    candidates = _rows(payload.get("candidate_scenes"))
    scene_ids = [_identifier(row.get("publisher_scene_id")) for row in candidates]
    if (
        not scene_ids
        or any(not scene_id for scene_id in scene_ids)
        or len(scene_ids) != len(set(scene_ids))
        or scene_ids != sorted(scene_ids)
    ):
        errors.append("dual_task_selection_scene_ledger_invalid")
    if any(row.get("method_outcomes_consulted") is not False for row in candidates):
        errors.append("dual_task_selection_candidate_outcome_leakage")
    if payload.get("selected_scene_id") is not None:
        errors.append("dual_task_selection_contains_scene_choice")
    if payload.get("articulation_rule") != (
        "complete_joint_graph_no_universal_joint_count_cap_reject_only_if_unbounded_"
        "uncontrollable_unresettable_unscoreable_or_collision_unqualified"
    ):
        errors.append("dual_task_selection_articulation_rule_invalid")
    if payload.get("task_count") != 2:
        errors.append("dual_task_selection_task_count_invalid")
    if payload.get("claim_ceiling") != "development_only_public_dataset_rehearsal":
        errors.append("dual_task_selection_claim_ceiling_invalid")
    expected = canonical_digest(payload, digest_field="preregistration_digest")
    if payload.get("preregistration_digest") != expected:
        errors.append("dual_task_selection_digest_invalid")
    if errors:
        raise DualTaskRehearsalContractError(errors)
    return payload


def validate_scene_freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _clone(value)
    errors: list[str] = []
    if payload.get("schema_version") != SCENE_FREEZE_SCHEMA_VERSION:
        errors.append("dual_task_scene_freeze_schema_invalid")
    if not _digest(payload.get("selection_preregistration_digest")):
        errors.append("dual_task_scene_freeze_preregistration_digest_invalid")
    if payload.get("learned_policy_outcomes_accessed") is not False:
        errors.append("dual_task_scene_freeze_outcome_leakage")
    ledger = _rows(payload.get("candidate_ledger"))
    selected = [row for row in ledger if row.get("decision") == "selected"]
    rejected = [row for row in ledger if row.get("decision") == "rejected"]
    if len(selected) != 1 or len(ledger) != len(selected) + len(rejected):
        errors.append("dual_task_scene_freeze_ledger_decisions_invalid")
    selected_scene_id = _identifier(payload.get("selected_scene_id"))
    if len(selected) == 1 and _identifier(selected[0].get("publisher_scene_id")) != selected_scene_id:
        errors.append("dual_task_scene_freeze_selected_scene_mismatch")
    scene_ids = [_identifier(row.get("publisher_scene_id")) for row in ledger]
    if any(not scene_id for scene_id in scene_ids) or len(scene_ids) != len(set(scene_ids)):
        errors.append("dual_task_scene_freeze_ledger_invalid")
    if any(not _identifier(row.get("reason")) for row in ledger):
        errors.append("dual_task_scene_freeze_reason_missing")
    if any(row.get("method_outcomes_consulted") is not False for row in ledger):
        errors.append("dual_task_scene_freeze_candidate_outcome_leakage")
    if len(selected) == 1 and selected[0].get("previously_used") is not False:
        errors.append("dual_task_scene_freeze_selected_scene_previously_used")
    sources = payload.get("source_components")
    if not isinstance(sources, Mapping) or set(sources) != {"interiorgs", "sage_collision"}:
        errors.append("dual_task_scene_freeze_sources_invalid")
    else:
        for role, row in sources.items():
            if not isinstance(row, Mapping):
                errors.append(f"dual_task_scene_freeze_source_invalid:{role}")
                continue
            if (
                not _identifier(row.get("repository"))
                or not _identifier(row.get("revision"))
                or not _digest(row.get("sha256"))
                or not isinstance(row.get("size_bytes"), int)
                or int(row.get("size_bytes")) <= 0
                or not _identifier(row.get("license"))
                or row.get("rights_admitted") is not True
            ):
                errors.append(f"dual_task_scene_freeze_source_invalid:{role}")
    criterion_results = payload.get("criterion_results")
    if not isinstance(criterion_results, Mapping) or set(criterion_results) != REQUIRED_SELECTION_CRITERIA:
        errors.append("dual_task_scene_freeze_criterion_results_invalid")
    elif any(value is not True for value in criterion_results.values()):
        errors.append("dual_task_scene_freeze_unpassed_criterion")
    if not _digest(payload.get("topology_survey_digest")):
        errors.append("dual_task_scene_freeze_topology_digest_invalid")
    if not _digest(payload.get("reconnaissance_render_digest")):
        errors.append("dual_task_scene_freeze_render_digest_invalid")
    expected = canonical_digest(payload, digest_field="scene_freeze_digest")
    if payload.get("scene_freeze_digest") != expected:
        errors.append("dual_task_scene_freeze_digest_invalid")
    if errors:
        raise DualTaskRehearsalContractError(errors)
    return payload


def validate_task_freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _clone(value)
    errors: list[str] = []
    if payload.get("schema_version") != TASK_FREEZE_SCHEMA_VERSION:
        errors.append("dual_task_task_freeze_schema_invalid")
    if not _identifier(payload.get("task_id")):
        errors.append("dual_task_task_id_invalid")
    task_kind = _identifier(payload.get("task_kind"))
    if task_kind not in TASK_KINDS:
        errors.append("dual_task_task_kind_invalid")
    if not _digest(payload.get("scene_freeze_digest")):
        errors.append("dual_task_task_scene_digest_invalid")
    if tuple(payload.get("candidate_ids") or []) != FROZEN_CANDIDATES:
        errors.append("dual_task_task_candidates_invalid")
    if payload.get("frozen_before_learned_policy_execution") is not True:
        errors.append("dual_task_task_not_preregistered")
    if payload.get("learned_policy_outcomes_accessed") is not False:
        errors.append("dual_task_task_outcome_leakage")
    source = payload.get("source_object")
    if not isinstance(source, Mapping):
        errors.append("dual_task_source_object_missing")
    else:
        bounds = source.get("observed_bounds_world_m")
        if (
            not _identifier(source.get("instance_id"))
            or not _identifier(source.get("semantic_label"))
            or not isinstance(bounds, Mapping)
            or not isinstance(bounds.get("minimum"), list)
            or not isinstance(bounds.get("maximum"), list)
            or len(bounds.get("minimum") or []) != 3
            or len(bounds.get("maximum") or []) != 3
            or any(_finite(item) is None for item in (bounds.get("minimum") or []) + (bounds.get("maximum") or []))
            or not _identifier(source.get("support_or_attachment_id"))
        ):
            errors.append("dual_task_source_object_invalid")
    removal = payload.get("removal_plan")
    removal_fields = (
        "removal_id",
        "mask_set_id",
        "source_collider_prim_path",
        "collider_deletion_id",
        "replacement_asset_id",
        "replacement_qualification_id",
    )
    if not isinstance(removal, Mapping) or any(
        not _identifier(removal.get(field)) for field in removal_fields
    ):
        errors.append("dual_task_removal_plan_invalid")
    cameras = payload.get("cameras")
    if not isinstance(cameras, Mapping) or set(cameras) != {"external", "wrist", "overview"}:
        errors.append("dual_task_cameras_invalid")
    elif (
        any(not _identifier(cameras[role]) for role in cameras)
        or payload.get("overview_camera_policy_input") is not False
        or payload.get("overview_camera_deterministic_scoring_input") is not False
    ):
        errors.append("dual_task_camera_roles_invalid")
    execution = payload.get("execution_contract")
    if not isinstance(execution, Mapping):
        errors.append("dual_task_execution_contract_missing")
    else:
        for field in ("control_frequency_hz", "maximum_steps", "settle_window_steps"):
            raw = execution.get(field)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                errors.append(f"dual_task_execution_{field}_invalid")
        seeds = execution.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
        ):
            errors.append("dual_task_execution_seeds_invalid")
        if not _identifier(execution.get("canonical_scenario_cell_id")):
            errors.append("dual_task_execution_canonical_cell_invalid")
        if not isinstance(execution.get("reset_state"), Mapping):
            errors.append("dual_task_execution_reset_invalid")
    predicates = payload.get("deterministic_success_predicates")
    failures = payload.get("failure_rungs")
    if not isinstance(predicates, list) or not predicates or any(
        not _identifier(item) for item in predicates
    ):
        errors.append("dual_task_success_predicates_invalid")
    if not isinstance(failures, list) or not failures or any(
        not _identifier(item) for item in failures
    ):
        errors.append("dual_task_failure_rungs_invalid")
    if task_kind == "articulated_interaction":
        graph = payload.get("articulation_graph")
        if not isinstance(graph, Mapping):
            errors.append("dual_task_articulation_graph_missing")
        else:
            try:
                validate_articulation_graph(graph)
            except ArticulationGraphContractError as exc:
                errors.extend(exc.errors)
    elif payload.get("articulation_graph") not in (None, {}):
        errors.append("dual_task_rigid_articulation_graph_present")
    expected = canonical_digest(payload, digest_field="task_freeze_digest")
    if payload.get("task_freeze_digest") != expected:
        errors.append("dual_task_task_freeze_digest_invalid")
    if errors:
        raise DualTaskRehearsalContractError(errors)
    return payload


def validate_task_freeze_join(task_freezes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(task_freezes, (str, bytes)) or len(task_freezes) != 2:
        raise DualTaskRehearsalContractError(["dual_task_join_requires_exactly_two_tasks"])
    tasks = [validate_task_freeze(value) for value in task_freezes]
    errors: list[str] = []
    if len({_identifier(task["task_id"]) for task in tasks}) != 2:
        errors.append("dual_task_join_task_ids_not_distinct")
    if {task["task_kind"] for task in tasks} != TASK_KINDS:
        errors.append("dual_task_join_kinds_not_materially_distinct")
    if len({task["scene_freeze_digest"] for task in tasks}) != 1:
        errors.append("dual_task_join_scene_mismatch")
    source_ids = {_identifier(task["source_object"]["instance_id"]) for task in tasks}
    if len(source_ids) != 2:
        errors.append("dual_task_join_source_objects_not_distinct")
    independent_fields = (
        "removal_id",
        "mask_set_id",
        "source_collider_prim_path",
        "collider_deletion_id",
        "replacement_asset_id",
        "replacement_qualification_id",
    )
    for field in independent_fields:
        if len({_identifier(task["removal_plan"][field]) for task in tasks}) != 2:
            errors.append(f"dual_task_join_shared_{field}")
    if errors:
        raise DualTaskRehearsalContractError(errors)
    result: dict[str, Any] = {
        "schema_version": JOIN_SCHEMA_VERSION,
        "scene_freeze_digest": tasks[0]["scene_freeze_digest"],
        "task_freeze_digests": sorted(task["task_freeze_digest"] for task in tasks),
        "task_ids": sorted(task["task_id"] for task in tasks),
        "candidate_ids": list(FROZEN_CANDIDATES),
        "independence_fields": list(independent_fields),
        "independent": True,
        "join_digest": "",
    }
    result["join_digest"] = canonical_digest(result, digest_field="join_digest")
    return result


__all__ = [
    "DualTaskRehearsalContractError",
    "FROZEN_CANDIDATES",
    "JOIN_SCHEMA_VERSION",
    "REQUIRED_SELECTION_CRITERIA",
    "SCENE_FREEZE_SCHEMA_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "TASK_FREEZE_SCHEMA_VERSION",
    "TASK_KINDS",
    "validate_scene_freeze",
    "validate_selection_preregistration",
    "validate_task_freeze",
    "validate_task_freeze_join",
]
