"""Reuse full native policy authorities and actual per-cell controls admission.

A frozen source spec identifies a candidate; it cannot confer the source cell's
construction/control qualification on a varied cell. Derived cell specs keep
the frozen model/interface unchanged and bind the actual new predecessors.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .adp009d_policy_episode import maximum_policy_queries_for_task_spec
from .adp009d_policy_rights import (
    build_candidate_policy_rights,
    validate_candidate_policy_rights_authorities,
)
from .adp_episode_evidence_index import _episode_row
from .adp_task_scoring import OUTCOME_NEVER_MOVED, score_task_episode_from_spec
from .decision_evidence_contracts import canonical_digest
from .exact_workcell_variation_matrix import REQUIRED_CONTROLS, ExactWorkcellVariationError
from .native_task_arena_controls_admission import validate_native_task_controls_admission
from .native_task_arena_policy_bundle import validate_native_task_policy_execution_spec
from .policy_scientific_reset import compare_reset_readbacks, validate_reset_readback
from .task_control_diagnostic_boundary import build_task_control_pair


def validate_native_source_policy_candidates(
    *, source_scene_plan: Mapping[str, Any], candidate_specs: Mapping[str, Mapping[str, Any]],
    candidate_identity_digests: Mapping[str, str], readiness: Mapping[str, Any],
    scenario_suite: Mapping[str, Any],
) -> dict[str, Any]:
    """Reprove full checkpoint/license authority before any native runtime work.

    The historical readiness/suite remain the exact existing license/access
    authorities. Their historical task labels are not new-site performance
    evidence. This helper does not widen those authorities or replace them with
    a narrow self-digested rights projection.
    """
    source = source_scene_plan
    if (source.get("schema_version") != "native_task_arena_scene_plan.v1"
            or source.get("plan_digest") != canonical_digest(source, digest_field="plan_digest")):
        raise ExactWorkcellVariationError(["full_matrix_source_plan_invalid"])
    if set(candidate_specs) != {"pi05_droid", "groot_n17_droid"} or set(candidate_identity_digests) != set(candidate_specs):
        raise ExactWorkcellVariationError(["full_matrix_droid_pair_required"])
    for candidate_id, spec in candidate_specs.items():
        validate_native_task_policy_execution_spec(spec)
        if (spec.get("candidate_id") != candidate_id
                or spec.get("execution_spec_digest") != candidate_identity_digests[candidate_id]
                or spec.get("scene_plan_digest") != source["plan_digest"]
                or spec.get("task_id") != source.get("task_id")
                or spec.get("prompt") != source.get("task_spec", {}).get("prompt")
                or spec.get("max_policy_queries") != maximum_policy_queries_for_task_spec(
                    source["task_spec"], open_loop_horizon=spec["open_loop_horizon"])):
            raise ExactWorkcellVariationError(["full_matrix_frozen_candidate_binding_mismatch"])
        validate_candidate_policy_rights_authorities(
            spec["candidate_rights_binding"], readiness=readiness, scenario_suite=scenario_suite,
            candidate_id=candidate_id, policy_spec=spec["policy_spec"],
            runtime_robot_id=source["robot"]["robot_id"], scene_plan_digest=source["plan_digest"],
        )
    result = {
        "schema_version": "exact_workcell_native_source_admission.v1", "status": "admitted_candidate_inputs",
        "source_plan_digest": source["plan_digest"], "candidate_identity_digests": dict(candidate_identity_digests),
        "readiness_digest": readiness["readiness_digest"], "scenario_suite_digest": scenario_suite["suite_digest"],
        "cell_controls_qualification_inferred": False, "provider_mutation_performed": False,
    }
    result["admission_digest"] = canonical_digest(result, digest_field="admission_digest")
    return result


def derive_native_cell_policy_admission(
    *, source_scene_plan: Mapping[str, Any], source_candidate_specs: Mapping[str, Mapping[str, Any]],
    candidate_identity_digests: Mapping[str, str], scene_plan: Mapping[str, Any],
    construction_result: Mapping[str, Any], control_plan: Mapping[str, Any],
    control_receipt_paths: Mapping[str, Path], media_root: Path,
    readiness: Mapping[str, Any], scenario_suite: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit policy calls only after this exact cell's real controls and media.

    Actual retained control receipts are independently re-scored; lossless
    frames and review media are verified by the existing evidence index. The
    shared controls admission revalidates the exact per-cell construction.
    Candidate checkpoints, adapters, endpoints, query budgets and license/access
    authorities remain frozen to the source specs. This does not allocate, query,
    publish, or grant physical-validation authority.
    """
    source_admission = validate_native_source_policy_candidates(
        source_scene_plan=source_scene_plan, candidate_specs=source_candidate_specs,
        candidate_identity_digests=candidate_identity_digests, readiness=readiness, scenario_suite=scenario_suite,
    )
    scenario = scene_plan.get("scenario") or {}
    binding = scenario.get("exact_workcell_binding") or {}
    task = scene_plan["task_spec"]
    if (scene_plan.get("plan_digest") != canonical_digest(scene_plan, digest_field="plan_digest")
            or binding.get("source_plan_digest") != source_scene_plan["plan_digest"]
            or scene_plan.get("scene_id") != source_scene_plan.get("scene_id")
            or scene_plan.get("task_id") != source_scene_plan.get("task_id")
            or task.get("task_success_contract") != source_scene_plan["task_spec"].get("task_success_contract")
            or control_plan.get("plan_digest") != canonical_digest(control_plan, digest_field="plan_digest")
            or control_plan.get("cell_id") != scenario.get("cell_id")
            or control_plan.get("task_spec_digest") != canonical_digest(task)):
        raise ExactWorkcellVariationError(["full_matrix_cell_controls_scope_mismatch"])
    if set(control_receipt_paths) != set(REQUIRED_CONTROLS):
        raise ExactWorkcellVariationError(["full_matrix_required_control_receipts_missing"])
    receipts, resets = [], []
    for control_id in REQUIRED_CONTROLS:
        path = Path(control_receipt_paths[control_id])
        if path.is_symlink():
            raise ExactWorkcellVariationError(["full_matrix_control_receipt_symlink_forbidden"])
        receipt = json.loads(path.read_text())
        expected_episode_id = scenario["cell_id"] + "." + control_id
        if (receipt.get("control_id") != control_id or receipt.get("episode_id") != expected_episode_id
                or receipt.get("task_spec_digest") != canonical_digest(task)
                or receipt.get("task_spec") != task
                or receipt.get("control_plan_digest") != control_plan["plan_digest"]
                or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
                or receipt.get("control_passed") is not True or receipt.get("blockers") != []
                or receipt.get("candidate_policy_queried") is not False
                or receipt.get("grader_authority") != "deterministic_simulator_state"
                or receipt.get("phase_execution_blocker") is not None
                or receipt.get("qualification_allowed", True) is not True
                or receipt.get("diagnostic_only", False) is not False):
            raise ExactWorkcellVariationError(["full_matrix_control_receipt_not_admitted:" + control_id])
        recomputed = score_task_episode_from_spec(task_spec=task, samples=receipt.get("state_trace") or [])
        expected_success = control_id == "deterministic_scripted_positive"
        if (recomputed != receipt.get("score") or recomputed.get("status") != "scored"
                or recomputed.get("task_succeeded") is not expected_success
                or not expected_success and recomputed.get("outcome") != OUTCOME_NEVER_MOVED):
            raise ExactWorkcellVariationError(["full_matrix_control_score_not_admitted:" + control_id])
        _episode_row(media_root.resolve(), path.resolve())
        reset = validate_reset_readback(receipt.get("scientific_reset") or {})
        identity = reset["binding"]
        if (reset.get("complete") is not True or identity.get("candidate_id") != control_id
                or identity.get("cell_id") != scenario["cell_id"] or identity.get("seed") != scenario["seed"]
                or identity.get("task_spec_digest") != canonical_digest(task)
                or identity.get("matrix_cell_digest") != binding.get("cell_digest")
                or identity.get("matrix_reset_digest") != binding.get("reset_digest")):
            raise ExactWorkcellVariationError(["full_matrix_control_reset_not_admitted:" + control_id])
        receipts.append(receipt)
        resets.append(reset)
    if not compare_reset_readbacks(*resets)["comparison_eligible"]:
        raise ExactWorkcellVariationError(["full_matrix_control_reset_parity_unproven"])
    pair = build_task_control_pair(
        plan=control_plan, task=task, receipts=receipts, qualification_allowed=True,
        required_controls=REQUIRED_CONTROLS, canonical_digest=canonical_digest,
    )
    controls = {
        "schema_version": "native_task_arena_control_result.v1", "status": "completed",
        "controls_qualified": pair["cell_admitted_for_policy_execution"],
        "blockers": pair["policy_execution_blockers"], "candidate_policy_queried": False,
        "scene_plan_digest": scene_plan["plan_digest"],
        "construction_result_digest": construction_result["result_digest"], "control_pair": pair,
        "packet_receipt_digest": construction_result.get("packet_receipt_digest"),
    }
    controls["result_digest"] = canonical_digest(controls, digest_field="result_digest")
    admission = validate_native_task_controls_admission(
        scene_plan=scene_plan, construction_result=construction_result, control_result=controls,
    )
    cell_specs = {}
    for candidate_id, source_spec in source_candidate_specs.items():
        spec = copy.deepcopy(dict(source_spec))
        spec.update({
            "cell_id": scenario["cell_id"], "scene_plan_digest": admission["scene_plan_digest"],
            "construction_result_digest": admission["construction_result_digest"],
            "control_result_digest": admission["control_result_digest"],
            "control_pair_digest": admission["control_pair_digest"],
            "candidate_rights_binding": build_candidate_policy_rights(
                readiness, candidate_id=candidate_id, policy_spec=spec["policy_spec"],
                runtime_robot_id=scene_plan["robot"]["robot_id"], scene_plan_digest=scene_plan["plan_digest"],
            ),
        })
        spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
        validate_native_task_policy_execution_spec(spec)
        validate_candidate_policy_rights_authorities(
            spec["candidate_rights_binding"], readiness=readiness, scenario_suite=scenario_suite,
            candidate_id=candidate_id, policy_spec=spec["policy_spec"],
            runtime_robot_id=scene_plan["robot"]["robot_id"], scene_plan_digest=scene_plan["plan_digest"],
        )
        if spec["max_policy_queries"] != maximum_policy_queries_for_task_spec(task, open_loop_horizon=spec["open_loop_horizon"]):
            raise ExactWorkcellVariationError(["full_matrix_cell_query_budget_changed"])
        cell_specs[candidate_id] = spec
    result = {
        "schema_version": "exact_workcell_native_cell_admission.v1", "status": "admitted_policy_inputs",
        "source_admission_digest": source_admission["admission_digest"],
        "source_candidate_identity_digests": dict(candidate_identity_digests),
        "candidate_specs": cell_specs, "control_result": controls, "admission": admission,
        "qualification_authorized": False, "provider_mutation_performed": False,
    }
    result["admission_digest"] = canonical_digest(result, digest_field="admission_digest")
    return result
