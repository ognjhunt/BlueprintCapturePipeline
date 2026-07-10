"""Claim-bounded WAM backend strategy catalog.

The catalog keeps paper/model-family facts separate from runtime readiness.
Runtime code can recommend a backend without implying that the model ran, that
rank fidelity has been calibrated, or that generated media is physical proof.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


WAM_BACKEND_STRATEGY_SCHEMA_VERSION = "wam_backend_strategy_manifest.v1"
PREFERRED_CONFIGURED_LEARNED_WAM_BACKEND = "cosmos3_wam"

COSMOS3_WAM_PRECONDITIONS_SCHEMA_VERSION = "cosmos3_wam_precondition_checks.v1"
COSMOS3_WAM_ADAPTER_MODULE = "blueprint_pipeline.cosmos3_wam_command_adapter"
# SPEC-06 owns the external episode-consistency scorer; until one of these
# modules exists (or a caller passes an explicit availability flag), the
# scorer precondition fails and cosmos3_wam stays aspirational.
COSMOS3_CONSISTENCY_SCORER_MODULES = (
    "blueprint_pipeline.sc3_consistency_scorer",
    "blueprint_pipeline.wam_consistency_scorer",
)
COSMOS3_PROVIDER_COMMAND_ENVS = (
    "BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND",
)
COSMOS3_CHECKPOINT_ENVS = (
    "BLUEPRINT_COSMOS3_WAM_CHECKPOINT",
    "BLUEPRINT_COSMOS3_NANO_CHECKPOINT",
)
COSMOS3_CALIBRATION_ANCHORS_PATH_ENV = "BLUEPRINT_COSMOS3_CALIBRATION_ANCHORS_PATH"

SOURCE_URLS = {
    "oscar": "https://arxiv.org/html/2606.04463v2",
    "sc3_eval": "https://arxiv.org/html/2606.18610v3",
    "cosmos3_report": "https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf",
    "cosmos3_page": "https://research.nvidia.com/labs/cosmos-lab/cosmos3/",
    "cosmos3_blog": "https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/",
    "cosmos_predict25": "https://github.com/nvidia-cosmos/cosmos-predict2.5",
}

OSCAR_POLICY_EVAL_METRICS = {
    "metric_scope": "OSCAR skeleton-conditioned RoboArena policy-eval table",
    "mmrv": 0.571,
    "spearman": 0.750,
    "pearson": 0.852,
    "success_rate_difference_pp": 1.73,
}

OSCAR_DATASET_FACTS = {
    "filtered_episode_count": 180657,
    "filtered_robot_episode_count": 94830,
    "filtered_human_egocentric_episode_count": 85827,
    "robot_sources": ["RH20T", "InternData-A1", "DROID", "AgiBot", "AIROA-MoMa"],
    "human_sources": ["EgoDex", "EPIC-Kitchens"],
    "embodiments": [
        "Franka Panda",
        "KUKA iiwa",
        "AgiBot G1",
        "Toyota HSR",
        "human hand",
    ],
}

OSCAR_SUCCESS_SCORER_CAVEAT = {
    "scorer": "GPT-5 VLM in OSCAR paper",
    "human_label_calibration_clip_count": 100,
    "human_label_agreement_count": 78,
    "specificity": 0.90,
    "recall_caveat": "misses_about_one_third_of_real_successes",
    "blueprint_implication": "external generated-video success labels remain separate from consistency and rank-fidelity calibration",
}

SC3_EVAL_METRICS = {
    "headline_closed_loop": {
        "pearson": 0.929,
        "mmrv": 0.119,
        "scope": "overall SC3-Eval closed-loop result reported by the paper",
    },
    "in_distribution_online": {
        "sc3_eval_pearson": 0.984,
        "sc3_eval_mmrv": 0.022,
        "cosmos_predict25_pearson": 0.897,
        "cosmos_predict25_mmrv": 0.090,
    },
    "out_of_distribution_online": {
        "sc3_eval_pearson": 0.870,
        "sc3_eval_mmrv": 0.171,
        "cosmos_predict25_pearson": 0.871,
        "cosmos_predict25_mmrv": 0.195,
        "caveat": "SC3 improves MMRV on the OOD online split, but Pearson is effectively tied/slightly lower than Cosmos-Predict2.5",
    },
}

SC3_SCOPE_CAVEAT = {
    "paper_version": "arXiv:2606.18610v3",
    "source_reverified_on": "2026-07-02",
    "training_hours": 381,
    "physical_scene_count": 1,
    "object_category_count": 12,
    "camera_view_count": 3,
    "camera_views": ["two_third_person_cameras", "one_wrist_camera"],
    "policy_checkpoint_count": 7,
    "max_rollout_seconds": 20,
    "action_representation": "7d_delta_end_effector_pose",
    "blueprint_implication": "not proof of universal all-task or all-scene grading",
}

SC3_RECIPE_CONTRACT = {
    "recipe_id": "sc3_eval_self_consistency_recipe",
    "reliability_signals": [
        "forward_inverse_dynamics_consistency",
        "cross_view_consistency",
        "uncertainty_driven_early_termination",
    ],
    "required_blueprint_layers": [
        "synchronized_multi_view_cameras",
        "robot_camera_profile",
        "action_chunks",
        "initial_observations",
        "generated_rollout_frames",
        "policy_requery_trace",
        "success_criteria",
        "failure_taxonomy",
        "configured_wam_rollout_adapter",
        "generated_rollout_visual_smoke",
        "external_episode_consistency_scorer",
        "accepted_anchor_calibration_for_rank_fidelity_claims",
    ],
    "claim_boundary": {
        "self_consistency_is_reliability_signal_only": True,
        "self_consistency_does_not_label_task_success": True,
        "self_consistency_does_not_prove_generated_world_rank_fidelity": True,
        "self_consistency_does_not_prove_physical_robot_readiness": True,
        "self_consistency_does_not_prove_safety_validation": True,
    },
}

COMMON_RUNTIME_GATES = {
    "requires_explicit_adapter_command": True,
    "requires_explicit_model_checkpoint_or_provider_runtime": True,
    "requires_explicit_run_gate": True,
    "auto_run_allowed_without_gate": False,
}

COMMON_CLAIM_BOUNDARY = {
    "model_choice_does_not_prove_task_success": True,
    "model_choice_does_not_prove_policy_success": True,
    "model_choice_does_not_prove_generated_world_rank_fidelity": True,
    "model_choice_does_not_prove_physical_robot_readiness": True,
    "model_choice_does_not_prove_deployment_approval": True,
    "model_choice_does_not_prove_safety_validation": True,
    "model_choice_does_not_prove_sensor_truth": True,
    "model_choice_does_not_prove_contact_truth": True,
    "model_choice_does_not_allow_public_claim_upgrade": True,
}

BACKEND_STRATEGY_CATALOG: dict[str, dict[str, Any]] = {
    "oscar_wam": {
        "backend_id": "oscar_wam",
        "display_name": "OSCAR WAM",
        "runtime_role": "action_conditioned_world_model_rollout_generator",
        "strategy_role": "baseline_compatibility_and_evidence_lineage",
        "recommendation_tier": "baseline_compat",
        "preferred_for_new_configured_learned_wam": False,
        "default_local_runtime_candidate": False,
        "base_model": "Cosmos-Predict2.5-2B",
        "paper_lineage": "OSCAR 2606.04463v2",
        "fine_tuning_summary": OSCAR_DATASET_FACTS,
        "policy_eval_metrics": OSCAR_POLICY_EVAL_METRICS,
        "success_scorer_caveat": OSCAR_SUCCESS_SCORER_CAVEAT,
        "portable_lessons": [
            "2D kinematic-skeleton action conditioning",
            "multi-embodiment robot plus human-egocentric data curation",
            "RoboArena-style rank-fidelity calibration when accepted anchors exist",
        ],
        "source_urls": [SOURCE_URLS["oscar"], SOURCE_URLS["cosmos_predict25"]],
        "runtime_gates": COMMON_RUNTIME_GATES,
        "claim_boundary": COMMON_CLAIM_BOUNDARY,
    },
    "cosmos_wam": {
        "backend_id": "cosmos_wam",
        "display_name": "Cosmos-Predict2.5 WAM",
        "runtime_role": "legacy_world_video_rollout_or_review_substrate",
        "strategy_role": "legacy_advisory_baseline",
        "recommendation_tier": "legacy_baseline",
        "preferred_for_new_configured_learned_wam": False,
        "default_local_runtime_candidate": False,
        "base_model": "Cosmos-Predict2.5",
        "migration_note": "NVIDIA's Cosmos-Predict2.5 repository says the line is no longer under active development and future support is focused on Cosmos 3.",
        "source_urls": [SOURCE_URLS["cosmos_predict25"]],
        "runtime_gates": COMMON_RUNTIME_GATES,
        "claim_boundary": COMMON_CLAIM_BOUNDARY,
    },
    "cosmos3_wam": {
        "backend_id": "cosmos3_wam",
        "display_name": "Cosmos3-Nano WAM",
        "runtime_role": "preferred_configured_world_action_model_evaluator_candidate",
        "strategy_role": "preferred_new_configured_learned_wam_evaluator_candidate",
        "recommendation_tier": "preferred_configured_default_candidate",
        "preferred_for_new_configured_learned_wam": True,
        "default_local_runtime_candidate": False,
        "base_model": "Cosmos3-Nano",
        "model_family": "Cosmos 3",
        "reported_parameter_scale": "16B model built on a dense 8B transformer",
        "release_status_from_primary_source": "released_in_cosmos3_technical_report",
        "sc3_eval_lineage": {
            "initializes_from": "Cosmos3-Nano",
            "metrics": SC3_EVAL_METRICS,
            "scope_caveat": SC3_SCOPE_CAVEAT,
            "recipe_contract": SC3_RECIPE_CONTRACT,
        },
        "why_preferred": [
            "current Cosmos 3 line is the forward-supported NVIDIA Physical AI family",
            "SC3-Eval's stronger rank-fidelity recipe initializes from Cosmos3-Nano",
            "Cosmos 3 supports forward dynamics, inverse dynamics, and policy modes for action generation",
        ],
        "requires_adapter_calibration_and_external_consistency_scorer": True,
        "not_universal_grading_proof": True,
        "source_urls": [
            SOURCE_URLS["sc3_eval"],
            SOURCE_URLS["cosmos3_report"],
            SOURCE_URLS["cosmos3_page"],
            SOURCE_URLS["cosmos3_blog"],
        ],
        "runtime_gates": COMMON_RUNTIME_GATES,
        "claim_boundary": COMMON_CLAIM_BOUNDARY,
    },
    "cosmos3_super": {
        "backend_id": "cosmos3_super",
        "display_name": "Cosmos3-Super",
        "runtime_role": "high_cost_adjudication_candidate",
        "strategy_role": "high_cost_contested_ranking_adjudicator_candidate",
        "recommendation_tier": "high_cost_adjudication",
        "preferred_for_new_configured_learned_wam": False,
        "default_local_runtime_candidate": False,
        "base_model": "Cosmos3-Super",
        "model_family": "Cosmos 3",
        "reported_parameter_scale": "64B model built on a dense 32B transformer",
        "release_status_from_primary_source": "released_in_cosmos3_technical_report",
        "recommended_use": "contested or high-stakes evaluator runs after cheaper screens pass",
        "not_default_reason": "cost and hardware profile make it unsuitable as the default local path",
        "source_urls": [SOURCE_URLS["cosmos3_report"], SOURCE_URLS["cosmos3_page"]],
        "runtime_gates": COMMON_RUNTIME_GATES,
        "claim_boundary": COMMON_CLAIM_BOUNDARY,
    },
    "cosmos3_edge": {
        "backend_id": "cosmos3_edge",
        "display_name": "Cosmos3-Edge",
        "runtime_role": "future_or_unavailable_edge_candidate",
        "strategy_role": "announced_edge_candidate_not_default",
        "recommendation_tier": "not_released_default",
        "preferred_for_new_configured_learned_wam": False,
        "default_local_runtime_candidate": False,
        "base_model": "Cosmos3-Edge",
        "model_family": "Cosmos 3",
        "reported_parameter_scale": "4B model built on a dense 2B transformer",
        "release_status_from_primary_source": "technical_report_says_included_in_later_release",
        "treat_as_released_default": False,
        "source_urls": [SOURCE_URLS["cosmos3_report"]],
        "runtime_gates": COMMON_RUNTIME_GATES,
        "claim_boundary": COMMON_CLAIM_BOUNDARY,
    },
}


def _module_present(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _calibration_anchor_rows_from_path(anchors_path: Path) -> int:
    if not anchors_path.is_file():
        return 0
    try:
        payload = json.loads(anchors_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        raw = payload.get("anchors") or payload.get("usable_anchor_count")
        if isinstance(raw, int):
            return max(raw, 0)
        rows = raw if isinstance(raw, list) else []
    else:
        rows = []
    return sum(1 for row in rows if isinstance(row, Mapping))


def evaluate_cosmos3_wam_preconditions(
    *,
    calibration_anchors_path: str | Path | None = None,
    consistency_scorer_available: bool | None = None,
) -> dict[str, Any]:
    """Machine-check the strategy-doc preconditions for cosmos3_wam.

    The aspirational/preferred-candidate state is DERIVED from these checks,
    never asserted: until every check passes, cosmos3_wam must report
    ``aspirational: true``.
    """

    adapter_present = _module_present(COSMOS3_WAM_ADAPTER_MODULE)
    provider_env_present = [
        name for name in COSMOS3_PROVIDER_COMMAND_ENVS if str(os.getenv(name) or "").strip()
    ]
    checkpoint_env_present = [
        name
        for name in COSMOS3_CHECKPOINT_ENVS
        if Path(str(os.getenv(name) or "").strip() or "/nonexistent").expanduser().exists()
    ]
    checkpoint_or_provider_configured = bool(provider_env_present or checkpoint_env_present)
    scorer_modules_present = [
        name for name in COSMOS3_CONSISTENCY_SCORER_MODULES if _module_present(name)
    ]
    scorer_available = (
        bool(consistency_scorer_available)
        if consistency_scorer_available is not None
        else bool(scorer_modules_present)
    )
    anchors_path = Path(
        str(
            calibration_anchors_path
            or os.getenv(COSMOS3_CALIBRATION_ANCHORS_PATH_ENV)
            or ""
        ).strip()
        or "/nonexistent"
    ).expanduser()
    anchor_row_count = _calibration_anchor_rows_from_path(anchors_path)
    checks = {
        "adapter_module_present": {
            "passed": adapter_present,
            "detail": COSMOS3_WAM_ADAPTER_MODULE,
        },
        "checkpoint_or_provider_runtime_configured": {
            "passed": checkpoint_or_provider_configured,
            "detail": {
                "provider_command_envs_present": provider_env_present,
                "checkpoint_envs_present": checkpoint_env_present,
            },
        },
        "explicit_run_gates_defined": {
            "passed": True,
            "detail": [
                "BLUEPRINT_ALLOW_LOCAL_WAM_MODEL",
                "BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER",
            ],
        },
        "consistency_scorer_available": {
            "passed": scorer_available,
            "detail": {
                "scorer_modules_present": scorer_modules_present,
                "explicit_flag_used": consistency_scorer_available is not None,
            },
        },
        "calibration_anchors_present": {
            "passed": anchor_row_count > 0,
            "detail": {
                "anchors_path": str(anchors_path) if anchors_path.is_file() else None,
                "usable_anchor_row_count": anchor_row_count,
            },
        },
    }
    all_met = all(bool(check["passed"]) for check in checks.values())
    return {
        "schema_version": COSMOS3_WAM_PRECONDITIONS_SCHEMA_VERSION,
        "backend_id": "cosmos3_wam",
        "checks": checks,
        "preconditions_met": all_met,
        "aspirational": not all_met,
        "preferred_candidate_state": (
            "preferred_configured_candidate" if all_met else "aspirational"
        ),
        "state_derived_from_machine_checks": True,
        "state_asserted_manually": False,
    }


def _attach_derived_cosmos3_state(row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("backend_id")) != "cosmos3_wam":
        return row
    preconditions = evaluate_cosmos3_wam_preconditions()
    row["preconditions"] = preconditions
    row["aspirational"] = preconditions["aspirational"]
    row["preferred_candidate_state"] = preconditions["preferred_candidate_state"]
    return row


def get_wam_backend_strategy(backend_id: str) -> dict[str, Any]:
    """Return a copy of one backend strategy row."""

    row = BACKEND_STRATEGY_CATALOG.get(str(backend_id))
    if row is None:
        return {
            "backend_id": str(backend_id),
            "strategy_role": "unknown_or_non_wam_policy_candidate",
            "recommendation_tier": "not_cataloged",
            "preferred_for_new_configured_learned_wam": False,
            "default_local_runtime_candidate": False,
            "runtime_gates": COMMON_RUNTIME_GATES,
            "claim_boundary": COMMON_CLAIM_BOUNDARY,
        }
    return _attach_derived_cosmos3_state(copy.deepcopy(row))


def wam_backend_strategy_rows(
    backend_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return catalog rows, optionally filtered and preserving requested order."""

    if backend_ids is None:
        return [
            _attach_derived_cosmos3_state(copy.deepcopy(row))
            for row in BACKEND_STRATEGY_CATALOG.values()
        ]
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for backend_id in backend_ids:
        key = str(backend_id)
        if key in seen:
            continue
        seen.add(key)
        if key in BACKEND_STRATEGY_CATALOG:
            rows.append(get_wam_backend_strategy(key))
    return rows


def _copy_with_selection_flags(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_backend_ids: set[str],
    configured_backend_ids: set[str],
) -> list[dict[str, Any]]:
    flagged: list[dict[str, Any]] = []
    for row_value in rows:
        row = copy.deepcopy(dict(row_value))
        backend_id = str(row.get("backend_id") or "")
        row["selected_for_this_run"] = backend_id in selected_backend_ids
        row["configured_for_this_run"] = backend_id in configured_backend_ids
        flagged.append(row)
    return flagged


def build_wam_backend_strategy_manifest(
    *,
    generated_at: str,
    selected_backend_ids: Sequence[str] = (),
    configured_backend_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a proof-bound strategy manifest for WAM backend selection."""

    selected_set = {str(item) for item in selected_backend_ids}
    configured_set = {str(item) for item in configured_backend_ids}
    rows = _copy_with_selection_flags(
        wam_backend_strategy_rows(),
        selected_backend_ids=selected_set,
        configured_backend_ids=configured_set,
    )
    cosmos3_preconditions = evaluate_cosmos3_wam_preconditions()
    return {
        "schema_version": WAM_BACKEND_STRATEGY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "strategy_catalog_defined",
        "source_reverified_on": "2026-07-02",
        "source_reverified_from_primary_sources": True,
        "preferred_configured_learned_wam_backend_candidate": (
            PREFERRED_CONFIGURED_LEARNED_WAM_BACKEND
        ),
        "cosmos3_wam_preconditions": cosmos3_preconditions,
        "cosmos3_wam_aspirational": cosmos3_preconditions["aspirational"],
        "cosmos3_wam_preferred_candidate_state": cosmos3_preconditions[
            "preferred_candidate_state"
        ],
        "preferred_configured_backend_is_not_permanent_dependency": True,
        "backend_swap_boundary": (
            "model adapters sit behind Blueprint capture, observation/action, "
            "WAM rollout, scoring, and calibration contracts"
        ),
        "recommended_backend_ids_by_use_case": {
            "new_configured_learned_wam_evaluator": "cosmos3_wam",
            "baseline_compatibility_and_oscar_lineage": "oscar_wam",
            "legacy_cosmos_predict25_baseline": "cosmos_wam",
            "high_cost_contested_adjudication": "cosmos3_super",
            "edge_runtime": "cosmos3_edge_when_released_and_reverified",
        },
        "selected_backend_ids": list(selected_backend_ids),
        "configured_backend_ids": list(configured_backend_ids),
        "backend_strategies": rows,
        "paper_metric_facts": {
            "oscar_policy_eval": OSCAR_POLICY_EVAL_METRICS,
            "oscar_dataset": OSCAR_DATASET_FACTS,
            "oscar_success_scorer_caveat": OSCAR_SUCCESS_SCORER_CAVEAT,
            "sc3_eval": SC3_EVAL_METRICS,
            "sc3_scope_caveat": SC3_SCOPE_CAVEAT,
        },
        "sc3_recipe_contract": SC3_RECIPE_CONTRACT,
        "global_requirements_to_claim_external_rank_fidelity": [
            "accepted_real_world_anchor.v1 rows joined on scenario_eval_run_id, policy_id, task_id, and scenario_variation_instance_id",
            "actual success/failure result plus owner evidence and accepted calibration decision",
            "MMRV/Spearman/Pearson computed by sim_vs_real_calibration_report.json",
        ],
        "claim_boundary": {
            **COMMON_CLAIM_BOUNDARY,
            "cosmos3_preference_is_backend_strategy_not_public_accuracy_claim": True,
            "cosmos3_preferred_candidate_state_is_machine_derived_not_asserted": True,
            "cosmos3_preference_does_not_prove_universal_all_task_grading": True,
            "cosmos3_wam_never_auto_runs_without_explicit_adapter_and_gates": True,
            "oscar_and_cosmos_predict25_remain_baseline_compatibility_lineage": True,
            "cosmos3_edge_treated_as_released_default": False,
        },
    }
