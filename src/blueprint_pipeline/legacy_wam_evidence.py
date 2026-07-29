"""Compatibility helpers for pre-router WAM/classical-simulation artifacts.

The legacy WAM cross-check artifact is a request for later measurements.  It is
not calibration evidence and cannot qualify either WAM or a simulator.  These
helpers preserve that boundary while allowing the Decision/Evidence Router to
surface the requested engines as explicit candidate methods.
"""

from __future__ import annotations

from typing import Any, Mapping


WAM_CLASSICAL_SIM_CROSS_CHECK_SCHEMA_VERSION = "wam_classical_sim_cross_check_plan.v1"
WAM_SCORECARD_SCHEMA_PREFIX = "policy_ranking_scorecard."


def translate_wam_cross_check_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a legacy optional cross-check plan without granting authority."""

    if value.get("schema_version") != WAM_CLASSICAL_SIM_CROSS_CHECK_SCHEMA_VERSION:
        raise ValueError("legacy_wam_cross_check_schema_mismatch")
    requested = value.get("recommended_cross_checks")
    if not isinstance(requested, list):
        raise ValueError("legacy_wam_cross_check_methods_missing")
    supported = {
        "classical_sim_mujoco": "traditional_simulation",
        "classical_sim_isaac": "traditional_simulation",
    }
    candidates: list[dict[str, Any]] = []
    for method_id in sorted({str(item).strip() for item in requested if str(item).strip()}):
        family = supported.get(method_id)
        if family is None:
            continue
        candidates.append(
            {
                "legacy_method_id": method_id,
                "method_family": family,
                "candidate_source": WAM_CLASSICAL_SIM_CROSS_CHECK_SCHEMA_VERSION,
                "candidate_only": True,
                "availability_asserted": False,
                "qualification_granted": False,
                "execution_asserted": False,
                "promotion_effect": "none_without_exact_method_profile_and_qualification",
            }
        )
    return {
        "schema_version": "legacy_wam_evidence_candidate_translation.v1",
        "job_id": value.get("job_id"),
        "primary_evaluation_substrate": value.get("primary_evaluation_substrate"),
        "candidate_methods": candidates,
        "shared_dependency_warning": {
            "source": "legacy_wam_cross_check_plan",
            "warning": "agreement_is_not_independent_without_shared_dependency_review",
        },
        "qualification_granted": False,
        "execution_started": False,
    }


def wam_scorecard_as_debug_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bound a legacy WAM scorecard to debug evidence for adapter normalization.

    The returned mapping intentionally matches the stable adapter raw-result
    shape.  It never turns the scorecard into comparative-ranking qualification.
    """

    schema = str(value.get("schema_version") or "")
    if not schema.startswith(WAM_SCORECARD_SCHEMA_PREFIX):
        raise ValueError("legacy_wam_scorecard_schema_mismatch")
    if value.get("single_best_policy_claimed") is True or value.get("top_policy_id"):
        raise ValueError("legacy_wam_scorecard_unproven_winner_forbidden")
    status = str(value.get("status") or "")
    blockers = [str(item) for item in value.get("blockers") or [] if str(item)]
    if "blocked" not in status and "inconclusive" not in status:
        blockers.append("legacy_wam_scorecard_not_qualified_for_comparative_ranking")
    return {
        "status": "uncertain",
        "supports_claim": None,
        "categorical_finding": "thesis_not_supported",
        "uncertainty": 1.0,
        "coverage": 0.0,
        "blockers": sorted(set(blockers + ["debug_evidence_only"])),
        "invalid_rollout_reasons": [],
        "raw_artifact_references": [],
        "provenance": {
            "legacy_schema_version": schema,
            "self_grading_used_for_qualification": False,
            "generated_rollout_upgraded_physical_claim": False,
        },
        "claim_ceiling": {
            "comparative_policy_ranking": False,
            "physical_success": False,
            "deployment_readiness": False,
            "safety_certification": False,
        },
        "false_safe_risk": 1.0,
    }


__all__ = ["translate_wam_cross_check_plan", "wam_scorecard_as_debug_evidence"]
