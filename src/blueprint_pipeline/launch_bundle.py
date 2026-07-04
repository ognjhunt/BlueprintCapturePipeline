"""Launch-critical qualification and trust bundle helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def build_buyer_trust_score(
    *,
    descriptor: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    metadata: Mapping[str, Any],
    provider_status: str,
    fidelity_review: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    score = 100
    reasons: list[str] = []

    completeness_status = str(scorecard.get("completeness_status") or "").strip().lower()
    if completeness_status != "sufficient":
        score -= 30
        reasons.append(
            "capture completeness is below launch threshold"
            if completeness_status
            else "capture completeness evidence is missing"
        )

    confidence = float(qualification_record.get("confidence") or 0.0)
    if confidence < 0.75:
        score -= 20
        reasons.append("qualification confidence is still moderate")

    pose_match_rate = float((descriptor.get("quality") or {}).get("pose_match_rate") or 0.0)
    if pose_match_rate < 0.75:
        score -= 10
        reasons.append("pose consistency is limited")

    consent_status = str(metadata.get("consent_status") or metadata.get("capture_rights", {}).get("consent_status") or "").strip().lower()
    if consent_status not in {"documented", "policy_only"}:
        score -= 20
        reasons.append("rights and consent are incomplete")

    permission_uri = str(metadata.get("permission_document_uri") or metadata.get("capture_rights", {}).get("permission_document_uri") or "").strip()
    if not permission_uri:
        score -= 5
        reasons.append("no permission document is attached")

    normalized_review = fidelity_review if isinstance(fidelity_review, Mapping) else {}
    review_status = str(normalized_review.get("status") or "").strip().lower()
    review_scores = normalized_review.get("scores") if isinstance(normalized_review.get("scores"), Mapping) else {}
    coverage_score = float(review_scores.get("coverage") or 0.0)
    world_model_fitness = float(review_scores.get("world_model_fitness") or 0.0)
    if review_status != "succeeded":
        score -= 15
        reasons.append("multimodal capture review is incomplete")
    else:
        if coverage_score < 0.7:
            score -= 10
            reasons.append("Gemini review found coverage gaps in the capture")
        if world_model_fitness < 0.65:
            score -= 10
            reasons.append("Gemini review found limited world-model fitness")

    score = max(0, min(100, score))
    band = "high" if score >= 80 else "medium" if score >= 60 else "low"
    return {"score": score, "band": band, "reasons": reasons}


def build_launch_qualification_bundle(
    *,
    descriptor: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    readiness_decision: Mapping[str, Any],
    site_intake: Mapping[str, Any],
    buyer_trust_score: Mapping[str, Any],
    provider_run: Mapping[str, Any],
    privacy_processing: Mapping[str, Any] | None = None,
    fidelity_review: Mapping[str, Any] | None = None,
    world_model_fit_summary: Mapping[str, Any] | None = None,
    capturer_payout_recommendation: Mapping[str, Any] | None = None,
    provenance_summary: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    capture_rights = site_intake.get("capture_rights") if isinstance(site_intake.get("capture_rights"), Mapping) else {}
    if not capture_rights and isinstance(descriptor.get("metadata"), Mapping):
        raw_rights = descriptor["metadata"].get("capture_rights")
        if isinstance(raw_rights, Mapping):
            capture_rights = raw_rights
    task_scope = site_intake.get("task_scope") if isinstance(site_intake.get("task_scope"), Mapping) else {}
    if not task_scope:
        task_context = site_intake.get("task_context")
        task_scope = dict(task_context) if isinstance(task_context, Mapping) else {}
    missing_evidence = _string_list(readiness_decision.get("missing_evidence")) or _string_list(scorecard.get("missing_evidence"))
    normalized_review = fidelity_review if isinstance(fidelity_review, Mapping) else {}
    review_scores = normalized_review.get("scores") if isinstance(normalized_review.get("scores"), Mapping) else {}
    review_findings = normalized_review.get("findings") if isinstance(normalized_review.get("findings"), Mapping) else {}
    review_assessments = normalized_review.get("assessments") if isinstance(normalized_review.get("assessments"), Mapping) else {}
    recapture_recommendations = _string_list(review_findings.get("recapture_recommendations"))
    preview_status = provider_run.get("status") or "not_requested"
    normalized_privacy = dict(privacy_processing) if isinstance(privacy_processing, Mapping) else {}

    return {
        "qualification_summary": {
            "readiness_state": qualification_record.get("readiness_state"),
            "confidence": qualification_record.get("confidence"),
            "task_statement": task_scope.get("task_statement"),
            "facility_template": task_scope.get("facility_template"),
            "alpha_scoring_status": normalized_review.get("status") or "missing",
            "risk_count": len(qualification_record.get("risks", []))
            if isinstance(qualification_record.get("risks"), list)
            else 0,
        },
        "buyer_trust_score": dict(buyer_trust_score),
        "rights_and_compliance_summary": {
            "consent_status": capture_rights.get("consent_status"),
            "permission_document_uri": capture_rights.get("permission_document_uri"),
            "consent_scope": _string_list(capture_rights.get("consent_scope")),
            "data_licensing_allowed": capture_rights.get("data_licensing_allowed"),
            "derived_scene_generation_allowed": capture_rights.get("derived_scene_generation_allowed"),
        },
        "capture_quality_summary": {
            "completeness_status": scorecard.get("completeness_status"),
            "pose_match_rate": (descriptor.get("quality") or {}).get("pose_match_rate"),
            "qa_status": descriptor.get("qa_status"),
            "coverage_plan": descriptor.get("coverage_plan"),
            "capture_modality": descriptor.get("capture_modality"),
            "evidence_tier": descriptor.get("evidence_tier"),
            "gemini_review_status": normalized_review.get("status"),
            "coverage_score": review_scores.get("coverage"),
            "visual_clarity_score": review_scores.get("visual_clarity"),
            "lighting_stability_score": review_scores.get("lighting_stability"),
            "motion_stability_score": review_scores.get("motion_stability"),
            "blur_assessment": review_assessments.get("blur"),
            "lighting_assessment": review_assessments.get("lighting"),
            "motion_speed_assessment": review_assessments.get("motion_speed"),
            "doubling_back_assessment": review_assessments.get("doubling_back"),
            "coverage_completeness_assessment": review_assessments.get("coverage_completeness"),
            "task_zone_completeness_assessment": review_assessments.get("task_zone_completeness"),
            "occlusion_and_hidden_zone_assessment": review_assessments.get("occlusion_and_hidden_zone"),
            "depth_and_spatial_conditioning_assessment": review_assessments.get("depth_and_spatial_conditioning"),
            "privacy_processing": normalized_privacy,
        },
        "recapture_requirements": {
            "required": bool(missing_evidence or recapture_recommendations or normalized_review.get("status") != "succeeded"),
            "missing_evidence": missing_evidence,
            "recommendations": recapture_recommendations,
        },
        "preview_status": preview_status,
        "provider_preview_status": {
            "status": preview_status,
            "provider_name": provider_run.get("provider_name"),
            "provider_model": provider_run.get("provider_model"),
            "provider_run_id": provider_run.get("provider_run_id"),
            "failure_reason": provider_run.get("failure_reason"),
            "labeling": dict(provider_run.get("labeling") or {}),
        },
        "privacy_processing": normalized_privacy,
        "world_model_fit_summary": dict(world_model_fit_summary or {}),
        "capturer_payout_recommendation": dict(capturer_payout_recommendation or {}),
        "provenance_summary": dict(provenance_summary or {}),
        "gemini_fidelity_review": dict(normalized_review or {}),
    }
