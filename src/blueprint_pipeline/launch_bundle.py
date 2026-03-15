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
) -> Dict[str, Any]:
    score = 100
    reasons: list[str] = []

    completeness_status = str(scorecard.get("completeness_status") or "").strip().lower()
    if completeness_status and completeness_status != "sufficient":
        score -= 30
        reasons.append("capture completeness is below launch threshold")

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

    if provider_status == "failed":
        score -= 5
        reasons.append("preview provider is unavailable")

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
) -> Dict[str, Any]:
    capture_rights = site_intake.get("capture_rights") if isinstance(site_intake.get("capture_rights"), Mapping) else {}
    task_scope = site_intake.get("task_scope") if isinstance(site_intake.get("task_scope"), Mapping) else {}
    missing_evidence = _string_list(readiness_decision.get("missing_evidence")) or _string_list(scorecard.get("missing_evidence"))

    return {
        "qualification_summary": {
            "readiness_state": qualification_record.get("readiness_state"),
            "confidence": qualification_record.get("confidence"),
            "task_statement": task_scope.get("task_statement"),
            "facility_template": task_scope.get("facility_template"),
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
        },
        "recapture_requirements": {
            "required": bool(missing_evidence),
            "missing_evidence": missing_evidence,
        },
        "preview_status": provider_run.get("status") or "not_requested",
    }
