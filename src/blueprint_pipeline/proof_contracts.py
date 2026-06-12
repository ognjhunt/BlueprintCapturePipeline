"""Machine-readable proof-path artifacts for exact-site package readiness."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .common import utc_now_iso


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _site_labeling(
    *,
    site_identity: Mapping[str, Any] | None,
    adjacent_systems: Sequence[str] | None,
) -> Dict[str, Any]:
    identity = dict(site_identity or {})
    adjacent = [item for item in _string_list(adjacent_systems) if item]
    exact_site_confident = bool(
        identity.get("site_id")
        or identity.get("site_name")
        or identity.get("address_full")
    )
    notes: list[str] = []
    if exact_site_confident:
        notes.append("Primary package scope is the captured exact site.")
    else:
        notes.append("Exact-site identity remains incomplete and needs operator review.")
    if adjacent:
        notes.append("Adjacent systems are contextual only and are not labeled as the primary site.")
    return {
        "site_scope": "exact_site" if exact_site_confident else "exact_site_review_required",
        "adjacent_context_included": bool(adjacent),
        "site_identity": identity,
        "adjacent_systems": adjacent,
        "notes": notes,
    }


def build_rights_provenance_review(
    *,
    rights_summary: Mapping[str, Any] | None,
    privacy_processing: Mapping[str, Any] | None,
    provenance_summary: Mapping[str, Any] | None,
    site_identity: Mapping[str, Any] | None,
    adjacent_systems: Sequence[str] | None,
    artifact_uris: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    rights = dict(rights_summary or {})
    privacy = dict(privacy_processing or {})
    provenance = dict(provenance_summary or {})
    consent_status = str(rights.get("consent_status") or "unknown").strip().lower()
    derived_generation_allowed = bool(rights.get("derived_scene_generation_allowed"))
    privacy_status = str(privacy.get("status") or "not_run").strip().lower()
    provenance_status = str(provenance.get("status") or "missing").strip().lower()

    rights_state = (
        "cleared"
        if derived_generation_allowed and consent_status in {"documented", "policy_only"}
        else "blocked"
        if not derived_generation_allowed
        else "needs_review"
    )
    privacy_state = (
        "cleared"
        if privacy_status
        in {
            "no_people_detected",
            "person_removed",
            "face_anonymized_fallback",
            "full_frame_redacted_local_proof",
        }
        else "blocked"
        if privacy_status == "failed_closed"
        else "needs_review"
    )
    provenance_state = (
        "grounded"
        if provenance_status == "grounded"
        and bool((provenance.get("record") or {}).get("canonical_truth"))
        else "needs_review"
    )

    blockers: list[str] = []
    if rights_state == "blocked":
        blockers.append("rights_not_sufficient_for_derived_generation")
    elif rights_state == "needs_review":
        blockers.append("rights_or_consent_requires_review")
    if privacy_state == "blocked":
        blockers.append("privacy_processing_failed_closed")
    elif privacy_state == "needs_review":
        blockers.append("privacy_processing_incomplete")
    if provenance_state != "grounded":
        blockers.append("provenance_not_grounded")

    overall_status = (
        "cleared"
        if not blockers
        else "blocked"
        if any(
            blocker in {"rights_not_sufficient_for_derived_generation", "privacy_processing_failed_closed"}
            for blocker in blockers
        )
        else "needs_review"
    )

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": overall_status,
        "site_labeling": _site_labeling(
            site_identity=site_identity,
            adjacent_systems=adjacent_systems,
        ),
        "rights": {
            "status": rights_state,
            "consent_status": rights.get("consent_status"),
            "permission_document_uri": rights.get("permission_document_uri"),
            "consent_scope": _string_list(rights.get("consent_scope")),
            "derived_scene_generation_allowed": derived_generation_allowed,
            "data_licensing_allowed": rights.get("data_licensing_allowed"),
        },
        "privacy": {
            "status": privacy_state,
            "pipeline_status": privacy.get("status"),
            "mode": privacy.get("mode"),
            "fail_closed": bool(privacy.get("fail_closed")),
            "raw_retained": bool(privacy.get("raw_retained")),
        },
        "provenance": {
            "status": provenance_state,
            "summary_status": provenance.get("status"),
            "grounding_level": (provenance.get("record") or {}).get("grounding_level"),
            "canonical_truth": bool((provenance.get("record") or {}).get("canonical_truth")),
        },
        "blockers": blockers,
        "artifacts": dict(artifact_uris or {}),
    }


def build_site_package_manifest(
    *,
    scene_id: str,
    capture_id: str,
    site_submission_id: str | None,
    opportunity_id: str | None,
    evaluation_prep_manifest: Mapping[str, Any],
    site_world_spec: Mapping[str, Any],
    site_world_registration: Mapping[str, Any],
    site_world_health: Mapping[str, Any],
    launchable_export_bundle: Mapping[str, Any],
    site_identity: Mapping[str, Any] | None,
    adjacent_systems: Sequence[str] | None,
    rights_review: Mapping[str, Any] | None,
    artifact_uris: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    canonical_package_status = str(
        evaluation_prep_manifest.get("canonical_package_status")
        or site_world_spec.get("canonical_package_status")
        or "unknown"
    ).strip()
    launchable_status = str(launchable_export_bundle.get("status") or "missing").strip().lower()
    runtime_launchable = bool(site_world_health.get("launchable"))
    blockers = [
        f"rights_review:{rights_review.get('status')}"
        for _ in [0]
        if isinstance(rights_review, Mapping) and str(rights_review.get("status") or "") == "blocked"
    ]
    if canonical_package_status == "registration_blocked":
        blockers.append("canonical_package_registration_blocked")
    if not site_world_spec:
        blockers.append("site_world_spec_missing")

    status = (
        "ready"
        if not blockers and canonical_package_status != "registration_blocked" and bool(site_world_spec)
        else "blocked"
    )

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "scene_id": scene_id,
        "capture_id": capture_id,
        "site_submission_id": site_submission_id,
        "opportunity_id": opportunity_id,
        "status": status,
        "site_labeling": _site_labeling(
            site_identity=site_identity,
            adjacent_systems=adjacent_systems,
        ),
        "package_status": {
            "canonical_package_status": canonical_package_status,
            "launchable_export_status": launchable_status,
            "runtime_launchable": runtime_launchable,
            "runtime_registration_status": site_world_registration.get("runtime_registration_status"),
            "site_world_health_status": site_world_health.get("status"),
        },
        "blockers": blockers,
        "artifacts": dict(artifact_uris or {}),
    }


def build_hosted_review_readiness(
    *,
    scene_id: str,
    capture_id: str,
    site_submission_id: str | None,
    opportunity_id: str | None,
    site_identity: Mapping[str, Any] | None,
    adjacent_systems: Sequence[str] | None,
    preview_manifest_uri: str | None,
    worldlabs_launch_url: str | None,
    runtime_demo_manifest_uri: str | None,
    demo_readiness_state: str,
    demo_blockers: Sequence[str] | None,
    site_world_health: Mapping[str, Any],
    launchable_export_bundle: Mapping[str, Any],
    artifact_uris: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    blockers = list(dict.fromkeys([str(item).strip() for item in (demo_blockers or []) if str(item).strip()]))
    if not preview_manifest_uri:
        blockers.append("preview_manifest_missing")
    if not worldlabs_launch_url:
        blockers.append("worldlabs_launch_url_missing")
    if not runtime_demo_manifest_uri:
        blockers.append("runtime_demo_manifest_missing")
    if not bool(site_world_health.get("launchable")):
        blockers.append("site_world_not_launchable")

    operator_ready = (
        demo_readiness_state == "ready"
        and bool(site_world_health.get("launchable"))
        and str(launchable_export_bundle.get("status") or "").strip().lower() in {"ready", "partial"}
    )
    webapp_ready = bool(preview_manifest_uri and worldlabs_launch_url and runtime_demo_manifest_uri)
    status = "ready" if operator_ready and webapp_ready and not blockers else "blocked"

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "scene_id": scene_id,
        "capture_id": capture_id,
        "site_submission_id": site_submission_id,
        "opportunity_id": opportunity_id,
        "status": status,
        "site_labeling": _site_labeling(
            site_identity=site_identity,
            adjacent_systems=adjacent_systems,
        ),
        "readiness": {
            "operator_ready": operator_ready,
            "webapp_ready": webapp_ready,
            "runtime_demo_readiness_state": demo_readiness_state,
            "runtime_launchable": bool(site_world_health.get("launchable")),
            "launchable_export_status": launchable_export_bundle.get("status"),
        },
        "blockers": blockers,
        "artifacts": {
            **dict(artifact_uris or {}),
            "preview_manifest_uri": preview_manifest_uri,
            "worldlabs_launch_url": worldlabs_launch_url,
            "runtime_demo_manifest_uri": runtime_demo_manifest_uri,
        },
    }


def build_proof_pack_manifest(
    *,
    scene_id: str,
    capture_id: str,
    site_submission_id: str | None,
    opportunity_id: str | None,
    site_package_manifest: Mapping[str, Any],
    rights_review: Mapping[str, Any],
    hosted_review_readiness: Mapping[str, Any],
    artifact_uris: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    proof_pack_ready = (
        str(site_package_manifest.get("status") or "") == "ready"
        and str(rights_review.get("status") or "") == "cleared"
    )
    blockers = []
    if str(site_package_manifest.get("status") or "") != "ready":
        blockers.append("site_package_not_ready")
    if str(rights_review.get("status") or "") != "cleared":
        blockers.append(f"rights_review:{rights_review.get('status')}")
    status = "ready" if proof_pack_ready else "blocked"

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "scene_id": scene_id,
        "capture_id": capture_id,
        "site_submission_id": site_submission_id,
        "opportunity_id": opportunity_id,
        "status": status,
        "proof_pack_ready": proof_pack_ready,
        "hosted_review_ready": str(hosted_review_readiness.get("status") or "") == "ready",
        "site_labeling": site_package_manifest.get("site_labeling") or {},
        "blockers": blockers,
        "artifacts": dict(artifact_uris or {}),
    }


def build_proof_path_status(
    *,
    scene_id: str,
    capture_id: str,
    site_submission_id: str | None,
    opportunity_id: str | None,
    rights_review: Mapping[str, Any],
    site_package_manifest: Mapping[str, Any],
    proof_pack_manifest: Mapping[str, Any],
    hosted_review_readiness: Mapping[str, Any],
) -> Dict[str, Any]:
    proof_pack_ready = str(proof_pack_manifest.get("status") or "") == "ready"
    hosted_review_ready = str(hosted_review_readiness.get("status") or "") == "ready"
    rights_cleared = str(rights_review.get("status") or "") == "cleared"
    event_statuses = []
    for order, (event_name, verified, detail) in enumerate(
        (
            (
                "proof_pack_delivered",
                proof_pack_ready and rights_cleared,
                "proof pack is ready and rights review is cleared",
            ),
            (
                "hosted_review_started",
                hosted_review_ready,
                "hosted review readiness is ready",
            ),
            (
                "hosted_review_follow_up_sent",
                proof_pack_ready and hosted_review_ready,
                "proof pack and hosted review are both ready",
            ),
            (
                "human_commercial_handoff_started",
                proof_pack_ready and hosted_review_ready and rights_cleared,
                "proof pack is ready, hosted review is ready, and rights are cleared",
            ),
        ),
        start=1,
    ):
        event_statuses.append(
            {
                "event_name": event_name,
                "status": "verified" if verified else "pending",
                "verified": bool(verified),
                "order": order,
                "detail": detail,
            }
        )
    next_step = (
        "clear_rights_or_privacy_review"
        if not rights_cleared
        else "finish_site_package"
        if str(site_package_manifest.get("status") or "") != "ready"
        else "prepare_hosted_review"
        if not hosted_review_ready
        else "operator_can_start_hosted_review"
    )
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "scene_id": scene_id,
        "capture_id": capture_id,
        "site_submission_id": site_submission_id,
        "opportunity_id": opportunity_id,
        "proof_pack_ready": proof_pack_ready,
        "hosted_review_ready": hosted_review_ready,
        "rights_cleared": rights_cleared,
        "next_truthful_step": next_step,
        "site_labeling": site_package_manifest.get("site_labeling") or rights_review.get("site_labeling") or {},
        "event_statuses": event_statuses,
    }
