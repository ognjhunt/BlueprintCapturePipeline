"""Machine-readable proof-path artifacts for exact-site package readiness."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .common import parse_bool, utc_now_iso
from .site_taxonomy import resolve_site_type


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


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


CONSENT_REVOKED_STATUSES = frozenset({"revoked", "withdrawn", "rescinded"})
KNOWN_CONSENT_USE_CLASSES = frozenset(
    {
        "derived_generation",
        "robot_evaluation",
        "model_training",
        "data_licensing",
        "commercial_licensing",
    }
)


def build_rights_provenance_review(
    *,
    rights_summary: Mapping[str, Any] | None,
    privacy_processing: Mapping[str, Any] | None,
    provenance_summary: Mapping[str, Any] | None,
    site_identity: Mapping[str, Any] | None,
    adjacent_systems: Sequence[str] | None,
    artifact_uris: Mapping[str, Any] | None = None,
    required_use_classes: Sequence[str] | None = None,
    site_type: str | None = None,
) -> Dict[str, Any]:
    rights = dict(rights_summary or {})
    privacy = dict(privacy_processing or {})
    provenance = dict(provenance_summary or {})
    consent_status = str(rights.get("consent_status") or "unknown").strip().lower()
    derived_generation_allowed = parse_bool(
        rights.get("derived_scene_generation_allowed"),
        default=False,
    )
    permission_document_uri = str(rights.get("permission_document_uri") or "").strip()
    privacy_status = str(privacy.get("status") or "not_run").strip().lower()
    provenance_status = str(provenance.get("status") or "missing").strip().lower()
    commercialization_terms = _mapping(
        rights.get("commercialization_terms")
        or rights.get("commercializationTerms")
        or rights.get("commercial_terms")
        or rights.get("commercialTerms")
    )
    revenue_share_terms = _mapping(
        rights.get("operator_revenue_terms")
        or rights.get("operatorRevenueTerms")
        or rights.get("revenue_share_terms")
        or rights.get("revenueShareTerms")
        or commercialization_terms.get("operator_revenue_terms")
        or commercialization_terms.get("revenue_share_terms")
        or commercialization_terms.get("revenue_share")
    )
    exclusivity_terms = _mapping(
        rights.get("exclusivity_terms")
        or rights.get("exclusivityTerms")
        or commercialization_terms.get("exclusivity_terms")
        or commercialization_terms.get("exclusivity")
    )

    # Revoked consent is an absolute stop: no downstream artifact may clear,
    # regardless of what the packet allowed before revocation.
    consent_revoked = (
        consent_status in CONSENT_REVOKED_STATUSES
        or parse_bool(rights.get("consent_revoked"), default=False)
        or bool(str(rights.get("consent_revoked_at") or "").strip())
    )

    # A "documented" consent claim without the document itself is an incomplete
    # rights packet, not documented consent — it must not clear.
    consent_evidence_complete = consent_status == "policy_only" or (
        consent_status == "documented" and bool(permission_document_uri)
    )

    # Use-class scope enforcement: when the caller declares what the artifact
    # is for (e.g. "robot_evaluation", "model_training", "derived_generation"),
    # an explicit consent_scope that omits that class is a "no" and blocks; an
    # unspecified scope cannot silently grant it and requires review.
    consent_scope = [item.lower() for item in _string_list(rights.get("consent_scope"))]
    consent_use_classes = [
        item for item in consent_scope if item in KNOWN_CONSENT_USE_CLASSES
    ]
    required_classes = [
        item.lower() for item in _string_list(required_use_classes)
    ]
    scope_blocked_classes: list[str] = []
    scope_unspecified = bool(required_classes) and not consent_use_classes
    if required_classes and consent_use_classes:
        scope_blocked_classes = [
            use_class
            for use_class in required_classes
            if use_class not in consent_use_classes
        ]

    rights_state = (
        "blocked"
        if consent_revoked or scope_blocked_classes
        else "cleared"
        if derived_generation_allowed and consent_evidence_complete and not scope_unspecified
        else "blocked"
        if not derived_generation_allowed
        else "needs_review"
    )
    fallback_redaction_used = privacy_status in {
        "face_anonymized_fallback",
        "full_frame_redacted_local_proof",
    }
    # R010: industrial sites (warehouses/factories/cold storage/stockrooms) expose
    # badges/screens/plates/signage that person-only redaction never targets. For
    # those sites, privacy may only clear when the industrial-sensitive classes
    # were actually handled. Industrial-ness is derived from the site_type text
    # and/or the privacy manifest's own ``is_industrial_site`` flag, so the gate
    # holds whether or not the site type is threaded through the call.
    site_type_resolution = resolve_site_type(site_type)
    is_industrial_site = site_type_resolution.is_industrial or parse_bool(
        privacy.get("is_industrial_site"), default=False
    )
    industrial_sensitive_classes_handled = parse_bool(
        privacy.get("industrial_sensitive_classes_handled"),
        default=False,
    )
    industrial_sensitive_gap = is_industrial_site and not industrial_sensitive_classes_handled
    privacy_would_clear = privacy_status in {
        "no_people_detected",
        "person_removed",
    }
    privacy_state = (
        "blocked"
        if privacy_status == "failed_closed"
        else "cleared"
        if privacy_would_clear and not industrial_sensitive_gap
        else "needs_review"
    )
    provenance_state = (
        "grounded"
        if provenance_status == "grounded"
        and bool((provenance.get("record") or {}).get("canonical_truth"))
        else "needs_review"
    )

    blockers: list[str] = []
    if consent_revoked:
        blockers.append("consent_revoked_takedown_required")
    for use_class in scope_blocked_classes:
        blockers.append(f"consent_scope_excludes_use_class:{use_class}")
    if scope_unspecified:
        blockers.append(
            "consent_scope_unspecified_for_required_use_classes:"
            + ",".join(required_classes)
        )
    if rights_state == "blocked" and not consent_revoked and not scope_blocked_classes:
        blockers.append("rights_not_sufficient_for_derived_generation")
    elif rights_state == "needs_review":
        blockers.append("rights_or_consent_requires_review")
        if consent_status == "documented" and not permission_document_uri:
            blockers.append("consent_documented_without_permission_document")
    if privacy_state == "blocked":
        blockers.append("privacy_processing_failed_closed")
    elif fallback_redaction_used:
        blockers.append("privacy_fallback_redaction_requires_manual_review")
    elif industrial_sensitive_gap:
        blockers.append("industrial_sensitive_classes_not_handled")
    elif privacy_state == "needs_review":
        blockers.append("privacy_processing_incomplete")
    if provenance_state != "grounded":
        blockers.append("provenance_not_grounded")

    overall_status = (
        "cleared"
        if not blockers
        else "blocked"
        if any(
            blocker
            in {
                "rights_not_sufficient_for_derived_generation",
                "privacy_processing_failed_closed",
                "consent_revoked_takedown_required",
                "industrial_sensitive_classes_not_handled",
            }
            or blocker.startswith("consent_scope_excludes_use_class:")
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
            "consent_revoked": consent_revoked,
            "consent_revoked_at": rights.get("consent_revoked_at"),
            "permission_document_uri": rights.get("permission_document_uri"),
            "consent_scope": _string_list(rights.get("consent_scope")),
            "consent_use_classes": consent_use_classes,
            "required_use_classes": required_classes,
            "scope_excluded_use_classes": scope_blocked_classes,
            "derived_scene_generation_allowed": derived_generation_allowed,
            "data_licensing_allowed": parse_bool(
                rights.get("data_licensing_allowed"),
                default=False,
            ),
            "commercialization_terms": commercialization_terms,
            "operator_revenue_terms": revenue_share_terms,
            "exclusivity_terms": exclusivity_terms,
            "revenue_share_commitment_made": False,
            "payout_commitment_allowed": False,
        },
        "privacy": {
            "status": privacy_state,
            "pipeline_status": privacy.get("status"),
            "mode": privacy.get("mode"),
            "fail_closed": parse_bool(privacy.get("fail_closed"), default=False),
            "raw_retained": parse_bool(privacy.get("raw_retained"), default=False),
            # Fallback redactions cleared the gate mechanically but were not
            # verified removals; they require human review before external
            # delivery or hosted review surfaces can clear.
            "fallback_redaction_used": fallback_redaction_used,
            "manual_review_recommended": fallback_redaction_used,
            "external_delivery_allowed": not fallback_redaction_used
            and privacy_state == "cleared",
            # R010: site-type-aware redaction coverage. For industrial sites the
            # industrial-sensitive classes (badges/screens/plates/signage) must be
            # handled before privacy can clear.
            "is_industrial_site": is_industrial_site,
            "industrial_sensitive_classes_handled": industrial_sensitive_classes_handled,
            "industrial_sensitive_classes": _string_list(
                privacy.get("industrial_sensitive_classes")
            ),
            "redaction_classes": _string_list(privacy.get("redaction_classes")),
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
    # Fail closed on rights/privacy. This manifest is synced to the WebApp as
    # authoritative state, so it must only read "ready" when rights+privacy review is
    # cleared. needs_review / blocked / missing must all block — matching
    # build_proof_pack_manifest — so an unverified-consent capture never projects as
    # ready. See beta-launch audit PIPE-02.
    rights_status = (
        str(rights_review.get("status") or "").strip()
        if isinstance(rights_review, Mapping)
        else ""
    )
    blockers: list[str] = []
    if rights_status != "cleared":
        blockers.append(f"rights_review:{rights_status or 'unavailable'}")
    if canonical_package_status == "registration_blocked":
        blockers.append("canonical_package_registration_blocked")
    if not site_world_spec:
        blockers.append("site_world_spec_missing")
    # A spec pointer is not runtime proof: the launchable export bundle and runtime
    # health must both back the claim before this manifest can read "ready".
    if launchable_status not in {"ready", "partial"}:
        blockers.append(f"launchable_export_not_ready:{launchable_status or 'missing'}")
    if not runtime_launchable:
        blockers.append("site_world_runtime_not_launchable")

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
    rights_review: Mapping[str, Any] | None = None,
    artifact_uris: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    blockers = list(dict.fromkeys([str(item).strip() for item in (demo_blockers or []) if str(item).strip()]))
    # Fail closed on rights/privacy. Hosted-review readiness is synced to the WebApp
    # as authoritative buyer/reviewer state, so it must not read "ready" unless
    # rights+privacy review is cleared. See beta-launch audit PIPE-02.
    rights_status = (
        str(rights_review.get("status") or "").strip()
        if isinstance(rights_review, Mapping)
        else ""
    )
    if rights_status != "cleared":
        blockers.append(f"rights_review:{rights_status or 'unavailable'}")
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
    rights_review: Mapping[str, Any] | None,
    hosted_review_readiness: Mapping[str, Any],
    artifact_uris: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    # Fail closed on a missing or malformed rights review: no review means the
    # rights packet is unavailable, never cleared.
    rights_status = (
        str(rights_review.get("status") or "").strip()
        if isinstance(rights_review, Mapping)
        else ""
    )
    proof_pack_ready = (
        str(site_package_manifest.get("status") or "") == "ready"
        and rights_status == "cleared"
    )
    blockers = []
    if str(site_package_manifest.get("status") or "") != "ready":
        blockers.append("site_package_not_ready")
    if rights_status != "cleared":
        blockers.append(f"rights_review:{rights_status or 'unavailable'}")
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
