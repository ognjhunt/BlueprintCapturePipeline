"""Buyer-facing readout for Task Evaluation Run / Post-Training Data Package exports.

A robot team deciding whether to trust a package needs one artifact that answers, in
plain language: what is in the package, what evidence backs it, what is still missing,
and what the package does NOT prove. This module derives that readout strictly from the
export manifest (plus an optional success-claim ledger) and fails closed: a package
missing any buyer-critical section produces a blocked readout, never a quiet pass.

The readout is a summary of existing evidence. It introduces no new claims, and its
claim boundary can only repeat or weaken what the success-claim ledger already proved.
"""

from __future__ import annotations

from typing import Any, Mapping

from .common import utc_now_iso
from .success_claim_contracts import CLAIM_LADDER

BUYER_PACKAGE_READOUT_SCHEMA_VERSION = "buyer_package_readout.v1"

# Sections a buyer-critical package must have. product_handoff is intentionally not
# here: pricing/entitlement wiring is out of band and must never gate evidence review.
BUYER_CRITICAL_SECTIONS: tuple[str, ...] = (
    "cards",
    "rights_privacy_provenance",
    "robot_pov_evidence",
    "failure_evidence",
    "task_success_criteria",
    "calibration",
    "media_provenance",
    "export_integrity",
    "replay_review",
)

_CARD_KEYS = ("site_card", "task_cards", "scenario_cards", "eval_cards")
_POV_KEYS = (
    "robot_pov_observation_manifest",
    "robot_pov_observations",
    "robot_pov_frame_sequence_manifest",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strict_bool(value: Any) -> bool:
    return value is True


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _section(status: str, blockers: list[str], **fields: Any) -> dict[str, Any]:
    return {"status": status, "blockers": sorted(set(blockers)), **fields}


def _presence_section(
    *, present: bool, missing_blocker: str, **fields: Any
) -> dict[str, Any]:
    blockers = [] if present else [missing_blocker]
    return _section("present" if present else "missing", blockers, **fields)


def build_buyer_package_readout(
    *,
    export_manifest: Mapping[str, Any],
    success_claim_ledger: Mapping[str, Any] | None = None,
    product_handoff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the buyer readout from a post-training data package export manifest.

    Fails closed: any buyer-critical section missing from the manifest yields
    ``status: blocked_incomplete_package`` with explicit blockers.
    """
    manifest = _mapping(export_manifest)
    included = _mapping(manifest.get("included_artifacts"))
    export_policy = _mapping(manifest.get("export_policy"))
    claim_boundary = _mapping(manifest.get("claim_boundary"))
    manifest_counts = _mapping(manifest.get("manifest_counts"))
    package_files = _mapping(manifest.get("package_files"))
    consent_evidence = _mapping(manifest.get("consent_evidence"))
    optional_exports = _mapping(manifest.get("optional_exports"))
    optional_formats = _mapping(optional_exports.get("formats"))
    video_bundle = _mapping(optional_formats.get("video_bundle"))
    lerobot_v3 = _mapping(optional_formats.get("lerobot_v3"))
    gr00t_lerobot = _mapping(optional_formats.get("gr00t_lerobot"))
    handoff = _mapping(product_handoff) or _mapping(manifest.get("product_handoff"))
    ledger = _mapping(success_claim_ledger)

    sections: dict[str, dict[str, Any]] = {}

    missing_cards = [key for key in _CARD_KEYS if key not in included]
    sections["cards"] = _section(
        "present" if not missing_cards else "missing",
        [f"card_missing:{key}" for key in missing_cards],
        site_card_included="site_card" in included,
        task_cards_included="task_cards" in included,
        scenario_cards_included="scenario_cards" in included,
        eval_cards_included="eval_cards" in included,
    )

    rights_present = "rights_packet" in included
    sections["rights_privacy_provenance"] = _presence_section(
        present=rights_present,
        missing_blocker="rights_packet_missing",
        rights_packet_included=rights_present,
        rights_privacy_scope_proven=_strict_bool(
            claim_boundary.get("rights_privacy_scope_proven")
        ),
        consent_evidence_record_included="consent_evidence" in included
        or bool(str(consent_evidence.get("path") or "").strip()),
        consent_evidence_present=consent_evidence.get("consent_evidence_present")
        is True,
        consent_evidence_status=consent_evidence.get("status"),
        provenance_chain_present="proof_boundaries" in included,
    )

    materialized_clip_count = _int_or_none(video_bundle.get("materialized_clip_count")) or 0
    missing_clip_file_count = _int_or_none(video_bundle.get("missing_clip_file_count")) or 0
    consumer_layout_complete = bool(
        lerobot_v3.get("consumer_layout_complete")
        or gr00t_lerobot.get("consumer_layout_complete")
    )
    pov_present = any(key in included for key in _POV_KEYS) or materialized_clip_count > 0
    pov_blockers: list[str] = []
    if not pov_present:
        pov_blockers.append("robot_pov_evidence_missing")
    if materialized_clip_count > 0 and missing_clip_file_count > 0:
        pov_blockers.append("declared_clip_files_missing")
    if materialized_clip_count > 0 and not consumer_layout_complete:
        pov_blockers.append("training_consumer_layout_incomplete")
    sections["robot_pov_evidence"] = _section(
        "present" if not pov_blockers else "missing",
        pov_blockers,
        pov_artifact_keys=[key for key in _POV_KEYS if key in included],
        clips_manifest_included=bool(export_policy.get("clips_manifest_included")),
        materialized_clip_count=materialized_clip_count,
        missing_clip_file_count=missing_clip_file_count,
        lerobot_v3_consumer_layout_complete=bool(
            lerobot_v3.get("consumer_layout_complete")
        ),
        gr00t_lerobot_consumer_layout_complete=bool(
            gr00t_lerobot.get("consumer_layout_complete")
        ),
    )

    failure_present = "failure_labels" in included
    sections["failure_evidence"] = _presence_section(
        present=failure_present,
        missing_blocker="failure_labels_missing",
        failure_label_count=_int_or_none(manifest_counts.get("failure_label_count")),
        failure_cases_preserved=failure_present,
    )

    criteria_present = "task_cards" in included and "eval_cards" in included
    sections["task_success_criteria"] = _presence_section(
        present=criteria_present,
        missing_blocker="task_success_criteria_source_missing",
        criteria_source="task_cards_and_eval_cards" if criteria_present else None,
    )

    calibration_present = "calibration_report" in included
    sections["calibration"] = _presence_section(
        present=calibration_present,
        missing_blocker="calibration_report_missing",
        calibration_report_included=calibration_present,
        camera_calibration_metadata_included="camera_calibration_metadata" in included,
    )

    media_blockers: list[str] = []
    generated_media_included = bool(export_policy.get("visual_augmentation_packet_included"))
    if generated_media_included and export_policy.get(
        "visual_augmentation_generated_videos_are_raw_capture_evidence"
    ) is not False:
        media_blockers.append("generated_media_not_segregated_from_raw_capture")
    sections["media_provenance"] = _section(
        "present" if not media_blockers else "missing",
        media_blockers,
        generated_media_included=generated_media_included,
        generated_media_is_raw_capture_evidence=False,
        raw_capture_is_authoritative=True,
    )

    integrity_blockers: list[str] = []
    if not str(manifest.get("checksums_path") or "").strip():
        integrity_blockers.append("checksums_manifest_missing")
    if not package_files:
        integrity_blockers.append("package_file_inventory_missing")
    sections["export_integrity"] = _section(
        "present" if not integrity_blockers else "missing",
        integrity_blockers,
        checksums_path=manifest.get("checksums_path"),
        package_file_count=len(package_files),
        schema_version=manifest.get("schema_version"),
    )

    replay_present = "replay_review_instructions" in included or bool(
        str(manifest.get("replay_review_instructions_path") or "").strip()
    )
    sections["replay_review"] = _presence_section(
        present=replay_present,
        missing_blocker="replay_review_instructions_missing",
        replay_review_instructions_path=manifest.get("replay_review_instructions_path")
        or included.get("replay_review_instructions"),
    )

    product_sku = str(handoff.get("product_sku") or "").strip() or None
    entitlement_id = str(handoff.get("entitlement_id") or "").strip() or None
    sections["product_handoff"] = _section(
        "optional_out_of_band",
        [],
        product_type=str(handoff.get("product_type") or "").strip() or None,
        product_sku=product_sku,
        entitlement_id=entitlement_id,
        buyer_review_url=str(handoff.get("buyer_review_url") or "").strip() or None,
        entitlement_wiring_present=bool(product_sku and entitlement_id),
        pricing_is_out_of_band=True,
    )

    blockers: list[str] = []
    export_status = str(manifest.get("status") or "").strip()
    if export_status != "export_ready_review_required":
        blockers.append(f"export_manifest_not_ready:{export_status or 'missing'}")
    for name in BUYER_CRITICAL_SECTIONS:
        blockers.extend(f"{name}:{blocker}" for blocker in sections[name]["blockers"])

    highest_claim = str(ledger.get("highest_truthful_claim") or "").strip()
    if highest_claim not in CLAIM_LADDER:
        highest_claim = "no_claim"

    readout_claim_boundary = {
        "highest_truthful_claim": highest_claim,
        "success_claim_ledger_present": bool(ledger),
        "physical_deployment_ready": highest_claim == "physical_deployment_ready",
        "package_purchase_is_not_deployment_approval": True,
        "generated_media_is_not_physical_proof": True,
        "simulator_results_are_not_real_world_outcomes": True,
        "readout_summarizes_existing_evidence_only": True,
    }

    status = "buyer_readout_ready_review_required" if not blockers else (
        "blocked_incomplete_package"
    )
    return {
        "schema_version": BUYER_PACKAGE_READOUT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "scene_id": manifest.get("scene_id"),
        "capture_id": manifest.get("capture_id"),
        "package_type": manifest.get("package_type"),
        "export_manifest_status": export_status or None,
        "status": status,
        "blockers": sorted(set(blockers)),
        "sections": sections,
        "claim_boundary": readout_claim_boundary,
    }


def render_buyer_package_readout_markdown(readout: Mapping[str, Any]) -> str:
    """Render the readout as a short buyer-readable markdown summary."""
    data = _mapping(readout)
    sections = _mapping(data.get("sections"))
    boundary = _mapping(data.get("claim_boundary"))
    lines = [
        "# Package Readout",
        "",
        f"- Scene: {data.get('scene_id') or 'unknown'}",
        f"- Capture: {data.get('capture_id') or 'unknown'}",
        f"- Readout status: {data.get('status')}",
        f"- Highest truthful claim: {boundary.get('highest_truthful_claim') or 'no_claim'}",
        "",
        "## Sections",
        "",
    ]
    for name, section in sections.items():
        section_map = _mapping(section)
        lines.append(f"- {name}: {section_map.get('status')}")
        for blocker in section_map.get("blockers") or []:
            lines.append(f"  - blocker: {blocker}")
    blockers = data.get("blockers") or []
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in blockers)
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- Purchasing or reviewing this package is not deployment approval.",
            "- Generated or simulator media in this package is not physical proof.",
            "- Simulator results are not real-world outcomes.",
            (
                "- The highest claim this package supports is: "
                f"{boundary.get('highest_truthful_claim') or 'no_claim'}."
            ),
        ]
    )
    return "\n".join(lines) + "\n"
