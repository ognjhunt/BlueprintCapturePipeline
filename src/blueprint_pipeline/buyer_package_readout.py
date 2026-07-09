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

from .common import parse_bool, utc_now_iso
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
_TASK_SUCCESS_REQUIRED_METRIC_KEYS = (
    "min_clearance_m",
    "clearance_threshold_m",
    "max_path_deviation_m",
)

BUYER_BLOCKER_CLASS_COPY: dict[str, dict[str, str]] = {
    "export_manifest_not_ready": {
        "headline": "Package is not ready for review.",
        "body": "The export manifest is missing, blocked, or not in the review-ready state.",
        "next_step": "Regenerate the package export and retry the buyer readout.",
    },
    "cards": {
        "headline": "Package cards are incomplete.",
        "body": "One or more site, task, scenario, or evaluation cards are missing.",
        "next_step": "Rebuild the robot-evaluation dataset cards before delivery.",
    },
    "rights_privacy_provenance": {
        "headline": "Rights, privacy, or provenance evidence is incomplete.",
        "body": "The package cannot be delivered until the rights packet and provenance boundary are present.",
        "next_step": "Resolve the rights/privacy review and regenerate the package.",
    },
    "robot_pov_evidence": {
        "headline": "Robot point-of-view evidence is incomplete.",
        "body": "The package is missing robot POV media, loadable training layout, or measured state provenance.",
        "next_step": "Materialize robot POV evidence and rerun export validation.",
    },
    "failure_evidence": {
        "headline": "Failure-case evidence is incomplete.",
        "body": "The package must include failure labels or an explicit reviewed-zero-failures attestation.",
        "next_step": "Add reviewed failure evidence before buyer delivery.",
    },
    "task_success_criteria": {
        "headline": "Task-success criteria are incomplete.",
        "body": "Required success metrics, coverage rows, or task/eval sources are missing.",
        "next_step": "Complete the task-success metrics and rerun the export.",
    },
    "calibration": {
        "headline": "Calibration evidence is incomplete.",
        "body": "The package cannot make calibration-bounded claims without a calibration report.",
        "next_step": "Attach the calibration report or keep calibration claims disabled.",
    },
    "media_provenance": {
        "headline": "Media provenance needs correction.",
        "body": "Generated or augmented media must stay separate from raw capture evidence.",
        "next_step": "Fix the media provenance labels and regenerate the package.",
    },
    "export_integrity": {
        "headline": "Export integrity is incomplete.",
        "body": "Checksums, file inventory, or round-trip validation are missing or blocked.",
        "next_step": "Regenerate the export and confirm the buyer-loadable files validate.",
    },
    "replay_review": {
        "headline": "Replay review instructions are missing.",
        "body": "A buyer cannot inspect the package without replay review instructions.",
        "next_step": "Add replay review instructions before delivery.",
    },
    "revocation_takedown": {
        "headline": "Consent revocation blocks package use.",
        "body": "The package must stay blocked until takedown and downstream-use actions are resolved.",
        "next_step": "Complete the rights/privacy takedown workflow before restoring access.",
    },
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strict_bool(value: Any) -> bool:
    return value is True


def _manifest_bool(value: Any) -> bool:
    return parse_bool(value, default=False)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        values: list[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, list | tuple | set):
        values = list(value)
    else:
        values = [value]
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _section(status: str, blockers: list[str], **fields: Any) -> dict[str, Any]:
    return {"status": status, "blockers": sorted(set(blockers)), **fields}


def _blocker_class(blocker: str) -> str:
    if blocker.startswith("export_manifest_not_ready:"):
        return "export_manifest_not_ready"
    prefix = blocker.split(":", 1)[0]
    return prefix if prefix in BUYER_BLOCKER_CLASS_COPY else "unknown"


def _customer_facing_status(
    *,
    status: str,
    blockers: list[str],
    claim_boundary: Mapping[str, Any],
) -> dict[str, Any]:
    blocker_messages: list[dict[str, str]] = []
    unmapped_blockers: list[str] = []
    for blocker in sorted(set(blockers)):
        blocker_class = _blocker_class(blocker)
        copy = BUYER_BLOCKER_CLASS_COPY.get(blocker_class)
        if copy is None:
            unmapped_blockers.append(blocker)
            copy = {
                "headline": "Package needs operator review.",
                "body": "An unmapped blocker is present; the package must remain blocked until an operator reviews it.",
                "next_step": "Ask Blueprint support to review the blocker before relying on this package.",
            }
        blocker_messages.append(
            {
                "blocker": blocker,
                "blocker_class": blocker_class,
                "headline": copy["headline"],
                "body": copy["body"],
                "next_step": copy["next_step"],
            }
        )

    blocked = bool(blockers)
    no_calibration_anchors = claim_boundary.get("no_real_world_calibration_anchors_present") is True
    return {
        "schema_version": "buyer_package_customer_status_copy.v1",
        "state": "blocked" if blocked else "review_required",
        "headline": (
            "Package is blocked and not ready for buyer review."
            if blocked
            else "Package is available for review with explicit claim limits."
        ),
        "body": (
            "Blueprint has found evidence gaps that must be resolved before this package can be delivered."
            if blocked
            else "The package can be reviewed, but it is not deployment approval or real-world performance proof."
        ),
        "primary_action": (
            "Resolve the listed blockers and regenerate the package readout."
            if blocked
            else "Review the evidence and claim boundary before using the package."
        ),
        "blocker_messages": blocker_messages,
        "unmapped_blockers": unmapped_blockers,
        "all_blocker_classes_have_customer_copy": not unmapped_blockers,
        "degraded_state_copy": [
            {
                "degraded_class": "no_real_world_calibration_anchors",
                "headline": "Simulator results are review-grade only.",
                "body": (
                    "No accepted real-world calibration anchors are included, so results must not be treated "
                    "as real-world performance predictions."
                ),
            }
        ]
        if no_calibration_anchors
        else [],
    }


def _presence_section(
    *, present: bool, missing_blocker: str, **fields: Any
) -> dict[str, Any]:
    blockers = [] if present else [missing_blocker]
    return _section("present" if present else "missing", blockers, **fields)


def _task_success_metrics(manifest: Mapping[str, Any]) -> dict[str, Any]:
    for key in (
        "task_success_metrics",
        "simulator_command_batch_metrics",
        "batch_metrics",
    ):
        metrics = _mapping(manifest.get(key))
        if metrics:
            return metrics
    return {}


def _lerobot_format_claimed(format_entry: Mapping[str, Any]) -> bool:
    entry = _mapping(format_entry)
    if not entry:
        return False
    status = str(entry.get("status") or "").strip().lower()
    return (
        _manifest_bool(entry.get("format_written"))
        or _manifest_bool(entry.get("consumer_layout_complete"))
        or bool(_mapping(entry.get("round_trip_validation")))
        or status.startswith("written")
    )


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
    revocation_takedown = _mapping(manifest.get("revocation_takedown"))
    downstream_takedown_execution_ledger = _mapping(
        manifest.get("downstream_takedown_execution_ledger")
    )
    revenue_share_review = _mapping(manifest.get("revenue_share_review"))
    data_processing_terms_review = _mapping(manifest.get("data_processing_terms_review"))
    optional_exports = _mapping(manifest.get("optional_exports"))
    optional_formats = _mapping(optional_exports.get("formats"))
    video_bundle = _mapping(optional_formats.get("video_bundle"))
    lerobot_v3 = _mapping(optional_formats.get("lerobot_v3"))
    gr00t_lerobot = _mapping(optional_formats.get("gr00t_lerobot"))
    handoff = _mapping(product_handoff) or _mapping(manifest.get("product_handoff"))
    ledger = _mapping(success_claim_ledger)
    manifest_blockers = _string_list(manifest.get("blockers"))
    consent_revoked = (
        _manifest_bool(consent_evidence.get("consent_revoked"))
        or _manifest_bool(revocation_takedown.get("consent_revoked"))
    )
    revocation_required = bool(
        consent_revoked
        or revocation_takedown.get("status") == "takedown_required"
        or manifest.get("status") == "blocked_consent_revoked_takedown_required"
        or "consent:consent_revoked_takedown_required" in manifest_blockers
    )
    webapp_takedown_executed = _strict_bool(
        revocation_takedown.get("webapp_takedown_executed")
    )
    hosted_session_takedown_executed = (
        _strict_bool(revocation_takedown.get("hosted_session_takedown_executed"))
    )
    downstream_takedown_artifacts = _mapping(
        revocation_takedown.get("downstream_takedown_artifacts")
        or manifest.get("downstream_takedown_artifacts")
    )

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
        consent_evidence_present=_manifest_bool(
            consent_evidence.get("consent_evidence_present")
        ),
        consent_evidence_status=consent_evidence.get("status"),
        revenue_share_review_included=bool(revenue_share_review),
        revenue_share_review_status=revenue_share_review.get("status"),
        owner_revenue_share_record_present=(
            _manifest_bool(revenue_share_review.get("owner_revenue_share_record_present"))
        ),
        operator_revenue_terms_present=bool(
            _mapping(revenue_share_review.get("operator_revenue_terms"))
        ),
        commercialization_terms_present=bool(
            _mapping(revenue_share_review.get("commercialization_terms"))
        ),
        exclusivity_terms_present=bool(
            _mapping(revenue_share_review.get("exclusivity_terms"))
        ),
        required_before_paid_reuse_or_resale=(
            _manifest_bool(
                revenue_share_review.get("required_before_paid_reuse_or_resale")
            )
        ),
        paid_reuse_or_resale_blocked=(
            not _manifest_bool(revenue_share_review.get("commercial_use_claim_allowed"))
            or not _manifest_bool(
                revenue_share_review.get("external_licensing_claim_allowed")
            )
        ),
        revenue_share_commitment_made=False,
        payout_commitment_allowed=False,
        commercial_use_claim_allowed=False,
        data_processing_terms_review_included=bool(data_processing_terms_review),
        data_processing_terms_review_status=data_processing_terms_review.get("status"),
        retention_policy_present=(
            _manifest_bool(data_processing_terms_review.get("retention_policy_present"))
        ),
        subprocessor_list_present=(
            _manifest_bool(data_processing_terms_review.get("subprocessor_list_present"))
        ),
        access_audit_terms_present=(
            _manifest_bool(data_processing_terms_review.get("access_audit_terms_present"))
        ),
        dpa_approval_claimed=False,
        external_delivery_claim_allowed=False,
        provenance_chain_present="proof_boundaries" in included,
    )
    sections["revocation_takedown"] = _section(
        "takedown_required" if revocation_required else "not_required",
        ["consent_revoked_takedown_required"] if revocation_required else [],
        revocation_takedown_manifest_included=(
            "revocation_takedown_manifest" in included
            or bool(str(revocation_takedown.get("path") or "").strip())
        ),
        revocation_takedown_manifest_path=revocation_takedown.get("path")
        or included.get("revocation_takedown_manifest"),
        consent_revoked=consent_revoked,
        consent_revoked_at=consent_evidence.get("consent_revoked_at")
        or revocation_takedown.get("consent_revoked_at"),
        local_package_access_revoked=_manifest_bool(
            revocation_takedown.get("local_package_access_revoked")
        )
        or revocation_required,
        delivery_blocked_by_consent_revocation=_manifest_bool(
            revocation_takedown.get("delivery_blocked")
        )
        or revocation_required,
        signed_access_revoked_by_consent=_manifest_bool(
            revocation_takedown.get("signed_access_revoked")
        )
        or revocation_required,
        downstream_takedown_required=_manifest_bool(
            revocation_takedown.get("downstream_takedown_required")
        )
        or revocation_required,
        webapp_takedown_executed=webapp_takedown_executed,
        hosted_session_takedown_executed=hosted_session_takedown_executed,
        webapp_or_hosted_takedown_execution_proven=bool(
            webapp_takedown_executed and hosted_session_takedown_executed
        ),
        downstream_takedown_artifacts=downstream_takedown_artifacts,
        downstream_takedown_execution_ledger_included=bool(
            downstream_takedown_execution_ledger
        ),
        downstream_takedown_execution_ledger_status=(
            downstream_takedown_execution_ledger.get("status")
        ),
        downstream_takedown_execution_ledger_path=(
            revocation_takedown.get("downstream_takedown_execution_ledger_path")
            or downstream_takedown_artifacts.get("downstream_takedown_execution_ledger")
        ),
        external_takedown_executor_present=(
            _manifest_bool(
                downstream_takedown_execution_ledger.get(
                    "external_takedown_executor_present"
                )
            )
        ),
        webapp_rights_privacy_takedown_notice_path=downstream_takedown_artifacts.get(
            "webapp_rights_privacy_takedown_notice"
        ),
        hosted_session_takedown_request_path=downstream_takedown_artifacts.get(
            "hosted_session_takedown_request"
        ),
        required_actions=_string_list(revocation_takedown.get("required_actions")),
        downstream_unexecuted_actions=_string_list(
            revocation_takedown.get("downstream_unexecuted_actions")
        ),
    )

    materialized_clip_count = _int_or_none(video_bundle.get("materialized_clip_count")) or 0
    missing_clip_file_count = _int_or_none(video_bundle.get("missing_clip_file_count")) or 0
    consumer_layout_complete = (
        _manifest_bool(lerobot_v3.get("consumer_layout_complete"))
        or _manifest_bool(gr00t_lerobot.get("consumer_layout_complete"))
    )
    pov_present = any(key in included for key in _POV_KEYS) or materialized_clip_count > 0
    pov_blockers: list[str] = []
    if not pov_present:
        pov_blockers.append("robot_pov_evidence_missing")
    if materialized_clip_count > 0 and missing_clip_file_count > 0:
        pov_blockers.append("declared_clip_files_missing")
    if materialized_clip_count > 0 and not consumer_layout_complete:
        pov_blockers.append("training_consumer_layout_incomplete")
    measured_state_fractions: dict[str, float] = {}
    measured_state_fraction_floors: list[float] = []
    for format_name, format_entry in (
        ("lerobot_v3", lerobot_v3),
        ("gr00t_lerobot", gr00t_lerobot),
    ):
        if not _lerobot_format_claimed(format_entry):
            continue
        state_action_provenance = _mapping(format_entry.get("state_action_provenance"))
        real_state_fraction = _float_or_none(
            state_action_provenance.get("real_state_fraction")
        )
        measured_state_fraction_floor = _float_or_none(
            state_action_provenance.get("measured_state_fraction_floor")
        )
        if measured_state_fraction_floor is not None:
            measured_state_fraction_floors.append(measured_state_fraction_floor)
        if real_state_fraction is None:
            pov_blockers.append(f"measured_state_fraction_unknown:{format_name}")
            continue
        measured_state_fractions[format_name] = real_state_fraction
        floor_passed = _strict_bool(
            state_action_provenance.get("measured_state_fraction_floor_passed")
        )
        if measured_state_fraction_floor is not None:
            floor_passed = floor_passed and real_state_fraction >= measured_state_fraction_floor
        if not floor_passed:
            pov_blockers.append(
                f"insufficient_measured_state_fraction:{format_name}"
            )
    sections["robot_pov_evidence"] = _section(
        "present" if not pov_blockers else "missing",
        pov_blockers,
        pov_artifact_keys=[key for key in _POV_KEYS if key in included],
        clips_manifest_included=_manifest_bool(
            export_policy.get("clips_manifest_included")
        ),
        materialized_clip_count=materialized_clip_count,
        missing_clip_file_count=missing_clip_file_count,
        lerobot_v3_consumer_layout_complete=_manifest_bool(
            lerobot_v3.get("consumer_layout_complete")
        ),
        gr00t_lerobot_consumer_layout_complete=_manifest_bool(
            gr00t_lerobot.get("consumer_layout_complete")
        ),
        measured_state_fractions=measured_state_fractions,
        measured_state_fraction_floor=(
            max(measured_state_fraction_floors)
            if measured_state_fraction_floors
            else None
        ),
    )

    failure_present = "failure_labels" in included
    failure_label_count = _int_or_none(manifest_counts.get("failure_label_count"))
    failure_review = _mapping(manifest.get("failure_evidence_review"))
    zero_failures_reviewed = _strict_bool(
        manifest_counts.get("zero_failures_reviewed")
    ) or _strict_bool(failure_review.get("zero_failures_reviewed"))
    failure_blockers: list[str] = []
    if not failure_present:
        failure_blockers.append("failure_labels_missing")
    if failure_present and failure_label_count is None:
        failure_blockers.append("failure_label_count_unknown")
    if failure_present and failure_label_count == 0 and not zero_failures_reviewed:
        failure_blockers.append("failure_labels_empty_without_zero_failures_reviewed")
    sections["failure_evidence"] = _section(
        "present" if not failure_blockers else "missing",
        failure_blockers,
        failure_label_count=_int_or_none(manifest_counts.get("failure_label_count")),
        zero_failures_reviewed=zero_failures_reviewed,
        failure_cases_preserved=failure_present,
    )

    criteria_present = "task_cards" in included and "eval_cards" in included
    task_metrics = _task_success_metrics(manifest)
    metric_keys = set(_string_list(task_metrics.get("required_metric_keys")))
    missing_metric_keys = sorted(
        key for key in _TASK_SUCCESS_REQUIRED_METRIC_KEYS if key not in metric_keys
    )
    attempt_metric_row_count = _int_or_none(task_metrics.get("attempt_metric_row_count")) or 0
    missing_metric_row_count = _int_or_none(task_metrics.get("missing_metric_row_count")) or 0
    task_success_blockers: list[str] = []
    if not criteria_present:
        task_success_blockers.append("task_success_criteria_source_missing")
    if (
        "simulator_command_batch_metrics" not in included
        and not str(task_metrics.get("source_artifact") or "").strip()
    ):
        task_success_blockers.append("task_success_metrics_artifact_missing")
    if not task_metrics:
        task_success_blockers.append("task_success_metrics_missing")
    if task_metrics and not _strict_bool(task_metrics.get("metric_coverage_complete")):
        task_success_blockers.append("task_success_metric_coverage_incomplete")
    if task_metrics and attempt_metric_row_count <= 0:
        task_success_blockers.append("task_success_metric_rows_missing")
    if task_metrics and missing_metric_row_count > 0:
        task_success_blockers.append("task_success_metric_rows_incomplete")
    task_success_blockers.extend(
        f"task_success_required_metric_missing:{key}" for key in missing_metric_keys
    )
    sections["task_success_criteria"] = _section(
        "present" if not task_success_blockers else "missing",
        task_success_blockers,
        criteria_source="task_cards_eval_cards_and_batch_metrics"
        if criteria_present and task_metrics
        else None,
        task_cards_included="task_cards" in included,
        eval_cards_included="eval_cards" in included,
        metrics_artifact_included="simulator_command_batch_metrics" in included,
        metric_coverage_complete=_strict_bool(
            task_metrics.get("metric_coverage_complete")
        ),
        attempt_metric_row_count=attempt_metric_row_count,
        missing_metric_row_count=missing_metric_row_count,
        required_metric_keys=sorted(metric_keys),
        required_metric_keys_present=not missing_metric_keys,
    )

    calibration_present = "calibration_report" in included
    calibration_report = _mapping(
        manifest.get("calibration_report")
        or manifest.get("sim_vs_real_calibration_report")
    )
    accepted_calibration_anchor_count = (
        _int_or_none(calibration_report.get("accepted_anchor_count")) or 0
    )
    sim_vs_real_calibration_score = _float_or_none(
        calibration_report.get("sim_vs_real_calibration_score")
    )
    sim_vs_real_calibration_claim_allowed = bool(
        calibration_report.get("status") == "completed"
        and accepted_calibration_anchor_count > 0
        and sim_vs_real_calibration_score is not None
    )
    sections["calibration"] = _presence_section(
        present=calibration_present,
        missing_blocker="calibration_report_missing",
        calibration_report_included=calibration_present,
        camera_calibration_metadata_included="camera_calibration_metadata" in included,
        calibration_report_status=calibration_report.get("status"),
        accepted_real_world_calibration_anchor_count=accepted_calibration_anchor_count,
        sim_vs_real_calibration_score=sim_vs_real_calibration_score,
        sim_vs_real_calibration_claim_allowed=sim_vs_real_calibration_claim_allowed,
        no_real_world_calibration_anchors_present=(
            accepted_calibration_anchor_count == 0
        ),
        results_are_not_real_world_performance_predictions=(
            not sim_vs_real_calibration_claim_allowed
        ),
        buyer_disclosure=(
            "No accepted real-world calibration anchors are included; simulator or "
            "generated results must not be presented as real-world performance "
            "predictions."
            if accepted_calibration_anchor_count == 0
            else (
                "Calibration anchors are present; use only the report's stated "
                "calibration score and claim boundary."
            )
        ),
    )

    media_blockers: list[str] = []
    generated_media_included = _manifest_bool(
        export_policy.get("visual_augmentation_packet_included")
    )
    generated_media_claimed_raw = parse_bool(
        export_policy.get("visual_augmentation_generated_videos_are_raw_capture_evidence"),
        default=True,
    )
    if generated_media_included and generated_media_claimed_raw:
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
    # A lerobot-format export the buyer cannot load back (LeRobotDataset round
    # trip) must never read "ready": require a passed round-trip verdict for
    # every lerobot-format export the package claims to include.
    lerobot_round_trip: dict[str, Any] = {}
    for format_name, format_entry in (
        ("lerobot_v3", lerobot_v3),
        ("gr00t_lerobot", gr00t_lerobot),
    ):
        if not _lerobot_format_claimed(format_entry):
            continue
        validation = _mapping(format_entry.get("round_trip_validation"))
        validation_status = str(validation.get("status") or "").strip() or None
        lerobot_round_trip[format_name] = validation_status
        if not validation:
            integrity_blockers.append(
                f"lerobot_round_trip_validation_missing:{format_name}"
            )
        elif validation_status != "passed":
            integrity_blockers.append(f"lerobot_export_not_loadable:{format_name}")
    sections["export_integrity"] = _section(
        "present" if not integrity_blockers else "missing",
        integrity_blockers,
        checksums_path=manifest.get("checksums_path"),
        package_file_count=len(package_files),
        schema_version=manifest.get("schema_version"),
        lerobot_round_trip_validation=lerobot_round_trip,
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
    scaniverse_support_included = _manifest_bool(
        export_policy.get("scaniverse_support_assets_included")
    )
    sections["derived_support_assets"] = _section(
        "present_support_only" if scaniverse_support_included else "not_included",
        [],
        scaniverse_support_assets_included=scaniverse_support_included,
        scaniverse_support_asset_manifest_path=manifest.get(
            "scaniverse_support_asset_manifest_path"
        ),
        scaniverse_assets_are_external_derived_support=_manifest_bool(
            export_policy.get("scaniverse_assets_are_external_derived_support")
        ),
        scaniverse_assets_are_raw_capture_evidence=False,
        scaniverse_assets_are_task_success_evidence=False,
        scaniverse_assets_are_physics_contact_evidence=False,
        owner_system_simulator_evidence_required_for_claim_upgrade=True,
    )

    blockers: list[str] = []
    export_status = str(manifest.get("status") or "").strip()
    if export_status != "export_ready_review_required":
        blockers.append(f"export_manifest_not_ready:{export_status or 'missing'}")
    for name in BUYER_CRITICAL_SECTIONS:
        blockers.extend(f"{name}:{blocker}" for blocker in sections[name]["blockers"])
    if revocation_required:
        blockers.append("revocation_takedown:consent_revoked_takedown_required")

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
        "accepted_real_world_calibration_anchor_count": (
            accepted_calibration_anchor_count
        ),
        "sim_vs_real_calibration_score": sim_vs_real_calibration_score,
        "sim_vs_real_calibration_claim_allowed": sim_vs_real_calibration_claim_allowed,
        "no_real_world_calibration_anchors_present": (
            accepted_calibration_anchor_count == 0
        ),
        "results_are_not_real_world_performance_predictions": (
            not sim_vs_real_calibration_claim_allowed
        ),
        "scaniverse_assets_are_raw_capture_evidence": False,
        "scaniverse_assets_are_task_success_evidence": False,
        "scaniverse_assets_are_physics_contact_evidence": False,
        "readout_summarizes_existing_evidence_only": True,
        "consent_revocation_blocks_downstream_use": revocation_required,
        "local_package_access_revoked": _manifest_bool(
            revocation_takedown.get("local_package_access_revoked")
        )
        or revocation_required,
        "delivery_blocked_by_consent_revocation": _manifest_bool(
            revocation_takedown.get("delivery_blocked")
        )
        or revocation_required,
        "signed_access_revoked_by_consent": _manifest_bool(
            revocation_takedown.get("signed_access_revoked")
        )
        or revocation_required,
        "webapp_or_hosted_takedown_execution_proven": bool(
            webapp_takedown_executed and hosted_session_takedown_executed
        ),
        "downstream_takedown_execution_ledger_present": bool(
            downstream_takedown_execution_ledger
        ),
        "readout_is_not_takedown_execution_proof": True,
    }

    status = "buyer_readout_ready_review_required" if not blockers else (
        "blocked_incomplete_package"
    )
    customer_facing_status = _customer_facing_status(
        status=status,
        blockers=blockers,
        claim_boundary=readout_claim_boundary,
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
        "customer_facing_status": customer_facing_status,
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
    customer_status = _mapping(data.get("customer_facing_status"))
    if customer_status:
        lines.extend(
            [
                "",
                "## Customer Status",
                "",
                f"- State: {customer_status.get('state')}",
                f"- {customer_status.get('headline')}",
                f"- {customer_status.get('body')}",
            ]
        )
        for message in customer_status.get("blocker_messages") or []:
            message_map = _mapping(message)
            lines.append(
                "- "
                f"{message_map.get('blocker_class')}: {message_map.get('headline')} "
                f"{message_map.get('next_step')}"
            )
        for degraded in customer_status.get("degraded_state_copy") or []:
            degraded_map = _mapping(degraded)
            lines.append(
                "- "
                f"{degraded_map.get('degraded_class')}: {degraded_map.get('headline')}"
            )
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
                "- No accepted sim-vs-real calibration anchors are included; "
                "results are not real-world performance predictions."
                if boundary.get("no_real_world_calibration_anchors_present") is True
                else (
                    "- Sim-vs-real calibration is bounded by the included calibration "
                    "report and does not approve deployment."
                )
            ),
            (
                "- The highest claim this package supports is: "
                f"{boundary.get('highest_truthful_claim') or 'no_claim'}."
            ),
        ]
    )
    if boundary.get("consent_revocation_blocks_downstream_use") is True:
        lines.extend(
            [
                "- Consent revocation blocks package delivery and training use.",
                (
                    "- This readout is not proof that WebApp or hosted-session "
                    "takedown has executed."
                ),
            ]
        )
    return "\n".join(lines) + "\n"
