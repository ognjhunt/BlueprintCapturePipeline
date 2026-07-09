from blueprint_pipeline.buyer_package_readout import (
    BUYER_CRITICAL_SECTIONS,
    BUYER_BLOCKER_CLASS_COPY,
    build_buyer_package_readout,
    render_buyer_package_readout_markdown,
)


def _complete_export_manifest() -> dict:
    return {
        "schema_version": "post_training_data_package_export.v1",
        "scene_id": "site-a",
        "capture_id": "capture-1",
        "package_type": "post_training_data_package",
        "status": "export_ready_review_required",
        "included_artifacts": {
            "site_card": "robot_eval_dataset/site_card.json",
            "task_cards": "robot_eval_dataset/task_cards.json",
            "scenario_cards": "robot_eval_dataset/scenario_cards.json",
            "eval_cards": "robot_eval_dataset/eval_cards.json",
            "rights_packet": "robot_eval_dataset/rights_packet.json",
            "proof_boundaries": "robot_eval_dataset/proof_boundaries.json",
            "failure_labels": "failure_labels.json",
            "calibration_report": "calibration_report.json",
            "robot_pov_observation_manifest": "robot_pov_observation_manifest.json",
            "replay_review_instructions": "replay_review_instructions.md",
            "simulator_command_batch_metrics": "simulator_command_batch_metrics.json",
        },
        "manifest_counts": {"failure_label_count": 3},
        "task_success_metrics": {
            "source_artifact": "simulator_command_batch_metrics.json",
            "metric_coverage_complete": True,
            "required_metric_keys": [
                "min_clearance_m",
                "clearance_threshold_m",
                "max_path_deviation_m",
            ],
            "attempt_metric_row_count": 1,
            "missing_metric_row_count": 0,
        },
        "export_policy": {
            "clips_manifest_included": True,
            "visual_augmentation_packet_included": True,
            "visual_augmentation_generated_videos_are_raw_capture_evidence": False,
        },
        "claim_boundary": {"rights_privacy_scope_proven": False},
        "checksums_path": "checksums.json",
        "replay_review_instructions_path": "replay_review_instructions.md",
        "package_files": {"dataset_card": {"path": "dataset_card.json"}},
    }


def test_empty_export_manifest_fails_closed() -> None:
    readout = build_buyer_package_readout(export_manifest={})

    assert readout["status"] == "blocked_incomplete_package"
    blockers = readout["blockers"]
    assert "export_manifest_not_ready:missing" in blockers
    assert "cards:card_missing:site_card" in blockers
    assert "rights_privacy_provenance:rights_packet_missing" in blockers
    assert "robot_pov_evidence:robot_pov_evidence_missing" in blockers
    assert "failure_evidence:failure_labels_missing" in blockers
    assert "task_success_criteria:task_success_criteria_source_missing" in blockers
    assert "calibration:calibration_report_missing" in blockers
    assert "export_integrity:checksums_manifest_missing" in blockers
    assert "replay_review:replay_review_instructions_missing" in blockers
    assert readout["claim_boundary"]["highest_truthful_claim"] == "no_claim"
    assert readout["claim_boundary"]["physical_deployment_ready"] is False
    customer_status = readout["customer_facing_status"]
    assert customer_status["state"] == "blocked"
    assert customer_status["all_blocker_classes_have_customer_copy"] is True
    assert customer_status["unmapped_blockers"] == []
    assert len(customer_status["blocker_messages"]) == len(blockers)


def test_complete_manifest_produces_buyer_readable_summary_without_overclaim() -> None:
    readout = build_buyer_package_readout(export_manifest=_complete_export_manifest())

    assert readout["status"] == "buyer_readout_ready_review_required"
    assert readout["blockers"] == []
    for name in BUYER_CRITICAL_SECTIONS:
        assert readout["sections"][name]["status"] == "present", name

    # No ledger provided: the readout must not upgrade any claim.
    boundary = readout["claim_boundary"]
    assert boundary["highest_truthful_claim"] == "no_claim"
    assert boundary["success_claim_ledger_present"] is False
    assert boundary["physical_deployment_ready"] is False
    assert boundary["package_purchase_is_not_deployment_approval"] is True
    assert boundary["accepted_real_world_calibration_anchor_count"] == 0
    assert boundary["sim_vs_real_calibration_claim_allowed"] is False
    assert boundary["no_real_world_calibration_anchors_present"] is True
    assert boundary["results_are_not_real_world_performance_predictions"] is True
    calibration = readout["sections"]["calibration"]
    assert calibration["no_real_world_calibration_anchors_present"] is True
    assert calibration["results_are_not_real_world_performance_predictions"] is True
    rights_section = readout["sections"]["rights_privacy_provenance"]
    assert rights_section["operator_revenue_terms_present"] is False
    assert rights_section["commercialization_terms_present"] is False
    assert rights_section["exclusivity_terms_present"] is False
    assert rights_section["paid_reuse_or_resale_blocked"] is True
    assert rights_section["data_processing_terms_review_included"] is False
    assert rights_section["retention_policy_present"] is False
    assert rights_section["subprocessor_list_present"] is False
    assert rights_section["access_audit_terms_present"] is False
    assert rights_section["dpa_approval_claimed"] is False
    customer_status = readout["customer_facing_status"]
    assert customer_status["state"] == "review_required"
    assert customer_status["all_blocker_classes_have_customer_copy"] is True
    assert customer_status["blocker_messages"] == []
    assert customer_status["degraded_state_copy"][0]["degraded_class"] == (
        "no_real_world_calibration_anchors"
    )

    markdown = render_buyer_package_readout_markdown(readout)
    assert "Highest truthful claim: no_claim" in markdown
    assert "Package is available for review with explicit claim limits." in markdown
    assert "not deployment approval" in markdown
    assert "Simulator results are not real-world outcomes." in markdown
    assert "not real-world performance predictions" in markdown


def test_readout_surfaces_calibration_anchor_claim_boundary() -> None:
    manifest = _complete_export_manifest()
    manifest["calibration_report"] = {
        "schema_version": "sim_vs_real_calibration_report.v1",
        "status": "completed",
        "accepted_anchor_count": 4,
        "sim_vs_real_calibration_score": 0.75,
    }

    readout = build_buyer_package_readout(export_manifest=manifest)

    calibration = readout["sections"]["calibration"]
    boundary = readout["claim_boundary"]
    assert calibration["accepted_real_world_calibration_anchor_count"] == 4
    assert calibration["sim_vs_real_calibration_score"] == 0.75
    assert calibration["sim_vs_real_calibration_claim_allowed"] is True
    assert calibration["no_real_world_calibration_anchors_present"] is False
    assert boundary["accepted_real_world_calibration_anchor_count"] == 4
    assert boundary["sim_vs_real_calibration_score"] == 0.75
    assert boundary["sim_vs_real_calibration_claim_allowed"] is True
    assert boundary["results_are_not_real_world_performance_predictions"] is False

    markdown = render_buyer_package_readout_markdown(readout)
    assert "bounded by the included calibration report" in markdown


def test_failure_evidence_requires_label_count_or_reviewed_zero_attestation() -> None:
    manifest = _complete_export_manifest()
    manifest["manifest_counts"]["failure_label_count"] = 0

    readout = build_buyer_package_readout(export_manifest=manifest)

    assert readout["status"] == "blocked_incomplete_package"
    assert (
        "failure_evidence:failure_labels_empty_without_zero_failures_reviewed"
        in readout["blockers"]
    )

    manifest["manifest_counts"]["zero_failures_reviewed"] = True
    passed = build_buyer_package_readout(export_manifest=manifest)
    assert passed["status"] == "buyer_readout_ready_review_required"
    assert passed["sections"]["failure_evidence"]["zero_failures_reviewed"] is True


def test_task_success_criteria_requires_batch_metric_coverage() -> None:
    manifest = _complete_export_manifest()
    manifest["included_artifacts"].pop("simulator_command_batch_metrics")
    manifest.pop("task_success_metrics")

    readout = build_buyer_package_readout(export_manifest=manifest)

    assert readout["status"] == "blocked_incomplete_package"
    assert (
        "task_success_criteria:task_success_metrics_artifact_missing"
        in readout["blockers"]
    )
    assert "task_success_criteria:task_success_metrics_missing" in readout["blockers"]

    manifest = _complete_export_manifest()
    manifest["task_success_metrics"]["required_metric_keys"] = ["min_clearance_m"]
    metric_gap = build_buyer_package_readout(export_manifest=manifest)
    assert metric_gap["status"] == "blocked_incomplete_package"
    assert (
        "task_success_criteria:task_success_required_metric_missing:max_path_deviation_m"
        in metric_gap["blockers"]
    )


def test_ledger_claim_is_echoed_but_never_invented() -> None:
    manifest = _complete_export_manifest()
    readout = build_buyer_package_readout(
        export_manifest=manifest,
        success_claim_ledger={"highest_truthful_claim": "simulator_task_success"},
    )
    assert readout["claim_boundary"]["highest_truthful_claim"] == "simulator_task_success"
    assert readout["claim_boundary"]["physical_deployment_ready"] is False

    bogus = build_buyer_package_readout(
        export_manifest=manifest,
        success_claim_ledger={"highest_truthful_claim": "definitely_deployment_ready"},
    )
    assert bogus["claim_boundary"]["highest_truthful_claim"] == "no_claim"


def test_generated_media_must_be_segregated() -> None:
    manifest = _complete_export_manifest()
    del manifest["export_policy"][
        "visual_augmentation_generated_videos_are_raw_capture_evidence"
    ]

    readout = build_buyer_package_readout(export_manifest=manifest)

    assert readout["status"] == "blocked_incomplete_package"
    assert (
        "media_provenance:generated_media_not_segregated_from_raw_capture"
        in readout["blockers"]
    )


def test_blocked_export_status_blocks_readout() -> None:
    manifest = _complete_export_manifest()
    manifest["status"] = "blocked_missing_inputs"

    readout = build_buyer_package_readout(export_manifest=manifest)

    assert readout["status"] == "blocked_incomplete_package"
    assert "export_manifest_not_ready:blocked_missing_inputs" in readout["blockers"]
    assert readout["customer_facing_status"]["blocker_messages"][0]["blocker_class"] == (
        "export_manifest_not_ready"
    )


def test_customer_facing_status_copy_covers_every_blocker_class() -> None:
    manifest = _complete_export_manifest()
    manifest["status"] = "blocked_missing_inputs"
    included = manifest["included_artifacts"]
    for key in (
        "site_card",
        "rights_packet",
        "proof_boundaries",
        "robot_pov_observation_manifest",
        "failure_labels",
        "simulator_command_batch_metrics",
        "calibration_report",
        "replay_review_instructions",
    ):
        included.pop(key, None)
    manifest["manifest_counts"]["failure_label_count"] = 0
    manifest.pop("task_success_metrics")
    manifest["export_policy"]["visual_augmentation_generated_videos_are_raw_capture_evidence"] = True
    manifest.pop("checksums_path")
    manifest.pop("replay_review_instructions_path")
    manifest["package_files"] = {}
    manifest["consent_evidence"] = {"consent_revoked": True}

    readout = build_buyer_package_readout(export_manifest=manifest)
    customer_status = readout["customer_facing_status"]
    message_classes = {
        message["blocker_class"]
        for message in customer_status["blocker_messages"]
    }

    assert readout["status"] == "blocked_incomplete_package"
    assert customer_status["state"] == "blocked"
    assert customer_status["all_blocker_classes_have_customer_copy"] is True
    assert customer_status["unmapped_blockers"] == []
    assert set(BUYER_BLOCKER_CLASS_COPY).issubset(message_classes)
    assert "unknown" not in message_classes
    assert all(message["headline"] for message in customer_status["blocker_messages"])
    assert all(message["next_step"] for message in customer_status["blocker_messages"])

    markdown = render_buyer_package_readout_markdown(readout)
    assert "## Customer Status" in markdown
    assert "rights_privacy_provenance: Rights, privacy, or provenance evidence is incomplete." in markdown


def test_revoked_consent_surfaces_takedown_blocker() -> None:
    manifest = _complete_export_manifest()
    manifest["status"] = "blocked_consent_revoked_takedown_required"
    manifest["blockers"] = ["consent:consent_revoked_takedown_required"]
    manifest["included_artifacts"]["revocation_takedown_manifest"] = (
        "revocation_takedown_manifest.json"
    )
    manifest["consent_evidence"] = {
        "consent_revoked": True,
        "consent_revoked_at": "2026-07-04T12:00:00Z",
    }
    manifest["revocation_takedown"] = {
        "status": "takedown_required",
        "path": "revocation_takedown_manifest.json",
        "consent_revoked": True,
        "local_package_access_revoked": True,
        "delivery_blocked": True,
        "signed_access_revoked": True,
        "downstream_takedown_required": True,
        "webapp_takedown_executed": False,
        "hosted_session_takedown_executed": False,
        "downstream_takedown_artifacts": {
            "webapp_rights_privacy_takedown_notice": (
                "webapp_rights_privacy_takedown_notice.json"
            ),
            "hosted_session_takedown_request": "hosted_session_takedown_request.json",
            "downstream_takedown_execution_ledger": (
                "downstream_takedown_execution_ledger.json"
            ),
        },
        "downstream_takedown_execution_ledger_path": (
            "downstream_takedown_execution_ledger.json"
        ),
        "required_actions": ["remove_or_expire_hosted_sessions"],
        "downstream_unexecuted_actions": ["notify_webapp_rights_privacy_blocking"],
    }
    manifest["downstream_takedown_execution_ledger"] = {
        "schema_version": "post_training_downstream_takedown_execution_ledger.v1",
        "status": "queued_unexecuted_downstream_takedown",
        "external_takedown_executor_present": False,
        "webapp_or_hosted_takedown_execution_proven": False,
    }

    readout = build_buyer_package_readout(export_manifest=manifest)

    assert readout["status"] == "blocked_incomplete_package"
    assert "revocation_takedown:consent_revoked_takedown_required" in readout["blockers"]
    takedown = readout["sections"]["revocation_takedown"]
    assert takedown["status"] == "takedown_required"
    assert takedown["local_package_access_revoked"] is True
    assert takedown["delivery_blocked_by_consent_revocation"] is True
    assert takedown["signed_access_revoked_by_consent"] is True
    assert takedown["webapp_takedown_executed"] is False
    assert takedown["hosted_session_takedown_executed"] is False
    assert takedown["webapp_rights_privacy_takedown_notice_path"] == (
        "webapp_rights_privacy_takedown_notice.json"
    )
    assert takedown["hosted_session_takedown_request_path"] == (
        "hosted_session_takedown_request.json"
    )
    assert takedown["downstream_takedown_execution_ledger_included"] is True
    assert takedown["downstream_takedown_execution_ledger_status"] == (
        "queued_unexecuted_downstream_takedown"
    )
    assert takedown["downstream_takedown_execution_ledger_path"] == (
        "downstream_takedown_execution_ledger.json"
    )
    assert takedown["external_takedown_executor_present"] is False
    assert readout["claim_boundary"]["consent_revocation_blocks_downstream_use"] is True
    assert (
        readout["claim_boundary"]["downstream_takedown_execution_ledger_present"]
        is True
    )
    assert readout["claim_boundary"]["webapp_or_hosted_takedown_execution_proven"] is False

    markdown = render_buyer_package_readout_markdown(readout)
    assert "Consent revocation blocks package delivery and training use." in markdown
    assert "not proof that WebApp or hosted-session takedown has executed" in markdown


def test_string_true_consent_revocation_blocks_readout() -> None:
    manifest = _complete_export_manifest()
    manifest["consent_evidence"] = {"consent_revoked": "true"}

    readout = build_buyer_package_readout(export_manifest=manifest)

    assert readout["status"] == "blocked_incomplete_package"
    assert "revocation_takedown:consent_revoked_takedown_required" in readout["blockers"]
    assert readout["sections"]["revocation_takedown"]["status"] == "takedown_required"
    assert readout["sections"]["revocation_takedown"]["consent_revoked"] is True
    assert readout["claim_boundary"]["consent_revocation_blocks_downstream_use"] is True


def test_string_true_takedown_execution_does_not_prove_execution() -> None:
    manifest = _complete_export_manifest()
    manifest["consent_evidence"] = {"consent_revoked": "true"}
    manifest["revocation_takedown"] = {
        "webapp_takedown_executed": "true",
        "hosted_session_takedown_executed": "true",
    }

    readout = build_buyer_package_readout(export_manifest=manifest)

    takedown = readout["sections"]["revocation_takedown"]
    assert takedown["consent_revoked"] is True
    assert takedown["webapp_takedown_executed"] is False
    assert takedown["hosted_session_takedown_executed"] is False
    assert takedown["webapp_or_hosted_takedown_execution_proven"] is False
    assert readout["claim_boundary"]["webapp_or_hosted_takedown_execution_proven"] is False


def test_string_false_manifest_booleans_do_not_overclaim() -> None:
    manifest = _complete_export_manifest()
    manifest["claim_boundary"]["rights_privacy_scope_proven"] = "false"
    manifest["consent_evidence"] = {
        "consent_evidence_present": "false",
        "consent_revoked": "false",
    }
    manifest["revocation_takedown"] = {
        "local_package_access_revoked": "false",
        "delivery_blocked": "false",
        "signed_access_revoked": "false",
        "downstream_takedown_required": "false",
        "webapp_takedown_executed": "false",
        "hosted_session_takedown_executed": "false",
    }
    manifest["revenue_share_review"] = {
        "owner_revenue_share_record_present": "false",
        "required_before_paid_reuse_or_resale": "false",
        "commercial_use_claim_allowed": "false",
        "external_licensing_claim_allowed": "false",
    }
    manifest["data_processing_terms_review"] = {
        "retention_policy_present": "false",
        "subprocessor_list_present": "false",
        "access_audit_terms_present": "false",
    }
    manifest["export_policy"] = {
        **manifest["export_policy"],
        "clips_manifest_included": "false",
        "visual_augmentation_packet_included": "false",
        "visual_augmentation_generated_videos_are_raw_capture_evidence": "false",
    }
    manifest["optional_exports"] = {
        "formats": {
            "lerobot_v3": {
                "format_written": "false",
                "consumer_layout_complete": "false",
                "status": "available_not_written",
            },
            "video_bundle": {
                "materialized_clip_count": 0,
                "missing_clip_file_count": 0,
            },
        }
    }

    readout = build_buyer_package_readout(export_manifest=manifest)

    assert readout["status"] == "buyer_readout_ready_review_required"
    rights = readout["sections"]["rights_privacy_provenance"]
    assert rights["rights_privacy_scope_proven"] is False
    assert rights["consent_evidence_present"] is False
    assert rights["owner_revenue_share_record_present"] is False
    assert rights["required_before_paid_reuse_or_resale"] is False
    assert rights["retention_policy_present"] is False
    takedown = readout["sections"]["revocation_takedown"]
    assert takedown["local_package_access_revoked"] is False
    assert takedown["delivery_blocked_by_consent_revocation"] is False
    assert takedown["signed_access_revoked_by_consent"] is False
    assert takedown["webapp_takedown_executed"] is False
    assert readout["sections"]["robot_pov_evidence"]["clips_manifest_included"] is False
    assert readout["sections"]["media_provenance"]["generated_media_included"] is False
    assert readout["claim_boundary"]["local_package_access_revoked"] is False
    assert readout["sections"]["export_integrity"][
        "lerobot_round_trip_validation"
    ] == {}


def test_lerobot_round_trip_validation_gates_export_integrity() -> None:
    manifest = _complete_export_manifest()
    manifest["optional_exports"] = {
        "formats": {
            "lerobot_v3": {
                "format_written": True,
                "status": "written_native",
                "consumer_layout_complete": True,
            }
        }
    }

    # A lerobot export with no round-trip validation verdict must fail closed.
    missing = build_buyer_package_readout(export_manifest=manifest)
    assert missing["status"] == "blocked_incomplete_package"
    assert (
        "export_integrity:lerobot_round_trip_validation_missing:lerobot_v3"
        in missing["blockers"]
    )

    # A blocked validation verdict means the buyer cannot load the dataset.
    manifest["optional_exports"]["formats"]["lerobot_v3"]["round_trip_validation"] = {
        "status": "blocked",
        "blockers": ["timestamps_not_monotonic:episode_0"],
    }
    blocked = build_buyer_package_readout(export_manifest=manifest)
    assert blocked["status"] == "blocked_incomplete_package"
    assert (
        "export_integrity:lerobot_export_not_loadable:lerobot_v3"
        in blocked["blockers"]
    )
    section = blocked["sections"]["export_integrity"]
    assert section["lerobot_round_trip_validation"]["lerobot_v3"] == "blocked"

    # A passed verdict restores the readout.
    manifest["optional_exports"]["formats"]["lerobot_v3"]["round_trip_validation"] = {
        "status": "passed",
        "blockers": [],
    }
    manifest["optional_exports"]["formats"]["lerobot_v3"]["state_action_provenance"] = {
        "real_state_fraction": 1.0,
        "real_action_fraction": 1.0,
        "measured_state_fraction_floor": 0.5,
        "measured_state_fraction_floor_passed": True,
    }
    passed = build_buyer_package_readout(export_manifest=manifest)
    assert passed["status"] == "buyer_readout_ready_review_required"
    assert passed["sections"]["export_integrity"]["lerobot_round_trip_validation"] == {
        "lerobot_v3": "passed"
    }


def test_unwritten_optional_lerobot_formats_do_not_require_round_trip_validation() -> None:
    manifest = _complete_export_manifest()
    manifest["optional_exports"] = {
        "formats": {
            "lerobot_v3": {
                "format_written": False,
                "status": "available_not_written",
            },
            "gr00t_lerobot": {
                "format_written": False,
                "status": "blocked_optional_dependency_missing",
            },
        }
    }

    readout = build_buyer_package_readout(export_manifest=manifest)

    assert readout["status"] == "buyer_readout_ready_review_required"
    assert readout["sections"]["export_integrity"][
        "lerobot_round_trip_validation"
    ] == {}
    assert not any(
        blocker.startswith("export_integrity:lerobot_round_trip_validation_missing")
        for blocker in readout["blockers"]
    )


def test_gr00t_round_trip_validation_also_gated() -> None:
    manifest = _complete_export_manifest()
    manifest["optional_exports"] = {
        "formats": {
            "gr00t_lerobot": {
                "format_written": True,
                "round_trip_validation": {
                    "status": "blocked",
                    "blockers": ["episode_length_mismatch:episode_0"],
                },
            }
        }
    }

    readout = build_buyer_package_readout(export_manifest=manifest)

    assert readout["status"] == "blocked_incomplete_package"
    assert (
        "export_integrity:lerobot_export_not_loadable:gr00t_lerobot"
        in readout["blockers"]
    )


def _claimed_lerobot_v3_entry(**provenance_overrides: object) -> dict:
    entry: dict = {
        "format_written": True,
        "status": "written_native",
        "consumer_layout_complete": True,
        "round_trip_validation": {"status": "passed", "blockers": []},
        "state_action_provenance": {
            "real_state_fraction": 1.0,
            "real_action_fraction": 1.0,
            "measured_state_fraction_floor": 0.5,
            "measured_state_fraction_floor_passed": True,
        },
    }
    entry["state_action_provenance"].update(provenance_overrides)
    return entry


def test_insufficient_measured_state_fraction_blocks_robot_pov_evidence() -> None:
    manifest = _complete_export_manifest()
    manifest["optional_exports"] = {
        "formats": {
            "lerobot_v3": _claimed_lerobot_v3_entry(
                real_state_fraction=0.2,
                measured_state_fraction_floor_passed=False,
            )
        }
    }

    readout = build_buyer_package_readout(export_manifest=manifest)

    assert readout["status"] == "blocked_incomplete_package"
    assert (
        "robot_pov_evidence:insufficient_measured_state_fraction:lerobot_v3"
        in readout["blockers"]
    )
    pov = readout["sections"]["robot_pov_evidence"]
    assert pov["status"] == "missing"
    assert pov["measured_state_fractions"]["lerobot_v3"] == 0.2


def test_claimed_lerobot_export_without_state_provenance_fails_closed() -> None:
    manifest = _complete_export_manifest()
    entry = _claimed_lerobot_v3_entry()
    del entry["state_action_provenance"]
    manifest["optional_exports"] = {"formats": {"lerobot_v3": entry}}

    readout = build_buyer_package_readout(export_manifest=manifest)

    assert readout["status"] == "blocked_incomplete_package"
    assert (
        "robot_pov_evidence:measured_state_fraction_unknown:lerobot_v3"
        in readout["blockers"]
    )


def test_measured_state_fraction_floor_passed_keeps_robot_pov_present() -> None:
    manifest = _complete_export_manifest()
    manifest["optional_exports"] = {
        "formats": {"lerobot_v3": _claimed_lerobot_v3_entry()}
    }

    readout = build_buyer_package_readout(export_manifest=manifest)

    assert readout["status"] == "buyer_readout_ready_review_required"
    pov = readout["sections"]["robot_pov_evidence"]
    assert pov["status"] == "present"
    assert pov["measured_state_fractions"]["lerobot_v3"] == 1.0
    assert pov["measured_state_fraction_floor"] == 0.5


def test_product_handoff_is_optional_and_never_blocks() -> None:
    manifest = _complete_export_manifest()

    without = build_buyer_package_readout(export_manifest=manifest)
    assert without["sections"]["product_handoff"]["entitlement_wiring_present"] is False
    assert without["status"] == "buyer_readout_ready_review_required"

    with_handoff = build_buyer_package_readout(
        export_manifest=manifest,
        product_handoff={
            "product_type": "post_training_data_package_v1",
            "product_sku": "PTDP-001",
            "entitlement_id": "ent-42",
            "buyer_review_url": "https://webapp.example/review/ent-42",
        },
    )
    section = with_handoff["sections"]["product_handoff"]
    assert section["entitlement_wiring_present"] is True
    assert section["pricing_is_out_of_band"] is True
