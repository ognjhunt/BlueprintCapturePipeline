from blueprint_pipeline.buyer_package_readout import (
    BUYER_CRITICAL_SECTIONS,
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
        },
        "manifest_counts": {"failure_label_count": 3},
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

    markdown = render_buyer_package_readout_markdown(readout)
    assert "Highest truthful claim: no_claim" in markdown
    assert "not deployment approval" in markdown
    assert "Simulator results are not real-world outcomes." in markdown


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
