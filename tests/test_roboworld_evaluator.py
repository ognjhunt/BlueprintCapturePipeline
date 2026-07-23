from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.evaluator_evidence_profiles import (
    EVALUATOR_EVIDENCE_PROFILES,
    required_evaluator_evidence_digest_fields,
)
from blueprint_pipeline.roboworld_evaluator import (
    ADMISSION_EVIDENCE_SCHEMA_VERSION,
    JUDGE_CALIBRATION_REQUEST_SCHEMA_VERSION,
    PROGRESS_SCORE_SCHEMA_VERSION,
    aggregate_segment_scores,
    build_default_progress_profile,
    build_roboworld_admission_checklist,
    build_segment_aggregation_ablation,
    run_judge_calibration_campaign,
    validate_progress_profile,
    validate_progress_score,
)


ROOT = Path(__file__).resolve().parents[1]
DIGEST = "a" * 64


def _score(
    value: int,
    *,
    segment_index: int = 0,
    rollout_id: str = "rollout-1",
    frame_scores: list[int] | None = None,
    failure: bool | None = None,
) -> dict:
    profile = build_default_progress_profile()
    failure = value in {1, 3} if failure is None else failure
    stage = {
        0: "no_task_directed_behavior",
        1: "approach",
        2: "approach",
        3: "target_contact",
        4: "near_completion",
        5: "completed",
    }[value]
    views = [
        {
            "view_id": "fixed_external_left",
            "roles_used": ["task_progress", *(["task_completion"] if value == 5 else [])],
            "evidence_refs": [f"video.mp4#segment={segment_index}&view=fixed_left"],
        }
    ]
    if failure:
        views.append(
            {
                "view_id": "wrist",
                "roles_used": ["world_model_failure_detection"],
                "evidence_refs": [f"video.mp4#segment={segment_index}&view=wrist"],
            }
        )
    return {
        "schema_version": PROGRESS_SCORE_SCHEMA_VERSION,
        "profile_id": profile["profile_id"],
        "profile_sha256": profile["profile_sha256"],
        "rollout_id": rollout_id,
        "segment_index": segment_index,
        "criterion_id": "place-target",
        "task_progress_score": value,
        "policy_progress_stage": stage,
        "world_model_failure_stage": (
            "during_approach" if value == 1 and failure else "upon_contact" if failure else "none"
        ),
        "world_model_failure_detected": failure,
        "criterion_evidence_refs": [f"video.mp4#segment={segment_index}"],
        "judge_confidence": 0.9,
        "judge_abstained": False,
        "prompt_sha256": DIGEST,
        "judge_model_sha256": "b" * 64,
        "calibration_set_sha256": "c" * 64,
        "view_evidence": views,
        "sampled_frame_scores": frame_scores if frame_scores is not None else [value, value],
    }


def _load_schema(name: str) -> dict:
    return json.loads((ROOT / "docs" / "schemas" / name).read_text(encoding="utf-8"))


def test_default_profile_is_frozen_valid_and_schema_conformant() -> None:
    profile = build_default_progress_profile()

    validation = validate_progress_profile(profile)

    assert validation["status"] == "validated"
    assert [row["score"] for row in profile["rubric"]] == list(range(6))
    assert profile["segment_aggregation"]["default_strategy"] != "maximum_experimental"
    assert profile["segment_aggregation"]["maximum_is_experimental"] is True
    assert profile["source_method"]["step_forcing_implemented"] is False
    jsonschema.validate(
        profile, _load_schema("roboworld_progress_evaluator_profile.schema.json")
    )


def test_tracked_default_profile_matches_implementation() -> None:
    tracked = json.loads(
        (
            ROOT
            / "docs"
            / "roboworld"
            / "roboworld_progress_evaluator_profile.v1.json"
        ).read_text(encoding="utf-8")
    )

    assert tracked == build_default_progress_profile()


def test_progress_score_preserves_requested_fields_and_view_authority() -> None:
    source = _score(3, failure=True)

    result = validate_progress_score(source)

    assert result["status"] == "validated"
    assert result["task_progress_score"] == 3
    assert result["policy_progress_stage"] == "target_contact"
    assert result["world_model_failure_stage"] == "upon_contact"
    assert result["world_model_failure_detected"] is True
    assert result["criterion_evidence_refs"]
    assert result["judge_confidence"] == 0.9
    assert result["judge_abstained"] is False
    assert result["prompt_sha256"] == DIGEST
    jsonschema.validate(source, _load_schema("roboworld_progress_score.schema.json"))


def test_progress_score_rejects_stage_mismatch_and_abstention_is_not_aggregated() -> None:
    mismatched = _score(5)
    mismatched["policy_progress_stage"] = "approach"
    mismatch_result = validate_progress_score(mismatched)

    assert mismatch_result["status"] == "blocked"
    assert "task_progress_score_policy_stage_mismatch" in mismatch_result["blockers"]

    abstained = _score(4)
    abstained["judge_abstained"] = True
    abstained["abstention_reason"] = "fixed views are fully occluded"
    abstention_result = validate_progress_score(abstained)
    aggregation = aggregate_segment_scores([abstained])

    assert abstention_result["status"] == "abstained"
    assert aggregation["status"] == "blocked"
    assert "segment_0:judge_abstained" in aggregation["blockers"]


def test_wrist_cannot_claim_success_without_calibrated_criterion_override() -> None:
    source = _score(5)
    source["view_evidence"] = [
        {
            "view_id": "wrist",
            "roles_used": ["task_progress", "task_completion"],
            "evidence_refs": ["video.mp4#view=wrist"],
        }
    ]

    blocked = validate_progress_score(source)

    assert blocked["status"] == "blocked"
    assert (
        "progress_score_view_role_unauthorized:wrist:task_completion" in blocked["blockers"]
    )
    assert "task_completion_requires_authorized_completion_view" in blocked["blockers"]

    profile = build_default_progress_profile()
    profile.pop("profile_sha256")
    profile["view_authority"]["criterion_overrides"] = [
        {
            "criterion_id": "place-target",
            "view_id": "wrist",
            "allowed_roles": ["task_progress", "task_completion"],
            "independently_accepted": True,
            "calibration_set_sha256": "d" * 64,
            "reason": "fixed views are fully occluded for this registered criterion",
        }
    ]
    from blueprint_pipeline.roboworld_evaluator import canonical_sha256

    profile["profile_sha256"] = canonical_sha256(profile)
    source["profile_sha256"] = profile["profile_sha256"]

    accepted = validate_progress_score(source, profile=profile)

    assert accepted["status"] == "validated"


def test_all_segment_strategies_are_reported_and_maximum_is_not_default() -> None:
    result = aggregate_segment_scores(
        [
            _score(5, segment_index=0, frame_scores=[4, 5]),
            _score(3, segment_index=1, frame_scores=[4, 3]),
        ]
    )

    assert result["status"] == "complete"
    assert result["aggregations"] == {
        "progress_then_regression_aware": 1.0,
        "terminal": 3.0,
        "mean": 4.0,
        "minimum": 3.0,
        "maximum_experimental": 5.0,
        "stable_maintenance": 3.0,
    }
    assert result["default_strategy"] == "terminal"
    assert result["selected_score"] == 3.0
    assert result["maximum_selected_as_default"] is False


def test_stable_success_requires_adjacent_terminal_frame_maintenance() -> None:
    unstable = aggregate_segment_scores(
        [_score(5, segment_index=0, frame_scores=[5, 5, 4])]
    )
    stable = aggregate_segment_scores(
        [_score(5, segment_index=0, frame_scores=[4, 5, 5])]
    )

    assert unstable["stable_success"] is False
    assert unstable["aggregations"]["stable_maintenance"] == 4.0
    assert stable["stable_success"] is True
    assert stable["aggregations"]["stable_maintenance"] == 5.0


def test_segment_ablation_compares_all_strategies_without_auto_promotion() -> None:
    rollouts = []
    for policy_index, value in enumerate((1, 3, 5), start=1):
        rollouts.append(
            {
                "policy_id": f"policy-{policy_index}",
                "reference_score": float(value),
                "reference_independently_accepted": True,
                "segment_scores": [
                    _score(value, segment_index=0, rollout_id=f"r-{policy_index}"),
                    _score(value, segment_index=1, rollout_id=f"r-{policy_index}"),
                ],
            }
        )

    report = build_segment_aggregation_ablation(rollouts)

    assert report["status"] == "measured"
    assert set(report["strategies"]) == {
        "progress_then_regression_aware",
        "terminal",
        "mean",
        "minimum",
        "maximum_experimental",
        "stable_maintenance",
    }
    assert report["strategies"]["maximum_experimental"]["experimental"] is True
    assert report["strategy_promotion_decision"] == "not_automatically_selected"
    jsonschema.validate(report, _load_schema("segment_aggregation_ablation.schema.json"))


def _campaign() -> dict:
    samples = []
    for index, human_score in enumerate((1, 3, 5), start=1):
        outputs = []
        for judge_id, family, score in (
            ("gpt-judge", "openai_gpt", human_score),
            ("gemini-judge", "google_gemini", 5 if index == 1 else human_score),
        ):
            outputs.append(
                {
                    "judge_id": judge_id,
                    "judge_family": family,
                    "score": score,
                    "confidence": 0.9,
                    "abstained": False,
                    "blinded_to_policy_identity": True,
                    "randomized_order": True,
                    "prompt_sha256": "1" * 64,
                    "judge_model_sha256": "2" * 64,
                    "calibration_set_sha256": "3" * 64,
                }
            )
        samples.append(
            {
                "sample_id": f"sample-{index}",
                "policy_id": f"policy-{index}",
                "task_id": "place-object",
                "view_condition": "three-view",
                "contact_stage": "before-contact" if index == 1 else "after-contact",
                "artifact_type": "none" if index == 3 else "object-morph",
                "human_reference": {
                    "score": human_score,
                    "reviewer_count": 2,
                    "blinded_to_policy_identity": True,
                    "randomized_order": True,
                    "label_artifact_sha256": "4" * 64,
                },
                "judge_outputs": outputs,
            }
        )
    return {
        "schema_version": JUDGE_CALIBRATION_REQUEST_SCHEMA_VERSION,
        "campaign_id": "campaign-1",
        "campaign_version": "1.0.0",
        "frozen": True,
        "sample_manifest_sha256": "5" * 64,
        "samples": samples,
    }


def test_judge_campaign_reports_confusion_calibration_rank_and_bias() -> None:
    campaign = _campaign()
    jsonschema.validate(campaign, _load_schema("judge_calibration_campaign_request.schema.json"))
    report = run_judge_calibration_campaign(campaign)

    assert report["status"] == "measured"
    assert report["sample_count"] == 3
    assert report["policy_count"] == 3
    judges = {row["judge_id"]: row for row in report["judges"]}
    assert judges["gpt-judge"]["confusion_matrix"]["5"]["5"] == 1
    assert judges["gpt-judge"]["policy_rank_stability"]["spearman"] == 1.0
    assert judges["gemini-judge"]["false_success_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert judges["gemini-judge"]["bias_breakdowns"]["artifact_type"]
    assert judges["gemini-judge"]["confidence_bins"]
    jsonschema.validate(
        report, _load_schema("judge_calibration_campaign_report.schema.json")
    )


def _complete_admission_evidence() -> dict:
    release = {
        "code_released": True,
        "weights_released": True,
        "source_uri": "https://example.test/roboworld.git",
        "source_revision": "1" * 40,
        "software_license": "Apache-2.0",
        "software_license_sha256": "1" * 64,
        "checkpoint_uri": "https://example.test/model.safetensors",
        "checkpoint_sha256": "2" * 64,
        "weights_license": "research-license",
        "weights_license_sha256": "3" * 64,
        "container_image_digest": f"example.test/roboworld@sha256:{'4' * 64}",
        "preprocessing_manifest_sha256": "5" * 64,
        "data_filter_manifest_sha256": "6" * 64,
        "action_normalization_manifest_sha256": "7" * 64,
        "training_schedule_manifest_sha256": "8" * 64,
        "evaluation_script_sha256": "9" * 64,
    }
    return {
        "schema_version": ADMISSION_EVIDENCE_SCHEMA_VERSION,
        "paper_version": "arXiv:2607.01060v4",
        "upstream_release": release,
        "diagnostic_reproduction": {
            "executed": True,
            "bair_protocol_sha256": "a" * 64,
            "step_forcing_checkpoint_sha256": "b" * 64,
            "metrics_artifact_sha256": "c" * 64,
        },
        "published_result_reproduction": {
            "executed": True,
            "policy_count": 8,
            "rollout_count": 4186,
            "roboarena_snapshot_sha256": "d" * 64,
            "policy_registry_sha256": "e" * 64,
            "initial_condition_manifest_sha256": "f" * 64,
            "rollout_results_sha256": "1" * 64,
            "judge_prompt_sha256": "2" * 64,
            "rank_report_sha256": "3" * 64,
        },
        "blueprint_comparison": {
            "executed": True,
            "frozen_benchmark_spec_sha256": "4" * 64,
            "environment_manifest_sha256": "5" * 64,
            "policy_registry_sha256": "6" * 64,
            "current_wam_results_sha256": "7" * 64,
            "physics_sim_results_sha256": "8" * 64,
            "roboworld_results_sha256": "9" * 64,
            "external_anchor_results_sha256": "a" * 64,
            "comparison_report_sha256": "b" * 64,
        },
    }


def test_admission_waits_for_code_and_complete_evidence_can_pass() -> None:
    pending = _complete_admission_evidence()
    pending["upstream_release"] = {"code_released": False, "weights_released": False}

    pending_result = build_roboworld_admission_checklist(pending)
    complete_result = build_roboworld_admission_checklist(_complete_admission_evidence())

    assert pending_result["status"] == "awaiting_upstream_release"
    assert "roboworld_upstream_code_not_released" in pending_result["blockers"]
    assert pending_result["deferred_work"]["step_forcing_training_reimplementation_authorized"] is False
    assert complete_result["status"] == "admitted_for_configured_backend_evaluation"
    assert complete_result["blockers"] == []
    jsonschema.validate(
        complete_result,
        _load_schema("roboworld_admission_reproduction_checklist.schema.json"),
    )


def test_tracked_admission_checklist_matches_frozen_evidence() -> None:
    evidence = json.loads(
        (
            ROOT / "docs" / "roboworld" / "roboworld_admission_evidence.v1.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(evidence, _load_schema("roboworld_admission_evidence.schema.json"))
    tracked = json.loads(
        (
            ROOT
            / "docs"
            / "roboworld"
            / "roboworld_admission_reproduction_checklist.v1.json"
        ).read_text(encoding="utf-8")
    )

    assert tracked == build_roboworld_admission_checklist(evidence)
    assert tracked["status"] == "awaiting_upstream_release"
    jsonschema.validate(
        tracked,
        _load_schema("roboworld_admission_reproduction_checklist.schema.json"),
    )


def test_roboworld_evidence_profile_requires_new_evaluator_artifacts() -> None:
    profile = EVALUATOR_EVIDENCE_PROFILES["roboworld_progress_v1"]
    fields = required_evaluator_evidence_digest_fields("roboworld_progress_v1")

    assert profile["step_forcing_backend_required"] is False
    assert profile["paper_metrics_inherited"] is False
    assert "progress_profile_manifest_sha256" in fields
    assert "view_authority_manifest_sha256" in fields
    assert "judge_calibration_set_sha256" in fields
    assert "segment_aggregation_report_sha256" in fields


def test_profile_digest_and_override_validation_fail_closed() -> None:
    profile = build_default_progress_profile()
    profile["segment_aggregation"]["default_strategy"] = "maximum_experimental"

    result = validate_progress_profile(profile)

    assert result["status"] == "blocked"
    assert "maximum_segment_aggregation_cannot_be_default" in result["blockers"]
    assert "progress_profile_digest_mismatch" in result["blockers"]


def test_malformed_campaign_fails_closed() -> None:
    campaign = copy.deepcopy(_campaign())
    campaign["samples"][0]["judge_outputs"][0]["blinded_to_policy_identity"] = False

    report = run_judge_calibration_campaign(campaign)

    assert report["status"] == "blocked"
    assert "judge_campaign_output_not_blinded:0:0" in report["blockers"]
