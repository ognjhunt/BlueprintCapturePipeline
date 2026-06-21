import json
from pathlib import Path

import pytest

from blueprint_pipeline.policy_improvement_run import (
    _failure_mode_summary,
    _float,
    _split_counts,
    _status_and_blockers,
    build_policy_improvement_run_offer,
    main,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _capture_root(tmp_path: Path) -> Path:
    capture_root = (
        tmp_path
        / "local-blueprint"
        / "scenes"
        / "policy-site"
        / "captures"
        / "capture-policy"
    )
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "policy-site", "capture_id": "capture-policy"},
    )
    robot_eval = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(robot_eval / "site_card.json", {"site_id": "policy-site"})
    _write_json(robot_eval / "task_cards.json", {"cards": [{"task_id": "tote-transfer"}]})
    _write_json(
        robot_eval / "scenario_cards.json",
        {"cards": [{"scenario_id": "blocked-aisle", "task_id": "tote-transfer"}]},
    )
    _write_json(robot_eval / "eval_cards.json", {"cards": [{"scenario_id": "blocked-aisle"}]})
    _write_json(robot_eval / "rights_packet.json", {"status": "review_required"})
    _write_json(robot_eval / "proof_boundaries.json", {"robot_readiness_proven": False})
    return capture_root


def _complete_job_dir(capture_root: Path) -> Path:
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "policy-improvement-job"
    _write_json(
        job_dir / "job_request.json",
        {
            "schema_version": "robot_eval_job_request.v1",
            "policy_package": {
                "policy_id": "customer-tote-policy",
                "action_interface": "joint_position_delta_20hz",
            },
            "robot": {"robot_model": "g1-humanoid"},
            "task": {"task_id": "tote-transfer"},
            "thresholds": {
                "target_success_rate": 0.95,
                "max_cycle_time_seconds": 90.0,
            },
        },
    )
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": [
                {
                    "scenario_eval_run_id": "dev-1",
                    "split": "train",
                    "task_id": "tote-transfer",
                },
                {
                    "scenario_eval_run_id": "val-1",
                    "split": "validation",
                    "task_id": "tote-transfer",
                },
                {
                    "scenario_eval_run_id": "heldout-1",
                    "split": "heldout",
                    "task_id": "tote-transfer",
                },
            ],
        },
    )
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "schema_version": "robot_eval_job_normalized_attempt_trace.v1",
            "attempt_count": 3,
        },
    )
    _write_json(
        job_dir / "failure_labels.json",
        {
            "schema_version": "robot_eval_job_failure_labels.v1",
            "label_count": 2,
            "labels": [
                {"attempt_id": "a1", "failure_mode_id": "grasp_alignment_miss"},
                {"attempt_id": "a2", "failure_mode_id": "grasp_alignment_miss"},
            ],
        },
    )
    _write_json(
        job_dir / "policy_package_manifest.json",
        {
            "schema_version": "robot_eval_policy_package_manifest.v1",
            "policy_id": "customer-tote-policy",
            "action_interface": "joint_position_delta_20hz",
        },
    )
    _write_json(
        job_dir / "post_training_data_package_export_manifest.json",
        {
            "schema_version": "post_training_data_package_export.v1",
            "status": "export_ready_review_required",
            "package_type": "post_training_data_package",
            "manifest_counts": {"attempt_count": 3, "failure_label_count": 2},
            "export_policy": {
                "scenario_eval_matrix_included": True,
                "failure_labels_included": True,
            },
        },
    )
    autoresearch = job_dir / "policy_autoresearch"
    _write_json(
        autoresearch / "policy_autoresearch_report.json",
        {
            "schema_version": "policy_autoresearch_report.v1",
            "status": "promoted",
            "target_success_reached": True,
            "baseline_heldout_success_rate": 0.82,
            "best_heldout_success_rate": 0.96,
            "frozen_verifier_sha256": "abc123",
        },
    )
    _write_json(
        autoresearch / "policy_candidate_package.json",
        {
            "schema_version": "policy_autoresearch_candidate_package.v1",
            "status": "promoted_sim_only_policy_candidate",
            "artifact_kind": "adapter",
            "frozen_verifier_sha256": "abc123",
            "simulator_execution_proven": True,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )
    _write_json(
        autoresearch / "heldout_eval_result.json",
        {
            "schema_version": "policy_autoresearch_eval_result.v1",
            "safety_contact_gate_passed": True,
            "task_success_summary": {"task_success_rate": 0.96},
        },
    )
    _write_json(autoresearch / "agent_idea_tree.json", {"ideas": []})
    _write_json(autoresearch / "followup_real_world_validation_request.json", {"status": "queued"})
    return job_dir


def test_policy_improvement_run_offer_binds_eval_post_training_and_candidate_artifacts(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    job_dir = _complete_job_dir(capture_root)

    result = build_policy_improvement_run_offer(
        capture_root=capture_root,
        job_dir=job_dir,
        access_level="config_adapter",
        improvement_targets=("adapter", "task_head"),
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert result["status"] == "improvement_candidate_ready_for_customer_review"
    assert result["product"]["extends"] == ["Task Evaluation Run", "Post-Training Data Package"]
    assert result["access_model"]["source_code_required"] is False
    assert result["customer_inputs"]["base_policy_or_model"]["value"] == "customer-tote-policy"
    assert result["customer_inputs"]["success_threshold"]["value"] == 0.95
    assert result["scenario_split_policy"]["split_counts"]["heldout"] == 1
    assert result["failure_mode_summary"]["dominant_failure_modes"][0] == {
        "failure_mode": "grasp_alignment_miss",
        "count": 2,
    }
    assert result["policy_autoresearch_summary"]["heldout_success_rate_delta"] == 0.14
    assert result["claim_boundary"]["simulator_execution_proven"] is True
    assert result["claim_boundary"]["robot_readiness_proven"] is False
    assert result["claim_boundary"]["public_claim_upgrade_allowed"] is False
    assert "policy_candidate_package" in result["included_artifacts"]

    persisted = _read_json(job_dir / "policy_improvement_run" / "policy_improvement_run_offer.json")
    brief = (job_dir / "policy_improvement_run" / "policy_improvement_run_offer.md").read_text(
        encoding="utf-8"
    )
    assert persisted["status"] == result["status"]
    assert "Policy Improvement Run" in brief
    assert "deployment approval" in brief


def test_policy_improvement_run_offer_accepts_promoted_wam_candidate(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    job_dir = _complete_job_dir(capture_root)
    autoresearch = job_dir / "policy_autoresearch"
    package = _read_json(autoresearch / "policy_candidate_package.json")
    package["status"] = "promoted_wam_policy_candidate"
    package["sim_only_policy_improvement_support_artifact"] = False
    package["wam_policy_improvement_support_artifact"] = True
    package["simulator_execution_proven"] = False
    _write_json(autoresearch / "policy_candidate_package.json", package)
    _write_json(
        job_dir / "policy_ranking_scorecard.json",
        {
            "schema_version": "policy_ranking_scorecard.v1",
            "status": "completed",
            "evaluation_substrate": "fixture_wam",
            "top_policy_id": "site_finetune_policy",
            "policy_count": 2,
            "scenario_attempt_count": 6,
        },
    )
    _write_json(
        job_dir / "wam_eval_claim_boundary.json",
        {
            "schema_version": "wam_eval_claim_boundary.v1",
            "evaluation_substrate": "fixture_wam",
            "generated_rollouts_are_model_derived_support_artifacts": True,
            "customer_specific_srcc_claimed": False,
        },
    )

    result = build_policy_improvement_run_offer(
        capture_root=capture_root,
        job_dir=job_dir,
        access_level="config_adapter",
        improvement_targets=("adapter",),
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "improvement_candidate_ready_for_customer_review"
    assert result["policy_autoresearch_summary"]["candidate_status"] == (
        "promoted_wam_policy_candidate"
    )
    assert result["wam_evaluation_summary"]["status"] == "completed"
    assert result["wam_evaluation_summary"]["evaluation_substrate"] == "fixture_wam"
    assert result["wam_evaluation_summary"]["customer_specific_srcc_claimed"] is False
    assert result["claim_boundary"]["simulator_execution_proven"] is False


def test_policy_improvement_run_offer_blocks_missing_customer_inputs_and_holdout(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "incomplete-policy-job"
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": [{"scenario_eval_run_id": "dev-1", "split": "train"}],
        },
    )

    result = build_policy_improvement_run_offer(
        capture_root=capture_root,
        job_dir=job_dir,
        access_level="black_box",
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert result["status"] == "blocked_missing_policy_improvement_inputs"
    assert "missing_customer_input_base_policy_or_model" in result["blockers"]
    assert "missing_customer_input_robot_embodiment" in result["blockers"]
    assert "missing_heldout_or_sealed_audit_split" in result["blockers"]
    assert result["claim_boundary"]["source_code_required_by_default"] is False


def test_policy_improvement_run_helpers_cover_parse_and_split_edges() -> None:
    assert _float("0.75") == 0.75
    assert _float("not-a-number") is None
    assert _split_counts({"runs": "not-a-list"}) == {
        "development": 0,
        "validation": 0,
        "heldout": 0,
        "sealed_audit": 0,
        "unknown": 0,
    }
    assert _split_counts(
        {
            "runs": [
                "bad-row",
                {"split": "sealed"},
                {"scenario_split": "something-new"},
            ]
        }
    ) == {
        "development": 0,
        "validation": 0,
        "heldout": 0,
        "sealed_audit": 1,
        "unknown": 2,
    }
    assert _failure_mode_summary(
        {"failures": [{"reason": "drop"}, {"label": "drop"}, "skip"]}
    )["dominant_failure_modes"] == [{"failure_mode": "drop", "count": 2}]


def test_policy_improvement_run_status_gate_progression() -> None:
    present_inputs = {
        key: {"status": "present", "value": "x"}
        for key in (
            "base_policy_or_model",
            "robot_embodiment",
            "action_interface",
            "target_task",
            "success_threshold",
            "cycle_time_threshold_seconds",
        )
    }
    split_counts = {"heldout": 1, "sealed_audit": 0}
    policy_summary = {
        "candidate_status": "promoted_sim_only_policy_candidate",
        "safety_contact_gate_passed": True,
    }
    ready_package = {"status": "export_ready_review_required"}

    assert _status_and_blockers(
        customer_inputs=present_inputs,
        included_artifacts={},
        split_counts=split_counts,
        policy_summary=policy_summary,
        post_training_package=ready_package,
    ) == (
        "blocked_missing_policy_improvement_inputs",
        ["missing_scenario_eval_matrix"],
    )
    assert _status_and_blockers(
        customer_inputs=present_inputs,
        included_artifacts={"scenario_eval_matrix": "matrix.json"},
        split_counts=split_counts,
        policy_summary=policy_summary,
        post_training_package=ready_package,
    ) == (
        "ready_for_baseline_evaluation",
        ["baseline_normalized_attempt_trace_missing"],
    )
    assert _status_and_blockers(
        customer_inputs=present_inputs,
        included_artifacts={
            "scenario_eval_matrix": "matrix.json",
            "normalized_attempt_trace": "trace.json",
        },
        split_counts=split_counts,
        policy_summary=policy_summary,
        post_training_package=ready_package,
    ) == ("ready_for_failure_diagnosis", ["failure_labels_missing"])
    assert _status_and_blockers(
        customer_inputs=present_inputs,
        included_artifacts={
            "scenario_eval_matrix": "matrix.json",
            "normalized_attempt_trace": "trace.json",
            "failure_labels": "labels.json",
        },
        split_counts=split_counts,
        policy_summary=policy_summary,
        post_training_package={"status": "missing"},
    ) == (
        "ready_for_post_training_data_package",
        ["post_training_data_package_export_not_ready"],
    )
    assert _status_and_blockers(
        customer_inputs=present_inputs,
        included_artifacts={
            "scenario_eval_matrix": "matrix.json",
            "normalized_attempt_trace": "trace.json",
            "failure_labels": "labels.json",
        },
        split_counts=split_counts,
        policy_summary=policy_summary,
        post_training_package=ready_package,
    ) == ("ready_for_policy_autoresearch", ["policy_autoresearch_report_missing"])
    assert _status_and_blockers(
        customer_inputs=present_inputs,
        included_artifacts={
            "scenario_eval_matrix": "matrix.json",
            "normalized_attempt_trace": "trace.json",
            "failure_labels": "labels.json",
            "policy_autoresearch_report": "report.json",
        },
        split_counts=split_counts,
        policy_summary={"candidate_status": "searched", "safety_contact_gate_passed": True},
        post_training_package=ready_package,
    ) == ("completed_no_promoted_candidate", ["promoted_policy_candidate_missing"])
    assert _status_and_blockers(
        customer_inputs=present_inputs,
        included_artifacts={
            "scenario_eval_matrix": "matrix.json",
            "normalized_attempt_trace": "trace.json",
            "failure_labels": "labels.json",
            "policy_autoresearch_report": "report.json",
        },
        split_counts=split_counts,
        policy_summary={
            "candidate_status": "promoted_sim_only_policy_candidate",
            "safety_contact_gate_passed": False,
        },
        post_training_package=ready_package,
    ) == (
        "blocked_candidate_failed_safety_contact_gate",
        ["policy_candidate_safety_contact_gate_not_passed"],
    )


def test_policy_improvement_run_rejects_unknown_access_and_targets(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    job_dir = _complete_job_dir(capture_root)

    with pytest.raises(ValueError, match="Unsupported access level"):
        build_policy_improvement_run_offer(
            capture_root=capture_root,
            job_dir=job_dir,
            access_level="unknown",
        )

    with pytest.raises(ValueError, match="Unsupported improvement target"):
        build_policy_improvement_run_offer(
            capture_root=capture_root,
            job_dir=job_dir,
            improvement_targets=("adapter", "unsupported"),
        )


def test_policy_improvement_run_main_returns_success_and_blocked_status(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    capture_root = _capture_root(tmp_path)
    job_dir = _complete_job_dir(capture_root)
    output_dir = tmp_path / "offer"

    assert (
        main(
            [
                "--capture-root",
                str(capture_root),
                "--job-dir",
                str(job_dir),
                "--output-dir",
                str(output_dir),
                "--access-level",
                "source_training",
                "--customer-policy-ref",
                "cli-policy",
                "--embodiment",
                "unitree-g1",
                "--action-interface",
                "joint-delta",
                "--target-task",
                "tote-transfer",
                "--success-threshold",
                "0.9",
                "--cycle-time-threshold-seconds",
                "80",
                "--improvement-target",
                "complete_policy",
            ]
        )
        == 0
    )
    assert "status=improvement_candidate_ready_for_customer_review" in capsys.readouterr().out

    blocked_job = capture_root / "pipeline" / "robot_eval_jobs" / "blocked"
    assert main(["--capture-root", str(capture_root), "--job-dir", str(blocked_job)]) == 1
    assert "status=blocked_missing_policy_improvement_inputs" in capsys.readouterr().out
