from __future__ import annotations

import json
import sys
import types
from pathlib import Path


import blueprint_pipeline.post_training_data_package as package_module
from blueprint_pipeline.post_training_data_package import (
    _artifact,
    _read_optional_mapping,
    _rows,
    _write_native_hdf5,
    _write_native_parquet,
    build_post_training_data_package_export,
    main,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    (capture_root / "raw").mkdir(parents=True)
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    return capture_root


def _seed_required_pipeline_artifacts(capture_root: Path) -> None:
    dataset_root = capture_root / "pipeline" / "robot_eval_dataset"
    for name in (
        "site_card.json",
        "task_cards.json",
        "scenario_cards.json",
        "eval_cards.json",
        "proof_boundaries.json",
    ):
        _write_json(dataset_root / name, {"name": name})


def _seed_ready_job(job_dir: Path) -> None:
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "attempt_count": 1,
            "attempts": [
                {
                    "attempt_id": "attempt-1",
                    "scenario_id": "scenario-1",
                    "task_id": "task-1",
                    "policy_id": "policy-1",
                    "success": True,
                    "status": "passed",
                    "metrics": {"score": 1.0},
                    "action_trace": [
                        {
                            "sc3_7d_delta_ee_pose": [
                                0.05,
                                0.0,
                                0.01,
                                0.0,
                                0.0,
                                0.02,
                                1.0,
                            ]
                        }
                    ],
                    "observation_refs": [{"frame": "000001"}],
                }
            ],
        },
    )
    _write_json(
        job_dir / "failure_labels.json",
        {
            "label_count": 1,
            "labels": [
                {"attempt_id": "attempt-1", "label": "nominal"},
                {"scenario_id": "scenario-1", "label": "scenario-only"},
            ],
        },
    )
    _write_json(job_dir / "arena_eval_metrics.json", {"score": 1.0, "attempt_count": 1})
    _write_json(
        job_dir / "clips_manifest.json",
        {
            "clip_count": 1,
            "clips": [
                {
                    "clip_id": "clip-1",
                    "clip_path": "clip-1.mp4",
                    "attempt_id": "attempt-1",
                    "frame_count": 32,
                    "camera_motion_m": 0.0,
                    "visible_skeleton_fraction": 0.9,
                    "sharpness_score": 72.0,
                    "semantic_dedup_key": "scene-1|task-1|attempt-1",
                }
            ],
        },
    )
    for name in (
        "prediction_outcome_ledger.json",
        "calibration_report.json",
        "breakage_library.json",
        "visual_review_ledger.json",
        "simulator_provider_adapter_manifest.json",
        "simulator_command_batch_attempt_trace.jsonl",
        "simulator_command_batch_contact_stream.jsonl",
        "simulator_command_batch_planner_state.jsonl",
        "simulator_command_batch_control_stream.jsonl",
        "sim_vs_real_calibration_report.json",
        "deployment_outcome_intake_manifest.json",
        "deployment_outcome_ledger.json",
        "real_world_validation_followup_plan.json",
        "real_world_validation_followup_request_queue.json",
        "live_eval_closure_manifest.json",
        "live_eval_closure_evidence.json",
        "robot_eval_report.json",
        "proof_boundary.json",
        "review_resolution_ledger.json",
        "accepted_failure_labels.json",
        "customer_handoff_report.json",
        "delivery_manifest.json",
        "signed_access_manifest.json",
    ):
        path = job_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    _write_json(
        job_dir / "live_eval_closure_manifest.json",
        {
            "status": "local_artifacts_ready_live_external_blocked",
            "gates": {
                "webapp_upstream_truth": {"passed": True, "blockers": []},
                "rights_privacy_scope": {"passed": True, "blockers": []},
                "review_acceptance": {"passed": True, "blockers": []},
                "signed_delivery_access": {"passed": False, "blockers": ["signed_url_missing"]},
            },
        },
    )


def test_post_training_data_package_blocks_with_manifest_only_defaults(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)

    manifest = build_post_training_data_package_export(capture_root=capture_root)

    assert manifest["status"] == "blocked_missing_inputs"
    assert "missing_normalized_attempt_trace" in manifest["blockers"]
    output_dir = capture_root / "pipeline" / "post_training_data_package"
    assert (output_dir / "data" / "attempts.jsonl").is_file()
    metrics = json.loads((output_dir / "data" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "missing_source_metrics"
    clips = json.loads((output_dir / "clips_manifest.json").read_text(encoding="utf-8"))
    assert clips["status"] == "missing_source_clips"
    optional = json.loads((output_dir / "optional_export_manifest.json").read_text(encoding="utf-8"))
    assert optional["formats"]["video_bundle"]["clip_count"] == 0
    assert manifest["claim_boundary"]["rank_fidelity_result_proven"] is False
    assert manifest["claim_boundary"]["deployment_approval_proven"] is False
    assert manifest["claim_boundary"]["package_delivery_is_deployment_approval"] is False
    assert manifest["claim_boundary"]["post_training_package_export_ready"] is False


def test_post_training_data_package_exports_ready_package_with_policy_flags(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    output_dir = tmp_path / "package"

    def _write_fake_hdf5(path: Path, _rows_arg: object) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake-hdf5", encoding="utf-8")
        return True

    def _write_fake_parquet(path: Path, _rows_arg: object) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake-parquet", encoding="utf-8")
        return True

    monkeypatch.setattr(package_module, "_write_native_hdf5", _write_fake_hdf5)
    monkeypatch.setattr(package_module, "_write_native_parquet", _write_fake_parquet)

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    assert manifest["status"] == "export_ready_review_required"
    assert manifest["blockers"] == []
    assert manifest["manifest_counts"]["attempt_count"] == 1
    assert manifest["manifest_counts"]["failure_label_count"] == 1
    assert manifest["manifest_counts"]["clip_count"] == 1
    assert manifest["manifest_counts"]["curated_clip_count"] == 1
    assert manifest["manifest_counts"]["rejected_clip_count"] == 0
    assert manifest["manifest_counts"]["semantic_duplicate_group_count"] == 0
    assert manifest["manifest_counts"]["valid_sc3_7d_action_count"] == 1
    assert manifest["included_artifacts"]["signed_access_manifest"] == (
        "signed_access_manifest.json"
    )
    assert manifest["handoff_records"]["proof_boundary_path"] == "proof_boundary.json"
    assert manifest["handoff_records"]["delivery_manifest_path"] == "delivery_manifest.json"
    assert manifest["handoff_records"]["signed_access_manifest_path"] == (
        "signed_access_manifest.json"
    )
    assert manifest["handoff_records"]["live_closure_gate_references"][
        "review_acceptance"
    ]["passed"] is True
    assert manifest["handoff_records"]["live_closure_gate_references"][
        "signed_delivery_access"
    ]["blockers"] == ["signed_url_missing"]
    assert manifest["claim_boundary"]["post_training_package_export_ready"] is True
    assert manifest["claim_boundary"]["review_acceptance_proven"] is True
    assert manifest["claim_boundary"]["signed_delivery_access_proven"] is False
    assert manifest["claim_boundary"]["deployment_approval_proven"] is False
    assert manifest["export_policy"]["simulator_command_batch_trace_streams_included"] is True
    assert manifest["export_policy"]["deployment_outcomes_included"] is True
    assert manifest["export_policy"]["real_world_validation_followup_queue_included"] is True
    assert manifest["export_policy"]["rl_post_training_handoff_included"] is True
    assert manifest["export_policy"]["concurrent_baseline_ab_plan_included"] is True
    assert manifest["export_policy"]["bottleneck_stage_detection_included"] is True
    assert manifest["export_policy"]["speed_curriculum_plan_included"] is True
    assert manifest["export_policy"]["action_chunk_continuity_qa_included"] is True
    assert manifest["export_policy"]["intervention_safety_ledger_included"] is True
    assert manifest["export_policy"]["oscar_style_curation_filters_passed"] is True
    assert manifest["export_policy"]["semantic_dedup_passed"] is True
    assert manifest["export_policy"]["sc3_7d_action_contract_passed"] is True
    assert manifest["claim_boundary"]["oscar_style_curation_filters_proven"] is True
    assert manifest["claim_boundary"]["semantic_dedup_proven"] is True
    assert manifest["claim_boundary"]["sc3_7d_action_contract_proven"] is True
    assert manifest["rl_post_training_handoff_packet_path"] == (
        "rl_post_training_handoff_packet.json"
    )
    rl_handoff = json.loads(
        (output_dir / "rl_post_training_handoff_packet.json").read_text(encoding="utf-8")
    )
    assert rl_handoff["schema_version"] == "rl_post_training_handoff_packet.v1"
    assert rl_handoff["success_definition"]["source"] == (
        "job_request.thresholds + evaluation_result.standard_policy_scorecard"
    )
    assert rl_handoff["sparse_reward_signal"]["reward_family"] == (
        "sparse_task_success_with_intervention_penalties"
    )
    assert rl_handoff["concurrent_baseline_ab"]["old_run_only_comparison_allowed"] is False
    assert rl_handoff["claim_boundary"]["speed_curriculum_is_plan_not_completed_training"] is True
    package_index = json.loads((output_dir / "package_index.json").read_text(encoding="utf-8"))
    assert package_index["files"]["rl_post_training_handoff_packet"] == (
        "rl_post_training_handoff_packet.json"
    )
    assert package_index["files"]["curation_report"] == "curation_report.json"
    assert package_index["files"]["semantic_dedup_report"] == "semantic_dedup_report.json"
    assert package_index["files"]["sc3_action_normalization_report"] == (
        "sc3_action_normalization_report.json"
    )
    curation = json.loads((output_dir / "curation_report.json").read_text(encoding="utf-8"))
    semantic_dedup = json.loads(
        (output_dir / "semantic_dedup_report.json").read_text(encoding="utf-8")
    )
    sc3_action = json.loads(
        (output_dir / "sc3_action_normalization_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert curation["status"] == "passed"
    assert semantic_dedup["status"] == "passed"
    assert sc3_action["status"] == "passed"
    assert sc3_action["claim_boundary"]["missing_actions_exported_as_identity_pose"] is False
    optional = json.loads((output_dir / "optional_export_manifest.json").read_text(encoding="utf-8"))
    assert optional["formats"]["hdf5"]["status"] == "written_native"
    assert optional["formats"]["parquet"]["status"] == "written_native"
    archive_members = json.loads((output_dir / "archive_manifest.json").read_text(encoding="utf-8"))
    assert "exports/rlds/episodes.jsonl" in archive_members["included_files"]
    assert "rl_post_training_handoff_packet.json" in archive_members["included_files"]
    assert "curation_report.json" in archive_members["included_files"]
    assert "semantic_dedup_report.json" in archive_members["included_files"]
    assert "sc3_action_normalization_report.json" in archive_members["included_files"]


def test_post_training_data_package_blocks_invalid_sc3_actions_and_source_filters(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "attempt_count": 1,
            "attempts": [
                {
                    "attempt_id": "attempt-1",
                    "scenario_id": "scenario-1",
                    "task_id": "task-1",
                    "policy_id": "policy-1",
                    "success": True,
                    "status": "passed",
                    "metrics": {"score": 1.0},
                    "action_trace": [{"joint": "arm"}],
                }
            ],
        },
    )
    _write_json(
        job_dir / "clips_manifest.json",
        {
            "clip_count": 1,
            "clips": [
                {
                    "clip_id": "clip-1",
                    "clip_path": "clip-1.mp4",
                    "attempt_id": "attempt-1",
                    "frame_count": 4,
                    "camera_motion_m": 0.0,
                    "action_motion_score": 0.1,
                    "visible_skeleton_fraction": 0.9,
                    "sharpness_score": 0.0,
                    "semantic_dedup_key": "scene-1|task-1|attempt-1",
                }
            ],
        },
    )

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    assert manifest["status"] == "blocked_package_quality_gates"
    assert "curation:clip-1:min_frame_count_failed" in manifest["blockers"]
    assert "curation:clip-1:blur_or_sharpness_evidence_failed" in manifest["blockers"]
    assert (
        "sc3_action:attempt-1:sc3_7d_delta_end_effector_pose_missing_or_invalid"
        in manifest["blockers"]
    )
    assert manifest["claim_boundary"]["post_training_package_export_ready"] is False
    assert manifest["claim_boundary"]["oscar_style_curation_filters_proven"] is False
    assert manifest["claim_boundary"]["sc3_7d_action_contract_proven"] is False


def test_post_training_data_package_blocks_semantic_duplicate_clips(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    _write_json(
        job_dir / "clips_manifest.json",
        {
            "clip_count": 2,
            "clips": [
                {
                    "clip_id": "clip-1",
                    "clip_path": "clip-1.mp4",
                    "attempt_id": "attempt-1",
                    "frame_count": 32,
                    "camera_motion_m": 0.0,
                    "action_motion_score": 0.1,
                    "visible_skeleton_fraction": 0.9,
                    "sharpness_score": 72.0,
                    "semantic_dedup_key": "duplicate-scene-task-trajectory",
                },
                {
                    "clip_id": "clip-2",
                    "clip_path": "clip-2.mp4",
                    "attempt_id": "attempt-1",
                    "frame_count": 32,
                    "camera_motion_m": 0.0,
                    "action_motion_score": 0.1,
                    "visible_skeleton_fraction": 0.9,
                    "sharpness_score": 72.0,
                    "semantic_dedup_key": "duplicate-scene-task-trajectory",
                },
            ],
        },
    )

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    assert manifest["status"] == "blocked_package_quality_gates"
    assert (
        "semantic_dedup:semantic_duplicate_group:duplicate-scene-task-trajectory"
        in manifest["blockers"]
    )
    assert manifest["manifest_counts"]["semantic_duplicate_group_count"] == 1
    assert manifest["claim_boundary"]["semantic_dedup_proven"] is False


def test_post_training_data_package_writes_blocked_customer_handoff_manifests(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    for name in (
        "customer_handoff_report.json",
        "delivery_manifest.json",
        "signed_access_manifest.json",
    ):
        (job_dir / name).unlink()
    _write_json(
        job_dir / "live_eval_closure_manifest.json",
        {
            "status": "blocked",
            "gates": {
                "webapp_upstream_truth": {
                    "passed": False,
                    "blockers": [
                        "missing_webapp_request_id",
                        "webapp_upstream_ids_not_grounded_in_capture_or_webapp_source",
                    ],
                    "evidence": {
                        "ids": {
                            "site_submission_id": "site-1",
                            "request_id": "",
                            "buyer_request_id": "buyer-1",
                            "capture_job_id": "capture-job-1",
                        }
                    },
                },
                "rights_privacy_scope": {"passed": True, "blockers": []},
                "review_acceptance": {
                    "passed": False,
                    "blockers": ["review_acceptance_evidence_missing"],
                },
                "signed_delivery_access": {
                    "passed": False,
                    "blockers": [
                        "signed_delivery_evidence_missing",
                        "signed_delivery_access_not_proven",
                    ],
                },
            },
        },
    )
    monkeypatch.setattr(package_module, "_write_native_hdf5", lambda path, rows: False)
    monkeypatch.setattr(package_module, "_write_native_parquet", lambda path, rows: False)

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    assert manifest["status"] == "export_ready_review_required"
    assert manifest["included_artifacts"]["customer_handoff_report"] == (
        "customer_handoff_report.json"
    )
    assert manifest["included_artifacts"]["delivery_manifest"] == "delivery_manifest.json"
    assert manifest["included_artifacts"]["signed_access_manifest"] == (
        "signed_access_manifest.json"
    )
    assert manifest["handoff_records"]["post_training_package_export_ready"] is True
    assert manifest["handoff_records"]["customer_handoff_ready"] is False
    assert (
        "webapp_upstream_truth:missing_webapp_request_id"
        in manifest["handoff_records"]["customer_handoff_blockers"]
    )
    assert (
        "review_acceptance:review_acceptance_evidence_missing"
        in manifest["handoff_records"]["customer_handoff_blockers"]
    )
    assert (
        "signed_delivery_access:signed_delivery_access_not_proven"
        in manifest["handoff_records"]["customer_handoff_blockers"]
    )
    assert manifest["claim_boundary"]["post_training_package_export_ready"] is True
    assert manifest["claim_boundary"]["customer_handoff_ready"] is False
    assert manifest["claim_boundary"]["hosted_access_ready"] is False
    assert manifest["claim_boundary"]["deployment_approval_proven"] is False
    assert manifest["claim_boundary"]["safety_validation_proven"] is False
    delivery = json.loads((job_dir / "delivery_manifest.json").read_text(encoding="utf-8"))
    signed_access = json.loads((job_dir / "signed_access_manifest.json").read_text(encoding="utf-8"))
    handoff = json.loads((job_dir / "customer_handoff_report.json").read_text(encoding="utf-8"))
    assert delivery["status"] == "export_ready_handoff_blocked"
    assert signed_access["status"] == "blocked_signed_delivery_access"
    assert handoff["post_training_data_package_handoff"]["customer_handoff_ready"] is False
    assert delivery["claim_boundary"]["delivery_access_is_deployment_approval"] is False
    assert signed_access["claim_boundary"]["physical_robot_readiness_proven"] is False
    closure = json.loads((job_dir / "live_eval_closure_manifest.json").read_text(encoding="utf-8"))
    assert closure["post_training_data_package_handoff"][
        "post_training_package_export_ready"
    ] is True
    assert closure["post_training_data_package_handoff"]["customer_handoff_ready"] is False
    assert closure["proof_boundary"]["post_training_package_export_ready"] is True
    assert closure["proof_boundary"]["customer_handoff_ready"] is False
    archive_members = json.loads((job_dir / "archive_manifest.json").read_text(encoding="utf-8"))
    assert "customer_handoff_report.json" in archive_members["included_files"]
    assert "delivery_manifest.json" in archive_members["included_files"]
    assert "signed_access_manifest.json" in archive_members["included_files"]


def test_post_training_data_package_includes_visual_augmentation_support_packet(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    packet_dir = job_dir / "oscar_visual_augmentation_packet"
    _write_json(
        packet_dir / "oscar_visual_augmentation_packet_manifest.json",
        {
            "schema_version": "oscar_visual_augmentation_packet.v1",
            "status": "completed_with_model_derived_generated_videos",
            "packet_type": "oscar_visual_augmentation_packet",
            "variant_count": 2,
            "generated_video_count": 1,
            "selected_backend_id": "oscar_wam",
            "claim_boundary": {
                "generated_videos_are_model_derived_support_assets": True,
                "generated_videos_are_raw_capture_evidence": False,
                "contact_physics_proven": False,
                "real_robot_readiness_proven": False,
                "deployment_safety_proven": False,
            },
        },
    )
    (packet_dir / "visual_augmentation_variant_requests.jsonl").write_text(
        '{"variant_id":"kitchen"}\n',
        encoding="utf-8",
    )
    _write_json(packet_dir / "model_backend_registry.json", {"backends": []})
    _write_json(packet_dir / "visual_distribution_shift_eval_protocol.json", {"status": "ready"})
    _write_json(packet_dir / "claim_boundary.json", {"model_derived_visual_augmentation": True})
    _write_json(
        packet_dir / "visual_augmentation_generation_run_manifest.json",
        {"status": "completed_with_model_derived_outputs"},
    )
    (packet_dir / "visual_augmentation_generation_results.jsonl").write_text(
        '{"variant_id":"kitchen","model_derived":true}\n',
        encoding="utf-8",
    )
    _write_json(
        packet_dir / "visual_augmentation_generation_qa_manifest.json",
        {"status": "passed_visual_qa_smoke"},
    )
    _write_json(
        packet_dir / "visual_augmentation_training_readiness_manifest.json",
        {"training_ready_without_review": False},
    )
    _write_json(
        packet_dir / "visual_augmentation_training_dataset_manifest.json",
        {"status": "candidate_dataset_written_requires_review"},
    )
    (packet_dir / "exports" / "visual_augmentation" / "episodes.jsonl").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (packet_dir / "exports" / "visual_augmentation" / "episodes.jsonl").write_text(
        '{"variant_id":"kitchen","use_status":"candidate_requires_review"}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "package"
    monkeypatch.setattr(package_module, "_write_native_hdf5", lambda path, rows: False)
    monkeypatch.setattr(package_module, "_write_native_parquet", lambda path, rows: False)

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    assert manifest["status"] == "export_ready_review_required"
    assert (
        manifest["included_artifacts"]["oscar_visual_augmentation_packet_manifest"]
        == "oscar_visual_augmentation_packet/oscar_visual_augmentation_packet_manifest.json"
    )
    assert manifest["included_artifacts"]["oscar_visual_augmentation_generation_run_manifest"] == (
        "oscar_visual_augmentation_packet/visual_augmentation_generation_run_manifest.json"
    )
    assert manifest["included_artifacts"]["oscar_visual_augmentation_training_episodes"] == (
        "oscar_visual_augmentation_packet/exports/visual_augmentation/episodes.jsonl"
    )
    assert manifest["export_policy"]["visual_augmentation_packet_included"] is True
    assert manifest["export_policy"]["visual_augmentation_is_model_derived_support"] is True
    assert (
        manifest["export_policy"]["visual_augmentation_generated_videos_are_raw_capture_evidence"]
        is False
    )
    assert manifest["manifest_counts"]["visual_augmentation_variant_count"] == 2
    assert manifest["visual_augmentation_support_manifest_path"] == (
        "visual_augmentation_support_manifest.json"
    )
    support = json.loads(
        (output_dir / "visual_augmentation_support_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert support["generated_videos_model_derived"] is True
    assert support["raw_capture_evidence"] is False
    assert support["claim_boundary"]["contact_physics_proven"] is False
    assert support["claim_boundary"]["real_robot_readiness_proven"] is False
    assert support["claim_boundary"]["deployment_safety_proven"] is False
    package_index = json.loads((output_dir / "package_index.json").read_text(encoding="utf-8"))
    assert package_index["files"]["visual_augmentation_support_manifest"] == (
        "visual_augmentation_support_manifest.json"
    )
    archive_members = json.loads((output_dir / "archive_manifest.json").read_text(encoding="utf-8"))
    assert "visual_augmentation_support_manifest.json" in archive_members["included_files"]


def test_post_training_data_package_main_returns_status_codes(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    capture_root = _capture_root(tmp_path)
    blocked_output = tmp_path / "blocked"

    assert main(["--capture-root", str(capture_root), "--output-dir", str(blocked_output)]) == 1
    assert "status=blocked_missing_inputs" in capsys.readouterr().out

    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    ready_output = tmp_path / "ready"
    monkeypatch.setattr(package_module, "_write_native_hdf5", lambda path, rows: False)
    monkeypatch.setattr(package_module, "_write_native_parquet", lambda path, rows: False)

    assert (
        main(
            [
                "--capture-root",
                str(capture_root),
                "--job-dir",
                str(job_dir),
                "--output-dir",
                str(ready_output),
            ]
        )
        == 0
    )
    assert "status=export_ready_review_required" in capsys.readouterr().out


def test_post_training_data_package_private_helpers_cover_optional_edges(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assert _read_optional_mapping(tmp_path / "missing.json") == {}
    non_mapping = tmp_path / "list.json"
    non_mapping.write_text("[]", encoding="utf-8")
    assert _read_optional_mapping(non_mapping) == {}
    assert _rows({"rows": [{"ok": True}, "skip"]}, "rows") == [{"ok": True}]
    assert _rows({"rows": "not-a-list"}, "rows") == []

    missing_artifact = _artifact(tmp_path, tmp_path / "missing.file")
    assert missing_artifact["exists"] is False
    assert missing_artifact["sha256"] is None

    monkeypatch.setitem(sys.modules, "h5py", None)
    assert _write_native_hdf5(tmp_path / "fallback" / "episodes.hdf5", []) is False

    class _FakeH5File:
        def __init__(self, path: Path, mode: str) -> None:
            self.path = path
            self.mode = mode
            self.attrs: dict[str, object] = {}

        def __enter__(self) -> "_FakeH5File":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("fake", encoding="utf-8")
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def create_dataset(self, _name: str, *, data: object, dtype: object) -> None:
            self.attrs["dataset"] = {"data": data, "dtype": dtype}

    monkeypatch.setitem(
        sys.modules,
        "h5py",
        types.SimpleNamespace(
            File=_FakeH5File,
            string_dtype=lambda encoding: f"string:{encoding}",
        ),
    )
    assert _write_native_hdf5(tmp_path / "native" / "episodes.hdf5", [{"episode": 1}]) is True

    monkeypatch.setattr(package_module.importlib.util, "find_spec", lambda _name: None)
    assert _write_native_parquet(tmp_path / "fallback" / "episodes.parquet", []) is False

    class _FakeDataFrame:
        def __init__(self, rows: object) -> None:
            self.rows = rows

        def to_parquet(self, path: Path, *, index: bool) -> None:
            assert index is False
            path.write_text(json.dumps(self.rows), encoding="utf-8")

    monkeypatch.setattr(package_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setitem(
        sys.modules,
        "pandas",
        types.SimpleNamespace(DataFrame=lambda rows: _FakeDataFrame(rows)),
    )
    assert _write_native_parquet(tmp_path / "native" / "episodes.parquet", [{"episode_id": "e"}]) is True


def test_dataset_card_surfaces_clip_curation_state(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)

    # No curation/dedup manifests -> explicit not_run QA state.
    build_post_training_data_package_export(capture_root=capture_root)
    output_dir = capture_root / "pipeline" / "post_training_data_package"
    dataset_card = json.loads((output_dir / "dataset_card.json").read_text(encoding="utf-8"))
    assert dataset_card["clip_curation"]["curation_status"] == "not_run"
    assert dataset_card["clip_curation"]["dedup_status"] == "not_run"

    # With manifests present -> counts and provider provenance carried through.
    curation_dir = capture_root / "derived" / "clip_curation"
    dedup_dir = capture_root / "derived" / "semantic_dedup"
    curation_dir.mkdir(parents=True)
    dedup_dir.mkdir(parents=True)
    (curation_dir / "clip_curation_manifest.json").write_text(
        json.dumps({"accepted_clip_count": 3, "rejected_clip_count": 2}), encoding="utf-8"
    )
    (curation_dir / "clip_rejection_manifest.json").write_text(
        json.dumps({"rejected_count": 2}), encoding="utf-8"
    )
    (dedup_dir / "semantic_dedup_manifest.json").write_text(
        json.dumps(
            {
                "coverage": {"kept_clip_count": 2, "dropped_clip_count": 1},
                "embedding_provider": {"name": "downsampled-pixel", "version": "1"},
            }
        ),
        encoding="utf-8",
    )
    manifest = build_post_training_data_package_export(capture_root=capture_root)
    dataset_card = json.loads((output_dir / "dataset_card.json").read_text(encoding="utf-8"))
    curation = dataset_card["clip_curation"]
    assert curation["curation_status"] == "run"
    assert curation["accepted_clip_count"] == 3
    assert curation["rejected_clip_count"] == 2
    assert curation["post_dedup_clip_count"] == 2
    assert curation["embedding_provider"]["name"] == "downsampled-pixel"
    assert "clip_curation_manifest" in manifest["included_artifacts"]
    assert "semantic_dedup_manifest" in manifest["included_artifacts"]
