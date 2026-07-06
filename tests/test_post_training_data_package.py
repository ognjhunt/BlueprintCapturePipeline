from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from tests.video_codec import require_video_codec_or_skip

import blueprint_pipeline.post_training_data_package as package_module
from blueprint_pipeline.scaniverse_asset_import import build_scaniverse_asset_import
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


def _write_valid_mp4_or_placeholder(path: Path, *, frame_count: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        path.write_bytes(b"fake-mp4")
        return
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 32))
    if not writer.isOpened():
        require_video_codec_or_skip("cv2 mp4 writer unavailable")
    try:
        for index in range(frame_count):
            writer.write(np.full((32, 32, 3), 40 + index, dtype=np.uint8))
    finally:
        writer.release()


def _write_ascii_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "1 0 0",
                "1 1 0.1",
                "0 1 0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


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
                    "observation_refs": [
                        {
                            "frame": "000001",
                            "state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                        }
                    ],
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
    clips_manifest_path = job_dir / "clips_manifest.json"
    clips_manifest = json.loads(clips_manifest_path.read_text(encoding="utf-8"))
    clips_manifest["clips"][0].update(
        {
            "consent_revoked": False,
            "delivery_blocked_by_consent_revocation": False,
            "signed_access_revoked_by_consent": False,
            "manual_rights_review_recommended": False,
            "commercial_use_claim_allowed": True,
            "external_licensing_claim_allowed": True,
        }
    )
    _write_json(clips_manifest_path, clips_manifest)
    (job_dir / "clip-1.mp4").write_bytes(b"fake-mp4")
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


def test_post_training_data_package_materializes_lerobot_v3_and_gr00t_exports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    clips_manifest_path = job_dir / "clips_manifest.json"
    clips_manifest = json.loads(clips_manifest_path.read_text(encoding="utf-8"))
    clips_manifest["clips"][0].update(
        {
            "observation_source": "raw_capture",
            "consent_scope": ["robot_evaluation", "model_training"],
            "redaction_status": "face_redacted",
            "license_status": "documented",
            "consent_revoked": "false",
            "commercial_use_claim_allowed": "false",
            "external_licensing_claim_allowed": "false",
        }
    )
    _write_json(clips_manifest_path, clips_manifest)
    _write_valid_mp4_or_placeholder(job_dir / "clip-1.mp4")
    output_dir = tmp_path / "package"

    def _write_fake_structured_parquet(path: Path, rows: object) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"row_count": len(list(rows))}), encoding="utf-8")
        return True

    monkeypatch.setattr(
        package_module,
        "_write_structured_parquet",
        _write_fake_structured_parquet,
    )
    monkeypatch.setattr(
        package_module,
        "_write_lerobot_tasks_parquet",
        _write_fake_structured_parquet,
    )

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    optional = manifest["optional_exports"]
    video_bundle = optional["formats"]["video_bundle"]
    assert video_bundle["status"] == "written_materialized"
    assert video_bundle["materialized_clip_count"] == 1
    assert video_bundle["missing_clip_file_count"] == 0
    assert video_bundle["all_declared_clips_materialized"] is True
    video_bundle_manifest = json.loads(
        (output_dir / "exports" / "video_bundle" / "clips_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    clip_row = video_bundle_manifest["clips"][0]
    assert clip_row["metadata_sidecar_path"].endswith(".mp4.metadata.json")
    assert clip_row["metadata_sidecar_schema_version"] == (
        "post_training_clip_metadata_sidecar.v1"
    )
    clip_sidecar = json.loads(
        (output_dir / clip_row["metadata_sidecar_path"]).read_text(encoding="utf-8")
    )
    assert clip_sidecar["clip_id"] == "clip-1"
    assert clip_sidecar["consent_scope"] == ["robot_evaluation", "model_training"]
    assert clip_sidecar["license_status"] == "documented"
    assert clip_sidecar["redaction_status"] == "face_redacted"
    assert clip_sidecar["rights_metadata"]["clip_metadata_source"] == (
        "clip_manifest_top_level_fields"
    )
    assert clip_sidecar["rights_metadata"]["license_status"] == "documented"
    assert clip_sidecar["consent_revoked"] is False
    assert clip_sidecar["commercial_use_claim_allowed"] is False
    assert clip_sidecar["external_licensing_claim_allowed"] is False
    assert clip_sidecar["claim_boundary"]["standalone_clip_requires_sidecar_review"] is True

    lerobot_v3 = optional["formats"]["lerobot_v3"]
    assert lerobot_v3["status"] == "written_native"
    assert lerobot_v3["native_parquet_written"] is True
    assert lerobot_v3["consumer_layout_complete"] is True
    lerobot_info = json.loads(
        (
            output_dir / "exports" / "lerobot_v3" / "meta" / "info.json"
        ).read_text(encoding="utf-8")
    )
    assert "observation_source" in lerobot_info["features"]
    assert "source_rights_metadata_json" in lerobot_info["features"]
    lerobot_stats = json.loads(
        (
            output_dir / "exports" / "lerobot_v3" / "meta" / "stats.json"
        ).read_text(encoding="utf-8")
    )
    assert lerobot_stats["raw_capture_frame_rows"] == 1
    assert lerobot_stats["rights_metadata_frame_rows"] == 1
    assert (
        output_dir
        / "exports"
        / "lerobot_v3"
        / "videos"
        / "observation.images.ego_view"
        / "chunk-000"
        / "file-000.mp4"
    ).is_file()

    gr00t = optional["formats"]["gr00t_lerobot"]
    assert gr00t["status"] == "written_native"
    assert gr00t["consumer_layout_complete"] is True
    modality = json.loads(
        (
            output_dir / "exports" / "gr00t_lerobot" / "meta" / "modality.json"
        ).read_text(encoding="utf-8")
    )
    assert modality["video"]["ego_view"]["original_key"] == "observation.images.ego_view"
    assert modality["metadata"]["observation_source"]["original_key"] == (
        "observation_source"
    )
    assert modality["metadata"]["source_rights_metadata_json"]["original_key"] == (
        "source_rights_metadata_json"
    )
    gr00t_stats = json.loads(
        (
            output_dir / "exports" / "gr00t_lerobot" / "meta" / "stats.json"
        ).read_text(encoding="utf-8")
    )
    assert gr00t_stats["raw_capture_frame_rows"] == 1
    episodes = [
        json.loads(line)
        for line in (
            output_dir / "exports" / "gr00t_lerobot" / "meta" / "episodes.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert episodes[0]["observation_source"] == "raw_capture"
    assert episodes[0]["source_rights_metadata"]["consent_scope"] == [
        "robot_evaluation",
        "model_training",
    ]
    assert episodes[0]["source_rights_metadata"]["commercial_use_claim_allowed"] is False
    assert (
        episodes[0]["source_rights_metadata"]["external_licensing_claim_allowed"]
        is False
    )

    package_index = json.loads((output_dir / "package_index.json").read_text(encoding="utf-8"))
    indexed_paths = set(package_index["files"].values())
    assert any(path.startswith("exports/video_bundle/clips/") for path in indexed_paths)
    assert clip_row["metadata_sidecar_path"] in indexed_paths
    assert "exports/gr00t_lerobot/meta/modality.json" in indexed_paths
    archive_members = json.loads((output_dir / "archive_manifest.json").read_text(encoding="utf-8"))
    assert clip_row["metadata_sidecar_path"] in archive_members["included_files"]

    # Fully-measured package: real-data fractions are 1.0 and the floor passes.
    for entry in (lerobot_v3, gr00t):
        provenance = entry["state_action_provenance"]
        assert provenance["real_state_fraction"] == 1.0
        assert provenance["real_action_fraction"] == 1.0
        assert provenance["measured_state_fraction_floor_passed"] is True
        assert "insufficient_measured_state_fraction" not in entry["blockers"]
    assert manifest["export_policy"]["lerobot_real_state_fraction"] == 1.0
    assert manifest["export_policy"]["lerobot_real_action_fraction"] == 1.0
    assert manifest["export_policy"]["measured_state_fraction_floor_passed"] is True
    assert manifest["claim_boundary"]["measured_state_fraction_floor_passed"] is True

    readout = json.loads((output_dir / "buyer_package_readout.json").read_text(encoding="utf-8"))
    pov = readout["sections"]["robot_pov_evidence"]
    assert pov["status"] == "present"
    assert pov["materialized_clip_count"] == 1
    assert pov["gr00t_lerobot_consumer_layout_complete"] is True
    assert pov["measured_state_fractions"]["lerobot_v3"] == 1.0
    assert pov["measured_state_fractions"]["gr00t_lerobot"] == 1.0


def test_video_bundle_sidecar_does_not_truthify_string_false_metadata(
    tmp_path: Path,
) -> None:
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video-bytes")
    output_dir = tmp_path / "package"

    result = package_module._materialize_video_bundle(
        output_dir=output_dir,
        clips={
            "clip_count": 1,
            "clips": [
                {
                    "clip_id": "clip-1",
                    "clip_path": "source.mp4",
                    "attempt_id": "attempt-1",
                    "observation_source_is_model_derived": "false",
                    "observation_source_is_raw_capture_evidence": "false",
                    "fallback_redaction_used": "false",
                    "manual_rights_review_recommended": "false",
                    "rights_metadata": {
                        "consent_revoked": "false",
                        "fallback_redaction_used": "false",
                        "manual_rights_review_recommended": "false",
                    },
                }
            ],
        },
        generated_at="2026-07-04T12:00:00Z",
        source_roots=[tmp_path],
    )

    clip = result["materialized_clips"][0]
    sidecar = json.loads(
        (output_dir / clip["metadata_sidecar_path"]).read_text(encoding="utf-8")
    )
    assert sidecar["observation_source_is_model_derived"] is False
    assert sidecar["observation_source_is_raw_capture_evidence"] is False
    assert sidecar["consent_revoked"] is False
    assert sidecar["fallback_redaction_used"] is False
    assert sidecar["manual_rights_review_recommended"] is False


def test_training_export_rows_preserve_rights_claim_boundary_metadata() -> None:
    frame_rows, episodes, _tasks, shape = package_module._training_export_rows(
        rows=[
            {
                "attempt_id": "attempt-1",
                "task_id": "task-1",
                "actions": [[0.05, 0.0, 0.01, 0.0, 0.0, 0.1, 0.0]],
                "observation.state": [0.0, 0.0, 0.79],
                "success": True,
            }
        ],
        materialized_clips=[
            {
                "attempt_id": "attempt-1",
                "clip_id": "clip-1",
                "materialized": True,
                "materialized_path": "exports/video_bundle/clips/clip-1.mp4",
                "rights_metadata": {
                    "metadata_source": "package_consent_evidence",
                    "license_status": "blocked_consent_revoked_takedown_required",
                    "consent_scope": ["robot_evaluation", "model_training"],
                    "consent_revoked": True,
                    "consent_revoked_at": "2026-07-04T12:00:00Z",
                    "delivery_blocked_by_consent_revocation": True,
                    "signed_access_revoked_by_consent": True,
                    "commercial_use_claim_allowed": "false",
                    "external_licensing_claim_allowed": False,
                    "manual_rights_review_recommended": True,
                },
            }
        ],
    )

    assert shape["rights_metadata_frame_rows"] == 1
    frame_rights = json.loads(frame_rows[0]["source_rights_metadata_json"])
    assert frame_rights["metadata_source"] == "package_consent_evidence"
    assert frame_rights["consent_revoked"] is True
    assert frame_rights["consent_revoked_at"] == "2026-07-04T12:00:00Z"
    assert frame_rights["delivery_blocked_by_consent_revocation"] is True
    assert frame_rights["signed_access_revoked_by_consent"] is True
    assert frame_rights["commercial_use_claim_allowed"] is False
    assert frame_rights["external_licensing_claim_allowed"] is False
    assert episodes[0]["source_rights_metadata"] == frame_rights


def test_training_export_rows_report_state_action_provenance() -> None:
    frame_rows, episodes, _tasks, shape = package_module._training_export_rows(
        rows=[
            {
                "attempt_id": "attempt-measured",
                "task_id": "task-1",
                "actions": [[0.05, 0.0, 0.01], [0.02, 0.0, 0.0]],
                "observation.state": [0.1, 0.2, 0.3],
                "success": True,
            },
            {
                # No actions and no measured state: the export synthesizes a
                # fallback action row and a zero-filled state vector.
                "attempt_id": "attempt-synthesized",
                "task_id": "task-1",
                "success": False,
            },
        ],
        materialized_clips=[],
    )

    assert shape["measured_state_rows"] == 2
    assert shape["synthesized_state_rows"] == 1
    assert shape["measured_action_rows"] == 2
    assert shape["synthesized_action_rows"] == 1
    assert shape["real_state_fraction"] == pytest.approx(2 / 3)
    assert shape["real_action_fraction"] == pytest.approx(2 / 3)

    measured = episodes[0]["state_action_provenance"]
    assert measured["measured_state_rows"] == 2
    assert measured["synthesized_state_rows"] == 0
    assert measured["measured_action_rows"] == 2
    assert measured["synthesized_action_rows"] == 0

    synthesized = episodes[1]["state_action_provenance"]
    assert synthesized["measured_state_rows"] == 0
    assert synthesized["synthesized_state_rows"] == 1
    assert synthesized["measured_action_rows"] == 0
    assert synthesized["synthesized_action_rows"] == 1

    assert frame_rows[0]["action_synthesized_fallback"] is False
    assert frame_rows[2]["action_synthesized_fallback"] is True
    assert frame_rows[2]["state_synthesized_zero_fill"] is True


def test_mostly_synthesized_state_package_is_downgraded_with_fraction_surfaced(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    # Strip the measured state: every exported observation.state row is now a
    # synthesized zero fill.
    trace_path = job_dir / "normalized_attempt_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    for attempt in trace["attempts"]:
        for ref in attempt["observation_refs"]:
            ref.pop("state", None)
    _write_json(trace_path, trace)
    _write_valid_mp4_or_placeholder(job_dir / "clip-1.mp4")
    output_dir = tmp_path / "package"

    def _write_fake_structured_parquet(path: Path, rows: object) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"row_count": len(list(rows))}), encoding="utf-8")
        return True

    monkeypatch.setattr(
        package_module,
        "_write_structured_parquet",
        _write_fake_structured_parquet,
    )
    monkeypatch.setattr(
        package_module,
        "_write_lerobot_tasks_parquet",
        _write_fake_structured_parquet,
    )

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    for format_name in ("lerobot_v3", "gr00t_lerobot"):
        entry = manifest["optional_exports"]["formats"][format_name]
        assert entry["status"] == "written_degraded"
        assert "insufficient_measured_state_fraction" in entry["blockers"]
        provenance = entry["state_action_provenance"]
        assert provenance["real_state_fraction"] == 0.0
        assert provenance["real_action_fraction"] == 1.0
        assert provenance["measured_state_fraction_floor"] == pytest.approx(0.5)
        assert provenance["measured_state_fraction_floor_passed"] is False

    export_manifest = json.loads(
        (
            output_dir / "exports" / "lerobot_v3" / "lerobot_v3_export_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert "insufficient_measured_state_fraction" in export_manifest["blockers"]
    per_episode = export_manifest["state_action_provenance"]["per_episode"]
    assert per_episode[0]["attempt_id"] == "attempt-1"
    assert per_episode[0]["measured_state_rows"] == 0
    assert per_episode[0]["synthesized_state_rows"] == 1

    assert manifest["export_policy"]["lerobot_real_state_fraction"] == 0.0
    assert manifest["export_policy"]["measured_state_fraction_floor_passed"] is False
    assert manifest["claim_boundary"]["measured_state_fraction_floor_passed"] is False

    readout = json.loads(
        (output_dir / "buyer_package_readout.json").read_text(encoding="utf-8")
    )
    assert readout["status"] == "blocked_incomplete_package"
    pov = readout["sections"]["robot_pov_evidence"]
    assert pov["status"] == "missing"
    assert "insufficient_measured_state_fraction:lerobot_v3" in pov["blockers"]
    assert "insufficient_measured_state_fraction:gr00t_lerobot" in pov["blockers"]
    assert pov["measured_state_fractions"]["lerobot_v3"] == 0.0


def test_measured_state_fraction_floor_is_env_configurable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BLUEPRINT_PTDP_MEASURED_STATE_FRACTION_FLOOR", "0.0")
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    trace_path = job_dir / "normalized_attempt_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    for attempt in trace["attempts"]:
        for ref in attempt["observation_refs"]:
            ref.pop("state", None)
    _write_json(trace_path, trace)
    (job_dir / "clip-1.mp4").write_bytes(b"fake-mp4")

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=tmp_path / "package",
    )

    lerobot_v3 = manifest["optional_exports"]["formats"]["lerobot_v3"]
    provenance = lerobot_v3["state_action_provenance"]
    assert provenance["measured_state_fraction_floor"] == 0.0
    assert provenance["real_state_fraction"] == 0.0
    assert provenance["measured_state_fraction_floor_passed"] is True
    assert "insufficient_measured_state_fraction" not in lerobot_v3["blockers"]


def test_lerobot_and_gr00t_exports_require_video_for_every_episode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "attempt_count": 2,
            "attempts": [
                {
                    "attempt_id": "attempt-1",
                    "scenario_id": "scenario-1",
                    "task_id": "task-1",
                    "policy_id": "policy-1",
                    "success": True,
                    "status": "passed",
                    "metrics": {"score": 1.0},
                    "action_trace": [{"sc3_7d_delta_ee_pose": [0, 0, 0, 0, 0, 0, 1]}],
                    "observation_refs": [{"frame": "000001"}],
                },
                {
                    "attempt_id": "attempt-2",
                    "scenario_id": "scenario-2",
                    "task_id": "task-2",
                    "policy_id": "policy-1",
                    "success": True,
                    "status": "passed",
                    "metrics": {"score": 1.0},
                    "action_trace": [{"sc3_7d_delta_ee_pose": [0, 0, 0, 0, 0, 0, 1]}],
                    "observation_refs": [{"frame": "000002"}],
                },
            ],
        },
    )
    _write_json(
        job_dir / "clips_manifest.json",
        {
            "clip_count": 2,
            "clips": [
                {
                    "clip_id": "clip-1",
                    "clip_path": "missing-clip-1.mp4",
                    "attempt_id": "attempt-1",
                    "frame_count": 1,
                    "camera_motion_m": 0.0,
                    "visible_skeleton_fraction": 0.9,
                    "sharpness_score": 72.0,
                    "semantic_dedup_key": "scene-1|task-1|attempt-1",
                },
                {
                    "clip_id": "clip-2",
                    "clip_path": "clip-2.mp4",
                    "attempt_id": "attempt-2",
                    "frame_count": 1,
                    "camera_motion_m": 0.0,
                    "visible_skeleton_fraction": 0.9,
                    "sharpness_score": 72.0,
                    "semantic_dedup_key": "scene-1|task-2|attempt-2",
                },
            ],
        },
    )
    _write_valid_mp4_or_placeholder(job_dir / "clip-2.mp4")

    def _write_fake_structured_parquet(path: Path, rows: object) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"row_count": len(list(rows))}), encoding="utf-8")
        return True

    monkeypatch.setattr(
        package_module,
        "_write_structured_parquet",
        _write_fake_structured_parquet,
    )
    monkeypatch.setattr(
        package_module,
        "_write_lerobot_tasks_parquet",
        _write_fake_structured_parquet,
    )

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=tmp_path / "package",
    )

    optional = manifest["optional_exports"]["formats"]
    assert optional["video_bundle"]["materialized_clip_count"] == 1
    assert optional["video_bundle"]["missing_clip_file_count"] == 1

    lerobot_v3 = optional["lerobot_v3"]
    assert lerobot_v3["consumer_layout_complete"] is False
    assert lerobot_v3["missing_video_episode_count"] == 1
    assert lerobot_v3["all_episode_videos_materialized"] is False
    assert "lerobot_v3_video_files_missing" in lerobot_v3["blockers"]

    gr00t = optional["gr00t_lerobot"]
    assert gr00t["consumer_layout_complete"] is False
    assert gr00t["missing_video_episode_count"] == 1
    assert gr00t["all_episode_videos_materialized"] is False
    assert "gr00t_lerobot_video_files_missing" in gr00t["blockers"]

    gr00t_videos = tmp_path / "package" / "exports" / "gr00t_lerobot" / "videos"
    assert (
        gr00t_videos
        / "chunk-000"
        / "observation.images.ego_view"
        / "episode_000001.mp4"
    ).is_file()
    assert not (
        gr00t_videos
        / "chunk-000"
        / "observation.images.ego_view"
        / "episode_000000.mp4"
    ).exists()


def test_post_training_package_runs_lerobot_round_trip_validation(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    _write_valid_mp4_or_placeholder(job_dir / "clip-1.mp4")
    output_dir = tmp_path / "package"

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    formats = manifest["optional_exports"]["formats"]
    for format_name in ("lerobot_v3", "gr00t_lerobot"):
        validation = formats[format_name]["round_trip_validation"]
        assert validation["status"] == "passed", (format_name, validation["blockers"])
        assert validation["path"] == (
            f"exports/{format_name}/round_trip_validation_report.json"
        )
        report_path = output_dir / "exports" / format_name / (
            "round_trip_validation_report.json"
        )
        assert report_path.is_file()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["status"] == "passed"
        assert (
            report["claim_boundary"][
                "validation_passed_is_not_data_quality_or_success_claim"
            ]
            is True
        )
    package_index = json.loads(
        (output_dir / "package_index.json").read_text(encoding="utf-8")
    )
    assert "exports/lerobot_v3/round_trip_validation_report.json" in set(
        package_index["files"].values()
    )

    readout = json.loads(
        (output_dir / "buyer_package_readout.json").read_text(encoding="utf-8")
    )
    integrity = readout["sections"]["export_integrity"]
    assert integrity["lerobot_round_trip_validation"] == {
        "lerobot_v3": "passed",
        "gr00t_lerobot": "passed",
    }
    assert not any(
        blocker.startswith("export_integrity:lerobot") for blocker in readout["blockers"]
    )


def test_misaligned_lerobot_export_downgrades_buyer_readout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    (job_dir / "clip-1.mp4").write_bytes(b"fake-mp4")
    output_dir = tmp_path / "package"

    original_write_jsonl = package_module._write_jsonl

    def _corrupt_lerobot_v3_episodes(path: Path, rows: object) -> None:
        payload = [dict(row) for row in rows]  # type: ignore[union-attr]
        if "lerobot_v3" in str(path) and "meta/episodes" in path.as_posix():
            for row in payload:
                if "length" in row:
                    # Lie about the episode length: video/parquet alignment is
                    # now off and the round trip must catch it.
                    row["length"] = 999
        original_write_jsonl(path, payload)

    monkeypatch.setattr(package_module, "_write_jsonl", _corrupt_lerobot_v3_episodes)

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    validation = manifest["optional_exports"]["formats"]["lerobot_v3"][
        "round_trip_validation"
    ]
    readout = json.loads(
        (output_dir / "buyer_package_readout.json").read_text(encoding="utf-8")
    )
    if validation["status"] == "passed":
        # Native parquet was written (pyarrow installed) so the jsonl corruption
        # never reached the validated rows; the gate stays green.
        assert validation["loader"] != "hermetic_jsonl_fallback"
    else:
        assert readout["status"] == "blocked_incomplete_package"
        assert (
            "export_integrity:lerobot_export_not_loadable:lerobot_v3"
            in readout["blockers"]
        )


def test_lerobot_v3_export_loads_with_installed_lerobot(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    LeRobotDataset = lerobot_dataset.LeRobotDataset
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    clip_path = job_dir / "clip-1.mp4"
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 32))
    if not writer.isOpened():
        require_video_codec_or_skip("cv2 mp4 writer unavailable")
    for index in range(3):
        writer.write(np.full((32, 32, 3), 40 + index * 40, dtype=np.uint8))
    writer.release()
    output_dir = tmp_path / "package"

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    lerobot_v3 = manifest["optional_exports"]["formats"]["lerobot_v3"]
    assert lerobot_v3["status"] == "written_native"
    assert lerobot_v3["consumer_layout_complete"] is True
    dataset = LeRobotDataset(
        repo_id="blueprint/local-test",
        root=output_dir / "exports" / "lerobot_v3",
        download_videos=False,
    )
    assert len(dataset) == 1
    assert dataset.meta.total_frames == 1
    assert dataset.meta.get_video_file_path(
        0,
        "observation.images.ego_view",
    ) == Path("videos/observation.images.ego_view/chunk-000/file-000.mp4")
    assert list(dataset.meta.tasks.index) == ["task-1"]


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


def test_post_training_data_package_rejects_boolean_only_curation_pass(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    _write_json(
        job_dir / "clips_manifest.json",
        {
            "clip_count": 1,
            "clips": [
                {
                    "clip_id": "clip-1",
                    "clip_path": "clip-1.mp4",
                    "attempt_id": "attempt-1",
                    "min_frame_filter_passed": True,
                    "static_camera_filter_passed": True,
                    "visible_skeleton_filter_passed": True,
                    "blur_filter_passed": True,
                    "semantic_dedup_key": "scene-1|task-1|attempt-1",
                }
            ],
        },
    )
    output_dir = tmp_path / "package"

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    assert manifest["status"] == "blocked_package_quality_gates"
    assert "curation:clip-1:min_frame_count_missing" in manifest["blockers"]
    assert any(
        blocker.startswith("curation:clip-1:min_frame_count_missing:")
        for blocker in manifest["blockers"]
    )
    curation = json.loads((output_dir / "curation_report.json").read_text(encoding="utf-8"))
    assert curation["status"] == "blocked"
    frame_evidence = curation["clips"][0]["gates"]["min_frame"]
    assert frame_evidence["explicit"] is True
    assert frame_evidence["explicit_boolean_is_not_measured_evidence"] is True


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


def test_post_training_data_package_labels_scaniverse_assets_as_support_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    export_dir = tmp_path / "scaniverse_exports"
    ply = export_dir / "pilot_splat.ply"
    usdz = export_dir / "pilot_scene.usdz"
    sidecar = export_dir / "blueprint_scaniverse_sidecar.json"
    _write_ascii_ply(ply)
    usdz.write_bytes(b"USDZ placeholder")
    _write_json(
        sidecar,
        {
            "blueprint_assignment_id": "assignment-1",
            "blueprint_scene_id": "scene-1",
            "blueprint_capture_id": "capture-1",
            "scaniverse_site_id": "scaniverse-site-1",
            "scaniverse_scan_id": "scaniverse-scan-1",
            "capture_hardware": "Insta360 X5",
            "source_video_filename": "pilot.insv",
            "metric_scale_calibrated": False,
            "export_created_at": "2026-07-06T12:00:00Z",
            "export_performed_by": "operator@example.test",
            "rights_scope": "pilot_review_only",
        },
    )
    import_result = build_scaniverse_asset_import(
        capture_root=capture_root,
        assets=[ply, usdz],
        source_manifest=sidecar,
    )
    assert import_result["status"] == "ready_for_review"

    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    output_dir = tmp_path / "package"
    monkeypatch.setattr(package_module, "_write_native_hdf5", lambda path, rows: False)
    monkeypatch.setattr(package_module, "_write_native_parquet", lambda path, rows: False)

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    assert manifest["status"] == "export_ready_review_required"
    assert manifest["included_artifacts"]["scaniverse_import_manifest"].endswith(
        "scaniverse_import_manifest.json"
    )
    assert manifest["export_policy"]["scaniverse_support_assets_included"] is True
    assert manifest["export_policy"]["scaniverse_assets_are_external_derived_support"] is True
    assert manifest["export_policy"]["scaniverse_assets_are_raw_capture_evidence"] is False
    assert manifest["export_policy"]["scaniverse_assets_are_task_success_evidence"] is False
    assert manifest["export_policy"]["scaniverse_assets_are_physics_contact_evidence"] is False
    assert manifest["scaniverse_support_asset_manifest_path"] == (
        "scaniverse_support_asset_manifest.json"
    )
    support = json.loads(
        (output_dir / "scaniverse_support_asset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert support["external_derived_support_asset"] is True
    assert support["raw_capture_evidence"] is False
    assert support["claim_boundary"]["isaac_sim_execution_proven"] is False
    assert support["claim_boundary"]["physics_contact_validated"] is False
    assert support["claim_boundary"]["scaniverse_assets_are_task_success_evidence"] is False
    package_index = json.loads((output_dir / "package_index.json").read_text(encoding="utf-8"))
    assert package_index["files"]["scaniverse_support_asset_manifest"] == (
        "scaniverse_support_asset_manifest.json"
    )
    buyer_readout = json.loads((output_dir / "buyer_package_readout.json").read_text(encoding="utf-8"))
    derived = buyer_readout["sections"]["derived_support_assets"]
    assert derived["status"] == "present_support_only"
    assert derived["scaniverse_assets_are_raw_capture_evidence"] is False
    assert derived["scaniverse_assets_are_task_success_evidence"] is False
    assert derived["scaniverse_assets_are_physics_contact_evidence"] is False
    archive_members = json.loads((output_dir / "archive_manifest.json").read_text(encoding="utf-8"))
    assert "scaniverse_support_asset_manifest.json" in archive_members["included_files"]


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


def test_post_training_data_package_writes_buyer_readout_and_replay_instructions(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    output_dir = tmp_path / "package"

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    assert manifest["status"] == "export_ready_review_required"
    assert manifest["replay_review_instructions_path"] == "replay_review_instructions.md"
    assert manifest["buyer_package_readout_path"] == "buyer_package_readout.json"
    assert manifest["buyer_package_summary_path"] == "buyer_package_summary.md"

    instructions = (output_dir / "replay_review_instructions.md").read_text(encoding="utf-8")
    assert "checksums.json" in instructions
    assert "deployment approval" in instructions
    package_index = json.loads((output_dir / "package_index.json").read_text(encoding="utf-8"))
    assert package_index["files"]["replay_review_instructions"] == (
        "replay_review_instructions.md"
    )

    readout = json.loads((output_dir / "buyer_package_readout.json").read_text(encoding="utf-8"))
    assert readout["schema_version"] == "buyer_package_readout.v1"
    assert readout["status"] == manifest["buyer_readout_status"]
    # This fixture ships no robot POV evidence, so the buyer readout must fail closed
    # even though the pipeline export itself is ready for review.
    assert readout["status"] == "blocked_incomplete_package"
    assert "robot_pov_evidence:robot_pov_evidence_missing" in readout["blockers"]
    assert readout["claim_boundary"]["highest_truthful_claim"] == "no_claim"

    summary = (output_dir / "buyer_package_summary.md").read_text(encoding="utf-8")
    assert "Highest truthful claim: no_claim" in summary
    assert "not deployment approval" in summary


def test_post_training_data_package_wires_consent_handoff_and_success_ledger(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    _write_json(
        capture_root / "raw" / "rights_consent.json",
        {
            "consent_status": "documented",
            "consent_revoked": "false",
            "consent_scope": ["robot_evaluation", "model_training"],
            "permission_document_uri": "s3://blueprint-consent/site-a.pdf",
        },
    )
    _write_json(
        capture_root / "pipeline" / "robot_eval_dataset" / "rights_packet.json",
        {
            "status": "review_required",
            "consent_revoked": "false",
            "record_count": 1,
            "records": [
                {
                    "rights_scope": "model_training",
                    "evidence_uri": "s3://blueprint-consent/site-a.pdf",
                }
            ],
        },
    )
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    _write_json(
        job_dir / "success_claim_ledger.json",
        {
            "schema_version": "success_claim_ledger.v1",
            "highest_truthful_claim": "simulator_task_success",
            "claims": {"simulator_task_success": True},
            "blockers": [],
        },
    )
    _write_json(
        job_dir / "webapp_robot_eval_status_projection.json",
        {
            "product_handoff": {
                "product_type": "post_training_data_package_v1",
                "product_sku": "PTDP-001",
                "entitlement_id": "ent-42",
                "buyer_review_url": "https://webapp.example/review/ent-42",
            }
        },
    )
    output_dir = tmp_path / "package"

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    assert manifest["included_artifacts"]["consent_evidence"] == "consent_evidence.json"
    assert manifest["consent_evidence"]["consent_evidence_present"] is True
    assert manifest["consent_evidence"]["consent_revoked"] is False
    assert manifest["claim_boundary"]["consent_revocation_blocks_downstream_use"] is False
    assert manifest["handoff_records"]["local_package_access_revoked"] is False
    assert manifest["export_policy"]["consent_revoked"] is False
    assert manifest["revocation_takedown"]["consent_revoked"] is False
    assert manifest["status"] == "export_ready_review_required"
    assert manifest["success_claim_ledger_path"] == "success_claim_ledger.json"
    assert manifest["product_handoff"]["entitlement_id"] == "ent-42"

    package_index = json.loads((output_dir / "package_index.json").read_text(encoding="utf-8"))
    assert package_index["files"]["consent_evidence"] == "consent_evidence.json"
    assert package_index["files"]["success_claim_ledger"] == "success_claim_ledger.json"
    archive_members = json.loads((output_dir / "archive_manifest.json").read_text(encoding="utf-8"))
    assert "consent_evidence.json" in archive_members["included_files"]
    assert "success_claim_ledger.json" in archive_members["included_files"]

    readout = json.loads((output_dir / "buyer_package_readout.json").read_text(encoding="utf-8"))
    assert (
        readout["claim_boundary"]["highest_truthful_claim"]
        == "simulator_task_success"
    )
    assert readout["claim_boundary"]["consent_revocation_blocks_downstream_use"] is False
    assert (
        readout["sections"]["rights_privacy_provenance"][
            "consent_evidence_present"
        ]
        is True
    )
    assert (
        readout["sections"]["product_handoff"]["entitlement_wiring_present"]
        is True
    )


def test_post_training_data_package_string_true_consent_revocation_blocks(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    _write_json(
        capture_root / "raw" / "rights_consent.json",
        {
            "consent_status": "documented",
            "consent_revoked": "true",
            "consent_scope": ["robot_evaluation", "model_training"],
            "permission_document_uri": "s3://blueprint-consent/site-a.pdf",
        },
    )
    _write_json(
        capture_root / "pipeline" / "robot_eval_dataset" / "rights_packet.json",
        {
            "status": "review_required",
            "consent_revoked": "false",
            "record_count": 1,
            "records": [
                {
                    "rights_scope": "model_training",
                    "evidence_uri": "s3://blueprint-consent/site-a.pdf",
                }
            ],
        },
    )
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    output_dir = tmp_path / "package"

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    assert manifest["status"] == "blocked_consent_revoked_takedown_required"
    assert "consent:consent_revoked_takedown_required" in manifest["blockers"]
    assert manifest["consent_evidence"]["consent_revoked"] is True
    assert manifest["claim_boundary"]["consent_revocation_blocks_downstream_use"] is True
    assert manifest["handoff_records"]["local_package_access_revoked"] is True
    assert manifest["export_policy"]["consent_revoked"] is True
    assert manifest["revocation_takedown"]["consent_revoked"] is True
    assert manifest["revocation_takedown"]["status"] == "takedown_required"

    readout = json.loads(
        (output_dir / "buyer_package_readout.json").read_text(encoding="utf-8")
    )
    assert readout["status"] == "blocked_incomplete_package"
    assert (
        "revocation_takedown:consent_revoked_takedown_required"
        in readout["blockers"]
    )
    assert readout["claim_boundary"]["consent_revocation_blocks_downstream_use"] is True


def test_post_training_data_package_propagates_revenue_terms_without_payout_claim(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    _write_json(
        capture_root / "raw" / "rights_consent.json",
        {
            "consent_status": "documented",
            "consent_scope": ["robot_evaluation", "model_training"],
            "permission_document_uri": "s3://blueprint-consent/site-a.pdf",
        },
    )
    revenue_review = {
        "schema_version": "real_site_robot_eval_revenue_share_review.v1",
        "status": "recorded_review_required",
        "required_before_paid_reuse_or_resale": True,
        "owner_revenue_share_record_present": True,
        "operator_revenue_terms": {
            "terms_uri": "owner://terms/revenue-share",
            "operator_revenue_share_bps": 1500,
            "payee_entity_id": "operator-1",
        },
        "commercialization_terms": {
            "license_model": "request_scoped",
            "commercial_use_classes": ["robot_evaluation"],
        },
        "exclusivity_terms": {"exclusive": False},
        "revenue_share_commitment_made": False,
        "payout_commitment_allowed": False,
    }
    data_processing_terms = {
        "retention_policy": {
            "raw_capture_retention_days": 30,
            "package_artifact_retention_days": 90,
        },
        "subprocessors": [
            {"name": "storage-provider", "purpose": "artifact_storage"}
        ],
        "access_audit_terms": {
            "audit_log_required": True,
            "operator_access_review_interval_days": 30,
        },
    }
    _write_json(
        capture_root / "pipeline" / "robot_eval_dataset" / "rights_packet.json",
        {
            "schema_version": "real_site_robot_eval_rights_packet.v1",
            "status": "review_required",
            "record_count": 3,
            "records": [
                {
                    "rights_scope": "commercial_licensing",
                    "terms_record_present": True,
                    "commercialization_terms": revenue_review[
                        "commercialization_terms"
                    ],
                },
                {
                    "rights_scope": "revenue_share",
                    "terms_record_present": True,
                    "operator_revenue_terms": revenue_review[
                        "operator_revenue_terms"
                    ],
                },
                {
                    "rights_scope": "exclusivity_limits",
                    "terms_record_present": True,
                    "exclusivity_terms": revenue_review["exclusivity_terms"],
                },
            ],
            "revenue_share_review": revenue_review,
            "data_processing_terms": data_processing_terms,
            "commercial_use_claim_allowed": False,
            "external_licensing_claim_allowed": False,
        },
    )
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    output_dir = tmp_path / "package"

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    review = manifest["revenue_share_review"]
    assert review["status"] == "recorded_review_required"
    assert review["source_schema_version"] == (
        "real_site_robot_eval_revenue_share_review.v1"
    )
    assert review["owner_revenue_share_record_present"] is True
    assert review["operator_revenue_terms"]["operator_revenue_share_bps"] == 1500
    assert review["operator_revenue_terms"]["payee_entity_id"] == "operator-1"
    assert review["commercialization_terms"]["license_model"] == "request_scoped"
    assert review["exclusivity_terms"]["exclusive"] is False
    assert review["commercial_use_claim_allowed"] is False
    assert review["external_licensing_claim_allowed"] is False
    assert review["revenue_share_commitment_made"] is False
    assert review["payout_commitment_allowed"] is False
    assert review["blockers"] == []
    assert review["claim_boundary"][
        "operator_revenue_terms_are_review_metadata_not_payment_or_resale_clearance"
    ] is True

    revenue_artifact = json.loads(
        (output_dir / "revenue_share_review.json").read_text(encoding="utf-8")
    )
    assert revenue_artifact == review
    consent = json.loads((output_dir / "consent_evidence.json").read_text(encoding="utf-8"))
    assert consent["revenue_share_review"]["owner_revenue_share_record_present"] is True
    data_processing = manifest["data_processing_terms_review"]
    assert data_processing["status"] == "recorded_review_required"
    assert data_processing["retention_policy_present"] is True
    assert data_processing["subprocessor_list_present"] is True
    assert data_processing["access_audit_terms_present"] is True
    assert data_processing["dpa_approval_claimed"] is False
    assert data_processing["external_delivery_claim_allowed"] is False
    assert data_processing["blockers"] == []
    assert (
        data_processing["claim_boundary"][
            "data_processing_terms_are_review_metadata_not_legal_approval"
        ]
        is True
    )
    data_processing_artifact = json.loads(
        (output_dir / "data_processing_terms_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert data_processing_artifact == data_processing
    assert consent["data_processing_terms_review"]["retention_policy_present"] is True
    license_manifest = json.loads(
        (output_dir / "license_manifest.json").read_text(encoding="utf-8")
    )
    assert license_manifest["revenue_share_review"]["operator_revenue_terms"][
        "operator_revenue_share_bps"
    ] == 1500
    assert (
        license_manifest["data_processing_terms_review"][
            "access_audit_terms_present"
        ]
        is True
    )
    assert license_manifest["commercial_use_requires_package_scope_clearance"] is True
    package_index = json.loads((output_dir / "package_index.json").read_text(encoding="utf-8"))
    assert package_index["files"]["revenue_share_review"] == "revenue_share_review.json"
    assert package_index["files"]["data_processing_terms_review"] == (
        "data_processing_terms_review.json"
    )

    readout = json.loads((output_dir / "buyer_package_readout.json").read_text(encoding="utf-8"))
    rights_section = readout["sections"]["rights_privacy_provenance"]
    assert rights_section["revenue_share_review_status"] == "recorded_review_required"
    assert rights_section["owner_revenue_share_record_present"] is True
    assert rights_section["operator_revenue_terms_present"] is True
    assert rights_section["commercialization_terms_present"] is True
    assert rights_section["exclusivity_terms_present"] is True
    assert rights_section["required_before_paid_reuse_or_resale"] is True
    assert rights_section["paid_reuse_or_resale_blocked"] is True
    assert rights_section["data_processing_terms_review_status"] == (
        "recorded_review_required"
    )
    assert rights_section["retention_policy_present"] is True
    assert rights_section["subprocessor_list_present"] is True
    assert rights_section["access_audit_terms_present"] is True
    assert rights_section["dpa_approval_claimed"] is False
    assert rights_section["revenue_share_commitment_made"] is False
    assert rights_section["payout_commitment_allowed"] is False


def test_post_training_data_package_blocks_revoked_consent_and_writes_revenue_review(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    _write_json(
        capture_root / "raw" / "rights_consent.json",
        {
            "consent_status": "revoked",
            "consent_revoked_at": "2026-07-04T12:00:00Z",
            "consent_scope": ["robot_evaluation", "model_training"],
            "permission_document_uri": "s3://blueprint-consent/site-a.pdf",
        },
    )
    _write_json(
        capture_root / "pipeline" / "robot_eval_dataset" / "rights_packet.json",
        {
            "status": "blocked",
            "consent_revoked": True,
            "consent_revoked_at": "2026-07-04T12:00:00Z",
            "record_count": 1,
            "records": [
                {
                    "rights_scope": "model_training",
                    "evidence_uri": "s3://blueprint-consent/site-a.pdf",
                }
            ],
        },
    )
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    output_dir = tmp_path / "package"

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    assert manifest["status"] == "blocked_consent_revoked_takedown_required"
    assert "consent:consent_revoked_takedown_required" in manifest["blockers"]
    assert manifest["included_artifacts"]["revocation_takedown_manifest"] == (
        "revocation_takedown_manifest.json"
    )
    assert manifest["consent_evidence"]["consent_revoked"] is True
    assert manifest["revocation_takedown"]["status"] == "takedown_required"
    assert manifest["revocation_takedown"]["local_package_access_revoked"] is True
    assert manifest["revocation_takedown"]["delivery_blocked"] is True
    assert manifest["revocation_takedown"]["signed_access_revoked"] is True
    assert manifest["revocation_takedown"]["webapp_takedown_executed"] is False
    assert (
        manifest["revocation_takedown"]["hosted_session_takedown_executed"]
        is False
    )
    assert "remove_or_expire_hosted_sessions" in manifest["revocation_takedown"][
        "required_actions"
    ]
    assert "notify_webapp_rights_privacy_blocking" in manifest["revocation_takedown"][
        "downstream_unexecuted_actions"
    ]
    assert manifest["downstream_takedown_artifacts"] == {
        "webapp_rights_privacy_takedown_notice": (
            "webapp_rights_privacy_takedown_notice.json"
        ),
        "hosted_session_takedown_request": "hosted_session_takedown_request.json",
        "downstream_takedown_execution_ledger": (
            "downstream_takedown_execution_ledger.json"
        ),
    }
    assert manifest["downstream_takedown_execution_ledger"]["status"] == (
        "queued_unexecuted_downstream_takedown"
    )
    assert manifest["downstream_takedown_execution_ledger"][
        "external_takedown_executor_present"
    ] is False
    assert manifest["downstream_takedown_execution_ledger"][
        "webapp_or_hosted_takedown_execution_proven"
    ] is False
    assert (
        manifest["export_policy"]["downstream_takedown_execution_ledger_included"]
        is True
    )
    assert manifest["included_artifacts"]["webapp_rights_privacy_takedown_notice"] == (
        "webapp_rights_privacy_takedown_notice.json"
    )
    assert manifest["included_artifacts"]["hosted_session_takedown_request"] == (
        "hosted_session_takedown_request.json"
    )
    assert (
        manifest["export_policy"]["webapp_rights_privacy_takedown_notice_included"]
        is True
    )
    assert manifest["export_policy"]["hosted_session_takedown_request_included"] is True
    assert manifest["revenue_share_review"]["status"] == "review_required"
    assert manifest["revenue_share_review"]["revenue_share_commitment_made"] is False
    assert manifest["handoff_records"]["delivery_manifest_status"] == (
        "revoked_consent_takedown_required"
    )
    assert manifest["handoff_records"]["signed_access_manifest_status"] == (
        "revoked_consent_takedown_required"
    )
    assert manifest["handoff_records"]["local_package_access_revoked"] is True
    video_bundle = json.loads(
        (output_dir / "exports" / "video_bundle" / "clips_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    clip_rights = video_bundle["clips"][0]["rights_metadata"]
    assert clip_rights["consent_revoked"] is True
    assert clip_rights["consent_revoked_at"] == "2026-07-04T12:00:00Z"
    assert clip_rights["delivery_blocked_by_consent_revocation"] is True
    assert clip_rights["signed_access_revoked_by_consent"] is True
    assert clip_rights["manual_rights_review_recommended"] is True
    assert clip_rights["commercial_use_claim_allowed"] is False
    assert clip_rights["external_licensing_claim_allowed"] is False

    takedown = json.loads(
        (output_dir / "revocation_takedown_manifest.json").read_text(encoding="utf-8")
    )
    assert takedown["status"] == "takedown_required"
    assert takedown["local_package_access_revoked"] is True
    assert takedown["downstream_takedown_required"] is True
    assert takedown["webapp_takedown_executed"] is False
    assert takedown["hosted_session_takedown_executed"] is False
    assert "delivery_manifest.json" in takedown["affected_artifacts"]
    assert "webapp_rights_privacy_takedown_notice.json" in takedown["affected_artifacts"]
    assert "hosted_session_takedown_request.json" in takedown["affected_artifacts"]
    assert "downstream_takedown_execution_ledger.json" in takedown["affected_artifacts"]
    assert takedown["downstream_takedown_execution_ledger_path"] == (
        "downstream_takedown_execution_ledger.json"
    )
    assert "remove_or_expire_hosted_sessions" in takedown["required_actions"]
    execution_ledger = json.loads(
        (output_dir / "downstream_takedown_execution_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution_ledger["schema_version"] == (
        "post_training_downstream_takedown_execution_ledger.v1"
    )
    assert execution_ledger["status"] == "queued_unexecuted_downstream_takedown"
    assert execution_ledger["local_package_access_revoked"] is True
    assert execution_ledger["delivery_blocked_by_consent_revocation"] is True
    assert execution_ledger["signed_access_revoked_by_consent"] is True
    assert execution_ledger["external_takedown_executor_present"] is False
    assert "external_takedown_executor_missing" in execution_ledger["blockers"]
    statuses_by_surface = {
        surface["surface"]: surface["status"] for surface in execution_ledger["surfaces"]
    }
    assert statuses_by_surface["post_training_data_package"] == "blocked_locally"
    assert statuses_by_surface["signed_delivery_access"] == "revoked_locally"
    assert statuses_by_surface["webapp_projection"] == "queued_unexecuted"
    assert statuses_by_surface["hosted_sessions"] == "queued_unexecuted"
    webapp_notice = json.loads(
        (output_dir / "webapp_rights_privacy_takedown_notice.json").read_text(
            encoding="utf-8"
        )
    )
    hosted_request = json.loads(
        (output_dir / "hosted_session_takedown_request.json").read_text(
            encoding="utf-8"
        )
    )
    assert webapp_notice["status"] == "queued_unexecuted_webapp_rights_privacy_blocking"
    assert webapp_notice["required_webapp_state"] == (
        "blocked_consent_revoked_takedown_required"
    )
    assert webapp_notice["webapp_takedown_executed"] is False
    assert webapp_notice["claim_boundary"]["webapp_takedown_execution_proven"] is False
    assert hosted_request["status"] == "queued_unexecuted_hosted_session_takedown"
    assert hosted_request["hosted_review_assets_access_allowed"] is False
    assert hosted_request["hosted_session_takedown_executed"] is False
    assert (
        hosted_request["claim_boundary"]["hosted_session_takedown_execution_proven"]
        is False
    )

    package_index = json.loads((output_dir / "package_index.json").read_text(encoding="utf-8"))
    assert package_index["status"] == "revoked_consent_takedown_required"
    assert package_index["local_package_access_revoked"] is True
    assert package_index["delivery_blocked_by_consent_revocation"] is True
    assert package_index["signed_access_revoked_by_consent"] is True
    assert package_index["files"]["revocation_takedown_manifest"] == (
        "revocation_takedown_manifest.json"
    )
    assert package_index["files"]["webapp_rights_privacy_takedown_notice"] == (
        "webapp_rights_privacy_takedown_notice.json"
    )
    assert package_index["files"]["hosted_session_takedown_request"] == (
        "hosted_session_takedown_request.json"
    )
    assert package_index["files"]["downstream_takedown_execution_ledger"] == (
        "downstream_takedown_execution_ledger.json"
    )
    assert package_index["files"]["revenue_share_review"] == "revenue_share_review.json"
    archive_members = json.loads((output_dir / "archive_manifest.json").read_text(encoding="utf-8"))
    assert archive_members["status"] == "created_revoked_consent_takedown_required"
    assert archive_members["local_package_access_revoked"] is True
    assert archive_members["delivery_blocked_by_consent_revocation"] is True
    assert archive_members["signed_access_revoked_by_consent"] is True
    assert "revocation_takedown_manifest.json" in archive_members["included_files"]
    assert "downstream_takedown_execution_ledger.json" in archive_members[
        "included_files"
    ]
    assert "webapp_rights_privacy_takedown_notice.json" in archive_members["included_files"]
    assert "hosted_session_takedown_request.json" in archive_members["included_files"]
    assert "revenue_share_review.json" in archive_members["included_files"]

    delivery = json.loads((output_dir / "delivery_manifest.json").read_text(encoding="utf-8"))
    signed_access = json.loads(
        (output_dir / "signed_access_manifest.json").read_text(encoding="utf-8")
    )
    handoff = json.loads(
        (output_dir / "customer_handoff_report.json").read_text(encoding="utf-8")
    )
    assert delivery["status"] == "revoked_consent_takedown_required"
    assert delivery["delivery_blocked_by_consent_revocation"] is True
    assert delivery["local_package_access_revoked"] is True
    assert signed_access["status"] == "revoked_consent_takedown_required"
    assert signed_access["signed_access_ready"] is False
    assert signed_access["signed_access_revoked_by_consent"] is True
    assert handoff["status"] == "revoked_consent_takedown_required"
    assert handoff["post_training_data_package_handoff"][
        "local_package_access_revoked"
    ] is True
    assert (
        handoff["claim_boundary"]["webapp_or_hosted_takedown_execution_proven"]
        is False
    )
    readout = json.loads((output_dir / "buyer_package_readout.json").read_text(encoding="utf-8"))
    assert "revocation_takedown:consent_revoked_takedown_required" in readout["blockers"]
    assert readout["sections"]["revocation_takedown"]["status"] == "takedown_required"
    assert (
        readout["sections"]["revocation_takedown"][
            "webapp_rights_privacy_takedown_notice_path"
        ]
        == "webapp_rights_privacy_takedown_notice.json"
    )
    assert (
        readout["sections"]["revocation_takedown"][
            "hosted_session_takedown_request_path"
        ]
        == "hosted_session_takedown_request.json"
    )
    assert readout["sections"]["revocation_takedown"][
        "downstream_takedown_execution_ledger_status"
    ] == "queued_unexecuted_downstream_takedown"
    assert readout["sections"]["revocation_takedown"][
        "downstream_takedown_execution_ledger_path"
    ] == "downstream_takedown_execution_ledger.json"
    assert readout["sections"]["revocation_takedown"][
        "external_takedown_executor_present"
    ] is False
    assert (
        readout["sections"]["revocation_takedown"][
            "webapp_or_hosted_takedown_execution_proven"
        ]
        is False
    )
    assert readout["claim_boundary"]["consent_revocation_blocks_downstream_use"] is True


def test_export_policy_rl_flags_track_actual_handoff_content(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)

    # An empty RL handoff packet must surface as *not included* in export_policy
    # instead of the flags being hardcoded True.
    monkeypatch.setattr(
        package_module,
        "build_rl_post_training_handoff_packet",
        lambda **_kwargs: {},
    )

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=tmp_path / "package",
    )

    export_policy = manifest["export_policy"]
    assert export_policy["rl_post_training_handoff_included"] is False
    assert export_policy["rl_sparse_reward_signal_included"] is False
    assert export_policy["concurrent_baseline_ab_plan_included"] is False
    assert export_policy["bottleneck_stage_detection_included"] is False
    assert export_policy["speed_curriculum_plan_included"] is False
    assert export_policy["action_chunk_continuity_qa_included"] is False
    assert export_policy["intervention_safety_ledger_included"] is False


def test_lerobot_episode_export_wired_into_package_export(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    control_rows = [
        {
            "stream_type": "control_action",
            "attempt_id": "attempt-1",
            "action_index": 0,
            "task_id": "task-1",
            "scenario_id": "scenario-1",
            "action": {
                "delta_position_m": [0.05, 0.0, 0.01],
                "delta_rotation_axis_angle": [0.0, 0.0, 0.1],
                "gripper": 0.0,
                "base_pose_7d": [0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0],
                "sim_time_s": 0.0,
            },
        }
    ]
    (job_dir / "simulator_command_batch_control_stream.jsonl").write_text(
        "\n".join(json.dumps(row) for row in control_rows) + "\n",
        encoding="utf-8",
    )
    (job_dir / "simulator_command_batch_attempt_trace.jsonl").write_text(
        json.dumps(
            {
                "attempt_id": "attempt-1",
                "episode_id": "episode-1",
                "scenario_eval_run_id": "run-1",
                "task_id": "task-1",
                "scenario_id": "scenario-1",
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (job_dir / "clip-1.mp4").write_bytes(b"fake-mp4")
    output_dir = tmp_path / "package"

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    export_policy = manifest["export_policy"]
    assert export_policy["lerobot_episode_export_included"] is True
    assert export_policy["lerobot_episode_export_status"] == "completed_review_required"
    assert export_policy["lerobot_episode_export_episode_count"] == 1
    assert export_policy["lerobot_gr00t_ready_episode_count"] == 1
    assert manifest["included_artifacts"]["lerobot_episode_export_manifest"] == (
        "lerobot_episode_export/lerobot_episode_export_manifest.json"
    )
    lerobot_manifest = json.loads(
        (
            output_dir
            / "lerobot_episode_export"
            / "lerobot_episode_export_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert lerobot_manifest["status"] == "completed_review_required"
    video_bundle = json.loads(
        (output_dir / "exports" / "video_bundle" / "clips_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    clip_rights = video_bundle["clips"][0]["rights_metadata"]
    assert clip_rights["metadata_source"] == "package_consent_evidence"
    assert clip_rights["license_status"] == "blocked_missing_consent_evidence"
    assert clip_rights["redaction_status"] == "not_declared"
    assert clip_rights["consent_revoked"] is False
    assert clip_rights["delivery_blocked_by_consent_revocation"] is False
    assert clip_rights["signed_access_revoked_by_consent"] is False
    assert clip_rights["commercial_use_claim_allowed"] is False
    assert clip_rights["external_licensing_claim_allowed"] is False
    assert clip_rights["manual_rights_review_recommended"] is True
    episodes = [
        json.loads(line)
        for line in (
            output_dir / "lerobot_episode_export" / "meta" / "episodes.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert episodes[0]["state_present"] is True
    assert episodes[0]["timestamps_present"] is True
    assert episodes[0]["video_present"] is True
    assert episodes[0]["gr00t_ready"] is True
    assert episodes[0]["gr00t_ready_missing"] == []
    assert (
        output_dir / "lerobot_episode_export" / episodes[0]["video_path"]
    ).is_file()
    lerobot_v3_stats = json.loads(
        (output_dir / "exports" / "lerobot_v3" / "meta" / "stats.json").read_text(
            encoding="utf-8"
        )
    )
    assert lerobot_v3_stats["rights_metadata_frame_rows"] > 0
    lerobot_v3_episodes_path = (
        output_dir
        / "exports"
        / "lerobot_v3"
        / "meta"
        / "episodes"
        / "chunk-000"
        / "file-000.parquet.jsonl"
    )
    if lerobot_v3_episodes_path.is_file():
        lerobot_v3_episode = json.loads(
            lerobot_v3_episodes_path.read_text(encoding="utf-8").splitlines()[0]
        )
        assert (
            lerobot_v3_episode["source_rights_metadata"]["metadata_source"]
            == "package_consent_evidence"
        )
        assert (
            lerobot_v3_episode["source_rights_metadata"][
                "manual_rights_review_recommended"
            ]
            is True
        )
        assert (
            lerobot_v3_episode["source_rights_metadata"]["consent_revoked"]
            is False
        )
        assert (
            lerobot_v3_episode["source_rights_metadata"][
                "delivery_blocked_by_consent_revocation"
            ]
            is False
        )
        assert (
            lerobot_v3_episode["source_rights_metadata"][
                "signed_access_revoked_by_consent"
            ]
            is False
        )
        assert (
            lerobot_v3_episode["source_rights_metadata"][
                "commercial_use_claim_allowed"
            ]
            is False
        )
        assert (
            lerobot_v3_episode["source_rights_metadata"][
                "external_licensing_claim_allowed"
            ]
            is False
        )
    assert (
        output_dir / "lerobot_episode_export" / "meta" / "modality.json"
    ).is_file()
    assert (
        output_dir / "lerobot_episode_export" / "data" / "episode_000000.jsonl"
    ).is_file()
