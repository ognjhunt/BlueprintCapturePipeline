from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.first_gpu_sample_video_stage import (
    FIRST_GPU_SAMPLE_VIDEO_STAGE_SCHEMA_VERSION,
    LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND,
    main,
    stage_first_gpu_sample_video,
)
from blueprint_pipeline.preflight_capture import build_capture_preflight_report


def test_stage_first_gpu_sample_video_writes_preflightable_bundle(tmp_path: Path) -> None:
    source_video = tmp_path / "sample.mp4"
    source_video.write_bytes(b"fake-video")

    result = stage_first_gpu_sample_video(
        source_video=source_video,
        storage_root=tmp_path / "storage",
        bucket="local-blueprint",
        scene_id="sample-scene",
        capture_id="sample-capture",
        workflow_name="Pick tote",
        task_steps=["approach tote", "pick tote"],
        site_submission_id="site-submission-1",
        request_id="request-1",
        buyer_request_id="buyer-request-1",
        capture_job_id="capture-job-1",
    )

    capture_root = Path(result["capture_root"])
    assert result["schema_version"] == FIRST_GPU_SAMPLE_VIDEO_STAGE_SCHEMA_VERSION
    assert result["preflight_missing_required_inputs"] == []
    assert Path(result["source_video_preflight_path"]).is_file()
    assert result["source_video_preflight_status"] == "ready"
    assert result["source_video_ready_for_worldlabs_first_clip"] is True
    assert (capture_root / "raw" / "walkthrough.mp4").read_bytes() == b"fake-video"
    manifest = json.loads((capture_root / "raw" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["requested_outputs"] == [
        "qualification",
        "robot_eval_dataset",
        "task_evaluation_run",
    ]
    assert manifest["capture_capabilities"]["camera_pose"] is False
    preflight = build_capture_preflight_report(capture_root)
    assert preflight["missing_required_inputs"] == []
    assert preflight["video_candidates"] == ["walkthrough.mp4"]
    assert "no_ready_first_gpu_candidates" in result["candidate_audit_blockers"]


def test_stage_first_gpu_sample_video_cli_writes_manifest(tmp_path: Path) -> None:
    source_video = tmp_path / "sample.mov"
    source_video.write_bytes(b"fake-video")
    output = tmp_path / "stage-result.json"

    exit_code = main(
        [
            "--source-video",
            str(source_video),
            "--storage-root",
            str(tmp_path / "storage"),
            "--scene-id",
            "sample-scene",
            "--capture-id",
            "sample-capture",
            "--workflow-name",
            "Pick tote",
            "--task-step",
            "approach tote",
            "--task-step",
            "pick tote",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == FIRST_GPU_SAMPLE_VIDEO_STAGE_SCHEMA_VERSION
    assert Path(payload["capture_root"]).is_dir()
    assert payload["source_video_preflight_path"].endswith("source_video_preflight_manifest.json")
    assert payload["claim_boundary"]["gpu_provisioning_performed"] is False


def test_stage_first_gpu_sample_video_can_write_gpu_handoff_summary(tmp_path: Path) -> None:
    source_video = tmp_path / "sample.mp4"
    source_video.write_bytes(b"fake-video")

    result = stage_first_gpu_sample_video(
        source_video=source_video,
        storage_root=tmp_path / "storage",
        bucket="local-blueprint",
        scene_id="sample-scene",
        capture_id="sample-capture",
        run_simulation_automation=True,
    )

    capture_root = Path(result["capture_root"])
    gpu_handoff_path = capture_root / "pipeline" / "simulation_automation" / "gpu_handoff_packet.json"
    assert result["simulation_automation_run"] is True
    assert result["simulation_automation_status"] == "blocked"
    assert result["gpu_handoff_packet_path"] == str(gpu_handoff_path)
    assert gpu_handoff_path.is_file()
    assert "owner_gpu_simulator_execution_not_run" in result["gpu_handoff_blockers"]
    assert "missing_local_scene_asset" in result["gpu_handoff_hard_preflight_blockers"]
    assert "missing_scene_frame_estimate" in result["gpu_handoff_hard_preflight_blockers"]
    assert "scene_bounds_missing_or_invalid" in result["gpu_handoff_hard_preflight_blockers"]
    details = {
        item["blocker_id"]: item
        for item in result["gpu_handoff_pre_gpu_blocker_details"]
    }
    assert details["missing_local_scene_asset"]["source_artifact"] == "scene_asset_preflight.json"
    assert details["missing_scene_frame_estimate"]["source_artifact"] == "scene_frame_estimate.json"
    assert details["scene_bounds_missing_or_invalid"]["source_artifact"] == (
        "spawn_pose_validation_manifest.json"
    )
    assert result["gpu_handoff_spawn_validation_summary"]["status"] == "blocked"
    assert result["claim_boundary"]["simulation_automation_artifacts_written"] is True
    assert result["claim_boundary"]["simulator_execution_performed"] is False


def test_stage_first_gpu_sample_video_can_use_explicit_scene_asset(
    tmp_path: Path,
) -> None:
    source_video = tmp_path / "sample.mp4"
    source_video.write_bytes(b"fake-video")
    scene_asset = tmp_path / "scene.obj"
    scene_asset.write_text(
        "\n".join(
            [
                "v -1.0 -1.0 0.0",
                "v 1.0 -1.0 0.0",
                "v 1.0 1.0 0.0",
                "v -1.0 1.0 0.0",
                "v 0.0 0.0 1.0",
                "f 1 2 5",
                "f 2 3 5",
                "f 3 4 5",
                "f 4 1 5",
            ]
        ),
        encoding="utf-8",
    )

    result = stage_first_gpu_sample_video(
        source_video=source_video,
        storage_root=tmp_path / "storage",
        bucket="local-blueprint",
        scene_id="sample-scene",
        capture_id="sample-capture",
        scene_assets=[scene_asset],
        run_simulation_automation=True,
    )

    assert result["scene_asset_inputs"] == [str(scene_asset)]
    assert result["gpu_handoff_status"] == "ready_for_owner_gpu_preflight_handoff"
    assert result["gpu_handoff_ready_for_owner_gpu_preflight"] is True
    assert result["gpu_handoff_blockers"] == ["owner_gpu_simulator_execution_not_run"]
    assert result["gpu_handoff_hard_preflight_blockers"] == []
    assert result["gpu_handoff_spawn_validation_summary"]["status"] == "review_required"
    assert result["claim_boundary"]["simulator_execution_performed"] is False


def test_stage_first_gpu_sample_video_can_write_local_webapp_rehearsal_request(
    tmp_path: Path,
) -> None:
    source_video = tmp_path / "sample.mp4"
    source_video.write_bytes(b"fake-video")

    result = stage_first_gpu_sample_video(
        source_video=source_video,
        storage_root=tmp_path / "storage",
        bucket="local-blueprint",
        scene_id="sample-scene",
        capture_id="sample-capture",
        site_submission_id="site-submission-1",
        request_id="request-1",
        buyer_request_id="buyer-request-1",
        capture_job_id="capture-job-1",
        stage_local_webapp_rehearsal_request=True,
        local_webapp_job_id="local-webapp-job-1",
    )

    local_request = result["local_webapp_rehearsal_request"]
    request_path = Path(local_request["request_path"])
    staged_inputs_path = Path(local_request["staged_inputs_path"])
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    staged_inputs = json.loads(staged_inputs_path.read_text(encoding="utf-8"))

    assert local_request["status"] == "staged"
    assert local_request["source_kind"] == LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND
    assert request_payload["source_kind"] == LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND
    assert request_payload["job_request"]["schema_version"] == "robot_eval_job_request.v1"
    assert staged_inputs["local_rehearsal_only"] is True
    assert staged_inputs["webapp_request"]["source_kind"] == LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND
    assert staged_inputs["proof_boundary"]["real_webapp_forwarding_proven"] is False
    assert result["claim_boundary"]["local_webapp_rehearsal_request_written"] is True
    assert result["claim_boundary"]["real_webapp_forwarding_proven"] is False


def test_stage_first_gpu_sample_video_rehearsal_requires_real_upstream_ids(
    tmp_path: Path,
) -> None:
    source_video = tmp_path / "sample.mp4"
    source_video.write_bytes(b"fake-video")

    with pytest.raises(PipelineError, match="requires real upstream IDs"):
        stage_first_gpu_sample_video(
            source_video=source_video,
            storage_root=tmp_path / "storage",
            bucket="local-blueprint",
            scene_id="sample-scene",
            capture_id="sample-capture",
            site_submission_id="site-submission-1",
            stage_local_webapp_rehearsal_request=True,
        )


def test_stage_first_gpu_sample_video_strict_source_preflight_does_not_replace_existing_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_video = tmp_path / "sample.mp4"
    source_video.write_bytes(b"fake-video")
    capture_root = (
        tmp_path
        / "storage"
        / "local-blueprint"
        / "scenes"
        / "sample-scene"
        / "captures"
        / "sample-capture"
    )
    sentinel = capture_root / "raw" / "sentinel.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep", encoding="utf-8")

    monkeypatch.setattr(
        "blueprint_pipeline.first_gpu_sample_video_stage.build_first_gpu_sample_video_preflight",
        lambda **kwargs: {
            "status": "blocked",
            "blockers": ["no_source_videos_ready_for_worldlabs_first_clip"],
            "ready_for_worldlabs_first_clip_count": 0,
            "candidates": [
                {
                    "staging_blockers": [],
                    "worldlabs_blockers": ["source_video_exceeds_worldlabs_duration_limit"],
                }
            ],
        },
    )

    with pytest.raises(PipelineError, match="Source video failed strict first-GPU preflight"):
        stage_first_gpu_sample_video(
            source_video=source_video,
            storage_root=tmp_path / "storage",
            bucket="local-blueprint",
            scene_id="sample-scene",
            capture_id="sample-capture",
            force=True,
            require_source_video_preflight=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_stage_first_gpu_sample_video_cli_accepts_strict_source_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_video = tmp_path / "sample.mov"
    source_video.write_bytes(b"fake-video")
    output = tmp_path / "stage-result.json"

    monkeypatch.setattr(
        "blueprint_pipeline.first_gpu_sample_video_stage.build_first_gpu_sample_video_preflight",
        lambda **kwargs: {
            "schema_version": "first_gpu_sample_video_preflight.v1",
            "status": "ready",
            "blockers": [],
            "ready_for_worldlabs_first_clip_count": 1,
            "candidates": [
                {
                    "path": str(source_video.resolve()),
                    "ready_for_worldlabs_first_clip": True,
                    "staging_blockers": [],
                    "worldlabs_blockers": [],
                }
            ],
        },
    )

    exit_code = main(
        [
            "--source-video",
            str(source_video),
            "--storage-root",
            str(tmp_path / "storage"),
            "--scene-id",
            "sample-scene",
            "--capture-id",
            "sample-capture",
            "--require-source-video-preflight",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source_video_preflight_status"] == "ready"
    assert payload["source_video_ready_for_worldlabs_first_clip"] is True
