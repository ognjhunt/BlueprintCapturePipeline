from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from blueprint_pipeline import first_gpu_sample_video_preflight as video_preflight
from blueprint_pipeline import first_gpu_sample_video_stage as stage_module
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
        "preview_simulation",
        "robot_eval_dataset",
        "task_evaluation_run",
    ]
    assert manifest["capture_capabilities"]["camera_pose"] is False
    assert manifest["capture_rights"] == {
        "derived_scene_generation_allowed": False,
        "data_licensing_allowed": False,
        "capture_contributor_payout_eligible": False,
        "consent_status": "unknown",
        "permission_document_uri": None,
        "consent_scope": [],
        "consent_notes": [],
    }
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


def test_stage_first_gpu_sample_video_can_carry_owner_rights_scope(tmp_path: Path) -> None:
    source_video = tmp_path / "sample.mp4"
    source_video.write_bytes(b"fake-video")

    result = stage_first_gpu_sample_video(
        source_video=source_video,
        storage_root=tmp_path / "storage",
        bucket="local-blueprint",
        scene_id="sample-scene",
        capture_id="sample-capture",
        derived_scene_generation_allowed=True,
        data_licensing_allowed=False,
        consent_status="documented",
        permission_document_uri="codex-thread://owner-run-approval/2026-06-12",
        consent_scope=[
            "owner_gpu_smoke",
            "worldlabs_generation_for_this_capture",
        ],
        consent_notes=["Owner explicitly requested this isolated first-GPU smoke."],
    )

    capture_root = Path(result["capture_root"])
    manifest = json.loads((capture_root / "raw" / "manifest.json").read_text(encoding="utf-8"))
    rights = manifest["capture_rights"]
    assert rights["derived_scene_generation_allowed"] is True
    assert rights["data_licensing_allowed"] is False
    assert rights["consent_status"] == "documented"
    assert rights["permission_document_uri"] == (
        "codex-thread://owner-run-approval/2026-06-12"
    )
    assert rights["consent_scope"] == [
        "owner_gpu_smoke",
        "worldlabs_generation_for_this_capture",
    ]
    assert result["capture_rights"] == rights

    preflight = build_capture_preflight_report(capture_root)
    descriptor_rights = preflight["descriptor_preview"]["metadata"]["capture_rights"]
    assert descriptor_rights["derived_scene_generation_allowed"] is True
    assert descriptor_rights["consent_status"] == "documented"
    assert descriptor_rights["permission_document_uri"] == (
        "codex-thread://owner-run-approval/2026-06-12"
    )
    intake_packet = json.loads((capture_root / "raw" / "intake_packet.json").read_text(encoding="utf-8"))
    upload_complete = json.loads(
        (capture_root / "raw" / "capture_upload_complete.json").read_text(encoding="utf-8")
    )
    assert intake_packet["capture_rights"] == rights
    assert intake_packet["owner_approval"]["status"] == "documented"
    assert upload_complete["capture_rights"] == rights
    assert upload_complete["owner_approval"]["approved_scope"] == [
        "owner_gpu_smoke",
        "worldlabs_generation_for_this_capture",
    ]
    assert upload_complete["source_video_sha256"] == manifest["source_video"]["sha256"]


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


def test_first_gpu_sample_video_preflight_media_and_cli_edges(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake-video")
    duplicate = tmp_path / "nested" / "duplicate.MOV"
    duplicate.parent.mkdir()
    duplicate.write_bytes(b"fake-video")

    assert video_preflight._float_or_none("bad") is None
    assert video_preflight._float_or_none(-1) is None
    assert video_preflight._int_or_none("bad") is None
    assert video_preflight._int_or_none(-1) is None
    assert video_preflight._discover_videos([tmp_path / "missing", video, tmp_path]) == [
        video.resolve(),
        duplicate.resolve(),
    ]

    monkeypatch.setattr(video_preflight.shutil, "which", lambda _name: None)
    unavailable = video_preflight._ffprobe_media_metadata(video)
    assert unavailable["status"] == "unavailable"
    assert "ffprobe_not_found" in unavailable["blockers"]

    monkeypatch.setattr(video_preflight.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        video_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=2, stderr="x" * 500, stdout=""),
    )
    failed = video_preflight._ffprobe_media_metadata(video)
    assert failed["status"] == "failed"
    assert failed["stderr_tail"] == "x" * 400

    monkeypatch.setattr(
        video_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr="", stdout="{not-json"),
    )
    assert video_preflight._ffprobe_media_metadata(video)["blockers"] == ["ffprobe_output_not_json"]

    monkeypatch.setattr(
        video_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "format": {"duration": "12.5", "format_name": "mov,mp4"},
                    "streams": [
                        {
                            "codec_type": "video",
                            "width": "1920",
                            "height": "1080",
                            "codec_name": "h264",
                        }
                    ],
                }
            ),
        ),
    )
    ready = video_preflight._ffprobe_media_metadata(video)
    assert ready["status"] == "ready"
    assert ready["duration_seconds"] == 12.5
    assert ready["width"] == 1920
    assert ready["height"] == 1080

    missing = video_preflight._audit_video(
        tmp_path / "missing.mp4",
        max_duration_seconds=30,
        max_size_bytes=100,
        require_probe=False,
    )
    assert "source_video_missing" in missing["staging_blockers"]
    unsupported = video_preflight._audit_video(
        tmp_path / "sample.txt",
        max_duration_seconds=30,
        max_size_bytes=100,
        require_probe=False,
    )
    assert "unsupported_video_suffix" in unsupported["staging_blockers"]
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    monkeypatch.setattr(
        video_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({})),
    )
    empty_audit = video_preflight._audit_video(
        empty,
        max_duration_seconds=30,
        max_size_bytes=100,
        require_probe=False,
    )
    assert "source_video_empty" in empty_audit["worldlabs_blockers"]
    assert "source_video_duration_unknown" in empty_audit["worldlabs_blockers"]
    monkeypatch.setattr(
        video_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps({"format": {"duration": "12.5"}, "streams": [{"codec_type": "video"}]}),
        ),
    )
    big = video_preflight._audit_video(
        video,
        max_duration_seconds=1,
        max_size_bytes=1,
        require_probe=True,
    )
    assert "source_video_exceeds_worldlabs_size_limit" in big["worldlabs_blockers"]
    assert "source_video_exceeds_worldlabs_duration_limit" in big["worldlabs_blockers"]

    no_sources = video_preflight.build_first_gpu_sample_video_preflight()
    assert no_sources["blockers"] == ["no_source_videos_found"]
    no_staging = video_preflight.build_first_gpu_sample_video_preflight(
        source_videos=[tmp_path / "sample.txt"],
    )
    assert no_staging["blockers"] == ["no_source_videos_ready_for_capture_staging"]
    monkeypatch.setattr(video_preflight.shutil, "which", lambda _name: None)
    no_worldlabs = video_preflight.build_first_gpu_sample_video_preflight(
        source_videos=[video],
        require_probe=True,
        output_path=tmp_path / "preflight.json",
    )
    assert no_worldlabs["blockers"] == ["no_source_videos_ready_for_worldlabs_first_clip"]
    assert no_worldlabs["output_path"].endswith("preflight.json")

    assert video_preflight.main(["--output", str(tmp_path / "empty-output.json")]) == 1
    assert "blockers=no_source_videos_found" in capsys.readouterr().out

    monkeypatch.setattr(
        video_preflight,
        "build_first_gpu_sample_video_preflight",
        lambda **kwargs: {
            "status": "ready",
            "source_video_count": 1,
            "ready_for_worldlabs_first_clip_count": 1,
            "blockers": [],
            "output_path": str(kwargs["output_path"]),
        },
    )
    assert video_preflight.main(["--source-video", str(video), "--output", str(tmp_path / "ready.json")]) == 0
    assert "ready_worldlabs=1" in capsys.readouterr().out


def test_stage_first_gpu_sample_video_validation_replacement_and_cli_print_edges(
    tmp_path: Path,
    capsys,
) -> None:
    source_video = tmp_path / "sample.mp4"
    source_video.write_bytes(b"fake-video")

    with pytest.raises(PipelineError, match="scene_id is required"):
        stage_module._safe_id("", field="scene_id")
    with pytest.raises(PipelineError, match="path-safe"):
        stage_module._safe_id("../bad", field="scene_id")
    with pytest.raises(PipelineError, match="Source video is missing"):
        stage_first_gpu_sample_video(
            source_video=tmp_path / "missing.mp4",
            storage_root=tmp_path / "storage",
            scene_id="scene",
            capture_id="capture",
        )
    bad_suffix = tmp_path / "sample.txt"
    bad_suffix.write_text("not-video", encoding="utf-8")
    with pytest.raises(PipelineError, match="Source video must use"):
        stage_first_gpu_sample_video(
            source_video=bad_suffix,
            storage_root=tmp_path / "storage",
            scene_id="scene",
            capture_id="capture",
        )

    link_target = tmp_path / "linked.mp4"
    stage_module._copy_or_link_video(source=source_video, target=link_target, mode="link")
    assert link_target.is_symlink()
    with pytest.raises(PipelineError, match="Unsupported staging mode"):
        stage_module._copy_or_link_video(source=source_video, target=tmp_path / "bad.mp4", mode="move")

    file_root = tmp_path / "file-root"
    file_root.write_text("stale", encoding="utf-8")
    stage_module._remove_existing_capture_root(file_root)
    assert not file_root.exists()
    dir_root = tmp_path / "dir-root"
    dir_root.mkdir()
    stage_module._remove_existing_capture_root(dir_root)
    assert not dir_root.exists()

    result = stage_first_gpu_sample_video(
        source_video=source_video,
        storage_root=tmp_path / "storage-existing",
        scene_id="scene",
        capture_id="capture",
    )
    capture_root = Path(result["capture_root"])
    with pytest.raises(PipelineError, match="already exists"):
        stage_first_gpu_sample_video(
            source_video=source_video,
            storage_root=tmp_path / "storage-existing",
            scene_id="scene",
            capture_id="capture",
        )
    replaced = stage_first_gpu_sample_video(
        source_video=source_video,
        storage_root=tmp_path / "storage-existing",
        scene_id="scene",
        capture_id="capture",
        force=True,
    )
    assert Path(replaced["capture_root"]) == capture_root

    assert main(
        [
            "--source-video",
            str(tmp_path / "missing.mp4"),
            "--storage-root",
            str(tmp_path / "storage"),
            "--scene-id",
            "scene",
            "--capture-id",
            "capture",
        ]
    ) == 1
    assert "[first-gpu-sample-stage] FAILED:" in capsys.readouterr().out

    local_output = tmp_path / "local-stage.json"
    assert main(
        [
            "--source-video",
            str(source_video),
            "--storage-root",
            str(tmp_path / "storage-local-cli"),
            "--scene-id",
            "scene",
            "--capture-id",
            "capture",
            "--site-submission-id",
            "site-1",
            "--request-id",
            "request-1",
            "--buyer-request-id",
            "buyer-1",
            "--capture-job-id",
            "capture-job-1",
            "--stage-local-webapp-rehearsal-request",
            "--output",
            str(local_output),
        ]
    ) == 0
    assert "local_webapp_rehearsal_staged_inputs=" in capsys.readouterr().out

    sim_output = tmp_path / "sim-stage.json"
    assert main(
        [
            "--source-video",
            str(source_video),
            "--storage-root",
            str(tmp_path / "storage-sim-cli"),
            "--scene-id",
            "scene",
            "--capture-id",
            "capture",
            "--run-simulation-automation",
            "--output",
            str(sim_output),
        ]
    ) == 0
    sim_stdout = capsys.readouterr().out
    assert "simulation_automation_status=blocked" in sim_stdout
    assert "gpu_handoff_blockers=owner_gpu_simulator_execution_not_run" in sim_stdout
    assert "gpu_handoff_hard_preflight_blockers=" in sim_stdout
