from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.local_bundle_workflow import (
    detect_bundle_identity,
    run_local_bundle_workflow,
    stage_local_bundle,
)
from blueprint_pipeline.materialization import capture_materialization_readiness
from blueprint_pipeline.preflight_capture import build_capture_preflight_report


def test_detect_bundle_identity_reads_raw_bundle_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    payloads = {
        "manifest.json": {"scene_id": "scene-1", "capture_id": "capture-1"},
        "capture_context.json": {"sceneId": "scene-1"},
        "capture_upload_complete.json": {"captureId": "capture-1"},
    }
    for name, payload in payloads.items():
        (raw_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    identity = detect_bundle_identity(tmp_path)
    assert identity.scene_id == "scene-1"
    assert identity.capture_id == "capture-1"


def test_stage_local_bundle_copy_strips_stale_object_index_derivatives(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    raw_dir = source_root / "raw"
    raw_dir.mkdir(parents=True)
    payloads = {
        "manifest.json": {"scene_id": "scene-1", "capture_id": "capture-1"},
        "capture_context.json": {"sceneId": "scene-1"},
        "capture_upload_complete.json": {"captureId": "capture-1"},
        "object_index.json": {"objects": [{"id": "old"}]},
        "object_index_build_report.json": {"status": "built"},
        "object_index_keyframes.json": {"keyframes": []},
        "object_grounding_hints.json": {"hints": []},
    }
    for name, payload in payloads.items():
        (raw_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (raw_dir / "object_index_artifacts").mkdir()
    (raw_dir / "object_index_artifacts" / "stale.txt").write_text("stale", encoding="utf-8")

    capture_root = stage_local_bundle(
        source_bundle=source_root,
        storage_root=tmp_path / "storage",
        bucket="local-blueprint",
        mode="copy",
    )

    staged_raw = capture_root / "raw"
    assert (staged_raw / "manifest.json").is_file()
    assert (staged_raw / "capture_context.json").is_file()
    assert (staged_raw / "capture_upload_complete.json").is_file()
    assert not (staged_raw / "object_index.json").exists()
    assert not (staged_raw / "object_index_build_report.json").exists()
    assert not (staged_raw / "object_index_keyframes.json").exists()
    assert not (staged_raw / "object_grounding_hints.json").exists()
    assert not (staged_raw / "object_index_artifacts").exists()


def test_preflight_waives_missing_intake_for_open_capture_with_accepted_hypothesis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "source"
    raw_dir = source_root / "raw"
    raw_dir.mkdir(parents=True)
    payloads = {
        "manifest.json": {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "special_task_type": "open_capture",
            "capture_source": "iphone",
            "capture_modality": "iphone_arkit_lidar",
            "video_uri": "raw/walkthrough.mov",
        },
        "capture_context.json": {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "special_task_type": "open_capture",
            "task_hypothesis_status": "accepted",
        },
        "capture_upload_complete.json": {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
        },
        "task_hypothesis.json": {
            "status": "accepted",
        },
    }
    for name, payload in payloads.items():
        (raw_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (raw_dir / "walkthrough.mov").write_bytes(b"video")

    capture_root = stage_local_bundle(
        source_bundle=source_root,
        storage_root=tmp_path / "storage",
        bucket="local-blueprint",
        mode="copy",
    )

    monkeypatch.setattr(
        "blueprint_pipeline.preflight_capture.preview_capture_bundle",
        lambda **_kwargs: {
            "descriptor": {
                "capture_modality": "iphone_arkit_lidar",
                "evidence_tier": "pre_screen_video",
            },
            "qa_report": {
                "status": "degraded",
                "escalation_recommendation": {"human_review_required": True},
            },
        },
    )

    report = build_capture_preflight_report(capture_root)

    assert report["missing_required_inputs"] == []
    assert report["intake_packet_waived"] is True
    assert report["status"] == "pre_screen_only"
    assert any("open-capture metadata" in note for note in report["notes"])


def test_materialization_readiness_accepts_walkthrough_mp4_without_mov(tmp_path: Path) -> None:
    raw_dir = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "manifest.json").write_text(
        json.dumps({"scene_id": "scene-1", "capture_id": "capture-1"}),
        encoding="utf-8",
    )
    (raw_dir / "walkthrough.mp4").write_bytes(b"video")

    readiness = capture_materialization_readiness(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        gcs_root=tmp_path / "storage",
    )

    assert readiness["ready"] is True
    assert readiness["issues"] == []
    assert readiness["video_candidates"] == ["walkthrough.mp4"]
    assert readiness["selected_video_path"].endswith("raw/walkthrough.mp4")


def test_run_local_bundle_workflow_can_request_scene_memory_lane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "source"
    raw_dir = source_root / "raw"
    raw_dir.mkdir(parents=True)
    payloads = {
        "manifest.json": {"scene_id": "scene-1", "capture_id": "capture-1"},
        "capture_context.json": {"scene_id": "scene-1", "capture_id": "capture-1"},
        "capture_upload_complete.json": {"scene_id": "scene-1", "capture_id": "capture-1"},
    }
    for name, payload in payloads.items():
        (raw_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "blueprint_pipeline.local_bundle_workflow.build_capture_preflight_report",
        lambda *_args, **_kwargs: {"status": "passed", "missing_required_inputs": []},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.local_bundle_workflow.materialize_capture_bundle",
        lambda **_kwargs: {"status": "ok"},
    )

    captured: dict[str, object] = {}

    def _run_capture_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"status": "completed", "lanes": [kwargs.get("lane")]}

    monkeypatch.setattr(
        "blueprint_pipeline.local_bundle_workflow.run_capture_pipeline",
        _run_capture_pipeline,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.local_bundle_workflow.run_evaluation_prep_stage",
        lambda **_kwargs: {"status": "not_ready_for_validation", "manifest_path": "evaluation_prep_manifest.json"},
    )

    result = run_local_bundle_workflow(
        source_bundle=source_root,
        storage_root=tmp_path / "storage",
        bucket="local-blueprint",
        mode="copy",
        run_qualification=True,
        run_evaluation_prep=True,
        pipeline_lane="scene_memory",
    )

    assert result["pipeline_lane"] == "scene_memory"
    assert captured["lane"] == "scene_memory"


def test_run_local_bundle_workflow_can_request_deeper_pipeline_lanes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "source"
    raw_dir = source_root / "raw"
    raw_dir.mkdir(parents=True)
    payloads = {
        "manifest.json": {"scene_id": "scene-1", "capture_id": "capture-1"},
        "capture_context.json": {"scene_id": "scene-1", "capture_id": "capture-1"},
        "capture_upload_complete.json": {"scene_id": "scene-1", "capture_id": "capture-1"},
    }
    for name, payload in payloads.items():
        (raw_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "blueprint_pipeline.local_bundle_workflow.build_capture_preflight_report",
        lambda *_args, **_kwargs: {"status": "passed", "missing_required_inputs": []},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.local_bundle_workflow.materialize_capture_bundle",
        lambda **_kwargs: {"status": "ok"},
    )

    captured: dict[str, object] = {}

    def _run_capture_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"status": "completed", "lanes": [kwargs.get("lane")]}

    monkeypatch.setattr(
        "blueprint_pipeline.local_bundle_workflow.run_capture_pipeline",
        _run_capture_pipeline,
    )

    result = run_local_bundle_workflow(
        source_bundle=source_root,
        storage_root=tmp_path / "storage",
        bucket="local-blueprint",
        mode="copy",
        run_qualification=True,
        pipeline_lane="evaluation_prep",
    )

    assert result["pipeline_lane"] == "evaluation_prep"
    assert captured["lane"] == "evaluation_prep"
