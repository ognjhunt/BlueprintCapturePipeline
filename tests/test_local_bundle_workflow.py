from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import local_bundle_workflow as local_workflow
from blueprint_pipeline import preflight_capture as preflight
from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.local_bundle_workflow import (
    detect_bundle_identity,
    run_local_bundle_workflow,
    stage_local_bundle,
)
from blueprint_pipeline.materialization import capture_materialization_readiness
from blueprint_pipeline.preflight_capture import build_capture_preflight_report
from tests.runpy_entrypoint import run_module_as_main


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_source_bundle(source_root: Path, *, scene_id: str = "scene-1", capture_id: str = "capture-1") -> None:
    raw_dir = source_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "manifest.json": {"scene_id": scene_id, "capture_id": capture_id},
        "capture_context.json": {"scene_id": scene_id, "capture_id": capture_id},
        "capture_upload_complete.json": {"scene_id": scene_id, "capture_id": capture_id},
    }
    for name, payload in payloads.items():
        _write_json(raw_dir / name, payload)


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


def test_stage_local_bundle_copy_preserves_raw_bundle_byte_for_byte(tmp_path: Path) -> None:
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

    source_snapshot = {
        path.relative_to(raw_dir).as_posix(): path.read_bytes()
        for path in raw_dir.rglob("*")
        if path.is_file()
    }
    capture_root = stage_local_bundle(
        source_bundle=source_root,
        storage_root=tmp_path / "storage",
        bucket="local-blueprint",
        mode="copy",
    )

    staged_raw = capture_root / "raw"
    staged_snapshot = {
        path.relative_to(staged_raw).as_posix(): path.read_bytes()
        for path in staged_raw.rglob("*")
        if path.is_file()
    }
    assert staged_snapshot == source_snapshot


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
    assert "--provider local" in result["commands"]["agent_review_local"]
    assert "agent_review_openai" not in result["commands"]
    assert result["remaining_runtime_requirements"]["agent_review_local"] == [
        "no LLM key required",
        "use --provider openai or --provider claude only for configured external review overrides",
    ]


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


def test_local_bundle_workflow_rejects_invalid_bundle_metadata_and_modes(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="missing raw"):
        detect_bundle_identity(tmp_path / "no-raw")

    missing = tmp_path / "missing"
    (missing / "raw").mkdir(parents=True)
    with pytest.raises(PipelineError, match="Required bundle file is missing"):
        local_workflow._read_json_object(missing / "raw" / "manifest.json")

    invalid = tmp_path / "invalid"
    (invalid / "raw").mkdir(parents=True)
    (invalid / "raw" / "manifest.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(PipelineError, match="Invalid JSON"):
        local_workflow._read_json_object(invalid / "raw" / "manifest.json")

    non_object = tmp_path / "non-object"
    (non_object / "raw").mkdir(parents=True)
    (non_object / "raw" / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(PipelineError, match="Expected JSON object"):
        local_workflow._read_json_object(non_object / "raw" / "manifest.json")

    source_root = tmp_path / "source-errors"
    _write_source_bundle(source_root)
    with pytest.raises(PipelineError, match="Unsupported staging mode"):
        stage_local_bundle(source_bundle=source_root, storage_root=tmp_path / "storage", mode="move")

    no_ids = tmp_path / "no-ids"
    _write_source_bundle(no_ids, scene_id="", capture_id="")
    with pytest.raises(PipelineError, match="Could not determine scene_id"):
        detect_bundle_identity(no_ids)

    no_capture_id = tmp_path / "no-capture-id"
    _write_source_bundle(no_capture_id, scene_id="scene-1", capture_id="")
    with pytest.raises(PipelineError, match="Could not determine capture_id"):
        detect_bundle_identity(no_capture_id)

    conflicting = tmp_path / "conflicting"
    _write_source_bundle(conflicting)
    _write_json(conflicting / "raw" / "capture_context.json", {"scene_id": "scene-2", "capture_id": "capture-1"})
    with pytest.raises(PipelineError, match="Conflicting scene IDs"):
        detect_bundle_identity(conflicting)

    conflicting_capture = tmp_path / "conflicting-capture"
    _write_source_bundle(conflicting_capture)
    _write_json(
        conflicting_capture / "raw" / "capture_upload_complete.json",
        {"scene_id": "scene-1", "capture_id": "capture-2"},
    )
    with pytest.raises(PipelineError, match="Conflicting capture IDs"):
        detect_bundle_identity(conflicting_capture)


def test_stage_local_bundle_replaces_existing_roots_and_can_symlink(tmp_path: Path) -> None:
    source_root = tmp_path / "source-stage"
    _write_source_bundle(source_root)

    linked_capture = stage_local_bundle(
        source_bundle=source_root,
        storage_root=tmp_path / "linked-storage",
        mode="link",
    )
    assert (linked_capture / "raw").is_symlink()

    copied_capture = stage_local_bundle(
        source_bundle=source_root,
        storage_root=tmp_path / "copy-storage",
        mode="copy",
    )
    with pytest.raises(PipelineError, match="already exists"):
        stage_local_bundle(source_bundle=source_root, storage_root=tmp_path / "copy-storage", mode="copy")
    replaced_capture = stage_local_bundle(
        source_bundle=source_root,
        storage_root=tmp_path / "copy-storage",
        mode="copy",
        force=True,
    )
    assert replaced_capture == copied_capture
    assert (replaced_capture / "raw" / "manifest.json").is_file()

    file_storage = tmp_path / "file-storage"
    target = file_storage / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    target.parent.mkdir(parents=True)
    target.write_text("stale", encoding="utf-8")
    unlinked_capture = stage_local_bundle(
        source_bundle=source_root,
        storage_root=file_storage,
        mode="copy",
        force=True,
    )
    assert unlinked_capture.is_dir()

    with pytest.raises(PipelineError, match="missing raw"):
        stage_local_bundle(source_bundle=tmp_path / "missing-raw", storage_root=tmp_path / "storage")


def test_run_local_bundle_workflow_validates_lane_and_preflight_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "source-workflow-errors"
    _write_source_bundle(source_root)

    with pytest.raises(PipelineError, match="requires --run-qualification"):
        run_local_bundle_workflow(
            source_bundle=source_root,
            storage_root=tmp_path / "storage-a",
            run_evaluation_prep=True,
        )
    with pytest.raises(PipelineError, match="Unsupported local workflow pipeline lane"):
        run_local_bundle_workflow(
            source_bundle=source_root,
            storage_root=tmp_path / "storage-b",
            pipeline_lane="not-a-lane",
        )

    monkeypatch.setattr(
        "blueprint_pipeline.local_bundle_workflow.build_capture_preflight_report",
        lambda *_args, **_kwargs: {"status": "blocked", "missing_required_inputs": ["video"]},
    )
    with pytest.raises(PipelineError, match="missing required inputs: video"):
        run_local_bundle_workflow(
            source_bundle=source_root,
            storage_root=tmp_path / "storage-c",
            mode="copy",
        )


def test_preflight_capture_reports_missing_intake_and_video_uri_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "source-preflight"
    _write_source_bundle(source_root)
    _write_json(
        source_root / "raw" / "manifest.json",
        {"scene_id": "scene-1", "capture_id": "capture-1", "video_uri": "gs://bucket/video.mov"},
    )
    capture_root = stage_local_bundle(
        source_bundle=source_root,
        storage_root=tmp_path / "storage-preflight",
        mode="copy",
    )
    (capture_root / "raw" / "intake_packet.json").unlink(missing_ok=True)
    monkeypatch.setattr(
        "blueprint_pipeline.preflight_capture.preview_capture_bundle",
        lambda **_kwargs: {
            "descriptor": {"capture_modality": "iphone_video_only", "evidence_tier": "pre_screen_video"},
            "qa_report": {"status": "passed", "escalation_recommendation": {"human_review_required": False}},
        },
    )

    assert preflight._string_value(None, "") == ""
    report = build_capture_preflight_report(capture_root)
    assert "intake_packet" in report["missing_required_inputs"]
    assert report["video_candidates"] == ["gs://bucket/video.mov"]

    _write_json(source_root / "raw" / "manifest.json", {"scene_id": "scene-1", "capture_id": "capture-1"})
    capture_without_video = stage_local_bundle(
        source_bundle=source_root,
        storage_root=tmp_path / "storage-preflight-missing-video",
        mode="copy",
    )
    (capture_without_video / "raw" / "intake_packet.json").unlink(missing_ok=True)
    missing_video = build_capture_preflight_report(capture_without_video)
    assert "video" in missing_video["missing_required_inputs"]


def test_preflight_capture_main_writes_output_and_reports_failures(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    success_report = {
        "status": "ready_for_materialization",
        "mode_decision": "qualified_metric_capture",
        "human_review_required": False,
        "missing_required_inputs": [],
    }
    monkeypatch.setattr(preflight, "build_capture_preflight_report", lambda _root: success_report)
    output = tmp_path / "preflight.json"

    assert preflight.main(["--capture-root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ready_for_materialization"

    blocked_report = {
        **success_report,
        "status": "blocked",
        "missing_required_inputs": ["video", "intake_packet"],
    }
    monkeypatch.setattr(preflight, "build_capture_preflight_report", lambda _root: blocked_report)
    assert preflight.main(["--capture-root", str(tmp_path)]) == 1
    assert "missing_required_inputs=video,intake_packet" in capsys.readouterr().out

    monkeypatch.setattr(
        preflight,
        "build_capture_preflight_report",
        lambda _root: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert preflight.main(["--capture-root", str(tmp_path)]) == 1
    assert "[capture-preflight] FAILED: boom" in capsys.readouterr().out


def test_preflight_capture_module_entrypoint_runs(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source-preflight-main"
    _write_source_bundle(source_root)
    _write_json(
        source_root / "raw" / "intake_packet.json",
        {"workflowName": "Inspect station", "taskSteps": ["walk"], "zone": "zone-a"},
    )
    (source_root / "raw" / "walkthrough.mov").write_bytes(b"video")
    capture_root = stage_local_bundle(
        source_bundle=source_root,
        storage_root=tmp_path / "storage-preflight-main",
        mode="copy",
    )
    monkeypatch.setattr(sys, "argv", ["preflight_capture", "--capture-root", str(capture_root)])

    with pytest.raises(SystemExit) as exc_info:
        run_module_as_main("blueprint_pipeline.preflight_capture")

    assert exc_info.value.code == 0
