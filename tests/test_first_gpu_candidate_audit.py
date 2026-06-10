from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.first_gpu_candidate_audit import (
    FIRST_GPU_CANDIDATE_AUDIT_SCHEMA_VERSION,
    build_first_gpu_candidate_audit,
    main,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path, *, scene_id: str, capture_id: str, video: bool = True) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / scene_id / "captures" / capture_id
    _write_json(
        capture_root / "raw" / "manifest.json",
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "requested_outputs": ["qualification", "robot_eval_dataset", "task_evaluation_run"],
            "site_submission_id": "site-submission-1",
            "request_id": "request-1",
            "buyer_request_id": "buyer-request-1",
            "capture_job_id": "capture-job-1",
            "capture_capabilities": {"camera_pose": True},
        },
    )
    _write_json(capture_root / "raw" / "capture_context.json", {"workflowName": "GPU smoke"})
    _write_json(capture_root / "raw" / "intake_packet.json", {"workflowName": "GPU smoke"})
    _write_json(
        capture_root / "raw" / "capture_upload_complete.json",
        {"scene_id": scene_id, "capture_id": capture_id},
    )
    if video:
        (capture_root / "raw" / "walkthrough.mp4").write_bytes(b"fake-video")
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "requested_outputs": ["qualification", "robot_eval_dataset", "task_evaluation_run"],
            "site_submission_id": "site-submission-1",
            "request_id": "request-1",
            "buyer_request_id": "buyer-request-1",
            "capture_job_id": "capture-job-1",
        },
    )
    return capture_root


def test_first_gpu_candidate_audit_discovers_video_backed_capture(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path, scene_id="scene-1", capture_id="capture-1")
    _capture_root(tmp_path, scene_id="scene-2", capture_id="capture-2", video=False)
    output = tmp_path / "audit.json"

    result = build_first_gpu_candidate_audit(
        search_roots=[tmp_path / "storage"],
        output_path=output,
        require_webapp_forwarding=False,
        require_webapp_staged_request=False,
        require_gpu_gates=False,
    )

    assert result["schema_version"] == FIRST_GPU_CANDIDATE_AUDIT_SCHEMA_VERSION
    assert result["candidate_count"] == 2
    assert result["video_backed_candidate_count"] == 1
    assert result["ready_candidate_count"] == 0
    assert result["blockers"] == ["no_ready_first_gpu_candidates"]
    video_candidate = next(
        candidate
        for candidate in result["candidates"]
        if candidate["capture_root"] == str(capture_root.resolve())
    )
    assert video_candidate["has_raw_video"] is True
    assert video_candidate["raw_video_paths"] == [str(capture_root / "raw" / "walkthrough.mp4")]
    assert output.is_file()


def test_first_gpu_candidate_audit_cli_writes_blocked_manifest(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path, scene_id="scene-1", capture_id="capture-1")
    output = tmp_path / "cli-audit.json"

    exit_code = main(
        [
            "--capture-root",
            str(capture_root),
            "--output",
            str(output),
            "--no-require-webapp-forwarding",
            "--no-require-webapp-staged-request",
            "--no-require-gpu-gates",
        ]
    )

    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == FIRST_GPU_CANDIDATE_AUDIT_SCHEMA_VERSION
    assert payload["candidate_count"] == 1
    assert payload["claim_boundary"]["gpu_provisioning_performed"] is False
