from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import capture_batch_registry as batch_registry
from blueprint_pipeline.capture_batch_registry import main, update_capture_batch_registry


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {"site_identity": {"site_id": "site-1"}},
        },
    )
    _write_json(
        root / "raw" / "manifest.json",
        {"scene_id": "scene-1", "capture_id": "capture-1", "site_identity": {"site_id": "site-1"}},
    )
    return root


def _write_stage_artifacts(root: Path) -> None:
    pipeline = root / "pipeline"
    automation = pipeline / "simulation_automation"
    job = pipeline / "robot_eval_jobs" / "job-1"
    _write_json(
        pipeline / "privacy_processing_manifest.json",
        {"status": "person_removed", "privacy_processed_video_uri": "gs://bucket/privacy/final_walkthrough.mov"},
    )
    _write_json(pipeline / "worldlabs_request_manifest.json", {"status": "ready_for_generation"})
    _write_json(pipeline / "worldlabs_operation_manifest.json", {"done": True, "status": "ready"})
    _write_json(pipeline / "worldlabs_world_manifest.json", {"world_id": "world-1"})
    _write_json(
        pipeline / "worldlabs_assets" / "materialized_assets_manifest.json",
        {"status": "complete", "download_count": 1},
    )
    _write_json(pipeline / "worldlabs_export_manifest.json", {"status": "complete"})
    _write_json(
        automation / "cpu_preflight_manifest.json",
        {"status": "ready_for_owner_gpu_preflight_handoff", "ready_for_owner_gpu_preflight": True},
    )
    _write_json(
        automation / "gpu_handoff_packet.json",
        {"status": "ready_for_owner_gpu_preflight_handoff", "blockers": ["owner_gpu_simulator_execution_not_run"]},
    )
    _write_json(job / "evaluation_result.json", {"status": "completed"})
    _write_json(
        job / "post_training_data_package_export_manifest.json",
        {"status": "export_ready_review_required"},
    )


def test_capture_batch_registry_tracks_per_site_status_and_retry_resume(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_stage_artifacts(root)
    registry_path = tmp_path / "site_capture_batch_registry.json"

    registry = update_capture_batch_registry(
        capture_roots=[root],
        registry_path=registry_path,
        resume=True,
    )

    capture = registry["sites"]["site-1"]["captures"]["capture-1"]
    assert capture["stage_statuses"]["privacy"]["status"] == "complete"
    assert capture["stage_statuses"]["worldlabs"]["status"] == "complete"
    assert capture["stage_statuses"]["materialization"]["status"] == "complete"
    assert capture["stage_statuses"]["cpu_preflight"]["status"] == "complete"
    assert capture["stage_statuses"]["gpu_handoff"]["status"] == "ready_except_owner_gpu"
    assert capture["stage_statuses"]["eval_result"]["status"] == "complete"
    assert capture["stage_statuses"]["data_package_export"]["status"] == "complete"
    assert capture["resume"]["next_stage"] == "owner_gpu_simulator_execution"

    _write_json(
        root / "pipeline" / "worldlabs_assets" / "materialized_assets_manifest.json",
        {"status": "blocked", "download_count": 0},
    )
    retried = update_capture_batch_registry(
        capture_roots=[root],
        registry_path=registry_path,
        resume=True,
        retry_stage="materialization",
    )
    retried_capture = retried["sites"]["site-1"]["captures"]["capture-1"]

    assert retried_capture["stage_statuses"]["materialization"]["status"] == "queued_for_retry"
    assert retried_capture["stage_statuses"]["materialization"]["previous_status"] == "blocked"
    assert retried_capture["attempts"]["materialization"]["attempt_count"] == 2
    assert retried_capture["resume"]["resume_from_stage"] == "materialization"
    assert _read_json(registry_path)["schema_version"] == "site_capture_batch_registry.v1"


def test_capture_batch_registry_handles_pending_complete_and_cli_paths(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "local-blueprint" / "scenes" / "scene-2" / "captures" / "capture-2"
    _write_json(root / "capture_descriptor.json", {"scene_id": "scene-2", "capture_id": "capture-2"})
    _write_json(root / "raw" / "manifest.json", {"scene_id": "scene-2", "capture_id": "capture-2"})
    registry_path = tmp_path / "registry.json"

    with pytest.raises(ValueError, match="retry_stage must be one of"):
        update_capture_batch_registry(
            capture_roots=[root],
            registry_path=registry_path,
            retry_stage="not-a-stage",
        )

    pending = update_capture_batch_registry(capture_roots=[root], registry_path=registry_path)
    capture = pending["sites"]["scene-2"]["captures"]["capture-2"]
    assert capture["stage_statuses"]["privacy"]["status"] == "pending"
    assert capture["resume"]["next_stage"] == "privacy"

    _write_stage_artifacts(root)
    _write_json(
        root / "pipeline" / "simulation_automation" / "gpu_handoff_packet.json",
        {"status": "complete", "owner_gpu_simulator_execution_proven": True, "blockers": []},
    )
    complete = update_capture_batch_registry(
        capture_roots=[root],
        registry_path=registry_path,
        resume=False,
    )
    complete_capture = complete["sites"]["scene-2"]["captures"]["capture-2"]
    assert complete_capture["stage_statuses"]["gpu_handoff"]["status"] == "complete"
    assert complete_capture["resume"]["next_stage"] is None
    assert batch_registry._discover_capture_roots(tmp_path / "local-blueprint") == [root]

    cli_path = tmp_path / "cli-registry.json"
    assert main(["--storage-root", str(tmp_path / "local-blueprint"), "--registry-path", str(cli_path), "--no-resume"]) == 0
    captured = capsys.readouterr()
    assert "[capture-batch-registry] registry=" in captured.out
    assert "[capture-batch-registry] site_count=1" in captured.out
    with pytest.raises(SystemExit, match="provide at least one"):
        main(["--registry-path", str(tmp_path / "empty.json")])
