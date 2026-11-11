from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import alpha_readiness as alpha

import pytest

pytestmark = pytest.mark.slow


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_alpha_readiness_defensive_helpers_and_legacy_sync_merge(tmp_path: Path) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{not-json", encoding="utf-8")
    pipeline_root = tmp_path / "pipeline"
    _write_json(
        pipeline_root / "webapp_sync_result.json",
        {
            "status": "succeeded",
            "stage": "qualification",
            "attachment_payload": {"site_submission_id": "site-1"},
        },
    )

    assert alpha._read_json_object(invalid_json) == {}
    assert alpha._bool_env({"FLAG": "true"}, "FLAG") is True
    assert alpha._mode_payload({"capture_mode": {"resolved_mode": "site_world_candidate"}}) == {
        "resolved_mode": "site_world_candidate"
    }
    assert alpha._latest_sync_payload(
        {
            "latest_stage": "qualification",
            "syncs": {"qualification": {"status": "succeeded", "stage": "qualification"}},
        }
    ) == {"status": "succeeded", "stage": "qualification"}
    assert alpha._latest_sync_payload({"status": "legacy"}) == {"status": "legacy"}

    legacy = alpha.write_pipeline_sync_result(
        pipeline_root=pipeline_root,
        stage="evaluation_prep",
        result={"status": "failed", "stage": "evaluation_prep"},
    )

    assert legacy["latest_stage"] == "evaluation_prep"
    assert legacy["syncs"]["qualification"]["status"] == "succeeded"
    assert legacy["syncs"]["qualification"]["attachment_payload"] == {"site_submission_id": "site-1"}
    assert legacy["syncs"]["evaluation_prep"]["status"] == "failed"


def test_alpha_evaluation_prep_sync_records_upstream_bootstrap_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "capture_source": "iphone",
            "capture_tier": "tier1_iphone",
            "capture_modality": "iphone_arkit_lidar",
            "raw_prefix_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/raw",
            "frames_index_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/frames/index.jsonl",
            "requested_outputs": ["evaluation_prep"],
            "arkit_poses_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/raw/arkit/poses.jsonl",
            "arkit_intrinsics_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/raw/arkit/intrinsics.json",
            "arkit_depth_prefix_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/raw/arkit/depth",
            "metadata": {
                "capture_mode": {"resolved_mode": "site_world_candidate"},
            },
        },
    )

    def fail_sync(**_kwargs: object) -> dict[str, object]:
        raise ValueError("missing upstream request bootstrap")

    monkeypatch.setattr(alpha, "sync_webapp_pipeline_attachment", fail_sync)

    result = alpha.sync_webapp_evaluation_prep(capture_root=capture_root, env={})

    failed = result["syncs"]["evaluation_prep"]
    assert failed["status"] == "failed"
    assert failed["blocker"] == "webapp_sync_requires_upstream_request_job_bootstrap"
    assert "missing upstream request bootstrap" in failed["reason"]


