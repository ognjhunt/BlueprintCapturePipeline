from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import alpha_readiness as alpha
from blueprint_pipeline import canonical_site_package as canonical

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


def test_canonical_site_package_defensive_inputs_and_missing_fields(tmp_path: Path) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{not-json", encoding="utf-8")

    assert canonical._string_list("restricted") == ["restricted"]
    assert canonical._read_optional_json(tmp_path / "missing.json") == {}
    assert canonical._read_optional_json(invalid_json) == {}
    assert canonical._world_labs_readiness(
        worldlabs_input={
            "status": "ready",
            "output_video_uri": "gs://bucket/privacy/world-model.mp4",
            "audit_payload": {"privacy_safe_input": True},
        },
        privacy_processing={"status": "person_removed"},
        rights_review={"status": "cleared"},
        provenance_summary={"status": "stale"},
    )["warnings"] == ["provenance_summary:stale"]

    package = canonical.build_blueprint_canonical_site_package(
        descriptor={
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {"privacy_restrictions": "no faces"},
        },
        capture_root=tmp_path / "capture",
        pipeline_dir=tmp_path / "capture" / "pipeline",
        bucket="local-blueprint",
        storage_root=tmp_path,
        pipeline_prefix="scenes/scene-1/captures/capture-1/pipeline",
        descriptor_uri="gs://local-blueprint/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        qa_report_uri="gs://local-blueprint/scenes/scene-1/captures/capture-1/qa_report.json",
        qa_report={"status": "passed"},
        privacy_processing={},
        worldlabs_input={},
        geometry_artifacts={},
        provenance_summary={},
        rights_provenance_review={},
        site_intake={},
        scorecard={},
        qualification_record={},
        task_targets_payload={"targets": [123, {"object_label": "tote"}]},
        task_hypothesis_report={},
    )

    assert package["semantic_task_context"]["restricted_areas"] == ["no faces"]
    assert package["semantic_task_context"]["target_objects"] == [
        {"label": "tote", "source": "task_targets", "raw": {"object_label": "tote"}}
    ]
    assert package["conditioning"]["rgb_video"]["restricted_raw_capture"] == {
        "present": False,
        "exported": False,
        "access_scope": "restricted_internal_evidence",
    }
    assert "conditioning.frames.frame_index_uri" in package["missing_fields"]
    assert "conditioning.camera.poses_uri" in package["missing_fields"]
    assert "conditioning.camera.intrinsics_uri" in package["missing_fields"]
    assert "conditioning.depth_confidence.depth" in package["missing_fields"]
