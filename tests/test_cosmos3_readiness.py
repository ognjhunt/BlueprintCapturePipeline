from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.synthesis import cosmos3_readiness as c3
from blueprint_pipeline.synthesis.cosmos3_readiness import (
    COSMOS3_READINESS_SCHEMA_VERSION,
    evaluate_cosmos3_capture_readiness,
    write_cosmos3_capture_readiness,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _capture_root(tmp_path: Path) -> Path:
    return tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"


def _write_base_capture(tmp_path: Path, *, include_held_out: bool = True) -> Path:
    capture_root = _capture_root(tmp_path)
    raw_root = capture_root / "raw"
    pipeline_root = capture_root / "pipeline"
    (raw_root / "arkit" / "depth").mkdir(parents=True)
    (pipeline_root / "privacy").mkdir(parents=True)
    (raw_root / "walkthrough.mov").write_bytes(b"video")
    (pipeline_root / "privacy" / "final_walkthrough.mov").write_bytes(b"privacy-safe-video")
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps({"sceneId": "scene-1", "captureId": "capture-1"}),
        encoding="utf-8",
    )
    (raw_root / "arkit" / "poses.jsonl").write_text(
        json.dumps({"frame_id": "000001", "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        raw_root / "arkit" / "intrinsics.json",
        {"fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480},
    )
    manifest = {
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "video_uri": "walkthrough.mov",
        "site_identity": {"site_id": "site-1", "site_id_source": "fixture"},
        "capture_topology": {"capture_session_id": "session-1", "pass_count": 2},
        "capture_rights": {
            "derived_scene_generation_allowed": True,
            "consent_status": "documented",
        },
        "world_model_candidate": True,
    }
    _write_json(raw_root / "manifest.json", manifest)
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "site_id": "site-1",
            "capture_source": "meta_glasses",
            "capture_modality": "glasses_video_only",
            "world_model_candidate": True,
            "quality": {"world_model_candidate": True},
            "capture_rights": manifest["capture_rights"],
            "metadata": {
                "site_identity": manifest["site_identity"],
                "capture_topology": manifest["capture_topology"],
                "capture_rights": manifest["capture_rights"],
            },
        },
    )
    _write_json(
        pipeline_root / "geometry" / "geometry_summary.json",
        {
            "status": "completed_degraded",
            "geometry_source": "local_sfm",
            "fallback_used": False,
            "provider_native_result": False,
            "contract_ready_for_world_model": True,
            "intrinsics_available": True,
            "pose_track_count": 6,
            "geometry_live_ready": False,
            "site_frame_available": False,
            "scale_resolved": False,
        },
    )
    site_root = tmp_path / "bucket" / "sites" / "site-1" / "reference_memory"
    site_root.mkdir(parents=True)
    (site_root / "site_reference_index.jsonl").write_text(
        json.dumps({"reference_id": "ref-1", "capture_id": "capture-1"}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        site_root / "site_reference_manifest.json",
        {
            "schema_version": "site_reference_database.v1",
            "site_id": "site-1",
            "total_reference_frames": 6,
            "capture_count": 1,
            "chunk_count": 1,
            "readiness": {"state": "degraded", "blockers": ["site_frame_not_established"]},
            "site_frame_established": False,
        },
    )
    _write_json(
        pipeline_root / "cosmos_training_export" / "manifest.json",
        {
            "schema_version": "v1",
            "status": "ready",
            "source_mode": "dense_index",
            "paired_example_count": 3,
            "val_count": 1,
            "trainer_config_path": str(pipeline_root / "cosmos_training_export" / "trainer_config.json"),
            "inference_backend_shape_path": str(pipeline_root / "cosmos_training_export" / "inference_backend_shape.json"),
        },
    )
    _write_json(
        pipeline_root
        / "cosmos_single_capture_smoke"
        / "cosmos_single_capture_smoke_manifest.json",
        {
            "status": "blocked",
            "reason": "cosmos_runtime_unavailable",
            "benchmark_family": "cosmos_single_capture_smoke",
        },
    )
    if include_held_out:
        _write_json(
            pipeline_root / "evaluation_prep" / "held_out_revisits.json",
            {"schema_version": "v1", "routes": [{"route_id": "heldout-1"}]},
        )
    return capture_root


def test_cosmos3_readiness_maps_existing_stack_without_runtime_claims(tmp_path: Path) -> None:
    capture_root = _write_base_capture(tmp_path)

    report = evaluate_cosmos3_capture_readiness(capture_root)

    assert report["schema_version"] == COSMOS3_READINESS_SCHEMA_VERSION
    assert report["provider_jobs_called"] is False
    assert report["model_download_required"] is False
    assert report["stack_checks"]["raw_capture_contract"]["state"] == "ready"
    assert report["stack_checks"]["geometry_lane"]["state"] == "degraded"
    assert report["stack_checks"]["site_reference_database"]["state"] == "degraded"
    assert report["stack_checks"]["cosmos_predict25_export"]["state"] == "ready"
    assert report["capabilities"]["reasoner_site_understanding"]["state"] == "data_ready"
    assert report["capabilities"]["generator_site_conditioning"]["state"] == "data_ready"
    assert report["capabilities"]["world_action_policy"]["state"] == "blocked"
    assert report["simulation_boundary"]["visual_site_review_without_full_digital_twin"] is True
    assert report["simulation_boundary"]["robot_action_or_collision_eval_without_sim_ready_twin"] is False
    assert "Generated outputs are ground truth" in report["simulation_boundary"]["blocked_claims"]


def test_cosmos3_readiness_blocks_fallback_geometry_from_grounding(tmp_path: Path) -> None:
    capture_root = _write_base_capture(tmp_path)
    _write_json(
        capture_root / "pipeline" / "geometry" / "geometry_summary.json",
        {
            "status": "completed_with_fallback",
            "geometry_source": "fallback_geometry",
            "fallback_used": True,
            "contract_ready_for_world_model": True,
            "intrinsics_available": True,
            "pose_track_count": 3,
            "geometry_live_ready": False,
        },
    )

    report = evaluate_cosmos3_capture_readiness(capture_root)

    geometry_check = report["stack_checks"]["geometry_lane"]
    assert geometry_check["state"] == "blocked"
    assert "fallback_geometry_not_allowed_for_cosmos3_grounding" in geometry_check["blockers"]
    assert report["capabilities"]["generator_site_conditioning"]["state"] == "blocked"


def test_cosmos3_readiness_requires_held_out_revisit_for_eval_claims(tmp_path: Path) -> None:
    capture_root = _write_base_capture(tmp_path, include_held_out=False)
    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["metadata"]["capture_topology"]["pass_count"] = 1
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    manifest_path = capture_root / "raw" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["capture_topology"]["pass_count"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate_cosmos3_capture_readiness(capture_root)

    validation_check = report["stack_checks"]["held_out_validation"]
    assert validation_check["state"] == "blocked"
    assert "missing_held_out_revisit_or_second_pass" in validation_check["blockers"]
    assert report["capabilities"]["evaluator_site_consistency"]["state"] == "blocked"


def test_write_cosmos3_readiness_artifacts(tmp_path: Path) -> None:
    capture_root = _write_base_capture(tmp_path)

    report = write_cosmos3_capture_readiness(capture_root)

    json_path = Path(report["artifact_paths"]["json"])
    markdown_path = Path(report["artifact_paths"]["markdown"])
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert "Cosmos 3 Capture-Grounded Readiness" in markdown_path.read_text(encoding="utf-8")


def test_cosmos3_readiness_blocks_when_capture_evidence_is_missing(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    capture_root.mkdir(parents=True)

    report = evaluate_cosmos3_capture_readiness(capture_root)

    raw_check = report["stack_checks"]["raw_capture_contract"]
    assert raw_check["state"] == "blocked"
    assert raw_check["blockers"] == [
        "missing_capture_descriptor_or_raw_manifest",
        "missing_capture_upload_complete",
        "missing_raw_walkthrough_video",
        "derived_scene_generation_not_allowed",
    ]
    assert raw_check["warnings"] == [
        "privacy_safe_conditioning_video_not_found",
        "raw_pose_stream_not_found",
        "raw_intrinsics_not_found",
        "raw_depth_dir_not_found",
    ]
    assert report["stack_checks"]["geometry_lane"]["blockers"] == ["missing_geometry_summary"]
    assert report["stack_checks"]["site_reference_database"]["blockers"] == ["missing_site_id"]
    assert report["stack_checks"]["cosmos_predict25_export"]["blockers"] == [
        "missing_cosmos_training_export_manifest"
    ]
    assert report["stack_checks"]["cosmos_predict25_benchmark"]["warnings"] == [
        "missing_cosmos_benchmark_manifest"
    ]
    assert report["public_claims"]["allowed_internal_claim"] == (
        "Cosmos 3 feasibility is still blocked on local evidence gaps"
    )


def test_cosmos3_readiness_recommends_stackable_work_for_missing_site_and_export(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    capture_root.mkdir(parents=True)

    report = evaluate_cosmos3_capture_readiness(capture_root, site_id="site-missing")

    assert report["stack_checks"]["site_reference_database"]["blockers"] == [
        "missing_site_reference_manifest"
    ]
    assert {item["lane"] for item in report["today_stackable_work"]} == {
        "provider_native_geometry",
        "site_reference_database",
        "cosmos_predict25_export",
    }


def test_cosmos3_readiness_reports_site_export_and_benchmark_edges(tmp_path: Path) -> None:
    capture_root = _write_base_capture(tmp_path)
    pipeline_root = capture_root / "pipeline"
    site_root = tmp_path / "bucket" / "sites" / "site-1" / "reference_memory"
    (site_root / "site_reference_index.jsonl").unlink()
    _write_json(
        site_root / "site_reference_manifest.json",
        {
            "schema_version": "site_reference_database.v1",
            "site_id": "site-1",
            "total_reference_frames": 0,
            "capture_count": 0,
            "readiness": {"blockers": ["stale_reference_memory"]},
            "site_frame_established": True,
        },
    )
    _write_json(
        pipeline_root / "cosmos_training_export" / "manifest.json",
        {
            "status": "blocked",
            "reason": "source_exports_missing",
            "paired_example_count": 0,
            "val_count": 0,
        },
    )
    (
        pipeline_root
        / "cosmos_single_capture_smoke"
        / "cosmos_single_capture_smoke_manifest.json"
    ).unlink()

    report = evaluate_cosmos3_capture_readiness(capture_root)

    site_check = report["stack_checks"]["site_reference_database"]
    assert site_check["blockers"] == [
        "missing_site_reference_index",
        "no_site_reference_frames",
        "no_site_reference_captures",
    ]
    assert site_check["warnings"] == ["stale_reference_memory"]
    export_check = report["stack_checks"]["cosmos_predict25_export"]
    assert export_check["blockers"] == ["source_exports_missing", "no_cosmos_paired_examples"]
    assert export_check["warnings"] == ["no_validation_split_examples"]
    assert report["stack_checks"]["cosmos_predict25_benchmark"]["warnings"] == [
        "missing_cosmos_benchmark_manifest"
    ]


def test_cosmos3_readiness_marks_action_eval_data_ready_with_action_evidence(
    tmp_path: Path,
) -> None:
    capture_root = _write_base_capture(tmp_path)
    pipeline_root = capture_root / "pipeline"
    _write_json(
        pipeline_root / "geometry" / "geometry_summary.json",
        {
            "status": "completed",
            "geometry_source": "video_to_world",
            "fallback_used": False,
            "geometry_live_ready": True,
            "site_frame_available": True,
            "scale_resolved": True,
            "pose_track_count": 12,
        },
    )
    site_root = tmp_path / "bucket" / "sites" / "site-1" / "reference_memory"
    _write_json(
        site_root / "site_reference_manifest.json",
        {
            "schema_version": "site_reference_database.v1",
            "site_id": "site-1",
            "total_reference_frames": 8,
            "capture_count": 2,
            "site_frame_established": True,
        },
    )
    (capture_root / "raw" / "action_labels.jsonl").write_text(
        json.dumps({"frame_id": "000001", "action": "open"}) + "\n",
        encoding="utf-8",
    )

    report = evaluate_cosmos3_capture_readiness(capture_root)

    assert report["stack_checks"]["geometry_lane"]["state"] == "ready"
    assert report["stack_checks"]["site_reference_database"]["state"] == "ready"
    assert report["stack_checks"]["action_evidence"]["state"] == "ready"
    assert report["capabilities"]["world_action_policy"]["state"] == "data_ready"
    assert report["simulation_boundary"]["robot_action_or_collision_eval_without_sim_ready_twin"] is True


def test_cosmos3_readiness_helper_fallbacks_and_normalizers(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True)
    context = c3.resolve_local_capture_context(capture_root)
    bad_json = raw_root / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    _write_json(raw_root / "site_identity.json", {"site_id": "sidecar-site"})

    class GetOnly:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def get(self, key: str, default: object = None) -> object:
            return self.payload.get(key, default)

    assert c3._read_optional_json(bad_json) == {}
    assert c3._merge_raw_sidecars({}, raw_root)["site_identity"] == {"site_id": "sidecar-site"}
    assert (
        c3._raw_video_path(
            context=context,
            descriptor={},
            raw_manifest={"raw_video_uri": "gs://bucket/raw-video.mov"},
        )
        == "gs://bucket/raw-video.mov"
    )
    assert (
        c3._privacy_safe_video_path(
            context=context,
            descriptor={"metadata": {"world_model_video_uri": "gs://bucket/privacy.mov"}},
        )
        == "gs://bucket/privacy.mov"
    )
    assert c3._derived_scene_generation_allowed(
        descriptor={
            "capture_rights": "not-a-mapping",
            "metadata": {"rights_lineage": {"derived_generation_allowed": True}},
        },
        raw_manifest={},
    )
    assert not c3._derived_scene_generation_allowed(descriptor={}, raw_manifest={})
    assert c3._world_model_candidate(
        descriptor={"quality": {"world_model_candidate": True}},
        raw_manifest={},
    )
    assert c3._resolve_site_reference_root(context=context, site_id=None) is None
    assert c3._resolve_site_reference_root(context=context, site_id="missing-site") == (
        context.storage_root / context.bucket / "sites" / "missing-site" / "reference_memory"
    )
    assert (
        c3._resolve_site_id(
            descriptor=GetOnly({"site_id": "", "metadata": {}}),
            raw_manifest=GetOnly({"site_id": ""}),
        )
        is None
    )
    assert c3._first_int(None, "", "not-an-int", [], "7") == 7
    assert c3._first_int("not-an-int") is None
    assert c3._string_list(None) == []
    assert c3._string_list(" one ") == ["one"]
    assert c3._string_list(5) == ["5"]
    assert c3._string_list(["one", " one ", "", "two"]) == ["one", "two"]
    markdown = c3._render_markdown(
        {
            "capture": "not-a-mapping",
            "stack_checks": {"ignored": "not-a-mapping"},
            "capabilities": {"ignored": "not-a-mapping"},
            "simulation_boundary": "not-a-mapping",
        }
    )
    assert "Cosmos 3 Capture-Grounded Readiness" in markdown
