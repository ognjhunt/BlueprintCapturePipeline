from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.evaluation_prep_stage import run_evaluation_prep_stage


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_capture(tmp_path: Path) -> Path:
    capture_root = tmp_path / "bucket" / "scenes" / "scene_eval" / "captures" / "cap_eval"
    pipeline_root = capture_root / "pipeline"
    raw_root = capture_root / "raw"
    (raw_root / "walkthrough.mov").parent.mkdir(parents=True, exist_ok=True)
    (raw_root / "walkthrough.mov").write_bytes(b"mov")
    (raw_root / "arkit").mkdir(parents=True, exist_ok=True)
    (raw_root / "arkit" / "poses.jsonl").write_text("{}\n", encoding="utf-8")
    (raw_root / "arkit" / "intrinsics.json").write_text("{}", encoding="utf-8")
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene_eval",
            "capture_id": "cap_eval",
            "capture_source": "iphone",
            "processing_profile": "pose_assisted",
            "scene_memory_capture": {
                "world_model_candidate": True,
                "sensor_availability": {
                    "arkit_poses": True,
                    "arkit_intrinsics": True,
                    "arkit_depth": False,
                    "arkit_confidence": False,
                    "arkit_meshes": False,
                    "motion": True,
                },
            },
            "metadata": {"task_statement": "Open and close the fridge door"},
        },
    )
    _write_json(
        raw_root / "object_index.json",
        {
            "objects": [
                {
                    "id": "1",
                    "label": "refrigerator",
                    "boundingBox": {
                        "center": [1.0, 0.0, 1.0],
                        "extents": [0.8, 0.8, 2.0],
                        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "orientationQuaternion": [1, 0, 0, 0],
                    },
                }
            ]
        },
    )
    _write_json(
        pipeline_root / "opportunity_handoff.json",
        {
            "schema_version": "v1",
            "site_submission_id": "site-sub-1",
            "opportunity_id": "opp-1",
            "qualification_state": "ready",
            "downstream_evaluation_eligibility": True,
            "operator_approved_summary": "Qualified fridge-door opportunity.",
            "scoped_task_definition": {
                "task_id": "task-1",
                "scoped_task_statement": "Open and close the fridge door",
                "success_criteria": ["door opens", "door closes"],
                "in_scope_zone": "kitchen_fridge_zone",
            },
            "site_constraints": {
                "operating_constraints": ["daytime only"],
                "privacy_security_constraints": ["no PII"],
                "known_blockers": ["none"],
            },
            "scene_memory_package": {
                "bundle_path": "scene_memory",
                "scene_memory_manifest_path": "scene_memory/scene_memory_manifest.json",
                "scene_memory_readiness_path": "scene_memory/scene_memory_readiness.json",
                "conditioning_bundle_path": "scene_memory/conditioning_bundle.json",
                "preview_simulation_manifest_path": "preview_simulation/preview_simulation_manifest.json",
                "gen3c_adapter_manifest_path": "scene_memory/adapter_manifests/gen3c.json",
                "neoverse_adapter_manifest_path": "scene_memory/adapter_manifests/neoverse.json",
                "cosmos_transfer_adapter_manifest_path": "scene_memory/adapter_manifests/cosmos_transfer.json",
            },
        },
    )
    _write_json(
        pipeline_root / "qualification_record.json",
        {"readiness_state": "ready", "confidence": 0.92},
    )
    _write_json(
        pipeline_root / "task_scope_record.json",
        {
            "task_statement": "Open and close the fridge door",
            "target_object_ids": ["1"],
            "articulation_required_ids": ["1"],
            "task_zone": {"center": [1.0, 0.0, 1.0]},
            "success_criteria": ["door opens", "door closes"],
        },
    )
    advanced_dir = pipeline_root / "advanced_geometry"
    advanced_dir.mkdir(parents=True, exist_ok=True)
    for name in ("3dgs_compressed.ply", "labels.json", "structure.json", "task_targets.synthetic.json"):
        (advanced_dir / name).write_text("{}" if name.endswith(".json") else "ply\n", encoding="utf-8")
    _write_json(advanced_dir / "advanced_geometry_bundle.json", {"schema_version": "v1"})
    scene_memory_dir = pipeline_root / "scene_memory"
    adapter_dir = scene_memory_dir / "adapter_manifests"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    _write_json(scene_memory_dir / "scene_memory_manifest.json", {"schema_version": "v1"})
    _write_json(scene_memory_dir / "scene_memory_readiness.json", {"schema_version": "v1", "status": "ready"})
    _write_json(
        scene_memory_dir / "conditioning_bundle.json",
        {
            "schema_version": "v1",
            "raw_video_uri": "gs://bucket/scenes/scene_eval/captures/cap_eval/raw/walkthrough.mov",
            "arkit": {
                "poses_uri": "gs://bucket/scenes/scene_eval/captures/cap_eval/raw/arkit/poses.jsonl",
                "intrinsics_uri": "gs://bucket/scenes/scene_eval/captures/cap_eval/raw/arkit/intrinsics.json",
            },
        },
    )
    _write_json(
        adapter_dir / "gen3c.json",
        {
            "schema_version": "v1",
            "status": "available_stage1_remote",
            "execution_mode": "remote_service",
            "required_conditioning": ["camera_poses", "intrinsics", "depth_or_explicit_geometry"],
            "service_contract_version": "stage1_world_model_remote_v1",
        },
    )
    _write_json(
        adapter_dir / "neoverse.json",
        {
            "schema_version": "v1",
            "status": "available_stage1_local",
            "execution_mode": "local_gpu_runtime",
            "required_conditioning": ["rgb_video"],
            "service_contract_version": "stage1_world_model_local_v1",
        },
    )
    _write_json(
        adapter_dir / "cosmos_transfer.json",
        {
            "schema_version": "v1",
            "status": "planned_phase3",
            "execution_mode": "planned_phase3",
            "required_conditioning": ["depth", "segmentation", "edge"],
            "service_contract_version": "reserved_phase3",
        },
    )
    _write_json(
        pipeline_root / "preview_simulation" / "preview_simulation_manifest.json",
        {
            "schema_version": "v1",
            "status": "prep_ready",
            "canonical_artifact_uri": "gs://bucket/scenes/scene_eval/captures/cap_eval/pipeline/scene_memory/scene_memory_manifest.json",
            "presentation_artifact_uri": "gs://bucket/scenes/scene_eval/captures/cap_eval/pipeline/preview_simulation/preview_simulation_manifest.json",
            "authoritative_record": False,
        },
    )
    _write_json(
        pipeline_root / "presentation_world" / "presentation_world_manifest.json",
        {
            "schema_version": "v1",
            "status": "available",
            "canonical_artifact_uri": "gs://bucket/scenes/scene_eval/captures/cap_eval/pipeline/scene_memory/scene_memory_manifest.json",
            "presentation_artifact_uri": "gs://bucket/scenes/scene_eval/captures/cap_eval/pipeline/presentation_world/presentation_world_manifest.json",
            "authoritative_record": False,
        },
    )
    _write_json(
        pipeline_root / "presentation_world" / "runtime_demo_manifest.json",
        {
            "schema_version": "v1",
            "status": "demo_ready",
            "canonical_artifact_uri": "gs://bucket/scenes/scene_eval/captures/cap_eval/pipeline/scene_memory/scene_memory_manifest.json",
            "presentation_artifact_uri": "gs://bucket/scenes/scene_eval/captures/cap_eval/pipeline/presentation_world/runtime_demo_manifest.json",
            "authoritative_record": False,
        },
    )
    return capture_root


def _configure_runtime_client(monkeypatch) -> None:
    monkeypatch.setenv("NEOVERSE_RUNTIME_SERVICE_URL", "http://runtime.local")

    class _FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def build_site_world(self, spec):
            return {
                "schema_version": "v1",
                "site_world_id": spec["site_world_id"],
                "build_id": "build-1",
                "scene_id": spec["scene_id"],
                "capture_id": spec["capture_id"],
                "status": "ready",
                "runtime_base_url": "http://runtime.local",
                "websocket_base_url": "ws://runtime.local",
                "vm_instance_id": "vast-123",
                "supported_cameras": ["head_rgb", "wrist_rgb"],
                "scenario_catalog": spec["scenario_catalog"],
                "start_state_catalog": spec["start_state_catalog"],
                "task_catalog": spec["task_catalog"],
                "robot_profiles": spec["robot_profiles"],
                "runtime_capabilities": {
                    "supports_step_rollout": True,
                    "supports_batch_rollout": True,
                    "supports_camera_views": True,
                    "supports_stream": True,
                },
                "health": {
                    "schema_version": "v1",
                    "site_world_id": spec["site_world_id"],
                    "build_id": "build-1",
                    "healthy": True,
                    "launchable": True,
                    "status": "healthy",
                    "blockers": [],
                    "warnings": [],
                    "last_heartbeat_at": "2026-03-12T00:00:00Z",
                },
            }

        def get_site_world_health(self, _site_world_id):
            return {
                "schema_version": "v1",
                "site_world_id": "siteworld",
                "build_id": "build-1",
                "healthy": True,
                "launchable": True,
                "status": "healthy",
                "blockers": [],
                "warnings": [],
                "last_heartbeat_at": "2026-03-12T00:00:00Z",
            }

        def create_session(self, *_args, **_kwargs):
            return {"session_id": "runtime-session-1", "build_id": "build-1"}

        def reset_session(self, *_args, **_kwargs):
            return {"episode": {"episodeId": "runtime-session-1"}}

    monkeypatch.setattr("blueprint_pipeline.evaluation_prep_stage.SiteWorldRuntimeServiceClient", _FakeClient)


def test_evaluation_prep_stage_writes_required_contract(tmp_path: Path, monkeypatch) -> None:
    capture_root = _build_capture(tmp_path)
    _configure_runtime_client(monkeypatch)

    result = run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rich_handoff = json.loads((capture_root / "pipeline" / "evaluation_prep" / "qualified_opportunity_handoff.json").read_text(encoding="utf-8"))
    anchors = json.loads((capture_root / "pipeline" / "evaluation_prep" / "task_anchor_manifest.json").read_text(encoding="utf-8"))
    site_world_spec = json.loads((capture_root / "pipeline" / "evaluation_prep" / "site_world_spec.json").read_text(encoding="utf-8"))
    site_world_registration = json.loads((capture_root / "pipeline" / "evaluation_prep" / "site_world_registration.json").read_text(encoding="utf-8"))
    site_world_health = json.loads((capture_root / "pipeline" / "evaluation_prep" / "site_world_health.json").read_text(encoding="utf-8"))
    summary = json.loads((capture_root / "pipeline" / "evaluation_prep" / "evaluation_prep_summary.json").read_text(encoding="utf-8"))
    scene_memory_bundle = json.loads((capture_root / "pipeline" / "evaluation_prep" / "scene_memory_bundle_manifest.json").read_text(encoding="utf-8"))
    protected_regions = json.loads((capture_root / "pipeline" / "evaluation_prep" / "protected_regions_manifest.json").read_text(encoding="utf-8"))
    hosted_session_runtime = json.loads((capture_root / "pipeline" / "evaluation_prep" / "hosted_session_runtime_manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "ready_for_validation"
    assert manifest["world_model_classification"] == "validated_site_world"
    assert manifest["canonical_output"]["authoritative_record"] is True
    assert manifest["presentation_output"]["authoritative_record"] is False
    assert manifest["artifacts"]["qualified_opportunity_handoff"] == "qualified_opportunity_handoff.json"
    assert manifest["artifacts"]["scene_memory_bundle_manifest"] == "scene_memory_bundle_manifest.json"
    assert manifest["artifacts"]["site_world_spec"] == "site_world_spec.json"
    assert manifest["artifacts"]["site_world_registration"] == "site_world_registration.json"
    assert manifest["artifacts"]["site_world_health"] == "site_world_health.json"
    assert manifest["artifacts"]["protected_regions_manifest"] == "protected_regions_manifest.json"
    assert manifest["artifacts"]["canonical_render_policy"] == "canonical_render_policy.json"
    assert manifest["artifacts"]["presentation_variance_policy"] == "presentation_variance_policy.json"
    assert manifest["artifacts"]["hosted_session_runtime_manifest"] == "hosted_session_runtime_manifest.json"
    assert manifest["artifacts"]["presentation_world_manifest"] == "../presentation_world/presentation_world_manifest.json"
    assert manifest["artifacts"]["runtime_demo_manifest"] == "../presentation_world/runtime_demo_manifest.json"
    assert scene_memory_bundle["presentation_world_manifest_path"] == "../presentation_world/presentation_world_manifest.json"
    assert scene_memory_bundle["runtime_demo_manifest_path"] == "../presentation_world/runtime_demo_manifest.json"
    assert scene_memory_bundle["protected_regions_manifest_path"] == "protected_regions_manifest.json"
    assert scene_memory_bundle["canonical_render_policy_path"] == "canonical_render_policy.json"
    assert scene_memory_bundle["presentation_variance_policy_path"] == "presentation_variance_policy.json"
    assert scene_memory_bundle["site_world_spec_path"] == "site_world_spec.json"
    assert rich_handoff["qualification_state"] == "ready"
    assert rich_handoff["downstream_evaluation_eligibility"] is True
    assert rich_handoff["scene_memory_package"]["scene_memory_manifest_path"] == "../scene_memory/scene_memory_manifest.json"
    assert rich_handoff["scene_memory_package"]["presentation_world_manifest_path"] == "../presentation_world/presentation_world_manifest.json"
    assert anchors["tasks"][0]["target_object_ids"] == ["1"]
    assert anchors["tasks"][0]["task_critical"] is True
    assert site_world_spec["runtime_eligibility"]["launchable"] is True
    assert site_world_spec["canonical_output"]["authoritative_record"] is True
    assert site_world_spec["presentation_output"]["authoritative_record"] is False
    assert site_world_spec["task_catalog"][0]["task_id"] == "task-1"
    assert site_world_spec["task_catalog"][0]["task_critical"] is True
    assert site_world_spec["runtime_layer_policy"]["protected_region_locking"] is True
    assert site_world_spec["canonical_package_version"]
    assert scene_memory_bundle["canonical_package_version"] == site_world_spec["canonical_package_version"]
    assert hosted_session_runtime["canonical_package_version"] == site_world_spec["canonical_package_version"]
    assert hosted_session_runtime["runtime_capabilities"]["protected_region_locking"] is True
    assert protected_regions["regions"][0]["classification"] == "locked"
    assert protected_regions["regions"][0]["task_critical"] is True
    assert site_world_registration["status"] == "ready"
    assert site_world_registration["world_model_classification"] == "validated_site_world"
    assert site_world_registration["runtime_base_url"] == "http://runtime.local"
    assert site_world_registration["runtime_capabilities"]["supports_camera_views"] is True
    assert site_world_health["launchable"] is True
    assert site_world_health["world_model_classification"] == "validated_site_world"
    assert summary["task_count"] == 1
    assert summary["object_count"] == 1
    assert summary["site_world_status"] == "healthy"
    assert summary["world_model_classification"] == "validated_site_world"
    assert summary["canonical_package_version"] == site_world_spec["canonical_package_version"]


def test_evaluation_prep_stage_accepts_scene_memory_without_geometry_bundle(tmp_path: Path, monkeypatch) -> None:
    capture_root = _build_capture(tmp_path)
    _configure_runtime_client(monkeypatch)
    advanced_dir = capture_root / "pipeline" / "advanced_geometry"
    for path in advanced_dir.iterdir():
        path.unlink()
    advanced_dir.rmdir()

    result = run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    review_queue = json.loads((capture_root / "pipeline" / "evaluation_prep" / "review_queue.json").read_text(encoding="utf-8"))
    site_world_health = json.loads((capture_root / "pipeline" / "evaluation_prep" / "site_world_health.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "ready_for_validation"
    assert "geometry_bundle:missing" not in manifest["degradation_reasons"]
    assert any(item["kind"] == "incomplete_geometry_bundle" and item["severity"] == "low" for item in review_queue["items"])
    assert site_world_health["launchable"] is True


def test_evaluation_prep_stage_degrades_when_object_index_is_missing(tmp_path: Path) -> None:
    capture_root = _build_capture(tmp_path)
    (capture_root / "raw" / "object_index.json").unlink()

    result = run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    object_geometry = json.loads((capture_root / "pipeline" / "evaluation_prep" / "object_geometry_manifest.json").read_text(encoding="utf-8"))
    anchors = json.loads((capture_root / "pipeline" / "evaluation_prep" / "task_anchor_manifest.json").read_text(encoding="utf-8"))
    site_world_health = json.loads((capture_root / "pipeline" / "evaluation_prep" / "site_world_health.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "degraded_but_usable"
    assert "object_geometry:missing" in manifest["degradation_reasons"]
    assert object_geometry["status"] == "missing_object_index"
    assert object_geometry["objects"] == []
    assert anchors["tasks"][0]["target_object_ids"] == ["1"]
    assert site_world_health["launchable"] is False
    assert "object_index_path_missing" in site_world_health["warnings"]


def test_evaluation_prep_stage_marks_empty_object_index_as_prototype(tmp_path: Path) -> None:
    capture_root = _build_capture(tmp_path)
    _write_json(capture_root / "raw" / "object_index.json", {"objects": []})

    result = run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    object_geometry = json.loads((capture_root / "pipeline" / "evaluation_prep" / "object_geometry_manifest.json").read_text(encoding="utf-8"))
    protected_regions = json.loads((capture_root / "pipeline" / "evaluation_prep" / "protected_regions_manifest.json").read_text(encoding="utf-8"))
    site_world_spec = json.loads((capture_root / "pipeline" / "evaluation_prep" / "site_world_spec.json").read_text(encoding="utf-8"))
    site_world_health = json.loads((capture_root / "pipeline" / "evaluation_prep" / "site_world_health.json").read_text(encoding="utf-8"))

    assert manifest["world_model_classification"] == "prototype_demo"
    assert object_geometry["status"] == "empty_object_index"
    assert object_geometry["object_index_present"] is True
    assert object_geometry["object_index_entry_count"] == 0
    assert protected_regions["grounding_status"] == "ungrounded"
    assert protected_regions["ungrounded_reason"] == "empty_object_index"
    assert protected_regions["regions"] == []
    assert site_world_spec["grounding_status"] == "ungrounded"
    assert site_world_spec["runtime_layer_policy"]["grounding_status"] == "ungrounded"
    assert site_world_health["world_model_classification"] == "prototype_demo"
