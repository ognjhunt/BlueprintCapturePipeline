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
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene_eval",
            "capture_id": "cap_eval",
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
    _write_json(scene_memory_dir / "conditioning_bundle.json", {"schema_version": "v1"})
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
        {"schema_version": "v1", "status": "prep_ready"},
    )
    return capture_root


def test_evaluation_prep_stage_writes_required_contract(tmp_path: Path) -> None:
    capture_root = _build_capture(tmp_path)

    result = run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rich_handoff = json.loads((capture_root / "pipeline" / "evaluation_prep" / "qualified_opportunity_handoff.json").read_text(encoding="utf-8"))
    anchors = json.loads((capture_root / "pipeline" / "evaluation_prep" / "task_anchor_manifest.json").read_text(encoding="utf-8"))
    hosted_runtime = json.loads((capture_root / "pipeline" / "evaluation_prep" / "hosted_session_runtime_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((capture_root / "pipeline" / "evaluation_prep" / "evaluation_prep_summary.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "ready_for_validation"
    assert manifest["artifacts"]["qualified_opportunity_handoff"] == "qualified_opportunity_handoff.json"
    assert manifest["artifacts"]["scene_memory_bundle_manifest"] == "scene_memory_bundle_manifest.json"
    assert manifest["artifacts"]["hosted_session_runtime_manifest"] == "hosted_session_runtime_manifest.json"
    assert rich_handoff["qualification_state"] == "ready"
    assert rich_handoff["downstream_evaluation_eligibility"] is True
    assert rich_handoff["scene_memory_package"]["scene_memory_manifest_path"] == "../scene_memory/scene_memory_manifest.json"
    assert anchors["tasks"][0]["target_object_ids"] == ["1"]
    assert hosted_runtime["launchable"] is True
    assert hosted_runtime["default_backend"] == "neoverse"
    assert hosted_runtime["launchable_backends"] == ["neoverse", "gen3c"]
    assert hosted_runtime["task_catalog"][0]["id"] == "task-1"
    assert hosted_runtime["start_state_catalog"][0]["id"].startswith("start_")
    assert hosted_runtime["scenario_catalog"][0]["id"] == "scenario_preview_simulation_default"
    assert hosted_runtime["default_robot_profile_id"] == "mobile_manipulator_rgb_v1"
    assert hosted_runtime["robot_profiles"][0]["allowed_policy_adapters"] == ["openvla_oft", "pi05", "dreamzero"]
    assert hosted_runtime["export_defaults"] == [
        "observation_frames",
        "action_trace",
        "reward",
        "summary_metrics",
        "rollout_video",
        "rlds_dataset",
    ]
    assert hosted_runtime["runtime_capabilities"]["supports_camera_views"] is True
    assert hosted_runtime["backend_launch_requirements"]["cosmos_transfer"]["status"] == "planned_phase3"
    assert "task-1" in hosted_runtime["task_ids"]
    assert summary["task_count"] == 1
    assert summary["object_count"] == 1


def test_evaluation_prep_stage_accepts_scene_memory_without_geometry_bundle(tmp_path: Path) -> None:
    capture_root = _build_capture(tmp_path)
    advanced_dir = capture_root / "pipeline" / "advanced_geometry"
    for path in advanced_dir.iterdir():
        path.unlink()
    advanced_dir.rmdir()

    result = run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    review_queue = json.loads((capture_root / "pipeline" / "evaluation_prep" / "review_queue.json").read_text(encoding="utf-8"))
    hosted_runtime = json.loads((capture_root / "pipeline" / "evaluation_prep" / "hosted_session_runtime_manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "ready_for_validation"
    assert "geometry_bundle:missing" not in manifest["degradation_reasons"]
    assert any(item["kind"] == "incomplete_geometry_bundle" and item["severity"] == "low" for item in review_queue["items"])
    assert hosted_runtime["launchable"] is True


def test_evaluation_prep_stage_degrades_when_object_index_is_missing(tmp_path: Path) -> None:
    capture_root = _build_capture(tmp_path)
    (capture_root / "raw" / "object_index.json").unlink()

    result = run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    object_geometry = json.loads((capture_root / "pipeline" / "evaluation_prep" / "object_geometry_manifest.json").read_text(encoding="utf-8"))
    anchors = json.loads((capture_root / "pipeline" / "evaluation_prep" / "task_anchor_manifest.json").read_text(encoding="utf-8"))
    hosted_runtime = json.loads((capture_root / "pipeline" / "evaluation_prep" / "hosted_session_runtime_manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "degraded_but_usable"
    assert "object_geometry:missing" in manifest["degradation_reasons"]
    assert object_geometry["status"] == "missing_object_index"
    assert object_geometry["objects"] == []
    assert anchors["tasks"][0]["target_object_ids"] == ["1"]
    assert hosted_runtime["launchable"] is True
