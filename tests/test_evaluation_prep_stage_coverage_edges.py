from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import blueprint_pipeline.evaluation_prep_stage as eps
from blueprint_pipeline.common import PipelineError, write_json
from blueprint_pipeline.world_model_policy import WorldModelPolicy


def _context(tmp_path: Path):
    capture_root = tmp_path / "storage" / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_root = capture_root / "pipeline"
    raw_root = capture_root / "raw"
    pipeline_root.mkdir(parents=True)
    raw_root.mkdir(parents=True)
    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor_path.write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        capture_root=capture_root,
        raw_root=raw_root,
        pipeline_root=pipeline_root,
        descriptor_path=descriptor_path,
        storage_root=tmp_path / "storage" / "local-blueprint",
        bucket="local-blueprint",
        scene_id="scene-1",
        capture_id="capture-1",
        capture_prefix="scenes/scene-1/captures/capture-1",
    )


def _runtime_status() -> dict[str, Any]:
    return {
        "launchable": True,
        "runtime_base_url": "http://runtime.test",
        "websocket_base_url": "ws://runtime.test",
        "blockers": [],
        "warnings": [],
        "grounding_status": "grounded",
        "object_index_backend_blockers": [],
    }


def _runtime_spec() -> dict[str, Any]:
    return {
        "site_world_id": "siteworld-test",
        "site_submission_id": "site-submission-1",
        "canonical_package_version": "local-version",
        "task_catalog": [{"id": "task-1", "task_id": "task-1", "task_text": "Pick object"}],
        "scenario_catalog": [{"id": "scenario-1"}],
        "start_state_catalog": [{"id": "start-1"}],
        "robot_profiles": [{"id": "mobile_manipulator_rgb_v1"}],
        "primary_runtime_backend": "site_world_runtime",
        "canonical_world_model": {"status": "ready"},
        "native_world_model_status": "primary_ready",
        "native_world_model_primary": True,
        "provider_fallback_preview_status": "not_requested",
        "provider_fallback_only": False,
        "artifact_families": {},
        "world_model_backend": "site_world_runtime",
        "scene_representation": "site_world_runtime_video_world_model_v1",
        "runtime_render_source": "site_world_runtime_full_capture",
        "fallback_mode": "none",
    }


def test_small_helper_edges(tmp_path: Path) -> None:
    context = _context(tmp_path)
    eval_dir = context.pipeline_root / "evaluation_prep"
    eval_dir.mkdir()

    assert eps._string_list({"fallback": True}) == ["{'fallback': True}"]
    assert eps._adapter_manifest_details({"site_world_runtime_adapter_manifest_path": ""}, eval_dir=eval_dir) == {}
    assert eps._task_category("Navigate to dock") == "navigate"
    assert eps._task_category("Pick up tote") == "pick"
    assert eps._task_category("Inspect shelf") == "generic"
    assert eps._default_task_id({"tasks": [{"task_id": "scope-task"}]}, {}, "capture") == "scope-task"
    assert eps._default_task_id({"tasks": [{}]}, {}, "capture") == "capture"
    assert eps._default_task_text({"task_statement": "Open cabinet"}, {}, "capture") == "Open cabinet"
    assert eps._default_task_text({}, {}, "capture") == "capture"
    assert eps._real_path_from_eval_dir(eval_dir, "") is None
    assert eps._runtime_readiness_state(launchable=False, blockers=[]) == "incomplete"
    assert eps._gate(["bad"], ["ok", "bad:thing"]) is False


def test_task_run_entries_and_anchor_manifest_cover_task_entry_paths(tmp_path: Path) -> None:
    context = _context(tmp_path)
    assert eps._load_task_run_entries(context.capture_root) == []
    write_json(
        context.pipeline_root / "task_scope_record.json",
        {
            "tasks": [{"task_id": "scope-task"}],
            "target_object_ids": ["missing-object"],
            "articulation_required_ids": ["hinge-1"],
            "task_zone": {"center": [1, 2, 3]},
        },
    )
    write_json(
        context.pipeline_root / "task_run_manifest.json",
        {
            "groups": {
                "bad": {"not": "a list"},
                "pick": [
                    "skip-me",
                    {
                        "capture_root": str(context.capture_root),
                        "capture_id": "capture-1",
                        "task_text": "Pick up tote",
                    },
                ],
            }
        },
    )

    entries = eps._load_task_run_entries(context.capture_root)
    assert entries == [
        {
            "task_id": "scope-task",
            "task_text": "Pick up tote",
            "task_category": "pick",
            "capture_root": str(context.capture_root.resolve()),
            "capture_id": "capture-1",
            "target_object_ids": ["missing-object"],
            "articulation_required_ids": ["hinge-1"],
        }
    ]

    manifest = eps._build_task_anchor_manifest(
        capture_root=context.capture_root,
        handoff={},
        scope_record={"task_zone": {"center": [1, 2, 3]}},
        task_run_entries=entries,
        object_geometry_manifest={"objects": []},
    )

    task = manifest["tasks"][0]
    assert task["task_id"] == "scope-task"
    assert task["task_zone"]["center"] == [1.0, 2.0, 3.0]
    assert task["goal_zone"] == [1.0, 2.0, 3.0]
    assert task["provenance"]["grounding_level"] == "reconstructed"


def test_bundle_handoff_review_and_runtime_variant_edges(tmp_path: Path) -> None:
    context = _context(tmp_path)
    eval_dir = context.pipeline_root / "evaluation_prep"
    eval_dir.mkdir()
    advanced_dir = context.pipeline_root / "advanced_geometry"
    advanced_dir.mkdir()
    for name in ("3dgs_compressed.ply", "labels.json", "structure.json", "task_targets.synthetic.json"):
        (advanced_dir / name).write_text("{}", encoding="utf-8")

    geometry_bundle = eps._build_geometry_bundle_manifest(pipeline_dir=context.pipeline_root, eval_dir=eval_dir)
    assert geometry_bundle["status"] == "complete"
    assert geometry_bundle["bundle_path"] == "../advanced_geometry"

    normalized = eps._normalize_rich_handoff(
        handoff={},
        scope_record={"task_statement": "Open door", "blockers": ["blocked aisle"]},
        qualification_record={"readiness_state": "ready"},
        capture_root=context.capture_root,
        geometry_bundle_manifest=geometry_bundle,
        scene_memory_bundle_manifest={"scene_memory_manifest_path": "scene_memory/scene_memory_manifest.json"},
    )
    assert normalized["scoped_task_definition"]["scoped_task_statement"] == "Open door"
    assert normalized["site_constraints"]["known_blockers"] == ["blocked aisle"]
    assert "missing_site_submission_id" in normalized["upstream_link_blockers"]
    assert normalized["geometry_package"]["ply_path"].endswith("3dgs_compressed.ply")

    review = eps._build_review_queue(
        object_geometry_manifest={"objects": ["skip", {}, {"object_id": "target-1"}, {"object_id": "other-1"}]},
        task_anchor_manifest={"tasks": [{"target_object_ids": ["target-1"]}]},
        simready_validation=None,
        geometry_bundle_manifest={"status": "complete"},
        scene_memory_bundle_manifest={"status": "missing"},
    )
    kinds = {item["kind"] for item in review["items"]}
    assert {"missing_selected_views", "missing_support_surfaces", "missing_collision_hulls"} <= kinds
    assert "incomplete_scene_memory_bundle" in kinds

    write_json(
        context.pipeline_root / "cosmos_zero_shot_validation" / "cosmos_zero_shot_benchmark.json",
        {"status": "failed", "reason": "benchmark down"},
    )
    write_json(context.pipeline_root / "cosmos_training_export" / "manifest.json", {"status": "ready"})
    write_json(
        context.pipeline_root / "cosmos_training_export" / "training_run_manifest.json",
        {"status": "failed", "reason": "training down"},
    )
    variants = eps._build_runtime_backend_variants(
        context=context,
        eval_dir=eval_dir,
        pipeline_dir=context.pipeline_root,
        scene_memory_bundle_manifest={"site_world_runtime_adapter_manifest_path": ""},
        native_semantics={"native_world_model_primary": False, "native_world_model_status": "not_ready"},
    )
    assert "benchmark down" in variants["cosmos_zero_shot_i2w"]["blockers"]
    assert "training down" in variants["cosmos_predict_lora_adapter"]["blockers"]
    assert "native_world_model_not_primary" in variants["cosmos_predict_lora_adapter"]["blockers"]


def test_world_model_descriptor_and_readiness_edges(tmp_path: Path) -> None:
    context = _context(tmp_path)
    write_json(
        context.pipeline_root / "presentation_world" / "authoritative_runtime_render_manifest.json",
        {
            "primary_asset_path": "/tmp/site.glb",
            "primary_asset_uri": "gs://bucket/site.glb",
            "supporting_assets": [{"name": "support"}, "skip"],
        },
    )
    ready = eps._canonical_world_model_payload(context=context, capture_orientation={"display_orientation": "landscape"})
    assert ready["status"] == "ready"
    assert ready["supporting_assets"] == [{"name": "support"}]
    assert eps._primary_runtime_render_descriptor(
        conditioning_bundle={},
        local_paths={},
        canonical_world_model=ready,
    )["scene_representation"] == "site_world_runtime_video_world_model_v1"
    assert eps._native_world_model_semantics(
        context=context,
        canonical_world_model=ready,
        runtime_render_descriptor={"runtime_render_source": "ignored"},
        scene_memory_bundle_manifest={},
    )["native_world_model_path"] == "authoritative_native_render"

    (context.pipeline_root / "presentation_world" / "authoritative_runtime_render_manifest.json").unlink()
    advanced_bundle = context.pipeline_root / "advanced_geometry" / "advanced_geometry_bundle.json"
    advanced_bundle.parent.mkdir(exist_ok=True)
    advanced_bundle.write_text("{}", encoding="utf-8")
    missing = eps._canonical_world_model_payload(context=context, capture_orientation={})
    assert missing["status"] == "missing"
    assert missing["supporting_assets"][0]["uri"].endswith("/advanced_geometry/advanced_geometry_bundle.json")

    geometry_descriptor = eps._primary_runtime_render_descriptor(
        conditioning_bundle={
            "raw_video_uri": "gs://bucket/raw.mov",
            "geometry": {
                "poses_uri": "gs://bucket/poses.jsonl",
                "intrinsics_uri": "gs://bucket/intrinsics.json",
                "depth_manifest_uri": "gs://bucket/depth.json",
            },
        },
        local_paths={},
        canonical_world_model=missing,
    )
    assert geometry_descriptor["fallback_mode"] == "geometry_lane_conditioning"
    assert eps._primary_runtime_render_descriptor(
        conditioning_bundle={},
        local_paths={},
        canonical_world_model=missing,
    )["runtime_render_source"] == "pending_world_model_service"
    assert eps._native_world_model_semantics(
        context=context,
        canonical_world_model=missing,
        runtime_render_descriptor={"runtime_render_source": "geometry_conditioned_capture"},
        scene_memory_bundle_manifest={},
    )["native_world_model_path"] == "geometry_conditioned_native_path"
    assert eps._native_world_model_semantics(
        context=context,
        canonical_world_model=missing,
        runtime_render_descriptor={"runtime_render_source": "unavailable"},
        scene_memory_bundle_manifest={"preview_simulation_manifest_path": "preview.json"},
    )["provider_fallback_only"] is True


def test_status_descriptor_presentation_and_provenance_edges(tmp_path: Path) -> None:
    context = _context(tmp_path)
    missing_artifact = context.pipeline_root / "missing.json"
    status = eps._canonical_site_world_runtime_status(
        qualification_state="ready",
        downstream_evaluation_eligibility=True,
        scene_memory_bundle_manifest={"status": "complete"},
        object_geometry_manifest={},
        protected_regions_manifest={},
        required_runtime_artifact_paths=[missing_artifact],
        runtime_service_url="http://runtime.test",
    )
    assert f"missing_runtime_artifact:{missing_artifact.name}" in status["blockers"]
    assert eps._descriptor_scene_memory_capture({"scene_memory_capture": {"sensor_availability": {"rgb": True}}}) == {
        "sensor_availability": {"rgb": True}
    }
    assert eps._descriptor_scene_memory_capture({}) == {}
    assert eps._descriptor_capture_orientation({"metadata": {"capture_orientation": {"display": "portrait"}}}) == {
        "display": "portrait"
    }
    assert eps._descriptor_capture_orientation({}, {"capture_orientation": {"display": "landscape"}}) == {
        "display": "landscape"
    }
    assert eps._descriptor_capture_orientation({}) == {}
    assert eps._presentation_demo_readiness({"ui_base_url": "https://demo.example"}) == {
        "readiness_state": "ready",
        "blockers": [],
    }
    refreshed = eps._refresh_presentation_contract_payload(
        payload={},
        context=context,
        canonical_package_version="version-1",
        derivation_policy={"policy": "limited"},
    )
    assert refreshed["derivation_policy"] == {"policy": "limited"}
    assert eps._object_geometry_has_provenance({"objects": []}) is False
    assert eps._object_geometry_has_provenance({"objects": ["skip", {"object_id": "missing-provenance"}]}) is False


def test_site_world_spec_skips_non_mapping_tasks(tmp_path: Path) -> None:
    context = _context(tmp_path)
    eval_dir = context.pipeline_root / "evaluation_prep"
    eval_dir.mkdir()
    spec = eps._build_site_world_spec(
        context=context,
        eval_dir=eval_dir,
        normalized_handoff={"site_submission_id": "site-submission-1", "qualification_state": "ready"},
        scene_memory_bundle_manifest={},
        object_geometry_manifest={},
        task_anchor_manifest={"tasks": ["skip"]},
        task_run_manifest={},
        protected_regions_manifest={},
        canonical_render_policy={},
        presentation_variance_policy={},
        canonical_runtime_status={"launchable": False, "blockers": ["missing_runtime_service_url"]},
    )
    assert spec["task_catalog"] == []


class _VerificationMismatchClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def register_site_world_package(self, *, spec: Mapping[str, Any], registration: Mapping[str, Any], health: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **dict(registration),
            "status": "ready",
            "build_id": "local-build",
            "runtime_base_url": "",
            "health": {**dict(health), "healthy": False, "launchable": False, "status": "degraded"},
        }

    def get_site_world(self, _site_world_id: str) -> dict[str, Any]:
        return {"build_id": "remote-build", "canonical_package_version": "remote-version"}


class _SmokeFailureClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def register_site_world_package(self, *, spec: Mapping[str, Any], registration: Mapping[str, Any], health: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **dict(registration),
            "status": "ready",
            "build_id": "local-build",
            "runtime_base_url": "http://runtime.test",
            "health": {**dict(health), "healthy": True, "launchable": True, "status": "healthy"},
        }

    def get_site_world(self, _site_world_id: str) -> dict[str, Any]:
        return {
            "build_id": "local-build",
            "canonical_package_version": "local-version",
            "runtime_base_url": "http://runtime.test",
        }

    def create_session(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("session down")


class _ReadyClient(_SmokeFailureClient):
    def create_session(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"session_id": "session-1"}

    def reset_session(self, _session_id: str) -> dict[str, Any]:
        return {"status": "reset"}

    def get_site_world_health(self, _site_world_id: str) -> dict[str, Any]:
        return {"healthy": True, "launchable": True, "status": "healthy", "runtime_base_url": "http://runtime.test"}


def test_runtime_records_cover_verification_blockers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(eps, "SiteWorldRuntimeServiceClient", _VerificationMismatchClient)

    registration, health = eps._build_site_world_runtime_records(
        context=context,
        spec=_runtime_spec(),
        canonical_runtime_status=_runtime_status(),
    )

    assert registration["status"] == "blocked"
    assert "runtime_registered_build_id_mismatch" in registration["blockers"]
    assert "runtime_registered_package_version_mismatch" in registration["blockers"]
    assert "runtime_base_url_missing_after_registration" in registration["blockers"]
    assert "runtime_health_not_launchable" in health["blockers"]
    assert health["runtime_capabilities"]["supports_step_rollout"] is False


def test_runtime_records_cover_smoke_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(eps, "SiteWorldRuntimeServiceClient", _SmokeFailureClient)

    registration, health = eps._build_site_world_runtime_records(
        context=context,
        spec=_runtime_spec(),
        canonical_runtime_status=_runtime_status(),
    )

    assert registration["runtime_smoke"]["status"] == "failed"
    assert health["runtime_capabilities"]["supports_step_rollout"] is False
    assert any(str(item).startswith("runtime_smoke_failed:session down") for item in health["blockers"])


def test_runtime_records_cover_missing_catalogs_and_required_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = _context(tmp_path)
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")
    monkeypatch.setattr(eps, "SiteWorldRuntimeServiceClient", _ReadyClient)
    spec = {**_runtime_spec(), "task_catalog": [], "scenario_catalog": [], "start_state_catalog": []}

    registration, health = eps._build_site_world_runtime_records(
        context=context,
        spec=spec,
        canonical_runtime_status=_runtime_status(),
    )

    assert registration["runtime_smoke"]["status"] == "blocked"
    assert "runtime_smoke_catalogs_missing" in registration["runtime_smoke"]["blockers"]
    assert "runtime_session_smoke_required" in registration["blockers"]
    assert health["launchable"] is False


def test_hosted_session_runtime_manifest_edge_catalogs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = _context(tmp_path)
    manifest = eps._build_hosted_session_runtime_manifest(
        context=context,
        normalized_handoff={"site_submission_id": "site-submission-1"},
        scene_memory_bundle_manifest={},
        task_anchor_manifest={"tasks": ["skip"]},
        task_run_manifest={"start_states": ["dock", "", "dock"]},
        canonical_runtime_status={"blockers": []},
    )
    assert manifest["start_state_catalog"] == [
        {"id": "start_dock", "name": "dock", "task_id": None, "source": "task_run_manifest"}
    ]
    assert "missing_task_anchor_manifest" in manifest["blockers"]
    assert "no_launchable_stage1_backend" in manifest["blockers"]

    monkeypatch.setattr(eps, "_build_runtime_backend_variants", lambda **_kwargs: {})
    empty_backend_manifest = eps._build_hosted_session_runtime_manifest(
        context=context,
        normalized_handoff={},
        scene_memory_bundle_manifest={},
        task_anchor_manifest={"tasks": []},
        task_run_manifest={},
        canonical_runtime_status={"blockers": []},
    )
    assert empty_backend_manifest["start_state_catalog"][0]["source"] == "runtime_default"
    assert "runtime_manifest_only" in empty_backend_manifest["blockers"]


def test_benchmark_recapture_and_validation_summary_edges(tmp_path: Path) -> None:
    context = _context(tmp_path)
    benchmark = eps._build_benchmark_suite_manifest(
        normalized_handoff={"scoped_task_definition": {"success_criteria": ["Done"]}},
        qualification_record={"risks": [{"detail": "Watch glare"}]},
        task_anchor_manifest={"tasks": ["skip"]},
        task_run_manifest={},
    )
    assert benchmark["status"] == "missing"

    previous = (
        context.capture_root.parent
        / "capture-0"
        / "pipeline"
        / "evaluation_prep"
        / "site_normalization_package.json"
    )
    write_json(
        context.capture_root / "pipeline" / "evaluation_prep" / "site_normalization_package.json",
        {"capture_id": "capture-1"},
    )
    write_json(
        previous,
        {
            "capture_id": "capture-0",
            "qualification_state": "ready",
            "scoped_task_definition": {"scoped_task_statement": "Old task"},
            "site_constraints": {"known_blockers": []},
            "measurements": {"minimum_route_width_m": 1.2},
        },
    )
    found = eps._find_previous_site_normalization_path(
        capture_root=context.capture_root,
        current_capture_id="capture-1",
    )
    assert found == previous
    diff = eps._build_recapture_diff(
        capture_root=context.capture_root,
        current_capture_id="capture-1",
        site_normalization_package={
            "qualification_state": "ready",
            "scoped_task_definition": {"scoped_task_statement": "New task"},
            "site_constraints": {"known_blockers": ["blocked"]},
            "measurements": {"minimum_route_width_m": 0.9},
        },
        benchmark_suite_manifest={"task_count": 1},
    )
    assert diff["status"] == "changed"
    assert diff["previous_capture_id"] == "capture-0"

    summary = eps._world_model_validation_summary(
        policy=WorldModelPolicy(),
        site_world_health={"launchable": True, "blockers": []},
        launchable_export_bundle={"bundles": {"world_model_runtime": {"launchable": True}}},
        runtime_demo_manifest={"ui_base_url": "https://demo.example"},
        object_geometry_manifest={
            "objects": [
                {
                    "object_id": "target-1",
                    "provenance": {"grounding_level": "reconstructed", "canonical_truth": True},
                }
            ]
        },
        geometry_bundle_manifest={"status": "complete"},
        scene_memory_bundle_manifest={
            "status": "complete",
            "geometry_summary_path": "../geometry/geometry_summary.json",
            "geometry_summary": {
                "geometry_source": "video_to_world",
                "ready_for_world_model": True,
                "provider_native_result": True,
                "site_frame_available": True,
                "scale_resolved": True,
                "geometry_live_ready": True,
            },
        },
        task_anchor_manifest={"tasks": [{"target_object_ids": ["target-1"]}]},
        review_queue={"items": []},
    )
    assert summary["world_model_classification"] == "validated_site_world"


def _write_minimal_evaluation_capture(capture_root: Path) -> None:
    pipeline_root = capture_root / "pipeline"
    raw_root = capture_root / "raw"
    pipeline_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(exist_ok=True)
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps({"metadata": {"site_identity": {"name": "Test site"}, "adjacent_systems": ["wms"]}}),
        encoding="utf-8",
    )
    (raw_root / "walkthrough.mov").write_bytes(b"video")
    write_json(
        pipeline_root / "opportunity_handoff.json",
        {
            "site_submission_id": "site-submission-1",
            "buyer_request_id": "buyer-request-1",
            "capture_job_id": "capture-job-1",
            "qualification_state": "ready",
            "downstream_evaluation_eligibility": True,
            "scoped_task_definition": {
                "task_id": "task-1",
                "scoped_task_statement": "Pick up tote",
                "success_criteria": ["Tote moved"],
            },
        },
    )
    write_json(
        pipeline_root / "qualification_record.json",
        {
            "readiness_state": "ready",
            "measurements": {"minimum_route_width_m": 1.2, "maximum_target_reach_m": 0.8},
            "risks": [],
            "blockers": [],
        },
    )
    write_json(
        pipeline_root / "task_scope_record.json",
        {
            "task_statement": "Pick up tote",
            "tasks": [{"task_id": "task-1"}],
            "target_object_ids": ["target-1"],
            "task_zone": {"center": [0.0, 0.0, 0.0]},
        },
    )
    write_json(
        pipeline_root / "task_run_manifest.json",
        {
            "groups": {
                "pick": [
                    {
                        "capture_root": str(capture_root),
                        "capture_id": "capture-1",
                        "task_text": "Pick up tote",
                    }
                ]
            },
            "start_states": ["ready"],
        },
    )
    write_json(
        pipeline_root / "marble_sim_assets" / "marble_simready_bridge.json",
        {"status": "bridge_ready"},
    )


def test_run_stage_covers_existing_manifest_marble_bridge_cosmos_and_degradation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    _write_minimal_evaluation_capture(context.capture_root)
    monkeypatch.setenv("BLUEPRINT_ALLOW_LEGACY_COSMOS_EVAL_PREP_EXPORT", "true")
    monkeypatch.setattr(
        eps,
        "_resolve_object_geometry_manifest",
        lambda **_kwargs: {
            "objects": [
                {
                    "object_id": "target-1",
                    "placement_bbox": {"center": [0.0, 0.0, 0.0]},
                    "provenance": {"grounding_level": "reconstructed", "canonical_truth": True},
                }
            ]
        },
    )
    for name in (
        "simulation_automation_evaluation_prep_surface",
        "palatial_physready_evaluation_prep_surface",
        "site_eval_director_evaluation_prep_surface",
        "robot_eval_job_evaluation_prep_surface",
    ):
        monkeypatch.setattr(
            eps,
            name,
            lambda **_kwargs: {
                "status": "stubbed",
                "artifacts": {},
                "artifact_uris": {},
                "simulator_execution_proven": False,
                "robot_readiness_proven": False,
                "job_count": 0,
                "model_derived_support_assets_present": False,
                "live_provider_calls_performed": False,
            },
        )
    monkeypatch.setattr(eps, "sync_webapp_evaluation_prep", lambda **_kwargs: {"status": "skipped"})
    monkeypatch.setattr(eps, "write_alpha_readiness_summary", lambda **_kwargs: {"status": "skipped"})

    import blueprint_pipeline.robot_eval_dataset as robot_eval_dataset
    from blueprint_pipeline.synthesis import cosmos_training_export

    monkeypatch.setattr(
        robot_eval_dataset,
        "build_real_site_robot_eval_dataset",
        lambda **_kwargs: {
            "status": "stubbed",
            "dataset_statuses": {},
            "recorded_trace_eval_status": "missing",
            "prediction_vs_actual_status": "missing",
            "rights_packet_status": "missing",
        },
    )
    monkeypatch.setattr(
        cosmos_training_export,
        "export_cosmos_training_substrate",
        lambda **_kwargs: {"status": "exported"},
    )

    result = eps.run_evaluation_prep_stage(capture_root=context.capture_root, provider_name="manual")

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert result["marble_sim_assets"]["status"] == "bridge_ready"
    assert manifest["status"] == "degraded_but_usable"
    assert "scene_memory_bundle:partial" in manifest["degradation_reasons"]
    assert "geometry_bundle:missing" in manifest["degradation_reasons"]


def test_run_stage_raises_when_handoff_missing(tmp_path: Path) -> None:
    context = _context(tmp_path)

    with pytest.raises(PipelineError, match="Missing opportunity_handoff"):
        eps.run_evaluation_prep_stage(capture_root=context.capture_root)


def test_main_success_failure_and_module_entrypoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        eps,
        "run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": "prep.json", "status": "ready_for_validation"},
    )
    assert eps.main(["--capture-root", str(tmp_path), "--provider", "manual"]) == 0
    assert "manifest=prep.json" in capsys.readouterr().out

    def _raise(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(eps, "run_evaluation_prep_stage", _raise)
    assert eps.main(["--capture-root", str(tmp_path), "--provider", "manual"]) == 1
    assert "FAILED: boom" in capsys.readouterr().out

    capture_root = tmp_path / "storage" / "local-blueprint" / "scenes" / "scene-entry" / "captures" / "capture-entry"
    monkeypatch.setattr(sys, "argv", ["evaluation_prep_stage.py", "--capture-root", str(capture_root)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("blueprint_pipeline.evaluation_prep_stage", run_name="__main__")
    assert exc.value.code == 1
