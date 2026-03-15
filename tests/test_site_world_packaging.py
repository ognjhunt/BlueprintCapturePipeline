from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blueprint_contracts.site_world_contract import load_site_world_bundle, validate_site_world_bundle
from blueprint_pipeline.capture_orchestrator import PipelineConfig, run_capture_pipeline
from blueprint_pipeline.evaluation_prep_stage import run_evaluation_prep_stage
from blueprint_pipeline.materialization import build_capture_bundle_records, materialize_capture_bundle
from blueprint_pipeline.qualification import _presentation_bundle_status, _presentation_primary_asset


def _successful_capture_review() -> dict[str, object]:
    return {
        "schema_version": "v1",
        "review_type": "gemini_multimodal_capture_review",
        "status": "succeeded",
        "generated_at": "2026-03-15T00:00:00+00:00",
        "provider_name": "gemini",
        "provider_model": "gemini-2.5-pro",
        "review_mode": "video_primary_frames_fallback",
        "confidence": 0.88,
        "summary": "Capture supports downstream work.",
        "scores": {
            "coverage": 0.88,
            "visual_clarity": 0.84,
            "lighting_stability": 0.82,
            "motion_stability": 0.8,
            "task_understanding": 0.85,
            "world_model_fitness": 0.83,
            "payout_quality": 0.78,
        },
        "bonus_signals": {
            "complete_coverage": {"score": 0.9, "reason": "Coverage is complete."},
            "multi_pass": {"score": 0.7, "reason": "Multiple views are present."},
            "lidar_depth": {"score": 1.0, "reason": "Depth-backed capture quality is strong."},
            "steady_walkthrough": {"score": 0.85, "reason": "The walkthrough is steady."},
        },
        "findings": {
            "missing_views": [],
            "blur_observations": [],
            "lighting_observations": [],
            "occlusion_observations": [],
            "task_scope_notes": [],
            "blocker_summaries": [],
            "recapture_recommendations": [],
        },
        "recommendations": {
            "world_model_recommendation": "good_candidate",
            "payout_recommendation": "baseline",
        },
        "provenance": {"provider_name": "gemini", "provider_model": "gemini-2.5-pro"},
    }


class _HealthyRuntimeClient:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def register_site_world_package(self, *, spec, registration, health):  # type: ignore[no-untyped-def]
        site_world_id = str(registration.get("site_world_id") or "siteworld-test")
        return {
            **dict(registration),
            "schema_version": "v1",
            "status": "ready",
            "site_world_id": site_world_id,
            "runtime_base_url": "http://runtime.test",
            "websocket_base_url": "ws://runtime.test",
            "runtime_capabilities": {
                "supports_step_rollout": True,
                "supports_batch_rollout": True,
                "supports_camera_views": True,
                "supports_stream": False,
                "supports_rlds_export": True,
                "supports_preview_render": True,
                "protected_region_locking": True,
                "runtime_layer_compositing": True,
                "debug_render_outputs": True,
            },
            "health": {
                **dict(health),
                "schema_version": "v1",
                "site_world_id": site_world_id,
                "healthy": True,
                "launchable": True,
                "status": "healthy",
                "blockers": [],
                "warnings": [],
            },
        }

    def get_site_world_health(self, site_world_id: str):  # type: ignore[no-untyped-def]
        return {
            "schema_version": "v1",
            "site_world_id": site_world_id,
            "healthy": True,
            "launchable": True,
            "status": "healthy",
            "blockers": [],
            "warnings": [],
        }

    def create_session(self, site_world_id: str, **_kwargs):  # type: ignore[no-untyped-def]
        return {"site_world_id": site_world_id, "session_id": "session-test"}

    def reset_session(self, _session_id: str, **_kwargs):  # type: ignore[no-untyped-def]
        return {"status": "ok"}


def _write_backend_script(path: Path, *, mode: str) -> None:
    if mode == "success":
        body = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
output_path = Path(sys.argv[2])
objects = [
    {
        "id": "cabinet_0001",
        "object_id": "cabinet_0001",
        "label": "cabinet",
        "boundingBox": {
            "center": [0.0, 0.0, 0.75],
            "extents": [0.8, 0.45, 0.9],
            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "orientationQuaternion": [1.0, 0.0, 0.0, 0.0],
        },
        "mean_confidence": 0.95,
        "confidence": 0.95,
        "n_total_detections": 3,
        "n_frame_detections": 2,
        "reference_crop": "",
        "all_crops": [],
        "task_relevance": {"score": 0.9, "matched_terms": ["cabinet"]},
        "articulation_hints": {"interactive": True, "kind": "cabinet", "confidence": 0.82},
        "evidence_frames": [0, 1],
        "source_prompts": ["cabinet"],
        "provenance": {"grounding_level": "observed", "canonical_truth": True},
        "mean_box_px": {"area": 64000.0, "width": 240.0, "height": 260.0},
    },
    {
        "id": "aisle_0001",
        "object_id": "aisle_0001",
        "label": "aisle",
        "boundingBox": {
            "center": [0.0, 0.0, 0.0],
            "extents": [1.2, 1.2, 0.1],
            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "orientationQuaternion": [1.0, 0.0, 0.0, 0.0],
        },
        "mean_confidence": 0.9,
        "confidence": 0.9,
        "n_total_detections": 2,
        "n_frame_detections": 2,
        "reference_crop": "",
        "all_crops": [],
        "task_relevance": {"score": 0.1, "matched_terms": []},
        "articulation_hints": {"interactive": False, "kind": "static", "confidence": 0.3},
        "evidence_frames": [0, 1],
        "source_prompts": ["aisle"],
        "provenance": {"grounding_level": "observed", "canonical_truth": True},
        "mean_box_px": {"area": 72000.0, "width": 280.0, "height": 280.0},
    },
]
output_path.write_text(json.dumps({"backend_status": "ok", "objects": objects}, indent=2), encoding="utf-8")
"""
    elif mode == "runtime-missing":
        body = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "backend_status": "skipped",
            "reason": "ultralytics_missing:stubbed-for-test",
            "detections": [],
        },
        indent=2,
    ),
    encoding="utf-8",
)
"""
    else:
        body = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "backend_status": "skipped",
            "reason": "sam3_not_installed",
            "detections": [],
            "objects": [],
        },
        indent=2,
    ),
    encoding="utf-8",
)
"""
    path.write_text(body, encoding="utf-8")


def _build_staged_capture(
    tmp_path: Path,
    *,
    manifest_overrides: dict[str, object] | None = None,
    context_overrides: dict[str, object] | None = None,
) -> tuple[Path, str]:
    bucket = "local-blueprint"
    scene_id = "scene-1"
    capture_id = "capture-1"
    capture_root = tmp_path / bucket / "scenes" / scene_id / "captures" / capture_id
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True)

    manifest_payload = {
        "scene_id": scene_id,
        "capture_id": capture_id,
        "video_uri": "walkthrough.mov",
        "has_lidar": True,
        "intended_space_type": "office",
        "width": 1920,
        "height": 1080,
    }
    if manifest_overrides:
        manifest_payload.update(manifest_overrides)
    (raw_root / "manifest.json").write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    (raw_root / "intake_packet.json").write_text(
        json.dumps(
            {
                "workflowName": "Open cabinet",
                "taskSteps": ["Walk to cabinet", "Open cabinet"],
                "zone": "cabinet zone",
                "owner": "ops",
                "targetKPI": "cabinet opened successfully",
            }
        ),
        encoding="utf-8",
    )
    context_payload = {
        "sceneId": scene_id,
        "captureId": capture_id,
        "captureSource": "iphone",
        "captureModality": "iphone_arkit_lidar",
        "hiddenZoneBound": 0.2,
        "captureOrientation": {
            "displayOrientation": "portrait",
            "displayWidth": 1080,
            "displayHeight": 1920,
            "rotationDegrees": 90,
        },
    }
    if context_overrides:
        context_payload.update(context_overrides)
    (raw_root / "capture_context.json").write_text(
        json.dumps(context_payload),
        encoding="utf-8",
    )
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps({"sceneId": scene_id, "captureId": capture_id}),
        encoding="utf-8",
    )
    (raw_root / "walkthrough.mov").write_bytes(b"not-a-real-video")
    arkit_root = raw_root / "arkit"
    (arkit_root / "depth").mkdir(parents=True)
    (arkit_root / "poses.jsonl").write_text(
        json.dumps({"frame_id": "000001", "t_device_sec": 0.0, "T_world_camera": np.eye(4).tolist()}) + "\n",
        encoding="utf-8",
    )
    (arkit_root / "intrinsics.json").write_text(
        json.dumps({"width": 1920, "height": 1080, "fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0}),
        encoding="utf-8",
    )
    (arkit_root / "depth" / "000001.png").write_bytes(b"depth")

    materialized = materialize_capture_bundle(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
    )
    return capture_root, str(materialized["descriptor_uri"])


def _assert_valid_production_bundle(eval_root: Path) -> dict[str, object]:
    registration_path = eval_root / "site_world_registration.json"
    bundle = load_site_world_bundle(registration_path, require_spec=True)
    errors = validate_site_world_bundle(bundle, production_mode=True)
    assert errors == []
    return bundle.spec


def test_site_world_packaging_emits_launchable_bundle(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(tmp_path)
    success_backend = tmp_path / "success_backend.py"
    sam3_backend = tmp_path / "sam3_backend.py"
    _write_backend_script(success_backend, mode="success")
    _write_backend_script(sam3_backend, mode="sam3-skip")

    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_SAM3_COMMAND", f"python3 {sam3_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("NEOVERSE_RUNTIME_SERVICE_URL", "http://runtime.test")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_UI_BASE_URL", "https://demo.example/internal")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_PUBLIC_UI_BASE_URL", "https://demo.example/public")
    monkeypatch.setattr("blueprint_pipeline.evaluation_prep_stage.SiteWorldRuntimeServiceClient", _HealthyRuntimeClient)
    monkeypatch.setattr("blueprint_pipeline.qualification.infer_capture_fidelity_review", lambda **_kwargs: _successful_capture_review())

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="scene_memory",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    evaluation = run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    pipeline_root = capture_root / "pipeline"
    eval_root = pipeline_root / "evaluation_prep"
    manifest = json.loads((eval_root / "evaluation_prep_manifest.json").read_text(encoding="utf-8"))
    health = json.loads((eval_root / "site_world_health.json").read_text(encoding="utf-8"))
    geometry = json.loads((eval_root / "object_geometry_manifest.json").read_text(encoding="utf-8"))

    assert evaluation["manifest_path"] == str(eval_root / "evaluation_prep_manifest.json")
    assert (eval_root / "site_world_spec.json").is_file()
    assert (eval_root / "site_world_registration.json").is_file()
    assert (eval_root / "site_world_health.json").is_file()
    assert (pipeline_root / "presentation_world" / "presentation_bundle.json").is_file()
    assert (pipeline_root / "presentation_world" / "presentation_world_manifest.json").is_file()
    assert (pipeline_root / "presentation_world" / "runtime_demo_manifest.json").is_file()
    presentation_bundle = json.loads((pipeline_root / "presentation_world" / "presentation_bundle.json").read_text(encoding="utf-8"))
    runtime_demo_manifest = json.loads((pipeline_root / "presentation_world" / "runtime_demo_manifest.json").read_text(encoding="utf-8"))
    presentation_world_manifest = json.loads((pipeline_root / "presentation_world" / "presentation_world_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["presentation_bundle"] == "../presentation_world/presentation_bundle.json"
    assert manifest["artifacts"]["presentation_world_manifest"] == "../presentation_world/presentation_world_manifest.json"
    assert manifest["artifacts"]["runtime_demo_manifest"] == "../presentation_world/runtime_demo_manifest.json"
    assert presentation_bundle["authoritative_record"] is False
    assert presentation_bundle["canonical_source"]["scene_memory_manifest_uri"].endswith("/scene_memory/scene_memory_manifest.json")
    assert presentation_bundle["render_inputs"]["scene_memory_manifest_uri"].endswith("/scene_memory/scene_memory_manifest.json")
    assert presentation_bundle["render_inputs"]["conditioning_bundle_uri"].endswith("/scene_memory/conditioning_bundle.json")
    assert presentation_bundle["status"] == "missing"
    assert presentation_bundle["bundle_type"] == ""
    assert presentation_bundle["renderer_backend"] == "neoverse"
    assert presentation_bundle["fallback_policy"] == "canonical_only"
    assert presentation_world_manifest["presentation_bundle_uri"].endswith("/presentation_world/presentation_bundle.json")
    assert presentation_world_manifest["bundle_type"] == ""
    assert presentation_world_manifest["renderer_backend"] == "neoverse"
    assert presentation_world_manifest["readiness"]["bundle_status"] == "missing"
    assert presentation_world_manifest["orientation"]["display_orientation"] == "portrait"
    assert presentation_world_manifest["orientation"]["display_rotation_degrees"] == 90
    assert runtime_demo_manifest["ui_base_url"] == "https://demo.example/internal"
    assert runtime_demo_manifest["public_ui_base_url"] == "https://demo.example/public"
    assert runtime_demo_manifest["presentation_world_manifest_uri"].endswith("/presentation_world/presentation_world_manifest.json")
    assert runtime_demo_manifest["renderer_backend"] == "gsplat"
    assert runtime_demo_manifest["bundle_status"] == "ready"
    assert runtime_demo_manifest["fallback_policy"] == "canonical_only"
    assert runtime_demo_manifest["interactive_demo"]["readiness_state"] == "ready"
    assert runtime_demo_manifest["interactive_demo"]["render_inputs"]["site_world_spec_uri"].endswith("/evaluation_prep/site_world_spec.json")
    assert health["launchable"] is True
    assert len(geometry["objects"]) >= 1
    bundle = load_site_world_bundle(eval_root / "site_world_registration.json", require_spec=True)
    assert validate_site_world_bundle(bundle, production_mode=False) == []
    spec = bundle.spec
    runtime_eligibility = dict(spec["runtime_eligibility"])
    assert runtime_eligibility["readiness_state"] == "launchable"
    assert spec["canonical_output"]["authoritative_record"] is True
    assert spec["presentation_output"]["authoritative_record"] is False
    assert spec["primary_runtime_backend"] == "neoverse"
    assert spec["canonical_world_model"]["world_model_backend"] == "neoverse"
    assert spec["canonical_world_model"]["scene_representation"] == "pending_world_model_service"
    assert spec["runtime_render_source"] == "pending_world_model_service"
    assert spec["fallback_mode"] == "none"
    assert spec["canonical_world_model"]["primary_asset_path"] == ""
    assert spec["presentation"]["bundle_type"] == ""
    assert spec["presentation"]["renderer_backend"] == "neoverse"
    assert spec["presentation"]["bundle_status"] == "missing"
    assert spec["presentation"]["primary_asset_path"] == ""
    assert spec["presentation"]["orientation"]["display_orientation"] == "portrait"
    summary = json.loads((eval_root / "evaluation_prep_summary.json").read_text(encoding="utf-8"))
    launchable_export = json.loads((eval_root / "launchable_export_bundle.json").read_text(encoding="utf-8"))
    assert summary["validation_gates"]["presentation_demo_ui_ready"]["passed"] is True
    assert launchable_export["bundles"]["presentation_demo_ui"]["launchable"] is True


def test_site_world_packaging_surfaces_runtime_missing_blockers(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(tmp_path)
    missing_backend = tmp_path / "missing_backend.py"
    sam3_backend = tmp_path / "sam3_backend.py"
    _write_backend_script(missing_backend, mode="runtime-missing")
    _write_backend_script(sam3_backend, mode="sam3-skip")

    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {missing_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {missing_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_SAM3_COMMAND", f"python3 {sam3_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("NEOVERSE_RUNTIME_SERVICE_URL", "http://runtime.test")
    monkeypatch.setattr("blueprint_pipeline.qualification.infer_capture_fidelity_review", lambda **_kwargs: _successful_capture_review())

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="scene_memory",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    pipeline_root = capture_root / "pipeline"
    raw_root = capture_root / "raw"
    eval_root = pipeline_root / "evaluation_prep"
    build_report = json.loads((raw_root / "object_index_build_report.json").read_text(encoding="utf-8"))
    geometry = json.loads((eval_root / "object_geometry_manifest.json").read_text(encoding="utf-8"))
    health = json.loads((eval_root / "site_world_health.json").read_text(encoding="utf-8"))
    manifest = json.loads((eval_root / "evaluation_prep_manifest.json").read_text(encoding="utf-8"))

    assert build_report["empty_index_cause"] == "runtime_missing"
    assert geometry["status"] == "empty_object_index"
    assert "object_index_backend:yolo_world:ultralytics_missing:stubbed-for-test" in geometry["object_index_backend_blockers"]
    assert health["launchable"] is False
    assert "object_index_backend:yolo_world:ultralytics_missing:stubbed-for-test" not in health["blockers"]
    assert "object_index_backend:yolo_world:ultralytics_missing:stubbed-for-test" in health["warnings"]
    assert "object_index_backend:yolo_world:ultralytics_missing:stubbed-for-test" in manifest["degradation_reasons"]
    bundle = load_site_world_bundle(eval_root / "site_world_registration.json", require_spec=True)
    assert validate_site_world_bundle(bundle, production_mode=False) == []
    spec = bundle.spec
    runtime_eligibility = dict(spec["runtime_eligibility"])
    assert runtime_eligibility["readiness_state"] == "launchable"
    assert "object_index_backend:yolo_world:ultralytics_missing:stubbed-for-test" in runtime_eligibility["warnings"]
    assert spec["canonical_output"]["authoritative_record"] is True
    assert spec["presentation_output"]["authoritative_record"] is False
    summary = json.loads((eval_root / "evaluation_prep_summary.json").read_text(encoding="utf-8"))
    runtime_demo_manifest = json.loads((pipeline_root / "presentation_world" / "runtime_demo_manifest.json").read_text(encoding="utf-8"))
    assert summary["validation_gates"]["presentation_demo_ui_ready"]["passed"] is False
    assert runtime_demo_manifest["status"] == "ready"
    assert runtime_demo_manifest["interactive_demo"]["readiness_state"] == "blocked"
    assert "missing_demo_ui_base_url" in runtime_demo_manifest["interactive_demo"]["blockers"]


def test_site_world_packaging_preserves_vertical_capture_orientation(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(
        tmp_path,
        context_overrides={
            "captureOrientation": {
                "display_orientation": "portrait",
                "rotation_degrees": 90,
                "display_size": {"width": 1080, "height": 1920},
            }
        },
    )
    success_backend = tmp_path / "success_backend.py"
    sam3_backend = tmp_path / "sam3_backend.py"
    _write_backend_script(success_backend, mode="success")
    _write_backend_script(sam3_backend, mode="sam3-skip")

    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_SAM3_COMMAND", f"python3 {sam3_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("NEOVERSE_RUNTIME_SERVICE_URL", "http://runtime.test")
    monkeypatch.setattr("blueprint_pipeline.evaluation_prep_stage.SiteWorldRuntimeServiceClient", _HealthyRuntimeClient)
    monkeypatch.setattr("blueprint_pipeline.qualification.infer_capture_fidelity_review", lambda **_kwargs: _successful_capture_review())

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="scene_memory",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    pipeline_root = capture_root / "pipeline"
    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    scene_memory_manifest = json.loads((pipeline_root / "scene_memory" / "scene_memory_manifest.json").read_text(encoding="utf-8"))
    presentation_bundle = json.loads((pipeline_root / "presentation_world" / "presentation_bundle.json").read_text(encoding="utf-8"))
    runtime_demo_manifest = json.loads((pipeline_root / "presentation_world" / "runtime_demo_manifest.json").read_text(encoding="utf-8"))
    eval_manifest = json.loads((pipeline_root / "evaluation_prep" / "evaluation_prep_manifest.json").read_text(encoding="utf-8"))
    site_world_spec = json.loads((pipeline_root / "evaluation_prep" / "site_world_spec.json").read_text(encoding="utf-8"))
    hosted_runtime_manifest = json.loads((pipeline_root / "evaluation_prep" / "hosted_session_runtime_manifest.json").read_text(encoding="utf-8"))

    assert descriptor["capture_orientation"]["display_orientation"] == "portrait"
    assert scene_memory_manifest["capture_orientation"]["display_orientation"] == "portrait"
    assert presentation_bundle["capture_orientation"]["display_orientation"] == "portrait"
    assert runtime_demo_manifest["capture_orientation"]["display_orientation"] == "portrait"
    assert eval_manifest["capture_orientation"]["display_orientation"] == "portrait"
    assert site_world_spec["capture_orientation"]["display_orientation"] == "portrait"
    assert site_world_spec["presentation"]["orientation"]["display_orientation"] == "portrait"
    assert site_world_spec["canonical_world_model"]["orientation"]["display_orientation"] == "portrait"
    assert site_world_spec["canonical_world_model"]["world_model_backend"] == "neoverse"
    assert scene_memory_manifest["primary_runtime_backend"] == "neoverse"
    assert scene_memory_manifest["canonical_world_model"]["scene_representation"] == "pending_world_model_service"
    assert hosted_runtime_manifest["primary_runtime_backend"] == "neoverse"
    assert hosted_runtime_manifest["canonical_world_model"]["primary_asset_path"] == ""
    assert hosted_runtime_manifest["capture_orientation"]["display_orientation"] == "portrait"


def test_materialization_capture_orientation_precedence(monkeypatch, tmp_path: Path) -> None:
    bucket = "local-blueprint"
    scene_id = "scene-1"
    capture_id = "capture-1"
    raw_root = tmp_path / bucket / "scenes" / scene_id / "captures" / capture_id / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / "walkthrough.mov").write_bytes(b"not-a-real-video")
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps({"sceneId": scene_id, "captureId": capture_id}),
        encoding="utf-8",
    )

    def _write_inputs(manifest_payload: dict[str, object], context_payload: dict[str, object]) -> None:
        (raw_root / "manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")
        (raw_root / "capture_context.json").write_text(json.dumps(context_payload), encoding="utf-8")

    class _Result:
        def __init__(self, stdout: str, returncode: int = 0) -> None:
            self.stdout = stdout
            self.returncode = returncode

    _write_inputs(
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "video_uri": "walkthrough.mov",
            "width": 1920,
            "height": 1080,
        },
        {
            "sceneId": scene_id,
            "captureId": capture_id,
            "captureOrientation": {"display_orientation": "portrait", "rotation_degrees": 90},
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.materialization.subprocess.run",
        lambda *args, **kwargs: _Result(
            json.dumps({"streams": [{"codec_type": "video", "width": 1920, "height": 1080, "tags": {"rotate": "0"}}]})
        ),
    )
    explicit = build_capture_bundle_records(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
        write_frames_index=False,
    )
    assert explicit["descriptor"]["capture_orientation"]["display_orientation"] == "portrait"
    assert explicit["descriptor"]["capture_orientation"]["source"] == "capture_context"
    assert explicit["descriptor"]["capture_orientation"]["display_rotation_degrees"] == 90
    assert explicit["descriptor"]["capture_orientation"]["normalization_applied"] is True
    assert explicit["descriptor"]["capture_orientation"]["declared_capture_width"] == 1080
    assert explicit["descriptor"]["capture_orientation"]["declared_capture_height"] == 1920

    _write_inputs(
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "video_uri": "walkthrough.mov",
            "width": 1920,
            "height": 1080,
        },
        {"sceneId": scene_id, "captureId": capture_id},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.materialization.subprocess.run",
        lambda *args, **kwargs: _Result(
            json.dumps({"streams": [{"codec_type": "video", "width": 1920, "height": 1080, "tags": {"rotate": "90"}}]})
        ),
    )
    probed = build_capture_bundle_records(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
        write_frames_index=False,
    )
    assert probed["descriptor"]["capture_orientation"]["display_orientation"] == "portrait"
    assert probed["descriptor"]["capture_orientation"]["source"] == "video_metadata"
    assert probed["descriptor"]["capture_orientation"]["display_rotation_degrees"] == 90
    assert probed["descriptor"]["capture_orientation"]["normalization_applied"] is True

    _write_inputs(
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "video_uri": "walkthrough.mov",
            "width": 720,
            "height": 1280,
        },
        {"sceneId": scene_id, "captureId": capture_id},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.materialization.subprocess.run",
        lambda *args, **kwargs: _Result("", returncode=1),
    )
    fallback = build_capture_bundle_records(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
        write_frames_index=False,
    )
    assert fallback["descriptor"]["capture_orientation"]["display_orientation"] == "portrait"
    assert fallback["descriptor"]["capture_orientation"]["source"] == "inferred"
    assert fallback["descriptor"]["capture_orientation"]["display_rotation_degrees"] == 0
    assert fallback["descriptor"]["capture_orientation"]["normalization_applied"] is False


def test_presentation_primary_asset_uses_advanced_geometry(tmp_path: Path) -> None:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_dir = capture_root / "pipeline"
    advanced_dir = pipeline_dir / "advanced_geometry"
    advanced_dir.mkdir(parents=True)
    (advanced_dir / "3dgs_compressed.ply").write_text("advanced", encoding="utf-8")

    primary_asset = _presentation_primary_asset(
        pipeline_dir=pipeline_dir,
        bucket="local-blueprint",
        storage_root=tmp_path,
    )

    assert primary_asset is not None
    assert str(primary_asset["path"]).endswith("3dgs_compressed.ply")
    assert primary_asset["source_name"] == "advanced_geometry_3dgs"


def test_presentation_bundle_status_missing_without_primary_asset() -> None:
    status = _presentation_bundle_status(
        emit_presentation=True,
        primary_asset=None,
        render_inputs={"missing_inputs": []},
    )

    assert status == "missing"
