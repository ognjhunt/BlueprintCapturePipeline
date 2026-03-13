from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.capture_orchestrator import PipelineConfig, run_capture_pipeline
from blueprint_pipeline.evaluation_prep_stage import run_evaluation_prep_stage
from blueprint_pipeline.materialization import materialize_capture_bundle


class _HealthyRuntimeClient:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def register_site_world_package(self, *, spec, registration, health):  # type: ignore[no-untyped-def]
        site_world_id = str(registration.get("site_world_id") or "siteworld-test")
        return {
            **dict(registration),
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


def _build_staged_capture(tmp_path: Path) -> tuple[Path, str]:
    bucket = "local-blueprint"
    scene_id = "scene-1"
    capture_id = "capture-1"
    capture_root = tmp_path / bucket / "scenes" / scene_id / "captures" / capture_id
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True)

    (raw_root / "manifest.json").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "capture_id": capture_id,
                "video_uri": "walkthrough.mov",
                "has_lidar": True,
                "intended_space_type": "office",
            }
        ),
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
    (raw_root / "capture_context.json").write_text(
        json.dumps(
            {
                "sceneId": scene_id,
                "captureId": capture_id,
                "captureSource": "iphone",
                "captureModality": "iphone_arkit_lidar",
                "hiddenZoneBound": 0.2,
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps({"sceneId": scene_id, "captureId": capture_id}),
        encoding="utf-8",
    )
    (raw_root / "walkthrough.mov").write_bytes(b"not-a-real-video")

    materialized = materialize_capture_bundle(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
    )
    return capture_root, str(materialized["descriptor_uri"])


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
    monkeypatch.setattr("blueprint_pipeline.evaluation_prep_stage.SiteWorldRuntimeServiceClient", _HealthyRuntimeClient)

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="qualification",
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
    assert (pipeline_root / "presentation_world" / "presentation_world_manifest.json").is_file()
    assert (pipeline_root / "presentation_world" / "runtime_demo_manifest.json").is_file()
    assert manifest["artifacts"]["presentation_world_manifest"] == "../presentation_world/presentation_world_manifest.json"
    assert manifest["artifacts"]["runtime_demo_manifest"] == "../presentation_world/runtime_demo_manifest.json"
    assert health["launchable"] is True
    assert len(geometry["objects"]) >= 1


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

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="qualification",
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
    assert "object_index_backend:yolo_world:ultralytics_missing:stubbed-for-test" in health["blockers"]
    assert "object_index_backend:yolo_world:ultralytics_missing:stubbed-for-test" in manifest["degradation_reasons"]
