from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.agent_runtime.orchestrator import run_agent_review
from blueprint_pipeline.interiorgs_phase2_adapter import (
    adapt_interiorgs_scene,
    adapt_interiorgs_task_runs,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_interiorgs_adapter_builds_reviewable_capture(tmp_path: Path) -> None:
    source_dir = tmp_path / "0436_840303"
    _write_json(
        source_dir / "labels.json",
        [
            {
                "ins_id": "60",
                "label": "door",
                "bounding_box": [
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                    {"x": 0.0, "y": 1.0, "z": 0.0},
                    {"x": 0.2, "y": 1.0, "z": 0.0},
                    {"x": 0.2, "y": 0.0, "z": 0.0},
                    {"x": 0.0, "y": 0.0, "z": 2.0},
                    {"x": 0.0, "y": 1.0, "z": 2.0},
                    {"x": 0.2, "y": 1.0, "z": 2.0},
                    {"x": 0.2, "y": 0.0, "z": 2.0},
                ],
            },
            {
                "ins_id": "70",
                "label": "table",
                "bounding_box": [
                    {"x": 1.0, "y": 1.0, "z": 0.0},
                    {"x": 1.0, "y": 1.6, "z": 0.0},
                    {"x": 1.8, "y": 1.6, "z": 0.0},
                    {"x": 1.8, "y": 1.0, "z": 0.0},
                    {"x": 1.0, "y": 1.0, "z": 0.9},
                    {"x": 1.0, "y": 1.6, "z": 0.9},
                    {"x": 1.8, "y": 1.6, "z": 0.9},
                    {"x": 1.8, "y": 1.0, "z": 0.9},
                ],
            },
        ],
    )
    _write_json(
        source_dir / "structure.json",
        {
            "rooms": [{"profile": [[0, 0], [2, 0], [2, 2], [0, 2]]}],
            "holes": [],
            "walls": [{"start": [0, 0], "end": [2, 0]}],
        },
    )
    _write_json(
        source_dir / "task_targets.synthetic.json",
        {
            "facility_name": "Kitchen Scene 0436 (InteriorGS)",
            "source": "interiorgs",
            "scene_type": "indoor",
            "tasks": [{"task_id": "Pick up table_70 and place it in the target zone"}],
            "manipulation_candidates": [
                {
                    "instance_id": "70",
                    "label": "table",
                    "category": "manipulation",
                    "confidence": 1.0,
                    "boundingBox": {
                        "center": [1.4, 1.3, 0.45],
                        "extents": [0.8, 0.6, 0.9],
                        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    },
                }
            ],
            "articulation_hints": [
                {
                    "instance_id": "60",
                    "label": "door",
                    "category": "articulation",
                    "confidence": 1.0,
                    "boundingBox": {
                        "center": [0.1, 0.5, 1.0],
                        "extents": [0.2, 1.0, 2.0],
                        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    },
                }
            ],
            "navigation_hints": [],
        },
    )

    result = adapt_interiorgs_scene(source_dir=source_dir, output_root=tmp_path / "out")

    pipeline_dir = result.capture_root / "pipeline"
    assert (result.capture_root / "capture_descriptor.json").is_file()
    descriptor = json.loads((result.capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    assert (pipeline_dir / "scene_graph.json").is_file()
    assert (pipeline_dir / "route_graph.json").is_file()
    assert (pipeline_dir / "readiness_decision.json").is_file()
    assert (pipeline_dir / "scene_memory/scene_memory_manifest.json").is_file()
    assert (pipeline_dir / "scene_memory/scene_memory_readiness.json").is_file()
    assert (pipeline_dir / "scene_memory/conditioning_bundle.json").is_file()
    assert (pipeline_dir / "preview_simulation/preview_simulation_manifest.json").is_file()
    assert descriptor["requested_lanes"] == ["qualification", "scene_memory", "advanced_geometry"]
    geometry = json.loads((pipeline_dir / "geometry_evidence.json").read_text(encoding="utf-8"))
    assert geometry["object_count"] == 2
    assert geometry["measured_route_width_m"] == 2.0
    assert geometry["adapter_route_width_source"] == "room_profile_lower_quartile"
    task_targets = json.loads((pipeline_dir / "task_targets.json").read_text(encoding="utf-8"))
    assert task_targets["target_object_ids"] == ["70"]
    handoff = json.loads((pipeline_dir / "opportunity_handoff.json").read_text(encoding="utf-8"))
    assert handoff["scene_memory_package"]["scene_memory_manifest_path"] == "scene_memory/scene_memory_manifest.json"

    review = run_agent_review(capture_root=result.capture_root, provider_name="openai")
    assert Path(review["final_bundle_path"]).is_file()


def test_interiorgs_adapter_creates_task_runs_manifest(tmp_path: Path) -> None:
    source_dir = tmp_path / "0787_841244"
    _write_json(
        source_dir / "labels.json",
        [
            {
                "ins_id": "88",
                "label": "cup",
                "bounding_box": [
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                    {"x": 0.0, "y": 0.1, "z": 0.0},
                    {"x": 0.1, "y": 0.1, "z": 0.0},
                    {"x": 0.1, "y": 0.0, "z": 0.0},
                    {"x": 0.0, "y": 0.0, "z": 0.2},
                    {"x": 0.0, "y": 0.1, "z": 0.2},
                    {"x": 0.1, "y": 0.1, "z": 0.2},
                    {"x": 0.1, "y": 0.0, "z": 0.2},
                ],
            },
            {
                "ins_id": "102",
                "label": "refrigerator",
                "bounding_box": [
                    {"x": 1.0, "y": 1.0, "z": 0.0},
                    {"x": 1.0, "y": 2.0, "z": 0.0},
                    {"x": 1.7, "y": 2.0, "z": 0.0},
                    {"x": 1.7, "y": 1.0, "z": 0.0},
                    {"x": 1.0, "y": 1.0, "z": 2.0},
                    {"x": 1.0, "y": 2.0, "z": 2.0},
                    {"x": 1.7, "y": 2.0, "z": 2.0},
                    {"x": 1.7, "y": 1.0, "z": 2.0},
                ],
            },
            {
                "ins_id": "214",
                "label": "floor lamp",
                "bounding_box": [
                    {"x": 3.0, "y": 3.0, "z": 0.0},
                    {"x": 3.0, "y": 3.2, "z": 0.0},
                    {"x": 3.2, "y": 3.2, "z": 0.0},
                    {"x": 3.2, "y": 3.0, "z": 0.0},
                    {"x": 3.0, "y": 3.0, "z": 1.2},
                    {"x": 3.0, "y": 3.2, "z": 1.2},
                    {"x": 3.2, "y": 3.2, "z": 1.2},
                    {"x": 3.2, "y": 3.0, "z": 1.2},
                ],
            },
        ],
    )
    _write_json(
        source_dir / "structure.json",
        {
            "rooms": [{"profile": [[0, 0], [4, 0], [4, 4], [0, 4]]}],
            "holes": [],
            "walls": [],
        },
    )
    _write_json(
        source_dir / "task_targets.synthetic.json",
        {
            "facility_name": "Kitchen Scene 0787 (InteriorGS)",
            "source": "interiorgs",
            "scene_type": "indoor",
            "tasks": [
                {"task_id": "Pick up cup_88 and place it in the target zone"},
                {"task_id": "Open and close refrigerator_102"},
                {"task_id": "Navigate to floor lamp_214"},
            ],
            "manipulation_candidates": [
                {
                    "instance_id": "88",
                    "label": "cup",
                    "category": "manipulation",
                    "confidence": 1.0,
                    "boundingBox": {
                        "center": [0.05, 0.05, 0.1],
                        "extents": [0.1, 0.1, 0.2],
                        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    },
                }
            ],
            "articulation_hints": [
                {
                    "instance_id": "102",
                    "label": "refrigerator",
                    "category": "articulation",
                    "confidence": 1.0,
                    "boundingBox": {
                        "center": [1.35, 1.5, 1.0],
                        "extents": [0.7, 1.0, 2.0],
                        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    },
                }
            ],
            "navigation_hints": [
                {
                    "instance_id": "214",
                    "label": "floor lamp",
                    "category": "navigation",
                    "confidence": 1.0,
                    "boundingBox": {
                        "center": [3.1, 3.1, 0.6],
                        "extents": [0.2, 0.2, 1.2],
                        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    },
                }
            ],
        },
    )

    scene_result = adapt_interiorgs_scene(source_dir=source_dir, output_root=tmp_path / "out")
    task_results = adapt_interiorgs_task_runs(
        source_dir=source_dir,
        output_root=tmp_path / "out",
        run_evaluation_prep=True,
        run_simready=True,
    )

    manifest_path = scene_result.capture_root / "pipeline" / "task_run_manifest.json"
    report_path = scene_result.capture_root / "pipeline" / "task_run_comparison_report.md"
    dashboard_path = scene_result.capture_root / "pipeline" / "dashboard_summary.json"
    deployment_summary_path = scene_result.capture_root / "pipeline" / "scene_deployment_summary.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(task_results) == 3
    assert len(manifest["groups"]["pick"]) == 1
    assert len(manifest["groups"]["open_close"]) == 1
    assert len(manifest["groups"]["navigate"]) == 1
    assert Path(manifest["groups"]["pick"][0]["evaluation_prep_manifest_path"]).is_file()
    assert Path(manifest["groups"]["pick"][0]["simready_scene_path"]).is_file()
    assert Path(manifest["groups"]["open_close"][0]["simready_manifest_path"]).is_file()
    assert report_path.is_file()
    assert dashboard_path.is_file()
    assert deployment_summary_path.is_file()

    open_close_capture_root = Path(manifest["groups"]["open_close"][0]["capture_root"])
    readiness = json.loads(
        (open_close_capture_root / "pipeline" / "readiness_decision.json").read_text(encoding="utf-8")
    )
    blocker_text = " ".join(item["detail"] for item in readiness.get("blockers", []))
    assert "0.0 m" not in blocker_text
    simready_assets = json.loads(
        (open_close_capture_root / "pipeline" / "simready" / "simready_assets.json").read_text(encoding="utf-8")
    )
    assert any(
        item["object_id"] == "102" and item["articulation_required"] is True
        for item in simready_assets["assets"]
    )
