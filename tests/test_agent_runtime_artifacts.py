from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.agent_runtime.artifacts import load_pipeline_review_artifacts


def _write_json(path: Path, payload: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {}), encoding="utf-8")


def test_load_pipeline_review_artifacts_includes_optional_supplemental_geometry(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_root = capture_root / "pipeline"
    capture_root.mkdir(parents=True)
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "capture_source": "iphone",
            "capture_tier": "tier1_iphone",
            "raw_prefix_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/raw",
            "frames_index_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/frames/index.jsonl",
        },
    )
    for filename in (
        "site_intake.json",
        "capture_package_manifest.json",
        "capture_qa_scorecard.json",
        "task_scope_record.json",
        "qualification_record.json",
        "qualification_brief.json",
        "scene_graph.json",
        "route_graph.json",
        "geometry_evidence.json",
        "capability_checks.json",
        "blocker_register.json",
        "readiness_decision.json",
        "opportunity_handoff.json",
        "human_actions_required.json",
        "task_hypothesis_report.json",
        "normalized_task_hypothesis.json",
    ):
        _write_json(pipeline_root / filename)
    (pipeline_root / "readiness_report.md").write_text("# Readiness\n", encoding="utf-8")
    _write_json(pipeline_root / "geometry" / "geometry_summary.json", {"status": "completed"})
    _write_json(
        pipeline_root / "worldlabs_assets" / "materialized_assets_manifest.json",
        {"status": "complete"},
    )

    artifacts = load_pipeline_review_artifacts(capture_root)

    assert [item["label"] for item in artifacts.supplemental_geometry] == [
        "geometry_summary",
        "worldlabs_materialized_assets",
    ]
