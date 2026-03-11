from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from blueprint_pipeline.interiorgs_phase2_summary import (
    _recommended_next_action,
    build_scene_deployment_summary,
    write_actionability_summary,
    write_blocker_theme_summary,
    write_dashboard_summary_json,
    write_recommended_next_actions_csv,
    write_recommended_next_actions_summary,
    write_scene_dashboard_summaries,
    write_scene_deployment_summaries,
    write_consolidated_summary,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _schema() -> dict:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "schemas"
        / "interiorgs_scene_dashboard_summary.schema.json"
    )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _seed_scene(tmp_path: Path) -> Path:
    pipeline_dir = tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303" / "pipeline"
    _write_json(pipeline_dir / "readiness_decision.json", {"status": "not_ready_yet", "confidence": 0.8})
    _write_json(
        pipeline_dir / "capture_qa_scorecard.json",
        {"completeness_status": "sufficient"},
    )
    _write_json(
        pipeline_dir / "task_run_manifest.json",
        {
            "groups": {
                "pick": [
                    {
                        "task_text": "Pick up pot_85 and place it in the target zone",
                        "capture_root": str(tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-pick-task"),
                        "capture_id": "840303-pick-task",
                        "final_memo_path": "/tmp/pick-memo.md",
                    }
                ],
                "open_close": [
                    {
                        "task_text": "Open and close door_60",
                        "capture_root": str(tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-open-task"),
                        "capture_id": "840303-open-task",
                        "final_memo_path": "/tmp/open-memo.md",
                    }
                ],
                "navigate": [
                    {
                        "task_text": "Navigate to station_33",
                        "capture_root": str(tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-nav-task"),
                        "capture_id": "840303-nav-task",
                        "final_memo_path": "/tmp/nav-memo.md",
                    }
                ],
            }
        },
    )
    _write_json(
        tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-pick-task" / "pipeline" / "readiness_decision.json",
        {
            "status": "not_ready_yet",
            "human_review_required": True,
            "hidden_zone_bound": 0.4,
            "blockers": [
                {
                    "category": "capture_coverage",
                    "resolution_path": "recapture",
                    "detail": "Hidden-zone bound is 0.4.",
                }
            ],
        },
    )
    _write_json(
        tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-open-task" / "pipeline" / "readiness_decision.json",
        {
            "status": "risky",
            "blockers": [
                {
                    "category": "geometry_clearance",
                    "resolution_path": "scope_change",
                    "detail": "Estimated workcell span is 3.7 m.",
                }
            ],
        },
    )
    _write_json(
        tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-nav-task" / "pipeline" / "readiness_decision.json",
        {
            "status": "not_ready_yet",
            "capability_checks": [
                {
                    "name": "reach",
                    "status": "blocked",
                    "detail": "Reach exceeds the bounded robot envelope.",
                }
            ],
            "blockers": [],
        },
    )
    return pipeline_dir


def test_write_consolidated_summary(tmp_path: Path) -> None:
    _seed_scene(tmp_path)
    summary_path = write_consolidated_summary(output_root=tmp_path)
    content = summary_path.read_text(encoding="utf-8")
    assert "Whole-home status: `not_ready_yet`" in content
    assert "Counts: `ready=0`, `risky=1`, `not_ready_yet=0`" in content
    assert "Pick up pot_85 and place it in the target zone" in content


def test_write_blocker_theme_summary(tmp_path: Path) -> None:
    _seed_scene(tmp_path)
    summary_path = write_blocker_theme_summary(output_root=tmp_path)
    content = summary_path.read_text(encoding="utf-8")
    assert "`hidden-zone coverage`: `1` tasks" in content
    assert "`route / clearance`: `1` tasks" in content
    assert "`reach`: `1` tasks" in content


def test_write_actionability_summary(tmp_path: Path) -> None:
    _seed_scene(tmp_path)
    summary_path = write_actionability_summary(output_root=tmp_path)
    content = summary_path.read_text(encoding="utf-8")
    assert "`recapture`: `1` theme hits" in content
    assert "`task redesign`: `1` theme hits" in content
    assert "`robot capability mismatch`: `1` theme hits" in content


def test_write_recommended_next_actions_summary(tmp_path: Path) -> None:
    _seed_scene(tmp_path)
    summary_path = write_recommended_next_actions_summary(output_root=tmp_path)
    content = summary_path.read_text(encoding="utf-8")
    assert "`recapture` Pick up pot_85 and place it in the target zone" in content
    assert "`redesign` Open and close door_60" in content
    assert "`defer` Navigate to station_33" in content


def test_write_recommended_next_actions_csv(tmp_path: Path) -> None:
    _seed_scene(tmp_path)
    csv_path = write_recommended_next_actions_csv(output_root=tmp_path)
    content = csv_path.read_text(encoding="utf-8")
    assert "scene,whole_home_capture_id,category,task_text,capture_id,status,next_action,memo_path" in content
    assert "0436,840303,pick,Pick up pot_85 and place it in the target zone,840303-pick-task,not_ready_yet,recapture,/tmp/pick-memo.md" in content


def test_write_dashboard_summary_json_validates_schema_and_rollup(tmp_path: Path) -> None:
    _seed_scene(tmp_path)
    summary_path = write_dashboard_summary_json(output_root=tmp_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(_schema())
    validator.validate(payload["scenes"][0])
    assert payload["schema_version"] == "v1"
    assert payload["scenes"][0]["deployment_summary"] == {
        "total_tasks": 3,
        "ready_now": 0,
        "needs_redesign": 1,
        "outside_robot_envelope": 1,
    }


def test_write_scene_dashboard_summaries_validates_schema(tmp_path: Path) -> None:
    _seed_scene(tmp_path)
    paths = write_scene_dashboard_summaries(output_root=tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    Draft202012Validator(_schema()).validate(payload)
    assert paths[0].name == "dashboard_summary.json"
    assert payload["whole_home"]["memo_uri"] == "gs://localbucket/scenes/0436/captures/840303/pipeline/agent_readiness_memo.md"


def test_write_scene_deployment_summaries_use_dashboard_payload(tmp_path: Path) -> None:
    _seed_scene(tmp_path)
    paths = write_scene_deployment_summaries(output_root=tmp_path)
    content = paths[0].read_text(encoding="utf-8")
    assert paths[0].name == "scene_deployment_summary.md"
    assert "- Ready now: `0`" in content
    assert "- Need redesign: `1`" in content
    assert "- Outside robot envelope: `1`" in content
    assert "`open_close` Open and close door_60 (`840303-open-task`)" in content


def test_build_scene_deployment_summary_handles_empty_sections() -> None:
    content = build_scene_deployment_summary(
        {
            "scene": "0436",
            "whole_home": {"capture_id": "840303", "status": "ready"},
            "deployment_summary": {
                "total_tasks": 1,
                "ready_now": 1,
                "needs_redesign": 0,
                "outside_robot_envelope": 0,
            },
            "categories": {
                "pick": {
                    "tasks": [
                        {
                            "task_text": "Pick item",
                            "capture_id": "pick-1",
                            "next_action": "advance to human signoff",
                        }
                    ]
                },
                "open_close": {"tasks": []},
                "navigate": {"tasks": []},
            },
        }
    )
    assert "## Need Redesign" in content
    assert "- none" in content


def test_recommended_next_action_prefers_structured_resolution_paths() -> None:
    assert _recommended_next_action(
        {
            "status": "ready",
            "human_review_required": True,
            "blockers": [
                {
                    "category": "capture_coverage",
                    "resolution_path": "recapture",
                    "detail": "Missing overhead capture.",
                }
            ],
        }
    ) == "recapture"
    assert _recommended_next_action(
        {
            "status": "risky",
            "blockers": [
                {
                    "category": "geometry_clearance",
                    "resolution_path": "scope_change",
                    "detail": "Route width is too narrow.",
                }
            ],
        }
    ) == "redesign"
    assert _recommended_next_action(
        {
            "status": "not_ready_yet",
            "blockers": [
                {
                    "category": "platform_limitation",
                    "resolution_path": "platform_change",
                    "detail": "Reach exceeds envelope.",
                }
            ],
        }
    ) == "defer"
    assert _recommended_next_action(
        {
            "status": "ready",
            "human_review_required": True,
            "blockers": [],
        }
    ) == "advance to human signoff"
