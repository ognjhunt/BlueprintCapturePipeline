from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.interiorgs_phase2_summary import (
    write_scene_dashboard_summaries,
    write_dashboard_summary_json,
    write_recommended_next_actions_csv,
    write_recommended_next_actions_summary,
    write_actionability_summary,
    write_blocker_theme_summary,
    write_consolidated_summary,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_write_consolidated_summary(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303" / "pipeline"
    _write_json(pipeline_dir / "readiness_decision.json", {"status": "not_ready_yet", "confidence": 0.8})
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
                "open_close": [],
                "navigate": [],
            }
        },
    )
    _write_json(
        tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-pick-task" / "pipeline" / "readiness_decision.json",
        {"status": "ready"},
    )

    summary_path = write_consolidated_summary(output_root=tmp_path)
    content = summary_path.read_text(encoding="utf-8")

    assert "Whole-home status: `not_ready_yet`" in content
    assert "Counts: `ready=1`, `risky=0`, `not_ready_yet=0`" in content
    assert "Pick up pot_85 and place it in the target zone" in content


def test_write_blocker_theme_summary(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303" / "pipeline"
    _write_json(pipeline_dir / "readiness_decision.json", {"status": "not_ready_yet", "confidence": 0.8})
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
                "open_close": [],
                "navigate": [],
            }
        },
    )
    _write_json(
        tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-pick-task" / "pipeline" / "readiness_decision.json",
        {
            "status": "not_ready_yet",
            "human_review_required": True,
            "blockers": [
                {"detail": "Inferred target reach distance 1.4 m exceeds the bounded pilot envelope."},
                {"detail": "Estimated workcell span is 3.7 m."},
            ],
        },
    )

    summary_path = write_blocker_theme_summary(output_root=tmp_path)
    content = summary_path.read_text(encoding="utf-8")

    assert "`reach`: `1` tasks" in content
    assert "`workcell span`: `1` tasks" in content
    assert "Pick up pot_85 and place it in the target zone" in content


def test_write_actionability_summary(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303" / "pipeline"
    _write_json(pipeline_dir / "readiness_decision.json", {"status": "not_ready_yet", "confidence": 0.8})
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
                "open_close": [],
                "navigate": [],
            }
        },
    )
    _write_json(
        tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-pick-task" / "pipeline" / "readiness_decision.json",
        {
            "status": "not_ready_yet",
            "human_review_required": True,
            "blockers": [
                {"detail": "Inferred target reach distance 1.4 m exceeds the bounded pilot envelope."},
                {"detail": "Estimated workcell span is 3.7 m."},
                {"detail": "Hidden-zone bound is 0.4."},
            ],
        },
    )

    summary_path = write_actionability_summary(output_root=tmp_path)
    content = summary_path.read_text(encoding="utf-8")

    assert "`robot capability mismatch`: `1` theme hits" in content
    assert "`task redesign`: `1` theme hits" in content
    assert "`recapture`: `1` theme hits" in content


def test_write_recommended_next_actions_summary(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303" / "pipeline"
    _write_json(pipeline_dir / "readiness_decision.json", {"status": "not_ready_yet", "confidence": 0.8})
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
                "navigate": [],
            }
        },
    )
    _write_json(
        tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-pick-task" / "pipeline" / "readiness_decision.json",
        {
            "status": "not_ready_yet",
            "blockers": [{"detail": "Hidden-zone bound is 0.4."}],
        },
    )
    _write_json(
        tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-open-task" / "pipeline" / "readiness_decision.json",
        {
            "status": "ready",
            "human_review_required": True,
            "blockers": [],
        },
    )

    summary_path = write_recommended_next_actions_summary(output_root=tmp_path)
    content = summary_path.read_text(encoding="utf-8")

    assert "`recapture` Pick up pot_85 and place it in the target zone" in content
    assert "`advance to human signoff` Open and close door_60" in content


def test_write_recommended_next_actions_csv(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303" / "pipeline"
    _write_json(pipeline_dir / "readiness_decision.json", {"status": "not_ready_yet", "confidence": 0.8})
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
                "open_close": [],
                "navigate": [],
            }
        },
    )
    _write_json(
        tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-pick-task" / "pipeline" / "readiness_decision.json",
        {
            "status": "not_ready_yet",
            "blockers": [{"detail": "Hidden-zone bound is 0.4."}],
        },
    )

    csv_path = write_recommended_next_actions_csv(output_root=tmp_path)
    content = csv_path.read_text(encoding="utf-8")

    assert "scene,whole_home_capture_id,category,task_text,capture_id,status,next_action,memo_path" in content
    assert "0436,840303,pick,Pick up pot_85 and place it in the target zone,840303-pick-task,not_ready_yet,recapture,/tmp/pick-memo.md" in content


def test_write_dashboard_summary_json(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303" / "pipeline"
    _write_json(pipeline_dir / "readiness_decision.json", {"status": "not_ready_yet", "confidence": 0.8})
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
                "open_close": [],
                "navigate": [],
            }
        },
    )
    _write_json(
        tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-pick-task" / "pipeline" / "readiness_decision.json",
        {
            "status": "not_ready_yet",
            "human_review_required": True,
            "blockers": [{"detail": "Hidden-zone bound is 0.4."}],
        },
    )

    summary_path = write_dashboard_summary_json(output_root=tmp_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "v1"
    assert payload["scenes"][0]["scene"] == "0436"
    assert payload["scenes"][0]["whole_home"]["status"] == "not_ready_yet"
    assert payload["scenes"][0]["categories"]["pick"]["tasks"][0]["next_action"] == "recapture"


def test_write_scene_dashboard_summaries(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303" / "pipeline"
    _write_json(pipeline_dir / "readiness_decision.json", {"status": "not_ready_yet", "confidence": 0.8})
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
                "open_close": [],
                "navigate": [],
            }
        },
    )
    _write_json(
        tmp_path / "localbucket" / "scenes" / "0436" / "captures" / "840303-pick-task" / "pipeline" / "readiness_decision.json",
        {
            "status": "not_ready_yet",
            "human_review_required": True,
            "blockers": [{"detail": "Hidden-zone bound is 0.4."}],
        },
    )

    paths = write_scene_dashboard_summaries(output_root=tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))

    assert paths[0].name == "dashboard_summary.json"
    assert payload["scene"] == "0436"
    assert payload["whole_home"]["capture_id"] == "840303"
