from __future__ import annotations

from types import SimpleNamespace

from blueprint_pipeline.agent_runtime.orchestrator import _agent_blocker_register, _recapture_plan


def test_agent_blocker_register_infers_resolution_paths_and_zone() -> None:
    artifacts = SimpleNamespace(
        descriptor=SimpleNamespace(
            scene_id="scene-1",
            capture_id="capture-1",
            capture_modality="iphone_arkit_lidar",
        ),
        task_scope_record={"task_zone": {"name": "West Aisle"}},
        blocker_register={
            "entries": [
                {
                    "id": "risk_1",
                    "severity": "high",
                    "category": "geometry",
                    "detail": "Route segment A3-B1 width measured at 2.4 m, below required shared-traffic clearance.",
                },
                {
                    "id": "risk_2",
                    "severity": "medium",
                    "category": "scoping",
                    "detail": "Task scope remains ambiguous from the available capture evidence.",
                },
            ]
        },
    )
    evidence_audit = {
        "evidence_gaps": [
            {
                "severity": "high",
                "category": "hidden_zone",
                "detail": "Restricted mezzanine access prevents verification of the east-side drop zone; escort required.",
                "source_artifacts": ["geometry_evidence.json"],
            }
        ]
    }

    enriched = _agent_blocker_register(artifacts, evidence_audit)
    entries = {item["id"]: item for item in enriched["entries"]}

    assert entries["risk_1"]["resolution_path"] == "recapture"
    assert entries["risk_1"]["zone"] == "West Aisle"
    assert entries["risk_2"]["resolution_path"] == "human_review"
    assert entries["evidence_gap_3"]["resolution_path"] == "recapture"
    assert "blocker_register.json" in entries["risk_1"]["source_artifacts"]
    assert "evidence_audit.json" in entries["evidence_gap_3"]["source_artifacts"]


def test_recapture_plan_filters_non_recapture_blockers_and_marks_access_pending() -> None:
    artifacts = SimpleNamespace(
        readiness_decision={"status": "not_ready_yet"},
        descriptor=SimpleNamespace(
            scene_id="scene-2",
            capture_id="capture-2",
            capture_modality="android_video_only",
        ),
        task_scope_record={"task_zone": {"name": "East Mezzanine"}},
    )
    blocker_register = {
        "entries": [
            {
                "id": "hidden_zone_1",
                "severity": "high",
                "category": "hidden_zone",
                "detail": "Restricted mezzanine access prevents verification of the east-side drop zone; escort required.",
                "zone": "East Mezzanine",
                "resolution_path": "recapture",
                "source_artifacts": ["geometry_evidence.json"],
            },
            {
                "id": "scope_1",
                "severity": "medium",
                "category": "scoping",
                "detail": "Task scope remains ambiguous from the available capture evidence.",
                "zone": "East Mezzanine",
                "resolution_path": "human_review",
                "source_artifacts": ["task_scope_record.json"],
            },
        ]
    }

    recapture_plan = _recapture_plan(
        artifacts,
        blocker_register,
        route_access_review={"overall_route_readiness": "blocked"},
        workcell_risk_review={"risks": ["hidden zone access"]},
    )

    assert recapture_plan["required"] is True
    assert recapture_plan["access_pending"] is True
    assert len(recapture_plan["steps"]) == 1
    assert recapture_plan["steps"][0]["detail"].startswith("Restricted mezzanine access prevents verification")
    assert recapture_plan["steps"][0]["preferred_capture_mode"] == "iphone_arkit_lidar"
    assert recapture_plan["priority_distribution"]["P1"] == 1
    assert "LiDAR scanner" in recapture_plan["equipment_list"]
    assert recapture_plan["capture_sessions"][0]["zone"] == "East Mezzanine"

