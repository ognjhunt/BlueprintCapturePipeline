from __future__ import annotations

from pathlib import Path

import pytest

import autoresearch.run_eval as run_eval
from autoresearch.common import REPO_ROOT, load_target_manifest
from autoresearch.runner import snapshot_candidate_files


def _snapshot_candidate(manifest_path: Path, tmp_path: Path) -> Path:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    snapshot_candidate_files(
        manifest=load_target_manifest(manifest_path),
        source_root=REPO_ROOT,
        destination_root=candidate_dir,
    )
    return candidate_dir


def test_intake_normalizer_eval_handles_both_frozen_cases(monkeypatch, tmp_path: Path) -> None:
    manifest_path = REPO_ROOT / "autoresearch" / "targets" / "intake_normalizer.json"
    candidate_dir = _snapshot_candidate(manifest_path, tmp_path)

    def fake_agent(**kwargs):  # type: ignore[no-untyped-def]
        prompt = kwargs["prompt"]
        if "Case: missing_required_fields" in prompt:
            return {
                "schema_version": "v1",
                "scene_id": "scene-missing",
                "capture_id": "capture-missing",
                "status": "needs_human_completion",
                "capture_modality": "video_only",
                "workflow": "",
                "zone": "",
                "owner": "",
                "success_criteria": [],
                "adjacent_systems": ["Manual pallet staging"],
                "non_routine_modes": [],
                "people_traffic_notes": ["Shared pedestrian path"],
                "privacy_restrictions": [],
                "security_restrictions": [],
                "known_blockers": [],
                "missing_required_fields": ["workflow", "task_zone", "success_criteria"],
            }
        return {
            "schema_version": "v1",
            "scene_id": "scene-complete",
            "capture_id": "capture-complete",
            "status": "normalized",
            "capture_modality": "metric_scan",
            "workflow": "Pick totes from AMR arrival station and place on outbound conveyor belt B3",
            "zone": "Pick Module 3",
            "owner": "site_ops_lead",
            "success_criteria": [
                "Route clearance verified for all primary workflow segments",
                "Pickup and placement targets remain visible and reachable",
            ],
            "adjacent_systems": ["AMR arrival station", "Outbound conveyor B3"],
            "non_routine_modes": ["Jam clearing at conveyor merge"],
            "people_traffic_notes": ["Forklift cross-traffic enters from the west aisle"],
            "privacy_restrictions": ["No faces in final artifacts"],
            "security_restrictions": ["Escort required after 18:00"],
            "known_blockers": ["West camera mast blocks one corner view"],
            "missing_required_fields": [],
        }

    monkeypatch.setattr(run_eval, "_invoke_generation_agent", fake_agent)
    eval_dir = tmp_path / "eval"
    payload = run_eval.evaluate_candidate(
        target_manifest_path=manifest_path,
        candidate_dir=candidate_dir,
        output_dir=eval_dir,
        agent_engine="codex",
    )
    assert payload["pytest"]["failed"] == 0
    assert payload["structured_checks"]["schema_valid"] is True
    assert (eval_dir / "eval.json").is_file()


def test_readiness_report_eval_enforces_required_sections(monkeypatch, tmp_path: Path) -> None:
    manifest_path = REPO_ROOT / "autoresearch" / "targets" / "readiness_report_writer.json"
    candidate_dir = _snapshot_candidate(manifest_path, tmp_path)

    def fake_agent(**kwargs):  # type: ignore[no-untyped-def]
        prompt = kwargs["prompt"]
        if "Case: pre_screen" in prompt:
            memo = """# Site Readiness Assessment: Pre-Screen Zone

## Readiness State: RISKY

PRE-SCREEN ASSESSMENT ONLY — NOT FOR QUALIFICATION DECISIONS

## Human Signoff Boundary

Human review is required before any qualification decision.

## Executive Summary

- pre_screen
- Video-only evidence cannot support decision-grade clearance checks.

## Evidence Assessment

pre_screen

## Required Human Actions

- Confirm whether a metric recapture is in scope before qualification review.

## Recapture Recommendations

- Return for metric capture of the primary route and workcell clearances.

## Next Steps

1. Schedule metric recapture.
"""
        else:
            memo = """# Site Readiness Assessment: Pick Module 3

## Readiness State: NOT READY YET

Hidden-zone bound 0.42 exceeds the readiness envelope.

## Human Signoff Boundary

Human review is required before any qualification decision.

## Executive Summary

- NOT READY YET
- Hidden-zone bound 0.42 exceeds the readiness envelope.

## Blockers

- Route segment A3-B1 width measured at 2.4 m, below the required 2.65 m shared-traffic clearance.

## Capability Assessment

- Route traversal remains blocked pending recapture.

## Evidence Assessment

- Hidden-zone bound 0.42 exceeds the readiness envelope.

## Required Human Actions

- Approve targeted recapture for the west aisle hidden zone.
- Review shared-traffic clearance with site safety lead.

## Recapture Recommendations

- Capture the west aisle hidden zone with metric-grade geometry coverage.

## Next Steps

1. Approve recapture.
2. Review shared-traffic clearance with site safety lead.
"""
        return {"memo_markdown": memo}

    monkeypatch.setattr(run_eval, "_invoke_generation_agent", fake_agent)
    eval_dir = tmp_path / "eval"
    payload = run_eval.evaluate_candidate(
        target_manifest_path=manifest_path,
        candidate_dir=candidate_dir,
        output_dir=eval_dir,
        agent_engine="codex",
    )
    assert payload["pytest"]["failed"] == 0
    assert payload["rubric"]["score"] > 0.0
    assert (eval_dir / "cases" / "pre_screen" / "readiness_report.md").is_file()


def test_recapture_planner_eval_rejects_uncited_geometry_drift(monkeypatch, tmp_path: Path) -> None:
    manifest_path = REPO_ROOT / "autoresearch" / "targets" / "recapture_planner.json"
    candidate_dir = _snapshot_candidate(manifest_path, tmp_path)

    def fake_agent(**kwargs):  # type: ignore[no-untyped-def]
        prompt = kwargs["prompt"]
        if "Case: access_constrained" in prompt:
            return {
                "schema_version": "v1",
                "scene_id": "scene-recapture-2",
                "capture_id": "capture-recapture-2",
                "required": True,
                "access_pending": True,
                "steps": [
                    {
                        "order": 1,
                        "detail": "Restricted mezzanine access prevents verification of the east-side drop zone; escort required.",
                        "preferred_capture_mode": "iphone_arkit_lidar",
                    }
                ],
            }
        if "Case: mixed_access" in prompt:
            return {
                "schema_version": "v1",
                "scene_id": "scene-recapture-3",
                "capture_id": "capture-recapture-3",
                "required": True,
                "access_pending": True,
                "steps": [
                    {
                        "order": 1,
                        "detail": "Restricted mezzanine access prevents verification of the east-side drop zone; escort required.",
                        "preferred_capture_mode": "iphone_arkit_lidar",
                    },
                    {
                        "order": 2,
                        "detail": "Route segment A3-B1 width measured at 2.4 m, below required shared-traffic clearance.",
                        "preferred_capture_mode": "iphone_arkit_lidar",
                    },
                ],
            }
        return {
            "schema_version": "v1",
            "scene_id": "scene-recapture-1",
            "capture_id": "capture-recapture-1",
            "required": True,
            "steps": [
                {
                    "order": 1,
                    "detail": "Hidden-zone bound 0.42 exceeds the readiness envelope.",
                    "preferred_capture_mode": "iphone_arkit_lidar",
                },
                {
                    "order": 2,
                    "detail": "Route segment A3-B1 width measured at 2.4 m, below required shared-traffic clearance.",
                    "preferred_capture_mode": "iphone_arkit_lidar",
                },
            ],
        }

    monkeypatch.setattr(run_eval, "_invoke_generation_agent", fake_agent)
    eval_dir = tmp_path / "eval"
    payload = run_eval.evaluate_candidate(
        target_manifest_path=manifest_path,
        candidate_dir=candidate_dir,
        output_dir=eval_dir,
        agent_engine="codex",
    )
    assert payload["pytest"]["failed"] == 0
    assert payload["structured_checks"]["schema_valid"] is True
    assert payload["rubric"]["groundedness"] == pytest.approx(1.0)
