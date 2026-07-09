from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_capture_root_by_site_coverage import validate_coverage


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _webapp_forwarding_preflight(path: Path, site_slugs: list[str]) -> Path:
    _write_json(
        path,
        {
            "schema_version": "blueprint.webapp.robot_eval_forwarding_readiness.v1",
            "status": "ready_for_required_forwarding_with_probe",
            "configured_env": {
                "capture_root_by_site_json": {
                    "configured": True,
                    "valid": True,
                    "site_count": len(site_slugs),
                    "site_slugs": site_slugs,
                }
            },
        },
    )
    return path


def test_capture_root_by_site_coverage_passes_for_all_expected_sites(tmp_path: Path) -> None:
    site_one = tmp_path / "sites" / "one" / "scenes" / "scene-1" / "captures" / "capture-1"
    site_two = tmp_path / "sites" / "two" / "scenes" / "scene-2" / "captures" / "capture-2"
    site_one.mkdir(parents=True)
    site_two.mkdir(parents=True)
    expected = {"site-one": str(site_one), "site-two": str(site_two)}
    preflight = _webapp_forwarding_preflight(
        tmp_path / "forwarding_preflight.json",
        ["site-one", "site-two"],
    )

    report = validate_coverage(
        expected_site_roots=expected,
        pipeline_site_roots=dict(expected),
        webapp_forwarding_preflight=preflight,
        require_paths_exist=True,
    )

    assert report["status"] == "passed"
    assert report["blockers"] == []
    assert report["expected_site_count"] == 2
    assert report["claim_boundary"]["live_forwarding_or_pipeline_processing_proven"] is False


def test_capture_root_by_site_coverage_blocks_missing_webapp_site(tmp_path: Path) -> None:
    expected = {"site-one": "/captures/site-one", "site-two": "/captures/site-two"}
    preflight = _webapp_forwarding_preflight(
        tmp_path / "forwarding_preflight.json",
        ["site-one"],
    )

    report = validate_coverage(
        expected_site_roots=expected,
        pipeline_site_roots=dict(expected),
        webapp_forwarding_preflight=preflight,
    )

    assert report["status"] == "blocked"
    assert "webapp_forwarding_preflight_missing_site:site-two" in report["blockers"]
    site_two = next(item for item in report["site_results"] if item["site_slug"] == "site-two")
    assert site_two["status"] == "blocked"


def test_capture_root_by_site_coverage_blocks_pipeline_root_mismatch(tmp_path: Path) -> None:
    expected = {"site-one": str(tmp_path / "expected" / "site-one")}
    pipeline = {"site-one": str(tmp_path / "other" / "site-one")}

    report = validate_coverage(
        expected_site_roots=expected,
        pipeline_site_roots=pipeline,
    )

    assert report["status"] == "blocked"
    assert "pipeline_capture_root_mismatch_for_site:site-one" in report["blockers"]
