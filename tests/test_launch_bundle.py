from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.launch_bundle import (
    build_buyer_trust_score,
    build_launch_qualification_bundle,
)
from blueprint_pipeline.provider_preview import run_preview_provider


def test_buyer_trust_score_penalizes_missing_rights_and_preview_failure() -> None:
    score = build_buyer_trust_score(
        descriptor={"quality": {"pose_match_rate": 0.6}},
        qualification_record={"confidence": 0.7},
        scorecard={"completeness_status": "need_more_evidence"},
        metadata={},
        provider_status="failed",
    )

    assert score["band"] == "low"
    assert score["score"] < 60
    assert score["reasons"]


def test_launch_bundle_uses_provider_status_for_preview_state() -> None:
    bundle = build_launch_qualification_bundle(
        descriptor={"quality": {}, "capture_modality": "iphone_arkit_lidar", "evidence_tier": "qualified_metric_capture"},
        qualification_record={"readiness_state": "ready", "confidence": 0.91, "risks": []},
        scorecard={"completeness_status": "sufficient"},
        readiness_decision={"missing_evidence": []},
        site_intake={"capture_rights": {"consent_status": "documented", "consent_scope": ["sales-floor"]}},
        buyer_trust_score={"score": 88, "band": "high", "reasons": []},
        provider_run={"status": "succeeded"},
    )

    assert bundle["preview_status"] == "succeeded"
    assert bundle["buyer_trust_score"]["score"] == 88


def test_launch_bundle_defaults_preview_status_when_not_requested() -> None:
    bundle = build_launch_qualification_bundle(
        descriptor={"quality": {}, "capture_modality": "iphone_arkit_lidar", "evidence_tier": "qualified_metric_capture"},
        qualification_record={"readiness_state": "ready", "confidence": 0.91, "risks": []},
        scorecard={"completeness_status": "sufficient"},
        readiness_decision={"missing_evidence": []},
        site_intake={"capture_rights": {"consent_status": "documented", "consent_scope": ["sales-floor"]}},
        buyer_trust_score={"score": 92, "band": "high", "reasons": []},
        provider_run={},
    )

    assert bundle["preview_status"] == "not_requested"
    assert bundle["recapture_requirements"]["required"] is False


def test_preview_provider_stub_writes_manifests(tmp_path: Path) -> None:
    result = run_preview_provider(
        provider_name="stub_preview",
        descriptor={"capture_id": "cap-1", "raw_prefix_uri": "gs://bucket/raw"},
        capture_root=tmp_path,
        pipeline_dir=tmp_path,
    )

    assert result["status"] == "succeeded"
    assert (tmp_path / "provider_run_manifest.json").is_file()
    assert (tmp_path / "preview_manifest.json").is_file()


def test_preview_provider_failure_is_captured_without_raising(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDLABS_API_KEY", "")
    monkeypatch.setenv("WORLDLABS_API_URL", "")

    result = run_preview_provider(
        provider_name="world_labs",
        descriptor={"capture_id": "cap-2", "raw_prefix_uri": "gs://bucket/raw"},
        capture_root=tmp_path,
        pipeline_dir=tmp_path,
    )

    assert result["status"] == "failed"
    assert result["failure_reason"]
    assert (tmp_path / "provider_run_manifest.json").is_file()
    assert (tmp_path / "preview_manifest.json").is_file()
