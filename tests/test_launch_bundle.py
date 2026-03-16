from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.launch_bundle import (
    build_buyer_trust_score,
    build_launch_qualification_bundle,
)
from blueprint_pipeline.provider_preview import (
    _DEFAULT_WORLDLABS_TEXT_PROMPT,
    WorldLabsPreviewProvider,
    run_preview_provider,
)


def test_buyer_trust_score_penalizes_missing_rights_without_preview_failure_penalty() -> None:
    score = build_buyer_trust_score(
        descriptor={"quality": {"pose_match_rate": 0.6}},
        qualification_record={"confidence": 0.7},
        scorecard={"completeness_status": "need_more_evidence"},
        metadata={},
        provider_status="failed",
        fidelity_review={"status": "succeeded", "scores": {"coverage": 0.9, "world_model_fitness": 0.9}},
    )

    assert score["band"] == "low"
    assert score["score"] < 60
    assert score["reasons"]
    assert "preview provider is unavailable" not in score["reasons"]


def test_launch_bundle_uses_provider_status_for_preview_state() -> None:
    bundle = build_launch_qualification_bundle(
        descriptor={"quality": {}, "capture_modality": "iphone_arkit_lidar", "evidence_tier": "qualified_metric_capture"},
        qualification_record={"readiness_state": "ready", "confidence": 0.91, "risks": []},
        scorecard={"completeness_status": "sufficient"},
        readiness_decision={"missing_evidence": []},
        site_intake={"capture_rights": {"consent_status": "documented", "consent_scope": ["sales-floor"]}},
        buyer_trust_score={"score": 88, "band": "high", "reasons": []},
        provider_run={"status": "succeeded"},
        fidelity_review={"status": "succeeded", "scores": {"coverage": 0.9}},
        world_model_fit_summary={"status": "good_candidate"},
        capturer_payout_recommendation={"status": "baseline"},
        provenance_summary={"status": "grounded"},
    )

    assert bundle["preview_status"] == "succeeded"
    assert bundle["provider_preview_status"]["status"] == "succeeded"
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
        fidelity_review={"status": "succeeded", "scores": {"coverage": 0.9}},
        world_model_fit_summary={"status": "good_candidate"},
        capturer_payout_recommendation={"status": "baseline"},
        provenance_summary={"status": "grounded"},
    )

    assert bundle["preview_status"] == "not_requested"
    assert bundle["provider_preview_status"]["status"] == "not_requested"
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
        descriptor={
            "capture_id": "cap-2",
            "raw_prefix_uri": "gs://bucket/raw",
            "metadata": {
                "worldlabs_input_video_uri": "gs://bucket/scenes/scene-2/captures/cap-2/pipeline/worldlabs_input/worldlabs_input.mp4",
            },
        },
        capture_root=tmp_path,
        pipeline_dir=tmp_path,
    )

    assert result["status"] == "queued"
    assert result["failure_reason"] is None
    assert (tmp_path / "provider_run_manifest.json").is_file()
    assert (tmp_path / "preview_manifest.json").is_file()
    assert (tmp_path / "worldlabs_request_manifest.json").is_file()


def test_worldlabs_preview_provider_uses_detailed_default_prompt() -> None:
    provider = WorldLabsPreviewProvider()

    payload = provider._build_request_manifest(
        descriptor={
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {
                "worldlabs_input_video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/pipeline/worldlabs_input/worldlabs_input.mp4",
            },
        },
        capture_root=Path("/tmp/capture-root"),
    )

    assert payload["generation_request"]["world_prompt"]["text_prompt"] == _DEFAULT_WORLDLABS_TEXT_PROMPT
