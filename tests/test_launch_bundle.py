from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.launch_bundle import (
    build_buyer_trust_score,
    build_launch_qualification_bundle,
)
from blueprint_pipeline.provider_preview import (
    _DEFAULT_WORLDLABS_TEXT_PROMPT,
    _worldlabs_api_request,
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

    assert result["status"] == "failed"
    assert result["failure_reason"]
    assert (tmp_path / "provider_run_manifest.json").is_file()
    assert (tmp_path / "preview_manifest.json").is_file()
    assert (tmp_path / "worldlabs_request_manifest.json").is_file()


def test_preview_provider_requires_explicit_selection(tmp_path: Path) -> None:
    result = run_preview_provider(
        provider_name="",
        descriptor={"capture_id": "cap-3", "raw_prefix_uri": "gs://bucket/raw"},
        capture_root=tmp_path,
        pipeline_dir=tmp_path,
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "preview_provider_not_configured"
    assert (tmp_path / "provider_run_manifest.json").is_file()
    assert (tmp_path / "preview_manifest.json").is_file()


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


def test_worldlabs_preview_provider_never_uses_raw_video() -> None:
    provider = WorldLabsPreviewProvider()

    payload = provider._build_request_manifest(
        descriptor={
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/walkthrough.mov",
            "privacy_status": "no_people_detected",
            "metadata": {},
        },
        capture_root=Path("/tmp/capture-root"),
    )

    assert payload["status"] == "blocked"
    assert payload["selected_video_uri"] is None
    assert all(item["source_id"] != "raw_video_uri" for item in payload["video_candidates"])


def test_worldlabs_preview_provider_labels_non_production_raw_bypass() -> None:
    provider = WorldLabsPreviewProvider()

    payload = provider._build_request_manifest(
        descriptor={
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {
                "worldlabs_input_video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/pipeline/worldlabs_input/worldlabs_input.mp4",
                "worldlabs_input_labeling": {
                    "non_production": True,
                    "unredacted_input": True,
                    "raw_video_bypass_used": True,
                    "review_state": "non_production_unredacted_raw_preview",
                },
            },
        },
        capture_root=Path("/tmp/capture-root"),
    )

    assert payload["input_labeling"]["non_production"] is True
    assert payload["input_labeling"]["unredacted_input"] is True
    assert "non-production-preview" in payload["generation_request"]["tags"]
    assert "unredacted-raw-input" in payload["generation_request"]["tags"]


def test_worldlabs_preview_provider_production_requires_privacy_audit(monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")
    provider = WorldLabsPreviewProvider()

    with pytest.raises(RuntimeError, match="production_worldlabs_input_audit_missing_or_invalid"):
        provider._build_request_manifest(
            descriptor={
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "metadata": {
                    "worldlabs_input_video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/pipeline/worldlabs_input/worldlabs_input.mp4",
                },
            },
            capture_root=Path("/tmp/capture-root"),
        )


def test_worldlabs_preview_provider_carries_privacy_audit_checksums(monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")
    provider = WorldLabsPreviewProvider()
    input_uri = "gs://bucket/scenes/scene-1/captures/capture-1/pipeline/worldlabs_input/worldlabs_input.mp4"

    payload = provider._build_request_manifest(
        descriptor={
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {
                "worldlabs_input_video_uri": input_uri,
                "worldlabs_input_manifest_uri": "gs://bucket/scenes/scene-1/captures/capture-1/pipeline/worldlabs_input/worldlabs_input_manifest.json",
                "worldlabs_input_audit_uri": "gs://bucket/scenes/scene-1/captures/capture-1/pipeline/worldlabs_input_audit.json",
                "worldlabs_input_audit": {
                    "privacy_safe_input": True,
                    "raw_video_bypass_used": False,
                    "source_manifest_uri": "gs://bucket/scenes/scene-1/captures/capture-1/pipeline/privacy_processing_manifest.json",
                    "output_video_uri": input_uri,
                    "output_checksum_sha256": "abc123",
                    "source_checksum_sha256": "def456",
                },
            },
        },
        capture_root=Path("/tmp/capture-root"),
    )

    assert payload["status"] == "ready_for_generation"
    assert payload["worldlabs_input_audit_uri"].endswith("/worldlabs_input_audit.json")
    assert payload["selected_input_checksum_sha256"] == "abc123"
    assert payload["source_input_checksum_sha256"] == "def456"
    assert payload["privacy_safe_input"] is True


def test_worldlabs_preview_provider_consumes_provider_adapter_input() -> None:
    provider = WorldLabsPreviewProvider()
    input_uri = "gs://bucket/scenes/scene-1/captures/capture-1/pipeline/worldlabs_input/worldlabs_input.mp4"

    payload = provider._build_request_manifest(
        descriptor={
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {
                "worldlabs_input_video_uri": "gs://bucket/legacy/should-not-be-used.mp4",
            },
        },
        capture_root=Path("/tmp/capture-root"),
        provider_adapter_input={
            "schema_version": "v1",
            "adapter_input_type": "ProviderAdapterInput",
            "provider": "world_labs",
            "adapter": "marble",
            "status": "ready",
            "canonical_site_package_uri": "gs://bucket/scenes/scene-1/captures/capture-1/pipeline/site_package/canonical_site_package.json",
            "provider_adapter_input_uri": "gs://bucket/scenes/scene-1/captures/capture-1/pipeline/site_package/provider_adapter_inputs/world_labs_marble.json",
            "conditioning_inputs": {
                "rgb_video": {
                    "uri": input_uri,
                    "source_id": "privacy_safe_world_model_input",
                    "privacy_safe": True,
                    "checksum_sha256": "abc123",
                    "source_checksum_sha256": "def456",
                    "source_manifest_uri": "gs://bucket/pipeline/privacy_processing_manifest.json",
                }
            },
            "generation": {
                "display_name": "Warehouse handoff lane",
                "text_prompt": "Preserve the handoff lane, docks, doors, and route anchors.",
                "tags": ["scene-1", "capture-1", "provider-adapter-input"],
            },
            "labeling": {"capture_grounded": True, "generated_output": False},
        },
    )

    assert payload["status"] == "ready_for_generation"
    assert payload["selected_video_uri"] == input_uri
    assert payload["selected_video_source_id"] == "privacy_safe_world_model_input"
    assert payload["canonical_site_package_uri"].endswith("/canonical_site_package.json")
    assert payload["provider_adapter_input_uri"].endswith("/world_labs_marble.json")
    assert payload["adapter_input_status"] == "ready"
    assert payload["selected_input_checksum_sha256"] == "abc123"
    assert payload["source_input_checksum_sha256"] == "def456"
    assert payload["generation_request"]["display_name"] == "Warehouse handoff lane"
    assert payload["generation_request"]["world_prompt"]["text_prompt"].startswith("Preserve")
    assert "provider-adapter-input" in payload["generation_request"]["tags"]


def test_worldlabs_preview_provider_blocks_blocked_provider_adapter_input() -> None:
    provider = WorldLabsPreviewProvider()

    payload = provider._build_request_manifest(
        descriptor={"scene_id": "scene-1", "capture_id": "capture-1", "metadata": {}},
        capture_root=Path("/tmp/capture-root"),
        provider_adapter_input={
            "schema_version": "v1",
            "adapter_input_type": "ProviderAdapterInput",
            "provider": "world_labs",
            "adapter": "marble",
            "status": "blocked",
            "blockers": ["missing_privacy_safe_world_model_input"],
            "canonical_site_package_uri": "gs://bucket/pipeline/site_package/canonical_site_package.json",
            "provider_adapter_input_uri": "gs://bucket/pipeline/site_package/provider_adapter_inputs/world_labs_marble.json",
        },
    )

    assert payload["status"] == "blocked"
    assert payload["selected_video_uri"] is None
    assert payload["adapter_input_status"] == "blocked"
    assert "missing_privacy_safe_world_model_input" in payload["blockers"]
    assert payload["generation_source_type"] is None


def test_worldlabs_poll_reads_world_from_operation_response(monkeypatch) -> None:
    provider = WorldLabsPreviewProvider()
    launch_url = "https://marble.worldlabs.ai/worlds/world-1"

    def _fake_worldlabs_api_request(path: str, *, method: str = "GET", body=None) -> dict[str, object]:
        assert method == "GET"
        assert body is None
        assert path == "/marble/v1/operations/op-1"
        return {
            "done": True,
            "operation_id": "op-1",
            "response": {
                "world_id": "world-1",
                "world_marble_url": launch_url,
                "model": "marble-1.1",
            },
        }

    monkeypatch.setattr(
        "blueprint_pipeline.provider_preview._worldlabs_api_request",
        _fake_worldlabs_api_request,
    )

    result = provider.poll(run_id="op-1")

    assert result["status"] == "ready"
    assert result["world_id"] == "world-1"
    assert result["launch_url"] == launch_url


def test_worldlabs_api_request_preserves_slash_after_base_url(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"ok": true}'

    def _fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setenv("WORLDLABS_API_KEY", "test-key")
    monkeypatch.setattr("blueprint_pipeline.provider_preview._urllib_request.urlopen", _fake_urlopen)

    result = _worldlabs_api_request(
        "/marble/v1/media-assets:prepare_upload",
        method="POST",
        body={"file_name": "clip.mp4", "extension": "mp4", "kind": "video"},
    )

    assert result == {"ok": True}
    assert captured["url"] == "https://api.worldlabs.ai/marble/v1/media-assets:prepare_upload"


def test_run_preview_provider_persists_worldlabs_launch_url_aliases(tmp_path: Path, monkeypatch) -> None:
    launch_url = "https://marble.worldlabs.ai/worlds/world-2"
    labeling = {
        "non_production": True,
        "unredacted_input": True,
        "raw_video_bypass_used": True,
        "review_state": "non_production_unredacted_raw_preview",
    }

    def _fake_submit(self, *, descriptor, capture_root):  # type: ignore[no-untyped-def]
        del descriptor, capture_root
        return {
            "provider_name": self.provider_name,
            "provider_model": self.provider_model,
            "provider_run_id": "op-2",
            "status": "processing",
            "artifact_uris": {},
            "cost_usd": 0.0,
            "latency_ms": 1,
            "worldlabs_operation_id": "op-2",
            "worldlabs_media_asset_id": "media-2",
            "worldlabs_upload_id": "upload-2",
            "selected_input_checksum_sha256": "abc123",
            "source_manifest_uri": "gs://bucket/pipeline/privacy_processing_manifest.json",
            "worldlabs_input_audit_uri": "gs://bucket/pipeline/worldlabs_input_audit.json",
            "privacy_safe_input": True,
            "labeling": labeling,
        }

    def _fake_poll(self, *, run_id):  # type: ignore[no-untyped-def]
        assert run_id == "op-2"
        return {
            "provider_run_id": run_id,
            "status": "ready",
            "world_id": "world-2",
            "launch_url": launch_url,
            "worldlabs_operation": {"done": True, "operation_id": run_id},
            "worldlabs_world": {
                "world_id": "world-2",
                "world_marble_url": launch_url,
            },
        }

    monkeypatch.setattr(WorldLabsPreviewProvider, "submit", _fake_submit)
    monkeypatch.setattr(WorldLabsPreviewProvider, "poll", _fake_poll)

    result = run_preview_provider(
        provider_name="world_labs",
        descriptor={"capture_id": "cap-4", "raw_prefix_uri": "gs://bucket/raw"},
        capture_root=tmp_path,
        pipeline_dir=tmp_path,
    )

    preview_manifest = json.loads((tmp_path / "preview_manifest.json").read_text(encoding="utf-8"))

    assert result["provider_model"] == "marble-1.1"
    assert result["world_id"] == "world-2"
    assert result["launch_url"] == launch_url
    assert result["worldlabs_launch_url"] == launch_url
    assert result["preview_launch_url"] == launch_url
    assert result["worldlabs_operation_id"] == "op-2"
    assert result["worldlabs_media_asset_id"] == "media-2"
    assert result["worldlabs_upload_id"] == "upload-2"
    assert result["selected_input_checksum_sha256"] == "abc123"
    assert result["worldlabs_input_audit_uri"].endswith("/worldlabs_input_audit.json")
    assert result["operation_terminal_status"] == "ready"
    assert result["labeling"] == labeling
    assert preview_manifest["worldlabs_launch_url"] == launch_url
    assert preview_manifest["preview_launch_url"] == launch_url
    assert preview_manifest["labeling"] == labeling
