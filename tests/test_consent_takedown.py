"""Consent-revocation takedown propagation: rights are authoritative continuously.

The build-time consent gate stops NEW exports, but a revocation that lands after
delivery must recall what already shipped. These tests pin the recall contract:

1. propagate_consent_takedown(capture) enumerates every downstream artifact whose
   lineage traces to the revoked capture — including derived-of-derived chains —
   and emits a fail-closed takedown_manifest.v1 with a tombstone per artifact.
2. The webapp receives an explicit revoked VERDICT, never mere absence.
3. A delivery-time gate refuses to serve/sync any artifact whose source capture
   has an open takedown, even before a takedown manifest has been generated.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

import blueprint_pipeline.consent_takedown as ct
import blueprint_pipeline.webapp_sync as webapp_sync


SCENE_ID = "scene-e2f1"
CAPTURE_ID = "capture-77aa"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _make_capture(tmp_path: Path, *, consent_status: str = "documented") -> Path:
    """A capture root that already produced a synced package and hosted artifacts."""
    capture_root = tmp_path / "scenes" / SCENE_ID / "captures" / CAPTURE_ID
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": SCENE_ID, "capture_id": CAPTURE_ID},
    )
    _write_json(
        capture_root / "raw" / "rights_consent.json",
        {
            "consent_status": consent_status,
            "consent_scope": ["robot_evaluation", "model_training"],
            "permission_document_uri": "gs://bucket/consent.pdf",
            **(
                {"consent_revoked": True, "consent_revoked_at": "2026-07-04T00:00:00Z"}
                if consent_status == "revoked"
                else {}
            ),
        },
    )
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": SCENE_ID, "capture_id": CAPTURE_ID},
    )
    pipeline = capture_root / "pipeline"
    _write_json(
        pipeline / "site_world_spec.json",
        {
            "scene_id": SCENE_ID,
            "capture_id": CAPTURE_ID,
            "canonical_package_uri": "gs://bucket/site_world/spec.json",
            "canonical_package_version": "abc123",
        },
    )
    _write_json(
        pipeline / "hosted_session_runtime_manifest.json",
        {
            "schema_version": "hosted_session_runtime_claim_boundary.v1",
            "scene_id": SCENE_ID,
            "capture_id": CAPTURE_ID,
            "default_backend": "isaac",
        },
    )
    _write_json(
        pipeline / "webapp_sync_result.json",
        {
            "status": "succeeded",
            "latest_stage": "completion",
            "syncs": {
                "completion": {
                    "status": "succeeded",
                    "webapp_response_ids": {
                        "attachment_id": "att_123",
                        "site_world_id": "sw_456",
                    },
                    "attachment_payload": {
                        "scene_id": SCENE_ID,
                        "capture_id": CAPTURE_ID,
                        "site_submission_id": "sub_real_1",
                        "request_id": "req_real_1",
                    },
                }
            },
        },
    )
    _write_json(
        capture_root
        / "robot_eval_jobs"
        / "job1"
        / "post_training_data_package_export_manifest.json",
        {
            "schema_version": "post_training_data_package_export.v1",
            "scene_id": SCENE_ID,
            "capture_id": CAPTURE_ID,
            "status": "exported",
        },
    )
    return capture_root


def _make_two_level_export_chain(tmp_path: Path) -> Path:
    """An exports root OUTSIDE the capture tree with a 2-deep derivation chain.

    generated_video_manifest.json carries the capture_id (depth 1).
    ptdp_clip_manifest.json carries NO capture/scene ids — it only references the
    generated video (depth 2). clip_0001.mp4 is referenced by the clip manifest.
    """
    exports_root = tmp_path / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    video_path = exports_root / "generated_video.mp4"
    video_path.write_bytes(b"fake-mp4-bytes")
    _write_json(
        exports_root / "generated_video_manifest.json",
        {
            "scene_id": SCENE_ID,
            "capture_id": CAPTURE_ID,
            "video_path": "generated_video.mp4",
        },
    )
    clip_path = exports_root / "clip_0001.mp4"
    clip_path.write_bytes(b"fake-clip-bytes")
    _write_json(
        exports_root / "ptdp_clip_manifest.json",
        {
            "clip_id": "clip_0001",
            "clip_path": "clip_0001.mp4",
            "source_video_path": str(exports_root / "generated_video.mp4"),
        },
    )
    return exports_root


def _revoke(capture_root: Path) -> None:
    _write_json(
        capture_root / "raw" / "rights_consent.json",
        {
            "consent_status": "revoked",
            "consent_revoked": True,
            "consent_revoked_at": "2026-07-04T12:00:00Z",
            "consent_scope": ["robot_evaluation", "model_training"],
        },
    )


def _artifact_relpaths(manifest: dict[str, Any]) -> set[str]:
    return {entry["relative_path"] for entry in manifest["artifacts"]}


# ---------------------------------------------------------------------------
# propagate_consent_takedown
# ---------------------------------------------------------------------------


def test_propagate_revoked_capture_lists_all_derived_artifacts(tmp_path: Path) -> None:
    capture_root = _make_capture(tmp_path, consent_status="revoked")
    manifest = ct.propagate_consent_takedown(capture_root=capture_root)

    assert manifest["schema_version"] == ct.TAKEDOWN_MANIFEST_SCHEMA_VERSION
    assert manifest["status"] == "takedown_open"
    assert manifest["takedown_open"] is True
    assert manifest["scene_id"] == SCENE_ID
    assert manifest["capture_id"] == CAPTURE_ID

    relpaths = _artifact_relpaths(manifest)
    assert "pipeline/site_world_spec.json" in relpaths
    assert "pipeline/hosted_session_runtime_manifest.json" in relpaths
    assert (
        "robot_eval_jobs/job1/post_training_data_package_export_manifest.json"
        in relpaths
    )

    for entry in manifest["artifacts"]:
        tombstone = entry["tombstone"]
        assert tombstone["status"] == "takedown_required"
        assert tombstone["reason"] == "consent_revoked"
        assert tombstone["capture_id"] == CAPTURE_ID
        assert tombstone["serve_allowed"] is False
        assert tombstone["training_use_allowed"] is False
        assert entry["lineage"]["traces_to_capture"] is True

    # The manifest is persisted where delivery gates can find it.
    written = (
        capture_root / "pipeline" / "consent_takedown" / "takedown_manifest.json"
    )
    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8"))["takedown_open"] is True


def test_propagate_excludes_raw_capture_truth(tmp_path: Path) -> None:
    capture_root = _make_capture(tmp_path, consent_status="revoked")
    manifest = ct.propagate_consent_takedown(capture_root=capture_root)
    assert not any(
        relpath.startswith("raw/") for relpath in _artifact_relpaths(manifest)
    )


def test_propagate_active_consent_is_not_required(tmp_path: Path) -> None:
    capture_root = _make_capture(tmp_path, consent_status="documented")
    manifest = ct.propagate_consent_takedown(capture_root=capture_root)
    assert manifest["status"] == "not_required"
    assert manifest["takedown_open"] is False
    assert all(entry["tombstone"] == {} for entry in manifest["artifacts"])


def test_propagate_unknown_consent_state_fails_closed(tmp_path: Path) -> None:
    capture_root = tmp_path / "scenes" / SCENE_ID / "captures" / CAPTURE_ID
    (capture_root / "pipeline").mkdir(parents=True)
    _write_json(capture_root / "pipeline" / "some_artifact.json", {"x": 1})

    manifest = ct.propagate_consent_takedown(capture_root=capture_root)
    assert manifest["status"] == "takedown_open_consent_state_unknown"
    assert manifest["takedown_open"] is True
    assert "consent_state_unverifiable" in manifest["blockers"]


def test_webapp_revocation_signal_is_explicit_verdict(tmp_path: Path) -> None:
    capture_root = _make_capture(tmp_path, consent_status="revoked")
    manifest = ct.propagate_consent_takedown(capture_root=capture_root)

    signal = manifest["webapp_revocation_signal"]
    assert signal["schema_version"] == ct.WEBAPP_REVOCATION_SIGNAL_SCHEMA_VERSION
    assert signal["verdict"] == "revoked"
    assert signal["verdict_is_explicit_not_absence"] is True
    assert signal["capture_id"] == CAPTURE_ID
    assert (
        signal["required_webapp_state"]
        == "blocked_consent_revoked_takedown_required"
    )
    # Already-synced webapp state is enumerated so the recall targets real ids.
    webapp_surface = manifest["external_surfaces"]["webapp"]
    assert webapp_surface["webapp_response_ids"].get("attachment_id") == "att_123"

    # Execution is never claimed locally: sync is queued until proven.
    assert manifest["webapp_revocation_sync"]["executed"] is False
    assert "webapp_revocation_sync_not_executed" in manifest["blockers"]


def test_two_level_derived_chain_fully_caught(tmp_path: Path) -> None:
    """Adversarial: generated video -> PTDP clip, clip carries no capture ids."""
    capture_root = _make_capture(tmp_path, consent_status="revoked")
    exports_root = _make_two_level_export_chain(tmp_path)

    manifest = ct.propagate_consent_takedown(
        capture_root=capture_root,
        additional_artifact_roots=[exports_root],
    )

    by_name = {Path(entry["path"]).name: entry for entry in manifest["artifacts"]}
    assert "generated_video_manifest.json" in by_name
    assert "generated_video.mp4" in by_name
    assert "ptdp_clip_manifest.json" in by_name
    assert "clip_0001.mp4" in by_name

    assert by_name["generated_video_manifest.json"]["lineage"]["depth"] == 1
    clip_lineage = by_name["ptdp_clip_manifest.json"]["lineage"]
    assert clip_lineage["depth"] == 2
    assert clip_lineage["traces_to_capture"] is True
    assert any("generated_video" in via for via in clip_lineage["via"])
    assert by_name["ptdp_clip_manifest.json"]["tombstone"]["status"] == "takedown_required"
    assert by_name["clip_0001.mp4"]["tombstone"]["status"] == "takedown_required"


# ---------------------------------------------------------------------------
# Delivery-time gate
# ---------------------------------------------------------------------------


def test_delivery_gate_allows_active_consent(tmp_path: Path) -> None:
    capture_root = _make_capture(tmp_path, consent_status="documented")
    verdict = ct.evaluate_delivery_time_takedown_gate(capture_root=capture_root)
    assert verdict["status"] == "allowed"
    assert verdict["serve_allowed"] is True
    assert verdict["blockers"] == []


def test_delivery_gate_treats_string_false_revocation_as_active(
    tmp_path: Path,
) -> None:
    capture_root = _make_capture(tmp_path, consent_status="documented")
    _write_json(
        capture_root / "raw" / "rights_consent.json",
        {
            "consent_status": "documented",
            "consent_revoked": "false",
            "consentRevoked": "false",
            "consent_scope": ["robot_evaluation", "model_training"],
        },
    )

    verdict = ct.evaluate_delivery_time_takedown_gate(capture_root=capture_root)

    assert verdict["status"] == "allowed"
    assert verdict["serve_allowed"] is True
    assert verdict["blockers"] == []


def test_delivery_gate_blocks_string_true_revocation(tmp_path: Path) -> None:
    capture_root = _make_capture(tmp_path, consent_status="documented")
    _write_json(
        capture_root / "raw" / "rights_consent.json",
        {
            "consent_status": "documented",
            "consent_revoked": "true",
            "consent_scope": ["robot_evaluation", "model_training"],
        },
    )

    verdict = ct.evaluate_delivery_time_takedown_gate(capture_root=capture_root)

    assert verdict["status"] == "blocked_open_consent_takedown"
    assert verdict["serve_allowed"] is False
    assert "consent_revoked_takedown_required" in verdict["blockers"]


def test_delivery_gate_blocks_revocation_even_without_manifest(tmp_path: Path) -> None:
    """Rights are authoritative continuously: the gate re-reads consent live."""
    capture_root = _make_capture(tmp_path, consent_status="documented")
    _revoke(capture_root)

    verdict = ct.evaluate_delivery_time_takedown_gate(capture_root=capture_root)
    assert verdict["status"] == "blocked_open_consent_takedown"
    assert verdict["serve_allowed"] is False
    assert "consent_revoked_takedown_required" in verdict["blockers"]


def test_delivery_gate_blocks_unknown_consent_state(tmp_path: Path) -> None:
    capture_root = tmp_path / "scenes" / SCENE_ID / "captures" / CAPTURE_ID
    (capture_root / "pipeline").mkdir(parents=True)
    verdict = ct.evaluate_delivery_time_takedown_gate(capture_root=capture_root)
    assert verdict["status"] == "blocked_consent_state_unverifiable"
    assert verdict["serve_allowed"] is False


def test_reserving_hosted_artifact_fails_closed_after_revocation(tmp_path: Path) -> None:
    capture_root = _make_capture(tmp_path, consent_status="documented")
    hosted = capture_root / "pipeline" / "hosted_session_runtime_manifest.json"

    before = ct.evaluate_delivery_time_takedown_gate(
        capture_root=capture_root, artifact_path=hosted, surface="hosted_sessions"
    )
    assert before["status"] == "allowed"

    _revoke(capture_root)
    ct.propagate_consent_takedown(capture_root=capture_root)

    after = ct.evaluate_delivery_time_takedown_gate(
        capture_root=capture_root, artifact_path=hosted, surface="hosted_sessions"
    )
    assert after["status"] == "blocked_open_consent_takedown"
    assert after["serve_allowed"] is False
    assert after["artifact_path"] == str(hosted)
    assert after["takedown_manifest_present"] is True


def test_delivery_gate_honors_open_manifest_even_if_consent_flips_back(
    tmp_path: Path,
) -> None:
    """A written takedown stays authoritative until explicitly re-propagated."""
    capture_root = _make_capture(tmp_path, consent_status="revoked")
    ct.propagate_consent_takedown(capture_root=capture_root)
    # Consent source flips back without re-running propagation.
    _write_json(
        capture_root / "raw" / "rights_consent.json",
        {"consent_status": "documented", "consent_scope": ["robot_evaluation"]},
    )
    verdict = ct.evaluate_delivery_time_takedown_gate(capture_root=capture_root)
    assert verdict["status"] == "blocked_open_consent_takedown"


# ---------------------------------------------------------------------------
# Webapp sync: revoked verdict + refuse-to-sync on open takedown
# ---------------------------------------------------------------------------


def test_sync_webapp_attachment_refuses_open_takedown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = _make_capture(tmp_path, consent_status="revoked")
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.example/sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "token")

    with pytest.raises(webapp_sync.WebappSyncError, match="open_consent_takedown"):
        webapp_sync.sync_webapp_pipeline_attachment(
            site_submission_id="sub_real_1",
            request_id="req_real_1",
            buyer_request_id="breq_real_1",
            capture_job_id="cj_real_1",
            scene_id=SCENE_ID,
            capture_id=CAPTURE_ID,
            pipeline_prefix="gs://bucket/pipeline",
            qualification_state="qualified_ready",
            opportunity_state="handoff_ready",
            artifacts={"site_world_spec": "gs://bucket/site_world/spec.json"},
            capture_root=capture_root,
        )


def test_sync_webapp_attachment_fails_closed_on_takedown_when_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = _make_capture(tmp_path, consent_status="revoked")
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)

    result = webapp_sync.sync_webapp_pipeline_attachment(
        site_submission_id="sub_real_1",
        request_id="req_real_1",
        buyer_request_id="breq_real_1",
        capture_job_id="cj_real_1",
        scene_id=SCENE_ID,
        capture_id=CAPTURE_ID,
        pipeline_prefix="gs://bucket/pipeline",
        qualification_state="qualified_ready",
        opportunity_state="handoff_ready",
        artifacts={},
        capture_root=capture_root,
    )
    assert result["status"] == "failed"
    assert result["blocker"] == "open_consent_takedown"


def test_sync_webapp_consent_revocation_not_configured_is_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = _make_capture(tmp_path, consent_status="revoked")
    manifest = ct.propagate_consent_takedown(capture_root=capture_root)
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)

    result = ct.sync_webapp_consent_revocation(takedown_manifest=manifest)
    assert result["status"] == "queued_unexecuted_webapp_revocation_sync"
    assert result["executed"] is False
    assert "webapp_revocation_sync_not_configured" in result["blockers"]


def test_sync_webapp_consent_revocation_posts_revoked_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = _make_capture(tmp_path, consent_status="revoked")
    manifest = ct.propagate_consent_takedown(capture_root=capture_root)
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.example/sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "token")

    posted: dict[str, Any] = {}

    def _fake_urlopen(request, timeout=None):  # noqa: ANN001
        posted["body"] = json.loads(request.data.decode("utf-8"))
        posted["headers"] = dict(request.header_items())
        return io.BytesIO(json.dumps({"ok": True, "id": "revocation_ack_1"}).encode())

    monkeypatch.setattr(ct.urllib_request, "urlopen", _fake_urlopen)

    result = ct.sync_webapp_consent_revocation(takedown_manifest=manifest)
    assert result["status"] == "executed"
    assert result["executed"] is True

    body = posted["body"]
    assert body["schema_version"] == ct.WEBAPP_REVOCATION_SIGNAL_SCHEMA_VERSION
    assert body["verdict"] == "revoked"
    assert body["capture_id"] == CAPTURE_ID
    assert body["verdict_is_explicit_not_absence"] is True
    assert body["artifact_count"] == manifest["artifact_count"]
    # The signed sync channel is reused, so the webapp can authenticate the verdict.
    header_names = {name.lower() for name in posted["headers"]}
    assert "x-blueprint-pipeline-signature" in header_names
