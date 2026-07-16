"""Takedown recall EXECUTION + drill (audit finding R049).

Enumeration is not recall. These tests pin the execution contract the finding
says was missing: a revoked capture's enumerated downstream DERIVED artifacts are
actively recalled (quarantined) with a per-artifact marker, the buyer-entitlement
revocation is handed off to the webapp, and an executed-recall audit record is
written. The drill runs the whole thing end-to-end against a fixture capture and
proves every enumerated target reaches a terminal recalled/quarantined state — or,
when a target cannot be recalled, an EXPLICIT blocked state (never a false success).

Capture-truth boundary: only derived deliverables are recalled. The raw capture
bundle is authoritative and is never modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import blueprint_pipeline.consent_takedown as ct


SCENE_ID = "scene-r049"
CAPTURE_ID = "capture-r049"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _make_revoked_capture(tmp_path: Path) -> Path:
    """A revoked capture that already produced synced + hosted derived artifacts."""
    capture_root = tmp_path / "scenes" / SCENE_ID / "captures" / CAPTURE_ID
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": SCENE_ID, "capture_id": CAPTURE_ID},
    )
    _write_json(
        capture_root / "raw" / "rights_consent.json",
        {
            "consent_status": "revoked",
            "consent_revoked": True,
            "consent_revoked_at": "2026-07-04T12:00:00Z",
            "consent_scope": ["robot_evaluation", "model_training"],
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
        {"scene_id": SCENE_ID, "capture_id": CAPTURE_ID, "default_backend": "isaac"},
    )
    _write_json(
        pipeline / "webapp_sync_result.json",
        {
            "status": "succeeded",
            "syncs": {
                "completion": {
                    "status": "succeeded",
                    "webapp_response_ids": {
                        "attachment_id": "att_123",
                        "entitlement_id": "ent_789",
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


def _make_export_chain(tmp_path: Path) -> Path:
    """Derived-of-derived exports OUTSIDE the capture tree (delivered package)."""
    exports_root = tmp_path / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    (exports_root / "generated_video.mp4").write_bytes(b"fake-mp4-bytes")
    _write_json(
        exports_root / "generated_video_manifest.json",
        {"scene_id": SCENE_ID, "capture_id": CAPTURE_ID, "video_path": "generated_video.mp4"},
    )
    (exports_root / "clip_0001.mp4").write_bytes(b"fake-clip-bytes")
    _write_json(
        exports_root / "ptdp_clip_manifest.json",
        {
            "clip_id": "clip_0001",
            "clip_path": "clip_0001.mp4",
            "source_video_path": str(exports_root / "generated_video.mp4"),
        },
    )
    return exports_root


# ---------------------------------------------------------------------------
# The drill: every enumerated target reaches a terminal recalled state.
# ---------------------------------------------------------------------------


def test_takedown_drill_recalls_every_enumerated_target(tmp_path: Path) -> None:
    capture_root = _make_revoked_capture(tmp_path)
    exports_root = _make_export_chain(tmp_path)

    drill = ct.run_takedown_drill(
        capture_root=capture_root,
        additional_artifact_roots=[exports_root],
    )

    # Drill passed: full coverage, everything recalled, nothing left unresolved.
    assert drill["schema_version"] == ct.TAKEDOWN_DRILL_SCHEMA_VERSION
    assert drill["status"] == "passed"
    assert drill["coverage_complete"] is True
    assert drill["all_recalled"] is True
    assert drill["unresolved_targets"] == []
    assert drill["enumerated_target_count"] > 0

    execution = drill["execution"]
    assert execution["schema_version"] == ct.RECALL_EXECUTION_SCHEMA_VERSION
    assert execution["status"] == "executed"
    assert execution["executed"] is True
    assert execution["blocked_count"] == 0
    assert execution["target_count"] == drill["enumerated_target_count"]

    # Every enumerated target reached a terminal recalled/quarantined state, and
    # each has a recall marker recording reason + timestamp.
    for target in execution["targets"]:
        assert target["terminal"] is True
        assert target["blocked"] is False
        assert target["outcome"] in {"quarantined", "recalled_absent"}
        assert target["recalled_at"]
        marker = json.loads(Path(target["marker_path"]).read_text(encoding="utf-8"))
        assert marker["schema_version"] == ct.RECALL_MARKER_SCHEMA_VERSION
        assert marker["reason"] == "consent_revoked"
        assert marker["serve_allowed"] is False
        assert marker["training_use_allowed"] is False

    # The delivered derived deliverables are actively gone from their delivery
    # locations (quarantined), not merely listed.
    assert not (capture_root / "pipeline" / "site_world_spec.json").exists()
    assert not (capture_root / "pipeline" / "hosted_session_runtime_manifest.json").exists()
    assert not (exports_root / "generated_video.mp4").exists()
    assert not (exports_root / "clip_0001.mp4").exists()

    # Quarantined bytes are preserved under the takedown quarantine dir.
    quarantined = list((capture_root / ct.QUARANTINE_RELATIVE_DIR).rglob("*"))
    assert any(p.is_file() for p in quarantined)

    # Webapp buyer-entitlement revocation is handed off (webapp-owned), with the
    # explicit revoked verdict — and honestly reported as not-yet-executed here.
    handoff = execution["webapp_handoff"]
    assert handoff["signal"]["verdict"] == "revoked"
    assert handoff["executed"] is False
    assert execution["webapp_entitlement_revocation_executed"] is False

    # Capture-truth: the raw bundle is untouched and unchanged.
    raw_consent = capture_root / "raw" / "rights_consent.json"
    raw_manifest = capture_root / "raw" / "manifest.json"
    assert raw_consent.exists() and raw_manifest.exists()
    assert json.loads(raw_consent.read_text())["consent_status"] == "revoked"
    assert not any(t["path"].startswith(str(capture_root / "raw")) for t in execution["targets"])
    assert execution["capture_truth_boundary"]["raw_capture_bundle_never_modified"] is True

    # The executed-recall audit record is persisted where operators can find it.
    record_path = capture_root / ct.RECALL_EXECUTION_RECORD_RELATIVE_PATH
    assert record_path.is_file()
    assert json.loads(record_path.read_text())["status"] == "executed"


def test_takedown_drill_surfaces_blocked_target_no_false_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-closed: a target that cannot be recalled is an EXPLICIT blocked
    state that drives the record to blocked_needs_operator — never a silent pass."""
    capture_root = _make_revoked_capture(tmp_path)

    real_move = ct._move_artifact_to_quarantine

    def _flaky_move(src: Path, dst: Path) -> None:
        if Path(src).name == "site_world_spec.json":
            raise PermissionError("simulated: cannot recall this target")
        real_move(src, dst)

    monkeypatch.setattr(ct, "_move_artifact_to_quarantine", _flaky_move)

    drill = ct.run_takedown_drill(capture_root=capture_root)

    # Coverage is still complete (the blocked target is explicit, not missed)...
    assert drill["coverage_complete"] is True
    assert drill["unresolved_targets"] == []
    # ...but the drill does NOT report a clean pass, and nothing claims success.
    assert drill["status"] == "blocked"
    assert drill["all_recalled"] is False

    execution = drill["execution"]
    assert execution["status"] == "blocked_needs_operator"
    assert execution["executed"] is False
    assert execution["blocked_count"] == 1

    blocked = [t for t in execution["targets"] if t["blocked"]]
    assert len(blocked) == 1
    assert blocked[0]["relative_path"] == "pipeline/site_world_spec.json"
    assert blocked[0]["outcome"] == "blocked"
    assert blocked[0]["needs_operator"] is True
    assert blocked[0]["terminal"] is False
    assert any(
        b.startswith("recall_blocked:") for b in execution["blockers"]
    )

    # The blocked target is still on disk (recall did not silently succeed) and
    # carries a marker recording the blocked state for the operator.
    assert (capture_root / "pipeline" / "site_world_spec.json").exists()
    marker = json.loads(Path(blocked[0]["marker_path"]).read_text())
    assert marker["status"] == "blocked"
    assert marker["needs_operator"] is True

    # Non-blocked targets still reached a terminal recalled state.
    recalled = [t for t in execution["targets"] if not t["blocked"]]
    assert recalled
    assert all(t["terminal"] for t in recalled)


def test_execute_active_consent_is_not_required(tmp_path: Path) -> None:
    """No revocation: nothing to recall, and the record says so plainly."""
    capture_root = tmp_path / "scenes" / SCENE_ID / "captures" / CAPTURE_ID
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": SCENE_ID, "capture_id": CAPTURE_ID},
    )
    _write_json(
        capture_root / "raw" / "rights_consent.json",
        {"consent_status": "documented", "consent_scope": ["robot_evaluation"]},
    )
    _write_json(capture_root / "pipeline" / "site_world_spec.json", {"x": 1})

    drill = ct.run_takedown_drill(capture_root=capture_root)
    assert drill["status"] == "not_required"
    assert drill["execution"]["status"] == "not_required"
    assert drill["execution"]["executed"] is False
    # The derived artifact is untouched when consent is intact.
    assert (capture_root / "pipeline" / "site_world_spec.json").exists()


def test_execute_syncs_webapp_revocation_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the signed channel is configured and sync is requested, the webapp
    entitlement-revocation handoff is actually pushed and reported executed."""
    import io

    capture_root = _make_revoked_capture(tmp_path)
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.example/sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "token")

    posted: dict[str, Any] = {}

    def _fake_urlopen(request, timeout=None):  # noqa: ANN001
        posted["body"] = json.loads(request.data.decode("utf-8"))
        return io.BytesIO(json.dumps({"ok": True, "id": "revocation_ack_1"}).encode())

    monkeypatch.setattr(ct.urllib_request, "urlopen", _fake_urlopen)

    execution = ct.execute_consent_takedown(capture_root=capture_root, sync_webapp=True)

    assert execution["status"] == "executed"
    assert execution["webapp_entitlement_revocation_executed"] is True
    assert execution["webapp_handoff"]["executed"] is True
    assert posted["body"]["verdict"] == "revoked"
    assert posted["body"]["capture_id"] == CAPTURE_ID
