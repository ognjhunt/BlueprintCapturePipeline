"""Website preparation skips optional reports while retaining admission boundaries."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import site_package_orchestrator as orchestrator
from blueprint_pipeline import website_scene_handoff, website_task_evidence
from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_qualification_coverage_edges import _descriptor, _write_descriptor


@pytest.fixture
def website_capture(tmp_path, monkeypatch):
    storage = tmp_path / "gcs"
    descriptor = _descriptor(
        capture_source="unknown", capture_modality="video_only", requested_outputs=[],
        raw_video_uri="gs://bucket/scenes/scene-1/captures/capture-1/raw/job.mov",
        metadata={"capture_entry_source": "browser_self_capture", "owner_note": "retain original"},
    )
    uri = _write_descriptor(storage, descriptor)
    capture = storage / "scenes/scene-1/captures/capture-1"
    source = capture / "raw/job.mov"
    source.parent.mkdir()
    source.write_bytes(b"retained-video-fixture")
    (capture / "capture_descriptor.json").write_text(json.dumps(descriptor.to_dict()))
    context = {
        "schema_version": "website_site_task_context.v1", "request_id": "",
        "scene_id": descriptor.scene_id, "capture_id": descriptor.capture_id,
        "purpose": "scene_preparation", "confirmed": False, "confirmed_at": None,
        "description": "Move the box to the shelf", "task_family": "rigid_pick_place",
        "capture_rights": {"derived_scene_generation_allowed": True},
    }
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    monkeypatch.setattr(orchestrator, "load_current_website_task_context", lambda **_: context)
    sponsorship = {"schema_version": "website_scene_sponsorship.v1", "purpose": "scene_preparation"}
    monkeypatch.setattr(orchestrator, "load_website_scene_sponsorship", lambda **_: sponsorship)
    clean_plate = {
        "status": "noop", "mode": "task_aware_prepared_images",
        "privacy_status": "website_capture_admitted", "privacy_verified": True,
    }
    monkeypatch.setattr(orchestrator, "run_clean_plate_stage", lambda **_: clean_plate)
    outbox = {"state": "forward_pending", "id": "native-scene-1"}
    monkeypatch.setattr(website_scene_handoff, "prepare_website_scene_handoff", lambda **_: {
        "status": "intake_ready", "website_intake_outbox": outbox,
    })
    return SimpleNamespace(
        storage=storage, capture=capture, source=source, uri=uri,
        context=context, clean_plate=clean_plate, outbox=outbox,
        config=SimpleNamespace(gcs_root=storage, emit_readiness_support_outputs=True),
    )


@pytest.mark.parametrize("entry_source", ["browser_self_capture", "site_self_capture"])
def test_website_handoff_succeeds_without_generic_reports_or_object_index(
    website_capture, monkeypatch, entry_source,
):
    fx = website_capture
    descriptor_path = fx.capture / "capture_descriptor.json"
    original = json.loads(descriptor_path.read_text())
    original["metadata"]["capture_entry_source"] = entry_source
    descriptor_path.write_text(json.dumps(original))

    def forbidden(*_args, **_kwargs):
        pytest.fail("website entered optional generic readiness/reporting")

    for name in (
        "load_raw_manifest", "ensure_object_index_stage", "infer_task_targets",
        "_build_runtime_preflight_report", "_build_completeness_scorecard",
        "_build_qualification_record", "_build_readiness_decision",
        "build_capture_enrichment_runner", "write_blueprint_canonical_site_package",
        "run_privacy_postprocess", "infer_capture_fidelity_review",
        "_prepare_worldlabs_input_video", "build_geometry_stage_contract",
        "sync_webapp_pipeline_attachment",
    ):
        monkeypatch.setattr(orchestrator, name, forbidden)
    monkeypatch.setattr(orchestrator.support_outputs, "write_alpha_readiness_summary_if_enabled", forbidden)
    handoffs = []

    def prepare_media(**kwargs):
        assert kwargs["website_source_video"] == fx.source
        assert kwargs["task_context"] == fx.context
        assert kwargs["policy"].enabled is True
        return fx.clean_plate

    def handoff(**kwargs):
        handoffs.append(kwargs)
        return {"status": "intake_ready", "website_intake_outbox": fx.outbox}

    monkeypatch.setattr(orchestrator, "run_clean_plate_stage", prepare_media)
    monkeypatch.setattr(website_scene_handoff, "prepare_website_scene_handoff", handoff)
    result = orchestrator.run_qualification_pipeline(descriptor_gcs_uri=fx.uri, config=fx.config)
    assert result["status"] == "completed"
    assert result["native_execution_complete"] is False
    assert len(handoffs) == 1
    retained = json.loads(descriptor_path.read_text())
    assert retained["raw_video_uri"] == original["raw_video_uri"]
    assert retained["metadata"]["owner_note"] == "retain original"
    assert retained["metadata"]["site_task_context"] == fx.context
    assert retained["metadata"]["website_scene_execution_authority"]["purpose"] == "scene_preparation"
    assert retained["metadata"]["privacy_processing"]["fail_closed"] is False
    assert retained["privacy_status"] == "website_capture_admitted"
    assert not any((fx.capture / "pipeline" / name).exists() for name in (
        "qualification_quality_report.json", "capture_qa_scorecard.json", "readiness_decision.json",
        "qualification_record.json", "qualification_weakness_summary.json", "recapture_instructions.json",
        "canonical_site_package.json", "buyer_trust_score.json",
    ))
    publication = json.loads((fx.capture / "pipeline/provider_run_manifest.json").read_text())
    assert publication["website_scene_preparation"]["website_intake_outbox"] == fx.outbox


@pytest.mark.parametrize("failure,reason", [
    ("withdrawn", "website_capture_withdrawn"),
    ("rights_denied", "website_item_consent_not_granted"),
    ("sponsorship", "website_scene_sponsorship_unavailable"),
    ("clean_plate", "website_source_geometry_unavailable"),
    ("privacy_unverified", "website_scene_preparation_pending"),
    ("privacy_failed", "website_scene_preparation_pending"),
])
def test_direct_website_path_preserves_source_grant_and_privacy_refusals(
    website_capture, monkeypatch, failure, reason,
):
    fx = website_capture
    if failure == "withdrawn":
        monkeypatch.setattr(website_task_evidence, "read_consent_state", lambda *_: {"state": "revoked"})
    elif failure == "rights_denied":
        fx.context["capture_rights"]["derived_scene_generation_allowed"] = False
        fx.context["context_digest"] = canonical_digest(fx.context, digest_field="context_digest")
    elif failure == "sponsorship":
        def missing_sponsorship(**_):
            raise ValueError(reason)
        monkeypatch.setattr(orchestrator, "load_website_scene_sponsorship", missing_sponsorship)
    elif failure == "clean_plate":
        fx.clean_plate.update(status="blocked", blockers=[reason])
    elif failure == "privacy_unverified":
        fx.clean_plate["privacy_verified"] = False
    elif failure == "privacy_failed":
        fx.clean_plate["privacy_status"] = "failed_closed"
    monkeypatch.setattr(website_scene_handoff, "prepare_website_scene_handoff", lambda **_: pytest.fail(
        "unadmitted source reached native scene handoff"))
    with pytest.raises(PipelineError, match=reason):
        orchestrator.run_qualification_pipeline(descriptor_gcs_uri=fx.uri, config=fx.config)
    report = json.loads((fx.capture / "pipeline/qualification_quality_report.json").read_text())
    assert report["status"] == "failed"
    assert report["error"].endswith(reason)
    if failure == "clean_plate":
        assert report["failed_stage"] == "clean_plate"
    elif failure.startswith("privacy_"):
        assert report["failed_stage"] == "privacy_postprocess"
