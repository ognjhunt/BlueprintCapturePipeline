from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.common import StageError, PipelineError
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
import blueprint_pipeline.site_package_orchestrator as q


def _confirmed_website_task(**identity):
    context = {**identity, "schema_version": "website_site_task_context.v1",
               "description": "Pick the box", "confirmed": True,
               "confirmed_at": "2026-10-06T00:00:00Z",
               "capture_rights": {"derived_scene_generation_allowed": True}}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    return context


def _descriptor(**overrides) -> CaptureDescriptor:
    defaults = {
        "schema_version": "v1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "capture_source": "iphone",
        "capture_tier": "candidate",
        "raw_prefix_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw",
        "frames_index_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/frames.json",
        "quality": {},
        "raw_video_uri": "gs://bucket/captures/scene-1/capture-1/raw/final_walkthrough.mov",
        "environment_type_hint": "warehouse",
        "capture_modality": "iphone_arkit_lidar",
        "evidence_tier": "pre_screen_video",
        "requested_lanes": ["qualification"],
        "requested_outputs": ["qualification"],
        "coverage_plan": ["dock aisle", "task zone"],
        "metadata": {},
    }
    defaults.update(overrides)
    return CaptureDescriptor(**defaults)


def _write_descriptor(storage_root: Path, descriptor: CaptureDescriptor) -> str:
    descriptor_uri = "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json"
    descriptor_path = (
        storage_root / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    )
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(json.dumps(descriptor.to_dict()), encoding="utf-8")
    return descriptor_uri


def _patch_pipeline_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing left to stub: the capture app's stages were removed on 2026-11-11."""
    del monkeypatch


def test_website_preparation_skips_legacy_privacy_video_and_geometry(tmp_path, monkeypatch):
    storage_root = tmp_path / "gcs"
    descriptor_uri = _write_descriptor(storage_root, _descriptor(
        capture_source="unknown", capture_modality="video_only", requested_outputs=["preview_simulation"],
        metadata={"capture_entry_source": "browser_self_capture", "capture_rights": {"derived_scene_generation_allowed": True}}))
    _patch_pipeline_side_effects(monkeypatch)
    monkeypatch.setattr(q, "load_current_website_task_context", _confirmed_website_task)
    monkeypatch.setattr(q, "load_website_scene_sponsorship", lambda **_: {"sponsor": "blueprint"})
    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        return {"status": "blocked", "mode": "source_geometry_blocked", "privacy_status": "no_people_detected",
                "privacy_verified": True, "blockers": ["website_source_geometry_unavailable"]}

    monkeypatch.setattr(q, "run_clean_plate_stage", prepare)
    with pytest.raises(PipelineError, match="website_source_geometry_unavailable"):
        q.run_qualification_pipeline(descriptor_gcs_uri=descriptor_uri, config=SimpleNamespace(gcs_root=storage_root, runtime_preflight_enabled=False))
    assert len(calls) == 1
    assert calls[0]["policy"].enabled is True
    assert calls[0]["privacy_processing"]["mode"] == "task_aware_prepared_images"
    # The capture's Atlas job is the scene; no Marble input is prepared.
    assert calls[0]["worldlabs_input"]["status"] == "not_requested"


@pytest.mark.parametrize("sponsorship, enabled", [
    ({"sponsor": "blueprint", "purpose": "scene_preparation"}, True),
    ({"sponsor": "blueprint"}, False),
])
def test_browser_delivery_without_requested_outputs_prepares_only_a_sponsored_scene(tmp_path, monkeypatch, sponsorship, enabled):
    # Real browser deliveries carry requested_outputs == []; only a scene_preparation grant requests the preview.
    storage_root = tmp_path / "gcs"
    descriptor_uri = _write_descriptor(storage_root, _descriptor(
        capture_source="unknown", capture_modality="video_only", requested_outputs=[],
        metadata={"capture_entry_source": "browser_self_capture", "capture_rights": {"derived_scene_generation_allowed": True}}))
    _patch_pipeline_side_effects(monkeypatch)
    monkeypatch.setattr(q, "load_current_website_task_context", _confirmed_website_task)
    monkeypatch.setattr(q, "load_website_scene_sponsorship", lambda **_: dict(sponsorship))
    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        return {"status": "blocked", "mode": "analysis_blocked", "privacy_status": "pending_website_review",
                "privacy_verified": False, "blockers": ["website_preparation_probe"]}

    monkeypatch.setattr(q, "run_clean_plate_stage", prepare)
    with pytest.raises(PipelineError, match="website_preparation_probe"):
        q.run_qualification_pipeline(descriptor_gcs_uri=descriptor_uri, config=SimpleNamespace(gcs_root=storage_root, runtime_preflight_enabled=False))
    assert calls[0]["policy"].enabled is enabled
    assert calls[0]["worldlabs_input"]["status"] == "not_requested"


def test_sponsored_browser_preparation_hands_off_to_atlas_and_never_calls_marble(tmp_path, monkeypatch):
    # Owner decision 2026-10-22: the capture's Atlas job is the website scene.
    storage_root = tmp_path / "gcs"
    descriptor_uri = _write_descriptor(storage_root, _descriptor(
        capture_source="unknown", capture_modality="video_only", requested_outputs=[],
        metadata={"capture_entry_source": "browser_self_capture", "capture_rights": {"derived_scene_generation_allowed": True}}))
    _patch_pipeline_side_effects(monkeypatch)
    monkeypatch.setattr(q, "load_current_website_task_context", _confirmed_website_task)
    monkeypatch.setattr(q, "load_website_scene_sponsorship",
                        lambda **_: {"sponsor": "blueprint", "purpose": "scene_preparation"})
    monkeypatch.setattr(q, "run_clean_plate_stage", lambda **_: {
        "status": "objects_removed", "mode": "task_masks_ready", "privacy_status": "website_capture_admitted",
        "privacy_verified": True, "blockers": []})
    handoffs = []

    def handoff(**kwargs):
        handoffs.append(kwargs)
        return {"status": "awaiting_reconstruction", "blockers": ["website_atlas_job_processing"]}

    monkeypatch.setattr("blueprint_pipeline.website_scene_handoff.prepare_website_scene_handoff", handoff)
    with pytest.raises(PipelineError, match="website_atlas_job_processing"):
        q.run_qualification_pipeline(descriptor_gcs_uri=descriptor_uri, config=SimpleNamespace(gcs_root=storage_root, runtime_preflight_enabled=False))
    assert len(handoffs) == 1 and "provider_run" not in handoffs[0]


def test_run_qualification_pipeline_suppresses_failure_writer_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(q, "_write_failure", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("disk full")))

    with pytest.raises(q.PipelineError):
        q.run_qualification_pipeline(
            descriptor_gcs_uri="not-a-gs-uri",
            config=SimpleNamespace(gcs_root=tmp_path / "gcs", runtime_preflight_enabled=False),
        )


@pytest.mark.parametrize("outbox", [{"state": "forward_pending", "id": "scene-1"}, {}])
def test_website_preparation_finishes_only_at_its_native_outbox(tmp_path, monkeypatch, outbox):
    import blueprint_pipeline.website_scene_handoff as handoff
    storage_root = tmp_path / "gcs"
    descriptor_uri = _write_descriptor(storage_root, _descriptor(
        capture_source="unknown", capture_modality="video_only", requested_outputs=["preview_simulation"],
        metadata={"capture_entry_source": "browser_self_capture", "capture_rights": {"derived_scene_generation_allowed": True}}))
    _patch_pipeline_side_effects(monkeypatch)
    monkeypatch.setattr(q, "load_current_website_task_context", _confirmed_website_task)
    monkeypatch.setattr(q, "load_website_scene_sponsorship", lambda **_: {"sponsor": "blueprint"})
    monkeypatch.setattr(q, "run_clean_plate_stage", lambda **_: {"status": "noop", "privacy_status": "no_people_detected", "privacy_verified": True})
    monkeypatch.setattr(handoff, "prepare_website_scene_handoff", lambda **_: {"status": "intake_ready", "website_intake_outbox": outbox})
    if outbox:
        result = q.run_qualification_pipeline(descriptor_gcs_uri=descriptor_uri, config=SimpleNamespace(gcs_root=storage_root, runtime_preflight_enabled=False))
        assert result["status"] == "completed"
        assert result["native_execution_complete"] is False
    else:
        with pytest.raises(PipelineError, match="website_native_intake_pending"):
            q.run_qualification_pipeline(descriptor_gcs_uri=descriptor_uri, config=SimpleNamespace(gcs_root=storage_root, runtime_preflight_enabled=False))


@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("mismatch", [False, True])
def test_website_native_profile_gap_finishes_preparation_without_native_execution(tmp_path, monkeypatch, stale, mismatch):
    import blueprint_pipeline.website_scene_handoff as handoff
    from blueprint_pipeline.website_task_preparation import native_task_preparation_gap

    storage_root = tmp_path / "gcs"
    descriptor_uri = _write_descriptor(storage_root, _descriptor(
        capture_source="unknown", capture_modality="video_only", requested_outputs=["preview_simulation"],
        metadata={"capture_entry_source": "browser_self_capture", "capture_rights": {"derived_scene_generation_allowed": True}}))
    _patch_pipeline_side_effects(monkeypatch)

    def current_task(**identity):
        # Folding has a task-family preparation path; this fixture exercises
        # the separate hold for a family with no implemented profile.
        context = {**_confirmed_website_task(**identity),
                   "task_family": "articulated_open_close" if mismatch else "unimplemented_task_family"}
        context["context_digest"] = canonical_digest(context, digest_field="context_digest")
        return context

    def held(**kwargs):
        context = kwargs["descriptor"]["metadata"]["site_task_context"]
        gap = native_task_preparation_gap(context, **({"inferred_profile_id": "rigid_pick_place"} if mismatch else {}))
        if stale:
            gap["task_context_digest"] = "sha256:" + "9" * 64
        value = {"schema_version": "website_scene_handoff.v1", "status": "needs_input",
                 "scene_id": context["scene_id"], "capture_id": context["capture_id"],
                 "blockers": [gap["code"]], "native_task_preparation_gap": gap,
                 "simulator_ready": False, "visual_reconstruction_ready": True,
                 **({} if mismatch else {"provider_mutation_performed": False})}
        value["digest"] = canonical_digest(value, digest_field="digest")
        return value

    monkeypatch.setattr(q, "load_current_website_task_context", current_task)
    monkeypatch.setattr(q, "load_website_scene_sponsorship", lambda **_: {"sponsor": "blueprint"})
    monkeypatch.setattr(q, "run_clean_plate_stage", lambda **_: {"status": "noop", "privacy_status": "no_people_detected", "privacy_verified": True})
    monkeypatch.setattr(handoff, "prepare_website_scene_handoff", held)
    if stale:
        with pytest.raises(PipelineError, match=("website_task_profile_mechanism_mismatch" if mismatch
                                                else "website_native_task_profile_preparation_required")):
            q.run_qualification_pipeline(descriptor_gcs_uri=descriptor_uri, config=SimpleNamespace(gcs_root=storage_root, runtime_preflight_enabled=False))
    else:
        result = q.run_qualification_pipeline(descriptor_gcs_uri=descriptor_uri, config=SimpleNamespace(gcs_root=storage_root, runtime_preflight_enabled=False))
        assert result["status"] == "completed" and result["native_execution_complete"] is False
        assert result["website_scene_preparation"]["status"] == "needs_input"
        assert "website_intake_outbox" not in result["website_scene_preparation"]


def test_shared_task_constraints_survive_a_stored_task_scope_record(tmp_path: Path) -> None:
    # The capture app's writer of task_scope_record.json was removed on 2026-11-11; a stored
    # record is still read by evaluation preparation, so its task constraints must carry over.
    from blueprint_pipeline.evaluation_prep_stage import _build_task_anchor_manifest

    metadata = {
        "assumptions": ["Dimensions estimated from video"],
        "missing_measurements": ["door hinge torque", "rack travel force"],
        "allowed_environmental_changes": {"dish_loading": False, "wash_cycle": False},
        "starting_conditions": {"door": "closed"},
        "success_criteria": {"door": "open to supplied target", "racks": "extend"},
    }
    scope = {
        "task_statement": "Manipulate dishwasher door and upper/lower racks",
        "tasks": [{"task_id": "door", "target_object_ids": ["door"]}],
        "success_criteria_authority": "supplied_unverified_task_criteria",
        **metadata,
    }
    anchors = _build_task_anchor_manifest(
        capture_root=tmp_path / "scenes" / "scene-1" / "captures" / "capture-1",
        handoff={}, scope_record=scope, task_run_entries=[], object_geometry_manifest={},
    )
    task = anchors["tasks"][0]
    for key in metadata:
        assert task[key] == metadata[key]
    assert task["success_criteria_authority"] == "supplied_unverified_task_criteria"


def test_a_descriptor_that_is_not_a_website_upload_ends_without_running_anything(tmp_path):
    storage_root = tmp_path / "gcs"
    descriptor_uri = _write_descriptor(storage_root, _descriptor(metadata={}))
    with pytest.raises(PipelineError, match="device_capture_pipeline_removed") as failure:
        q.run_qualification_pipeline(descriptor_gcs_uri=descriptor_uri,
                                     config=SimpleNamespace(gcs_root=storage_root))
    assert isinstance(failure.value.__cause__, StageError)
    failure_record = storage_root / "scenes" / "scene-1" / "captures" / "capture-1" / "pipeline"
    assert not (failure_record / "qualification_summary.json").exists()
