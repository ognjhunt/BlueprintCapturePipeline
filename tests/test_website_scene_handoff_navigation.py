"""A walking job gets the first look on paper and no simulation.

The robot moves itself, so no simulation is built for this kind of job: the scene
handoff holds it with the plain record a job whose family has no native task
profile gets (``website_native_task_profile_preparation_required``), which the
site already understands. It writes no navigation record, registers nothing,
sends the website nothing and buys nothing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_candidate_registry_reader import REGISTRY_ENV
from blueprint_pipeline.website_preparation_revision import workspace
from blueprint_pipeline.website_task_context_contracts import holds_native_task_preparation
from blueprint_pipeline.website_task_preparation import profile_preparation_gap
from tests.test_website_capture_withdrawal_team_scene import _registry

NOW = 1_800_000_000.0
ATLAS = "blueprint_pipeline.website_atlas_scene_source.atlas_scene_source"
RIGHTS = {"consent_status": "granted", "consent_revoked": False, "consent_scope": ["robot_evaluation"],
          "derived_scene_generation_allowed": True}
OWNER = {"user_id": "blueprint", "organization_id": "blueprint"}


def _context() -> dict:
    context = {"schema_version": "website_site_task_context.v1", "request_id": "req1", "scene_id": "site-req1",
               "capture_id": "walkthrough-req1", "task_family": "navigation", "confirmed": True,
               "confirmed_at": "2026-11-06T18:30:00Z",
               "description": "Walk down the lane to its far end without touching either side.",
               "capture_rights": dict(RIGHTS)}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    return context


def _authority(context: dict) -> dict:
    return {"schema_version": "website_scene_sponsorship.v1", "sponsor": "blueprint", "owner": dict(OWNER),
            "request_id": "req1", "scene_id": "site-req1", "capture_id": "walkthrough-req1",
            "task_context_digest": context["context_digest"], "max_total_spend_usd": 20, "max_paid_attempts": 2,
            "expires_at_epoch": NOW + 3600, "authoring_provider": "openai",
            "consent": {"rights_reference": canonical_digest(context["capture_rights"]),
                        "provider_terms_reference": "sha256:" + "c" * 64, "accepted_by": OWNER["user_id"],
                        "accepted_at_epoch": NOW - 60, "private_processing_authorized": True,
                        "provider_training_authorized": False, "task_confirmed": True,
                        "spend_authorized": True}}


@pytest.fixture
def job(tmp_path, monkeypatch):
    """One walking job's capture, its consent records, a ready Atlas scene and the registry."""
    capture = tmp_path / "pubsub/source-bucket/scenes/site-req1/captures/walkthrough-req1"
    write_json(capture / "raw/manifest.json", {"site_submission_id": "req1", "scene_id": "site-req1",
                                               "capture_id": "walkthrough-req1", "consent_status": "granted"})
    removal = capture / "pipeline/removal.json"
    write_json(removal, {"entries": []})
    base = {"collision_mesh_path": str(tmp_path / "scene.glb"), "collision_mesh_digest": "sha256:" + "1" * 64}
    monkeypatch.setattr(ATLAS, lambda **_: {
        "status": "ready", "base_scene": base, "alignment": "anchored", "camera_path_levelled_m": [],
        "source_geometry": {"digest": "atlas-geometry", "geometry_backend": "world_labs_atlas"}})
    registry, derived = tmp_path / "etc/team-evaluation-registry.json", tmp_path / "inputs/team-scene-profiles"
    _registry(registry, derived)
    monkeypatch.setenv(REGISTRY_ENV, str(registry))
    sent = []
    monkeypatch.setattr("blueprint_pipeline.website_task_context.website_webapp_request",
                        lambda **kwargs: sent.append(kwargs) or {})
    return {"capture": capture, "removal": removal, "derived": derived, "sent": sent, "root": workspace(capture)}


def _run(job):
    context = _context()
    descriptor = {"capture_id": "walkthrough-req1", "scene_id": "site-req1", "metadata": {
        "site_task_context": context, "website_scene_execution_authority": _authority(context)}}
    clean_plate = {"privacy_verified": True, "status": "noop", "source_geometry": None, "task_masks": {},
                   "removal_manifest_path": str(job["removal"])}
    return handoff.prepare_website_scene_handoff(descriptor=descriptor, clean_plate=clean_plate,
                                                 capture_root=job["capture"], now=NOW), context


def test_a_walking_job_holds_with_the_plain_no_simulation_reason(job):
    result, context = _run(job)
    gap = result["native_task_preparation_gap"]
    assert gap == profile_preparation_gap(context)
    assert gap["code"] == "website_native_task_profile_preparation_required"
    assert gap["job_disposition"] == "held_for_preparation"
    assert result["provider_mutation_performed"] is False
    # The site's existing reading of the hold accepts it as this job's own.
    assert holds_native_task_preparation(result, context)


def test_a_walking_job_writes_no_navigation_record_registers_nothing_and_sends_nothing(job):
    result, _context_ = _run(job)
    assert not any(key.startswith("navigation_") for key in result)
    assert not list(Path(job["root"]).rglob("*navigation*"))
    assert not (job["derived"].is_dir() and list(job["derived"].glob("*.json")))
    assert job["sent"] == []


def test_a_second_handoff_holds_the_same_way(job):
    first, _ = _run(job)
    second, _ = _run(job)
    assert second["native_task_preparation_gap"] == first["native_task_preparation_gap"]
