"""A brief confirmed after preparation is its sealed context's confirmed successor, and its readers bind on the parent.

Decision (lead, 2026-11-07, on the owner's first-principles direction): confirming a brief is consent
state, not task content. A context that differs from the sealed one only in ``confirmed`` false to true
(with the website's ``confirmed_at``) keeps everything bound to the sealed digest valid, and nothing paid
is bought again. Any other difference keeps today's behaviour. Synthetic contexts only; no provider,
website, money or host.
"""
# Covers: src/blueprint_pipeline/website_task_context_lineage.py
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from blueprint_pipeline import website_task_semantics as semantics
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_task_context_lineage import (
    accepts_bound_digest,
    artifact_context,
    confirmation_successor,
    read_confirmation_parent,
    record_confirmation_lineage,
)
from tests.test_website_task_semantics import semantics_fixture

CONFIRMED_AT = "2026-11-07T14:30:45Z"


def _sealed(value: dict) -> dict:
    value = {key: item for key, item in value.items() if key != "context_digest"}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


def successor_of(parent: dict, **change) -> dict:
    """The website's own confirmed reading of the same brief (the site confirmed it)."""
    return _sealed({**parent, "confirmed": True, "confirmed_at": CONFIRMED_AT, **change})


def test_a_context_that_differs_only_in_its_confirmation_is_the_sealed_ones_successor():
    parent, _, _ = semantics_fixture()
    current = successor_of(parent)
    assert confirmation_successor(parent, current) == {
        "schema_version": "website_task_context_confirmation_lineage.v1",
        "confirmation_parent_context_digest": parent["context_digest"],
        "context_digest": current["context_digest"], "confirmed_at": CONFIRMED_AT}
    assert accepts_bound_digest(current, parent["context_digest"])
    assert accepts_bound_digest(current, current["context_digest"])
    # An older projection without confirmed_at in the unconfirmed context is the same parent.
    legacy = _sealed({key: value for key, value in parent.items() if key != "confirmed_at"})
    assert accepts_bound_digest(current, legacy["context_digest"])


@pytest.mark.parametrize("change", ["edited", "family_dropped", "unconfirmed", "no_time", "reverse", "tampered"])
def test_anything_else_is_not_a_successor_and_binds_nothing(change):
    parent, _, _ = semantics_fixture()
    current = successor_of(parent)
    if change == "edited":
        current = successor_of(parent, description=parent["description"] + " Twice.")
    elif change == "family_dropped":
        current = _sealed({key: value for key, value in current.items() if key != "task_family"})
    elif change == "unconfirmed":
        current = _sealed({**parent, "confirmed_at": CONFIRMED_AT})
    elif change == "no_time":
        current = _sealed({**current, "confirmed_at": None})
    elif change == "reverse":
        parent, current = current, parent
    else:
        current = {**current, "confirmed_at": "2026-11-08T00:00:00Z"}  # seal no longer matches
    assert confirmation_successor(parent, current) is None
    assert not accepts_bound_digest(current, parent["context_digest"])


def test_the_lineage_is_kept_beside_the_capture_and_re_verified_on_every_read(tmp_path):
    parent, _, _ = semantics_fixture()
    current = successor_of(parent)
    pipeline = tmp_path / "scenes/site-synthetic-job/captures/walkthrough-synthetic-job/pipeline"
    record = record_confirmation_lineage(pipeline, parent=parent, successor=current)
    assert record["confirmation_parent_context_digest"] == parent["context_digest"]
    assert record["confirmed_at"] == CONFIRMED_AT and record["parent_context"] == parent
    assert read_confirmation_parent(pipeline, current) == parent
    assert artifact_context(pipeline, current) == parent
    # An edited task records nothing and reads as itself.
    edited = successor_of(parent, description="Another job")
    assert record_confirmation_lineage(pipeline, parent=parent, successor=edited) is None
    assert artifact_context(pipeline, edited) == edited
    # A rewritten record no longer verifies, so nothing binds through it.
    path = next((pipeline / "website_task_context_lineage").glob("*.json"))
    value = json.loads(path.read_text())
    value["parent_context"]["description"] = "Another job"
    path.write_text(json.dumps(value))
    assert read_confirmation_parent(pipeline, current) is None and artifact_context(pipeline, current) == current


def test_task_semantics_bought_for_the_parent_load_for_its_confirmed_successor(tmp_path):
    parent, source, receipt = semantics_fixture()
    artifact = semantics.retain_task_semantics(job_dir=tmp_path, receipt=receipt, task_context=parent,
                                               source_video_digest=source)
    (tmp_path / "selection_binding.json").write_text(json.dumps({"planner_receipt": receipt}))
    current = successor_of(parent)
    loaded = semantics.load_task_semantics(job_dir=tmp_path, task_context=current, source_video_digest=source)
    assert loaded == artifact and loaded["binding"]["task_context_digest"] == parent["context_digest"]
    assert semantics.effective_task_profile(task_context=current, task_semantics=loaded,
                                            source_video_digest=source) == "rigid_pick_place"
    # An edited task cannot read the parent's semantics.
    with pytest.raises(ValueError, match="website_task_semantics_planner_binding_invalid"):
        semantics.load_task_semantics(job_dir=tmp_path, task_context=successor_of(parent, description="Other"),
                                      source_video_digest=source)


def _ranking(pipeline: Path, context: dict) -> dict:
    from blueprint_pipeline.website_candidate_ranking import RANKING_FILENAME, SCHEMA_VERSION

    value = {"schema_version": SCHEMA_VERSION, "task_context_digest": context["context_digest"],
             "claim_ceiling": "development_only", "basis": "advisory_order_never_exclusion", "status": "ranked",
             "candidates": [], "priority_order": ["candidate-b", "candidate-a"], "preparation_requested": []}
    value["ranking_digest"] = canonical_digest(value, digest_field="ranking_digest")
    pipeline.mkdir(parents=True, exist_ok=True)
    (pipeline / RANKING_FILENAME).write_text(json.dumps(value))
    return value


def test_the_parents_candidate_ranking_is_the_successors_and_no_model_is_called(tmp_path, monkeypatch):
    from blueprint_pipeline import website_candidate_ranking as ranking

    parent, _, _ = semantics_fixture()
    current = successor_of(parent)
    pipeline = tmp_path / "pipeline"
    retained = _ranking(pipeline, parent)
    monkeypatch.setattr(ranking, "_gemini", lambda prompt: pytest.fail("a confirmed successor is never ranked again"))
    assert ranking.rank_website_candidates(task_context=current, task_masks={"targets": []}, pipeline_dir=pipeline,
                                           capture_root=tmp_path) == retained
    assert ranking.load_ranking(pipeline, task_context_digest=current["context_digest"]) is None  # needs the context
    (pipeline / "website_task_context.json").write_text(json.dumps(current))
    assert ranking.retained_priority(pipeline, task_context_digest=current["context_digest"]) == [
        "candidate-b", "candidate-a"]
    edited = successor_of(parent, description="Other")
    assert ranking.load_ranking(pipeline, task_context_digest=edited["context_digest"], task_context=edited) is None


@pytest.mark.parametrize("current_kind", ["successor", "edited"])
def test_screening_reads_a_source_bound_to_the_parent_against_the_confirmed_successor(tmp_path, monkeypatch,
                                                                                      current_kind):
    """The intake receipt names the record's sealed context; the website now reads the confirmed successor."""
    from blueprint_pipeline import beta_policy_screening_worker as worker
    from blueprint_pipeline import team_scene_packet_worker as packets
    from blueprint_pipeline import website_task_context as reader

    rights = {"consent_status": "granted", "consent_revoked": False, "consent_scope": ["robot_evaluation"],
              "derived_scene_generation_allowed": True}
    parent = _sealed({"schema_version": "website_site_task_context.v1", "request_id": "req1",
                      "scene_id": "site-req1", "capture_id": "walkthrough-req1", "task_family": "navigation",
                      "description": "Walk to the end of the aisle.", "confirmed": False, "confirmed_at": None,
                      "purpose": "scene_preparation", "capture_rights": rights})
    current = successor_of(parent) if current_kind == "successor" else successor_of(parent, description="Other")
    capture = tmp_path / "capture"
    (capture / "pipeline").mkdir(parents=True)
    source = {"scene_intent_id": "scene-" + "a" * 64, "scene_intent_digest": "sha256:" + "a" * 64,
              "capture_id": "walkthrough-req1", "task_id": "website-" + parent["context_digest"][7:27],
              "task_context_digest": parent["context_digest"], "scene_revision_digest": "sha256:" + "b" * 64}
    monkeypatch.setattr(packets, "resolve_capture_root", lambda _source: capture)
    monkeypatch.setattr(reader, "load_current_website_task_context", lambda **_: deepcopy(current))
    for retained in (parent, current):  # before and after the replay rewrote the retained context
        (capture / "pipeline/website_task_context.json").write_text(json.dumps(retained))
        if current_kind == "successor":
            assert worker.require_current_source(source, site_id="site-req1") == current
        else:
            with pytest.raises(ValueError, match="beta_screening_(current_task_context_changed|source_context_mismatch)"):
                worker.require_current_source(source, site_id="site-req1")


def test_the_entry_keeps_the_lineage_before_it_replaces_the_sealed_context(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_evidence as module
    from blueprint_pipeline.capture_bridge import CaptureDescriptor

    parent, _, _ = semantics_fixture()
    current = successor_of(parent)
    pipeline = tmp_path / "scenes/site-synthetic-job/captures/walkthrough-synthetic-job/pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "website_task_context.json").write_text(json.dumps(parent))
    monkeypatch.setattr(module, "ingest_website_task_evidence", lambda **_: {"items": []})
    monkeypatch.setattr(module, "publish_website_item_evidence", lambda **_: None)
    descriptor = CaptureDescriptor(schema_version="v1", scene_id=parent["scene_id"], capture_id=parent["capture_id"],
        capture_source="unknown", capture_tier="candidate", raw_prefix_uri="gs://bucket/raw",
        frames_index_uri="gs://bucket/raw/frames.json", site_submission_id=parent["request_id"], metadata={})
    result = module.prepare_website_task_descriptor(descriptor=descriptor, capture_root=pipeline.parent,
        pipeline_dir=pipeline, bucket="bucket", load_task_context=lambda **_: current,
        load_sponsorship=lambda **_: {"sponsor": "synthetic"})
    assert result.metadata["site_task_context"] == current
    assert json.loads((pipeline / "website_task_context.json").read_text()) == current
    assert read_confirmation_parent(pipeline, current) == parent
