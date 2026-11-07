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
    PARENT_FIELD,
    accepts_bound_digest,
    artifact_context,
    binding_contexts,
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
    # A record decided before the confirmation is matched against the context, then its verified parent.
    assert binding_contexts(pipeline, current) == (current, parent)
    assert binding_contexts(pipeline, parent) == (parent,) and binding_contexts(None, current) == (current,)
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
    assert binding_contexts(pipeline, current) == (current,)


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


# The website's own projections (WebApp ``projectWebsiteTaskContext``; the same brief in
# server/tests/website-task-context-confirmation.test.ts), exactly as it serves them: a navigation brief
# prepared while unconfirmed (no answers, four open gates, general_task), then confirmed by accepting
# Blueprint's description-based proposal ``taskShape: single`` with no gate unknown and success criteria
# that state nothing (the library job's shape); and the same confirmation correcting that answer.
WEBSITE_SEALED = json.loads("{\"schema_version\":\"website_site_task_context.v1\",\"purpose\":\"scene_preparation\",\"request_id\":\"synthetic-aisle\",\"scene_id\":\"site-synthetic-aisle\",\"capture_id\":\"walkthrough-synthetic-aisle\",\"description\":\"Walk down the aisle to its far end without touching the shelves.\",\"task_family\":\"general_task\",\"confirmed\":false,\"confirmed_at\":null,\"operator_answers\":{},\"unresolved\":[\"sceneStability\",\"objectVariety\",\"deploymentTimeline\",\"accessWindow\"],\"capture_rights\":{\"derived_scene_generation_allowed\":true,\"data_licensing_allowed\":false,\"capture_contributor_payout_eligible\":false,\"consent_status\":\"granted\",\"consent_revoked\":false,\"consent_scope\":[\"derived_scene_generation\",\"robot_evaluation\"],\"statement_version\":\"2026-11-02.v2\",\"recorded_at_iso\":\"2026-11-06T10:00:00.000Z\"},\"context_digest\":\"sha256:bcfc3febeb9de94846c5500aae00aa129bb6dffe74a91fbfc4f485dd2f917c88\"}")
WEBSITE_SUCCESSOR = json.loads("{\"schema_version\":\"website_site_task_context.v1\",\"purpose\":\"scene_preparation\",\"request_id\":\"synthetic-aisle\",\"scene_id\":\"site-synthetic-aisle\",\"capture_id\":\"walkthrough-synthetic-aisle\",\"description\":\"Walk down the aisle to its far end without touching the shelves.\",\"task_family\":\"general_task\",\"confirmed\":true,\"confirmed_at\":\"2026-11-07T14:30:45.016Z\",\"operator_answers\":{\"taskShape\":\"single\"},\"unresolved\":[],\"capture_rights\":{\"derived_scene_generation_allowed\":true,\"data_licensing_allowed\":false,\"capture_contributor_payout_eligible\":false,\"consent_status\":\"granted\",\"consent_revoked\":false,\"consent_scope\":[\"derived_scene_generation\",\"robot_evaluation\"],\"statement_version\":\"2026-11-02.v2\",\"recorded_at_iso\":\"2026-11-06T10:00:00.000Z\"},\"success_criteria\":{\"successDefinition\":null,\"successRate\":null,\"cycleTimeSeconds\":null,\"unknown\":true},\"task_evidence_provenance\":{\"item_details\":\"owner_stated_unverified\",\"success_criteria\":\"owner_stated_target\",\"success_rate_unit\":\"percent\",\"cycle_time_unit\":\"seconds\",\"physical_measurements_verified\":false},\"confirmation_parent_context_digest\":\"sha256:bcfc3febeb9de94846c5500aae00aa129bb6dffe74a91fbfc4f485dd2f917c88\",\"context_digest\":\"sha256:0deac30a4fe0a0c3b8c6da19224891249f2dc7ccc95b47427283ba977131b06c\"}")
WEBSITE_CORRECTED = json.loads("{\"schema_version\":\"website_site_task_context.v1\",\"purpose\":\"scene_preparation\",\"request_id\":\"synthetic-aisle\",\"scene_id\":\"site-synthetic-aisle\",\"capture_id\":\"walkthrough-synthetic-aisle\",\"description\":\"Walk down the aisle to its far end without touching the shelves.\",\"task_family\":\"general_task\",\"confirmed\":true,\"confirmed_at\":\"2026-11-07T14:30:45.016Z\",\"operator_answers\":{\"taskShape\":\"multiple\"},\"unresolved\":[],\"capture_rights\":{\"derived_scene_generation_allowed\":true,\"data_licensing_allowed\":false,\"capture_contributor_payout_eligible\":false,\"consent_status\":\"granted\",\"consent_revoked\":false,\"consent_scope\":[\"derived_scene_generation\",\"robot_evaluation\"],\"statement_version\":\"2026-11-02.v2\",\"recorded_at_iso\":\"2026-11-06T10:00:00.000Z\"},\"success_criteria\":{\"successDefinition\":null,\"successRate\":null,\"cycleTimeSeconds\":null,\"unknown\":true},\"task_evidence_provenance\":{\"item_details\":\"owner_stated_unverified\",\"success_criteria\":\"owner_stated_target\",\"success_rate_unit\":\"percent\",\"cycle_time_unit\":\"seconds\",\"physical_measurements_verified\":false},\"context_digest\":\"sha256:c364877e9324ea0301f20e6307bab7f8b618b5f25c230a07025d12813a949778\"}")


def test_the_websites_digests_are_these_bytes_and_the_successor_names_the_sealed_one():
    for context in (WEBSITE_SEALED, WEBSITE_SUCCESSOR, WEBSITE_CORRECTED):
        assert canonical_digest(context, digest_field="context_digest") == context["context_digest"]
    assert WEBSITE_SUCCESSOR[PARENT_FIELD] == WEBSITE_SEALED["context_digest"]
    assert PARENT_FIELD not in WEBSITE_CORRECTED


def test_a_confirmation_that_accepts_the_proposal_is_the_sealed_ones_successor(tmp_path):
    """The library shape: answers, open gates and empty success criteria changed; the task did not."""
    from blueprint_pipeline.website_assessment_resume import _context_upgrade_allowed
    from blueprint_pipeline.website_task_context_contracts import validate_website_task_context

    sealed, current = deepcopy(WEBSITE_SEALED), deepcopy(WEBSITE_SUCCESSOR)
    assert {key for key in set(sealed) | set(current) if sealed.get(key) != current.get(key)} == {
        "confirmed", "confirmed_at", "operator_answers", "unresolved", "success_criteria",
        "task_evidence_provenance", PARENT_FIELD, "context_digest"}
    assert confirmation_successor(sealed, current)["confirmation_parent_context_digest"] == sealed["context_digest"]
    # Browser admission (where the library job's resume was refused) now lets the successor replace it.
    assert _context_upgrade_allowed(sealed, current)
    assert validate_website_task_context(current, request_id=current["request_id"], scene_id=current["scene_id"],
                                         capture_id=current["capture_id"], purpose="scene_preparation") == current
    assert accepts_bound_digest(current, sealed["context_digest"])
    pipeline = tmp_path / "scenes/site-synthetic-aisle/captures/walkthrough-synthetic-aisle/pipeline"
    assert record_confirmation_lineage(pipeline, parent=sealed, successor=current) is not None
    assert artifact_context(pipeline, current) == sealed


def _restated(change: str) -> dict:
    current = deepcopy(WEBSITE_SUCCESSOR)
    if change == "description":
        current["description"] += " Twice."
    elif change == "family":
        current["task_family"] = "navigation"
    elif change == "success_definition":
        current["success_criteria"] = {**current["success_criteria"], "successDefinition": "Reaches the end", "unknown": False}
    elif change == "success_rate":
        current["success_criteria"] = {**current["success_criteria"], "successRate": 90, "unknown": False}
    elif change == "item_details":
        current["operator_task_details"] = {"item_weight": "2 kg"}
    elif change == "rights":
        current["capture_rights"] = {**current["capture_rights"], "consent_scope": ["robot_evaluation"]}
    elif change == "identity":
        current["capture_id"] = "walkthrough-other"
    elif change == "another_parent":
        current[PARENT_FIELD] = "sha256:" + "e" * 64
    return _sealed(current)


@pytest.mark.parametrize("change", ["correction", "description", "family", "success_definition", "success_rate",
                                    "item_details", "rights", "identity", "another_parent"])
def test_a_correction_or_any_task_change_is_not_a_successor_even_when_a_parent_is_named(change):
    from blueprint_pipeline.website_assessment_resume import _context_upgrade_allowed

    current = deepcopy(WEBSITE_CORRECTED) if change == "correction" else _restated(change)
    assert confirmation_successor(WEBSITE_SEALED, current) is None
    assert not _context_upgrade_allowed(WEBSITE_SEALED, current)
    if change in {"correction", "another_parent"}:
        assert not accepts_bound_digest(current, WEBSITE_SEALED["context_digest"])


@pytest.mark.parametrize("value", [{"confirmed": False, "confirmed_at": None}, {PARENT_FIELD: "sha256:not-a-digest"}])
def test_only_a_confirmed_context_names_a_parent(value):
    from blueprint_pipeline.website_task_context_contracts import validate_website_task_context

    current = _sealed({**WEBSITE_SUCCESSOR, **value})
    with pytest.raises(ValueError, match="website_task_context_confirmation_parent_invalid"):
        validate_website_task_context(current, request_id=current["request_id"], scene_id=current["scene_id"],
                                      capture_id=current["capture_id"], purpose="scene_preparation")


@pytest.mark.parametrize("answered", [True, False])
def test_semantics_bought_for_the_parent_load_for_a_successor_that_accepted_the_proposal(tmp_path, answered):
    parent, source, receipt = semantics_fixture()
    artifact = semantics.retain_task_semantics(job_dir=tmp_path, receipt=receipt, task_context=parent,
                                               source_video_digest=source)
    (tmp_path / "selection_binding.json").write_text(json.dumps({"planner_receipt": receipt}))
    answers = {"operator_answers": {"taskShape": "single"}, "unresolved": []}
    current = successor_of(parent, **answers, **({PARENT_FIELD: parent["context_digest"]} if answered else {}))
    if answered:
        assert semantics.load_task_semantics(job_dir=tmp_path, task_context=current, source_video_digest=source) == artifact
    else:  # the same answers with no parent named by the website: a corrected brief binds nothing of the parent
        with pytest.raises(ValueError, match="website_task_semantics_planner_binding_invalid"):
            semantics.load_task_semantics(job_dir=tmp_path, task_context=current, source_video_digest=source)


@pytest.mark.parametrize("current_kind", ["answered", "corrected"])
def test_screening_reads_a_parent_bound_source_against_a_successor_that_accepted_the_proposal(tmp_path, monkeypatch,
                                                                                               current_kind):
    from blueprint_pipeline import beta_policy_screening_worker as worker
    from blueprint_pipeline import team_scene_packet_worker as packets
    from blueprint_pipeline import website_task_context as reader

    parent = WEBSITE_SEALED
    current = WEBSITE_SUCCESSOR if current_kind == "answered" else WEBSITE_CORRECTED
    site = parent["scene_id"]
    capture = tmp_path / "capture"
    (capture / "pipeline").mkdir(parents=True)
    source = {"scene_intent_id": "scene-" + "a" * 64, "scene_intent_digest": "sha256:" + "a" * 64,
              "capture_id": parent["capture_id"], "task_id": "website-" + parent["context_digest"][7:27],
              "task_context_digest": parent["context_digest"], "scene_revision_digest": "sha256:" + "b" * 64}
    monkeypatch.setattr(packets, "resolve_capture_root", lambda _source: capture)
    monkeypatch.setattr(reader, "load_current_website_task_context", lambda **_: deepcopy(current))
    (capture / "pipeline/website_task_context.json").write_text(json.dumps(current))
    if current_kind == "answered":
        assert worker.require_current_source(source, site_id=site) == current
    else:
        with pytest.raises(ValueError, match="beta_screening_source_context_mismatch"):
            worker.require_current_source(source, site_id=site)
