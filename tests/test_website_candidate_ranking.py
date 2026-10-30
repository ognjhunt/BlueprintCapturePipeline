"""Intake ranking orders every compatible candidate in one retained, admitted call; no live model calls."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from blueprint_pipeline import website_candidate_ranking as ranking
from blueprint_pipeline import website_gemini_receipts as receipts
from blueprint_pipeline.beta_policy_candidate_catalog import catalog_candidate, load_default_catalog, seal_catalog
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION, PaidResourceAdmissionBlocked, require_paid_resource_admission,
)


def context(**fields):
    value = {"schema_version": "website_site_task_context.v1", "request_id": "job1", "scene_id": "site-job1",
             "capture_id": "walkthrough-job1", "confirmed": True, "confirmed_at": "2026-10-30T00:00:00Z",
             "description": "Move the item from its start support to the destination.",
             "task_family": "rigid_pick_place", "operator_task_details": {"item_weight": "about 2 kg"}, **fields}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


MASKS = {"targets": [
    {"target_id": "subject", "target_role": "task_object", "semantic_label": "item", "task_effect": "manipulated",
     "placement_relation": "on_top_of", "articulated_part": "", "articulation_kind": "", "disposition": "remove",
     "estimated_visible_bounds": {"minimum": [0.0, 0.0, 0.8], "maximum": [0.2, 0.1, 1.1],
                                  "basis": "visible_masked_surfaces_with_model_estimated_depth_and_cameras"}},
    {"target_id": "support", "task_effect": "static_contact", "semantic_label": "support", "disposition": "keep",
     "estimated_visible_bounds": {"minimum": [-0.5, -0.4, 0.0], "maximum": [0.6, 0.4, 0.8], "basis": "estimate"}},
    {"target_id": "person-1", "target_class": "person", "disposition": "remove", "semantic_label": "person"},
]}


def catalog():
    """The real catalog plus one row that declares only another task kind."""
    base = load_default_catalog()
    other = {**deepcopy(catalog_candidate(base, "pi05_droid")), "candidate_id": "other_kind_only",
             "supported_task_kinds": ["articulated_open_close"]}
    return seal_catalog({**base, "candidates": [*base["candidates"], other]})


def admitted(calls, refuse=False):
    def reserve(**kwargs):
        calls.append(("reserve", kwargs))
        if refuse:
            raise PaidResourceAdmissionBlocked(["website_preparation_allowance_exhausted"])
        admission = {"schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION, "resource_class": kwargs["resource_class"],
                     "status": "admitted", "allocation_binding_digest": kwargs["binding_digest"], "blockers": []}
        return admission, require_paid_resource_admission(admission, resource_class=kwargs["resource_class"],
                                                          expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    return reserve


def model(calls, answer):
    def call(prompt):
        calls.append(("prompt", prompt))

        def invoke():
            calls.append(("invoke", None))
            if isinstance(answer, Exception):
                raise answer
            return {"finish_reason": "STOP", "text": json.dumps(answer(prompt))}
        return (lambda: None), invoke
    return call


def partial_answer(prompt):
    """A model that ranks three, repeats one, names an unknown id and omits the rest."""
    return {"rankings": [
        {"candidate_id": "groot_n17_droid", "priority_score": 90, "rationale": "Enabled; interface matches."},
        {"candidate_id": "jacobi_fanuc_m710ic_45m", "priority_score": 90, "rationale": "Large payload body."},
        {"candidate_id": "pi05_droid", "priority_score": 40, "rationale": "Enabled single arm."},
        {"candidate_id": "pi05_droid", "priority_score": 99, "rationale": "Duplicate."},
        {"candidate_id": "invented_robot", "priority_score": 100, "rationale": "Not in the catalog."},
        {"candidate_id": "molmoact2_droid", "priority_score": 101, "rationale": "Out of range."}]}


def rank(tmp_path, calls, answer=partial_answer, **kwargs):
    return ranking.rank_website_candidates(task_context=kwargs.pop("task_context", context()), task_masks=MASKS,
        pipeline_dir=tmp_path / "pipeline", capture_root=tmp_path, catalog=kwargs.pop("catalog", catalog()),
        model_call=model(calls, answer), **kwargs)


def test_every_compatible_candidate_is_ranked_or_kept_and_none_is_dropped(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(receipts, "reserve_website_preparation_spend", admitted(calls))
    value = rank(tmp_path, calls)
    every = {row["candidate_id"] for row in catalog()["candidates"]}
    assert value["status"] == "ranked" and set(value["priority_order"]) == every
    assert len(value["priority_order"]) == len(every) == len(value["candidates"])
    # Ranked first by score, ties by identifier; the model's omissions keep a place after them.
    assert value["priority_order"][:3] == ["groot_n17_droid", "jacobi_fanuc_m710ic_45m", "pi05_droid"]
    assert value["candidates"][2]["rationale"] == "Enabled single arm."
    omitted = [row for row in value["candidates"] if row["basis"] == "model_omitted"]
    assert [row["candidate_id"] for row in omitted] == sorted(row["candidate_id"] for row in omitted)
    assert "molmoact2_droid" in {row["candidate_id"] for row in omitted}
    # A row naming only another task kind is listed last with its reason, never silently removed.
    assert value["candidates"][-1] == {**value["candidates"][-1], "candidate_id": "other_kind_only",
        "basis": "not_compatible", "compatibility_blocker": "task_kind_not_supported_by_candidate"}
    assert value["flags"] == ["model_entry_invalid", "model_named_unknown_candidate", "model_omitted_candidate",
                              "model_ranked_candidate_twice"]
    assert [row["priority_rank"] for row in value["candidates"]] == list(range(1, len(every) + 1))
    jacobi = next(row for row in value["candidates"] if row["candidate_id"] == "jacobi_fanuc_m710ic_45m")
    assert jacobi["deployment_companies"] == ["Jacobi Robotics", "Delta Technology"]
    # The model saw the task facts, every compatible row and the cited business brief; never a person.
    prompt = next(text for kind, text in calls if kind == "prompt")
    assert all(f'"candidate_id": "{key}"' in prompt for key in every - {"other_kind_only"})
    assert '"other_kind_only"' not in prompt and "person-1" not in prompt
    assert "What Blueprint Does" in prompt and '"estimated_visible_extent_m": [0.2, 0.1, 0.3]' in prompt
    assert value["facts"]["estimated_task_region_extent_m"] == [1.1, 0.8, 1.1]
    assert value["ranking_digest"] == canonical_digest(value, digest_field="ranking_digest")


def test_ranking_call_is_admitted_once_and_a_restart_never_buys_it_again(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(receipts, "reserve_website_preparation_spend", admitted(calls))
    first = rank(tmp_path, calls)
    [reservation] = [kwargs for kind, kwargs in calls if kind == "reserve"]
    assert reservation["provider"] == "google" and reservation["resource_class"] == "evaluator_api"
    assert reservation["request_count"] == 1 and 0 < reservation["maximum_cost_usd"] < 1
    receipt = json.loads((tmp_path / "pipeline" / "website_candidate_ranking" / "receipts"
                          / (first["request_digest"][7:] + ".json")).read_text())
    assert receipt["status"] == "completed" and receipt["admission"]["status"] == "admitted"
    assert receipt["request"]["binding"]["business_context"]["source_file"] == "PLATFORM_CONTEXT.md"
    assert first["receipt_path"].endswith(first["request_digest"][7:] + ".json")
    count = len(calls)
    # Restart: the retained ranking is reused before any model or admission step.
    assert rank(tmp_path, calls) == first and len(calls) == count
    # Lost ranking file: the retained receipt answers without another reservation or call.
    (tmp_path / "pipeline" / ranking.RANKING_FILENAME).unlink()
    again = rank(tmp_path, calls)
    assert again["priority_order"] == first["priority_order"]
    assert [kind for kind, _ in calls[count:]] == ["prompt"]
    assert ranking.retained_priority(tmp_path / "pipeline", task_context_digest=context()["context_digest"]) \
        == first["priority_order"]


def test_uncertain_call_is_flagged_and_never_bought_again(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(receipts, "reserve_website_preparation_spend", admitted(calls))
    value = rank(tmp_path, calls, answer=TimeoutError("provider may have charged"))
    assert value["status"] == "not_ranked" and value["priority_order"] == []
    again = rank(tmp_path, calls, answer=partial_answer)
    assert again["status"] == "not_ranked" and again["reason"] == "website_gemini_requires_reconciliation"
    assert [kind for kind, _ in calls].count("reserve") == 1
    assert [kind for kind, _ in calls].count("invoke") == 1
    assert ranking.retained_priority(tmp_path / "pipeline", task_context_digest=context()["context_digest"]) == []


def test_refused_admission_is_recorded_and_the_job_keeps_its_default_order(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(receipts, "reserve_website_preparation_spend", admitted(calls, refuse=True))
    value = rank(tmp_path, calls)
    assert value["status"] == "not_ranked" and value["reason"] == "paid_resource_admission_blocked"
    assert value["reason_codes"] == ["website_preparation_allowance_exhausted"]
    assert "invoke" not in [kind for kind, _ in calls]
    stored = json.loads((tmp_path / "pipeline" / ranking.RANKING_FILENAME).read_text())
    assert stored == value and value["preparation"]["status"] == "not_started"


def test_preparation_is_requested_in_priority_order_and_the_missing_hook_is_named(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(receipts, "reserve_website_preparation_spend", admitted(calls))
    value = rank(tmp_path, calls)
    requested = [row["candidate_id"] for row in value["preparation_requested"]]
    disabled = [row["candidate_id"] for row in value["candidates"]
                if not row["enabled"] and row["basis"] != "not_compatible"]
    assert requested == disabled and requested[0] == "jacobi_fanuc_m710ic_45m"
    assert value["preparation_requested"][0]["blockers"] == catalog_candidate(
        load_default_catalog(), "jacobi_fanuc_m710ic_45m")["eligibility"]["blockers"]
    assert value["preparation"] == {"status": "not_started", "reason": ranking.PREPARATION_HOOK_MISSING}


def test_unknown_task_family_ranks_every_row_and_unknown_facts_stay_unknown(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(receipts, "reserve_website_preparation_spend", admitted(calls))
    value = ranking.rank_website_candidates(task_context=context(task_family="general_task", operator_task_details=None),
        task_masks=None, pipeline_dir=tmp_path / "pipeline", capture_root=tmp_path, catalog=catalog(),
        model_call=model(calls, partial_answer))
    assert value["task_kind"] is None and "not_compatible" not in {row["basis"] for row in value["candidates"]}
    assert value["facts"]["unknown"] == ["grounded_targets", "estimated_task_region_extent_m",
                                         "owner_stated_item_details"]


def test_business_brief_is_the_shared_platform_block_cited_by_digest(tmp_path):
    brief = ranking.business_context()
    assert brief["source_file"] == "PLATFORM_CONTEXT.md" and brief["section"] == "shared_platform_context"
    assert brief["text"].startswith("## Shared Platform Doctrine") and "Embodiment Rule" not in brief["text"]
    assert brief["sha256"] == canonical_digest({"text": brief["text"]})
    (tmp_path / "PLATFORM_CONTEXT.md").write_text("no shared block")
    with pytest.raises(ValueError, match="business_context_missing"):
        ranking.business_context(tmp_path / "PLATFORM_CONTEXT.md")


def test_preparation_ranks_once_grounded_targets_exist_and_before_the_native_handoff(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from blueprint_pipeline import website_scene_handoff
    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    from tests.test_qualification_coverage_edges import _descriptor

    events = []
    monkeypatch.setattr(ranking, "rank_website_candidates", lambda **kwargs: events.append(("rank", kwargs)))
    monkeypatch.setattr(website_scene_handoff, "prepare_website_scene_handoff",
                        lambda **_: events.append(("native", None)) or {"website_intake_outbox": {
                            "state": "forward_pending"}})
    pipeline = tmp_path / "capture" / "pipeline"
    pipeline.mkdir(parents=True)
    for allowed in (True, False):
        run_website_capture_preparation(
            descriptor=_descriptor(metadata={"site_task_context": context()}), capture_root=tmp_path / "capture",
            pipeline_dir=pipeline, source_video=None, derived_scene_generation_allowed=allowed, gates=[],
            prepare_media=lambda **_: {"status": "noop", "privacy_status": "website_capture_admitted",
                                       "privacy_verified": True, "task_masks": MASKS},
            check_privacy=lambda **_: SimpleNamespace(passed=True),
            make_gate=lambda _name, passed, _detail: SimpleNamespace(passed=passed))
    # Only a capture whose rights allow derived processing is ranked; the handoff always follows.
    assert [kind for kind, _ in events] == ["rank", "native", "native"]
    call = events[0][1]
    assert call["task_masks"] == MASKS and call["pipeline_dir"] == pipeline
    assert call["task_context"]["context_digest"] == context()["context_digest"]
