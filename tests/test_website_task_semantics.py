# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_task_semantics.py
"""Beta scene preparation: derived routing never rewrites the owner job."""
from copy import deepcopy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import website_task_semantics as semantics


def semantics_fixture():
    """A source-bound planner completion, reusable by handoff/compiler tests."""
    context = {
        "schema_version": "website_site_task_context.v1",
        "request_id": "synthetic-job", "scene_id": "site-synthetic-job",
        "capture_id": "walkthrough-synthetic-job", "task_family": "general_task",
        "description": (
            "Add a new synthetic ball on the floor, then have a robot pick it up and place it on the desktop. "
            "Preserve all captured obstacles; use the existing scene pipeline. Budget: USD 35; no physical claim."
        ),
        "confirmed": False, "confirmed_at": None,
    }
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    source = "sha256:" + "a" * 64
    proposal = {
        "status": "proposed",
        "task_semantics": {
            "operative_job": "Lift the new ball from the floor and place it on the desktop.",
            "task_family": "rigid_pick_place",
            "steps": [{
                "operation": "rigid_pick_place",
                "description": "Pick up the ball from the floor and place it on the desktop.",
                "task_quote": "have a robot pick it up and place it on the desktop",
            }],
            "covers_entire_job": True, "unresolved_aspects": [],
            "preparation_notes": [
                "Add a new synthetic ball on the floor", "Preserve all captured obstacles",
                "Budget: USD 35; no physical claim.",
            ],
        },
    }
    binding = {
        "task_context_digest": context["context_digest"], "source_video_digest": source,
        "survey_digest": "sha256:" + "b" * 64, "model": "fake-review-model",
    }
    receipt = {
        "status": "completed", "binding": binding, "binding_digest": canonical_digest(binding),
        "proposal": proposal, "proposal_digest": canonical_digest(proposal),
        "source_commit": "c" * 40, "usage": {"input_tokens": 123, "output_tokens": 456},
        "cost_usd": 0.01, "claim_ceiling": "development_only",
    }
    return context, source, receipt


def _artifact(context, source, receipt):
    return semantics.task_semantics_from_receipt(
        receipt=receipt, task_context=context, source_video_digest=source)


def _artifact_path(job_dir, artifact):
    return job_dir / "task_semantics" / (artifact["digest"][7:] + ".json")


def _refresh_receipt(receipt):
    receipt["binding_digest"] = canonical_digest(receipt["binding"])
    receipt["proposal_digest"] = canonical_digest(receipt["proposal"])


def test_bound_setup_and_one_robot_operation_resolve_without_changing_the_owner_context(tmp_path):
    context, source, receipt = semantics_fixture()
    original = deepcopy(context)
    artifact = semantics.retain_task_semantics(
        job_dir=tmp_path, receipt=receipt, task_context=context, source_video_digest=source)
    assert artifact["binding"] == {
        "task_context_digest": context["context_digest"], "source_video_digest": source,
        "planner_binding_digest": receipt["binding_digest"],
        "planner_proposal_digest": receipt["proposal_digest"],
        "planner_receipt_digest": canonical_digest(receipt),
    }
    assert artifact["claim_ceiling"] == "development_only"
    assert artifact["planner_receipt"] == receipt
    assert artifact["semantics"]["preparation_notes"]
    assert json.loads(_artifact_path(tmp_path, artifact).read_text()) == artifact
    assert semantics.resolved_task_profile(
        artifact, task_context=context, source_video_digest=source) == "rigid_pick_place"
    assert context == original and context["task_family"] == "general_task"


@pytest.mark.parametrize("changed", ["context_body", "context_digest", "context_identity", "video", "status",
                                     "binding", "binding_digest", "proposal", "proposal_digest"])
def test_receipt_binding_refuses_changed_owner_source_or_completion(changed):
    context, source, receipt = semantics_fixture()
    if changed == "context_body":
        context["description"] += " Wash the ball afterwards."
    elif changed == "context_digest":
        context["context_digest"] = "sha256:" + "d" * 64
    elif changed == "context_identity":
        context["capture_id"] = "walkthrough-another-job"
        context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    elif changed == "video":
        source = "sha256:" + "d" * 64
    elif changed == "status":
        receipt["status"] = "invalid_output"
    elif changed == "binding":
        receipt["binding"]["source_video_digest"] = "sha256:" + "d" * 64
    elif changed == "binding_digest":
        receipt["binding_digest"] = "sha256:" + "d" * 64
    elif changed == "proposal":
        receipt["proposal"]["task_semantics"]["task_family"] = "articulated_open_close"
    else:
        receipt["proposal_digest"] = "sha256:" + "d" * 64
    with pytest.raises(ValueError, match="website_task_semantics_planner_binding_invalid"):
        _artifact(context, source, receipt)


@pytest.mark.parametrize("quote", ["pick up the ball", "Pick it up and place it on the desktop", "Wash the ball"])
def test_operation_quote_must_copy_the_original_task_verbatim(quote):
    context, source, receipt = semantics_fixture()
    receipt["proposal"]["task_semantics"]["steps"][0]["task_quote"] = quote
    _refresh_receipt(receipt)
    with pytest.raises(ValueError, match="website_task_semantics_task_quote_invalid"):
        _artifact(context, source, receipt)


@pytest.mark.parametrize("changed", ["family", "notes", "binding", "receipt_usage", "receipt_proposal", "digest"])
def test_self_hashed_artifact_cannot_substitute_for_the_bound_planner_completion(changed):
    context, source, receipt = semantics_fixture()
    artifact = _artifact(context, source, receipt)
    if changed == "family":
        artifact["semantics"]["task_family"] = "articulated_open_close"
    elif changed == "notes":
        artifact["semantics"]["preparation_notes"] = []
    elif changed == "binding":
        artifact["binding"]["planner_binding_digest"] = "sha256:" + "d" * 64
    elif changed == "receipt_usage":
        artifact["planner_receipt"]["usage"]["output_tokens"] += 1
    elif changed == "receipt_proposal":
        artifact["planner_receipt"]["proposal"]["task_semantics"]["operative_job"] = "Another task"
        _refresh_receipt(artifact["planner_receipt"])
    else:
        artifact["digest"] = "sha256:" + "d" * 64
    if changed != "digest":
        artifact["digest"] = canonical_digest(artifact, digest_field="digest")
    with pytest.raises(ValueError, match="website_task_semantics_changed"):
        semantics.validate_task_semantics(artifact, task_context=context, source_video_digest=source)


@pytest.mark.parametrize("change", ["incomplete", "unresolved", "general", "unknown", "unsupported",
                                    "no_steps", "different_profile", "multiple_supported", "additional_unsupported"])
def test_ambiguous_or_more_than_one_operation_stays_held(change):
    context, source, receipt = semantics_fixture()
    reading = receipt["proposal"]["task_semantics"]
    if change == "incomplete":
        reading["covers_entire_job"] = False
    elif change == "unresolved":
        reading["unresolved_aspects"] = ["The requested second operation is unclear."]
    elif change == "general":
        reading["task_family"] = "general_task"
    elif change in {"unknown", "unsupported"}:
        reading["steps"][0]["operation"] = change
    elif change == "no_steps":
        reading["steps"] = []
    elif change == "different_profile":
        reading["task_family"] = "articulated_open_close"
    else:
        extra = deepcopy(reading["steps"][0])
        if change == "additional_unsupported":
            extra["operation"] = "unsupported"
        reading["steps"].append(extra)
    _refresh_receipt(receipt)
    artifact = _artifact(context, source, receipt)
    assert semantics.resolved_task_profile(artifact, task_context=context, source_video_digest=source) is None


@pytest.mark.parametrize("description,family", [
    ("Load the washer and wash the clothes.", "washing"),
    ("Fold the towels and place them on the shelf.", "laundry_folding"),
    ("Close the cabinet door.", "articulated_closing"),
])
def test_washing_folding_and_closing_do_not_become_pick_place(description, family):
    context, source, receipt = semantics_fixture()
    context["description"] = description
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    receipt["binding"]["task_context_digest"] = context["context_digest"]
    receipt["proposal"]["task_semantics"] = {
        "operative_job": description, "task_family": family,
        "steps": [{"operation": "unsupported", "description": description, "task_quote": description}],
        "covers_entire_job": True, "unresolved_aspects": [], "preparation_notes": [],
    }
    _refresh_receipt(receipt)
    artifact = _artifact(context, source, receipt)
    assert semantics.resolved_task_profile(artifact, task_context=context, source_video_digest=source) is None


@pytest.mark.parametrize("declared", ["articulated_open_close", "rigid_pick_place", "laundry_folding"])
def test_explicit_declared_families_remain_authoritative(declared):
    from blueprint_pipeline.website_task_preparation import native_task_preparation_gap
    context, source, receipt = semantics_fixture()
    context["task_family"] = declared
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    receipt["binding"]["task_context_digest"] = context["context_digest"]
    _refresh_receipt(receipt)
    artifact = _artifact(context, source, receipt)
    assert semantics.resolved_task_profile(artifact, task_context=context, source_video_digest=source) is None
    gap = native_task_preparation_gap(context, inferred_profile_id="rigid_pick_place",
        task_semantics=artifact, source_video_digest=source)
    if declared == "rigid_pick_place":
        assert gap is None
    else:
        assert gap["code"] == ("website_task_profile_mechanism_mismatch" if declared == "articulated_open_close"
                               else "website_native_task_profile_preparation_required")
        assert gap["task_family"] == declared


def test_loader_anchors_to_the_selected_completion_and_detects_changed_artifacts(tmp_path):
    context, source, receipt = semantics_fixture()
    artifact = semantics.retain_task_semantics(
        job_dir=tmp_path, receipt=receipt, task_context=context, source_video_digest=source)
    assert semantics.load_task_semantics(job_dir=tmp_path, task_context=context, source_video_digest=source) is None
    selection = tmp_path / "selection_binding.json"
    selection.write_text(json.dumps({"planner_receipt": receipt}))
    assert semantics.load_task_semantics(job_dir=tmp_path, task_context=context, source_video_digest=source) == artifact
    retained_path = _artifact_path(tmp_path, artifact)
    artifact["semantics"]["task_family"] = "articulated_open_close"
    artifact["digest"] = canonical_digest(artifact, digest_field="digest")
    retained_path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="website_task_semantics_changed"):
        semantics.load_task_semantics(job_dir=tmp_path, task_context=context, source_video_digest=source)


@pytest.mark.parametrize("missing", ["artifact", "symlink_artifact", "symlink_selection"])
def test_selected_semantics_cannot_use_missing_or_symlinked_evidence(tmp_path, missing):
    context, source, receipt = semantics_fixture()
    artifact = semantics.retain_task_semantics(
        job_dir=tmp_path, receipt=receipt, task_context=context, source_video_digest=source)
    selection = tmp_path / "selection_binding.json"
    selection.write_text(json.dumps({"planner_receipt": receipt}))
    target = _artifact_path(tmp_path, artifact)
    if missing == "symlink_selection":
        saved = tmp_path / "selected.json"
        selection.rename(saved)
        selection.symlink_to(saved)
    else:
        target.unlink()
        if missing == "symlink_artifact":
            saved = tmp_path / "foreign.json"
            saved.write_text(json.dumps(artifact))
            target.symlink_to(saved)
    with pytest.raises(ValueError, match="website_task_semantics_(selection_invalid|artifact_missing)"):
        semantics.load_task_semantics(job_dir=tmp_path, task_context=context, source_video_digest=source)


def test_completed_legacy_plan_is_reusable_and_cannot_inherit_unrelated_new_semantics(tmp_path):
    context, source, receipt = semantics_fixture()
    semantics.retain_task_semantics(job_dir=tmp_path, receipt=receipt,
        task_context=context, source_video_digest=source)
    del receipt["proposal"]["task_semantics"]
    _refresh_receipt(receipt)
    before = deepcopy(receipt)
    assert _artifact(context, source, receipt) is None
    (tmp_path / "selection_binding.json").write_text(json.dumps({"planner_receipt": receipt}))
    assert semantics.load_task_semantics(job_dir=tmp_path, task_context=context, source_video_digest=source) is None
    assert receipt == before
    assert semantics.resolved_task_profile(None, task_context=context, source_video_digest=source) is None


@pytest.mark.parametrize("field", ["success_criteria", "operator_answers", "operator_task_details"])
def test_verbatim_requirement_quotes_are_bound_to_the_complete_current_context(field):
    context, source, receipt = semantics_fixture()
    quote = "Lift the rigid workpiece and place it on the desktop."
    context["description"] = "Transfer the rigid workpiece as specified in the task requirements."
    context[field] = {"requirement": quote}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    receipt["binding"]["task_context_digest"] = context["context_digest"]
    receipt["proposal"]["task_semantics"]["steps"][0]["task_quote"] = quote
    _refresh_receipt(receipt)
    artifact = _artifact(context, source, receipt)
    assert semantics.resolved_task_profile(artifact, task_context=context, source_video_digest=source) == "rigid_pick_place"
    context[field]["requirement"] = "Wash the workpiece before transferring it."
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    with pytest.raises(ValueError, match="website_task_semantics_planner_binding_invalid"):
        semantics.validate_task_semantics(artifact, task_context=context, source_video_digest=source)


def test_missing_video_identity_cannot_admit_derived_semantics():
    context, source, receipt = semantics_fixture()
    with pytest.raises(ValueError, match="website_task_semantics_source_video_missing"):
        semantics.resolved_task_profile(_artifact(context, source, receipt),
            task_context=context, source_video_digest=None)


def test_immutable_retention_reuses_each_completion_and_keeps_distinct_receipts(tmp_path):
    context, source, receipt = semantics_fixture()
    first = semantics.retain_task_semantics(job_dir=tmp_path, receipt=receipt,
        task_context=context, source_video_digest=source)
    path = _artifact_path(tmp_path, first)
    before = path.read_bytes()
    second = semantics.retain_task_semantics(job_dir=tmp_path, receipt=receipt,
        task_context=context, source_video_digest=source)
    assert second == first and path.read_bytes() == before
    receipt["usage"]["output_tokens"] += 1
    newer = semantics.retain_task_semantics(job_dir=tmp_path, receipt=receipt,
        task_context=context, source_video_digest=source)
    assert newer["digest"] != first["digest"]
    assert path.read_bytes() == before and _artifact_path(tmp_path, newer).is_file()
    assert len(list((tmp_path / "task_semantics").glob("*.json"))) == 2


def test_selected_completion_cannot_inherit_another_completion_semantics(tmp_path):
    context, source, receipt = semantics_fixture()
    semantics.retain_task_semantics(job_dir=tmp_path, receipt=receipt,
        task_context=context, source_video_digest=source)
    # A different completion has its own immutable identity, even if its
    # operative wording and proposed profile happen to agree.
    selected = deepcopy(receipt)
    selected["usage"]["output_tokens"] += 1
    (tmp_path / "selection_binding.json").write_text(json.dumps({"planner_receipt": selected}))
    with pytest.raises(ValueError, match="website_task_semantics_artifact_missing"):
        semantics.load_task_semantics(job_dir=tmp_path, task_context=context, source_video_digest=source)
