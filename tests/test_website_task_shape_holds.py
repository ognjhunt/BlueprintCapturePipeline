# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_profile_contracts.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/website_task_preparation.py
#   src/blueprint_pipeline/website_task_context_contracts.py
"""Website task shapes the one rigid-object path cannot represent hold by name; nothing is paid.

Several task objects, or one soft object, under the rigid pick/place or articulated
open/close profile used to compile on (a towel authored as a rigid body) or fail late
as an untyped ``website_single_manipulated_subject_required``. They are now typed holds
before coverage, spec research, registration and authoring, and the stage keeps them.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline import website_task_semantics as semantics
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_profile_contracts import deformable_subject_noun
from blueprint_pipeline.website_task_context_contracts import holds_native_task_preparation
from tests.test_website_scene_handoff import ATLAS, atlas_ready, inputs
from tests.test_website_task_preparation import REMOVAL, _arguments
from tests.test_website_task_semantics import semantics_fixture

MULTI = "website_multi_object_task_preparation_required"
DEFORMABLE = "website_deformable_task_object_preparation_required"


class Reached(Exception):
    """The job passed the subject holds and reached registration."""


@pytest.mark.parametrize("label,noun", [
    ("folded bath towel", "towel"), ("stack of towels", "towel"), ("pile of clothes", "clothes"),
    ("blue plastic shopping bag", "bag"), ("USB charging cable", "cable"), ("t-shirt, folded", "t-shirt"),
    ("two cushions", "cushion"), ("the grey blanket on the sofa", "blanket"),
    ("towel rack", None), ("wire basket", None), ("cable box on the shelf", None), ("white ceramic mug", None),
    ("mug on the towel", None), ("wooden tray", None), ("cardboard boxes", None), ("", None), (None, None),
])
def test_only_the_task_objects_own_head_noun_reads_as_soft(label, noun):
    assert deformable_subject_noun(label) == noun


def _context(arguments, family):
    context = arguments["task_context"]
    context["task_family"] = family
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    return context


def _removal(*entries):
    base = REMOVAL["entries"][0]
    return {**REMOVAL, "entries": [{**deepcopy(base), **entry} for entry in entries]}


def _read_as_rigid(arguments):
    """A general job the source-bound reading resolved to rigid_pick_place (the WebApp's soft-object hint)."""
    context = _context(arguments, "general_task")
    _, source, receipt = semantics_fixture()
    receipt["binding"]["task_context_digest"] = context["context_digest"]
    receipt["binding_digest"] = canonical_digest(receipt["binding"])
    receipt["proposal"]["task_semantics"]["steps"][0]["task_quote"] = context["description"]
    receipt["proposal_digest"] = canonical_digest(receipt["proposal"])
    geometry = arguments["source_geometry"]
    geometry["binding"] = {"source_video_digest": source}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    arguments["task_masks"]["source_geometry_digest"] = geometry["digest"]
    arguments["task_masks"]["digest"] = canonical_digest(arguments["task_masks"], digest_field="digest")
    arguments["task_semantics"] = semantics.task_semantics_from_receipt(receipt=receipt, task_context=context,
                                                                       source_video_digest=source)


@pytest.mark.parametrize("family", ["rigid_pick_and_place", "general_task"])
def test_a_soft_task_object_is_held_before_registration_not_authored_rigid(tmp_path, monkeypatch, family):
    arguments = _arguments(tmp_path, removal_manifest=_removal({"semantic_label": "folded bath towel"}))
    if family == "general_task":
        _read_as_rigid(arguments)
    else:
        _context(arguments, family)
    monkeypatch.setattr(preparation, "register_source_to_runtime",
                        lambda **_: pytest.fail("A soft object must not be registered or authored as a rigid body"))
    result = preparation.compile_website_scene_preparation(**arguments)
    assert result["status"] == "needs_input" and result["blockers"] == [DEFORMABLE]
    assert result["intake_request"] is None and result["provider_mutation_performed"] is False
    gap = result["native_task_preparation_gap"]
    assert gap["job_disposition"] == "held_for_preparation" and gap["inferred_profile_id"] == "rigid_pick_place"
    assert gap["subject_target_ids"] == ["cup-1"] and gap["deformable_noun"] == "towel"
    assert gap["subject_basis"] == "manipulated_target_label_text_screen"
    assert ("task_semantics" in gap) is (family == "general_task")


@pytest.mark.parametrize("label", ["towel rack", "white ceramic mug"])
def test_a_rigid_object_with_a_soft_word_nearby_still_compiles(tmp_path, monkeypatch, label):
    arguments = _arguments(tmp_path, removal_manifest=_removal({"semantic_label": label}))
    _context(arguments, "rigid_pick_and_place")

    def registration(**_):
        raise Reached()

    monkeypatch.setattr(preparation, "register_source_to_runtime", registration)
    with pytest.raises(Reached):
        preparation.compile_website_scene_preparation(**arguments)


@pytest.mark.parametrize("family,profile", [("rigid_pick_and_place", "rigid_pick_place"),
                                            ("articulated_open_close", "articulated_open_close")])
def test_several_task_objects_are_held_not_narrowed_or_failed(tmp_path, monkeypatch, family, profile):
    arguments = _arguments(tmp_path, removal_manifest=_removal({}, {"target_id": "cup-2"}))
    _context(arguments, family)
    monkeypatch.setattr(preparation, "register_source_to_runtime", lambda **_: pytest.fail("No registration"))
    result = preparation.compile_website_scene_preparation(**arguments)
    assert result["blockers"] == [MULTI] and result["intake_request"] is None
    gap = result["native_task_preparation_gap"]
    assert gap["subject_target_ids"] == ["cup-1", "cup-2"] and gap["inferred_profile_id"] == profile
    assert "deformable_noun" not in gap


def test_a_legacy_context_without_a_family_keeps_its_compatibility_refusal(tmp_path):
    arguments = _arguments(tmp_path, removal_manifest=_removal({}, {"target_id": "cup-2"}))
    with pytest.raises(ValueError, match="website_single_manipulated_subject_required"):
        preparation.compile_website_scene_preparation(**arguments)


def test_an_articulated_fixture_and_a_family_task_never_take_the_soft_object_hold():
    context = {"task_family": "articulated_open_close", "description": "Open the curtain cabinet."}
    curtain = [{"target_id": "c1", "semantic_label": "curtain"}]
    assert preparation.native_task_subject_gap(context, subjects=curtain, targets={}) is None
    family = {"task_family": "cloth_folding", "description": "Fold the towel."}
    assert preparation.native_task_subject_gap(family, subjects=[{"target_id": "t", "semantic_label": "towel"}],
                                               targets={}) is None
    # An unread general job is held by its own profile gap, never by this one.
    general = {"task_family": "general_task", "description": "Fold the towel."}
    assert preparation.native_task_subject_gap(general, subjects=[{"target_id": "t", "semantic_label": "towel"}],
                                               targets={}) is None


@pytest.fixture
def ready(tmp_path, monkeypatch):
    value = atlas_ready(tmp_path)
    monkeypatch.setattr(ATLAS, lambda **_: value)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry",
                        lambda **kw: {**kw["task_masks"], "source_geometry_digest": "atlas-geometry"})
    return value


@pytest.mark.parametrize("entries,code", [
    ([{"target_id": "towel-1", "task_effect": "manipulated", "semantic_label": "folded bath towel"}], DEFORMABLE),
    ([{"target_id": "mug-1", "task_effect": "manipulated", "semantic_label": "mug"},
      {"target_id": "mug-2", "task_effect": "manipulated", "semantic_label": "mug"}], MULTI),
])
def test_the_handoff_holds_before_coverage_specs_and_authoring_and_the_stage_keeps_it(
        tmp_path, monkeypatch, ready, entries, code):
    kwargs = inputs(tmp_path)
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    context.update(schema_version="website_site_task_context.v1", task_family="rigid_pick_and_place",
                   description="Put it on the shelf.")
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    kwargs["clean_plate"]["task_masks"] = {"targets": [{"target_id": row["target_id"]} for row in entries]}
    write_json(tmp_path / "pipeline/removal.json", {"entries": entries})

    def unexpected(**_):
        pytest.fail("A held task object buys no coverage, spec research or authoring")

    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", unexpected)
    monkeypatch.setattr("blueprint_pipeline.website_object_spec_research.attach_object_specs", unexpected)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", unexpected)
    monkeypatch.setattr(handoff, "prepare_website_runtime_inputs", unexpected)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "needs_input" and result["blockers"] == [code]
    assert result["visual_reconstruction_ready"] is True and result["simulator_ready"] is False
    assert "runtime_inputs" not in result and "website_intake_outbox" not in result
    assert result["source_geometry_digest"] == ready["source_geometry"]["digest"]
    # The capture stage and the status reader accept it as held, not failed.
    assert holds_native_task_preparation(result, context)

    def tampered(change):
        value = deepcopy(result)
        change(value["native_task_preparation_gap"])
        value["digest"] = canonical_digest(value, digest_field="digest")
        return holds_native_task_preparation(value, context)

    assert not tampered(lambda gap: gap.update(inferred_profile_id="articulated_open_close"))
    assert not tampered(lambda gap: gap.update(subject_target_ids=gap["subject_target_ids"][:1] * 2))
    if code == DEFORMABLE:
        assert not tampered(lambda gap: gap.update(subject_label="white ceramic mug"))
    else:
        assert not tampered(lambda gap: gap.update(subject_target_ids=gap["subject_target_ids"][:1]))


def test_a_soft_object_read_from_a_general_job_is_a_verified_hold():
    context, source, receipt = semantics_fixture()
    artifact = semantics.task_semantics_from_receipt(receipt=receipt, task_context=context, source_video_digest=source)
    gap = preparation.native_task_subject_gap(context, subjects=[{"target_id": "towel", "semantic_label": "bath towel"}],
                                              targets={}, task_semantics=artifact, source_video_digest=source)
    assert gap["code"] == DEFORMABLE and gap["task_semantics"] == artifact
    value = {"schema_version": "website_scene_handoff.v1", "status": "needs_input",
             "scene_id": context["scene_id"], "capture_id": context["capture_id"], "blockers": [gap["code"]],
             "native_task_preparation_gap": gap, "simulator_ready": False, "visual_reconstruction_ready": True}
    value["digest"] = canonical_digest(value, digest_field="digest")
    assert holds_native_task_preparation(value, context)
    gap["source_video_digest"] = "sha256:" + "f" * 64
    value["digest"] = canonical_digest(value, digest_field="digest")
    assert not holds_native_task_preparation(value, context)
