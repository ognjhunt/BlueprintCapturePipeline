# Covers (for impacted-test selection):
#   src/blueprint_pipeline/atlas_scene_planner.py
#   src/blueprint_pipeline/website_atlas_pipeline.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/website_task_preparation.py
#   src/blueprint_pipeline/website_task_context_contracts.py
"""Beta scene→native preparation consumes source-bound operation evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace

from agents import AgentOutputSchema
import pytest

from blueprint_pipeline import atlas_scene_planner as planner
from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline import website_task_semantics as semantics
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_atlas_scene_planner import fixture as planner_fixture
from tests.test_website_task_semantics import semantics_fixture


@pytest.mark.parametrize("schema", [planner.SemanticSceneProposal, planner.SemanticRefinementProposal])
def test_operation_semantics_compile_in_real_strict_agents_schema(schema):
    compiled = AgentOutputSchema(schema).json_schema()
    assert "task_semantics" in compiled["required"]
    assert compiled["$defs"]["TaskSemantics"]["additionalProperties"] is False


def test_existing_planner_call_retains_semantics_and_adopts_legacy_completion(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_context as control
    from blueprint_pipeline import paid_resource_admission as admission
    path, survey, proposal, policy = planner_fixture(tmp_path)
    context, _, semantic_receipt = semantics_fixture()
    proposal["task_semantics"] = semantic_receipt["proposal"]["task_semantics"]
    context["success_criteria"] = {"successDefinition": "The ball stays on the desktop."}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    bought = []
    monkeypatch.setattr(admission, "require_paid_resource_admission_grant", lambda *_, **__: None)
    monkeypatch.setattr(control, "reserve_website_preparation_spend", lambda **kw: (bought.append(kw) or {}, object()))
    monkeypatch.setattr(control, "website_webapp_request", lambda **_: {"status": "settled"})

    class Invoker:
        def configure_reservation_audit(self, **_):
            pass

        def invoke(self, spec, inputs):
            assert spec.output_type is planner.SemanticSceneProposal
            task = json.loads(inputs[0]["content"][0]["text"])
            assert task["task_requirements"]["success_criteria"] == context["success_criteria"]
            return SimpleNamespace(output=proposal, cost_usd=.03, usage={"input_tokens": 123},
                                   model=spec.model, sdk_version="fake")

    options = dict(survey_path=path, task_context=context, configuration={"policy": policy},
                   output_root=tmp_path / "plans", invoker=Invoker(), release_blockers=lambda: ("c" * 40, []))
    receipt = planner.review_survey(**options)
    assert planner.review_survey(**options) == receipt and len(bought) == 1
    artifact = semantics.task_semantics_from_receipt(receipt=receipt, task_context=context,
                                                     source_video_digest=survey["source_video_digest"])
    assert semantics.resolved_task_profile(artifact, task_context=context,
                                           source_video_digest=survey["source_video_digest"]) == "rigid_pick_place"
    # Simulate the exact retained completed pre-semantics request. A schema
    # deployment must adopt it, leaving its missing semantics unresolved.
    old = deepcopy(receipt)
    old["binding"]["instructions"] = old["binding"]["instructions"].replace(semantics.TASK_SEMANTICS_INSTRUCTIONS, "")
    old["binding"]["output_schema_digest"] = canonical_digest(planner.CurrentSceneProposal.model_json_schema())
    old["binding_digest"] = canonical_digest(old["binding"])
    old["proposal"].pop("task_semantics")
    old["proposal_digest"] = canonical_digest(old["proposal"])
    (options["output_root"] / (receipt["binding_digest"][7:] + ".json")).unlink()
    write_json(options["output_root"] / (old["binding_digest"][7:] + ".json"), old)
    assert planner.review_survey(**options) == old and len(bought) == 1
    assert semantics.task_semantics_from_receipt(receipt=old, task_context=context,
                                                 source_video_digest=survey["source_video_digest"]) is None


def _compiler_arguments(tmp_path):
    from tests.test_website_task_preparation import _arguments, synthetic_arguments
    args = synthetic_arguments(_arguments(tmp_path))
    _, source, receipt = semantics_fixture()
    context = args["task_context"]
    context["task_family"] = "general_task"
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    receipt["binding"]["task_context_digest"] = context["context_digest"]
    receipt["binding_digest"] = canonical_digest(receipt["binding"])
    step = receipt["proposal"]["task_semantics"]["steps"][0]
    step["task_quote"] = context["description"]
    receipt["proposal_digest"] = canonical_digest(receipt["proposal"])
    geometry = args["source_geometry"]
    geometry["binding"] = {"source_video_digest": source}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    args["task_masks"]["source_geometry_digest"] = geometry["digest"]
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    args["task_semantics"] = semantics.task_semantics_from_receipt(receipt=receipt, task_context=context,
                                                                  source_video_digest=source)
    return args


def test_general_job_compiler_uses_semantics_without_changing_owner_context(tmp_path):
    args = _compiler_arguments(tmp_path)
    original = deepcopy(args["task_context"])
    result = preparation.compile_website_scene_preparation(**args)
    assert "native_task_preparation_gap" not in result
    assert result["task_semantics"] == args["task_semantics"]
    assert result["binding"]["task_context_digest"] == original["context_digest"]
    assert args["task_context"] == original and original["task_family"] == "general_task"


@pytest.mark.parametrize("fault", ["missing", "video", "context", "multiple", "unsupported", "mechanism"])
def test_general_job_compiler_holds_bad_or_inapplicable_semantics_before_registration(tmp_path, monkeypatch, fault):
    args = _compiler_arguments(tmp_path)
    value = args["task_semantics"]
    if fault == "missing":
        args.pop("task_semantics")
    elif fault == "video":
        args["source_geometry"]["binding"]["source_video_digest"] = "sha256:" + "f" * 64
    elif fault == "context":
        args["task_context"]["description"] += " Different requirements."
        args["task_context"]["context_digest"] = canonical_digest(args["task_context"], digest_field="context_digest")
    else:
        receipt = value["planner_receipt"]
        proposed = receipt["proposal"]["task_semantics"]
        if fault == "multiple":
            proposed["steps"] *= 2
        elif fault == "unsupported":
            proposed["steps"][0]["operation"] = "unsupported"
        else:
            proposed["task_family"] = "articulated_open_close"
            proposed["steps"][0]["operation"] = "articulated_open_close"
        receipt["proposal_digest"] = canonical_digest(receipt["proposal"])
        args["task_semantics"] = semantics.task_semantics_from_receipt(receipt=receipt,
            task_context=args["task_context"], source_video_digest=value["binding"]["source_video_digest"])
    args["source_geometry"]["digest"] = canonical_digest(args["source_geometry"], digest_field="digest")
    monkeypatch.setattr(preparation, "register_source_to_runtime", lambda **_: pytest.fail("No registration before valid routing"))
    if fault in {"context", "video"}:
        with pytest.raises(ValueError, match="semantics_planner_binding_invalid"):
            preparation.compile_website_scene_preparation(**args)
    else:
        result = preparation.compile_website_scene_preparation(**args)
        assert result["intake_request"] is None
        assert result["blockers"] == ["website_task_profile_mechanism_mismatch" if fault == "mechanism"
                                       else "website_native_task_profile_preparation_required"]


def test_selected_semantics_pass_both_handoff_gaps_and_reach_compiler(tmp_path, monkeypatch):
    from blueprint_pipeline import website_scene_handoff as handoff
    from tests.test_website_scene_handoff import inputs, atlas_ready, ATLAS
    kwargs = inputs(tmp_path)
    context, source, receipt = semantics_fixture()
    context.update(capture_id="cap-1", scene_id="scene-1")
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    receipt["binding"]["task_context_digest"] = context["context_digest"]
    receipt["binding_digest"] = canonical_digest(receipt["binding"])
    kwargs["descriptor"]["metadata"]["site_task_context"] = context
    ready = atlas_ready(tmp_path)
    ready["source_geometry"]["binding"] = {"source_video_digest": source}
    monkeypatch.setattr(ATLAS, lambda **_: ready)
    job = tmp_path / "pipeline/atlas"
    write_json(job / "selection_binding.json", {"planner_receipt": receipt})
    artifact = semantics.retain_task_semantics(job_dir=job, receipt=receipt, task_context=context,
                                                source_video_digest=source)
    kwargs["clean_plate"]["task_masks"] = {"targets": [{"target_id": "ball"}]}
    write_json(tmp_path / "pipeline/removal.json", {"entries": [{"target_id": "ball", "task_effect": "manipulated"}]})
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **kw: kw["task_masks"])
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", lambda **kw: kw["task_masks"])
    monkeypatch.setattr("blueprint_pipeline.website_object_spec_research.attach_object_specs", lambda **kw: kw["task_masks"])

    class CompilerReached(Exception):
        pass

    def compile_here(**kw):
        assert kw["task_semantics"] == artifact and kw["task_context"] == context
        raise CompilerReached()

    monkeypatch.setattr(handoff, "compile_website_scene_preparation", compile_here)
    with pytest.raises(CompilerReached):
        handoff.prepare_website_scene_handoff(**kwargs)


def test_resolved_general_task_mechanism_gap_remains_a_verified_hold(tmp_path):
    from blueprint_pipeline.website_task_context_contracts import holds_native_task_preparation
    context, source, receipt = semantics_fixture()
    artifact = semantics.task_semantics_from_receipt(receipt=receipt, task_context=context, source_video_digest=source)
    gap = preparation.native_task_preparation_gap(context, inferred_profile_id="articulated_open_close",
                                                task_semantics=artifact, source_video_digest=source)
    handoff = {"schema_version": "website_scene_handoff.v1", "status": "needs_input",
        "scene_id": context["scene_id"], "capture_id": context["capture_id"], "blockers": [gap["code"]],
        "native_task_preparation_gap": gap, "simulator_ready": False, "visual_reconstruction_ready": True}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    assert holds_native_task_preparation(handoff, context)
    gap["source_video_digest"] = "sha256:" + "f" * 64
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    assert not holds_native_task_preparation(handoff, context)


@pytest.mark.parametrize("family", ["cloth_folding", "multi_step"])
def test_derived_registered_family_crosses_handoff_and_compiler_to_specific_preparation(tmp_path, monkeypatch, family):
    from blueprint_pipeline import website_scene_handoff as handoff
    from blueprint_pipeline.website_task_context_contracts import retains_task_family_preparation
    from tests.test_task_semantics_catalog_composition import _family_arguments
    args = _family_arguments(tmp_path, family)
    context = args["task_context"]
    context.update(confirmed=True, confirmed_at="2026-10-27T00:00:00Z")
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    receipt = args["task_semantics"]["planner_receipt"]
    receipt["binding"]["task_context_digest"] = context["context_digest"]
    receipt["binding_digest"] = canonical_digest(receipt["binding"])
    geometry = args["source_geometry"]
    geometry["geometry_backend"] = "world_labs_atlas"
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    args["task_masks"]["source_geometry_digest"] = geometry["digest"]
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    job = tmp_path / "pipeline/atlas"
    write_json(job / "selection_binding.json", {"planner_receipt": receipt})
    semantics.retain_task_semantics(job_dir=job, receipt=receipt, task_context=context,
                                    source_video_digest=geometry["binding"]["source_video_digest"])
    removal = tmp_path / "pipeline/removal.json"
    write_json(removal, args["removal_manifest"])
    monkeypatch.setattr("blueprint_pipeline.website_atlas_scene_source.atlas_scene_source", lambda **_: {
        "status": "ready", "base_scene": args["base_scene"], "source_geometry": geometry,
        "source_pose_digest": "source", "background_pose_digest": "background", "alignment": "anchored"})
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **kw: kw["task_masks"])
    original = deepcopy(context)
    result = handoff.prepare_website_scene_handoff(descriptor={"scene_id": context["scene_id"],
        "capture_id": context["capture_id"], "metadata": {"site_task_context": context}},
        clean_plate={"privacy_verified": True, "status": "noop", "task_masks": args["task_masks"],
                     "removal_manifest_path": str(removal)}, capture_root=tmp_path, now=1800000000)
    assert result["status"] == "needs_input"
    assert result["blockers"] == ["website_task_family_success_contract_required",
                                  "website_task_family_native_runtime_inputs_required"]
    assert result["task_evaluation_inputs"]["task_family"] == family
    assert retains_task_family_preparation(result, context)
    assert context == original and context["task_family"] == "general_task"
