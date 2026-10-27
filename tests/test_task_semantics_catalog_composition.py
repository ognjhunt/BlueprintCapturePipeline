# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_task_semantics.py
#   src/blueprint_pipeline/task_family_preparation.py
#   src/blueprint_pipeline/website_task_context_contracts.py
"""Derived operation hints compose with the registered task-family contracts."""
from copy import deepcopy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.task_family_preparation import prepare_task_family
from blueprint_pipeline.task_family_scoring import FAMILY_ALIASES
from blueprint_pipeline import website_task_semantics as semantics
from blueprint_pipeline.website_task_context_contracts import retains_task_family_preparation
from tests.test_website_task_semantics import semantics_fixture


_FAMILY_OPERATIONS = {
    "pouring": "Pour the liquid into the receiver.",
    "wiping": "Wipe the work surface.",
    "surface_wiping": "Wipe the other work surface.",
    "insertion": "Insert the peg into its socket.",
    "stacking": "Stack the rigid block on the other block.",
    "button_press": "Press the start button.",
    "cloth_folding": "Fold the cloth along its marked line.",
    "mining": "Mine the rock with a pickaxe.",
    "unknown": "Perform the unspecified finishing step.",
    "rigid_pick_place": "Pick up the ball and place it on the desktop.",
    "articulated_open_close": "Open the cabinet door.",
}


def catalog_semantics_fixture(family, *, operations=None, declared_family="general_task"):
    """Return original context, video identity and derived completion evidence."""
    context, source, receipt = semantics_fixture()
    operations = operations or (["pouring", "wiping"] if family == "multi_step" else [family])
    phrases = [_FAMILY_OPERATIONS[operation] for operation in operations]
    context["description"] = " ".join(phrases) + " Preserve the captured environment."
    context["task_family"] = declared_family
    context["capture_rights"] = {"derived_scene_generation_allowed": True}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    receipt["binding"]["task_context_digest"] = context["context_digest"]
    receipt["proposal"]["task_semantics"] = {
        "operative_job": " ".join(phrases), "task_family": family,
        "steps": [{"operation": operation, "description": phrase, "task_quote": phrase}
                  for operation, phrase in zip(operations, phrases, strict=True)],
        "covers_entire_job": True, "unresolved_aspects": [],
        "preparation_notes": ["Preserve the captured environment."],
    }
    receipt["binding_digest"] = canonical_digest(receipt["binding"])
    receipt["proposal_digest"] = canonical_digest(receipt["proposal"])
    artifact = semantics.task_semantics_from_receipt(
        receipt=receipt, task_context=context, source_video_digest=source)
    return context, source, artifact


def _family_arguments(tmp_path, family="cloth_folding", *, operations=None):
    context, source, artifact = catalog_semantics_fixture(family, operations=operations)
    splat, collision = tmp_path / "scene.ply", tmp_path / "collision.glb"
    splat.write_bytes(b"development-only-source-appearance")
    collision.write_bytes(b"development-only-source-collision")
    geometry = {"binding": {"source_video_digest": source}, "frames": []}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    masks = {"source_geometry_digest": geometry["digest"], "targets": []}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    return dict(task_context=context, task_masks=masks, removal_manifest={"entries": []},
        source_geometry=geometry, base_scene={"splat_path": str(splat), "splat_digest": _sha256_file(splat),
            "collision_mesh_path": str(collision), "collision_mesh_digest": _sha256_file(collision),
            "up_axis": "Y", "meters_per_unit": 1.0},
        output_root=tmp_path / "preparation", task_semantics=artifact)


def _handoff(preparation, context, source_geometry):
    value = {"schema_version": "website_scene_handoff.v1", "scene_id": context["scene_id"],
        "capture_id": context["capture_id"], "visual_reconstruction_ready": True,
        "simulator_ready": False, "status": preparation["status"], "blockers": preparation["blockers"],
        "job_disposition": preparation["job_disposition"],
        "source_geometry_digest": source_geometry["digest"],
        "task_evaluation_inputs": deepcopy(preparation["task_evaluation_inputs"])}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


@pytest.mark.parametrize("family", FAMILY_ALIASES)
def test_general_task_uses_current_catalog_and_requests_exact_missing_family_contracts(tmp_path, family):
    arguments = _family_arguments(tmp_path, family)
    original = deepcopy(arguments["task_context"])
    result = prepare_task_family(**arguments)
    assert result["status"] == "needs_input" and result["job_disposition"] == "accepted"
    assert result["blockers"] == ["website_task_family_success_contract_required",
                                  "website_task_family_native_runtime_inputs_required"]
    assert result["intake_request"] is None and result["provider_mutation_performed"] is False
    inputs = result["task_evaluation_inputs"]
    assert inputs["task_family"] == family
    assert inputs["runtime_adapter"] == "task_evaluation_task_family_native_adapter"
    assert inputs["task_semantics"] == arguments["task_semantics"]
    assert inputs["source_video_digest"] == arguments["source_geometry"]["binding"]["source_video_digest"]
    assert inputs["scope"]["task_context_digest"] == original["context_digest"]
    assert inputs["source_bindings"]["task_context_digest"] == original["context_digest"]
    assert inputs["automatic_paid_execution_authorized"] is False
    assert arguments["task_context"] == original and original["task_family"] == "general_task"
    assert retains_task_family_preparation(_handoff(result, original, arguments["source_geometry"]), original)


def test_synthetic_ball_still_resolves_to_its_existing_rigid_profile():
    from blueprint_pipeline.website_task_preparation import native_task_preparation_gap
    context, source, receipt = semantics_fixture()
    artifact = semantics.task_semantics_from_receipt(receipt=receipt,
        task_context=context, source_video_digest=source)
    original = deepcopy(context)
    assert semantics.effective_task_profile(task_context=context, task_semantics=artifact,
        source_video_digest=source) == "rigid_pick_place"
    assert native_task_preparation_gap(context, inferred_profile_id="rigid_pick_place",
        task_semantics=artifact, source_video_digest=source) is None
    assert context == original


@pytest.mark.parametrize("declared", ["button_press", "cloth_folding", "rigid_pick_place", "mining"])
def test_explicit_original_family_wins_over_a_different_derived_family(declared):
    context, source, artifact = catalog_semantics_fixture("pouring", declared_family=declared)
    original = deepcopy(context)
    profile = semantics.effective_task_profile(task_context=context, task_semantics=artifact,
                                               source_video_digest=source)
    assert profile == (None if declared == "mining" else declared)
    assert context == original


@pytest.mark.parametrize("family", ["unknown", "mining"])
def test_unregistered_derived_operations_stay_held_without_a_narrower_fallback(family):
    from blueprint_pipeline.website_task_preparation import native_task_preparation_gap
    context, source, artifact = catalog_semantics_fixture(family)
    assert semantics.effective_task_profile(task_context=context, task_semantics=artifact,
                                            source_video_digest=source) is None
    gap = native_task_preparation_gap(context, inferred_profile_id="rigid_pick_place",
        task_semantics=artifact, source_video_digest=source)
    assert gap["code"] == "website_native_task_profile_preparation_required"
    assert gap["task_family"] == "general_task" and gap["description"] == context["description"]


def test_known_multi_step_preserves_each_ordered_operation_until_contract_preparation(tmp_path):
    arguments = _family_arguments(tmp_path, "multi_step", operations=["pouring", "wiping"])
    result = prepare_task_family(**arguments)
    inputs = result["task_evaluation_inputs"]
    assert inputs["task_family"] == "multi_step"
    assert [step["operation"] for step in inputs["task_semantics"]["semantics"]["steps"]] == ["pouring", "wiping"]
    assert inputs["requirements"]["multi_step"]["ordered_stages_required"] is True
    assert inputs["task_success_contract"] is None and inputs["native_runtime_contract"] is None
    assert result["blockers"] == ["website_task_family_success_contract_required",
                                  "website_task_family_native_runtime_inputs_required"]


@pytest.mark.parametrize("operations", [["pouring", "unknown"], ["pouring", "mining"],
                                         ["rigid_pick_place", "wiping"],
                                         ["articulated_open_close", "button_press"], ["pouring"],
                                         ["wiping", "wiping"], ["wiping", "surface_wiping"]])
def test_multi_step_never_discards_unknown_work_or_crosses_runtime_adapter_boundaries(operations):
    context, source, artifact = catalog_semantics_fixture("multi_step", operations=operations)
    assert semantics.effective_task_profile(task_context=context, task_semantics=artifact,
                                            source_video_digest=source) is None


@pytest.mark.parametrize("changed", ["video", "scope", "family", "context", "semantics_context",
                                     "semantics_receipt", "semantic_body", "missing_semantics",
                                     "source_binding_context", "source_binding_geometry"])
def test_retained_general_task_family_requires_the_same_derived_source_bindings(tmp_path, changed):
    arguments = _family_arguments(tmp_path)
    preparation = prepare_task_family(**arguments)
    context = deepcopy(arguments["task_context"])
    handoff = _handoff(preparation, context, arguments["source_geometry"])
    assert retains_task_family_preparation(handoff, context)
    inputs = handoff["task_evaluation_inputs"]
    if changed == "video":
        inputs["source_video_digest"] = "sha256:" + "d" * 64
    elif changed == "scope":
        inputs["scope"]["task_context_digest"] = "sha256:" + "d" * 64
    elif changed == "family":
        inputs["task_family"] = "pouring"
    elif changed == "context":
        context["description"] += " Wash the cloth afterwards."
        context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    elif changed == "semantics_context":
        inputs["task_semantics"]["binding"]["task_context_digest"] = "sha256:" + "d" * 64
    elif changed == "semantics_receipt":
        inputs["task_semantics"]["planner_receipt"]["usage"]["output_tokens"] += 1
    elif changed == "semantic_body":
        inputs["task_semantics"]["semantics"]["steps"][0]["operation"] = "rigid_pick_place"
    elif changed == "source_binding_context":
        inputs["source_bindings"]["task_context_digest"] = "sha256:" + "d" * 64
    elif changed == "source_binding_geometry":
        inputs["source_bindings"]["source_geometry_digest"] = "sha256:" + "d" * 64
    else:
        del inputs["task_semantics"]
    if changed.startswith("semantics_") or changed == "semantic_body":
        artifact = inputs["task_semantics"]
        artifact["digest"] = canonical_digest(artifact, digest_field="digest")
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    assert not retains_task_family_preparation(handoff, context)


def _ordered_family_contract(context, order):
    from blueprint_pipeline.task_family_scoring import seal_task_family_contract
    definitions = {
        "pouring": {
            "criteria": {"minimum_transferred_volume_ml": 1, "maximum_spilled_volume_ml": 1,
                "maximum_volume_balance_error_ml": 1, "maximum_final_in_flight_volume_ml": 0},
            "reset": {"source_volume_ml": 10, "receiver_volume_ml": 0, "in_flight_volume_ml": 0,
                "spilled_volume_ml": 0},
        },
        "wiping": {
            "criteria": {"minimum_cleaned_fraction": .5, "maximum_remaining_dirty_area_m2": 1},
            "reset": {"dirty_area_m2": 2},
        },
    }
    return seal_task_family_contract(task_family="multi_step",
        scope={"scene_id": context["scene_id"], "task_id": "website-" + context["context_digest"][7:27],
            "task_context_digest": context["context_digest"]},
        runtime_identity_digest="sha256:" + "d" * 64,
        stages=[{"stage_id": f"stage-{index}", "task_family": family, "minimum_hold_seconds": 0,
            "criteria": definitions[family]["criteria"]} for index, family in enumerate(order)],
        reset={f"stage-{index}": definitions[family]["reset"] for index, family in enumerate(order)})


def test_derived_multi_step_compiler_refuses_a_valid_contract_that_reorders_the_job(tmp_path):
    arguments = _family_arguments(tmp_path, "multi_step", operations=["pouring", "wiping"])
    context = arguments["task_context"]
    matching = _ordered_family_contract(context, ["pouring", "wiping"])
    preparation = prepare_task_family(**arguments, evaluation_inputs={"task_success_contract": matching})
    assert preparation["blockers"] == ["website_task_family_native_runtime_inputs_required"]
    reordered = _ordered_family_contract(context, ["wiping", "pouring"])
    with pytest.raises(ValueError, match="semantics|stage|operation"):
        prepare_task_family(**arguments, evaluation_inputs={"task_success_contract": reordered})


def test_retained_multi_step_revalidates_contract_order_against_original_semantics(tmp_path):
    arguments = _family_arguments(tmp_path, "multi_step", operations=["pouring", "wiping"])
    context = arguments["task_context"]
    contract = _ordered_family_contract(context, ["pouring", "wiping"])
    preparation = prepare_task_family(**arguments, evaluation_inputs={"task_success_contract": contract})
    handoff = _handoff(preparation, context, arguments["source_geometry"])
    assert retains_task_family_preparation(handoff, context)
    handoff["task_evaluation_inputs"]["task_success_contract"] = _ordered_family_contract(
        context, ["wiping", "pouring"])
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    assert not retains_task_family_preparation(handoff, context)
