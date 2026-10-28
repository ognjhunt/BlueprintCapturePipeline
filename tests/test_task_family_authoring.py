"""AUT-12 precursor retains missing facts; it never manufactures native inputs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_family_authoring as authoring
from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_task_semantics import task_semantics_from_receipt
from tests.test_task_family_preparation import _arguments_for_family, _evaluation_inputs


def _artifact(result):
    return result["task_evaluation_inputs"]["authoring"]


def test_ordinary_layout_and_job_retain_one_actionable_fact_without_native_injection(tmp_path, monkeypatch):
    args = _arguments_for_family(tmp_path, "stacking")
    monkeypatch.setattr(preparation, "register_source_to_runtime", lambda **_: pytest.fail("No rigid or provider work"))
    result = preparation.compile_website_scene_preparation(**args)
    value = _artifact(result)["request"]
    assert result["status"] == "needs_input" and result["job_disposition"] == "accepted"
    assert result["task_evaluation_inputs"]["task_success_contract"] is None
    assert result["task_evaluation_inputs"]["native_runtime_contract"] is None
    assert value["status"] == "needs_authoring" and value["precursor_only"]
    assert value["native_inputs_authored"] is False and value["owner_facts_confirmed"] is False
    assert value["next_action"]["quantity"] == "minimum_support_margin_m"
    assert value["next_action"]["category"] == "task_fact"
    assert "centre of mass" in value["next_action"]["question"]
    assert {row["category"] for row in value["requirements"]} == {
        "task_fact", "source_measurement", "runtime_integration"}
    assert value["source_frame_bindings"][0]["frame_id"] == "frame-0"
    assert value["source_targets"][0]["target_id"] == "cup-1"
    assert value["provider_mutation_performed"] is False
    assert value["automatic_paid_execution_authorized"] is False
    assert value["claim_ceiling"] == "development_only"


def test_restart_adopts_identical_requirement_bytes_without_a_new_generation(tmp_path):
    args = _arguments_for_family(tmp_path)
    first = preparation.compile_website_scene_preparation(**args)
    path = Path(_artifact(first)["reference"]["path"])
    before = path.read_bytes(), path.stat().st_mtime_ns
    second = preparation.compile_website_scene_preparation(**args)
    assert second == first
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert len(list((args["output_root"] / "family_authoring").glob("*/requirements.json"))) == 1


def test_changed_owner_facts_create_a_generation_without_overwriting_history(tmp_path):
    args = _arguments_for_family(tmp_path)
    first = _artifact(preparation.compile_website_scene_preparation(**args))
    original = Path(first["reference"]["path"]).read_bytes()
    args["task_context"]["success_criteria"] = {
        "successDefinition": "Press the button gently.", "successRate": 95,
        "cycleTimeSeconds": 10, "unknown": False}
    args["task_context"]["context_digest"] = canonical_digest(args["task_context"], digest_field="context_digest")
    second = _artifact(preparation.compile_website_scene_preparation(**args))
    assert second["reference"]["path"] != first["reference"]["path"]
    assert Path(first["reference"]["path"]).read_bytes() == original
    assert second["request"]["owner_criteria"] == args["task_context"]["success_criteria"]
    assert second["request"]["next_action"]["quantity"] == "minimum_button_travel_m"
    assert second["request"]["owner_facts_confirmed"] is False  # Prose did not become numbers.


def test_supplied_contract_is_adopted_while_native_fact_gaps_remain(tmp_path):
    args = _arguments_for_family(tmp_path)
    supplied = _evaluation_inputs(args)
    del supplied["native_runtime_inputs"]
    result = preparation.compile_website_scene_preparation(**args, evaluation_inputs=supplied)
    value = _artifact(result)["request"]
    assert result["status"] == "needs_input"
    assert result["task_evaluation_inputs"]["task_success_contract"] == supplied["task_success_contract"]
    assert value["retained_stage_inputs"] == {"task_success_contract": True, "native_runtime_contract": False}
    assert value["operations"][0]["stage_id"] == "stage1"
    assert all(row["category"] != "task_fact" for row in value["requirements"])
    assert value["next_action"]["quantity"] == "button_joint_axis_and_rest"


def _ordered_job(tmp_path):
    args = _arguments_for_family(tmp_path, "multi_step")
    context = args["task_context"]
    context["description"] = "Press the button, then wipe the table."
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    video = "sha256:" + "e" * 64
    args["source_geometry"]["binding"] = {"source_video_digest": video}
    args["source_geometry"]["digest"] = canonical_digest(args["source_geometry"], digest_field="digest")
    args["task_masks"]["source_geometry_digest"] = args["source_geometry"]["digest"]
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    binding = {"task_context_digest": context["context_digest"], "source_video_digest": video}
    semantics = {"operative_job": context["description"], "task_family": "multi_step",
        "steps": [{"operation": "button_press", "description": "Press the button", "task_quote": "Press the button"},
                  {"operation": "wiping", "description": "Wipe the table", "task_quote": "wipe the table"}],
        "covers_entire_job": True, "unresolved_aspects": [], "preparation_notes": []}
    proposal = {"status": "proposed", "task_semantics": semantics}
    receipt = {"status": "completed", "binding": binding, "binding_digest": canonical_digest(binding),
        "proposal": proposal, "proposal_digest": canonical_digest(proposal)}
    args["task_semantics"] = task_semantics_from_receipt(receipt=receipt,
        task_context=context, source_video_digest=video)
    return args


def test_source_bound_multistep_order_and_step_identities_survive_restart(tmp_path):
    args = _ordered_job(tmp_path)
    result = preparation.compile_website_scene_preparation(**args)
    value = _artifact(result)["request"]
    assert [row["task_family"] for row in value["operations"]] == ["button_press", "wiping"]
    assert [row["order"] for row in value["operations"]] == [1, 2]
    assert value["operations"][0]["source_operation"]["task_quote"] == "Press the button"
    assert value["operations"][1]["source_operation"]["task_quote"] == "wipe the table"
    assert len({row["stage_id"] for row in value["operations"]}) == 2
    assert preparation.compile_website_scene_preparation(**args) == result
    assert result["status"] == "needs_input" and value["native_inputs_authored"] is False


def test_unknown_multistep_order_asks_for_operations_instead_of_collapsing_the_job(tmp_path):
    args = _arguments_for_family(tmp_path, "multi_step")
    value = _artifact(preparation.compile_website_scene_preparation(**args))["request"]
    assert value["operations"] == []
    assert value["next_action"]["requirement_id"] == "ordered_operations"
    assert value["native_inputs_authored"] is False


def test_tampered_derived_operation_order_is_not_an_authoring_source(tmp_path):
    args = _ordered_job(tmp_path)
    args["task_semantics"] = copy.deepcopy(args["task_semantics"])
    args["task_semantics"]["semantics"]["steps"].reverse()
    args["task_semantics"]["digest"] = canonical_digest(args["task_semantics"], digest_field="digest")
    with pytest.raises(ValueError, match="website_task_semantics_changed"):
        preparation.compile_website_scene_preparation(**args)


def test_crash_after_requirement_persistence_replays_the_same_artifact(tmp_path, monkeypatch):
    args = _arguments_for_family(tmp_path)
    writer = authoring.write_exclusive_private_bytes
    def crash(path, payload):
        writer(path, payload)
        raise OSError("injected crash after immutable write")
    monkeypatch.setattr(authoring, "write_exclusive_private_bytes", crash)
    with pytest.raises(OSError, match="injected crash"):
        preparation.compile_website_scene_preparation(**args)
    files = list((args["output_root"] / "family_authoring").glob("*/requirements.json"))
    assert len(files) == 1
    before = files[0].read_bytes(), files[0].stat().st_mtime_ns
    monkeypatch.setattr(authoring, "write_exclusive_private_bytes", writer)
    result = preparation.compile_website_scene_preparation(**args)
    assert (files[0].read_bytes(), files[0].stat().st_mtime_ns) == before
    assert json.loads(before[0]) == _artifact(result)["request"]


def test_mutated_retained_requirement_is_refused_instead_of_overwritten(tmp_path):
    args = _arguments_for_family(tmp_path)
    first = _artifact(preparation.compile_website_scene_preparation(**args))
    path = Path(first["reference"]["path"])
    path.write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="immutable_input_staging_conflict"):
        preparation.compile_website_scene_preparation(**args)
    assert path.read_text() == '{"tampered":true}'
