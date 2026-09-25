"""Completed drawer CAD reuse is source-bound and performs no model call."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_partial_astra_successor import (
    SCHEMA_VERSION, prepare_completed_articulated_successor, restore_partial_astra,
    semantic_articulated_requests,
)
from blueprint_pipeline.task_evaluation_scene_configuration_astra_phase_adoption import _inventory
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError, file_record, validate_request

META = json.loads((Path(__file__).parent / "fixtures/partial_astra_eae_metadata.json").read_text())
SOURCE_ENVELOPE = "sha256:" + "1" * 64
SUCCESSOR_ENVELOPE = "sha256:" + "2" * 64
SOURCE_GEOMETRY = {"source_candidate_digest": "sha256:" + "3" * 64,
                   "construction_envelope_digest": SOURCE_ENVELOPE,
                   "configuration_digest": "sha256:" + "4" * 64,
                   "source_aabb_min_xyz_m": [0, 0, 0],
                   "source_aabb_max_xyz_m": [1, 1, 1]}


def sealed(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


@pytest.fixture
def completed(tmp_path):
    prior = tmp_path / "original" / "astra_cad_blender_runtime"
    successor = tmp_path / "successor"
    old, new = {}, {}
    parts = {}
    for part in ("carcass", "drawer"):
        base = copy.deepcopy(META["request"])
        frame = tmp_path / "source-frame.png"
        frame.write_bytes(b"source-frame")
        for source in base["source_frames"]:
            source["path"] = str(frame)
            source["sha256"] = file_record(frame)["sha256"]
        base["object_id"] = "office_cabinet__" + part
        base["physical_review_input"]["object_id"] = base["object_id"]
        base["run_id"] = "original-run"
        source_constraints = {"source_geometry_receipt": SOURCE_GEOMETRY}
        base["construction_constraints"] = json.dumps(source_constraints)
        old[part] = sealed(base, "request_digest")
        validate_request(old[part])
        successor_constraints = copy.deepcopy(source_constraints)
        successor_constraints["source_geometry_receipt"]["construction_envelope_digest"] = SUCCESSOR_ENVELOPE
        new[part] = sealed(dict(old[part], run_id="successor-run",
                                expected_production_commit="b" * 40,
                                construction_constraints=json.dumps(successor_constraints)), "request_digest")
        validate_request(new[part])
        directory = prior / "authoring/parts" / part
        directory.mkdir(parents=True)
        (directory / "request.json").write_text(json.dumps(old[part]))
        records = {}
        for name in ("asset", "final_visual_mesh", "final_visual_mesh_receipt", "physical_review",
                     "physical_review_input", "geometry_readback"):
            path = directory / (name + ".json")
            path.write_text(name)
            records[name] = file_record(path)
        result = sealed({"schema_version": "task_object_astra_authoring_result.v1",
                         "status": "candidate_authored_pending_native_qualification",
                         "request_digest": old[part]["request_digest"], "model": "gpt-6-sol",
                         **records}, "result_digest")
        (directory / "result.json").write_text(json.dumps(result))
        parts[part] = result
        receipt = sealed({"schema_version": "task_asset_agents_api_stage_receipt.v1",
                          "provider": "openai", "model": "gpt-6-sol", "runtime": "openai_agents_api",
                          "run_id": "original-run", "object_id": old[part]["object_id"],
                          "session_cleanup": "deleted", "result_digest": result["result_digest"]}, "receipt_digest")
        receipt_path = prior / "inference/agents_api/parts" / part / "agents_api_stage_receipt.json"
        receipt_path.parent.mkdir(parents=True)
        receipt_path.write_text(json.dumps(receipt))
    source_plan = {"schema_version": "development_drawer_plan.v1", "parts": sorted(parts),
                   "source_geometry_receipt": SOURCE_GEOMETRY}
    plan = copy.deepcopy(source_plan)
    plan["source_geometry_receipt"]["construction_envelope_digest"] = SUCCESSOR_ENVELOPE
    authored = sealed({"schema_version": "task_object_astra_articulated_authoring_result.v1",
                       "status": "parts_authored_pending_native_qualification", "provider": "openai",
                       "agent_runtime": "openai_agents_api", "model": "gpt-6-sol", "plan": source_plan,
                       "parts": parts, "part_request_digests": {
                           part: row["request_digest"] for part, row in old.items()}}, "result_digest")
    (prior / "authoring/result.json").write_text(json.dumps(authored))
    base_binding = {"schema_version": "astra_same_run_source_binding.v1", "run_id": "original-run",
                    "authoring_input_digest": canonical_digest({part: {
                        key: value for key, value in row.items()
                        if key not in {"request_digest", "expected_production_commit"}}
                        for part, row in old.items()}),
                    "source_candidate": META["stage_source_binding"]["source_candidate"],
                    "rights_admission": META["stage_source_binding"]["rights_admission"],
                    "configuration_sha256": META["stage_source_binding"]["configuration_sha256"],
                    "assembly_parts": sorted(parts)}
    binding = sealed(base_binding, "binding_digest")
    (prior / "stage_source_binding.json").write_text(json.dumps(binding))
    current_binding = sealed(dict(binding, run_id="successor-run",
        authoring_input_digest=canonical_digest({part: {
            key: value for key, value in row.items()
            if key not in {"request_digest", "expected_production_commit"}}
            for part, row in new.items()})), "binding_digest")
    lineage = {"source_run_id": "original-run", "successor_run_id": "successor-run",
               "owner_id": "owner", "stable_intent_id": "intent",
               "stable_intent_digest": "sha256:" + "c" * 64}
    descriptor = sealed({"schema_version": SCHEMA_VERSION,
        "adoption_kind": "completed_articulated_agents_api",
        "source_run_id": "original-run", "successor_run_id": "successor-run",
        "original_runtime_root": str(prior),
        "source_request_digest": canonical_digest({part: row["request_digest"] for part, row in old.items()}),
        "semantic_request_digest": canonical_digest(semantic_articulated_requests(new)),
        "source_construction_envelope_digest": SOURCE_ENVELOPE,
        "source_stage_binding_digest": binding["binding_digest"],
        "owner_intent_lineage": lineage, "retained_files": _inventory(prior)}, "adoption_digest")
    return {"value": descriptor, "part_requests": {part: validate_request(row) for part, row in new.items()},
            "plan": plan, "source_binding": current_binding, "verified_lineage": lineage,
            "successor_envelope_digest": SUCCESSOR_ENVELOPE,
            "runtime": successor}, prior


def test_completed_articulated_successor_preserves_source_and_spends_nothing(completed):
    arguments, prior = completed
    before = _inventory(prior)
    arguments["runtime"].joinpath("authoring").mkdir(parents=True)
    prepared = prepare_completed_articulated_successor(**arguments)
    assert set(prepared["source_part_requests"]) == {"carcass", "drawer"}
    assert prepared["lineage"]["new_provider_calls"] == 0
    assert prepared["lineage"]["cad_execution_repeated"] is False
    assert prepared["lineage"]["blender_execution_repeated"] is False
    assert prepared["lineage"]["source_construction_envelope_digest"] == SOURCE_ENVELOPE
    assert prepared["lineage"]["successor_construction_envelope_digest"] == SUCCESSOR_ENVELOPE
    assert _inventory(prior) == before
    for part in ("carcass", "drawer"):
        assert prepared["source_part_requests"][part].request_digest != arguments["part_requests"][part].request_digest
        assert (arguments["runtime"] / "inference/agents_api/parts" / part /
                "agents_api_stage_receipt.json").is_file()


def test_completed_articulated_archive_restores_only_at_original_root(completed, tmp_path):
    import shutil
    import zipfile

    arguments, prior = completed
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for row in arguments["value"]["retained_files"]:
            output.write(prior / row["relative_path"], row["relative_path"])
    arguments["value"]["retained_runtime_archive"] = {
        key: value for key, value in file_record(archive).items() if key != "path"}
    sealed(arguments["value"], "adoption_digest")
    before = arguments["value"]["retained_files"]
    shutil.rmtree(prior)
    restore_partial_astra(value=arguments["value"], request_value={
        "run_id": "successor-run", "part_requests": {
            part: request.model_dump(mode="json") for part, request in arguments["part_requests"].items()}},
        original_root=prior, verified_lineage=arguments["verified_lineage"], archive_path=archive)
    assert _inventory(prior) == before


@pytest.mark.parametrize("change", ["source_envelope", "successor_envelope", "geometry_bounds"])
def test_completed_articulated_successor_checks_envelope_lineage_and_geometry(completed, change):
    arguments, _ = completed
    if change == "source_envelope":
        arguments["value"]["source_construction_envelope_digest"] = SUCCESSOR_ENVELOPE
        sealed(arguments["value"], "adoption_digest")
    elif change == "successor_envelope":
        arguments["successor_envelope_digest"] = SOURCE_ENVELOPE
    else:
        arguments["plan"]["source_geometry_receipt"]["source_aabb_max_xyz_m"] = [2, 1, 1]
    arguments["runtime"].joinpath("authoring").mkdir(parents=True)
    with pytest.raises(AssetAuthoringError):
        prepare_completed_articulated_successor(**arguments)


@pytest.mark.parametrize("change", ["owner", "description", "rights", "part_result", "asset"])
def test_completed_articulated_successor_rejects_changed_inputs_or_bytes(completed, change):
    arguments, prior = completed
    if change == "owner":
        arguments["verified_lineage"] = dict(arguments["verified_lineage"], owner_id="foreign")
    elif change == "description":
        row = arguments["part_requests"]["drawer"].model_dump(mode="json")
        row["owner_description"] += " with an altered source appearance"
        row = sealed(row, "request_digest")
        arguments["part_requests"]["drawer"] = validate_request(row)
    elif change == "rights":
        arguments["source_binding"]["rights_admission"] = {"changed": True}
    elif change == "part_result":
        (prior / "authoring/parts/drawer/result.json").write_text("{}")
        arguments["value"]["retained_files"] = _inventory(prior)
        sealed(arguments["value"], "adoption_digest")
    else:
        (prior / "authoring/parts/drawer/asset.json").write_text("changed")
        arguments["value"]["retained_files"] = _inventory(prior)
        sealed(arguments["value"], "adoption_digest")
    arguments["runtime"].joinpath("authoring").mkdir(parents=True)
    with pytest.raises(AssetAuthoringError):
        prepare_completed_articulated_successor(**arguments)
