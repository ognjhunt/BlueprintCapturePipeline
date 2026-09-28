"""Frozen Astra administrative contracts and same-run completed-stage reuse."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_contract import (
    TaskEvaluationLaunchPreparationContractError, validate_retained_preparation_request,
)
from blueprint_pipeline.task_evaluation_retained_preparation_contract import retained_contract_identity, retained_schema
from blueprint_pipeline.task_evaluation_scene_configuration_provider_runtime import execute_scene_configuration_stage_chain
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError
from tests.test_task_evaluation_scene_configuration_provider_runtime import _inputs, _registry, _producers

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_astra_request_retains_exact_bytes_and_grants_no_new_authority():
    path = ROOT / "tests/fixtures/scene_configuration/astra_preparation_request.v1.json"
    original = path.read_bytes()
    value = json.loads(original)
    assert validate_retained_preparation_request(value) == value
    identity = retained_contract_identity(value)
    assert identity["schema_version"] == "task_evaluation_retained_preparation_contract_identity.v2"
    assert identity["contract_id"] == "scene_preparation_27000_astra.v1"
    assert identity["new_execution_authorized"] is False and identity["new_spend_authorized"] is False
    assert identity["contract_digest"] == canonical_digest(identity, digest_field="contract_digest")
    assert path.read_bytes() == original
    assert "replacement_authoring_backend" not in retained_schema()["properties"]


@pytest.mark.parametrize("mutation", ["backend", "total", "author_minimum", "author_maximum", "review", "rights"])
def test_retained_astra_contract_does_not_relax_spend_or_rights(mutation):
    value = json.loads((ROOT / "tests/fixtures/scene_configuration/astra_preparation_request.v1.json").read_text())
    if mutation == "backend":
        value.pop("replacement_authoring_backend")
    elif mutation == "total":
        value["spend"]["hard_cap_usd"] = 26.77
    elif mutation.startswith("author"):
        value["spend"]["external_service_caps"]["openai"]["stage_max_cost_usd"]["content_agents"] = 4.99 if mutation.endswith("minimum") else 15.01
    elif mutation == "review":
        value["spend"]["external_service_caps"]["openai"]["stage_max_cost_usd"]["artifixer_visual_review"] = .64
    else:
        value["scene"]["rights"]["provider_disclosure_scope"] = "source_and_derived"
        value["scene"]["rights"]["source_bytes_redistributable"] = False
    with pytest.raises(TaskEvaluationLaunchPreparationContractError):
        validate_retained_preparation_request(value)


def _astra_inputs(tmp_path):
    envelope, configurations = _inputs(tmp_path)
    envelope["expected_production_commit"] = "a" * 40
    config, path = configurations["stage-3"]
    config["authoring_backend"] = "astra_cad_blender_v1"
    path.write_text(json.dumps(config))
    root = tmp_path / "output"
    root.mkdir()
    return envelope, configurations, root


def test_provider_resume_adopts_completed_prefix_and_runs_only_failed_astra_onward(tmp_path):
    envelope, configs, root = _astra_inputs(tmp_path)
    observed, produced = [], []
    producer = _producers()
    def execute(**kwargs):
        stage = kwargs["stage"]["stage_id"]
        produced.append(stage)
        if stage == "stage-3" and produced.count(stage) == 1:
            raise RuntimeError("interrupted Astra producer")
        return producer.execute(**kwargs)
    arguments = dict(envelope=envelope, configurations=configs, output_root=root,
        registry=_registry(observed), producer_registry=SimpleNamespace(execute=execute))
    with pytest.raises(RuntimeError, match="interrupted"):
        execute_scene_configuration_stage_chain(**arguments)
    first = (root / "stage-1/completed_stage_checkpoint.json").read_bytes()
    result = execute_scene_configuration_stage_chain(**arguments)
    assert result["status"] == "completed"
    assert observed == [f"stage-{n}" for n in range(1, 7)]
    assert produced == ["stage-1", "stage-3", "stage-3", "stage-5"]
    assert (root / "stage-1/completed_stage_checkpoint.json").read_bytes() == first
    assert execute_scene_configuration_stage_chain(**arguments) == result
    assert len(produced) == 4


def test_provider_resume_refuses_changed_input_or_extended_deadline(tmp_path):
    envelope, configs, root = _astra_inputs(tmp_path)
    def interrupt(**kwargs):
        raise RuntimeError("fixture interruption")
    args = dict(envelope=envelope, configurations=configs, output_root=root,
        producer_registry=SimpleNamespace(execute=interrupt), parent_deadline_epoch=100000., clock=lambda: 0)
    with pytest.raises(RuntimeError, match="fixture"):
        execute_scene_configuration_stage_chain(**args)
    for changes in ({"parent_deadline_epoch": 100001.}, {"envelope": {**deepcopy(envelope), "run_id": "different"}}):
        with pytest.raises(AssetAuthoringError, match="immutable_binding_changed"):
            execute_scene_configuration_stage_chain(**{**args, **changes})
