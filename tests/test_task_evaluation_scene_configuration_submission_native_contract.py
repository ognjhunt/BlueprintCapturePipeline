"""Submission-to-native adapter contract rehearsal; no simulator or provider runs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import (
    TaskEvaluationRigidRelocationNativeAdapterError,
    adapt_rigid_relocation_task_template,
)
from tests.test_task_evaluation_rigid_relocation_native_adapter import (
    DEFINITION,
    EXECUTION,
    NATIVE_IMPORT,
    SOURCE_OBJECT,
    STATIC,
    SUCCESS,
    SUPPORT,
    _case,
    _documents,
    _rewrite,
)
from tests.test_task_evaluation_scene_configuration_submission import (
    SHA,
    _materialize,
    production_fixture,
)


def _assembled_native_case(tmp_path: Path, task_mutator=None):
    fixture = production_fixture(tmp_path)
    if task_mutator is not None:
        task = json.loads(fixture["task_request"].read_text())
        task_mutator(task)
        fixture["task_request"].write_text(json.dumps(task))
    result = _materialize(fixture)
    staging = Path(result["staging_root"])
    assembled = json.loads(
        (staging / "scene_configuration_preparation_request.v1.json").read_text()
    )
    adapter_root = tmp_path / "adapter"
    adapter_root.mkdir()
    launch, configured, references, _ = _case(adapter_root)
    task_identity = assembled["task"]["identity"]
    subject_identity = assembled["task"]["subject"]["identity"]
    configured["team_namespace"] = assembled["team_namespace"]
    launch["team_namespace"] = assembled["team_namespace"]
    configured["source_commit"] = SHA
    configured["scene_identity"] = assembled["scene"]["identity"]
    configured["task_template"]["identity"] = task_identity
    configured["replacement"]["identity"] = subject_identity
    launch["expected_production_commit"] = SHA
    launch["scene"]["identity"] = configured["scene_identity"]
    launch["task"]["identity"] = task_identity
    launch["task"]["subject"]["identity"] = subject_identity
    launch["task"]["strategy"] = "pick_and_place"
    destination = copy.deepcopy(assembled["task"]["destination"])
    destination.pop("native_probe")
    # These synthetic qualifications represent later stage outputs, not claims
    # about the real scene. This test exercises only the pure native adapter.
    for field in ("native_import_qualification", "geometry", "placement_qualification"):
        destination[field] = copy.deepcopy(destination["static_qualification"])
    launch["task"]["destination"] = destination
    docs = _documents(task_identity, subject_identity)
    for contract, relative in (
        (DEFINITION, "task_template.v1.json"),
        (SUCCESS, "task_success_criteria.v1.json"),
        (EXECUTION, "task_execution_spec.v1.json"),
        (SUPPORT, "support_plane_input.v1.json"),
        (SOURCE_OBJECT, "source_object_selection.v1.json"),
    ):
        docs[contract] = json.loads((staging / "configuration" / relative).read_text())
    half_height = (
        docs[SOURCE_OBJECT]["aabb_max_xyz_m"][2]
        - docs[SOURCE_OBJECT]["aabb_min_xyz_m"][2]
    ) / 2.0
    docs[STATIC]["observed_structure"]["center_of_mass_m"] = [0.0, 0.0, half_height]
    docs[STATIC]["result_digest"] = canonical_digest(
        docs[STATIC], digest_field="result_digest"
    )
    for contract, document in docs.items():
        _rewrite(
            tmp_path=adapter_root, configured=configured, references=references,
            contract_path=contract, document=document,
        )
    launch["task"]["configured_scene_revision_digest"] = configured["revision_digest"]
    return launch, configured, references, docs, adapter_root


def test_assembled_task_reaches_native_adapter_with_exact_instruction_and_seed(
    tmp_path: Path,
) -> None:
    launch, configured, references, docs, _ = _assembled_native_case(tmp_path)
    result = adapt_rigid_relocation_task_template(
        request=launch, configured_revision=configured,
        materialized_references=references,
    )
    native = result["native_task_definition"]["task_spec"]
    assert native["prompt"] == docs[DEFINITION]["instruction"]
    assert "open book" in native["prompt"] and "blue document tray" in native["prompt"]
    assert native["control_frequency_hz"] == 15
    assert native["minimum_lift_m"] == docs[DEFINITION]["interaction_affordance"]["minimum_lift_m"]
    assert native["configured_success_criteria"]["owner_success_contract_required"] is True
    assert result["native_episode_execution"]["scenario"]["seed"] == 1
    assert result["source_documents"]["documents"]["native_import_qualification"] == docs[NATIVE_IMPORT]


@pytest.mark.parametrize(
    "contract,field,value,error",
    [
        (DEFINITION, "status", "candidate_pending_scene_construction", "identity_or_strategy_mismatch"),
        (EXECUTION, "resolved_seed", 0, "invalid:resolved_seed"),
    ],
)
def test_native_adapter_refuses_old_submission_status_or_zero_seed(
    tmp_path: Path, contract: str, field: str, value, error: str,
) -> None:
    launch, configured, references, docs, adapter_root = _assembled_native_case(tmp_path)
    document = copy.deepcopy(docs[contract])
    document[field] = value
    _rewrite(
        tmp_path=adapter_root, configured=configured, references=references,
        contract_path=contract, document=document,
    )
    launch["task"]["configured_scene_revision_digest"] = configured["revision_digest"]
    with pytest.raises(TaskEvaluationRigidRelocationNativeAdapterError, match=error):
        adapt_rigid_relocation_task_template(
            request=launch, configured_revision=configured,
            materialized_references=references,
        )
