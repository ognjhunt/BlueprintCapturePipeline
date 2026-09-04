from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_adapters import (
    TaskEvaluationSceneConfigurationAdapterError,
)
from blueprint_pipeline.task_evaluation_scene_configuration_supplemental_destination import (
    supplemental_destination_inputs,
    supplemental_destination_static_artifacts,
)

from tests.test_task_evaluation_scene_configuration_builtin_adapters import (
    DESTINATION_IDENTITY,
    sha256,
    supplemental_destination_inputs as _fixture_inputs,
)


pytest.importorskip("pxr")


def _envelope(destination: dict) -> dict:
    return {
        "recipe": {
            "subject_identity": {"id": "replacement-mug", "version": "v1"},
            "supplemental_destination": destination["recipe_supplemental_destination"],
        },
        "materialized_references": destination["materialized_references"],
    }


def test_no_destination_declared_means_no_destination_artifacts(tmp_path: Path) -> None:
    assert supplemental_destination_inputs({"recipe": {}}) is None
    assert supplemental_destination_static_artifacts(
        envelope={"recipe": {}}, output_root=tmp_path
    ) == []


def test_inputs_cross_bind_every_declared_destination_reference(tmp_path: Path) -> None:
    destination = _fixture_inputs(tmp_path / "destination")
    inputs = supplemental_destination_inputs(_envelope(destination))
    assert inputs is not None
    assert inputs["identity"] == DESTINATION_IDENTITY
    assert inputs["asset_digest"] == sha256(destination["asset"])
    assert inputs["simready_result"]["intended_support_prim_paths"] == ["/Asset/Colliders/Bottom"]


def test_inputs_refuse_a_recipe_reference_that_names_different_bytes(tmp_path: Path) -> None:
    destination = _fixture_inputs(tmp_path / "destination")
    destination["recipe_supplemental_destination"]["asset"]["digest"] = "sha256:" + "7" * 64
    with pytest.raises(
        TaskEvaluationSceneConfigurationAdapterError,
        match="simready_supplemental_destination_recipe_binding_invalid:asset",
    ):
        supplemental_destination_inputs(_envelope(destination))


def test_inputs_refuse_a_missing_materialized_reference(tmp_path: Path) -> None:
    destination = _fixture_inputs(tmp_path / "destination")
    destination["materialized_references"] = [
        row
        for row in destination["materialized_references"]
        if row["contract_path"] != "construction.recipe.supplemental_destination.simready_result"
    ]
    with pytest.raises(
        TaskEvaluationSceneConfigurationAdapterError,
        match="scene_configuration_materialized_reference_missing:construction.recipe.supplemental_destination.simready_result",
    ):
        supplemental_destination_inputs(_envelope(destination))


def test_inputs_refuse_a_support_prim_that_is_not_a_declared_collider(tmp_path: Path) -> None:
    destination = _fixture_inputs(tmp_path / "destination")
    simready = json.loads(destination["simready_result"].read_text())
    simready["intended_support_prim_paths"] = ["/Asset/Colliders/Lid"]
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    simready["result_digest"] = canonical_digest(simready, digest_field="result_digest")
    destination["simready_result"].write_text(json.dumps(simready, sort_keys=True))
    for row in destination["materialized_references"]:
        if row["contract_path"].endswith("simready_result"):
            row["digest"] = sha256(destination["simready_result"])
            row["size_bytes"] = destination["simready_result"].stat().st_size
    destination["recipe_supplemental_destination"]["simready_result"].update(
        digest=sha256(destination["simready_result"]),
        size_bytes=destination["simready_result"].stat().st_size,
    )
    with pytest.raises(
        TaskEvaluationSceneConfigurationAdapterError,
        match="simready_supplemental_destination_binding_invalid",
    ):
        supplemental_destination_inputs(_envelope(destination))
