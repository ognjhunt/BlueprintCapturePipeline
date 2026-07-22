from __future__ import annotations

import json

import pytest

from blueprint_pipeline.artifact_contracts import (
    ArtifactContractError,
    SELLABLE_ARTIFACT_CONTRACTS,
    sellable_artifact_json_schemas,
    validate_sellable_artifact,
    main,
)


def test_sellable_contracts_export_json_schema() -> None:
    schemas = sellable_artifact_json_schemas()
    assert set(schemas) == set(SELLABLE_ARTIFACT_CONTRACTS)
    for name, schema in schemas.items():
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"] == name
        assert "schema_version" in schema["required"]
        assert schema["additionalProperties"] is True


def test_sellable_contracts_write_deterministic_schema_bundle(
    tmp_path,
    capsys,
) -> None:
    output = tmp_path / "schemas" / "sellable-artifacts.json"
    assert main(["--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sellable_artifact_json_schema_bundle.v1"
    assert set(payload["schemas"]) == set(SELLABLE_ARTIFACT_CONTRACTS)
    assert str(output.resolve()) in capsys.readouterr().out


def test_card_collection_rejects_wrong_nested_schema() -> None:
    with pytest.raises(
        ArtifactContractError,
        match=r"cards\[0\].schema_version:must_be:real_site_robot_eval_task_card.v0.1",
    ):
        validate_sellable_artifact(
            "task_cards",
            {
                "schema_version": "real_site_robot_eval_task_cards.v0.1",
                "task_card_count": 1,
                "cards": [{"schema_version": "wrong"}],
            },
        )


def test_post_training_manifest_rejects_missing_claim_boundary() -> None:
    with pytest.raises(
        ArtifactContractError,
        match="claim_boundary:must_be_mapping",
    ):
        validate_sellable_artifact(
            "post_training_data_package_export",
            {
                "schema_version": "post_training_data_package_export.v1",
                "status": "blocked",
                "blockers": [],
            },
        )


def test_evaluation_run_requires_all_replaceable_components() -> None:
    with pytest.raises(ArtifactContractError, match="proof_contract:must_be_mapping"):
        validate_sellable_artifact(
            "evaluation_run",
            {
                "schema_version": "evaluation_run.v1",
                "run_id": "run-1",
                "mode": "evaluate",
                "scene_bundle": {},
                "robot_adapter": {},
                "task_scenario_pack": {},
                "policy_adapter": {},
                "runtime_provider_profile": {},
            },
        )
