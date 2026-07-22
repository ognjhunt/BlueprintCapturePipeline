"""Typed, machine-readable contracts for sellable pipeline boundaries."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ArtifactContractError(ValueError):
    """Raised before a malformed sellable artifact crosses a stage boundary."""


@dataclass(frozen=True)
class ArtifactContract:
    artifact_type: str
    schema_version: str
    required_fields: tuple[str, ...]
    required_mapping_fields: tuple[str, ...] = ()
    required_list_fields: tuple[str, ...] = ()
    item_schema_version: str | None = None

    def validate(self, value: Mapping[str, Any]) -> None:
        errors: list[str] = []
        if value.get("schema_version") != self.schema_version:
            errors.append(f"schema_version:must_be:{self.schema_version}")
        for field in self.required_fields:
            field_value = value.get(field)
            if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
                errors.append(f"{field}:missing")
        for field in self.required_mapping_fields:
            if not isinstance(value.get(field), Mapping):
                errors.append(f"{field}:must_be_mapping")
        for field in self.required_list_fields:
            rows = value.get(field)
            if not isinstance(rows, list):
                errors.append(f"{field}:must_be_list")
                continue
            if self.item_schema_version:
                for index, row in enumerate(rows):
                    if not isinstance(row, Mapping):
                        errors.append(f"{field}[{index}]:must_be_mapping")
                    elif row.get("schema_version") != self.item_schema_version:
                        errors.append(
                            f"{field}[{index}].schema_version:must_be:"
                            f"{self.item_schema_version}"
                        )
        if errors:
            raise ArtifactContractError(
                f"{self.artifact_type}_contract_invalid:" + ",".join(errors)
            )

    def json_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "schema_version": {"const": self.schema_version},
        }
        properties.update({field: {} for field in self.required_fields})
        properties.update(
            {field: {"type": "object"} for field in self.required_mapping_fields}
        )
        for field in self.required_list_fields:
            item_schema: dict[str, Any] = {"type": "object"}
            if self.item_schema_version:
                item_schema["properties"] = {
                    "schema_version": {"const": self.item_schema_version}
                }
                item_schema["required"] = ["schema_version"]
            properties[field] = {"type": "array", "items": item_schema}
        required = [
            "schema_version",
            *self.required_fields,
            *self.required_mapping_fields,
            *self.required_list_fields,
        ]
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://blueprint.example/schemas/{self.artifact_type}.schema.json",
            "title": self.artifact_type,
            "type": "object",
            "properties": properties,
            "required": list(dict.fromkeys(required)),
            "additionalProperties": True,
        }


SELLABLE_ARTIFACT_CONTRACTS = {
    "site_card": ArtifactContract(
        artifact_type="site_card",
        schema_version="real_site_robot_eval_site_card.v0.1",
        required_fields=("site_card_id", "scene_id", "capture_id"),
        required_mapping_fields=("geometry", "provenance_rights_review_status"),
    ),
    "task_cards": ArtifactContract(
        artifact_type="task_cards",
        schema_version="real_site_robot_eval_task_cards.v0.1",
        required_fields=("task_card_count",),
        required_list_fields=("cards",),
        item_schema_version="real_site_robot_eval_task_card.v0.1",
    ),
    "scenario_cards": ArtifactContract(
        artifact_type="scenario_cards",
        schema_version="real_site_robot_eval_scenario_cards.v0.1",
        required_fields=("scenario_card_count",),
        required_list_fields=("cards",),
        item_schema_version="real_site_robot_eval_scenario_card.v0.1",
    ),
    "eval_cards": ArtifactContract(
        artifact_type="eval_cards",
        schema_version="real_site_robot_eval_eval_cards.v0.1",
        required_fields=("eval_card_count",),
        required_list_fields=("cards",),
        item_schema_version="real_site_robot_eval_eval_card.v0.1",
    ),
    "evaluation_run": ArtifactContract(
        artifact_type="evaluation_run",
        schema_version="evaluation_run.v1",
        required_fields=("run_id", "mode"),
        required_mapping_fields=(
            "scene_bundle",
            "robot_adapter",
            "task_scenario_pack",
            "policy_adapter",
            "runtime_provider_profile",
            "proof_contract",
        ),
    ),
    "post_training_data_package_export": ArtifactContract(
        artifact_type="post_training_data_package_export",
        schema_version="post_training_data_package_export.v1",
        required_fields=("status",),
        required_mapping_fields=("claim_boundary",),
        required_list_fields=("blockers",),
    ),
}


def validate_sellable_artifact(
    artifact_type: str,
    value: Mapping[str, Any],
) -> None:
    try:
        contract = SELLABLE_ARTIFACT_CONTRACTS[artifact_type]
    except KeyError as exc:
        raise ArtifactContractError(f"unknown_artifact_contract:{artifact_type}") from exc
    contract.validate(value)


def sellable_artifact_json_schemas() -> dict[str, dict[str, Any]]:
    return {
        name: contract.json_schema()
        for name, contract in sorted(SELLABLE_ARTIFACT_CONTRACTS.items())
    }


def write_sellable_artifact_json_schemas(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "sellable_artifact_json_schema_bundle.v1",
        "schemas": sellable_artifact_json_schemas(),
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = write_sellable_artifact_json_schemas(args.output.expanduser().resolve())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
