from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_suite_index import (
    REQUIRED_ROLE_PROJECTS,
    build_public_scene_suite_index_receipt,
)


SCHEMA_PATH = (
    Path(__file__).parents[1]
    / "docs"
    / "schemas"
    / "public_scene_suite_index.v1.schema.json"
)
EVALUATED_ON = dt.date(2026, 8, 4)


def _component(
    role: str,
    source_project_id: str,
    *,
    digest_character: str,
    revision: dict | None = None,
    status: str = "admitted",
    blockers: list[str] | None = None,
) -> dict:
    return {
        "role": role,
        "source_project_id": source_project_id,
        "component_manifest_digest": "sha256:" + digest_character * 64,
        "component_admission_receipt_digest": "sha256:" + digest_character * 64,
        "exact_revision": revision
        or {"kind": "git_commit", "value": digest_character * 40},
        "exact_artifact_digest": "sha256:" + digest_character * 64,
        "status": status,
        "blockers": list(blockers or []),
    }


def _index() -> dict:
    value = {
        "schema_version": "public_scene_suite_index.v1",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009",
        "index_id": "adp009-exact-public-suite-v1",
        "components": [
            _component(
                "inpaint360_author_smoke",
                "Inpaint360GS",
                digest_character="1",
                revision={
                    "kind": "git_commit",
                    "value": "d54c893285c6cb27788e05cce607e7d3cca6388a",
                },
            ),
            _component(
                "infusion_primary_adapter",
                "InFusion",
                digest_character="2",
                revision={
                    "kind": "git_commit",
                    "value": "788da7f40cad4314831a053b7419df277d7814c4",
                },
            ),
            _component(
                "aurafusion360_quality_challenger",
                "AuraFusion360",
                digest_character="3",
                revision={
                    "kind": "git_commit",
                    "value": "f23b26c44ba84608306ba952510533ebf4c7877d",
                },
            ),
            _component(
                "interiorgs_appearance_scene",
                "InteriorGS",
                digest_character="4",
                revision={"kind": "release_tag", "value": "dataset-v1.0"},
            ),
            _component(
                "sage3d_collision_companion",
                "SAGE-3D",
                digest_character="5",
                revision={"kind": "content_digest", "value": "sha256:" + "5" * 64},
            ),
            _component(
                "controlled_background_truth",
                "Blueprint-controlled",
                digest_character="6",
                revision={"kind": "content_digest", "value": "sha256:" + "6" * 64},
            ),
            _component(
                "exact_simready_object",
                "Blueprint-controlled",
                digest_character="7",
                revision={"kind": "content_digest", "value": "sha256:" + "7" * 64},
            ),
            _component(
                "usd_content_agents_candidate",
                "NVIDIA-Omniverse/usd-content-agents",
                digest_character="a",
                revision={"kind": "release_tag", "value": "v0.5.2"},
            ),
            _component(
                "physics_positive_control",
                "Blueprint-controlled",
                digest_character="8",
                revision={"kind": "content_digest", "value": "sha256:" + "8" * 64},
            ),
            _component(
                "scannetpp_real_transfer",
                "ScanNet++",
                digest_character="9",
                revision={"kind": "release_tag", "value": "dataset-v2.0"},
            ),
        ],
        "claim_ceiling": "development_only",
        "claim_boundaries": {
            "exact_public_suite_binding": True,
            "public_scene_software_qualified": False,
            "metric_geometry_qualified": False,
            "task_physics_qualified": False,
            "partner_capture_qualified": False,
            "prospective_validation": False,
            "physical_evidence": False,
            "digital_twin": False,
            "deployment_readiness": False,
            "physical_safety": False,
            "customer_value": False,
            "general_sim_to_real_fidelity": False,
        },
    }
    value["index_digest"] = canonical_digest(value, digest_field="index_digest")
    return value


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _redigest(value: dict) -> None:
    value["index_digest"] = canonical_digest(value, digest_field="index_digest")


def _receipt(value: dict) -> dict:
    return build_public_scene_suite_index_receipt(value, evaluated_on=EVALUATED_ON)


def _component_for_role(value: dict, role: str) -> dict:
    return next(row for row in value["components"] if row["role"] == role)


def test_exact_admitted_matrix_is_schema_valid_and_complete() -> None:
    value = _index()
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(value)

    receipt = _receipt(value)

    assert receipt["status"] == "matrix_complete"
    assert receipt["blockers"] == []
    assert receipt["adp009_matrix_complete"] is True
    assert receipt["required_role_count"] == 10
    assert receipt["declared_component_count"] == 10
    assert receipt["admitted_role_count"] == 10
    assert receipt["blocked_roles"] == []
    assert {row["role"] for row in receipt["role_bindings"]} == set(
        REQUIRED_ROLE_PROJECTS
    )
    assert receipt["claim_ceiling"] == "development_only"
    assert receipt["artifact_bytes_opened"] is False
    assert receipt["artifact_bytes_verified"] is False
    assert receipt["public_scene_software_qualified"] is False
    assert receipt["metric_geometry_qualified"] is False
    assert receipt["task_physics_qualified"] is False
    assert receipt["partner_capture_qualified"] is False
    assert receipt["prospective_validation"] is False
    assert receipt["physical_evidence_created"] is False
    assert receipt["deployment_readiness"] is False
    assert receipt["customer_value"] is False
    assert receipt == _receipt(value)
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_blocked_component_is_schema_valid_but_matrix_is_incomplete() -> None:
    value = _index()
    transfer = _component_for_role(value, "scannetpp_real_transfer")
    transfer["status"] = "blocked"
    transfer["blockers"] = ["dataset_access_not_admitted"]
    _redigest(value)

    jsonschema.Draft202012Validator(_schema()).validate(value)
    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert receipt["adp009_matrix_complete"] is False
    assert receipt["admitted_role_count"] == 9
    assert receipt["blocked_roles"] == ["scannetpp_real_transfer"]
    assert "components[9].status:blocked" in receipt["blockers"]


@pytest.mark.parametrize(
    ("role", "substitute"),
    [
        ("interiorgs_appearance_scene", "ARKitScenes"),
        ("scannetpp_real_transfer", "WildRGB-D"),
        ("interiorgs_appearance_scene", "Blueprint-controlled"),
        ("sage3d_collision_companion", "Blueprint-controlled"),
        ("scannetpp_real_transfer", "Blueprint-controlled"),
        ("usd_content_agents_candidate", "SimReadyGen"),
    ],
)
def test_dataset_and_authored_substitutions_fail_closed(
    role: str, substitute: str
) -> None:
    value = _index()
    _component_for_role(value, role)["source_project_id"] = substitute
    _redigest(value)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_schema()).validate(value)

    receipt = _receipt(value)
    expected_project = REQUIRED_ROLE_PROJECTS[role]
    assert receipt["status"] == "blocked"
    assert receipt["adp009_matrix_complete"] is False
    assert any(
        blocker.endswith(f"source_project_id:must_be:{expected_project}")
        for blocker in receipt["blockers"]
    )


def test_duplicate_role_and_missing_role_are_both_blocked() -> None:
    value = _index()
    duplicate = value["components"][-1]
    duplicate["role"] = "interiorgs_appearance_scene"
    duplicate["source_project_id"] = "InteriorGS"
    _redigest(value)

    # JSON Schema checks each role/project pair, while the runtime boundary
    # enforces exact cardinality across the whole index.
    jsonschema.Draft202012Validator(_schema()).validate(value)
    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert "components:duplicate_role:interiorgs_appearance_scene" in receipt["blockers"]
    assert "components:missing_role:scannetpp_real_transfer" in receipt["blockers"]


def test_digest_tamper_is_blocked_even_when_changed_digest_is_well_formed() -> None:
    value = _index()
    value["components"][0]["exact_artifact_digest"] = "sha256:" + "a" * 64

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert receipt["adp009_matrix_complete"] is False
    assert "index_digest:mismatch" in receipt["blockers"]
    assert receipt["index_digest"] != receipt["supplied_index_digest"]


def test_component_admission_receipt_digest_is_mandatory() -> None:
    value = _index()
    value["components"][0]["component_admission_receipt_digest"] = None
    _redigest(value)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_schema()).validate(value)
    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "components[0].component_admission_receipt_digest:invalid"
        in receipt["blockers"]
    )


@pytest.mark.parametrize("location", ["top", "component", "revision", "boundary"])
def test_unknown_fields_are_rejected_by_schema_and_runtime(location: str) -> None:
    value = _index()
    if location == "top":
        value["unexpected"] = True
        expected = "unexpected:unknown_property"
    elif location == "component":
        value["components"][0]["unexpected"] = True
        expected = "components[0].unexpected:unknown_property"
    elif location == "revision":
        value["components"][0]["exact_revision"]["branch"] = "main"
        expected = "components[0].exact_revision.branch:unknown_property"
    else:
        value["claim_boundaries"]["physical_trial_qualified"] = True
        expected = "claim_boundaries.physical_trial_qualified:unknown_property"
    _redigest(value)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_schema()).validate(value)

    receipt = _receipt(value)
    assert receipt["status"] == "blocked"
    assert expected in receipt["blockers"]


def test_admitted_component_cannot_hide_blockers() -> None:
    value = _index()
    value["components"][0]["blockers"] = ["license_review_pending"]
    _redigest(value)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_schema()).validate(value)
    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert "components[0].blockers:admitted_must_be_empty" in receipt["blockers"]


def test_input_is_not_mutated() -> None:
    value = _index()
    before = copy.deepcopy(value)

    _receipt(value)

    assert value == before
