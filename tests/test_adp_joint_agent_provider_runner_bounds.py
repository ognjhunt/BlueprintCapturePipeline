"""The GPU runner must accept both documents that can carry link bounds.

The Joint Agent runner verifies the bounds file it was shipped by recomputing
that file's self-digest.  It only ever knew one document: a source-mesh receipt,
which seals itself with ``receipt_digest``.

Feeding the agent our authored replacement changes which document travels.  An
authored-link components document seals itself with ``components_digest``, so a
runner that checks only ``receipt_digest`` reads a perfectly valid input as
tampered -- and it does so on the GPU, after the instance is running and
billing, which is the most expensive place in the system to discover a naming
mismatch.

This pins both shapes against the digest rule the runner actually applies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest


RUNNER = (
    Path(__file__).resolve().parents[1] / "scripts" / "adp_joint_agent_provider_runner.py"
)

COMPONENTS_SCHEMA = "authored_link_source_components.v1"


def _seal(payload: dict, field: str) -> dict:
    payload[field] = ""
    payload[field] = canonical_digest(payload, digest_field=field)
    return payload


def _mesh_receipt() -> dict:
    return _seal(
        {
            "schema_version": "articulated_source_asset.v1",
            "connected_components": [
                {
                    "component_index": 0,
                    "aabb_min_asset_m": [0.0, 0.0, 0.0],
                    "aabb_max_asset_m": [1.0, 1.0, 1.0],
                }
            ],
            "receipt_digest": "",
        },
        "receipt_digest",
    )


def _components_document() -> dict:
    return _seal(
        {
            "schema_version": COMPONENTS_SCHEMA,
            "status": "authored_link_bounds_synthesized",
            "connected_components": [
                {
                    "component_index": 0,
                    "link_id": "door",
                    "aabb_min_asset_m": [0.0, 0.0, 0.0],
                    "aabb_max_asset_m": [1.0, 1.0, 1.0],
                }
            ],
            "components_digest": "",
        },
        "components_digest",
    )


def _runner_digest_field(document: dict) -> str:
    """The field the runner selects, mirrored from its source."""

    source = RUNNER.read_text(encoding="utf-8")
    assert '"components_digest"' in source and '"receipt_digest"' in source, (
        "the runner must still choose between the two self-digest fields"
    )
    assert f'== "{COMPONENTS_SCHEMA}"' in source, (
        "the runner must select on the authored-link components schema"
    )
    return (
        "components_digest"
        if document.get("schema_version") == COMPONENTS_SCHEMA
        else "receipt_digest"
    )


@pytest.mark.parametrize(
    "document, expected_field",
    [
        (_mesh_receipt(), "receipt_digest"),
        (_components_document(), "components_digest"),
    ],
)
def test_both_bounds_documents_verify_under_their_own_digest_field(
    document: dict, expected_field: str
) -> None:
    field = _runner_digest_field(document)
    assert field == expected_field
    assert canonical_digest(document, digest_field=field) == document[field]


def test_an_authored_components_document_has_no_receipt_digest_to_check() -> None:
    """Which is exactly why checking only `receipt_digest` refuses it."""

    document = _components_document()
    assert "receipt_digest" not in document
    assert canonical_digest(document, digest_field="receipt_digest") != document.get(
        "receipt_digest"
    )


def test_a_tampered_components_document_still_fails(tmp_path: Path) -> None:
    """Accepting a second digest field must not weaken the check itself."""

    document = _components_document()
    document["connected_components"][0]["aabb_max_asset_m"] = [9.0, 9.0, 9.0]
    path = tmp_path / "articulated_source_receipt.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    field = _runner_digest_field(reloaded)
    assert canonical_digest(reloaded, digest_field=field) != reloaded[field]
