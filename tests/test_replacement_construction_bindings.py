from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.replacement_construction_bindings import (
    ReplacementConstructionBindingsError,
    seal_replacement_construction_bindings,
    validate_replacement_construction_bindings,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _row(suffix: str, digest_characters: str) -> dict:
    values = iter(digest_characters)
    return {
        "task_id": f"task_{suffix}",
        "asset_id": f"replacement_{suffix}",
        "task_freeze_digest": _sha(next(values)),
        "source_object_instance_id": f"source_{suffix}",
        "removal_id": f"removal_{suffix}",
        "mask_set_id": f"masks_{suffix}",
        "mask_set_receipt_digest": _sha(next(values)),
        "source_removal_receipt_digest": _sha(next(values)),
        "source_removal_qualified": True,
        "collider_deletion_id": f"collider_{suffix}",
        "source_collider_prim_path": f"/Root/source_{suffix}",
        "collider_deletion_receipt_digest": _sha(next(values)),
        "collider_deletion_qualified": True,
        "replacement_qualification_id": f"qualification_{suffix}",
        "replacement_qualification_receipt_digest": _sha(next(values)),
        "replacement_asset_sha256": _sha(next(values)),
        "replacement_simulator_import_qualified": True,
    }


def _sealed() -> dict:
    return seal_replacement_construction_bindings(
        scene_freeze_digest=_sha("1"),
        task_freeze_join_digest=_sha("2"),
        bindings=[_row("a", "345678"), _row("b", "9abcde")],
    )


def test_two_independent_removal_replacement_lanes_seal() -> None:
    sealed = _sealed()

    assert validate_replacement_construction_bindings(sealed) == sealed
    assert [row["asset_id"] for row in sealed["bindings"]] == [
        "replacement_a",
        "replacement_b",
    ]


@pytest.mark.parametrize(
    "field",
    [
        "mask_set_id",
        "source_removal_receipt_digest",
        "collider_deletion_id",
        "collider_deletion_receipt_digest",
        "replacement_qualification_id",
        "replacement_qualification_receipt_digest",
        "replacement_asset_sha256",
    ],
)
def test_shared_removal_or_replacement_identity_is_rejected(field: str) -> None:
    sealed = _sealed()
    sealed["bindings"][1][field] = sealed["bindings"][0][field]
    sealed["construction_digest"] = canonical_digest(
        sealed, digest_field="construction_digest"
    )

    with pytest.raises(ReplacementConstructionBindingsError) as excinfo:
        validate_replacement_construction_bindings(sealed)

    assert f"replacement_construction_shared_identity:{field}" in excinfo.value.errors


def test_unqualified_receipt_and_digest_mutation_fail_closed() -> None:
    sealed = _sealed()
    sealed["bindings"][0]["source_removal_qualified"] = False

    with pytest.raises(ReplacementConstructionBindingsError) as excinfo:
        validate_replacement_construction_bindings(sealed)

    assert (
        "replacement_construction_qualification_missing:0:source_removal_qualified"
        in excinfo.value.errors
    )
    assert "replacement_construction_digest_invalid" in excinfo.value.errors


def test_swapped_task_freeze_binding_changes_construction_seal() -> None:
    sealed = _sealed()
    swapped = copy.deepcopy(sealed)
    swapped["bindings"][0]["task_freeze_digest"], swapped["bindings"][1][
        "task_freeze_digest"
    ] = (
        swapped["bindings"][1]["task_freeze_digest"],
        swapped["bindings"][0]["task_freeze_digest"],
    )

    with pytest.raises(ReplacementConstructionBindingsError) as excinfo:
        validate_replacement_construction_bindings(swapped)

    assert excinfo.value.errors == ("replacement_construction_digest_invalid",)
