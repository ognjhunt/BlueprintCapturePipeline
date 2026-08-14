"""A stage may not name a producer that does not exist.

`fresh_scene_paired_target_preparation.STAGE_CONTRACTS` is the chain a fresh
scene walks to become an ArtiFixer3D input set, and each stage names the module
that produces it. Two stages -- `segment_sweep_freezes` and `segment_cutout_set`
-- name `blueprint_pipeline.public_scene_segment_contribution_cutout`, which is
not in the tree.

That is the whole reason a fresh scene cannot be prepared by code today. The
chain reports it honestly (`fresh_scene_segment_cutout_set_missing`, at the
right stage rather than several steps downstream), but a reader checking
"is there a producer for every stage" sees thirteen `implementation` strings
and reasonably concludes yes.

An import is the cheapest possible proof that a named producer is real, and it
rediscovers the set from `STAGE_CONTRACTS` so a stage added later cannot ship
with a dangling implementation.
"""

from __future__ import annotations

import importlib

import pytest

from blueprint_pipeline.fresh_scene_paired_target_preparation import STAGE_CONTRACTS

#: Stages whose producer is known-missing, with the work item that closes it.
#: A stage listed here still fails the "declared" check below, so the gap
#: cannot be widened silently -- only an entry removed when the module lands.
KNOWN_MISSING_PRODUCERS: dict[str, str] = {
    "blueprint_pipeline.public_scene_segment_contribution_cutout": (
        "the deterministic exact-index standard-3DGS subset writer that emits "
        "adp009d_segment_contribution_cutout_set.v1; nothing in the tree "
        "produces a segment cutout set, which is what blocks ArtiFixer3D"
    ),
}


def test_the_stage_table_is_discoverable() -> None:
    assert STAGE_CONTRACTS, "no fresh-scene stage contracts found"
    assert all(stage.get("stage_id") for stage in STAGE_CONTRACTS)
    assert all(stage.get("implementation") for stage in STAGE_CONTRACTS)


@pytest.mark.parametrize(
    "stage", STAGE_CONTRACTS, ids=[stage["stage_id"] for stage in STAGE_CONTRACTS]
)
def test_every_stage_names_a_producer_that_exists_or_a_recorded_gap(stage) -> None:
    implementation = stage["implementation"]
    try:
        importlib.import_module(implementation)
    except ModuleNotFoundError:
        assert implementation in KNOWN_MISSING_PRODUCERS, (
            f"stage {stage['stage_id']!r} names producer {implementation!r}, "
            "which is not in the tree and is not a recorded gap. Add the module, "
            "or record it in KNOWN_MISSING_PRODUCERS with what closes it."
        )


def test_no_recorded_gap_outlives_the_module_that_closes_it() -> None:
    """Once the producer lands, the entry must go, or the ledger starts lying."""

    for implementation in sorted(KNOWN_MISSING_PRODUCERS):
        try:
            importlib.import_module(implementation)
        except ModuleNotFoundError:
            continue
        raise AssertionError(
            f"{implementation} now exists; remove it from KNOWN_MISSING_PRODUCERS "
            "so the remaining gaps stay an accurate list of what is missing."
        )


def test_every_recorded_gap_is_actually_named_by_a_stage() -> None:
    """A gap nothing references is a stale note, not a work item."""

    declared = {stage["implementation"] for stage in STAGE_CONTRACTS}
    orphaned = sorted(set(KNOWN_MISSING_PRODUCERS) - declared)
    assert not orphaned, f"KNOWN_MISSING_PRODUCERS names unreferenced modules: {orphaned}"
