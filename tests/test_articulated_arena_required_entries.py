from __future__ import annotations

from blueprint_pipeline.vast_provider_adapter import (
    ADP009D_ARTICULATED_ARENA_REQUIRED_ENTRIES,
    ADP009D_ISAAC_REQUIRED_ENTRIES,
)


def test_it_requires_what_this_payload_actually_ships() -> None:
    """The scene spec and the twin, which the rigid lane has no notion of."""

    for entry in (
        "provider_runtime/run_adp_arena_provider_runtime.sh",
        "provider_runtime/adp_arena_provider_runner.py",
        "provider_runtime/adp_arena_provider_manifest.json",
        "provider_runtime/native/adp009d_articulated_scene_spec.json",
    ):
        assert entry in ADP009D_ARTICULATED_ARENA_REQUIRED_ENTRIES, entry


def test_it_does_not_require_the_sealed_scenes_overlay() -> None:
    """The one entry a non-SAGE collision scene cannot produce."""

    assert (
        "provider_runtime/assets/sage_collision_overlay.usda"
        not in ADP009D_ARTICULATED_ARENA_REQUIRED_ENTRIES
    )


def test_the_rigid_lane_still_requires_its_overlay() -> None:
    """Adding a kind must not quietly relax the one already qualified."""

    assert (
        "provider_runtime/assets/sage_collision_overlay.usda"
        in ADP009D_ISAAC_REQUIRED_ENTRIES
    )


def test_both_kinds_still_demand_the_runner_and_its_entrypoint() -> None:
    """Whatever else differs, a bundle without a runner cannot run."""

    for entries in (
        ADP009D_ARTICULATED_ARENA_REQUIRED_ENTRIES,
        ADP009D_ISAAC_REQUIRED_ENTRIES,
    ):
        assert "provider_runtime/adp_arena_provider_runner.py" in entries
        assert "provider_runtime/run_adp_arena_provider_runtime.sh" in entries


def test_the_articulated_set_is_not_a_superset_of_the_rigid_one() -> None:
    """If it required everything the rigid lane does, it would need no kind."""

    assert not ADP009D_ISAAC_REQUIRED_ENTRIES.issubset(
        ADP009D_ARTICULATED_ARENA_REQUIRED_ENTRIES
    )
