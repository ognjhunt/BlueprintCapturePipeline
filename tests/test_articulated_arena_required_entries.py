from __future__ import annotations
from pathlib import Path

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


from blueprint_pipeline.vast_provider_adapter import (  # noqa: E402
    _is_isaac_provider_bundle,
)


def test_the_articulated_kind_is_recognised_as_an_isaac_bundle() -> None:
    """It runs the Isaac image, so it needs the Isaac CUDA-safety path.

    Isaac images do not necessarily expose a standalone libcudart to their
    bundled Python; they prove driver compatibility through SimulationApp plus
    Warp device enumeration instead. A bundle missing from this predicate gets
    the strict standalone probe, reports cudart_missing, and is refused - after
    booting Isaac and initialising the GPU perfectly well. Two launches died
    that way, on different hosts, reading as host faults.
    """

    assert _is_isaac_provider_bundle("adp009d_articulated_arena") is True


def test_the_other_isaac_kinds_are_unchanged() -> None:
    for kind in ("isaac", "adp_simready_isaac", "adp_arena", "adp009d_isaac"):
        assert _is_isaac_provider_bundle(kind) is True


def test_a_non_isaac_kind_is_still_not_one() -> None:
    """The predicate selects a safety path; it must stay narrow."""

    for kind in ("wam", "evaluator", "adp_joint_agent", "adp_content_agents"):
        assert _is_isaac_provider_bundle(kind) is False


def test_every_place_the_kind_is_switched_on_knows_about_it() -> None:
    """Adding a bundle kind means adding it everywhere the kind is branched on.

    This one cost three launches. The contract and the entry table knew about
    adp009d_articulated_arena; the CUDA-safety predicate did not, so the run
    got the strict libcudart probe an Isaac image legitimately fails. Fixing
    that exposed the next site: the on-host extraction branch, which unpacks to
    adp_arena_provider_bundle and looks for run_adp_arena_provider_runtime.sh -
    a kind missing from it lands in no branch and reports entrypoint_missing
    for an entrypoint that is present in the zip.

    Grepping the adapter for the sibling kind is how you find them all.
    """

    source = (
        Path(__file__).resolve().parents[1]
        / "src/blueprint_pipeline/vast_provider_adapter.py"
    ).read_text(encoding="utf-8")

    for line in source.splitlines():
        if '"adp009d_isaac"' in line and "adp_arena" in line and "{" in line:
            assert '"adp009d_articulated_arena"' in line, (
                "kind-switch missing the articulated kind: " + line.strip()[:120]
            )
