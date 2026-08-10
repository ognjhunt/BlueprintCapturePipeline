from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.robot_asset_discovery import (
    ROBOT_ASSET_DISCOVERY_SCHEMA_VERSION,
    RobotAssetDiscoveryError,
    FRANKA_CANDIDATE_RELATIVE_PATHS,
    discover_robot_asset,
)


def test_the_first_candidate_that_exists_wins(tmp_path: Path) -> None:
    root = tmp_path / "isaac"
    (root / "b").mkdir(parents=True)
    (root / "b" / "franka.usd").write_text("x", encoding="utf-8")

    found = discover_robot_asset(
        search_roots=[root], relative_candidates=["a/franka.usd", "b/franka.usd"]
    )

    assert found["resolved_path"].endswith("b/franka.usd")
    assert found["schema_version"] == ROBOT_ASSET_DISCOVERY_SCHEMA_VERSION


def test_every_place_looked_is_reported_when_nothing_is_found(
    tmp_path: Path,
) -> None:
    """A launch that finds no robot must say where it looked.

    Otherwise the next attempt guesses at paths too, and each guess is another
    container boot. The search itself is the useful output when it fails.
    """

    with pytest.raises(RobotAssetDiscoveryError) as excinfo:
        discover_robot_asset(
            search_roots=[tmp_path / "isaac"],
            relative_candidates=["a/franka.usd", "b/franka.usd"],
        )

    joined = ";".join(excinfo.value.errors)
    assert "a/franka.usd" in joined and "b/franka.usd" in joined
    assert "robot_asset_not_found" in joined


def test_a_directory_masquerading_as_the_asset_is_not_accepted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "isaac"
    (root / "a" / "franka.usd").mkdir(parents=True)

    with pytest.raises(RobotAssetDiscoveryError):
        discover_robot_asset(
            search_roots=[root], relative_candidates=["a/franka.usd"]
        )


def test_the_shipped_candidate_list_covers_the_known_layouts() -> None:
    """Isaac has moved its robot assets between releases more than once."""

    joined = " ".join(FRANKA_CANDIDATE_RELATIVE_PATHS)
    assert "Isaac/Robots/Franka" in joined
    assert any(path.endswith(".usd") for path in FRANKA_CANDIDATE_RELATIVE_PATHS)
    assert len(set(FRANKA_CANDIDATE_RELATIVE_PATHS)) == len(
        FRANKA_CANDIDATE_RELATIVE_PATHS
    )


def test_discovery_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "isaac"
    (root / "b").mkdir(parents=True)
    (root / "b" / "franka.usd").write_text("x", encoding="utf-8")
    args = {"search_roots": [root], "relative_candidates": ["b/franka.usd"]}

    assert discover_robot_asset(**args) == discover_robot_asset(**args)


from blueprint_pipeline.robot_asset_discovery import (  # noqa: E402
    is_usable_robot_asset,
)


def test_a_viewport_test_fixture_is_not_a_robot_asset() -> None:
    """The probe found exactly one Franka-named USD and it was a bolt test.

    /isaac-sim/extscache/omni.kit.viewport.actions-.../data/tests/usd/referenced/
    M20_Bolt_Tight_R512_Franka_SI.usda is a viewport regression scene. Counting
    it as a robot asset reported raw_usd_composition as viable on an image that
    cannot support it, which is worse than reporting nothing - it would have
    sent the next launch down a dead route.
    """

    assert (
        is_usable_robot_asset(
            "/isaac-sim/extscache/omni.kit.viewport.actions-110.0.0+f9bf0dda"
            "/data/tests/usd/referenced/M20_Bolt_Tight_R512_Franka_SI.usda"
        )
        is False
    )


def test_a_real_robot_asset_path_is_accepted() -> None:
    assert is_usable_robot_asset("/isaac-sim/assets/Isaac/Robots/Franka/franka.usd")


def test_anything_under_a_tests_or_extscache_tree_is_rejected() -> None:
    for path in (
        "/x/data/tests/franka.usd",
        "/x/extscache/some.ext/data/franka.usd",
        "/x/unittests/franka.usd",
    ):
        assert is_usable_robot_asset(path) is False, path
