"""A payload must not die on a provider because an asset was renamed."""

from __future__ import annotations

import pytest

from blueprint_pipeline.runtime_asset_resolution import (
    RuntimeAssetResolutionError,
    resolve_runtime_asset,
)


def _write(directory, name: str) -> None:
    (directory / name).write_text("#usda 1.0\n", encoding="utf-8")


def test_resolves_the_declared_filename_when_it_is_present(tmp_path):
    _write(tmp_path, "scene.usda")

    resolved = resolve_runtime_asset(
        runtime_dir=tmp_path, declared_filename="scene.usda", role="scene_collision"
    )

    assert resolved["resolved_path"] == str(tmp_path / "scene.usda")
    assert resolved["matched_on"] == "declared_filename"


def test_resolves_an_alias_when_the_bundle_renamed_the_asset(tmp_path):
    """Asset bindings rename files into the bundle; the payload still needs them.

    The scene collision ships as ``sage_collision.usd`` no matter what it was
    called when it was authored, so a spec written against the authoring name
    finds nothing.
    """

    _write(tmp_path, "sage_collision.usd")

    resolved = resolve_runtime_asset(
        runtime_dir=tmp_path,
        declared_filename="840796_collision_without_refrigerator.usda",
        aliases=("sage_collision.usd",),
        role="scene_collision",
    )

    assert resolved["resolved_path"] == str(tmp_path / "sage_collision.usd")
    assert resolved["matched_on"] == "alias"
    assert resolved["matched_alias"] == "sage_collision.usd"


def test_searches_a_nested_assets_directory(tmp_path):
    """Bundles put assets under provider_runtime/assets, not beside the spec."""

    nested = tmp_path / "assets"
    nested.mkdir()
    _write(nested, "sage_collision.usd")

    resolved = resolve_runtime_asset(
        runtime_dir=tmp_path,
        declared_filename="sage_collision.usd",
        role="scene_collision",
    )

    assert resolved["resolved_path"] == str(nested / "sage_collision.usd")


def test_declared_filename_wins_over_an_alias(tmp_path):
    _write(tmp_path, "scene.usda")
    _write(tmp_path, "sage_collision.usd")

    resolved = resolve_runtime_asset(
        runtime_dir=tmp_path,
        declared_filename="scene.usda",
        aliases=("sage_collision.usd",),
        role="scene_collision",
    )

    assert resolved["matched_on"] == "declared_filename"


def test_a_miss_reports_every_usd_actually_present(tmp_path):
    """The failure must map the layout, so one launch resolves it for good.

    A bare "asset missing" costs a launch to learn the name and another to use
    it. Listing what is there makes the next attempt the last one.
    """

    _write(tmp_path, "sage_collision.usd")
    _write(tmp_path, "approved_can.usda")

    with pytest.raises(RuntimeAssetResolutionError) as excinfo:
        resolve_runtime_asset(
            runtime_dir=tmp_path,
            declared_filename="840796_collision_without_refrigerator.usda",
            role="scene_collision",
        )

    joined = ";".join(excinfo.value.errors)
    assert "runtime_asset_not_found:scene_collision" in joined
    assert "sage_collision.usd" in joined
    assert "approved_can.usda" in joined


def test_a_missing_runtime_directory_fails_closed(tmp_path):
    with pytest.raises(RuntimeAssetResolutionError) as excinfo:
        resolve_runtime_asset(
            runtime_dir=tmp_path / "nope",
            declared_filename="scene.usda",
            role="scene_collision",
        )

    assert any("runtime_dir_missing" in error for error in excinfo.value.errors)


def test_a_directory_named_like_the_asset_is_not_the_asset(tmp_path):
    (tmp_path / "scene.usda").mkdir()

    with pytest.raises(RuntimeAssetResolutionError):
        resolve_runtime_asset(
            runtime_dir=tmp_path, declared_filename="scene.usda", role="scene_collision"
        )
