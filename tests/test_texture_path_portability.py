"""Texture references must survive the trip to the provider.

The twin's materials referenced its PBR textures by absolute laptop paths.
Locally every viewer resolves them; on the provider none exist, so the
appliance renders untextured - silently, because a missing texture is a
warning, not an error. Same class as the cousin-scene USD digests that were
checkout-path-bound, one asset type over.
"""

from __future__ import annotations

import textwrap

import pytest

from blueprint_pipeline.texture_path_portability import (
    TexturePathPortabilityError,
    audit_texture_path_portability,
)


def _usda(tmp_path, body):
    path = tmp_path / "asset.usda"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_an_absolute_texture_path_is_refused(tmp_path):
    asset = _usda(
        tmp_path,
        """\
        #usda 1.0
        def Shader "albedo_texture"
        {
            asset inputs:file = @/Users/someone/textures/albedo.png@
        }
        """,
    )

    with pytest.raises(TexturePathPortabilityError) as excinfo:
        audit_texture_path_portability(asset_path=asset)

    assert any("absolute_texture_path" in e for e in excinfo.value.errors)


def test_a_relative_reference_must_resolve_beside_the_layer(tmp_path):
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "albedo.png").write_bytes(b"png")
    asset = _usda(
        tmp_path,
        """\
        #usda 1.0
        def Shader "albedo_texture"
        {
            asset inputs:file = @./textures/albedo.png@
        }
        """,
    )

    receipt = audit_texture_path_portability(asset_path=asset)

    assert receipt["texture_count"] == 1
    assert receipt["all_relative_and_resolvable"] is True


def test_a_dangling_relative_reference_is_refused(tmp_path):
    asset = _usda(
        tmp_path,
        """\
        #usda 1.0
        def Shader "albedo_texture"
        {
            asset inputs:file = @./textures/missing.png@
        }
        """,
    )

    with pytest.raises(TexturePathPortabilityError) as excinfo:
        audit_texture_path_portability(asset_path=asset)

    assert any("texture_unresolvable" in e for e in excinfo.value.errors)


def test_a_simulated_runtime_layout_checks_the_provider_side_truth(tmp_path):
    """`../native/x.png` is right for the bundle and wrong beside the source.

    The reference resolves against where the layer will SIT on the provider,
    not where it sits on the laptop, so the audit takes the runtime layout as
    an explicit remap.
    """

    runtime = tmp_path / "runtime"
    (runtime / "assets").mkdir(parents=True)
    (runtime / "native").mkdir()
    (runtime / "native" / "albedo.png").write_bytes(b"png")
    asset = _usda(
        tmp_path,
        """\
        #usda 1.0
        def Shader "albedo_texture"
        {
            asset inputs:file = @../native/albedo.png@
        }
        """,
    )

    receipt = audit_texture_path_portability(
        asset_path=asset, resolve_as_if_layer_lived_in=runtime / "assets"
    )

    assert receipt["all_relative_and_resolvable"] is True


def test_an_asset_with_no_textures_passes_trivially(tmp_path):
    asset = _usda(tmp_path, "#usda 1.0\n")

    receipt = audit_texture_path_portability(asset_path=asset)

    assert receipt["texture_count"] == 0
