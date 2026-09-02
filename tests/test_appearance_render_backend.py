"""Typed appearance render backend contract (Scene 839873 render audit)."""

from __future__ import annotations

import pytest

from blueprint_pipeline.appearance_render_backend import (
    BACKEND_ISAAC_NATIVE_NUREC,
    BACKEND_NRE_NATIVE_GRPC,
    BACKEND_PARTICLEFIELD_3DGRUT_TRANSCODE,
    BACKEND_PARTICLEFIELD_BLUEPRINT_PRIVATE,
    AppearanceRenderBackendError,
    backend_launch_render_path,
    build_appearance_render_backend,
    validate_appearance_render_backend,
)

SOURCE = "sha256:" + "9" * 64
DERIVED = "sha256:" + "1" * 64


def _particlefield(**overrides):
    base = dict(
        kind=BACKEND_PARTICLEFIELD_3DGRUT_TRANSCODE,
        source_asset_digest=SOURCE,
        derived_asset_digest=DERIVED,
        renderer_identity="isaac-sim:6.0.1",
        conversion_identity="threedgrut.export.scripts.transcode@a37ef721",
        camera_frame_contract="registered_world",
    )
    base.update(overrides)
    return build_appearance_render_backend(**base)


def test_backend_contract_is_digest_bound_and_round_trips() -> None:
    contract = _particlefield()
    assert contract["launch_render_path"] == "particlefield_3d_gaussian_splat"
    assert contract["development_only"] is False
    assert validate_appearance_render_backend(contract) == contract
    assert backend_launch_render_path(contract) == "particlefield_3d_gaussian_splat"

    tampered = {**contract, "derived_asset_digest": "sha256:" + "2" * 64}
    with pytest.raises(AppearanceRenderBackendError) as excinfo:
        validate_appearance_render_backend(tampered)
    assert "appearance_render_backend_receipt_digest_mismatch" in excinfo.value.errors


def test_blueprint_private_conversion_is_a_declared_development_comparator_only() -> None:
    with pytest.raises(AppearanceRenderBackendError) as excinfo:
        _particlefield(kind=BACKEND_PARTICLEFIELD_BLUEPRINT_PRIVATE)
    assert (
        "appearance_render_backend_development_kind_requires_declaration"
        in excinfo.value.errors
    )
    contract = _particlefield(kind=BACKEND_PARTICLEFIELD_BLUEPRINT_PRIVATE, development_only=True)
    assert contract["development_only"] is True
    with pytest.raises(AppearanceRenderBackendError):
        _particlefield(development_only=True)


def test_direct_backends_cannot_describe_a_conversion_that_did_not_happen() -> None:
    native = build_appearance_render_backend(
        kind=BACKEND_ISAAC_NATIVE_NUREC,
        source_asset_digest=SOURCE,
        derived_asset_digest=None,
        renderer_identity="isaac-sim:6.0.1",
        conversion_identity=None,
        camera_frame_contract="registered_world",
    )
    assert native["launch_render_path"] == "plain_nurec_volume"
    with pytest.raises(AppearanceRenderBackendError) as excinfo:
        build_appearance_render_backend(
            kind=BACKEND_NRE_NATIVE_GRPC,
            source_asset_digest=SOURCE,
            derived_asset_digest=DERIVED,
            renderer_identity="nvcr.io/nvidia/nre/nre@sha256:...",
            conversion_identity=None,
            camera_frame_contract="nurec_space",
        )
    assert (
        "appearance_render_backend_conversion_declared_without_conversion"
        in excinfo.value.errors
    )
    nre = build_appearance_render_backend(
        kind=BACKEND_NRE_NATIVE_GRPC,
        source_asset_digest=SOURCE,
        derived_asset_digest=None,
        renderer_identity="nvcr.io/nvidia/nre/nre@sha256:pinned",
        conversion_identity=None,
        camera_frame_contract="nurec_space",
    )
    assert nre["launch_render_path"] is None
    with pytest.raises(AppearanceRenderBackendError):
        backend_launch_render_path(nre)


def test_converted_backend_requires_derived_digest_and_conversion_identity() -> None:
    with pytest.raises(AppearanceRenderBackendError) as excinfo:
        _particlefield(derived_asset_digest=None)
    assert "appearance_render_backend_conversion_identity_missing" in excinfo.value.errors
    with pytest.raises(AppearanceRenderBackendError):
        _particlefield(camera_frame_contract="camera_local")
