from __future__ import annotations

from blueprint_pipeline.paid_resource_allocator import (
    ADP_SIMREADY_ISAAC_ADMITTED_IMAGES,
)
from blueprint_pipeline.adp009d_native_microcheck_bundle import (
    DEFAULT_IMAGE as ARENA_IMAGE,
)
from blueprint_pipeline.public_scene_simready_isaac_bundle import (
    DEFAULT_IMAGE as BARE_IMAGE,
)


def test_both_pinned_isaac_images_are_admitted() -> None:
    """An Arena payload needs the image that has Arena on it.

    The bare image carries isaacsim alone, which is right for a raw PhysX
    probe and cannot run an Arena composition at all. Both are digest-pinned
    NVIDIA images already used by lanes in this repository.
    """

    assert BARE_IMAGE in ADP_SIMREADY_ISAAC_ADMITTED_IMAGES
    assert ARENA_IMAGE in ADP_SIMREADY_ISAAC_ADMITTED_IMAGES


def test_every_admitted_image_is_digest_pinned() -> None:
    """A floating tag would let the image change under a sealed receipt."""

    for image in ADP_SIMREADY_ISAAC_ADMITTED_IMAGES:
        assert "@sha256:" in image, image


def test_an_unpinned_or_unknown_image_is_not_admitted() -> None:
    """Widening from one image to two must not widen to anything."""

    for image in (
        "nvcr.io/nvidia/isaac-sim:6.0.1",
        "nvcr.io/nvidia/isaac-sim:latest",
        "docker.io/library/ubuntu@sha256:" + "0" * 64,
        "",
    ):
        assert image not in ADP_SIMREADY_ISAAC_ADMITTED_IMAGES


def test_the_admitted_set_is_small_and_explicit() -> None:
    """This is an allowlist; it should be readable at a glance."""

    assert len(ADP_SIMREADY_ISAAC_ADMITTED_IMAGES) == 2
