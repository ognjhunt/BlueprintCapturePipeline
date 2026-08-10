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


def test_the_simready_lane_admits_only_the_image_it_can_actually_run() -> None:
    """Measured, after admitting one it could not.

    Widening this allowlist to include the Arena image let three launches
    through that produced 0, 1 and 0 lines of container log respectively -
    nothing ever started. The same transport on the bare image produces 870
    lines, and the Arena transport on the Arena image produces 1000. The image
    is fine and the bare image is fine here; this transport and that image
    together are not.

    Admitting a combination that cannot run is worse than refusing it: the
    guard stops meaning anything and the failures arrive as silence.
    """

    assert BARE_IMAGE in ADP_SIMREADY_ISAAC_ADMITTED_IMAGES
    assert ARENA_IMAGE not in ADP_SIMREADY_ISAAC_ADMITTED_IMAGES


def test_every_admitted_image_is_digest_pinned() -> None:
    """A floating tag would let the image change under a sealed receipt."""

    for image in ADP_SIMREADY_ISAAC_ADMITTED_IMAGES:
        assert "@sha256:" in image, image


def test_an_unpinned_or_unknown_image_is_not_admitted() -> None:
    for image in (
        "nvcr.io/nvidia/isaac-sim:6.0.1",
        "nvcr.io/nvidia/isaac-sim:latest",
        "docker.io/library/ubuntu@sha256:" + "0" * 64,
        "",
    ):
        assert image not in ADP_SIMREADY_ISAAC_ADMITTED_IMAGES


def test_the_admitted_set_stays_an_allowlist() -> None:
    """Kept as a tuple so adding an image is a deliberate, reviewable edit."""

    assert isinstance(ADP_SIMREADY_ISAAC_ADMITTED_IMAGES, tuple)
    assert len(ADP_SIMREADY_ISAAC_ADMITTED_IMAGES) == 1
