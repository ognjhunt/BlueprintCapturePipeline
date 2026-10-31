"""Generated gap views rebuilt around what the real frames observed.

capture-447ae4f4 (2026-10-30): World Labs returned whole new images for two
gap cameras; outside the reviewed gap they disagreed with the capture, so the
whole-view review refused both and the filled floor never reached the splat.
"""
import numpy as np
import pytest

from blueprint_pipeline import atlas_depth_splat as depth
from blueprint_pipeline import atlas_gap_composite as composite
from tests.test_atlas_depth_splat import camera

BOX_DEPTH, BOX_HALF = 1.5, .15


def scene(cam, frame_id, *, box=True, confidence=None):
    """A gradient floor plane at z=-2 and, optionally, an observed box in front of it.

    Colour follows the world position, so every camera sees the same surface.
    """
    k, e = cam["intrinsics"], cam["extrinsics"]
    u = (np.arange(k["width"]) + .5 - k["cx"]) / k["fx"]
    v = (np.arange(k["height"]) + .5 - k["cy"]) / k["fy"]
    rays = np.stack(np.broadcast_arrays(u[None, :], -v[:, None], -1.0), axis=-1)
    rays = rays @ depth._rotation_matrix(e["quaternion"]).T
    centre = np.asarray(e["position"], float)
    planar = (-2.0 - centre[2]) / rays[..., 2]
    hit = centre + rays * planar[..., None]
    rgb = np.zeros((*planar.shape, 3), np.uint8)
    rgb[..., 0] = 200
    rgb[..., 1] = np.clip((hit[..., 0] + 1) * 120, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip((hit[..., 1] + 1) * 120, 0, 255).astype(np.uint8)
    if box:
        near = (-BOX_DEPTH - centre[2]) / rays[..., 2]
        face = centre + rays * near[..., None]
        inside = (np.abs(face[..., 0] - .05) < BOX_HALF) & (np.abs(face[..., 1]) < BOX_HALF)
        planar = np.where(inside, near, planar)
        rgb[inside] = (0, 255, 0)
    conf = np.full(planar.shape, 255, np.uint8) if confidence is None else confidence
    return depth.posed_frame(frame_id=frame_id, rgb=rgb, depth=planar.astype(np.float32), confidence=conf,
                             camera=cam)


def real_frames():
    return [scene(camera(), "real-0"), scene(camera((.2, 0, 0)), "real-1")]


def provider_view(cam, admitted):
    """A generated view that erased the observed box (what a provider may do) and was admitted in part."""
    generated = scene(cam, "generated", box=False)
    generated["rgb"][..., 2] = 40  # Its own colours too: nothing outside the admitted region is kept.
    return depth.restrict_frame(generated, admitted)


def columns(cam, start, stop):
    mask = np.zeros((cam["intrinsics"]["height"], cam["intrinsics"]["width"]), bool)
    mask[:, start:stop] = True
    return mask


def test_an_observed_object_the_provider_erased_returns_and_the_admitted_gap_stays_generated():
    cam = camera((.1, .05, .1))
    view = provider_view(cam, columns(cam, 0, 6))
    references = composite.observation_references(real_frames())
    built, receipt = composite.composite_view(view, references)
    truth = scene(cam, "truth")
    box = (truth["rgb"] == (0, 255, 0)).all(axis=-1)
    assert box.sum() > 100 and not (view["rgb"] == (0, 255, 0)).all(axis=-1).any()
    # The box the capture observed is back, at its observed depth.
    core = box.copy()
    core[1:-1, 1:-1] &= box[:-2, 1:-1] & box[2:, 1:-1] & box[1:-1, :-2] & box[1:-1, 2:]
    assert (built["rgb"][core] == (0, 255, 0)).all(axis=-1).mean() > .9
    assert np.allclose(built["depth"][core], truth["depth"][core], rtol=.02)
    # The admitted gap keeps the provider's pixels and depth exactly.
    admitted = view["keep_mask"]
    assert np.array_equal(built["rgb"][admitted], view["rgb"][admitted])
    assert np.array_equal(built["depth"][admitted], view["depth"][admitted])
    # Beyond the fill's feathered edge, observed pixels are the observation, not a blend.
    from scipy.ndimage import distance_transform_edt
    rgb, observed_depth, observed, basis = composite.render_observed(references, cam)
    far = observed & (distance_transform_edt(~admitted) > composite.FEATHER_PIXELS + 1)
    assert basis == "recorded_depth_splats" and far.sum() > 1000
    assert np.array_equal(built["rgb"][far], rgb[far])
    assert np.array_equal(built["depth"][far], observed_depth[far])
    # Admission, camera and confidence are unchanged; the receipt accounts for every pixel.
    assert np.array_equal(built["keep_mask"], admitted) and built["extrinsics"] == view["extrinsics"]
    assert np.array_equal(built["confidence"], view["confidence"])
    assert (receipt["observed_pixel_count"] + receipt["admitted_generated_pixel_count"]
            + receipt["unobserved_generated_pixel_count"]) == receipt["total_pixel_count"]
    assert receipt["admitted_generated_pixel_count"] == admitted.sum()
    assert receipt["candidate_basis"] == composite.CANDIDATE_BASIS
    assert receipt["reference_frame_ids"] == ["real-0", "real-1"]
    assert receipt["replaced_mean_absolute_difference"] > 0
    assert receipt["rgb_sha256"] == composite._digest(built["rgb"])
    assert receipt["claim_ceiling"] == "development_only"


def test_unobserved_pixels_keep_the_provider_and_nothing_is_rendered_from_behind_the_camera():
    # Turned away from everything the real frames saw except a sliver.
    cam = camera((0, 0, -1.2), yaw_degrees=170)
    view = provider_view(cam, columns(cam, 0, 4))
    built, receipt = composite.composite_view(view, composite.observation_references(real_frames()))
    assert receipt["observed_pixel_count"] == 0
    assert np.array_equal(built["rgb"], view["rgb"]) and np.array_equal(built["depth"], view["depth"])


def test_at_a_recorded_centre_photographed_pixels_return_even_without_depth():
    # No real frame has confident depth on the box: photographed but unmeasured.
    unmeasured = []
    for cam, frame_id in ((camera(), "real-0"), (camera((.2, 0, 0)), "real-1")):
        box = (scene(cam, frame_id)["rgb"] == (0, 255, 0)).all(axis=-1)
        unmeasured.append(scene(cam, frame_id, confidence=np.where(box, 0, 255).astype(np.uint8)))
    cam = camera()
    box = (unmeasured[0]["rgb"] == (0, 255, 0)).all(axis=-1)
    view = provider_view(cam, columns(cam, 40, 48))
    built, receipt = composite.composite_view(view, composite.observation_references(unmeasured))
    assert receipt["observed_basis"] == "recorded_centre_photographs"
    outside = ~columns(cam, 35, 48)
    assert np.array_equal(built["rgb"][outside], unmeasured[0]["rgb"][outside])
    # Its depth stays the provider's where no frame measured it.
    assert np.array_equal(built["depth"][box & outside], view["depth"][box & outside])


def test_a_changed_grid_refuses_instead_of_posting_a_misaligned_frame():
    cam = camera((.1, .05, .1))
    view = provider_view(cam, columns(cam, 0, 6))
    view["rgb"] = view["rgb"][:-1]
    with pytest.raises(ValueError, match="grid_changed"):
        composite.composite_view(view, composite.observation_references(real_frames()))
