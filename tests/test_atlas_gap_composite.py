"""Generated gap views corrected only where they contradict a well-observed capture.

capture-447ae4f4 (2026-10-30): World Labs returned whole new images for two
gap cameras and the whole-view review refused both. A capture pixel replaces
the provider's only where the capture saw the scene well and the provider
contradicts it beyond what content known to agree shows.
"""
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from blueprint_pipeline import atlas_depth_splat as depth
from blueprint_pipeline import atlas_gap_composite as composite
from tests.test_atlas_depth_splat import camera

BOX_DEPTH, BOX_HALF = 1.5, .15
GREEN = (0, 255, 0)


def scene(cam, frame_id, *, box=True, confidence=None):
    """A gradient plane at z=-2 and, optionally, an observed box in front of it.

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
        rgb[inside] = GREEN
    conf = np.full(planar.shape, 255, np.uint8) if confidence is None else confidence
    return depth.posed_frame(frame_id=frame_id, rgb=rgb, depth=planar.astype(np.float32), confidence=conf,
                             camera=cam)


POSITIONS = ((0, 0, 0), (.2, 0, 0), (.1, .1, 0), (.05, -.08, 0))


def real_frames(confidence=255):
    return [scene(camera(p), f"real-{i}", confidence=np.full((64, 48), confidence, np.uint8))
            for i, p in enumerate(POSITIONS)]


def columns(cam, start, stop):
    mask = np.zeros((cam["intrinsics"]["height"], cam["intrinsics"]["width"]), bool)
    mask[:, start:stop] = True
    return mask


def provider_view(cam, admitted):
    """A generated view that erased the observed box, as completion hands it on.

    Its own pixels agree with the capture elsewhere, within tolerance; completion
    pasted sparse context into ``rgb``, and the provider's image travels as ``provider_rgb``.
    """
    generated = scene(cam, "generated", box=False)
    generated["rgb"][..., 0] -= 3
    pasted = generated["rgb"].copy()
    pasted[::3, ::3] = (255, 255, 255)
    return depth.restrict_frame(generated, admitted, rgb=pasted)


def box_core(cam):
    box = (scene(cam, "truth")["rgb"] == GREEN).all(axis=-1)
    core = box.copy()
    core[1:-1, 1:-1] &= box[:-2, 1:-1] & box[2:, 1:-1] & box[1:-1, :-2] & box[1:-1, 2:]
    return box, core


def test_a_well_observed_object_the_provider_erased_returns_and_agreeing_pixels_stay_the_providers():
    cam = camera((.1, .05, .1))
    view = provider_view(cam, columns(cam, 0, 6))
    built, receipt = composite.composite_view(view, composite.observation_references(real_frames()))
    box, core = box_core(cam)
    provider = view["provider_rgb"]
    assert core.sum() > 50 and not (provider == GREEN).all(axis=-1).any()
    # The box the capture observed well returns, at its observed depth.
    assert (built["rgb"][core] == GREEN).all(axis=-1).mean() > .9
    assert np.allclose(built["depth"][core], scene(cam, "truth")["depth"][core], rtol=.02)
    # Everywhere it agrees, and in the admitted gap, the provider's own pixels and depth stay exactly.
    agree = distance_transform_edt(~box) > composite.FEATHER_PIXELS + 2
    assert agree.sum() > 1500
    assert np.array_equal(built["rgb"][agree], provider[agree])
    assert np.array_equal(built["depth"][agree], view["depth"][agree])
    admitted = view["keep_mask"]
    assert np.array_equal(built["rgb"][admitted], provider[admitted])
    # Not the context completion pasted into its frame.
    assert not (built["rgb"][::3, ::3][agree[::3, ::3]] == 255).all(axis=-1).any()
    assert np.array_equal(built["keep_mask"], admitted) and built["extrinsics"] == view["extrinsics"]
    assert receipt["depth_contradiction_pixel_count"] > 0 and receipt["replaced_pixel_count"] >= core.sum()
    assert (receipt["provider_pixel_count"] + receipt["replaced_pixel_count"]
            + receipt["feathered_pixel_count"]) == receipt["total_pixel_count"]
    assert receipt["thresholds"]["agree_relative_depth"] == composite.AGREE_RELATIVE_DEPTH
    assert receipt["provider_rgb_sha256"] == composite._digest(provider)
    assert receipt["reviewed_input_rgb_sha256"] == composite._digest(view["rgb"])
    assert receipt["rgb_sha256"] == composite._digest(built["rgb"]) and receipt["claim_ceiling"] == "development_only"


@pytest.mark.parametrize("case", ["low_confidence", "magnified"])
def test_a_poorly_observed_capture_never_overrides_the_provider(case):
    if case == "low_confidence":
        # Below the calibrated depth confidence.
        references, cam = real_frames(confidence=int(composite.MIN_CONFIDENCE * 255) - 1), camera((.1, .05, .1))
    else:
        # Close up, one reference pixel spans more target pixels than the gate allows.
        references, cam = real_frames(), camera((.1, .05, -1.1))
    view = provider_view(cam, columns(cam, 0, 6))
    rendered = composite.render_observed(composite.observation_references(references), cam)
    assert rendered["observed"].mean() > .3 and not composite.well_observed(rendered).any()
    built, receipt = composite.composite_view(view, composite.observation_references(references))
    assert np.array_equal(built["rgb"], view["provider_rgb"]) and np.array_equal(built["depth"], view["depth"])
    assert receipt["replaced_pixel_count"] == 0 and receipt["well_observed_pixel_count"] == 0


def test_unobserved_pixels_keep_the_provider_and_nothing_is_rendered_from_behind_the_camera():
    cam = camera((0, 0, -1.2), yaw_degrees=170)
    view = provider_view(cam, columns(cam, 0, 4))
    built, receipt = composite.composite_view(view, composite.observation_references(real_frames()))
    assert receipt["observed_pixel_count"] == 0
    assert np.array_equal(built["rgb"], view["provider_rgb"]) and np.array_equal(built["depth"], view["depth"])


def test_at_a_recorded_centre_a_photographed_object_returns_even_without_depth():
    # No real frame has confident depth on the box: photographed but unmeasured.
    unmeasured = []
    for i, position in enumerate(POSITIONS):
        cam = camera(position)
        box = (scene(cam, "probe")["rgb"] == GREEN).all(axis=-1)
        unmeasured.append(scene(cam, f"real-{i}", confidence=np.where(box, 0, 255).astype(np.uint8)))
    cam = camera()
    view = provider_view(cam, columns(cam, 40, 48))
    built, receipt = composite.composite_view(view, composite.observation_references(unmeasured))
    assert receipt["observed_basis"] == "recorded_centre_photographs" and receipt["colour_contradiction_pixel_count"]
    box, core = box_core(cam)
    assert (built["rgb"][core] == GREEN).all(axis=-1).all()
    # Its depth stays the provider's where no frame measured it.
    assert np.array_equal(built["depth"][box], view["depth"][box])


def test_a_changed_grid_refuses_instead_of_posting_a_misaligned_frame():
    cam = camera((.1, .05, .1))
    view = provider_view(cam, columns(cam, 0, 6))
    view["provider_rgb"] = view["provider_rgb"][:-1]
    with pytest.raises(ValueError, match="grid_changed"):
        composite.composite_view(view, composite.observation_references(real_frames()))


def test_completion_keeps_the_providers_own_image_beside_its_composite():
    generated = scene(camera(), "generated", box=False)
    pasted = generated["rgb"].copy()
    pasted[0] = 0
    restricted = depth.restrict_frame(generated, np.ones((64, 48), bool), rgb=pasted)
    assert np.array_equal(restricted["provider_rgb"], generated["rgb"]) and np.array_equal(restricted["rgb"], pasted)
    again = depth.restrict_frame(restricted, np.ones((64, 48), bool), rgb=pasted * 0)
    assert np.array_equal(again["provider_rgb"], generated["rgb"])
    assert "provider_rgb" not in depth.restrict_frame(generated, np.ones((64, 48), bool))
