"""Faster support dilation preserves original pixels, PNGs and receipts."""
import io
import json

import numpy as np
from PIL import Image, ImageFilter
import pytest

from blueprint_pipeline import public_scene_inpainting_finalize as finalize
from blueprint_pipeline import public_scene_inpainting_inputs as inputs
from tests.test_public_scene_inpainting_inputs import _fake_sealed_render, _write_v2_fixture


def _pillow(pixels, radius):
    if radius == 0:
        return pixels.copy()  # The original finalizer skips filtering at zero.
    return np.asarray(Image.fromarray(pixels).filter(ImageFilter.MaxFilter(2 * radius + 1)))


def _png(pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG", optimize=False)
    return stream.getvalue()


@pytest.mark.parametrize("radius", [0, 1, 8, 17, 64])
@pytest.mark.parametrize("shape", [(1, 1), (1, 19), (23, 1), (17, 29), (35, 31)])
@pytest.mark.parametrize("kind", ["empty", "full", "border", "sparse", "dense", "random", "grayscale"])
def test_dilation_matches_pillow_pixels_and_png_bytes(radius, shape, kind):
    rng = np.random.default_rng(219)
    pixels = np.zeros(shape, dtype=np.uint8)
    if kind == "full":
        pixels.fill(255)
    elif kind == "border":
        pixels[0, ::2] = 255
        pixels[-1, ::3] = 255
        pixels[::2, 0] = 255
        pixels[::3, -1] = 255
    elif kind in {"sparse", "dense", "random"}:
        pixels[:] = (rng.random(shape) < {"sparse": .02, "dense": .95, "random": .5}[kind]) * 255
    elif kind == "grayscale":
        pixels[:] = rng.integers(0, 256, shape, dtype=np.uint8)
    # Reversed views exercise noncontiguous input and a read-only source.
    pixels = pixels[:, ::-1]
    pixels.setflags(write=False)
    original = pixels.copy()
    expected = _pillow(pixels, radius)
    actual = finalize._dilate_support_mask(pixels, radius)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(pixels, original)
    assert actual.dtype == np.uint8
    assert _png(actual) == _png(expected)


def test_finalizer_preserves_receipt_and_retained_mask_bytes(tmp_path, monkeypatch):
    paths = _write_v2_fixture(tmp_path, include_background=True)
    request = json.loads((paths["repo"] / "request.json").read_text())
    request.pop("request_digest")
    request["mask_policy"]["dilation_pixels"] = 8
    request["rendering"].update(width=1280, height=1280)
    request_path = paths["data"] / "dilation-request.json"
    request_path.write_text(json.dumps(inputs.build_public_scene_inpainting_input_request(request)))
    monkeypatch.setattr(inputs, "render_splat_at_exact_cameras", _fake_sealed_render)
    optimized = finalize._dilate_support_mask
    original_finish = finalize.finish_prepared_inputs
    captured = {}

    def capture(context, **kwargs):
        captured.update(context=context, kwargs=kwargs)
        return original_finish(context, **kwargs)

    monkeypatch.setattr(finalize, "finish_prepared_inputs", capture)
    monkeypatch.setattr(finalize, "_dilate_support_mask", _pillow)
    expected = inputs.materialize_public_scene_inpainting_inputs(
        request_path=request_path, repo_root=paths["repo"], data_root=paths["data"],
        output_root=paths["output"])
    masks = {p: p.read_bytes() for p in (paths["output"] / "masks").glob("*.png")}
    receipt = paths["output"] / "public_scene_interiorgs_edit_input_receipt.v2.json"
    receipt_bytes = receipt.read_bytes()
    assert masks
    monkeypatch.setattr(finalize, "_dilate_support_mask", optimized)
    actual = original_finish(captured["context"], **captured["kwargs"])
    assert actual == expected
    assert receipt.read_bytes() == receipt_bytes
    assert all(p.read_bytes() == raw for p, raw in masks.items())
    retained = original_finish(captured["context"], **captured["kwargs"], validate_only=True)
    assert retained == expected
    assert all(p.read_bytes() == raw for p, raw in masks.items())
