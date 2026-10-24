"""An edit that left its source camera fails before any paid review.

The 2026-10-22 dishwasher concept run accepted three reframed or copied edits
of six; these synthetic cases reproduce both patterns without a provider.
"""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pytest

from blueprint_pipeline import website_image_completion as completion
from blueprint_pipeline.edited_view_camera import edited_view_camera, main as audit_main
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

SIZE = (480, 640)  # height, width


def scene(seed):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 255, (*SIZE, 3), dtype=np.uint8)).filter(ImageFilter.GaussianBlur(2))
    draw = ImageDraw.Draw(image)
    for _ in range(40):
        x, y = rng.integers(0, SIZE[1]), rng.integers(0, SIZE[0])
        draw.rectangle([x, y, x + rng.integers(10, 60), y + rng.integers(10, 60)],
                       fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    return image


def removal_mask(box=(220, 160, 420, 360)):
    mask = Image.new("L", (SIZE[1], SIZE[0]), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def repaint(image, mask):
    """What a faithful edit does: new pixels inside the removal, mild re-synthesis noise elsewhere."""
    pixels = np.asarray(image).astype(np.int16)
    pixels = pixels + np.random.default_rng(9).integers(-2, 3, pixels.shape)
    pixels[np.asarray(mask) > 0] = (120, 110, 100)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def reframe(image, zoom):
    width, height = image.size
    w, h = round(width / zoom), round(height / zoom)
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h)).resize((width, height), Image.Resampling.LANCZOS)


def case(tmp_path, edited, mask=None):
    mask = removal_mask() if mask is None else mask
    paths = {name: tmp_path / f"{name}.png" for name in ("original", "edited", "mask")}
    scene(1).save(paths["original"])
    edited.save(paths["edited"])
    mask.save(paths["mask"])
    return paths


@pytest.mark.parametrize("edit,expected", [
    (lambda original, mask: repaint(original, mask), "kept"),
    # The editor zoomed to fill its letterboxed canvas (frame decoded-000000119).
    (lambda original, mask: repaint(reframe(original, 1.1), mask), "moved"),
    (lambda original, mask: repaint(Image.fromarray(np.roll(np.asarray(original), 40, axis=1)), mask), "moved"),
    # The editor copied the reference view's composition (frames 161 and 223).
    (lambda original, mask: repaint(scene(2), mask), "moved"),
])
def test_edits_keep_or_leave_the_source_camera(tmp_path, edit, expected):
    paths = case(tmp_path, edit(scene(1), removal_mask()))
    result = edited_view_camera(paths["original"], paths["edited"], paths["mask"])
    assert result["status"] == expected, result


def test_an_edit_that_leaves_too_little_to_compare_stays_with_the_model_review(tmp_path):
    mask = removal_mask((4, 4, SIZE[1] - 5, SIZE[0] - 5))
    paths = case(tmp_path, repaint(scene(2), mask), mask)
    result = edited_view_camera(paths["original"], paths["edited"], paths["mask"])
    assert result["status"] == "unverifiable" and result["reason"] == "too_few_features_outside_mask"


def frames(paths):
    edited = [{"frame_id": "view", "image_path": str(paths["edited"]), "image_digest": _sha256_file(paths["edited"]),
               "generated_pixels_present": True, "remaining_mask_path": str(paths["mask"]),
               "remaining_mask_digest": _sha256_file(paths["mask"])}]
    originals = [{"frame_id": "view", "image_path": str(paths["original"]),
                  "image_digest": _sha256_file(paths["original"])}]
    return edited, originals


def test_moved_edit_is_blocked_without_buying_a_review(tmp_path, monkeypatch):
    def paid_review(**_):
        raise AssertionError("a moved camera must not reach the paid review")
    monkeypatch.setattr(completion, "_verify_completed_background", paid_review)
    edited, originals = frames(case(tmp_path, repaint(reframe(scene(1), 1.1), removal_mask())))
    review = completion.verify_completed_background(frames=edited, original_frames=originals,
                                                    plan={"targets": []}, output_root=tmp_path / "review")
    assert review["status"] == "blocked" and review["basis"] == "deterministic_view_geometry"
    assert review["review"]["camera_changed_frame_ids"] == ["view"]
    assert review["review"]["consistent_background"] is False
    again = completion.verify_completed_background(frames=edited, original_frames=originals,
                                                   plan={"targets": []}, output_root=tmp_path / "review")
    assert again["request_digest"] == review["request_digest"]


def test_kept_edit_still_goes_to_the_paid_review(tmp_path, monkeypatch):
    monkeypatch.setattr(completion, "_verify_completed_background", lambda **_: {"status": "passed", "paid": True})
    edited, originals = frames(case(tmp_path, repaint(scene(1), removal_mask())))
    assert completion.verify_completed_background(frames=edited, original_frames=originals,
        plan={"targets": []}, output_root=tmp_path / "review") == {"status": "passed", "paid": True}


def test_changed_mask_bytes_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(completion, "_verify_completed_background", lambda **_: {"status": "passed"})
    paths = case(tmp_path, repaint(scene(1), removal_mask()))
    edited, originals = frames(paths)
    removal_mask((0, 0, 50, 50)).save(paths["mask"])
    with pytest.raises(ValueError, match="review_source_changed"):
        completion.verify_completed_background(frames=edited, original_frames=originals,
                                               plan={"targets": []}, output_root=tmp_path / "review")


def test_audit_flags_an_approved_background_view_that_left_its_camera(tmp_path, capsys):
    rows = []
    for name, edit in (("kept", lambda o, m: repaint(o, m)), ("copied", lambda o, m: repaint(scene(2), m))):
        root = tmp_path / name
        root.mkdir()
        paths = case(root, edit(scene(1), removal_mask()))
        edited, originals = frames(paths)
        rows.append({**edited[0], "frame_id": name, "original_image_path": originals[0]["image_path"],
                     "original_image_digest": originals[0]["image_digest"]})
    rows.append({"frame_id": "unedited", "image_path": rows[0]["original_image_path"]})
    retained = tmp_path / "prepared_background.json"
    retained.write_text(json.dumps({"background": {"frames": rows}}))
    assert audit_main([str(retained)]) == 2
    status = {row["frame_id"]: row["status"] for row in json.loads(capsys.readouterr().out)}
    assert status == {"kept": "kept", "copied": "moved", "unedited": "not_edited"}


def test_reused_view_whose_mask_left_with_its_run_goes_to_the_model_review(tmp_path, monkeypatch):
    monkeypatch.setattr(completion, "_verify_completed_background", lambda **_: {"status": "passed", "paid": True})
    paths = case(tmp_path, repaint(reframe(scene(1), 1.1), removal_mask()))
    edited, originals = frames(paths)
    paths["mask"].unlink()
    assert completion.verify_completed_background(frames=edited, original_frames=originals,
        plan={"targets": []}, output_root=tmp_path / "review") == {"status": "passed", "paid": True}
