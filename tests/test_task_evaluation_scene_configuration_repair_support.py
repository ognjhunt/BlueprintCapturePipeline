import pytest
from PIL import Image

from blueprint_pipeline.task_evaluation_scene_configuration_repair_support import materialize_repair_support


def test_missing_sam_never_falls_back_to_the_projected_box(tmp_path):
    frame, calibrated, sam, _ = _scene(tmp_path)
    Image.new("L", (160, 140), 0).save(sam)
    before = {p: p.read_bytes() for p in (frame, calibrated, sam)}
    with pytest.raises(ValueError, match="sam_core_missing"):
        _run(tmp_path, frame, calibrated, sam)
    assert not (tmp_path / "repair").exists()
    assert all(p.read_bytes() == value for p, value in before.items())


def test_empty_calibrated_and_sam_support_fails_before_output(tmp_path):
    frame = tmp_path / "frame.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (20, 20), "white").save(frame)
    Image.new("L", (20, 20), 0).save(mask)
    with pytest.raises(ValueError, match="core_missing"):
        materialize_repair_support(calibrated_mask_path=mask, sam_mask_path=mask,
            source_frame_path=frame, calibration_digest="sha256:" + "a" * 64,
            output_root=tmp_path / "repair")
    assert not (tmp_path / "repair").exists()


def test_object_core_never_blends_original_object_back_into_target(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_configuration_semantic_locality import seal_semantic_teacher_frame
    source, teacher, support, core, output = [tmp_path / name for name in
        ("source.png", "teacher.png", "support.png", "core.png", "sealed.png")]
    Image.new("RGB", (100, 100), "red").save(source)
    Image.new("RGB", (100, 100), "blue").save(teacher)
    mask = Image.new("L", (100, 100), 0)
    mask.paste(255, (20, 20, 80, 80))
    mask.save(support)
    # A core close to the support boundary used to receive feathered source
    # pixels, restoring a visible edge of the removed object.
    mask = Image.new("L", (100, 100), 0)
    mask.paste(255, (21, 21, 79, 79))
    mask.save(core)
    seal_semantic_teacher_frame(source_path=source, mask_path=support,
        raw_teacher_path=teacher, mask_encoding="binary_white_edit_region_png",
        output_path=output, object_core_mask_path=core)
    with Image.open(output) as sealed:
        assert set(sealed.crop((21, 21, 79, 79)).getdata()) == {(0, 0, 255)}
        assert sealed.getpixel((19, 50)) == (255, 0, 0)


def test_core_outside_admitted_support_is_refused(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_configuration_semantic_locality import seal_semantic_teacher_frame
    source, support, core = [tmp_path / name for name in ("source.png", "support.png", "core.png")]
    Image.new("RGB", (100, 100), "red").save(source)
    mask = Image.new("L", (100, 100), 0)
    mask.paste(255, (20, 20, 80, 80))
    mask.save(support)
    mask.putpixel((19, 50), 255)
    mask.save(core)
    with pytest.raises(RuntimeError, match="core_outside_support"):
        seal_semantic_teacher_frame(source_path=source, mask_path=support,
            raw_teacher_path=source, mask_encoding="binary_white_edit_region_png",
            output_path=tmp_path / "output.png", object_core_mask_path=core)


def _scene(tmp_path, *, surface=160, sam_box=(50, 40, 80, 100), calibrated_box=(20, 20, 140, 120)):
    """A dark object on a wood-toned surface; the calibrated projection is far larger than the silhouette."""
    import numpy as np
    frame, calibrated, sam = [tmp_path / name for name in ("frame.png", "calibrated.png", "sam.png")]
    canvas = np.full((140, 160), surface, dtype=np.uint8)
    left, top, right, bottom = sam_box
    canvas[top:bottom, left:right] = 60  # the object itself
    Image.fromarray(canvas, mode="L").convert("RGB").save(frame)
    mask = Image.new("L", (160, 140), 0)
    mask.paste(255, sam_box)
    mask.save(sam)
    mask = Image.new("L", (160, 140), 0)
    mask.paste(255, calibrated_box)
    mask.save(calibrated)
    return frame, calibrated, sam, canvas


def _run(tmp_path, frame, calibrated, sam):
    import numpy as np
    result = materialize_repair_support(calibrated_mask_path=calibrated, sam_mask_path=sam,
        source_frame_path=frame, calibration_digest="sha256:" + "a" * 64, output_root=tmp_path / "repair")
    with Image.open(result["repair_object_core"]["path"]) as image:
        core = np.asarray(image) > 0
    with Image.open(result["repair_support_mask"]["path"]) as image:
        support = np.asarray(image) > 0
    return result, core, support


def test_support_is_exactly_the_sam_silhouette(tmp_path):
    """Owner decision 2026-09-13: the editor understands the object from its outline; keep only that."""
    import numpy as np
    frame, calibrated, sam, canvas = _scene(tmp_path)
    canvas[40:100, 115:135] = 205  # a pale neighbour inside the calibrated projection
    canvas[100:112, 50:80] = 125   # the contact shadow
    Image.fromarray(canvas, mode="L").convert("RGB").save(frame)
    result, core, support = _run(tmp_path, frame, calibrated, sam)
    silhouette = np.zeros_like(core)
    silhouette[40:100, 50:80] = True
    assert np.array_equal(core, silhouette)
    assert np.array_equal(support, silhouette)
    prov = result["repair_support_mask"]["provenance"]
    assert prov["policy"] == "sam_silhouette_exact_no_fallback_v4"
    assert prov["guard_band_pixels"] == 0 and prov["calibrated_reach_pixels"] == 0
    assert prov["calibrated_pixels_beyond_reach_dropped"] == int((np.asarray(Image.open(calibrated)) > 0).sum()) - int(silhouette.sum())
    assert prov["guard_band_pixels_offered"] == 0 and prov["guard_band_pixels_admitted"] == 0
    assert result["repair_support_mask"]["digest"] == result["repair_object_core"]["digest"]


def test_full_frame_sam_is_refused_before_an_edit_request(tmp_path):
    frame, calibrated, sam, _ = _scene(tmp_path)
    Image.new("L", (160, 140), 255).save(sam)
    with pytest.raises(ValueError, match="sam_core_full_frame"):
        _run(tmp_path, frame, calibrated, sam)
    assert not (tmp_path / "repair").exists()
