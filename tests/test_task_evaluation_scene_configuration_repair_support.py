import pytest
from PIL import Image

from blueprint_pipeline.task_evaluation_scene_configuration_repair_support import materialize_repair_support


def test_calibrated_full_object_repairs_a_missed_sam_view_without_altering_sam(tmp_path):
    frame, calibrated, sam = [tmp_path / name for name in ("frame.png", "calibrated.png", "sam.png")]
    Image.new("RGB", (120, 120), "white").save(frame)
    core = Image.new("L", (120, 120), 0)
    core.paste(255, (40, 45, 70, 65))
    core.save(calibrated)
    Image.new("L", (120, 120), 0).save(sam)
    original = sam.read_bytes()
    result = materialize_repair_support(calibrated_mask_path=calibrated, sam_mask_path=sam,
        source_frame_path=frame, calibration_digest="sha256:" + "a" * 64,
        output_root=tmp_path / "repair")
    with Image.open(result["repair_support_mask"]["path"]) as support:
        assert support.getbbox() == (8, 13, 102, 97)
    with Image.open(result["repair_object_core"]["path"]) as actual_core:
        assert actual_core.tobytes() == core.tobytes()
    assert sam.read_bytes() == original
    assert result["repair_support_mask"]["provenance"]["whole_object_coverage_visually_qualified"] is False


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


def test_calibrated_coverage_far_from_the_silhouette_is_dropped_and_the_band_keeps_the_neighbour_out(tmp_path):
    """2026-09-13 source-08: the calibrated projection annexed the bottle beside the vase."""
    frame, calibrated, sam, canvas = _scene(tmp_path)
    # A pale neighbour inside the calibrated projection, well beyond reach of the silhouette,
    # a contact shadow touching the object, and a detached shadow-toned patch.
    canvas[40:100, 115:135] = 205
    canvas[100:112, 50:80] = 125
    canvas[10:18, 60:70] = 30  # a dark neighbour standing within the guard band above the object
    Image.fromarray(canvas, mode="L").convert("RGB").save(frame)
    result, core, support = _run(tmp_path, frame, calibrated, sam)
    assert core[40:100, 50:80].all()                      # the silhouette
    assert core[40:100, 80:104].all()                      # calibrated coverage within reach stays
    assert not core[40:100, 115:135].any()                 # the neighbour is no longer object core
    assert not support[40:100, 115:135].any()              # nor guard band
    assert support[100:112, 50:80].all()                   # the contact shadow is admitted
    assert not support[10:18, 60:70].any()                 # a dark neighbour in the band is not
    prov = result["repair_support_mask"]["provenance"]
    assert prov["policy"] == "calibrated_object_near_sam_core_surface_band_guard_v2"
    assert prov["calibrated_pixels_beyond_reach_dropped"] > 0
    assert prov["guard_band_pixels_admitted"] < prov["guard_band_pixels_offered"]
    assert prov["luminance_ceiling"] < 1.1 * prov["surface_reference_luminance"]
    assert result["repair_support_mask"]["digest"] != result["repair_object_core"]["digest"]


def test_view_sam_missed_entirely_keeps_the_whole_calibrated_projection(tmp_path):
    frame, calibrated, sam, canvas = _scene(tmp_path)
    Image.new("L", (160, 140), 0).save(sam)
    result, core, support = _run(tmp_path, frame, calibrated, sam)
    assert core[20:120, 20:140].all() and not core[0:20, :].any()
    assert result["repair_support_mask"]["provenance"]["calibrated_pixels_beyond_reach_dropped"] == 0
    assert support.sum() > core.sum()
