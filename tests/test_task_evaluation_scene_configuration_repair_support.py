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
