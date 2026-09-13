"""Repair allowances must not promote visible neighbours into forced replacement."""
from PIL import Image
from tests.test_task_evaluation_scene_configuration_repair_support import _scene, _run


def test_pale_neighbor_touching_sam_is_not_promoted_to_opaque_object_core(tmp_path):
    frame, calibrated, sam, canvas = _scene(tmp_path)
    canvas[40:100, 80:98] = 205  # immediately beside the SAM silhouette, inside the 24px reach
    Image.fromarray(canvas).convert('RGB').save(frame)
    _, core, support = _run(tmp_path, frame, calibrated, sam)
    assert not core[40:100, 80:98].any()
    assert not support[40:100, 80:98].any()



def test_sam_core_is_full_opacity_and_every_other_source_pixel_is_exact(tmp_path):
    import numpy as np
    from blueprint_pipeline.task_evaluation_scene_configuration_semantic_locality import seal_semantic_teacher_frame
    frame, calibrated, sam, _ = _scene(tmp_path)
    result, core, support = _run(tmp_path, frame, calibrated, sam)
    raw = tmp_path / "raw.png"
    Image.new("RGB", (160, 140), (10, 90, 200)).save(raw)
    output = tmp_path / "sealed.png"
    seal_semantic_teacher_frame(source_path=frame, raw_teacher_path=raw,
        mask_path=result["repair_support_mask"]["path"],
        object_core_mask_path=result["repair_object_core"]["path"],
        mask_encoding="binary_white_edit_region_png", output_path=output)
    with Image.open(output) as image, Image.open(frame) as original, Image.open(raw) as generated:
        actual = np.asarray(image)
        assert np.array_equal(actual[~support], np.asarray(original)[~support])
        assert np.array_equal(actual[core], np.asarray(generated)[core])
    assert np.array_equal(core, support)
