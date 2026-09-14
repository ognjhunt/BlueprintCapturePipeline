"""ADP-009D: corrected targets only, bounded repair, preserved neighboring scene."""
from pathlib import Path
import json

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.artifixer_appearance_freeze import (
    TRAINING_POLICY, freeze_source_appearance, local_appearance_mask, verify_frozen_appearance,
)
from tests.test_artifixer_background_initialization import _model_fixture, splat
from tests.test_public_scene_artifixer3d_dual_target_runner import _runner_module
from tests.test_public_scene_artifixer3d_dual_target_inputs import _dual_candidate


def policy():
    return {"mode": TRAINING_POLICY, "region_rule": "target_or_registered_tabletop_3sigma_v1", "target_lower_m": [0, 0, 0],
            "target_upper_m": [.08, .08, .2], "support_top_z_m": 0.0}


def test_local_color_region_protects_tall_neighbor_and_distant_room():
    reference = splat([[.04,.04,.1], [.13,.04,0], [.13,.04,.04],
                      [.5,.04,0], [.13,.04,0], [0,0,0]],
                     [[.01]*3, [.01,.01,.001], [.01]*3,
                      [.01,.01,.001], [.01,.01,.04], [.01]*3])
    partition = {"frozen_source_count": 5, "generated_support_count": 1,
                 "total_count": 6, "reused_source_vertex_rows_byte_exact": True,
                 "local_appearance_policy": policy()}
    assert local_appearance_mask(reference, partition).tolist() == [True, True, False, False, False]
    # Rotation makes a formerly flat splat vertical; it must cease to be editable.
    reference.quats[1] = [2**-.5, 2**-.5, 0, 0]
    assert not local_appearance_mask(reference, partition)[1]


def test_adam_updates_only_declared_source_region_and_generated_colors():
    torch, reference, partition, Model = _model_fixture()
    partition["local_appearance_policy"] = {
        "mode": TRAINING_POLICY, "region_rule": "target_or_registered_tabletop_3sigma_v1", "target_lower_m": [-.1,-.1,.9],
        "target_upper_m": [.1,.1,1.1], "support_top_z_m": .9}
    with freeze_source_appearance(Model, reference=reference, partition=partition):
        model = Model()
        before = model.features_albedo.detach().clone()
        optimizer = torch.optim.Adam([model.features_albedo, model.features_specular], lr=.1)
        for _ in range(4):
            optimizer.zero_grad()
            (model.features_albedo.sum()+model.features_specular.sum()).backward()
            optimizer.step()
        assert not torch.equal(model.features_albedo[0], before[0])
        assert torch.equal(model.features_albedo[1], before[1])
        assert not torch.equal(model.features_albedo[2], before[2])
    result = verify_frozen_appearance(model=model, reference=reference, partition=partition)
    assert result["editable_source_count"] == 1
    assert result["exact_source_appearance_prefix_match"] is False
    assert result["exact_protected_source_appearance_match"] is True
    with torch.no_grad():
        model.features_albedo[1,0] += .1
    with pytest.raises(ValueError, match="frozen_appearance_changed"):
        verify_frozen_appearance(model=model, reference=reference, partition=partition)


def test_corrected_training_omits_originals_and_rejected_camera(tmp_path):
    runner = _runner_module()
    _, _, _, dual = _dual_candidate(tmp_path)
    task = dual["tasks"][0]
    staged = Path(task["scene_directory"])
    # The second camera is rejected; neither its original nor edit can train.
    task["frames"][1]["semantic_teacher_excluded_from_training"] = True
    task["semantic_teacher_indices"] = [1]
    teacher_root, _ = runner._prepare_dual_target_teacher_frames(
        task=task, staged_task=staged, task_output=tmp_path/"training")
    transforms = staged/task["transforms"]["relative_path"]
    path, selected, teachers, rows = runner._prepare_corrected_only_training(
        task=task, transforms_path=transforms, teacher_root=teacher_root, staged_task=staged)
    assert json.loads(selected.read_text()) == []
    actual = json.loads(path.read_text())
    assert len(actual["frames"]) == len(rows) == 1
    assert rows[0]["camera_id"] == task["frames"][0]["camera_id"]
    original = json.loads(transforms.read_text())["frames"][1]
    for key in original.keys()-{"file_path", "training_role"}:
        assert actual["frames"][0][key] == original[key]
    assert [p.name for p in teachers.iterdir()] == ["00000.png"]
    with Image.open(teachers/"00000.png") as actual_image, Image.open(teacher_root/"00001.png") as expected:
        assert np.array_equal(actual_image, expected)
    assert runner._sha256(teachers/"00000.png") == task["frames"][0]["semantic_teacher_rgb"]["sha256"]


def test_native_export_accepts_only_explicit_local_protection_proof():
    from blueprint_pipeline.public_scene_artifixer3d_native_exports import geometry_protection_is_qualified
    proof = {"mode": "freeze_declared_appearance_initialization", "status": "qualified",
             "exact_position_tensor_match": True, "exact_rotation_tensor_match": True,
             "exact_scale_tensor_match": True, "blockers": [],
             "initialization_receipt_digest": "sha256:"+"a"*64,
             "exact_full_density_tensor_match": True, "local_appearance_policy": policy(),
             "exact_protected_source_appearance_match": True,
             "editable_source_count": 2, "protected_source_count": 8, "frozen_source_count": 10}
    assert geometry_protection_is_qualified(proof)
    assert not geometry_protection_is_qualified({**proof, "protected_source_count": 7})
    assert not geometry_protection_is_qualified({**proof, "exact_protected_source_appearance_match": False})


def test_completed_training_reuse_matches_admitted_subset_only():
    from blueprint_pipeline.artifixer_completed_training_reuse import admitted_teacher_training_images
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    frames = [{"camera_id": str(i), "whole_frame_semantic_teacher": {"sha256": str(i)}} for i in range(10)]
    selection = {"schema_version": "semantic_target_training_selection.v1",
                 "status": "admitted_for_training_only", "minimum_approved_views": 8,
                 "approved_camera_ids": [str(i) for i in range(8)], "excluded_camera_ids": ["8", "9"]}
    selection["selection_digest"] = canonical_digest(selection, digest_field="selection_digest")
    teacher = {"frames": frames, "editor_identity": {"training_view_selection": selection}}
    assert admitted_teacher_training_images(teacher) == {str(i):str(i) for i in range(8)}
    selection["excluded_camera_ids"] = ["9"]
    with pytest.raises(ValueError, match="selection_invalid"):
        admitted_teacher_training_images(teacher)


def test_unknown_training_policy_refuses_before_image_calls():
    from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_driver import (
        _artifixer_tuning, TaskEvaluationSceneConfigurationArtifixerError,
    )
    with pytest.raises(TaskEvaluationSceneConfigurationArtifixerError, match="training_policy_invalid"):
        _artifixer_tuning({"artifixer_training_policy": "typo"})
