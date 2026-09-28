"""SAM is editor input guidance, never a post-edit pixel boundary."""
from pathlib import Path

from blueprint_pipeline.task_evaluation_scene_configuration_semantic_locality import (
    EDITOR_OUTPUT_POLICY, valid_teacher_output_policy,
)
from tests.test_task_evaluation_scene_configuration_artifixer_selective_repair import _fixture


def test_full_editor_output_reaches_training_and_records_real_outside_changes(tmp_path):
    fixture = _fixture(tmp_path, preserve_editor_output=True)
    seal = fixture['locality']
    receipt = seal['receipt']
    assert receipt['policy'] == EDITOR_OUTPUT_POLICY
    assert valid_teacher_output_policy(receipt)
    assert receipt['all_editor_output_bytes_preserved_exactly'] is True
    assert receipt['all_non_target_source_pixels_preserved_exactly'] is False
    assert receipt['semantic_object_absence_review_passed'] is False
    assert receipt['generated_output_is_capture_or_physical_evidence'] is False
    for row in receipt['frames']:
        raw = Path(row['raw_semantic_teacher']['path'])
        staged = Path(seal['receipt_path']).parent / row['sealed_semantic_teacher']['relative_path']
        assert staged.read_bytes() == raw.read_bytes()
        assert row['inner_feather_radius_pixels'] == 0
    altered = receipt['frames'][1]
    assert altered['outside_exact_support_changed_pixels_after_seal'] == 62
    assert altered['non_target_source_pixels_preserved_exactly'] is False
    assert altered['deterministic_selective_repair_required'] is True


def test_editor_output_policy_requires_truthful_output_binding(tmp_path):
    receipt = _fixture(tmp_path, preserve_editor_output=True)['locality']['receipt']
    assert not valid_teacher_output_policy({**receipt, 'all_editor_output_bytes_preserved_exactly': False})
    assert not valid_teacher_output_policy({**receipt, 'status': 'semantic_teacher_exact_support_locality_sealed'})
