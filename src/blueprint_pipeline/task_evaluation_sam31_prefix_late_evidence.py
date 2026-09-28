"""Read-only review, mask, freeze and contribution joins for partial SAM reuse."""
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_sam31_prefix_evidence import reuse_verdict
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, require


def validate_late_prefix(artifacts, *, phase_count):
    """Use the downstream consumer's validators; never reconstruct a paid stage."""
    return reuse_verdict("sam31_prefix_late", (canonical_digest(artifacts), phase_count), artifacts,
                         lambda: _validate_late_prefix(artifacts, phase_count=phase_count))


def _validate_late_prefix(artifacts, *, phase_count):
    if phase_count < 6:
        return
    required = {'selection_inputs', 'task_selection', 'track_selection_review', 'review_execution'}
    if phase_count >= 7:
        required.add('calibrated_mask_set')
    if phase_count >= 8:
        required.update(('segment_sweep_freeze', 'standard_splat'))
    if phase_count >= 9:
        required.add('gaussian_contribution_evidence')
    if phase_count >= 10:
        required.add('segment_cutout_set')
    require(required.issubset(artifacts), 'sam31_adoption_late_artifacts_missing')
    from .public_scene_sam31_track_selection_review import (
        AI_RECEIPT_SCHEMA_VERSION, load_validated_sam31_track_selection_inputs,
        validate_sam31_track_selection_review,
    )
    from .task_evaluation_scene_configuration_sam31_inputs import (
        _cutout, validate_reviewed_mask_inputs, validate_retained_sweep,
        validate_retained_contribution,
    )

    def path(name):
        record = artifacts[name]
        return checked_file(record["path"], record)

    freezes, inputs, selected = load_validated_sam31_track_selection_inputs(path("selection_inputs"))
    require(freezes == [str(path("task_selection"))], "sam31_adoption_review_task_changed")
    review = validate_sam31_track_selection_review(receipt_path=path("track_selection_review"),
        task_freeze_paths=freezes, task_inputs=inputs, selected_track_ids_by_task=selected)
    require(review["schema_version"] == AI_RECEIPT_SCHEMA_VERSION,
            "sam31_adoption_independent_review_required")
    execution = review["review_execution_receipt"]
    require(all(execution[key] == artifacts["review_execution"][key]
                for key in ("path", "sha256", "size_bytes")), "sam31_adoption_review_execution_changed")
    if phase_count < 7:
        return
    task = read(freezes[0], digest_field="task_freeze_digest")
    _, _, camera_ids, _, mask_paths = validate_reviewed_mask_inputs(
        mask_set_path=path("calibrated_mask_set"), review=review, review_kind="ai", task=task,
        inputs=inputs, selected=selected, minimum_views=16,
        source_object={"publisher_instance_id": task["source_object"]["instance_id"]})
    if phase_count < 8:
        return
    sweep_ref = artifacts["segment_sweep_freeze"]
    frozen = read(path("segment_sweep_freeze"), digest_field="freeze_digest")
    source_ref = frozen["source_standard_splat"]
    source = checked_file(source_ref["path"], source_ref)
    require(all(source_ref[key] == artifacts["standard_splat"][key]
                for key in ("sha256", "size_bytes")), "sam31_adoption_freeze_source_changed")
    sweep, original = validate_retained_sweep(sweep_reference=sweep_ref, source=source,
        task=task, camera_ids=camera_ids, mask_paths=mask_paths)
    if phase_count < 9:
        return
    validate_retained_contribution(manifest_reference=artifacts["gaussian_contribution_evidence"],
        sweep=sweep, original=original, source=source, camera_ids=camera_ids)
    if phase_count >= 10:
        candidate_path = path("segment_cutout_set")
        candidate = read(candidate_path, digest_field="receipt_digest")
        _cutout(Path(candidate_path), candidate, source, task, camera_ids, mask_paths)
