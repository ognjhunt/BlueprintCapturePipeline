"""Real read-only late-prefix science on small outputs from the CPU lifecycle."""
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_sam31_prefix_late_evidence import validate_late_prefix
from tests.test_task_evaluation_sam31_preparation_lifecycle import (
    test_real_cpu_sam_review_mask_freeze_cutout_lifecycle as run_synthetic_lifecycle,
)


@pytest.fixture
def retained(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_sam31_preparation_review_stages as reviews
    from blueprint_pipeline import task_evaluation_sam31_preparation_cpu_stages as cpu
    from tests.test_sam31_provider_launch_packet import _profile
    from tests.test_sam31_prefix_adoption import write
    profile_root = tmp_path / 'full-provider-sources'
    profile_root.mkdir()
    packet, _, _ = _profile(profile_root)
    provider_ref = write(profile_root / 'profile.json', packet)
    real_cpu = cpu.execute_cpu_stage
    def with_provider_sources(job):
        if job['stage_id'] == 'sam31_inputs':
            job['inputs']['sam31_provider_profile'] = provider_ref
        return real_cpu(job)
    monkeypatch.setattr(cpu, 'execute_cpu_stage', with_provider_sources)
    real = reviews.execute_review_stage
    observed = {}
    def capture(job):
        outcome = real(job)
        if job['stage_id'] == 'segment_cutout':
            observed.update(job['inputs'])
            observed.update(outcome['artifacts'])
        return outcome
    monkeypatch.setattr(reviews, 'execute_review_stage', capture)
    run_synthetic_lifecycle(tmp_path, monkeypatch)
    assert observed['segment_cutout_set']
    # Everything after setup is read-only, using real receipt/science validators.
    monkeypatch.setattr(reviews, 'run_sam31_ai_visual_review',
        lambda **kwargs: pytest.fail('a retained-prefix reader must not invoke a model'))
    return observed


@pytest.mark.parametrize('phase_count', [6, 7, 8, 9, 10])
def test_each_completed_late_phase_reopens_real_science_without_writes(retained, phase_count):
    before = {Path(row['path']): Path(row['path']).read_bytes() for row in retained.values()}
    validate_late_prefix(retained, phase_count=phase_count)
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_prepared_sam_request_reopens_profile_and_every_frame(retained):
    from blueprint_pipeline.task_evaluation_sam31_prefix_evidence import validate_sam_inputs
    profile = {'artifact_references': {'sam31_provider_profile': retained['sam31_provider_profile']}}
    validate_sam_inputs(retained, profile, retained['sam31_provider_profile']['path'], 'a' * 40)


def test_resealed_mask_pixels_cannot_override_the_original_sam_union(retained):
    import json
    from PIL import Image
    from blueprint_pipeline.task_evaluation_sam31_prefix_adoption import record
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
    mask_set_path = Path(retained['calibrated_mask_set']['path'])
    mask_set = json.loads(mask_set_path.read_text())
    ref = mask_set['tasks'][0]['masks'][0]['mask']
    path = mask_set_path.parent / ref['relative_path']
    with Image.open(path) as image:
        pixels = image.convert('L')
    pixels.putpixel((0, 0), 255 if pixels.getpixel((0, 0)) == 0 else 0)
    pixels.save(path)
    ref.update({key: value for key, value in record(path).items() if key != 'path'})
    mask_set['receipt_digest'] = canonical_digest(mask_set, digest_field='receipt_digest')
    mask_set_path.write_text(canonical_json(mask_set))
    retained['calibrated_mask_set'] = record(mask_set_path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='exact_mask_changed'):
        validate_late_prefix(retained, phase_count=7)
    assert path.read_bytes() == before


def test_resealed_sweep_cannot_change_the_task_object(retained):
    import json
    from blueprint_pipeline.task_evaluation_sam31_prefix_adoption import record
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
    path = Path(retained['segment_sweep_freeze']['path'])
    sweep = json.loads(path.read_text())
    sweep['scene']['target_instance_id'] = 'different-object'
    sweep['freeze_digest'] = canonical_digest(sweep, digest_field='freeze_digest')
    path.write_text(canonical_json(sweep))
    retained['segment_sweep_freeze'] = record(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='sweep_source_join_invalid'):
        validate_late_prefix(retained, phase_count=8)
    assert path.read_bytes() == before


def test_prepared_sam_request_refuses_a_different_checkpoint(retained, tmp_path):
    import json
    from blueprint_pipeline.task_evaluation_sam31_prefix_evidence import validate_sam_inputs
    from tests.test_sam31_prefix_adoption import write
    old = retained['sam31_provider_profile']
    current = json.loads(Path(old['path']).read_text())
    current.update(checkpoint_digest='sha256:' + 'f' * 64, model_digest='sha256:' + 'f' * 64)
    changed = write(tmp_path / 'different-checkpoint.json', current, 'profile_digest')
    before = Path(old['path']).read_bytes()
    with pytest.raises(ValueError, match='sam31_adoption_model_changed'):
        validate_sam_inputs(retained, {'artifact_references': {'sam31_provider_profile': old}}, changed['path'], 'a' * 40)
    assert Path(old['path']).read_bytes() == before


def test_resealed_nonfinite_contribution_array_is_not_reusable(retained):
    import json
    import numpy as np
    from blueprint_pipeline.task_evaluation_sam31_prefix_adoption import record
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
    path = Path(retained['gaussian_contribution_evidence']['path'])
    manifest = json.loads(path.read_text())
    ref = manifest['repetitions'][0]
    array_path = path.parent / ref['relative_path']
    with np.load(array_path, allow_pickle=False) as archive:
        values = np.array(archive['per_view_class_contribution'], dtype=np.float64)
    values.flat[0] = np.nan
    np.savez(array_path, per_view_class_contribution=values)
    ref.update({key: value for key, value in record(array_path).items() if key != 'path'})
    manifest['manifest_digest'] = canonical_digest(manifest, digest_field='manifest_digest')
    path.write_text(canonical_json(manifest))
    retained['gaussian_contribution_evidence'] = record(path)
    before = array_path.read_bytes()
    with pytest.raises(ValueError, match='segment_cutout_repetition_invalid'):
        validate_late_prefix(retained, phase_count=9)
    assert array_path.read_bytes() == before
