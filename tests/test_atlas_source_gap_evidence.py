"""Source visibility, empty space and missing depth are separate decisions."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image

import pytest

from blueprint_pipeline.atlas_source_gap_evidence import combine_source_surveys, repair_depth_outcomes, source_actions, source_gallery
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def entry(visibility, ids):
    return [{'camera_id': 'work-surface-view', 'gap_ids': ['gap-1'], 'source_evidence': [
        {'gap_id': 'gap-1', 'visibility': visibility, 'source_frame_ids': ids,
         'reason': 'Evidence from the recorded work area.'}]}]


def test_source_recovery_precedes_generation_and_open_space_stays_empty():
    gallery = {'reviewed_source_frame_ids': ['recorded-a', 'recorded-b'],
               'conditioned_frame_ids': ['recorded-a']}
    needed = source_actions(entry('observed_elsewhere_in_capture', ['recorded-b']), gallery)
    assert needed['additional_source_frame_ids'] == ['recorded-b']
    opening = source_actions(entry('physical_opening', ['recorded-a']), gallery)
    assert opening['physical_opening_gap_ids'] == {'work-surface-view': ['gap-1']}
    assert source_actions(entry('uncertain', []), gallery)['unresolved_gap_ids'] == ['gap-1']
    observed = source_actions(entry('observed_in_conditioning', ['recorded-a']), gallery)
    assert not observed['additional_source_frame_ids']
    assert observed['source_coverage_is_not_depth_coverage'] is True


@pytest.mark.parametrize('change', ['missing_gap', 'invented_frame', 'unconditioned', 'no_observation', 'empty_gaps'])
def test_source_evidence_cannot_be_fabricated_or_omitted(change):
    gallery = {'reviewed_source_frame_ids': ['a', 'b'], 'conditioned_frame_ids': ['a']}
    rows = deepcopy(entry('observed_in_conditioning', ['a']))
    if change == 'missing_gap':
        rows[0]['source_evidence'] = []
    if change == 'invented_frame':
        rows[0]['source_evidence'][0]['source_frame_ids'] = ['imagined']
    if change == 'unconditioned':
        rows[0]['source_evidence'][0]['source_frame_ids'] = ['b']
    if change == 'no_observation':
        rows[0]['source_evidence'][0]['source_frame_ids'] = []
    if change == 'empty_gaps':
        rows[0].update(gap_ids=[], source_evidence=[])
    with pytest.raises(ValueError, match='atlas_source_gap_'):
        source_actions(rows, gallery)


def test_absent_full_capture_survey_refuses_before_any_generation(tmp_path):
    with pytest.raises(ValueError, match='capture_survey_required'):
        source_gallery(survey_path=None, source_video_digest='video',
                       conditioned_frame_ids=['selected'], root=tmp_path)


def test_partial_depth_failure_is_visible_even_when_another_repair_succeeds(tmp_path):
    mask = tmp_path / 'gap.png'
    Image.new('L', (4, 4), 255).save(mask)
    plan = {'prepared_views': [{'frame_id': key, 'completion_gap_ids': [key + '-gap'],
        'completion_gap_mask_path': str(mask), 'completion_gap_mask_digest': _sha256_file(mask)}
        for key in ('front', 'side')]}
    plan['digest'] = canonical_digest(plan, digest_field='digest')
    path = tmp_path / 'rgbd-completion-plan.json'
    path.write_text(json.dumps(plan))
    report = {'generation': {'mask_plan_digest': plan['digest'], 'preservation_paths': [str(path)]}}
    frames = [{'frame_id': 'completion-root-front', 'depth': np.full((4, 4), np.nan),
               'confidence': np.zeros((4, 4))},
              {'frame_id': 'completion-root-side', 'depth': np.full((4, 4), 2.),
               'confidence': np.ones((4, 4))}]
    outcomes = repair_depth_outcomes(frames, report)
    assert outcomes[0]['native_depth_available_in_gap'] == 0
    assert outcomes[0]['admitted_depth_in_gap'] == 0
    assert outcomes[1]['native_depth_available_in_gap'] == outcomes[1]['admitted_depth_in_gap'] == 16
    assert len(outcomes) == 2, 'One successful camera cannot hide another camera\'s missing depth'


def test_nested_source_evidence_is_accepted_by_the_real_agents_schema():
    from agents import AgentOutputSchema
    from blueprint_pipeline.atlas_scene_planner import CoverageCompletionProposal

    AgentOutputSchema(CoverageCompletionProposal)


def test_dense_task_interval_does_not_hide_later_video_views(tmp_path):
    original = tmp_path / 'source.png'
    Image.new('RGB', (12, 8), 'white').save(original)
    def survey(name, rows):
        value = {'schema_version': 'atlas_video_survey.v1', 'source_video_digest': 'video',
                 'display_rotation_degrees': 0, 'heldout_pixels_included': False,
                 'frames': [{'frame_id': key, 'timestamp_seconds': timestamp,
                             'source_image_path': str(original), 'source_image_digest': _sha256_file(original)}
                            for key, timestamp in rows]}
        value['digest'] = canonical_digest(value, digest_field='digest')
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path
    full = survey('full.json', [('first', 0), ('later-angle', 20)])
    dense = survey('dense.json', [('first', 0), ('task-closeup', 1)])
    path = combine_source_surveys(survey_paths=[full, dense], output=tmp_path / 'combined.json')
    gallery, previews = source_gallery(survey_path=path, source_video_digest='video',
                                      conditioned_frame_ids=['first'], root=tmp_path / 'previews')
    assert gallery['reviewed_source_frame_ids'] == ['first', 'task-closeup', 'later-angle']
    assert len(previews) == 3
    assert gallery['candidate_gallery_does_not_prove_every_video_frame_was_reviewed'] is True
    Image.new('RGB', (12, 8), 'black').save(original)
    with pytest.raises(ValueError, match='capture_image_changed'):
        source_gallery(survey_path=path, source_video_digest='video', conditioned_frame_ids=['first'],
                       root=tmp_path / 'changed-previews')
