"""A completion whose survey and aiming no step can finish takes the floor from counts, before anything is bought.

Austin (itest-cce2beee, 2026-11-25), a room kept as filmed: 204 posed views, 121 reference views, 201 survey cameras
and a 16.2 M Gaussian draft. Its survey and aiming took 78 minutes offline with one BLAS thread, and the host runs 2.4
to 3 times slower than that and ends a step 65 minutes after its unit starts: every retry began the survey again.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from blueprint_pipeline import atlas_completion_work as work
from blueprint_pipeline.atlas_automation_plan import COMPLETION_FLOOR_ERRORS
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

PORTRAIT = {'width': 720, 'height': 1280, 'fx': 602.3, 'fy': 602.3, 'cx': 360., 'cy': 640.}


def _view(index, intrinsics=PORTRAIT):
    position = np.array([.05 * index, 0., .02 * index])
    return {'frame_id': f'decoded-{index:09d}', 'intrinsics': dict(intrinsics), 'position': position,
            'extrinsics': {'position': position.tolist(), 'quaternion': [0., 0., 0., 1.]}}


def _camera(view, *, offset=0.):
    return {'camera_id': view['frame_id'] + '-wide', 'camera': {'intrinsics': dict(view['intrinsics']),
            'extrinsics': {'position': (view['position'] + offset).tolist(), 'quaternion': [0., 0., 0., 1.]}}}


def _export(tmp_path, gaussians):
    root = tmp_path / 'splat'
    root.mkdir(parents=True)
    (root / 'atlas_depth_splat_manifest.json').write_text(json.dumps({'gaussian_count': gaussians}))
    return root


def _austin(tmp_path):
    views = [_view(index) for index in range(204)]
    references = [view for index, view in enumerate(views) if index % 5 not in (1, 3)][:121]
    return dict(root=tmp_path / 'completion', frames=views, photographs=references, geometry_frames=views,
                candidates=[_camera(view) for view in views[3:]], splat_root=_export(tmp_path, 16_195_148))


def _lamp_size(tmp_path):
    """A captured object's room (itest-314bd4bc, 30 views): its whole camera pool of 281 surveyed, no export."""
    landscape = {'width': 1280, 'height': 720, 'fx': 960., 'fy': 960., 'cx': 640., 'cy': 360.}
    views = [_view(index, landscape) for index in range(30)]
    at_centres = [_camera(view) for view in views for _ in range(5)]  # wide, left, right, lower, higher
    away = [_camera(view, offset=.3) for view in views[:29]] + [_camera(views[i % 30], offset=.5 + i)
                                                              for i in range(102)]
    return dict(root=tmp_path / 'completion', frames=views, photographs=views, geometry_frames=views,
                candidates=at_centres + away)


def test_austin_takes_the_floor_before_its_survey_from_counts_alone(tmp_path):
    estimate = work.CompletionWork(**_austin(tmp_path))
    with pytest.raises(ValueError, match=f'^{work.WORK_EXCEEDS_STEP}$'):
        estimate.admit_survey()
    assert work.WORK_EXCEEDS_STEP in COMPLETION_FLOOR_ERRORS
    record = json.loads((tmp_path / 'completion' / work.RECORD).read_text())
    assert record['digest'] == canonical_digest(record, digest_field='digest')
    assert record['stage'] == 'before_gap_survey' and record['basis'] == 'counts_times_measured_unit_costs_not_the_clock'
    assert record['counts'] == {'views': 204, 'photographs': 121, 'geometry_views': 204, 'survey_cameras': 201,
                                'gaussians': 16_195_148, 'scored_placements': 0, 'aimed_survey_cameras': 0}
    # Past both what a step can finish and what a later step could do again before its planner, before a placement
    # is scored; and the same answer every time: nothing here reads the clock.
    available = record['available_reference_seconds']
    assert record['reference_seconds'] > 1.5 * available['whole_pass']
    assert record['repeated_reference_seconds'] > available['repeated_before_planner']
    assert work.CompletionWork(**_austin(tmp_path / 'again')).phases == estimate.phases


def test_a_normal_room_kept_as_filmed_is_well_inside_what_a_step_can_do(tmp_path):
    """30 views with a 3 M Gaussian export, every survey camera at a recorded centre, 40 gaps aimed at."""
    views = [_view(index) for index in range(30)]
    estimate = work.CompletionWork(root=tmp_path / 'completion', frames=views, photographs=views,
                                   geometry_frames=views, candidates=[_camera(view) for view in views],
                                   splat_root=_export(tmp_path, 3_000_000))
    estimate.admit_survey()
    for _ in range(40):
        estimate.admit_placements(placements=30, samples=30 * 14400)
    estimate.admit_aimed_survey([_camera(views[0], offset=1. + i) for i in range(40)])
    assert estimate.total() < .3 * work.step_seconds()['whole_pass']


def test_a_lamp_size_room_with_a_captured_object_fits(tmp_path):
    """itest-314bd4bc's room: its whole pool of 281 cameras surveyed from the views (no export), then 53 aimed. On the
    host that pass took 49 minutes from its field-of-view audit to its planner's inputs: inside a step's 65."""
    estimate = work.CompletionWork(**_lamp_size(tmp_path))
    assert estimate.counts['survey_cameras'] == 281
    estimate.admit_survey()
    for _ in range(20):  # its aiming took 3.3 minutes on the host
        estimate.admit_placements(placements=27, samples=30 * 14400)
    estimate.admit_aimed_survey([_camera(estimate.frames[i % 30], offset=1. + i) for i in range(53)])
    assert estimate.fits() and not (tmp_path / 'completion' / work.RECORD).exists()


def test_placement_scoring_no_step_can_finish_takes_the_floor_before_the_next_gap_is_scored(tmp_path):
    estimate = work.CompletionWork(**_lamp_size(tmp_path))
    estimate.admit_survey()
    scored = 0
    with pytest.raises(ValueError, match=f'^{work.WORK_EXCEEDS_STEP}$'):
        while True:
            estimate.admit_placements(placements=300, samples=204 * 14400)
            scored += 1
    record = json.loads((tmp_path / 'completion' / work.RECORD).read_text())
    assert record['stage'] == 'before_placement_scoring' and record['counts']['scored_placements'] == 300 * (scored + 1)
    # The count it stops at is the same every time.
    again = work.CompletionWork(**_lamp_size(tmp_path / 'again'))
    again.admit_survey()
    for _ in range(scored):
        again.admit_placements(placements=300, samples=204 * 14400)
    with pytest.raises(ValueError, match=f'^{work.WORK_EXCEEDS_STEP}$'):
        again.admit_placements(placements=300, samples=204 * 14400)


def test_the_budget_is_what_a_step_can_finish_and_reach_its_planner_in_at_the_hosts_pace():
    from blueprint_pipeline.website_atlas_preview import STEP_END_SECONDS, STEP_START_BUDGET_SECONDS

    assert work.step_seconds() == {'whole_pass': STEP_END_SECONDS / work.HOST_SLOWDOWN,
                                   'repeated_before_planner': STEP_START_BUDGET_SECONDS / work.HOST_SLOWDOWN}
    assert (STEP_END_SECONDS, STEP_START_BUDGET_SECONDS) == (3900, 2700)
    assert work.HOST_SLOWDOWN >= 38 / 15.7  # Austin's fusion: 38 minutes on the host, 15.7 here


def test_the_completion_checks_before_its_survey_and_never_asks_the_planner(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_context_fov, atlas_scene_gaps
    from blueprint_pipeline import atlas_scene_completion as completion
    from tests.test_atlas_scene_completion import frame

    monkeypatch.setattr(atlas_context_fov, 'verify_pose_context_fov', lambda **_: {'digest': 'audit'})
    monkeypatch.setattr(atlas_scene_gaps, 'survey_gaps', lambda **_: pytest.fail('surveyed'))
    monkeypatch.setattr(work, 'step_seconds', lambda: {'whole_pass': 0., 'repeated_before_planner': 0.})
    review = {'subject_origin': 'no_manipulated_subject', 'automation': {'objects': [], 'removal_instruction': ''}}
    with pytest.raises(ValueError, match=f'^{work.WORK_EXCEEDS_STEP}$'):
        completion.complete_scene(pose_root=tmp_path, pose={'digest': 'pose', 'binding': {
            'source_video_digest': 'source'}}, frames=[frame(), frame('b', .2)], review=review,
            root=tmp_path / 'completion', context={'context_digest': 'task'}, configuration={},
            task_runner=lambda **_: pytest.fail('bought'), planner_runner=lambda **_: pytest.fail('planned'),
            release_blockers=lambda: pytest.fail('released'), max_polls=1, context_basis='preserved_capture')
    assert json.loads((tmp_path / 'completion' / work.RECORD).read_text())['stage'] == 'before_gap_survey'


def test_a_pass_that_reached_its_planner_or_a_coverage_pass_is_not_bounded(tmp_path, monkeypatch):
    """Such a pass did its survey in a step already, and may hold a paid answer; a coverage pass has no floor."""
    from blueprint_pipeline import atlas_context_fov, atlas_scene_gaps
    from blueprint_pipeline import atlas_scene_completion as completion
    from tests.test_atlas_scene_completion import frame

    class Surveyed(Exception):
        pass

    def survey(**_):
        raise Surveyed
    monkeypatch.setattr(atlas_context_fov, 'verify_pose_context_fov', lambda **_: {'digest': 'audit'})
    monkeypatch.setattr(atlas_scene_gaps, 'survey_gaps', survey)
    monkeypatch.setattr(work, 'step_seconds', lambda: {'whole_pass': 0., 'repeated_before_planner': 0.})
    review = {'subject_origin': 'no_manipulated_subject', 'automation': {'objects': [], 'removal_instruction': ''}}
    args = dict(pose_root=tmp_path, pose={'digest': 'pose', 'binding': {'source_video_digest': 'source'}},
                frames=[frame(), frame('b', .2)], review=review, context={'context_digest': 'task'},
                configuration={}, task_runner=None, planner_runner=None, release_blockers=None, max_polls=1,
                context_basis='preserved_capture')
    (tmp_path / 'planned').mkdir()
    (tmp_path / 'planned/plan-inputs-source-0123456789abcdef.json').write_text('{}')
    for root, extra in ((tmp_path / 'planned', {}), (tmp_path / 'coverage', {'coverage_feedback': {}})):
        with pytest.raises(Surveyed):
            completion.complete_scene(root=root, **args, **extra)
        assert not (root / work.RECORD).exists()
