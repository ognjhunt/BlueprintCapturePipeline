"""Count the unfinished path without issuing grants or blocking retained work."""
from copy import deepcopy
import json

import pytest

from blueprint_pipeline.atlas_pricing import novel_view_headroom, require_headroom


def configuration(remaining=40):
    return {'remaining_requests': remaining, 'authority': {'maximum_requests': 40},
            'policy': {'maximum_review_cost_usd': 4, 'refined_splat_cost_usd': 5,
                       'mesh_export_cost_usd': .5}}


def forecast(tmp_path, config, count=1, cost=0.):
    return novel_view_headroom(config,
        batches=[(tmp_path / 'generated' / str(i), cost) for i in range(count)],
        final_output_root=tmp_path, context={'context_digest': 'context'}, source_digest='source')


@pytest.mark.parametrize('remaining,allowed', [(2, False), (5, False), (6, True), (40, True)])
def test_forecast_includes_review_surfaces_and_final_outputs(tmp_path, remaining, allowed):
    report = forecast(tmp_path, configuration(remaining))
    assert report['minimum_remaining_requests'] == 6
    assert report['remaining_requests'] == remaining
    assert report['sufficient_for_known_calls'] is allowed
    assert report['reservation_created'] is False
    assert report['additional_mask_or_retry_calls'] == 'not_yet_known'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('remaining', [None, True, -1, 2.5, '40'])
def test_unknown_or_malformed_remaining_allowance_cannot_authorize_new_views(tmp_path, remaining):
    with pytest.raises(ValueError, match='remaining_requests_invalid'):
        forecast(tmp_path, configuration(remaining))


def test_batches_are_counted_from_the_actual_plan_and_output_policy(tmp_path):
    config = configuration(7)
    assert forecast(tmp_path, config, count=3)['minimum_remaining_requests'] == 8
    del config['policy']['refined_splat_cost_usd']
    assert forecast(tmp_path, config, count=3)['minimum_remaining_requests'] == 7


def test_retained_submissions_reconcile_even_with_zero_remaining_requests(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_context

    submission = tmp_path / 'generated/0/atlas_pose/submission.json'
    submission.parent.mkdir(parents=True)
    submission.write_text(json.dumps({'status': 'submitted', 'operation_id': 'retained'}))
    config = {**configuration(0), 'schema_version': 'website_atlas_configuration.v1'}
    monkeypatch.setattr(website_task_context, 'load_atlas_configuration',
                        lambda *a, **kw: pytest.fail('Forecast blocked a retained submission'))
    before = submission.read_bytes()
    assert forecast(tmp_path, config) is None
    assert submission.read_bytes() == before


def test_completed_and_admitted_stages_are_not_counted_a_second_time(tmp_path):
    for relative, value in [
            ('generated/0/result.json', {'status': 'completed'}),
            ('refined/atlas_pose/submission.json', {'status': 'submitted',
                'binding': {'task': 'posedImages2Splats'}}),
            ('mesh/atlas_pose/submission.json', {'status': 'completed',
                'binding': {'task': 'splats2Mesh'}})]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
    report = forecast(tmp_path, configuration(5), count=2)
    assert report['minimum_remaining_requests'] == 4
    assert report['unfinished_stages']['novel_view_batches'] == 1
    assert 'refined_splat' not in report['unfinished_stages']
    assert 'mesh_export' not in report['unfinished_stages']


def test_production_forecast_reads_current_noncreating_allowance_after_planning(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_context

    config = {**configuration(40), 'schema_version': 'website_atlas_configuration.v1'}
    current = {**deepcopy(config), 'remaining_requests': 2}
    calls = []
    def read(context, source_digest, *, create):
        calls.append((context, source_digest, create))
        return current
    monkeypatch.setattr(website_task_context, 'load_atlas_configuration', read)
    report = forecast(tmp_path, config)
    assert calls == [({'context_digest': 'context'}, 'source', False)]
    assert report['remaining_requests'] == 2 and not report['sufficient_for_known_calls']
    assert config['remaining_requests'] == 40
    # The owner raising this grant's limits continues the job; any other change stops it.
    current['authority']['maximum_requests'] = 400
    assert forecast(tmp_path, config)['sufficient_for_known_calls'] is False
    current['authority']['provider_retention'] = 'project_library_retention_approved'
    with pytest.raises(ValueError, match='authority_or_policy_changed'):
        forecast(tmp_path, config)


def test_dollars_and_requests_share_one_forecast_without_counting_bought_work_twice(tmp_path):
    config = {**configuration(40), 'remaining_cost_usd': 7.5}
    # Two $.5 batches + refined splat $5 + mesh $.5 + one review margin = $7.5.
    report = forecast(tmp_path, config, count=2, cost=.5)
    assert report['minimum_remaining_cost_usd'] == 7.5 and report['sufficient_for_known_calls']
    config['remaining_cost_usd'] = 7.49
    with pytest.raises(ValueError, match='remaining_budget_insufficient'):
        require_headroom(forecast(tmp_path, config, count=2, cost=.5), tmp_path / 'forecast.json')
    # An in-flight batch is already held by admission; a bought splat is already paid.
    for relative, value in [('generated/0/atlas_pose/submission.json', {'status': 'submitted'}),
                            ('refined/atlas_pose/submission.json', {'status': 'submitted',
                                'binding': {'task': 'posedImages2Splats'}})]:
        (tmp_path / relative).parent.mkdir(parents=True)
        (tmp_path / relative).write_text(json.dumps(value))
    config['remaining_cost_usd'] = 2.
    assert forecast(tmp_path, config, count=2, cost=.5)['minimum_remaining_cost_usd'] == 2


@pytest.mark.parametrize('change,allowed', [
    ({'maximum_requests': 60}, True),
    ({'maximum_requests': 60, 'maximum_cost_usd': 50, 'approval_reference': 'owner raise',
      'expires_at_epoch': 2, 'authorization_digest': 'sha256:amended'}, True),
    ({'maximum_requests': None}, True),
    ({'maximum_requests': 39}, False),
    ({'maximum_cost_usd': '40'}, False),
    ({'maximum_cost_usd': 39}, False),
    ({'maximum_requests': 60, 'source_video_digest': 'other'}, False),
    ({'maximum_requests': 60, 'provider_training_authorized': True}, False),
    ({'maximum_requests': 60, 'planner': {'provider': 'other'}}, False),
])
def test_owner_may_raise_a_running_jobs_limits_but_not_change_its_grant(change, allowed):
    from blueprint_pipeline.website_atlas_pipeline import resumed_policy

    authority = {'request_id': 'r', 'capture_id': 'c', 'source_video_digest': 'v',
                 'provider_training_authorized': False, 'planner': {'provider': 'openai'},
                 'approval_reference': 'owner', 'expires_at_epoch': 1, 'authorization_digest': 'sha256:a',
                 'maximum_cost_usd': 40, 'maximum_requests': 40}
    policy = {'maximum_review_cost_usd': 1, 'pose_input_cost_usd': .02}
    stored = {'authority': authority, 'policy': policy}
    current = {'authority': {k: v for k, v in {**authority, **change}.items() if v is not None},
               'policy': dict(policy)}
    if allowed:
        assert resumed_policy(stored, current) == {'policy': policy}
    else:
        with pytest.raises(ValueError, match='authority_or_policy_changed'):
            resumed_policy(stored, current)
