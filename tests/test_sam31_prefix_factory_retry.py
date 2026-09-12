"""The real attempt factory must pass an exact prefix to the retry reader."""
import inspect
import json

import pytest

from blueprint_pipeline import task_evaluation_public_scene_attempt_factory as factory
from blueprint_pipeline import task_evaluation_sam31_prefix_adoption as adoption
from tests.test_task_evaluation_public_scene_attempt_factory import context as context, write, ref


class RetainedReaderReached(Exception):
    pass


@pytest.mark.parametrize('candidate_phase', ['sam31_tracking', 'contribution_sweep', 'segment_cutout'])
def test_factory_retry_supplies_the_sealed_adoption_phase(context, tmp_path, monkeypatch, candidate_phase):
    args, _ = context
    first = factory.materialize_public_scene_attempt(**args)
    plan = args['output_root'] / 'submission/configuration/sam31_preparation_plan.v1.json'
    profile = args['output_root'] / 'sam31_preparation_profile.json'
    candidate = {'source_plan': ref(plan), 'source_profile': ref(profile),
                 'parent_request_digest': 'sha256:' + 'a' * 64}
    machinery = json.loads(args['machinery_path'].read_text())
    machinery.update(child_queue_root=str(tmp_path / 'children'), parent_queue_root=str(tmp_path / 'parents'),
        execution_root=str(tmp_path / 'executions'), release_retention_binding_root=str(tmp_path / 'pins'))
    path = write(tmp_path / 'retry-machinery.json', machinery, 'machinery_digest')
    output = tmp_path / 'retry-factory'
    output.mkdir()
    sealed = output / 'completed_prefix_adoption.json'
    write(sealed, {'synthetic_signature_probe_only': True, 'through_phase': 'sam31_tracking',
        'source_plan': candidate['source_plan'], 'source_profile': candidate['source_profile'],
        'original_parent_request_digest': candidate['parent_request_digest']}, 'adoption_digest')
    before = sealed.read_bytes()
    monkeypatch.setattr(factory, '_prefix_candidates', lambda *args: [candidate])
    zero = write(tmp_path / 'observation.json', {'synthetic_signature_probe_only': True})
    monkeypatch.setattr('blueprint_pipeline.task_evaluation_prefix_observation.selection_observation',
                        lambda root: (zero, 1001.))
    monkeypatch.setattr(adoption, 'select_completed_prefix_adoption', lambda **kwargs: {
        'status': 'reusable_prefix_selected', 'through_phase': candidate_phase})
    signature = inspect.signature(adoption.materialize_completed_prefix_adoption)
    def exact_reader_boundary(**kwargs):
        # No scientific validation is faked as successful: stop at the real
        # materializer's signature boundary, after the actual factory branch.
        signature.bind(**kwargs)
        assert kwargs['through_phase'] == 'sam31_tracking'
        assert kwargs['expected_source_commit'] == first['source_commit']
        raise RetainedReaderReached
    monkeypatch.setattr(adoption, 'materialize_completed_prefix_adoption', exact_reader_boundary)
    with pytest.raises(RetainedReaderReached):
        factory.materialize_public_scene_attempt(**{**args, 'machinery_path': path, 'output_root': output})
    assert sealed.read_bytes() == before


def test_factory_does_not_swallow_verified_pending_billing(context, tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_sam31_prefix_billing import Sam31PrefixBillingPending
    args, _ = context
    factory.materialize_public_scene_attempt(**args)
    candidate = {'source_plan': ref(args['output_root'] / 'submission/configuration/sam31_preparation_plan.v1.json'),
                 'source_profile': ref(args['output_root'] / 'sam31_preparation_profile.json'),
                 'parent_request_digest': 'sha256:' + 'a' * 64}
    machinery = json.loads(args['machinery_path'].read_text())
    machinery.update(child_queue_root=str(tmp_path / 'children'), parent_queue_root=str(tmp_path / 'parents'),
        execution_root=str(tmp_path / 'executions'), release_retention_binding_root=str(tmp_path / 'pins'))
    path = write(tmp_path / 'pending-machinery.json', machinery, 'machinery_digest')
    output = tmp_path / 'pending-factory'
    monkeypatch.setattr(factory, '_prefix_candidates', lambda *args: [candidate, candidate])
    zero = write(tmp_path / 'observation.json', {'synthetic_signature_probe_only': True})
    monkeypatch.setattr('blueprint_pipeline.task_evaluation_prefix_observation.selection_observation', lambda root: (zero, 1001.))
    calls = []
    def pending(**kwargs):
        calls.append(kwargs)
        raise Sam31PrefixBillingPending('sam31_adoption_official_billing_pending')
    monkeypatch.setattr(adoption, 'select_completed_prefix_adoption', pending)
    with pytest.raises(Sam31PrefixBillingPending):
        factory.materialize_public_scene_attempt(**{**args, 'machinery_path': path, 'output_root': output})
    assert len(calls) == 1
    assert not (output / 'prefix_selection.json').exists()
    assert not (output / 'sam31_preparation_profile.json').exists()
    assert not (output / 'submission').exists()
