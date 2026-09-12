"""Two-hop immutable provenance with explicit synthetic media/authority seams.

Real adoption recursion, task/source comparison, queue/parent identities and
file digests run. Existing topology fixtures replace scientific GPU closures;
the standing-authority and media validators have separate real coverage.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_sam31_prefix_adoption as adoption
from blueprint_pipeline import task_evaluation_sam31_preparation_review_authority as authority
from tests import test_restart_adoption_topology as fixture
from tests.test_restart_adoption_topology import topology as topology, A, B, C
from tests.test_sam31_prefix_adoption import prefix as prefix, write


def _case(topology, tmp_path, monkeypatch, fault=None):
    build, materialize, _calls = topology
    original_write = fixture.write
    def selection_with_source(path, value, field=None):
        if path.name == 'source_selections.json':
            task = adoption.record(path.parent / 'task.json')
            if fault == 'unrelated_origin_task':
                other = tmp_path / 'same-task-different-source.json'
                other.write_bytes(Path(task['path']).read_bytes())
                task = adoption.record(other)
            scene = {'synthetic_topology': True, 'source_evidence': {'task_request': task}}
            scene_ref = original_write(path.with_name('selected-scene.json'), scene, 'scene_freeze_digest')
            value = {**value, 'task_id': 'frozen-task', 'scene_selection': scene_ref,
                     'scene_freeze_digest': scene['scene_freeze_digest']}
        return original_write(path, value, field)
    monkeypatch.setattr(fixture, 'write', selection_with_source)
    first = build(A, 'calibrated_views')
    _, first_ref, _ = materialize(first, B, 'calibrated_views', 'first-adoption')
    tracker = build(B, 'sam31_tracking', first_ref)
    second, second_ref, _ = materialize(tracker, C, 'sam31_tracking', 'second-adoption')
    observed = adoption.validate_completed_prefix_adoption(second_ref['path'], expected_source_commit=C,
                                                           approved_roots=(tmp_path,))
    assert observed['selection_origin'] == {
        'task_request': first['plan']['host_inputs']['task_request'], 'source_commit': A,
        'source_plan': first['plan_ref'], 'source_profile': first['profile_ref'],
        'parent_request_digest': first['digest']}
    assert observed['record']['source_plan'] == tracker['plan_ref']
    assert observed['selection_origin']['task_request'] != tracker['plan']['host_inputs']['task_request']
    freeze_ref = observed['artifacts']['task_selection']
    if fault == 'different_freeze_path':
        copied = tmp_path / 'same-freeze-different-path.json'
        copied.write_bytes(Path(freeze_ref['path']).read_bytes())
        freeze_ref = adoption.record(copied)
    candidate = tmp_path / 'candidate.json'
    write(candidate, {'selection_bindings': [{'task_freeze': freeze_ref}]}, 'candidate_digest')
    task_ref = second['current_host_inputs']['task_request']
    if fault == 'unrelated_current_task':
        copied = tmp_path / 'same-current-task-different-path.json'
        copied.write_bytes(Path(task_ref['path']).read_bytes())
        task_ref = adoption.record(copied)
    terms = write(tmp_path / 'terms.json', {'synthetic_authority_boundary': True})
    standing = {'schema_version': authority.SCHEMA, 'source_commit': C, 'task_request': task_ref,
        'scene_intent_authority': {'synthetic_current_owner': True}, 'accepted_by': 'fixture-owner',
        'accepted_on': '2026-09-12', 'human_authority_reference': 'current-owner-only',
        'provider_terms_evidence': terms}
    standing_path = tmp_path / 'standing.json'
    write(standing_path, standing, 'authority_digest')
    monkeypatch.setattr(authority, 'validate_sam31_review_authority', lambda *a, **kw: deepcopy(standing))
    monkeypatch.setattr(authority.review, 'load_validated_sam31_track_selection_review_candidate',
                        lambda path: (Path(path), json.loads(Path(path).read_text())))
    monkeypatch.setattr('blueprint_pipeline.public_scene_removal_selection.validate_removal_task_selection', lambda v: v)
    # Only translate the production roots to this test's owned fixture root;
    # every real recursive identity/digest validator remains active.
    validate = adoption.validate_completed_prefix_adoption
    monkeypatch.setattr(adoption, 'validate_completed_prefix_adoption',
                        lambda path, **kw: validate(path, **{**kw, 'approved_roots': (tmp_path,)}))
    rights_calls = []
    def materialize_rights(**kwargs):
        rights_calls.append(kwargs)
        value = {'synthetic_rights_sealing_boundary': True, 'attestation_digest': 'sha256:' + 'f' * 64}
        Path(kwargs['output_path']).write_text(json.dumps(value))
        return value
    monkeypatch.setattr(authority.review, 'materialize_sam31_ai_visual_review_rights', materialize_rights)
    if fault == 'nested_adoption_bytes':
        path = Path(first_ref['path'])
        path.write_bytes(path.read_bytes() + b' ')
    if fault == 'original_task_bytes':
        path = Path(first['plan']['host_inputs']['task_request']['path'])
        path.write_bytes(path.read_bytes() + b' ')
    if fault == 'forged_origin_field':
        second['selection_origin'] = {'task_request': task_ref, 'source_commit': C}
        second_ref = write(Path(second_ref['path']), second, 'adoption_digest')
    binding = {'standing_authority': adoption.record(standing_path), 'completed_prefix_adoption': second_ref}
    return standing, standing_path, candidate, second_ref, binding, rights_calls


@pytest.mark.parametrize('route', ['resolve', 'validate'])
@pytest.mark.parametrize('fault', [None, 'forged_origin_field', 'unrelated_origin_task', 'unrelated_current_task',
                                   'different_freeze_path', 'nested_adoption_bytes', 'original_task_bytes'])
def test_two_hop_review_uses_exact_validated_selection_origin(topology, tmp_path, monkeypatch, route, fault):
    standing, path, candidate, adoption_ref, binding, calls = _case(topology, tmp_path, monkeypatch, fault)
    output = tmp_path / 'rights.json'
    def run():
        if route == 'resolve':
            return authority.resolve_sam31_review_rights(authority_path=path,
                task_request_path=standing['task_request']['path'], candidate_path=candidate,
                output_path=output, completed_prefix_adoption_path=adoption_ref['path'])
        return authority.validate_scene_review_binding(binding, candidate_path=candidate,
            accepted_by=standing['accepted_by'], accepted_on=standing['accepted_on'],
            human_authority_reference=standing['human_authority_reference'])
    if fault in (None, 'forged_origin_field'):
        result = run()
        if route == 'resolve':
            assert result == output and len(calls) == 1
            assert calls[0]['scene_owner_authority'] == binding
            derived = json.loads(output.with_suffix('.derivation.json').read_text())
            assert derived['completed_prefix_adoption'] == adoption_ref
            assert derived['task_request'] == standing['task_request']
        else:
            assert result == standing and not calls
    else:
        with pytest.raises(ValueError):
            run()
        assert not output.exists() and not calls


def test_submission_input_refusal_keeps_only_safe_typed_code():
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import SceneConfigurationSubmissionError
    from blueprint_pipeline.task_evaluation_sam31_preparation_review_stages import _failure_blocker
    code = 'scene_configuration_submission_input_digest_mismatch'
    assert code in _failure_blocker(SceneConfigurationSubmissionError(code))
    hidden = _failure_blocker(SceneConfigurationSubmissionError('Provider response contained secret fixture-key-value'))
    assert 'SceneConfigurationSubmissionError' in hidden
    assert 'fixture-key-value' not in hidden and 'Provider response' not in hidden
