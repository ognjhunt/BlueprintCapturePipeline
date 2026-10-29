"""Corrections preserve retained artifacts and share the original native limits."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from blueprint_pipeline import website_preparation_revision as revisions
from blueprint_pipeline import website_task_target_choice as choices
from blueprint_pipeline import website_preparation_authority as authority
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import checked_file
from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import record
from blueprint_pipeline.task_evaluation_scene_intake import SceneIntakeError
from tests.test_task_evaluation_scene_intake import request, stage, attempt


@pytest.fixture(autouse=True)
def trusted_issuer(monkeypatch):
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS", "webapp")


def preparation_request():
    value = request()
    value['task']['task_id'] = 'website-test-task'
    value['execution'].update(purpose='scene_preparation', policy_candidates=[])
    value['consent']['task_confirmed'] = False
    return value


def bind(tmp_path, monkeypatch, value, digest='a'):
    monkeypatch.setattr(choices, 'correction_binding_digest', lambda _: 'sha256:' + digest * 64)
    preparation = {'intake_request': copy.deepcopy(value), 'status': 'intake_ready'}
    preparation['digest'] = canonical_digest(preparation, digest_field='digest')
    return revisions.bind_preparation(preparation, output_root=revisions.workspace(tmp_path), capture_root=tmp_path)


def test_correction_retains_original_bytes_and_exact_authority(tmp_path, monkeypatch):
    value = preparation_request()
    base = tmp_path / 'pipeline' / 'website_scene_preparation'
    base.mkdir(parents=True)
    old = base / 'preparation.json'
    old.write_text(json.dumps({'original': True}))
    original = record(old)
    prepared = bind(tmp_path, monkeypatch, value)
    corrected = prepared['intake_request']
    assert corrected['submission_id'] != value['submission_id']
    assert revisions.capture_id(corrected) == value['submission_id']
    assert corrected['execution'] == value['execution']
    assert corrected['consent'] == value['consent']
    assert corrected['owner'] == value['owner']
    assert corrected['source'] == value['source']
    assert checked_file(original['path'], original) == old
    revised = record(revisions.workspace(tmp_path) / 'preparation.json')
    second = bind(tmp_path, monkeypatch, value, 'b')
    assert second['intake_request']['submission_id'] != corrected['submission_id']
    assert checked_file(revised['path'], revised).is_file()
    assert checked_file(original['path'], original).is_file()
    assert revisions.capture_root_for_preparation(revised['path']) == tmp_path
    assert revisions.capture_root_for_preparation(old) == tmp_path
    assert revisions.capture_root_for_preparation(base / 'revisions' / 'bad' / 'preparation.json') is None


@pytest.mark.parametrize('field', ['capture_id', 'corrections_digest', 'purpose', 'submission_id'])
def test_revision_identity_cannot_be_rebound(tmp_path, monkeypatch, field):
    value = bind(tmp_path, monkeypatch, preparation_request())['intake_request']
    if field == 'purpose':
        del value['execution']['purpose']
    elif field == 'submission_id':
        value['submission_id'] = 'forged'
    else:
        value['task']['subject'][revisions.FIELD][field] = 'other'
    with pytest.raises(ValueError, match='website_preparation_revision_invalid'):
        revisions.revision(value)


def test_both_original_and_corrected_intakes_reopen_current_authority(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(authority, 'require_retained_preparation_authority',
                        lambda **kw: calls.append(kw['request']) or kw['now'])
    original = preparation_request()
    first = stage(tmp_path / 'queue', original)
    corrected = bind(tmp_path, monkeypatch, original)['intake_request']
    second = stage(tmp_path / 'queue', corrected)
    assert stage(tmp_path / 'queue', corrected) == second
    assert first['intent_id'] != second['intent_id']
    assert original in calls and corrected in calls
    monkeypatch.setattr(authority, 'require_retained_preparation_authority',
                        lambda **_: (_ for _ in ()).throw(ValueError('source_revoked')))
    with pytest.raises(SceneIntakeError, match='scene_intake_preparation_authority_invalid'):
        attempt(tmp_path / 'queue', second)


def test_correction_does_not_reset_attempts_in_either_direction(tmp_path, monkeypatch):
    monkeypatch.setattr(authority, 'require_retained_preparation_authority', lambda **kw: kw['now'])
    original = preparation_request()
    first = stage(tmp_path / 'queue', original)
    corrected = bind(tmp_path, monkeypatch, original)['intake_request']
    second = stage(tmp_path / 'queue', corrected)
    old_attempt = attempt(tmp_path / 'queue', first, cost=1)
    attempt(tmp_path / 'queue', second, cost=1)
    for intent in (first, second):
        with pytest.raises(SceneIntakeError, match='attempt_cap_exhausted'):
            attempt(tmp_path / 'queue', intent, 'another', cost=1)
    old_path = tmp_path / 'queue' / first['intent_id'] / 'attempts/a1.json'
    assert json.loads(old_path.read_text()) == old_attempt


def test_correction_keeps_unknown_spend_and_serializes_sibling_reservations(tmp_path, monkeypatch):
    monkeypatch.setattr(authority, 'require_retained_preparation_authority', lambda **kw: kw['now'])
    original = preparation_request()
    original['execution']['max_paid_attempts'] = 3
    first = stage(tmp_path / 'queue', original)
    corrected = bind(tmp_path, monkeypatch, original)['intake_request']
    second = stage(tmp_path / 'queue', corrected)
    attempt(tmp_path / 'queue', first, cost=3)
    def reserve(intent):
        try:
            attempt(tmp_path / 'queue', intent, 'race', cost=1)
            return 'reserved'
        except SceneIntakeError as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, (first, second)))
    assert sorted(results) == ['reserved', 'scene_intake_spend_cap_exhausted']


def test_both_registrations_resolve_after_same_capture_correction(tmp_path, monkeypatch):
    from blueprint_pipeline import website_native_background as native
    from blueprint_pipeline import task_evaluation_scene_retirement_generations as births
    from blueprint_pipeline.website_scene_dispatch import register_website_preparation, resolve_website_source
    monkeypatch.setattr(native, 'construction_rights_admission', lambda **_: {})
    monkeypatch.setattr(native, 'prepare_construction_stages', lambda **_: {'references': []})
    monkeypatch.setattr(births, 'capture_birth_source_projection', lambda *_a, **_kw: None)
    capture = tmp_path / 'capture'
    base = capture / 'pipeline' / 'website_scene_preparation'
    base.mkdir(parents=True)
    original = preparation_request()
    context = {'capture_id': original['submission_id']}
    context['context_digest'] = canonical_digest(context, digest_field='context_digest')
    prepared = {'status': 'intake_ready', 'intake_request': original, 'binding': {'task_context_digest': context['context_digest']}}
    prepared['digest'] = canonical_digest(prepared, digest_field='digest')
    (base / 'preparation.json').write_text(json.dumps(prepared))
    def register(root):
        (root / 'runtime.json').write_text('{}')
        (root / 'task_context.json').write_text(json.dumps(context))
        return register_website_preparation(preparation_path=root / 'preparation.json',
            runtime_inputs_path=root / 'runtime.json', task_context_path=root / 'task_context.json',
            root=tmp_path / 'bindings', now=100)
    old = register(base)
    monkeypatch.setattr(choices, 'correction_binding_digest', lambda _: 'sha256:' + 'a' * 64)
    root = revisions.workspace(capture)
    corrected = revisions.bind_preparation(prepared, output_root=root, capture_root=capture)['intake_request']
    new = register(root)
    assert old != new
    assert register(root) == new
    for index, request_value in enumerate((original, corrected)):
        result = resolve_website_source(intent={'request': request_value, 'intent_id': f'scene-{index}',
            'intent_digest': 'sha256:' + str(index) * 64,
            'task_content_digest': canonical_digest(request_value['task'])}, config={
                'website_source_binding_root': str(tmp_path / 'bindings'),
                'factory_output_root': str(tmp_path / 'factory'),
                'website_source_machinery_path': str(tmp_path / 'machinery.json')})
        assert result.status == 'resolved'


def test_cross_runtime_revision_vector():
    from pathlib import Path
    vector = json.loads((Path(__file__).parent / 'fixtures/website-preparation-revision.v1.json').read_text())
    assert revisions.revision(vector['original']) is None
    assert revisions.capture_id(vector['corrected']) == vector['original']['submission_id']
    assert vector['corrected']['execution'] == vector['original']['execution']
    assert vector['corrected']['consent'] == vector['original']['consent']


def test_confirmed_revision_still_reopens_current_preparation_grant(tmp_path, monkeypatch):
    value = preparation_request()
    value['consent']['task_confirmed'] = True
    corrected = bind(tmp_path, monkeypatch, value)['intake_request']
    calls = []
    monkeypatch.setattr(authority, 'require_retained_preparation_authority',
                        lambda **kw: calls.append(kw['request']) or kw['now'])
    receipt = stage(tmp_path / 'queue', corrected)
    attempt(tmp_path / 'queue', receipt, cost=1)
    assert len(calls) == 2 and all(item == corrected for item in calls)
