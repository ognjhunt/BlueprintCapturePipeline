# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_launch_births.py
#   src/blueprint_pipeline/task_evaluation_launch_dispatcher.py
"""ADP-009D/day28: actual launch output birth uses its original paid owner tuple."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_scene_retirement_real_participants import access_fixture, authenticated_birth_refs
from tests.test_scene_retirement_connected_acceptance import _sealed_file
from tests.test_task_evaluation_launch_dispatcher import _profile, _write_profile_and_request
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def owned_launch(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_execution_authority as execution
    _, policy, _ = access_fixture(tmp_path, monkeypatch)
    _, owner, attempt = authenticated_birth_refs(tmp_path, monkeypatch)
    native = tmp_path / 'native'
    native.mkdir()
    profile = _profile(native)
    record = json.loads(Path(attempt['path']).read_bytes())
    profile.update(execution.bind_scene_attempt(record), source_commit=record['source_commit'])
    profile['allocator']['argv'].extend(['--provider', 'vast'])
    profile['profile_digest'] = canonical_digest(profile, digest_field='profile_digest')
    profiles, request_path = _write_profile_and_request(native, profile)
    outputs = tmp_path / 'launches'
    outputs.mkdir()
    policy['roots'] = [dict(root=str(outputs), storage_class='evidence', device=outputs.stat().st_dev)]
    _sealed_file(tmp_path / 'policy.json', policy, 'policy_digest', mode=0o644)
    monkeypatch.setattr(execution.time, 'time', lambda: 101)
    request = json.loads(request_path.read_bytes())
    return policy, owner, attempt, profile, request, profiles, request_path, outputs


def test_original_authenticated_launch_tuple_births_empty_output(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_launch_births import enroll_launch_output
    _, owner, attempt, profile, request, _, _, outputs = owned_launch(tmp_path, monkeypatch)
    target = outputs / request['launch_id']
    born = enroll_launch_output(target, request=request, profile=profile, now=101)
    assert list(target.iterdir()) == []
    assert born['owner_raw_ref'] == owner and born['birth_request_raw_ref'] == attempt
    assert born['ino'] == target.stat().st_ino


@pytest.mark.parametrize('change', ['profile-digest', 'source-commit', 'source-bundle',
    'evaluation-spec', 'launch-id', 'owner', 'attempt'])
def test_launch_cannot_borrow_resealed_foreign_native_or_owner_tuple(tmp_path, monkeypatch, change):
    from blueprint_pipeline.task_evaluation_scene_retirement_launch_births import enroll_launch_output
    _, _, _, profile, request, _, _, outputs = owned_launch(tmp_path, monkeypatch)
    target = outputs / request['launch_id']
    if change == 'profile-digest':
        request['launch_profile_digest'] = 'sha256:' + 'f' * 64
    elif change == 'source-commit':
        request['source_commit'] = 'f' * 40
    elif change == 'source-bundle':
        request['source_bundle'] = dict(request['source_bundle'], digest='sha256:' + 'f' * 64)
    elif change == 'evaluation-spec':
        request['evaluation_run_spec'] = dict(request['evaluation_run_spec'], digest='sha256:' + 'f' * 64)
    elif change == 'launch-id':
        request['launch_id'] = 'another-launch'
    else:
        profile['scene_attempt_binding'] = dict(profile['scene_attempt_binding'])
        profile['scene_attempt_binding']['intent_id' if change == 'owner' else 'attempt_id'] = 'foreign'
        profile['profile_digest'] = canonical_digest(profile, digest_field='profile_digest')
        request['launch_profile_digest'] = profile['profile_digest']
    request['request_digest'] = canonical_digest(request, digest_field='request_digest')
    with pytest.raises(ValueError):
        enroll_launch_output(target, request=request, profile=profile, now=101)
    assert not target.exists()


def test_expired_owner_cannot_create_new_launch_output(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_launch_births import enroll_launch_output
    _, _, _, profile, request, _, _, outputs = owned_launch(tmp_path, monkeypatch)
    target = outputs / request['launch_id']
    with pytest.raises(ValueError):
        enroll_launch_output(target, request=request, profile=profile, now=1001)
    assert not target.exists()


def test_actual_dispatcher_born_before_first_immutable_launch_record(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_launch_dispatcher as dispatcher
    policy, owner, attempt, _, request, profiles, request_path, outputs = owned_launch(tmp_path, monkeypatch)
    original_write = dispatcher._write_immutable
    observed = []
    def first_write(path, value):
        target = outputs / request['launch_id']
        key = hashlib.sha256(str(target).encode()).hexdigest() + '.json'
        generation = json.loads((Path(policy['generation_store']) / key).read_bytes())
        assert generation['owner_raw_ref'] == owner and generation['birth_request_raw_ref'] == attempt
        observed.append(generation)
        raise RuntimeError('stop before launch work')
    monkeypatch.setattr(dispatcher, '_write_immutable', first_write)
    with pytest.raises(RuntimeError, match='stop before launch'):
        dispatcher.dispatch_launch_request(request_path=request_path, profile_dir=profiles,
            state_root=outputs, execute=False, allocator_runner=lambda argv: pytest.fail('no paid execution'))
    assert len(observed) == 1 and original_write is not None


def test_legacy_profile_cannot_adopt_existing_launch_directory(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_launch_births import enroll_launch_output
    policy, _, _, profile, request, _, _, outputs = owned_launch(tmp_path, monkeypatch)
    for field in ('scene_attempt_binding', 'scene_attempt_id', 'scene_intent_digest'):
        del profile[field]
    target = outputs / request['launch_id']
    target.mkdir()
    (target / 'legacy').write_bytes(b'keep')
    assert enroll_launch_output(target, request=request, profile=profile, now=101) is None
    assert (target / 'legacy').read_bytes() == b'keep'
    assert list(Path(policy['generation_store']).glob('*.json')) == []


def test_actual_existing_launch_output_preserves_birth_after_owner_expiry(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_launch_dispatcher as dispatcher
    from blueprint_pipeline import task_evaluation_scene_execution_authority as execution
    from blueprint_pipeline.task_evaluation_scene_retirement_launch_births import enroll_launch_output
    policy, _, _, profile, request, profiles, request_path, outputs = owned_launch(tmp_path, monkeypatch)
    target = outputs / request['launch_id']
    original = enroll_launch_output(target, request=request, profile=profile, now=101)
    monkeypatch.setattr(execution.time, 'time', lambda: 1001)
    key = hashlib.sha256(str(target).encode()).hexdigest() + '.json'
    def retained_metadata(path, value):
        assert json.loads((Path(policy['generation_store']) / key).read_bytes()) == original
        raise RuntimeError('existing launch remains original without a new birth')
    monkeypatch.setattr(dispatcher, '_write_immutable', retained_metadata)
    with pytest.raises(RuntimeError, match='existing launch remains'):
        dispatcher.dispatch_launch_request(request_path=request_path, profile_dir=profiles,
            state_root=outputs, execute=False, allocator_runner=lambda argv: pytest.fail('no paid execution'))


def test_existing_owned_launch_cannot_be_borrowed_by_foreign_attempt(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_launch_dispatcher as dispatcher
    from blueprint_pipeline.task_evaluation_scene_retirement_launch_births import enroll_launch_output
    _, _, _, profile, request, profiles, request_path, outputs = owned_launch(tmp_path, monkeypatch)
    target = outputs / request['launch_id']
    enroll_launch_output(target, request=request, profile=profile, now=101)
    profile['scene_attempt_binding']['attempt_id'] = profile['scene_attempt_id'] = 'foreign'
    profile['profile_digest'] = canonical_digest(profile, digest_field='profile_digest')
    _write_profile_and_request(request_path.parent, profile)
    monkeypatch.setattr(dispatcher, '_write_immutable', lambda *a, **k: pytest.fail('foreign metadata write'))
    with pytest.raises(ValueError):
        dispatcher.dispatch_launch_request(request_path=request_path, profile_dir=profiles,
            state_root=outputs, execute=False, allocator_runner=lambda argv: pytest.fail('no paid execution'))
