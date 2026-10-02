"""Actual queue and HTTP response lifetimes use the retirement exclusion."""
import asyncio
import threading
from importlib import import_module

import pytest

from tests.test_scene_retirement_real_participants import access_fixture, run_paused, StopFixture


@pytest.mark.parametrize('role', ['preparation', 'compilation', 'sam-phase'])
def test_real_worker_holds_outer_fence_before_native_worker_lock(tmp_path, monkeypatch, role):
    access, _, member = access_fixture(tmp_path, monkeypatch)
    entered, finish, release = threading.Event(), threading.Event(), threading.Event()
    errors = []
    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        raise StopFixture()
    if role == 'preparation':
        module = import_module('blueprint_pipeline.task_evaluation_launch_preparation_worker')
        monkeypatch.setattr(module, 'ensure_launch_preparation_queue_root', paused)
        def operation():
            module.process_launch_preparation_queue(queue_root=member, input_root=member,
                allowed_uri_prefixes=['s3://fixture/'], service_account='fixture', source_commit='a'*40)
    elif role == 'compilation':
        module = import_module('blueprint_pipeline.task_evaluation_episode_compilation_worker')
        monkeypatch.setattr(module, 'ensure_scene_construction_queue_root', paused)
        def operation():
            module.process_episode_compilation_queue(queue_root=member, input_root=member,
                output_root=member, source_commit='a'*40)
    else:
        module = import_module('blueprint_pipeline.task_evaluation_sam31_preparation_execution')
        monkeypatch.setattr(module, '_verified_checkout_head', lambda: 'a'*40)
        monkeypatch.setattr(module, '_ensure', paused)
        def operation():
            module.process_sam31_phase_queue(queue_root=member, parent_queue_root=member,
                preparation_input_root=member, execution_root=member, source_commit='a'*40)
    thread = threading.Thread(target=run_paused, args=(operation, entered, finish, errors))
    thread.start()
    try:
        assert entered.wait(3), errors
        with pytest.raises(access.SceneRetirementAccessError, match='scene_retirement_reader_active'):
            with access.exclusive_scene_access():
                pytest.fail('exclusive mutation overlapped actual worker before its inner lock')
    finally:
        release.set()
        thread.join(4)
    assert not thread.is_alive() and finish.is_set() and errors == []


@pytest.mark.parametrize('send_failure', [False, True])
def test_actual_result_response_retains_outer_fence_until_send_or_disconnect(tmp_path, monkeypatch, send_failure):
    access, _, member = access_fixture(tmp_path, monkeypatch)
    from blueprint_pipeline import live_pipeline_result_artifact_resolution as resolution
    from blueprint_pipeline.live_pipeline_result_artifact_response import ResultArtifactFileResponse
    payload = member / 'result.bin'
    payload.write_bytes(b'actual-result')
    monkeypatch.setattr(resolution, '_policy_canary_run_root', lambda **kwargs: member)
    monkeypatch.setattr(resolution, '_registered_operator_run_root', lambda **kwargs: None)
    monkeypatch.setattr(resolution, 'resolve_task_evaluation_result_artifact',
                        lambda **kwargs: (payload, {'sha256': 'sha256:'+'a'*64}))
    path, record = resolution.resolve_live_pipeline_result_artifact(
        legacy_state_root=tmp_path/'legacy', policy_canary_result_root=member,
        run_id='fixture-run', artifact_id='fixture', retain_read_lease=True)
    response = ResultArtifactFileResponse(path, artifact_cleanup=record.get('_artifact_cleanup'))
    observed = []
    async def send(message):
        with pytest.raises(access.SceneRetirementAccessError, match='scene_retirement_reader_active'):
            with access.exclusive_scene_access():
                pytest.fail('response dropped exclusion while bytes could still be sent')
        observed.append(message['type'])
        if send_failure:
            raise StopFixture('actual ASGI send disconnected')
    async def receive():
        return {'type': 'http.disconnect'}
    scope = {'type':'http','method':'GET','asgi': {'spec_version':'2.4'}, 'extensions': {}, 'headers': []}
    if send_failure:
        with pytest.raises(StopFixture):
            asyncio.run(response(scope, receive, send))
    else:
        asyncio.run(response(scope, receive, send))
    assert observed
    with access.exclusive_scene_access():
        pass


def test_actual_pin_publisher_denies_reference_to_retired_generation(tmp_path, monkeypatch):
    import hashlib
    import json
    from pathlib import Path
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.control_plane_storage_pins import write_storage_pin
    from tests.test_scene_retirement_real_participants import authenticated_birth_refs
    access, policy, member = access_fixture(tmp_path,monkeypatch)
    intent_id, owner, request = authenticated_birth_refs(tmp_path,monkeypatch)
    member.rmdir()
    state = access.birth_scene_member(member,owner_intent_id=intent_id,
        owner_raw_ref=owner,birth_request_raw_ref=request,now=101)
    state.update(state='retired',retirement_token='2'*32,journal_sha256='sha256:'+'c'*64)
    state['state_digest'] = canonical_digest(state,digest_field='state_digest')
    record = Path(policy['generation_store'])/(hashlib.sha256(str(member).encode()).hexdigest()+'.json')
    record.write_text(json.dumps(state))
    pins = tmp_path/'pins'
    with pytest.raises(access.SceneRetirementAccessError,match='scene_retirement_generation_unavailable'):
        write_storage_pin(pins_root=pins,kind='preparation',owner_id='new-pin',paths=[member])
    assert not pins.exists() or list(pins.rglob('*.json')) == []
