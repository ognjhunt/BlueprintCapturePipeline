"""ADP-009D/day28: original-owner capture birth stays separate from intent births."""

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from tests.test_capture_original_owner_observer import observation
from tests.test_scene_retirement_connected_acceptance import _sealed_file
from tests.test_scene_retirement_real_participants import access_fixture


def _fixture(tmp_path, monkeypatch, *, prepare_parent=True):
    access, policy, root = access_fixture(tmp_path, monkeypatch)
    owner = observation()
    marker = owner["completion_marker"]
    video = owner["producer_delivery"]["raw_video"]
    prefix = f"scenes/{owner['scene_id']}/captures/{owner['capture_id']}"
    manifest_name = f"{prefix}/raw/manifest.json"
    delivery_key = hashlib.sha256(json.dumps(
        [owner["bucket"], marker["object_name"], marker["generation"]],
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    selector = {"object_name": f"{prefix}/deliveries/{delivery_key}/capture_delivery_membership.json",
                "generation": "17000000000000000003"}
    membership = {
        "schema_version": "capture_delivery_membership.v1", "delivery_key": delivery_key,
        "source_finalize": {"bucket": owner["bucket"], "object_name": marker["object_name"],
                            "generation": marker["generation"]},
        "producer_delivery": {"kind": "browser",
                              "receipt_object_name": owner["producer_delivery"]["server_record"]["object_name"],
                              "receipt_generation": owner["producer_delivery"]["server_record"]["generation"],
                              "receipt_size_bytes": owner["producer_delivery"]["server_record"]["size_bytes"],
                              "receipt_sha256": owner["producer_delivery"]["server_record"]["sha256"]},
        "raw": [
            {"object_name": marker["object_name"], "relative_path": "raw/capture_upload_complete.json",
             "generation": marker["generation"], "size_bytes": marker["size_bytes"],
             "crc32c": "AAAAAA==", "sha256": marker["sha256"]},
            {"object_name": manifest_name, "relative_path": "raw/manifest.json",
             "generation": "17000000000000000004", "size_bytes": 12,
             "crc32c": "AAAAAA==", "sha256": "sha256:" + "e" * 64},
            {"object_name": video["object_name"], "relative_path": "raw/walkthrough.mov",
             "generation": video["generation"], "size_bytes": video["size_bytes"],
             "crc32c": video["crc32c"], "sha256": "sha256:" + "f" * 64},
        ], "derived": [],
    }
    encoded = json.dumps(membership, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    selector["size_bytes"] = len(encoded)
    selector["sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    target = (root / owner["bucket"] / "scenes" / owner["scene_id"] / "captures" /
              owner["capture_id"])
    if prepare_parent:
        target.parent.mkdir(parents=True)
    return access, policy, target, owner, selector, encoded


def _next_delivery(owner, membership_raw, *, new_video):
    owner = json.loads(json.dumps(owner))
    member = json.loads(membership_raw)
    marker = owner['completion_marker']
    marker['generation'] = '17000000000000000011'
    if new_video:
        video = owner['producer_delivery']['raw_video']
        video['generation'] = '17000000000000000010'
        receipt = owner['producer_delivery']['server_record']
        receipt['object_name'] = receipt['object_name'].replace(
            '17000000000000000000', video['generation'])
        receipt['generation'] = '17000000000000000012'
        owner['producer_delivery']['delivery_key'] = 'sha256:' + '9' * 64
    source_fields = ('request_id', 'scene_id', 'capture_id', 'bucket', 'raw_prefix_uri',
                     'capture_owner', 'ownership_record', 'consent_attestation',
                     'capture_rights', 'completion_marker', 'producer_delivery')
    owner['source_projection_digest'] = cross_runtime_canonical_digest({
        key: owner[key] for key in source_fields})
    owner['observation_digest'] = cross_runtime_canonical_digest(
        owner, digest_field='observation_digest')
    member['source_finalize']['generation'] = marker['generation']
    member['delivery_key'] = hashlib.sha256(json.dumps([
        owner['bucket'], marker['object_name'], marker['generation']],
        separators=(',', ':')).encode()).hexdigest()
    member['producer_delivery']['receipt_object_name'] = owner['producer_delivery']['server_record']['object_name']
    member['producer_delivery']['receipt_generation'] = owner['producer_delivery']['server_record']['generation']
    member['producer_delivery']['receipt_size_bytes'] = owner['producer_delivery']['server_record']['size_bytes']
    member['producer_delivery']['receipt_sha256'] = owner['producer_delivery']['server_record']['sha256']
    for row in member['raw']:
        if row['relative_path'] == 'raw/capture_upload_complete.json':
            row['generation'] = marker['generation']
        elif row['relative_path'] == 'raw/walkthrough.mov':
            row['generation'] = owner['producer_delivery']['raw_video']['generation']
    encoded = json.dumps(member, sort_keys=True, separators=(',', ':')).encode()
    prefix = f"scenes/{owner['scene_id']}/captures/{owner['capture_id']}"
    selector = {'object_name': f"{prefix}/deliveries/{member['delivery_key']}/capture_delivery_membership.json",
                'generation': '17000000000000000013', 'size_bytes': len(encoded),
                'sha256': 'sha256:' + hashlib.sha256(encoded).hexdigest()}
    return owner, selector, encoded


def test_capture_birth_retains_original_proofs_before_empty_target(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member

    _, policy, target, owner, selector, membership_raw = _fixture(tmp_path, monkeypatch)
    born = birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership_raw=membership_raw)
    assert born["schema_version"] == "scene_capture_generation.v1"
    assert born["capture_owner_user_id"] == "owner-1"
    assert "owner_intent_id" not in born and "birth_request_raw_ref" not in born
    assert born["pinned_marker"] == owner["completion_marker"]
    assert born["dev"] == target.stat().st_dev and born["ino"] == target.stat().st_ino
    assert list(target.iterdir()) == []
    for field in ("owner_observation_raw_ref", "birth_delivery_raw_ref"):
        ref = born[field]
        raw = Path(ref["path"]).read_bytes()
        assert Path(ref["path"]).parent == Path(policy["generation_store"])
        assert ref == {"path": ref["path"], "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                       "size_bytes": len(raw)}
    birth_delivery = json.loads(Path(born['birth_delivery_raw_ref']['path']).read_bytes())
    member_ref = birth_delivery['source_membership_raw_ref']
    assert Path(member_ref['path']).read_bytes() == membership_raw
    assert Path(member_ref['path']).parent == Path(policy['generation_store'])
    assert birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership_raw=membership_raw) == born


def test_normal_capture_birth_prepares_absent_parent_under_owned_root(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member

    _, policy, target, owner, selector, membership_raw = _fixture(
        tmp_path, monkeypatch, prepare_parent=False)
    assert not target.parent.exists()
    born = birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership_raw=membership_raw)
    assert born['state'] == 'active' and target.is_dir()
    assert list(target.iterdir()) == []
    for parent in (target.parent, *list(target.parents)[1:5]):
        assert parent.stat().st_uid == Path(policy['generation_store']).stat().st_uid


def test_capture_birth_source_projection_reopens_exact_retained_membership(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import (
        birth_capture_member, capture_birth_source_projection,
    )

    _, _, target, owner, selector, membership_raw = _fixture(tmp_path, monkeypatch)
    born = birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership_raw=membership_raw)
    proof = capture_birth_source_projection(target)
    assert proof['generation_id'] == born['generation_id']
    assert proof['source_membership_selector'] == selector
    assert proof['delivery_key'] == owner['producer_delivery']['delivery_key']
    assert proof['raw_video']['generation'] == owner['producer_delivery']['raw_video']['generation']
    assert proof['raw_video']['sha256'] == json.loads(membership_raw)['raw'][2]['sha256']
    assert proof['owner_observation_raw_ref'] == born['owner_observation_raw_ref']
    Path(json.loads(Path(born['birth_delivery_raw_ref']['path']).read_bytes())[
        'source_membership_raw_ref']['path']).unlink()
    with pytest.raises(ValueError):
        capture_birth_source_projection(target)


def test_native_website_registration_binds_original_delivery_and_sponsor_rights(
        tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member
    from blueprint_pipeline.website_scene_dispatch import (
        register_website_preparation, resolve_website_source,
    )
    from blueprint_pipeline import website_native_background as native
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    _, _, target, owner, selector, membership_raw = _fixture(tmp_path, monkeypatch)
    born = birth_capture_member(target, observation=owner,
                                membership_selector=selector, membership_raw=membership_raw)
    base = target / 'pipeline' / 'website_scene_preparation'
    native_dir = base / 'native'
    native_dir.mkdir(parents=True)
    context = {'schema_version': 'website_site_task_context.v1',
               'request_id': owner['request_id'], 'scene_id': owner['scene_id'],
               'capture_id': owner['capture_id'], 'capture_rights': owner['capture_rights']}
    context['context_digest'] = canonical_digest(context, digest_field='context_digest')
    request = {'owner': {'user_id': 'sponsor-2'},
               'consent': {'rights_reference': cross_runtime_canonical_digest(owner['capture_rights'])},
               'source': {'binding_id': 'website-source',
                          'content_digest': 'sha256:' + '4' * 64}}
    preparation = {'intake_request': request,
                   'binding': {'task_context_digest': context['context_digest']}}
    preparation['digest'] = canonical_digest(preparation, digest_field='digest')
    preparation_path = base / 'preparation.json'
    runtime_path = native_dir / 'runtime_inputs.json'
    context_path = base / 'task_context.json'
    preparation_path.write_text(json.dumps(preparation))
    runtime_path.write_text('{}')
    context_path.write_text(json.dumps(context))
    monkeypatch.setattr(native, 'construction_rights_admission', lambda **_: {})
    monkeypatch.setattr(native, 'prepare_construction_stages',
                        lambda **_: {'references': []})
    selected = register_website_preparation(
        preparation_path=preparation_path, runtime_inputs_path=runtime_path,
        task_context_path=context_path, root=tmp_path / 'bindings', now=1)
    registration = json.loads(Path(selected['path']).read_bytes())
    assert registration['capture_source']['generation_id'] == born['generation_id']
    assert registration['capture_source']['source_membership_selector'] == selector
    assert registration['capture_source']['capture_owner_user_id'] == 'owner-1'
    assert request['owner']['user_id'] != owner['capture_owner']['user_id']
    intent = {'request': request, 'intent_id': 'scene-' + '1' * 32,
              'intent_digest': 'sha256:' + '2' * 64,
              'task_content_digest': 'sha256:' + '3' * 64}
    resolution = resolve_website_source(intent=intent, config={
        'website_source_binding_root': str(tmp_path / 'bindings'),
        'factory_output_root': str(tmp_path / 'factory'),
        'website_source_machinery_path': str(tmp_path / 'machinery.json'),
    })
    binding = json.loads(Path(resolution.binding_path).read_bytes())
    assert binding['capture_source'] == registration['capture_source']
    context['capture_rights']['consent_revoked'] = True
    context['context_digest'] = canonical_digest(context, digest_field='context_digest')
    context_path.write_text(json.dumps(context))
    with pytest.raises(ValueError, match='website_capture_source_rights_mismatch'):
        register_website_preparation(
            preparation_path=preparation_path, runtime_inputs_path=runtime_path,
            task_context_path=context_path, root=tmp_path / 'other-bindings', now=1)


def test_materialization_refuses_changed_native_capture_binding_before_attempt_birth(
        tmp_path, monkeypatch):
    from blueprint_pipeline import website_scene_dispatch as dispatch
    from blueprint_pipeline import task_evaluation_scene_owner_authority as authority
    from blueprint_pipeline import task_evaluation_scene_intake as intake
    from blueprint_pipeline import task_evaluation_scene_preparation_attempts as attempts

    intent_path = tmp_path / 'intent' / 'intent.json'
    intent_path.parent.mkdir()
    intent_path.write_text('{}')
    binding_path = tmp_path / 'binding.json'
    machinery_path = tmp_path / 'machinery.json'
    release_path = tmp_path / 'release.json'
    intent = {'intent_id': 'scene-' + '1' * 32, 'intent_digest': 'sha256:' + '2' * 64,
              'task_content_digest': 'sha256:' + '3' * 64,
              'request': {'owner': {'user_id': 'sponsor-2'}}}
    binding = {'schema_version': 'website_scene_source_binding.v1',
               'intent_digest': intent['intent_digest'],
               'task_digest': intent['task_content_digest'],
               'owner': intent['request']['owner'],
               'capture_source': {'delivery_key': 'sha256:' + 'a' * 64},
               'references': {'preparation': {'path': str(tmp_path / 'preparation.json')},
                              'task_context': {'path': str(tmp_path / 'context.json')}}}
    monkeypatch.setattr(authority, 'reopen_scene_intent', lambda *_args, **_kwargs: intent)
    monkeypatch.setattr(dispatch, 'read', lambda path, **_kwargs: (
        binding if Path(path) == binding_path else
        {'schema_version': dispatch.RELEASE_SCHEMA} if Path(path) == release_path else
        {'schema_version': 'task_evaluation_website_scene_machinery.v1'}))
    monkeypatch.setattr(attempts, 'preparation_attempt_path',
                        lambda *_args: tmp_path / 'attempt.json')
    monkeypatch.setattr(intake, '_read', lambda *_args, **_kwargs: {
        'intent_digest': intent['intent_digest'], 'input_digest': 'sha256:' + '5' * 64,
        'source_commit': 'commit'})
    monkeypatch.setattr(dispatch, '_selected_capture_source',
                        lambda *_args: {'delivery_key': 'sha256:' + 'b' * 64})
    with pytest.raises(ValueError, match='website_capture_source_binding_changed'):
        dispatch.materialize_website_attempt(
            intent_path=intent_path, source_binding_path=binding_path,
            machinery_path=machinery_path, release_binding_path=release_path,
            output_root=tmp_path / 'output', attempt_id='attempt-1')


def test_capture_birth_rejects_missing_member_or_changed_delivery_without_target(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member

    _, _, target, owner, selector, membership_raw = _fixture(tmp_path, monkeypatch)
    membership = json.loads(membership_raw)
    membership["raw"] = membership["raw"][:1]
    membership_raw = json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()
    selector["size_bytes"] = len(membership_raw)
    selector["sha256"] = "sha256:" + hashlib.sha256(membership_raw).hexdigest()
    with pytest.raises(ValueError):
        birth_capture_member(target, observation=owner, membership_selector=selector,
                             membership_raw=membership_raw)
    assert not target.exists()


def test_active_capture_refuses_another_valid_delivery_on_occupied_target(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member

    _, policy, target, owner, selector, membership_raw = _fixture(tmp_path, monkeypatch)
    born = birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership_raw=membership_raw)
    next_owner, next_selector, next_raw = _next_delivery(owner, membership_raw,
                                                          new_video=True)
    with pytest.raises(ValueError, match='scene_capture_active_delivery_conflict'):
        birth_capture_member(target, observation=next_owner,
                             membership_selector=next_selector,
                             membership_raw=next_raw)
    key = hashlib.sha256(str(target).encode()).hexdigest() + '.json'
    assert json.loads((Path(policy['generation_store']) / key).read_bytes()) == born


def test_app_bundle_member_requires_exact_server_completion_receipt(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member

    _, _, target, owner, selector, membership_raw = _fixture(tmp_path, monkeypatch)
    member = json.loads(membership_raw)
    producer = owner['producer_delivery']
    producer['kind'] = 'website_capture_link_bundle'
    producer['server_record']['object_name'] = (
        f"scenes/{owner['scene_id']}/captures/{owner['capture_id']}/upload/bundle_completion.json")
    member['producer_delivery']['kind'] = producer['kind']
    member['producer_delivery']['receipt_object_name'] = producer['server_record']['object_name']
    source_fields = ('request_id', 'scene_id', 'capture_id', 'bucket', 'raw_prefix_uri',
                     'capture_owner', 'ownership_record', 'consent_attestation',
                     'capture_rights', 'completion_marker', 'producer_delivery')
    owner['source_projection_digest'] = cross_runtime_canonical_digest({
        key: owner[key] for key in source_fields})
    owner['observation_digest'] = cross_runtime_canonical_digest(
        owner, digest_field='observation_digest')
    raw = json.dumps(member, sort_keys=True, separators=(',', ':')).encode()
    selector['size_bytes'] = len(raw)
    selector['sha256'] = 'sha256:' + hashlib.sha256(raw).hexdigest()
    bad = json.loads(raw)
    bad['producer_delivery']['receipt_sha256'] = 'sha256:' + '0' * 64
    bad_raw = json.dumps(bad, sort_keys=True, separators=(',', ':')).encode()
    bad_selector = dict(selector, size_bytes=len(bad_raw),
                        sha256='sha256:' + hashlib.sha256(bad_raw).hexdigest())
    with pytest.raises(ValueError, match='capture_membership_receipt_mismatch'):
        birth_capture_member(target, observation=owner,
                             membership_selector=bad_selector, membership_raw=bad_raw)
    assert not target.exists()
    born = birth_capture_member(target, observation=owner,
                                membership_selector=selector, membership_raw=raw)
    assert born['schema_version'] == 'scene_capture_generation.v1'


def test_retired_capture_requires_new_raw_delivery_and_marker(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member

    _, policy, target, owner, selector, membership_raw = _fixture(tmp_path, monkeypatch)
    born = birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership_raw=membership_raw)
    key = hashlib.sha256(str(target).encode()).hexdigest() + ".json"
    retired = dict(born, state="retired", state_sequence=born["state_sequence"] + 1)
    _sealed_file(Path(policy["generation_store"]) / key, retired, "state_digest", mode=0o600)
    target.rmdir()
    with pytest.raises(ValueError):
        birth_capture_member(target, observation=owner, membership_selector=selector,
                             membership_raw=membership_raw)
    assert not target.exists()
    rewrite, rewrite_selector, rewrite_raw = _next_delivery(owner, membership_raw,
                                                              new_video=False)
    with pytest.raises(ValueError):
        birth_capture_member(target, observation=rewrite,
                             membership_selector=rewrite_selector,
                             membership_raw=rewrite_raw)
    assert not target.exists()
    next_owner, next_selector, next_raw = _next_delivery(owner, membership_raw,
                                                          new_video=True)
    newer = birth_capture_member(target, observation=next_owner,
                                 membership_selector=next_selector,
                                 membership_raw=next_raw)
    assert newer['state'] == 'active' and newer['generation_id'] != born['generation_id']
    assert newer['previous_generation_id'] == born['generation_id']
    assert newer['ino'] == target.stat().st_ino
    assert Path(born['owner_observation_raw_ref']['path']).is_file()
    assert Path(born['birth_delivery_raw_ref']['path']).is_file()


@pytest.mark.parametrize('terminal', [False, True])
def test_direct_selected_stage_reads_only_pinned_members_without_prefix_list(
        tmp_path, monkeypatch, terminal):
    from blueprint_pipeline import capture_original_owner_observer as observer
    from blueprint_pipeline import pubsub_handoff_listener as listener

    _, policy, target, owner, selector, membership_raw = _fixture(tmp_path, monkeypatch,
                                                                    prepare_parent=False)
    membership = json.loads(membership_raw)
    contents = {
        owner['completion_marker']['object_name']: b'{"done":true}',
        owner['producer_delivery']['raw_video']['object_name']: b'video0000',
        next(row['object_name'] for row in membership['raw']
             if row['relative_path'] == 'raw/manifest.json'): b'{}',
    }
    marker = owner['completion_marker']
    marker['size_bytes'] = len(contents[marker['object_name']])
    marker['sha256'] = 'sha256:' + hashlib.sha256(contents[marker['object_name']]).hexdigest()
    for row in membership['raw']:
        body = contents[row['object_name']]
        row['size_bytes'] = len(body)
        row['sha256'] = 'sha256:' + hashlib.sha256(body).hexdigest()
    source_fields = ('request_id', 'scene_id', 'capture_id', 'bucket', 'raw_prefix_uri',
                     'capture_owner', 'ownership_record', 'consent_attestation',
                     'capture_rights', 'completion_marker', 'producer_delivery')
    owner['source_projection_digest'] = cross_runtime_canonical_digest({
        key: owner[key] for key in source_fields})
    owner['observation_digest'] = cross_runtime_canonical_digest(
        owner, digest_field='observation_digest')
    membership_raw = json.dumps(membership, sort_keys=True, separators=(',', ':')).encode()
    selector['size_bytes'] = len(membership_raw)
    selector['sha256'] = 'sha256:' + hashlib.sha256(membership_raw).hexdigest()
    contents[selector['object_name']] = membership_raw
    rows = {row['object_name']: row for row in membership['raw']}

    class Blob:
        def __init__(self, name, generation):
            self.name = name
            self.generation = generation
            self.size = len(contents[name])
            self.crc32c = rows[name]['crc32c'] if name in rows else 'AAAAAA=='

        def reload(self, **kwargs):
            assert kwargs['if_generation_match'] == self.generation
            assert kwargs['retry'] is None

        def download_as_bytes(self, **kwargs):
            assert kwargs['if_generation_match'] == self.generation
            return contents[self.name]

    class Bucket:
        def blob(self, name, generation):
            assert name in contents
            expected = rows[name]['generation'] if name in rows else selector['generation']
            assert str(generation) == expected
            return Blob(name, generation)

    class Client:
        def bucket(self, name):
            assert name == owner['bucket']
            return Bucket()

        def list_blobs(self, *_args, **_kwargs):
            pytest.fail('selected source cannot list current prefix')

    def download(*, downloads, manifest_rows, selected_generations, **_kwargs):
        assert set(selected_generations) == set(rows)
        assert {row['name'] for row in manifest_rows} == set(rows)
        for blob, destination in downloads:
            destination.write_bytes(contents[blob.name])

    monkeypatch.setattr(observer, 'load_original_owner_observation', lambda **_: owner)
    monkeypatch.setattr(listener, 'download_with_reservation', download)
    payload = {
        'bucket': owner['bucket'], 'scene_id': owner['scene_id'],
        'capture_id': owner['capture_id'], 'raw_prefix_uri': owner['raw_prefix_uri'],
        'source_finalize': {**membership['source_finalize'], 'event_id': 'evt-1',
                            'event_source': 'storage'},
        'source_membership_selector': selector,
    }
    staged = listener.stage_handoff_capture(listener.parse_handoff_payload(payload),
                                             storage_root=target.parents[4], storage_client=Client())
    assert staged == target
    assert (target / 'raw' / 'manifest.json').read_bytes() == b'{}'
    staged_manifest = json.loads((target / listener.STAGING_MANIFEST_FILENAME).read_bytes())
    assert staged_manifest['delivery_key'] == membership['delivery_key']
    assert staged_manifest['source_membership_selector'] == selector
    assert staged_manifest['local_generation_id']
    assert not (target / 'derived').exists()
    runs = []

    def consume(**kwargs):
        runs.append(kwargs)
        assert kwargs['capture_root'] == str(target)
        assert json.loads((target / listener.STAGING_MANIFEST_FILENAME).read_bytes())[
            'local_generation_id'] == staged_manifest['local_generation_id']
        if terminal:
            raise ValueError('website_control_scene-sponsorship_http_409:consent_expired')
        return {'status': 'completed'}

    first = listener.process_handoff_payload(
        payload, storage_root=target.parents[4], storage_client=Client(),
        provider='local', run_e2e=consume)
    assert first['status'] == ('terminal_authority_ended' if terminal else 'processed')
    ledger = json.loads((target / listener.JOB_LEDGER_FILENAME).read_bytes())
    assert ledger['producer_delivery_key'] == owner['producer_delivery']['delivery_key']
    assert ledger['source_payload_sha256'] == listener.payload_sha256(payload)
    if terminal:
        assert ledger['terminal_producer_delivery_key'] == owner['producer_delivery']['delivery_key']
        assert ledger['attempt_history'][-1]['producer_delivery_key'] == owner['producer_delivery']['delivery_key']
    assert len(runs) == 1
    replay = {**payload, 'source_finalize': {**payload['source_finalize'], 'event_id': 'evt-2'}}
    second = listener.process_handoff_payload(
        replay, storage_root=target.parents[4], storage_client=Client(),
        provider='local', run_e2e=consume)
    assert second['status'] == ('skipped_terminal_authority_ended' if terminal
                                else 'skipped_already_processed')
    assert len(runs) == 1


def test_capture_lease_binds_semantic_delivery_across_changed_payload_bytes(tmp_path):
    from blueprint_pipeline import pubsub_handoff_listener as listener

    target = tmp_path / 'capture'
    target.mkdir()
    key = 'sha256:' + 'a' * 64
    first, ledger = listener._claim_job_lease(
        target, scene_id='scene-1', capture_id='cap-1', owner='worker-1',
        lease_seconds=60, payload_sha256='b' * 64,
        producer_delivery_key=key, create_capture_root=False)
    assert first == 'claimed' and ledger['producer_delivery_key'] == key
    assert ledger['source_payload_sha256'] == 'b' * 64
    listener._finish_job_lease(target, owner='worker-1', token=ledger['lease_token'],
                               update={'status': listener.TERMINAL_AUTHORITY_STATUS,
                                       'terminal_payload_sha256': 'b' * 64,
                                       'terminal_producer_delivery_key': key})
    second, unchanged = listener._claim_job_lease(
        target, scene_id='scene-1', capture_id='cap-1', owner='worker-2',
        lease_seconds=60, payload_sha256='c' * 64,
        producer_delivery_key=key, create_capture_root=False)
    assert second == 'terminal' and unchanged['terminal_producer_delivery_key'] == key
    assert unchanged['revision'] == 2


def test_retired_capture_selector_uses_original_producer_key_before_any_stage(tmp_path, monkeypatch):
    from blueprint_pipeline import capture_original_owner_observer as observer
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import website_scene_workspace_retention as retention

    _, _, target, owner, selector, membership_raw = _fixture(tmp_path, monkeypatch,
                                                               prepare_parent=False)
    membership = json.loads(membership_raw)
    payload = {'bucket': owner['bucket'], 'scene_id': owner['scene_id'],
               'capture_id': owner['capture_id'], 'raw_prefix_uri': owner['raw_prefix_uri'],
               'source_finalize': {**membership['source_finalize'], 'event_id': 'evt-1',
                                   'event_source': 'storage'},
               'source_membership_selector': selector}
    monkeypatch.setattr(observer, 'load_original_owner_observation', lambda **_: owner)
    calls = []
    monkeypatch.setattr(listener, 'stage_handoff_capture',
                        lambda *_args, **_kwargs: calls.append('stage') or target)
    key = owner['producer_delivery']['delivery_key']
    retired = {'status': listener.TERMINAL_AUTHORITY_STATUS,
               'queue_disposition': listener.TERMINAL_AUTHORITY_STATUS,
               'receipt': 'retained-receipt'}
    monkeypatch.setattr(retention, 'retired_capture_status',
                        lambda **_: {**retired, 'producer_delivery_keys': [key]})
    result = listener.process_handoff_payload(payload, storage_root=target.parents[4],
                                               provider='openai', run_e2e=lambda **_: pytest.fail('ran'))
    assert result['status'] == 'skipped_retired_terminal' and calls == []
    monkeypatch.setattr(retention, 'retired_capture_status',
                        lambda **_: {**retired, 'producer_delivery_keys': []})
    result = listener.process_handoff_payload(payload, storage_root=target.parents[4],
                                               provider='openai', run_e2e=lambda **_: pytest.fail('ran'))
    assert result['status'] == 'retired_delivery_identity_unproven_retryable' and calls == []
    assert not target.exists()


def test_retired_status_collects_delivery_keys_from_current_and_history():
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import website_scene_workspace_retention as retention

    old, current = 'sha256:' + 'a' * 64, 'sha256:' + 'b' * 64
    ledger = {'status': listener.TERMINAL_AUTHORITY_STATUS,
              'terminal_producer_delivery_key': current,
              'attempt_history': [{'status': listener.TERMINAL_AUTHORITY_STATUS,
                                   'producer_delivery_key': old}]}
    assert retention.ended_producer_delivery_keys(ledger) == {old, current}
    assert listener._ended_delivery_keys(ledger) == {old, current}
