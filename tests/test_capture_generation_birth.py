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
        "producer_delivery": {"kind": owner["producer_delivery"]["kind"],
                              "receipt_object_name": owner["producer_delivery"]["server_record"]["object_name"],
                              "receipt_generation": owner["producer_delivery"]["server_record"]["generation"]},
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


def test_direct_selected_stage_reads_only_pinned_members_without_prefix_list(tmp_path, monkeypatch):
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
