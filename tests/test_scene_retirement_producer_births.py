# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_producer_births.py
#   src/blueprint_pipeline/task_evaluation_launch_activation_worker.py
#   src/blueprint_pipeline/task_evaluation_episode_compilation_worker.py
"""ADP-009D/day28: real child creation inherits exact authenticated preparation."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_scene_retirement_normal_cache import owner_submission
from tests.test_scene_retirement_connected_acceptance import _sealed_file


def preparation(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
    access, policy, request, _, queue, inputs, proofs = owner_submission(tmp_path, monkeypatch)
    cache.publish_preparation_storage_authority(queue_root=queue, request=request, now=101, **proofs)
    receipt = stage_launch_preparation_request(value=request, queue_root=queue, submitted_by='scene-progression')
    generation = cache.enroll_preparation_storage(queue_path=receipt['queue_path'], input_root=inputs, now=101)
    return access, policy, request, inputs / request['preparation_id'], generation


def test_authenticated_child_is_born_before_producer_payload(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    _, _, request, parent, prior = preparation(tmp_path, monkeypatch)
    target = parent.parent / 'actual-activation'
    result = enroll_preparation_child(target, preparation_root=parent, request=request, now=101)
    assert target.is_dir() and list(target.iterdir()) == []
    assert result['state'] == 'active'
    assert result['owner_raw_ref'] == prior['owner_raw_ref']
    assert result['birth_request_raw_ref'] == prior['birth_request_raw_ref']
    assert target.stat().st_ino == result['ino']


@pytest.mark.parametrize('change', ['owner', 'attempt', 'request', 'retired', 'inode'])
def test_child_cannot_borrow_changed_parent_or_authority(tmp_path, monkeypatch, change):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    _, policy, request, parent, generation = preparation(tmp_path, monkeypatch)
    target = parent.parent / 'must-not-exist'
    key = hashlib.sha256(str(parent).encode()).hexdigest() + '.json'
    if change == 'owner':
        generation['owner_intent_id'] = 'foreign-owner'
    elif change == 'attempt':
        generation['birth_request_raw_ref'] = generation['owner_raw_ref']
    elif change == 'request':
        request = dict(request, run_id='foreign-run')
    elif change == 'retired':
        generation['state'] = 'retired'
    else:
        parent.rmdir()
        parent.mkdir()
        generation['ino'] = parent.stat().st_ino + 1
    _sealed_file(Path(policy['generation_store']) / key, generation, 'state_digest', mode=0o600)
    with pytest.raises(ValueError):
        enroll_preparation_child(target, preparation_root=parent, request=request, now=101)
    assert not target.exists()


def test_unregistered_preparation_never_adopts_child(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    _, policy, request, parent, _ = preparation(tmp_path, monkeypatch)
    (Path(policy['generation_store']) / (hashlib.sha256(str(parent).encode()).hexdigest() + '.json')).unlink()
    target = parent.parent / 'legacy-child'
    assert enroll_preparation_child(target, preparation_root=parent, request=request, now=101) is None
    assert not target.exists()


def test_verified_materialized_rows_must_belong_to_exact_preparation(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    _, _, request, parent, _ = preparation(tmp_path, monkeypatch)
    foreign = parent.parent / 'foreign-bytes'
    foreign.write_bytes(b'foreign')
    target = parent.parent / 'must-not-exist'
    with pytest.raises(ValueError):
        enroll_preparation_child(target, preparation_root=parent, request=request,
                                 verified_paths=[foreign], now=101)
    assert not target.exists()


def test_disabled_birth_preserves_original_producer_creation(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement_producer_births import enroll_preparation_child
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', raising=False)
    target = tmp_path / 'original-target'
    assert enroll_preparation_child(target, preparation_root=tmp_path / 'missing', request={}) is None
    assert not target.exists()
