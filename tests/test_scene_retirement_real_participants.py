# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_access.py
#   src/blueprint_pipeline/artifixer_completed_training_reuse.py
#   src/blueprint_pipeline/task_evaluation_scene_configuration_submission_publication.py
"""Real existing reader/publisher lifetime, development-only local fixtures."""
import hashlib
import json
import os
import threading
from pathlib import Path

import pytest


class StopFixture(RuntimeError):
    pass


def access_fixture(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    coordinator = tmp_path / 'coordination'
    coordinator.mkdir(mode=0o755)
    generations = tmp_path / 'generations'
    generations.mkdir(mode=0o700)
    member = tmp_path / 'scene'
    member.mkdir()
    value = {
        'schema_version': 'scene_retirement_policy.v1', 'enabled': True,
        'policy_id': 'test-policy', 'roots': [{'root': str(member), 'storage_class': 'evidence',
                                               'device': member.stat().st_dev}],
        'coordinator_path': str(coordinator), 'generation_store': str(generations),
        'journal_store': str(tmp_path / 'journals'), 'consumer_cohort': [],
        'principals': [], 'private_archive_allowed_classes': [], 'limits': {},
    }
    value['policy_digest'] = canonical_digest(value, digest_field='policy_digest')
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps(value))
    policy.chmod(0o644)
    monkeypatch.setenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', str(policy))
    # Authority is a current-user hermetic fixture, never a production override.
    monkeypatch.setattr(access, '_POLICY_UID', os.getuid())
    return access, value, member


def run_paused(operation, entered, finish, errors):
    try:
        operation()
    except StopFixture:
        pass
    except BaseException as exc:
        errors.append(exc)
    finally:
        finish.set()


@pytest.mark.parametrize('role', ['completed-review-reader', 'submission-publisher'])
def test_real_participant_holds_fence_through_actual_body(tmp_path, monkeypatch, role):
    access, _, member = access_fixture(tmp_path, monkeypatch)
    entered, finish, release = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        raise StopFixture()

    if role == 'completed-review-reader':
        from blueprint_pipeline import artifixer_completed_training_reuse as existing
        monkeypatch.setattr(existing, '_read', paused)
        operation = lambda: existing.stage_completed_review(source_root=member, output_root=tmp_path / 'out')
    else:
        from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as existing
        manifest = member / 'manifest.json'
        manifest.write_text('{"input_namespace":"scene-test"}')
        locks = tmp_path / 'publisher-locks'
        locks.mkdir()
        monkeypatch.setattr(existing, '_publish_locked', paused)
        operation = lambda: existing.publish_scene_configuration_submission(
            manifest_path=manifest, receipt_path=member / 'publication.json',
            expected_source_commit='a' * 40, lock_root=locks)
    worker = threading.Thread(target=run_paused, args=(operation, entered, finish, errors))
    worker.start()
    try:
        assert entered.wait(3), errors
        with pytest.raises(access.SceneRetirementAccessError, match='scene_retirement_reader_active'):
            with access.exclusive_scene_access():
                pytest.fail('retirement entered while real consumer/publisher body was active')
    finally:
        release.set()
        worker.join(4)
    assert not worker.is_alive() and finish.is_set() and errors == []
    with access.exclusive_scene_access():
        pass


def test_real_reader_cannot_reopen_retired_generation(tmp_path, monkeypatch):
    access, policy, member = access_fixture(tmp_path, monkeypatch)
    from blueprint_pipeline import artifixer_completed_training_reuse as existing
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    info = member.stat()
    generation = {
        'schema_version': 'scene_member_generation.v1', 'canonical_path': str(member),
        'generation_id': '1' * 32, 'previous_generation_id': None,
        'owner_intent_id': 'intent-1', 'owner_raw_ref': {'path': str(tmp_path / 'owner.json'),
            'sha256': 'sha256:' + 'a' * 64, 'size_bytes': 1},
        'birth_request_raw_ref': {'path': str(tmp_path / 'birth.json'),
            'sha256': 'sha256:' + 'b' * 64, 'size_bytes': 1},
        'state': 'retired', 'dev': info.st_dev, 'ino': info.st_ino, 'mode': info.st_mode,
        'inventory_sha256': 'sha256:' + 'c' * 64, 'retirement_token': '2' * 32,
        'journal_sha256': 'sha256:' + 'd' * 64, 'state_sequence': 1,
    }
    generation['state_digest'] = canonical_digest(generation, digest_field='state_digest')
    name = hashlib.sha256(str(member).encode()).hexdigest() + '.json'
    path = Path(policy['generation_store']) / name
    path.write_text(json.dumps(generation))
    path.chmod(0o600)
    entered = []
    monkeypatch.setattr(existing, '_read', lambda *a: entered.append(a))
    with pytest.raises(access.SceneRetirementAccessError, match='scene_retirement_generation_unavailable'):
        existing.stage_completed_review(source_root=member, output_root=tmp_path / 'out')
    assert entered == []


def test_disabled_policy_preserves_real_reader_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE', raising=False)
    from blueprint_pipeline import artifixer_completed_training_reuse as existing
    monkeypatch.setattr(existing, '_read', lambda *a: (_ for _ in ()).throw(StopFixture()))
    with pytest.raises(StopFixture):
        existing.stage_completed_review(source_root=tmp_path, output_root=tmp_path / 'out')
