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
    container = tmp_path / 'payloads'
    container.mkdir()
    member = container / 'scene'
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
        def operation():
            return existing.stage_completed_review(source_root=member, output_root=tmp_path / 'out')
    else:
        from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as existing
        manifest = member / 'manifest.json'
        manifest.write_text('{"input_namespace":"scene-test"}')
        locks = tmp_path / 'publisher-locks'
        locks.mkdir()
        monkeypatch.setattr(existing, '_publish_locked', paused)
        def operation():
            return existing.publish_scene_configuration_submission(
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


def test_fifo_record_is_refused_before_open(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    fifo = tmp_path / 'fifo'
    os.mkfifo(fifo)
    original = os.open
    attempted = []
    def guarded(name, *args, **kwargs):
        if str(name) == 'fifo':
            attempted.append(name)
            raise StopFixture('FIFO open must never be attempted')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(os, 'open', guarded)
    with pytest.raises(access.SceneRetirementAccessError):
        access._read(fifo)
    assert attempted == []


def test_foreign_first_successful_fd_is_never_closed(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    target = tmp_path / 'foreign'
    target.write_bytes(b'not-the-coordinator')
    foreign = os.open(target, os.O_RDONLY)
    original = os.open
    def reused(name, *args, **kwargs):
        return foreign if str(name) == '/' else original(name, *args, **kwargs)
    monkeypatch.setattr(os, 'open', reused)
    try:
        with pytest.raises(access.SceneRetirementAccessError):
            with access._opened('/', directory=True):
                pytest.fail('foreign initial descriptor adopted')
        assert os.fstat(foreign).st_ino == target.stat().st_ino
    finally:
        os.close(foreign)


def test_real_reader_denies_retired_nested_member_under_container_root(tmp_path, monkeypatch):
    access, policy, member = access_fixture(tmp_path, monkeypatch)
    from blueprint_pipeline import artifixer_completed_training_reuse as existing
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    policy['roots'][0]['root'] = str(member.parent)
    policy['policy_digest'] = canonical_digest(policy, digest_field='policy_digest')
    Path(os.environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE']).write_text(json.dumps(policy))
    info = member.stat()
    generation = {
        'schema_version': 'scene_member_generation.v1', 'canonical_path': str(member),
        'state': 'retired', 'generation_id': '1' * 32,
        'dev': info.st_dev, 'ino': info.st_ino, 'mode': info.st_mode,
    }
    generation['state_digest'] = canonical_digest(generation, digest_field='state_digest')
    key = hashlib.sha256(str(member).encode()).hexdigest() + '.json'
    record = Path(policy['generation_store']) / key
    record.write_text(json.dumps(generation))
    record.chmod(0o600)
    attempted = []
    def must_not_read(*args):
        attempted.append(args)
        raise StopFixture('stale nested member reopened')
    monkeypatch.setattr(existing, '_read', must_not_read)
    with access.exclusive_scene_access():
        pass
    with pytest.raises(access.SceneRetirementAccessError, match='scene_retirement_generation_unavailable'):
        existing.stage_completed_review(source_root=member, output_root=tmp_path / 'out')
    assert attempted == []


def test_reused_retained_parent_is_refused_before_child_lookup(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    foreign_tree = tmp_path / 'foreign'
    foreign_tree.mkdir()
    (foreign_tree / 'private').mkdir()
    foreign = os.open(foreign_tree, os.O_RDONLY | os.O_DIRECTORY)
    original = access._open_owned
    reused = []
    def switched(name, *args, **kwargs):
        fd, info = original(name, *args, **kwargs)
        if name == '/':
            os.close(fd)
            os.dup2(foreign, fd)
            reused.append(fd)
        return fd, info
    monkeypatch.setattr(access, '_open_owned', switched)
    lookups = []
    original_stat = os.stat
    def named(name, *args, **kwargs):
        if str(name) == 'private':
            lookups.append(name)
        return original_stat(name, *args, **kwargs)
    monkeypatch.setattr(os, 'stat', named)
    try:
        with pytest.raises(access.SceneRetirementAccessError):
            with access._opened('/private', directory=True):
                pytest.fail('foreign parent traversal adopted')
        assert lookups == []
        assert os.fstat(reused[0]).st_ino == foreign_tree.stat().st_ino
    finally:
        for fd in reused:
            os.close(fd)
        os.close(foreign)


def test_known_close_failure_refuses_and_still_cleans_other_tokens(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    original = os.close
    failed, closed = [], []
    def fault(fd):
        if not failed:
            failed.append(fd)
            raise OSError('injected owned close fault')
        closed.append(fd)
        return original(fd)
    monkeypatch.setattr(os, 'close', fault)
    try:
        with pytest.raises(access.SceneRetirementAccessError, match='scene_retirement_descriptor_cleanup_failed'):
            with access._opened('/private', directory=True):
                pass
        assert failed and closed
    finally:
        for fd in failed:
            original(fd)

