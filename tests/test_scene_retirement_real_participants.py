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
    monkeypatch.setattr(access, '_SERVICE_IDENTITY', (os.getuid(),os.getgid()))
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



def test_real_materializer_lifetime_blocks_retirement(tmp_path, monkeypatch):
    access, _, member = access_fixture(tmp_path, monkeypatch)
    from blueprint_pipeline import website_scene_dispatch as existing
    from blueprint_pipeline import task_evaluation_scene_owner_authority as authority
    entered, finish, release = threading.Event(), threading.Event(), threading.Event()
    errors = []
    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        raise StopFixture()
    monkeypatch.setattr(authority, 'reopen_scene_intent', paused)
    (tmp_path / 'intent.json').write_bytes(b'{}')
    def operation():
        return existing.materialize_website_attempt(intent_path=tmp_path / 'intent.json',
            source_binding_path=tmp_path / 'binding.json', machinery_path=tmp_path / 'machinery.json',
            release_binding_path=tmp_path / 'release.json', output_root=member, attempt_id='attempt-1')
    worker = threading.Thread(target=run_paused, args=(operation, entered, finish, errors))
    worker.start()
    try:
        assert entered.wait(3), errors
        with pytest.raises(access.SceneRetirementAccessError, match='scene_retirement_reader_active'):
            with access.exclusive_scene_access():
                pytest.fail('materializer lifetime not fenced')
    finally:
        release.set()
        worker.join(4)
    assert not worker.is_alive() and errors == []


def test_birth_cannot_recreate_same_retired_request_generation(tmp_path, monkeypatch):
    access, policy, member = access_fixture(tmp_path, monkeypatch)
    intent_id, owner, birth = authenticated_birth_refs(tmp_path, monkeypatch)
    member.rmdir()
    with access.scene_access():
        generation = access.birth_scene_member(member, owner_intent_id=intent_id,
                                               owner_raw_ref=owner, birth_request_raw_ref=birth, now=101)
    assert member.is_dir() and generation['state'] == 'active'
    assert generation['canonical_path'] == str(member)
    member.rmdir()
    # This is the exact authoritative state transition the action engine will
    # publish under EX, not a caller all-safe flag or a deletion implementation.
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    key = hashlib.sha256(str(member).encode()).hexdigest() + '.json'
    record = Path(policy['generation_store']) / key
    state = json.loads(record.read_text())
    state.update(state='retired', retirement_token='2' * 32, journal_sha256='sha256:' + 'c' * 64)
    state['state_digest'] = canonical_digest(state, digest_field='state_digest')
    record.write_text(json.dumps(state))
    with access.scene_access():
        with pytest.raises(access.SceneRetirementAccessError, match='scene_retirement_generation_unavailable'):
            access.birth_scene_member(member, owner_intent_id=intent_id,
                                     owner_raw_ref=owner, birth_request_raw_ref=birth, now=101)
    assert not member.exists()


def test_actual_birth_holds_same_outer_fence_through_directory_creation(tmp_path, monkeypatch):
    access, _, member = access_fixture(tmp_path, monkeypatch)
    member.rmdir()
    original = os.mkdir
    entered = []
    def mkdir(name, *args, **kwargs):
        if str(name) == member.name:
            with pytest.raises(access.SceneRetirementAccessError, match='scene_retirement_reader_active'):
                with access.exclusive_scene_access():
                    pytest.fail('retirement overlapped actual birth mutation')
            entered.append(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(os, 'mkdir', mkdir)
    intent_id, owner, birth = authenticated_birth_refs(tmp_path, monkeypatch)
    with access.scene_access():
        state = access.birth_scene_member(member, owner_intent_id=intent_id,
                                        owner_raw_ref=owner, birth_request_raw_ref=birth, now=101)
    assert entered == [member.name] and state['state'] == 'active'


def authenticated_birth_refs(tmp_path, monkeypatch):
    from tests.test_task_evaluation_scene_intake import stage, attempt
    from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import record
    root = tmp_path / 'intents'
    intent = stage(root)
    attempt(root, intent)
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT', str(root))
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS', 'webapp')
    return (intent['intent_id'], record(root / intent['intent_id'] / 'intent.json'),
            record(root / intent['intent_id'] / 'attempts' / 'a1.json'))


def test_actual_birth_interprets_only_the_verified_raw_request_bytes(tmp_path, monkeypatch):
    access, _, member = access_fixture(tmp_path, monkeypatch)
    from blueprint_pipeline import task_evaluation_scene_retirement_generations as generations
    intent_id, owner, birth = authenticated_birth_refs(tmp_path, monkeypatch)
    member.rmdir()
    original = generations._read
    repeated = []
    def forbid_unverified_second_parse(path, *args, **kwargs):
        if str(path) in {owner['path'], birth['path']}:
            repeated.append(str(path))
            raise StopFixture('interpreted a separately reacquired unverified raw version')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(generations, '_read', forbid_unverified_second_parse)
    with access.scene_access():
        state = access.birth_scene_member(member, owner_intent_id=intent_id,
                                         owner_raw_ref=owner, birth_request_raw_ref=birth, now=101)
    assert state['state'] == 'active' and repeated == []


def test_actual_birth_cleans_owned_birth_gate_on_store_fsync_failure(tmp_path, monkeypatch):
    access, policy, member = access_fixture(tmp_path, monkeypatch)
    intent_id, owner, birth = authenticated_birth_refs(tmp_path, monkeypatch)
    member.rmdir()
    store_inode = Path(policy['generation_store']).stat().st_ino
    original_open, original_fsync = os.open, os.fsync
    gates = []
    def opened(name, *args, **kwargs):
        fd = original_open(name, *args, **kwargs)
        if str(name).endswith('.lock'):
            gates.append(fd)
        return fd
    def fault(fd):
        if os.fstat(fd).st_ino == store_inode:
            raise OSError('injected store fsync failure')
        return original_fsync(fd)
    monkeypatch.setattr(os, 'open', opened)
    monkeypatch.setattr(os, 'fsync', fault)
    try:
        with pytest.raises(OSError, match='injected store fsync failure'):
            access.birth_scene_member(member, owner_intent_id=intent_id,
                                     owner_raw_ref=owner, birth_request_raw_ref=birth, now=101)
        assert gates
        for fd in gates:
            with pytest.raises(OSError):
                os.fstat(fd)
        assert not member.exists()
    finally:
        for fd in gates:
            try:
                os.fstat(fd)
            except OSError:
                continue
            os.close(fd)
