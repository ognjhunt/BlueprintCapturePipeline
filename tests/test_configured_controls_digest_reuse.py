"""ADP-030: retained robot bytes are hashed once per CPU operation, not per ancestor."""
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_configured_controls_autostart_support as support
from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as worker
from blueprint_pipeline import task_evaluation_retained_controls_evidence as retained
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import validation_file_digests as digests
from tests.test_task_evaluation_retained_model_compatibility import _json, _lineage
from tests.test_task_evaluation_scene_intake import stage, attempt


@pytest.mark.parametrize("hash_file", [support._sha256, worker._sha256])
def test_retained_robot_reuse_still_detects_same_size_rewrite(tmp_path, hash_file):
    path = tmp_path / "robot.usd"
    path.write_bytes(b"a" * digests.MINIMUM_BYTES)
    before = path.stat()
    with digests.file_digest_scope():
        original = hash_file(path)
        for _ in range(30):
            assert hash_file(path) == original
        assert digests.digest_scope_stats()["bytes_hashed"] == before.st_size
        assert digests.digest_scope_stats()["cache_hits"] == 30
        path.write_bytes(b"b" * before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert hash_file(path) != original
        assert digests.digest_scope_stats()["bytes_hashed"] == 2 * before.st_size
    with digests.file_digest_scope():
        hash_file(path)
        assert digests.digest_scope_stats()["bytes_hashed"] == before.st_size


def test_status_reuses_robot_bytes_across_attempts_but_not_requests(tmp_path, monkeypatch):
    root = tmp_path / "intents"
    intent = stage(root)
    attempt(root, intent, "first")
    attempt(root, intent, "second")
    robot = tmp_path / "robot.usd"
    robot.write_bytes(b"a" * digests.MINIMUM_BYTES)
    observations = []

    def cancellation(directory, attempt):
        support._sha256(robot)
        observations.append(digests.digest_scope_stats())
        return None

    monkeypatch.setattr(retained, "validated_cancellation", cancellation)
    for _ in range(2):
        result = intake.scene_intent_status(queue_root=root, intent_id=intent["intent_id"], now=102)
        assert len(result["attempts"]) == 2
        assert digests.digest_scope_stats() is None
    assert [row["bytes_hashed"] for row in observations] == [robot.stat().st_size] * 4
    assert [row["cache_hits"] for row in observations] == [0, 1, 0, 1]


@pytest.mark.parametrize('case', ['validated_parent', 'standalone', 'different_parent', 'changed_intent'])
def test_result_reuses_only_its_exact_just_validated_parent(monkeypatch, case):
    from blueprint_pipeline import task_evaluation_configured_controls_autostart_validation as validation
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    parent = {'adoption_digest': 'sha256:' + 'a' * 64}
    intent = {'completed_placement_adoption': parent}
    intent['intent_digest'] = canonical_digest(intent, digest_field='intent_digest')
    result = {'completed_placement_adoption': dict(parent), 'placement_calls_reexecuted': False}
    calls = []
    monkeypatch.setattr(retained, 'validate_placement_adoption', lambda value: calls.append(value))
    kwargs = {'expected_intent_digest': intent['intent_digest'], 'expected_scene_binding_digest': 'scene',
              'expected_task_binding_digest': 'task', 'expected_cpu_checkpoint_binding_digest': 'checkpoint'}
    if case != 'standalone':
        kwargs['_validated_intent'] = intent
    if case == 'different_parent':
        result['completed_placement_adoption']['adoption_digest'] = 'sha256:' + 'b' * 64
    if case == 'changed_intent':
        intent['changed'] = True
    # Reusing the parent never bypasses the independent result validation.
    reason = 'completed_adoption_invalid' if case in {'different_parent', 'changed_intent'} else 'autostart_result_invalid'
    with pytest.raises(validation.TaskEvaluationConfiguredControlsAutostartError, match=reason):
        validation._validate_result(result, **kwargs)
    assert len(calls) == (1 if case == 'standalone' else 0)


def test_status_walks_each_retained_adoption_once_per_read(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_configured_controls_autostart_validation as validation
    owner, cancellation = _lineage(tmp_path / "history", "gpt-6-sol")
    root = tmp_path / "intents"
    intent = stage(root)
    attempt(root, intent, "first")
    attempt(root, intent, "second")
    walked = []
    validate_intent = validation.validate_configured_controls_autostart_intent
    monkeypatch.setattr(validation, "validate_configured_controls_autostart_intent",
                        lambda value: walked.append(value["intent_digest"]) or validate_intent(value))
    monkeypatch.setattr(retained, "validated_cancellation", lambda directory, row: (
        retained.validate_placement_cancellation(receipt=cancellation, attempt=owner) or cancellation))
    for _ in range(2):
        result = intake.scene_intent_status(queue_root=root, intent_id=intent["intent_id"], now=102)
        assert [row["status"] for row in result["attempts"]] == ["cancelled_before_execution"] * 2
    # Three release generations: walked once per status read, not once per cancelled
    # attempt (12 walks before), and never reused across reads.
    assert len(walked) == 6 and len(set(walked)) == 3


def test_reused_adoption_rechecks_owner_seal_and_reopened_bytes(tmp_path):
    owner, cancellation = _lineage(tmp_path / "history", "gpt-6-sol")
    adoption = cancellation["completed_placement_adoption"]
    with digests.file_digest_scope():
        verdict = retained.validate_placement_adoption(adoption, expected_owner_digest=owner["intent_digest"])
        verdict["intent"]["changed_by_caller"] = True
        assert "changed_by_caller" not in retained.validate_placement_adoption(adoption)["intent"]
        with pytest.raises(ValueError, match="completed_placement_adoption_owner_mismatch"):
            retained.validate_placement_adoption(adoption, expected_owner_digest="sha256:" + "b" * 64)
        forged = {**adoption, "source_launch_id": "forged"}  # still claims the reused seal
        with pytest.raises(ValueError, match="completed_placement_adoption_invalid"):
            retained.validate_placement_adoption(forged)
        result_path = Path(adoption["source_result"]["path"])
        _json(result_path, {**json.loads(result_path.read_text()), "rewritten": True})
        with pytest.raises(ValueError, match="completed_placement_adoption_reference_changed"):
            retained.validate_placement_adoption(adoption)
