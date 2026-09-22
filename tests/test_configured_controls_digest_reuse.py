"""ADP-030: retained robot bytes are hashed once per CPU operation, not per ancestor."""
import os

import pytest

from blueprint_pipeline import task_evaluation_configured_controls_autostart_support as support
from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as worker
from blueprint_pipeline import task_evaluation_retained_controls_evidence as retained
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import validation_file_digests as digests
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
