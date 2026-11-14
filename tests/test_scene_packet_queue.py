# Covers: src/blueprint_pipeline/scene_packet_queue.py
# Covers: src/blueprint_pipeline/scene_packet.py
"""The scene packet queue claims a request, writes its status, retries by itself and recovers a killed run.

The world is the old worker's test world (``tests/test_team_scene_packet_worker._world``): the real rigid
fixture scene, its published revision and bundle beside a source launch, a website capture in the
partition, the operator registry, and one request enqueued by scene preparation's own enqueue. Nothing
reaches the network or spends.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import scene_packet
from blueprint_pipeline import scene_packet_queue as queue
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture, scene_facts
from tests.test_team_scene_packet_worker import REQUEST_ID, SCENE_ID, _sealed, _world

RELEASE_1, RELEASE_2 = "1" * 40, "2" * 40


def _run(world: dict[str, Any], monkeypatch, *, release: str | None = None) -> dict[str, Any]:
    monkeypatch.setattr(queue, "running_release", lambda: release)
    return queue.process_queue(queue_root=world["queue"],
                               fetcher=lambda reference: world["scene"]["store"][reference["uri"]])


def _status(world: dict[str, Any]) -> dict[str, Any]:
    return json.loads((world["queue"] / "status" / f"{world['key']}.json").read_text())


def _failing(monkeypatch, error: BaseException) -> None:
    def prepare(**kwargs: Any) -> Any:
        (Path(kwargs["output_root"]) / "partial").write_text("half-written")
        raise error

    monkeypatch.setattr(scene_packet, "prepare", prepare)


def test_a_request_is_claimed_compiled_recorded_and_reused(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    run = _run(world, monkeypatch)
    [status] = run["results"]
    assert (status["status"], status["outcome"], status["blockers"]) == ("completed", "compiled", [])
    assert status == _status(world) and run["provider_mutation_performed"] is False
    assert (world["queue"] / "processed" / f"{world['key']}.json").is_file()
    assert not list((world["queue"] / "pending").iterdir()) and not list((world["queue"] / "processing").iterdir())
    # The derived row points at the half, which the scripted check's reader admits; the check is queued.
    row = profile_for_capture(world["capture"])
    assert row["team_scene_inputs_path"] == status["team_scene_inputs_path"]
    assert row["robot_free_probe_root"] == str(world["run"].resolve()) and row["scenario_id"] == "nominal"
    assert scene_facts(row)["scene"]["task_spec"]["task_success_contract"]
    assert list((tmp_path / "rigid-robot-free-probes" / "pending").glob("*.json"))
    # The same request again reuses the half: nothing is compiled twice.
    monkeypatch.setattr(scene_packet, "prepare", lambda **_: pytest.fail("compiled twice"))
    shutil.copyfile(world["queue"] / "processed" / f"{world['key']}.json",
                    world["queue"] / "pending" / f"{world['key']}.json")
    [again] = _run(world, monkeypatch)["results"]
    assert (again["status"], again["outcome"]) == ("completed", "already_compiled")
    assert again["team_scene_inputs_path"] == status["team_scene_inputs_path"]


def test_a_withdrawn_capture_is_refused_before_anything_compiles_and_never_retried(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    monkeypatch.setattr(scene_packet, "prepare", lambda **_: pytest.fail("compiled a withdrawn capture"))
    marker = world["capture"].parent.parent / "website_withdrawal" / "tombstone.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps(_sealed({"schema_version": "website_capture_withdrawal_tombstone.v1",
                                          "scene_id": SCENE_ID, "request_id": REQUEST_ID}, "digest")))
    [status] = _run(world, monkeypatch)["results"]
    assert status["blockers"] == ["team_scene_capture_withdrawn"]
    assert queue.refusal_retry_cause(status) is None and "refusal_retry" not in status
    assert profile_for_capture(world["capture"]) is None and not world["halves"].exists()
    later = datetime.now(timezone.utc) + timedelta(days=3)
    assert queue.requeue_due(world["queue"], later, None) == []


@pytest.mark.parametrize("blocker", ["team_scene_capture_withdrawn", "team_scene_runtime_source_rights_missing",
                                     "website_preparation_budget_exhausted", "team_scene_owner_contract_invalid:x"])
def test_a_privacy_rights_money_or_owner_contract_block_is_never_timed_retried(blocker):
    assert queue.refusal_retry_cause({"status": "blocked", "blockers": [blocker]}) is None
    assert queue.transient_cause({"status": "blocked", "blockers": [blocker]}) is None


def test_a_transient_failure_returns_after_its_backoff_at_most_three_times(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    _failing(monkeypatch, OSError("disk"))
    [status] = _run(world, monkeypatch)["results"]
    assert status["blockers"] == ["team_scene_packet_failed:OSError"] and status["error"]
    retry = status["automatic_retry"]
    assert (retry["attempts"], retry["limit"]) == (0, 3)
    # The failed attempt's directory is gone; nothing names it.
    assert not any(world["halves"].iterdir())
    not_before = datetime.fromisoformat(retry["not_before"])
    assert queue.requeue_due(world["queue"], not_before - timedelta(seconds=1), None) == []
    for attempt, wait in ((1, 1800), (2, 7200), (3, None)):
        [row] = queue.requeue_due(world["queue"], not_before, None)
        assert row == {"request_key": world["key"], "reason": "team_scene_packet_failed:OSError", "attempt": attempt}
        [status] = _run(world, monkeypatch)["results"]
        retry = status["automatic_retry"]
        if wait is None:
            assert retry["exhausted"] is True
        else:
            not_before = datetime.fromisoformat(retry["not_before"])
    assert queue.requeue_due(world["queue"], not_before + timedelta(days=1), None) == []


def test_a_refusal_returns_on_the_shared_schedule(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    _failing(monkeypatch, scene_packet.ScenePacketError("team_scene_task_kind_unsupported"))
    [status] = _run(world, monkeypatch)["results"]
    record = status["refusal_retry"]
    assert (status["blockers"], record["attempts"], record["limit"]) == (["team_scene_task_kind_unsupported"], 0, 6)
    not_before = datetime.fromisoformat(record["not_before"])
    assert queue.requeue_due(world["queue"], not_before - timedelta(seconds=1), None) == []
    assert queue.requeue_due(world["queue"], not_before, None) == [
        {"request_key": world["key"], "reason": "refusal_retry", "attempt": 1}]
    assert _status(world)["refusal_retry"]["attempts"] == 1


def test_a_new_release_retries_every_block_once(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    _failing(monkeypatch, scene_packet.ScenePacketError("team_scene_capture_withdrawn"))
    [status] = _run(world, monkeypatch, release=RELEASE_1)["results"]
    assert status["release_commit"] == RELEASE_1
    now = datetime.now(timezone.utc)
    assert queue.requeue_due(world["queue"], now, RELEASE_1) == []
    assert queue.requeue_due(world["queue"], now, RELEASE_2) == [
        {"request_key": world["key"], "reason": "new_release", "release_commit": RELEASE_2,
         "blocked_release_commit": RELEASE_1}]
    (world["queue"] / "pending" / f"{world['key']}.json").rename(world["queue"] / "processed" / f"{world['key']}.json")
    assert queue.requeue_due(world["queue"], now, RELEASE_2) == []
    assert queue.requeue_due(world["queue"], now, None) == []


def test_an_interrupted_run_is_recovered_and_its_attempt_removed(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    key = world["key"]
    attempt = world["halves"] / f"{'a' * 16}-{'b' * 8}"
    (attempt / "scene").mkdir(parents=True)
    (world["queue"] / "pending" / f"{key}.json").rename(world["queue"] / "processing" / f"{key}.json")
    (world["queue"] / "status" / f"{key}.json").write_text(json.dumps({
        "schema_version": queue.STATUS_SCHEMA, "request_key": key, "status": "processing",
        "attempt_root": str(attempt), "capture_root": str(world["capture"]), "capture_id": "walkthrough-req-1"}))
    run = _run(world, monkeypatch)
    [recovered] = run["recovered"]
    assert recovered["blockers"] == ["team_scene_packet_interrupted"]
    assert recovered["automatic_retry"]["attempts"] == 0 and not attempt.exists()
    assert (world["queue"] / "processed" / f"{key}.json").is_file() and run["results"] == []
    # The timer's pass returns it after its backoff and claims nothing itself.
    monkeypatch.setattr(queue, "running_release", lambda: None)
    assert queue.recover(queue_root=world["queue"], website_transport=lambda notice: pytest.fail("sent"))[
        "requeued"] == []
    not_before = datetime.fromisoformat(recovered["automatic_retry"]["not_before"])
    assert queue.requeue_due(world["queue"], not_before, None)[0]["reason"] == "team_scene_packet_interrupted"


def test_a_cold_storage_refusal_returns_once_its_run_is_back(tmp_path, monkeypatch):
    from blueprint_pipeline.scene_run_cold_storage import COLD_STORAGE

    world = _world(tmp_path, monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.scene_run_cold_storage.in_cold_storage", lambda run: True)
    away = world["run"].with_name("away")
    world["run"].rename(away)
    [status] = _run(world, monkeypatch)["results"]
    assert status["blockers"] == [COLD_STORAGE]
    now = datetime.now(timezone.utc)
    assert queue.requeue_due(world["queue"], now, None) == []
    away.rename(world["run"])
    assert queue.requeue_due(world["queue"], now, None) == [{"request_key": world["key"],
                                                             "reason": "scene_run_restored"}]


def test_the_command_needs_a_queue_root(capsys, monkeypatch):
    monkeypatch.delenv(queue.QUEUE_ROOT_ENV, raising=False)
    assert queue.main(["--queue-root", ""]) == 2
    assert "team_scene_packet_queue_root_not_configured" in capsys.readouterr().out
