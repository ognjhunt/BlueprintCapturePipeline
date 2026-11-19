"""A finished scene intent's preparation output goes behind a pointer once every reader is done with it.

Owner approval 2026-11-19: "Give `completed-scene-preparation` a cleanup rule".
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/completed_scene_preparation_retention.py

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import completed_scene_preparation_retention as preparation
from blueprint_pipeline import control_plane_evidence_offload as offload
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from tests.test_control_plane_evidence_streaming import MultipartClient
from tests.test_website_scene_workspace_retention import REQUEST, _intent, _request

DAY = 86400
BUCKET = "blueprint-task-evaluation-artifacts-test"


@pytest.fixture(autouse=True)
def roomy_disk(tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk

    monkeypatch.setattr(offload, "reserve_control_plane_disk", functools.partial(
        reserve_control_plane_disk, disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, free=80 * 1024**3)))
    monkeypatch.setattr(offload, "DEFAULT_RESERVATION_ROOT", tmp_path / "disk-reservations")


def _aged(path: Path, days: float) -> None:
    stamp = time.time() - days * DAY
    for item in sorted(path.rglob("*"), reverse=True):
        os.utime(item, (stamp, stamp), follow_symlinks=False)
    os.utime(path, (stamp, stamp))


def _folder(tmp_path: Path, intent_id: str, *, days: float = 8) -> Path:
    folder = tmp_path / "completed-scene-preparation" / intent_id
    (folder / "source-1").mkdir(parents=True)
    (folder / "source-1" / "scene.usdc").write_bytes(b"usd" * 1000)
    (folder / "website-source").mkdir()
    (folder / "website-source" / "binding.json").write_text('{"intent": "%s"}' % intent_id, encoding="utf-8")
    _aged(folder, days)
    return folder


def _intent_named(tmp_path: Path, submission: str, **kwargs) -> str:
    for name in ("intents", "bindings", "pins", "queue/pending", "queue/processing"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return _intent(tmp_path, {**_request(time.time() + 30 * DAY), "submission_id": submission}, **kwargs)


def _run(tmp_path: Path, client: MultipartClient | None = None, *, enabled: bool = True, **kwargs) -> dict:
    return preparation.offload_finished_intent_outputs(
        roots=[tmp_path / "completed-scene-preparation"], pins_root=tmp_path / "pins",
        queue_roots=[tmp_path / "queue"], intent_root=tmp_path / "intents", binding_root=tmp_path / "bindings",
        apply=True, enabled=enabled, now=time.time(), classifier=lambda *_args, **_kwargs: None,
        stream_publisher=functools.partial(store.publish_configured_scene_stream, client=client or MultipartClient(),
                                           bucket=BUCKET),
        **{"process_checker": lambda _path: False, **kwargs})


def test_a_finished_intents_idle_folder_goes_behind_a_pointer_and_restores_byte_for_byte(tmp_path):
    intent_id = _intent_named(tmp_path, "done", finished="completed")
    folder = _folder(tmp_path, intent_id)
    before = {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    client = MultipartClient()

    report = _run(tmp_path, client)

    assert report["status"] == "applied" and report["offloaded_count"] == 1 and not folder.exists()
    pointer = folder.parent / f"{intent_id}{offload.POINTER_SUFFIX}"
    assert json.loads(pointer.read_text(encoding="utf-8"))["evidence_deleted"] is False
    offload.restore_offloaded_evidence(pointer_path=pointer, destination=folder, materializer=functools.partial(
        store.materialize_configured_scene_artifact, client=client, bucket=BUCKET))
    assert {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()} == before


def test_an_open_unknown_recent_or_queued_intents_folder_stays(tmp_path):
    open_intent = _intent_named(tmp_path, "open")
    queued = _intent_named(tmp_path, "queued", finished="completed")
    recent = _intent_named(tmp_path, "recent", finished="revoked")
    folders = [_folder(tmp_path, open_intent), _folder(tmp_path, "scene-" + "0" * 64), _folder(tmp_path, queued),
               _folder(tmp_path, recent, days=2)]
    (tmp_path / "queue" / "pending" / "row.json").write_text(json.dumps({"intent_id": queued}), encoding="utf-8")

    report = _run(tmp_path, process_checker=lambda _path: False)

    assert report["candidate_count"] == 0 and report["status"] == "dry_run"
    assert {reason: row["count"] for reason, row in report["retained_by_reason"].items()} == {
        "intent_open": 1, "intent_unknown": 1, "queue_referenced": 1, "unsealed_recent": 1}
    assert all(folder.is_dir() for folder in folders)


def test_a_folder_in_use_stays_and_unreadable_intents_keep_every_folder(tmp_path):
    done = _intent_named(tmp_path, "done", finished="completed")
    folder = _folder(tmp_path, done)
    assert _run(tmp_path, process_checker=lambda path: path == folder)["retained_by_reason"]["in_use"]["count"] == 1
    (tmp_path / "intents" / done / "intent.json").write_text("{", encoding="utf-8")
    report = _run(tmp_path)
    assert list(report["retained_by_reason"]) == ["reference_index_unreadable"] and folder.is_dir()


def test_switched_off_it_only_plans_and_each_tick_is_bounded(tmp_path):
    folders = [_folder(tmp_path, _intent_named(tmp_path, f"done-{index}", finished="completed")) for index in range(3)]
    planned = _run(tmp_path, enabled=False)
    assert (planned["status"], planned["candidate_count"]) == ("dry_run", 3) and all(path.is_dir() for path in folders)
    applied = _run(tmp_path, max_per_tick=2)
    assert (applied["offloaded_count"], applied["deferred_count"]) == (2, 1)
    assert sum(path.is_dir() for path in folders) == 1


def test_no_root_is_no_phase_and_the_switch_is_on_by_default(tmp_path):
    assert preparation.offload_finished_intent_outputs(
        roots=[tmp_path / "absent"], pins_root=tmp_path, queue_roots=[], intent_root=None, binding_root=None,
        apply=True, enabled=True, now=time.time()) is None
    assert preparation.setting({}) == (True, None)
    assert preparation.setting({preparation.SWITCH_ENV: "0"}) == (False, None)
    assert preparation.configured_roots({}) == [preparation.ROOT]
    assert REQUEST["source"]["kind"] == "capture_bundle"  # the intents above are the retention tests' own
