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

import pytest

from blueprint_pipeline import completed_scene_preparation_retention as preparation
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from tests.test_control_plane_evidence_streaming import MultipartClient
from tests.test_website_scene_workspace_retention import REQUEST, _intent, _request

DAY = 86400
BUCKET = "blueprint-task-evaluation-artifacts-test"


MIB = 1024 * 1024


def _aged(path: Path, days: float) -> None:
    stamp = time.time() - days * DAY
    for item in sorted(path.rglob("*"), reverse=True):
        os.utime(item, (stamp, stamp), follow_symlinks=False)
    os.utime(path, (stamp, stamp))


def _folder(tmp_path: Path, intent_id: str, *, days: float = 8) -> Path:
    """One attempt's output: its records, a materialized file a record names, and materialized bulk none does."""
    folder = tmp_path / "completed-scene-preparation" / intent_id
    attempt = folder / "source-1"
    (attempt / "materialized" / "submission").mkdir(parents=True)
    (attempt / "materialized" / "scene.usdc").write_bytes(b"u" * (2 * MIB))
    (attempt / "materialized" / "frames.bin").write_bytes(b"f" * MIB)
    (attempt / "materialized" / "submission" / "bundle.ply").write_bytes(b"p" * MIB)
    request = attempt / "materialized" / "submission" / "request.json"
    request.write_text('{"preparation_id": "p"}', encoding="utf-8")
    (attempt / "factory.json").write_text(json.dumps({
        "submission_request": {"path": str(request)}, "bundle": {"path": str(attempt / "materialized/submission/bundle.ply")}}),
        encoding="utf-8")
    _aged(folder, days)
    return folder


MEMBERS = {"source-1/materialized/scene.usdc", "source-1/materialized/frames.bin"}


def _intent_named(tmp_path: Path, submission: str, **kwargs) -> str:
    for name in ("intents", "bindings", "pins", "queue/pending", "queue/processing"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return _intent(tmp_path, {**_request(time.time() + 30 * DAY), "submission_id": submission}, **kwargs)


def _run(tmp_path: Path, client: MultipartClient | None = None, *, enabled: bool = True, **kwargs) -> dict:
    return preparation.offload_finished_intent_outputs(
        roots=[tmp_path / "completed-scene-preparation"], pins_root=tmp_path / "pins",
        queue_roots=[tmp_path / "queue"], intent_root=tmp_path / "intents", binding_root=tmp_path / "bindings",
        apply=True, enabled=enabled, now=time.time(), classifier=lambda *_args, **_kwargs: None,
        **{"stream_publisher": functools.partial(store.publish_configured_scene_stream,
                                                 client=client or MultipartClient(), bucket=BUCKET),
           "process_checker": lambda _path: False, "retract": lambda uri: pytest.fail(f"retracted {uri}"), **kwargs})


def test_a_finished_intents_unnamed_bulk_is_archived_records_stay_and_it_restores_byte_for_byte(tmp_path):
    intent_id = _intent_named(tmp_path, "done", finished="completed")
    folder = _folder(tmp_path, intent_id)
    before = {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    client = MultipartClient()

    report = _run(tmp_path, client)

    assert report["offloaded_count"] == 1 and report["offloaded_bytes"] == 3 * MIB
    assert report["kept_bulky_counts"] == {"named_by_record": 1}
    left = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
    assert left == set(before) - MEMBERS | {"/".join(preparation.RECORD_PARTS)}
    record = json.loads(preparation.record_path(folder).read_text(encoding="utf-8"))
    assert record["uri"] == record["archive"]["uri"] and record["evidence_deleted"] is False
    assert {row["relative_path"] for row in record["archive"]["members"]} == MEMBERS
    assert _run(tmp_path, client)["retained_counts"] == {"already_archived": 1}
    restored = preparation.restore_folder(folder=folder, materializer=functools.partial(
        store.materialize_configured_scene_artifact, client=client, bucket=BUCKET))
    assert restored["placed_count"] == 2
    assert all((folder / name).read_bytes() == data for name, data in before.items())


def test_a_withdrawal_that_deletes_the_folder_during_the_upload_takes_the_upload_back(tmp_path):
    import shutil
    intent_id = _intent_named(tmp_path, "done", finished="completed")
    folder = _folder(tmp_path, intent_id)
    publish = functools.partial(store.publish_configured_scene_stream, client=MultipartClient(), bucket=BUCKET)
    retracted = []

    def withdrawn_meanwhile(**kwargs):
        reference = publish(**kwargs)
        shutil.rmtree(folder)
        return reference

    report = _run(tmp_path, stream_publisher=withdrawn_meanwhile, retract=retracted.append)

    (outcome,) = report["results"]
    assert outcome["reason"] == "folder_changed_during_archive" and retracted == [outcome["retracted"]["uri"]]
    assert not folder.exists()


def test_an_open_unknown_recent_or_queued_intents_folder_stays(tmp_path):
    open_intent = _intent_named(tmp_path, "open")
    queued = _intent_named(tmp_path, "queued", finished="completed")
    recent = _intent_named(tmp_path, "recent", finished="revoked")
    folders = [_folder(tmp_path, open_intent), _folder(tmp_path, "scene-" + "0" * 64), _folder(tmp_path, queued),
               _folder(tmp_path, recent, days=2)]
    (tmp_path / "queue" / "pending" / "row.json").write_text(json.dumps({"intent_id": queued}), encoding="utf-8")

    report = _run(tmp_path)

    assert report["candidate_count"] == 0 and report["offloaded_count"] == 0
    assert report["retained_counts"] == {"intent_open": 1, "intent_unknown": 1, "queue_referenced": 1,
                                         "recently_active": 1}
    assert all((folder / name).is_file() for folder in folders for name in MEMBERS)


def test_a_folder_in_use_stays_and_unreadable_intents_keep_every_folder(tmp_path):
    done = _intent_named(tmp_path, "done", finished="completed")
    folder = _folder(tmp_path, done)
    assert _run(tmp_path, process_checker=lambda path: path == folder)["retained_counts"] == {"in_use": 1}
    (tmp_path / "intents" / done / "intent.json").write_text("{", encoding="utf-8")
    assert _run(tmp_path)["retained_counts"] == {"reference_index_unreadable": 1}
    assert all((folder / name).is_file() for name in MEMBERS)


def test_switched_off_it_only_plans_and_each_tick_is_bounded(tmp_path):
    folders = [_folder(tmp_path, _intent_named(tmp_path, f"done-{index}", finished="completed")) for index in range(3)]
    planned = _run(tmp_path, enabled=False)
    assert (planned["status"], planned["candidate_count"], planned["offloaded_count"]) == ("dry_run", 3, 0)
    applied = _run(tmp_path, max_per_tick=2)
    assert (applied["offloaded_count"], applied["deferred_count"]) == (2, 1)
    assert sum(preparation.record_path(folder).is_file() for folder in folders) == 2


def test_no_root_is_no_phase_and_the_switch_is_on_by_default(tmp_path):
    assert preparation.offload_finished_intent_outputs(
        roots=[tmp_path / "absent"], pins_root=tmp_path, queue_roots=[], intent_root=None, binding_root=None,
        apply=True, enabled=True, now=time.time()) is None
    assert preparation.setting({}) == (True, None)
    assert preparation.setting({preparation.SWITCH_ENV: "0"}) == (False, None)
    assert preparation.configured_roots({}) == [preparation.ROOT]
    assert REQUEST["source"]["kind"] == "capture_bundle"  # the intents above are the retention tests' own
