"""Review tests for the rule that clears offloaded runs older than the withdrawn site (2026-11-27).

The rule compares each pointer's ``offloaded_at_epoch`` with the site's earliest record on the host, read
from what is under the site directory today. These tests ask what happens when today's files are not the
site's first: a workspace retired and then born again, a workspace restored, and what the pointer's time
really is.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_capture_withdrawal_deletion.py
#   src/blueprint_pipeline/website_capture_withdrawal_automatic.py

from __future__ import annotations

import functools
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from blueprint_pipeline import control_plane_evidence_offload as offload
from blueprint_pipeline import pubsub_handoff_disk_admission as disk_admission
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline import website_capture_withdrawal as withdrawal
from blueprint_pipeline import website_capture_withdrawal_automatic as automatic
from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
from blueprint_pipeline import website_capture_withdrawal_drill as drill
from blueprint_pipeline import website_scene_workspace_retention as retention
from tests.test_control_plane_evidence_streaming import MultipartClient
from tests.test_pubsub_handoff_listener import SUBSCRIPTION, FakeBlob, FakeStorageClient
from tests.test_task_evaluation_configured_scene_object_store import _ContentAddressedClient
from tests.test_website_scene_workspace_retention import FakeCloud

REPO = Path(__file__).resolve().parents[1]
BUCKET, REQUEST = "capture-bucket", "reborn1"
SCENE, CAPTURE = f"site-{REQUEST}", f"walkthrough-{REQUEST}"
ARTIFACT_BUCKET = "blueprint-task-evaluation-artifacts-test"
HOUR, DAY = 3600, 86400
UNDETERMINED = "website_withdrawal_deletion_offloaded_run_binding_undetermined"
# The site's own launch run, named as the host names them; nothing in its name or receipt names the site.
SITE_RUN = "website-0a1b2c3d4e5f60718293a4b5-6ad7b73d-20261018t045356z-c94a716392a9-activation-auto-launch"


@pytest.fixture(autouse=True)
def isolated_disk_ledger(tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk

    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT", str(tmp_path / "disk-reservations"))
    roomy = functools.partial(reserve_control_plane_disk,
                              disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, free=80 * 1024**3))
    for module in (retention, disk_admission, offload):
        monkeypatch.setattr(module, "reserve_control_plane_disk", roomy)
    monkeypatch.setattr(retention, "DEFAULT_RESERVATION_ROOT", tmp_path / "disk-reservations")
    monkeypatch.setattr(offload, "DEFAULT_RESERVATION_ROOT", tmp_path / "disk-reservations")


# --- the host, as the real listener, clean-up and retention code leave it ------------------------------


def _payload(**extra: Any) -> bytes:
    return json.dumps({"bucket": BUCKET, "scene_id": SCENE, "capture_id": CAPTURE,
                       "raw_prefix_uri": f"gs://{BUCKET}/scenes/{SCENE}/captures/{CAPTURE}/raw", **extra}).encode()


def _bundle(video: bytes) -> list[FakeBlob]:
    prefix = f"scenes/{SCENE}/captures/{CAPTURE}"
    manifest = {"scene_id": SCENE, "capture_id": CAPTURE, "capture_source": "browser_self_capture",
                "site_submission_id": REQUEST}
    return [FakeBlob(f"{prefix}/raw/manifest.json", json.dumps(manifest).encode()),
            FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
            FakeBlob(f"{prefix}/raw/walkthrough.mov", video)]


def _authority_ended(**_kwargs):
    try:
        raise ValueError("website_control_scene-sponsorship_http_409:consent_expired")
    except ValueError as exc:
        raise listener.PipelineError("website scene failed") from exc


def _deliver(tmp_path: Path, cloud: FakeCloud, payload: bytes, *, ended: bool, video: bytes) -> Path:
    """One delivery through the real handoff listener, acknowledged, as production leaves it."""

    blobs = _bundle(video)
    for blob in blobs:
        cloud.put(blob.name, blob._data)
    result = listener.process_handoff_payload(
        payload, storage_root=tmp_path / "pubsub-handoffs", provider="openai",
        storage_client=FakeStorageClient(blobs), stage_control_plane=True, run_e2e_enabled=False,
        run_e2e=_authority_ended if ended else (lambda **_: {"status": "completed"}))
    capture = tmp_path / "pubsub-handoffs" / BUCKET / "scenes" / SCENE / "captures" / CAPTURE
    assert capture.is_dir(), result
    assert listener._write_ack_receipt(
        capture_root=capture, subscription=SUBSCRIPTION, message_id=f"msg-{len(payload)}",
        payload_digest=listener.payload_sha256(payload), delivery_attempt=1,
        disposition="terminal_authority_ended" if ended else "terminal_success") is None
    return capture


def _move_ledger_back(capture: Path, seconds: float) -> float:
    """The first life happened ``seconds`` ago: every time in the listener's ledger moves back by that much."""

    path = capture / "pipeline_job_ledger.json"

    def moved(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: moved(item) for key, item in value.items()}
        if isinstance(value, list):
            return [moved(item) for item in value]
        if isinstance(value, str) and re.fullmatch(r"\d{4}-\d\d-\d\dT[0-9:.]+(Z|[+-]\d\d:\d\d)", value):
            at = datetime.fromisoformat(value.replace("Z", "+00:00")) - timedelta(seconds=seconds)
            return at.isoformat().replace("+00:00", "Z") if value.endswith("Z") else at.isoformat()
        return value

    ledger = moved(json.loads(path.read_text()))
    path.write_text(json.dumps(ledger))
    return datetime.fromisoformat(ledger["started_at"].replace("Z", "+00:00")).timestamp()


def _context(tmp_path: Path, **overrides: Any) -> retention.RetentionContext:
    for name in ("pins", "queue/pending", "queue/processing", "intents", "bindings"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return retention.RetentionContext(
        storage_root=tmp_path / "pubsub-handoffs", pins_root=tmp_path / "pins", queue_roots=(tmp_path / "queue",),
        intent_root=tmp_path / "intents", binding_root=tmp_path / "bindings", **overrides)


def _retire(tmp_path: Path, cloud: FakeCloud, context: retention.RetentionContext, client: MultipartClient) -> dict:
    """The retention timer's own plan and apply for the scene."""

    plan = retention.plan_scene_workspace_retirement(
        context=context, bucket=BUCKET, scene_id=SCENE, now=time.time() + 72 * HOUR, cloud=cloud,
        process_checker=lambda _path: False)
    assert plan["status"] == "retirable", plan["reasons"]
    return retention.apply_scene_workspace_retirement(
        plan, context=context, ack=retention.RETIRE_ACK, cloud=cloud, now=plan["observed_at_epoch"],
        stream_publisher=functools.partial(store.publish_configured_scene_stream, client=client,
                                           bucket=ARTIFACT_BUCKET),
        process_checker=lambda _path: False)


def _offload_the_sites_run(launches: Path, video: bytes, *, last_written: float, tick: float) -> dict:
    """The site's launch run, sealed, then archived and removed by the real evidence offload at ``tick``."""

    run = launches / SITE_RUN
    (run / "frames").mkdir(parents=True)
    (run / "frames/frame-0001.bin").write_bytes(video)  # the site's own footage, in a file the plan never reads
    (run / "launch_receipt.json").write_text(json.dumps({"status": "completed", "profile": "construction"}))
    for path in (run, *run.rglob("*")):
        os.utime(path, (last_written, last_written))
    manifest = offload.build_evidence_offload_manifest(
        evidence_roots=[launches], hot_window_seconds=2 * DAY, now=lambda: tick, classifier=lambda *_a, **_k: None)
    assert [row["name"] for row in manifest["candidates"]] == [SITE_RUN]
    client = _ContentAddressedClient()
    result = offload.apply_evidence_offload(
        manifest, ack=offload.EXECUTE_ACK, now=lambda: tick,
        publisher=lambda **kwargs: store.publish_configured_scene_artifact(**kwargs, client=client,
                                                                          bucket="blueprint-production-inputs"))
    assert result["offloaded_count"] == 1 and not run.exists(), result
    pointer = json.loads((launches / f"{SITE_RUN}{offload.POINTER_SUFFIX}").read_text())
    return {"pointer": pointer, "client": client}


def _withdraw(capture: Path) -> None:
    withdrawal.acknowledge_withdrawal(capture_root=capture, command={
        "schema_version": "website_capture_withdrawal.v1", "request_id": REQUEST, "scene_id": SCENE,
        "capture_id": CAPTURE, "withdrawal_id": f"withdrawal-{REQUEST}",
        "requested_at_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")})


def _resolution(launches: Path, tmp_path: Path) -> dict:
    """The production resolution's shape: an object store, a source bucket, no archive scan."""

    return {"run_roots": [launches], "intent_root": tmp_path / "intents", "binding_root": tmp_path / "bindings",
            "object_store": deletion.B2ObjectVersions(drill.MemoryB2(), drill.BUCKET),
            "cloud_storage": drill.MemoryGcs([]), "uncovered_roots": (), "screening_queue_root": None,
            "screening_work_root": None, "native_g1_registration_root": None, "team_scene_halves_root": None,
            "team_registry": None, "team_scene_packet_queue_root": None, "rigid_probe_queue_root": None,
            "rigid_probe_work_root": None}


def _plan(capture: Path, launches: Path, tmp_path: Path) -> dict:
    return deletion.plan_withdrawal_deletion(capture_root=capture, **_resolution(launches, tmp_path))


def _rows(plan: dict) -> tuple[list[str], list[str]]:
    cleared = [Path(row["path"]).name for row in plan["offloaded_runs_older_than_site"]["runs"]]
    undetermined = [Path(row["path"]).name for row in plan["unresolved"] if row.get("reason") == UNDETERMINED]
    return cleared, undetermined


def _first_life(tmp_path: Path) -> dict:
    """A site that first came 40 days ago and whose job then ended; its run was offloaded 20 days ago."""

    cloud, client = FakeCloud(), MultipartClient()
    video = b"the site's first walkthrough video"
    capture = _deliver(tmp_path, cloud, _payload(), ended=True, video=video)
    first = _move_ledger_back(capture, 40 * DAY)
    launches = tmp_path / "control-plane/task-evaluation-launch-runs"
    # The run got the site's footage a day after the site came; the clean-up archived it 20 days ago.
    offloaded = _offload_the_sites_run(launches, video, last_written=first + DAY, tick=time.time() - 20 * DAY)
    assert first < offloaded["pointer"]["offloaded_at_epoch"] < time.time() - 19 * DAY
    # Nothing in the sealed pointer names the site, as on the host's 144 pointers.
    assert REQUEST.encode() not in json.dumps(offloaded["pointer"]).encode()
    return {"cloud": cloud, "client": client, "capture": capture, "launches": launches, "first": first,
            "offloaded": offloaded, "video": video}


def _first_life_then_retired(tmp_path: Path) -> dict:
    """The same site, its workspace then retired by the retention code.

    The first life ended for lost authority, the only ending after which the listener takes a new message
    for the same capture. No production entry point retires such a workspace today
    (``allow_authority_ended_archive`` is False everywhere outside tests), so this state needs that switch,
    a legacy capture delivered again, or an operator. Nothing in the rule depends on which.
    """

    world = _first_life(tmp_path)
    context = _context(tmp_path, allow_authority_ended_archive=True)
    retired = _retire(tmp_path, world["cloud"], context, world["client"])
    assert retired["status"] == "retired" and not world["capture"].exists()
    return {**world, "context": context, "receipt": Path(retired["receipt"])}


def _renewed() -> bytes:
    """A different message for the same capture: a new request, as after renewed consent."""

    return _payload(pipeline_handoff_uri=f"gs://{BUCKET}/scenes/{SCENE}/captures/{CAPTURE}/pipeline_handoff.json")


# --- what the review found ------------------------------------------------------------------------------


def test_a_site_born_again_after_its_workspace_was_retired_does_not_clear_the_run_of_its_first_life(tmp_path):
    world = _first_life_then_retired(tmp_path)
    launches, scenes = world["launches"], world["capture"].parents[2]
    # A different message for the ended capture is a new request (the listener's own rule): the workspace is
    # born again beside the retirement receipt, with a new ledger that starts now.
    again = _deliver(tmp_path, world["cloud"], _renewed(), ended=False, video=b"the site's second walkthrough video")
    ledger = json.loads((again / "pipeline_job_ledger.json").read_text())
    assert ledger["status"] == "completed"
    assert datetime.fromisoformat(ledger["started_at"].replace("Z", "+00:00")).timestamp() > time.time() - HOUR
    assert [row.get("source") for row in ledger["attempt_history"]][:1] == ["scene_retirement_receipt"]

    # While the old receipt still has its name the rule says it cannot know, and nothing is cleared.
    _withdraw(again)
    cleared, undetermined = _rows(_plan(again, launches, tmp_path))
    assert (cleared, undetermined) == ([], [SITE_RUN])
    # This first plan must not have sealed anything a later plan would trust.
    journal = again.parents[1] / "website_withdrawal" / deletion.SITE_RECORD_FILENAME
    assert not journal.exists()

    # Since 2026-11-27 retention never touches a workspace that holds a withdrawal tombstone, so after the
    # withdrawal the old receipt keeps its name.
    plan = retention.plan_scene_workspace_retirement(
        context=world["context"], bucket=BUCKET, scene_id=SCENE, now=time.time() + 72 * HOUR,
        cloud=world["cloud"], process_checker=lambda _path: False)
    assert plan["status"] == "retained" and retention.WITHDRAWAL_TOMBSTONE_REASON in plan["reasons"]
    assert world["receipt"].is_file()

    # Before the withdrawal it still can: the retention timer's pass over the born-again workspace finds the
    # old receipt beside it, sees that it does not describe this workspace, and moves it aside
    # (receipt_workspace_mismatch). The same site, in a world where that pass came first:
    world = _first_life_then_retired(tmp_path / "moved-aside-first")
    launches, scenes = world["launches"], world["capture"].parents[2]
    again = _deliver(tmp_path / "moved-aside-first", world["cloud"], _renewed(), ended=False,
                     video=b"the site's second walkthrough video")
    plan = retention.plan_scene_workspace_retirement(
        context=world["context"], bucket=BUCKET, scene_id=SCENE, now=time.time() + 72 * HOUR,
        cloud=world["cloud"], process_checker=lambda _path: False)
    assert plan["status"] == "retirable", plan["reasons"]
    result = retention.apply_scene_workspace_retirement(
        plan, context=world["context"], ack=retention.RETIRE_ACK, cloud=world["cloud"], now=plan["observed_at_epoch"],
        stream_publisher=lambda **_: pytest.fail("nothing is archived on this path"),
        process_checker=lambda _path: False)
    assert (result["status"], result["reason"]) == ("skipped", "receipt_workspace_mismatch"), result
    assert not world["receipt"].exists() and Path(result["quarantined_receipt"]).is_file()
    # What is beside the live workspace now: the receipt under another name, and nothing the rule looks for.
    assert sorted(path.name for path in scenes.iterdir()) == [Path(result["quarantined_receipt"]).name, SCENE]
    assert Path(result["quarantined_receipt"]).name.startswith(f".{SCENE}{retention.RETIRED_SUFFIX}.")
    _withdraw(again)
    journal = again.parents[1] / "website_withdrawal" / deletion.SITE_RECORD_FILENAME

    # The site's run of its first life holds its footage, was offloaded after the site first came, and the
    # hourly pass never reads the archive: it must stay undetermined, and the apply must refuse.
    plan = _plan(again, launches, tmp_path)
    cleared, undetermined = _rows(plan)
    record = plan["offloaded_runs_older_than_site"]["site_earliest_record"]
    assert world["first"] < world["offloaded"]["pointer"]["offloaded_at_epoch"]
    assert cleared == [], (cleared, record)
    assert undetermined == [SITE_RUN]
    # It stays undetermined because the rule sees the retirement, not by chance of the times.
    assert (record["epoch"], record["unavailable_reason"]) == (None, "site_workspace_retired"), record
    assert not journal.exists()
    with pytest.raises(deletion.WithdrawalDeletionError, match="unresolved_copies"):
        deletion.apply_withdrawal_deletion(capture_root=again, expected_plan_digest=plan["digest"],
                                           authorize_complete_deletion=True, **_resolution(launches, tmp_path))
    assert again.is_dir()


def _born_again_and_receipt_moved_aside(tmp_path: Path) -> dict:
    """Born again, and retention's pass moved the old receipt aside before the site withdrew (after it, the
    tombstone stops retention)."""
    world = _first_life_then_retired(tmp_path)
    again = _deliver(tmp_path, world["cloud"], _renewed(), ended=False, video=b"the site's second walkthrough video")
    plan = retention.plan_scene_workspace_retirement(
        context=world["context"], bucket=BUCKET, scene_id=SCENE, now=time.time() + 72 * HOUR,
        cloud=world["cloud"], process_checker=lambda _path: False)
    result = retention.apply_scene_workspace_retirement(
        plan, context=world["context"], ack=retention.RETIRE_ACK, cloud=world["cloud"], now=plan["observed_at_epoch"],
        stream_publisher=lambda **_: pytest.fail("nothing is archived on this path"),
        process_checker=lambda _path: False)
    assert result["reason"] == "receipt_workspace_mismatch", result
    _withdraw(again)
    return {**world, "again": again, "moved_aside": Path(result["quarantined_receipt"])}


@pytest.mark.parametrize("left", ["receipt_moved_aside_only", "ledger_row_only", "receipt_temporary_only"])
def test_each_trace_of_the_retirement_alone_keeps_the_run_of_the_first_life_undetermined(tmp_path, left):
    world = _born_again_and_receipt_moved_aside(tmp_path)
    again, moved = world["again"], world["moved_aside"]
    ledger_path = again / "pipeline_job_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if left != "ledger_row_only":
        # Without the listener's row, only the file beside the workspace says the site was retired.
        ledger["attempt_history"] = [row for row in ledger["attempt_history"]
                                     if row.get("source") != "scene_retirement_receipt"]
        ledger_path.write_text(json.dumps(ledger))
    if left == "ledger_row_only":
        moved.unlink()
    elif left == "receipt_temporary_only":
        moved.rename(moved.with_name(f".{SCENE}{retention.RETIRED_SUFFIX}.1234.0badc0de.tmp"))
    plan = _plan(again, world["launches"], tmp_path)
    record = plan["offloaded_runs_older_than_site"]["site_earliest_record"]
    assert (record["epoch"], record["unavailable_reason"]) == (None, "site_workspace_retired"), record
    assert _rows(plan) == ([], [SITE_RUN])


def test_an_ended_capture_reopened_in_place_keeps_its_first_start(tmp_path):
    # What production does today with a capture that ended for lost authority: its workspace stays on the
    # host, and a new message reopens the same ledger.
    world = _first_life(tmp_path)
    again = _deliver(tmp_path, world["cloud"], _renewed(), ended=False, video=b"the site's second walkthrough video")
    ledger = json.loads((again / "pipeline_job_ledger.json").read_text())
    assert "reopened_after_terminal_authority" in [row.get("status") for row in ledger["attempt_history"]]
    _withdraw(again)
    plan = _plan(again, world["launches"], tmp_path)
    assert plan["offloaded_runs_older_than_site"]["site_earliest_record"]["epoch"] == pytest.approx(
        world["first"], abs=1.0)
    assert _rows(plan) == ([], [SITE_RUN])


def test_a_restored_workspace_keeps_its_first_start_so_the_run_of_its_first_life_stays_undetermined(tmp_path):
    world = _first_life_then_retired(tmp_path)
    capture, launches = world["capture"], world["launches"]
    scene = capture.parents[1]
    # The operator door's restore brings the workspace back and moves the receipt aside.
    retention.restore_scene_workspace(
        receipt_path=world["receipt"], destination=scene, cloud=world["cloud"],
        materializer=functools.partial(store.materialize_configured_scene_artifact, client=world["client"],
                                       bucket=ARTIFACT_BUCKET))
    assert capture.is_dir() and not world["receipt"].exists()
    _withdraw(capture)
    plan = _plan(capture, launches, tmp_path)
    cleared, undetermined = _rows(plan)
    # The restored ledger still says when the site first came, though every raw file was downloaded today.
    # (A rule that says it cannot know after a restore is as safe; a later time is not.)
    record = plan["offloaded_runs_older_than_site"]["site_earliest_record"]
    assert record["epoch"] is None or record["epoch"] == pytest.approx(world["first"], abs=1.0)
    assert (cleared, undetermined) == ([], [SITE_RUN])


def test_the_pointer_time_is_the_clean_up_ticks_start_and_the_margin_is_longer_than_a_tick_may_run(tmp_path):
    launches = tmp_path / "launch-runs"
    run = launches / "run-1"
    run.mkdir(parents=True)
    (run / "launch_receipt.json").write_text("{}")
    tick = time.time() - 5 * DAY
    for path in (run, run / "launch_receipt.json"):
        os.utime(path, (tick - 3 * DAY, tick - 3 * DAY))
    manifest = offload.build_evidence_offload_manifest(
        evidence_roots=[launches], hot_window_seconds=2 * DAY, now=lambda: tick, classifier=lambda *_a, **_k: None)
    # The storage clean-up passes one clock for the whole tick (``clock = lambda: observed_at``): a file that
    # reaches a sealed run after the tick began is archived, and the pointer still carries the tick's start.
    (run / "late.bin").write_bytes(b"written after the tick began")
    client = _ContentAddressedClient()
    result = offload.apply_evidence_offload(
        manifest, ack=offload.EXECUTE_ACK, now=lambda: tick,
        publisher=lambda **kwargs: store.publish_configured_scene_artifact(**kwargs, client=client,
                                                                          bucket="blueprint-production-inputs"))
    assert result["offloaded_count"] == 1
    pointer = json.loads((launches / f"run-1{offload.POINTER_SUFFIX}").read_text())
    assert pointer["offloaded_at_epoch"] == tick
    assert "late.bin" in {row["relative_path"] for row in pointer["members"]}
    source = (REPO / "src/blueprint_pipeline/control_plane_storage_gc.py").read_text()
    assert "clock = lambda: observed_at" in source and "apply_evidence_offload(" in source
    # So the margin has to be longer than one clean-up tick may run; its unit stops it at this limit.
    unit = (REPO / "deploy/systemd/blueprint-control-plane-storage-gc.service").read_text()
    (limit,) = re.findall(r"^TimeoutStartSec=(\d+)h$", unit, flags=re.MULTILINE)
    assert deletion.PREDATES_SITE_MARGIN_SECONDS > int(limit) * HOUR


def test_a_sealed_journal_with_a_later_time_cannot_clear_more_while_the_sites_files_are_there(tmp_path):
    from tests.test_website_capture_withdrawal_deletion import (
        _cleared,
        _host_plan,
        _host_shaped,
        _older_pointer,
        _undetermined,
    )

    world = _host_shaped(tmp_path)
    first = world["first"]
    # The site's own run, offloaded a day after the site came, naming nothing locally.
    _older_pointer(world, world["launches"], "launch-website-suspect-0200", first + DAY, "7")
    journal = world["site"] / "website_withdrawal" / deletion.SITE_RECORD_FILENAME
    tombstone = json.loads((world["site"] / "website_withdrawal/tombstone.json").read_text())
    later = drill._sealed({"schema_version": deletion.SITE_RECORD_SCHEMA, "scene_id": world["site"].name,
                           "tombstone_digest": tombstone["digest"], "epoch": first + 30 * DAY,
                           "ledger_epoch": first + 30 * DAY, "earliest_mtime_epoch": first + 30 * DAY}, "digest")
    drill._write(journal, later)
    plan = _host_plan(world)
    # With the ledger on disk the earlier of the two counts: a correctly sealed later journal changes nothing.
    assert plan["offloaded_runs_older_than_site"]["site_earliest_record"]["epoch"] == pytest.approx(first, abs=1e-3)
    assert _undetermined(plan) == ["launch-website-suspect-0200"] and _cleared(plan) == world["older"]
    assert json.loads(journal.read_text()) == later  # and the plan does not rewrite it


def test_an_apply_that_fails_on_every_pass_is_paged_before_the_withdrawal_is_three_weeks_old(tmp_path):
    from tests.test_website_capture_withdrawal_automatic import Harness

    harness = Harness(tmp_path)
    harness.approve()

    def refused(**_arguments):
        raise deletion.WithdrawalDeletionError("website_withdrawal_deletion_object_delete_refused")

    codes: set[str] = set()
    for hour in range(0, 96, 2):
        summary = automatic.run_pass(
            site_roots=[harness.partition], report_root=harness.report_root, resolution=lambda: harness.resolution,
            quiet=list, approval_path=harness.approval_path, approval_owner_uid=os.getuid(),
            now=harness.now + hour * HOUR, webhook_url="https://alerts.invalid/hook",
            poster=lambda url, report: harness.posts.append(report), applier=refused,
            site_index_root=harness.index_root, indexed_roots=[harness.uncovered])
        codes |= {alert["code"] for alert in summary["alerts"]}
    state = harness.state()
    assert state["status"] == "failed" and state["last_error"] == "website_withdrawal_deletion_object_delete_refused"
    # Four days of failed applies: somebody must have been told.
    assert codes & {"website_withdrawal_deletion_failing", "website_withdrawal_deletion_unresolved"}, (
        codes, state.get("failure_count"), state.get("blocked_since_epoch"))


def test_a_run_offloaded_while_the_withdrawal_waits_puts_it_back_to_unresolved_and_nothing_is_applied(tmp_path):
    from tests.test_website_capture_withdrawal_automatic import Harness, _older_than_the_site

    harness = Harness(tmp_path)
    _older_than_the_site(harness)
    harness.approve(cool_off_hours=72)
    harness.run()
    assert harness.state()["status"] == "planned" and harness.state()["waiting_on"] == ["cool_off"]
    # The clean-up goes on offloading other jobs' runs every day, and their sealed pointers name no site either
    # (none of the host's readable pointers does). One offloaded during the cool-off left the host after the
    # site came, so the rule cannot clear it: the plan is unresolved again and stays so, whatever was sealed
    # in the journal by the first plan.
    launches = harness.world["other_run"].parent
    uri = drill._uri("control-plane-evidence", "9", "evidence.tar")
    harness.world["store"].put(uri, b"another job's run, archived during the cool-off")
    name = "website-another-job-offloaded-during-the-cool-off"
    drill._write(launches / f"{name}.offloaded.v1.json", drill._sealed({
        "schema_version": "control_plane_evidence_offload_pointer.v1", "status": "offloaded", "directory": name,
        "uri": uri, "members": [], "retained_receipt_bytes": {},
        "offloaded_at_epoch": harness.now + 24 * HOUR}, "pointer_digest"))
    harness.run(offset=72 * HOUR + 60)
    state = harness.state()
    assert state["status"] == "unresolved" and harness.applies == [], state
    assert state["unresolved_reasons"] == {UNDETERMINED: 1}
    assert harness.site_files() > 0


# --- second review (2026-11-27): the workspace archive of a receipt moved aside, and a retired tombstone -------


def test_the_workspace_archive_of_a_retired_site_is_in_the_plan_under_the_receipts_own_name(tmp_path):
    world = _first_life_then_retired(tmp_path)
    archive = json.loads(world["receipt"].read_text())["archive"]["uri"]
    _withdraw(world["capture"])
    plan = _plan(world["capture"], world["launches"], tmp_path)
    assert plan["scene_workspace"]["receipt"] == str(world["receipt"])
    assert plan["scene_workspace"]["archive_uri"] == archive


@pytest.mark.xfail(strict=True, reason=(
    "REVIEW FINDING, older than commit 4b2652b: _scene_workspace reads the workspace archive only from "
    "<site>.retired.v1.json. Once that receipt is moved aside (a restore, or a workspace born again) nothing in "
    "the plan names the archive it wrote to the artifact store, so an apply leaves it and no receipt lists it. "
    "With any pointer-only run on the host the narrowed rule keeps such a plan unresolved; that is by chance, "
    "not by design. Remove this marker with the fix."))
@pytest.mark.parametrize("how", ["restored_by_the_operator", "born_again"])
def test_the_workspace_archive_of_a_receipt_moved_aside_is_in_the_plan(tmp_path, how):
    world = _first_life_then_retired(tmp_path)
    capture, launches = world["capture"], world["launches"]
    archive = json.loads(world["receipt"].read_text())["archive"]["uri"]
    # No pointer-only run stands in the way: on such a host the plan below resolves and the apply runs.
    (launches / f"{SITE_RUN}{offload.POINTER_SUFFIX}").unlink()
    if how == "restored_by_the_operator":
        retention.restore_scene_workspace(
            receipt_path=world["receipt"], destination=capture.parents[1], cloud=world["cloud"],
            materializer=functools.partial(store.materialize_configured_scene_artifact, client=world["client"],
                                           bucket=ARTIFACT_BUCKET))
    else:
        _deliver(tmp_path, world["cloud"], _renewed(), ended=False, video=b"the site's second walkthrough video")
        plan = retention.plan_scene_workspace_retirement(
            context=world["context"], bucket=BUCKET, scene_id=SCENE, now=time.time() + 72 * HOUR,
            cloud=world["cloud"], process_checker=lambda _path: False)
        result = retention.apply_scene_workspace_retirement(
            plan, context=world["context"], ack=retention.RETIRE_ACK, cloud=world["cloud"],
            now=plan["observed_at_epoch"], stream_publisher=lambda **_: pytest.fail("nothing is archived here"),
            process_checker=lambda _path: False)
        assert result["reason"] == "receipt_workspace_mismatch", result
    assert not world["receipt"].exists()
    assert [path.name for path in capture.parents[2].glob(f".{SCENE}{retention.RETIRED_SUFFIX}.*")]
    # The archive is still in the artifact store: neither a restore nor the move aside deletes it.
    assert any(archive.endswith(key) for _bucket, key in world["client"].objects)
    _withdraw(capture)
    plan = _plan(capture, launches, tmp_path)
    # Some part of the plan must name it: as an object to delete, or as a row that stops the apply.
    assert archive in json.dumps(plan), (plan["scene_workspace"], plan["unresolved"])


def test_retention_keeps_the_workspace_of_a_withdrawn_site_so_its_tombstone_stays_where_the_pass_looks(tmp_path):
    cloud = FakeCloud()
    capture = _deliver(tmp_path, cloud, _payload(), ended=False, video=b"the site's walkthrough video")
    _withdraw(capture)
    sites, problems = automatic.discover_withdrawals([tmp_path / "pubsub-handoffs"])
    assert [site.name for site in sites] == [SCENE] and not problems
    # The reclaim timer's own context (timer_context_factory), three days after anything last touched the site.
    context = _context(tmp_path, keep_shared_workspaces=True, retire_internal_test_jobs=True)
    plan = retention.plan_scene_workspace_retirement(
        context=context, bucket=BUCKET, scene_id=SCENE, now=time.time() + 72 * HOUR, cloud=cloud,
        process_checker=lambda _path: False)
    assert "website_withdrawal/tombstone.json" not in [row["relative_path"] for row in plan["archive"]], plan["status"]
    assert plan["status"] == "retained", plan["status"]
    assert retention.WITHDRAWAL_TOMBSTONE_REASON in plan["reasons"], plan["reasons"]
    # The reason is the tombstone: the same workspace without it would be retired (next test).
    assert plan["reasons"] == [retention.WITHDRAWAL_TOMBSTONE_REASON], plan["reasons"]


def test_retention_retires_a_workspace_with_no_tombstone_exactly_as_before(tmp_path):
    cloud, client = FakeCloud(), MultipartClient()
    capture = _deliver(tmp_path, cloud, _payload(), ended=False, video=b"the site's walkthrough video")
    context = _context(tmp_path, keep_shared_workspaces=True, retire_internal_test_jobs=True)
    plan = retention.plan_scene_workspace_retirement(
        context=context, bucket=BUCKET, scene_id=SCENE, now=time.time() + 72 * HOUR, cloud=cloud,
        process_checker=lambda _path: False)
    assert plan["status"] == "retirable" and plan["reasons"] == [], plan["reasons"]
    result = retention.apply_scene_workspace_retirement(
        plan, context=context, ack=retention.RETIRE_ACK, cloud=cloud, now=plan["observed_at_epoch"],
        stream_publisher=functools.partial(store.publish_configured_scene_stream, client=client,
                                           bucket=ARTIFACT_BUCKET),
        process_checker=lambda _path: False)
    assert result["status"] == "retired" and not capture.exists(), result


def test_a_tombstone_written_after_the_plan_stops_the_apply(tmp_path):
    cloud, client = FakeCloud(), MultipartClient()
    capture = _deliver(tmp_path, cloud, _payload(), ended=False, video=b"the site's walkthrough video")
    context = _context(tmp_path, keep_shared_workspaces=True, retire_internal_test_jobs=True)
    plan = retention.plan_scene_workspace_retirement(
        context=context, bucket=BUCKET, scene_id=SCENE, now=time.time() + 72 * HOUR, cloud=cloud,
        process_checker=lambda _path: False)
    assert plan["status"] == "retirable", plan["reasons"]
    # The site withdraws between the reclaim timer's plan and its apply. (The apply's unchanged-files check
    # also stops this; the tombstone check is the reason that does not depend on it.)
    _withdraw(capture)
    result = retention.apply_scene_workspace_retirement(
        plan, context=context, ack=retention.RETIRE_ACK, cloud=cloud, now=plan["observed_at_epoch"],
        stream_publisher=functools.partial(store.publish_configured_scene_stream, client=client,
                                           bucket=ARTIFACT_BUCKET),
        process_checker=lambda _path: False)
    assert result["status"] == "skipped" and capture.is_dir(), result
    assert (capture.parents[1] / "website_withdrawal" / "tombstone.json").is_file()
