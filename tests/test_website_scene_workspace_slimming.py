"""A finished website job's bulky Atlas intermediates go behind an archive; what sharing and the site read stays.

The reclaim timer never retires a workspace that is shared with deployment companies (their routes and the rights
re-check read it); it slims it instead.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_scene_workspace_slimming.py
#   src/blueprint_pipeline/website_scene_workspace_retention.py

from __future__ import annotations

import functools
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline import website_scene_workspace_retention as retention
from blueprint_pipeline import website_scene_workspace_slimming as slimming
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_website_scene_workspace_retention import (  # noqa: F401 - the autouse disk ledger applies here too
    ARTIFACT_BUCKET,
    BUCKET,
    CAPTURE,
    DAY,
    HOUR,
    SCENE,
    FakeCloud,
    MultipartClient,
    _context,
    _idle,
    _plan,
    _publisher,
    _scene,
    isolated_disk_ledger,
)

MIB = 1024 * 1024


def _big(path: Path, seed: bytes, extra: int = 0) -> bytes:
    data = (seed * (MIB // len(seed) + 1))[:MIB] + seed * (extra + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _atlas_job(capture: Path, **overrides) -> None:
    job = {"state": "ready", "enqueued_at_epoch": time.time() - 6 * DAY, "retry_at_epoch": None,
           "provider_cleanup": {"state": "complete"}, **overrides}
    _json(capture / "pipeline" / "atlas" / "preview_job.json", job)
    (capture / "pipeline" / "atlas" / "preview.lock").touch()  # the Atlas timer's job lock, made on its first step


def _job(tmp_path: Path, *, shared: bool = True, ack: bool = True, **job) -> tuple[Path, Path, dict[str, bytes]]:
    """A finished website job as the host keeps it: Atlas intermediates, the scene preparation and the shared clip."""

    scene, _cloud = _scene(tmp_path, ack=ack)
    capture = scene / "captures" / CAPTURE
    atlas = capture / "pipeline" / "atlas"
    prep = capture / "pipeline" / "website_scene_preparation"
    files = {
        # Bulky intermediates nobody reads once the job is done.
        "pipeline/atlas/downloads/atlas-scene-1.zip": _big(atlas / "downloads" / "atlas-scene-1.zip", b"zip"),
        "pipeline/atlas/pose/frame-0001.png": _big(atlas / "pose" / "frame-0001.png", b"one"),
        "pipeline/atlas/pose/frame-0002.png": _big(atlas / "pose" / "frame-0002.png", b"two"),
        # The provider splat the base scene was levelled from: named by a record that stays.
        "pipeline/atlas/downloads/atlas-scene-1/splat/atlas_refined_splat.ply":
            _big(atlas / "downloads" / "atlas-scene-1" / "splat" / "atlas_refined_splat.ply", b"provider"),
        # A record stays whatever its size, and so does a small file.
        "pipeline/atlas/survey/capture_source_survey.json": b'{"rows": "' + b"x" * (2 * MIB) + b'"}',
        "pipeline/atlas/thumb.png": b"small",
        # What sharing and the site's own view read.
        "pipeline/website_scene_preparation/level/levelled.ply": _big(prep / "level" / "levelled.ply", b"level"),
    }
    for relative in ("pipeline/atlas/survey/capture_source_survey.json", "pipeline/atlas/thumb.png"):
        (capture / relative).parent.mkdir(parents=True, exist_ok=True)
        (capture / relative).write_bytes(files[relative])
    levelled = prep / "level" / "levelled.ply"
    provider = atlas / "downloads" / "atlas-scene-1" / "splat" / "atlas_refined_splat.ply"
    _json(prep / "base_scene.json", {
        "splat_path": str(levelled), "splat_digest": "sha256:" + hashlib.sha256(levelled.read_bytes()).hexdigest(),
        "levelling": {"provider_splat_path": str(provider),
                      "provider_splat_digest": "sha256:" + hashlib.sha256(provider.read_bytes()).hexdigest()}})
    _atlas_job(capture, **job)
    if shared:
        _json(capture / "pipeline" / "company_clip" / "company_clip.json", {
            "schema_version": "website_company_clip_manifest.v1",
            "archive": {"uri": "s3://artifacts/website-company-clip/walkthrough-1.mp4", "digest": "sha256:" + "a" * 64}})
        _json(capture / "pipeline" / "company_clip" / "statement_check.json", {"outcome": "covers_sharing"})
    return scene, capture, files


def _slim_plan(tmp_path: Path, *, age: float = 72 * HOUR, **kwargs) -> dict:
    return slimming.plan_scene_workspace_slim(
        context=kwargs.pop("context", None) or _context(tmp_path, keep_shared_workspaces=True),
        bucket=BUCKET, scene_id=SCENE, now=time.time() + age, process_checker=_idle, **kwargs)


def _slim(tmp_path: Path, plan: dict, client: MultipartClient, **kwargs) -> dict:
    return slimming.apply_scene_workspace_slim(
        plan, context=_context(tmp_path, keep_shared_workspaces=True), ack=kwargs.pop("ack", slimming.SLIM_ACK),
        now=plan["observed_at_epoch"], stream_publisher=kwargs.pop("stream_publisher", None) or _publisher(client),
        process_checker=_idle)


MEMBERS = {"pipeline/atlas/downloads/atlas-scene-1.zip", "pipeline/atlas/pose/frame-0001.png",
           "pipeline/atlas/pose/frame-0002.png"}


def test_the_timer_keeps_a_shared_workspace_that_a_retirement_by_hand_may_retire(tmp_path):
    scene, _capture, _files = _job(tmp_path)
    cloud = FakeCloud()
    timer = _plan(tmp_path, cloud, context=_context(tmp_path, keep_shared_workspaces=True))
    assert timer["status"] == "retained" and timer["reasons"] == [f"shared_with_companies:{CAPTURE}"]
    by_hand = _plan(tmp_path, cloud)
    assert "shared_with_companies" not in " ".join(by_hand["reasons"])


def test_a_copy_taken_back_no_longer_counts_as_shared(tmp_path):
    scene, capture, _files = _job(tmp_path)
    manifest = capture / "pipeline" / "company_clip" / "company_clip.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["archive"]["retracted"] = True
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert not retention.shared_with_companies(capture)
    value["archive"] = {**value["archive"], "retracted": False, "retraction": "pending"}
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert not retention.shared_with_companies(capture)
    value["superseded_archives"] = [{"uri": "s3://artifacts/older.mp4"}]
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert retention.shared_with_companies(capture)
    manifest.write_text("{not json", encoding="utf-8")
    assert retention.shared_with_companies(capture)  # nothing is retired on doubt


def test_a_finished_shared_job_unacknowledged_plans_only_its_unnamed_bulky_atlas_files(tmp_path):
    _job(tmp_path, ack=False)
    plan = _slim_plan(tmp_path)
    assert plan["status"] == "slimmable" and plan["reasons"] == []
    (capture,) = plan["captures"]
    assert {row["relative_path"] for row in capture["members"]} == MEMBERS
    assert capture["member_bytes"] == sum(row["size_bytes"] for row in capture["members"])
    # The provider splat stays: the base scene names it, and the site's preview rebuilds from that record.
    assert plan["kept_bulky_counts"] == {"named_by_kept_record": 1}
    assert plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest")


def test_slimming_archives_with_readback_writes_the_receipt_removes_and_restores_byte_for_byte(tmp_path):
    scene, capture, files = _job(tmp_path)
    plan = _slim_plan(tmp_path)
    client = MultipartClient()

    result = _slim(tmp_path, plan, client)

    (outcome,) = result["captures"]
    assert result["status"] == "slimmed" and outcome["status"] == "slimmed"
    assert result["freed_bytes"] == sum(len(files[name]) for name in MEMBERS)
    for name, data in files.items():
        assert (capture / name).exists() is (name not in MEMBERS)
        if name not in MEMBERS:
            assert (capture / name).read_bytes() == data
    # Sharing's records are untouched, and the Atlas job's folder keeps its records.
    assert (capture / "pipeline" / "company_clip" / "company_clip.json").is_file()
    assert (capture / "pipeline" / "atlas" / "preview_job.json").is_file()
    assert not (capture / "pipeline" / "atlas" / "pose").exists()  # emptied folders go
    receipt = json.loads(slimming.receipt_path(capture).read_text(encoding="utf-8"))
    assert receipt["schema_version"] == slimming.RECEIPT_SCHEMA and receipt["evidence_deleted"] is False
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    archive = receipt["archive"]
    assert archive["filename"] == f"{CAPTURE}.tar" and archive["uri"].startswith(f"s3://{ARTIFACT_BUCKET}/")
    assert f"/{slimming.ARTIFACT_KIND}/sha256/" in archive["uri"]
    assert {row["relative_path"]: row["sha256"] for row in archive["members"]} == {
        name: "sha256:" + hashlib.sha256(files[name]).hexdigest() for name in MEMBERS}
    # Once slimmed, a capture is not planned again.
    again = _slim_plan(tmp_path, age=96 * HOUR)
    assert again["status"] == "nothing_to_slim" and again["captures"][0]["already_slimmed"] is True

    restored = slimming.restore_scene_workspace_slim(capture_root=capture, materializer=functools.partial(
        store.materialize_configured_scene_artifact, client=client, bucket=ARTIFACT_BUCKET))

    assert restored["status"] == "restored" and restored["placed_count"] == len(MEMBERS)
    for name, data in files.items():
        assert (capture / name).read_bytes() == data
    assert not slimming.receipt_path(capture).exists() and Path(restored["historical_receipt"]).is_file()


@pytest.mark.parametrize(("job", "reason"), [
    ({"enqueued_at_epoch": time.time()}, "atlas_job_window_open"),
    ({"provider_cleanup": {"state": "pending"}}, "atlas_provider_cleanup_pending"),
    ({"provider_cleanup": None}, "atlas_provider_cleanup_pending"),
    ({"state": "processing"}, "atlas_job_open"),
    ({"retry_at_epoch": 1}, "atlas_job_open"),
    ({"step_in_flight": {"since_epoch": 1}}, "atlas_job_open"),
])
def test_an_atlas_job_that_may_still_move_by_itself_keeps_everything(tmp_path, job, reason):
    _job(tmp_path, **job)
    plan = _slim_plan(tmp_path)
    assert plan["status"] == "retained" and plan["reasons"] == [f"{reason}:{CAPTURE}"]
    assert plan["captures"] == []


def test_a_recently_active_or_pinned_job_keeps_everything(tmp_path):
    _job(tmp_path)
    assert _slim_plan(tmp_path, age=HOUR)["reasons"] == ["recently_active"]


def test_an_authority_ended_capture_keeps_everything(tmp_path):
    scene, _cloud = _scene(tmp_path, status="terminal_authority_ended")
    _atlas_job(scene / "captures" / CAPTURE)
    _big(scene / "captures" / CAPTURE / "pipeline" / "atlas" / "pose" / "frame.png", b"f")
    assert _slim_plan(tmp_path)["reasons"] == ["authority_ended_capture_kept_local"]


def test_hard_linked_and_preview_source_files_stay(tmp_path):
    scene, capture, files = _job(tmp_path)
    os.link(capture / "pipeline/atlas/pose/frame-0001.png", capture / "pipeline/atlas/pose/frame-0001-copy.png")
    plan = _slim_plan(tmp_path)
    names = {row["relative_path"] for row in plan["captures"][0]["members"]}
    assert names == MEMBERS - {"pipeline/atlas/pose/frame-0001.png"}
    assert plan["kept_bulky_counts"] == {"hard_linked": 2, "named_by_kept_record": 1}


def test_a_member_that_changes_after_the_upload_keeps_every_local_byte(tmp_path):
    scene, capture, files = _job(tmp_path)
    plan = _slim_plan(tmp_path)
    client = MultipartClient()
    publish = _publisher(client)

    def publisher_then_touch(**kwargs):
        reference = publish(**kwargs)
        os.utime(capture / "pipeline/atlas/pose/frame-0002.png", ns=(1, 1))
        return reference

    result = _slim(tmp_path, plan, client, stream_publisher=publisher_then_touch)

    assert result["status"] == "skipped" and result["captures"][0]["reason"] == "candidate_changed"
    assert "published_archive" in result["captures"][0]
    assert all((capture / name).exists() for name in MEMBERS) and not slimming.receipt_path(capture).exists()


def test_apply_refuses_a_plan_it_did_not_make(tmp_path):
    _job(tmp_path)
    plan = _slim_plan(tmp_path)
    with pytest.raises(slimming.WebsiteSceneWorkspaceSlimError):
        _slim(tmp_path, plan, MultipartClient(), ack="retire-scene-workspace")
    forged = {**plan, "captures": [{**plan["captures"][0], "members": [
        {"relative_path": "pipeline/company_clip/company_clip.json", "size_bytes": 1, "identity": [0, 0, 0]}]}]}
    forged["plan_digest"] = canonical_digest(forged, digest_field="plan_digest")
    with pytest.raises(slimming.WebsiteSceneWorkspaceSlimError):
        _slim(tmp_path, forged, MultipartClient())


def test_the_timer_phase_slims_when_enabled_and_only_plans_when_not(tmp_path):
    scene, capture, files = _job(tmp_path)
    client = MultipartClient()

    def run(enabled: bool) -> dict:
        return slimming.slim_scene_workspaces(
            storage_roots=[tmp_path / "pubsub-handoffs"], apply=True, enabled=enabled, now=time.time() + 72 * HOUR,
            context_factory=lambda root: _context(tmp_path, keep_shared_workspaces=True),
            stream_publisher=_publisher(client), process_checker=_idle)

    planned = run(False)
    assert planned["status"] == "dry_run" and planned["candidate_count"] == 1 and planned["slimmed_count"] == 0
    assert all((capture / name).exists() for name in MEMBERS)
    applied = run(True)
    assert applied["slimmed_count"] == 1 and applied["removed_bytes"] == sum(len(files[name]) for name in MEMBERS)
    assert not any((capture / name).exists() for name in MEMBERS)


def test_the_switch_is_on_by_default_and_an_invalid_value_keeps_it_planning():
    assert slimming.slimming_setting({}) == (True, None)
    assert slimming.slimming_setting({slimming.SLIMMING_ENV: "0"}) == (False, None)
    assert slimming.slimming_setting({slimming.SLIMMING_ENV: "maybe"}) == (False, slimming.SLIMMING_INVALID)


def test_one_reclaim_tick_keeps_a_shared_job_whole_and_slims_it(tmp_path):
    from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, run_storage_gc
    from blueprint_pipeline.control_plane_storage_gc_reasons import build_storage_gc_summary

    scene, capture, files = _job(tmp_path)
    _context(tmp_path)  # the pins, queue, intent and binding roots
    client = MultipartClient()
    now = time.time() + 72 * HOUR
    report = run_storage_gc(
        content_store_roots=[], derived_roots=[], queue_roots=[tmp_path / "queue"], pins_root=tmp_path / "pins",
        scene_workspace_roots=[tmp_path / "pubsub-handoffs"], scene_intent_root=tmp_path / "intents",
        scene_binding_root=tmp_path / "bindings", scene_cloud_factory=FakeCloud,
        scene_stream_publisher=_publisher(client), scene_process_checker=_idle, now=lambda: now,
        classifier=lambda *_args, **_kwargs: None, apply=True, ack=RUN_ACK,
        scene_workspace_retirement_enabled=True, scene_workspace_slimming_enabled=True)

    assert report["opt_in"]["scene_workspace_slimming"] is True and "phase_errors" not in report
    assert report["scene_workspaces"]["retained_counts"] == {"shared_with_companies": 1} and scene.is_dir()
    slim = report["scene_workspace_slimming"]
    assert (slim["slimmed_count"], slim["removed_bytes"]) == (1, sum(len(files[name]) for name in MEMBERS))
    assert build_storage_gc_summary(report)["phases"]["scene_workspace_slimming"]["removed_or_offloaded_bytes"] \
        == slim["removed_bytes"]
    assert (capture / "pipeline" / "company_clip" / "company_clip.json").is_file()


def test_a_job_with_no_bulky_atlas_file_left_is_not_searched_every_tick(tmp_path, monkeypatch):
    scene, capture, _files = _job(tmp_path)
    for name in MEMBERS | {"pipeline/atlas/downloads/atlas-scene-1/splat/atlas_refined_splat.ply"}:
        (capture / name).unlink()
    monkeypatch.setattr(slimming, "_named", lambda *_args: pytest.fail("searched with nothing to slim"))
    plan = _slim_plan(tmp_path)
    assert plan["status"] == "nothing_to_slim" and plan["captures"][0]["members"] == []


def test_a_held_or_missing_atlas_preview_lock_uploads_nothing(tmp_path):
    # Withdrawal deletion holds each capture's Atlas preview lock from its listing to its deletion.
    import fcntl
    scene, capture, files = _job(tmp_path)
    plan = _slim_plan(tmp_path)
    never = lambda **_kwargs: pytest.fail("uploaded while the fence was not held")  # noqa: E731
    with (capture / "pipeline" / "atlas" / "preview.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _slim(tmp_path, plan, MultipartClient(), stream_publisher=never)
    assert (result["status"], result["reason"], result["captures"]) == ("skipped", "atlas_preview_busy", [])
    (capture / "pipeline" / "atlas" / "preview.lock").unlink()
    result = _slim(tmp_path, plan, MultipartClient(), stream_publisher=never)
    assert (result["status"], result["reason"]) == ("skipped", "atlas_preview_lock_missing")
    assert all((capture / name).exists() for name in MEMBERS)


def test_a_withdrawal_acknowledged_before_the_upload_uploads_nothing(tmp_path):
    scene, capture, files = _job(tmp_path)
    plan = _slim_plan(tmp_path)
    _json(scene / "website_withdrawal" / "tombstone.json", {"schema_version": "website_capture_withdrawal_tombstone.v1"})
    result = _slim(tmp_path, plan, MultipartClient(), stream_publisher=lambda **_kwargs: pytest.fail("uploaded"))
    assert result["status"] == "skipped"
    assert result["captures"] == [{"capture_id": CAPTURE, "status": "skipped", "reason": "capture_withdrawn"}]
    assert all((capture / name).exists() for name in MEMBERS)


def test_the_fence_is_held_for_the_whole_upload(tmp_path):
    import fcntl
    scene, capture, files = _job(tmp_path)
    plan = _slim_plan(tmp_path)
    publish = _publisher(MultipartClient())
    attempts = []

    def publisher(**kwargs):
        with (capture / "pipeline" / "atlas" / "preview.lock").open("rb") as lock:
            with pytest.raises(BlockingIOError):  # what withdrawal deletion meets: atlas_preview_busy
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempts.append(1)
        return publish(**kwargs)

    assert _slim(tmp_path, plan, MultipartClient(), stream_publisher=publisher)["status"] == "slimmed"
    assert attempts == [1]


# --- owner approval 2026-11-19: the timer retires Blueprint's own internal test jobs once done -----------------


def _timer(tmp_path: Path) -> retention.RetentionContext:
    return retention.timer_context_factory(pins_root=tmp_path / "pins", queue_roots=[tmp_path / "queue"],
                                           intent_root=tmp_path / "intents",
                                           binding_root=tmp_path / "bindings")(tmp_path / "pubsub-handoffs")


def _shared_job(tmp_path: Path, *, internal_test: bool, basis: str = "internal_test") -> tuple[Path, FakeCloud]:
    """A finished, unacknowledged job whose clip is out: an internal test job's, or a site's."""

    scene, cloud = _scene(tmp_path, ack=False)
    _context(tmp_path)  # the pins, queue, intent and binding roots
    capture = scene / "captures" / CAPTURE
    rights = ({"statement_version": "internal_test.v1"} if internal_test
              else {"statement_version": "2026-11-12.v4"})
    marker = {"label": "nav-doorway-96", "footage_id": "reference-job"}
    _json(capture / "pipeline" / "website_task_context.json",
          {"capture_rights": rights, **({"internal_test": marker} if internal_test else {})})
    _json(capture / "pipeline" / "company_clip" / "company_clip.json", {
        "schema_version": "website_company_clip_manifest.v1", "permission": {"permission_basis": basis},
        "archive": {"uri": "s3://artifacts/website-company-clip/walkthrough-1.mp4", "digest": "sha256:" + "a" * 64}})
    return scene, cloud


def test_the_timer_retires_a_done_internal_test_job_although_shared_and_unacknowledged(tmp_path):
    scene, cloud = _shared_job(tmp_path, internal_test=True)
    assert retention.internal_test_job(scene / "captures" / CAPTURE)
    plan = _plan(tmp_path, cloud, context=_timer(tmp_path))
    assert plan["status"] == "retirable" and plan["reasons"] == []
    # Retirement by hand keeps the existing acknowledgement proof.
    assert _plan(tmp_path, cloud)["reasons"] == [f"acknowledgement_unproven:{CAPTURE}"]


def test_a_site_job_stays_whole_while_shared_whatever_its_files_say(tmp_path):
    scene, cloud = _shared_job(tmp_path, internal_test=False, basis="site_sharing_statement")
    assert not retention.internal_test_job(scene / "captures" / CAPTURE)
    assert _plan(tmp_path, cloud, context=_timer(tmp_path))["reasons"] == [
        f"acknowledgement_unproven:{CAPTURE}", f"shared_with_companies:{CAPTURE}"]


def test_a_test_job_with_a_copy_shared_on_another_basis_stays_whole(tmp_path):
    scene, cloud = _shared_job(tmp_path, internal_test=True, basis="site_sharing_statement")
    assert not retention.internal_test_job(scene / "captures" / CAPTURE)
    assert f"shared_with_companies:{CAPTURE}" in _plan(tmp_path, cloud, context=_timer(tmp_path))["reasons"]
