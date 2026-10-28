"""Hermetic withdrawal deletion: temporary roots, a fake B2 client and a fake GCS bucket only.

Beta journey step: site withdrawal. Nothing here reaches a provider, a bucket or
the control-plane host; the fakes model the boto3 S3 version API the existing
artifact store client exposes and the existing exact GCS cleanup storage.
"""
# Covers: src/blueprint_pipeline/website_capture_withdrawal_deletion.py
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from blueprint_pipeline import website_capture_withdrawal as withdrawal
from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)

BUCKET = "blueprint-artifacts"
PREFIX = "blueprint/arm-decision-proof-v1/configured-scenes/artifacts"



def _sealed(value: dict, field: str) -> dict:
    """Seal a record the way its writer does."""

    value = {**value, field: ""}
    value[field] = canonical_digest(value, digest_field=field)
    return value

def _uri(kind: str, digit: str, name: str) -> str:
    return f"s3://{BUCKET}/{PREFIX}/{kind}/sha256/{digit * 64}/{name}"


class FakeB2:
    """``list_object_versions``/``delete_object``/``get_object`` as boto3 answers them on B2."""

    def __init__(self):
        self.objects: dict[str, list[dict]] = {}
        self.deleted: list[tuple[str, str]] = []
        self.reads: list[str] = []

    def put(self, uri: str, body: bytes, *, versions: int = 1, marker: bool = False) -> None:
        key = uri.split(f"s3://{BUCKET}/", 1)[1]
        rows = [{"VersionId": f"v{index}", "Body": body, "marker": False} for index in range(versions)]
        if marker:
            rows.append({"VersionId": "dm", "Body": b"", "marker": True})
        self.objects[key] = rows

    def list_object_versions(self, *, Bucket, Prefix, **_markers):
        assert Bucket == BUCKET
        versions, markers = [], []
        for key, rows in self.objects.items():
            if key.startswith(Prefix):
                for row in rows:
                    (markers if row["marker"] else versions).append({"Key": key, "VersionId": row["VersionId"]})
        return {"Versions": versions, "DeleteMarkers": markers, "IsTruncated": False}

    def delete_object(self, *, Bucket, Key, VersionId):
        assert Bucket == BUCKET
        self.deleted.append((Key, VersionId))
        self.objects[Key] = [row for row in self.objects[Key] if row["VersionId"] != VersionId]

    def get_object(self, *, Bucket, Key):
        self.reads.append(Key)
        return {"Body": io.BytesIO(next(row["Body"] for row in self.objects[Key] if not row["marker"]))}

    def present(self, uri: str) -> bool:
        return bool(self.objects.get(uri.split(f"s3://{BUCKET}/", 1)[1]))


class FakeGcs:
    bucket_name = "synthetic-source"

    def __init__(self):
        self.rows = [{"name": "scenes/site-req1/captures/walkthrough-req1/raw/video.mp4", "generation": "7",
                      "size": 15, "crc32c": "AAAAAA=="}]

    def objects(self, prefix):
        return [dict(row) for row in self.rows if row["name"].startswith(prefix)]

    def soft_deleted_objects(self, prefix):
        return []

    def delete(self, name, generation):
        self.rows = [row for row in self.rows if (row["name"], row["generation"]) != (name, generation)]


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return path


def _archive(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


@pytest.fixture
def world(tmp_path):
    capture = tmp_path / "pubsub/source-bucket/scenes/site-req1/captures/walkthrough-req1"
    _write(capture / "raw/manifest.json", {"site_submission_id": "req1", "scene_id": "site-req1",
                                           "capture_id": "walkthrough-req1", "consent_status": "granted"})
    _write(capture / "raw/video.mp4", b"synthetic-video")
    _write(capture / "pipeline/website_scene_preparation/preparation.json", {"scene_id": "site-req1"})
    withdrawal.acknowledge_withdrawal(capture_root=capture, command={
        "schema_version": "website_capture_withdrawal.v1", "request_id": "req1", "scene_id": "site-req1",
        "capture_id": "walkthrough-req1", "withdrawal_id": "withdrawal-req1",
        "requested_at_iso": "2026-10-15T00:00:00Z"})
    control, inputs = tmp_path / "control-plane", tmp_path / "inputs"
    launches, canaries = control / "task-evaluation-launch-runs", control / "task-evaluation-policy-canaries"
    prepared = inputs / "completed-scene-preparation"
    for root in (launches, canaries, prepared):
        root.mkdir(parents=True)
    # Lineage: the site's website source registration and the scene intent it serves.
    request = {"owner": {"tenant": "site-tenant"}, "submission_id": "sub-1", "source": {"kind": "website"}}
    intent_id = "scene-" + "c" * 64
    _write(control / "task-evaluation-scene-intents" / intent_id / "intent.json",
           {"intent_id": intent_id, "request": request})
    _write(control / "task-evaluation-scene-intents" / ("scene-" + "d" * 64) / "intent.json",
           {"intent_id": "scene-" + "d" * 64, "request": {**request, "submission_id": "other"}})
    _write(control / "website-source-bindings" / "registration.json", _sealed({
        "schema_version": "website_scene_source_registration.v1",
        "request_digest": cross_runtime_canonical_digest(request),
        "capture_source": {"request_id": "req1", "scene_id": "site-req1", "capture_id": "walkthrough-req1"},
        "references": {"preparation": {"path": str(capture / "pipeline/website_scene_preparation/preparation.json")}}},
        "registration_digest"))
    # A launch run that names only its scene intent, as a scene-owned launch profile does.
    construction = launches / "launch-req1-construction-0001"
    _write(construction / "launch_profile.json", {"scene_attempt_binding": {"intent_id": intent_id},
                                                  "allocator": {"provider": "vast"}})
    _write(construction / "launch_receipt.json", {"launch_id": construction.name, "status": "completed"})
    _write(construction / "frames/0001.png", b"\x89PNG synthetic frame")
    _write(launches / f".{construction.name}.offload-1234.tar", b"interrupted staging archive")
    # A team run that names its source launch, with a registered remote result artifact on B2.
    canary = canaries / "team-scored-run-00000001"
    _write(canary / "dispatch_receipt.json", {"source_launch_id": construction.name,
                                              "controls_launch_id": "launch-req1-controls-0002"})
    # This site's own file names are not another site.
    _write(canary / "inputs.jsonl", b'{"video": "uploads/walkthrough-req1.mov"}\n'
                                    b'{"receipt": "/x/scenes/site-req1.retired.v1.json"}\n')
    review = _uri("task-evaluation-result", "a", "review.mp4")
    _write(canary / "artifacts/result_delivery/remote_artifacts/review.json",
           {"schema_version": "task_evaluation_result_artifact_remote_reference.v1", "uri": review})
    # An offloaded launch run: only its pointer (with retained receipt bytes) is local; the
    # launch profile naming its scene intent is inside the archive.
    offloaded = _uri("control-plane-evidence", "b", "evidence.tar")
    receipt = json.dumps({"launch_id": "launch-req1-controls-0002", "status": "completed"}).encode()
    _write(launches / "launch-req1-controls-0002.offloaded.v1.json", _sealed({
        "schema_version": "control_plane_evidence_offload_pointer.v1", "directory": "launch-req1-controls-0002",
        "uri": offloaded, "members": [], "retained_receipt_bytes": {
            "launch_receipt.json": base64.b64encode(receipt).decode()}}, "pointer_digest"))
    # The intent-keyed scene preparation output.
    _write(prepared / intent_id / "website-source/binding.json", {"intent_digest": "sha256:" + "e" * 64})
    # Another site's run and archive: never touched.
    other = launches / "launch-req2-construction-0003"
    _write(other / "launch_profile.json", {"capture_root": "/x/scenes/site-req2/captures/walkthrough-req2"})
    other_uri = _uri("control-plane-evidence", "f", "evidence.tar")
    _write(launches / "launch-req2-controls-0004.offloaded.v1.json", _sealed({
        "schema_version": "control_plane_evidence_offload_pointer.v1", "directory": "launch-req2-controls-0004",
        "uri": other_uri, "members": [], "retained_receipt_bytes": {}, "scene_id": "site-req2"}, "pointer_digest"))
    # The retired scene workspace's receipt and archive.
    workspace = _uri("website-scene-workspace", "9", "workspace.tar")
    retired = {"schema_version": "website_scene_workspace_retired.v1", "bucket": "source-bucket",
               "scene_id": "site-req1", "archive": {"uri": workspace, "members": []}, "receipt_digest": ""}
    retired["receipt_digest"] = canonical_digest(retired, digest_field="receipt_digest")
    _write(capture.parents[2] / "site-req1.retired.v1.json", retired)
    store = FakeB2()
    store.put(review, b"review bytes")
    store.put(offloaded, _archive({"launch_profile.json": json.dumps(
        {"scene_attempt_binding": {"intent_id": intent_id}}).encode(), "frames/1.png": b"\x89PNG"}),
        versions=2, marker=True)
    store.put(workspace, b"workspace archive")
    store.put(other_uri, _archive({"launch_profile.json": b'{"scene_id": "site-req2"}'}))
    uncovered = inputs / "sam31-preparations"
    uncovered.mkdir()
    resolution = {"run_roots": (launches, canaries, prepared), "intent_root": control / "task-evaluation-scene-intents",
                  "binding_root": control / "website-source-bindings",
                  "object_store": deletion.B2ObjectVersions(store, BUCKET), "uncovered_roots": (uncovered,),
                  "scan_offloaded_archives": True}
    return {"capture": capture, "site": capture.parents[1], "store": store, "resolution": resolution,
            "construction": construction, "canary": canary, "other": other, "launches": launches,
            "prepared": prepared / intent_id, "review": review, "offloaded": offloaded, "workspace": workspace,
            "other_uri": other_uri, "intent_id": intent_id, "uncovered": uncovered}


def _plan(world, **overrides):
    return deletion.plan_withdrawal_deletion(capture_root=world["capture"], **{**world["resolution"], **overrides})


def _apply(world, plan, **overrides):
    return deletion.apply_withdrawal_deletion(capture_root=world["capture"], expected_plan_digest=plan["digest"],
                                              authorize_complete_deletion=True,
                                              **{**world["resolution"], **overrides})


def _snapshot(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def test_dry_run_resolves_every_pipeline_copy_and_deletes_nothing(world, tmp_path):
    before, objects = _snapshot(tmp_path), json.dumps(world["store"].objects, default=str, sort_keys=True)
    plan = _plan(world)
    assert plan["unresolved"] == []
    assert {run["name"] for run in plan["runs"]} == {
        "launch-req1-construction-0001", "team-scored-run-00000001", "launch-req1-controls-0002",
        world["intent_id"]}
    assert {row["uri"] for row in plan["objects"]} == {world["review"], world["offloaded"], world["workspace"]}
    offloaded = next(row for row in plan["objects"] if row["uri"] == world["offloaded"])
    assert len(offloaded["versions"]) == 3  # two versions and a delete marker
    assert plan["lineage"]["intents"] == [world["intent_id"]]
    assert plan["providers_named_in_run_evidence"] == ["vast"]
    assert plan["scene_files"]["count"] == 3
    assert _snapshot(tmp_path) == before
    assert json.dumps(world["store"].objects, default=str, sort_keys=True) == objects
    assert world["store"].deleted == []
    # Archives are read only for offloaded runs whose local records cannot show their site.
    assert sorted(key.split("/sha256/")[1][0] for key in world["store"].reads) == ["b"]


def test_without_reading_archives_an_offloaded_run_stays_undetermined(world):
    plan = _plan(world, scan_offloaded_archives=False)
    assert [row["reason"] for row in plan["unresolved"]] == [
        "website_withdrawal_deletion_offloaded_run_binding_undetermined"]
    assert plan["unresolved"][0]["path"].endswith("launch-req1-controls-0002")
    with pytest.raises(deletion.WithdrawalDeletionError, match="unresolved_copies"):
        _apply(world, plan, scan_offloaded_archives=False)
    assert world["store"].deleted == []


def test_apply_deletes_exactly_the_plan_and_names_what_remains(world):
    gcs = FakeGcs()
    plan = _plan(world, cloud_storage=gcs)
    receipt = _apply(world, plan, cloud_storage=gcs)
    for path in (world["construction"], world["canary"], world["prepared"],
                 world["launches"] / "launch-req1-controls-0002.offloaded.v1.json",
                 world["launches"] / ".launch-req1-construction-0001.offload-1234.tar",
                 world["site"].parent / "site-req1.retired.v1.json"):
        assert not path.exists(), path
    assert not any(world["store"].present(uri) for uri in (world["review"], world["offloaded"], world["workspace"]))
    assert len(world["store"].deleted) == 5
    # Another site's run, pointer and archive are untouched; so is the retained audit journal.
    assert (world["other"] / "launch_profile.json").exists() and world["store"].present(world["other_uri"])
    assert (world["launches"] / "launch-req2-controls-0004.offloaded.v1.json").exists()
    assert [path for path in world["site"].rglob("*")
            if path.is_file() and "website_withdrawal" not in path.parts] == []
    assert gcs.rows == []
    assert receipt["covered_copies_deleted"] is True and receipt["provider_copies_deleted"] is False
    assert receipt["deleted"]["scene_files"] == 3
    holders = {row["holder"]: row for row in receipt["remaining"]}
    assert holders["providers_named_in_run_evidence"]["names"] == ["vast"]
    assert {"gpu_and_remote_cpu_providers", "model_and_reconstruction_providers",
            "object_store_provider_physical_erasure", "webapp_and_delivered_copies"} <= set(holders)
    assert holders["pipeline_root_not_covered"]["path"] == str(world["uncovered"])
    assert "capture_source_bucket" not in holders
    assert receipt["pipeline_controlled_deletion_complete"] is False  # a root it does not cover is present
    sealed = world["site"] / "website_withdrawal/deletion_receipts" / f"{plan['digest'][7:]}.json"
    assert json.loads(sealed.read_text()) == receipt
    assert receipt["digest"] == canonical_digest(receipt, digest_field="digest")


def test_archived_atlas_previews_of_the_walkthrough_are_found_by_name_even_after_retirement(world):
    # The workspace is retired (its receipt is all that is local), yet the preview archive is found.
    mine = _uri("atlas-scene-bundle", "3", "walkthrough-req1.zip")
    theirs = _uri("atlas-scene-bundle", "4", "walkthrough-req2.zip")
    world["store"].put(mine, b"preview", versions=2, marker=True)
    world["store"].put(theirs, b"another site's preview")
    plan = _plan(world)
    assert plan["unresolved"] == []
    found = {row["uri"]: row for row in plan["objects"]}
    assert len(found[mine]["versions"]) == 3 and theirs not in found
    _apply(world, plan)
    assert not world["store"].present(mine) and world["store"].present(theirs)


def test_withdrawal_finds_only_its_job_scoped_revision_previews_without_local_capture_records(world):
    child = "supplement-" + "a" * 40
    mine = [_uri("atlas-scene-bundle", digit, f"req1-{capture}.zip")
            for digit, capture in (("5", child), ("6", "supplement-" + "b" * 40))]
    others = [_uri("atlas-scene-bundle", "7", f"{request}-{child}.zip")
              for request in ("req2", "req11")]
    others.append(_uri("atlas-scene-bundle", "8", "req1-supplement-invalid.zip"))
    for uri in [*mine, *others]:
        world["store"].put(uri, b"synthetic preview", versions=2, marker=True)
    assert not list(world["capture"].parent.glob("supplement-*"))
    plan = _plan(world)
    assert plan["unresolved"] == []
    found = {row["uri"] for row in plan["objects"]}
    assert set(mine) <= found and not set(others) & found
    _apply(world, plan)
    assert all(not world["store"].present(uri) for uri in mine)
    assert all(world["store"].present(uri) for uri in others)


@pytest.mark.parametrize("capture_id", ["walkthrough-req1", "supplement-" + "a" * 40])
def test_deletion_waits_for_a_running_preview_step_and_then_holds_it_off(world, capture_id):
    import fcntl
    from blueprint_pipeline import website_atlas_preview as preview
    job_dir = world["capture"].parent / capture_id / "pipeline/atlas"
    job_dir.mkdir(parents=True)
    (job_dir / preview.LOCK_FILE).touch()
    (job_dir / preview.JOB_FILE).write_text("{}")
    plan = _plan(world)
    with (job_dir / preview.LOCK_FILE).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(deletion.WithdrawalDeletionError, match="atlas_preview_busy"):
            _apply(world, plan)
    assert world["store"].deleted == []
    _apply(world, plan)
    # The local cleanup removes the job and its lock with the site's other files.
    assert not (job_dir / preview.JOB_FILE).exists() and not (job_dir / preview.LOCK_FILE).exists()


def test_deletion_holds_every_revision_lock_and_releases_earlier_locks_on_busy_refusal(world, monkeypatch):
    import fcntl
    from blueprint_pipeline.website_atlas_preview_store import JOB_FILE, LOCK_FILE
    job_dirs = [world["capture"] / "pipeline/atlas", *[
        world["capture"].parent / ("supplement-" + char * 40) / "pipeline/atlas" for char in ("a", "b")]]
    for directory in job_dirs:
        directory.mkdir(parents=True)
        (directory / LOCK_FILE).touch()
        (directory / JOB_FILE).write_text("{}")
    plan = _plan(world)
    with (job_dirs[-1] / LOCK_FILE).open("r") as busy:
        fcntl.flock(busy, fcntl.LOCK_EX)
        with pytest.raises(deletion.WithdrawalDeletionError, match="atlas_preview_busy"):
            _apply(world, plan)
    assert world["store"].deleted == []
    for directory in job_dirs:
        with (directory / LOCK_FILE).open("r") as released:
            fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)

    object_store = world["resolution"]["object_store"]
    delete = object_store.delete
    def while_locked(key, version):
        for directory in job_dirs:
            with (directory / LOCK_FILE).open("r") as held:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        delete(key, version)
    monkeypatch.setattr(object_store, "delete", while_locked)
    _apply(world, plan)
    assert all(not (directory / JOB_FILE).exists() for directory in job_dirs)


def test_missing_revision_preview_lock_refuses_before_any_deletion(world):
    from blueprint_pipeline.website_atlas_preview_store import JOB_FILE
    job = world["capture"].parent / ("supplement-" + "a" * 40) / "pipeline/atlas" / JOB_FILE
    _write(job, {})
    with pytest.raises(deletion.WithdrawalDeletionError, match="atlas_preview_lock_missing"):
        _apply(world, _plan(world))
    assert world["store"].deleted == [] and job.is_file()


def test_previews_cannot_be_ruled_out_without_the_object_store(world):
    plan = _plan(world, object_store=None)
    assert {"prefix": f"{PREFIX}/atlas-scene-bundle/sha256/",
            "reason": "website_withdrawal_deletion_object_store_unconfigured"} in plan["unresolved"]


def test_complete_only_with_source_bucket_and_no_uncovered_root(world):
    plan = _plan(world, uncovered_roots=())
    receipt = _apply(world, plan, uncovered_roots=())
    holders = {row["holder"] for row in receipt["remaining"]}
    assert "capture_source_bucket" in holders and receipt["pipeline_controlled_deletion_complete"] is False
    gcs = FakeGcs()
    world["resolution"]["uncovered_roots"] = ()
    plan = _plan(world, cloud_storage=gcs)
    assert _apply(world, plan, cloud_storage=gcs)["pipeline_controlled_deletion_complete"] is True


@pytest.mark.parametrize("fault", ["unauthorized", "digest", "changed"])
def test_apply_needs_authorization_and_the_unchanged_reviewed_plan(world, tmp_path, fault):
    plan = _plan(world)
    if fault == "changed":
        _write(world["construction"] / "late.json", {"written": "after review"})
    before = _snapshot(tmp_path)
    with pytest.raises(deletion.WithdrawalDeletionError) as error:
        deletion.apply_withdrawal_deletion(
            capture_root=world["capture"], authorize_complete_deletion=fault != "unauthorized",
            expected_plan_digest="sha256:" + "0" * 64 if fault == "digest" else plan["digest"], **world["resolution"])
    assert str(error.value).endswith({"unauthorized": "explicit_authorization_required", "digest": "plan_changed",
                                      "changed": "plan_changed"}[fault])
    assert _snapshot(tmp_path) == before and world["store"].deleted == []


def test_local_cleanup_deletes_only_the_reviewed_scene_files(world, tmp_path, monkeypatch):
    plan = _plan(world)
    real = withdrawal.plan_local_cleanup

    def late_writer(**kwargs):
        # A file written after the reviewed plan was re-resolved, before the local cleanup.
        _write(world["capture"] / "late-upload.mov", b"never reviewed")
        return real(**kwargs)

    monkeypatch.setattr(withdrawal, "plan_local_cleanup", late_writer)
    with pytest.raises(deletion.WithdrawalDeletionError, match="scene_files_changed"):
        _apply(world, plan)
    assert (world["capture"] / "late-upload.mov").exists()


@pytest.mark.parametrize("fault,reason", [
    ("symlink", "unsafe_entry"),
    ("another_site", "run_names_another_site"),
    ("undetermined_pointer", "offloaded_run_binding_undetermined"),
    ("unreadable_archive", "offloaded_archive_unreadable"),
    ("foreign_bucket", "object_outside_artifact_bucket"),
    ("shared_object", "object_shared_with_unbound_run"),
    ("no_object_store", "object_store_unconfigured"),
    ("unreadable_intent", "record_unreadable"),
    ("unsealed_registration", "registration_unsealed"),
    ("forged_pointer", "pointer_unsealed"),
])
def test_anything_unresolved_fails_closed_before_any_deletion(world, tmp_path, fault, reason):
    overrides = {}
    if fault == "symlink":
        os.symlink("/etc", world["construction"] / "escape")
    elif fault == "another_site":
        _write(world["canary"] / "mixed.json", {"scene_id": "site-req9"})
    elif fault in ("undetermined_pointer", "unreadable_archive"):
        _write(world["launches"] / "launch-unknown-0005.offloaded.v1.json", _sealed({
            "schema_version": "control_plane_evidence_offload_pointer.v1", "directory": "launch-unknown-0005",
            "uri": _uri("control-plane-evidence", "8", "evidence.tar"), "members": [], "retained_receipt_bytes": {}},
            "pointer_digest"))
        if fault == "undetermined_pointer":
            overrides["scan_offloaded_archives"] = False
    elif fault == "foreign_bucket":
        _write(world["canary"] / "artifacts/result_delivery/remote_artifacts/legacy.json",
               {"uri": "s3://legacy-spaces-bucket/review.mp4"})
    elif fault == "shared_object":
        _write(world["other"] / "artifacts/result_delivery/remote_artifacts/review.json", {"uri": world["review"]})
    elif fault == "no_object_store":
        overrides["object_store"] = None
    elif fault == "unsealed_registration":
        # A stale registration naming this scene but another intent must not seed deletion.
        _write(Path(world["resolution"]["binding_root"]) / "stale.json", {
            "schema_version": "website_scene_source_registration.v1", "request_digest": "sha256:" + "d" * 64,
            "capture_source": {"scene_id": "site-req1"}, "references": {}})
    elif fault == "forged_pointer":
        # A bound run's pointer rewritten to name another run's archive, without the writer's seal.
        path = world["launches"] / "launch-req1-controls-0002.offloaded.v1.json"
        _write(path, {**json.loads(path.read_text()), "uri": world["other_uri"]})
    else:
        _write(Path(world["resolution"]["intent_root"]) / ("scene-" + "1" * 64) / "intent.json", b"{not json")
    plan = _plan(world, **overrides)
    assert any(row["reason"].endswith(reason) for row in plan["unresolved"]), plan["unresolved"]
    before = _snapshot(tmp_path)
    with pytest.raises(deletion.WithdrawalDeletionError, match="unresolved_copies"):
        _apply(world, plan, **overrides)
    assert _snapshot(tmp_path) == before and world["store"].deleted == []


@pytest.mark.parametrize("names_site", [True, False])
def test_reading_an_offloaded_archive_resolves_an_undetermined_pointer(world, names_site):
    unknown = _uri("control-plane-evidence", "7", "evidence.tar")
    _write(world["launches"] / "launch-unknown-0006.offloaded.v1.json", _sealed({
        "schema_version": "control_plane_evidence_offload_pointer.v1", "directory": "launch-unknown-0006",
        "uri": unknown, "members": [], "retained_receipt_bytes": {}}, "pointer_digest"))
    body = b'{"capture_root": "/x/scenes/site-req1/captures/walkthrough-req1"}' if names_site \
        else b'{"scene_id": "site-req3"}'
    world["store"].put(unknown, _archive({"run/launch_profile.json": body, "run/frame.png": b"\x00binary"}))
    assert any("undetermined" in row["reason"] for row in _plan(world, scan_offloaded_archives=False)["unresolved"])
    plan = _plan(world, scan_offloaded_archives=True)
    assert plan["unresolved"] == []
    assert ("launch-unknown-0006" in {run["name"] for run in plan["runs"]}) is names_site
    _apply(world, plan, scan_offloaded_archives=True)
    assert world["store"].present(unknown) is not names_site


def test_legal_hold_and_missing_tombstone_refuse(world, tmp_path):
    _write(world["site"] / "website_withdrawal/legal_hold.json", {"legal_hold": True})
    plan = _plan(world)
    with pytest.raises(deletion.WithdrawalDeletionError, match="legal_hold"):
        _apply(world, plan)
    assert world["store"].deleted == [] and world["construction"].exists()
    fresh = tmp_path / "fresh/scenes/site-req5/captures/walkthrough-req5"
    fresh.mkdir(parents=True)
    with pytest.raises(deletion.WithdrawalDeletionError, match="tombstone_required"):
        deletion.plan_withdrawal_deletion(capture_root=fresh, **world["resolution"])


def test_cli_delete_is_a_dry_run_unless_applied_with_authorization(world, tmp_path, monkeypatch, capsys):
    roots = []
    for root in world["resolution"]["run_roots"]:
        roots += ["--run-root", str(root)]
    argv = ["website_capture_withdrawal", "delete", "--capture-root", str(world["capture"]), *roots,
            "--intent-root", str(world["resolution"]["intent_root"]),
            "--binding-root", str(world["resolution"]["binding_root"])]
    before = _snapshot(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    withdrawal.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "dry_run" and printed["plan"]["schema_version"] == deletion.PLAN_SCHEMA
    # Without --object-store the B2 copies cannot be deleted and the offloaded run cannot be read.
    assert {row["reason"] for row in printed["plan"]["unresolved"]} == {
        "website_withdrawal_deletion_object_store_unconfigured",
        "website_withdrawal_deletion_offloaded_run_binding_undetermined"}
    assert _snapshot(tmp_path) == before
    monkeypatch.setattr(sys, "argv", [*argv, "--apply", "--expected-plan-digest", printed["plan"]["digest"]])
    with pytest.raises(deletion.WithdrawalDeletionError, match="explicit_authorization_required"):
        withdrawal.main()
    assert _snapshot(tmp_path) == before
