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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import website_capture_withdrawal as withdrawal
from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
from blueprint_pipeline import website_capture_withdrawal_drill as drill
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
                  "scan_offloaded_archives": True, "screening_queue_root": control / "beta-policy-screening",
                  "screening_work_root": tmp_path / "work/beta-policy-screening-work",
                  # Named but absent until a test writes the site's G1 registration, a team scene half, the
                  # team registry or one of the two team queues.
                  "native_g1_registration_root": control / "native-g1-site-registrations",
                  "team_scene_halves_root": inputs / "team-scene-halves",
                  "team_registry": tmp_path / "etc/team-evaluation-registry.json",
                  "team_scene_packet_queue_root": control / "team-scene-packets",
                  "rigid_probe_queue_root": control / "rigid-robot-free-probes",
                  "rigid_probe_work_root": tmp_path / "work/rigid-robot-free-probe-work",
                  "unclaimed_handoff_root": tmp_path / "handoffs/.website_preparation_unclaimed"}
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


def test_published_scene_previews_of_the_job_are_found_by_name_and_deleted(world):
    # The job page's web-sized scene previews (website_scene_preview): the walkthrough's and an
    # update's, found by name even with no local capture records; other jobs' stay.
    child = "supplement-" + "a" * 40
    mine = [_uri("website-scene-preview", "3", "walkthrough-req1.ply"),
            _uri("website-scene-preview", "4", f"req1-{child}.ply")]
    others = [_uri("website-scene-preview", "5", "walkthrough-req2.ply"),
              _uri("website-scene-preview", "6", f"req11-{child}.ply"),
              _uri("website-scene-preview", "7", f"req1-{child}.zip")]
    for uri in [*mine, *others]:
        world["store"].put(uri, b"synthetic preview", versions=2, marker=True)
    plan = _plan(world)
    assert plan["unresolved"] == []
    found = {row["uri"]: row for row in plan["objects"]}
    assert set(mine) <= set(found) and not set(others) & set(found)
    assert len(found[mine[0]]["versions"]) == 3
    receipt = _apply(world, plan)
    assert all(not world["store"].present(uri) for uri in mine)
    assert all(world["store"].present(uri) for uri in others)
    assert set(mine) <= {row["uri"] for row in receipt["deleted"]["objects"]}


def test_company_clips_of_the_job_are_found_by_name_and_deleted_with_the_local_copy(world):
    # The silent clip deployment companies watch (website_company_clip): every archived version of the
    # walkthrough's and a supplement's is deleted by name, the host copy goes with the scene, and a pass
    # holding the clip lock makes deletion wait. Other jobs' clips stay.
    import fcntl
    from blueprint_pipeline import website_company_clip as company_clip
    child = "supplement-" + "a" * 40
    mine = [_uri("website-company-clip", "3", "walkthrough-req1.mp4"),
            _uri("website-company-clip", "4", f"req1-{child}.mp4")]
    others = [_uri("website-company-clip", "5", "walkthrough-req2.mp4"),
              _uri("website-company-clip", "6", f"req11-{child}.mp4")]
    for uri in [*mine, *others]:
        world["store"].put(uri, b"synthetic clip", versions=2, marker=True)
    directory = world["capture"].joinpath(*company_clip.DIRECTORY)
    _write(directory / company_clip.CLIP_FILE, b"synthetic clip")
    _write(directory / company_clip.MANIFEST_FILE, {"schema_version": company_clip.MANIFEST_SCHEMA})
    (directory / company_clip.LOCK_FILE).touch()
    plan = _plan(world)
    assert plan["unresolved"] == []
    found = {row["uri"]: row for row in plan["objects"]}
    assert set(mine) <= set(found) and not set(others) & set(found)
    assert len(found[mine[0]]["versions"]) == 3
    with (directory / company_clip.LOCK_FILE).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(deletion.WithdrawalDeletionError, match="company_clip_busy"):
            _apply(world, plan)
    assert world["store"].deleted == [] and (directory / company_clip.CLIP_FILE).exists()
    receipt = _apply(world, plan)
    assert all(not world["store"].present(uri) for uri in mine)
    assert all(world["store"].present(uri) for uri in others)
    assert set(mine) <= {row["uri"] for row in receipt["deleted"]["objects"]}
    assert not (directory / company_clip.CLIP_FILE).exists()
    assert not (directory / company_clip.MANIFEST_FILE).exists()


def test_company_clips_cannot_be_ruled_out_without_the_object_store(world):
    plan = _plan(world, object_store=None)
    assert {"prefix": f"{PREFIX}/website-company-clip/sha256/",
            "reason": "website_withdrawal_deletion_object_store_unconfigured"} in plan["unresolved"]


def test_simulation_packages_of_the_job_are_found_by_name_and_deleted_with_the_local_copy(world):
    # The simulation package deployment companies download (website_company_simulation): every archived
    # version of the walkthrough's and a supplement's is deleted by name, the host copy goes with the scene,
    # and a pass holding the package lock makes deletion wait. Other jobs' packages stay.
    import fcntl
    from blueprint_pipeline import website_company_simulation as company_simulation
    child = "supplement-" + "b" * 40
    mine = [_uri("website-company-simulation", "3", "walkthrough-req1.zip"),
            _uri("website-company-simulation", "4", f"req1-{child}.zip")]
    others = [_uri("website-company-simulation", "5", "walkthrough-req2.zip"),
              _uri("website-company-simulation", "6", f"req11-{child}.zip")]
    for uri in [*mine, *others]:
        world["store"].put(uri, b"synthetic package", versions=2, marker=True)
    directory = world["capture"].joinpath(*company_simulation.DIRECTORY)
    _write(directory / company_simulation.PACKAGE_FILE, b"synthetic package")
    _write(directory / company_simulation.MANIFEST_FILE, {"schema_version": company_simulation.MANIFEST_SCHEMA})
    (directory / company_simulation.LOCK_FILE).touch()
    plan = _plan(world)
    assert plan["unresolved"] == []
    found = {row["uri"]: row for row in plan["objects"]}
    assert set(mine) <= set(found) and not set(others) & set(found)
    assert len(found[mine[0]]["versions"]) == 3
    with (directory / company_simulation.LOCK_FILE).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(deletion.WithdrawalDeletionError, match="company_simulation_busy"):
            _apply(world, plan)
    assert world["store"].deleted == [] and (directory / company_simulation.PACKAGE_FILE).exists()
    receipt = _apply(world, plan)
    assert all(not world["store"].present(uri) for uri in mine)
    assert all(world["store"].present(uri) for uri in others)
    assert set(mine) <= {row["uri"] for row in receipt["deleted"]["objects"]}
    assert not (directory / company_simulation.PACKAGE_FILE).exists()
    assert not (directory / company_simulation.MANIFEST_FILE).exists()


def test_slimmed_atlas_archives_of_the_job_are_found_by_name_and_deleted(world):
    # A finished job's slimmed Atlas files (website_scene_workspace_slimming) sit in the store under the
    # capture's name, as its clip does: every version of the walkthrough's and a supplement's goes, also once
    # the workspace is gone. Other jobs' archives stay.
    child = "supplement-" + "b" * 40
    mine = [_uri("website-scene-workspace-slim", "7", "walkthrough-req1.tar"),
            _uri("website-scene-workspace-slim", "8", f"req1-{child}.tar")]
    others = [_uri("website-scene-workspace-slim", "9", "walkthrough-req2.tar"),
              _uri("website-scene-workspace-slim", "a", f"req11-{child}.tar")]
    for uri in [*mine, *others]:
        world["store"].put(uri, b"synthetic archive", versions=2, marker=True)
    plan = _plan(world)
    assert plan["unresolved"] == []
    found = {row["uri"] for row in plan["objects"]}
    assert set(mine) <= found and not set(others) & found
    receipt = _apply(world, plan)
    assert all(not world["store"].present(uri) for uri in mine)
    assert all(world["store"].present(uri) for uri in others)
    assert set(mine) <= {row["uri"] for row in receipt["deleted"]["objects"]}


def test_slimmed_atlas_archives_cannot_be_ruled_out_without_the_object_store(world):
    plan = _plan(world, object_store=None)
    assert {"prefix": f"{PREFIX}/website-scene-workspace-slim/sha256/",
            "reason": "website_withdrawal_deletion_object_store_unconfigured"} in plan["unresolved"]


def test_simulation_packages_cannot_be_ruled_out_without_the_object_store(world):
    plan = _plan(world, object_store=None)
    assert {"prefix": f"{PREFIX}/website-company-simulation/sha256/",
            "reason": "website_withdrawal_deletion_object_store_unconfigured"} in plan["unresolved"]


def test_scene_previews_cannot_be_ruled_out_without_the_object_store(world):
    plan = _plan(world, object_store=None)
    assert {"prefix": f"{PREFIX}/website-scene-preview/sha256/",
            "reason": "website_withdrawal_deletion_object_store_unconfigured"} in plan["unresolved"]


def test_deletion_waits_for_a_scene_preview_pass_and_removes_its_local_copy(world):
    import fcntl
    from blueprint_pipeline import website_scene_preview as scene_preview
    directory = world["capture"].joinpath(*scene_preview.DIRECTORY)
    _write(directory / scene_preview.PREVIEW_FILE, b"ply synthetic preview")
    _write(directory / scene_preview.MANIFEST_FILE, {"schema_version": scene_preview.MANIFEST_SCHEMA})
    (directory / scene_preview.LOCK_FILE).touch()
    plan = _plan(world)
    with (directory / scene_preview.LOCK_FILE).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(deletion.WithdrawalDeletionError, match="scene_preview_busy"):
            _apply(world, plan)
    assert world["store"].deleted == [] and (directory / scene_preview.PREVIEW_FILE).exists()
    _apply(world, plan)
    assert not (directory / scene_preview.PREVIEW_FILE).exists()
    assert not (directory / scene_preview.MANIFEST_FILE).exists()


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


def test_deletion_waits_for_a_review_video_retention_pass_and_then_deletes_its_videos(world):
    import fcntl
    from blueprint_pipeline.screening_retained_ledger import REVIEW_VIDEOS_DIRNAME, REVIEW_VIDEOS_LOCK_NAME
    base = world["capture"] / "pipeline" / REVIEW_VIDEOS_DIRNAME
    (base / "screen-0123/videos").mkdir(parents=True)
    (base / REVIEW_VIDEOS_LOCK_NAME).touch()
    (base / "screen-0123/videos/00-run-external.mp4").write_bytes(b"synthetic render")
    plan = _plan(world)
    with (base / REVIEW_VIDEOS_LOCK_NAME).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(deletion.WithdrawalDeletionError, match="review_videos_busy"):
            _apply(world, plan)
    assert world["store"].deleted == []
    _apply(world, plan)
    assert not (base / "screen-0123/videos/00-run-external.mp4").exists() and not (base / REVIEW_VIDEOS_LOCK_NAME).exists()


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


def test_the_withdrawal_command_points_delete_at_its_own_command(world, monkeypatch, capsys):
    # The intake's withdrawal route imports website_capture_withdrawal; deletion stays out of it.
    monkeypatch.setattr(sys, "argv", ["website_capture_withdrawal", "delete", "--capture-root", str(world["capture"])])
    with pytest.raises(SystemExit):
        withdrawal.main()
    assert "python -m blueprint_pipeline.website_capture_withdrawal_deletion" in capsys.readouterr().err


def test_cli_delete_is_a_dry_run_unless_applied_with_authorization(world, tmp_path, monkeypatch, capsys):
    roots = []
    for root in world["resolution"]["run_roots"]:
        roots += ["--run-root", str(root)]
    argv = ["website_capture_withdrawal_deletion", "--capture-root", str(world["capture"]), *roots,
            "--intent-root", str(world["resolution"]["intent_root"]),
            "--binding-root", str(world["resolution"]["binding_root"]),
            "--screening-queue-root", str(world["resolution"]["screening_queue_root"]),
            "--screening-work-root", str(world["resolution"]["screening_work_root"]),
            "--native-g1-registration-root", str(world["resolution"]["native_g1_registration_root"]),
            "--team-scene-halves-root", str(world["resolution"]["team_scene_halves_root"]),
            "--team-registry", str(world["resolution"]["team_registry"]),
            "--team-scene-packet-queue-root", str(world["resolution"]["team_scene_packet_queue_root"]),
            "--rigid-probe-queue-root", str(world["resolution"]["rigid_probe_queue_root"]),
            "--rigid-probe-work-root", str(world["resolution"]["rigid_probe_work_root"]),
            "--unclaimed-handoff-root", str(world["resolution"]["unclaimed_handoff_root"])]
    before = _snapshot(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    deletion.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "dry_run" and printed["plan"]["schema_version"] == deletion.PLAN_SCHEMA
    # Without --object-store the B2 copies cannot be deleted and the offloaded run cannot be read.
    assert {row["reason"] for row in printed["plan"]["unresolved"]} == {
        "website_withdrawal_deletion_object_store_unconfigured",
        "website_withdrawal_deletion_offloaded_run_binding_undetermined"}
    assert _snapshot(tmp_path) == before
    monkeypatch.setattr(sys, "argv", [*argv, "--apply", "--expected-plan-digest", printed["plan"]["digest"]])
    with pytest.raises(deletion.WithdrawalDeletionError, match="explicit_authorization_required"):
        deletion.main()
    assert _snapshot(tmp_path) == before


def test_a_withdrawal_still_deletes_a_scene_run_the_clean_up_keeps_for_an_open_job(world, tmp_path, monkeypatch):
    """Stage 8 keeps an open website job's scene run on the host (``open_job_source_runs``). Privacy wins: the
    withdrawal deletion never reads that reason, and the kept run goes at once with the site's other copies."""
    import time

    from blueprint_pipeline import completed_replay_cache_retention as retention
    from blueprint_pipeline.control_plane_storage_gc_reasons import (
        PROTECTED_OPEN_JOB_SOURCE,
        evidence_protection_reason,
    )

    run = world["construction"]
    intent = _sealed({"intent_id": world["intent_id"],
                      "request": {"execution": {"expires_at_epoch": time.time() + 86_400}}}, "intent_digest")
    intent_root = tmp_path / "gc-scene-intents"
    _write(intent_root / world["intent_id"] / "intent.json", intent)
    packets = tmp_path / "gc-team-scene-packets"
    _write(packets / "processed" / "request.json", {"intent_id": world["intent_id"],
                                                    "intent_digest": intent["intent_digest"],
                                                    "source_launch_root": str(run)})
    monkeypatch.setattr(retention, "process_reference", lambda _root, **_kwargs: None)
    from types import SimpleNamespace

    from blueprint_pipeline import open_job_source_runs as open_jobs

    # The work disk has room, so the open job's run is kept (under the floor it may be moved, oldest first).
    monkeypatch.setattr(open_jobs, "measure_disk", lambda _path: SimpleNamespace(total=100 * 1024**3,
                                                                                 free=90 * 1024**3, used=0))
    # The job is open (its window runs): the clean-up keeps the run.
    assert evidence_protection_reason(
        run, settlement_roots=(), pins_root=tmp_path / "pins", queue_roots=[], now=time.time,
        team_scene_packet_queue_root=packets,
        scene_intent_root=intent_root) == PROTECTED_OPEN_JOB_SOURCE
    gcs = FakeGcs()
    plan = _plan(world, cloud_storage=gcs)
    assert run.name in {row["name"] for row in plan["runs"]}
    _apply(world, plan, cloud_storage=gcs)
    assert not run.exists()


def test_a_withdrawal_also_deletes_a_scene_runs_restore_record(world, tmp_path):
    """Review 8c (7): ``<run>.restore.v1.json`` (attempt counts and times) goes with the site's run."""
    from blueprint_pipeline import scene_run_restore as restore

    run = world["construction"]
    restore._write_state(run, {"attempts": 1, "status": "refused", "error_type": "ValueError"})
    record = run.parent / (run.name + restore.STATE_SUFFIX)
    assert json.loads(record.read_text())["schema_version"] == restore.STATE_SCHEMA
    gcs = FakeGcs()
    plan = _plan(world, cloud_storage=gcs)
    _apply(world, plan, cloud_storage=gcs)
    assert not run.exists() and not record.exists()


@pytest.mark.parametrize("content", ['{"attempts": 1, "status": "ref', "", "\x00\xff not json"])
def test_a_torn_or_unreadable_restore_record_and_the_restore_lock_never_stop_a_withdrawal(world, content):
    """Review 8d (2), privacy: the run's own restore leftovers (record, lock, temporary) go with it, unread."""
    from blueprint_pipeline import scene_run_restore as restore

    run = world["construction"]
    record = run.parent / (run.name + restore.STATE_SUFFIX)
    record.write_bytes(content.encode("utf-8", "surrogateescape"))  # what a crash in a plain write leaves
    lock = run.parent / f".{run.name}.restore.lock"
    lock.write_bytes(b"")
    temporary = run.parent / f".{run.name}.restore-state.tmp"
    temporary.write_text('{"half', encoding="utf-8")
    gcs = FakeGcs()
    plan = _plan(world, cloud_storage=gcs)
    assert not plan["unresolved"]
    _apply(world, plan, cloud_storage=gcs)
    assert not any(path.exists() for path in (run, record, lock, temporary))


def test_the_restore_record_is_written_whole_or_not_at_all(tmp_path, monkeypatch):
    """A crash mid-write leaves the previous record, never a torn one, and no temporary beside it."""
    from blueprint_pipeline import scene_run_restore as restore

    run = tmp_path / "launch-1"
    restore._write_state(run, {"attempts": 1, "status": "refused"})
    record = run.parent / (run.name + restore.STATE_SUFFIX)
    before = record.read_bytes()

    def crash(*_args, **_kwargs):
        raise OSError("disk went away mid-write")

    monkeypatch.setattr(restore.os, "fsync", crash)
    restore._write_state(run, {"attempts": 2, "status": "refused"})
    assert record.read_bytes() == before and json.loads(before)["attempts"] == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == [record.name]


@pytest.mark.parametrize("name", ["{run}.restore.v1.json", ".{run}.restore.lock", ".{run}.restore-state.tmp"])
def test_a_symlinked_restore_leftover_is_removed_itself_and_its_target_outside_the_site_is_untouched(
        world, tmp_path, name):
    """Review 8e (1), privacy: the link goes (never followed), the run goes, the link's target stays."""
    import os

    run = world["construction"]
    outside = tmp_path / "not-this-site"
    outside.mkdir()
    target = outside / "keep.json"
    target.write_text('{"another": "owner"}', encoding="utf-8")
    before = (target.read_bytes(), os.stat(target).st_mtime_ns)
    link = run.parent / name.format(run=run.name)
    link.symlink_to(target)
    gcs = FakeGcs()
    plan = _plan(world, cloud_storage=gcs)
    assert not plan["unresolved"]
    _apply(world, plan, cloud_storage=gcs)
    assert not run.exists() and not os.path.lexists(link)
    assert (target.read_bytes(), os.stat(target).st_mtime_ns) == before and sorted(outside.iterdir()) == [target]


def test_any_other_symlinked_leftover_still_refuses_the_withdrawal(world, tmp_path):
    run = world["construction"]
    target = tmp_path / "elsewhere.txt"
    target.write_text("x", encoding="utf-8")
    (run.parent / f".{run.name}.staging").symlink_to(target)
    plan = _plan(world, cloud_storage=FakeGcs())
    assert any(row["reason"] == "website_withdrawal_deletion_unsafe_entry" or "unsafe_entry" in row["reason"]
               for row in plan["unresolved"])
    assert target.exists()


def test_a_withdrawal_lists_and_deletes_a_killed_restores_staging_with_the_run(world):
    """PR review, privacy: a restore killed mid-way leaves ``.<run>.restoring`` (its partial download and
    extraction inside); the withdrawal's plan lists it as the run's leftover and the apply removes it."""
    from blueprint_pipeline import scene_run_restore as restore

    run = world["construction"]
    staging = restore.staging_path(run)
    partial = staging / run.name / ".restore-x1y2z3" / "evidence.tar"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"part of the site's archive")
    (staging / run.name / ".restore-x1y2z3" / "tree" / "allocator").mkdir(parents=True)
    (staging / run.name / ".restore-x1y2z3" / "tree" / "launch_receipt.json").write_text("{}")
    gcs = FakeGcs()
    plan = _plan(world, cloud_storage=gcs)
    assert not plan["unresolved"]
    [row] = [row for row in plan["runs"] if row["name"] == run.name]
    assert str(staging) in row["leftovers"]
    _apply(world, plan, cloud_storage=gcs)
    assert not run.exists() and not staging.exists()


def test_a_withdrawal_during_a_slimming_upload_leaves_no_slim_archive_once_deletion_completes(world, monkeypatch):
    # Codex P1 on #3651: a withdrawal that lands while slimming uploads a capture's Atlas files must not list the
    # site's archives before that one is visible and then report deletion complete. Slimming holds the Atlas
    # preview lock this deletion holds, across its withdrawal check and the upload; the deletion that meets it
    # refuses (atlas_preview_busy), and the one after it finds the archive by name and deletes it.
    import hashlib
    from blueprint_pipeline import website_scene_workspace_retention as retention
    from blueprint_pipeline import website_scene_workspace_slimming as slimming
    from blueprint_pipeline.task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX

    capture, store, site = world["capture"], world["store"], world["site"]
    _write(capture / "pipeline/atlas/preview.lock", b"")
    _write(capture / "pipeline/atlas/pose/frame.png", b"f" * (1 << 20))
    _write(capture / ".pipeline_job_ledger.json.lock", b"")
    tombstone = site / "website_withdrawal" / "tombstone.json"
    acknowledged = tombstone.read_bytes()
    tombstone.unlink()  # the site has not withdrawn yet when slimming starts

    def finished_job(*, scene, **_kwargs):  # the job's retention checks pass (covered by the slimming tests)
        walk = retention._walk(scene)
        return retention._Evaluation([], walk.files, [], ["walkthrough-req1"], 0.0, walk.allocated_bytes)

    monkeypatch.setattr(retention, "_evaluate", finished_job)
    context = retention.RetentionContext(storage_root=site.parents[2], pins_root=site.parents[3] / "pins",
                                         queue_roots=(), intent_root=None, binding_root=None)
    plan = slimming.plan_scene_workspace_slim(context=context, bucket="source-bucket", scene_id="site-req1",
                                              now=time.time(), process_checker=lambda _path: False)
    assert [row["relative_path"] for row in plan["captures"][0]["members"]] == ["pipeline/atlas/pose/frame.png"]
    met: list[str] = []
    uploaded: list[str] = []

    def upload_while_the_site_withdraws(*, write_stream, digest, size_bytes, filename, artifact_kind):
        data = io.BytesIO()
        write_stream(data)
        # Mid-upload: the site withdraws and its deletion runs. It must not get past the fence.
        _write(tombstone, acknowledged)
        with pytest.raises(deletion.WithdrawalDeletionError) as refused:
            _apply(world, _plan(world))
        met.append(str(refused.value))
        uri = (f"s3://{BUCKET}/{LARGE_ARTIFACT_KEY_PREFIX}/{artifact_kind}/sha256/"
               f"{hashlib.sha256(data.getvalue()).hexdigest()}/{filename}")
        store.put(uri, data.getvalue())  # the upload completes and is visible
        uploaded.append(uri)
        return {"uri": uri, "digest": digest, "size_bytes": size_bytes,
                "full_byte_service_account_readback_passed": True}

    result = slimming.apply_scene_workspace_slim(plan, context=context, ack=slimming.SLIM_ACK, now=time.time(),
                                                 stream_publisher=upload_while_the_site_withdraws,
                                                 process_checker=lambda _path: False)

    assert met == ["website_withdrawal_deletion_atlas_preview_busy"] and store.present(uploaded[0])
    # The withdrawal seen under the listener's locks: nothing local is removed, and the archive is reported.
    (outcome,) = result["captures"]
    assert outcome["status"] == "skipped" and outcome["published_archive"]["uri"] == uploaded[0]
    assert (capture / "pipeline/atlas/pose/frame.png").is_file()
    # Once slimming is done the deletion runs: its listing finds the archive by name, and nothing survives it.
    plan = _plan(world)
    assert uploaded[0] in {row["uri"] for row in plan["objects"]}
    receipt = _apply(world, plan)
    assert not store.present(uploaded[0])
    assert uploaded[0] in {row["uri"] for row in receipt["deleted"]["objects"]}


# --- The host's shape on 2026-11-27: offloaded runs that name no site and were offloaded before the site came ---
#
# The production pass never reads archives. On the host, 144 readable pointer-only runs name no site in their
# sealed pointers, and every one was offloaded hours before the withdrawn site's first record on the host. Each
# one used to be an unresolved row, so the plan could never apply. The rule: such a run is not the site's when
# its sealed evidence-offload pointer says it left the host more than PREDATES_SITE_MARGIN_SECONDS before the
# site's earliest record there. Anything missing, unsealed, odd or too close keeps it undetermined.

HOST_REQUEST = "capture-0f0e0d0c-0b0a-4908-8706-050403020100"
MARGIN = 6 * 3600
UNDETERMINED = "website_withdrawal_deletion_offloaded_run_binding_undetermined"
PREDATES = "website_withdrawal_deletion_offloaded_run_predates_site"


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def _ledger(capture: Path, started_at: float, history: tuple[float, ...] = ()) -> Path:
    """The handoff listener's job ledger, as it writes it at its first lease claim and carries it forward."""

    return drill._write(capture / "pipeline_job_ledger.json", {
        "schema_version": "pipeline_job_ledger.v1", "revision": 7, "status": "completed",
        "started_at": _iso(started_at),
        "attempt_history": [{"attempt_number": index + 1, "status": "failed_retryable", "started_at": _iso(at)}
                            for index, at in enumerate(history)]})


def _older_pointer(world: dict, root: Path, name: str, offloaded_at: Any, digit: str, *,
                   suffix: str = ".offloaded.v1.json", drop: tuple[str, ...] = (), **fields: Any) -> Path:
    """A sealed pointer-only run that names no site, as the offload writes it; its archive object is on B2."""

    uri = drill._uri("control-plane-evidence", digit, "evidence.tar")
    world["store"].put(uri, b"an older run's archive")
    if suffix == ".residue.v1.json":
        value = {"schema_version": "control_plane_result_residue_pointer.v1", "run": name, "uri": uri}
    else:
        value = {"schema_version": "control_plane_evidence_offload_pointer.v1", "status": "offloaded",
                 "directory": name, "terminal_receipt": None, "uri": uri, "digest": "sha256:" + digit * 64,
                 "size_bytes": 22, "member_count": 0, "members": [], "retained_receipt_bytes": {},
                 "evidence_deleted": False}
    value = {**value, "offloaded_at_epoch": offloaded_at, **fields}
    for key in drop:
        value.pop(key, None)
    world.setdefault("older_uris", {})[name] = uri
    return drill._write(root / f"{name}{suffix}", drill._sealed(value, "pointer_digest"))


def _host_shaped(tmp_path: Path) -> dict:
    """The withdrawn site with its own runs, another site's run, and three older runs that name no site."""

    world = drill.build_world(tmp_path, HOST_REQUEST)
    withdrawal.acknowledge_withdrawal(capture_root=world["capture"], command={
        "schema_version": "website_capture_withdrawal.v1", "request_id": HOST_REQUEST,
        "scene_id": f"site-{HOST_REQUEST}", "capture_id": f"walkthrough-{HOST_REQUEST}",
        "withdrawal_id": f"withdrawal-{HOST_REQUEST}", "requested_at_iso": "2026-11-25T02:25:31Z"})
    # The site's first record on the host: the listener's first lease claim, 10 days ago (the drill's own
    # unrelated run left 30 days ago). Every file under the site was written now, so the ledger is the earliest.
    first = float(int(time.time()) - 10 * 86400)  # whole seconds, so the ledger's text holds it exactly
    _ledger(world["capture"], first)
    launches, canaries = world["other_run"].parent, world["site_runs"][1].parent
    # Offloaded 14 hours before the site's first record, as the host's newest no-site pointer was.
    older = {"launch-website-legacy-trial-0101": launches, "launch-website-legacy-trial-0102": launches,
             "team-scored-run-legacy-0103": canaries}
    for (name, root), digit in zip(older.items(), "124"):
        _older_pointer(world, root, name, first - 14 * 3600, digit)
    drill_older = world.get("unrelated_older_pointer")
    world.update(first=first, launches=launches, canaries=canaries, older_added=sorted(older),
                 older=sorted([*older, *([drill_older["name"]] if drill_older else [])]),
                 resolution={**world["resolution"], "object_store": deletion.B2ObjectVersions(world["store"],
                                                                                              drill.BUCKET),
                             "cloud_storage": world["gcs"]})
    return world


def _host_plan(world: dict, **overrides: Any) -> dict:
    # The production resolution: no scan_offloaded_archives.
    return deletion.plan_withdrawal_deletion(capture_root=world["capture"], **{**world["resolution"], **overrides})


def _undetermined(plan: dict) -> list[str]:
    return sorted(Path(row["path"]).name for row in plan["unresolved"] if row["reason"] == UNDETERMINED)


def _cleared(plan: dict) -> list[str]:
    return sorted(Path(row["path"]).name for row in plan["offloaded_runs_older_than_site"]["runs"])


def test_offloaded_runs_older_than_the_site_that_name_no_site_do_not_block_its_withdrawal(tmp_path):
    world = _host_shaped(tmp_path)
    plan = _host_plan(world)
    assert plan["unresolved"] == [], plan["unresolved"]
    section = plan["offloaded_runs_older_than_site"]
    assert _cleared(plan) == world["older"]
    assert section["margin_seconds"] == MARGIN
    assert section["site_earliest_record"]["epoch"] == pytest.approx(world["first"], abs=1e-3)
    assert section["site_earliest_record"]["unavailable_reason"] is None
    for row in section["runs"]:
        # Each cleared run says why, with the two times compared.
        assert row["reason"] == PREDATES
        assert row["site_earliest_record_epoch"] == section["site_earliest_record"]["epoch"]
        assert row["offloaded_at_epoch"] + MARGIN < row["site_earliest_record_epoch"]
    # The site's own runs are still planned for deletion, and nothing else is.
    request = HOST_REQUEST
    assert {run["name"] for run in plan["runs"]} == {
        f"launch-{request}-construction-0001", "team-scored-run-drill-0001", f"launch-{request}-controls-0002",
        "scene-" + __import__("hashlib").sha256(request.encode()).hexdigest()}
    assert not set(world["older_uris"].values()) & {row["uri"] for row in plan["objects"]}


def test_the_rule_adds_nothing_to_what_is_deleted(tmp_path):
    world = _host_shaped(tmp_path)
    plan = _host_plan(world)
    pointers = sorted(path for root in (world["launches"], world["canaries"])
                      for path in root.glob("*.offloaded.v1.json")
                      if path.name[: -len(".offloaded.v1.json")] in world["older"])
    moved = tmp_path / "moved-away"
    moved.mkdir()
    for path in pointers:
        path.rename(moved / path.name)
    without = _host_plan(world)
    for path in pointers:
        (moved / path.name).rename(path)
    # The same runs, objects and site files with and without the older runs: the rule only empties a list.
    assert [run["name"] for run in plan["runs"]] == [run["name"] for run in without["runs"]]
    assert [row["uri"] for row in plan["objects"]] == [row["uri"] for row in without["objects"]]
    assert plan["scene_files"] == without["scene_files"]
    assert without["offloaded_runs_older_than_site"]["runs"] == []
    # Applied: every older pointer and its archive stay; the receipt names them as kept.
    receipt = deletion.apply_withdrawal_deletion(capture_root=world["capture"], expected_plan_digest=plan["digest"],
                                                 authorize_complete_deletion=True, **world["resolution"])
    assert all(path.is_file() for path in pointers)
    assert all(world["store"].versions_of(uri) == 1 for uri in world["older_uris"].values())
    assert sorted(Path(row["path"]).name for row in receipt["offloaded_runs_older_than_site"]["runs"]) \
        == world["older"]
    kept = [row for row in receipt["remaining"] if row["holder"] == "offloaded_runs_older_than_site"]
    assert len(kept) == 1 and kept[0]["count"] == len(world["older"]) and kept[0]["reason"]
    assert not any(str(path) in receipt["deleted"]["files"] for path in pointers)


def test_an_object_a_cleared_run_also_names_is_still_not_the_sites_to_delete(tmp_path):
    world = _host_shaped(tmp_path)
    # The site's team run names an archive that an older, cleared run also names.
    shared = world["older_uris"]["launch-website-legacy-trial-0101"]
    drill._write(world["site_runs"][1] / "artifacts/result_delivery/remote_artifacts/shared.json", {"uri": shared})
    plan = _host_plan(world)
    assert {"uri": shared, "reason": "website_withdrawal_deletion_object_shared_with_unbound_run"} in plan["unresolved"]
    assert shared not in {row["uri"] for row in plan["objects"]}


@pytest.mark.parametrize("fault", [
    "offloaded_after_the_site",  # the site's own run, offloaded after its data came, naming nothing locally
    "inside_the_margin", "exactly_the_margin", "missing_time", "boolean_time", "text_time", "not_a_number_time",
    "negative_time", "not_offloaded_status", "residue_pointer", "leftover_beside_it", "directory_present",
])
def test_a_run_that_could_hold_the_sites_data_stays_undetermined(tmp_path, fault):
    world = _host_shaped(tmp_path)
    first, name, root = world["first"], "launch-website-suspect-0200", world["launches"]
    kwargs: dict[str, Any] = {}
    at: Any = first - 14 * 3600
    if fault == "offloaded_after_the_site":
        at = first + 60
    elif fault == "inside_the_margin":
        at = first - MARGIN + 0.5  # a clock off by less than the margin must not clear it
    elif fault == "exactly_the_margin":
        at = first - MARGIN  # equal is the safe side
    elif fault == "missing_time":
        kwargs["drop"] = ("offloaded_at_epoch",)
    elif fault == "boolean_time":
        at = True
    elif fault == "text_time":
        at = str(first - 14 * 3600)
    elif fault == "not_a_number_time":
        at = float("nan")
    elif fault == "negative_time":
        at = -1.0
    elif fault == "not_offloaded_status":
        kwargs["status"] = "restored"
    elif fault == "residue_pointer":
        kwargs["suffix"] = ".residue.v1.json"
    _older_pointer(world, root, name, at, "7", **kwargs)
    if fault == "leftover_beside_it":
        drill._write(root / f".{name}.pointer-123.tmp", b"a pointer temporary")
    elif fault == "directory_present":
        drill._write(root / name / "launch_profile.json", {"note": "a run directory that came back"})
    plan = _host_plan(world)
    if fault == "directory_present":
        # A run directory is read, never judged by its pointer: it names nothing, so it is simply not the site's.
        assert name not in _cleared(plan) and name not in {run["name"] for run in plan["runs"]}
        return
    assert _undetermined(plan) == [name], plan["unresolved"]
    assert name not in _cleared(plan)
    assert _cleared(plan) == world["older"]  # the others still clear
    with pytest.raises(deletion.WithdrawalDeletionError, match="unresolved_copies"):
        deletion.apply_withdrawal_deletion(capture_root=world["capture"], expected_plan_digest=plan["digest"],
                                           authorize_complete_deletion=True, **world["resolution"])
    assert world["store"].deleted == []


def test_a_pointer_rewritten_without_its_writers_seal_is_never_cleared(tmp_path):
    world = _host_shaped(tmp_path)
    path = world["launches"] / "launch-website-legacy-trial-0101.offloaded.v1.json"
    # Back-dated after the fact: the seal no longer matches.
    drill._write(path, {**json.loads(path.read_text()), "offloaded_at_epoch": 1.0})
    plan = _host_plan(world)
    assert {"path": str(path.parent / "launch-website-legacy-trial-0101"),
            "reason": "website_withdrawal_deletion_pointer_unsealed"} in plan["unresolved"]
    assert "launch-website-legacy-trial-0101" not in _cleared(plan)


def test_a_pointer_the_service_cannot_read_stays_unresolved_whatever_its_file_time(tmp_path, monkeypatch):
    world = _host_shaped(tmp_path)
    path = world["launches"] / "launch-website-legacy-trial-0102.offloaded.v1.json"
    os.utime(path, (world["first"] - 30 * 86400, world["first"] - 30 * 86400))  # a file time is not evidence
    original = Path.read_bytes

    def read_bytes(self):
        if self == path:
            raise PermissionError(13, "Permission denied", str(self))
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    plan = _host_plan(world)
    assert {"path": str(path.parent / "launch-website-legacy-trial-0102"),
            "reason": "website_withdrawal_deletion_record_unreadable"} in plan["unresolved"]
    assert "launch-website-legacy-trial-0102" not in _cleared(plan)


@pytest.mark.parametrize("fault, reason", [
    ("no_ledger", "site_job_ledger_unreadable"),
    ("ledger_not_json", "site_job_ledger_unreadable"),
    ("ledger_wrong_schema", "site_job_ledger_unreadable"),
    ("ledger_without_started_at", "site_job_ledger_unreadable"),
    ("ledger_started_at_without_zone", "site_job_ledger_unreadable"),
    ("ledger_symlink", "site_job_ledger_unreadable"),
    ("second_capture_without_ledger", "site_job_ledger_unreadable"),
    ("retired_workspace", "site_workspace_retired"),
    ("retiring_leftover", "site_workspace_retired"),
])
def test_without_the_sites_earliest_record_nothing_is_cleared(tmp_path, fault, reason):
    world = _host_shaped(tmp_path)
    capture, site = world["capture"], world["site"]
    ledger = capture / "pipeline_job_ledger.json"
    if fault == "no_ledger":
        ledger.unlink()
    elif fault == "ledger_not_json":
        ledger.write_bytes(b"{not json")
    elif fault == "ledger_wrong_schema":
        drill._write(ledger, {**json.loads(ledger.read_text()), "schema_version": "pipeline_job_ledger.v0"})
    elif fault == "ledger_without_started_at":
        value = json.loads(ledger.read_text())
        value.pop("started_at")
        drill._write(ledger, value)
    elif fault == "ledger_started_at_without_zone":
        drill._write(ledger, {**json.loads(ledger.read_text()), "started_at": "2026-10-28T15:22:51"})
    elif fault == "ledger_symlink":
        target = tmp_path / "elsewhere.json"
        ledger.rename(target)
        ledger.symlink_to(target)
    elif fault == "second_capture_without_ledger":
        drill._write(capture.parent / f"walkthrough-{HOST_REQUEST}-supplement" / "raw/video.mp4", b"more video")
    elif fault == "retired_workspace":
        drill._write(site.parent / f"{site.name}.retired.v1.json", {"note": "a workspace retirement receipt"})
    else:
        (site.parent / f".retiring-{site.name}-1").mkdir()
    record = deletion._site_earliest_record(site, {"scene_id": site.name})
    assert record["epoch"] is None and record["unavailable_reason"] == reason
    if fault == "ledger_symlink":
        # A link inside the site stops the whole plan before anything is judged.
        with pytest.raises(ValueError, match="symlink"):
            _host_plan(world)
        return
    plan = _host_plan(world)
    section = plan["offloaded_runs_older_than_site"]
    assert section["site_earliest_record"]["epoch"] is None
    assert section["site_earliest_record"]["unavailable_reason"] == reason
    assert section["runs"] == []
    assert set(world["older"]) <= set(_undetermined(plan))


@pytest.mark.parametrize("earlier", ["a_file_under_the_site", "an_earlier_attempt_in_the_ledger"])
def test_any_earlier_trace_of_the_site_moves_the_comparison_earlier(tmp_path, earlier):
    world = _host_shaped(tmp_path)
    old = world["first"] - 20 * 3600  # before the older runs left the host
    if earlier == "a_file_under_the_site":
        os.utime(world["capture"] / "raw/video.mp4", (old, old))
    else:
        _ledger(world["capture"], world["first"], history=(old,))
    plan = _host_plan(world)
    assert plan["offloaded_runs_older_than_site"]["site_earliest_record"]["epoch"] == pytest.approx(old, abs=1e-3)
    # The runs that left 14 hours before the ledger's start now left after that trace: undetermined again. Only
    # the drill's run, offloaded 30 days ago, is still older than everything.
    assert not set(world["older_added"]) & set(_cleared(plan))
    assert set(world["older_added"]) <= set(_undetermined(plan))


@pytest.mark.parametrize("journal", ["as_sealed", "moved_later_without_its_seal", "of_another_tombstone"])
def test_after_the_apply_the_journal_keeps_the_sites_earliest_record_and_only_its_seal_is_trusted(tmp_path, journal):
    world = _host_shaped(tmp_path)
    plan = _host_plan(world)
    path = world["site"] / "website_withdrawal" / deletion.SITE_RECORD_FILENAME
    kept = json.loads(path.read_text())
    assert kept["epoch"] == pytest.approx(world["first"], abs=1e-3)
    deletion.apply_withdrawal_deletion(capture_root=world["capture"], expected_plan_digest=plan["digest"],
                                       authorize_complete_deletion=True, **world["resolution"])
    # The apply removed the site's files, its job ledger with them; the journal stays.
    assert not (world["capture"] / "pipeline_job_ledger.json").exists() and path.is_file()
    if journal == "moved_later_without_its_seal":
        path.write_text(json.dumps({**kept, "epoch": world["first"] + 400 * 86400}))
    elif journal == "of_another_tombstone":
        path.write_text(json.dumps(drill._sealed({**kept, "tombstone_digest": "sha256:" + "0" * 64}, "digest")))
    again = _host_plan(world)
    if journal == "as_sealed":
        assert again["unresolved"] == [] and _cleared(again) == world["older"]
        assert again["offloaded_runs_older_than_site"]["site_earliest_record"]["journal_epoch"] == kept["epoch"]
    else:
        assert again["offloaded_runs_older_than_site"]["site_earliest_record"]["epoch"] is None
        assert _cleared(again) == [] and set(world["older"]) <= set(_undetermined(again))
