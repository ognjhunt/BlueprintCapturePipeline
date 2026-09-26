"""A finished website scene workspace is retired only when cloud storage can restore it."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_scene_workspace_retention.py

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import google_crc32c
import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import website_scene_workspace_retention as retention
from blueprint_pipeline.control_plane_storage_pins import write_storage_pin
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests.test_pubsub_handoff_listener import FakeBlob, FakeStorageClient, SUBSCRIPTION


BUCKET = "capture-bucket"
SCENE = "scene-1"
CAPTURE = "capture-1"
HOUR = 3600
DAY = 24 * HOUR


def _b64(digest: bytes) -> str:
    return base64.b64encode(digest).decode("ascii")


class FakeCloud:
    """Firebase Storage as the retention sees it: a listing with GCS metadata, and downloads."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, retention.CloudObject]] = {}
        self.listings: list[tuple[str, str]] = []

    def put(self, name: str, data: bytes, *, bucket: str = BUCKET, generation: str = "1",
            md5: bool = True, crc32c: bool = True) -> None:
        self.objects[(bucket, name)] = (data, retention.CloudObject(
            name=name, size=len(data), generation=generation,
            md5_hash=_b64(hashlib.md5(data).digest()) if md5 else None,
            crc32c=_b64(google_crc32c.value(data).to_bytes(4, "big")) if crc32c else None,
        ))

    def list_objects(self, bucket: str, prefix: str) -> dict[str, retention.CloudObject]:
        self.listings.append((bucket, prefix))
        return {name: meta for (b, name), (_, meta) in self.objects.items()
                if b == bucket and name.startswith(prefix)}

    def download(self, bucket: str, name: str, destination: Path) -> None:
        destination.write_bytes(self.objects[(bucket, name)][0])


def _payload(capture: str) -> bytes:
    return json.dumps({
        "bucket": BUCKET, "scene_id": SCENE, "capture_id": capture,
        "raw_prefix_uri": f"gs://{BUCKET}/scenes/{SCENE}/captures/{capture}/raw",
    }).encode("utf-8")


def _bundle(capture: str) -> list[FakeBlob]:
    prefix = f"scenes/{SCENE}/captures/{capture}"
    manifest = {"scene_id": SCENE, "capture_id": capture, "capture_source": "browser_self_capture",
                "site_submission_id": "request-1"}
    return [
        FakeBlob(f"{prefix}/raw/manifest.json", json.dumps(manifest).encode("utf-8")),
        FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
        FakeBlob(f"{prefix}/raw/walkthrough.mov", b"raw walkthrough video bytes"),
    ]


def _authority_ended(**_kwargs):
    try:
        raise ValueError("website_control_scene-sponsorship_http_409:consent_expired")
    except ValueError as exc:
        raise listener.PipelineError("website scene failed") from exc


def _stage(storage: Path, cloud: FakeCloud, capture: str, *, status: str = "completed", ack: bool = True) -> Path:
    """Stage one capture through the real listener, exactly as production leaves it."""

    blobs = _bundle(capture)
    for blob in blobs:
        cloud.put(blob.name, blob._data)
    run_e2e = _authority_ended if status == "terminal_authority_ended" else (lambda **_: {"status": "completed"})
    listener.process_handoff_payload(_payload(capture), storage_root=storage, provider="openai",
                                     storage_client=FakeStorageClient(blobs), run_e2e=run_e2e,
                                     stage_control_plane=True, run_e2e_enabled=False)
    capture_root = storage / BUCKET / "scenes" / SCENE / "captures" / capture
    if ack:
        assert listener._write_ack_receipt(
            capture_root=capture_root, subscription=SUBSCRIPTION, message_id=f"msg-{capture}",
            payload_digest=listener.payload_sha256(_payload(capture)), delivery_attempt=1,
            disposition="terminal_authority_ended" if status == "terminal_authority_ended" else "terminal_success")
    # A local-only pipeline output: the website preparation writes these and uploads nothing.
    (capture_root / "pipeline").mkdir(exist_ok=True)
    (capture_root / "pipeline" / "preparation.json").write_text(json.dumps({"stage": "prepared"}), encoding="utf-8")
    return capture_root


def _scene(tmp_path: Path, *, status: str = "completed", ack: bool = True,
           captures: tuple[str, ...] = (CAPTURE,)) -> tuple[Path, FakeCloud]:
    cloud = FakeCloud()
    for capture in captures:
        _stage(tmp_path / "pubsub-handoffs", cloud, capture, status=status, ack=ack)
    return tmp_path / "pubsub-handoffs" / BUCKET / "scenes" / SCENE, cloud


def _context(tmp_path: Path, **overrides) -> retention.RetentionContext:
    for name in ("pins", "queue/pending", "queue/processing", "intents", "bindings"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    context = retention.RetentionContext(
        storage_root=tmp_path / "pubsub-handoffs", pins_root=tmp_path / "pins",
        queue_roots=(tmp_path / "queue",), intent_root=tmp_path / "intents",
        binding_root=tmp_path / "bindings")
    return replace(context, **overrides)


def _idle(_path: Path) -> bool:
    return False


def _plan(tmp_path: Path, cloud: FakeCloud, *, age: float = 72 * HOUR, context=None, **kwargs) -> dict:
    return retention.plan_scene_workspace_retirement(
        context=context or _context(tmp_path), bucket=BUCKET, scene_id=SCENE,
        now=time.time() + age, cloud=cloud, process_checker=kwargs.pop("process_checker", _idle), **kwargs)


# --- the scene intent and website source registration tree --------------------------------------

REQUEST = {
    "schema_version": "task_evaluation_scene_intake_request.v1", "submission_id": "submission-1",
    "owner": {"user_id": "user-1", "organization_id": "org-1"},
    "source": {"kind": "capture_bundle", "binding_id": "website-scene-1", "content_digest": "sha256:" + "c" * 64},
    "task": {"strategy": "pick_and_place", "task_id": "task-1"},
    "execution": {"expires_at_epoch": 0},
    "consent": {"accepted_by": "user-1"},
}


def _request(expires_at: float) -> dict:
    """The owner's intake request; its execution window is part of it."""

    return {**REQUEST, "execution": {"expires_at_epoch": expires_at}}


def _register(tmp_path: Path, scene: Path, request: dict) -> Path:
    """What website_scene_dispatch.register_website_preparation writes."""

    capture_root = scene / "captures" / CAPTURE
    references = {}
    for role, relative in (("preparation", "pipeline/preparation.json"),
                           ("runtime_inputs", "pipeline/preparation.json"),
                           ("task_context", "pipeline_handoff.json")):
        path = capture_root / relative
        references[role] = {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                            "size_bytes": path.stat().st_size}
    value = {"schema_version": "website_scene_source_registration.v1",
             "request_digest": cross_runtime_canonical_digest(request), "references": references,
             "provider_mutation_performed": False, "execution_authority_granted": False,
             "claim_ceiling": "development_only"}
    value["registration_digest"] = canonical_digest(value, digest_field="registration_digest")
    path = tmp_path / "bindings" / (value["request_digest"][7:] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _intent(tmp_path: Path, request: dict, *, finished: str | None = None) -> str:
    intent_id = "scene-" + cross_runtime_canonical_digest({"owner": request["owner"], "sub": request["submission_id"]})[7:]
    directory = tmp_path / "intents" / intent_id
    directory.mkdir(parents=True)
    intent = intake._seal({"schema_version": "task_evaluation_scene_intent.v1", "intent_id": intent_id,
                           "request": request, "authenticated_issuer": "webapp", "accepted_at_epoch": 1.0,
                           "source_content_digest": request["source"]["content_digest"],
                           "task_content_digest": cross_runtime_canonical_digest(request["task"]),
                           "provider_mutation_performed": False}, "intent_digest")
    (directory / "intent.json").write_text(json.dumps(intent), encoding="utf-8")
    if finished == "completed":
        projection = intake._seal({"schema_version": "task_evaluation_scene_progression.v1",
                                   "intent_id": intent_id, "intent_digest": intent["intent_digest"],
                                   "event_sequence": 1, "last_event_digest": "sha256:" + "e" * 64,
                                   "status": "completed", "phase": "terminal", "blockers": [],
                                   "result_reference": None, "state": {}, "updated_at_epoch": 1.0,
                                   "provider_allocation_performed": False}, "progression_digest")
        (directory / "progression.json").write_text(json.dumps(projection), encoding="utf-8")
    elif finished == "revoked":
        (directory / "revoked.json").write_text("{}", encoding="utf-8")
    return intent_id


# --- plan -----------------------------------------------------------------------------------------


def test_terminal_acknowledged_idle_scene_is_retirable(tmp_path):
    scene, cloud = _scene(tmp_path)

    plan = _plan(tmp_path, cloud)

    assert plan["schema_version"] == retention.PLAN_SCHEMA
    assert plan["status"] == "retirable" and plan["reasons"] == []
    assert (plan["bucket"], plan["scene_id"], plan["workspace"]) == (BUCKET, SCENE, str(scene))
    assert plan["captures"] == [{"capture_id": CAPTURE, "ledger_status": "completed",
                                 "acknowledgement": "ack_receipt"}]
    raw = [f"captures/{CAPTURE}/raw/{name}" for name in ("capture_upload_complete.json", "manifest.json",
                                                         "walkthrough.mov")]
    assert [row["relative_path"] for row in plan["cloud_verified"]] == raw
    video = next(row for row in plan["cloud_verified"] if row["relative_path"].endswith("walkthrough.mov"))
    data = b"raw walkthrough video bytes"
    assert video == {"relative_path": raw[2], "uri": f"gs://{BUCKET}/scenes/{SCENE}/{raw[2]}", "generation": "1",
                     "size": len(data), "md5_hash": _b64(hashlib.md5(data).digest()),
                     "crc32c": _b64(google_crc32c.value(data).to_bytes(4, "big"))}
    archived = {row["relative_path"]: row for row in plan["archive"]}
    local_only = {f"captures/{CAPTURE}/{name}" for name in (
        "pipeline/preparation.json", "pipeline_job_ledger.json", ".pipeline_job_ledger.json.lock",
        "pipeline_job_output_commit.json", "pipeline_job_ack_receipt.json", "pipeline_staging_manifest.json",
        "pipeline_handoff.json")}
    assert local_only <= set(archived)
    assert not any("/raw/" in name for name in archived), "raw capture bytes are never archived"
    preparation = scene / "captures" / CAPTURE / "pipeline" / "preparation.json"
    assert archived[f"captures/{CAPTURE}/pipeline/preparation.json"] == {
        "relative_path": f"captures/{CAPTURE}/pipeline/preparation.json",
        "size_bytes": preparation.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(preparation.read_bytes()).hexdigest()}
    assert len(plan["snapshot"]) == len(plan["cloud_verified"]) + len(plan["archive"])
    assert plan["totals"]["file_count"] == len(plan["snapshot"])
    assert plan["totals"]["cloud_verified_bytes"] == sum(row["size"] for row in plan["cloud_verified"])
    assert plan["totals"]["archive_bytes"] == sum(row["size_bytes"] for row in plan["archive"])
    assert plan["totals"]["workspace_allocated_bytes"] > 0
    assert plan["idle_seconds"] >= 71 * HOUR
    assert plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest")
    assert cloud.listings == [(BUCKET, f"scenes/{SCENE}/")]


def test_authority_ended_captures_are_terminal_too(tmp_path):
    _, cloud = _scene(tmp_path, status="terminal_authority_ended")

    plan = _plan(tmp_path, cloud)

    assert plan["status"] == "retirable", plan["reasons"]
    assert plan["captures"][0]["ledger_status"] == "terminal_authority_ended"


def test_not_retired_while_any_capture_is_not_terminal(tmp_path):
    scene, cloud = _scene(tmp_path, captures=(CAPTURE, "capture-2"))
    ledger_path = scene / "captures" / "capture-2" / "pipeline_job_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    ledger_path.write_text(json.dumps({**ledger, "status": "failed_retryable"}), encoding="utf-8")
    assert _plan(tmp_path, cloud)["reasons"] == ["capture_not_terminal:capture-2"]

    lease = time.time() + 30 * DAY
    ledger_path.write_text(json.dumps({**ledger, "status": "processing", "lease_expires_at":
                                       listener._iso_at(listener.datetime.fromtimestamp(lease, listener.timezone.utc))}),
                           encoding="utf-8")
    assert _plan(tmp_path, cloud)["reasons"] == ["capture_lease_held:capture-2", "capture_not_terminal:capture-2"]

    ledger_path.write_text("{not json", encoding="utf-8")
    plan = _plan(tmp_path, cloud)
    assert plan["status"] == "retained" and plan["reasons"] == ["capture_ledger_unreadable:capture-2"]
    assert plan["cloud_verified"] == [] and plan["archive"] == [], "no cloud inventory once a check retains"
    assert cloud.listings == []


def test_a_terminal_ledger_needs_its_own_proof(tmp_path):
    scene, cloud = _scene(tmp_path)
    (scene / "captures" / CAPTURE / "pipeline_job_output_commit.json").unlink()
    assert _plan(tmp_path, cloud)["reasons"] == [f"capture_not_terminal:{CAPTURE}"]


def test_an_authority_ending_without_its_receipt_is_not_terminal(tmp_path):
    scene, cloud = _scene(tmp_path, status="terminal_authority_ended")
    (scene / "captures" / CAPTURE / "pipeline_job_terminal_receipt.json").unlink()
    assert _plan(tmp_path, cloud)["reasons"] == [f"capture_not_terminal:{CAPTURE}"]


def test_a_scene_without_captures_is_retained(tmp_path):
    scene = tmp_path / "pubsub-handoffs" / BUCKET / "scenes" / SCENE
    (scene / "captures").mkdir(parents=True)
    assert _plan(tmp_path, FakeCloud())["reasons"] == ["scene_has_no_captures"]


def test_an_unacknowledged_capture_is_retained_until_pubsub_forgets_it(tmp_path):
    _, cloud = _scene(tmp_path, ack=False)
    assert _plan(tmp_path, cloud)["reasons"] == [f"acknowledgement_unproven:{CAPTURE}"]


def test_legacy_ledger_counts_as_acknowledged_after_pubsub_retention(tmp_path):
    _, cloud = _scene(tmp_path, ack=False)

    assert _plan(tmp_path, cloud, age=6 * DAY)["reasons"] == [f"acknowledgement_unproven:{CAPTURE}"]
    plan = _plan(tmp_path, cloud, age=8 * DAY)

    assert plan["status"] == "retirable", plan["reasons"]
    assert plan["captures"][0]["acknowledgement"] == "pubsub_retention_elapsed"


def test_an_ack_receipt_must_match_the_terminal_kind_and_follow_it(tmp_path):
    scene, cloud = _scene(tmp_path)
    ack_path = scene / "captures" / CAPTURE / "pipeline_job_ack_receipt.json"
    ack = json.loads(ack_path.read_text(encoding="utf-8"))

    ack_path.write_text(json.dumps({**ack, "disposition": "terminal_authority_ended"}), encoding="utf-8")
    assert _plan(tmp_path, cloud)["reasons"] == [f"acknowledgement_unproven:{CAPTURE}"]

    # An acknowledgement older than the terminal state acknowledged an earlier message.
    ack_path.write_text(json.dumps({**ack, "acknowledged_at": "2020-01-01T00:00:00+00:00"}), encoding="utf-8")
    assert _plan(tmp_path, cloud)["reasons"] == [f"acknowledgement_unproven:{CAPTURE}"]


def test_a_recently_touched_scene_is_retained(tmp_path):
    _, cloud = _scene(tmp_path)
    assert _plan(tmp_path, cloud, age=HOUR)["reasons"] == ["recently_active"]


def test_not_retired_while_pinned(tmp_path):
    scene, cloud = _scene(tmp_path)
    context = _context(tmp_path)
    write_storage_pin(pins_root=context.pins_root, kind="preparation", owner_id="prep-1",
                      paths=[str(scene / "captures" / CAPTURE / "pipeline")])

    assert _plan(tmp_path, cloud, context=context)["reasons"] == ["pinned"]


def test_not_retired_while_queue_references_the_scene(tmp_path):
    _, cloud = _scene(tmp_path)
    (tmp_path / "queue" / "processing").mkdir(parents=True, exist_ok=True)
    (tmp_path / "queue" / "processing" / "job.json").write_text(json.dumps({"scene_id": SCENE}), encoding="utf-8")

    assert _plan(tmp_path, cloud)["reasons"] == ["queue_referenced"]


def test_not_retired_while_a_process_uses_it(tmp_path):
    scene, cloud = _scene(tmp_path)
    seen: list[Path] = []

    def busy(path: Path) -> bool:
        seen.append(path)
        return True

    def broken(_path: Path) -> bool:
        raise ValueError("replay_cache_process_inventory_unavailable")

    assert _plan(tmp_path, cloud, process_checker=busy)["reasons"] == ["in_use"]
    assert seen == [scene]
    assert _plan(tmp_path, cloud, process_checker=broken)["reasons"] == ["in_use"]


def test_not_retired_while_an_open_intent_can_resolve_its_source(tmp_path):
    scene, cloud = _scene(tmp_path)
    now = time.time() + 72 * HOUR
    request = _request(now + HOUR)
    _register(tmp_path, scene, request)

    # Registered but not yet claimed by any intent: the website may still create one.
    assert _plan(tmp_path, cloud, age=HOUR * 60)["reasons"] == ["unclaimed_source_registration"]

    intent_id = _intent(tmp_path, request)
    plan = retention.plan_scene_workspace_retirement(
        context=_context(tmp_path), bucket=BUCKET, scene_id=SCENE, now=now, cloud=cloud, process_checker=_idle)
    assert plan["reasons"] == [f"open_scene_intent:{intent_id}"]


def test_an_old_registration_no_intent_claimed_does_not_hold_the_scene(tmp_path):
    scene, cloud = _scene(tmp_path)
    _register(tmp_path, scene, _request(time.time() + 30 * DAY))
    assert _plan(tmp_path, cloud, age=4 * DAY)["status"] == "retirable"


@pytest.mark.parametrize("finished", ["completed", "revoked", "expired"])
def test_retired_once_the_intent_is_completed_revoked_or_expired(tmp_path, finished):
    scene, cloud = _scene(tmp_path)
    now = time.time() + 60 * HOUR
    request = _request(now - 1 if finished == "expired" else now + DAY)
    _register(tmp_path, scene, request)
    _intent(tmp_path, request, finished=None if finished == "expired" else finished)

    plan = retention.plan_scene_workspace_retirement(
        context=_context(tmp_path), bucket=BUCKET, scene_id=SCENE, now=now, cloud=cloud, process_checker=_idle)

    assert plan["status"] == "retirable", plan["reasons"]


def test_a_registration_elsewhere_keeps_the_scene_it_names(tmp_path):
    """A workspace that names a registration the index did not read cannot prove no intent needs it."""

    scene, cloud = _scene(tmp_path)
    elsewhere = tmp_path / "other-bindings"
    registration = _register(tmp_path, scene, _request(time.time() + 30 * DAY))
    elsewhere.mkdir()
    moved = elsewhere / registration.name
    registration.rename(moved)
    handoff = scene / "captures" / CAPTURE / "pipeline" / "website_scene_preparation" / "handoff.json"
    handoff.parent.mkdir(parents=True)
    handoff.write_text(json.dumps({"schema_version": "website_scene_handoff.v1",
                                   "source_registration": {"path": str(moved)}}), encoding="utf-8")

    assert _plan(tmp_path, cloud, age=4 * DAY)["reasons"] == [f"source_registration_unindexed:{CAPTURE}"]
    moved.rename(registration)
    handoff.write_text(json.dumps({"schema_version": "website_scene_handoff.v1",
                                   "source_registration": {"path": str(registration)}}), encoding="utf-8")
    assert _plan(tmp_path, cloud, age=4 * DAY)["status"] == "retirable"


def test_raw_bytes_that_do_not_verify_in_the_cloud_block_retirement(tmp_path):
    _, cloud = _scene(tmp_path)
    video = f"scenes/{SCENE}/captures/{CAPTURE}/raw/walkthrough.mov"
    cloud.put(video, b"different bytes of equal size!!"[: len(b"raw walkthrough video bytes")])

    plan = _plan(tmp_path, cloud)

    assert plan["status"] == "retained"
    assert plan["reasons"] == [f"raw_not_verified_in_cloud:captures/{CAPTURE}/raw/walkthrough.mov"]
    assert not any("/raw/" in row["relative_path"] for row in plan["archive"])

    del cloud.objects[(BUCKET, video)]
    assert _plan(tmp_path, cloud)["reasons"] == [
        f"raw_not_verified_in_cloud:captures/{CAPTURE}/raw/walkthrough.mov"]


def test_crc32c_verifies_an_object_without_md5_and_nothing_verifies_without_either(tmp_path):
    _, cloud = _scene(tmp_path)
    video = f"scenes/{SCENE}/captures/{CAPTURE}/raw/walkthrough.mov"
    data = cloud.objects[(BUCKET, video)][0]

    cloud.put(video, data, md5=False)  # a composite object carries only CRC32C
    assert _plan(tmp_path, cloud)["status"] == "retirable"

    cloud.put(video, data, md5=False, crc32c=False)
    assert _plan(tmp_path, cloud)["reasons"] == [
        f"raw_not_verified_in_cloud:captures/{CAPTURE}/raw/walkthrough.mov"]


def test_unreadable_reference_index_fails_closed(tmp_path):
    scene, cloud = _scene(tmp_path)
    (tmp_path / "bindings").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bindings" / ("0" * 64 + ".json")).write_text("{not json", encoding="utf-8")
    assert _plan(tmp_path, cloud)["reasons"] == ["reference_index_unreadable"]

    (tmp_path / "bindings" / ("0" * 64 + ".json")).unlink()
    _register(tmp_path, scene, _request(time.time() + 30 * DAY))
    broken = tmp_path / "intents" / "scene-broken"
    broken.mkdir(parents=True)
    (broken / "intent.json").write_text(json.dumps({"intent_id": "scene-broken", "intent_digest": "sha256:0"}),
                                        encoding="utf-8")
    assert _plan(tmp_path, cloud, age=4 * DAY)["reasons"] == ["reference_index_unreadable"]

    # An intent root or binding root the context does not know is unreadable too.
    assert _plan(tmp_path, cloud, context=_context(tmp_path, intent_root=None))["reasons"] == [
        "reference_index_unreadable"]


def test_symlink_inside_the_workspace_blocks_retirement(tmp_path):
    scene, cloud = _scene(tmp_path)
    (scene / "captures" / CAPTURE / "pipeline" / "link").symlink_to(tmp_path)

    plan = _plan(tmp_path, cloud)

    assert plan["reasons"] == [f"unsafe_entry:captures/{CAPTURE}/pipeline/link"]


def test_a_symlinked_workspace_is_never_walked(tmp_path):
    scene, cloud = _scene(tmp_path)
    real = tmp_path / "real-scene"
    scene.rename(real)
    scene.symlink_to(real)
    assert _plan(tmp_path, cloud)["reasons"] == ["workspace_path_unsafe"]


def test_scene_workspaces_lists_every_scene_but_not_receipts(tmp_path):
    scene, _ = _scene(tmp_path)
    (scene.parent / f"{SCENE}-old{retention.RETIRED_SUFFIX}").write_text("{}", encoding="utf-8")
    (tmp_path / "pubsub-handoffs" / ".pubsub_delivery_evidence").mkdir()

    assert retention.scene_workspaces(tmp_path / "pubsub-handoffs") == [(BUCKET, SCENE, scene)]


def test_listener_file_names_have_not_drifted():
    assert retention.LISTENER_FILES == {
        "ledger": listener.JOB_LEDGER_FILENAME, "output_commit": listener.JOB_OUTPUT_COMMIT_FILENAME,
        "terminal_receipt": listener.JOB_TERMINAL_RECEIPT_FILENAME, "ack_receipt": listener.JOB_ACK_RECEIPT_FILENAME,
        "staging_manifest": listener.STAGING_MANIFEST_FILENAME,
    }
    assert retention.LISTENER_SCHEMAS == {
        "output_commit": listener.JOB_OUTPUT_COMMIT_SCHEMA_VERSION,
        "terminal_receipt": listener.JOB_TERMINAL_RECEIPT_SCHEMA_VERSION,
        "ack_receipt": listener.JOB_ACK_RECEIPT_SCHEMA_VERSION,
        "staging_manifest": listener.STAGING_MANIFEST_SCHEMA_VERSION,
    }
    assert retention.TERMINAL_AUTHORITY_STATUS == listener.TERMINAL_AUTHORITY_STATUS


def test_invalid_identities_are_refused_before_any_path_use(tmp_path):
    for bucket, scene_id in (("capture-bucket", ".."), ("capture-bucket", "a/b"), ("Bad_Bucket", SCENE)):
        with pytest.raises(retention.WebsiteSceneWorkspaceRetentionError, match="identity_invalid"):
            retention.plan_scene_workspace_retirement(
                context=_context(tmp_path), bucket=bucket, scene_id=scene_id, now=time.time(),
                cloud=FakeCloud(), process_checker=_idle)


def test_plan_does_not_touch_the_workspace(tmp_path):
    scene, cloud = _scene(tmp_path)
    before = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in scene.rglob("*")}
    _plan(tmp_path, cloud)
    assert {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in scene.rglob("*")} == before
    assert os.listdir(scene.parent) == [SCENE]
