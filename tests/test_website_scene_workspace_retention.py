"""A finished website scene workspace is retired only when cloud storage can restore it."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_scene_workspace_retention.py
#   src/blueprint_pipeline/control_plane_evidence_offload.py

from __future__ import annotations

import base64
import fcntl
import functools
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import google_crc32c
import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import website_scene_workspace_retention as retention
from blueprint_pipeline.control_plane_storage_pins import write_storage_pin
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests.test_control_plane_evidence_streaming import MultipartClient
from tests.test_pubsub_handoff_listener import FakeBlob, FakeStorageClient, SUBSCRIPTION


BUCKET = "capture-bucket"
SCENE = "scene-1"
CAPTURE = "capture-1"
HOUR = 3600
DAY = 24 * HOUR
ARTIFACT_BUCKET = "blueprint-task-evaluation-artifacts-test"


@pytest.fixture(autouse=True)
def isolated_disk_ledger(tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk

    monkeypatch.setattr(retention, "reserve_control_plane_disk", functools.partial(
        reserve_control_plane_disk, disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, free=80 * 1024**3)))
    monkeypatch.setattr(retention, "DEFAULT_RESERVATION_ROOT", tmp_path / "disk-reservations")


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


def test_a_terminal_receipt_left_by_an_earlier_payload_does_not_prove_the_ending(tmp_path):
    """A job reopened by a new payload can leave the old payload's receipt behind."""

    scene, cloud = _scene(tmp_path, status="terminal_authority_ended")
    ledger_path = scene / "captures" / CAPTURE / "pipeline_job_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    receipt = json.loads((scene / "captures" / CAPTURE / "pipeline_job_terminal_receipt.json").read_text())
    assert receipt["payload_sha256"] == ledger["terminal_payload_sha256"]

    ledger_path.write_text(json.dumps({**ledger, "terminal_payload_sha256": "b" * 64}), encoding="utf-8")

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
    # An expired intent counts as finished only after the grace period in which it may be extended.
    request = _request(now - retention.DEFAULT_EXPIRED_GRACE_SECONDS - 1 if finished == "expired" else now + DAY)
    _register(tmp_path, scene, request)
    _intent(tmp_path, request, finished=None if finished == "expired" else finished)

    plan = retention.plan_scene_workspace_retirement(
        context=_context(tmp_path), bucket=BUCKET, scene_id=SCENE, now=now, cloud=cloud, process_checker=_idle)

    assert plan["status"] == "retirable", plan["reasons"]


def _attempt(tmp_path: Path, intent_id: str, *, cancelled: bool = False) -> None:
    """A reserved attempt row, as reserve_scene_attempt seals it; optionally cancelled before it started."""

    directory = tmp_path / "intents" / intent_id
    intent = json.loads((directory / "intent.json").read_text(encoding="utf-8"))
    row = intake._seal({"schema_version": "task_evaluation_scene_attempt.v1", "intent_id": intent_id,
                        "intent_digest": intent["intent_digest"], "attempt_id": "attempt-1",
                        "source_commit": "a" * 40, "runtime_digest": "sha256:" + "d" * 64,
                        "input_digest": "sha256:" + "e" * 64, "provider": "vast", "maximum_spend_usd": 5.0,
                        "status": "reserved", "reserved_at_epoch": 1.0}, "attempt_digest")
    (directory / "attempts").mkdir(exist_ok=True)
    (directory / "attempts" / "attempt-1.json").write_text(json.dumps(row), encoding="utf-8")
    if cancelled:
        original = {"schema_version": "task_evaluation_launch_receipt.v1", "status": "blocked",
                    "source_commit": "a" * 40}
        original["receipt_digest"] = cross_runtime_canonical_digest(original, digest_field="receipt_digest")
        cancellation = {"schema_version": "task_evaluation_unstarted_controls_cancellation.v1",
                        "status": "cancelled_before_controls_eligibility", "attempt_id": "attempt-1",
                        "attempt_digest": row["attempt_digest"], "intent_digest": row["intent_digest"],
                        "maximum_spend_usd": 5.0, "provider": "vast", "original_blocked_launch_receipt": original,
                        "downstream_execution_eligible": False, "provider_mutation_performed": False}
        cancellation["receipt_digest"] = canonical_digest(cancellation, digest_field="receipt_digest")
        (directory / "cancelled-unstarted-controls").mkdir(exist_ok=True)
        (directory / "cancelled-unstarted-controls" / "attempt-1.json").write_text(json.dumps(cancellation),
                                                                                     encoding="utf-8")


def test_an_expired_intent_stays_open_while_it_could_still_be_extended(tmp_path):
    """An owner may extend an expired intent's window, and it would then resolve its source again."""

    scene, cloud = _scene(tmp_path)
    now = time.time() + 60 * HOUR
    request = _request(now - DAY)
    _register(tmp_path, scene, request)
    intent_id = _intent(tmp_path, request)

    def plan(**overrides):
        return retention.plan_scene_workspace_retirement(
            context=_context(tmp_path, **overrides), bucket=BUCKET, scene_id=SCENE, now=now, cloud=cloud,
            process_checker=_idle)

    assert plan()["reasons"] == [f"open_scene_intent:{intent_id}"]
    assert plan(expired_grace_seconds=DAY - 1)["status"] == "retirable"


@pytest.mark.parametrize("finished", ["revoked", "expired"])
def test_a_live_attempt_keeps_a_revoked_or_expired_intents_workspace(tmp_path, finished):
    scene, cloud = _scene(tmp_path)
    now = time.time() + 60 * HOUR
    request = _request(now - 8 * DAY if finished == "expired" else now + DAY)
    _register(tmp_path, scene, request)
    intent_id = _intent(tmp_path, request, finished="revoked" if finished == "revoked" else None)
    _attempt(tmp_path, intent_id)

    def plan():
        return retention.plan_scene_workspace_retirement(
            context=_context(tmp_path), bucket=BUCKET, scene_id=SCENE, now=now, cloud=cloud, process_checker=_idle)

    assert plan()["reasons"] == [f"open_scene_attempt:{intent_id}/attempt-1"]
    _attempt(tmp_path, intent_id, cancelled=True)  # progression's own terminal proof for the row
    assert plan()["status"] == "retirable"


def test_a_completed_intents_attempt_rows_do_not_hold_its_workspace(tmp_path):
    """Only retired predecessors are ever settled; a completed run's own row stays a spend hold forever.

    Progression completes an intent only after joining its attempt's terminal result, and a
    website attempt copies every workspace input into its own submission when it materializes.
    """

    scene, cloud = _scene(tmp_path)
    now = time.time() + 60 * HOUR
    request = _request(now + DAY)
    _register(tmp_path, scene, request)
    intent_id = _intent(tmp_path, request, finished="completed")
    _attempt(tmp_path, intent_id)

    plan = retention.plan_scene_workspace_retirement(
        context=_context(tmp_path), bucket=BUCKET, scene_id=SCENE, now=now, cloud=cloud, process_checker=_idle)

    assert plan["status"] == "retirable", plan["reasons"]


def test_an_attempt_that_cannot_be_read_protects_every_scene(tmp_path):
    scene, cloud = _scene(tmp_path)
    now = time.time() + 60 * HOUR
    request = _request(now - 8 * DAY)
    _register(tmp_path, scene, request)
    intent_id = _intent(tmp_path, request)
    (tmp_path / "intents" / intent_id / "attempts").mkdir()
    (tmp_path / "intents" / intent_id / "attempts" / "attempt-1.json").write_text("{}", encoding="utf-8")

    plan = retention.plan_scene_workspace_retirement(
        context=_context(tmp_path), bucket=BUCKET, scene_id=SCENE, now=now, cloud=cloud, process_checker=_idle)

    assert plan["reasons"] == ["reference_index_unreadable"]


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


def test_listener_file_names_have_not_drifted(tmp_path):
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
    # The listener builds its lock name inline; retirement must take the very same lock.
    with listener._locked_job_ledger(tmp_path / "capture"):
        pass
    assert os.listdir(tmp_path / "capture") == [retention.LEDGER_LOCK]


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


# --- apply and restore ----------------------------------------------------------------------------


def _digests(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _publisher(client: MultipartClient):
    return functools.partial(store.publish_configured_scene_stream, client=client, bucket=ARTIFACT_BUCKET)


def _retire(tmp_path: Path, cloud: FakeCloud, plan: dict, *, client: MultipartClient | None = None, **kwargs) -> dict:
    return retention.apply_scene_workspace_retirement(
        plan, context=kwargs.pop("context", None) or _context(tmp_path), ack=kwargs.pop("ack", retention.RETIRE_ACK),
        cloud=cloud, now=plan["observed_at_epoch"],
        stream_publisher=kwargs.pop("stream_publisher", None) or _publisher(client or MultipartClient()),
        process_checker=_idle, **kwargs)


def test_apply_archives_local_only_files_verifies_readback_writes_receipt_and_removes(tmp_path):
    scene, cloud = _scene(tmp_path)
    original = _digests(scene)
    plan = _plan(tmp_path, cloud)
    client = MultipartClient()

    result = _retire(tmp_path, cloud, plan, client=client)

    receipt_path = scene.parent / f"{SCENE}{retention.RETIRED_SUFFIX}"
    assert result["status"] == "retired" and result["receipt"] == str(receipt_path)
    assert result["freed_allocated_bytes"] == plan["totals"]["workspace_allocated_bytes"] > 0
    assert not scene.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == retention.RETIRED_SCHEMA
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert (receipt["bucket"], receipt["scene_id"], receipt["workspace"]) == (BUCKET, SCENE, str(scene))
    assert receipt["source_plan_digest"] == plan["plan_digest"] and receipt["evidence_deleted"] is False
    assert receipt["cloud_verified"] == plan["cloud_verified"]
    archive = receipt["archive"]
    assert [row["relative_path"] for row in archive["members"]] == [row["relative_path"] for row in plan["archive"]]
    assert archive["member_count"] == len(plan["archive"]) and result["archive_bytes"] == archive["size_bytes"]
    assert archive["uri"].startswith(f"s3://{ARTIFACT_BUCKET}/") and f"/{retention.ARTIFACT_KIND}/sha256/" in archive["uri"]
    assert archive["uri"].endswith("/workspace.tar") and not client.pending and client.upload_count == 1
    stored = client.objects[(ARTIFACT_BUCKET, archive["uri"].split("/", 3)[3])]
    assert "sha256:" + hashlib.sha256(stored).hexdigest() == archive["digest"]
    # The capture records travel as parsed JSON, so a retired capture stays idempotent.
    [capture] = receipt["captures"]
    root = f"captures/{CAPTURE}"
    assert capture["capture_id"] == CAPTURE
    assert capture["ledger"]["status"] == "completed" and capture["output_commit"]["status"] == "committed"
    assert capture["ack_receipt"]["disposition"] == "terminal_success"
    assert capture["staging_manifest"]["schema_version"] == "pipeline_handoff_staging_manifest.v1"
    assert "terminal_receipt" not in capture
    assert {row["relative_path"] for row in archive["members"]} == {
        name for name in original if not name.startswith(f"{root}/raw/")}
    assert oct(receipt_path.stat().st_mode & 0o777) == oct(0o640)
    assert receipt_path.stat().st_uid == scene.parent.stat().st_uid
    assert not list(scene.parent.glob(".*")), "no temporary is left beside the receipt"


def test_apply_deletes_nothing_when_archive_readback_fails(tmp_path):
    scene, cloud = _scene(tmp_path)
    before = _digests(scene)
    plan = _plan(tmp_path, cloud)

    def lying(**kwargs):
        reference = _publisher(MultipartClient())(**kwargs)
        return {**reference, "full_byte_service_account_readback_passed": False}

    def raising(**_kwargs):
        raise store.TaskEvaluationConfiguredSceneObjectStoreError("configured_scene_artifact_readback_mismatch")

    def wrong_digest(**kwargs):
        return {**_publisher(MultipartClient())(**kwargs), "digest": "sha256:" + "0" * 64}

    for publisher in (lying, raising, wrong_digest):
        result = _retire(tmp_path, cloud, plan, stream_publisher=publisher)
        assert result["status"] == "skipped" and result["reason"] == "archive_readback_failed"
        assert _digests(scene) == before
        assert not (scene.parent / f"{SCENE}{retention.RETIRED_SUFFIX}").exists()


def test_apply_skips_when_the_listener_holds_a_capture_lock(tmp_path):
    scene, cloud = _scene(tmp_path)
    plan = _plan(tmp_path, cloud)
    lock = os.open(scene / "captures" / CAPTURE / ".pipeline_job_ledger.json.lock", os.O_RDWR)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)  # what the listener holds while it reads or commits the ledger
        result = _retire(tmp_path, cloud, plan)
    finally:
        os.close(lock)

    assert result["status"] == "skipped" and result["reason"] == "candidate_busy"
    assert scene.is_dir() and not (scene.parent / f"{SCENE}{retention.RETIRED_SUFFIX}").exists()
    assert _retire(tmp_path, cloud, plan)["status"] == "retired"


@pytest.mark.parametrize("change", ["rewritten", "added", "pinned"])
def test_apply_skips_when_the_workspace_changed_since_the_plan(tmp_path, change):
    scene, cloud = _scene(tmp_path)
    plan = _plan(tmp_path, cloud)
    preparation = scene / "captures" / CAPTURE / "pipeline" / "preparation.json"
    if change == "rewritten":
        preparation.write_text(json.dumps({"stage": "rewritten"}), encoding="utf-8")
    elif change == "added":
        (preparation.parent / "late.json").write_text("{}", encoding="utf-8")
    else:
        write_storage_pin(pins_root=tmp_path / "pins", kind="compilation", owner_id="c-1", paths=[str(scene)])

    result = _retire(tmp_path, cloud, plan)

    assert result["status"] == "skipped" and result["reason"] == "candidate_changed"
    assert scene.is_dir() and not (scene.parent / f"{SCENE}{retention.RETIRED_SUFFIX}").exists()


def test_apply_skips_when_cloud_objects_changed(tmp_path):
    scene, cloud = _scene(tmp_path)
    plan = _plan(tmp_path, cloud)
    video = f"scenes/{SCENE}/captures/{CAPTURE}/raw/walkthrough.mov"
    cloud.put(video, cloud.objects[(BUCKET, video)][0], generation="2")  # overwritten, even with equal bytes

    result = _retire(tmp_path, cloud, plan)

    assert result["status"] == "skipped" and result["reason"] == "cloud_changed"
    assert scene.is_dir() and not (scene.parent / f"{SCENE}{retention.RETIRED_SUFFIX}").exists()


def test_apply_requires_the_ack_and_an_unaltered_retirable_plan(tmp_path):
    scene, cloud = _scene(tmp_path)
    plan = _plan(tmp_path, cloud)
    retained = _plan(tmp_path, cloud, age=HOUR)
    elsewhere = {**plan, "workspace": str(tmp_path / "elsewhere")}
    elsewhere["plan_digest"] = canonical_digest(elsewhere, digest_field="plan_digest")

    for candidate, ack in ((plan, "yes"), ({**plan, "archive": []}, retention.RETIRE_ACK),
                           (retained, retention.RETIRE_ACK), (elsewhere, retention.RETIRE_ACK)):
        with pytest.raises(retention.WebsiteSceneWorkspaceRetentionError, match="apply_not_authorized"):
            _retire(tmp_path, cloud, candidate, ack=ack)
    assert scene.is_dir()


def test_an_existing_receipt_is_never_replaced(tmp_path):
    scene, cloud = _scene(tmp_path)
    plan = _plan(tmp_path, cloud)
    receipt = scene.parent / f"{SCENE}{retention.RETIRED_SUFFIX}"
    receipt.write_text("{}", encoding="utf-8")

    result = _retire(tmp_path, cloud, plan)

    assert result["status"] == "skipped" and result["reason"] == "already_retired"
    assert receipt.read_text(encoding="utf-8") == "{}" and scene.is_dir()


def test_a_workspace_with_nothing_to_archive_publishes_nothing(tmp_path, monkeypatch):
    scene, cloud = _scene(tmp_path)
    for relative in _digests(scene):  # pretend every file was uploaded
        cloud.put(f"scenes/{SCENE}/{relative}", (scene / relative).read_bytes())
    plan = _plan(tmp_path, cloud)
    assert plan["archive"] == [] and plan["status"] == "retirable"

    result = _retire(tmp_path, cloud, plan, stream_publisher=lambda **_: pytest.fail("nothing to archive"))

    assert result["status"] == "retired" and result["archive_bytes"] == 0
    assert json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))["archive"] is None


def test_retire_receipt_replays_to_identical_bytes(tmp_path):
    scene, cloud = _scene(tmp_path, status="terminal_authority_ended")
    original = _digests(scene)
    client = MultipartClient()
    result = _retire(tmp_path, cloud, _plan(tmp_path, cloud), client=client)
    assert result["status"] == "retired" and not scene.exists()
    destination = tmp_path / "restored" / SCENE

    restored = retention.restore_scene_workspace(
        receipt_path=Path(result["receipt"]), destination=destination, cloud=cloud,
        materializer=functools.partial(store.materialize_configured_scene_artifact, client=client,
                                       bucket=ARTIFACT_BUCKET))

    assert restored["schema_version"] == retention.RESTORE_SCHEMA and restored["status"] == "restored"
    assert restored["file_count"] == len(original)
    assert _digests(destination) == original
    assert not [path for path in destination.parent.iterdir() if path.name.startswith(".")]


def test_restore_refuses_a_tampered_receipt_or_changed_cloud_bytes(tmp_path):
    scene, cloud = _scene(tmp_path)
    client = MultipartClient()
    receipt_path = Path(_retire(tmp_path, cloud, _plan(tmp_path, cloud), client=client)["receipt"])
    materializer = functools.partial(store.materialize_configured_scene_artifact, client=client, bucket=ARTIFACT_BUCKET)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps({**receipt, "scene_id": "scene-2"}), encoding="utf-8")

    with pytest.raises(retention.WebsiteSceneWorkspaceRetentionError, match="receipt_invalid"):
        retention.restore_scene_workspace(receipt_path=tampered, destination=tmp_path / "a", cloud=cloud,
                                          materializer=materializer)
    video = f"scenes/{SCENE}/captures/{CAPTURE}/raw/walkthrough.mov"
    cloud.put(video, b"x" * len(cloud.objects[(BUCKET, video)][0]))
    with pytest.raises(retention.WebsiteSceneWorkspaceRetentionError, match="restore_cloud_mismatch"):
        retention.restore_scene_workspace(receipt_path=receipt_path, destination=tmp_path / "b", cloud=cloud,
                                          materializer=materializer)
    assert not (tmp_path / "b").exists() and not list(tmp_path.glob(".restore-*"))
    (tmp_path / "c").mkdir()
    with pytest.raises(retention.WebsiteSceneWorkspaceRetentionError, match="destination_exists"):
        retention.restore_scene_workspace(receipt_path=receipt_path, destination=tmp_path / "c", cloud=cloud,
                                          materializer=materializer)


# --- the command line the operator door runs ------------------------------------------------------


@pytest.fixture()
def cli(tmp_path, monkeypatch):
    """The module's command line against temporary roots, a fake cloud and a fake artifact store."""

    context = _context(tmp_path)
    for name, value in (("BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT", context.storage_root),
                        ("BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT", context.pins_root),
                        ("BLUEPRINT_CONTROL_PLANE_GC_SCENE_INTENT_ROOT", context.intent_root),
                        ("BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT", context.binding_root),
                        ("BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS", str(tmp_path / "queue"))):
        monkeypatch.setenv(name, str(value))
    state = SimpleNamespace(cloud=FakeCloud(), client=MultipartClient(), now=time.time() + 72 * HOUR)
    monkeypatch.setattr(retention, "_cloud_inventory", lambda: state.cloud)
    monkeypatch.setattr(retention, "publish_configured_scene_stream", _publisher(state.client))
    monkeypatch.setattr(retention, "_process_in_use", _idle)
    monkeypatch.setattr(retention, "_now", lambda: state.now)
    monkeypatch.setattr(retention, "materialize_configured_scene_artifact", functools.partial(
        store.materialize_configured_scene_artifact, client=state.client, bucket=ARTIFACT_BUCKET))
    return state


def _cli(tmp_path: Path, *argv: str) -> tuple[int, dict]:
    out = tmp_path / "result.json"
    code = retention.main([*argv, "--result-out", str(out)])
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_cli_plans_then_retires_with_the_ack(tmp_path, cli):
    scene, cli.cloud = _scene(tmp_path)

    code, planned = _cli(tmp_path, "retire", "--scene-id", SCENE)
    assert code == 0 and planned["status"] == "planned" and planned["plan"]["status"] == "retirable"
    assert scene.is_dir()

    code, refused = _cli(tmp_path, "retire", "--scene-id", SCENE, "--apply")
    assert code == 1 and refused["status"] == "failed" and refused["code"].endswith("apply_not_authorized")
    assert scene.is_dir()

    code, retired = _cli(tmp_path, "retire", "--scene-id", SCENE, "--apply", "--ack", retention.RETIRE_ACK)
    assert code == 0 and retired["status"] == "retired" and retired["bucket"] == BUCKET
    assert not scene.exists() and Path(retired["receipt"]).is_file()

    code, again = _cli(tmp_path, "retire", "--scene-id", SCENE, "--apply", "--ack", retention.RETIRE_ACK)
    assert code == 0 and again["status"] == "retired" and again["already_retired"] is True


def test_cli_reports_why_a_scene_is_retained(tmp_path, cli):
    scene, cli.cloud = _scene(tmp_path, ack=False)

    code, result = _cli(tmp_path, "retire", "--scene-id", SCENE, "--bucket", BUCKET, "--apply",
                        "--ack", retention.RETIRE_ACK)

    assert code == 0 and result["status"] == "retained"
    assert result["reasons"] == [f"acknowledgement_unproven:{CAPTURE}"] and scene.is_dir()


def test_cli_fails_with_a_typed_code(tmp_path, cli):
    code, result = _cli(tmp_path, "retire", "--scene-id", "no-such-scene")
    assert code == 1 and result == {"status": "failed", "code": "scene_workspace_not_found"}
    code, result = _cli(tmp_path, "retire", "--scene-id", "../etc")
    assert code == 1 and result["code"] == "website_scene_workspace_identity_invalid"


def test_cli_restores_a_receipt(tmp_path, cli):
    scene, cli.cloud = _scene(tmp_path)
    original = _digests(scene)
    _, retired = _cli(tmp_path, "retire", "--scene-id", SCENE, "--apply", "--ack", retention.RETIRE_ACK)

    code, restored = _cli(tmp_path, "restore", "--receipt", retired["receipt"], "--destination", str(scene))

    assert code == 0 and restored["status"] == "restored" and _digests(scene) == original


def test_cli_defaults_protect_every_queue_the_reclaim_timer_protects():
    unit = (Path(__file__).resolve().parents[1] / "deploy/systemd/blueprint-control-plane-storage-gc.service"
            ).read_text(encoding="utf-8")
    line = next(row for row in unit.splitlines() if row.startswith("Environment=BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS="))
    gc_roots = [item for item in line.split("=", 2)[2].split(":") if item]
    defaults = retention.scene_queue_roots(retention.DEFAULT_QUEUE_ROOTS, retention.DEFAULT_INTENT_ROOT)
    assert set(map(Path, gc_roots)) <= set(defaults)
    control_plane = Path("/var/lib/blueprint/pipeline-control-plane")
    assert {control_plane / "sam31-preparation-executions",
            control_plane / "task-evaluation-scene-configuration-activation-intents"} <= set(defaults)


# --- what the listener reads back ------------------------------------------------------------------


def test_a_retirement_receipt_answers_for_the_messages_it_proves_terminal(tmp_path):
    scene, cloud = _scene(tmp_path, status="terminal_authority_ended")
    result = _retire(tmp_path, cloud, _plan(tmp_path, cloud))
    storage = tmp_path / "pubsub-handoffs"

    status = retention.retired_capture_status(storage_root=storage, bucket=BUCKET, scene_id=SCENE,
                                              capture_id=CAPTURE)

    assert status is not None and status["receipt"] == result["receipt"]
    assert (status["status"], status["queue_disposition"]) == ("terminal_authority_ended", "terminal_authority_ended")
    assert status["payload_sha256s"] == [listener.payload_sha256(_payload(CAPTURE))]
    for other in ({"capture_id": "capture-9"}, {"scene_id": "scene-2"}, {"bucket": "other-bucket"},
                  {"capture_id": "../x"}):
        query = {"storage_root": storage, "bucket": BUCKET, "scene_id": SCENE, "capture_id": CAPTURE, **other}
        assert retention.retired_capture_status(**query) is None
    receipt = Path(result["receipt"])
    document = json.loads(receipt.read_text(encoding="utf-8"))
    receipt.chmod(0o640)
    receipt.write_text(json.dumps({**document, "retired_at_epoch": 0}), encoding="utf-8")
    assert retention.retired_capture_status(storage_root=storage, bucket=BUCKET, scene_id=SCENE,
                                            capture_id=CAPTURE) is None
