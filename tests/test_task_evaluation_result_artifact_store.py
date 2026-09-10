from __future__ import annotations

import asyncio
import hashlib
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_evaluation_result_artifact_store as offload
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk
from blueprint_pipeline.control_plane_evidence_offload import build_evidence_offload_manifest
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.task_evaluation_result_delivery import (
    resolve_task_evaluation_result_artifact,
)
from tests.test_live_pipeline_result_artifact_api import (
    ARTIFACT_ID,
    _configure,
    _get,
    _headers,
    _registry,
)
from tests.test_task_evaluation_configured_scene_object_store import _ContentAddressedClient


@pytest.fixture
def setup(tmp_path, monkeypatch):
    client, token, roots = _configure(tmp_path, monkeypatch)
    run_id = "remote-result-test"
    root = roots / f"{run_id}-activation"
    payload = b"0123456789" * 8000
    _registry(root, run_id=run_id, content=payload)
    registry_path = root / "artifacts/result_delivery/artifact_registry.json"
    registry = json.loads(registry_path.read_text())
    closure = {}
    for name in ("billing", "teardown", "provider_zero"):
        data = b'{"status":"closed"}'
        path = root / f"{name}.json"
        path.write_bytes(data)
        sha = "sha256:" + hashlib.sha256(data).hexdigest()
        record = {
            "artifact_id": name,
            "role": "closure_" + name,
            "relative_path": path.name,
            "evidence_root": str(root),
            "sha256": sha,
            "size_bytes": len(data),
        }
        registry["artifacts"].append(record)
        closure[name + "_receipt"] = {"artifact_id": name, "digest": sha, "size_bytes": len(data)}
        if name == "provider_zero":
            closure[name + "_receipt"]["provider_zero_verified"] = True
    delivery = {
        "schema_version": "task_evaluation_result_delivery.v2",
        "run_id": run_id,
        "result_status": "blocked",
        "reproducibility": closure,
        "delivery_digest": "",
    }
    delivery["delivery_digest"] = cross_runtime_canonical_digest(
        delivery, digest_field="delivery_digest"
    )
    registry["delivery_digest"] = delivery["delivery_digest"]
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    registry_path.write_text(json.dumps(registry))
    (registry_path.parent / "delivery.json").write_text(json.dumps(delivery))
    object_client = _ContentAddressedClient()
    monkeypatch.setattr(
        store, "_artifact_object_store_client", lambda: (object_client, "private-bucket")
    )
    monkeypatch.setattr(
        offload, "_artifact_object_store_client", lambda: (object_client, "private-bucket")
    )
    cache = tmp_path / "cache"
    monkeypatch.setenv(offload.CACHE_ROOT_ENV, str(cache))
    ledger = tmp_path / "ledger"
    monkeypatch.setattr(
        offload,
        "reserve_control_plane_disk",
        partial(
            reserve_control_plane_disk,
            reservation_root=ledger,
            disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, free=80 * 1024**3),
        ),
    )
    return SimpleNamespace(
        client=client,
        token=token,
        run_id=run_id,
        root=root,
        payload=payload,
        objects=object_client,
        cache=cache,
        ledger=ledger,
        registry=registry,
        registry_path=registry_path,
        path=root / "evidence/external.mp4",
    )


def apply(f, **kwargs):
    return offload.offload_result_artifacts(
        run_root=f.root, apply=True, ack=offload.APPLY_ACK, hot_window_seconds=0, **kwargs
    )


def test_dry_run_then_verified_eviction_signed_download_and_range(setup):
    f = setup
    before = f.registry_path.read_bytes()
    dry = offload.offload_result_artifacts(run_root=f.root, hot_window_seconds=0)
    assert dry["candidate_bytes"] == len(f.payload)
    assert f.path.exists() and not f.objects.objects
    result = apply(f)
    assert result["offloaded_count"] == 1 and not result["skipped"]
    assert not f.path.exists()
    assert f.registry_path.read_bytes() == before
    assert (f.root / "billing.json").exists()
    again = apply(f)
    assert again["already_remote_count"] == 1 and f.objects.upload_count == 1
    response = _get(f.client, f.token, f.run_id, "remote-whole")
    assert response.status_code == 200 and response.content == f.payload
    assert response.headers["x-blueprint-artifact-sha256"] == f.registry["artifacts"][0]["sha256"]
    assert list(f.cache.iterdir()) == []
    assert not list(f.ledger.glob("*.json"))
    response = f.client.get(
        f"/api/live-pipeline/task-evaluation-runs/{f.run_id}/artifacts/{ARTIFACT_ID}",
        headers={**_headers(f.token, "remote-range"), "Range": "bytes=7-17"},
    )
    assert response.status_code == 206 and response.content == f.payload[7:18]
    assert response.headers["content-range"] == f"bytes 7-17/{len(f.payload)}"
    assert list(f.cache.iterdir()) == []


def test_corrupt_upload_or_reference_write_never_evicts(setup, monkeypatch):
    f = setup
    f.objects.corrupt_readback = True
    result = apply(f)
    assert result["offloaded_count"] == 0 and result["skipped"]
    assert f.path.read_bytes() == f.payload
    f.objects.corrupt_readback = False

    def fail(*args):
        raise OSError("disk write refused")

    monkeypatch.setattr(offload, "_durable_reference", fail)
    result = apply(f)
    assert result["offloaded_count"] == 0 and result["skipped"]
    assert f.path.read_bytes() == f.payload


def test_remote_corruption_fails_before_exposing_bytes_and_cleans_up(setup):
    f = setup
    apply(f)
    f.objects.corrupt_readback = True
    response = _get(f.client, f.token, f.run_id, "remote-corrupt")
    assert response.status_code == 503
    assert f.payload not in response.content
    assert list(f.cache.iterdir()) == [] and not list(f.ledger.glob("*.json"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "different-run"),
        ("registry_digest", "sha256:" + "a" * 64),
        ("relative_path", "other.mp4"),
    ],
)
def test_remote_reference_cannot_cross_run_or_registry(setup, field, value):
    f = setup
    apply(f)
    pointer = next((f.registry_path.parent / offload.REMOTE_DIRECTORY).glob("*.json"))
    body = json.loads(pointer.read_text())
    body[field] = value
    body["reference_digest"] = canonical_digest(body, digest_field="reference_digest")
    pointer.chmod(0o600)
    pointer.write_text(json.dumps(body))
    assert _get(f.client, f.token, f.run_id, "remote-wrong-binding").status_code == 404
    assert not f.cache.exists()


def test_remote_reference_wrong_bucket_is_refused(setup):
    f = setup
    apply(f)
    pointer = next((f.registry_path.parent / offload.REMOTE_DIRECTORY).glob("*.json"))
    body = json.loads(pointer.read_text())
    body["reference"]["uri"] = body["reference"]["uri"].replace("private-bucket", "other-tenant")
    body["reference_digest"] = canonical_digest(body, digest_field="reference_digest")
    pointer.chmod(0o600)
    pointer.write_text(json.dumps(body))
    assert _get(f.client, f.token, f.run_id, "remote-wrong-bucket").status_code == 503
    assert list(f.cache.iterdir()) == []


def test_changed_source_after_publication_remains_local(setup):
    f = setup

    def publish(**kwargs):
        result = store.publish_configured_scene_artifact(
            client=f.objects, bucket="private-bucket", **kwargs
        )
        f.path.write_bytes(b"replacement")
        return result

    result = apply(f, publisher=publish)
    assert result["offloaded_count"] == 0 and result["skipped"]
    assert f.path.read_bytes() == b"replacement"


def test_hot_active_or_reactivated_runs_are_retained(setup):
    f = setup
    assert offload.offload_result_artifacts(run_root=f.root)["status"] == "retained_hot_or_active"
    assert apply(f, protection_checker=lambda _: True)["status"] == "retained_hot_or_active"
    calls = []

    def protected(_):
        calls.append(1)
        return len(calls) > 1

    result = apply(f, protection_checker=protected)
    assert result["offloaded_count"] == 0 and result["skipped"]
    assert f.path.read_bytes() == f.payload


def test_bulk_aliases_share_one_reference_and_protected_alias_keeps_local(setup):
    f = setup
    registry = f.registry
    registry["artifacts"].append({**registry["artifacts"][0], "artifact_id": "alias"})
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    f.registry_path.write_text(json.dumps(registry))
    assert apply(f)["offloaded_count"] == 1
    path, record = resolve_task_evaluation_result_artifact(
        run_root=f.root, run_id=f.run_id, artifact_id="alias"
    )
    assert path.read_bytes() == f.payload
    record["_artifact_cleanup"]()


def test_protected_alias_and_unregistered_bytes_are_never_evicted(setup):
    f = setup
    f.registry["artifacts"].append(
        {**f.registry["artifacts"][0], "artifact_id": "alias", "role": "episode_receipt"}
    )
    f.registry["registry_digest"] = canonical_digest(f.registry, digest_field="registry_digest")
    f.registry_path.write_text(json.dumps(f.registry))
    unknown = f.root / "unknown.png"
    unknown.write_bytes(f.payload)
    assert apply(f)["candidate_count"] == 0
    assert f.path.exists() and unknown.exists()


def test_generic_archive_gc_preserves_registered_download_root(setup):
    f = setup
    (f.root / "dispatch_receipt.json").write_text("{}")
    result = build_evidence_offload_manifest(
        evidence_roots=[f.root.parent], hot_window_seconds=0, classifier=lambda *a, **k: None
    )
    assert result["candidate_count"] == 0


def test_missing_auth_does_not_fetch_or_allocate_cache(setup):
    f = setup
    apply(f)
    response = f.client.get(
        f"/api/live-pipeline/task-evaluation-runs/{f.run_id}/artifacts/{ARTIFACT_ID}"
    )
    assert response.status_code in (401, 403)
    assert not f.cache.exists()


def test_symlinked_reference_directory_is_refused(setup, tmp_path):
    f = setup
    apply(f)
    references = f.registry_path.parent / offload.REMOTE_DIRECTORY
    moved = tmp_path / "redirected-references"
    references.rename(moved)
    references.symlink_to(moved, target_is_directory=True)
    assert _get(f.client, f.token, f.run_id, "remote-symlink").status_code == 404
    assert not f.cache.exists()


def test_file_response_cleanup_runs_on_cancelled_send(setup):
    from blueprint_pipeline.live_pipeline_result_artifact_response import ResultArtifactFileResponse

    f = setup
    apply(f)
    path, record = resolve_task_evaluation_result_artifact(
        run_root=f.root, run_id=f.run_id, artifact_id=ARTIFACT_ID
    )
    response = ResultArtifactFileResponse(path, artifact_cleanup=record["_artifact_cleanup"])

    async def send(_):
        raise asyncio.CancelledError

    async def receive():
        return {"type": "http.disconnect"}

    async def execute():
        with pytest.raises(asyncio.CancelledError):
            await response(
                {"type": "http", "method": "GET", "headers": [], "extensions": {}}, receive, send
            )

    asyncio.run(execute())
    assert list(f.cache.iterdir()) == [] and not list(f.ledger.glob("*.json"))


def test_intake_uses_same_private_artifact_store_bindings_as_gc():
    repo = Path(__file__).resolve().parents[1]
    gc = (repo / "deploy/systemd/blueprint-control-plane-storage-gc.service").read_text()
    intake = (repo / "deploy/systemd/blueprint-pipeline-intake.service").read_text()
    for line in gc.splitlines():
        if line.startswith("Environment=BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_"):
            assert line in intake


def test_existing_gc_tick_uses_per_artifact_offload(setup, monkeypatch, tmp_path):
    from blueprint_pipeline.control_plane_storage_gc import run_storage_gc, RUN_ACK

    f = setup
    monkeypatch.setattr(
        "blueprint_pipeline.completed_replay_cache_retention.active_reference",
        lambda _, **kw: False,
    )
    report = run_storage_gc(
        content_store_roots=[],
        derived_roots=[],
        queue_roots=[],
        pins_root=tmp_path / "pins",
        evidence_roots=[f.root.parent],
        offload_enabled=True,
        apply=True,
        ack=RUN_ACK,
        hot_window_seconds=0,
        classifier=lambda *a, **kw: None,
        publisher=partial(
            store.publish_configured_scene_artifact, client=f.objects, bucket="private-bucket"
        ),
    )
    assert report["result_artifact_offload"][0]["offloaded_count"] == 1
    assert report["evidence_offload"]["offloaded_count"] == 0
    assert f.registry_path.exists() and not f.path.exists()
    assert _get(f.client, f.token, f.run_id, "gc-remote").content == f.payload


def test_disk_reservation_refusal_does_not_upload_or_delete(setup, monkeypatch):
    from blueprint_pipeline.control_plane_disk_budget import ControlPlaneDiskBudgetError

    f = setup

    def refused(*args, **kwargs):
        raise ControlPlaneDiskBudgetError("control_plane_disk_budget_exceeded")

    monkeypatch.setattr(offload, "reserve_control_plane_disk", refused)
    with pytest.raises(ControlPlaneDiskBudgetError):
        apply(f)
    assert f.path.read_bytes() == f.payload and not f.objects.objects


def test_dead_process_cache_reclamation_preserves_active_and_unknown_dirs(tmp_path, monkeypatch):
    dead = tmp_path / "download-9999999-a123"
    live = tmp_path / "download-100-b123"
    unknown = tmp_path / "user-owned"
    for p in (dead, live, unknown):
        p.mkdir()
        (p / "data").write_bytes(b"retain except dead")

    def exists(pid, signal):
        if pid == 9999999:
            raise ProcessLookupError

    monkeypatch.setattr(offload.os, "kill", exists)
    offload._reap_abandoned_downloads(tmp_path)
    assert not dead.exists() and live.exists() and unknown.exists()


def test_mutation_during_reference_commit_is_preserved(setup, monkeypatch):
    f = setup
    original = offload._durable_reference

    def write(*args):
        original(*args)
        f.path.write_bytes(b"new local truth")

    monkeypatch.setattr(offload, "_durable_reference", write)
    report = apply(f)
    assert report["offloaded_count"] == 0 and report["skipped"]
    assert f.path.read_bytes() == b"new local truth"


def test_eviction_waits_for_an_active_authenticated_download_lease(setup):
    import threading
    from blueprint_pipeline.live_pipeline_result_artifact_resolution import (
        resolve_live_pipeline_result_artifact,
    )

    f = setup
    path, record = resolve_live_pipeline_result_artifact(
        legacy_state_root=f.root.parent / "unused",
        policy_canary_result_root=f.root.parent,
        run_id=f.run_id,
        artifact_id=ARTIFACT_ID,
        retain_read_lease=True,
    )
    published = threading.Event()
    finished = threading.Event()
    results = []

    def publish(**kwargs):
        result = store.publish_configured_scene_artifact(
            client=f.objects, bucket="private-bucket", **kwargs
        )
        published.set()
        return result

    def migrate():
        try:
            results.append(apply(f, publisher=publish))
        finally:
            finished.set()

    worker = threading.Thread(target=migrate)
    worker.start()
    try:
        assert published.wait(5)
        assert not finished.wait(0.1)
        assert path.read_bytes() == f.payload
    finally:
        record["_artifact_cleanup"]()
        worker.join(5)
    assert finished.is_set() and results[0]["offloaded_count"] == 1
    assert not f.path.exists()
