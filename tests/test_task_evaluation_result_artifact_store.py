from __future__ import annotations

import asyncio
import errno
import fcntl
import hashlib
import json
import os
import stat
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
        "blueprint_pipeline.completed_replay_cache_retention.process_reference",
        lambda _, **kw: None,
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


def _gc_tick(f, tmp_path):
    from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, run_storage_gc

    return run_storage_gc(
        content_store_roots=[], derived_roots=[], queue_roots=[], pins_root=tmp_path / "pins",
        evidence_roots=[f.root.parent], offload_enabled=True, apply=True, ack=RUN_ACK, hot_window_seconds=0,
        classifier=lambda *a, **kw: None,
        publisher=partial(store.publish_configured_scene_artifact, client=f.objects, bucket="private-bucket"),
    )


@pytest.mark.parametrize(
    ("stage", "error_type", "error_number"),
    [
        ("registry", "TaskEvaluationResultDeliveryError", None),
        # The registry verified, but a registered file no longer has its recorded bytes.
        ("plan", "TaskEvaluationResultDeliveryError", None),
        ("protection", "PermissionError", errno.EACCES),
        ("publish", "ControlPlaneDiskBudgetError", None),
        ("evict", "PermissionError", errno.EPERM),
    ],
)
def test_result_artifact_offload_failure_records_stage_and_errno(
    setup, monkeypatch, tmp_path, stage, error_type, error_number
):
    """2026-09-27: six registry runs failed result-artifact offload and the GC recorded only
    ``PermissionError``: not the step that raised it, nor the errno that tells a refused
    /proc read (EACCES) from a refused chmod (EPERM). Neither message nor path is kept."""
    from blueprint_pipeline.control_plane_disk_budget import ControlPlaneDiskBudgetError

    f = setup
    host_path = str(tmp_path / "host-only-path")
    process = "blueprint_pipeline.completed_replay_cache_retention.process_reference"
    monkeypatch.setattr(process, lambda _, **kw: None)

    def refuse(error):
        def raising(*_args, **_kwargs):
            raise error
        return raising

    payload = f.payload
    if stage == "registry":
        (f.registry_path.parent / "delivery.json").write_text("{}")
    elif stage == "plan":
        payload = b"x" * len(f.payload)
        f.path.write_bytes(payload)
    elif stage == "protection":
        monkeypatch.setattr(process, refuse(PermissionError(errno.EACCES, "Permission denied", host_path)))
    elif stage == "publish":
        monkeypatch.setattr(offload, "reserve_control_plane_disk",
                            refuse(ControlPlaneDiskBudgetError("control_plane_disk_budget_exceeded")))
    else:
        monkeypatch.setattr(offload, "acquire_artifact_read_lease",
                            refuse(PermissionError(errno.EPERM, "Operation not permitted", host_path)))

    report = _gc_tick(f, tmp_path)

    assert report["result_artifact_offload"] == [{
        "status": "retained", "run_directory": f.root.name, "reason": error_type,
        "error_type": error_type, "errno": error_number, "stage": stage,
    }]
    assert f.path.read_bytes() == payload and f.registry_path.exists()
    assert host_path not in json.dumps(report)


def test_a_skipped_artifact_records_its_stage_and_errno(setup, monkeypatch, tmp_path):
    f = setup
    monkeypatch.setattr(
        "blueprint_pipeline.completed_replay_cache_retention.process_reference", lambda _, **kw: None)

    def full_disk(**_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device", str(tmp_path / "host-only-path"))

    result = apply(f, publisher=full_disk)

    assert result["offloaded_count"] == 0
    assert result["skipped"] == [{
        "relative_path": "evidence/external.mp4", "reason": "OSError",
        "error_type": "OSError", "errno": errno.ENOSPC, "stage": "publish",
    }]
    assert "host-only-path" not in json.dumps(result)
    assert f.path.read_bytes() == f.payload


def test_a_retained_run_says_why(setup, monkeypatch, tmp_path):
    f = setup
    assert offload.offload_result_artifacts(run_root=f.root)["retained_reason"] == "hot"
    assert apply(f, protection_checker=lambda _: True)["retained_reason"] == "protected"
    assert apply(f, protection_checker=lambda _: "protected_pin")["retained_reason"] == "protected_pin"
    # The GC passes its evidence protection reason through.
    monkeypatch.setattr("blueprint_pipeline.completed_replay_cache_retention.process_reference",
                        lambda _, **kw: "inventory_unreadable")
    [row] = _gc_tick(f, tmp_path)["result_artifact_offload"]
    assert (row["status"], row["retained_reason"]) == (
        "retained_hot_or_active", "protected_process_inventory_unreadable")
    assert f.path.read_bytes() == f.payload


class _SimulatedOwnership:
    """How an account without CAP_FOWNER sees ownership: ``os.geteuid``, ``open``, ``fstat``,
    ``fchmod``, ``chmod`` (and ``Path.chmod``), ``fchown`` and ``chown``.

    Owner, group and mode live in a table keyed by inode, and nothing real is chowned or
    chmodded. A file the simulated account creates through ``os.open`` (as ``mkstemp``
    and ``Path.touch`` do) is its own, its mode masked by the GC unit's ``UMask=0077``.
    Every other file belongs to ``default_owner`` (the registry owner) with its real
    mode until ``own()`` says otherwise. Changing a mode needs ownership and fails with
    EPERM otherwise; root (CAP_CHOWN) may chown anything, anyone else only what it keeps.
    """

    def __init__(self, monkeypatch, *, euid: int, egid: int, default_owner: tuple[int, int]):
        self.euid, self.egid, self.default_owner = euid, egid, default_owner
        self.files: dict[tuple[int, int], dict[str, int]] = {}
        self.calls: list[tuple[str, tuple[int, int]]] = []
        self._fstat, self._stat, self._open = os.fstat, os.stat, os.open
        monkeypatch.setattr(os, "geteuid", lambda: self.euid)
        monkeypatch.setattr(os, "open", self.open)
        monkeypatch.setattr(os, "fstat", self.fstat)
        monkeypatch.setattr(os, "fchmod", lambda fd, mode: self._chmod("fchmod", self._fstat(fd), mode))
        monkeypatch.setattr(os, "chmod", lambda path, mode, *, dir_fd=None, follow_symlinks=True: self._chmod(
            "chmod", self._stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks), mode))
        monkeypatch.setattr(Path, "chmod", lambda path, mode, *, follow_symlinks=True: self._chmod(
            "chmod", self._stat(path, follow_symlinks=follow_symlinks), mode))
        monkeypatch.setattr(os, "fchown", lambda fd, uid, gid: self._chown("fchown", self._fstat(fd), uid, gid))
        monkeypatch.setattr(os, "chown", lambda path, uid, gid, *, dir_fd=None, follow_symlinks=True: self._chown(
            "chown", self._stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks), uid, gid))

    def open(self, path, flags, mode=0o777, *, dir_fd=None):
        created = False
        if flags & os.O_CREAT:
            try:
                self._stat(path, dir_fd=dir_fd, follow_symlinks=False)
            except FileNotFoundError:
                created = True
        descriptor = self._open(path, flags, mode, dir_fd=dir_fd)
        if created:
            real = self._fstat(descriptor)
            self.files[(real.st_dev, real.st_ino)] = {"uid": self.euid, "gid": self.egid, "mode": mode & ~0o077}
        return descriptor

    def _entry(self, real) -> dict[str, int]:
        uid, gid = self.default_owner
        return self.files.setdefault((real.st_dev, real.st_ino),
                                     {"uid": uid, "gid": gid, "mode": stat.S_IMODE(real.st_mode)})

    def own(self, path: Path, *, uid: int, gid: int, mode: int) -> None:
        real = self._stat(path)
        self.files[(real.st_dev, real.st_ino)] = {"uid": uid, "gid": gid, "mode": mode}

    def of(self, path: Path) -> dict[str, int]:
        return self._entry(self._stat(path))

    def calls_on(self, path: Path) -> list[str]:
        real = self._stat(path)
        return [name for name, key in self.calls if key == (real.st_dev, real.st_ino)]

    def fstat(self, fd):
        real = self._fstat(fd)
        entry = self._entry(real)
        fields = {name: getattr(real, name) for name in dir(real) if name.startswith("st_")}
        fields.update(st_mode=stat.S_IFMT(real.st_mode) | entry["mode"], st_uid=entry["uid"], st_gid=entry["gid"])
        return SimpleNamespace(**fields)

    def _chmod(self, name: str, real, mode: int) -> None:
        self.calls.append((name, (real.st_dev, real.st_ino)))
        entry = self._entry(real)
        if entry["uid"] != self.euid:
            raise PermissionError(errno.EPERM, "Operation not permitted")
        entry["mode"] = stat.S_IMODE(mode)

    def _chown(self, name: str, real, uid: int, gid: int) -> None:
        self.calls.append((name, (real.st_dev, real.st_ino)))
        entry = self._entry(real)
        if self.euid != 0 and uid not in (-1, entry["uid"]):
            raise PermissionError(errno.EPERM, "Operation not permitted")
        entry["uid"] = entry["uid"] if uid == -1 else uid
        entry["gid"] = entry["gid"] if gid == -1 else gid


def _lease_lock(f) -> Path:
    return f.registry_path.parent / ".artifact-readers.lock"


def _flock_is_held(lock: Path) -> bool:
    probe = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(probe)
    return False


def _owner(f) -> tuple[int, int]:
    registry = f.registry_path.stat()
    return registry.st_uid, registry.st_gid


def _lease_with_existing_lock(f, monkeypatch, *, euid: int, egid: int, uid: int, gid: int, mode: int):
    """Take a shared lease on a lock that already exists as ``uid:gid`` with ``mode``."""

    _lease_lock(f).touch()
    ownership = _SimulatedOwnership(monkeypatch, euid=euid, egid=egid, default_owner=_owner(f))
    ownership.own(_lease_lock(f), uid=uid, gid=gid, mode=mode)
    return ownership, offload.acquire_artifact_read_lease(f.root)


def test_root_eviction_lease_needs_no_fowner(setup, monkeypatch):
    """2026-09-27: the GC unit is root with CAP_CHOWN and CAP_DAC_OVERRIDE but not CAP_FOWNER.
    The lease gave its new lock to the registry owner and then chmodded it, which only
    that owner may do, so every eviction failed with EPERM after the upload. The mode
    is now set while root still owns the file, and the owner changed last."""

    f = setup
    uid, gid = _owner(f)
    ownership = _SimulatedOwnership(monkeypatch, euid=0, egid=0, default_owner=(uid, gid))

    release = offload.acquire_artifact_read_lease(f.root, exclusive=True)
    try:
        assert ownership.of(_lease_lock(f)) == {"uid": uid, "gid": gid, "mode": 0o660}
        assert ownership.calls_on(_lease_lock(f)) == ["fchmod", "fchown"]
        assert _flock_is_held(_lease_lock(f))
    finally:
        release()
    assert not _flock_is_held(_lease_lock(f))


@pytest.mark.parametrize("mode", [0o660, 0o600], ids=["0660", "0600-left-by-a-failed-tick"])
def test_existing_lock_owned_by_registry_owner_is_not_chmodded(setup, monkeypatch, mode):
    """A lock the registry owner already holds is neither chmodded nor chowned by root: at
    0660 it needs nothing, and at 0600 (left by a tick that failed its chmod) root may
    not repair it; the open and the flock are the access check, and the owner's next
    lease repairs the mode."""

    f = setup
    uid, gid = _owner(f)
    ownership, release = _lease_with_existing_lock(f, monkeypatch, euid=0, egid=0, uid=uid, gid=gid, mode=mode)
    try:
        assert ownership.calls_on(_lease_lock(f)) == []
        assert ownership.of(_lease_lock(f)) == {"uid": uid, "gid": gid, "mode": mode}
        assert _flock_is_held(_lease_lock(f))
    finally:
        release()
    assert not _flock_is_held(_lease_lock(f))


def test_root_gives_a_third_users_lock_to_the_registry_owner_without_chmod(setup, monkeypatch):
    """A lock some other account owns is chowned to the registry owner, which needs only
    CAP_CHOWN, and keeps its mode: root may not chmod what it does not own."""

    f = setup
    uid, gid = _owner(f)
    ownership, release = _lease_with_existing_lock(f, monkeypatch, euid=0, egid=0, uid=4242, gid=4242, mode=0o600)
    try:
        assert ownership.calls_on(_lease_lock(f)) == ["fchown"]
        assert ownership.of(_lease_lock(f)) == {"uid": uid, "gid": gid, "mode": 0o600}
        assert _flock_is_held(_lease_lock(f))
    finally:
        release()


def test_non_root_reader_lease_is_unchanged(setup, monkeypatch):
    """The service account (the WebApp-facing resolver) creates the lock as its owner, sets its
    mode, and never chowns it."""

    f = setup
    uid, gid = _owner(f)
    ownership = _SimulatedOwnership(monkeypatch, euid=uid, egid=gid, default_owner=(uid, gid))

    release = offload.acquire_artifact_read_lease(f.root)
    try:
        assert ownership.calls_on(_lease_lock(f)) == ["fchmod"]
        assert ownership.of(_lease_lock(f)) == {"uid": uid, "gid": gid, "mode": 0o660}
        assert _flock_is_held(_lease_lock(f))
    finally:
        release()


def test_non_root_reader_takes_the_flock_on_a_lock_someone_else_owns(setup, monkeypatch):
    """The one intended change for a non-root reader. It used to chmod the lock whoever owned
    it, so a group-writable lock another account owns failed with EPERM. It now leaves
    that lock's mode alone and takes the flock; the open and the flock are the check."""

    f = setup
    uid, gid = _owner(f)
    ownership, release = _lease_with_existing_lock(f, monkeypatch, euid=uid, egid=gid, uid=4242, gid=gid, mode=0o664)
    try:
        assert ownership.calls_on(_lease_lock(f)) == []
        assert ownership.of(_lease_lock(f)) == {"uid": 4242, "gid": gid, "mode": 0o664}
        assert _flock_is_held(_lease_lock(f))
    finally:
        release()


def test_the_owners_next_lease_repairs_a_lock_a_failed_tick_left_0600(setup, monkeypatch):
    f = setup
    uid, gid = _owner(f)
    ownership, release = _lease_with_existing_lock(f, monkeypatch, euid=uid, egid=gid, uid=uid, gid=gid, mode=0o600)
    try:
        assert ownership.calls_on(_lease_lock(f)) == ["fchmod"]
        assert ownership.of(_lease_lock(f)) == {"uid": uid, "gid": gid, "mode": 0o660}
    finally:
        release()


def test_root_gc_tick_evicts_without_fowner(setup, monkeypatch, tmp_path):
    """The six production registry runs with eviction candidates published their bulk
    artifacts on every hourly tick and were then retained with PermissionError at the
    evict stage. Under the same privileges a tick now evicts them, and every mode or
    owner change the tick makes is one those privileges allow."""

    f = setup
    uid, gid = _owner(f)
    monkeypatch.setattr(
        "blueprint_pipeline.completed_replay_cache_retention.process_reference", lambda _, **kw: None)
    ownership = _SimulatedOwnership(monkeypatch, euid=0, egid=0, default_owner=(uid, gid))

    [row] = _gc_tick(f, tmp_path)["result_artifact_offload"]

    assert (row["status"], row["offloaded_count"], row["skipped"]) == ("applied", 1, [])
    assert not f.path.exists() and f.objects.upload_count == 1
    assert ownership.of(_lease_lock(f)) == {"uid": uid, "gid": gid, "mode": 0o660}
    # The ledger, the reservation, the reference and the lease all went through the fake.
    assert {name for name, _key in ownership.calls} == {"chmod", "fchmod", "fchown", "chown"}


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
