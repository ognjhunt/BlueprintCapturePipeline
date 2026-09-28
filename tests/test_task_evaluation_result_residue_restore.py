# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_result_residue_restore.py
"""An offloaded residue comes back member by member, never over a different file."""

from __future__ import annotations

import functools
import json
import os
import stat
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline import task_evaluation_result_residue_offload as residue
from blueprint_pipeline import task_evaluation_result_residue_restore as restore
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_result_residue_offload import (  # noqa: F401 - hermetic_host is autouse here too
    BUCKET,
    NOW,
    OLD,
    RESIDUE,
    _local_files,
    _offload,
    _sealed_run,
    hermetic_host,
)


def test_residue_restore_round_trips(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    os.chmod(f.run / "logs" / "worker.log", 0o600)
    modes = {relative: stat.S_IMODE((f.run / relative).stat().st_mode) for relative in RESIDUE}
    before = _local_files(f.run)
    assert _offload(f)["offloaded_count"] == len(RESIDUE)
    pointer_bytes = f.pointer.read_bytes()
    materializer = functools.partial(store.materialize_configured_scene_artifact, client=f.client, bucket=BUCKET)

    restored = residue.restore_result_residue(run_root=f.run, materializer=materializer, now=lambda: NOW)

    assert (restored["status"], restored["restored_count"], restored["conflicts"]) == ("restored", len(RESIDUE), [])
    assert restored["restored_bytes"] == sum(len(data) for data in RESIDUE.values())
    assert _local_files(f.run) == before
    assert {relative: stat.S_IMODE((f.run / relative).stat().st_mode) for relative in RESIDUE} == modes
    assert (f.run / "logs/worker.log").stat().st_mtime == OLD
    assert f.pointer.read_bytes() == pointer_bytes
    receipt_path = f.evidence / f"{f.run.name}{residue.RESTORE_RECEIPT_SUFFIX}"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt == {**restored} and receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest")
    assert not [path for path in f.evidence.iterdir() if path.name.startswith(".")]

    # A second restore finds every member in place; a different local file is never overwritten.
    (f.run / "logs" / "worker.log").chmod(0o644)
    (f.run / "logs" / "worker.log").write_bytes(b"newer local truth\n")
    again = residue.restore_result_residue(run_root=f.run, materializer=materializer, now=lambda: NOW)
    assert (again["status"], again["restored_count"], again["already_present_count"]) == (
        "restored_with_conflicts", 0, len(RESIDUE) - 1)
    assert again["conflicts"] == [{"relative_path": "logs/worker.log", "reason": "existing_file_differs"}]
    assert (f.run / "logs" / "worker.log").read_bytes() == b"newer local truth\n"

    # A tampered pointer restores nothing.
    tampered = json.loads(pointer_bytes)
    tampered["members"][0]["size_bytes"] += 1
    f.pointer.chmod(0o640)
    f.pointer.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(residue.ResultResidueOffloadError, match="pointer_invalid"):
        residue.restore_result_residue(run_root=f.run, materializer=materializer, now=lambda: NOW)


def test_the_service_owner_is_given_each_file_last(tmp_path, monkeypatch) -> None:
    """The GC is root with CAP_CHOWN but not CAP_FOWNER: once a file belongs to the service
    user, root may no longer set its mode or times. So both happen first, the owner last."""

    f = _sealed_run(tmp_path / "canaries")
    calls: list[tuple] = []
    stranger = SimpleNamespace(st_uid=os.getuid() + 1, st_gid=os.getgid() + 1)
    descriptor = os.open(tmp_path / "probe", os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        with monkeypatch.context() as patched:
            patched.setattr(os, "fchmod", lambda fd, mode: calls.append(("chmod", mode)))
            patched.setattr(os, "fchown", lambda fd, uid, gid: calls.append(("chown", uid, gid)))
            residue._adopt_owner(descriptor, stranger, mode=0o440)
    finally:
        os.close(descriptor)
    assert calls == [("chmod", 0o440), ("chown", stranger.st_uid, stranger.st_gid)]

    _offload(f)
    order: list[str] = []
    real_utime, real_adopt = os.utime, residue._adopt_owner
    monkeypatch.setattr(os, "utime", lambda *args, **kwargs: (order.append("utime"), real_utime(*args, **kwargs))[1])
    recording = lambda *args, **kwargs: (order.append("adopt"), real_adopt(*args, **kwargs))[1]  # noqa: E731
    monkeypatch.setattr(residue, "_adopt_owner", recording)
    monkeypatch.setattr(restore, "_adopt_owner", recording)
    residue.restore_result_residue(run_root=f.run, now=lambda: NOW, materializer=functools.partial(
        store.materialize_configured_scene_artifact, client=f.client, bucket=BUCKET))
    # One (utime, adopt) pair per restored file, then the receipt's own adopt.
    assert order == ["utime", "adopt"] * len(RESIDUE) + ["adopt"]


def _materializer(f):
    return functools.partial(store.materialize_configured_scene_artifact, client=f.client, bucket=BUCKET)


def _receipt(f) -> dict:
    return json.loads((f.evidence / f"{f.run.name}{residue.RESTORE_RECEIPT_SUFFIX}").read_text(encoding="utf-8"))


def test_restore_restores_the_rest_past_a_member_it_cannot_place(tmp_path) -> None:
    """A member whose directory became a file cannot come back; it is a typed conflict, the
    others are restored, and the receipt is written."""

    f = _sealed_run(tmp_path / "canaries")
    _offload(f)
    (f.run / "work" / "stage").rmdir()
    (f.run / "work" / "stage").write_bytes(b"a file where a directory was")

    restored = restore.restore_result_residue(run_root=f.run, materializer=_materializer(f), now=lambda: NOW)

    assert restored["status"] == "restored_with_conflicts"
    assert restored["conflicts"] == [
        {"relative_path": "work/stage/state.npz", "reason": "restore_failed:NotADirectoryError"}]
    assert restored["restored_count"] == len(RESIDUE) - 1
    assert (f.run / "logs" / "worker.log").read_bytes() == RESIDUE["logs/worker.log"]
    assert (f.run / "work" / "stage").read_bytes() == b"a file where a directory was"
    assert _receipt(f) == restored


def test_restore_needs_no_sealed_registry_only_the_same_run(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    _offload(f)
    delivery = f.run / "artifacts/result_delivery/delivery.json"
    value = json.loads(delivery.read_text(encoding="utf-8"))
    delivery.write_text(json.dumps({**value, "result_status": "running"}), encoding="utf-8")

    restored = restore.restore_result_residue(run_root=f.run, materializer=_materializer(f), now=lambda: NOW)

    assert (restored["status"], restored["restored_count"]) == ("restored", len(RESIDUE))
    registry_path = f.run / "artifacts/result_delivery/artifact_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_path.write_text(json.dumps({**registry, "run_id": "another-run"}), encoding="utf-8")
    with pytest.raises(residue.ResultResidueOffloadError, match="run_mismatch"):
        restore.restore_result_residue(run_root=f.run, materializer=_materializer(f), now=lambda: NOW)


def test_restore_links_a_hard_link_group_again(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    os.link(f.run / "logs" / "worker.log", f.run / "logs" / "worker-copy.log")
    os.link(f.run / "logs" / "worker.log", f.run / "work" / "worker-elsewhere.log")
    _offload(f)
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    groups = {row["relative_path"]: row["group"] for row in pointer["members"]}
    assert groups["logs/worker.log"] == groups["logs/worker-copy.log"] == groups["work/worker-elsewhere.log"]
    assert groups["logs/worker.log"] != groups["provider/outputs.zip"]

    restored = restore.restore_result_residue(run_root=f.run, materializer=_materializer(f), now=lambda: NOW)

    assert restored["restored_count"] == len(RESIDUE) + 2
    linked = [f.run / "logs" / "worker.log", f.run / "logs" / "worker-copy.log", f.run / "work" / "worker-elsewhere.log"]
    identities = {(path.stat().st_dev, path.stat().st_ino) for path in linked}
    assert len(identities) == 1 and linked[0].stat().st_nlink == 3
    assert all(path.read_bytes() == RESIDUE["logs/worker.log"] for path in linked)


def test_the_receipt_is_written_even_when_the_archive_cannot_be_fetched(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    _offload(f)

    def unreachable(**_kwargs):
        raise OSError(5, "Input/output error")

    with pytest.raises(OSError):
        restore.restore_result_residue(run_root=f.run, materializer=unreachable, now=lambda: NOW)

    receipt = _receipt(f)
    assert (receipt["status"], receipt["restored_count"]) == ("failed", 0)
    assert receipt["failure"] == {"error_type": "OSError", "errno": 5, "stage": "restore"}
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert not [path for path in f.evidence.iterdir() if path.name.startswith(".")]


def test_restore_fsyncs_each_directory_it_places_a_member_in(tmp_path, monkeypatch) -> None:
    f = _sealed_run(tmp_path / "canaries")
    _offload(f)
    (f.run / "provider").rmdir()
    synced: set[tuple[int, int]] = set()
    real_fsync = os.fsync

    def recording(descriptor):
        metadata = os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):
            synced.add((metadata.st_dev, metadata.st_ino))
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording)
    restore.restore_result_residue(run_root=f.run, materializer=_materializer(f), now=lambda: NOW)

    for directory in (f.run, f.run / "logs", f.run / "work" / "stage", f.run / "provider"):
        assert (directory.stat().st_dev, directory.stat().st_ino) in synced, directory
